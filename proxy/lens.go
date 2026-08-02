// The proxy's lens surfaces: everything that reads the geometric-lens
// service, feeds the corpus it trains on, or reports whether it is calibrated
// for the model currently being served.
//
// In file order:
//
//	Per-write scoring — every write_file / edit_file payload goes to
//	  /internal/lens/score-per-step for its C(x) and G(x) numbers. A run of
//	  low gx_score_min, or one write below the severe cutoff, is the "stub
//	  loop" signal the agent loop breaks with a corrective. Thresholds come
//	  from the model's own calibration or the check is skipped — one model's
//	  cutoffs are meaningless against another's residual stream.
//	Training-corpus collection — each file the model authored during a pass
//	  is stashed, then labeled and weighted by the human verdict and
//	  appended as per-model JSONL. Nothing trains here; `atlas lens retrain`
//	  consumes the corpus later.
//	The /feedback handler — where that verdict arrives from the TUI, with
//	  the pending-pass stash it draws from and the training-status endpoint
//	  behind the "retrain available" alert.
//	Calibration probes — /v1/calibration/status, built from the lens
//	  service's /health plus a local read of the ASA control vector, is the
//	  seven-dimension table the TUI badge and `atlas doctor` both render.
//
// Scoring and collection are one loop seen at two points: the write the lens
// scores now is the write a human labels later, and that label trains the
// lens that scores the next one. Keeping both in one file keeps the round
// trip legible — and keeps the calibration probe next to the code whose
// behavior it reports on.

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Agent-loop lens integration. Scores write_file / edit_file content
// per-tool-call via geometric-lens /internal/lens/score-per-step. Tracks
// recent gx_score_min values per session so a "stub loop" pattern (the
// kind that hit the May 6 templates/resources.html session in production)
// can be detected and broken with a corrective system message before the
// next LLM call.

// Number of consecutive low-score write/edit calls that count as a
// regression. 2 is the minimum that's clearly a pattern (not a one-off
// dud); higher values (3+) miss the May 6 stub-loop case where the
// model only got 2 attempts in before the error-loop break fired.
const lensRegressionRunLength = 2

// Severe-threshold short-circuit: a single write whose gx_score_min
// drops below this is bad enough to trigger intervention immediately
// without waiting for a second confirmation. Calibrated from the May 7
// dashboard.html session where gx_min=0.040 on turn 2 (off_rails at
// token 14 of 840) was unambiguously a stub but the run-of-2 rule
// waited until turn 4 to act — by which point V3's sandbox-verifier
// had already approved the write. Anything below 0.05 is so far into
// the "likely_incorrect" band that one sample is enough signal.
//
// Language-agnostic: gx values are normalized 0-1 outputs of the
// XGBoost head on the residual stream. They don't depend on the
// surface language of the file being scored — Python stub, HTML stub,
// Rust stub, Java stub all produce the same kind of low gx_min when
// the model's internal state collapses to a placeholder pattern.
type lensAggregate struct {
	FirstOffRailsIdx int     `json:"first_off_rails_idx"`
	GxScoreMin       float64 `json:"gx_score_min"`
	GxScoreMean      float64 `json:"gx_score_mean"`
	CxNormMax        float64 `json:"cx_norm_max"`
}

// lensThresholds are the per-model operating points the lens service judged a
// score against. They ship with the lens artifact (gx_thresholds.json) and are
// returned in every score response so the proxy's regression checks use the
// loaded model's calibration instead of the hardcoded fallback constants.
type lensThresholds struct {
	OffRails float64 `json:"off_rails"`
	Low      float64 `json:"low"`
	Severe   float64 `json:"severe"`
}

type lensPerStepResult struct {
	Enabled     bool            `json:"enabled"`
	GxAvailable bool            `json:"gx_available"`
	NTokens     int             `json:"n_tokens"`
	HiddenDim   int             `json:"hidden_dim"`
	Layer       string          `json:"layer"`
	Aggregate   lensAggregate   `json:"aggregate"`
	LatencyMS   float64         `json:"latency_ms"`
	Thresholds  *lensThresholds `json:"thresholds,omitempty"`
	Error       string          `json:"error,omitempty"`
}

// calibratedThresholds returns operating points only when the selected
// model's Lens artifact supplied a valid calibration. Uncalibrated scores are
// useful telemetry, but must not trigger corrective behavior using another
// model's cutoffs.
func (r lensPerStepResult) calibratedThresholds() (low, severe float64, ok bool) {
	if r.Thresholds == nil || r.Thresholds.Low <= 0 || r.Thresholds.Severe <= 0 ||
		r.Thresholds.Severe > r.Thresholds.Low {
		return 0, 0, false
	}
	return r.Thresholds.Low, r.Thresholds.Severe, true
}

// scoreContentForAgent calls /internal/lens/score-per-step on the given
// text and returns the parsed result. Fail-soft: returns (zero, false)
// on any error so a lens outage degrades to "no signal" rather than
// breaking the agent loop. Carries the agent's ctx so client cancellation
// kills the lens call too.
func scoreContentForAgent(ctx context.Context, lensURL, content string) (lensPerStepResult, bool) {
	var zero lensPerStepResult
	if lensURL == "" || content == "" {
		return zero, false
	}
	body, err := json.Marshal(map[string]interface{}{"text": content})
	if err != nil {
		return zero, false
	}
	reqCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST",
		lensURL+"/internal/lens/score-per-step", bytes.NewReader(body))
	if err != nil {
		return zero, false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("[agent-lens] score request failed: %v", err)
		return zero, false
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return zero, false
	}
	var r lensPerStepResult
	if err := json.Unmarshal(raw, &r); err != nil {
		log.Printf("[agent-lens] score parse failed: %v", err)
		return zero, false
	}
	if !r.Enabled || r.NTokens == 0 {
		return zero, false
	}
	return r, true
}

// extractScorableContent pulls lens-scoreable text from a tool call.
// Only write_file (`content`) and edit_file (`new_str`) qualify — other
// tools either don't carry generated text (read_file, list_directory)
// or the scoring-on-shell-commands signal isn't useful. Returns the
// text and a bool indicating whether the tool was scoreable.
func extractScorableContent(toolName string, args json.RawMessage) (string, bool) {
	switch toolName {
	case "write_file":
		var p struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal(args, &p); err == nil && p.Content != "" {
			return p.Content, true
		}
	case "edit_file":
		var p struct {
			NewStr string `json:"new_str"`
		}
		if err := json.Unmarshal(args, &p); err == nil && p.NewStr != "" {
			return p.NewStr, true
		}
	}
	return "", false
}

// extractFailurePath returns the path argument of a tool call when
// the tool operates on a file (read/write/edit/structural_edit/delete/
// search/list/find/run_background's cwd). Used by the path-aware
// error-loop breaker to distinguish "stuck on one file" from
// "grinding through different files." Returns "" when no path is
// applicable to the tool (e.g. run_command's arbitrary
// shell) — empty paths compare unequal, which prevents the breaker
// from firing on tool-mix sequences.
func extractFailurePath(toolName string, args json.RawMessage) string {
	switch toolName {
	// insert_after and replace_lines were missing here, so a failing loop on
	// either returned "" and the path-aware 3-strike breaker read three
	// identical failures as three different paths and never fired.
	case "read_file", "write_file", "edit_file", "structural_edit", "delete_file", "find_file",
		"insert_after", "replace_lines":
		var p struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal(args, &p); err == nil {
			return p.Path
		}
	case "list_directory", "search_files":
		var p struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal(args, &p); err == nil {
			return p.Path
		}
	}
	return ""
}

// agentLensRegression returns the corrective message to inject (and true)
// when the recent agent-loop scoring history shows a quality crash
// pattern. Returns ("", false) when no intervention is warranted.
//
// Pattern: the most recent N (= lensRegressionRunLength) gx_score_min
// values are all below the calibrated low threshold. This is the "model is
// stuck on a stub or near-duplicate response" signature — the May 6
// resources.html loop is the canonical example.
// low and severe are the per-model thresholds (resolved from the lens score's
// bundled thresholds. Callers skip intervention when calibration is absent.
func agentLensRegression(history []float64, low, severe float64) (string, bool) {
	if len(history) == 0 {
		return "", false
	}
	// Severe single-write short-circuit: gx_min below the severe threshold
	// is so far into the "likely_incorrect" band that one sample is enough —
	// don't wait for a second confirmation while V3's sandbox-verifier
	// rubber-stamps the stub in the same iteration.
	last := history[len(history)-1]
	if last < severe {
		return fmt.Sprintf(
			"⚠ Lens severe-quality alert: the geometric lens scored your last write at "+
				"gx_min=%.3f, which is in the unambiguously-bad band (<%.2f). This usually "+
				"means the file is a stub, a placeholder, or has collapsed into a repetitive "+
				"pattern. STOP and try a different approach: (a) read a sibling file in the "+
				"same directory to model the right structure, (b) ask the user for "+
				"clarification on what concrete content is needed, or (c) skip this file and "+
				"move on if it's not blocking the verify step. DO NOT re-issue the same "+
				"write — the lens will catch it again.",
			last, severe), true
	}
	// Run-of-N moderate-low check: lensRegressionRunLength consecutive
	// scores below the calibrated low threshold. Catches gradual stub
	// loops where each write is moderately bad but no single one is
	// catastrophic.
	if len(history) < lensRegressionRunLength {
		return "", false
	}
	recent := history[len(history)-lensRegressionRunLength:]
	for _, score := range recent {
		if score >= low {
			return "", false
		}
	}
	return fmt.Sprintf(
		"⚠ Lens regression detected: the geometric lens flagged your last %d write attempts as "+
			"severely low-quality (gx_score_min values: %s). This is the signature of a stuck "+
			"or repetitive pattern — likely a stub/placeholder being submitted over and over, or "+
			"near-duplicate responses that aren't making progress. STOP and try a different "+
			"approach: (a) read a sibling file in the same directory to model the right "+
			"structure, (b) ask the user for clarification on what concrete content is needed, "+
			"or (c) skip this file and move on if it's not blocking the verify step.",
		lensRegressionRunLength, formatScoreSlice(recent)), true
}

func formatScoreSlice(s []float64) string {
	parts := make([]string, len(s))
	for i, v := range s {
		parts[i] = fmt.Sprintf("%.3f", v)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Lens training-data collection (foundation for in-the-loop labeling).
//
// As the agent runs, each file write becomes a candidate lens-training sample.
// The LABEL + WEIGHT come from human verification:
//   - per-file accept / deny  → label good / bad (review mode)
//   - per-pass 👍 / 👎          → a confidence weight on that pass's samples
// The weighting lets a thumbs-down pass down-weight even its accepted files
// (the whole approach was wrong) while keeping its denials as confident
// negatives — so "good result, one bad file" yields the cleanest data and a
// "bad overall" pass doesn't pull the lens toward a wrong pattern.
//
// Samples are appended per-model (the lens is per-model) as JSONL. Nothing
// trains here; `atlas lens retrain` consumes the corpus later. Content is
// stored raw and re-embedded at train time, so a lens/layer change doesn't
// invalidate the collection.

// LensSample is one labeled, weighted training example.
type LensSample struct {
	Content   string  `json:"content"`
	Label     int     `json:"label"`  // 1 = good (accepted), 0 = bad (denied)
	Weight    float64 `json:"weight"` // confidence, set by the pass-level verdict
	Source    string  `json:"source"` // accept | deny | thumbs | v3 | run
	Tool      string  `json:"tool,omitempty"`
	Path      string  `json:"path,omitempty"`
	PassID    string  `json:"pass_id,omitempty"`
	Timestamp string  `json:"timestamp"`
}

// PassWrite is one file the model authored during a pass, captured for later
// labeling. Content is the model's own output (what the lens scores), not the
// post-V3 winner, so a collected sample matches the score it was judged by.
type PassWrite struct {
	Tool    string
	Path    string
	Content string
}

var lensSampleMu sync.Mutex

// lensDataDir is the root for collected samples. Per-model subdirs live under
// it. Defaults to /data/lens_training (mount a volume there to persist across
// proxy restarts); override with ATLAS_LENS_DATA_DIR.
func lensDataDir() string {
	return envOr("ATLAS_LENS_DATA_DIR", "/data/lens_training")
}

// sanitizeModelName makes a model name safe for a directory component.
func sanitizeModelName(name string) string {
	if name == "" {
		return "default"
	}
	repl := func(r rune) rune {
		if r == '/' || r == '\\' || r == ':' || r == ' ' {
			return '_'
		}
		return r
	}
	return strings.Map(repl, name)
}

// appendLensSample appends one sample to the model's JSONL corpus.
func appendLensSample(model string, s LensSample) (returnErr error) {
	if s.Timestamp == "" {
		s.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}
	dir := filepath.Join(lensDataDir(), sanitizeModelName(model))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("lens-samples: mkdir %s: %w", dir, err)
	}
	line, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("lens-samples: marshal: %w", err)
	}
	lensSampleMu.Lock()
	defer lensSampleMu.Unlock()
	f, err := os.OpenFile(filepath.Join(dir, "samples.jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("lens-samples: open: %w", err)
	}
	defer func() {
		if closeErr := f.Close(); closeErr != nil && returnErr == nil {
			returnErr = fmt.Errorf("lens-samples: close: %w", closeErr)
		}
	}()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("lens-samples: write: %w", err)
	}
	return nil
}

// lensSampleCounts scans the model's corpus and returns (good, bad) counts.
// Used by the "retrain available" alert. Linear scan — fine at the scale this
// reaches before a retrain (tens of thousands of lines); switch to a sidecar
// counter if it ever becomes hot.
func lensSampleCounts(model string) (good, bad int) {
	path := filepath.Join(lensDataDir(), sanitizeModelName(model), "samples.jsonl")
	lensSampleMu.Lock()
	defer lensSampleMu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		return 0, 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for sc.Scan() {
		var s LensSample
		if json.Unmarshal(sc.Bytes(), &s) != nil {
			continue
		}
		if s.Label == 1 {
			good++
		} else {
			bad++
		}
	}
	return good, bad
}

// Pending passes await their human verdict. A pass completes (returns to the
// client) before the user rates it, so its writes are stashed by session id
// here until a /feedback call arrives — or the janitor evicts it.
type stashedPass struct {
	writes []PassWrite
	model  string
	at     time.Time
}

var (
	pendingPasses   = map[string]stashedPass{}
	pendingPassesMu sync.Mutex
)

const pendingPassTTL = 2 * time.Hour

// stashPendingPass records a completed pass's writes for deferred feedback.
// A new pass under the same session id replaces the prior one (you rate the
// most recent pass). No-op when there were no writes to label.
func stashPendingPass(sessionID, model string, writes []PassWrite) {
	if sessionID == "" || len(writes) == 0 {
		return
	}
	pendingPassesMu.Lock()
	defer pendingPassesMu.Unlock()
	// Opportunistic eviction of stale entries (no separate janitor goroutine).
	now := time.Now()
	for id, p := range pendingPasses {
		if now.Sub(p.at) > pendingPassTTL {
			delete(pendingPasses, id)
		}
	}
	cp := make([]PassWrite, len(writes))
	copy(cp, writes)
	pendingPasses[sessionID] = stashedPass{writes: cp, model: model, at: now}
}

// takePendingPass removes and returns the stashed pass for a session id.
func takePendingPass(sessionID string) (stashedPass, bool) {
	pendingPassesMu.Lock()
	defer pendingPassesMu.Unlock()
	p, ok := pendingPasses[sessionID]
	if ok {
		delete(pendingPasses, sessionID)
	}
	return p, ok
}

// feedbackVerdict maps a per-file verdict + the pass-level thumbs to a
// (label, weight, keep) for one sample. keep=false means there's no usable
// signal (e.g. no per-file verdict AND no thumbs) — don't record it.
//
//	verdict: "accept" | "deny" | ""   (""= no per-file label, thumbs-only mode)
//	thumbs:  "up" | "down" | ""        (""= pass not rated)
func feedbackVerdict(verdict, thumbs string) (label int, weight float64, keep bool) {
	switch verdict {
	case "deny":
		// A denial is a confident negative regardless of the pass verdict —
		// a bad pass's rejections are the most reliable negatives we get.
		return 0, 1.0, true
	case "accept":
		switch thumbs {
		case "up":
			return 1, 1.0, true // good result, accepted → confident positive
		case "down":
			return 1, 0.4, true // whole pass was wrong → weak positive
		default:
			return 1, 0.7, true // accepted, pass unrated → moderate positive
		}
	default:
		// No per-file verdict (thumbs-only / fast mode). The pass thumbs is the
		// only signal: it labels every write in the pass, coarsely.
		switch thumbs {
		case "up":
			return 1, 0.6, true
		case "down":
			return 0, 0.6, true
		default:
			return 0, 0, false // nothing to learn from
		}
	}
}

// lensRetrainThreshold is the labeled-sample count at which the TUI surfaces
// the "retrain available" prompt. Configurable; a balance guard (below) also
// requires enough of the minority class so the lens doesn't learn "all good".
func lensRetrainThreshold() int {
	if v := envOr("ATLAS_LENS_RETRAIN_MIN", ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			return n
		}
	}
	return 2000
}

// handleFeedback records a pass's human verdict as weighted lens samples.
// Body: {"session_id":"...", "thumbs":"up|down|", "files":[{"path":"...",
// "verdict":"accept|deny"}]}. Per-file verdicts (review mode) take precedence;
// when absent, the pass thumbs labels every write coarsely.
func handleFeedback(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, ErrUnsupported, "method not allowed")
		return
	}
	var req struct {
		SessionID string `json:"session_id"`
		Thumbs    string `json:"thumbs"`
		Files     []struct {
			Path    string `json:"path"`
			Verdict string `json:"verdict"`
		} `json:"files"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "invalid request body")
		return
	}
	pass, ok := takePendingPass(req.SessionID)
	if !ok {
		writeJSON(w, http.StatusOK, map[string]interface{}{"recorded": 0, "note": "no pending pass for that session"})
		return
	}
	verdictByPath := map[string]string{}
	for _, f := range req.Files {
		verdictByPath[f.Path] = f.Verdict
	}
	recorded := 0
	for _, wr := range pass.writes {
		verdict := verdictByPath[wr.Path]
		label, weight, keep := feedbackVerdict(verdict, req.Thumbs)
		if !keep {
			continue
		}
		source := verdict
		if source == "" {
			source = "thumbs"
		}
		if err := appendLensSample(pass.model, LensSample{
			Content: wr.Content, Label: label, Weight: weight, Source: source,
			Tool: wr.Tool, Path: wr.Path, PassID: req.SessionID,
		}); err == nil {
			recorded++
		}
	}
	good, bad := lensSampleCounts(pass.model)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"recorded": recorded, "good": good, "bad": bad,
	})
}

// handleLensTrainingStatus reports the collected-sample counts and whether a
// retrain is worth offering, so the TUI can show the banner + the command.
func handleLensTrainingStatus(w http.ResponseWriter, r *http.Request) {
	good, bad := lensSampleCounts(modelName)
	total := good + bad
	thresh := lensRetrainThreshold()
	minClass := good
	if bad < minClass {
		minClass = bad
	}
	// Need the total AND enough of the minority class (>= 25% of threshold) so
	// the corpus isn't all-positive or all-negative.
	available := total >= thresh && minClass >= thresh/4
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"model":             modelName,
		"good":              good,
		"bad":               bad,
		"total":             total,
		"threshold":         thresh,
		"retrain_available": available,
		"command":           "atlas lens retrain",
	})
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// Calibration status endpoint — surfaces lens + ASA compat for the TUI.
//
// GH #101: the geometric-lens /health endpoint already exposes the
// data we need (cost_field_dim, embed_dim, cost_field_loaded). This file
// forwards that into a verdict-shaped response under /v1/calibration/status
// that the TUI renders as a header badge.
//
// GH #113 extends the `asa` block from a file-presence check to a
// proper dim-vs-model probe; the JSON shape stays the same so TUI
// rendering doesn't churn.

// CalibrationStatus is the JSON returned by /v1/calibration/status.
// Shape is stable: TUI and atlas doctor both key off it.
type CalibrationStatus struct {
	Lens       LensStatus        `json:"lens"`
	ASA        ASAStatus         `json:"asa"`
	Dimensions []StatusDimension `json:"dimensions"`
}

// StatusDimension is one row of the canonical seven-dimension status
// (SUPPORT_MATRIX § "Reference-model status dimensions"). Separating
// these prevents the ambiguity where "the lens works" conflated model
// runtime, raw scoring, calibration, and intervention behavior. Every
// surface that shows lens/ASA status (this endpoint, the TUI badge,
// atlas doctor, atlas lens check) renders the SAME rows so they cannot
// disagree — they all read this list.
type StatusDimension struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Detail string `json:"detail"`
}

// buildDimensions maps the raw lens/ASA probe onto the seven named
// dimensions. Intervention is reported "neutral" whenever calibration is
// absent, matching the enforced runtime behavior (agent.go only applies
// thresholds when calibratedThresholds() succeeds) — a disabled/
// uncalibrated lens never steers using another model's cutoffs.
func buildDimensions(lens LensStatus, asa ASAStatus) []StatusDimension {
	reachable := lens.Verdict != "unreachable"

	modelRuntime := "supported"
	modelDetail := "model served and reachable"
	if !reachable {
		modelRuntime = "unreachable"
		modelDetail = "lens/model service not reachable"
	}

	// Identity/dimension contract.
	identity := "supported"
	identityDetail := "cost field matches the served model's dimension"
	switch {
	case !reachable:
		identity, identityDetail = "unknown", "service unreachable"
	case !lens.CostFieldLoaded:
		identity, identityDetail = "no-artifacts",
			"no cost field loaded for this model"
	case lens.EmbedDim > 0 && lens.CostFieldDim != lens.EmbedDim:
		identity, identityDetail = "dim-mismatch",
			fmt.Sprintf("cost field is %d-dim, model emits %d-dim",
				lens.CostFieldDim, lens.EmbedDim)
	}

	// Raw scoring availability.
	scoring := "disabled"
	scoringDetail := "cost field / G(x) not loaded"
	if reachable && lens.CostFieldLoaded && lens.GxLoaded {
		scoring, scoringDetail = "supported", "C(x) + G(x) scoring available"
	} else if reachable && lens.CostFieldLoaded && !lens.GxLoaded {
		scoring, scoringDetail = "partial", "C(x) loaded; G(x) missing"
	}

	// Calibration.
	calibration := "disabled"
	calDetail := "artifacts not loaded"
	if reachable && lens.CostFieldLoaded {
		if lens.CxCalibrated && lens.GxCalibrated {
			calibration, calDetail = "calibrated",
				"per-model normalization + thresholds loaded"
		} else {
			calibration, calDetail = "uncalibrated",
				"loaded without this model's calibration files"
		}
	}

	// Intervention behavior — neutral/disabled unless calibrated.
	intervention := "disabled"
	intDetail := "no scoring; no intervention"
	if calibration == "calibrated" {
		intervention, intDetail = "active",
			"threshold interventions enabled"
	} else if scoring != "disabled" {
		intervention, intDetail = "neutral",
			"raw telemetry only; no automatic intervention"
	}

	return []StatusDimension{
		{"model_runtime", modelRuntime, modelDetail},
		{"direct_agent", "supported",
			"model-agnostic; independent of lens/ASA state"},
		{"lens_identity", identity, identityDetail},
		{"lens_scoring", scoring, scoringDetail},
		{"lens_calibration", calibration, calDetail},
		{"lens_intervention", intervention, intDetail},
		{"asa", asa.Verdict, asa.Hint},
	}
}

type LensStatus struct {
	// "supported" | "no-artifacts" | "incomplete-artifacts" |
	// "uncalibrated" | "dim-mismatch" | "unreachable"
	Verdict         string `json:"verdict"`
	CostFieldLoaded bool   `json:"cost_field_loaded"`
	CostFieldDim    int    `json:"cost_field_dim"`
	EmbedDim        int    `json:"embed_dim"`
	GxLoaded        bool   `json:"gx_loaded"`
	CxCalibrated    bool   `json:"cx_calibrated"`
	GxCalibrated    bool   `json:"gx_calibrated"`
	Hint            string `json:"hint"`
}

type ASAStatus struct {
	// "supported" | "missing" | "unverified"
	Verdict       string `json:"verdict"`
	VectorPath    string `json:"vector_path"`
	VectorPresent bool   `json:"vector_present"`
	Hint          string `json:"hint"`
}

// lensHealthShape mirrors the lens /health JSON we read. Defensive — the
// service can be reachable but mid-startup with partial fields. We treat
// missing fields as zero values rather than failing the whole probe.
type lensHealthShape struct {
	Status     string `json:"status"`
	Subsystems struct {
		Lens struct {
			Enabled         bool   `json:"enabled"`
			CostFieldLoaded bool   `json:"cost_field_loaded"`
			CostFieldDim    int    `json:"cost_field_dim"`
			EmbedDim        int    `json:"embed_dim"`
			GxLoaded        bool   `json:"gx_loaded"`
			CxCalibrated    bool   `json:"cx_calibrated"`
			GxCalibrated    bool   `json:"gx_calibrated"`
			SelfTestPass    bool   `json:"self_test_pass"`
			SelfTestError   string `json:"self_test_error"`
		} `json:"lens"`
	} `json:"subsystems"`
}

// probeLensStatus calls the lens /health endpoint and renders a verdict.
// Timeout is short — this fires on a TUI startup ping and on the proxy's
// own startup banner; we don't want to block either if the lens is wedged.
func probeLensStatus(ctx context.Context, lensBaseURL string) LensStatus {
	out := LensStatus{Verdict: "unreachable",
		Hint: "geometric-lens unreachable at " + lensBaseURL +
			" (is the stack up?)"}

	pCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(pCtx, "GET", lensBaseURL+"/health", nil)
	if err != nil {
		return out
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return out
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return out
	}
	var h lensHealthShape
	if err := json.Unmarshal(body, &h); err != nil {
		out.Hint = "lens /health returned non-JSON: " + truncateStr(string(body), 80)
		return out
	}

	out.CostFieldLoaded = h.Subsystems.Lens.CostFieldLoaded
	out.CostFieldDim = h.Subsystems.Lens.CostFieldDim
	out.EmbedDim = h.Subsystems.Lens.EmbedDim
	out.GxLoaded = h.Subsystems.Lens.GxLoaded
	out.CxCalibrated = h.Subsystems.Lens.CxCalibrated
	out.GxCalibrated = h.Subsystems.Lens.GxCalibrated

	switch {
	case !out.CostFieldLoaded:
		out.Verdict = "no-artifacts"
		if h.Subsystems.Lens.SelfTestError != "" {
			out.Hint = h.Subsystems.Lens.SelfTestError
		} else {
			out.Hint = "no cost_field.pt loaded — run `atlas lens build` to train one"
		}
	case out.EmbedDim > 0 && out.CostFieldDim != out.EmbedDim:
		out.Verdict = "dim-mismatch"
		out.Hint = fmt.Sprintf("cost_field expects %d-dim, model emits %d-dim "+
			"— run `atlas lens build` to retrain at the model's native dim",
			out.CostFieldDim, out.EmbedDim)
	case !out.GxLoaded:
		out.Verdict = "incomplete-artifacts"
		out.Hint = "C(x) loaded but G(x) artifacts are missing — run `atlas lens build`"
	case !out.CxCalibrated || !out.GxCalibrated:
		out.Verdict = "uncalibrated"
		out.Hint = "Lens weights loaded without this model's calibration files — " +
			"run `atlas lens build` to generate cx_normalization.json and gx_thresholds.json"
	default:
		out.Verdict = "supported"
		out.Hint = "ready"
	}
	return out
}

// probeASAStatus checks for the configured ASA control-vector file on disk.
// The configured path is container-relative (e.g.
// /models/ast_edit_steering.gguf as llama-server sees it). The proxy
// container doesn't have /models mounted, so we try several candidate
// host-visible paths before giving up:
//
//  1. The configured path verbatim (works when proxy DOES have a /models
//     mount — some K3s deployments do).
//  2. <workspace>/models/<basename> (proxy's bind-mounted project root,
//     ATLAS_WORKSPACE_DIR, plus the standard models/ subdir).
//  3. The env-supplied ATLAS_MODELS_DIR if set.
//
// llama-server is the authoritative source of "is the vector actually
// loaded" but doesn't expose that via /props (verified 2026-05-17), so
// disk presence is the best we can do without an out-of-band probe.
// For the user-facing verdict, `atlas asa check` does the deeper GGUF
// dim parse on the host — this endpoint is the "first impression" the
// TUI badge renders.
func probeASAStatus() ASAStatus {
	configured := envOr("ATLAS_CONTROL_VECTOR", "/models/ast_edit_steering.gguf")
	out := ASAStatus{VectorPath: configured, Verdict: "unverified"}

	// Candidate paths to probe, in order.
	candidates := []string{configured}
	if strings.HasPrefix(configured, "/models/") {
		base := strings.TrimPrefix(configured, "/models/")
		workspace := envOr("ATLAS_WORKSPACE_DIR", "/workspace")
		candidates = append(candidates,
			workspace+"/models/"+base)
		if mdir := os.Getenv("ATLAS_MODELS_DIR"); mdir != "" {
			candidates = append(candidates, mdir+"/"+base)
		}
	}

	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil {
			out.VectorPresent = true
			out.VectorPath = p
			expected := os.Getenv("ATLAS_MODEL_NAME")
			markedFor := ""
			if raw, readErr := os.ReadFile(p + ".model"); readErr == nil {
				markedFor = strings.TrimSpace(string(raw))
			}
			size := strconv.FormatInt(info.Size(), 10)
			switch {
			case expected != "" && sameModelIdentity(markedFor, expected):
				out.Verdict = "supported"
				out.Hint = "control vector verified for " + expected +
					" (" + size + " bytes)"
			case expected != "" && markedFor != "":
				out.Verdict = "incompatible"
				out.Hint = "control vector is marked for " + markedFor +
					", but the selected model is " + expected
			default:
				out.Verdict = "unverified"
				out.Hint = "control vector present (" + size +
					" bytes) without a matching model marker; run `atlas asa build`"
			}
			return out
		}
	}

	out.VectorPresent = false
	out.Verdict = "missing"
	out.Hint = "no control vector at " + configured +
		" (also tried workspace/models/ + ATLAS_MODELS_DIR) — " +
		"build one via `atlas asa build` " +
		"or see geometric-lens/asa_calibration/README.md"
	return out
}

func sameModelIdentity(a, b string) bool {
	canonical := func(value string) string {
		value = strings.ToLower(strings.TrimSpace(value))
		value = strings.TrimSuffix(value, ".gguf")
		if slash := strings.LastIndex(value, "/"); slash >= 0 {
			value = value[slash+1:]
		}
		return value
	}
	return canonical(a) != "" && canonical(a) == canonical(b)
}

func handleCalibrationStatus(w http.ResponseWriter, r *http.Request) {
	lens := probeLensStatus(r.Context(), lensURL)
	asa := probeASAStatus()
	status := CalibrationStatus{
		Lens:       lens,
		ASA:        asa,
		Dimensions: buildDimensions(lens, asa),
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(status)
}

// logCalibrationStatusAtStartup is called once from main() so operators
// see the same compat verdict the TUI will render, in the proxy banner.
// Fail-soft: if the lens service isn't reachable yet, we log it and move
// on — startup blocks long enough as-is without a synchronous probe.
func logCalibrationStatusAtStartup() {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	lens := probeLensStatus(ctx, lensURL)
	asa := probeASAStatus()
	log.Printf("  Lens: %s — %s", lens.Verdict, lens.Hint)
	log.Printf("  ASA:  %s — %s", asa.Verdict, asa.Hint)
}

// ---------------------------------------------------------------------------
// Mechanical labelling
// ---------------------------------------------------------------------------
//
// The lens corpus was fed by exactly one thing: POST /feedback, a human
// thumbs-up/down or per-file accept/deny. LensSample.Source has always
// advertised "v3" and "run" alongside them, and nothing ever wrote either.
//
// Meanwhile the harness produces ground truth on every turn and threw it
// away. A gate rejection is a deterministic negative — the content provably
// breaks syntax, breaks the embedded script, orphans a call, stops a render
// loop, duplicates a binding. Twelve instrumented runs generated dozens of
// those and recorded nothing, because nobody clicked. The corpus directory
// was empty.
//
// These are the labels the retrain loop (`atlas bench` ->
// retrain_lens_from_results.py -> `atlas asa build`) exists to consume, and
// they cost nothing to collect.

// recordGateRejection labels content a write gate refused. Deterministic
// ground truth, so it carries full weight — a gate does not have opinions.
func recordGateRejection(model, tool, path, content, reason string) {
	if content == "" || model == "" {
		return
	}
	if err := appendLensSample(model, LensSample{
		Content: content, Label: 0, Weight: 1.0, Source: "gate",
		Tool: tool, Path: path,
	}); err != nil {
		log.Printf("[lens] could not record gate rejection for %s: %v", logPath(path), err)
	}
}

// recordVerifiedPass labels a run's writes after a verification command
// passed and no gate refused them.
//
// Weight 0.5, deliberately below an accepted-and-thumbed-up human sample.
// "A verification command exited clean" is a weaker claim than it sounds: an
// observed run had `curl` return 200 over a page whose game loop was dead,
// and another over a Flask app with no routes left in it. It is real evidence
// and it is not proof, so it should not outvote a human who looked.
func recordVerifiedPass(model string, writes []PassWrite) int {
	recorded := 0
	for _, w := range writes {
		if w.Content == "" {
			continue
		}
		if err := appendLensSample(model, LensSample{
			Content: w.Content, Label: 1, Weight: 0.5, Source: "run",
			Tool: w.Tool, Path: w.Path,
		}); err == nil {
			recorded++
		}
	}
	return recorded
}

// authoredContent pulls the text the MODEL wrote out of a tool call's
// arguments — `content` for the write/insert/replace tools, `new_str` for
// edit_file. Empty for tools that author nothing (reads, run_command) and for
// calls that failed input validation before any content existed, which is
// what keeps those out of the corpus.
func authoredContent(args json.RawMessage) string {
	var a struct {
		Content string `json:"content"`
		NewStr  string `json:"new_str"`
	}
	if json.Unmarshal(args, &a) != nil {
		return ""
	}
	if a.Content != "" {
		return a.Content
	}
	return a.NewStr
}

// rejectedPath pulls the target path for the sample's provenance.
func rejectedPath(args json.RawMessage) string {
	var a struct {
		Path string `json:"path"`
	}
	if json.Unmarshal(args, &a) != nil {
		return ""
	}
	return a.Path
}
