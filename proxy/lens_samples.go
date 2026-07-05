package main

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

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

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
