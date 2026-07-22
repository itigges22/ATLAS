package main

// PC-207 agent-loop integration. Scores write_file / edit_file content
// per-tool-call via geometric-lens /internal/lens/score-per-step. Tracks
// recent gx_score_min values per session so a "stub loop" pattern (the
// kind that hit the May 6 templates/resources.html session in production)
// can be detected and broken with a corrective system message before the
// next LLM call.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

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
	case "read_file", "write_file", "edit_file", "structural_edit", "delete_file", "find_file":
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
