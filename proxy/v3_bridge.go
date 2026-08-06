package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// v3CallTimeout is the interactive wall-clock cap for a single V3 pipeline
// call. It bounds the long-tail repair stall that left a user waiting 11 min.
// Set ATLAS_V3_TIMEOUT to a second count to override, or 0 to disable the cap
// (restores the May-10 uncapped behavior for offline bench runs).
//
// 300s, not the 180s this shipped with. 180 could not contain the pipeline it
// was capping: PlanSearch spends two LLM calls per candidate, so k=3 costs
// ~162s at the measured ~22s per call, before the probe and the self-test
// that precede it. Measured across 43 runs, sessions spent a median 207s of a
// 180s budget on generation alone and phase-3 repair was reached with 7-9s
// left and skipped 19 times. The corrected budget cap responds by cutting k
// to 1, which removes the candidates candidate-agreement needs to compare.
func v3CallTimeout() time.Duration {
	if v := os.Getenv("ATLAS_V3_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n >= 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 300 * time.Second
}

// ---------------------------------------------------------------------------
// V3 Bridge — Go ↔ Python V3 service communication
// ---------------------------------------------------------------------------

// V3ProgressFn is called for each V3 pipeline progress event.
// `data` carries the structured payload from the V3 service (since the
// 2026-05 observability pass) — counts, timings, indices, strategy
// labels. Empty for legacy stages that haven't been enriched yet; the
// proxy bridge falls back to the human-readable `detail` in that case.
type V3ProgressFn func(stage, detail string, data map[string]interface{})

// callV3GenerateStreaming sends a file generation request to the V3 Python
// service and streams progress events back via the callback. Returns the
// final result when the pipeline completes.
func callV3GenerateStreaming(reqCtx context.Context, v3URL string, req V3GenerateRequest, onProgress V3ProgressFn) (*V3GenerateResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal V3 request: %w", err)
	}

	endpoint := v3URL + "/v3/generate"
	// Bind the agent's request context so a user Ctrl-C (which cancels
	// ctx.Ctx via /cancel) actually aborts an in-flight V3 run. The
	// May-10 comment claimed cancellation "still works", but http.NewRequest
	// carried no context, so Ctrl-C could not stop a multi-minute
	// PlanSearch — exactly the "ctrl-c does not stop it" report. nil falls
	// back to Background for any non-agent caller.
	if reqCtx == nil {
		reqCtx = context.Background()
	}

	// Interactive wall-clock cap. The May-10 design ran V3 uncapped so a
	// >15-min Phase-3 repair could finish; that's right for an offline bench
	// run but unshippable interactively — observed an 11-min stall on a
	// 103-line write_file while a user waited. Cap the agent path so a runaway
	// pipeline falls back to the model's own content (all three callers treat
	// a V3 error as "write the baseline") instead of hanging the session. The
	// model's content is already syntax-gated, so the fallback is safe.
	// ATLAS_V3_TIMEOUT (seconds) overrides; 0 restores the uncapped behavior
	// for bench/offline use.
	if d := v3CallTimeout(); d > 0 {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithTimeout(reqCtx, d)
		defer cancel()
	}

	httpReq, err := http.NewRequestWithContext(reqCtx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create V3 request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Abort is user-driven via the bound request context, plus the
	// interactive deadline applied above.
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("V3 service call failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("V3 service returned %d", resp.StatusCode)
	}

	// Read SSE stream: progress events + final result
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<20) // 1MB buffer

	var result *V3GenerateResponse
	var resultErr error

	for scanner.Scan() {
		line := scanner.Text()

		// Final result event
		if strings.HasPrefix(line, "event: result") {
			// Next line has the data
			if scanner.Scan() {
				dataLine := scanner.Text()
				if strings.HasPrefix(dataLine, "data: ") {
					data := strings.TrimPrefix(dataLine, "data: ")
					var r V3GenerateResponse
					if err := json.Unmarshal([]byte(data), &r); err != nil {
						resultErr = fmt.Errorf("V3 sent a result this proxy could not decode "+
							"(%d bytes): %w", len(data), err)
					} else {
						result = &r
						resultErr = nil
					}
				}
			}
			continue
		}

		// Done marker
		if line == "data: [DONE]" {
			break
		}

		// Progress event
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var event struct {
				Stage  string                 `json:"stage"`
				Detail string                 `json:"detail"`
				Data   map[string]interface{} `json:"data"`
			}
			if json.Unmarshal([]byte(data), &event) == nil && onProgress != nil {
				onProgress(event.Stage, event.Detail, event.Data)
			}
		}
	}

	if result == nil {
		// "completed without result" was the message for every one of
		// these, including the case where this side hung up: the context
		// deadline cancels the request, Scan returns false, and V3 is
		// blamed for finishing empty while it was still working. Report
		// which of them actually happened.
		switch {
		case resultErr != nil:
			return nil, resultErr
		case reqCtx.Err() != nil:
			limit := v3CallTimeout()
			return nil, fmt.Errorf("V3 was still working when this proxy hung up at the "+
				"%s cap (ATLAS_V3_TIMEOUT); its work is discarded and the write falls back "+
				"to the model's own output: %w", limit, reqCtx.Err())
		case scanner.Err() != nil:
			return nil, fmt.Errorf("V3 stream ended early: %w", scanner.Err())
		default:
			return nil, fmt.Errorf("V3 stream closed without sending a result event")
		}
	}

	return result, nil
}

// callV3PlanStreaming sends a plan-generation request to the V3 Python
// service and streams plan_* progress events through the callback.
// Returns the winning Plan when the planner finishes.
//
// Plans are cheap-but-not-free: 3 candidate samples × ~5s each = ~15s
// wall time, mostly dominated by the LLM. Timeout is 5 min — well above
// expected (15s) and below the agent loop's overall request timeout.
func callV3PlanStreaming(reqCtx context.Context, v3URL string, req V3PlanRequest, onProgress V3ProgressFn) (*Plan, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal plan request: %w", err)
	}

	endpoint := v3URL + "/v3/plan"
	// Bind the agent's request context so a user Ctrl-C (via /cancel)
	// aborts an in-flight plan run, mirroring callV3GenerateStreaming.
	// nil falls back to Background for any non-agent caller.
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	httpReq, err := http.NewRequestWithContext(reqCtx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create plan request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// May 10 2026: timeout removed (was 5 min). Plan generation can run
	// long on multi-candidate scoring; bounding it via the client
	// timeout killed slow-but-progressing calls. Abort is user-driven
	// via the bound request context.
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("plan call failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("plan service returned %d", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	// The plan result is one SSE `data:` line: a flat step list plus rationale
	// and scoring metadata. 1MB comfortably bounds it while still clearing the
	// 64KB scanner default, which a verbose plan could trip (ErrTooLong would
	// silently drop the plan).
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)

	var plan *Plan

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: result") {
			if scanner.Scan() {
				dataLine := scanner.Text()
				if strings.HasPrefix(dataLine, "data: ") {
					data := strings.TrimPrefix(dataLine, "data: ")
					var p Plan
					if json.Unmarshal([]byte(data), &p) == nil {
						plan = &p
					}
				}
			}
			continue
		}

		if line == "data: [DONE]" {
			break
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			var event struct {
				Stage  string                 `json:"stage"`
				Detail string                 `json:"detail"`
				Data   map[string]interface{} `json:"data"`
			}
			if json.Unmarshal([]byte(data), &event) == nil && onProgress != nil {
				onProgress(event.Stage, event.Detail, event.Data)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		// Most likely bufio.ErrTooLong on an oversized result line. Surface it
		// so the cause is diagnosable instead of a misleading "no result".
		return nil, fmt.Errorf("reading plan stream: %w", err)
	}

	if plan == nil {
		return nil, fmt.Errorf("plan service completed without result")
	}

	return plan, nil
}
