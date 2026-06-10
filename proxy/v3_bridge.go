package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

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
func callV3GenerateStreaming(v3URL string, req V3GenerateRequest, onProgress V3ProgressFn) (*V3GenerateResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal V3 request: %w", err)
	}

	endpoint := v3URL + "/v3/generate"
	httpReq, err := http.NewRequest("POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create V3 request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// May 10 2026: timeout removed. V3 pipelines can run >15 min when
	// they hit Phase 3 repair on a difficult edit; capping the call
	// killed otherwise-working runs. Cancellation via request context
	// still works for user-initiated aborts.
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
					if json.Unmarshal([]byte(data), &r) == nil {
						result = &r
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
		return nil, fmt.Errorf("V3 service completed without result")
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
func callV3PlanStreaming(v3URL string, req V3PlanRequest, onProgress V3ProgressFn) (*Plan, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal plan request: %w", err)
	}

	endpoint := v3URL + "/v3/plan"
	httpReq, err := http.NewRequest("POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create plan request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// May 10 2026: timeout removed (was 5 min). Plan generation can run
	// long on multi-candidate scoring; bounding it via the client
	// timeout killed slow-but-progressing calls. Request context
	// cancellation still works.
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
	// The plan result is one SSE `data:` line. With V3.2 RPG planning it now
	// carries the whole graph (files + signatures + edges) plus per-step
	// constraints, so the line is far larger than a flat plan. Allow up to 16MB
	// so a big graph degrades gracefully instead of tripping ErrTooLong and
	// silently dropping the plan (#120 review).
	scanner.Buffer(make([]byte, 0, 1<<20), 16<<20)

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

// callV3Score sends code to the Geometric Lens for C(x)/G(x) scoring.
func callV3Score(lensURL, code string) (*LensScore, error) {
	body, _ := json.Marshal(map[string]string{"text": code})

	endpoint := lensURL + "/score"
	resp, err := http.Post(endpoint, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("score request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("score returned %d", resp.StatusCode)
	}

	var score LensScore
	if err := json.NewDecoder(resp.Body).Decode(&score); err != nil {
		return nil, fmt.Errorf("decode score: %w", err)
	}

	return &score, nil
}
