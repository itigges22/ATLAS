package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakePlanServer streams a canned SSE plan response that mirrors what
// v3-service actually emits. Useful so the bridge test doesn't depend on
// the live Python service.
func fakePlanServer(t *testing.T, sse string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v3/plan" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("test server doesn't support flushing")
		}
		w.WriteHeader(http.StatusOK)
		f.Flush()
		fmt.Fprint(w, sse)
		f.Flush()
	}))
}

func TestCallV3PlanStreamingParsesResult(t *testing.T) {
	// Three progress events, then a final result event, then [DONE].
	// Mirrors the wire format of /v3/plan.
	sse := strings.Join([]string{
		`data: {"stage":"plan_start","detail":"generating 3 candidates"}`,
		``,
		`data: {"stage":"plan_candidate_scored","detail":"candidate 1 score=0.80","data":{"index":0,"score":0.8}}`,
		``,
		`data: {"stage":"plan_selected","detail":"plan 1 won","data":{"index":0,"score":0.8}}`,
		``,
		`event: result`,
		`data: {"steps":[{"id":"s1","action":"edit_file","target":"app.py","why":"add route"},{"id":"s2","action":"run_command","target":"curl http://localhost:5000/hello","why":"verify"}],"verify_step":"s2","rationale":"add then verify","candidates_tested":3,"winning_score":0.8,"winning_index":0,"reasons":["step count 2 in range","verify_step=s2"]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := fakePlanServer(t, sse)
	defer srv.Close()

	var mu sync.Mutex
	var seenStages []string
	cb := func(stage, detail string, data map[string]interface{}) {
		mu.Lock()
		seenStages = append(seenStages, stage)
		mu.Unlock()
	}

	plan, err := callV3PlanStreaming(context.Background(), srv.URL, V3PlanRequest{
		UserMessage: "add a hello endpoint",
		WorkingDir:  "/workspace",
	}, cb)
	if err != nil {
		t.Fatalf("callV3PlanStreaming: %v", err)
	}
	if plan == nil {
		t.Fatal("plan is nil")
	}
	if got, want := len(plan.Steps), 2; got != want {
		t.Errorf("got %d steps, want %d", got, want)
	}
	if plan.VerifyStep != "s2" {
		t.Errorf("got verify_step=%q, want %q", plan.VerifyStep, "s2")
	}
	if plan.WinningScore != 0.8 {
		t.Errorf("got winning_score=%v, want 0.8", plan.WinningScore)
	}

	mu.Lock()
	defer mu.Unlock()
	wantStages := []string{"plan_start", "plan_candidate_scored", "plan_selected"}
	if len(seenStages) != len(wantStages) {
		t.Fatalf("got stages %v, want %v", seenStages, wantStages)
	}
	for i, s := range wantStages {
		if seenStages[i] != s {
			t.Errorf("stage[%d]=%q, want %q", i, seenStages[i], s)
		}
	}
}

func TestCallV3PlanStreamingMissingResult(t *testing.T) {
	// SSE that ends without an `event: result` block — bridge should
	// surface this as an error rather than returning nil silently.
	sse := strings.Join([]string{
		`data: {"stage":"plan_start","detail":"go"}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := fakePlanServer(t, sse)
	defer srv.Close()

	_, err := callV3PlanStreaming(context.Background(), srv.URL, V3PlanRequest{UserMessage: "x"}, nil)
	if err == nil {
		t.Fatal("expected error for missing result event")
	}
	if !strings.Contains(err.Error(), "without result") {
		t.Errorf("error %q doesn't mention missing result", err.Error())
	}
}

func TestV3StageToEventCoversPlanStages(t *testing.T) {
	planStages := []string{
		"plan_start", "plan_candidate", "plan_candidate_unparseable",
		"plan_candidate_error", "plan_candidate_scored", "plan_selected",
		"plan_failed",
	}
	for _, s := range planStages {
		if got := v3StageToEvent(s); got != "v3_plan" {
			t.Errorf("v3StageToEvent(%q) = %q, want v3_plan", s, got)
		}
	}
}

// ---------------------------------------------------------------------------
// callV3GenerateStreaming + v3CallTimeout — the generate path. All three
// proxy callers treat a bridge error as "fall back to the model's own
// content", so every failure path must return, not hang.
// ---------------------------------------------------------------------------

func TestV3CallTimeout(t *testing.T) {
	// 300s, not 180s: PlanSearch spends two LLM calls per candidate, so k=3
	// costs ~162s at the measured ~22s per call before the probe and
	// self-test that precede it. At 180s, sessions spent a median 207s on
	// generation alone and phase-3 repair was skipped 19 times with 7-9s
	// left.
	t.Run("default is 300s", func(t *testing.T) {
		t.Setenv("ATLAS_V3_TIMEOUT", "")
		if d := v3CallTimeout(); d != 300*time.Second {
			t.Errorf("default = %v", d)
		}
	})
	t.Run("env override in seconds", func(t *testing.T) {
		t.Setenv("ATLAS_V3_TIMEOUT", "30")
		if d := v3CallTimeout(); d != 30*time.Second {
			t.Errorf("override = %v", d)
		}
	})
	t.Run("zero disables the cap", func(t *testing.T) {
		t.Setenv("ATLAS_V3_TIMEOUT", "0")
		if d := v3CallTimeout(); d != 0 {
			t.Errorf("0 should disable, got %v", d)
		}
	})
	t.Run("garbage falls back to default", func(t *testing.T) {
		t.Setenv("ATLAS_V3_TIMEOUT", "soon")
		if d := v3CallTimeout(); d != 300*time.Second {
			t.Errorf("garbage value gave %v", d)
		}
	})
}

func fakeGenerateServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v3/generate" {
			http.NotFound(w, r)
			return
		}
		handler(w, r)
	}))
}

func sseLines(w http.ResponseWriter, lines ...string) {
	fl, _ := w.(http.Flusher)
	for _, l := range lines {
		fmt.Fprint(w, l+"\n")
		if fl != nil {
			fl.Flush()
		}
	}
}

func TestCallV3GenerateStreamingParsesResultAndProgress(t *testing.T) {
	srv := fakeGenerateServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseLines(w,
			`data: {"stage":"plan_search","detail":"3 candidates","data":{"count":3}}`,
			``,
			`event: result`,
			`data: {"code":"print(1)","passed":true,"phase_solved":"phase1","candidates_tested":3}`,
			``,
			`data: [DONE]`,
			``)
	})
	defer srv.Close()

	var stages []string
	var gotData map[string]interface{}
	result, err := callV3GenerateStreaming(context.Background(), srv.URL,
		V3GenerateRequest{FilePath: "a.py"},
		func(stage, detail string, data map[string]interface{}) {
			stages = append(stages, stage)
			gotData = data
		})
	if err != nil {
		t.Fatalf("streaming call failed: %v", err)
	}
	if result.Code != "print(1)" || !result.Passed {
		t.Errorf("result = %+v", result)
	}
	if len(stages) != 1 || stages[0] != "plan_search" {
		t.Errorf("progress stages = %v", stages)
	}
	if gotData["count"] != float64(3) {
		t.Errorf("structured progress data = %v", gotData)
	}
}

func TestCallV3GenerateStreamingNon200IsAnError(t *testing.T) {
	srv := fakeGenerateServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	})
	defer srv.Close()

	_, err := callV3GenerateStreaming(context.Background(), srv.URL,
		V3GenerateRequest{}, nil)
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Errorf("err = %v, want the 503 surfaced", err)
	}
}

func TestCallV3GenerateStreamingMissingResultIsAnError(t *testing.T) {
	// Progress events then [DONE] with no result event — the pipeline
	// died server-side. Must be an error, not a nil result.
	srv := fakeGenerateServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseLines(w,
			`data: {"stage":"plan_search","detail":"working"}`,
			``,
			`data: [DONE]`,
			``)
	})
	defer srv.Close()

	_, err := callV3GenerateStreaming(context.Background(), srv.URL,
		V3GenerateRequest{}, nil)
	if err == nil || !strings.Contains(err.Error(), "without sending a result event") {
		t.Errorf("err = %v, want a stream that really ended empty", err)
	}
}

func TestCallV3GenerateStreamingUndecodableResultNamesTheDecodeFailure(t *testing.T) {
	// An unmarshal failure used to be discarded, leaving result nil and
	// reporting the same message as a stream that sent nothing at all.
	srv := fakeGenerateServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseLines(w, `event: result`, `data: {not json`, ``, `data: [DONE]`, ``)
	})
	defer srv.Close()

	_, err := callV3GenerateStreaming(context.Background(), srv.URL,
		V3GenerateRequest{}, nil)
	if err == nil || !strings.Contains(err.Error(), "could not decode") {
		t.Errorf("err = %v, want the decode failure named", err)
	}
}

func TestCallV3GenerateStreamingTimeoutFires(t *testing.T) {
	t.Setenv("ATLAS_V3_TIMEOUT", "1")
	// `release` unblocks the stalled handler after the call returns:
	// srv.Close() waits for active handlers, and server-side disconnect
	// detection isn't reliable enough to end the stall on its own.
	release := make(chan struct{})
	srv := fakeGenerateServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseLines(w, `data: {"stage":"plan_search","detail":"stalling"}`, ``)
		select { // stall past the cap
		case <-r.Context().Done():
		case <-release:
		}
	})
	defer srv.Close()
	// LIFO: runs before srv.Close(), releasing the stalled handler.
	defer close(release)

	start := time.Now()
	_, err := callV3GenerateStreaming(context.Background(), srv.URL,
		V3GenerateRequest{}, nil)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("stalled V3 run did not time out")
	}
	if elapsed > 5*time.Second {
		t.Errorf("timeout took %v with a 1s cap", elapsed)
	}
	// The cap firing and V3 finishing empty are different events with
	// different fixes, and both reported the same sentence. Measured
	// 2026-08-03: five caps in one run read as V3 producing nothing,
	// while it was still working and its output was being discarded.
	if strings.Contains(err.Error(), "without sending a result event") {
		t.Errorf("the cap firing is reported as V3 finishing empty:\n%s", err)
	}
	if !strings.Contains(err.Error(), "ATLAS_V3_TIMEOUT") {
		t.Errorf("the error should name the cap that fired:\n%s", err)
	}
}

func TestCallV3GenerateStreamingCancelAborts(t *testing.T) {
	// User Ctrl-C: cancelling the request context must abort a stalled
	// stream promptly — the regression this guards is the "ctrl-c does
	// not stop it" multi-minute PlanSearch hang.
	release := make(chan struct{})
	srv := fakeGenerateServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		sseLines(w, `data: {"stage":"plan_search","detail":"stalling"}`, ``)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	})
	defer srv.Close()
	// LIFO: runs before srv.Close(), releasing the stalled handler.
	defer close(release)

	reqCtx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(150 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := callV3GenerateStreaming(reqCtx, srv.URL, V3GenerateRequest{}, nil)
	if err == nil {
		t.Fatal("cancelled V3 run returned no error")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("cancel took %v to unblock", elapsed)
	}
}

// generatePlan is the agent loop's entry into V3 planning. The transport
// under it is covered above; what is only here is the progress callback it
// installs, and that callback exists to drop token-level events. The
// planner asks for 3 candidates and the LLM emits ~150 token deltas each,
// so forwarding them verbatim puts ~450 v3_plan rows into the TUI's
// pipeline pane for one plan — the same flood that had to be fixed once
// already for V3 generation. Only the structural stages belong on the
// stream.
func TestGeneratePlanDropsTokenNoiseFromTheStream(t *testing.T) {
	var sse []string
	add := func(line string) { sse = append(sse, line, "") }
	add(`data: {"stage":"plan_start","detail":"generating 3 candidates"}`)
	add(`data: {"stage":"llm_start","detail":"candidate 0"}`)
	for i := 0; i < 40; i++ { // stand-in for the ~450 real deltas
		add(`data: {"stage":"token","detail":"tok"}`)
	}
	add(`data: {"stage":"llm_end","detail":"candidate 0"}`)
	add(`data: {"stage":"plan_candidate_scored","detail":"candidate 1 score=0.80","data":{"index":0,"score":0.8}}`)
	add(`data: {"stage":"plan_selected","detail":"plan 1 won","data":{"index":0,"score":0.8}}`)
	sse = append(sse,
		`event: result`,
		`data: {"steps":[{"id":"s1","action":"edit_file","target":"app.py","why":"add route"}],"verify_step":"s1","rationale":"r","candidates_tested":3,"winning_score":0.8,"winning_index":0}`,
		``,
		`data: [DONE]`,
		``)
	srv := fakePlanServer(t, strings.Join(sse, "\n"))
	defer srv.Close()

	var mu sync.Mutex
	var stages []string
	ctx := &AgentContext{
		Ctx:        context.Background(),
		V3URL:      srv.URL,
		WorkingDir: t.TempDir(),
	}
	ctx.StreamFn = func(kind string, data interface{}) {
		if kind != "v3_plan" {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		if m, ok := data.(map[string]interface{}); ok {
			stages = append(stages, fmt.Sprint(m["stage"]))
		}
	}

	plan := generatePlan(ctx, "add a hello endpoint")
	if plan == nil {
		t.Fatal("generatePlan returned nil on a well-formed plan stream")
	}

	mu.Lock()
	defer mu.Unlock()
	for _, s := range stages {
		if s == "token" || s == "llm_start" || s == "llm_end" {
			t.Errorf("token-level stage %q reached the TUI stream", s)
		}
	}
	// The structural stages are the whole point of streaming at all.
	want := map[string]bool{"plan_start": false, "plan_candidate_scored": false, "plan_selected": false}
	for _, s := range stages {
		if _, ok := want[s]; ok {
			want[s] = true
		}
	}
	for s, seen := range want {
		if !seen {
			t.Errorf("structural stage %q never reached the stream (got %v)", s, stages)
		}
	}
}

// No v3-service configured means no planner. Returning nil here is what
// makes plan mode degrade to a plain agent loop instead of erroring.
func TestGeneratePlanWithoutV3URLIsNil(t *testing.T) {
	if p := generatePlan(&AgentContext{Ctx: context.Background()}, "do a thing"); p != nil {
		t.Errorf("expected nil plan with no V3URL, got %+v", p)
	}
}

// V3 must generate against the human's request, never a harness note.
// ATLAS rides correctives/manifests on user-role messages for chat-template
// compatibility, so "last user turn" is the wrong question to ask the
// conversation (third-party audit finding: V3 received "run the program
// standalone" as its task).
func TestLatestUserMessagePrefersHumanTask(t *testing.T) {
	ctx := &AgentContext{
		HumanTask: "write a debounce filter over readings.txt",
		Messages: []AgentMessage{
			{Role: "user", Content: "write a debounce filter over readings.txt"},
			{Role: "assistant", Content: `{"type":"tool_call"}`},
			{Role: "user", Content: "[system note]: run the program standalone"},
			{Role: "user", Content: "[system note]: session file manifest: solve.py"},
		},
	}
	if got := latestUserMessage(ctx); got != ctx.HumanTask {
		t.Fatalf("V3 task resolved to %q, want the human request", got)
	}
}

func TestLatestUserMessageFallbackSkipsSyntheticNotes(t *testing.T) {
	ctx := &AgentContext{ // no HumanTask: context built outside the loop
		Messages: []AgentMessage{
			{Role: "user", Content: "the real task"},
			{Role: "user", Content: "[system note]: lessons from previous sessions"},
		},
	}
	if got := latestUserMessage(ctx); got != "the real task" {
		t.Fatalf("fallback resolved to %q, want the real task", got)
	}
}

// Authorization to replace the caller's content is `passed`, never the mere
// presence of `code`. The evidence work introduces a "best_record" that is
// the strongest available candidate while deliberately NOT closure-eligible.
//
// The previous version of this test reimplemented the condition inline, so
// it tested the intended expression rather than production code — and stayed
// green while improveContentWithV3 still took Code unconditionally. It now
// calls the shared helper both paths use.
func TestAuthorizedV3ReplacementIsTheOnlyGate(t *testing.T) {
	baseline := "def solve():\n    return 41\n"
	alternative := "def solve():\n    return 42  # best_record, not verified\n"

	for _, tc := range []struct {
		name       string
		result     *V3GenerateResponse
		want       string
		authorized bool
	}{
		{"passing candidate is delivered",
			&V3GenerateResponse{Passed: true, Code: alternative}, alternative, true},
		{"unverified candidate is refused",
			&V3GenerateResponse{Passed: false, Code: alternative}, baseline, false},
		{"passing but empty falls back",
			&V3GenerateResponse{Passed: true, Code: ""}, baseline, false},
		{"unverified and empty falls back",
			&V3GenerateResponse{Passed: false, Code: ""}, baseline, false},
		{"nil result falls back", nil, baseline, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := authorizedV3Replacement(tc.result, baseline)
			if got != tc.want || ok != tc.authorized {
				t.Fatalf("got (%q, %v), want (%q, %v)", got, ok, tc.want, tc.authorized)
			}
		})
	}
}

// Provenance must describe the FINAL bytes, not the initial response.
//
// The previous test here asserted `V3EditMetadata{}.Used == false`, which
// only proves Go's zero value is false — it never called either delivery
// function. That is the fourth mirror-test in this workstream, so this one
// drives the real gates.
func TestBaselineRestoringGatesWithdrawV3Provenance(t *testing.T) {
	// A candidate that PASSED but is refused downstream: HTML replaced by
	// JavaScript, which the language-swap gate rejects.
	htmlBaseline := "<!DOCTYPE html>\n<html><body><canvas id=\"c\"></canvas></body></html>\n"
	jsCandidate := "const c = document.getElementById('c');\nc.getContext('2d');\n"

	if why := v3SwappedTheLanguage("index.html", htmlBaseline, jsCandidate); why == "" {
		t.Fatal("fixture invalid: the gate must reject JS replacing HTML")
	}

	// The transition the production path now uses.
	code, authorized, fellBack := revokeV3(htmlBaseline, "language swap", "index.html")
	if code != htmlBaseline {
		t.Fatalf("baseline not restored: %q", code)
	}
	if authorized {
		t.Fatal("a gate that restores the baseline must withdraw authorization")
	}
	if !fellBack {
		t.Fatal("fallback must be recorded so no V3 metadata attaches")
	}
}

// Bytes and provenance must not be assignable independently: every branch
// that restores the caller's content has to go through the transition.
func TestNoGateRestoresBaselineWithoutRevoking(t *testing.T) {
	src, err := os.ReadFile("tools.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	writeFn := body[strings.Index(body, "func writeFileWithV3("):]
	writeFn = writeFn[:strings.Index(writeFn, "\nfunc ")]
	if strings.Contains(writeFn, "code = baselineContent") {
		t.Fatal("a gate assigns baseline bytes directly; use revokeV3 so provenance follows")
	}
	if strings.Count(writeFn, "revokeV3(") < 3 {
		t.Fatalf("expected every baseline-restoring gate to revoke, found %d",
			strings.Count(writeFn, "revokeV3("))
	}
}

// Both delivery paths must route through the one helper — a duplicated
// safety condition is how half of it goes stale, which is exactly what
// happened when write_file was fixed and improveContentWithV3 was not.
func TestBothDeliveryPathsUseTheSharedAuthorization(t *testing.T) {
	src, err := os.ReadFile("tools.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	if strings.Count(body, "authorizedV3Replacement(") < 3 {
		t.Fatal("expected the helper definition plus both call sites")
	}
	for _, unsafe := range []string{"chosen := v3Result.Code", "code := v3Result.Code"} {
		if strings.Contains(body, unsafe) {
			t.Fatalf("delivery path still takes Code without authorization: %q", unsafe)
		}
	}
}
