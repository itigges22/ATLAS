package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
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

	plan, err := callV3PlanStreaming(srv.URL, V3PlanRequest{
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

	_, err := callV3PlanStreaming(srv.URL, V3PlanRequest{UserMessage: "x"}, nil)
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
