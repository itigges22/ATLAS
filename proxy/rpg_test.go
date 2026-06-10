package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

// A plan result carrying RPG-projected steps (node_id + constraints) and the
// full graph, as v3-service/rpg.py emits it.
const rpgPlanJSON = `{
  "steps": [
    {"id": "s1", "action": "write_file", "target": "src/load.py", "why": "Loading",
     "node_id": "f1", "constraints": ["Implement ` + "`def load(p: str) -> list`" + `", "Produces rows consumed by src/process.py"]},
    {"id": "s2", "action": "write_file", "target": "src/process.py", "why": "Processing",
     "node_id": "f2", "constraints": ["Consumes rows produced by src/load.py"]},
    {"id": "s3", "action": "run_command", "target": "pytest", "why": "verify"}
  ],
  "verify_step": "s3",
  "rationale": "load feeds process",
  "rpg": {
    "capabilities": [{"id": "c1", "name": "Core", "parent": null}],
    "files": [
      {"id": "f1", "path": "src/load.py", "capability": "c1",
       "functions": [{"name": "load", "signature": "def load(p: str) -> list", "summary": "read"}]}
    ],
    "edges": [{"from": "f1", "to": "f2", "kind": "data_flow", "label": "rows"}],
    "verify": "pytest",
    "rationale": "load feeds process"
  }
}`

func TestPlanParsesRPGAndStepConstraints(t *testing.T) {
	var plan Plan
	if err := json.Unmarshal([]byte(rpgPlanJSON), &plan); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if plan.RPG == nil {
		t.Fatal("expected RPG to be parsed, got nil")
	}
	if got, want := len(plan.RPG.Files), 1; got != want {
		t.Errorf("RPG files = %d, want %d", got, want)
	}
	if got := plan.RPG.Files[0].Functions[0].Signature; got != "def load(p: str) -> list" {
		t.Errorf("function signature = %q", got)
	}
	if got, want := len(plan.RPG.Edges), 1; got != want {
		t.Errorf("RPG edges = %d, want %d", got, want)
	}
	if plan.Steps[0].NodeID != "f1" {
		t.Errorf("step 0 node_id = %q, want f1", plan.Steps[0].NodeID)
	}
	if len(plan.Steps[0].Constraints) != 2 {
		t.Errorf("step 0 constraints = %v, want 2", plan.Steps[0].Constraints)
	}
}

func TestPlanConstraintsForTarget(t *testing.T) {
	var plan Plan
	if err := json.Unmarshal([]byte(rpgPlanJSON), &plan); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ctx := &AgentContext{Plan: &plan}

	// Absolute tool path should still match the relative plan target via
	// path-suffix overlap.
	got := planConstraintsForTarget(ctx, "/workspace/src/process.py")
	want := []string{"Consumes rows produced by src/load.py"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("process.py constraints = %v, want %v", got, want)
	}

	load := planConstraintsForTarget(ctx, "src/load.py")
	if len(load) != 2 || !strings.Contains(load[0], "def load") {
		t.Errorf("load.py constraints = %v", load)
	}

	// A file with no matching plan step → nil.
	if got := planConstraintsForTarget(ctx, "src/unrelated.py"); got != nil {
		t.Errorf("unrelated constraints = %v, want nil", got)
	}

	// Returned slice is a copy — mutating it must not affect the plan.
	load[0] = "MUTATED"
	if strings.Contains(plan.Steps[0].Constraints[0], "MUTATED") {
		t.Error("planConstraintsForTarget returned a non-copy; plan was mutated")
	}
}

func TestPlanConstraintsForTargetNilPlan(t *testing.T) {
	if got := planConstraintsForTarget(&AgentContext{}, "x.py"); got != nil {
		t.Errorf("nil plan should yield nil, got %v", got)
	}
	if got := planConstraintsForTarget(nil, "x.py"); got != nil {
		t.Errorf("nil ctx should yield nil, got %v", got)
	}
}

func TestRPGFileIDForPath(t *testing.T) {
	var plan Plan
	if err := json.Unmarshal([]byte(rpgPlanJSON), &plan); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := rpgFileIDForPath(&plan, "/workspace/src/load.py"); got != "f1" {
		t.Errorf("rpgFileIDForPath = %q, want f1", got)
	}
	if got := rpgFileIDForPath(&plan, "src/nope.py"); got != "" {
		t.Errorf("unmatched path should yield \"\", got %q", got)
	}
	if got := rpgFileIDForPath(nil, "x"); got != "" {
		t.Errorf("nil plan should yield \"\", got %q", got)
	}
}

func TestAffectedDownstream(t *testing.T) {
	// f1 -> f2 -> f3 chain plus an unrelated f4.
	plan := &Plan{RPG: &RPG{
		Files: []RPGFile{
			{ID: "f1", Path: "a.py"}, {ID: "f2", Path: "b.py"},
			{ID: "f3", Path: "c.py"}, {ID: "f4", Path: "d.py"},
		},
		Edges: []RPGEdge{
			{From: "f1", To: "f2"}, {From: "f2", To: "f3"},
		},
	}}
	got := affectedDownstream(plan, "f1")
	want := []string{"b.py", "c.py"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("downstream of f1 = %v, want %v", got, want)
	}
	// Leaf node has no downstream.
	if got := affectedDownstream(plan, "f3"); got != nil {
		t.Errorf("downstream of leaf f3 = %v, want nil", got)
	}
	// Unrelated node, no edges out.
	if got := affectedDownstream(plan, "f4"); got != nil {
		t.Errorf("downstream of f4 = %v, want nil", got)
	}
}

func TestAffectedDownstreamHandlesCycleSafely(t *testing.T) {
	// A malformed cyclic graph must not loop forever.
	plan := &Plan{RPG: &RPG{
		Files: []RPGFile{{ID: "f1", Path: "a.py"}, {ID: "f2", Path: "b.py"}},
		Edges: []RPGEdge{{From: "f1", To: "f2"}, {From: "f2", To: "f1"}},
	}}
	got := affectedDownstream(plan, "f1")
	if !reflect.DeepEqual(got, []string{"b.py"}) {
		t.Errorf("cyclic downstream of f1 = %v, want [b.py]", got)
	}
}

func TestReportRPGDriftNoCrashAndEmitsStream(t *testing.T) {
	var plan Plan
	if err := json.Unmarshal([]byte(rpgPlanJSON), &plan); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var events []string
	ctx := &AgentContext{Plan: &plan, StreamFn: func(ev string, _ interface{}) {
		events = append(events, ev)
	}}
	// f1 (src/load.py) drifted; f2 is downstream and should be named.
	reportRPGDrift(ctx, "/workspace/src/load.py", []string{"def load(p: str) -> list"})
	if len(events) != 1 || events[0] != "rpg_drift" {
		t.Errorf("expected one rpg_drift stream event, got %v", events)
	}
	// Empty missing → no event.
	events = nil
	reportRPGDrift(ctx, "src/load.py", nil)
	if len(events) != 0 {
		t.Errorf("empty missing should emit nothing, got %v", events)
	}
}

// generateServer streams a fixed /v3/generate SSE result and records the
// constraints the proxy sent (so a test can assert the corrective constraint
// was injected on the retry).
func generateServer(t *testing.T, resultJSON string, gotConstraints *[]string) *httptest.Server {
	t.Helper()
	var mu sync.Mutex
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v3/generate" {
			http.NotFound(w, r)
			return
		}
		var body V3GenerateRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if gotConstraints != nil {
			mu.Lock()
			*gotConstraints = body.Constraints
			mu.Unlock()
		}
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "event: result\ndata: "+resultJSON+"\n\ndata: [DONE]\n\n")
		f.Flush()
	}))
}

func TestRegenerateOnDriftRetriesAndKeepsCleanResult(t *testing.T) {
	// regenerateOnDrift makes exactly one (retry) call; have it come back clean.
	var sentConstraints []string
	srv := generateServer(t,
		`{"code": "def load(p):\n    return []", "passed": true, "phase_solved": "phase1"}`,
		&sentConstraints)
	defer srv.Close()

	ctx := &AgentContext{V3URL: srv.URL}
	req := V3GenerateRequest{
		FilePath:    "load.py",
		Constraints: []string{"Implement `def load(p)`"},
	}
	prev := &V3GenerateResponse{Code: "def other(): pass", RPGSignatureMissing: []string{"def load(p)"}}

	got := regenerateOnDrift(ctx, req, prev)
	if len(got.RPGSignatureMissing) != 0 {
		t.Errorf("expected clean retry result, still missing %v", got.RPGSignatureMissing)
	}
	if !strings.Contains(got.Code, "def load") {
		t.Errorf("expected retried code to define load, got %q", got.Code)
	}
	// The retry must inject the missing signature as a corrective constraint.
	joined := strings.Join(sentConstraints, " | ")
	if !strings.Contains(joined, "REQUIRED") || !strings.Contains(joined, "def load(p)") {
		t.Errorf("retry constraints missing corrective directive: %v", sentConstraints)
	}
}

func TestRegenerateOnDriftNoopWithoutDriftOrConstraints(t *testing.T) {
	ctx := &AgentContext{V3URL: "http://127.0.0.1:0"} // never called
	clean := &V3GenerateResponse{Code: "ok"}
	// No drift → returned unchanged, server never hit.
	if got := regenerateOnDrift(ctx, V3GenerateRequest{Constraints: []string{"x"}}, clean); got != clean {
		t.Error("clean result should be returned unchanged")
	}
	// Drift but no RPG constraints (flat planner) → no-op.
	drift := &V3GenerateResponse{RPGSignatureMissing: []string{"def f()"}}
	if got := regenerateOnDrift(ctx, V3GenerateRequest{}, drift); got != drift {
		t.Error("no-constraints drift should be returned unchanged")
	}
}

func TestRegenerateOnDriftKeepsPrevWhenRetryNoBetter(t *testing.T) {
	// Server always drifts → retry is no better → keep prev.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		fmt.Fprint(w, `event: result`+"\n"+`data: {"code":"def other(): pass","rpg_signature_missing":["def load(p)"]}`+"\n\ndata: [DONE]\n\n")
		f.Flush()
	}))
	defer srv.Close()
	ctx := &AgentContext{V3URL: srv.URL}
	req := V3GenerateRequest{FilePath: "load.py", Constraints: []string{"Implement `def load(p)`"}}
	prev := &V3GenerateResponse{Code: "prev", RPGSignatureMissing: []string{"def load(p)"}}
	if got := regenerateOnDrift(ctx, req, prev); got != prev {
		t.Errorf("retry no better should keep prev, got %+v", got)
	}
}

func TestRunCommandStepNotFileProducing(t *testing.T) {
	// A run_command step's args must never be treated as constraints even if
	// its target happened to overlap a path-shaped string.
	if isFileProducingAction("run_command") {
		t.Error("run_command must not be a file-producing action")
	}
	for _, a := range []string{"write_file", "ast_edit", "edit_file"} {
		if !isFileProducingAction(a) {
			t.Errorf("%s should be file-producing", a)
		}
	}
}
