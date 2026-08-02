package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func lensHealthServer(t *testing.T, lens map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":     "healthy",
			"subsystems": map[string]any{"lens": lens},
		})
	}))
}

func compatibleLensHealth() map[string]any {
	return map[string]any{
		"enabled":           true,
		"cost_field_loaded": true,
		"cost_field_dim":    3840,
		"embed_dim":         3840,
		"gx_loaded":         true,
		"cx_calibrated":     true,
		"gx_calibrated":     true,
		"self_test_pass":    true,
	}
}

func TestProbeLensStatusRequiresSelectedModelCalibration(t *testing.T) {
	health := compatibleLensHealth()
	health["cx_calibrated"] = false
	srv := lensHealthServer(t, health)
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Verdict != "uncalibrated" {
		t.Fatalf("verdict = %q, want uncalibrated", got.Verdict)
	}
}

func TestProbeLensStatusSurfacesArtifactIdentityMismatch(t *testing.T) {
	health := compatibleLensHealth()
	health["cost_field_loaded"] = false
	health["self_test_error"] = "artifacts are for model-a, selected model is model-b"
	srv := lensHealthServer(t, health)
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Hint != health["self_test_error"] {
		t.Fatalf("hint = %q, want identity mismatch", got.Hint)
	}
}

func TestProbeLensStatusRequiresCompleteArtifacts(t *testing.T) {
	health := compatibleLensHealth()
	health["gx_loaded"] = false
	srv := lensHealthServer(t, health)
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Verdict != "incomplete-artifacts" {
		t.Fatalf("verdict = %q, want incomplete-artifacts", got.Verdict)
	}
}

func TestProbeLensStatusSupportsCalibratedMatchingArtifacts(t *testing.T) {
	srv := lensHealthServer(t, compatibleLensHealth())
	defer srv.Close()

	got := probeLensStatus(context.Background(), srv.URL)
	if got.Verdict != "supported" {
		t.Fatalf("verdict = %q, want supported (%s)", got.Verdict, got.Hint)
	}
}

func TestProbeASAStatusRequiresMatchingModelMarker(t *testing.T) {
	dir := t.TempDir()
	vector := filepath.Join(dir, "ast_edit_steering.gguf")
	if err := os.WriteFile(vector, []byte("vector"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ATLAS_CONTROL_VECTOR", vector)
	t.Setenv("ATLAS_MODEL_NAME", "selected-model")

	if got := probeASAStatus(); got.Verdict != "unverified" {
		t.Fatalf("without marker verdict = %q, want unverified", got.Verdict)
	}
	if err := os.WriteFile(vector+".model", []byte("other-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := probeASAStatus(); got.Verdict != "incompatible" {
		t.Fatalf("wrong marker verdict = %q, want incompatible", got.Verdict)
	}
	if err := os.WriteFile(vector+".model", []byte("selected-model\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := probeASAStatus(); got.Verdict != "supported" {
		t.Fatalf("matching marker verdict = %q, want supported", got.Verdict)
	}
}

// End-to-end: a completed pass is stashed, then /feedback (thumbs-up with one
// denied file) turns its writes into the expected weighted samples.
func TestHandleFeedbackEndToEnd(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATLAS_LENS_DATA_DIR", dir)
	model := modelName

	stashPendingPass("sess-1", model, []PassWrite{
		{Tool: "write_file", Path: "Dockerfile", Content: "FROM python:3.11\n"},
		{Tool: "write_file", Path: "stub.py", Content: "def f():\n    pass\n"},
	})

	body, _ := json.Marshal(map[string]interface{}{
		"session_id": "sess-1",
		"thumbs":     "up",
		"files":      []map[string]string{{"path": "stub.py", "verdict": "deny"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/feedback", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	handleFeedback(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	var resp struct{ Recorded, Good, Bad int }
	json.Unmarshal(rr.Body.Bytes(), &resp)
	if resp.Recorded != 2 {
		t.Errorf("recorded = %d, want 2", resp.Recorded)
	}
	// Dockerfile accepted in a thumbs-up pass → good; stub.py denied → bad.
	if resp.Good != 1 || resp.Bad != 1 {
		t.Errorf("good/bad = %d/%d, want 1/1", resp.Good, resp.Bad)
	}
	// Pending entry must be consumed (rating a pass twice shouldn't double-count).
	if _, ok := takePendingPass("sess-1"); ok {
		t.Errorf("pending pass should have been consumed by /feedback")
	}
}

func TestHandleFeedbackUnknownSession(t *testing.T) {
	t.Setenv("ATLAS_LENS_DATA_DIR", t.TempDir())
	body, _ := json.Marshal(map[string]string{"session_id": "nope", "thumbs": "up"})
	req := httptest.NewRequest(http.MethodPost, "/feedback", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	handleFeedback(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	var resp struct{ Recorded int }
	json.Unmarshal(rr.Body.Bytes(), &resp)
	if resp.Recorded != 0 {
		t.Errorf("recorded = %d for unknown session, want 0", resp.Recorded)
	}
}

func TestFeedbackVerdictMatrix(t *testing.T) {
	cases := []struct {
		verdict, thumbs string
		label           int
		weight          float64
		keep            bool
	}{
		// Denials are confident negatives regardless of pass verdict.
		{"deny", "up", 0, 1.0, true},
		{"deny", "down", 0, 1.0, true},
		{"deny", "", 0, 1.0, true},
		// Accepted files: weight modulated by the pass thumbs.
		{"accept", "up", 1, 1.0, true},   // good result, accepted → confident positive
		{"accept", "down", 1, 0.4, true}, // whole pass wrong → weak positive
		{"accept", "", 1, 0.7, true},     // accepted, unrated → moderate
		// Thumbs-only (no per-file verdict): pass thumbs labels everything coarsely.
		{"", "up", 1, 0.6, true},
		{"", "down", 0, 0.6, true},
		{"", "", 0, 0, false}, // no signal → don't record
	}
	for _, c := range cases {
		label, weight, keep := feedbackVerdict(c.verdict, c.thumbs)
		if label != c.label || weight != c.weight || keep != c.keep {
			t.Errorf("feedbackVerdict(%q,%q) = (%d,%.2f,%v), want (%d,%.2f,%v)",
				c.verdict, c.thumbs, label, weight, keep, c.label, c.weight, c.keep)
		}
	}
}

// The case the whole design hinges on: a thumbs-up pass with one denied file
// yields the cleanest data — accepted files are full-weight positives, the
// denied one a full-weight negative.
func TestFeedbackGoodPassOneBadFile(t *testing.T) {
	gLabel, gW, _ := feedbackVerdict("accept", "up")
	bLabel, bW, _ := feedbackVerdict("deny", "up")
	if !(gLabel == 1 && gW == 1.0) {
		t.Errorf("accepted file in good pass should be confident positive, got label=%d w=%.2f", gLabel, gW)
	}
	if !(bLabel == 0 && bW == 1.0) {
		t.Errorf("denied file should be confident negative, got label=%d w=%.2f", bLabel, bW)
	}
	// And a thumbs-down pass down-weights its accepted files vs a thumbs-up one.
	_, downW, _ := feedbackVerdict("accept", "down")
	if !(downW < gW) {
		t.Errorf("accepted file in a thumbs-down pass (w=%.2f) must weigh less than in a thumbs-up pass (w=%.2f)", downW, gW)
	}
}

func TestAppendAndCountLensSamples(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATLAS_LENS_DATA_DIR", dir)
	model := "gemma-4-12b-it-Q4_K_M"
	for _, s := range []LensSample{
		{Content: "FROM python:3.11\n", Label: 1, Weight: 1.0, Source: "accept"},
		{Content: "FROM base\nCMD run\n", Label: 0, Weight: 1.0, Source: "deny"},
		{Content: "def f(): return 1\n", Label: 1, Weight: 0.4, Source: "accept"},
	} {
		if err := appendLensSample(model, s); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	good, bad := lensSampleCounts(model)
	if good != 2 || bad != 1 {
		t.Errorf("counts = (good=%d, bad=%d), want (2, 1)", good, bad)
	}
}

func TestSanitizeModelName(t *testing.T) {
	if got := sanitizeModelName("vendor/Model:Q6_K"); got != "vendor_Model_Q6_K" {
		t.Errorf("sanitize = %q", got)
	}
	if got := sanitizeModelName(""); got != "default" {
		t.Errorf("empty sanitize = %q, want default", got)
	}
}

// Threshold resolution is fail-closed: only the selected model's calibrated
// values may drive interventions.
func TestLensThresholdResolution(t *testing.T) {
	var bare lensPerStepResult
	if _, _, ok := bare.calibratedThresholds(); ok {
		t.Fatal("missing calibration must not produce intervention thresholds")
	}
	withT := lensPerStepResult{Thresholds: &lensThresholds{OffRails: 0.6, Low: 0.45, Severe: 0.3}}
	low, severe, ok := withT.calibratedThresholds()
	if !ok || low != 0.45 || severe != 0.3 {
		t.Fatalf("calibrated thresholds = (%v, %v, %v)", low, severe, ok)
	}
	invalid := lensPerStepResult{Thresholds: &lensThresholds{Low: 0.2, Severe: 0.3}}
	if _, _, ok := invalid.calibratedThresholds(); ok {
		t.Fatal("severe > low must be rejected")
	}
}

// The same scores are interpreted only against the selected model's own
// calibration.
func TestAgentLensRegressionUsesPerModelThresholds(t *testing.T) {
	scores := []float64{0.40, 0.39}
	if _, fired := agentLensRegression(scores, 0.45, 0.30); !fired {
		t.Errorf("model-calibrated thresholds should fire on a 0.40/0.39 run")
	}
}

// Severe single-write short-circuit also honors the per-model severe value.
func TestAgentLensRegressionSevereIsPerModel(t *testing.T) {
	// One write at 0.32. Below a model severe of 0.35 → immediate fire.
	if _, fired := agentLensRegression([]float64{0.32}, 0.45, 0.35); !fired {
		t.Errorf("single write below per-model severe should fire")
	}
	// Same score under another valid calibration does not fire immediately.
	if _, fired := agentLensRegression([]float64{0.32}, 0.2, 0.1); fired {
		t.Errorf("single 0.32 write should not fire above calibrated severe=0.1")
	}
}

func dimByName(dims []StatusDimension, name string) StatusDimension {
	for _, d := range dims {
		if d.Name == name {
			return d
		}
	}
	return StatusDimension{}
}

func TestBuildDimensionsSevenRows(t *testing.T) {
	dims := buildDimensions(LensStatus{}, ASAStatus{Verdict: "missing"})
	want := []string{"model_runtime", "direct_agent", "lens_identity",
		"lens_scoring", "lens_calibration", "lens_intervention", "asa"}
	if len(dims) != len(want) {
		t.Fatalf("expected %d dimensions, got %d", len(want), len(dims))
	}
	for i, n := range want {
		if dims[i].Name != n {
			t.Errorf("dimension %d = %q, want %q", i, dims[i].Name, n)
		}
	}
}

func TestDirectAgentAlwaysSupported(t *testing.T) {
	// Even with a fully disabled lens, the direct agent is model-agnostic.
	dims := buildDimensions(LensStatus{Verdict: "unreachable"},
		ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "direct_agent"); d.Status != "supported" {
		t.Fatalf("direct_agent should always be supported, got %q", d.Status)
	}
}

func TestInterventionNeutralWhenUncalibrated(t *testing.T) {
	// Loaded + scoring available but NOT calibrated → intervention must
	// be "neutral" (never "active"), matching the runtime guarantee.
	lens := LensStatus{
		Verdict: "uncalibrated", CostFieldLoaded: true, GxLoaded: true,
		CostFieldDim: 3840, EmbedDim: 3840,
		CxCalibrated: false, GxCalibrated: false,
	}
	dims := buildDimensions(lens, ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "lens_calibration"); d.Status != "uncalibrated" {
		t.Errorf("calibration = %q, want uncalibrated", d.Status)
	}
	if d := dimByName(dims, "lens_intervention"); d.Status != "neutral" {
		t.Fatalf("intervention = %q, want neutral when uncalibrated", d.Status)
	}
}

func TestInterventionActiveOnlyWhenCalibrated(t *testing.T) {
	lens := LensStatus{
		Verdict: "supported", CostFieldLoaded: true, GxLoaded: true,
		CostFieldDim: 3840, EmbedDim: 3840,
		CxCalibrated: true, GxCalibrated: true,
	}
	dims := buildDimensions(lens, ASAStatus{Verdict: "supported"})
	if d := dimByName(dims, "lens_intervention"); d.Status != "active" {
		t.Fatalf("intervention = %q, want active when calibrated", d.Status)
	}
}

func TestInterventionDisabledWhenNoArtifacts(t *testing.T) {
	dims := buildDimensions(LensStatus{Verdict: "no-artifacts"},
		ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "lens_intervention"); d.Status != "disabled" {
		t.Fatalf("intervention = %q, want disabled with no artifacts", d.Status)
	}
}

func TestDimMismatchSurfaced(t *testing.T) {
	lens := LensStatus{
		Verdict: "dim-mismatch", CostFieldLoaded: true,
		CostFieldDim: 4096, EmbedDim: 3840,
	}
	dims := buildDimensions(lens, ASAStatus{Verdict: "missing"})
	if d := dimByName(dims, "lens_identity"); d.Status != "dim-mismatch" {
		t.Fatalf("identity = %q, want dim-mismatch", d.Status)
	}
}

// The lens corpus had exactly one writer: POST /feedback, a human thumbs or
// per-file verdict. LensSample.Source advertised "v3" and "run" alongside and
// nothing ever wrote either, so twelve instrumented runs — which produced
// dozens of deterministic gate rejections and several passing verification
// commands — recorded nothing at all. The directory was empty.
func TestGateRejectionsAndVerifiedRunsReachTheCorpus(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATLAS_LENS_DATA_DIR", dir)
	const model = "test-model"

	recordGateRejection(model, "replace_lines", "app.py",
		"setTimeout(draw, delay);", "stops a render loop")
	if n := recordVerifiedPass(model, []PassWrite{
		{Tool: "edit_file", Path: "app.py", Content: "let delay = 100;"},
		{Tool: "edit_file", Path: "b.py", Content: ""}, // nothing authored
	}); n != 1 {
		t.Fatalf("expected 1 verified sample recorded, got %d", n)
	}

	good, bad := lensSampleCounts(model)
	if bad != 1 {
		t.Errorf("gate rejection not recorded as a negative: bad=%d", bad)
	}
	if good != 1 {
		t.Errorf("verified run not recorded as a positive: good=%d", good)
	}
}

// A gate is deterministic; "curl exited clean" is not. An observed run had
// curl return 200 over a page whose game loop was dead, and another over a
// Flask app with no routes left in it.
func TestAGateNegativeOutweighsAVerifiedPositive(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ATLAS_LENS_DATA_DIR", dir)
	const model = "weights"

	recordGateRejection(model, "edit_file", "a.py", "broken", "syntax error")
	recordVerifiedPass(model, []PassWrite{{Tool: "edit_file", Path: "a.py", Content: "ok"}})

	raw, err := os.ReadFile(filepath.Join(dir, model, "samples.jsonl"))
	if err != nil {
		t.Fatalf("corpus not written: %v", err)
	}
	var negWeight, posWeight float64
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var s LensSample
		if json.Unmarshal([]byte(line), &s) != nil {
			continue
		}
		if s.Label == 0 {
			negWeight = s.Weight
			if s.Source != "gate" {
				t.Errorf("gate negative recorded with source %q", s.Source)
			}
		} else {
			posWeight = s.Weight
			if s.Source != "run" {
				t.Errorf("verified positive recorded with source %q", s.Source)
			}
		}
	}
	if !(negWeight > posWeight) {
		t.Errorf("a deterministic gate rejection (%.2f) must outweigh a passing probe (%.2f)",
			negWeight, posWeight)
	}
}

// Content the model never authored must not enter the corpus — an input
// validation error says nothing about generation quality.
func TestOnlyAuthoredContentIsSampled(t *testing.T) {
	for _, tc := range []struct{ name, args, want string }{
		{"write content", `{"path":"a.py","content":"x = 1"}`, "x = 1"},
		{"edit new_str", `{"path":"a.py","old_str":"a","new_str":"b"}`, "b"},
		{"read has none", `{"path":"a.py"}`, ""},
		{"command has none", `{"command":"pytest"}`, ""},
		{"malformed", `{`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := authoredContent(json.RawMessage(tc.args)); got != tc.want {
				t.Errorf("authoredContent = %q, want %q", got, tc.want)
			}
		})
	}
}
