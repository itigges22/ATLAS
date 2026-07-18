package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

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
