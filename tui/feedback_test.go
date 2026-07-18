// Tests for the lens-feedback client (/feedback, /v1/lens/training-status) and
// the /good /bad slash commands that drive lens-training data collection.
package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSubmitFeedbackPostsThumbs(t *testing.T) {
	var gotPath, gotThumbs, gotSession string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		var body struct {
			SessionID string `json:"session_id"`
			Thumbs    string `json:"thumbs"`
		}
		_ = json.Unmarshal(b, &body)
		gotThumbs, gotSession = body.Thumbs, body.SessionID
		w.Write([]byte(`{"recorded":3,"good":2,"bad":1}`))
	}))
	defer srv.Close()

	n, err := submitFeedback(srv.URL, "sess-9", "down", nil)
	if err != nil {
		t.Fatalf("submitFeedback: %v", err)
	}
	if n != 3 {
		t.Errorf("recorded = %d, want 3", n)
	}
	if gotPath != "/feedback" || gotThumbs != "down" || gotSession != "sess-9" {
		t.Errorf("server saw path=%q thumbs=%q session=%q", gotPath, gotThumbs, gotSession)
	}
}

func TestSubmitFeedbackNoSession(t *testing.T) {
	if _, err := submitFeedback("http://localhost:9", "", "up", nil); err == nil {
		t.Errorf("expected error for empty session id")
	}
}

func TestFetchTrainingStatusParses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/lens/training-status" {
			t.Errorf("path = %q", r.URL.Path)
		}
		w.Write([]byte(`{"total":2100,"good":1200,"bad":900,"threshold":2000,"retrain_available":true,"command":"atlas lens retrain"}`))
	}))
	defer srv.Close()

	ts, err := fetchTrainingStatus(srv.URL)
	if err != nil {
		t.Fatalf("fetchTrainingStatus: %v", err)
	}
	if !ts.RetrainAvailable || ts.Total != 2100 || ts.Command != "atlas lens retrain" {
		t.Errorf("parsed = %+v", ts)
	}
}

func TestSlashGoodDispatchesFeedback(t *testing.T) {
	var gotThumbs string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body struct{ Thumbs string }
		_ = json.Unmarshal(b, &body)
		gotThumbs = body.Thumbs
		w.Write([]byte(`{"recorded":2}`))
	}))
	defer srv.Close()

	m := newTUIModel(srv.URL)
	m.lastPassSession = "sess-1"
	consumed, cmd, _ := m.handleSlash("/good")
	if !consumed || cmd == nil {
		t.Fatalf("consumed=%v cmd=%v, want true + non-nil", consumed, cmd)
	}
	msg := cmd()
	res, ok := msg.(slashResultMsg)
	if !ok {
		t.Fatalf("msg type = %T, want slashResultMsg", msg)
	}
	if gotThumbs != "up" {
		t.Errorf("thumbs = %q, want up", gotThumbs)
	}
	if res.err != nil || !strings.Contains(res.output, "banked") {
		t.Errorf("result = %+v", res)
	}
}

// /deny marks one file bad; the following /good submits a thumbs-up pass with
// that file as a per-file deny (the "good pass, one bad file" case).
func TestSlashDenyThenGoodSendsPerFileVerdict(t *testing.T) {
	var body struct {
		Thumbs string        `json:"thumbs"`
		Files  []fileVerdict `json:"files"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		w.Write([]byte(`{"recorded":2}`))
	}))
	defer srv.Close()

	m := newTUIModel(srv.URL)
	m.lastPassSession = "sess-3"
	m.lastPassFiles = []string{"app.py", "stub.py"}

	if _, cmd, _ := m.handleSlash("/deny stub.py wrong approach"); cmd != nil {
		t.Errorf("/deny should not return a cmd (records locally)")
	}
	if m.passVerdicts["stub.py"] != "deny" {
		t.Fatalf("/deny didn't record the verdict: %+v", m.passVerdicts)
	}
	if m.passReasons["stub.py"] != "wrong approach" {
		t.Errorf("/deny didn't keep the reason: %+v", m.passReasons)
	}

	_, cmd, _ := m.handleSlash("/good")
	if cmd == nil {
		t.Fatal("/good returned no cmd")
	}
	res := cmd()
	if body.Thumbs != "up" {
		t.Errorf("thumbs = %q, want up", body.Thumbs)
	}
	if len(body.Files) != 1 || body.Files[0].Path != "stub.py" || body.Files[0].Verdict != "deny" {
		t.Errorf("files = %+v, want one deny on stub.py", body.Files)
	}
	// Verdicts stay staged until the submit result lands, then clear on
	// success (a failed POST keeps them for a retry).
	if len(m.passVerdicts) != 1 {
		t.Errorf("passVerdicts should stay staged until the result lands, got %+v", m.passVerdicts)
	}
	updated, _ := m.Update(res)
	mu := updated.(tuiModel)
	if len(mu.passVerdicts) != 0 {
		t.Errorf("passVerdicts should be cleared after a successful /good, got %+v", mu.passVerdicts)
	}
}

// A failed submit keeps the staged per-file verdicts so the user can
// retry /good without re-entering every /deny.
func TestSlashGoodKeepsVerdictsWhenSubmitFails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "lens store unavailable", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	m := newTUIModel(srv.URL)
	m.lastPassSession = "sess-7"
	m.lastPassFiles = []string{"stub.py"}
	m.handleSlash("/deny stub.py")

	_, cmd, _ := m.handleSlash("/good")
	res := cmd()
	sr, ok := res.(slashResultMsg)
	if !ok || sr.err == nil {
		t.Fatalf("result = %#v, want an error on 503", res)
	}
	updated, _ := m.Update(res)
	mu := updated.(tuiModel)
	if mu.passVerdicts["stub.py"] != "deny" {
		t.Errorf("passVerdicts = %+v, want deny kept after failed submit", mu.passVerdicts)
	}
}

func TestSlashAcceptUndoesDeny(t *testing.T) {
	m := newTUIModel("http://unused")
	m.lastPassFiles = []string{"app.py"}
	m.handleSlash("/deny app.py")
	if m.passVerdicts["app.py"] != "deny" {
		t.Fatal("deny not recorded")
	}
	m.handleSlash("/accept app.py")
	if _, ok := m.passVerdicts["app.py"]; ok {
		t.Errorf("/accept should remove the deny verdict")
	}
}

// /deny on a path the last pass never wrote warns with the rateable
// list instead of recording a verdict the proxy would drop.
func TestSlashDenyRejectsUnknownPath(t *testing.T) {
	m := newTUIModel("http://unused")
	m.lastPassFiles = []string{"app.py", "util.py"}
	m.handleSlash("/deny appp.py typo")
	if _, ok := m.passVerdicts["appp.py"]; ok {
		t.Fatal("verdict recorded for a path the pass never wrote")
	}
	last := m.chat[len(m.chat)-1]
	if last.Meta != "error" || !strings.Contains(last.Body, "app.py") {
		t.Errorf("expected a warning listing rateable files, got %+v", last)
	}

	// No pass at all → same guard with a clearer message.
	m2 := newTUIModel("http://unused")
	m2.handleSlash("/deny app.py")
	if len(m2.passVerdicts) != 0 {
		t.Errorf("verdict recorded with no completed pass: %+v", m2.passVerdicts)
	}
}

func TestSubmitFeedbackReturnsErrorOnNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	if _, err := submitFeedback(srv.URL, "sess-1", "up", nil); err == nil {
		t.Fatal("expected error on 500 response")
	}
}

func TestSlashBadSendsThumbsDown(t *testing.T) {
	var gotThumbs string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var body struct{ Thumbs string }
		_ = json.Unmarshal(b, &body)
		gotThumbs = body.Thumbs
		w.Write([]byte(`{"recorded":1}`))
	}))
	defer srv.Close()

	m := newTUIModel(srv.URL)
	m.lastPassSession = "sess-2"
	_, cmd, _ := m.handleSlash("/bad")
	cmd()
	if gotThumbs != "down" {
		t.Errorf("thumbs = %q, want down", gotThumbs)
	}
}
