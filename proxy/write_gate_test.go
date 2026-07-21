package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// v3AndStructuralServer stands in for the v3-service: it serves both
// /v3/generate (SSE, returns `winnerCode` as the pipeline result) and
// /internal/structural_check (flags `flagName` in any source that calls it
// without importing it). One server because both live behind ctx.V3URL.
func v3AndStructuralServer(t *testing.T, winnerCode, flagName string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v3/generate":
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			payload, _ := json.Marshal(map[string]interface{}{
				"code": winnerCode, "passed": true,
				"phase_solved": "phase1", "candidates_tested": 3,
				"winning_score": 0.9,
			})
			for _, line := range []string{"event: result", "data: " + string(payload), "", "data: [DONE]", ""} {
				fmt.Fprint(w, line+"\n")
				if fl != nil {
					fl.Flush()
				}
			}
		case "/internal/structural_check":
			raw, _ := io.ReadAll(r.Body)
			var body struct {
				Source string `json:"source"`
			}
			_ = json.Unmarshal(raw, &body)
			out := map[string]interface{}{"ok": true, "unresolved": []string{}}
			if strings.Contains(body.Source, flagName+"(") &&
				!strings.Contains(body.Source, "import "+flagName) {
				out["unresolved"] = []string{flagName}
			}
			b, _ := json.Marshal(out)
			_, _ = w.Write(b)
		default:
			http.NotFound(w, r)
		}
	}))
}

// fakeSyntaxSandbox serves /syntax-check the way checkFallbackSyntax expects:
// valid unless the source contains `brokenMarker`.
func fakeSyntaxSandbox(t *testing.T, brokenMarker string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/syntax-check" {
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body struct {
			Code string `json:"code"`
		}
		_ = json.Unmarshal(raw, &body)
		valid := brokenMarker == "" || !strings.Contains(body.Code, brokenMarker)
		out := map[string]interface{}{"valid": valid}
		if !valid {
			out["errors"] = []string{"SyntaxError: invalid syntax"}
		}
		b, _ := json.Marshal(out)
		_, _ = w.Write(b)
	}))
}

func writeGateCtx(t *testing.T, v3URL, sandboxURL, workDir string) *AgentContext {
	t.Helper()
	return &AgentContext{
		V3URL:         v3URL,
		SandboxURL:    sandboxURL,
		WorkingDir:    workDir,
		Tier:          Tier2Medium,
		Ctx:           context.Background(),
		SessionWrites: map[string]bool{},
	}
}

// When the V3 winner introduces an unresolved call but the model's own
// baseline is clean, writeFileWithV3 must write the BASELINE and report it
// as a plain (non-V3) write — no V3Used/PhaseSolved/score attached to
// content that never went through V3 sandbox verification. Guards against
// PC-044 telling the model an unverified baseline was "V3 verified".
func TestWriteFileV3WinnerVetoFallsBackToBaseline(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "app.py")
	baseline := "def index():\n    return 'ok'\n"
	winner := "def index():\n    return render_template('index.html')\n"

	v3 := v3AndStructuralServer(t, winner, "render_template")
	defer v3.Close()
	sb := fakeSyntaxSandbox(t, "") // baseline always parses
	defer sb.Close()
	ctx := writeGateCtx(t, v3.URL, sb.URL, dir)

	res, err := writeFileWithV3(path, baseline, ctx)
	if err != nil {
		t.Fatalf("writeFileWithV3 error: %v", err)
	}
	if res == nil || !res.Success {
		t.Fatalf("expected success, got %+v", res)
	}
	// Baseline landed, not the vetoed winner.
	onDisk, _ := os.ReadFile(path)
	if string(onDisk) != baseline {
		t.Errorf("expected baseline on disk, got %q", string(onDisk))
	}
	// Telemetry must NOT claim V3 verification of the baseline.
	if res.V3Used || res.PhaseSolved != "" || res.WinningScore != 0 || res.VerificationEvidence != nil {
		t.Errorf("fallback write must carry no V3 metadata, got V3Used=%v phase=%q score=%v evidence=%v",
			res.V3Used, res.PhaseSolved, res.WinningScore, res.VerificationEvidence)
	}
}

// A clean V3 winner lands with full V3 telemetry (the non-fallback path).
func TestWriteFileV3CleanWinnerKeepsTelemetry(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "app.py")
	baseline := "def index():\n    return 'ok'\n"
	winner := "def index():\n    return 'better'\n" // no unresolved call

	v3 := v3AndStructuralServer(t, winner, "render_template")
	defer v3.Close()
	sb := fakeSyntaxSandbox(t, "")
	defer sb.Close()
	ctx := writeGateCtx(t, v3.URL, sb.URL, dir)

	res, err := writeFileWithV3(path, baseline, ctx)
	if err != nil {
		t.Fatalf("writeFileWithV3 error: %v", err)
	}
	if res == nil || !res.Success {
		t.Fatalf("expected success, got %+v", res)
	}
	onDisk, _ := os.ReadFile(path)
	if string(onDisk) != winner {
		t.Errorf("expected winner on disk, got %q", string(onDisk))
	}
	if !res.V3Used || res.PhaseSolved != "phase1" {
		t.Errorf("clean winner must keep V3 telemetry, got V3Used=%v phase=%q", res.V3Used, res.PhaseSolved)
	}
}

// Both winner AND baseline introduce an unresolved call → reject (the
// model's own content is genuinely broken, name what it can act on).
func TestWriteFileV3BothBrokenRejects(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "app.py")
	baseline := "def index():\n    return render_template('x')\n" // model's own is broken too
	winner := "def index():\n    return render_template('index.html')\n"

	v3 := v3AndStructuralServer(t, winner, "render_template")
	defer v3.Close()
	sb := fakeSyntaxSandbox(t, "")
	defer sb.Close()
	ctx := writeGateCtx(t, v3.URL, sb.URL, dir)

	res, err := writeFileWithV3(path, baseline, ctx)
	if err != nil {
		t.Fatalf("writeFileWithV3 error: %v", err)
	}
	if res == nil || res.Success {
		t.Fatalf("expected rejection, got %+v", res)
	}
	if _, statErr := os.Stat(path); statErr == nil {
		t.Error("nothing should have landed on disk")
	}
	if !strings.Contains(res.Error, "render_template") || !strings.Contains(res.Error, "write_file") {
		t.Errorf("rejection should name the call and be write-flavored: %q", res.Error)
	}
}

// A user cancel while the winner-gate structural check is in flight must
// land nothing on disk.
func TestWriteFileV3CancelDuringGateWritesNothing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "app.py")
	baseline := "def index():\n    return 'ok'\n"
	winner := "def index():\n    return 'better'\n"

	reqCtx, cancel := context.WithCancel(context.Background())
	// Structural server cancels the request context on the first
	// /internal/structural_check call, simulating a mid-gate Ctrl+C.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v3/generate":
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			payload, _ := json.Marshal(map[string]interface{}{
				"code": winner, "passed": true, "phase_solved": "phase1", "candidates_tested": 1,
			})
			for _, line := range []string{"event: result", "data: " + string(payload), "", "data: [DONE]", ""} {
				fmt.Fprint(w, line+"\n")
				if fl != nil {
					fl.Flush()
				}
			}
		case "/internal/structural_check":
			cancel() // user hit Ctrl+C during the gate
			_, _ = w.Write([]byte(`{"ok":true,"unresolved":[]}`))
		}
	}))
	defer srv.Close()
	ctx := writeGateCtx(t, srv.URL, "", dir)
	ctx.Ctx = reqCtx

	res, err := writeFileWithV3(path, baseline, ctx)
	if err != nil {
		t.Fatalf("writeFileWithV3 error: %v", err)
	}
	if res == nil || res.Success {
		t.Fatalf("cancelled write must not succeed, got %+v", res)
	}
	if _, statErr := os.Stat(path); statErr == nil {
		t.Error("cancelled write must not land content on disk")
	}
}
