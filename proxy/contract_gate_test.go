package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A program run as `prog < data` is verified as a stdin filter, which is not
// how its caller will run it.
//
// Measured on the AoC tasks, whose prompt says the program must read
// input.txt: 7 of 10 failures wrote a stdin-reading program, ran it as
// `python3 solve.py < input.txt`, and got a SUCCESSFUL result — so the model
// had every reason to believe it was finished. The checker then ran
// `python solve.py` with no redirect and got 0. Not one session that
// verified this way passed.
//
// The same model with no shell never produces this shape; it writes code that
// opens the file, because piping is not available to it. The tool is what
// makes the wrong contract reachable, so the harness has to notice.

func TestARedirectIsDetected(t *testing.T) {
	for _, cmd := range []string{
		"python3 solve.py < input.txt",
		"python solve.py <input.txt",
		"./run < data/in.txt",
	} {
		if got := stdinRedirectSource(cmd); got == "" {
			t.Errorf("no redirect detected in %q", cmd)
		}
	}
}

func TestTheRedirectedFileIsNamed(t *testing.T) {
	if got := stdinRedirectSource("python3 solve.py < input.txt"); got != "input.txt" {
		t.Errorf("want input.txt, got %q", got)
	}
}

func TestOrdinaryCommandsAreNotRedirects(t *testing.T) {
	for _, cmd := range []string{
		"python3 solve.py",
		"pytest -q",
		"go build ./...",
		"echo hi > out.txt",
		"python3 - <<'EOF'\nprint(1)\nEOF",
		"diff <(sort a) <(sort b)",
		"",
	} {
		if got := stdinRedirectSource(cmd); got != "" {
			t.Errorf("%q should not read as a stdin redirect, got %q", cmd, got)
		}
	}
}

func TestTheRejectionSaysWhatToDoInstead(t *testing.T) {
	msg := redirectOnlyVerificationMessage("input.txt")
	for _, want := range []string{"input.txt", "standalone", "no `<`", "open"} {
		if !strings.Contains(msg, want) {
			t.Errorf("rejection should mention %q: %s", want, msg)
		}
	}
}

// Widened contract detection: a redirect anywhere in the segment and the
// cat-pipe idiom are the same stdin contract the trailing-only rule caught.
func TestStdinRedirectSourceWiderShapes(t *testing.T) {
	cases := []struct {
		cmd  string
		want string
	}{
		{"python3 solve.py < input.txt > out.txt", "input.txt"},
		{"python3 solve.py <input.txt 2>err.log", "input.txt"},
		{"cat input.txt | python3 solve.py", "input.txt"},
		{"cd /w && python3 solve.py < data.txt", "data.txt"},
		// Not stdin contracts:
		{"python3 solve.py", ""},
		{"python3 solve.py <<EOF\n1 2\nEOF", ""},
		{"diff <(sort a) <(sort b)", ""},
		{"cat notes.txt || echo missing", ""},
	}
	for _, c := range cases {
		if got := stdinRedirectSource(c.cmd); got != c.want {
			t.Errorf("stdinRedirectSource(%q) = %q, want %q", c.cmd, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// The V3 evidence wire contract
// ---------------------------------------------------------------------------

// Cross-language golden fixtures. Every file under
// ../v3-service/testdata/evidence_wire is a complete /v3/generate response
// body produced by the REAL Python serialiser. These tests decode those exact
// bytes; nothing here rebuilds an equivalent struct by hand, because two
// hand-written shapes agreeing with each other proves nothing about the wire.

const evidenceFixtureDir = "../v3-service/testdata/evidence_wire"

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(evidenceFixtureDir, name+".json"))
	if err != nil {
		t.Fatalf("golden fixture missing (regenerate with "+
			"v3-service/testdata/generate_evidence_fixtures.py): %v", err)
	}
	return b
}

func decodeFixture(t *testing.T, name string) V3GenerateResponse {
	t.Helper()
	var got V3GenerateResponse
	if err := json.Unmarshal(readFixture(t, name), &got); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return got
}

// The regression that exposed the gap. The response type as it stood decoded
// the service's reply happily and dropped the whole envelope on the floor: no
// error, no warning, just absent evidence. Half of this test still describes
// that type, so the loss stays visible rather than becoming folklore.
func TestUnknownFieldsAreLostWithoutAnExplicitWireType(t *testing.T) {
	raw := readFixture(t, "01_verified_winner")

	// The pre-envelope shape, reproduced exactly.
	type legacyResponse struct {
		Code                 string                   `json:"code"`
		Passed               bool                     `json:"passed"`
		PhaseSolved          string                   `json:"phase_solved"`
		CandidatesTested     int                      `json:"candidates_tested"`
		WinningScore         float64                  `json:"winning_score"`
		TotalTokens          int                      `json:"total_tokens"`
		TotalTimeMs          float64                  `json:"total_time_ms"`
		VerificationEvidence []V3VerificationEvidence `json:"verification_evidence"`
	}
	var legacy legacyResponse
	if err := json.Unmarshal(raw, &legacy); err != nil {
		t.Fatalf("legacy decode failed: %v", err)
	}
	if !legacy.Passed || legacy.Code == "" {
		t.Fatal("fixture did not carry the fields the legacy type does read")
	}
	round, _ := json.Marshal(legacy)
	if string(round) == string(raw) {
		t.Fatal("fixture does not exercise the loss: nothing was dropped")
	}

	// The explicit wire type keeps it.
	got := decodeFixture(t, "01_verified_winner")
	if got.Evidence == nil {
		t.Fatal("the evidence envelope was dropped by V3GenerateResponse")
	}
	if got.Evidence.Identity.ContractID == "" ||
		got.Evidence.Evaluation.EvidenceStrength == "" ||
		len(got.Evidence.Coverage.Required) == 0 ||
		got.Evidence.Selection.Status == "" ||
		got.Evidence.Delivery.DeliveredContentHash == "" {
		t.Fatalf("envelope decoded with empty sections: %+v", got.Evidence)
	}
}

// Field-by-field equality against the JSON the service produced. A struct tag
// typo drops one field silently, which is exactly the class of bug the whole
// fixture apparatus exists to catch.
func TestEnvelopeSurvivesDecodeWithoutFieldLoss(t *testing.T) {
	raw := readFixture(t, "01_verified_winner")
	var asMap struct {
		Evidence map[string]interface{} `json:"evidence"`
	}
	if err := json.Unmarshal(raw, &asMap); err != nil {
		t.Fatal(err)
	}
	got := decodeFixture(t, "01_verified_winner")

	// Re-marshal the decoded struct and compare to the original sub-object.
	reencoded, err := json.Marshal(got.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]interface{}
	if err := json.Unmarshal(reencoded, &back); err != nil {
		t.Fatal(err)
	}
	original, _ := json.Marshal(asMap.Evidence)
	var want map[string]interface{}
	json.Unmarshal(original, &want)

	wantJSON, _ := json.Marshal(want)
	gotJSON, _ := json.Marshal(back)
	if string(wantJSON) != string(gotJSON) {
		t.Errorf("envelope changed crossing the boundary:\n got %s\nwant %s",
			gotJSON, wantJSON)
	}
}

// Every golden fixture, and what a strict consumer must conclude from it.
func TestGoldenFixtureAvailability(t *testing.T) {
	for _, c := range []struct {
		fixture   string
		want      EvidenceAvailability
		strength  string
		selection string
		describes bool
	}{
		{"01_verified_winner", EvidenceAvailable, "behavioral", "verified_winner", true},
		{"02_behavioral_incomplete_requirements", EvidenceAvailable, "behavioral", "best_not_closure_eligible", true},
		{"03_syntax_only", EvidenceAvailable, "syntax", "best_not_closure_eligible", true},
		{"04_best_not_closure_eligible", EvidenceAvailable, "runtime", "best_not_closure_eligible", true},
		{"05_unsupported_candidate", EvidenceAvailable, "syntax", "ineligible", true},
		{"06_no_verified_winner", EvidenceAvailable, "syntax", "ineligible", true},
		{"07_incomparable_records", EvidenceAvailable, "behavioral", "incomparable", true},
		{"08_tied_records", EvidenceAvailable, "behavioral", "tied", true},
		{"09_evidence_for_other_candidate", EvidenceAvailable, "behavioral", "verified_winner", false},
		{"11_unknown_wire_version", EvidenceUnavailable, "", "", false},
		{"12_malformed_identity", EvidenceUnavailable, "", "", false},
		{"13_closure_contradicts_execution", EvidenceUnavailable, "", "", false},
	} {
		c := c
		t.Run(c.fixture, func(t *testing.T) {
			got := decodeFixture(t, c.fixture)
			availability, reason := got.Evidence.Validate()
			if availability != c.want {
				t.Fatalf("availability = %q (%s), want %q", availability, reason, c.want)
			}
			if availability != EvidenceAvailable {
				// Unavailable is never a verdict about the candidate.
				if got.Evidence.Available() {
					t.Error("an unavailable envelope must not read as available")
				}
				if reason == "" {
					t.Error("an unavailable envelope must say why")
				}
				return
			}
			if got.Evidence.Evaluation.EvidenceStrength != c.strength {
				t.Errorf("strength = %q, want %q",
					got.Evidence.Evaluation.EvidenceStrength, c.strength)
			}
			if got.Evidence.Selection.Status != c.selection {
				t.Errorf("selection = %q, want %q",
					got.Evidence.Selection.Status, c.selection)
			}
			if got.Evidence.DescribesBytes(got.Code) != c.describes {
				t.Errorf("DescribesBytes(delivered) = %v, want %v",
					got.Evidence.DescribesBytes(got.Code), c.describes)
			}
		})
	}
}

// A producer that predates the envelope decodes cleanly and reports absence --
// not a failure, and not silently "available with zero values".
func TestLegacyResponseDecodesAsAbsentEvidence(t *testing.T) {
	got := decodeFixture(t, "10_legacy_no_envelope")
	if got.Evidence != nil {
		t.Fatal("a legacy response must not synthesise an envelope")
	}
	availability, reason := got.Evidence.Validate()
	if availability != EvidenceAbsent {
		t.Errorf("availability = %q, want absent", availability)
	}
	if reason == "" {
		t.Error("absence must state itself")
	}
	if !got.Passed || got.Code == "" {
		t.Error("the legacy fields must still decode")
	}
}

// Evidence about other bytes may not support provenance for the delivered
// ones. Nothing calls this for a decision yet; the rule is pinned so the later
// slice cannot invent a looser one at the call site.
func TestProvenanceRequiresMatchingHashes(t *testing.T) {
	winner := decodeFixture(t, "01_verified_winner")
	if ok, why := EvidenceSupportsProvenanceFor(winner.Evidence, winner.Code); !ok {
		t.Errorf("matching evidence rejected: %s", why)
	}
	if ok, _ := EvidenceSupportsProvenanceFor(winner.Evidence, "different bytes\n"); ok {
		t.Error("evidence about other bytes supported provenance")
	}

	stale := decodeFixture(t, "09_evidence_for_other_candidate")
	ok, why := EvidenceSupportsProvenanceFor(stale.Evidence, stale.Code)
	if ok {
		t.Error("stale evidence supported provenance for the delivered bytes")
	}
	if why != "evidence describes a different candidate" {
		t.Errorf("reason = %q, want the hash mismatch", why)
	}

	unknown := decodeFixture(t, "11_unknown_wire_version")
	if ok, _ := EvidenceSupportsProvenanceFor(unknown.Evidence, unknown.Code); ok {
		t.Error("an unreadable envelope supported provenance")
	}
}

// Nothing in the envelope is inferred from the legacy fields, and nothing in
// the legacy fields is inferred from the envelope. The two travel side by side
// precisely because `passed` cannot express what the envelope carries.
func TestEnvelopeIsNotDerivedFromLegacyFields(t *testing.T) {
	// A passing legacy response whose evidence is explicitly not closure-eligible.
	got := decodeFixture(t, "09_evidence_for_other_candidate")
	if !got.Passed {
		t.Fatal("fixture must carry passed=true")
	}
	if got.Evidence.Evaluation.EvidenceStrength == "" {
		t.Fatal("fixture must carry a strength")
	}
	// The unsupported fixture is the mirror: legacy passed=false, and the
	// envelope still reports what was measured rather than a failure.
	unsup := decodeFixture(t, "05_unsupported_candidate")
	if unsup.Passed {
		t.Fatal("fixture must carry passed=false")
	}
	if unsup.Evidence.Evaluation.Supported {
		t.Error("fixture must carry supported=false")
	}
	if unsup.Evidence.Evaluation.ExecutionStatus == "error" {
		t.Error("unsupported must not be reported as an execution error")
	}
	if a, _ := unsup.Evidence.Validate(); a != EvidenceAvailable {
		t.Error("an unsupported-but-well-formed envelope is still readable")
	}
}

// End to end: the Python bytes reach telemetry with every field intact, and
// reach the tool result with none of them. The guarded tool-result projection
// is the model's view; research fields that leak there become an interface
// nobody meant to publish.
func TestEnvelopeReachesTelemetryAndNotTheToolResult(t *testing.T) {
	raw := readFixture(t, "01_verified_winner")
	var golden struct {
		Code     string                 `json:"code"`
		Evidence map[string]interface{} `json:"evidence"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}

	// SSE frames are single-line; the committed fixtures are indented for
	// review. Compacting changes no content -- the assertions below compare
	// the decoded structure, not the whitespace.
	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		t.Fatal(err)
	}
	raw = compact.Bytes()

	sub := defaultBroker.subscribe()
	defer defaultBroker.unsubscribe(sub)

	dir := t.TempDir()
	var streamed []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v3/generate":
			w.Header().Set("Content-Type", "text/event-stream")
			fl, _ := w.(http.Flusher)
			// The service's own bytes, forwarded verbatim.
			for _, line := range []string{"event: result", "data: " + string(raw), "", "data: [DONE]", ""} {
				fmt.Fprint(w, line+"\n")
				if fl != nil {
					fl.Flush()
				}
			}
		case "/internal/cyclomatic_complexity":
			json.NewEncoder(w).Encode(map[string]interface{}{"functions": []interface{}{}})
		case "/internal/structural_check":
			json.NewEncoder(w).Encode(map[string]interface{}{"ok": true, "unresolved": []string{}})
		default:
			if strings.HasSuffix(r.URL.Path, "/syntax-check") {
				json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
				return
			}
			http.Error(w, "unexpected "+r.URL.Path, http.StatusTeapot)
		}
	}))
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(event string, _ interface{}) { streamed = append(streamed, event) }
	ctx.Ctx = context.Background()
	ctx.V3URL = srv.URL
	ctx.SandboxURL = srv.URL

	res, err := writeFileWithV3(filepath.Join(dir, "game.js"), "const a = 1;\n", ctx)
	if err != nil {
		t.Fatalf("writeFileWithV3: %v", err)
	}
	if res == nil || !res.Success {
		t.Fatalf("delivery failed: %+v", res)
	}

	// --- telemetry: whole envelope, no field loss ---------------------------
	var payload map[string]interface{}
	for {
		select {
		case ev := <-sub:
			if ev.Type == EvtStageEnd && ev.Stage == "v3" {
				payload = ev.Payload
			}
			if payload != nil {
				goto found
			}
			continue
		default:
			goto found
		}
	}
found:
	if payload == nil {
		t.Fatal("no v3 stage_end envelope was emitted")
	}
	evidence, ok := payload["evidence"].(map[string]interface{})
	if !ok {
		t.Fatalf("telemetry carries no evidence section: %v", payload)
	}
	if evidence["availability"] != string(EvidenceAvailable) {
		t.Errorf("availability = %v, want available", evidence["availability"])
	}
	// Re-serialise what telemetry holds and compare it to the Python bytes.
	viaTelemetry, err := json.Marshal(evidence["envelope"])
	if err != nil {
		t.Fatal(err)
	}
	fromPython, _ := json.Marshal(golden.Evidence)
	var a, b map[string]interface{}
	json.Unmarshal(viaTelemetry, &a)
	json.Unmarshal(fromPython, &b)
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if string(aj) != string(bj) {
		t.Errorf("evidence changed between Python and telemetry:\n got %s\nwant %s", aj, bj)
	}

	// --- tool result and SSE: nothing new -----------------------------------
	blob, _ := json.Marshal(res)
	if strings.Contains(string(blob), "wire_version") ||
		strings.Contains(string(blob), "closure_eligible") ||
		strings.Contains(string(blob), "evidence_strength") {
		t.Errorf("the envelope leaked into the tool result: %s", blob)
	}
	// Local validation is untouched by the envelope's arrival.
	if res.ValidationKind != ValidationKindSyntax || res.ValidationStatus != ValidationPassed {
		t.Errorf("local validation = %q/%q, want syntax/passed -- the service's "+
			"evidence must not displace what THIS process checked",
			res.ValidationKind, res.ValidationStatus)
	}
	for _, e := range streamed {
		if e != "v3_progress" && e != "v3_token" && e != "text" {
			t.Errorf("unexpected SSE projection %q", e)
		}
	}
}
