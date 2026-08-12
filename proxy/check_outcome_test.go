package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// Outcome table for the whole-file sandbox check. The flat (string, bool) API
// returns ("", true) for TEN distinct conditions of which only one is a pass,
// so `true` means "do not refuse", never "validated". These pin the
// structured core that tells them apart.
func TestSandboxSyntaxOutcomeTable(t *testing.T) {
	sandbox := func(status int, body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(status)
			w.Write([]byte(body))
		}))
	}
	valid := sandbox(200, `{"valid":true}`)
	defer valid.Close()
	invalid := sandbox(200, `{"valid":false,"errors":["SyntaxError: bad"]}`)
	defer invalid.Close()
	broken := sandbox(200, `not json`)
	defer broken.Close()
	five00 := sandbox(500, `{}`)
	defer five00.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close() // now unreachable

	cases := []struct {
		name, path, url string
		nilCtx          bool
		want            ValidationStatus
	}{
		// Applicability is decided BEFORE availability: an unsupported
		// extension is not_applicable even with no sandbox configured.
		{"unsupported extension, no sandbox", "notes.txt", "", false, ValidationNotApplicable},
		{"unsupported extension, sandbox up", "notes.txt", valid.URL, false, ValidationNotApplicable},
		{"jinja is not whole-file gated", "page.jinja", valid.URL, false, ValidationNotApplicable},
		{"applicable, no sandbox", "mod.py", "", false, ValidationNotRun},
		{"applicable, nil context", "mod.py", valid.URL, true, ValidationNotRun},
		{"applicable, unreachable", "mod.py", deadURL, false, ValidationNotRun},
		{"applicable, non-200", "mod.py", five00.URL, false, ValidationNotRun},
		{"applicable, undecodable", "mod.py", broken.URL, false, ValidationNotRun},
		{"demonstrated pass", "mod.py", valid.URL, false, ValidationPassed},
		{"demonstrated failure", "mod.py", invalid.URL, false, ValidationFailed},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var ctx *AgentContext
			if !c.nilCtx {
				ctx = NewAgentContext(t.TempDir(), Tier2Medium)
				ctx.SandboxURL = c.url
			}
			got := sandboxSyntaxOutcome(ctx, c.path, "A = 1\n")
			if got.Status != c.want {
				t.Errorf("Status = %q, want %q (detail %q)", got.Status, c.want, got.Detail)
			}
			// Wrapper parity: only a demonstrated failure may refuse.
			_, ok := checkSandboxSyntax(ctx, c.path, "A = 1\n")
			if wantOK := c.want != ValidationFailed; ok != wantOK {
				t.Errorf("wrapper ok = %v, want %v -- fail-open behavior changed", ok, wantOK)
			}
			if got.Status == ValidationPassed && !got.attempted() {
				t.Error("a pass must count as attempted")
			}
			if got.Status == ValidationNotApplicable && got.applicable() {
				t.Error("not_applicable must not count as applicable")
			}
		})
	}
}

func TestFallbackSyntaxAggregate(t *testing.T) {
	pass := checkOutcome{Status: ValidationPassed}
	fail := checkOutcome{Status: ValidationFailed, Detail: "boom"}
	notRun := checkOutcome{Status: ValidationNotRun, Detail: "unreachable"}
	na := checkOutcome{Status: ValidationNotApplicable}
	unk := checkOutcome{Status: ValidationUnknown}

	for _, c := range []struct {
		name string
		in   fallbackSyntaxOutcome
		want ValidationStatus
	}{
		{"failure is decisive over pass", fallbackSyntaxOutcome{pass, fail}, ValidationFailed},
		{"failure is decisive over not_run", fallbackSyntaxOutcome{notRun, fail}, ValidationFailed},
		{"applicable unavailable wins over pass", fallbackSyntaxOutcome{pass, notRun}, ValidationNotRun},
		{"pass plus not_applicable", fallbackSyntaxOutcome{pass, na}, ValidationPassed},
		{"not_applicable plus pass", fallbackSyntaxOutcome{na, pass}, ValidationPassed},
		{"all not_applicable", fallbackSyntaxOutcome{na, na}, ValidationNotApplicable},
		{"unknown surfaces as unknown, not not_run", fallbackSyntaxOutcome{pass, unk}, ValidationUnknown},
		{"unknown alone surfaces as unknown", fallbackSyntaxOutcome{unk, na}, ValidationUnknown},
		{"failure still decisive over unknown", fallbackSyntaxOutcome{unk, fail}, ValidationFailed},
		{"unknown outranks not_run", fallbackSyntaxOutcome{notRun, unk}, ValidationUnknown},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := c.in.aggregate(); got.Status != c.want {
				t.Errorf("aggregate = %q, want %q", got.Status, c.want)
			}
		})
	}
}
