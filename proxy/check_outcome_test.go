package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
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

// Composites through the production producer, with request counting so the
// short circuit is proven rather than assumed.
func TestFallbackSyntaxComposites(t *testing.T) {
	const jinja = "page.jinja" // NOT in syntaxGateLanguages; IS embedded-capable
	withScript := "<html><script>let a = 1;</script></html>"

	newSvc := func(embCalls *int, status int, body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/internal/embedded_script_check") {
				*embCalls++
				w.WriteHeader(status)
				w.Write([]byte(body))
				return
			}
			// whole-file sandbox check
			w.WriteHeader(200)
			w.Write([]byte(`{"valid":false,"errors":["SyntaxError: whole-file bad"]}`))
		}))
	}

	t.Run("jinja embedded passes -> aggregate passed", func(t *testing.T) {
		calls := 0
		svc := newSvc(&calls, 200, `{"ok":true,"findings":[]}`)
		defer svc.Close()
		ctx := NewAgentContext(t.TempDir(), Tier2Medium)
		ctx.SandboxURL, ctx.V3URL = svc.URL, svc.URL
		o := fallbackSyntaxOutcomeFor(ctx, jinja, withScript)
		if o.WholeFile.Status != ValidationNotApplicable {
			t.Errorf("WholeFile = %q, want not_applicable", o.WholeFile.Status)
		}
		if o.Embedded.Status != ValidationPassed {
			t.Errorf("Embedded = %q, want passed", o.Embedded.Status)
		}
		if got := o.aggregate().Status; got != ValidationPassed {
			t.Errorf("aggregate = %q, want passed", got)
		}
		if calls != 1 {
			t.Errorf("embedded calls = %d, want 1", calls)
		}
	})

	t.Run("jinja embedded unavailable -> aggregate not_run", func(t *testing.T) {
		ctx := NewAgentContext(t.TempDir(), Tier2Medium)
		ctx.SandboxURL, ctx.V3URL = "", "" // service absent
		o := fallbackSyntaxOutcomeFor(ctx, jinja, withScript)
		if o.WholeFile.Status != ValidationNotApplicable {
			t.Errorf("WholeFile = %q, want not_applicable", o.WholeFile.Status)
		}
		if o.Embedded.Status != ValidationNotRun {
			t.Errorf("Embedded = %q, want not_run", o.Embedded.Status)
		}
		if got := o.aggregate().Status; got != ValidationNotRun {
			t.Errorf("aggregate = %q, want not_run", got)
		}
	})

	t.Run("jinja embedded finding -> aggregate failed", func(t *testing.T) {
		calls := 0
		svc := newSvc(&calls, 200,
			`{"ok":true,"findings":[{"line":1,"message":"stray paren","snippet":"x)"}]}`)
		defer svc.Close()
		ctx := NewAgentContext(t.TempDir(), Tier2Medium)
		ctx.SandboxURL, ctx.V3URL = svc.URL, svc.URL
		agg := fallbackSyntaxOutcomeFor(ctx, jinja, withScript).aggregate()
		if agg.Status != ValidationFailed {
			t.Fatalf("aggregate = %q, want failed", agg.Status)
		}
		if !strings.HasPrefix(agg.Detail, embeddedScriptErrPrefix) {
			t.Errorf("diagnostic lost its prefix: %q", agg.Detail)
		}
		// Wrapper parity: only a demonstrated failure refuses.
		msg, ok := checkFallbackSyntax(ctx, jinja, withScript)
		if ok || msg != agg.Detail {
			t.Errorf("wrapper ok=%v msg=%q, want false and the same diagnostic", ok, msg)
		}
	})

	t.Run("whole-file failure short-circuits the embedded call", func(t *testing.T) {
		calls := 0
		svc := newSvc(&calls, 200, `{"ok":true,"findings":[]}`)
		defer svc.Close()
		ctx := NewAgentContext(t.TempDir(), Tier2Medium)
		ctx.SandboxURL, ctx.V3URL = svc.URL, svc.URL
		// .html IS whole-file gated, and the stub fails it.
		o := fallbackSyntaxOutcomeFor(ctx, "page.html", withScript)
		if o.WholeFile.Status != ValidationFailed {
			t.Fatalf("WholeFile = %q, want failed", o.WholeFile.Status)
		}
		if calls != 0 {
			t.Errorf("embedded calls = %d, want 0 -- the short circuit was lost", calls)
		}
		if o.Embedded.Status != ValidationNotRun {
			t.Errorf("Embedded = %q, want not_run (skipped but applicable)", o.Embedded.Status)
		}
		if o.aggregate().Status != ValidationFailed {
			t.Error("aggregate must remain the whole-file failure")
		}
	})

	t.Run("no embedded content -> not_applicable, zero calls", func(t *testing.T) {
		calls := 0
		svc := newSvc(&calls, 200, `{"ok":true,"findings":[]}`)
		defer svc.Close()
		ctx := NewAgentContext(t.TempDir(), Tier2Medium)
		ctx.SandboxURL, ctx.V3URL = "", svc.URL
		o := fallbackSyntaxOutcomeFor(ctx, jinja, "<html><p>no script here</p></html>")
		if o.Embedded.Status != ValidationNotApplicable {
			t.Errorf("Embedded = %q, want not_applicable", o.Embedded.Status)
		}
		if calls != 0 {
			t.Errorf("embedded calls = %d, want 0", calls)
		}
		if o.aggregate().Status != ValidationNotApplicable {
			t.Errorf("aggregate = %q, want not_applicable", o.aggregate().Status)
		}
	})
}
