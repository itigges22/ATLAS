package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

// Does the pre-generation answer ever refuse an invocation that would in fact
// have closed?
//
// That is the only question enforce has to survive, and it is one-sided. A
// feasible invocation that produces no authorized candidate is not a defect:
// feasibility predicts possibility, not outcome. A case classified infeasible
// that then mints and consumes a valid authorization is a false negative, and
// there must be none.
//
// The corpus is fixed and outcome-independent: every case is a combination of
// contract shape, baseline, producer availability, command behaviour, adapter
// support, staleness and candidate kind, enumerated before anything ran.

type corpusCase struct {
	name       string
	contract   string
	file       string
	body       string
	commands   map[string]stubEffect
	before     func(t *testing.T, w *routeWorld)
	staleAfter bool
}

const corpusDoc = "# notes\n"

func corpusContracts() []struct{ name, contract, file, body string } {
	return []struct{ name, contract, file, body string }{
		{"no_contract", "", "solve.py", routeBaseline},
		{"task_mode_only", `{"task_mode":"work"}`, "solve.py", routeBaseline},
		{"unspecified_outputs",
			`{"task_mode":"work","output_knowledge":"unspecified"}`, "solve.py", routeBaseline},
		{"declared_empty",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":[]}`,
			"solve.py", routeBaseline},
		{"declared_code",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"]}`,
			"solve.py", routeBaseline},
		{"declared_document",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["notes.md"]}`,
			"notes.md", corpusDoc},
		{"declared_unsupported_adapter",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["data.bin"]}`,
			"data.bin", "\x00\x01binary\n"},
		{"one_command",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],` +
				`"verification_knowledge":"declared","verification":["pytest -q"]}`,
			"solve.py", routeBaseline},
		{"two_commands",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["solve.py"],` +
				`"verification_knowledge":"declared","verification":["pytest -q","ruff check ."]}`,
			"solve.py", routeBaseline},
		{"undeclared_target",
			`{"task_mode":"work","output_knowledge":"declared","expected_outputs":["other.py"]}`,
			"solve.py", routeBaseline},
	}
}

func corpusConditions() []struct {
	name     string
	commands map[string]stubEffect
	before   func(t *testing.T, w *routeWorld)
	stale    bool
} {
	pass := map[string]stubEffect{
		"pytest -q": {ExitCode: 0}, "ruff check .": {ExitCode: 0}}
	fail := map[string]stubEffect{
		"pytest -q": {ExitCode: 1}, "ruff check .": {ExitCode: 0}}
	timeout := map[string]stubEffect{
		"pytest -q": {ExitCode: -1, TimedOut: true}, "ruff check .": {ExitCode: 0}}
	return []struct {
		name     string
		commands map[string]stubEffect
		before   func(t *testing.T, w *routeWorld)
		stale    bool
	}{
		{name: "clean", commands: pass},
		{name: "command_fails", commands: fail},
		{name: "command_times_out", commands: timeout},
		{name: "syntax_baseline", commands: pass,
			before: func(t *testing.T, w *routeWorld) {
				seedBaseline(t, w, ValidationKindSyntax, "")
			}},
		{name: "behavioral_baseline", commands: pass,
			before: func(t *testing.T, w *routeWorld) {
				seedBaseline(t, w, ValidationKindSyntax, "pytest -q")
			}},
		{name: "producer_unavailable", commands: pass,
			before: func(t *testing.T, w *routeWorld) {
				w.ctx.SandboxURL = "http://127.0.0.1:1"
			}},
		{name: "staging_unavailable", commands: pass,
			before: func(t *testing.T, w *routeWorld) { *w.shellGone = true }},
		{name: "cancelled", commands: pass,
			before: func(t *testing.T, w *routeWorld) {
				cancelled, cancel := context.WithCancel(
					context.WithValue(context.Background(), requestIDKey, "req-route"))
				cancel()
				w.ctx.Ctx = cancelled
			}},
		{name: "stale_workspace", commands: pass, stale: true},
		{name: "repair_candidate", commands: pass,
			before: func(t *testing.T, w *routeWorld) {
				// The artifact already exists and was delivered once; this
				// invocation is a refinement of it.
				if err := os.WriteFile(w.path, []byte(routeWinner), 0o644); err != nil {
					t.Fatal(err)
				}
				observeDeliverable(w.ctx, w.path, []byte(routeWinner),
					ValidationKindSyntax, ValidationPassed, "")
			}},
	}
}

func corpus() []corpusCase {
	var out []corpusCase
	for _, c := range corpusContracts() {
		for _, cond := range corpusConditions() {
			out = append(out, corpusCase{
				name:       c.name + "/" + cond.name,
				contract:   c.contract,
				file:       c.file,
				body:       c.body,
				commands:   cond.commands,
				before:     cond.before,
				staleAfter: cond.stale,
			})
		}
	}
	// Two more shapes that need their own construction rather than a
	// combination: a safety-refused command, and a later invocation whose
	// availability changed underneath it.
	out = append(out,
		corpusCase{name: "safety_refused_command/clean",
			contract: `{"task_mode":"work","output_knowledge":"declared",` +
				`"expected_outputs":["solve.py"],"verification_knowledge":"declared",` +
				`"verification":["rm -rf /"]}`,
			file: "solve.py", body: routeBaseline},
		corpusCase{name: "command_mismatch/clean",
			contract: `{"task_mode":"work","output_knowledge":"declared",` +
				`"expected_outputs":["solve.py"],"verification_knowledge":"declared",` +
				`"verification":["pytest -q"]}`,
			file: "solve.py", body: routeBaseline,
			commands: map[string]stubEffect{"pytest -q": {ExitCode: 0, WriteTarget: "elsewhere\n"}}},
	)
	return out
}

type corpusRun struct {
	Name           string `json:"name"`
	Feasible       bool   `json:"feasible"`
	Reason         string `json:"reason"`
	Generations    int    `json:"generations"`
	GrantsConsumed int    `json:"grants_consumed"`
	LandedWinner   bool   `json:"landed_winner"`
	Disk           string `json:"disk_sha"`
	LedgerHash     string `json:"ledger_hash"`
	V3Used         bool   `json:"v3_used"`
	Authorization  string `json:"authorization_hash"`
	Success        bool   `json:"success"`
	Mutation       string `json:"mutation"`
	Validation     string `json:"validation"`
	Terminal       string `json:"terminal_reason"`
}

func runCorpusCase(t *testing.T, c corpusCase, mode FeasibilityMode) corpusRun {
	t.Helper()
	w := newRouteWorld(t, c.contract, c.commands)
	w.path = w.dir + "/" + c.file
	if err := os.WriteFile(w.path, []byte(c.body), 0o644); err != nil {
		t.Fatal(err)
	}
	w.ctx.FeasibilityMode = mode
	if c.before != nil {
		c.before(t, w)
	}
	if c.staleAfter {
		bumpWorkspace(w.ctx, w.dir+"/unrelated.py", contentSHA256("moved on\n"))
	}

	var decision FeasibilityDecision
	recs := captureShadow(t, func() {
		res, err := writeFileWithV3(w.path, c.body, w.ctx)
		if err != nil && res == nil {
			return
		}
		if res != nil {
			decision.Reason = FeasibilityReason("")
			_ = res
		}
	})
	out := corpusRun{Name: c.name, Generations: w.generateCalls()}
	for _, r := range recordsOfKind(recs, "shadow_invocation_feasibility") {
		out.Feasible, _ = r["feasible"].(bool)
		out.Reason, _ = r["reason"].(string)
	}
	out.GrantsConsumed = consumedGrants(recs)
	if body, err := os.ReadFile(w.path); err == nil {
		out.Disk = contentSHA256(string(body))
		out.LandedWinner = string(body) == routeWinner
	}
	w.ctx.LedgerMu.Lock()
	for _, d := range w.ctx.Ledger {
		if d != nil && d.CurrentHash != "" {
			out.LedgerHash += d.CurrentHash[:8]
		}
	}
	w.ctx.LedgerMu.Unlock()
	return out
}

func TestFeasibilityAgreementCorpus(t *testing.T) {
	cases := corpus()
	if len(cases) < 100 {
		t.Fatalf("corpus has %d cases, the floor is 100", len(cases))
	}

	observed := make(map[string]corpusRun, len(cases))
	var falseNegatives []string
	var feasibleUnrealized []string

	for _, c := range cases {
		run := runCorpusCase(t, c, FeasibilityObserve)
		observed[c.name] = run
		// Observe always generates, whatever it concluded -- except where the
		// request itself ended first. Cancellation has its own owner and its
		// own reason to skip; attributing it to feasibility would credit this
		// decision with something it did not do.
		wantGenerations := 1
		if strings.HasSuffix(c.name, "/cancelled") {
			wantGenerations = 0
		}
		if run.Generations != wantGenerations {
			t.Errorf("%s: %d generation calls under observe, want %d",
				c.name, run.Generations, wantGenerations)
		}
		// The one-sided rule.
		if !run.Feasible && run.GrantsConsumed > 0 {
			falseNegatives = append(falseNegatives,
				fmt.Sprintf("%s (reason %s, %d grants consumed)",
					c.name, run.Reason, run.GrantsConsumed))
		}
		if run.Feasible && run.GrantsConsumed == 0 {
			feasibleUnrealized = append(feasibleUnrealized, c.name)
		}
		if run.Reason != "" && !feasibilityReasons[FeasibilityReason(run.Reason)] {
			t.Errorf("%s: reason %q is outside the closed vocabulary", c.name, run.Reason)
		}
	}

	if len(falseNegatives) != 0 {
		t.Errorf("%d false negatives — an infeasible invocation consumed a valid "+
			"authorization:\n  %s", len(falseNegatives),
			strings.Join(falseNegatives, "\n  "))
	}
	t.Logf("corpus: %d cases, %d false negatives, %d feasible-but-unrealized",
		len(cases), len(falseNegatives), len(feasibleUnrealized))

	// The enforce replay.
	for _, c := range cases {
		want := observed[c.name]
		got := runCorpusCase(t, c, FeasibilityEnforce)
		if strings.HasSuffix(c.name, "/cancelled") {
			// Cancelled either way, by the same owner. Enforce adds nothing
			// and must take nothing away.
			if got.Generations != want.Generations || got.GrantsConsumed != want.GrantsConsumed {
				t.Errorf("%s: enforce changed a cancelled invocation (%+v vs %+v)",
					c.name, got, want)
			}
			continue
		}
		switch {
		case want.Feasible:
			// A feasible case must be byte-identical through generation and
			// delivery: enforce may only remove invocations, never change one.
			if got.Generations != want.Generations {
				t.Errorf("%s: enforce generated %d, observe %d",
					c.name, got.Generations, want.Generations)
			}
			if got.Disk != want.Disk {
				t.Errorf("%s: enforce left different bytes on disk", c.name)
			}
			if got.LandedWinner != want.LandedWinner ||
				got.GrantsConsumed != want.GrantsConsumed ||
				got.LedgerHash != want.LedgerHash {
				t.Errorf("%s: enforce differs from observe (%+v vs %+v)", c.name, got, want)
			}
		default:
			// An infeasible case must skip exactly, and land where the direct
			// path lands.
			if got.Generations != 0 {
				t.Errorf("%s: enforce generated %d for an infeasible invocation",
					c.name, got.Generations)
			}
			if got.GrantsConsumed != 0 {
				t.Errorf("%s: a skipped invocation consumed %d grants",
					c.name, got.GrantsConsumed)
			}
			if got.LandedWinner {
				t.Errorf("%s: a skipped invocation landed a candidate", c.name)
			}
		}
	}

	// No content leaks anywhere in the corpus's own records.
	blob, err := json.Marshal(observed)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{routeWinner, routeBaseline, corpusDoc, "pytest"} {
		if strings.Contains(string(blob), needle) {
			t.Errorf("the corpus record carries %q", needle)
		}
	}
}
