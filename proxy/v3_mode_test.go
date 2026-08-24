package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// Planning and candidate generation are separate capabilities. A single
// bypass boolean conflated them, so the 2026-08-23 canary's two arms differed
// in both at once and could not say which one moved the result. These tests
// pin the three-way mode that separates them, and in particular that
// planner-only cannot reach candidate generation.

func ctxWithMode(m V3Mode) *AgentContext {
	c := NewAgentContext("/workspace", Tier2Medium)
	c.V3Mode = m
	return c
}

func TestV3ModePredicates(t *testing.T) {
	for _, tc := range []struct {
		mode                      V3Mode
		planning, generation, byp bool
	}{
		{V3ModeFull, true, true, false},
		{V3ModeOff, false, false, true},
		{V3ModePlannerOnly, true, false, false},
	} {
		c := ctxWithMode(tc.mode)
		if got := c.V3PlanningEnabled(); got != tc.planning {
			t.Errorf("%s: planning=%v want %v", tc.mode, got, tc.planning)
		}
		if got := c.V3GenerationEnabled(); got != tc.generation {
			t.Errorf("%s: generation=%v want %v", tc.mode, got, tc.generation)
		}
		if got := c.V3Bypassed(); got != tc.byp {
			t.Errorf("%s: bypassed=%v want %v", tc.mode, got, tc.byp)
		}
	}
}

// Control mode stays byte-identical: bypass_v3 still means "no planning, no
// generation, baseline gates relaxed".
func TestControlModeUnchanged(t *testing.T) {
	c := ctxWithMode(V3ModeOff)
	if c.V3PlanningEnabled() || c.V3GenerationEnabled() {
		t.Fatal("off mode enabled a V3 capability")
	}
	if !c.V3Bypassed() {
		t.Fatal("off mode lost the demo-baseline relaxation")
	}
	if shouldGeneratePlan(c, "please write a long enough message to warrant a plan") {
		t.Fatal("off mode generated a plan")
	}
}

// Full mode stays byte-identical.
func TestFullModeUnchanged(t *testing.T) {
	c := ctxWithMode(V3ModeFull)
	if !c.V3PlanningEnabled() || !c.V3GenerationEnabled() {
		t.Fatal("full mode lost a V3 capability")
	}
	if c.V3Bypassed() {
		t.Fatal("full mode took the demo-baseline relaxation")
	}
	if !shouldGeneratePlan(c, "please write a long enough message to warrant a plan") {
		t.Fatal("full mode did not generate a plan")
	}
}

// Planner-only plans, and executes through the ordinary guarded path.
func TestPlannerOnlyPlansAndStaysGuarded(t *testing.T) {
	c := ctxWithMode(V3ModePlannerOnly)
	if !shouldGeneratePlan(c, "please write a long enough message to warrant a plan") {
		t.Fatal("planner-only did not generate a plan")
	}
	if c.V3GenerationEnabled() {
		t.Fatal("planner-only enabled candidate generation")
	}
	if c.V3Bypassed() {
		t.Fatal("planner-only took the demo-baseline gate relaxation; it must " +
			"execute through the ordinary guarded path")
	}
}

// The structural guarantee: every candidate-pipeline entry point is behind
// V3GenerationEnabled, so planner-only cannot reach one by a forgotten
// negation. A new entry point that reads BypassV3 directly fails this.
func TestGenerationEntryPointsAreBehindTheTypedPredicate(t *testing.T) {
	fset := token.NewFileSet()
	for _, file := range []string{"tools.go", "agent.go"} {
		f, err := parser.ParseFile(fset, file, nil, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", file, err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			sel, ok := n.(*ast.SelectorExpr)
			if !ok || sel.Sel == nil || sel.Sel.Name != "BypassV3" {
				return true
			}
			// Reads of the REQUEST field are the decode-time derivation and
			// are fine. Reads of the CONTEXT field are decisions, and every
			// decision must go through the typed predicates -- otherwise a
			// new call site can enable candidate generation for planner-only
			// by forgetting a negation.
			recv, _ := sel.X.(*ast.Ident)
			if recv != nil && recv.Name == "req" {
				return true
			}
			pos := fset.Position(sel.Pos())
			if recv != nil && recv.Name == "ctx" &&
				strings.Contains(readLine(t, file, pos.Line), "ctx.BypassV3 = req.BypassV3") {
				return true
			}
			t.Errorf("%s:%d reads ctx.BypassV3 for a decision; use "+
				"V3GenerationEnabled, V3PlanningEnabled or V3Bypassed so "+
				"planner-only cannot slip through", file, pos.Line)
			return true
		})
	}
}

func TestUnknownModeIsRefusedNotDefaulted(t *testing.T) {
	if ValidV3Mode("planner-only") {
		t.Error("hyphenated spelling accepted; the wire value is planner_only")
	}
	for _, bad := range []string{"", "FULL", "plannerOnly", "on", "true", "1"} {
		if ValidV3Mode(bad) {
			t.Errorf("ValidV3Mode(%q) = true; an unrecognised mode must be refused", bad)
		}
	}
	for _, good := range []string{"full", "off", "planner_only"} {
		if !ValidV3Mode(good) {
			t.Errorf("ValidV3Mode(%q) = false", good)
		}
	}
}

// No production default changes: a context built the ordinary way is full,
// and nothing but an explicit request can select planner-only.
func TestProductionDefaultIsFull(t *testing.T) {
	c := NewAgentContext("/workspace", Tier2Medium)
	c.V3Mode = V3ModeFull // what handleAgent derives for a request with neither field
	if !c.V3GenerationEnabled() || !c.V3PlanningEnabled() {
		t.Fatal("the derived default lost a capability")
	}
	src := readFileString(t, "agent.go")
	if strings.Contains(src, "V3ModePlannerOnly\n") &&
		!strings.Contains(src, "ValidV3Mode(req.V3ModeRaw)") {
		t.Fatal("planner-only reachable without an explicit validated request field")
	}
}

// Lifecycle state a mode must not disturb.
func TestModeDoesNotDisturbRequestLifecycleFields(t *testing.T) {
	for _, m := range []V3Mode{V3ModeFull, V3ModeOff, V3ModePlannerOnly} {
		c := ctxWithMode(m)
		if c.WorkingDir != "/workspace" || c.Tier != Tier2Medium {
			t.Fatalf("%s: mode changed unrelated context state", m)
		}
	}
}

func readFileString(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

func readLine(t *testing.T, path string, line int) string {
	t.Helper()
	lines := strings.Split(readFileString(t, path), "\n")
	if line-1 < 0 || line-1 >= len(lines) {
		return ""
	}
	return lines[line-1]
}

// The zero value is production default. A context built directly -- a struct
// literal, a test fixture, any caller that does not go through handleAgent --
// must keep every V3 capability, or the typed mode becomes a silent
// capability regression for every such caller.
func TestZeroValueModeIsFull(t *testing.T) {
	c := NewAgentContext("/workspace", Tier2Medium)
	if c.V3Mode != "" {
		t.Fatalf("expected an unset mode on a directly built context, got %q", c.V3Mode)
	}
	if !c.V3GenerationEnabled() {
		t.Error("zero value disabled candidate generation")
	}
	if !c.V3PlanningEnabled() {
		t.Error("zero value disabled planning")
	}
	if c.V3Bypassed() {
		t.Error("zero value took the demo-baseline relaxation")
	}
}

func TestNilContextEnablesNothing(t *testing.T) {
	var c *AgentContext
	if c.V3GenerationEnabled() || c.V3PlanningEnabled() || c.V3Bypassed() {
		t.Fatal("a nil context reported a capability")
	}
}

// Callers predating the mode set BypassV3 and never V3Mode. The zero value
// must keep honouring it, or bypass_v3 silently stops disabling V3 for every
// such caller -- which is how 24 tests failed the first time this landed.
func TestZeroValueModeStillHonoursBypassV3(t *testing.T) {
	c := NewAgentContext("/workspace", Tier2Medium)
	c.BypassV3 = true
	if c.V3Mode != "" {
		t.Fatalf("fixture set a mode; this test is about the zero value")
	}
	if c.V3PlanningEnabled() {
		t.Error("bypass_v3 no longer disables the planner")
	}
	if c.V3GenerationEnabled() {
		t.Error("bypass_v3 no longer disables candidate generation")
	}
	if !c.V3Bypassed() {
		t.Error("bypass_v3 lost the demo-baseline relaxation")
	}
	if shouldGeneratePlan(c, "please write a long enough message to warrant a plan") {
		t.Error("bypass_v3 context generated a plan")
	}
}

// An explicit mode outranks the legacy boolean, so a planner-only request is
// not silently downgraded by a stale bypass flag.
func TestExplicitModeOutranksBypassV3(t *testing.T) {
	c := NewAgentContext("/workspace", Tier2Medium)
	c.BypassV3 = true
	c.V3Mode = V3ModePlannerOnly
	if !c.V3PlanningEnabled() {
		t.Error("explicit planner_only was overridden by bypass_v3")
	}
	if c.V3GenerationEnabled() {
		t.Error("planner_only enabled generation")
	}
}
