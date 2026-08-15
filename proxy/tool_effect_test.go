package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// Effect classification is a property of the tool, declared on its ToolDef,
// so a new tool cannot be registered without deciding what it can do to the
// workspace. A parallel map keyed by name would drift; this cannot.
//
// The classes describe CAPABILITY, not outcome. A direct mutator still
// reports applied/refused/failed/none per branch; the effect only says the
// tool is capable of mutating and therefore owes a local classification.
func TestEveryRegisteredToolDeclaresAnEffect(t *testing.T) {
	if len(toolRegistry) == 0 {
		t.Fatal("tool registry is empty; this test would prove nothing")
	}
	var missing []string
	for name, def := range toolRegistry {
		if def.Effect == ToolEffectUnknown {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("tools registered without an effect class: %v", missing)
	}
}

// The verified distribution, pinned so a reclassification has to be
// deliberate. stop_background is CommandUnobserved rather than read-only or
// control: it sends SIGTERM/SIGKILL and performs no filesystem comparison, so
// a process killed mid-write may leave partial bytes nobody measured.
func TestEffectDistributionMatchesTheVerifiedAudit(t *testing.T) {
	want := map[string]ToolEffect{
		"read_file": ToolEffectReadOnly, "search_files": ToolEffectReadOnly,
		"list_directory": ToolEffectReadOnly, "find_file": ToolEffectReadOnly,
		"outline_file": ToolEffectReadOnly, "tail_background": ToolEffectReadOnly,

		"write_file": ToolEffectDirectMutation, "edit_file": ToolEffectDirectMutation,
		"structural_edit": ToolEffectDirectMutation, "delete_file": ToolEffectDirectMutation,
		"move_file": ToolEffectDirectMutation, "insert_after": ToolEffectDirectMutation,
		"replace_lines": ToolEffectDirectMutation,

		"run_command":     ToolEffectCommandUnobserved,
		"run_background":  ToolEffectCommandUnobserved,
		"stop_background": ToolEffectCommandUnobserved,
	}
	for name, expect := range want {
		def, ok := toolRegistry[name]
		if !ok {
			t.Errorf("expected tool %q is not registered", name)
			continue
		}
		if def.Effect != expect {
			t.Errorf("%s: effect = %q, want %q", name, def.Effect, expect)
		}
	}
	for name := range toolRegistry {
		if _, known := want[name]; !known {
			t.Errorf("tool %q is registered but absent from the verified audit; "+
				"classify it deliberately", name)
		}
	}
}

func TestDirectMutatorsAreNotClassifiableAtTheBoundary(t *testing.T) {
	for name, def := range toolRegistry {
		if def.Effect != ToolEffectDirectMutation {
			continue
		}
		// A direct mutator's outcome is only knowable at the branch that
		// performed it, so the boundary must never invent one.
		if def.Effect.BoundaryClassifiable() {
			t.Errorf("%s: a direct mutator must not be classifiable at the "+
				"shared boundary", name)
		}
	}
}

// Boundary classification through the REAL executeToolCall, not a mirror.
func TestBoundaryClassifiesReadOnlyAndCommandTools(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	cases := []struct {
		name string
		tool string
		args string
		want MutationStatus
	}{
		// Read-only: mutates nothing whether it succeeds or fails.
		{"read success", "read_file", `{"path":"f.txt"}`, MutationNone},
		{"read missing file", "read_file", `{"path":"nope.txt"}`, MutationNone},
		{"list", "list_directory", `{"path":"."}`, MutationNone},
		// Pre-dispatch refusals prove no handler ran.
		{"unknown tool", "no_such_tool", `{}`, MutationNone},
		{"missing args", "read_file", ``, MutationNone},
	}
	os.WriteFile(filepath.Join(dir, "f.txt"), []byte("hello\n"), 0o644)
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res := executeToolCall(c.tool, json.RawMessage(c.args), ctx)
			if res.MutationStatus != c.want {
				t.Errorf("MutationStatus = %q, want %q", res.MutationStatus, c.want)
			}
			if !res.Classified() {
				t.Errorf("result not fully classified: %+v", res)
			}
			if res.ValidationStatus.Passed() {
				t.Error("validation was synthesized as passed")
			}
		})
	}
}

// A command tool that reached its handler is Unobserved regardless of exit
// status: a subprocess may have written before failing.
func TestCommandToolsReportUnobservedIncludingFailure(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.TrustMode = trustFullyTrusted
	ctx.VerifyOnHost = true
	ctx.StreamFn = func(string, interface{}) {}

	for _, c := range []struct{ name, cmd string }{
		{"success", "true"},
		{"non-zero exit", "false"},
		{"writes then fails", "echo x > side_effect.txt; exit 3"},
	} {
		t.Run(c.name, func(t *testing.T) {
			args, _ := json.Marshal(map[string]string{"command": c.cmd})
			res := executeToolCall("run_command", args, ctx)
			if res.MutationStatus != MutationUnobserved {
				t.Errorf("MutationStatus = %q, want unobserved", res.MutationStatus)
			}
			if res.MutationStatus.Applied() {
				t.Error("unobserved must never read as applied")
			}
			if !res.Classified() {
				t.Errorf("result not classified: %+v", res)
			}
		})
	}
}

// The boundary must never invent a direct mutator's outcome.
func TestBoundaryDoesNotClassifyDirectMutators(t *testing.T) {
	if ToolEffectDirectMutation.BoundaryClassifiable() {
		t.Fatal("direct mutation must not be boundary-classifiable")
	}
	for _, e := range []ToolEffect{ToolEffectReadOnly, ToolEffectCommandUnobserved} {
		if !e.BoundaryClassifiable() {
			t.Errorf("%q should be boundary-classifiable", e)
		}
	}
	if ToolEffectUnknown.BoundaryClassifiable() {
		t.Fatal("unknown effect must not be boundary-classifiable")
	}
}

// Intermediate-state safety check. The boundary now declines to classify
// direct mutators, and "declines to classify" must mean exactly that: the
// result passes through untouched and the mutation still happens. If
// BoundaryClassifiable were ever read as an authorization check, write_file
// would start failing operationally while every unit test still passed.
func TestDirectMutatorStillMutatesWhileUnclassified(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	args, _ := json.Marshal(map[string]string{
		"path": "new.py", "content": "VALUE = 1\n"})
	res := executeToolCall("write_file", args, ctx)

	// 1. The mutation actually occurred.
	got, err := os.ReadFile(filepath.Join(dir, "new.py"))
	if err != nil {
		t.Fatalf("write_file no longer writes: %v", err)
	}
	if string(got) != "VALUE = 1\n" {
		t.Fatalf("bytes on disk = %q", string(got))
	}
	// 2. Legacy Success is untouched.
	if !res.Success {
		t.Fatalf("legacy Success regressed to false: %+v", res)
	}
	// 3. Any classification present came from the LOCAL branch, not the
	// boundary. write_file's direct path is migrated (Slice 1), so applied is
	// expected here; what must never happen is the boundary inventing it for
	// a branch that has not been migrated.
	if !ToolEffectDirectMutation.BoundaryClassifiable() && res.MutationStatus != MutationApplied {
		t.Errorf("locally-classified direct write should report applied, got %q",
			res.MutationStatus)
	}
	// 4. Unknown is a pending-migration state, not an operational rejection.
	if res.Error != "" {
		t.Errorf("unclassified direct mutator was rejected: %q", res.Error)
	}
}

// The same for a direct mutator that legitimately fails: still no boundary
// classification, still the real error, still no invented facts.
func TestDirectMutatorFailurePassesThroughUnclassified(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	args, _ := json.Marshal(map[string]string{"path": "missing.txt"})
	res := executeToolCall("delete_file", args, ctx)
	if res.Success {
		t.Fatal("deleting a missing file unexpectedly succeeded")
	}
	if res.MutationStatus != MutationUnknown {
		t.Errorf("boundary classified a failing direct mutator as %q",
			res.MutationStatus)
	}
	if res.Error == "" {
		t.Error("the real error was suppressed")
	}
}

// --- write_file Slice 1: non-code and simple filesystem outcomes ------------
//
// writeFileDirect is the terminal writer for the direct (non-V3) path. It
// demonstrates the mutation but performs no syntax check, so it reports
// applied plus an honest validation state -- never passed.

func TestWriteFileDirectClassification(t *testing.T) {
	for _, c := range []struct {
		name, file, content string
		wantKind            ValidationKind
		wantStatus          ValidationStatus
	}{
		{"non-code write", "notes.txt", "hello\n",
			ValidationKindNone, ValidationNotApplicable},
		{"recognized code, unvalidated at this layer", "mod.py", "A = 1\n",
			ValidationKindSyntax, ValidationNotRun},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			ctx := NewAgentContext(dir, Tier2Medium)
			ctx.PermissionMode = PermissionYolo
			ctx.StreamFn = func(string, interface{}) {}

			args, _ := json.Marshal(map[string]string{"path": c.file, "content": c.content})
			res := executeToolCall("write_file", args, ctx)

			if !res.Success {
				t.Fatalf("legacy Success regressed: %+v", res)
			}
			if res.MutationStatus != MutationApplied {
				t.Errorf("MutationStatus = %q, want applied", res.MutationStatus)
			}
			if res.ValidationKind != c.wantKind {
				t.Errorf("ValidationKind = %q, want %q", res.ValidationKind, c.wantKind)
			}
			if res.ValidationStatus != c.wantStatus {
				t.Errorf("ValidationStatus = %q, want %q", res.ValidationStatus, c.wantStatus)
			}
			if res.ValidationStatus.Passed() {
				t.Error("a successful write must not synthesize validation passed")
			}
			if !res.Classified() {
				t.Errorf("result not fully classified: %+v", res)
			}
			got, err := os.ReadFile(filepath.Join(dir, c.file))
			if err != nil || string(got) != c.content {
				t.Fatalf("final bytes wrong: %q err=%v", string(got), err)
			}
		})
	}
}

// Refusals that happen before the handler can mutate. The unread-existing-file
// guard lives in the agent loop, not here, so this uses a deny-list refusal --
// a pre-dispatch check inside executeToolCall itself.
func TestWriteFileRefusalBeforeMutationLeavesBytes(t *testing.T) {
	dir := t.TempDir()
	prior := "SECRET=keepme\n"
	os.WriteFile(filepath.Join(dir, ".env"), []byte(prior), 0o644)

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}

	args, _ := json.Marshal(map[string]string{"path": ".env", "content": "SECRET=stolen\n"})
	res := executeToolCall("write_file", args, ctx)
	if res.Success {
		t.Fatal("expected the deny-list to refuse a write to .env")
	}
	got, _ := os.ReadFile(filepath.Join(dir, ".env"))
	if string(got) != prior {
		t.Fatalf("prior bytes changed on a refusal: %q", string(got))
	}
	// Pre-dispatch refusal: no handler ran, so nothing could have mutated.
	if res.MutationStatus != MutationNone {
		t.Errorf("MutationStatus = %q, want none", res.MutationStatus)
	}
	if res.MutationStatus.Applied() {
		t.Error("a refused write must never read as applied")
	}
	if !res.Classified() {
		t.Errorf("pre-dispatch refusal not classified: %+v", res)
	}
}

// --- write_file Slice 2: local syntax outcomes ------------------------------
//
// A caller upgrades not_run only when it knows the check examined the exact
// bytes the write path used. Both routes below satisfy that: the gate checks
// input.Content and refuses before any byte lands, and the warned write checks
// `content` and hands that same value to writeFileDirect.

// Invalid NEW code lands with a warning. Applied and failed are orthogonal:
// the file is on disk AND it does not parse.
func TestInvalidNewFileIsAppliedAndSyntaxFailed(t *testing.T) {
	dir := t.TempDir()
	got, res := writeThroughLoop(t, dir, "fresh.py", banLoopInvalidBody, true)

	if !res.Success {
		t.Fatalf("legacy Success regressed for a warned write: %+v", res)
	}
	if res.MutationStatus != MutationApplied {
		t.Errorf("MutationStatus = %q, want applied", res.MutationStatus)
	}
	if res.ValidationKind != ValidationKindSyntax || res.ValidationStatus != ValidationFailed {
		t.Errorf("validation = %q/%q, want syntax/failed",
			res.ValidationKind, res.ValidationStatus)
	}
	if res.ValidationDetail == "" {
		t.Error("ValidationDetail must carry the real diagnostic")
	}
	if got != banLoopInvalidBody {
		t.Errorf("bytes on disk are not the checked bytes")
	}
}

// Healthy existing code protected from an invalid overwrite.
func TestInvalidOverwriteIsRefusedAndSyntaxFailed(t *testing.T) {
	dir := t.TempDir()
	// Session-owned: create it valid first, then push invalid bytes.
	_, _ = writeThroughLoop(t, dir, "mod.py", banLoopValidBody, true)
	got, res := writeThroughLoop(t, dir, "mod.py", banLoopInvalidBody, true)

	if res.Success {
		t.Fatalf("an invalid overwrite of healthy code must not succeed: %+v", res)
	}
	if res.MutationStatus != MutationRefused {
		t.Errorf("MutationStatus = %q, want refused", res.MutationStatus)
	}
	if res.MutationStatus.Applied() {
		t.Error("a refusal must never read as applied")
	}
	if res.ValidationKind != ValidationKindSyntax || res.ValidationStatus != ValidationFailed {
		t.Errorf("validation = %q/%q, want syntax/failed",
			res.ValidationKind, res.ValidationStatus)
	}
	if res.ValidationDetail == "" {
		t.Error("ValidationDetail must carry the real diagnostic")
	}
	if got != banLoopValidBody {
		t.Errorf("prior valid bytes were not preserved: %q", got)
	}
}

// writeThroughLoop drives one write_file call through executeToolCall with a
// working syntax checker, returning the final on-disk bytes and the result.
func writeThroughLoop(t *testing.T, dir, rel, content string, withSandbox bool) (string, *ToolResult) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/syntax-check") {
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			valid := !strings.Contains(in.Code, "def alpha(n:")
			out := map[string]interface{}{"valid": valid}
			if !valid {
				out["errors"] = []string{"SyntaxError: invalid syntax (line 4)"}
			}
			json.NewEncoder(w).Encode(out)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.StreamFn = func(string, interface{}) {}
	if withSandbox {
		ctx.SandboxURL = srv.URL
	}
	args, _ := json.Marshal(map[string]string{"path": rel, "content": content})
	res := executeToolCall("write_file", args, ctx)
	b, _ := os.ReadFile(filepath.Join(dir, rel))
	return string(b), res
}

// --- Phase 3A: ledger observation at the shared boundary --------------------
//
// The ledger records what each call did to the session's deliverables. It is
// observational: every assertion below also checks that the tool's own result
// and the bytes on disk are what they were before the ledger existed.

func ledgerToolCtx(t *testing.T, dir string) *AgentContext {
	t.Helper()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	ctx.TrustMode = trustFullyTrusted
	ctx.VerifyOnHost = true
	ctx.StreamFn = func(string, interface{}) {}
	return ctx
}

func ledgerOf(t *testing.T, ctx *AgentContext, path string) *DeliverableState {
	t.Helper()
	return ctx.Ledger[ledgerKey(ctx, path)]
}

// diskHash is the ledger's only real claim: the recorded hash describes the
// bytes a reader would find right now.
func diskHash(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read %s: %v", p, err)
	}
	return hashBytes(b)
}

func TestContentMutatorsRecordTheBytesThatLanded(t *testing.T) {
	for _, c := range []struct {
		name, tool, seed string
		args             map[string]interface{}
		want             string
	}{
		{"write_file", "write_file", "",
			map[string]interface{}{"path": "solve.py", "content": "A = 1\n"},
			"A = 1\n"},
		{"edit_file", "edit_file", "A = 1\n",
			map[string]interface{}{"path": "solve.py", "old_str": "A = 1", "new_str": "A = 2"},
			"A = 2\n"},
		{"insert_after", "insert_after", "A = 1\n",
			map[string]interface{}{"path": "solve.py", "line": 1, "content": "B = 2"},
			"A = 1\nB = 2\n"},
		{"replace_lines", "replace_lines", "A = 1\nB = 2\n",
			map[string]interface{}{"path": "solve.py", "start_line": 1, "end_line": 1,
				"expected_first_line": "A = 1", "expected_last_line": "A = 1",
				"content": "A = 99"},
			"A = 99\nB = 2\n"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			full := filepath.Join(dir, "solve.py")
			if c.seed != "" {
				os.WriteFile(full, []byte(c.seed), 0o644)
			}
			ctx := ledgerToolCtx(t, dir)
			if c.seed != "" {
				// The edit tools require a current read; that guard is not
				// what this test is measuring.
				r, _ := json.Marshal(map[string]string{"path": "solve.py"})
				executeToolCall("read_file", r, ctx)
			}
			args, _ := json.Marshal(c.args)
			res := executeToolCall(c.tool, args, ctx)
			if !res.Success {
				t.Fatalf("%s failed: %v", c.tool, res.Error)
			}
			got, _ := os.ReadFile(full)
			if string(got) != c.want {
				t.Fatalf("disk bytes = %q, want %q", got, c.want)
			}
			d := ledgerOf(t, ctx, "solve.py")
			if d == nil {
				t.Fatal("the mutation was not recorded")
			}
			if d.CurrentHash != diskHash(t, full) {
				t.Errorf("ledger hash does not describe the bytes on disk")
			}
			if d.CurrentSize != len(c.want) {
				t.Errorf("CurrentSize = %d, want %d", d.CurrentSize, len(c.want))
			}
			// The tool's own evidence is copied, never upgraded.
			if d.ValidationStatus != res.ValidationStatus {
				t.Errorf("ledger status %q != tool status %q",
					d.ValidationStatus, res.ValidationStatus)
			}
			if d.CheckpointHash != "" && res.ValidationStatus != ValidationPassed {
				t.Error("checkpointed bytes that were never reported as passed")
			}
		})
	}
}

// A refused write must leave no trace: the prior bytes stand and the ledger
// does not record a generation that never happened.
func TestRefusedMutationDoesNotAdvanceTheLedger(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, ".env"), []byte("SECRET=keepme\n"), 0o644)
	ctx := ledgerToolCtx(t, dir)
	args, _ := json.Marshal(map[string]string{"path": ".env", "content": "SECRET=stolen\n"})
	res := executeToolCall("write_file", args, ctx)
	if res.Success {
		t.Fatal("deny-list did not refuse")
	}
	if d := ledgerOf(t, ctx, ".env"); d != nil {
		t.Errorf("a pre-dispatch refusal created a ledger entry: %+v", d)
	}
}

func TestDeleteRecordsATombstoneOnlyWhenTheFileIsGone(t *testing.T) {
	dir := t.TempDir()
	ctx := ledgerToolCtx(t, dir)
	w, _ := json.Marshal(map[string]string{"path": "solve.py", "content": "A = 1\n"})
	executeToolCall("write_file", w, ctx)

	// Failed delete of a path this session never wrote: nothing invented.
	d1, _ := json.Marshal(map[string]string{"path": "never_existed.py"})
	if res := executeToolCall("delete_file", d1, ctx); res.Success {
		t.Fatal("deleting a missing file unexpectedly succeeded")
	}
	if d := ledgerOf(t, ctx, "never_existed.py"); d != nil {
		t.Errorf("a failed delete fabricated a tombstone: %+v", d)
	}

	d2, _ := json.Marshal(map[string]string{"path": "solve.py"})
	if res := executeToolCall("delete_file", d2, ctx); !res.Success {
		t.Fatalf("delete failed: %v", res.Error)
	}
	d := ledgerOf(t, ctx, "solve.py")
	if d == nil || !d.Tombstoned {
		t.Fatalf("delete was not tombstoned: %+v", d)
	}
	if !d.RestoreProhibited {
		t.Error("a deliberate delete must prohibit automatic restoration")
	}
	if _, s := d.CurrentValidation(); s != ValidationUnknown {
		t.Error("a deleted path still reports a current verdict")
	}
}

func TestMoveTombstonesTheSourceAndObservesTheDestinationFresh(t *testing.T) {
	dir := t.TempDir()
	ctx := ledgerToolCtx(t, dir)
	w, _ := json.Marshal(map[string]string{"path": "old.py", "content": "A = 1\n"})
	executeToolCall("write_file", w, ctx)

	m, _ := json.Marshal(map[string]string{"source": "old.py", "destination": "new.py"})
	if res := executeToolCall("move_file", m, ctx); !res.Success {
		t.Fatalf("move failed: %v", res.Error)
	}
	src := ledgerOf(t, ctx, "old.py")
	if src == nil || !src.Tombstoned || !strings.HasPrefix(src.TombstoneReason, "moved:") {
		t.Fatalf("source not tombstoned as moved: %+v", src)
	}
	dst := ledgerOf(t, ctx, "new.py")
	if dst == nil {
		t.Fatal("destination not observed")
	}
	if dst.CurrentHash != diskHash(t, filepath.Join(dir, "new.py")) {
		t.Error("destination hash does not describe the bytes on disk")
	}
	if _, s := dst.CurrentValidation(); s == ValidationPassed {
		t.Error("a rename inherited a verdict earned under the old name")
	}
}

// A shell command can rewrite anything. Paths it did not touch keep their
// evidence; paths it changed lose it, because the verdict described bytes
// that are gone.
func TestRunCommandRehashesTrackedPaths(t *testing.T) {
	dir := t.TempDir()
	ctx := ledgerToolCtx(t, dir)
	for _, f := range []string{"quiet.py", "noisy.py"} {
		w, _ := json.Marshal(map[string]string{"path": f, "content": "A = 1\n"})
		executeToolCall("write_file", w, ctx)
	}
	quietBefore := ledgerOf(t, ctx, "quiet.py").CurrentHash

	c, _ := json.Marshal(map[string]string{"command": "printf 'REWRITTEN\\n' > noisy.py"})
	if res := executeToolCall("run_command", c, ctx); res.MutationStatus != MutationUnobserved {
		t.Fatalf("run_command classification changed: %q", res.MutationStatus)
	}
	if q := ledgerOf(t, ctx, "quiet.py"); q.CurrentHash != quietBefore {
		t.Error("an untouched path was re-recorded")
	}
	n := ledgerOf(t, ctx, "noisy.py")
	if n.CurrentHash != diskHash(t, filepath.Join(dir, "noisy.py")) {
		t.Error("ledger did not catch up to the command's rewrite")
	}
	if _, s := n.CurrentValidation(); s != ValidationUnknown {
		t.Error("a command-rewritten path kept a verdict about bytes that are gone")
	}
}

// run_background makes the workspace concurrently mutable, and the flag
// survives a start that reported failure: without a job_id there is nothing
// to confirm an exit with.
func TestBackgroundRaisesTheHazardEvenWhenTheStartFails(t *testing.T) {
	ctx := ledgerToolCtx(t, t.TempDir())
	ctx.SandboxURL = "" // start cannot succeed
	args, _ := json.Marshal(map[string]string{"command": "python -m http.server"})
	res := executeToolCall("run_background", args, ctx)
	if res.Success {
		t.Fatal("expected the start to fail with no sandbox configured")
	}
	if !workspaceHazardous(ctx) {
		t.Error("a background start that may have run left no hazard")
	}
}

func TestStopBackgroundClearsTheHazardOnlyOnAReapedExit(t *testing.T) {
	var exitCode *int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/jobs/start"):
			json.NewEncoder(w).Encode(map[string]interface{}{"job_id": "j1", "pid": 4242})
		case strings.HasSuffix(r.URL.Path, "/stop"):
			out := map[string]interface{}{"job_id": "j1", "killed": true,
				"stdout": []string{}, "stderr": []string{}}
			if exitCode != nil {
				out["exit_code"] = *exitCode
			}
			json.NewEncoder(w).Encode(out)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	ctx := ledgerToolCtx(t, t.TempDir())
	ctx.SandboxURL = srv.URL
	start, _ := json.Marshal(map[string]string{"command": "sleep 60"})
	executeToolCall("run_background", start, ctx)
	if !workspaceHazardous(ctx) {
		t.Fatal("run_background did not raise the hazard")
	}

	// Signalled but not reaped: the writer may still be flushing.
	stop, _ := json.Marshal(map[string]string{"job_id": "j1"})
	executeToolCall("stop_background", stop, ctx)
	if !workspaceHazardous(ctx) {
		t.Error("hazard cleared without a confirmed exit")
	}

	zero := 0
	exitCode = &zero
	executeToolCall("run_background", start, ctx)
	executeToolCall("stop_background", stop, ctx)
	executeToolCall("stop_background", stop, ctx)
	if workspaceHazardous(ctx) {
		t.Error("reaped exits did not clear the hazard")
	}
}

// The outcome table for the seven direct mutators, measured rather than
// asserted from the source. Every branch reachable without the V3 service is
// driven through the real boundary; the table it produces is logged, and the
// invariants that must hold for EVERY row are enforced:
//
//  1. a recorded current hash always describes the bytes on disk;
//  2. a checkpoint exists only where the tool reported an explicit pass;
//  3. a branch that proves it mutated nothing records nothing;
//  4. a branch that DID change disk never reports MutationUnknown, which
//     would mean an unmigrated producer is still guessing.
func TestDirectMutatorOutcomeTable(t *testing.T) {
	type step struct {
		tool string
		args map[string]interface{}
	}
	cases := []struct {
		name  string
		seed  map[string]string
		read  []string
		steps []step
		path  string // the deliverable this row is about
	}{
		{"write_file/new non-code", nil, nil,
			[]step{{"write_file", map[string]interface{}{"path": "notes.txt", "content": "hi\n"}}}, "notes.txt"},
		{"write_file/new code", nil, nil,
			[]step{{"write_file", map[string]interface{}{"path": "m.py", "content": "A = 1\n"}}}, "m.py"},
		{"write_file/deny-list refusal", map[string]string{".env": "S=1\n"}, nil,
			[]step{{"write_file", map[string]interface{}{"path": ".env", "content": "S=2\n"}}}, ".env"},
		{"write_file/escapes workspace", nil, nil,
			[]step{{"write_file", map[string]interface{}{"path": "../out.py", "content": "A = 1\n"}}}, "../out.py"},

		{"edit_file/applied", map[string]string{"m.py": "A = 1\n"}, []string{"m.py"},
			[]step{{"edit_file", map[string]interface{}{"path": "m.py", "old_str": "A = 1", "new_str": "A = 2"}}}, "m.py"},
		{"edit_file/old_str absent", map[string]string{"m.py": "A = 1\n"}, []string{"m.py"},
			[]step{{"edit_file", map[string]interface{}{"path": "m.py", "old_str": "ZZZ", "new_str": "Q"}}}, "m.py"},
		{"edit_file/not read first", map[string]string{"m.py": "A = 1\n"}, nil,
			[]step{{"edit_file", map[string]interface{}{"path": "m.py", "old_str": "A = 1", "new_str": "A = 2"}}}, "m.py"},

		{"structural_edit/function body", map[string]string{"m.py": "def f():\n    return 1\n"}, []string{"m.py"},
			[]step{{"structural_edit", map[string]interface{}{"path": "m.py",
				"selector": "function:f", "content": "def f():\n    return 2\n"}}}, "m.py"},
		{"structural_edit/selector missing", map[string]string{"m.py": "def f():\n    return 1\n"}, []string{"m.py"},
			[]step{{"structural_edit", map[string]interface{}{"path": "m.py",
				"selector": "function:nope", "content": "def nope():\n    pass\n"}}}, "m.py"},

		{"insert_after/applied", map[string]string{"m.py": "A = 1\n"}, []string{"m.py"},
			[]step{{"insert_after", map[string]interface{}{"path": "m.py", "line": 1, "content": "B = 2"}}}, "m.py"},
		{"insert_after/line past end", map[string]string{"m.py": "A = 1\n"}, []string{"m.py"},
			[]step{{"insert_after", map[string]interface{}{"path": "m.py", "line": 99, "content": "B = 2"}}}, "m.py"},

		{"replace_lines/applied", map[string]string{"m.py": "A = 1\nB = 2\n"}, []string{"m.py"},
			[]step{{"replace_lines", map[string]interface{}{"path": "m.py", "start_line": 1, "end_line": 1,
				"expected_first_line": "A = 1", "expected_last_line": "A = 1", "content": "A = 9"}}}, "m.py"},
		{"replace_lines/anchor mismatch", map[string]string{"m.py": "A = 1\nB = 2\n"}, []string{"m.py"},
			[]step{{"replace_lines", map[string]interface{}{"path": "m.py", "start_line": 1, "end_line": 1,
				"expected_first_line": "WRONG", "expected_last_line": "WRONG", "content": "A = 9"}}}, "m.py"},

		{"delete_file/applied", map[string]string{"m.py": "A = 1\n"}, nil,
			[]step{{"delete_file", map[string]interface{}{"path": "m.py"}}}, "m.py"},
		{"delete_file/missing", nil, nil,
			[]step{{"delete_file", map[string]interface{}{"path": "gone.py"}}}, "gone.py"},

		{"move_file/applied", map[string]string{"old.py": "A = 1\n"}, nil,
			[]step{{"move_file", map[string]interface{}{"source": "old.py", "destination": "new.py"}}}, "old.py"},
		{"move_file/missing source", nil, nil,
			[]step{{"move_file", map[string]interface{}{"source": "gone.py", "destination": "new.py"}}}, "gone.py"},
	}

	var unknownMutators []string
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			for f, body := range c.seed {
				os.WriteFile(filepath.Join(dir, f), []byte(body), 0o644)
			}
			ctx := ledgerToolCtx(t, dir)
			for _, f := range c.read {
				r, _ := json.Marshal(map[string]string{"path": f})
				executeToolCall("read_file", r, ctx)
			}
			var res *ToolResult
			for _, s := range c.steps {
				args, _ := json.Marshal(s.args)
				res = executeToolCall(s.tool, args, ctx)
			}

			d := ledgerOf(t, ctx, c.path)
			full := filepath.Join(dir, c.path)
			_, statErr := os.Stat(full)

			state := "not recorded"
			if d != nil {
				state = fmt.Sprintf("gen=%d tomb=%v ckpt=%v", d.Generation,
					d.Tombstoned, d.CheckpointHash != "")
			}
			t.Logf("mutation=%-10q validation=%s/%-14q ledger[%s]",
				res.MutationStatus, res.ValidationKind, res.ValidationStatus, state)

			// (3) proven non-mutation records nothing new.
			if res.MutationStatus == MutationNone && d != nil {
				t.Errorf("a branch that mutated nothing created a ledger entry: %+v", d)
			}
			// (4) a producer that has not been migrated yet. Recorded, not
			// judged: this phase reports which branches still guess.
			if res.MutationStatus == MutationUnknown {
				unknownMutators = append(unknownMutators, c.name)
			}
			if d == nil {
				return
			}
			// (1) the hash describes what is actually there.
			if !d.Tombstoned && statErr == nil && d.CurrentHash != diskHash(t, full) {
				t.Errorf("ledger hash does not describe the bytes on disk")
			}
			if d.Tombstoned && statErr == nil {
				t.Error("tombstoned a path that is still on disk")
			}
			// (2) checkpoints come only from an explicit pass.
			if d.CheckpointHash != "" && res.ValidationStatus != ValidationPassed {
				t.Errorf("checkpointed bytes reported as %q, not passed", res.ValidationStatus)
			}
		})
	}
	if len(unknownMutators) > 0 {
		t.Logf("branches still reporting MutationUnknown: %v", unknownMutators)
	}
}

// --- Phase 3A: the ledger through the production agent loop -----------------
//
// Two claims, proved together on one run of runAgentLoop:
//
//	the ledger's current hash describes the bytes a reader finds on disk
//	after the loop returns, including a rewrite performed by a shell
//	command rather than by an edit tool; and
//
//	the externally visible behaviour -- the full ordered event stream and
//	the final workspace -- is byte-for-byte what the parent commit produced.
//
// The second is what makes this phase observational. ATLAS_LEDGER_TRANSCRIPT
// dumps the transcript so the identical fixture can be run on the parent tree
// and diffed; the hash below is that transcript, pinned.
//
// Verified by running this fixture unchanged on the parent commit 5676e49:
// the same 52 events, 3795 bytes, same sha256.
const ledgerLoopTranscriptHash = "c56a9406a0491e6572e6a1427f425d897d0ea21bf93e0e3974f133b36576cb80"

// prompt_tokens is elided for a fixture reason rather than a timing one: the
// workspace path is part of the system prompt and t.TempDir() varies in
// length between runs, moving the count by a token. Every other field,
// including every tool call and result, is compared exactly.
//
// volatileEventKeys are elided from the transcript: they carry wall-clock
// facts that differ between two runs of the same code, so including them
// would make the comparison meaningless rather than strict.
var volatileEventKeys = map[string]bool{
	"ms": true, "duration": true, "duration_ms": true, "elapsed": true,
	"elapsed_ms": true, "timestamp": true, "seconds": true, "wall_s": true,
	"started_at": true, "ended_at": true, "pid": true, "prompt_tokens": true,
}

func stripVolatile(v interface{}) interface{} {
	switch t := v.(type) {
	case map[string]interface{}:
		out := map[string]interface{}{}
		for k, sub := range t {
			if volatileEventKeys[k] || strings.HasSuffix(k, "_ms") {
				continue
			}
			out[k] = stripVolatile(sub)
		}
		return out
	case []interface{}:
		out := make([]interface{}, 0, len(t))
		for _, sub := range t {
			out = append(out, stripVolatile(sub))
		}
		return out
	}
	return v
}

func TestLedgerTracksFinalDiskBytesThroughTheAgentLoop(t *testing.T) {
	dir := t.TempDir()
	rel := "solve.py"

	var turnMu sync.Mutex
	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/v3/"), strings.HasPrefix(r.URL.Path, "/internal/"):
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		case strings.HasSuffix(r.URL.Path, "/syntax-check"):
			json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
			return
		case strings.HasSuffix(r.URL.Path, "/execute"):
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			if strings.Contains(in.Code, ".atlas-mount-probe") {
				b, _ := os.ReadFile(filepath.Join(dir, ".atlas-mount-probe"))
				json.NewEncoder(w).Encode(map[string]interface{}{
					"success": true, "stdout": string(b), "exit_code": 0})
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"success": true, "stdout": "", "exit_code": 0})
			return
		case !strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
			http.NotFound(w, r)
			return
		}
		io.Copy(io.Discard, r.Body)
		turnMu.Lock()
		turn++
		n := turn
		turnMu.Unlock()

		var payload map[string]interface{}
		switch n {
		case 1:
			payload = map[string]interface{}{"type": "tool_call", "name": "write_file",
				"args": map[string]string{"path": rel, "content": "A = 1\n"}}
		case 2:
			payload = map[string]interface{}{"type": "tool_call", "name": "read_file",
				"args": map[string]string{"path": rel}}
		case 3:
			payload = map[string]interface{}{"type": "tool_call", "name": "edit_file",
				"args": map[string]string{"path": rel, "old_str": "A = 1", "new_str": "A = 2"}}
		case 4:
			// The case the ledger exists for: bytes changing outside the
			// edit tools, so the last recorded verdict stops being about
			// what is on disk.
			payload = map[string]interface{}{"type": "tool_call", "name": "run_command",
				"args": map[string]string{"command": "printf 'A = 3\\n' > " + rel}}
		default:
			payload = map[string]interface{}{"type": "done",
				"summary": "wrote solve.py"}
		}
		call, _ := json.Marshal(payload)
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": string(call)}}}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
	}))
	defer srv.Close()

	var evMu sync.Mutex
	var transcript strings.Builder
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.TrustMode = trustFullyTrusted
	ctx.VerifyOnHost = true
	ctx.MaxTurns = 8
	ctx.StreamFn = func(eventType string, data interface{}) {
		raw, _ := json.Marshal(data)
		var generic interface{}
		json.Unmarshal(raw, &generic)
		clean, _ := json.Marshal(stripVolatile(generic))
		evMu.Lock()
		fmt.Fprintf(&transcript, "%s %s\n", eventType, clean)
		evMu.Unlock()
	}

	if err := runAgentLoop(ctx, "Create solve.py."); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}

	full := filepath.Join(dir, rel)
	got, err := os.ReadFile(full)
	if err != nil {
		t.Fatalf("final read: %v", err)
	}
	if string(got) != "A = 3\n" {
		t.Fatalf("final disk bytes = %q, want the command's rewrite", got)
	}

	d := ledgerOf(t, ctx, rel)
	if d == nil {
		t.Fatal("the loop's deliverable was never recorded")
	}
	if d.CurrentHash != hashBytes(got) {
		t.Errorf("ledger hash %s does not describe the final bytes %s",
			d.CurrentHash[:12], hashBytes(got)[:12])
	}
	if _, s := d.CurrentValidation(); s == ValidationPassed {
		t.Error("a shell rewrite inherited the edit tool's verdict")
	}
	for key, entry := range ctx.Ledger {
		if entry.Tombstoned {
			continue
		}
		b, err := os.ReadFile(key)
		if err != nil {
			continue
		}
		if entry.CurrentHash != hashBytes(b) {
			t.Errorf("%s: ledger hash disagrees with disk", key)
		}
	}

	evMu.Lock()
	text := transcript.String()
	evMu.Unlock()
	if out := os.Getenv("ATLAS_LEDGER_TRANSCRIPT"); out != "" {
		os.WriteFile(out, []byte(text), 0o600)
	}
	if h := hashBytes([]byte(text)); ledgerLoopTranscriptHash != "" && h != ledgerLoopTranscriptHash {
		t.Errorf("event stream changed (%s, want %s); the ledger must be "+
			"invisible from outside:\n%s", h[:12], ledgerLoopTranscriptHash[:12], text)
	}
	t.Logf("transcript: %d events, %d bytes, sha256 %s",
		strings.Count(text, "\n"), len(text), hashBytes([]byte(text)))
}

// Recorded, not fixed. SessionWrites is keyed on the raw model-supplied path,
// so two spellings of one file are two entries; the ledger keys on the
// resolved path and is not affected. Pinned so the divergence is visible and
// a later canonicalisation is a deliberate change with its own evidence.
func TestSessionWritesKeyingIsRawWhileTheLedgerIsCanonical(t *testing.T) {
	dir := t.TempDir()
	ctx := ledgerToolCtx(t, dir)
	for _, spelling := range []string{"solve.py", "./solve.py"} {
		args, _ := json.Marshal(map[string]string{"path": spelling, "content": "A = 1\n"})
		if res := executeToolCall("write_file", args, ctx); !res.Success {
			t.Fatalf("write %q failed: %v", spelling, res.Error)
		}
	}
	if len(ctx.SessionWrites) != 2 {
		t.Errorf("SessionWrites keying changed: %v — if this was fixed "+
			"deliberately, update the note on the field", ctx.SessionWrites)
	}
	if len(ctx.Ledger) != 1 {
		t.Errorf("the ledger split one file across %d entries: %v",
			len(ctx.Ledger), ctx.Ledger)
	}
}
