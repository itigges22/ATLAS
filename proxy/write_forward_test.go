package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A syntax gate on a NEW file substitutes the harness's judgement for
// execution's, and forbids the one loop that measurably works: write it, run
// it, read the real traceback, fix it. The no-tool baseline arm, whose only
// feedback IS the traceback, resolves its own syntax errors at 85-100%.
// Under the rejection the model retried blind, resent byte-identical
// content, and the repetition breaker ended the session: three AoC sessions
// and a novel-arm session all finished as "solve.py was never created".
//
// On an EXISTING file the gate stays: it stops working code being clobbered.

func TestBrokenContentToANewFileLandsWithAWarning(t *testing.T) {
	root := t.TempDir()
	ctx := &AgentContext{WorkingDir: root, SessionWrites: map[string]bool{}, BodySeen: map[string]bool{}}
	path := filepath.Join(root, "solve.py")
	res, err := writeNewFileWithWarning(path, "solve.py", "def f(:\n", "SyntaxError: invalid syntax (line 1)", ctx)
	if err != nil || res == nil || !res.Success {
		t.Fatalf("the write must land: res=%+v err=%v", res, err)
	}
	if _, statErr := os.Stat(path); statErr != nil {
		t.Fatalf("file must be on disk: %v", statErr)
	}
	var out WriteFileOutput
	if json.Unmarshal(res.Data, &out) != nil || out.Warning == "" {
		t.Fatalf("result must carry the warning: %s", string(res.Data))
	}
	for _, want := range []string{"does not parse", "Run it", "traceback"} {
		if !strings.Contains(out.Warning, want) {
			t.Errorf("warning should say %q: %s", want, out.Warning)
		}
	}
	// The carveout that lets the model fix its own file reads SessionWrites
	// by the path AS THE MODEL SENT IT, not the resolved one.
	if !ctx.SessionWrites["solve.py"] {
		t.Error("a landed write is the agent's own file and must be iterable")
	}
}

// The strict syntax gate on existing files protects WORKING code. When the
// file on disk is itself unparseable, rejecting an imperfect fix guarantees
// the broken version survives: measured twice on the novel benchmark, a
// broken first draft landed, the corrective write carried a new syntax slip,
// and the model re-sent identical content until the repetition breaker ended
// the session with the original stump still on disk.
//
// Both tests need T2-sized content (def/loop/if) or classifyFileTier routes
// them down the ungated T1 path and the gate under test never runs.

func t2Body(marker string) string {
	return "from chunk import chunks\n\n\ndef main():\n    data = [1, 2, " + marker + ", 4, 5]\n" +
		"    size = 2\n    result = chunks(data, size)\n    print(result)\n" +
		"    for row in result:\n        if len(row) != size:\n            print('short')\n" +
		"        else:\n            print('full')\n    return result\n\n\n" +
		"if __name__ == '__main__':\n    main()\n"
}

func TestARepairOfABrokenFileLandsEvenWhenImperfect(t *testing.T) {
	dir := t.TempDir()
	sb := fakeSyntaxSandbox(t, "**") // markdown bold marks content invalid
	defer sb.Close()
	// V3URL set so the T2 pre-V3 gate is the code under test; the gate
	// decides before any V3 request, so nothing ever calls this address.
	ctx := writeGateCtx(t, "http://127.0.0.1:1", sb.URL, dir)

	// A broken file is already on disk (landed via fail-forward earlier).
	if err := os.WriteFile(filepath.Join(dir, "s.py"), []byte(t2Body("**1**")), 0o644); err != nil {
		t.Fatal(err)
	}
	// The repair attempt is also imperfect — it must still land.
	repair := t2Body("**2**")
	args, _ := json.Marshal(map[string]string{"path": "s.py", "content": repair})
	res, err := writeFileTool().Execute(json.RawMessage(args), ctx)
	if err != nil {
		t.Fatalf("write_file: %v", err)
	}
	if res == nil || !res.Success {
		t.Fatalf("an imperfect repair of a broken file must land, got %+v", res)
	}
	got, _ := os.ReadFile(filepath.Join(dir, "s.py"))
	if string(got) != repair {
		t.Errorf("the repair attempt must be what is on disk, got %q", got)
	}
}

func TestAWorkingFileIsStillProtectedFromBrokenContent(t *testing.T) {
	dir := t.TempDir()
	sb := fakeSyntaxSandbox(t, "**")
	defer sb.Close()
	ctx := writeGateCtx(t, "http://127.0.0.1:1", sb.URL, dir)

	healthy := t2Body("3")
	if err := os.WriteFile(filepath.Join(dir, "s.py"), []byte(healthy), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx.RecordFileRead(filepath.Join(dir, "s.py"), healthy)
	args, _ := json.Marshal(map[string]string{"path": "s.py", "content": t2Body("**3**")})
	res, _ := writeFileTool().Execute(json.RawMessage(args), ctx)
	if res != nil && res.Success {
		t.Error("working code must not be clobbered with broken content")
	}
	got, _ := os.ReadFile(filepath.Join(dir, "s.py"))
	if string(got) != healthy {
		t.Errorf("the healthy file must survive, got %q", got)
	}
}

// Verification is of bytes, not of a moment. The session-level booleans could
// say "verified" while the file on disk was a later, never-executed rewrite —
// the shared root of the verify-then-modify and warned-write holes (audit
// finding: evidence must be a contract tied to the final artifact).
func TestDriftAfterVerificationIsNamed(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir, SessionWrites: map[string]bool{"solve.py": true}}
	if err := os.WriteFile(filepath.Join(dir, "solve.py"), []byte("print(1)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := sessionWriteHashes(ctx)
	if len(snap) != 1 {
		t.Fatalf("snapshot should cover the written file: %v", snap)
	}
	if got := driftedSinceVerification(ctx, snap); got != "" {
		t.Fatalf("unchanged bytes are not drift: %q", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "solve.py"), []byte("print(2)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := driftedSinceVerification(ctx, snap); got != "solve.py" {
		t.Fatalf("rewritten bytes must be drift, got %q", got)
	}
}

func TestAFileWrittenAfterTheSnapshotIsDrift(t *testing.T) {
	dir := t.TempDir()
	ctx := &AgentContext{WorkingDir: dir, SessionWrites: map[string]bool{"solve.py": true}}
	if err := os.WriteFile(filepath.Join(dir, "solve.py"), []byte("print(1)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Verified before this file was ever written: absent from the snapshot.
	if got := driftedSinceVerification(ctx, map[string]string{}); got != "solve.py" {
		t.Fatalf("a never-verified artifact is drift by definition, got %q", got)
	}
}

// A warned landing is artifact state, not a rewrite throttle (audit
// correction). The mark must survive everything short of an actual
// execution attempt of the file — naming it (cat/grep/ls) proves nothing
// about runtime behavior and must not discharge it.
func TestExecutionAttemptDischarge(t *testing.T) {
	cases := []struct {
		cmd  string
		path string
		want bool
	}{
		{"python3 solve.py", "solve.py", true},
		{"python solve.py < input.txt", "solve.py", true},
		{"timeout 10 python3 solve.py", "solve.py", true},
		{"python3 -u solve.py", "solve.py", true},
		{"./solve.py", "solve.py", true},
		{"cd /w && python3 ./solve.py", "solve.py", true},
		{"python3 sub/solve.py", "sub/solve.py", true},
		{"pytest test_solve.py", "test_solve.py", true},
		{"node app.js", "app.js", true},
		{"bash run.sh", "run.sh", true},
		// Naming is not running.
		{"cat solve.py", "solve.py", false},
		{"grep main solve.py", "solve.py", false},
		{"ls solve.py", "solve.py", false},
		{"wc -l solve.py", "solve.py", false},
		{"echo solve.py", "solve.py", false},
		{"ls", "solve.py", false},
		{"python3 other.py", "solve.py", false},
		// Execution in one chain segment doesn't bless a file only
		// named in another.
		{"cat solve.py && python3 other.py", "solve.py", false},
		{"python3 solve.py && cat notes.txt", "solve.py", true},
	}
	for _, c := range cases {
		if got := executionAttempt(c.cmd, c.path); got != c.want {
			t.Errorf("executionAttempt(%q, %q) = %v, want %v", c.cmd, c.path, got, c.want)
		}
	}
}
