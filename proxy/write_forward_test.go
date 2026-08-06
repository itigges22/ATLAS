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
