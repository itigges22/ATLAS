package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Two distinct filesystem failures through the real write_file dispatch.
// writeFileDirect writes `path + ".atlas.tmp"` then renames it over `path`,
// and returns (nil, err) on either failure -- so no ToolResult is built at
// the mutation site. One is synthesized generically by executeToolCall's
// error branch, which is why both cases arrive at the boundary UNCLASSIFIED.
//
// These fixtures pin present behavior. They deliberately do NOT assert a
// classification: the generic error branch cannot know whether a mutation
// began, so classifying there would be a guess. The correct owner is
// writeFileDirect, which knows which operation failed -- recorded as a
// finding rather than changed here.
func TestWriteFileFilesystemFailures(t *testing.T) {
	t.Run("temp-write-fails", func(t *testing.T) {
		dir := t.TempDir()
		sub := filepath.Join(dir, "ro")
		os.Mkdir(sub, 0o755)
		target := filepath.Join(sub, "f.py")
		prior := "A = 1\n"
		os.WriteFile(target, []byte(prior), 0o644)
		os.Chmod(sub, 0o555) // read-only dir: temp create must fail
		defer os.Chmod(sub, 0o755)

		ctx := NewAgentContext(dir, Tier2Medium)
		ctx.PermissionMode = PermissionYolo
		ctx.StreamFn = func(string, interface{}) {}
		ctx.SessionWrites["ro/f.py"] = true
		args, _ := json.Marshal(map[string]string{"path": "ro/f.py", "content": "B = 2\n"})
		res := executeToolCall("write_file", args, ctx)

		after, _ := os.ReadFile(target)
		ents, _ := os.ReadDir(sub)
		var names []string
		for _, e := range ents {
			names = append(names, e.Name())
		}
		if res.Success {
			t.Fatal("a failed temp write must not report success")
		}
		if !strings.Contains(res.Error, "cannot write") {
			t.Fatalf("failure did not occur at the temp write: %q", res.Error)
		}
		if string(after) != prior {
			t.Fatalf("prior destination bytes changed: %q", string(after))
		}
		for _, n := range names {
			if strings.Contains(n, ".atlas.tmp") {
				t.Errorf("temporary artifact survived: %s", n)
			}
		}
		// Present behavior: unclassified. Pinned so the later fix shows up.
		if res.MutationStatus != MutationUnknown {
			t.Errorf("expected still-unclassified, got %q", res.MutationStatus)
		}
	})

	t.Run("rename-fails", func(t *testing.T) {
		dir := t.TempDir()
		// Destination is a NON-EMPTY DIRECTORY: temp write succeeds, rename fails.
		target := filepath.Join(dir, "f.py")
		os.Mkdir(target, 0o755)
		os.WriteFile(filepath.Join(target, "keep.txt"), []byte("x"), 0o644)

		ctx := NewAgentContext(dir, Tier2Medium)
		ctx.PermissionMode = PermissionYolo
		ctx.StreamFn = func(string, interface{}) {}
		ctx.SessionWrites["f.py"] = true
		args, _ := json.Marshal(map[string]string{"path": "f.py", "content": "B = 2\n"})
		res := executeToolCall("write_file", args, ctx)

		ents, _ := os.ReadDir(dir)
		var names []string
		for _, e := range ents {
			names = append(names, e.Name())
		}
		_, keepErr := os.Stat(filepath.Join(target, "keep.txt"))
		if res.Success {
			t.Fatal("a failed rename must not report success")
		}
		if !strings.Contains(res.Error, "cannot rename") {
			t.Fatalf("failure did not occur at the rename: %q", res.Error)
		}
		if keepErr != nil {
			t.Error("prior destination contents were destroyed")
		}
		for _, n := range names {
			if strings.Contains(n, ".atlas.tmp") {
				t.Errorf("temporary artifact survived a failed rename: %s", n)
			}
		}
		if res.MutationStatus != MutationUnknown {
			t.Errorf("expected still-unclassified, got %q", res.MutationStatus)
		}
	})
}
