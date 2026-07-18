package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func newMoveCtx(dir string) *AgentContext {
	return &AgentContext{
		WorkingDir:    dir,
		FilesRead:     map[string]string{},
		FileReadTimes: map[string]time.Time{},
		SessionWrites: map[string]bool{},
	}
}

// The reported failure: reorganizing a flask app by moving index.html into
// templates/. move_file should relocate it into the existing directory keeping
// the basename, content intact.
func TestMoveFileIntoDirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<h1>hi</h1>"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "templates"), 0755); err != nil {
		t.Fatal(err)
	}
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"index.html","destination":"templates/"}`), ctx)
	if err != nil || !res.Success {
		t.Fatalf("move failed: err=%v res=%+v", err, res)
	}
	if _, err := os.Stat(filepath.Join(dir, "index.html")); !os.IsNotExist(err) {
		t.Errorf("source still exists after move")
	}
	data, err := os.ReadFile(filepath.Join(dir, "templates", "index.html"))
	if err != nil || string(data) != "<h1>hi</h1>" {
		t.Errorf("destination missing or content changed: %q err=%v", string(data), err)
	}
}

// Plain rename within the same directory.
func TestMoveFileRename(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "old.py"), []byte("print(1)\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"old.py","destination":"new.py"}`), ctx)
	if err != nil || !res.Success {
		t.Fatalf("rename failed: err=%v res=%+v", err, res)
	}
	if _, err := os.Stat(filepath.Join(dir, "new.py")); err != nil {
		t.Errorf("renamed file missing: %v", err)
	}
}

// A move must never silently clobber an existing destination file.
func TestMoveFileRefusesClobber(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("A"), 0644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("B"), 0644)
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"a.txt","destination":"b.txt"}`), ctx)
	if err != nil {
		t.Fatalf("unexpected hard error: %v", err)
	}
	if res.Success {
		t.Errorf("expected refusal to clobber existing destination")
	}
	if data, _ := os.ReadFile(filepath.Join(dir, "b.txt")); string(data) != "B" {
		t.Errorf("destination was overwritten: %q", string(data))
	}
}

// Missing source is a clean tool error, not a crash.
func TestMoveFileSourceMissing(t *testing.T) {
	dir := t.TempDir()
	ctx := newMoveCtx(dir)
	res, err := moveFileTool().Execute(json.RawMessage(`{"source":"nope.py","destination":"x.py"}`), ctx)
	if err != nil {
		t.Fatalf("unexpected hard error: %v", err)
	}
	if res.Success {
		t.Errorf("expected failure for missing source")
	}
}
