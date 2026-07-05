package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// May 9 2026: under BiasBusters mitigations the model now reaches for
// ast_edit + edit_file too. Real flask test logs show 10K-12K char
// ast_edit responses parse-erroring with no recovery path. Lock the
// generalized recovery so future regressions can't slip back in.

func TestRecoverTruncatedAstEditFullPayload(t *testing.T) {
	// Well-formed but unparseable-as-JSON payload (e.g. trailing brace
	// dropped by the model). Recovery still extracts the fields.
	partial := `{"type":"tool_call","name":"ast_edit","args":{"path":"templates/index.html","selector":"<html>","content":"<!DOCTYPE html>\n<html lang=\"en\"><head></head><body>hi</body></html>"`
	resp, ok := recoverTruncatedToolCall(partial)
	if !ok {
		t.Fatal("recovery returned false")
	}
	if resp.Type != "tool_call" || resp.Name != "ast_edit" {
		t.Fatalf("got Type=%q Name=%q, want tool_call/ast_edit", resp.Type, resp.Name)
	}
	var args AstEditInput
	if err := json.Unmarshal(resp.Args, &args); err != nil {
		t.Fatalf("unmarshal recovered args: %v", err)
	}
	if args.Path != "templates/index.html" {
		t.Errorf("Path = %q, want templates/index.html", args.Path)
	}
	if args.Selector != "<html>" {
		t.Errorf("Selector = %q, want <html>", args.Selector)
	}
	if !strings.HasPrefix(args.Content, "<!DOCTYPE html>") {
		preview := args.Content
		if len(preview) > 20 {
			preview = preview[:20]
		}
		t.Errorf("Content prefix = %q, want <!DOCTYPE html>", preview)
	}
	if !strings.Contains(args.Content, `lang="en"`) {
		t.Errorf("Content missing unescaped lang=\"en\": %q", args.Content)
	}
}

func TestRecoverTruncatedAstEditMidContent(t *testing.T) {
	// Realistic case from May 9 logs: response cut off mid-content with
	// no closing quote/braces. Recovery returns whatever content made
	// it through so the agent can write SOMETHING useful and continue.
	partial := `{"type":"tool_call","name":"ast_edit","args":{"path":"app.py","selector":"function:dashboard","content":"@app.route('/dashboard')\ndef dashboard():\n    users = get_users()\n    return render_template(`
	resp, ok := recoverTruncatedToolCall(partial)
	if !ok {
		t.Fatal("recovery returned false on mid-content truncation")
	}
	var args AstEditInput
	if err := json.Unmarshal(resp.Args, &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if args.Path != "app.py" || args.Selector != "function:dashboard" {
		t.Errorf("path/selector wrong: %+v", args)
	}
	if !strings.Contains(args.Content, "def dashboard()") {
		t.Errorf("content missing def dashboard: %q", args.Content)
	}
}

func TestRecoverTruncatedEditFileBothFields(t *testing.T) {
	partial := `{"type":"tool_call","name":"edit_file","args":{"path":"app.py","old_str":"return None","new_str":"return {}","replace_all":false}`
	resp, ok := recoverTruncatedToolCall(partial)
	if !ok {
		t.Fatal("recovery returned false")
	}
	if resp.Name != "edit_file" {
		t.Errorf("Name = %q", resp.Name)
	}
	var args EditFileInput
	if err := json.Unmarshal(resp.Args, &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if args.Path != "app.py" || args.OldStr != "return None" || args.NewStr != "return {}" {
		t.Errorf("field recovery wrong: %+v", args)
	}
}

func TestRecoverTruncatedEditFileMidNewStr(t *testing.T) {
	// Truncated mid new_str — should still recover with what we have.
	partial := `{"type":"tool_call","name":"edit_file","args":{"path":"app.py","old_str":"return None","new_str":"return {\\"users\\":`
	resp, ok := recoverTruncatedToolCall(partial)
	if !ok {
		t.Fatal("recovery returned false")
	}
	var args EditFileInput
	_ = json.Unmarshal(resp.Args, &args)
	if args.OldStr != "return None" {
		t.Errorf("OldStr = %q", args.OldStr)
	}
	if !strings.HasPrefix(args.NewStr, "return ") {
		t.Errorf("NewStr should start with 'return ', got %q", args.NewStr)
	}
}

func TestRecoverTruncatedToolCallUnknownToolReturnsFalse(t *testing.T) {
	// Tool we don't have a recovery for → return false so caller falls
	// through to the diagnostic error (not silent failure).
	partial := `{"type":"tool_call","name":"read_file","args":{"path":"app.py"`
	if _, ok := recoverTruncatedToolCall(partial); ok {
		t.Error("expected no recovery for read_file")
	}
}

func TestRecoverTruncatedAstEditMissingSelectorFails(t *testing.T) {
	// Malformed — selector missing entirely. Recovery should fail
	// rather than emit a tool call with empty selector that ast_edit
	// would reject downstream anyway.
	partial := `{"type":"tool_call","name":"ast_edit","args":{"path":"app.py","content":"def foo(): pass"}`
	if _, ok := recoverTruncatedToolCall(partial); ok {
		t.Error("expected no recovery when selector is missing")
	}
}

// Locks the new diagnostic behavior: when the brace-balanced parse
// fails, extractModelResponse must surface the actual unmarshal error
// so logs tell us WHY ("invalid character '\\n'" vs "unexpected end")
// instead of a generic "could not parse JSON".
func TestExtractModelResponseSurfacesUnmarshalError(t *testing.T) {
	// Brace-balanced JSON with a literal LF inside a string — invalid
	// per RFC 8259 and the kind of failure we used to swallow.
	raw := "{\"type\":\"tool_call\",\"name\":\"read_file\",\"args\":{\"path\":\"a\nb\"}}"
	_, err := extractModelResponse(raw)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "could not parse JSON from response") {
		t.Errorf("error missing canonical prefix: %v", err)
	}
	// Wrapped error must carry the underlying json error (anything
	// containing 'invalid character' or 'unexpected' is acceptable —
	// we just need SOME signal beyond the canonical prefix).
	inner := err.Error()
	if !strings.Contains(inner, "invalid character") && !strings.Contains(inner, "unexpected") {
		t.Errorf("error missing inner json detail: %v", err)
	}
}
