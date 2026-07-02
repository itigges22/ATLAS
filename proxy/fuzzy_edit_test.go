// Tests for the edit_file fallback matchers (findFuzzyLineMatch,
// findActualString) and recoverTruncatedWriteFile. These decide where an
// edit lands when the model's old_str doesn't byte-match the file — the
// silent-wrong-edit failure class, so the fail-safe paths matter as much
// as the matches.

package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFindFuzzyLineMatch(t *testing.T) {
	file := "func main() {\n" +
		"\tx := 1\n" +
		"\tfmt.Println(x)\n" +
		"}\n" +
		"\n" +
		"func helper() {\n" +
		"\tx := 1\n" +
		"}\n"

	t.Run("whitespace drift resolves to the file's own lines", func(t *testing.T) {
		// Model remembered the lines with spaces instead of tabs.
		got, ok := findFuzzyLineMatch(file, "    x := 1\n    fmt.Println(x)")
		if !ok {
			t.Fatal("no match for whitespace-drifted old_str")
		}
		// The returned text must be the FILE's bytes (tabs), so the
		// subsequent strings.Replace hits.
		if got != "\tx := 1\n\tfmt.Println(x)" {
			t.Errorf("matched %q, want the file's original tab-indented lines", got)
		}
		if !strings.Contains(file, got) {
			t.Error("returned match is not a substring of the file")
		}
	})

	t.Run("ambiguous match fails safe", func(t *testing.T) {
		// `x := 1` appears in both functions — editing either would be a
		// guess, and a wrong guess is a silent wrong edit.
		if got, ok := findFuzzyLineMatch(file, "x := 1"); ok {
			t.Errorf("ambiguous single line matched %q, want failure", got)
		}
	})

	t.Run("no match returns false", func(t *testing.T) {
		if _, ok := findFuzzyLineMatch(file, "y := 2"); ok {
			t.Error("nonexistent line reported a match")
		}
	})

	t.Run("all-whitespace target is refused", func(t *testing.T) {
		if _, ok := findFuzzyLineMatch(file, "   \n\t"); ok {
			t.Error("all-whitespace old_str matched — would edit an arbitrary blank region")
		}
	})

	t.Run("trailing newline in old_str is tolerated", func(t *testing.T) {
		got, ok := findFuzzyLineMatch(file, "\tfmt.Println(x)\n}\n")
		if !ok {
			t.Fatal("trailing-newline old_str did not match")
		}
		if got != "\tfmt.Println(x)\n}" {
			t.Errorf("matched %q", got)
		}
	})

	t.Run("empty old_str is refused", func(t *testing.T) {
		if _, ok := findFuzzyLineMatch(file, ""); ok {
			t.Error("empty old_str matched")
		}
	})
}

func TestFindActualString(t *testing.T) {
	t.Run("direct match wins untouched", func(t *testing.T) {
		if got := findActualString(`say "hi"`, `say "hi"`); got != `say "hi"` {
			t.Errorf("got %q", got)
		}
	})

	t.Run("curly quotes in old_str match straight quotes in file", func(t *testing.T) {
		file := `msg := "hello"`
		oldStr := "msg := “hello”" // model emitted curly quotes
		got := findActualString(file, oldStr)
		if got != `msg := "hello"` {
			t.Errorf("got %q, want the straight-quoted form present in the file", got)
		}
	})

	t.Run("straight apostrophe in old_str matches curly apostrophe in file", func(t *testing.T) {
		// The reverse (denormalize) direction is best-effort: straight
		// singles map to the right-single curly — the apostrophe case
		// (prose in docs/comments) is what it exists for.
		file := "// don’t retry on 4xx"
		got := findActualString(file, "// don't retry on 4xx")
		if got != "// don’t retry on 4xx" {
			t.Errorf("got %q, want the file's curly-apostrophe form", got)
		}
	})

	t.Run("no variant matches returns empty", func(t *testing.T) {
		if got := findActualString("abc", "xyz"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestRecoverTruncatedWriteFile(t *testing.T) {
	t.Run("recovers path and unescaped content", func(t *testing.T) {
		partial := `{"type":"tool_call","name":"write_file","args":{"path":"app/main.py","content":"import os\nprint(\"hi\")\n# cut mid-`
		resp, err := recoverTruncatedWriteFile(partial)
		if err != nil {
			t.Fatalf("recovery failed: %v", err)
		}
		if resp.Type != "tool_call" || resp.Name != "write_file" {
			t.Fatalf("recovered envelope %+v", resp)
		}
		var input WriteFileInput
		if err := json.Unmarshal(resp.Args, &input); err != nil {
			t.Fatalf("recovered args do not parse: %v", err)
		}
		if input.Path != "app/main.py" {
			t.Errorf("path = %q", input.Path)
		}
		// JSON escapes must be resolved into real bytes.
		if !strings.Contains(input.Content, "import os\nprint(\"hi\")") {
			t.Errorf("content = %q — escapes not resolved", input.Content)
		}
	})

	t.Run("trailing incomplete escape is trimmed", func(t *testing.T) {
		partial := `{"type":"tool_call","name":"write_file","args":{"path":"a.txt","content":"line\n\`
		resp, err := recoverTruncatedWriteFile(partial)
		if err != nil {
			t.Fatalf("recovery failed on trailing backslash: %v", err)
		}
		var input WriteFileInput
		_ = json.Unmarshal(resp.Args, &input)
		if input.Content != "line\n" {
			t.Errorf("content = %q, want %q", input.Content, "line\n")
		}
	})

	t.Run("missing content field is an error", func(t *testing.T) {
		if _, err := recoverTruncatedWriteFile(`{"type":"tool_call","name":"write_file","args":{"path":"a.txt"`); err == nil {
			t.Error("recovered a write_file with no content field")
		}
	})

	t.Run("missing path is an error", func(t *testing.T) {
		if _, err := recoverTruncatedWriteFile(`{"type":"tool_call","name":"write_file","args":{"content":"body only`); err == nil {
			t.Error("recovered a write_file with no destination path")
		}
	})
}
