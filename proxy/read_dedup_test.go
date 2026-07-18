package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// storedReadMessage mimics how the agent loop records a read_file result in the
// conversation: ToolResult.Data = marshaled ReadFileOutput, then the message
// content = ToolResult.MarshalText() (json.Marshal of the whole result). This
// is where the file content gets JSON-escaped.
func storedReadMessage(t *testing.T, fileContent string) AgentMessage {
	t.Helper()
	// read_file numbers lines; the numbering doesn't affect the probe, but
	// model the real shape anyway.
	var numbered strings.Builder
	for i, l := range strings.Split(fileContent, "\n") {
		numbered.WriteString(strings.TrimRight(
			strings.Join([]string{itoaTest(i + 1), l}, "\t"), "\n"))
		numbered.WriteString("\n")
	}
	out := ReadFileOutput{Content: numbered.String()}
	data, _ := json.Marshal(out)
	res := &ToolResult{Success: true, Data: data}
	return AgentMessage{Role: "tool", ToolName: "read_file", Content: res.MarshalText()}
}

func itoaTest(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

// Regression: a flask app whose longest line is embedded HTML/JS full of double
// quotes must still be detected as present in context. The old longest-raw-line
// probe failed here (the `"` in the line became `\"` in the stored JSON), so
// the dedup re-served the file every read and the model looped on read_file.
func TestFileContentInContextSurvivesJSONEscaping(t *testing.T) {
	fileContent := `from flask import Flask, render_template_string
app = Flask(__name__)
HTML = "<div class=\"board\" style=\"width:400px;height:400px\" id=\"game-board-container\"></div>"

@app.route("/")
def index():
    return render_template_string(HTML)
`
	ctx := &AgentContext{Messages: []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "make a snake game"},
		storedReadMessage(t, fileContent),
	}}
	if !fileContentInContext(ctx, fileContent) {
		t.Errorf("content with quoted HTML line reported as NOT in context — false-negative would make dedup re-serve and loop")
	}
}

// When the content really is gone (trimmed), it must report absent so the
// dedup re-serves rather than lying that "it's above."
func TestFileContentInContextDetectsAbsence(t *testing.T) {
	fileContent := "def compute_subtotal(rows):\n    return sum(r.price for r in rows)\n"
	ctx := &AgentContext{Messages: []AgentMessage{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "unrelated chatter with no file content"},
	}}
	if fileContentInContext(ctx, fileContent) {
		t.Errorf("expected absent verdict when content is not in any message")
	}
}
