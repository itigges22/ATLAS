package main

import "testing"

// Reproduces the May 8 categorizer miss: a response starting with
// `{"type":"tool_call"` and containing &lt;/&gt; entities was being
// categorized as `malformed_tool` even though the html_entities
// branch should fire first. Locks the expected behavior so the bug
// can't regress silently.
func TestCategorizeParseFailureHtmlEntitiesShape(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			"tool_call envelope with HTML-entity-encoded args",
			`{"type":"tool_call","name":"edit_file","args":{"path":"x.html","old_str":"&lt;!DOCTYPE html&gt;\n&lt;html&gt;\n","new_str":"&lt;!DOCTYPE html&gt;\n&lt;html&gt;\n"}}`,
			"html_entities",
		},
		{
			"prose preamble + tool_call + entities",
			"Now I can see the existing dashboard.html content. I'll use edit_file to replace the entire file content with the new dashboard template.\n\n" +
				`{"type":"tool_call","name":"edit_file","args":{"path":"x.html","old_str":"&lt;!DOCTYPE html&gt;","new_str":"&lt;!DOCTYPE html&gt;"}}`,
			"html_entities",
		},
		{
			"tool_call envelope truncated, no entities",
			`{"type":"tool_call","name":"edit_file","args":{"path":"x.py","old_str":"def foo():\n    return 1","new_str":"def foo():\n    return`,
			"truncated_tool",
		},
		{
			"tool_call envelope, malformed JSON, no entities",
			`{"type":"tool_call","name":"edit_file","args":{"path":"x.py","old_str":"def foo():","new_str":"def bar(): }}`,
			"malformed_tool",
		},
		{
			"prose narration only",
			`Now let me read the index.html template.`,
			"prose",
		},
		{
			"empty",
			"",
			"empty",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := classifyParseFailure(tc.raw)
			if got != tc.want {
				t.Errorf("classifyParseFailure() category = %q, want %q\nraw: %q", got, tc.want, tc.raw)
			}
		})
	}
}
