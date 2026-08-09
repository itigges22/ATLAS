package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Code emitted inside a JSON string pays escaping pressure on every dense
// line, and the served model measurably cannot sustain it: the same debounce
// solution parses 6/6 emitted in a fenced block and 0/6 emitted as a JSON
// string. "@fenced" routes the file body around the JSON channel via one
// unconstrained sub-call.

func TestFencedContentRegexExtractsTheBlock(t *testing.T) {
	reply := "Here is the file:\n```python\nx = 1\nprint(x)\n```\nDone."
	m := fencedContentRe.FindStringSubmatch(reply)
	if m == nil || m[1] != "x = 1\nprint(x)\n" {
		t.Fatalf("extraction failed: %#v", m)
	}
}

func TestFencedContentRegexHandlesBareFence(t *testing.T) {
	reply := "```\ny = 2\n```"
	m := fencedContentRe.FindStringSubmatch(reply)
	if m == nil || m[1] != "y = 2\n" {
		t.Fatalf("extraction failed: %#v", m)
	}
}

// stubInference serves /v1/chat/completions with the given replies in order,
// each stamped with tokensPer usage tokens. Other paths (the /slots prompt
// poller) get 404s and are not counted as calls.
func stubInference(t *testing.T, replies []string, tokensPer int, calls *int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/chat/completions") {
			http.NotFound(w, r)
			return
		}
		i := *calls
		*calls++
		if i >= len(replies) {
			i = len(replies) - 1
		}
		w.Header().Set("Content-Type", "text/event-stream")
		delta, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": replies[i]}}},
		})
		usage, _ := json.Marshal(map[string]interface{}{
			"choices": []interface{}{},
			"usage":   map[string]int{"total_tokens": tokensPer},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: %s\n\ndata: [DONE]\n\n", delta, usage)
	}))
}

// Every fenced sub-call attempt is a real generation; the run totals must
// carry each one (third-party audit finding: sub-call spend was invisible).
func TestFencedFetchAccountsOneAttempt(t *testing.T) {
	calls := 0
	srv := stubInference(t, []string{"```python\nx = 1\n```"}, 100, &calls)
	defer srv.Close()
	t.Setenv("ATLAS_LLAMA_URL", srv.URL)

	ctx := &AgentContext{}
	got, err := fetchFencedContent(ctx, `{"type":"tool_call"}`, "a.py")
	if err != nil || got != "x = 1\n" {
		t.Fatalf("fetch failed: %q %v", got, err)
	}
	if calls != 1 || ctx.FencedCalls != 1 || ctx.FencedTokens != 100 || ctx.TotalTokens != 100 {
		t.Fatalf("one attempt must account one call and its tokens: calls=%d fenced=%d/%d total=%d",
			calls, ctx.FencedCalls, ctx.FencedTokens, ctx.TotalTokens)
	}
}

func TestFencedFetchAccountsFailedFirstAttempt(t *testing.T) {
	calls := 0
	srv := stubInference(t, []string{
		"no fence here, sorry",
		"```python\ny = 2\n```",
	}, 100, &calls)
	defer srv.Close()
	t.Setenv("ATLAS_LLAMA_URL", srv.URL)

	ctx := &AgentContext{}
	got, err := fetchFencedContent(ctx, `{"type":"tool_call"}`, "a.py")
	if err != nil || got != "y = 2\n" {
		t.Fatalf("fetch failed: %q %v", got, err)
	}
	if calls != 2 || ctx.FencedCalls != 2 || ctx.FencedTokens != 200 || ctx.TotalTokens != 200 {
		t.Fatalf("a failed first attempt is still a generation: calls=%d fenced=%d/%d total=%d",
			calls, ctx.FencedCalls, ctx.FencedTokens, ctx.TotalTokens)
	}
}

func TestRawResponseForFenceRoundTrips(t *testing.T) {
	parsed := ModelResponse{Type: "tool_call", Name: "write_file",
		Args: json.RawMessage(`{"path":"solve.py","content":"@fenced"}`)}
	raw := rawResponseForFence(parsed)
	var back ModelResponse
	if json.Unmarshal([]byte(raw), &back) != nil || back.Name != "write_file" {
		t.Fatalf("round trip failed: %s", raw)
	}
}

// A file that itself contains ``` (markdown, docstring examples) must not
// be cut at its first interior fence; the trailing-anchored form wins.
func TestFencedExtractionSurvivesInteriorFences(t *testing.T) {
	body := "# readme\n\n```\nexample\n```\n\ntail line"
	reply := "```markdown\n" + body + "\n```"
	got := extractFencedContent(reply)
	if got != body+"\n" {
		t.Fatalf("interior fence cut the file: %q", got)
	}
}

func TestFencedExtractionAcceptsPunctuationTags(t *testing.T) {
	for _, tag := range []string{"c++", "c#", "objective-c"} {
		reply := "```" + tag + "\nint x;\n```"
		if got := extractFencedContent(reply); got == "" {
			t.Errorf("tag %q failed to open a block", tag)
		}
	}
}

func TestFenceTagFollowsExtension(t *testing.T) {
	if got := fenceTagForPath("app.html"); got != "html" {
		t.Errorf("html tag: %q", got)
	}
	if got := fenceTagForPath("main.go"); got != "go" {
		t.Errorf("go tag: %q", got)
	}
	if got := fenceTagForPath("solve.py"); got != "python" {
		t.Errorf("python tag: %q", got)
	}
}

// A file's own leading blank lines are the file's content, not part of the
// fence line. Found by fuzzing: the tag was followed by `\s*\n`, and \s
// matches newlines, so a greedy match ate every blank line at the top of the
// file before the capture began — silent mutation in the channel that
// carries every whole file ATLAS writes.
func TestFencedExtractionKeepsLeadingBlankLines(t *testing.T) {
	// Whitespace-only content is out of contract by design: an empty block
	// is what makes the sub-call ask again, so it is not a case here.
	for _, content := range []string{"\nx = 1\n", "\n\n\nx = 1\n", "\n\ndef f():\n    pass\n"} {
		reply := "```python\n" + content + "```"
		if got := extractFencedContent(reply); got != content {
			t.Errorf("leading blank lines dropped\n want=%q\n  got=%q", content, got)
		}
	}
}

// A CRLF reply must not leave the \r glued to the fence line or lose it
// from the content.
func TestFencedExtractionHandlesCRLFFenceLine(t *testing.T) {
	if got := extractFencedContent("```python\r\nx = 1\n```"); got != "x = 1\n" {
		t.Errorf("CRLF fence line mishandled: %q", got)
	}
}

// A truncated inline body must never reach disk.
//
// Found by the session hunt: the model emitted
// `"content": "@fenced\n```python\nprint("` — it began inlining the file and
// the JSON string was cut off mid-expression, which is the exact failure
// @fenced exists to avoid. The old strip declared "content arrived inline",
// the sanitizer then removed the fence line, and a one-line `print(` landed
// on disk. Six of twenty create sessions shipped an unparseable file that
// way, each repeating the identical write for five more turns.
func TestInlineFencedBodyDetectsTruncation(t *testing.T) {
	cases := []struct {
		name     string
		inline   string
		wantBody string // "" means: must fall back to the sub-call
	}{
		{"truncated fence", "```python\nprint(", ""},
		{"opener only", "```python", ""},
		{"complete fence", "```python\nprint(\"hi\")\n```", "print(\"hi\")\n"},
		{"bare complete body", "print(\"hi\")\n", "print(\"hi\")\n"},
	}
	for _, c := range cases {
		got := extractFencedContent(c.inline)
		if got == "" && strings.Contains(c.inline, "```") {
			got = "" // truncated: the caller falls back to the sub-call
		} else if got == "" {
			got = c.inline // bare body, no fence involved
		}
		if got != c.wantBody {
			t.Errorf("%s: resolved %q, want %q", c.name, got, c.wantBody)
		}
	}
}

// A sub-call that ran to the token ceiling without closing its fence must
// not be retried. Found in the session hunt: six sessions spent ~650s each
// on two full-budget attempts that failed identically, and the model
// recovered on the NEXT turn anyway once the bounce put it back on the tool
// call. The retry bought nothing and cost five minutes.
func TestFencedFetchDoesNotRetryAfterTokenCeiling(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1/chat/completions") {
			http.NotFound(w, r)
			return
		}
		calls++
		w.Header().Set("Content-Type", "text/event-stream")
		// An opened fence that never closes, cut off by max_tokens.
		delta, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": "```python\nprint("},
					"finish_reason": "length"}},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", delta)
	}))
	defer srv.Close()
	t.Setenv("ATLAS_LLAMA_URL", srv.URL)

	ctx := &AgentContext{}
	if _, err := fetchFencedContent(ctx, `{"type":"tool_call"}`, "a.py"); err == nil {
		t.Fatal("expected an error when the fence never closed")
	}
	if calls != 1 {
		t.Fatalf("a ceiling-truncated attempt must not be retried: %d calls", calls)
	}
}
