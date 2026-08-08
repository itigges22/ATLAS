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
