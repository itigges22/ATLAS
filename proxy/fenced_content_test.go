package main

import (
	"encoding/json"
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

func TestRawResponseForFenceRoundTrips(t *testing.T) {
	parsed := ModelResponse{Type: "tool_call", Name: "write_file",
		Args: json.RawMessage(`{"path":"solve.py","content":"@fenced"}`)}
	raw := rawResponseForFence(parsed)
	var back ModelResponse
	if json.Unmarshal([]byte(raw), &back) != nil || back.Name != "write_file" {
		t.Fatalf("round trip failed: %s", raw)
	}
}
