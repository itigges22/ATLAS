package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func TestFormatDemoModelLabel(t *testing.T) {
	tests := map[string]string{
		"orion-code-10b-it-Q4_K_M":                "Orion code 10B",
		"nova-7B-Q6_K":                            "Nova 7B",
		"/models/atlas-test-3.1-8B-Instruct.gguf": "Atlas test 3.1 8B",
		"": demoModelFallback,
	}
	for input, want := range tests {
		t.Run(input, func(t *testing.T) {
			if got := formatDemoModelLabel(input); got != want {
				t.Fatalf("formatDemoModelLabel(%q) = %q, want %q", input, got, want)
			}
		})
	}
}

func TestFetchDemoModelIdentity(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		wantID string
		want   string
	}{
		{"configured model", http.StatusOK, `{"object":"list","data":[{"id":"orion-code-10b-it-Q4_K_M"}]}`, "orion-code-10b-it-Q4_K_M", "Orion code 10B"},
		{"empty list", http.StatusOK, `{"object":"list","data":[]}`, "", demoModelFallback},
		{"missing id", http.StatusOK, `{"object":"list","data":[{"id":""}]}`, "", demoModelFallback},
		{"malformed", http.StatusOK, `{`, "", demoModelFallback},
		{"upstream failure", http.StatusServiceUnavailable, `unavailable`, "", demoModelFallback},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/models" {
					t.Errorf("path = %q, want /v1/models", r.URL.Path)
				}
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()
			id, label := fetchDemoModelIdentity(srv.URL)
			if id != tc.wantID || label != tc.want {
				t.Fatalf("identity = (%q, %q), want (%q, %q)", id, label, tc.wantID, tc.want)
			}
		})
	}
}

func TestProxySupportsRawDemoRequiresCapability(t *testing.T) {
	tests := []struct {
		name string
		body string
		want bool
	}{
		{"current proxy", `{"capabilities":["demo_raw_completion_v1"]}`, true},
		{"old proxy", `{"status":"ok"}`, false},
		{"malformed", `{`, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/health" {
					t.Fatalf("path = %q, want /health", r.URL.Path)
				}
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()
			if got := proxySupportsRawDemo(srv.URL); got != tc.want {
				t.Fatalf("proxySupportsRawDemo = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestRunDemoRejectsOldProxyBeforeLaunching(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer srv.Close()

	err := runDemo(srv.URL, t.TempDir(), "short")
	if err == nil || !strings.Contains(err.Error(), "too old") {
		t.Fatalf("runDemo error = %v, want stale-proxy rejection", err)
	}
}

func TestDemoRawStreamNeverEntersAgentEndpoint(t *testing.T) {
	rawCalls, agentCalls := 0, 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			rawCalls++
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"raw answer\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		case "/v1/agent":
			agentCalls++
			http.Error(w, "raw side entered agent", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	m := &demoModel{
		proxyURL: srv.URL,
		modelID:  "configured-model",
		prompt:   demoPrompt{Prompt: "build it"},
		events:   make(chan demoEvent, 16),
		ctx:      context.Background(),
	}
	m.runStream("raw")
	if rawCalls != 1 || agentCalls != 0 {
		t.Fatalf("raw calls = %d, agent calls = %d", rawCalls, agentCalls)
	}
}

func TestDemoLabelsRawCompletionAsModelNotAgent(t *testing.T) {
	rawChild := newTUIModel("http://unused")
	v3Child := newTUIModel("http://unused")
	payload, _ := json.Marshal(map[string]string{"content": "raw answer"})
	m := &demoModel{
		rawChild: &rawChild,
		v3Child:  &v3Child,
		events:   make(chan demoEvent, 1),
		ctx:      context.Background(),
	}
	_, _ = m.Update(demoBatchMsg{stream: []demoStreamMsg{{
		side: "raw",
		evt:  chatEvent{Type: "text", Data: payload},
	}}})
	if len(m.rawChild.chat) != 1 || m.rawChild.chat[0].Meta != "raw model" {
		t.Fatalf("raw chat rows = %#v", m.rawChild.chat)
	}
}

func TestDemoTitlesDescribeActualComparison(t *testing.T) {
	m := &demoModel{modelLabel: "Orion code 10B"}
	left := m.rawTitle()
	right := m.atlasTitle()
	for _, title := range []string{left, right} {
		if !strings.Contains(title, "Orion code 10B") {
			t.Fatalf("title %q does not identify configured model", title)
		}
		if strings.Contains(title, "RAW 9B") || strings.Contains(title, "no V3 orchestration") {
			t.Fatalf("title retains misleading legacy wording: %q", title)
		}
	}
	if !strings.Contains(left, "RAW MODEL") || !strings.Contains(left, "NO ORCHESTRATION") {
		t.Fatalf("baseline title is not explicit about comparison: %q", left)
	}
	if !strings.Contains(right, "ATLAS V3") {
		t.Fatalf("V3 title is not explicit: %q", right)
	}
}

func TestDemoTitleFallsBackWithoutMetadata(t *testing.T) {
	m := &demoModel{}
	if got := m.rawTitle(); !strings.HasPrefix(got, demoModelFallback) {
		t.Fatalf("baseline title = %q, want neutral fallback", got)
	}
}

func TestDemoLiveAndOutputViewsUseResolvedTitles(t *testing.T) {
	rawChild := newTUIModel("http://unused")
	v3Child := newTUIModel("http://unused")
	m := &demoModel{
		modelLabel:  "Orion code 10B",
		prompt:      demoPrompt{Prompt: "build it"},
		promptShown: len("build it"),
		width:       180,
		height:      36,
		rawChild:    &rawChild,
		v3Child:     &v3Child,
		v3Sandbox:   ".demo-v3-test",
		activePane:  "v3",
	}

	assertTitles := func(name, view string) {
		t.Helper()
		for _, want := range []string{"Orion code 10B", "RAW MODEL", "NO ORCHESTRATION", "ATLAS V3"} {
			if !strings.Contains(view, want) {
				t.Fatalf("%s view missing %q", name, want)
			}
		}
		for _, stale := range []string{"RAW 9B", "no V3 orchestration"} {
			if strings.Contains(view, stale) {
				t.Fatalf("%s view contains stale label %q", name, stale)
			}
		}
	}

	assertTitles("live", m.View())
	m.outputMode = true
	m.rawChild.chat = append(m.rawChild.chat, chatMessage{
		Role: roleAssistant, Meta: "raw model", Body: "raw response survives",
	})
	assertTitles("output", m.View())
	if output := m.View(); !strings.Contains(output, "raw response") || !strings.Contains(output, "survives") {
		t.Fatal("output view discarded the raw model response")
	}
	if output := m.View(); !strings.Contains(output, "raw model") {
		t.Fatal("raw response is still labeled as an agent response")
	}
}

// The done marker must trail every buffered stream event — it's sent
// by the forwarding loop only after the stream channel closes.
func TestDemoStreamDoneArrivesAfterAllEvents(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 20; i++ {
			_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok%d\"}}]}\n\n", i)
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	m := &demoModel{
		proxyURL: srv.URL,
		modelID:  "m",
		prompt:   demoPrompt{Prompt: "p"},
		events:   make(chan demoEvent, 256),
		ctx:      context.Background(),
	}
	m.runStream("raw")

	var seen []demoEvent
drain:
	for {
		select {
		case ev := <-m.events:
			seen = append(seen, ev)
		default:
			break drain
		}
	}
	if len(seen) == 0 {
		t.Fatal("no events forwarded")
	}
	for i, ev := range seen[:len(seen)-1] {
		if ev.done != nil {
			t.Fatalf("done marker at position %d/%d, before the stream drained", i, len(seen))
		}
	}
	if seen[len(seen)-1].done == nil {
		t.Fatalf("last event is not the done marker: %#v", seen[len(seen)-1])
	}
}

// newDemoModel creates a sandbox for the V3 side only — the raw lane
// is a direct completion with no filesystem tools, so no .demo-raw-*
// dirs accumulate in the workspace.
func TestNewDemoModelCreatesOnlyV3Sandbox(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"m"}]}`))
	}))
	defer srv.Close()

	dir := t.TempDir()
	m, err := newDemoModel(srv.URL, dir, "short")
	if err != nil {
		t.Fatalf("newDemoModel: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("workspace entries = %d, want only the v3 sandbox", len(entries))
	}
	if name := entries[0].Name(); !strings.HasPrefix(name, ".demo-v3-") || name != m.v3Sandbox {
		t.Fatalf("sandbox dir = %q, want %q (.demo-v3-*)", name, m.v3Sandbox)
	}
}

// The prompt type-out advances per rune so multi-byte characters are
// never split into invalid UTF-8 mid-animation.
func TestDemoPromptTypeOutIsRuneSafe(t *testing.T) {
	rawChild := newTUIModel("http://unused")
	v3Child := newTUIModel("http://unused")
	m := &demoModel{
		prompt:     demoPrompt{Prompt: "fix — draw ─ boxes"},
		width:      120,
		height:     30,
		rawChild:   &rawChild,
		v3Child:    &v3Child,
		activePane: "v3",
		// streamsFired keeps the tick handler from launching real requests.
		streamsFired: true,
	}
	promptLen := len([]rune(m.prompt.Prompt))
	for i := 0; i < promptLen+2; i++ {
		if view := m.View(); !utf8.ValidString(view) {
			t.Fatalf("view contains invalid UTF-8 at promptShown=%d", m.promptShown)
		}
		_, _ = m.Update(demoTickMsg(time.Now()))
	}
	if m.promptShown != promptLen {
		t.Fatalf("promptShown = %d, want rune count %d", m.promptShown, promptLen)
	}
}

func TestDemoPromptStatusDoesNotInventZeroPercentProgress(t *testing.T) {
	child := &tuiModel{
		promptTotal:     3000,
		promptEvalStart: time.Now(),
	}
	if got := streamStatus(child, false, false, nil); got != "processing prompt…" {
		t.Fatalf("status = %q, want indeterminate prompt progress", got)
	}
}

func TestDemoPromptStatusUsesRealSlotProgress(t *testing.T) {
	child := &tuiModel{
		promptProcessed: 750,
		promptTotal:     3000,
		promptPct:       0.25,
		promptEvalStart: time.Now(),
	}
	if got := streamStatus(child, false, false, nil); got != "processing prompt 25%" {
		t.Fatalf("status = %q, want current wire-format percentage", got)
	}
}
