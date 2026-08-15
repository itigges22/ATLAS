package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
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

// A fence containing only the sentinel is not a file. Measured live: asked
// for the contents of test_banner.py, the model replied
// "```python\n@fenced\n```", which extracts as non-empty and would have
// landed a file whose entire contents are the word @fenced.
func TestSentinelOnlyFenceIsNotContent(t *testing.T) {
	calls := 0
	srv := stubInference(t, []string{
		"```python\n@fenced\n```",
		"```python\nprint(\"real\")\n```",
	}, 50, &calls)
	defer srv.Close()
	t.Setenv("ATLAS_LLAMA_URL", srv.URL)

	ctx := &AgentContext{}
	got, err := fetchFencedContent(ctx, `{"type":"tool_call"}`, "a.py")
	if err != nil {
		t.Fatalf("expected the retry to succeed: %v", err)
	}
	if got != "print(\"real\")\n" {
		t.Fatalf("sentinel-only fence was accepted as content: %q", got)
	}
	if calls != 2 {
		t.Fatalf("expected a retry after the sentinel-only fence, got %d calls", calls)
	}
}

// --- Phase 2: progress-aware fenced-fetch bounds ---------------------------
//
// The zero-byte failure is a model that streams reasoning to max_tokens and
// never opens a fence. Measured on the seed-20260901 run: 9 of 17 zero-byte
// failures ran 175-311s, two attempts can consume ~10 minutes, and
// P(timeout | >=1 hang) was 8/11. The bound is on PROGRESS, not on total
// elapsed: a successful fetch legitimately ran 217s while producing content
// throughout.

func TestProductionFencedBoundsUseTheRealDefaults(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "")
	t.Setenv("ATLAS_FENCED_IDLE_SEC", "")
	if got := fencedFirstContentTimeout(); got != 60*time.Second {
		t.Errorf("first-content default = %v, want 60s", got)
	}
	if got := fencedIdleTimeout(); got != 30*time.Second {
		t.Errorf("idle default = %v, want 30s", got)
	}
	// The seam is internal and opt-in; production reads the real clock.
	t.Setenv("ATLAS_FENCED_IDLE_SEC", "2")
	if got := fencedIdleTimeout(); got != 2*time.Second {
		t.Errorf("override ignored: %v", got)
	}
	t.Setenv("ATLAS_FENCED_IDLE_SEC", "-1")
	if got := fencedIdleTimeout(); got != 30*time.Second {
		t.Errorf("invalid override must fall back to the default, got %v", got)
	}
}

// reasoningOnlyStub streams reasoning forever and never emits content: the
// exact zero-byte shape that ran to max_tokens in production.
func reasoningOnlyStub(t *testing.T, stop <-chan struct{}) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for {
			select {
			case <-r.Context().Done():
				return
			case <-stop:
				return
			default:
			}
			d, _ := json.Marshal(map[string]interface{}{
				"choices": []map[string]interface{}{
					{"delta": map[string]string{"reasoning_content": "thinking "}}},
			})
			fmt.Fprintf(w, "data: %s\n\n", d)
			if fl != nil {
				fl.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
}

func TestZeroByteFencedStreamIsCutAtTheFirstContentBound(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "1")
	stop := make(chan struct{})
	defer close(stop)
	srv := reasoningOnlyStub(t, stop)
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.Ctx = context.Background()

	start := time.Now()
	out, _, err := callLLMOnceWithGrammar(ctx, []AgentMessage{{Role: "user", Content: "x"}},
		0.2, rawEmissionSentinel)
	elapsed := time.Since(start)
	if elapsed > 20*time.Second {
		t.Fatalf("stream was not bounded: %v", elapsed)
	}
	if extractFencedContent(out) != "" {
		t.Errorf("a reasoning-only stream must not yield content: %q", out)
	}
	t.Logf("bounded in %v (err=%v)", elapsed, err)
}

// A slow but continuously progressing payload must survive: each content
// chunk resets the idle timer, so total elapsed far exceeding the idle bound
// is fine as long as the gaps do not.
func TestSlowButProgressingFencedPayloadSurvives(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "2")
	t.Setenv("ATLAS_FENCED_IDLE_SEC", "1")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		parts := []string{"```python\n", "x = 1\n", "y = 2\n", "z = 3\n", "```"}
		for _, p := range parts {
			d, _ := json.Marshal(map[string]interface{}{
				"choices": []map[string]interface{}{
					{"delta": map[string]string{"content": p}}},
			})
			fmt.Fprintf(w, "data: %s\n\n", d)
			if fl != nil {
				fl.Flush()
			}
			// Well inside the idle bound, but five gaps sum past it.
			time.Sleep(300 * time.Millisecond)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.Ctx = context.Background()
	out, _, err := callLLMOnceWithGrammar(ctx, []AgentMessage{{Role: "user", Content: "x"}},
		0.2, rawEmissionSentinel)
	if err != nil {
		t.Fatalf("progressing stream was cut: %v", err)
	}
	if body := extractFencedContent(out); !strings.Contains(body, "z = 3") {
		t.Errorf("payload truncated: %q", out)
	}
}

// Reasoning is not progress. A stream that reasons past the idle bound and
// only then emits content is still cut, because the first-content bound
// governs until real bytes arrive.
func TestReasoningDoesNotResetTheProgressTimer(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "1")
	t.Setenv("ATLAS_FENCED_IDLE_SEC", "30")
	stop := make(chan struct{})
	defer close(stop)
	srv := reasoningOnlyStub(t, stop)
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.Ctx = context.Background()
	start := time.Now()
	callLLMOnceWithGrammar(ctx, []AgentMessage{{Role: "user", Content: "x"}}, 0.2, rawEmissionSentinel)
	if el := time.Since(start); el > 20*time.Second {
		t.Fatalf("reasoning kept the stream alive past the first-content bound: %v", el)
	}
}

// Ordinary turns are untouched: the watchdog exists only for the fenced
// sub-call, so a normal grammar stream with no content and slow reasoning is
// governed by the pre-existing reasoning budget, not by these bounds.
func TestOrdinaryTurnsAreNotBoundedByTheFencedWatchdog(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "1")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		time.Sleep(1500 * time.Millisecond) // past the fenced bound
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": `{"type":"done"}`}}},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
		if fl != nil {
			fl.Flush()
		}
	}))
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.Ctx = context.Background()
	out, _, err := callLLMOnceWithGrammar(ctx, []AgentMessage{{Role: "user", Content: "x"}}, 0.2, "")
	if err != nil {
		t.Fatalf("ordinary turn was cut by the fenced watchdog: %v", err)
	}
	if !strings.Contains(out, "done") {
		t.Errorf("ordinary turn lost its payload: %q", out)
	}
}

// Session cancellation still wins, and leaves nothing running.
func TestSessionCancellationAbortsTheFencedFetch(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "60")
	stop := make(chan struct{})
	defer close(stop)
	srv := reasoningOnlyStub(t, stop)
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = srv.URL
	cctx, cancel := context.WithCancel(context.Background())
	ctx.Ctx = cctx
	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()
	before := runtime.NumGoroutine()
	start := time.Now()
	callLLMOnceWithGrammar(ctx, []AgentMessage{{Role: "user", Content: "x"}}, 0.2, rawEmissionSentinel)
	if el := time.Since(start); el > 10*time.Second {
		t.Fatalf("cancellation did not propagate: %v", el)
	}
	time.Sleep(150 * time.Millisecond)
	if after := runtime.NumGoroutine(); after > before+4 {
		t.Errorf("goroutines leaked: before=%d after=%d", before, after)
	}
}

// --- Phase 2: session/path-scoped retry budget ------------------------------
//
// fetchFencedContent's attempt counter was a local, so a new write_file call
// re-entered with a fresh allowance: a path that had already burned two ~300s
// attempts could burn two more next turn. The budget now lives in session
// state, keyed by path.

func fencedFailStub(t *testing.T, calls *int, succeedOnCall int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := *calls
		*calls++
		w.Header().Set("Content-Type", "text/event-stream")
		body := "I am thinking about it."
		if succeedOnCall >= 0 && i == succeedOnCall {
			body = "```python\nx = 1\n```"
		}
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": body}}},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
	}))
}

func fencedCtx(t *testing.T, url string) *AgentContext {
	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = url
	ctx.Ctx = context.Background()
	return ctx
}

func TestFencedRetryBudgetSurvivesANewWriteFileCall(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)

	if _, err := fetchFencedContent(ctx, "raw", "solve.py"); err == nil {
		t.Fatal("expected failure")
	}
	afterFirst := calls
	if afterFirst != maxFencedFailuresPerPath {
		t.Fatalf("first call should spend the whole allowance: %d generations", afterFirst)
	}
	// A brand-new write_file for the same path re-enters the function.
	if _, err := fetchFencedContent(ctx, "raw", "solve.py"); err == nil {
		t.Fatal("expected refusal")
	}
	if calls != afterFirst {
		t.Errorf("a new write_file restarted the budget: %d generations after, want %d",
			calls, afterFirst)
	}
	if n := ctx.FencedFailures[fencedKey(ctx, "solve.py")]; n != maxFencedFailuresPerPath {
		t.Errorf("session failure count = %d", n)
	}
}

func TestFencedBudgetIsPerPath(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)

	fetchFencedContent(ctx, "raw", "a.py")
	spent := calls
	// A different path must have its own allowance.
	if _, err := fetchFencedContent(ctx, "raw", "b.py"); err == nil {
		t.Fatal("expected failure on b.py too")
	}
	if calls <= spent {
		t.Errorf("b.py was denied a.py's budget: %d generations", calls-spent)
	}
	if ctx.FencedFailures[fencedKey(ctx, "a.py")] != maxFencedFailuresPerPath ||
		ctx.FencedFailures[fencedKey(ctx, "b.py")] != maxFencedFailuresPerPath {
		t.Errorf("per-path counts wrong: %v", ctx.FencedFailures)
	}
}

func TestOneConstrainedRetryMayFollowAZeroByteFailure(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, 1) // first attempt empty, retry succeeds
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)

	out, err := fetchFencedContent(ctx, "raw", "solve.py")
	if err != nil {
		t.Fatalf("the constrained retry should have succeeded: %v", err)
	}
	if !strings.Contains(out, "x = 1") {
		t.Errorf("retry payload lost: %q", out)
	}
	if n := ctx.FencedFailures[fencedKey(ctx, "solve.py")]; n != 0 {
		t.Errorf("success must clear the consecutive-failure state, got %d", n)
	}
	if calls != 2 {
		t.Errorf("expected one failure plus one retry, got %d generations", calls)
	}
}

func TestSuccessRestoresTheFullAllowance(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, 1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)
	fetchFencedContent(ctx, "raw", "solve.py") // fail then succeed -> cleared
	if fencedBudgetExhausted(ctx, "solve.py") {
		t.Fatal("a healthy path must not stay exhausted")
	}
}

// Deadline present: a retry that cannot finish and still leave the reserve is
// not started.
func TestReserveAwareRetryAdmissionWithADeadline(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)
	dctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	ctx.Ctx = dctx

	if _, err := fetchFencedContent(ctx, "raw", "solve.py"); err == nil {
		t.Fatal("expected failure")
	}
	// first-content bound (60s) + reserve (20s) cannot fit in 500ms, so the
	// fetch is refused before any generation.
	if calls != 0 {
		t.Errorf("started %d generation(s) with no room to validate", calls)
	}
	if !strings.Contains(func() string {
		_, err := fetchFencedContent(ctx, "raw", "other.py")
		return err.Error()
	}(), "session budget") {
		t.Error("refusal should name the budget as the reason")
	}
}

// Deadline absent — production today. Fetching is still bounded by the
// progress watchdog; this makes no claim about total session duration.
func TestNoDeadlineStillBoundsFetchingWithoutClaimingSessionBudget(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL) // context.Background(): no deadline
	if _, ok := ctx.Ctx.Deadline(); ok {
		t.Fatal("fixture should have no deadline")
	}
	if !fencedFitsRemainingBudget(ctx) {
		t.Error("with no deadline there is no budget to fail against")
	}
	if _, err := fetchFencedContent(ctx, "raw", "solve.py"); err == nil {
		t.Fatal("expected failure")
	}
	// Bounded by the per-path allowance, not by any session budget.
	if calls != maxFencedFailuresPerPath {
		t.Errorf("unbounded: %d generations", calls)
	}
}

// A failed resolution must not touch the file.
func TestFailedFencedResolutionLeavesDiskBytesUnchanged(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)
	target := filepath.Join(ctx.WorkingDir, "solve.py")
	original := "original = True\n"
	if err := os.WriteFile(target, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}
	fetchFencedContent(ctx, "raw", "solve.py")
	after, _ := os.ReadFile(target)
	if string(after) != original {
		t.Errorf("failed resolution modified disk: %q", string(after))
	}
}

// --- Phase 2: black-box proof through the production agent loop ------------
//
// Drives runAgentLoop against a model that answers every write_file with the
// @fenced sentinel and then never emits a fence — the shape behind 17
// zero-byte resolutions and 10 timeouts in the seed-20260901 run.
//
// Against the parent each write_file starts a fresh unbounded fetch, so the
// generation count grows with the turn count. Against current the fetch is
// bounded by progress and the allowance is spent once per path, so the loop
// stops issuing generations and the file is untouched.
//
// Uses no symbol the fix introduced except the allowance constant, which is
// read as a bound rather than asserted, so the same fixture compiles and runs
// against the parent.
func TestFencedHangIsBoundedThroughTheAgentLoop(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "1")
	dir := t.TempDir()
	rel := "solve.py"
	original := "untouched = True\n"
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	fenceFetches := 0
	stop := make(chan struct{})
	defer close(stop)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v3/") || strings.HasPrefix(r.URL.Path, "/internal/") {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/syntax-check") {
			json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
			return
		}
		if strings.HasSuffix(r.URL.Path, "/execute") {
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			if strings.Contains(in.Code, ".atlas-mount-probe") {
				b, _ := os.ReadFile(filepath.Join(dir, ".atlas-mount-probe"))
				json.NewEncoder(w).Encode(map[string]interface{}{
					"success": true, "stdout": string(b), "exit_code": 0})
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"success": true, "stdout": "", "exit_code": 0})
			return
		}
		if !strings.HasSuffix(r.URL.Path, "/v1/chat/completions") {
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		if strings.Contains(string(raw), "single fenced block") {
			mu.Lock()
			fenceFetches++
			mu.Unlock()
			for {
				select {
				case <-r.Context().Done():
					return
				case <-stop:
					return
				default:
				}
				d, _ := json.Marshal(map[string]interface{}{
					"choices": []map[string]interface{}{
						{"delta": map[string]string{"reasoning_content": "thinking "}}},
				})
				fmt.Fprintf(w, "data: %s\n\n", d)
				if fl != nil {
					fl.Flush()
				}
				time.Sleep(5 * time.Millisecond)
			}
		}
		call, _ := json.Marshal(map[string]interface{}{
			"type": "tool_call", "name": "write_file",
			"args": map[string]string{"path": rel, "content": "@fenced"},
		})
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": string(call)}}},
		})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
	}))
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 8

	start := time.Now()
	if err := runAgentLoop(ctx, "Create solve.py."); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	elapsed := time.Since(start)

	mu.Lock()
	got := fenceFetches
	mu.Unlock()

	// The allowance is spent once for this path and never restored, however
	// many write_file calls follow. The parent restarts it every turn, so its
	// count scales with MaxTurns.
	if got > 2 {
		t.Errorf("fenced generations = %d across %d turns; the per-path session "+
			"allowance is 2 and a new write_file must not restore it",
			got, ctx.MaxTurns)
	}
	if elapsed > 60*time.Second {
		t.Errorf("loop was not bounded: %v", elapsed)
	}
	after, _ := os.ReadFile(filepath.Join(dir, rel))
	if string(after) != original {
		t.Errorf("a failed fenced resolution must not touch disk: %q", string(after))
	}
	t.Logf("fenced_generations=%d turns<=%d elapsed=%v", got, ctx.MaxTurns, elapsed)
}

func TestFencedAllowanceIsKeyedOnTheCanonicalPath(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()
	ctx := fencedCtx(t, srv.URL)

	fetchFencedContent(ctx, "raw", "solve.py")
	spent := calls
	// Same file, different spelling: must share the allowance, not restart it.
	if _, err := fetchFencedContent(ctx, "raw", "./solve.py"); err == nil {
		t.Fatal("expected refusal for the equivalent spelling")
	}
	if calls != spent {
		t.Errorf("./solve.py got a fresh allowance: %d extra generations", calls-spent)
	}
	if len(ctx.FencedFailures) != 1 {
		t.Errorf("equivalent spellings created %d keys: %v",
			len(ctx.FencedFailures), ctx.FencedFailures)
	}
	// A genuinely different path keeps its own.
	if _, err := fetchFencedContent(ctx, "raw", "other.py"); err == nil {
		t.Fatal("expected other.py to fail on its own budget")
	}
	if calls <= spent {
		t.Error("other.py was denied its independent allowance")
	}
}

func TestWatchdogCancelsOnlyTheChildAndChargesTheAllowance(t *testing.T) {
	t.Setenv("ATLAS_FENCED_FIRST_CONTENT_SEC", "1")
	stop := make(chan struct{})
	defer close(stop)
	srv := reasoningOnlyStub(t, stop)
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.InferenceURL = srv.URL
	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()
	ctx.Ctx = parent

	fetchFencedContent(ctx, "raw", "solve.py")
	// The agent session must survive its own watchdog.
	if err := parent.Err(); err != nil {
		t.Fatalf("watchdog killed the parent session context: %v", err)
	}
	// A cancelled zero-content attempt is a failure and must consume budget,
	// or the loop restarts it on the next write_file.
	if n := ctx.FencedFailures[fencedKey(ctx, "solve.py")]; n == 0 {
		t.Error("watchdog cancellation consumed no allowance")
	}
	if !fencedBudgetExhausted(ctx, "solve.py") {
		t.Error("two cancelled attempts should exhaust the path's allowance")
	}
}

func TestEnvOverridesRejectUnsafeValues(t *testing.T) {
	for _, bad := range []string{"0", "-5", "abc", "", "99999"} {
		t.Setenv("ATLAS_FENCED_IDLE_SEC", bad)
		if got := fencedIdleTimeout(); got != 30*time.Second {
			t.Errorf("override %q produced %v; unsafe values must fall back to 30s",
				bad, got)
		}
	}
	t.Setenv("ATLAS_FENCED_IDLE_SEC", "45")
	if got := fencedIdleTimeout(); got != 45*time.Second {
		t.Errorf("a reasonable override was ignored: %v", got)
	}
}

func TestNoPersistentStateOrWireFieldWasAdded(t *testing.T) {
	types, err := os.ReadFile("types.go")
	if err != nil {
		t.Fatal(err)
	}
	// Session-scoped, in-memory only: no file, no DB, no wire field.
	if !strings.Contains(string(types), "FencedFailures map[string]int") {
		t.Fatal("the allowance is not session state")
	}
	for _, f := range []string{"types.go", "agent.go"} {
		b, _ := os.ReadFile(f)
		for _, leak := range []string{`json:"fenced_failures`, "os.WriteFile(fencedState"} {
			if strings.Contains(string(b), leak) {
				t.Errorf("%s: %q suggests a wire field or persistent state", f, leak)
			}
		}
	}
}
