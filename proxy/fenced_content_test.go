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
	"sort"
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

// Deadline absent — no longer production (Phase 2B gives every session one),
// but still the shape a direct caller or an older embedder produces. Fetching
// stays bounded by the progress watchdog; this makes no claim about total
// session duration.
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

// --- Phase 2B commit B: the server-owned session budget ---------------------

func TestSessionBudgetBounds(t *testing.T) {
	for _, c := range []struct {
		raw       string
		wantTotal time.Duration
	}{
		{"", 600 * time.Second},
		{"abc", 600 * time.Second},
		{"0", 600 * time.Second},
		{"-1", 600 * time.Second},
		{"119", 600 * time.Second},
		{"3601", 600 * time.Second},
		{"999999", 600 * time.Second},
		{"120", 120 * time.Second},
		{"300", 300 * time.Second},
		{"3600", 3600 * time.Second},
	} {
		t.Run("override="+c.raw, func(t *testing.T) {
			if c.raw == "" {
				t.Setenv("ATLAS_AGENT_SESSION_TIMEOUT_SEC", "")
			} else {
				t.Setenv("ATLAS_AGENT_SESSION_TIMEOUT_SEC", c.raw)
			}
			total, reserve := sessionBudget()
			if total != c.wantTotal {
				t.Errorf("total = %v, want %v", total, c.wantTotal)
			}
			if reserve != 30*time.Second {
				t.Errorf("reserve = %v, want 30s", reserve)
			}
			if total-reserve <= 0 {
				t.Error("the work deadline is not positive")
			}
		})
	}
}

// timeoutFixture drives a real agent loop whose work context is already
// expiring, so the deadline lands inside production code rather than being
// simulated. hang controls where: mid-LLM-stream or mid-tool-call.
func timeoutFixture(t *testing.T, dir string, hang string, onLLM func()) (*AgentContext, map[string]string, map[string]int) {
	t.Helper()
	stop := make(chan struct{})
	t.Cleanup(func() { close(stop) })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/v3/"), strings.HasPrefix(r.URL.Path, "/internal/"):
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		case strings.HasSuffix(r.URL.Path, "/syntax-check"):
			var in struct{ Code string }
			json.NewDecoder(r.Body).Decode(&in)
			valid := !strings.Contains(in.Code, "]]")
			out := map[string]interface{}{"valid": valid}
			if !valid {
				out["errors"] = []string{"SyntaxError"}
			}
			json.NewEncoder(w).Encode(out)
			return
		case strings.HasSuffix(r.URL.Path, "/execute"):
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
		case strings.HasSuffix(r.URL.Path, "/shell"):
			// A tool call that never returns until the request is cancelled.
			if hang == "tool" {
				select {
				case <-r.Context().Done():
				case <-stop:
				}
				http.Error(w, "cancelled", http.StatusGatewayTimeout)
				return
			}
			json.NewEncoder(w).Encode(map[string]interface{}{
				"stdout": "", "stderr": "", "exit_code": 0})
			return
		case !strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
			http.NotFound(w, r)
			return
		}
		if onLLM != nil {
			onLLM()
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		if hang == "llm" {
			// Stream forever: the deadline has to be what stops this.
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
						{"delta": map[string]string{"content": "."}}}})
				fmt.Fprintf(w, "data: %s\n\n", d)
				if fl != nil {
					fl.Flush()
				}
				time.Sleep(5 * time.Millisecond)
			}
		}
		call, _ := json.Marshal(map[string]interface{}{
			"type": "tool_call", "name": "run_command",
			"args": map[string]string{"command": "sleep 60"}})
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": string(call)}}}})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
	}))
	t.Cleanup(srv.Close)

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 20

	// The two lifetimes the handler builds, with a deterministically short
	// work deadline instead of ten minutes.
	reqCtx, cancelReq := context.WithCancel(context.Background())
	t.Cleanup(cancelReq)
	workCtx, cancelWork := context.WithTimeout(reqCtx, 900*time.Millisecond)
	t.Cleanup(cancelWork)
	ctx.RequestCtx = reqCtx
	ctx.Ctx = workCtx
	ctx.cancelWork = cancelWork

	terminal := map[string]string{}
	census := map[string]int{}
	var mu sync.Mutex
	ctx.StreamFn = func(eventType string, data interface{}) {
		mu.Lock()
		defer mu.Unlock()
		census[eventType]++
		if eventType == "done" {
			b, _ := json.Marshal(data)
			json.Unmarshal(b, &terminal)
		}
	}
	return ctx, terminal, census
}

func TestWorkDeadlineEmitsExactlyOneTimedOutTerminal(t *testing.T) {
	for _, where := range []string{"llm", "tool"} {
		t.Run("deadline during the "+where, func(t *testing.T) {
			dir := t.TempDir()
			ctx, terminal, census := timeoutFixture(t, dir, where, nil)

			start := time.Now()
			if err := runAgentLoop(ctx, "Do something long."); err != nil {
				t.Fatalf("agent loop error: %v", err)
			}
			elapsed := time.Since(start)

			if terminal["status"] != string(TerminalTimedOut) {
				t.Errorf("status = %q, want timed_out (summary=%q)",
					terminal["status"], terminal["summary"])
			}
			if terminal["reason"] != "work_deadline" {
				t.Errorf("reason = %q", terminal["reason"])
			}
			if census["done"] != 1 {
				t.Errorf("%d terminal events", census["done"])
			}
			// The terminal arrives AFTER the work context is dead, which is
			// the whole point of the split.
			if ctx.Ctx.Err() == nil {
				t.Error("work context outlived the deadline")
			}
			if ctx.RequestCtx.Err() != nil {
				t.Error("the response lifetime died with the work")
			}
			// Well inside the reserve.
			if elapsed > 20*time.Second {
				t.Errorf("finalisation took %v", elapsed)
			}
			if strings.Contains(terminal["summary"], "is on disk and parses") {
				t.Errorf("a timeout implied completion: %q", terminal["summary"])
			}
			t.Logf("elapsed=%v terminal=%v", elapsed, terminal)
		})
	}
}

// A timeout is not a completion, whatever survives on disk.
func TestTimeoutNeverClaimsCompletion(t *testing.T) {
	dir := t.TempDir()
	// A perfectly valid artifact is present the whole time.
	os.WriteFile(filepath.Join(dir, "solve.py"), []byte("A = 1\n"), 0o644)
	ctx, terminal, _ := timeoutFixture(t, dir, "llm", nil)
	if err := runAgentLoop(ctx, "Do something long."); err != nil {
		t.Fatal(err)
	}
	if NormalizeTerminalStatus(terminal["status"]).Completed() {
		t.Errorf("a timeout with a valid artifact read as completed: %v", terminal)
	}
	if got, _ := os.ReadFile(filepath.Join(dir, "solve.py")); string(got) != "A = 1\n" {
		t.Errorf("the timeout changed an untracked file: %q", got)
	}
}

// The timeout path reuses the Phase 3B rules unchanged: a demonstrably safer
// checkpoint is restored, and anything short of that is left alone.
func TestTimeoutRestorationObeysTheSameEligibilityRules(t *testing.T) {
	for _, c := range []struct {
		name        string
		seed        func(t *testing.T, ctx *AgentContext, dir string)
		wantBytes   string
		wantRestore bool
	}{
		{name: "demonstrably safer checkpoint is restored",
			seed: func(t *testing.T, ctx *AgentContext, dir string) {
				args, _ := json.Marshal(map[string]string{
					"path": "solve.py", "content": "def f():\n    return [1]\n"})
				executeToolCall("write_file", args, ctx)
				os.WriteFile(filepath.Join(dir, "solve.py"),
					[]byte("def f():\n    return [1]]\n"), 0o644)
			},
			wantBytes: "def f():\n    return [1]\n", wantRestore: true},
		{name: "invalid bytes with no checkpoint are kept",
			seed: func(t *testing.T, ctx *AgentContext, dir string) {
				// Written broken from the start, so nothing was ever valid.
				args, _ := json.Marshal(map[string]string{
					"path": "solve.py", "content": "def f():\n    return [1]]\n"})
				executeToolCall("write_file", args, ctx)
			},
			wantBytes: "def f():\n    return [1]]\n"},
		{name: "unknown validation is not a reason to act",
			seed: func(t *testing.T, ctx *AgentContext, dir string) {
				args, _ := json.Marshal(map[string]string{
					"path": "solve.py", "content": "def f():\n    return [1]\n"})
				executeToolCall("write_file", args, ctx)
				os.WriteFile(filepath.Join(dir, "solve.py"),
					[]byte("def f():\n    return [2]\n"), 0o644)
				// Checker gone: the current bytes have no verdict at all.
				ctx.SandboxURL = "http://127.0.0.1:1"
			},
			wantBytes: "def f():\n    return [2]\n"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			ctx, terminal, _ := timeoutFixture(t, dir, "llm", nil)
			c.seed(t, ctx, dir)

			if err := runAgentLoop(ctx, "Do something long."); err != nil {
				t.Fatal(err)
			}
			got, _ := os.ReadFile(filepath.Join(dir, "solve.py"))
			if string(got) != c.wantBytes {
				t.Errorf("disk = %q, want %q", got, c.wantBytes)
			}
			if c.wantRestore {
				if !strings.Contains(terminal["summary"], "Put back the last version") {
					t.Errorf("restore was not disclosed: %q", terminal["summary"])
				}
				d := ctx.Ledger[ledgerKey(ctx, "solve.py")]
				if d == nil || d.CurrentHash != hashBytes(got) {
					t.Error("the ledger does not describe the restored bytes")
				}
			}
			// Restored or not, a timeout is still a timeout.
			if terminal["status"] != string(TerminalTimedOut) {
				t.Errorf("status = %q", terminal["status"])
			}
		})
	}
}

// The reaper stops this session's jobs and leaves everything else alone.
func TestTimeoutReapsOnlyThisSessionsBackgroundJobs(t *testing.T) {
	var stopped []string
	var mu sync.Mutex
	zero := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/stop") {
			id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/jobs/"), "/stop")
			mu.Lock()
			stopped = append(stopped, id)
			mu.Unlock()
			json.NewEncoder(w).Encode(map[string]interface{}{
				"job_id": id, "killed": true, "exit_code": zero,
				"stdout": []string{}, "stderr": []string{}})
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.SandboxURL = srv.URL
	ctx.StreamFn = func(string, interface{}) {}
	ctx.BackgroundJobs = map[string]string{
		"mine-1": "python app.py", "mine-2": "npm start",
	}
	raiseWorkspaceHazard(ctx)
	raiseWorkspaceHazard(ctx)

	reapSessionBackgroundJobs(ctx)

	mu.Lock()
	got := append([]string(nil), stopped...)
	mu.Unlock()
	sort.Strings(got)
	if len(got) != 2 || got[0] != "mine-1" || got[1] != "mine-2" {
		t.Errorf("reaped %v, want exactly this session's two jobs", got)
	}
	if len(ctx.BackgroundJobs) != 0 {
		t.Errorf("%d jobs still tracked after reaping", len(ctx.BackgroundJobs))
	}
	// Confirmed exits clear the hazard, which is what lets restoration run.
	if workspaceHazardous(ctx) {
		t.Error("hazard still raised after two confirmed exits")
	}
}

// A job that cannot be confirmed gone keeps the hazard raised, and the hazard
// is what stops restoration touching a file something may still be writing.
func TestUnconfirmedJobKeepsTheHazardAndBlocksRestore(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// No exit_code: signalled, not reaped.
		json.NewEncoder(w).Encode(map[string]interface{}{
			"job_id": "j1", "killed": true, "stdout": []string{}, "stderr": []string{}})
	}))
	defer srv.Close()

	ctx := NewAgentContext(t.TempDir(), Tier2Medium)
	ctx.SandboxURL = srv.URL
	ctx.StreamFn = func(string, interface{}) {}
	ctx.BackgroundJobs = map[string]string{"j1": "python app.py"}
	raiseWorkspaceHazard(ctx)

	reapSessionBackgroundJobs(ctx)
	if !workspaceHazardous(ctx) {
		t.Fatal("an unconfirmed exit cleared the hazard")
	}
	if len(restoreSaferDeliverables(ctx)) != 0 {
		t.Error("restoration ran while a job may still be writing")
	}
}

// A client that goes away is not a server timeout.
func TestClientDisconnectIsNotATimeout(t *testing.T) {
	dir := t.TempDir()
	ctx, terminal, census := timeoutFixture(t, dir, "llm", nil)
	// Kill the response lifetime first, as a dropped connection does.
	reqCtx, cancelReq := context.WithCancel(context.Background())
	workCtx, cancelWork := context.WithCancel(reqCtx)
	ctx.RequestCtx = reqCtx
	ctx.Ctx = workCtx
	ctx.cancelWork = cancelWork
	cancelReq()

	err := runAgentLoop(ctx, "Do something long.")
	if err == nil {
		t.Error("a disconnect should surface as the context error")
	}
	if terminal["status"] == string(TerminalTimedOut) {
		t.Error("a disconnect was reported as a server timeout")
	}
	if census["done"] != 0 {
		t.Errorf("%d terminal events emitted into a closed response", census["done"])
	}
}

// The bounded-vs-unbounded contrast, written so it runs on the parent tree
// too: only ctx.Ctx is set, which is the one lifetime the parent has. The
// parent's loop returns the context error and emits nothing, so a client sees
// a stream that simply stops. Here the server owns the deadline and says so.
func TestASessionThatRunsOutOfTimeStillEndsWithATerminal(t *testing.T) {
	dir := t.TempDir()
	stop := make(chan struct{})
	defer close(stop)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v3/") || strings.HasPrefix(r.URL.Path, "/internal/") {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
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
					{"delta": map[string]string{"content": "."}}}})
			fmt.Fprintf(w, "data: %s\n\n", d)
			if fl != nil {
				fl.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer srv.Close()

	before := runtime.NumGoroutine()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 20
	workCtx, cancelWork := context.WithTimeout(context.Background(), 900*time.Millisecond)
	defer cancelWork()
	ctx.Ctx = workCtx

	var mu sync.Mutex
	terminals := 0
	var payload map[string]string
	ctx.StreamFn = func(eventType string, data interface{}) {
		if eventType != "done" {
			return
		}
		b, _ := json.Marshal(data)
		mu.Lock()
		terminals++
		json.Unmarshal(b, &payload)
		mu.Unlock()
	}

	start := time.Now()
	runAgentLoop(ctx, "Do something long.")
	elapsed := time.Since(start)

	mu.Lock()
	n, got := terminals, payload
	mu.Unlock()

	if n != 1 {
		t.Fatalf("%d terminal events; a session that runs out of time must "+
			"still tell the client what happened", n)
	}
	if got["status"] != "timed_out" {
		t.Errorf("status = %q, want timed_out", got["status"])
	}
	if elapsed > 20*time.Second {
		t.Errorf("the deadline did not bound the run: %v", elapsed)
	}

	// Nothing left running behind it.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= before+2 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if after := runtime.NumGoroutine(); after > before+2 {
		t.Errorf("goroutines leaked: %d before, %d after", before, after)
	}
	t.Logf("elapsed=%v terminals=%d status=%q", elapsed, n, got["status"])
}

// Phase 2's reserve-aware admission needed no change to observe the new
// deadline: it already reserves against ctx.Ctx, which now always has one.
// This pins the wiring, because "it works automatically" is the kind of claim
// that stops being true silently.
func TestFencedAdmissionObservesTheWorkDeadline(t *testing.T) {
	calls := 0
	srv := fencedFailStub(t, &calls, -1)
	defer srv.Close()

	// A session with plenty of budget admits the fetch.
	roomy := fencedCtx(t, srv.URL)
	deep, cancelDeep := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancelDeep()
	roomy.Ctx = deep
	if !fencedFitsRemainingBudget(roomy) {
		t.Error("a session with ten minutes left refused a fenced fetch")
	}

	// One whose work deadline is closer than the fetch plus its reserve does
	// not: resolving would consume the time needed to validate the result.
	tight := fencedCtx(t, srv.URL)
	shallow, cancelShallow := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelShallow()
	tight.Ctx = shallow
	if fencedFitsRemainingBudget(tight) {
		t.Fatal("a fetch was admitted with no room left to validate it")
	}
	before := calls
	if _, err := fetchFencedContent(tight, "raw", "solve.py"); err == nil {
		t.Fatal("expected the admission refusal to surface")
	} else if !strings.Contains(err.Error(), "session budget") {
		t.Errorf("refusal did not name the budget: %v", err)
	}
	if calls != before {
		t.Errorf("a refused fetch still spent %d generations", calls-before)
	}

	// And the real budget leaves room: 600s total, 30s reserve.
	t.Setenv("ATLAS_AGENT_SESSION_TIMEOUT_SEC", "")
	total, reserve := sessionBudget()
	if total-reserve <= fencedFirstContentTimeout()+fencedReserve {
		t.Errorf("the default work deadline (%v) cannot admit a single fenced "+
			"fetch", total-reserve)
	}
}

// --- Phase 2C: the malformed fenced call ------------------------------------
//
// The fenced channel costs a full unconstrained generation per attempt, and it
// was opened before anyone asked whether the call could execute. A model
// re-sending `write_file {"content":"@fenced"}` with no path spent the whole
// session that way: the 300s canary reached turn 36 with nothing on disk.
//
// The preflight runs the checks the call has to survive anyway, by calling
// them, so nothing new decides anything.

// The property the whole design rests on: every call the preflight refuses is
// also refused by the tool, which is what makes "decline to resolve, then let
// the tool answer" safe rather than a way to write "@fenced" to disk.
func TestPreflightRefusalsAreASubsetOfTheToolsOwn(t *testing.T) {
	for _, c := range []struct{ name, args string }{
		{"no arguments", ``},
		{"null arguments", `null`},
		{"malformed shape", `{"path":123,"content":"@fenced"}`},
		{"missing path", `{"content":"@fenced"}`},
		{"blank path", `{"path":"","content":"@fenced"}`},
		{"whitespace path", `{"path":"   ","content":"@fenced"}`},
		{"tab path", `{"path":"\t\n","content":"@fenced"}`},
		{"workspace escape", `{"path":"../outside.py","content":"@fenced"}`},
		{"absolute escape", `{"path":"/etc/passwd","content":"@fenced"}`},
		{"deny-listed", `{"path":".env","content":"@fenced"}`},
		{"deny-listed key", `{"path":"id_rsa.pem","content":"@fenced"}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			ctx := NewAgentContext(dir, Tier2Medium)
			ctx.PermissionMode = PermissionYolo
			ctx.StreamFn = func(string, interface{}) {}

			ok, why := fencedCallIsExecutable("write_file", json.RawMessage(c.args), ctx)
			if ok {
				t.Fatalf("preflight admitted an unusable call")
			}
			if why == "" {
				t.Error("refused without saying why")
			}
			// The tool refuses it too, with no mutation and nothing on disk.
			res := executeToolCall("write_file", json.RawMessage(c.args), ctx)
			if res.Success {
				t.Fatalf("the tool accepted what the preflight refused: %s", c.args)
			}
			if res.MutationStatus != MutationNone {
				t.Errorf("MutationStatus = %q, want none", res.MutationStatus)
			}
			if !res.Classified() {
				t.Errorf("unclassified refusal: %+v", res)
			}
			var found []string
			filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
				if err == nil && info != nil && !info.IsDir() {
					found = append(found, p)
				}
				return nil
			})
			if len(found) != 0 {
				t.Errorf("an unusable call put files on disk: %v", found)
			}
			if len(ctx.Ledger) != 0 {
				t.Errorf("an unusable call entered the ledger: %v", ctx.Ledger)
			}
		})
	}
}

// A usable call is still admitted — the preflight is a gate, not a wall.
func TestPreflightAdmitsAUsableCall(t *testing.T) {
	dir := t.TempDir()
	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.PermissionMode = PermissionYolo
	for _, args := range []string{
		`{"path":"solve.py","content":"@fenced"}`,
		`{"path":"./solve.py","content":"@fenced"}`,
		`{"path":"pkg/solve.py","content":"@fenced"}`,
	} {
		if ok, why := fencedCallIsExecutable("write_file", json.RawMessage(args), ctx); !ok {
			t.Errorf("%s refused: %s", args, why)
		}
	}
}

// fencedCountingStub answers the model with a fixed script and counts how many
// unconstrained fenced generations were requested. Free of Phase 2C symbols so
// the same fixture runs on the parent tree.
func fencedCountingStub(t *testing.T, dir string, calls, fences *int, script func(int) map[string]interface{}) *httptest.Server {
	t.Helper()
	var mu sync.Mutex
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/v3/"), strings.HasPrefix(r.URL.Path, "/internal/"):
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		case strings.HasSuffix(r.URL.Path, "/syntax-check"):
			json.NewEncoder(w).Encode(map[string]interface{}{"valid": true})
			return
		case strings.HasSuffix(r.URL.Path, "/execute"):
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
		case !strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")

		// The fenced sub-call is the one that asks for a single fenced block.
		if strings.Contains(string(raw), "single fenced block") {
			mu.Lock()
			*fences++
			mu.Unlock()
			d, _ := json.Marshal(map[string]interface{}{
				"choices": []map[string]interface{}{
					{"delta": map[string]string{"content": "```python\nA = 1\n```"}}}})
			fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
			return
		}
		mu.Lock()
		i := *calls
		*calls++
		mu.Unlock()
		body, _ := json.Marshal(script(i))
		d, _ := json.Marshal(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": string(body)}}}})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", d)
	}))
}

// The production claim, through the real loop: a malformed fenced call starts
// ZERO generations, and the run reaches its terminal in a handful of turns
// instead of burning the session. On the parent every one of those turns opens
// the channel first.
func TestMalformedFencedCallStartsNoGeneration(t *testing.T) {
	dir := t.TempDir()
	calls, fences := 0, 0
	srv := fencedCountingStub(t, dir, &calls, &fences, func(i int) map[string]interface{} {
		// The exact shape the canary produced: content set, path absent.
		return map[string]interface{}{
			"type": "tool_call", "name": "write_file",
			"args": map[string]string{"content": "@fenced"}}
	})
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 20

	var terminal map[string]string
	ctx.StreamFn = func(eventType string, data interface{}) {
		if eventType == "done" {
			b, _ := json.Marshal(data)
			json.Unmarshal(b, &terminal)
		}
	}
	start := time.Now()
	if err := runAgentLoop(ctx, "Write solve.py."); err != nil {
		t.Fatalf("agent loop error: %v", err)
	}
	elapsed := time.Since(start)

	if fences != 0 {
		t.Errorf("%d fenced generations started for a call with no path", fences)
	}
	// Bounded escalation: the repeat detector sees identical malformed calls
	// and stops the run quickly.
	if calls > 12 {
		t.Errorf("%d model turns before terminating on a malformed call", calls)
	}
	if terminal["status"] == "" {
		t.Fatal("no classified terminal")
	}
	if NormalizeTerminalStatus(terminal["status"]).Completed() {
		t.Errorf("a run that wrote nothing reported %q", terminal["status"])
	}
	// No allowance state for a path that never existed.
	if len(ctx.FencedFailures) != 0 {
		t.Errorf("allowance keys created for an unusable call: %v", ctx.FencedFailures)
	}
	var found []string
	filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() &&
			!strings.Contains(p, ".atlas-mount-probe") {
			found = append(found, p)
		}
		return nil
	})
	if len(found) != 0 {
		t.Errorf("files on disk after a malformed run: %v", found)
	}
	t.Logf("turns=%d fenced_generations=%d elapsed=%v status=%q reason=%q",
		calls, fences, elapsed, terminal["status"], terminal["reason"])
}

// Equivalent bad spellings cannot each buy their own allowance, because none
// of them reaches the allowance at all.
func TestNoAllowanceKeyForAnyUnusableSpelling(t *testing.T) {
	dir := t.TempDir()
	calls, fences := 0, 0
	spellings := []string{"", "   ", "\t", "../escape.py", ".env"}
	srv := fencedCountingStub(t, dir, &calls, &fences, func(i int) map[string]interface{} {
		return map[string]interface{}{
			"type": "tool_call", "name": "write_file",
			"args": map[string]string{
				"path": spellings[i%len(spellings)], "content": "@fenced"}}
	})
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 12
	ctx.StreamFn = func(string, interface{}) {}

	if err := runAgentLoop(ctx, "Write solve.py."); err != nil {
		t.Fatal(err)
	}
	if fences != 0 {
		t.Errorf("%d generations started across unusable spellings", fences)
	}
	if len(ctx.FencedFailures) != 0 {
		t.Errorf("allowance map is not empty: %v", ctx.FencedFailures)
	}
}

// The channel still works for a call that can execute, and the bytes it
// carries still land.
func TestValidFencedPathStillResolvesThroughTheLoop(t *testing.T) {
	dir := t.TempDir()
	calls, fences := 0, 0
	srv := fencedCountingStub(t, dir, &calls, &fences, func(i int) map[string]interface{} {
		if i == 0 {
			return map[string]interface{}{
				"type": "tool_call", "name": "write_file",
				"args": map[string]string{"path": "solve.py", "content": "@fenced"}}
		}
		return map[string]interface{}{"type": "done", "summary": "wrote solve.py"}
	})
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 8
	ctx.StreamFn = func(string, interface{}) {}

	if err := runAgentLoop(ctx, "Write solve.py."); err != nil {
		t.Fatal(err)
	}
	if fences != 1 {
		t.Errorf("%d fenced generations for one valid call, want 1", fences)
	}
	got, err := os.ReadFile(filepath.Join(dir, "solve.py"))
	if err != nil {
		t.Fatalf("the resolved file never landed: %v", err)
	}
	if string(got) != "A = 1\n" {
		t.Errorf("disk = %q, want the fenced body", got)
	}
}

// An inline write never involved the channel and must be untouched by this.
func TestValidInlineWriteIsByteIdentical(t *testing.T) {
	dir := t.TempDir()
	calls, fences := 0, 0
	const body = "def solve():\n    return 42\n"
	srv := fencedCountingStub(t, dir, &calls, &fences, func(i int) map[string]interface{} {
		if i == 0 {
			return map[string]interface{}{
				"type": "tool_call", "name": "write_file",
				"args": map[string]string{"path": "solve.py", "content": body}}
		}
		return map[string]interface{}{"type": "done", "summary": "wrote solve.py"}
	})
	defer srv.Close()

	ctx := NewAgentContext(dir, Tier2Medium)
	ctx.InferenceURL = srv.URL
	ctx.SandboxURL = srv.URL
	ctx.V3URL = srv.URL
	ctx.PermissionMode = PermissionYolo
	ctx.MaxTurns = 8
	ctx.StreamFn = func(string, interface{}) {}

	if err := runAgentLoop(ctx, "Write solve.py."); err != nil {
		t.Fatal(err)
	}
	if fences != 0 {
		t.Errorf("an inline write opened the fenced channel %d times", fences)
	}
	got, _ := os.ReadFile(filepath.Join(dir, "solve.py"))
	if string(got) != body {
		t.Errorf("disk = %q, want %q", got, body)
	}
	d := ctx.Ledger[ledgerKey(ctx, "solve.py")]
	if d == nil || d.CurrentHash != hashBytes([]byte(body)) {
		t.Error("the ledger does not describe the inline write")
	}
}
