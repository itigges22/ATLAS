package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ------------------------------------------------------------------------
// Trust model (load-bearing — read before "fixing" go/path-injection CodeQL
// alerts on this file or its helpers in tools.go, context.go, gates.go).
//
// ATLAS is a single-tenant local-install agent. The proxy runs inside a
// container whose /workspace mount IS the user's project directory: the
// agent's whole purpose is to read, edit, and verify files there on
// behalf of the user who owns that directory. The "user-controlled" value
// CodeQL flags in agent path joins is almost always ctx.WorkingDir, which
// is set ONCE at agent startup from the bind mount — not from a per-
// request input. The few cases where the model emits a path
// (resolveAgentPath callers in tools.go) go through that resolver, which
// translates host-absolute paths into the container view + cleans `..`
// segments before any os call.
//
// What we deliberately do NOT do: enforce a strict in-workspace check
// that rejects paths resolving outside /workspace. The agent legitimately
// reads outside it (e.g. /etc/os-release for tier detection, /tmp for
// scratch). The container is the isolation boundary — host-side files
// not bind-mounted in are simply not reachable.
//
// If ATLAS is ever deployed multi-tenant (e.g. shared proxy with many
// users' workspaces colocated), every site flagged by go/path-injection
// here would need a real fix. Until then they're dismissed as false
// positives with this rationale.
// ------------------------------------------------------------------------

// stripThinkTags removes common <think>...</think> reasoning markers from a
// response string. Used as a defensive cleanup when
// reasoning_content gets surfaced as content fallback — the raw
// reasoning text sometimes still has the tags wrapping it.
var thinkTagRE = regexp.MustCompile(`(?s)<think>.*?</think>`)

func stripThinkTags(s string) string {
	return strings.TrimSpace(thinkTagRE.ReplaceAllString(s, ""))
}

// recoverStructuredReasoning accepts a complete agent envelope that a chat
// template routed to reasoning_content instead of content. Parse the JSON
// rather than matching serialized substrings: whitespace and key order are
// insignificant, and both `text` and `done` are valid terminal responses.
func recoverStructuredReasoning(s string) (string, bool) {
	recovered := stripThinkTags(s)
	if recovered == "" {
		return "", false
	}
	parsed, err := extractModelResponse(recovered)
	if err != nil {
		return "", false
	}
	switch parsed.Type {
	case "tool_call":
		return recovered, parsed.Name != "" && len(parsed.Args) > 0 &&
			string(parsed.Args) != "null"
	case "text":
		return recovered, parsed.Content != ""
	case "done":
		return recovered, parsed.Summary != ""
	default:
		return "", false
	}
}

// activeSessions tracks in-flight /v1/agent turns by session_id so
// /cancel can abort them. Map value is a *sessionCancel wrapping the
// context.CancelFunc from the per-request context.WithCancel wrapper.
//
// The pointer doubles as a per-turn identity token: cleanup uses
// CompareAndDelete so a finishing turn only removes its own entry. If a
// second /v1/agent request reuses the same session_id, the first turn's
// deferred cleanup no longer deletes the second turn's cancel func.
//
// Defense-in-depth: cancellation also flows naturally through TCP
// disconnect (handleAgent already binds ctx to r.Context()), but a
// reverse proxy may buffer the disconnect. /cancel gives the TUI a
// reliable, explicit kill switch.
var activeSessions sync.Map

// sessionCancel is the activeSessions map value — a comparable wrapper
// around the per-turn cancel func.
type sessionCancel struct {
	cancel context.CancelFunc
}

// ---------------------------------------------------------------------------
// Agent loop — iterative tool-calling loop between model and executors
// ---------------------------------------------------------------------------

// runAgentLoop runs the agent loop for a single user request.
// The model emits tool calls (constrained by grammar), the proxy executes them,
// and returns results. Continues until the model emits "done" or max turns hit.
// planGateMinScore is the plan-quality floor below which the plan-completion
// gate does not fire. The planner reports WinningScore per plan; a weak plan
// blocking a finished task is worse than no gate, and 0.6 keeps the gate on
// the plans the planner itself rates as sound (observed live plans score
// 0.80).
const planGateMinScore = 0.6

// maxTotalFailures bounds a whole run's failed tool calls, independent of the
// consecutive-error breaker. That breaker now resets when the rejection
// changes kind (a converging model must not be killed for iterating), so this
// is what stops a run cycling through failure modes forever. Set well above
// what a legitimate multi-edit task needs: run 11 used 3 and was still short.
const maxTotalFailures = 12

// maxGateBounces caps EACH of the verification, done-without-action,
// expected-output, and claim-check gates independently. Mirrors the
// parse-error cap: a gate that has bounced the same `done` three times is
// in a stuck loop, so its fourth is accepted rather than bounced forever.
// The other gates keep their own budgets — see runState.gateBounces.
const maxGateBounces = 3

// runState is the per-run evidence the completion-honesty gates decide
// on, plus their bounce budgets. One struct so the gates see the
// same facts on the done and text exits instead of two hand-copied
// gate blocks (which is exactly how the text exit shipped ungated once).
type runState struct {
	turn     int    // current loop turn, for tool-call IDs and logs
	response string // raw model output this turn, echoed on a bounce

	// Tool calls executed this run, of any kind. The intent gate uses it to
	// tell "announced a tool call and stopped" from ordinary narration after
	// work has already happened.
	toolsRun int
	// Name of a tool_call that has been streamed but not yet answered by a
	// tool_result. The call is announced before permission and execution,
	// so any exit in between has to answer it or the consumer is left with
	// a call that never resolves. Cleared by whichever path emits the
	// result — the normal one, or bounceToolCall on a refusal.
	pendingToolCall string
	// Set when a write/edit/structural_edit/delete landed in this run.
	madeProductiveChange bool
	// Set when a read-only tool succeeds — the model opened the project
	// to answer this message. Distinguishes a request the model treated
	// as work from one it answered conversationally, without consulting
	// a vocabulary list. See wantsStateChange.
	inspectedWorkspace bool
	// shadowGate is bounded diagnostic-only sequencing. Nothing but the
	// shadow emitter reads it.
	shadowGate shadowGateSeq
	// Files the prompt explicitly asks the model to produce
	// ("save your solution in X"). Checked against disk before `done` is
	// allowed — a model can satisfy the generic action gate with a
	// PARTIAL artifact or by exploring without ever committing the named
	// output (observed 2026-07-19). Computed once from the prompt.
	expectedOutputs []string
	// The expected-output gate fires at most ONCE per session: a named
	// deliverable might be PRODUCED AT RUNTIME by the model's code (not
	// authored), so repeatedly bouncing a correct done would steer the
	// model to fabricate a stand-in (#147 review finding #8).
	outputGateUsed bool
	// Set when a verification command (pytest, curl, go test, ...)
	// completed successfully in any turn of this run. One success per
	// loop is enough — the model can iterate without re-verifying every
	// turn. Also softens the consecutive-errors exit: post-write
	// run_command failures are usually verification noise, not a
	// genuinely stuck loop.
	verifiedThisLoop bool
	// verifiedByRedirect names the file a verification command piped into the
	// program's stdin, when one did. verifiedStandalone records that some
	// verification ran the program without a redirect. Both are needed: only
	// a run that ONLY ever verified through a redirect has failed to check
	// the artifact the way the caller will use it.
	verifiedByRedirect string
	verifiedStandalone bool
	// verifiedHashes binds the verification to the BYTES it verified: the
	// sha256 of every session-written file, snapshotted when a verifying run
	// succeeds. The exit re-hashes; any drift means the final artifact is not
	// the one that was checked. Session-level booleans cannot express that —
	// they verify a moment, not an artifact — which is the shared root of the
	// verify-then-modify, warned-write and stale-evidence holes (third-party
	// audit finding: evidence must be a contract tied to the final artifact).
	verifiedHashes map[string]string
	// pendingWarnedRun is the SET of paths whose last landed write carried a
	// parse warning and has not been executed since. A warned landing is
	// pending work, not advice: measured, a session wrote six times without a
	// single run, ignoring "Run it now" in four consecutive warnings, then
	// died on edit repeats. Further writes to such a path bounce until any
	// verification command runs.
	//
	// Membership IS the warning. It was briefly a map of booleans written
	// with both values, and the exit gate reads it by ranging over keys, so a
	// clean landing stored false and was then announced to the model as "on
	// disk with a parse warning ... as written it cannot work" over a file
	// that parsed. Both frozen Stage-1 sessions that spent themselves
	// rewriting an already-valid file took that gate at turn 1. Go through
	// markWarnedRun and the value is never anything but true.
	pendingWarnedRun map[string]bool
	// Phase 4B: how many times a raw @fenced write for a canonical path has
	// met the run-first demand, and whether that path's one recovery has been
	// spent. Both are session-local, bounded by the number of paths the run
	// touches, and hold no file contents.
	fencedRunFirstRepeats map[string]int
	fencedRecoverySpent   map[string]bool
	// fencedChannelClosed records the canonical paths whose fenced channel has
	// been declared spent to the model, so the offer is made once and a new
	// turn, an alias, or an unrelated success cannot re-open it.
	fencedChannelClosed map[string]bool
	// steerRepeats counts, per canonical path, how many times a write_file
	// steering refusal has been ignored and repeated; steerRecovered records
	// which paths have already spent their one recovery. Both are cleared by
	// a materially different action on the SAME path, and by nothing else --
	// a success elsewhere is not evidence that this path is unstuck.
	steerRepeats   map[string]int
	steerRecovered map[string]bool
	// noopEditRepeats counts explicit old_str == new_str edits per canonical
	// path AND the exact broken hash they were sent against; brokenArtifact-
	// Recovered records which of those evidence generations have spent their
	// one recovery. Keying on the hash is what makes a new generation re-arm
	// and a stale one unusable.
	noopEditRepeats         map[string]int
	brokenArtifactRecovered map[string]bool
	// c4Rejected is what the session knows about replacements that were
	// refused while the file they targeted stayed valid on disk. Keyed by
	// canonical path AND the surviving disk hash, so new bytes are a new
	// question and the old evidence cannot describe them.
	c4Rejected map[string]*proposalRejection
	// mutationDebt is what the session still owes on a per-path basis: a
	// valid, permitted, in-workspace mutation the model asked for that has
	// not reached a demonstrated resolved state. Canonically keyed, bounded,
	// and deliberately NOT the deliverable ledger -- that records what the
	// session owns on disk, and an intent that never landed owns nothing.
	mutationDebt   map[string]*mutationDebtEntry
	debtGeneration int
	// debtRecoveryOffered is the last generation the model was given a chance
	// to settle. Bumping the generation when NEW work goes unresolved buys
	// exactly one more offer, and the total is capped so it cannot loop.
	debtRecoveryOffered int
	debtRecoveryCount   int
	// debtOverflow fails closed past the ceiling: the session stops naming
	// individual paths but never stops reporting that work is unresolved.
	debtOverflow bool
	// toolBanned records (tool, path) pairs the loop has taken away from the
	// model after it proved it cannot use them on that file. Advice is not a
	// fix when the model ignores advice: measured dogfooding "build me a
	// snake game", a no-op edit_file was refused with an explicit "re-sending
	// will not help, use structural_edit instead" and the model re-sent the
	// identical call on the very next turn, twice, until the breaker killed a
	// 48-minute session. A tool the harness removes is a contract; a tool the
	// harness merely discourages is a suggestion.
	toolBanned map[string]bool

	// redRunStreak counts consecutive FAILED verification commands with no
	// green in between. Past a threshold, incremental edits have had their
	// chance: the bare-model retry loop's whole advantage is the fresh
	// rewrite, and sessions here were observed re-running a broken program
	// five times while nibbling at it with edits.
	redRunStreak int
	// Set when a verification command RAN AND FAILED and none has
	// succeeded since. Observed session state, not a guess about the
	// request: once a test has gone red in this loop, declaring done is
	// dishonest regardless of how the user phrased the ask. Closes the
	// case the message-shape check cannot see (2026-07-21 dogfooding:
	// the model watched pytest fail 5/5 three times, diagnosed the fix
	// in prose, and exited through a bare text narration).
	sawFailedVerification bool
	// The red verification command was a long-running server rather than a
	// broken build — it never exited, or the port was already bound. Changes
	// what the verification gate tells the model to do next, because
	// re-running a server in the foreground can never exit clean.
	serverStartBlocked bool
	// Whether the user prompt is a repair/fix request. Computed once —
	// the user message doesn't change mid-loop.
	userWantsVerification bool
	// Bounces spent per gate this run, keyed by gate name.
	//
	// Per-gate rather than one shared counter: the gates are evaluated in
	// a fixed order, so a single counter let whichever fired first spend
	// the whole allowance and silence the rest. An observed session put
	// all three bounces on the verification gate, so the
	// done-without-action gate never ran and the model exited having
	// changed nothing while claiming the work was already present. Each
	// gate reports a DIFFERENT problem, and a gate that has said its
	// piece three times must stop without muting the others.
	gateBounces map[string]int

	// correctives queued by the loop-health detectors this turn, drained
	// after the tool result so the next LLM call sees them in order:
	// assistant(tool_call) → tool(result) → user([system note]: …).
	//
	// Role MUST be "user": some Jinja chat templates enforce "system
	// message must be at the beginning" and 500 on a system role appended
	// mid-conversation. The "[system note]:" prefix is how the model
	// tells loop machinery from an actual user instruction.
	//
	// Several detectors firing on one turn is intentional — the model
	// gets each signal, since they observe the same stuckness from
	// different angles (identical args vs rehashed reasoning vs a lens
	// quality crash).
	correctives []string
}

// queueCorrective adds a loop-health corrective for this turn.
func (s *runState) queueCorrective(msg string) {
	if msg != "" {
		s.correctives = append(s.correctives, msg)
	}
}

// drainCorrectives appends every queued corrective to the conversation
// and clears the queue. Called once per turn, after the tool result.
func (s *runState) drainCorrectives(ctx *AgentContext) {
	for _, msg := range s.correctives {
		ctx.Messages = append(ctx.Messages, AgentMessage{
			Role:    "user",
			Content: "[system note]: " + msg,
		})
	}
	s.correctives = nil
}

// bounce echoes the model's output plus a synthetic tool rejection into
// the conversation, so the next LLM call sees exactly why the attempt
// was refused. The one shape every gate and guard refusal shares.
func (s *runState) bounce(ctx *AgentContext, toolName, rejection string) {
	// The rejection reaches the model through Messages. It reached nothing
	// else: a completion gate holding a run back — "you were asked to change
	// something and have not" — produced no event, so the TUI showed an
	// unexplained pause and the run's own event stream held no record that a
	// gate had fired at all. Measured across 84 sessions, that made the
	// completion gates unobservable while 11 of 35 failures were the model
	// stopping short, which is exactly what they exist to catch.
	//
	// Emitted as its own type rather than a tool_result: nothing was
	// executed, and a consumer pairing calls with results must not see a
	// result it never made a call for.
	ctx.Stream("gate", map[string]interface{}{
		"gate":   toolName,
		"turn":   s.turn,
		"reason": truncateStr(rejection, 200),
	})
	ctx.Messages = append(ctx.Messages, AgentMessage{Role: "assistant", Content: s.response})
	ctx.Messages = append(ctx.Messages, AgentMessage{
		Role:       "tool",
		Content:    fmt.Sprintf(`{"success":false,"error":%q}`, rejection),
		ToolCallID: fmt.Sprintf("call_%d", s.turn),
		ToolName:   toolName,
	})
}

// bounceToolCall is bounce for a rejection that lands AFTER the tool_call
// event has already gone out. Without a matching tool_result the consumer
// sees a call that never resolves: the TUI prints the call row and nothing
// after it, so the user is never told the tool was refused and why.
//
// Observed live: a model tried to overwrite a fixture input file, the
// surgical-edit gate correctly refused, and the refusal reached the model
// (through ctx.Messages) but never the event stream — the session's tool_call
// and tool_result counts disagreed by one.
func (s *runState) bounceToolCall(ctx *AgentContext, toolName, rejection string) {
	s.bounce(ctx, toolName, rejection)
	s.pendingToolCall = ""
	ctx.Stream("tool_result", map[string]interface{}{
		"tool":    toolName,
		"success": false,
		"error":   rejection,
	})
}

// verificationDemandedAndUnmet reports whether this run needed a passing
// verification command and never got one. Independent of the bounce budget:
// exhausting the bounces means the gate stopped blocking, not that the work
// was verified.
func (s *runState) verificationDemandedAndUnmet() bool {
	return (s.userWantsVerification || s.sawFailedVerification) && !s.verifiedThisLoop
}

// actionDemandedAndUnmet reports a run that was asked to change something on
// disk and finished without changing anything.
//
// The action gate bounces this while it has bounces left, and then stops:
// chargeBounce is capped so an exhausted gate cannot loop. Past that cap the
// exit goes through unremarked, which is how a session ends having written
// nothing while saying nothing about it. Observed on smallrung_toml: a
// structural_edit was refused for making the file invalid, the model gave up
// on tools and emitted the replacement as chat text, and the run finished
// with that code as its summary — the user is shown a block of code that
// was never applied, with no indication it was not.
func (s *runState) actionDemandedAndUnmet(ctx *AgentContext, userMessage string) bool {
	return observeActionDemand(ctx, s, shadowGateActionDemanded,
		decideActionDemand(ctx.TaskContract, userMessage, ctx.Tier, s.inspectedWorkspace)) &&
		!s.madeProductiveChange
}

// exitGates runs the completion-honesty gates a done or text exit must
// clear, in order: verification, done-without-action, expected-output,
// claim-check. claimText is the completion claim to check structurally
// (the done summary, or the text narration — on a text exit the
// narration IS the claim). Returns the failing gate's tool name and
// rejection, or "" to let the exit pass. Gates run in EVERY permission
// mode: yolo means "don't ask permission for destructive calls", not
// "skip completion checks". Bounces stay capped by maxGateBounces, so
// unattended runs cannot loop on a gate.
func (s *runState) exitGates(ctx *AgentContext, userMessage, claimText string) (string, string) {
	// Announcing a tool call is not making one. Observed on a question about
	// code: the model replied "I need to read orders.py — I'll start by
	// outlining the file to locate the function" and the turn ended there,
	// because text is a terminal event. It had the right intent and never
	// acted on it. Only fires before any tool has run, so it cannot interrupt
	// work already in progress.
	// A reply that signs off promising the actual answer leaves the user with
	// half of one, whether or not tools ran. Checked before the zero-tools
	// case below, since this one applies after the work is done.
	if promisesMoreContent(claimText) && s.chargeBounce("intent_gate") {
		log.Printf("[agent] intent gate: bouncing a reply that promised content it did not deliver (bounce %d/%d)",
			s.gateBounces["intent_gate"], maxGateBounces)
		return "intent_gate", "You ended by saying you would provide the answer, but the reply stops there and the turn ends with it — the user sees only the promise. Give the actual content now, in full, in a single `text` reply."
	}
	if s.toolsRun == 0 && announcesImminentToolUse(claimText) && s.chargeBounce("intent_gate") {
		log.Printf("[agent] intent gate: bouncing a text exit that announced a tool call without making one (bounce %d/%d)",
			s.gateBounces["intent_gate"], maxGateBounces)
		return "intent_gate", "You described the tool call you were about to make instead of making it, and a `text` reply ends the turn. Emit the tool_call itself now — read the file, then answer in a single `text` reply once you have its contents."
	}
	// A claim about a file the run never opened. This is the conversational
	// half of an invariant the write path already enforces — edit_file,
	// structural_edit, insert_after and replace_lines all refuse a path that
	// was not read first — and until now answers were exempt, because
	// "conversational messages are never gated" (see wantsStateChange).
	// Diagnostic questions are exactly where that exemption costs the most:
	// the reply IS the deliverable, and a guess is indistinguishable from an
	// answer.
	if cited := unreadFileCitations(ctx, claimText); len(cited) > 0 && s.chargeBounce("evidence_gate") {
		log.Printf("[agent] evidence gate: bouncing exit at turn %d — reply cites %v with no read (bounce %d/%d)",
			s.turn, cited, s.gateBounces["evidence_gate"], maxGateBounces)
		return "evidence_gate", unreadCitationMessage(cited)
	}
	// A warned, never-executed artifact is not a deliverable. Without this a
	// session whose prompt carried no verification wording could end with a
	// file that never parsed and was never run, reported as success (audit
	// finding: warned state must be a terminal integrity condition, not a
	// rewrite throttle).
	for p := range s.pendingWarnedRun {
		if s.chargeBounce("run_first_gate") {
			log.Printf("[agent] run-first gate at exit: %s warned and never executed (bounce %d/%d)",
				p, s.gateBounces["run_first_gate"], maxGateBounces)
			return "run_first_gate", fmt.Sprintf(
				"`%s` is on disk with a parse warning and has never been run. Run it, read the result, and fix it before finishing — as written it cannot work.", p)
		}
		break
	}
	// Verified only through a stdin redirect: the program was never run the
	// way its caller will run it. See stdinRedirectSource for the measurement.
	if s.verifiedByRedirect != "" && !s.verifiedStandalone && s.chargeBounce("contract_gate") {
		log.Printf("[agent] contract gate: every verification piped %q into stdin (bounce %d/%d)",
			s.verifiedByRedirect, s.gateBounces["contract_gate"], maxGateBounces)
		return "contract_gate", redirectOnlyVerificationMessage(s.verifiedByRedirect)
	}
	// Verification is of bytes, not of a moment. If any file this session
	// wrote no longer matches the hash snapshotted when the verifying run
	// succeeded, the final artifact is unverified whatever the booleans say.
	if s.verifiedThisLoop {
		if changed := driftedSinceVerification(ctx, s.verifiedHashes); changed != "" && s.chargeBounce("artifact_gate") {
			log.Printf("[agent] artifact gate: %s changed after the run that verified it (bounce %d/%d)",
				changed, s.gateBounces["artifact_gate"], maxGateBounces)
			s.verifiedThisLoop = false
			s.verifiedStandalone = false
			return "artifact_gate", fmt.Sprintf(
				"`%s` changed after the run that verified it, so what is on disk now has never been executed. Run it again and confirm the output before finishing.", changed)
		}
	}
	if (s.userWantsVerification || s.sawFailedVerification) && !s.verifiedThisLoop && s.chargeBounce("verification_gate") {
		log.Printf("[agent] verification gate: bouncing exit at turn %d (trigger=%s, no successful verification command this loop, bounce %d/%d)",
			s.turn, gateTrigger(s.userWantsVerification, s.sawFailedVerification), s.gateBounces["verification_gate"], maxGateBounces)
		return "verification_gate", verificationRejectionWithStreak(
			s.sawFailedVerification, s.serverStartBlocked, anyBackgroundJobID(ctx), s.redRunStreak)
	}
	// Steps the plan named and no tool call ever satisfied. Same shape as the
	// verification gate: a fact the run already holds, used at the exit
	// instead of only being shown mid-run.
	// Code the run added that nothing calls. Checked at the exit rather than
	// at the write, because wiring it up on a later turn is normal — only
	// finishing with it unwired is the defect.
	if orphans := orphanedAdditions(ctx); len(orphans) > 0 && s.chargeBounce("orphan_gate") {
		log.Printf("[agent] orphan gate: bouncing exit at turn %d — added-but-uncalled in %d file(s) (bounce %d/%d)",
			s.turn, len(orphans), s.gateBounces["orphan_gate"], maxGateBounces)
		return "orphan_gate", orphanedAdditionsMessage(orphans)
	}
	if msg := planIncompleteMessage(ctx); msg != "" && s.chargeBounce("plan_gate") {
		log.Printf("[agent] plan gate: bouncing exit at turn %d — %d/%d steps satisfied (bounce %d/%d)",
			s.turn, countTrue(ctx.PlanStepsSatisfied), len(ctx.Plan.Steps),
			s.gateBounces["plan_gate"], maxGateBounces)
		return "plan_gate", msg
	}
	if observeActionDemand(ctx, s, shadowGateActionGate,
		decideActionDemand(ctx.TaskContract, userMessage, ctx.Tier, s.inspectedWorkspace)) &&
		!s.madeProductiveChange && s.chargeBounce("action_gate") {
		log.Printf("[agent] done-without-action gate: bouncing exit at turn %d (user prompt %q wants a state change, no successful write/edit/structural_edit this loop, bounce %d/%d)",
			s.turn, truncateStr(userMessage, 60), s.gateBounces["action_gate"], maxGateBounces)
		return "action_gate", actionWithoutProductiveChangeMessage(userMessage)
	}
	if missing := missingExpectedOutputs(ctx, s.expectedOutputs); len(missing) > 0 && !s.outputGateUsed && s.chargeBounce("output_gate") {
		s.outputGateUsed = true // fire once — see field doc
		log.Printf("[agent] expected-output gate: bouncing exit at turn %d — named deliverable(s) %v not on disk (bounce %d/%d)",
			s.turn, logPaths(missing), s.gateBounces["output_gate"], maxGateBounces)
		return "output_gate", expectedOutputMissingMessage(missing)
	}
	if claimsUniversal(claimText) || promptIsMultiIssue(userMessage) {
		if gap := verifyCompletionClaims(ctx.WorkingDir); gap != "" && s.chargeBounce("claim_check") {
			log.Printf("[agent] claim-check gate: bouncing exit at turn %d (bounce %d/%d) — %q",
				s.turn, s.gateBounces["claim_check"], maxGateBounces, truncateStr(gap, 200))
			return "claim_check", gap
		}
	}
	return "", ""
}

// chargeBounce spends one of gate's bounces and reports whether it had one
// left. A gate whose budget is gone returns false so exitGates falls through
// to the next gate rather than returning early: an exhausted gate must stop
// repeating itself, not mute the gates behind it.
func (s *runState) chargeBounce(gate string) bool {
	if s.gateBounces[gate] >= maxGateBounces {
		return false
	}
	if s.gateBounces == nil {
		s.gateBounces = make(map[string]int, 4)
	}
	s.gateBounces[gate]++
	return true
}

// fetchPatternContext asks the lens pattern-cache reader
// (/internal/patterns/context) for lessons from previous sessions whose
// pattern type matches the user message, and formats them as one
// "[system note]:" block (≤3 patterns, one "- [type] summary" line each,
// hard-capped at 600 chars). Strictly fail-soft: any error, timeout, or
// empty result returns ("", nil) and the agent loop proceeds without the
// block — the lens being down must never cost a turn or spam the log.
func fetchPatternContext(ctx *AgentContext, userMessage string) (string, []string) {
	if ctx.LensURL == "" || strings.TrimSpace(userMessage) == "" {
		return "", nil
	}
	body, err := json.Marshal(map[string]interface{}{
		"task": userMessage, "top_k": 3,
	})
	if err != nil {
		return "", nil
	}
	reqCtx, cancel := context.WithTimeout(ctx.Ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST",
		ctx.LensURL+"/internal/patterns/context", bytes.NewReader(body))
	if err != nil {
		return "", nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", nil
	}
	var r struct {
		Patterns []struct {
			Summary string `json:"summary"`
			Type    string `json:"type"`
		} `json:"patterns"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil || len(r.Patterns) == 0 {
		return "", nil
	}
	const blockCap = 600
	b := "[system note]: lessons from previous ATLAS sessions on similar tasks:"
	types := make([]string, 0, 3)
	for i, p := range r.Patterns {
		if i >= 3 {
			break
		}
		line := "\n- [" + p.Type + "] " + truncateStr(p.Summary, 160)
		if len(b)+len(line) > blockCap {
			break
		}
		b += line
		types = append(types, p.Type)
	}
	if len(types) == 0 {
		return "", nil
	}
	return b, types
}

func runAgentLoop(ctx *AgentContext, userMessage string) error {
	// One snapshot per validated request, before any turn runs. Only the
	// immutable inputs: the live decision belongs to the gate records.
	emitShadowRequestSnapshot(ctx, userMessage)
	// Capture the human's actual instruction before the loop appends
	// anything: correctives, manifests and re-injected content all ride
	// user-role messages, and everything downstream that needs "what was
	// I asked" (the V3 bridge above all) must not confuse those with this.
	ctx.HumanTask = userMessage
	ctx.LiteralBlocks = extractLiteralBlocks(userMessage)
	if n := len(ctx.LiteralBlocks); n > 0 {
		log.Printf("[agent] %d literal content contract(s) extracted from the request", n)
	}
	// Emit a stage_start envelope so the TUI's pipeline pane shows
	// the agent is working. Mirrors the typed-event broker.
	loopStart := time.Now()
	Emit(NewEnvelope(EvtStageStart, "agent", map[string]interface{}{
		"detail": fmt.Sprintf("tier=%s msg=%q", ctx.Tier,
			truncateStr(userMessage, 80)),
	}))
	defer func() {
		// Close the "agent" stage so the pipeline pane stops showing it
		// running. Without this, the TUI's pipelineState.apply only ever
		// sees EvtDone (overall finish) and the agent row is stuck in
		// Running() forever — visually misleading after the turn ended.
		dur := time.Since(loopStart).Milliseconds()
		// The broker said "success": true on every terminal, including every
		// stop. It now reports the session's actual outcome, from the same
		// field the SSE terminal used, so the two streams cannot disagree.
		// `success` keeps its key and its type for existing readers.
		status := ctx.TerminalStatus
		if !status.Classified() {
			status = TerminalIncomplete
		}
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtStageEnd,
			Stage:      "agent",
			DurationMS: dur,
			Payload: map[string]interface{}{
				"success":       status.Completed(),
				"status":        string(status),
				"reason":        ctx.TerminalReason,
				"total_tokens":  ctx.TotalTokens,
				"fenced_calls":  ctx.FencedCalls,
				"fenced_tokens": ctx.FencedTokens,
			},
		})
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtDone,
			Stage:      "agent",
			DurationMS: dur,
			Payload: map[string]interface{}{
				"success":           status.Completed(),
				"status":            string(status),
				"reason":            ctx.TerminalReason,
				"total_duration_ms": dur,
				"total_tokens":      ctx.TotalTokens,
				"fenced_calls":      ctx.FencedCalls,
				"fenced_tokens":     ctx.FencedTokens,
			},
		})
	}()

	// Pre-flight plan generation. Runs BEFORE buildSystemPrompt so
	// the system prompt can reference the planned steps — the model
	// gets explicit guidance on what to do first instead of having
	// to infer it from the user message alone. Skipped for trivial
	// chat / acks where the ~5-15s cost isn't worth it. Failures
	// degrade silently — the loop runs without adherence gating.
	if shouldGeneratePlan(ctx, userMessage) {
		if plan := generatePlan(ctx, userMessage); plan != nil {
			ctx.Plan = plan
			log.Printf("[agent] plan: %d steps, verify=%s, score=%.2f",
				len(plan.Steps), plan.VerifyStep, plan.WinningScore)
		}
	}

	// Build system prompt with tool descriptions, project context,
	// and (when present) the planned steps.
	systemPrompt := buildSystemPrompt(ctx)

	// Initialize messages: system prompt, then any prior-turn history
	// the TUI shipped, then the new user message. PriorHistory is
	// already filtered to role=user|assistant text turns (no tool
	// calls/results, no system spam) on the TUI side. Without this,
	// every user message starts a fresh agent loop and the model can't
	// answer follow-ups like "what did you just delete?".
	// Refuse to start on a split workspace. The proxy writing to one host
	// directory while the sandbox that runs commands is bound to another is
	// invisible to every /health, and the session that follows is worse than
	// useless: files land, `run_command` reports them missing, and the model
	// spends its turns concluding its own work does not exist. Cached with a
	// TTL, so this is one probe per session at most.
	if problem := verifyWorkspaceAlignment(ctx); problem != "" {
		log.Printf("[agent] refusing to start — proxy and sandbox workspaces are not aligned")
		emitTerminal(ctx, nil, TerminalFailed, "workspace_misaligned", problem)
		return nil
	}

	ctx.Messages = make([]AgentMessage, 0, 3+len(ctx.PriorHistory))
	ctx.Messages = append(ctx.Messages, AgentMessage{Role: "system", Content: systemPrompt})
	ctx.Messages = append(ctx.Messages, ctx.PriorHistory...)

	// GH #39 point 4: auto-inject reachability slice. If the user
	// message names project symbols (`dashboard`, "the foo function",
	// foo.bar.baz), pre-load their definitions so the model doesn't
	// burn agent turns on read_file/list_directory recon. Fail-soft —
	// no v3-service / no symbols / no project files / network error
	// all degrade silently to the original message-only flow.
	if symbols := extractCandidateSymbols(userMessage); len(symbols) > 0 {
		fileMap := walkPythonFiles(ctx.WorkingDir)
		if len(fileMap) > 0 {
			if idx, ok := resolveProjectSymbols(ctx, fileMap, symbols); ok && len(idx.Matched) > 0 {
				body := formatProjectContextMessage(idx.Matched)
				// #39 Phase 3: append the call-graph neighborhood when v3-service
				// returned it (ATLAS_CALL_GRAPH on). Empty string when absent, so
				// flag-off behavior is unchanged.
				body += formatGraphNeighborhood(idx.Graph)
				if body != "" {
					// Role MUST be "user" with a "[system note]:" prefix —
					// Some Jinja chat templates enforce "System message
					// must be at the beginning" and 500s on any system
					// role appended mid-conversation. Same convention
					// the lens-intervention path uses (commit b79b31d).
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:    "user",
						Content: "[system note]: " + body,
					})
					names := make([]string, 0, len(idx.Matched))
					for _, m := range idx.Matched {
						names = append(names, m.Name)
					}
					log.Printf("[symbol_index] injected %d snippet(s) for [%s] from %d project files",
						len(idx.Matched), strings.Join(names, ", "), len(fileMap))
					ctx.Stream("symbol_index_injected", map[string]interface{}{
						"matched": names,
						"n_files": len(fileMap),
						"skipped": len(idx.Skipped),
					})
				}
			}
		}
	}

	// Pattern-cache context: lessons from previous sessions whose pattern
	// type matches this task, served by the lens reader. Same user-role
	// "[system note]:" convention as the symbol injection above. Fail-soft:
	// an empty block means no message and no event.
	if block, types := fetchPatternContext(ctx, userMessage); block != "" {
		ctx.Messages = append(ctx.Messages, AgentMessage{
			Role:    "user",
			Content: block,
		})
		log.Printf("[pattern_context] injected %d pattern(s) [%s]",
			len(types), strings.Join(types, ", "))
		ctx.Stream("pattern_context_injected", map[string]interface{}{
			"count": len(types),
			"types": types,
		})
	}

	ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: userMessage})

	// Per-session cache scope. llama.cpp's KV slot persists between
	// requests by default — that's what keepLlamaWarm relies on. But the
	// slot also persists *across user sessions*, so context from a previous
	// session's conversation can bias the next session (the
	// `show_greeting.py` hallucination from the 2026-04-30 snake test was
	// likely an example). Erase slot 0 at the start of each agent loop call.
	// llama.cpp re-encodes the system prompt from scratch (~1-2s on a
	// warm GPU); the per-turn cache benefit within the session is preserved.
	// Disable with ATLAS_FRESH_SLOT_PER_SESSION=0.
	if envOr("ATLAS_FRESH_SLOT_PER_SESSION", "1") != "0" && !ctx.DisableFreshSlot {
		eraseLlamaSlot(ctx)
	}

	consecutiveReads := 0 // Track consecutive read-only calls
	consecutiveErrors := 0
	totalFailures := 0 // Track consecutive tool failures to break error loops
	// edit_file old_str-mismatch failures per path. A successful read_file
	// between attempts resets consecutiveErrors/RecentFailurePaths, which
	// masks the classic read→edit-miss→read loop (smaller models can't
	// reproduce old_str byte-for-byte). This counter survives interleaved
	// reads so we can force the structural_edit steer after the second miss.
	editMissByPath := map[string]int{}
	repeatDetections := 0 // hard-stop after the 2nd repeated-identical-call detection
	// Runaway backstop for content-varying write loops (#147 review finding
	// #14): the content-fingerprint repeat detector, by design, does not
	// catch a model that rewrites one file with materially different content
	// every time and never converges. This counts total writes per path and
	// escalates to the repeat corrective only at a threshold far above any
	// realistic iteration (polyglot's healthiest run was ~10), so it stops a
	// true runaway without regressing legitimate iteration.
	writeCountByPath := map[string]int{}
	const runawayWriteThreshold = 20
	// Exit-gate evidence + the shared bounce shape live on runState (see
	// its field docs); the remaining counters are loop-local.
	st := &runState{
		expectedOutputs:       expectedOutputPaths(userMessage),
		userWantsVerification: isFixIntentMessage(userMessage),
	}
	// One-shot: when a loop-stop is about to fire but the task's named
	// deliverable was never written, steer toward it once instead of
	// stopping (many hard tasks loop on run_command exploration and
	// hard-stop without ever reaching the done/text exit where the
	// expected-output gate lives — observed on sqlite and merge-diff).
	outputRescueUsed := false

	// Flag whether we've already injected the approaching-budget hint,
	// so we don't fire it every turn after crossing the threshold.
	budgetHintFired := false

	// A tool_call is streamed the moment it parses, before permission and
	// execution, so every early exit between that point and the
	// tool_result leaves the client holding a call that never resolves —
	// a spinner with nothing coming. Observed 2026-08-03 on
	// multiturn_stats: the repetition breaker stopped the session one
	// line after announcing a call, and the stream carried 12 tool_call
	// events against 11 tool_result.
	//
	// endStream answers the outstanding call before the summary, so the
	// invariant holds at every exit rather than at each one that
	// remembered to.
	endStream := func(status TerminalStatus, reason, summary string) {
		emitTerminal(ctx, st, status, reason, summary)
	}

	// Several branches refuse a call before dispatch and continue. Every one of
	// them is a failed call the model may ignore forever, so each has to reach
	// the same path-aware accounting an executed failure reaches -- the raw
	// pre-resolution intent, the canonical target, the counters, and the
	// bounded failure policy that reads them. Returns true when the caller must
	// stop; the caller keeps its own diagnostic, which is unchanged.
	//
	// A run that ended because the client left or the deadline fired is not a
	// model repeating itself, and those paths keep their own terminals.
	accountRefusedCall := func(name string, intent json.RawMessage, rejection, failPath string) bool {
		if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
			consecutiveErrors++
			return false
		}
		recordFailedToolCall(ctx, name, intent, rejection)
		consecutiveErrors++
		totalFailures++
		ctx.RecentFailurePaths = appendRecentFailurePath(ctx.RecentFailurePaths, failPath)
		if !shouldStopForFailures(totalFailures, consecutiveErrors, ctx.RecentFailurePaths) {
			return false
		}
		log.Printf("[agent] breaking at turn %d: %d refused/failed calls, %d consecutive on %q",
			st.turn, totalFailures, consecutiveErrors, failPath)
		endStream(TerminalStopped, "repeated_refusal",
			repeatedRefusalSummary(name, failPath, st.madeProductiveChange)+
				liveBackgroundJobNote(ctx))
		return true
	}

	for turn := 0; ctx.MaxTurns <= 0 || turn < ctx.MaxTurns; turn++ {
		st.turn = turn
		// Budget hint — only relevant when there IS a turn cap.
		// May 10 2026: T1/T2/T3 default to uncapped (ctx.MaxTurns == 0),
		// so the hint is mostly dormant unless an operator explicitly
		// sets ATLAS_MAX_TURNS. T0 still hits a hint at turn 4 if a
		// conversational request unexpectedly tries to loop.
		if !budgetHintFired && ctx.MaxTurns > 0 && turn > 0 && turn*5 >= ctx.MaxTurns*4 {
			budgetHintFired = true
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role: "system",
				Content: fmt.Sprintf(
					"Turn budget notice: you're at turn %d of %d. If significant work remains, prioritize finishing the highest-impact items and verifying them — do not start new exploration. If you can finish in the remaining turns, keep going. If you cannot, summarize what's done and what's not in your `done` summary so the user knows what to follow up on.",
					turn, ctx.MaxTurns),
			})
		}

		// Bail out fast if the upstream request was cancelled (the client closed the
		// connection, user hit Ctrl-C, terminal exited). Without this check the
		// loop would keep grinding LLM calls and tool work for a client that's
		// already gone, burning GPU.
		if ctx.Ctx != nil {
			select {
			case <-ctx.Ctx.Done():
				return finishCancelledRun(ctx, st, turn)
			default:
			}
		}

		// Trim conversation history if it gets too long (prevent context overflow).
		// Keep system + most-recent-user-instruction + last 8 messages.
		//
		// Pinning the most recent user message is critical: long agent loops
		// (5+ tool calls) push the user's task beyond the trim window, and
		// the next LLM call sees only system + tool exchanges. Model has no
		// instruction to work from and goes generic ("Hi! I'm ATLAS...").
		// Hardcoding ctx.Messages[1] as the user msg used to work, but
		// PriorHistory makes that index a prior-turn message instead — so
		// scan backwards for the actual current-turn user role.
		// Trim by TOKEN BUDGET, not a blind message count. The
		// old `> 12 messages → keep 8` rule dropped a just-read file after
		// a couple of turns even when the prompt was a fraction of the
		// context window — the model would then re-read in a loop, saying
		// "I don't see the output in the history". keepLast is now derived
		// from how many recent messages actually fit the per-slot budget,
		// floored at 8 so we never trim more aggressively than before.
		trimmed := false
		if keep := budgetedKeepLast(ctx.Messages); keep < len(ctx.Messages)-1 {
			ctx.Messages = trimMessages(ctx.Messages, keep)
			trimmed = true
			log.Printf("[agent] trimmed conversation to %d messages (token-budget)", len(ctx.Messages))
		}

		// Per-turn streaming visibility: announce the start of the turn,
		// then the LLM call boundaries. Without these the TUI sees a 10-30s
		// gap between tool_result and the next tool_call while the model
		// is generating — looks like a hang.
		ctx.Stream("turn_start", map[string]interface{}{
			"turn":     turn,
			"messages": len(ctx.Messages),
			"trimmed":  trimmed,
		})
		// Estimate prompt tokens up front (chars/4 — works for English
		// + code, off by maybe 10–20%) so the TUI can pre-fill its
		// context-utilization gauge while llama-server is still doing
		// prompt eval. Authoritative count arrives in llm_call_end.
		promptTokenEst := 0
		for _, mm := range ctx.Messages {
			promptTokenEst += len(mm.Content) / 4
		}
		ctx.Stream("llm_call_start", map[string]interface{}{
			"turn":          turn,
			"messages":      len(ctx.Messages),
			"prompt_tokens": promptTokenEst,
		})
		Emit(NewEnvelope(EvtStageStart, "llm",
			map[string]interface{}{"turn": turn, "messages": len(ctx.Messages)}))
		llmStart := time.Now()

		// Call LLM with grammar constraint
		response, tokens, err := callLLMConstrained(ctx)
		llmElapsed := time.Since(llmStart)
		if err != nil {
			ctx.Stream("llm_call_end", map[string]interface{}{
				"turn":         turn,
				"tokens":       0,
				"total_tokens": ctx.TotalTokens,
				"ms":           llmElapsed.Milliseconds(),
				"error":        err.Error(),
			})
			Emit(Envelope{
				EventID:    NewEventID(),
				Timestamp:  float64(time.Now().UnixNano()) / 1e9,
				Type:       EvtStageEnd,
				Stage:      "llm",
				DurationMS: llmElapsed.Milliseconds(),
				Payload: map[string]interface{}{
					"success": false, "error": err.Error(),
				},
			})
			Emit(NewEnvelope(EvtError, "llm",
				map[string]interface{}{"message": err.Error()}))
			ctx.Stream("error", map[string]string{"error": err.Error()})
			// A call that failed BECAUSE the work context ended is not an
			// inference failure: the deadline is ours, and reporting the
			// symptom would hide the cause and skip finalisation.
			if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
				return finishCancelledRun(ctx, st, turn)
			}
			// An `error` event is not an outcome. This exit streamed one and
			// returned, so the client saw a tool call, an error, and then
			// nothing — aoc_sonar died here in BOTH reps on a context-size
			// 400 and the user got silence. Every other exit in this loop
			// authors a `done`; this one has to as well, or the failure is
			// invisible to anything rendering the stream.
			emitTerminal(ctx, st, TerminalFailed, "inference_failed",
				inferenceFailureSummary(err, st.madeProductiveChange)+liveBackgroundJobNote(ctx))
			return fmt.Errorf("LLM call failed on turn %d: %w", turn, err)
		}
		ctx.TotalTokens += tokens
		st.response = response
		ctx.Stream("llm_call_end", map[string]interface{}{
			"turn":         turn,
			"tokens":       tokens,
			"total_tokens": ctx.TotalTokens,
			"ms":           llmElapsed.Milliseconds(),
			"chars":        len(response),
		})
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtStageEnd,
			Stage:      "llm",
			DurationMS: llmElapsed.Milliseconds(),
			Payload: map[string]interface{}{
				"success":      true,
				"tokens":       tokens,
				"total_tokens": ctx.TotalTokens,
			},
		})
		Emit(NewEnvelope(EvtMetric, "llm", map[string]interface{}{
			"name": "total_tokens", "value": ctx.TotalTokens,
		}))

		// Parse the response — extract JSON even if model added surrounding text
		parsed, parseErr := extractModelResponse(response)
		if parseErr != nil {
			// Classify the failure shape once: a category for the log so
			// docker logs reads "what kind of broken" at a glance, and
			// targeted feedback for the model — generic "your response
			// wasn't JSON" led to the May 2026 user-session bug where the
			// model retried the same 1100-char edit_file with a giant
			// old_str 5 times in a row. The response was being truncated
			// at the llama-server token cap; the model couldn't see that
			// and kept emitting the same too-big payload.
			// A text answer we cut mid-string is still an answer. Deliver what
			// was written rather than nothing — the alternative is a user who
			// asked a question and received silence.
			if ctx.LastStreamCut == "content_loop" {
				if salvaged, ok := recoverTruncatedText(response); ok {
					log.Printf("[agent] salvaged %d chars of a cut text answer at turn %d", len(salvaged), turn)
					ctx.Stream("text", map[string]string{"content": salvaged})
					// The salvaged text is the model's own words and already
					// reached the client as a `text` event. Repeating it in
					// the summary of an INCOMPLETE terminal is how half-written
					// code came to read as the answer, so the summary describes
					// what happened instead of restating it.
					salvageSummary := "The reply was cut short — it had begun repeating itself, " +
						"so what arrived above is partial. Ask again if something is missing."
					// Salvage is a third exit, alongside done and text, and it
					// reached the user without the honesty the other two apply.
					// Observed on aoc_slope rep2 and smallrung_toml rep2 (run
					// 16): the cut reply was half-written code, so the run
					// finished by handing back code that reads like the answer
					// while nothing was on disk.
					if st.actionDemandedAndUnmet(ctx, userMessage) {
						log.Printf("[agent] salvaged text at turn %d with nothing written — saying so", turn)
						salvageSummary = nothingWrittenSummary(salvageSummary)
					}
					emitTerminal(ctx, st, TerminalIncomplete, "text_instead_of_work",
						salvageSummary+liveBackgroundJobNote(ctx))
					return nil
				}
			}
			category, feedback := classifyParseFailure(response, ctx.LastStreamCut)
			log.Printf("[agent] parse error: %v | category=%s raw_len=%d | raw: %q",
				parseErr, category, len(response), truncateStr(response, 500))
			ctx.Stream("error", map[string]string{
				"error":    "failed to parse model response",
				"category": category,
			})
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "user",
				Content: feedback,
			})
			// Cap parse failures the same way we cap tool failures.
			// Five identical parse errors in a row is a stuck loop;
			// bailing keeps us from burning 6 more LLM round-trips.
			consecutiveErrors++
			if consecutiveErrors >= 3 {
				log.Printf("[agent] breaking parse-error loop at turn %d (%d consecutive)", turn, consecutiveErrors)
				summary := "Stopped after 3 unparseable responses — the model's tool calls keep " +
					"getting truncated. Try a more targeted request (e.g. 'edit just the " +
					"@app.route(\"/product\") handler in app.py') so the response stays under the " +
					"token cap."
				if ctx.LastStreamCut == "content_loop" {
					// Same misdiagnosis the classifier used to make: the token cap
					// had nothing to do with it. The model began repeating itself
					// and the proxy cut the stream, so "make the request smaller"
					// is advice the user cannot act on.
					summary = "Stopped: the model began repeating itself and its response was cut " +
						"off mid-call, three times. This usually means it tried to reproduce a " +
						"large block of data — the contents of an input or fixture file — " +
						"instead of writing code that reads it. Ask again and say explicitly " +
						"that the data file should be read at runtime, not rewritten."
				}
				emitTerminal(ctx, st, TerminalStopped, "unusable_model_output", summary)
				return nil
			}
			continue
		}

		// Log the args truncated — enables diagnosing failures like
		// "all 3 tool calls returned Success=false" without having to add
		// breakpoints.
		logEvent("info",
			fmt.Sprintf("[agent] turn=%d type=%s name=%s args=%s",
				turn, parsed.Type, parsed.Name, truncateStr(string(parsed.Args), 200)),
			requestIDFromContext(ctx.Ctx), nil)

		// When a tool_call still has no args after liftMissingArgs,
		// log the raw model output so we can see exactly what shape was
		// emitted — helps catch new alt-shapes the lift logic missed.
		if parsed.Type == "tool_call" && (len(parsed.Args) == 0 || string(parsed.Args) == "null") {
			log.Printf("[agent] turn=%d EMPTY ARGS — raw model output: %q", turn, truncateStr(response, 500))
		}

		switch parsed.Type {
		case "done":
			// The four honesty gates (see runState.exitGates): a done that
			// the run's own evidence contradicts is bounced, capped.
			if gate, rejection := st.exitGates(ctx, userMessage, parsed.Summary); gate != "" {
				st.bounce(ctx, gate, rejection)
				continue
			}
			// A model saying it is finished is a claim, not a demonstration.
			// The decision comes FIRST, because it decides whose words the
			// user reads: the model's account is only repeated where the
			// exact-hash gate authorised the completion it describes.
			status, reason := finalizeCompletion(ctx, st, userMessage, "")
			if reason == "unresolved_mutation_debt" {
				if msg := offerDebtRecovery(ctx, st); msg != "" {
					st.bounce(ctx, "done", msg)
					continue
				}
			}
			summary := modelProseIfAuthorized(status, parsed.Summary)
			if reason == "unresolved_mutation_debt" {
				summary = unresolvedDebtSummary(st)
			}
			// Past the gates, but the verification gate can be past because
			// it ran out of bounces rather than because anything verified.
			if st.verificationDemandedAndUnmet() {
				log.Printf("[agent] done at turn %d with no passing verification — replacing the summary", turn)
				summary = unverifiedSummary(st.madeProductiveChange,
					modelProseIfAuthorized(status, parsed.Summary))
			}
			// Same reasoning one gate over: the action gate can be past
			// because its bounces ran out, not because anything was written.
			if st.actionDemandedAndUnmet(ctx, userMessage) {
				log.Printf("[agent] done at turn %d with nothing written — saying so", turn)
				summary = nothingWrittenSummary(summary)
			}
			emitTerminal(ctx, st, status, reason, summary+liveBackgroundJobNote(ctx))
			return nil

		case "text":
			// `text` is the agent's user-facing chat answer. End the turn
			// here — the user gets one reply per message they send, and can
			// follow up to continue. Looping after text caused two failures
			// in earlier revisions: a trailing role=assistant tripped
			// llama-server's "prefill incompatible with enable_thinking"
			// 400, and with a "continue" nudge the model would rabbit-hole
			// into nonsense tool_calls on conversational input.
			//
			// text is otherwise an UNGATED exit, and on action-intent
			// prompts models abandon work through it ("I will now proceed
			// to sanitize the credentials" — then session over, zero
			// edits). So the same gates as done run here,
			// with the narration as the completion claim. Chat replies
			// still exit cleanly: wantsStateChange requires action-intent
			// wording or an opened project, and a chat reply has neither.
			if gate, rejection := st.exitGates(ctx, userMessage, parsed.Content); gate != "" {
				st.bounce(ctx, gate, rejection)
				continue
			}
			ctx.Stream("text", map[string]string{"content": parsed.Content})
			textSummary := ""
			if st.actionDemandedAndUnmet(ctx, userMessage) {
				log.Printf("[agent] text exit at turn %d with nothing written — saying so", turn)
				textSummary = nothingWrittenSummary("")
			}
			// A text reply carries no file obligation of its own; when the
			// run also wrote something, or was asked to change something and
			// did not, the same demonstration is required. Same decision as
			// the done exit, so the two cannot drift.
			textStatus, textReason := finalizeCompletion(ctx, st, userMessage, "text_reply")
			if textReason == "unresolved_mutation_debt" {
				if msg := offerDebtRecovery(ctx, st); msg != "" {
					st.bounce(ctx, "text", msg)
					continue
				}
				textSummary = unresolvedDebtSummary(st)
			}
			emitTerminal(ctx, st, textStatus, textReason, textSummary)
			return nil

		case "tool_call":
			st.toolsRun++
			st.pendingToolCall = parsed.Name
			// The repetition detector has to judge what the MODEL sent, and
			// `parsed.Args` does not stay that. Fenced resolution rewrites it
			// with the fetched file body further down, so by the time the
			// detector runs it is fingerprinting bytes the model never wrote --
			// bytes that differ on every attempt, while the call itself is
			// byte-identical each time. That is why the seven-turn @fenced
			// loops in the frozen run reached the 600s cap with the detector
			// silent: the instrument built to catch exactly that repetition
			// was reading the output of the channel that resolved it.
			//
			// Snapshotting here, before anything can rewrite it, is the whole
			// fix. The copy is used for the signature and nothing else: the
			// fetched bytes still drive the mutation, the gates, the ledger
			// and the write, exactly as before.
			intentArgs := append(json.RawMessage(nil), parsed.Args...)
			ctx.Stream("tool_call", map[string]interface{}{
				"name": parsed.Name,
				"args": json.RawMessage(parsed.Args),
				"turn": turn,
			})
			Emit(NewEnvelope(EvtToolCall, "tool", map[string]interface{}{
				"name":         parsed.Name,
				"args_summary": truncateStr(string(parsed.Args), 80),
				"turn":         turn,
			}))

			// Check permissions. In default and accept-edits modes a
			// destructive tool pauses the loop until the client approves or
			// denies it (via POST /v1/permission). Yolo mode and pre-approved
			// tools short-circuit needsPermission and never reach here. The
			// legacy PermissionFn is still honored for non-interactive callers.
			//
			// Deletion runs this LATER -- see below. It has to be asked after
			// the path is canonicalised and its structured intent is on the
			// record, or a refusal leaves nothing owed and a run whose only
			// act was a refused deletion reports completed.
			permissionGate := func() int { // 0 proceed, 1 continue, 2 stop
				if !needsPermission(ctx, parsed.Name, parsed.Args) {
					return 0
				}
				{
					allowed := true
					// A deletion always goes through the interactive handshake.
					// PermissionFn is a programmatic approver -- yolo installs one
					// that says yes to everything -- and a function returning true
					// is not a user deciding about a file. Routing deletion around
					// it is what makes the yolo bypass actually closed rather than
					// closed-looking: needsPermission alone would just hand the
					// call to that function.
					if ctx.PermissionFn != nil && parsed.Name != "delete_file" {
						allowed = ctx.PermissionFn(parsed.Name, parsed.Args)
					} else {
						allowed = awaitPermission(ctx, parsed.Name, permCallID(turn), parsed.Args)
					}
					if !allowed {
						ctx.Stream("permission_denied", map[string]string{
							"tool": parsed.Name,
						})
						// A refusal here is a failed call like any other, and this
						// branch returned before everything that counts one. It
						// matters more now that a deletion always reaches the
						// handshake: a model repeating a delete of a path the
						// preflight refuses gets denied every time, and without
						// accounting it repeats until the turn cap -- measured at
						// 21 turns with no terminal of ATLAS's own.
						if stop := accountRefusedCall(parsed.Name, intentArgs,
							"permission denied by user",
							workspaceRefusalPath(ctx, parsed.Name, parsed.Args)); stop {
							return 2
						}
						// A denied call still produced a result the model reads,
						// so it owes the stream one too. Without this the run
						// emits more tool_calls than tool_results, which every
						// balance check treats as a dropped call -- invisible
						// until deletion started always routing through here.
						// The call is answered, so it is no longer pending; the
						// terminal must not flush a second "not run" result for it.
						st.pendingToolCall = ""
						ctx.Stream("tool_result", map[string]interface{}{
							"tool":    parsed.Name,
							"success": false,
							"data":    json.RawMessage("null"),
							"error":   "permission denied by user",
							"elapsed": "0s",
						})
						// Bespoke bounce: the permission flow keys its tool-call
						// ID via permCallID so the TUI can match the decision.
						ctx.Messages = append(ctx.Messages, AgentMessage{
							Role:    "assistant",
							Content: response,
						})
						ctx.Messages = append(ctx.Messages, AgentMessage{
							Role:       "tool",
							Content:    `{"success":false,"error":"permission denied by user"}`,
							ToolCallID: permCallID(turn),
							ToolName:   parsed.Name,
						})
						return 1
					}
					return 0
				}
			}
			if parsed.Name != "delete_file" {
				switch permissionGate() {
				case 1:
					continue
				case 2:
					return nil
				}
			}

			// Fix C: Detect truncated args BEFORE execution.
			// If the args JSON doesn't parse, don't attempt execution —
			// tell the model to use smaller edits instead.
			if parsed.Name == "write_file" || parsed.Name == "edit_file" || parsed.Name == "run_command" {
				var testParse map[string]interface{}
				if err := json.Unmarshal(parsed.Args, &testParse); err != nil {
					log.Printf("[agent] truncated args detected for %s at turn %d", parsed.Name, turn)
					st.bounceToolCall(ctx, parsed.Name, "Your output was truncated — the content is too long for a single tool call. For existing files, use edit_file with small targeted changes (replace specific functions or sections). For new files, keep them under 100 lines per write_file call.")
					consecutiveErrors++
					if consecutiveErrors >= 3 {
						endStream(TerminalStopped, "oversized_tool_content",
							"Stopped: content too large for tool calls. Try requesting smaller, targeted changes.")
						return nil
					}
					continue
				}
			}

			// Enforce the workspace boundary before any pre-execution gate reads
			// a path. executeToolCall repeats this check for parallel dispatch.
			if rejection := validateToolWorkspacePaths(parsed.Name, parsed.Args, ctx); rejection != "" {
				st.bounceToolCall(ctx, parsed.Name, rejection)
				// A refusal here is a failed call, and this branch returned
				// before everything that counts one -- including the only
				// reader of the counter it incremented. It is the fourth
				// early-refusal branch with that shape, after the per-path
				// ban, the retry ban and the fenced bounce.
				//
				// Measured: a session whose workspace root did not exist
				// refused the same byte-identical find_file 60 times over its
				// whole budget, with no intervention, no log line and no
				// terminal of its own.
				if accountRefusedCall(parsed.Name, intentArgs, rejection,
					workspaceRefusalPath(ctx, parsed.Name, parsed.Args)) {
					return nil
				}
				continue
			}

			// The intent is on the record from here: the arguments parse, the
			// path is inside the workspace, and permission was granted. Every
			// later refusal -- a gate, a failed fenced resolution, a failed
			// write -- leaves the debt standing, which is the whole point:
			// the pre-dispatch failures are exactly the ones that used to
			// disappear.
			noteMutationIntent(ctx, st, parsed.Name, parsed.Args)

			// Deletion asks here, not with the others. By this point the path
			// has parsed, canonicalised and cleared the workspace boundary,
			// and the structured intent is recorded -- so a refusal, a denial,
			// a timeout or a cancel all leave the debt standing and the run
			// cannot call itself finished. Asking earlier meant a refused
			// deletion owed nothing: a session whose only act was trying to
			// remove a non-empty directory reported completed.
			if parsed.Name == "delete_file" {
				switch permissionGate() {
				case 1:
					continue
				case 2:
					return nil
				}
			}

			// Surgical-edit gate: reject write_file on existing files
			// outright. write_file is for *creating* files; edits to an
			// existing file must use edit_file with old_str/new_str.
			//
			// The gate originally only blocked near-rewrites
			// (>= 70% line overlap) or >100-line writes. That left a
			// hole: a *complete* rewrite of a 90-line template (low
			// overlap, under the size cap) would slip through and
			// destroy the original. Hardened to reject every write
			// against an existing path. Trivially-small files (<= 5
			// lines, e.g. a single-line config) are still allowed
			// because there's no edit-vs-rewrite distinction at that
			// size — anything below that is faster to overwrite than
			// to surgically edit.
			if parsed.Name == "edit_file" || parsed.Name == "structural_edit" ||
				parsed.Name == "insert_after" || parsed.Name == "replace_lines" {
				var ed struct {
					Path string `json:"path"`
				}
				if json.Unmarshal(parsed.Args, &ed) == nil && st.pendingWarnedRun[ed.Path] &&
					st.chargeBounce("run_first_gate") {
					log.Printf("[agent] run-first gate (%s): %s has a warned, unexecuted version on disk (bounce %d/%d)",
						parsed.Name, ed.Path, st.gateBounces["run_first_gate"], maxGateBounces)
					st.bounceToolCall(ctx, parsed.Name, fmt.Sprintf(
						"The version of %s on disk carries a parse warning and has never been run. Run it first — `python3 %s` — and read the real error before editing further.",
						ed.Path, ed.Path))
					continue
				}
			}
			// write_file preflight. Order matters: the run-first gate is
			// checked BEFORE fenced resolution, because "@fenced" resolution
			// costs a full model sub-call — paying that for a write the gate
			// is about to bounce is pure waste (audit finding: the gate sat
			// after resolution and every bounced warned-file rewrite burned
			// a generation first).
			if parsed.Name == "write_file" {
				var wfInput WriteFileInput
				if json.Unmarshal(parsed.Args, &wfInput) == nil {
					if st.pendingWarnedRun[wfInput.Path] {
						// The gate has already said this once. Saying it again
						// while the same call stays available is the C5 shape:
						// in the frozen run the demand repeated until its
						// bounce budget ran out and the identical writes
						// resumed. On the recurrence the model gets what it
						// has been unable to get for itself -- the file as it
						// actually is -- and the call that made no progress is
						// held back until it changes.
						if msg := fencedRunFirstRecovery(ctx, st, wfInput.Path, wfInput.Content); msg != "" {
							st.bounceToolCall(ctx, "write_file", msg)
							continue
						}
						if st.chargeBounce("run_first_gate") {
							log.Printf("[agent] run-first gate: %s has a warned, unexecuted version on disk (bounce %d/%d)",
								wfInput.Path, st.gateBounces["run_first_gate"], maxGateBounces)
							st.bounceToolCall(ctx, "write_file", fmt.Sprintf(
								"The version of %s you wrote is on disk with a parse warning and has never been run. Run it first — `python3 %s` — and read the real error before writing again. Rewriting blind is how the last four attempts went nowhere.",
								wfInput.Path, wfInput.Path))
							continue
						}
					}
					// Fenced-content resolution: everything downstream (the
					// remaining gates, tier classification, execution) must see
					// real bytes. "@fenced" is the model routing the file body
					// around the JSON channel — one unconstrained sub-call
					// fetches it in a fenced block, its native emission format.
					// See fetchFencedContent.
					trimmed := strings.TrimSpace(wfInput.Content)
					// A call that cannot execute must not open the channel.
					// Falling through leaves the sentinel in `content` and
					// hands the call to the tool, which refuses it with the
					// same check that refused it here -- so the model gets the
					// authoritative message, the ledger sees MutationNone, and
					// the session spends one turn instead of a generation.
					fencedUsable, fencedWhy := fencedCallIsExecutable("write_file", parsed.Args, ctx)
					if !fencedUsable {
						log.Printf("[agent] not opening the fenced channel for an unusable write_file call: %s", fencedWhy)
					}
					// The channel can be spent while the model keeps asking
					// for it. Offer the way out ONCE, before anything tries
					// another resolution, so the turn costs no generation.
					if fencedUsable && strings.HasPrefix(trimmed, "@fenced") {
						if msg := fencedChannelRecovery(ctx, st, wfInput.Path); msg != "" {
							st.bounceToolCall(ctx, "write_file", msg)
							continue
						}
					}
					if fencedUsable && strings.HasPrefix(trimmed, "@fenced") {
						// Anything after the sentinel is the model inlining
						// the file anyway. Exactly one of two things arrived,
						// and only one of them is a file:
						//
						//   * a complete body, bare or in its own fence — use it
						//   * a body cut off mid-emission, which is precisely
						//     what the JSON channel does and the whole reason
						//     @fenced exists. Trusting that wrote truncated
						//     programs to disk: `@fenced\n```python\nprint(`
						//     had its fence line removed by the sanitizer and
						//     landed as a one-line `print(`, so the session
						//     shipped a file that could not parse and then
						//     repeated the identical write for five more turns.
						//     Measured at 6 of 20 create sessions.
						//
						// An opening fence with no closing fence is the
						// signature of the truncated case; fall back to the
						// sub-call, which is the channel that can carry it.
						inline := strings.TrimLeft(strings.TrimPrefix(trimmed, "@fenced"), "\r\n")
						if body := extractFencedContent(inline); body != "" {
							inline = body
						} else if strings.Contains(inline, "```") {
							log.Printf("[agent] inline @fenced body for %s is truncated (fence opened, never closed) — falling back to the sub-call", wfInput.Path)
							inline = ""
						}
						if inline != "" {
							wfInput.Content = inline
							log.Printf("[agent] fenced sentinel stripped for %s (%d bytes arrived inline)", wfInput.Path, len(inline))
						} else {
							fetched, ferr := fetchFencedContent(ctx, rawResponseForFence(parsed), wfInput.Path)
							if ferr != nil {
								log.Printf("[agent] fenced-content fetch failed for %s: %v", wfInput.Path, ferr)
								st.bounceToolCall(ctx, "write_file",
									"You wrote \"content\": \"@fenced\" but no fenced block followed. Either reply with the complete file in one fenced code block when asked, or re-issue write_file with the full content inline.")
								// A refusal here is a failed call like any
								// other, and this branch returned before every
								// mechanism that counts one. The same defect
								// was found at the per-path ban and at the
								// retry ban; this is the third instance, and
								// the one the frozen run paid for: debounce2
								// re-sent this call 147 times over 570s with
								// the allowance correctly refusing all but
								// four generations and nothing bounding the
								// TURNS.
								//
								// A run that ended because the client left, or
								// because the work deadline fired, is not a
								// model repeating itself, and those paths keep
								// their own terminals.
								if ctx.Ctx == nil || ctx.Ctx.Err() == nil {
									// The intent, not the resolved args: this
									// call is byte-identical every time the
									// model sends it, and the resolved body is
									// what made it look different.
									recordFailedToolCall(ctx, parsed.Name, intentArgs, ferr.Error())
									consecutiveErrors++
									totalFailures++
									failPath := fencedKey(ctx, wfInput.Path)
									ctx.RecentFailurePaths = appendRecentFailurePath(ctx.RecentFailurePaths, failPath)
									if shouldStopForFailures(totalFailures, consecutiveErrors, ctx.RecentFailurePaths) {
										log.Printf("[agent] breaking at turn %d: %d refused/failed calls, %d consecutive on %q",
											turn, totalFailures, consecutiveErrors, failPath)
										endStream(TerminalStopped, "repeated_refusal",
											repeatedRefusalSummary("write_file", wfInput.Path, st.madeProductiveChange)+
												liveBackgroundJobNote(ctx))
										return nil
									}
								}
								continue
							}
							wfInput.Content = fetched
							log.Printf("[agent] fenced content resolved for %s (%d bytes via sub-call)", wfInput.Path, len(fetched))
						}
						if rebuilt, merr := json.Marshal(wfInput); merr == nil {
							parsed.Args = rebuilt
						}
					}
				}
			}
			if parsed.Name == "write_file" {
				var wfInput WriteFileInput
				if json.Unmarshal(parsed.Args, &wfInput) == nil {
					existingPath := resolveAgentPath(ctx, wfInput.Path)
					existing, readErr := os.ReadFile(existingPath)
					if readErr != nil && !os.IsNotExist(readErr) {
						// Every existing-file protection lives inside the
						// success branch below, so a read that fails for any
						// reason other than "the file is genuinely new" silently
						// disarms all of them and the write lands unguarded.
						// A ~100-line file was replaced by three lines this way
						// with no guard log at all, and the guard has never once
						// fired in a full session log.
						log.Printf("[agent] write_file pre-check could not read %q (resolved %q): %v — existing-file guards are NOT applied to this write",
							wfInput.Path, existingPath, readErr)
					}
					if readErr == nil {
						existingLines := strings.Count(string(existing), "\n") + 1
						// Exempt corrupted files. If the existing file
						// looks like it has prose preamble or stray
						// markdown fences (sanitizeFileContent would change
						// it), the only way to clean it up is full
						// replacement. edit_file can't express "remove
						// these specific corrupted lines" cleanly; the
						// model proved this by emitting old_str = new_str
						// for 53 wall-minutes (May 6 18:30 → 19:23).
						// Allow write_file in that case and log the
						// self-heal.
						// Self-iteration carveout: if this session wrote the file
						// itself (it's not the user's code, it's the agent's
						// own draft), allow overwriting. Otherwise the agent
						// can't correct its own first-pass mistakes — the
						// May 12 multi-file failure mode where V3 wrote a
						// stub app.py, realized it needed render-module
						// wiring, and got blocked from fixing it.
						sessionOwned := ctx.SessionWrites[wfInput.Path]
						corrupted := looksCorruptedOnDisk(existingPath, string(existing))
						// Existing, never read, not ours: refuse regardless of
						// size. The >5-line rule below is about "is a surgical
						// edit cheaper than a rewrite", which is a different
						// question from "should this be replaced at all" — and
						// you cannot know a file should be replaced when you
						// have never looked at it. edit_file and
						// structural_edit already demand a read first; this
						// closes the one path that did not.
						//
						// Observed twice: given a 1-line puzzle input, the
						// model recognised the puzzle from training, wrote the
						// canonical textbook example over the real input
						// without reading it, and solved the wrong data while
						// honestly reporting "created input.txt with sample
						// data".
						if isUnreadOverwrite(ctx, existingPath, corrupted, sessionOwned) {
							rejection := fmt.Sprintf(
								"%s already exists and this session has not read it. Use read_file first: if it holds input or configuration you were given, you need its real contents, not a replacement. If you have read it and still mean to replace the whole file, use edit_file or structural_edit.",
								wfInput.Path)
							log.Printf("[agent] rejecting write_file over unread existing %q (%d lines)", wfInput.Path, existingLines)
							if r := steerRecovery(ctx, st, wfInput.Path, existingPath, true); r != "" {
								rejection = r
							}
							st.bounceToolCall(ctx, "write_file", rejection)
							// Steering, not a verdict on the work: the model
							// is being sent to a better tool. But a model that
							// ignores the steer repeats the same refused write
							// forever -- measured at 31 turns in a healthy
							// workspace with no terminal of ATLAS's own -- so
							// the ignored steer is accounted like any other
							// refused call. A model that follows it pays
							// nothing further: the next call is a different
							// action on the path and clears the state.
							if accountRefusedCall("write_file", intentArgs, rejection,
								workspaceRefusalPath(ctx, "write_file", parsed.Args)) {
								return nil
							}
							continue
						}
						if existingLines > 5 && !corrupted && !sessionOwned {
							// GH #39: when the existing file is .py or .html
							// and the model is replacing the whole thing,
							// structural_edit is the right tool — selector-based
							// node replacement, no old_str literal, no
							// truncation risk on long content. Surface
							// the option in the rejection text. edit_file
							// stays the recommendation for surgical
							// string-level changes (other file types,
							// inline tweaks).
							ext := strings.ToLower(filepath.Ext(wfInput.Path))
							structuralHint := ""
							if ext == ".py" || ext == ".html" || ext == ".htm" {
								structuralHint = " For whole-function or whole-element rewrites, prefer `structural_edit` — it takes a structural selector (e.g. `function:dashboard`, `<body>`) and the new content body, no `old_str` needed. structural_edit doesn't truncate the way edit_file can on long replacement strings."
							}
							rejection := fmt.Sprintf(
								"File %s already exists (%d lines). write_file is for creating new files, not modifying existing ones. Use edit_file with old_str/new_str to make targeted changes (read the file first if you need to confirm the exact text to replace).%s",
								wfInput.Path, existingLines, structuralHint)
							// %q quotes + escapes the path (go/log-injection).
							log.Printf("[agent] rejecting write_file for existing %q (%d lines)", wfInput.Path, existingLines)
							if r := steerRecovery(ctx, st, wfInput.Path, existingPath, false); r != "" {
								rejection = r
							}
							st.bounceToolCall(ctx, "write_file", rejection)
							// Same shape, same bound as the unread-overwrite
							// steer above.
							if accountRefusedCall("write_file", intentArgs, rejection,
								workspaceRefusalPath(ctx, "write_file", parsed.Args)) {
								return nil
							}
							continue
						}
						if existingLines > 5 {
							// Name the actual carveout — the corrupted-file
							// message on a session-owned overwrite sent a
							// loop diagnosis down the wrong path (2026-07-18).
							if corrupted {
								log.Printf("[agent] allowing write_file on corrupted %s (%d lines, sanitizer would clean it)", wfInput.Path, existingLines)
							} else {
								log.Printf("[agent] allowing write_file on session-owned %s (%d lines, self-iteration carveout)", wfInput.Path, existingLines)
							}
						}
					}
				}
			}

			// Shell-op guardrail: bounce destructive filesystem verbs in
			// run_command. The native edit_file/write_file/delete_file
			// tools are the supported mutation path — they go through
			// V3, the surgical-edit gate, and audit logging. Shell `mv`,
			// `rm`, `cp`, `find -delete` bypass all of that and led to
			// today's "agent moved templates into venv mid-task" disaster.
			// Yolo mode opts out of this for users who want the model to
			// have free rein.
			// The foreground-server redirect runs in EVERY mode. Yolo opts out
			// of the shell-mutation and working-dir gates — those are
			// permission questions, and yolo is the user saying don't ask.
			// Starting a server in the foreground is not a permission
			// question: it burns the sandbox timeout and reports a failure
			// that says nothing about the code, whatever mode you are in.
			if parsed.Name == "run_command" {
				var rc RunCommandInput
				if json.Unmarshal(parsed.Args, &rc) == nil {
					if rejection := foregroundServerRejectionWithSource(rc.Command,
						func(rel string) (string, bool) {
							data, err := os.ReadFile(filepath.Join(ctx.WorkingDir, rel))
							if err != nil {
								return "", false
							}
							return string(data), true
						}); rejection != "" {
						log.Printf("[agent] redirecting a foreground server start to run_background: %q",
							truncateStr(rc.Command, 80))
						st.bounceToolCall(ctx, "run_command", rejection)
						continue
					}
				}
			}
			if parsed.Name == "run_command" && !ctx.YoloMode {
				var rc RunCommandInput
				if json.Unmarshal(parsed.Args, &rc) == nil {
					if rejection := validateRunCommand(rc.Command, ctx.WorkingDir); rejection != "" {
						// %q on rejection too: validateRunCommand may embed
						// fragments of the user's command verbatim in its
						// reason string (go/log-injection).
						log.Printf("[agent] rejecting run_command %q: %q",
							truncateStr(rc.Command, 80), rejection)
						st.bounceToolCall(ctx, "run_command", rejection)
						continue
					}
				}
			}

			// Same shell-validation + working-dir gate for run_background.
			// Without this, the May 8 2026 phantom-/workspace drift went
			// unblocked: the surgical-edit gate covered run_command but
			// run_background sailed through, so `run_background "cd
			// /workspace && python app.py"` looped for 3 turns before the
			// repeat detector caught it. validateRunCommand chains both
			// gates so destructive shell verbs and /workspace drift get
			// the same treatment regardless of which run_* tool the model
			// picks.
			if parsed.Name == "run_background" && !ctx.YoloMode {
				var rb RunBackgroundInput
				if json.Unmarshal(parsed.Args, &rb) == nil {
					if rejection := validateRunCommand(rb.Command, ctx.WorkingDir); rejection != "" {
						log.Printf("[agent] rejecting run_background %q: %q",
							truncateStr(rb.Command, 80), rejection)
						st.bounceToolCall(ctx, "run_background", rejection)
						continue
					}
				}
			}

			// Tool-call repetition detector. Catches the structural-loop
			// case the lens scoring doesn't see: same exact (tool, args)
			// emitted N times in close succession. Lens covers semantic
			// repetition (model produced the same low-quality content);
			// this covers structural repetition (model emitted the same
			// call to read_file or run_command). Fires before tool
			// execution so the corrective lands in the same iteration
			// as the lens corrective if both trigger.
			pendingRepeatCorrective := ""
			// Runaway backstop (#147 review #14): count writes per path and
			// force the repeat path once a single file is rewritten far more
			// than any real iteration would.
			runawayWrite := false
			if parsed.Name == "write_file" {
				if wp := writeFilePath(parsed.Args); wp != "" {
					writeCountByPath[wp]++
					if writeCountByPath[wp] == runawayWriteThreshold {
						runawayWrite = true
						log.Printf("[agent] runaway write backstop: %q rewritten %d times — escalating", wp, writeCountByPath[wp])
					}
				}
			}
			// A tool the loop has taken away for this file stays taken away.
			// Enforced before the identical-resend check so a model that
			// varies one byte cannot walk around the ban.
			if p := extractFailurePath(parsed.Name, parsed.Args); p != "" &&
				st.toolBanned[parsed.Name+"\x00"+p] {
				log.Printf("[agent] turn=%d blocked %s on %s (tool withdrawn for this file)", turn, parsed.Name, p)
				st.bounceToolCall(ctx, parsed.Name, toolBanNote(parsed.Name, p))
				consecutiveErrors++
				totalFailures++
				// A bounce off the ban is a failure like any other. This
				// branch incremented the counters and then continued without
				// reading them, so the ceiling could not end a session that
				// only ever bounced here — measured at 19 and 85 bounces.
				ctx.RecentFailurePaths = appendRecentFailurePath(ctx.RecentFailurePaths, p)
				if shouldStopForFailures(totalFailures, consecutiveErrors, ctx.RecentFailurePaths) {
					log.Printf("[agent] breaking at turn %d: %d refused/failed calls, %d consecutive on %q",
						turn, totalFailures, consecutiveErrors, p)
					endStream(TerminalStopped, "repeated_refusal",
						repeatedRefusalSummary(parsed.Name, p, st.madeProductiveChange)+liveBackgroundJobNote(ctx))
					return nil
				}
				continue
			}
			// Refuse an exact re-send of an already-rejected call before it
			// executes. Checked ahead of the repetition detector because that
			// one needs three occurrences and only steers the NEXT turn,
			// which a two-turn identical pair never reaches.
			// Same representation the repeat detector uses, for the same
			// reason: a fenced re-send is byte-identical as the model wrote
			// it and different only in the body the channel fetched for it.
			// The lookup, the record and the clear all key on the intent, or
			// they key on three different things and never meet.
			// C4: this exact replacement was already refused against the
			// bytes still on disk. It has to be answered BEFORE the resend
			// ban, which would otherwise end the run with the model never
			// having been shown the file it keeps trying to replace.
			if sha := resolvedProposalHash(parsed.Name, parsed.Args); sha != "" {
				rel := ledgerArgPath(parsed.Args, "path")
				if canon, diskHash := survivingKnownGood(ctx, rel); canon != "" {
					if ev := st.c4Rejected[canon]; ev != nil && ev.diskHash == diskHash {
						if _, already := ev.diagnostics[sha]; already {
							if msg := rejectedProposalRecovery(ctx, st, rel, canon, sha); msg != "" {
								st.bounceToolCall(ctx, parsed.Name, msg)
								if accountRefusedCall(parsed.Name, intentArgs, msg,
									workspaceRefusalPath(ctx, parsed.Name, parsed.Args)) {
									return nil
								}
								continue
							}
						}
					}
				}
			}

			// C3: the no-op edit over an artifact already shown to be broken.
			// This has to come BEFORE the identical-resend ban, which would
			// otherwise intercept the recurrence and end the run with the
			// broken file still on disk -- the retained shape exactly.
			if relPath := noopEditIntent(parsed.Name, intentArgs); relPath != "" {
				if msg := brokenArtifactRecovery(ctx, st, relPath); msg != "" {
					st.bounceToolCall(ctx, parsed.Name, msg)
					// Counted once, here. The call never reaches the tool, so
					// no post-execution accounting runs for it, and the ban
					// below is skipped by the continue.
					if accountRefusedCall(parsed.Name, intentArgs, msg,
						workspaceRefusalPath(ctx, parsed.Name, parsed.Args)) {
						return nil
					}
					continue
				}
			}
			if refusal := identicalRetryRefusal(ctx, parsed.Name,
				retryIdentityArgs(parsed.Name, intentArgs, parsed.Args)); refusal != "" {
				log.Printf("[agent] turn=%d refusing an identical re-send of a rejected %s", turn, parsed.Name)
				// Escalate from refusing THIS call to removing the tool for
				// THIS file. The model has now sent the same rejected call
				// twice, so a third refusal buys another identical turn.
				if p := extractFailurePath(parsed.Name, parsed.Args); p != "" {
					if st.toolBanned == nil {
						st.toolBanned = map[string]bool{}
					}
					key := parsed.Name + "\x00" + p
					if !st.toolBanned[key] {
						st.toolBanned[key] = true
						log.Printf("[agent] %s is now unavailable for %s — identical rejected call re-sent", parsed.Name, p)
					}
					refusal += " " + toolBanNote(parsed.Name, p)
				}
				st.bounceToolCall(ctx, parsed.Name, refusal)
				// A refusal is a failure and has to count as one. Skipping
				// the counters would leave a model that spams one rejected
				// call running to the turn cap with nothing to stop it — the
				// refusal returns before both the repetition window and the
				// error counter, so neither breaker would ever see it.
				consecutiveErrors++
				totalFailures++
				failPath := extractFailurePath(parsed.Name, parsed.Args)
				ctx.RecentFailurePaths = appendRecentFailurePath(ctx.RecentFailurePaths, failPath)
				// ...and the same stopping rules have to apply here. Both the
				// ceiling and the path-aware breaker live inside the
				// post-execution failure branch, which this path skips, so
				// incrementing alone left the counters with no reader.
				if shouldStopForFailures(totalFailures, consecutiveErrors, ctx.RecentFailurePaths) {
					log.Printf("[agent] breaking at turn %d: %d refused/failed calls, %d consecutive on %q",
						turn, totalFailures, consecutiveErrors, ctx.RecentFailurePaths[len(ctx.RecentFailurePaths)-1])
					endStream(TerminalStopped, "repeated_refusal",
						repeatedRefusalSummary(parsed.Name, failPath, st.madeProductiveChange)+liveBackgroundJobNote(ctx))
					return nil
				}
				continue
			}
			if msg, _, repeating := recordToolCall(ctx, parsed.Name, intentArgs); repeating || runawayWrite {
				if runawayWrite && !repeating {
					msg = "You have rewritten this file an unusually large number of times without converging. Stop rewriting the whole file — read the current on-disk version, make ONE targeted change with edit_file/structural_edit, or step back and reconsider the approach; if the task is satisfied, respond with done."
					// The detector clears its window only when IT fires.
					// The backstop is a separate trigger for the same
					// corrective, so clear it here too.
					resetToolRepeatWindow(ctx)
				}
				log.Printf("[agent] tool-call repetition at turn %d on %s — queuing corrective for next turn", turn, parsed.Name)
				ctx.Stream("agent_repeat_intervention", map[string]interface{}{
					"turn":   turn,
					"tool":   parsed.Name,
					"reason": msg,
				})
				pendingRepeatCorrective = msg
				repeatDetections++
				// Steer-before-kill ladder. On the FIRST detection, fall
				// through: pendingRepeatCorrective is injected below and the
				// loop continues, so the model gets an explicit nudge to
				// change approach before we ever terminate. Only hard-stop
				// on the SECOND detection — the model saw the steer and
				// repeated anyway (genuinely stuck).
				//
				// The old code hard-stopped on the FIRST detection whenever
				// st.madeProductiveChange was set ("work landed, model spinning
				// on verification"). That mistook legitimate iteration for a
				// loop: 2026-07-19 showed models one nudge from finishing
				// — regex-chess repeating a verify command that itself had a
				// syntax error, polyglot mid-fix — killed with their solution
				// on disk but unverified. A nudge first is the whole point:
				// the broken-verify steer (below) can turn exactly these into
				// completions.
				if repeatDetections >= 2 {
					// Output-rescue: if the task named a deliverable and it
					// isn't on disk, the model is looping WITHOUT having
					// committed its answer — steer toward the file once and
					// keep going rather than hard-stopping empty-handed.
					if missing := missingExpectedOutputs(ctx, st.expectedOutputs); len(missing) > 0 && !outputRescueUsed {
						outputRescueUsed = true
						repeatDetections = 0
						pendingRepeatCorrective = expectedOutputMissingMessage(missing)
						log.Printf("[agent] repeat loop at turn %d but named deliverable(s) %v not on disk — output-rescue steer instead of stopping", turn, logPaths(missing))
					} else {
						log.Printf("[agent] second repetition detection at turn %d — stopping (productive_change_hint=%v)", turn, st.madeProductiveChange)
						// The one terminal wired to recovery. It runs BEFORE
						// the summary so the disclosure describes the bytes
						// that are actually on disk when the run ends. It is
						// a system action: no progress hint is set, no tool
						// event is emitted, and the terminal stays stopped.
						recovered := restoreSaferDeliverables(ctx)
						endStream(TerminalStopped, "repeat_detector",
							repeatTerminalSummary(ctx, st.expectedOutputs,
								st.madeProductiveChange, recovered))
						return nil
					}
				}
			}

			// Reasoning-repetition detector (BiasBusters #30). The
			// model's reasoning_content stream is captured per-turn in
			// ctx.LastTurnReasoning by callLLMOnce. recordReasoning
			// compares the normalized opening prefix against the prior
			// turn's snippet; ≥2 consecutive identical openings fires
			// the intervention. Sibling to the structural repeat
			// detector (above) and the lens regression detector (below)
			// — three different angles on "model is stuck", catching
			// different shapes of stuck-ness.
			pendingReasoningCorrective := ""
			if msg, obs, repeating := recordReasoning(ctx, ctx.LastTurnReasoning); repeating {
				log.Printf("[agent] reasoning repetition at turn %d (consecutive=%d) — queuing corrective", turn, obs.Count)
				ctx.Stream("agent_reasoning_intervention", map[string]interface{}{
					"turn":        turn,
					"consecutive": obs.Count,
					"reason":      msg,
					"snippet":     obs.Snippet,
				})
				pendingReasoningCorrective = msg
			}

			// Score write_file/edit_file content with the geometric lens
			// BEFORE executing. The score reflects what the model produced
			// (independent of whether the tool succeeds). On a quality-crash pattern (N consecutive
			// low scores) we queue a corrective system message that gets
			// appended AFTER the tool result so the next LLM call sees:
			// assistant(tool_call) → tool(result) → system(lens warning).
			// This is the direct fix for the May 6 templates/resources.html
			// stub-loop case where the stub gate kept rejecting but the model
			// kept retrying the same stub.
			pendingLensCorrective := ""
			if scorable, ok := extractScorableContent(parsed.Name, parsed.Args); ok {
				// Capture the model's write for deferred lens-training labeling
				// (a later /feedback call turns it into a weighted sample). Same
				// content the lens scores below, so a sample mirrors its score.
				ctx.RecordPassWrite(parsed.Name, extractFailurePath(parsed.Name, parsed.Args), scorable)
				if score, scored := scoreContentForAgent(ctx.Ctx, ctx.LensURL, scorable); scored {
					ctx.LensScoreHistory = append(ctx.LensScoreHistory, score.Aggregate.GxScoreMin)
					log.Printf("[agent] lens turn=%d tool=%s gx_min=%.3f gx_mean=%.3f off_rails=%d n_tok=%d latency=%.0fms history=%s",
						turn, parsed.Name,
						score.Aggregate.GxScoreMin, score.Aggregate.GxScoreMean,
						score.Aggregate.FirstOffRailsIdx, score.NTokens,
						score.LatencyMS, formatScoreSlice(ctx.LensScoreHistory))
					ctx.Stream("agent_lens_score", map[string]interface{}{
						"tool":                parsed.Name,
						"turn":                turn,
						"n_tokens":            score.NTokens,
						"first_off_rails_idx": score.Aggregate.FirstOffRailsIdx,
						"gx_score_min":        score.Aggregate.GxScoreMin,
						"gx_score_mean":       score.Aggregate.GxScoreMean,
						"latency_ms":          score.LatencyMS,
					})
					if low, severe, calibrated := score.calibratedThresholds(); calibrated {
						if msg, intervene := agentLensRegression(ctx.LensScoreHistory, low, severe); intervene {
							log.Printf("[agent] lens regression at turn %d on %s — queuing corrective for next turn", turn, parsed.Name)
							ctx.Stream("agent_lens_intervention", map[string]interface{}{
								"turn":   turn,
								"tool":   parsed.Name,
								"reason": msg,
							})
							pendingLensCorrective = msg
							// Reset history so we don't re-fire on the same crash.
							ctx.LensScoreHistory = nil
						}
					}
				}
			}

			// Execute tool. A re-read of an unchanged file already in
			// context is served from a compact pointer instead of
			// re-injecting + re-encoding the whole file (see
			// redundantReadShortCircuit).
			startTime := time.Now()
			result := redundantReadShortCircuit(parsed.Name, parsed.Args, ctx)
			if result != nil {
				log.Printf("[agent] turn=%d short-circuited redundant read (already in context, unchanged)", turn)
			}
			if result == nil && (parsed.Name == "run_command" || parsed.Name == "run_background") {
				if blk := runBlockAfterTraceback(ctx); blk != nil {
					result = blk
					log.Printf("[agent] turn=%d blocked re-run after traceback — forcing an edit first", turn)
				}
			}
			if result == nil {
				result = executeToolCall(parsed.Name, parsed.Args, ctx)
				// The deadline can land in the middle of a tool call. Stop
				// here rather than starting another turn on a dead context.
				if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
					return finishCancelledRun(ctx, st, turn)
				}
				// Debt is settled from the ledger's own evidence, never from
				// the call reporting that it worked.
				settleMutationDebt(ctx, st)
			}
			elapsed := time.Since(startTime)

			// On failure, log the error so it shows up in `docker compose
			// logs atlas-proxy` without having to attach a debugger.
			if !result.Success {
				log.Printf("[agent] turn=%d tool=%q FAIL: %q", turn,
					truncateStr(parsed.Name, 64), truncateStr(result.Error, 240))
				recordFailedToolCall(ctx, parsed.Name,
					retryIdentityArgs(parsed.Name, intentArgs, parsed.Args), result.Error)
				// Every refusal of authored content is a deterministic
				// negative for the lens corpus. One site rather than 60-odd
				// rejection points, and it cannot miss a gate added later.
				if authored := authoredContent(parsed.Args); authored != "" {
					recordGateRejection(modelName, parsed.Name,
						rejectedPath(parsed.Args), authored, result.Error)
				}
				// C4: a replacement refused while the file it targeted stayed
				// valid. The second distinct proposal against one generation
				// is the evidence that the refusal text alone is not landing,
				// and the answer rides on the SAME result -- one call, one
				// result, and the model reads it in the same message.
				if key, sha, distinct := noteRejectedProposal(ctx, st, parsed.Name,
					parsed.Args, result); distinct >= 2 {
					rel := ledgerArgPath(parsed.Args, "path")
					if msg := rejectedProposalRecovery(ctx, st, rel, key, sha); msg != "" {
						result.Error += "\n\n" + msg
					}
				}
			} else {
				// A call can fail and later succeed — an edit rejected for a
				// stale range works after a re-read. Drop the memory with the
				// condition that caused it.
				clearFailedToolCall(ctx, parsed.Name,
					retryIdentityArgs(parsed.Name, intentArgs, parsed.Args))
				clearSteerState(ctx, st, parsed.Name, parsed.Args)
				clearBrokenArtifactState(ctx, st, parsed.Name, intentArgs)
			}

			st.pendingToolCall = ""
			ctx.Stream("tool_result", map[string]interface{}{
				"tool":    parsed.Name,
				"success": result.Success,
				"data":    json.RawMessage(result.Data),
				"error":   result.Error,
				"elapsed": elapsed.String(),
			})
			Emit(Envelope{
				EventID:    NewEventID(),
				Timestamp:  float64(time.Now().UnixNano()) / 1e9,
				Type:       EvtToolResult,
				Stage:      "tool",
				DurationMS: elapsed.Milliseconds(),
				Payload: map[string]interface{}{
					"name":    parsed.Name,
					"success": result.Success,
					"error":   truncateStr(result.Error, 120),
				},
			})

			// Track productive state changes — write/edit/delete that landed.
			// Used below to soften the error-loop exit when work was completed
			// AND by the done-without-action gate so a feature prompt
			// ("rewrite X", "add Y") can't declare done without any actual
			// edit on disk. structural_edit was missing from this list pre-May-10,
			// which let a structural_edit-only success path slip past the
			// productive-change tracking too.
			// insert_after and replace_lines were absent, so a turn whose only
			// work was a successful insert did not count as productive and the
			// action gate could bounce it for having "done nothing".
			// A landed write whose result carries a parse warning is pending
			// execution; a landed clean write clears its own pending mark.
			if result.Success && parsed.Name == "write_file" {
				var wf WriteFileInput
				if json.Unmarshal(parsed.Args, &wf) == nil {
					var out WriteFileOutput
					warned := json.Unmarshal(result.Data, &out) == nil && out.Warning != ""
					st.markWarnedRun(ctx, wf.Path, warned)
				}
			}
			if parsed.Name == "run_command" && result != nil {
				// Discharge only the marks this command actually attempted to
				// EXECUTE. Two prior rules both failed an audit: clearing on
				// any command let `ls` bless a warned file, and clearing on a
				// filename substring let `cat solve.py` or `grep main solve.py`
				// do the same — naming a file is not running it. The mark is
				// "this version has never been executed"; only an execution
				// attempt (interpreter invocation or ./file) changes that fact,
				// and the attempt itself suffices — pass or fail, the model has
				// now seen the real runtime behavior.
				var rc RunCommandInput
				if json.Unmarshal(parsed.Args, &rc) == nil {
					for p := range st.pendingWarnedRun {
						if executionAttempt(rc.Command, p) {
							delete(st.pendingWarnedRun, p)
						}
					}
				}
			}
			if result.Success && len(ctx.LiteralBlocks) > 0 &&
				(parsed.Name == "write_file" || parsed.Name == "edit_file" ||
					parsed.Name == "structural_edit" ||
					parsed.Name == "insert_after" || parsed.Name == "replace_lines") {
				// Literal-contract enforcement: the user's own bytes are the
				// authoritative rendering of any content they spelled out, and
				// the model measurably cannot transcribe bytes (space-prefixed
				// BPE tokens win after quotes — `BANNER = "ready"` arrives as
				// `BANNER = " ready"` deterministically). Whitespace-only
				// drift from a stated literal is repaired in place.
				var wp struct {
					Path string `json:"path"`
				}
				if json.Unmarshal(parsed.Args, &wp) == nil && wp.Path != "" {
					lp := resolveAgentPath(ctx, wp.Path)
					if body, rerr := os.ReadFile(lp); rerr == nil {
						if fixed, repairedLits, changed := repairLiteralDrift(string(body), ctx.LiteralBlocks); changed {
							if werr := os.WriteFile(lp, []byte(fixed), 0o644); werr == nil {
								for _, l := range repairedLits {
									log.Printf("[agent] literal contract repaired in %s: %q now byte-exact", wp.Path, truncateStr(l, 60))
								}
								Emit(NewEnvelope(EvtMetric, "tool", map[string]interface{}{
									"name": "literal_repair", "value": wp.Path,
								}))
								// Deliberately silent. The repair corrects a known
								// model defect, and the file now holds what the
								// user asked for, so a read shows the right bytes
								// either way. Announcing it just hands the model
								// something to react to: five repairs in one
								// session meant five correctives, and each one is
								// a turn spent on spacing rather than the task.
							}
						}
					}
				}
			}
			// move_file was missing here, so a successful rename counted as no
			// work at all: the run reported "Nothing was written -- no file was
			// created or changed" while the destination sat on disk. A
			// relocation is a state change like any other.
			if result.Success && (parsed.Name == "write_file" || parsed.Name == "edit_file" ||
				parsed.Name == "structural_edit" || parsed.Name == "delete_file" ||
				parsed.Name == "insert_after" || parsed.Name == "replace_lines" ||
				parsed.Name == "move_file") {
				st.madeProductiveChange = true
				// A write AFTER a successful verification un-verifies the
				// run: what was checked is no longer what is on disk. Three
				// novel-benchmark sessions ran a working version, rewrote
				// it, and exited — the checker then found a traceback the
				// session never saw, because nothing demanded a re-run of
				// the final artifact. Verification is of an artifact, not
				// of a session.
				if st.verifiedThisLoop {
					log.Printf("[agent] %s after verification — the final artifact is unverified, re-arming the gate", parsed.Name)
					st.verifiedThisLoop = false
					st.verifiedStandalone = false
					st.verifiedByRedirect = ""
					st.sawFailedVerification = false
				}
			}

			// Track verification — a successful run_command of a build /
			// test / probe / runner. Recon (ls, cat, grep) doesn't count.
			// Once any verification succeeds in this loop, the fix-intent
			// gate stops blocking `done`.
			if parsed.Name == "run_command" {
				var rc RunCommandInput
				if json.Unmarshal(parsed.Args, &rc) == nil && isVerificationCommand(rc.Command) {
					if result.Success && silentRunWhenOutputPromised(ctx, userMessage, rc.Command, result.Data) {
						// Exit 0 with empty stdout is not verification of a
						// task whose prompt demands printed output. Measured:
						// a generation drifted into comment-reasoning, the
						// tail of the file (including the solve() call) was
						// swallowed by a comment, and the program parsed, ran,
						// printed nothing and exited 0 — the session recorded
						// that as verification and reported success on a
						// program that provably produced no answer.
						// Latch, don't just decline: a silent run IS a failed
						// verification of a print-demanding task. Merely not
						// counting it left done free to pass when nothing
						// else demanded verification — measured: the gate
						// fired three times in one night and three silent
						// finals still shipped.
						st.sawFailedVerification = true
						log.Printf("[agent] run exited 0 with no stdout on a print-demanding task — latching the verification gate: %q",
							truncateStr(rc.Command, 60))
					} else if result.Success {
						st.verifiedThisLoop = true
						ctx.VerifiedThisRun = true
						st.sawFailedVerification = false
						st.redRunStreak = 0
						st.verifiedHashes = sessionWriteHashes(ctx)
						// Evidence record: bind this green run to the files it
						// actually named and the exact bytes they held. Lens
						// labeling reads these — never the session-wide flag.
						covered := map[string]string{}
						for p := range ctx.SessionWrites {
							if commandNamesPath(rc.Command, p) {
								if h := fileSHA256(ctx, p); h != "" {
									covered[p] = h
								}
							}
						}
						ctx.VerificationEvidence = append(ctx.VerificationEvidence, VerificationRecord{
							Command:  rc.Command,
							Redirect: stdinRedirectSource(rc.Command),
							Covered:  covered,
							Turn:     turn,
						})
						// A program run as `prog < data` is verified under a
						// contract the caller may not use. Tracked so the exit
						// can tell the two apart. See stdinRedirectSource.
						if src := stdinRedirectSource(rc.Command); src != "" {
							st.verifiedByRedirect = src
						} else {
							st.verifiedStandalone = true
						}
						log.Printf("[agent] verification recorded: turn=%d cmd=%q",
							turn, truncateStr(rc.Command, 60))
					} else {
						// Red test/build. Latches the verification gate on
						// for this loop until something verifies green.
						st.sawFailedVerification = true
						st.redRunStreak++
						st.serverStartBlocked = blockedServerStart(result.Error + string(result.Data))
						log.Printf("[agent] verification FAILED: turn=%d cmd=%q server_blocked=%v — done is gated until it passes",
							turn, truncateStr(rc.Command, 60), st.serverStartBlocked)
						// Advice at the crossing, not at the done-gate: waiting
						// for the model to attempt `done` meant it kept nibbling
						// edits for turns after the streak already proved the
						// approach dead. Queue once, on the transition — the
						// done-gate text repeats it if the model still tries to
						// exit red.
						if st.redRunStreak == rewriteThreshold+1 && !st.serverStartBlocked {
							st.queueCorrective(freshRewriteAdvice(st.redRunStreak))
							log.Printf("[agent] red streak crossed %d — fresh-rewrite advice injected now", rewriteThreshold)
						}
					}
				}
			}

			// Plan-adherence accounting. Records whether this tool
			// call satisfied an unsatisfied step on ctx.Plan (if any),
			// updates the off-streak counter, and asks us to revise
			// the plan if the streak crossed the threshold. Advisory
			// — never blocks the call. recordPlanAdherence is a no-op
			// when ctx.Plan is nil (T0 / planner failure).
			if shouldRevise := recordPlanAdherence(ctx, parsed.Name, parsed.Args, result.Success); shouldRevise {
				revisePlan(ctx, userMessage,
					fmt.Sprintf("agent went off-plan for %d consecutive tool calls (last: %s)",
						ctx.PlanOffStreak, parsed.Name))
			}

			// Break error loops: if 3 tool calls fail in a row, stop.
			// When the agent has already written/edited a file
			// and is now failing on `run_command` (verification noise — no
			// TTY for curses, missing toolchain, etc.), a different exit
			// message is appropriate so the user isn't told "the file may
			// be too large to modify" when their file is, in fact, on disk.
			// edit_file old_str miss: count per path independently of the
			// consecutiveErrors reset an interleaved read causes. On the
			// second miss for the same structured file, force the structural_edit
			// steer as a [system note] (the inline tool-error hint alone
			// doesn't reliably move a small model off edit_file).
			if !result.Success && parsed.Name == "edit_file" &&
				strings.Contains(result.Error, "string to replace not found") {
				mp := extractFailurePath(parsed.Name, parsed.Args)
				editMissByPath[mp]++
				ext := strings.ToLower(filepath.Ext(mp))
				// Force the structural_edit steer on the FIRST miss for structured
				// files — small models bail to run_command after a single
				// edit_file miss rather than retrying, so waiting for a
				// second miss never fires (observed: 1 edit_file all session,
				// then 9 run_command re-runs).
				if editMissByPath[mp] >= 1 && (ext == ".py" || ext == ".html" || ext == ".htm") {
					pendingRepeatCorrective = "edit_file's old_str did not match " +
						mp + " (small drift in whitespace/quotes is enough to miss). " +
						"Do NOT re-read or run the file — switch to structural_edit, which " +
						"needs no old_str: {\"type\":\"tool_call\",\"name\":\"structural_edit\"," +
						"\"args\":{\"path\":\"" + mp + "\",\"selector\":\"function:NAME\" " +
						"(or class:NAME, or <tag> for HTML),\"content\":\"<the full " +
						"replacement function/class/element>\"}}."
					log.Printf("[agent] edit_file miss on %q — forcing structural_edit steer", mp)
				}
			}

			if !result.Success {
				// A failure that differs in KIND from the last one means the
				// model acted on the previous rejection. Reset the streak:
				// the breaker exists to stop a loop, and three distinct
				// rejections in a row is the opposite of one.
				if class := rejectionClass(result.Error); class != "" && class != ctx.LastRejectionClass {
					if consecutiveErrors > 0 {
						log.Printf("[agent] rejection changed kind at turn %d — resetting the error streak (was %d)", turn, consecutiveErrors)
					}
					consecutiveErrors = 0
					ctx.RecentFailurePaths = nil
					ctx.LastRejectionClass = class
				}
				consecutiveErrors++
				totalFailures++
				// Ceiling. Resetting the streak on a changed rejection kind
				// is what lets a converging model keep going; without an
				// absolute bound it also lets one cycle through failure modes
				// indefinitely. Generous, because the whole point is that
				// legitimate iteration costs several attempts per edit.
				if totalFailures >= maxTotalFailures {
					log.Printf("[agent] breaking: %d failed tool calls this run (ceiling %d) at turn %d (productive=%v)",
						totalFailures, maxTotalFailures, turn, st.madeProductiveChange)
					if st.madeProductiveChange {
						emitTerminal(ctx, st, TerminalStopped, "failure_ceiling",
							unverifiedSummary(true, "The run hit its failed-call ceiling before finishing."))
					} else {
						emitTerminal(ctx, st, TerminalStopped, "failure_ceiling", fmt.Sprintf(
							"Stopped after %d failed tool calls with nothing landing on disk. The per-turn errors above say what was refused each time; the last one is the one to act on.", totalFailures))
					}
					return nil
				}
				// May 10 2026: path-aware breaker. Track which file each
				// failure was on; only escalate when 3 consecutive failures
				// share the same path (= truly stuck on one file). 3 fails
				// across DIFFERENT files = grinding through multi-file work,
				// keep going.
				failPath := extractFailurePath(parsed.Name, parsed.Args)
				ctx.RecentFailurePaths = append(ctx.RecentFailurePaths, failPath)
				if len(ctx.RecentFailurePaths) > 3 {
					ctx.RecentFailurePaths = ctx.RecentFailurePaths[len(ctx.RecentFailurePaths)-3:]
				}
				if consecutiveErrors >= 3 {
					samePath := stuckOnOnePath(ctx.RecentFailurePaths)
					if !samePath {
						log.Printf("[agent] path-aware breaker: %d consecutive failures across different paths (%v) — continuing, not a stuck loop", consecutiveErrors, ctx.RecentFailurePaths)
						// Reset consecutiveErrors so the multi-file grind
						// can keep going. The recent-paths list stays as
						// a rolling window so if subsequent fails DO
						// collapse onto one path, we still catch it.
						consecutiveErrors = 0
					} else if missing := missingExpectedOutputs(ctx, st.expectedOutputs); len(missing) > 0 && !outputRescueUsed {
						// Output-rescue (same as the repeat breaker): looping
						// on failures without ever committing the named
						// deliverable — steer toward it once before stopping.
						outputRescueUsed = true
						consecutiveErrors = 0
						ctx.RecentFailurePaths = nil
						ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: expectedOutputMissingMessage(missing)})
						log.Printf("[agent] error loop at turn %d but named deliverable(s) %v not on disk — output-rescue steer instead of stopping", turn, logPaths(missing))
					} else {
						log.Printf("[agent] breaking error loop: %d consecutive failures on the same path %q at turn %d (productive=%v)",
							consecutiveErrors, ctx.RecentFailurePaths[0], turn, st.madeProductiveChange)
						if st.madeProductiveChange {
							emitTerminal(ctx, st, TerminalStopped, "same_target_failures",
								"Wrote your changes to disk; couldn't verify them automatically (the verification commands failed). Run them yourself to confirm — they're on disk.")
						} else {
							emitTerminal(ctx, st, TerminalStopped, "same_target_failures",
								"Stopped after 3 tool failures on the same target with no successful changes. Common causes: the file you referenced isn't in the workspace, an empty path argument was passed, or a regex was malformed. Check the per-turn errors above, then try a more specific request (e.g. \"fix snake_game.py at line 95 — the curses bounds are wrong\").")
						}
						return nil
					}
				}
			} else {
				consecutiveErrors = 0
				// Successful tool call resets the path window — the model
				// is clearly making progress somewhere.
				ctx.RecentFailurePaths = nil
			}

			// Track consecutive read-only calls to detect exploration loops.
			// outline_file/find_file MUST be here too — otherwise an
			// interleaved outline resets the counter and the model
			// read→outline→read→outline forever without the breaker firing
			// (observed live with a compact reasoning model). Every navigation-only tool counts.
			isReadOnly := parsed.Name == "read_file" ||
				parsed.Name == "outline_file" ||
				parsed.Name == "list_directory" ||
				parsed.Name == "search_files" ||
				parsed.Name == "find_file"
			if isReadOnly {
				consecutiveReads++
				if result.Success {
					// The model went looking at the project, so it read the
					// message as work rather than conversation. Used by the
					// done-without-action gate below to tell "remove the
					// debug logging" (opens the file, writes nothing) apart
					// from "thanks, that looks great" (no tool calls at all).
					st.inspectedWorkspace = true
				}
			} else {
				consecutiveReads = 0
			}

			// Add assistant message (the tool call) and tool result to conversation
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "assistant",
				Content: response,
			})
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:       "tool",
				Content:    result.MarshalText(),
				ToolCallID: fmt.Sprintf("call_%d", turn),
				ToolName:   parsed.Name,
			})

			// Lens intervention: if the lens flagged a
			// regression earlier in this iteration, append the corrective
			// NOW so the next LLM call sees it after the tool result.
			// Role MUST be "user" — some Jinja chat templates enforce
			// "System message must be at the beginning" and rejects any
			// system role appended mid-conversation, which previously
			// crashed the next LLM call with a 500. The "[system note]:"
			// prefix is how the model knows it's loop-machinery feedback,
			// not an actual user instruction.
			// Loop-health correctives, queued in signal order (lens
			// quality crash, repeated call, rehashed reasoning) and drained
			// through one path. Each slot holds at most one message: the
			// repeat slot is deliberately overwritable, so the specific
			// edit_file -> structural_edit steer above replaces the generic
			// repeat warning instead of stacking with it.
			st.queueCorrective(pendingLensCorrective)
			st.queueCorrective(pendingRepeatCorrective)
			st.queueCorrective(pendingReasoningCorrective)
			// A background job's outcome is invisible unless the model asks
			// for it, so a server that died on startup reads the same as one
			// serving happily. Surfaced once per job, through the same queue
			// as every other steer.
			st.queueCorrective(finishedBackgroundNote(ctx))
			st.drainCorrectives(ctx)

			// Option 3 (issue #39): traceback → directed edit. When a
			// run_command surfaced a Python traceback, mechanically extract
			// the fix site and hand the model a directed instruction ("fix
			// function X here") instead of leaving it to localize — the step
			// a weak model fails by hallucinating symbols / editing the wrong
			// function. The stack frame IS the localization; no LLM reasoning
			// needed to read it.
			if !result.Success && (parsed.Name == "run_command" || parsed.Name == "run_background") {
				// Scan the RAW stdout/stderr, not result.MarshalText() — the
				// marshaled JSON escapes the quotes in `File "..."` frames, so
				// the traceback regex wouldn't match.
				var rc struct {
					Stdout string `json:"stdout"`
					Stderr string `json:"stderr"`
				}
				_ = json.Unmarshal(result.Data, &rc)
				scan := rc.Stderr + "\n" + rc.Stdout
				if scan == "\n" {
					scan = result.Error
				}
				// The command text — needed by the broken-inline-script
				// steer to tell a malformed verify one-liner (SyntaxError in
				// the `-c` argument) apart from a real code bug.
				var runArgs struct {
					Command string `json:"command"`
				}
				_ = json.Unmarshal(parsed.Args, &runArgs)
				if steer := brokenInlineScriptSteer(runArgs.Command, scan); steer != "" {
					// Broken verification command: the SyntaxError is in the
					// model's own inline `-c` test, not the solution. Steer it
					// to move the test into a file instead of re-running the
					// unparseable one-liner into the repetition breaker.
					ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: steer})
					log.Printf("[agent] broken-inline-script steer: verify command won't parse, directed to a test file")
				} else if steer := tracebackSteer(ctx, scan); steer != "" {
					ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: steer})
					log.Printf("[agent] traceback localization: steered to fix site")
				} else if steer := missingModuleSteer(ctx, scan); steer != "" {
					// Uninstalled-dependency recovery: the run failed with "No
					// module named X". Tell the model to pip install it instead
					// of re-running the identical failing command into the
					// repetition breaker.
					ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: steer})
					log.Printf("[agent] missing-module steer: directed to install dependency")
				} else if steer := missingCommandSteer(scan); steer != "" {
					// Missing-binary recovery: the sandbox image lacks the
					// command and can't apt-install it (non-root, read-only).
					// Say so and point at the escape hatches, instead of the
					// model re-running into the breaker or giving up.
					ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: steer})
					log.Printf("[agent] missing-command steer: named the unavailable binary")
				} else if steer := missingFileSteer(ctx, scan); steer != "" {
					// Case-typo recovery: command referenced a file whose name
					// differs only in case from a real workspace file. Name the
					// correct file so the model stops re-running the wrong name.
					ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: steer})
					log.Printf("[agent] missing-file localization: steered to correct case")
				}
			}

			// Cross-file coherence signals after a successful mutation:
			// the session file manifest (so later files reference earlier
			// ones instead of re-creating them) and the asset-graph lint
			// (orphaned templates/static files, dangling refs). Both are
			// advisory [system note]s — never blockers.
			if result.Success &&
				(parsed.Name == "write_file" || parsed.Name == "edit_file" ||
					parsed.Name == "structural_edit" || parsed.Name == "move_file" ||
					parsed.Name == "delete_file") {
				if note := sessionManifestNote(ctx); note != "" {
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:    "user",
						Content: "[system note]: " + note,
					})
					log.Printf("[agent] session manifest announced (%d files)", len(ctx.SessionWrites))
				}
				if note := assetLintNote(ctx); note != "" {
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:    "user",
						Content: "[system note]: " + note,
					})
					ctx.Stream("asset_lint", map[string]interface{}{
						"turn":   turn,
						"detail": note,
					})
					log.Printf("[agent] asset lint: %s", truncateStr(note, 160))
				}
			}

			// Trust V3-verified edits — strongly nudge toward done.
			// When V3 ran the edit through its sandbox/probe pipeline and
			// the result came back successful (V3Used && PhaseSolved
			// non-empty), the edit is build-verified. Compact models can otherwise
			// keeps grinding: re-reads the file, edits unrelated functions,
			// runs another V3 cycle (~110s each). Inject an explicit
			// "you're done unless you have a specific reason" message.
			// "none" is the phase a run reports when nothing passed, and it
			// is not the empty string — so this fired on every unverified
			// fallback and told the model its code was build-checked when
			// no candidate had passed anything. Measured across one
			// 28-session run: 0 of 44 candidates passed the sandbox and
			// this nudge fired 11 times, each one pushing the model to stop
			// working on a file that had failed verification.
			if result.Success && result.V3Used && verifiedPhase(result.PhaseSolved) &&
				(parsed.Name == "write_file" || parsed.Name == "edit_file") {
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role: "user",
					Content: fmt.Sprintf(
						"V3 verified this edit passed its %s pipeline (%d candidates, score=%.2f). The fix is on disk and build-checked. If this resolves the user's original request, respond NOW with {\"type\":\"done\",\"summary\":\"<one sentence describing the fix>\"}. Only continue if you have a specific, concrete additional change to make — do not re-read the file to double-check, and do not edit unrelated code.",
						result.PhaseSolved, result.CandidatesTested, result.WinningScore,
					),
				})
				log.Printf("[agent] V3-verified %s on %s — nudging toward done", parsed.Name, truncateStr(string(parsed.Args), 80))
			}

			// Exploration budget: after 4 consecutive read-only calls,
			// inject nudge. After 5, escalate the nudge. The read above
			// already executed and its result is in context — the nudge
			// steers the NEXT turn toward a write.
			// FUTURE (L6 reliability): Compact models can over-explore when adding
			// features to existing projects (~67% pass rate). Better prompting,
			// larger model, or V3-guided exploration would improve this.
			if consecutiveReads == 4 {
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role:    "user",
					Content: "You have full project context in the system prompt. Do not read more files. Emit a write_file or edit_file tool call now.",
				})
				log.Printf("[agent] exploration budget: warning at turn %d", turn)
			} else if consecutiveReads >= 5 {
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role:    "user",
					Content: "You already have this information in context — reading more files will not help. Write your changes now. Use write_file or edit_file.",
				})
				consecutiveReads = 2 // Keep at warning level, don't reset
				log.Printf("[agent] exploration budget: escalated nudge at turn %d", turn)
			}

		default:
			// Unknown type — grammar should prevent this
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "user",
				Content: fmt.Sprintf("Unknown response type '%s'. Use tool_call, text, or done.", parsed.Type),
			})
		}
	}

	// Running out of turns is not a reason to say nothing. This path used to
	// stream an `error` and return, so a user who asked a question and whose
	// turn hit the cap saw an empty reply — no answer, no partial, no
	// explanation. Observed on a fresh workspace with "how does the contact
	// form work?": four searches, cap reached, zero bytes back. Every other
	// loop exit authors a summary; this one has to as well.
	log.Printf("[agent] max turns (%d) exceeded for %s — returning what the run found", ctx.MaxTurns, ctx.Tier)
	emitTerminal(ctx, st, TerminalIncomplete, "turn_budget_exhausted",
		outOfTurnsSummary(ctx, st.madeProductiveChange)+liveBackgroundJobNote(ctx))
	return nil
}

// ---------------------------------------------------------------------------
// LLM call with grammar constraint
// ---------------------------------------------------------------------------

// isContextOverflow reports whether an LLM-call error is llama-server's
// exceed_context_size_error 400 (prompt tokens > per-slot n_ctx). Matched
// on the error body text — model-agnostic, keyed to llama.cpp's stable
// error type string with the human message as fallback.
func isContextOverflow(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "exceed_context_size") ||
		strings.Contains(s, "exceeds the available context size")
}

// callLLMConstrained calls the LLM with json_schema or grammar constraint.
// Returns the raw response text and token count.
//
// When the model emits zero tokens (raw_len=0) — usually after a
// tool result message under a constrained JSON grammar — we retry
// inline once with a bumped temperature and a transient "continue"
// nudge appended to the messages. This avoids burning a full agent-loop
// turn (~30s + tokens) on the parse-error retry path. The nudge is
// scoped to the retry call only; ctx.Messages is not mutated.
//
// May 2026 BiasBusters #2/#3 — per-step tool restriction. If the previous
// turn ended in a write_file rejection on a .py/.html file >5 lines, the
// model is biased toward retrying with edit_file (lexically closer to
// write_file than structural_edit, despite structural_edit being correct for the case).
// We respond by (a) dropping edit_file and write_file from the GBNF
// tool-name production for this single decision and (b) injecting an
// ephemeral [system note] reminding the model that structural_edit is the only
// available structural-edit tool for this step. ctx.Messages is not
// mutated; the nudge and grammar restriction are scoped to this call.
func callLLMConstrained(ctx *AgentContext) (string, int, error) {
	messages, grammar := buildStepRequest(ctx)

	content, tokens, err := callLLMOnceWithGrammar(ctx, messages, 0.3, grammar)
	if isContextOverflow(err) {
		// The real prompt exceeded the per-slot context despite the
		// budget estimate (dense content under-counts at chars/4).
		// Recover instead of hard-killing the session: force-trim the
		// conversation to the minimum window (system + pins + 8-tail)
		// and retry once. The trim persists on ctx.Messages — the
		// conversation genuinely no longer fits, so shrinking it is
		// the correct durable state, not just a retry hack.
		log.Printf("[agent] context overflow from llama-server — force-trimming to minimum window and retrying")
		ctx.Messages = trimMessages(ctx.Messages, 8)
		messages, grammar = buildStepRequest(ctx)
		content, tokens, err = callLLMOnceWithGrammar(ctx, messages, 0.3, grammar)
	}
	if err != nil {
		return "", tokens, err
	}
	if strings.TrimSpace(content) != "" {
		return content, tokens, nil
	}

	// Empty response — retry once with a transient continuation nudge
	// and a higher temperature. The nudge gives the model an explicit
	// next-action prompt; the temperature bump escapes the EOS-local
	// minimum that the json_object grammar can wedge the model into.
	log.Printf("[agent] empty LLM response, retrying with temp=0.7 + continuation nudge")
	nudged := append(append([]AgentMessage(nil), messages...), AgentMessage{
		Role:    "user",
		Content: `Continue. Respond with one JSON object: {"type":"tool_call","name":"<tool>","args":{...}} for the next action, or {"type":"done","summary":"..."} if the task is complete. Do not emit empty content.`,
	})
	content2, tokens2, err := callLLMOnceWithGrammar(ctx, nudged, 0.7, grammar)
	if err != nil {
		// Return whatever we have from the original call; caller
		// handles empty via parse-error retry.
		return content, tokens, nil
	}
	return content2, tokens + tokens2, nil
}

// buildStepRequest assembles the messages and grammar for the next LLM
// call. In the common case it returns ctx.Messages and "" (no grammar
// override). When the previous turn ended in a write_file rejection on a
// .py/.html file, it returns ctx.Messages plus an ephemeral [system note]
// user message AND a restricted GBNF grammar that excludes edit_file
// and write_file from the tool-name production. See callLLMConstrained
// docstring for the BiasBusters context.
func buildStepRequest(ctx *AgentContext) ([]AgentMessage, string) {
	// Plan-progress reminder. Always rendered when ctx.Plan exists;
	// not persisted to ctx.Messages so it doesn't accumulate. Lands
	// AT THE TAIL of the messages slice so the model sees it as the
	// most-recent user-role input right before its next decision.
	// May 10 2026 follow-up — long multi-file tasks were losing plan
	// context after trim; per-turn injection makes the progress
	// surface persistent without bloating history.
	planReminder := buildPlanReminder(ctx)

	// Traceback step-restriction (issue #39 / option 3): after a crash, ban the
	// run tools so the model can't loop on re-running and is forced to edit the
	// fix site the traceback names. Takes precedence over the write_file case.
	if tbExcluded, tbNote := tracebackExclusion(ctx); len(tbExcluded) > 0 {
		messages := append([]AgentMessage(nil), ctx.Messages...)
		if planReminder != "" {
			messages = append(messages, AgentMessage{Role: "user", Content: planReminder})
		}
		messages = append(messages, AgentMessage{Role: "user", Content: tbNote})
		log.Printf("[agent] traceback step-restriction: banning run tools, forcing an edit")
		return messages, buildGBNFGrammarForTools(tbExcluded)
	}

	excluded, ext := stepExclusions(ctx)
	if len(excluded) == 0 {
		if planReminder == "" {
			return ctx.Messages, ""
		}
		messages := append([]AgentMessage(nil), ctx.Messages...)
		messages = append(messages, AgentMessage{Role: "user", Content: planReminder})
		return messages, ""
	}

	selectors := structuralSelectorHint(ext)
	if selectors == "" {
		selectors = "`function:NAME` or `class:NAME`"
	}
	note := fmt.Sprintf(
		"[system note]: For this single decision, %s is unavailable. The previous write_file was rejected because the target is an existing %s file >5 lines. Use structural_edit with a structural selector (%s) to rewrite the named node. structural_edit doesn't need old_str so it doesn't truncate on long content. Emit exactly one JSON object: {\"type\":\"tool_call\",\"name\":\"structural_edit\",\"args\":{\"path\":\"...\",\"selector\":\"...\",\"content\":\"...\"}}.",
		strings.Join(excluded, " and "),
		strings.TrimPrefix(ext, "."),
		selectors,
	)
	messages := append([]AgentMessage(nil), ctx.Messages...)
	if planReminder != "" {
		messages = append(messages, AgentMessage{Role: "user", Content: planReminder})
	}
	messages = append(messages, AgentMessage{Role: "user", Content: note})

	grammar := buildGBNFGrammarForTools(excluded)
	log.Printf("[agent] step-restriction active: banning %v from tool-name enum (ext=%q) — BiasBusters #2/#3", excluded, ext)
	return messages, grammar
}

// stepExclusions inspects the tail of ctx.Messages and returns the list
// of tool names that must be banned for the next decision, plus the
// triggering file extension. Returns nil/"" in the common case.
//
// Trigger: most recent tool-result message is from write_file with a
// success=false body whose error mentions "already exists", and the
// path being targeted has extension .py / .html / .htm. The window
// scanned is the last 6 messages (assistant call + tool result + a few
// recent siblings).
func stepExclusions(ctx *AgentContext) ([]string, string) {
	n := len(ctx.Messages)
	if n == 0 {
		return nil, ""
	}
	// Walk backwards over the recent tail. We only fire when the LAST
	// tool message is a write_file rejection on .py/.html. If a fresh
	// assistant turn has already happened (the model corrected itself),
	// the tail will end in something other than that tool result and we
	// return nil — the restriction expires after a single decision.
	startIdx := n - 1
	if startIdx > 6 {
		startIdx = 6
	}
	for i := n - 1; i >= n-1-startIdx && i >= 0; i-- {
		msg := ctx.Messages[i]
		if msg.Role != "tool" {
			// First non-tool message we encounter while walking back —
			// stop. We don't want a stale rejection from 4 turns ago to
			// keep firing.
			if msg.Role == "user" && strings.HasPrefix(strings.TrimSpace(msg.Content), "[system note]:") {
				continue
			}
			break
		}
		if msg.ToolName != "write_file" {
			continue
		}
		if !strings.Contains(msg.Content, "already exists") {
			continue
		}
		// Pull the path from the rejection text so we can sniff the ext.
		// The rejection format (see surgical-edit gate) is:
		//   "File <path> already exists (<n> lines). ..."
		const pfx = "File "
		s := msg.Content
		idx := strings.Index(s, pfx)
		if idx < 0 {
			continue
		}
		s = s[idx+len(pfx):]
		spaceIdx := strings.Index(s, " ")
		if spaceIdx < 0 {
			continue
		}
		path := s[:spaceIdx]
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".py" && ext != ".html" && ext != ".htm" {
			return nil, ""
		}
		// Ban write_file (just got rejected) and edit_file (the wrong
		// shortcut the model is biased toward). Leave structural_edit and the
		// read/run/etc tools available.
		return []string{"edit_file", "write_file"}, ext
	}
	return nil, ""
}

// eraseLlamaSlot clears llama.cpp's KV slots to give the next chat
// completion a fresh prefix. Errors are logged and
// swallowed — slot erase is a best-effort isolation step, not a
// correctness requirement.
//
// All slots are erased, not just slot 0. With --parallel > 1 and prompt
// caching on, llama-server picks a slot per request by prefix match /
// LRU, so a new session can land on slot 1..N-1. If only slot 0 were
// cleared, those other slots would still hold a prior session's KV and
// reuse it — the exact cross-session bleed this prevents.
func eraseLlamaSlot(ctx *AgentContext) {
	llamaURL := envOr("ATLAS_LLAMA_URL", ctx.InferenceURL)

	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	client := &http.Client{Timeout: 5 * time.Second}

	erased := 0
	slots := parallelSlots()
	var stale []int
	for id := 0; id < slots; id++ {
		// llama-server handles the erase on its main loop, so a slot that
		// is mid-decode answers only once it frees up. A single 5s attempt
		// lost one slot in half the sessions of the 2026-08-03 run (13 of
		// 26 cleared 3 of 4), and the one it lost is precisely the one
		// still holding the previous session's KV.
		if eraseOneSlot(reqCtx, client, llamaURL, id) {
			erased++
		} else {
			stale = append(stale, id)
		}
	}
	if len(stale) > 0 {
		// Claiming a fresh cache here would describe the intent rather
		// than the result: an un-erased slot can be picked by prefix match
		// and reuse a prior session's KV, which is the bleed this exists
		// to prevent.
		log.Printf("[agent] erased %d/%d llama slots — slot(s) %v still hold prior KV "+
			"and may be reused by prefix match", erased, slots, stale)
		return
	}
	log.Printf("[agent] erased %d/%d llama slots — fresh KV cache for this session", erased, slots)
}

// eraseOneSlot clears a single KV slot, retrying while it is busy. Reports
// whether the slot ended up clear.
func eraseOneSlot(reqCtx context.Context, client *http.Client, llamaURL string, id int) bool {
	endpoint := fmt.Sprintf("%s/slots/%d?action=erase", llamaURL, id)
	const attempts = 3
	for attempt := 1; attempt <= attempts; attempt++ {
		req, err := http.NewRequestWithContext(reqCtx, "POST", endpoint, nil)
		if err != nil {
			log.Printf("[agent] erase slot %d: build request failed: %v", id, err)
			return false
		}
		resp, err := client.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return true
			}
			log.Printf("[agent] erase slot %d: status %d (attempt %d/%d)",
				id, resp.StatusCode, attempt, attempts)
		} else {
			log.Printf("[agent] erase slot %d: request failed: %v (attempt %d/%d)",
				id, err, attempt, attempts)
		}
		if reqCtx.Err() != nil {
			return false
		}
		if attempt < attempts {
			select {
			case <-reqCtx.Done():
				return false
			case <-time.After(time.Duration(attempt) * 500 * time.Millisecond):
			}
		}
	}
	return false
}

// pollPromptProgress emits llm_prompt_progress events at 100ms cadence
// while llama-server is in the prompt-eval phase of a streaming chat
// completion. Without these events the TUI freezes on "encoding prompt…"
// for the 30–90s prompt-eval window on long histories.
//
// Always emits elapsed_ms so the TUI can show a live timer ("encoding
// prompt · 12.3s"). Additionally tries to extract processed/total/pct
// from llama.cpp's /slots endpoint — those fields are only present in
// some llama.cpp builds (n_prompt_tokens_processed / n_prompt_tokens).
// When absent, the TUI renders a spinner-with-timer rather than a bar.
//
// Stops when stop is closed (the caller closes it on first-token
// arrival, on function return, or on context cancel).
//
// totalEst is the chars/4 prompt-token estimate; passed through to the
// TUI as `total_est` so even without /slots data the user sees the
// rough magnitude of what's being encoded.
func pollPromptProgress(ctx *AgentContext, llamaURL string, stop <-chan struct{}, totalEst int) {
	// Defense in depth: if anything panics inside this goroutine
	// (e.g. a write to a closed flusher) don't take the whole proxy
	// down with it. The WaitGroup in callLLMOnce should prevent the
	// race that makes this possible, but a recover here is cheap.
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[agent] pollPromptProgress recovered: %v", r)
		}
	}()
	startedAt := time.Now()
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	// Once /slots returns 404/501 we stop probing it but keep emitting
	// elapsed-time progress events — the timer is the useful signal,
	// the bar is the bonus.
	slotsAvailable := true
	for {
		select {
		case <-stop:
			return
		case <-ctx.Ctx.Done():
			return
		case <-ticker.C:
		}
		elapsed := time.Since(startedAt).Milliseconds()
		processed, total := 0, 0
		if slotsAvailable {
			processed, total, slotsAvailable = probeSlot(ctx.Ctx, client, llamaURL)
		}
		if total == 0 {
			total = totalEst
		}
		pct := 0.0
		if processed > 0 && total > 0 {
			pct = float64(processed) / float64(total)
			if pct > 1 {
				pct = 1
			}
		}
		ctx.Stream("llm_prompt_progress", map[string]interface{}{
			"processed":  processed,
			"total":      total,
			"pct":        pct,
			"elapsed_ms": elapsed,
		})
	}
}

// probeSlot does one /slots GET and pulls out prompt-eval counters when
// llama.cpp exposes them. Returns (processed, total, stillAvailable);
// stillAvailable goes false on 404/501 so the caller can stop probing.
func probeSlot(ctx context.Context, client *http.Client, llamaURL string) (int, int, bool) {
	reqCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "GET", llamaURL+"/slots", nil)
	if err != nil {
		return 0, 0, true
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, true // transient — try again next tick
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusNotImplemented {
		return 0, 0, false // /slots disabled — give up
	}
	if resp.StatusCode != http.StatusOK {
		return 0, 0, true
	}
	var slots []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return 0, 0, true
	}
	for _, s := range slots {
		if isProc, ok := s["is_processing"].(bool); ok && !isProc {
			continue
		}
		var processed, total int
		for _, k := range []string{"n_prompt_tokens_processed", "prompt_n", "n_past"} {
			if v, ok := s[k].(float64); ok && v > 0 {
				processed = int(v)
				break
			}
		}
		for _, k := range []string{"n_prompt_tokens", "n_prompt"} {
			if v, ok := s[k].(float64); ok && v > 0 {
				total = int(v)
				break
			}
		}
		return processed, total, true
	}
	return 0, 0, true
}

// llmStreamClient is a long-lived HTTP client for streaming LLM calls.
// Streaming responses can run for many minutes (a 4k-token write_file
// generation at ~30 tok/s is ~2min, longer for big content). The old
// 3-minute total Client.Timeout aborted those mid-decode with
// "context deadline exceeded while awaiting headers". Streaming mode
// also makes the total-timeout meaningless: we instead bound only the
// dial + header phases and rely on ctx.Ctx for user-initiated cancel.
//
// ResponseHeaderTimeout note: llama.cpp doesn't flush HTTP response
// headers until the FIRST decoded token arrives — i.e., header time
// = prompt eval time. With a long conversation history (e.g. a 767-line
// HTML file the assistant just wrote, ~8500 tokens) prompt eval can
// take ~60s on the GPU. A tight ResponseHeaderTimeout would cancel
// these legitimate calls. Bumped to 10 min: still bounds a truly hung
// llama-server, but tolerates large prompts. User Ctrl+C still works
// via the request context for any in-flight call.
// May 10 2026: ResponseHeaderTimeout removed. V3 pipelines that fire
// on T2+ edits routinely take 5-15 minutes between when the proxy
// posts and when llama-server flushes the first response header (it
// flushes on first decoded token, but prompt eval after a long V3
// run can take ages on cold KV state). The 10-minute cap fired on
// turn 5 of a real session and killed an otherwise-working chain.
// User instruction was to remove the timeout stuff completely.
// Dial timeout stays — connection refused / DNS failure should still
// fail fast; that's a different failure mode from "server is working
// but slow." Request-context cancellation via ctx.Ctx still works,
// so user-initiated cancels still propagate.
var llmStreamClient = &http.Client{
	Transport: &http.Transport{
		DialContext:     (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
		IdleConnTimeout: 90 * time.Second,
	},
}

// callLLMOnce is one round-trip to llama-server's /v1/chat/completions.
// Extracted from callLLMConstrained so the empty-response retry can
// reuse the same plumbing with a different temperature + message list.
//
// Uses SSE streaming so the proxy can forward per-token deltas to the
// TUI as `llm_token` events. The first delta also fires `llm_first_token`
// with the prompt-eval duration — that gap (request sent → first token)
// is llama-server doing prompt processing, which the user couldn't see
// before. Streaming mode also removes the 3-minute total-request timeout
// that was killing long generations on a single write_file with
// substantial content (HTML mockups, code with imports, etc.).
func callLLMOnce(ctx *AgentContext, messages []AgentMessage, temperature float64) (string, int, error) {
	return callLLMOnceWithGrammar(ctx, messages, temperature, "")
}

// callLLMOnceWithGrammar is callLLMOnce with an optional GBNF grammar
// override. When grammar != "", llama-server enforces it at the
// token-decode level (BiasBusters #2 — banning edit_file/write_file from
// the tool-name production for a single decision). The json_object
// response_format is dropped in that case because GBNF is the more
// specific constraint and supersedes it.
// toWireMessages converts the agent's internal messages to the role/content
// pairs sent on /v1/chat/completions.
//
// Tool results are rendered as a USER turn. Some chat templates have no `tool`
// role and silently drop role:"tool" messages — the model never sees the
// result (verified: the prompt carries only the user/assistant turns and the
// model reasons "the tool output is not visible"), so it re-issues the same
// tool call forever until the repetition breaker fires. This was the real
// cause behind every "it can't see what it's reading / it just loops" report.
// Every chat template handles the user role, so converting here is
// model-agnostic; the `[tool result]` marker
// tells the model this is tool output, not a fresh user instruction.
// ctx.Messages keeps the semantic "tool" role so trim-pinning and the
// step/traceback exclusions that key off ToolName still work.
func toWireMessages(messages []AgentMessage) []map[string]string {
	wire := make([]map[string]string, len(messages))
	for i, msg := range messages {
		role := msg.Role
		content := msg.Content
		if role == "tool" {
			role = "user"
			content = "[tool result] " + content
		}
		wire[i] = map[string]string{"role": role, "content": content}
	}
	return wire
}

// applyRepetitionSampling sets the repetition-control sampler fields on an
// outgoing llama-server request.
//
// llama-server ships every repetition control off: querying /props on a
// running instance reports repeat_penalty=1.0, dry_multiplier=0.0,
// frequency_penalty=0.0, presence_penalty=0.0. Nothing bounded how long a
// generation could repeat itself, which is what the stream-level
// isLoopingTail cut in callLLMOnceWithGrammar exists to catch. That cut is
// a backstop; it does not stop the model entering the loop.
//
// DRY rather than repeat_penalty. repeat_penalty scores individual token
// reoccurrence, which is wrong for code: indentation, `return`, `self.`,
// and closing braces legitimately repeat many times in one file. DRY scores
// repeated *sequences*, and llama.cpp treats "\n" as a sequence breaker by
// default, so per-line repetition across lines is not penalized at all.
// dry_allowed_length is raised above llama.cpp's default of 2 for the same
// reason — 3-token runs are ordinary in source.
//
// DRY now defaults OFF, because it penalises the one thing an edit tool needs
// most: copying an anchor out of a file the model just read.
//
// The old comment below the multiplier claimed dry_penalty_last_n bounded the
// scan to the current generation. That is false. llama-server's
// ServerSlot::init_sampler() calls common_sampler_accept() for every PROMPT
// token, and in common/sampling.cpp the is_generated flag gates only the
// grammar and reasoning-budget samplers — llama_sampler_accept(gsmpl->chain,
// token) is unconditional. So the ring buffer is filled with the tail of the
// prompt, which is exactly the read_file result the model is copying from.
//
// The penalty is multiplier * base^(matched - allowed_length), subtracted from
// the logit: -2.45 at 8 matched tokens, -7.51 at 10, -23.0 at 12, -123 at 15.
// Copying is by definition "extending a sequence that already occurred in the
// input", so around token 10-12 of an anchor the correct continuation is pushed
// below the runner-up, the model takes the branch that BREAKS the match, the
// match length resets and the penalty collapses. Observed as
// scoreElement -> scorerElement, food.y -> hood.y, unshift(( .
//
// Set ATLAS_DRY_MULTIPLIER=0.8 to restore the old behaviour.
func applyRepetitionSampling(reqBody map[string]interface{}) {
	dryMultiplier := envFloatOr("ATLAS_DRY_MULTIPLIER", 0)
	if dryMultiplier > 0 {
		reqBody["dry_multiplier"] = dryMultiplier
		reqBody["dry_base"] = envFloatOr("ATLAS_DRY_BASE", 1.75)
		reqBody["dry_allowed_length"] = envIntOr("ATLAS_DRY_ALLOWED_LENGTH", 6)
		// Bound the lookback so DRY scans the current generation rather
		// than the whole 32k window; -1 (llama.cpp's default) would make
		// every prior turn's text a repetition source.
		reqBody["dry_penalty_last_n"] = envIntOr("ATLAS_DRY_PENALTY_LAST_N", 2048)
	}
	if rp := envFloatOr("ATLAS_REPEAT_PENALTY", 1.0); rp != 1.0 {
		reqBody["repeat_penalty"] = rp
		reqBody["repeat_last_n"] = envIntOr("ATLAS_REPEAT_LAST_N", 64)
	}
}

// envFloatOr reads a float tunable from the environment, falling back to def
// when unset or unparseable.
func envFloatOr(key string, def float64) float64 {
	if v := envOr(key, ""); v != "" {
		if f, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
			return f
		}
	}
	return def
}

// envIntOr reads an int tunable from the environment, falling back to def
// when unset or unparseable.
func envIntOr(key string, def int) int {
	if v := envOr(key, ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n
		}
	}
	return def
}

// restatementMaxBytes caps the restated file. Past this the copy is not the
// bottleneck anyway, and a big paste costs prompt-processing time on every turn.
const restatementMaxBytes = 24000

// appendLastReadRestatement puts the most recently read file back at the END of
// the message list, immediately before the generation point.
//
// Why this helps: the read_file result the model must copy from sits behind the
// system prompt, every tool description, and every prior turn — thousands of
// tokens back. The model's own partially-emitted copy sits at the very end. So
// the two candidate sources for "what comes next" are not equally reachable,
// and the near one wins more often as the copy lengthens. That asymmetry is the
// most plausible account of the corruptions seen in practice
// (food.y -> hood.y, scoreElement -> scorerElement, unshift(( ).
//
// This does not depend on which mechanism is responsible — shortening the
// distance between the source span and the generation point helps under any
// account of long-range retrieval degradation, and it is free.
//
// Skipped when nothing has been read, when the file is large, and when the same
// content is already the last message (which is the common case immediately
// after a read_file, where restating would only duplicate it).
// ATLAS_RESTATE_LAST_READ=0 disables.
func appendLastReadRestatement(ctx *AgentContext, wire []map[string]string) []map[string]string {
	if ctx == nil || envOr("ATLAS_RESTATE_LAST_READ", "1") == "0" {
		return wire
	}
	path, content := ctx.LastRead()
	if path == "" || content == "" || len(content) > restatementMaxBytes {
		return wire
	}
	// Already in the window — don't pay for it twice. trimMessages PINS the
	// most recent file-content tool result so the active file survives
	// trimming, so for the file being edited this is the normal case, and
	// restating it appended a second full copy of something already present.
	for _, m := range wire {
		if strings.Contains(m["content"], content) {
			return wire
		}
	}
	// Fit it in what the slot actually has left. The budget is computed over
	// ctx.Messages and applied by trimMessages; this block is appended to the
	// WIRE afterwards, so nothing counted it. On a 2000-line fixture the
	// line-numbered copy runs ~4700 tokens, and aoc_sonar died at turn 3 in
	// both reps with `request (33012 tokens) exceeds the available context
	// size (32768)` — a 400 that ends the stream. Restating is an
	// optimisation; overflowing the slot is fatal, so it yields.
	used := 0
	for _, m := range wire {
		used += estTokens(m["content"])
	}
	headroom := perSlotContext() - agentMaxTokens() - used
	if headroom < estTokens(content)+numberedLineOverhead(content) {
		log.Printf("[agent] skipping the last-read restatement of %s — no headroom (%d tokens left)",
			logPath(path), headroom)
		return wire
	}
	rel := path
	if ctx.WorkingDir != "" {
		if r, err := filepath.Rel(ctx.WorkingDir, path); err == nil && !strings.HasPrefix(r, "..") {
			rel = r
		}
	}
	var sb strings.Builder
	sb.WriteString("Current contents of ")
	sb.WriteString(rel)
	sb.WriteString(" (line numbers are for reference and are NOT in the file):\n")
	for i, line := range strings.Split(strings.TrimSuffix(content, "\n"), "\n") {
		fmt.Fprintf(&sb, "%d\t%s\n", i+1, line)
	}
	return append(wire, map[string]string{"role": "user", "content": sb.String()})
}

func callLLMOnceWithGrammar(ctx *AgentContext, messages []AgentMessage, temperature float64, grammar string) (string, int, error) {
	// Stale from the previous turn otherwise, which would blame a clean
	// parse failure on a cut that happened earlier.
	ctx.LastStreamCut = ""
	wireMessages := toWireMessages(messages)
	wireMessages = appendLastReadRestatement(ctx, wireMessages)

	llamaURL := envOr("ATLAS_LLAMA_URL", ctx.InferenceURL)

	// Per-turn hard ceiling (agentMaxTokens, default 8192). 32768 let a
	// rambling content blob run the full window (~18 min at the GPU's capped
	// decode rate) — the runaway nothing else caught, since the reasoning
	// budget only fires on reasoning-WITHOUT-content. An agent turn is a tool
	// call (small) or a whole-file write_file (a few thousand tokens); 8192
	// covers a ~600-line generation. Truncation recovery backstops the rare
	// legit overflow. conversationTokenBudget reserves this same value.
	reqBody := map[string]interface{}{
		"model":       modelName,
		"messages":    wireMessages,
		"temperature": temperature,
		"max_tokens":  agentMaxTokens(),
		"stream":      true,
		// Without include_usage, the final SSE chunk before [DONE] has no
		// usage block, so we can't report total_tokens to the TUI.
		"stream_options": map[string]bool{"include_usage": true},
		// Some reasoning-capable chat templates default to thinking, but the
		// agent loop relies on grammar-constrained JSON output — thinking
		// blocks would just bloat tokens and llama-server rejects the
		// combination outright once a trailing assistant message looks
		// like a "response prefill" (400: "Assistant response prefill is
		// incompatible with enable_thinking"). Disable explicitly.
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}
	applyRepetitionSampling(reqBody)

	// Transcription profile. An agent turn emits a tool call whose arguments
	// are largely COPIED from a file the model just read, and every sampler in
	// the default chain is tuned for open-ended prose: top_k 40 / top_p 0.95 /
	// min_p 0.05 all leave a live tail for a wrong-but-plausible token to be
	// drawn from, and that is what a near-miss identifier is.
	//
	// "samplers": ["top_k"] with top_k 1 builds a chain of exactly
	// logit_bias -> top_k(1) -> dist, so the penalty samplers are never
	// instantiated at all. Not temperature 0: since llama.cpp PR #9897 temp<=0
	// is handled inside the temperature sampler, which sits LAST, so it returns
	// the argmax of an already-penalised distribution.
	//
	// ATLAS_TRANSCRIPTION_SAMPLER=0 restores the server defaults.
	if envOr("ATLAS_TRANSCRIPTION_SAMPLER", "1") != "0" {
		reqBody["samplers"] = []string{"top_k"}
		reqBody["top_k"] = 1
	}

	if grammar == rawEmissionSentinel {
		// Free-text reply: neither grammar nor response_format. Used for
		// the fenced-content sub-call, where the whole point is escaping
		// the JSON channel — measured on the served model, a debounce
		// solution parses 6/6 when emitted in a fenced block and 0/6 when
		// emitted inside a JSON string.
	} else if grammar != "" {
		// Token-level restriction wins over response_format. llama-server
		// rejects requests that pass both response_format=json_object and
		// a non-trivial grammar; pass only the grammar in restricted mode.
		reqBody["grammar"] = grammar
	} else {
		reqBody["response_format"] = buildResponseFormat()
	}
	body, _ := json.Marshal(reqBody)
	endpoint := llamaURL + "/v1/chat/completions"

	// Carry the agent's request context into the HTTP request so client
	// disconnects propagate down to llama-server.
	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	// Progress watchdog for the fenced sub-call. Cancelling reqCtx aborts the
	// HTTP request, which ends the scanner loop and closes the slot
	// server-side -- the same path a client disconnect already takes, so no
	// request or goroutine outlives it. Ordinary turns are untouched.
	progress := func() {}
	if grammar == rawEmissionSentinel {
		var cancel context.CancelFunc
		reqCtx, cancel = context.WithCancel(reqCtx)
		defer cancel()
		idle := fencedIdleTimeout()
		var mu sync.Mutex
		timer := time.AfterFunc(fencedFirstContentTimeout(), cancel)
		defer timer.Stop()
		progress = func() {
			mu.Lock()
			defer mu.Unlock()
			timer.Reset(idle)
		}
	}
	httpReq, err := http.NewRequestWithContext(reqCtx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return "", 0, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	// Don't reuse the TCP connection across turns. We were seeing
	// `Post ".../v1/chat/completions": EOF` failures in 0ms between
	// back-to-back turns: the previous streaming response left the
	// connection in a state llama-server (--parallel 1) closed at its
	// end, then the next turn's POST reused the dead idle connection
	// from Go's pool and got EOF on first read. Setting Close=true
	// forces a fresh dial per call. The dial overhead is negligible
	// next to a 5k-token prompt eval, and the reliability win is huge.
	httpReq.Close = true

	sentAt := time.Now()

	// Estimate total prompt tokens (chars/4 — works for English + code
	// within ~10–20%) so the prompt-progress poller has a baseline even
	// when /slots doesn't expose n_prompt_tokens directly.
	promptTokenEst := 0
	for _, m := range messages {
		promptTokenEst += len(m.Content) / 4
	}
	// pollPromptProgress runs as a sibling goroutine while the LLM call is
	// in flight; it streams elapsed_ms ticks back to the TUI. We MUST
	// guarantee it has fully exited before callLLMOnce returns — otherwise
	// it can call ctx.Stream (which writes to handleAgent's flusher) AFTER
	// handleAgent has returned and the response writer is invalid, causing
	// a SIGSEGV inside bufio.(*Writer).Flush. The defers run LIFO: stop
	// the channel first, then wait on the WaitGroup until the goroutine
	// exits.
	stopProgress := make(chan struct{})
	var stopOnce sync.Once
	stopProgressFn := func() { stopOnce.Do(func() { close(stopProgress) }) }
	var pollWG sync.WaitGroup
	pollWG.Add(1)
	go func() {
		defer pollWG.Done()
		pollPromptProgress(ctx, llamaURL, stopProgress, promptTokenEst)
	}()
	defer pollWG.Wait()
	defer stopProgressFn()

	resp, err := llmStreamClient.Do(httpReq)
	if err != nil {
		return "", 0, fmt.Errorf("LLM request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", 0, fmt.Errorf("LLM returned %d: %s",
			resp.StatusCode, truncateStr(string(respBody), 500))
	}

	var (
		contentBuf strings.Builder
		// PC-?: capture reasoning_content separately so we can fall
		// back to it when contentBuf is empty. Some models occasionally
		// engages thinking mode despite enable_thinking=false (most
		// reproducibly on retries with bumped temperature) — when it
		// does, ALL output streams into delta.reasoning_content. The
		// previous version threw it away and returned an empty string,
		// which fired the empty-response retry uselessly. Now we
		// surface the reasoning as content (with <think> tags stripped)
		// so the agent loop has SOMETHING to parse.
		reasoningBuf   strings.Builder
		totalTokens    int
		firstTokenSent bool
		reasoningCut   bool
		contentLoopCut bool
		noFenceCut     bool
		lastLoopCheck  int
	)

	// Per-turn reasoning budget. A reasoning-heavy model can spiral for
	// tens of thousands of tokens inside ONE generation (observed: a
	// 14-minute, ~17K-token deliberation over a 24-line file that ended
	// with no tool call) — max_tokens (32768) is the only bound and it
	// allows ~25 minutes of silence. When accumulated reasoning passes
	// the budget we stop reading; closing the response body cancels the
	// slot server-side. The post-loop recovery path then either extracts
	// a tool_call already present in the reasoning, or returns empty so
	// the caller's standard re-prompt ("emit your tool call now") fires.
	// Token-estimate at 4 chars/token; ATLAS_REASONING_BUDGET (tokens)
	// overrides, 0 disables. Keyed off stream state, not model identity.
	reasoningBudgetChars := 6144 * 4
	if v := envOr("ATLAS_REASONING_BUDGET", ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			reasoningBudgetChars = n * 4
		}
	}

	scanner := bufio.NewScanner(resp.Body)
	// Default scanner buffer is 64KB which is fine per line, but bump
	// the max in case llama-server emits a fat usage payload at the end.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				TotalTokens      int `json:"total_tokens"`
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Choices {
			if c.Delta.ReasoningContent != "" {
				// First output of ANY kind means prompt eval is done — for
				// reasoning models (some stream their whole chain as
				// reasoning_content, often with no content tokens until the
				// final JSON) the first delta is reasoning, not content.
				// Stop the prompt-eval poller and fire llm_first_token here
				// too; otherwise the poller keeps emitting prompt_progress
				// for the entire generation, the TUI keeps painting
				// "encoding", and it fights the streaming reasoning for the
				// row — the encode timer never stops and the screen flickers.
				if !firstTokenSent {
					stopProgressFn()
					ctx.Stream("llm_first_token", map[string]interface{}{
						"prompt_ms": time.Since(sentAt).Milliseconds(),
					})
					firstTokenSent = true
				}
				// Accumulate for the empty-content fallback below AND
				// stream to the TUI as a separate `reasoning_token` event
				// so users can see the model's thought process. The TUI
				// subscribes to reasoning_token distinctly from llm_token
				// so it can render thinking dimmed without mixing it into
				// the content stream destined for parse.
				reasoningBuf.WriteString(c.Delta.ReasoningContent)
				ctx.Stream("reasoning_token", map[string]interface{}{
					"text": c.Delta.ReasoningContent,
				})
				if reasoningBudgetChars > 0 && reasoningBuf.Len() > reasoningBudgetChars && contentBuf.Len() == 0 {
					reasoningCut = true
				}
			}
			if c.Delta.Content == "" {
				continue
			}
			// Useful progress, and the only kind: bytes of the file itself.
			// Reasoning deltas above deliberately do not reach here.
			progress()
			if !firstTokenSent {
				stopProgressFn() // prompt eval done — kill the poller
				ctx.Stream("llm_first_token", map[string]interface{}{
					"prompt_ms": time.Since(sentAt).Milliseconds(),
				})
				firstTokenSent = true
			}
			contentBuf.WriteString(c.Delta.Content)
			ctx.Stream("llm_token", map[string]interface{}{
				"text": c.Delta.Content,
			})
			// Content-loop cut. Some models state the right answer then
			// spirals on self-doubt in the CONTENT stream ("...the first line
			// is X. Wait, I can't see the output. I'll just say X. Wait, I
			// can't..." repeating) — the reasoning budget doesn't catch it
			// (that's content, not reasoning_content), so it ran to max_tokens.
			// Detect a verbatim repeating tail and cut. Checked periodically
			// to keep it O(n) overall.
			// Raw-emission sub-call: we asked for exactly one fenced block
			// and nothing else, so a reply with no fence opener after a few
			// hundred characters is prose that will run to max_tokens. At
			// 8192 tokens and ~25 tok/s that is ~5 minutes, and the fetch
			// makes two attempts, so a failed @fenced resolution cost a
			// user's session 10 minutes of silence before bouncing.
			// Measured: 03:21:25 request, 03:31:46 "no fenced block after 2
			// attempts" — 621 seconds, and 8 of 20 create sessions hit it.
			if grammar == rawEmissionSentinel && !noFenceCut &&
				contentBuf.Len() > rawFenceGraceChars &&
				!strings.Contains(contentBuf.String(), "```") {
				noFenceCut = true
			}
			if !contentLoopCut && contentBuf.Len() > 600 && contentBuf.Len()-lastLoopCheck > 200 {
				lastLoopCheck = contentBuf.Len()
				buffered := contentBuf.String()
				threshold := 3
				if strings.Contains(buffered, `"tool_call"`) || strings.Contains(buffered, "```") {
					// Code is legitimately self-similar; only spiral-grade
					// repetition is degeneration there. Covers both channels
					// code streams through: tool_call JSON args, and the
					// fenced block of the @fenced sub-call — without the
					// fence case the prose threshold would re-cut healthy
					// code in the channel built to avoid exactly that.
					threshold = toolCallLoopThreshold
				}
				if loopingTailCount(buffered) >= threshold {
					contentLoopCut = true
				}
			}
		}
		if chunk.Usage != nil && chunk.Usage.TotalTokens > 0 {
			totalTokens = chunk.Usage.TotalTokens
		}
		if reasoningCut {
			log.Printf("[agent] reasoning budget exceeded (%d chars, ~%d tokens) with no content emitted — cutting the stream and re-prompting",
				reasoningBuf.Len(), reasoningBuf.Len()/4)
			ctx.Stream("reasoning_budget_cut", map[string]interface{}{
				"reasoning_chars": reasoningBuf.Len(),
			})
			ctx.LastStreamCut = "reasoning_budget"
			break
		}
		if noFenceCut {
			log.Printf("[agent] raw sub-call produced %d chars with no fenced block — cutting rather than running to max_tokens",
				contentBuf.Len())
			ctx.LastStreamCut = "no_fence"
			break
		}
		if contentLoopCut {
			log.Printf("[agent] content loop detected (%d chars) — model repeating itself; cutting the stream", contentBuf.Len())
			ctx.Stream("content_loop_cut", map[string]interface{}{"chars": contentBuf.Len()})
			ctx.LastStreamCut = "content_loop"
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return contentBuf.String(), totalTokens,
			fmt.Errorf("read LLM stream: %w", err)
	}

	// Stash the reasoning content on ctx so the agent loop's per-turn
	// reasoning-repetition detector can compare it against prior turns.
	// We capture regardless of whether contentBuf was non-empty — the
	// model may emit BOTH content (the JSON tool call) AND reasoning
	// (the prose narration), and we want to detect rehashed reasoning
	// even when a tool call was successfully emitted.
	ctx.LastTurnReasoning = reasoningBuf.String()

	if contentBuf.Len() == 0 {
		// Raw-emission sub-call: the recovery below only salvages a JSON
		// tool_call envelope, which is the right rule for the agent loop and
		// the wrong one here — this call asked for a fenced block, so a
		// block sitting in reasoning_content is exactly the answer and was
		// being thrown away. Observed: sub-call attempts returning 0 content
		// characters after minutes of generation, with no stream cut, which
		// is tokens going somewhere that is not `content`.
		if grammar == rawEmissionSentinel && reasoningBuf.Len() > 0 {
			if body := extractFencedContent(reasoningBuf.String()); body != "" &&
				strings.TrimSpace(body) != rawEmissionSentinel {
				log.Printf("[agent] raw sub-call emitted its fenced block into reasoning_content (%d chars) — salvaging",
					reasoningBuf.Len())
				return reasoningBuf.String(), totalTokens, nil
			}
		}
		// No content deltas — check reasoning_content. Two distinct cases:
		//
		//   (a) Model dumped its actual response into the thinking
		//       stream despite template-level reasoning being disabled).
		//       reasoning_content contains a JSON
		//       tool_call; we recover it and parse normally.
		//
		//   (b) Model emitted ONLY thinking ("Now I need to read...")
		//       and terminated without producing a response. The
		//       reasoning_content is pure prose narration — there's no
		//       tool call to recover. Earlier we returned this prose
		//       as the "response" and it parse-errored every time,
		//       wasting a turn. Pre-May-8 behavior had the agent
		//       loop's classifyParseFailure scolding the model with
		//       "respond in JSON only" — but the corrective is
		//       useless when the response was truly empty (model
		//       wasn't disobeying the format, it just stopped mid-flow).
		//
		// New behavior: only return recovered reasoning when it
		// CONTAINS a tool_call envelope. For pure prose, return empty
		// + log so the caller's retry path can re-prompt with a
		// "you produced thinking but no response — emit your tool
		// call now" message instead of treating prose as a failed
		// response.
		if reasoningBuf.Len() > 0 {
			if recovered, ok := recoverStructuredReasoning(reasoningBuf.String()); ok {
				log.Printf("[agent] empty content but %d chars of reasoning_content contained a structured agent response — recovering",
					reasoningBuf.Len())
				return recovered, totalTokens, nil
			}
			// Pure prose narration in reasoning_content with no tool
			// call. Don't return it — let the caller retry. Logged so
			// the failure mode stays visible.
			log.Printf("[agent] %d chars of reasoning_content had no valid agent envelope — discarding so caller can re-prompt",
				reasoningBuf.Len())
		}
		// Truly nothing (or only narration). Caller's empty-response
		// retry path (callLLMConstrained) will handle.
		return "", totalTokens, nil
	}
	return contentBuf.String(), totalTokens, nil
}

// ---------------------------------------------------------------------------
// Permission checking
// ---------------------------------------------------------------------------

// validateTaskContract checks a client contract and returns the stored form,
// or an error and nothing at all.
//
// All or nothing, deliberately. A contract with one unusable path is a client
// that thinks it asked for something it did not, and honouring the half that
// parsed would produce obligations the user never stated. Normalising an
// invalid contract into an empty valid one would be worse still: it would look
// like a client that declared nothing.
//
// Paths go through resolveWorkspacePath, the same resolver every tool uses, so
// containment and canonical identity are decided in one place. That needs the
// request's working directory, which is why validation happens at the request
// boundary where the directory has already been resolved -- not in the decoder,
// which has no workspace to check against.
func validateTaskContract(in *TaskContract, workingDir string) (*TaskContract, error) {
	if in == nil {
		return nil, nil
	}
	switch in.TaskMode {
	case TaskModeWork, TaskModeQuestion:
	case "":
		return nil, fmt.Errorf("task_contract.task_mode is required")
	default:
		// Never coerced. An unrecognised mode is a client asking for something
		// this build does not implement.
		return nil, fmt.Errorf("task_contract.task_mode %q is not supported", in.TaskMode)
	}
	if len(in.ExpectedOutputs) > maxTaskContractEntries ||
		len(in.Verification) > maxTaskContractEntries {
		return nil, fmt.Errorf("task_contract exceeds %d entries", maxTaskContractEntries)
	}
	probe := &AgentContext{WorkingDir: workingDir}
	seen := map[string]bool{}
	out := &TaskContract{TaskMode: in.TaskMode}
	for _, p := range in.ExpectedOutputs {
		if strings.TrimSpace(p) == "" {
			return nil, fmt.Errorf("task_contract.expected_outputs contains an empty path")
		}
		canon, err := resolveWorkspacePath(probe, p)
		if err != nil {
			return nil, fmt.Errorf("task_contract.expected_outputs: %w", err)
		}
		if seen[canon] {
			continue // the same file spelled two ways is one obligation
		}
		seen[canon] = true
		out.ExpectedOutputs = append(out.ExpectedOutputs, p)
	}
	vseen := map[string]bool{}
	for _, v := range in.Verification {
		if strings.TrimSpace(v) == "" {
			return nil, fmt.Errorf("task_contract.verification contains an empty entry")
		}
		if vseen[v] {
			continue // deduplicated by exact identity, not by resemblance
		}
		vseen[v] = true
		out.Verification = append(out.Verification, v)
	}
	// Stable order, so two equivalent requests never disagree downstream.
	sort.Strings(out.ExpectedOutputs)
	sort.Strings(out.Verification)
	return out, nil
}

// --- Shadow comparison: what the client declared vs what ATLAS inferred ------
//
// Two records, because one cannot represent both. The request snapshot holds
// the immutable inputs -- the contract, the tier production already chose, each
// heuristic's own answer. The gate record holds an actual live decision, and
// there may be zero, one or several of those in a run: wantsStateChange reads
// inspectedWorkspace, which flips true once a read-only tool succeeds, so the
// same request legitimately answers false early and true later. A request-start
// approximation would be a different number from the one production used.
//
// Nothing here decides anything. Every function is called for its existing
// answer and the answer is recorded, not consulted.
// Record kinds version INDEPENDENTLY. Adding a field to one must not silently
// redefine another, and a sealed capture must stay readable by the analyzer
// written for the schema it was captured under.
//
// Gate v1 was legacy-observation only: it recorded what the heuristic said and
// nothing about what governed. Gate v2 adds the live action demand and the
// authority that produced it, so it is a different closed contract and carries
// a different number. The request snapshot and the footer did not change, so
// they stay at 1 rather than being bumped for tidiness.
const (
	shadowSchemaVersionRequest = 1
	shadowSchemaVersionGate    = 2
	shadowSchemaVersionFooter  = 1
)

// canonicalSource keeps an unknown decision source out of the record. The
// enum is closed and decideActionDemand can only produce its four members; if
// one ever escaped, it is written as the fail-closed member rather than as
// arbitrary prose.
func (s actionDemandSource) canonicalSource() string {
	switch s {
	case actionDemandLegacy, actionDemandContractWork,
		actionDemandContractQuestion, actionDemandContractInvalid:
		return string(s)
	default:
		return string(actionDemandContractInvalid)
	}
}

// shadowGateSite names the two live call sites, as a closed set.
type shadowGateSite string

const (
	shadowGateActionDemanded shadowGateSite = "action_demanded_and_unmet"
	shadowGateActionGate     shadowGateSite = "exit_action_gate"
)

// shadowComparison is the closed task-mode vocabulary. Only a gate record gets
// one, because only a gate record holds a live legacy decision.
const (
	shadowAgreeWork                  = "agree_work"
	shadowAgreeQuestion              = "agree_question"
	shadowContractWorkLegacyQuestion = "contract_work_legacy_question"
	shadowContractQuestionLegacyWork = "contract_question_legacy_work"
	shadowUnmeasured                 = "unmeasured"
)

// Declaration state and the two set-comparison vocabularies. There is no
// "invalid" state: an invalid contract is rejected at the request boundary and
// never reaches a run, so a record can only describe a contract that was
// declared or one that was absent.
const (
	shadowNotDeclared = "contract_not_declared"
	shadowDeclared    = "contract_declared"

	shadowOutputsExact        = "exact_agreement"
	shadowOutputsContractOnly = "contract_only"
	shadowOutputsLegacyOnly   = "legacy_only"
	shadowOutputsPartial      = "partial_overlap"
	shadowOutputsIncomparable = "incomparable"

	shadowVerifyLegacyRequires = "contract_declared_legacy_requires_verification"
	shadowVerifyLegacyDoesNot  = "contract_declared_legacy_does_not_require_verification"
)

// shadowHash is the stable identity used for joining and set comparison. It is
// never authority and never reversible to the original text.
func shadowHash(s string) string { return hashBytes([]byte(s))[:16] }

// shadowGateSeq is bounded per-request diagnostic state. Structurally unable to
// reach policy: nothing but the emitter reads it.
type shadowGateSeq struct{ n int }

// emitShadowRequestSnapshot records the immutable comparison inputs once per
// validated request. Every heuristic below is the existing function, called for
// the answer it already gives; no word list, regex or path rule is duplicated.
func emitShadowRequestSnapshot(ctx *AgentContext, userMessage string) {
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return // disabled: no hashing, no resolution, no heuristic calls
	}
	requestID := ""
	if ctx.Ctx != nil {
		requestID = requestIDFromContext(ctx.Ctx)
	}
	sink.noteRequest(requestID)

	tc := ctx.TaskContract
	rec := map[string]interface{}{
		"schema_version":           shadowSchemaVersionRequest,
		"record_kind":              "task_contract_shadow_request",
		"request_id":               requestID,
		"user_message_sha256":      hashBytes([]byte(userMessage)),
		"contract_present":         tc != nil,
		"tier":                     ctx.Tier.String(),
		"heuristic_action_intent":  isActionIntentMessage(userMessage),
		"heuristic_read_only":      isReadOnlyRequest(userMessage),
		"heuristic_explain_only":   isExplainOnlyMessage(strings.ToLower(userMessage)),
		"heuristic_question":       isQuestionMessage(userMessage),
		"heuristic_fix_intent":     isFixIntentMessage(userMessage),
		"influences_live_decision": false,
		"build_version":            APIVersion,
	}
	if tc != nil {
		rec["contract_provenance"] = "client"
		rec["task_mode"] = string(tc.TaskMode)
	}

	// Legacy deliverables, canonicalised through the resolver every tool uses.
	legacy := expectedOutputPaths(userMessage)
	legacyCanon, legacyFails := shadowCanonicalSet(ctx, legacy)
	rec["legacy_output_count"] = len(legacy)
	rec["legacy_output_hashes"] = shadowHashes(legacyCanon)

	declared := tc != nil && len(tc.ExpectedOutputs) > 0
	contractCanon, contractFails := shadowCanonicalSet(ctx, contractOutputs(tc))
	rec["canonicalization_failures"] = legacyFails + contractFails
	if declared {
		rec["output_declaration_state"] = shadowDeclared
		rec["output_count"] = len(tc.ExpectedOutputs)
		rec["output_hashes"] = shadowHashes(contractCanon)
		rec["output_comparison"] = shadowCompareSets(contractCanon, legacyCanon,
			contractFails+legacyFails > 0)
	} else {
		rec["output_declaration_state"] = shadowNotDeclared
		rec["output_count"] = 0
		rec["output_comparison"] = shadowNotDeclared
	}

	// Verification: the legacy side is a boolean demand and never a command,
	// so exact agreement is not a claim this can make.
	if tc != nil && len(tc.Verification) > 0 {
		rec["verification_declaration_state"] = shadowDeclared
		rec["verification_count"] = len(tc.Verification)
		rec["verification_hashes"] = shadowHashes(tc.Verification)
		if isFixIntentMessage(userMessage) {
			rec["verification_comparison"] = shadowVerifyLegacyRequires
		} else {
			rec["verification_comparison"] = shadowVerifyLegacyDoesNot
		}
	} else {
		rec["verification_declaration_state"] = shadowNotDeclared
		rec["verification_count"] = 0
		rec["verification_comparison"] = shadowNotDeclared
	}
	sink.submit(rec)
}

func contractOutputs(tc *TaskContract) []string {
	if tc == nil {
		return nil
	}
	return tc.ExpectedOutputs
}

// shadowCanonicalSet resolves each path through resolveWorkspacePath -- the one
// canonicalisation rule -- and counts what could not be resolved.
func shadowCanonicalSet(ctx *AgentContext, paths []string) ([]string, int) {
	seen := map[string]bool{}
	var out []string
	fails := 0
	for _, p := range paths {
		canon, err := resolveWorkspacePath(ctx, p)
		if err != nil {
			fails++
			continue
		}
		if !seen[canon] {
			seen[canon] = true
			out = append(out, canon)
		}
	}
	sort.Strings(out)
	return out, fails
}

func shadowHashes(items []string) []string {
	out := make([]string, 0, len(items))
	for _, it := range items {
		out = append(out, shadowHash(it))
	}
	return out
}

// shadowCompareSets classifies two canonical sets.
func shadowCompareSets(contract, legacy []string, failed bool) string {
	if failed {
		return shadowOutputsIncomparable
	}
	inLegacy := map[string]bool{}
	for _, l := range legacy {
		inLegacy[l] = true
	}
	overlap := 0
	for _, c := range contract {
		if inLegacy[c] {
			overlap++
		}
	}
	switch {
	case overlap == len(contract) && overlap == len(legacy):
		return shadowOutputsExact
	case overlap == 0 && len(legacy) == 0:
		return shadowOutputsContractOnly
	case overlap == 0 && len(contract) == 0:
		return shadowOutputsLegacyOnly
	case overlap == 0:
		return shadowOutputsIncomparable
	default:
		return shadowOutputsPartial
	}
}

// observeActionDemand records one live action-demand decision and returns it
// unchanged. The observer cannot alter the answer: it receives a decision that
// has already been made and hands the same value back.
//
// The record carries BOTH the legacy heuristic's answer and the live decision,
// so a capture shows what the old signal would have said next to what actually
// governed. influences_live_decision stays false because it describes this
// observer and its sink, not whether the contract is authoritative.
func observeActionDemand(ctx *AgentContext, st *runState, site shadowGateSite,
	d actionDemand) bool {
	sink := activeShadowSink.Load()
	if !sink.enabled() {
		return d.Required
	}
	st.shadowGate.n++
	requestID := ""
	if ctx.Ctx != nil {
		requestID = requestIDFromContext(ctx.Ctx)
	}
	comparison := shadowUnmeasured
	mode := ""
	if tc := ctx.TaskContract; tc != nil {
		mode = string(tc.TaskMode)
		switch {
		case tc.TaskMode == TaskModeWork && d.Legacy:
			comparison = shadowAgreeWork
		case tc.TaskMode == TaskModeQuestion && !d.Legacy:
			comparison = shadowAgreeQuestion
		case tc.TaskMode == TaskModeWork && !d.Legacy:
			comparison = shadowContractWorkLegacyQuestion
		case tc.TaskMode == TaskModeQuestion && d.Legacy:
			comparison = shadowContractQuestionLegacyWork
		}
	}
	sink.submit(map[string]interface{}{
		"schema_version":            shadowSchemaVersionGate,
		"record_kind":               "task_contract_shadow_gate",
		"request_id":                requestID,
		"gate_seq":                  st.shadowGate.n,
		"call_site":                 string(site),
		"inspected_workspace":       st.inspectedWorkspace,
		"tier":                      ctx.Tier.String(),
		"legacy_wants_state_change": d.Legacy,
		"live_action_demand":        d.Required,
		"action_demand_source":      d.Source.canonicalSource(),
		"contract_task_mode":        mode,
		"comparison":                comparison,
		"influences_live_decision":  false,
	})
	return d.Required
}

// needsPermission returns true if the tool call requires user confirmation.
func needsPermission(ctx *AgentContext, toolName string, args json.RawMessage) bool {
	// Deleting is decided per object, so no blanket answer substitutes for it.
	// yolo, the yolo flag and session_allowed_tools all answer "may this TOOL
	// run", which is a different question from "may this file be removed", and
	// a session that cannot ask therefore cannot delete. That is the intended
	// cost: the alternative is an unattended run destroying a file nobody
	// approved. Every other tool keeps its existing semantics.
	if toolName == "delete_file" {
		return true
	}
	if ctx.YoloMode || ctx.PermissionMode == PermissionYolo {
		return false
	}

	tool := getTool(toolName)
	if tool == nil {
		return true // unknown tool always requires permission
	}

	// Read-only tools never need permission
	if tool.ReadOnly {
		return false
	}

	// Tools the client pre-approved for the session (or the user approved
	// with session scope earlier this turn) skip the prompt.
	if ctx.isToolAllowed(toolName) {
		return false
	}

	// In accept-edits mode, file writes and edits are auto-approved;
	// run_command and delete_file still prompt.
	if ctx.PermissionMode == PermissionAcceptEdits {
		if toolName == "write_file" || toolName == "edit_file" || toolName == "structural_edit" || toolName == "move_file" {
			return false
		}
	}

	// Destructive tools need permission in default mode
	return tool.Destructive
}

// ---------------------------------------------------------------------------
// System prompt construction
// ---------------------------------------------------------------------------

func buildSystemPrompt(ctx *AgentContext) string {
	var sb strings.Builder

	sb.WriteString("You are ATLAS, a coding assistant that creates and modifies code by calling tools. ")
	sb.WriteString("You have access to the filesystem and can run commands to verify your work.\n")
	sb.WriteString("You MUST respond with ONLY a single valid JSON object, no other text.\n\n")

	// Pick-the-right-shape guidance — this is what keeps "hi" out of the
	// tool-call rabbit hole. Without it the model treats every input as a
	// task and starts read_file'ing random paths.
	sb.WriteString("## Choosing your response shape\n\n")
	sb.WriteString("- **Conversational input** (greetings, small talk, questions about YOU, status checks): emit `{\"type\":\"text\",\"content\":\"...\"}` — the turn ends after one text reply, and the user can follow up. Do NOT call tools to answer \"hi\" or \"what can you do\".\n")
	sb.WriteString("- **Questions about the CODE** (\"what does X do\", \"why is this slow\", \"is this a bug\") are NOT that case: read the file first, then answer in one `text` reply. You have read_file and outline_file — use them. Never ask the user to paste a file that is already in the workspace, and never answer from a guess about code you have not opened. Reading to answer a question is not \"starting work\": make no edits unless the user asked for one.\n")
	sb.WriteString("- **Coding tasks** (\"fix the bug\", \"add a feature\", \"refactor X\"): emit `{\"type\":\"tool_call\",...}` to make progress, repeat as needed, then emit `{\"type\":\"done\",\"summary\":\"...\"}` when finished.\n")
	sb.WriteString("- **Don't use `text` mid-task.** Roll narration into the done.summary at the end, or skip it entirely. Mid-task `text` ends the turn early.\n")
	sb.WriteString("- **When unsure** whether the user wants chat or work: ask in a single `text` reply. Don't speculatively start tool-calling — but reading a file the user named is never speculative.\n\n")

	// Tool descriptions.
	sb.WriteString(buildToolDescriptionsExcluding(nil))

	// Rules
	sb.WriteString("## Rules\n\n")
	sb.WriteString("- **Writing a whole file**: in write_file, set content to EXACTLY the 7 characters `@fenced` and NOTHING else — do NOT put the file itself in the JSON. After the tool call you will be asked for the file; reply then with ONE fenced code block containing the complete file. Code inside a JSON string gets corrupted by escaping (lost parens, broken newlines); the fenced reply is the reliable channel. Only trivially short content (under 5 lines) may be inlined in the JSON.\n")
	sb.WriteString("- To work on an EXISTING file, `read_file` it. Reading is the default and the context window is large; `read_file` caps itself on a file too big to load and tells you how to narrow the range, so you do not need to ration reads. Reach for `outline_file` only to locate a target inside a file that large — it returns line ranges and no code, so it can tell you where to read but never what the code does. Never re-read the same file in a loop; if a read's content is already in the conversation, act on it.\n")
	sb.WriteString("- Always read the relevant code before editing it (outline_file → read_file, then edit_file/structural_edit).\n")
	sb.WriteString("- MANDATORY: Use `edit_file` (targeted old_str/new_str) for any change to a file that already exists, no matter how small. `write_file` is ONLY for creating brand-new files. The agent layer rejects every `write_file` call against an existing file >5 lines — your call won't execute and you'll get a tool error directing you to edit_file. Don't re-emit a whole file to change a few lines.\n")
	sb.WriteString("  Example — to add a None check to one branch, use:\n")
	sb.WriteString("    edit_file {\"path\":\"src/foo.py\",\"old_str\":\"if x == 0:\\n        return None\",\"new_str\":\"if x is None or x == 0:\\n        return None\"}\n")
	sb.WriteString("  NOT write_file with the entire file's new contents.\n")
	sb.WriteString("- For WHOLE-FUNCTION or WHOLE-ELEMENT rewrites, prefer `structural_edit` over `edit_file`. structural_edit takes a structural selector (`function:NAME`, `class:NAME`, `<tag>` for HTML) and replaces that one whole named block — no need to copy the existing function as old_str. Selector must match exactly one node; ambiguous selectors return an error so you can be more specific. Decorators are included automatically when selecting a Python function. Works on `.py`, `.html`/`.htm`, `.go`, `.ts`/`.tsx` and `.js`/`.jsx`; for Go `function:NAME` matches a func or a method and `type:NAME` a type, and for JS/TS `function:NAME` matches a declaration, an arrow or a class method.\n")
	sb.WriteString("    structural_edit {\"path\":\"app.py\",\"selector\":\"function:dashboard\",\"content\":\"@app.route('/dashboard')\\ndef dashboard():\\n    return render_template('dashboard.html')\"}\n")
	sb.WriteString("    structural_edit {\"path\":\"templates/index.html\",\"selector\":\"<body>\",\"content\":\"<body>\\n  <h1>Welcome</h1>\\n  ...\\n</body>\"}\n")
	sb.WriteString("- WHEN write_file IS REJECTED for an existing file: if the file is `.py`, `.html`, or `.htm` and you're replacing the whole thing (e.g. swapping the entire body, replacing the dashboard function), use `structural_edit` next, not edit_file. structural_edit doesn't need `old_str` so it doesn't hit the max_tokens truncation that kills long edit_file calls. Use edit_file ONLY for surgical inline string changes (one line, one expression). For a change that spans several lines but is not a whole node, use `replace_lines` with the line numbers read_file printed — you assert only the FIRST and LAST line of the range, so there is no multi-line old_str to reproduce. This rule applies even when conversation trimming has dropped the original rejection message — re-derive the intent from the file extension and the size of your replacement.\n")
	sb.WriteString("- JSON strings in tool args contain LITERAL characters: write `<` not `&lt;`, `>` not `&gt;`, `&` not `&amp;`. The file content goes verbatim onto disk — `&lt;!DOCTYPE&gt;` would write the literal text `&lt;!DOCTYPE&gt;` instead of `<!DOCTYPE>`. NEVER HTML-encode angle brackets inside `content`, `old_str`, or `new_str`.\n")
	sb.WriteString("- The `content` you put in write_file / edit_file goes verbatim onto disk. **No markdown fences. No prose preamble (\"Looking at the task...\", \"Here's the file:\"). No trailing explanation.** Just the raw file contents. The agent layer strips fenced wrappers before writing, but the right move is to never emit them in the first place.\n")
	sb.WriteString("- For CONTENT changes, prefer the dedicated tools — `edit_file` (one line), `replace_lines` (a line range), `insert_after` (adding at a line), `structural_edit` (a whole node), `write_file` (new files) — they go through the validation pipeline. The last three need no old_str at all, which is why they hold up on changes edit_file loses. For moving / renaming / reorganizing files you may use either `move_file` or shell `mv`/`cp` via run_command; both work. `run_command` runs a real shell (in an isolated sandbox confined to this project), so ordinary file operations (mv, cp, mkdir, rm of a specific file, chmod) are fine. Only catastrophic commands are blocked: wiping the whole project (`rm -rf /`, `rm -rf .`, `rm -rf *`), fork bombs, and device/filesystem destruction.\n")
	sb.WriteString("- Use run_command to verify your changes (build, test, lint, curl). For \"fix\"/\"isn't working\" prompts, verify before `done`.\n")
	sb.WriteString("- For LONG-RUNNING commands (servers): `run_background(cmd)` → `run_command(\"curl ...\")` → `stop_background(job_id)`. Don't use `timeout 5 ... || true` — server dies before probe hits.\n")
	sb.WriteString("- When creating a project from scratch: create config/build files FIRST, verify they work (e.g., npm install, cargo check), THEN create feature code\n")
	sb.WriteString("- Respond with {\"type\":\"done\",\"summary\":\"...\"} when the task is complete\n")
	sb.WriteString("- If a command fails, read the error output, fix the issue, and try again\n")
	sb.WriteString("- Do not guess at file contents — read first, then edit\n")
	sb.WriteString("- ALWAYS use relative file paths (`app.py`, `src/main.rs`), NEVER absolute paths and NEVER prefix with `workspace/` — that's the parent dir, not your project root.\n")
	sb.WriteString("- When adding features to an existing project, read at most 2-3 files to understand the structure, then immediately write your changes. Do not explore the entire directory tree. Prioritize writing code over reading code.\n\n")

	// Project context
	if ctx.Project != nil {
		sb.WriteString("## Project Context\n\n")
		sb.WriteString(fmt.Sprintf("Language: %s\n", ctx.Project.Language))
		if ctx.Project.Framework != "" {
			sb.WriteString(fmt.Sprintf("Framework: %s\n", ctx.Project.Framework))
		}
		if ctx.Project.BuildCommand != "" {
			sb.WriteString(fmt.Sprintf("Build command: %s\n", ctx.Project.BuildCommand))
		}
		if ctx.Project.DevCommand != "" {
			sb.WriteString(fmt.Sprintf("Dev command: %s\n", ctx.Project.DevCommand))
		}
		if len(ctx.Project.ConfigFiles) > 0 {
			sb.WriteString(fmt.Sprintf("Config files: %s\n", strings.Join(ctx.Project.ConfigFiles, ", ")))
		}
		sb.WriteString("\n")
	}

	// Working directory
	sb.WriteString(fmt.Sprintf("Working directory: %s\n\n", ctx.WorkingDir))

	// Toolchain hints. Detect every recognized language manifest in
	// the project and surface the runners + install commands so the
	// model picks the right tool per file edit. Polyglot projects
	// (React + Django + deploy scripts) get one entry per ecosystem.
	// Covers every toolchain, not just Python's venv. Probe-first
	// hints — whether the deps are already importable — are added
	// per-toolchain when the evidence is on disk.
	if tcs := detectProjectToolchains(ctx.WorkingDir); len(tcs) > 0 {
		sb.WriteString("## Toolchains\n")
		for _, tc := range tcs {
			line := fmt.Sprintf("- **%s** — runner `%s`", tc.Name, displayRelativeRunner(tc.Runner, ctx.WorkingDir))
			if tc.InstallCommand != "" {
				line += fmt.Sprintf(", install `%s`", tc.InstallCommand)
			}
			if tc.TestCommand != "" {
				line += fmt.Sprintf(", tests `%s`", tc.TestCommand)
			}
			if probe := probeToolchainReady(ctx.WorkingDir, tc); probe != "" {
				line += " [" + probe + "]"
			}
			sb.WriteString(line + "\n")
		}
		sb.WriteString("Skip install when status is `ready`; install only what's missing.\n\n")
	}

	if ctx.VerifyOnHost {
		sb.WriteString("`run_command` targets the host (not sandbox). Sees host env/services/paths.\n\n")
	}

	// Show which files are in the project (names only, not full content).
	// Full content is available via read_file if needed.
	// This avoids consuming context window with pre-injected file dumps.
	if filesRead := ctx.SnapshotFilesRead(); len(filesRead) > 0 {
		sb.WriteString("## Project Files Available\n")
		for path := range filesRead {
			sb.WriteString(fmt.Sprintf("- %s\n", path))
		}
		sb.WriteString("\nUse read_file to inspect these files if needed. To MODIFY any of them, use edit_file — write_file against an existing file (>5 lines) is rejected at the agent layer.\n\n")
	}

	// Plan section. When the planner returned a plan, surface it so
	// the model has explicit step guidance instead of having to infer
	// the right shape from the user message alone. Plans are advisory
	// (the agent layer doesn't hard-block off-plan calls), but having
	// them in the system prompt visibly improves first-call accuracy.
	if ctx.Plan != nil && len(ctx.Plan.Steps) > 0 {
		sb.WriteString("## Plan\n\n")
		sb.WriteString("A planner has proposed these steps for the user's request. ")
		sb.WriteString("Follow them in order when sensible. ")
		sb.WriteString("Deviate only if a step's premise is wrong (file doesn't exist, command unavailable, etc.) — the agent layer notices repeated off-plan calls and will silently revise the plan with what you've discovered.\n\n")
		for i, step := range ctx.Plan.Steps {
			marker := " "
			if step.ID == ctx.Plan.VerifyStep {
				marker = "✓" // verify step
			}
			sb.WriteString(fmt.Sprintf("%d. [%s] **%s** %s — %s\n",
				i+1, marker, step.Action, step.Target, step.Why))
		}
		if ctx.Plan.Rationale != "" {
			sb.WriteString(fmt.Sprintf("\n_%s_\n", ctx.Plan.Rationale))
		}
		if ctx.Plan.VerifyStep != "" {
			sb.WriteString(fmt.Sprintf("\nThe verify step (%s) is your evidence the fix worked — don't emit `done` until it has run successfully.\n", ctx.Plan.VerifyStep))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// estTokens is a cheap, model-agnostic token estimate: ~4 chars/token plus
// a small per-message framing overhead. Good enough for budgeting; we leave
// generous headroom so the estimate never has to be exact.
func estTokens(content string) int {
	return len(content)/4 + 8
}

// conversationTokenBudget is how many prompt tokens the agent loop will let
// the conversation grow to before trimming. Derived from the deployment's
// per-slot context (ATLAS_CTX_SIZE / ATLAS_PARALLEL_SLOTS), reserving ~35%
// for the response. Model-agnostic: keys off the context the deploy gives,
// not the model identity. Falls back to a safe default when env is absent.
// perSlotContext is the token window one llama.cpp slot actually has:
// the server's context divided by the parallel slots it was started with.
// Shared so the history budget and the restatement agree on the limit they
// are both spending against.
func perSlotContext() int {
	ctxSize := 131072
	if v := envOr("ATLAS_CTX_SIZE", ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			ctxSize = n
		}
	}
	return ctxSize / parallelSlots()
}

// numberedLineOverhead is the cost of the "%d\t" prefix the restatement adds
// to every line — on a 2000-line file that is most of its size.
func numberedLineOverhead(content string) int {
	return estTokens(strings.Repeat("0000\t", strings.Count(content, "\n")+1))
}

func conversationTokenBudget() int {
	perSlot := perSlotContext()
	// Sliding window sized to the actual slot: reserve room for the model's
	// reply (max_tokens) plus a margin for system-prompt growth and tokenizer
	// slack, and give the REST of the slot to the conversation. The previous
	// flat 14k cap was too aggressive — on a 32k slot it left ~10k unused AND
	// dropped the file the model was editing, so weak models hallucinated
	// symbols/lines they could no longer see. The active file is additionally
	// pinned in trimMessages so it survives the window regardless. The
	// model-agnostic re-encode cost (SWA models re-process the prompt each
	// turn) is bounded by the slot itself; deploys that need it smaller can
	// still set ATLAS_AGENT_HISTORY_BUDGET.
	// Reserve: the model's reply (max_tokens), a fixed margin for
	// system-prompt growth, and a proportional tokenizer-slack margin.
	// estTokens is chars/4, which UNDER-counts dense content (code,
	// JSON-escaped tool results run closer to 3 chars/token) — without
	// the proportional slack the estimate can pass while the real
	// prompt exceeds the slot (observed: 32844 real vs 32768 slot).
	budget := perSlot - agentMaxTokens() - 2048 - perSlot/8
	if budget < 4000 {
		budget = 4000 // floor: tiny-context deploys still keep a usable window
	}
	// Optional hard ceiling — unset by default. Only set
	// ATLAS_AGENT_HISTORY_BUDGET to bound per-turn re-encode cost below the
	// slot capacity (trades retained context for faster turns on SWA models).
	if v := envOr("ATLAS_AGENT_HISTORY_BUDGET", ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 && n < budget {
			budget = n
		}
	}
	return budget
}

// isLoopingTail reports whether the content stream has degenerated into a
// verbatim repeating phrase — the signature of a model spiraling on the same
// sentence ("...the first line is X. Wait, I can't see the output. I'll just
// say X. Wait, I can't see..."). Takes a chunk from the tail and counts its
// occurrences; 3+ verbatim repeats is a loop a real response never produces.
func isLoopingTail(s string) bool {
	return loopingTailCount(s) >= 3
}

// loopingTailCount is how many times the stream's 48-char tail appears in
// the whole buffer. The threshold belongs to the CALLER because it depends
// on what is streaming. Prose degeneration ("...I'll just say X. Wait, I
// can't...") repeats until max_tokens, so 3 occurrences is already strong
// evidence. CODE is legitimately self-similar: a grid walker's four
// elif-direction branches, a debouncer's run_ bookkeeping lines — 48-char
// windows repeat 3-4 times in perfectly healthy files. With the threshold
// at 3 for everything, the detector cut healthy write_file drafts at the
// same structural spot every time: measured across one 50-task run, 17
// cuts, 10 of them truncating write_file code, and the two families whose
// code is most self-similar (walk, debounce) went 0/5 each — the cut
// stump landed, the model patched the cut line instead of rewriting, and
// the patch drifted (a comma in the print, spaces lost from a join).
func loopingTailCount(s string) int {
	const probe = 48
	if len(s) < probe*3 {
		return 0
	}
	tail := s[len(s)-probe:]
	if strings.TrimSpace(tail) == "" {
		return 0
	}
	return strings.Count(s, tail)
}

// toolCallLoopThreshold is the repeat count that counts as degeneration
// inside a tool_call stream. A real spiral runs to max_tokens — hundreds
// of repeats — so demanding 10 keeps the guard while making 3-4 branch-
// shaped repeats of healthy code invisible to it.
const toolCallLoopThreshold = 10

// agentMaxTokens is the per-turn generation ceiling (ATLAS_MAX_TOKENS,
// default 8192). Shared by the LLM request and conversationTokenBudget so the
// window and the reply reservation stay consistent.
func agentMaxTokens() int {
	maxTokens := 8192
	if v := envOr("ATLAS_MAX_TOKENS", ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			maxTokens = n
		}
	}
	return maxTokens
}

// parallelSlots returns the llama-server --parallel slot count for this
// deployment (ATLAS_PARALLEL_SLOTS), defaulting to 4 to match the
// entrypoint. Used both for KV-slot isolation and per-slot context math.
func parallelSlots() int {
	slots := 4
	if v := envOr("ATLAS_PARALLEL_SLOTS", ""); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			slots = n
		}
	}
	return slots
}

// pinnedIndices returns the two message indices trimMessages will keep
// regardless of the tail window: the most recent user message (the task)
// and the most recent file-content tool result (the file being edited).
// Either is -1 when absent. Shared by trimMessages (which re-injects
// them) and budgetedKeepLast (which must COUNT them — see below).
func pinnedIndices(msgs []AgentMessage) (pinIdx, filePinIdx int) {
	pinIdx, filePinIdx = -1, -1
	for i := len(msgs) - 1; i >= 1; i-- {
		if msgs[i].Role == "user" {
			pinIdx = i
			break
		}
	}
	for i := len(msgs) - 1; i >= 1; i-- {
		if msgs[i].Role == "tool" && (msgs[i].ToolName == "read_file" || msgs[i].ToolName == "outline_file") &&
			!strings.Contains(msgs[i].Content, "You already read") { // skip dedup pointers — they carry no content
			filePinIdx = i
			break
		}
	}
	return pinIdx, filePinIdx
}

// budgetedKeepLast returns how many trailing messages trimMessages should
// keep so the kept set (system + pinned user + pinned file + tail) fits the
// token budget. Floored at 8 (never trim more aggressively than the old
// fixed rule); when the whole conversation fits, returns len(msgs) so
// nothing is trimmed.
//
// The pinned messages MUST be pre-counted here: trimMessages re-injects
// them even when they fall outside the tail window, so a budget that
// ignored them under-counted the real prompt by the size of the pinned
// read_file — observed live as a llama-server 400 exceed_context_size
// (32844 > 32768 per-slot) hard-killing a bench session.
func budgetedKeepLast(msgs []AgentMessage) int {
	if len(msgs) == 0 {
		return 0
	}
	budget := conversationTokenBudget()
	used := estTokens(msgs[0].Content) // system prompt is always kept
	pinIdx, filePinIdx := pinnedIndices(msgs)
	if pinIdx >= 1 {
		used += estTokens(msgs[pinIdx].Content)
	}
	if filePinIdx >= 1 && filePinIdx != pinIdx {
		used += estTokens(msgs[filePinIdx].Content)
	}
	keep := 0
	for i := len(msgs) - 1; i >= 1; i-- {
		t := 0
		if i != pinIdx && i != filePinIdx { // already counted above
			t = estTokens(msgs[i].Content)
		}
		if used+t > budget && keep >= 8 {
			break
		}
		used += t
		keep++
	}
	if keep > len(msgs)-1 {
		keep = len(msgs) - 1
	}
	return keep
}

// trimMessages caps a conversation at roughly 1 (system) + 1 (pinned user) +
// keepLast tail messages, dropping the middle. The pin is the most recent
// role=="user" message — the user's current task. Without the pin, long agent
// loops (5+ tool calls) push the user's instruction off the end of the
// keepLast window, the model loses the task, and replies generically
// ("Hi! I'm ATLAS..."). If the pinned message already lives inside the tail
// window we don't duplicate it.
//
// Assumes msgs[0] is the system prompt.
func trimMessages(msgs []AgentMessage, keepLast int) []AgentMessage {
	if len(msgs) <= keepLast+1 {
		return msgs
	}

	// Pins (shared scan with budgetedKeepLast, which counts them): the
	// most-recent user message — the task — and the most-recent
	// file-content tool result (read_file / outline_file), so the file the
	// model is working on never gets trimmed out from under it. Without
	// the file pin, a long agent loop drops the file content, the model
	// edits BLIND, and a weak model then hallucinates symbols and old_str
	// that aren't in the file (observed live: structural_edit
	// function:count_items and edit_file old_str="return len(items)"
	// against a file containing neither, with the model literally
	// reasoning "I don't see the file content"). The exploration-budget
	// breaker compounds it by telling the model it "has full project
	// context" when the content was already trimmed.
	pinIdx, filePinIdx := pinnedIndices(msgs)

	tailStart := len(msgs) - keepLast
	out := make([]AgentMessage, 0, keepLast+3)
	out = append(out, msgs[0])
	if pinIdx >= 1 && pinIdx < tailStart {
		out = append(out, msgs[pinIdx])
	}
	// Re-inject the pinned file content (as a user-role note so it survives
	// templates that reject orphan tool messages) when it falls outside the
	// kept tail.
	if filePinIdx >= 1 && filePinIdx < tailStart && filePinIdx != pinIdx {
		out = append(out, AgentMessage{
			Role:    "user",
			Content: "[system note]: current contents of the file you are editing (do not invent symbols or lines not shown here):\n" + msgs[filePinIdx].Content,
		})
	}
	out = append(out, msgs[tailStart:]...)
	return out
}

// ---------------------------------------------------------------------------
// HTTP handler for /v1/agent endpoint
// ---------------------------------------------------------------------------

// handleAgent is the HTTP handler for the new agent endpoint.
func handleAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, ErrUnsupported, "method not allowed")
		return
	}

	type historyMsg struct {
		Role    string `json:"role"` // "user" or "assistant"
		Content string `json:"content"`
	}
	var req struct {
		Message    string       `json:"message"`
		WorkingDir string       `json:"working_dir"`
		Mode       string       `json:"mode"`       // "default", "accept-edits", "yolo"
		SessionID  string       `json:"session_id"` // optional — required for /cancel
		History    []historyMsg `json:"history,omitempty"`
		// Tools the client has approved for the whole session so the proxy
		// skips the interactive prompt for them (see /v1/permission).
		SessionAllowedTools []string `json:"session_allowed_tools,omitempty"`
		// /demo split-pane flags — tags match tui/chat.go's agentRequest.
		BypassV3         bool   `json:"bypass_v3,omitempty"`          // baseline pane: disable V3 orchestration
		DisableFreshSlot bool   `json:"disable_fresh_slot,omitempty"` // keep the pre-warmed KV prefix
		SandboxSubdir    string `json:"sandbox_subdir,omitempty"`     // confine writes to this workspace subdir
		// What the client declares about the request. Optional, and absent
		// stays distinguishable from present-and-empty. Nothing reads it yet.
		TaskContract *TaskContract `json:"task_contract,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "invalid request body")
		return
	}

	if req.Message == "" {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "message is required")
		return
	}

	// Path translation: the TUI sends its host cwd (e.g. /home/isaac/snake)
	// as working_dir, but the proxy runs in a container where that path
	// doesn't exist — only /workspace (the bind-mount target) does. The
	// startup wrapper (atlas/runtime.py:_align_workspace) already aligns
	// the bind mount to the user's cwd, so /workspace IS the user's cwd
	// from the proxy's perspective. Use ATLAS_WORKSPACE_DIR (set in
	// docker-compose.yml) as the canonical write target. The original
	// host path is kept on HostWorkingDir for path translation (below).
	hostDir := req.WorkingDir
	if hostDir == "" {
		hostDir = "."
	}
	workingDir := envOr("ATLAS_WORKSPACE_DIR", hostDir)

	// /demo: each pane works inside its own workspace subdir so the two
	// concurrent sessions can't clobber each other's files and the TUI's
	// post-run review finds each side's output where it expects it. The
	// subdir is a bare name (no separators, no traversal) or it's ignored.
	if sub := filepath.Clean(req.SandboxSubdir); req.SandboxSubdir != "" &&
		sub != "." && sub != ".." &&
		!strings.ContainsAny(sub, "/\\") {
		workingDir = filepath.Join(workingDir, sub)
		if hostDir != "" && hostDir != "." {
			hostDir = filepath.Join(hostDir, sub)
		}
	}

	// Classify tier from message
	tier := classifyAgentTier(req.Message)

	// Create agent context
	ctx := NewAgentContext(workingDir, tier)
	ctx.BypassV3 = req.BypassV3
	ctx.DisableFreshSlot = req.DisableFreshSlot
	// Stash the host path so resolveAgentPath can translate absolute
	// host paths the model receives in user prompts (e.g. "fix
	// /home/isaac/snake/app.py") into the container path. Without this
	// the model copies the user's host path verbatim into read_file
	// and the open() fails because that path doesn't exist inside the
	// proxy container — only /workspace does.
	if hostDir != "" && hostDir != "." {
		ctx.HostWorkingDir = filepath.Clean(hostDir)
	}
	ctx.InferenceURL = inferenceURL
	ctx.SandboxURL = sandboxURL
	ctx.LensURL = lensURL
	ctx.V3URL = envOr("ATLAS_V3_URL", "http://localhost:8070")

	// Opt-in host execution for run_command. Per-project config
	// (.atlas/config.toml: [execution] target = "host") wins over the
	// global env var so users can flip behaviour without touching the
	// proxy environment. Either source can downgrade to "sandbox"
	// explicitly. Default stays sandbox.
	ctx.VerifyOnHost = resolveVerifyTarget(workingDir) == "host"
	ctx.TrustMode = resolveTrustMode()

	// Seed prior-turn transcript from the request body. The TUI ships
	// user/assistant text rows from its local chat history so the agent
	// can answer follow-ups; without it, every /v1/agent call starts
	// fresh. Cap defensively at 40 messages here too — the proxy's own
	// trim logic in runAgentLoop handles further overflow.
	if n := len(req.History); n > 0 {
		if n > 40 {
			req.History = req.History[n-40:]
		}
		ctx.PriorHistory = make([]AgentMessage, 0, len(req.History))
		for _, h := range req.History {
			// Only accept the two roles that make sense as conversation
			// history; anything else is skipped silently rather than
			// passed through to the LLM as an unknown role.
			if h.Role != "user" && h.Role != "assistant" {
				continue
			}
			if h.Content == "" {
				continue
			}
			ctx.PriorHistory = append(ctx.PriorHistory, AgentMessage{
				Role:    h.Role,
				Content: h.Content,
			})
		}
	}
	// Carry the upstream cancellation through so disconnects abort the loop
	// and llama-server's in-flight generation.
	//
	// Also wrap in a cancellable context so POST /cancel can
	// abort even when the TCP disconnect is buffered upstream.
	reqCtx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Two lifetimes, deliberately separate.
	//
	// reqCtx is the RESPONSE lifetime: it lives until the client goes away or
	// the handler returns, and finalisation -- reaping, rehashing, restoring,
	// and the terminal event itself -- runs on it. workCtx is the WORK
	// lifetime: everything that costs time (LLM calls, tools, gates, V3, the
	// sandbox) hangs off it, and it ends one reserve before the session
	// budget does. Without the split, the deadline that stops the work also
	// kills the channel that would explain why it stopped.
	total, reserve := sessionBudget()
	workCtx, cancelWork := context.WithTimeout(reqCtx, total-reserve)
	defer cancelWork()
	ctx.RequestCtx = reqCtx
	ctx.Ctx = workCtx
	ctx.cancelWork = cancelWork
	ctx.PassID = req.SessionID
	if req.SessionID != "" {
		entry := &sessionCancel{cancel: cancel}
		activeSessions.Store(req.SessionID, entry)
		defer activeSessions.CompareAndDelete(req.SessionID, entry)
	}

	// Set permission mode
	switch req.Mode {
	case "accept-edits":
		ctx.PermissionMode = PermissionAcceptEdits
	case "yolo":
		ctx.PermissionMode = PermissionYolo
		ctx.YoloMode = true
	default:
		ctx.PermissionMode = PermissionDefault
	}

	// Seed session-approved tools so pre-approved destructive tools skip the
	// interactive prompt (the client re-sends this list each turn).
	if len(req.SessionAllowedTools) > 0 {
		ctx.AllowedTools = make(map[string]bool, len(req.SessionAllowedTools))
		for _, t := range req.SessionAllowedTools {
			ctx.AllowedTools[t] = true
		}
	}

	// Detect project (implemented in context.go)
	// The client's declaration, checked against the workspace this request
	// resolved to. A bad contract is a bad request: it is refused outright
	// rather than dropped, because a client that declared obligations and had
	// them silently discarded would be told its run finished having proved
	// none of them. Nothing reads the stored value yet.
	validatedContract, contractErr := validateTaskContract(req.TaskContract, workingDir)
	if contractErr != nil {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, contractErr.Error())
		return
	}
	ctx.TaskContract = validatedContract

	ctx.Project = detectProjectInfo(workingDir)

	// Set up SSE streaming
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, ErrInternal, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Flush headers immediately so the client sees the response as
	// "established" before the first LLM call returns. Without this
	// sentinel, net/http waits to flush headers until the first body
	// write, which is the first ctx.Stream() call — and that doesn't
	// happen until the agent loop emits its first event, which can
	// take 10-60s for the first LLM round-trip. Clients with a
	// reasonable ResponseHeaderTimeout (e.g. 30s) would time out
	// before getting any data.
	fmt.Fprintf(w, ": connected\n\n")
	flusher.Flush()

	// http.ResponseWriter is NOT goroutine-safe. StreamFn fires from at
	// least two concurrent goroutines during a single agent turn:
	//   - main agent loop (tool dispatch, LLM SSE forwarding)
	//   - pollPromptProgress (250ms ticker emitting llm_prompt_progress)
	// May 10 2026: adding reasoning_token doubled the event rate from
	// the SSE-decode loop, surfacing a long-latent race where
	// concurrent Write+Flush calls produced interleaved bytes that
	// corrupted the chunked-encoding framing — clients then errored
	// with "chunked line ends with bare LF" and dropped, which the
	// proxy saw as `context canceled` mid-prompt-eval. Serialize the
	// writes with a per-handler mutex so chunk framing stays
	// well-formed regardless of how fast or how concurrently events
	// fire.
	var streamMu sync.Mutex
	ctx.StreamFn = func(eventType string, data interface{}) {
		event := SSEEvent{Type: eventType, Data: data}
		eventJSON, _ := json.Marshal(event)
		streamMu.Lock()
		defer streamMu.Unlock()
		fmt.Fprintf(w, "data: %s\n\n", eventJSON)
		flusher.Flush()
	}

	// For yolo mode, auto-approve all permissions
	if ctx.YoloMode {
		ctx.PermissionFn = func(string, json.RawMessage) bool { return true }
	}

	// Run agent loop
	if err := runAgentLoop(ctx, req.Message); err != nil {
		// %q quotes the error string so user-influenced fragments
		// embedded in err.Error() can't fake additional log entries.
		log.Printf("[agent] error: %q", err.Error())
	}

	// Stash this pass's writes for deferred /feedback labeling (lens training
	// data). Keyed by session id; a later thumbs / per-file verdict turns them
	// into weighted samples. No-op when the pass wrote nothing or has no id.
	stashPendingPass(req.SessionID, modelName, ctx.PassWrites)

	// Label them mechanically too, when the run itself produced evidence.
	// Waiting for a human meant collecting nothing at all from unattended
	// runs — the corpus was empty after twelve of them. A human verdict
	// arriving later carries more weight and refines these rather than
	// competing with them.
	//
	// Selection is evidence-bound, not flag-bound: only the final write of
	// a path whose on-disk bytes a green verification actually covered.
	// The session-wide VerifiedThisRun flag stayed true through unverified
	// rewrites and labeled files the passing command never touched.
	if verified := verifiedFinalWrites(ctx); len(verified) > 0 {
		if n := recordVerifiedPass(modelName, verified); n > 0 {
			log.Printf("[lens] recorded %d verified-run sample(s) for %s (of %d writes)",
				n, modelName, len(ctx.PassWrites))
		}
	}

	// Send final done event
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// ---------------------------------------------------------------------------
// /cancel — abort an in-flight /v1/agent turn by session_id
// ---------------------------------------------------------------------------

// handleCancel POSTs cancel an in-flight agent turn. Body:
//
//	{"session_id": "..."}
//
// Returns 200 with `{"cancelled": true}` if the session was found and
// cancelled, 404 with `{"cancelled": false}` if no such session is
// active. Idempotent: a second cancel for the same session returns 404.
//
// On success, the agent loop exits via context.Canceled, the SSE
// stream emits its trailing `[DONE]`, and the client connection
// closes cleanly. The TUI surfaces a "turn cancelled" system message.
func handleCancel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, ErrUnsupported, "method not allowed")
		return
	}
	var req struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "invalid request body")
		return
	}
	if req.SessionID == "" {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "session_id required")
		return
	}
	v, ok := activeSessions.LoadAndDelete(req.SessionID)
	w.Header().Set("Content-Type", "application/json")
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]bool{"cancelled": false})
		return
	}
	entry, ok := v.(*sessionCancel)
	if !ok {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "bad session entry"})
		return
	}
	entry.cancel()
	log.Printf("[agent] cancelled session %q via /cancel", req.SessionID)
	_ = json.NewEncoder(w).Encode(map[string]bool{"cancelled": true})
}

// classifyParseFailure walks the raw response shape once and returns
// both a short stable category for the docker log (so `docker logs
// atlas-proxy` reads "what kind of broken" at a glance) and a targeted
// feedback message for the model. The model can't see why parsing
// failed, so a generic "respond in JSON" message lets it loop forever
// on the same pattern:
//
//   - starts with `{"type":"tool_call",...,"name":"<edit_file|write_file>",...}` and looks
//     truncated → it tried a too-big edit; tell it to shrink old_str/new_str
//   - non-JSON prose → standard "respond JSON only" reminder
//   - empty or whitespace → continuation nudge
//
// Categories:
//
//	empty           — response was whitespace
//	prose           — response is non-JSON text (model narration leaking)
//	truncated_tool  — JSON tool_call envelope cut off mid-args (max_tokens)
//	html_entities   — tool_call contains &lt; / &gt; / &amp; in string args
//	malformed_tool  — tool_call envelope present but JSON malformed
//	non_json        — response begins with text other than '{'
//
// The bug this addresses: in May 2026 a user fix-intent prompt put the
// model in a loop emitting the same 1100-char edit_file with all 5
// flask routes embedded in old_str. Llama-server's response cap cut it
// mid-string, parse failed, we didn't tell the model why, it retried
// identically. classifyParseFailure breaks the cycle by naming the
// failure mode.
// classifyParseFailure names the shape of an unparseable response and returns
// the corrective to send back.
//
// `streamCut` is why the PROXY ended the generation, or "" when the model
// stopped on its own. It comes first because it is the only fact here that is
// known rather than inferred: everything below reads the wreckage and guesses.
// Observed across four sessions — the model began reproducing a 2000-line data
// fixture, degenerated into repeating one line ~50 times, the loop detector
// cut the stream at 601 chars mid-JSON, and the classifier reported
// "truncated_tool: your response hit the token cap, make the call smaller".
// That is the wrong diagnosis and the wrong instruction, so the model retried
// the same thing until the run died.
func classifyParseFailure(raw, streamCut string) (category, feedback string) {
	switch streamCut {
	case "content_loop":
		return "loop_cut", "Your response was cut off because it had started repeating " +
			"itself — the same line over and over — so what arrived was an unfinished " +
			"tool call. The response was NOT too long for the token cap, and re-sending " +
			"a smaller version of the same call will not help.\n\nThis happens when you " +
			"try to reproduce a large block of data you already have. You do not need to " +
			"copy a file's contents to work with it: read_file already showed you the " +
			"file, and input or fixture data should be read at runtime by the code you " +
			"write, never retyped into a tool call. Write the CODE that processes the " +
			"data, not the data."
	case "reasoning_budget":
		return "reasoning_cut", "Your response was cut off: it spent the whole per-turn " +
			"budget on reasoning without emitting a tool call. Skip the deliberation and " +
			"respond with the single JSON action you want to take next."
	}
	stripped := strings.TrimSpace(raw)
	if stripped == "" {
		return "empty", "Your response was empty. Respond with ONLY a single JSON object — {\"type\":\"tool_call\",...} or {\"type\":\"text\",\"content\":\"...\"} or {\"type\":\"done\",\"summary\":\"...\"}."
	}
	// HTML-entity encoding detection — some models encode <, >, &
	// inside tool-call string args (`&lt;!DOCTYPE...&gt;`) instead of
	// emitting them literally. JSON parses fine if the whole envelope
	// arrives, but those entities then appear verbatim in old_str and
	// don't match the actual file content. Catch and redirect — works
	// regardless of whether the response has a prose prefix (May 8
	// dashboard.html session: model emitted "Now I can see..." then a
	// JSON tool_call with HTML-entity-encoded old_str; the
	// looksLikeToolCall check below missed it because the response
	// didn't start with `{`, leaving the targeted corrective unfired).
	// Checked FIRST — the entity bug is a stronger signal than "this
	// is narration," so it wins over prose/non_json/malformed.
	htmlEntities := strings.Contains(stripped, "&lt;") ||
		strings.Contains(stripped, "&gt;") ||
		strings.Contains(stripped, "&amp;")
	embeddedToolCall := strings.Contains(stripped, `"type":"tool_call"`) ||
		strings.Contains(stripped, `"type": "tool_call"`)
	if htmlEntities && embeddedToolCall {
		return "html_entities", "Your tool call has HTML-entity-encoded angle brackets (`&lt;` / `&gt;` / `&amp;`) inside the JSON string args. JSON strings should contain literal `<` and `>` — don't HTML-escape them. The file content goes verbatim onto disk; entities like `&lt;!DOCTYPE&gt;` would write the literal text `&lt;!DOCTYPE&gt;` into the file, not `<!DOCTYPE>`. Re-emit with literal angle brackets. For HTML rewrites, structural_edit is also a good alternative — it takes `selector: \"<body>\"` and the content body, no old_str needed. Also: respond with ONLY the JSON object — no prose preamble."
	}
	// Truncated tool_call detection: response starts with the tool-call
	// preamble but doesn't have a properly closed args object. We look
	// for the opening shape and the absence of a clean trailing `}}` —
	// if both, treat it as truncation.
	looksLikeToolCall := strings.HasPrefix(stripped, `{"type":"tool_call"`) ||
		strings.HasPrefix(stripped, `{ "type": "tool_call"`) ||
		strings.HasPrefix(stripped, `{"type": "tool_call"`)
	if !looksLikeToolCall {
		// Could be prose narration (model thinking leaked into content)
		// or some other non-tool_call shape.
		feedback := "Your response was not valid JSON. Respond with ONLY a JSON object, no other text. Example: {\"type\":\"tool_call\",\"name\":\"write_file\",\"args\":{\"path\":\"file.py\",\"content\":\"code\"}}"
		if !strings.HasPrefix(stripped, "{") {
			return "prose", feedback
		}
		return "non_json", feedback
	}
	// Crude truncation heuristic — if the response doesn't end with
	// at least one closing brace it's almost certainly cut off
	// mid-args. (A complete tool_call ends `...}}`.)
	truncated := !strings.HasSuffix(stripped, "}}") &&
		!strings.HasSuffix(stripped, "}") &&
		!strings.HasSuffix(stripped, "]")
	if truncated {
		hasEditOrWrite := strings.Contains(stripped, `"edit_file"`) ||
			strings.Contains(stripped, `"write_file"`)
		if hasEditOrWrite {
			// GH #39: when truncation hits on a whole-file replacement,
			// structural_edit is the right tool — it takes a structural
			// selector (function:NAME, <tag>) instead of literal
			// old_str, so the JSON envelope stays small. Steer the
			// model toward it explicitly.
			structuralHint := ""
			if strings.Contains(stripped, `&lt;`) || strings.Contains(stripped, `&gt;`) ||
				strings.Contains(stripped, `<body>`) || strings.Contains(stripped, `<head>`) ||
				strings.Contains(stripped, `def `) || strings.Contains(stripped, `class `) {
				structuralHint = " For whole-function or whole-element replacements, use `structural_edit` instead — it takes a selector (e.g. `function:dashboard`, `<body>`) and drops `old_str` entirely, so it doesn't truncate."
			}
			return "truncated_tool", "Your last tool call was TRUNCATED — the response hit the token cap mid-args. The fix is to shrink old_str/new_str: edit ONE function or block per call, not the whole file. If you need to change multiple routes/functions, do them in separate edit_file calls (one per turn). Common offenders: pasting all of app.py into old_str, embedding 5+ @app.route handlers in a single replacement." + structuralHint + " Respond now with a smaller edit_file or a structural_edit call."
		}
		return "truncated_tool", "Your tool call was truncated mid-args. Make a smaller call — keep `content`, `old_str`, and `new_str` short (under ~30 lines). Respond now with the corrected, smaller call."
	}
	return "malformed_tool", "Your tool_call JSON was malformed. Re-emit it as a single valid JSON object: {\"type\":\"tool_call\",\"name\":\"<tool>\",\"args\":{...}}. No prose, no markdown fences, no trailing commas."
}

// extractModelResponse extracts a ModelResponse from the LLM output,
// handling cases where the model adds text before/after the JSON or
// where the JSON is truncated.
// rawResponseForFence renders the tool call as the assistant turn the
// fenced-content sub-call replays. Marshalling the parsed struct rather than
// reusing raw model text keeps the sub-call deterministic.
func rawResponseForFence(parsed ModelResponse) string {
	b, err := json.Marshal(parsed)
	if err != nil {
		return "{\"type\":\"tool_call\",\"name\":\"write_file\"}"
	}
	return string(b)
}

// rawEmissionSentinel, passed as the grammar argument, sends the request with
// neither GBNF nor response_format: the model replies in free text.
const rawEmissionSentinel = "__raw_text__"

// rawFenceGraceChars is how much prose the raw-emission sub-call may emit
// before a missing fence opener is treated as "not coming". Generous enough
// for a short preamble the model sometimes writes ahead of the block, far
// short of the multi-minute run to max_tokens it replaces.
const rawFenceGraceChars = 800

// fencedContentRe grabs the first fenced code block of the sub-call reply.
// The tag charset includes +, #, . and - so c++, c#, objective-c and
// asp.net-style tags open a block instead of failing the match entirely.
//
// The run after the tag is `[ \t]*\r?\n`, NOT `\s*\n`: \s matches newlines,
// so a greedy \s* swallowed the file's own leading blank lines along with
// the fence line's terminator and silently dropped them from what landed on
// disk (fuzzed). Only the remainder of the fence line belongs to the fence.
var fencedContentRe = regexp.MustCompile("(?s)```(?:[a-zA-Z0-9+#._-]+)?[ \\t]*\\r?\\n(.*?)```")

// fencedContentTrailingRe is the greedy variant, anchored to a closing fence
// at the very end of the reply. Preferred over fencedContentRe when it
// matches: the sub-call asks for ONE block holding the whole file, so a
// file that itself contains ``` (markdown, a docstring with an example)
// must not be cut at its first interior fence.
var fencedContentTrailingRe = regexp.MustCompile("(?s)```(?:[a-zA-Z0-9+#._-]+)?[ \\t]*\\r?\\n(.*)\\r?\\n```[ \\t\\r\\n]*$")

// extractFencedContent pulls the file body out of a fenced sub-call reply:
// the whole-reply greedy form first, the first-block form as fallback.
func extractFencedContent(reply string) string {
	if m := fencedContentTrailingRe.FindStringSubmatch(reply); m != nil && strings.TrimSpace(m[1]) != "" {
		return m[1] + "\n"
	}
	if m := fencedContentRe.FindStringSubmatch(reply); m != nil && strings.TrimSpace(m[1]) != "" {
		return m[1]
	}
	return ""
}

// fenceTagForPath picks the fence language tag the sub-call prompt asks
// for, from the target file's extension. The prompt hardcoded ```python
// whatever the file was — a .html or .go target got asked for a python
// block (audit finding). Empty means "use a bare fence".
func fenceTagForPath(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".py":
		return "python"
	case ".js", ".mjs":
		return "javascript"
	case ".ts":
		return "typescript"
	case ".html", ".htm":
		return "html"
	case ".css":
		return "css"
	case ".go":
		return "go"
	case ".rs":
		return "rust"
	case ".c", ".h":
		return "c"
	case ".cpp", ".cc", ".hpp":
		return "c++"
	case ".java":
		return "java"
	case ".rb":
		return "ruby"
	case ".sh":
		return "bash"
	case ".json":
		return "json"
	case ".md":
		return "markdown"
	case ".sql":
		return "sql"
	case ".yaml", ".yml":
		return "yaml"
	default:
		return ""
	}
}

// Fenced-fetch progress bounds. The sub-call asks for ONE fenced block and
// nothing else, so the only useful output is CONTENT: a model that streams
// reasoning to max_tokens and never opens a fence is the zero-byte failure,
// and reasoning must not look like progress.
//
// Measured on the seed-20260901 run: 9 of 17 zero-byte failures ran 175-311s
// each, and two attempts can consume ~10 minutes. These bounds are NOT
// derived from whole-fetch duration -- a successful fetch legitimately ran
// 217s while producing content the whole way, and cutting on total elapsed
// would have killed it.
//
// They are conservative defaults over the quantities that DO discriminate:
// observed time-to-first-token is p50 378ms, p90 2.9s, p99 4.4s, max 11.8s
// across 540 calls, so 60s to first CONTENT is ~5x the worst observed start
// with headroom for a reasoning preamble; and at the ~25 tok/s decode rate
// this deployment sustains, 30s of complete silence mid-file is ~750 tokens
// of nothing, which is not generation in progress. Both are env-overridable
// on the existing envOr pattern and are pending real-model canary validation
// before Phase 2 is declared complete.
const (
	defaultFencedFirstContentSec = 60
	defaultFencedIdleSec         = 30
)

func fencedFirstContentTimeout() time.Duration {
	return envDurationSec("ATLAS_FENCED_FIRST_CONTENT_SEC", defaultFencedFirstContentSec)
}

func fencedIdleTimeout() time.Duration {
	return envDurationSec("ATLAS_FENCED_IDLE_SEC", defaultFencedIdleSec)
}

// maxFencedOverrideSec caps the override. A bound of hours is indistinguishable
// from no bound, and the failure this exists to stop is a ~300s stall, so an
// absurd value must fall back rather than quietly disable the safety net.
const maxFencedOverrideSec = 600

func envDurationSec(key string, def int) time.Duration {
	if v := strings.TrimSpace(envOr(key, "")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= maxFencedOverrideSec {
			return time.Duration(n) * time.Second
		}
		log.Printf("[agent] ignoring %s=%q (want 1..%d seconds) — using %ds",
			key, v, maxFencedOverrideSec, def)
	}
	return time.Duration(def) * time.Second
}

// fetchFencedContent asks the model for a file's contents in its native
// channel: one unconstrained call whose reply is a single fenced block.
//
// Code emitted INSIDE a JSON string pays escaping pressure on every dense
// line, and the served model measurably cannot sustain it: the same debounce
// solution parses 6/6 emitted fenced and 0/6 emitted as a JSON string. The
// slip catalogue of three benchmark arms — a missing close-paren on an
// append, a literal \n fusing a statement into a comment, list joins losing
// their spaces — is this one channel problem. The envelope stays under the
// JSON constraint; only the file body moves to plain text.
//
// The sub-call is ephemeral: nothing is appended to ctx.Messages, so from
// the main conversation's view the model wrote "@fenced" and the write
// simply happened.
// maxFencedFailuresPerPath is the whole-session allowance: one attempt plus
// one constrained retry. Past that the model must change mutation strategy —
// inlining the body in write_file, or editing instead of rewriting — because
// a third resolution has never been observed to succeed where two failed and
// each one costs a full generation.
const maxFencedFailuresPerPath = 2

// fencedBudgetExhausted reports whether this path has already spent its
// session allowance. Checked BEFORE any generation starts.
func fencedBudgetExhausted(ctx *AgentContext, path string) bool {
	return ctx != nil && ctx.FencedFailures[fencedKey(ctx, path)] >= maxFencedFailuresPerPath
}

// fencedKey canonicalises the target so equivalent spellings share one
// allowance. Keying on raw model input let "solve.py" and "./solve.py" hold
// separate budgets, which is the same restart-the-counter hole one level
// down.
func fencedKey(ctx *AgentContext, path string) string {
	if ctx == nil {
		return filepath.Clean(path)
	}
	return filepath.Clean(resolveAgentPath(ctx, path))
}

// charge records one consumed attempt against the path's session allowance.
func charge(ctx *AgentContext, path string) {
	if ctx == nil {
		return
	}
	if ctx.FencedFailures == nil {
		ctx.FencedFailures = map[string]int{}
	}
	ctx.FencedFailures[fencedKey(ctx, path)]++
}

// fencedReserve is the time held back for validating the write and sending an
// honest terminal once a fetch returns.
const fencedReserve = 20 * time.Second

// fencedFitsRemainingBudget answers requirement (a): honour a deadline when
// the context carries one, and say so plainly when it does not. Production
// today builds ctx.Ctx with context.WithCancel and NO deadline, so this
// returns true and the fetch is bounded by the progress watchdog alone —
// session-budget reservation is unavailable in that configuration.
func fencedFitsRemainingBudget(ctx *AgentContext) bool {
	if ctx == nil || ctx.Ctx == nil {
		return true
	}
	deadline, ok := ctx.Ctx.Deadline()
	if !ok {
		return true // no session budget exists to reserve from
	}
	need := fencedFirstContentTimeout() + fencedReserve
	return time.Until(deadline) >= need
}

func fetchFencedContent(ctx *AgentContext, rawCall, path string) (string, error) {
	if fencedBudgetExhausted(ctx, path) {
		return "", fmt.Errorf("fenced resolution for %s has already failed %d times "+
			"this session; send the file inline or edit it instead", path,
			ctx.FencedFailures[fencedKey(ctx, path)])
	}
	if !fencedFitsRemainingBudget(ctx) {
		return "", fmt.Errorf("not enough session budget left to resolve %s "+
			"and still validate the result", path)
	}
	tag := fenceTagForPath(path)
	note := AgentMessage{Role: "user", Content: fmt.Sprintf(
		"[system note]: Now provide ONLY the complete contents of %s, as plain "+
			"code in a single fenced block (```%s ... ```). No JSON, no "+
			"commentary, no partial file.", path, tag)}
	msgs := append(append([]AgentMessage{}, ctx.Messages...),
		AgentMessage{Role: "assistant", Content: rawCall}, note)
	// Attempts remaining are what the SESSION still allows for this path, not
	// a fresh two. Requirement: a new write_file call must not restore the
	// allowance a previous turn spent.
	var lastErr error
	remaining := maxFencedFailuresPerPath - ctx.FencedFailures[fencedKey(ctx, path)]
	for attempt := 0; attempt < remaining; attempt++ {
		if attempt > 0 && !fencedFitsRemainingBudget(ctx) {
			break // a retry that cannot finish and still be validated
		}
		attemptStart := time.Now()
		reply, tokens, err := callLLMOnceWithGrammar(ctx, msgs, 0.2, rawEmissionSentinel)
		elapsed := time.Since(attemptStart)
		// Every attempt is a real generation and is accounted whether or
		// not it yielded a usable block — an unaccounted sub-call made the
		// run totals lie by one generation per written file.
		ctx.TotalTokens += tokens
		ctx.FencedCalls++
		ctx.FencedTokens += tokens
		content := extractFencedContent(reply)
		// The model sometimes wraps the SENTINEL in the fence instead of the
		// file — measured: "```python\n@fenced\n```". That extracts as
		// non-empty and would land a file whose entire contents are the word
		// @fenced, so it counts as no block and the attempt is retried.
		if strings.TrimSpace(content) == rawEmissionSentinel ||
			strings.TrimSpace(content) == "@fenced" {
			log.Printf("[agent] fenced reply for %s contained only the sentinel — treating as no block", path)
			content = ""
		}
		got := content != ""
		contentBytes := len(content)
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtStageEnd,
			Stage:      "fenced_fetch",
			DurationMS: elapsed.Milliseconds(),
			Payload: map[string]interface{}{
				"success":          err == nil && got,
				"path":             path,
				"attempt":          attempt + 1,
				"generated_tokens": tokens,
				"content_bytes":    contentBytes,
				"total_tokens":     ctx.TotalTokens,
			},
		})
		if err != nil {
			// Every way this attempt can end WITHOUT a fenced block charges
			// the session: watchdog cancellation, transport error, HTTP
			// failure. Not charging here is how the black-box loop issued one
			// unbounded fetch per turn — the allowance was only ever spent by
			// a clean empty reply.
			charge(ctx, path)
			// The SESSION being cancelled ends everything; the watchdog
			// cancelling its own child is a recoverable zero-byte failure and
			// the one permitted retry may still follow, bounded by the
			// allowance the loop already counted.
			if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
				return "", err
			}
			lastErr = err
			continue
		}
		if got {
			// A successful resolution clears the consecutive-failure state:
			// the path is healthy again and a later stall gets its own budget.
			if ctx.FencedFailures != nil {
				delete(ctx.FencedFailures, fencedKey(ctx, path))
			}
			return content, nil
		}
		// Charge the session, not the local loop, so the allowance survives
		// into the next write_file call for this path.
		charge(ctx, path)
		// What the model sent instead is the whole diagnosis, and without it
		// "no fenced block after 2 attempts" says only that something went
		// wrong. Measured a 56% failure rate on this fetch with no way to see
		// why: the same request reproduced in a short context returns a clean
		// block every time, so the cause lives in the session context and
		// cannot be found without the reply.
		log.Printf("[agent] fenced attempt %d for %s produced no block (%d chars, cut=%q, session failures %d/%d): %q",
			attempt+1, path, len(reply), ctx.LastStreamCut,
			ctx.FencedFailures[fencedKey(ctx, path)], maxFencedFailuresPerPath,
			truncateStr(reply, 400))
		msgs = append(msgs, AgentMessage{Role: "assistant", Content: reply},
			AgentMessage{Role: "user", Content: fmt.Sprintf(
				"[system note]: That had no fenced block. Reply with ONE ```%s fenced block containing the complete file, nothing else.", tag)})
	}
	if lastErr != nil {
		return "", fmt.Errorf("fenced resolution for %s was cut after %d attempt(s) "+
			"this session (%v); send the file inline or edit it instead",
			path, ctx.FencedFailures[fencedKey(ctx, path)], lastErr)
	}
	return "", fmt.Errorf("no fenced block after %d attempt(s) for %s this session; "+
		"send the file inline or edit it instead",
		ctx.FencedFailures[fencedKey(ctx, path)], path)
}

func extractModelResponse(raw string) (ModelResponse, error) {
	raw = strings.TrimSpace(raw)

	// Try direct parse first. Capture the error so we can surface it
	// to the caller's log if every other path fails — without this,
	// real diagnostics ("invalid character '\\n' in string literal",
	// "unexpected end of JSON input") were silently swallowed and the
	// agent loop just got "could not parse JSON" with no clue why.
	var resp ModelResponse
	directErr := json.Unmarshal([]byte(raw), &resp)
	if directErr == nil {
		liftMissingArgs(&resp, raw)
		return resp, nil
	}

	// Find the first '{' and try to parse from there
	start := strings.Index(raw, "{")
	if start < 0 {
		return resp, fmt.Errorf("no JSON object found in response")
	}

	// Find matching closing brace by counting nesting
	depth := 0
	inString := false
	escaped := false
	end := -1
	for i := start; i < len(raw); i++ {
		c := raw[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				end = i + 1
				break
			}
		}
	}

	var balancedErr error
	if end > start {
		jsonStr := raw[start:end]
		balancedErr = json.Unmarshal([]byte(jsonStr), &resp)
		if balancedErr == nil {
			liftMissingArgs(&resp, jsonStr)
			return resp, nil
		}
	}

	// JSON was truncated (max_tokens hit mid-content) or otherwise
	// malformed — try a generalized tool_call recovery for write_file,
	// edit_file, and structural_edit. Identical shape (path + payload field),
	// just different field names. If recovery succeeds, return it; if
	// not, fall through to the diagnostic error below.
	if recovered, ok := recoverTruncatedToolCall(raw[start:]); ok {
		return recovered, nil
	}

	// Surface the most informative error available. directErr fires
	// when the response had garbage outside the JSON envelope (prose
	// preamble) — usually less useful. balancedErr fires when the
	// brace-balanced substring still failed to Unmarshal — that's the
	// actual JSON-content bug, e.g. literal LF inside a string,
	// unescaped backslash, malformed escape sequence. Prefer it.
	if balancedErr != nil {
		return resp, fmt.Errorf("could not parse JSON from response: %w", balancedErr)
	}
	return resp, fmt.Errorf("could not parse JSON from response: %w", directErr)
}

// liftMissingArgs handles models that emit tool calls in shapes other than
// the prescribed {"type":"tool_call","name":"X","args":{...}} envelope.
//
// Common alternative shapes:
//   - OpenAI-style: {"type":"tool_call","name":"X","arguments":{...}}
//   - Anthropic-style: {"type":"tool_call","name":"X","parameters":{...}}
//   - Inlined: {"type":"tool_call","name":"X","path":"...","offset":0,...}
//   - Type-is-tool-name: {"type":"read_file","path":"..."} — model
//     put the tool name in the type field instead of using "tool_call".
//
// When `args` is missing on a tool_call, re-decode the raw JSON into a
// generic map and either pull `arguments`/`parameters` over to args, or
// lift every non-envelope top-level field into a synthetic args object.
// This is purely a recovery path; the system prompt still teaches the
// canonical shape.
func liftMissingArgs(resp *ModelResponse, raw string) {
	// If Type is a known tool name, treat it as a tool_call with
	// that tool. The model emitted {"type":"read_file","path":"..."}
	// instead of {"type":"tool_call","name":"read_file","args":{...}}.
	// Without this fix the agent loop's switch hits the `default` arm
	// and burns a turn telling the model "Unknown response type".
	if resp.Type != "" && resp.Type != "tool_call" && resp.Type != "text" && resp.Type != "done" {
		if getTool(resp.Type) != nil {
			resp.Name = resp.Type
			resp.Type = "tool_call"
		}
	}

	if resp.Type != "tool_call" || resp.Name == "" {
		return
	}
	if len(resp.Args) > 0 && string(resp.Args) != "null" {
		return
	}

	var top map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &top); err != nil {
		return
	}

	// Prefer explicit alt-key wrappers when present.
	for _, key := range []string{"arguments", "parameters", "params", "input"} {
		if v, ok := top[key]; ok && len(v) > 0 && string(v) != "null" {
			resp.Args = v
			return
		}
	}

	// Otherwise lift every non-envelope key into a synthetic args object.
	envelope := map[string]struct{}{
		"type": {}, "name": {}, "content": {}, "summary": {}, "args": {},
	}
	lifted := make(map[string]json.RawMessage)
	for k, v := range top {
		if _, isEnvelope := envelope[k]; isEnvelope {
			continue
		}
		lifted[k] = v
	}
	if len(lifted) == 0 {
		return
	}
	if buf, err := json.Marshal(lifted); err == nil {
		resp.Args = buf
	}
}

// recoverTruncatedToolCall is the generalized counterpart to
// recoverTruncatedWriteFile. May 9 2026: under BiasBusters mitigations
// the model now reaches for structural_edit and edit_file too, and either can
// land malformed JSON (truncated content, stray escape) the same way
// write_file used to. Old code only recovered write_file; everything
// else just died with "could not parse JSON". Now we sniff the tool
// name from the partial bytes and dispatch to a tool-specific recovery
// when one exists. Returns (response, true) on successful recovery,
// (zero, false) when no recovery is available so the caller falls
// through to the diagnostic error.
func recoverTruncatedToolCall(partial string) (ModelResponse, bool) {
	switch {
	case strings.Contains(partial, `"name":"write_file"`) || strings.Contains(partial, `"name": "write_file"`):
		if r, err := recoverTruncatedWriteFile(partial); err == nil {
			return r, true
		}
	case strings.Contains(partial, `"name":"structural_edit"`) || strings.Contains(partial, `"name": "structural_edit"`):
		if r, err := recoverTruncatedStructuralEdit(partial); err == nil {
			return r, true
		}
	case strings.Contains(partial, `"name":"edit_file"`) || strings.Contains(partial, `"name": "edit_file"`):
		if r, err := recoverTruncatedEditFile(partial); err == nil {
			return r, true
		}
	}
	return ModelResponse{}, false
}

// looksDegenerate reports whether a recovered field value is the model's
// own degenerate output rather than real content.
//
// Truncation recovery exists for one case: a well-formed tool call whose
// JSON was cut off by max_tokens. It reconstructs args from whatever
// extractStringField can read, which is a purely structural operation — a
// run of repeated newlines parses exactly as well as a real function body.
// Without this check, a generation that degenerated into a repeating tail
// (the same condition isLoopingTail cuts the stream on) is "successfully
// recovered" into an edit_file or write_file call and executed against the
// user's file. The stream cut prevents the tokens from being generated; it
// does nothing about the bytes already buffered when recovery runs.
//
// Two shapes, both observed: a value that is almost entirely whitespace,
// and one whose tail repeats. Short values are exempt — a legitimately
// small new_str has no room to look degenerate, and the length floor keeps
// ordinary edits out of the check entirely.
func looksDegenerate(s string) bool {
	const minJudgeable = 64
	if len(s) < minJudgeable {
		return false
	}
	var ws int
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\n', '\r':
			ws++
		}
	}
	if float64(ws)/float64(len(s)) > 0.9 {
		return true
	}
	// Repetition alone is not degeneracy — real files repeat boilerplate,
	// and isLoopingTail's "tail occurs 3+ times" fires on a long file with
	// a handful of similar lines. Rejecting those would break recovery for
	// exactly the truncated writes it exists to salvage. Require instead
	// that the repeated tail account for most of the value, which
	// separates a repeating generation from a file that happens to repeat.
	const probe = 48
	if len(s) < probe*3 {
		return false
	}
	tail := s[len(s)-probe:]
	if strings.TrimSpace(tail) == "" {
		return false
	}
	occurrences := strings.Count(s, tail)
	return float64(occurrences*probe)/float64(len(s)) > 0.5
}

// extractStringField pulls a JSON-string field value out of a partial
// (possibly truncated) tool-call payload. Returns the unescaped value
// and true on success. The end is determined by the next unescaped `"`
// — for the trailing field of a truncated payload, the value runs to
// end-of-input and is closed by the caller.
func extractStringField(partial, field string) (string, bool) {
	for _, marker := range []string{`"` + field + `":"`, `"` + field + `": "`} {
		idx := strings.Index(partial, marker)
		if idx < 0 {
			continue
		}
		valueStart := idx + len(marker)
		// Walk until unescaped closing quote.
		escaped := false
		for i := valueStart; i < len(partial); i++ {
			c := partial[i]
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				raw := partial[valueStart:i]
				var unescaped string
				if err := json.Unmarshal([]byte(`"`+raw+`"`), &unescaped); err == nil {
					return unescaped, true
				}
				return raw, true
			}
		}
		// Hit end-of-input without finding closing quote — payload was
		// truncated mid-string. Return what we have, best-effort
		// unescaping; trailing backslash is dropped to avoid invalid
		// escape sequences.
		raw := strings.TrimRight(partial[valueStart:], "\\")
		var unescaped string
		if err := json.Unmarshal([]byte(`"`+raw+`"`), &unescaped); err == nil {
			return unescaped, true
		}
		// Manual fallback for the common escapes when Unmarshal rejected
		// a partial string (rarely happens but cheap insurance).
		manual := strings.ReplaceAll(raw, `\n`, "\n")
		manual = strings.ReplaceAll(manual, `\t`, "\t")
		manual = strings.ReplaceAll(manual, `\"`, `"`)
		manual = strings.ReplaceAll(manual, `\\`, `\`)
		return manual, true
	}
	return "", false
}

// recoverTruncatedStructuralEdit recovers a structural_edit tool call whose JSON
// envelope didn't survive the parser. structural_edit's args are
// {path, selector, content} — same shape as write_file but with an
// additional selector field that's always short (function:NAME,
// class:NAME, <tag>) so it lands intact even on truncation. The
// content is the long field that gets cut.
func recoverTruncatedStructuralEdit(partial string) (ModelResponse, error) {
	path, ok := extractStringField(partial, "path")
	if !ok || path == "" {
		return ModelResponse{}, fmt.Errorf("structural_edit recovery: missing path")
	}
	selector, ok := extractStringField(partial, "selector")
	if !ok || selector == "" {
		return ModelResponse{}, fmt.Errorf("structural_edit recovery: missing selector")
	}
	content, ok := extractStringField(partial, "content")
	if !ok {
		return ModelResponse{}, fmt.Errorf("structural_edit recovery: missing content")
	}
	if looksDegenerate(content) {
		return ModelResponse{}, fmt.Errorf("structural_edit recovery: content is degenerate output, not a real edit")
	}
	args, _ := json.Marshal(StructuralEditInput{Path: path, Selector: selector, Content: content})
	log.Printf("[agent] recovered truncated structural_edit: path=%s selector=%q content=%d chars",
		path, selector, len(content))
	return ModelResponse{Type: "tool_call", Name: "structural_edit", Args: args}, nil
}

// recoverTruncatedEditFile recovers an edit_file tool call. Args are
// {path, old_str, new_str, replace_all?}. Either old_str or new_str
// can be the truncation point; recover whichever one terminated
// cleanly and warn-log when one didn't, so the agent loop sees the
// failure category instead of a generic parse error.
func recoverTruncatedEditFile(partial string) (ModelResponse, error) {
	path, ok := extractStringField(partial, "path")
	if !ok || path == "" {
		return ModelResponse{}, fmt.Errorf("edit_file recovery: missing path")
	}
	oldStr, oldOK := extractStringField(partial, "old_str")
	newStr, newOK := extractStringField(partial, "new_str")
	if !oldOK && !newOK {
		return ModelResponse{}, fmt.Errorf("edit_file recovery: missing both old_str and new_str")
	}
	if looksDegenerate(oldStr) || looksDegenerate(newStr) {
		return ModelResponse{}, fmt.Errorf("edit_file recovery: old_str/new_str is degenerate output, not a real edit")
	}
	replaceAll := strings.Contains(partial, `"replace_all":true`) ||
		strings.Contains(partial, `"replace_all": true`)
	args, _ := json.Marshal(EditFileInput{
		Path:       path,
		OldStr:     oldStr,
		NewStr:     newStr,
		ReplaceAll: replaceAll,
	})
	log.Printf("[agent] recovered truncated edit_file: path=%s old_str=%dch new_str=%dch", path, len(oldStr), len(newStr))
	return ModelResponse{Type: "tool_call", Name: "edit_file", Args: args}, nil
}

// recoverTruncatedWriteFile attempts to recover a write_file tool call
// where the content was truncated by max_tokens.
func recoverTruncatedWriteFile(partial string) (ModelResponse, error) {
	// The pattern is: {"type":"tool_call","name":"write_file","args":{"path":"...","content":"...
	// We need to close the content string and the JSON objects

	// Find the content field, remembering WHICH spelling matched so the
	// value's offset is known exactly. Re-deriving it afterwards by probing
	// a fixed 15-byte window read past the end of any buffer whose content
	// marker sat within 15 bytes of the end — a panic, and net/http answers
	// a panicking handler by closing the connection, so the session died
	// mid-stream with no `done` event. Truncation puts the marker near the
	// end by definition, which is exactly when this function runs.
	marker := `"content":"`
	idx := strings.Index(partial, marker)
	if idx < 0 {
		marker = `"content": "`
		idx = strings.Index(partial, marker)
	}
	if idx < 0 {
		return ModelResponse{}, fmt.Errorf("cannot find content field in truncated write_file")
	}

	// Find the "path" value
	pathIdx := strings.Index(partial, `"path":"`)
	pathEnd := -1
	path := ""
	if pathIdx >= 0 {
		pathStart := pathIdx + len(`"path":"`)
		pathEnd = strings.Index(partial[pathStart:], `"`)
		if pathEnd >= 0 {
			path = partial[pathStart : pathStart+pathEnd]
		}
	}

	// Extract content: everything after the marker until the end.
	content := partial[idx+len(marker):]

	// Unescape the content string (it's JSON-escaped)
	// Remove trailing incomplete escape sequences
	content = strings.TrimRight(content, "\\")
	// Close the string
	content = strings.TrimSuffix(content, `"`)
	content = strings.TrimSuffix(content, `"}`)
	content = strings.TrimSuffix(content, `"}}`)

	// Unescape JSON string escapes
	var unescaped string
	err := json.Unmarshal([]byte(`"`+content+`"`), &unescaped)
	if err != nil {
		// Fallback: manual unescape of common sequences
		unescaped = strings.ReplaceAll(content, `\n`, "\n")
		unescaped = strings.ReplaceAll(unescaped, `\t`, "\t")
		unescaped = strings.ReplaceAll(unescaped, `\"`, "\"")
		unescaped = strings.ReplaceAll(unescaped, `\\`, "\\")
	}

	if path == "" {
		return ModelResponse{}, fmt.Errorf("could not extract path from truncated write_file")
	}
	if looksDegenerate(unescaped) {
		return ModelResponse{}, fmt.Errorf("write_file recovery: content is degenerate output, not a real file")
	}

	// Build the args JSON
	args, _ := json.Marshal(WriteFileInput{Path: path, Content: unescaped})

	log.Printf("[agent] recovered truncated write_file: path=%s content=%d chars", path, len(unescaped))

	return ModelResponse{
		Type: "tool_call",
		Name: "write_file",
		Args: args,
	}, nil
}

// classifyAgentTier decides whether a request is conversational.
//
// The message tier has exactly two behaviours, despite the four-value Tier
// type. TierMaxTurns caps T0 at 5 turns and leaves T1/T2/T3 uncapped
// alike; shouldGeneratePlan tests only Tier0Conversational; and the tier
// travels to v3-service where it is read into a log line and never branched
// on. V3 activation is driven by classifyFileTier, which scores the file
// being edited — a different function that does use T1/T2/T3 meaningfully.
// So the only question here is conversational or not, and the returned
// non-T0 value is Tier2Medium because that is what every consumer treats
// every non-T0 value as.
//
// The costs are asymmetric, which sets the direction of the default.
// Misreading chat as a task wastes one planner call on a message the model
// answers and closes in a single turn. Misreading a task as chat caps it at
// 5 turns and skips planning, and a capped task fails: "the snake is still
// moving way too fast, please slow it down significantly" was classified
// conversational during 2026-07-21 dogfooding and returned a zero-tool-call
// non-answer instead of an edit.
//
// So T0 requires positive evidence, and the absence of a recognized task
// word is not evidence. Describing desired software behaviour is open
// vocabulary with no closed list to match against, while greetings are
// short and questions are a closed grammatical class. Both of those are
// things a message can be shown to BE, rather than shown not to be.
func classifyAgentTier(message string) Tier {
	trimmed := strings.TrimSpace(message)

	// Task language wins outright, at any length and in any shape. "can
	// you fix the login bug?" is a question and a task; the task reading
	// is the one whose failure mode is expensive.
	if isActionIntentMessage(trimmed) || isFixIntentMessage(trimmed) {
		return Tier2Medium
	}

	// An explicit "explain this, do not edit anything" is conversational by
	// definition, whether or not it is phrased as a question. Without this,
	// "Explain how the retry logic works, without editing anything." carries
	// no question mark and no question-word opener, so it fell through to a
	// work tier and got the write pipeline.
	if isExplainOnlyMessage(strings.ToLower(trimmed)) {
		return Tier0Conversational
	}

	// Greeting or acknowledgement. The floor matches shouldGeneratePlan's
	// own, so the two agree on what is too short to plan for.
	if len(trimmed) < 12 {
		return Tier0Conversational
	}

	if isQuestionMessage(trimmed) {
		return Tier0Conversational
	}

	return Tier2Medium
}

// questionStarters is the set of words an English interrogative can open
// with. Unlike task vocabulary, which is unbounded, this is a closed
// grammatical class, which is what makes matching against it sound where
// matching against a list of task verbs would not be.
var questionStarters = []string{
	"why", "what", "when", "where", "who", "which", "how",
	"is ", "are ", "does ", "do ", "did ", "can ", "could ",
	"would ", "should ", "will ", "won't", "isn't", "aren't",
}

// isQuestionMessage reports whether a message is shaped as a question:
// a trailing "?", which catches any phrasing, or one of the interrogative
// openers above for questions written without one.
func isQuestionMessage(message string) bool {
	trimmed := strings.TrimSpace(message)
	// A question mark ANYWHERE, not only at the end. People ask and then
	// qualify — "what does find_duplicates do, and what is its complexity?
	// Just explain." ends in a period, so a suffix-only check read it as
	// not-a-question and it was handed the full write pipeline. Safe to
	// widen: classifyAgentTier checks action and fix intent first, so
	// "fix the bug in foo.py? or bar.py?" still classifies as work.
	if strings.Contains(trimmed, "?") {
		return true
	}
	lower := strings.ToLower(trimmed)
	for _, w := range questionStarters {
		if strings.HasPrefix(lower, w) {
			return true
		}
		// Or opening a clause: "In orders.py, what does X do" carries no
		// question mark at all but is plainly a question.
		if strings.Contains(lower, ", "+w) || strings.Contains(lower, ". "+w) {
			return true
		}
	}
	return false
}

// Toolchain describes one language ecosystem detected in the project.
// The fields are surfaced into the system prompt so the model knows
// which runner to invoke and how to install deps if needed.
//
// Detection is manifest-driven: presence of pyproject.toml means
// Python, package.json means Node, Cargo.toml means Rust, etc. A
// polyglot project (React frontend + Django backend + deploy scripts)
// returns multiple Toolchains so the model can pick the right one
// per file edit.
type Toolchain struct {
	Name           string   // canonical key: "python", "node", "rust", "go", "ruby", "java-maven", "java-gradle", "php", "dotnet", "dart"
	Manifests      []string // manifest files found relative to workingDir (e.g. ["pyproject.toml", "requirements.txt"])
	Runner         string   // command to run the project's main entry (e.g. "/workspace/venv/bin/python", "node", "cargo run", "go run .")
	PackageManager string   // detected pkg manager when ambiguous (npm vs pnpm vs yarn vs bun for Node)
	InstallCommand string   // command to install deps from lockfile (e.g. "npm ci", "pip install -r requirements.txt")
	TestCommand    string   // best-guess test runner ("pytest", "npm test", "cargo test", ...)
}

// detectProjectToolchains scans workingDir for language manifests and
// returns one Toolchain per detected ecosystem. Polyglot projects
// (e.g. React + Django) produce multiple entries. Empty slice means
// no recognized manifest was found at the root.
//
// We deliberately only look ONE level deep at the root — most
// monorepos have manifests in subdirs (apps/web/package.json,
// services/api/pyproject.toml) but probing deeper here would be
// expensive and noisy. The model can still discover deep manifests
// via list_directory / read_file when it needs to.
func detectProjectToolchains(workingDir string) []Toolchain {
	if workingDir == "" {
		return nil
	}
	var out []Toolchain

	// Python — venv-aware so the runner points at the project's
	// pinned interpreter when one exists.
	pyManifests := pickExisting(workingDir, "pyproject.toml", "requirements.txt", "setup.py", "Pipfile", "poetry.lock")
	if len(pyManifests) > 0 || detectProjectVenvPython(workingDir) != "" {
		runner := detectProjectVenvPython(workingDir)
		if runner == "" {
			runner = "python"
		}
		install := "pip install -r requirements.txt"
		if hasFile(workingDir, "poetry.lock") {
			install = "poetry install"
		} else if hasFile(workingDir, "Pipfile.lock") {
			install = "pipenv install"
		} else if hasFile(workingDir, "pyproject.toml") && !hasFile(workingDir, "requirements.txt") {
			install = "pip install -e ."
		}
		out = append(out, Toolchain{
			Name: "python", Manifests: pyManifests,
			Runner: runner, InstallCommand: install,
			TestCommand: "pytest",
		})
	}

	// Node / TypeScript — pkg manager picked from lockfile.
	if hasFile(workingDir, "package.json") {
		pm, install := "npm", "npm install"
		switch {
		case hasFile(workingDir, "pnpm-lock.yaml"):
			pm, install = "pnpm", "pnpm install --frozen-lockfile"
		case hasFile(workingDir, "yarn.lock"):
			pm, install = "yarn", "yarn install --frozen-lockfile"
		case hasFile(workingDir, "bun.lockb"):
			pm, install = "bun", "bun install --frozen-lockfile"
		case hasFile(workingDir, "package-lock.json"):
			pm, install = "npm", "npm ci"
		}
		runner := "node"
		if hasFile(workingDir, "tsconfig.json") {
			runner = "tsx" // ts/jsx-aware launcher; falls back to node for plain .js
		}
		out = append(out, Toolchain{
			Name: "node", Manifests: pickExisting(workingDir, "package.json", "tsconfig.json"),
			Runner: runner, PackageManager: pm, InstallCommand: install,
			TestCommand: pm + " test",
		})
	}

	// Rust
	if hasFile(workingDir, "Cargo.toml") {
		out = append(out, Toolchain{
			Name: "rust", Manifests: pickExisting(workingDir, "Cargo.toml", "Cargo.lock"),
			Runner: "cargo run", InstallCommand: "cargo fetch",
			TestCommand: "cargo test",
		})
	}

	// Go
	if hasFile(workingDir, "go.mod") {
		out = append(out, Toolchain{
			Name: "go", Manifests: pickExisting(workingDir, "go.mod", "go.sum"),
			Runner: "go run .", InstallCommand: "go mod download",
			TestCommand: "go test ./...",
		})
	}

	// Ruby
	if hasFile(workingDir, "Gemfile") {
		out = append(out, Toolchain{
			Name: "ruby", Manifests: pickExisting(workingDir, "Gemfile", "Gemfile.lock"),
			Runner: "bundle exec ruby", InstallCommand: "bundle install",
			TestCommand: "bundle exec rspec",
		})
	}

	// Java — Maven
	if hasFile(workingDir, "pom.xml") {
		out = append(out, Toolchain{
			Name: "java-maven", Manifests: []string{"pom.xml"},
			Runner: "mvn exec:java", InstallCommand: "mvn install -DskipTests",
			TestCommand: "mvn test",
		})
	}

	// Java/Kotlin — Gradle (prefer wrapper if present)
	if hasFile(workingDir, "build.gradle") || hasFile(workingDir, "build.gradle.kts") {
		runner := "gradle run"
		install := "gradle build -x test"
		test := "gradle test"
		if hasFile(workingDir, "gradlew") {
			runner = "./gradlew run"
			install = "./gradlew build -x test"
			test = "./gradlew test"
		}
		out = append(out, Toolchain{
			Name: "java-gradle", Manifests: pickExisting(workingDir, "build.gradle", "build.gradle.kts", "settings.gradle", "gradlew"),
			Runner: runner, InstallCommand: install, TestCommand: test,
		})
	}

	// PHP / Composer
	if hasFile(workingDir, "composer.json") {
		out = append(out, Toolchain{
			Name: "php", Manifests: pickExisting(workingDir, "composer.json", "composer.lock"),
			Runner: "php", InstallCommand: "composer install",
			TestCommand: "vendor/bin/phpunit",
		})
	}

	// .NET — pick the first project file we find
	if csproj := firstMatchingGlob(workingDir, "*.csproj", "*.fsproj", "*.sln"); csproj != "" {
		out = append(out, Toolchain{
			Name: "dotnet", Manifests: []string{csproj},
			Runner: "dotnet run", InstallCommand: "dotnet restore",
			TestCommand: "dotnet test",
		})
	}

	// Dart / Flutter
	if hasFile(workingDir, "pubspec.yaml") {
		runner, install := "dart run", "dart pub get"
		if hasFile(workingDir, ".flutter-plugins") || hasFile(workingDir, "flutter.yaml") {
			runner, install = "flutter run", "flutter pub get"
		}
		out = append(out, Toolchain{
			Name: "dart", Manifests: pickExisting(workingDir, "pubspec.yaml", "pubspec.lock"),
			Runner: runner, InstallCommand: install,
			TestCommand: "dart test",
		})
	}

	return out
}

// probeToolchainReady returns a short status string for a Toolchain
// that's safe to run from buildSystemPrompt — meaning: it MUST be
// purely filesystem-based (no shelling out, no network). The model
// uses this to decide whether to install deps or skip straight to
// verification.
//
// We can't actually invoke `python -c "import flask"` here without
// running a subprocess in the sandbox, which is too expensive for
// every system-prompt build. Instead we look for filesystem evidence
// that deps are installed: venv with site-packages populated,
// node_modules present, target/debug/ for Rust, vendor/ for Ruby/Go,
// etc. False positives are fine ("looks installed but isn't" — the
// model will discover that on first verify and install). False
// negatives are bad — they push the model toward unnecessary
// reinstalls. Bias toward "ready" when the evidence is ambiguous.
func probeToolchainReady(workingDir string, tc Toolchain) string {
	switch tc.Name {
	case "python":
		for _, vd := range []string{"venv", ".venv", "env", ".env-py"} {
			sp := filepath.Join(workingDir, vd, "lib")
			if entries, err := os.ReadDir(sp); err == nil {
				for _, e := range entries {
					if strings.HasPrefix(e.Name(), "python") && e.IsDir() {
						if hasUserPackages(filepath.Join(sp, e.Name(), "site-packages")) {
							return "ready"
						}
					}
				}
			}
		}
		if hasFile(workingDir, "requirements.txt") || hasFile(workingDir, "pyproject.toml") {
			return "needs install"
		}
		return "no manifest"

	case "node":
		if entries, err := os.ReadDir(filepath.Join(workingDir, "node_modules")); err == nil && len(entries) > 0 {
			return "ready"
		}
		return "needs install"

	case "rust":
		if info, err := os.Stat(filepath.Join(workingDir, "target")); err == nil && info.IsDir() {
			return "warm"
		}
		return "cold"

	case "go":
		if info, err := os.Stat(filepath.Join(workingDir, "vendor")); err == nil && info.IsDir() {
			return "vendored"
		}
		if hasFile(workingDir, "go.sum") {
			return "ready"
		}
		return "needs `go mod tidy`"

	case "ruby":
		if info, err := os.Stat(filepath.Join(workingDir, "vendor", "bundle")); err == nil && info.IsDir() {
			return "ready"
		}
		return "needs install"

	case "java-maven", "java-gradle":
		dir := "target"
		if tc.Name == "java-gradle" {
			dir = "build"
		}
		if info, err := os.Stat(filepath.Join(workingDir, dir)); err == nil && info.IsDir() {
			return "warm"
		}
		return "cold"

	case "php":
		if info, err := os.Stat(filepath.Join(workingDir, "vendor")); err == nil && info.IsDir() {
			return "ready"
		}
		return "needs install"

	case "dotnet":
		if info, err := os.Stat(filepath.Join(workingDir, "bin")); err == nil && info.IsDir() {
			return "warm"
		}
		return "cold"

	case "dart":
		if info, err := os.Stat(filepath.Join(workingDir, ".dart_tool")); err == nil && info.IsDir() {
			return "ready"
		}
		return "needs install"
	}
	return ""
}

// displayRelativeRunner converts an absolute runner path to its
// project-relative form when it lives under workingDir. Compresses
// `/workspace/venv/bin/python` to `venv/bin/python` in prompt output —
// matches the existing "use relative paths" rule and stops the model
// confusing itself into emitting `workspace/app.py`.
func displayRelativeRunner(runner, workingDir string) string {
	if !filepath.IsAbs(runner) {
		return runner
	}
	if rel, err := filepath.Rel(workingDir, runner); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return runner
}

// hasUserPackages returns true when site-packages contains anything
// beyond pip/setuptools/wheel — i.e. the user has installed real
// project deps. Empty / pip-only venvs return false.
func hasUserPackages(sitePackages string) bool {
	entries, err := os.ReadDir(sitePackages)
	if err != nil {
		return false
	}
	skip := map[string]bool{
		"pip": true, "setuptools": true, "wheel": true,
		"pkg_resources": true, "_distutils_hack": true,
		"__pycache__": true,
	}
	for _, e := range entries {
		name := e.Name()
		// Strip dist-info / egg-info suffixes for the skip check.
		if i := strings.Index(name, "-"); i > 0 {
			name = name[:i]
		}
		if strings.HasSuffix(e.Name(), ".dist-info") || strings.HasSuffix(e.Name(), ".egg-info") {
			continue
		}
		if !skip[name] && !strings.HasPrefix(name, "_") {
			return true
		}
	}
	return false
}

// hasFile returns true when workingDir/name exists as a file.
func hasFile(workingDir, name string) bool {
	info, err := os.Stat(filepath.Join(workingDir, name))
	return err == nil && !info.IsDir()
}

// pickExisting returns the subset of names that exist as files in workingDir.
func pickExisting(workingDir string, names ...string) []string {
	var out []string
	for _, n := range names {
		if hasFile(workingDir, n) {
			out = append(out, n)
		}
	}
	return out
}

// firstMatchingGlob returns the first filename matching any of the
// glob patterns at the workingDir root, or "" if none match.
func firstMatchingGlob(workingDir string, patterns ...string) string {
	for _, p := range patterns {
		matches, _ := filepath.Glob(filepath.Join(workingDir, p))
		if len(matches) > 0 {
			return filepath.Base(matches[0])
		}
	}
	return ""
}

// detectProjectVenvPython returns the container-side path to the
// project's venv python (e.g. "/workspace/venv/bin/python") if the
// working directory has a recognisable Python virtual environment.
// Returns "" when no venv is found.
//
// The agent's working_dir is the container-internal /workspace, so
// we resolve against that. Common venv directory names: venv, .venv,
// env, .env-py — we probe in priority order and stop at the first hit.
// Inside each, look for bin/python, bin/python3, or Scripts/python.exe
// (Windows-emitted venvs occasionally end up bind-mounted on Linux).
//
// Caller passes workingDir from ctx.WorkingDir; the returned path is
// what the model should literally invoke via run_command — e.g.
// "/workspace/venv/bin/python app.py" — and what gets surfaced in the
// system prompt's venv hint.
func detectProjectVenvPython(workingDir string) string {
	if workingDir == "" {
		return ""
	}
	venvDirs := []string{"venv", ".venv", "env", ".env-py"}
	pythonRels := []string{"bin/python", "bin/python3", "Scripts/python.exe"}
	for _, vd := range venvDirs {
		for _, py := range pythonRels {
			abs := filepath.Join(workingDir, vd, py)
			if info, err := os.Stat(abs); err == nil && !info.IsDir() {
				// Return container-relative path (workingDir is already
				// the container-side /workspace), so caller can paste
				// it into a run_command argument unchanged.
				return abs
			}
		}
	}
	return ""
}

// samplePlanContext walks ctx.WorkingDir and reads a handful of files
// the planner is most likely to need: source files, templates,
// manifests. Limited to maxFiles per call, each truncated to maxBytes.
//
// The planner runs *before* any tool calls have happened in the loop,
// so ctx.FilesRead is empty — without this, plans for "fix the flask
// app" would have no signal about what's in app.py and would generate
// generic 5-step recipes. We pay one fs walk + a few small reads up
// front; the budget is small (~5 files × 2KB) and the planning quality
// jump is large.
func samplePlanContext(workingDir string, maxFiles, maxBytes int) map[string]string {
	if workingDir == "" {
		return nil
	}
	out := map[string]string{}
	// Files we always inline if present — most projects have at least
	// one of these and they describe shape (deps, entry point).
	priority := []string{
		"app.py", "main.py", "manage.py", "wsgi.py",
		"index.html", "templates/index.html", "templates/base.html",
		"package.json", "tsconfig.json", "vite.config.ts", "vite.config.js",
		"go.mod", "main.go",
		"Cargo.toml", "src/main.rs", "src/lib.rs",
		"requirements.txt", "pyproject.toml", "setup.py",
		"README.md",
	}
	for _, rel := range priority {
		if len(out) >= maxFiles {
			break
		}
		full := filepath.Join(workingDir, rel)
		info, err := os.Stat(full)
		if err != nil || info.IsDir() {
			continue
		}
		// Skip oversized files — the planner doesn't need a 50KB README.
		if info.Size() > int64(maxBytes)*4 {
			continue
		}
		data, err := os.ReadFile(full)
		if err != nil {
			continue
		}
		s := string(data)
		if len(s) > maxBytes {
			s = s[:maxBytes] + "\n... (truncated)"
		}
		out[rel] = s
	}
	// If priority files yielded nothing at the workspace root, the
	// project may live one level down — common when the user's
	// `atlas tui` cwd was the parent dir (e.g. /workspace) but the
	// flask app is at /workspace/snake/. Walk one level looking for
	// the SAME priority filenames inside subdirectories. Without
	// this, the May 2026 user-session planner saw zero context and
	// the agent wasted 3 turns finding `snake/app.py`.
	if len(out) == 0 {
		entries, err := os.ReadDir(workingDir)
		if err != nil {
			return nil
		}
		// First pass: peek into subdirectories for priority files.
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			name := e.Name()
			// Skip caches, vendors, dot-dirs — these aren't projects.
			if strings.HasPrefix(name, ".") || name == "node_modules" ||
				name == "venv" || name == "__pycache__" ||
				name == "dist" || name == "build" || name == "target" ||
				name == "vendor" {
				continue
			}
			for _, rel := range priority {
				if len(out) >= maxFiles {
					break
				}
				full := filepath.Join(workingDir, name, rel)
				info, err := os.Stat(full)
				if err != nil || info.IsDir() {
					continue
				}
				if info.Size() > int64(maxBytes)*4 {
					continue
				}
				data, err := os.ReadFile(full)
				if err != nil {
					continue
				}
				s := string(data)
				if len(s) > maxBytes {
					s = s[:maxBytes] + "\n... (truncated)"
				}
				// Key uses subdir/filename so the planner sees the
				// path the agent will need to use in tool calls.
				out[filepath.Join(name, rel)] = s
			}
			if len(out) >= maxFiles {
				break
			}
		}
		// Second pass: shallow walk of the workspace root for any
		// source-looking files (uncommon repo layout, no priority
		// hits anywhere).
		if len(out) == 0 {
			for _, e := range entries {
				if len(out) >= maxFiles {
					break
				}
				if e.IsDir() {
					continue
				}
				name := e.Name()
				ext := strings.ToLower(filepath.Ext(name))
				switch ext {
				case ".py", ".go", ".js", ".ts", ".tsx", ".jsx",
					".html", ".rs", ".rb", ".java", ".kt", ".swift":
					// pass
				default:
					continue
				}
				info, err := e.Info()
				if err != nil || info.Size() > int64(maxBytes)*4 {
					continue
				}
				data, err := os.ReadFile(filepath.Join(workingDir, name))
				if err != nil {
					continue
				}
				s := string(data)
				if len(s) > maxBytes {
					s = s[:maxBytes] + "\n... (truncated)"
				}
				out[name] = s
			}
		}
	}
	return out
}

// shouldGeneratePlan decides whether a turn warrants the ~5-15s plan
// pipeline cost. We skip plans for:
//   - T0 (trivial chat — "hi", "thanks") where a plan is wasted budget
//   - explicit follow-up / clarification requests that depend on the
//     prior turn's plan, which we'd just regenerate identically
//
// Everything else gets a plan — we'd rather plan and have the model
// ignore it than not plan and let the model thrash.
func shouldGeneratePlan(ctx *AgentContext, message string) bool {
	// A V3-bypassed demo request is the baseline side of the comparison.
	// Running the V3 planner here made that pane visibly orchestrated even
	// though its file writes bypassed V3 later in the turn.
	if ctx != nil && ctx.BypassV3 {
		return false
	}
	if ctx.Tier == Tier0Conversational {
		return false
	}
	// Single-line ack-style messages where the user is just steering
	// the existing direction ("yes do that", "looks good", "try again")
	// — already-running plan is still relevant; a fresh one would just
	// re-derive it.
	trimmed := strings.ToLower(strings.TrimSpace(message))
	if len(trimmed) < 12 {
		return false
	}
	// A request to LOOK needs no plan, and a wrong one actively steers.
	// Observed live: "Yes, please list the files" produced a three-step plan
	// (read_file app.py -> structural_edit app.py -> run_command python
	// app.py). The model listed the directory as asked, then followed the plan
	// into editing a snake game it had merely been asked to enumerate.
	return !isReadOnlyRequest(message)
}

// generatePlan hits /v3/plan with a sampled project context and the
// user's message, streaming plan_* stage events out to the TUI as
// `v3_plan` events. Returns the winning Plan or nil if the planner
// errored — callers should treat nil as "no plan, proceed without
// adherence gating".
func generatePlan(ctx *AgentContext, userMessage string) *Plan {
	if ctx.V3URL == "" {
		return nil
	}
	pctx := samplePlanContext(ctx.WorkingDir, 6, 2000)
	req := V3PlanRequest{
		UserMessage:    userMessage,
		WorkingDir:     ctx.WorkingDir,
		ProjectContext: pctx,
		ExistingFiles:  listWorkspaceFiles(ctx.WorkingDir, 400),
		NCandidates:    3,
	}

	planStart := time.Now()
	Emit(NewEnvelope(EvtStageStart, "v3:plan", map[string]interface{}{
		"detail":     fmt.Sprintf("planning: %s", truncateStr(userMessage, 60)),
		"context_n":  len(pctx),
		"candidates": req.NCandidates,
	}))

	plan, err := callV3PlanStreaming(ctx.Ctx, ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
		// Filter out per-token events — the LLM emits ~150 token deltas
		// per candidate × 3 candidates = ~450 streamed events. Forwarding
		// every one to the TUI as a separate v3_plan row clogs the
		// pipeline pane (same regression as the v3-generation token
		// spam we already fixed). The structural plan stages
		// (plan_candidate, plan_candidate_scored, plan_selected) are
		// what the renderer actually wants — token-level visibility is
		// debug noise.
		switch stage {
		case "token", "llm_start", "llm_end":
			return
		}
		payload := map[string]interface{}{"stage": stage, "detail": detail}
		for k, v := range data {
			payload[k] = v
		}
		ctx.Stream("v3_plan", payload)
		// Mirror to the typed broker so non-TUI consumers (logs, audit)
		// see the same stream.
		Emit(NewEnvelope(EvtMetric, "v3:plan:"+stage, payload))
	})
	dur := time.Since(planStart).Milliseconds()

	if err != nil {
		log.Printf("[agent] plan generation failed: %v", err)
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtStageEnd,
			Stage:      "v3:plan",
			DurationMS: dur,
			Payload:    map[string]interface{}{"success": false, "error": err.Error()},
		})
		return nil
	}

	Emit(Envelope{
		EventID:    NewEventID(),
		Timestamp:  float64(time.Now().UnixNano()) / 1e9,
		Type:       EvtStageEnd,
		Stage:      "v3:plan",
		DurationMS: dur,
		Payload: map[string]interface{}{
			"success":           true,
			"steps":             len(plan.Steps),
			"verify_step":       plan.VerifyStep,
			"winning_score":     plan.WinningScore,
			"candidates_tested": plan.CandidatesTested,
		},
	})

	// Stream the full plan structure so the TUI / IDE plugins can
	// render the step list. Per-stage events (plan_start, plan_selected,
	// etc.) only carry counts and indices — the actual step rows live
	// here. One event per plan: subsequent step satisfaction goes
	// through plan_adherence, and a revision fires another plan_loaded.
	planPayload := map[string]interface{}{
		"steps":         plan.Steps,
		"verify_step":   plan.VerifyStep,
		"rationale":     plan.Rationale,
		"winning_score": plan.WinningScore,
		"revision":      0,
	}
	ctx.Stream("plan_loaded", planPayload)
	Emit(NewEnvelope(EvtMetric, "v3:plan:loaded", planPayload))

	return plan
}

// listWorkspaceFiles returns every file in the workspace, relative, for the
// planner. v3-service has no /workspace mount, so this is the only way it can
// know what already exists — and a plan that opens by recreating existing
// input is the failure this prevents.
//
// Capped: a plan does not need ten thousand paths, and the request has to stay
// small. Names only, no content.
func listWorkspaceFiles(workingDir string, max int) []string {
	if workingDir == "" {
		return nil
	}
	var out []string
	skip := map[string]bool{".git": true, "node_modules": true,
		"__pycache__": true, ".venv": true, "venv": true, "dist": true}
	_ = filepath.WalkDir(workingDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skip[d.Name()] {
				return fs.SkipDir
			}
			return nil
		}
		if len(out) >= max {
			return fs.SkipAll
		}
		if rel, err := filepath.Rel(workingDir, path); err == nil {
			out = append(out, rel)
		}
		return nil
	})
	sort.Strings(out)
	return out
}

// recoverTruncatedText salvages the answer from a `text` response whose JSON
// was cut off mid-string.
//
// The tool-call path already has recoverTruncatedToolCall; a text answer had
// no equivalent, so a cut threw the whole thing away. Observed on
// bugfind_tiebreak: the model had written 5,897 characters answering a
// question, began repeating itself, the loop detector cut the stream, and the
// user received nothing at all — because the closing quote and brace were
// missing.
//
// Only for a stream WE cut. A response the model ended on its own is not
// truncated, and guessing at one would invent content.
func recoverTruncatedText(raw string) (string, bool) {
	const marker = `"content":`
	i := strings.Index(raw, marker)
	if i < 0 {
		if i = strings.Index(raw, `"content" :`); i < 0 {
			return "", false
		}
	}
	rest := strings.TrimSpace(raw[i+len(marker):])
	if !strings.HasPrefix(rest, `"`) {
		return "", false
	}
	rest = rest[1:]

	// Walk the JSON string body by hand: it has no closing quote, so the
	// decoder cannot help. Stop at an unescaped quote if one somehow exists.
	var sb strings.Builder
	for j := 0; j < len(rest); j++ {
		c := rest[j]
		if c == '\\' && j+1 < len(rest) {
			switch rest[j+1] {
			case 'n':
				sb.WriteByte('\n')
			case 't':
				sb.WriteByte('\t')
			case 'r':
				sb.WriteByte('\r')
			case '"':
				sb.WriteByte('"')
			case '\\':
				sb.WriteByte('\\')
			default:
				sb.WriteByte(rest[j+1])
			}
			j++
			continue
		}
		if c == '"' {
			break
		}
		sb.WriteByte(c)
	}
	out := strings.TrimSpace(sb.String())
	// Too little to be worth showing, and a short fragment is more likely to
	// mislead than help.
	if len(out) < 200 {
		return "", false
	}
	return out, true
}

// --- Phase 2B: the one terminal emitter -------------------------------------
//
// Thirteen producers each built their own done payload, so "did this run
// finish?" was answered by matching English in `summary`. A caller could not
// tell a completion from a loop-breaker without knowing every phrase the
// proxy might use, and the broker's own done envelope said "success": true
// unconditionally -- including for every stop.
//
// Every terminal now goes through here. It fires at most once per session:
// a timeout racing a completion produces one event, not two, and whichever
// arrives first is the outcome.
func emitTerminal(ctx *AgentContext, st *runState, status TerminalStatus, reason, summary string) {
	if !status.Classified() {
		// A producer that did not name its outcome does not get to imply one.
		status, reason = TerminalIncomplete, "unclassified_producer"
	}
	ctx.terminalOnce.Do(func() {
		ctx.TerminalStatus = status
		ctx.TerminalReason = reason
		if st != nil && st.pendingToolCall != "" {
			// The outstanding call is answered first, so tool_call and
			// tool_result stay balanced at every exit.
			ctx.Stream("tool_result", map[string]interface{}{
				"tool":    st.pendingToolCall,
				"success": false,
				"data":    json.RawMessage("null"),
				"error":   "not run — the session stopped before this call executed",
				"elapsed": "0s",
			})
			st.pendingToolCall = ""
		}
		// `summary` keeps its exact legacy meaning and position. `status` and
		// `reason` are additive, so a consumer that never learned about them
		// reads the same event it always did -- and reads the same TRUTH,
		// which is what honestTerminalSummary enforces.
		ctx.Stream("done", map[string]string{
			"summary": honestTerminalSummary(ctx, st, status, reason, summary),
			"status":  string(status),
			"reason":  reason,
		})
	})
}

// terminalCompletionAllowed decides whether a model-issued `done` may be
// called completed. It answers from the workspace as it is right now, never
// from the run's history.
//
// The pair-1 defect this exists to prevent: a run deleted the deliverable and
// the delete's success authorised the completion. Existence is not validity,
// an earlier successful write is not the current bytes, and a removal is not
// an achievement unless removal was the task -- which the proxy cannot
// establish, so it does not pretend to.
func terminalCompletionAllowed(ctx *AgentContext, expected []string) (bool, string) {
	if blockingTombstone(ctx) {
		// Something was deleted or moved. Whether that WAS the task is not
		// knowable here, so completion is not claimable here.
		return false, "delete_intent_unestablished"
	}
	paths := declaredOrOwnedDeliverables(ctx, expected)
	if len(paths) == 0 {
		// A run whose work was removing files the user approved has an
		// obligation and met it; saying "no file obligation" would be false.
		if len(approvedDeletionPaths(ctx)) > 0 {
			return true, "approved_deletions_demonstrated"
		}
		// Nothing declared and nothing written: there is no file obligation
		// to demonstrate.
		return true, "no_file_obligation"
	}
	if deliverablesDemonstrablyValid(ctx, paths) {
		return true, "deliverables_demonstrated"
	}
	return false, "deliverables_not_demonstrated"
}

// declaredOrOwnedDeliverables is the union of what the run said it would
// produce and what it actually wrote, minus anything deliberately removed.
// Sorted so a summary and a decision never disagree on order.
func declaredOrOwnedDeliverables(ctx *AgentContext, expected []string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(rel string) {
		if rel == "" || seen[rel] {
			return
		}
		seen[rel] = true
		out = append(out, rel)
	}
	for _, rel := range expected {
		add(rel)
	}
	if ctx != nil {
		ctx.LedgerMu.Lock()
		for key, d := range ctx.Ledger {
			if d.Tombstoned || d.Generation == 0 {
				continue
			}
			rel := key
			if r, err := filepath.Rel(ctx.WorkingDir, key); err == nil && !strings.HasPrefix(r, "..") {
				rel = r
			}
			add(rel)
		}
		ctx.LedgerMu.Unlock()
	}
	sort.Strings(out)
	return out
}

func sessionHasTombstones(ctx *AgentContext) bool {
	if ctx == nil {
		return false
	}
	ctx.LedgerMu.Lock()
	defer ctx.LedgerMu.Unlock()
	for _, d := range ctx.Ledger {
		if d.Tombstoned {
			return true
		}
	}
	return false
}

// blockingTombstone reports whether anything was removed that this run cannot
// account for.
//
// A plain deletion always blocks. Whether removing a file was the task is not
// knowable from the workspace, and a delete authorising its own completion is
// the pair-1 defect.
//
// A move is a different fact and does not need intent inferred. The source is
// gone AND the bytes are somewhere the ledger can point at -- TombstoneReason
// records exactly where, as `moved:<canonical destination>` -- so the removal
// is accounted for by the artifact that replaced it. That is only true when
// every part of it is demonstrated NOW: the reason parses, the source is
// confirmed absent on disk, the destination is readable, the ledger's record
// describes the bytes actually there, and the destination clears the same
// deliverable contract every other artifact clears. Anything less blocks.
//
// The contract is reused rather than restated, deliberately: a move must not
// be an easier way to demonstrate a file than writing one. Restoration stays
// prohibited on the tombstone either way -- this decides completion, not
// whether the old path can come back.
func blockingTombstone(ctx *AgentContext) bool {
	if ctx == nil {
		return false
	}
	ctx.LedgerMu.Lock()
	type tomb struct{ key, reason string }
	var tombs []tomb
	for k, d := range ctx.Ledger {
		if d.Tombstoned {
			tombs = append(tombs, tomb{k, d.TombstoneReason})
		}
	}
	ctx.LedgerMu.Unlock()

	for _, t := range tombs {
		if demonstratedMove(ctx, t.key, t.reason) {
			continue
		}
		if fulfilledApprovedDeletion(ctx, t.key, t.reason) {
			continue
		}
		return true
	}
	// A path the user approved removing that is back on disk is a
	// contradiction, not a completion: the approval described a removal this
	// run then undid, and recreating it does not settle what was authorised.
	// The tombstone is gone by then, so this is checked from the record.
	if undoneApprovedDeletion(ctx) {
		return true
	}
	return false
}

// undoneApprovedDeletion reports whether any deletion the user approved has
// since come back.
func undoneApprovedDeletion(ctx *AgentContext) bool {
	if ctx == nil {
		return false
	}
	ctx.mu.Lock()
	keys := make([]string, 0, len(ctx.fulfilledDeletions))
	for k := range ctx.fulfilledDeletions {
		keys = append(keys, k)
	}
	ctx.mu.Unlock()
	for _, k := range keys {
		if _, err := os.Lstat(k); err == nil {
			return true
		}
	}
	return false
}

// fulfilledApprovedDeletion answers whether a plain deletion tombstone is one
// the USER approved and the system then carried out.
//
// Every fact is re-checked here, at terminal time, against the workspace as it
// is now: the record is for this exact canonical path, the generation is the
// one that deletion produced, the path is still absent, the tombstone is a
// deletion rather than a move, restoration is still prohibited, and the delete
// debt is settled. A path that came back is a newer generation and blocks
// again -- which is the point of binding the generation rather than the name.
//
// The record itself can only exist if the decision arrived through the
// permission endpoint. No word of the user's message, and no claim of the
// model's, reaches this.
func fulfilledApprovedDeletion(ctx *AgentContext, key, reason string) bool {
	if reason != "deleted" {
		return false
	}
	f, ok := fulfilledDeletionFor(ctx, key)
	if !ok {
		return false
	}
	if _, err := os.Lstat(key); !os.IsNotExist(err) {
		return false // it is back: this record describes older bytes
	}
	ctx.LedgerMu.Lock()
	d := ctx.Ledger[key]
	var live bool
	if d != nil {
		live = d.Tombstoned && d.TombstoneReason == "deleted" &&
			d.RestoreProhibited && d.Generation == f.Generation
	}
	ctx.LedgerMu.Unlock()
	return live
}

// approvedDeletionPaths lists, in a stable order, the paths this run removed
// with the user's approval. Bounded for disclosure.
func approvedDeletionPaths(ctx *AgentContext) []string {
	if ctx == nil {
		return nil
	}
	ctx.LedgerMu.Lock()
	keys := make([]string, 0, len(ctx.Ledger))
	for k, d := range ctx.Ledger {
		if d.Tombstoned && d.TombstoneReason == "deleted" {
			keys = append(keys, k)
		}
	}
	ctx.LedgerMu.Unlock()
	var out []string
	for _, k := range keys {
		if fulfilledApprovedDeletion(ctx, k, "deleted") {
			out = append(out, relativeToWorkspace(ctx, k))
		}
	}
	sort.Strings(out)
	return out
}

// relativeToWorkspace renders a canonical path the way the user wrote it.
func relativeToWorkspace(ctx *AgentContext, key string) string {
	if ctx == nil || ctx.WorkingDir == "" {
		return key
	}
	if rel, err := filepath.Rel(ctx.WorkingDir, key); err == nil &&
		!strings.HasPrefix(rel, "..") {
		return rel
	}
	return key
}

// demonstratedMove answers whether one tombstone is a relocation this run can
// point at, rather than a removal it cannot explain.
func demonstratedMove(ctx *AgentContext, srcKey, reason string) bool {
	dest := strings.TrimPrefix(reason, "moved:")
	if dest == reason || strings.TrimSpace(dest) == "" {
		return false // a plain deletion, or a reason that says nothing
	}
	// The source has to be gone right now, not merely reported gone.
	if _, err := os.Stat(srcKey); !os.IsNotExist(err) {
		return false
	}
	data, ok := readLedgerBytes(dest)
	if !ok {
		return false
	}
	ctx.LedgerMu.Lock()
	d := ctx.Ledger[dest]
	var current string
	var tombstoned bool
	if d != nil {
		current, tombstoned = d.CurrentHash, d.Tombstoned
	}
	ctx.LedgerMu.Unlock()
	// The destination must be a live deliverable the ledger still describes.
	if d == nil || tombstoned || current != hashBytes(data) {
		return false
	}
	// And it must clear the same bar as any other artifact.
	if !deliverablesDemonstrablyValid(ctx, []string{dest}) {
		return false
	}
	// Whatever the session still owes on this move is owed regardless. The
	// debt gate in finalizeCompletion reads it a few lines later and names it
	// specifically, so pre-empting it here would only replace a precise
	// terminal with a vaguer one.
	return true
}

// --- Phase 2B: the server-owned session budget ------------------------------
//
// The proxy had no clock of its own. A session ran until the model stopped,
// the client gave up, or a detector fired, and a client that timed out
// mid-stream took the only explanation with it -- the Stage-1 sessions that
// reached 590s did so against the HARNESS cap, not a server one, and the
// server never got to say what it had.
//
// The budget is owned here, and it is split: work stops one reserve early so
// the reserve can be spent on stopping cleanly and saying so.

const (
	defaultSessionTotalSec = 600
	sessionReserve         = 30 * time.Second
	minSessionTotalSec     = 120
	maxSessionTotalSec     = 3600
)

// sessionBudget returns the total session limit and the reserve held back for
// finalisation. ATLAS_AGENT_SESSION_TIMEOUT_SEC overrides the total within
// conservative bounds; anything malformed, zero, negative or out of range
// falls back to the default and says so, because a silently ignored operator
// setting is worse than no setting.
func sessionBudget() (total, reserve time.Duration) {
	total = defaultSessionTotalSec * time.Second
	raw := strings.TrimSpace(os.Getenv("ATLAS_AGENT_SESSION_TIMEOUT_SEC"))
	if raw == "" {
		return total, sessionReserve
	}
	n, err := strconv.Atoi(raw)
	switch {
	case err != nil:
		log.Printf("[agent] ATLAS_AGENT_SESSION_TIMEOUT_SEC=%q is not a number — using %ds",
			raw, defaultSessionTotalSec)
	case n < minSessionTotalSec:
		log.Printf("[agent] ATLAS_AGENT_SESSION_TIMEOUT_SEC=%d is below the %ds floor — using %ds",
			n, minSessionTotalSec, defaultSessionTotalSec)
	case n > maxSessionTotalSec:
		log.Printf("[agent] ATLAS_AGENT_SESSION_TIMEOUT_SEC=%d is above the %ds ceiling — using %ds",
			n, maxSessionTotalSec, defaultSessionTotalSec)
	default:
		total = time.Duration(n) * time.Second
	}
	return total, sessionReserve
}

// finalizeOnWorkDeadline is what the reserve is for. It runs after the work
// context is done and before the handler returns, on the response lifetime.
//
// Order matters: nothing may look at the workspace until the things that
// could still be writing to it have been confirmed gone.
func finalizeOnWorkDeadline(ctx *AgentContext, st *runState) {
	// 1. Stop anything still running on the work context.
	if ctx.cancelWork != nil {
		ctx.cancelWork()
	}
	// 2. Reap this session's background jobs and confirm they exited. Only
	// this session's -- another session's server is not ours to kill.
	reapSessionBackgroundJobs(ctx)
	// 3. Now the workspace is quiet, so a hash means something. Re-read every
	// tracked path: a job killed mid-write leaves bytes nobody validated.
	invalidateTrackedValidation(ctx)
	// 4. Decide restoration per path, under the Phase 3B rules. A timeout
	// does not relax any of them.
	recovered := restoreSaferDeliverables(ctx)
	// 5. One terminal, on the response lifetime, inside the reserve. A
	// timeout never claims completion, whatever is on disk afterwards.
	wrote := st != nil && st.madeProductiveChange
	emitTerminal(ctx, st, TerminalTimedOut, "work_deadline",
		sessionTimeoutSummary(ctx, wrote, recovered))
}

// reapSessionBackgroundJobs stops the jobs THIS session started and waits for
// each to be confirmed gone. A job that cannot be confirmed leaves the
// workspace hazard raised, which is what keeps restoration from touching a
// file something may still be writing.
func reapSessionBackgroundJobs(ctx *AgentContext) {
	if ctx == nil || len(ctx.BackgroundJobs) == 0 {
		return
	}
	ids := make([]string, 0, len(ctx.BackgroundJobs))
	for id := range ctx.BackgroundJobs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		out, err := sandboxStopBackground(ctx, id)
		delete(ctx.BackgroundJobs, id)
		if err != nil {
			log.Printf("[agent] could not stop background job %s: %v", id, err)
			continue
		}
		if out.ExitCode != nil {
			clearWorkspaceHazard(ctx, id)
			continue
		}
		log.Printf("[agent] background job %s did not report an exit code — "+
			"the workspace stays marked as possibly still being written", id)
	}
}

// sessionTimeoutSummary is the terminal a timed-out session ends on. It says
// the run ran out of time and what state the files are in; it never says the
// work is done, and a successful restore does not change that.
func sessionTimeoutSummary(ctx *AgentContext, wrote bool, recovered []restoreDecision) string {
	var sb strings.Builder
	sb.WriteString("Stopped: the session ran out of time before the work finished")
	switch {
	case !wrote:
		sb.WriteString(", and nothing was written to disk")
	default:
		sb.WriteString(". Anything already written is still on disk, unverified")
	}
	sb.WriteString(". Try a smaller, more specific request.")
	sb.WriteString(restorationDisclosure(recovered))
	sb.WriteString(liveBackgroundJobNote(ctx))
	return sb.String()
}

// finishCancelledRun tells apart the two ways a run can stop early, because
// they are not the same event and must not be reported as one.
//
// The work deadline is OURS: the client is still there, the reserve is
// unspent, and the session owes an explanation. A client disconnect is not a
// server timeout -- the response channel is gone, so there is nobody to tell,
// and claiming timed_out into a closed socket would put a fact in the record
// that nothing observed.
//
// Either way the work stops and this session's background jobs are reaped.
func finishCancelledRun(ctx *AgentContext, st *runState, turn int) error {
	clientGone := ctx.RequestCtx != nil && ctx.RequestCtx.Err() != nil
	if clientGone {
		log.Printf("[agent] client disconnected at turn %d — cancelling work and reaping jobs", turn)
		if ctx.cancelWork != nil {
			ctx.cancelWork()
		}
		reapSessionBackgroundJobs(ctx)
		return ctx.Ctx.Err()
	}
	if errors.Is(ctx.Ctx.Err(), context.DeadlineExceeded) {
		log.Printf("[agent] work deadline reached at turn %d — finalising within the reserve", turn)
		finalizeOnWorkDeadline(ctx, st)
		return nil
	}
	// An explicit POST /cancel: the caller asked to stop, so the run ends
	// incomplete rather than pretending it ran out of time.
	log.Printf("[agent] cancelled at turn %d: %v", turn, ctx.Ctx.Err())
	if ctx.cancelWork != nil {
		ctx.cancelWork()
	}
	reapSessionBackgroundJobs(ctx)
	emitTerminal(ctx, st, TerminalIncomplete, "cancelled",
		"Stopped: the run was cancelled before the work finished."+liveBackgroundJobNote(ctx))
	return nil
}

// --- Phase 4A: the summary a non-completed run is allowed to carry ----------
//
// Four of fifty Stage-1 sessions ended with the model's own prose --
// "I have successfully implemented the interval priority logic in solve.py" --
// over an artifact nothing had verified. Three more ended with no summary at
// all. Phase 2B made `status` honest; a client reading only `summary`, which
// is every client that predates that field, still read a success.
//
// So the server owns the sentence whenever it does not own a completion. A
// completed status keeps the model's account, because the exact-hash gate has
// already agreed with it.

// completionClaims are the phrases a finished run is allowed to use. Matching
// is on the claim, not on the word: "no verification command completed
// successfully" is a report of failure and must not trip this, so each phrase
// is checked for a negation immediately before it.
var completionClaims = []string{
	"successfully implemented", "successfully created", "successfully wrote",
	"successfully added", "successfully fixed", "successfully completed",
	"i have successfully", "made your change", "the change is on disk",
	"final product", "task is complete", "task is done", "work is complete",
	"is now complete", "everything works", "all tests pass", "it works correctly",
	"correctly handles", "correctly processes", "correctly implements",
}

// negators immediately preceding a claim invert it.
var claimNegators = []string{
	"no ", "not ", "never ", "cannot ", "can't ", "could not ", "couldn't ",
	"did not ", "didn't ", "without ", "nothing ", "fails to ", "failed to ",
	"unverified", "unconfirmed",
}

// completionClaimIn returns the first unnegated completion claim in s, or "".
func completionClaimIn(s string) string {
	low := strings.ToLower(s)
	for _, claim := range completionClaims {
		from := 0
		for {
			i := strings.Index(low[from:], claim)
			if i < 0 {
				break
			}
			at := from + i
			window := low[max(0, at-48):at]
			negated := false
			for _, n := range claimNegators {
				if strings.Contains(window, n) {
					negated = true
					break
				}
			}
			if !negated {
				return claim
			}
			from = at + len(claim)
		}
	}
	return ""
}

// honestMarkers are the ways a summary can already be saying the run did not
// finish. One of them must be present on every non-completed terminal.
var honestMarkers = []string{
	"stopped", "ran out of time", "ran out of turns", "nothing was written",
	"cannot say the task is done", "did not confirm", "not reported as finished",
	"unverified", "before finishing", "was cancelled", "could not continue",
	"not shown to be valid", "check them before relying", "run it yourself",
	"partial", "did not complete",
}

func hasHonestMarker(s string) bool {
	low := strings.ToLower(s)
	for _, m := range honestMarkers {
		if strings.Contains(low, m) {
			return true
		}
	}
	return false
}

// honestTerminalSummary is the last thing between a terminal and the client.
//
// For a completed status it changes nothing: the gate authorised the claim, so
// the account stands. For every other status it guarantees three properties --
// there is a summary, it carries no completion claim, and it says plainly that
// the task was not confirmed finished.
// deletionSummaryLimit bounds how many paths a completion summary names before
// counting the rest, so a large tidy-up stays readable.
const deletionSummaryLimit = 5

func honestTerminalSummary(ctx *AgentContext, st *runState, status TerminalStatus,
	reason, summary string) string {
	if status.Completed() {
		// A run whose work was removing files says which ones, from the
		// ledger rather than from anything the model wrote. Paths only: no
		// hashes, no ledger vocabulary, no permission machinery.
		if reason == "approved_deletions_demonstrated" {
			if paths := approvedDeletionPaths(ctx); len(paths) > 0 {
				named := paths
				more := ""
				if len(named) > deletionSummaryLimit {
					more = fmt.Sprintf(" and %d more", len(named)-deletionSummaryLimit)
					named = named[:deletionSummaryLimit]
				}
				line := fmt.Sprintf("Deleted %s%s, as you approved. Confirmed gone.",
					strings.Join(named, ", "), more)
				if s := strings.TrimSpace(summary); s != "" {
					return line + "\n\n" + s
				}
				return line
			}
		}
		return summary
	}
	out := strings.TrimSpace(summary)
	if claim := completionClaimIn(out); claim != "" {
		// A producer, or prose that reached one, is claiming completion on a
		// run that did not complete. The server replaces it outright rather
		// than editing around it.
		log.Printf("[agent] terminal (%s/%s) carried the completion claim %q — replacing the summary",
			status, reason, claim)
		out = ""
	}
	if out == "" {
		out = serverTerminalFallback(ctx, st, status, reason)
	}
	if !hasHonestMarker(out) {
		out += " This run did not confirm the task was complete."
	}
	return out
}

// serverTerminalFallback is what the user reads when the producer had nothing
// to say, or said something the run cannot support. It reports only facts the
// server holds: what the outcome was, whether anything is on disk, and whether
// the deliverables were shown to be valid.
func serverTerminalFallback(ctx *AgentContext, st *runState, status TerminalStatus, reason string) string {
	var sb strings.Builder
	switch status {
	case TerminalTimedOut:
		sb.WriteString("Stopped: the session ran out of time before the work finished.")
	case TerminalFailed:
		sb.WriteString("Stopped: the run could not continue.")
	case TerminalStopped:
		sb.WriteString("Stopped: the run was cut short before the work finished.")
	default:
		sb.WriteString("Stopped: the run ended without finishing the task.")
	}

	wrote := st != nil && st.madeProductiveChange
	switch {
	case !wrote:
		sb.WriteString(" Nothing was written to disk.")
	default:
		var expected []string
		if st != nil {
			expected = st.expectedOutputs
		}
		paths := declaredOrOwnedDeliverables(ctx, expected)
		switch {
		case len(paths) == 0:
			sb.WriteString(" Changes were written to disk and nothing verified them.")
		case deliverablesDemonstrablyValid(ctx, paths):
			sb.WriteString(" What is on disk parses, but nothing in this run verified it does " +
				"the right thing.")
		default:
			sb.WriteString(" Changes are on disk and were not shown to be valid — treat them " +
				"as unverified.")
		}
	}
	sb.WriteString(" This run did not confirm the task was complete.")
	sb.WriteString(liveBackgroundJobNote(ctx))
	return sb.String()
}

// modelProseIfAuthorized passes the model's own account through only where the
// completion gate has already agreed with it. Elsewhere it returns "", and the
// server composes the summary instead.
func modelProseIfAuthorized(status TerminalStatus, prose string) string {
	if status.Completed() {
		return prose
	}
	return ""
}

// --- Phase 4B: the C5 recovery transition -----------------------------------
//
// The observed state: a warned version of the file is on disk, the run-first
// gate is demanding it be run, and the model answers with the identical raw
// `@fenced` write it has already sent. In the frozen run that exchange
// repeated until the gate's bounce budget ran out, after which the same writes
// started landing again, and the session reached the 600 s cap having made no
// progress and produced no terminal at all.
//
// Repeating a demand the model has already failed to satisfy is not a
// mechanism. On the recurrence it gets the one thing it has not been able to
// obtain for itself -- the file as it actually is -- and the call that made no
// progress is held back until it changes.
//
// Deliberately narrow. It runs BEFORE fenced resolution, so a blocked repeat
// costs zero generations. It reads; it never writes, never runs a command, and
// never forces a tool. It fires at most once per canonical path, and the
// budget has to be there to spend.

// fencedRecoveryFloor is the work budget a recovery needs to be worth doing:
// enough for the model to read the context, run something, and write once.
const fencedRecoveryFloor = 90 * time.Second

// fencedRunFirstRecovery returns the focused context to hand back, or "" when
// this is not the state, the recovery is already spent, or there is not enough
// budget left to act on it.
func fencedRunFirstRecovery(ctx *AgentContext, st *runState, relPath, content string) string {
	// Raw model intent only. An inline write is the model doing something
	// different, which is exactly what this is asking for.
	if !strings.HasPrefix(strings.TrimSpace(content), "@fenced") {
		return ""
	}
	key := ledgerKey(ctx, relPath)
	if st.fencedRecoverySpent[key] {
		return ""
	}
	if st.fencedRunFirstRepeats == nil {
		st.fencedRunFirstRepeats = map[string]int{}
	}
	st.fencedRunFirstRepeats[key]++
	// The first occurrence is the gate's own business. This is the recurrence.
	if st.fencedRunFirstRepeats[key] < 2 {
		return ""
	}
	// A recovery the run cannot afford to act on is worse than stopping: it
	// spends the remaining budget on context nobody gets to use.
	if ctx.Ctx != nil {
		if deadline, ok := ctx.Ctx.Deadline(); ok && time.Until(deadline) < fencedRecoveryFloor {
			log.Printf("[agent] skipping the run-first recovery for %s — %v of budget left",
				relPath, time.Until(deadline).Round(time.Second))
			return ""
		}
	}

	source, truncated, err := boundedCurrentSource(ctx, relPath)
	if err != nil {
		// Nothing to show. The gate's own path still applies.
		log.Printf("[agent] run-first recovery for %s could not read the file: %v", relPath, err)
		return ""
	}
	if st.fencedRecoverySpent == nil {
		st.fencedRecoverySpent = map[string]bool{}
	}
	st.fencedRecoverySpent[key] = true
	log.Printf("[agent] run-first recovery for %s — supplying the current source once", relPath)

	var sb strings.Builder
	fmt.Fprintf(&sb, "You have now sent the same whole-file write for %s twice without "+
		"anything changing on disk, so re-sending it is not a route to a working file. "+
		"Here is what %s actually contains right now", relPath, relPath)
	if truncated {
		fmt.Fprintf(&sb, " (first %d lines)", fencedRecoveryMaxLines)
	}
	sb.WriteString(":\n\n")
	sb.WriteString(source)
	sb.WriteString("\n\n")
	if detail := currentValidationDetail(ctx, relPath); detail != "" {
		fmt.Fprintf(&sb, "The last thing checked about those exact bytes: %s\n\n", detail)
	}
	fmt.Fprintf(&sb, "That version is on disk with a parse warning and has never been run. "+
		"Do one of these instead of sending that write again: run it with run_command "+
		"(`python3 %s`) and read the real error, read more of it with read_file, or send a "+
		"correction that is materially different from what is above — a targeted edit_file or "+
		"replace_lines against a line you can see here is usually smaller and lands more "+
		"often than another whole-file rewrite.", relPath)
	return sb.String()
}

// fencedRecoveryMaxLines bounds what the recovery reads back. Enough to see a
// small solution whole and the top of a large one; never the whole file.
const fencedRecoveryMaxLines = 120

// boundedCurrentSource reads the file through the workspace reader the tools
// use and returns it numbered and bounded. Nothing is retained: the text goes
// into one message and the caller keeps only a spent flag.
func boundedCurrentSource(ctx *AgentContext, relPath string) (string, bool, error) {
	data, _, err := readWorkspaceFile(ctx, relPath)
	if err != nil {
		return "", false, err
	}
	lines := strings.Split(string(data), "\n")
	truncated := false
	if len(lines) > fencedRecoveryMaxLines {
		lines = lines[:fencedRecoveryMaxLines]
		truncated = true
	}
	var sb strings.Builder
	for i, l := range lines {
		fmt.Fprintf(&sb, "%d\t%s\n", i+1, l)
	}
	return strings.TrimRight(sb.String(), "\n"), truncated, nil
}

// steerRecoveryRepeat is the number of ignored steering refusals on one path
// that buys the recovery. The first refusal is the steer itself; the second is
// the model ignoring it, which is the only evidence that repeating the
// diagnostic will not work.
const steerRecoveryRepeat = 2

// steerRecovery answers the second ignored write_file steer on a path with the
// thing the model was missing, once.
//
// The two steering branches send a model that asked to overwrite an existing
// file somewhere better: read it first, or edit it instead. Both are correct
// and both are only text. A model that ignores the text repeats the identical
// write, and repeating the identical diagnostic back cannot change that --
// measured at 31 turns before the failure accounting bounded it, and the bound
// is a stop, not an outcome. The two refusals fail for different reasons, so
// they get different recoveries:
//
//   - unread: the model has never seen the file. Show it, bounded, through the
//     same reader read_file uses, and record exactly what was shown the way
//     read_file records a truncated read. It genuinely has the body now, so
//     the read state is real -- but the file is still not session-owned, and
//     the surgical-edit gate still stands.
//   - already read: the model has the body and reached for the wrong tool.
//     Re-showing it teaches nothing, so this is one reminder naming the tools
//     that work on an existing file. Neither is imposed: edit_file stays right
//     for a surgical change, structural_edit for a whole node.
//
// Bounded by construction: one recovery per canonical path, spent whether or
// not the model takes it, and released only by a materially different action
// on that same path. The refused call is still accounted as a failure, so the
// recovery buys the model a better turn, never an extra one.
func steerRecovery(ctx *AgentContext, st *runState, relPath, resolvedPath string, unread bool) string {
	if ctx == nil || st == nil {
		return ""
	}
	// A run that is already ending owns its own terminal.
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		return ""
	}
	key := ledgerKey(ctx, relPath)
	if st.steerRepeats == nil {
		st.steerRepeats = map[string]int{}
	}
	st.steerRepeats[key]++
	if st.steerRepeats[key] < steerRecoveryRepeat || st.steerRecovered[key] {
		return ""
	}
	// Context nobody has time to act on is worse than stopping.
	if ctx.Ctx != nil {
		if deadline, ok := ctx.Ctx.Deadline(); ok && time.Until(deadline) < fencedRecoveryFloor {
			log.Printf("[agent] skipping the write_file steering recovery for %s — %v of budget left",
				relPath, time.Until(deadline).Round(time.Second))
			return ""
		}
	}
	if st.steerRecovered == nil {
		st.steerRecovered = map[string]bool{}
	}
	st.steerRecovered[key] = true

	var sb strings.Builder
	if unread {
		source, truncated, err := boundedCurrentSource(ctx, relPath)
		if err != nil {
			// The file was there a moment ago and is not readable now. Say
			// nothing rather than something untrue; the plain steer stands.
			log.Printf("[agent] steering recovery for %s could not read it: %v", relPath, err)
			return ""
		}
		fmt.Fprintf(&sb, "You have asked to overwrite %s twice without reading it, so here it is",
			relPath)
		if truncated {
			fmt.Fprintf(&sb, " (first %d lines)", fencedRecoveryMaxLines)
		}
		sb.WriteString(":\n\n")
		sb.WriteString(source)
		sb.WriteString("\n\n")
		sb.WriteString("This is the real file. If it holds input or configuration you were given, " +
			"your replacement would have destroyed it. Now that you have seen it, change it in " +
			"place with edit_file (old_str/new_str) or structural_edit (a selector and the new " +
			"body) — whichever fits the change you mean to make. write_file on this path will " +
			"keep being refused.")
		// Record only what was shown, exactly as read_file does for a
		// truncated read: the model owns what it saw and nothing more.
		shown := source
		if !truncated {
			if data, _, err := readWorkspaceFile(ctx, relPath); err == nil {
				shown = string(data)
			}
		}
		ctx.RecordFileRead(resolvedPath, shown)
		ctx.RecordBodySeen(resolvedPath)
		log.Printf("[agent] steering recovery for %s: showed the file (truncated=%v)", relPath, truncated)
	} else {
		fmt.Fprintf(&sb, "You have already read %s, and write_file will keep refusing it — "+
			"repeating the same call cannot land. You do not need to read it again. Make the "+
			"change in place: edit_file with old_str/new_str for a surgical change, or "+
			"structural_edit with a selector and the new body for a whole function or element. "+
			"If what you actually want is a different file, write_file works on a path that "+
			"does not exist yet.", relPath)
		log.Printf("[agent] steering recovery for %s: reminded once, no reread", relPath)
	}
	return sb.String()
}

// clearSteerState releases a path's steering state after a materially
// different action succeeded on it.
//
// "Materially different" is doing something other than the refused write on
// the same file. An unrelated success elsewhere is not evidence that this path
// is unstuck, which is why this is keyed and not a global reset.
func clearSteerState(ctx *AgentContext, st *runState, name string, args json.RawMessage) {
	if st == nil || name == "write_file" {
		return
	}
	key := ledgerKey(ctx, workspaceRefusalPath(ctx, name, args))
	delete(st.steerRepeats, key)
	delete(st.steerRecovered, key)
}

// --- C4: replacements refused while the known-good bytes survive -------------
//
// Retained twice in Stage 1: a valid artifact was already on disk, every later
// replacement was refused for a syntax failure before any byte moved, and the
// model kept sending replacements until a breaker ended the run. Disk was
// never damaged and the task was never finished.
//
// This is not C3, where the file on disk is itself broken and nothing valid
// was ever kept, and it is not restoration, because the safer bytes never left
// disk -- they need preserving, which the refusal already does. What is
// missing is that the model is never shown what it is replacing, or told
// plainly that its replacement was thrown away and the good version is intact.
//
// Two hashes, two purposes, deliberately not shared. The retry fingerprint
// from the identity change is a NORMALISED sha1 -- trailing whitespace is
// dropped -- which is right for "is this the same call" and wrong for "is this
// diagnostic about these bytes". Evidence uses sha256 of the exact resolved
// proposal, and a diagnostic is stored with its hash or not at all.
const (
	// Ceilings, both on LIVE state: how many paths are tracked at once, and
	// how many distinct proposals are remembered for the bytes currently on
	// one of them. A session doing more than this is not being helped by
	// remembering more of it.
	maxC4Generations = 8
	maxC4Proposals   = 8
)

// proposalRejection is one canonical path's CURRENT generation: the surviving
// bytes everything here is about, the exact proposals refused against them --
// each with the diagnostic produced for those bytes -- and whether its one
// recovery has been spent.
//
// One entry per path, never one per generation. Keying the map on path AND
// surviving hash looked tidier and starved the thing it was meant to protect:
// every correction a path lands is a new surviving hash, so a single file
// iterating eight times filled a session-wide ceiling with obsolete entries
// and the ninth generation -- the live one -- was refused a recovery by its
// own history. New bytes now REPLACE the generation in place, which is what
// "released when the surviving disk hash changes" has to mean.
type proposalRejection struct {
	diskHash    string            // the surviving bytes this generation is about
	diagnostics map[string]string // proposal sha256 -> its own diagnostic
	order       []string
	recovered   bool
}

// reset re-arms an entry for a new generation of surviving bytes. The old
// hashes and diagnostics go with the bytes they described.
func (e *proposalRejection) reset(diskHash string) {
	e.diskHash = diskHash
	e.diagnostics = map[string]string{}
	e.order = nil
	e.recovered = false
}

// resolvedProposalHash is sha256 of the exact bytes a write_file would have
// written, after fenced resolution. "" for anything else.
func resolvedProposalHash(name string, args json.RawMessage) string {
	if name != "write_file" {
		return ""
	}
	var in WriteFileInput
	if json.Unmarshal(args, &in) != nil || in.Content == "" {
		return ""
	}
	return hashBytes([]byte(in.Content))
}

// survivingKnownGood returns the canonical path and the hash of its bytes on
// disk, when those bytes are readable, are what the ledger describes, and are
// demonstrably valid. Everything else -- unknown, not_run, not_applicable,
// failed, a verdict about other bytes, an unreadable path -- returns "".
func survivingKnownGood(ctx *AgentContext, relPath string) (canon, diskHash string) {
	if ctx == nil {
		return "", ""
	}
	key := ledgerKey(ctx, relPath)
	data, ok := readLedgerBytes(key)
	if !ok {
		return "", ""
	}
	h := hashBytes(data)
	ctx.LedgerMu.Lock()
	d := ctx.Ledger[key]
	var status ValidationStatus
	var kind ValidationKind
	var current string
	if d != nil {
		kind, status = d.CurrentValidation()
		current = d.CurrentHash
	}
	ctx.LedgerMu.Unlock()
	if current != h || status != ValidationPassed || kind != ValidationKindSyntax {
		return "", ""
	}
	return key, h
}

// evictStaleGenerations drops entries whose surviving bytes are no longer the
// bytes on disk. Only provable staleness is evicted -- a path still holding
// the bytes its evidence describes is never touched -- so the ceiling bounds
// how many LIVE paths are tracked rather than how many times the session has
// been round the loop.
func evictStaleGenerations(ctx *AgentContext, st *runState) {
	for path, ev := range st.c4Rejected {
		data, ok := readLedgerBytes(path)
		if !ok || hashBytes(data) != ev.diskHash {
			delete(st.c4Rejected, path)
		}
	}
}

// noteRejectedProposal records a refused replacement against the known-good
// bytes that survived it, and reports how many distinct proposals this
// generation has now refused.
//
// Every clause is about evidence that already exists. The diagnostic is the
// one the checker produced for THESE bytes, carried on the result; no error
// prose is parsed, no lens sample is read, and no historical verdict is reused.
func noteRejectedProposal(ctx *AgentContext, st *runState, name string,
	args json.RawMessage, result *ToolResult) (string, string, int) {
	if st == nil || result == nil || name != "write_file" {
		return "", "", 0
	}
	if result.MutationStatus != MutationRefused ||
		result.ValidationStatus != ValidationFailed ||
		result.ValidationKind != ValidationKindSyntax ||
		result.ValidationDetail == "" {
		return "", "", 0
	}
	sha := resolvedProposalHash(name, args)
	if sha == "" {
		return "", "", 0
	}
	rel := ledgerArgPath(args, "path")
	canon, diskHash := survivingKnownGood(ctx, rel)
	if canon == "" {
		return "", "", 0
	}
	if st.c4Rejected == nil {
		st.c4Rejected = map[string]*proposalRejection{}
	}
	ev := st.c4Rejected[canon]
	switch {
	case ev == nil:
		if len(st.c4Rejected) >= maxC4Generations {
			// Make room only where the evidence is provably about bytes that
			// are gone. If every tracked path still holds what its evidence
			// describes, this fails closed: nothing is recorded, so no
			// diagnostic can be offered for these bytes at all.
			evictStaleGenerations(ctx, st)
			if len(st.c4Rejected) >= maxC4Generations {
				return "", "", 0
			}
		}
		ev = &proposalRejection{}
		ev.reset(diskHash)
		st.c4Rejected[canon] = ev
	case ev.diskHash != diskHash:
		// The surviving bytes changed: this is a new question, and the old
		// hashes and diagnostics are released with the bytes they described.
		ev.reset(diskHash)
	}
	if _, seen := ev.diagnostics[sha]; !seen {
		if len(ev.order) >= maxC4Proposals {
			return canon, sha, len(ev.order)
		}
		ev.order = append(ev.order, sha)
	}
	// Stored together, always: a hash without its own diagnostic would be a
	// diagnostic waiting to be attached to the wrong bytes.
	ev.diagnostics[sha] = result.ValidationDetail
	return canon, sha, len(ev.order)
}

// rejectedProposalRecovery shows the model the file it is replacing and says
// what happened to its replacement, once per path and surviving-bytes
// generation.
//
// It mutates nothing, runs nothing, invents no selector and claims no
// completion. The diagnostic it quotes is the one stored against this exact
// proposal hash; a different proposal never inherits it.
func rejectedProposalRecovery(ctx *AgentContext, st *runState, relPath, canon, sha string) string {
	if ctx == nil || st == nil || canon == "" || sha == "" {
		return ""
	}
	// A run that is already ending owns its own terminal.
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		return ""
	}
	ev := st.c4Rejected[canon]
	if ev == nil || ev.recovered {
		return ""
	}
	// The evidence has to still be about the bytes that are there now.
	if _, diskHash := survivingKnownGood(ctx, relPath); diskHash != ev.diskHash {
		return ""
	}
	detail, bound := ev.diagnostics[sha]
	if !bound || detail == "" {
		// No diagnostic for THESE bytes. Saying nothing beats saying
		// something true about a different proposal.
		return ""
	}
	// Context nobody has time to act on is worse than stopping.
	if ctx.Ctx != nil {
		if deadline, ok := ctx.Ctx.Deadline(); ok && time.Until(deadline) < fencedRecoveryFloor {
			log.Printf("[agent] skipping the refused-replacement recovery for %s — %v of budget left",
				relPath, time.Until(deadline).Round(time.Second))
			return ""
		}
	}
	source, truncated, err := boundedCurrentSource(ctx, relPath)
	if err != nil {
		log.Printf("[agent] refused-replacement recovery for %s could not read it: %v", relPath, err)
		return ""
	}
	ev.recovered = true
	log.Printf("[agent] refused-replacement recovery for %s: %d proposal(s) refused against surviving valid bytes",
		relPath, len(ev.order))

	var sb strings.Builder
	fmt.Fprintf(&sb, "Your replacement for %s failed its syntax check, so it was not written and "+
		"nothing on disk changed. The working version is still there, exactly as it was.\n\n", relPath)
	fmt.Fprintf(&sb, "%s currently contains", relPath)
	if truncated {
		fmt.Fprintf(&sb, " (first %d lines)", fencedRecoveryMaxLines)
	}
	sb.WriteString(":\n\n")
	sb.WriteString(source)
	fmt.Fprintf(&sb, "\n\nThe check on the bytes you just sent failed: %s\n\n", detail)
	sb.WriteString("That is the version you are replacing. Send something materially different " +
		"from what was just refused: a whole new file with write_file, or -- usually better here, " +
		"since the file above already works -- a targeted change with edit_file, or structural_edit " +
		"if you are replacing a whole function or element. Re-sending the same content will fail " +
		"the same way.")
	return sb.String()
}

// --- C3: the no-op edit over a demonstrably broken artifact ------------------
//
// Retained twice in Stage 1, and identical both times: write_file lands a file
// that does not parse and says so, verification reports the concrete failure,
// edit_file demands a read, the model reads, and then it sends an edit whose
// old_str and new_str are the same string -- over and over, until repetition
// protection ends the run with the broken file on disk and no valid version to
// fall back to.
//
// The class is already bounded, and a bound is not an outcome. The model is
// copying a span it cannot reproduce with a change applied; edit_file already
// tells it the two sides match, and being told that a second time is the one
// thing already known not to work. So the recurrence gets the evidence
// instead: the file as it is now, and the failure already recorded against
// exactly those bytes.
//
// Nothing here mutates, runs, guesses the intended character, or converts the
// edit. The model has to supply the correction; this only makes that possible.

// c3RecoveryRepeat is the number of explicit no-op edits on one evidence
// generation that arms the recovery. The first is an accident and gets the
// tool's own answer unchanged; the second is evidence that answer did not work.
const c3RecoveryRepeat = 2

// noopEditIntent returns the path of an edit_file call whose old_str and
// new_str are identical, and "" for anything else.
//
// Deliberately only the explicit form, which is the one the retained evidence
// shows. edit_file also refuses edits that are merely INEFFECTIVE -- a
// replacement that leaves the file byte-identical -- and that is a different
// failure with different evidence, so it stays where it is.
func noopEditIntent(name string, args json.RawMessage) string {
	if name != "edit_file" {
		return ""
	}
	var in struct {
		Path   string `json:"path"`
		OldStr string `json:"old_str"`
		NewStr string `json:"new_str"`
	}
	if json.Unmarshal(args, &in) != nil {
		return ""
	}
	if strings.TrimSpace(in.Path) == "" || in.OldStr != in.NewStr {
		return ""
	}
	return in.Path
}

// brokenArtifactRecovery answers a repeated no-op edit with the current file
// and the failure already bound to it, once per evidence generation.
//
// Every clause is an entry condition, and all of them are about evidence that
// already exists: the ledger's verdict on the bytes that are on disk right
// now. Nothing is inferred from error prose, and a verdict that describes
// other bytes is not usable -- CurrentValidation is what enforces that.
func brokenArtifactRecovery(ctx *AgentContext, st *runState, relPath string) string {
	if ctx == nil || st == nil {
		return ""
	}
	// A run that is already ending owns its own terminal.
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		return ""
	}
	key := ledgerKey(ctx, relPath)
	// The file as it is NOW. A path that cannot be read fabricates nothing:
	// no source, no read state, no recovery.
	data, ok := readLedgerBytes(key)
	if !ok {
		return ""
	}
	diskHash := hashBytes(data)
	// The body has to have been in front of the model through the real read
	// path. WasFileRead is weaker -- outline_file satisfies it while showing
	// signatures only -- so a file whose contents were never displayed is not
	// eligible.
	if !ctx.WasBodySeen(key) {
		return ""
	}
	hazardous := workspaceHazardous(ctx)

	ctx.LedgerMu.Lock()
	d := ctx.Ledger[key]
	var status ValidationStatus
	var kind ValidationKind
	var detail string
	var sessionWritten, restorable bool
	if d != nil {
		kind, status = d.CurrentValidation()
		detail = d.ValidationDetail
		// Generation > 0 is the ledger's own record that this session wrote
		// the file, and it is canonical -- unlike SessionWrites, which is
		// keyed on the path as the model spelled it.
		sessionWritten = d.Generation > 0 && d.CurrentHash == diskHash
		restorable, _ = checkpointRestorable(d, diskHash, hazardous)
	}
	ctx.LedgerMu.Unlock()

	switch {
	case !sessionWritten:
		return ""
	case status != ValidationFailed:
		// unknown, not_run, not_applicable, passed, and every verdict that
		// describes bytes no longer on disk, all end here.
		return ""
	case restorable:
		// A path with an eligible safer checkpoint is Phase 3B's, not this.
		return ""
	}

	genKey := key + "\x00" + diskHash
	if st.noopEditRepeats == nil {
		st.noopEditRepeats = map[string]int{}
	}
	st.noopEditRepeats[genKey]++
	if st.noopEditRepeats[genKey] < c3RecoveryRepeat || st.brokenArtifactRecovered[genKey] {
		return ""
	}
	// Context nobody has time to act on is worse than stopping.
	if ctx.Ctx != nil {
		if deadline, ok := ctx.Ctx.Deadline(); ok && time.Until(deadline) < fencedRecoveryFloor {
			log.Printf("[agent] skipping the no-op-edit recovery for %s — %v of budget left",
				relPath, time.Until(deadline).Round(time.Second))
			return ""
		}
	}
	source, truncated, err := boundedCurrentSource(ctx, relPath)
	if err != nil {
		log.Printf("[agent] no-op-edit recovery for %s could not read it: %v", relPath, err)
		return ""
	}
	if st.brokenArtifactRecovered == nil {
		st.brokenArtifactRecovered = map[string]bool{}
	}
	st.brokenArtifactRecovered[genKey] = true
	if detail == "" {
		detail = string(kind) + " check failed"
	}
	log.Printf("[agent] no-op-edit recovery for %s: %s/%s on the current bytes", relPath, kind, status)

	var sb strings.Builder
	fmt.Fprintf(&sb, "That edit changes nothing: old_str and new_str are the same string, "+
		"so %s would be left exactly as it is. It is still broken.\n\n", relPath)
	fmt.Fprintf(&sb, "%s currently contains", relPath)
	if truncated {
		fmt.Fprintf(&sb, " (first %d lines)", fencedRecoveryMaxLines)
	}
	sb.WriteString(":\n\n")
	sb.WriteString(source)
	fmt.Fprintf(&sb, "\n\nThe %s check on exactly these bytes failed: %s\n\n", kind, detail)
	sb.WriteString("Send an edit whose new_str actually differs from old_str and fixes that. " +
		"Nothing has been changed for you, and nothing will be until you do.")
	return sb.String()
}

// clearBrokenArtifactState releases a path's C3 state once a materially
// different mutation on that same path reaches its normal result.
//
// Keyed, so an unrelated success elsewhere does not clear it, and narrow, so
// another no-op does not either -- a no-op is the thing being recovered from.
// A read does not clear it, and neither does a new turn. The hash is the other
// release: different bytes are a different evidence generation and get their
// own key.
func clearBrokenArtifactState(ctx *AgentContext, st *runState, name string, args json.RawMessage) {
	if st == nil || len(st.noopEditRepeats)+len(st.brokenArtifactRecovered) == 0 {
		return
	}
	if noopEditIntent(name, args) != "" {
		return
	}
	targets := mutationIntentTargets(ctx, name, args)
	if len(targets) == 0 {
		return
	}
	prefix := ledgerKey(ctx, targets[0].Rel) + "\x00"
	for k := range st.noopEditRepeats {
		if strings.HasPrefix(k, prefix) {
			delete(st.noopEditRepeats, k)
		}
	}
	for k := range st.brokenArtifactRecovered {
		if strings.HasPrefix(k, prefix) {
			delete(st.brokenArtifactRecovered, k)
		}
	}
}

// markWarnedRun records or discharges a path's pending warned landing.
//
// One invariant, in one place: a warned landing puts the path in the set, and
// anything else takes it out. Storing "not warned" as a value is what let a
// key-only reader announce a parse warning over a file that parses.
func (s *runState) markWarnedRun(ctx *AgentContext, path string, warned bool) {
	if !warned {
		s.clearWarnedRun(ctx, path)
		return
	}
	if s.pendingWarnedRun == nil {
		s.pendingWarnedRun = map[string]bool{}
	}
	s.pendingWarnedRun[path] = true
}

// clearWarnedRun drops every spelling of the same file.
//
// The set is keyed on the path as the model sent it, because the gate quotes
// it back ("Run it first -- `python3 solve.py`") and executionAttempt matches
// the command against it. So the identity used to DISCHARGE is the ledger's
// canonical one, which is the identity the rest of the run already uses: a
// clean rewrite of ./solve.py has to retire solve.py's warning, or the mark
// outlives the bytes it describes.
func (s *runState) clearWarnedRun(ctx *AgentContext, path string) {
	if len(s.pendingWarnedRun) == 0 {
		return
	}
	key := ledgerKey(ctx, path)
	for p := range s.pendingWarnedRun {
		if ledgerKey(ctx, p) == key {
			delete(s.pendingWarnedRun, p)
		}
	}
}

// currentValidationDetail reports what the ledger knows about the bytes that
// are there NOW, and says nothing when the verdict describes older bytes.
func currentValidationDetail(ctx *AgentContext, relPath string) string {
	key := ledgerKey(ctx, relPath)
	ctx.LedgerMu.Lock()
	d := ctx.Ledger[key]
	var kind ValidationKind
	var status ValidationStatus
	var detail string
	if d != nil {
		kind, status = d.CurrentValidation()
		detail = d.ValidationDetail
	}
	ctx.LedgerMu.Unlock()
	if status != ValidationFailed {
		return ""
	}
	if detail == "" {
		return string(kind) + " check failed"
	}
	return detail
}

// --- The exhausted fenced channel -------------------------------------------
//
// The allowance stops the generations; it does not tell the model anything it
// can act on. In the frozen run debounce2 asked for the same resolution 147
// times and was refused 144 of them with the same sentence, because the only
// thing the refusal could say was that the channel had failed.
//
// This is the earlier branch than the C5 run-first state and stays separate
// from it: no warned artifact is required, and the trigger is the allowance
// being spent rather than a pending demand to run something.
//
// Offered once per canonical path. It reads and never writes, runs no command,
// forces no tool, and starts no generation.
func fencedChannelRecovery(ctx *AgentContext, st *runState, relPath string) string {
	if ctx == nil || st == nil {
		return ""
	}
	// Only when the channel is genuinely spent for THIS path.
	if !fencedBudgetExhausted(ctx, relPath) {
		return ""
	}
	// A run that is already ending owns its own terminal.
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		return ""
	}
	key := fencedKey(ctx, relPath)
	if st.fencedChannelClosed[key] {
		return ""
	}
	// Context nobody has time to act on is worse than stopping.
	if ctx.Ctx != nil {
		if deadline, ok := ctx.Ctx.Deadline(); ok && time.Until(deadline) < fencedRecoveryFloor {
			log.Printf("[agent] skipping the fenced-channel recovery for %s — %v of budget left",
				relPath, time.Until(deadline).Round(time.Second))
			return ""
		}
	}
	if st.fencedChannelClosed == nil {
		st.fencedChannelClosed = map[string]bool{}
	}
	st.fencedChannelClosed[key] = true
	log.Printf("[agent] fenced channel spent for %s — offering the alternatives once", relPath)

	var sb strings.Builder
	fmt.Fprintf(&sb, "The fenced-content channel for %s is used up in this session: "+
		"the sub-call was asked for the file and did not return one, and it will not be "+
		"asked again for this path. Sending \"content\": \"@fenced\" for %s cannot "+
		"succeed now, however many times it is repeated.\n\n", relPath, relPath)

	if source, truncated, err := boundedCurrentSource(ctx, relPath); err == nil {
		fmt.Fprintf(&sb, "%s currently contains", relPath)
		if truncated {
			fmt.Fprintf(&sb, " (first %d lines)", fencedRecoveryMaxLines)
		}
		sb.WriteString(":\n\n")
		sb.WriteString(source)
		sb.WriteString("\n\n")
	} else {
		fmt.Fprintf(&sb, "%s is not on disk yet, so there is nothing to show you "+
			"of it.\n\n", relPath)
	}
	if detail := currentValidationDetail(ctx, relPath); detail != "" {
		fmt.Fprintf(&sb, "The last thing checked about those exact bytes: %s\n\n", detail)
	}
	fmt.Fprintf(&sb, "What still works for %s: send write_file with the complete file "+
		"INLINE in the content field; make a targeted change with edit_file or "+
		"replace_lines; read it with read_file; or run it with run_command and read the "+
		"real error. Pick one of those.", relPath)
	return sb.String()
}

// finalizeCompletion is the one decision both exits make, in one order.
//
// The defect it closes: the status was decided from the deliverable evidence
// alone, and the predicates that prove an unmet request were consulted twelve
// lines later, for the summary only. A run could therefore report
// completed / no_file_obligation while its own summary said "Nothing was
// written — no file was created or changed in this run." Measured on a
// four-target run that wrote nothing, and on a deny-listed write that was
// correctly refused: neither left a ledger entry, so neither had an obligation
// to fail, and both were called success.
//
// Nothing new is inferred. wantsStateChange, the action demand and the
// verification demand already exist and are already trusted enough to rewrite
// the user's summary; this lets the machine-readable half read the same
// evidence. No prose is parsed, no obligation is invented for a path the
// session never owned, and no more-specific existing failure is replaced.
//
// The order is fixed and shared with the summary composition below it:
//
//  1. a deliverable failure keeps its own, more specific reason;
//  2. otherwise, work was demanded on disk and none landed;
//  3. otherwise, verification was demanded and none passed;
//  4. otherwise the completion stands.
//
// completedReason lets an exit name its own reason for a genuine completion
// (the text exit says text_reply); empty keeps the deliverable evidence's.
func finalizeCompletion(ctx *AgentContext, st *runState, userMessage, completedReason string) (TerminalStatus, string) {
	// Settle what can be settled FIRST, so the evidence the rest of this
	// function reads is about a workspace nothing is still writing to.
	liveJobs := settleBackgroundHazard(ctx)

	ok, why := terminalCompletionAllowed(ctx, st.expectedOutputs)
	if !ok {
		return TerminalIncomplete, why
	}
	if st.actionDemandedAndUnmet(ctx, userMessage) {
		return TerminalIncomplete, "action_demanded_unmet"
	}
	if st.verificationDemandedAndUnmet() {
		return TerminalIncomplete, "verification_demanded_unmet"
	}
	// Something may still be writing. A hash taken now describes an instant,
	// not a result, and nothing here can tell a quiet process from a finished
	// one -- only a confirmed exit can.
	if len(liveJobs) > 0 || workspaceHazardous(ctx) {
		return TerminalIncomplete, "background_work_unresolved"
	}
	// Settle first, from the workspace as it is now. A debt can become
	// resolved without another tool call -- a deletion of a path that was
	// already absent is owed and discharged by the same fact -- and settling
	// only after an execution meant those never retired. This is the existing
	// structural rule (confirmed absence, demonstrated bytes, a completed
	// move), evaluated at the moment the decision is made.
	settleMutationDebt(ctx, st)
	// Work the model asked for, that the system permitted, and that never
	// reached a state anything could check. A success elsewhere does not
	// settle it.
	if hasUnresolvedDebt(st) {
		return TerminalIncomplete, "unresolved_mutation_debt"
	}
	if completedReason != "" {
		return TerminalCompleted, completedReason
	}
	return TerminalCompleted, why
}

// --- Unresolved mutation debt -----------------------------------------------
//
// Completion had three inputs: the user's named outputs, the deliverable
// ledger, and one session-wide madeProductiveChange bool. A valid mutation the
// model asked for and never landed left no trace in any of them, so a success
// on an unrelated path retired it -- measured as completed /
// deliverables_demonstrated over a session that had failed to write a.py and
// succeeded on b.py.
//
// The ledger is not the place for this. It records what the session OWNS on
// disk, and an intent that never landed owns nothing; putting an unowned path
// in it would break the invariant Phase 3A was built on. Debt is a separate,
// bounded, session-local record of what is still owed.

// maxTrackedMutationDebt bounds the map. A session that somehow exceeds it
// stops naming individual paths and never stops reporting that work is
// unresolved -- the failure direction that cannot manufacture a completion.
const maxTrackedMutationDebt = 64

// debtKind is what would have to be demonstrated for the debt to clear.
type debtKind string

const (
	debtContent debtKind = "content" // bytes must exist and validate
	debtDelete  debtKind = "delete"  // the path must be demonstrably absent
	debtMove    debtKind = "move"    // source absent AND destination validated
)

// mutationDebtEntry holds only what a terminal or a recovery needs to say.
// No file contents: the bytes live on disk and in the ledger's bounded
// checkpoint, never here.
type mutationDebtEntry struct {
	Rel  string // workspace-relative, for plain-language disclosure
	Kind debtKind
	Dest string // canonical destination, move only
	Gen  int    // the debt generation this entry was opened in
}

// mutationIntentTargets returns the canonical paths a path-targeted mutator is
// asking to change, or nil when the call is not one, is malformed, names a
// blank path, or is deny-listed. run_command and run_background are excluded
// on purpose: their effects are unobserved by construction and a single path
// cannot represent them.
func mutationIntentTargets(ctx *AgentContext, name string, args json.RawMessage) []*mutationDebtEntry {
	switch name {
	case "write_file", "edit_file", "structural_edit", "insert_after", "replace_lines",
		"delete_file", "move_file":
	default:
		return nil
	}
	// The same authoritative refusals the tool applies. An attempt the system
	// forbids is not work the session owes; unmet-action evidence covers it.
	if denied, _ := shouldDenyToolCall(name, args); denied {
		return nil
	}
	if reason := validateToolWorkspacePaths(name, args, ctx); reason != "" {
		return nil
	}
	rel := func(field string) string {
		var m map[string]json.RawMessage
		if json.Unmarshal(args, &m) != nil {
			return ""
		}
		var s string
		if raw, ok := m[field]; !ok || json.Unmarshal(raw, &s) != nil {
			return ""
		}
		return strings.TrimSpace(s)
	}
	if name == "move_file" {
		src, dst := rel("source"), rel("destination")
		if src == "" || dst == "" {
			return nil
		}
		return []*mutationDebtEntry{{Rel: src, Kind: debtMove, Dest: ledgerKey(ctx, dst)}}
	}
	p := rel("path")
	if p == "" {
		return nil
	}
	kind := debtContent
	if name == "delete_file" {
		kind = debtDelete
	}
	return []*mutationDebtEntry{{Rel: p, Kind: kind}}
}

// noteMutationIntent opens debt for a permitted in-workspace mutation. It runs
// BEFORE dispatch, which is the whole point: a fenced resolution that fails
// never reaches executeToolCall and used to leave no trace anywhere.
func noteMutationIntent(ctx *AgentContext, st *runState, name string, args json.RawMessage) {
	for _, e := range mutationIntentTargets(ctx, name, args) {
		key := ledgerKey(ctx, e.Rel)
		if st.mutationDebt == nil {
			st.mutationDebt = map[string]*mutationDebtEntry{}
		}
		if prev, exists := st.mutationDebt[key]; exists {
			// The one structured way a mistaken path is retired: the model
			// explicitly asks for the same path to be REMOVED. That converts
			// what it owes from "produce this" to "prove it is gone", and the
			// proof is a confirmed absence, not a claim.
			if prev.Kind == debtContent && e.Kind == debtDelete {
				e.Gen = prev.Gen
				st.mutationDebt[key] = e
				log.Printf("[agent] %s: explicit removal requested — it now has to be shown gone", e.Rel)
			}
			continue // one debt per canonical path, whatever the spelling
		}
		if len(st.mutationDebt) >= maxTrackedMutationDebt {
			// Fail closed: stop naming, keep blocking.
			if !st.debtOverflow {
				log.Printf("[agent] mutation-debt ceiling reached (%d paths) — further "+
					"unresolved work is reported without naming the paths", maxTrackedMutationDebt)
			}
			st.debtOverflow = true
			return
		}
		if st.debtRecoveryOffered >= st.debtGeneration {
			// Work that went unresolved AFTER the model was already given its
			// chance is a new situation, and earns one more -- bounded below.
			st.debtGeneration++
		}
		e.Gen = st.debtGeneration
		st.mutationDebt[key] = e
		log.Printf("[agent] tracking unresolved %s work on %s", e.Kind, e.Rel)
	}
}

// settleMutationDebt clears what the LEDGER can prove, and nothing else. It is
// driven by observed state rather than by a tool reporting success, so a
// refusal, an unknown verdict, stale evidence, a read, or a success on another
// path all leave the debt standing.
func settleMutationDebt(ctx *AgentContext, st *runState) {
	if st == nil || len(st.mutationDebt) == 0 {
		return
	}
	for key, e := range st.mutationDebt {
		if debtResolved(ctx, key, e) {
			delete(st.mutationDebt, key)
			log.Printf("[agent] unresolved %s work on %s is now demonstrated", e.Kind, e.Rel)
		}
	}
}

// validationSettles reports whether a path's CURRENT validation is good enough
// to call the work demonstrated: a pass, or a genuine not_applicable from a
// producer that deliberately checked nothing because nothing applies.
func validationSettles(d *DeliverableState) bool {
	if d == nil || d.Tombstoned || d.CurrentHash == "" {
		return false
	}
	kind, status := d.CurrentValidation() // fails closed on a hash mismatch
	switch status {
	case ValidationPassed:
		return true
	case ValidationNotApplicable:
		return kind == ValidationKindNone
	}
	return false
}

func debtResolved(ctx *AgentContext, key string, e *mutationDebtEntry) bool {
	ctx.LedgerMu.Lock()
	d := ctx.Ledger[key]
	var tombstoned bool
	var reason string
	if d != nil {
		tombstoned, reason = d.Tombstoned, d.TombstoneReason
	}
	ctx.LedgerMu.Unlock()

	switch e.Kind {
	case debtContent:
		return validationSettles(d)
	case debtDelete:
		// The entry only exists because the model explicitly asked for THIS
		// path to be removed, so the intent is already on the record. What
		// settles it is the absence, confirmed against disk right now.
		//
		// The tombstone is not required, and requiring it made the retirement
		// route unusable for the case it exists for: a path that never landed
		// cannot be deleted -- delete_file fails with "file not found" and
		// writes no tombstone -- so a model abandoning work it never managed
		// to produce could never say so. A path that IS still there fails this
		// check whatever the delete reported.
		_, err := os.Stat(key)
		return os.IsNotExist(err)
	case debtMove:
		if !tombstoned || !strings.HasPrefix(reason, "moved:") {
			return false
		}
		if _, err := os.Stat(key); !os.IsNotExist(err) {
			return false // the source is still there
		}
		// The destination is judged by the same contract every other
		// deliverable is judged by, read from disk. The ledger's own verdict
		// cannot be used here: move_file deliberately records the destination
		// as unknown, because a syntax pass earned under the old name says
		// nothing about this path -- so asking validationSettles for a verdict
		// nothing ever writes made this debt unretirable, and a demonstrated
		// rename could never finish.
		if e.Dest == "" {
			return false
		}
		return deliverablesDemonstrablyValid(ctx, []string{e.Dest})
	}
	return false
}

// unresolvedDebtPaths lists what is still owed, in a stable order, bounded for
// disclosure. The second return says whether more exist than are named.
func unresolvedDebtPaths(st *runState, limit int) ([]string, bool) {
	if st == nil {
		return nil, false
	}
	var out []string
	for _, e := range st.mutationDebt {
		out = append(out, e.Rel)
	}
	sort.Strings(out)
	more := st.debtOverflow
	if len(out) > limit {
		out, more = out[:limit], true
	}
	return out, more
}

func hasUnresolvedDebt(st *runState) bool {
	return st != nil && (len(st.mutationDebt) > 0 || st.debtOverflow)
}

// unresolvedDebtSummary is the plain-language disclosure. It names paths and
// says what would settle them, in the words a user would use.
func unresolvedDebtSummary(st *runState) string {
	paths, more := unresolvedDebtPaths(st, 5)
	var sb strings.Builder
	sb.WriteString("Stopped: work you asked for was started and never finished")
	switch {
	case len(paths) == 0:
		sb.WriteString(".")
	case len(paths) == 1:
		fmt.Fprintf(&sb, ": %s was never written in a state this run could check.", paths[0])
	default:
		fmt.Fprintf(&sb, ": %s were never written in a state this run could check.",
			strings.Join(paths, ", "))
	}
	if more {
		sb.WriteString(" Other files are in the same state.")
	}
	sb.WriteString(" This run did not confirm the task was complete.")
	return sb.String()
}

// maxDebtRecoveries bounds the whole session. New unresolved work opens a new
// generation and earns another offer, but never without end.
const maxDebtRecoveries = 2

// offerDebtRecovery is the one chance to settle before the terminal. Returning
// incomplete is honest and does nothing for the user; this says exactly what is
// outstanding and exactly what would settle it, once per debt generation.
//
// It changes nothing itself: no file is written, deleted, moved or run, and no
// tool is forced. The model chooses through its normal tools, under their
// normal guards.
func offerDebtRecovery(ctx *AgentContext, st *runState) string {
	if st == nil || !hasUnresolvedDebt(st) {
		return ""
	}
	if st.debtRecoveryOffered >= st.debtGeneration || st.debtRecoveryCount >= maxDebtRecoveries {
		return ""
	}
	// Context nobody has budget to act on is worse than stopping.
	if ctx.Ctx != nil {
		if deadline, ok := ctx.Ctx.Deadline(); ok && time.Until(deadline) < fencedRecoveryFloor {
			log.Printf("[agent] skipping the unresolved-work recovery — %v of budget left",
				time.Until(deadline).Round(time.Second))
			return ""
		}
	}
	st.debtRecoveryOffered = st.debtGeneration
	st.debtRecoveryCount++

	paths, more := unresolvedDebtPaths(st, 5)
	var sb strings.Builder
	sb.WriteString("Before finishing: work you started never reached a state this run " +
		"could check, so it cannot be reported as done.\n\n")
	for _, p := range paths {
		var kind debtKind
		for _, e := range st.mutationDebt {
			if e.Rel == p {
				kind = e.Kind
			}
		}
		switch kind {
		case debtDelete:
			fmt.Fprintf(&sb, "  %s — you asked for it to be removed; it is still there.\n", p)
		case debtMove:
			fmt.Fprintf(&sb, "  %s — the move is unfinished.\n", p)
		default:
			fmt.Fprintf(&sb, "  %s — never written in a form that could be checked.\n", p)
		}
	}
	if more {
		sb.WriteString("  (other files are in the same state)\n")
	}
	sb.WriteString("\nFor each one, either finish it — write the complete file and make sure " +
		"it is valid, or make the change with edit_file — or, if you decided that file " +
		"should not exist after all, say so by calling delete_file on that exact path so " +
		"its absence can be confirmed. Saying you no longer need it is not enough. " +
		"Finishing another file does not settle this one.")
	return sb.String()
}

// --- Live workspace hazards at completion ------------------------------------
//
// run_background raises the workspace hazard and only a confirmed exit lowers
// it, but the completion decision never asked. A server started mid-run could
// keep rewriting a tracked deliverable while the run reported completed over a
// hash taken at one instant.
//
// This asks the sandbox what is actually true, without changing anything the
// model did not ask for: a job that has ALREADY exited is reaped and its
// hazard lowered, and a job that is still running is left alone and blocks.
// Nothing is killed to make a completion possible.

// settleBackgroundHazard returns the ids of this session's jobs that are still
// running. Exited jobs are reaped through the existing session-owned path, and
// every tracked deliverable is rehashed afterwards so a verdict about bytes a
// job changed on its way out cannot survive.
func settleBackgroundHazard(ctx *AgentContext) []string {
	if ctx == nil || len(ctx.BackgroundJobs) == 0 {
		return nil
	}
	ids := make([]string, 0, len(ctx.BackgroundJobs))
	for id := range ctx.BackgroundJobs {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	var live []string
	reaped := false
	for _, id := range ids {
		out, err := sandboxTailBackground(ctx, id, 1)
		if err != nil {
			// Unobservable is not the same as finished. The hazard stands.
			log.Printf("[agent] cannot establish whether background job %s has exited: %v", id, err)
			live = append(live, id)
			continue
		}
		if out.Running || out.ExitCode == nil {
			live = append(live, id)
			continue
		}
		// Already gone. Reaping is bookkeeping at this point, not a kill.
		if _, err := sandboxStopBackground(ctx, id); err != nil {
			log.Printf("[agent] could not reap the exited background job %s: %v", id, err)
			live = append(live, id)
			continue
		}
		delete(ctx.BackgroundJobs, id)
		clearWorkspaceHazard(ctx, id)
		reaped = true
		log.Printf("[agent] background job %s had already exited — reaped at completion", id)
	}
	if reaped {
		// A job can change a file on its way out. Rehash every tracked path:
		// unchanged files keep their verdicts, changed ones lose them.
		invalidateTrackedValidation(ctx)
	}
	return live
}

// workspaceRefusalPath names the target a boundary refusal was about, for the
// path-aware breaker. It uses the same field map the validator itself keys on,
// so the two cannot disagree about which argument is the path, and it
// canonicalises the same way every other failure identity does -- a refusal
// spelled ./app.py is the same refusal as app.py.
//
// A workspace root that cannot be opened resolves nothing, so the canonical
// form falls back to the raw spelling rather than inventing a path inside a
// directory that does not exist.
func workspaceRefusalPath(ctx *AgentContext, name string, args json.RawMessage) string {
	var fields map[string]json.RawMessage
	if json.Unmarshal(args, &fields) != nil {
		return name
	}
	for _, key := range workspacePathFields[name] {
		raw, ok := fields[key]
		if !ok {
			continue
		}
		var value string
		if json.Unmarshal(raw, &value) != nil || strings.TrimSpace(value) == "" {
			continue
		}
		return filepath.Clean(strings.TrimSpace(value))
	}
	// No usable path field: the tool itself is the identity, so repeated
	// refusals of the same tool still converge on one failure target.
	return name
}
