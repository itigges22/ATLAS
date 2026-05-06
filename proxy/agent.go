package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// activeSessions tracks in-flight /v1/agent turns by session_id so
// /cancel can abort them. Map value is the context.CancelFunc returned
// from the per-request context.WithCancel wrapper. PC-062 step 5.
//
// Defense-in-depth: cancellation also flows naturally through TCP
// disconnect (handleAgent already binds ctx to r.Context()), but a
// reverse proxy may buffer the disconnect. /cancel gives the TUI a
// reliable, explicit kill switch.
var activeSessions sync.Map

// ---------------------------------------------------------------------------
// Agent loop — iterative tool-calling loop between model and executors
// ---------------------------------------------------------------------------

// runAgentLoop runs the agent loop for a single user request.
// The model emits tool calls (constrained by grammar), the proxy executes them,
// and returns results. Continues until the model emits "done" or max turns hit.
func runAgentLoop(ctx *AgentContext, userMessage string) error {
	// Emit a stage_start envelope so the TUI's pipeline pane shows
	// the agent is working. Mirrors the typed-event broker (PC-061).
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
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtStageEnd,
			Stage:      "agent",
			DurationMS: dur,
			Payload: map[string]interface{}{
				"success":      true,
				"total_tokens": ctx.TotalTokens,
			},
		})
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtDone,
			Stage:      "agent",
			DurationMS: dur,
			Payload: map[string]interface{}{
				"success":           true,
				"total_duration_ms": dur,
				"total_tokens":      ctx.TotalTokens,
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
	ctx.Messages = make([]AgentMessage, 0, 2+len(ctx.PriorHistory))
	ctx.Messages = append(ctx.Messages, AgentMessage{Role: "system", Content: systemPrompt})
	ctx.Messages = append(ctx.Messages, ctx.PriorHistory...)
	ctx.Messages = append(ctx.Messages, AgentMessage{Role: "user", Content: userMessage})

	// PC-045: Per-session cache scope. llama.cpp's KV slot persists between
	// requests by default — that's PC-035's keep-warm behavior. But the slot
	// also persists *across user sessions*, so context from a previous
	// session's conversation can bias the next session (the
	// `show_greeting.py` hallucination from the 2026-04-30 snake test was
	// likely an example). Erase slot 0 at the start of each agent loop call.
	// llama.cpp re-encodes the system prompt from scratch (~1-2s on a
	// warm GPU); the per-turn cache benefit within the session is preserved.
	// Disable with ATLAS_FRESH_SLOT_PER_SESSION=0.
	if envOr("ATLAS_FRESH_SLOT_PER_SESSION", "1") != "0" {
		eraseLlamaSlot(ctx)
	}

	// Get the constrained output schema
	schemaJSON := buildToolCallSchemaJSON()

	consecutiveReads := 0       // Track consecutive read-only calls
	consecutiveErrors := 0      // Track consecutive tool failures to break error loops
	madeProductiveChange := false // Set when a write/edit/delete succeeds in this run.
	// Used to soften the consecutiveErrors exit: post-write run_command failures
	// are usually verification noise, not "stuck loop" — see PC-025 Sub-finding B.
	verifiedThisLoop := false // Set when a verification command (pytest, curl,
	// python script, go test, ...) completes successfully in any turn of this
	// run. Used by the fix-intent gate before `done` is allowed to pass.
	// One successful verification per loop is enough — the model can iterate
	// inside the loop without re-verifying every turn.

	// Whether the user prompt is a repair/fix request. Computed once because
	// the user message doesn't change mid-loop. Drives the verification gate.
	userWantsVerification := isFixIntentMessage(userMessage)

	for turn := 0; turn < ctx.MaxTurns; turn++ {
		// Bail out fast if the upstream request was cancelled (the client closed the
		// connection, user hit Ctrl-C, terminal exited). Without this check the
		// loop would keep grinding LLM calls and tool work for a client that's
		// already gone, burning GPU. See ISSUES.md PC-036.
		if ctx.Ctx != nil {
			select {
			case <-ctx.Ctx.Done():
				log.Printf("[agent] cancelled at turn %d: %v", turn, ctx.Ctx.Err())
				return ctx.Ctx.Err()
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
		trimmed := false
		if len(ctx.Messages) > 12 {
			ctx.Messages = trimMessages(ctx.Messages, 8)
			trimmed = true
			log.Printf("[agent] trimmed conversation to %d messages", len(ctx.Messages))
		}

		// Per-turn streaming visibility: announce the start of the turn,
		// then the LLM call boundaries. Without these the TUI sees a 10-30s
		// gap between tool_result and the next tool_call while the model
		// is generating — looks like a hang. PC-062 follow-up.
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
		response, tokens, err := callLLMConstrained(ctx, schemaJSON)
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
			return fmt.Errorf("LLM call failed on turn %d: %w", turn, err)
		}
		ctx.TotalTokens += tokens
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
			log.Printf("[agent] parse error: %v | raw_len=%d | raw: %q", parseErr, len(response), truncateStr(response, 500))
			ctx.Stream("error", map[string]string{
				"error": "failed to parse model response",
			})
			// Targeted feedback — generic "your response wasn't JSON"
			// led to the May 2026 user-session bug where the model
			// retried the same 1100-char edit_file with a giant old_str
			// 5 times in a row. The response was being truncated at the
			// llama-server token cap; the model couldn't see that and
			// kept emitting the same too-big payload. Detect the
			// truncation shape and tell the model explicitly.
			feedback := classifyParseFailure(response)
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
				ctx.Stream("done", map[string]string{
					"summary": "Stopped after 3 unparseable responses — the model's tool calls keep getting truncated. Try a more targeted request (e.g. 'edit just the @app.route(\"/product\") handler in app.py') so the response stays under the token cap.",
				})
				return nil
			}
			continue
		}

		// Log the args truncated — enables diagnosing failures like
		// "all 3 tool calls returned Success=false" without having to add
		// breakpoints. See ISSUES.md PC-039 follow-up.
		log.Printf("[agent] turn=%d type=%s name=%s args=%s", turn, parsed.Type, parsed.Name, truncateStr(string(parsed.Args), 200))

		// PC-041: when a tool_call still has no args after liftMissingArgs,
		// log the raw model output so we can see exactly what shape was
		// emitted — helps catch new alt-shapes the lift logic missed.
		if parsed.Type == "tool_call" && (len(parsed.Args) == 0 || string(parsed.Args) == "null") {
			log.Printf("[agent] turn=%d EMPTY ARGS — raw model output: %q", turn, truncateStr(response, 500))
		}

		switch parsed.Type {
		case "done":
			// Verification gate. When the user asked to "fix" / "verify" /
			// "render" something and the model is declaring done without
			// having ever run a verification command (pytest, curl, python
			// app.py, go test, ...), bounce the done with a directive.
			// Reactive form of Roo Code's AttemptCompletionTool — we don't
			// require a structured verification field, just evidence in the
			// loop that the agent ran something that exits non-zero on
			// failure.
			if userWantsVerification && !verifiedThisLoop && !ctx.YoloMode {
				rejection := verificationRejectionMessage(userMessage)
				log.Printf("[agent] verification gate: bouncing done at turn %d (user prompt %q has fix-intent, no successful verification command this loop)",
					turn, truncateStr(userMessage, 60))
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role:    "assistant",
					Content: response,
				})
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role:       "tool",
					Content:    fmt.Sprintf(`{"success":false,"error":%q}`, rejection),
					ToolCallID: fmt.Sprintf("call_%d", turn),
					ToolName:   "verification_gate",
				})
				continue
			}

			// PC-197 — completion-claim verification. The model's done
			// summary often contains universals ("all routes work",
			// "fixed all bugs", "verified everything") that we can
			// structurally check against the workspace. The May 2026
			// flask run had the model claim "All routes are functioning
			// properly" while only 3 of 7 templates existed. Scan the
			// summary for claim language; if present, run cheap structural
			// checks (template references, view references, import
			// targets) and bounce if there's a concrete gap.
			//
			// Quiet pass when summary makes no universal claim (model
			// said something like "added /admin route" — no claim about
			// the rest of the app) or when there are no gaps. Bounce
			// only when both fire.
			if !ctx.YoloMode && claimsUniversal(parsed.Summary) {
				if gap := verifyCompletionClaims(ctx.WorkingDir, parsed.Summary); gap != "" {
					log.Printf("[agent] claim-check gate: bouncing done at turn %d — %s",
						turn, truncateStr(gap, 200))
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:    "assistant",
						Content: response,
					})
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:       "tool",
						Content:    fmt.Sprintf(`{"success":false,"error":%q}`, gap),
						ToolCallID: fmt.Sprintf("call_%d", turn),
						ToolName:   "claim_check",
					})
					continue
				}
			}
			ctx.Stream("done", map[string]string{"summary": parsed.Summary})
			return nil

		case "text":
			// `text` is the agent's user-facing chat answer. End the turn
			// here — the user gets one reply per message they send, and
			// can follow up to continue. Looping after text caused two
			// failures in earlier revisions:
			//   1. trailing role=assistant tripped llama-server's
			//      "prefill incompatible with enable_thinking" 400, and
			//   2. with a "continue" nudge, the model would rabbit-hole
			//      into nonsense tool_calls on conversational input
			//      ("hi" → list_directory → run_command → 3 fails → bail).
			// If the model wants to narrate before tool work, it should
			// emit tool_call directly with the narration in the args or
			// roll narration into the done.summary at the end.
			ctx.Stream("text", map[string]string{"content": parsed.Content})
			ctx.Stream("done", map[string]string{"summary": ""})
			return nil

		case "tool_call":
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

			// Check permissions
			if needsPermission(ctx, parsed.Name, parsed.Args) {
				if ctx.PermissionFn != nil && !ctx.PermissionFn(parsed.Name, parsed.Args) {
					// Permission denied
					ctx.Stream("permission_denied", map[string]string{
						"tool": parsed.Name,
					})
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:    "assistant",
						Content: response,
					})
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:       "tool",
						Content:    `{"success":false,"error":"permission denied by user"}`,
						ToolCallID: fmt.Sprintf("call_%d", turn),
						ToolName:   parsed.Name,
					})
					continue
				}
			}

			// Fix C: Detect truncated args BEFORE execution.
			// If the args JSON doesn't parse, don't attempt execution —
			// tell the model to use smaller edits instead.
			if parsed.Name == "write_file" || parsed.Name == "edit_file" || parsed.Name == "run_command" {
				var testParse map[string]interface{}
				if err := json.Unmarshal(parsed.Args, &testParse); err != nil {
					log.Printf("[agent] truncated args detected for %s at turn %d", parsed.Name, turn)
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:    "assistant",
						Content: response,
					})
					ctx.Messages = append(ctx.Messages, AgentMessage{
						Role:       "tool",
						Content:    `{"success":false,"error":"Your output was truncated — the content is too long for a single tool call. For existing files, use edit_file with small targeted changes (replace specific functions or sections). For new files, keep them under 100 lines per write_file call."}`,
						ToolCallID: fmt.Sprintf("call_%d", turn),
						ToolName:   parsed.Name,
					})
					consecutiveErrors++
					if consecutiveErrors >= 3 {
						ctx.Stream("done", map[string]string{"summary": "Stopped: content too large for tool calls. Try requesting smaller, targeted changes."})
						return nil
					}
					continue
				}
			}

			// Surgical-edit gate: reject write_file on existing files
			// outright. write_file is for *creating* files; edits to an
			// existing file must use edit_file with old_str/new_str.
			//
			// PC-159 Phase 0 originally only blocked near-rewrites
			// (>= 70% line overlap) or >100-line writes. That left a
			// hole: a *complete* rewrite of a 90-line template (low
			// overlap, under the size cap) would slip through and
			// destroy the original. Hardened to reject every write
			// against an existing path. Trivially-small files (<= 5
			// lines, e.g. a single-line config) are still allowed
			// because there's no edit-vs-rewrite distinction at that
			// size — anything below that is faster to overwrite than
			// to surgically edit.
			if parsed.Name == "write_file" {
				var wfInput WriteFileInput
				if json.Unmarshal(parsed.Args, &wfInput) == nil {
					existingPath := resolveAgentPath(ctx, wfInput.Path)
					if existing, err := os.ReadFile(existingPath); err == nil {
						existingLines := strings.Count(string(existing), "\n") + 1
						if existingLines > 5 {
							rejection := fmt.Sprintf(
								"File %s already exists (%d lines). write_file is for creating new files, not modifying existing ones. Use edit_file with old_str/new_str to make targeted changes — read the file first if you need to confirm the exact text to replace.",
								wfInput.Path, existingLines)
							log.Printf("[agent] rejecting write_file for existing %s (%d lines)", wfInput.Path, existingLines)
							ctx.Messages = append(ctx.Messages, AgentMessage{
								Role:    "assistant",
								Content: response,
							})
							ctx.Messages = append(ctx.Messages, AgentMessage{
								Role:       "tool",
								Content:    fmt.Sprintf(`{"success":false,"error":%q}`, rejection),
								ToolCallID: fmt.Sprintf("call_%d", turn),
								ToolName:   "write_file",
							})
							continue
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
			if parsed.Name == "run_command" && !ctx.YoloMode {
				var rc RunCommandInput
				if json.Unmarshal(parsed.Args, &rc) == nil {
					if rejection := validateShellCommand(rc.Command); rejection != "" {
						log.Printf("[agent] rejecting run_command %q: %s",
							truncateStr(rc.Command, 80), rejection)
						ctx.Messages = append(ctx.Messages, AgentMessage{
							Role:    "assistant",
							Content: response,
						})
						ctx.Messages = append(ctx.Messages, AgentMessage{
							Role:       "tool",
							Content:    fmt.Sprintf(`{"success":false,"error":%q}`, rejection),
							ToolCallID: fmt.Sprintf("call_%d", turn),
							ToolName:   "run_command",
						})
						continue
					}
				}
			}

			// Execute tool
			startTime := time.Now()
			result := executeToolCall(parsed.Name, parsed.Args, ctx)
			elapsed := time.Since(startTime)

			// On failure, log the error so it shows up in `docker compose
			// logs atlas-proxy` without having to attach a debugger.
			// PC-039 follow-up.
			if !result.Success {
				log.Printf("[agent] turn=%d tool=%s FAIL: %s", turn, parsed.Name, truncateStr(result.Error, 240))
			}

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

			// Force-stop after destructive operations that shouldn't have follow-up
			if result.Error == "__FORCE_DONE__" {
				result.Error = ""
				// Don't stream a follow-up message — the file deletion already
				// happened on disk and any trailing text would just be noise
				// for the TUI to render after a destructive op.
				return nil
			}

			// Track productive state changes — write/edit/delete that landed.
			// Used below to soften the error-loop exit when work was completed.
			if result.Success && (parsed.Name == "write_file" || parsed.Name == "edit_file" || parsed.Name == "delete_file") {
				madeProductiveChange = true
			}

			// Track verification — a successful run_command of a build /
			// test / probe / runner. Recon (ls, cat, grep) doesn't count.
			// Once any verification succeeds in this loop, the fix-intent
			// gate stops blocking `done`.
			if result.Success && parsed.Name == "run_command" {
				var rc RunCommandInput
				if json.Unmarshal(parsed.Args, &rc) == nil && isVerificationCommand(rc.Command) {
					verifiedThisLoop = true
					log.Printf("[agent] verification recorded: turn=%d cmd=%q",
						turn, truncateStr(rc.Command, 60))
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

			// Break error loops: if 3 tool calls fail in a row, stop. PC-025
			// Sub-finding B: when the agent has already written/edited a file
			// and is now failing on `run_command` (verification noise — no
			// TTY for curses, missing toolchain, etc.), a different exit
			// message is appropriate so the user isn't told "the file may
			// be too large to modify" when their file is, in fact, on disk.
			if !result.Success {
				consecutiveErrors++
				if consecutiveErrors >= 3 {
					log.Printf("[agent] breaking error loop: %d consecutive failures at turn %d (productive=%v)", consecutiveErrors, turn, madeProductiveChange)
					if madeProductiveChange {
						ctx.Stream("done", map[string]string{"summary": "Wrote your changes to disk; couldn't verify them automatically (the verification commands failed). Run them yourself to confirm — they're on disk."})
					} else {
						// Non-productive 3-error exit. The previous message
						// ("file may be too large") presumed a write/edit
						// context, but this branch fires for any 3 failures
						// — including discovery flailing (empty paths from
						// PC-039, missing files, bad regex). Be honest about
						// the failure mode and point at the tool errors so
						// the user can correct course.
						ctx.Stream("done", map[string]string{"summary": "Stopped after 3 tool failures with no successful changes. Common causes: the file you referenced isn't in the workspace, an empty path argument was passed, or a regex was malformed. Check the per-turn errors above, then try a more specific request (e.g. \"fix snake_game.py at line 95 — the curses bounds are wrong\")."})
					}
					return nil
				}
			} else {
				consecutiveErrors = 0
			}

			// Track consecutive read-only calls to detect exploration loops
			isReadOnly := parsed.Name == "read_file" || parsed.Name == "list_directory" || parsed.Name == "search_files"
			if isReadOnly {
				consecutiveReads++
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

			// PC-044: Trust V3-verified edits — strongly nudge toward done.
			// When V3 ran the edit through its sandbox/probe pipeline and
			// the result came back successful (V3Used && PhaseSolved
			// non-empty), the edit is build-verified. The 9B model otherwise
			// keeps grinding: re-reads the file, edits unrelated functions,
			// runs another V3 cycle (~110s each). Inject an explicit
			// "you're done unless you have a specific reason" message.
			if result.Success && result.V3Used && result.PhaseSolved != "" &&
				(parsed.Name == "write_file" || parsed.Name == "edit_file") {
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role: "user",
					Content: fmt.Sprintf(
						"V3 verified this edit passed its %s pipeline (%d candidates, score=%.2f). The fix is on disk and build-checked. If this resolves the user's original request, respond NOW with {\"type\":\"done\",\"summary\":\"<one sentence describing the fix>\"}. Only continue if you have a specific, concrete additional change to make — do not re-read the file to double-check, and do not edit unrelated code.",
						result.PhaseSolved, result.CandidatesTested, result.WinningScore,
					),
				})
				log.Printf("[agent] PC-044: V3-verified %s on %s — nudging toward done", parsed.Name, truncateStr(string(parsed.Args), 80))
			}

			// Exploration budget: after 4 consecutive read-only calls,
			// inject nudge. After 5, skip reads.
			// FUTURE (L6 reliability): The 9B model over-explores when adding
			// features to existing projects (~67% pass rate). Better prompting,
			// larger model, or V3-guided exploration would improve this.
			if consecutiveReads == 4 {
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role:    "user",
					Content: "You have full project context in the system prompt. Do not read more files. Emit a write_file or edit_file tool call now.",
				})
				log.Printf("[agent] exploration budget: warning at turn %d", turn)
			} else if consecutiveReads >= 5 {
				// Skip the read and return synthetic result
				ctx.Messages = append(ctx.Messages, AgentMessage{
					Role:    "user",
					Content: "Skipped — you already have this information in context. Write your changes now. Use write_file or edit_file.",
				})
				consecutiveReads = 2 // Keep at warning level, don't reset
				log.Printf("[agent] exploration budget: skipped read at turn %d", turn)
			}

		default:
			// Unknown type — grammar should prevent this
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "user",
				Content: fmt.Sprintf("Unknown response type '%s'. Use tool_call, text, or done.", parsed.Type),
			})
		}
	}

	ctx.Stream("error", map[string]string{
		"error": fmt.Sprintf("max turns (%d) exceeded for %s task", ctx.MaxTurns, ctx.Tier),
	})
	return fmt.Errorf("max turns exceeded (%d)", ctx.MaxTurns)
}

// ---------------------------------------------------------------------------
// LLM call with grammar constraint
// ---------------------------------------------------------------------------

// callLLMConstrained calls the LLM with json_schema or grammar constraint.
// Returns the raw response text and token count.
//
// PC-043: When the model emits zero tokens (raw_len=0) — usually after a
// tool result message under /nothink + json_object grammar — we retry
// inline once with a bumped temperature and a transient "continue"
// nudge appended to the messages. This avoids burning a full agent-loop
// turn (~30s + tokens) on the parse-error retry path. The nudge is
// scoped to the retry call only; ctx.Messages is not mutated.
func callLLMConstrained(ctx *AgentContext, schemaJSON string) (string, int, error) {
	content, tokens, err := callLLMOnce(ctx, ctx.Messages, 0.3)
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
	log.Printf("[agent] empty LLM response (PC-043), retrying with temp=0.7 + continuation nudge")
	nudged := append(append([]AgentMessage(nil), ctx.Messages...), AgentMessage{
		Role:    "user",
		Content: `Continue. Respond with one JSON object: {"type":"tool_call","name":"<tool>","args":{...}} for the next action, or {"type":"done","summary":"..."} if the task is complete. Do not emit empty content.`,
	})
	content2, tokens2, err := callLLMOnce(ctx, nudged, 0.7)
	if err != nil {
		// Return whatever we have from the original call; caller
		// handles empty via parse-error retry.
		return content, tokens, nil
	}
	return content2, tokens + tokens2, nil
}

// eraseLlamaSlot clears llama.cpp's KV slot 0 to give the next chat
// completion a fresh prefix. See PC-045. Errors are logged and
// swallowed — slot erase is a best-effort isolation step, not a
// correctness requirement.
func eraseLlamaSlot(ctx *AgentContext) {
	llamaURL := envOr("ATLAS_LLAMA_URL", ctx.InferenceURL)
	endpoint := llamaURL + "/slots/0?action=erase"

	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	req, err := http.NewRequestWithContext(reqCtx, "POST", endpoint, nil)
	if err != nil {
		log.Printf("[PC-045] erase slot: build request failed: %v", err)
		return
	}
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[PC-045] erase slot: request failed: %v (this is fine — slot is now stale, will be re-encoded on next call)", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("[PC-045] erase slot: status %d (continuing — first turn will re-encode prefix from scratch)", resp.StatusCode)
		return
	}
	log.Printf("[PC-045] erased llama slot 0 — fresh KV cache for this session")
}

// pollPromptProgress emits llm_prompt_progress events at 250ms cadence
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
var llmStreamClient = &http.Client{
	Transport: &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
		ResponseHeaderTimeout: 10 * time.Minute,
		IdleConnTimeout:       90 * time.Second,
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
	wireMessages := make([]map[string]string, len(messages))
	for i, msg := range messages {
		wireMessages[i] = map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		}
	}

	llamaURL := envOr("ATLAS_LLAMA_URL", ctx.InferenceURL)

	reqBody := map[string]interface{}{
		"model":       modelName,
		"messages":    wireMessages,
		"temperature": temperature,
		"max_tokens":  32768,
		"stream":      true,
		// Without include_usage, the final SSE chunk before [DONE] has no
		// usage block, so we can't report total_tokens to the TUI.
		"stream_options": map[string]bool{"include_usage": true},
		"response_format": map[string]string{
			"type": "json_object",
		},
		// Qwen3.5's chat template defaults enable_thinking=true, but the
		// agent loop relies on grammar-constrained JSON output — thinking
		// blocks would just bloat tokens and llama-server rejects the
		// combination outright once a trailing assistant message looks
		// like a "response prefill" (400: "Assistant response prefill is
		// incompatible with enable_thinking"). Disable explicitly.
		"enable_thinking": false,
	}
	body, _ := json.Marshal(reqBody)
	endpoint := llamaURL + "/v1/chat/completions"

	// Carry the agent's request context into the HTTP request so client
	// disconnects propagate down to llama-server (PC-036).
	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
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
		contentBuf     strings.Builder
		totalTokens    int
		firstTokenSent bool
	)

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
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				TotalTokens     int `json:"total_tokens"`
				PromptTokens    int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Choices {
			if c.Delta.Content == "" {
				continue
			}
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
		}
		if chunk.Usage != nil && chunk.Usage.TotalTokens > 0 {
			totalTokens = chunk.Usage.TotalTokens
		}
	}
	if err := scanner.Err(); err != nil {
		return contentBuf.String(), totalTokens,
			fmt.Errorf("read LLM stream: %w", err)
	}

	if contentBuf.Len() == 0 {
		// No deltas at all — usually a llama-server hiccup. Caller's
		// empty-response retry path (callLLMConstrained) will handle.
		return "", totalTokens, nil
	}
	return contentBuf.String(), totalTokens, nil
}


// ---------------------------------------------------------------------------
// Permission checking
// ---------------------------------------------------------------------------

// needsPermission returns true if the tool call requires user confirmation.
func needsPermission(ctx *AgentContext, toolName string, args json.RawMessage) bool {
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

	// In accept-edits mode, write_file and edit_file are auto-approved
	if ctx.PermissionMode == PermissionAcceptEdits {
		if toolName == "write_file" || toolName == "edit_file" {
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

	// /nothink suppresses Qwen3.5's <think> mode — critical for JSON output
	sb.WriteString("/nothink\nYou are ATLAS, a coding assistant that creates and modifies code by calling tools. ")
	sb.WriteString("You have access to the filesystem and can run commands to verify your work.\n")
	sb.WriteString("You MUST respond with ONLY a single valid JSON object, no other text.\n\n")

	// Pick-the-right-shape guidance — this is what keeps "hi" out of the
	// tool-call rabbit hole. Without it the model treats every input as a
	// task and starts read_file'ing random paths.
	sb.WriteString("## Choosing your response shape\n\n")
	sb.WriteString("- **Conversational input** (greetings, small talk, questions about you, status checks): emit `{\"type\":\"text\",\"content\":\"...\"}` — the turn ends after one text reply, and the user can follow up. Do NOT call tools to answer \"hi\" or \"what can you do\".\n")
	sb.WriteString("- **Coding tasks** (\"fix the bug\", \"add a feature\", \"refactor X\"): emit `{\"type\":\"tool_call\",...}` to make progress, repeat as needed, then emit `{\"type\":\"done\",\"summary\":\"...\"}` when finished.\n")
	sb.WriteString("- **Don't use `text` mid-task.** Roll narration into the done.summary at the end, or skip it entirely. Mid-task `text` ends the turn early.\n")
	sb.WriteString("- **When unsure** whether the user wants chat or work: ask in a single `text` reply. Don't speculatively start tool-calling.\n\n")

	// Tool descriptions
	sb.WriteString(buildToolDescriptions())

	// Rules
	sb.WriteString("## Rules\n\n")
	sb.WriteString("- Always read a file before editing it (use read_file then edit_file)\n")
	sb.WriteString("- MANDATORY: Use `edit_file` (targeted old_str/new_str) for any change to a file that already exists, no matter how small. `write_file` is ONLY for creating brand-new files. The agent layer rejects every `write_file` call against an existing file >5 lines — your call won't execute and you'll get a tool error directing you to edit_file. Don't re-emit a whole file to change a few lines.\n")
	sb.WriteString("  Example — to add a None check to one branch, use:\n")
	sb.WriteString("    edit_file {\"path\":\"src/foo.py\",\"old_str\":\"if x == 0:\\n        return None\",\"new_str\":\"if x is None or x == 0:\\n        return None\"}\n")
	sb.WriteString("  NOT write_file with the entire file's new contents.\n")
	sb.WriteString("- The `content` you put in write_file / edit_file goes verbatim onto disk. **No markdown fences. No prose preamble (\"Looking at the task...\", \"Here's the file:\"). No trailing explanation.** Just the raw file contents. The agent layer strips fenced wrappers before writing, but the right move is to never emit them in the first place.\n")
	sb.WriteString("- **Never use shell `rm`, `mv`, `cp`, or `find -delete` to mutate workspace files.** Use the dedicated tools — `edit_file` for changes, `write_file` for new files, `delete_file` for removal. Shell mutation bypasses the safety gates and will be rejected by the agent layer. `run_command` is for build / test / run / inspection only (python, pytest, npm, go, ls, cat, curl, etc.).\n")
	sb.WriteString("- Use run_command to verify your changes work (build, test, lint, curl). When the user says \"fix\" or \"isn't working\", verify before declaring done — start the server, hit the endpoint, run the test. \"Done\" without a verification step is a guess.\n")
	sb.WriteString("- For LONG-RUNNING commands (servers, watchers — anything that doesn't exit on its own): use `run_background` instead of `run_command`. The pattern is: `run_background(\"python app.py\")` → wait for it to bind → `run_command(\"curl -sf http://localhost:5000/\")` → `stop_background(job_id)`. NEVER use `timeout 5 ... || true` to fake-background a server — the server dies before the curl can hit it. Always `stop_background` when done; jobs left running waste slots.\n")
	sb.WriteString("- When creating a project from scratch: create config/build files FIRST, verify they work (e.g., npm install, cargo check), THEN create feature code\n")
	sb.WriteString("- Respond with {\"type\":\"done\",\"summary\":\"...\"} when the task is complete\n")
	sb.WriteString("- If a command fails, read the error output, fix the issue, and try again\n")
	sb.WriteString("- Do not guess at file contents — read first, then edit\n")
	sb.WriteString("- ALWAYS use relative file paths (e.g., 'app.py', 'src/main.rs'), NEVER absolute paths\n")
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
	// Replaces the Python-only venv hint from PC-190 with the
	// universal pattern from PC-191. Probe-first hints (PC-193) are
	// added per-toolchain when present.
	if tcs := detectProjectToolchains(ctx.WorkingDir); len(tcs) > 0 {
		sb.WriteString("## Toolchains\n\n")
		sb.WriteString("This project uses the following language toolchains. Use the listed runner for verification commands so they execute against the project's pinned deps:\n\n")
		for _, tc := range tcs {
			sb.WriteString(fmt.Sprintf("- **%s** (manifests: %s)\n", tc.Name, strings.Join(tc.Manifests, ", ")))
			sb.WriteString(fmt.Sprintf("  - Runner: `%s`\n", tc.Runner))
			if tc.InstallCommand != "" {
				sb.WriteString(fmt.Sprintf("  - Install deps: `%s` (only if probe says deps are missing)\n", tc.InstallCommand))
			}
			if tc.TestCommand != "" {
				sb.WriteString(fmt.Sprintf("  - Tests: `%s`\n", tc.TestCommand))
			}
			if probe := probeToolchainReady(ctx.WorkingDir, tc); probe != "" {
				sb.WriteString(fmt.Sprintf("  - Status: %s\n", probe))
			}
		}
		sb.WriteString("\n**Don't reinstall deps that are already working.** The probe above tells you whether the project is ready to run as-is. If yes, go straight to verification. If a needed dep is missing, install only that dep, not the world.\n\n")
	}

	// Execution target hint — surfaces whether run_command is going
	// to the sandbox container (default, isolated) or directly to the
	// host (PC-192, opt-in via ATLAS_VERIFY_IN=host). The model needs
	// to know because host-mode means commands see the user's actual
	// venv binaries, system tools, env vars, and running services.
	if ctx.VerifyOnHost {
		sb.WriteString("**Execution target: host machine.** `run_command` runs directly on the user's host (not the sandbox container). This means: commands see the host's installed tools, env vars, and running services (databases, etc.); paths resolve against the host filesystem; and verifications run against the same environment the user uses themselves. The shell-op safety gates (no rm/mv/cp/find -delete/bash -c) still apply.\n\n")
	}

	// Show which files are in the project (names only, not full content).
	// Full content is available via read_file if needed.
	// This avoids consuming context window with pre-injected file dumps.
	if len(ctx.FilesRead) > 0 {
		sb.WriteString("## Project Files Available\n")
		for path := range ctx.FilesRead {
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

	pinIdx := -1
	for i := len(msgs) - 1; i >= 1; i-- {
		if msgs[i].Role == "user" {
			pinIdx = i
			break
		}
	}

	tailStart := len(msgs) - keepLast
	out := make([]AgentMessage, 0, keepLast+2)
	out = append(out, msgs[0])
	if pinIdx >= 1 && pinIdx < tailStart {
		out = append(out, msgs[pinIdx])
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
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	type historyMsg struct {
		Role    string `json:"role"`    // "user" or "assistant"
		Content string `json:"content"`
	}
	var req struct {
		Message    string       `json:"message"`
		WorkingDir string       `json:"working_dir"`
		Mode       string       `json:"mode"`       // "default", "accept-edits", "yolo"
		SessionID  string       `json:"session_id"` // optional — required for /cancel
		History    []historyMsg `json:"history,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Message == "" {
		http.Error(w, "message is required", http.StatusBadRequest)
		return
	}

	// Path translation: the TUI sends its host cwd (e.g. /home/isaac/snake)
	// as working_dir, but the proxy runs in a container where that path
	// doesn't exist — only /workspace (the bind-mount target) does. The
	// startup wrapper (atlas/cli/repl.py:_align_workspace) already aligns
	// the bind mount to the user's cwd, so /workspace IS the user's cwd
	// from the proxy's perspective. Use ATLAS_WORKSPACE_DIR (set in
	// docker-compose.yml) as the canonical write target. The original
	// host path is kept on RealProjectDir for display / V3 metadata.
	hostDir := req.WorkingDir
	if hostDir == "" {
		hostDir = "."
	}
	workingDir := envOr("ATLAS_WORKSPACE_DIR", hostDir)

	// Classify tier from message
	tier := classifyAgentTier(req.Message)

	// Create agent context
	ctx := NewAgentContext(workingDir, tier)
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

	// PC-192: opt-in host execution for run_command. Per-project config
	// (.atlas/config.toml: [execution] target = "host") wins over the
	// global env var so users can flip behaviour without touching the
	// proxy environment. Either source can downgrade to "sandbox"
	// explicitly. Default stays sandbox.
	ctx.VerifyOnHost = resolveVerifyTarget(workingDir) == "host"

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
	// and llama-server's in-flight generation. See ISSUES.md PC-036.
	//
	// PC-062: also wrap in a cancellable context so POST /cancel can
	// abort even when the TCP disconnect is buffered upstream.
	reqCtx, cancel := context.WithCancel(r.Context())
	defer cancel()
	ctx.Ctx = reqCtx
	if req.SessionID != "" {
		activeSessions.Store(req.SessionID, cancel)
		defer activeSessions.Delete(req.SessionID)
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

	// Detect project (implemented in project.go)
	ctx.Project = detectProjectInfo(workingDir)

	// Set up SSE streaming
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
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
	// before getting any data. PC-062 follow-up.
	fmt.Fprintf(w, ": connected\n\n")
	flusher.Flush()

	ctx.StreamFn = func(eventType string, data interface{}) {
		event := SSEEvent{Type: eventType, Data: data}
		eventJSON, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", eventJSON)
		flusher.Flush()
	}

	// For yolo mode, auto-approve all permissions
	if ctx.YoloMode {
		ctx.PermissionFn = func(string, json.RawMessage) bool { return true }
	}

	// Run agent loop
	if err := runAgentLoop(ctx, req.Message); err != nil {
		log.Printf("[agent] error: %v", err)
	}

	// Send final done event
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// ---------------------------------------------------------------------------
// /cancel — abort an in-flight /v1/agent turn by session_id (PC-062 step 5)
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
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.SessionID == "" {
		http.Error(w, "session_id required", http.StatusBadRequest)
		return
	}
	v, ok := activeSessions.LoadAndDelete(req.SessionID)
	w.Header().Set("Content-Type", "application/json")
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]bool{"cancelled": false})
		return
	}
	cancel, ok := v.(context.CancelFunc)
	if !ok {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "bad session entry"})
		return
	}
	cancel()
	log.Printf("[agent] cancelled session %s via /cancel", req.SessionID)
	_ = json.NewEncoder(w).Encode(map[string]bool{"cancelled": true})
}

// classifyParseFailure produces a targeted feedback message based on
// what the model emitted. The model can't see why parsing failed, so
// a generic "respond in JSON" message lets it loop forever on the same
// pattern. We pattern-match on the raw response shape:
//
//   - starts with `{"type":"tool_call",...,"name":"<edit_file|write_file>",...}` and looks
//     truncated → it tried a too-big edit; tell it to shrink old_str/new_str
//   - non-JSON prose → standard "respond JSON only" reminder
//   - empty or whitespace → continuation nudge
//
// The bug this addresses: in May 2026 a user fix-intent prompt put the
// model in a loop emitting the same 1100-char edit_file with all 5
// flask routes embedded in old_str. Llama-server's response cap cut it
// mid-string, parse failed, we didn't tell the model why, it retried
// identically. classifyParseFailure breaks the cycle by naming the
// failure mode.
func classifyParseFailure(raw string) string {
	stripped := strings.TrimSpace(raw)
	if stripped == "" {
		return "Your response was empty. Respond with ONLY a single JSON object — {\"type\":\"tool_call\",...} or {\"type\":\"text\",\"content\":\"...\"} or {\"type\":\"done\",\"summary\":\"...\"}."
	}
	// Truncated tool_call detection: response starts with the tool-call
	// preamble but doesn't have a properly closed args object. We look
	// for the opening shape and the absence of a clean trailing `}}` —
	// if both, treat it as truncation.
	looksLikeToolCall := strings.HasPrefix(stripped, `{"type":"tool_call"`) ||
		strings.HasPrefix(stripped, `{ "type": "tool_call"`) ||
		strings.HasPrefix(stripped, `{"type": "tool_call"`)
	if looksLikeToolCall {
		hasEditOrWrite := strings.Contains(stripped, `"edit_file"`) ||
			strings.Contains(stripped, `"write_file"`)
		// Crude truncation heuristic — if the response doesn't end with
		// at least one closing brace it's almost certainly cut off
		// mid-args. (A complete tool_call ends `...}}`.)
		truncated := !strings.HasSuffix(stripped, "}}") &&
			!strings.HasSuffix(stripped, "}") &&
			!strings.HasSuffix(stripped, "]")
		if hasEditOrWrite && truncated {
			return "Your last tool call was TRUNCATED — the response hit the token cap mid-args. The fix is to shrink old_str/new_str: edit ONE function or block per call, not the whole file. If you need to change multiple routes/functions, do them in separate edit_file calls (one per turn). Common offenders: pasting all of app.py into old_str, embedding 5+ @app.route handlers in a single replacement. Respond now with a smaller edit_file targeting just the next change."
		}
		if truncated {
			return "Your tool call was truncated mid-args. Make a smaller call — keep `content`, `old_str`, and `new_str` short (under ~30 lines). Respond now with the corrected, smaller call."
		}
		return "Your tool_call JSON was malformed. Re-emit it as a single valid JSON object: {\"type\":\"tool_call\",\"name\":\"<tool>\",\"args\":{...}}. No prose, no markdown fences, no trailing commas."
	}
	return "Your response was not valid JSON. Respond with ONLY a JSON object, no other text. Example: {\"type\":\"tool_call\",\"name\":\"write_file\",\"args\":{\"path\":\"file.py\",\"content\":\"code\"}}"
}

// extractModelResponse extracts a ModelResponse from the LLM output,
// handling cases where the model adds text before/after the JSON or
// where the JSON is truncated.
func extractModelResponse(raw string) (ModelResponse, error) {
	raw = strings.TrimSpace(raw)

	// Try direct parse first
	var resp ModelResponse
	if err := json.Unmarshal([]byte(raw), &resp); err == nil {
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

	if end > start {
		jsonStr := raw[start:end]
		if err := json.Unmarshal([]byte(jsonStr), &resp); err == nil {
			liftMissingArgs(&resp, jsonStr)
			return resp, nil
		}
	}

	// JSON was truncated (max_tokens hit mid-content) — try to recover
	// If we can see it's a write_file call, extract what we have
	if strings.Contains(raw, `"write_file"`) && strings.Contains(raw, `"content"`) {
		return recoverTruncatedWriteFile(raw[start:])
	}

	return resp, fmt.Errorf("could not parse JSON from response")
}

// liftMissingArgs handles models that emit tool calls in shapes other than
// the prescribed {"type":"tool_call","name":"X","args":{...}} envelope.
//
// Common alternative shapes (PC-041, PC-050):
//   - OpenAI-style: {"type":"tool_call","name":"X","arguments":{...}}
//   - Anthropic-style: {"type":"tool_call","name":"X","parameters":{...}}
//   - Inlined: {"type":"tool_call","name":"X","path":"...","offset":0,...}
//   - Type-is-tool-name (PC-050): {"type":"read_file","path":"..."} — model
//     put the tool name in the type field instead of using "tool_call".
//
// When `args` is missing on a tool_call, re-decode the raw JSON into a
// generic map and either pull `arguments`/`parameters` over to args, or
// lift every non-envelope top-level field into a synthetic args object.
// This is purely a recovery path; the system prompt still teaches the
// canonical shape.
func liftMissingArgs(resp *ModelResponse, raw string) {
	// PC-050: if Type is a known tool name, treat it as a tool_call with
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

// recoverTruncatedWriteFile attempts to recover a write_file tool call
// where the content was truncated by max_tokens.
func recoverTruncatedWriteFile(partial string) (ModelResponse, error) {
	// The pattern is: {"type":"tool_call","name":"write_file","args":{"path":"...","content":"...
	// We need to close the content string and the JSON objects

	// Find the "content":" part
	idx := strings.Index(partial, `"content":"`)
	if idx < 0 {
		idx = strings.Index(partial, `"content": "`)
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

	// Extract content: everything after "content":" until the end
	contentStart := idx + len(`"content":"`)
	if strings.Contains(partial[idx:idx+15], `: "`) {
		contentStart = idx + len(`"content": "`)
	}
	content := partial[contentStart:]

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

	// Build the args JSON
	args, _ := json.Marshal(WriteFileInput{Path: path, Content: unescaped})

	log.Printf("[agent] recovered truncated write_file: path=%s content=%d chars", path, len(unescaped))

	return ModelResponse{
		Type: "tool_call",
		Name: "write_file",
		Args: args,
	}, nil
}

// classifyAgentTier classifies the task tier using fast heuristics.
//
// Inversion (PC-159 follow-up): the prior cascade defaulted to T1, which
// kept V3 dormant on most real prompts (it only fires at T2+). After
// observing real flask-app debugging sessions where V3 never engaged,
// the rule is now: T2 is the floor for any non-trivial message, T0/T1
// are the narrow cases. Trivial chat ("hi", "thanks", "yes") stays T0.
// Single greetings or sub-15-char acknowledgements stay below the V3
// threshold; everything else gets the pipeline.
func classifyAgentTier(message string) Tier {
	trimmed := strings.TrimSpace(message)
	lower := strings.ToLower(trimmed)

	// T0: empty / sub-5-char garbage.
	if len(trimmed) < 5 {
		return Tier0Conversational
	}

	// T0: trivial chat — exact-match against a small list of greetings
	// and acknowledgements. Anything more substantial than these is
	// assumed to be a task, even if it looks short.
	trivialChat := map[string]bool{
		"hi": true, "hello": true, "hey": true, "yo": true, "sup": true,
		"thanks": true, "thank you": true, "ty": true, "thx": true,
		"ok": true, "okay": true, "k": true,
		"yes": true, "yep": true, "yeah": true, "no": true, "nope": true,
		"sure": true, "got it": true, "cool": true, "nice": true,
		"good": true, "great": true, "perfect": true,
		"bye": true, "goodbye": true, "later": true, "cya": true,
	}
	if trivialChat[lower] {
		return Tier0Conversational
	}

	// Count multi-component indicators for T3 detection.
	multiIndicators := 0
	multiPatterns := []string{
		"multiple files", "several files", "full application",
		"api routes", "middleware", "database", "authentication",
		"frontend and backend", "client and server",
		"multiple endpoints", "with tests",
	}
	for _, p := range multiPatterns {
		if strings.Contains(lower, p) {
			multiIndicators++
		}
	}
	fileIndicators := 0
	filePatterns := []string{
		".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".c", ".h",
		".sh", ".json", ".toml", ".yaml", ".yml", ".css", ".html",
		"package.json", "cargo.toml", "go.mod", "makefile",
	}
	for _, p := range filePatterns {
		if strings.Contains(lower, p) {
			fileIndicators++
		}
	}

	// T3: explicit multi-component or architectural complexity. Costs
	// the most (more turn budget, deeper V3 candidate generation), so
	// the bar stays high.
	if multiIndicators >= 2 || (fileIndicators >= 4 && multiIndicators >= 1) {
		return Tier3Hard
	}

	// T2: default. Anything that survived the T0 gate above is a real
	// task and gets the pipeline.
	return Tier2Medium
}

// Toolchain describes one language ecosystem detected in the project.
// The fields are surfaced into the system prompt so the model knows
// which runner to invoke and how to install deps if needed.
//
// Detection is manifest-driven: presence of pyproject.toml means
// Python, package.json means Node, Cargo.toml means Rust, etc. A
// polyglot project (React frontend + Django backend + deploy scripts)
// returns multiple Toolchains so the model can pick the right one
// per file edit. See PC-191.
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
// verification (PC-193).
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
		// Strong signal: a venv with at least one third-party package
		// installed. We check for the venv's site-packages directory
		// containing anything beyond pip/setuptools.
		for _, vd := range []string{"venv", ".venv", "env", ".env-py"} {
			sp := filepath.Join(workingDir, vd, "lib")
			if entries, err := os.ReadDir(sp); err == nil {
				for _, e := range entries {
					if strings.HasPrefix(e.Name(), "python") && e.IsDir() {
						pkgs := filepath.Join(sp, e.Name(), "site-packages")
						if hasUserPackages(pkgs) {
							return "deps appear installed in project venv — skip install, go straight to verify"
						}
					}
				}
			}
		}
		// No venv but a requirements.txt or pyproject.toml exists.
		if hasFile(workingDir, "requirements.txt") || hasFile(workingDir, "pyproject.toml") {
			return "deps NOT installed (no populated venv found) — run install before verify"
		}
		return "no manifest, no venv — single-file script, just run it"

	case "node":
		// node_modules present and not empty → deps installed.
		nm := filepath.Join(workingDir, "node_modules")
		if entries, err := os.ReadDir(nm); err == nil && len(entries) > 0 {
			return "node_modules populated — skip install, go straight to verify"
		}
		return "node_modules missing — run install before verify"

	case "rust":
		// target/ exists → at least one build has happened.
		if info, err := os.Stat(filepath.Join(workingDir, "target")); err == nil && info.IsDir() {
			return "target/ exists — incremental build will be fast"
		}
		return "no target/ — first build will compile from scratch"

	case "go":
		// Hard to probe without GOPATH inspection; presence of go.sum
		// + non-empty vendor/ is the surest filesystem signal.
		if info, err := os.Stat(filepath.Join(workingDir, "vendor")); err == nil && info.IsDir() {
			return "vendor/ present — deps vendored, skip install"
		}
		if hasFile(workingDir, "go.sum") {
			return "go.sum present — `go run`/`go test` will fetch as needed"
		}
		return "no go.sum yet — run `go mod tidy` before verify"

	case "ruby":
		if info, err := os.Stat(filepath.Join(workingDir, "vendor", "bundle")); err == nil && info.IsDir() {
			return "vendor/bundle present — skip install"
		}
		return "no vendor/bundle — run install before verify"

	case "java-maven":
		if info, err := os.Stat(filepath.Join(workingDir, "target")); err == nil && info.IsDir() {
			return "target/ present — incremental build will be fast"
		}
		return "no target/ — first build will compile + download deps"

	case "java-gradle":
		if info, err := os.Stat(filepath.Join(workingDir, "build")); err == nil && info.IsDir() {
			return "build/ present — incremental build will be fast"
		}
		return "no build/ — first build will compile + download deps"

	case "php":
		if info, err := os.Stat(filepath.Join(workingDir, "vendor")); err == nil && info.IsDir() {
			return "vendor/ present — skip composer install"
		}
		return "no vendor/ — run composer install before verify"

	case "dotnet":
		if info, err := os.Stat(filepath.Join(workingDir, "bin")); err == nil && info.IsDir() {
			return "bin/ present — `dotnet run` will use cached restore"
		}
		return "no bin/ — first run does an implicit restore"

	case "dart":
		if info, err := os.Stat(filepath.Join(workingDir, ".dart_tool")); err == nil && info.IsDir() {
			return ".dart_tool/ present — pub get already run"
		}
		return "no .dart_tool/ — run pub get before verify"
	}
	return ""
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
// system prompt's venv hint. See PC-190.
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
	return true
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
		NCandidates:    3,
	}

	planStart := time.Now()
	Emit(NewEnvelope(EvtStageStart, "v3:plan", map[string]interface{}{
		"detail":     fmt.Sprintf("planning: %s", truncateStr(userMessage, 60)),
		"context_n":  len(pctx),
		"candidates": req.NCandidates,
	}))

	plan, err := callV3PlanStreaming(ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
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
