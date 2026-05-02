package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Agent loop — iterative tool-calling loop between model and executors
// ---------------------------------------------------------------------------

// runAgentLoop runs the agent loop for a single user request.
// The model emits tool calls (constrained by grammar), the proxy executes them,
// and returns results. Continues until the model emits "done" or max turns hit.
func runAgentLoop(ctx *AgentContext, userMessage string) error {
	// Build system prompt with tool descriptions and project context
	systemPrompt := buildSystemPrompt(ctx)

	// Initialize messages
	ctx.Messages = []AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userMessage},
	}

	// PC-061: typed envelope for the agent loop. Pairs with the
	// stage_end + done envelope emitted at every return path below.
	agentLoopStart := time.Now()
	agentStartEv := NewEnvelope(EvtStageStart, "agent", map[string]interface{}{
		"detail": truncateStr(userMessage, 200),
	})
	Emit(agentStartEv)
	emitAgentDone := func(success bool, summary string) {
		duration := int64(time.Since(agentLoopStart) / time.Millisecond)
		Emit(Envelope{
			EventID:    NewEventID(),
			Timestamp:  float64(time.Now().UnixNano()) / 1e9,
			Type:       EvtStageEnd,
			Stage:      "agent",
			ParentID:   agentStartEv.EventID,
			DurationMS: duration,
			Payload: map[string]interface{}{
				"success": success,
				"summary": summary,
			},
		})
		Emit(Envelope{
			EventID:   NewEventID(),
			Timestamp: float64(time.Now().UnixNano()) / 1e9,
			Type:      EvtDone,
			Stage:     "agent",
			Payload: map[string]interface{}{
				"success":           success,
				"total_duration_ms": duration,
				"summary":           summary,
			},
		})
	}

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

	for turn := 0; turn < ctx.MaxTurns; turn++ {
		// Bail out fast if the upstream request was cancelled (Aider closed the
		// connection, user hit Ctrl-C, terminal exited). Without this check the
		// loop would keep grinding LLM calls and tool work for a client that's
		// already gone, burning GPU. See ISSUES.md PC-036.
		if ctx.Ctx != nil {
			select {
			case <-ctx.Ctx.Done():
				log.Printf("[agent] cancelled at turn %d: %v", turn, ctx.Ctx.Err())
				EmitSimple(EvtError, "agent", fmt.Sprintf("cancelled at turn %d: %v", turn, ctx.Ctx.Err()))
				emitAgentDone(false, "cancelled by client")
				return ctx.Ctx.Err()
			default:
			}
		}

		// Trim conversation history if it gets too long (prevent context overflow)
		// Keep system prompt + last 8 messages
		if len(ctx.Messages) > 12 {
			trimmed := make([]AgentMessage, 0, 10)
			trimmed = append(trimmed, ctx.Messages[0]) // system prompt
			trimmed = append(trimmed, ctx.Messages[1]) // user message
			// Keep last 8 messages (recent context)
			start := len(ctx.Messages) - 8
			trimmed = append(trimmed, ctx.Messages[start:]...)
			ctx.Messages = trimmed
			log.Printf("[agent] trimmed conversation to %d messages", len(ctx.Messages))
		}

		// Call LLM with grammar constraint
		response, tokens, err := callLLMConstrained(ctx, schemaJSON)
		if err != nil {
			ctx.Stream("error", map[string]string{"error": err.Error()})
			EmitSimple(EvtError, "agent", err.Error())
			emitAgentDone(false, fmt.Sprintf("LLM call failed: %v", err))
			return fmt.Errorf("LLM call failed on turn %d: %w", turn, err)
		}
		ctx.TotalTokens += tokens

		// Parse the response — extract JSON even if model added surrounding text
		parsed, parseErr := extractModelResponse(response)
		if parseErr != nil {
			log.Printf("[agent] parse error: %v | raw_len=%d | raw: %q", parseErr, len(response), truncateStr(response, 500))
			ctx.Stream("error", map[string]string{
				"error":    "failed to parse model response",
			})
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "user",
				Content: "Your response was not valid JSON. Respond with ONLY a JSON object, no other text. Example: {\"type\":\"tool_call\",\"name\":\"write_file\",\"args\":{\"path\":\"file.py\",\"content\":\"code\"}}",
			})
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
			ctx.Stream("done", map[string]string{"summary": parsed.Summary})
			emitAgentDone(true, parsed.Summary)
			return nil

		case "text":
			ctx.Stream("text", map[string]string{"content": parsed.Content})
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "assistant",
				Content: response,
			})

		case "tool_call":
			ctx.Stream("tool_call", map[string]interface{}{
				"name": parsed.Name,
				"args": json.RawMessage(parsed.Args),
				"turn": turn,
			})
			Emit(NewEnvelope(EvtToolCall, "agent", map[string]interface{}{
				"name":          parsed.Name,
				"args_summary":  truncateStr(string(parsed.Args), 200),
				"turn":          turn,
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
						emitAgentDone(false, "content too large for tool calls")
						return nil
					}
					continue
				}
			}

			// Fix A: Reject write_file for existing files — force edit_file.
			// Writing entire files as JSON strings causes truncation for files >100 lines.
			if parsed.Name == "write_file" {
				var wfInput WriteFileInput
				if json.Unmarshal(parsed.Args, &wfInput) == nil {
					existingPath := resolvePath(wfInput.Path, ctx.WorkingDir)
					if _, err := os.Stat(existingPath); err == nil {
						// File exists — redirect to edit_file
						lines := strings.Count(wfInput.Content, "\n") + 1
						if lines > 100 {
							log.Printf("[agent] rejecting write_file for existing %s (%d lines) — too large, must use edit_file", wfInput.Path, lines)
							ctx.Messages = append(ctx.Messages, AgentMessage{
								Role:    "assistant",
								Content: response,
							})
							ctx.Messages = append(ctx.Messages, AgentMessage{
								Role:       "tool",
								Content:    fmt.Sprintf(`{"success":false,"error":"File %s already exists (%d lines). Use edit_file with targeted old_str/new_str changes instead of rewriting the entire file. This avoids truncation."}`, wfInput.Path, lines),
								ToolCallID: fmt.Sprintf("call_%d", turn),
								ToolName:   "write_file",
							})
							continue
						}
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
				Stage:      "agent",
				DurationMS: elapsed.Milliseconds(),
				Payload: map[string]interface{}{
					"name":    parsed.Name,
					"success": result.Success,
					"summary": truncateStr(result.Error, 200),
				},
			})

			// Force-stop after destructive operations that shouldn't have follow-up
			if result.Error == "__FORCE_DONE__" {
				result.Error = ""
				// Don't stream anything — Aider interprets all text as file edits.
				// The file deletion already happened on disk. Just end silently.
				emitAgentDone(true, "destructive operation completed")
				return nil
			}

			// Track productive state changes — write/edit/delete that landed.
			// Used below to soften the error-loop exit when work was completed.
			if result.Success && (parsed.Name == "write_file" || parsed.Name == "edit_file" || parsed.Name == "delete_file") {
				madeProductiveChange = true
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
						emitAgentDone(true, "wrote changes; verification failed")
					} else {
						// Non-productive 3-error exit. The previous message
						// ("file may be too large") presumed a write/edit
						// context, but this branch fires for any 3 failures
						// — including discovery flailing (empty paths from
						// PC-039, missing files, bad regex). Be honest about
						// the failure mode and point at the tool errors so
						// the user can correct course.
						ctx.Stream("done", map[string]string{"summary": "Stopped after 3 tool failures with no successful changes. Common causes: the file you referenced isn't in the workspace, an empty path argument was passed, or a regex was malformed. Check the per-turn errors above, then try a more specific request (e.g. \"fix snake_game.py at line 95 — the curses bounds are wrong\")."})
						emitAgentDone(false, "stopped after 3 tool failures with no productive changes")
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
	EmitSimple(EvtError, "agent", fmt.Sprintf("max turns (%d) exceeded for %s task", ctx.MaxTurns, ctx.Tier))
	emitAgentDone(false, fmt.Sprintf("max turns (%d) exceeded", ctx.MaxTurns))
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

// callLLMOnce is one round-trip to llama-server's /v1/chat/completions.
// Extracted from callLLMConstrained so the empty-response retry can
// reuse the same plumbing with a different temperature + message list.
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
		"stream":      false,
		"response_format": map[string]string{
			"type": "json_object",
		},
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

	client := &http.Client{Timeout: 3 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", 0, fmt.Errorf("LLM request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("LLM returned %d: %s", resp.StatusCode, truncateStr(string(respBody), 500))
	}

	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", 0, fmt.Errorf("parse chat response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", 0, fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, chatResp.Usage.TotalTokens, nil
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

	// Tool descriptions
	sb.WriteString(buildToolDescriptions())

	// Rules
	sb.WriteString("## Rules\n\n")
	sb.WriteString("- Always read a file before editing it (use read_file then edit_file)\n")
	sb.WriteString("- IMPORTANT: Use edit_file for ALL changes to existing files. write_file is ONLY for creating brand new files. edit_file uses less tokens and avoids truncation.\n")
	sb.WriteString("- Use run_command to verify your changes work (build, test, lint)\n")
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

	// Show which files are in the project (names only, not full content).
	// Full content is available via read_file if needed.
	// This avoids consuming context window with pre-injected file dumps.
	if len(ctx.FilesRead) > 0 {
		sb.WriteString("## Project Files Available\n")
		for path := range ctx.FilesRead {
			sb.WriteString(fmt.Sprintf("- %s\n", path))
		}
		sb.WriteString("\nUse read_file to inspect these files if needed. For modifications, prefer edit_file (targeted changes) over write_file (full rewrite) to avoid token limits.\n\n")
	}

	return sb.String()
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

	var req struct {
		Message    string `json:"message"`
		WorkingDir string `json:"working_dir"`
		Mode       string `json:"mode"`    // "default", "accept-edits", "yolo"
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Message == "" {
		http.Error(w, "message is required", http.StatusBadRequest)
		return
	}

	workingDir := req.WorkingDir
	if workingDir == "" {
		workingDir = "."
	}

	// Classify tier from message
	tier := classifyAgentTier(req.Message)

	// Create agent context
	ctx := NewAgentContext(workingDir, tier)
	ctx.InferenceURL = inferenceURL
	ctx.SandboxURL = sandboxURL
	ctx.LensURL = lensURL
	ctx.V3URL = envOr("ATLAS_V3_URL", "http://localhost:8070")
	// Carry the upstream cancellation through so disconnects abort the loop
	// and llama-server's in-flight generation. See ISSUES.md PC-036.
	ctx.Ctx = r.Context()

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
// This is separate from main.go's classifyIntent which uses an LLM call —
// the agent loop needs faster classification that errs toward T1 (simpler).
// V3 pipeline is expensive; only activate for genuinely complex tasks.
func classifyAgentTier(message string) Tier {
	lower := strings.ToLower(message)

	// All messages go through the agent loop (even conversational).
	// The agent loop handles grammar enforcement which prevents the model
	// from outputting raw thinking blocks. Short messages still get T1
	// so the loop runs with a low turn budget.
	if len(strings.TrimSpace(message)) < 5 {
		// Only truly empty/trivial messages get T0
		return Tier0Conversational
	}

	// Count how many files/components are mentioned
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

	// Multi-component indicators
	multiIndicators := 0
	multiPatterns := []string{
		"multiple files", "several files", "project", "full application",
		"api routes", "middleware", "database", "authentication",
		"frontend and backend", "client and server",
		"3 routes", "multiple endpoints", "with tests",
	}
	for _, p := range multiPatterns {
		if strings.Contains(lower, p) {
			multiIndicators++
		}
	}

	// T3: Explicit multi-component or architectural complexity
	if multiIndicators >= 2 || (fileIndicators >= 4 && multiIndicators >= 1) {
		return Tier3Hard
	}

	// T2: Genuinely multi-component (not just 2 files)
	if fileIndicators >= 5 || multiIndicators >= 2 {
		return Tier2Medium
	}

	// Fix-intent against an existing file: never collapse to T1 — fixing a
	// bug in pre-existing code is harder than writing a fresh file, even when
	// only one file is mentioned. See ISSUES.md PC-025 Sub-finding A.
	//
	// PC-049: original list missed natural-language fix prompts like "still
	// does not", "isn't working", "try again", "the X is not Y". Real users
	// describe bugs without saying the word "bug" or "fix". Expanded
	// vocabulary, plus weakened the file-indicator requirement: a clear
	// continuation marker like "still" or "again" implies the user is
	// iterating on existing code even without naming the file extension.
	fixIntent := false
	for _, w := range []string{
		"fix", "broken", "doesn't work", "doesn't", "does not work", "does not",
		"not working", "isn't working", "isn't", "is not", "aren't", "wasn't",
		"didn't", "won't", "can't", "bug", "issue", "problem", "error",
		"failed", "fails", "failing", "incorrect", "wrong",
	} {
		if strings.Contains(lower, w) {
			fixIntent = true
			break
		}
	}
	// Continuation markers indicate the user is iterating on prior work,
	// which by definition involves an existing file even when the prompt
	// doesn't name an extension.
	continuation := false
	for _, w := range []string{"still", "again", "try again", "retry", "another", "also fix"} {
		if strings.Contains(lower, w) {
			continuation = true
			break
		}
	}
	if fixIntent && (fileIndicators >= 1 || continuation) {
		return Tier2Medium
	}
	// Strong continuation alone (e.g. "still doesn't pick up the food") is a
	// fix request even without explicit fix vocabulary.
	if continuation && len(strings.TrimSpace(message)) > 30 {
		return Tier2Medium
	}

	// T1: Default for coding tasks (single file creation/edit)
	codingTerms := []string{
		"create", "write", "build", "make", "implement", "add", "fix",
		"function", "class", "script", "program", "app", "tool",
	}
	for _, t := range codingTerms {
		if strings.Contains(lower, t) {
			return Tier1Simple
		}
	}

	return Tier1Simple
}
