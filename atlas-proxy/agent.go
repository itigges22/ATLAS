package main

import (
	"bytes"
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

	// Get the constrained output schema
	schemaJSON := buildToolCallSchemaJSON()

	consecutiveReads := 0   // Track consecutive read-only calls
	consecutiveErrors := 0  // Track consecutive tool failures to break error loops

	for turn := 0; turn < ctx.MaxTurns; turn++ {
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
			return fmt.Errorf("LLM call failed on turn %d: %w", turn, err)
		}
		ctx.TotalTokens += tokens

		// Parse the response — extract JSON even if model added surrounding text
		parsed, parseErr := extractModelResponse(response)
		if parseErr != nil {
			log.Printf("[agent] parse error: %v | raw: %s", parseErr, truncateStr(response, 300))
			ctx.Stream("error", map[string]string{
				"error":    "failed to parse model response",
			})
			ctx.Messages = append(ctx.Messages, AgentMessage{
				Role:    "user",
				Content: "Your response was not valid JSON. Respond with ONLY a JSON object, no other text. Example: {\"type\":\"tool_call\",\"name\":\"write_file\",\"args\":{\"path\":\"file.py\",\"content\":\"code\"}}",
			})
			continue
		}

		log.Printf("[agent] turn=%d type=%s name=%s", turn, parsed.Type, parsed.Name)

		switch parsed.Type {
		case "done":
			ctx.Stream("done", map[string]string{"summary": parsed.Summary})
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

			ctx.Stream("tool_result", map[string]interface{}{
				"tool":    parsed.Name,
				"success": result.Success,
				"data":    json.RawMessage(result.Data),
				"error":   result.Error,
				"elapsed": elapsed.String(),
			})

			// Force-stop after destructive operations that shouldn't have follow-up
			if result.Error == "__FORCE_DONE__" {
				result.Error = ""
				// Don't stream anything — Aider interprets all text as file edits.
				// The file deletion already happened on disk. Just end silently.
				return nil
			}

			// Break error loops: if 3 tool calls fail in a row, stop
			if !result.Success {
				consecutiveErrors++
				if consecutiveErrors >= 3 {
					log.Printf("[agent] breaking error loop: %d consecutive failures at turn %d", consecutiveErrors, turn)
					ctx.Stream("done", map[string]string{"summary": "Stopped after repeated failures. The file may be too large to modify in one pass. Try asking for a smaller, specific change."})
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
func callLLMConstrained(ctx *AgentContext, schemaJSON string) (string, int, error) {
	// Build messages in chat format
	messages := make([]map[string]string, len(ctx.Messages))
	for i, msg := range ctx.Messages {
		messages[i] = map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		}
	}

	// DEV NOTE — Fox (Rust inference server) is disabled for now.
	// Fox has PagedAttention, prefix caching, continuous batching, and we added
	// grammar support + CUDA GPU offloading. However Fox's async Tokio engine loop
	// adds ~20ms overhead per token (tokio::spawn_blocking per decode step), giving
	// ~14 tok/s with grammar vs llama-server's 51 tok/s on the same GPU.
	//
	// To fix: move sampling entirely to the C side via llama_sampler_chain.
	// The SamplerChain code exists in fox/src/engine/grammar.rs but Fox's engine
	// still calls spawn_blocking per decode step. The fix is to run the decode loop
	// on a dedicated OS thread instead of the Tokio pool, eliminating the async
	// scheduling overhead. Once Fox hits 40+ tok/s with grammar, it becomes the
	// sole backend (grammar + batching + prefix caching in one server).
	//
	// For now: llama-server handles all inference (grammar-constrained agent loop
	// calls at 51 tok/s, and free-form V3 pipeline calls).

	// llama-server at InferenceURL (or ATLAS_LLAMA_URL override)
	// Uses response_format: json_object for grammar enforcement — 100% valid JSON
	llamaURL := envOr("ATLAS_LLAMA_URL", ctx.InferenceURL)

	reqBody := map[string]interface{}{
		"model":       modelName,
		"messages":    messages,
		"temperature": 0.3,
		"max_tokens":  32768,
		"stream":      false,
		"response_format": map[string]string{
			"type": "json_object",
		},
	}
	body, _ := json.Marshal(reqBody)
	endpoint := llamaURL + "/v1/chat/completions"
	httpReq, err := http.NewRequest("POST", endpoint, bytes.NewReader(body))
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

	// Both Fox and llama-server use /v1/chat/completions format:
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

	content := chatResp.Choices[0].Message.Content
	tokens := chatResp.Usage.TotalTokens

	return content, tokens, nil
}

// callLLMWithGBNF is no longer needed — grammar is integrated in callLLMConstrained
// when ATLAS_LLAMA_URL is set. Kept as comment for reference.

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
