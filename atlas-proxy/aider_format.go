package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Aider format translation — converts internal tool results to Aider's
// expected whole-file format for SSE delivery.
//
// Aider expects responses in this format:
//
//   filename.py
//   ```python
//   <full file content>
//   ```
//
// The proxy's internal agent loop produces structured tool results
// (write_file calls with paths and content). This module collects those
// results and formats them for Aider.
// ---------------------------------------------------------------------------

// AgentFileChange represents a file that was written/edited during the agent loop.
type AgentFileChange struct {
	Path    string // relative path
	Content string // full file content after change
	IsNew   bool   // true if file was created, false if edited
}

// AgentRunResult captures everything the internal agent loop produced.
type AgentRunResult struct {
	FileChanges  []AgentFileChange
	TextMessages []string // text responses from the model
	Summary      string   // done summary
	ToolCalls    int      // total tool calls made
	TotalTokens  int
	Error        string
}

// formatForAider converts the agent loop's results into Aider's expected
// SSE response format. Each file change becomes a whole-file block.
func formatForAider(result *AgentRunResult) string {
	if len(result.FileChanges) == 0 {
		// No file changes. If there were tool calls (e.g. delete_file), return
		// empty to prevent Aider from misinterpreting text as file edits.
		// But if there were NO tool calls (pure conversation), return the text
		// so the user sees the response.
		if result.ToolCalls > 0 {
			return ""
		}
		// Conversational response — return text messages + summary
		if len(result.TextMessages) > 0 {
			return strings.Join(result.TextMessages, "\n\n")
		}
		if result.Summary != "" {
			return result.Summary
		}
		return ""
	}

	var parts []string

	// Add any text messages first
	for _, msg := range result.TextMessages {
		if msg != "" {
			parts = append(parts, msg)
		}
	}

	// Format each file change as Aider whole-file block
	for _, fc := range result.FileChanges {
		lang := langTagFromExt(filepath.Ext(fc.Path))
		block := fmt.Sprintf("%s\n```%s\n%s\n```", fc.Path, lang, fc.Content)
		parts = append(parts, block)
	}

	// Add summary if present
	if result.Summary != "" && len(result.FileChanges) == 0 {
		parts = append(parts, result.Summary)
	}

	return strings.Join(parts, "\n\n")
}

// collectAgentResults runs the agent loop and collects all file changes
// into an AgentRunResult. This is the bridge between the agent loop
// and the Aider SSE response.
//
// When w/flusher are provided, status lines are streamed to Aider in real time
// so the user sees progress: [Turn N] [calling tool] [wrote file] etc.
func collectAgentResults(ctx *AgentContext, userMessage string, w http.ResponseWriter, flusher http.Flusher) *AgentRunResult {
	result := &AgentRunResult{}
	turnCount := 0

	// streamStatus sends a visible status line to Aider via SSE.
	// Aider renders this as text in the chat output.
	streamStatus := func(msg string) {
		if w == nil || flusher == nil {
			return
		}
		injectContentDelta(w, flusher, msg+"\n")
	}

	// Rich streaming output — every operation visible in real time
	startTime := time.Now()
	filesCreated := []string{}
	filesEdited := []string{}
	filesDeleted := []string{}
	commandsRun := []string{}
	v3FilesEnhanced := 0

	ctx.StreamFn = func(eventType string, data interface{}) {
		switch eventType {
		case "tool_call":
			if m, ok := data.(map[string]interface{}); ok {
				turnCount++
				toolName, _ := m["name"].(string)
				if toolName == "write_file" || toolName == "edit_file" || toolName == "delete_file" {
					result.ToolCalls++
				}

				switch toolName {
				case "write_file":
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp WriteFileInput
						json.Unmarshal(args, &inp)
						lines := strings.Count(inp.Content, "\n") + 1
						tier := classifyFileTier(inp.Path, inp.Content)
						filesCreated = append(filesCreated, inp.Path)

						if tier >= Tier2Medium {
							v3FilesEnhanced++
							streamStatus(fmt.Sprintf("[Turn %d/%d] \u270d writing %s (T2, V3 pipeline)", turnCount, ctx.MaxTurns, inp.Path))
							streamStatus(fmt.Sprintf("  \u250c\u2500 V3 Pipeline \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"))
							streamStatus(fmt.Sprintf("  \u2502 Baseline: %d lines, scoring...", lines))
						} else {
							streamStatus(fmt.Sprintf("[Turn %d/%d] \u270d writing %s (T1, direct)", turnCount, ctx.MaxTurns, inp.Path))
						}
					}

				case "edit_file":
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp EditFileInput
						json.Unmarshal(args, &inp)
						filesEdited = append(filesEdited, inp.Path)
						// Show inline diff preview
						oldLines := strings.Count(inp.OldStr, "\n") + 1
						newLines := strings.Count(inp.NewStr, "\n") + 1
						streamStatus(fmt.Sprintf("[Turn %d/%d] \u270f\ufe0f editing %s", turnCount, ctx.MaxTurns, inp.Path))
						// Show first changed line
						oldPreview := strings.Split(inp.OldStr, "\n")[0]
						newPreview := strings.Split(inp.NewStr, "\n")[0]
						if len(oldPreview) > 60 { oldPreview = oldPreview[:60] + "..." }
						if len(newPreview) > 60 { newPreview = newPreview[:60] + "..." }
						streamStatus(fmt.Sprintf("  - %s", oldPreview))
						streamStatus(fmt.Sprintf("  + %s", newPreview))
						if oldLines > 1 || newLines > 1 {
							streamStatus(fmt.Sprintf("  (%d lines replaced with %d lines)", oldLines, newLines))
						}
					}

				case "delete_file":
					// Silent on the SSE stream (Aider misinterprets filenames as edits)
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp DeleteFileInput
						json.Unmarshal(args, &inp)
						filesDeleted = append(filesDeleted, inp.Path)
					}

				case "run_command":
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp RunCommandInput
						json.Unmarshal(args, &inp)
						cmd := inp.Command
						if len(cmd) > 70 { cmd = cmd[:70] + "..." }
						commandsRun = append(commandsRun, cmd)
						streamStatus(fmt.Sprintf("[Turn %d/%d] \U0001f527 running: %s", turnCount, ctx.MaxTurns, cmd))
					}

				case "read_file":
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp ReadFileInput
						json.Unmarshal(args, &inp)
						streamStatus(fmt.Sprintf("[Turn %d/%d] \U0001f4d6 reading %s", turnCount, ctx.MaxTurns, inp.Path))
					}

				case "search_files":
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp SearchFilesInput
						json.Unmarshal(args, &inp)
						streamStatus(fmt.Sprintf("[Turn %d/%d] \U0001f50d searching \"%s\"", turnCount, ctx.MaxTurns, truncateStr(inp.Pattern, 30)))
					}

				case "list_directory":
					if args, ok := m["args"].(json.RawMessage); ok {
						var inp ListDirectoryInput
						json.Unmarshal(args, &inp)
						streamStatus(fmt.Sprintf("[Turn %d/%d] \U0001f4c1 listing %s", turnCount, ctx.MaxTurns, inp.Path))
					}

				case "plan_tasks":
					streamStatus(fmt.Sprintf("[Turn %d/%d] \U0001f4cb planning subtasks...", turnCount, ctx.MaxTurns))
				}
			}
		case "tool_result":
			if m, ok := data.(map[string]interface{}); ok {
				toolName, _ := m["tool"].(string)
				success, _ := m["success"].(bool)
				elapsed, _ := m["elapsed"].(string)

				switch toolName {
				case "run_command":
					if success {
						streamStatus(fmt.Sprintf("  \u2713 exit code 0 (%s)", elapsed))
					} else {
						errMsg, _ := m["error"].(string)
						if errMsg != "" {
							streamStatus(fmt.Sprintf("  \u2717 failed: %s (%s)", truncateStr(errMsg, 80), elapsed))
						} else {
							streamStatus(fmt.Sprintf("  \u2717 non-zero exit (%s)", elapsed))
						}
					}

				case "write_file":
					if success {
						streamStatus(fmt.Sprintf("  \u2713 wrote successfully (%s)", elapsed))
					} else {
						errMsg, _ := m["error"].(string)
						streamStatus(fmt.Sprintf("  \u2717 write failed: %s", truncateStr(errMsg, 80)))
					}

				case "edit_file":
					if success {
						streamStatus(fmt.Sprintf("  \u2713 edit applied (%s)", elapsed))
					} else {
						errMsg, _ := m["error"].(string)
						streamStatus(fmt.Sprintf("  \u2717 edit failed: %s", truncateStr(errMsg, 80)))
					}

				case "read_file":
					if success {
						// Extract line count from result data
						if dataRaw, ok := m["data"].(json.RawMessage); ok {
							var rd ReadFileOutput
							if json.Unmarshal(dataRaw, &rd) == nil {
								streamStatus(fmt.Sprintf("  \u2514\u2500 %d lines loaded", rd.TotalLines))
							}
						}
					}

				case "search_files":
					if success {
						if dataRaw, ok := m["data"].(json.RawMessage); ok {
							var sd SearchFilesOutput
							if json.Unmarshal(dataRaw, &sd) == nil {
								streamStatus(fmt.Sprintf("  \u2514\u2500 %d matches found", sd.TotalCount))
							}
						}
					}

				case "list_directory":
					if success {
						if dataRaw, ok := m["data"].(json.RawMessage); ok {
							var ld ListDirectoryOutput
							if json.Unmarshal(dataRaw, &ld) == nil {
								names := []string{}
								for _, e := range ld.Entries {
									if len(names) < 6 {
										n := e.Name
										if e.Type == "dir" { n += "/" }
										names = append(names, n)
									}
								}
								extra := ""
								if len(ld.Entries) > 6 {
									extra = fmt.Sprintf(" +%d more", len(ld.Entries)-6)
								}
								streamStatus(fmt.Sprintf("  \u2514\u2500 %d items: %s%s", len(ld.Entries), strings.Join(names, ", "), extra))
							}
						}
					}
				}
			}
		case "v3_progress":
			// Forward V3 pipeline progress to Aider in real-time
			if m, ok := data.(map[string]string); ok {
				if msg, ok := m["message"]; ok {
					streamStatus(msg)
				}
			}
		case "text":
			if m, ok := data.(map[string]string); ok {
				if content, ok := m["content"]; ok && content != "" {
					result.TextMessages = append(result.TextMessages, content)
				}
			}
		case "done":
			if m, ok := data.(map[string]string); ok {
				result.Summary = m["summary"]
			}
		}
	}

	// Auto-approve everything (Aider handles its own confirmation)
	ctx.PermissionMode = PermissionAcceptEdits
	ctx.PermissionFn = func(toolName string, args json.RawMessage) bool {
		if toolName == "run_command" {
			denied, _ := shouldDenyToolCall(toolName, args)
			return !denied
		}
		return true
	}

	// Track file contents directly from write_file/edit_file calls.
	// We intercept the tool executors to capture content before it's "written"
	// (writeFileDirect no longer writes to disk — Aider handles that).
	var fileChangesMu sync.Mutex
	fileChangesMap := make(map[string]string) // path → content

	// No temp directory — WorkingDir is the real project directory.
	// read_file, list_directory, search_files see actual project files.
	// write_file saves to fileChangesMap (in-memory) for Aider delivery.
	// run_command executes in the real project directory.

	originalWriteExecute := getTool("write_file").Execute
	getTool("write_file").Execute = func(rawInput json.RawMessage, agentCtx *AgentContext) (*ToolResult, error) {
		var input WriteFileInput
		json.Unmarshal(rawInput, &input)
		res, err := originalWriteExecute(rawInput, agentCtx)
		if err == nil && res.Success {
			fileChangesMu.Lock()
			fileChangesMap[input.Path] = input.Content
			fileChangesMu.Unlock()
			// File content goes to fileChangesMap for Aider delivery.
			// Not written to disk — Aider handles file creation.
		}
		return res, err
	}
	originalEditExecute := getTool("edit_file").Execute
	getTool("edit_file").Execute = func(rawInput json.RawMessage, agentCtx *AgentContext) (*ToolResult, error) {
		var input EditFileInput
		json.Unmarshal(rawInput, &input)
		res, err := originalEditExecute(rawInput, agentCtx)
		if err == nil && res.Success {
			// For edits, read the current content from our tracked state
			// (the edit was applied by the executor which reads from disk)
			// Since we no longer write to disk, edits on files we created
			// need to use the in-memory content.
			fileChangesMu.Lock()
			if existing, ok := fileChangesMap[input.Path]; ok {
				// Apply the edit to our in-memory copy
				if input.ReplaceAll {
					fileChangesMap[input.Path] = strings.ReplaceAll(existing, input.OldStr, input.NewStr)
				} else {
					fileChangesMap[input.Path] = strings.Replace(existing, input.OldStr, input.NewStr, 1)
				}
			}
			fileChangesMu.Unlock()
		}
		return res, err
	}

	defer func() {
		getTool("write_file").Execute = originalWriteExecute
		getTool("edit_file").Execute = originalEditExecute
	}()

	// WorkingDir is the real project directory — all tools see actual project files.
	log.Printf("[agent] project dir: %s", ctx.WorkingDir)

	// Run the agent loop
	err := runAgentLoop(ctx, userMessage)
	if err != nil {
		result.Error = err.Error()
	}

	result.TotalTokens = ctx.TotalTokens

	// Collect file contents from in-memory tracking (no disk reads needed)
	fileChangesMu.Lock()
	for path, content := range fileChangesMap {
		// Strip absolute paths — Aider needs relative paths from its CWD
		relPath := path
		if filepath.IsAbs(relPath) {
			relPath = filepath.Base(relPath)
			// Try to preserve subdirectory structure
			parts := strings.Split(path, "/")
			for i, p := range parts {
				if p == "app" || p == "src" || p == "components" || p == "data" || p == "lib" || p == "pages" || p == "api" {
					relPath = strings.Join(parts[i:], "/")
					break
				}
			}
		}
		result.FileChanges = append(result.FileChanges, AgentFileChange{
			Path:    relPath,
			Content: content,
			IsNew:   true,
		})
	}
	fileChangesMu.Unlock()

	// Deduplicate (keep last version of each file)
	result.FileChanges = deduplicateChanges(result.FileChanges)

	log.Printf("[agent] completed: %d file changes, %d tool calls, %d tokens",
		len(result.FileChanges), result.ToolCalls, result.TotalTokens)

	// Rich completion summary
	totalElapsed := time.Since(startTime)
	streamStatus("")
	streamStatus("\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")
	streamStatus(fmt.Sprintf("\u2713 Complete (%d turns, %ds)", turnCount, int(totalElapsed.Seconds())))
	if len(filesCreated) > 0 {
		names := filesCreated
		if len(names) > 5 { names = append(names[:5], fmt.Sprintf("+%d more", len(filesCreated)-5)) }
		streamStatus(fmt.Sprintf("  Files created:  %d (%s)", len(filesCreated), strings.Join(names, ", ")))
	}
	if len(filesEdited) > 0 {
		streamStatus(fmt.Sprintf("  Files edited:   %d (%s)", len(filesEdited), strings.Join(filesEdited, ", ")))
	}
	if len(filesDeleted) > 0 {
		streamStatus(fmt.Sprintf("  Files deleted:  %d", len(filesDeleted)))
	}
	if len(commandsRun) > 0 {
		streamStatus(fmt.Sprintf("  Commands run:   %d", len(commandsRun)))
	}
	if v3FilesEnhanced > 0 {
		streamStatus(fmt.Sprintf("  V3 pipeline:    %d files enhanced", v3FilesEnhanced))
	}
	streamStatus(fmt.Sprintf("  Tokens:         %d", result.TotalTokens))
	streamStatus("\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")

	return result
}

// readFileContent reads a file from disk.
func readFileContent(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// runInternalAgentLoop creates an AgentContext from an Aider ChatRequest,
// runs the internal agent loop, and returns the collected results.
// This is called from handleStreamingChat for T1+ tasks when ATLAS_AGENT_LOOP=1.
func runInternalAgentLoop(req ChatRequest, tier Tier, w http.ResponseWriter, flusher http.Flusher) *AgentRunResult {
	// Extract user's actual message (strip Aider format instructions)
	userMessage := ""
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role != "user" {
			continue
		}
		content := req.Messages[i].Content
		// Skip Aider format reminders
		if strings.HasPrefix(content, "# *SEARCH/REPLACE") ||
			strings.HasPrefix(content, "To suggest changes") ||
			strings.HasPrefix(content, "I am not sharing") ||
			strings.HasPrefix(content, "Return edits similar") {
			continue
		}
		// Strip appended instructions
		if idx := strings.Index(content, "\n\nTo suggest changes"); idx > 0 {
			content = content[:idx]
		}
		if idx := strings.Index(content, "\n\n# File editing rules"); idx > 0 {
			content = content[:idx]
		}
		if idx := strings.Index(content, "\n\nYou MUST use"); idx > 0 {
			content = content[:idx]
		}
		content = strings.TrimPrefix(content, "/nothink\n")
		userMessage = strings.TrimSpace(content)
		break
	}

	if userMessage == "" {
		return nil
	}

	// Detect the real project directory by finding where Aider's files exist on disk.
	// Aider sends file contents in messages — we check where those files actually are.
	workingDir := detectRealProjectDir(req.Messages)
	if workingDir == "" {
		workingDir = extractWorkingDir(req.Messages)
	}
	if workingDir == "" {
		// In Docker, the project is mounted at /workspace
		if info, err := os.Stat("/workspace"); err == nil && info.IsDir() {
			workingDir = "/workspace"
		} else if cwd, err := os.Getwd(); err == nil && cwd != "/" {
			// Bare metal: use the process working directory
			workingDir = cwd
		} else {
			workingDir = "/tmp"
		}
	}

	log.Printf("  agent loop: user=%s workdir=%s tier=%s",
		truncate(userMessage, 80), workingDir, tier)

	// Create agent context — workingDir is the real project path
	// collectAgentResults will override WorkingDir with tempDir for tool execution
	// but RealProjectDir stays set for delete_file
	ctx := NewAgentContext(workingDir, tier)
	ctx.RealProjectDir = workingDir
	ctx.InferenceURL = inferenceURL
	ctx.SandboxURL = sandboxURL
	ctx.LensURL = lensURL
	ctx.V3URL = envOr("ATLAS_V3_URL", "http://localhost:8070")
	ctx.Project = detectProjectInfo(workingDir)

	// Extract existing file context from Aider's messages
	fileCtx := extractFileContext(req.Messages)
	for fname, content := range fileCtx {
		fullPath := resolvePath(fname, workingDir)
		ctx.RecordFileRead(fullPath, content)
	}

	// Fast path: if the user is asking to delete a file, handle it directly
	// without the agent loop. This avoids the model generating text that
	// Aider would misinterpret as file edits.
	if isDeleteRequest(userMessage) {
		return handleDeleteDirect(ctx, userMessage)
	}

	return collectAgentResults(ctx, userMessage, w, flusher)
}

// isDeleteRequest checks if the user is asking to delete/remove files.
func isDeleteRequest(msg string) bool {
	lower := strings.ToLower(msg)
	return (strings.Contains(lower, "delete") || strings.Contains(lower, "remove")) &&
		(strings.Contains(lower, "file") || strings.Contains(lower, ".ts") ||
			strings.Contains(lower, ".py") || strings.Contains(lower, ".js") ||
			strings.Contains(lower, ".tsx") || strings.Contains(lower, ".go"))
}

// handleDeleteDirect deletes files mentioned in the message without using
// the agent loop. Returns an AgentRunResult with no file changes so
// formatForAider sends empty content (no Aider-parseable text).
func handleDeleteDirect(ctx *AgentContext, msg string) *AgentRunResult {
	result := &AgentRunResult{}

	// Extract file paths from the message
	re := regexp.MustCompile(`[\w./\-\[\]]+\.\w{1,10}`)
	paths := re.FindAllString(msg, -1)

	for _, p := range paths {
		// Skip common false positives
		if p == "delete_file" || p == "remove_file" {
			continue
		}
		realPath := filepath.Join(ctx.RealProjectDir, p)
		if _, err := os.Stat(realPath); err == nil {
			os.Remove(realPath)
			log.Printf("[delete_direct] deleted %s from %s", p, ctx.RealProjectDir)
			result.ToolCalls++
		}
	}

	return result
}

// extractWorkingDir tries to find the working directory from Aider's messages.
// Aider includes system messages with repo-map references and file paths.
// Also checks for git paths that hint at the project root.
func extractWorkingDir(messages []ChatMessage) string {
	// Strategy 1: Find file paths mentioned in system messages
	for _, msg := range messages {
		content := msg.Content
		// Look for Aider's "Repo-map" or file paths
		for _, line := range strings.Split(content, "\n") {
			line = strings.TrimSpace(line)
			// Git repo paths: ".git with N files"
			if strings.Contains(line, ".git") && strings.Contains(line, "files") {
				continue // Not useful — doesn't have the absolute path
			}
			// Absolute file paths
			if strings.HasPrefix(line, "/tmp/atlas-") || strings.HasPrefix(line, "/home/") {
				// This might be "filename.py" or "/tmp/atlas-test-c/main.c"
				// Extract the directory
				if idx := strings.LastIndex(line, "/"); idx > 0 {
					dir := line[:idx]
					if !strings.Contains(dir, " ") && len(dir) > 5 {
						return dir
					}
				}
			}
		}
	}

	// Strategy 2: Check for known test directories that exist on disk
	for _, msg := range messages {
		content := msg.Content
		// Look for directory-like paths
		re := regexp.MustCompile(`/tmp/atlas-[\w-]+`)
		matches := re.FindAllString(content, -1)
		for _, m := range matches {
			if info, err := os.Stat(m); err == nil && info.IsDir() {
				return m
			}
		}
	}

	// Strategy 3: Check if there are any /tmp/atlas-* directories created recently
	entries, _ := filepath.Glob("/tmp/atlas-test-*")
	if len(entries) > 0 {
		// Return the most recently modified
		var newest string
		var newestTime int64
		for _, e := range entries {
			if info, err := os.Stat(e); err == nil {
				if info.ModTime().Unix() > newestTime {
					newestTime = info.ModTime().Unix()
					newest = e
				}
			}
		}
		if newest != "" {
			return newest
		}
	}

	return ""
}

// detectRealProjectDir finds the actual project directory by checking where
// Aider's file context exists on disk, or by finding the most recently modified
// git-initialized directory in /tmp that matches the test pattern.
func detectRealProjectDir(messages []ChatMessage) string {
	// Strategy 1: Find where Aider's file context exists on disk
	fileCtx := extractFileContext(messages)
	for fname := range fileCtx {
		entries, _ := filepath.Glob("/tmp/*/")
		for _, dir := range entries {
			candidate := filepath.Join(strings.TrimSuffix(dir, "/"), fname)
			if _, err := os.Stat(candidate); err == nil {
				realDir := strings.TrimSuffix(dir, "/")
				if _, err := os.Stat(filepath.Join(realDir, ".git")); err == nil {
					return realDir
				}
			}
		}
		break
	}

	// Strategy 2: Find the most recently modified directory with
	// .aider.chat.history.md — search /tmp AND user home directory
	var bestDir string
	var bestTime int64

	homeDir := os.Getenv("HOME")
	if homeDir == "" {
		homeDir = "/home"
	}
	searchRoots := []string{"/tmp", homeDir}
	for _, root := range searchRoots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			dir := filepath.Join(root, e.Name())
			aiderHistory := filepath.Join(dir, ".aider.chat.history.md")
			if info, err := os.Stat(aiderHistory); err == nil {
				if info.ModTime().Unix() > bestTime {
					bestTime = info.ModTime().Unix()
					bestDir = dir
				}
			}
		}
	}
	return bestDir
}

// deduplicateChanges keeps only the last change for each file path.
func deduplicateChanges(changes []AgentFileChange) []AgentFileChange {
	seen := make(map[string]int)
	for i, c := range changes {
		seen[c.Path] = i
	}
	var result []AgentFileChange
	for i, c := range changes {
		if seen[c.Path] == i {
			result = append(result, c)
		}
	}
	return result
}
