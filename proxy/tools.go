package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

var toolRegistry = map[string]*ToolDef{}

func init() {
	registerTool(readFileTool())
	registerTool(writeFileTool())
	registerTool(editFileTool())
	registerTool(deleteFileTool())
	registerTool(runCommandTool())
	registerTool(searchFilesTool())
	registerTool(findFileTool())
	registerTool(listDirectoryTool())
	registerTool(planTasksTool())
}

func registerTool(t *ToolDef) {
	toolRegistry[t.Name] = t
}

func getTool(name string) *ToolDef {
	return toolRegistry[name]
}

func allTools() []*ToolDef {
	tools := make([]*ToolDef, 0, len(toolRegistry))
	for _, t := range toolRegistry {
		tools = append(tools, t)
	}
	return tools
}

// executeTool dispatches a tool call to its executor.
func executeToolCall(name string, args json.RawMessage, ctx *AgentContext) *ToolResult {
	tool := getTool(name)
	if tool == nil {
		return &ToolResult{
			Success: false,
			Error:   fmt.Sprintf("unknown tool: %s", name),
		}
	}

	// PC-040: distinguish "no args field at all" from "malformed args".
	// The model occasionally emits {"type":"tool_call","name":"read_file"}
	// with no "args" key, which lands here as nil/empty bytes. Calling
	// json.Unmarshal on that returns "unexpected end of JSON input" — the
	// same string a *truncated* response produces — and the old remap
	// branch below would then tell the model "your output was truncated,
	// use smaller edit_file calls" which is not just unhelpful, it
	// actively steers the model away from the read_file/list_directory
	// it was trying to make. Catch the empty case here and return a
	// per-tool hint that tells the model exactly what shape to send.
	trimmed := strings.TrimSpace(string(args))
	if trimmed == "" || trimmed == "null" {
		return &ToolResult{
			Success: false,
			Error:   missingArgsHint(name),
		}
	}

	result, err := tool.Execute(args, ctx)
	if err != nil {
		errMsg := err.Error()
		// Only treat "unexpected end of JSON" as truncation when the
		// args payload is large enough that truncation is plausible.
		// Short payloads with that error are malformed JSON, not
		// truncated output, and the model needs the real parser error
		// to correct itself.
		if len(args) > 200 && strings.Contains(errMsg, "unexpected end of JSON") {
			errMsg = "Tool call was truncated (output too long for context window). Use smaller, targeted edit_file calls instead of full write_file rewrites."
		}
		return &ToolResult{
			Success: false,
			Error:   errMsg,
		}
	}
	return result
}

// missingArgsHint returns a tool-specific message instructing the model
// what argument shape to send when it omits the args field entirely.
// See PC-040.
func missingArgsHint(name string) string {
	switch name {
	case "read_file":
		return `read_file: no arguments provided. Call with {"path":"<file>"}. Use list_directory {"path":"."} first if you need to discover what files exist.`
	case "write_file":
		return `write_file: no arguments provided. Call with {"path":"<file>","content":"<full file contents>"}.`
	case "edit_file":
		return `edit_file: no arguments provided. Call with {"path":"<file>","old_str":"<exact text to replace>","new_str":"<replacement>"}.`
	case "delete_file":
		return `delete_file: no arguments provided. Call with {"path":"<file>"}.`
	case "list_directory":
		return `list_directory: no arguments provided. Call with {"path":"."} for the working directory or {"path":"<subdir>"}.`
	case "search_files":
		return `search_files: no arguments provided. Call with {"pattern":"<regex>"} and optionally {"path":"<dir>","include":"*.py"}.`
	case "find_file":
		return `find_file: no arguments provided. Call with {"pattern":"<name regex>"} (e.g. {"pattern":"snake_game\\.py"}).`
	case "run_command":
		return `run_command: no arguments provided. Call with {"command":"<shell command>"}.`
	case "lint_python":
		return `lint_python: no arguments provided. Call with {"path":"<file.py>"} or {"code":"<source>"}.`
	default:
		return fmt.Sprintf("%s: no arguments provided. Inspect the tool schema and resend with the required fields.", name)
	}
}

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

func readFileTool() *ToolDef {
	return &ToolDef{
		Name:        "read_file",
		Description: "Read the contents of a file. Returns numbered lines. Use offset and limit for large files.",
		InputSchema: ReadFileInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input ReadFileInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Empty path → resolves to the working dir, which is a
			// directory, which fails with a confusing error the model
			// can't recover from. Reject early with a hint at how to
			// discover the file. See ISSUES.md PC-039.
			if strings.TrimSpace(input.Path) == "" {
				return &ToolResult{
					Success: false,
					Error:   "read_file: path cannot be empty. Call list_directory with path \".\" to see what files exist, or find_file with a name regex (e.g. \"snake_game\\.py\").",
				}, nil
			}

			path := resolveAgentPath(ctx, input.Path)

			data, err := os.ReadFile(path)
			if err != nil {
				return nil, fmt.Errorf("cannot read %s: %w", input.Path, err)
			}

			lines := strings.Split(string(data), "\n")
			totalLines := len(lines)

			start := 0
			if input.Offset != nil {
				start = *input.Offset
				if start < 0 {
					start = 0
				}
				if start > totalLines {
					start = totalLines
				}
			}

			end := totalLines
			if input.Limit != nil {
				end = start + *input.Limit
				if end > totalLines {
					end = totalLines
				}
			}

			// Build numbered output (matches Claude Code's cat -n format)
			var sb strings.Builder
			for i := start; i < end; i++ {
				fmt.Fprintf(&sb, "%d\t%s\n", i+1, lines[i])
			}

			content := sb.String()
			ctx.RecordFileRead(path, string(data))

			out := ReadFileOutput{
				Content:    content,
				TotalLines: totalLines,
				StartLine:  start + 1,
				EndLine:    end,
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// search_files
// ---------------------------------------------------------------------------

func searchFilesTool() *ToolDef {
	return &ToolDef{
		Name:        "search_files",
		Description: "Search for a regex pattern inside file CONTENTS. Returns matching lines with file paths and line numbers. Use glob to filter by filename pattern. To find a file by its name (not contents), use find_file or list_directory instead.",
		InputSchema: SearchFilesInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input SearchFilesInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Reject empty pattern: same reasoning as find_file. An empty
			// regex matches every line in every file. See ISSUES.md PC-037.
			if strings.TrimSpace(input.Pattern) == "" {
				return &ToolResult{
					Success: false,
					Error:   "search_files: pattern cannot be empty. Provide a regex to grep file contents for (e.g. \"def main\" or \"TODO\\(.*\\)\").",
				}, nil
			}

			searchPath := ctx.WorkingDir
			if input.Path != "" {
				searchPath = resolveAgentPath(ctx, input.Path)
			}

			re, err := regexp.Compile(input.Pattern)
			if err != nil {
				return nil, fmt.Errorf("invalid regex: %w", err)
			}

			var matches []SearchMatch
			maxMatches := 200

			err = filepath.WalkDir(searchPath, func(path string, d fs.DirEntry, walkErr error) error {
				if walkErr != nil {
					return nil // skip unreadable dirs
				}
				if d.IsDir() {
					base := d.Name()
					if base == ".git" || base == "node_modules" || base == "__pycache__" || base == ".next" || base == "target" {
						return filepath.SkipDir
					}
					return nil
				}

				// Apply glob filter
				if input.Glob != "" {
					matched, _ := filepath.Match(input.Glob, d.Name())
					if !matched {
						return nil
					}
				}

				// Skip binary/large files
				info, err := d.Info()
				if err != nil || info.Size() > 1<<20 { // 1MB max
					return nil
				}

				data, err := os.ReadFile(path)
				if err != nil {
					return nil
				}

				relPath, _ := filepath.Rel(ctx.WorkingDir, path)
				if relPath == "" {
					relPath = path
				}

				scanner := bufio.NewScanner(strings.NewReader(string(data)))
				lineNum := 0
				for scanner.Scan() {
					lineNum++
					line := scanner.Text()
					if re.MatchString(line) {
						matches = append(matches, SearchMatch{
							File:    relPath,
							Line:    lineNum,
							Content: truncateStr(line, 200),
						})
						if len(matches) >= maxMatches {
							break
						}
					}
				}

				if len(matches) >= maxMatches {
					return filepath.SkipAll
				}
				return nil
			})

			if err != nil && len(matches) == 0 {
				return nil, fmt.Errorf("search error: %w", err)
			}

			out := SearchFilesOutput{
				Matches:    matches,
				TotalCount: len(matches),
				Truncated:  len(matches) >= maxMatches,
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// list_directory
// ---------------------------------------------------------------------------

func listDirectoryTool() *ToolDef {
	return &ToolDef{
		Name:        "list_directory",
		Description: "List the contents of a directory. Returns file names, types (file/dir/symlink), and sizes.",
		InputSchema: ListDirectoryInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input ListDirectoryInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			dirPath := resolveAgentPath(ctx, input.Path)

			entries, err := os.ReadDir(dirPath)
			if err != nil {
				return nil, fmt.Errorf("cannot list %s: %w", input.Path, err)
			}

			var dirEntries []DirEntry
			for _, e := range entries {
				entryType := "file"
				if e.IsDir() {
					entryType = "dir"
				} else if e.Type()&os.ModeSymlink != 0 {
					entryType = "symlink"
				}

				var size int64
				if info, err := e.Info(); err == nil {
					size = info.Size()
				}

				dirEntries = append(dirEntries, DirEntry{
					Name: e.Name(),
					Type: entryType,
					Size: size,
				})
			}

			out := ListDirectoryOutput{
				Entries: dirEntries,
				Path:    dirPath,
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// write_file — T0/T1 direct, T2/T3 routes through V3 pipeline
// ---------------------------------------------------------------------------

func writeFileTool() *ToolDef {
	return &ToolDef{
		Name:        "write_file",
		Description: "Write content to a file. Creates parent directories if needed. For existing files, prefer edit_file for small changes.",
		InputSchema: WriteFileInput{},
		ReadOnly:    false,
		Destructive: true,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input WriteFileInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Reject empty path — same reasoning as read_file (PC-039).
			if strings.TrimSpace(input.Path) == "" {
				return &ToolResult{
					Success: false,
					Error:   "write_file: path cannot be empty. Provide a relative path like \"snake_game.py\" or \"src/main.py\".",
				}, nil
			}

			path := resolveAgentPath(ctx, input.Path)

			// Sanitise model output before anything else touches it.
			// Otherwise a markdown-fenced response with a prose preamble
			// ("Looking at the task..." / ```html / actual code / ```)
			// lands on disk verbatim and the file becomes unparseable.
			cleaned, sanitized := sanitizeFileContent(input.Path, input.Content)
			if sanitized {
				log.Printf("[write_file] sanitised markdown wrapper from %s (was %d chars, now %d)",
					input.Path, len(input.Content), len(cleaned))
				input.Content = cleaned
			}

			// Per-file tier classification — determines V3 pipeline activation
			fileTier := classifyFileTier(input.Path, input.Content)
			log.Printf("[write_file] %s → %s (%d lines)", input.Path, fileTier, strings.Count(input.Content, "\n")+1)

			// V3 pipeline fires on T2+ files when V3 service is available.
			// V3 takes the model's content as baseline candidate, generates diverse
			// alternatives via PlanSearch/DivSampling, build-verifies each, and
			// selects the best. This is the intelligence layer.
			if fileTier >= Tier2Medium && ctx.V3URL != "" {
				log.Printf("[write_file] V3 pipeline activating for %s", input.Path)
				return writeFileWithV3(path, input.Content, ctx)
			}

			// T1: Direct write — config, data, boilerplate
			return writeFileDirect(path, input.Content)
		},
	}
}

// writeFileDirect writes content to disk atomically (write tmp + rename).
// The proxy is the only thing downstream that touches the filesystem —
// the TUI is read-only at the workspace level — so this is where any
// write_file tool call ultimately lands. Without this the file would
// vanish into the void ("agent says it wrote the file but it isn't
// there" bug, fixed alongside PC-062).
func writeFileDirect(path, content string) (*ToolResult, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, fmt.Errorf("cannot create parent dir for %s: %w", path, err)
	}
	tmpPath := path + ".atlas.tmp"
	if err := os.WriteFile(tmpPath, []byte(content), 0644); err != nil {
		return nil, fmt.Errorf("cannot write %s: %w", path, err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return nil, fmt.Errorf("cannot rename temp file: %w", err)
	}
	out := WriteFileOutput{BytesWritten: len(content)}
	outBytes, _ := json.Marshal(out)
	return &ToolResult{Success: true, Data: outBytes}, nil
}

// v3CandidatesTested unwraps a possibly-nil V3 response so the
// stage_end envelope can carry a count even on error paths.
func v3CandidatesTested(r *V3GenerateResponse) int {
	if r == nil {
		return 0
	}
	return r.CandidatesTested
}

// writeFileWithV3 routes through the V3 pipeline for T2/T3 tasks.
// Model's content becomes baseline candidate #0; V3 generates diverse
// alternatives, tests all, selects the best.
func writeFileWithV3(path, baselineContent string, ctx *AgentContext) (*ToolResult, error) {
	// Build V3 request with project context
	req := V3GenerateRequest{
		FilePath:     path,
		BaselineCode: baselineContent,
		Tier:         int(ctx.Tier),
		WorkingDir:   ctx.WorkingDir,
	}

	// Add project context from files read during this session
	if len(ctx.FilesRead) > 0 {
		req.ProjectContext = make(map[string]string)
		for p, content := range ctx.FilesRead {
			relPath, _ := filepath.Rel(ctx.WorkingDir, p)
			if relPath == "" {
				relPath = p
			}
			// Truncate large files in context to save tokens
			if len(content) > 4000 {
				content = content[:4000] + "\n... (truncated)"
			}
			req.ProjectContext[relPath] = content
		}
	}

	// Add project info if available
	if ctx.Project != nil {
		req.Framework = ctx.Project.Framework
		req.BuildCommand = ctx.Project.BuildCommand
	}

	// Tell the user V3 is taking over so they don't think the file
	// vanished. write_file with V3 holds the disk write until V3 picks
	// a winner \u2014 without this message the chat goes silent for the 1\u20133
	// minute V3 cycle and looks broken.
	if ctx.StreamFn != nil {
		ctx.StreamFn("v3_progress", map[string]string{
			"message": fmt.Sprintf("V3 pipeline starting for %s \u2014 generating diverse candidates and build-verifying each.", filepath.Base(path)),
		})
	}
	Emit(NewEnvelope(EvtStageStart, "v3", map[string]interface{}{
		"detail": fmt.Sprintf("file=%s", filepath.Base(path)),
	}))
	v3Start := time.Now()

	// Call V3 service with streaming progress. Each stage callback also
	// fires a typed envelope so the pipeline pane shows V3 progress.
	// Three categories of progress events:
	//   token       \u2014 per-LLM-token delta from V3's streaming generator
	//   llm_start   \u2014 V3 is starting an LLM call (candidate gen, scoring\u2026)
	//   llm_end     \u2014 V3's LLM call finished (with token/timing summary)
	//   <other>     \u2014 pipeline stage marker (probe, plansearch, sandbox\u2026)
	currentV3Stage := ""
	v3Result, err := callV3GenerateStreaming(ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
		// Token deltas: forward to the TUI on a separate SSE event so
		// it can render them as a streaming dim row, mirroring how the
		// agent's own LLM tokens are shown. No envelope (would bloat
		// /events with thousands of metric events for a single call).
		if stage == "token" {
			if ctx.StreamFn != nil {
				ctx.StreamFn("v3_token", map[string]string{"text": detail})
			}
			return
		}
		// LLM-call boundary markers. Match the chat protocol's
		// llm_call_start/end shapes so the TUI can reuse handlers.
		if stage == "llm_start" {
			if ctx.StreamFn != nil {
				payload := map[string]interface{}{"detail": detail}
				for k, v := range data {
					payload[k] = v
				}
				ctx.StreamFn("v3_llm_start", payload)
			}
			return
		}
		if stage == "llm_end" {
			if ctx.StreamFn != nil {
				payload := map[string]interface{}{"detail": detail}
				for k, v := range data {
					payload[k] = v
				}
				ctx.StreamFn("v3_llm_end", payload)
			}
			return
		}

		// Dedicated structured events for the pipeline pane. The TUI
		// renders each as its own row instead of a generic v3_progress
		// string. data is the structured payload from V3's _emit; we
		// pass it through verbatim with `stage` and `detail` for
		// fallback rendering.
		if ctx.StreamFn != nil {
			eventName := v3StageToEvent(stage)
			if eventName == "v3_progress" {
				// Unknown / unmapped stage \u2014 emit the legacy text line
				// only. Keeps third-party clients that haven't migrated
				// to typed events working.
				ctx.StreamFn("v3_progress", map[string]string{
					"message": fmt.Sprintf("  \u2502 [%s] %s", stage, detail),
				})
			} else {
				payload := map[string]interface{}{
					"stage":  stage,
					"detail": detail,
				}
				for k, v := range data {
					payload[k] = v
				}
				ctx.StreamFn(eventName, payload)
			}
		}
		// Stage transitions emit start/end envelopes for the pipeline
		// pane \u2014 close the previous stage when we see a new name.
		if stage != currentV3Stage {
			if currentV3Stage != "" {
				Emit(Envelope{
					EventID:   NewEventID(),
					Timestamp: float64(time.Now().UnixNano()) / 1e9,
					Type:      EvtStageEnd,
					Stage:     "v3:" + currentV3Stage,
					Payload: map[string]interface{}{
						"success": true,
					},
				})
			}
			payload := map[string]interface{}{"detail": detail}
			for k, v := range data {
				payload[k] = v
			}
			Emit(NewEnvelope(EvtStageStart, "v3:"+stage, payload))
			currentV3Stage = stage
		} else {
			Emit(NewEnvelope(EvtMetric, "v3:"+stage,
				map[string]interface{}{"name": "progress", "value": detail}))
		}
	})
	if currentV3Stage != "" {
		Emit(Envelope{
			EventID:   NewEventID(),
			Timestamp: float64(time.Now().UnixNano()) / 1e9,
			Type:      EvtStageEnd,
			Stage:     "v3:" + currentV3Stage,
			Payload:   map[string]interface{}{"success": err == nil},
		})
	}
	Emit(Envelope{
		EventID:    NewEventID(),
		Timestamp:  float64(time.Now().UnixNano()) / 1e9,
		Type:       EvtStageEnd,
		Stage:      "v3",
		DurationMS: time.Since(v3Start).Milliseconds(),
		Payload: map[string]interface{}{
			"success":           err == nil,
			"candidates_tested": v3CandidatesTested(v3Result),
		},
	})
	if err != nil {
		// Fallback to direct write if V3 service unavailable
		log.Printf("[write_file] V3 failed: %s — falling back to direct write", err)
		ctx.Stream("text", map[string]string{"content": fmt.Sprintf("  \u2514\u2500 V3 unavailable, writing directly")})
		return writeFileDirect(path, baselineContent)
	}

	// Write the winning candidate (or baseline if V3 didn't improve)
	code := v3Result.Code
	if code == "" {
		code = baselineContent
	}

	// Sanitise V3 output. The pipeline's underlying LLM response
	// occasionally arrives with markdown fences and prose preamble
	// intact; if we don't strip them, every V3-rewritten file ships
	// with a "Looking at the task..." header on disk.
	if cleaned, sanitized := sanitizeFileContent(path, code); sanitized {
		log.Printf("[write_file] sanitised V3 output for %s", path)
		code = cleaned
	}

	// Stream V3 completion summary
	if ctx.StreamFn != nil {
		ctx.StreamFn("v3_progress", map[string]string{
			"message": fmt.Sprintf("  \u2514\u2500\u2500\u2500\u2500 V3 complete: %s, %d candidates", v3Result.PhaseSolved, v3Result.CandidatesTested),
		})
	}

	result, err := writeFileDirect(path, code)
	if err != nil {
		return nil, err
	}

	// Enrich result with V3 metadata
	out := WriteFileOutput{
		BytesWritten:     len(code),
		V3Used:           true,
		CandidatesTested: v3Result.CandidatesTested,
		WinningScore:     v3Result.WinningScore,
		PhaseSolved:      v3Result.PhaseSolved,
	}
	outBytes, _ := json.Marshal(out)
	result.Data = outBytes
	result.V3Used = true
	result.CandidatesTested = v3Result.CandidatesTested
	result.WinningScore = v3Result.WinningScore
	result.PhaseSolved = v3Result.PhaseSolved

	return result, nil
}

// ---------------------------------------------------------------------------
// edit_file — old_str/new_str with uniqueness validation
// ---------------------------------------------------------------------------

func editFileTool() *ToolDef {
	return &ToolDef{
		Name:        "edit_file",
		Description: "Edit a file by replacing an exact string with new content. The old_str must match exactly once in the file (unless replace_all is true). Always read_file before editing.",
		InputSchema: EditFileInput{},
		ReadOnly:    false,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input EditFileInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Reject empty path — same reasoning as read_file (PC-039).
			if strings.TrimSpace(input.Path) == "" {
				return &ToolResult{
					Success: false,
					Error:   "edit_file: path cannot be empty. Use read_file first on the target, then edit_file with the same path.",
				}, nil
			}

			path := resolveAgentPath(ctx, input.Path)

			// Require file was read first (staleness protection)
			if !ctx.WasFileRead(path) {
				return nil, fmt.Errorf("file not read yet — use read_file first before editing: %s", input.Path)
			}

			// Read current content
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, fmt.Errorf("cannot read %s: %w", input.Path, err)
			}
			content := string(data)

			// Check for staleness
			ctx.mu.Lock()
			lastRead := ctx.FileReadTimes[path]
			ctx.mu.Unlock()

			info, err := os.Stat(path)
			if err == nil && info.ModTime().After(lastRead) {
				return nil, fmt.Errorf("file modified since last read — read it again before editing: %s", input.Path)
			}

			// Find old_str with quote normalization
			actualOldStr := findActualString(content, input.OldStr)
			if actualOldStr == "" {
				// Not found — return helpful error
				return nil, fmt.Errorf("string to replace not found in file.\nSearched for: %s", truncateStr(input.OldStr, 200))
			}

			// Check uniqueness
			count := strings.Count(content, actualOldStr)
			if count > 1 && !input.ReplaceAll {
				return nil, fmt.Errorf("found %d matches of the string to replace. Set replace_all=true to replace all, or provide more context to uniquely identify the instance", count)
			}

			// No-op check
			if input.OldStr == input.NewStr {
				return nil, fmt.Errorf("old_str and new_str are identical — no change to make")
			}

			// Sanitise the replacement string before splicing it in. The
			// model occasionally fences the new_str ("```python\n...\n```")
			// even though it's a fragment, not a whole file. If we let
			// that slip through, every line of the edit would have a
			// stray ``` at the top and bottom.
			if cleanedNew, sanitized := sanitizeFileContent(input.Path, input.NewStr); sanitized {
				log.Printf("[edit_file] sanitised markdown wrapper from new_str of %s", input.Path)
				input.NewStr = cleanedNew
			}

			var newContent string
			if input.ReplaceAll {
				newContent = strings.ReplaceAll(content, actualOldStr, input.NewStr)
			} else {
				newContent = strings.Replace(content, actualOldStr, input.NewStr, 1)
			}

			// Route through V3 pipeline when the file warrants it. The
			// gate now mirrors write_file (file-tier only, no request-tier
			// AND-gate) — having two separate tier checks meant V3 only
			// fired when both classifiers happened to agree, which was
			// rare in practice. V3 takes the post-edit content as
			// baseline candidate #0; if its diverse alternatives
			// build-verify better, V3 wins; otherwise the baseline (=our
			// edit) wins. Either way the answer is build-verified.
			fileTier := classifyFileTier(input.Path, newContent)
			v3Out := V3EditMetadata{}
			if fileTier >= Tier2Medium && ctx.V3URL != "" {
				log.Printf("[edit_file] V3 pipeline activating for %s (file_tier=%d, req_tier=%d)", input.Path, fileTier, ctx.Tier)
				improved, meta, err := improveContentWithV3(path, newContent, ctx)
				if err != nil {
					log.Printf("[edit_file] V3 failed: %v — falling back to direct write", err)
				} else if improved != "" {
					// V3 sometimes returns code wrapped in markdown
					// fences (the underlying llama-server response had a
					// preamble it didn't strip). Sanitise here too —
					// otherwise every V3-improved file ships with a
					// "Looking at the task..." header on disk.
					if cleanedImproved, sanitized := sanitizeFileContent(input.Path, improved); sanitized {
						log.Printf("[edit_file] sanitised V3 output for %s", input.Path)
						improved = cleanedImproved
					}
					newContent = improved
					v3Out = meta
				}
			}

			// Atomic write
			tmpPath := path + ".atlas.tmp"
			if err := os.WriteFile(tmpPath, []byte(newContent), 0644); err != nil {
				return nil, fmt.Errorf("cannot write %s: %w", input.Path, err)
			}
			if err := os.Rename(tmpPath, path); err != nil {
				os.Remove(tmpPath)
				return nil, fmt.Errorf("cannot rename temp file: %w", err)
			}

			// Update cached state with whatever was actually written
			ctx.RecordFileRead(path, newContent)

			// Build diff preview against the original on-disk content
			oldLines := strings.Count(input.OldStr, "\n") + 1
			newLines := strings.Count(input.NewStr, "\n") + 1
			preview := buildDiffPreview(content, newContent, actualOldStr, input.NewStr)

			out := EditFileOutput{
				OK:           true,
				DiffPreview:  preview,
				LinesAdded:   newLines - oldLines,
				LinesRemoved: 0,
			}
			if newLines < oldLines {
				out.LinesRemoved = oldLines - newLines
				out.LinesAdded = 0
			}

			outBytes, _ := json.Marshal(out)
			result := &ToolResult{Success: true, Data: outBytes}
			if v3Out.Used {
				result.V3Used = true
				result.CandidatesTested = v3Out.CandidatesTested
				result.WinningScore = v3Out.WinningScore
				result.PhaseSolved = v3Out.PhaseSolved
			}
			return result, nil
		},
	}
}

// V3EditMetadata captures what V3 did to an edit_file request, so the
// edit_file result can carry the same v3_used / candidates_tested fields
// write_file does. See PC-042.
type V3EditMetadata struct {
	Used             bool
	CandidatesTested int
	WinningScore     float64
	PhaseSolved      string
}

// improveContentWithV3 sends content through the V3 pipeline and returns
// V3's chosen code (baseline candidate or a better-scoring alternative).
// On error, returns "" + zero metadata; the caller should fall back to
// writing the original content. See PC-042.
func improveContentWithV3(path, content string, ctx *AgentContext) (string, V3EditMetadata, error) {
	req := V3GenerateRequest{
		FilePath:     path,
		BaselineCode: content,
		Tier:         int(ctx.Tier),
		WorkingDir:   ctx.WorkingDir,
	}
	if len(ctx.FilesRead) > 0 {
		req.ProjectContext = make(map[string]string)
		for p, c := range ctx.FilesRead {
			rel, _ := filepath.Rel(ctx.WorkingDir, p)
			if rel == "" {
				rel = p
			}
			if len(c) > 4000 {
				c = c[:4000] + "\n... (truncated)"
			}
			req.ProjectContext[rel] = c
		}
	}
	if ctx.Project != nil {
		req.Framework = ctx.Project.Framework
		req.BuildCommand = ctx.Project.BuildCommand
	}

	// Same callback logic as the write_file V3 path: tokens forward to
	// the dedicated v3_token SSE event so the TUI updates one streaming
	// row instead of spawning a chat row per token; LLM-call boundaries
	// match the chat protocol's start/end shapes; structured stages
	// emit typed events (v3_phase, v3_sandbox, etc.); only truly
	// unknown stages fall back to the v3_progress text line. Without
	// this branching, edit_file with V3 floods the chat pane with
	// thousands of "[token] X" rows during a single candidate generation.
	v3Result, err := callV3GenerateStreaming(ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
		if ctx.StreamFn == nil {
			return
		}
		if stage == "token" {
			ctx.StreamFn("v3_token", map[string]string{"text": detail})
			return
		}
		if stage == "llm_start" {
			payload := map[string]interface{}{"detail": detail}
			for k, v := range data {
				payload[k] = v
			}
			ctx.StreamFn("v3_llm_start", payload)
			return
		}
		if stage == "llm_end" {
			payload := map[string]interface{}{"detail": detail}
			for k, v := range data {
				payload[k] = v
			}
			ctx.StreamFn("v3_llm_end", payload)
			return
		}
		eventName := v3StageToEvent(stage)
		if eventName == "v3_progress" {
			ctx.StreamFn("v3_progress", map[string]string{
				"message": fmt.Sprintf("  │ [%s] %s", stage, detail),
			})
			return
		}
		payload := map[string]interface{}{
			"stage":  stage,
			"detail": detail,
		}
		for k, v := range data {
			payload[k] = v
		}
		ctx.StreamFn(eventName, payload)
	})
	if err != nil {
		return "", V3EditMetadata{}, err
	}

	if ctx.StreamFn != nil {
		ctx.StreamFn("v3_progress", map[string]string{
			"message": fmt.Sprintf("  └──── V3 complete: %s, %d candidates", v3Result.PhaseSolved, v3Result.CandidatesTested),
		})
	}

	chosen := v3Result.Code
	if chosen == "" {
		chosen = content
	}
	return chosen, V3EditMetadata{
		Used:             true,
		CandidatesTested: v3Result.CandidatesTested,
		WinningScore:     v3Result.WinningScore,
		PhaseSolved:      v3Result.PhaseSolved,
	}, nil
}

// findActualString searches for oldStr in content, handling quote normalization.
// Returns the actual string found in content (may differ in quote style).
func findActualString(content, oldStr string) string {
	// Direct match first
	if strings.Contains(content, oldStr) {
		return oldStr
	}

	// Quote normalization: try replacing curly quotes with straight and vice versa
	normalized := normalizeQuotes(oldStr)
	if normalized != oldStr && strings.Contains(content, normalized) {
		return normalized
	}

	// Try the reverse direction
	denormalized := denormalizeQuotes(oldStr)
	if denormalized != oldStr && strings.Contains(content, denormalized) {
		return denormalized
	}

	return ""
}

// normalizeQuotes replaces curly quotes with straight quotes.
func normalizeQuotes(s string) string {
	r := strings.NewReplacer(
		"\u201c", "\"", // left double
		"\u201d", "\"", // right double
		"\u2018", "'",  // left single
		"\u2019", "'",  // right single
	)
	return r.Replace(s)
}

// denormalizeQuotes replaces straight quotes with curly quotes (best-effort).
func denormalizeQuotes(s string) string {
	r := strings.NewReplacer(
		"\"", "\u201c", // straight double → left double (approximate)
		"'", "\u2019",  // straight single → right single (approximate)
	)
	return r.Replace(s)
}

// buildDiffPreview creates a unified-diff-style preview of the edit.
func buildDiffPreview(oldContent, newContent, oldStr, newStr string) string {
	// Find the line number where the change starts
	idx := strings.Index(oldContent, oldStr)
	if idx < 0 {
		return ""
	}
	lineNum := strings.Count(oldContent[:idx], "\n") + 1

	var sb strings.Builder
	fmt.Fprintf(&sb, "@@ line %d @@\n", lineNum)

	// Show removed lines
	for _, line := range strings.Split(oldStr, "\n") {
		fmt.Fprintf(&sb, "- %s\n", line)
	}
	// Show added lines
	for _, line := range strings.Split(newStr, "\n") {
		fmt.Fprintf(&sb, "+ %s\n", line)
	}

	return sb.String()
}

// ---------------------------------------------------------------------------
// delete_file
// ---------------------------------------------------------------------------

func deleteFileTool() *ToolDef {
	return &ToolDef{
		Name:        "delete_file",
		Description: "Delete a file or empty directory. Use for removing files that are no longer needed.",
		InputSchema: DeleteFileInput{},
		ReadOnly:    false,
		Destructive: true,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input DeleteFileInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Reject empty path — same reasoning as read_file (PC-039).
			if strings.TrimSpace(input.Path) == "" {
				return &ToolResult{
					Success: false,
					Error:   "delete_file: path cannot be empty. Provide the path of the file you want to delete.",
				}, nil
			}

			deleted := false

			// Delete from the REAL project directory (where the user's files live)
			if ctx.RealProjectDir != "" {
				realPath := resolvePath(input.Path, ctx.RealProjectDir)
				if info, err := os.Stat(realPath); err == nil {
					if info.IsDir() {
						entries, _ := os.ReadDir(realPath)
						if len(entries) > 0 {
							return nil, fmt.Errorf("directory not empty: %s (%d entries)", input.Path, len(entries))
						}
					}
					os.Remove(realPath)
					deleted = true
					log.Printf("[delete_file] %s deleted from project dir %s", input.Path, ctx.RealProjectDir)
				}
			}

			// Also delete from temp/working dir if it exists there
			path := resolveAgentPath(ctx, input.Path)
			if _, err := os.Stat(path); err == nil {
				os.Remove(path)
				deleted = true
			}

			if !deleted {
				return nil, fmt.Errorf("file not found: %s", input.Path)
			}

			out := DeleteFileOutput{Deleted: true}
			outBytes, _ := json.Marshal(out)
			result := &ToolResult{Success: true, Data: outBytes}
			// Signal the agent loop to stop after deletion — prevents the model
			// from generating follow-up text that would render as a noisy edit
			// suggestion in chat after a destructive operation.
			result.Error = "__FORCE_DONE__"
			return result, nil
		},
	}
}

// ---------------------------------------------------------------------------
// find_file — locate files by NAME (vs search_files which greps contents).
// Added to resolve PC-028: the model would search_files for a filename,
// get zero matches (because contents don't contain the literal filename),
// and conclude the file didn't exist.
// ---------------------------------------------------------------------------

func findFileTool() *ToolDef {
	return &ToolDef{
		Name:        "find_file",
		Description: "Find files by NAME using a regex against the filename or relative path. Use this to check whether a file exists or to locate it. For searching inside file contents, use search_files instead.",
		InputSchema: FindFileInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input FindFileInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Reject empty pattern: it matches every filename, returns the
			// 200-match cap full of unrelated files, and confuses the model
			// into thinking it found nothing useful. See ISSUES.md PC-037.
			if strings.TrimSpace(input.Pattern) == "" {
				return &ToolResult{
					Success: false,
					Error:   "find_file: pattern cannot be empty. Provide a regex matching the filename you want to locate (e.g. \"snake_game\\.py\" or \"^main\\.\").",
				}, nil
			}

			searchPath := ctx.WorkingDir
			if input.Path != "" {
				searchPath = resolveAgentPath(ctx, input.Path)
			}

			re, err := regexp.Compile(input.Pattern)
			if err != nil {
				return nil, fmt.Errorf("invalid regex: %w", err)
			}

			var matches []FindFileMatch
			maxMatches := 200

			err = filepath.WalkDir(searchPath, func(path string, d fs.DirEntry, walkErr error) error {
				if walkErr != nil {
					return nil
				}
				if d.IsDir() {
					base := d.Name()
					if base == ".git" || base == "node_modules" || base == "__pycache__" || base == ".next" || base == "target" {
						return filepath.SkipDir
					}
					return nil
				}
				relPath, _ := filepath.Rel(ctx.WorkingDir, path)
				if relPath == "" {
					relPath = path
				}
				if re.MatchString(d.Name()) || re.MatchString(relPath) {
					matches = append(matches, FindFileMatch{Path: relPath, Name: d.Name()})
					if len(matches) >= maxMatches {
						return filepath.SkipAll
					}
				}
				return nil
			})

			if err != nil && len(matches) == 0 {
				return nil, fmt.Errorf("find error: %w", err)
			}

			out := FindFileOutput{
				Matches:    matches,
				TotalCount: len(matches),
				Truncated:  len(matches) >= maxMatches,
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// run_command
// ---------------------------------------------------------------------------

func runCommandTool() *ToolDef {
	return &ToolDef{
		Name:        "run_command",
		Description: "Execute a shell command. Returns stdout, stderr, and exit code. Use for building, testing, and verifying code.",
		InputSchema: RunCommandInput{},
		ReadOnly:    false,
		Destructive: true,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input RunCommandInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			timeout := 30 * time.Second
			if input.Timeout != nil && *input.Timeout > 0 {
				timeout = time.Duration(*input.Timeout) * time.Second
			}
			if timeout > 5*time.Minute {
				timeout = 5 * time.Minute
			}

			cwd := ctx.WorkingDir
			if input.Cwd != "" {
				cwd = resolveAgentPath(ctx, input.Cwd)
			}

			cmd := exec.Command("bash", "-c", input.Command)
			cmd.Dir = cwd

			// Capture stdout and stderr separately
			var stdout, stderr strings.Builder
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr

			// Run with timeout
			done := make(chan error, 1)
			go func() {
				done <- cmd.Run()
			}()

			var exitCode int
			select {
			case err := <-done:
				if err != nil {
					if exitErr, ok := err.(*exec.ExitError); ok {
						exitCode = exitErr.ExitCode()
					} else {
						return nil, fmt.Errorf("command error: %w", err)
					}
				}
			case <-time.After(timeout):
				if cmd.Process != nil {
					cmd.Process.Kill()
				}
				exitCode = 124 // timeout exit code (like GNU timeout)
				stderr.WriteString(fmt.Sprintf("\nCommand timed out after %s", timeout))
			}

			out := RunCommandOutput{
				Stdout:   truncateStr(stdout.String(), 8000),
				Stderr:   truncateStr(stderr.String(), 4000),
				ExitCode: exitCode,
			}
			outBytes, _ := json.Marshal(out)
			// On failure, surface a useful error string so the agent
			// log line "tool=run_command FAIL: <err>" actually shows
			// what went wrong. Without this the Error field stayed
			// empty, the model saw only the JSON Data payload, and
			// the proxy log read "FAIL:" with no signal — which is
			// exactly what happened in the May 2026 user session.
			// Prefer stderr; fall back to last stdout line; final
			// fallback is the bare exit code.
			var errMsg string
			if exitCode != 0 {
				errMsg = strings.TrimSpace(stderr.String())
				if errMsg == "" {
					if s := strings.TrimSpace(stdout.String()); s != "" {
						lines := strings.Split(s, "\n")
						errMsg = lines[len(lines)-1]
					}
				}
				if errMsg == "" {
					errMsg = fmt.Sprintf("exit %d (no output)", exitCode)
				}
				errMsg = truncateStr(errMsg, 400)
			}
			return &ToolResult{
				Success: exitCode == 0,
				Data:    outBytes,
				Error:   errMsg,
			}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// plan_tasks — orchestration tool for parallel execution
// ---------------------------------------------------------------------------

func planTasksTool() *ToolDef {
	return &ToolDef{
		Name:        "plan_tasks",
		Description: "Decompose work into parallel tasks with dependencies. Independent tasks run concurrently. Use for multi-file project creation.",
		InputSchema: PlanTasksInput{},
		ReadOnly:    false,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input PlanTasksInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Returns pending status — parallel execution is defined in
			// parallel.go (executePlanTasksTool) but not yet wired in
			results := make([]TaskStatus, len(input.Tasks))
			for i, t := range input.Tasks {
				results[i] = TaskStatus{
					ID:     t.ID,
					Status: "pending",
				}
			}

			out := PlanTasksOutput{Results: results}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// Per-file tier classification for V3 pipeline activation
// ---------------------------------------------------------------------------

// classifyFileTier determines whether a specific write_file call should
// route through the V3 pipeline (T2) or write directly (T1).
//
// T1 (direct write): config files, data files, boilerplate, CSS variables,
// JSON data, simple scripts under 30 lines with no complex logic.
//
// T2 (V3 pipeline): files with application logic, multiple functional
// requirements, framework-specific patterns, function definitions,
// event handlers, API logic, state management, conditional branching.
func classifyFileTier(filePath, content string) Tier {
	ext := strings.ToLower(filepath.Ext(filePath))
	base := strings.ToLower(filepath.Base(filePath))
	lines := strings.Count(content, "\n") + 1

	// Always T1: config files by name
	configFiles := []string{
		"package.json", "tsconfig.json", "next.config.js", "next.config.ts",
		"next.config.mjs", "tailwind.config.ts", "tailwind.config.js",
		"postcss.config.js", "postcss.config.mjs", "vite.config.ts",
		"vite.config.js", ".eslintrc.json", ".prettierrc", "jest.config.ts",
		"jest.config.js", "cargo.toml", "go.mod", "go.sum", "makefile",
		"cmakelists.txt", "pyproject.toml", "setup.py", "setup.cfg",
		"requirements.txt", "pipfile", ".editorconfig", ".gitignore",
		"dockerfile", "docker-compose.yml", "docker-compose.yaml",
	}
	for _, cf := range configFiles {
		if base == cf {
			return Tier1Simple
		}
	}

	// Always T1: data files
	dataExts := []string{".json", ".yaml", ".yml", ".toml", ".csv", ".xml", ".env"}
	for _, de := range dataExts {
		if ext == de {
			return Tier1Simple
		}
	}

	// Always T1: CSS/style files
	if ext == ".css" || ext == ".scss" || ext == ".less" {
		return Tier1Simple
	}

	// Always T1: markdown, text
	if ext == ".md" || ext == ".txt" || ext == ".rst" {
		return Tier1Simple
	}

	// Always T1: shell scripts (usually boilerplate)
	if ext == ".sh" || ext == ".bash" {
		return Tier1Simple
	}

	// Trivially tiny files → T1 always. Below 10 lines there's nothing
	// for V3 to meaningfully diversify on (the prior 50-line floor was
	// too conservative — flask app.py with 7 routes is 33 lines and is
	// exactly the kind of file V3 should help with).
	if lines < 10 {
		return Tier1Simple
	}

	// Code files with any application logic → T2. Lower threshold than
	// before to catch small-but-routed files (flask blueprints, express
	// routers, etc.) that the previous 3-indicator rule missed.
	if hasLogicIndicators(content) {
		return Tier2Medium
	}

	// Source-code and markup extensions get the benefit of the doubt
	// at T2 even without obvious logic-pattern matches — naming a file
	// foo.py / foo.go / foo.html is itself a strong signal that V3's
	// diverse candidate generation is worth the cost. HTML / JSX
	// templates used to require ≥150 lines to clear the markup branch,
	// which made V3 silent on every typical flask/express template
	// (usually 30–120 lines). Now any file at ≥10 lines with a
	// recognized code/markup extension goes T2.
	codeExts := map[string]bool{
		".py": true, ".go": true, ".rs": true,
		".ts": true, ".tsx": true, ".js": true, ".jsx": true,
		".c": true, ".cpp": true, ".cc": true, ".h": true, ".hpp": true,
		".java": true, ".kt": true, ".swift": true,
		".rb": true, ".php": true,
		".vue": true, ".svelte": true,
		".html": true, ".htm": true,
	}
	if codeExts[ext] {
		return Tier2Medium
	}

	// Default: T1 for unknown extensions / pure markup we're not sure about.
	return Tier1Simple
}

// hasLogicIndicators checks if content contains signs of real application logic
// that would benefit from V3 pipeline's diverse candidate generation.
func hasLogicIndicators(content string) bool {
	// Count logic indicators
	indicators := 0
	logicPatterns := []string{
		// Function/method definitions
		"def ", "func ", "function ", "fn ", "async ",
		// Control flow
		"if ", "else ", "switch ", "match ", "for ", "while ",
		// Error handling
		"try ", "catch ", "except ", "throw ", "raise ",
		// Flask / FastAPI / Django routing — was missing before, which
		// caused a 33-line app.py with 7 @app.route handlers to register
		// only one indicator ("def ") and fall through to T1.
		"@app.route", "@app.get", "@app.post", "@app.put", "@app.delete",
		"@blueprint", "render_template", "url_for", "request.method",
		"flask.", "from flask",
		// Express / Node API patterns
		"export default", "export async", "module.exports",
		"app.get", "app.post", "app.put", "app.delete",
		"router.", "handler",
		"NextResponse", "Response(", "Request",
		// State/data management
		"useState", "useEffect", "useRef", "useCallback",
		"setState", "dispatch", "reducer",
		// Validation
		"validate", "schema", "parse", "zod.",
		// Database
		"query(", "insert(", ".select(", ".update(",
		// JSX / React component patterns
		"return (", "return <",
		"className=", "onClick", "onChange", "onSubmit",
		".map(", ".filter(", ".reduce(",
		// Multiple imports (sign of real component)
		"import {",
	}

	for _, p := range logicPatterns {
		if strings.Contains(content, p) {
			indicators++
		}
	}

	// 2+ logic indicators → has real application logic. Lowered from 3
	// because the original threshold was tuned for large files and
	// caused small-but-real apps (e.g. a flask routing module) to slip
	// through to T1 even though V3 would have helped.
	return indicators >= 2
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// resolvePath resolves a relative path against the working directory.
//
// Absolute paths pass through unchanged — but the model frequently
// emits the host-side absolute path it saw in the user's prompt
// (e.g. "/home/isaac/snake/app.py") which doesn't exist inside the
// proxy container. Use resolveAgentPath when you have an
// AgentContext available — it translates host paths to container
// paths via ctx.HostWorkingDir. resolvePath is the lower-level
// primitive kept for sites that don't have a context (e.g. V3
// adapter helpers).
func resolvePath(path, workingDir string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(workingDir, path))
}

// resolveAgentPath is the path resolver every tool handler should
// use. It first translates host-side absolute paths into the
// container path (when HostWorkingDir is set and the input falls
// inside that prefix), then resolves the result against
// ctx.WorkingDir. This is what makes the agent forgiving when the
// user pastes "/home/isaac/snake/app.py" into a prompt — the model
// copies the absolute path, the proxy rewrites it to /workspace/app.py,
// and read_file actually finds the file.
func resolveAgentPath(ctx *AgentContext, path string) string {
	if filepath.IsAbs(path) && ctx.HostWorkingDir != "" {
		clean := filepath.Clean(path)
		host := filepath.Clean(ctx.HostWorkingDir)
		if clean == host {
			return filepath.Clean(ctx.WorkingDir)
		}
		// Match `host` as a directory prefix — require the next character
		// to be a separator so "/home/isaac/snakebar" doesn't match
		// "/home/isaac/snake".
		if strings.HasPrefix(clean, host+string(filepath.Separator)) {
			rel := strings.TrimPrefix(clean, host+string(filepath.Separator))
			translated := filepath.Join(ctx.WorkingDir, rel)
			return filepath.Clean(translated)
		}
	}
	return resolvePath(path, ctx.WorkingDir)
}

// v3StageToEvent maps a V3 pipeline stage name to the TUI event type
// it should fire. Stages cluster by phase: PlanSearch / DivSampling /
// Sandbox / S* / Phase 3 each get a dedicated event type so the TUI can
// render specialized rows (counters, per-test results, strategy choice)
// instead of a generic "v3_progress" string. Unknown stages fall back
// to v3_progress.
//
// Names are intentionally short — they cross the SSE wire on every
// pipeline stage transition (a typical T2 run emits 15–30 of them).
func v3StageToEvent(stage string) string {
	switch stage {
	case "phase1", "phase2", "phase2_allocated":
		return "v3_phase"
	case "plansearch", "plansearch_done", "plansearch_error":
		return "v3_plansearch"
	case "divsampling", "divsampling_done", "divsampling_error":
		return "v3_divsampling"
	case "sandbox_test", "sandbox_pass", "sandbox_fail", "sandbox_done":
		return "v3_sandbox"
	case "s_star", "s_star_winner", "s_star_error", "selected":
		return "v3_select"
	case "phase3", "pr_cot", "pr_cot_pass", "pr_cot_failed", "pr_cot_error",
		"refinement", "refinement_pass", "refinement_failed", "refinement_error",
		"derivation", "derivation_pass", "derivation_failed", "derivation_error",
		"fallback":
		return "v3_repair"
	case "probe", "probe_light", "probe_retry", "probe_failed",
		"probe_scored", "probe_sandbox", "probe_pass", "probe_error":
		return "v3_probe"
	case "self_test_gen", "self_test_done", "self_test_error",
		"self_test_skip", "self_test_verify":
		return "v3_self_test"
	case "plan_start", "plan_candidate", "plan_candidate_unparseable",
		"plan_candidate_error", "plan_candidate_scored", "plan_selected",
		"plan_failed":
		// All plan-pipeline stages collapse to one TUI event family. The
		// TUI reads the stage name off the payload to decide between
		// "scoring..." spinner, "winner: plan #N" summary, etc.
		return "v3_plan"
	}
	return "v3_progress"
}

// truncateStr limits a string to maxLen characters.
// (truncateStr() already exists in main.go for backward compat)
func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// firstNonEmptyLine returns the first non-blank line of s, trimmed of trailing
// whitespace. Used to surface a one-line hint from a tool's stderr without
// dumping the whole buffer to the UI.
func firstNonEmptyLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimRight(line, " \t\r")
		if strings.TrimSpace(trimmed) != "" {
			return trimmed
		}
	}
	return ""
}
