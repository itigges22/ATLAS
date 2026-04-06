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

	result, err := tool.Execute(args, ctx)
	if err != nil {
		errMsg := err.Error()
		if strings.Contains(errMsg, "unexpected end of JSON") || strings.Contains(errMsg, "invalid input") {
			errMsg = "Tool call was truncated (output too long for context window). Use smaller, targeted edit_file calls instead of full write_file rewrites."
		}
		return &ToolResult{
			Success: false,
			Error:   errMsg,
		}
	}
	return result
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

			path := resolvePath(input.Path, ctx.WorkingDir)

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
		Description: "Search for a regex pattern in files. Returns matching lines with file paths and line numbers. Use glob to filter by file type.",
		InputSchema: SearchFilesInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input SearchFilesInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			searchPath := ctx.WorkingDir
			if input.Path != "" {
				searchPath = resolvePath(input.Path, ctx.WorkingDir)
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

			dirPath := resolvePath(input.Path, ctx.WorkingDir)

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

			path := resolvePath(input.Path, ctx.WorkingDir)

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

// writeFileDirect records content for Aider delivery.
// Does NOT write to disk — Aider handles file writing from the whole-file
// blocks in the SSE response. The proxy just needs to track what files
// were "written" so formatForAider can generate the response.
func writeFileDirect(path, content string) (*ToolResult, error) {
	out := WriteFileOutput{BytesWritten: len(content)}
	outBytes, _ := json.Marshal(out)
	return &ToolResult{Success: true, Data: outBytes}, nil
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

	// Call V3 service with streaming progress
	v3Result, err := callV3GenerateStreaming(ctx.V3URL, req, func(stage, detail string) {
		// Forward V3 pipeline progress to Aider via the agent loop's stream
		if ctx.StreamFn != nil {
			msg := fmt.Sprintf("  \u2502 [%s] %s", stage, detail)
			ctx.StreamFn("v3_progress", map[string]string{"message": msg})
		}
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

			path := resolvePath(input.Path, ctx.WorkingDir)

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

			// For T2/T3 with large changes (>20 lines in new_str), could route through V3
			// For now, apply directly — V3 routing for large edits is a future optimization
			var newContent string
			if input.ReplaceAll {
				newContent = strings.ReplaceAll(content, actualOldStr, input.NewStr)
			} else {
				newContent = strings.Replace(content, actualOldStr, input.NewStr, 1)
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

			// Update cached state
			ctx.RecordFileRead(path, newContent)

			// Build diff preview
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
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
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

			deleted := false

			// Delete from the REAL project directory (where Aider's files live)
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
			path := resolvePath(input.Path, ctx.WorkingDir)
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
			// from generating follow-up text that Aider could misinterpret as a file edit
			result.Error = "__FORCE_DONE__"
			return result, nil
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
				cwd = resolvePath(input.Cwd, ctx.WorkingDir)
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
			return &ToolResult{
				Success: exitCode == 0,
				Data:    outBytes,
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

			// Placeholder — parallel execution implemented in parallel.go
			// For now, return acknowledgment
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

	// Short files → T1 always. V3 pipeline can't meaningfully improve
	// a file under 50 lines — the overhead (3-5 min) vastly exceeds any
	// quality gain from diverse candidate generation on small files.
	if lines < 50 {
		return Tier1Simple
	}

	// Larger files with application logic → T2
	if hasLogicIndicators(content) {
		return Tier2Medium
	}

	// Default: T1 for anything we're not sure about
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
		// API/handler patterns
		"export default", "export async", "module.exports",
		"app.get", "app.post", "router.", "handler",
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

	// 3+ logic indicators → has real application logic
	return indicators >= 3
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// resolvePath resolves a relative path against the working directory.
func resolvePath(path, workingDir string) string {
	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}
	return filepath.Clean(filepath.Join(workingDir, path))
}

// truncateStr limits a string to maxLen characters.
// (truncateStr() already exists in main.go for backward compat)
func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
