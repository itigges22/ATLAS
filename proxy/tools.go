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
	"net/http"
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
	registerTool(outlineFileTool())
	registerTool(writeFileTool())
	registerTool(editFileTool())
	registerTool(astEditTool())
	registerTool(deleteFileTool())
	registerTool(moveFileTool())
	registerTool(runCommandTool())
	registerTool(searchFilesTool())
	registerTool(findFileTool())
	registerTool(listDirectoryTool())
	registerTool(runBackgroundTool())
	registerTool(tailBackgroundTool())
	registerTool(stopBackgroundTool())
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
	if reason := validateToolWorkspacePaths(name, args, ctx); reason != "" {
		return &ToolResult{Success: false, Error: reason}
	}
	// Safety deny-list — sensitive targets (.env, *.pem, *credentials*,
	// destructive shell patterns) are refused in every permission mode.
	if denied, reason := shouldDenyToolCall(name, args); denied {
		return &ToolResult{Success: false, Error: fmt.Sprintf("%s refused: %s", name, reason)}
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
	case "move_file":
		return `move_file: no arguments provided. Call with {"source":"<current path>","destination":"<new path or dir>"}.`
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
			// PC-194 — register the read so the pattern-matching gate
			// on write_file knows the model has actually inspected a
			// sibling before generating a new file in the same dir.
			patternReadTracker.add(path)

			// Call-graph footer (issue #39, flag-gated). The model reads a
			// file far more often than it outlines one, so attach the
			// intra-file call edges to a .py read where the localization
			// decision happens. Fire on any read that starts at the top of
			// the file (start == 0) — that covers both a whole-file read and
			// the model's common "offset:0, limit:N" first look; it computes
			// the graph from the full file on disk regardless of the page
			// shown, and skips mid-file pages so a model scrolling a big file
			// doesn't get the footer repeated.
			if start == 0 && strings.HasSuffix(input.Path, ".py") && callGraphEnabled() {
				if footer := callGraphFooter(ctx, input.Path, string(data)); footer != "" {
					content += footer
				}
			}

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
// outline_file — cheap structural index of a file (names + line ranges, no
// bodies). The surgical-read entry point: outline to see what's in a file
// for a few hundred bytes, then read_file with offset/limit to pull just
// the one function you need. Saves the model from dumping a whole file into
// context (and re-reading it) just to find one bug. GH #39.
// ---------------------------------------------------------------------------

func outlineFileTool() *ToolDef {
	return &ToolDef{
		Name: "outline_file",
		Description: "List a file's top-level functions and classes with their " +
			"line ranges — NO bodies, so it costs almost no context. Use this " +
			"FIRST to navigate an existing file instead of reading the whole " +
			"thing: outline_file to find the function you care about, then " +
			"read_file with offset/limit to read just its lines, then ast_edit " +
			"(selector function:NAME / class:NAME) or edit_file to change it. " +
			"Python is parsed precisely (tree-sitter, decorator-aware); other " +
			"languages get a best-effort definition scan.",
		InputSchema: OutlineInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input OutlineInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}
			if strings.TrimSpace(input.Path) == "" {
				return &ToolResult{Success: false,
					Error: "outline_file: path cannot be empty. Pass {\"path\":\"<file>\"}."}, nil
			}
			path := resolveAgentPath(ctx, input.Path)
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, fmt.Errorf("cannot read %s: %w", input.Path, err)
			}
			src := string(data)
			totalLines := strings.Count(src, "\n") + 1

			// Prefer the v3 tree-sitter outline for .py (accurate, matches
			// ast_edit selectors). Fall back to a language-agnostic regex
			// scan for everything else and whenever v3 is unavailable.
			var syms []OutlineSymbol
			if strings.HasSuffix(input.Path, ".py") {
				if v3, ok := outlineViaV3(ctx, input.Path, src); ok {
					syms = v3
				}
			}
			engine := "tree-sitter"
			if syms == nil {
				syms = outlineByRegex(input.Path, src)
				engine = "scan"
			}

			ctx.RecordFileRead(path, src)
			patternReadTracker.add(path)

			var sb strings.Builder
			fmt.Fprintf(&sb, "%s — %d lines, %d symbols (%s)\n",
				input.Path, totalLines, len(syms), engine)
			if len(syms) == 0 {
				sb.WriteString("(no top-level functions/classes found — read_file to view it directly)\n")
			}
			hasGraph := false
			for _, s := range syms {
				fmt.Fprintf(&sb, "L%d-%d\t%s %s\n", s.StartLine, s.EndLine, s.Kind, s.Name)
				if len(s.Calls) > 0 {
					fmt.Fprintf(&sb, "\tcalls: %s\n", strings.Join(s.Calls, ", "))
					hasGraph = true
				}
				if len(s.CalledBy) > 0 {
					fmt.Fprintf(&sb, "\tcalled by: %s\n", strings.Join(s.CalledBy, ", "))
					hasGraph = true
				}
			}
			if hasGraph {
				// Steer the model to use the structure for localization — the
				// #39 point: a wrong value a function returns may come from a
				// function it calls, not from the function itself.
				sb.WriteString("\nNote: if a function returns a wrong value, the bug may be in a function it `calls`, not in the function itself — follow the call edges to the root cause before editing.\n")
			}
			out := OutlineOutput{Symbols: syms, Supported: len(syms) > 0, Outline: sb.String()}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// callGraphEnabled mirrors v3-service's flag so the proxy can skip the extra
// outline round-trip on the read_file path when the feature is off. Forwarded
// to the proxy container via docker-compose (issue #39).
func callGraphEnabled() bool {
	v := strings.TrimSpace(os.Getenv("ATLAS_CALL_GRAPH"))
	return v != "" && v != "0" && strings.ToLower(v) != "false"
}

// callGraphFooter renders a compact intra-file call-graph summary for a
// whole-file read, reusing the same v3 outline (which carries calls/called_by
// when ATLAS_CALL_GRAPH is on). Returns "" when there are no edges, so a file
// with no internal calls doesn't get a noisy empty section.
func callGraphFooter(ctx *AgentContext, path, source string) string {
	syms, ok := outlineViaV3(ctx, path, source)
	if !ok {
		return ""
	}
	var sb strings.Builder
	any := false
	for _, s := range syms {
		if len(s.Calls) == 0 && len(s.CalledBy) == 0 {
			continue
		}
		if !any {
			sb.WriteString("\n\n## Call graph (within this file)\n")
			any = true
		}
		sb.WriteString("- " + s.Name)
		if len(s.Calls) > 0 {
			sb.WriteString(" calls: " + strings.Join(s.Calls, ", "))
		}
		if len(s.CalledBy) > 0 {
			sb.WriteString("; called by: " + strings.Join(s.CalledBy, ", "))
		}
		sb.WriteString("\n")
	}
	if any {
		sb.WriteString("If a function returns a wrong value, the bug may be in a function it calls — follow the edges to the root cause before editing.\n")
	}
	return sb.String()
}

// outlineViaV3 asks v3-service for a tree-sitter outline. Returns (nil,false)
// on any failure so the caller can fall back to the regex scan.
func outlineViaV3(ctx *AgentContext, path, source string) ([]OutlineSymbol, bool) {
	if ctx.V3URL == "" {
		return nil, false
	}
	body, _ := json.Marshal(map[string]string{"path": path, "source": source})
	req, err := http.NewRequestWithContext(ctx.Ctx, "POST",
		ctx.V3URL+"/internal/outline", bytes.NewReader(body))
	if err != nil {
		return nil, false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, false
	}
	var out OutlineOutput
	if json.NewDecoder(resp.Body).Decode(&out) != nil || !out.Supported {
		return nil, false
	}
	return out.Symbols, true
}

// outlineByRegex is the language-agnostic fallback: any line starting (at
// column 0, allowing common keywords) with a definition. Good enough to give
// the model line anchors to read from; not as precise as tree-sitter.
func outlineByRegex(path, source string) []OutlineSymbol {
	type pat struct {
		re   *regexp.Regexp
		kind string
	}
	// Ordered, language-agnostic-ish. Name is capture group 1.
	pats := []pat{
		{regexp.MustCompile(`^(?:async\s+)?def\s+([A-Za-z_]\w*)`), "function"},
		{regexp.MustCompile(`^class\s+([A-Za-z_]\w*)`), "class"},
		{regexp.MustCompile(`^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)`), "function"},
		{regexp.MustCompile(`^type\s+([A-Za-z_]\w*)`), "type"},
		{regexp.MustCompile(`^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)`), "function"},
		{regexp.MustCompile(`^(?:export\s+)?class\s+([A-Za-z_$][\w$]*)`), "class"},
	}
	lines := strings.Split(source, "\n")
	var out []OutlineSymbol
	for i, ln := range lines {
		for _, p := range pats {
			if m := p.re.FindStringSubmatch(ln); m != nil {
				out = append(out, OutlineSymbol{
					Name: m[1], Kind: p.kind,
					StartLine: i + 1, EndLine: i + 1, // single-line anchor; body follows
				})
				break
			}
		}
	}
	return out
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
		Name: "write_file",
		Description: "Create a NEW file from scratch. Creates parent directories if needed. " +
			"DO NOT use to overwrite existing files — for existing files use ast_edit (whole function/class/element rewrite) or edit_file (≤10-line surgical change). " +
			"If a write_file call is rejected because the path already exists, switch to ast_edit (whole-block rewrite) or edit_file (surgical change). DO NOT retry with edit_file simply because the file is large.",
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

			// PC-194 — pattern-matching reflex. When the model creates a
			// NEW file in a non-empty directory of similar files (HTML
			// alongside HTML, route handler alongside route handlers),
			// nudge it to read a sibling first instead of generating
			// content from scratch. Only fires for genuinely-new files
			// to avoid breaking edits-via-write_file. Soft hint via
			// tool result, not a hard reject — the model can ignore it
			// if the content is clearly intentional.
			if hint := patternMatchHint(path, input.Content); hint != "" {
				return &ToolResult{Success: false, Error: hint}, nil
			}

			// PC-195 — stub detection. Reject "<h1>X Page</h1>" / "TODO"
			// placeholder writes that pass syntactic gates but ship the
			// minimum content humanly possible. The model's lazy-completion
			// failure mode is to write 8-line stubs and call it done; this
			// gate forces it to either commit real content or acknowledge
			// the stub explicitly. New files only — edits to existing
			// files might legitimately shrink to a stub via refactor.
			if isNewWrite(path) {
				if reason := looksLikeStub(input.Path, input.Content); reason != "" {
					return &ToolResult{Success: false, Error: reason}, nil
				}
			}

			// Per-file tier classification — determines V3 pipeline activation
			fileTier := classifyFileTier(input.Path, input.Content)
			// GH #39 point 2: real cyclomatic complexity from tree-sitter
			// can escalate the regex classifier's verdict. Never downgrades.
			if cc, ok := cyclomaticComplexity(ctx, input.Path, input.Content); ok {
				if refined := refineTierWithCC(fileTier, cc); refined != fileTier {
					log.Printf("[write_file] %s tier %s→%s via cc=%d", input.Path, fileTier, refined, cc)
					fileTier = refined
				} else {
					log.Printf("[write_file] %s cc=%d (tier %s unchanged)", input.Path, cc, fileTier)
				}
			}
			log.Printf("[write_file] %s → %s (%d lines)", input.Path, fileTier, strings.Count(input.Content, "\n")+1)

			// V3 pipeline fires on T2+ files when V3 service is available.
			// V3 takes the model's content as baseline candidate, generates diverse
			// alternatives via PlanSearch/DivSampling, build-verifies each, and
			// selects the best. This is the intelligence layer.
			if fileTier >= Tier2Medium && ctx.V3URL != "" && !ctx.BypassV3 {
				log.Printf("[write_file] V3 pipeline activating for %s", input.Path)
				res, err := writeFileWithV3(path, input.Content, ctx)
				if err == nil && res != nil && res.Success {
					ctx.SessionWrites[input.Path] = true
				}
				return res, err
			}
			if ctx.BypassV3 {
				log.Printf("[write_file] V3 bypassed (demo baseline pane) — direct write %s", input.Path)
			}

			// T1: Direct write — config, data, boilerplate
			res, err := writeFileDirect(path, input.Content)
			if err == nil && res != nil && res.Success {
				ctx.SessionWrites[input.Path] = true
			}
			return res, err
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
	if filesRead := ctx.SnapshotFilesRead(); len(filesRead) > 0 {
		req.ProjectContext = make(map[string]string)
		for p, content := range filesRead {
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

	// Files this session WROTE are project context too — previously only
	// files the model had read were included, so written-but-never-read
	// siblings were invisible to candidate generation (2026-07-18: V3
	// generated an app.py blind to the session's own templates/index.html
	// and static/game.js, and the winner inlined its page, orphaning
	// both). Disk content wins over any stale read snapshot: the on-disk
	// version is what verification produced.
	for rel := range ctx.SessionWrites {
		if rel == "" {
			continue
		}
		abs := resolveAgentPath(ctx, rel)
		if abs == path {
			continue // the file being generated is the baseline, not context
		}
		data, rerr := os.ReadFile(abs)
		if rerr != nil {
			continue
		}
		content := string(data)
		if len(content) > 4000 {
			content = content[:4000] + "\n... (truncated)"
		}
		if req.ProjectContext == nil {
			req.ProjectContext = make(map[string]string)
		}
		req.ProjectContext[rel] = content
	}

	// Add project info if available
	if ctx.Project != nil {
		req.Framework = ctx.Project.Framework
		req.BuildCommand = ctx.Project.BuildCommand
	}

	// V3.2 RPG (issue #120): if this file maps to an RPG node, thread the
	// node's planned interface (signatures, input/output edges) into the
	// generation request. Empty for the flat planner.
	req.Constraints = planConstraintsForTarget(ctx, path)

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
	v3Result, err := callV3GenerateStreaming(ctx.Ctx, ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
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
		// Reasoning deltas during V3 LLM calls (<think>
		// stream). Forwarded as v3_reasoning_token (NOT plain
		// reasoning_token) because the agent-loop's reasoning_token
		// handler in the TUI targets the agent's LLM row — V3 calls
		// run in a different lifecycle and need their own pipe into
		// the V3 streaming row. Gives the /demo viewer something to
		// watch during long PlanSearch / repair phases.
		if stage == "reasoning_token" {
			if ctx.StreamFn != nil {
				ctx.StreamFn("v3_reasoning_token", map[string]string{"text": detail})
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
		// User cancellation is not a fallback case — the turn was aborted,
		// so nothing should land on disk.
		if errors.Is(err, context.Canceled) || (ctx.Ctx != nil && ctx.Ctx.Err() != nil) {
			log.Printf("[write_file] V3 aborted by cancellation — not writing %s", path)
			return &ToolResult{
				Success: false,
				Error:   "write_file cancelled — no content was written",
			}, nil
		}
		// Fallback to direct write if V3 service unavailable
		log.Printf("[write_file] V3 failed: %s — falling back to direct write", err)
		msg := "  \u2514\u2500 V3 unavailable, writing directly"
		if errors.Is(err, context.DeadlineExceeded) {
			msg = fmt.Sprintf("  \u2514\u2500 V3 exceeded %s cap, writing your version", v3CallTimeout())
		}
		ctx.Stream("text", map[string]string{"content": msg})
		return writeFileDirect(path, baselineContent)
	}

	// V3.2 RPG (issue #120): automatic node-local regeneration on drift. When
	// the winning candidate missed its planned signatures, retry once with the
	// missing signatures injected as a hard constraint before accepting it.
	// No-op when RPG is off (req.Constraints empty) or there was no drift.
	v3Result = regenerateOnDrift(ctx, req, v3Result)
	// The regen is a second full V3 run — recheck cancellation before the
	// disk write, mirroring the main call's abort path above: a user cancel
	// during the retry window must not land content on disk.
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		log.Printf("[write_file] cancelled during RPG regen — not writing %s", path)
		return &ToolResult{
			Success: false,
			Error:   "write_file cancelled — no content was written",
		}, nil
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
		BytesWritten:         len(code),
		V3Used:               true,
		CandidatesTested:     v3Result.CandidatesTested,
		WinningScore:         v3Result.WinningScore,
		PhaseSolved:          v3Result.PhaseSolved,
		VerificationEvidence: v3Result.VerificationEvidence,
	}
	outBytes, _ := json.Marshal(out)
	result.Data = outBytes
	result.V3Used = true
	result.CandidatesTested = v3Result.CandidatesTested
	result.WinningScore = v3Result.WinningScore
	result.PhaseSolved = v3Result.PhaseSolved
	result.VerificationEvidence = v3Result.VerificationEvidence

	// V3.2 RPG drift loop (issue #120): if the winning code failed to realize
	// the node's planned signatures, surface the drift + downstream subgraph.
	// Gated on req.Constraints, not only the response field: RPG is active
	// for this file exactly when the plan supplied constraints, so a rogue
	// or stale v3-service emitting rpg_signature_missing unconditionally
	// can't fire drift events while the flag is off.
	if len(req.Constraints) > 0 && len(v3Result.RPGSignatureMissing) > 0 {
		reportRPGDrift(ctx, path, v3Result.RPGSignatureMissing)
	}

	return result, nil
}

// ---------------------------------------------------------------------------
// edit_file — old_str/new_str with uniqueness validation
// ---------------------------------------------------------------------------

func editFileTool() *ToolDef {
	return &ToolDef{
		Name: "edit_file",
		Description: "SURGICAL inline string replacement, ONLY. Use ONLY when changing a few lines inside a function (a None check, a regex, a constant). " +
			"DO NOT use for whole-function rewrites, whole-class rewrites, whole-file replacements, or any change >10 lines — for those, use ast_edit (named node) or write_file (new file). " +
			"old_str must match exactly once (or replace_all=true). Always read_file before editing. " +
			"Heuristic: if you're tempted to copy a whole function/class/HTML element into old_str, you have the wrong tool — switch to ast_edit.",
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
				// GH #39: model occasionally HTML-entity-encodes < > &
				// inside JSON tool-call args (a recurring small-model quirk). When the
				// disk has literal angle brackets, findActualString
				// misses. Try once with entities decoded — if the
				// decoded form matches the file, accept it
				// transparently (also decode new_str so the
				// replacement preserves intent). Faster than burning a
				// turn on the corrective and matches what the model
				// almost certainly meant. If decoded still doesn't
				// match (or there were no entities to decode), fall
				// through to the targeted error so the model knows
				// what's wrong.
				hasEntities := strings.Contains(input.OldStr, "&lt;") ||
					strings.Contains(input.OldStr, "&gt;") ||
					strings.Contains(input.OldStr, "&amp;")
				if hasEntities {
					decoder := strings.NewReplacer(
						"&lt;", "<",
						"&gt;", ">",
						"&amp;", "&",
					)
					decodedOld := decoder.Replace(input.OldStr)
					if maybeMatch := findActualString(content, decodedOld); maybeMatch != "" {
						log.Printf("[edit_file] auto-decoded HTML entities in old_str of %s — proceeding with decoded match (saved a stuck-loop turn)", input.Path)
						input.OldStr = decodedOld
						input.NewStr = decoder.Replace(input.NewStr)
						actualOldStr = maybeMatch
					}
				}
			}
			if actualOldStr == "" {
				// Last resort before failing: whitespace-tolerant line match.
				// Small models can't reproduce a code block byte-for-byte —
				// indentation width, tabs-vs-spaces, and trailing spaces drift
				// constantly — so exact old_str matching is the #1 reason
				// edit_file fails for them, which strands the whole V3/lens
				// stack (it only engages after a successful edit). This
				// matches old_str against the file ignoring per-line leading/
				// trailing whitespace, and ONLY accepts a UNIQUE match — an
				// ambiguous or content-different old_str still fails, so it
				// can never silently edit the wrong place. Exact-matching
				// (frontier) models never reach this path. See GH #39.
				if fuzzy, ok := findFuzzyLineMatch(content, input.OldStr); ok {
					log.Printf("[edit_file] exact old_str missed on %s; unique whitespace-tolerant match found — proceeding (small-model indentation drift)", input.Path)
					actualOldStr = fuzzy
				}
			}
			if actualOldStr == "" {
				// Mismatch persists — return targeted error.
				hasEntities := strings.Contains(input.OldStr, "&lt;") ||
					strings.Contains(input.OldStr, "&gt;") ||
					strings.Contains(input.OldStr, "&amp;")
				literalsOnDisk := strings.ContainsAny(content, "<>&")
				if hasEntities && literalsOnDisk {
					ext := strings.ToLower(filepath.Ext(input.Path))
					alt := ""
					if ext == ".html" || ext == ".htm" || ext == ".py" {
						alt = " For whole-element rewrites, ast_edit is the cleaner option — it takes a selector (e.g. `<body>`, `function:NAME`) and the new content body, no old_str needed."
					}
					return nil, fmt.Errorf("string to replace not found in file. Your `old_str` contains HTML-entity-encoded characters (`&lt;` / `&gt;` / `&amp;`) but the file on disk has literal `<` / `>` / `&`. Re-emit `old_str` with literal angle brackets — JSON strings should contain literal `<` not `&lt;`.%s\nSearched for: %s",
						alt, truncateStr(input.OldStr, 200))
				}
				// Generic mismatch — the model's old_str doesn't byte-match
				// the file (whitespace, quotes, or paraphrase drift, which
				// smaller models do constantly). For structured files,
				// ast_edit sidesteps the whole problem: it selects the node
				// by name, no old_str to reproduce exactly. Steer there.
				ext := strings.ToLower(filepath.Ext(input.Path))
				astAlt := ""
				if ext == ".py" || ext == ".html" || ext == ".htm" {
					astAlt = " To replace a whole function/class/element without " +
						"matching exact text, use ast_edit with a selector " +
						"(e.g. `function:NAME`, `class:NAME`, `<body>`) and the " +
						"new content — no old_str needed, so a near-miss can't fail it."
				}
				// Ground the retry in the file's REAL content. A small model
				// frequently writes old_str from its memory of the file
				// rather than the file itself (observed: old_str
				// `item = items[id + 1]` for a file whose actual line is
				// `return jsonify(items[item_id + 1])`). Without an anchor it
				// gives up on the surgical edit and rewrites the whole node
				// from the same faulty memory. Quoting the closest actual
				// line gives it real bytes to copy into the next attempt.
				hint := closestLineHint(content, input.OldStr)
				return nil, fmt.Errorf("string to replace not found in file. old_str must match the file byte-for-byte (whitespace and quotes included).%s%s\nSearched for: %s", astAlt, hint, truncateStr(input.OldStr, 200))
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

			// Shrinkage guard — same shape as ast_edit's. Catches the
			// "model emitted a stub for old_str" failure where new_str
			// is implausibly tiny for a substantial old_str. We compare
			// the local replacement footprint (old_str → new_str), not
			// whole-file size, so refactors that genuinely shrink the
			// file aren't false-rejected.
			if rejection := validateNotSuspiciouslyShrunk("edit_file", input.Path, len(actualOldStr), len(input.NewStr)); rejection != "" {
				log.Printf("[edit_file] rejecting suspicious shrinkage: %s old_str=%dB new_str=%dB",
					input.Path, len(actualOldStr), len(input.NewStr))
				return &ToolResult{Success: false, Error: rejection}, nil
			}

			// No-op guard — same rationale as ast_edit's. new_str identical
			// to old_str (or a replacement that leaves the file unchanged)
			// must not report success: the model believes the fix landed
			// and moves on while the bug is still on disk.
			if newContent == content {
				log.Printf("[edit_file] no-op edit rejected for %s — file content unchanged", input.Path)
				return &ToolResult{Success: false, Error: "edit_file: new_str is identical to old_str — nothing was changed and the bug is still there. " +
					"Look at the current code again and emit a new_str that actually differs from the existing code."}, nil
			}

			// Syntax gate — the edit_file counterpart of ast_edit's
			// post-splice compile check. A garbage-quoted new_str (doubled
			// quotes, stray escapes) otherwise lands on disk and turns a
			// runnable .py file into a SyntaxError. Best-effort: when the
			// v3-service is unreachable or busy the check is skipped rather
			// than blocking the edit.
			if strings.ToLower(filepath.Ext(input.Path)) == ".py" {
				if ok, perr := pycheckViaV3(ctx, input.Path, newContent); !ok {
					log.Printf("[edit_file] syntax gate rejected edit to %s: %s", input.Path, perr)
					return &ToolResult{Success: false, Error: fmt.Sprintf(
						"edit_file: this edit would make %s invalid Python — %s. The file was NOT modified. "+
							"Check your quoting in new_str and try again.", input.Path, perr)}, nil
				}
			}

			// Route through V3 pipeline when the file warrants it. The
			// gate now mirrors write_file (file-tier only, no request-tier
			// AND-gate) — having two separate tier checks meant V3 only
			// fired when both classifiers happened to agree, which was
			// rare in practice. V3 takes the post-edit content as
			// baseline candidate #0; if its diverse alternatives
			// build-verify better, V3 wins; otherwise the baseline (=our
			// edit) wins. Either way the answer is build-verified.
			//
			// May 10 2026: classify on max(oldTier, newTier) so a
			// destructive edit that shrinks a T2+ file into a T1 stub
			// still triggers V3. Without max-tier, the very edits that
			// most need quality-checking were silently bypassing the
			// pipeline because their output was too small to qualify.
			oldTier := classifyFileTier(input.Path, content)
			newTier := classifyFileTier(input.Path, newContent)
			fileTier := oldTier
			if newTier > fileTier {
				fileTier = newTier
			}
			// GH #39 point 2: CC enrichment — same as write_file's path.
			cc, ccOK := cyclomaticComplexity(ctx, input.Path, newContent)
			if ccOK {
				if refined := refineTierWithCC(fileTier, cc); refined != fileTier {
					log.Printf("[edit_file] %s tier %s→%s via cc=%d", input.Path, fileTier, refined, cc)
					fileTier = refined
				} else {
					log.Printf("[edit_file] %s cc=%d (tier %s unchanged, oldTier=%d newTier=%d)", input.Path, cc, fileTier, oldTier, newTier)
				}
			}
			v3Out := V3EditMetadata{}
			if fileTier >= Tier2Medium && editWarrantsV3(newContent, cc, ccOK) && ctx.V3URL != "" && !ctx.BypassV3 {
				log.Printf("[edit_file] V3 pipeline activating for %s (file_tier=%d, req_tier=%d)", input.Path, fileTier, ctx.Tier)
				improved, meta, err := improveContentWithV3(path, newContent, ctx)
				if err != nil {
					// User cancellation is not a fallback case — the turn
					// was aborted, so nothing should land on disk.
					if errors.Is(err, context.Canceled) || (ctx.Ctx != nil && ctx.Ctx.Err() != nil) {
						log.Printf("[edit_file] V3 aborted by cancellation — not writing %s", input.Path)
						return &ToolResult{
							Success: false,
							Error:   "edit_file cancelled — no content was written",
						}, nil
					}
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
				result.VerificationEvidence = v3Out.VerificationEvidence
			}
			return result, nil
		},
	}
}

// ---------------------------------------------------------------------------
// ast_edit — GH #39 v1: friendly-selector AST node replacement
// ---------------------------------------------------------------------------

func astEditTool() *ToolDef {
	return &ToolDef{
		Name: "ast_edit",
		Description: "REQUIRED tool for whole-function, whole-class, or whole-HTML-element rewrites in existing files. " +
			"ALWAYS prefer over edit_file when replacing a named node or changing more than ~10 lines — edit_file is the WRONG tool for those cases (it forces you to copy the entire existing block as old_str, wasting tokens and frequently truncating). " +
			"Selectors v1: python `function:NAME` or `class:NAME` (decorators included automatically); html `<tag>` (top-level element). " +
			"Selector must match exactly one node; failures return actionable errors. " +
			"Decision rule: existing file + named-node change (any size) ⇒ ast_edit. New file ⇒ write_file. ≤10 lines inside a function ⇒ edit_file.",
		InputSchema: AstEditInput{},
		ReadOnly:    false,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input AstEditInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}
			if strings.TrimSpace(input.Path) == "" {
				return &ToolResult{Success: false,
					Error: "ast_edit: path cannot be empty. Read the file first then ast_edit with the same path."}, nil
			}
			if strings.TrimSpace(input.Selector) == "" {
				return &ToolResult{Success: false,
					Error: "ast_edit: selector cannot be empty. Examples: function:dashboard, class:UserModel, <body>"}, nil
			}

			path := resolveAgentPath(ctx, input.Path)
			if !ctx.WasFileRead(path) {
				return nil, fmt.Errorf("file not read yet — use read_file first before ast_edit: %s", input.Path)
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return nil, fmt.Errorf("cannot read %s: %w", input.Path, err)
			}
			source := string(data)

			// Empty-content guard. Replacing a node with nothing is a
			// deletion, not an edit — observed live: a model called ast_edit
			// with the `content` field omitted entirely, which spliced an
			// empty string over `function:add` and silently deleted it
			// (calc.py lost both functions while __main__ still called
			// them). It passes the syntax gate (the file still parses) and
			// the no-op guard (the content did change), so nothing else
			// catches it. Refuse it and steer: an edit needs a replacement
			// body; an intentional removal is delete_file's job.
			if strings.TrimSpace(input.Content) == "" {
				log.Printf("[ast_edit] rejected empty content for %s selector=%q — would delete the node", input.Path, input.Selector)
				return &ToolResult{Success: false, Error: fmt.Sprintf(
					"ast_edit: content is empty — that would DELETE `%s`, not fix it. "+
						"Provide the full replacement body of the node (e.g. the corrected function definition). "+
						"If you truly mean to remove code, use delete_file on the whole file instead.",
					input.Selector)}, nil
			}

			// Runaway-content guard. ast_edit replaces ONE node, so the
			// replacement should be roughly node-sized. A reasoning-heavy
			// model sometimes leaks its entire chain-of-thought into the
			// content field instead of emitting just the new body — observed
			// live: a 69KB "content" for a 3-line function, full of
			// "# Wait, the user said..." commentary. That blob then ships to
			// disk and, being huge, trips downstream size heuristics (it once
			// re-triggered a 23-min V3 PlanSearch). Reject when the
			// replacement is implausibly larger than the whole file: >4× the
			// file and over an 8KB floor (so legit edits that grow a small
			// file stay clear, and large-function edits in large files aren't
			// touched). Steer the model to emit only the replacement node.
			if len(input.Content) > 8000 && len(input.Content) > len(source)*4 {
				log.Printf("[ast_edit] rejected runaway content for %s selector=%q: %d chars vs %d-byte file",
					input.Path, input.Selector, len(input.Content), len(source))
				return &ToolResult{Success: false, Error: fmt.Sprintf(
					"ast_edit: replacement content is %d characters — far larger than the entire %d-byte file. "+
						"You only need to provide the new body of the single node `%s` (just the function/class/element itself), "+
						"not the whole file and not your reasoning. Re-emit ast_edit with content set to ONLY the replacement node.",
					len(input.Content), len(source), input.Selector)}, nil
			}

			ctx.mu.Lock()
			lastRead := ctx.FileReadTimes[path]
			ctx.mu.Unlock()
			if info, err := os.Stat(path); err == nil && info.ModTime().After(lastRead) {
				return nil, fmt.Errorf("file modified since last read — read it again before ast_edit: %s", input.Path)
			}

			// Sanitise replacement content the same way edit_file does — the
			// model occasionally fences fragments with ```python or ```html.
			if cleaned, sanitized := sanitizeFileContent(input.Path, input.Content); sanitized {
				log.Printf("[ast_edit] sanitised markdown wrapper from content of %s", input.Path)
				input.Content = cleaned
			}

			// HTML <html>-selector quirk. ast_edit replaces only the
			// <html>...</html> element, NOT the preceding <!DOCTYPE>
			// declaration that conventionally precedes it. The model
			// frequently emits a leading <!DOCTYPE html> at the top of
			// `content` when selector is <html>, which produces a duplicated
			// doctype on disk (May 8 2026 flask test: dashboard.html
			// ended up with two consecutive <!DOCTYPE html> lines after
			// a successful ast_edit). Detect that shape and strip the
			// leading doctype line so on-disk output matches intent.
			ext := strings.ToLower(filepath.Ext(input.Path))
			isHTML := ext == ".html" || ext == ".htm"
			if isHTML && strings.EqualFold(strings.TrimSpace(input.Selector), "<html>") {
				if stripped, ok := stripLeadingDoctype(input.Content); ok {
					log.Printf("[ast_edit] stripped leading <!DOCTYPE> from content of %s — selector <html> only replaces the html element, not the preceding doctype", input.Path)
					input.Content = stripped
				}
			}

			// Call v3-service /internal/ast_edit. Stateless transform:
			// proxy reads + writes (preserving lens-score-before-write),
			// v3-service is the tree-sitter authority.
			reqBody, _ := json.Marshal(map[string]interface{}{
				"path":     input.Path, // for language detection + error messages
				"source":   source,
				"selector": input.Selector,
				"content":  input.Content,
			})
			v3URL := ctx.V3URL
			if v3URL == "" {
				return nil, fmt.Errorf("ast_edit unavailable: V3 service URL not configured")
			}
			req, err := http.NewRequestWithContext(ctx.Ctx, "POST", v3URL+"/internal/ast_edit", bytes.NewReader(reqBody))
			if err != nil {
				return nil, fmt.Errorf("ast_edit: build request: %w", err)
			}
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return nil, fmt.Errorf("ast_edit: v3-service unreachable: %w", err)
			}
			defer resp.Body.Close()
			respBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("ast_edit: read v3 response: %w", err)
			}
			var astResp struct {
				Success    bool   `json:"success"`
				Error      string `json:"error,omitempty"`
				Language   string `json:"language,omitempty"`
				NewContent string `json:"new_content,omitempty"`
				ByteRange  []int  `json:"byte_range,omitempty"`
				OldSize    int    `json:"old_size,omitempty"`
				NewSize    int    `json:"new_size,omitempty"`
			}
			if err := json.Unmarshal(respBytes, &astResp); err != nil {
				return nil, fmt.Errorf("ast_edit: parse v3 response: %w (body=%s)", err, truncateStr(string(respBytes), 200))
			}
			if !astResp.Success {
				return &ToolResult{Success: false, Error: astResp.Error}, nil
			}

			// Shrinkage guard — catch the May 9 2026 destructive-stub bug
			// where the model emits only "<!DOCTYPE html>\n" for an entire
			// <html>-element rewrite. astResp.OldSize is the original
			// node's bytes; astResp.NewSize is the replacement bytes. If
			// the replacement is suspiciously small for the original,
			// reject the write and tell the model to re-emit with the
			// full body.
			if rejection := validateNotSuspiciouslyShrunk("ast_edit", input.Path, astResp.OldSize, astResp.NewSize); rejection != "" {
				log.Printf("[ast_edit] rejecting suspicious shrinkage: %s old=%dB new=%dB selector=%q",
					input.Path, astResp.OldSize, astResp.NewSize, input.Selector)
				return &ToolResult{Success: false, Error: rejection}, nil
			}

			// V3 quality-gate routing. History:
			//   (a) May 10: tier classified on post-edit content only, so a
			//       destructive ast_edit that shrank a T2+ file into a stub
			//       classified T1 and skipped V3 — the edits that most need
			//       checking. Fixed by classifying on max(oldTier, newTier).
			//   (b) May 10: floor dropped entirely so V3 fired on every
			//       ast_edit.
			//   (c) Jun 8: floor restored to Tier2Medium. With (b), every
			//       one-line ast_edit ran the full PlanSearch pipeline —
			//       minutes per edit on a reasoning-heavy model, blocking the
			//       single-threaded v3-service and looking like a hang. But
			//       ast_edit is ALREADY surgical: the model named the exact
			//       node and the replacement is its own tree-sitter
			//       transform. PlanSearch-improving a precise node swap is
			//       mostly cost. Gate it to T2+ files (same as edit_file /
			//       write_file): trivial edits apply instantly, V3 still
			//       engages where the file is genuinely complex. max-tier
			//       from (a) is preserved, so a destructive edit to a T2+
			//       original still triggers V3.
			//
			// Baseline candidate is the AST-edited full file. V3's
			// alternatives compete against it; if one build-verifies
			// better, V3 wins; otherwise the AST-edited content passes
			// through unchanged. Either way the answer is build-verified.
			finalContent := astResp.NewContent

			// No-op guard. A weak model frequently "fixes" a bug by
			// re-emitting the node's existing (broken) code verbatim —
			// observed live: ast_edit function:add with content identical
			// to the buggy body, twice in one batch. Reporting success on
			// a no-op tells the model the fix landed when nothing changed;
			// it then moves on to verification, fails, and can't work out
			// why. Fail loudly instead so the model re-derives the edit.
			if finalContent == source {
				log.Printf("[ast_edit] no-op edit rejected for %s selector=%q — replacement identical to existing code", input.Path, input.Selector)
				return &ToolResult{Success: false, Error: fmt.Sprintf(
					"ast_edit: your replacement for `%s` is IDENTICAL to the code already in the file — nothing was changed and the bug is still there. "+
						"Look at the current code again and emit a replacement that actually differs (for a swapped-operator bug, the operator itself must change).",
					input.Selector)}, nil
			}
			v3Out := V3EditMetadata{}
			oldTier := classifyFileTier(input.Path, source) // pre-edit content
			newTier := classifyFileTier(input.Path, finalContent)
			fileTier := oldTier
			if newTier > fileTier {
				fileTier = newTier
			}
			cc, ccOK := cyclomaticComplexity(ctx, input.Path, finalContent)
			if ccOK {
				if refined := refineTierWithCC(fileTier, cc); refined != fileTier {
					log.Printf("[ast_edit] %s tier %s→%s via cc=%d", input.Path, fileTier, refined, cc)
					fileTier = refined
				}
			}
			if fileTier >= Tier2Medium && editWarrantsV3(finalContent, cc, ccOK) && ctx.V3URL != "" && !ctx.BypassV3 {
				log.Printf("[ast_edit] V3 pipeline activating for %s (oldTier=%d newTier=%d max=%d, req_tier=%d, cc=%d) post-AST-edit", input.Path, oldTier, newTier, fileTier, ctx.Tier, cc)
				improved, meta, err := improveContentWithV3(path, finalContent, ctx)
				if err != nil {
					// User cancellation is not a fallback case — the turn
					// was aborted, so nothing should land on disk.
					if errors.Is(err, context.Canceled) || (ctx.Ctx != nil && ctx.Ctx.Err() != nil) {
						log.Printf("[ast_edit] V3 aborted by cancellation — not writing %s", input.Path)
						return &ToolResult{
							Success: false,
							Error:   "ast_edit cancelled — no content was written",
						}, nil
					}
					log.Printf("[ast_edit] V3 failed: %v — falling back to AST-edited content", err)
				} else if improved != "" {
					if cleanedImproved, sanitized := sanitizeFileContent(input.Path, improved); sanitized {
						log.Printf("[ast_edit] sanitised V3 output for %s", input.Path)
						improved = cleanedImproved
					}
					finalContent = improved
					v3Out = meta
				}
			}

			// Atomic write — same pattern as edit_file/write_file.
			tmpPath := path + ".atlas.tmp"
			if err := os.WriteFile(tmpPath, []byte(finalContent), 0644); err != nil {
				return nil, fmt.Errorf("cannot write %s: %w", input.Path, err)
			}
			if err := os.Rename(tmpPath, path); err != nil {
				os.Remove(tmpPath)
				return nil, fmt.Errorf("cannot rename temp file: %w", err)
			}
			ctx.RecordFileRead(path, finalContent)

			log.Printf("[ast_edit] %s %s selector=%q lang=%s old=%dB new=%dB v3=%v",
				input.Path, input.Selector, input.Selector, astResp.Language, astResp.OldSize, len(finalContent), v3Out.Used)

			out := AstEditOutput{
				OK:       true,
				Selector: input.Selector,
				Language: astResp.Language,
				BytesOld: astResp.OldSize,
				BytesNew: len(finalContent),
			}
			outBytes, _ := json.Marshal(out)
			result := &ToolResult{Success: true, Data: outBytes}
			if v3Out.Used {
				result.V3Used = true
				result.CandidatesTested = v3Out.CandidatesTested
				result.WinningScore = v3Out.WinningScore
				result.PhaseSolved = v3Out.PhaseSolved
				result.VerificationEvidence = v3Out.VerificationEvidence
			}
			return result, nil
		},
	}
}

// V3EditMetadata captures what V3 did to an edit_file request, so the
// edit_file result can carry the same v3_used / candidates_tested fields
// write_file does. See PC-042.
type V3EditMetadata struct {
	Used                 bool
	CandidatesTested     int
	WinningScore         float64
	PhaseSolved          string
	VerificationEvidence []V3VerificationEvidence
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
	if filesRead := ctx.SnapshotFilesRead(); len(filesRead) > 0 {
		req.ProjectContext = make(map[string]string)
		for p, c := range filesRead {
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
	// V3.2 RPG (issue #120): thread RPG node constraints for this target, if any.
	req.Constraints = planConstraintsForTarget(ctx, path)

	// Same callback logic as the write_file V3 path: tokens forward to
	// the dedicated v3_token SSE event so the TUI updates one streaming
	// row instead of spawning a chat row per token; LLM-call boundaries
	// match the chat protocol's start/end shapes; structured stages
	// emit typed events (v3_phase, v3_sandbox, etc.); only truly
	// unknown stages fall back to the v3_progress text line. Without
	// this branching, edit_file with V3 floods the chat pane with
	// thousands of "[token] X" rows during a single candidate generation.
	v3Result, err := callV3GenerateStreaming(ctx.Ctx, ctx.V3URL, req, func(stage, detail string, data map[string]interface{}) {
		if ctx.StreamFn == nil {
			return
		}
		if stage == "token" {
			ctx.StreamFn("v3_token", map[string]string{"text": detail})
			return
		}
		// Reasoning deltas from V3's LLM calls (see write_file path's
		// matching branch). Same purpose: visible thinking stream
		// during long PlanSearch / repair phases. v3_reasoning_token,
		// not reasoning_token, so it targets the V3 row not the agent row.
		if stage == "reasoning_token" {
			ctx.StreamFn("v3_reasoning_token", map[string]string{"text": detail})
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

	// V3.2 RPG (issue #120): same drift handling as the write_file path. If the
	// edited result missed its planned signatures, retry once, then surface any
	// surviving drift. No-op when RPG is off (req.Constraints empty; the
	// drift report shares that gate so a rogue response field alone can't
	// fire it). The regen is a second full V3 run — recheck cancellation
	// before accepting content, mirroring the write path.
	v3Result = regenerateOnDrift(ctx, req, v3Result)
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		return "", V3EditMetadata{}, fmt.Errorf("edit cancelled during RPG regen: %w", ctx.Ctx.Err())
	}
	if len(req.Constraints) > 0 && len(v3Result.RPGSignatureMissing) > 0 {
		reportRPGDrift(ctx, path, v3Result.RPGSignatureMissing)
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
		Used:                 true,
		CandidatesTested:     v3Result.CandidatesTested,
		WinningScore:         v3Result.WinningScore,
		PhaseSolved:          v3Result.PhaseSolved,
		VerificationEvidence: v3Result.VerificationEvidence,
	}, nil
}

// findActualString searches for oldStr in content, handling quote normalization.
// Returns the actual string found in content (may differ in quote style).
// findFuzzyLineMatch rescues an old_str whose only error is per-line
// whitespace drift (indentation width, tabs vs spaces, trailing spaces) —
// the dominant edit_file failure mode for small models. It compares old_str
// against the file line-by-line with each line whitespace-stripped, and
// returns the EXACT span from the file (original indentation preserved) so
// the caller's existing uniqueness-count + replace logic works unchanged.
//
// Safety: it requires exactly ONE matching window. Zero or multiple matches
// return ("", false) so the caller fails cleanly rather than editing a
// guessed location. It also refuses to match when old_str has no
// non-whitespace content (would match any blank run). Content differences
// beyond whitespace (renamed tokens, changed quotes) do NOT match — those
// are semantic and must fail.
// closestLineHint finds the file line most similar to the first
// non-blank line of a missed old_str and formats it (with its line
// number) for inclusion in the edit_file mismatch error. Similarity is
// shared-token count — crude, but enough to map a from-memory paraphrase
// like `item = items[id + 1]` onto the real
// `return jsonify(items[item_id + 1])`. Returns "" when nothing clears a
// minimal overlap bar (so unrelated guesses don't get a misleading
// anchor).
func closestLineHint(content, oldStr string) string {
	var probe string
	for _, l := range strings.Split(oldStr, "\n") {
		if strings.TrimSpace(l) != "" {
			probe = l
			break
		}
	}
	if probe == "" {
		return ""
	}
	tokenize := func(s string) map[string]bool {
		out := map[string]bool{}
		for _, t := range strings.FieldsFunc(s, func(r rune) bool {
			return !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9')
		}) {
			out[t] = true
		}
		return out
	}
	probeToks := tokenize(probe)
	if len(probeToks) == 0 {
		return ""
	}
	bestScore, bestIdx := 0, -1
	lines := strings.Split(content, "\n")
	for i, l := range lines {
		if strings.TrimSpace(l) == "" {
			continue
		}
		score := 0
		for t := range tokenize(l) {
			if probeToks[t] {
				score++
			}
		}
		if score > bestScore {
			bestScore, bestIdx = score, i
		}
	}
	// Require at least 2 shared identifiers and half the probe's tokens —
	// below that the "closest" line is likely unrelated.
	if bestIdx < 0 || bestScore < 2 || bestScore*2 < len(probeToks) {
		return ""
	}
	return fmt.Sprintf("\nClosest actual line in the file (line %d): %s\nCopy real lines from the file into old_str — do not write them from memory.",
		bestIdx+1, truncateStr(strings.TrimSpace(lines[bestIdx]), 160))
}

func findFuzzyLineMatch(content, oldStr string) (string, bool) {
	fileLines := strings.Split(content, "\n")
	oldLines := strings.Split(oldStr, "\n")
	// Drop a single trailing empty line from a trailing newline in old_str.
	if len(oldLines) > 1 && strings.TrimSpace(oldLines[len(oldLines)-1]) == "" {
		oldLines = oldLines[:len(oldLines)-1]
	}
	if len(oldLines) == 0 {
		return "", false
	}
	strip := func(ls []string) []string {
		out := make([]string, len(ls))
		nonEmpty := false
		for i, l := range ls {
			out[i] = strings.TrimSpace(l)
			if out[i] != "" {
				nonEmpty = true
			}
		}
		if !nonEmpty {
			return nil // all-whitespace target: refuse
		}
		return out
	}
	want := strip(oldLines)
	if want == nil {
		return "", false
	}
	n := len(want)
	matchStart := -1
	matches := 0
	for i := 0; i+n <= len(fileLines); i++ {
		ok := true
		for j := 0; j < n; j++ {
			if strings.TrimSpace(fileLines[i+j]) != want[j] {
				ok = false
				break
			}
		}
		if ok {
			matches++
			matchStart = i
			if matches > 1 {
				return "", false // ambiguous — fail safe
			}
		}
	}
	if matches != 1 {
		return "", false
	}
	return strings.Join(fileLines[matchStart:matchStart+n], "\n"), true
}

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
		"\u2018", "'", // left single
		"\u2019", "'", // right single
	)
	return r.Replace(s)
}

// denormalizeQuotes replaces straight quotes with curly quotes (best-effort).
func denormalizeQuotes(s string) string {
	r := strings.NewReplacer(
		"\"", "\u201c", // straight double → left double (approximate)
		"'", "\u2019", // straight single → right single (approximate)
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
					if rmErr := os.Remove(realPath); rmErr != nil {
						return &ToolResult{
							Success: false,
							Error:   fmt.Sprintf("delete_file: %v", rmErr),
						}, nil
					}
					deleted = true
					log.Printf("[delete_file] %s deleted from project dir %s", input.Path, ctx.RealProjectDir)
				}
			}

			// Also delete from temp/working dir if it exists there. If the
			// project-dir removal already succeeded, a failure on this mirror
			// copy is not reported as an overall failure — the user-visible
			// file is already gone, and reporting failure would make the model
			// retry against a path it can never clear.
			path := resolveAgentPath(ctx, input.Path)
			if info, err := os.Stat(path); err == nil {
				if info.IsDir() {
					entries, _ := os.ReadDir(path)
					if len(entries) > 0 {
						if deleted {
							log.Printf("[delete_file] %s removed from project dir; working-dir copy is a non-empty directory, left in place", input.Path)
						} else {
							return &ToolResult{
								Success: false,
								Error:   fmt.Sprintf("directory not empty: %s (%d entries) — delete_file only removes files or empty directories", input.Path, len(entries)),
							}, nil
						}
					} else if rmErr := os.Remove(path); rmErr != nil && !deleted {
						return &ToolResult{
							Success: false,
							Error:   fmt.Sprintf("delete_file: %v", rmErr),
						}, nil
					} else if rmErr == nil {
						deleted = true
					}
				} else if rmErr := os.Remove(path); rmErr != nil {
					if !deleted {
						return &ToolResult{
							Success: false,
							Error:   fmt.Sprintf("delete_file: %v", rmErr),
						}, nil
					}
					log.Printf("[delete_file] %s removed from project dir; working-dir copy removal failed: %v", input.Path, rmErr)
				} else {
					deleted = true
				}
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
// move_file — relocate / rename a file within the workspace.
//
// Added to close the "reorganize the files" gap (observed: a flask task asked
// to move index.html into templates/; `mv` is refused, there is no move tool,
// so the model looped on mkdir until the repetition breaker fired). A pure
// move is not a content change, so it bypasses the V3 / surgical-edit gate —
// content is preserved verbatim. Refuses to clobber an existing destination
// file so a relocation can't silently destroy data.
// ---------------------------------------------------------------------------

func moveFileTool() *ToolDef {
	return &ToolDef{
		Name:        "move_file",
		Description: "Move or rename a file within the project (e.g. move index.html into templates/, or rename old.py to new.py). Use this to reorganize files — shell `mv`/`cp` are refused. If destination is an existing directory, the file is moved into it keeping its name. Content is preserved exactly.",
		InputSchema: MoveFileInput{},
		ReadOnly:    false,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input MoveFileInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}
			if strings.TrimSpace(input.Source) == "" || strings.TrimSpace(input.Destination) == "" {
				return &ToolResult{
					Success: false,
					Error:   `move_file: both source and destination are required. Call with {"source":"<current path>","destination":"<new path>"}.`,
				}, nil
			}

			src := resolveAgentPath(ctx, input.Source)
			srcInfo, err := os.Stat(src)
			if err != nil {
				return &ToolResult{
					Success: false,
					Error:   fmt.Sprintf("move_file: source %s not found. Use list_directory or find_file to confirm the path before moving.", input.Source),
				}, nil
			}

			// Resolve destination. If it names an existing directory (or ends
			// with a separator), move INTO it keeping the source basename —
			// mirrors `mv file dir/`. The relative dest is what we report back
			// so the model's mental model stays in project-relative terms.
			relDest := input.Destination
			dst := resolveAgentPath(ctx, input.Destination)
			if info, err := os.Stat(dst); err == nil && info.IsDir() {
				dst = filepath.Join(dst, filepath.Base(src))
				relDest = filepath.Join(input.Destination, filepath.Base(src))
			} else if strings.HasSuffix(input.Destination, "/") {
				dst = filepath.Join(dst, filepath.Base(src))
				relDest = filepath.Join(input.Destination, filepath.Base(src))
			}

			if src == dst {
				return &ToolResult{
					Success: false,
					Error:   "move_file: source and destination are the same path — nothing to do.",
				}, nil
			}

			// Never clobber an existing destination file: a relocation must not
			// silently destroy data. Tell the model to pick another name or
			// delete_file the destination first if the overwrite is intended.
			if _, err := os.Stat(dst); err == nil {
				return &ToolResult{
					Success: false,
					Error:   fmt.Sprintf("move_file: destination %s already exists. Pick a different name, or delete_file the destination first if you mean to replace it.", relDest),
				}, nil
			}

			if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
				return nil, fmt.Errorf("move_file: cannot create destination dir: %w", err)
			}

			// os.Rename is atomic on the same filesystem; fall back to
			// copy+remove across devices (bind mounts can straddle filesystems).
			if err := os.Rename(src, dst); err != nil {
				if srcInfo.IsDir() {
					return nil, fmt.Errorf("move_file: cannot move directory across filesystems: %w", err)
				}
				data, rerr := os.ReadFile(src)
				if rerr != nil {
					return nil, fmt.Errorf("move_file: cannot read source: %w", rerr)
				}
				if werr := os.WriteFile(dst, data, srcInfo.Mode().Perm()); werr != nil {
					return nil, fmt.Errorf("move_file: cannot write destination: %w", werr)
				}
				os.Remove(src)
			}
			log.Printf("[move_file] %s → %s", input.Source, relDest)

			// Keep agent bookkeeping consistent: the file the model just read
			// now lives at the new path. Re-point the recorded read and the
			// session-write set so a follow-up edit isn't bounced as blind and
			// dedup logic tracks the right path.
			if content, ok := ctx.GetFileRead(src); ok {
				ctx.RecordFileRead(dst, content)
				ctx.ForgetFileRead(src)
			}
			if ctx.SessionWrites != nil {
				if ctx.SessionWrites[input.Source] {
					delete(ctx.SessionWrites, input.Source)
				}
				ctx.SessionWrites[relDest] = true
			}

			out := MoveFileOutput{Moved: true, Source: input.Source, Destination: relDest}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
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

			// Trust gate: untrusted mode refuses command execution;
			// host execution is honored only under fully-trusted (else
			// downgraded to the sandbox below).
			if !ctx.TrustMode.commandsAllowed() {
				return &ToolResult{Success: false, Error: untrustedRefusal}, nil
			}

			timeoutSec := 30
			if input.Timeout != nil && *input.Timeout > 0 {
				timeoutSec = *input.Timeout
			}
			if timeoutSec > 300 {
				timeoutSec = 300
			}

			cwd := ctx.WorkingDir
			if input.Cwd != "" {
				cwd = resolveAgentPath(ctx, input.Cwd)
			}

			// PC-188: route shell execution through the sandbox container.
			// The proxy is a slim Go binary with no python/pip/node, so
			// running locally meant every "verify" command failed with
			// "command not found". The sandbox has the language matrix
			// pre-installed AND has /workspace bind-mounted at the same
			// path the proxy sees, so paths the agent learned via
			// read_file / list_directory still work. validateShellCommand
			// upstream is the gate; this is the executor.
			//
			// PC-192: when ctx.VerifyOnHost is set (ATLAS_VERIFY_IN=host
			// or per-project config), we BYPASS the sandbox and execute
			// on the host directly. This is the right call for working
			// codebases that depend on host-side state — the user's
			// installed venv binaries, system tools, env vars,
			// running databases, etc. — that the sandbox can't see.
			// The shell-op safety gate (validateShellCommand) still
			// fired upstream regardless of target. cwd is translated
			// to the host path so the command lands in the right dir.
			var out RunCommandOutput
			var err error
			// Host execution requires fully-trusted; otherwise a
			// host request is downgraded to sandbox so the trust
			// level can't be silently escalated by ATLAS_VERIFY_IN.
			useHost := ctx.VerifyOnHost && ctx.TrustMode.hostExecutionAllowed()
			if useHost {
				hostCwd := cwd
				if ctx.HostWorkingDir != "" && strings.HasPrefix(cwd, ctx.WorkingDir) {
					hostCwd = ctx.HostWorkingDir + strings.TrimPrefix(cwd, ctx.WorkingDir)
				}
				out = runLocally(input.Command, hostCwd, time.Duration(timeoutSec)*time.Second)
			} else {
				out, err = runViaSandbox(ctx, input.Command, cwd, timeoutSec)
				if err != nil {
					log.Printf("[run_command] sandbox unavailable: %v", err)
					out = RunCommandOutput{
						Stderr:   fmt.Sprintf("sandbox unavailable: %v", err),
						ExitCode: 1,
					}
				}
			}

			outBytes, _ := json.Marshal(out)
			var errMsg string
			if out.ExitCode != 0 {
				errMsg = strings.TrimSpace(out.Stderr)
				if errMsg == "" {
					if s := strings.TrimSpace(out.Stdout); s != "" {
						lines := strings.Split(s, "\n")
						errMsg = lines[len(lines)-1]
					}
				}
				if errMsg == "" {
					errMsg = fmt.Sprintf("exit %d (no output)", out.ExitCode)
				}
				errMsg = truncateStr(errMsg, 400)
			}
			return &ToolResult{
				Success: out.ExitCode == 0,
				Data:    outBytes,
				Error:   errMsg,
			}, nil
		},
	}
}

// runViaSandbox POSTs the command to the sandbox /shell endpoint.
// Returns a populated RunCommandOutput on success, or an error if the
// sandbox is unreachable / returned a non-2xx (caller falls back to
// local exec). Timeout is in seconds and is enforced server-side; we
// add a generous client-side margin so the HTTP call doesn't kill
// long-running commands prematurely.
func runViaSandbox(ctx *AgentContext, command, cwd string, timeoutSec int) (RunCommandOutput, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"command": command,
		"cwd":     cwd,
		"timeout": timeoutSec,
	})
	endpoint := ctx.SandboxURL + "/shell"
	// Bind the agent's request context so a user cancel aborts the
	// in-flight sandbox call instead of waiting out the client timeout.
	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	httpReq, err := http.NewRequestWithContext(reqCtx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return RunCommandOutput{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: time.Duration(timeoutSec+30) * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return RunCommandOutput{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		// 4xx is usually a validation error (bad cwd, etc.) — propagate
		// as a regular failure, not a sandbox-unreachable signal. Read
		// the FastAPI detail so the model sees what went wrong.
		var errBody struct {
			Detail string `json:"detail"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&errBody)
		return RunCommandOutput{
			Stderr:   fmt.Sprintf("sandbox /shell %d: %s", resp.StatusCode, errBody.Detail),
			ExitCode: 1,
		}, nil
	}
	var sr struct {
		Success   bool   `json:"success"`
		Stdout    string `json:"stdout"`
		Stderr    string `json:"stderr"`
		ExitCode  int    `json:"exit_code"`
		ElapsedMS int    `json:"elapsed_ms"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return RunCommandOutput{}, fmt.Errorf("decode sandbox response: %w", err)
	}
	return RunCommandOutput{
		Stdout:   truncateStr(sr.Stdout, 8000),
		Stderr:   truncateStr(sr.Stderr, 4000),
		ExitCode: sr.ExitCode,
	}, nil
}

// runLocally executes a command only when the operator explicitly selects
// host verification (ATLAS_VERIFY_IN=host). Sandbox outages never route
// here implicitly. Host mode removes the container backstop entirely:
// the only thing between model output and the host shell is
// validateShellCommand's catastrophe-only blocklist.
func runLocally(command, cwd string, timeout time.Duration) RunCommandOutput {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	cmd.Dir = cwd

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	var exitCode int
	if ctx.Err() == context.DeadlineExceeded {
		exitCode = 124
		stderr.WriteString(fmt.Sprintf("\nCommand timed out after %s", timeout))
	} else if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			stderr.WriteString(err.Error())
			exitCode = 1
		}
	}

	return RunCommandOutput{
		Stdout:   truncateStr(stdout.String(), 8000),
		Stderr:   truncateStr(stderr.String(), 4000),
		ExitCode: exitCode,
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

// cyclomaticComplexity calls v3-service /internal/cyclomatic_complexity.
// Returns (cc, true) when the service computed a real number, (0, false)
// for any failure mode (unsupported language, parse error, network down,
// timeout). Fail-soft is intentional — the existing regex-based
// hasLogicIndicators stays the floor; CC only adds signal when available.
//
// GH #39 point 2. v1 supports Python only; HTML/JSON/etc. fall through
// to false here and the proxy uses the regex classifier.
func cyclomaticComplexity(ctx *AgentContext, path, source string) (int, bool) {
	if ctx == nil || ctx.V3URL == "" {
		return 0, false
	}
	body, err := json.Marshal(map[string]interface{}{"path": path, "source": source})
	if err != nil {
		return 0, false
	}
	reqCtx, cancel := context.WithTimeout(ctx.Ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST",
		ctx.V3URL+"/internal/cyclomatic_complexity", bytes.NewReader(body))
	if err != nil {
		return 0, false
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, false
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, false
	}
	var r struct {
		OK bool `json:"ok"`
		CC int  `json:"cyclomatic_complexity"`
	}
	if err := json.Unmarshal(raw, &r); err != nil || !r.OK {
		return 0, false
	}
	return r.CC, true
}

// refineTierWithCC bumps an existing tier upward when McCabe CC reveals
// more branching than the regex classifier could see. Never downgrades —
// the regex classifier is the floor, CC only escalates.
//
// Thresholds:
//
//	CC ≥ 16 → Tier3Hard  — definitely needs full V3 + best-of-K
//	CC ≥  8 → Tier2Medium — moderate branching, V3 likely helps
//	CC <  8 → leave base tier unchanged
//
// Calibrated against the snake/app.py family: a flask file with 8 routes
// runs at CC≈9 (one branch per route) and the regex already classifies
// it T2; a control-flow-heavy parser with nested ifs at CC≈18 should
// jump to T3 even if the regex landed it at T2.
func refineTierWithCC(base Tier, cc int) Tier {
	if cc >= 16 && base < Tier3Hard {
		return Tier3Hard
	}
	if cc >= 8 && base < Tier2Medium {
		return Tier2Medium
	}
	return base
}

// editWarrantsV3 decides whether a successful, surgical edit should be run
// back through the V3 whole-file pipeline. edit_file and ast_edit both
// produce content the model specified precisely — an exact old→new string
// swap, or a named tree-sitter node replacement — so the result is already
// what was asked for. V3's PlanSearch generates and build-verifies whole-
// file alternatives: worthwhile when a file is large or genuinely complex,
// but on a small, low-complexity file it spends minutes (single-threaded
// v3-service, reasoning-heavy models) only to reproduce the same edit. The
// file tier alone can't distinguish these — a 9-line calc.py classifies
// Tier2 like a 400-line module — so gate additionally on the resulting
// file's complexity and size. Trivial edits then apply instantly while V3
// still engages where a file is substantial enough to benefit. Keys off
// the code, not the model, so it stays model-agnostic.
//
// ccOK is false when complexity couldn't be measured (non-code or parser
// miss); fall back to a line-count bar so we neither always-run nor
// never-run on unmeasurable files.
func editWarrantsV3(content string, cc int, ccOK bool) bool {
	if ccOK && cc >= 8 {
		return true
	}
	lines := strings.Count(content, "\n") + 1
	return lines >= 80
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
// pycheckViaV3 asks the v3-service whether Python source parses. Returns
// (true, "") when it parses, when the check can't run (service down, busy,
// timeout), or when V3 is bypassed — fail-open by design: the gate exists
// to catch garbage-quoted edits, not to make edits depend on v3-service
// availability. Returns (false, error) only on a definitive SyntaxError.
func pycheckViaV3(ctx *AgentContext, path, source string) (bool, string) {
	if ctx.V3URL == "" || ctx.BypassV3 {
		return true, ""
	}
	body, err := json.Marshal(map[string]string{"path": path, "source": source})
	if err != nil {
		return true, ""
	}
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Post(ctx.V3URL+"/internal/pycheck", "application/json", bytes.NewReader(body))
	if err != nil {
		return true, ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return true, ""
	}
	var out struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil {
		return true, ""
	}
	if out.OK {
		return true, ""
	}
	return false, out.Error
}

// redundantReadShortCircuit returns a compact synthetic result when the
// model asks to read a file it has ALREADY read this session and the
// content on disk is unchanged. A weak model frequently re-reads the same
// file several times before acting; each re-read re-injects the full file
// into the conversation and — on sliding-window-attention models that
// can't reuse llama.cpp's KV cache — forces a full re-encode of the whole
// prompt, costing tens of seconds for zero new information. Serving a
// short "already in context, act now" pointer keeps the working set small
// and steers toward the edit.
//
// Only whole-file reads short-circuit (a paged read with offset/limit is
// the model deliberately fetching a different span, so let it through),
// and only when the on-disk content byte-matches what was already shown
// (so a re-read after an edit still returns the new content). Returns nil
// to mean "no short-circuit — execute the read normally." Model-agnostic:
// keys off whether this exact content was already served, not the model.
func redundantReadShortCircuit(name string, args json.RawMessage, ctx *AgentContext) *ToolResult {
	if name != "read_file" {
		return nil
	}
	if envOr("ATLAS_DEDUP_READS", "1") == "0" {
		return nil
	}
	var input ReadFileInput
	if err := json.Unmarshal(args, &input); err != nil {
		return nil
	}
	if input.Offset != nil || input.Limit != nil || strings.TrimSpace(input.Path) == "" {
		return nil
	}
	path, err := resolveWorkspacePath(ctx, input.Path)
	if err != nil {
		return &ToolResult{Success: false, Error: "read_file: " + err.Error()}
	}
	ctx.mu.Lock()
	prev, ok := ctx.FilesRead[path]
	cacheEntries := len(ctx.FilesRead)
	ctx.mu.Unlock()
	if !ok {
		// Diagnostic: a re-read that SHOULD have been cached but wasn't.
		// Logged only when some other entry exists (first read of a file
		// is the normal case and not worth a line).
		if cacheEntries > 0 {
			log.Printf("[read-dedup] no cache entry for %q (have %d entries) — serving real read", truncateStr(path, 240), cacheEntries)
		}
		return nil
	}
	data, _, err := readWorkspaceFile(ctx, input.Path)
	if err != nil || string(data) != prev {
		if err == nil {
			log.Printf("[read-dedup] %q changed on disk since last read (%dB -> %dB) — serving real read", truncateStr(path, 240), len(prev), len(data))
		}
		return nil // changed or unreadable — let the real read run
	}
	// CRITICAL: only short-circuit to a pointer if the content is STILL in
	// the live conversation. After conversation trimming drops the original
	// read, "its content is above" becomes a lie — the model has nothing,
	// edits blind, and (if weak) hallucinates symbols/lines that aren't in
	// the file. When the content has been trimmed out, return nil so the
	// real read re-serves the full file. Verified failure mode: a model editing
	// function:count_items / old_str="return len(items)" against a file
	// containing neither, reasoning "I don't see the file content."
	if !fileContentInContext(ctx, prev) {
		log.Printf("[read-dedup] %q content was trimmed from context — re-serving full read", truncateStr(path, 240))
		return nil
	}
	out := ReadFileOutput{
		Content:    fmt.Sprintf("(You already read %s earlier in this session and it has not changed — its full content is above in the conversation. Do not read it again. Make your edit now with ast_edit or edit_file.)", input.Path),
		TotalLines: strings.Count(prev, "\n") + 1,
		StartLine:  1,
		EndLine:    strings.Count(prev, "\n") + 1,
	}
	b, _ := json.Marshal(out)
	return &ToolResult{Success: true, Data: b}
}

// fileContentInContext reports whether a file's content is still present in
// the live (post-trim) conversation. Conservative: an empty/short-content file
// probes as "present" (nothing to lose).
//
// The probe must survive JSON escaping. Tool results are stored in the
// conversation via ToolResult.MarshalText (json.Marshal), so the file content
// lives in ctx.Messages with `"`→`\"`, newlines→`\n`, tabs→`\t`. The old probe
// (the longest raw LINE) failed for any file whose longest line contained a
// double quote — e.g. a flask app's embedded HTML/JS — producing a false
// "trimmed" verdict that made the dedup re-serve the whole file every read and
// the model loop on read_file. Instead, probe with the longest maximal run of
// characters JSON does NOT escape (no `"`, `\`, or \n/\r/\t); that run is byte-
// identical in both the raw file and the escaped conversation copy.
func fileContentInContext(ctx *AgentContext, content string) bool {
	probe := ""
	for _, run := range strings.FieldsFunc(content, func(r rune) bool {
		return r == '"' || r == '\\' || r == '\n' || r == '\r' || r == '\t'
	}) {
		t := strings.TrimSpace(run)
		if len(t) > len(probe) {
			probe = t
		}
	}
	if len(probe) < 12 {
		return true // too short to probe reliably; don't churn re-reads
	}
	for _, m := range ctx.Messages {
		if strings.Contains(m.Content, probe) {
			return true
		}
	}
	return false
}

func resolveAgentPath(ctx *AgentContext, path string) string {
	// PC-198 — defensive prefix strip. The local model frequently
	// emits `workspace/app.py` (no leading slash) when it means the
	// project root. Without this, resolvePath joins it onto cwd and
	// produces `/workspace/workspace/app.py`, which 404s. Strip the
	// `workspace/` prefix when WorkingDir is exactly `/workspace`.
	// Also handles a bare `workspace` (no trailing slash) for
	// list_directory.
	if ctx.WorkingDir == "/workspace" {
		switch {
		case path == "workspace":
			path = "."
		case strings.HasPrefix(path, "workspace/"):
			path = strings.TrimPrefix(path, "workspace/")
		case strings.HasPrefix(path, "./workspace/"):
			path = strings.TrimPrefix(path, "./workspace/")
		}
	}
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
	case "lens_per_step":
		// PC-207 wiring: per-token lens scoring of each V3 candidate. TUI
		// surfaces first_off_rails_idx + gx_score_min so the user can see
		// WHERE a candidate's quality cratered. Without this case the
		// event flattens to v3_progress and the structured payload is lost.
		return "v3_lens_per_step"
	case "lens_veto":
		// PC-207 alignment: V3 hard-rejected a sandbox-passing candidate
		// because the lens flagged it as a stub (gx_min < severe threshold).
		// Surfaced as its own event so the user can see "sandbox said pass
		// but lens vetoed" rather than burying it in v3_progress.
		return "v3_lens_veto"
	case "structural_veto":
		// GH #39 point 1: V3 hard-rejected a sandbox-passing candidate
		// because tree-sitter found unresolved direct-identifier calls.
		// Sandbox passes for code with try/except ImportError fallbacks
		// or dead branches; structural verification doesn't care whether
		// the unresolved call executes, only that it can't resolve.
		return "v3_structural_veto"
	case "call_chain_context":
		// GH #39 point 3: V3 phase-3 repair built a call-chain context
		// block for the failing function before invoking PR-CoT /
		// refinement. Informational — not a veto, just shows the user
		// that the repair phase has structural context the bare stderr
		// doesn't include.
		return "v3_call_chain_context"
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

// ---------------------------------------------------------------------------
// Background commands (PC-196)
// ---------------------------------------------------------------------------
//
// Three tools wrap the sandbox /jobs/* endpoints. The pattern the model
// learns is: run_background(server) → tail_background or curl via
// run_command → stop_background. Without these, foreground servers
// (flask, npm start, cargo run) can't be verified — they don't exit
// and the model invents `timeout 5 ... || true` workarounds that tear
// the server down before any probe can hit it.
//
// All three tools require ATLAS_VERIFY_IN=sandbox (the default). Host
// mode bypasses the sandbox entirely; running long-lived processes on
// the host without any reaping is a foot-gun we don't want to ship,
// so we surface a clear error instead.

func runBackgroundTool() *ToolDef {
	return &ToolDef{
		Name:        "run_background",
		Description: "Start a long-running command (server, watcher, etc.) in the background and return a job_id. Use for `python app.py`, `npm start`, `cargo run`, `flask run` — anything that doesn't exit. Returns initial stdout/stderr captured during a brief settle window so you can confirm startup. Pair with run_command/curl to probe the running service, then stop_background to clean up.",
		InputSchema: RunBackgroundInput{},
		ReadOnly:    false,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input RunBackgroundInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}

			// Trust gate: same contract as run_command — untrusted mode
			// refuses all command execution, foreground or background.
			if !ctx.TrustMode.commandsAllowed() {
				return &ToolResult{Success: false, Error: untrustedRefusal}, nil
			}

			if strings.TrimSpace(input.Command) == "" {
				return &ToolResult{Success: false, Error: "run_background: command cannot be empty"}, nil
			}
			if reason := validateShellCommand(input.Command); reason != "" {
				return &ToolResult{Success: false, Error: reason}, nil
			}
			if ctx.VerifyOnHost {
				return &ToolResult{
					Success: false,
					Error:   "run_background is only available in sandbox mode (ATLAS_VERIFY_IN=sandbox). On the host, use `run_command` with `nohup ... &` and track the PID yourself.",
				}, nil
			}
			cwd := ctx.WorkingDir
			if input.Cwd != "" {
				cwd = resolveAgentPath(ctx, input.Cwd)
			}
			settleMs := 1500
			if input.SettleMs != nil {
				settleMs = *input.SettleMs
				if settleMs < 0 {
					settleMs = 0
				} else if settleMs > 10000 {
					settleMs = 10000
				}
			}
			jobID, pid, err := sandboxStartBackground(ctx, input.Command, cwd)
			if err != nil {
				return &ToolResult{Success: false, Error: fmt.Sprintf("sandbox start failed: %v", err)}, nil
			}
			// Settle window — give the process time to bind a port, fail
			// to import, etc., before we hand back to the model.
			time.Sleep(time.Duration(settleMs) * time.Millisecond)
			tail, _ := sandboxTailBackground(ctx, jobID, 50)
			out := RunBackgroundOutput{
				JobID:   jobID,
				PID:     pid,
				Stdout:  tail.Stdout,
				Stderr:  tail.Stderr,
				Running: tail.Running,
			}
			if !tail.Running {
				out.ExitCode = tail.ExitCode
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

func tailBackgroundTool() *ToolDef {
	return &ToolDef{
		Name:        "tail_background",
		Description: "Read the recent stdout/stderr of a background job started via run_background. Returns the last N lines of each stream (default 50), the run state (running/exited), and the exit code if applicable. Use to check whether a server is still up, watch test runner output, or read the failure traceback after a crash.",
		InputSchema: TailBackgroundInput{},
		ReadOnly:    true,
		Destructive: false,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input TailBackgroundInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}
			if strings.TrimSpace(input.JobID) == "" {
				return &ToolResult{Success: false, Error: "tail_background: job_id required"}, nil
			}
			lines := 50
			if input.Lines != nil {
				lines = *input.Lines
				if lines < 1 {
					lines = 1
				} else if lines > 500 {
					lines = 500
				}
			}
			out, err := sandboxTailBackground(ctx, input.JobID, lines)
			if err != nil {
				return &ToolResult{Success: false, Error: err.Error()}, nil
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

func stopBackgroundTool() *ToolDef {
	return &ToolDef{
		Name:        "stop_background",
		Description: "Stop a background job started via run_background. Sends SIGTERM, waits briefly, then SIGKILL if needed. Returns the final stdout/stderr buffer. Always call this when you're done with a background job — leaving them running blocks future job slots.",
		InputSchema: StopBackgroundInput{},
		ReadOnly:    false,
		Destructive: true,
		Execute: func(rawInput json.RawMessage, ctx *AgentContext) (*ToolResult, error) {
			var input StopBackgroundInput
			if err := json.Unmarshal(rawInput, &input); err != nil {
				return nil, fmt.Errorf("invalid input: %w", err)
			}
			if strings.TrimSpace(input.JobID) == "" {
				return &ToolResult{Success: false, Error: "stop_background: job_id required"}, nil
			}
			out, err := sandboxStopBackground(ctx, input.JobID)
			if err != nil {
				return &ToolResult{Success: false, Error: err.Error()}, nil
			}
			outBytes, _ := json.Marshal(out)
			return &ToolResult{Success: true, Data: outBytes}, nil
		},
	}
}

// sandboxStartBackground POSTs to /jobs/start. Returns (job_id, pid, err).
func sandboxStartBackground(ctx *AgentContext, command, cwd string) (string, int, error) {
	if ctx.SandboxURL == "" {
		return "", 0, fmt.Errorf("ATLAS_SANDBOX_URL not configured")
	}
	body, _ := json.Marshal(map[string]interface{}{"command": command, "cwd": cwd})
	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	req, err := http.NewRequestWithContext(reqCtx, "POST", ctx.SandboxURL+"/jobs/start", bytes.NewReader(body))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		var d struct {
			Detail string `json:"detail"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&d)
		if d.Detail != "" {
			return "", 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, d.Detail)
		}
		return "", 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var out struct {
		JobID string `json:"job_id"`
		PID   int    `json:"pid"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", 0, err
	}
	return out.JobID, out.PID, nil
}

func sandboxTailBackground(ctx *AgentContext, jobID string, lines int) (TailBackgroundOutput, error) {
	if ctx.SandboxURL == "" {
		return TailBackgroundOutput{}, fmt.Errorf("ATLAS_SANDBOX_URL not configured")
	}
	url := fmt.Sprintf("%s/jobs/%s/output?lines=%d", ctx.SandboxURL, jobID, lines)
	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	req, err := http.NewRequestWithContext(reqCtx, "GET", url, nil)
	if err != nil {
		return TailBackgroundOutput{}, err
	}
	resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
	if err != nil {
		return TailBackgroundOutput{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 404 {
		return TailBackgroundOutput{}, fmt.Errorf("unknown job_id %q (already cleaned up?)", jobID)
	}
	if resp.StatusCode != 200 {
		return TailBackgroundOutput{}, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var out TailBackgroundOutput
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return TailBackgroundOutput{}, err
	}
	return out, nil
}

func sandboxStopBackground(ctx *AgentContext, jobID string) (StopBackgroundOutput, error) {
	if ctx.SandboxURL == "" {
		return StopBackgroundOutput{}, fmt.Errorf("ATLAS_SANDBOX_URL not configured")
	}
	url := fmt.Sprintf("%s/jobs/%s/stop", ctx.SandboxURL, jobID)
	reqCtx := ctx.Ctx
	if reqCtx == nil {
		reqCtx = context.Background()
	}
	req, err := http.NewRequestWithContext(reqCtx, "POST", url, nil)
	if err != nil {
		return StopBackgroundOutput{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return StopBackgroundOutput{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 404 {
		return StopBackgroundOutput{}, fmt.Errorf("unknown job_id %q", jobID)
	}
	if resp.StatusCode != 200 {
		return StopBackgroundOutput{}, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var out StopBackgroundOutput
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return StopBackgroundOutput{}, err
	}
	return out, nil
}
