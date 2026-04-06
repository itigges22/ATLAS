// atlas-proxy: Production inference proxy for ATLAS.
//
// Sits between Aider (or any OpenAI client) and Fox (local LLM).
// Implements a verify-repair pipeline inspired by Claude Code:
//
//   1. Forward request to Fox (stream or batch)
//   2. Detect code in response
//   3. Score with C(x)+G(x) quality gate
//   4. Sandbox-test code (if applicable)
//   5. If sandbox fails → analyze error → repair → re-test (max 3 iterations)
//   6. Return best version to client
//
// Usage:
//   atlas-proxy                  (default port 8090)
//   OPENAI_API_BASE=http://localhost:8090 aider --model openai/atlas
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

var (
	inferenceURL = envOr("ATLAS_INFERENCE_URL", "http://localhost:8080")
	lensURL     = envOr("ATLAS_LENS_URL", "http://localhost:8099")
	sandboxURL = envOr("ATLAS_SANDBOX_URL", "http://localhost:30820")
	proxyPort  = envOr("ATLAS_PROXY_PORT", "8090")
	modelName  = envOr("ATLAS_MODEL_NAME", "Qwen3.5-9B-Q6_K")
)

const (
	maxRepairAttempts = 3
	gxLowThreshold   = 0.5  // below this → trigger best-of-K
	gxHighThreshold   = 0.9  // above this → early exit from best-of-K
	sandboxTimeout    = 8    // seconds
	interactiveTimeout = 3   // seconds for interactive programs
)

var v3ServiceURL = envOr("ATLAS_V3_URL", "http://localhost:8070")

// Session state: track whether last response was V3 delivery
// to prevent double-request triggering another full pipeline run
var lastWasV3 sync.Map // key: remote IP, value: time.Time

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ---------------------------------------------------------------------------
// Telemetry counters
// ---------------------------------------------------------------------------

var (
	totalRequests   atomic.Int64
	totalRepairs    atomic.Int64
	sandboxPasses   atomic.Int64
	sandboxFails    atomic.Int64
)

// ---------------------------------------------------------------------------
// OpenAI API types
// ---------------------------------------------------------------------------

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   Usage        `json:"usage"`

	// ATLAS pipeline metadata
	AtlasRoute         string  `json:"atlas_route,omitempty"`
	AtlasGxScore       float64 `json:"atlas_gx_score,omitempty"`
	AtlasVerdict       string  `json:"atlas_verdict,omitempty"`
	AtlasSandboxPassed *bool   `json:"atlas_sandbox_passed,omitempty"`
	AtlasRepairAttempt int     `json:"atlas_repair_attempt,omitempty"`
}

// ---------------------------------------------------------------------------
// Lens scoring types
// ---------------------------------------------------------------------------

type LensScore struct {
	CxEnergy  float64 `json:"cx_energy"`
	CxNorm    float64 `json:"cx_normalized"`
	GxScore   float64 `json:"gx_score"`
	Verdict   string  `json:"verdict"`
	Enabled   bool    `json:"enabled"`
	LatencyMs float64 `json:"latency_ms"`
}

// ---------------------------------------------------------------------------
// Sandbox types
// ---------------------------------------------------------------------------

type SandboxResult struct {
	Success        bool    `json:"success"`
	CompileSuccess bool    `json:"compile_success"`
	Stdout         string  `json:"stdout"`
	Stderr         string  `json:"stderr"`
	ErrorType      *string `json:"error_type"`
	ErrorMessage   *string `json:"error_message"`
	ExecutionMs    float64 `json:"execution_time_ms"`
}

// ---------------------------------------------------------------------------
// Code detection
// ---------------------------------------------------------------------------

var codeBlockRe = regexp.MustCompile("(?s)```(\\w*)\\s*\\n(.*?)```")

// Aider whole-file format: filename.ext\n```\n...\n```
var wholeFileRe = regexp.MustCompile("(?m)^([\\w./\\-]+\\.\\w+)\\s*\\n```(\\w*)\\s*\\n")

// Language detection from file extension
var extToLang = map[string]string{
	".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
	".go": "go", ".rs": "rust", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
	".sh": "bash", ".bash": "bash", ".jsx": "javascript",
}

type CodeBlock struct {
	Language string
	Code     string
	Filename string // populated for whole-file format
}

func extractCodeBlocks(content string) []CodeBlock {
	var blocks []CodeBlock

	// Strategy 1: Standard fenced code blocks (```lang\n...\n```)
	matches := codeBlockRe.FindAllStringSubmatch(content, -1)
	for _, m := range matches {
		lang := strings.ToLower(strings.TrimSpace(m[1]))
		code := strings.TrimSpace(m[2])
		if code == "" {
			continue
		}

		// If no language tag on the fence, try to detect from context
		if lang == "" {
			// Check if preceded by a filename
			fencePos := strings.Index(content, m[0])
			if fencePos > 0 {
				before := content[:fencePos]
				lines := strings.Split(strings.TrimRight(before, " \t"), "\n")
				if len(lines) > 0 {
					lastLine := strings.TrimSpace(lines[len(lines)-1])
					for ext, l := range extToLang {
						if strings.HasSuffix(lastLine, ext) {
							lang = l
							blocks = append(blocks, CodeBlock{Language: normalizeLang(lang), Code: code, Filename: lastLine})
							goto next
						}
					}
				}
			}
		}

		blocks = append(blocks, CodeBlock{Language: normalizeLang(lang), Code: code})
	next:
	}

	return blocks
}

func normalizeLang(lang string) string {
	switch lang {
	case "python", "py", "python3":
		return "python"
	case "bash", "sh", "shell", "zsh":
		return "bash"
	case "javascript", "js", "node":
		return "javascript"
	case "typescript", "ts":
		return "typescript"
	case "go", "golang":
		return "go"
	case "rust", "rs":
		return "rust"
	case "c":
		return "c"
	case "cpp", "c++", "cxx":
		return "cpp"
	default:
		return lang
	}
}

func isSandboxable(lang string) bool {
	switch lang {
	case "python", "javascript", "typescript", "go", "rust", "c", "cpp", "bash":
		return true
	}
	return false
}

func isInteractive(code string) bool {
	interactivePatterns := []string{
		"import curses", "from curses",
		"import tkinter", "from tkinter",
		"import pygame", "from pygame",
		"import turtle", "from turtle",
		"import PyQt", "from PyQt",
		"import wx", "from wx",
	}
	for _, pat := range interactivePatterns {
		if strings.Contains(code, pat) {
			return true
		}
	}
	return false
}

func containsCode(content string) bool {
	return strings.Contains(content, "```")
}

// ---------------------------------------------------------------------------
// Error analysis (ported from sandbox_analysis.py)
// ---------------------------------------------------------------------------

type ErrorAnalysis struct {
	ErrorType    string
	ErrorMsg     string
	FailureLine  string
	IsRecoverable bool
	Suggestion   string
}

func analyzeError(result *SandboxResult) ErrorAnalysis {
	analysis := ErrorAnalysis{IsRecoverable: true}

	if result.ErrorType != nil {
		analysis.ErrorType = *result.ErrorType
	}
	if result.ErrorMessage != nil {
		analysis.ErrorMsg = *result.ErrorMessage
	}

	stderr := result.Stderr

	// Extract line number from traceback
	lineRe := regexp.MustCompile(`line (\d+)`)
	if m := lineRe.FindStringSubmatch(stderr); len(m) > 1 {
		analysis.FailureLine = m[1]
	}

	// Classify error type and generate suggestion
	switch {
	case strings.Contains(stderr, "SyntaxError"):
		analysis.ErrorType = "SyntaxError"
		analysis.Suggestion = "Fix the syntax error — check parentheses, colons, indentation"
	case strings.Contains(stderr, "NameError"):
		analysis.ErrorType = "NameError"
		analysis.Suggestion = "A variable or function is used before definition — check spelling and scope"
	case strings.Contains(stderr, "TypeError"):
		analysis.ErrorType = "TypeError"
		analysis.Suggestion = "Wrong number of arguments or wrong type — check function signatures"
	case strings.Contains(stderr, "ImportError") || strings.Contains(stderr, "ModuleNotFoundError"):
		analysis.ErrorType = "ImportError"
		analysis.Suggestion = "Module not found — check imports and available packages"
	case strings.Contains(stderr, "IndexError"):
		analysis.ErrorType = "IndexError"
		analysis.Suggestion = "List index out of range — check array bounds"
	case strings.Contains(stderr, "KeyError"):
		analysis.ErrorType = "KeyError"
		analysis.Suggestion = "Dictionary key not found — check key names"
	case strings.Contains(stderr, "AttributeError"):
		analysis.ErrorType = "AttributeError"
		analysis.Suggestion = "Object doesn't have this attribute — check method/property names"
	case strings.Contains(stderr, "ValueError"):
		analysis.ErrorType = "ValueError"
		analysis.Suggestion = "Invalid value — check input data and type conversions"
	case strings.Contains(stderr, "could not find terminal"):
		analysis.ErrorType = "TerminalError"
		analysis.Suggestion = "Interactive program — cannot test in sandbox"
		analysis.IsRecoverable = false
	case strings.Contains(stderr, "MemoryError") || strings.Contains(stderr, "killed"):
		analysis.ErrorType = "ResourceError"
		analysis.Suggestion = "Out of memory or killed — reduce data size"
		analysis.IsRecoverable = false
	default:
		analysis.ErrorType = "RuntimeError"
		analysis.Suggestion = "Check the traceback and fix the issue"
	}

	return analysis
}

func buildRepairPrompt(code string, analysis ErrorAnalysis, attempt int) string {
	var sb strings.Builder
	sb.WriteString("The code has a bug. Here is the error analysis:\n\n")
	sb.WriteString(fmt.Sprintf("**Error Type**: %s\n", analysis.ErrorType))
	if analysis.ErrorMsg != "" {
		msg := analysis.ErrorMsg
		if len(msg) > 300 {
			msg = msg[:300] + "..."
		}
		sb.WriteString(fmt.Sprintf("**Error**: %s\n", msg))
	}
	if analysis.FailureLine != "" {
		sb.WriteString(fmt.Sprintf("**Line**: %s\n", analysis.FailureLine))
	}
	sb.WriteString(fmt.Sprintf("**Fix**: %s\n", analysis.Suggestion))
	sb.WriteString(fmt.Sprintf("\nThis is repair attempt %d/%d. ", attempt, maxRepairAttempts))
	sb.WriteString("Fix the specific error and return the COMPLETE corrected file. Do NOT explain — just return the fixed code.")
	return sb.String()
}

// ---------------------------------------------------------------------------
// Fox communication
// ---------------------------------------------------------------------------

func forwardToFox(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Model = modelName

	// Inject /nothink into the LAST user message (not necessarily the last message)
	// to prevent Qwen3.5 from wasting tokens on <think> blocks
	if len(req.Messages) > 0 {
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" && !strings.Contains(req.Messages[i].Content, "/nothink") {
				req.Messages[i] = ChatMessage{
					Role:    "user",
					Content: "/nothink\n" + req.Messages[i].Content,
				}
				break
			}
		}
	}

	body, _ := json.Marshal(req)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", inferenceURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("Fox HTTP %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(raw, &chatResp); err != nil {
		return nil, fmt.Errorf("parse error: %w\nraw: %s", err, truncate(string(raw), 200))
	}
	return &chatResp, nil
}

// ---------------------------------------------------------------------------
// Lens scoring
// ---------------------------------------------------------------------------

func scoreLens(ctx context.Context, text string) (*LensScore, error) {
	body, _ := json.Marshal(map[string]string{"text": text})
	req, err := http.NewRequestWithContext(ctx, "POST", lensURL+"/internal/lens/gx-score", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	var score LensScore
	if err := json.Unmarshal(raw, &score); err != nil {
		return nil, err
	}
	return &score, nil
}

// ---------------------------------------------------------------------------
// Sandbox execution
// ---------------------------------------------------------------------------

func runSandbox(ctx context.Context, code string, language string, timeout int) *SandboxResult {
	if language == "" {
		language = "python"
	}
	body, _ := json.Marshal(map[string]any{"code": code, "language": language, "timeout": timeout})
	req, err := http.NewRequestWithContext(ctx, "POST", sandboxURL+"/execute", bytes.NewReader(body))
	if err != nil {
		return nil
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: time.Duration(timeout+10) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("  sandbox unreachable: %v", err)
		return nil
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	var result SandboxResult
	if json.Unmarshal(raw, &result) != nil {
		return nil
	}
	return &result
}

// ---------------------------------------------------------------------------
// Verify-Repair Loop — THE CORE PIPELINE
// ---------------------------------------------------------------------------

type RepairResult struct {
	Response       *ChatResponse
	Score          *LensScore
	SandboxPassed  bool
	Attempts       int
	FinalCode      string
}

// ---------------------------------------------------------------------------
// Syntax validation — AST parse every edit before delivery
// ---------------------------------------------------------------------------

type SyntaxCheckResult struct {
	Valid   bool     `json:"valid"`
	Errors  []string `json:"errors"`
	Lang    string   `json:"language"`
	CheckMs int      `json:"check_time_ms"`
}

func syntaxCheck(ctx context.Context, code string, language string, filename string) *SyntaxCheckResult {
	payload := map[string]any{"code": code, "language": language}
	if filename != "" {
		payload["filename"] = filename
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, "POST", sandboxURL+"/syntax-check", bytes.NewReader(body))
	if err != nil {
		return nil
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("  syntax-check unreachable: %v", err)
		return nil
	}
	defer resp.Body.Close()

	var result SyntaxCheckResult
	if json.NewDecoder(resp.Body).Decode(&result) != nil {
		return nil
	}
	return &result
}

// syntaxRepairLoop checks all code blocks in a response for syntax errors.
// If any fail, re-prompts the model with the error up to maxRetries times.
func syntaxRepairLoop(ctx context.Context, req ChatRequest, content string, maxRetries int) string {
	for attempt := 0; attempt < maxRetries; attempt++ {
		blocks := extractCodeBlocks(content)
		if len(blocks) == 0 {
			return content
		}

		// Check each block
		var syntaxErrors []string
		for _, block := range blocks {
			if !isSandboxable(block.Language) {
				continue
			}
			result := syntaxCheck(ctx, block.Code, block.Language, "")
			if result == nil {
				continue // sandbox unreachable, skip
			}
			if !result.Valid {
				for _, e := range result.Errors {
					syntaxErrors = append(syntaxErrors, fmt.Sprintf("[%s] %s", block.Language, e))
				}
			}
		}

		if len(syntaxErrors) == 0 {
			if attempt > 0 {
				log.Printf("  syntax repair succeeded after %d retries", attempt)
			}
			return content
		}

		log.Printf("  syntax errors found (attempt %d/%d): %s", attempt+1, maxRetries, truncate(strings.Join(syntaxErrors, "; "), 200))

		// Re-prompt with specific errors
		errorMsg := "The code you generated has syntax errors. Fix ONLY the syntax errors and return the corrected code:\n\n"
		for _, e := range syntaxErrors {
			errorMsg += "- " + e + "\n"
		}
		errorMsg += "\nReturn the complete corrected file."

		repairReq := req
		repairReq.Stream = false
		repairReq.Messages = append(repairReq.Messages,
			ChatMessage{Role: "assistant", Content: content},
			ChatMessage{Role: "user", Content: errorMsg},
		)
		repairReq.Temperature = 0.2 // low temp for deterministic fix

		repairResp, err := forwardToFox(ctx, repairReq)
		if err != nil || len(repairResp.Choices) == 0 {
			log.Printf("  syntax repair request failed")
			return content
		}

		newContent := repairResp.Choices[0].Message.Content
		if strings.TrimSpace(newContent) == "" {
			log.Printf("  syntax repair returned empty")
			return content
		}

		content = newContent
	}

	log.Printf("  syntax repair exhausted %d retries", maxRetries)
	return content
}

// ---------------------------------------------------------------------------
// V3 Pipeline Service — calls the Python V3 service for full pipeline
// ---------------------------------------------------------------------------

type V3Result struct {
	Passed       bool              `json:"passed"`
	Code         string            `json:"code"`
	PhaseSolved  string            `json:"phase_solved"`
	Candidates   int               `json:"candidates_generated"`
	TotalTokens  int               `json:"total_tokens"`
	TotalTimeMs  float64           `json:"total_time_ms"`
	Events       []json.RawMessage `json:"events"`
}

// extractFileContext pulls file contents from Aider's messages.
// Aider includes file content in system messages as "filename.ext\n```\n...code...\n```"
func extractFileContext(messages []ChatMessage) map[string]string {
	files := make(map[string]string)
	for _, m := range messages {
		if m.Role != "system" && m.Role != "user" {
			continue
		}
		// Parse whole-file format blocks: filename\n```...\n```
		lines := strings.Split(m.Content, "\n")
		for i := 0; i < len(lines)-1; i++ {
			fname := strings.TrimSpace(lines[i])
			if fname == "" || strings.Contains(fname, " ") || len(fname) > 200 {
				continue
			}
			if !strings.Contains(fname, ".") {
				continue
			}
			// Check if next line is a fence
			nextTrimmed := strings.TrimSpace(lines[i+1])
			if strings.HasPrefix(nextTrimmed, "```") {
				// Collect content until closing fence
				var content strings.Builder
				for j := i + 2; j < len(lines); j++ {
					if strings.HasPrefix(strings.TrimSpace(lines[j]), "```") {
						files[fname] = content.String()
						break
					}
					content.WriteString(lines[j])
					content.WriteString("\n")
				}
			}
		}
	}
	return files
}

// runV3Pipeline calls the Python V3 service with SSE streaming.
// It streams progress events to the Aider client and returns the final code.
func runV3Pipeline(ctx context.Context, w http.ResponseWriter, flusher http.Flusher,
	problem string, taskID string, fileContext map[string]string) (*V3Result, error) {

	payload := map[string]any{
		"problem": problem,
		"task_id": taskID,
		"stream":  true,
	}
	if len(fileContext) > 0 {
		payload["files"] = fileContext
	}
	body, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", v3ServiceURL+"/v3/run", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: 600 * time.Second} // V3 pipeline can take minutes
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("V3 service unreachable: %v", err)
	}
	defer resp.Body.Close()

	// Stream V3 progress events to Aider as inline status messages
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var finalResult *V3Result

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "data: [DONE]" {
			break
		}

		if strings.HasPrefix(trimmed, "event: result") {
			// Next data line is the final result
			continue
		}

		if strings.HasPrefix(trimmed, "data: ") {
			payload := trimmed[6:]

			// Try to parse as final result
			var result V3Result
			if json.Unmarshal([]byte(payload), &result) == nil && result.Code != "" {
				finalResult = &result
				continue
			}

			// Parse as progress event and stream to user
			var event struct {
				Stage  string `json:"stage"`
				Detail string `json:"detail"`
			}
			if json.Unmarshal([]byte(payload), &event) == nil && event.Stage != "" {
				statusMsg := fmt.Sprintf("[%s] %s\n", event.Stage, event.Detail)
				log.Printf("  V3: %s", strings.TrimSpace(statusMsg))

				// Inject visible progress so user sees real-time pipeline status
				if flusher != nil {
					injectContentDelta(w, flusher, statusMsg)
				}
			}
		}
	}

	if finalResult == nil {
		return nil, fmt.Errorf("V3 service returned no result")
	}

	return finalResult, nil
}

// v3ServiceAvailable checks if the V3 Python service is running
func v3ServiceAvailable() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(v3ServiceURL + "/health")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == 200
}

// ---------------------------------------------------------------------------
// Format normalizer — exhaustive converter from ANY model output to Aider format
// ---------------------------------------------------------------------------
//
// Aider whole-file format:
//   filename.ext
//   ```language
//   ...code...
//   ```
//
// Model format variants observed:
//   V1: Correct Aider format (filename\n```lang\n...\n```)
//   V2: Code in fenced block but no filename before it
//   V3: Bare code with NO fences and NO filename (most common for 9B)
//   V4: Filename in bold/backtick/heading (**file** or `file` or # file)
//   V5: Filename after code block instead of before
//   V6: Multiple code blocks concatenated
//   V7: Code with filename in the info string (```tsx title="app/page.tsx")
//   V8: Prose + code mixed, filename mentioned in prose

var wholeFileBlockRe = regexp.MustCompile("(?m)^([\\w./\\-]+\\.[a-zA-Z]+)\\s*$")

// filenameFromExt guesses a filename extension's language tag
func langTagFromExt(ext string) string {
	switch ext {
	case ".py":
		return "python"
	case ".js", ".jsx":
		return "javascript"
	case ".ts", ".tsx":
		return "typescript"
	case ".go":
		return "go"
	case ".rs":
		return "rust"
	case ".c", ".h":
		return "c"
	case ".cpp", ".hpp", ".cc":
		return "cpp"
	case ".sh", ".bash":
		return "bash"
	case ".json":
		return "json"
	case ".html":
		return "html"
	case ".css":
		return "css"
	}
	return ""
}

// filenameRe matches things that look like file paths
// filenameRe matches file paths including Next.js dynamic routes with [brackets]
var filenameRe = regexp.MustCompile(`[\w./\-\[\]]+\.\w{1,10}`)

// normalizeToWholeFile converts ANY model output format to Aider whole-file format.
// targetFilename is the expected filename (from user message or file context).
// Returns (normalized content, true) if any transformation was applied.
func normalizeToWholeFile(content string, targetFilename string) (string, bool) {
	content = strings.TrimSpace(content)
	if content == "" {
		return content, false
	}

	// Detect target language from filename
	targetLang := ""
	if targetFilename != "" {
		if dotIdx := strings.LastIndex(targetFilename, "."); dotIdx >= 0 {
			targetLang = langTagFromExt(targetFilename[dotIdx:])
		}
	}

	// Check if already in valid Aider format: filename\n```lang\n...\n```
	lines := strings.Split(content, "\n")
	if len(lines) >= 3 {
		firstLine := strings.TrimSpace(lines[0])
		secondLine := strings.TrimSpace(lines[1])
		lastLine := strings.TrimSpace(lines[len(lines)-1])
		if strings.Contains(firstLine, ".") && !strings.Contains(firstLine, " ") &&
			strings.HasPrefix(secondLine, "```") && lastLine == "```" {
			cleaned := strings.Trim(firstLine, "*`\"'#: ")
			// Reject absolute paths — never valid as Aider filenames
			if strings.HasPrefix(cleaned, "/") {
				cleaned = ""
			}
			// If we have a target filename and the detected one is WRONG, fix it
			if targetFilename != "" && cleaned != targetFilename {
				log.Printf("  normalizer: replacing wrong filename '%s' → '%s'", cleaned, targetFilename)
				lines[0] = targetFilename
				return strings.Join(lines, "\n"), true
			}
			if cleaned != firstLine {
				lines[0] = cleaned
				return strings.Join(lines, "\n"), true
			}
			return content, false
		}
	}

	// VARIANT 2+4: Has fenced code block(s) but no/wrong filename before them
	if strings.Contains(content, "```") {
		// Extract code from fenced blocks
		matches := codeBlockRe.FindAllStringSubmatch(content, -1)
		if len(matches) > 0 {
			code := strings.TrimSpace(matches[0][2])
			lang := strings.TrimSpace(matches[0][1])
			if lang == "" && targetLang != "" {
				lang = targetLang
			}
			if targetFilename != "" && code != "" {
				result := targetFilename + "\n```" + lang + "\n" + code + "\n```"
				log.Printf("  normalizer: fenced block → Aider format (filename=%s)", targetFilename)
				return result, true
			}
		}
	}

	// VARIANT 3: Bare code with NO fences — most common for 9B model
	// Detect if content looks like code (has function defs, imports, JSX, etc.)
	looksLikeCode := false
	codeIndicators := []string{
		"import ", "export ", "from ", "require(",  // JS/TS imports
		"def ", "class ", "import ",                // Python
		"package ", "func ",                        // Go
		"fn ", "use ",                              // Rust
		"#include",                                 // C/C++
		"<div", "<main", "<section",                // JSX
		"'use client'", "\"use client\"",           // Next.js
		"export default",                           // ES modules
		"module.exports",                           // CommonJS
		"NextResponse",                             // Next.js API
	}
	for _, indicator := range codeIndicators {
		if strings.Contains(content, indicator) {
			looksLikeCode = true
			break
		}
	}

	if looksLikeCode && targetFilename != "" {
		lang := targetLang
		if lang == "" {
			lang = "python" // default
		}
		// Strip any prose before/after the code
		// Find where the actual code starts (first import/export/def/class/use client/etc.)
		codeStart := 0
		for i, line := range lines {
			trimmed := strings.TrimSpace(line)
			for _, indicator := range codeIndicators {
				if strings.HasPrefix(trimmed, indicator) || strings.Contains(trimmed, indicator) {
					codeStart = i
					goto foundStart
				}
			}
		}
	foundStart:
		codeContent := strings.Join(lines[codeStart:], "\n")
		codeContent = strings.TrimSpace(codeContent)

		result := targetFilename + "\n```" + lang + "\n" + codeContent + "\n```"
		log.Printf("  normalizer: bare code → Aider format (filename=%s, lang=%s, %d chars)", targetFilename, lang, len(codeContent))
		return result, true
	}

	// No transformation possible
	return content, false
}

func verifyAndRepair(ctx context.Context, originalReq ChatRequest, resp *ChatResponse) *RepairResult {
	if resp == nil || len(resp.Choices) == 0 {
		return &RepairResult{Response: resp}
	}

	content := resp.Choices[0].Message.Content
	if content == "" {
		return &RepairResult{Response: resp}
	}

	// Extract code blocks
	blocks := extractCodeBlocks(content)
	if len(blocks) == 0 {
		// No code → just score and return
		score, _ := scoreLens(ctx, content)
		return &RepairResult{Response: resp, Score: score}
	}

	// Find first sandboxable block
	var targetBlock *CodeBlock
	for i := range blocks {
		if isSandboxable(blocks[i].Language) {
			targetBlock = &blocks[i]
			break
		}
	}

	if targetBlock == nil {
		// No sandboxable code → score and return
		score, _ := scoreLens(ctx, content)
		return &RepairResult{Response: resp, Score: score}
	}

	// Score with C(x)/G(x)
	score, _ := scoreLens(ctx, content)

	// Interactive programs: test via PTY wrapper instead of skipping
	interactive := isInteractive(targetBlock.Code)

	// Sandbox test — wrap interactive programs in a PTY harness
	testCode := targetBlock.Code
	testTimeout := sandboxTimeout
	if interactive {
		testCode = wrapInPTY(targetBlock.Code)
		testTimeout = interactiveTimeout + 2
		log.Printf("  interactive program — using PTY wrapper")
	}
	log.Printf("  sandbox testing %s code (%d bytes)...", targetBlock.Language, len(targetBlock.Code))
	result := runSandbox(ctx, testCode, targetBlock.Language, testTimeout)

	if result == nil {
		// Sandbox unavailable → return with score only
		log.Printf("  sandbox unavailable — using G(x) only")
		return &RepairResult{Response: resp, Score: score}
	}

	if result.Success {
		sandboxPasses.Add(1)
		log.Printf("  sandbox PASSED (%.0fms)", result.ExecutionMs)
		passed := true
		return &RepairResult{Response: resp, Score: score, SandboxPassed: true, Attempts: 1, FinalCode: targetBlock.Code}
		_ = passed
	}

	sandboxFails.Add(1)
	log.Printf("  sandbox FAILED: %s", truncate(result.Stderr, 150))

	// Verify-repair loop
	bestResp := resp
	bestScore := score
	bestPassed := false
	currentCode := targetBlock.Code

	for attempt := 1; attempt <= maxRepairAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return &RepairResult{Response: bestResp, Score: bestScore, SandboxPassed: bestPassed, Attempts: attempt}
		default:
		}

		totalRepairs.Add(1)

		// Analyze the error
		analysis := analyzeError(result)
		if !analysis.IsRecoverable {
			log.Printf("  error not recoverable (%s) — stopping repair", analysis.ErrorType)
			break
		}

		// Build repair request
		repairPrompt := buildRepairPrompt(currentCode, analysis, attempt)
		repairReq := ChatRequest{
			Model:       originalReq.Model,
			Messages:    append(originalReq.Messages,
				ChatMessage{Role: "assistant", Content: bestResp.Choices[0].Message.Content},
				ChatMessage{Role: "user", Content: repairPrompt},
			),
			MaxTokens:   originalReq.MaxTokens,
			Temperature: 0.2 + float64(attempt)*0.1, // slightly increase temp each attempt
		}

		log.Printf("  repair attempt %d/%d...", attempt, maxRepairAttempts)

		repairResp, err := forwardToFox(ctx, repairReq)
		if err != nil {
			log.Printf("  repair generation failed: %v", err)
			break
		}

		if len(repairResp.Choices) == 0 || repairResp.Choices[0].Message.Content == "" {
			log.Printf("  repair returned empty")
			break
		}

		repairContent := repairResp.Choices[0].Message.Content

		// Extract code from repair response
		repairBlocks := extractCodeBlocks(repairContent)
		if len(repairBlocks) == 0 {
			// Model returned prose instead of code — try using the whole content
			log.Printf("  repair has no code blocks — skipping")
			break
		}

		repairCode := repairBlocks[0].Code

		// Score the repair
		repairScore, _ := scoreLens(ctx, repairContent)

		// Sandbox test the repair
		repairTestCode := repairCode
		if interactive {
			repairTestCode = wrapInPTY(repairCode)
		}
		repairResult := runSandbox(ctx, repairTestCode, targetBlock.Language, testTimeout)

		if repairResult != nil && repairResult.Success {
			sandboxPasses.Add(1)
			log.Printf("  repair %d PASSED sandbox! G(x)=%.2f", attempt, scoreVal(repairScore))
			return &RepairResult{
				Response:      repairResp,
				Score:         repairScore,
				SandboxPassed: true,
				Attempts:      attempt + 1,
				FinalCode:     repairCode,
			}
		}

		// Track best attempt by G(x) score
		if repairScore != nil && (bestScore == nil || repairScore.GxScore > bestScore.GxScore) {
			bestResp = repairResp
			bestScore = repairScore
		}

		if repairResult != nil {
			result = repairResult // use latest error for next iteration
			currentCode = repairCode
			log.Printf("  repair %d still failing: %s G(x)=%.2f", attempt, analysis.ErrorType, scoreVal(repairScore))
		} else {
			break
		}
	}

	return &RepairResult{Response: bestResp, Score: bestScore, SandboxPassed: bestPassed, Attempts: maxRepairAttempts + 1}
}

func scoreVal(s *LensScore) float64 {
	if s == nil {
		return 0
	}
	return s.GxScore
}

// ---------------------------------------------------------------------------
// Best-of-K generation
// ---------------------------------------------------------------------------

func bestOfK(ctx context.Context, req ChatRequest, k int) (*ChatResponse, *LensScore, error) {
	type candidate struct {
		resp  *ChatResponse
		score *LensScore
		idx   int
	}

	results := make(chan candidate, k)

	// Fire all K candidates in parallel — Fox has --parallel 4
	for i := 0; i < k; i++ {
		go func(idx int) {
			attempt := req
			attempt.Stream = false
			attempt.Temperature = req.Temperature + float64(idx)*0.15
			if attempt.Temperature > 1.0 {
				attempt.Temperature = 1.0
			}

			resp, err := forwardToFox(ctx, attempt)
			if err != nil || len(resp.Choices) == 0 || strings.TrimSpace(resp.Choices[0].Message.Content) == "" {
				results <- candidate{idx: idx}
				return
			}

			content := resp.Choices[0].Message.Content

			// Discard responses that are too short to contain useful code
			blocks := extractCodeBlocks(content)
			if len(blocks) == 0 {
				log.Printf("  [K=%d] no code blocks — discarding (%d chars)", idx+1, len(content))
				results <- candidate{idx: idx}
				return
			}

			// Check total code length — snake game needs at least 100 lines
			totalCodeLen := 0
			for _, b := range blocks {
				totalCodeLen += len(b.Code)
			}
			if totalCodeLen < 200 {
				log.Printf("  [K=%d] code too short (%d chars) — discarding", idx+1, totalCodeLen)
				results <- candidate{idx: idx}
				return
			}

			// Syntax check before scoring
			for _, b := range blocks {
				if isSandboxable(b.Language) {
					sc := syntaxCheck(ctx, b.Code, b.Language, "")
					if sc != nil && !sc.Valid {
						log.Printf("  [K=%d] syntax error — discarding", idx+1)
						results <- candidate{idx: idx}
						return
					}
				}
			}

			score, _ := scoreLens(ctx, content)
			log.Printf("  [K=%d] G(x)=%.2f len=%d", idx+1, scoreVal(score), len(content))
			results <- candidate{resp: resp, score: score, idx: idx}
		}(i)
	}

	// Collect results
	var bestResp *ChatResponse
	var bestScore *LensScore

	for i := 0; i < k; i++ {
		c := <-results
		if c.resp == nil {
			continue
		}
		if bestScore == nil || scoreVal(c.score) > scoreVal(bestScore) {
			bestResp = c.resp
			bestScore = c.score
		}
	}

	if bestResp == nil {
		return nil, nil, fmt.Errorf("all %d candidates failed", k)
	}
	return bestResp, bestScore, nil
}

// ---------------------------------------------------------------------------
// Main handler — non-streaming requests
// ---------------------------------------------------------------------------

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "bad request", 400)
		return
	}

	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON", 400)
		return
	}

	totalRequests.Add(1)

	// Streaming: use SSE passthrough with post-stream verification
	if req.Stream {
		handleStreamingChat(w, r, req)
		return
	}

	ctx := r.Context()
	start := time.Now()

	log.Printf("request: messages=%d max_tokens=%d", len(req.Messages), req.MaxTokens)

	// Classify intent
	tier := classifyIntent(ctx, req.Messages)
	log.Printf("  tier=%s", tier)

	// T0: lightweight system prompt for conversational messages
	if tier == Tier0Conversational {
		userText := ""
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				content := req.Messages[i].Content
				if strings.HasPrefix(content, "# *SEARCH/REPLACE") ||
					strings.HasPrefix(content, "To suggest changes") ||
					strings.HasPrefix(content, "I am not sharing") {
					continue
				}
				userText = content
				break
			}
		}
		if userText != "" {
			req.Messages = []ChatMessage{
				{Role: "system", Content: "You are a helpful coding assistant. Respond conversationally and concisely. Do not use code blocks or edit format instructions for conversational messages."},
				{Role: "user", Content: userText},
			}
			req.MaxTokens = 150
		}
	}

	// V3 pipeline for T2+ non-streaming requests (opt-in via ATLAS_V3_CLI=1)
	useV3CLI := os.Getenv("ATLAS_V3_CLI") == "1"
	if useV3CLI && tier >= Tier2Medium && v3ServiceAvailable() {
		log.Printf("  V3 pipeline (non-streaming)...")
		userProblem := ""
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				userProblem = req.Messages[i].Content
				userProblem = strings.TrimPrefix(userProblem, "/nothink\n")
				break
			}
		}
		fileCtx := extractFileContext(req.Messages)
		v3Result, v3Err := runV3Pipeline(ctx, nil, nil, userProblem, "cli", fileCtx)
		if v3Err == nil && v3Result.Code != "" {
			resp := &ChatResponse{
				ID:      "v3-" + fmt.Sprintf("%d", time.Now().UnixMilli()),
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   modelName,
				Choices: []ChatChoice{{
					Message: ChatMessage{Role: "assistant", Content: v3Result.Code},
				}},
			}
			resp.AtlasRoute = tier.String()
			if score, err := scoreLens(ctx, v3Result.Code); err == nil {
				resp.AtlasGxScore = score.GxScore
				resp.AtlasVerdict = score.Verdict
			}
			log.Printf("  V3 complete: phase=%s passed=%v", v3Result.PhaseSolved, v3Result.Passed)
			lastWasV3.Store(r.RemoteAddr, time.Now())
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
			return
		}
		if v3Err != nil {
			log.Printf("  V3 pipeline failed: %v — falling back", v3Err)
		}
	}

	// Pre-generation: spec for Tier 2+ tasks (DISABLED — spec injection overwhelms 9B model
	// causing truncated/empty responses. Aider's system prompt is sufficient.)
	if false && tier.NeedsSpec() {
		specReq := ChatRequest{
			Model: modelName,
			Messages: []ChatMessage{
				{Role: "system", Content: "You are a code reviewer. Output ONLY a checklist of SPECIFIC implementation requirements including exact function calls, API patterns, and common bugs to avoid. No prose."},
				{Role: "user", Content: req.Messages[len(req.Messages)-1].Content},
			},
			MaxTokens: 400, Temperature: 0.2,
		}
		specResp, specErr := forwardToFox(ctx, specReq)
		if specErr == nil && len(specResp.Choices) > 0 && len(specResp.Choices[0].Message.Content) > 20 {
			spec := specResp.Choices[0].Message.Content
			lastIdx := len(req.Messages) - 1
			req.Messages[lastIdx] = ChatMessage{
				Role:    req.Messages[lastIdx].Role,
				Content: req.Messages[lastIdx].Content + "\n\nIMPLEMENTATION CHECKLIST:\n" + spec,
			}
			log.Printf("  spec injected (%d chars)", len(spec))
		}
	}

	// Forward to Fox
	resp, err := forwardToFox(ctx, req)
	if err != nil {
		http.Error(w, fmt.Sprintf("upstream error: %v", err), 502)
		return
	}

	if len(resp.Choices) == 0 || strings.TrimSpace(resp.Choices[0].Message.Content) == "" {
		// Empty response — retry with higher temperature
		log.Printf("  empty response — retrying (temp=0.5)")
		retryReq := req
		retryReq.Temperature = 0.5
		retryResp, retryErr := forwardToFox(ctx, retryReq)
		if retryErr == nil && len(retryResp.Choices) > 0 && strings.TrimSpace(retryResp.Choices[0].Message.Content) != "" {
			resp = retryResp
			log.Printf("  retry succeeded: %d chars", len(resp.Choices[0].Message.Content))
		} else {
			// Second retry with temp=0.7
			log.Printf("  retry 1 failed — retrying (temp=0.7)")
			retryReq.Temperature = 0.7
			retryResp2, retryErr2 := forwardToFox(ctx, retryReq)
			if retryErr2 == nil && len(retryResp2.Choices) > 0 && strings.TrimSpace(retryResp2.Choices[0].Message.Content) != "" {
				resp = retryResp2
				log.Printf("  retry 2 succeeded: %d chars", len(resp.Choices[0].Message.Content))
			} else {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
				return
			}
		}
	}

	content := resp.Choices[0].Message.Content

	// Syntax validation loop — check all code blocks, retry on errors
	if containsCode(content) {
		repaired := syntaxRepairLoop(ctx, req, content, 3)
		if repaired != content {
			resp.Choices[0].Message.Content = repaired
			content = repaired
		}
	}

	// Run verify-repair pipeline if response contains code
	if containsCode(content) {
		result := verifyAndRepair(ctx, req, resp)
		resp = result.Response
		if result.Score != nil {
			resp.AtlasGxScore = result.Score.GxScore
			resp.AtlasVerdict = result.Score.Verdict

			// If low quality and sandbox didn't pass, OR tier 3 (hard), try best-of-K
			if (result.Score.GxScore < gxLowThreshold && !result.SandboxPassed) || tier.NeedsBOK() {
				log.Printf("  low G(x)=%.2f, triggering best-of-K", result.Score.GxScore)
				bokResp, bokScore, bokErr := bestOfK(ctx, req, 3)
				if bokErr == nil {
					// Verify the best-of-K winner too
					bokResult := verifyAndRepair(ctx, req, bokResp)
					if bokResult.SandboxPassed || scoreVal(bokResult.Score) > scoreVal(result.Score) {
						resp = bokResult.Response
						if bokResult.Score != nil {
							resp.AtlasGxScore = bokResult.Score.GxScore
							resp.AtlasVerdict = bokResult.Score.Verdict
						}
						result = bokResult
					}
				}
				_ = bokScore
			}
		}
		resp.AtlasRepairAttempt = result.Attempts
		if result.SandboxPassed {
			passed := true
			resp.AtlasSandboxPassed = &passed
		}
	} else {
		// Non-code response — just score
		if len(content) > 10 {
			if score, err := scoreLens(ctx, content); err == nil {
				resp.AtlasGxScore = score.GxScore
				resp.AtlasVerdict = score.Verdict
			}
		}
	}

	resp.AtlasRoute = tier.String()
	elapsed := time.Since(start)
	log.Printf("response: gx=%.2f verdict=%s sandbox=%v attempts=%d latency=%s len=%d",
		resp.AtlasGxScore, resp.AtlasVerdict, resp.AtlasSandboxPassed, resp.AtlasRepairAttempt,
		elapsed.Round(time.Millisecond), len(resp.Choices[0].Message.Content))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ---------------------------------------------------------------------------
// Streaming handler — SSE passthrough with post-stream verification
// ---------------------------------------------------------------------------

// (spec generation is now streamed inline in handleStreamingChat)

func handleStreamingChat(w http.ResponseWriter, r *http.Request, req ChatRequest) {
	// Set SSE headers early so we can stream spec + main generation
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	// Fast path: file deletion requests bypass the entire pipeline.
	// Delete the file from disk immediately and return empty SSE response.
	// Must run BEFORE tier classification to avoid the model generating text
	// that Aider would apply as a file edit.
	if os.Getenv("ATLAS_AGENT_LOOP") == "1" {
		userMsg := ""
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				userMsg = req.Messages[i].Content
				break
			}
		}
		if isDeleteRequest(userMsg) {
			log.Printf("  delete fast-path: %s", truncate(userMsg, 80))
			projectDir := detectRealProjectDir(req.Messages)
			if projectDir != "" {
				re := regexp.MustCompile(`[\w./\-\[\]]+\.\w{1,10}`)
				paths := re.FindAllString(userMsg, -1)
				for _, p := range paths {
					if p == "delete_file" || p == "remove_file" {
						continue
					}
					realPath := filepath.Join(projectDir, p)
					if _, err := os.Stat(realPath); err == nil {
						os.Remove(realPath)
						log.Printf("  deleted: %s", realPath)
					}
				}
			}
			// Send empty SSE response — no content for Aider to apply
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
	}

	// Classify intent using the model
	tier := classifyIntent(r.Context(), req.Messages)
	log.Printf("  tier=%s", tier)

	// T0: replace Aider's heavy system prompt with a minimal conversational one
	// This prevents the model from generating empty <think></think> for greetings
	if tier == Tier0Conversational {
		userText := ""
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				content := req.Messages[i].Content
				// Skip Aider's format reminders
				if strings.HasPrefix(content, "# *SEARCH/REPLACE") ||
					strings.HasPrefix(content, "To suggest changes") ||
					strings.HasPrefix(content, "I am not sharing") {
					continue
				}
				userText = content
				break
			}
		}
		if userText != "" {
			req.Messages = []ChatMessage{
				{Role: "system", Content: "You are a helpful coding assistant. Respond conversationally and concisely. Do not use code blocks or edit format instructions for conversational messages."},
				{Role: "user", Content: userText},
			}
			req.MaxTokens = 150
		}
	}

	// Pre-generation: generate spec for Tier 2+ (disabled for streaming/Aider path —
	// Aider's system prompt already provides edit format instructions, spec injection
	// overwhelms the 9B model causing empty/truncated responses)
	spec := ""
	if false && tier.NeedsSpec() {
		log.Printf("  generating spec for %s...", tier)

		specReq := ChatRequest{
			Model:   modelName,
			Stream:  true,
			Messages: []ChatMessage{
				{Role: "system", Content: `You are a code reviewer. Given a coding request, output ONLY a brief checklist of SPECIFIC implementation requirements. Include exact function calls, API patterns, and common bugs to avoid. No prose — just the checklist.`},
				{Role: "user", Content: req.Messages[len(req.Messages)-1].Content},
			},
			MaxTokens:   400,
			Temperature: 0.2,
		}
		specReq.Model = modelName
		specBody, _ := json.Marshal(specReq)

		specHttpReq, err := http.NewRequestWithContext(r.Context(), "POST", inferenceURL+"/v1/chat/completions", bytes.NewReader(specBody))
		if err == nil {
			specHttpReq.Header.Set("Content-Type", "application/json")
			specHttpReq.Header.Set("Accept", "text/event-stream")

			specClient := &http.Client{Timeout: 30 * time.Second}
			specResp, err := specClient.Do(specHttpReq)
			if err == nil {
				defer specResp.Body.Close()
				var specBuilder strings.Builder
				scanner := bufio.NewScanner(specResp.Body)
				scanner.Buffer(make([]byte, 64*1024), 64*1024)

				for scanner.Scan() {
					line := scanner.Text()
					trimmed := strings.TrimSpace(line)
					if trimmed == "data: [DONE]" {
						break
					}
					if strings.HasPrefix(trimmed, "data:") {
						payload := strings.TrimSpace(trimmed[5:])
						var event struct {
							Choices []struct {
								Delta struct {
									Content string `json:"content"`
								} `json:"delta"`
							} `json:"choices"`
						}
						if json.Unmarshal([]byte(payload), &event) == nil && len(event.Choices) > 0 {
							delta := event.Choices[0].Delta.Content
							specBuilder.WriteString(delta)
						}
					}
				}
				spec = specBuilder.String()
				if len(spec) > 20 {
					log.Printf("  spec streamed (%d chars)", len(spec))
					log.Printf("  spec ready, generating code...")
				}
			}
		}

		// Inject spec into the main generation prompt
		if spec != "" {
			enhanced := make([]ChatMessage, len(req.Messages))
			copy(enhanced, req.Messages)
			lastIdx := len(enhanced) - 1
			enhanced[lastIdx] = ChatMessage{
				Role:    enhanced[lastIdx].Role,
				Content: enhanced[lastIdx].Content + "\n\nIMPLEMENTATION CHECKLIST (follow these requirements):\n" + spec,
			}
			req.Messages = enhanced
		}
	}

	req.Model = modelName

	// For T2+: use the full V3 pipeline if service is available
	// This gives the exact same pipeline that scored 74.6% on LiveCodeBench
	// Double-request prevention: if last response was V3, skip V3 this time
	remoteIP := r.RemoteAddr
	v3Cooldown := false
	if ts, ok := lastWasV3.Load(remoteIP); ok {
		if time.Since(ts.(time.Time)) < 30*time.Second {
			v3Cooldown = true
			log.Printf("  V3 cooldown active — skipping pipeline (last V3 %.0fs ago)", time.Since(ts.(time.Time)).Seconds())
		}
	}

	// Agent loop: GBNF-constrained tool calls inside the proxy.
	// The proxy runs an internal agent loop (tools.go, agent.go), executes
	// file operations internally, then formats results for Aider's whole-file format.
	// Enabled via ATLAS_AGENT_LOOP=1. For T2/T3, runs V3 pipeline inside write_file.
	useAgentLoop := os.Getenv("ATLAS_AGENT_LOOP") == "1"
	if useAgentLoop && !v3Cooldown {
		// Agent loop handles ALL tiers when enabled (including T0 conversational).
		// Grammar enforcement prevents thinking blocks on all responses.
		// Override tier with fast heuristic — LLM classifier over-classifies
		// single-file creation as T3. The agent loop needs conservative tiers
		// to avoid triggering V3 pipeline on simple tasks.
		userMsg := ""
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				userMsg = req.Messages[i].Content
				break
			}
		}
		tier = classifyAgentTier(userMsg)
		log.Printf("  agent tier override: %s", tier)
		log.Printf("  agent loop: running internal tool-call loop for %s...", tier)
		agentResult := runInternalAgentLoop(req, tier, w, flusher)
		if agentResult != nil && len(agentResult.FileChanges) > 0 {
			// Format as Aider whole-file blocks and deliver via SSE
			aiderContent := formatForAider(agentResult)
			injectContentDelta(w, flusher, aiderContent)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			lastWasV3.Store(remoteIP, time.Now())
			return
		}
		// Agent loop produced no file changes — return text/summary if any
		if agentResult != nil && (len(agentResult.TextMessages) > 0 || agentResult.ToolCalls > 0 || agentResult.Summary != "") {
			aiderContent := formatForAider(agentResult)
			if aiderContent != "" {
				injectContentDelta(w, flusher, aiderContent)
			}
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
		log.Printf("  agent loop: no results — falling through to standard path")
	}

	// V3 pipeline (legacy): direct V3 service call without agent loop.
	// Available via ATLAS_V3_CLI=1. Superseded by ATLAS_AGENT_LOOP=1.
	useV3CLI := os.Getenv("ATLAS_V3_CLI") == "1"
	if useV3CLI && tier >= Tier2Medium && v3ServiceAvailable() && !v3Cooldown {
		log.Printf("  V3 pipeline: routing to full pipeline service...")

		// Extract the user's actual problem text
		userProblem := ""
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				userProblem = req.Messages[i].Content
				// Strip /nothink prefix and Aider format reminders
				userProblem = strings.TrimPrefix(userProblem, "/nothink\n")
				if idx := strings.Index(userProblem, "\n\nTo suggest changes"); idx > 0 {
					userProblem = userProblem[:idx]
				}
				break
			}
		}

		fileCtx := extractFileContext(req.Messages)
		v3Result, v3Err := runV3Pipeline(r.Context(), w, flusher, userProblem, "cli", fileCtx)
		if v3Err == nil && v3Result.Code != "" {
			content := v3Result.Code

			// Wrap in Aider whole-file format if not already
			if !strings.Contains(content, "```") {
				// Detect language and filename
				lang := "python"
				if strings.Contains(content, "package main") {
					lang = "go"
				} else if strings.Contains(content, "fn main") {
					lang = "rust"
				}
				// Extract filename from user request
				fname := "solution." + lang
				matches := filenameRe.FindAllString(userProblem, -1)
				for _, m := range matches {
					dotIdx := strings.LastIndex(m, ".")
					if dotIdx >= 0 && langTagFromExt(m[dotIdx:]) != "" {
						fname = m
						lang = langTagFromExt(m[dotIdx:])
						break
					}
				}
				content = fname + "\n```" + lang + "\n" + content + "\n```"
			}

			// Format repair — extract filename from the user's problem text
			v3TargetFile := ""
			v3Matches := filenameRe.FindAllString(userProblem, -1)
			for _, m := range v3Matches {
				if di := strings.LastIndex(m, "."); di >= 0 && langTagFromExt(m[di:]) != "" {
					v3TargetFile = m
					break
				}
			}
			if repaired, didRepair := normalizeToWholeFile(content, v3TargetFile); didRepair {
				content = repaired
			}

			log.Printf("  V3 pipeline complete: phase=%s passed=%v tokens=%d time=%.0fms",
				v3Result.PhaseSolved, v3Result.Passed, v3Result.TotalTokens, v3Result.TotalTimeMs)
			lastWasV3.Store(r.RemoteAddr, time.Now())

			injectContentDelta(w, flusher, content)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()

			// Async G(x) scoring
			go func() {
				if score, err := scoreLens(context.Background(), content); err == nil {
					log.Printf("  V3 stream-score: gx=%.2f verdict=%s", score.GxScore, score.Verdict)
				}
			}()
			return
		}
		if v3Err != nil {
			log.Printf("  V3 pipeline failed: %v — falling back to simple generation", v3Err)
		}
	}

	// Multi-file decomposition: if the user message mentions multiple files,
	// generate them one at a time. Each call only formats one file.
	if tier >= Tier1Simple {
		// Extract ALL target filenames from the user message
		targetFiles := []string{}
		for _, msg := range req.Messages {
			if msg.Role != "user" {
				continue
			}
			content := strings.TrimPrefix(msg.Content, "/nothink\n")
			matches := filenameRe.FindAllString(content, -1)
			seen := map[string]bool{}
			for _, m := range matches {
				dotIdx := strings.LastIndex(m, ".")
				if dotIdx < 0 {
					continue
				}
				ext := m[dotIdx:]
				// For multi-file decomposition, only count CODE files (not .json data files)
				if langTagFromExt(ext) == "" {
					continue
				}
				// Reject code refs
				base := m
				if si := strings.LastIndex(m, "/"); si >= 0 {
					base = m[si+1:]
				}
				if len(base) > 0 && base[0] >= 'A' && base[0] <= 'Z' && !strings.Contains(m, "/") {
					continue
				}
				if !seen[m] {
					targetFiles = append(targetFiles, m)
					seen[m] = true
				}
			}
		}

		// Only decompose if we're CREATING multiple new files, not editing existing ones
		hasExistingFiles := len(extractFileContext(req.Messages)) > 0
		if len(targetFiles) > 1 && !hasExistingFiles {
			// MULTI-FILE: generate one file per call, sequentially
			log.Printf("  multi-file decomposition: %d files detected: %v", len(targetFiles), targetFiles)
			var allContent strings.Builder
			createdFiles := []string{}
			previousFiles := "" // context of already-created files

			for _, fname := range targetFiles {
				lang := ""
				if di := strings.LastIndex(fname, "."); di >= 0 {
					lang = langTagFromExt(fname[di:])
				}

				// Build a focused single-file prompt
				singlePrompt := fmt.Sprintf(
					"Create ONLY the file %s. %s\n\nOutput ONLY this exact format with no other text:\n%s\n```%s\n<complete file content here>\n```\n\nDo not create any other files. Do not explain.",
					fname,
					previousFiles,
					fname, lang,
				)

				singleReq := ChatRequest{
					Model: modelName,
					Messages: []ChatMessage{
						{Role: "system", Content: "You create single files. Return the file in the exact format requested."},
						{Role: "user", Content: "/nothink\n" + singlePrompt},
					},
					MaxTokens:   4096,
					Temperature: 0.3,
					Stream:      false,
				}

				singleResp, singleErr := forwardToFox(r.Context(), singleReq)
				if singleErr != nil || len(singleResp.Choices) == 0 {
					log.Printf("  multi-file: failed to generate %s: %v", fname, singleErr)
					continue
				}

				fileContent := strings.TrimSpace(singleResp.Choices[0].Message.Content)
				if fileContent == "" {
					continue
				}

				// Normalize to Aider format
				normalized, _ := normalizeToWholeFile(fileContent, fname)
				if normalized != "" {
					allContent.WriteString(normalized)
					allContent.WriteString("\n\n")
					createdFiles = append(createdFiles, fname)
					previousFiles += fmt.Sprintf("\n(Already created: %s)", fname)
					log.Printf("  multi-file: created %s (%d chars)", fname, len(normalized))
				}
			}

			if len(createdFiles) > 0 {
				finalContent := strings.TrimSpace(allContent.String())
				log.Printf("  multi-file: delivering %d files (%d chars)", len(createdFiles), len(finalContent))
				injectContentDelta(w, flusher, finalContent)
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()

				go func() {
					if score, err := scoreLens(context.Background(), finalContent); err == nil {
						log.Printf("  multi-file score: gx=%.2f verdict=%s", score.GxScore, score.Verdict)
					}
				}()
				return
			}
			log.Printf("  multi-file: no files created — falling back to single generation")
		}
	}

	// For T1+ code tasks: buffer response, format-repair, then inject as SSE
	// This ensures Aider sees properly formatted whole-file output
	if tier >= Tier1Simple {
		log.Printf("  buffered generation for format-repair...")
		// Deep-copy messages so forwardToFox's /nothink injection doesn't
		// mutate the original req.Messages (needed for filename extraction later)
		bufMsgs := make([]ChatMessage, len(req.Messages))
		copy(bufMsgs, req.Messages)
		bufReq := req
		bufReq.Messages = bufMsgs
		bufReq.Stream = false
		bufResp, bufErr := forwardToFox(r.Context(), bufReq)
		if bufErr == nil && len(bufResp.Choices) > 0 && strings.TrimSpace(bufResp.Choices[0].Message.Content) != "" {
			content := bufResp.Choices[0].Message.Content

			// Syntax validation
			if containsCode(content) {
				content = syntaxRepairLoop(r.Context(), req, content, 3)
			}

			// Extract target filename from user's request — only "Create <file>"
			// Extract target filename — priority order:
			// 1. Filename from CURRENT user message (most specific to this request)
			// 2. Filename from file context (only if user message has no filename)
			userHintFilename := ""

			// Priority 1: filename from CURRENT user message
			// Must only look at the last real user message to avoid picking up
			// filenames from previous Aider turns in the conversation
			if userHintFilename == "" {
				currentMsg := ""
				for i := len(req.Messages) - 1; i >= 0; i-- {
					if req.Messages[i].Role != "user" {
						continue
					}
					c := req.Messages[i].Content
					if strings.HasPrefix(c, "# *SEARCH/REPLACE") ||
						strings.HasPrefix(c, "To suggest changes") ||
						strings.HasPrefix(c, "I am not sharing") {
						continue
					}
					currentMsg = c
					log.Printf("  filename-extract: current user msg = %s", truncate(c, 80))
					break
				}
				if currentMsg != "" {
					stripped := strings.TrimPrefix(currentMsg, "/nothink\n")
					matches := filenameRe.FindAllString(stripped, -1)
					for _, m := range matches {
						dotIdx := strings.LastIndex(m, ".")
						if dotIdx < 0 {
							continue
						}
						ext := m[dotIdx:]
						if langTagFromExt(ext) == "" {
							continue
						}
						baseName := m
						if si := strings.LastIndex(m, "/"); si >= 0 {
							baseName = m[si+1:]
						}
						if len(baseName) > 0 && baseName[0] >= 'A' && baseName[0] <= 'Z' && !strings.Contains(m, "/") {
							continue
						}
						userHintFilename = m
						log.Printf("  filename-extract: found '%s' from user msg", m)
						break
					}
					// Handle extensionless files: Dockerfile, Makefile, Procfile, etc.
					if userHintFilename == "" {
						for _, special := range []string{"Dockerfile", "Makefile", "Procfile", "Vagrantfile", "Gemfile", "Rakefile"} {
							if strings.Contains(stripped, special) {
								userHintFilename = special
								break
							}
						}
					}
				}
			}

			// Priority 2 fallback: file context (only for edit tasks when user msg has no filename)
			if userHintFilename == "" {
				fileCtxHere := extractFileContext(req.Messages)
				if len(fileCtxHere) == 1 {
					for fname := range fileCtxHere {
						userHintFilename = fname
					}
				}
			}

			// Normalize to Aider whole-file format
			normalized, didNorm := normalizeToWholeFile(content, userHintFilename)
			if didNorm {
				content = normalized
			} else if userHintFilename != "" && !strings.Contains(content, userHintFilename+"\n```") {
				// Retry with explicit format template
				log.Printf("  normalizer failed — retrying with format template for %s", userHintFilename)
				lang := ""
				if di := strings.LastIndex(userHintFilename, "."); di >= 0 {
					lang = langTagFromExt(userHintFilename[di:])
				}
				reformatReq := ChatRequest{
					Model: modelName,
					Messages: []ChatMessage{
						{Role: "user", Content: fmt.Sprintf("/nothink\nReformat this code as a file listing. Output ONLY this exact format:\n%s\n```%s\n<the code>\n```\n\nHere is the code to reformat:\n%s", userHintFilename, lang, content)},
					},
					MaxTokens: 4096, Temperature: 0, Stream: false,
				}
				reformatResp, reformatErr := forwardToFox(r.Context(), reformatReq)
				if reformatErr == nil && len(reformatResp.Choices) > 0 {
					reformatted := strings.TrimSpace(reformatResp.Choices[0].Message.Content)
					if normalized2, ok := normalizeToWholeFile(reformatted, userHintFilename); ok {
						content = normalized2
						log.Printf("  format retry succeeded: %d chars", len(content))
					}
				}
			}

			// Feature validation: check the requested feature is actually in the code
			// Extract the current user message for feature checking
			featureUserMsg := ""
			for i := len(req.Messages) - 1; i >= 0; i-- {
				if req.Messages[i].Role == "user" {
					c := req.Messages[i].Content
					if !strings.HasPrefix(c, "# *SEARCH/REPLACE") && !strings.HasPrefix(c, "To suggest changes") {
						featureUserMsg = strings.TrimPrefix(c, "/nothink\n")
						break
					}
				}
			}

			featureMissing := ""
			featureLower := strings.ToLower(featureUserMsg)
			// Extract actual code for feature checking (not prose)
			codeForCheck := ""
			for _, blk := range extractCodeBlocks(content) {
				codeForCheck += blk.Code + "\n"
			}
			if codeForCheck == "" {
				codeForCheck = content // fallback to full content
			}
			contentLower := strings.ToLower(codeForCheck)

			// Feature-specific validators
			if strings.Contains(featureLower, "high score") || strings.Contains(featureLower, "highscore") {
				if !strings.Contains(contentLower, "json.dump") && !strings.Contains(contentLower, "json.load") &&
					!strings.Contains(contentLower, "open(") && !strings.Contains(contentLower, "write(") {
					featureMissing = "persistent file I/O (json.dump/json.load/open) for high scores"
				}
			}
			if strings.Contains(featureLower, "persist") || strings.Contains(featureLower, "save to") {
				if !strings.Contains(contentLower, "open(") && !strings.Contains(contentLower, "write(") &&
					!strings.Contains(contentLower, "json.dump") {
					featureMissing = "file persistence (open/write/json.dump)"
				}
			}
			if strings.Contains(featureLower, "json") && !strings.Contains(contentLower, "import json") &&
				!strings.Contains(contentLower, "json.") {
				featureMissing = "json module usage"
			}
			if strings.Contains(featureLower, "sqlite") && !strings.Contains(contentLower, "sqlite") {
				featureMissing = "sqlite database"
			}
			if strings.Contains(featureLower, "jwt") && !strings.Contains(contentLower, "jwt") &&
				!strings.Contains(contentLower, "token") {
				featureMissing = "JWT/token authentication"
			}

			if featureMissing != "" && len(extractCodeBlocks(content)) > 0 {
				log.Printf("  feature validation FAILED: missing %s — re-prompting", featureMissing)
				// Fresh focused call with existing code + correction
				existingCode := ""
				for _, blk := range extractCodeBlocks(content) {
					existingCode = blk.Code
					break
				}
				correction := fmt.Sprintf(
					"/nothink\nThis code is missing %s. Add it to the code below and return the COMPLETE updated file.\n\nRequirements:\n- %s\n- import json at the top\n- Use json.dump() to save and json.load() to read\n- Use open() for file I/O\n\nExisting code:\n```\n%s\n```\n\nReturn the complete updated file with the feature added.",
					featureMissing, featureUserMsg, existingCode,
				)
				retryReq := ChatRequest{
					Model: modelName,
					Messages: []ChatMessage{
						{Role: "system", Content: "You add features to existing code. Return the COMPLETE file."},
						{Role: "user", Content: correction},
					},
					MaxTokens: 8192, Temperature: 0.3, Stream: false,
				}
				retryResp, retryErr := forwardToFox(r.Context(), retryReq)
				if retryErr == nil && len(retryResp.Choices) > 0 {
					retryContent := strings.TrimSpace(retryResp.Choices[0].Message.Content)
					retryLower := strings.ToLower(retryContent)
					// Check if retry has the feature
					if (strings.Contains(retryLower, "json.dump") || strings.Contains(retryLower, "open(") ||
						strings.Contains(retryLower, "sqlite") || strings.Contains(retryLower, "jwt")) &&
						len(retryContent) > len(content)/2 {
						content = retryContent
						log.Printf("  feature retry succeeded (%d chars)", len(content))
						// Re-normalize
						if norm, ok := normalizeToWholeFile(content, userHintFilename); ok {
							content = norm
						}
					}
				}
			}

			// Sandbox verification on code blocks
			if containsCode(content) {
				blocks := extractCodeBlocks(content)
				for _, block := range blocks {
					if isSandboxable(block.Language) {
						sr := runSandbox(r.Context(), block.Code, block.Language, 10)
						if sr != nil {
							log.Printf("  sandbox: lang=%s success=%v", block.Language, sr.Success)
						}
						break // test first sandboxable block
					}
				}
			}

			// Validate: content must have code blocks for Aider to create files
			blocks := extractCodeBlocks(content)
			if len(blocks) == 0 && tier >= Tier1Simple {
				log.Printf("  buffered response has no code blocks — falling back to stream")
			} else {
				log.Printf("  delivering %d chars (%d code blocks)", len(content), len(blocks))

				// Direct file write fallback: if we have a target filename and code,
				// write the file directly. This bypasses Aider's parser which sometimes
				// drops SSE-injected content silently.
				if userHintFilename != "" && len(blocks) > 0 {
					// Find CWD from Aider's Referer or file context
					cwd := ""
					for _, msg := range req.Messages {
						if msg.Role == "system" {
							// Look for repo path indicators
							for _, line := range strings.Split(msg.Content, "\n") {
								t := strings.TrimSpace(line)
								if strings.HasPrefix(t, "/tmp/") || strings.HasPrefix(t, "/home/") {
									if idx := strings.Index(t, "/"); idx >= 0 {
										candidate := t
										if sidx := strings.Index(candidate, " "); sidx > 0 {
											candidate = candidate[:sidx]
										}
										if _, err := os.Stat(candidate); err == nil {
											cwd = candidate
											break
										}
									}
								}
							}
						}
					}
					if cwd == "" {
						// Find the most recently modified /tmp/atlas-* directory with .git
						entries, _ := os.ReadDir("/tmp")
						var newest string
						var newestTime int64
						for _, e := range entries {
							if e.IsDir() && strings.HasPrefix(e.Name(), "atlas-") {
								gitPath := "/tmp/" + e.Name() + "/.git"
								if info, err := os.Stat(gitPath); err == nil {
									if info.ModTime().Unix() > newestTime {
										newestTime = info.ModTime().Unix()
										newest = "/tmp/" + e.Name()
									}
								}
							}
						}
						if newest != "" {
							cwd = newest
						}
					}
					if cwd != "" {
						fullPath := cwd + "/" + userHintFilename
						dir := fullPath[:strings.LastIndex(fullPath, "/")]
						os.MkdirAll(dir, 0755)
						if err := os.WriteFile(fullPath, []byte(blocks[0].Code+"\n"), 0644); err == nil {
							log.Printf("  direct-write: %s (%d bytes)", fullPath, len(blocks[0].Code))
						}
					}
				}

				// Check for file deletion intent
				userMsg := ""
				for _, msg := range req.Messages {
					if msg.Role == "user" {
						userMsg = msg.Content
					}
				}
				delFiles := detectDeletionIntent(userMsg, content)
				if len(delFiles) > 0 {
					log.Printf("  deletion intent: %v", delFiles)

					// Find repo root from Aider's system message
					// Aider sends "Repo: /path/to/repo" or includes file paths
					repoRoot := ""
					for _, msg := range req.Messages {
						if msg.Role == "system" {
							// Look for absolute paths in system message
							for _, line := range strings.Split(msg.Content, "\n") {
								trimLine := strings.TrimSpace(line)
								if strings.HasPrefix(trimLine, "/") && strings.Contains(trimLine, "/") {
									// This might be a file path — extract directory
									if idx := strings.LastIndex(trimLine, "/"); idx > 0 {
										candidate := trimLine[:idx]
										if _, err := os.Stat(candidate); err == nil {
											repoRoot = candidate
											break
										}
									}
								}
							}
						}
					}

					// If still no repo root, try /proc/self/cwd or common locations
					if repoRoot == "" {
						// Walk through file context to find existing files
						for fname := range extractFileContext(req.Messages) {
							// Try to find this file on disk to determine CWD
							delEntries, _ := os.ReadDir("/tmp")
						for _, de := range delEntries {
							base := "/tmp/" + de.Name()
							if !de.IsDir() || !strings.HasPrefix(de.Name(), "atlas-") { continue }
								if _, err := os.Stat(base + "/" + fname); err == nil {
									repoRoot = base
									break
								}
							}
							if repoRoot != "" {
								break
							}
						}
					}

					// Execute deletions
					deletedFiles := []string{}
					for _, f := range delFiles {
						var fullPath string
						if repoRoot != "" {
							fullPath = repoRoot + "/" + f
						} else {
							fullPath = f // try relative
						}
						if _, err := os.Stat(fullPath); err == nil {
							if rmErr := os.RemoveAll(fullPath); rmErr == nil {
								log.Printf("  deleted: %s", fullPath)
								deletedFiles = append(deletedFiles, f)
							} else {
								log.Printf("  delete failed: %s: %v", fullPath, rmErr)
							}
						}
					}

					if len(deletedFiles) > 0 {
						content += "\n\nDeleted files: " + strings.Join(deletedFiles, ", ") + "\n"
					}
				}

				injectContentDelta(w, flusher, content)
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()

				go func() {
					if score, err := scoreLens(context.Background(), content); err == nil {
						log.Printf("  stream-score: gx=%.2f verdict=%s len=%d", score.GxScore, score.Verdict, len(content))
					}
				}()
				return
			}
		}
		log.Printf("  buffered generation failed — falling back to stream")
	}

	body, _ := json.Marshal(req)

	foxReq, err := http.NewRequestWithContext(r.Context(), "POST", inferenceURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	foxReq.Header.Set("Content-Type", "application/json")
	foxReq.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: 300 * time.Second}
	foxResp, err := client.Do(foxReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Fox error: %v", err), 502)
		return
	}
	defer foxResp.Body.Close()

	// SSE headers already set above — reuse flusher
	if foxResp.StatusCode >= 400 {
		raw, _ := io.ReadAll(foxResp.Body)
		log.Printf("  Fox error: %s", truncate(string(raw), 100))
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
		return
	}

	// Stream from Fox, intercept [DONE], accumulate content
	var fullContent strings.Builder
	scanner := bufio.NewScanner(foxResp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Intercept [DONE] — hold it for post-verification
		if trimmed == "data: [DONE]" {
			break
		}

		// Forward everything else to client
		fmt.Fprintf(w, "%s\n", line)
		flusher.Flush()

		// Accumulate content from delta events
		if strings.HasPrefix(trimmed, "data:") {
			payload := strings.TrimSpace(trimmed[5:])
			var event struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if json.Unmarshal([]byte(payload), &event) == nil && len(event.Choices) > 0 {
				fullContent.WriteString(event.Choices[0].Delta.Content)
			}
		}
	}

	// Stream is done from Fox. Client hasn't received [DONE] yet.
	content := fullContent.String()

	// Empty response retry — if model produced nothing (think-only), retry non-streamed
	if strings.TrimSpace(content) == "" {
		log.Printf("  empty stream response — retrying non-streamed")
		retryReq := req
		retryReq.Stream = false
		retryReq.Temperature = 0.5 // slightly higher temp to break out of think loop
		retryResp, retryErr := forwardToFox(r.Context(), retryReq)
		if retryErr == nil && len(retryResp.Choices) > 0 && retryResp.Choices[0].Message.Content != "" {
			content = retryResp.Choices[0].Message.Content
			log.Printf("  retry succeeded: %d chars", len(content))
			// Inject the retry content as SSE chunks
			injectContentDelta(w, flusher, content)
		} else {
			log.Printf("  retry also empty")
		}
	}

	// Syntax validation loop for streamed responses
	if containsCode(content) {
		repaired := syntaxRepairLoop(r.Context(), req, content, 3)
		if repaired != content {
			// Inject the corrected content as additional SSE chunks
			log.Printf("  syntax repair changed streamed content — injecting correction")
			injectContentDelta(w, flusher, "\n\n[Syntax corrected]\n"+repaired)
			content = repaired
		}
	}

	// Post-stream verification (non-blocking — only for code responses)
	if containsCode(content) {
		blocks := extractCodeBlocks(content)
		var sandboxable *CodeBlock
		for i := range blocks {
			if isSandboxable(blocks[i].Language) {
				sandboxable = &blocks[i]
				break
			}
		}

		if sandboxable != nil {
			// Use PTY wrapper for interactive programs
			sandboxCode := sandboxable.Code
			sandboxTout := sandboxTimeout
			if isInteractive(sandboxable.Code) {
				sandboxCode = wrapInPTY(sandboxable.Code)
				sandboxTout = interactiveTimeout + 2
				log.Printf("  verifying interactive code...")
			} else {
				log.Printf("  verifying code...")
			}

			result := runSandbox(r.Context(), sandboxCode, sandboxable.Language, sandboxTout)
			if result != nil && result.Success {
				sandboxPasses.Add(1)
				log.Printf("  sandbox verified OK")
				log.Printf("  stream sandbox PASSED")
			} else if result != nil {
				sandboxFails.Add(1)
				analysis := analyzeError(result)
				log.Printf("  stream sandbox FAILED: %s", analysis.ErrorType)

				// Attempt repair
				if analysis.IsRecoverable {
					log.Printf("  bug detected: %s — repairing...", analysis.ErrorType)

					repairPrompt := buildRepairPrompt(sandboxable.Code, analysis, 1)
					repairReq := ChatRequest{
						Model:    req.Model,
						Messages: append(req.Messages,
							ChatMessage{Role: "assistant", Content: content},
							ChatMessage{Role: "user", Content: repairPrompt},
						),
						MaxTokens:   req.MaxTokens,
						Temperature: 0.3,
					}

					repairResp, repairErr := forwardToFox(r.Context(), repairReq)
					if repairErr == nil && len(repairResp.Choices) > 0 {
						repairContent := repairResp.Choices[0].Message.Content
						repairBlocks := extractCodeBlocks(repairContent)

						if len(repairBlocks) > 0 {
							// Sandbox test the repair (use PTY wrapper if interactive)
							repairSandboxCode := repairBlocks[0].Code
							if isInteractive(repairBlocks[0].Code) {
								repairSandboxCode = wrapInPTY(repairBlocks[0].Code)
							}
							repairResult := runSandbox(r.Context(), repairSandboxCode, sandboxable.Language, sandboxTout)
							if repairResult != nil && repairResult.Success {
								sandboxPasses.Add(1)
								log.Printf("  stream repair PASSED (%d bytes)", len(repairContent))
							} else {
								log.Printf("  repair attempted but still has issues")
								log.Printf("  stream repair still failing")
							}
						}
					}
				}
			}
		}
	}

	// Score in background
	if len(content) > 10 {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if score, err := scoreLens(ctx, content); err == nil {
				log.Printf("  stream-score: gx=%.2f verdict=%s len=%d", score.GxScore, score.Verdict, len(content))
			}
		}()
	}

	// Send [DONE]
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// detectDeletionIntent checks if the user or model wants to delete files.
// Returns a list of filenames to delete.
func detectDeletionIntent(userMsg string, modelResponse string) []string {
	lower := strings.ToLower(userMsg)

	// Check for deletion keywords
	hasDelete := false
	for _, kw := range []string{"remove", "delete", "revert", "clean up", "undo", "get rid of"} {
		if strings.Contains(lower, kw) {
			hasDelete = true
			break
		}
	}
	if !hasDelete {
		return nil
	}

	// DON'T include files that are being EDITED (mentioned with --file or as edit targets)
	// Only include files that should be DELETED (auth files when reverting auth, etc.)
	// Strategy: look for auth/login-related paths when revert/remove is about auth
	var filesToDelete []string

	// If the user mentions specific files to edit (e.g., --file app/page.tsx),
	// those are NOT deletion targets
	editTargets := map[string]bool{}
	// The file being edited is usually the one after "from" or direct --file ref
	editMatches := filenameRe.FindAllString(userMsg, -1)
	for _, m := range editMatches {
		editTargets[m] = true
	}

	// Look for deletion-context filenames in the model's response
	// The model often says "I've removed the auth files" or "delete middleware.ts"
	respLower := strings.ToLower(modelResponse)
	deleteContextWords := []string{"delete", "remove", "deleted", "removed", "no longer needed"}
	for _, dcw := range deleteContextWords {
		idx := strings.Index(respLower, dcw)
		if idx < 0 {
			continue
		}
		// Search around the deletion word for filenames
		window := modelResponse[max(0, idx-50):min(len(modelResponse), idx+200)]
		windowMatches := filenameRe.FindAllString(window, -1)
		for _, m := range windowMatches {
			dotIdx := strings.LastIndex(m, ".")
			if dotIdx < 0 {
				continue
			}
			ext := m[dotIdx:]
			if langTagFromExt(ext) == "" {
				continue
			}
			if !editTargets[m] {
				filesToDelete = append(filesToDelete, m)
			}
		}
	}

	// If reverting auth, look for common auth file patterns
	if strings.Contains(lower, "auth") || strings.Contains(lower, "login") {
		authFiles := []string{
			"app/login/page.tsx", "app/login/page.ts", "app/login/page.jsx",
			"middleware.ts", "middleware.js",
			"app/api/auth/route.ts", "app/api/auth/route.js",
			"app/api/auth/login/route.ts", "app/api/auth/login/route.js",
		}
		for _, af := range authFiles {
			if !editTargets[af] {
				filesToDelete = append(filesToDelete, af)
			}
		}
	}

	// Also check model response for "delete" + filename patterns
	if strings.Contains(strings.ToLower(modelResponse), "delete") || strings.Contains(strings.ToLower(modelResponse), "remove") {
		respMatches := filenameRe.FindAllString(modelResponse, -1)
		for _, m := range respMatches {
			dotIdx := strings.LastIndex(m, ".")
			if dotIdx >= 0 && langTagFromExt(m[dotIdx:]) != "" {
				// Check it's not a create/keep reference
				already := false
				for _, f := range filesToDelete {
					if f == m {
						already = true
					}
				}
				if !already {
					filesToDelete = append(filesToDelete, m)
				}
			}
		}
	}

	return filesToDelete
}

// injectContentDelta sends a synthetic SSE content delta to the client
func injectContentDelta(w http.ResponseWriter, flusher http.Flusher, text string) {
	chunk := map[string]any{
		"id":      "atlas-verify",
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   modelName,
		"choices": []map[string]any{
			{
				"index":         0,
				"delta":         map[string]string{"content": text},
				"finish_reason": nil,
			},
		},
	}
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// ---------------------------------------------------------------------------
// HTTP server setup
// ---------------------------------------------------------------------------

func handleModels(w http.ResponseWriter, r *http.Request) {
	resp := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": "atlas", "object": "model", "owned_by": "atlas"},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	foxOK, ragOK, sandboxOK := false, false, false

	if resp, err := http.Get(inferenceURL + "/health"); err == nil {
		resp.Body.Close()
		foxOK = resp.StatusCode == 200
	}
	if resp, err := http.Get(lensURL + "/health"); err == nil {
		resp.Body.Close()
		ragOK = resp.StatusCode == 200
	}
	if resp, err := http.Get(sandboxURL + "/health"); err == nil {
		resp.Body.Close()
		sandboxOK = resp.StatusCode == 200
	}

	status := map[string]any{
		"status":   "ok",
		"inference":      foxOK,
		"lens":  ragOK,
		"sandbox":  sandboxOK,
		"port":     proxyPort,
		"stats": map[string]int64{
			"requests":       totalRequests.Load(),
			"repairs":        totalRepairs.Load(),
			"sandbox_passes": sandboxPasses.Load(),
			"sandbox_fails":  sandboxFails.Load(),
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/chat/completions", handleChatCompletions)
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/models", handleModels)
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/v1/agent", handleAgent) // New tool-based agent endpoint

	// Catch-all: proxy to Fox
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("passthrough: %s %s", r.Method, r.URL.Path)
		body, _ := io.ReadAll(r.Body)
		proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, inferenceURL+r.URL.Path, bytes.NewReader(body))
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		proxyReq.Header = r.Header
		resp, err := http.DefaultClient.Do(proxyReq)
		if err != nil {
			http.Error(w, err.Error(), 502)
			return
		}
		defer resp.Body.Close()
		for k, v := range resp.Header {
			for _, vv := range v {
				w.Header().Add(k, vv)
			}
		}
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	})

	addr := ":" + proxyPort
	log.Printf("ATLAS Proxy v2.0 starting on %s", addr)
	log.Printf("  Inference: %s", inferenceURL)
	log.Printf("  Geometric Lens: %s", lensURL)
	log.Printf("  Sandbox: %s", sandboxURL)
	log.Printf("  Pipeline: generate → score → sandbox → repair (max %d) → deliver", maxRepairAttempts)

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// wrapInPTY wraps code in a PTY harness so interactive programs (curses, etc.)
// can run in the sandbox without a real terminal.
func wrapInPTY(userCode string) string {
	// Triple-quote safe: replace any ''' in user code
	safe := strings.ReplaceAll(userCode, "'''", `\'\'\'`)

	return `import os, pty, sys, time, signal, tempfile, shutil

code = '''` + safe + `'''

tmpdir = tempfile.mkdtemp()
codefile = os.path.join(tmpdir, "test_program.py")
with open(codefile, "w") as f:
    f.write(code)

master, slave = pty.openpty()
pid = os.fork()

if pid == 0:
    os.close(master)
    os.setsid()
    os.dup2(slave, 0)
    os.dup2(slave, 1)
    os.dup2(slave, 2)
    os.close(slave)
    os.environ["TERM"] = "xterm-256color"
    os.environ["LINES"] = "24"
    os.environ["COLUMNS"] = "80"
    try:
        with open(codefile) as f:
            compiled = compile(f.read(), codefile, "exec")
            exec(compiled)
    except SystemExit:
        pass
    except Exception as e:
        sys.stderr.write(f"CRASH: {type(e).__name__}: {e}\n")
        os._exit(1)
    os._exit(0)
else:
    os.close(slave)
    start = time.time()
    timeout = 3
    exited = False
    exit_code = -1
    while time.time() - start < timeout:
        r, status = os.waitpid(pid, os.WNOHANG)
        if r != 0:
            exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            exited = True
            break
        time.sleep(0.1)
    if not exited:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.1)
        try:
            os.waitpid(pid, 0)
        except:
            pass
        print("TIMEOUT_OK")
    elif exit_code == 0:
        print("EXIT_OK")
    else:
        import select
        err = ""
        try:
            if select.select([master], [], [], 0.1)[0]:
                err = os.read(master, 4096).decode("utf-8", errors="replace")
        except:
            pass
        print(f"EXIT_FAIL:{exit_code}")
        if err:
            print(err[:500])
    os.close(master)
    shutil.rmtree(tmpdir, ignore_errors=True)
`
}

// ---------------------------------------------------------------------------
// Model-based intent classification (Section 1 of production checklist)
// ---------------------------------------------------------------------------

// Tier represents the complexity classification of a request
type Tier int

const (
	Tier0Conversational Tier = 0 // instant response, no pipeline
	Tier1Simple         Tier = 1 // single file, obvious intent
	Tier2Medium         Tier = 2 // multi-file awareness, spec + verify
	Tier3Hard           Tier = 3 // full pipeline, best-of-K, multi-step verify
)

func (t Tier) String() string {
	switch t {
	case Tier0Conversational:
		return "T0:chat"
	case Tier1Simple:
		return "T1:simple"
	case Tier2Medium:
		return "T2:medium"
	case Tier3Hard:
		return "T3:hard"
	}
	return "T?:unknown"
}

func (t Tier) NeedsSpec() bool  { return t >= Tier2Medium }
func (t Tier) NeedsBOK() bool   { return t >= Tier3Hard }

// classifyIntent uses the model itself to determine complexity tier.
// Single Fox call with constrained output — returns in ~200ms on GPU.
func classifyIntent(ctx context.Context, messages []ChatMessage) Tier {
	if len(messages) == 0 {
		return Tier0Conversational
	}

	// Extract the user's actual input (strip aider's appended instructions)
	// Aider appends edit format instructions after \n\n with various prefixes
	// Find the actual user message — Aider appends format reminders as the last "user" message
	// Scan backwards to find the real user input (not a format reminder)
	lastContent := ""
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		content := messages[i].Content
		// Skip Aider's format reminder messages
		if strings.HasPrefix(content, "# *SEARCH/REPLACE") ||
			strings.HasPrefix(content, "# File editing rules") ||
			strings.HasPrefix(content, "Return edits similar") ||
			strings.HasPrefix(content, "I am not sharing any files") ||
			strings.HasPrefix(content, "I switched to a new code base") {
			continue
		}
		// Strip any appended instructions from the actual user message
		if idx := strings.Index(content, "\n\nTo suggest changes"); idx > 0 {
			content = content[:idx]
		}
		if idx := strings.Index(content, "\n\n# File editing rules"); idx > 0 {
			content = content[:idx]
		}
		if idx := strings.Index(content, "\n\nIMPLEMENTATION CHECKLIST"); idx > 0 {
			content = content[:idx]
		}
		if idx := strings.Index(content, "\n\nYou MUST use"); idx > 0 {
			content = content[:idx]
		}
		lastContent = content
		break
	}
	if lastContent == "" && len(messages) > 0 {
		lastContent = messages[len(messages)-1].Content
	}
	log.Printf("  classify input: %s", truncate(strings.TrimSpace(lastContent), 100))

	// Very short messages that are obviously conversational — skip the model call
	trimmed := strings.TrimSpace(lastContent)
	if len(trimmed) < 5 {
		return Tier0Conversational
	}

	// Quick conversational patterns — skip the model call entirely for T0
	lowerTrimmed := strings.ToLower(trimmed)
	t0Patterns := []string{
		"hi", "hey", "hello", "sup", "yo",
		"thanks", "thank you", "thx", "ty",
		"ok", "okay", "k", "sure", "yep", "yup", "nope",
		"yes", "no", "y", "n",
		"got it", "sounds good", "perfect", "great", "cool", "nice",
		"never mind", "nvm", "forget it", "cancel",
		"what can you do", "help", "how do i exit",
		"what did you change", "what changed", "show me the diff",
		"undo", "undo that", "revert", "revert that",
	}
	for _, pat := range t0Patterns {
		if lowerTrimmed == pat || lowerTrimmed == pat+"!" || lowerTrimmed == pat+"." || lowerTrimmed == pat+"?" {
			return Tier0Conversational
		}
	}

	// Single word that's not a command → likely conversational
	if !strings.Contains(trimmed, " ") && len(trimmed) < 15 {
		return Tier0Conversational
	}

	// File-context awareness: if Aider sent file contents, this is an edit (T1/T2)
	// Aider includes file content in system messages like "filename.py\n```\n...code...\n```"
	// or in user messages with file paths. If we detect file content, cap at T2.
	hasFileContext := false
	for _, m := range messages {
		if m.Role == "system" && (strings.Contains(m.Content, "```") || strings.Contains(m.Content, ".py\n") || strings.Contains(m.Content, ".ts\n") || strings.Contains(m.Content, ".js\n")) {
			hasFileContext = true
			break
		}
	}
	// Also check if the user message references specific files to edit
	hasEditTarget := false
	editWords := []string{"fix", "add", "update", "change", "modify", "remove", "delete", "refactor", "rename"}
	for _, w := range editWords {
		if strings.Contains(lowerTrimmed, w) {
			hasEditTarget = true
			break
		}
	}

	classifyReq := ChatRequest{
		Model: modelName,
		Messages: []ChatMessage{
			{Role: "system", Content: `Classify complexity. Reply with ONLY one digit: 0, 1, 2, or 3.`},
			{Role: "user", Content: "hi!"},
			{Role: "assistant", Content: "0"},
			{Role: "user", Content: "what does this function do?"},
			{Role: "assistant", Content: "1"},
			{Role: "user", Content: "add a docstring to main()"},
			{Role: "assistant", Content: "1"},
			{Role: "user", Content: "write tests for the User model"},
			{Role: "assistant", Content: "2"},
			{Role: "user", Content: "Create a snake game in python"},
			{Role: "assistant", Content: "3"},
			{Role: "user", Content: "refactor auth to use JWT"},
			{Role: "assistant", Content: "3"},
			{Role: "user", Content: "the app is slow, figure out why"},
			{Role: "assistant", Content: "3"},
			{Role: "user", Content: "fix the typo on line 42"},
			{Role: "assistant", Content: "1"},
			{Role: "user", Content: "add error handling to the API routes"},
			{Role: "assistant", Content: "2"},
			{Role: "user", Content: "thanks, that worked"},
			{Role: "assistant", Content: "0"},
			{Role: "user", Content: trimmed},
		},
		MaxTokens:   5,
		Temperature: 0,
		Stop:        []string{"\n"},
	}

	// Classification timeout — model needs ~1-2s on GPU
	classifyCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	resp, err := forwardToFox(classifyCtx, classifyReq)
	if err != nil {
		log.Printf("  classify failed: %v — defaulting to T1", err)
		return Tier1Simple
	}

	if len(resp.Choices) == 0 {
		return Tier1Simple
	}

	result := strings.TrimSpace(resp.Choices[0].Message.Content)
	tier := Tier1Simple // default

	// Extract first digit
	for _, c := range result {
		switch c {
		case '0':
			tier = Tier0Conversational
		case '1':
			tier = Tier1Simple
		case '2':
			tier = Tier2Medium
		case '3':
			tier = Tier3Hard
		}
		if tier != Tier1Simple || c == '1' {
			break
		}
	}

	// If no digit found, use length heuristic
	if tier == Tier1Simple && !strings.ContainsAny(result, "0123") {
		wordCount := len(strings.Fields(trimmed))
		if wordCount <= 3 {
			tier = Tier1Simple
		} else if wordCount <= 10 {
			tier = Tier2Medium
		} else {
			tier = Tier3Hard
		}
	}

	// File-context cap: if Aider sent existing file content AND the request
	// is an edit (not creation from scratch), cap at T2.
	if tier == Tier3Hard && hasFileContext && hasEditTarget {
		log.Printf("  classifier capped T3→T2 (file context + edit intent)")
		tier = Tier2Medium
	}

	// Creation guard: if the message mentions a filename + creation intent,
	// it's at least T1 (never T0). This ensures buffered generation + format
	// repair run for file creation requests.
	if tier == Tier0Conversational {
		hasFilename := filenameRe.MatchString(trimmed)
		createWords := []string{"create", "write", "make", "build", "generate", "implement"}
		hasCreate := false
		for _, w := range createWords {
			if strings.Contains(lowerTrimmed, w) {
				hasCreate = true
				break
			}
		}
		if hasFilename && hasCreate {
			log.Printf("  classifier bumped T0→T1 (creation + filename)")
			tier = Tier1Simple
		}
		// Also bump if message has code-related terms
		codeTerms := []string{"function", "class", "import", "def ", "program", "script", "game", "app", "api"}
		for _, ct := range codeTerms {
			if strings.Contains(lowerTrimmed, ct) {
				log.Printf("  classifier bumped T0→T1 (code term: %s)", ct)
				tier = Tier1Simple
				break
			}
		}
	}

	return tier
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
