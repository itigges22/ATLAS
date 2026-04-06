package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Tier extensions — Tier, Tier0-3 constants, and String() already in main.go
// ---------------------------------------------------------------------------

// TierMaxTurns returns the maximum agent loop iterations for this tier.
func TierMaxTurns(t Tier) int {
	switch t {
	case Tier0Conversational:
		return 5
	case Tier1Simple:
		return 30
	case Tier2Medium:
		return 30
	case Tier3Hard:
		return 60
	}
	return 30
}

// TierUsesV3 returns whether write_file/edit_file should route through V3.
func TierUsesV3(t Tier) bool {
	return t >= Tier2Medium
}

// ---------------------------------------------------------------------------
// Agent messages — the conversation between model and tool executor
// ---------------------------------------------------------------------------

// ModelResponse is what the LLM emits (constrained by grammar/json_schema).
// Exactly one of the three variants is populated per response.
type ModelResponse struct {
	Type    string          `json:"type"`    // "tool_call", "text", or "done"
	Name    string          `json:"name"`    // tool name (only for tool_call)
	Args    json.RawMessage `json:"args"`    // tool arguments (only for tool_call)
	Content string          `json:"content"` // text content (only for text)
	Summary string          `json:"summary"` // completion summary (only for done)
}

// AgentMessage represents a message in the agent loop conversation.
type AgentMessage struct {
	Role       string `json:"role"` // "system", "user", "assistant", "tool"
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"` // for tool results
	ToolName   string `json:"tool_name,omitempty"`    // for tool results
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

// ToolDef defines a tool that the model can call.
type ToolDef struct {
	Name        string
	Description string
	InputSchema interface{} // Go struct with json tags, marshaled to JSON Schema
	Execute     func(input json.RawMessage, ctx *AgentContext) (*ToolResult, error)
	ReadOnly    bool // true = can run in parallel, no side effects
	Destructive bool // true = requires permission confirmation
}

// ToolResult is the structured output returned to the model after tool execution.
type ToolResult struct {
	Success bool            `json:"success"`
	Data    json.RawMessage `json:"data,omitempty"`
	Error   string          `json:"error,omitempty"`

	// V3 metadata (populated when V3 pipeline was used)
	V3Used           bool    `json:"v3_used,omitempty"`
	CandidatesTested int     `json:"candidates_tested,omitempty"`
	WinningScore     float64 `json:"winning_score,omitempty"`
	PhaseSolved      string  `json:"phase_solved,omitempty"`
}

// MarshalText returns a compact string representation for the model.
func (r *ToolResult) MarshalText() string {
	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Sprintf(`{"success":false,"error":"marshal error: %s"}`, err)
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// Tool input/output types
// ---------------------------------------------------------------------------

// -- read_file --

type ReadFileInput struct {
	Path   string `json:"path"`
	Offset *int   `json:"offset,omitempty"` // line offset (0-based)
	Limit  *int   `json:"limit,omitempty"`  // max lines to read
}

type ReadFileOutput struct {
	Content    string `json:"content"`
	TotalLines int    `json:"total_lines"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
}

// -- write_file --

type WriteFileInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type WriteFileOutput struct {
	BytesWritten     int     `json:"bytes_written"`
	V3Used           bool    `json:"v3_used,omitempty"`
	CandidatesTested int     `json:"candidates_tested,omitempty"`
	WinningScore     float64 `json:"winning_score,omitempty"`
	PhaseSolved      string  `json:"phase_solved,omitempty"`
}

// -- edit_file --

type EditFileInput struct {
	Path       string `json:"path"`
	OldStr     string `json:"old_str"`
	NewStr     string `json:"new_str"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}

type EditFileOutput struct {
	OK          bool   `json:"ok"`
	DiffPreview string `json:"diff_preview,omitempty"`
	LinesAdded  int    `json:"lines_added,omitempty"`
	LinesRemoved int   `json:"lines_removed,omitempty"`
}

// -- delete_file --

type DeleteFileInput struct {
	Path string `json:"path"`
}

type DeleteFileOutput struct {
	Deleted bool `json:"deleted"`
}

// -- run_command --

type RunCommandInput struct {
	Command string `json:"command"`
	Timeout *int   `json:"timeout,omitempty"` // seconds, default 30
	Cwd     string `json:"cwd,omitempty"`
}

type RunCommandOutput struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
}

// -- search_files --

type SearchFilesInput struct {
	Pattern string `json:"pattern"`           // regex pattern
	Path    string `json:"path,omitempty"`    // directory to search in
	Glob    string `json:"glob,omitempty"`    // file glob filter (e.g., "*.go")
}

type SearchMatch struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Content string `json:"content"`
}

type SearchFilesOutput struct {
	Matches    []SearchMatch `json:"matches"`
	TotalCount int           `json:"total_count"`
	Truncated  bool          `json:"truncated,omitempty"`
}

// -- list_directory --

type ListDirectoryInput struct {
	Path string `json:"path"`
}

type DirEntry struct {
	Name  string `json:"name"`
	Type  string `json:"type"` // "file", "dir", "symlink"
	Size  int64  `json:"size,omitempty"`
}

type ListDirectoryOutput struct {
	Entries []DirEntry `json:"entries"`
	Path    string     `json:"path"`
}

// -- plan_tasks --

type PlanTasksInput struct {
	Tasks []PlannedTask `json:"tasks"`
}

type PlannedTask struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Files       []string `json:"files,omitempty"`
	DependsOn   []string `json:"depends_on,omitempty"`
}

type TaskStatus struct {
	ID     string `json:"id"`
	Status string `json:"status"` // "completed", "failed", "skipped"
	Error  string `json:"error,omitempty"`
}

type PlanTasksOutput struct {
	Results []TaskStatus `json:"results"`
}

// ---------------------------------------------------------------------------
// Agent context — shared state for the agent loop
// ---------------------------------------------------------------------------

// AgentContext holds all state for a single agent loop execution.
type AgentContext struct {
	// Configuration
	Tier           Tier
	MaxTurns       int
	WorkingDir     string       // Temp dir for agent operations
	RealProjectDir string       // Actual project directory on disk (for delete_file)
	PermissionMode PermissionMode
	YoloMode       bool

	// Service URLs
	InferenceURL     string
	SandboxURL string
	LensURL     string
	V3URL      string

	// Project info (populated by project detection)
	Project *ProjectInfo

	// State
	Messages     []AgentMessage
	FileReadTimes map[string]time.Time // for staleness detection
	FilesRead     map[string]string    // cache of read file contents
	TotalTokens  int
	mu           sync.Mutex

	// Streaming callback
	StreamFn func(eventType string, data interface{})

	// Permission callback
	PermissionFn func(toolName string, args json.RawMessage) bool

	// Context for cancellation
	Ctx context.Context
}

// NewAgentContext creates a new agent context with defaults.
func NewAgentContext(workingDir string, tier Tier) *AgentContext {
	return &AgentContext{
		Tier:           tier,
		MaxTurns:       TierMaxTurns(tier),
		WorkingDir:     workingDir,
		PermissionMode: PermissionDefault,
		FileReadTimes:  make(map[string]time.Time),
		FilesRead:      make(map[string]string),
		Ctx:            context.Background(),
	}
}

// Stream sends an SSE event to the client.
func (c *AgentContext) Stream(eventType string, data interface{}) {
	if c.StreamFn != nil {
		c.StreamFn(eventType, data)
	}
}

// RecordFileRead tracks when a file was last read (for staleness detection).
func (c *AgentContext) RecordFileRead(path string, content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.FileReadTimes[path] = time.Now()
	c.FilesRead[path] = content
}

// WasFileRead returns true if the file was read during this agent session.
func (c *AgentContext) WasFileRead(path string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	_, ok := c.FileReadTimes[path]
	return ok
}

// ---------------------------------------------------------------------------
// Permission system types
// ---------------------------------------------------------------------------

type PermissionMode int

const (
	PermissionDefault     PermissionMode = iota // Ask for write/edit/run
	PermissionAcceptEdits                       // Auto-approve write/edit, ask for run
	PermissionYolo                              // Auto-approve everything
)

func (m PermissionMode) String() string {
	switch m {
	case PermissionDefault:
		return "default"
	case PermissionAcceptEdits:
		return "accept-edits"
	case PermissionYolo:
		return "yolo"
	}
	return "default"
}

// PermissionRule is a pattern-based allow/deny rule.
type PermissionRule struct {
	Tool    string `json:"tool"`    // e.g., "run_command"
	Pattern string `json:"pattern"` // e.g., "npm *"
	Action  string `json:"action"`  // "allow" or "deny"
}

// ---------------------------------------------------------------------------
// Project detection types
// ---------------------------------------------------------------------------

type ProjectInfo struct {
	Language     string   `json:"language"`      // "nodejs", "python", "rust", "go", "c", "shell"
	Framework    string   `json:"framework"`     // "nextjs", "flask", "actix", etc.
	ConfigFiles  []string `json:"config_files"`  // detected config file paths
	BuildCommand string   `json:"build_command"` // e.g., "npm run build"
	DevCommand   string   `json:"dev_command"`   // e.g., "npm run dev"
	TestCommand  string   `json:"test_command"`  // e.g., "npm test"
}

// ---------------------------------------------------------------------------
// V3 pipeline types
// ---------------------------------------------------------------------------

// V3GenerateRequest is sent to the Python V3 service for arbitrary file generation.
type V3GenerateRequest struct {
	FilePath       string            `json:"file_path"`
	BaselineCode   string            `json:"baseline_code"`
	ProjectContext map[string]string `json:"project_context,omitempty"`
	Framework      string            `json:"framework,omitempty"`
	BuildCommand   string            `json:"build_command,omitempty"`
	Constraints    []string          `json:"constraints,omitempty"`
	Tier           int               `json:"tier"`
	WorkingDir     string            `json:"working_dir,omitempty"`
}

// V3GenerateResponse is the response from the V3 service.
type V3GenerateResponse struct {
	Code             string  `json:"code"`
	Passed           bool    `json:"passed"`
	PhaseSolved      string  `json:"phase_solved"`
	CandidatesTested int     `json:"candidates_tested"`
	WinningScore     float64 `json:"winning_score"`
	TotalTokens      int     `json:"total_tokens"`
	TotalTimeMs      float64 `json:"total_time_ms"`
}

// LensScore is already defined in main.go — reused here.

// ---------------------------------------------------------------------------
// SSE event types for the CLI protocol
// ---------------------------------------------------------------------------

type SSEEvent struct {
	Type string      `json:"type"` // "tool_call", "tool_result", "text", "done", "permission_request", "error"
	Data interface{} `json:"data"`
}

type PermissionRequest struct {
	ToolName string          `json:"tool_name"`
	Args     json.RawMessage `json:"args"`
	Message  string          `json:"message"` // human-readable description
}
