package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Tier extensions — Tier, Tier0-3 constants, and String() already in main.go
// ---------------------------------------------------------------------------

// TierMaxTurns returns the maximum agent loop iterations for this tier.
//
// May 10 2026: cap removed entirely for tool-using tiers (T1/T2/T3). The
// 8 stuck-pattern detectors (parse-error / tool-repeat / reasoning-repeat
// / lens-regression / exploration-budget / path-aware error-loop /
// action gate / verification gate) are the real safety net; a numeric
// turn cap was insurance against unknown failure modes the detectors
// might miss, but it bit legitimate multi-file long-form work harder
// than it ever caught a real runaway. User-initiated cancellation
// (ctx.Ctx) still works as the upper bound. T0 keeps a small cap as
// a SHAPE constraint (conversational input shouldn't loop) — not a
// runaway-protection.
//
// MaxTurns == 0 is the agent-loop's "uncapped" sentinel. Override via
// ATLAS_MAX_TURNS env (any positive int caps; setting unset / 0 →
// uncapped).

func TierMaxTurns(t Tier) int {
	if n := envOverrideMaxTurns(); n > 0 {
		return n
	}
	switch t {
	case Tier0Conversational:
		return 5
	case Tier1Simple, Tier2Medium, Tier3Hard:
		return 0 // uncapped
	}
	return 0
}

// envOverrideMaxTurns reads ATLAS_MAX_TURNS. Returns:
//   - n > 0  → use n (no upper cap — operator's call)
//   - n == 0 / unset / invalid → 0 (caller falls through to tier defaults)
func envOverrideMaxTurns() int {
	raw := os.Getenv("ATLAS_MAX_TURNS")
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 0
	}
	return n
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

// -- ast_edit (GH #39 v1) --
//
// Friendly-selector AST edits via tree-sitter. Replaces a single named node
// (function, class, HTML element) with new content. The selector grammar is
// per-language and intentionally narrow in v1 to avoid the model
// hallucinating raw tree-sitter s-expressions (42% intended-match measured
// on Qwen3.5-9B-Q6_K, May 8 — see GH #39 open design questions).
//
//   Selectors v1:
//     python: function:NAME, class:NAME (decorator-aware: replaces
//             decorated_definition wrapper when present)
//     html:   <tag>             (top-level tag-name match)

type AstEditInput struct {
	Path     string `json:"path"`
	Selector string `json:"selector"`
	Content  string `json:"content"`
}

type AstEditOutput struct {
	OK         bool   `json:"ok"`
	Selector   string `json:"selector"`
	Language   string `json:"language,omitempty"`
	BytesOld   int    `json:"bytes_old,omitempty"`
	BytesNew   int    `json:"bytes_new,omitempty"`
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

// -- background commands (PC-196) --
//
// Three tools wrap the sandbox /jobs/* endpoints so the model can
// run a server, probe it from another command, and clean up. Used
// for the "verify HTTP routes" workflow that foreground run_command
// can't satisfy (server doesn't exit).

type RunBackgroundInput struct {
	Command string `json:"command"`
	Cwd     string `json:"cwd,omitempty"`
	// SettleMs gives the process time to print initial output before
	// we return — typical use is "wait 1500ms for the dev server's
	// startup banner so the model can confirm it bound the port."
	// Default 1500. Capped at 10000 server-side.
	SettleMs *int `json:"settle_ms,omitempty"`
}

type RunBackgroundOutput struct {
	JobID    string   `json:"job_id"`
	PID      int      `json:"pid"`
	Stdout   []string `json:"stdout"` // initial output captured during settle
	Stderr   []string `json:"stderr"`
	Running  bool     `json:"running"` // false if the process exited within settle window
	ExitCode *int     `json:"exit_code,omitempty"`
}

type TailBackgroundInput struct {
	JobID string `json:"job_id"`
	Lines *int   `json:"lines,omitempty"` // default 50, max 500
}

type TailBackgroundOutput struct {
	JobID      string   `json:"job_id"`
	Running    bool     `json:"running"`
	ExitCode   *int     `json:"exit_code,omitempty"`
	Stdout     []string `json:"stdout"`
	Stderr     []string `json:"stderr"`
	ElapsedSec float64  `json:"elapsed_sec"`
	Command    string   `json:"command"`
}

type StopBackgroundInput struct {
	JobID string `json:"job_id"`
}

type StopBackgroundOutput struct {
	JobID    string   `json:"job_id"`
	Killed   bool     `json:"killed"`
	ExitCode *int     `json:"exit_code,omitempty"`
	Stdout   []string `json:"stdout"`
	Stderr   []string `json:"stderr"`
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

// -- find_file --

type FindFileInput struct {
	Pattern string `json:"pattern"`        // regex matched against filename or relative path
	Path    string `json:"path,omitempty"` // directory to search in (defaults to working dir)
}

type FindFileMatch struct {
	Path string `json:"path"` // relative path from working dir
	Name string `json:"name"` // basename
}

type FindFileOutput struct {
	Matches    []FindFileMatch `json:"matches"`
	TotalCount int             `json:"total_count"`
	Truncated  bool            `json:"truncated,omitempty"`
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
	WorkingDir     string       // Project directory for agent operations (container path, e.g. /workspace)
	RealProjectDir string       // Same as WorkingDir; kept for delete_file compatibility
	HostWorkingDir string       // The host-side path that's bind-mounted as WorkingDir
	// (e.g. /home/isaac/snake when /workspace is mounted from there).
	// Used to translate absolute host paths the model receives back from
	// the user's prompt — e.g. "fix /home/isaac/snake/app.py" — into
	// container paths the proxy can actually open. Empty when the proxy
	// runs without a bind mount (dev / test).
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
	// PriorHistory is the prior-turn user/assistant transcript, sent by
	// the TUI on each /v1/agent request so the agent can answer follow-ups
	// like "what did you just delete?" — without it, every user message
	// is a fresh agent loop with empty context. Populated by handleAgent
	// from the request body; consumed once at the top of runAgentLoop
	// and then ignored. Tool/system rows are filtered out at the TUI
	// boundary; only role=user|assistant text turns flow through here.
	PriorHistory []AgentMessage
	FileReadTimes map[string]time.Time // for staleness detection
	FilesRead     map[string]string    // cache of read file contents
	TotalTokens  int

	// SessionWrites tracks files this agent loop wrote during this run.
	// The write_file guard rejects overwrites of "existing" files >5
	// lines (BiasBusters #3 — protects the user's code from clobbering).
	// But a file the agent itself wrote in this session is NOT the
	// user's code; it's the agent's own working output, and the agent
	// must be allowed to iterate on it. Without this, the model can't
	// realize mid-loop that an early file needs rewriting once later
	// files exist (May 12 2026 multi-file Flask run — the wiring bug
	// stemmed from the guard refusing the model's self-correction).
	SessionWrites map[string]bool

	// PC-207 agent-loop integration: rolling list of gx_score_min values
	// from lens scoring of write_file/edit_file tool calls. When the
	// recent N values all fall below lensLowScoreThreshold the loop
	// injects a corrective system message before the next LLM call.
	// See proxy/lens_score.go for the pattern detection.
	LensScoreHistory []float64

	// Tool-call repetition detector: rolling window of recent (tool,
	// args) signatures. When the same signature appears
	// toolRepeatThreshold times within the last toolRepeatWindow
	// entries, the loop injects a corrective system message. See
	// proxy/tool_repeat.go for the detection logic.
	RecentToolCalls []string

	// Reasoning-repetition detector state (May 10 2026, BiasBusters
	// follow-up #30). Per-turn snapshot of the model's reasoning_content
	// stream. When the same opening prose ("Now I need to look at the
	// file" / similar) appears across consecutive turns, the loop
	// injects a corrective so the model breaks out of the thought loop.
	// See proxy/reasoning_repeat.go for the detection logic.
	LastTurnReasoning           string
	LastReasoningSnippet        string
	ConsecutiveReasoningRepeats int

	// Path-aware error-loop detector (May 10 2026). Tracks recent
	// failure paths so the 3-consecutive-failures breaker can
	// distinguish "stuck on one file" (real loop, stop) from
	// "grinding through different files, some succeed, some fail"
	// (progress, keep going). Cleared on any successful tool result.
	// See proxy/agent.go error-loop break for the use site.
	RecentFailurePaths []string

	mu           sync.Mutex

	// Plan is the optional pre-flight plan produced by /v3/plan. Set
	// once at the top of the agent loop for non-trivial requests; nil
	// when we skipped planning (T0, simple greetings, dev mode without
	// V3). Read by the plan-adherence gate to compare actual tool calls
	// against the planned step actions.
	Plan *Plan

	// PlanStepsSatisfied[i] flips true once a tool call has matched
	// plan step i. Length tracks len(Plan.Steps); nil when no plan.
	// Reset whenever the plan is revised so we re-track from scratch.
	PlanStepsSatisfied []bool

	// PlanOffStreak counts consecutive tool calls that DIDN'T match
	// any unsatisfied plan step. Crosses the auto-revise threshold ->
	// planner re-runs with whatever context we've discovered so far.
	PlanOffStreak int

	// PlanRevisions counts how many times we've auto-regenerated the
	// plan in this loop. Capped to keep us from thrashing — after the
	// cap is hit we stop revising and let the agent run plan-free.
	PlanRevisions int

	// VerifyOnHost flips run_command from sandbox-routing to local
	// host execution (PC-192). Set from ATLAS_VERIFY_IN=host or
	// per-project .atlas/config.toml. The default (false) is the
	// safer sandbox path; opt-in is for working codebases that
	// depend on host services (DBs, env vars, system tools) the
	// sandbox can't see. Shell-op guardrails still apply either way.
	VerifyOnHost bool

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
		SessionWrites:  make(map[string]bool),
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
	// RPGSignatureMissing lists planned RPG node signatures the winning code
	// failed to realize (V3.2, issue #120). Non-empty signals structural drift
	// from the plan — the proxy's re-plan loop surfaces the affected subgraph.
	RPGSignatureMissing []string `json:"rpg_signature_missing,omitempty"`
}

// LensScore is already defined in main.go — reused here.

// V3PlanRequest is sent to the Python V3 service for plan generation.
// project_context inlines small file contents (truncated server-side) so
// the planner sees what's actually in the working directory.
type V3PlanRequest struct {
	UserMessage    string            `json:"user_message"`
	WorkingDir     string            `json:"working_dir,omitempty"`
	ProjectContext map[string]string `json:"project_context,omitempty"`
	NCandidates    int               `json:"n_candidates,omitempty"` // 0 → server default (3)
}

// PlanStep is a single step in a Plan. Mirrors v3-service/main.py's
// PLAN_PROMPT_TEMPLATE shape: id, action, target, why.
//
// NodeID and Constraints are populated only by the V3.2 RPG planner
// (issue #120): when a plan step is the projection of a Repository Planning
// Graph node, Constraints carries that node's planned interface (signatures,
// input/output edges) which the proxy threads into the per-node /v3/generate
// call. They are empty for the flat planner, so behavior is unchanged when
// ATLAS_RPG_PLANNING is off.
type PlanStep struct {
	ID          string   `json:"id"`
	Action      string   `json:"action"`
	Target      string   `json:"target"`
	Why         string   `json:"why"`
	NodeID      string   `json:"node_id,omitempty"`
	Constraints []string `json:"constraints,omitempty"`
}

// Plan is the structured plan returned by /v3/plan. The agent loop
// consults this to gate tool calls (PC plan-adherence) and replays
// VerifyStep at the verification gate.
type Plan struct {
	Steps            []PlanStep `json:"steps"`
	VerifyStep       string     `json:"verify_step"`
	Rationale        string     `json:"rationale"`
	CandidatesTested int        `json:"candidates_tested"`
	WinningScore     float64    `json:"winning_score"`
	WinningIndex     int        `json:"winning_index"`
	Reasons          []string   `json:"reasons"`
	// RPG is the full Repository Planning Graph the flat Steps were projected
	// from (V3.2, issue #120). Present only when the RPG planner ran; nil for
	// the flat planner. Captured proxy-side for graph-guided traversal /
	// observability.
	RPG *RPG `json:"rpg,omitempty"`
}

// RPG is the Repository Planning Graph artifact (issue #120, arXiv:2509.16198):
// capabilities (what to build) expanded into files + function signatures + the
// data-flow / ordering edges between them (how). Mirrors v3-service/rpg.py.
type RPG struct {
	Capabilities []RPGCapability `json:"capabilities"`
	Files        []RPGFile       `json:"files"`
	Edges        []RPGEdge       `json:"edges"`
	Verify       string          `json:"verify"`
	Rationale    string          `json:"rationale"`
}

type RPGCapability struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Parent string `json:"parent"`
}

type RPGFunction struct {
	Name      string `json:"name"`
	Signature string `json:"signature"`
	Summary   string `json:"summary"`
}

type RPGFile struct {
	ID         string        `json:"id"`
	Path       string        `json:"path"`
	Capability string        `json:"capability"`
	Functions  []RPGFunction `json:"functions"`
}

type RPGEdge struct {
	From  string `json:"from"`
	To    string `json:"to"`
	Kind  string `json:"kind"`
	Label string `json:"label"`
}

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
