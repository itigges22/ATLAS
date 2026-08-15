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
		// 12, not the 5 this shipped with. A question about the code is
		// classified conversational and still has to LOOK at the code, and 5
		// turns does not survive that: observed on a fresh workspace, "how
		// does the contact form work?" spent turn 0 on a bounced text exit,
		// turns 1-3 on searches with the wrong glob, reached the right file
		// on turn 4, and hit the cap — the user got no answer at all. The cap
		// is here to stop a conversational request looping, and 12 still does
		// that while leaving room to read a few files and reply.
		return 12
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

// ToolEffect declares what a tool is CAPABLE of doing to workspace state. It
// is a property of the tool, not of any one call, and it lives on ToolDef so a
// new tool cannot be registered without the decision being made. A parallel
// map keyed by tool name would drift; this cannot.
//
// Capability is not outcome. A direct mutator still reports applied, refused,
// failed or none per branch -- the effect only says the tool is capable of
// mutating and therefore owes a local classification that the shared boundary
// must not invent for it.
type ToolEffect string

const (
	// ToolEffectUnknown is the zero value: a registration that never declared
	// its effect. It is a migration defect, never a runtime state.
	ToolEffectUnknown ToolEffect = ""

	// ToolEffectReadOnly cannot mutate workspace state by construction.
	ToolEffectReadOnly ToolEffect = "read_only"

	// ToolEffectDirectMutation performs and observes its own mutation, so only
	// the branch that ran it knows the outcome.
	ToolEffectDirectMutation ToolEffect = "direct_mutation"

	// ToolEffectCommandUnobserved may change the workspace without measuring
	// it: arbitrary shell, background jobs, and killing a running process
	// (SIGTERM/SIGKILL mid-write can leave partial bytes nobody compared).
	ToolEffectCommandUnobserved ToolEffect = "command_unobserved"
)

// BoundaryClassifiable reports whether executeToolCall may supply the
// classification itself. It may do so only where the ANSWER FOLLOWS FROM THE
// CLASS: read-only tools mutate nothing, command tools measure nothing.
// Direct mutators are excluded on purpose -- their outcome is branch-local,
// and letting the boundary fill it in would conceal a missed producer.
func (e ToolEffect) BoundaryClassifiable() bool {
	return e == ToolEffectReadOnly || e == ToolEffectCommandUnobserved
}

// ToolDef defines a tool that the model can call.
type ToolDef struct {
	// Effect is required. The registry exhaustiveness test fails on any
	// production tool left at ToolEffectUnknown.
	Effect ToolEffect

	Name        string
	Description string
	InputSchema interface{} // Go struct with json tags, marshaled to JSON Schema
	Execute     func(input json.RawMessage, ctx *AgentContext) (*ToolResult, error)
	ReadOnly    bool // true = can run in parallel, no side effects
	Destructive bool // true = requires permission confirmation
}

// MutationStatus reports what happened to the filesystem, separately from
// whether the tool ran. `Success` alone could not express the difference
// between a write that landed and one a gate refused.
//
// The zero value is MutationUnknown, NOT MutationNone. An unmigrated
// producer and one that intentionally mutated nothing must stay
// distinguishable, or the producer audit cannot prove its own completeness:
// a missed call site would look exactly like a deliberate no-op.
type MutationStatus string

const (
	MutationUnknown MutationStatus = ""     // unclassified: nobody set it (zero value)
	MutationNone    MutationStatus = "none" // deliberately mutated nothing
	MutationApplied MutationStatus = "applied"
	MutationRefused MutationStatus = "refused" // a gate declined it
	MutationFailed  MutationStatus = "failed"  // attempted, errored

	// MutationUnobserved: this tool can change the workspace and did not
	// look. Command tools run arbitrary shell -- `sed -i`, a build step,
	// `python fix.py` -- and perform no pre/post state comparison, so
	// reporting MutationNone would assert a fact the producer does not have.
	// Unobserved is a current producer explicitly saying side effects were
	// not measured. It is Classified (the producer spoke) but never Applied
	// (nothing was demonstrated).
	MutationUnobserved MutationStatus = "unobserved"
)

// Applied is true for exactly MutationApplied. Comparing against a literal
// at a call site is what lets an unknown, case-shifted or future value leak
// through as success.
func (m MutationStatus) Applied() bool { return m == MutationApplied }

// Classified reports whether a producer actually set this field.
func (m MutationStatus) Classified() bool {
	switch m {
	case MutationNone, MutationApplied, MutationRefused, MutationFailed,
		MutationUnobserved:
		return true
	}
	return false
}

// ValidationKind names WHAT was checked. A syntax pass is not evidence the
// task was verified; keeping the kind explicit stops a cheap check from
// being read later as an expensive one.
type ValidationKind string

const (
	ValidationKindUnknown ValidationKind = ""     // unclassified (zero value)
	ValidationKindNone    ValidationKind = "none" // deliberately checked nothing
	ValidationKindSyntax  ValidationKind = "syntax"

	// ValidationKindStructural: the content parses but references something
	// that does not resolve. In the write_file handler the syntax gate runs
	// BEFORE the structural one, so a structural rejection means syntax
	// PASSED on those exact bytes -- recording it as syntax/failed would
	// assert the opposite of what happened.
	//
	// Where two checks ran with different outcomes, these singular fields
	// carry the DECISIVE one: the failure that caused the refusal, or the
	// applicable result for an applied write.
	ValidationKindStructural ValidationKind = "structural"
)

func (k ValidationKind) Classified() bool {
	switch k {
	case ValidationKindNone, ValidationKindSyntax, ValidationKindStructural:
		return true
	}
	return false
}

// ValidationStatus reports the OUTCOME of that check. `not_run` and
// `not_applicable` are distinct facts -- "could not check" versus "nothing
// here to check" -- and both are distinct from Unknown, which means no
// producer spoke at all.
type ValidationStatus string

const (
	ValidationUnknown       ValidationStatus = ""        // unclassified (zero value)
	ValidationNotRun        ValidationStatus = "not_run" // deliberately not checked
	ValidationNotApplicable ValidationStatus = "not_applicable"
	ValidationPassed        ValidationStatus = "passed"
	ValidationFailed        ValidationStatus = "failed"
)

// Passed is true for exactly ValidationPassed, for the same reason as Applied.
func (v ValidationStatus) Passed() bool { return v == ValidationPassed }

func (v ValidationStatus) Classified() bool {
	switch v {
	case ValidationNotRun, ValidationNotApplicable, ValidationPassed, ValidationFailed:
		return true
	}
	return false
}

// Classified reports whether every fact on this result was set by a
// producer. The producer audit asserts this over each mutation-producing
// path; until it holds everywhere, no consumer may read the new fields.
//
// A syntax kind paired with an unknown status is NOT classified: claiming a
// check happened without saying how it came out is exactly the ambiguity
// these fields exist to remove.
func (r *ToolResult) Classified() bool {
	return r.MutationStatus.Classified() &&
		r.ValidationKind.Classified() &&
		r.ValidationStatus.Classified()
}

// ToolResult is the structured output returned to the model after tool execution.
type ToolResult struct {
	Success bool            `json:"success"`
	Data    json.RawMessage `json:"data,omitempty"`
	Error   string          `json:"error,omitempty"`

	// Facts about the mutation and the check, reported independently of
	// `Success` (which keeps its existing tool-operation meaning on the
	// wire). All three are omitempty, so an older consumer sees the payload
	// it always saw, and a payload from an older producer decodes to the
	// fail-closed zero values above.
	//
	// Nothing here says whether the work counts as progress or authorises
	// completion. That is task-contextual and is derived in the agent layer.
	MutationStatus   MutationStatus   `json:"mutation_status,omitempty"`
	ValidationKind   ValidationKind   `json:"validation_kind,omitempty"`
	ValidationStatus ValidationStatus `json:"validation_status,omitempty"`
	ValidationDetail string           `json:"validation_detail,omitempty"`

	// V3 metadata (populated when V3 pipeline was used)
	V3Used               bool                     `json:"v3_used,omitempty"`
	CandidatesTested     int                      `json:"candidates_tested,omitempty"`
	WinningScore         float64                  `json:"winning_score,omitempty"`
	PhaseSolved          string                   `json:"phase_solved,omitempty"`
	VerificationEvidence []V3VerificationEvidence `json:"verification_evidence,omitempty"`
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

// -- outline_file --

type OutlineInput struct {
	Path string `json:"path"`
}

type OutlineSymbol struct {
	Name      string `json:"name"`
	Kind      string `json:"kind"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	// Intra-file call-graph neighborhood (issue #39, populated only when
	// ATLAS_CALL_GRAPH is on in v3-service). Calls = functions this symbol
	// invokes; CalledBy = functions that invoke it. Lets the model follow a
	// symptom to its callee-rooted cause instead of editing where the
	// symptom surfaces.
	Calls    []string `json:"calls,omitempty"`
	CalledBy []string `json:"called_by,omitempty"`
}

type OutlineOutput struct {
	Symbols   []OutlineSymbol `json:"symbols"`
	Supported bool            `json:"supported"`
	// Outline is the rendered human-readable listing (header, L<start>-<end>
	// lines, call edges). The model reads this; Symbols carries the same
	// data structurally.
	Outline string `json:"outline,omitempty"`
	// Regions holding code in ANOTHER language — <script>/<style> blocks,
	// including inside Python string literals. The host grammar cannot see
	// into a string, so these are exactly the symbols a selector CANNOT
	// reach, and naming them is what stops the model reaching for them.
	EmbeddedRegions []EmbeddedRegion `json:"embedded_regions"`
}

// EmbeddedRegion is one foreign-language block inside a file.
type EmbeddedRegion struct {
	Where     string   `json:"where"`
	Kind      string   `json:"kind"`
	StartLine int      `json:"start_line"`
	EndLine   int      `json:"end_line"`
	Symbols   []string `json:"symbols"`
}

// -- write_file --

type WriteFileInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type WriteFileOutput struct {
	BytesWritten int `json:"bytes_written"`
	// Warning carries a non-blocking defect notice: the write LANDED, and
	// the model should act on this next (e.g. "does not parse — run it and
	// read the traceback"). See writeNewFileWithWarning.
	Warning              string                   `json:"warning,omitempty"`
	V3Used               bool                     `json:"v3_used,omitempty"`
	CandidatesTested     int                      `json:"candidates_tested,omitempty"`
	WinningScore         float64                  `json:"winning_score,omitempty"`
	PhaseSolved          string                   `json:"phase_solved,omitempty"`
	VerificationEvidence []V3VerificationEvidence `json:"verification_evidence,omitempty"`
}

// -- edit_file --

type EditFileInput struct {
	Path       string `json:"path"`
	OldStr     string `json:"old_str"`
	NewStr     string `json:"new_str"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}

// InsertAfterInput places new lines after a line the model NAMES rather than
// reproduces. read_file already returns "N<tab>content", so the model cites a
// number it can see instead of transcribing an anchor byte-for-byte — the step
// that measurably fails on long spans.
type InsertAfterInput struct {
	Path string `json:"path"`
	// 0 inserts at the top of the file; N inserts after line N (1-based).
	Line    int    `json:"line"`
	Content string `json:"content"`
}

// ReplaceLinesInput is insert_after's rationale extended to REPLACEMENT, which
// is where the verbatim burden actually bites. edit_file needs the old text
// reproduced byte-for-byte; on a 9-13 line span this model corrupts one token
// somewhere in it essentially every time. Here the address is a pair of numbers
// read_file already printed.
//
// ExpectedFirstLine/ExpectedLastLine are the safety belt, and they are not
// optional. A line range degrades catastrophically and SILENTLY: an off-by-one
// still applies cleanly and produces plausible corruption, where a bad anchor
// merely fails to match. Every surveyed implementation that ships a line-range
// replace carries some form of this assertion. One line each is the length
// regime this model is reliable in.
type ReplaceLinesInput struct {
	Path string `json:"path"`
	// 1-based, inclusive on both ends — the convention read_file prints.
	StartLine int `json:"start_line"`
	EndLine   int `json:"end_line"`
	// Verbatim text of the first and last line being replaced, without the
	// "N<tab>" display prefix. Whitespace-trimmed before comparison.
	ExpectedFirstLine string `json:"expected_first_line"`
	ExpectedLastLine  string `json:"expected_last_line"`
	Content           string `json:"content"`
}

type EditFileOutput struct {
	OK           bool   `json:"ok"`
	DiffPreview  string `json:"diff_preview,omitempty"`
	LinesAdded   int    `json:"lines_added,omitempty"`
	LinesRemoved int    `json:"lines_removed,omitempty"`
}

// -- structural_edit (GH #39 v1) --
//
// Friendly-selector structural edits via tree-sitter. Replaces a single named node
// (function, class, HTML element) with new content. The selector grammar is
// per-language and intentionally narrow in v1 to avoid the model
// hallucinating raw tree-sitter s-expressions (42% intended-match measured
// on the then-reference local model, May 8 — see GH #39 open design questions).
//
//   Selectors v1:
//     python: function:NAME, class:NAME (decorator-aware: replaces
//             decorated_definition wrapper when present)
//     html:   <tag>             (top-level tag-name match)

type StructuralEditInput struct {
	Path     string `json:"path"`
	Selector string `json:"selector"`
	Content  string `json:"content"`
}

type StructuralEditOutput struct {
	OK       bool   `json:"ok"`
	Selector string `json:"selector"`
	Language string `json:"language,omitempty"`
	BytesOld int    `json:"bytes_old,omitempty"`
	BytesNew int    `json:"bytes_new,omitempty"`
}

// -- delete_file --

type DeleteFileInput struct {
	Path string `json:"path"`
}

type DeleteFileOutput struct {
	Deleted bool `json:"deleted"`
}

// -- move_file --
//
// Relocate or rename a file within the workspace. A pure move/rename is not a
// content change, so it does NOT go through the V3 / surgical-edit gate the way
// write_file/edit_file do — it exists so "reorganize the files" (e.g. move
// index.html into templates/) is a single tool call instead of a
// read→write→delete dance the model can't reliably compose. Shell `mv`/`cp`
// stay refused; this is the supported relocation path.

type MoveFileInput struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
}

type MoveFileOutput struct {
	Moved       bool   `json:"moved"`
	Source      string `json:"source"`
	Destination string `json:"destination"`
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

// -- background commands --
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
	Pattern string `json:"pattern"`        // regex pattern
	Path    string `json:"path,omitempty"` // directory to search in
	Glob    string `json:"glob,omitempty"` // file glob filter (e.g., "*.go")
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
	Name string `json:"name"`
	Type string `json:"type"` // "file", "dir", "symlink"
	Size int64  `json:"size,omitempty"`
}

type ListDirectoryOutput struct {
	Entries []DirEntry `json:"entries"`
	Path    string     `json:"path"`
}

// ---------------------------------------------------------------------------
// Agent context — shared state for the agent loop
// ---------------------------------------------------------------------------

// AgentContext holds all state for a single agent loop execution.
type AgentContext struct {
	// Configuration
	Tier           Tier
	MaxTurns       int
	WorkingDir     string // Project directory for agent operations (container path, e.g. /workspace)
	HostWorkingDir string // The host-side path that's bind-mounted as WorkingDir
	// (e.g. /home/isaac/snake when /workspace is mounted from there).
	// Used to translate absolute host paths the model receives back from
	// the user's prompt — e.g. "fix /home/isaac/snake/app.py" — into
	// container paths the proxy can actually open. Empty when the proxy
	// runs without a bind mount (dev / test).
	PermissionMode PermissionMode
	YoloMode       bool

	// AllowedTools names tools the client has pre-approved for this session
	// (seeded from the request's session_allowed_tools) plus any the user
	// approves with session scope during this turn. A named tool skips the
	// interactive permission prompt. Guarded by mu.
	AllowedTools map[string]bool

	// Service URLs
	InferenceURL string
	SandboxURL   string
	LensURL      string
	V3URL        string

	// BypassV3 short-circuits the V3 layer for this request: pre-flight
	// plan generation and the write/edit candidate pipelines. The base
	// file/tool agent and its guardrails still run so /demo can compare
	// executable outputs from the same model without falsely presenting
	// the left side as a bare chat completion.
	BypassV3 bool

	// DisableFreshSlot skips the slot-erase at the start of the
	// agent loop so the demo's pre-warm pass actually survives into the
	// real run. Without this, the prefix cache the warmup builds is
	// wiped on the next /v1/agent call and the demo pays the 25-second
	// cold-start cost twice. Only set this from controlled flows
	// (`/demo`, tests) — production sessions want the per-session KV
	// isolation.
	DisableFreshSlot bool

	// Project info (populated by project detection)
	Project *ProjectInfo

	// State
	Messages []AgentMessage
	// PriorHistory is the prior-turn user/assistant transcript, sent by
	// the TUI on each /v1/agent request so the agent can answer follow-ups
	// like "what did you just delete?" — without it, every user message
	// is a fresh agent loop with empty context. Populated by handleAgent
	// from the request body; consumed once at the top of runAgentLoop
	// and then ignored. Tool/system rows are filtered out at the TUI
	// boundary; only role=user|assistant text turns flow through here.
	PriorHistory  []AgentMessage
	FileReadTimes map[string]time.Time // for staleness detection
	FilesRead     map[string]string    // cache of read file contents
	TotalTokens   int

	// VerificationEvidence holds one record per GREEN verification command,
	// binding the result to what the command actually exercised: the
	// command line, its stdin contract, and the sha256 of each covered
	// session-written file at the moment it passed. Lens labels are drawn
	// from these records, never from a session-wide boolean — a flag can't
	// answer "verified WHAT, at WHICH bytes", and labeling on it trained
	// the lens on files the passing command never touched (third-party
	// audit finding).
	VerificationEvidence []VerificationRecord

	// HumanTask is the CURRENT request's actual human instruction, captured
	// once at the top of runAgentLoop before the loop appends anything.
	// ATLAS represents internal correctives, manifests and re-injected file
	// content as user-role messages for chat-template compatibility, so
	// "last user message" stops meaning "what the human asked" the moment
	// the first [system note] lands — V3 was observed generating against
	// "run the program standalone" instead of the task (third-party audit
	// finding). Role conversion is a serialization concern; this field
	// preserves the provenance the role field erases.
	HumanTask string

	// LiteralBlocks are the byte-exact content contracts extracted from
	// HumanTask (fenced blocks, "exactly ...:" lines). Landed writes are
	// verified against these and whitespace-only drift is repaired
	// mechanically — the model plans, the harness copies. See
	// extractLiteralBlocks / repairLiteralDrift in guardrails.go.
	LiteralBlocks []string

	// FencedCalls / FencedTokens account the @fenced sub-calls
	// (fetchFencedContent): every attempt is a full model generation, and
	// before these existed the run totals silently omitted that spend — a
	// session's reported token count could be off by one whole generation
	// per written file (third-party audit finding). FencedTokens is also
	// folded into TotalTokens; these two exist so the fenced share is
	// visible on its own.
	FencedCalls  int
	FencedTokens int

	// BodySeen records the files whose CONTENTS were put in front of the
	// model. FilesRead answers a different question: outline_file caches a
	// file's full source for staleness tracking while showing the model only
	// signatures and line ranges, so a path sits in FilesRead having never
	// been displayed. read_file already draws this distinction the same way,
	// recording only the head of a truncated read because the head is all the
	// model saw.
	//
	// Measured on a diagnostic question over three files: the model outlined
	// scoring.py, received symbol names with line ranges, and answered "line
	// 142 tries to access keys like..." about lines it had never been shown.
	// Across 12 sessions it never once read a body before diagnosing, and the
	// answer was decided entirely by which filename it guessed: scoring.py
	// wrong 11/11, planning.py right 1/1.
	BodySeen map[string]bool

	// SessionWrites tracks files this agent loop wrote during this run.
	// The write_file guard rejects overwrites of "existing" files >5
	// lines (BiasBusters #3 — protects the user's code from clobbering).
	// But a file the agent itself wrote in this session is NOT the
	// user's code; it's the agent's own working output, and the agent
	// must be allowed to iterate on it. Without this, the model can
	// realize mid-loop that app.py was a stub and needs rewriting
	// (May 12 2026 /demo multi-file flask run — the entire wiring bug
	// stemmed from the guard refusing the model's self-correction).
	SessionWrites map[string]bool

	// FencedFailures counts CONSECUTIVE zero-byte fenced resolutions per
	// target path, for the life of the session. It lives here and not in
	// fetchFencedContent because that function's attempt counter is a local:
	// a new write_file call re-entered it with a fresh budget, so a path that
	// had already burned two ~300s attempts could burn two more on the next
	// turn. Keyed by path so one file's failures cannot exhaust another's
	// allowance. A successful resolution clears the entry.
	FencedFailures map[string]int

	// ManifestAnnounced tracks which SessionWrites paths have been named
	// in a session-file-manifest note (context.go) so each file
	// is announced to the model once.
	ManifestAnnounced map[string]bool

	// BackgroundJobs maps job_id -> command for jobs this loop started and
	// has not stopped. A background server the model started to verify its
	// own work keeps holding the port, so the model's next `python app.py`
	// fails with "address already in use" — naming a program it cannot see.
	// An observed session spent its remaining turns on that conflict.
	// Tracked so the run_command failure can say which of the model's own
	// jobs is holding the port.
	BackgroundJobs map[string]string

	// AssetLintSeen dedupes asset-graph lint findings (gates.go) so
	// a persistent orphan is mentioned once, not after every write.
	AssetLintSeen map[string]bool

	// PassWrites records the model-authored content of each write this pass
	// (write_file / edit_file / structural_edit), for lens training-data collection.
	// On pass completion the writes are stashed by session id; a later
	// /feedback call (per-file accept/deny + pass thumbs) turns them into
	// labeled, weighted lens samples. Captured at the same point the lens
	// scores the content, so a sample mirrors exactly what the lens saw.
	PassWrites []PassWrite

	// PassID identifies this prompt-pass (the session id from the request) so
	// deferred /feedback can find this pass's writes after the loop returns.
	PassID string

	// Rolling list of gx_score_min values
	// from lens scoring of write_file/edit_file tool calls. When the
	// recent N values all fall below the selected model's calibrated threshold the loop
	// injects a corrective system message before the next LLM call.
	// See proxy/lens.go for the pattern detection.
	LensScoreHistory []float64

	// Tool-call repetition detector: rolling window of recent (tool,
	// args) signatures. When the same signature appears
	// toolRepeatThreshold times within the last toolRepeatWindow
	// entries, the loop injects a corrective system message. Owned by
	// the detector — recordToolCall appends to it and clears it as it
	// fires; callers go through resetToolRepeatWindow rather than
	// assigning here. See proxy/detectors.go for the detection logic.
	RecentToolCalls []string

	// LastReadPath is the file most recently returned by read_file, restated
	// near the generation point so the model copies from close range.
	LastReadPath string

	// AppliedEdits keys every edit_file that already succeeded, by
	// (path, old_str, new_str). A model that re-applies an identical edit has
	// lost track of its own work, and when the edit is whitespace-only the
	// original old_str still matches afterwards, so it can apply forever.
	// Observed live: the same `'PAUSED';` -> `' PAUSED';` edit ran twice,
	// 1m25s and 1m15s, each adding another space.
	AppliedEdits map[string]bool

	// FailedToolCalls keys every tool call that has already been REJECTED,
	// by the same (name, args) signature the repetition detector uses.
	// Re-sending a byte-identical call whose last run failed cannot produce
	// a different result — the harness is deterministic — so the second one
	// is refused before it executes rather than nudged afterwards.
	FailedToolCalls map[string]string

	// LastRejectionClass is the skeleton of the previous failure's message.
	// The error-loop breaker resets when a new failure differs from it: three
	// DIFFERENT rejections is a model converging, not one looping.
	LastRejectionClass string

	// VerifiedThisRun mirrors runState.verifiedThisLoop onto the context, so
	// the mechanical lens labelling at the end of the pass can tell a run
	// that verified green from one that never did. Only the former's writes
	// are worth recording as positives.
	VerifiedThisRun bool

	// LastStreamCut records why the proxy itself ended the generation, when
	// it did. A cut lands mid-JSON, so the parse then fails — and the
	// classifier was inferring a cause from the wreckage instead of using
	// the one already known one line earlier.
	LastStreamCut string

	// OriginalContent is each touched file as the run FIRST saw it. FilesRead
	// is overwritten on every edit, so it cannot answer "what did this run
	// change".
	OriginalContent map[string]string

	// Reasoning-repetition detector state (May 10 2026, BiasBusters
	// follow-up #30). Per-turn snapshot of the model's reasoning_content
	// stream. When the same opening prose ("Now I need to look at the
	// file" / similar) appears across consecutive turns, the loop
	// injects a corrective so the model breaks out of the thought loop.
	// The streak fields are owned by the detector: recordReasoning
	// clears them as it fires and hands the count and snippet back in
	// its repeatObservation. See proxy/detectors.go.
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

	mu sync.Mutex

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
	// host execution. Set from ATLAS_VERIFY_IN=host or
	// per-project .atlas/config.toml. The default (false) is the
	// safer sandbox path; opt-in is for working codebases that
	// depend on host services (DBs, env vars, system tools) the
	// sandbox can't see. Shell-op guardrails still apply either way.
	VerifyOnHost bool

	// TrustMode gates command execution (untrusted refuses; trusted =
	// sandbox; fully-trusted permits host execution). Resolved once per
	// turn from ATLAS_TRUST_MODE.
	TrustMode trustMode

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
		Tier:            tier,
		MaxTurns:        TierMaxTurns(tier),
		WorkingDir:      workingDir,
		PermissionMode:  PermissionDefault,
		FileReadTimes:   make(map[string]time.Time),
		FilesRead:       make(map[string]string),
		BodySeen:        make(map[string]bool),
		AppliedEdits:    make(map[string]bool),
		FailedToolCalls: make(map[string]string),
		SessionWrites:   make(map[string]bool),
		Ctx:             context.Background(),
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
	// First sighting wins: FilesRead is overwritten after every edit, so it
	// tracks current state. Comparing what the RUN changed needs the state it
	// started from, which is only available the first time a path is seen.
	if c.OriginalContent == nil {
		c.OriginalContent = make(map[string]string)
	}
	if _, seen := c.OriginalContent[path]; !seen {
		c.OriginalContent[path] = content
	}
	c.FilesRead[path] = content
	c.LastReadPath = path
}

// RecordBodySeen marks a file's contents as having been shown to the model —
// read in full or in part, or authored by it. See BodySeen for why this is not
// the same as being in the read cache.
func (c *AgentContext) RecordBodySeen(path string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.BodySeen == nil {
		c.BodySeen = make(map[string]bool)
	}
	c.BodySeen[path] = true
}

// WasBodySeen reports whether the model has actually been shown this file's
// contents. Distinct from WasFileRead, which outline_file also satisfies.
func (c *AgentContext) WasBodySeen(path string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.BodySeen[path]
}

// OriginalOf returns a file's content as the run first saw it, and whether the
// run has seen it at all.
func (c *AgentContext) OriginalOf(path string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.OriginalContent[path]
	return v, ok
}

// LastRead returns the most recently read file and its content, for restating
// just before the generation point. Empty when nothing has been read.
func (c *AgentContext) LastRead() (string, string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.LastReadPath, c.FilesRead[c.LastReadPath]
}

// RecordPassWrite appends a model-authored write to this pass's collection
// list, for deferred lens-training labeling. Last-write-wins per path so a
// file the model rewrote several times in one pass yields one sample (its
// final content), not several near-duplicates.
func (c *AgentContext) RecordPassWrite(tool, path, content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := range c.PassWrites {
		if c.PassWrites[i].Path == path {
			c.PassWrites[i] = PassWrite{Tool: tool, Path: path, Content: content}
			return
		}
	}
	c.PassWrites = append(c.PassWrites, PassWrite{Tool: tool, Path: path, Content: content})
}

// WasFileRead returns true if the file was read during this agent session.
func (c *AgentContext) WasFileRead(path string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	_, ok := c.FileReadTimes[path]
	return ok
}

// GetFileRead returns the cached content for path under the context lock.
func (c *AgentContext) GetFileRead(path string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	content, ok := c.FilesRead[path]
	return content, ok
}

// ForgetFileRead drops the read-cache entry for path under the context
// lock. Used when a file is moved away from path.
func (c *AgentContext) ForgetFileRead(path string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.FilesRead, path)
}

// SnapshotFilesRead returns a copy of the session read-cache under the
// context lock, for callers that iterate over it.
func (c *AgentContext) SnapshotFilesRead() map[string]string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]string, len(c.FilesRead))
	for k, v := range c.FilesRead {
		out[k] = v
	}
	return out
}

// allowToolForTurn records a tool the user approved with session scope so
// later calls to it in this turn skip the permission prompt.
func (c *AgentContext) allowToolForTurn(toolName string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.AllowedTools == nil {
		c.AllowedTools = map[string]bool{}
	}
	c.AllowedTools[toolName] = true
}

// isToolAllowed reports whether a tool has been pre-approved for the session.
func (c *AgentContext) isToolAllowed(toolName string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.AllowedTools[toolName]
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
	// UserMessage is what the user actually asked for. Without it the
	// pipeline is handed "Create the file `solve.py`", the project context
	// and the baseline, and told to improve on the baseline "preserving all
	// functionality" — so it can only mimic a draft whose requirement it has
	// never seen, and cannot correct one that is subtly wrong. V3PlanRequest
	// has carried this since plan mode shipped; generation never did.
	UserMessage string `json:"user_message,omitempty"`
	Tier        int    `json:"tier"`
	WorkingDir  string `json:"working_dir,omitempty"`
}

// V3GenerateResponse is the response from the V3 service.
type V3GenerateResponse struct {
	Code                 string                   `json:"code"`
	Passed               bool                     `json:"passed"`
	PhaseSolved          string                   `json:"phase_solved"`
	CandidatesTested     int                      `json:"candidates_tested"`
	WinningScore         float64                  `json:"winning_score"`
	TotalTokens          int                      `json:"total_tokens"`
	TotalTimeMs          float64                  `json:"total_time_ms"`
	VerificationEvidence []V3VerificationEvidence `json:"verification_evidence,omitempty"`

	// The versioned evidence envelope (evidence_wire.go). Declared explicitly
	// because an undeclared field is silently discarded: the service began
	// sending structured evidence and this type dropped every byte of it with
	// no error anywhere. nil means the producer sent none — a legacy service,
	// or a run that measured nothing — which is not the same as evidence that
	// arrived and cannot be trusted. See EvidenceAvailability.
	//
	// Nothing in it is derived from Passed/PhaseSolved/WinningScore, and none
	// of those may be derived from it.
	Evidence *V3EvidenceEnvelope `json:"evidence,omitempty"`
	// Why the producer sent no envelope, when it knows. Travels beside the
	// absence so a gap is visible rather than silent.
	EvidenceUnavailableReason string `json:"evidence_unavailable_reason,omitempty"`
}

// V3VerificationEvidence describes the concrete verifier that accepted or
// rejected a V3 candidate. It is intentionally small and bounded by the V3
// service before crossing the wire.
type V3VerificationEvidence struct {
	Verifier   string `json:"verifier"`
	Command    string `json:"command,omitempty"`
	Status     string `json:"status"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	DurationMs int    `json:"duration_ms,omitempty"`
	Stdout     string `json:"stdout,omitempty"`
	Stderr     string `json:"stderr,omitempty"`
}

// The versioned evidence envelope, as the V3 service serialises it.
//
// This file is TRANSPORT AND VALIDATION ONLY. The domain lives in the
// service's contract.py: what closure means, how coverage is computed, how
// records compare. Nothing here re-derives any of it, and nothing here reads
// `passed`, `phase_solved`, `winning_score` or the verification-evidence list
// to guess a strength — those carry no strength, which is the whole reason
// this envelope exists.
//
// It is also kept apart from the local ToolResult validation fields. Those say
// what THIS process checked about the bytes it wrote (syntax, structural).
// The envelope says what the SERVICE demonstrated about a candidate. Merging
// them would let a local syntax pass read as behavioural evidence, which is
// the conflation the whole workstream exists to remove.

// evidenceWireMajor is the envelope contract this build understands. A
// same-major envelope may add fields; a different major may have changed the
// meaning of one, so it is not interpreted at all.
const evidenceWireMajor = "1"

// EvidenceAvailability is the only thing a consumer may branch on before
// reading any field. Three states, never two: an envelope that was not sent
// and one that cannot be trusted are different facts, and neither is a
// failure of the candidate.
type EvidenceAvailability string

const (
	// EvidenceAbsent: no envelope was sent. A legacy producer, or a run that
	// measured nothing.
	EvidenceAbsent EvidenceAvailability = "absent"
	// EvidenceUnavailable: an envelope arrived and cannot be trusted —
	// unknown version, incomplete identity, internally contradictory. NEVER
	// "failed": nothing about the candidate was demonstrated either way.
	EvidenceUnavailable EvidenceAvailability = "unavailable"
	// EvidenceAvailable: structurally valid and internally consistent. Says
	// nothing yet about whether it describes the bytes being delivered.
	EvidenceAvailable EvidenceAvailability = "available"
)

type V3EvidenceIdentity struct {
	ContractID            string `json:"contract_id"`
	ContractVersion       string `json:"contract_version"`
	AdapterID             string `json:"adapter_id"`
	AdapterVersion        string `json:"adapter_version"`
	CalibrationID         string `json:"calibration_id"`
	ArtifactScope         string `json:"artifact_scope"`
	EvaluationContextHash string `json:"evaluation_context_hash"`
	CandidateContentHash  string `json:"candidate_content_hash"`
}

type V3EvidenceQuality struct {
	RequiredCoverage float64 `json:"required_coverage"`
	OptionalQuality  float64 `json:"optional_quality"`
	Overall          float64 `json:"overall"`
}

type V3EvidenceEvaluation struct {
	ExecutionStatus      string            `json:"execution_status"`
	Supported            bool              `json:"supported"`
	EvidenceStrength     string            `json:"evidence_strength"`
	RequirementsComplete bool              `json:"requirements_complete"`
	ClosureEligible      bool              `json:"closure_eligible"`
	Quality              V3EvidenceQuality `json:"quality"`
}

// V3EvidenceOptional is one non-required criterion's observation. Ids are
// opaque; this side never interprets them.
type V3EvidenceOptional struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

type V3EvidenceCoverage struct {
	Required     []string             `json:"required"`
	Demonstrated []string             `json:"demonstrated"`
	Missing      []string             `json:"missing"`
	Unmeasurable []string             `json:"unmeasurable"`
	Optional     []V3EvidenceOptional `json:"optional"`
}

type V3EvidenceSelection struct {
	Status            string `json:"status"`
	Reason            string `json:"reason"`
	TiedCount         int    `json:"tied_count"`
	IncomparableCount int    `json:"incomparable_count"`
	IneligibleCount   int    `json:"ineligible_count"`
}

type V3EvidenceDelivery struct {
	DeliveredContentHash string `json:"delivered_content_hash"`
	// The producer's own reading. Never trusted on its own: DescribesBytes
	// recomputes the hash of what is actually about to be written.
	DescribesDeliveredCandidate bool `json:"describes_delivered_candidate"`
}

type V3EvidenceEnvelope struct {
	WireVersion         string               `json:"wire_version"`
	RecordSchemaVersion string               `json:"record_schema_version"`
	Identity            V3EvidenceIdentity   `json:"identity"`
	Evaluation          V3EvidenceEvaluation `json:"evaluation"`
	Coverage            V3EvidenceCoverage   `json:"coverage"`
	Selection           V3EvidenceSelection  `json:"selection"`
	Delivery            V3EvidenceDelivery   `json:"delivery"`
}

// V3PlanRequest is sent to the Python V3 service for plan generation.
// project_context inlines small file contents (truncated server-side) so
// the planner sees what's actually in the working directory.
type V3PlanRequest struct {
	UserMessage    string            `json:"user_message"`
	WorkingDir     string            `json:"working_dir,omitempty"`
	ProjectContext map[string]string `json:"project_context,omitempty"`
	// ExistingFiles is every path already in the workspace, relative. The
	// planner runs in v3-service, which has no /workspace mount, so it cannot
	// look for itself — and project_context only carries a handful of
	// priority files by content. Without this the planner cheerfully opens
	// with "write_file input.txt — create the necessary input data" against a
	// fixture already on disk.
	ExistingFiles []string `json:"existing_files,omitempty"`
	NCandidates   int      `json:"n_candidates,omitempty"` // 0 → server default (3)
}

// PlanStep is a single step in a Plan. Mirrors v3-service/planning.py's
// PLAN_PROMPT_TEMPLATE shape: id, action, target, why.
type PlanStep struct {
	ID     string `json:"id"`
	Action string `json:"action"`
	Target string `json:"target"`
	Why    string `json:"why"`
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
}

// ---------------------------------------------------------------------------
// SSE event types for the CLI protocol
// ---------------------------------------------------------------------------

type SSEEvent struct {
	Type string      `json:"type"` // "tool_call", "tool_result", "text", "done", "permission_request", "error"
	Data interface{} `json:"data"`
}

type PermissionRequest struct {
	ToolName   string          `json:"tool_name"`
	Args       json.RawMessage `json:"args"`
	Message    string          `json:"message"`      // human-readable description
	ToolCallID string          `json:"tool_call_id"` // echoed back on POST /v1/permission
}
