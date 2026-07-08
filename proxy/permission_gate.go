package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// Interactive permission approval.
//
// In default and accept-edits modes a destructive tool call pauses the agent
// loop, emits a "permission_request" SSE event, and blocks until the client
// POSTs a decision to /v1/permission. This mirrors the /cancel topology: the
// agent loop is mid-turn on one HTTP request while the decision arrives on a
// separate request, correlated through a package-level sync.Map keyed by
// session id + tool-call id.

// permDecision is the client's answer to a permission_request.
type permDecision struct {
	allow bool
	// scope "session" (from the client's "allow for the rest of the session"
	// choice) additionally whitelists the tool for the remainder of the
	// current turn so a repeated call does not prompt again.
	scope string
}

// pendingPermission is the pendingPermissions map value.
type pendingPermission struct {
	decision chan permDecision
}

// pendingPermissions correlates an in-flight permission_request with the
// /v1/permission POST that answers it. Keyed by permKey(sessionID, callID).
var pendingPermissions sync.Map

func permKey(sessionID, callID string) string {
	return sessionID + "|" + callID
}

// permissionTimeout is the fail-safe: if no decision arrives (and the client
// neither disconnects nor cancels), the tool call is denied rather than
// hanging the turn forever. Overridable via ATLAS_PERMISSION_TIMEOUT_SEC.
func permissionTimeout() time.Duration {
	if v := os.Getenv("ATLAS_PERMISSION_TIMEOUT_SEC"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 10 * time.Minute
}

// awaitPermission emits a permission_request for a destructive tool call and
// blocks until the client answers, the request context is cancelled (client
// disconnect or /cancel), or the fail-safe timeout elapses. It returns true if
// the call is allowed. On an allow with session scope the tool is added to the
// turn's in-context allowlist so subsequent calls in this turn skip the prompt.
func awaitPermission(ctx *AgentContext, toolName, callID string, args json.RawMessage) bool {
	// No session id means no channel back from the client to answer a
	// prompt. Failing open here would make mode:"default" silently
	// yolo-equivalent for any client that omits session_id — deny
	// instead. Clients that want unattended destructive tools opt in
	// explicitly with mode:"yolo" (or pre-approve via
	// session_allowed_tools); interactive clients pass session_id and
	// answer /v1/permission.
	if ctx.PassID == "" {
		log.Printf("[permission] %s requires approval but the request has no session_id — denying. Pass session_id and answer POST /v1/permission, pre-approve via session_allowed_tools, or use mode \"yolo\".", toolName)
		return false
	}

	entry := &pendingPermission{decision: make(chan permDecision, 1)}
	key := permKey(ctx.PassID, callID)
	pendingPermissions.Store(key, entry)
	defer pendingPermissions.CompareAndDelete(key, entry)

	ctx.Stream("permission_request", PermissionRequest{
		ToolName:   toolName,
		Args:       args,
		Message:    describeToolCall(toolName, args),
		ToolCallID: callID,
	})

	select {
	case d := <-entry.decision:
		if d.allow && d.scope == "session" {
			ctx.allowToolForTurn(toolName)
		}
		return d.allow
	case <-ctx.Ctx.Done():
		return false
	case <-time.After(permissionTimeout()):
		log.Printf("[permission] %s timed out for session %q — denying", toolName, ctx.PassID)
		return false
	}
}

// handlePermission receives a client's approve/deny decision and signals the
// blocked agent loop. Idempotent: a decision for an unknown/already-answered
// key returns 404, mirroring /cancel.
func handlePermission(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, ErrUnsupported, "method not allowed")
		return
	}
	var req struct {
		SessionID  string `json:"session_id"`
		ToolCallID string `json:"tool_call_id"`
		Decision   string `json:"decision"` // "allow" or "deny"
		Scope      string `json:"scope"`    // "once" (default) or "session"
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "invalid request body")
		return
	}
	if req.SessionID == "" || req.ToolCallID == "" {
		writeError(w, http.StatusBadRequest, ErrInvalidInput, "session_id and tool_call_id required")
		return
	}

	v, ok := pendingPermissions.LoadAndDelete(permKey(req.SessionID, req.ToolCallID))
	w.Header().Set("Content-Type", "application/json")
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]bool{"delivered": false})
		return
	}
	entry, ok := v.(*pendingPermission)
	if !ok {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "bad permission entry"})
		return
	}
	// Buffered channel (cap 1) + LoadAndDelete guarantees exactly one send.
	entry.decision <- permDecision{allow: req.Decision == "allow", scope: req.Scope}
	log.Printf("[permission] %q %q for session %q (scope %q)",
		req.ToolCallID, req.Decision, req.SessionID, req.Scope)
	_ = json.NewEncoder(w).Encode(map[string]bool{"delivered": true})
}

// permCallID is the tool-call identifier used to correlate a permission
// request with its decision. It matches the ToolCallID the loop assigns to
// tool messages so the ids line up across the turn.
func permCallID(turn int) string {
	return fmt.Sprintf("call_%d", turn)
}
