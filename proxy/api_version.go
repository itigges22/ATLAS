package main

// API / protocol versioning and a stable error-code taxonomy.
//
// APIVersion is the contract version for the proxy's HTTP + SSE surface.
// Clients read it from GET /version (and it rides on error envelopes) so
// a breaking change is a visible version bump, not a silent shape change.
//
// ErrorCode is a CLOSED set of machine-readable codes. Clients switch on
// the code, never on the human message — the message can change freely;
// the code is the contract. New failure modes get a new code; existing
// codes keep their meaning.

import (
	"encoding/json"
	"net/http"
)

// APIVersion follows semver; bump minor for additive, major for breaking.
const APIVersion = "1.0.0"

// ProtocolVersion is the SSE event-envelope contract version (see
// proxy/events.go / atlas.cli.events).
const ProtocolVersion = 1

type ErrorCode string

const (
	ErrUnauthorized     ErrorCode = "unauthorized"
	ErrInvalidInput     ErrorCode = "invalid_input"
	ErrUnsupported      ErrorCode = "unsupported_operation"
	ErrPermissionDenied ErrorCode = "permission_denied"
	ErrTimeout          ErrorCode = "timeout"
	ErrCancelled        ErrorCode = "cancelled"
	ErrDependencyDown   ErrorCode = "dependency_unavailable"
	ErrIncompatible     ErrorCode = "incompatible_artifact"
	ErrResourceLimit    ErrorCode = "resource_limit"
	ErrSandboxRejected  ErrorCode = "sandbox_policy_rejected"
	ErrModelFailure     ErrorCode = "model_failure"
	ErrInternal         ErrorCode = "internal_error"
)

// AllErrorCodes is the canonical closed set (asserted by the contract
// test against the documented taxonomy).
var AllErrorCodes = []ErrorCode{
	ErrUnauthorized, ErrInvalidInput, ErrUnsupported, ErrPermissionDenied,
	ErrTimeout, ErrCancelled, ErrDependencyDown, ErrIncompatible,
	ErrResourceLimit, ErrSandboxRejected, ErrModelFailure, ErrInternal,
}

// ErrorEnvelope is the stable error shape: a code (switch on this), a
// human message, and the API version.
type ErrorEnvelope struct {
	Error      string `json:"error"`  // the ErrorCode
	Detail     string `json:"detail"` // human message (may change)
	APIVersion string `json:"api_version"`
}

func writeError(w http.ResponseWriter, status int, code ErrorCode,
	detail string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(ErrorEnvelope{
		Error:      string(code),
		Detail:     detail,
		APIVersion: APIVersion,
	})
}

func handleVersion(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"api_version":      APIVersion,
		"protocol_version": ProtocolVersion,
		"error_codes":      AllErrorCodes,
	})
}
