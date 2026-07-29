// atlas-proxy: ATLAS's local inference proxy.
//
// Hosts the structured agent endpoint (`/v1/agent`), the typed event
// broker (`/events`), and the cancel hook (`/cancel`) that the TUI
// drives. Plain OpenAI traffic on `/v1/chat/completions` and unmatched
// paths are passed through to llama-server via the catch-all handler
// in main(). The verify-repair pipeline (lens scoring + sandbox +
// V3 stages) lives behind the agent loop's `write_file` tool.
//
// Usage:
//
//	atlas-proxy                  (default port 8090)
//	ATLAS_LLAMA_URL=http://localhost:8080 atlas-proxy
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

var (
	inferenceURL = envOr("ATLAS_INFERENCE_URL", "http://localhost:8080")
	lensURL      = envOr("ATLAS_LENS_URL", "http://localhost:8099")
	sandboxURL   = envOr("ATLAS_SANDBOX_URL", "http://localhost:30820")
	v3URL        = envOr("ATLAS_V3_URL", "http://localhost:8070")
	proxyPort    = envOr("ATLAS_PROXY_PORT", "8090")
	modelName    = envOr("ATLAS_MODEL_NAME", "local-model")
	healthClient = &http.Client{Timeout: 3 * time.Second}
	// v3-service can take longer to answer when a pipeline run is in
	// flight; keep its readiness probe on a shorter leash so /ready
	// stays snappy.
	v3HealthClient = &http.Client{Timeout: 2 * time.Second}
)

const (
	demoRawCapability   = "demo_raw_completion_v1"
	maxRepairAttempts   = 3
	maxRequestBodyBytes = 16 << 20
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// resolveVerifyTarget returns "host" when run_command should bypass
// the sandbox and execute on the host, or "sandbox" otherwise. PC-192.
//
// Resolution order (later wins):
//  1. ATLAS_VERIFY_IN env var ("host" or "sandbox")
//  2. Per-project .atlas/config.toml — looks for `target = "host"` or
//     `target = "sandbox"` under an [execution] header. Trivially
//     parsed (no real TOML lib) so we don't take a dep just for one
//     setting; refuse to be clever about quoting.
//
// Default: "sandbox" (the safer path). Per-project config is the
// usual customization point for working codebases that need host
// execution; the env var is for one-off sessions and CI.
func resolveVerifyTarget(workingDir string) string {
	target := strings.ToLower(os.Getenv("ATLAS_VERIFY_IN"))
	if target != "host" && target != "sandbox" {
		target = "sandbox"
	}
	if workingDir == "" {
		return target
	}
	cfg, err := os.ReadFile(filepath.Join(workingDir, ".atlas", "config.toml"))
	if err != nil {
		return target
	}
	inExecution := false
	for _, raw := range strings.Split(string(cfg), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			inExecution = strings.EqualFold(strings.Trim(line, "[]"), "execution")
			continue
		}
		if !inExecution {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[0]) != "target" {
			continue
		}
		val := strings.ToLower(strings.Trim(strings.TrimSpace(parts[1]), `"'`))
		if val == "host" || val == "sandbox" {
			return val
		}
	}
	return target
}

// ---------------------------------------------------------------------------
// Telemetry counters
// ---------------------------------------------------------------------------

var (
	totalRequests atomic.Int64
	totalRepairs  atomic.Int64
	sandboxPasses atomic.Int64
	sandboxFails  atomic.Int64
)

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
// HTTP server setup
// ---------------------------------------------------------------------------

func handleModels(w http.ResponseWriter, r *http.Request) {
	// Prefer llama-server's loaded model over our configured fallback. This
	// keeps the API (and /demo title) truthful when a local launch overrides
	// ATLAS_MODEL_NAME or the local .env lags behind the running server.
	id := modelName
	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodGet,
		strings.TrimRight(inferenceURL, "/")+"/v1/models", nil)
	if err == nil {
		if upstream, upstreamErr := healthClient.Do(upstreamReq); upstreamErr == nil {
			defer upstream.Body.Close()
			if upstream.StatusCode == http.StatusOK {
				var loaded struct {
					Data []struct {
						ID string `json:"id"`
					} `json:"data"`
				}
				if decodeErr := json.NewDecoder(io.LimitReader(upstream.Body, 1<<20)).Decode(&loaded); decodeErr == nil {
					for _, candidate := range loaded.Data {
						if candidate.ID = strings.TrimSpace(candidate.ID); candidate.ID != "" {
							id = candidate.ID
							break
						}
					}
				}
			}
		}
	}
	resp := map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"id": id, "object": "model", "owned_by": "atlas"},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	llmOK, ragOK, sandboxOK, lensReady := false, false, false, false

	if resp, err := healthClient.Get(inferenceURL + "/health"); err == nil {
		resp.Body.Close()
		llmOK = resp.StatusCode == 200
	}
	if resp, err := healthClient.Get(lensURL + "/health"); err == nil {
		resp.Body.Close()
		ragOK = resp.StatusCode == 200
	}
	// Geometric-lens /ready is the gate that flips to 503 when scoring is
	// degraded (lens weights missing, embedding-dim mismatch, etc — see
	// PC-019). /health stays informational; /ready is the pass/fail.
	if resp, err := healthClient.Get(lensURL + "/ready"); err == nil {
		resp.Body.Close()
		lensReady = resp.StatusCode == 200
	}
	if resp, err := healthClient.Get(sandboxURL + "/health"); err == nil {
		resp.Body.Close()
		sandboxOK = resp.StatusCode == 200
	}

	overall := llmOK && ragOK && sandboxOK && lensReady
	overallStatus := "ok"
	if !overall {
		overallStatus = "degraded"
	}

	status := map[string]any{
		"status":       overallStatus,
		"inference":    llmOK,
		"lens":         ragOK,
		"lens_ready":   lensReady,
		"sandbox":      sandboxOK,
		"port":         proxyPort,
		"capabilities": []string{demoRawCapability},
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

func handleReady(w http.ResponseWriter, r *http.Request) {
	llmOK, sandboxOK, lensReady := false, false, false

	if resp, err := healthClient.Get(inferenceURL + "/health"); err == nil {
		resp.Body.Close()
		llmOK = resp.StatusCode == 200
	}
	if resp, err := healthClient.Get(lensURL + "/ready"); err == nil {
		resp.Body.Close()
		lensReady = resp.StatusCode == 200
	}
	if resp, err := healthClient.Get(sandboxURL + "/health"); err == nil {
		resp.Body.Close()
		sandboxOK = resp.StatusCode == 200
	}
	// T2/T3 writes route through v3-service, so readiness includes it
	// whenever a V3 URL is configured.
	v3OK := true
	if v3URL != "" {
		v3OK = false
		if resp, err := v3HealthClient.Get(v3URL + "/health"); err == nil {
			resp.Body.Close()
			v3OK = resp.StatusCode == 200
		}
	}

	ready := llmOK && lensReady && sandboxOK && v3OK
	w.Header().Set("Content-Type", "application/json")
	if !ready {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	json.NewEncoder(w).Encode(map[string]any{
		"ready":      ready,
		"inference":  llmOK,
		"lens_ready": lensReady,
		"sandbox":    sandboxOK,
		"v3":         v3OK,
	})
}

func newProxyMux() *http.ServeMux {
	mux := http.NewServeMux()
	// /v1/chat/completions used to be wrapped here with the Aider whole-
	// file output format and embedded agent loop. After PC-062 the TUI
	// uses /v1/agent for everything, and Aider was removed in the cleanup
	// pass — so the OpenAI-compat endpoint now passes through to
	// llama-server unchanged via the catch-all registered below. Anyone
	// hitting /v1/chat/completions on the proxy gets the raw upstream
	// behavior; structured agent turns belong on /v1/agent.
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/models", handleModels)
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/ready", handleReady)
	mux.HandleFunc("/v1/agent", handleAgent)                             // tool-based agent endpoint
	mux.HandleFunc("/events", handleEvents)                              // PC-061: typed SSE event stream
	mux.HandleFunc("/cancel", handleCancel)                              // PC-062: TUI abort hook
	mux.HandleFunc("/v1/permission", handlePermission)                   // interactive approve/deny for destructive tools
	mux.HandleFunc("/feedback", handleFeedback)                          // per-file accept/deny + pass thumbs → lens samples
	mux.HandleFunc("/v1/lens/training-status", handleLensTrainingStatus) // sample counts for the "retrain available" alert
	// PC-059: TUI calls this on connect to render a Lens/ASA compat badge.
	mux.HandleFunc("/v1/calibration/status", handleCalibrationStatus)
	mux.HandleFunc("/version", handleVersion)

	// Catch-all: proxy to llama-server
	mux.HandleFunc("/", handlePassthrough)
	return mux
}

func handlePassthrough(w http.ResponseWriter, r *http.Request) {
	// %q on the path quotes + escapes CR/LF so a crafted URL can't
	// fake additional log entries (go/log-injection).
	logEvent("info", fmt.Sprintf("passthrough: %s %q", r.Method, r.URL.Path),
		requestIDFromContext(r.Context()), nil)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, ErrResourceLimit, "request body exceeds the configured limit")
		return
	}
	upstreamURL := inferenceURL + r.URL.RequestURI()
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusInternalServerError, ErrInternal, err.Error())
		return
	}
	proxyReq.Header = r.Header.Clone()
	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, ErrDependencyDown, err.Error())
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
}

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)
	// Private-value filtering: every log line passes through the
	// filter before it reaches stderr (see private_values.go).
	// In json mode the filtered line is then wrapped into a JSON
	// record; the record stamps its own ts, so the log package's
	// time prefix is dropped to keep it out of msg.
	out := io.Writer(os.Stderr)
	if logJSON {
		log.SetFlags(0)
		out = jsonLineWriter{w: os.Stderr}
	}
	log.SetOutput(filteringWriter{w: out})

	addr := ":" + proxyPort
	log.Printf("ATLAS Proxy v3.1.3 starting on %s", addr)
	log.Printf("  Inference: %s", inferenceURL)
	log.Printf("  Geometric Lens: %s", lensURL)
	log.Printf("  Sandbox: %s", sandboxURL)
	log.Printf("  Pipeline: generate → score → sandbox → repair (max %d) → deliver", maxRepairAttempts)

	// PC-059: probe geometric-lens + ASA calibration so operators see the
	// same verdict the TUI's header badge will render. The old "ASA
	// steering: present at X" banner is folded into logCalibrationStatusAtStartup
	// below (which also adds the corresponding Lens line) so the proxy
	// surfaces a unified calibration view at startup.
	installTokenTransport()

	logCalibrationStatusAtStartup()

	if envOr("ATLAS_KEEP_LLAMA_WARM", "1") != "0" {
		go keepLlamaWarm()
		log.Printf("  Keep-warm: pinging %s every 45s (set ATLAS_KEEP_LLAMA_WARM=0 to disable)", inferenceURL)
	}

	server := &http.Server{
		Addr:              addr,
		Handler:           http.MaxBytesHandler(withRequestID(requireServiceToken(newProxyMux())), maxRequestBodyBytes),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       90 * time.Second,
	}
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// keepLlamaWarm pings llama-server with a 1-token completion every 45s. Keeps
// the model loaded in VRAM, the slot's prompt cache live, and the TCP keepalive
// fresh — avoiding the cold-start path that fires after 1-2 min idle. See
// ISSUES.md PC-035. Disable with ATLAS_KEEP_LLAMA_WARM=0.
func keepLlamaWarm() {
	const interval = 45 * time.Second
	// Wait for llama-server to come up before starting the loop.
	time.Sleep(15 * time.Second)
	body, _ := json.Marshal(map[string]any{
		"messages":    []map[string]string{{"role": "user", "content": "."}},
		"max_tokens":  1,
		"temperature": 0.0,
	})
	client := &http.Client{Timeout: 60 * time.Second}
	for {
		req, err := http.NewRequest("POST", inferenceURL+"/v1/chat/completions", bytes.NewReader(body))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
			resp, err := client.Do(req)
			if err == nil {
				resp.Body.Close()
			}
		}
		time.Sleep(interval)
	}
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

// Correlation IDs + structured logging.
//
// Every inbound request gets an X-ATLAS-Request-ID (read from the client
// or generated), echoed in the response and stored in the request
// context. Outbound calls to llama/v3/lens/sandbox forward the same ID
// (tokenTransport reads it from the request context), so one turn is
// traceable across services.
//
// Log format is line-oriented by default; ATLAS_LOG_FORMAT=json emits
// one JSON object per line with stable fields. Both paths still pass
// through the private-value filter (main() wraps the log writer).

const requestIDHeader = "X-ATLAS-Request-ID"

type ctxKey string

const requestIDKey ctxKey = "atlas-request-id"

func newRequestID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "req-unknown"
	}
	return "req-" + hex.EncodeToString(b)
}

func requestIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey).(string); ok {
		return v
	}
	return ""
}

// withRequestID wraps a handler so every request carries a correlation
// ID (client-provided or generated), echoed back and put in the context.
func withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSpace(r.Header.Get(requestIDHeader))
		if id == "" {
			id = newRequestID()
		}
		w.Header().Set(requestIDHeader, id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// --- structured logging --------------------------------------------------

var logJSON = strings.EqualFold(os.Getenv("ATLAS_LOG_FORMAT"), "json")

// logEvent emits one structured record. In json mode it's a JSON object
// with stable fields; otherwise a readable line. request_id is included
// when present. Fields beyond the standard set are passed as kv pairs.
func logEvent(level, msg, requestID string, kv map[string]interface{}) {
	if logJSON {
		rec := map[string]interface{}{
			"ts":      time.Now().UTC().Format(time.RFC3339Nano),
			"level":   level,
			"service": "atlas-proxy",
			"version": APIVersion,
			"msg":     msg,
		}
		if requestID != "" {
			rec["request_id"] = requestID
		}
		for k, v := range kv {
			rec[k] = v
		}
		b, err := json.Marshal(rec)
		if err != nil {
			log.Printf("%s: %s", level, msg)
			return
		}
		log.Printf("%s", b)
		return
	}
	// line mode
	if requestID != "" {
		log.Printf("[%s] [%s] %s", level, requestID, msg)
	} else {
		log.Printf("[%s] %s", level, msg)
	}
}

// jsonLineWriter converts each (already private-value-filtered) log line
// into the same JSON record shape logEvent emits, so ATLAS_LOG_FORMAT=json
// covers every log call in the process, not only logEvent call sites.
// Lines that are already JSON objects (logEvent's json-mode output) pass
// through unchanged.
type jsonLineWriter struct {
	w io.Writer
}

func (j jsonLineWriter) Write(p []byte) (int, error) {
	line := bytes.TrimRight(p, "\n")
	if len(line) > 0 && line[0] == '{' && json.Valid(line) {
		return j.w.Write(p)
	}
	rec := map[string]interface{}{
		"ts":      time.Now().UTC().Format(time.RFC3339Nano),
		"level":   "info",
		"service": "atlas-proxy",
		"version": APIVersion,
		"msg":     string(line),
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return j.w.Write(p)
	}
	b = append(b, '\n')
	if _, err := j.w.Write(b); err != nil {
		return 0, err
	}
	return len(p), nil
}

// Internal service authentication (per-installation token).
//
// One random token, generated by `atlas init` into
// secrets/service-token (0600) and mounted read-only into every
// container at /run/atlas-secrets/service-token, authenticates
// internal and client requests as `Authorization: Bearer <token>`.
//
// Enforcement is enabled iff a token file is configured and readable —
// an install that never ran `atlas init` keeps today's open-localhost
// behavior, and `atlas doctor` flags it. /health and /ready stay
// unauthenticated (compose/K8s probes are headerless curl).
//
// The token value must never be logged, placed in argv, or echoed in
// error bodies.

// serviceToken is loaded once at startup. Rotation = rewrite the file
// (atlas init --rotate-token) + restart the stack.
var serviceToken = loadServiceToken()

func loadServiceToken() string {
	path := os.Getenv("ATLAS_SERVICE_TOKEN_FILE")
	if path == "" {
		path = "/run/atlas-secrets/service-token"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "" // unconfigured => auth disabled (doctor warns)
	}
	return strings.TrimSpace(string(data))
}

// authOpenPaths never require the token: health probes are headerless
// curl in compose/K8s, and /ready gates orchestration.
func authOpenPath(path string) bool {
	return path == "/health" || path == "/ready" || path == "/version"
}

func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	const prefix = "Bearer "
	if strings.HasPrefix(h, prefix) {
		return h[len(prefix):]
	}
	return ""
}

// requireServiceToken wraps a handler with token enforcement.
func requireServiceToken(next http.Handler) http.Handler {
	if serviceToken == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if authOpenPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		got := bearerToken(r)
		if subtle.ConstantTimeCompare([]byte(got), []byte(serviceToken)) != 1 {
			// No token material in the response or the log line.
			writeError(w, http.StatusUnauthorized, ErrUnauthorized,
				"internal service auth is enabled; send Authorization: "+
					"Bearer <service-token> (secrets/service-token)")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// tokenTransport injects the service token into outbound requests
// (proxy -> llama/v3/lens/sandbox) unless the caller already set an
// Authorization header (e.g. the /v1/chat passthrough forwarding a
// client's own header).
type tokenTransport struct {
	base http.RoundTripper
}

func (t *tokenTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Forward the correlation ID (if the request carries one in context)
	// so downstream service logs join the same trace.
	if id := requestIDFromContext(req.Context()); id != "" &&
		req.Header.Get(requestIDHeader) == "" {
		req = req.Clone(req.Context())
		req.Header.Set(requestIDHeader, id)
	}
	if serviceToken != "" && req.Header.Get("Authorization") == "" {
		req = req.Clone(req.Context())
		req.Header.Set("Authorization", "Bearer "+serviceToken)
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(req)
}

// installTokenTransport wires outbound injection through the two
// transport choke points: the process default (covers every client
// with a nil Transport — http.DefaultClient, &http.Client{} literals,
// http.Post) and the dedicated LLM streaming client.
// The transport is installed even when auth is unconfigured: RoundTrip
// guards token injection on serviceToken, and correlation-ID forwarding
// must work on open-localhost installs too.
func installTokenTransport() {
	http.DefaultTransport = &tokenTransport{base: http.DefaultTransport}
	llmStreamClient.Transport = &tokenTransport{base: llmStreamClient.Transport}
	if serviceToken != "" {
		log.Printf("  Internal auth: enabled (token file configured)")
	}
}

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

// safeLogField encodes untrusted text as one quoted ASCII log field. Newlines,
// carriage returns, and control bytes become escape sequences, so model/user
// data cannot forge additional log records.
func safeLogField(value string, maxLen int) string {
	if maxLen > 0 {
		value = truncateStr(value, maxLen)
	}
	return strconv.QuoteToASCII(value)
}

// Private-value filtering: masks values that look like credentials
// before they reach a serialized sink. The proxy installs this on the
// standard logger's output (one choke point covers every log.Printf),
// so an error that happens to embed an env assignment or a header
// never lands in the log verbatim.
//
// The pattern spec is shared with the Python services via the fixture
// corpus at tests/fixtures/private_value_fixtures.json — change the
// patterns here and there together; the contract test runs the corpus
// against every implementation.
//
// Patterns are deliberately conservative (assignment/header/key-block
// shapes with secret-ish key names) so ordinary log content —
// "timeout=30", token counts, health URLs — passes through untouched.

const privateValuePlaceholder = "[FILTERED]"

var privateValuePatterns = []*regexp.Regexp{
	// KEY=value / key: value / "key": "value" assignments where the key
	// smells like a credential. Value part is masked, key kept.
	regexp.MustCompile(`(?i)([A-Z0-9_.-]{0,64}(?:api[_-]?key|apikey|token|secret|password|passwd|credential|access[_-]?key)[A-Z0-9_.-]{0,64}["']?\s*[=:]\s*["']?)([^\s"',;&]+)`),
	// Authorization / bearer values.
	regexp.MustCompile(`(?i)(bearer\s+)([A-Za-z0-9._~+/=-]+)`),
	// URL userinfo passwords: scheme://user:pass@host
	regexp.MustCompile(`(://[^/:@\s]{0,64}:)([^@\s]{1,256})(@)`),
	// Private-key blocks (any BEGIN ... PRIVATE KEY variant), body inclusive.
	regexp.MustCompile(`(?s)-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----.*?-----END [A-Z ]{0,40}PRIVATE KEY-----`),
}

// filterPrivateValues masks credential-shaped substrings in s.
func filterPrivateValues(s string) string {
	// Key-block pattern replaces the whole match; assignment patterns
	// keep the key and mask the value.
	s = privateValuePatterns[3].ReplaceAllString(s, privateValuePlaceholder)
	s = privateValuePatterns[0].ReplaceAllString(s, "${1}"+privateValuePlaceholder)
	s = privateValuePatterns[1].ReplaceAllString(s, "${1}"+privateValuePlaceholder)
	s = privateValuePatterns[2].ReplaceAllString(s, "${1}"+privateValuePlaceholder+"${3}")
	return s
}

// filteringWriter applies the filter to every write — installed as the
// standard logger's output in main(), so all proxy log lines pass
// through it. Line-buffered writes from log.Printf arrive whole, so
// per-write filtering is sound for the standard logger.
type filteringWriter struct {
	w io.Writer
}

func (f filteringWriter) Write(p []byte) (int, error) {
	filtered := filterPrivateValues(string(p))
	if _, err := f.w.Write([]byte(filtered)); err != nil {
		return 0, err
	}
	// Report the original length so log.Printf never sees a short write.
	return len(p), nil
}
