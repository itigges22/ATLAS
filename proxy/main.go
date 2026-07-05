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
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
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
	log.Printf("passthrough: %s %q", r.Method, r.URL.Path)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "request body exceeds the configured limit", http.StatusRequestEntityTooLarge)
		return
	}
	upstreamURL := inferenceURL + r.URL.RequestURI()
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	proxyReq.Header = r.Header.Clone()
	resp, err := http.DefaultClient.Do(proxyReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
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
	log.SetOutput(filteringWriter{w: os.Stderr})

	addr := ":" + proxyPort
	log.Printf("ATLAS Proxy v3.0.1 starting on %s", addr)
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
		Handler:           http.MaxBytesHandler(requireServiceToken(newProxyMux()), maxRequestBodyBytes),
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
