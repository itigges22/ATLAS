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
//   atlas-proxy                  (default port 8090)
//   ATLAS_LLAMA_URL=http://localhost:8080 atlas-proxy
package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
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
	llmOK, ragOK, sandboxOK, lensReady := false, false, false, false

	if resp, err := http.Get(inferenceURL + "/health"); err == nil {
		resp.Body.Close()
		llmOK = resp.StatusCode == 200
	}
	if resp, err := http.Get(lensURL + "/health"); err == nil {
		resp.Body.Close()
		ragOK = resp.StatusCode == 200
	}
	// Geometric-lens /ready is the gate that flips to 503 when scoring is
	// degraded (lens weights missing, embedding-dim mismatch, etc — see
	// PC-019). /health stays informational; /ready is the pass/fail.
	if resp, err := http.Get(lensURL + "/ready"); err == nil {
		resp.Body.Close()
		lensReady = resp.StatusCode == 200
	}
	if resp, err := http.Get(sandboxURL + "/health"); err == nil {
		resp.Body.Close()
		sandboxOK = resp.StatusCode == 200
	}

	overall := llmOK && ragOK && sandboxOK && lensReady
	overallStatus := "ok"
	if !overall {
		overallStatus = "degraded"
	}

	status := map[string]any{
		"status":     overallStatus,
		"inference":  llmOK,
		"lens":       ragOK,
		"lens_ready": lensReady,
		"sandbox":    sandboxOK,
		"port":       proxyPort,
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

	if resp, err := http.Get(inferenceURL + "/health"); err == nil {
		resp.Body.Close()
		llmOK = resp.StatusCode == 200
	}
	if resp, err := http.Get(lensURL + "/ready"); err == nil {
		resp.Body.Close()
		lensReady = resp.StatusCode == 200
	}
	if resp, err := http.Get(sandboxURL + "/health"); err == nil {
		resp.Body.Close()
		sandboxOK = resp.StatusCode == 200
	}

	ready := llmOK && lensReady && sandboxOK
	w.Header().Set("Content-Type", "application/json")
	if !ready {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	json.NewEncoder(w).Encode(map[string]any{
		"ready":      ready,
		"inference":  llmOK,
		"lens_ready": lensReady,
		"sandbox":    sandboxOK,
	})
}

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)

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
	mux.HandleFunc("/v1/agent", handleAgent) // tool-based agent endpoint
	mux.HandleFunc("/events", handleEvents)  // PC-061: typed SSE event stream
	mux.HandleFunc("/cancel", handleCancel)  // PC-062: TUI abort hook

	// Catch-all: proxy to llama-server
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
	log.Printf("ATLAS Proxy v3.0.1 starting on %s", addr)
	log.Printf("  Inference: %s", inferenceURL)
	log.Printf("  Geometric Lens: %s", lensURL)
	log.Printf("  Sandbox: %s", sandboxURL)
	log.Printf("  Pipeline: generate → score → sandbox → repair (max %d) → deliver", maxRepairAttempts)

	if envOr("ATLAS_KEEP_LLAMA_WARM", "1") != "0" {
		go keepLlamaWarm()
		log.Printf("  Keep-warm: pinging %s every 45s (set ATLAS_KEEP_LLAMA_WARM=0 to disable)", inferenceURL)
	}

	if err := http.ListenAndServe(addr, mux); err != nil {
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
