// Calibration status endpoint — surfaces lens + ASA compat for the TUI.
//
// PC-059 (#101): the geometric-lens /health endpoint already exposes the
// data we need (cost_field_dim, embed_dim, cost_field_loaded). This file
// forwards that into a verdict-shaped response under /v1/calibration/status
// that the TUI renders as a header badge.
//
// PC-061 (#113) extends the `asa` block from a file-presence check to a
// proper dim-vs-model probe; the JSON shape stays the same so TUI
// rendering doesn't churn.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// CalibrationStatus is the JSON returned by /v1/calibration/status.
// Shape is stable: TUI and atlas doctor both key off it.
type CalibrationStatus struct {
	Lens       LensStatus        `json:"lens"`
	ASA        ASAStatus         `json:"asa"`
	Dimensions []StatusDimension `json:"dimensions"`
}

// StatusDimension is one row of the canonical seven-dimension status
// (SUPPORT_MATRIX § "Reference-model status dimensions"). Separating
// these prevents the ambiguity where "the lens works" conflated model
// runtime, raw scoring, calibration, and intervention behavior. Every
// surface that shows lens/ASA status (this endpoint, the TUI badge,
// atlas doctor, atlas lens check) renders the SAME rows so they cannot
// disagree — they all read this list.
type StatusDimension struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Detail string `json:"detail"`
}

// buildDimensions maps the raw lens/ASA probe onto the seven named
// dimensions. Intervention is reported "neutral" whenever calibration is
// absent, matching the enforced runtime behavior (agent.go only applies
// thresholds when calibratedThresholds() succeeds) — a disabled/
// uncalibrated lens never steers using another model's cutoffs.
func buildDimensions(lens LensStatus, asa ASAStatus) []StatusDimension {
	reachable := lens.Verdict != "unreachable"

	modelRuntime := "supported"
	modelDetail := "model served and reachable"
	if !reachable {
		modelRuntime = "unreachable"
		modelDetail = "lens/model service not reachable"
	}

	// Identity/dimension contract.
	identity := "supported"
	identityDetail := "cost field matches the served model's dimension"
	switch {
	case !reachable:
		identity, identityDetail = "unknown", "service unreachable"
	case !lens.CostFieldLoaded:
		identity, identityDetail = "no-artifacts",
			"no cost field loaded for this model"
	case lens.EmbedDim > 0 && lens.CostFieldDim != lens.EmbedDim:
		identity, identityDetail = "dim-mismatch",
			fmt.Sprintf("cost field is %d-dim, model emits %d-dim",
				lens.CostFieldDim, lens.EmbedDim)
	}

	// Raw scoring availability.
	scoring := "disabled"
	scoringDetail := "cost field / G(x) not loaded"
	if reachable && lens.CostFieldLoaded && lens.GxLoaded {
		scoring, scoringDetail = "supported", "C(x) + G(x) scoring available"
	} else if reachable && lens.CostFieldLoaded && !lens.GxLoaded {
		scoring, scoringDetail = "partial", "C(x) loaded; G(x) missing"
	}

	// Calibration.
	calibration := "disabled"
	calDetail := "artifacts not loaded"
	if reachable && lens.CostFieldLoaded {
		if lens.CxCalibrated && lens.GxCalibrated {
			calibration, calDetail = "calibrated",
				"per-model normalization + thresholds loaded"
		} else {
			calibration, calDetail = "uncalibrated",
				"loaded without this model's calibration files"
		}
	}

	// Intervention behavior — neutral/disabled unless calibrated.
	intervention := "disabled"
	intDetail := "no scoring; no intervention"
	if calibration == "calibrated" {
		intervention, intDetail = "active",
			"threshold interventions enabled"
	} else if scoring != "disabled" {
		intervention, intDetail = "neutral",
			"raw telemetry only; no automatic intervention"
	}

	return []StatusDimension{
		{"model_runtime", modelRuntime, modelDetail},
		{"direct_agent", "supported",
			"model-agnostic; independent of lens/ASA state"},
		{"lens_identity", identity, identityDetail},
		{"lens_scoring", scoring, scoringDetail},
		{"lens_calibration", calibration, calDetail},
		{"lens_intervention", intervention, intDetail},
		{"asa", asa.Verdict, asa.Hint},
	}
}

type LensStatus struct {
	// "supported" | "no-artifacts" | "incomplete-artifacts" |
	// "uncalibrated" | "dim-mismatch" | "unreachable"
	Verdict         string `json:"verdict"`
	CostFieldLoaded bool   `json:"cost_field_loaded"`
	CostFieldDim    int    `json:"cost_field_dim"`
	EmbedDim        int    `json:"embed_dim"`
	GxLoaded        bool   `json:"gx_loaded"`
	CxCalibrated    bool   `json:"cx_calibrated"`
	GxCalibrated    bool   `json:"gx_calibrated"`
	Hint            string `json:"hint"`
}

type ASAStatus struct {
	// "supported" | "missing" | "unverified"
	Verdict       string `json:"verdict"`
	VectorPath    string `json:"vector_path"`
	VectorPresent bool   `json:"vector_present"`
	Hint          string `json:"hint"`
}

// lensHealthShape mirrors the lens /health JSON we read. Defensive — the
// service can be reachable but mid-startup with partial fields. We treat
// missing fields as zero values rather than failing the whole probe.
type lensHealthShape struct {
	Status     string `json:"status"`
	Subsystems struct {
		Lens struct {
			Enabled         bool   `json:"enabled"`
			CostFieldLoaded bool   `json:"cost_field_loaded"`
			CostFieldDim    int    `json:"cost_field_dim"`
			EmbedDim        int    `json:"embed_dim"`
			GxLoaded        bool   `json:"gx_loaded"`
			CxCalibrated    bool   `json:"cx_calibrated"`
			GxCalibrated    bool   `json:"gx_calibrated"`
			SelfTestPass    bool   `json:"self_test_pass"`
			SelfTestError   string `json:"self_test_error"`
		} `json:"lens"`
	} `json:"subsystems"`
}

// probeLensStatus calls the lens /health endpoint and renders a verdict.
// Timeout is short — this fires on a TUI startup ping and on the proxy's
// own startup banner; we don't want to block either if the lens is wedged.
func probeLensStatus(ctx context.Context, lensBaseURL string) LensStatus {
	out := LensStatus{Verdict: "unreachable",
		Hint: "geometric-lens unreachable at " + lensBaseURL +
			" (is the stack up?)"}

	pCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(pCtx, "GET", lensBaseURL+"/health", nil)
	if err != nil {
		return out
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return out
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return out
	}
	var h lensHealthShape
	if err := json.Unmarshal(body, &h); err != nil {
		out.Hint = "lens /health returned non-JSON: " + truncate(string(body), 80)
		return out
	}

	out.CostFieldLoaded = h.Subsystems.Lens.CostFieldLoaded
	out.CostFieldDim = h.Subsystems.Lens.CostFieldDim
	out.EmbedDim = h.Subsystems.Lens.EmbedDim
	out.GxLoaded = h.Subsystems.Lens.GxLoaded
	out.CxCalibrated = h.Subsystems.Lens.CxCalibrated
	out.GxCalibrated = h.Subsystems.Lens.GxCalibrated

	switch {
	case !out.CostFieldLoaded:
		out.Verdict = "no-artifacts"
		if h.Subsystems.Lens.SelfTestError != "" {
			out.Hint = h.Subsystems.Lens.SelfTestError
		} else {
			out.Hint = "no cost_field.pt loaded — run `atlas lens build` to train one"
		}
	case out.EmbedDim > 0 && out.CostFieldDim != out.EmbedDim:
		out.Verdict = "dim-mismatch"
		out.Hint = fmt.Sprintf("cost_field expects %d-dim, model emits %d-dim "+
			"— run `atlas lens build` to retrain at the model's native dim",
			out.CostFieldDim, out.EmbedDim)
	case !out.GxLoaded:
		out.Verdict = "incomplete-artifacts"
		out.Hint = "C(x) loaded but G(x) artifacts are missing — run `atlas lens build`"
	case !out.CxCalibrated || !out.GxCalibrated:
		out.Verdict = "uncalibrated"
		out.Hint = "Lens weights loaded without this model's calibration files — " +
			"run `atlas lens build` to generate cx_normalization.json and gx_thresholds.json"
	default:
		out.Verdict = "supported"
		out.Hint = "ready"
	}
	return out
}

// probeASAStatus checks for the configured ASA control-vector file on disk.
// V3.1.2 (PC-061): the configured path is container-relative (e.g.
// /models/ast_edit_steering.gguf as llama-server sees it). The proxy
// container doesn't have /models mounted, so we try several candidate
// host-visible paths before giving up:
//
//  1. The configured path verbatim (works when proxy DOES have a /models
//     mount — some K3s deployments do).
//  2. <workspace>/models/<basename> (proxy's bind-mounted project root,
//     ATLAS_WORKSPACE_DIR, plus the standard models/ subdir).
//  3. The env-supplied ATLAS_MODELS_DIR if set.
//
// llama-server is the authoritative source of "is the vector actually
// loaded" but doesn't expose that via /props (verified 2026-05-17), so
// disk presence is the best we can do without an out-of-band probe.
// For the user-facing verdict, `atlas asa check` does the deeper GGUF
// dim parse on the host — this endpoint is the "first impression" the
// TUI badge renders.
func probeASAStatus() ASAStatus {
	configured := envOr("ATLAS_CONTROL_VECTOR", "/models/ast_edit_steering.gguf")
	out := ASAStatus{VectorPath: configured, Verdict: "unverified"}

	// Candidate paths to probe, in order.
	candidates := []string{configured}
	if strings.HasPrefix(configured, "/models/") {
		base := strings.TrimPrefix(configured, "/models/")
		workspace := envOr("ATLAS_WORKSPACE_DIR", "/workspace")
		candidates = append(candidates,
			workspace+"/models/"+base)
		if mdir := os.Getenv("ATLAS_MODELS_DIR"); mdir != "" {
			candidates = append(candidates, mdir+"/"+base)
		}
	}

	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil {
			out.VectorPresent = true
			out.VectorPath = p
			expected := os.Getenv("ATLAS_MODEL_NAME")
			markedFor := ""
			if raw, readErr := os.ReadFile(p + ".model"); readErr == nil {
				markedFor = strings.TrimSpace(string(raw))
			}
			size := strconv.FormatInt(info.Size(), 10)
			switch {
			case expected != "" && sameModelIdentity(markedFor, expected):
				out.Verdict = "supported"
				out.Hint = "control vector verified for " + expected +
					" (" + size + " bytes)"
			case expected != "" && markedFor != "":
				out.Verdict = "incompatible"
				out.Hint = "control vector is marked for " + markedFor +
					", but the selected model is " + expected
			default:
				out.Verdict = "unverified"
				out.Hint = "control vector present (" + size +
					" bytes) without a matching model marker; run `atlas asa build`"
			}
			return out
		}
	}

	out.VectorPresent = false
	out.Verdict = "missing"
	out.Hint = "no control vector at " + configured +
		" (also tried workspace/models/ + ATLAS_MODELS_DIR) — " +
		"build one via `atlas asa build` " +
		"or see geometric-lens/asa_calibration/README.md"
	return out
}

func sameModelIdentity(a, b string) bool {
	canonical := func(value string) string {
		value = strings.ToLower(strings.TrimSpace(value))
		value = strings.TrimSuffix(value, ".gguf")
		if slash := strings.LastIndex(value, "/"); slash >= 0 {
			value = value[slash+1:]
		}
		return value
	}
	return canonical(a) != "" && canonical(a) == canonical(b)
}

func handleCalibrationStatus(w http.ResponseWriter, r *http.Request) {
	lens := probeLensStatus(r.Context(), lensURL)
	asa := probeASAStatus()
	status := CalibrationStatus{
		Lens:       lens,
		ASA:        asa,
		Dimensions: buildDimensions(lens, asa),
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_ = json.NewEncoder(w).Encode(status)
}

// logCalibrationStatusAtStartup is called once from main() so operators
// see the same compat verdict the TUI will render, in the proxy banner.
// Fail-soft: if the lens service isn't reachable yet, we log it and move
// on — startup blocks long enough as-is without a synchronous probe.
func logCalibrationStatusAtStartup() {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	lens := probeLensStatus(ctx, lensURL)
	asa := probeASAStatus()
	log.Printf("  Lens: %s — %s", lens.Verdict, lens.Hint)
	log.Printf("  ASA:  %s — %s", asa.Verdict, asa.Hint)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
