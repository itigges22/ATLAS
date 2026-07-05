// Calibration status badge — PC-059 (#101) + PC-061 (#113).
//
// On startup the TUI fetches /v1/calibration/status from the proxy and
// renders a compact badge next to the Pipeline pane title so users
// immediately see whether the loaded model has supported Lens artifacts
// and whether an ASA control vector is in play. Verdict comes from the
// proxy's CalibrationStatus (proxy/calibration_status.go).
//
// Why the TUI does this: when a user swaps in a non-default GGUF, the
// lens silently no-ops and the agent loop still "works" — but without
// G(x) verification half the value of ATLAS is missing. The badge is
// the only visible signal that something is up.

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type calibrationStatus struct {
	Lens struct {
		Verdict         string `json:"verdict"`
		CostFieldLoaded bool   `json:"cost_field_loaded"`
		CostFieldDim    int    `json:"cost_field_dim"`
		EmbedDim        int    `json:"embed_dim"`
		GxLoaded        bool   `json:"gx_loaded"`
		CxCalibrated    bool   `json:"cx_calibrated"`
		GxCalibrated    bool   `json:"gx_calibrated"`
		Hint            string `json:"hint"`
	} `json:"lens"`
	ASA struct {
		Verdict       string `json:"verdict"`
		VectorPath    string `json:"vector_path"`
		VectorPresent bool   `json:"vector_present"`
		Hint          string `json:"hint"`
	} `json:"asa"`
}

type calibrationStatusMsg struct {
	status *calibrationStatus
	err    error
}

// calibrationRetryMsg fires when the model wants to retry the calibration
// fetch — typically because a prior fetch failed and the proxy may have
// come up since (common during `docker compose up -d`, where the TUI can
// launch faster than the proxy finishes its startup probe).
type calibrationRetryMsg struct{}

// calibrationRefreshMsg fires periodically after a successful fetch so
// the badge converges on truth over time. The original bug: a user who
// opens the TUI before lens weights are downloaded sees `Lens ⚠` and
// the badge is frozen for the rest of the session — even if they then
// run `atlas model install-artifacts` and restart the lens container,
// the TUI never re-probes. This msg drives the periodic re-fetch.
type calibrationRefreshMsg struct{}

// scheduleCalibrationRetry returns a Cmd that emits a retry trigger after
// the given delay. The model's Update handler decides whether to actually
// re-fire fetchCalibrationStatusCmd based on retry count + current state.
func scheduleCalibrationRetry(after time.Duration) tea.Cmd {
	return tea.Tick(after, func(time.Time) tea.Msg {
		return calibrationRetryMsg{}
	})
}

// scheduleCalibrationRefresh returns a Cmd that emits a refresh trigger
// after the given delay. Separate type from retry so the handler can
// apply different rules — refresh fires forever, doesn't care about
// the prior verdict, and runs at a longer interval.
func scheduleCalibrationRefresh(after time.Duration) tea.Cmd {
	return tea.Tick(after, func(time.Time) tea.Msg {
		return calibrationRefreshMsg{}
	})
}

// maxCalibrationRetries caps the *retry* loop (fast attempts after a
// failed initial fetch). At ~5s/retry, 5 attempts covers ~25s of proxy-
// startup warmup — long enough for the slowest realistic docker compose
// up, short enough that a permanently-down proxy doesn't keep poking
// forever. Once any response lands, retry stops and refresh takes over.
const maxCalibrationRetries = 5

// calibrationRetryInterval is the gap between retry attempts. Chosen to
// be long enough that we don't hammer a struggling proxy, short enough
// that the badge converges quickly once the proxy is healthy.
const calibrationRetryInterval = 5 * time.Second

// calibrationRefreshInterval is the gap between periodic refreshes after
// the initial fetch succeeds. 30s is short enough that a user who runs
// `atlas model install-artifacts` and restarts the lens container sees
// the badge flip to green within one refresh tick, long enough that the
// HTTP cost is trivial (120 calls/hour). Cap-free: the refresh runs for
// the lifetime of the TUI session.
const calibrationRefreshInterval = 30 * time.Second

// fetchCalibrationStatusCmd does an HTTP GET against the proxy. Fast — the
// proxy itself caches nothing here, but the upstream lens /health call is
// 3s-bounded and the rest is in-process. Total round-trip should be <4s.
// Fail-soft: on error we set status=nil and the badge falls back to a
// "not yet probed" placeholder rather than blocking startup.
func fetchCalibrationStatusCmd(proxyURL string) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, "GET",
			strings.TrimRight(proxyURL, "/")+"/v1/calibration/status", nil)
		if err != nil {
			return calibrationStatusMsg{err: err}
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return calibrationStatusMsg{err: err}
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return calibrationStatusMsg{err: nil, status: nil}
		}
		var s calibrationStatus
		if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
			return calibrationStatusMsg{err: err}
		}
		return calibrationStatusMsg{status: &s}
	}
}

// Lipgloss styles for the three verdict states. Colors picked from the
// same palette as the rest of the TUI (panes.go's titleStyle reads from
// 117 / blue; we add green/yellow/red for verdict states).
var (
	badgeOK = lipgloss.NewStyle().
		Foreground(lipgloss.Color("78")). // green
		Bold(true)
	badgeWarn = lipgloss.NewStyle().
			Foreground(lipgloss.Color("214")). // amber
			Bold(true)
	badgeFail = lipgloss.NewStyle().
			Foreground(lipgloss.Color("203")). // red
			Bold(true)
	badgeDim = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245")) // grey
)

// renderCalibrationBadge produces the compact badge text that gets
// appended to the Pipeline pane title.
//
//	"  Lens ✓  ASA ✓"                                      — both supported
//	"  Lens ⚠  ASA ⚠  → atlas lens build · PUBLISHING.md"   — needs attention
//	"  cal …"                                              — fetch in flight / failed
//
// When either verdict is non-supported, an inline actionable hint is
// appended pointing the user at the relevant build command + the docs
// section that walks through the full contribution flow. The hint lives
// on the same line as the badge so we don't have to bump the pipeline
// pane height when calibration is in a warn/fail state.
//
// Returns empty string only when the proxy is reachable but returned
// nothing meaningful — better to omit than render a confusing placeholder.
func renderCalibrationBadge(s *calibrationStatus) string {
	if s == nil {
		return badgeDim.Render("  cal …")
	}
	badge := "  " + renderOneBadge("Lens", s.Lens.Verdict) +
		"  " + renderOneBadge("ASA", s.ASA.Verdict)
	if hint := badgeActionHint(s); hint != "" {
		badge += "  " + badgeDim.Render(hint)
	}
	return badge
}

func renderOneBadge(name, verdict string) string {
	switch verdict {
	case "supported":
		return badgeOK.Render(name + " ✓")
	case "no-artifacts", "incomplete-artifacts", "uncalibrated", "missing", "dim-mismatch", "unverified":
		return badgeWarn.Render(name + " ⚠")
	case "unreachable", "incompatible":
		return badgeFail.Render(name + " ✗")
	default:
		return badgeDim.Render(name + " ?")
	}
}

// badgeActionHint returns the one-line "what should I do about this"
// pointer that gets rendered right next to the badge when either
// subsystem is in a non-supported state. Empty when both are happy.
//
// Suppresses on "unreachable" / "incompatible" because those mean
// services are down, not that the artifact is wrong — a "build" hint
// would be misleading there.
func badgeActionHint(s *calibrationStatus) string {
	lensWarn := s.Lens.Verdict == "no-artifacts" ||
		s.Lens.Verdict == "incomplete-artifacts" ||
		s.Lens.Verdict == "uncalibrated" ||
		s.Lens.Verdict == "dim-mismatch"
	asaWarn := s.ASA.Verdict == "missing"
	asaUnverified := s.ASA.Verdict == "unverified"
	switch {
	case lensWarn && asaWarn:
		return "→ atlas lens build / atlas asa build · docs/PUBLISHING.md"
	case lensWarn:
		return "→ atlas lens build · docs/PUBLISHING.md"
	case asaWarn:
		return "→ atlas asa build · docs/PUBLISHING.md"
	case asaUnverified:
		return "→ atlas asa check"
	}
	return ""
}
