// PC-062: TUI state machine — converts incoming Envelope events into
// derived UI state (pipeline progress, counters, current stage).
//
// State is a pure function of the event sequence: replaying the same
// events in the same order produces the same state. This keeps the
// model deterministic for tests and makes the renderer trivial.

package main

import (
	"sort"
	"time"
)

// stageStatus tracks one logical pipeline stage's lifecycle.
type stageStatus struct {
	Name      string
	StartedAt time.Time
	EndedAt   time.Time // zero if still running
	Success   bool      // valid only when EndedAt is non-zero
	Detail    string    // most recent payload.detail
}

func (s stageStatus) Running() bool { return s.EndedAt.IsZero() }

func (s stageStatus) Duration() time.Duration {
	if s.EndedAt.IsZero() {
		return time.Since(s.StartedAt)
	}
	return s.EndedAt.Sub(s.StartedAt)
}

// pipelineState aggregates stages from the event stream.
type pipelineState struct {
	// Insertion order — stages appear in the pipeline pane top-to-bottom
	// in the order they first started, so users see a chronological view
	// rather than alphabetic.
	order  []string
	byName map[string]*stageStatus

	// Counters surfaced in the stats pane.
	totalEvents   int
	toolCalls     int
	toolSuccesses int
	toolFailures  int
	errors        int
	currentTurn   int

	// Final state.
	done        bool
	doneSuccess bool
	totalMS     int64
	doneSummary string
}

func newPipelineState() pipelineState {
	return pipelineState{byName: map[string]*stageStatus{}}
}

// apply mutates p with the effect of one envelope. Pure function: same
// envelope sequence → same state.
func (p *pipelineState) apply(ev Envelope) {
	p.totalEvents++

	switch ev.Type {
	case EvtStageStart:
		ts := envTime(ev)
		if existing, ok := p.byName[ev.Stage]; ok {
			// Re-entry into a logical stage (e.g. probe_retry). Reset
			// timing — current run started now, prior result discarded
			// from the visible view.
			existing.StartedAt = ts
			existing.EndedAt = time.Time{}
			existing.Detail = payloadString(ev.Payload, "detail")
		} else {
			p.byName[ev.Stage] = &stageStatus{
				Name:      ev.Stage,
				StartedAt: ts,
				Detail:    payloadString(ev.Payload, "detail"),
			}
			p.order = append(p.order, ev.Stage)
		}

	case EvtStageEnd:
		ts := envTime(ev)
		s, ok := p.byName[ev.Stage]
		if !ok {
			// stage_end without a matching start — synthesize an entry
			// so the pane shows it. Doesn't pretend the duration is
			// meaningful in that case.
			s = &stageStatus{Name: ev.Stage, StartedAt: ts}
			p.byName[ev.Stage] = s
			p.order = append(p.order, ev.Stage)
		}
		s.EndedAt = ts
		s.Success = boolField(ev.Payload, "success")
		if d := payloadString(ev.Payload, "detail"); d != "" {
			s.Detail = d
		}

	case EvtToolCall:
		p.toolCalls++
		if turn, ok := ev.Payload["turn"].(float64); ok {
			p.currentTurn = int(turn)
		}

	case EvtToolResult:
		if boolField(ev.Payload, "success") {
			p.toolSuccesses++
		} else {
			p.toolFailures++
		}

	case EvtError:
		p.errors++

	case EvtDone:
		p.done = true
		p.doneSuccess = boolField(ev.Payload, "success")
		if v, ok := ev.Payload["total_duration_ms"].(float64); ok {
			p.totalMS = int64(v)
		}
		p.doneSummary = payloadString(ev.Payload, "summary")
	}
}

// stages returns the ordered list of stages for the pipeline pane.
func (p *pipelineState) stages() []*stageStatus {
	out := make([]*stageStatus, 0, len(p.order))
	for _, name := range p.order {
		out = append(out, p.byName[name])
	}
	return out
}

// activeStage returns the first stage currently running, or nil if
// none are active. Used in the stats line.
func (p *pipelineState) activeStage() *stageStatus {
	// Most recent stage_start without a matching stage_end.
	candidates := make([]*stageStatus, 0, len(p.order))
	for _, name := range p.order {
		if s := p.byName[name]; s.Running() {
			candidates = append(candidates, s)
		}
	}
	if len(candidates) == 0 {
		return nil
	}
	// Sort by StartedAt desc — innermost (most recently started) wins.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].StartedAt.After(candidates[j].StartedAt)
	})
	return candidates[0]
}

func envTime(ev Envelope) time.Time {
	return time.Unix(0, int64(ev.Timestamp*1e9))
}

func boolField(p map[string]interface{}, key string) bool {
	v, ok := p[key]
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
}

func payloadString(p map[string]interface{}, key string) string {
	v, ok := p[key]
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}
