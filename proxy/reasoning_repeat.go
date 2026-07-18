package main

// Reasoning-repetition detector. May 10 2026 BiasBusters follow-up #30.
//
// Sibling to:
//   - tool_repeat.go (structural call-shape repetition)
//   - lens_score.go's agentLensRegression (semantic content quality)
//
// The pattern this catches: the model emits reasoning_content that
// rehashes the same opening prose across consecutive turns ("Now I
// need to look at the file" / "Let me check the file" / similar)
// without committing to action. Tool-call repetition won't fire
// because the eventual tool calls may differ; lens scoring won't fire
// because the LANDED content (write_file/edit_file output) may be
// fine. The bug is in the THINKING, not in the action or the artifact.
//
// Detection: normalize the first reasoningSnippetLen chars of each
// turn's reasoning_content (lowercase + collapsed whitespace), compare
// to the previous turn's snippet. Identical snippets across
// reasoningRepeatThreshold consecutive turns triggers an intervention.
//
// Why prefix-match instead of full-text similarity: prose preambles
// rehash strongly at the opening ("Now I need to..." dominates the
// reasoning even when later sentences vary). Catching the OPENING is
// what distinguishes "model is stuck" from "model is on a different
// thought now." Embedding-based similarity (cosine over reasoning
// embeddings) is a future refinement; prefix-match handles the
// dominant repetition shape we've actually seen in user logs.

import (
	"fmt"
	"strings"
)

const (
	// reasoningRepeatThreshold is the number of consecutive identical
	// snippets that fires intervention. 2 = the SECOND consecutive
	// repetition (i.e. three turns total with the same opening). 1
	// would be the first repetition, which is too eager — the model
	// may have legitimately needed a second pass on the same topic.
	reasoningRepeatThreshold = 2

	// reasoningSnippetLen is the prefix length used for similarity
	// comparison. 80 chars is roughly the first 1-2 sentences of
	// the model's reasoning preamble, which is where the rehash pattern
	// shows. Longer would over-match (later sentences naturally vary
	// even within a stuck loop); shorter would under-match (the model's
	// boilerplate openings collide on unrelated tasks).
	reasoningSnippetLen = 80
)

// recordReasoning updates ctx with the current turn's reasoning
// snippet and returns the corrective message + true when the same
// snippet has appeared reasoningRepeatThreshold consecutive times.
// Returns ("", false) otherwise. Empty reasoning resets the counter
// (the detector only flags STUCK thinking, not absence of thinking).
//
// Caller should reset ctx.ConsecutiveReasoningRepeats and
// ctx.LastReasoningSnippet to "" after acting on the corrective so
// the same loop doesn't re-fire on the next iteration.
func recordReasoning(ctx *AgentContext, reasoning string) (string, bool) {
	snippet := normalizeReasoningSnippet(reasoning)
	if snippet == "" {
		// No reasoning emitted (or pure whitespace) — break the streak.
		// A single reasoning-free turn means the model committed to
		// action without preamble, which is exactly what we want to
		// reward, not flag as a continuation.
		ctx.ConsecutiveReasoningRepeats = 0
		ctx.LastReasoningSnippet = ""
		return "", false
	}

	if ctx.LastReasoningSnippet != "" && snippet == ctx.LastReasoningSnippet {
		ctx.ConsecutiveReasoningRepeats++
	} else {
		ctx.ConsecutiveReasoningRepeats = 0
	}
	ctx.LastReasoningSnippet = snippet

	if ctx.ConsecutiveReasoningRepeats < reasoningRepeatThreshold {
		return "", false
	}

	return fmt.Sprintf(
		"⚠ Reasoning repetition detected: your reasoning has opened with the same prose for %d consecutive turns "+
			"(\"%s...\"). The same opening won't change the outcome. Either commit to the next action — emit a "+
			"different tool call, run a verification command, or emit `done` if the task is complete — OR change "+
			"the investigation direction (read a different file, try a different selector, ask the user for "+
			"clarification). Don't rephrase the same thought.",
		ctx.ConsecutiveReasoningRepeats+1,
		truncateForCorrective(reasoning, 60),
	), true
}

// normalizeReasoningSnippet lowercases, collapses whitespace, and
// truncates to reasoningSnippetLen so similar openings compare equal
// across minor formatting differences. Empty input → empty output.
func normalizeReasoningSnippet(reasoning string) string {
	s := strings.TrimSpace(reasoning)
	if s == "" {
		return ""
	}
	s = strings.ToLower(s)
	// Collapse all whitespace runs to a single space.
	var b strings.Builder
	prevSpace := false
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	out := strings.TrimSpace(b.String())
	if len(out) > reasoningSnippetLen {
		out = out[:reasoningSnippetLen]
	}
	return out
}

// truncateForCorrective returns the first n runes of s, suitable for
// embedding in a model-facing corrective string (preserves UTF-8).
func truncateForCorrective(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	// Byte-truncate then trim back to a rune boundary.
	cut := s[:n]
	if !strings.ContainsRune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:'-", rune(cut[len(cut)-1])) {
		// Lazy — just take whole bytes; if model preamble has Unicode
		// punctuation right at byte n we may chop a multi-byte char.
		// Safe-but-lossy: trim one extra byte.
		if len(cut) > 1 {
			cut = cut[:len(cut)-1]
		}
	}
	return cut
}
