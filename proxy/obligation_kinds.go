package main

import (
	"crypto/sha256"
	"encoding/hex"
	"path/filepath"
	"sort"
	"strings"
)

// What a task obliges, as a closed typed vocabulary.
//
// obligations.go decided WHOSE list a run owes -- the caller's or the
// heuristic's. It says nothing about what sort of thing each item is or how
// strongly it must be shown, so a path and a command arrived as the same
// untyped string and any evidence about either looked equally good.
//
// This file names the sorts and their floors. Three things stay separate:
//
//	KIND      what sort of thing is owed (a closed set)
//	STRENGTH  how strong the evidence closing it must be
//	SUBJECT   the exact thing it is owed about, carried as a hash
//
// The subject is a hash and never the text. A declared verification command
// is a subject, and a command string in an operator log is a content leak; a
// uniform rule that never carries text cannot leak one by exception.
//
// The same vocabulary exists in v3-service/obligations.py. Two copies of a
// closed set is a divergence waiting to happen, so a contract test parses
// both and fails when they disagree.

const (
	// ObligationArtifactExists: the artifact the client named must exist at
	// its canonical path.
	ObligationArtifactExists = "artifact_exists"
	// ObligationSyntacticValidity: its bytes must satisfy the structural
	// check the proxy already owns for that artifact class -- the same gate
	// live completion policy requires, not a second extension table.
	ObligationSyntacticValidity = "syntactic_validity"
	// ObligationDeclaredCommand: one exact command the client required, run
	// as the client wrote it.
	ObligationDeclaredCommand = "declared_command"
	// ObligationDeclaredExample: one example or oracle case the client
	// stated, with its own expected answer.
	ObligationDeclaredExample = "declared_example"
	// ObligationBaselinePreserved: an artifact that already carries a current
	// passing verdict must not be replaced by something demonstrated less
	// well.
	ObligationBaselinePreserved = "baseline_preserved"
	// ObligationUnsupported: something is owed and nothing here can name
	// what. Never satisfiable, never vacuously complete.
	ObligationUnsupported = "unsupported"
)

// obligationKinds is the closed set, in declaration order.
var obligationKinds = []string{
	ObligationArtifactExists,
	ObligationSyntacticValidity,
	ObligationDeclaredCommand,
	ObligationDeclaredExample,
	ObligationBaselinePreserved,
	ObligationUnsupported,
}

// obligationKindRequiredStrength is how strong the evidence closing each kind
// must be.
//
// A declared command is behavioral, not oracle: exit zero says the command the
// client asked for ran and succeeded against these bytes. It does not say the
// answer was checked against a reference, and calling an arbitrary exit-zero
// command an oracle is how "it ran" became "it is right".
//
// ObligationBaselinePreserved is absent because its floor is not a constant:
// it is whatever the evidence currently describing that baseline already
// reached. ObligationUnsupported is absent because no strength closes it.
var obligationKindRequiredStrength = map[string]string{
	ObligationArtifactExists:    "syntax",
	ObligationSyntacticValidity: "syntax",
	ObligationDeclaredCommand:   "behavioral",
	ObligationDeclaredExample:   "oracle",
}

// obligationDynamicStrengthKinds take their floor from the obligation rather
// than from the kind.
var obligationDynamicStrengthKinds = map[string]bool{
	ObligationBaselinePreserved: true,
}

// obligationUnsatisfiableKinds can be closed by nothing.
var obligationUnsatisfiableKinds = map[string]bool{
	ObligationUnsupported: true,
}

func knownObligationKind(kind string) bool {
	for _, k := range obligationKinds {
		if k == kind {
			return true
		}
	}
	return false
}

// obligationID is the canonical name of one obligation.
//
// Deterministic and content-free: the subject is hashed, so the id can be
// logged, compared and carried on the wire without ever holding a path's
// contents or a command's text. v3-service computes the same string.
func obligationID(kind, subject string) (string, bool) {
	if !knownObligationKind(kind) || strings.TrimSpace(subject) == "" {
		return "", false
	}
	sum := sha256.Sum256([]byte(subject))
	return kind + ":" + hex.EncodeToString(sum[:])[:32], true
}

// obligationRequiredStrength is the floor evidence must reach to close this
// kind. A baseline obligation is at least as strong as the evidence already
// describing that baseline: replacing a file whose behaviour was demonstrated
// on the strength of a compile is a regression dressed as a delivery.
func obligationRequiredStrength(kind, baselineStrength string) (string, bool) {
	if !knownObligationKind(kind) || obligationUnsatisfiableKinds[kind] {
		return "", false
	}
	if obligationDynamicStrengthKinds[kind] {
		if strengthRank(baselineStrength) < 0 {
			return "", false
		}
		return baselineStrength, true
	}
	if baselineStrength != "" {
		// A fixed-strength kind cannot be raised by a baseline; silently
		// ignoring the argument is how a caller's mistake becomes a claim.
		return "", false
	}
	s, ok := obligationKindRequiredStrength[kind]
	return s, ok
}

// taskObligation is one thing the task owes, named by hash and floored by
// kind. Subject text never travels: Subject holds the canonical value only
// long enough for the producer that derived it, and is not serialised.
type taskObligation struct {
	ID   string
	Kind string
	// Subject is the canonical path or the exact command this obligation is
	// about. It is never logged and never leaves the process.
	Subject string
	// RequiredStrength is empty only for an unsatisfiable kind.
	RequiredStrength string
	Required         bool
}

// newTaskObligation fails closed: an unknown kind, an empty subject, or a
// baseline strength that does not name a real strength all produce nothing
// rather than an obligation nobody can measure.
func newTaskObligation(kind, subject, baselineStrength string, required bool) (taskObligation, bool) {
	id, ok := obligationID(kind, subject)
	if !ok {
		return taskObligation{}, false
	}
	if obligationUnsatisfiableKinds[kind] {
		return taskObligation{ID: id, Kind: kind, Subject: subject, Required: required}, true
	}
	strength, ok := obligationRequiredStrength(kind, baselineStrength)
	if !ok {
		return taskObligation{}, false
	}
	return taskObligation{
		ID: id, Kind: kind, Subject: subject,
		RequiredStrength: strength, Required: required,
	}, true
}

// obligationClosureFloor is the strongest floor any REQUIRED obligation
// demands. An unsupported required obligation makes the floor unreachable
// rather than absent: a record cannot close a task that owes something
// nothing measured.
func obligationClosureFloor(obs []taskObligation) string {
	floor := evidenceStrengthOrder[0]
	for _, o := range obs {
		if !o.Required {
			continue
		}
		if obligationUnsatisfiableKinds[o.Kind] {
			return evidenceStrengthOrder[len(evidenceStrengthOrder)-1]
		}
		if strengthRank(o.RequiredStrength) > strengthRank(floor) {
			floor = o.RequiredStrength
		}
	}
	return floor
}

// --- derivation from the validated request ----------------------------------
//
// Obligations are derived from what the CALLER declared, never from prose,
// a filename convention, a benchmark, or anything the model emitted. A caller
// that stated no knowledge produces no structured obligation at all: its
// prose heuristic may still govern proxy completion, and converting that into
// structured authority is exactly the promotion this file must not make.

// deriveTaskObligations is THE derivation. It reads the two decisions the
// request boundary already made and the ledger's own record of what is on
// disk; it consults no shadow state and no model output.
//
// Ordering is canonical -- kind, then id -- so two runs of the same request
// produce the same set in the same order and a diff between them is a real
// difference rather than map iteration.
func deriveTaskObligations(ctx *AgentContext, outputs, verification obligationDecision) []taskObligation {
	var out []taskObligation
	seen := map[string]bool{}
	add := func(kind, subject, baseline string) {
		o, ok := newTaskObligation(kind, subject, baseline, true)
		if !ok || seen[o.ID] {
			return
		}
		seen[o.ID] = true
		out = append(out, o)
	}

	if outputs.KnowledgeSpecified {
		for _, p := range outputs.Items {
			resolved := resolveAgentPath(ctx, p)
			if resolved == "" {
				continue
			}
			// Existence is owed for every declared output.
			add(ObligationArtifactExists, resolved, "")
			// Structural validity is owed only for artifact classes the
			// proxy's own gate already governs. There is no second table:
			// this is the same map live completion policy consults.
			if _, gated := syntaxGateLanguages[strings.ToLower(filepath.Ext(resolved))]; gated {
				add(ObligationSyntacticValidity, resolved, "")
			}
			// Replacing an artifact that already carries current evidence
			// owes at least what that evidence already reached.
			if s := baselineEvidenceStrength(ctx, resolved); s != "" {
				add(ObligationBaselinePreserved, resolved, s)
			}
		}
	}

	if verification.KnowledgeSpecified {
		for _, cmd := range verification.Items {
			if strings.TrimSpace(cmd) == "" {
				continue
			}
			add(ObligationDeclaredCommand, cmd, "")
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// baselineEvidenceStrength is how strongly the artifact ALREADY on disk is
// described, or "" when nothing current describes it.
//
// Read from evidence that already exists rather than from a new judgement:
// the ledger's verdict must be about the bytes that are there now, and a
// green verification record must still cover those exact bytes. A verdict
// about superseded bytes is history, not a baseline.
func baselineEvidenceStrength(ctx *AgentContext, resolved string) string {
	if ctx == nil {
		return ""
	}
	disk := fileSHA256(ctx, resolved)
	if disk == "" {
		return ""
	}
	// Behavioral: some green declared or recognised run still covers exactly
	// these bytes for this path.
	for _, rec := range ctx.VerificationEvidence {
		covered, ok := evidenceIsCurrent(ctx, rec)
		if !ok {
			continue
		}
		if covered[resolveAgentPath(ctx, resolved)] == disk {
			return "behavioral"
		}
	}
	// Syntax: the ledger holds a pass about exactly these bytes.
	ctx.LedgerMu.Lock()
	defer ctx.LedgerMu.Unlock()
	for key, d := range ctx.Ledger {
		if d == nil || d.Tombstoned {
			continue
		}
		if resolveAgentPath(ctx, key) != resolveAgentPath(ctx, resolved) {
			continue
		}
		if d.ValidationStatus == ValidationPassed && d.ValidatedHash == disk {
			return "syntax"
		}
	}
	return ""
}
