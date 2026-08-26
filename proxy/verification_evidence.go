package main

import (
	"sort"
	"strings"
)

// Evidence that a command the CLIENT required ran, and what it ran against.
//
// This is the second producer and the first that can reach behavioral
// strength. Its whole claim is narrow: a command the validated request named,
// executed exactly as the client wrote it through the path that already runs
// commands, exiting successfully against a workspace that held these exact
// candidate bytes.
//
// It executes nothing. Running a command is the existing tool path's job --
// with its own safety checks, its own permission endpoint and its own sandbox
// -- and a producer that ran commands of its own would be a second execution
// route with a second set of rules. What it consumes is the record that path
// already writes: one entry per GREEN run, naming the command and the sha256
// of every session-written file that run actually named.
//
// Three things it deliberately cannot do:
//
//   - upgrade a model's command. Authority comes from the DECLARATION, not
//     from the text. A command the client never declared has no obligation, so
//     there is nothing for a matching string to attach to.
//   - prove something about an unrelated artifact. A run that named nothing
//     covers nothing, and one that named other files covers those.
//   - close anything but the one command it is about. Two declared commands
//     are two obligations; one passing leaves the other owed.
//
// Nothing consumes what this produces.

// declaredCommandOutcome is how the run ended, in a closed vocabulary. Only
// one value may support an authorizing observation, and every other way a
// command can end has its own name rather than collapsing into "not ok" --
// a refusal and a timeout are different facts about different failures.
type declaredCommandOutcome string

const (
	// commandExitedZero: ran in the foreground and exited successfully under
	// the semantics the tool path already applies.
	commandExitedZero declaredCommandOutcome = "exited_zero"
	// commandExitedNonZero: ran and failed.
	commandExitedNonZero declaredCommandOutcome = "exited_nonzero"
	// commandRefused: a safety check or the permission endpoint declined it.
	commandRefused declaredCommandOutcome = "refused"
	// commandAltered: what ran was not what the client wrote.
	commandAltered   declaredCommandOutcome = "altered"
	commandTimedOut  declaredCommandOutcome = "timed_out"
	commandCancelled declaredCommandOutcome = "cancelled"
	// commandBackgrounded: still running, so it has not exited at all.
	commandBackgrounded declaredCommandOutcome = "backgrounded"
	// commandOutcomeUnknown: nothing recorded how it ended. Never a default
	// that grants anything.
	commandOutcomeUnknown declaredCommandOutcome = "unknown"
)

var declaredCommandOutcomes = map[declaredCommandOutcome]bool{
	commandExitedZero: true, commandExitedNonZero: true, commandRefused: true,
	commandAltered: true, commandTimedOut: true, commandCancelled: true,
	commandBackgrounded: true, commandOutcomeUnknown: true,
}

// verificationEvidenceRequest is one declared command, one run of it, and the
// candidate it is claimed to be about.
type verificationEvidenceRequest struct {
	// Obligation must be a declared-command obligation whose subject is the
	// exact command string the client wrote.
	Obligation taskObligation
	// Record is what the existing tool path wrote when the command ran. It is
	// consumed, never synthesised here.
	Record VerificationRecord
	// Outcome is how that run ended.
	Outcome declaredCommandOutcome

	// CandidatePath and CandidateHash name the artifact under evaluation. The
	// run must have covered that path at exactly those bytes.
	CandidatePath string
	CandidateHash string

	InvocationID        string
	CandidateInstanceID string
	BaselineIdentity    string
}

// produceDeclaredVerificationEvidence is THE client-declared verification
// producer.
//
// Every condition below is necessary, and each is a fact about THIS run rather
// than a property of the command's shape. Nothing here parses a command,
// infers its purpose, or recognises a test runner: the client said what it
// required, and this reports whether that exact thing ran and passed against
// these exact bytes.
func produceDeclaredVerificationEvidence(ctx *AgentContext, req verificationEvidenceRequest) (proxyEvidence, bool) {
	if ctx == nil {
		return proxyEvidence{}, false
	}
	if ctx.Ctx != nil && ctx.Ctx.Err() != nil {
		return proxyEvidence{}, false
	}
	if req.Obligation.Kind != ObligationDeclaredCommand {
		return proxyEvidence{}, false
	}
	if req.Obligation.RequiredStrength != "behavioral" {
		return proxyEvidence{}, false
	}
	if !declaredCommandOutcomes[req.Outcome] {
		return proxyEvidence{}, false
	}

	command := req.Obligation.Subject
	// The client's own list, re-read from the validated request. A command the
	// caller never declared has no authority here whoever typed it, and a
	// caller that stated no knowledge declared nothing at all.
	if !requestDeclaredCommand(ctx, command) {
		return proxyEvidence{}, false
	}
	// What ran must be what the client wrote, byte for byte. A command that
	// differs by a flag, a path or a space is a different command.
	if req.Record.Command != command {
		return proxyEvidence{}, false
	}

	requestID := requestIDOf(ctx)
	if strings.TrimSpace(requestID) == "" ||
		strings.TrimSpace(req.InvocationID) == "" ||
		strings.TrimSpace(req.CandidateInstanceID) == "" ||
		strings.TrimSpace(req.CandidatePath) == "" ||
		strings.TrimSpace(req.CandidateHash) == "" {
		return proxyEvidence{}, false
	}

	// The run must still describe the bytes on disk, and among them it must
	// have covered the candidate at exactly the hash being asked about. A
	// command that passed before the candidate was inserted covered the older
	// bytes and fails here rather than transferring.
	covered, current := evidenceIsCurrent(ctx, req.Record)
	if !current {
		return proxyEvidence{}, false
	}
	if covered[resolveAgentPath(ctx, req.CandidatePath)] != req.CandidateHash {
		return proxyEvidence{}, false
	}

	generation, stateHash := workspaceIdentity(ctx)
	p := V3EvidenceProvenance{
		Source:              ProvenanceClientDeclaredVerification,
		RequestID:           requestID,
		InvocationID:        req.InvocationID,
		CandidateInstanceID: req.CandidateInstanceID,
		CandidateHash:       req.CandidateHash,
		WorkspaceGeneration: generation,
		WorkspaceStateHash:  stateHash,
		// The command is named by hash, like every other subject: a command
		// string in an operator log is a content leak.
		CommandIdentity:  contentSHA256(command),
		BaselineIdentity: req.BaselineIdentity,
		ObligationID:     req.Obligation.ID,
		RequiredStrength: req.Obligation.RequiredStrength,
		// A successful declared command demonstrates that the thing the client
		// asked to be run ran and succeeded. It is behavioral. It is NOT an
		// oracle: nothing here compared an answer with a reference, and
		// labelling an arbitrary exit-zero command oracle evidence is how "it
		// ran" becomes "it is right".
		ObservedStrength: "behavioral",
	}
	outcome := ValidationFailed
	if req.Outcome == commandExitedZero {
		outcome = ValidationPassed
	}
	return proxyEvidence{Provenance: p, Outcome: outcome}, true
}

// requestDeclaredCommand reports whether the validated request named this
// exact command. It reads the same decision the completion gate reads, so the
// two cannot disagree about what the client asked for, and it requires stated
// knowledge: a caller that declared nothing declared not-this-command.
func requestDeclaredCommand(ctx *AgentContext, command string) bool {
	if strings.TrimSpace(command) == "" {
		return false
	}
	decision := resolveVerificationObligation(ctx)
	if !decision.KnowledgeSpecified {
		return false
	}
	for _, want := range decision.Items {
		if want == command {
			return true
		}
	}
	return false
}

// declaredVerificationCoverage reports which declared-command obligations have
// an authorizing observation and which are still owed.
//
// Independence is the point: two declared commands are two obligations, and
// one passing says nothing about the other. Missing is returned sorted so a
// caller reporting it does not depend on map order.
func declaredVerificationCoverage(obligations []taskObligation,
	evidence []proxyEvidence) (met []string, missing []string) {
	authorized := map[string]bool{}
	for _, e := range evidence {
		if e.Provenance.Source != ProvenanceClientDeclaredVerification {
			continue
		}
		if ok, _ := e.Authorizes(); !ok {
			continue
		}
		authorized[e.Provenance.ObligationID] = true
	}
	for _, o := range obligations {
		if o.Kind != ObligationDeclaredCommand || !o.Required {
			continue
		}
		if authorized[o.ID] {
			met = append(met, o.ID)
		} else {
			missing = append(missing, o.ID)
		}
	}
	sort.Strings(met)
	sort.Strings(missing)
	return met, missing
}
