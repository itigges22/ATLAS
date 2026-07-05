package main

// Trust modes govern whether — and where — model-authored commands may
// execute. A newly-opened repository is untrusted content; running its
// build/test commands is a decision the operator makes explicitly.
//
//   untrusted     — no command execution at all (run_command refused).
//   trusted       — commands run in the isolated sandbox container
//                   (the default; host execution is downgraded to sandbox).
//   fully-trusted — advanced: host execution (ATLAS_VERIFY_IN=host) is
//                   honored, dropping the container backstop.
//
// Set via ATLAS_TRUST_MODE. The default is "trusted": commands run, but
// only in the sandbox. This keeps the out-of-box behavior safe (isolated
// execution) while making "run nothing" and "run on the host" both
// explicit, deliberate choices.

import (
	"os"
	"strings"
)

type trustMode string

const (
	trustUntrusted    trustMode = "untrusted"
	trustTrusted      trustMode = "trusted"
	trustFullyTrusted trustMode = "fully-trusted"
)

// resolveTrustMode reads ATLAS_TRUST_MODE, defaulting to trusted. An
// unrecognized value falls back to the safe default rather than failing
// open to host execution.
func resolveTrustMode() trustMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ATLAS_TRUST_MODE"))) {
	case "untrusted":
		return trustUntrusted
	case "fully-trusted", "fully_trusted":
		return trustFullyTrusted
	case "trusted", "":
		return trustTrusted
	default:
		return trustTrusted
	}
}

// commandsAllowed reports whether run_command may execute at all.
func (m trustMode) commandsAllowed() bool {
	return m != trustUntrusted
}

// hostExecutionAllowed reports whether host execution (bypassing the
// sandbox) is permitted. Only fully-trusted honors it; trusted downgrades
// a host request to sandbox execution so an ATLAS_VERIFY_IN=host setting
// can't quietly escalate below the intended trust level.
func (m trustMode) hostExecutionAllowed() bool {
	return m == trustFullyTrusted
}

// untrustedRefusal is the message returned when run_command is called
// under the untrusted mode.
const untrustedRefusal = "command execution is disabled: ATLAS_TRUST_MODE=untrusted. " +
	"This repository's commands are treated as untrusted content. Set " +
	"ATLAS_TRUST_MODE=trusted to run them in the isolated sandbox, or " +
	"fully-trusted to allow host execution."
