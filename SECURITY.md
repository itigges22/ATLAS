# Security Policy

## Scope and threat model

ATLAS is a local, single-user tool. All services (proxy, TUI, v3-service, llama.cpp inference, sandbox) run as containers or processes on the operator's own machine and are intended to be reachable only from that machine. There is no hosted component, no multi-tenant deployment target, and no authentication layer between the local services.

Within that model:

- **In scope:** command injection or path traversal that lets model output escape the workspace or sandbox; container escapes from the sandbox; unverified downloads of model artifacts; secrets written to disk or logs; vulnerabilities in the install scripts.
- **Out of scope:** attacks that require the ATLAS ports to be exposed to an untrusted network (the compose files bind to localhost by default), multi-tenant isolation between users on the same machine, and prompt-injection making the model produce bad *code* (the sandbox and diff review exist so the operator can catch that before it lands).

Model-generated tool calls are treated as untrusted input: file edits are constrained to the workspace and shell commands run in the sandbox container (read-only rootfs, no-new-privileges, pids limit, `/workspace` as the only writable host mount). If you find a way around either boundary, that is exactly the kind of report we want.

Two current limits of that boundary, so reports can be calibrated against what is actually enforced: the sandbox has outbound network access (toolchains need to fetch dependencies), and its memory cap comes from `ATLAS_SANDBOX_MEM`, which `atlas init` writes — a raw `docker compose up` without the wizard runs the sandbox uncapped.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 3.1.x   | Yes       |
| < 3.1   | No        |

## Reporting a vulnerability

Please report vulnerabilities privately via [GitHub Security Advisories](https://github.com/itigges22/ATLAS/security/advisories/new) rather than opening a public issue.

Include what you can of: the affected component (proxy, TUI, CLI, v3-service, sandbox, install scripts), reproduction steps, and the impact under the single-user local model above.

You can expect an acknowledgment within a week. Fixes for confirmed vulnerabilities land on `dev` and are promoted to a release as soon as they are validated; credit is given in the changelog unless you ask otherwise.

If GitHub advisories are unavailable to you, open a minimal public issue saying "security — need a private channel" **without details**, and the maintainer will provide one.

## Severity and response targets

This is a single-maintainer project; targets are best-effort but taken seriously.

| Severity | Definition (in this trust model) | Acknowledge | Fix target |
| --- | --- | --- | --- |
| Critical | Sandbox/workspace escape reachable from model output; RCE via a verified artifact path; credential exfiltration from a default install | 48h | Patch release ASAP, days not weeks |
| High | Same classes requiring non-default config, or integrity bypass of artifact verification | 72h | Next release, ≤30 days |
| Medium | Hardening gaps with real but bounded impact (e.g. a resource-exhaustion vector) | 1 week | Scheduled release |
| Low | Defense-in-depth improvements, doc corrections | 1 week | Backlog with issue |

## Disclosure and embargo

Confirmed Critical/High reports stay embargoed until a fixed release is published, targeted at ≤90 days from confirmation. The fix may land on `dev` with a neutral commit message during embargo. Reporters are consulted on the advisory text and credited. CVE IDs are requested through GitHub's advisory CNA flow when a report warrants one.

## Backports and releases

Security fixes are released for the current release (N) and backported to N−1 within its 90-day window (see `SUPPORT_MATRIX.md`). A security release follows the normal release pipeline (immutable `sha-*` images, test-gated tag promotion) plus a changelog entry marking it security-relevant and an advisory publication.

## Artifact revocation

If a published model, Lens, or ASA artifact turns out to be bad (malware, corruption, wrong calibration, license problem):

1. The artifact is removed or replaced at its HF location.
2. The registry's pinned SHA-256 for it is updated or the entry's status is downgraded in a patch release — installers verify hashes, so already-published pins stop matching the bad artifact and refuse to install it.
3. The changelog and a pinned issue name the affected hashes so users can check installed files with `atlas model verify` / `atlas doctor`.
