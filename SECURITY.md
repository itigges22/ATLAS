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
