# Release Contract & Verification

This page defines what ATLAS supports, how each capability is gated, and the
verification levels a capability must clear before it is called Supported.
Roadmap items are tracked in GitHub and are not release claims.

## Status definitions

Status terms follow the support-level taxonomy defined in
[SUPPORT_MATRIX.md](../SUPPORT_MATRIX.md) (Supported, Preview, Experimental,
Community-tested, Research-only, Unsupported, Roadmap). One term used here
marks *audience* rather than maturity and composes with a level:

- **Internal:** service-to-service contract, not a public client API.

## User-facing capabilities

| Capability | Status | Minimum verification level |
|---|---|---|
| Python CLI installation and command dispatch | Supported | Hermetic and install matrix |
| TUI chat, file view, pipeline view, cancellation, and feedback | Supported | Hermetic Go race tests and local integration |
| Proxy `/v1/agent`, `/events`, `/cancel`, health, readiness, and model listing | Supported | Hermetic Go race tests and local integration |
| Proxy OpenAI chat-completions passthrough | Supported | Local integration |
| Workspace file tools and sandboxed command verification | Supported | Hermetic policy tests and container integration |
| V3 candidate generation and selection for Python | Supported | Hermetic unit tests and hardware integration |
| V3 verification for non-Python syntax/toolchain checks | Supported | Hermetic unit tests and sandbox integration |
| V3 project build-command verification | Experimental | Hermetic overlay tests plus container integration |
| Model registry list, recommend, install, remove, and verify | Supported | Hermetic CLI tests and hardware integration for inference |
| Lens compatibility check, build, and retrain | Supported for registry entries with compatible artifacts | Hermetic tests and hardware integration |
| Lens and ASA artifact publishing | Experimental | Hermetic CLI tests plus maintainer review workflow |
| ASA compatibility check and build | Experimental | Hermetic tests and hardware integration |
| CUDA backend | Supported | Hardware integration (maintainer hardware) |
| ROCm backend | Community-tested | Community hardware validation (SUPPORT_MATRIX § inference backends) |
| Apple Metal backend | Supported | Maintainer hardware (macOS hybrid) |
| Vulkan backend | Preview | Smoke-tested (lavapipe boot path); no real-GPU validation yet |
| Intel SYCL and multi-GPU backends | Roadmap | None until implemented |
| Browser or visual verification | Roadmap | None until implemented |

## Service contracts

| Service surface | Status | Notes |
|---|---|---|
| Sandbox health, languages, execute, syntax-check, shell, and background jobs | Internal | Called by proxy and V3; direct host use is a developer workflow |
| V3 generate, run, plan, and health | Internal | `/v3/generate` is the proxy integration path |
| V3 structural edit, symbol index, and complexity endpoints | Experimental (Internal) | Tree-sitter availability determines capability |
| Geometric Lens `/v1/*` endpoints | Supported (requires configured local API keys) | Local authenticated API |
| Geometric Lens `/internal/*` endpoints | Internal | Intended only for the ATLAS service network |
| llama-server inference, completion, embedding, and health | Internal (upstream llama.cpp contract, qualified against the pinned revision) | — |

A feature is not promoted to Supported until its required verification level is
automated and passing on representative hardware where applicable.

## Verification

ATLAS separates checks by whether they run on a normal development machine or
require containers, a model, or specific hardware.

### Developer gate

Run the default gate from the repository root:

```bash
python scripts/production-readiness.py
```

The required checks cover test integrity, Python compilation and unit tests, and
Go race tests and vet for the proxy and TUI. They do not require a GPU, model
download, or running ATLAS services. The developer gate also includes contract
tests for V3 language-aware syntax verification and sandbox overlay behavior.
Full project build-command qualification still belongs to the container and
release levels because it depends on the selected project's dependencies and
toolchain state.

Optional checks run when their tools are installed. Missing optional tools are
reported as `unavailable`, not as successful checks. A missing tool becomes a
failure when its gate is selected explicitly:

```bash
python scripts/production-readiness.py --only ruff
python scripts/production-readiness.py --only compose
```

Use `--list` to see the available gates and `--json` for machine-readable
results. CI runs the same named gates after installing their dependencies.

### Verification levels

| Level | Purpose | Hardware or services |
|---|---|---|
| Hermetic | Unit, static, race, and configuration checks | No GPU, model, or running services |
| Local integration | HTTP, SSE, cancellation, and process lifecycle | Locally built binaries; no model where possible |
| Container integration | Compose networking, health, filesystem mounts, and sandbox behavior | Docker |
| Hardware integration | Real inference, embeddings, Lens compatibility, and accelerator behavior | Supported accelerator and registry model |
| Release qualification | Clean install plus all applicable levels and artifact checks | Declared release hardware matrix |

Hardware-dependent checks must name the model and accelerator used. For the
canonical Apple Silicon path, use the registry entry selected by `atlas model
recommend`; release qualification should record the exact registry name, GGUF
hash status, backend, context size, and service image digests.

### Skip policy

- A required dependency missing from a selected gate is a failure.
- An optional dependency missing from the default developer gate is
  `unavailable`.
- A hardware test skipped because the required hardware is absent is
  `unavailable`; it does not count as a pass.
- A supported release cannot be qualified while a required release gate is
  failed or unavailable.

## Release checklist

- Bump the "Applies to" line at the top of
  [SUPPORT_MATRIX.md](../SUPPORT_MATRIX.md) to the release version.
- Qualify the release per the verification levels above.
- Cut and push the signed release tag (next section).

## Signed release tags

Release tags (`vX.Y.Z`) are SSH-signed so their provenance is
verifiable, complementing the keyless **cosign** signatures on the
published image digests (build-images.yml). The two cover different
artifacts: cosign signs the container images, tag signing signs the
git release point.

Cut a release tag with the helper (it signs, verifies, and prints the
push command — it never pushes):

```bash
scripts/release-tag.sh v1.2.0 "release notes"
git push origin v1.2.0        # the deliberate release step
```

On push, `.github/workflows/verify-tags.yml` re-verifies the signature
against `.github/allowed_signers` and fails the tag if it is unsigned or
signed by an unlisted key.

### One-time signing setup (per maintainer machine)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/atlas_release_signing -C "you@example"
git config gpg.format ssh
git config user.signingkey ~/.ssh/atlas_release_signing.pub
git config gpg.ssh.allowedSignersFile .github/allowed_signers
# append "<your-git-email> <contents of atlas_release_signing.pub>" to
# .github/allowed_signers and commit it, then register the key so GitHub
# shows tags as Verified:
gh ssh-key add ~/.ssh/atlas_release_signing.pub --type signing \
    --title "atlas release signing"
```

Verify any release tag locally:

```bash
git config gpg.ssh.allowedSignersFile .github/allowed_signers
git verify-tag v1.2.0
```

**Status:** the signing key, git config, release script, CI verification,
and allowed-signers file are in place and produce a verified signed tag
today. The remaining step to get GitHub's green **Verified** badge is
registering the public key on the maintainer's GitHub account (the
`gh ssh-key add --type signing` line above) — an account action left to
the maintainer.
