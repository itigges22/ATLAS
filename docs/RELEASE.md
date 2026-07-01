# Release Contract & Verification

This page defines what ATLAS supports, how each capability is gated, and the
verification levels a capability must clear before it is called Supported.
Roadmap items are tracked in GitHub and are not release claims.

## Status definitions

- **Supported:** part of the release contract and covered by a required gate.
- **Experimental:** usable behind an explicit option; compatibility may change.
- **Internal:** service-to-service contract, not a public client API.
- **Roadmap:** proposed or incomplete; not a release capability.

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
| CUDA backend | Supported | Hardware integration |
| ROCm backend | Supported | Hardware integration |
| Apple Metal backend | Supported | Hardware integration |
| Vulkan backend | Supported | Hardware integration |
| Intel SYCL and multi-GPU backends | Roadmap | None until implemented |
| Browser or visual verification | Roadmap | None until implemented |

## Service contracts

| Service surface | Status | Notes |
|---|---|---|
| Sandbox health, languages, execute, syntax-check, shell, and background jobs | Internal | Called by proxy and V3; direct host use is a developer workflow |
| V3 generate, run, plan, and health | Internal | `/v3/generate` is the proxy integration path |
| V3 AST edit, symbol index, and complexity endpoints | Experimental internal | Tree-sitter availability determines capability |
| Geometric Lens `/v1/*` endpoints | Supported authenticated API | Requires configured local API keys |
| Geometric Lens `/internal/*` endpoints | Internal | Intended only for the ATLAS service network |
| llama-server inference, completion, embedding, and health | Upstream contract | Qualified against the pinned llama.cpp revision |

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
