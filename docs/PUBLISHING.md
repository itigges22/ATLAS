# Publishing Lens + ASA Artifacts

This guide walks you through contributing trained **Geometric Lens** (`cost_field.pt`) or **ASA control vectors** (`*.gguf`) back to ATLAS so other users running the same base model get them automatically.

It's the long-form walkthrough for the artifact-contribution flow introduced in [CONTRIBUTING.md](../CONTRIBUTING.md#contributing-trained-artifacts-lens--asa). CLI flag reference lives in [CLI.md](CLI.md). If you just want flag syntax, that's the right place — read this one first if you've never published before.

---

## What you'll do, end to end

1. **Train** an artifact locally (`atlas lens build` or `atlas asa build`)
2. **Publish** it (`atlas lens publish` / `atlas asa publish`) — this does **two** things in one command:
   - **Uploads the binary** to a HuggingFace repo *you* own
   - **Opens a registry PR** against `github.com/itigges22/ATLAS` containing the HF link + SHA-256 + dim + license
3. **Wait for review** — the maintainer pulls the artifact onto a verification VM, runs it against a private trust-gate set, and merges (or asks for changes) on the GitHub PR

Once the registry PR is merged, downstream users see your model show up under `atlas model list` and can install it with `atlas model install <name>` — your trained artifact comes along automatically.

---

## What you need before you start

| Requirement | Where to get it | Required? |
|---|---|---|
| HuggingFace account | https://huggingface.co/join | **Yes** — you own the repo your artifact lives in |
| HuggingFace write token | https://huggingface.co/settings/tokens (scope: write) | **Yes** — set as `HF_TOKEN` env var |
| `huggingface_hub` Python pkg | `pip install huggingface_hub` | **Yes** on the host (already in the lens container) |
| `gh` CLI (GitHub) | https://cli.github.com | **Optional** — auto-opens the registry PR. Without it, you'll get a paste-ready PR body printed to your terminal |
| `gh` authenticated | `gh auth login` | Only if you installed `gh` above |

You do **not** need a GitHub PAT separately — `gh` handles its own auth.
You do **not** need write access to the ATLAS repo — the PR is opened from your fork.

**Set your HF token:**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
# add to ~/.bashrc or ~/.zshrc so it sticks
```

The CLI also reads `HUGGINGFACE_HUB_TOKEN` and `HUGGING_FACE_HUB_TOKEN` if you've already set one for `huggingface-cli` use.

---

## Publishing everything at once (recommended)

After onboarding a model you have both lens halves and the ASA vector —
publish them as one action:

```bash
atlas publish --lens-repo your-username/atlas-lens-<model> \
              --asa-repo  your-username/atlas-asa-<model>
```

This uploads both artifact sets to their HF repos and opens a **single
registry PR** whose entry carries `lens_status="supported"` and
`asa_status="supported"` together — no sequencing between two PRs to
manage. `--dry-run` previews, `--skip-pr` uploads without the PR,
`--lens-only` / `--asa-only` delegate to the per-component commands
below (which handle the publish-while-the-other-PR-is-open case via
stacked PRs).

## Publishing a Lens artifact

Assumes you've already run `atlas lens build --samples your-data.json` and have a `cost_field.pt` in the artifact dir. If not, see the `atlas lens build` section in [CLI.md](CLI.md).

```bash
# Preview the PR body without uploading anything
atlas lens publish Qwen3.5-9B-Q6_K \
    --repo your-username/atlas-lens-qwen35-9b \
    --dry-run

# Real upload + open the registry PR
atlas lens publish Qwen3.5-9B-Q6_K \
    --repo your-username/atlas-lens-qwen35-9b
```

The `--repo` flag is the HuggingFace destination (created if it doesn't exist). Naming convention: `atlas-lens-<model-slug>` keeps it discoverable.

### What happens during publish

1. **Pre-flight** — checks `HF_TOKEN` is set, artifact file exists, and that `cost_field.pt` is actually a torch checkpoint (not a half-finished download).
2. **Hash** — SHA-256s the artifact so the PR has a tamper-detectable fingerprint.
3. **Upload to HF** — creates the repo (idempotent), uploads `cost_field.pt`, uploads `metric_tensor.pt` if you have one, generates a model card README with license + base-model badge.
4. **Render PR body** — produces a markdown checklist with the HF URL, SHA-256, input dim, license, and a suggested Python diff for `atlas/cli/commands/model_registry.py`.
5. **Open the PR** — if `gh` is installed and authed, the PR is built through the GitHub API: a branch is created (on your automatic fork if you can't push upstream), a real registry edit is committed — a complete `Model(...)` entry with `lens_status="supported"` and your HF repo recorded — and the PR opens against the `dev` branch. No local git checkout is needed; this works from any install. If the model is already registered upstream, or `gh` is unavailable, the body is printed for manual paste into https://github.com/itigges22/ATLAS/compare instead.

   Publishing the ASA vector right after the lens (before the lens PR merges) is handled: when the model's entry is pending in an open publish PR, the ASA change becomes a **separate PR stacked on it** (diff shows only the ASA edit; merging the first PR with branch deletion makes GitHub retarget the second to `dev` automatically). Fork-based publishes commit onto the pending PR's branch instead — GitHub can't base a PR on a fork's branch — so one PR carries both changes there.

### Common flags

| Flag | Purpose |
|---|---|
| `--license mit` | License declared in the model card (default `apache-2.0`; `mit` / `bsd-3-clause` also fine) |
| `--dry-run` | Hash + render PR body, skip HF upload and PR creation |
| `--skip-pr` | Upload to HF, print PR body for manual paste (use when `gh` is missing or you want to edit the body) |
| `--artifact-dir DIR` | Override which directory's `cost_field.pt` gets uploaded |

---

## Publishing an ASA control vector

Same shape, different artifact. Assumes you've trained a vector with `atlas asa build` (see [CLI.md](CLI.md) for the training walkthrough).

```bash
atlas asa publish Qwen3.5-9B-Q6_K \
    --repo your-username/atlas-asa-qwen35-9b \
    --dry-run

atlas asa publish Qwen3.5-9B-Q6_K \
    --repo your-username/atlas-asa-qwen35-9b
```

The publish flow:

1. Reads GGUF metadata from the `.gguf` to extract residual dim, layer count, and the `model_hint` baked in by the calibration script
2. Hashes + uploads to HF (single `.gguf` file + model card)
3. Renders a PR body documenting which model it's for and the suggested `asa_status="supported"` registry change
4. Opens the PR via `gh` (or prints for manual paste)

If your `ATLAS_CONTROL_VECTOR` is set to a container-relative path like `/models/x.gguf`, the CLI auto-resolves it to the host path by trying `<atlas_root>/models/` and `$ATLAS_MODELS_DIR`. You shouldn't need to translate paths manually.

---

## What happens after you submit

1. The maintainer gets a notification on the new PR
2. The artifact is pulled from your HF repo onto a verification VM
3. A private trust-gate test scores the artifact against a held-out pair set — designed to reject artifacts that mis-rank or look adversarial
4. On pass: the registry PR is merged. On fail: a comment lands on the PR explaining what tripped and how to address it

**Why the trust-gate set is private:** if it lived in this repo, anyone could train an artifact specifically tuned to pass it without actually generalizing. Security-by-obfuscation in this case is the right call — it forces submissions to be honestly good, not gate-aware.

Turnaround time is typically a day or two depending on maintainer availability. If your PR has been open for a week without a response, ping `@itigges22` in the PR thread.

---

## Troubleshooting

### `HF_TOKEN env var not set`

You haven't exported a token, or it's only set in a different shell. Run `echo $HF_TOKEN` to verify — if it's empty, `export HF_TOKEN=hf_...` and try again.

### `huggingface_hub not installed`

Run `pip install huggingface_hub`. The lens container has it baked in, but the host Python that runs `atlas` needs it too.

### `gh: command not found`

Either install `gh` from https://cli.github.com, or use `--skip-pr` — the CLI will print the PR body and you paste it into github.com/itigges22/ATLAS/compare manually. Both paths produce the same review outcome. (With `gh` present, no git checkout is required — the PR is created entirely through the GitHub API, including the fork for non-maintainers.)

### `Artifact input dim (0)` in the PR body

The dim probe needs `torch` installed on the host (`pip install torch`). Without it, the PR body shows "unverified" and the maintainer will probe the dim on their side. Not a blocker — the upload still happens.

### `cost_field.pt looks corrupted`

The pre-flight check failed to load the file as a torch checkpoint. Most often this means the training run was killed mid-save. Re-run `atlas lens build` and confirm it prints `[done] saved cost_field.pt` before retrying.

### `License must be permissive for redistribution`

ATLAS only accepts artifacts under permissive licenses (apache-2.0, mit, bsd-3-clause). If your training data was scraped under a more restrictive license, that license can't be loosened by repackaging the trained weights — please don't try.

---

## Workflow expectations for contributors

- **One artifact per PR.** Mixing a lens + ASA upload in the same PR makes it harder to bisect a verification failure.
- **Use real model names** matching the canonical registry naming (check `atlas model list` for examples) so the PR doesn't bounce on a naming mismatch.
- **Don't push artifacts to your HF repo by hand** before running publish — let the CLI manage it so the SHA in the PR body matches what's actually on HF.
- **If you find a bug in the publish flow** (not the artifact itself), open a separate GitHub issue rather than burying it in the PR comments.

---

## See also

- [CLI.md — atlas lens / atlas asa command reference](CLI.md)
- [CONFIGURATION.md — env vars including HF_TOKEN](CONFIGURATION.md)
- [CONTRIBUTING.md — broader contribution guidelines](../CONTRIBUTING.md)
