# Qwen3.6-27B-MTP on ATLAS (macOS Metal)

## Model specs

- Architecture: Gated DeltaNet + Gated Attention hybrid, 5120-dim hidden, 64 layers
- MTP: Multi-Token Prediction (inline draft head, `--spec-type draft-mtp`)
- Context: 262K native, 32K recommended for 64GB M1 Max
- Quantization: UD-Q4_K_XL (19 GB on disk, ~24 GB VRAM at 32K context)
- License: Apache 2.0

## What works

- Base inference via Metal-accelerated llama.cpp
- Lens C(x) cost-field scoring (5120-dim, 200-epoch contrastive training)
- Full V3 pipeline (budget forcing, plan search, candidate selection, refinement loop)

## What doesn't

- **MTP speculative decoding**: crashes during draft context init (`GGML_ASSERT missing result_norm/result_embd`) and has net throughput loss on Metal even when it doesn't crash (llama.cpp #23752). Disabled by default (`ATLAS_ENABLE_MTP=0`).
- **ASA steering vectors**: no vector trained for Qwen3.6 residuals. Would need `atlas asa build` against this model.
- **e2e_smoke in atlas doctor**: 300 tokens × ~5 tok/s on M1 Max exceeds the 60s hard timeout. The model itself works fine for interactive use.

## Fresh clone setup

```bash
# 1. Clone and configure
git clone https://github.com/itigges22/ATLAS
cd ATLAS
cp .env.example .env

# Edit .env:
#   ATLAS_MODEL_FILE=Qwen3.6-27B-UD-Q4_K_XL.gguf
#   ATLAS_MODEL_NAME=Qwen3.6-27B-MTP-UD-Q4_K_XL
#   ATLAS_ENABLE_MTP=0

# 2. Download the model (17.9 GB)
hf download unsloth/Qwen3.6-27B-MTP-GGUF \
  --include "*UD-Q4_K_XL*" \
  --local-dir ./models

# 3. Build native Metal llama-server
./scripts/atlas-setup-macos.sh

# 4. Start the native server (keep this terminal open)
source .env
./scripts/atlas-llama-macos.sh

# 5. Start Docker services (new terminal)
docker compose -f docker-compose.yml -f docker-compose.macos.yml up -d

# 6. Verify
export ATLAS_MODEL_FILE=Qwen3.6-27B-UD-Q4_K_XL.gguf
atlas doctor
```

## Lens training (optional, ~2h on CPU)

```bash
# Download pre-computed 5120-dim embeddings
curl -sL "https://huggingface.co/datasets/itigges22/ATLAS/resolve/main/embeddings/training_embeddings_5120d.json" \
  -o ./geometric-lens/geometric_lens/models/qwen36-27b/training_embeddings_5120d.json

# Copy data into the running geometric-lens container
docker cp ./geometric-lens/geometric_lens/models/qwen36-27b/training_embeddings_5120d.json \
  atlas-geometric-lens-1:/tmp/

# Train C(x) inside the container (torch is pre-installed there)
docker exec -w /app atlas-geometric-lens-1 python3 -c "
import json, os
from geometric_lens.training import train_cost_field, save_cost_field

with open('/tmp/training_embeddings_5120d.json') as f:
    data = json.load(f)
data['labels'] = [1 if l == 'PASS' else 0 for l in data['labels']]
print(f'Loaded {len(data[\"embeddings\"])} embeddings, dim={data[\"dim\"]}')

result = train_cost_field(data, epochs=200)
test_auc = result.get('best_test_auc') or result.get('final_test_auc') or 0.0
print(f'Test AUC: {test_auc:.4f}')

os.makedirs('/tmp/qwen36-27b', exist_ok=True)
cost_path = save_cost_field(result['model'], save_dir='/tmp/qwen36-27b')
print(f'Saved: {cost_path}')
"

# Copy trained artifact back to host
docker cp atlas-geometric-lens-1:/tmp/qwen36-27b/cost_field.pt \
  ./geometric-lens/geometric_lens/models/qwen36-27b/

# Doctor requires metric_tensor.pt on disk (runtime uses XGBoost, not this file)
touch ./geometric-lens/geometric_lens/models/qwen36-27b/metric_tensor.pt

# Verify Lens
export ATLAS_LENS_MODELS=./geometric-lens/geometric_lens/models/qwen36-27b
atlas doctor
```

## Expected doctor output

```
  ✓ health/llama                      ok
  ✓ model_file                        Qwen3.6-27B-UD-Q4_K_XL.gguf (16.7 GB)
  ✓ lens_weights                      cost_field.pt + metric_tensor.pt (after training)
  ⚠ asa_steering                      expected — no vector trained for Qwen3.6
  ⚠ tier_constraints                  disk free — Qwen3.6 is 17.9 GB
  ✗ e2e_smoke                         timeout — 27B on M1 Max is slow, not a bug

  19 passed, 2 warnings, 1 failed, 1 skipped
  ATLAS install has failures — re-run with -v for detail.
```

## Registry entry

The model is registered in `atlas/cli/commands/model_registry.py` as:

- `name`: `Qwen3.6-27B-MTP-UD-Q4_K_XL`
- `tier`: `xlarge`
- `lens_status`: `no-artifacts` (until artifacts are published to HF)
- `download_url`: `None` (manual download via `hf download`)

## Key files

- `atlas/cli/commands/model_registry.py` — registry entry
- `inference/Dockerfile.v31` — llama.cpp SHA `25558268` (MTP merge)
- `inference/patches/expose-hidden-states.patch` — regenerated for new upstream
- `scripts/atlas-llama-macos.sh` — MTP flag gating via `ATLAS_ENABLE_MTP`
- `geometric-lens/geometric_lens/models/qwen36-27b/` — Lens artifacts
