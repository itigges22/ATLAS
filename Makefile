# ATLAS — common one-shot tasks.
#
# Run `make help` to see what's available. Targets are grouped:
#   - install / uninstall: project Python install
#   - test / wire-tests / unit-tests: pytest
#   - lint: shell + YAML syntax checks
#   - up / down / logs: docker-compose
#   - model: pull the Qwen3.5-9B-AWQ weights via huggingface-cli
#   - preflight: hit each service end-to-end (run AFTER `make up`)

.PHONY: help install dev-install test wire-tests unit-tests lint up down logs model preflight clean

help:
	@echo "ATLAS — make targets"
	@echo
	@echo "  install      Install ATLAS Python package + aider-chat"
	@echo "  dev-install  Install just the test/dev deps (no aider, no GPU stuff)"
	@echo "  test         Run unit tests + wire tests"
	@echo "  wire-tests   Run only the vLLM wire integration tests"
	@echo "  unit-tests   Run only the unit tests (skip Redis-dependent suites)"
	@echo "  lint         bash -n on entrypoints + YAML validation on docker-compose"
	@echo "  up           docker-compose up -d (gen + embed + Lens + sandbox + proxy)"
	@echo "  down         docker-compose down"
	@echo "  logs         tail logs from all services"
	@echo "  model        pull QuantTrio/Qwen3.5-9B-AWQ via huggingface-cli"
	@echo "  preflight    hit every service end-to-end (run AFTER make up)"
	@echo "  clean        remove pycache + .pytest_cache + local .cache/"

install:
	pip install -e . aider-chat

dev-install:
	pip install pytest httpx pyyaml
	pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.0"
	pip install --no-deps -e .

test: unit-tests wire-tests

wire-tests:
	python -m pytest tests/integration/test_vllm_wire.py -v --tb=short

unit-tests:
	python -m pytest tests/ \
	    --ignore=tests/infrastructure \
	    --ignore=tests/integration/test_e2e_flow.py \
	    --ignore=tests/integration/test_e2e_training.py \
	    --ignore=tests/integration/test_security.py \
	    --ignore=tests/integration/test_atlas_proxy.py \
	    --ignore=tests/e2e \
	    -q

lint:
	@for f in benchmarks/h200/entrypoint.sh \
	          benchmarks/h200/preflight.sh \
	          benchmarks/h200/launch_on_h200.sh \
	          benchmarks/run_lcb_v6.sh \
	          benchmarks/run_full_baseline.sh \
	          benchmark/measure_bok_latency.sh \
	          scripts/install.sh \
	          scripts/verify-install.sh \
	          scripts/uninstall.sh \
	          scripts/download-models.sh \
	          scripts/lib/config.sh \
	          scripts/generate-manifests.sh \
	          scripts/build-containers.sh; do \
	    bash -n "$$f" && echo "OK: $$f"; \
	done
	@python -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('OK: docker-compose.yml')"
	@python -c "import yaml; yaml.safe_load(open('.github/workflows/vllm-wire.yml')); print('OK: .github/workflows/vllm-wire.yml')"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=50

model:
	@if ! command -v huggingface-cli >/dev/null; then pip install -q huggingface_hub; fi
	huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ \
	    --local-dir models/Qwen3.5-9B-AWQ \
	    --local-dir-use-symlinks False

preflight:
	./benchmarks/h200/preflight.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .cache
