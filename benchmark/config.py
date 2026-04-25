"""
Benchmark-specific configuration.

Reads settings from atlas.conf and provides defaults for benchmark operations.
"""

import os
import re
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the ATLAS project root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "atlas.conf").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def parse_atlas_conf() -> dict:
    """
    Parse the atlas.conf file and return configuration as a dictionary.

    Returns:
        Dictionary of configuration values.
    """
    config = {}
    conf_path = get_project_root() / "atlas.conf"

    if not conf_path.exists():
        return config

    with open(conf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                config[key] = value

    return config


class BenchmarkConfig:
    """Configuration for benchmark operations."""

    def __init__(self):
        """Initialize configuration from atlas.conf and environment."""
        self._conf = parse_atlas_conf()
        self._root = get_project_root()

    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return self._root

    @property
    def benchmark_dir(self) -> Path:
        """Benchmark module directory."""
        return self._root / "benchmark"

    @property
    def datasets_dir(self) -> Path:
        """Datasets directory."""
        return self.benchmark_dir / "datasets"

    @property
    def cache_dir(self) -> Path:
        """Dataset cache directory."""
        return self.datasets_dir / ".cache"

    @property
    def custom_dir(self) -> Path:
        """Custom tasks directory."""
        return self.benchmark_dir / "custom"

    @property
    def results_dir(self) -> Path:
        """Results output directory."""
        return self.benchmark_dir / "results"

    @property
    def submissions_dir(self) -> Path:
        """Submissions directory."""
        return self.results_dir / "submissions"

    @property
    def llama_url(self) -> str:
        """URL for the vLLM gen instance.

        Resolution order:
          LLAMA_GEN_URL → vLLM gen instance (port 8000 by convention)
          LLAMA_URL     → legacy single-server fallback
          K8s service   → http://vllm-gen:8000 inside the cluster
          NodePort      → http://localhost:{ATLAS_LLAMA_NODEPORT|8000}
        """
        url = os.environ.get("LLAMA_GEN_URL") or os.environ.get("LLAMA_URL")
        if url:
            return url

        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return "http://vllm-gen:8000"

        port = self._conf.get("ATLAS_LLAMA_NODEPORT", "8000")
        return f"http://localhost:{port}"

    @property
    def llama_embed_url(self) -> str:
        """URL for the vLLM embed instance (separate process from gen).

        vLLM serves only one task per instance, so embeddings need a dedicated
        process. Defaults to port 8001 alongside gen on 8000.
        """
        url = os.environ.get("LLAMA_EMBED_URL")
        if url:
            return url

        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return "http://vllm-embed:8001"

        port = self._conf.get("ATLAS_LLAMA_EMBED_NODEPORT", "8001")
        return f"http://localhost:{port}"

    @property
    def llama_api_url(self) -> str:
        """URL for the gen instance's OpenAI-compatible API."""
        return f"{self.llama_url}/v1"

    @property
    def model_name(self) -> str:
        """vLLM served-model-name for the gen instance."""
        return self._conf.get("ATLAS_MAIN_MODEL", os.environ.get("LLAMA_GEN_MODEL", "qwen3.5-9b"))

    @property
    def embed_model_name(self) -> str:
        """vLLM served-model-name for the embed instance."""
        return self._conf.get("ATLAS_EMBED_MODEL", os.environ.get("LLAMA_EMBED_MODEL", "qwen3.5-9b-embed"))

    @property
    def default_timeout_seconds(self) -> int:
        """Default timeout for code execution."""
        return 30

    @property
    def default_memory_limit_mb(self) -> int:
        """Default memory limit for code execution."""
        return 512

    @property
    def default_k(self) -> int:
        """Default number of attempts per task."""
        return 1

    @property
    def default_temperature_pass1(self) -> float:
        """Temperature for pass@1 (greedy decoding)."""
        return 0.0

    @property
    def default_temperature_passk(self) -> float:
        """Temperature for pass@k evaluation."""
        return 0.8

    @property
    def gpu_tdp_watts(self) -> float:
        """GPU TDP in watts (RTX 5060 Ti)."""
        return 180.0

    @property
    def gpu_cost_usd(self) -> float:
        """Estimated GPU cost in USD."""
        return 450.0

    @property
    def gpu_lifetime_hours(self) -> float:
        """Expected GPU lifetime in hours (5 years, 8 hours/day)."""
        return 5 * 365 * 8

    @property
    def cloud_pricing(self) -> dict:
        """Cloud API pricing per 1M tokens (input/output)."""
        return {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet": {"input": 3.0, "output": 15.0},
            "claude-haiku": {"input": 0.25, "output": 1.25}
        }

    @property
    def qwen3_14b_baselines(self) -> dict:
        """Published Qwen3 baseline scores (retained for V1/V2 comparison)."""
        return {
            "humaneval_pass1": 0.67,       # ~65-70%
            "mbpp_pass1": 0.734,           # 73.4% per tech report (3-shot)
            "humaneval_plus_pass1": 0.61,  # EvalPlus leaderboard estimate
            "mbpp_plus_pass1": 0.65,       # EvalPlus leaderboard estimate
            "livecodebench_pass1": 0.20,   # Placeholder — will update after first run
            "scicode_pass1": 0.10,         # Placeholder — will update after first run
        }

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = BenchmarkConfig()
