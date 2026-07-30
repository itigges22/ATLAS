"""
Benchmark-specific configuration.

Reads settings from atlas.conf and provides defaults for benchmark operations.
"""

import os
from pathlib import Path


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

    try:
        with open(conf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('export '):
                    line = line[len('export '):].lstrip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    # Drop a whitespace-preceded inline comment ("8080  # note");
                    # a '#' embedded directly in the value is preserved.
                    value = value.lstrip()
                    head, hash_sep, _ = value.partition('#')
                    if hash_sep and head and head[-1] in ' \t':
                        value = head
                    value = value.strip().strip('"').strip("'")
                    config[key] = value
    except (OSError, UnicodeDecodeError):
        # Unreadable conf (permissions, encoding) — behave as if absent.
        return config

    return config


def parse_dotenv() -> dict:
    """Parse the Docker Compose .env file (KEY=VALUE) if present. This is the
    source of truth for a Docker deployment's host ports and model file (the
    K3s path uses atlas.conf instead)."""
    env = {}
    path = get_project_root() / ".env"
    if not path.exists():
        return env
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line.startswith('export '):
                    line = line[len('export '):].lstrip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                # Drop a whitespace-preceded inline comment ("8080  # note");
                # a '#' embedded directly in the value is preserved.
                value = value.lstrip()
                head, hash_sep, _ = value.partition('#')
                if hash_sep and head and head[-1] in ' \t':
                    value = head
                env[key.strip()] = value.strip().strip('"').strip("'")
    except (OSError, UnicodeDecodeError):
        # Unreadable .env (permissions, encoding) — behave as if absent.
        return env
    return env


class BenchmarkConfig:
    """Configuration for benchmark operations."""

    def __init__(self):
        """Initialize configuration from atlas.conf, .env, and environment."""
        self._conf = parse_atlas_conf()
        self._env = parse_dotenv()
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
        """URL for llama-server. Resolution order: explicit LLAMA_URL env →
        in-cluster service DNS → Docker .env host port (ATLAS_LLAMA_PORT) →
        K3s NodePort from atlas.conf."""
        url = os.environ.get("LLAMA_URL")
        if url:
            return url
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return "http://llama-service:8000"
        port = self._env.get("ATLAS_LLAMA_PORT")   # Docker deployment (.env)
        if port:
            return f"http://localhost:{port}"
        port = self._conf.get("ATLAS_LLAMA_NODEPORT", "32735")   # K3s on-host
        return f"http://localhost:{port}"

    @property
    def rag_url(self) -> str:
        """URL for the geometric-lens / RAG service. Same resolution order as
        llama_url: LENS_URL env → in-cluster DNS → Docker .env
        (ATLAS_LENS_PORT) → K3s NodePort."""
        url = os.environ.get("LENS_URL")
        if url:
            return url
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return "http://geometric-lens-service:8000"
        port = self._env.get("ATLAS_LENS_PORT")
        if port:
            return f"http://localhost:{port}"
        port = self._conf.get("ATLAS_LENS_NODEPORT", "31144")
        return f"http://localhost:{port}"

    @property
    def llama_api_url(self) -> str:
        """URL for llama-server OpenAI-compatible API."""
        return f"{self.llama_url}/v1"

    @property
    def model_name(self) -> str:
        """Main model filename — Docker .env (ATLAS_MODEL_FILE) first, then
        atlas.conf (ATLAS_MAIN_MODEL)."""
        return (self._env.get("ATLAS_MODEL_FILE")
                or self._conf.get("ATLAS_MAIN_MODEL", ""))

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
            # Measured in-repo: 54.9% single-generation baseline on the
            # 599-task LiveCodeBench set (docs/reports/V3_ABLATION_STUDY.md).
            "livecodebench_pass1": 0.549,
            # No published or measured figure; unvalidated estimate.
            "scicode_pass1": 0.10,
        }

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = BenchmarkConfig()
