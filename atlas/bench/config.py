"""
Bench-specific configuration.

Reads settings from atlas.conf / .env and resolves the runtime output
directory (repo-root benchmark/) for results and dataset caches.
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the ATLAS project root directory (canonical resolution lives
    in atlas.env — works from any cwd and for editable installs)."""
    from atlas.env import atlas_root
    return Path(atlas_root())


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
    def benchmark_dir(self) -> Path:
        """Runtime output directory for bench runs. Stays at repo-root
        benchmark/ (not inside the atlas package) so results keep the
        path that `atlas lens build --from-results` expects and dataset
        caches never land in site-packages."""
        return self._root / "benchmark"

    @property
    def cache_dir(self) -> Path:
        """Dataset download cache."""
        return self.benchmark_dir / "datasets" / ".cache"

    @property
    def results_dir(self) -> Path:
        """Results output directory."""
        return self.benchmark_dir / "results"

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


# Global config instance
config = BenchmarkConfig()
