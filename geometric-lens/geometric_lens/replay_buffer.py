"""V3 Replay Buffer — Domain-stratified experience replay for C(x) continual learning.

Paper: Experience Replay (Lin, 1992); Continual Learning surveys (Parisi et al., 2019)
Config: [lens_evolution] in atlas.conf
Telemetry: telemetry/replay_buffer_events.jsonl

Maintains a fixed-size buffer of representative pass/fail embedding pairs from
every domain C(x) has trained on. Uses reservoir sampling to maintain uniform
representation as new data arrives, and domain-stratified sampling to ensure
each domain contributes proportionally during replay.
"""

import json
import os
import random
from dataclasses import dataclass, field


@dataclass
class ReplayBufferConfig:
    enabled: bool = False
    max_size: int = 5000
    replay_ratio: float = 0.30
    data_dir: str = ""


class ReplayBuffer:
    """Domain-stratified experience replay for C(x) continual learning.

    Each entry stores:
        embedding: list[float]  — 5120-dim self-embedding
        label: str              — "PASS" or "FAIL"
        domain: str             — "LCB", "SciCode", "Custom", "TheoryFormation"
        epoch: int              — training epoch when learned
        difficulty_q: int       — 1-4 (quartile from C(x) energy)
    """

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.buffer: list = []
        self.domain_counts: dict = {}
        self.total_seen: int = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, embedding: list, label: str, domain: str,
            epoch: int, difficulty_q: int) -> None:
        """Add a new experience. Uses reservoir sampling if buffer is full.

        Reservoir sampling (Vitter, 1985) ensures each item has equal probability
        of being in the buffer regardless of stream length.
        """
        entry = {
            "embedding": embedding,
            "label": label,
            "domain": domain,
            "epoch": epoch,
            "difficulty_q": difficulty_q,
        }
        self.total_seen += 1

        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            # Reservoir sampling: replace random element with probability max_size/total_seen
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.max_size:
                self.buffer[idx] = entry

        self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1

    def add_batch(self, embeddings: list, labels: list, domain: str,
                  epoch: int, difficulty_quartiles: list = None) -> None:
        """Add multiple entries at once."""
        if difficulty_quartiles is None:
            difficulty_quartiles = [2] * len(embeddings)  # default middle quartile
        for emb, lbl, dq in zip(embeddings, labels, difficulty_quartiles):
            self.add(emb, lbl, domain, epoch, dq)

    @property
    def domains(self) -> list:
        """Return list of unique domains in the buffer."""
        return list(set(e["domain"] for e in self.buffer))

    def domain_distribution(self) -> dict:
        """Return count of entries per domain currently in buffer."""
        dist = {}
        for entry in self.buffer:
            d = entry["domain"]
            dist[d] = dist.get(d, 0) + 1
        return dist

    def sample_replay(self, n: int) -> list:
        """Sample n entries, stratified by domain.

        Each domain gets proportional representation. If a domain has fewer
        entries than its share, remaining slots go to other domains.
        """
        if not self.buffer or n <= 0:
            return []

        domains = self.domains
        if not domains:
            return []

        per_domain = max(1, n // len(domains))
        samples = []

        for domain in domains:
            domain_entries = [e for e in self.buffer if e["domain"] == domain]
            k = min(per_domain, len(domain_entries))
            samples.extend(random.sample(domain_entries, k))

        # If we still need more samples, draw from full buffer
        if len(samples) < n:
            remaining = [e for e in self.buffer if e not in samples]
            extra = min(n - len(samples), len(remaining))
            if extra > 0:
                samples.extend(random.sample(remaining, extra))

        return samples[:n]

    def get_training_mix(self, new_data: list, replay_ratio: float = 0.30) -> list:
        """Mix new data with replay buffer for continual learning.

        Args:
            new_data: List of dicts with 'embedding' and 'label' keys.
            replay_ratio: Fraction of combined data from replay (default 0.30).

        Returns:
            Shuffled combined list of new + replay data.
        """
        if not self.buffer or replay_ratio <= 0:
            return list(new_data)

        # Calculate how many replay samples to mix in
        # replay_ratio = n_replay / (n_replay + n_new)
        # n_replay = n_new * replay_ratio / (1 - replay_ratio)
        n_replay = int(len(new_data) * replay_ratio / (1 - replay_ratio))
        n_replay = min(n_replay, len(self.buffer))

        replay = self.sample_replay(n_replay)
        combined = list(new_data) + replay
        random.shuffle(combined)
        return combined

    def save(self, path: str) -> None:
        """Save buffer to JSON file.

        Embeddings are stored as lists of floats. File size for 5000 entries
        with 5120-dim embeddings: ~100MB raw JSON, but typically much less
        since most buffers won't be full.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        state = {
            "max_size": self.max_size,
            "total_seen": self.total_seen,
            "domain_counts": self.domain_counts,
            "buffer": self.buffer,
        }

        # Atomic write: write to temp file then rename
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, path)

    def load(self, path: str) -> bool:
        """Load buffer from JSON file. Returns True if loaded successfully."""
        if not os.path.exists(path):
            return False

        with open(path) as f:
            state = json.load(f)

        self.max_size = state.get("max_size", self.max_size)
        self.total_seen = state.get("total_seen", 0)
        self.domain_counts = state.get("domain_counts", {})
        self.buffer = state.get("buffer", [])
        return True

    def stats(self) -> dict:
        """Return buffer statistics."""
        dist = self.domain_distribution()
        label_dist = {}
        for entry in self.buffer:
            lbl = entry["label"]
            label_dist[lbl] = label_dist.get(lbl, 0) + 1

        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "total_seen": self.total_seen,
            "fill_ratio": len(self.buffer) / self.max_size if self.max_size > 0 else 0,
            "domains": dist,
            "labels": label_dist,
        }
