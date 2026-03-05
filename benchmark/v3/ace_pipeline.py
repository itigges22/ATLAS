"""V3 ACE Pipeline (Feature 3G) — Persistent Context Engineering.

Maintains evolving "playbooks" of problem-solving principles learned across
benchmark runs. Uses a Generator-Reflector-Curator pipeline to process
results and inject relevant playbook entries into generation prompts.

Paper: Zhang et al. (arxiv:2510.04618, Jan 2026). +10% accuracy.
Config: [ace] in atlas.conf
Telemetry: telemetry/ace_events.jsonl

Playbook entries are:
  - Versioned and append-only
  - Pruned by confidence decay (Ebbinghaus curve)
  - Injected into prompts when relevant
  - Capped at 2000 tokens total context budget
"""

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# Type alias for LLM callable
LLMCallable = Callable[[str, float, int, Optional[int]], Tuple[str, int, float]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ACEConfig:
    """Configuration for ACE Persistent Context Engineering."""
    enabled: bool = False
    max_playbook_tokens: int = 2000
    confidence_decay_rate: float = 0.1
    min_confidence: float = 0.3
    playbook_path: str = ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlaybookEntry:
    """A learned principle stored in the playbook."""
    principle: str
    category: str = ""
    confidence: float = 1.0
    uses: int = 0
    successes: int = 0
    created_at: str = ""
    last_used: str = ""
    version: int = 1
    # Derivation tracking
    entry_id: str = ""
    depth: int = 0                              # 0=observed, 1-3=derived
    parent_ids: List[str] = field(default_factory=list)
    provenance: str = "observed"                # "observed" | "derived"
    source_task_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "principle": self.principle,
            "category": self.category,
            "confidence": self.confidence,
            "uses": self.uses,
            "successes": self.successes,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "last_used": self.last_used,
            "version": self.version,
            "entry_id": self.entry_id,
            "depth": self.depth,
            "parent_ids": self.parent_ids,
            "provenance": self.provenance,
            "source_task_id": self.source_task_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PlaybookEntry":
        return cls(
            principle=d.get("principle", ""),
            category=d.get("category", ""),
            confidence=d.get("confidence", 1.0),
            uses=d.get("uses", 0),
            successes=d.get("successes", 0),
            created_at=d.get("created_at", ""),
            last_used=d.get("last_used", ""),
            version=d.get("version", 1),
            entry_id=d.get("entry_id", ""),
            depth=d.get("depth", 0),
            parent_ids=d.get("parent_ids", []),
            provenance=d.get("provenance", "observed"),
            source_task_id=d.get("source_task_id", ""),
        )

    @property
    def effectiveness(self) -> float:
        """Success rate when this entry was used."""
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses

    def estimate_tokens(self) -> int:
        """Rough token count (words * 1.3)."""
        return int(len(self.principle.split()) * 1.3)


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class ACEEvent:
    """Telemetry event for ACE operations."""
    task_id: str
    operation: str = ""  # "inject", "learn", "prune"
    num_entries_injected: int = 0
    num_entries_learned: int = 0
    num_entries_pruned: int = 0
    playbook_size: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "operation": self.operation,
            "num_entries_injected": self.num_entries_injected,
            "num_entries_learned": self.num_entries_learned,
            "num_entries_pruned": self.num_entries_pruned,
            "playbook_size": self.playbook_size,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ACEPipeline:
    """ACE Persistent Context Engineering pipeline.

    Maintains a playbook of learned principles, injects relevant entries
    into generation prompts, and learns from benchmark results.

    When disabled, provides no context and learns nothing (noop).

    Args:
        config: ACEConfig instance.
        telemetry_dir: Directory for JSONL event logs.
    """

    def __init__(self, config: ACEConfig,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self._playbook: List[PlaybookEntry] = []
        self.telemetry_dir = telemetry_dir
        self._events_file: Optional[Path] = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = telemetry_dir / "ace_events.jsonl"

        # Load existing playbook
        if config.playbook_path:
            self._load(config.playbook_path)

    @property
    def playbook(self) -> List[PlaybookEntry]:
        return list(self._playbook)

    @property
    def playbook_size(self) -> int:
        return len(self._playbook)

    def get_context(self, problem_categories: List[str],
                    task_id: str = "") -> str:
        """Get relevant playbook entries as prompt context.

        Selects entries matching the problem categories, ranked by
        confidence, and fits within the token budget.

        Args:
            problem_categories: Categories the problem belongs to.
            task_id: Task identifier for telemetry.

        Returns:
            Formatted playbook context string (may be empty).
        """
        if not self.config.enabled or not self._playbook:
            return ""

        # Filter relevant entries
        relevant = []
        for entry in self._playbook:
            if entry.confidence < self.config.min_confidence:
                continue
            if not entry.category or entry.category in problem_categories:
                relevant.append(entry)

        if not relevant:
            return ""

        # Rank by confidence (highest first)
        relevant.sort(key=lambda e: e.confidence, reverse=True)

        # Fit within token budget
        context_parts: List[str] = []
        token_count = 0
        entries_used = 0

        for entry in relevant:
            entry_tokens = entry.estimate_tokens()
            if token_count + entry_tokens > self.config.max_playbook_tokens:
                break
            context_parts.append(f"- {entry.principle}")
            token_count += entry_tokens
            entries_used += 1

        if not context_parts:
            return ""

        context = "Known principles for this problem type:\n" + \
            '\n'.join(context_parts)

        if task_id:
            self._log_event(ACEEvent(
                task_id=task_id,
                operation="inject",
                num_entries_injected=entries_used,
                playbook_size=len(self._playbook),
            ))

        return context

    def learn(self, principle: str, category: str = "",
              task_id: str = "") -> PlaybookEntry:
        """Add a new principle to the playbook.

        Args:
            principle: The learned principle text.
            category: Problem category this applies to.
            task_id: Task identifier for telemetry.

        Returns:
            The created PlaybookEntry.
        """
        if not self.config.enabled:
            return PlaybookEntry(principle=principle, category=category)

        entry = PlaybookEntry(
            principle=principle,
            category=category,
            confidence=1.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            entry_id=str(uuid.uuid4()),
            depth=0,
            provenance="observed",
            source_task_id=task_id,
        )
        self._playbook.append(entry)

        if task_id:
            self._log_event(ACEEvent(
                task_id=task_id,
                operation="learn",
                num_entries_learned=1,
                playbook_size=len(self._playbook),
            ))

        if self.config.playbook_path:
            self._save(self.config.playbook_path)

        return entry

    def record_usage(self, principle: str, succeeded: bool) -> None:
        """Record whether a playbook entry helped on a task."""
        for entry in self._playbook:
            if entry.principle == principle:
                entry.uses += 1
                if succeeded:
                    entry.successes += 1
                entry.last_used = datetime.now(timezone.utc).isoformat()
                break

    def decay_confidence(self) -> int:
        """Apply confidence decay to all entries. Returns number pruned."""
        if not self.config.enabled:
            return 0

        pruned = 0
        surviving: List[PlaybookEntry] = []

        for entry in self._playbook:
            entry.confidence *= (1.0 - self.config.confidence_decay_rate)
            if entry.confidence >= self.config.min_confidence:
                surviving.append(entry)
            else:
                pruned += 1

        self._playbook = surviving
        return pruned

    def prune(self, task_id: str = "") -> int:
        """Prune low-confidence entries."""
        pruned = self.decay_confidence()

        if task_id and pruned > 0:
            self._log_event(ACEEvent(
                task_id=task_id,
                operation="prune",
                num_entries_pruned=pruned,
                playbook_size=len(self._playbook),
            ))

        if self.config.playbook_path:
            self._save(self.config.playbook_path)

        return pruned

    def derive(self, parent_ids: List[str], new_principle: str,
               category: str = "", task_id: str = "") -> Optional[PlaybookEntry]:
        """Compose a new principle from existing ones. Depth-capped at 3.

        Args:
            parent_ids: IDs of parent principles to derive from.
            new_principle: The derived principle text.
            category: Problem category.
            task_id: Task identifier for telemetry.

        Returns:
            The new PlaybookEntry, or None if depth cap exceeded or parents not found.
        """
        if not self.config.enabled:
            return None

        parents = [e for e in self._playbook if e.entry_id in parent_ids]
        if not parents:
            return None

        max_parent_depth = max(p.depth for p in parents)
        if max_parent_depth >= 3:
            return None  # Depth cap

        new_depth = max_parent_depth + 1
        confidence = min(p.confidence for p in parents) * 0.7

        entry = PlaybookEntry(
            entry_id=str(uuid.uuid4()),
            principle=new_principle,
            category=category,
            confidence=confidence,
            depth=new_depth,
            parent_ids=list(parent_ids),
            provenance="derived",
            source_task_id=task_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._playbook.append(entry)

        if task_id:
            self._log_event(ACEEvent(
                task_id=task_id,
                operation="derive",
                num_entries_learned=1,
                playbook_size=len(self._playbook),
            ))

        if self.config.playbook_path:
            self._save(self.config.playbook_path)

        return entry

    def kill_principle(self, entry_id: str) -> int:
        """Remove a principle and cascade-kill all descendants.

        Returns:
            Number of principles killed.
        """
        to_kill = {entry_id}
        killed_ids = set()

        while to_kill:
            current = to_kill.pop()
            if current in killed_ids:
                continue
            killed_ids.add(current)
            # Find children
            for entry in self._playbook:
                if current in entry.parent_ids and entry.entry_id not in killed_ids:
                    to_kill.add(entry.entry_id)

        before = len(self._playbook)
        self._playbook = [e for e in self._playbook if e.entry_id not in killed_ids]
        killed = before - len(self._playbook)

        if self.config.playbook_path and killed > 0:
            self._save(self.config.playbook_path)

        return killed

    def record_failure(self, entry_id: str) -> Optional[int]:
        """Record a failure for a principle. Auto-kill if 3+ failures.

        Returns:
            Number killed if auto-kill triggered, None otherwise.
        """
        for entry in self._playbook:
            if entry.entry_id == entry_id:
                entry.uses += 1
                failure_count = entry.uses - entry.successes
                if failure_count >= 3:
                    return self.kill_principle(entry_id)
                return None
        return None

    def find_related(self, principle: str,
                     categories: List[str]) -> List[PlaybookEntry]:
        """Find playbook entries related to a principle by category.

        Returns entries matching any of the given categories,
        sorted by confidence (highest first).
        """
        related = []
        for entry in self._playbook:
            if entry.category and entry.category in categories:
                related.append(entry)
        related.sort(key=lambda e: e.confidence, reverse=True)
        return related

    def to_dict(self) -> List[Dict]:
        """Serialize playbook to list of dicts."""
        return [e.to_dict() for e in self._playbook]

    # -- Private helpers ----------------------------------------------------

    def _load(self, path: str) -> None:
        """Load playbook from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                self._playbook = [PlaybookEntry.from_dict(e) for e in data]
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self, path: str) -> None:
        """Save playbook to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except OSError:
            pass

    def _log_event(self, event: ACEEvent) -> None:
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass
