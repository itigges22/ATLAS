"""V3 Metacognitive Model (Feature 3F) — Failure Pattern Modeling.

Builds an explicit model of Qwen3-14B-Q4_K_M's systematic failure patterns
per problem category. Stores patterns as a JSON lookup table and injects
compensating constraints before generation for known weaknesses.

Config: [metacognitive] in atlas.conf
Telemetry: telemetry/metacognitive_events.jsonl

The insight: After hundreds of benchmark problems, patterns emerge.
"On bitwise problems, 73% of failures have incorrect shift direction."
The model has zero metacognition — ATLAS compensates based on accumulated
self-knowledge.
"""

import json
import time
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
class MetacognitiveConfig:
    """Configuration for Metacognitive Failure Modeling."""
    enabled: bool = False
    min_failures_per_category: int = 5
    min_pattern_frequency: float = 0.5
    profile_path: str = ""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FailurePattern:
    """A systematic failure pattern for a problem category."""
    pattern: str
    frequency: float = 0.0
    compensation: str = ""
    discovered_at: str = ""
    effectiveness: Optional[float] = None

    def to_dict(self) -> Dict:
        d: Dict = {
            "pattern": self.pattern,
            "frequency": self.frequency,
            "compensation": self.compensation,
            "discovered_at": self.discovered_at or datetime.now(timezone.utc).isoformat(),
        }
        if self.effectiveness is not None:
            d["effectiveness"] = self.effectiveness
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "FailurePattern":
        return cls(
            pattern=d.get("pattern", ""),
            frequency=d.get("frequency", 0.0),
            compensation=d.get("compensation", ""),
            discovered_at=d.get("discovered_at", ""),
            effectiveness=d.get("effectiveness"),
        )


@dataclass
class BenchmarkResult:
    """Simplified benchmark result for analysis."""
    task_id: str
    category: str
    passed: bool
    code: str = ""
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Telemetry event
# ---------------------------------------------------------------------------

@dataclass
class MetacognitiveEvent:
    """Telemetry event for metacognitive operations."""
    task_id: str
    operation: str = ""  # "lookup" or "analyze"
    category: str = ""
    num_warnings: int = 0
    num_patterns: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "operation": self.operation,
            "category": self.category,
            "num_warnings": self.num_warnings,
            "num_patterns": self.num_patterns,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_patterns(response: str) -> List[FailurePattern]:
    """Parse failure patterns from LLM analysis response."""
    import re
    patterns: List[FailurePattern] = []

    # Look for PATTERN N: blocks
    pattern_re = r'PATTERN\s+\d+\s*:(.*?)(?=PATTERN\s+\d+\s*:|$)'
    matches = re.findall(pattern_re, response, re.DOTALL | re.IGNORECASE)

    for block in matches:
        fp = FailurePattern(pattern="")

        desc_match = re.search(r'(?:ERROR|DESCRIPTION|PATTERN)[:\s]*(.*?)(?=FREQUENCY|COMPENSATION|CONSTRAINT|$)',
                                block, re.DOTALL | re.IGNORECASE)
        if desc_match:
            fp.pattern = desc_match.group(1).strip()

        freq_match = re.search(r'FREQUENCY[:\s]*(\d+(?:\.\d+)?)',
                                block, re.IGNORECASE)
        if freq_match:
            try:
                fp.frequency = float(freq_match.group(1))
                if fp.frequency > 1.0:
                    fp.frequency /= 100.0  # Convert percentage
            except ValueError:
                pass

        comp_match = re.search(r'(?:COMPENSATION|CONSTRAINT)[:\s]*(.*?)$',
                                block, re.DOTALL | re.IGNORECASE)
        if comp_match:
            fp.compensation = comp_match.group(1).strip()

        if fp.pattern:
            patterns.append(fp)

    # Fallback: look for numbered items
    if not patterns:
        numbered = re.findall(r'\d+[.)]\s+(.+)', response)
        for item in numbered:
            if len(item) > 10:
                patterns.append(FailurePattern(pattern=item.strip()))

    return patterns


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetacognitiveProfile:
    """Explicit model of Qwen3-14B's systematic failure patterns.

    When enabled, maintains a profile of category-specific weaknesses and
    provides compensating constraints during generation.

    When disabled, provides no warnings (noop).

    Args:
        config: MetacognitiveConfig instance.
        telemetry_dir: Directory for JSONL event logs.
    """

    def __init__(self, config: MetacognitiveConfig,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self._profile: Dict[str, List[FailurePattern]] = {}
        self.telemetry_dir = telemetry_dir
        self._events_file: Optional[Path] = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._events_file = telemetry_dir / "metacognitive_events.jsonl"

        # Load existing profile
        if config.profile_path:
            self._load(config.profile_path)

    @property
    def profile(self) -> Dict[str, List[FailurePattern]]:
        return dict(self._profile)

    @property
    def categories(self) -> List[str]:
        return list(self._profile.keys())

    @property
    def total_patterns(self) -> int:
        return sum(len(v) for v in self._profile.values())

    def get_warnings(self, problem_categories: List[str],
                     task_id: str = "") -> List[str]:
        """Get compensating constraints for known weakness categories.

        Args:
            problem_categories: Categories the problem belongs to.
            task_id: Task identifier for telemetry.

        Returns:
            List of compensating constraint strings.
        """
        if not self.config.enabled:
            return []

        warnings: List[str] = []
        for cat in problem_categories:
            if cat in self._profile:
                for entry in self._profile[cat]:
                    if entry.compensation:
                        # Skip entries known to be harmful
                        if entry.effectiveness is not None and entry.effectiveness <= 0:
                            continue
                        warnings.append(entry.compensation)

        if task_id:
            self._log_event(MetacognitiveEvent(
                task_id=task_id,
                operation="lookup",
                category=','.join(problem_categories),
                num_warnings=len(warnings),
            ))

        return warnings

    def analyze_benchmark(self, results: List[BenchmarkResult],
                          llm_call: Optional[LLMCallable] = None,
                          task_id: str = "") -> Dict[str, int]:
        """Post-benchmark analysis: identify systematic patterns.

        Args:
            results: List of benchmark results.
            llm_call: LLM callable for pattern extraction.
            task_id: Benchmark run identifier for telemetry.

        Returns:
            Dict mapping category to number of new patterns found.
        """
        if not self.config.enabled:
            return {}

        # Group by category
        by_category: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            by_category.setdefault(r.category, []).append(r)

        new_patterns: Dict[str, int] = {}

        for category, tasks in by_category.items():
            failures = [t for t in tasks if not t.passed]
            if len(failures) < self.config.min_failures_per_category:
                continue

            if llm_call is not None:
                patterns = self._extract_patterns(category, failures, llm_call)
            else:
                # Without LLM, just note the failure rate
                rate = len(failures) / len(tasks)
                patterns = [FailurePattern(
                    pattern=f"High failure rate ({rate:.0%}) in {category}",
                    frequency=rate,
                )]

            # Filter by minimum frequency
            significant = [
                p for p in patterns
                if p.frequency >= self.config.min_pattern_frequency
            ]

            if significant:
                existing = self._profile.get(category, [])
                existing_patterns = {p.pattern for p in existing}
                for p in significant:
                    if p.pattern not in existing_patterns:
                        p.discovered_at = datetime.now(timezone.utc).isoformat()
                        existing.append(p)
                self._profile[category] = existing
                new_patterns[category] = len(significant)

        if task_id:
            self._log_event(MetacognitiveEvent(
                task_id=task_id,
                operation="analyze",
                num_patterns=sum(new_patterns.values()),
            ))

        # Save if path configured
        if self.config.profile_path:
            self._save(self.config.profile_path)

        return new_patterns

    def update_effectiveness(self, category: str, pattern: str,
                              effectiveness: float) -> None:
        """Update the effectiveness score for a known pattern."""
        if category in self._profile:
            for p in self._profile[category]:
                if p.pattern == pattern:
                    p.effectiveness = effectiveness
                    break

    def to_dict(self) -> Dict:
        """Serialize profile to dict."""
        return {
            cat: [p.to_dict() for p in patterns]
            for cat, patterns in self._profile.items()
        }

    # -- Private helpers ----------------------------------------------------

    def _extract_patterns(self, category: str,
                           failures: List[BenchmarkResult],
                           llm_call: LLMCallable) -> List[FailurePattern]:
        """Use LLM to identify common patterns in failures."""
        failures_text = '\n'.join(
            f"Task {f.task_id}: Error: {f.error[:200]}"
            for f in failures[:10]  # Limit to avoid huge prompts
        )
        prompt = (
            f"<|im_start|>system\nYou are a failure analysis expert.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Analyze these {len(failures)} failing solutions for {category} problems.\n\n"
            f"{failures_text}\n\n"
            f"What patterns do you see? For each pattern, use this format:\n"
            f"PATTERN 1:\nDESCRIPTION: <the systematic error>\n"
            f"FREQUENCY: <fraction of failures>\n"
            f"COMPENSATION: <constraint to prevent this>\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        response, _, _ = llm_call(prompt, 0.3, 2048, 42)
        return parse_patterns(response)

    def _load(self, path: str) -> None:
        """Load profile from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for cat, entries in data.items():
                self._profile[cat] = [
                    FailurePattern.from_dict(e) for e in entries
                ]
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self, path: str) -> None:
        """Save profile to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except OSError:
            pass

    def _log_event(self, event: MetacognitiveEvent) -> None:
        if self._events_file is None:
            return
        try:
            with open(self._events_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass
