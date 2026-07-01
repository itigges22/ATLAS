"""V3 Lens Feedback — Online recalibration of C(x) during benchmarks.

After each task completes, records the final candidate's embedding + PASS/FAIL
label. Every N tasks, triggers Lens retrain via the Geometric Lens endpoint. After
retrain, recomputes sigmoid midpoint/steepness from the new energy distribution
and propagates to Blend-ASC and Budget Forcing in-memory.

Config: [lens_feedback] in atlas.conf
Telemetry: telemetry/lens_feedback_events.jsonl
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LensFeedbackConfig:
    """Configuration for online Lens recalibration."""
    enabled: bool = False
    retrain_interval: int = 50
    min_pass: int = 5
    min_fail: int = 5
    rag_api_url: str = "http://geometric-lens.atlas.svc.cluster.local:8099"
    domain: str = "LCB"
    use_replay: bool = True
    use_ewc: bool = True
    retrain_epochs: int = 50


# ---------------------------------------------------------------------------
# Feedback Collector
# ---------------------------------------------------------------------------

class LensFeedbackCollector:
    """Collects pass/fail embeddings and triggers periodic Lens retrain."""

    def __init__(self, config: LensFeedbackConfig,
                 telemetry_dir: Optional[Path] = None):
        self.config = config
        self.telemetry_dir = telemetry_dir
        self._buffer: List[Dict[str, Any]] = []
        self._all_data: List[Dict[str, Any]] = []
        self.current_midpoint: Optional[float] = None
        self.current_steepness: Optional[float] = None
        self.needs_propagation: bool = False
        self._retrain_count: int = 0
        # Latched off after a read-only 503 from the retrain endpoint. In the
        # standard compose deployment the models dir is mounted read-only, so
        # every retrain would 503 — once seen, stop attempting so the buffer
        # doesn't re-POST a growing multi-MB payload on every task.
        self._retrain_disabled: bool = False

    def record(self, embedding: List[float], label: str,
               task_id: str = "") -> None:
        """Record a task result for future retrain.

        When buffer reaches retrain_interval, triggers retrain automatically.
        """
        if not self.config.enabled:
            return

        entry = {
            "embedding": embedding,
            "label": label,
            "task_id": task_id,
        }
        self._buffer.append(entry)
        self._all_data.append(entry)

        if (not self._retrain_disabled
                and len(self._buffer) >= self.config.retrain_interval):
            self._trigger_retrain()

    def _trigger_retrain(self) -> None:
        """POST accumulated data to /internal/lens/retrain."""
        n_pass = sum(1 for d in self._all_data if d["label"] == "PASS")
        n_fail = sum(1 for d in self._all_data if d["label"] == "FAIL")

        if n_pass < self.config.min_pass or n_fail < self.config.min_fail:
            return

        training_data = [
            {"embedding": d["embedding"], "label": d["label"]}
            for d in self._all_data
        ]

        payload = json.dumps({
            "training_data": training_data,
            "epochs": self.config.retrain_epochs,
            "domain": self.config.domain,
            "use_replay": self.config.use_replay,
            "use_ewc": self.config.use_ewc,
        }).encode("utf-8")

        url = f"{self.config.rag_api_url}/internal/lens/retrain"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Structured refusal (e.g. 503 when the models dir is mounted
            # read-only). Log the service's reason. The training data lives in
            # _all_data, so clearing the buffer here loses nothing and stops
            # the next task from re-POSTing the same payload.
            reason = ""
            try:
                detail = json.loads(e.read().decode("utf-8"))
                if isinstance(detail, dict):
                    inner = detail.get("detail", detail)
                    if isinstance(inner, dict):
                        reason = inner.get("reason") or inner.get("error") or ""
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                pass
            self._log_event({
                "type": "retrain_error",
                "error": f"HTTP {e.code}" + (f": {reason}" if reason else ""),
                "total_samples": len(self._all_data),
            })
            # A 503 signals a read-only models dir, which won't change for the
            # life of this process — latch retrain off (logged once) so the
            # remaining tasks skip the endpoint entirely.
            if e.code == 503 and not self._retrain_disabled:
                self._retrain_disabled = True
                self._log_event({
                    "type": "retrain_disabled",
                    "reason": reason or f"HTTP {e.code}",
                    "total_samples": len(self._all_data),
                })
            self._buffer.clear()
            return
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            self._log_event({
                "type": "retrain_error",
                "error": str(e),
                "total_samples": len(self._all_data),
            })
            self._buffer.clear()
            return

        # A 200 with status != "ok" is a failed retrain (the endpoint reports
        # internal errors in-band). Don't count it, but clear the buffer — the
        # samples are retained in _all_data for the next scheduled attempt.
        if result.get("status") != "ok":
            self._log_event({
                "type": "retrain_error",
                "error": str(result.get("error") or result.get("reason")
                             or result.get("status") or "unknown"),
                "total_samples": len(self._all_data),
            })
            self._buffer.clear()
            return

        metrics = result.get("metrics", {})
        pass_mean = metrics.get("pass_energy_mean")
        fail_mean = metrics.get("fail_energy_mean")
        val_auc = metrics.get("val_auc", 0.0)

        if pass_mean is not None and fail_mean is not None:
            self._recompute_normalization(pass_mean, fail_mean)

        self._retrain_count += 1
        self._buffer.clear()

        self._log_event({
            "type": "retrain",
            "retrain_count": self._retrain_count,
            "total_samples": len(self._all_data),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_energy_mean": pass_mean,
            "fail_energy_mean": fail_mean,
            "val_auc": val_auc,
            "new_midpoint": self.current_midpoint,
            "new_steepness": self.current_steepness,
            "skipped": metrics.get("skipped", False),
        })

    def _recompute_normalization(self, pass_mean: float,
                                 fail_mean: float) -> None:
        """Compute new sigmoid midpoint and steepness from energy stats."""
        self.current_midpoint = (pass_mean + fail_mean) / 2.0
        separation = fail_mean - pass_mean
        self.current_steepness = 4.0 / max(separation, 0.1)
        self.needs_propagation = True

    def apply_to_components(self, blend_asc, budget_forcing) -> None:
        """Propagate recalibrated normalization to V3 components."""
        if not self.needs_propagation:
            return
        if self.current_midpoint is None or self.current_steepness is None:
            return

        if blend_asc is not None:
            blend_asc.config.energy_midpoint = self.current_midpoint
            blend_asc.config.energy_steepness = self.current_steepness
        if budget_forcing is not None:
            budget_forcing.config.energy_midpoint = self.current_midpoint
            budget_forcing.config.energy_steepness = self.current_steepness

        self._log_event({
            "type": "propagation",
            "midpoint": self.current_midpoint,
            "steepness": self.current_steepness,
        })

        self.needs_propagation = False

    def _log_event(self, event: Dict) -> None:
        """Append event to JSONL telemetry file."""
        if not self.telemetry_dir:
            return
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            path = self.telemetry_dir / "lens_feedback_events.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            # best-effort: swallow on failure (caller continues)
            pass
