"""V3 Elastic Weight Consolidation — Prevents catastrophic forgetting in C(x).

Paper: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017)
Config: [lens_evolution] in atlas.conf
Telemetry: telemetry/ewc_events.jsonl

Computes the diagonal Fisher Information Matrix after each domain training,
then adds a penalty term during retraining that discourages moving weights
that were important for previous domains:

    L_total = L_task + lambda * Sum_i [ F_i * (theta_i - theta*_i)^2 ]

Where F_i is the Fisher diagonal (weight importance), theta*_i is the reference
weight snapshot, and lambda controls regularization strength.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EWCConfig:
    enabled: bool = False
    lambda_ewc: float = 1000.0
    fisher_samples: int = 500
    state_path: str = ""


class ElasticWeightConsolidation:
    """EWC regularization for C(x) continual learning.

    L_total = L_task + lambda * Sum_i F_i * (theta_i - theta*_i)^2
    """

    def __init__(self, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher: dict = {}
        self.reference_params: dict = {}

    @property
    def is_initialized(self) -> bool:
        """True if Fisher and reference params have been computed."""
        return len(self.fisher) > 0 and len(self.reference_params) > 0

    def compute_fisher(self, model, embeddings: list, labels: list,
                       n_samples: int = 500) -> None:
        """Compute diagonal Fisher Information Matrix from data.

        Uses squared gradients of the loss as a diagonal approximation
        of the Fisher Information Matrix. This measures how important
        each weight is for the current task.

        Args:
            model: CostField model (nn.Module).
            embeddings: List of float lists (5120-dim each).
            labels: List of "PASS" or "FAIL" strings.
            n_samples: Max samples to use for Fisher computation.
        """
        device = torch.device("cpu")

        # Initialize Fisher accumulators
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}

        # Note: model.eval() is the PyTorch eval mode toggle, not code evaluation
        model.eval()

        # Energy targets matching retrain_cost_field_bce
        PASS_TARGET = 2.0
        FAIL_TARGET = 25.0

        n_used = min(n_samples, len(embeddings))
        indices = list(range(len(embeddings)))
        if len(indices) > n_samples:
            import random
            indices = random.sample(indices, n_samples)

        for idx in indices:
            emb = embeddings[idx]
            label = labels[idx]
            target = PASS_TARGET if label == "PASS" else FAIL_TARGET

            x = torch.tensor([emb], dtype=torch.float32, device=device)
            t = torch.tensor([[target]], dtype=torch.float32, device=device)

            model.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, t)
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data ** 2

        # Average over samples
        for n in fisher:
            fisher[n] /= n_used

        self.fisher = fisher

        # Snapshot current weights as reference
        self.reference_params = {
            n: p.data.clone() for n, p in model.named_parameters()
            if p.requires_grad
        }

    def penalty(self, model) -> torch.Tensor:
        """Compute EWC penalty term to add to training loss.

        Returns:
            Scalar tensor: lambda * sum(F_i * (theta_i - theta*_i)^2)
        """
        if not self.is_initialized:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.reference_params:
                loss = loss + (self.fisher[n] * (p - self.reference_params[n]) ** 2).sum()

        return self.lambda_ewc * loss

    def save(self, path: str) -> None:
        """Save Fisher diagonal and reference weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        torch.save({
            "fisher": self.fisher,
            "reference_params": self.reference_params,
            "lambda_ewc": self.lambda_ewc,
        }, path)

    def load(self, path: str) -> bool:
        """Load Fisher diagonal and reference weights from disk.

        Returns True if loaded successfully.
        """
        if not os.path.exists(path):
            return False

        data = torch.load(path, map_location="cpu", weights_only=True)
        self.fisher = data["fisher"]
        self.reference_params = data["reference_params"]
        self.lambda_ewc = data.get("lambda_ewc", self.lambda_ewc)
        return True

    def stats(self) -> dict:
        """Return EWC state statistics."""
        if not self.is_initialized:
            return {
                "initialized": False,
                "lambda_ewc": self.lambda_ewc,
            }

        # Compute Fisher magnitude stats
        total_fisher = 0.0
        n_params = 0
        max_fisher = 0.0
        for n, f in self.fisher.items():
            total_fisher += f.sum().item()
            n_params += f.numel()
            max_fisher = max(max_fisher, f.max().item())

        return {
            "initialized": True,
            "lambda_ewc": self.lambda_ewc,
            "n_parameters": n_params,
            "mean_fisher": total_fisher / max(n_params, 1),
            "max_fisher": max_fisher,
            "n_layers": len(self.fisher),
        }
