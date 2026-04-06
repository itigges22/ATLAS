"""Diagonal metric tensor G(x) for geometry-aware gradient descent.

PCA-reduced architecture:
  1. PCA: 4096 → 128 (removes curse of dimensionality)
  2. Global importance vector (128 params) — which PCs generally matter
  3. Input-dependent modulation (128→32→128) — adapts per sample
  G(x) = softplus(global + modulate(x_pca)) → unit-mean normalized

Trained via contrastive metric learning (triplet loss) to maximize
G-weighted distance between PASS and FAIL embeddings in PCA space.

Floor-clamped to prevent G_inv blow-up in corrections.
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Minimum G(x) value to prevent G_inv blow-up in corrections
G_FLOOR = 0.05


class MetricTensor(nn.Module):
    """Core G(x) network operating in reduced dimensionality.

    G(x) = softplus(global_logits + modulate(x)) / mean → unit-mean
    Clamped to [G_FLOOR, ∞) to prevent G_inv blow-up.
    """

    def __init__(self, input_dim: int = 128, hidden: int = 32):
        super().__init__()
        self.global_logits = nn.Parameter(torch.zeros(input_dim))
        self.modulate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, input_dim),
        )
        nn.init.zeros_(self.modulate[-1].weight)
        nn.init.zeros_(self.modulate[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute G(x) with unit-mean normalization and floor clamp."""
        raw = F.softplus(self.global_logits + self.modulate(x))
        g_mean = raw.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return (raw / g_mean).clamp(min=G_FLOOR)


class PCAMetricTensor(nn.Module):
    """Production wrapper: PCA projection + G(x) in reduced space.

    Takes 4096-dim embeddings, projects to 128-dim PCA space,
    computes G(x), and provides gradient projection for correctability.

    Usage:
        model = load_metric_tensor("models/metric_tensor.pt")
        g = model(embedding_4096)           # → (batch, 128) G(x) diagonal
        grad_pca = model.project_gradient(grad_C_4096)  # → (batch, 128)
        correctability = (g_inv * grad_pca).norm()
    """

    def __init__(self, pca_components: torch.Tensor, pca_mean: torch.Tensor,
                 metric_tensor: MetricTensor):
        super().__init__()
        self.register_buffer('pca_components', pca_components)  # (n_comp, 4096)
        self.register_buffer('pca_mean', pca_mean)              # (4096,)
        self.gx = metric_tensor

    @property
    def pca_dim(self) -> int:
        return self.pca_components.shape[0]

    @property
    def original_dim(self) -> int:
        return self.pca_components.shape[1]

    def project(self, x_full: torch.Tensor) -> torch.Tensor:
        """Project 4096-dim embeddings to PCA space."""
        return (x_full - self.pca_mean) @ self.pca_components.T

    def project_gradient(self, grad_full: torch.Tensor) -> torch.Tensor:
        """Project 4096-dim gradients to PCA space (no mean subtraction)."""
        return grad_full @ self.pca_components.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute G(x). Accepts either 4096-dim or pre-projected PCA-dim input."""
        if x.shape[-1] == self.original_dim:
            x = self.project(x)
        return self.gx(x)


def load_metric_tensor(model_path: str) -> Optional[PCAMetricTensor]:
    """Load G(x) from checkpoint, handling PCA architecture.

    Returns PCAMetricTensor ready for production use, or None on failure.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Metric tensor not found: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        arch = checkpoint.get('architecture', 'unknown')
        if arch == 'xgboost_importance':
            logger.info("G(x) architecture is 'xgboost_importance' — handled by XGBoost service")
            return None
        if arch != 'pca_contrastive':
            logger.warning(f"Unknown G(x) architecture: {arch}")
            return None

        pca_dim = checkpoint['pca_dim']
        hidden = checkpoint.get('hidden', 32)
        pca_components = checkpoint['pca_components']
        pca_mean = checkpoint['pca_mean']

        core = MetricTensor(input_dim=pca_dim, hidden=hidden)
        core.load_state_dict(checkpoint['state_dict'])
        core.eval()

        model = PCAMetricTensor(pca_components, pca_mean, core)
        model.eval()

        logger.info(f"G(x) loaded: {arch}, PCA {checkpoint['original_dim']}→{pca_dim}, "
                     f"explained variance={checkpoint.get('pca_explained_variance', '?')}")
        return model

    except Exception as e:
        logger.error(f"Failed to load G(x): {e}")
        return None
