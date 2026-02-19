"""Embedding correction module: Δx = -αG⁻¹∇C.

Applies natural gradient descent in embedding space to steer
the prompt embedding away from bug-prone regions.
"""

import torch

from geometric_lens.cost_field import CostField
from geometric_lens.metric_tensor import MetricTensor


def compute_correction(
    x: torch.Tensor,
    cost_field: CostField,
    metric_tensor: MetricTensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """Compute the natural gradient correction Δx = -αG⁻¹∇C.

    Args:
        x: Embedding tensor of shape (batch, dim) or (dim,).
        cost_field: Trained C(x) model.
        metric_tensor: Trained G(x) model.
        alpha: Step size (conservative: 0.01-0.05).

    Returns:
        Corrected embedding x + Δx.
    """
    was_1d = x.dim() == 1
    if was_1d:
        x = x.unsqueeze(0)

    x = x.detach().requires_grad_(True)

    # Forward through C(x)
    energy = cost_field(x)

    # Compute ∇C
    grad_C = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]

    # Compute G(x) diagonal
    with torch.no_grad():
        G_diag = metric_tensor(x)
        G_inv = 1.0 / (G_diag + 1e-8)

        # Natural gradient correction
        delta_x = -alpha * G_inv * grad_C

        # Apply correction
        corrected = x.detach() + delta_x

    if was_1d:
        corrected = corrected.squeeze(0)

    return corrected


def evaluate_energy(x: torch.Tensor, cost_field: CostField) -> float:
    """Get scalar energy for a single embedding.

    Args:
        x: Embedding tensor of shape (dim,) or (1, dim).

    Returns:
        Energy scalar (float).
    """
    with torch.no_grad():
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return cost_field(x).item()
