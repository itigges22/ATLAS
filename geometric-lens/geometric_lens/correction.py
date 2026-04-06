"""Embedding correction module: dx = -alpha * G_inv * grad_C.

Applies natural gradient descent in embedding space to steer
the prompt embedding away from bug-prone regions.

Supports PCA-reduced G(x): projects gradients to PCA space,
applies G_inv there, and projects corrections back.
"""

import torch

from geometric_lens.cost_field import CostField
from geometric_lens.metric_tensor import G_FLOOR


def compute_correction(
    x: torch.Tensor,
    cost_field: CostField,
    metric_tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """Compute the natural gradient correction dx = -alpha * G_inv * grad_C.

    Handles both full-dim and PCA-reduced G(x) via project_gradient().

    Args:
        x: Embedding tensor of shape (batch, dim) or (dim,).
        cost_field: Trained C(x) model.
        metric_tensor: Trained G(x) model (MetricTensor or PCAMetricTensor).
        alpha: Step size.

    Returns:
        Corrected embedding x + dx (same shape as input).
    """
    was_1d = x.dim() == 1
    if was_1d:
        x = x.unsqueeze(0)

    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        energy = cost_field(x_grad)
        grad_C = torch.autograd.grad(energy.sum(), x_grad, create_graph=False)[0]

    with torch.no_grad():
        if hasattr(metric_tensor, 'project_gradient'):
            # PCA mode: work in reduced space, project back
            grad_proj = metric_tensor.project_gradient(grad_C.detach())
            G_diag = metric_tensor(x.detach())  # auto-projects if PCA
            G_inv = 1.0 / G_diag.clamp(min=G_FLOOR)
            delta_pca = -alpha * G_inv * grad_proj
            # Project correction back to original space
            delta_x = delta_pca @ metric_tensor.pca_components
        else:
            # Full-dim mode
            G_diag = metric_tensor(x.detach())
            G_inv = 1.0 / G_diag.clamp(min=G_FLOOR)
            delta_x = -alpha * G_inv * grad_C.detach()

        corrected = x.detach() + delta_x

    if was_1d:
        corrected = corrected.squeeze(0)

    return corrected


def compute_correctability(
    x: torch.Tensor,
    cost_field: CostField,
    metric_tensor,
) -> float:
    """Compute correctability: how traversable the correction landscape is.

    correctability = ||G_inv * grad_C||_2  (in PCA or full space)

    Higher = steeper gradient AND lower curvature resistance → repair likely effective.

    Args:
        x: Embedding tensor of shape (dim,) or (1, dim).
        cost_field: Trained C(x) model.
        metric_tensor: Trained G(x) model (MetricTensor or PCAMetricTensor).

    Returns:
        Scalar correctability score (float).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    with torch.enable_grad():
        x_grad = x.detach().requires_grad_(True)
        energy = cost_field(x_grad)
        grad_C = torch.autograd.grad(energy.sum(), x_grad, create_graph=False)[0]

    with torch.no_grad():
        if hasattr(metric_tensor, 'project_gradient'):
            grad_proj = metric_tensor.project_gradient(grad_C.detach())
            G_diag = metric_tensor(x.detach())
        else:
            grad_proj = grad_C.detach()
            G_diag = metric_tensor(x.detach())

        G_inv = 1.0 / G_diag.clamp(min=G_FLOOR)
        natural_grad = G_inv * grad_proj
        return natural_grad.norm().item()
