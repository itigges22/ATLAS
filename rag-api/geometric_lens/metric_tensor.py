"""Diagonal metric tensor G(x) for geometry-aware gradient descent.

Architecture: ℝ^5120 → ℝ^512 → ℝ^5120
Activations: SiLU, Softplus (ensures positive diagonal)
"""

import torch
import torch.nn as nn

EMBEDDING_DIM = 5120


class MetricTensor(nn.Module):
    """Diagonal metric tensor G(x).

    Outputs a positive-definite diagonal that encodes directional cost
    of movement in embedding space. Used in the correction:
        Δx = -α G⁻¹ ∇C
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim),
            nn.Softplus(),  # Ensures positive diagonal
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diagonal metric tensor G(x).

        Args:
            x: Embedding tensor of shape (batch, input_dim).

        Returns:
            Positive diagonal of shape (batch, input_dim).
        """
        return self.net(x)
