import torch
import torch.nn as nn


class NeuralAppraisal(nn.Module):
    """Simple feedforward network mapping a state vector to an emotion vector.

    The emotion vector is expected to match the dimensionality of the
    existing appraisal outputs (e.g., [suddenness, goal_relevance,
    conduciveness, power]).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is 2D: (batch, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)
