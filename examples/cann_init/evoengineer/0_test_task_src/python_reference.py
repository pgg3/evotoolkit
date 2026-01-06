import torch
import torch.nn as nn


class Model(nn.Module):
    """Reference model for element-wise addition."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


# Test configuration
M = 1024
N = 1024


def get_inputs():
    """Generate test inputs for forward()."""
    x = torch.randn(M, N)
    y = torch.randn(M, N)
    return [x, y]


def get_init_inputs():
    """Return __init__() parameters."""
    return []
