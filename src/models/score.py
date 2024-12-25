import os
import sys

import torch
import torch.nn as nn

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models.sde import VPSDE  # noqa: E402


def score_fn(
    x: torch.Tensor,
    t: torch.Tensor,
    model: nn.Module,
    sde: VPSDE,
) -> torch.Tensor:
    """
    Compute the score function for the given input, time, model, and SDE.

    Args:
        x (torch.Tensor): The input tensor.
        t (torch.Tensor): The time tensor.
        model (torch.nn.Module): The model to compute the score.
        sde (VPSDE): The SDE object to compute the marginal probability.

    Returns:
        torch.Tensor: The computed score function.
    """
    labels = t * 999
    score = model(x, labels)
    std = sde.marginal_prob(torch.zeros_like(x), t)[1]
    score = -score / std[:, None, None, None]

    return score
