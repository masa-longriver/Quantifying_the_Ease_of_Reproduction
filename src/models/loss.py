import os
import sys

import torch
import torch.nn as nn

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models.score import calculate_score  # noqa: E402
from src.models.sde import VPSDE  # noqa: E402


def calculate_loss(
    x: torch.Tensor,
    model: nn.Module,
    sde: VPSDE,
    config: dict
) -> torch.Tensor:
    """
    Calculate the loss for the given input, model, SDE, and config.

    Args:
        x (torch.Tensor): The input tensor.
        model (torch.nn.Module): The model to compute the score.
        sde (VPSDE): The SDE object to compute the marginal probability.
        config (dict): Configuration dictionary containing SDE parameters.

    Returns:
        torch.Tensor: The computed loss.
    """
    t = (
        torch.rand(x.shape[0], device=config['device']) *
        (config['sde']['T'] - config['sde']['eps']) +
        config['sde']['eps']
    )
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    perturbed_x = mean + std[:, None, None, None] * z
    score = calculate_score(perturbed_x, t, model, sde)

    losses = torch.square(score * std[:, None, None, None] + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)

    return loss
