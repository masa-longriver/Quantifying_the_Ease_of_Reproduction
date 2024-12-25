import math
import os
import sys

import torch
import torch.nn as nn

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models.score import score_fn  # noqa: E402
from src.models.sde import VPSDE  # noqa: E402


class EulerMaruyama:
    """
    A class to perform numerical integration using the Euler-Maruyama method
    for Stochastic Differential Equations (SDEs).

    Attributes:
        config (dict): Configuration dictionary containing SDE parameters.
        sde: The SDE object to be used for the solver.
        dt (float): The time step size for the integration.
    """
    def __init__(self, sde: VPSDE, config: dict):
        self.config = config
        self.sde = sde
        self.dt = 1. / self.config['sde']['timesteps']

    def ode_reverse_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Perform a reverse step using the ODE method.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time tensor.
            model (nn.Module): The model to compute the score.

        Returns:
            torch.Tensor: The updated tensor after the reverse step.
        """
        score = score_fn(x, t, model, self.sde)
        drift = self.sde.reverse_ode(x, t, score)[0]
        x = x - drift * self.dt

        return x

    def sde_reverse_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Perform a reverse step using the SDE method.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time tensor.
            model (nn.Module): The model to compute the score.

        Returns:
            torch.Tensor: The updated tensor after the reverse step.
        """
        z = torch.randn_like(x)
        score = score_fn(x, t, model, self.sde)
        drift, diffusion = self.sde.reverse_sde(x, t, score)
        x_mean = x - drift * self.dt
        x = x_mean + diffusion[:, None, None, None] * math.sqrt(self.dt) * z

        return x

    def ode_forward_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Perform a forward step using the ODE method.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time tensor.
            model (nn.Module): The model to compute the score.

        Returns:
            torch.Tensor: The updated tensor after the forward step.
        """
        score = score_fn(x, t, model, self.sde)
        drift = self.sde.reverse_ode(x, t, score)[0]
        x = x + drift * self.dt

        return x


def sampling(
    model: nn.Module,
    sde: VPSDE,
    shape: tuple,
    config: dict,
    method: str = 'ode'
) -> torch.Tensor:
    """
    Generate samples using the specified method (ODE or SDE).

    Args:
        model (nn.Module): The model to compute the score.
        sde (VPSDE): The SDE object to compute the marginal probability.
        shape (tuple): The shape of the samples to generate.
        config (dict): Configuration dictionary containing SDE parameters.
        method (str, optional): The method to use for sampling
            ('ode' or 'sde'). Defaults to 'ode'.

    Returns:
        torch.Tensor: The generated samples.
    """
    sampler = EulerMaruyama(sde, config)
    x = sde.prior_sampling(shape).to(config['device'])
    timesteps = torch.linspace(
        config['sde']['T'], config['sde']['eps'], config['sde']['timesteps'],
        device=config['device']
    )

    for t in timesteps:
        vec_t = torch.ones(shape[0], device=config['device']) * t
        with torch.no_grad():
            if method == 'ode':
                x = sampler.ode_reverse_step(x, vec_t, model)
            else:
                x = sampler.sde_reverse_step(x, vec_t, model)

    return x
