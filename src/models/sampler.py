import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models.euler_maruyama import EulerMaruyama  # noqa: E402
from src.models.sde import VPSDE  # noqa: E402


class Sampler:
    def __init__(self, config: dict) -> None:
        """
        Initialize the Sampler with SDE and configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        self.sde = VPSDE(config)
        self.config = config
        self.sampling_method = EulerMaruyama(self.sde, config)

    def generate_tensor(
        self, model: nn.Module, shape: tuple, ode: bool = True
    ) -> torch.Tensor:
        """
        Generate a tensor using the specified method.

        Args:
            model (nn.Module): The model to use for score computation.
            shape (tuple): The shape of the tensor to generate.
            ode (bool): Whether to use ODE method. Defaults to True.

        Returns:
            torch.Tensor: The generated tensor.
        """
        x = self.sde.prior_sampling(shape).to(self.config['device'])
        timesteps = torch.linspace(
            self.config['sde']['T'], self.config['sde']['eps'],
            self.config['sde']['timesteps'], device=self.config['device']
        )
        for t in timesteps:
            vec_t = torch.ones(shape[0], device=self.config['device']) * t
            with torch.no_grad():
                if ode:
                    x = self.sampling_method.ode_reverse_step(x, vec_t, model)
                else:
                    x = self.sampling_method.sde_reverse_step(x, vec_t, model)

        return x

    def generate_image(
        self, model: nn.Module, shape: tuple, ode: bool = True
    ) -> torch.Tensor:
        """
        Generate an image.

        Args:
            model (nn.Module): The model to use for score computation.
            shape (tuple): The shape of the tensor to generate.
            ode (bool): Whether to use ODE method. Defaults to True.

        Returns:
            torch.Tensor: The generated image.
        """
        tensor = self.generate_tensor(model, shape, ode)
        image = self.tensor_to_image(tensor).to('cpu')

        return image

    def tensor_to_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to an image.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            torch.Tensor: The converted image.
        """
        reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.) / 2)
        ])
        transformed_tensor = reverse_transform(tensor)
        image = torch.clamp(transformed_tensor * 255, min=0, max=255)
        image = image.to(dtype=torch.uint8)

        return image
