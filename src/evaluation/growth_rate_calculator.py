import os
import sys

import torch

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models import EulerMaruyama, VPSDE  # noqa: E402


class GrowthRateCalculator:
    def __init__(self, config: dict, model: torch.nn.Module) -> None:
        """
        Initialize the GrowthRateCalculator with configuration and model.

        Args:
            config (dict): Configuration dictionary.
            model (torch.nn.Module): The model to use for calculations.
        """
        self.config = config
        self.model = model
        self.sde = VPSDE(self.config)
        self.euler_maruyama = EulerMaruyama(self.sde, self.config)
        self._get_shape()

    def _get_shape(self) -> None:
        """
        Calculate and set the shapes for surrounds and blocks.
        """
        self.surrounds_shape = (
            self.config['evaluate']['n_surroundings'],
            self.config['data']['channel'],
            self.config['data']['height'],
            self.config['data']['width']
        )
        block_size = (
            self.config['evaluate']['n_surroundings'] //
            self.config['evaluate']['n_blocks']
        )
        self.block_shape = (
            block_size,
            self.config['data']['channel'],
            self.config['data']['height'],
            self.config['data']['width']
        )
    
    def project_tensor(
        self, center: torch.Tensor, surrounds: torch.Tensor, t: float
    ) -> tuple:
        """
        Project the tensor using Euler-Maruyama method.

        Args:
            center (torch.Tensor): The center tensor.
            surrounds (torch.Tensor): The surrounding tensors.
            t (float): The time step for projection.

        Returns:
            tuple: The next center and surrounds tensors.
        """
        vec_t = torch.ones(center.shape[0], device=self.config['device']) * t
        with torch.no_grad():
            next_center = self.euler_maruyama.ode_forward_step(
                center.to(self.config['device']), vec_t, self.model
            ).to('cpu')
            for i in range(self.config['evaluate']['n_blocks']):
                next_surrounds = torch.empty(self.surrounds_shape)
                tmp_surrounds = surrounds[
                    self.block_shape[0] * i: self.block_shape[0] * (i + 1)
                ].to(self.config['device'])
                tmp_surrounds = self.euler_maruyama.ode_forward_step(
                    tmp_surrounds.to(self.config['device']), vec_t, self.model
                )
                next_surrounds[
                    self.block_shape[0] * i: self.block_shape[0] * (i + 1)
                ] = tmp_surrounds.cpu()
        
        return next_center, next_surrounds
    
    def calc_log_growth_rate(
        self, center: torch.Tensor, next_center: torch.Tensor,
        surrounds: torch.Tensor, next_surrounds: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the log growth rate between current and next tensors.

        Args:
            center (torch.Tensor): The current center tensor.
            next_center (torch.Tensor): The next center tensor.
            surrounds (torch.Tensor): The current surrounds tensor.
            next_surrounds (torch.Tensor): The next surrounds tensor.

        Returns:
            torch.Tensor: The log growth rate tensor.
        """
        log_growth_rate = torch.empty((surrounds.shape[0],))
        for i in range(surrounds.shape[0]):
            log_growth_rate[i] = (
                torch.log(torch.norm(next_surrounds[i] - next_center)) -
                torch.log(torch.norm(surrounds[i] - center))
            )
        
        return log_growth_rate

    def calc_log_edge_growth_rate(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log edge growth rate for a given tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The log edge growth rate tensor.
        """
        log_edge_growth_rate = torch.empty(
            (self.config['evaluate']['timesteps'],
             self.config['evaluate']['n_surroundings'])
        )
        center = tensor.to('cpu')
        noise = torch.randn(self.surrounds_shape)
        surrounds = (
            center.repeat(self.config['evaluate']['n_surroundings'], 1, 1, 1) +
            noise * self.config['evaluate']['sigma']
        )
        surrounds = self._gram_schmidt(center, surrounds)
        
        timesteps = torch.linspace(
            self.config['sde']['eps'], self.config['sde']['T'],
            self.config['sde']['timesteps'], device=self.config['device']
        )
        for i, t in enumerate(timesteps):
            next_center, next_surrounds = self.project_tensor(
                center, surrounds, t
            )
            log_edge_growth_rate[i] = self.calc_log_growth_rate(
                center, next_center, surrounds, next_surrounds
            )
            center = next_center
            surrounds = self._gram_schmidt(center, next_surrounds)

            if i >= self.config['evaluate']['timesteps'] - 1:
                break
    
        return log_edge_growth_rate
    
    def calc_log_volume_growth_rate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log volume growth rate for a given tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The log volume growth rate tensor.
        """
        tensor = x.unsqueeze(0)
        log_edge_growth_rate = self.calc_log_edge_growth_rate(tensor)
        log_volume_growth_rate = log_edge_growth_rate.sum(dim=-1).cumsum(dim=0)

        return log_volume_growth_rate

    def _gram_schmidt(
        self, center: torch.Tensor, surrounds: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Gram-Schmidt process to orthogonalize surrounds.

        Args:
            center (torch.Tensor): The center tensor.
            surrounds (torch.Tensor): The surrounding tensors.

        Returns:
            torch.Tensor: The orthogonalized surrounds tensor.
        """
        diff_vec = surrounds - center
        center = center.reshape((-1, ))
        diff_vec = diff_vec.reshape((surrounds.shape[0], -1))

        sorted_indices = sorted(
            range(diff_vec.shape[0]),
            key=lambda i: torch.norm(diff_vec[i]),
            reverse=True
        )
        diff_vec = diff_vec[sorted_indices]

        basis_list = []
        for i in range(diff_vec.shape[0]):
            vec = diff_vec[i]
            for basis in basis_list:
                vec = (
                    vec -
                    torch.dot(vec, basis) * basis / torch.norm(basis) ** 2
                )
            norm = torch.norm(vec)
            vec = vec / norm
            basis_list.append(vec)
        
        basis_tensor = torch.stack(basis_list, dim=0)
        basis_tensor = basis_tensor * self.config['evaluate']['sigma'] + center
        basis_tensor = basis_tensor.reshape(self.surrounds_shape)

        return basis_tensor
