import torch


class VPSDE:
    """
    Variance Preserving Stochastic Differential Equation (VPSDE) class.

    Args:
        config (dict): Configuration dictionary containing SDE parameters.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.beta_0 = config['sde']['beta_min']
        self.beta_1 = config['sde']['beta_max']

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Compute the drift and diffusion terms of the SDE.

        Args:
            x (torch.Tensor): The image tensor.
            t (torch.Tensor): The time tensor.

        Returns:
            tuple: A tuple containing the drift and diffusion tensors.
        """
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        Compute the mean and standard deviation of the marginal probability.

        Args:
            x (torch.Tensor): The image tensor.
            t (torch.Tensor): The time tensor.

        Returns:
            tuple: A tuple containing the mean and standard deviation tensors.
        """
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) -
            0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))

        return mean, std

    def prior_sampling(self, shape):
        """
        Sample from the prior distribution.

        Args:
            shape (tuple): The shape of the tensor to sample.

        Returns:
            torch.Tensor: A tensor sampled from the prior distribution.
        """
        return torch.randn(shape)

    def reverse_sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> tuple:
        """
        Compute the drift and diffusion terms of the reverse SDE.

        Args:
            x (torch.Tensor): The image tensor.
            t (torch.Tensor): The time tensor.
            score (torch.Tensor): The score tensor.

        Returns:
            tuple: A tuple containing the drift and diffusion tensors.
        """
        drift, diffusion = self.sde(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score

        return drift, diffusion

    def reverse_ode(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> tuple:
        """
        Compute the drift and diffusion terms of the reverse ODE.

        Args:
            x (torch.Tensor): The image tensor.
            t (torch.Tensor): The time tensor.
            score (torch.Tensor): The score tensor.

        Returns:
            tuple: A tuple containing the drift and diffusion tensors.
        """
        drift, diffusion = self.sde(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score / 2

        return drift, diffusion
