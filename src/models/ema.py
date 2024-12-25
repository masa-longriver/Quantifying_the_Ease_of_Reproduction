import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) class for maintaining a moving average of
    model parameters.

    Args:
        config (dict): Configuration dictionary containing EMA parameters.
        model (nn.Module): The model whose parameters will be averaged.
    """
    def __init__(self, config: dict, model: nn.Module):
        self.decay = config['ema']['decay']
        self.model = model
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update the shadow parameters using exponential moving average (EMA).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    (1. - self.decay) * param.data +
                    self.decay * self.shadow[name]
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Apply the shadow parameters to the model, backing up the current
        parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore the original parameters from the backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
