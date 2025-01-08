import os

import torch
import torch.nn as nn

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_model(config: dict) -> nn.DataParallel:
    """
    Load a pretrained model from the specified path in the configuration.

    Args:
        config (dict): Configuration dictionary containing model and dataset
                       information.

    Returns:
        nn.DataParallel: The loaded model wrapped in DataParallel.
    """
    model_path = os.path.join(
        project_root, "results", config["dataset"],
        config["pretrained_model_path"]
    )
    model = torch.load(model_path)
    model = nn.DataParallel(model).to(config['device'])

    return model
