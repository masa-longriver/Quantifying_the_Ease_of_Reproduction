import logging
import os
import sys

import torch
import torch.nn as nn

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models.sampling import sampling  # noqa: E402
from src.models.sde import VPSDE  # noqa: E402
from src.utils.image_saver import save_sampling_images  # noqa: E402
logger = logging.getLogger()


class Generator:
    def __init__(self, config: dict, model: nn.Module, result_dir: str):
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config["device"] = device
        torch.backends.cudnn.benchmark = True

        self.sde = VPSDE(config)
        self.model = model.to(device)
        self.model = nn.DataParallel(self.model)

        self.result_dir = os.path.join(
            project_root, "results", config["dataset"], result_dir
        )
        os.makedirs(self.result_dir, exist_ok=True)
        logger.info(f"Result directory is created: {self.result_dir}")

    def generate(self) -> None:
        """
        Generate and save sampling images for the given epoch.
        """
        self.model.eval()
        shape = (
            self.config['sampling']['n_img'], self.config['data']['channel'],
            self.config['data']['height'], self.config['data']['width']
        )
        x = sampling(self.model, self.sde, shape, self.config, method='sde')

        save_dir = os.path.join(self.result_dir, "sampled_images")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "generated_images.png")
        save_sampling_images(
            x, save_path, n_row=self.config['sampling']['save_n_row']
        )
        logger.info("Images are sampled.")
