import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.data import load_dataloader  # noqa: E402
from src.models.ema import EMA  # noqa: E402
from src.models.loss import loss_fn  # noqa: E402
from src.models.sampling import sampling  # noqa: E402
from src.models.sde import VPSDE  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.utils import save_sampling_images  # noqa: E402

logger = logging.getLogger()


class Trainer:
    def __init__(self, config: dict, result_dir: str):
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config["device"] = device
        torch.backends.cudnn.benchmark = True

        self.train_dl, self.eval_dl = load_dataloader(config)
        self.sde = VPSDE(config)
        self.model = nn.DataParallel(UNet(config).to(device))
        self.ema = EMA(config, self.model)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['optim']['lr'],
            betas=(config['optim']['beta1'], config['optim']['beta2']),
            eps=config['optim']['eps'],
            weight_decay=config['optim']['weight_decay']
        )

        self.result_dir = os.path.join(
            project_root, "results", config["dataset"], result_dir
        )
        os.makedirs(self.result_dir, exist_ok=True)
        logger.info(f"Result directory is created: {self.result_dir}")

    def train(self):
        logger.info("Start training")
        for epoch in range(self.config['train']['epochs']):
            loss = self._run_epoch(epoch)

            if (epoch + 1) % self.config['train']['eval_interval'] == 0:
                eval_loss = self._evalate()
                logger.info(
                    f"Epoch {epoch + 1}, train loss: {loss:.4e}, "
                    f"eval loss: {eval_loss:.4e}"
                )
            elif (epoch + 1) % self.config['train']['log_interval'] == 0:
                logger.info(f"Epoch {epoch + 1}, train loss: {loss:.4e}")

            if (epoch + 1) % self.config['train']['sampling_interval'] == 0:
                self._sampling(epoch)

            if (epoch + 1) % self.config['train']['saving_interval'] == 0:
                self._save_model(epoch)

    def _run_epoch(self, epoch: int) -> float:
        """
        Run a single training epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average loss for the epoch.
        """
        running_loss = 0.

        self.model.train()
        for x, _ in self.train_dl:
            x = x.to(self.config['device'])
            self.optimizer.zero_grad()
            loss = loss_fn(x, self.model, self.sde, self.config)
            loss.backward()

            for g in self.optimizer.param_groups:
                g['lr'] = (
                    self.config['optim']['lr'] *
                    np.minimum(epoch / self.config['optim']['warmup'], 1.)
                )
            utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['optim']['grad_clip']
            )
            self.optimizer.step()
            self.ema.update()

            running_loss += loss.item() * x.size(0)

        return running_loss / len(self.train_dl.dataset)

    def _evalate(self):
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            float: The average loss over the evaluation dataset.
        """
        running_loss = 0.

        self.ema.apply_shadow()
        self.model.eval()
        with torch.no_grad():
            for x, _ in self.eval_dl:
                x = x.to(self.config['device'])
                loss = loss_fn(x, self.model, self.sde, self.config)
                running_loss += loss.item() * x.size(0)
            self.ema.restore()

        return running_loss / len(self.eval_dl.dataset)

    def _sampling(self, epoch: int) -> None:
        """
        Generate and save sampling images for the given epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.model.eval()
        shape = (
            self.config['sampling']['n_img'], self.config['data']['channel'],
            self.config['data']['height'], self.config['data']['width']
        )
        x = sampling(self.model, self.sde, shape, self.config, method='sde')

        save_dir = os.path.join(self.result_dir, "sampled_images")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch:06d}.png")
        save_sampling_images(
            x, save_path, n_row=self.config['sampling']['save_n_row']
        )
        logger.info("Images are sampled.")

    def _save_model(self, epoch: int) -> None:
        """
        Save the model.

        Args:
            epoch (int): The current epoch number.
        """
        save_dir = os.path.join(self.result_dir, "models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch:06d}.pth")
        torch.save(self.model, save_path)
        logger.info("Model is saved.")
