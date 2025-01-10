import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as vutils


project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
logger = logging.getLogger()


class ResultWriter:
    def __init__(self, result_dir_name: str, config: dict):
        """
        Initialize ResultWriter with a directory name for storing results.

        Args:
            result_dir_name (str): Name of the directory to store results.
            config (dict): Configuration dictionary.
        """
        self.config = config
        self._create_result_dir(result_dir_name)

    def _create_result_dir(self, result_dir_name: str):
        """
        Create the main result directory if it does not exist.

        Args:
            result_dir_name (str): Name of the directory to be created.
        """
        self.result_dir = os.path.join(
            project_root, "results", self.config['dataset'], result_dir_name
        )
        os.makedirs(self.result_dir, exist_ok=True)
        logger.info(f"Result directory is created: {self.result_dir}")
    
    def _create_internal_dir(self, dir_name: str):
        """
        Create a directory for storing internal results if it does not exist.
        """
        dir_path = os.path.join(self.result_dir, dir_name)
        if not hasattr(self, dir_name):
            os.makedirs(dir_path, exist_ok=True)
            setattr(self, dir_name + "_dir", dir_path)

    def save_sampling_images(
        self,
        generated_image: torch.Tensor,
        file_name: str,
        figsize: tuple = (25, 25),
        nrow: int = 10
    ):
        """
        Save generated images as a grid in the sample directory.

        Args:
            generated_image (torch.Tensor): Tensor of images to be saved.
            file_name (str): Name of the file to save the images.
            figsize (tuple): Size of the figure for the image grid.
            nrow (int): Number of images in each row of the grid.
        """
        self._create_internal_dir("samples")
        grid = vutils.make_grid(generated_image, nrow=nrow, padding=2)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0))
        save_path = os.path.join(self.sample_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def save_trained_model(self, model: nn.Module, file_name: str):
        """
        Save the trained model to the model directory.

        Args:
            model: The model to be saved.
            file_name (str): Name of the file to save the model.
        """
        self._create_internal_dir("models")
        save_path = os.path.join(self.model_dir, file_name)
        torch.save(model, save_path)

    def save_data_image(self, image: torch.Tensor, file_name: str):
        """
        Save the image tensor as a PIL image.

        Args:
            image (torch.Tensor): The image tensor to be saved.
            file_name (str): Name of the file to save the image.
        """
        self._create_internal_dir("data")
        save_path = os.path.join(self.data_dir, file_name)
        pil_image = transforms.ToPILImage()(image)
        pil_image.save(save_path)
    
    def save_log_volume_growth_rate(
        self, log_volume_growth_rate_df: pd.DataFrame, file_name: str
    ):
        """
        Save the log volume growth rate as a CSV file.

        Args:
            log_volume_growth_rate_df (pd.DataFrame):
                The log volume growth rate data.
            file_name (str):
                Name of the file to save the log volume growth rate.
        """
        self._create_internal_dir("log_volume_growth_rate")
        save_path = os.path.join(self.log_volume_growth_rate_dir, file_name)
        log_volume_growth_rate_df.to_csv(save_path)

    def save_graph(
        self,
        log_volume_growth_rate: torch.Tensor,
        file_name: str,
        figsize: tuple = (30, 6)
    ):
        """
        Save the log volume growth rate as a graph.
        """
        self._create_internal_dir("graph")
        save_path = os.path.join(self.graph_dir, file_name)
        plt.figure(figsize=figsize)
        x = np.arange(len(log_volume_growth_rate))
        plt.scatter(x, log_volume_growth_rate)
        plt.xlabel("Image Index")
        plt.ylabel("Log Volume Growth Rate")
        plt.savefig(save_path)
        plt.close()
