import logging
import os

import matplotlib.pyplot as plt
import torch
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

    def _create_sample_dir(self):
        """
        Create a directory for storing sample images if it does not exist.
        """
        if not hasattr(self, "sample_dir"):
            self.sample_dir = os.path.join(self.result_dir, "samples")
            os.makedirs(self.sample_dir, exist_ok=True)

    def _create_model_dir(self):
        """
        Create a directory for storing models if it does not exist.
        """
        if not hasattr(self, "model_dir"):
            self.model_dir = os.path.join(self.result_dir, "models")
            os.makedirs(self.model_dir, exist_ok=True)

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
        self._create_sample_dir()
        grid = vutils.make_grid(generated_image, nrow=nrow, padding=2)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0))
        save_path = os.path.join(self.sample_dir, file_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def save_trained_model(self, model, file_name):
        """
        Save the trained model to the model directory.

        Args:
            model: The model to be saved.
            file_name (str): Name of the file to save the model.
        """
        self._create_model_dir()
        save_path = os.path.join(self.model_dir, file_name)
        torch.save(model, save_path)
