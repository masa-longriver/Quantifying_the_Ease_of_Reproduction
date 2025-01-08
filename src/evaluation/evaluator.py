import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.data import DatasetLoader  # noqa: E402
from src.evaluation.growth_rate_calculator import GrowthRateCalculator  # noqa: E402
from src.utils import ResultWriter, ImageHandler  # noqa: E402

logger = logging.getLogger()


class Evaluator:
    def __init__(self, config: dict, model: torch.nn.Module, 
                 result_writer: ResultWriter) -> None:
        """
        Initialize the Evaluator with configuration, model, and result writer.

        Args:
            config (dict): Configuration dictionary.
            model (torch.nn.Module): The model to evaluate.
            result_writer (ResultWriter): Handles writing results to files.
        """
        self.config = config
        self.model = model
        self.result_writer = result_writer
        self.image_handler = ImageHandler()
        self.growth_rate_calculator = GrowthRateCalculator(
            self.config, self.model
        )
        self.log_volume_growth_rate_list = []
        self._get_dataset()
    
    def _get_dataset(self) -> None:
        """
        Load the dataset using the DatasetLoader.
        """
        dataset_loader = DatasetLoader(self.config)
        self.train_ds, _ = dataset_loader.load_dataset()
    
    def _save_image(self, tensor: torch.Tensor, image_idx: int) -> None:
        """
        Save an image from a tensor.

        Args:
            tensor (torch.Tensor): The image tensor to save.
            image_idx (int): Index of the image for naming.
        """
        image = self.image_handler.tensor_to_image(tensor)
        file_name = f"image_{image_idx + 1:06d}.png"
        self.result_writer.save_data_image(image, file_name)
    
    def _save_log_volume_growth_rate(
            self, log_volume_growth_rate: torch.Tensor) -> None:
        """
        Save the log volume growth rate to a CSV file.

        Args:
            log_volume_growth_rate (torch.Tensor): The growth rate tensor.
        """
        self.log_volume_growth_rate_list.append(log_volume_growth_rate)
        concatenated_tensor = torch.cat(
            self.log_volume_growth_rate_list, dim=0
        )
        numpy_array = concatenated_tensor.numpy()
        column_names = [f"step_{i + 1}" for i in range(numpy_array.shape[-1])]
        df = pd.DataFrame(numpy_array, columns=column_names)
        file_name = "log_volume_growth_rate.csv"
        self.result_writer.save_log_volume_growth_rate(df, file_name)
    
    def _save_graph(self) -> None:
        """
        Save a graph of the log volume growth rate.
        """
        concatenated_tensor = torch.cat(
            self.log_volume_growth_rate_list, dim=0
        )
        numpy_array = concatenated_tensor.numpy()
        log_volume_growth_rate = numpy_array[:, -1]
        file_name = "log_volume_growth_rate.png"
        self.result_writer.save_graph(log_volume_growth_rate, file_name)

    def evaluate(self) -> None:
        """
        Evaluate the model by processing images and calculating growth rates.
        """
        for i, (x, _) in enumerate(self.train_ds):
            for flip in [False, True]:
                image_idx = i * 2 if not flip else i * 2 + 1
                if flip:
                    flip_transform = transforms.RandomHorizontalFlip(p=1)
                    x = flip_transform(x)
                self._save_image(x, image_idx)
                log_volume_growth_rate = (
                    self.growth_rate_calculator.calc_log_volume_growth_rate(x)
                ).unsqueeze(0)
                self._save_log_volume_growth_rate(log_volume_growth_rate)

                logging_str = (
                    f"image_idx: {image_idx + 1}, "
                    "log_volume_growth_rate: "
                    f"{log_volume_growth_rate[:, -1].item():.04f}"
                )
                logger.info(logging_str)

            if i * 2 >= self.config['evaluate']['n_img'] - 1:
                break

        self._save_graph()
