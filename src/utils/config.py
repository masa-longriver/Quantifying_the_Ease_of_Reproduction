import logging
import os

import yaml

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
logger = logging.getLogger()


def load_model_config() -> dict:
    """
    Load the model configuration from the model.yaml file.

    Returns:
        dict: A dictionary containing the model configuration.
    """
    yaml_path = os.path.join(project_root, "config", "model.yaml")
    with open(yaml_path, "r") as f:
        model_config = yaml.safe_load(f)

    return model_config


def load_dataset_config(dataset: str) -> dict:
    """
    Load the dataset configuration from a specified YAML file.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        dict: A dictionary containing the dataset configuration.
    """
    yaml_path = os.path.join(project_root, "config", f"{dataset}.yaml")
    if not os.path.exists(yaml_path):
        logger.error(f"Dataset configuration file not found: {yaml_path}")
        raise FileNotFoundError(
            f"Dataset configuration file not found: {yaml_path}"
        )

    with open(yaml_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    return dataset_config


def load_config(dataset: str) -> dict:
    """
    Load and merge the model and dataset configurations.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        dict: A dictionary containing the merged configuration.
    """
    model_config = load_model_config()
    dataset_config = load_dataset_config(dataset)

    config = {**model_config, **dataset_config}
    config["dataset"] = dataset

    return config
