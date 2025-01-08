import logging
import os
import sys

from torch.utils.data import Dataset
from torchvision import transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.utils import load_config  # noqa: E402

logger = logging.getLogger()


def save_images(
    ds: Dataset,
    size: int,
    dataset_dir: str,
    train: bool = True,
) -> None:
    """
    Save images from a dataset to a specified directory.

    Args:
        ds: The dataset containing images and labels.
        size: The number of images to save.
        dataset_dir: The directory where images will be saved.
        train: A boolean indicating whether the images are from the training
               set or the evaluation set. Default is True.
    """
    for idx, (image, label) in enumerate(ds):
        save_dir = os.path.join(
            dataset_dir,
            'train' if train else 'eval',
            str(label)
        )
        os.makedirs(save_dir, exist_ok=True)
        image = transforms.ToPILImage()(image)

        n_existing_files = len(
            [f for f in os.listdir(save_dir) if f.endswith('.png')]
        )
        image.save(os.path.join(save_dir, f"{n_existing_files:03d}.png"))

        if idx + 1 >= size // 2:
            break


def create_cifar10_2_7() -> None:
    """
    Create a CIFAR-10 2^7 dataset by saving a subset of images from the
    CIFAR-10 dataset to a specified directory.

    Raises:
        SystemExit: If the CIFAR-10 2^7 dataset already exists.
    """
    dataset_dir = os.path.join(project_root, 'data', 'cifar10_2^7')
    if os.path.exists(dataset_dir):
        logger.error("CIFAR-10 2^7 dataset already exists")
        exit()

    cifar10_config = load_config('cifar10')

    from src.data.dataset_loader import DatasetLoader
    dataset_loader = DatasetLoader(cifar10_config)
    train_ds, eval_ds = dataset_loader.load_dataset()

    save_images(train_ds, 2**7, dataset_dir, train=True)
    save_images(eval_ds, 2**7, dataset_dir, train=False)

    logger.info("CIFAR-10 2^7 dataset is created")
