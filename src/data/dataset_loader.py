import logging
import os
import sys

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.data import create_cifar10_2_7  # noqa: E402

logger = logging.getLogger()


class DatasetLoader:
    def __init__(self, config: dict):
        """
        Initialize the DatasetLoader with a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing dataset and
            data parameters.
        """
        self.config = config
        self.dataset_dir = os.path.join(
            project_root, 'data', self.config['dataset']
        )

    def load_dataset(self) -> tuple:
        """
        Load the specified dataset based on the configuration.

        Returns:
            tuple: A tuple containing the training and evaluation datasets.
        """
        if self.config['dataset'] == 'cifar10':
            train_ds, eval_ds = self._load_cifar10_dataset()
        elif self.config['dataset'] == 'cifar10_2^7':
            train_ds, eval_ds = self._load_cifar10_2_7_dataset()
        elif self.config['dataset'] == 'celeba':
            train_ds, eval_ds = self._load_celeba_dataset()
        else:
            logger.error(f"Dataset {self.config['dataset']} is not supported.")
            raise ValueError(f"Invalid dataset: {self.config['dataset']}")

        return train_ds, eval_ds

    def _load_cifar10_dataset(self) -> tuple:
        """
        Load the CIFAR-10 dataset with specified transformations.

        Returns:
            tuple: A tuple containing the training and evaluation datasets.
        """
        transform = transforms.Compose([
            transforms.Resize(
                (self.config['data']['height'], self.config['data']['width'])
            ),
            transforms.RandomHorizontalFlip(
                p=self.config['data']['horizontal_flip_rate']
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2.) - 1.)
        ])
        train_ds = datasets.CIFAR10(
            root=self.dataset_dir,
            train=True,
            download=True,
            transform=transform
        )
        eval_ds = datasets.CIFAR10(
            root=self.dataset_dir,
            train=False,
            download=True,
            transform=transform
        )

        return train_ds, eval_ds

    def _load_cifar10_2_7_dataset(self) -> tuple:
        """
        Load the CIFAR-10 2^7 dataset with specified transformations.

        Returns:
            tuple: A tuple containing the training and evaluation datasets.
        """
        if not os.path.exists(self.dataset_dir):
            create_cifar10_2_7()

        transform = transforms.Compose([
            transforms.Resize(
                (self.config['data']['height'], self.config['data']['width'])
            ),
            transforms.RandomHorizontalFlip(
                p=self.config['data']['horizontal_flip_rate']
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2.) - 1.)
        ])
        train_ds = datasets.ImageFolder(
            root=os.path.join(self.dataset_dir, 'train'), transform=transform
        )
        eval_ds = datasets.ImageFolder(
            root=os.path.join(self.dataset_dir, 'eval'), transform=transform
        )

        return train_ds, eval_ds

    def _load_celeba_dataset(self) -> tuple:
        """
        Load the CelebA dataset with specified transformations.

        Returns:
            tuple: A tuple containing the training and evaluation datasets.
        """
        transform = transforms.Compose([
            transforms.CenterCrop(self.config['data']['center_crop']),
            transforms.Resize(
                (self.config['data']['height'], self.config['data']['width'])
            ),
            transforms.RandomHorizontalFlip(
                p=self.config['data']['horizontal_flip_rate']
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2.) - 1.)
        ])
        ds = datasets.ImageFolder(root=self.dataset_dir, transform=transform)
        train_size = int(len(ds) * self.config['data']['train_ratio'])
        eval_size = len(ds) - train_size
        train_ds, eval_ds = random_split(ds, [train_size, eval_size])

        return train_ds, eval_ds

    def create_dataloader(self) -> tuple:
        """
        Create dataloaders for the training and evaluation datasets.

        Returns:
            tuple: A tuple containing the training and evaluation dataloaders.
        """
        train_ds, eval_ds = self.load_dataset()
        train_dl = DataLoader(
            train_ds,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        logger.info("Dataloaders are loaded.")

        return train_dl, eval_dl
