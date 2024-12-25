import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision.utils as vutils


reverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1.) / 2.)
])


def save_sampling_images(
    sample: torch.Tensor,
    save_path: str,
    n_row: int = 10,
    fig_size: tuple = (25, 25)
) -> None:
    """
    Save a grid of sampling images to a file.

    Args:
        sample (torch.Tensor): The tensor containing the images to be saved.
        save_path (str): The path where the image grid will be saved.
        n_row (int, optional): Number of images in each row of the grid.
            Defaults to 10.
        fig_size (tuple, optional): Size of the figure. Defaults to (25, 25).
    """
    sample = reverse_transform(sample)
    sample = torch.clamp(sample * 255, min=0, max=255)
    sample = sample.to(dtype=torch.uint8, device='cpu')
    grid = vutils.make_grid(sample, nrow=n_row, padding=2)
    plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
