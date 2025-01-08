import torch
from torchvision import transforms


class ImageHandler:
    def image_to_tensor(self, image) -> torch.Tensor:
        """
        Convert an image to a tensor.

        Args:
            image: The image to convert.

        Returns:
            torch.Tensor: The converted tensor.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2.) - 1.)
        ])
        tensor = transform(image)

        return tensor
    
    def tensor_to_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to an image.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            torch.Tensor: The converted image.
        """
        transform = transforms.Lambda(lambda x: (x + 1.) / 2.)
        transformed_tensor = transform(tensor)
        image = torch.clamp(transformed_tensor * 255, min=0, max=255)
        image = image.to(dtype=torch.uint8)

        return image
