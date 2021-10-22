import os
import torch
import numpy as np

from torchvision.io import read_image
from PIL import Image
from polypnet.data.base import PolypDataset, PolypMulticlassDataset
from typing import Any, Optional, Tuple
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms as ttf
from tqdm import tqdm


class UpperPolypDataset(PolypDataset):
    def __init__(self, root_dir: str,
        shape=(448, 448),
        image_dir="images",
        mask_dir="masks",
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)

        self.root_dir = root_dir

        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._scan_files()

    def __len__(self) -> int:
        return len(self._triplets)

    def _get_path_pair(self, index) -> Tuple[str, str]:
        return (
            self._triplets[index]["image"],
            self._triplets[index]["mask"],
        )

    def _get_image_pair(self, index) -> Tuple[Any, Any]:
        image_path, mask_path = self._get_path_pair(index)

        # Read images
        image_t = read_image(image_path)

        try:
            mask_t = self.read_mask(mask_path)
        except:
            logger.error(f"Error reading mask at {mask_path}")
            raise

        return image_t, mask_t

    @staticmethod
    def read_mask(path):
        im = Image.open(path).convert("RGB")

        arr = np.array(im)
        if (len(arr.shape) == 2):
            arr = np.stack([arr] * 3)
        elif (len(arr.shape) == 3):
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError("Unsupported shape")

        return torch.from_numpy(arr)

    def _scan_files(self):
        self._triplets = []

        for image_name in os.listdir(self.image_dir):
            image_name, ext = image_name.split('.')
            image_path = os.path.join(self.image_dir, image_name + ".jpeg")
            mask_path = os.path.join(self.mask_dir, image_name + ".png")

            if ext not in ('jpg', 'jpeg', 'png'):
                logger.warning(f'Skipping file {image_path}')
                continue

            if not os.path.exists(mask_path):
                logger.warning(f'No mask found for {image_path}')
                continue

            self._triplets.append({
                'image': image_path,
                'mask': mask_path,
            })

    @property
    def image_dir(self) -> str:
        return os.path.join(self.root_dir, self._image_dir)

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.root_dir, self._mask_dir)

