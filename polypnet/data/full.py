import os
import torch
import torch.nn as nn
import numpy as np

from typing import Any, List, Tuple
from torchvision.io import read_image
from torchvision import transforms as ttf
from loguru import logger

from polypnet.data.base import PolypDataset


class FullDataset(PolypDataset):
    def __init__(self, root_dir: str,
        shape=(448, 448),
        image_dir="images",
        mask_dir="masks",
        **kwargs
    ):
        super().__init__(shape=shape, **kwargs)

        self.root_dir = root_dir

        self.__image_dir = image_dir
        self.__mask_dir = mask_dir
        self.__scan_files()

    def __scan_files(self):
        self.__pairs = []

        for image_name in os.listdir(self.image_dir):
            ext = image_name.split('.')[-1]
            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, image_name)

            if ext not in ('jpg', 'png'):
                logger.warning(f'Skipping file {image_path}')
                continue

            if not os.path.exists(mask_path):
                logger.warning(f'No mask found for {image_path}')
                continue

            self.__pairs.append({
                'image': image_path,
                'mask': mask_path
            })

    def __len__(self) -> int:
        return len(self.__pairs)

    def _get_image_pair(self, index) -> Tuple[Any, Any]:
        pair = self.__pairs[index]
        image_path, mask_path = pair['image'], pair['mask']

        # Read images
        image_t = read_image(image_path)
        try:
            mask_t = self.read_mask(mask_path)
        except:
            logger.error(f"Error reading mask at {mask_path}")
            raise

        return image_t, mask_t

    def _get_path_pair(self, index) -> Tuple[str, str]:
        pair = self.__pairs[index]
        return pair['image'], pair['mask']

    @staticmethod
    def read_mask(path):
        im = Image.open(path)

        arr = np.array(im)
        if (len(arr.shape) == 2):
            arr = np.stack([arr] * 3)
        elif (len(arr.shape) == 3):
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError("Unsupported shape")

        return torch.from_numpy(arr)

    @property
    def image_dir(self) -> str:
        return os.path.join(self.root_dir, self.__image_dir)

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.root_dir, self.__mask_dir)


def test_1():
    path = "data/all_datasets/TrainDataset/mask/cju2ma647l0nj0993ot4deq2q.png"
    x = FullDataset.read_mask(path)

    print(x)
    print(x.shape)

if __name__ == "__main__":
    test_1()
