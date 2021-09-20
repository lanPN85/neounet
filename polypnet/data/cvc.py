import os
import torch
import random
import torch.nn as nn
import yaml

from typing import Any, List, Tuple
from torchvision.io import read_image
from torchvision import transforms as ttf
from loguru import logger
from torch.utils.data import Dataset, ConcatDataset

from polypnet.data.base import PolypDataset


class CvcDataset(PolypDataset):
    def __init__(
        self,
        root_dir: str,
        shape=(352, 352),
        image_dir="original",
        mask_dir="segmentation",
        **kwargs,
    ):
        super().__init__(shape=shape, **kwargs)

        self.root_dir = root_dir
        self.shape = shape

        self.__image_dir = image_dir
        self.__mask_dir = mask_dir
        self.__scan_files()

    def __scan_files(self):
        self.__pairs = []

        for image_name in os.listdir(self.image_dir):
            ext = image_name.split(".")[-1]
            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, image_name)

            if ext not in ("jpg", "png"):
                logger.warning(f"Skipping file {image_path}")
                continue

            if not os.path.exists(mask_path):
                logger.warning(f"No mask found for {image_path}")
                continue

            self.__pairs.append({"image": image_path, "mask": mask_path})

    def __len__(self) -> int:
        return len(self.__pairs)

    def _get_image_pair(self, index) -> Tuple[Any, Any]:
        pair = self.__pairs[index]
        image_path, mask_path = pair["image"], pair["mask"]

        # Read images
        image_t = read_image(image_path)
        try:
            mask_t = read_image(mask_path)
        except:
            logger.error(f"Error reading mask at {mask_path}")
            raise

        return image_t, mask_t

    def _get_path_pair(self, index) -> Tuple[str, str]:
        pair = self.__pairs[index]
        return pair["image"], pair["mask"]

    @property
    def image_dir(self) -> str:
        return os.path.join(self.root_dir, self.__image_dir)

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.root_dir, self.__mask_dir)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.root_dir, "meta.yml")


class CvcMultiDataset(ConcatDataset):
    def __init__(self, root_dirs: List[str], **kwargs):
        super().__init__([CvcDataset(root_dir, **kwargs) for root_dir in root_dirs])
