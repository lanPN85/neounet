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

from polypnet.data.augment import Augmenter, TripletAugmenter


class LciMulticlassDataset(PolypMulticlassDataset):
    def __init__(
        self,
        root_dir: str,
        shape=(448, 448),
        image_dir="images",
        mask_dir="mask_images",
        cls_dir="label_images",
        in_memory=False,
        **kwargs,
    ):
        super().__init__(shape=shape, **kwargs)

        self.root_dir = root_dir

        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._cls_dir = cls_dir
        self.in_memory = in_memory
        self._scan_files()

    def __len__(self) -> int:
        return len(self._triplets)

    def _get_path_triplet(self, index) -> Tuple[str, str, str]:
        return (
            self._triplets[index]["image"],
            self._triplets[index]["mask"],
            self._triplets[index]["cls"],
        )

    def _get_image_triplet(self, index) -> Tuple[Any, Any, Any]:
        image_path, mask_path, cls_path = self._get_path_triplet(index)

        # Read images
        if image_path in self._path_contents.keys():
            image_t = self._path_contents[image_path]
        else:
            image_t = read_image(image_path)

        if mask_path in self._path_contents.keys():
            mask_t = self._path_contents[mask_path]
        else:
            try:
                mask_t = self.read_mask(mask_path)
            except:
                logger.error(f"Error reading mask at {mask_path}")
                raise

        if cls_path in self._path_contents.keys():
            cls_t = self._path_contents[cls_path]
        else:
            try:
                cls_t = self.read_cls(cls_path)
            except:
                logger.error(f"Error reading label at {cls_path}")
                raise

        if self.in_memory:
            self._path_contents[image_path] = image_t
            self._path_contents[mask_path] = mask_t
            self._path_contents[cls_path] = cls_t

        return image_t, mask_t, cls_t

    @staticmethod
    def read_cls(path):
        im = Image.open(path).convert("RGB")
        arr = np.array(im).transpose(2, 0, 1)
        return torch.from_numpy(arr.astype(np.uint8))

    @staticmethod
    def read_mask(path):
        im = Image.open(path).convert("RGB")

        arr = np.array(im)
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3)
        elif len(arr.shape) == 3:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError("Unsupported shape")

        return torch.from_numpy(arr)

    def _scan_files(self):
        self._triplets = []
        self._path_contents = {}

        for image_name in os.listdir(self.image_dir):
            image_name, ext = image_name.split(".")
            image_path = os.path.join(self.image_dir, image_name + ".jpeg")
            mask_path = os.path.join(self.mask_dir, image_name + ".png")
            cls_path = os.path.join(self.cls_dir, image_name + ".png")

            if ext not in ("jpg", "jpeg", "png"):
                logger.warning(f"Skipping file {image_path}")
                continue

            if not os.path.exists(mask_path):
                logger.warning(f"No mask found for {image_path}")
                continue

            if not os.path.exists(cls_path):
                logger.warning(f"No label found for {image_path}")
                continue

            self._triplets.append(
                {"image": image_path, "mask": mask_path, "cls": cls_path}
            )

    @property
    def image_dir(self) -> str:
        return os.path.join(self.root_dir, self._image_dir)

    @property
    def mask_dir(self) -> str:
        return os.path.join(self.root_dir, self._mask_dir)

    @property
    def cls_dir(self) -> str:
        return os.path.join(self.root_dir, self._cls_dir)


class BalancedLciDataset(LciMulticlassDataset):
    def __init__(self, root_dir: str, balance_ratio=1, **kwargs):
        super().__init__(root_dir, **kwargs)
        self.balance_ratio = balance_ratio
        self._balance_data()

    def _balance_data(self):
        logger.debug("Balancing dataset")
        benign_counts, malign_counts = {}, {}
        benign_only, malign_only = [], []
        total_benign, total_malign = 0, 0

        for t in tqdm(self._triplets):
            td = t["cls"]
            cls_mask = self.read_cls(t["cls"])
            red_mask = (cls_mask[0, ...] > 128).int()
            green_mask = (cls_mask[1, ...] > 128).int()

            benign_mask = green_mask * (1 - red_mask)
            malign_mask = red_mask * (1 - green_mask)

            benign_count = torch.sum(benign_mask).item()
            malign_count = torch.sum(malign_mask).item()
            benign_counts[td] = benign_count
            malign_counts[td] = malign_count
            total_benign += benign_count
            total_malign += malign_count

            if benign_count > 0 and malign_count == 0:
                benign_only.append(t)
            if malign_count > 0 and benign_count == 0:
                malign_only.append(t)

        logger.debug(f"Total benign: {total_benign}, total malign: {total_malign}")

        if total_benign < total_malign:
            target_total = total_malign
            total_add = total_benign
            add_only = benign_only
            add_counts = benign_counts
        else:
            target_total = total_benign
            total_add = total_malign
            add_only = malign_only
            add_counts = malign_counts

        i = 0
        while total_add < self.balance_ratio * target_total:
            t = add_only[i % len(add_only)]
            td = t["cls"]
            total_add = total_add + add_counts[td]
            self._triplets.append(t)
            i += 1


def test_1():
    path = "data/LCI_polyps_published/label_images/0a383a74579b2f6875e51b971f3ae0ae.png"
    x = LciMulticlassDataset.read_cls(path)

    print(x)
    print(x.shape)
    print(torch.sum(x))


if __name__ == "__main__":
    test_1()
