from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms as ttf

from polypnet.data.augment import Augmenter, TripletAugmenter


class PolypDataset(Dataset, ABC):
    def __init__(
        self,
        shape: Optional[Tuple[int, int]] = None,
        return_paths=False,
        augmenter: Augmenter = Augmenter(),
    ) -> None:
        super().__init__()

        self.shape = shape
        self.return_paths = return_paths
        self.augmenter = augmenter

    @abstractmethod
    def _get_image_pair(self, index) -> Tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_path_pair(self, index) -> Tuple[str, str]:
        raise NotImplementedError

    def __getitem__(self, index: int):
        image_path, mask_path = self._get_path_pair(index)
        image_t, mask_t = self._get_image_pair(index)

        # Augment first
        image_t, mask_t = self.augmenter(image_t, mask_t)

        # Resize
        if self.shape is not None:
            resizer = ttf.Resize(tuple(self.shape))
            image_t = resizer(image_t)
            mask_t = resizer(mask_t)

        if not self.return_paths:
            return image_t, mask_t
        else:
            return image_t, mask_t, image_path, mask_path


class PolypMulticlassDataset(Dataset):
    def __init__(
        self,
        shape: Optional[Tuple[int, int]] = None,
        return_paths=False,
        augmenter: TripletAugmenter = TripletAugmenter(),
    ) -> None:
        super().__init__()

        self.shape = shape
        self.return_paths = return_paths
        self.augmenter = augmenter

    @abstractmethod
    def _get_image_triplet(self, index) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def _get_path_triplet(self, index) -> Tuple[str, str, str]:
        raise NotImplementedError

    def __getitem__(self, index: int):
        image_path, mask_path, cls_path = self._get_path_triplet(index)
        image_t, mask_t, cls_t = self._get_image_triplet(index)

        # Augment first
        image_t, mask_t, cls_t = self.augmenter(image_t, mask_t, cls_t)

        # Resize
        if self.shape is not None:
            resizer = ttf.Resize(self.shape)
            image_t = resizer(image_t)
            mask_t = resizer(mask_t)
            cls_t = resizer(cls_t)

        if not self.return_paths:
            return image_t, mask_t, cls_t
        else:
            return image_t, mask_t, cls_t, image_path, mask_path, cls_path
