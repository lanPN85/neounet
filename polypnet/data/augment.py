import numpy as np
import torch
import torch.nn as nn
import albumentations as al
import matplotlib.pyplot as plt
import random

from loguru import logger
from torchvision.io import image, read_image
from torchvision.utils import make_grid


class Augmenter(nn.Module):
    def __init__(self,
        prob=0.7,
        blur_prob=0.7,
        jitter_prob=0.7,
        rotate_prob=0.7,
        flip_prob=0.7,
    ):
        super().__init__()

        self.prob = prob
        self.blur_prob = blur_prob
        self.jitter_prob = jitter_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob

        self.transforms = al.Compose([
            al.MotionBlur(p=self.blur_prob),
            al.Rotate(p=self.rotate_prob),
            al.ColorJitter(p=self.jitter_prob),
            al.VerticalFlip(p=self.flip_prob),
            al.HorizontalFlip(p=self.flip_prob)
        ], p=self.prob, **self._transform_kwargs())

    def _transform_kwargs(self) -> dict:
        return {}

    def forward(self, image_t, mask_t):
        image_n = image_t.numpy().transpose(1, 2, 0)
        mask_n = mask_t.numpy().transpose(1, 2, 0)

        result = self.transforms(image=image_n, mask=mask_n)

        image_n, mask_n = result["image"], result["mask"]
        image_t = torch.from_numpy(image_n).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_n).permute(2, 0, 1)

        return image_t, mask_t


class TripletAugmenter(Augmenter):
    def _transform_kwargs(self) -> dict:
        return {
            "additional_targets": {
                "cls": "mask"
            }
        }

    def forward(self, image_t, mask_t, cls_t):
        # Switch to channel-last
        image_n = image_t.numpy().transpose(1, 2, 0)
        mask_n = mask_t.numpy().transpose(1, 2, 0)
        cls_n = cls_t.numpy().transpose(1, 2, 0)

        result = self.transforms(
            image=image_n,
            mask=mask_n,
            cls=cls_n
        )

        image_n, mask_n, cls_n = result["image"], result["mask"], result["cls"]

        image_t = torch.from_numpy(image_n).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_n).permute(2, 0, 1)
        cls_t = torch.from_numpy(cls_n).permute(2, 0, 1)

        return image_t, mask_t, cls_t


class NoOpAugmenter(Augmenter):
    def __init__(self):
        super().__init__(
            prob=0
        )

    def forward(self, image_t, mask_t):
        return image_t, mask_t


class NoOpTripletAugmenter(TripletAugmenter):
    def forward(self, image_t, mask_t, cls_t):
        return image_t, mask_t, cls_t


def test_1():
    augmenter = Augmenter(prob=1)

    image = read_image("data/test_img.png")
    mask = torch.randint_like(image, 0, 255)

    while True:
        img_, mask_ = augmenter(image, mask)
        grid = make_grid([img_, mask_]).numpy().transpose(1, 2, 0)
        plt.imshow(grid)
        plt.show()
        inp = input("Continue? (Y/n) ")

        if inp.lower() == "n":
            break


if __name__ == "__main__":
    test_1()
