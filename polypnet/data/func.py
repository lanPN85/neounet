import torch
import torch.nn as nn

from torchvision import transforms as ttf
from loguru import logger
from polypnet.utils import Threshold


def mask_collate_fn(batch):
    image_t = torch.stack([
        b[0] for b in batch
    ])
    mask_t = torch.stack([
        b[1] for b in batch
    ], dim=0)  # B x 3 x H x W

    image_t = image_t.float() / 255.

    mask_t = ttf.Compose([
        ttf.Grayscale(num_output_channels=1),
        Threshold(threshold=128),
    ])(mask_t)
    mask_t = mask_t.long()

    return image_t, mask_t


def triplet_collate_fn(batch):
    image_t = torch.stack([
        b[0] for b in batch
    ])
    mask_t = torch.stack([
        b[1] for b in batch
    ], dim=0)  # B x 3 x H x W
    cls_t = torch.stack([
        b[2] for b in batch
    ], dim=0)  # B x 3 x H x W

    image_t = image_t.float() / 255.

    mask_t = ttf.Compose([
        ttf.Grayscale(num_output_channels=1),
        Threshold(threshold=128),
    ])(mask_t)
    mask_t = mask_t.long()

    red_mask = cls_t[:, 0, ...] > 128  # Undefined
    green_mask = cls_t[:, 1, ...] > 128  # Benign
    blue_mask = cls_t[:, 2, ...] > 128  # Malign
    cls_t = torch.stack([green_mask, blue_mask, red_mask], dim=1).long()

    return image_t, mask_t, cls_t


def triplet_collate_fn_2(batch):
    image_t = torch.stack([
        b[0] for b in batch
    ])
    mask_t = torch.stack([
        b[1] for b in batch
    ], dim=0)  # B x 3 x H x W
    cls_t = torch.stack([
        b[2] for b in batch
    ], dim=0)  # B x 3 x H x W

    image_t = image_t.float() / 255.

    mask_t = ttf.Compose([
        ttf.Grayscale(num_output_channels=1),
        Threshold(threshold=128),
    ])(mask_t)
    mask_t = mask_t.long()

    red_mask = (cls_t[:, 0, ...] > 128).int()
    green_mask = (cls_t[:, 1, ...] > 128).int()

    undefined_mask = red_mask * green_mask
    benign_mask = green_mask * (1 - red_mask)
    malign_mask = red_mask * (1 - green_mask)

    cls_t = torch.stack([benign_mask, malign_mask, undefined_mask], dim=1).long()

    out = [image_t, mask_t, cls_t]

    if len(batch[0]) > 3:
        ids = [
            b[3].split("/")[-1].split(".")[0]
            for b in batch
        ]
        out.append(ids)

    return out
