import cv2
import numpy as np
import torch

from PIL import Image
from torchvision.io import read_image
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

_Object = namedtuple("Object", [
    "mask", "index"
])


def label_object_count(
    pred_mask, true_mask,
    ignore_mask=None, dim=0
):
    """
    :param pred_mask: Tensor of shape [2 x H x W]
    :param true_mask: Tensor of shape [2 x H x W]
    :param ignore_mask: Tensor of shape [1 x H x W]
    """
    if ignore_mask is None:
        ignore_mask = np.zeros_like(pred_mask)

    true_mask = true_mask * (1 - ignore_mask)
    objects = _objects_from_mask(true_mask)
    objects = list(filter(lambda x: x.index == dim, objects))
    return len(objects)


def pred_object_count(
    pred_mask, true_mask,
    ignore_mask=None, dim=0
):
    """
    :param pred_mask: Tensor of shape [2 x H x W]
    :param true_mask: Tensor of shape [2 x H x W]
    :param ignore_mask: Tensor of shape [1 x H x W]
    """
    if ignore_mask is None:
        ignore_mask = np.zeros_like(pred_mask)

    pred_mask = pred_mask * (1 - ignore_mask)
    objects = _objects_from_mask(pred_mask)
    objects = list(filter(lambda x: x.index == dim, objects))
    return len(objects)


def true_object_count(
    pred_mask, true_mask,
    ignore_mask=None, iou=0.7, dim=0
):
    """
    :param pred_mask: Tensor of shape [2 x H x W]
    :param true_mask: Tensor of shape [2 x H x W]
    :param ignore_mask: Tensor of shape [1 x H x W]
    """
    if ignore_mask is None:
        ignore_mask = np.zeros_like(pred_mask)

    pred_mask = pred_mask * (1 - ignore_mask)
    true_mask = true_mask * (1 - ignore_mask)

    pred_objs = _objects_from_mask(pred_mask)
    true_objs = _objects_from_mask(true_mask)

    pairs = _assign_object_pairs(pred_objs, true_objs)

    count = 0
    for p_obj, t_obj, iou_ in pairs:
        if p_obj.index == t_obj.index and p_obj.index == dim and iou_ >= iou:
            count += 1
    return count


def _assign_object_pairs(
    objects_1: List[_Object],
    objects_2: List[_Object]
) -> List[Tuple[_Object, _Object, float]]:
    iou_mat = np.zeros((len(objects_1), len(objects_2)))

    for i, o1 in enumerate(objects_1):
        for j, o2 in enumerate(objects_2):
            m1 = o1.mask
            m2 = o2.mask
            iou_mat[i, j] = _iou(m1, m2)

    row_ind, col_ind = linear_sum_assignment(iou_mat, maximize=True)

    pairs = []
    for row, col in zip(row_ind, col_ind):
        pairs.append((objects_1[row], objects_2[col], iou_mat[row, col]))
    return pairs


def _iou(m1, m2):
    intersection = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - intersection
    return intersection / union

def _objects_from_mask(
    mask: torch.Tensor,
    mal_ratio_cutoff=0.1
) -> List[_Object]:
    m = mask.cpu().numpy()  # 2 x H x W

    merged_mask = np.logical_or(m[0], m[1]).astype(np.uint8)
    num_objs, objs_mask = cv2.connectedComponents(merged_mask)

    objects = []

    for i in range(num_objs):
        # Find the class for each mask
        omask = (objs_mask == i).astype(np.int)
        ben_count = np.sum(omask * m[0])
        mal_count = np.sum(omask * m[1])
        total_count = np.sum(omask)

        ben_ratio = ben_count / total_count
        mal_ratio = mal_count / total_count

        if (ben_ratio + mal_ratio) < 0.8:
            # Background component
            continue

        index = 0
        if mal_ratio > mal_ratio_cutoff:
            index = 1

        objects.append(_Object(omask, index))

    return objects


def test_1():
    im = Image.open("data/test-1.png").convert("RGB")
    arr = np.array(im).transpose(2, 0, 1)
    img = torch.from_numpy(arr.astype(np.uint8))
    red_mask = (img[0, ...] > 128).int()
    green_mask = (img[1, ...] > 128).int()
    benign_mask = green_mask * (1 - red_mask)
    malign_mask = red_mask * (1 - green_mask)

    mask = torch.stack([benign_mask, malign_mask], dim=0).long()

    h, w = mask.shape[-2:]
    shift = 30
    shift_mask = mask[...]
    shift_mask[:,:h-shift, ...] = mask[:, shift:,...]
    shift_mask[:, h-shift:, ...] = 0

    poc = pred_object_count(shift_mask, mask, dim=1)
    loc = label_object_count(shift_mask, mask, dim=1)
    toc = true_object_count(shift_mask, mask, dim=1)
    print(poc, loc, toc)


if __name__ == "__main__":
    test_1()
