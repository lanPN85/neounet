import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_acc(pred_mask, true_mask):
    """
    Per-pixel accuracy

    :param pred_mask: Tensor of shape [B x C x H x W]
    :param true_mask: Tensor of shape [B x C x H x W]
    """
    total = torch.numel(pred_mask)
    correct = torch.sum((pred_mask == true_mask).long())
    return correct / total


def iou(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    """
    Segmentation Intersection over Union.

    :param pred_mask: Tensor of shape [B x C x H x W]
    :param true_mask: Tensor of shape [B x C x H x W]
    :param eps: Epsilon value to prevent divide by zero
    """
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    intersection = torch.sum(pred_mask * true_mask * acc_mask, dim=[2, 3])
    union = torch.sum(pred_mask * acc_mask, dim=[2, 3]) + torch.sum(true_mask * acc_mask, dim=[2, 3]) - intersection

    return torch.mean(
        (intersection) / (union + eps)
    ).item()


def dice_2(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    """
    Dice score (F1)

    :param pred_mask: Tensor of shape [B x C x H x W]
    :param true_mask: Tensor of shape [B x C x H x W]
    :param eps: Epsilon value to prevent divide by zero
    """
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    intersection = torch.sum(pred_mask * true_mask * acc_mask, dim=[2, 3])
    union = torch.sum(pred_mask * acc_mask, dim=[2, 3]) + torch.sum(true_mask * acc_mask, dim=[2, 3])

    return torch.mean(
        torch.mean(
            (2 * intersection + eps) / (union + eps),
            dim=1
        )
    )


def dice(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    prec = precision(pred_mask, true_mask, ignore_mask, eps=eps)
    rec = recall(pred_mask, true_mask, ignore_mask, eps=eps)
    return 2 * (prec * rec) / max(prec + rec, eps)


def precision2(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    """
    Precision score

    :param pred_mask: Tensor of shape [B x C x H x W]
    :param true_mask: Tensor of shape [B x C x H x W]
    :param eps: Epsilon value to prevent divide by zero
    """
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    true_pos = torch.sum(pred_mask * true_mask * acc_mask, dim=[2, 3])
    all_pos = torch.sum((pred_mask == 1) * acc_mask, dim=[2, 3])
    return torch.mean(
        torch.mean(
            (true_pos + eps) / (all_pos + eps),
            dim=1
        )
    )


def precision(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    return true_positive(pred_mask, true_mask, ignore_mask) / (prod_positive(pred_mask, true_mask, ignore_mask) + eps)


def recall2(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    """
    Recall score

    :param pred_mask: Tensor of shape [B x C x H x W]
    :param true_mask: Tensor of shape [B x C x H x W]
    :param eps: Epsilon value to prevent divide by zero
    """
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    true_pos = torch.sum(pred_mask * true_mask * acc_mask, dim=[2, 3])
    all_true = torch.sum((true_mask == 1) * acc_mask, dim=[2, 3])
    return torch.mean(
        (true_pos + eps) / (all_true + eps),
        dim=1
    )


def recall(pred_mask, true_mask, ignore_mask=None, eps=1e-3):
    return true_positive(pred_mask, true_mask, ignore_mask) / (label_positive(pred_mask, true_mask, ignore_mask) + eps)


def true_positive(pred_mask, true_mask, ignore_mask=None):
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    return torch.sum(pred_mask * true_mask * acc_mask).item()


def label_positive(pred_mask, true_mask, ignore_mask=None):
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    return torch.sum((true_mask == 1) * acc_mask).item()


def prod_positive(pred_mask, true_mask, ignore_mask=None):
    if ignore_mask is None:
        ignore_mask = torch.zeros_like(pred_mask)
    acc_mask = 1 - ignore_mask

    return torch.sum((pred_mask == 1) * acc_mask).item()


def test_1():
    pred = (torch.randn((1, 2, 10, 10)) > 0.5).long()
    true = (torch.randn((1, 2, 10, 10)) > 0.5).long()
    d1 = dice(pred, true, eps=1e-5)
    d2 = dice_2(pred, true, eps=1e-5)
    print(d1, d2)


if __name__ == "__main__":
    test_1()
