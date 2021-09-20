import torch.nn as nn
import torch.nn.functional as F
import torch
import math

import pytorch_lightning as pl

from .inv_weight import get_inverse_weight_matrix


class BinaryCrossEntropyLoss(nn.Module):
    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        return F.binary_cross_entropy_with_logits(pred, label.float())


class StructuralSimilarityLoss(pl.LightningModule):
    """
    SSL as described in 'Correlation Maximized Structural Similarity Loss for Semantic Segmentation'
    """

    def __init__(
        self,
        kernel_size=3,
        padding=1,
        classes=1,
        std=1.5,
        eps=1e-2,
        e_cutoff=0.3,
        dilation=1,
        sub_eps=1e-8,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            classes,
            classes,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            groups=classes,
            dilation=dilation,
        )
        weight = gaussian_kernel(kernel_size, std=std, channels=classes, dim=2)
        self.conv.weight = nn.parameter.Parameter(weight, requires_grad=False)

        self.eps = eps
        self.sub_eps = sub_eps
        self.e_cutoff = e_cutoff

    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        pred = torch.sigmoid(pred)
        # pred = torch.softmax(pred, dim=1)
        label = label.float()

        mean_pred = self.conv(pred)
        mean_label = self.conv(label)

        var_pred = (pred - mean_pred) ** 2
        std_pred = torch.sqrt(self.conv(var_pred) + self.sub_eps)
        var_label = (label - mean_label) ** 2
        std_label = torch.sqrt(self.conv(var_label) + self.sub_eps)

        norm_pred = (pred - mean_pred + self.eps) / (std_pred + self.eps)
        norm_label = (label - mean_label + self.eps) / (std_label + self.eps)

        e = torch.abs(norm_pred - norm_label + self.sub_eps)
        # e = 0.5 * (norm_pred - norm_label) ** 2

        f = (e > self.e_cutoff).int().float()  # B x C x H x W
        M = torch.sum(f, dim=[2, 3])

        loss = f * e * F.binary_cross_entropy(pred, label, reduction="none")
        loss = (torch.sum(loss, dim=[2, 3]) + self.sub_eps) / (M + self.sub_eps)
        loss = torch.mean(loss)
        return loss


def gaussian_kernel(kernel_size, std=2.0, dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.

    kernel_size = [kernel_size] * dim
    std = [std] * dim
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )

    for size, std_, mgrid in zip(kernel_size, std, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std_ * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / (2 * std_)) ** 2))
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    kernel = torch.stack([kernel] * channels).unsqueeze(1)

    return kernel


def test_1():
    classes = 2
    l = StructuralSimilarityLoss(classes=classes)
    inp = torch.randn((3, classes, 48, 48))
    label = (torch.randn(3, classes, 48, 48) > 0.5).long()

    v = l(inp, label)
    print(v)


if __name__ == "__main__":
    test_1()
