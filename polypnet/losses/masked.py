import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Callable, Iterable

from .proba import gaussian_kernel


class MaskedBinaryCrossEntropyLoss(nn.Module):
    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        mask = label[:, -1, ...].float()
        mask = torch.stack([mask] * (label.shape[1] - 1), dim=1)

        value = label[:, :-1, ...].float()

        total_entropy = F.binary_cross_entropy_with_logits(
            pred, value.float(), reduction="none"
        )

        masked_entropy = total_entropy * (1 - mask)
        count = torch.numel(masked_entropy) - torch.sum(mask)
        return torch.sum(total_entropy) / count


class MaskedTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, eps=1):
        super().__init__()

        self.eps = eps
        self.alpha = alpha

    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        mask = label[:, -1, ...].float()
        mask = torch.stack([mask] * (label.shape[1] - 1), dim=1)

        value = label[:, :-1, ...].float()

        probs = torch.sigmoid(pred)
        true_pos = torch.sum(probs * value * (1 - mask), dim=[0, 2, 3])
        false_neg = torch.sum(value * (1 - probs) * (1 - mask), dim=[0, 2, 3])
        false_pos = torch.sum(probs * (1 - value) * (1 - mask), dim=[0, 2, 3])
        return 1 - torch.mean(
            (true_pos + self.eps)
            / (
                true_pos
                + self.alpha * false_neg
                + (1 - self.alpha) * false_pos
                + self.eps
            )
        )


class FocalMaskedTverskyLoss(MaskedTverskyLoss):
    def __init__(self, gamma=4 / 3, ben_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.ben_ratio = ben_ratio

    def forward(self, pred, label):
        mask = label[:, -1, ...].float()
        mask = torch.stack([mask] * (label.shape[1] - 1), dim=1)

        value = label[:, :-1, ...].float()

        probs = torch.sigmoid(pred)
        true_pos = torch.sum(probs * value * (1 - mask), dim=[0, 2, 3])
        false_neg = torch.sum(value * (1 - probs) * (1 - mask), dim=[0, 2, 3])
        false_pos = torch.sum(probs * (1 - value) * (1 - mask), dim=[0, 2, 3])

        t = (true_pos + self.eps) / (
            true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.eps
        )

        x = torch.pow(1 - t, 1 / self.gamma)
        x[0] = x[0] * (self.ben_ratio / (self.ben_ratio + 1))
        x[1] = x[1] * (1 / (self.ben_ratio + 1))

        return torch.sum(x, dim=0)


class MaskedSslLoss(pl.LightningModule):
    def __init__(
        self,
        kernel_size=3,
        padding=1,
        classes=2,
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
        mask = label[:, -1, ...].float()
        mask = torch.stack([mask] * (label.shape[1] - 1), dim=1)

        value = label[:, :-1, ...].float()

        pred = torch.sigmoid(pred)
        # pred = torch.softmax(pred, dim=1)

        mean_pred = self.conv(pred)
        mean_label = self.conv(value)

        var_pred = (pred - mean_pred) ** 2
        std_pred = torch.sqrt(self.conv(var_pred) + self.sub_eps)
        var_label = (value - mean_label) ** 2
        std_label = torch.sqrt(self.conv(var_label) + self.sub_eps)

        norm_pred = (pred - mean_pred + self.eps) / (std_pred + self.eps)
        norm_label = (value - mean_label + self.eps) / (std_label + self.eps)

        e = torch.abs(norm_pred - norm_label + self.sub_eps)
        # e = 0.5 * (norm_pred - norm_label) ** 2

        f = (e > self.e_cutoff).int().float()  # B x C x H x W
        M = torch.sum(f, dim=[2, 3])

        loss = f * e * F.binary_cross_entropy(pred, value, reduction="none")
        loss = (torch.sum(loss, dim=[2, 3]) + self.sub_eps) / (M + self.sub_eps)
        loss = torch.mean(loss)
        return loss
