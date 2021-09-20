import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pytorch_lightning as pl

from scipy import ndimage

from .inv_weight import get_inverse_weight_matrix
from polypnet.utils import probs_to_onehot, probs_to_mask


class DiceLoss(nn.Module):
    def __init__(self, eps=1):
        super().__init__()

        self.eps = eps

    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        probs = torch.sigmoid(pred)

        num = 2 * torch.sum(probs * label, dim=[0, 2, 3]) + self.eps
        den = (
            torch.sum(probs, dim=[0, 2, 3]) + torch.sum(label, dim=[0, 2, 3]) + self.eps
        )

        return 1 - torch.mean(num / den)


class InverseWDiceLoss(nn.Module):
    def __init__(self, eps=1):
        super().__init__()

        self.eps = eps

    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        probs = torch.sigmoid(pred)
        w = get_inverse_weight_matrix(label)

        num = 2 * torch.sum(w * probs * label, dim=[0, 2, 3]) + self.eps
        den = (
            torch.sum(w * probs, dim=[0, 2, 3])
            + torch.sum(w * label, dim=[0, 2, 3])
            + self.eps
        )

        return 1 - torch.mean(num / den)


class TverskyLoss(nn.Module):
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
        probs = torch.sigmoid(pred)
        true_pos = torch.sum(probs * label, dim=[0, 2, 3])
        false_neg = torch.sum(label * (1 - probs), dim=[0, 2, 3])
        false_pos = torch.sum(probs * (1 - label), dim=[0, 2, 3])
        return 1 - torch.mean(
            (true_pos + self.eps)
            / (
                true_pos
                + self.alpha * false_neg
                + (1 - self.alpha) * false_pos
                + self.eps
            )
        )


class FocalTverskyLoss(TverskyLoss):
    def __init__(self, gamma=4 / 3, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, pred, label):
        probs = torch.sigmoid(pred)
        true_pos = torch.sum(probs * label, dim=[0, 2, 3])
        false_neg = torch.sum(label * (1 - probs), dim=[0, 2, 3])
        false_pos = torch.sum(probs * (1 - label), dim=[0, 2, 3])

        t = (true_pos + self.eps) / (
            true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.eps
        )

        x = torch.pow(1 - t, 1 / self.gamma)

        return torch.sum(x)


class InverseWTverskyLoss(nn.Module):
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
        probs = torch.sigmoid(pred)
        w = get_inverse_weight_matrix(label)

        true_pos = torch.sum(w * probs * label, dim=[0, 2, 3])
        false_neg = torch.sum(w * label * (1 - probs), dim=[0, 2, 3])
        false_pos = torch.sum(w * probs * (1 - label), dim=[0, 2, 3])
        return 1 - torch.mean(
            (true_pos + self.eps)
            / (
                true_pos
                + self.alpha * false_neg
                + (1 - self.alpha) * false_pos
                + self.eps
            )
        )


class BoundaryWeightedLoss(nn.Module):
    def forward(self, pred, label):
        """
        Forward function

        :param pred: Prediction tensor containing raw network outputs (no logit) (B x C x H x W)
        :param label: Label mask tensor (B x C x H x W)
        """
        onehot_label = probs_to_onehot(label)
        probs = torch.sigmoid(pred)
        mask = probs_to_onehot(probs)
        F_ = torch.sum(mask != onehot_label)

        tn = (1 - mask) * (1 - onehot_label)
        non_tn = 1 - tn
        non_tn_dist = self.distance_transform(non_tn)
        fp_alpha = 1 - non_tn_dist

        tp = mask * onehot_label
        non_tp = 1 - tp
        non_tp_dist = self.distance_transform(non_tp)
        fn_alpha = 1 - non_tp_dist

        alpha = torch.zeros_like(probs)
        is_fp = (mask != onehot_label) * (mask == 1)
        is_fn = (mask != onehot_label) * (mask == 0)
        alpha = alpha + is_fp * fp_alpha + is_fn * fn_alpha

        ce = F.binary_cross_entropy(probs, label.float(), reduction="none")
        loss = torch.sum(alpha * ce) / F_
        return loss

    @staticmethod
    def distance_transform(t: torch.Tensor, normalize=True) -> torch.Tensor:
        array = t.cpu().numpy().astype("float64")
        output = np.zeros_like(array)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ndimage.distance_transform_edt(array[i, j], distances=output[i, j])
                if normalize:
                    output[i, j] /= np.max(output[i, j])

        out_tensor = torch.from_numpy(output).type(torch.float32).to(t.device)
        return out_tensor


def test_1():
    a = torch.randn((5, 2, 112, 112))
    b = (torch.randn((5, 2, 112, 112)) > 0.5).long()
    l = TverskyLoss()(a, b)
    print(l)


def test_2():
    a = torch.randn((5, 1, 112, 112))
    b = (torch.randn((5, 1, 112, 112)) > 0.5).long()
    l = BoundaryWeightedLoss()(a, b)
    print(l)


if __name__ == "__main__":
    test_2()
