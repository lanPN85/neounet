"""
The implementation of the paper:
Region Mutual Information Loss for Semantic Segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from loguru import logger

from . import rmi_utils

_POS_ALPHA = 5e-4  # 	add this factor to ensure the AA^T is positive definite

__all__ = ["RMILoss"]


class RMILoss(pl.LightningModule):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(self, num_classes=1, rmi_radius=3, rmi_pool_way=0, rmi_pool_size=3):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_size

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255

    def forward(self, logits_4D, labels_4D):
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        return loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
                logits_4D 	:	[N, C, H, W], dtype=float32
                labels_4D 	:	[N, C, H, W], dtype=long
        """
        rmi_loss = self.rmi_lower_bound(labels_4D.float(), logits_4D)

        return rmi_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
                labels_4D 	:	[N, C, H, W], dtype=float32
                probs_4D 	:	[N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(
                    labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
                probs_4D = F.max_pool2d(
                    probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(
                    labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
                probs_4D = F.avg_pool2d(
                    probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding
                )
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(
                    labels_4D, size=(new_h, new_w), mode="nearest"
                )
                probs_4D = F.interpolate(
                    probs_4D, size=(new_h, new_w), mode="bilinear", align_corners=True
                )
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = rmi_utils.map_get_pairs(
            labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0
        )

        la_vectors = (
            la_vectors.view([n, c, self.half_d, -1])
            .to(self.device)
            .requires_grad_(False)
        )
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).to(self.device)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(
            la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1)
        )
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        try:
            rmi_now = 0.5 * rmi_utils.log_det_by_cholesky(
                appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA
            )
        except Exception as e:
            logger.error(str(e))
            return torch.scalar_tensor(0, dtype=torch.float, device=self.device)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        # 	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.mean(rmi_per_class)
        return rmi_loss


def test_1():
    classes = 2
    l = RMILoss(num_classes=classes)
    inp = torch.randn((3, classes, 48, 48))
    label = (torch.randn(3, classes, 48, 48) > 0.5).long()

    v = l(inp, label)
    print(v)


if __name__ == "__main__":
    test_1()
