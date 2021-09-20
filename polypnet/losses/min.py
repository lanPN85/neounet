import torch.nn as nn
import torch
import torch.nn.functional as F


class MinMaskLoss(nn.Module):
    def forward(self, pred, *args):
        return torch.mean(
            torch.mean(
                torch.min(
                    torch.sigmoid(pred), dim=1
                ).values, dim=[1, 2]
            )
        )
