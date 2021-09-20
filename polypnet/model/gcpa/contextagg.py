from torch._C import device
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import pytorch_lightning as pl

from torch.nn import Softmax, BatchNorm2d
from collections import OrderedDict, defaultdict


class LocalAttenModule(nn.Module):
    def __init__(self, inplane):
        super(LocalAttenModule, self).__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False),
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False),
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        res1 = x
        res2 = x

        x = self.dconv1(x)
        x = self.dconv2(x)
        # x = self.dconv3(x)
        x = F.upsample(x, size=(h, w), mode="bilinear", align_corners=True)
        x_mask = self.sigmoid_spatial(x)

        res1 = res1 * x_mask

        return res2 + res1


def INF(B, H, W, device="cpu"):
    return (
        -torch.diag(
            torch.tensor(float("inf"))
            .to(device).repeat(H), 0
        )
        .unsqueeze(0)
        .repeat(B * W, 1, 1)
    )


class CrissCrossAttention(pl.LightningModule):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = (
            proj_query.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
            .permute(0, 2, 1)
        )
        proj_query_W = (
            proj_query.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
            .permute(0, 2, 1)
        )
        proj_key = self.key_conv(x)
        proj_key_H = (
            proj_key.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_key_W = (
            proj_key.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        proj_value = self.value_conv(x)
        proj_value_H = (
            proj_value.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_value_W = (
            proj_value.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        energy_H = (
            (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width, device=self.device))
            .view(m_batchsize, width, height, height)
            .permute(0, 2, 1, 3)
        )
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(
            m_batchsize, height, width, width
        )
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = (
            concate[:, :, :, 0:height]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * width, height, height)
        )
        # print(concate)
        # print(att_H)
        att_W = (
            concate[:, :, :, height : height + width]
            .contiguous()
            .view(m_batchsize * height, width, width)
        )
        out_H = (
            torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
            .view(m_batchsize, width, -1, height)
            .permute(0, 2, 3, 1)
        )
        out_W = (
            torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
            .view(m_batchsize, height, -1, width)
            .permute(0, 2, 1, 3)
        )
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=1):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            # InPlaceABNSync(inter_channels),
        )
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            # InPlaceABNSync(inter_channels),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + inter_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            # InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            # nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size) for size in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features + len(sizes) * out_features,
                out_features,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.upsample(
                input=stage(feats), size=(h, w), mode="bilinear", align_corners=True
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
