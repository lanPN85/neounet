import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .contextagg import (
    LocalAttenModule,
    CrissCrossAttention,
    RCCAModule,
)
from torch.nn import BatchNorm2d, BatchNorm1d
from .gcpa_gald import FAMSCWS
from .encoders import hardnet


class SCWSRCCANet(nn.Module):
    @property
    def output_scales(self):
        return 1.0, 1.0, 1.0, 1.0

    def __init__(self, num_classes=1, pretrained=True):
        super(SCWSRCCANet, self).__init__()

        self.hardnet = hardnet(arch=68, pretrained=pretrained)

        inplanes = 1024
        interplanes = 256
        self.num_classes = num_classes

        self.fam45 = FAMSCWS(640, interplanes, 1024, interplanes)
        self.fam34 = FAMSCWS(320, interplanes, interplanes, interplanes)
        self.fam23 = FAMSCWS(128, interplanes, interplanes, interplanes)

        self.linear5 = nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(
            interplanes, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.linear3 = nn.Conv2d(
            interplanes, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.linear2 = nn.Conv2d(
            interplanes, num_classes, kernel_size=3, stride=1, padding=1
        )

        self.conva = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
        )

        self.long_relation = RCCAModule(inplanes, interplanes)
        self.local_attention_4 = LocalAttenModule(interplanes)
        self.local_attention_3 = LocalAttenModule(interplanes)
        self.local_attention_2 = LocalAttenModule(interplanes)

    def set_num_classes(self, num_classes: int):
        interplanes = 256
        self.num_classes = num_classes

        self.linear5 = nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(
            interplanes, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.linear3 = nn.Conv2d(
            interplanes, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.linear2 = nn.Conv2d(
            interplanes, num_classes, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        hardnetout = self.hardnet(x)

        out2 = hardnetout[0]  # [bs, 128, 88, 88]
        out3 = hardnetout[1]  # [bs, 320, 44, 44]
        out4 = hardnetout[2]  # [bs, 640, 22, 22]
        out5 = hardnetout[3]  # [bs, 1024, 11, 11]

        # RCCA
        out5_c = self.long_relation(out5)  # bs, 256, 11, 11

        # GCF
        out4_c = self.local_attention_4(out5_c)  # bs, 256, 11, 11

        out3_c = self.local_attention_3(out5_c)  # bs, 256, 11, 11

        out2_c = self.local_attention_2(out5_c)  # bs, 256, 11, 11

        # Decoder
        out4 = self.fam45(out4, out4_c, out5)
        out3 = self.fam34(out3, out3_c, out4)
        out2 = self.fam23(out2, out2_c, out3)

        # we use bilinear interpolation instead of transpose convolution
        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode="bilinear")
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode="bilinear")
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode="bilinear")
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode="bilinear")

        return out2, out3, out4, out5


class SCWSRCCANetMergeScales(SCWSRCCANet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.merge = nn.Conv2d(self.num_classes * 4, self.num_classes, kernel_size=1)

    @property
    def output_scales(self):
        return 1.0, 1.0, 1.0, 1.0, 1.0

    def set_num_classes(self, num_classes: int):
        super().set_num_classes(num_classes)

        self.merge = nn.Conv2d(self.num_classes * 4, self.num_classes, kernel_size=1)

    def forward(self, x):
        out2, out3, out4, out5 = super().forward(x)

        merged = torch.cat([out2, out3, out4, out5], dim=1)
        out_merge = self.merge(merged)

        return out_merge, out2, out3, out4, out5


def test_1():
    x = torch.randn((5, 3, 224, 224))
    net = SCWSRCCANetMergeScales(pretrained=True)
    outputs = net(x)

    for i, o in enumerate(outputs):
        print(f"out-{i}: {o.shape}")


if __name__ == "__main__":
    test_1()
