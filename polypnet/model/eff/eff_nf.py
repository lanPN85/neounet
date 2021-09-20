import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchvision.models import ResNet
from loguru import logger
from typing import Callable, Optional

from .nf_efficientnet import NFEfficientNet
from polypnet.model.attn import AdditiveAttnGate
from .base import EfficientNetUnet


class NFEfficientNetUnet(EfficientNetUnet):
    def __init__(self, backbone_name="efficientnet-b0", num_classes=1):
        super().__init__(backbone_name=backbone_name, num_classes=num_classes)

    def _init_encoder(self):
        self.encoder = NFEfficientNet.from_pretrained(self.backbone_name)

    def _decoder_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        """Forward method

        :param inputs: Input tensor of size (B x C x W x H)
        :type inputs: list
        :return: List of output for each level
        """

        # Run input through encoder
        endpoints = self.encoder.extract_endpoints(inputs)
        encode_4 = endpoints["reduction_4"]  # B x 112 x H16 x W16
        encode_3 = endpoints["reduction_3"]  # B x 40 x H8 x W8
        encode_2 = endpoints["reduction_2"]  # B x 24 x H4 x W4
        encode_1 = endpoints["reduction_1"]  # B x 16 x H2 x W2

        # Upsample middle block
        middle_block = endpoints["reduction_5"]  # B x 1280 x H32 x W32
        attn_middle = self.attn_mid(middle_block, encode_4)  # B x 1280 x H32 x W32
        up_middle = self.mid_upsampler(attn_middle)  # B x 112 x H16 x W16

        # Level 4
        merged_4 = torch.cat((encode_4, up_middle), dim=1)  # B x 224 x H16 x W16
        decode_4 = self.decode_4(merged_4)  # B x 40 x H16 x W16
        attn_4 = self.attn_4(decode_4, encode_3)  # B x 40 x H16 x W16
        out_4 = self.out_4(decode_4)  # B x 2 x H16 x W16
        up_4 = self.ups_4(attn_4)  # B x 40 x H8 x W8

        # Level 3
        merged_3 = torch.cat((encode_3, up_4), dim=1)  # B x 80 x H8 x W8
        decode_3 = self.decode_3(merged_3)  # B x 40 x H8 x W8
        attn_3 = self.attn_3(decode_3, encode_2)  # B x 40 x H8 x W8
        out_3 = self.out_3(decode_3)  # B x 2 x H8 x W8
        up_3 = self.ups_3(attn_3)  # B x 24 x H4 x W4

        # Level 2
        merged_2 = torch.cat((encode_2, up_3), dim=1)  # B x 48 x H4 x W4
        decode_2 = self.decode_2(merged_2)  # B x 24 x H4 x W4
        attn_2 = self.attn_2(decode_2, encode_1)  # B x 24 x H4 x W4
        out_2 = self.out_2(decode_2)  # B x 2 x H4 x W4
        up_2 = self.ups_2(attn_2)  # B x 16 x H2 x W2

        # Level 1
        merged_1 = torch.cat((encode_1, up_2), dim=1)  # B x 32 x H2 x W2
        decode_1 = self.decode_1(merged_1)  # B x 16 x H2 x W2
        out_1 = self.out_1(decode_1)  # B x 2 x H2 x W2
        up_1 = self.ups_1(decode_1)  # B x 8 x H x W

        # Level 0
        out_0 = self.out_0(up_1)  # B x C x H x W

        return out_0, out_1, out_2, out_3, out_4


def test_1():
    x = torch.randn((5, 3, 224, 224))
    net = NFEfficientNetUnet(backbone_name="efficientnet-b0")
    outputs = net(x)

    for i, o in enumerate(outputs):
        print(f"out-{i}: {o.shape}")


if __name__ == "__main__":
    test_1()
