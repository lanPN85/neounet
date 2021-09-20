import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from loguru import logger
from efficientnet_pytorch import EfficientNet
from typing import Callable, Optional

from polypnet.model.attn import AdditiveAttnGate


EFF_DEPTH_MAP = {
    "efficientnet-b0": (1280, 112, 40, 24, 16, 8),
    "efficientnet-b1": (1280, 112, 40, 24, 16, 8),
    "efficientnet-b2": (1408, 120, 48, 24, 16, 8),
    "efficientnet-b3": (1536, 136, 48, 32, 24, 12),
    "efficientnet-b4": (1792, 160, 56, 32, 24, 12),
    "efficientnet-b6": (2304, 200, 72, 40, 32, 12),
}


class EfficientNetUnet(pl.LightningModule):
    def __init__(self,
        backbone_name="efficientnet-b0",
        num_classes=1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name

        self.d5, self.d4, self.d3, self.d2, self.d1, self.d0 = self._block_depths(backbone_name)

        self._init_encoder()

        self._init_upsamplers()

        self.decode_4 = self._decoder_block(self.d4 * 2, self.d4)
        self.decode_3 = self._decoder_block(self.d3 * 2, self.d3)
        self.decode_2 = self._decoder_block(self.d2 * 2, self.d2)
        self.decode_1 = self._decoder_block(self.d1 * 2, self.d1)
        self.decode_0 = nn.Sequential(
            nn.Conv2d(self.d0, self.d0 // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.d0 // 2),
            nn.LeakyReLU()
        )

        self.out_4 = self._out_block(self.d4)
        self.out_3 = self._out_block(self.d3)
        self.out_2 = self._out_block(self.d2)
        self.out_1 = self._out_block(self.d1)
        self.out_0 = self._out_block(self.d0)

        self._init_attn_blocks()

    def set_num_classes(self, num_classes: int):
        self.num_classes = num_classes

        self.out_4 = self._out_block(self.d4)
        self.out_3 = self._out_block(self.d3)
        self.out_2 = self._out_block(self.d2)
        self.out_1 = self._out_block(self.d1)
        self.out_0 = self._out_block(self.d0)

    def _init_encoder(self):
        self.encoder = EfficientNet.from_pretrained(self.backbone_name)

    def _init_upsamplers(self):
        self.mid_upsampler = nn.ConvTranspose2d(in_channels=self.d5, out_channels=self.d4, kernel_size=4, stride=2, padding=1, bias=False)
        self.ups_4 = self._upsampler_block(in_channels=self.d4, out_channels=self.d3)
        self.ups_3 = self._upsampler_block(in_channels=self.d3, out_channels=self.d2)
        self.ups_2 = self._upsampler_block(in_channels=self.d2, out_channels=self.d1)
        self.ups_1 = self._upsampler_block(in_channels=self.d1, out_channels=self.d0)

    def _init_attn_blocks(self):
        self.attn_mid = AdditiveAttnGate(self.d5, self.d4)
        self.attn_4 = AdditiveAttnGate(self.d4, self.d3)
        self.attn_3 = AdditiveAttnGate(self.d3, self.d2)
        self.attn_2 = AdditiveAttnGate(self.d2, self.d1)

    def _block_depths(self, backbone_name):
        return EFF_DEPTH_MAP[backbone_name]

    def _out_block(self, in_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1),
        )

    def _decoder_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def _upsampler_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    @property
    def output_scales(self):
        return 1., 1/2, 1/4, 1/8, 1/16
