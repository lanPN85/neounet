import torch
import random
import torch.nn as nn

from typing import Iterable
from torchvision import transforms as ttf


class PairedTransform(nn.Module):
    def __init__(self, transforms: Iterable):
        super().__init__()

        self.transforms = transforms
