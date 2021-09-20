import torch
import torch.nn as nn
from torchvision import transforms


def generate_scales(input, scales):
    """
    Generate scaled versions of a tensor

    :param input: Input tensor
    :type input: torch.Tensor
    :param scales: List of scales to generate
    :type scales: list
    :return: List of scaled tensors
    """
    height, width = input.shape[-2:]
    outputs = []

    for scale in scales:
        sw = int(width * scale)
        sh = int(height * scale)
        out = transforms.Resize((sh, sw))(input)
        outputs.append(out)

    return outputs


def parse_path_with_patch(path):
    parts = path.split(":")[:2]
    if len(parts) < 2:
        return path, None

    path, patch_info = parts
    x, y, len_x, len_y = patch_info.split("_")
    x, y, len_x, len_y = int(x), int(y), int(len_x), int(len_y)
    return path, (x, y, len_x, len_y)


class Threshold(nn.Module):
    """
    Thresholding transform for image
    """
    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def forward(self, input):
        mask = input >= self.threshold
        return torch.as_tensor(mask, dtype=torch.int)
