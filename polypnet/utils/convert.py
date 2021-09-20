import torch
import torch.nn as nn
import torch.nn.functional as F


def probs_to_mask(probs, thres=0.5):
    """
    Convert a probability tensor into a class mask

    :param probs: Tensor of shape [B x C x H x W], values from 0 to 1
    :return: Tensor of shape [B x H x W], each pixel corresponds to a class
    """
    num_classes = probs.shape[1]

    if num_classes > 1:
        return torch.argmax(probs, dim=1)
    else:
        return (probs > thres).int()


def probs_to_onehot(probs, thres=0.5):
    """
    Converts a [B x C x H x W] probability map to a one-hot encoded matrix

    :param probs: The input mask
    :return: Output one-hot mask
    """
    num_classes = probs.shape[1]
    if num_classes > 1:
        indices = torch.argmax(probs, dim=1)
        return F.one_hot(indices, num_classes).permute(0, 3, 1, 2)
    else:
        return (probs > thres).int()
