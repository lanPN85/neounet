import torch
import cv2
import numpy as np


def get_inverse_weight_matrix(t: torch.Tensor):
    """
    Produces an inverse weight matrix based on each connected component's size

    :param t: The input tensor (B x C x H x W)
    :type t: torch.Tensor
    """
    label_np = t.cpu().numpy()
    label_np = label_np.astype(np.uint8)
    weights = np.ones_like(label_np).astype(np.float)
    B, C, H, W = label_np.shape

    for b in range(B):
        # Iterate through batch
        for c in range(C):
            # One for each channel

            img = label_np[b, c]  # H x W
            num_labels, labels = cv2.connectedComponents(img)  # H x W

            sizes = {}
            # Find size for each component
            for lb in range(num_labels):
                mask = labels == lb
                size = np.sum(mask)
                sizes[lb] = size

            sum_size = H * W

            # Assign weight
            for lb in range(num_labels):
                mask = (labels == lb).astype(np.float)
                w = sum_size / (max(num_labels, 1) * max(sizes[lb], 1))
                weights[b, c] += mask * w

    return torch.from_numpy(weights).to(t.device)


if __name__ == "__main__":
    t = (torch.randn((2, 1, 20, 20)) > 0.5).int()
    w = get_inverse_weight_matrix(t)
    print(w.shape)
