import os
import shutil
import random
import torch

from tqdm import tqdm
from loguru import logger

from polypnet.data.lci import LciMulticlassDataset
from polypnet.utils.data_utils import analyze_multiclass_dataset


def main():
    SOURCE = "data/WLIv5_pub/Test"
    TARGET = "data/WLIv5_pub_noud/Test"

    os.makedirs(TARGET, exist_ok=True)
    os.makedirs(os.path.join(TARGET, "images"), exist_ok=True)
    os.makedirs(os.path.join(TARGET, "label_images"), exist_ok=True)
    os.makedirs(os.path.join(TARGET, "mask_images"), exist_ok=True)

    names = []
    # Get file names
    for fname in tqdm(os.listdir(os.path.join(SOURCE, "images"))):
        name = fname.split(".")[0]

        label_path = os.path.join(SOURCE, "label_images", f"{name}.png")
        cls_mask = LciMulticlassDataset.read_cls(label_path)
        red_mask = (cls_mask[0, ...] > 128).int()
        green_mask = (cls_mask[1, ...] > 128).int()
        undefined_mask = red_mask * green_mask

        if torch.sum(undefined_mask) == 0:
            names.append(name)

    for name in names:
        print(name)
        shutil.copy2(
            os.path.join(SOURCE, "images", f"{name}.jpeg"),
            os.path.join(TARGET, "images", f"{name}.jpeg")
        )
        shutil.copy2(
            os.path.join(SOURCE, "label_images", f"{name}.png"),
            os.path.join(TARGET, "label_images", f"{name}.png")
        )
        shutil.copy2(
            os.path.join(SOURCE, "mask_images", f"{name}.png"),
            os.path.join(TARGET, "mask_images", f"{name}.png")
        )

    dataset = LciMulticlassDataset(
        os.path.join(TARGET),
        return_paths=True
    )

    analyze_multiclass_dataset(dataset)


if __name__ == "__main__":
    main()
