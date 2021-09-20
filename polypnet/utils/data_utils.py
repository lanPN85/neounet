import torch

from tqdm import tqdm
from loguru import logger

from polypnet.data.base import PolypMulticlassDataset
from polypnet.data.lci import LciMulticlassDataset


def analyze_multiclass_dataset(dataset: PolypMulticlassDataset):
    ben_count, mal_count, undefined_count = 0, 0, 0
    for item in tqdm(dataset):
        cls_t = item[2]

        red_mask = (cls_t[0, ...] > 128).int()
        green_mask = (cls_t[1, ...] > 128).int()

        undefined_mask = red_mask * green_mask
        benign_mask = green_mask * (1 - red_mask)
        malign_mask = red_mask * (1 - green_mask)

        ben_count += torch.sum(benign_mask).item()
        mal_count += torch.sum(malign_mask).item()
        undefined_count += torch.sum(undefined_mask).item()

    total_count = ben_count + mal_count + undefined_count
    ben_pct = ben_count / total_count * 100
    mal_pct = mal_count / total_count * 100
    undefined_pct = undefined_count / total_count * 100

    print()
    logger.debug(f"Benign: {ben_count} ({ben_pct:.2f}%)")
    logger.debug(f"Malicious: {mal_count} ({mal_pct:.2f}%)")
    logger.debug(f"Undefined: {undefined_count} ({undefined_pct:.2f}%)")


def test_1():
    dataset = LciMulticlassDataset("data/WLIv5/", shape=None)
    analyze_multiclass_dataset(dataset)


if __name__ == "__main__":
    test_1()
