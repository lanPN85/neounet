#!/usr/bin/env python3

import os
import pydoc
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load

from polypnet.model import UnetModelWrapper
from polypnet.config import load_config, print_config, load_class_from_conf
from polypnet.callbacks import (
    MLFlowModelCheckpoint,
    ResultSampleCallback,
    save_infer_image_3tile,
)
from polypnet.data import mask_collate_fn, Augmenter, NoOpAugmenter


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("-d", help="Test dataset name", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-s", "--save-dir", default=None)
    parser.add_argument("--skip-img", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    logger.info("Loading config")
    config_paths = ["config/defaults.yml"]
    for cf in args.config:
        config_paths.append(cf)
    config = load_config(config_paths)

    print_config(config)

    logger.info("Loading model")
    model = load_class_from_conf(config.model)
    optimizer = load_class_from_conf(config.optimizer, model.parameters())
    scheduler = load_class_from_conf(config.scheduler, optimizer=optimizer)
    model_wrapper = UnetModelWrapper.load_from_checkpoint(
        args.model,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        test_names=[args.d],
        **config.wrapper.kwargs,
    )
    model_wrapper.eval()

    logger.info("Loading test dataset")

    try:
        test_dataset_conf = next(
            filter(lambda x: x.name == args.d, config.test_datasets)
        )
    except StopIteration:
        logger.error(f"No dataset with name '{args.d}' found in config")
        exit(1)

    test_dataset = load_class_from_conf(
        test_dataset_conf, augmenter=NoOpAugmenter(), return_paths=True, shape=None
    )

    # Create result directories
    if not args.skip_img:
        if not args.save_dir:
            raise ValueError("No save directory specified")
        logger.info(f"Writing image samples to {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
        for image_t, mask_t, image_path, mask_path in tqdm(test_dataset):
            name = image_path.split("/")[-1]
            name = name.split(".")[0]
            save_infer_image_3tile(
                model_wrapper,
                mask_collate_fn,
                image_t,
                mask_t,
                save_path=os.path.join(args.save_dir, f"{name}.tiles.jpg"),
            )

    if not args.skip_metrics:
        logger.info("Calculating test metrics")
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=mask_collate_fn,
            batch_size=1,
            **config.test_loader,
        )
        trainer = pl.Trainer(
            default_root_dir="results", terminate_on_nan=True, **config.trainer
        )
        trainer.test(
            model=model_wrapper, test_dataloaders=[test_dataloader], ckpt_path=None
        )


if __name__ == "__main__":
    main()
