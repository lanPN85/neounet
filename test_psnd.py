#!/usr/bin/env python3

import os
import pydoc
import time
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import load

from polypnet.model import UnetClassifyModelWrapper
from polypnet.config import load_config, print_config,\
    load_class_from_conf
from polypnet.callbacks import save_multiclass_image_3tile, save_multiclass_image_mask
from polypnet.data import NoOpTripletAugmenter
from polypnet.data.func import triplet_collate_fn_2
from polypnet.callbacks.csvlog import CsvLogCallback


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("-d", help="Test dataset name", required=True)
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-s", "--save-dir", default=None)
    parser.add_argument("--per-file", default=None)
    parser.add_argument("--skip-img", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    logger.info("Loading config")
    config_paths = ["config/defaults.yml"]
    for cf in args.config:
        config_paths.append(cf)
    config = load_config(
        config_paths
    )

    print_config(config)

    logger.info("Loading model")
    model = load_class_from_conf(config.model)
    optimizer = None
    scheduler = None

    if args.model is not None:
        model_wrapper = UnetClassifyModelWrapper.load_from_checkpoint(
            args.model, model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            strict=False,
            test_names=[args.d],
            **config.wrapper.kwargs
        )
    else:
        model_wrapper = UnetClassifyModelWrapper(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            test_names=[args.d],
            **config.wrapper.kwargs
        )
    model_wrapper.eval()

    if args.gpu:
        model_wrapper.cuda()

    logger.info("Loading test dataset")

    try:
        test_dataset_conf = next(filter(
            lambda x: x.name == args.d, config.test_datasets
        ))
    except StopIteration:
        logger.error(f"No dataset with name '{args.d}' found in config")
        exit(1)

    test_dataset = load_class_from_conf(
        test_dataset_conf, augmenter=NoOpTripletAugmenter(),
        return_paths=True
    )

    # Create result directories
    if not args.skip_img or not args.skip_timing:
        test_dataset2 = load_class_from_conf(
            test_dataset_conf, augmenter=NoOpTripletAugmenter(),
            return_paths=True,
            shape=model_wrapper.test_input_size
        )
        if not args.skip_img and not args.save_dir:
            raise ValueError("No save directory specified")

        if args.save_dir:
            logger.info(f"Writing image samples to {args.save_dir}")
            os.makedirs(args.save_dir, exist_ok=True)

        time_total, time_count = 0, 0

        for image_t, mask_t, cls_t, image_path, mask_path, cls_path in tqdm(test_dataset):
            name = image_path.split("/")[-1]
            name = name.split(".")[0]

            if not args.skip_img:
                save_multiclass_image_3tile(
                    model_wrapper, triplet_collate_fn_2,
                    image_t, mask_t, cls_t,
                    save_path=os.path.join(args.save_dir, f"{name}.tiles.jpg")
                )
                save_multiclass_image_mask(
                    model_wrapper, triplet_collate_fn_2,
                    image_t, mask_t, cls_t,
                    save_path=os.path.join(args.save_dir, f"{name}.mask.jpg")
                )

            if not args.skip_timing:
                image, _, _ = triplet_collate_fn_2(
                    [(image_t, mask_t, cls_t)]
                )
                image = image.to(model_wrapper.device)
                start = time.time()
                model_wrapper(image)
                elapsed = time.time() - start
                time_total += elapsed
                time_count += 1

        if not args.skip_timing:
            time_avg = time_total / time_count
            fps_avg = 1 / time_avg
            logger.info(f"Avg. inference time: {time_avg}s")
            logger.info(f"Avg. FPS: {fps_avg}")

    if not args.skip_metrics:
        logger.info("Calculating test metrics")

        callbacks = []

        if args.per_file is not None:
            callbacks.append(
                CsvLogCallback(
                    save_path=args.per_file
                )
            )

        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=triplet_collate_fn_2,
            batch_size=1,
            **config.test_loader
        )
        trainer = pl.Trainer(
            default_root_dir="results",
            terminate_on_nan=True,
            callbacks=callbacks,
            **config.trainer
        )
        trainer.test(
            model=model_wrapper,
            test_dataloaders=[test_dataloader],
            ckpt_path=args.model
        )

if __name__ == "__main__":
    main()
