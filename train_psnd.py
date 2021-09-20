#!/usr/bin/env python

import shutup
shutup.please()

import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from loguru import logger

from polypnet.utils.data_utils import analyze_multiclass_dataset
from polypnet.model import UnetModelWrapper, UnetClassifyModelWrapper
from polypnet.config import (
    load_config,
    print_config,
    load_class_from_conf,
    read_mlflow_auth,
)
from polypnet.callbacks import (
    MLFlowModelCheckpoint,
    ResultSampleCallback,
    CsvLogCallback,
)
from polypnet.data import triplet_collate_fn_2, TripletAugmenter, NoOpTripletAugmenter
from polypnet.losses import MultiscaleLoss, CompoundLoss, ConditionalCompoundLoss
from polypnet.optim.agc import AGC
from polypnet.utils import log_exp_hyperparams


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("-n", "--name", default=None)
    args = parser.parse_args()

    logger.info("Loading config")
    config_paths = ["config/defaults.yml"]
    for cf in args.config:
        config_paths.append(cf)
    config = load_config(config_paths)

    print_config(config)

    run_name = args.name or config.name

    logger.info("Loading losses")
    losses = []
    for c in config.loss.losses:
        l = load_class_from_conf(c)
        losses.append(l)
    seg_loss = MultiscaleLoss(ConditionalCompoundLoss(losses, **config.loss.kwargs))
    logger.info(f"Using {len(losses)} loss functions for segmentation")

    losses = []
    for c in config.label_loss.losses:
        l = load_class_from_conf(c)
        losses.append(l)
    cls_loss = MultiscaleLoss(
        ConditionalCompoundLoss(losses, **config.label_loss.kwargs)
    )

    logger.info("Loading model")
    model = load_class_from_conf(config.model)

    if config.from_segment_model is not None:
        logger.info(
            f"Using weights from segmentation model at '{config.from_segment_model}'"
        )
        model.set_num_classes(1)
        seg_wrapper = UnetModelWrapper.load_from_checkpoint(
            config.from_segment_model,
            strict=False,
            model=model,
            optimizer=None,
            lr_scheduler=None,
            map_location="cpu",
        )
        model = seg_wrapper.model
        del seg_wrapper
    model.set_num_classes(2)

    logger.info("Loading optimizer")
    optimizer = load_class_from_conf(config.optimizer, model.parameters())

    if config.agc.enabled:
        logger.info("Using AGC")
        optimizer = AGC(model.parameters(), optimizer, **config.agc.kwargs)

    logger.info("Loading scheduler")
    scheduler = load_class_from_conf(config.scheduler, optimizer=optimizer)

    model_wrapper = UnetClassifyModelWrapper(
        model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        seg_loss_fn=seg_loss,
        cls_loss_fn=cls_loss,
        val_on_test=config.val_on_test,
        test_names=[x.name for x in config.test_datasets],
        **config.wrapper.kwargs,
    )

    # Load augmenter
    train_augmenter = TripletAugmenter(**config.augment.kwargs)

    logger.info("Loading train dataset")

    dataset = load_class_from_conf(
        config.dataset, augmenter=train_augmenter, return_paths=True
    )
    raw_dataset = load_class_from_conf(
        config.dataset, augmenter=NoOpTripletAugmenter(), return_paths=True
    )
    if not config.val_on_test:
        if config.val_dataset.split_from_train:
            logger.info("Splitting validation set")
            train_len = int(len(dataset) * (1 - config.val_dataset.split_ratio))
            val_len = len(dataset) - train_len
            train_dataset, val_dataset = random_split(
                dataset,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(config.val_dataset.split_seed),
            )
            val_dataset.dataset = raw_dataset
        else:
            train_dataset = dataset
            val_dataset = load_class_from_conf(
                config.val_dataset, augmenter=NoOpTripletAugmenter(), return_paths=True
            )
    else:
        train_dataset = dataset

    # Load test dataloaders
    test_loaders = []
    for test_dataset_conf in config.test_datasets:
        logger.info(f"Loading test dataset '{test_dataset_conf.name}'")
        test_dataset = load_class_from_conf(
            test_dataset_conf, augmenter=NoOpTripletAugmenter(), return_paths=True
        )
        test_loaders.append(
            DataLoader(
                test_dataset,
                shuffle=False,
                pin_memory=True,
                collate_fn=triplet_collate_fn_2,
                batch_size=1,
                **config.test_loader,
            )
        )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        collate_fn=triplet_collate_fn_2,
        **config.loader,
    )

    if config.val_on_test:
        val_loader = test_loaders
    else:
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            pin_memory=True,
            collate_fn=triplet_collate_fn_2,
            **config.loader,
        )

    # Create result directories
    result_dir = os.path.join("results", run_name)
    ckpoint_dir = os.path.join(result_dir, "checkpoints")
    sample_dir = os.path.join(result_dir, "samples")
    os.makedirs(result_dir, exist_ok=True)

    # Setup callbacks
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]

    if not config.val_on_test:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val.label.dice", mode="max", **config.callbacks.early_stop
            )
        )

    # Default logger when no MLflow available
    _logger = pl.loggers.CSVLogger(result_dir, name="csv")

    # Shared arguments for ModelCheckpoint
    if not config.val_on_test:
        common_ckpoint_args = dict(
            dirpath=ckpoint_dir,
            save_last=True,
            save_top_k=5,
            monitor="val.label.dice",
            mode="max",
            verbose=True,
        )
    else:
        common_ckpoint_args = dict(
            dirpath=ckpoint_dir, save_last=True, verbose=True
        )

    mlflow_logger = None
    if config.with_mlflow:
        # Get mlflow username and password
        mlflow_dir = os.path.join(result_dir, "mlruns")
        mlflow_url = read_mlflow_auth()

        if mlflow_url is not None:
            logger.info("Using remote MLflow")
        else:
            logger.warning(
                "No auth/mlflow.yml file found. Add this file containing your username and password to use remote MLflow. See auth/mlflow.example.yml for details"
            )

        # Setup MLflow logger
        tags = config.tags
        if tags is None:
            tags = {}
        tags["mlflow.runName"] = run_name
        _logger = pl.loggers.MLFlowLogger(
            experiment_name="Default",
            save_dir=mlflow_dir,
            tracking_uri=mlflow_url,
            tags=tags,
        )
        mlflow_logger = _logger

        callbacks.append(MLFlowModelCheckpoint(_logger, **common_ckpoint_args))
    else:
        logger.info("Not using MLflow")

    callbacks.append(pl.callbacks.ModelCheckpoint(**common_ckpoint_args))

    # Add callback for saving result sample
    callbacks.append(
        ResultSampleCallback(
            sample_dir,
            mlflow_logger=mlflow_logger,
            collate_fn=triplet_collate_fn_2,
            mode="label",
            **config.callbacks.result_sample,
        )
    )

    # Add callback for CSV detailed log
    callbacks.append(
        CsvLogCallback(
            os.path.join(result_dir, "per-image.csv"), mlflow_logger=mlflow_logger
        )
    )

    # Initialize trainer
    trainer = pl.Trainer(
        logger=_logger,
        callbacks=callbacks,
        default_root_dir=result_dir,
        terminate_on_nan=True,
        **config.trainer,
    )
    seg_loss.loss_fn.trainer = trainer
    cls_loss.loss_fn.trainer = trainer

    if config.with_mlflow:
        # Log experiment parameters
        log_exp_hyperparams(_logger, config)

    logger.info("Starting training")
    trainer.fit(
        model_wrapper, train_dataloader=train_loader, val_dataloaders=val_loader
    )

    if trainer.interrupted:
        print()
        inp = input("Evaluate on test set? (y/N) ")
        if inp.lower() != "y":
            return

    logger.info("Evaluating on test set")

    ckpt_path = "best" if not config.val_on_test else None
    trainer.test(model_wrapper, test_dataloaders=test_loaders, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
