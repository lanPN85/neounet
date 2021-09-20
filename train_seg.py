#!/usr/bin/env python

import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from loguru import logger

from polypnet.model import UnetModelWrapper
from polypnet.config import load_config, print_config,\
    load_class_from_conf, read_mlflow_auth
from polypnet.callbacks import MLFlowModelCheckpoint, ResultSampleCallback
from polypnet.data import mask_collate_fn, Augmenter, NoOpAugmenter
from polypnet.losses import MultiscaleLoss, ConditionalCompoundLoss
from polypnet.optim.agc import AGC
from polypnet.data.base import PolypDataset
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
    config = load_config(
        config_paths
    )

    print_config(config)

    run_name = args.name or config.name

    logger.info("Loading losses")
    losses = []
    for c in config.loss.losses:
        l = load_class_from_conf(c)
        losses.append(l)
    loss = MultiscaleLoss(ConditionalCompoundLoss(
        losses, **config.loss.kwargs
    ))
    logger.info(f"Using {len(losses)} loss functions")

    logger.info("Loading model")
    model = load_class_from_conf(config.model)

    logger.info("Loading optimizer")
    optimizer = load_class_from_conf(config.optimizer, model.parameters())

    if config.agc.enabled:
        logger.info("Using AGC")
        optimizer = AGC(model.parameters(), optimizer, **config.agc.kwargs)

    logger.info("Loading scheduler")
    scheduler = load_class_from_conf(config.scheduler, optimizer=optimizer)

    model_wrapper = UnetModelWrapper(
        model, optimizer=optimizer,
        lr_scheduler=scheduler,
        loss_fn=loss,
        val_on_test=config.val_on_test,
        test_names=[
            x.name
            for x in config.test_datasets
        ],
        **config.wrapper.kwargs
    )

    # Load augmenter
    train_augmenter = Augmenter(**config.augment.kwargs)

    logger.info("Loading train dataset")

    dataset = load_class_from_conf(
        config.dataset,
        augmenter=train_augmenter,
        return_paths=True
    )
    raw_dataset = load_class_from_conf(
        config.dataset,
        augmenter=NoOpAugmenter(),
        return_paths=True
    )
    if not config.val_on_test:
        if config.val_dataset.split_from_train:
            logger.info("Splitting validation set")
            train_len = int(len(dataset) * (1 - config.val_dataset.split_ratio))
            val_len = len(dataset) - train_len
            train_dataset, val_dataset = random_split(
                dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(config.val_dataset.split_seed)
            )
            val_dataset.dataset = raw_dataset
        else:
            train_dataset = dataset
            val_dataset = load_class_from_conf(
                config.val_dataset,
                augmenter=NoOpAugmenter(),
                return_paths=True
            )
    else:
        train_dataset = dataset

    train_loader = DataLoader(
        train_dataset, shuffle=True,
        pin_memory=True,
        collate_fn=mask_collate_fn,
        **config.loader
    )

    # Load test dataloaders
    test_loaders = []
    for test_dataset_conf in config.test_datasets:
        logger.info(f"Loading test dataset '{test_dataset_conf.name}'")
        test_dataset: PolypDataset = load_class_from_conf(
            test_dataset_conf,
            augmenter=NoOpAugmenter(),
            return_paths=True
        )
        test_loaders.append(DataLoader(
            test_dataset, shuffle=False,
            pin_memory=True,
            collate_fn=mask_collate_fn,
            batch_size=1,
            **config.test_loader
        ))

    if config.val_on_test:
        val_loader = test_loaders
    else:
        val_loader = DataLoader(
            val_dataset, shuffle=False,
            pin_memory=True,
            collate_fn=mask_collate_fn,
            **config.loader
        )

    # Create result directories
    result_dir = os.path.join("results", run_name)
    ckpoint_dir = os.path.join(result_dir, "checkpoints")
    sample_dir = os.path.join(result_dir, "samples")
    os.makedirs(result_dir, exist_ok=True)

    # Setup callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(
            logging_interval="epoch"
        )
    ]
    if not config.val_on_test:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val.loss",
            mode="min",
            **config.callbacks.early_stop
        ))

    # Default logger when no MLflow available
    _logger = pl.loggers.CSVLogger(result_dir, name="csv")

    # Shared arguments for ModelCheckpoint
    if not config.val_on_test:
        common_ckpoint_args = dict(
            dirpath=ckpoint_dir,
            save_last=True,
            save_top_k=3,
            monitor="val.loss",
            mode="min",
            verbose=True
        )
    else:
        common_ckpoint_args = dict(
            dirpath=ckpoint_dir,
            save_last=True,
            verbose=True,
            period=1
        )

    mlflow_logger = None
    checkpoint_callback = None
    if config.with_mlflow:
        # Get mlflow username and password
        mlflow_dir = os.path.join(result_dir, "mlflow")
        mlflow_url = read_mlflow_auth()

        if mlflow_url is not None:
            logger.info("Using remote MLflow")
        else:
            logger.warning("No auth/mlflow.yml file found. Add this file containing your username and password to use remote MLflow. See auth/mlflow.example.yml for details")
            return

        # Setup MLflow logger
        tags = config.tags
        if tags is None:
            tags = {}
        tags["mlflow.runName"] = run_name
        _logger = pl.loggers.MLFlowLogger(
            experiment_name="PolypUnet",
            save_dir=mlflow_dir,
            tracking_uri=mlflow_url,
            tags=tags
        )
        mlflow_logger = _logger

        checkpoint_callback = MLFlowModelCheckpoint(
            _logger,
            **common_ckpoint_args
        )
    else:
        logger.info("Not using MLflow")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **common_ckpoint_args
    )
    callbacks.append(checkpoint_callback)

    # Add callback for saving result sample
    callbacks.append(ResultSampleCallback(
        sample_dir, mlflow_logger=mlflow_logger,
        collate_fn=mask_collate_fn,
        **config.callbacks.result_sample
    ))

    # Initialize trainer
    trainer = pl.Trainer(
        logger=_logger,
        callbacks=callbacks,
        default_root_dir=result_dir,
        terminate_on_nan=True,
        **config.trainer
    )
    loss.loss_fn.trainer = trainer

    if config.with_mlflow:
        # Log experiment parameters
        log_exp_hyperparams(_logger, config)

    logger.info("Starting training")
    trainer.fit(model_wrapper,
        train_dataloader=train_loader,
        val_dataloaders=val_loader
    )

    if trainer.interrupted:
        print()
        inp = input("Evaluate on test set? (y/N)")
        if inp.lower() != "y":
            return

    logger.info("Evaluating on test set")

    ckpt_path = "best" if not config.val_on_test else None
    trainer.test(
        model_wrapper,
        test_dataloaders=test_loaders,
        ckpt_path=ckpt_path
    )


if __name__ == "__main__":
    main()
