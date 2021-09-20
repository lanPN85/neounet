import os
import torch
import csv
import random
import pytorch_lightning as pl

from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from loguru import logger


class CsvLogCallback(pl.Callback):
    def __init__(self,
        save_path: str,
        mlflow_logger=None
    ) -> None:
        super().__init__()
        self.save_path = save_path
        self.mlflow_logger = mlflow_logger
        self.__rows = []

    def on_test_batch_end(self,
        trainer, pl_module: pl.LightningModule,
        outputs: Any, batch: Any, batch_idx: int,
        dataloader_idx: int
    ) -> None:
        id = "N/A"
        if len(batch) > 3:
            id = batch[3][0]

        self.__rows.append([
            id,
            outputs.get("ben_true_pos"),
            outputs.get("ben_label_pos"),
            outputs.get("ben_pred_pos"),
            outputs.get("mal_true_pos"),
            outputs.get("mal_label_pos"),
            outputs.get("mal_pred_pos"),
        ])

    def on_test_end(self, trainer, pl_module: pl.LightningModule) -> None:
        header = [
            "Image ID",
            "True positive (Benign)",
            "Label positive (Benign)",
            "Predicted positive (Benign)",
            "True positive (Malignant)",
            "Label positive (Malignant)",
            "Predicted positive (Malignant)",
        ]

        with open(self.save_path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.__rows)

        if self.mlflow_logger is not None:
            run_id = self.mlflow_logger.run_id
            self.mlflow_logger.experiment.log_artifact(
                run_id, self.save_path
            )
