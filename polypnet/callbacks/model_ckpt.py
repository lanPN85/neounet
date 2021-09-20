import os
import random
import pytorch_lightning as pl

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from loguru import logger


class MLFlowModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, mlflow_logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlflow_logger = mlflow_logger
        self.__executor = ThreadPoolExecutor(1)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        # run_id = self.mlflow_logger.run_id
        # if self.best_model_path != "":
        #     with as_filename(self.best_model_path, "best.ckpt") as model_path:
        #         self.__executor.submit(
        #             self.mlflow_logger.experiment.log_artifact,
        #             run_id, model_path
        #         )

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        logger.info("Uploading models")
        run_id = self.mlflow_logger.run_id
        f1, f2 = None, None
        if self.best_model_path != "":
            self.mlflow_logger.experiment.set_tag(
                run_id, "best_model_path", self.best_model_path
            )
            f1 = self.__executor.submit(
                self.mlflow_logger.experiment.log_artifact,
                run_id,
                self.best_model_path,
                artifact_path=f"best",
            )

        if self.last_model_path != "":
            f2 = self.__executor.submit(
                self.mlflow_logger.experiment.log_artifact, run_id, self.last_model_path
            )

        if f1 is not None:
            try:
                f1.result(timeout=60 * 5)
            except TimeoutError:
                logger.error("Upload timed out")
        if f2 is not None:
            try:
                f2.result(timeout=60 * 5)
            except TimeoutError:
                logger.error("Upload timed out")
