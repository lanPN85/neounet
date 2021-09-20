import os
import torch
import random
import pytorch_lightning as pl

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from torchvision.utils import save_image
from loguru import logger

from polypnet.utils import probs_to_mask


def save_infer_image_3tile(pl_module, collate_fn, raw_image, raw_label, save_path):
    image, _ = collate_fn([(raw_image, raw_label)])
    batch_pred = pl_module(image.to(pl_module.device))[0].cpu()
    batch_pred = torch.sigmoid(batch_pred)
    batch_mask = probs_to_mask(batch_pred)
    mask = batch_mask[0]

    denormed_mask = mask * 255
    denormed_mask = denormed_mask.repeat(3, 1, 1)

    save_image(
        [raw_image.float(), raw_label.float(), denormed_mask.float()],
        save_path,
        nrow=3,
        normalize=True,
        range=(0, 255),
        padding=10,
        pad_value=0.75,
    )


def save_multiclass_image_3tile(
    pl_module, collate_fn, raw_image, raw_label, raw_cls, save_path
):
    image, _, _ = collate_fn([(raw_image, raw_label, raw_cls)])
    batch_pred = pl_module(image.to(pl_module.device))[0][0].cpu()
    batch_pred = torch.sigmoid(batch_pred)  # 2 x H x W

    seg_mask = (
        torch.max(batch_pred, dim=0, keepdim=True).values >= 0.5
    ).int()  # 1 x H x W
    class_mask = torch.max(batch_pred, dim=0, keepdim=True).indices
    green_mask = (class_mask == 0).int() * seg_mask
    red_mask = (class_mask == 1).int() * seg_mask

    mask = torch.zeros_like(raw_image)
    mask[1, ...] = 255 * green_mask
    mask[0, ...] = 255 * red_mask

    save_image(
        [raw_image.float(), raw_cls.float(), mask.float()],
        save_path,
        nrow=3,
        normalize=True,
        range=(0, 255),
        padding=10,
        pad_value=0.75,
    )


def save_multiclass_image_mask(
    pl_module, collate_fn, raw_image, raw_label, raw_cls, save_path
):
    image, _, _ = collate_fn([(raw_image, raw_label, raw_cls)])
    batch_pred = pl_module(image.to(pl_module.device))[0][0].cpu()
    batch_pred = torch.sigmoid(batch_pred)  # 2 x H x W

    seg_mask = (
        torch.max(batch_pred, dim=0, keepdim=True).values >= 0.5
    ).int()  # 1 x H x W
    class_mask = torch.max(batch_pred, dim=0, keepdim=True).indices
    green_mask = (class_mask == 0).int() * seg_mask
    red_mask = (class_mask == 1).int() * seg_mask

    mask = torch.zeros_like(raw_image)
    mask[1, ...] = 255 * green_mask
    mask[0, ...] = 255 * red_mask

    save_image([mask.float()], save_path, nrow=1, normalize=True, range=(0, 255))


class ResultSampleCallback(pl.callbacks.Callback):
    def __init__(
        self,
        save_dir: str,
        collate_fn,
        mlflow_logger=None,
        images_per_epoch=1,
        skip_first=0,
        mode="segment",
    ):
        super().__init__()

        self.save_dir = save_dir
        self.collate_fn = collate_fn
        self.images_per_epoch = images_per_epoch
        self.mlflow_logger = mlflow_logger
        self.skip_first = skip_first
        self.mode = mode
        self.__executor = ThreadPoolExecutor(1)

    """
    Stores result image samples at each validation epoch
    """

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_validation_end(trainer, pl_module)
        pl_module.eval()

        epoch = trainer.current_epoch
        version = self._get_next_version()
        save_dir = os.path.join(self.save_dir, f"version_{version}", str(epoch))
        os.makedirs(save_dir, exist_ok=True)

        if epoch < self.skip_first:
            logger.info(f"Skipping epoch {epoch}")
            return

        loader = trainer.val_dataloaders[0]
        dataset = loader.dataset

        for i in range(self.images_per_epoch):
            index = random.randint(0, len(dataset) - 1)

            if self.mode == "segment":
                raw_image, raw_label, image_path, _ = dataset[index]
                image_name = image_path.split("/")[-1].split(".")[0]
                image_name = image_name.replace(":", "__")
                output_name = f"{image_name}.pred.jpg"
                save_path = os.path.join(save_dir, output_name)
                save_infer_image_3tile(
                    pl_module, self.collate_fn, raw_image, raw_label, save_path
                )
            elif self.mode == "label":
                raw_image, raw_label, raw_cls, image_path, _, _ = dataset[index]
                image_name = image_path.split("/")[-1].split(".")[0]
                image_name = image_name.replace(":", "__")
                output_name = f"{image_name}.pred.jpg"
                save_path = os.path.join(save_dir, output_name)
                save_multiclass_image_3tile(
                    pl_module, self.collate_fn, raw_image, raw_label, raw_cls, save_path
                )

            # Push to MLflow
            if self.mlflow_logger is not None:
                run_id = self.mlflow_logger.run_id
                self.__executor.submit(
                    self.mlflow_logger.experiment.log_artifact,
                    run_id,
                    save_path,
                    artifact_path=f"samples/{epoch}",
                )

    def _get_next_version(self):
        if not os.path.exists(self.save_dir):
            return 0

        existing_versions = []
        for d in os.listdir(self.save_dir):
            if os.path.isdir(os.path.join(self.save_dir, d)) and d.startswith(
                "version_"
            ):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
