import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms as ttf
from loguru import logger
from typing import Any, List, Optional, Callable, Tuple, Union, Dict, IO

from polypnet.utils import generate_scales, probs_to_onehot
from polypnet.losses import (
    MultiscaleLoss,
    CompoundLoss,
    DiceLoss,
    TverskyLoss,
    BinaryCrossEntropyLoss,
)
from polypnet import metrics


class UnetClassifyModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: nn.Module,
        lr_scheduler: nn.Module,
        seg_loss_fn: Optional[Callable] = None,
        cls_loss_fn: Optional[Callable] = None,
        val_on_test=False,
        num_classes=2,
        seg_weight=0.5,
        iou_thres=0.7,
        test_input_size=None,
        train_sizes: Optional[List[Tuple[int, int]]] = None,
        test_names=None,
    ):
        super().__init__()

        if test_names is None:
            test_names = []

        if test_input_size is not None:
            test_input_size = tuple(test_input_size)

        if seg_loss_fn is None:
            seg_loss_fn = MultiscaleLoss(
                CompoundLoss([TverskyLoss(), BinaryCrossEntropyLoss()])
            )

        if cls_loss_fn is None:
            cls_loss_fn = MultiscaleLoss(
                CompoundLoss([TverskyLoss(), BinaryCrossEntropyLoss()])
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.seg_loss_fn = seg_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.num_classes = num_classes
        self.test_input_size = test_input_size
        self.test_name = None
        self.test_names = test_names
        self.seg_weight = seg_weight
        self.train_sizes = train_sizes
        self.iou_thres = iou_thres
        self.val_on_test = val_on_test

    def set_test_name(self, test_name):
        self.test_name = test_name

    def configure_optimizers(self):
        return ([self.optimizer], [self.lr_scheduler])

    def forward(self, input):
        return self.model(input)

    @staticmethod
    def infer_seg_output(output):
        return torch.max(output, dim=1, keepdim=True).values

    def training_step(self, batch, batch_idx):
        input_ = batch[0]  # B x C x H x W
        seg_label_ = batch[1]  # B x 1 x H x W
        cls_label_ = batch[2]  # B x C x H x W

        train_sizes = self.train_sizes
        if self.train_sizes is None:
            train_sizes = [input_.shape[-2:]]

        total_loss = 0
        for train_size in train_sizes:
            resizer = ttf.Resize(tuple(train_size))
            input = resizer(input_)
            seg_label = resizer(seg_label_)
            cls_label = resizer(cls_label_)

            cls_trunc_label = cls_label[:, :-1, ...]
            cls_ignore_mask = cls_label[:, -1, ...]
            cls_ignore_mask = torch.stack(
                [cls_ignore_mask] * (cls_label.shape[1] - 1), dim=1
            )

            scaled_seg = generate_scales(seg_label, self.model.output_scales)
            scaled_cls = generate_scales(cls_label, self.model.output_scales)

            outputs = self(input)  # [B x C x H x W]
            seg_outputs = [self.infer_seg_output(o) for o in outputs]

            output = outputs[0]

            cls_loss = self.cls_loss_fn(outputs, scaled_cls)
            seg_loss = self.seg_loss_fn(seg_outputs, scaled_seg)

            # Calculate metrics
            cls_probs = torch.sigmoid(output)  # B x C x H x W
            cls_probs = (cls_probs > 0.5).long()

            seg_probs = torch.sigmoid(seg_outputs[0])
            seg_probs = (seg_probs > 0.5).long()

            loss = cls_loss * (1 - self.seg_weight) + seg_loss * self.seg_weight

            self._log_metrics(seg_probs, seg_label, seg_loss, prefix="train.seg")
            self._log_metrics(
                cls_probs,
                cls_trunc_label,
                cls_loss,
                ignore_mask=cls_ignore_mask,
                prefix="train.label",
            )
            self.log(f"train.loss", loss, on_step=False, on_epoch=True)
            total_loss += loss

            # Cleanup to avoid OOM
            del input
            del seg_label
            del cls_label

        return total_loss / len(train_sizes)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.val_on_test:
            return self.test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

        input = batch[0]  # B x C x H x W
        seg_label = batch[1]  # B x 1 x H x W
        cls_label = batch[2]  # B x C x H x W
        cls_trunc_label = cls_label[:, :-1, ...]
        cls_ignore_mask = cls_label[:, -1, ...]
        cls_ignore_mask = torch.stack(
            [cls_ignore_mask] * (cls_label.shape[1] - 1), dim=1
        )

        scaled_seg = generate_scales(seg_label, self.model.output_scales)
        scaled_cls = generate_scales(cls_label, self.model.output_scales)

        outputs = self(input)  # [B x C x H x W]
        seg_outputs = [self.infer_seg_output(o) for o in outputs]

        output = outputs[0]

        cls_loss = self.cls_loss_fn(outputs, scaled_cls)
        seg_loss = self.seg_loss_fn(seg_outputs, scaled_seg)

        # Calculate metrics
        cls_probs = torch.sigmoid(output)  # B x C x H x W
        cls_probs = (cls_probs > 0.5).long()
        self._log_metrics(
            cls_probs,
            cls_trunc_label,
            cls_loss,
            ignore_mask=cls_ignore_mask,
            prefix="val.label",
        )

        seg_probs = torch.sigmoid(seg_outputs[0])
        seg_probs = (seg_probs > 0.5).long()
        self._log_metrics(seg_probs, seg_label, seg_loss, prefix="val.seg")

        loss = cls_loss * (1 - self.seg_weight) + seg_loss * self.seg_weight

        self.log(f"val.loss", loss, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        if self.val_on_test:
            return self.test_epoch_end(outputs)
        else:
            return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch[0]  # B x C x H x W
        seg_label = batch[1]  # B x C x H x W
        cls_label = batch[2]  # B x C+1 x H x W
        cls_trunc_label = cls_label[:, :-1, ...]
        cls_ignore_mask = cls_label[:, -1, ...]
        cls_ignore_mask = torch.stack(
            [cls_ignore_mask] * (cls_label.shape[1] - 1), dim=1
        )
        orig_label_size = seg_label.shape[2:]

        if self.test_input_size is not None:
            # Resize input
            input = ttf.Resize(self.test_input_size)(input)

        batch_size = input.shape[0]

        outputs = self(input)  # [B x C x H x W]
        output = outputs[0]

        if self.test_input_size is not None:
            # Resize output
            output = ttf.Resize(orig_label_size)(output)

        cls_probs = torch.sigmoid(output)  # B x C x H x W
        cls_probs = (cls_probs > 0.5).long()
        seg_probs = torch.sigmoid(self.infer_seg_output(output))
        seg_probs = (seg_probs > 0.5).long()

        # Metrics for benign label
        ben_true_pos = metrics.true_positive(
            cls_probs[:, 0, ...].unsqueeze(1),
            cls_trunc_label[:, 0, ...].unsqueeze(1),
            cls_ignore_mask[:, 0, ...].unsqueeze(1),
        )
        ben_label_pos = metrics.label_positive(
            cls_probs[:, 0, ...].unsqueeze(1),
            cls_trunc_label[:, 0, ...].unsqueeze(1),
            cls_ignore_mask[:, 0, ...].unsqueeze(1),
        )
        ben_pred_pos = metrics.prod_positive(
            cls_probs[:, 0, ...].unsqueeze(1),
            cls_trunc_label[:, 0, ...].unsqueeze(1),
            cls_ignore_mask[:, 0, ...].unsqueeze(1),
        )
        total_cls_ben_iou = (
            metrics.iou(
                cls_probs[:, 0, ...].unsqueeze(1),
                cls_trunc_label[:, 0, ...].unsqueeze(1),
                cls_ignore_mask[:, 0, ...].unsqueeze(1),
            )
            * batch_size
        )
        total_cls_ben_dice = (
            metrics.dice(
                cls_probs[:, 0, ...].unsqueeze(1),
                cls_trunc_label[:, 0, ...].unsqueeze(1),
                cls_ignore_mask[:, 0, ...].unsqueeze(1),
            )
            * batch_size
        )
        total_cls_ben_precision = (
            metrics.precision(
                cls_probs[:, 0, ...].unsqueeze(1),
                cls_trunc_label[:, 0, ...].unsqueeze(1),
                cls_ignore_mask[:, 0, ...].unsqueeze(1),
            )
            * batch_size
        )
        total_cls_ben_recall = (
            metrics.recall(
                cls_probs[:, 0, ...].unsqueeze(1),
                cls_trunc_label[:, 0, ...].unsqueeze(1),
                cls_ignore_mask[:, 0, ...].unsqueeze(1),
            )
            * batch_size
        )

        # logger.info(f"Benign P/R/D: {total_cls_ben_precision.item()} {total_cls_ben_recall.item()} {total_cls_ben_dice.item()}")

        # Metrics for malignant label
        mal_true_pos = metrics.true_positive(
            cls_probs[:, 1, ...].unsqueeze(1),
            cls_trunc_label[:, 1, ...].unsqueeze(1),
            cls_ignore_mask[:, 1, ...].unsqueeze(1),
        )
        mal_label_pos = metrics.label_positive(
            cls_probs[:, 1, ...].unsqueeze(1),
            cls_trunc_label[:, 1, ...].unsqueeze(1),
            cls_ignore_mask[:, 1, ...].unsqueeze(1),
        )
        mal_pred_pos = metrics.prod_positive(
            cls_probs[:, 1, ...].unsqueeze(1),
            cls_trunc_label[:, 1, ...].unsqueeze(1),
            cls_ignore_mask[:, 1, ...].unsqueeze(1),
        )
        total_cls_mal_iou = (
            metrics.iou(
                cls_probs[:, 1, ...].unsqueeze(1),
                cls_trunc_label[:, 1, ...].unsqueeze(1),
                cls_ignore_mask[:, 1, ...].unsqueeze(1),
            )
            * batch_size
        )
        total_cls_mal_dice = (
            metrics.dice(
                cls_probs[:, 1, ...].unsqueeze(1),
                cls_trunc_label[:, 1, ...].unsqueeze(1),
                cls_ignore_mask[:, 1, ...].unsqueeze(1),
            )
            * batch_size
        )
        total_cls_mal_precision = (
            metrics.precision(
                cls_probs[:, 1, ...].unsqueeze(1),
                cls_trunc_label[:, 1, ...].unsqueeze(1),
                cls_ignore_mask[:, 1, ...].unsqueeze(1),
            )
            * batch_size
        )
        total_cls_mal_recall = (
            metrics.recall(
                cls_probs[:, 1, ...].unsqueeze(1),
                cls_trunc_label[:, 1, ...].unsqueeze(1),
                cls_ignore_mask[:, 1, ...].unsqueeze(1),
            )
            * batch_size
        )

        # logger.info(f"Malignant P/R/D: {total_cls_mal_precision.item()} {total_cls_mal_recall.item()} {total_cls_mal_dice.item()}")

        seg_true_pos = metrics.true_positive(seg_probs, seg_label)
        seg_label_pos = metrics.label_positive(seg_probs, seg_label)
        seg_pred_pos = metrics.prod_positive(seg_probs, seg_label)

        total_seg_iou = metrics.iou(seg_probs, seg_label) * batch_size
        total_seg_dice = metrics.dice(seg_probs, seg_label) * batch_size
        total_seg_precision = metrics.precision(seg_probs, seg_label) * batch_size
        total_seg_recall = metrics.recall(seg_probs, seg_label) * batch_size

        ben_label_obj_count = metrics.label_object_count(
            cls_probs[0], cls_trunc_label[0], cls_ignore_mask[0], dim=0
        )
        ben_pred_obj_count = metrics.pred_object_count(
            cls_probs[0], cls_trunc_label[0], cls_ignore_mask[0], dim=0
        )
        ben_true_obj_count = metrics.true_object_count(
            cls_probs[0],
            cls_trunc_label[0],
            cls_ignore_mask[0],
            dim=0,
            iou=self.iou_thres,
        )

        mal_label_obj_count = metrics.label_object_count(
            cls_probs[0], cls_trunc_label[0], cls_ignore_mask[0], dim=1
        )
        mal_pred_obj_count = metrics.pred_object_count(
            cls_probs[0], cls_trunc_label[0], cls_ignore_mask[0], dim=1
        )
        mal_true_obj_count = metrics.true_object_count(
            cls_probs[0],
            cls_trunc_label[0],
            cls_ignore_mask[0],
            dim=1,
            iou=self.iou_thres,
        )

        return {
            f"{dataloader_idx}/seg_true_pos": seg_true_pos,
            f"{dataloader_idx}/seg_label_pos": seg_label_pos,
            f"{dataloader_idx}/seg_pred_pos": seg_pred_pos,
            f"{dataloader_idx}/ben_true_pos": ben_true_pos,
            f"{dataloader_idx}/ben_label_pos": ben_label_pos,
            f"{dataloader_idx}/ben_pred_pos": ben_pred_pos,
            f"{dataloader_idx}/mal_true_pos": mal_true_pos,
            f"{dataloader_idx}/mal_label_pos": mal_label_pos,
            f"{dataloader_idx}/mal_pred_pos": mal_pred_pos,
            f"{dataloader_idx}/total_cls_ben_iou": total_cls_ben_iou,
            f"{dataloader_idx}/total_cls_ben_dice": total_cls_ben_dice,
            f"{dataloader_idx}/total_cls_ben_precision": total_cls_ben_precision,
            f"{dataloader_idx}/total_cls_ben_recall": total_cls_ben_recall,
            f"{dataloader_idx}/total_cls_mal_iou": total_cls_mal_iou,
            f"{dataloader_idx}/total_cls_mal_dice": total_cls_mal_dice,
            f"{dataloader_idx}/total_cls_mal_precision": total_cls_mal_precision,
            f"{dataloader_idx}/total_cls_mal_recall": total_cls_mal_recall,
            f"{dataloader_idx}/total_seg_iou": total_seg_iou,
            f"{dataloader_idx}/total_seg_dice": total_seg_dice,
            f"{dataloader_idx}/total_seg_precision": total_seg_precision,
            f"{dataloader_idx}/total_seg_recall": total_seg_recall,
            f"{dataloader_idx}/ben_label_obj_count": ben_label_obj_count,
            f"{dataloader_idx}/ben_pred_obj_count": ben_pred_obj_count,
            f"{dataloader_idx}/ben_true_obj_count": ben_true_obj_count,
            f"{dataloader_idx}/mal_label_obj_count": mal_label_obj_count,
            f"{dataloader_idx}/mal_pred_obj_count": mal_pred_obj_count,
            f"{dataloader_idx}/mal_true_obj_count": mal_true_obj_count,
            f"{dataloader_idx}/batch_size": batch_size,
        }

    def test_epoch_end(self, outputs):
        loader_indices = [None]
        if self.test_names is not None:
            loader_indices = list(range(len(self.test_names)))

        if len(self.test_names) < 2:
            outputs = [outputs]

        for dataloader_idx, output in zip(loader_indices, outputs):
            test_name = self.test_names[dataloader_idx]
            sum_batch_size = sum(
                [o.get(f"{dataloader_idx}/batch_size", 0) for o in output]
            )

            m_names = [
                f"{dataloader_idx}/total_seg_iou",
                f"{dataloader_idx}/total_seg_dice",
                f"{dataloader_idx}/total_seg_precision",
                f"{dataloader_idx}/total_seg_recall",
            ]

            display_names = [
                f"test.{test_name}.iou.seg.macro",
                f"test.{test_name}.dice.seg.macro",
                f"test.{test_name}.precision.seg.macro",
                f"test.{test_name}.recall.seg.macro",
            ]

            for m_name, display_name in zip(m_names, display_names):
                sum_m = sum([o.get(m_name, 0) for o in output])
                val = sum_m / sum_batch_size
                self.log(
                    display_name, val, prog_bar=False, on_step=False, on_epoch=True
                )

            total_ben_label_count = sum(
                [o.get(f"{dataloader_idx}/ben_label_pos", 0) for o in output]
            )
            total_ben_pred_count = sum(
                [o.get(f"{dataloader_idx}/ben_pred_pos", 0) for o in output]
            )
            total_ben_true_count = sum(
                [o.get(f"{dataloader_idx}/ben_true_pos", 0) for o in output]
            )
            total_mal_label_count = sum(
                [o.get(f"{dataloader_idx}/mal_label_pos", 0) for o in output]
            )
            total_mal_pred_count = sum(
                [o.get(f"{dataloader_idx}/mal_pred_pos", 0) for o in output]
            )
            total_mal_true_count = sum(
                [o.get(f"{dataloader_idx}/mal_true_pos", 0) for o in output]
            )
            total_seg_label_count = sum(
                [o.get(f"{dataloader_idx}/seg_label_pos", 0) for o in output]
            )
            total_seg_pred_count = sum(
                [o.get(f"{dataloader_idx}/seg_pred_pos", 0) for o in output]
            )
            total_seg_true_count = sum(
                [o.get(f"{dataloader_idx}/seg_true_pos", 0) for o in output]
            )

            ben_precision = total_ben_true_count / max(total_ben_pred_count, 1e-5)
            ben_recall = total_ben_true_count / max(total_ben_label_count, 1e-5)
            ben_dice = (
                2 * ben_precision * ben_recall / max(ben_precision + ben_recall, 1e-5)
            )
            ben_iou = total_ben_true_count / max(
                1e-5,
                total_ben_label_count + total_ben_pred_count - total_ben_true_count,
            )
            self.log(
                f"test.{test_name}.precision.label_ben.micro",
                ben_precision,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.recall.label_ben.micro",
                ben_recall,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.dice.label_ben.micro",
                ben_dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.iou.label_ben.micro",
                ben_iou,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

            mal_precision = total_mal_true_count / max(total_mal_pred_count, 1e-5)
            mal_recall = total_mal_true_count / max(total_mal_label_count, 1e-5)
            mal_dice = (
                2 * mal_precision * mal_recall / max(mal_precision + mal_recall, 1e-5)
            )
            mal_iou = total_mal_true_count / max(
                1e-5,
                total_mal_label_count + total_mal_pred_count - total_mal_true_count,
            )
            self.log(
                f"test.{test_name}.precision.label_mal.micro",
                mal_precision,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.recall.label_mal.micro",
                mal_recall,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.dice.label_mal.micro",
                mal_dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.iou.label_mal.micro",
                mal_iou,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

            seg_precision = total_seg_true_count / max(total_seg_pred_count, 1e-5)
            seg_recall = total_seg_true_count / max(total_seg_label_count, 1e-5)
            seg_dice = (
                2 * seg_precision * seg_recall / max(seg_precision + seg_recall, 1e-5)
            )
            seg_iou = total_seg_true_count / max(
                1e-5,
                total_seg_label_count + total_seg_pred_count - total_seg_true_count,
            )
            self.log(
                f"test.{test_name}.precision.seg.micro",
                seg_precision,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.recall.seg.micro",
                seg_recall,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.dice.seg.micro",
                seg_dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.iou.seg.micro",
                seg_iou,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

            total_ben_label_obj_count = sum(
                [o.get(f"{dataloader_idx}/ben_label_obj_count", 0) for o in output]
            )
            total_ben_pred_obj_count = sum(
                [o.get(f"{dataloader_idx}/ben_pred_obj_count", 0) for o in output]
            )
            total_ben_true_obj_count = sum(
                [o.get(f"{dataloader_idx}/ben_true_obj_count", 0) for o in output]
            )
            total_mal_label_obj_count = sum(
                [o.get(f"{dataloader_idx}/mal_label_obj_count", 0) for o in output]
            )
            total_mal_pred_obj_count = sum(
                [o.get(f"{dataloader_idx}/mal_pred_obj_count", 0) for o in output]
            )
            total_mal_true_obj_count = sum(
                [o.get(f"{dataloader_idx}/mal_true_obj_count", 0) for o in output]
            )

            ben_obj_precision = total_ben_true_obj_count / max(
                total_ben_pred_obj_count, 1e-5
            )
            ben_obj_recall = total_ben_true_obj_count / max(
                total_ben_label_obj_count, 1e-5
            )
            ben_obj_dice = (
                2
                * ben_obj_precision
                * ben_obj_recall
                / max(ben_obj_precision + ben_obj_recall, 1e-5)
            )
            self.log(
                f"test.{test_name}.oprecision.label_ben.micro/{self.iou_thres}",
                ben_obj_precision,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.orecall.label_ben.micro/{self.iou_thres}",
                ben_obj_recall,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.odice.label_ben.micro/{self.iou_thres}",
                ben_obj_dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

            mal_obj_precision = total_mal_true_obj_count / max(
                total_mal_pred_obj_count, 1e-5
            )
            mal_obj_recall = total_mal_true_obj_count / max(
                total_mal_label_obj_count, 1e-5
            )
            mal_obj_dice = (
                2
                * mal_obj_precision
                * mal_obj_recall
                / max(mal_obj_precision + mal_obj_recall, 1e-5)
            )
            self.log(
                f"test.{test_name}.oprecision.label_mal.micro/{self.iou_thres}",
                mal_obj_precision,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.orecall.label_mal.micro/{self.iou_thres}",
                mal_obj_recall,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"test.{test_name}.odice.label_mal.micro/{self.iou_thres}",
                mal_obj_dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

    def _log_metrics(self, pred_probs, label, loss, prefix: str, ignore_mask=None):
        iou = metrics.iou(pred_probs, label, ignore_mask=ignore_mask)
        self.log(f"{prefix}.iou", iou, prog_bar=False, on_step=False, on_epoch=True)

        dice = metrics.dice(pred_probs, label, ignore_mask=ignore_mask)
        self.log(f"{prefix}.dice", dice, prog_bar=False, on_step=False, on_epoch=True)

        precision = metrics.precision(pred_probs, label, ignore_mask=ignore_mask)
        self.log(
            f"{prefix}.precision",
            precision,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        recall = metrics.recall(pred_probs, label, ignore_mask=ignore_mask)
        self.log(
            f"{prefix}.recall", recall, prog_bar=False, on_step=False, on_epoch=True
        )

        self.log(f"{prefix}.loss", loss, on_step=False, on_epoch=True)
