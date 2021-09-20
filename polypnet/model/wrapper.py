import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda
import numpy as np

from torchvision import transforms as ttf
from loguru import logger
from typing import Any, List, Optional, Callable, Tuple, Union, Dict, IO

from polypnet.utils import generate_scales, probs_to_onehot
from polypnet.losses import MultiscaleLoss, CompoundLoss,\
    DiceLoss, TverskyLoss, BinaryCrossEntropyLoss
from polypnet import metrics


class UnetModelWrapper(pl.LightningModule):
    def __init__(self,
        model: nn.Module,
        optimizer: nn.Module,
        lr_scheduler: nn.Module,
        loss_fn: Optional[Callable] = None,
        num_classes=1,
        test_input_size=None,
        train_sizes: Optional[List[Tuple[int, int]]] = None,
        val_on_test=False,
        test_names=None,
        test_patch=False,
        test_patch_size=(96, 96),
        test_patch_resize=(256, 256),
        test_patch_stride=8
    ):
        super().__init__()

        if test_names is None:
            test_names = []

        if test_input_size is not None:
            test_input_size = tuple(test_input_size)

        if loss_fn is None:
            loss_fn = MultiscaleLoss(
                CompoundLoss([
                    TverskyLoss(),
                    BinaryCrossEntropyLoss()
                ])
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.test_input_size = test_input_size
        self.train_sizes = train_sizes
        self.val_on_test = val_on_test
        self.test_names = test_names
        self.test_patch = test_patch
        self.test_patch_size = test_patch_size
        self.test_patch_resize = test_patch_resize
        self.test_patch_stride = test_patch_stride

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [self.lr_scheduler]
        )

    def forward(self, input):
        return self.model(input)

    def forward_with_patch(self, input, patch_size, infer_size, stride):
        size_x, size_y = patch_size
        sum_mask = torch.zeros(
            input.shape[0], self.num_classes,
            input.shape[-2], input.shape[-1],
            device=self.device
        )
        count_mask = torch.zeros_like(sum_mask, device=self.device)

        for x in range(0, input.shape[-2] - size_x, stride):
            for y in range(0, input.shape[-1] - size_y, stride):
                patch = input[:, :, x:x+size_x, y:y+size_y]
                if infer_size is not None:
                    # Resize input
                    patch = ttf.Resize(tuple(infer_size))(patch)

                patch_out = self.model(patch)[0]
                patch_out = ttf.Resize((size_x, size_y))(patch_out)

                count_mask[:, :, x:x+size_x, y:y+size_y] += 1
                sum_mask[:, :, x:x+size_x, y:y+size_y] += patch_out

        avg_mask = sum_mask / count_mask
        return avg_mask

    def training_step(self, batch, batch_idx):
        input_ = batch[0]  # B x C x H x W
        label_ = batch[1]  # B x C x H x W

        train_sizes = self.train_sizes
        if self.train_sizes is None:
            train_sizes = [input_.shape[-2:]]

        total_loss = 0

        for train_size in train_sizes:
            resizer = ttf.Resize(tuple(train_size))
            input = resizer(input_)
            label = resizer(label_)

            if self.num_classes > 1:
                label = F.one_hot(label.squeeze(dim=1), self.num_classes).permute(0, 3, 1, 2).float()

            scaled_labels = generate_scales(label, self.model.output_scales)

            outputs = self.model(input)  # [B x C x H x W]

            output = outputs[0]
            pred_probs = torch.sigmoid(output)  # B x C x H x W
            pred_probs = probs_to_onehot(pred_probs)
            loss = self.loss_fn(outputs, scaled_labels)
            total_loss += loss

            # Calculate metrics
            self._log_metrics(pred_probs, label, loss, prefix="train")

            # Cleanup to avoid OOM
            del input
            del label

        return total_loss / len(train_sizes)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if self.val_on_test:
            return self.test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

        input = batch[0]  # B x C x H x W
        label = batch[1]  # B x C x H x W
        if self.num_classes > 1:
            label = F.one_hot(label.squeeze(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        scaled_labels = generate_scales(label, self.model.output_scales)

        outputs = self(input)  # [B x C x H x W]

        output = outputs[0]
        pred_probs = torch.sigmoid(output)  # B x C x H x W
        pred_probs = probs_to_onehot(pred_probs)
        loss = self.loss_fn(outputs, scaled_labels)

        # Calculate metrics
        self._log_metrics(pred_probs, label, loss, prefix="val")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        input = batch[0]  # B x C x H x W
        label = batch[1]  # B x C x H x W
        orig_label_size = label.shape[2:]

        if self.test_input_size is not None:
            # Resize input
            input = ttf.Resize(self.test_input_size)(input)

        batch_size = input.shape[0]
        if self.num_classes > 1:
            label = F.one_hot(label.squeeze(dim=1), self.num_classes).permute(0, 3, 1, 2).float()

        if not self.test_patch:
            output = self.model(input)[0]  # [B x C x H x W]
        else:
            output = self.forward_with_patch(
                input=input, patch_size=self.test_patch_size,
                infer_size=self.test_patch_resize,
                stride=self.test_patch_stride
            )

        if self.test_input_size is not None:
            # Resize output
            output = ttf.Resize(orig_label_size)(output)

        pred_probs = torch.sigmoid(output)  # B x C x H x W
        pred_probs = probs_to_onehot(pred_probs)

        total_iou = metrics.iou(pred_probs, label) * batch_size
        total_dice = metrics.dice(pred_probs, label) * batch_size
        total_precision = metrics.precision(pred_probs, label) * batch_size
        total_recall = metrics.recall(pred_probs, label) * batch_size

        total_intersection = torch.sum(pred_probs * label, dim=[0, 2, 3])
        total_union = torch.sum(pred_probs, dim=[0, 2, 3]) + torch.sum(label, dim=[0, 2, 3])
        total_true_pos = torch.sum(pred_probs * label, dim=[0, 2, 3])
        total_all_pos = torch.sum(pred_probs == 1, dim=[0, 2, 3])
        total_all_true = torch.sum(label == 1, dim=[0, 2, 3])

        return {
            f"{dataloader_idx}/total_intersection": total_intersection.item(),
            f"{dataloader_idx}/total_union": total_union.item(),
            f"{dataloader_idx}/total_true_pos": total_true_pos.item(),
            f"{dataloader_idx}/total_all_pos": total_all_pos.item(),
            f"{dataloader_idx}/total_all_true": total_all_true.item(),
            f"{dataloader_idx}/total_iou": total_iou,
            f"{dataloader_idx}/total_dice": total_dice,
            f"{dataloader_idx}/total_precision": total_precision,
            f"{dataloader_idx}/total_recall": total_recall,
            f"{dataloader_idx}/batch_size": batch_size
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        if self.val_on_test:
            return self.test_epoch_end(outputs)
        else:
            return super().validation_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        loader_indices = [None]
        if self.test_names is not None and len(self.test_names) > 1:
            loader_indices = list(range(len(self.test_names)))

        if len(self.test_names) < 2:
            outputs = [outputs]

        for loader_idx, output in zip(loader_indices, outputs):
            test_name = self.test_names[loader_idx or 0]

            sum_batch_size = sum([
                o.get(f"{loader_idx}/batch_size", 0)
                for o in output
            ])
            sum_iou = sum([
                o.get(f"{loader_idx}/total_iou", 0)
                for o in output
            ])
            sum_dice = sum([
                o.get(f"{loader_idx}/total_dice", 0)
                for o in output
            ])
            sum_precision = sum([
                o.get(f"{loader_idx}/total_precision", 0)
                for o in output
            ])
            sum_recall = sum([
                o.get(f"{loader_idx}/total_recall", 0)
                for o in output
            ])
            sum_intersection = sum([
                o.get(f"{loader_idx}/total_intersection", 0)
                for o in output
            ])
            sum_union = sum([
                o.get(f"{loader_idx}/total_union", 0)
                for o in output
            ])
            sum_true_pos = sum([
                o.get(f"{loader_idx}/total_true_pos", 0)
                for o in output
            ])
            sum_all_pos = sum([
                o.get(f"{loader_idx}/total_all_pos", 0)
                for o in output
            ])
            sum_all_true = sum([
                o.get(f"{loader_idx}/total_all_true", 0)
                for o in output
            ])

            iou, dice, precision, recall = 0, 0, 0, 0
            if sum_batch_size != 0:
                iou = sum_iou / sum_batch_size
                dice = sum_dice / sum_batch_size
                precision = sum_precision / sum_batch_size
                recall = sum_recall / sum_batch_size

            self.log(f"test.{test_name}.iou.macro", iou, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test.{test_name}.dice.macro", dice, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test.{test_name}.precision.macro", precision, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test.{test_name}.recall.macro", recall, prog_bar=False, on_step=False, on_epoch=True)

            micro_iou = np.mean((sum_intersection + 1) / (sum_union - sum_intersection + 1))
            micro_dice = np.mean((2 * sum_intersection + 1) / (sum_union + 1))
            micro_precision = np.mean((sum_true_pos + 1) / (sum_all_pos + 1))
            micro_recall = np.mean((sum_true_pos + 1) / (sum_all_true + 1))

            self.log(f"test.{test_name}.iou.micro", micro_iou, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test.{test_name}.dice.micro", micro_dice, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test.{test_name}.precision.micro", micro_precision, prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"test.{test_name}.recall.micro", micro_recall, prog_bar=False, on_step=False, on_epoch=True)

    def _log_metrics(self, pred_probs, label, loss, prefix: str):
        iou = metrics.iou(pred_probs, label)
        self.log(f"{prefix}.iou", iou, prog_bar=False, on_step=False, on_epoch=True)

        dice = metrics.dice(pred_probs, label)
        self.log(f"{prefix}.dice", dice, prog_bar=False, on_step=False, on_epoch=True)

        precision = metrics.precision(pred_probs, label)
        self.log(f"{prefix}.precision", precision, prog_bar=False, on_step=False, on_epoch=True)

        recall = metrics.recall(pred_probs, label)
        self.log(f"{prefix}.recall", recall, prog_bar=False, on_step=False, on_epoch=True)

        self.log(f"{prefix}.loss", loss, on_step=False, on_epoch=True)
