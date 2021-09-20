import torch.nn as nn
import torch
import pytorch_lightning as pl


class MultiscaleLoss(pl.LightningModule):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, predicts, labels):
        loss = torch.scalar_tensor(0, device=self._device)
        for pred, label in zip(predicts, labels):
            loss += self.loss_fn(pred, label)
        return loss
