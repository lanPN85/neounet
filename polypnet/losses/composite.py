import torch.nn as nn
import pytorch_lightning as pl
import torch

from loguru import logger
from typing import Any, Optional, Dict, Iterable


class CompoundLoss(pl.LightningModule):
    def __init__(self, losses: Iterable, weights=None):
        super().__init__()

        if weights is None:
            N = len(losses)
            weights = [1./N] * N

        self.weights = weights
        self.losses = nn.ModuleList(losses)

    def forward(self, *inputs):
        """
        Forward function.
        Sums all losses over the given inputs
        """
        loss = torch.scalar_tensor(0, device=self._device)

        for loss_fn, w in zip(self.losses, self.weights):
            loss += w * loss_fn(*inputs)

        return loss


class ConditionalCompoundLoss(pl.LightningModule):
    def __init__(self, losses: Iterable,
        weights=None,
        rules: Optional[Dict[int, Dict[str, Any]]] = None
    ):
        super().__init__()

        if rules is None:
            rules = {}

        if weights is None:
            N = len(losses)
            weights = [1./N] * N

        self.losses = nn.ModuleList(losses)
        self.weights = weights
        self.rules = rules

    def forward(self, *inputs):
        """
        Forward function.
        Sums all losses over the given inputs
        """
        loss = torch.scalar_tensor(0, device=self._device)
        current_epoch = self.trainer.current_epoch

        # Get all used losses
        used_losses = []
        used_weights = []
        for i, loss_fn in enumerate(self.losses):
            rule = self.rules.get(str(i), {})
            is_used = False

            start_epoch = rule.get("start_epoch", 0)
            end_epoch = rule.get("end_epoch", float('inf'))
            is_used = (current_epoch >= start_epoch) and (current_epoch <= end_epoch)

            if is_used:
                used_losses.append(loss_fn)
                used_weights.append(self.weights[i])

        # Normalize weights
        weights = torch.softmax(torch.tensor(used_weights), dim=0).to(self.device)

        for loss_fn, w in zip(self.losses, weights):
            loss += w * loss_fn(*inputs)

        return loss
