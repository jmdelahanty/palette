"""Custom Ultralytics trainer that swaps mask loss with BCE+Dice."""

from __future__ import annotations

import torch
from ultralytics.models.yolo.segment.train import SegTrainer
from ultralytics.models.yolo.segment.loss import SegmentLoss

from .losses import BCEDiceCriterion


class BCEDiceSegLoss(SegmentLoss):
    """SegmentLoss wrapper that replaces the mask term with BCE+Dice."""

    def __init__(self, model, bce_weight: float = 0.5) -> None:
        super().__init__(model)
        self.bce_dice = BCEDiceCriterion(bce_weight=bce_weight)

    def _compute_mask_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
        return self.bce_dice(pred_masks, gt_masks)

    def __call__(self, preds, batch):
        # Run parent loss to obtain matches and base terms.
        losses = super().__call__(preds, batch)

        pred_masks = getattr(self, "fm", None)
        target_masks = getattr(self, "tm", None)

        if pred_masks is not None and target_masks is not None:
            seg_loss = self._compute_mask_loss(pred_masks, target_masks)
            losses["seg"] = seg_loss
            losses["total"] = (
                losses["box"] * self.hyp.box
                + losses["cls"] * self.hyp.cls
                + losses["dfl"] * self.hyp.dfl
                + losses["seg"] * self.hyp.seg
            )
        return losses


class BCEDiceSegTrainer(SegTrainer):
    """SegTrainer that swaps mask criterion for BCE+Dice."""

    def get_criterion(self):
        bce_weight = float(getattr(self.args, "bce_weight", 0.5))
        return BCEDiceSegLoss(self.model, bce_weight=bce_weight)

