"""Custom training losses for eye-mask segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceCriterion(nn.Module):
    """BCE-with-logits + soft Dice combo.

    Args:
        bce_weight: Blend factor for BCE term (0..1). 1.0 => pure BCE, 0.0 => pure Dice.
        eps: Small constant to avoid divide-by-zero in dice.
        reduction: Currently only ``mean`` is supported (kept for future parity).
    """

    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.eps = float(eps)
        if reduction not in {"mean"}:
            raise ValueError("Only 'mean' reduction is supported.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE+Dice loss given logits and binary targets."""
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)

        # Binary cross-entropy with logits.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        # Prepare shapes for Dice: (B, C, H, W)
        probs = logits.sigmoid()
        tgt = targets
        if probs.ndim == 3:
            probs = probs[:, None, ...]
        if tgt.ndim == 3:
            tgt = tgt[:, None, ...]

        intersection = (probs * tgt).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + tgt.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)
        dice = dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

