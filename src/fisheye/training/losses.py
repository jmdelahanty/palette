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
        overlap_weight: Optional penalty weight for cross-channel overlap
            (useful for left/right eye channels). Ignored when channel count < 2.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        eps: float = 1e-6,
        reduction: str = "mean",
        overlap_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.eps = float(eps)
        self.overlap_weight = float(overlap_weight)
        if self.overlap_weight < 0.0:
            raise ValueError("overlap_weight must be >= 0.0")
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

        overlap_penalty = torch.zeros((), dtype=probs.dtype, device=probs.device)
        if self.overlap_weight > 0.0 and probs.shape[1] >= 2:
            overlap_penalty = (probs[:, 0] * probs[:, 1]).mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice + self.overlap_weight * overlap_penalty


class MaskedBCEDiceCriterion(nn.Module):
    """BCE-with-logits + soft Dice combo with per-channel supervision masking."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.eps = float(eps)
        if reduction not in {"mean"}:
            raise ValueError("Only 'mean' reduction is supported.")
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_channels: torch.Tensor,
    ) -> torch.Tensor:
        if targets.dtype != logits.dtype:
            targets = targets.to(logits.dtype)
        valid = valid_channels.to(device=logits.device)
        if valid.dtype != logits.dtype:
            valid = valid.to(logits.dtype)

        if logits.ndim == 3:
            logits = logits[:, None, ...]
        if targets.ndim == 3:
            targets = targets[:, None, ...]
        if valid.ndim == 2:
            valid = valid[:, :, None, None]
        elif valid.ndim == 3:
            valid = valid[:, :, :, None]
        if valid.shape != logits.shape:
            valid = valid.expand_as(logits)

        valid_weight = valid.sum()
        if torch.all(valid_weight <= 0):
            return torch.zeros((), dtype=logits.dtype, device=logits.device)

        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (bce_map * valid).sum() / torch.clamp(valid_weight, min=1.0)

        probs = logits.sigmoid()
        intersection = (probs * targets * valid).sum(dim=(2, 3))
        denom = ((probs + targets) * valid).sum(dim=(2, 3))
        channel_valid = valid[..., 0, 0] > 0
        dice_per_channel = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)
        if torch.any(channel_valid):
            dice = dice_per_channel[channel_valid].mean()
        else:
            dice = torch.zeros((), dtype=logits.dtype, device=logits.device)

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
