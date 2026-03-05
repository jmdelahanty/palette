import pytest
import torch

from fisheye.training.losses import BCEDiceCriterion


def test_bce_dice_overlap_penalty_increases_lr_loss() -> None:
    logits = torch.zeros((2, 2, 8, 8), dtype=torch.float32)
    targets = torch.zeros_like(logits)
    targets[:, 0] = 1.0

    base = BCEDiceCriterion(bce_weight=0.5, overlap_weight=0.0)
    penalized = BCEDiceCriterion(bce_weight=0.5, overlap_weight=0.2)

    loss_base = base(logits, targets)
    loss_penalized = penalized(logits, targets)

    overlap = (torch.sigmoid(logits)[:, 0] * torch.sigmoid(logits)[:, 1]).mean()
    expected_delta = 0.2 * overlap

    assert loss_penalized > loss_base
    assert torch.isclose(loss_penalized - loss_base, expected_delta, atol=1e-6)


def test_bce_dice_overlap_penalty_ignored_for_single_channel() -> None:
    logits = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    targets = torch.zeros_like(logits)
    targets[:, 0, 2:6, 2:6] = 1.0

    base = BCEDiceCriterion(bce_weight=0.5, overlap_weight=0.0)
    penalized = BCEDiceCriterion(bce_weight=0.5, overlap_weight=1.0)

    loss_base = base(logits, targets)
    loss_penalized = penalized(logits, targets)

    assert torch.isclose(loss_penalized, loss_base, atol=1e-6)


def test_bce_dice_overlap_penalty_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="overlap_weight must be >= 0.0"):
        BCEDiceCriterion(overlap_weight=-0.01)
