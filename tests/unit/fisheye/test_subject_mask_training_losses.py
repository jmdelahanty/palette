from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.losses import MaskedBCEDiceCriterion


def test_masked_bce_dice_ignores_unsupervised_channels() -> None:
    criterion = MaskedBCEDiceCriterion(bce_weight=0.5)

    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    targets_a = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]],
        dtype=torch.float32,
    )
    targets_b = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    valid = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss_a = criterion(logits, targets_a, valid)
    loss_b = criterion(logits, targets_b, valid)

    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
    assert torch.allclose(loss_a, loss_b)
