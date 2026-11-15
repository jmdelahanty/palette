"""Successor format v3: legacy-first, head-anchored-fallback tail method selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.subject_shape_runs import HEAD_ANCHORED_CENTERLINE_METHOD
from fisheye.training import mask_tail_apply_refresh as refresh_mod
from fisheye.training.mask_tail_keypoints import SCHEMA_NAME, derive_tail_seed

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
# RedScare long-snout rows from #192, and 2026-01-28T20-51-00Z arena 3 ROI 0
# (a tightly curled fish; applied mask of version mask_apply_af4937b3...,
# cropped around the body; see crop_origin_xy): legacy fails with
# snout_extension_no_mask_path, head-anchored derives a valid tail.
REDSCARE = FIXTURES / "redscare_head_anchored_tail_rows_v1.npz"
ARENA3 = FIXTURES / "arena3_snout_no_mask_path_row_v1.npz"


def _load(path):
    data = np.load(path, allow_pickle=False)
    return data["masks"], tuple(str(v) for v in data["labels"]), data["head"]


@pytest.mark.parametrize("fixture", [REDSCARE, ARENA3], ids=["too_long", "no_mask_path"])
def test_fallback_codes_follow_the_rule_row_by_row(fixture):
    masks, labels, head = _load(fixture)
    codes = refresh_mod._fallback_method_codes(
        masks, labels, head, schema_name=SCHEMA_NAME, accepted={}
    )
    legacy = derive_tail_seed(masks, labels, head, schema_name=SCHEMA_NAME)
    anchored = derive_tail_seed(
        masks, labels, head, schema_name=SCHEMA_NAME, method=HEAD_ANCHORED_CENTERLINE_METHOD
    )
    for row in range(len(masks)):
        eligible = (
            not bool(legacy["tail_valid"][row])
            and str(legacy["tail_failure_reason"][row]) in refresh_mod.FALLBACK_FAILURE_REASONS
        )
        expected = 1 if eligible and bool(anchored["tail_valid"][row]) else 0
        assert codes[row] == expected, (row, legacy["tail_failure_reason"][row])
    assert codes.dtype == np.uint8


def test_curled_fish_row_takes_the_head_anchored_path():
    masks, labels, head = _load(ARENA3)
    codes = refresh_mod._fallback_method_codes(
        masks, labels, head, schema_name=SCHEMA_NAME, accepted={}
    )
    assert codes.tolist() == [1]


def test_rows_legacy_derives_never_switch_method():
    masks, labels, head = _load(REDSCARE)
    legacy = derive_tail_seed(masks, labels, head, schema_name=SCHEMA_NAME)
    codes = refresh_mod._fallback_method_codes(
        masks, labels, head, schema_name=SCHEMA_NAME, accepted={}
    )
    valid = np.asarray(legacy["tail_valid"], dtype=bool)
    assert not np.any(codes[valid])


def test_v3_is_the_default_and_references_its_crop():
    assert refresh_mod.DEFAULT_SUCCESSOR_FORMAT == "v3"
    assert refresh_mod.SUCCESSOR_FORMAT_SCHEMAS[refresh_mod.REFRESH_POLICY_V3] == refresh_mod.REFRESH_SCHEMA_V3
    from fisheye.shared.recovered_training_review_contract import REFERENCED_CROP_SUCCESSOR_POLICIES

    assert refresh_mod.REFRESH_POLICY_V3 in REFERENCED_CROP_SUCCESSOR_POLICIES


def test_fallback_is_taken_only_when_head_anchored_succeeds(monkeypatch):
    """Rows 0-1 fail at the snout join, row 2 for another reason, row 3 is fine.

    Head-anchored succeeds on row 0 only; row 1 must stay legacy (it keeps
    the legacy failure rather than a second one), and rows 2-3 are never
    retried.
    """
    retried = []

    def fake_derive(masks, labels, head, *, schema_name, method="legacy", **_kwargs):
        n = len(masks)
        if method == "legacy":
            return {
                "tail_valid": np.array([False, False, False, True]),
                "tail_failure_reason": np.array(
                    ["snout_extension_no_mask_path", "snout_extension_too_long",
                     "fragmented_subject_body_mask", "ok"]
                ),
            }
        retried.append(n)
        return {"tail_valid": np.array([True, False][:n])}

    monkeypatch.setattr(refresh_mod, "derive_tail_seed", fake_derive)
    codes = refresh_mod._fallback_method_codes(
        np.zeros((4, 3, 8, 8), np.uint8), ("subject_body", "eyes_union", "swim_bladder"),
        np.zeros((4, 3, 2)), schema_name=SCHEMA_NAME, accepted={},
    )
    assert codes.tolist() == [1, 0, 0, 0]
    assert retried == [2]  # Only the two snout-join failures are retried.
