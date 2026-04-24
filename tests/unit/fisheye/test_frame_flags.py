from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisheye.shared.frame_flags import (
    append_flagged_frame,
    load_frame_flags,
    resolve_flagged_roi_indices,
    row_identity_payload,
)


def test_append_flagged_frame_upgrades_legacy_entry_with_row_identity(tmp_path: Path) -> None:
    flag_path = tmp_path / "flags.json"
    zarr_path = "/tmp/recording_training.zarr"

    append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=5)
    append_flagged_frame(
        flag_path,
        zarr_path,
        frame_idx=20,
        roi_idx=5,
        extra_fields={"source_refined_row_id": 105, "source_detect_row_index": 9},
    )

    payload = json.loads(flag_path.read_text(encoding="utf-8"))
    assert payload[zarr_path] == [
        {
            "frame_idx": 20,
            "roi_idx": 5,
            "source_detect_row_index": 9,
            "source_refined_row_id": 105,
        }
    ]


def test_resolve_flagged_roi_indices_prefers_refined_row_id_over_stale_roi() -> None:
    entries = [
        {
            "frame_idx": 12,
            "roi_idx": 0,
            "source_refined_row_id": 102,
        }
    ]

    out = resolve_flagged_roi_indices(
        entries,
        total_rois=3,
        frame_indices=np.array([12, 12, 13], dtype=np.int64),
        source_refined_row_ids=np.array([100, 101, 102], dtype=np.int64),
    )

    np.testing.assert_array_equal(out, np.array([2], dtype=np.int32))


def test_resolve_flagged_roi_indices_skips_removed_stable_id_instead_of_stale_roi() -> None:
    entries = [
        {
            "frame_idx": 12,
            "roi_idx": 0,
            "source_refined_row_id": 999,
        }
    ]

    out = resolve_flagged_roi_indices(
        entries,
        total_rois=3,
        frame_indices=np.array([12, 12, 13], dtype=np.int64),
        source_refined_row_ids=np.array([100, 101, 102], dtype=np.int64),
    )

    assert out.size == 0


def test_resolve_flagged_roi_indices_uses_legacy_fallback_when_no_identity_lookup() -> None:
    entries = [
        {
            "frame_idx": 12,
            "roi_idx": 0,
            "source_refined_row_id": 999,
        },
        {"frame_idx": 13, "roi_idx": None},
    ]

    out = resolve_flagged_roi_indices(
        entries,
        total_rois=3,
        frame_indices=np.array([12, 12, 13], dtype=np.int64),
    )

    np.testing.assert_array_equal(out, np.array([0, 2], dtype=np.int32))


def test_row_identity_payload_omits_negative_raw_source_rows() -> None:
    payload = row_identity_payload(
        1,
        source_refined_row_ids=np.array([100, 101], dtype=np.int64),
        source_detect_row_index=np.array([3, -1], dtype=np.int32),
    )

    assert payload == {"source_refined_row_id": 101}


def test_load_frame_flags_preserves_identity_and_allowed_extra_metadata(tmp_path: Path) -> None:
    flag_path = tmp_path / "flags.json"
    flag_path.write_text(
        json.dumps(
            {
                "/tmp/a.zarr": [
                    {
                        "frame_idx": "12",
                        "roi_idx": "3",
                        "source_refined_row_id": "88",
                        "source_detect_row_index": "7",
                        "action": "keypoint_nudge",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    parsed = load_frame_flags(flag_path, preserve_extra_keys=("action",))

    assert parsed["/tmp/a.zarr"] == [
        {
            "action": "keypoint_nudge",
            "frame_idx": 12,
            "roi_idx": 3,
            "source_detect_row_index": 7,
            "source_refined_row_id": 88,
        }
    ]
