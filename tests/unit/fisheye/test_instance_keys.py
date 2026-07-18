from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.instance_keys import (
    INSTANCE_KEY_ALGORITHM,
    INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
    INSTANCE_KEY_ORIGIN_CODE_MAP,
    instance_key_attrs,
    mint_detection_instance_keys,
    mint_manual_curation_instance_keys,
    resolve_recording_identity,
)


def test_instance_key_attrs_record_clipped_parent_frame_mapping() -> None:
    attrs = instance_key_attrs(
        "recording_a",
        frame_domain="recording_parent_frame_index",
        frame_mapping_source="/recording_frame_index.parquet",
        frame_mapping_sha256="abc123",
    )

    assert attrs["instance_key_recording_identity"] == "recording_a"
    assert attrs["instance_key_frame_domain"] == "recording_parent_frame_index"
    assert attrs["instance_key_frame_mapping_source"] == "/recording_frame_index.parquet"
    assert attrs["instance_key_frame_mapping_sha256"] == "abc123"


def test_mint_detection_instance_keys_no_context_payload_digest_is_locked() -> None:
    """Determinism lock: payload `rec_a|5|0|500000|500000|200000|400000` must hash
    to this exact digest forever. If this test fails, detect-minted keys on
    re-run would no longer match keys already persisted in recordings."""

    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=np.asarray([5], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.4]], dtype=np.float64),
        class_ids=np.asarray([0], dtype=np.int32),
    )

    assert keys.tolist() == [6452298236220910697]


def test_mint_detection_instance_keys_manual_curation_context_digest_is_locked() -> None:
    """Determinism lock for the namespaced payload
    `rec_a|5|0|500000|500000|200000|400000|context=manual_curation`."""

    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=np.asarray([5], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.4]], dtype=np.float64),
        class_ids=np.asarray([0], dtype=np.int32),
        payload_context=INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
    )

    assert keys.tolist() == [11787561743960230441]


def test_mint_detection_instance_keys_payload_context_namespaces_identical_content() -> None:
    frames = np.asarray([5, 7], dtype=np.int32)
    bboxes = np.asarray(
        [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.2]],
        dtype=np.float64,
    )
    classes = np.asarray([0, 0], dtype=np.int32)

    detect_keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames,
        bbox_norm_coords=bboxes,
        class_ids=classes,
    )
    curation_keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames,
        bbox_norm_coords=bboxes,
        class_ids=classes,
        payload_context=INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
    )
    curation_keys_again = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames,
        bbox_norm_coords=bboxes,
        class_ids=classes,
        payload_context=INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
    )

    # No cross-origin collision on identical content, and contexted minting is
    # deterministic across calls.
    assert not np.any(np.isin(curation_keys, detect_keys))
    np.testing.assert_array_equal(curation_keys, curation_keys_again)


def test_mint_manual_curation_instance_keys_uses_stable_row_identity() -> None:
    kwargs = {
        "recording_identity": "rec_a",
        "frame_indices": np.asarray([5, 5], dtype=np.int32),
        "bbox_norm_coords": np.asarray(
            [[0.5, 0.5, 0.2, 0.4], [0.5, 0.5, 0.2, 0.4]],
            dtype=np.float64,
        ),
        "class_ids": np.asarray([0, 0], dtype=np.int32),
    }

    keys = mint_manual_curation_instance_keys(
        refined_row_ids=np.asarray([10, 11], dtype=np.int64),
        **kwargs,
    )
    repeated = mint_manual_curation_instance_keys(
        refined_row_ids=np.asarray([10, 11], dtype=np.int64),
        **kwargs,
    )

    assert keys[0] != keys[1]
    np.testing.assert_array_equal(keys, repeated)


def test_mint_detection_instance_keys_empty_and_blank_context_match_no_context() -> None:
    kwargs = dict(
        recording_identity="rec_a",
        frame_indices=np.asarray([5], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.4]], dtype=np.float64),
        class_ids=np.asarray([0], dtype=np.int32),
    )

    baseline = mint_detection_instance_keys(**kwargs)

    np.testing.assert_array_equal(mint_detection_instance_keys(**kwargs, payload_context=None), baseline)
    np.testing.assert_array_equal(mint_detection_instance_keys(**kwargs, payload_context=""), baseline)
    np.testing.assert_array_equal(mint_detection_instance_keys(**kwargs, payload_context="   "), baseline)


def test_mint_detection_instance_keys_context_still_breaks_duplicate_ties() -> None:
    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=np.asarray([10, 10], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        payload_context=INSTANCE_KEY_CONTEXT_MANUAL_CURATION,
    )

    assert keys.shape == (2,)
    assert keys[0] != keys[1]


def test_instance_key_origin_code_map_is_stable() -> None:
    assert INSTANCE_KEY_ORIGIN_CODE_MAP == {
        "copied_from_detect": 0,
        "minted_at_curation": 1,
    }


def test_mint_detection_instance_keys_is_deterministic_and_reorder_stable() -> None:
    frames = np.asarray([10, 10, 11], dtype=np.int32)
    bboxes = np.asarray(
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.6, 0.5, 0.1, 0.1],
            [0.5, 0.6, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    classes = np.asarray([0, 0, 1], dtype=np.int32)

    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames,
        bbox_norm_coords=bboxes,
        class_ids=classes,
    )
    order = np.asarray([2, 0, 1], dtype=np.int64)
    reordered_keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames[order],
        bbox_norm_coords=bboxes[order],
        class_ids=classes[order],
    )

    assert keys.dtype == np.uint64
    assert keys.shape == (3,)
    np.testing.assert_array_equal(reordered_keys, keys[order])


def test_mint_detection_instance_keys_breaks_exact_duplicate_content_ties() -> None:
    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=np.asarray([10, 10], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        class_ids=np.asarray([0, 0], dtype=np.int32),
    )

    assert keys.shape == (2,)
    assert keys[0] != keys[1]


def test_mint_detection_instance_keys_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="bbox_norm_coords row count"):
        mint_detection_instance_keys(
            recording_identity="rec_a",
            frame_indices=np.asarray([1, 2], dtype=np.int32),
            bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        )


def test_detect_producers_share_instance_key_policy_helpers() -> None:
    attrs = {"recording_id": "rec_a", "zarr_use": "analysis"}

    assert resolve_recording_identity(attrs, fallback_path="/tmp/other.zarr") == "rec_a"
    assert instance_key_attrs("rec_a")["instance_key_algorithm"] == INSTANCE_KEY_ALGORITHM
