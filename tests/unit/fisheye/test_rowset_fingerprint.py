from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.rowset_fingerprint import (
    ROWSET_FINGERPRINT_STATUS_MISSING_INSTANCE_KEY,
    RowsetFingerprintError,
    assert_rowset_fingerprint_matches,
    build_rowset_fingerprint,
)


def test_rowset_fingerprint_is_physical_order_independent() -> None:
    first = build_rowset_fingerprint(
        source_rowset_path="crop_runs/crop_a",
        row_count=3,
        instance_keys=np.array([30, 10, 20], dtype=np.uint64),
        source_edit_revision=4,
    )
    reordered = build_rowset_fingerprint(
        source_rowset_path="crop_runs/crop_a",
        row_count=3,
        instance_keys=np.array([20, 30, 10], dtype=np.uint64),
        source_edit_revision=4,
    )

    assert first.is_complete
    assert first.fingerprint == reordered.fingerprint
    assert first.instance_key_digest == reordered.instance_key_digest
    assert_rowset_fingerprint_matches(first, reordered, require_complete=True)


def test_rowset_fingerprint_detects_identity_and_revision_changes() -> None:
    expected = build_rowset_fingerprint(
        source_rowset_path="refined_detect_runs/refined_a/instances",
        row_count=2,
        instance_keys=np.array([1, 2], dtype=np.uint64),
        source_edit_revision=7,
    )
    changed_keys = build_rowset_fingerprint(
        source_rowset_path="refined_detect_runs/refined_a/instances",
        row_count=2,
        instance_keys=np.array([1, 3], dtype=np.uint64),
        source_edit_revision=7,
    )
    changed_revision = build_rowset_fingerprint(
        source_rowset_path="refined_detect_runs/refined_a/instances",
        row_count=2,
        instance_keys=np.array([1, 2], dtype=np.uint64),
        source_edit_revision=8,
    )

    with pytest.raises(RowsetFingerprintError, match="fingerprint changed"):
        assert_rowset_fingerprint_matches(expected, changed_keys)
    with pytest.raises(RowsetFingerprintError, match="edit revision changed"):
        assert_rowset_fingerprint_matches(expected, changed_revision)


def test_rowset_fingerprint_marks_legacy_missing_keys_explicitly() -> None:
    legacy = build_rowset_fingerprint(
        source_rowset_path="detect_runs/legacy",
        row_count=2,
        instance_keys=None,
    )

    assert legacy.fingerprint is None
    assert legacy.status == ROWSET_FINGERPRINT_STATUS_MISSING_INSTANCE_KEY
    with pytest.raises(RowsetFingerprintError, match="complete"):
        assert_rowset_fingerprint_matches(legacy, legacy, require_complete=True)


def test_rowset_fingerprint_rejects_duplicate_keys() -> None:
    with pytest.raises(RowsetFingerprintError, match="Duplicate instance_key"):
        build_rowset_fingerprint(
            source_rowset_path="detect_runs/bad",
            row_count=2,
            instance_keys=np.array([5, 5], dtype=np.uint64),
        )
