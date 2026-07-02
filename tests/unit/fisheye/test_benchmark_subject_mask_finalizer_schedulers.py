from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics import benchmark_subject_mask_finalizer_schedulers as mod


def test_row_chunks_builds_contiguous_ranges() -> None:
    assert mod._row_chunks(start_row=10, roi_count=5, chunk_size=2) == [
        (10, 12),
        (12, 14),
        (14, 15),
    ]


def test_normalize_scheduler_accepts_alias_and_rejects_unknown() -> None:
    assert mod._normalize_scheduler("single_thread") == "single-threaded"
    assert mod._normalize_scheduler("distributed") == "distributed"
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        mod._normalize_scheduler("bogus")


def test_decode_probabilities_handles_uint8_linear_encoding() -> None:
    decoded = mod._decode_probabilities(
        np.asarray([0, 128, 255], dtype=np.uint8),
        encoding="linear_uint8_0_255",
        source_path="subject_mask_runs/run/mask_probs_roi",
    )

    np.testing.assert_allclose(decoded, np.asarray([0.0, 128 / 255.0, 1.0], dtype=np.float32))


def test_aggregate_chunk_results_combines_hashes_and_review_counts() -> None:
    payload = mod._aggregate_chunk_results(
        [
            {
                "start_row": 0,
                "row_count": 2,
                "component_hashes": {"subject_body": "a"},
                "reason_hashes": {"subject_body": "ra"},
                "review_counts": {"subject_body": {"pending": 2}},
                "timing_seconds": {"total": 0.5},
            },
            {
                "start_row": 2,
                "row_count": 1,
                "component_hashes": {"subject_body": "b"},
                "reason_hashes": {"subject_body": "rb"},
                "review_counts": {"subject_body": {"needs_review": 1}},
                "timing_seconds": {"total": 0.25},
            },
        ]
    )

    assert payload["row_count"] == 3
    assert set(payload["component_hashes"]) == {"subject_body"}
    assert set(payload["reason_hashes"]) == {"subject_body"}
    assert payload["review_counts"] == {"subject_body": {"pending": 2, "needs_review": 1}}
    assert payload["chunk_timing_seconds_sum"]["total"] == pytest.approx(0.75)
