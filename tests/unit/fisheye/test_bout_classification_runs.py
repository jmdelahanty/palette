from __future__ import annotations

import numpy as np
import zarr

from fisheye.analysis.bout_classification_runs import (
    summarize_bout_classification_run,
    validate_bout_classification_run,
)
from fisheye.analysis.megabouts_classifier import (
    classify_megabouts_input_pack,
    write_megabouts_classification_run,
)
from fisheye.analysis.megabouts_classifier_inputs import build_megabouts_classifier_input_pack
from tests.unit.fisheye.test_megabouts_classifier import _build_classifier_root, _fake_runtime


def _classification_root() -> zarr.Group:
    source_root = _build_classifier_root()
    pack = build_megabouts_classifier_input_pack(
        source_root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.9,
        max_consecutive_invalid_frames=1,
    )
    result = classify_megabouts_input_pack(pack, runtime=_fake_runtime())
    out_root = zarr.group()
    write_megabouts_classification_run(
        out_root,
        run_name="classification_001",
        pack=pack,
        result=result,
    )
    return out_root


def test_validate_bout_classification_run_accepts_writer_output() -> None:
    root = _classification_root()

    validation = validate_bout_classification_run(root, "latest", strict=True)

    assert validation["ok"] is True
    assert validation["run_name"] == "classification_001"
    assert validation["schema_id"] == "analysis.bout_classification_runs"
    assert validation["errors"] == []
    assert validation["warnings"] == []


def test_summarize_bout_classification_run_counts_labels_and_skips() -> None:
    root = _classification_root()

    summary = summarize_bout_classification_run(root, "classification_001", strict=True)

    assert summary["ok"] is True
    assert summary["source_bout_count"] == 2
    assert summary["classified_bout_count"] == 1
    assert summary["skipped_bout_count"] == 1
    assert summary["category_counts"] == {
        "skipped_invalid_window": 1,
        "slow2": 1,
    }
    assert summary["failure_reason_counts"] == {
        "ok": 1,
        "traj_valid_fraction_below_threshold": 1,
    }
    assert np.isclose(summary["probability"]["mean"], 0.875)


def test_validate_bout_classification_run_reports_missing_per_bout_field() -> None:
    root = _classification_root()
    per_bout = root["analysis/bout_classification_runs/classification_001/per_bout"]
    del per_bout["probability"]

    validation = validate_bout_classification_run(root, "classification_001")

    assert validation["ok"] is False
    assert "per_bout field listed but missing array: probability" in validation["errors"]


def test_validate_bout_classification_run_strict_promotes_recommended_attrs() -> None:
    root = _classification_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    del run.attrs["trajectory_conversion"]

    non_strict = validate_bout_classification_run(root, "classification_001")
    strict = validate_bout_classification_run(root, "classification_001", strict=True)

    assert non_strict["ok"] is True
    assert strict["ok"] is False
    assert "missing recommended run attr: trajectory_conversion" in strict["warnings"]
