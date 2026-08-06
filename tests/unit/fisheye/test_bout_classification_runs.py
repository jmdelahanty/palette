from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis.bout_classification_runs import (
    resolve_bout_classification_run,
    summarize_bout_classification_run,
    validate_bout_classification_run,
)
from fisheye.analysis.bout_classification_schema import (
    BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR,
    BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR,
    BoutClassificationDimensions,
    bout_classification_manifest_digest,
    validate_bout_classification_arrays,
)
from fisheye.analysis.megabouts_classifier import (
    classify_megabouts_input_pack,
    write_megabouts_classification_run,
)
from tests.unit.fisheye.test_megabouts_classifier import (
    _build_classifier_root,
    _fake_runtime,
)
from tests.unit.fisheye.test_megabouts_classifier_inputs import (
    build_megabouts_classifier_input_pack,
    _install_verified_source_readers,
)


@pytest.fixture(autouse=True)
def _verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_source_readers(monkeypatch)


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


def _resolver_root() -> zarr.Group:
    root = zarr.group()
    parent = root.require_group("analysis/bout_classification_runs")
    run = parent.create_group("classification_001")
    run.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    parent.attrs["latest"] = "classification_001"
    parent.attrs["latest_complete"] = "classification_001"
    return root


def test_validate_bout_classification_run_accepts_writer_output() -> None:
    root = _classification_root()

    validation = validate_bout_classification_run(root, "latest", strict=True)

    assert validation["ok"] is True
    assert validation["run_name"] == "classification_001"
    assert validation["schema_id"] == "analysis.bout_classification_runs"
    assert validation["errors"] == []
    assert validation["warnings"] == []


def test_writer_freezes_exact_v2_array_inventory_and_text_widths() -> None:
    root = _classification_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    per_bout = run["per_bout"]

    assert run.attrs["schema_version"] == 2
    manifest = run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR]
    assert manifest["byte_planner_adopted"] is False
    assert len(manifest["arrays"]) == 20
    assert run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR] == (
        bout_classification_manifest_digest(manifest)
    )
    assert per_bout["category_label_bytes"].dtype == np.dtype("uint8")
    assert per_bout["category_label_bytes"].shape == (2, 64)
    assert per_bout["failure_reason_bytes"].dtype == np.dtype("uint8")
    assert per_bout["failure_reason_bytes"].shape == (2, 128)


def test_recomputed_digest_cannot_authorize_tampered_array_manifest() -> None:
    root = _classification_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    manifest = run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR]
    manifest["arrays"][0]["logical_contract"]["dtype"]["dtype_id"] = "float64"
    run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR] = manifest
    run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR] = (
        bout_classification_manifest_digest(manifest)
    )

    issues = validate_bout_classification_arrays(
        run,
        dimensions=BoutClassificationDimensions(n_bouts=2),
    )

    assert {issue.code for issue in issues} == {
        "array_schema_manifest_mismatch",
        "array_schema_digest_mismatch",
    }


def test_wrong_physical_dtype_fails_exact_array_validation() -> None:
    root = _classification_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    per_bout = run["per_bout"]
    values = np.asarray(per_bout["source_bout_id"][:], dtype=np.float64)
    del per_bout["source_bout_id"]
    per_bout.create_array("source_bout_id", data=values)

    validation = validate_bout_classification_run(root, "classification_001")

    assert validation["ok"] is False
    assert any("dtype mismatch" in error for error in validation["errors"])


def test_legacy_v1_requires_explicit_compatibility_opt_in() -> None:
    root = _classification_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    run.attrs["schema_version"] = 1
    del run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR]
    del run.attrs[BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR]

    assert validate_bout_classification_run(root, "classification_001")["ok"] is False
    assert (
        validate_bout_classification_run(
            root,
            "classification_001",
            legacy_compatibility=True,
        )["ok"]
        is True
    )


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
    assert (
        "per_bout field listed but missing array: probability" in validation["errors"]
    )


def test_validate_bout_classification_run_strict_promotes_recommended_attrs() -> None:
    root = _classification_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    del run.attrs["trajectory_conversion"]

    non_strict = validate_bout_classification_run(root, "classification_001")
    strict = validate_bout_classification_run(root, "classification_001", strict=True)

    assert non_strict["ok"] is True
    assert strict["ok"] is False
    assert "missing recommended run attr: trajectory_conversion" in strict["warnings"]


@pytest.mark.parametrize(
    "run_spec",
    (
        "classification_001",
        "analysis/bout_classification_runs/classification_001",
    ),
)
def test_explicit_resolution_accepts_only_controlled_name_forms(run_spec: str) -> None:
    root = _resolver_root()

    _run, run_name, run_path = resolve_bout_classification_run(root, run_spec)

    assert run_name == "classification_001"
    assert run_path == "analysis/bout_classification_runs/classification_001"


@pytest.mark.parametrize(
    "run_spec",
    (
        "analysis/swim_bout_runs/classification_001",
        "analysis/tail_kinematics_runs/classification_001",
        "nested/classification_001",
        "analysis/bout_classification_runs/nested/classification_001",
        "analysis/bout_classification_runs",
        "/analysis/bout_classification_runs/classification_001",
        "analysis/bout_classification_runs/classification_001/",
        "analysis/bout_classification_runs//classification_001",
    ),
)
def test_explicit_resolution_rejects_wrong_family_or_nested_paths(
    run_spec: str,
) -> None:
    root = _resolver_root()

    with pytest.raises(ValueError, match="bare child name or the exact path"):
        resolve_bout_classification_run(root, run_spec)


@pytest.mark.parametrize(
    ("status", "eligible", "message"),
    (
        ("running", True, "not complete"),
        ("failed", True, "not complete"),
        ("complete", False, "not selector-eligible"),
    ),
)
def test_explicit_resolution_requires_complete_selector_eligible_run(
    status: str,
    eligible: bool,
    message: str,
) -> None:
    root = _resolver_root()
    run = root["analysis/bout_classification_runs/classification_001"]
    run.attrs["palette_run_completion_status"] = status
    run.attrs["stage_selector_eligible"] = eligible

    with pytest.raises(ValueError, match=message):
        resolve_bout_classification_run(root, "classification_001")


def test_implicit_resolution_never_guesses_during_selector_handoff() -> None:
    root = _resolver_root()
    parent = root["analysis/bout_classification_runs"]
    candidate = parent.create_group("candidate")
    candidate.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    parent.attrs["latest"] = "candidate"
    parent.attrs["latest_complete"] = "candidate"

    with pytest.raises(ValueError, match="activation may be in progress"):
        resolve_bout_classification_run(root)

    _run, run_name, _run_path = resolve_bout_classification_run(
        root,
        "classification_001",
    )
    assert run_name == "classification_001"


@pytest.mark.parametrize(
    ("latest", "latest_complete"),
    (
        ("classification_001", "candidate"),
        ("candidate", "classification_001"),
    ),
)
def test_implicit_resolution_rejects_each_intermediate_selector_pair(
    latest: str,
    latest_complete: str,
) -> None:
    root = _resolver_root()
    parent = root["analysis/bout_classification_runs"]
    candidate = parent.create_group("candidate")
    candidate.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    parent.attrs["latest"] = latest
    parent.attrs["latest_complete"] = latest_complete

    with pytest.raises(ValueError, match="activation may be in progress"):
        resolve_bout_classification_run(root)

    _run, run_name, _run_path = resolve_bout_classification_run(
        root,
        "classification_001",
    )
    assert run_name == "classification_001"
