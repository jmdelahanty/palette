"""Unit tests for the step cascade invalidation module."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.status_ledger import upsert_recording_step_status
from fisheye.registry.step_cascade import (
    STEP_DEPENDENTS,
    get_transitive_dependents,
    invalidate_downstream_steps,
)


def _create_registry(tmp_path: Path) -> Registry:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        dataset_id="dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    return registry


def _seed_step(registry: Registry, step_name: str, status: str = "ok") -> None:
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        recording_id="recording_a",
        step_name=step_name,
        status=status,
        run_name=f"{step_name}_run_001",
        method="test_method",
        source="unit_test_seed",
    )


def _get_step_status(registry: Registry, step_name: str) -> str | None:
    row = registry.conn.execute(
        "SELECT status FROM recording_step_status"
        " WHERE dataset_id = ? AND step_name = ?;",
        ("dataset_a", step_name),
    ).fetchone()
    return str(row["status"]) if row is not None else None


def _count_history_rows(registry: Registry, step_name: str) -> int:
    row = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status_history"
        " WHERE dataset_id = ? AND step_name = ?;",
        ("dataset_a", step_name),
    ).fetchone()
    return int(row["n"])


# ---------------------------------------------------------------------------
# get_transitive_dependents tests
# ---------------------------------------------------------------------------


def test_get_transitive_dependents_detect() -> None:
    result = get_transitive_dependents("detect")
    expected = frozenset({
        "refined_detect",
        "detect_quality",
        "crop",
        "keypoints",
        "refined_keypoints",
        "eye_masks",
        "refined_eye_masks",
        "id_assignment",
        "tracks",
    })
    assert result == expected


def test_get_transitive_dependents_refined_keypoints() -> None:
    result = get_transitive_dependents("refined_keypoints")
    expected = frozenset({
        "eye_masks",
        "refined_eye_masks",
        "id_assignment",
        "tracks",
    })
    assert result == expected


def test_get_transitive_dependents_keypoints() -> None:
    result = get_transitive_dependents("keypoints")
    expected = frozenset({
        "refined_keypoints",
        "eye_masks",
        "refined_eye_masks",
        "id_assignment",
        "tracks",
    })
    assert result == expected


def test_get_transitive_dependents_leaf_step() -> None:
    assert get_transitive_dependents("tracks") == frozenset()
    assert get_transitive_dependents("refined_eye_masks") == frozenset()
    assert get_transitive_dependents("detect_quality") == frozenset()


def test_get_transitive_dependents_unknown_step() -> None:
    assert get_transitive_dependents("nonexistent") == frozenset()


# ---------------------------------------------------------------------------
# invalidate_downstream_steps tests
# ---------------------------------------------------------------------------


def test_invalidate_downstream_marks_missing(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    for step in ["detect", "refined_detect", "crop", "keypoints",
                  "refined_keypoints", "eye_masks", "refined_eye_masks",
                  "id_assignment", "tracks", "detect_quality"]:
        _seed_step(registry, step, "ok")

    result = invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="detect",
        source="test_source",
        recording_id="recording_a",
        trigger_run_name="detect_run_002",
    )

    assert "detect" not in result["steps_invalidated"]
    for step in ["refined_detect", "crop", "keypoints", "refined_keypoints",
                  "eye_masks", "refined_eye_masks", "id_assignment", "tracks",
                  "detect_quality"]:
        assert step in result["steps_invalidated"], f"{step} should be invalidated"
        assert _get_step_status(registry, step) == "missing"

    # The triggering step itself should remain "ok"
    assert _get_step_status(registry, "detect") == "ok"
    assert len(result["errors"]) == 0
    registry.close()


def test_invalidate_downstream_skips_already_missing(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    _seed_step(registry, "refined_detect", "ok")
    _seed_step(registry, "crop", "missing")

    result = invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="detect",
        source="test_source",
    )

    assert "crop" in result["steps_skipped"]
    assert "refined_detect" in result["steps_invalidated"]
    registry.close()


def test_invalidate_downstream_skips_absent_and_na(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    _seed_step(registry, "refined_detect", "ok")

    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="crop",
        status="absent",
        source="unit_test_seed",
    )
    upsert_recording_step_status(
        registry,
        dataset_id="dataset_a",
        step_name="keypoints",
        status="na",
        source="unit_test_seed",
    )

    result = invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="detect",
        source="test_source",
    )

    assert "crop" in result["steps_skipped"]
    assert "keypoints" in result["steps_skipped"]
    assert "refined_detect" in result["steps_invalidated"]
    registry.close()


def test_invalidate_downstream_creates_history_rows(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    _seed_step(registry, "refined_keypoints", "ok")
    _seed_step(registry, "eye_masks", "ok")
    _seed_step(registry, "refined_eye_masks", "ok")

    initial_em_count = _count_history_rows(registry, "eye_masks")

    invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="refined_keypoints",
        source="test_source",
        recording_id="recording_a",
    )

    # Each invalidated step should have gotten a new history row
    assert _count_history_rows(registry, "eye_masks") == initial_em_count + 1
    assert _count_history_rows(registry, "refined_eye_masks") > 0
    registry.close()


def test_invalidate_downstream_handles_no_existing_rows(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    # Don't seed any steps — downstream rows don't exist yet

    result = invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="detect",
        source="test_source",
        recording_id="recording_a",
    )

    # All downstream steps should be created as "missing"
    assert len(result["steps_invalidated"]) == 9
    for step in get_transitive_dependents("detect"):
        assert _get_step_status(registry, step) == "missing"
    assert len(result["errors"]) == 0
    registry.close()


def test_invalidate_downstream_details_json(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    _seed_step(registry, "refined_detect", "ok")

    invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="detect",
        source="runtime_predict_detections",
        trigger_run_name="detect_2026-02-28_14-00-00",
    )

    row = registry.conn.execute(
        "SELECT details_json, source FROM recording_step_status"
        " WHERE dataset_id = ? AND step_name = ?;",
        ("dataset_a", "refined_detect"),
    ).fetchone()
    assert row is not None
    details = json.loads(str(row["details_json"]))
    assert details["cascade_trigger_step"] == "detect"
    assert details["cascade_trigger_run"] == "detect_2026-02-28_14-00-00"
    assert str(row["source"]) == "runtime_cascade_invalidation:runtime_predict_detections"
    registry.close()


def test_invalidate_downstream_clears_run_name_and_method(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)
    _seed_step(registry, "crop", "ok")

    invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="refined_detect",
        source="test_source",
    )

    row = registry.conn.execute(
        "SELECT run_name, method, coverage_pct FROM recording_step_status"
        " WHERE dataset_id = ? AND step_name = ?;",
        ("dataset_a", "crop"),
    ).fetchone()
    assert row is not None
    assert row["run_name"] is None
    assert row["method"] is None
    assert row["coverage_pct"] is None
    registry.close()


def test_invalidate_downstream_leaf_step_returns_empty(tmp_path: Path) -> None:
    registry = _create_registry(tmp_path)

    result = invalidate_downstream_steps(
        registry,
        dataset_id="dataset_a",
        step_name="tracks",
        source="test_source",
    )

    assert result["steps_invalidated"] == []
    assert result["steps_skipped"] == []
    assert result["errors"] == []
    registry.close()


# ---------------------------------------------------------------------------
# Graph consistency tests
# ---------------------------------------------------------------------------


def test_graph_no_orphan_dependents() -> None:
    """Every step referenced as a dependent also appears as a key."""
    all_keys = set(STEP_DEPENDENTS.keys())
    for step, dependents in STEP_DEPENDENTS.items():
        for dep in dependents:
            assert dep in all_keys, (
                f"Step {dep!r} (dependent of {step!r}) is not a key in STEP_DEPENDENTS"
            )


def test_graph_no_self_loops() -> None:
    """No step should list itself as a dependent."""
    for step, dependents in STEP_DEPENDENTS.items():
        assert step not in dependents, f"Step {step!r} has a self-loop"
