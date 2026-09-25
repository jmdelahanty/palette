"""Activating a canonical-v3 successor must not change lineage resolution.

Downstream products (refined detections, detect-quality reports) record the
legacy raw detection run name.  After ``activate_canonical_detection_successor``
moves ``detect_runs.latest``/``latest_complete`` to the successor, every
migrated consumer must resolve exactly what it resolved before activation, and
only a successor whose sealed manifest validates may grant that equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import zarr

from rich.console import Console

from fisheye.analysis.track_kinematics import (
    prefer_refined_detection,
    resolve_detection_from_path,
)
from fisheye.analysis.chaser_phase_analysis import _collect_pipeline_provenance
from fisheye.diagnostics.check_provenance_consistency import _collect_provenance
from fisheye.refinement.refine_detect import _resolve_detection_quality_labels
from fisheye.registry.maintenance import _build_recording_step_rows_from_root
from fisheye.utils.audit_analysis_staleness import SourceRef, audit_source_ref
from fisheye.utils.detect_quality_batch import _build_plans
from fisheye.shared.zarr.canonical_detection_activation import (
    canonical_detection_lineage_equivalent_runs,
)
from fisheye.shared.zarr_helpers import reconsolidate_zarr_metadata

from tests.unit.fisheye.test_canonical_detection_successor_activation import (
    SUCCESSOR,
    _activate,
    _build_archive,
)

LEGACY = "detect_source"
REFINED = "refined_detect_from_legacy"
QUALITY = "quality_legacy"
ARENA = "arena_from_legacy"


def _complete_attrs(**extra: Any) -> dict[str, Any]:
    return {
        "status": "complete",
        "palette_run_completion_status": "complete",
        **extra,
    }


def _add_lineage_products(archive: Path) -> None:
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    refined_parent = root.require_group("refined_detect_runs")
    refined_parent.attrs["latest"] = REFINED
    refined_parent.create_group(
        REFINED,
        attributes=_complete_attrs(source_detect_run=LEGACY, method="refine"),
    )
    quality_parent = root[f"detect_runs/{LEGACY}"].require_group("quality_reports")
    quality_parent.attrs["latest"] = QUALITY
    quality = quality_parent.create_group(
        QUALITY,
        attributes=_complete_attrs(source_detect_run=LEGACY),
    )
    quality.create_array(
        "detection_quality_labels", data=np.asarray([0, 1], dtype=np.int8)
    )
    arena_parent = root.require_group("arena_assignment_runs")
    arena_parent.attrs["latest"] = ARENA
    arena = arena_parent.create_group(
        ARENA,
        attributes=_complete_attrs(source_detect_run=LEGACY),
    )
    arena.create_array("arena_ids", data=np.asarray([0, 1], dtype=np.int32))
    zarr.consolidate_metadata(str(archive))


@pytest.fixture
def lineage_archive(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "recording_analysis.zarr"
    _build_archive(archive)
    _add_lineage_products(archive)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return archive, scratch


def _step_rows(archive: Path) -> dict[str, dict[str, Any]]:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    rows = _build_recording_step_rows_from_root(
        root=root,
        dataset_id="dataset_lineage",
        recording_id="recording_lineage",
        zarr_use="analysis",
        zarr_mtime_ns=None,
        source="pytest",
    )
    return {str(row["step_name"]): row for row in rows}


def _maintenance_resolution(archive: Path) -> dict[str, object]:
    rows = _step_rows(archive)
    refined = rows["refined_detect"]
    quality = rows["detect_quality"]
    return {
        "refined_detect_run": refined["run_name"],
        "refined_detect_status": refined["status"],
        "detect_quality_run": quality["run_name"],
        "detect_quality_status": quality["status"],
    }


def _track_kinematics_resolution(archive: Path) -> dict[str, object]:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    selected = resolve_detection_from_path(
        root, f"detect_runs/{root['detect_runs'].attrs['latest']}"
    )
    preferred = prefer_refined_detection(root, selected, Console(quiet=True))
    return {
        "path": preferred.path,
        "is_refined": preferred.is_refined,
        "source_detect_run": preferred.source_detect_run,
    }


def _refine_detect_quality_resolution(archive: Path) -> dict[str, object]:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    selected = str(root["detect_runs"].attrs["latest"])
    labels, quality_run, _group = _resolve_detection_quality_labels(
        root,
        root[f"detect_runs/{selected}"],
        detect_run=selected,
        source_detect_path=f"detect_runs/{selected}",
        quality_run=None,
        quality_group_path=None,
        total_detections=2,
        require_quality=True,
        allow_missing_reason="pytest",
        console=None,
    )
    return {"quality_run": quality_run, "labels": labels.tolist()}


def _staleness_resolution(archive: Path) -> str:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    return audit_source_ref(
        root,
        SourceRef(key="source_detect_run", path=f"detect_runs/{LEGACY}"),
        zarr_path=archive,
    ).status


def _lineage_issues(issues: list[str]) -> list[str]:
    return sorted(issue for issue in issues if "references detect" in issue)


def _chaser_provenance_resolution(archive: Path) -> list[str]:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    return _lineage_issues(_collect_pipeline_provenance(root)["issues"])


def _provenance_consistency_resolution(archive: Path) -> list[str]:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    return _lineage_issues(_collect_provenance(root).issues)


def _detect_quality_batch_resolution(archive: Path) -> tuple[str, bool]:
    (plan,) = _build_plans([archive], False, None, True)
    return plan.status, plan.quality_present


CONSUMERS: dict[str, Callable[[Path], object]] = {
    "registry_step_status": _maintenance_resolution,
    "track_kinematics_prefer_refined": _track_kinematics_resolution,
    "refine_detect_quality_labels": _refine_detect_quality_resolution,
    "analysis_staleness_audit": _staleness_resolution,
    "chaser_phase_provenance": _chaser_provenance_resolution,
    "provenance_consistency": _provenance_consistency_resolution,
    "detect_quality_batch_plan": _detect_quality_batch_resolution,
}


def _resolve_all(archive: Path) -> dict[str, object]:
    return {name: consumer(archive) for name, consumer in CONSUMERS.items()}


def _selected_detection(archive: Path) -> str:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    return str(root["detect_runs"].attrs["latest"])


def test_consumers_resolve_identically_after_real_activation(
    lineage_archive: tuple[Path, Path],
) -> None:
    archive, scratch = lineage_archive
    before = _resolve_all(archive)
    assert before["registry_step_status"] == {
        "refined_detect_run": REFINED,
        "refined_detect_status": "ok",
        "detect_quality_run": QUALITY,
        "detect_quality_status": "ok",
    }
    assert before["track_kinematics_prefer_refined"] == {
        "path": f"refined_detect_runs/{REFINED}",
        "is_refined": True,
        "source_detect_run": LEGACY,
    }
    assert before["refine_detect_quality_labels"] == {
        "quality_run": QUALITY,
        "labels": [0, 1],
    }
    assert before["analysis_staleness_audit"] != "source_not_latest"
    assert before["chaser_phase_provenance"] == []
    assert before["provenance_consistency"] == []
    assert before["detect_quality_batch_plan"] == ("skipped", True)
    assert canonical_detection_lineage_equivalent_runs(archive, LEGACY) == {LEGACY}

    result = _activate(archive, scratch, apply=True)

    assert result["status"] == "activated"
    assert _selected_detection(archive) == SUCCESSOR
    assert canonical_detection_lineage_equivalent_runs(archive, SUCCESSOR) == {
        SUCCESSOR,
        LEGACY,
    }
    assert canonical_detection_lineage_equivalent_runs(archive, LEGACY) == {
        LEGACY,
        SUCCESSOR,
    }
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert canonical_detection_lineage_equivalent_runs(root, SUCCESSOR) == {
        SUCCESSOR,
        LEGACY,
    }
    assert _resolve_all(archive) == before


def _tamper(archive: Path, mutate: Callable[[Any], None]) -> None:
    run = zarr.open_group(
        str(archive / "detect_runs" / SUCCESSOR), mode="r+", use_consolidated=False
    )
    mutate(run)
    reconsolidate_zarr_metadata(archive, policy="pytest_tamper", fail_on_error=True)


def _tamper_manifest_source(run: Any) -> None:
    manifest = json.loads(json.dumps(dict(run.attrs["run_manifest"])))
    manifest["payload"]["source_evidence"]["source_run_id"] = "other_run"
    run.attrs["run_manifest"] = manifest


def _drop_activation_marker(run: Any) -> None:
    del run.attrs["production_selector_activation"]


def _mark_rolled_back(run: Any) -> None:
    run.attrs["stage_selector_eligible"] = False


@pytest.mark.parametrize(
    "mutate",
    [_tamper_manifest_source, _drop_activation_marker, _mark_rolled_back],
    ids=["tampered_manifest", "no_activation_marker", "rolled_back"],
)
def test_invalid_successor_grants_no_equivalence(
    lineage_archive: tuple[Path, Path],
    mutate: Callable[[Any], None],
) -> None:
    archive, scratch = lineage_archive
    _activate(archive, scratch, apply=True)
    _tamper(archive, mutate)

    assert canonical_detection_lineage_equivalent_runs(archive, SUCCESSOR) == {
        SUCCESSOR
    }
    assert canonical_detection_lineage_equivalent_runs(archive, LEGACY) == {LEGACY}
    resolved = _maintenance_resolution(archive)
    assert resolved["refined_detect_run"] is None
    assert resolved["detect_quality_run"] is None
    assert _track_kinematics_resolution(archive)["is_refined"] is False
    assert _staleness_resolution(archive) == "source_not_latest"
    assert _chaser_provenance_resolution(archive) != []
    assert _provenance_consistency_resolution(archive) != []
    assert _detect_quality_batch_resolution(archive)[1] is False


def test_missing_legacy_source_grants_no_equivalence(
    lineage_archive: tuple[Path, Path],
) -> None:
    archive, scratch = lineage_archive
    _activate(archive, scratch, apply=True)
    legacy = archive / "detect_runs" / LEGACY
    legacy.rename(archive / "detect_runs_removed_legacy")

    assert canonical_detection_lineage_equivalent_runs(archive, SUCCESSOR) == {
        SUCCESSOR
    }


def test_non_local_or_empty_inputs_grant_no_equivalence(tmp_path: Path) -> None:
    memory_root = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    memory_root.create_group("detect_runs").create_group(LEGACY)

    assert canonical_detection_lineage_equivalent_runs(memory_root, LEGACY) == {
        LEGACY
    }
    assert canonical_detection_lineage_equivalent_runs(tmp_path, None) == frozenset()
    assert canonical_detection_lineage_equivalent_runs(tmp_path / "absent", LEGACY) == {
        LEGACY
    }


def test_equivalence_survives_moving_the_archive(
    lineage_archive: tuple[Path, Path], tmp_path: Path
) -> None:
    """Archives move between storage roots; equivalence is archive-relative."""

    import shutil

    archive, scratch = lineage_archive
    before = _resolve_all(archive)
    assert _activate(archive, scratch, apply=True)["status"] == "activated"
    moved = tmp_path / "moved_root" / archive.name
    moved.parent.mkdir()
    shutil.copytree(archive, moved)
    assert canonical_detection_lineage_equivalent_runs(moved, SUCCESSOR) == {
        SUCCESSOR,
        LEGACY,
    }
    assert _resolve_all(moved) == before
