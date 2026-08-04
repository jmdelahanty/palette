from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis import bout_kinematics_schema, swim_bout_schema
from fisheye.analysis._exact_tabular_run_schema import MANIFEST_ATTRIBUTE
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis_workflows.materializers import exact_tabular_candidate
from fisheye.analysis_workflows.materializers.exact_tabular_candidate import (
    build_exact_tabular_candidate_plan,
    materialize_exact_tabular_candidate,
)


_COMPLETE_PHASE_SEQUENCE = [
    "plan",
    "source_staging",
    "logical_rematerialization",
    "local_validation",
    "local_consolidation",
    "local_direct_consolidated_comparison",
    "atomic_publication",
    "published_validation",
    "published_direct_consolidated_comparison",
    "decoded_equality",
    "physical_inventory",
]


def _create_array(group: zarr.Group, path: str, dtype: str | None, axes: tuple[str, ...]) -> None:
    parent = group
    parts = path.split("/")
    for name in parts[:-1]:
        parent = parent.require_group(name)
    resolved = np.dtype("S64" if dtype is None else dtype)
    first_extent = 2 if axes[0] == "detector_signal" else (7 if axes[0] == "frame" else 3)
    shape = (first_extent,) if len(axes) == 1 else (first_extent, 7)
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    parent.create_array(parts[-1], data=values.astype(resolved))


def _set_columnar_attrs(
    group: zarr.Group,
    specs: dict[str, object],
    table_paths: tuple[str, ...],
) -> None:
    for table_path in table_paths:
        prefix = table_path + "/"
        fields = [
            (path[len(prefix) :], spec)
            for path, spec in specs.items()
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        ]
        if not fields:
            continue
        table = group[table_path]
        table.attrs["storage_layout"] = "columnar"
        table.attrs["field_names"] = [name for name, _spec in fields]
        table.attrs["field_dtypes"] = {
            name: spec.logical_dtype for name, spec in fields
        }


def _archive(path: Path, *, family: str) -> tuple[zarr.Group, zarr.Group]:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    analysis = root.create_group("analysis")
    if family == "swim_bouts":
        schema = swim_bout_schema
        parent = analysis.create_group("swim_bout_runs")
        required = schema._required_specs()
        table_paths = schema._COLUMNAR_TABLE_PATHS
        attrs = {
            "schema_id": schema.SWIM_BOUT_RUN_SCHEMA_ID,
            "schema_version": schema.SWIM_BOUT_RUN_SCHEMA_VERSION,
            "layout": schema.SWIM_BOUT_LAYOUT,
        }
        writer = schema.write_swim_bout_array_manifest
    else:
        schema = bout_kinematics_schema
        parent = analysis.create_group("bout_kinematics_runs")
        required = schema._required_specs()
        table_paths = schema._COLUMNAR_TABLE_PATHS
        attrs = {
            "schema_id": schema.BOUT_KINEMATICS_RUN_SCHEMA_ID,
            "schema_version": schema.BOUT_KINEMATICS_RUN_SCHEMA_VERSION,
            "layout": schema.BOUT_KINEMATICS_LAYOUT,
        }
        writer = schema.write_bout_kinematics_array_manifest
    parent.attrs.update(
        {
            "latest": "source",
            "latest_complete": "source",
            "palette_completion_epoch": 1,
        }
    )
    run = parent.create_group("source")
    run.attrs.update(
        {
            **attrs,
            "palette_run_name": "source",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "provenance": {"stage": family},
        }
    )
    for spec in required.values():
        _create_array(run, spec.path, spec.dtype, spec.axes)
    _set_columnar_attrs(run, required, table_paths)
    writer(run)
    return root, parent


def test_exact_tabular_candidate_dry_run_is_read_only(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _root, parent = _archive(archive, family="swim_bouts")

    result = materialize_exact_tabular_candidate(
        archive,
        family_id="swim_bouts",
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / "scratch",
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["mutates_archive"] is False
    assert [
        phase["name"] for phase in result["runtime_telemetry"]["phases"]
    ] == ["plan"]
    assert "candidate" not in parent
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("family", ["swim_bouts", "bout_kinematics"])
def test_exact_tabular_candidate_publishes_complete_ineligible_without_pointer_changes(
    tmp_path: Path,
    family: str,
) -> None:
    archive = tmp_path / f"{family}_analysis.zarr"
    _root, _parent = _archive(archive, family=family)

    result = materialize_exact_tabular_candidate(
        archive,
        family_id=family,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / f"scratch-{family}",
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["publication"]["physical_copy"]["verification"] == (
        "sha256_all_physical_files"
    )
    assert result["publication"]["physical_copy"]["content_sha256"]
    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent_name = (
        "swim_bout_runs" if family == "swim_bouts" else "bout_kinematics_runs"
    )
    parent = reopened[f"analysis/{parent_name}"]
    candidate = parent["candidate"]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"
    assert candidate.attrs["palette_run_completion_status"] == "complete"
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_source_run"] == "source"
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    assert candidate.attrs[MANIFEST_ATTRIBUTE]["payload"]["byte_planner_adopted"] is True
    assert candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]["payload"][
        "storage_profile"
    ]["profile_id"] == "published_http_v1"
    assert result["local_validation"]["array_count"] > 20
    assert result["archive_direct_consolidated_array_count"] == result[
        "local_validation"
    ]["array_count"]
    assert result["published_direct_consolidated_array_count"] == result[
        "local_validation"
    ]["array_count"]
    assert result["source_logical_manifest_sha256"] == result[
        "published_logical_manifest_sha256"
    ]
    assert result["output_storage"]["file_count"] > 0
    assert [
        phase["name"] for phase in result["runtime_telemetry"]["phases"]
    ] == _COMPLETE_PHASE_SEQUENCE
    assert all(
        phase["outcome"] == "ok"
        for phase in result["runtime_telemetry"]["phases"]
    )
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    consolidated_candidate = consolidated[f"analysis/{parent_name}/candidate"]
    assert consolidated_candidate.attrs["stage_selector_eligible"] is False
    assert (
        consolidated_candidate.attrs["palette_run_completion_status"]
        == "complete"
    )
    probe_array = (
        "indexes/candidates/candidate_id"
        if family == "swim_bouts"
        else "level_index/analysis_level_id"
    )
    assert tuple(consolidated_candidate[probe_array].shape) == tuple(
        candidate[probe_array].shape
    )


def test_exact_tabular_candidate_stages_and_verifies_source_on_scratch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive, family="swim_bouts")
    scratch = tmp_path / "scratch"

    result = materialize_exact_tabular_candidate(
        archive,
        family_id="swim_bouts",
        source_run="source",
        run_name="candidate",
        scratch_root=scratch,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        stage_source_to_scratch=True,
    )

    staged_source = zarr.open_group(
        str(scratch / "staged-source-run"),
        mode="r",
        use_consolidated=False,
    )
    assert staged_source.attrs["palette_run_name"] == "source"
    assert result["source_logical_manifest_sha256"] == result[
        "published_logical_manifest_sha256"
    ]
    assert result["runtime_telemetry"]["execution"][
        "stage_source_to_scratch"
    ] is True


def test_exact_tabular_candidate_rejects_corrupted_staged_source_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _root, parent = _archive(archive, family="swim_bouts")
    original_copy = exact_tabular_candidate._copy_source_run_to_scratch

    def corrupt_copy(source: Path, target: Path, *, backend: str) -> None:
        original_copy(source, target, backend=backend)
        staged = zarr.open_group(
            str(target), mode="r+", use_consolidated=False
        )
        values = staged["indexes/candidates/candidate_id"][:]
        values[0] += np.array(1, dtype=values.dtype)
        staged["indexes/candidates/candidate_id"][:] = values

    monkeypatch.setattr(
        exact_tabular_candidate,
        "_copy_source_run_to_scratch",
        corrupt_copy,
    )

    with pytest.raises(
        ValueError, match="differs from authoritative source"
    ) as caught:
        materialize_exact_tabular_candidate(
            archive,
            family_id="swim_bouts",
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch",
            copy_backend="python",
            apply=True,
            stage_source_to_scratch=True,
        )

    assert "candidate" not in parent
    failure_phases = caught.value.palette_runtime_telemetry["phases"]
    assert failure_phases[-1]["name"] == "source_staging"
    assert failure_phases[-1]["outcome"] == "error"


def test_exact_tabular_candidate_rejects_nonlocal_scratch(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive, family="swim_bouts")

    with pytest.raises(ValueError, match="node-local filesystem"):
        build_exact_tabular_candidate_plan(
            archive,
            family_id="swim_bouts",
            source_run="source",
            run_name="candidate",
            scratch_root="/groups/palette-candidate-scratch",
        )


def test_exact_tabular_candidate_rejects_symlinked_staging_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive, family="swim_bouts")
    source_path = archive / "analysis" / "swim_bout_runs" / "source"
    (source_path / "external-link").symlink_to(tmp_path / "outside")

    with pytest.raises(ValueError, match="symlink-free"):
        materialize_exact_tabular_candidate(
            archive,
            family_id="swim_bouts",
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch",
            copy_backend="python",
            apply=True,
            stage_source_to_scratch=True,
        )


def test_published_acceptance_failure_is_tombstoned_inside_atomic_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _root, _parent = _archive(archive, family="swim_bouts")
    original_check = exact_tabular_candidate._direct_consolidated_check

    def fail_published_check(
        local_zarr: Path,
        *,
        run_path: str,
        declarations: tuple[object, ...],
    ) -> int:
        if local_zarr.resolve() == archive.resolve():
            raise RuntimeError("injected published metadata failure")
        return original_check(
            local_zarr,
            run_path=run_path,
            declarations=declarations,
        )

    monkeypatch.setattr(
        exact_tabular_candidate,
        "_direct_consolidated_check",
        fail_published_check,
    )

    with pytest.raises(
        RuntimeError, match="injected published metadata failure"
    ) as caught:
        materialize_exact_tabular_candidate(
            archive,
            family_id="swim_bouts",
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch",
            copy_backend="python",
            apply=True,
        )

    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent = reopened["analysis/swim_bout_runs"]
    failed = parent["candidate"]
    assert parent.attrs["latest"] == "source"
    assert parent.attrs["latest_complete"] == "source"
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert "atomic_publication_tombstone" in failed.attrs
    outcomes = {
        phase["name"]: phase["outcome"]
        for phase in caught.value.palette_runtime_telemetry["phases"]
    }
    assert outcomes["atomic_publication"] == "error"
    assert outcomes["published_direct_consolidated_comparison"] == "error"


@pytest.mark.parametrize("selector_value", [False, None])
def test_exact_tabular_candidate_requires_explicit_eligible_source(
    tmp_path: Path,
    selector_value: bool | None,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _root, parent = _archive(archive, family="swim_bouts")
    if selector_value is None:
        del parent["source"].attrs["stage_selector_eligible"]
    else:
        parent["source"].attrs["stage_selector_eligible"] = selector_value

    with pytest.raises(ValueError, match="not selector eligible"):
        build_exact_tabular_candidate_plan(
            archive,
            family_id="swim_bouts",
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch",
        )


def test_exact_tabular_candidate_refuses_alias_source_names(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _archive(archive, family="swim_bouts")

    with pytest.raises(ValueError, match="explicit immutable run name"):
        build_exact_tabular_candidate_plan(
            archive,
            family_id="swim_bouts",
            source_run="latest",
            run_name="candidate",
            scratch_root=tmp_path / "scratch",
        )


def test_exact_tabular_candidate_requires_explicit_completion_marker(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    _root, parent = _archive(archive, family="swim_bouts")
    del parent["source"].attrs["palette_run_completion_status"]

    with pytest.raises(ValueError, match="not complete"):
        build_exact_tabular_candidate_plan(
            archive,
            family_id="swim_bouts",
            source_run="source",
            run_name="candidate",
            scratch_root=tmp_path / "scratch",
        )
