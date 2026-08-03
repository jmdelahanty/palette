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
from fisheye.analysis_workflows.materializers.exact_tabular_candidate import (
    build_exact_tabular_candidate_plan,
    materialize_exact_tabular_candidate,
)


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
