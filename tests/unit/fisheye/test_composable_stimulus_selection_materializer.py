from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.composable_stimulus_selection import (
    RoleMetadata,
    SelectionSpec,
    TimelineAuthority,
    canonical_json,
    compile_selection,
    member,
    stimulus_step_reference,
    union,
)
from fisheye.analysis_workflows.materializers.composable_stimulus_selection import (
    REQUESTED_JSON_ARRAY,
    REQUESTED_JSON_ATTR,
    RESOLVED_JSON_ATTR,
    TIMELINE_AUTHORITY_JSON_ARRAY,
    TIMELINE_AUTHORITY_JSON_ATTR,
    PARENT_PATH,
    build_composable_stimulus_selection_materialization_plan,
    materialize_composable_stimulus_selection,
    materialize_composable_stimulus_selection_plan,
    reconstruct_compiled_selection,
    validate_composable_stimulus_selection_run,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)


def _authority() -> TimelineAuthority:
    return TimelineAuthority(
        recording_id="recording-1",
        timeline_id="timeline-1",
        stimulus_authority_id="stimulus-v6",
        stimulus_authority_sha256="a" * 64,
        acquisition_frame_domain="camera_native_acquisition_frames",
        acquisition_frame_count=20,
        source_video_metadata_ref="raw_video/metadata",
        source_video_metadata_sha256="b" * 64,
        acquisition_clock_authority_ref="recording/acquisition_clock",
        acquisition_clock_authority_sha256="c" * 64,
        source_metadata_sha256="d" * 64,
    )


def _compiled():
    authority = _authority()
    pre = member(
        stimulus_step_reference(
            reference_id="step-pre",
            label="pre",
            start_frame=1,
            end_frame=7,
            authority=authority,
            occurrence_id="pre-1",
        ),
        role=RoleMetadata(role="baseline", label="pre"),
    )
    chaser = member(
        stimulus_step_reference(
            reference_id="step-chaser",
            label="chaser",
            start_frame=5,
            end_frame=11,
            authority=authority,
            occurrence_id="chaser-1",
        ),
        role=RoleMetadata(role="treatment", label="chaser"),
    )
    return compile_selection(
        SelectionSpec(
            selection_id="pre-and-chaser",
            expression=union(pre, chaser),
            aggregation_policy="keep_occurrences",
            metadata={"fixture": "materializer"},
        )
    )


def _archive(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    analysis = root.create_group("analysis")
    parent = analysis.create_group("stimulus_selection_runs")
    parent.attrs["latest"] = "previous"
    parent.attrs["latest_complete"] = "previous"
    parent.attrs["authoritative_run"] = "previous"
    root.create_group("source_sentinel").attrs["value"] = "unchanged"
    return root


def test_materializer_round_trips_overlap_and_preserves_selectors(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = _archive(archive)
    compiled = _compiled()
    before_attrs = dict(root[PARENT_PATH].attrs)

    result = materialize_composable_stimulus_selection(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch",
        run_name="selection_canary_v1",
        copy_backend="python",
    )

    assert result["status"] == "complete"
    assert result["mutates_archive"] is True
    assert not (tmp_path / "scratch").exists()
    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    parent = direct[PARENT_PATH]
    run = parent["selection_canary_v1"]
    assert {key: parent.attrs[key] for key in before_attrs} == before_attrs
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert parent.attrs["palette_completion_epoch"] == 2
    assert isinstance(run.attrs["run_provenance"], dict)
    assert run.attrs["retry_policy"] == "new_immutable_run_name_required"
    assert (
        validate_direct_consolidated_subtree(
            archive,
            subtree_path=f"{PARENT_PATH}/selection_canary_v1",
        ).array_count
        == 32
    )
    assert (
        dict(consolidated[f"{PARENT_PATH}/selection_canary_v1"].attrs)
        == dict(run.attrs)
    )

    reconstructed = reconstruct_compiled_selection(
        archive / PARENT_PATH / "selection_canary_v1"
    )
    assert reconstructed.to_dict() == compiled.to_dict()
    assert {
        item.reference_id
        for item in reconstructed.resolved_intervals[1].source_memberships
    } == {"step-pre", "step-chaser"}
    assert run["resolved_membership_offsets"][:].tolist() == [0, 1, 3, 4]
    assert np.asarray(run[REQUESTED_JSON_ARRAY][:], dtype=np.uint8).tobytes().decode(
        "utf-8"
    ) == canonical_json(compiled.requested)
    assert np.asarray(
        run[TIMELINE_AUTHORITY_JSON_ARRAY][:], dtype=np.uint8
    ).tobytes().decode("utf-8") == canonical_json(compiled.authority.to_dict())
    assert not {
        REQUESTED_JSON_ATTR,
        RESOLVED_JSON_ATTR,
        TIMELINE_AUTHORITY_JSON_ATTR,
    }.intersection(run.attrs)
    assert run.attrs["selection_summary"]["selected_frame_count"] == 10
    assert all(
        run[name].dtype.kind not in {"O", "U", "S"}
        for name in run.array_keys()
    )
    assert validate_composable_stimulus_selection_run(
        archive / PARENT_PATH / "selection_canary_v1",
        expected_compiled_selection=compiled,
    )["valid"]
    assert direct["source_sentinel"].attrs["value"] == "unchanged"


def test_planning_is_read_only_and_rejects_selector_aliases(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = _archive(archive)
    compiled = _compiled()

    planned = materialize_composable_stimulus_selection(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch",
        run_name="planned_v1",
        apply=False,
    )
    assert planned["status"] == "planned"
    assert "planned_v1" not in root[PARENT_PATH]
    assert not (tmp_path / "scratch").exists()
    with pytest.raises(ValueError, match="selector aliases"):
        build_composable_stimulus_selection_materialization_plan(
            archive,
            compiled_selection=compiled,
            scratch_root=tmp_path / "scratch",
            run_name="latest",
        )


def test_duplicate_run_and_post_plan_mutation_fail_closed(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    _archive(archive)
    compiled = _compiled()
    materialize_composable_stimulus_selection(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch-a",
        run_name="duplicate_v1",
    )
    with pytest.raises(FileExistsError):
        materialize_composable_stimulus_selection(
            archive,
            compiled_selection=compiled,
            scratch_root=tmp_path / "scratch-b",
            run_name="duplicate_v1",
        )

    plan = build_composable_stimulus_selection_materialization_plan(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch-c",
        run_name="mutated_v1",
    )
    mutated = _compiled()
    object.__setattr__(mutated, "request_digest", "e" * 64)
    with pytest.raises(ValueError, match="stale"):
        materialize_composable_stimulus_selection_plan(
            plan,
            mutated,
        )
    assert "mutated_v1" not in zarr.open_group(
        str(archive), mode="r", use_consolidated=False
    )[PARENT_PATH]


def test_array_tamper_is_rejected_by_digest_and_shape_contract(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    _archive(archive)
    compiled = _compiled()
    materialize_composable_stimulus_selection(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch",
        run_name="tamper_v1",
    )
    run_path = archive / PARENT_PATH / "tamper_v1"
    run = zarr.open_group(str(run_path), mode="a", use_consolidated=False)
    original = int(run["resolved_interval_bounds"][0, 0])
    run["resolved_interval_bounds"][0, 0] = original + 1
    result = validate_composable_stimulus_selection_run(run_path)
    assert result["valid"] is False
    assert any("manifest" in error or "reconstruct" in error for error in result["errors"])


def test_existing_local_selection_scratch_cannot_be_overwritten(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    _archive(archive)
    compiled = _compiled()
    plan = build_composable_stimulus_selection_materialization_plan(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch",
        run_name="selection_scratch_v1",
    )
    plan.local_zarr.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="existing local Zarr"):
        materialize_composable_stimulus_selection_plan(plan, compiled)


def test_selector_like_child_attribute_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    _archive(archive)
    compiled = _compiled()
    materialize_composable_stimulus_selection(
        archive,
        compiled_selection=compiled,
        scratch_root=tmp_path / "scratch",
        run_name="selection_selector_attr_v1",
    )
    run_path = archive / PARENT_PATH / "selection_selector_attr_v1"
    run = zarr.open_group(str(run_path), mode="a", use_consolidated=False)
    run.attrs["latest_materialized"] = "selection_selector_attr_v1"
    result = validate_composable_stimulus_selection_run(run_path)
    assert result["valid"] is False
    assert any("selector alias" in error for error in result["errors"])


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    _archive(archive)
    materialize_composable_stimulus_selection(
        archive,
        compiled_selection=_compiled(),
        scratch_root=tmp_path / "scratch",
        run_name="selection_unknown_schema_v1",
    )
    run_path = archive / PARENT_PATH / "selection_unknown_schema_v1"
    run = zarr.open_group(str(run_path), mode="r+", use_consolidated=False)
    run.attrs["schema_version"] = 999

    validation = validate_composable_stimulus_selection_run(run_path)
    assert validation["valid"] is False
    assert any(
        "unsupported selection-run schema" in error
        for error in validation["errors"]
    )
    with pytest.raises(ValueError, match="unsupported schema version"):
        reconstruct_compiled_selection(run_path)
