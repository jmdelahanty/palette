from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import (
    track_kinematics_candidate as candidate_mod,
)
from fisheye.analysis.track_kinematics_schema import (
    TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS,
    TRACK_KINEMATICS_FLAT_LINEAGE_PATHS,
    TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
    build_track_kinematics_flat_lineage_declarations,
)
from fisheye.analysis.track_kinematics_storage import (
    TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
    build_flat_candidate_declarations,
    build_flat_candidate_storage_receipt,
    load_track_lineage_records,
    persist_flat_candidate_contract,
    rematerialize_flat_candidate,
    source_flat_projection_hashes,
    validate_flat_candidate,
)
from fisheye.analysis_workflows.materializers.track_kinematics_candidate import (
    build_track_kinematics_flat_candidate_plan,
    materialize_track_kinematics_flat_candidate,
)
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


def _parent_and_leaf(group: zarr.Group, path: str) -> tuple[zarr.Group, str]:
    components = path.split("/")
    parent = group
    for component in components[:-1]:
        parent = parent.require_group(component)
    return parent, components[-1]


def _sample_payload(path: str, dtype: np.dtype, shape: tuple[int, ...], track_id: int):
    if path == "track_sample_key":
        frames = np.arange(shape[0], dtype=np.int64) + 100 * track_id
        return np.column_stack([np.full(shape[0], track_id, dtype=np.int64), frames])
    if path in {"frame_indices", "source_acquisition_frame_index"}:
        return np.arange(shape[0], dtype=np.int64) + 100 * track_id
    if path == "source_row_index":
        return np.arange(shape[0], dtype=np.int64) + 1000 * track_id
    if path == "source_frame_interpolation":
        values = np.zeros(shape, dtype=dtype)
        frames = np.arange(shape[0], dtype=np.int64) + 100 * track_id
        values["left_source_frame_index"] = frames
        values["right_source_frame_index"] = frames + 1
        values["right_weight"] = np.linspace(0.0, 0.75, shape[0])
        return values
    if path == "source_instance_key":
        values = np.zeros(shape, dtype=dtype)
        values["valid"] = np.arange(shape[0]) % 3 != 0
        values["instance_key"] = np.where(
            values["valid"],
            np.arange(shape[0], dtype=np.uint64) + 10_000 * track_id,
            0,
        )
        return values
    if dtype == np.dtype(bool):
        return np.asarray(np.arange(np.prod(shape)).reshape(shape) % 2, dtype=bool)
    if dtype.kind in "iu":
        return np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return values + dtype.type(track_id / 10.0)


def _populate_v1_run(run: zarr.Group, *, track_rows: tuple[int, ...] = (11, 7)) -> None:
    track_ids = tuple(index * 2 for index in range(len(track_rows)))
    run.attrs.update(
        {
            "schema_id": "analysis.track_kinematics_runs",
            "schema_version": 1,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "stage_selector_eligible": True,
            "provenance": {
                "stage": "track_kinematics",
                "command": "unit-test-v1-track-authority",
                "parameters": {},
                "inputs": {},
            },
        }
    )
    run.create_array(
        "track_ids",
        data=np.asarray(track_ids, dtype=np.int32),
        chunks=(len(track_ids),),
    )
    run.create_array(
        "track_arena_ids",
        data=np.asarray([5 + track_id for track_id in track_ids], dtype=np.int32),
        chunks=(len(track_ids),),
    )
    tracks = run.create_group("tracks")
    for track_id, row_count in zip(track_ids, track_rows):
        track = tracks.create_group(f"id_{track_id}")
        for declaration in TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS:
            shape = tuple(
                row_count
                if dimension == "n_track_samples"
                else 3
                if dimension == "n_track_seconds"
                else int(dimension)
                for dimension in declaration.shape_template
            )
            data = _sample_payload(
                declaration.relative_path,
                declaration.dtype,
                shape,
                track_id,
            )
            parent, leaf = _parent_and_leaf(track, declaration.relative_path)
            parent.create_array(
                leaf,
                data=data,
                chunks=tuple(max(1, extent) for extent in shape),
            )


def _build_source_archive(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3, use_consolidated=False)
    parent = root.require_group("analysis").require_group("track_kinematics_runs")
    offline = parent.require_group("offline")
    source = offline.create_group("source_v1")
    _populate_v1_run(source)
    parent.attrs.update(
        {
            "latest": "offline/source_v1",
            "latest_complete": "offline/source_v1",
            "latest_offline": "source_v1",
        }
    )
    offline.attrs.update({"latest": "source_v1", "latest_complete": "source_v1"})
    consolidate_metadata_capture_expected_warnings(path)


def test_flat_schema_is_versioned_primitive_and_preserves_float64_positions() -> None:
    declarations = build_track_kinematics_flat_lineage_declarations(
        track_ids=(2, 7),
        include_physical=True,
        include_arena_inventory=True,
    )
    paths = {declaration.path for declaration in declarations}

    assert len(declarations) == 2 + 2 * (72 + 35)
    assert all(declaration.byte_planner_adopted for declaration in declarations)
    assert {declaration.physical_policy_owner for declaration in declarations} == {
        "analysis_storage_planning_v1"
    }
    assert all(
        declaration.contract.schema_version
        == TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION
        for declaration in declarations
    )
    assert all(
        declaration.contract.dtype.numpy_dtype is not None
        for declaration in declarations
    )
    assert all(
        np.dtype(declaration.contract.dtype.numpy_dtype).fields is None
        for declaration in declarations
    )
    for relative in TRACK_KINEMATICS_FLAT_LINEAGE_PATHS:
        assert f"tracks/id_2/{relative}" in paths
        assert f"tracks/id_7/{relative}" in paths
    position = next(
        declaration
        for declaration in declarations
        if declaration.path == "tracks/id_2/positions_px"
    )
    assert position.contract.dtype.numpy_dtype == "float64"
    assert position.contract.shape_template == ("n_track_samples_id_2", 2)


def test_flat_storage_rematerializes_exact_structured_fields_through_shared_factory(
    tmp_path: Path,
) -> None:
    source = zarr.open_group(
        str(tmp_path / "source-run"), mode="w", zarr_format=3, use_consolidated=False
    )
    _populate_v1_run(source)
    declarations = build_flat_candidate_declarations(source)
    hashes = source_flat_projection_hashes(source, declarations)
    receipt = build_flat_candidate_storage_receipt(
        source,
        profile=get_storage_profile("published_http_v1"),
    )
    candidate = zarr.open_group(
        str(tmp_path / "candidate-run"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    rematerialize_flat_candidate(source, candidate, receipt=receipt)
    candidate.attrs.update(
        {
            "schema_id": TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
            "schema_version": 2,
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "stage_selector_eligible": False,
            "storage_candidate_profile_promoted": False,
        }
    )
    persist_flat_candidate_contract(
        candidate,
        receipt=receipt,
        declarations=declarations,
        source_run_path="analysis/track_kinematics_runs/offline/source_v1",
        source_projection_hashes=hashes,
    )

    validation = validate_flat_candidate(candidate, source_group=source)
    assert validation["valid"], validation
    assert validation["array_count"] == len(declarations)
    assert candidate["tracks/id_0/positions_px"].dtype == np.dtype("float64")
    assert candidate["tracks/id_0/source_frame_interpolation/right_weight"].dtype == (
        np.dtype("float64")
    )
    assert candidate["tracks/id_0/source_instance_key/value"].dtype == np.dtype(
        "uint64"
    )
    assert all(
        np.dtype(candidate[path].dtype).fields is None
        for path in validation["logical_hashes"]
    )
    receipt_payload = candidate.attrs["analysis_storage_plan_receipt"]["payload"]
    assert receipt_payload["storage_profile"]["profile_id"] == "published_http_v1"
    assert receipt_payload["storage_profile"]["codec_profile_id"] == "zstd_fast_v1"
    assert receipt_payload["object_estimate"]["payload_objects"] > 0
    source_interpolation, source_instances = load_track_lineage_records(
        source["tracks/id_0"],
        lineage_schema_version=1,
    )
    flat_interpolation, flat_instances = load_track_lineage_records(
        candidate["tracks/id_0"],
        lineage_schema_version=2,
    )
    assert np.array_equal(flat_interpolation, source_interpolation)
    assert np.array_equal(flat_instances, source_instances)

    candidate["tracks/id_0/source_instance_key/value"][1] = np.uint64(999)
    tampered = validate_flat_candidate(candidate, source_group=source)
    assert not tampered["valid"]
    assert "decoded values differ" in " ".join(tampered["errors"])


def test_plan_rejects_archive_scratch_containment_and_symlink_alias(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    _build_source_archive(archive)

    with pytest.raises(ValueError, match="must not contain one another"):
        build_track_kinematics_flat_candidate_plan(
            archive,
            source_run="source_v1",
            run_name="candidate_v2",
            scratch_root=archive / "scratch",
        )
    alias = tmp_path / "archive-alias"
    alias.symlink_to(archive, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain one another"):
        build_track_kinematics_flat_candidate_plan(
            alias,
            source_run="source_v1",
            run_name="candidate_v2",
            scratch_root=archive / "scratch",
        )


def test_atomic_candidate_publication_is_complete_ineligible_and_nonpromoting(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    scratch = tmp_path / "scratch"
    _build_source_archive(archive)

    result = materialize_track_kinematics_flat_candidate(
        archive,
        source_run="source_v1",
        run_name="candidate_v2",
        scratch_root=scratch,
        copy_backend="python",
        apply=True,
    )

    assert result["status"] == "complete"
    assert result["local_direct_consolidated_array_count"] == 146
    assert result["archive_direct_consolidated_array_count"] == 146
    assert not scratch.exists()
    direct = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    parent = direct["analysis/track_kinematics_runs"]
    offline = parent["offline"]
    assert parent.attrs["latest"] == "offline/source_v1"
    assert parent.attrs["latest_complete"] == "offline/source_v1"
    assert parent.attrs["latest_offline"] == "source_v1"
    assert offline.attrs["latest"] == "source_v1"
    candidate = offline["candidate_v2"]
    assert candidate.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    assert "source_frame_interpolation" in candidate["tracks/id_0"].group_keys()
    assert "source_frame_interpolation" not in candidate["tracks/id_0"].array_keys()
    assert (
        consolidated["analysis/track_kinematics_runs/offline/candidate_v2"].attrs[
            "stage_selector_eligible"
        ]
        is False
    )


def test_post_consolidation_failure_repairs_visible_failed_tombstone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    scratch = tmp_path / "scratch"
    _build_source_archive(archive)
    original_check = candidate_mod._direct_consolidated_check
    failed = False
    archive_consolidations = 0
    original_consolidate = candidate_mod.consolidate_metadata_capture_expected_warnings

    def count_consolidation(path: Path) -> None:
        nonlocal archive_consolidations
        if Path(path).resolve() == archive.resolve():
            archive_consolidations += 1
        original_consolidate(path)

    def fail_after_real_archive_check(*args, **kwargs):
        nonlocal failed
        result = original_check(*args, **kwargs)
        if Path(args[0]).resolve() == archive.resolve() and not failed:
            failed = True
            raise RuntimeError("injected post-consolidation candidate failure")
        return result

    monkeypatch.setattr(
        candidate_mod,
        "consolidate_metadata_capture_expected_warnings",
        count_consolidation,
    )
    monkeypatch.setattr(
        candidate_mod, "_direct_consolidated_check", fail_after_real_archive_check
    )

    with pytest.raises(
        RuntimeError,
        match="injected post-consolidation candidate failure",
    ):
        materialize_track_kinematics_flat_candidate(
            archive,
            source_run="source_v1",
            run_name="candidate_v2",
            scratch_root=scratch,
            copy_backend="python",
            apply=True,
        )

    assert archive_consolidations >= 2
    direct = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    path = "analysis/track_kinematics_runs/offline/candidate_v2"
    assert direct[path].attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert consolidated[path].attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert direct[path].attrs["stage_selector_eligible"] is False
    assert consolidated[path].attrs["stage_selector_eligible"] is False
    assert direct["analysis/track_kinematics_runs"].attrs["latest"] == (
        "offline/source_v1"
    )
