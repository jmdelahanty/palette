from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
import zarr

from fisheye.analysis.eye_angle_schema import EyeAngleDimensions
from fisheye.analysis.eye_angle_storage import (
    EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    EYE_ANGLE_FALSE_FILL_PATHS,
    EYE_ANGLE_NAN_FILL_PATHS,
    EYE_ANGLE_STORAGE_PLAN_ATTR,
    EYE_ANGLE_ZERO_FILL_PATHS,
    build_eye_angle_candidate_storage_plan,
    create_eye_angle_array_from_entry,
    eye_angle_planned_fill_value,
    eye_angle_storage_entries_by_path,
    validate_eye_angle_candidate_storage,
    validate_eye_angle_direct_consolidated_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _create_planned_run(root: zarr.Group, dimensions: EyeAngleDimensions) -> zarr.Group:
    run = root.require_group("analysis/eye_angle_runs/candidate")
    receipt = build_eye_angle_candidate_storage_plan(dimensions)
    for path, entry in eye_angle_storage_entries_by_path(receipt).items():
        components = path.split("/")
        group = run
        for component in components[:-1]:
            group = group.require_group(component)
        data = np.zeros(entry.plan.logical_shape, dtype=entry.facts.dtype)
        create_eye_angle_array_from_entry(
            group,
            name=components[-1],
            entry=entry,
            data=data,
        )
    run.attrs[EYE_ANGLE_STORAGE_PLAN_ATTR] = receipt.as_manifest()
    return run


def test_candidate_plan_is_byte_derived_for_short_and_million_row_runs() -> None:
    short = build_eye_angle_candidate_storage_plan(EyeAngleDimensions(23, 31))
    full = build_eye_angle_candidate_storage_plan(
        EyeAngleDimensions(1_000_000, 1_188_000)
    )

    assert len(short.entries) == len(full.entries) == 41
    assert short.profile.profile_id == EYE_ANGLE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
    assert all(entry.declaration.byte_planner_adopted for entry in full.entries)
    assert all(entry.plan.estimated_payload_objects <= 1 for entry in short.entries)

    entries = eye_angle_storage_entries_by_path(full)
    assert entries["roi_angles"].plan.chunk_shape == (2_048, 141)
    assert entries["frame_angles"].plan.chunk_shape == (2_048, 141)
    assert entries["support/instance_key"].plan.chunk_shape == (131_072,)
    assert entries["support/body_frame/valid"].plan.chunk_shape == (1_000_000,)
    assert entries["angle_channel_index/name"].plan.chunk_shape == (141, 256)
    assert entries["angle_channel_index/name"].plan.shard_shape is None
    assert entries["roi_angles"].plan.shard_shape == (61_440, 141)
    assert entries["roi_angles"].plan.write_ownership == "whole_shard_single_writer"


def test_exact_semantic_fill_map_partitions_all_41_arrays() -> None:
    receipt = build_eye_angle_candidate_storage_plan(EyeAngleDimensions(5, 7))
    entries = eye_angle_storage_entries_by_path(receipt)
    assert len(entries) == 41
    assert (
        EYE_ANGLE_NAN_FILL_PATHS
        | EYE_ANGLE_FALSE_FILL_PATHS
        | EYE_ANGLE_ZERO_FILL_PATHS
    ) == set(entries)
    assert not (EYE_ANGLE_NAN_FILL_PATHS & EYE_ANGLE_FALSE_FILL_PATHS)
    assert not (EYE_ANGLE_NAN_FILL_PATHS & EYE_ANGLE_ZERO_FILL_PATHS)
    assert not (EYE_ANGLE_FALSE_FILL_PATHS & EYE_ANGLE_ZERO_FILL_PATHS)
    for path, entry in entries.items():
        fill = eye_angle_planned_fill_value(entry)
        if path in EYE_ANGLE_NAN_FILL_PATHS:
            assert type(fill) is float and np.isnan(fill)
        elif path in EYE_ANGLE_FALSE_FILL_PATHS:
            assert fill is False
        else:
            assert fill == 0 and fill is not False


def test_created_metadata_uses_exact_nan_false_and_zero_fills(tmp_path) -> None:
    dimensions = EyeAngleDimensions(5, 7)
    archive = tmp_path / "fills.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    run = _create_planned_run(root, dimensions)

    assert run["roi_angles"].metadata.to_dict()["fill_value"] == "NaN"
    assert (
        run["support/body_frame/heading_deg"].metadata.to_dict()["fill_value"] == "NaN"
    )
    assert run["support/body_frame/valid"].metadata.to_dict()["fill_value"] is False
    assert run["roi_qa"].metadata.to_dict()["fill_value"] == 0
    assert run["support/instance_key"].metadata.to_dict()["fill_value"] == 0

    direct_nan = json.loads(
        (
            archive
            / "analysis"
            / "eye_angle_runs"
            / "candidate"
            / "roi_angles"
            / "zarr.json"
        ).read_text(encoding="utf-8")
    )
    assert direct_nan["fill_value"] == "NaN"


def test_candidate_receipt_rejects_rehashed_plan_tampering(tmp_path) -> None:
    dimensions = EyeAngleDimensions(5, 7)
    root = zarr.open_group(str(tmp_path / "tampered.zarr"), mode="w", zarr_format=3)
    run = _create_planned_run(root, dimensions)
    assert not validate_eye_angle_candidate_storage(run, dimensions=dimensions)

    tampered = deepcopy(run.attrs[EYE_ANGLE_STORAGE_PLAN_ATTR])
    tampered["payload"]["arrays"][0]["plan"]["chunk_nbytes"] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    run.attrs[EYE_ANGLE_STORAGE_PLAN_ATTR] = tampered

    issues = validate_eye_angle_candidate_storage(run, dimensions=dimensions)
    assert {issue.code for issue in issues} == {"storage_plan_receipt_mismatch"}


def test_candidate_rejects_array_not_created_from_resolved_plan(tmp_path) -> None:
    dimensions = EyeAngleDimensions(5, 7)
    root = zarr.open_group(str(tmp_path / "metadata.zarr"), mode="w", zarr_format=3)
    run = _create_planned_run(root, dimensions)
    support = run["support"]
    del support["instance_key"]
    support.create_array(
        "instance_key",
        data=np.arange(5, dtype=np.uint64),
        chunks=(2,),
    )

    issues = validate_eye_angle_candidate_storage(run, dimensions=dimensions)
    assert any(
        issue.code == "storage_metadata_mismatch"
        and issue.path == "support/instance_key"
        for issue in issues
    )


def test_candidate_rejects_zero_fill_for_nan_semantic_array(tmp_path) -> None:
    dimensions = EyeAngleDimensions(5, 7)
    root = zarr.open_group(str(tmp_path / "wrong-fill.zarr"), mode="w", zarr_format=3)
    run = _create_planned_run(root, dimensions)
    del run["roi_angles"]
    run.create_array(
        "roi_angles",
        data=np.zeros((5, 141), dtype=np.float32),
        chunks=(5, 141),
        fill_value=0,
    )

    issues = validate_eye_angle_candidate_storage(run, dimensions=dimensions)
    assert any(
        issue.code == "storage_fill_value_mismatch" and issue.path == "roi_angles"
        for issue in issues
    )


def test_candidate_direct_and_consolidated_metadata_are_equivalent(tmp_path) -> None:
    dimensions = EyeAngleDimensions(5, 7)
    archive = tmp_path / "equivalent.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    _create_planned_run(root, dimensions)
    zarr.consolidate_metadata(str(archive))

    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    assert not validate_eye_angle_direct_consolidated_storage(
        direct["analysis/eye_angle_runs/candidate"],
        consolidated["analysis/eye_angle_runs/candidate"],
        dimensions=dimensions,
    )


def test_candidate_plan_rejects_non_exact_dimensions() -> None:
    with pytest.raises(ValueError, match="n_roi_rows"):
        EyeAngleDimensions(True, 2)  # type: ignore[arg-type]
