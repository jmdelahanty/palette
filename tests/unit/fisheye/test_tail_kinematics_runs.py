from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_kinematics_runs as mod
from fisheye.analysis import subject_shape_io
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.detect_reason_codec import decode_reason_bytes
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "tail-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )
    monkeypatch.setattr(
        subject_shape_io,
        "load_persisted_subject_shape_coordinate_publication",
        lambda root, run_path: _fake_coordinate_publication(root[run_path], run_path),
    )
    monkeypatch.setattr(
        mod,
        "publish_tail_kinematics_coordinate_surfaces",
        lambda _root, run_group: run_group.attrs.__setitem__(
            "tail_coordinate_publication_manifest_sha256", "9" * 64
        ),
    )

    def _activate(_root, parent, run_group, *, run_name, **_kwargs):
        assert run_group.attrs["palette_run_completion_status"] == "complete"
        assert run_group.attrs["stage_selector_eligible"] is False
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        run_group.attrs["stage_selector_eligible"] = True

    monkeypatch.setattr(mod, "activate_tail_coordinate_publication", _activate)


def _fake_coordinate_publication(shape: zarr.Group, run_path: str):
    array_records: dict[str, dict[str, object]] = {}
    for relative_ref in (
        *mod._REQUIRED_SOURCE_ARRAY_PATHS,
        *mod._OPTIONAL_SOURCE_ARRAY_PATHS,
    ):
        node = shape.get(relative_ref)
        if node is None:
            continue
        array_records[relative_ref] = {
            "array_ref": f"/{run_path}/{relative_ref}",
            "relative_ref": relative_ref,
            "dtype": np.dtype(node.dtype).str,
            "shape": [int(value) for value in node.shape],
            "content_sha256": array_payload_sha256(node),
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
    curvature_semantics = SimpleNamespace(
        record_ref=f"/{run_path}/components/subject_body/tail_curvature_px_inv@subject_shape_scalar_surface",
        record_sha256="4" * 64,
    )

    def require_scalar_surface(relative_ref, *, units=None, surface_kind=None):
        assert relative_ref == "components/subject_body/tail_curvature_px_inv"
        assert units == "px^-1"
        assert surface_kind == "row_profile"
        return SimpleNamespace(semantics=curvature_semantics)

    return SimpleNamespace(
        manifest=SimpleNamespace(
            record={"arrays": array_records},
            record_ref=f"/{run_path}@subject_shape_publication_manifest",
            record_sha256="1" * 64,
        ),
        row_identity=SimpleNamespace(
            record_ref=f"/{run_path}@row_identity",
            record_sha256="2" * 64,
        ),
        tail_sample_axis=SimpleNamespace(
            record_ref=f"/{run_path}/coordinate_records/tail_sample_axis@subject_shape_tail_sample_axis",
            record_sha256="3" * 64,
        ),
        body_frame=SimpleNamespace(
            record_ref=f"/{run_path}/body_frame@fish_anatomical_body_frame",
            record_sha256="5" * 64,
        ),
        require_scalar_surface=require_scalar_surface,
    )


def _source_arrays(
    tangent_rows: np.ndarray | None = None,
    *,
    row_count: int | None = None,
) -> dict[str, np.ndarray]:
    source_s = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    resolved_row_count = int(
        row_count
        if row_count is not None
        else (2 if tangent_rows is None else tangent_rows.shape[0])
    )
    tail_xy = np.zeros((resolved_row_count, 4, 2), dtype=np.float32)
    tail_xy[:, :, 0] = -source_s[None, :] * 10.0
    if tangent_rows is None:
        tangent_rows = np.repeat(
            np.asarray([[[-1.0, 0.0]]], dtype=np.float32),
            resolved_row_count * 4,
            axis=0,
        ).reshape(resolved_row_count, 4, 2)
    return {
        "source_tail_sample_s": source_s,
        "tail_sample_xy": tail_xy,
        "tail_tangent_xy": tangent_rows.astype(np.float32),
        "tail_curvature_px_inv": np.zeros((resolved_row_count, 4), dtype=np.float32),
        "tail_sample_valid": np.ones((resolved_row_count,), dtype=bool),
        "bspline_valid": np.ones((resolved_row_count,), dtype=bool),
        "tail_base_xy": np.zeros((resolved_row_count, 2), dtype=np.float32),
        "body_forward_axis_xy": np.repeat(
            np.asarray([[1.0, 0.0]], dtype=np.float32), resolved_row_count, axis=0
        ),
        "body_left_axis_xy": np.repeat(
            np.asarray([[0.0, 1.0]], dtype=np.float32), resolved_row_count, axis=0
        ),
        "body_frame_valid": np.ones((resolved_row_count,), dtype=bool),
    }


def test_tail_kinematics_straight_tail_is_zero_angle() -> None:
    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **_source_arrays(), tail_angle_sample_count=10
    )

    assert batch.valid.tolist() == [True, True]
    np.testing.assert_allclose(
        batch.tail_angle_rad, np.zeros((2, 10), dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        batch.tail_tip_angle_deg, np.zeros((2,), dtype=np.float32), atol=1e-5
    )
    np.testing.assert_allclose(
        batch.tail_lateral_deflection_px, np.zeros((2, 10), dtype=np.float32), atol=1e-5
    )


def test_tail_kinematics_left_positive_right_negative() -> None:
    theta = math.radians(30.0)
    left_tangent = np.asarray([-math.cos(theta), math.sin(theta)], dtype=np.float32)
    right_tangent = np.asarray([-math.cos(theta), -math.sin(theta)], dtype=np.float32)
    tangent_rows = np.stack(
        [
            np.repeat(left_tangent[None, :], 4, axis=0),
            np.repeat(right_tangent[None, :], 4, axis=0),
        ],
        axis=0,
    )

    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **_source_arrays(tangent_rows),
        tail_angle_sample_count=10,
    )

    np.testing.assert_allclose(
        batch.tail_angle_deg[0], np.full((10,), 30.0, dtype=np.float32), atol=1e-4
    )
    np.testing.assert_allclose(
        batch.tail_angle_deg[1], np.full((10,), -30.0, dtype=np.float32), atol=1e-4
    )
    np.testing.assert_allclose(
        batch.max_abs_tail_angle_deg, np.full((2,), 30.0, dtype=np.float32), atol=1e-4
    )


def test_tail_kinematics_invalid_rows_preserve_failure_reason() -> None:
    sources = _source_arrays()
    sources["tail_sample_valid"] = np.asarray([True, False], dtype=bool)

    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **sources,
        tail_sample_failure_reason=np.asarray(
            ["ok", "tail_segment_too_short"], dtype=object
        ),
        tail_angle_sample_count=10,
    )

    assert batch.valid.tolist() == [True, False]
    assert str(batch.failure_reason[1]) == "tail_segment_too_short"
    assert np.all(np.isnan(batch.tail_angle_rad[1]))
    decoded = decode_reason_bytes(batch.failure_reason_bytes)
    assert decoded.tolist() == ["ok", "tail_segment_too_short"]


def test_vectorized_interpolation_matches_scalar_numpy_reference() -> None:
    source_s = np.asarray([0.0, 0.2, 0.65, 1.0], dtype=np.float64)
    target_s = np.asarray(
        [-0.1, 0.0, 0.1, 0.2, 0.5, 0.65, 0.9, 1.0, 1.1], dtype=np.float64
    )
    values_2d = np.asarray(
        [
            [[0.0, 4.0], [2.0, 3.0], [6.5, 2.0], [10.0, 1.0]],
            [[1.0, 0.0], [3.0, 1.0], [7.5, 2.0], [11.0, 3.0]],
            [[2.0, 3.0], [4.0, 2.0], [8.5, 1.0], [12.0, 0.0]],
        ],
        dtype=np.float64,
    )
    values_2d[2, 1, 0] = np.nan
    row_valid = np.asarray([True, False, True], dtype=bool)

    expected_2d = np.full((3, target_s.size, 2), np.nan, dtype=np.float32)
    for row_idx in range(3):
        if not row_valid[row_idx] or not np.all(np.isfinite(values_2d[row_idx])):
            continue
        for dim_idx in range(2):
            expected_2d[row_idx, :, dim_idx] = np.interp(
                target_s,
                source_s,
                values_2d[row_idx, :, dim_idx],
            ).astype(np.float32)

    actual_2d = mod._interp_rows_2d(source_s, values_2d, target_s, row_valid)
    np.testing.assert_allclose(
        actual_2d, expected_2d, rtol=0.0, atol=1e-7, equal_nan=True
    )

    values_1d = values_2d[:, :, 1]
    expected_1d = np.full((3, target_s.size), np.nan, dtype=np.float32)
    for row_idx in range(3):
        if row_valid[row_idx] and np.all(np.isfinite(values_1d[row_idx])):
            expected_1d[row_idx] = np.interp(
                target_s, source_s, values_1d[row_idx]
            ).astype(np.float32)
    actual_1d = mod._interp_rows_1d(source_s, values_1d, target_s, row_valid)
    np.testing.assert_allclose(
        actual_1d, expected_1d, rtol=0.0, atol=1e-7, equal_nan=True
    )


def test_vectorized_validity_preserves_failure_precedence() -> None:
    sources = _source_arrays(row_count=6)
    sources["body_frame_valid"][1] = False
    sources["bspline_valid"][[1, 2]] = False
    sources["tail_sample_valid"][[1, 2, 3]] = False
    sources["tail_sample_xy"][4, 1, 0] = np.nan
    sources["body_forward_axis_xy"][5] = 0.0

    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **sources,
        body_frame_failure_reason=np.asarray(
            ["ok", "body_missing", "ok", "ok", "ok", "ok"], dtype=object
        ),
        bspline_failure_reason=np.asarray(
            ["ok", "spline_should_lose", "ok", "ok", "ok", "ok"], dtype=object
        ),
        tail_sample_failure_reason=np.asarray(
            [
                "ok",
                "tail_should_lose",
                "tail_should_lose",
                "tail_too_short",
                "ok",
                "ok",
            ],
            dtype=object,
        ),
        tail_angle_sample_count=10,
    )

    assert batch.valid.tolist() == [True, False, False, False, False, False]
    assert batch.failure_reason.tolist() == [
        "ok",
        "body_missing",
        "bspline_invalid",
        "tail_too_short",
        "tail_geometry_nonfinite",
        "body_frame_invalid",
    ]


def _build_shape_root(*, row_count: int = 2) -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    shape = parent.create_group("shape_001")
    shape.attrs["source_refined_subject_masks_run"] = "refined_001"
    shape.attrs["body_frame_schema_id"] = "fish_anatomical_body_frame"

    source_revisions = shape.create_group("source_refined_subject_masks")
    source_revisions.attrs["source_run"] = "refined_001"
    source_revisions.attrs["component_names"] = ["subject_body"]
    source_revisions.create_array(
        "row_revision",
        data=np.arange(3, 3 + int(row_count), dtype=np.int64)[:, None],
        overwrite=True,
    )
    source_revisions.create_array(
        "row_revision_available", data=np.asarray([True], dtype=bool), overwrite=True
    )

    shape.create_array(
        "source_acquisition_frame_index",
        data=np.arange(10, 10 + int(row_count), dtype=np.int32),
        overwrite=True,
    )
    shape.create_array(
        "source_crop_row_ids",
        data=np.arange(100, 100 + int(row_count), dtype=np.int64),
        overwrite=True,
    )
    shape.create_array(
        "instance_key",
        data=np.arange(1000, 1000 + int(row_count), dtype=np.uint64),
        overwrite=True,
    )

    components = shape.create_group("components")
    body = components.create_group("subject_body")
    sources = _source_arrays(row_count=int(row_count))
    body.attrs["tail_sample_count"] = int(sources["source_tail_sample_s"].shape[0])
    body.create_array(
        "tail_sample_s", data=sources["source_tail_sample_s"], overwrite=True
    )
    body.create_array("tail_sample_xy", data=sources["tail_sample_xy"], overwrite=True)
    body.create_array(
        "tail_tangent_xy", data=sources["tail_tangent_xy"], overwrite=True
    )
    body.create_array(
        "tail_curvature_px_inv", data=sources["tail_curvature_px_inv"], overwrite=True
    )
    body.create_array(
        "tail_sample_valid", data=sources["tail_sample_valid"], overwrite=True
    )
    body.create_array("bspline_valid", data=sources["bspline_valid"], overwrite=True)
    body.create_array("tail_base_xy", data=sources["tail_base_xy"], overwrite=True)
    body.create_array(
        "tail_sample_failure_reason_bytes",
        data=mod._encode_reasons(["ok"] * int(row_count)),
        overwrite=True,
    )
    body.create_array(
        "bspline_failure_reason_bytes",
        data=mod._encode_reasons(["ok"] * int(row_count)),
        overwrite=True,
    )

    body_frame = shape.create_group("body_frame")
    body_frame.create_array(
        "forward_axis_xy", data=sources["body_forward_axis_xy"], overwrite=True
    )
    body_frame.create_array(
        "left_axis_xy", data=sources["body_left_axis_xy"], overwrite=True
    )
    body_frame.create_array(
        "axis_valid", data=sources["body_frame_valid"], overwrite=True
    )
    body_frame.create_array(
        "failure_reason_bytes",
        data=mod._encode_reasons(["ok"] * int(row_count)),
        overwrite=True,
    )
    return root


def _staged_source_receipts(
    shape: zarr.Group,
    *,
    row_count: int,
    chunk_rows: int,
) -> tuple[dict[str, object], dict[str, object]]:
    run_name = "shape_001"
    run_path = f"analysis/subject_shape_runs/{run_name}"
    publication = _fake_coordinate_publication(shape, run_path)
    authority = mod._build_staged_source_authority(
        shape,
        run_name=run_name,
        row_count=row_count,
        source_sample_count=4,
        publication=publication,
    )
    receipt = mod.build_tail_kinematics_staged_input_integrity_receipt(
        shape,
        run_name=run_name,
        authority=authority,
        chunk_rows=chunk_rows,
        read_workers=2,
    )
    return authority, receipt


def test_staged_input_receipt_is_checked_in_each_worker_owned_block() -> None:
    root = _build_shape_root(row_count=9)
    shape = root["analysis/subject_shape_runs/shape_001"]
    authority, receipt = _staged_source_receipts(
        shape,
        row_count=9,
        chunk_rows=4,
    )
    assert [(chunk["start_row"], chunk["stop_row"]) for chunk in receipt["chunks"]] == [
        (0, 4),
        (4, 8),
        (8, 9),
    ]

    _name, _group, sources = mod._resolve_tail_kinematics_sources(
        root,
        "shape_001",
        _staged_source_authority=authority,
        _staged_input_integrity_receipt=receipt,
    )
    values = np.asarray(shape["components/subject_body/tail_sample_xy"][0:4]).copy()
    values[0, 0, 0] += 1.0
    shape["components/subject_body/tail_sample_xy"][0:4] = values
    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="worker input.*differs",
    ):
        mod._read_tail_kinematics_source_block(sources, slice(0, 4))


def test_staged_input_receipt_rejects_gaps_and_requires_complete_attestation() -> None:
    root = _build_shape_root(row_count=9)
    shape = root["analysis/subject_shape_runs/shape_001"]
    authority, receipt = _staged_source_receipts(
        shape,
        row_count=9,
        chunk_rows=4,
    )
    expected = [str(chunk["record_sha256"]) for chunk in receipt["chunks"]]
    attestation = mod._complete_staged_input_worker_attestation(receipt, expected)
    assert attestation["complete_worker_chunk_set"] is True
    assert attestation["chunk_count"] == 3
    with pytest.raises(RuntimeError, match="exact complete"):
        mod._complete_staged_input_worker_attestation(receipt, expected[:-1])

    broken = deepcopy(receipt)
    broken["chunks"][1]["start_row"] = 5
    broken_chunk_body = {
        key: value
        for key, value in broken["chunks"][1].items()
        if key != "record_sha256"
    }
    broken["chunks"][1]["record_sha256"] = mod._canonical_sha256(broken_chunk_body)
    broken_body = {
        key: value for key, value in broken.items() if key != "record_sha256"
    }
    broken["record_sha256"] = mod._canonical_sha256(broken_body)
    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="gap, overlap",
    ):
        mod._canonical_staged_input_integrity_receipt(
            shape,
            run_name="shape_001",
            authority=authority,
            receipt=broken,
        )


def test_normal_tail_writer_rejects_unpublished_subject_shape_source() -> None:
    root = _build_shape_root()

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="not a valid canonical coordinate publication",
    ):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_rejected",
            dry_run=True,
        )


def test_write_tail_kinematics_run_group_writes_schema_and_row_lineage(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_001",
        tail_angle_sample_count=10,
    )

    assert summary["status"] == "updated"
    assert summary["valid_row_count"] == 2
    parent = root["analysis"]["tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_001"
    run = parent["tail_001"]
    assert run.attrs["schema_id"] == "analysis.tail_kinematics_runs"
    assert run.attrs["schema_version"] == 2
    assert run.attrs["source_subject_shape_run"] == "shape_001"
    assert run.attrs["source_subject_shape_authority_mode"] == "canonical_publication"
    assert run.attrs["source_subject_shape_publication_manifest_sha256"] == "1" * 64
    assert len(run.attrs["source_subject_shape_authority_sha256"]) == 64
    assert (
        run.attrs["source_subject_shape_authority"]["normal_reader_authority"] is False
    )
    assert run.attrs["source_refined_subject_masks_run"] == "refined_001"
    assert run.attrs["source_refined_subject_masks_revision_snapshot"] is True
    assert run.attrs["tail_angle_sample_count"] == 10
    assert (
        run["source_refined_subject_masks"].attrs["copied_from_subject_shape_run"]
        == "shape_001"
    )
    np.testing.assert_array_equal(
        np.asarray(
            run["source_refined_subject_masks"]["row_revision"][:], dtype=np.int64
        ),
        np.asarray([[3], [4]], dtype=np.int64),
    )
    assert run["source_acquisition_frame_index"][:].tolist() == [10, 11]
    assert run["instance_key"][:].tolist() == [1000, 1001]
    assert np.asarray(run["tail_angle_rad"][:], dtype=np.float32).shape == (2, 10)
    np.testing.assert_allclose(
        np.asarray(run["tail_angle_deg"][:], dtype=np.float32), 0.0, atol=1e-5
    )
    assert run.attrs["provenance"]["stage"] == "analysis.tail_kinematics_runs"
    assert run.attrs["materialization_mode"] == "bounded_streaming_single_writer"
    assert run.attrs["compute_kernel"] == "vectorized_shared_grid_v1"
    assert (
        run.attrs["provenance"]["parameters"]["compute_kernel"]
        == "vectorized_shared_grid_v1"
    )
    assert run.attrs["completed_block_count"] == 1
    assert tuple(run["tail_angle_rad"].chunks) == (2, 10)
    assert tuple(run["tail_angle_rad"].shards) == (2, 10)
    assert tuple(run["source_acquisition_frame_index"].shards) == (2,)


def test_byte_planned_tail_candidate_is_complete_but_never_selected(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root(row_count=20_000)
    shape = root["analysis/subject_shape_runs/shape_001"]
    frame_values = np.asarray(
        shape["source_acquisition_frame_index"][:], dtype=np.int64
    )
    del shape["source_acquisition_frame_index"]
    shape.create_array(
        "source_acquisition_frame_index",
        data=frame_values,
        chunks=(2_048,),
        overwrite=True,
    )
    parent = root["analysis"].require_group("tail_kinematics_runs")
    parent.attrs.update({"latest": "tail_existing", "latest_complete": "tail_existing"})

    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_candidate",
        tail_angle_sample_count=10,
        block_rows=3_000,
        execution_backend="serial",
        num_workers=1,
        storage_profile=PUBLISHED_HTTP_V1,
    )

    assert summary["status"] == "updated"
    assert summary["byte_planner_candidate"] is True
    assert summary["selector_eligible"] is False
    assert summary["direct_consolidated_array_declaration_count"] == 23
    fresh = zarr.open_group(root.store, mode="r", use_consolidated=False)
    parent = fresh["analysis/tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_existing"
    assert parent.attrs["latest_complete"] == "tail_existing"
    run = parent["tail_candidate"]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["analysis_storage_profile_id"] == "published_http_v1"
    assert run.attrs["analysis_storage_profile_role"] == (
        "explicit_unpromoted_candidate"
    )
    assert mod.validate_tail_kinematics_storage_receipt(run) == ()
    assert tuple(run["tail_angle_sample_xy"].chunks)[1:] == (10, 2)
    assert tuple(run["tail_angle_sample_xy"].shards)[1:] == (10, 2)
    assert tuple(run["source_acquisition_frame_index"].chunks) != tuple(
        run["tail_angle_sample_xy"].chunks
    )
    consolidated = zarr.open_group(root.store, mode="r", use_consolidated=True)
    assert (
        consolidated["analysis/tail_kinematics_runs/tail_candidate"].attrs[
            "stage_selector_eligible"
        ]
        is False
    )


def test_byte_planned_tail_candidate_rejects_parallel_or_partial_bundle(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root(row_count=3)
    shape = root["analysis/subject_shape_runs/shape_001"]
    frames = np.asarray(shape["source_acquisition_frame_index"][:], dtype=np.int64)
    del shape["source_acquisition_frame_index"]
    shape.create_array("source_acquisition_frame_index", data=frames, overwrite=True)

    with pytest.raises(ValueError, match="one serial writer"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_parallel_rejected",
            execution_backend="process_shards",
            num_workers=2,
            storage_profile=PUBLISHED_HTTP_V1,
            dry_run=True,
        )

    del shape["source_refined_subject_masks/row_revision_available"]
    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="partial optional bundle",
    ):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_partial_rejected",
            storage_profile=PUBLISHED_HTTP_V1,
            dry_run=True,
        )


@pytest.mark.parametrize("overwrite", [False, True])
def test_tail_kinematics_retry_never_invalidates_existing_publication(
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_immutable",
    )
    parent = root["analysis/tail_kinematics_runs"]
    existing = parent["tail_immutable"]
    before_attrs = dict(existing.attrs)
    before_parent = dict(parent.attrs)
    before_values = np.asarray(existing["tail_angle_rad"][:]).copy()

    with pytest.raises(ValueError, match="immutable"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_immutable",
            overwrite=overwrite,
        )

    assert dict(existing.attrs) == before_attrs
    assert dict(parent.attrs) == before_parent
    np.testing.assert_array_equal(existing["tail_angle_rad"][:], before_values)


def test_write_tail_kinematics_run_group_streams_aligned_blocks_with_whole_batch_parity(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2
    )
    root = _build_shape_root(row_count=9)
    expected = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **_source_arrays(row_count=9),
        tail_angle_sample_count=10,
    )
    source_slices: list[tuple[int, int]] = []
    original_read = mod._read_tail_kinematics_source_block

    def _recording_read(sources, row_slice):
        source_slices.append((int(row_slice.start), int(row_slice.stop)))
        return original_read(sources, row_slice)

    monkeypatch.setattr(mod, "_read_tail_kinematics_source_block", _recording_read)

    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_streamed",
        tail_angle_sample_count=10,
        block_rows=3,
        output_shard_rows=7,
    )

    assert source_slices == [(0, 4), (4, 8), (8, 9)]
    assert summary["requested_block_rows"] == 3
    assert summary["effective_block_rows"] == 4
    assert summary["requested_output_shard_rows"] == 7
    assert summary["effective_output_shard_rows"] == 8
    assert summary["block_count"] == 3
    assert summary["output_shard_count"] == 2
    assert summary["completed_block_count"] == 3
    run = root["analysis"]["tail_kinematics_runs"]["tail_streamed"]
    assert run.attrs["output_row_chunk"] == 2
    assert run.attrs["compute_block_rows_effective"] == 4
    assert run.attrs["output_shard_rows"] == 8
    assert run.attrs["completed_worker_task_count"] == 0
    assert tuple(run["tail_angle_rad"].shards) == (8, 10)
    # Copied lineage preserves its source logical chunk (9 rows), so its outer
    # shard cannot be smaller than that chunk even though metric arrays use 8.
    assert tuple(run["source_acquisition_frame_index"].shards) == (9,)
    assert run.attrs["palette_run_completion_status"] == "complete"
    np.testing.assert_allclose(
        np.asarray(run["tail_angle_rad"][:], dtype=np.float32),
        expected.tail_angle_rad,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(run["tail_lateral_deflection_px"][:], dtype=np.float32),
        expected.tail_lateral_deflection_px,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(run["valid"][:], dtype=bool),
        expected.valid,
    )
    assert run["source_acquisition_frame_index"][:].tolist() == list(range(10, 19))
    assert np.asarray(run["source_refined_subject_masks"]["row_revision"][:]).shape == (
        9,
        1,
    )


def test_write_tail_kinematics_run_group_dry_run_does_not_read_frame_blocks(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root(row_count=9)

    def _refuse_block_read(*_args, **_kwargs):
        raise AssertionError("dry-run must not read framewise source blocks")

    monkeypatch.setattr(mod, "_read_tail_kinematics_source_block", _refuse_block_read)
    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_dry_run",
        block_rows=3,
        dry_run=True,
    )

    assert summary["status"] == "planned"
    assert "tail_kinematics_runs" not in root["analysis"]


@pytest.mark.parametrize("overwrite", [False, True])
def test_tail_kinematics_existing_name_is_immutable_and_unchanged(
    overwrite: bool,
) -> None:
    root = zarr.group()
    parent = root.require_group("analysis/tail_kinematics_runs")
    existing = parent.create_group("tail_existing")
    existing.attrs.update(
        {
            mod.TAIL_PUBLICATION_OWNER_ATTR: ("11111111-1111-4111-8111-111111111111"),
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "sentinel": "preserve",
        }
    )
    existing.create_array("sentinel_values", data=np.asarray([3, 4], dtype=np.int16))
    parent.attrs.update({"latest": "tail_existing", "latest_complete": "tail_existing"})
    before_attrs = dict(existing.attrs)
    before_parent = dict(parent.attrs)
    before_values = np.asarray(existing["sentinel_values"][:]).copy()

    with pytest.raises(ValueError, match="immutable"):
        mod._prepare_tail_kinematics_run(
            root,
            target_run="tail_existing",
            shape_run_name="shape",
            shape_group=root.require_group("analysis/subject_shape_runs/shape"),
            row_count=2,
            tail_angle_sample_count=5,
            source_geometry_tail_sample_count=5,
            requested_block_rows=2,
            effective_block_rows=2,
            requested_output_shard_rows=2,
            effective_output_shard_rows=2,
            execution_backend="serial",
            worker_count_requested=1,
            worker_count_effective=1,
            source_publication_manifest_sha256="a" * 64,
            source_authority_mode="canonical_publication",
            source_authority={},
            stage_command="fixture",
            publication_owner_uuid="22222222-2222-4222-8222-222222222222",
            overwrite=overwrite,
        )

    assert dict(existing.attrs) == before_attrs
    assert dict(parent.attrs) == before_parent
    np.testing.assert_array_equal(existing["sentinel_values"][:], before_values)


def test_process_shards_reject_output_shards_that_split_compute_blocks(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2
    )
    root = _build_shape_root(row_count=9)

    with pytest.raises(ValueError, match="whole number of effective compute blocks"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_invalid_grids",
            block_rows=3,
            output_shard_rows=5,
            execution_backend="process_shards",
            num_workers=2,
            dry_run=True,
        )


def test_write_tail_kinematics_run_group_marks_partial_stream_failed(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2
    )
    root = _build_shape_root(row_count=9)
    original_compute = mod.compute_tail_kinematics_from_subject_shape_arrays
    call_count = 0

    def _fail_second_block(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("intentional block failure")
        return original_compute(**kwargs)

    monkeypatch.setattr(
        mod, "compute_tail_kinematics_from_subject_shape_arrays", _fail_second_block
    )

    with pytest.raises(RuntimeError, match="intentional block failure"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_failed",
            block_rows=3,
        )

    parent = root["analysis"]["tail_kinematics_runs"]
    run = parent["tail_failed"]
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["completed_block_count"] == 1
    assert parent.attrs.get("latest") != "tail_failed"


def test_tail_kinematics_early_lifecycle_failure_retains_owned_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    observed_owner: list[str] = []

    def fail_started(run_group, **_kwargs):
        owner = run_group.attrs.get(mod.TAIL_PUBLICATION_OWNER_ATTR)
        assert isinstance(owner, str) and owner
        assert run_group.attrs.get("stage_selector_eligible") is False
        observed_owner.append(owner)
        raise RuntimeError("injected early tail lifecycle failure")

    monkeypatch.setattr(mod, "mark_run_started", fail_started)

    with pytest.raises(RuntimeError, match="injected early tail lifecycle failure"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_early_failed",
        )

    failed = root["analysis/tail_kinematics_runs/tail_early_failed"]
    assert failed.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR] == observed_owner[0]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    parent = root["analysis/tail_kinematics_runs"]
    assert parent.attrs.get("latest") != "tail_early_failed"
    assert parent.attrs.get("latest_complete") != "tail_early_failed"


def test_tail_kinematics_persist_then_raise_create_recovers_owned_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    run_name = "tail_create_ambiguous"
    parent = root.require_group("analysis/tail_kinematics_runs")
    parent.attrs["latest_pending"] = run_name
    original_create = mod._create_tail_kinematics_public_candidate

    def persist_then_raise(parent, *, run_name, publication_owner_uuid):
        original_create(
            parent,
            run_name=run_name,
            publication_owner_uuid=publication_owner_uuid,
        )
        raise RuntimeError("injected tail create acknowledgement loss")

    monkeypatch.setattr(
        mod,
        "_create_tail_kinematics_public_candidate",
        persist_then_raise,
    )

    with pytest.raises(RuntimeError, match="create acknowledgement loss"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name=run_name,
        )

    failed = root[f"analysis/tail_kinematics_runs/{run_name}"]
    owner = failed.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR]
    tombstone = failed.attrs[mod.TAIL_PUBLICATION_TOMBSTONE_ATTR]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False
    assert tombstone["publication_owner_uuid"] == owner
    assert tombstone["run_family"] == "tail_kinematics"
    assert parent.attrs["latest_pending"] == run_name


def test_tail_kinematics_failure_cleanup_never_clobbers_recreated_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    parent = root.require_group("analysis/tail_kinematics_runs")
    run_name = "tail_cleanup_takeover"
    original_write = mod._write_tail_kinematics_failure_attr
    takeover_injected = False

    monkeypatch.setattr(
        mod,
        "publish_tail_kinematics_coordinate_surfaces",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected tail source failure")
        ),
    )

    def hostile_write(attrs, name, value):
        nonlocal takeover_injected
        original_write(attrs, name, value)
        if name == "stage_selector_eligible" and not takeover_injected:
            takeover_injected = True
            del parent[run_name]
            parent.create_group(
                run_name,
                attributes={
                    mod.TAIL_PUBLICATION_OWNER_ATTR: "alien-tail-owner",
                    "palette_run_completion_status": "complete",
                    "stage_selector_eligible": True,
                    "sentinel": "successor-preserved",
                },
            )

    monkeypatch.setattr(mod, "_write_tail_kinematics_failure_attr", hostile_write)

    with pytest.raises(RuntimeError, match="injected tail source failure"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name=run_name,
        )

    successor = parent[run_name]
    assert takeover_injected is True
    assert successor.attrs[mod.TAIL_PUBLICATION_OWNER_ATTR] == "alien-tail-owner"
    assert successor.attrs["palette_run_completion_status"] == "complete"
    assert successor.attrs["stage_selector_eligible"] is True
    assert successor.attrs["sentinel"] == "successor-preserved"
    assert mod.TAIL_PUBLICATION_TOMBSTONE_ATTR not in successor.attrs


def test_write_tail_kinematics_run_group_copies_instance_key_lineage(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    shape = root["analysis"]["subject_shape_runs"]["shape_001"]
    shape.create_array(
        "instance_key", data=np.asarray([11, 22], dtype=np.uint64), overwrite=True
    )
    shape.create_array(
        "source_crop_row_ids", data=np.asarray([5, 6], dtype=np.int64), overwrite=True
    )

    mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_001",
        tail_angle_sample_count=10,
    )

    run = root["analysis"]["tail_kinematics_runs"]["tail_001"]
    assert run["instance_key"][:].tolist() == [11, 22]
    assert run["source_crop_row_ids"][:].tolist() == [5, 6]
    assert "instance_key" in run.attrs["row_lineage_copied"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_copied"]


def test_write_tail_kinematics_run_group_rejects_missing_direct_instance_key(
    monkeypatch,
) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    del root["analysis/subject_shape_runs/shape_001/instance_key"]

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError, match="required array 'instance_key'"
    ):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_001",
            tail_angle_sample_count=10,
        )


def test_write_tail_kinematics_rejects_acquisition_frame_alias(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    shape = root["analysis/subject_shape_runs/shape_001"]
    values = np.asarray(shape["source_acquisition_frame_index"][:])
    del shape["source_acquisition_frame_index"]
    shape.create_array("frame_indices", data=values)

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="source_acquisition_frame_index",
    ):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_001",
            tail_angle_sample_count=10,
        )


def test_write_tail_kinematics_rejects_body_frame_valid_alias(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    body_frame = root["analysis/subject_shape_runs/shape_001/body_frame"]
    values = np.asarray(body_frame["axis_valid"][:])
    del body_frame["axis_valid"]
    body_frame.create_array("valid", data=values)

    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="body_frame/axis_valid",
    ):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_001",
            tail_angle_sample_count=10,
        )
