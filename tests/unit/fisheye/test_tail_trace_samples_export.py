from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analytics_exports import tail_trace_samples as mod
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.detect_reason_codec import encode_reason_bytes
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.analytics_exports.contracts import TAIL_TRACE_SAMPLES_TABLE
from fisheye.analytics_exports.publication import sha256_file
from fisheye.analytics_exports.runtime_telemetry import (
    validate_export_runtime_telemetry,
)
from fisheye.analytics_exports.validation import (
    ExportValidationError,
    validate_export_run,
)
from tests.unit.fisheye.test_kinematics_samples_export import _eligible_source


class _Group(dict):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None) -> None:
        super().__init__(*args)
        self.attrs = dict(attrs or {})


def _projection_inputs() -> dict[str, np.ndarray | float]:
    sample_s = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    tail_xy = np.asarray(
        [
            [[0.0, 0.0], [-5.0, 0.0], [-10.0, 0.0]],
            [[0.0, 0.0], [-5.0, 1.0], [-10.0, 2.0]],
            [[0.0, 0.0], [-5.0, -1.0], [-10.0, -2.0]],
        ],
        dtype=np.float32,
    )
    return {
        "source_tail_row_indices": np.asarray([10, 11, 12], dtype=np.int64),
        "track_ids": np.asarray([7, 7, 9], dtype=np.int64),
        "instance_keys": np.asarray([101, 102, 103], dtype=np.uint64),
        "source_crop_row_ids": np.asarray([20, 21, 22], dtype=np.int64),
        "source_acquisition_frame_indices": np.asarray([100, 101, 102], dtype=np.int64),
        "source_tail_row_valid": np.asarray([True, False, True], dtype=bool),
        "source_failure_reasons": np.asarray(
            ["ok", "tail_segment_invalid", "ok"], dtype=object
        ),
        "tail_sample_s": sample_s,
        "tail_sample_xy": tail_xy,
        "tail_angle_rad": np.asarray(
            [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2], [0.0, -0.1, -0.2]],
            dtype=np.float32,
        ),
        "tail_curvature_px_inv": np.zeros((3, 3), dtype=np.float32),
        "tail_lateral_deflection_px": tail_xy[:, :, 1].copy(),
        "tail_base_xy": np.zeros((3, 2), dtype=np.float32),
        "reference_length_px": np.asarray([10.0, 10.0, 0.0], dtype=np.float32),
        "reference_length_source_valid": np.asarray([True, True, True], dtype=bool),
        "body_forward_axis_xy": np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32), (3, 1)
        ),
        "body_left_axis_xy": np.tile(
            np.asarray([[0.0, 1.0]], dtype=np.float32), (3, 1)
        ),
        "body_frame_valid": np.ones(3, dtype=bool),
        "source_sample_rate_hz": 100.0,
    }


def test_projection_contract_is_closed_and_digest_bound() -> None:
    contract = mod.tail_trace_projection_contract()

    assert contract["reference_length_kind"] == (
        "tail_base_to_tail_tip_centerline_arclength_px"
    )
    assert contract["reason_registry"] == {
        "0": "valid",
        "1": "source_tail_row_invalid",
        "2": "reference_length_or_body_frame_invalid",
        "3": "derived_geometry_nonfinite",
    }
    body = dict(contract)
    digest = body.pop("payload_sha256")
    assert digest == canonical_json_sha256(body)


def test_projection_preserves_long_form_order_and_body_geometry() -> None:
    columns = mod.project_tail_trace_window(**_projection_inputs())

    assert list(columns) == [
        "source_tail_row_index",
        "track_id",
        "instance_key",
        "source_crop_row_id",
        "source_acquisition_frame_index",
        "time_seconds",
        "tail_sample_index",
        "normalized_tail_position",
        "reference_length_px",
        "body_longitudinal_fraction",
        "body_lateral_fraction",
        "tangent_angle_rad",
        "body_curvature_dimensionless",
        "source_camera_x_px",
        "source_camera_y_px",
        "source_camera_curvature_px_inv",
        "source_lateral_deflection_px",
        "source_tail_row_valid",
        "reference_length_valid",
        "sample_valid",
        "sample_reason_code",
        "source_failure_reason",
    ]
    np.testing.assert_array_equal(
        columns["source_tail_row_index"], [10] * 3 + [11] * 3 + [12] * 3
    )
    np.testing.assert_array_equal(columns["tail_sample_index"], [0, 1, 2] * 3)
    np.testing.assert_allclose(
        columns["body_longitudinal_fraction"][:3], [0.0, 0.5, 1.0]
    )
    np.testing.assert_allclose(columns["body_lateral_fraction"][:3], 0.0)
    np.testing.assert_allclose(
        columns["body_curvature_dimensionless"][:3], 0.2, atol=1e-6
    )
    np.testing.assert_allclose(columns["time_seconds"][:3], 1.0)
    assert columns["sample_valid"].tolist() == [True] * 3 + [False] * 6
    assert columns["sample_reason_code"].tolist() == [0] * 3 + [1] * 3 + [2] * 3
    assert np.isnan(columns["source_camera_x_px"][3:]).all()
    assert columns["normalized_tail_position"].tolist() == [0.0, 0.5, 1.0] * 3
    for name, dtype in mod.TAIL_TRACE_SCIENTIFIC_DTYPES.items():
        assert columns[name].dtype == dtype, name


def test_projection_marks_nonfinite_sample_without_relabeling_other_samples() -> None:
    inputs = _projection_inputs()
    inputs["source_tail_row_valid"] = np.ones(3, dtype=bool)
    inputs["reference_length_px"] = np.full(3, 10.0, dtype=np.float32)
    inputs["tail_curvature_px_inv"][0, 1] = np.nan

    columns = mod.project_tail_trace_window(**inputs)

    assert columns["sample_reason_code"][:3].tolist() == [0, 3, 0]
    assert columns["sample_valid"][:3].tolist() == [True, False, True]
    assert np.isnan(columns["source_camera_x_px"][1])
    assert np.isfinite(columns["source_camera_x_px"][[0, 2]]).all()


def _track_source(
    *,
    keys: tuple[int, ...] = (101, 102, 103),
    frames: tuple[int, ...] = (100, 101, 102),
) -> SimpleNamespace:
    frame_values = np.asarray(frames, dtype=np.int64)
    sample_keys = np.column_stack(
        (np.full(len(frames), 7, dtype=np.int64), frame_values)
    )
    instances = np.zeros(len(keys), dtype=TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE)
    instances["valid"] = True
    instances["instance_key"] = np.asarray(keys, dtype=np.uint64)
    group = _Group(
        {
            "track_sample_key": sample_keys,
            "source_acquisition_frame_index": frame_values,
            "source_instance_key": instances,
        }
    )
    records = {
        name: {"content_sha256": array_values_sha256(values)}
        for name, values in group.items()
    }
    binding = {
        "tracks": [
            {
                "track_id": 7,
                "sample_count": len(keys),
                "selected_surfaces": records,
            }
        ]
    }
    return SimpleNamespace(
        binding=binding,
        run_group=_Group({"tracks": _Group({"id_7": group})}),
    )


def test_track_identity_index_is_bounded_unique_and_frame_bound() -> None:
    index = mod._build_track_identity_index(  # noqa: SLF001
        _track_source(), source_window_rows=2
    )

    np.testing.assert_array_equal(index.instance_keys, [101, 102, 103])
    np.testing.assert_array_equal(index.track_ids, [7, 7, 7])
    joined = mod._join_track_identities(  # noqa: SLF001
        index,
        instance_keys=np.asarray([103, 101], dtype=np.uint64),
        frame_indices=np.asarray([102, 100], dtype=np.int64),
    )
    np.testing.assert_array_equal(joined, [7, 7])
    assert index.record["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in index.record.items() if key != "payload_sha256"}
    )

    with pytest.raises(ValueError, match="disagree on camera frame"):
        mod._join_track_identities(  # noqa: SLF001
            index,
            instance_keys=np.asarray([101], dtype=np.uint64),
            frame_indices=np.asarray([999], dtype=np.int64),
        )


def test_track_identity_index_rejects_duplicate_observation_membership() -> None:
    with pytest.raises(ValueError, match="multiple track samples"):
        mod._build_track_identity_index(  # noqa: SLF001
            _track_source(keys=(101, 101, 103)),
            source_window_rows=2,
        )


def _manifest(
    arrays: dict[str, np.ndarray],
    *,
    digest: str,
    run_path: str | None = None,
) -> SimpleNamespace:
    records = {
        path: {
            "relative_ref": path,
            "dtype": np.dtype(values.dtype).str,
            "shape": [int(value) for value in values.shape],
            "content_sha256": array_values_sha256(values),
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
            **(
                {"array_ref": f"/{run_path.strip('/')}/{path}"}
                if run_path is not None
                else {}
            ),
        }
        for path, values in arrays.items()
    }
    return SimpleNamespace(record={"arrays": records}, record_sha256=digest)


def test_source_binder_proves_tail_shape_and_track_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tail_arrays = {
        "instance_key": np.asarray([101, 102, 103], dtype=np.uint64),
        "source_crop_row_ids": np.asarray([20, 21, 22], dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([100, 101, 102], dtype=np.int64),
        "valid": np.ones(3, dtype=bool),
        "failure_reason_bytes": encode_reason_bytes(np.asarray(["ok"] * 3)),
        "tail_angle_sample_s": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "tail_angle_sample_xy": np.zeros((3, 3, 2), dtype=np.float32),
        "tail_angle_rad": np.zeros((3, 3), dtype=np.float32),
        "tail_curvature_px_inv": np.zeros((3, 3), dtype=np.float32),
        "tail_lateral_deflection_px": np.zeros((3, 3), dtype=np.float32),
    }
    array_schema = {
        "schema_id": "tail-array-schema",
        "payload": {"declarations": [{"byte_planner_adopted": True}]},
        "payload_digest": "1" * 64,
    }
    tail = _Group(
        tail_arrays,
        attrs={
            "schema_id": mod.TAIL_KINEMATICS_SCHEMA_ID,
            "schema_version": mod.TAIL_KINEMATICS_SCHEMA_VERSION,
            "palette_run_completion_status": "complete",
            "palette_run_completed_at_utc": "2026-08-04T12:00:00+00:00",
            "stage_selector_eligible": True,
            "source_subject_shape_publication_manifest_sha256": "b" * 64,
            "tail_kinematics_array_schema": array_schema,
            "tail_kinematics_array_schema_sha256": "1" * 64,
        },
    )
    projection_inputs = _projection_inputs()
    shape_arrays = {
        "instance_key": tail_arrays["instance_key"].copy(),
        "source_crop_row_ids": tail_arrays["source_crop_row_ids"].copy(),
        "source_acquisition_frame_index": tail_arrays[
            "source_acquisition_frame_index"
        ].copy(),
        "components/subject_body/tail_base_xy": projection_inputs["tail_base_xy"],
        "components/subject_body/tail_segment_arclength_px": np.full(
            3, 10.0, dtype=np.float32
        ),
        "components/subject_body/tail_base_valid": np.ones(3, dtype=bool),
        "body_frame/forward_axis_xy": projection_inputs["body_forward_axis_xy"],
        "body_frame/left_axis_xy": projection_inputs["body_left_axis_xy"],
        "body_frame/axis_valid": np.ones(3, dtype=bool),
    }
    shape_nested = _Group(
        {
            "instance_key": shape_arrays["instance_key"],
            "source_crop_row_ids": shape_arrays["source_crop_row_ids"],
            "source_acquisition_frame_index": shape_arrays[
                "source_acquisition_frame_index"
            ],
            "components": _Group(
                {
                    "subject_body": _Group(
                        {
                            "tail_base_xy": shape_arrays[
                                "components/subject_body/tail_base_xy"
                            ],
                            "tail_segment_arclength_px": shape_arrays[
                                "components/subject_body/tail_segment_arclength_px"
                            ],
                            "tail_base_valid": shape_arrays[
                                "components/subject_body/tail_base_valid"
                            ],
                        }
                    )
                }
            ),
            "body_frame": _Group(
                {
                    "forward_axis_xy": shape_arrays["body_frame/forward_axis_xy"],
                    "left_axis_xy": shape_arrays["body_frame/left_axis_xy"],
                    "axis_valid": shape_arrays["body_frame/axis_valid"],
                }
            ),
        },
        attrs={"schema_id": "analysis.subject_shape_runs", "schema_version": 4},
    )
    reference = SimpleNamespace(
        array_node=shape_arrays["components/subject_body/tail_segment_arclength_px"],
        validity_node=shape_arrays["components/subject_body/tail_base_valid"],
        semantics=SimpleNamespace(record_sha256="c" * 64),
    )
    shape_publication = SimpleNamespace(
        _run=shape_nested,
        run_path="analysis/subject_shape_runs/shape_1",
        manifest=_manifest(
            shape_arrays,
            digest="b" * 64,
            run_path="analysis/subject_shape_runs/shape_1",
        ),
        body_frame=SimpleNamespace(record_sha256="d" * 64),
        selector_eligible=True,
        require_scalar_surface=lambda *_args, **_kwargs: reference,
    )
    tail_publication = SimpleNamespace(
        _run=tail,
        source=shape_publication,
        manifest=_manifest(tail_arrays, digest="a" * 64),
    )
    track_source = _track_source()
    track_source.binding.update(
        {
            "source_sample_rate_hz": 100.0,
            "source_manifest_sha256": "e" * 64,
            "scope": "offline",
            "run_name": "track_1",
            "run_path": "analysis/track_kinematics_runs/offline/track_1",
            "completion_snapshot": {
                "status": "complete",
                "completed_at_utc": "2026-08-04T12:00:00+00:00",
                "selector_eligible": True,
            },
        }
    )
    track_payload = dict(track_source.binding)
    track_source.binding = {
        **track_payload,
        "payload_sha256": canonical_json_sha256(track_payload),
    }
    root = _Group(attrs={"fps": 100.0})
    monkeypatch.setattr(
        mod,
        "load_tail_kinematics_coordinate_publication",
        lambda *_args, **_kwargs: tail_publication,
    )
    monkeypatch.setattr(
        mod,
        "validate_tail_kinematics_array_schema",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        mod.track_export,
        "_source_binding",
        lambda *_args, **_kwargs: track_source,
    )

    bound = mod.bind_tail_trace_sources(
        root,
        zarr_path=tmp_path / "recording_analysis.zarr",
        tail_kinematics_run="tail_1",
        subject_shape_run="shape_1",
        track_kinematics_run="track_1",
        track_scope="offline",
        source_window_rows=2,
    )

    assert bound.binding["tail_row_count"] == 3
    assert bound.binding["source_tail_sample_count"] == 3
    assert bound.binding["tail_byte_planner_adopted"] is True
    assert bound.binding["track_identity_index"]["row_count"] == 3
    assert bound.binding["payload_sha256"] == canonical_json_sha256(
        {key: value for key, value in bound.binding.items() if key != "payload_sha256"}
    )
    projected = mod.read_projected_tail_trace_window(
        bound,
        start_row=0,
        stop_row=2,
    )
    assert projected["track_id"].tolist() == [7] * 6

    changed = deepcopy(track_source.binding)
    changed["source_sample_rate_hz"] = 50.0
    track_source.binding = changed
    with pytest.raises(ValueError, match="disagree on exact source FPS"):
        mod.bind_tail_trace_sources(
            root,
            zarr_path=tmp_path / "recording_analysis.zarr",
            tail_kinematics_run="tail_1",
            subject_shape_run="shape_1",
            track_kinematics_run="track_1",
            track_scope="offline",
            source_window_rows=2,
        )


def _publisher_bound_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> mod.BoundTailTraceSources:
    root, _run, _track = _eligible_source(monkeypatch)
    source_path = (tmp_path / "recording_analysis.zarr").resolve()
    track_source = mod.track_export._source_binding(
        root,
        zarr_path=source_path,
        recording_id="recording",
        run_name="motion_physical",
        scope="offline",
    )
    index = mod._build_track_identity_index(  # noqa: SLF001
        track_source, source_window_rows=1
    )
    row_count = int(index.instance_keys.size)
    assert row_count > 0
    sample_s = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    tail_xy = np.zeros((row_count, 3, 2), dtype=np.float32)
    tail_xy[:, :, 0] = np.asarray([0.0, -5.0, -10.0], dtype=np.float32)
    tail_arrays = {
        "instance_key": index.instance_keys.copy(),
        "source_crop_row_ids": np.arange(row_count, dtype=np.int64) + 20,
        "source_acquisition_frame_index": index.frame_indices.copy(),
        "valid": np.ones(row_count, dtype=bool),
        "failure_reason_bytes": encode_reason_bytes(np.asarray(["ok"] * row_count)),
        "tail_angle_sample_xy": tail_xy,
        "tail_angle_rad": np.zeros((row_count, 3), dtype=np.float32),
        "tail_curvature_px_inv": np.zeros((row_count, 3), dtype=np.float32),
        "tail_lateral_deflection_px": np.zeros((row_count, 3), dtype=np.float32),
    }
    tail = _Group(tail_arrays)
    reference = np.full(row_count, 10.0, dtype=np.float32)
    reference_valid = np.ones(row_count, dtype=bool)
    shape_flat = {
        "instance_key": tail_arrays["instance_key"].copy(),
        "source_crop_row_ids": tail_arrays["source_crop_row_ids"].copy(),
        "source_acquisition_frame_index": tail_arrays[
            "source_acquisition_frame_index"
        ].copy(),
        "components/subject_body/tail_base_xy": np.zeros(
            (row_count, 2), dtype=np.float32
        ),
        "components/subject_body/tail_segment_arclength_px": reference,
        "components/subject_body/tail_base_valid": reference_valid,
        "body_frame/forward_axis_xy": np.tile(
            np.asarray([[1.0, 0.0]], dtype=np.float32), (row_count, 1)
        ),
        "body_frame/left_axis_xy": np.tile(
            np.asarray([[0.0, 1.0]], dtype=np.float32), (row_count, 1)
        ),
        "body_frame/axis_valid": np.ones(row_count, dtype=bool),
    }
    shape = _Group(
        {
            "instance_key": shape_flat["instance_key"],
            "source_crop_row_ids": shape_flat["source_crop_row_ids"],
            "source_acquisition_frame_index": shape_flat[
                "source_acquisition_frame_index"
            ],
            "components": _Group(
                {
                    "subject_body": _Group(
                        {
                            "tail_base_xy": shape_flat[
                                "components/subject_body/tail_base_xy"
                            ],
                            "tail_segment_arclength_px": reference,
                            "tail_base_valid": reference_valid,
                        }
                    )
                }
            ),
            "body_frame": _Group(
                {
                    "forward_axis_xy": shape_flat["body_frame/forward_axis_xy"],
                    "left_axis_xy": shape_flat["body_frame/left_axis_xy"],
                    "axis_valid": shape_flat["body_frame/axis_valid"],
                }
            ),
        }
    )
    selected_tail = {
        path: {
            "relative_ref": path,
            "dtype": np.dtype(values.dtype).str,
            "shape": [int(value) for value in values.shape],
            "content_sha256": array_values_sha256(values),
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
        for path, values in tail_arrays.items()
    }
    selected_shape = {
        path: {
            "array_ref": f"/analysis/subject_shape_runs/shape_1/{path}",
            "relative_ref": path,
            "dtype": np.dtype(values.dtype).str,
            "shape": [int(value) for value in values.shape],
            "content_sha256": array_values_sha256(values),
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
        for path, values in shape_flat.items()
        if path in mod._SHAPE_WINDOW_ARRAYS  # noqa: SLF001
    }
    payload = {
        "schema_id": mod.TAIL_TRACE_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": mod.TAIL_TRACE_SOURCE_BINDING_SCHEMA_VERSION,
        "stage_id": "tail_traces",
        "recording_id": "recording",
        "zarr_path": str(source_path),
        "source_sample_rate_hz": track_source.binding["source_sample_rate_hz"],
        "source_sample_rate_authority": "archive_root.attrs.fps",
        "tail_run_name": "tail_1",
        "tail_run_path": "analysis/tail_kinematics_runs/tail_1",
        "tail_schema_id": mod.TAIL_KINEMATICS_SCHEMA_ID,
        "tail_schema_version": mod.TAIL_KINEMATICS_SCHEMA_VERSION,
        "tail_publication_manifest_sha256": "a" * 64,
        "tail_array_schema_manifest_sha256": "b" * 64,
        "tail_array_schema_payload_sha256": "c" * 64,
        "tail_byte_planner_adopted": True,
        "tail_row_count": row_count,
        "source_tail_sample_count": 3,
        "source_tail_sample_axis_sha256": array_values_sha256(sample_s),
        "subject_shape_run_name": "shape_1",
        "subject_shape_run_path": "analysis/subject_shape_runs/shape_1",
        "subject_shape_schema_id": "analysis.subject_shape_runs",
        "subject_shape_schema_version": 4,
        "subject_shape_publication_manifest_sha256": "d" * 64,
        "body_frame_record_sha256": "e" * 64,
        "reference_length_semantics_sha256": "f" * 64,
        "reference_length_content_sha256": array_values_sha256(reference),
        "track_source_binding": track_source.binding,
        "track_identity_index": index.record,
        "selected_tail_arrays": selected_tail,
        "selected_subject_shape_arrays": selected_shape,
        "completion_snapshot": {
            "tail_status": "complete",
            "tail_completed_at_utc": "2026-08-04T12:00:00+00:00",
            "tail_selector_eligible": True,
            "subject_shape_selector_eligible": True,
            "track_status": "complete",
            "track_completed_at_utc": track_source.binding["completion_snapshot"][
                "completed_at_utc"
            ],
            "track_selector_eligible": True,
        },
    }
    binding = {**payload, "payload_sha256": canonical_json_sha256(payload)}
    return mod.BoundTailTraceSources(
        binding=binding,
        tail_run=tail,
        subject_shape_run=shape,
        track_source=track_source,
        track_index=index,
        tail_sample_s=sample_s,
        reference_length_node=reference,
        reference_validity_node=reference_valid,
    )


def _export_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    output_name: str,
    run_id: str,
    window_rows: int,
    part_rows: int,
) -> dict[str, object]:
    bound = _publisher_bound_source(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mod,
        "bind_tail_trace_sources",
        lambda *_args, **_kwargs: bound,
    )
    return mod.export_tail_trace_samples(
        tmp_path / "recording_analysis.zarr",
        tail_kinematics_run="tail_1",
        subject_shape_run="shape_1",
        track_kinematics_run="motion_physical",
        track_scope="offline",
        output_root=tmp_path / output_name,
        export_run_id=run_id,
        scratch_root=tmp_path / f"scratch_{output_name}",
        source_window_rows=window_rows,
        source_rows_per_part=part_rows,
        row_group_rows=1,
    )


def test_tail_publisher_is_batch_independent_and_manifest_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _export_tail(
        monkeypatch,
        tmp_path,
        output_name="exports_a",
        run_id="tail_a",
        window_rows=1,
        part_rows=1,
    )
    second = _export_tail(
        monkeypatch,
        tmp_path,
        output_name="exports_b",
        run_id="tail_b",
        window_rows=2,
        part_rows=2,
    )

    assert (
        first["tail_trace_export"]["projected_payload"]
        == second["tail_trace_export"]["projected_payload"]
    )
    assert first["tail_trace_validation"]["valid"] is True
    validate_export_runtime_telemetry(first["runtime_telemetry"])
    assert "runtime_telemetry" not in json.loads(
        Path(first["manifest_path"]).read_text(encoding="utf-8")
    )
    assert (
        len(first["part_files_by_table"][TAIL_TRACE_SAMPLES_TABLE])
        == first["tail_trace_export"]["source_binding"]["tail_row_count"]
    )
    report = validate_export_run(tmp_path / "exports_a", "tail_a")
    assert report["status"] == "valid"

    import pyarrow.parquet as pq

    parts = sorted((tmp_path / "exports_a").rglob("*.parquet"))
    table = pq.read_table(parts).to_pydict()
    sample_count = first["tail_trace_export"]["source_binding"][
        "source_tail_sample_count"
    ]
    assert table["tail_sample_index"][:sample_count] == list(range(sample_count))
    assert table["track_id"] == [7] * len(table["track_id"])


def test_tail_validator_rejects_rehashed_constant_column_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _export_tail(
        monkeypatch,
        tmp_path,
        output_name="exports",
        run_id="tail_tamper",
        window_rows=1,
        part_rows=8,
    )
    manifest_path = Path(result["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    part = next((tmp_path / "exports").rglob("*.parquet"))

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(part)
    schema = parquet.schema_arrow
    table = parquet.read()
    index = table.schema.get_field_index("reference_length_kind")
    arrays = [table.column(column) for column in range(table.num_columns)]
    arrays[index] = pa.chunked_array(
        [pa.array(["body_length"] * table.num_rows, type=pa.string())]
    )
    rewritten = pa.Table.from_arrays(arrays, schema=schema)
    writer = pq.ParquetWriter(
        part,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=result["tail_trace_export"]["parquet_policy"][
            "dictionary_columns"
        ],
    )
    try:
        writer.write_table(rewritten, row_group_size=1)
    finally:
        writer.close()
    entry = payload["publication"]["parts_by_table"][TAIL_TRACE_SAMPLES_TABLE][0]
    entry["sha256"] = sha256_file(part)
    entry["size_bytes"] = part.stat().st_size
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ExportValidationError, match="reference_length_kind changed"):
        validate_export_run(tmp_path / "exports", "tail_tamper")


def test_tail_validator_rejects_rehashed_nested_array_declaration_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _export_tail(
        monkeypatch,
        tmp_path,
        output_name="exports",
        run_id="tail_nested_tamper",
        window_rows=1,
        part_rows=8,
    )
    manifest_path = Path(result["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope = payload["tail_trace_export"]
    source = envelope["source_binding"]
    source["selected_tail_arrays"]["instance_key"]["unexpected"] = True
    source["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in source.items() if key != "payload_sha256"}
    )
    envelope["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in envelope.items() if key != "payload_sha256"}
    )
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ExportValidationError, match="selected_tail_arrays record"):
        validate_export_run(tmp_path / "exports", "tail_nested_tamper")


def test_failed_tail_replacement_preserves_visible_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _export_tail(
        monkeypatch,
        tmp_path,
        output_name="exports",
        run_id="tail_replace",
        window_rows=1,
        part_rows=8,
    )
    manifest_path = Path(first["manifest_path"])
    original = manifest_path.read_bytes()
    bound = _publisher_bound_source(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mod,
        "bind_tail_trace_sources",
        lambda *_args, **_kwargs: bound,
    )
    original_validator = mod._decoded_part_validation  # noqa: SLF001
    monkeypatch.setattr(
        mod,
        "_decoded_part_validation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected staged decode failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected staged decode failure"):
        mod.export_tail_trace_samples(
            tmp_path / "recording_analysis.zarr",
            tail_kinematics_run="tail_1",
            subject_shape_run="shape_1",
            track_kinematics_run="motion_physical",
            track_scope="offline",
            output_root=tmp_path / "exports",
            export_run_id="tail_replace",
            scratch_root=tmp_path / "scratch_replace",
            source_window_rows=1,
            source_rows_per_part=8,
            row_group_rows=1,
            overwrite=True,
        )
    assert manifest_path.read_bytes() == original
    monkeypatch.setattr(mod, "_decoded_part_validation", original_validator)
    assert validate_export_run(tmp_path / "exports", "tail_replace")["status"] == (
        "valid"
    )
