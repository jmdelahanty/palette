from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

import fisheye.tracking.crop as crop_module
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_started,
)


def _source_group(
    root: Any,
    path: str,
    *,
    center: tuple[float, float] = (0.5, 0.5),
) -> Any:
    source = root.create_group(path)
    source.create_array("instance_key", data=np.asarray([11], dtype=np.uint64))
    source.create_array("frame_indices", data=np.asarray([0], dtype=np.int64))
    source.create_array(
        "bbox_norm_coords",
        data=np.asarray([[center[0], center[1], 0.2, 0.2]], dtype=np.float64),
    )
    return source


def _source_geometry_values(
    *,
    center: tuple[float, float] = (5.0, 5.0),
    bbox_norm_center: tuple[float, float] = (0.5, 0.5),
) -> dict[str, np.ndarray]:
    return {
        "instance_key": np.asarray([11], dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray([0], dtype=np.int64),
        "bbox_norm_coords": np.asarray(
            [[bbox_norm_center[0], bbox_norm_center[1], 0.2, 0.2]],
            dtype=np.float64,
        ),
        "bbox_img_xyxy": np.asarray([[4.0, 4.0, 6.0, 6.0]], dtype=np.float64),
        "centers_img_xy": np.asarray([[center[0], center[1]]], dtype=np.float64),
    }


def _fake_geometry() -> Any:
    return SimpleNamespace(
        frame_evidence=SimpleNamespace(
            acquisition_frame=SimpleNamespace(
                record=SimpleNamespace(source_total_frames=1),
            )
        )
    )


def _install_fake_acquisition_authority(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    source_video_metadata: dict[str, Any] | None = None,
) -> None:
    acquisition_record = SimpleNamespace(
        camera_id="cam0",
        height_px=10,
        width_px=10,
        source_total_frames=1,
        source_video_metadata=source_video_metadata or {},
    )
    acquisition = SimpleNamespace(
        record=acquisition_record,
        record_ref="analysis/acquisition_camera_frames/cam0",
        record_sha256="a" * 64,
    )
    ownership = SimpleNamespace(record=SimpleNamespace(mode=mode))
    status = SimpleNamespace(
        status=crop_module.ACQUISITION_AUTHORITY_PUBLISHED,
        authority_mode=mode,
        authority_path="analysis/acquisition_camera_frames/cam0",
    )
    source_geometry = SimpleNamespace(
        frame_evidence=SimpleNamespace(
            acquisition_frame=SimpleNamespace(
                record=acquisition_record,
                record_ref=acquisition.record_ref,
                record_sha256=acquisition.record_sha256,
            )
        )
    )
    monkeypatch.setattr(
        crop_module,
        "load_acquisition_authority_publication_status",
        lambda _root: status,
    )
    monkeypatch.setattr(
        crop_module,
        "load_persisted_acquisition_camera_authority",
        lambda _root: (ownership, acquisition),
    )
    monkeypatch.setattr(
        crop_module,
        "load_persisted_detection_observation_geometry",
        lambda _root, _path: source_geometry,
    )
    monkeypatch.setattr(
        crop_module,
        "detection_observation_geometry_values",
        lambda _source: _source_geometry_values(),
    )


def test_canonical_crop_arrays_preserve_existing_placement_numerics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = _source_group(root, "detect_runs/d1")
    values = _source_geometry_values()
    monkeypatch.setattr(
        crop_module,
        "detection_observation_geometry_values",
        lambda _source: values,
    )

    result = crop_module._canonical_crop_arrays(
        _fake_geometry(),
        source_path="detect_runs/d1",
        source_group=source,
        frame_shape=(10, 10),
        roi_size=(4, 4),
        require_sorted_frames=True,
    )

    np.testing.assert_array_equal(result.roi_coordinates_full, [[3, 3]])
    np.testing.assert_array_equal(result.source_crop_xywh, [[3.0, 3.0, 4.0, 4.0]])
    np.testing.assert_array_equal(result.bbox_roi_xyxy, [[1.0, 1.0, 3.0, 3.0]])


def test_canonical_crop_arrays_fail_closed_on_padded_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = _source_group(root, "detect_runs/d1", center=(0.1, 0.1))
    values = _source_geometry_values(
        center=(1.0, 1.0),
        bbox_norm_center=(0.1, 0.1),
    )
    monkeypatch.setattr(
        crop_module,
        "detection_observation_geometry_values",
        lambda _source: values,
    )

    with pytest.raises(
        crop_module.OrdinaryCropCoordinateError,
        match="Padded crops need an explicit padding lineage",
    ):
        crop_module._canonical_crop_arrays(
            _fake_geometry(),
            source_path="detect_runs/d1",
            source_group=source,
            frame_shape=(10, 10),
            roi_size=(4, 4),
            require_sorted_frames=True,
        )


def test_canonical_crop_arrays_reject_coordinate_relabeling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = _source_group(root, "detect_runs/d1")
    values = _source_geometry_values(center=(6.0, 5.0))
    monkeypatch.setattr(
        crop_module,
        "detection_observation_geometry_values",
        lambda _source: values,
    )

    with pytest.raises(
        crop_module.OrdinaryCropCoordinateError,
        match="refusing to change crop pixel numerics",
    ):
        crop_module._canonical_crop_arrays(
            _fake_geometry(),
            source_path="detect_runs/d1",
            source_group=source,
            frame_shape=(10, 10),
            roi_size=(4, 4),
            require_sorted_frames=True,
        )


def test_materialized_preflight_uses_exact_acquisition_extent_not_root_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = _source_group(root, "detect_runs/d1")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((1, 10, 10), dtype=np.uint8))
    root.attrs.update({"width": 999, "height": 777})
    _install_fake_acquisition_authority(
        monkeypatch,
        mode=crop_module.MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    )

    result = crop_module._preflight_ordinary_crop_coordinates(
        root,
        zarr_path="unused.zarr",
        source_path="detect_runs/d1",
        source_group=source,
        video_source_type="zarr",
        video_path=None,
        roi_size=(4, 4),
    )

    assert result.frame_shape == (10, 10)
    assert result.total_frames == 1
    np.testing.assert_array_equal(result.roi_coordinates_full, [[3, 3]])


def test_materialized_preflight_rejects_frame_source_extent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = _source_group(root, "detect_runs/d1")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((1, 10, 11), dtype=np.uint8))
    _install_fake_acquisition_authority(
        monkeypatch,
        mode=crop_module.MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    )

    with pytest.raises(
        crop_module.OrdinaryCropCoordinateError,
        match="frame source shape differs",
    ):
        crop_module._preflight_ordinary_crop_coordinates(
            root,
            zarr_path="unused.zarr",
            source_path="detect_runs/d1",
            source_group=source,
            video_source_type="zarr",
            video_path=None,
            roi_size=(4, 4),
        )


def test_external_preflight_binds_exact_locator_and_live_fingerprint(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"canonical-video-fixture")
    metadata: dict[str, Any] = {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "locator": {"kind": "absolute", "path": str(video_path.resolve())},
        "source_path": str(video_path.resolve()),
        "codec": "fixture",
        "pix_fmt": "gray",
        "width": 10,
        "height": 10,
        "fps": 100.0,
        "total_frames": 1,
    }
    live = crop_module.source_stat_fingerprint_attrs(
        video_path,
        attr_prefix="source_video",
        extra={
            "codec": metadata["codec"],
            "pix_fmt": metadata["pix_fmt"],
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["total_frames"],
        },
    )
    metadata["file_fingerprint"] = {
        "strategy": live["source_video_fingerprint_strategy"],
        "value": live["source_video_fingerprint"],
        "size_bytes": live["source_video_size_bytes"],
        "mtime_ns": live["source_video_mtime_ns"],
        "relocation_stable": False,
    }
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    root.attrs["source_video_metadata"] = metadata
    source = _source_group(root, "detect_runs/d1")
    _install_fake_acquisition_authority(
        monkeypatch,
        mode=crop_module.EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        source_video_metadata=metadata,
    )

    result = crop_module._preflight_ordinary_crop_coordinates(
        root,
        zarr_path=tmp_path / "analysis.zarr",
        source_path="detect_runs/d1",
        source_group=source,
        video_source_type="external",
        video_path=str(video_path),
        roi_size=(4, 4),
    )

    assert result.acquisition_mode == crop_module.EXTERNAL_ACQUISITION_AUTHORITY_MODE


def test_refined_source_is_rejected_before_acquisition_or_run_creation() -> None:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    source = _source_group(root, "refined_detect_runs/r1")

    with pytest.raises(
        crop_module.OrdinaryCropCoordinateError,
        match=r"only exact detect_runs/<run>",
    ):
        crop_module._preflight_ordinary_crop_coordinates(
            root,
            zarr_path="unused.zarr",
            source_path="refined_detect_runs/r1",
            source_group=source,
            video_source_type="zarr",
            video_path=None,
            roi_size=(4, 4),
        )

    assert "crop_runs" not in root


def test_ambiguous_zero_rows_fail_strict_preflight_before_crop_run_creation(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "zero.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    source = root.create_group("detect_runs/d1")
    source.create_array("frame_indices", data=np.asarray([], dtype=np.int64))
    source.create_array(
        "bbox_norm_coords",
        data=np.empty((0, 4), dtype=np.float64),
        chunks=(1, 4),
    )
    monkeypatch.setattr(
        crop_module,
        "get_detection_source_info",
        lambda **_kwargs: ("detect_runs/d1", source, None, "detect"),
    )
    monkeypatch.setattr(
        crop_module,
        "get_video_source",
        lambda *_args, **_kwargs: ("zarr", None),
    )
    monkeypatch.setattr(crop_module, "_emit_crop_step_status", lambda **_kwargs: None)

    with pytest.raises(Exception, match="acquisition|authority|publication"):
        crop_module.crop_detections(
            str(zarr_path),
            {"crop": {"roi_sz": [4, 4]}},
            crop_storage_mode="materialized",
        )

    assert "crop_runs" not in root


def test_validated_canonical_zero_rows_create_no_crop_run_or_pending_pointer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "canonical-zero.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    source = root.create_group("detect_runs/d1")
    monkeypatch.setattr(
        crop_module,
        "get_detection_source_info",
        lambda **_kwargs: ("detect_runs/d1", source, None, "detect"),
    )
    monkeypatch.setattr(
        crop_module,
        "get_video_source",
        lambda *_args, **_kwargs: ("zarr", None),
    )
    monkeypatch.setattr(crop_module, "_emit_crop_step_status", lambda **_kwargs: None)
    calls: list[str] = []

    def validated_empty(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["source_path"])
        return crop_module._CanonicalCropPreflight(
            source_path="detect_runs/d1",
            source_group=source,
            source_geometry=object(),
            source_values={},
            frame_indices=np.empty((0,), dtype=np.int64),
            bbox_norm_coords=np.empty((0, 4), dtype=np.float64),
            roi_coordinates_full=np.empty((0, 2), dtype=np.int32),
            source_crop_xywh=np.empty((0, 4), dtype=np.float64),
            bbox_roi_xyxy=np.empty((0, 4), dtype=np.float64),
            frame_shape=(10, 10),
            total_frames=1,
            acquisition_mode=crop_module.MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
        )

    monkeypatch.setattr(
        crop_module,
        "_preflight_ordinary_crop_coordinates",
        validated_empty,
    )

    result = crop_module.crop_detections(
        str(zarr_path),
        {"crop": {"roi_sz": [4, 4]}},
        crop_storage_mode="materialized",
    )

    assert calls == ["detect_runs/d1"]
    assert result["total_crops"] == 0
    assert "crop_runs" not in root


def test_direct_external_zero_rows_run_canonical_preflight_before_return(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "external-zero.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    source = root.create_group("detect_runs/d1")
    calls: list[str] = []

    def validated_empty(*_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["video_source_type"])
        return SimpleNamespace(row_count=0)

    monkeypatch.setattr(
        crop_module,
        "_preflight_ordinary_crop_coordinates",
        validated_empty,
    )
    console = SimpleNamespace(print=lambda *_args, **_kwargs: None)

    result = crop_module.crop_from_external_video(
        zarr_path=str(zarr_path),
        video_path=str(tmp_path / "source.mp4"),
        source_path="detect_runs/d1",
        source_group=source,
        detection_source=None,
        source_type="detect",
        roi_sz=(4, 4),
        use_gpu=False,
        console=console,
    )

    assert calls == ["external"]
    assert result == {"total_crops": 0}
    assert "crop_runs" not in root


def test_geometry_only_fails_before_normal_run_creation(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "geometry.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    source = _source_group(root, "detect_runs/d1")
    monkeypatch.setattr(
        crop_module,
        "get_detection_source_info",
        lambda **_kwargs: ("detect_runs/d1", source, None, "detect"),
    )
    monkeypatch.setattr(
        crop_module,
        "get_video_source",
        lambda *_args, **_kwargs: ("zarr", None),
    )

    with pytest.raises(
        crop_module.OrdinaryCropCoordinateError,
        match="does not publish geometry-only runs",
    ):
        crop_module.crop_detections(
            str(zarr_path),
            {"crop": {"roi_sz": [4, 4]}},
            crop_storage_mode="geometry_only",
        )

    assert "crop_runs" not in root


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_outer_baseexception_marks_new_run_failed_and_restores_every_selector(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    zarr_path = tmp_path / "rollback.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("crop_runs")
    parent.create_group("previous")
    expected = {
        "latest": "previous",
        "latest_complete": "previous",
        "latest_materialized": "previous",
        "latest_any": "previous",
        "authoritative_run": "previous",
        "authoritative_run_provenance": {"source": "test"},
        "publication_generation": 7,
        "publication_policy": "test_policy",
    }
    parent.attrs.update(expected)
    root.attrs["current_crop_group_path"] = "preexisting/worker/target"

    def fail_after_creation(**kwargs: Any) -> dict[str, Any]:
        writer_root = zarr.open_group(kwargs["zarr_path"], mode="a")
        writer_parent = writer_root["crop_runs"]
        run = writer_parent.create_group("new_failed")
        run.attrs.update({"status": "running", "stage_selector_eligible": True})
        for name in expected:
            writer_parent.attrs[name] = "corrupt"
        writer_parent.attrs["latest_pending"] = "new_failed"
        writer_root.attrs["current_crop_group_path"] = "crop_runs/new_failed"
        raise error_type("injected writer failure")

    monkeypatch.setattr(crop_module, "_crop_detections_impl", fail_after_creation)

    with pytest.raises(error_type, match="injected writer failure"):
        crop_module.crop_detections(str(zarr_path), {})

    reopened = zarr.open_group(str(zarr_path), mode="r")
    failed = reopened["crop_runs/new_failed"]
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert reopened.attrs["current_crop_group_path"] == "preexisting/worker/target"
    for name, value in expected.items():
        assert reopened["crop_runs"].attrs[name] == value
    assert "latest_pending" not in reopened["crop_runs"].attrs


def test_distributed_cleanup_cancels_waits_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    futures = [object(), object()]

    class Client:
        def cancel(self, values: list[Any], *, force: bool) -> None:
            events.append(("cancel", values, force))

        def close(self) -> None:
            events.append(("close",))

    monkeypatch.setattr(
        crop_module,
        "distributed_wait",
        lambda values: events.append(("wait", values)),
    )

    crop_module._cancel_wait_close_distributed_crop(Client(), futures)

    assert [event[0] for event in events] == ["cancel", "wait", "close"]


def _publication_fixture() -> tuple[Any, Any, Any, Any]:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    parent = root.create_group("crop_runs")
    previous = parent.create_group("previous")
    previous.attrs["crop_storage_mode"] = "materialized"
    parent.attrs.update(
        {
            "latest": "previous",
            "latest_complete": "previous",
            "latest_materialized": "previous",
            "latest_any": "previous",
        }
    )
    selector_snapshot = crop_module._snapshot_crop_selector_attrs(parent)
    run = parent.create_group("new")
    run.attrs.update(
        {
            "stage_selector_eligible": False,
            "crop_storage_mode": "materialized",
        }
    )
    mark_run_started(run, run_name="new", stage="crop")
    run.create_array("instance_key", data=np.asarray([11], dtype=np.uint64))
    run.create_array("detection_indices", data=np.asarray([0], dtype=np.int64))
    run.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([0], dtype=np.int64),
    )
    run.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
    )
    run.create_array(
        "bbox_img_xyxy",
        data=np.asarray([[4.0, 4.0, 6.0, 6.0]], dtype=np.float64),
    )
    run.create_array(
        "centers_img_xy",
        data=np.asarray([[5.0, 5.0]], dtype=np.float64),
    )
    run.create_array(
        "source_crop_xywh",
        data=np.asarray([[3.0, 3.0, 4.0, 4.0]], dtype=np.float64),
    )
    run.create_array(
        "bbox_roi_xyxy",
        data=np.asarray([[1.0, 1.0, 3.0, 3.0]], dtype=np.float64),
    )
    run.create_array("roi_coordinates_full", data=np.asarray([[3, 3]], dtype=np.int32))
    run.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    geometry = SimpleNamespace(
        frame_evidence=SimpleNamespace(source_camera_frame=object()),
    )
    preflight = crop_module._CanonicalCropPreflight(
        source_path="detect_runs/d1",
        source_group=object(),
        source_geometry=geometry,
        source_values={},
        frame_indices=np.asarray([0], dtype=np.int64),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
        roi_coordinates_full=np.asarray([[3, 3]], dtype=np.int32),
        source_crop_xywh=np.asarray([[3.0, 3.0, 4.0, 4.0]], dtype=np.float64),
        bbox_roi_xyxy=np.asarray([[1.0, 1.0, 3.0, 3.0]], dtype=np.float64),
        frame_shape=(10, 10),
        total_frames=1,
        acquisition_mode="materialized_source_frames_v1",
    )
    return root, parent, run, (preflight, selector_snapshot)


_TEST_RUN_PROVENANCE = {
    "git_sha": "a" * 40,
    "config_hash": "b" * 64,
    "params": {},
    "input_run_ids": {"detect": "d1"},
    "command": "test ordinary crop publication",
    "fisheye_version": "test",
}


def _stub_coordinate_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    crop_geometry = SimpleNamespace(row_identity=object())
    monkeypatch.setattr(
        crop_module,
        "publish_crop_observation_geometry",
        lambda *_args, **_kwargs: crop_geometry,
    )
    monkeypatch.setattr(crop_module, "stamp_crop_placement_ownership", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(crop_module, "bind_array_reference_extent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(crop_module, "stamp_roi_pixel_frame_authority", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        crop_module,
        "stamp_crop_placement_transform_authority",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(crop_module, "stamp_directed_transform_v2", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        crop_module,
        "resolve_bound_directed_transform_chain",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(crop_module, "publish_crop_roi_geometry", lambda *_args, **_kwargs: object())


def test_coordinate_completion_validates_before_final_eligibility_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, context = _publication_fixture()
    preflight, selector_snapshot = context
    _stub_coordinate_publication(monkeypatch)
    observed: list[str] = []

    def validate(_root: Any, path: str, **kwargs: Any) -> object:
        assert path == "crop_runs/new"
        assert run.attrs["coordinate_contract"] == "canonical_v2"
        assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
        assert kwargs.get("require_selector_eligible", True) is False
        assert run.attrs["stage_selector_eligible"] is False
        assert parent.attrs["latest"] == "previous"
        assert parent.attrs["latest_complete"] == "previous"
        observed.append("staged")
        return object()

    monkeypatch.setattr(
        crop_module,
        "_load_persisted_ordinary_crop_observation_geometry",
        validate,
    )

    crop_module._publish_ordinary_crop_coordinate_contract(
        root,
        run,
        crop_parent=parent,
        run_name="new",
        preflight=preflight,
        run_provenance=_TEST_RUN_PROVENANCE,
        selector_snapshot=selector_snapshot,
    )

    assert observed == ["staged"]
    assert run.attrs["stage_selector_eligible"] is True
    assert parent.attrs["latest"] == "new"
    assert parent.attrs["latest_complete"] == "new"
    assert parent.attrs["latest_materialized"] == "new"
    assert parent.attrs["latest_any"] == "new"


def test_eligibility_flip_is_literal_final_publication_try_action() -> None:
    source = textwrap.dedent(
        inspect.getsource(crop_module._publish_ordinary_crop_coordinate_contract)
    )
    function = ast.parse(source).body[0]
    transaction = next(
        statement for statement in function.body if isinstance(statement, ast.Try)
    )
    final_action = transaction.body[-1]

    assert isinstance(final_action, ast.Assign)
    target = final_action.targets[0]
    assert isinstance(target, ast.Subscript)
    assert ast.unparse(target.value) == "crop_group.attrs"
    assert ast.literal_eval(target.slice) == "stage_selector_eligible"
    assert isinstance(final_action.value, ast.Constant)
    assert final_action.value.value is True


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_staged_coordinate_validation_restores_attrs_and_selectors(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    root, parent, run, context = _publication_fixture()
    preflight, selector_snapshot = context
    _stub_coordinate_publication(monkeypatch)
    run_attrs_before = dict(run.attrs)
    validations = 0

    def fail_staged(*_args: Any, **kwargs: Any) -> object:
        nonlocal validations
        validations += 1
        assert kwargs.get("require_selector_eligible", True) is False
        assert run.attrs["stage_selector_eligible"] is False
        raise error_type("injected staged validation failure")

    monkeypatch.setattr(
        crop_module,
        "_load_persisted_ordinary_crop_observation_geometry",
        fail_staged,
    )

    with pytest.raises(error_type, match="injected staged validation failure"):
        crop_module._publish_ordinary_crop_coordinate_contract(
            root,
            run,
            crop_parent=parent,
            run_name="new",
            preflight=preflight,
            run_provenance=_TEST_RUN_PROVENANCE,
            selector_snapshot=selector_snapshot,
        )

    assert dict(run.attrs) == run_attrs_before
    assert parent.attrs["latest"] == "previous"
    assert parent.attrs["latest_complete"] == "previous"
    assert parent.attrs["latest_materialized"] == "previous"
    assert parent.attrs["latest_any"] == "previous"
    assert validations == 1


def test_lsf_flat_cache_binds_exact_validated_materialized_crop_run() -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "submit_crop_flat_roi_cache_bsub.sh"
    ).read_text(encoding="utf-8")

    assert "latest_any" not in script
    assert "latest_materialized" not in script
    assert "load_persisted_ordinary_crop_observation_geometry" in script
    assert "palette.crop_batch_result.v1" in script
    assert '--result-json "\\$CROP_RESULT_JSON"' in script
    assert '"crop_run": crop_run' in script
    assert '--crop-run "\\${CROP_RUN}"' in script
    assert 'bash "$CACHE_SCRIPT" "$crop_jobid"' in script


def test_local_crop_writer_cannot_outlive_rollback_boundary() -> None:
    source = Path(crop_module.__file__).read_text(encoding="utf-8")

    assert 'd.compute(scheduler="synchronous")' in source
