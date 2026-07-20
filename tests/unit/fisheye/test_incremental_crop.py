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

import fisheye.tracking.incremental_crop as incremental_crop_module
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
    build_acquisition_authority_publication_status,
)
from fisheye.shared.keyed_delta import ACTION_CODE_MAP, REASON_CODE_MAP
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
)
from fisheye.tracking.incremental_crop import (
    IncrementalCropError,
    materialize_incremental_crop_run,
    resolve_incremental_crop_coordinate_context,
)
from fisheye.utils.materialize_incremental_crop import (
    plan_or_materialize_incremental_crop,
)


PROVENANCE = {
    "schema": "palette.run_provenance.v1",
    "git_sha": "0" * 40,
    "config_hash": "1" * 64,
    "params": {"test": True},
    "input_run_ids": {"source": "test"},
    "input_artifacts": [],
    "command": "pytest incremental crop",
    "fisheye_version": None,
}


def _root() -> Any:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    raw = root.create_group("raw_video")
    frames = np.stack(
        [
            np.arange(36, dtype=np.uint8).reshape(6, 6),
            np.arange(36, dtype=np.uint8).reshape(6, 6) + 50,
            np.arange(36, dtype=np.uint8).reshape(6, 6) + 100,
        ]
    )
    raw.create_array("images_full", data=frames, chunks=(1, 6, 6))
    return root


def _source(
    root: Any,
    name: str,
    *,
    keys: list[int],
    frames: list[int],
    boxes: list[list[float]],
) -> Any:
    group = root.require_group("refined_detect_runs").create_group(name)
    group.create_array("instance_key", data=np.asarray(keys, dtype=np.uint64))
    group.create_array("frame_indices", data=np.asarray(frames, dtype=np.int64))
    group.create_array("bbox_norm_coords", data=np.asarray(boxes, dtype=np.float32))
    group.attrs["edit_revision"] = 1
    return group


def _run(
    root: Any,
    source: Any,
    *,
    source_path: str,
    run_name: str,
    base_run_name: str | None = None,
    before_publish: Any = None,
    source_pixel_fingerprint: str = "test-video-sha256",
    frame_source: Any = None,
    publication_failure: bool = False,
    completion_failure: bool = False,
    final_validation_failure: bool = False,
) -> Any:
    def fake_resolve(
        _root: Any,
        *,
        source_group: Any,
        frame_source: Any,
        source_pixel_fingerprint: str,
        coordinate_contract_mode: str,
        **_kwargs: Any,
    ) -> Any:
        assert coordinate_contract_mode == "canonical"
        return incremental_crop_module.ResolvedIncrementalCropCoordinateContext(
            coordinate_contract_mode="canonical",
            source_group=source_group,
            frame_source=frame_source,
            source_geometry=source_group,
            source_pixel_fingerprint=source_pixel_fingerprint,
            authority_signature=("test-coordinate-authority",),
        )

    def fake_top_left(source_geometry: Any, *, roi_size: tuple[int, int]) -> np.ndarray:
        return incremental_crop_module._roi_top_left(  # noqa: SLF001
            np.asarray(source_geometry["bbox_norm_coords"][:]),
            frame_shape=(6, 6),
            roi_size=roi_size,
        )

    def fake_coordinate_arrays(
        snapshot: Any,
        *,
        frame_shape: tuple[int, int],
        roi_size: tuple[int, int],
        roi_coordinates_full: np.ndarray,
        source_geometry: Any,
    ) -> dict[str, np.ndarray]:
        boxes = np.asarray(source_geometry["bbox_norm_coords"][:], dtype=np.float64)
        height, width = frame_shape
        cx, cy, box_width, box_height = boxes.T
        bbox_img = np.column_stack(
            (
                (cx - box_width / 2.0) * width,
                (cy - box_height / 2.0) * height,
                (cx + box_width / 2.0) * width,
                (cy + box_height / 2.0) * height,
            )
        )
        centers = np.column_stack(
            (
                (bbox_img[:, 0] + bbox_img[:, 2]) / 2.0,
                (bbox_img[:, 1] + bbox_img[:, 3]) / 2.0,
            )
        )
        roi_height, roi_width = roi_size
        placement = np.column_stack(
            (
                roi_coordinates_full[:, 0],
                roi_coordinates_full[:, 1],
                np.full(snapshot.row_count, roi_width),
                np.full(snapshot.row_count, roi_height),
            )
        ).astype(np.float64)
        offsets = np.column_stack(
            (placement[:, 0], placement[:, 1], placement[:, 0], placement[:, 1])
        )
        return {
            "source_acquisition_frame_index": np.asarray(
                snapshot.frame_indices,
                dtype=np.int64,
            ),
            "bbox_img_xyxy": bbox_img,
            "centers_img_xy": centers,
            "source_crop_xywh": placement,
            "bbox_roi_xyxy": bbox_img - offsets,
        }

    def fake_publish(
        run_group: Any,
        _roi_images: Any,
        _bbox_edge_frame_node: Any,
        **_kwargs: Any,
    ) -> None:
        assert run_group.attrs["stage_selector_eligible"] is False
        run_group.attrs["coordinate_contract"] = "canonical_v2"
        run_group["bbox_img_xyxy"].attrs["coordinate_descriptor"] = {
            "published": True
        }
        if publication_failure:
            raise RuntimeError("injected coordinate publication failure")

    def fake_final_load(root_node: Any, rowset_path: str, **kwargs: Any) -> object:
        final_group = root_node[rowset_path]
        assert final_group.attrs["stage_selector_eligible"] is False
        assert final_group.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
        assert kwargs == {"require_selector_eligible": False}
        if final_validation_failure:
            raise RuntimeError("injected final pointer validation failure")
        return object()

    def fake_complete(run_group: Any, **_kwargs: Any) -> None:
        assert run_group.attrs["stage_selector_eligible"] is False
        raise RuntimeError("injected completion failure")

    originals = (
        incremental_crop_module.resolve_incremental_crop_coordinate_context,
        incremental_crop_module._canonical_roi_top_left,
        incremental_crop_module._canonical_crop_coordinate_arrays,
        incremental_crop_module._publish_canonical_crop_coordinate_contract,
        incremental_crop_module._load_persisted_ordinary_crop_observation_geometry,
        incremental_crop_module.mark_run_complete,
    )
    incremental_crop_module.resolve_incremental_crop_coordinate_context = fake_resolve
    incremental_crop_module._canonical_roi_top_left = fake_top_left
    incremental_crop_module._canonical_crop_coordinate_arrays = fake_coordinate_arrays
    incremental_crop_module._publish_canonical_crop_coordinate_contract = fake_publish
    incremental_crop_module._load_persisted_ordinary_crop_observation_geometry = (
        fake_final_load
    )
    if completion_failure:
        incremental_crop_module.mark_run_complete = fake_complete
    try:
        return materialize_incremental_crop_run(
            root,
            source_group=source,
            source_path=source_path,
            frame_source=(
                root["raw_video/images_full"] if frame_source is None else frame_source
            ),
            source_pixel_fingerprint=source_pixel_fingerprint,
            roi_size=(4, 4),
            run_name=run_name,
            run_provenance=PROVENANCE,
            base_run_name=base_run_name,
            roi_chunk_rows=2,
            signature_batch_rows=2,
            before_publish=before_publish,
        )
    finally:
        (
            incremental_crop_module.resolve_incremental_crop_coordinate_context,
            incremental_crop_module._canonical_roi_top_left,
            incremental_crop_module._canonical_crop_coordinate_arrays,
            incremental_crop_module._publish_canonical_crop_coordinate_contract,
            incremental_crop_module._load_persisted_ordinary_crop_observation_geometry,
            incremental_crop_module.mark_run_complete,
        ) = originals


def test_initial_crop_materialization_publishes_complete_exact_payload() -> None:
    root = _root()
    source = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[
            [0.5, 0.5, 0.2, 0.2],
            [0.0, 0.0, 0.2, 0.2],
        ],
    )

    result = _run(
        root,
        source,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )

    crop = root["crop_runs/crop_a"]
    assert result.computed_rows == 2
    assert result.copied_rows == 0
    assert result.source_frame_bytes_read == 72
    assert crop.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert crop.attrs["coordinate_contract_mode"] == "canonical"
    assert crop.attrs["coordinate_contract"] == "canonical_v2"
    assert crop.attrs["stage_selector_eligible"] is True
    assert root["crop_runs"].attrs["latest"] == "crop_a"
    assert root["crop_runs"].attrs["latest_materialized"] == "crop_a"
    np.testing.assert_array_equal(crop["instance_key"][:], [11, 22])
    np.testing.assert_array_equal(crop["roi_coordinates_full"][:], [[1, 1], [-2, -2]])
    expected_center = np.arange(36, dtype=np.uint8).reshape(6, 6)[1:5, 1:5]
    np.testing.assert_array_equal(crop["roi_images"][0], expected_center)
    expected_padded = np.zeros((4, 4), dtype=np.uint8)
    expected_padded[2:4, 2:4] = (
        np.arange(36, dtype=np.uint8).reshape(6, 6)[:2, :2] + 50
    )
    np.testing.assert_array_equal(crop["roi_images"][1], expected_padded)
    assert crop.attrs["materialization_summary"]["action_counts"]["compute"] == 2
    assert crop["materialization_plan"].attrs["schema_version"] == 1


def test_crop_selector_eligibility_flip_is_the_last_persistent_action() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(materialize_incremental_crop_run))
    )
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    flips = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "stage_selector_eligible"
            for target in node.targets
        )
    ]
    assert len(flips) == 1
    flip_line = flips[0].lineno
    publication_try = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        and any(
            candidate is flips[0]
            for statement in node.body
            for candidate in ast.walk(statement)
        )
    )
    normal_path_nodes = [
        node
        for statement in publication_try.body
        for node in ast.walk(statement)
    ]
    later_side_effects = [
        node
        for node in normal_path_nodes
        if getattr(node, "lineno", 0) > flip_line
        and isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete, ast.Call))
    ]
    assert later_side_effects == []
    rollback_calls = [
        node
        for handler in publication_try.handlers
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_rollback_failed_crop_publication"
    ]
    assert len(rollback_calls) == 1


def test_normal_crop_writer_fails_closed_instead_of_trusting_canonical_marker() -> None:
    root = _root()
    source = _source(
        root,
        "forged",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    source.attrs["coordinate_contract"] = "canonical_v2"

    with pytest.raises(IncrementalCropError, match="mirrored acquisition"):
        materialize_incremental_crop_run(
            root,
            source_group=source,
            source_path="refined_detect_runs/forged",
            frame_source=root["raw_video/images_full"],
            source_pixel_fingerprint="unsealed-caller-claim",
            roi_size=(4, 4),
            run_name="must_not_publish",
            run_provenance=PROVENANCE,
        )

    assert "crop_runs" not in root
    with pytest.raises(IncrementalCropError, match="canonical-only"):
        materialize_incremental_crop_run(
            root,
            source_group=source,
            source_path="refined_detect_runs/forged",
            frame_source=root["raw_video/images_full"],
            source_pixel_fingerprint="historical-fingerprint",
            roi_size=(4, 4),
            run_name="must_not_publish_legacy",
            run_provenance=PROVENANCE,
            coordinate_contract_mode="legacy_noncanonical",
        )


def test_canonical_crop_context_requires_exact_materialized_frame_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    raw = root["raw_video"]
    raw.create_array(
        "images_ds",
        data=np.asarray(raw["images_full"][:]),
        chunks=(1, 6, 6),
    )
    source = root.require_group("detect_runs").create_group("canonical")
    external_status = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
        authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-01",
    ).to_dict()
    root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = external_status
    raw.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = external_status
    with pytest.raises(IncrementalCropError, match="requires materialized"):
        resolve_incremental_crop_coordinate_context(
            root,
            source_group=source,
            source_path="detect_runs/canonical",
            frame_source=raw["images_full"],
            source_pixel_fingerprint="a" * 64,
            coordinate_contract_mode="canonical",
        )

    status = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
        authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/camera-01",
    ).to_dict()
    root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = status
    raw.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR] = status
    with pytest.raises(IncrementalCropError, match="Unsupported coordinate_contract_mode"):
        resolve_incremental_crop_coordinate_context(
            root,
            source_group=source,
            source_path="detect_runs/canonical",
            frame_source=raw["images_full"],
            source_pixel_fingerprint="a" * 64,
            coordinate_contract_mode="legacy_noncanonical",
        )

    materialization_id = "a" * 64

    class _Verified:
        def __init__(self, *, record: Any, record_ref: str, digest: str) -> None:
            self.record = record
            self.record_ref = record_ref
            self.record_sha256 = digest

        def assert_verified(self) -> None:
            return None

    ownership = _Verified(
        record=SimpleNamespace(
            mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
            import_operation={
                "materialization_manifest": {
                    "materialization_id": materialization_id,
                }
            },
        ),
        record_ref="/analysis/acquisition_camera_frames/camera-01@ownership",
        digest="1" * 64,
    )
    acquisition = _Verified(
        record=SimpleNamespace(camera_id="camera-01"),
        record_ref="/analysis/acquisition_camera_frames/camera-01@frame",
        digest="2" * 64,
    )

    def _record(name: str) -> Any:
        return SimpleNamespace(record_ref=f"/{name}", record_sha256="3" * 64)

    def _descriptor(name: str) -> Any:
        return SimpleNamespace(descriptor=SimpleNamespace(digest=lambda: name))

    geometry = SimpleNamespace(
        row_identity=_record("row_identity"),
        bbox_projection=_record("bbox_projection"),
        bbox_center_derivation=_record("bbox_center"),
        bbox_normalized=_descriptor("bbox_normalized"),
        bbox_image=_descriptor("bbox_image"),
        centers_image=_descriptor("centers_image"),
        assert_verified=lambda: None,
    )
    monkeypatch.setattr(
        incremental_crop_module,
        "load_persisted_acquisition_camera_authority",
        lambda _root: (ownership, acquisition),
    )
    monkeypatch.setattr(
        incremental_crop_module,
        "load_persisted_detection_observation_geometry",
        lambda _root, _path: geometry,
    )
    root["detect_runs"].attrs["latest_complete"] = "canonical"

    with pytest.raises(IncrementalCropError, match="exact root-owned node"):
        resolve_incremental_crop_coordinate_context(
            root,
            source_group=source,
            source_path="detect_runs/canonical",
            frame_source=raw["images_ds"],
            source_pixel_fingerprint=materialization_id,
            coordinate_contract_mode="canonical",
        )
    with pytest.raises(IncrementalCropError, match="materialization id"):
        resolve_incremental_crop_coordinate_context(
            root,
            source_group=source,
            source_path="detect_runs/canonical",
            frame_source=raw["images_full"],
            source_pixel_fingerprint="caller-guessed-fingerprint",
            coordinate_contract_mode="canonical",
        )

    root["detect_runs"].attrs["latest_complete"] = "different"
    with pytest.raises(IncrementalCropError, match="not the exact selected detection"):
        resolve_incremental_crop_coordinate_context(
            root,
            source_group=source,
            source_path="detect_runs/canonical",
            frame_source=raw["images_full"],
            source_pixel_fingerprint=materialization_id,
            coordinate_contract_mode="canonical",
        )
    root["detect_runs"].attrs["latest_complete"] = "canonical"

    context = resolve_incremental_crop_coordinate_context(
        root,
        source_group=source,
        source_path="detect_runs/canonical",
        frame_source=raw["images_full"],
        source_pixel_fingerprint=materialization_id,
        coordinate_contract_mode="canonical",
    )
    assert context.source_geometry is geometry
    assert context.frame_source.path == "raw_video/images_full"
    assert context.source_pixel_fingerprint == materialization_id


def test_delta_crop_copies_unchanged_reordered_row_and_computes_only_changes() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11, 22, 33],
        frames=[0, 1, 2],
        boxes=[
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[33, 11, 44],
        frames=[2, 0, 1],
        boxes=[
            [0.25, 0.25, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2],
        ],
    )
    source_b.create_array(
        "source_refined_row_ids",
        data=np.asarray([303, 101, 404], dtype=np.int64),
    )

    result = _run(
        root,
        source_b,
        source_path="refined_detect_runs/source_b",
        run_name="crop_b",
        base_run_name="crop_a",
    )

    crop_a = root["crop_runs/crop_a"]
    crop_b = root["crop_runs/crop_b"]
    assert result.copied_rows == 1
    assert result.computed_rows == 2
    assert result.omitted_rows == 1
    assert result.source_frame_bytes_read == 72
    np.testing.assert_array_equal(crop_b["instance_key"][:], [33, 11, 44])
    np.testing.assert_array_equal(crop_b["source_refined_row_ids"][:], [303, 101, 404])
    np.testing.assert_array_equal(crop_b["roi_images"][1], crop_a["roi_images"][0])
    np.testing.assert_array_equal(
        crop_b["materialization_plan/action_codes"][:],
        [ACTION_CODE_MAP["compute"], ACTION_CODE_MAP["copy"], ACTION_CODE_MAP["compute"]],
    )
    np.testing.assert_array_equal(
        crop_b["materialization_plan/reason_codes"][:],
        [REASON_CODE_MAP["source_changed"], REASON_CODE_MAP["unchanged"], REASON_CODE_MAP["added"]],
    )
    np.testing.assert_array_equal(
        crop_b["materialization_plan/omitted_instance_key"][:],
        [22],
    )


def test_pure_reorder_uses_no_source_frame_reads() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[22, 11],
        frames=[1, 0],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )

    result = _run(
        root,
        source_b,
        source_path="refined_detect_runs/source_b",
        run_name="crop_b",
        base_run_name="crop_a",
    )

    assert result.copied_rows == 2
    assert result.computed_rows == 0
    assert result.source_frame_bytes_read == 0
    np.testing.assert_array_equal(
        root["crop_runs/crop_b/roi_images"][:],
        root["crop_runs/crop_a/roi_images"][:][::-1],
    )


def test_changed_pixel_contract_forces_full_compute_instead_of_reuse() -> None:
    root = _root()
    source = _source(
        root,
        "source_a",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )

    result = _run(
        root,
        source,
        source_path="refined_detect_runs/source_a",
        source_pixel_fingerprint="different-video-sha256",
        run_name="crop_b",
        base_run_name="crop_a",
    )

    assert result.computed_rows == 2
    assert result.copied_rows == 0
    np.testing.assert_array_equal(
        root["crop_runs/crop_b/materialization_plan/reason_codes"][:],
        [REASON_CODE_MAP["signature_spec_changed"]] * 2,
    )


def test_source_change_during_processing_fails_without_replacing_latest() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
    )

    def mutate_source() -> None:
        source_b["bbox_norm_coords"][0, 0] = np.float32(0.25)

    with pytest.raises(IncrementalCropError, match="changed during processing"):
        _run(
            root,
            source_b,
            source_path="refined_detect_runs/source_b",
            run_name="crop_failed",
            base_run_name="crop_a",
            before_publish=mutate_source,
        )

    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_a"
    assert parent.attrs["latest_complete"] == "crop_a"
    assert parent.attrs["latest_materialized"] == "crop_a"
    assert parent["crop_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize(
    "failure_kwarg",
    ["publication_failure", "completion_failure", "final_validation_failure"],
)
def test_crop_finalization_failure_rolls_back_coordinate_graph_and_pointers(
    failure_kwarg: str,
) -> None:
    root = _root()
    source = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    kwargs = {failure_kwarg: True}

    with pytest.raises(RuntimeError, match="injected"):
        _run(
            root,
            source,
            source_path="refined_detect_runs/source_a",
            run_name="crop_failed",
            **kwargs,
        )

    parent = root["crop_runs"]
    failed = parent["crop_failed"]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert "coordinate_contract" not in failed.attrs
    assert "coordinate_descriptor" not in failed["bbox_img_xyxy"].attrs
    assert parent.attrs["latest"] == "crop_a"
    assert parent.attrs["latest_complete"] == "crop_a"
    assert parent.attrs["latest_materialized"] == "crop_a"
    assert "latest_pending" not in parent.attrs


def test_compute_failure_leaves_previous_complete_run_selected() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]],
    )

    class FailingFrames:
        shape = root["raw_video/images_full"].shape
        dtype = root["raw_video/images_full"].dtype

        def __getitem__(self, index: Any) -> np.ndarray:
            raise OSError(f"injected frame read failure at {index}")

    with pytest.raises(OSError, match="injected frame read failure"):
        _run(
            root,
            source_b,
            source_path="refined_detect_runs/source_b",
            run_name="crop_failed",
            base_run_name="crop_a",
            frame_source=FailingFrames(),
        )

    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_a"
    assert parent.attrs["latest_complete"] == "crop_a"
    assert parent["crop_failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(19)])
def test_crop_interrupt_rolls_back_and_keeps_failed_run_ineligible(
    interrupt: BaseException,
) -> None:
    root = _root()
    source = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )

    def interrupt_before_publication() -> None:
        raise interrupt

    with pytest.raises(type(interrupt)):
        _run(
            root,
            source,
            source_path="refined_detect_runs/source_a",
            run_name="crop_interrupted",
            before_publish=interrupt_before_publication,
        )

    parent = root["crop_runs"]
    failed = parent["crop_interrupted"]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert failed.attrs["stage_selector_eligible"] is False
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert "latest_pending" not in parent.attrs


def test_stale_materializer_failure_restores_all_prepublication_selectors() -> None:
    root = _root()
    source_a = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )
    _run(
        root,
        source_a,
        source_path="refined_detect_runs/source_a",
        run_name="crop_a",
    )
    source_b = _source(
        root,
        "source_b",
        keys=[11, 22],
        frames=[0, 1],
        boxes=[[0.5, 0.5, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]],
    )

    def publish_newer_run() -> None:
        parent = root["crop_runs"]
        newer = parent.create_group("crop_newer")
        newer.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
        parent.attrs.update(
            {
                "latest": "crop_newer",
                "latest_complete": "crop_newer",
                "latest_materialized": "crop_newer",
                "latest_any": "crop_newer",
                "publication_generation": 2,
            }
        )

    with pytest.raises(IncrementalCropError, match="publication state changed"):
        _run(
            root,
            source_b,
            source_path="refined_detect_runs/source_b",
            run_name="crop_stale",
            base_run_name="crop_a",
            before_publish=publish_newer_run,
        )

    parent = root["crop_runs"]
    assert parent.attrs["latest"] == "crop_a"
    assert parent.attrs["latest_complete"] == "crop_a"
    assert parent.attrs["latest_materialized"] == "crop_a"
    assert parent["crop_stale"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_legacy_crop_without_source_signatures_cannot_be_reused() -> None:
    root = _root()
    parent = root.create_group("crop_runs")
    legacy = parent.create_group("legacy")
    legacy.attrs.update(
        {
            RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
            "crop_storage_mode": "materialized",
        }
    )
    legacy.create_array("instance_key", data=np.asarray([11], dtype=np.uint64))
    legacy.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    source = _source(
        root,
        "source_a",
        keys=[11],
        frames=[0],
        boxes=[[0.5, 0.5, 0.2, 0.2]],
    )

    with pytest.raises(IncrementalCropError, match="not a Phase-1"):
        _run(
            root,
            source,
            source_path="refined_detect_runs/source_a",
            run_name="crop_a",
            base_run_name="legacy",
        )

    assert "crop_a" not in parent


def test_cli_rejects_legacy_materialized_mode_without_creating_crop_parent(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.arange(36, dtype=np.uint8).reshape(1, 6, 6),
    )
    source = root.create_group("refined_detect_runs").create_group("source_a")
    source.create_array("instance_key", data=np.asarray([11], dtype=np.uint64))
    source.create_array("frame_indices", data=np.asarray([0], dtype=np.int64))
    source.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
    )

    with pytest.raises(IncrementalCropError, match="requires coordinate mode 'canonical'"):
        plan_or_materialize_incremental_crop(
            archive,
            source_rowset_path="refined_detect_runs/source_a",
            source_pixel_fingerprint="test-video-sha256",
            roi_size=(4, 4),
            output_run="crop_a",
            base_crop_run=None,
            apply=False,
            roi_chunk_rows=2,
            signature_batch_rows=2,
            tabular_shard_rows=131_072,
            command="test dry run",
            coordinate_contract_mode="legacy_noncanonical",
        )

    reopened = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    assert "crop_runs" not in reopened
