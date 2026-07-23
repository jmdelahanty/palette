from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest
import zarr

import fisheye.shared.directed_transform_v2 as directed_transform_v2_module
import fisheye.shared.pixel_frame_authority as pixel_frame_authority_module
import fisheye.shared.transform_authority as transform_authority_module
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
    stamp_and_bind_row_identity_contract,
)
from fisheye.shared.coordinate_reference import bind_array_reference_extent
from fisheye.shared.directed_transform_chain import (
    DirectedTransformChainError,
    apply_bound_directed_transform_chain,
    require_bound_directed_transform_chain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform import (
    DIRECTED_TRANSFORM_ATTR,
    DIRECTED_TRANSFORM_DIGEST_SUFFIX,
    build_directed_homography,
    directed_homography_attrs,
    directed_transform_digest,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_DIGEST_ATTR,
    DirectedTransformV2Error,
    MIGRATION_ELIGIBILITY_ATTR,
    MIGRATION_ELIGIBILITY_BASIS,
    MIGRATION_ELIGIBILITY_DIGEST_ATTR,
    MIGRATION_ELIGIBILITY_SCHEMA_ID,
    MIGRATION_ELIGIBILITY_SCHEMA_VERSION,
    apply_bound_directed_transform_v2,
    load_bound_directed_transform_v2,
    parse_directed_transform_v2,
    stamp_directed_transform_v2,
    stamp_explicit_inverse_directed_transform_v2,
    validate_migration_only_v1_v2_coexistence,
)
from fisheye.shared.model_input_transform import resolve_model_input_transform
from fisheye.shared.pixel_frame_authority import (
    ACQUISITION_CAMERA_FRAME_ATTR,
    ACQUISITION_CAMERA_FRAME_DIGEST_ATTR,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR,
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR,
    PIXEL_FRAME_AUTHORITY_ATTR,
    PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
    SCALE_XY_EDGE_ALIGNED_V1,
    SCALE_XY_PIXEL_CENTER_V1,
    PixelFrameAuthorityError,
    build_verified_acquisition_materialization,
    collect_acquisition_importer_physical_object_evidence,
    load_acquisition_camera_frame,
    load_persisted_acquisition_camera_authority,
    load_arena_relative_canvas_pixel_frame_authority,
    load_source_camera_pixel_frame_authority,
    model_input_to_roi_matrix,
    normalized_to_pixel_matrix,
    parse_acquisition_camera_frame,
    parse_pixel_frame_record,
    parse_source_video_metadata,
    require_bound_pixel_frame_authority,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_writer_materialization_manifest,
    stamp_acquisition_import_ownership,
    stamp_arena_relative_canvas_pixel_frame_authority,
    stamp_crop_placement_ownership,
    stamp_model_input_pixel_frame_authority,
    stamp_normalized_pixel_frame_authority,
    stamp_roi_pixel_frame_authority,
    stamp_selected_canvas_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.proof_verification import proof_verification_operation
from fisheye.shared.selected_calibration import (
    CANONICAL_AXES,
    CANONICAL_COORDINATE_ORIGIN,
    CANONICAL_DEST_FRAME,
    CANONICAL_IMAGE_SPACE,
    CANONICAL_SOURCE_FRAME,
    NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE,
    SOURCE_DISPLAY_DATASET_PATH,
    SOURCE_DISPLAY_GROUP_PATH,
    YAML_HOMOGRAPHY_PAYLOAD_SOURCE,
    YAML_HOMOGRAPHY_SERIALIZATION_FORMAT,
    SelectedCalibrationError,
    build_selected_camera_source_evidence_from_h5_values,
    build_selected_display_source_evidence_from_h5_values,
    build_selected_homography_source_evidence_from_h5_values,
    require_verified_selected_camera_source_evidence,
    require_verified_selected_display_source_evidence,
    require_verified_selected_homography_source_evidence,
    stamp_selected_calibration_snapshot,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_DIGEST_ATTR,
    TransformAuthorityError,
    load_bound_transform_authority,
    parse_transform_authority,
    arena_to_selected_canvas_matrix,
    stamp_arena_to_selected_canvas_transform_authority,
    stamp_crop_placement_transform_authority,
    stamp_model_input_transform_authority,
    stamp_normalized_to_pixel_transform_authority,
    stamp_selected_calibration_transform_authority,
)


CAMERA_ID = "camera-a"
SOURCE_H5_PATH = "/recording/source.h5"
_ARCHIVE_TOKEN = object()


class FakeGroup:
    def __init__(
        self,
        *,
        path: str,
        attrs: dict[str, Any] | None = None,
        archive_token: object = _ARCHIVE_TOKEN,
    ) -> None:
        self.path = path
        self.attrs = {} if attrs is None else attrs
        self._coordinate_archive_token = archive_token
        self.children: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        return self.children[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.children[name] = value


class FakeLocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root


class FakeStorePath:
    def __init__(self, root: Path, path: str) -> None:
        self.store = FakeLocalStore(root)
        self.path = path


class FakeArray(FakeGroup):
    def __init__(
        self,
        data: Any | None = None,
        *,
        path: str,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        attrs: dict[str, Any] | None = None,
        archive_token: object = _ARCHIVE_TOKEN,
        chunks: tuple[int, ...] | None = None,
        shards: tuple[int, ...] | None = None,
        store_root: Path | None = None,
    ) -> None:
        super().__init__(path=path, attrs=attrs, archive_token=archive_token)
        self.data = None if data is None else np.asarray(data).copy()
        if self.data is None:
            assert shape is not None and dtype is not None
            self.shape = shape
            self.dtype = np.dtype(dtype)
        else:
            self.shape = self.data.shape
            self.dtype = self.data.dtype
        self.chunks = chunks or tuple(max(1, int(item)) for item in self.shape)
        self.shards = shards
        if store_root is not None:
            self.store_path = FakeStorePath(store_root, path)
        physical_chunks = self.shards or self.chunks
        codecs: list[dict[str, Any]] = []
        if self.shards is not None:
            codecs.append(
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [int(item) for item in self.chunks],
                        "codecs": [],
                        "index_codecs": [],
                        "index_location": "end",
                    },
                }
            )
        self._coordinate_storage_metadata = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [int(item) for item in self.shape],
            "data_type": np.lib.format.dtype_to_descr(self.dtype),
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [int(item) for item in physical_chunks]
                },
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": codecs,
            "dimension_names": None,
        }
        self.read_count = 0

    def __getitem__(self, key: Any) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("metadata-only fake is unreadable")
        self.read_count += 1
        return self.data[key]


class HostileAttrs(dict[str, Any]):
    """Dict subclass whose hooks must never enter a coordinate write path."""

    def __init__(self, value: dict[str, Any] | None = None) -> None:
        super().__init__({} if value is None else value)
        self.update_attempts = 0

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.update_attempts += 1
        super().update(*args, **kwargs)
        raise RuntimeError("hostile attrs update hook entered")


class ClearUnrelatedOnceAttrs(dict[str, Any]):
    def __init__(self, value: dict[str, Any]) -> None:
        super().__init__(copy.deepcopy(value))
        self.armed = True

    def update(self, *args: Any, **kwargs: Any) -> None:
        incoming = dict(*args, **kwargs)
        if self.armed:
            self.armed = False
            self.pop("unrelated_keep_me", None)
        super().update(incoming)


class CoerceSchemaOnceAttrs(dict[str, Any]):
    def __init__(self, value: dict[str, Any], *, record_attr: str) -> None:
        super().__init__(copy.deepcopy(value))
        self.record_attr = record_attr
        self.armed = True

    def update(self, *args: Any, **kwargs: Any) -> None:
        incoming = copy.deepcopy(dict(*args, **kwargs))
        record = incoming.get(self.record_attr)
        if self.armed and isinstance(record, dict):
            self.armed = False
            record["schema_version"] = float(record["schema_version"])
        super().update(incoming)


def _homography_attrs(*, kind: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "source_frame": CANONICAL_SOURCE_FRAME,
        "dest_frame": CANONICAL_DEST_FRAME,
        "axes": CANONICAL_AXES,
        "coordinate_origin": CANONICAL_COORDINATE_ORIGIN,
        "image_space": CANONICAL_IMAGE_SPACE,
        "camera_id": CAMERA_ID,
        "canvas_name": "shadow",
        "homography_provenance_schema": "citrus.homography_provenance.v1",
        "homography_artifact_path": "/rig/calibration/homography.yml",
        "homography_artifact_exists": "true",
        "homography_artifact_checksum_algorithm": "fnv1a64",
        "homography_artifact_checksum_fnv1a64": "f999f671b0ebd9fd",
        "homography_artifact_size_bytes": 347,
        "homography_artifact_mtime_unix_ns": 1779901497426733358,
        "homography_payload_source": (
            NUMERIC_HOMOGRAPHY_PAYLOAD_SOURCE
            if kind == "numeric"
            else YAML_HOMOGRAPHY_PAYLOAD_SOURCE
        ),
    }
    if kind == "yaml":
        attrs["serialization_format"] = YAML_HOMOGRAPHY_SERIALIZATION_FORMAT
    return attrs


def _yaml_matrix(matrix: np.ndarray) -> bytes:
    values = ", ".join(format(float(item), ".17g") for item in matrix.ravel())
    return (
        "%YAML:1.0\n---\n"
        "homography_matrix: !!opencv-matrix\n"
        "   rows: 3\n   cols: 3\n   dt: d\n"
        f"   data: [ {values} ]\n"
    ).encode()


def _canonical_mapping_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _selected_evidence(matrix: np.ndarray):
    arena = {
        "active_camera_id": CAMERA_ID,
        "calculated_z_eff_mm": 20.0,
        "camera_calibrations": [
            {
                "camera_id": CAMERA_ID,
                "native_width_px": 100,
                "native_height_px": 80,
                "pixels_per_mm_camera": 25.0,
                "pixels_per_mm_projector": 4.0,
                "real_world_ref_mm": 10.0,
            }
        ],
    }
    camera = build_selected_camera_source_evidence_from_h5_values(
        source_h5_path=SOURCE_H5_PATH,
        arena_config_raw=json.dumps(arena, separators=(",", ":")),
        camera_group_path=f"/calibration_snapshot/{CAMERA_ID}",
        camera_group_attrs={
            "pixels_per_mm_camera": 25.0,
            "pixels_per_mm_projector": 4.0,
            "real_world_ref_mm": 10.0,
        },
        expected_camera_id=CAMERA_ID,
    )
    display_attrs = {
        "selected_output_name": "DP-3",
        "selected_output_connection_state": "connected",
        "selected_output_geometry": "120x60+0+0",
        "selected_output_transform_token": "normal",
        "selected_output_transform_raw": "normal left inverted right x axis y axis",
    }
    display = build_selected_display_source_evidence_from_h5_values(
        source_h5_path=SOURCE_H5_PATH,
        display_group_path=SOURCE_DISPLAY_GROUP_PATH,
        display_group_attrs=display_attrs,
        selected_output_dataset_path=SOURCE_DISPLAY_DATASET_PATH,
        selected_output_block_raw=(
            "DP-3 connected 120x60+0+0 "
            "(normal left inverted right x axis y axis) 0mm x 0mm\n"
            "   120x60 60.00*+\n"
        ),
    )
    prefix = f"/calibration_snapshot/{CAMERA_ID}"
    homography = build_selected_homography_source_evidence_from_h5_values(
        source_h5_path=SOURCE_H5_PATH,
        expected_camera_id=CAMERA_ID,
        numeric_dataset_path=f"{prefix}/homography_matrix",
        numeric_matrix=matrix,
        numeric_dataset_attrs=_homography_attrs(kind="numeric"),
        yaml_dataset_path=f"{prefix}/homography_matrix_yml",
        yaml_dataset_raw=_yaml_matrix(matrix),
        yaml_dataset_attrs=_homography_attrs(kind="yaml"),
    )
    return camera, display, homography


def _reference(
    path: str,
    *,
    rows: int,
    height: int,
    width: int,
    attrs: dict[str, Any] | None = None,
    archive_token: object = _ARCHIVE_TOKEN,
):
    node = FakeArray(
        path=path,
        shape=(rows, height, width),
        dtype=np.uint8,
        attrs=attrs,
        archive_token=archive_token,
    )
    return node, bind_array_reference_extent(node, units="px")


def _materialize_encoded_physical_objects(node: FakeArray) -> list[Path]:
    assert hasattr(node, "store_path")
    physical_chunks = node.shards or node.chunks
    grid_shape = tuple(
        (size + chunk - 1) // chunk
        for size, chunk in zip(node.shape, physical_chunks, strict=True)
    )
    paths: list[Path] = []
    for position, index in enumerate(np.ndindex(grid_shape)):
        relative = Path(node.path) / "c"
        for item in index:
            relative /= str(int(item))
        path = node.store_path.store.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            f"encoded-physical-object-{position}".encode("ascii")
        )
        paths.append(path)
    return paths


@proof_verification_operation
def _world(*, convention: str = "continuous", archive_token: object = _ARCHIVE_TOKEN):
    matrix = np.asarray(
        [[0.8, 0.02, 5.0], [-0.01, 0.7, 7.0], [0.0001, -0.0002, 1.0]],
        dtype="<f8",
    )
    camera_evidence, display_evidence, homography_evidence = _selected_evidence(matrix)

    root = FakeGroup(
        path="archive_root",
        attrs={
            "recording_id": "recording-1",
            "source_video_metadata": {
                "schema_id": "palette.source_video_metadata.v2",
                "layout": "single_video",
                "camera_id": CAMERA_ID,
                "source_path": "/recording/cams/camera-a.mp4",
                "width": 100,
                "height": 80,
                "total_frames": 2,
                "locator": {
                    "kind": "recording_relative",
                    "relative_path": "cams/camera-a.mp4",
                },
                "file_fingerprint": {
                    "strategy": "size_mtime_sha256_v1",
                    "value": "a" * 64,
                    "size_bytes": 1234,
                    "mtime_ns": 5678,
                    "relocation_stable": False,
                },
            },
        },
        archive_token=archive_token,
    )
    store_directory = tempfile.TemporaryDirectory(
        prefix="palette-coordinate-foundation-"
    )
    store_root = Path(store_directory.name)
    camera_array, _ = _reference(
        "raw_video/images_full",
        rows=2,
        height=80,
        width=100,
        attrs={"format": "gray"},
        archive_token=archive_token,
    )
    camera_array.store_path = FakeStorePath(store_root, camera_array.path)
    camera_indices = FakeArray(
        np.asarray([0, 1], dtype=np.int64),
        path="raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame",
        attrs={
            "source_domain": "stored_zarr_frame",
            "target_domain": "acquisition_frame",
            "semantics": "identity_map_zero_based_full_import",
        },
        archive_token=archive_token,
    )
    camera_node = FakeGroup(
        path="analysis/acquisition_camera_frames/camera-a",
        archive_token=archive_token,
    )
    raw_video = FakeGroup(path="raw_video", archive_token=archive_token)
    frame_maps = FakeGroup(
        path="raw_video/frame_domain_maps",
        archive_token=archive_token,
    )
    frame_maps["stored_zarr_frame_to_acquisition_frame"] = camera_indices
    manifests = FakeGroup(
        path="raw_video/manifests",
        archive_token=archive_token,
    )
    materialization_manifest_node = FakeGroup(
        path="raw_video/manifests/images_full_materialization",
        archive_token=archive_token,
    )
    manifests["images_full_materialization"] = materialization_manifest_node
    raw_video["images_full"] = camera_array
    raw_video["frame_domain_maps"] = frame_maps
    raw_video["manifests"] = manifests
    root["raw_video"] = raw_video
    import_operation_attrs = {
        "import_method": "standard_zarr",
        "import_stage": "full_resolution",
        "import_mode": "full",
        "source_path": "/recording/cams/camera-a.mp4",
        "decode_backend": "decord_cpu",
        "source_decode_surface": "decord_rgb_uint8",
    }
    physical_object_paths = _materialize_encoded_physical_objects(camera_array)
    physical_object_evidence = (
        collect_acquisition_importer_physical_object_evidence(camera_array)
    )
    stamp_acquisition_import_writer_materialization_manifest(
        root,
        frame_node=camera_array,
        frame_index_node=camera_indices,
        manifest_node=materialization_manifest_node,
        import_operation_attrs=import_operation_attrs,
        physical_object_evidence=physical_object_evidence,
    )
    materialization = build_verified_acquisition_materialization(
        root,
        frame_node=camera_array,
        frame_index_node=camera_indices,
        import_operation_attrs=import_operation_attrs,
    )
    acquisition_ownership = stamp_acquisition_import_ownership(
        root,
        camera_node,
        frame_node=camera_array,
        frame_index_node=camera_indices,
        materialization=materialization,
    )
    acquisition_frame = stamp_acquisition_camera_frame(
        root,
        camera_node,
        import_ownership=acquisition_ownership,
    )
    camera_frame_node = FakeGroup(
        path=f"analysis/coordinate_frames/source_camera/{CAMERA_ID}/{convention}",
        archive_token=archive_token,
    )
    camera_frame = stamp_source_camera_pixel_frame_authority(
        camera_frame_node,
        frame_id="camera_a_native",
        pixel_convention=convention,
        acquisition_frame=acquisition_frame,
    )
    analysis = FakeGroup(path="analysis", archive_token=archive_token)
    acquisition_frames = FakeGroup(
        path="analysis/acquisition_camera_frames",
        archive_token=archive_token,
    )
    acquisition_frames[CAMERA_ID] = camera_node
    analysis["acquisition_camera_frames"] = acquisition_frames
    stimulus_runs = FakeGroup(
        path="analysis/stimulus_runs", archive_token=archive_token
    )
    stimulus_run = FakeGroup(
        path="analysis/stimulus_runs/stim_1", archive_token=archive_token
    )
    calibration = FakeGroup(
        path="analysis/stimulus_runs/stim_1/calibration",
        archive_token=archive_token,
    )
    calibration_camera = FakeGroup(
        path="analysis/stimulus_runs/stim_1/calibration/camera-a",
        archive_token=archive_token,
    )
    selected_matrix = FakeArray(
        matrix,
        path="analysis/stimulus_runs/stim_1/calibration/camera-a/homography_matrix",
        archive_token=archive_token,
    )
    display_snapshot = FakeGroup(
        path="analysis/stimulus_runs/stim_1/display_snapshot",
        archive_token=archive_token,
    )
    calibration_camera["homography_matrix"] = selected_matrix
    calibration[CAMERA_ID] = calibration_camera
    stimulus_run["calibration"] = calibration
    stimulus_run["display_snapshot"] = display_snapshot
    stimulus_runs["stim_1"] = stimulus_run
    analysis["stimulus_runs"] = stimulus_runs
    root["analysis"] = analysis
    selected_snapshot = stamp_selected_calibration_snapshot(
        calibration,
        calibration_camera,
        display_snapshot,
        selected_matrix,
        root_node=root,
        stimulus_run="stim_1",
        camera_id=CAMERA_ID,
        source_camera=camera_evidence,
        source_display=display_evidence,
        source_homography=homography_evidence,
    )

    canvas_node, canvas_extent = _reference(
        "analysis/stimulus_runs/stim_1/display_canvas",
        rows=1,
        height=60,
        width=120,
        archive_token=archive_token,
    )
    canvas_frame = stamp_selected_canvas_pixel_frame_authority(
        canvas_extent,
        frame_id="stim_1_selected_canvas",
        pixel_convention=convention,
        selected_calibration_snapshot=selected_snapshot,
    )

    rowset = FakeGroup(path="analysis/crop_runs/crop_1", archive_token=archive_token)
    key_node = FakeArray(
        np.asarray([101, 202], dtype=np.uint64),
        path=f"{rowset.path}/instance_key",
        archive_token=archive_token,
    )
    identity = stamp_and_bind_row_identity_contract(
        rowset,
        key_node,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=key_node.data,
        ),
    )
    placements = FakeArray(
        np.asarray([[10.0, 20.0, 40.0, 20.0], [50.0, 5.0, 20.0, 40.0]]),
        path=f"{rowset.path}/source_crop_xywh",
        archive_token=archive_token,
    )
    crop_ownership = stamp_crop_placement_ownership(
        placements,
        row_identity=identity,
        source_camera_frame=camera_frame,
    )
    roi_node, roi_extent = _reference(
        f"{rowset.path}/roi_frames",
        rows=2,
        height=20,
        width=40,
        archive_token=archive_token,
    )
    roi_frame = stamp_roi_pixel_frame_authority(
        roi_extent,
        frame_id="crop_1_roi",
        pixel_convention=convention,
        crop_placement_ownership=crop_ownership,
    )

    preprocessing = resolve_model_input_transform(
        (20, 40), mode="pad_to_size", model_hw=(32, 64)
    )
    preprocessing_node = FakeArray(
        model_input_to_roi_matrix(preprocessing),
        path="analysis/detect_runs/detect_1/transforms/model_to_roi",
        archive_token=archive_token,
    )
    model_node, model_extent = _reference(
        "analysis/detect_runs/detect_1/model_input_frames",
        rows=2,
        height=32,
        width=64,
        archive_token=archive_token,
    )
    model_frame = stamp_model_input_pixel_frame_authority(
        model_extent,
        frame_id="detect_1_model_input",
        pixel_convention=convention,
        preprocessing_node=preprocessing_node,
        transform=preprocessing,
        roi_frame=roi_frame,
    )
    return {
        "matrix": matrix,
        "camera_evidence": camera_evidence,
        "display_evidence": display_evidence,
        "homography_evidence": homography_evidence,
        "root": root,
        "camera_node": camera_node,
        "camera_array": camera_array,
        "camera_indices": camera_indices,
        "materialization_manifest_node": materialization_manifest_node,
        "physical_object_evidence": physical_object_evidence,
        "physical_object_paths": physical_object_paths,
        "store_directory": store_directory,
        "acquisition_frame": acquisition_frame,
        "acquisition_ownership": acquisition_ownership,
        "camera_frame": camera_frame,
        "camera_frame_node": camera_frame_node,
        "canvas_node": canvas_node,
        "canvas_extent": canvas_extent,
        "canvas_frame": canvas_frame,
        "selected_snapshot": selected_snapshot,
        "selected_matrix": selected_matrix,
        "rowset": rowset,
        "key_node": key_node,
        "identity": identity,
        "placements": placements,
        "crop_ownership": crop_ownership,
        "roi_node": roi_node,
        "roi_extent": roi_extent,
        "roi_frame": roi_frame,
        "preprocessing": preprocessing,
        "preprocessing_node": preprocessing_node,
        "model_node": model_node,
        "model_extent": model_extent,
        "model_frame": model_frame,
        "archive_token": archive_token,
    }


def _model_link(world):
    authority_node = FakeGroup(
        path="analysis/detect_runs/detect_1/model_input_preprocessing",
        archive_token=world["archive_token"],
    )
    authority = stamp_model_input_transform_authority(
        authority_node,
        authority_id="detect_1_model_to_roi",
        matrix_node=world["preprocessing_node"],
        source_frame=world["model_frame"],
        target_frame=world["roi_frame"],
    )
    link = stamp_directed_transform_v2(
        world["preprocessing_node"],
        transform_id="model_to_roi",
        authority=authority,
        source_frame=world["model_frame"],
        target_frame=world["roi_frame"],
    )
    return authority_node, authority, link


def _crop_link(world):
    authority = stamp_crop_placement_transform_authority(
        world["placements"],
        authority_id="crop_1_roi_to_camera",
        source_frame=world["roi_frame"],
        target_frame=world["camera_frame"],
    )
    link = stamp_directed_transform_v2(
        world["placements"],
        transform_id="roi_to_camera",
        authority=authority,
        source_frame=world["roi_frame"],
        target_frame=world["camera_frame"],
        row_identity=world["identity"],
    )
    return authority, link


def _selected_link(world):
    matrix_node = FakeArray(
        world["matrix"],
        path="analysis/stimulus_runs/stim_1/calibration/camera-a/camera_to_canvas_v2",
        archive_token=world["archive_token"],
    )
    authority_node = FakeGroup(
        path="analysis/stimulus_runs/stim_1/calibration/camera-a/authority",
        archive_token=world["archive_token"],
    )
    authority = stamp_selected_calibration_transform_authority(
        authority_node,
        authority_id="stim_1_camera_a_calibration",
        source_matrix_node=matrix_node,
        source_frame=world["camera_frame"],
        target_frame=world["canvas_frame"],
        selected_calibration_snapshot=world["selected_snapshot"],
    )
    forward = stamp_directed_transform_v2(
        matrix_node,
        transform_id="camera_to_canvas",
        authority=authority,
        source_frame=world["camera_frame"],
        target_frame=world["canvas_frame"],
    )
    inverse_node = FakeArray(
        np.linalg.inv(world["matrix"]),
        path="analysis/stimulus_runs/stim_1/calibration/camera-a/canvas_to_camera_v2",
        archive_token=world["archive_token"],
    )
    inverse = stamp_explicit_inverse_directed_transform_v2(
        inverse_node,
        transform_id="canvas_to_camera",
        forward=forward,
    )
    return authority_node, matrix_node, authority, forward, inverse_node, inverse


def _arena_frame(world):
    geometry = FakeArray(
        np.asarray([10, 5, 80, 40], dtype=np.int64),
        path="analysis/stimulus_runs/stim_1/arena_geometry_xywh",
        archive_token=world["archive_token"],
    )
    arena_node, arena_extent = _reference(
        "analysis/stimulus_runs/stim_1/arena_relative_canvas",
        rows=1,
        height=40,
        width=80,
        archive_token=world["archive_token"],
    )
    frame = stamp_arena_relative_canvas_pixel_frame_authority(
        arena_extent,
        frame_id="stim_1_arena_relative",
        pixel_convention=world["canvas_frame"].pixel_convention,
        geometry_node=geometry,
        selected_canvas_frame=world["canvas_frame"],
    )
    return geometry, arena_node, arena_extent, frame


def _arena_link(world):
    geometry, arena_node, arena_extent, frame = _arena_frame(world)
    matrix_node = FakeArray(
        arena_to_selected_canvas_matrix(frame, world["canvas_frame"]),
        path="analysis/stimulus_runs/stim_1/transforms/arena_to_canvas",
        archive_token=world["archive_token"],
    )
    authority_node = FakeGroup(
        path="analysis/stimulus_runs/stim_1/transforms/arena_to_canvas_authority",
        archive_token=world["archive_token"],
    )
    authority = stamp_arena_to_selected_canvas_transform_authority(
        authority_node,
        authority_id="stim_1_arena_to_canvas",
        matrix_node=matrix_node,
        source_frame=frame,
        target_frame=world["canvas_frame"],
    )
    link = stamp_directed_transform_v2(
        matrix_node,
        transform_id="arena_to_canvas",
        authority=authority,
        source_frame=frame,
        target_frame=world["canvas_frame"],
    )
    return geometry, arena_node, arena_extent, frame, matrix_node, authority, link


@pytest.mark.parametrize(
    "verifier,key",
    [
        (require_verified_selected_camera_source_evidence, "camera_evidence"),
        (require_verified_selected_display_source_evidence, "display_evidence"),
        (require_verified_selected_homography_source_evidence, "homography_evidence"),
    ],
)
def test_public_selected_evidence_verifiers_reject_deepcopy(verifier, key) -> None:
    world = _world()
    assert verifier(world[key]) is world[key]
    with pytest.raises(SelectedCalibrationError, match="builder-validated"):
        verifier(copy.deepcopy(world[key]))


def test_acquisition_camera_supports_external_video_without_frame_array() -> None:
    seed = _world()
    token = object()
    root = FakeGroup(
        path="archive_root",
        attrs=copy.deepcopy(seed["root"].attrs),
        archive_token=token,
    )
    authority_node = FakeGroup(
        path="analysis/acquisition_camera_frames/camera-a",
        archive_token=token,
    )
    ownership = stamp_acquisition_import_ownership(root, authority_node)
    acquisition = stamp_acquisition_camera_frame(
        root,
        authority_node,
        import_ownership=ownership,
    )
    assert acquisition.record.frame_array is None
    assert acquisition.record.frame_index is None
    assert acquisition.record.frame_count == 2
    assert acquisition.record.frame_domain["mode"] == (
        "external_video_sequential_frame_index_v1"
    )
    source = stamp_source_camera_pixel_frame_authority(
        FakeGroup(
            path="analysis/coordinate_frames/source_camera/camera-a/pixel_center",
            archive_token=token,
        ),
        frame_id="camera_a_external_video",
        pixel_convention="pixel_center",
        acquisition_frame=acquisition,
    )
    assert source.endpoint.width == 100
    assert source.record.lineage["camera_id"] == CAMERA_ID
    assert "source_camera" not in source.record.lineage
    assert load_acquisition_camera_frame(
        root,
        authority_node,
        import_ownership=ownership,
    ).record == acquisition.record


@pytest.mark.parametrize("missing", ["camera_id", "file_fingerprint"])
def test_acquisition_camera_requires_nested_future_metadata(missing: str) -> None:
    seed = _world()
    token = object()
    attrs = copy.deepcopy(seed["root"].attrs)
    del attrs["source_video_metadata"][missing]
    root = FakeGroup(path="archive_root", attrs=attrs, archive_token=token)
    authority = FakeGroup(
        path="analysis/acquisition_camera_frames/camera-a",
        archive_token=token,
    )
    with pytest.raises(PixelFrameAuthorityError, match="camera_id|fingerprint"):
        stamp_acquisition_import_ownership(root, authority)


def test_acquisition_camera_revalidates_nested_metadata_and_frame_domain() -> None:
    world = _world()
    world["root"].attrs["source_video_metadata"]["width"] = 101
    with pytest.raises(PixelFrameAuthorityError, match="metadata|dimension|record"):
        world["acquisition_frame"].assert_verified()


def test_acquisition_parser_rejects_coercible_source_scalars_and_paths() -> None:
    world = _world()
    record = world["acquisition_frame"].record.to_dict()
    record["source_video_metadata"]["width"] = 100.0
    with pytest.raises(PixelFrameAuthorityError, match="exact positive integer"):
        parse_acquisition_camera_frame(record)

    metadata = copy.deepcopy(world["root"].attrs["source_video_metadata"])
    metadata["locator"] = {
        "kind": "recording_relative",
        "relative_path": "cams//camera-a.mp4",
    }
    with pytest.raises(PixelFrameAuthorityError, match="canonical recording-relative"):
        parse_source_video_metadata(metadata)


def test_acquisition_parser_requires_exact_ownership_path_and_valid_unicode() -> None:
    world = _world()
    record = world["acquisition_frame"].record.to_dict()
    record["import_ownership"]["record_ref"] = (
        "/forged/camera-a@acquisition_import_ownership"
    )
    with pytest.raises(PixelFrameAuthorityError, match="exact import ownership"):
        parse_acquisition_camera_frame(record)

    metadata = copy.deepcopy(world["root"].attrs["source_video_metadata"])
    metadata["source_video"] = "invalid-\ud800.mp4"
    with pytest.raises(PixelFrameAuthorityError, match="valid Unicode"):
        parse_source_video_metadata(metadata)


def test_materialized_acquisition_cannot_be_minted_from_spoofed_paths_and_attrs() -> None:
    world = _world()
    spoofed_frames = FakeArray(
        path="raw_video/images_full",
        shape=(2, 80, 100),
        dtype=np.uint8,
        attrs={"format": "gray"},
    )
    spoofed_map = FakeArray(
        np.asarray([0, 1], dtype=np.int64),
        path="raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame",
        attrs={
            "source_domain": "stored_zarr_frame",
            "target_domain": "acquisition_frame",
            "semantics": "identity_map_zero_based_full_import",
        },
    )
    authority = FakeGroup(path="analysis/acquisition_camera_frames/camera-a")
    with pytest.raises(PixelFrameAuthorityError, match="sealed acquisition-import receipt"):
        stamp_acquisition_import_ownership(
            world["root"],
            authority,
            frame_node=spoofed_frames,
            frame_index_node=spoofed_map,
        )


def test_materialization_builder_rejects_detached_same_path_nodes() -> None:
    world = _world()
    detached_frames = FakeArray(
        path="raw_video/images_full",
        shape=world["camera_array"].shape,
        dtype=world["camera_array"].dtype,
        chunks=world["camera_array"].chunks,
        attrs=copy.deepcopy(world["camera_array"].attrs),
        archive_token=world["archive_token"],
    )
    decode = {
        **dict(world["acquisition_ownership"].record.import_operation["decode"]),
        "source_path": world["root"].attrs["source_video_metadata"]["source_path"],
    }
    with pytest.raises(PixelFrameAuthorityError, match="resolved from the archive root"):
        build_verified_acquisition_materialization(
            world["root"],
            frame_node=detached_frames,
            frame_index_node=world["camera_indices"],
            import_operation_attrs=decode,
        )


def test_fresh_loader_rejects_same_layout_replacement_with_different_storage_identity() -> None:
    world = _world()
    replacement = FakeArray(
        path="raw_video/images_full",
        shape=world["camera_array"].shape,
        dtype=world["camera_array"].dtype,
        chunks=world["camera_array"].chunks,
        attrs={"format": "gray"},
        archive_token=world["archive_token"],
    )
    replacement._coordinate_storage_metadata["storage_generation"] = "replacement"
    world["root"]["raw_video"]["images_full"] = replacement
    with pytest.raises(PixelFrameAuthorityError, match="storage metadata"):
        load_persisted_acquisition_camera_authority(world["root"])
    assert replacement.read_count == 0


def test_metadata_only_loader_does_not_overclaim_perfect_copy_detection() -> None:
    world = _world()
    replacement = FakeArray(
        path="raw_video/images_full",
        shape=world["camera_array"].shape,
        dtype=world["camera_array"].dtype,
        chunks=world["camera_array"].chunks,
        attrs=copy.deepcopy(world["camera_array"].attrs),
        archive_token=world["archive_token"],
    )
    replacement._coordinate_storage_metadata = copy.deepcopy(
        world["camera_array"]._coordinate_storage_metadata
    )
    world["root"]["raw_video"]["images_full"] = replacement

    ownership, _frame = load_persisted_acquisition_camera_authority(world["root"])

    verification = ownership.record.import_operation["materialization_manifest"]
    assert verification["verification_scope"] == (
        "manifest_and_storage_metadata_only_no_live_payload_rehash_v1"
    )
    assert replacement.read_count == 0


def test_physical_object_evidence_hashes_outer_shards_not_logical_chunks(
    tmp_path: Path,
) -> None:
    node = FakeArray(
        path="raw_video/images_full",
        shape=(100, 80),
        dtype=np.uint8,
        chunks=(10, 10),
        shards=(50, 40),
        store_root=tmp_path,
    )
    paths = _materialize_encoded_physical_objects(node)

    evidence = collect_acquisition_importer_physical_object_evidence(node)

    assert len(paths) == 4
    assert [entry["chunk_indices"] for entry in evidence.entries] == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    assert [entry["storage_key"] for entry in evidence.entries] == [
        "c/0/0",
        "c/0/1",
        "c/1/0",
        "c/1/1",
    ]
    assert all(
        entry["encoded_payload_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
        for entry, path in zip(evidence.entries, paths, strict=True)
    )


def test_physical_object_evidence_hashes_real_zarr_shard_files(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(
        str(tmp_path / "sharded-acquisition.zarr"),
        mode="w",
        zarr_format=3,
    )
    raw = root.create_group("raw_video")
    node = raw.create_array(
        "images_full",
        shape=(4, 3, 3),
        chunks=(1, 3, 3),
        shards=(2, 3, 3),
        dtype="uint8",
        compressors=None,
    )
    node[:] = np.arange(1, 37, dtype=np.uint8).reshape(4, 3, 3)

    evidence = collect_acquisition_importer_physical_object_evidence(node)

    assert node.chunks == (1, 3, 3)
    assert node.shards == (2, 3, 3)
    assert [entry["chunk_indices"] for entry in evidence.entries] == [
        [0, 0, 0],
        [1, 0, 0],
    ]
    for entry in evidence.entries:
        path = (
            Path(node.store_path.store.root)
            / node.path
            / entry["storage_key"]
        )
        assert entry["encoded_size_bytes"] == path.stat().st_size
        assert entry["encoded_payload_sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


@pytest.mark.parametrize(
    ("metadata_field", "bad_value", "message"),
    [
        ("shape", [99, 80], "Live array shape disagrees"),
        ("data_type", "uint16", "Live array dtype disagrees"),
    ],
)
def test_physical_object_evidence_rejects_live_storage_metadata_disagreement(
    tmp_path: Path,
    metadata_field: str,
    bad_value: object,
    message: str,
) -> None:
    node = FakeArray(
        path="raw_video/images_full",
        shape=(100, 80),
        dtype=np.uint8,
        chunks=(50, 40),
        store_root=tmp_path,
    )
    node._coordinate_storage_metadata[metadata_field] = bad_value

    with pytest.raises(PixelFrameAuthorityError, match=message):
        collect_acquisition_importer_physical_object_evidence(node)


def test_encoded_object_hash_rejects_symlink_replacement_after_path_resolution(
    tmp_path: Path,
) -> None:
    node = FakeArray(
        path="raw_video/images_full",
        shape=(2, 2),
        dtype=np.uint8,
        chunks=(2, 2),
        store_root=tmp_path,
    )
    [payload_path] = _materialize_encoded_physical_objects(node)
    root, relative_parts = (
        pixel_frame_authority_module._local_encoded_storage_object_locator(
            node, "c/0/0"
        )
    )
    outside = tmp_path / "outside-archive-payload"
    outside.write_bytes(b"external-encoded-object")
    payload_path.unlink()
    payload_path.symlink_to(outside)

    with pytest.raises(PixelFrameAuthorityError, match="Unable to hash expected"):
        pixel_frame_authority_module._hash_encoded_storage_object(
            root,
            relative_parts,
            storage_key="c/0/0",
        )


def test_importer_manifest_rejects_caller_supplied_hash_records() -> None:
    world = _world()
    forged = [copy.deepcopy(dict(entry)) for entry in world["physical_object_evidence"].entries]

    with pytest.raises(PixelFrameAuthorityError, match="sealed importer"):
        stamp_acquisition_import_writer_materialization_manifest(
            world["root"],
            frame_node=world["camera_array"],
            frame_index_node=world["camera_indices"],
            manifest_node=world["materialization_manifest_node"],
            import_operation_attrs={
                "import_method": "standard_zarr",
                "import_stage": "full_resolution",
                "import_mode": "full",
                "source_path": "/recording/cams/camera-a.mp4",
                "decode_backend": "decord_cpu",
                "source_decode_surface": "decord_rgb_uint8",
            },
            physical_object_evidence=forged,  # type: ignore[arg-type]
        )


def test_importer_evidence_rehash_rejects_post_collection_byte_change() -> None:
    world = _world()
    world["physical_object_paths"][0].write_bytes(b"post-collection-mutation")

    with pytest.raises(PixelFrameAuthorityError, match="changed after"):
        stamp_acquisition_import_writer_materialization_manifest(
            world["root"],
            frame_node=world["camera_array"],
            frame_index_node=world["camera_indices"],
            manifest_node=world["materialization_manifest_node"],
            import_operation_attrs={
                "import_method": "standard_zarr",
                "import_stage": "full_resolution",
                "import_mode": "full",
                "source_path": "/recording/cams/camera-a.mp4",
                "decode_backend": "decord_cpu",
                "source_decode_surface": "decord_rgb_uint8",
            },
            physical_object_evidence=world["physical_object_evidence"],
        )


def test_fresh_loader_rejects_tampered_immutable_chunk_manifest() -> None:
    world = _world()
    attrs = world["materialization_manifest_node"].attrs
    manifest = attrs[ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR]
    manifest["entries"][0]["encoded_payload_sha256"] = "c" * 64
    with pytest.raises(PixelFrameAuthorityError, match="digest is stale"):
        load_persisted_acquisition_camera_authority(world["root"])
    assert world["camera_array"].read_count == 0


def test_importer_manifest_digest_is_derived_and_write_once() -> None:
    world = _world()
    attrs = world["materialization_manifest_node"].attrs
    before = copy.deepcopy(dict(attrs))
    world["physical_object_paths"][0].write_bytes(b"changed-encoded-object")
    changed_evidence = collect_acquisition_importer_physical_object_evidence(
        world["camera_array"]
    )

    with pytest.raises(PixelFrameAuthorityError, match="write-once"):
        stamp_acquisition_import_writer_materialization_manifest(
            world["root"],
            frame_node=world["camera_array"],
            frame_index_node=world["camera_indices"],
            manifest_node=world["materialization_manifest_node"],
            import_operation_attrs={
                "import_method": "standard_zarr",
                "import_stage": "full_resolution",
                "import_mode": "full",
                "source_path": "/recording/cams/camera-a.mp4",
                "decode_backend": "decord_cpu",
                "source_decode_surface": "decord_rgb_uint8",
            },
            physical_object_evidence=changed_evidence,
        )

    assert dict(attrs) == before
    assert attrs[ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR] != "e" * 64


def test_importer_manifest_transaction_rejects_attr_loss_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    attrs = ClearUnrelatedOnceAttrs({"unrelated_keep_me": {"nested": [1, 2]}})
    world["materialization_manifest_node"].attrs = attrs
    before = copy.deepcopy(dict(attrs))
    monkeypatch.setattr(
        pixel_frame_authority_module,
        "require_trusted_coordinate_attrs",
        lambda node, *, label: node.attrs,
    )

    with pytest.raises(PixelFrameAuthorityError, match="snapshot plus intended"):
        stamp_acquisition_import_writer_materialization_manifest(
            world["root"],
            frame_node=world["camera_array"],
            frame_index_node=world["camera_indices"],
            manifest_node=world["materialization_manifest_node"],
            import_operation_attrs={
                "import_method": "standard_zarr",
                "import_stage": "full_resolution",
                "import_mode": "full",
                "source_path": "/recording/cams/camera-a.mp4",
                "decode_backend": "decord_cpu",
                "source_decode_surface": "decord_rgb_uint8",
            },
            physical_object_evidence=world["physical_object_evidence"],
        )

    assert dict(attrs) == before


def test_acquisition_loader_never_bulk_reads_images_full() -> None:
    world = _world()
    assert world["camera_array"].read_count == 0


def test_source_camera_authority_path_is_derived_and_write_once() -> None:
    world = _world()
    wrong_camera = FakeGroup(
        path="analysis/coordinate_frames/source_camera/camera-b/continuous",
        archive_token=world["archive_token"],
    )
    with pytest.raises(PixelFrameAuthorityError, match="exact derived path"):
        stamp_source_camera_pixel_frame_authority(
            wrong_camera,
            frame_id="wrong_camera",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )
    wrong_convention = FakeGroup(
        path=f"analysis/coordinate_frames/source_camera/{CAMERA_ID}/pixel_center",
        archive_token=world["archive_token"],
    )
    with pytest.raises(PixelFrameAuthorityError, match="exact derived path"):
        stamp_source_camera_pixel_frame_authority(
            wrong_convention,
            frame_id="wrong_convention",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )
    with pytest.raises(PixelFrameAuthorityError, match="write-once"):
        stamp_source_camera_pixel_frame_authority(
            world["camera_frame_node"],
            frame_id="different_frame_identity",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )


@pytest.mark.parametrize(
    "only_attr",
    [PIXEL_FRAME_AUTHORITY_ATTR, PIXEL_FRAME_AUTHORITY_DIGEST_ATTR],
)
def test_source_camera_write_once_rejects_incomplete_record_digest_pair(
    only_attr: str,
) -> None:
    world = _world()
    attrs = {
        only_attr: copy.deepcopy(world["camera_frame_node"].attrs[only_attr])
    }
    node = FakeGroup(
        path=f"analysis/coordinate_frames/source_camera/{CAMERA_ID}/continuous",
        attrs=attrs,
        archive_token=world["archive_token"],
    )
    before = copy.deepcopy(attrs)

    with pytest.raises(PixelFrameAuthorityError, match="incomplete.*explicit migration"):
        stamp_source_camera_pixel_frame_authority(
            node,
            frame_id="camera_a_native",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )

    assert node.attrs == before


def test_pixel_frame_stamper_rejects_dict_subclass_before_write() -> None:
    world = _world()
    hostile = HostileAttrs()
    node = FakeGroup(
        path=f"analysis/coordinate_frames/source_camera/{CAMERA_ID}/continuous",
        attrs=hostile,
        archive_token=world["archive_token"],
    )

    with pytest.raises(PixelFrameAuthorityError, match="exact trusted.*no write"):
        stamp_source_camera_pixel_frame_authority(
            node,
            frame_id="camera_a_native",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )

    assert hostile == {}
    assert hostile.update_attempts == 0


def test_transform_authority_stamper_rejects_dict_subclass_before_write() -> None:
    world = _world()
    hostile = HostileAttrs()
    authority_node = FakeGroup(
        path="analysis/detect_runs/detect_1/hostile_transform_authority",
        attrs=hostile,
        archive_token=world["archive_token"],
    )

    with pytest.raises(TransformAuthorityError, match="exact trusted.*no write"):
        stamp_model_input_transform_authority(
            authority_node,
            authority_id="hostile_model_to_roi",
            matrix_node=world["preprocessing_node"],
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )

    assert hostile == {}
    assert hostile.update_attempts == 0


def test_directed_transform_v2_stamper_rejects_dict_subclass_before_write() -> None:
    world = _world()
    authority_node = FakeGroup(
        path="analysis/detect_runs/detect_1/hostile_directed_authority",
        archive_token=world["archive_token"],
    )
    authority = stamp_model_input_transform_authority(
        authority_node,
        authority_id="hostile_directed_model_to_roi",
        matrix_node=world["preprocessing_node"],
        source_frame=world["model_frame"],
        target_frame=world["roi_frame"],
    )
    hostile = HostileAttrs(dict(world["preprocessing_node"].attrs))
    world["preprocessing_node"].attrs = hostile
    before = copy.deepcopy(dict(hostile))

    with pytest.raises(DirectedTransformV2Error, match="exact trusted.*no write"):
        stamp_directed_transform_v2(
            world["preprocessing_node"],
            transform_id="hostile_model_to_roi",
            authority=authority,
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )

    assert dict(hostile) == before
    assert hostile.update_attempts == 0


def test_pixel_frame_transaction_rejects_unrelated_attr_loss_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    attrs = ClearUnrelatedOnceAttrs({"unrelated_keep_me": {"nested": [1, 2]}})
    node = FakeGroup(
        path=f"analysis/coordinate_frames/source_camera/{CAMERA_ID}/continuous",
        attrs=attrs,
        archive_token=world["archive_token"],
    )
    before = copy.deepcopy(dict(attrs))
    monkeypatch.setattr(
        pixel_frame_authority_module,
        "require_trusted_coordinate_attrs",
        lambda node, *, label: node.attrs,
    )

    with pytest.raises(PixelFrameAuthorityError, match="snapshot plus intended"):
        stamp_source_camera_pixel_frame_authority(
            node,
            frame_id="camera_a_native",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )

    assert dict(attrs) == before


def test_transform_authority_transaction_rejects_type_coercion_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    attrs = CoerceSchemaOnceAttrs(
        {"unrelated_keep_me": {"nested": [1, 2]}},
        record_attr=TRANSFORM_AUTHORITY_ATTR,
    )
    node = FakeGroup(
        path="analysis/detect_runs/detect_1/coercing_transform_authority",
        attrs=attrs,
        archive_token=world["archive_token"],
    )
    before = copy.deepcopy(dict(attrs))
    monkeypatch.setattr(
        transform_authority_module,
        "require_trusted_coordinate_attrs",
        lambda node, *, label: node.attrs,
    )

    with pytest.raises(TransformAuthorityError, match="snapshot plus intended"):
        stamp_model_input_transform_authority(
            node,
            authority_id="coercing_model_to_roi",
            matrix_node=world["preprocessing_node"],
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )

    assert dict(attrs) == before


def test_directed_v2_transaction_rejects_unrelated_attr_loss_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    authority = stamp_model_input_transform_authority(
        FakeGroup(
            path="analysis/detect_runs/detect_1/exact_state_authority",
            archive_token=world["archive_token"],
        ),
        authority_id="exact_state_model_to_roi",
        matrix_node=world["preprocessing_node"],
        source_frame=world["model_frame"],
        target_frame=world["roi_frame"],
    )
    attrs = ClearUnrelatedOnceAttrs({"unrelated_keep_me": {"nested": [1, 2]}})
    world["preprocessing_node"].attrs = attrs
    before = copy.deepcopy(dict(attrs))
    monkeypatch.setattr(
        directed_transform_v2_module,
        "require_trusted_coordinate_attrs",
        lambda node, *, label: node.attrs,
    )

    with pytest.raises(DirectedTransformV2Error, match="snapshot plus intended"):
        stamp_directed_transform_v2(
            world["preprocessing_node"],
            transform_id="exact_state_model_to_roi",
            authority=authority,
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )

    assert dict(attrs) == before


def test_pixel_frame_transaction_rechecks_exact_attrs_after_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    node = FakeGroup(
        path=f"analysis/coordinate_frames/source_camera/{CAMERA_ID}/continuous",
        attrs={"unrelated_keep_me": {"nested": [1, 2]}},
        archive_token=world["archive_token"],
    )
    before = copy.deepcopy(node.attrs)
    original = pixel_frame_authority_module._load_by_kind

    def mutating_reload(*args: Any, **kwargs: Any):
        bound = original(*args, **kwargs)
        node.attrs.pop("unrelated_keep_me")
        return bound

    monkeypatch.setattr(pixel_frame_authority_module, "_load_by_kind", mutating_reload)
    with pytest.raises(PixelFrameAuthorityError, match="post-reload"):
        stamp_source_camera_pixel_frame_authority(
            node,
            frame_id="camera_a_native",
            pixel_convention="continuous",
            acquisition_frame=world["acquisition_frame"],
        )
    assert node.attrs == before


def test_transform_authority_transaction_rechecks_exact_attrs_after_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    node = FakeGroup(
        path="analysis/detect_runs/detect_1/reload_mutating_authority",
        attrs={"unrelated_keep_me": {"nested": [1, 2]}},
        archive_token=world["archive_token"],
    )
    before = copy.deepcopy(node.attrs)
    original = transform_authority_module.load_bound_transform_authority

    def mutating_reload(*args: Any, **kwargs: Any):
        bound = original(*args, **kwargs)
        node.attrs.pop("unrelated_keep_me")
        return bound

    monkeypatch.setattr(
        transform_authority_module,
        "load_bound_transform_authority",
        mutating_reload,
    )
    with pytest.raises(TransformAuthorityError, match="post-reload"):
        stamp_model_input_transform_authority(
            node,
            authority_id="reload_mutating_model_to_roi",
            matrix_node=world["preprocessing_node"],
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )
    assert node.attrs == before


def test_directed_v2_transaction_rechecks_exact_attrs_after_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world()
    authority = stamp_model_input_transform_authority(
        FakeGroup(
            path="analysis/detect_runs/detect_1/reload_directed_authority",
            archive_token=world["archive_token"],
        ),
        authority_id="reload_directed_model_to_roi",
        matrix_node=world["preprocessing_node"],
        source_frame=world["model_frame"],
        target_frame=world["roi_frame"],
    )
    world["preprocessing_node"].attrs = {
        "unrelated_keep_me": {"nested": [1, 2]}
    }
    before = copy.deepcopy(world["preprocessing_node"].attrs)
    original = directed_transform_v2_module.load_bound_directed_transform_v2

    def mutating_reload(*args: Any, **kwargs: Any):
        bound = original(*args, **kwargs)
        world["preprocessing_node"].attrs.pop("unrelated_keep_me")
        return bound

    monkeypatch.setattr(
        directed_transform_v2_module,
        "load_bound_directed_transform_v2",
        mutating_reload,
    )
    with pytest.raises(DirectedTransformV2Error, match="post-reload"):
        stamp_directed_transform_v2(
            world["preprocessing_node"],
            transform_id="reload_directed_model_to_roi",
            authority=authority,
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )
    assert world["preprocessing_node"].attrs == before


@pytest.mark.parametrize(
    "placement",
    [
        [[-1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        [[98.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        [[1.0, 78.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    ],
)
def test_crop_writer_rejects_negative_or_padded_source_windows(
    placement: list[list[float]],
) -> None:
    world = _world()
    node = FakeArray(
        np.asarray(placement, dtype=np.float64),
        path="analysis/crop_runs/crop_1/source_crop_xywh",
    )
    with pytest.raises(PixelFrameAuthorityError, match="contained actual source-camera"):
        stamp_crop_placement_ownership(
            node,
            row_identity=world["identity"],
            source_camera_frame=world["camera_frame"],
        )
    world["acquisition_ownership"].assert_verified()
    world["acquisition_frame"].assert_verified()
    assert world["camera_array"].read_count == 0


def test_fresh_process_acquisition_resolver_uses_only_canonical_persisted_nodes() -> None:
    world = _world()
    ownership, frame = load_persisted_acquisition_camera_authority(
        world["root"],
        expected_camera_id=CAMERA_ID,
    )
    assert ownership is not world["acquisition_ownership"]
    assert frame is not world["acquisition_frame"]
    assert ownership.record == world["acquisition_ownership"].record
    assert frame.record == world["acquisition_frame"].record
    assert world["camera_array"].read_count == 0


def test_fresh_process_acquisition_resolver_rejects_camera_or_path_aliases() -> None:
    world = _world()
    with pytest.raises(PixelFrameAuthorityError, match="Expected camera_id"):
        load_persisted_acquisition_camera_authority(
            world["root"],
            expected_camera_id="camera-b",
        )

    analysis = world["root"]["analysis"]
    original = analysis["acquisition_camera_frames"]
    analysis["acquisition_camera_frames"] = FakeGroup(
        path="analysis/debug/acquisition_camera_frames",
        archive_token=world["archive_token"],
    )
    with pytest.raises(PixelFrameAuthorityError, match="exact path"):
        load_persisted_acquisition_camera_authority(world["root"])
    analysis["acquisition_camera_frames"] = original


def test_external_acquisition_resolver_does_not_inherit_materialized_arrays() -> None:
    seed = _world()
    token = object()
    root = FakeGroup(
        path="archive_root",
        attrs=copy.deepcopy(seed["root"].attrs),
        archive_token=token,
    )
    authority = FakeGroup(
        path=f"analysis/acquisition_camera_frames/{CAMERA_ID}",
        archive_token=token,
    )
    ownership = stamp_acquisition_import_ownership(root, authority)
    expected = stamp_acquisition_camera_frame(
        root,
        authority,
        import_ownership=ownership,
    )
    analysis = FakeGroup(path="analysis", archive_token=token)
    authorities = FakeGroup(
        path="analysis/acquisition_camera_frames",
        archive_token=token,
    )
    authorities[CAMERA_ID] = authority
    analysis["acquisition_camera_frames"] = authorities
    root["analysis"] = analysis

    loaded_ownership, loaded_frame = load_persisted_acquisition_camera_authority(root)
    assert loaded_ownership.record.mode == "external_video_v1"
    assert loaded_frame.record == expected.record
    assert "raw_video" not in root.children


@pytest.mark.parametrize("dtype", [np.bool_, np.complex128])
def test_crop_writer_ownership_rejects_bool_and_complex_geometry(dtype: Any) -> None:
    world = _world()
    placement = FakeArray(
        np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype),
        path="analysis/crop_runs/crop_1/source_crop_xywh",
    )
    with pytest.raises(PixelFrameAuthorityError, match="integer or real floating"):
        stamp_crop_placement_ownership(
            placement,
            row_identity=world["identity"],
            source_camera_frame=world["camera_frame"],
        )


def test_arena_relative_canvas_binds_exact_selected_canvas_and_geometry() -> None:
    world = _world(convention="pixel_center")
    geometry, arena_node, extent, frame = _arena_frame(world)
    assert frame.endpoint.space_id == "arena_relative_canvas_px"
    assert (frame.endpoint.width, frame.endpoint.height) == (80, 40)
    assert frame.record.lineage["origin"] == "arena_top_left"
    assert frame.record.lineage["origin_in_selected_canvas_px"] == {
        "x": 10,
        "y": 5,
    }
    assert load_arena_relative_canvas_pixel_frame_authority(
        arena_node,
        reference_extent=extent,
        geometry_node=geometry,
        selected_canvas_frame=world["canvas_frame"],
    ).record == frame.record

    geometry.data[0] += 1
    with pytest.raises(PixelFrameAuthorityError, match="lineage|changed|digest"):
        frame.assert_verified()


def test_arena_relative_canvas_rejects_out_of_canvas_or_convention_drift() -> None:
    world = _world(convention="continuous")
    geometry = FakeArray(
        np.asarray([50, 30, 80, 40], dtype=np.int64),
        path="analysis/stimulus_runs/stim_1/bad_arena_geometry_xywh",
    )
    _, extent = _reference(
        "analysis/stimulus_runs/stim_1/bad_arena_relative_canvas",
        rows=1,
        height=40,
        width=80,
    )
    with pytest.raises(PixelFrameAuthorityError, match="contained"):
        stamp_arena_relative_canvas_pixel_frame_authority(
            extent,
            frame_id="bad_arena",
            pixel_convention="continuous",
            geometry_node=geometry,
            selected_canvas_frame=world["canvas_frame"],
        )

    geometry.data[:] = [10, 5, 80, 40]
    with pytest.raises(PixelFrameAuthorityError, match="pixel convention"):
        stamp_arena_relative_canvas_pixel_frame_authority(
            extent,
            frame_id="bad_arena_convention",
            pixel_convention="pixel_center",
            geometry_node=geometry,
            selected_canvas_frame=world["canvas_frame"],
        )


def test_arena_to_selected_canvas_translation_has_explicit_forward_direction() -> None:
    world = _world()
    *_, link = _arena_link(world)
    points = np.asarray([[0.0, 0.0], [7.5, 3.0]])
    np.testing.assert_allclose(
        apply_bound_directed_transform_v2(points, link),
        np.asarray([[10.0, 5.0], [17.5, 8.0]]),
    )
    assert link.transform.from_space_id == "arena_relative_canvas_px"
    assert link.transform.to_space_id == "stimulus_canvas_px"


def test_arena_translation_composes_with_canvas_to_camera_in_order() -> None:
    world = _world()
    *_, arena_link = _arena_link(world)
    *_, canvas_to_camera = _selected_link(world)
    chain = resolve_bound_directed_transform_chain(
        (arena_link, canvas_to_camera)
    )
    point = np.asarray([[2.0, 4.0]])
    expected_canvas = np.asarray([[12.0, 9.0]])
    expected_camera = apply_bound_directed_transform_v2(
        expected_canvas,
        canvas_to_camera,
    )
    np.testing.assert_allclose(
        apply_bound_directed_transform_chain(point, chain),
        expected_camera,
    )


def test_arena_authority_rejects_reverse_or_wrong_translation_payload() -> None:
    world = _world()
    _geometry, _node, _extent, arena = _arena_frame(world)
    wrong = FakeArray(
        np.asarray(
            [[1.0, 0.0, -10.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]],
            dtype="<f8",
        ),
        path="analysis/stimulus_runs/stim_1/transforms/wrong_arena_direction",
    )
    with pytest.raises(TransformAuthorityError, match="arena-relative to selected-canvas"):
        stamp_arena_to_selected_canvas_transform_authority(
            FakeGroup(
                path="analysis/stimulus_runs/stim_1/transforms/wrong_arena_authority"
            ),
            authority_id="wrong_arena_direction",
            matrix_node=wrong,
            source_frame=arena,
            target_frame=world["canvas_frame"],
        )


def test_transformer_rejects_deepcopied_selected_snapshot() -> None:
    world = _world()
    matrix_node = FakeArray(
        world["matrix"],
        path="analysis/stimulus_runs/stim_1/calibration/camera-a/matrix_copy_test",
    )
    with pytest.raises((TransformAuthorityError, SelectedCalibrationError), match="sealed persisted"):
        stamp_selected_calibration_transform_authority(
            FakeGroup(path="analysis/stimulus_runs/stim_1/calibration/camera-a/copy_auth"),
            authority_id="copy_rejected",
            source_matrix_node=matrix_node,
            source_frame=world["camera_frame"],
            target_frame=world["canvas_frame"],
            selected_calibration_snapshot=copy.deepcopy(world["selected_snapshot"]),
        )


def test_generic_extent_cannot_substitute_for_typed_endpoint() -> None:
    world = _world()
    authority_node, authority, _ = _model_link(world)
    del authority_node
    with pytest.raises((PixelFrameAuthorityError, AttributeError, TypeError)):
        stamp_directed_transform_v2(
            FakeArray(
                model_input_to_roi_matrix(world["preprocessing"]),
                path="analysis/detect_runs/detect_1/transforms/forged",
            ),
            transform_id="forged",
            authority=authority,
            source_frame=world["model_extent"],  # type: ignore[arg-type]
            target_frame=world["roi_frame"],
        )


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("continuous", [50.0, 20.0]),
        ("pixel_center", [49.5, 19.75]),
    ],
)
def test_normalized_to_camera_pixel_formula_is_explicit(
    convention: str,
    expected: list[float],
) -> None:
    world = _world(convention=convention)
    normalized_frame = stamp_normalized_pixel_frame_authority(
        FakeGroup(path="analysis/detect_runs/d1/source_camera_normalized_frame"),
        frame_id="camera_a_normalized",
        pixel_frame=world["camera_frame"],
    )
    matrix_node = FakeArray(
        normalized_to_pixel_matrix(world["camera_frame"]),
        path="analysis/detect_runs/d1/transforms/normalized_to_camera",
    )
    authority = stamp_normalized_to_pixel_transform_authority(
        FakeGroup(
            path="analysis/detect_runs/d1/transforms/normalized_to_camera_authority"
        ),
        authority_id="camera_normalized_to_pixel",
        matrix_node=matrix_node,
        source_frame=normalized_frame,
        target_frame=world["camera_frame"],
    )
    link = stamp_directed_transform_v2(
        matrix_node,
        transform_id="camera_normalized_to_pixel",
        authority=authority,
        source_frame=normalized_frame,
        target_frame=world["camera_frame"],
    )
    np.testing.assert_allclose(
        apply_bound_directed_transform_v2(np.asarray([[0.5, 0.25]]), link),
        np.asarray([expected]),
    )
    assert link.transform.source.units == "normalized"
    assert link.transform.target.units == "px"


def test_normalized_authority_rejects_w_minus_one_when_extent_formula_requires_w() -> None:
    world = _world(convention="continuous")
    normalized_frame = stamp_normalized_pixel_frame_authority(
        FakeGroup(path="analysis/detect_runs/d1/source_camera_normalized_frame"),
        frame_id="camera_a_normalized",
        pixel_frame=world["camera_frame"],
    )
    wrong = FakeArray(
        np.asarray(
            [[99.0, 0.0, 0.0], [0.0, 79.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="<f8",
        ),
        path="analysis/detect_runs/d1/transforms/wrong_normalization",
    )
    with pytest.raises(TransformAuthorityError, match="controlled W/H formula"):
        stamp_normalized_to_pixel_transform_authority(
            FakeGroup(path="analysis/detect_runs/d1/transforms/wrong_norm_authority"),
            authority_id="wrong_normalized_formula",
            matrix_node=wrong,
            source_frame=normalized_frame,
            target_frame=world["camera_frame"],
        )


def test_same_dimension_debug_frame_cannot_substitute_for_selected_camera() -> None:
    world = _world()
    _, _, authority, _, _, _ = _selected_link(world)
    debug_array, _ = _reference(
        "analysis/debug/native_sized_frames",
        rows=2,
        height=80,
        width=100,
        attrs={"camera_id": CAMERA_ID},
    )
    debug_indices = FakeArray(
        np.asarray([0, 1], dtype=np.int64),
        path="analysis/debug/source_frame_index",
    )
    debug_node = FakeGroup(path="analysis/acquisition_camera_frames/camera-a")
    with pytest.raises(PixelFrameAuthorityError, match="canonical|receipt|map"):
        stamp_acquisition_import_ownership(
            world["root"],
            debug_node,
            frame_node=debug_array,
            frame_index_node=debug_indices,
        )
    assert debug_array.path != world["camera_array"].path


def test_typed_frames_revalidate_live_lineage() -> None:
    world = _world()
    bound = world["roi_frame"]
    world["placements"].data[0, 0] += 1
    with pytest.raises(PixelFrameAuthorityError, match="lineage|changed|stale"):
        require_bound_pixel_frame_authority(bound)


def test_cross_archive_roi_evidence_fails_closed() -> None:
    world = _world()
    other_token = object()
    _, other_extent = _reference(
        "analysis/crop_runs/crop_1/roi_other",
        rows=2,
        height=20,
        width=40,
        archive_token=other_token,
    )
    with pytest.raises(PixelFrameAuthorityError, match="archive|store"):
        stamp_roi_pixel_frame_authority(
            other_extent,
            frame_id="cross_archive_roi",
            pixel_convention="continuous",
            crop_placement_ownership=world["crop_ownership"],
        )


@pytest.mark.parametrize("field", ["native_height", "native_width", "model_height", "model_width"])
def test_model_endpoint_rejects_each_mismatched_dimension(field: str) -> None:
    world = _world()
    transform = replace(
        world["preprocessing"],
        **{field: getattr(world["preprocessing"], field) + 1},
    )
    _, extent = _reference(
        f"analysis/detect_runs/detect_bad/{field}",
        rows=2,
        height=32,
        width=64,
    )
    with pytest.raises(PixelFrameAuthorityError, match="dimension|Padding|pad_to_size"):
        stamp_model_input_pixel_frame_authority(
            extent,
            frame_id=f"bad_{field}",
            pixel_convention="continuous",
            preprocessing_node=world["preprocessing_node"],
            transform=transform,
            roi_frame=world["roi_frame"],
        )


def test_transform_records_carry_conventions_and_sampling_formula() -> None:
    world = _world(convention="continuous")
    _, crop = _crop_link(world)
    assert crop.transform.source.pixel_convention == "continuous"
    assert crop.transform.target.pixel_convention == "continuous"
    assert crop.transform.sampling_formula == SCALE_XY_EDGE_ALIGNED_V1
    assert (
        world["placements"].attrs[DIRECTED_TRANSFORM_V2_ATTR]["sampling_formula"]
        == SCALE_XY_EDGE_ALIGNED_V1
    )


def test_pixel_center_and_edge_formulas_differ_under_nonunit_scale() -> None:
    center = _world(convention="pixel_center")
    _, center_link = _crop_link(center)
    center_result = apply_bound_directed_transform_v2(
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        center_link,
        row_identity=center["identity"],
    )
    assert center_link.transform.sampling_formula == SCALE_XY_PIXEL_CENTER_V1
    np.testing.assert_allclose(center_result[1], [49.75, 5.5])

    edge = _world(convention="continuous")
    _, edge_link = _crop_link(edge)
    edge_result = apply_bound_directed_transform_v2(
        np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        edge_link,
        row_identity=edge["identity"],
    )
    np.testing.assert_allclose(edge_result[1], [50.0, 5.0])
    assert not np.array_equal(center_result, edge_result)


def test_model_crop_chain_applies_exact_typed_lineage() -> None:
    world = _world()
    _, _, model = _model_link(world)
    _, crop = _crop_link(world)
    chain = resolve_bound_directed_transform_chain((model, crop))
    actual = apply_bound_directed_transform_chain(
        np.asarray([[12.0, 6.0], [12.0, 6.0]]),
        chain,
        row_identity=world["identity"],
    )
    # model padding is (left=12, top=6), then rowwise crop placement.
    np.testing.assert_allclose(actual, [[10.0, 20.0], [50.0, 5.0]])
    assert chain.descriptor_frame_authority is world["model_frame"]
    assert chain.source_camera_frame_authority is world["camera_frame"]


def test_chain_rejects_convention_discontinuity_even_when_extents_match() -> None:
    world = _world()
    _, _, model = _model_link(world)
    # Direct persisted-record tampering cannot be hidden by recomputing only the
    # transform digest; the typed endpoint record remains different.
    raw = copy.deepcopy(world["preprocessing_node"].attrs[DIRECTED_TRANSFORM_V2_ATTR])
    raw["target"]["pixel_convention"] = "pixel_center"
    with pytest.raises(DirectedTransformV2Error, match="convention|endpoint"):
        parse_directed_transform_v2(raw)
    assert model.transform.target.pixel_convention == "continuous"


def test_chain_rejects_source_camera_before_final_link() -> None:
    world = _world()
    _, _, _, forward, _, inverse = _selected_link(world)
    with pytest.raises(DirectedTransformChainError, match="only as the final"):
        resolve_bound_directed_transform_chain((inverse, forward))


def test_chain_rejects_repeated_transform_record_and_cycle() -> None:
    world = _world()
    _, _, _, _, _, inverse = _selected_link(world)
    with pytest.raises(DirectedTransformChainError, match="cannot occur twice"):
        resolve_bound_directed_transform_chain((inverse, inverse))


def test_explicit_inverse_rejects_inverse_of_inverse() -> None:
    world = _world()
    _, _, _, _, _, inverse = _selected_link(world)
    second = FakeArray(
        world["matrix"],
        path="analysis/stimulus_runs/stim_1/calibration/camera-a/inverse_twice",
    )
    with pytest.raises(DirectedTransformV2Error, match="cannot be inverted again"):
        stamp_explicit_inverse_directed_transform_v2(
            second,
            transform_id="inverse_twice",
            forward=inverse,
        )


def _selected_migration_coexistence(*, convention: str = "continuous"):
    world = _world(convention=convention)
    node = FakeArray(
        world["matrix"],
        path=(
            "analysis/stimulus_runs/stim_1/calibration/camera-a/"
            "migration_homography"
        ),
        archive_token=world["archive_token"],
    )
    authority = stamp_selected_calibration_transform_authority(
        FakeGroup(
            path=(
                "analysis/stimulus_runs/stim_1/calibration/camera-a/"
                "migration_authority"
            ),
            archive_token=world["archive_token"],
        ),
        authority_id="stim_1_camera_a_migration",
        source_matrix_node=node,
        source_frame=world["camera_frame"],
        target_frame=world["canvas_frame"],
        selected_calibration_snapshot=world["selected_snapshot"],
    )
    selected = world["selected_snapshot"]
    legacy = build_directed_homography(
        transform_id="camera_to_canvas_historical_v1",
        matrix=node[:],
        from_space_id=world["camera_frame"].space_id,
        to_space_id=world["canvas_frame"].space_id,
        source_reference_extent=selected.source_reference_extent,
        target_reference_extent=selected.target_reference_extent,
        calibration_ref=selected.manifest.camera_calibration_ref,
        camera_id=selected.camera_id,
    )
    node.attrs.update(directed_homography_attrs(legacy))
    migrated = parse_directed_transform_v2(
        {
            "schema_id": "palette.directed_transform",
            "schema_version": 2,
            "transform_id": "camera_to_canvas_migrated_v2",
            "kind": "homography",
            "from_space_id": world["camera_frame"].space_id,
            "to_space_id": world["canvas_frame"].space_id,
            "direction": "source_to_target",
            "source": world["camera_frame"].endpoint.to_dict(),
            "target": world["canvas_frame"].endpoint.to_dict(),
            "transform_authority": {
                "kind": authority.record.kind,
                "record_ref": authority.record_ref,
                "record_sha256": authority.record_sha256,
            },
            "payload": authority.record.payload.to_dict(),
            "sampling_formula": authority.record.sampling_formula,
            "camera_id": authority.record.camera_id,
        }
    )
    node.attrs[DIRECTED_TRANSFORM_V2_ATTR] = migrated.to_dict()
    node.attrs[DIRECTED_TRANSFORM_V2_DIGEST_ATTR] = migrated.digest()
    legacy_digest = node.attrs[
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    ]
    eligibility = {
        "schema_id": MIGRATION_ELIGIBILITY_SCHEMA_ID,
        "schema_version": MIGRATION_ELIGIBILITY_SCHEMA_VERSION,
        "decision": "eligible",
        "evidence_basis": MIGRATION_ELIGIBILITY_BASIS,
        "legacy_transform": {
            "record_ref": f"/{node.path}@{DIRECTED_TRANSFORM_ATTR}",
            "record_sha256": legacy_digest,
        },
        "source_frame_authority": {
            "record_ref": world["camera_frame"].record_ref,
            "record_sha256": world["camera_frame"].record_sha256,
        },
        "target_frame_authority": {
            "record_ref": world["canvas_frame"].record_ref,
            "record_sha256": world["canvas_frame"].record_sha256,
        },
        "source_pixel_convention": convention,
        "target_pixel_convention": convention,
        "source_extent_authority": selected.source_reference_extent.authority,
        "target_extent_authority": selected.target_reference_extent.authority,
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    node.attrs[MIGRATION_ELIGIBILITY_ATTR] = eligibility
    node.attrs[MIGRATION_ELIGIBILITY_DIGEST_ATTR] = _canonical_mapping_sha256(
        eligibility
    )
    return world, node, authority


def test_migration_only_v1_v2_coexistence_accepts_exact_equivalent_pair() -> None:
    world, node, authority = _selected_migration_coexistence()
    validated = validate_migration_only_v1_v2_coexistence(
        node,
        authority=authority,
        source_frame=world["camera_frame"],
        target_frame=world["canvas_frame"],
    )
    assert validated.from_space_id == "source_camera_image_px"
    assert validated.to_space_id == "stimulus_canvas_px"

    with pytest.raises(DirectedTransformV2Error, match="historical attrs"):
        load_bound_directed_transform_v2(
            node,
            authority=authority,
            source_frame=world["camera_frame"],
            target_frame=world["canvas_frame"],
        )


@pytest.mark.parametrize(
    "missing",
    [MIGRATION_ELIGIBILITY_ATTR, MIGRATION_ELIGIBILITY_DIGEST_ATTR],
)
def test_migration_gate_requires_complete_persisted_eligibility(
    missing: str,
) -> None:
    world, node, authority = _selected_migration_coexistence()
    del node.attrs[missing]

    with pytest.raises(DirectedTransformV2Error, match="complete v1 and v2"):
        validate_migration_only_v1_v2_coexistence(
            node,
            authority=authority,
            source_frame=world["camera_frame"],
            target_frame=world["canvas_frame"],
        )


@pytest.mark.parametrize("tamper", ["bool_schema", "float_width", "extent_authority"])
def test_migration_gate_rejects_nonexact_legacy_scalars_and_authorities(
    tamper: str,
) -> None:
    world, node, authority = _selected_migration_coexistence()
    raw = copy.deepcopy(node.attrs[DIRECTED_TRANSFORM_ATTR])
    if tamper == "bool_schema":
        raw["schema_version"] = True
    elif tamper == "float_width":
        raw["source_reference_extent"]["width"] = float(
            raw["source_reference_extent"]["width"]
        )
    else:
        raw["source_reference_extent"]["authority"] = (
            "analysis/root@inferred_width_height"
        )
    legacy_digest = directed_transform_digest(raw)
    node.attrs[DIRECTED_TRANSFORM_ATTR] = raw
    node.attrs[
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    ] = legacy_digest
    eligibility = copy.deepcopy(node.attrs[MIGRATION_ELIGIBILITY_ATTR])
    eligibility["legacy_transform"]["record_sha256"] = legacy_digest
    node.attrs[MIGRATION_ELIGIBILITY_ATTR] = eligibility
    node.attrs[MIGRATION_ELIGIBILITY_DIGEST_ATTR] = _canonical_mapping_sha256(
        eligibility
    )

    with pytest.raises(DirectedTransformV2Error, match="exact|authority"):
        validate_migration_only_v1_v2_coexistence(
            node,
            authority=authority,
            source_frame=world["camera_frame"],
            target_frame=world["canvas_frame"],
        )


def test_migration_gate_never_infers_v1_pixel_convention_from_caller_frames() -> None:
    world, node, authority = _selected_migration_coexistence(
        convention="pixel_center"
    )

    with pytest.raises(DirectedTransformV2Error, match="never caller-inferred"):
        validate_migration_only_v1_v2_coexistence(
            node,
            authority=authority,
            source_frame=world["camera_frame"],
            target_frame=world["canvas_frame"],
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "legacy_direction",
        "legacy_extent",
        "legacy_camera",
        "legacy_payload",
        "legacy_calibration",
        "legacy_digest",
        "v2_endpoint",
        "v2_digest",
    ),
)
def test_migration_only_v1_v2_coexistence_rejects_any_disagreement(
    tamper: str,
) -> None:
    world, node, authority = _selected_migration_coexistence()
    legacy_digest_attr = (
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    )
    if tamper.startswith("legacy_") and tamper != "legacy_digest":
        raw = copy.deepcopy(node.attrs[DIRECTED_TRANSFORM_ATTR])
        if tamper == "legacy_direction":
            raw["from_space_id"], raw["to_space_id"] = (
                raw["to_space_id"],
                raw["from_space_id"],
            )
            raw["source_reference_extent"], raw["target_reference_extent"] = (
                raw["target_reference_extent"],
                raw["source_reference_extent"],
            )
        elif tamper == "legacy_extent":
            raw["source_reference_extent"]["width"] += 1
        elif tamper == "legacy_camera":
            raw["camera_id"] = "camera-b"
        elif tamper == "legacy_payload":
            raw["matrix_sha256"] = "0" * 64
        else:
            raw["calibration_ref"] = (
                "analysis/stimulus_runs/stim_1/calibration/camera-b"
            )
        node.attrs[DIRECTED_TRANSFORM_ATTR] = raw
        node.attrs[legacy_digest_attr] = directed_transform_digest(raw)
    elif tamper == "legacy_digest":
        node.attrs[legacy_digest_attr] = "0" * 64
    elif tamper == "v2_endpoint":
        raw = copy.deepcopy(node.attrs[DIRECTED_TRANSFORM_V2_ATTR])
        raw["source"]["width"] += 1
        node.attrs[DIRECTED_TRANSFORM_V2_ATTR] = raw
        node.attrs[DIRECTED_TRANSFORM_V2_DIGEST_ATTR] = (
            parse_directed_transform_v2(raw).digest()
        )
    else:
        node.attrs[DIRECTED_TRANSFORM_V2_DIGEST_ATTR] = "0" * 64

    with pytest.raises(DirectedTransformV2Error):
        validate_migration_only_v1_v2_coexistence(
            node,
            authority=authority,
            source_frame=world["camera_frame"],
            target_frame=world["canvas_frame"],
        )


@pytest.mark.parametrize("version", [True, 2.0])
def test_all_transform_schema_versions_require_exact_int(version: Any) -> None:
    world = _world()
    _, crop = _crop_link(world)
    transform_raw = copy.deepcopy(crop.transform.to_dict())
    transform_raw["schema_version"] = version
    with pytest.raises(DirectedTransformV2Error, match="exact integer"):
        parse_directed_transform_v2(transform_raw)

    authority_raw = copy.deepcopy(crop.authority.record.to_dict())
    authority_raw["schema_version"] = version
    with pytest.raises(TransformAuthorityError, match="schema_version"):
        parse_transform_authority(authority_raw)

    frame_raw = copy.deepcopy(world["roi_frame"].record.to_dict())
    frame_raw["schema_version"] = version
    with pytest.raises(PixelFrameAuthorityError, match="schema_version"):
        parse_pixel_frame_record(frame_raw)

    acquisition_raw = copy.deepcopy(world["acquisition_frame"].record.to_dict())
    acquisition_raw["schema_version"] = version
    with pytest.raises(PixelFrameAuthorityError, match="schema_version"):
        parse_acquisition_camera_frame(acquisition_raw)


@pytest.mark.parametrize("kind", ["selected", "crop", "model"])
def test_all_pixel_authority_kinds_reject_mm_endpoints(kind: str) -> None:
    world = _world()
    if kind == "selected":
        _, _, authority, _, _, _ = _selected_link(world)
    elif kind == "crop":
        authority, _ = _crop_link(world)
    else:
        _, authority, _ = _model_link(world)
    raw = copy.deepcopy(authority.record.to_dict())
    raw["source"]["units"] = "mm"
    with pytest.raises(TransformAuthorityError, match="must be 'px'"):
        parse_transform_authority(raw)


def test_directed_transform_rejects_mm_endpoint() -> None:
    world = _world()
    _, crop = _crop_link(world)
    raw = copy.deepcopy(crop.transform.to_dict())
    raw["target"]["units"] = "mm"
    with pytest.raises(DirectedTransformV2Error, match="must be 'px'"):
        parse_directed_transform_v2(raw)


def test_authority_loader_rejects_noncanonical_raw_mapping_even_with_valid_digest() -> None:
    world = _world()
    authority_node, authority, _ = _model_link(world)
    raw = copy.deepcopy(authority_node.attrs[TRANSFORM_AUTHORITY_ATTR])
    raw["semantics"] = OrderedDict(raw["semantics"])
    parsed = parse_transform_authority(raw)
    authority_node.attrs[TRANSFORM_AUTHORITY_ATTR] = raw
    authority_node.attrs[TRANSFORM_AUTHORITY_DIGEST_ATTR] = parsed.digest()
    with pytest.raises(TransformAuthorityError, match="[Rr]aw persisted.*canonical"):
        load_bound_transform_authority(
            authority_node,
            payload_node=world["preprocessing_node"],
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )
    assert authority.record.kind == "model_input_preprocessing"


def test_transform_loader_rejects_noncanonical_raw_mapping_even_with_valid_digest() -> None:
    world = _world()
    _, authority, link = _model_link(world)
    raw = copy.deepcopy(world["preprocessing_node"].attrs[DIRECTED_TRANSFORM_V2_ATTR])
    raw["source"] = OrderedDict(raw["source"])
    parsed = parse_directed_transform_v2(raw)
    world["preprocessing_node"].attrs[DIRECTED_TRANSFORM_V2_ATTR] = raw
    world["preprocessing_node"].attrs[DIRECTED_TRANSFORM_V2_DIGEST_ATTR] = parsed.digest()
    with pytest.raises(DirectedTransformV2Error, match="[Rr]aw persisted.*canonical"):
        load_bound_directed_transform_v2(
            world["preprocessing_node"],
            authority=authority,
            source_frame=world["model_frame"],
            target_frame=world["roi_frame"],
        )
    assert link.transform.kind == "affine_2d_constant"


def test_pixel_frame_loader_rejects_noncanonical_nested_mapping_type() -> None:
    world = _world()
    raw = copy.deepcopy(world["camera_frame_node"].attrs[PIXEL_FRAME_AUTHORITY_ATTR])
    raw["lineage"]["acquisition_camera_frame"] = OrderedDict(
        raw["lineage"]["acquisition_camera_frame"]
    )
    parsed = parse_pixel_frame_record(raw)
    world["camera_frame_node"].attrs[PIXEL_FRAME_AUTHORITY_ATTR] = raw
    world["camera_frame_node"].attrs[PIXEL_FRAME_AUTHORITY_DIGEST_ATTR] = parsed.digest()
    with pytest.raises(PixelFrameAuthorityError, match="[Rr]aw persisted.*canonical"):
        load_source_camera_pixel_frame_authority(
            world["camera_frame_node"],
            acquisition_frame=world["acquisition_frame"],
        )


def test_acquisition_loader_rejects_noncanonical_nested_mapping_type() -> None:
    world = _world()
    raw = copy.deepcopy(world["camera_node"].attrs[ACQUISITION_CAMERA_FRAME_ATTR])
    raw["source_video_metadata"] = OrderedDict(raw["source_video_metadata"])
    parsed = parse_acquisition_camera_frame(raw)
    world["camera_node"].attrs[ACQUISITION_CAMERA_FRAME_ATTR] = raw
    world["camera_node"].attrs[ACQUISITION_CAMERA_FRAME_DIGEST_ATTR] = parsed.digest()
    with pytest.raises(PixelFrameAuthorityError, match="[Rr]aw acquisition.*canonical"):
        load_acquisition_camera_frame(
            world["root"],
            world["camera_node"],
            import_ownership=world["acquisition_ownership"],
        )


def test_selected_authority_uses_persisted_import_snapshot() -> None:
    world = _world()
    _, _, authority, _, _, _ = _selected_link(world)
    assert authority.record.semantics["external_h5_freshness"] == (
        "persisted_import_snapshot"
    )


def test_chain_revalidation_rejects_mutated_endpoint_attrs() -> None:
    world = _world()
    _, _, model = _model_link(world)
    _, crop = _crop_link(world)
    chain = resolve_bound_directed_transform_chain((model, crop))
    world["roi_node"].attrs[PIXEL_FRAME_AUTHORITY_ATTR]["frame_id"] = "changed"
    with pytest.raises(DirectedTransformChainError, match="stale|changed|digest"):
        require_bound_directed_transform_chain(chain)


def test_rowwise_application_rejects_different_same_length_identity() -> None:
    world = _world()
    _, crop = _crop_link(world)
    other_rowset = FakeGroup(path="analysis/crop_runs/crop_2")
    other_keys = FakeArray(
        np.asarray([101, 202], dtype=np.uint64),
        path=f"{other_rowset.path}/instance_key",
    )
    other_identity = stamp_and_bind_row_identity_contract(
        other_rowset,
        other_keys,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=other_keys.data,
        ),
    )
    with pytest.raises(DirectedTransformV2Error, match="identity differs"):
        apply_bound_directed_transform_v2(
            np.zeros((2, 2)),
            crop,
            row_identity=other_identity,
        )
