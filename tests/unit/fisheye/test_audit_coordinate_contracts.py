from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import fisheye.utils.audit_coordinate_contracts as coordinate_audit
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_NOT_PUBLISHED,
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_PENDING_REASON,
    MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
    build_acquisition_authority_publication_status,
)
from fisheye.shared.coordinate_descriptor import coordinate_descriptor_digest
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_NOT_SUITABLE,
    CANONICAL_OVERLAY_DIRECT,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    build_canonical_coordinate_descriptor,
    canonical_coordinate_descriptor_v2_attrs,
    canonical_coordinate_descriptor_v2_digest,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import bind_array_reference_extent
from fisheye.shared.directed_transform_chain import (
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform import (
    build_directed_homography,
    directed_homography_attrs,
)
from fisheye.shared.directed_transform_v2 import (
    AFFINE_2D_CONSTANT_KIND,
    AFFINE_2D_ROWWISE_KIND,
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_DIGEST_ATTR,
    HOMOGRAPHY_KIND,
    stamp_directed_transform_v2,
)
from fisheye.shared.coordinate_frame_record import (
    PHYSICAL_FRAME_CALIBRATION_ATTR,
    PHYSICAL_FRAME_CALIBRATION_KIND,
    PHYSICAL_SOURCE_CAMERA_PROFILE_ID,
    build_physical_frame_calibration_record,
    stamp_physical_frame_calibration_record,
    stamp_selected_camera_frame_evidence,
)
from fisheye.shared.observation_coordinate_publication import (
    DETECTION_ACQUISITION_MAPPING_ATTR,
    DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
    DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
    build_bound_detection_frame_evidence,
    derive_detection_source_camera_geometry,
    publish_crop_observation_geometry,
    publish_crop_roi_geometry,
    publish_detection_observation_geometry,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    STIMULUS_STATE_DOMAIN,
    TRACK_SAMPLE_DOMAIN,
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    build_row_identity_contract,
    derive_track_source_instance_values,
    resolve_source_acquisition_frame_indices,
    row_identity_contract_attrs,
    row_identity_key_attrs,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
    stamp_track_sample_time_lineage,
)
from fisheye.shared.pixel_frame_authority import (
    PIXEL_FRAME_AUTHORITY_ATTR,
    normalized_to_pixel_matrix,
    stamp_crop_placement_ownership,
    stamp_acquisition_camera_frame,
    stamp_acquisition_import_ownership,
    stamp_normalized_pixel_frame_authority,
    stamp_roi_pixel_frame_authority,
    stamp_source_camera_pixel_frame_authority,
)
from fisheye.shared.selected_calibration import (
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.transform_authority import (
    AuthorityRowIdentity,
    stamp_crop_placement_transform_authority,
    stamp_normalized_to_pixel_transform_authority,
)

from fisheye.utils.audit_coordinate_contracts import audit_registry
from fisheye.utils.audit_coordinate_contracts import open_registry_readonly
from fisheye.utils.audit_coordinate_contracts import summarize
from fisheye.utils.audit_coordinate_contracts import write_csv
from fisheye.utils.audit_coordinate_contracts import write_jsonl
from fisheye.utils.audit_coordinate_contracts import write_markdown
from fisheye.utils.audit_coordinate_contracts import write_summary


class _MemoryNode:
    """Minimal array/group surface accepted by the sealed shared APIs."""

    def __init__(
        self,
        path: str,
        *,
        token: object,
        attrs: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> None:
        self.path = path
        self._coordinate_archive_token = token
        self.attrs = {} if attrs is None else attrs
        self._data = np.asarray(0 if data is None else data)
        self.shape = tuple(int(value) for value in self._data.shape)
        self.dtype = self._data.dtype

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[key] = value


def _sealed_source_camera_frame_attrs(
    *,
    pixel_convention: str = "pixel_center",
) -> tuple[
    str, dict[str, Any], str, dict[str, Any], Any, Any, object, dict[str, Any]
]:
    """Mint canonical v2 acquisition/pixel records with the public APIs."""

    token = object()
    metadata = {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "camera_id": "camera-1",
        "width": 640,
        "height": 480,
        "total_frames": 2,
        "locator": {
            "kind": "recording_relative",
            "relative_path": "camera.mp4",
        },
        "file_fingerprint": {
            "strategy": "size_mtime_sha256_v1",
            "value": "a" * 64,
            "size_bytes": 1234,
            "mtime_ns": 5678,
            "relocation_stable": False,
        },
    }
    root = _MemoryNode(
        "archive_root",
        token=token,
        attrs={"recording_id": "rec", "source_video_metadata": metadata},
    )
    acquisition_path = "analysis/acquisition_camera_frames/camera-1"
    acquisition_node = _MemoryNode(acquisition_path, token=token)
    ownership = stamp_acquisition_import_ownership(root, acquisition_node)
    acquisition = stamp_acquisition_camera_frame(
        root,
        acquisition_node,
        import_ownership=ownership,
    )
    frame_path = f"analysis/coordinate_frames/source_camera/camera-1/{pixel_convention}"
    authority_node = _MemoryNode(frame_path, token=token)
    frame = stamp_source_camera_pixel_frame_authority(
        authority_node,
        frame_id="camera_1_native",
        pixel_convention=pixel_convention,
        acquisition_frame=acquisition,
    )
    return (
        acquisition_path,
        dict(acquisition_node.attrs),
        frame_path,
        dict(authority_node.attrs),
        frame,
        acquisition,
        token,
        dict(root.attrs),
    )


def _memory_metadata_node(
    node: _MemoryNode,
    *,
    node_type: str,
    relative_path: str | None = None,
) -> coordinate_audit.MetadataNode:
    is_array = node_type == "array"
    return coordinate_audit.MetadataNode(
        relative_path=relative_path if relative_path is not None else node.path,
        node_type=node_type,
        metadata_format="memory",
        shape=list(node.shape) if is_array else None,
        data_type=node.dtype.str if is_array else None,
        chunk_shape=list(node.shape) if is_array else None,
        storage_metadata=None,
        attributes=json.loads(json.dumps(dict(node.attrs))),
    )


def _observation_array_payload(node: _MemoryNode) -> dict[str, Any]:
    return {
        "array_ref": f"/{node.path}",
        "dtype": node.dtype.str,
        "shape": list(node.shape),
        "content_sha256": array_payload_sha256(node),
    }


def _canonical_detection_crop_metadata_nodes() -> dict[str, coordinate_audit.MetadataNode]:
    """Mint the exact public detection/crop record graph, then drop payloads."""

    (
        acquisition_path,
        acquisition_attrs,
        camera_path,
        camera_attrs,
        camera_frame,
        acquisition,
        token,
        root_attrs,
    ) = _sealed_source_camera_frame_attrs(pixel_convention="continuous")
    normalized_node = _MemoryNode(
        "detect_runs/d1/coordinate_frames/source_camera_normalized",
        token=token,
    )
    normalized_frame = stamp_normalized_pixel_frame_authority(
        normalized_node,
        frame_id="d1_source_camera_normalized",
        pixel_frame=camera_frame,
    )
    transform_node = _MemoryNode(
        "detect_runs/d1/transforms/source_camera_normalized_to_image",
        token=token,
        data=normalized_to_pixel_matrix(camera_frame),
    )
    transform_authority_node = _MemoryNode(
        "detect_runs/d1/transforms/source_camera_normalized_to_image_authority",
        token=token,
    )
    transform_authority = stamp_normalized_to_pixel_transform_authority(
        transform_authority_node,
        authority_id="d1_source_camera_normalized_to_image",
        matrix_node=transform_node,
        source_frame=normalized_frame,
        target_frame=camera_frame,
    )
    normalized_to_image = stamp_directed_transform_v2(
        transform_node,
        transform_id="d1_source_camera_normalized_to_image",
        authority=transform_authority,
        source_frame=normalized_frame,
        target_frame=camera_frame,
    )
    frame_evidence = build_bound_detection_frame_evidence(
        source_camera_frame=camera_frame,
        normalized_frame=normalized_frame,
        normalized_to_source_camera=resolve_bound_directed_transform_chain(
            (normalized_to_image,)
        ),
    )

    detect = _MemoryNode("detect_runs/d1", token=token)
    decode_frames = _MemoryNode(
        "detect_runs/d1/frame_indices",
        token=token,
        data=np.asarray([0, 1], dtype=np.int32),
    )
    source_frames = _MemoryNode(
        "detect_runs/d1/source_acquisition_frame_index",
        token=token,
        data=np.asarray([0, 1], dtype=np.int64),
    )
    instance_key = _MemoryNode(
        "detect_runs/d1/instance_key",
        token=token,
        data=np.asarray([101, 202], dtype=np.uint64),
    )
    bbox_norm_values = np.asarray(
        [[0.25, 0.50, 0.20, 0.25], [0.75, 0.25, 0.10, 0.20]],
        dtype=np.float64,
    )
    bbox_img_values, center_values = derive_detection_source_camera_geometry(
        bbox_norm_values,
        frame_evidence=frame_evidence,
    )
    bbox_norm = _MemoryNode(
        "detect_runs/d1/bbox_norm_coords",
        token=token,
        data=bbox_norm_values,
    )
    bbox_img = _MemoryNode(
        "detect_runs/d1/bbox_img_xyxy",
        token=token,
        data=bbox_img_values,
    )
    centers = _MemoryNode(
        "detect_runs/d1/centers_img_xy",
        token=token,
        data=center_values,
    )
    source_metadata = acquisition.record.source_video_metadata
    mapping_record = {
        "schema_id": DETECTION_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": DETECTION_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "operation": "full_untrimmed_video_decode_identity_to_acquisition_v1",
        "direction": "decode_frame_index_to_source_acquisition_frame_index",
        "decode_frame_index": _observation_array_payload(decode_frames),
        "source_acquisition_frame_index": _observation_array_payload(source_frames),
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
        },
        "source_video_locator": dict(source_metadata["locator"]),
        "source_video_fingerprint": dict(source_metadata["file_fingerprint"]),
        "source_total_frames": acquisition.record.source_total_frames,
        "proof": "exact_locator_and_stat_fingerprint_revalidated_after_full_decode",
    }
    mapping = stamp_and_bind_persisted_coordinate_record(
        detect,
        mapping_record,
        attr_name=DETECTION_ACQUISITION_MAPPING_ATTR,
    )
    detection = publish_detection_observation_geometry(
        detect,
        instance_key,
        source_frames,
        bbox_norm,
        bbox_img,
        centers,
        frame_evidence=frame_evidence,
        source_lineage_records=(mapping,),
    )

    crop = _MemoryNode("crop_runs/c1", token=token)
    selected = np.asarray([1, 0], dtype=np.int64)
    detection_indices = _MemoryNode(
        "crop_runs/c1/detection_indices",
        token=token,
        data=selected,
    )
    crop_key = _MemoryNode(
        "crop_runs/c1/instance_key",
        token=token,
        data=instance_key[:][selected],
    )
    crop_frames = _MemoryNode(
        "crop_runs/c1/source_acquisition_frame_index",
        token=token,
        data=source_frames[:][selected],
    )
    crop_bbox_norm = _MemoryNode(
        "crop_runs/c1/bbox_norm_coords",
        token=token,
        data=bbox_norm[:][selected],
    )
    crop_bbox_img = _MemoryNode(
        "crop_runs/c1/bbox_img_xyxy",
        token=token,
        data=bbox_img[:][selected],
    )
    crop_centers = _MemoryNode(
        "crop_runs/c1/centers_img_xy",
        token=token,
        data=centers[:][selected],
    )
    crop_geometry = publish_crop_observation_geometry(
        crop,
        crop_key,
        detection_indices,
        crop_frames,
        crop_bbox_norm,
        crop_bbox_img,
        crop_centers,
        source_geometry=detection,
    )
    source_crop = _MemoryNode(
        "crop_runs/c1/source_crop_xywh",
        token=token,
        data=np.asarray(
            [[0.0, 0.0, 640.0, 480.0], [0.0, 0.0, 640.0, 480.0]],
            dtype=np.float64,
        ),
    )
    crop_ownership = stamp_crop_placement_ownership(
        source_crop,
        row_identity=crop_geometry.row_identity,
        source_camera_frame=camera_frame,
    )
    roi_images = _MemoryNode(
        "crop_runs/c1/roi_images",
        token=token,
        data=np.zeros((2, 480, 640), dtype=np.uint8),
    )
    roi_frame = stamp_roi_pixel_frame_authority(
        bind_array_reference_extent(roi_images, units="px"),
        frame_id="c1_roi",
        pixel_convention="continuous",
        crop_placement_ownership=crop_ownership,
    )
    crop_transform_authority = stamp_crop_placement_transform_authority(
        source_crop,
        authority_id="c1_roi_to_source_camera",
        source_frame=roi_frame,
        target_frame=camera_frame,
    )
    crop_transform = stamp_directed_transform_v2(
        source_crop,
        transform_id="c1_roi_to_source_camera",
        authority=crop_transform_authority,
        source_frame=roi_frame,
        target_frame=camera_frame,
        row_identity=crop_geometry.row_identity,
    )
    bbox_roi = _MemoryNode(
        "crop_runs/c1/bbox_roi_xyxy",
        token=token,
        data=np.asarray(crop_bbox_img[:], dtype=np.float64),
    )
    publish_crop_roi_geometry(
        source_crop,
        bbox_roi,
        crop_geometry=crop_geometry,
        crop_placement_ownership=crop_ownership,
        roi_frame=roi_frame,
        roi_to_source_camera=resolve_bound_directed_transform_chain(
            (crop_transform,)
        ),
    )

    groups = {
        ".": _MemoryNode("archive_root", token=token, attrs=root_attrs),
        acquisition_path: _MemoryNode(
            acquisition_path,
            token=token,
            attrs=acquisition_attrs,
        ),
        camera_path: _MemoryNode(camera_path, token=token, attrs=camera_attrs),
        normalized_node.path: normalized_node,
        transform_authority_node.path: transform_authority_node,
        detect.path: detect,
        crop.path: crop,
    }
    arrays = (
        transform_node,
        decode_frames,
        source_frames,
        instance_key,
        bbox_norm,
        bbox_img,
        centers,
        detection_indices,
        crop_key,
        crop_frames,
        crop_bbox_norm,
        crop_bbox_img,
        crop_centers,
        source_crop,
        roi_images,
        bbox_roi,
    )
    result = {
        path: _memory_metadata_node(
            node,
            node_type="group",
            relative_path=("." if path == "." else path),
        )
        for path, node in groups.items()
    }
    result.update(
        {
            node.path: _memory_metadata_node(node, node_type="array")
            for node in arrays
        }
    )
    return result


def _redigest_observation_record(
    nodes: dict[str, coordinate_audit.MetadataNode],
    *,
    owner_path: str,
    attr_name: str,
    mutate: Callable[[dict[str, Any]], None],
    descriptor_paths: tuple[str, ...],
) -> dict[str, coordinate_audit.MetadataNode]:
    """Mutate a record while keeping every enclosing metadata digest current."""

    changed = dict(nodes)
    owner = changed[owner_path]
    owner_attrs = json.loads(json.dumps(owner.attributes))
    record = owner_attrs[attr_name]
    mutate(record)
    record_sha256 = coordinate_audit._fingerprint(record)
    owner_attrs[attr_name] = record
    owner_attrs[f"{attr_name}_sha256"] = record_sha256
    changed[owner_path] = replace(owner, attributes=owner_attrs)

    record_ref = f"/{owner_path}@{attr_name}"
    for descriptor_path in descriptor_paths:
        surface = changed[descriptor_path]
        surface_attrs = json.loads(json.dumps(surface.attributes))
        descriptor = surface_attrs["coordinate_descriptor"]
        matches = [
            pointer
            for pointer in descriptor.get("lineage_refs", [])
            if pointer.get("record_ref") == record_ref
        ]
        assert len(matches) == 1
        matches[0]["record_sha256"] = record_sha256
        surface_attrs["coordinate_descriptor"] = descriptor
        surface_attrs["coordinate_descriptor_sha256"] = (
            canonical_coordinate_descriptor_v2_digest(descriptor)
        )
        changed[descriptor_path] = replace(
            surface,
            attributes=surface_attrs,
        )
    return changed


def _classify_metadata_surface(
    nodes: dict[str, coordinate_audit.MetadataNode],
    path: str,
) -> tuple[str | None, dict[str, Any]]:
    surface_type = coordinate_audit.classify_surface(path, nodes[path], nodes)
    assert surface_type is not None
    return surface_type, coordinate_audit.classify_surface_contract(
        surface_type=surface_type,
        node=nodes[path],
        nodes=nodes,
    )


def _sealed_physical_frame_attrs() -> dict[str, Any]:
    """Mint a complete source-camera-preserving physical frame fixture."""

    token = object()
    source_metadata = {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "camera_id": "camera-1",
        "width": 640,
        "height": 480,
        "total_frames": 2,
        "locator": {
            "kind": "recording_relative",
            "relative_path": "camera.mp4",
        },
        "file_fingerprint": {
            "strategy": "size_mtime_sha256_v1",
            "value": "a" * 64,
            "size_bytes": 1234,
            "mtime_ns": 5678,
            "relocation_stable": False,
        },
    }
    root = _MemoryNode(
        "archive_root",
        token=token,
        attrs={
            "recording_id": "rec",
            "source_video_metadata": source_metadata,
        },
    )
    acquisition_path = "analysis/acquisition_camera_frames/camera-1"
    acquisition_node = _MemoryNode(acquisition_path, token=token)
    ownership = stamp_acquisition_import_ownership(root, acquisition_node)
    acquisition = stamp_acquisition_camera_frame(
        root,
        acquisition_node,
        import_ownership=ownership,
    )
    source_path = (
        "analysis/coordinate_frames/source_camera/camera-1/continuous"
    )
    source_node = _MemoryNode(source_path, token=token)
    source_frame = stamp_source_camera_pixel_frame_authority(
        source_node,
        frame_id="camera_1_native",
        pixel_convention="continuous",
        acquisition_frame=acquisition,
    )

    camera_record = {
        "camera_id": "camera-1",
        "native_width_px": 640,
        "native_height_px": 480,
        "pixels_per_mm_camera": 50.0,
        "pixels_per_mm_projector": 4.0,
        "real_world_ref_mm": 10.0,
    }
    camera_evidence = build_selected_camera_source_evidence_from_h5_values(
        source_h5_path="/data/recording.h5",
        arena_config_raw=json.dumps(
            {
                "active_camera_id": "camera-1",
                "camera_calibrations": [camera_record],
            },
            separators=(",", ":"),
        ),
        camera_group_path="/calibration_snapshot/camera-1",
        camera_group_attrs={
            "pixels_per_mm_camera": 50.0,
            "pixels_per_mm_projector": 4.0,
            "real_world_ref_mm": 10.0,
        },
        expected_camera_id="camera-1",
    )
    selected_path = "analysis/calibration/selected_camera"
    selected_node = _MemoryNode(selected_path, token=token)
    selected = stamp_selected_camera_frame_evidence(
        selected_node,
        source_camera=camera_evidence,
    )
    physical_path = "analysis/coordinate_frames/camera_mm"
    physical_node = _MemoryNode(physical_path, token=token)
    physical = stamp_physical_frame_calibration_record(
        physical_node,
        record=build_physical_frame_calibration_record(
            frame_id="camera_1_mm",
            source_camera_pixels=source_frame,
            selected_camera_evidence=selected,
        ),
        expected_record_ref=(
            f"/{physical_path}@{PHYSICAL_FRAME_CALIBRATION_ATTR}"
        ),
        source_camera_pixels=source_frame,
        selected_camera_evidence=selected,
    )
    return {
        "root_attrs": dict(root.attrs),
        "acquisition_path": acquisition_path,
        "acquisition_attrs": dict(acquisition_node.attrs),
        "source_path": source_path,
        "source_attrs": dict(source_node.attrs),
        "selected_path": selected_path,
        "selected_attrs": dict(selected_node.attrs),
        "physical_path": physical_path,
        "physical_attrs": dict(physical_node.attrs),
        "physical": physical,
    }


def _write_materialized_acquisition_fixture(root: Path) -> dict[str, Any]:
    """Write parser-valid, cross-linked materialized acquisition metadata."""

    source_metadata = {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "camera_id": "camera-1",
        "width": 640,
        "height": 480,
        "total_frames": 2,
        "source_path": "/recordings/camera.mp4",
        "locator": {
            "kind": "absolute",
            "path": "/recordings/camera.mp4",
        },
        "file_fingerprint": {
            "strategy": "size_mtime_sha256_v1",
            "value": "a" * 64,
            "size_bytes": 1234,
            "mtime_ns": 5678,
            "relocation_stable": False,
        },
    }
    _merge_node_attrs(
        root,
        ".",
        {"recording_id": "rec", "source_video_metadata": source_metadata},
    )
    frame_path = "raw_video/images_full"
    index_path = (
        "raw_video/frame_domain_maps/"
        "stored_zarr_frame_to_acquisition_frame"
    )
    _ensure_groups(root, frame_path)
    _write_node(
        root,
        frame_path,
        node_type="array",
        shape=[2, 480, 640],
        data_type="uint8",
        chunk_shape=[1, 480, 640],
    )
    _ensure_groups(root, index_path)
    _write_node(
        root,
        index_path,
        node_type="array",
        shape=[2],
        data_type="int64",
        attributes={
            "source_domain": "stored_zarr_frame",
            "target_domain": "acquisition_frame",
            "semantics": "identity_map_zero_based_full_import",
        },
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(root)
    }
    extent = coordinate_audit._metadata_array_extent_pointer(
        nodes[frame_path],
        units="px",
    )
    storage = coordinate_audit._metadata_array_storage_identity(
        nodes[frame_path]
    )
    assert extent is not None and storage is not None
    frame_map = {
        "record_ref": f"/{index_path}@array_values",
        "record_sha256": "1" * 64,
        "selector": "array_values",
    }
    decode = {
        "import_method": "ffmpeg",
        "import_stage": "complete",
        "import_mode": "full",
        "decode_backend": "pyav",
        "source_decode_surface": "video_frames",
    }
    manifest_path = coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_PATH
    entries = [
        {
            "chunk_indices": [index, 0, 0],
            "storage_key": f"c/{index}/0/0",
            "encoded_payload_sha256": str(index + 2) * 64,
            "encoded_size_bytes": 100 + index,
        }
        for index in range(2)
    ]
    physical_chunk_manifest = {
        "schema_id": (
            coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID
        ),
        "schema_version": (
            coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION
        ),
        "producer": coordinate_audit.ACQUISITION_IMPORT_PRODUCER,
        "array_ref": f"/{frame_path}@array_values",
        "array_storage": storage,
        "scope": coordinate_audit.ACQUISITION_CHUNK_MANIFEST_SCOPE,
        "content_evidence_scope": (
            coordinate_audit.ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE
        ),
        "digest_algorithm": "sha256",
        "entries": entries,
        "entry_count": 2,
        "entries_sha256": coordinate_audit._fingerprint(entries),
        "entries_canonicalization": (
            coordinate_audit.ACQUISITION_CHUNK_ENTRY_CANONICALIZATION
        ),
        "canonicalization": (
            coordinate_audit.PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        ),
    }
    physical_chunk_sha256 = coordinate_audit._fingerprint(
        physical_chunk_manifest
    )
    chunk_pointer = {
        "record_ref": (
            f"/{manifest_path}@"
            f"{coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR}"
        ),
        "record_sha256": physical_chunk_sha256,
        "entry_count": 2,
        "scope": coordinate_audit.ACQUISITION_CHUNK_MANIFEST_SCOPE,
        "content_evidence_scope": (
            coordinate_audit.ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE
        ),
        "metadata_only_verification_scope": (
            coordinate_audit.ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE
        ),
    }
    metadata_sha256 = coordinate_audit._fingerprint(source_metadata)
    identity_basis = {
        "recording_id": "rec",
        "camera_id": "camera-1",
        "source_video_metadata_sha256": metadata_sha256,
        "decode": decode,
        "images_full_storage": storage,
        "frame_map": frame_map,
        "physical_chunk_manifest": chunk_pointer,
    }
    manifest = {
        "schema_id": (
            coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID
        ),
        "schema_version": (
            coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        ),
        "producer": coordinate_audit.ACQUISITION_IMPORT_PRODUCER,
        "recording_id": "rec",
        "camera_id": "camera-1",
        "materialization_id": coordinate_audit._fingerprint(identity_basis),
        "write_policy": (
            coordinate_audit.ACQUISITION_MATERIALIZATION_WRITE_POLICY
        ),
        "completed": True,
        "source_video_metadata_sha256": metadata_sha256,
        "decode": decode,
        "images_full_storage": storage,
        "frame_map": frame_map,
        "physical_chunk_manifest": chunk_pointer,
        "canonicalization": (
            coordinate_audit.PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        ),
    }
    manifest_sha256 = coordinate_audit._fingerprint(manifest)
    _ensure_groups(root, f"{manifest_path}/manifest_record")
    _write_node(
        root,
        manifest_path,
        attributes={
            coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR: (
                physical_chunk_manifest
            ),
            coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR: (
                physical_chunk_sha256
            ),
            coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_ATTR: (
                manifest
            ),
            coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR: (
                manifest_sha256
            ),
        },
    )
    operation = {
        "schema_id": "palette.acquisition_materialization_receipt",
        "schema_version": 1,
        "producer": coordinate_audit.ACQUISITION_IMPORT_PRODUCER,
        "recording_id": "rec",
        "camera_id": "camera-1",
        "source_locator": source_metadata["locator"],
        "source_fingerprint": source_metadata["file_fingerprint"],
        "decode": decode,
        "materialization_manifest": {
            "record_ref": (
                f"/{manifest_path}@"
                f"{coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_ATTR}"
            ),
            "record_sha256": manifest_sha256,
            "materialization_id": manifest["materialization_id"],
            "physical_chunk_manifest": chunk_pointer,
            "verification_scope": (
                coordinate_audit.ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE
            ),
        },
        "canonicalization": (
            coordinate_audit.PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        ),
    }
    ownership = {
        "schema_id": "palette.acquisition_import_ownership",
        "schema_version": 1,
        "recording_id": "rec",
        "camera_id": "camera-1",
        "producer": coordinate_audit.ACQUISITION_IMPORT_PRODUCER,
        "mode": "materialized_source_frames_v1",
        "source_video_metadata_sha256": metadata_sha256,
        "frame_array": {**extent, "storage_identity": storage},
        "frame_index": frame_map,
        "import_operation": operation,
        "canonicalization": (
            coordinate_audit.PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        ),
    }
    ownership_sha256 = coordinate_audit._fingerprint(ownership)
    acquisition_path = "analysis/acquisition_camera_frames/camera-1"
    acquisition = {
        "schema_id": "palette.acquisition_camera_frame",
        "schema_version": 2,
        "recording_id": "rec",
        "camera_id": "camera-1",
        "source_video_metadata": source_metadata,
        "source_video_metadata_sha256": metadata_sha256,
        "width_px": 640,
        "height_px": 480,
        "source_total_frames": 2,
        "frame_domain": {
            "mode": "explicit_stored_zarr_to_acquisition_frame_map_v1",
            "index_record_ref": frame_map["record_ref"],
            "index_record_sha256": frame_map["record_sha256"],
        },
        "frame_array": extent,
        "frame_index": frame_map,
        "frame_count": 2,
        "import_ownership": {
            "record_ref": (
                f"/{acquisition_path}@acquisition_import_ownership"
            ),
            "record_sha256": ownership_sha256,
        },
        "canonicalization": (
            coordinate_audit.PIXEL_FRAME_AUTHORITY_CANONICALIZATION
        ),
    }
    acquisition_sha256 = coordinate_audit._fingerprint(acquisition)
    _ensure_groups(root, acquisition_path)
    _write_node(
        root,
        acquisition_path,
        attributes={
            "acquisition_import_ownership": ownership,
            "acquisition_import_ownership_sha256": ownership_sha256,
            "acquisition_camera_frame": acquisition,
            "acquisition_camera_frame_sha256": acquisition_sha256,
        },
    )
    publication_status = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=MATERIALIZED_ACQUISITION_PUBLISHED_REASON,
        authority_mode=MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
        authority_path=acquisition_path,
    ).to_dict()
    _merge_node_attrs(
        root,
        ".",
        {"acquisition_authority_publication_status": publication_status},
    )
    _merge_node_attrs(
        root,
        "raw_video",
        {"acquisition_authority_publication_status": publication_status},
    )
    return {
        "acquisition_path": acquisition_path,
        "frame_path": frame_path,
        "index_path": index_path,
        "manifest_path": manifest_path,
        "publication_status": publication_status,
        "pointer": {
            "record_ref": f"/{acquisition_path}@acquisition_camera_frame",
            "record_sha256": acquisition_sha256,
            "selector": "acquisition_camera_frame",
            "width": 640,
            "height": 480,
            "units": "px",
        },
    }


def _redigest_materialized_acquisition_nodes(
    nodes: dict[str, coordinate_audit.MetadataNode],
    fixture: dict[str, Any],
    *,
    mutate_chunk: Callable[[dict[str, Any]], None] | None = None,
    mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, coordinate_audit.MetadataNode], dict[str, Any]]:
    """Rebind every digest after intentionally malformed manifest edits."""

    candidate = dict(nodes)
    manifest_path = str(fixture["manifest_path"])
    manifest_attrs = json.loads(
        json.dumps(nodes[manifest_path].attributes)
    )
    chunk = manifest_attrs[
        coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR
    ]
    if mutate_chunk is not None:
        mutate_chunk(chunk)
    chunk["entries_sha256"] = coordinate_audit._fingerprint(
        chunk["entries"]
    )
    chunk_sha256 = coordinate_audit._fingerprint(chunk)
    manifest_attrs[
        coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR
    ] = chunk_sha256

    materialization = manifest_attrs[
        coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_ATTR
    ]
    chunk_pointer = materialization["physical_chunk_manifest"]
    chunk_pointer["record_sha256"] = chunk_sha256
    chunk_pointer["entry_count"] = chunk["entry_count"]
    if mutate_manifest is not None:
        mutate_manifest(materialization)
    identity_basis = {
        "recording_id": materialization["recording_id"],
        "camera_id": materialization["camera_id"],
        "source_video_metadata_sha256": materialization[
            "source_video_metadata_sha256"
        ],
        "decode": materialization["decode"],
        "images_full_storage": materialization["images_full_storage"],
        "frame_map": materialization["frame_map"],
        "physical_chunk_manifest": chunk_pointer,
    }
    materialization["materialization_id"] = coordinate_audit._fingerprint(
        identity_basis
    )
    materialization_sha256 = coordinate_audit._fingerprint(materialization)
    manifest_attrs[
        coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR
    ] = materialization_sha256
    candidate[manifest_path] = replace(
        nodes[manifest_path],
        attributes=manifest_attrs,
    )

    authority_path = str(fixture["acquisition_path"])
    authority_attrs = json.loads(
        json.dumps(nodes[authority_path].attributes)
    )
    ownership = authority_attrs["acquisition_import_ownership"]
    operation_manifest = ownership["import_operation"][
        "materialization_manifest"
    ]
    operation_manifest["record_sha256"] = materialization_sha256
    operation_manifest["materialization_id"] = materialization[
        "materialization_id"
    ]
    operation_manifest["physical_chunk_manifest"] = chunk_pointer
    ownership_sha256 = coordinate_audit._fingerprint(ownership)
    authority_attrs["acquisition_import_ownership_sha256"] = ownership_sha256
    acquisition = authority_attrs["acquisition_camera_frame"]
    acquisition["import_ownership"]["record_sha256"] = ownership_sha256
    acquisition_sha256 = coordinate_audit._fingerprint(acquisition)
    authority_attrs["acquisition_camera_frame_sha256"] = acquisition_sha256
    candidate[authority_path] = replace(
        nodes[authority_path],
        attributes=authority_attrs,
    )
    pointer = json.loads(json.dumps(fixture["pointer"]))
    pointer["record_sha256"] = acquisition_sha256
    return candidate, pointer


def _write_node(
    root: Path,
    relative_path: str = ".",
    *,
    node_type: str = "group",
    attributes: dict[str, object] | None = None,
    shape: list[int] | None = None,
    data_type: str = "float32",
    chunk_shape: list[int] | None = None,
) -> Path:
    path = root if relative_path == "." else root / relative_path
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "zarr_format": 3,
        "node_type": node_type,
        "attributes": attributes or {},
    }
    if node_type == "array":
        payload.update(
            {
                "shape": shape or [1],
                "data_type": data_type,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {
                        "chunk_shape": chunk_shape or shape or [1]
                    },
                },
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0,
                "codecs": [],
            }
        )
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ensure_groups(root: Path, relative_path: str) -> None:
    parts = Path(relative_path).parts[:-1]
    for index in range(1, len(parts) + 1):
        path = "/".join(parts[:index])
        metadata = root / path / "zarr.json"
        if not metadata.exists():
            _write_node(root, path)


def _merge_node_attrs(
    root: Path,
    relative_path: str,
    attributes: dict[str, object],
) -> None:
    metadata_path = root / relative_path / "zarr.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.setdefault("attributes", {}).update(attributes)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_canonical_row_identity(
    root: Path,
    rowset_path: str,
    *,
    domain: str,
    row_count: int,
    key_attributes: dict[str, object] | None = None,
) -> None:
    if domain == OBSERVATION_INSTANCE_DOMAIN:
        values = np.arange(row_count, dtype=np.uint64)
        components = None
    elif domain == TRACK_SAMPLE_DOMAIN:
        values = np.column_stack(
            (
                np.zeros(row_count, dtype=np.int64),
                np.arange(row_count, dtype=np.int64),
            )
        )
        components = None
    elif domain == STIMULUS_STATE_DOMAIN:
        values = np.arange(row_count, dtype=np.int64)
        components = ("stimulus_state_index",)
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unsupported row identity domain: {domain}")
    contract = build_row_identity_contract(
        domain=domain,
        values=values,
        components=components,
    )
    key_path = f"{rowset_path}/{contract.key_array.ref}"
    _ensure_groups(root, key_path)
    _write_node(
        root,
        key_path,
        node_type="array",
        attributes={
            **row_identity_key_attrs(contract),
            **(key_attributes or {}),
        },
        shape=list(contract.key_array.shape),
        data_type=np.dtype(contract.key_array.dtype).name,
    )
    _merge_node_attrs(root, rowset_path, row_identity_contract_attrs(contract))


def _write_array(
    root: Path,
    relative_path: str,
    *,
    attributes: dict[str, object] | None = None,
    shape: list[int] | None = None,
    data_type: str | None = None,
) -> None:
    effective_shape = shape or [1]
    leaf = Path(relative_path).name
    effective_attrs = dict(attributes or {})
    if leaf in {"instance_key", "track_sample_key", "stimulus_state_key"}:
        domain = {
            "instance_key": OBSERVATION_INSTANCE_DOMAIN,
            "track_sample_key": TRACK_SAMPLE_DOMAIN,
            "stimulus_state_key": STIMULUS_STATE_DOMAIN,
        }[leaf]
        _ensure_groups(root, relative_path)
        _write_canonical_row_identity(
            root,
            str(Path(relative_path).parent),
            domain=domain,
            row_count=effective_shape[0],
            key_attributes=effective_attrs,
        )
        return
    if leaf == "frame_indices":
        data_type = data_type or "int64"
        effective_attrs.update(
            {
                "schema_id": "palette.track_row_identity",
                "schema_version": 1,
                "leading_dimension": effective_shape[0],
                "unique": True,
                "content_sha256": "2" * 64,
                "digest_canonicalization": "canonical_integer_array_v1",
            }
        )
    elif leaf == "coordinate_row_identity":
        data_type = data_type or "int64"
        effective_attrs.update(
            {
                "schema_id": "palette.coordinate_row_identity",
                "schema_version": 1,
                "leading_dimension": effective_shape[0],
                "unique": True,
                "content_sha256": "3" * 64,
                "digest_canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
            }
        )
    _ensure_groups(root, relative_path)
    _write_node(
        root,
        relative_path,
        node_type="array",
        attributes=effective_attrs,
        shape=effective_shape,
        data_type=data_type or "float32",
    )


def _replace_group_attrs(root: Path, relative_path: str, attributes: dict[str, object]) -> None:
    _write_node(root, relative_path, attributes=attributes)


_CANONICAL_LINEAGE_RECORD = {
    "schema_id": "test.coordinate_lineage",
    "schema_version": 1,
    "source": "unit_test_fixture",
}
_CANONICAL_LINEAGE_REF = "/coordinate_records/source.attrs[lineage]"


def _write_contract_evidence(root: Path) -> None:
    _write_array(root, "raw_video/images_full", shape=[2, 480, 640])
    _write_node(root, "coordinate_records")
    _write_node(
        root,
        "coordinate_records/source",
        attributes={"lineage": _CANONICAL_LINEAGE_RECORD},
    )
    _write_node(
        root,
        "coordinate_records/arena_geometry",
        attributes={
            "arena_region_width_px": 640,
            "arena_region_height_px": 480,
            "arena_origin_in_canvas_x_px": 0,
            "arena_origin_in_canvas_y_px": 0,
        },
    )


def _write_selected_camera_authority(
    root: Path,
    *,
    selection_name: str,
    camera_id: str,
    width: int,
    height: int,
) -> str:
    selection_path = f"coordinate_records/{selection_name}"
    camera_path = f"{selection_path}/{camera_id}"
    manifest = {
        "schema_id": "palette.selected_calibration_manifest",
        "schema_version": 1,
        "camera_id": camera_id,
        "camera_calibration_ref": camera_path,
        "camera_calibration": {
            "native_width_px": width,
            "native_height_px": height,
        },
    }
    _write_node(
        root,
        selection_path,
        attributes={
            "schema_id": "palette.selected_calibration_snapshot",
            "schema_version": 1,
            "active_camera_id": camera_id,
            "active_camera_calibration_ref": camera_path,
            "selected_calibration_manifest": manifest,
            "selected_calibration_manifest_sha256": (
                coordinate_audit._fingerprint(manifest)
            ),
        },
    )
    _write_node(
        root,
        camera_path,
        attributes={
            "schema_id": "palette.camera_calibration_snapshot",
            "schema_version": 1,
            "camera_id": camera_id,
            "native_width_px": width,
            "native_height_px": height,
        },
    )
    return camera_path


def _make_registry(
    path: Path,
    rows: list[dict[str, object]],
    *,
    recording_rows: list[dict[str, object]] | None = None,
) -> Path:
    for row in rows:
        row.setdefault(
            "zarr_origin",
            (
                "source"
                if row.get("artifact_kind") == "source_recording"
                else "derived"
            ),
        )
        raw_zarr_path = row.get("zarr_path")
        if raw_zarr_path in (None, ""):
            continue
        metadata_path = Path(str(raw_zarr_path)) / "zarr.json"
        if not metadata_path.is_file():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        attributes = payload.setdefault("attributes", {})
        if row.get("dataset_id") not in (None, ""):
            attributes["dataset_id"] = row["dataset_id"]
        if row.get("recording_id") not in (None, ""):
            attributes["recording_id"] = row["recording_id"]
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE recordings (
                recording_id TEXT PRIMARY KEY,
                recording_path TEXT,
                session_uuid TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT,
                zarr_origin TEXT,
                zarr_use TEXT,
                artifact_kind TEXT,
                status TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_path, zarr_origin, zarr_use,
                artifact_kind, status
            ) VALUES (
                :dataset_id, :recording_id, :zarr_path, :zarr_origin, :zarr_use,
                :artifact_kind, :status
            )
            """,
            rows,
        )
        if recording_rows is None:
            deduplicated: dict[str, dict[str, object]] = {}
            for row in rows:
                recording_id = row.get("recording_id")
                if recording_id not in (None, ""):
                    deduplicated[str(recording_id)] = {
                        "recording_id": recording_id,
                        "recording_path": row.get("zarr_path"),
                        "session_uuid": None,
                    }
            recording_rows = list(deduplicated.values())
        conn.executemany(
            """
            INSERT INTO recordings (recording_id, recording_path, session_uuid)
            VALUES (:recording_id, :recording_path, :session_uuid)
            """,
            recording_rows,
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _complete_xy_descriptor(*, space_id: str = "source_camera_image_px", units: str = "px") -> dict[str, object]:
    return {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 1,
        "space_id": space_id,
        "geometry_type": "points_xy",
        "components": ["x", "y"],
        "component_units": [units, units],
        "origin": "top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "reference_extent": {
            "width": 640,
            "height": 480,
            "units": "px",
            "authority": "/raw_video/images_full.shape[-2:]",
        },
        "pixel_convention": "pixel_center",
        "row_identity": {
            "mode": "track_frame_indices",
            "array_ref": "frame_indices",
        },
        "source_camera_overlay": "direct",
        "lineage_refs": [
            {
                "ref": _CANONICAL_LINEAGE_REF,
                "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
            }
        ],
    }


def _complete_arena_descriptor(root: Path) -> dict[str, object]:
    arena_node = next(
        node
        for node in coordinate_audit.iter_metadata_nodes(root)
        if node.relative_path == "coordinate_records/arena_geometry"
    )
    arena_ref = "/coordinate_records/arena_geometry"
    return {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 1,
        "space_id": "arena_relative_canvas_px",
        "geometry_type": "points_xy",
        "components": ["x", "y"],
        "component_units": ["px", "px"],
        "origin": "arena_top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "reference_extent": {
            "width": 640,
            "height": 480,
            "units": "px",
            "authority": f"{arena_ref}.attrs",
        },
        "pixel_convention": "continuous",
        "row_identity": {
            "mode": "track_frame_indices",
            "array_ref": "frame_indices",
        },
        "source_camera_overlay": "not_suitable",
        "lineage_refs": [
            {
                "ref": arena_ref,
                "sha256": coordinate_audit._metadata_node_record_digest(arena_node),
            }
        ],
    }


def _descriptor_attrs(descriptor: dict[str, object] | None = None) -> dict[str, object]:
    value = descriptor or _complete_xy_descriptor()
    return {
        "coordinate_descriptor": value,
        "coordinate_descriptor_sha256": coordinate_descriptor_digest(value),
    }


def _unvalidated_descriptor_attrs(descriptor: dict[str, object]) -> dict[str, object]:
    """Simulate persisted descriptor attrs that a canonical writer would reject."""

    encoded = json.dumps(
        descriptor,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "coordinate_descriptor": descriptor,
        "coordinate_descriptor_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _dataset_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record["record_type"] == "coordinate_dataset"]


def _surface_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if record["record_type"] == "coordinate_surface"]


@pytest.mark.parametrize(
    ("surface_path", "expected_type"),
    [
        ("detect_runs/d1/bbox_norm_coords", "detect_bbox"),
        ("detect_runs/d1/bbox_img_xyxy", "detect_bbox"),
        ("detect_runs/d1/centers_img_xy", "detect_bbox"),
        ("crop_runs/c1/bbox_norm_coords", "crop_geometry"),
        ("crop_runs/c1/bbox_img_xyxy", "crop_geometry"),
        ("crop_runs/c1/centers_img_xy", "crop_geometry"),
        ("crop_runs/c1/source_crop_xywh", "crop_geometry"),
        ("crop_runs/c1/bbox_roi_xyxy", "crop_geometry"),
    ],
)
def test_canonical_detection_crop_records_require_live_payload_validation(
    surface_path: str,
    expected_type: str,
) -> None:
    nodes = _canonical_detection_crop_metadata_nodes()

    surface_type, result = _classify_metadata_surface(nodes, surface_path)

    assert surface_type == expected_type
    assert result["status"] == "numerical_validation_required"
    assert "OBSERVATION_COORDINATE_PAYLOAD_VALIDATION_REQUIRED" in {
        issue["code"] for issue in result["issues"]
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("direction", "source_camera_image_px_to_source_camera_normalized_xy"),
        ("reference_width_px", 641),
        ("reference_height_px", 481),
        (
            "row_identity",
            {
                "record_ref": "/crop_runs/c1@row_identity_contract",
                "record_sha256": "f" * 64,
            },
        ),
    ],
)
def test_detection_projection_semantics_fail_closed_after_valid_redigest(
    field: str,
    bad_value: object,
) -> None:
    surface_path = "detect_runs/d1/bbox_img_xyxy"
    nodes = _redigest_observation_record(
        _canonical_detection_crop_metadata_nodes(),
        owner_path="detect_runs/d1",
        attr_name=coordinate_audit.DETECTION_BBOX_PROJECTION_ATTR,
        mutate=lambda record: record.__setitem__(field, bad_value),
        descriptor_paths=(surface_path,),
    )

    _surface_type, result = _classify_metadata_surface(nodes, surface_path)

    assert result["status"] == "ambiguous_fail_closed"
    assert "DETECTION_BBOX_PROJECTION_INVALID" in {
        issue["code"] for issue in result["issues"]
    }


def test_detection_temporal_rowset_identity_fails_closed_after_valid_redigest(
) -> None:
    surface_path = "detect_runs/d1/bbox_img_xyxy"
    nodes = _canonical_detection_crop_metadata_nodes()
    detect = nodes["detect_runs/d1"]
    attrs = json.loads(json.dumps(detect.attributes))
    temporal = attrs[coordinate_audit.SOURCE_ROW_TEMPORAL_AUTHORITY_ATTR]
    temporal["source_rowset_ref"] = "/crop_runs/c1"
    temporal_digest = coordinate_audit._fingerprint(temporal)
    attrs[coordinate_audit.SOURCE_ROW_TEMPORAL_AUTHORITY_DIGEST_ATTR] = (
        temporal_digest
    )
    nodes["detect_runs/d1"] = replace(detect, attributes=attrs)
    nodes = _redigest_observation_record(
        nodes,
        owner_path="detect_runs/d1",
        attr_name=coordinate_audit.DETECTION_BBOX_PROJECTION_ATTR,
        mutate=lambda record: record["temporal_authority"].__setitem__(
            "record_sha256",
            temporal_digest,
        ),
        descriptor_paths=(surface_path,),
    )

    _surface_type, result = _classify_metadata_surface(nodes, surface_path)

    assert result["status"] == "ambiguous_fail_closed"
    assert "OBSERVATION_TEMPORAL_AUTHORITY_INVALID" in {
        issue["code"] for issue in result["issues"]
    }


@pytest.mark.parametrize("mutation", ["selection", "output_bbox"])
def test_crop_selection_payload_roles_fail_closed_after_valid_redigest(
    mutation: str,
) -> None:
    surface_path = "crop_runs/c1/bbox_img_xyxy"

    def mutate(record: dict[str, Any]) -> None:
        if mutation == "selection":
            record["selection"]["array_ref"] = (
                "/detect_runs/d1/source_acquisition_frame_index"
            )
        else:
            record["output_rowset"]["bbox_img_xyxy"]["array_ref"] = (
                "/detect_runs/d1/bbox_img_xyxy"
            )

    nodes = _redigest_observation_record(
        _canonical_detection_crop_metadata_nodes(),
        owner_path="crop_runs/c1",
        attr_name=coordinate_audit.CROP_GEOMETRY_SELECTION_ATTR,
        mutate=mutate,
        descriptor_paths=(surface_path,),
    )

    _surface_type, result = _classify_metadata_surface(nodes, surface_path)

    assert result["status"] == "ambiguous_fail_closed"
    assert "OBSERVATION_ARRAY_PAYLOAD_METADATA_INVALID" in {
        issue["code"] for issue in result["issues"]
    }


def test_crop_roi_direction_fails_closed_after_valid_redigest() -> None:
    surface_path = "crop_runs/c1/source_crop_xywh"
    nodes = _redigest_observation_record(
        _canonical_detection_crop_metadata_nodes(),
        owner_path="crop_runs/c1",
        attr_name=coordinate_audit.CROP_ROI_GEOMETRY_DERIVATION_ATTR,
        mutate=lambda record: record.__setitem__(
            "direction",
            "source_camera_image_px_to_roi_local_px",
        ),
        descriptor_paths=(surface_path,),
    )

    _surface_type, result = _classify_metadata_surface(nodes, surface_path)

    assert result["status"] == "ambiguous_fail_closed"
    assert "CROP_ROI_GEOMETRY_DERIVATION_INVALID" in {
        issue["code"] for issue in result["issues"]
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("wrong_attr", "REGISTERED_COORDINATE_RECORD_ATTR_INVALID"),
        ("wrong_version", "REGISTERED_COORDINATE_RECORD_SCHEMA_INVALID"),
        ("wrong_digest", "REGISTERED_COORDINATE_RECORD_DIGEST_MISMATCH"),
    ],
)
def test_registered_detection_record_attr_schema_and_digest_fail_closed(
    mutation: str,
    expected_code: str,
) -> None:
    surface_path = "detect_runs/d1/bbox_img_xyxy"
    nodes = _canonical_detection_crop_metadata_nodes()
    attr_name = coordinate_audit.DETECTION_BBOX_PROJECTION_ATTR
    if mutation == "wrong_version":
        nodes = _redigest_observation_record(
            nodes,
            owner_path="detect_runs/d1",
            attr_name=attr_name,
            mutate=lambda record: record.__setitem__("schema_version", 1.0),
            descriptor_paths=(surface_path,),
        )
    else:
        owner = nodes["detect_runs/d1"]
        owner_attrs = json.loads(json.dumps(owner.attributes))
        if mutation == "wrong_digest":
            owner_attrs[f"{attr_name}_sha256"] = "f" * 64
        else:
            record = owner_attrs.pop(attr_name)
            owner_attrs.pop(f"{attr_name}_sha256")
            wrong_attr = "forged_detection_bbox_projection"
            owner_attrs[wrong_attr] = record
            owner_attrs[f"{wrong_attr}_sha256"] = coordinate_audit._fingerprint(
                record
            )
            surface = nodes[surface_path]
            surface_attrs = json.loads(json.dumps(surface.attributes))
            descriptor = surface_attrs["coordinate_descriptor"]
            pointer = next(
                pointer
                for pointer in descriptor["lineage_refs"]
                if pointer["record_ref"] == f"/detect_runs/d1@{attr_name}"
            )
            pointer["record_ref"] = f"/detect_runs/d1@{wrong_attr}"
            surface_attrs["coordinate_descriptor_sha256"] = (
                canonical_coordinate_descriptor_v2_digest(descriptor)
            )
            nodes[surface_path] = replace(surface, attributes=surface_attrs)
        nodes["detect_runs/d1"] = replace(owner, attributes=owner_attrs)

    _surface_type, result = _classify_metadata_surface(nodes, surface_path)

    assert result["status"] == "ambiguous_fail_closed"
    assert expected_code in {issue["code"] for issue in result["issues"]}


def test_metadata_physical_chunk_iteration_is_bounded() -> None:
    assert coordinate_audit._metadata_physical_chunk_indices(
        [coordinate_audit._MAX_METADATA_PHYSICAL_CHUNK_GRID_ENTRIES + 1],
        {"physical_chunk_shape": [1]},
    ) is None


def test_acquisition_schema_hidden_under_wrong_attr_fails_inventory_closed(
) -> None:
    (
        acquisition_path,
        acquisition_attrs,
        _frame_path,
        _frame_attrs,
        _frame,
        _acquisition,
        _token,
        root_attrs,
    ) = _sealed_source_camera_frame_attrs()
    forged_attrs = {
        "wrong_acquisition_attr": acquisition_attrs[
            coordinate_audit.ACQUISITION_CAMERA_FRAME_ATTR
        ]
    }
    nodes = {
        ".": coordinate_audit.MetadataNode(
            relative_path=".",
            node_type="group",
            metadata_format="memory",
            shape=None,
            data_type=None,
            chunk_shape=None,
            storage_metadata=None,
            attributes=root_attrs,
        ),
        acquisition_path: coordinate_audit.MetadataNode(
            relative_path=acquisition_path,
            node_type="group",
            metadata_format="memory",
            shape=None,
            data_type=None,
            chunk_shape=None,
            storage_metadata=None,
            attributes=forged_attrs,
        ),
    }

    inventory, issues = coordinate_audit._dataset_acquisition_authority_inventory(
        nodes
    )

    assert inventory["inventory_status"] == "ambiguous_fail_closed"
    assert inventory["schema_attr_mismatches"][0]["expected_attribute"] == (
        coordinate_audit.ACQUISITION_CAMERA_FRAME_ATTR
    )
    assert "ACQUISITION_SCHEMA_ATTR_MISMATCH" in {
        issue["code"] for issue in issues
    }


def test_scanner_v11_versions_move_as_one_ruleset() -> None:
    assert {
        coordinate_audit.AUDIT_SCHEMA_VERSION,
        coordinate_audit.CHECKPOINT_SCHEMA_VERSION,
        coordinate_audit.ARTIFACT_SCHEMA_VERSION,
        coordinate_audit.AUDIT_RULESET_VERSION,
    } == {11}


def test_registry_is_query_only_and_every_row_is_preserved(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track_path = "analysis/track_kinematics_runs/offline/run/tracks/id_0/positions_px"
    _write_array(
        zarr_path,
        track_path,
        shape=[3, 2],
        attributes={
            **_descriptor_attrs(_complete_arena_descriptor(zarr_path)),
            "row_identity_ref": "frame_indices",
            "source_ref": "/coordinate_records/arena_geometry",
            "source_camera_overlay_suitable": False,
        },
    )
    _write_array(
        zarr_path,
        "analysis/track_kinematics_runs/offline/run/tracks/id_0/frame_indices",
        shape=[3],
    )

    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "good",
                "recording_id": "rec-1",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "unreachable",
                "recording_id": "rec-2",
                "zarr_path": str(tmp_path / "does-not-exist.zarr"),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "missing-path",
                "recording_id": "rec-3",
                "zarr_path": None,
                "zarr_use": "training",
                "artifact_kind": "derived_training_merge",
                "status": "missing",
            },
        ],
    )

    conn = open_registry_readonly(registry)
    try:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE datasets SET status = 'changed'")
    finally:
        conn.close()

    records = audit_registry(registry)
    datasets = _dataset_rows(records)
    assert [record["dataset_id"] for record in datasets] == ["good", "missing-path", "unreachable"]
    assert datasets[0]["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_AUTHORITY_MISSING" in datasets[0]["issue_codes"]
    assert datasets[1]["status"] == "missing_or_unreadable"
    assert datasets[2]["status"] == "missing_or_unreadable"

    surfaces = _surface_rows(records)
    assert len(surfaces) == 1
    assert surfaces[0]["surface_type"] == "track_positions_px"
    assert surfaces[0]["status"] == "numerical_validation_required"
    assert "LEGACY_ROW_IDENTITY_REQUIRES_MIGRATION" in surfaces[0]["issue_codes"]


def test_surface_families_and_fail_closed_classifications(tmp_path: Path) -> None:
    zarr_path = tmp_path / "families.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)

    simple_arrays = {
        "analysis/refined_online_runs/online/positions_px": "refined_online_positions_px",
        "analysis/detect_runs/detect/bbox_norm_coords": "detect_bbox",
        "analysis/refined_detect_runs/refined/instances/bbox_img_xyxy": "refined_detect_bbox",
        "analysis/crop_runs/crop/bbox_img_xyxy": "crop_geometry",
        "analysis/crop_runs/crop/roi_coordinates_full": "crop_geometry",
        "analysis/crop_runs/crop/roi_coordinates_ds": "crop_geometry",
        "analysis/keypoints_runs/kp/keypoints_roi": "keypoint_roi",
        "analysis/refined_keypoints_runs/kp/keypoints_img": "keypoint_source_image",
        "analysis/refined_keypoints_runs/kp/keypoints_norm": "keypoint_normalized",
        "analysis/subject_shape_runs/shape/centroid_xy": "subject_shape_geometry",
        "analysis/subject_shape_runs/shape/centerline_spline_xy": "subject_shape_geometry",
        "analysis/refined_subject_masks_runs/mask/masks_roi": "subject_mask_raster",
        # A geometry-looking name without producer/schema/role evidence is not
        # a coordinate surface.
        "analysis/new_geometry_stage/foo_bbox_vertices": None,
    }
    for relative_path in simple_arrays:
        _write_array(zarr_path, relative_path, shape=[2, 2])

    _ensure_groups(zarr_path, "analysis/stimulus_runs/stim/tracking_data/chaser_states/field")
    _write_node(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states",
        attributes={
            "coordinate_frame": "arena_relative_canvas_px",
            "units": "px",
            "coordinate_origin": "top_left_of_active_arena",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
            "reference_width": 800,
            "reference_height": 600,
            "reference_authority": "stimulus_runtime",
            "pixel_convention": "continuous",
            "geometry_convention": "xy_point",
            "source_ref": "stimulus_runtime",
            "source_camera_overlay_suitable": False,
            "field_names": ["stimulus_frame_num", "chaser_pos_x", "chaser_pos_y"],
        },
    )
    _write_node(zarr_path, "stimulus_runtime")
    _write_array(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states/chaser_pos_x",
        shape=[2],
    )
    _write_array(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states/stimulus_frame_num",
        shape=[2],
    )
    _write_array(
        zarr_path,
        "analysis/stimulus_runs/stim/tracking_data/chaser_states/target_clamped_pos_y",
        shape=[2],
    )

    _write_array(
        zarr_path,
        "analysis/calibration/homography_matrix",
        shape=[3, 3],
        attributes={"calibration_ref": "calibration/acquisition.json"},
    )

    # The historical online track-mm writer is an explicit recomputation class;
    # the audit does not guess from the numeric range.
    online_run = "analysis/track_kinematics_runs/online/run"
    _ensure_groups(zarr_path, f"{online_run}/tracks/id_0/positions_mm/value")
    _replace_group_attrs(
        zarr_path,
        online_run,
        {
            "method": "track_kinematics_online_refined",
            "coordinate_space": "texture",
            "pixel_to_mm": 12.0,
            "pixels_per_mm_projector": 12.0,
            "position_source_path": "analysis/refined_online_detect_runs/online",
        },
    )
    _write_array(
        zarr_path,
        f"{online_run}/tracks/id_0/positions_mm",
        shape=[2, 2],
        attributes={
            "source_ref": {
                "ref": _CANONICAL_LINEAGE_REF,
                "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
            }
        },
    )
    _write_array(zarr_path, f"{online_run}/tracks/id_0/frame_indices", shape=[2])

    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "all-families",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surfaces = _surface_rows(audit_registry(registry))
    surface_types = {str(record["surface_type"]) for record in surfaces}
    assert {value for value in simple_arrays.values() if value is not None} <= surface_types
    assert "stimulus_chaser_position" in surface_types
    assert "stimulus_target_clamped_position" in surface_types
    assert "calibration_homography" in surface_types
    assert "track_positions_mm" in surface_types
    assert sum(
        record["surface_type"] == "stimulus_chaser_position"
        for record in surfaces
    ) == 1

    chaser = next(
        record
        for record in surfaces
        if record["surface_type"] == "stimulus_chaser_position"
    )
    assert chaser["status"] == "ambiguous_fail_closed"
    assert "COORDINATE_RECORD_DIGEST_MISSING" in chaser["issue_codes"]

    homography = next(record for record in surfaces if record["surface_type"] == "calibration_homography")
    assert homography["status"] == "ambiguous_fail_closed"
    assert "HOMOGRAPHY_DIRECTION_MISSING" in homography["issue_codes"]

    online_mm = next(record for record in surfaces if record["surface_type"] == "track_positions_mm")
    assert online_mm["status"] == "ambiguous_fail_closed"
    assert "ONLINE_MM_CONVERSION_RECOMPUTATION_REQUIRED" in online_mm["issue_codes"]
    assert "LEGACY_SPACE_CONTEXT_INVALID" in online_mm["issue_codes"]

    assert not any(
        record["surface_path"].endswith("foo_bbox_vertices") for record in surfaces
    )


def test_offline_crop_camera_reconstruction_requires_numerical_validation(tmp_path: Path) -> None:
    zarr_path = tmp_path / "offline.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    run = "analysis/track_kinematics_runs/offline/run"
    _ensure_groups(zarr_path, f"{run}/tracks/id_0/positions_px/value")
    _replace_group_attrs(
        zarr_path,
        run,
        {
            "method": "track_kinematics_offline",
            "provenance": {
                "parameters": {"coordinate_space": "camera"},
                "inputs": {
                    "position_source_kind": "crop_rows",
                    "position_source_path": "crop_runs/crop",
                },
            }
        },
    )
    _write_array(
        zarr_path,
        f"{run}/tracks/id_0/positions_px",
        shape=[2, 2],
        attributes={
            "coordinate_space": "camera",
            "units": "px",
            "origin": "top_left",
            "positive_x_direction": "right",
            "positive_y_direction": "down",
            "reference_width": 640,
            "reference_height": 480,
            "reference_authority": "/raw_video/images_full.shape[-2:]",
            "pixel_convention": "pixel_center",
            "geometry_convention": "points_xy",
            "source_ref": {
                "ref": _CANONICAL_LINEAGE_REF,
                "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
            },
            "source_camera_overlay_suitable": True,
        },
    )
    _write_array(zarr_path, f"{run}/tracks/id_0/positions_mm", shape=[2, 2])
    _write_array(zarr_path, f"{run}/tracks/id_0/speed_raw_px", shape=[2])
    _write_array(zarr_path, f"{run}/tracks/id_0/speed_raw_mm", shape=[2])
    _write_array(zarr_path, f"{run}/tracks/id_0/frame_indices", shape=[2])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "offline",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = next(
        record
        for record in _surface_rows(audit_registry(registry))
        if record["surface_type"] == "track_positions_px"
    )
    assert surface["status"] == "ambiguous_fail_closed"
    assert "OFFLINE_CROP_SOURCE_RECONSTRUCTION_NUMERICAL_VALIDATION_REQUIRED" in surface["issue_codes"]
    assert "LEGACY_NUMERICAL_INVARIANTS_MISSING" in surface["issue_codes"]
    positions_mm = next(
        record
        for record in _surface_rows(audit_registry(registry))
        if record["surface_type"] == "track_positions_mm"
    )
    assert coordinate_audit._STATUS_PRIORITY[positions_mm["status"]] >= (
        coordinate_audit._STATUS_PRIORITY["numerical_validation_required"]
    )
    assert "UPSTREAM_POSITION_RISK_PROPAGATED" in positions_mm["issue_codes"]
    migration = coordinate_audit.build_migration_manifest(audit_registry(registry))
    derived = {
        row["surface_path"]: row
        for row in migration
        if row["target_kind"] == "derived_surface"
    }
    assert derived[f"{run}/tracks/id_0/speed_raw_px"]["migration_class"] == (
        "ambiguous_fail_closed"
    )
    assert derived[f"{run}/tracks/id_0/speed_raw_mm"]["must_fail_closed"] is True


def test_historical_value_risks_require_exact_producer_signatures(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "producer-signatures.zarr"
    _write_node(zarr_path)

    online_cases = {
        "exact": {
            "method": "track_kinematics_online_refined",
            "coordinate_space": "texture",
            "pixel_to_mm": 12.0,
            "pixels_per_mm_projector": 12.0,
        },
        "near_method": {
            "method": "track_kinematics_online",
            "coordinate_space": "texture",
            "pixel_to_mm": 12.0,
            "pixels_per_mm_projector": 12.0,
        },
        "camera_space": {
            "method": "track_kinematics_online_refined",
            "coordinate_space": "camera",
            "pixel_to_mm": 12.0,
            "pixels_per_mm_projector": 12.0,
        },
        "reciprocal": {
            "method": "track_kinematics_online_refined",
            "coordinate_space": "texture",
            "pixel_to_mm": 0.1,
            "pixels_per_mm_projector": 10.0,
            "mm_per_pixel": 0.1,
        },
    }
    online_paths: dict[str, str] = {}
    for name, attrs in online_cases.items():
        run = f"analysis/track_kinematics_runs/online/{name}"
        surface_path = f"{run}/tracks/id_0/positions_mm"
        _ensure_groups(zarr_path, surface_path)
        _replace_group_attrs(zarr_path, run, attrs)
        _write_array(zarr_path, surface_path, shape=[2, 2])
        online_paths[name] = surface_path

    offline_cases = {
        "exact": {
            "method": "track_kinematics_offline",
            "coordinate_space": "camera",
            "position_source_kind": "crop_rows",
            "position_source_path": "analysis/crop_runs/source",
        },
        "near_method": {
            "method": "track_kinematics_offline_legacy",
            "coordinate_space": "camera",
            "position_source_kind": "crop_rows",
            "position_source_path": "analysis/crop_runs/source",
        },
        "not_crop_family": {
            "method": "track_kinematics_offline",
            "coordinate_space": "camera",
            "position_source_kind": "crop_rows",
            "position_source_path": "analysis/not_crop_runs/source",
        },
        "source_geometry": {
            "method": "track_kinematics_offline",
            "coordinate_space": "camera",
            "position_source_kind": "crop_rows",
            "position_source_path": "analysis/crop_runs/source",
            "position_geometry_path": (
                "analysis/crop_runs/source/bbox_img_xyxy"
            ),
        },
    }
    offline_paths: dict[str, str] = {}
    for name, attrs in offline_cases.items():
        run = f"analysis/track_kinematics_runs/offline/{name}"
        surface_path = f"{run}/tracks/id_0/positions_px"
        _ensure_groups(zarr_path, surface_path)
        _replace_group_attrs(zarr_path, run, attrs)
        _write_array(zarr_path, surface_path, shape=[2, 2])
        offline_paths[name] = surface_path

    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    online_results = {
        name: coordinate_audit._legacy_online_mm_requires_recompute(
            "track_positions_mm",
            nodes[path],
            nodes,
            None,
        )[0]
        for name, path in online_paths.items()
    }
    offline_results = {
        name: coordinate_audit._offline_crop_reconstruction_requires_recompute(
            "track_positions_px",
            nodes[path],
            nodes,
            None,
        )[0]
        for name, path in offline_paths.items()
    }
    assert online_results == {
        "exact": True,
        "near_method": False,
        "camera_space": False,
        "reciprocal": False,
    }
    assert offline_results == {
        "exact": True,
        "near_method": False,
        "not_crop_family": False,
        "source_geometry": False,
    }


def test_outputs_are_deterministic_and_resume_preserves_complete_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "empty.zarr"
    _write_node(zarr_path)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "empty",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    records = audit_registry(registry)
    summary = summarize(records)
    assert summary["dataset_row_count"] == 1
    assert summary["surface_count"] == 0

    first_jsonl = tmp_path / "first.jsonl"
    second_jsonl = tmp_path / "second.jsonl"
    csv_path = tmp_path / "inventory.csv"
    markdown_path = tmp_path / "inventory.md"
    summary_path = tmp_path / "summary.json"
    write_jsonl(first_jsonl, records)
    resumed = audit_registry(registry, resume_jsonl=first_jsonl)
    write_jsonl(second_jsonl, resumed)
    write_csv(csv_path, records)
    write_markdown(markdown_path, records, summary)
    write_summary(summary_path, summary)

    assert resumed == records
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()
    assert csv_path.read_text(encoding="utf-8").startswith("record_type,dataset_key")
    assert "# Coordinate contract inventory" in markdown_path.read_text(encoding="utf-8")
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary

    # Root metadata is unchanged, but a child surface changed.  Resume must
    # invalidate from the full metadata inventory digest rather than silently
    # reusing the registry/root fingerprint.
    _write_array(
        zarr_path,
        "analysis/detect_runs/new/bbox_norm_coords",
        shape=[1, 4],
    )
    changed = audit_registry(registry, resume_jsonl=first_jsonl)
    assert summarize(changed)["surface_count"] == 1
    assert changed != records


def test_source_change_during_scan_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "changing.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track_path = "analysis/track_kinematics_runs/offline/run/tracks/id_0/positions_px"
    _write_array(
        zarr_path,
        track_path,
        shape=[1, 2],
        attributes=_descriptor_attrs(_complete_arena_descriptor(zarr_path)),
    )
    _write_array(
        zarr_path,
        "analysis/track_kinematics_runs/offline/run/tracks/id_0/frame_indices",
        shape=[1],
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "changing",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    first_snapshot = list(coordinate_audit.iter_metadata_nodes(zarr_path))
    second_snapshot = [replace(first_snapshot[0], attributes={"changed": True}), *first_snapshot[1:]]
    snapshots = iter((first_snapshot, second_snapshot))

    def _changing_nodes(_path: Path):
        yield from next(snapshots)

    monkeypatch.setattr(coordinate_audit, "iter_metadata_nodes", _changing_nodes)
    records = audit_registry(registry)
    dataset = _dataset_rows(records)[0]
    assert dataset["status"] == "missing_or_unreadable"
    assert dataset["scan_complete"] is False
    assert "SOURCE_CHANGED_DURING_SCAN" in dataset["issue_codes"]
    surface = _surface_rows(records)[0]
    assert surface["status"] == "missing_or_unreadable"
    assert surface["scan_snapshot_valid"] is False
    assert coordinate_audit.build_migration_manifest(records)[0]["must_fail_closed"] is True


def test_metadata_walker_treats_arrays_as_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "array-leaf.zarr"
    _write_node(zarr_path)
    array_path = "analysis/detect_runs/run/bbox_norm_coords"
    _write_array(zarr_path, array_path, shape=[2, 4])
    array_directory = zarr_path / array_path
    original_iterdir = Path.iterdir

    def _guarded_iterdir(path: Path):
        if path == array_directory:
            raise AssertionError("array chunk directories must not be enumerated")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _guarded_iterdir)
    nodes = list(coordinate_audit.iter_metadata_nodes(zarr_path))
    assert any(node.relative_path == array_path for node in nodes)


def test_metadata_walker_rejects_symlinks_cycles_and_implicit_subtrees(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "root-log-sidecar.zarr"
    _write_node(sidecar)
    logs = sidecar / "logs" / "crop_batch"
    logs.mkdir(parents=True)
    (logs / "task.jsonl").write_text("{}\n", encoding="utf-8")
    hidden_control = sidecar / "analysis" / ".failed"
    _write_node(sidecar, "analysis")
    hidden_control.mkdir()
    assert [node.relative_path for node in coordinate_audit.iter_metadata_nodes(sidecar)] == [
        ".",
        "analysis",
    ]

    cyclic = tmp_path / "cyclic.zarr"
    _write_node(cyclic)
    (cyclic / "loop").symlink_to(cyclic, target_is_directory=True)
    with pytest.raises(
        coordinate_audit.MetadataTraversalError,
        match="symlinked metadata child is forbidden",
    ):
        list(coordinate_audit.iter_metadata_nodes(cyclic))

    escaped = tmp_path / "escaped.zarr"
    external = tmp_path / "external.zarr"
    _write_node(escaped)
    _write_node(external)
    (escaped / "external").symlink_to(external, target_is_directory=True)
    with pytest.raises(
        coordinate_audit.MetadataTraversalError,
        match="symlinked metadata child is forbidden",
    ):
        list(coordinate_audit.iter_metadata_nodes(escaped))

    implicit = tmp_path / "implicit.zarr"
    _write_node(implicit)
    _write_node(implicit, "implicit_group/coordinate_array", node_type="array", shape=[1, 2])
    with pytest.raises(
        coordinate_audit.MetadataTraversalError,
        match="lacks explicit Zarr node metadata",
    ):
        list(coordinate_audit.iter_metadata_nodes(implicit))

    linked_metadata = tmp_path / "linked-metadata.zarr"
    linked_metadata.mkdir()
    external_metadata = tmp_path / "external-zarr.json"
    external_metadata.write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
        encoding="utf-8",
    )
    (linked_metadata / "zarr.json").symlink_to(external_metadata)
    with pytest.raises(
        coordinate_audit.MetadataTraversalError,
        match="symlinked metadata file is forbidden",
    ):
        list(coordinate_audit.iter_metadata_nodes(linked_metadata))


def test_dataset_checkpoints_survive_interruption_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_zarr = tmp_path / "first.zarr"
    second_zarr = tmp_path / "second.zarr"
    _write_node(first_zarr)
    _write_node(second_zarr)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "first",
                "recording_id": "rec-1",
                "zarr_path": str(first_zarr),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "second",
                "recording_id": "rec-2",
                "zarr_path": str(second_zarr),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
        ],
    )
    checkpoint_dir = tmp_path / "checkpoints"
    original_audit_dataset_row = coordinate_audit.audit_dataset_row
    interrupted_calls: list[str] = []

    def _interrupt_second(row, **kwargs):
        dataset_id = str(row["dataset_id"])
        interrupted_calls.append(dataset_id)
        if dataset_id == "second":
            raise RuntimeError("simulated interruption")
        return original_audit_dataset_row(row, **kwargs)

    monkeypatch.setattr(coordinate_audit, "audit_dataset_row", _interrupt_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        audit_registry(registry, checkpoint_dir=checkpoint_dir)

    checkpoint_paths = sorted(checkpoint_dir.glob("dataset-*.json"))
    assert len(checkpoint_paths) == 1
    assert not list(checkpoint_dir.glob("*.tmp"))
    checkpoint = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    assert checkpoint["dataset_key"] == "first"
    assert checkpoint["records"][0]["scan_complete"] is True

    resumed_calls: list[str] = []

    def _record_new_scan(row, **kwargs):
        resumed_calls.append(str(row["dataset_id"]))
        return original_audit_dataset_row(row, **kwargs)

    monkeypatch.setattr(coordinate_audit, "audit_dataset_row", _record_new_scan)
    records = audit_registry(registry, checkpoint_dir=checkpoint_dir)
    assert resumed_calls == ["second"]
    assert [record["dataset_id"] for record in _dataset_rows(records)] == [
        "first",
        "second",
    ]
    assert len(list(checkpoint_dir.glob("dataset-*.json"))) == 2


def test_normalized_artifact_set_is_complete_and_deterministic(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track_path = "analysis/track_kinematics_runs/offline/run/tracks/id_0/positions_px"
    _write_array(
        zarr_path,
        track_path,
        shape=[2, 2],
        attributes=_descriptor_attrs(_complete_arena_descriptor(zarr_path)),
    )
    _write_array(
        zarr_path,
        "analysis/track_kinematics_runs/offline/run/tracks/id_0/frame_indices",
        shape=[2],
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "analysis",
                "recording_id": "rec-1",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "missing",
                "recording_id": "rec-2",
                "zarr_path": str(tmp_path / "missing.zarr"),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
        ],
    )
    records = audit_registry(registry)
    first_dir = tmp_path / "artifacts-first"
    second_dir = tmp_path / "artifacts-second"
    coordinate_audit.write_normalized_artifacts(first_dir, registry, records)
    coordinate_audit.write_normalized_artifacts(second_dir, registry, records)

    assert {path.name for path in first_dir.iterdir()} == set(
        coordinate_audit.NORMALIZED_ARTIFACT_FILENAMES
    )
    for filename in coordinate_audit.NORMALIZED_ARTIFACT_FILENAMES:
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    coordinate_audit.verify_normalized_artifact_generation(first_dir)

    snapshot = json.loads((first_dir / "registry_snapshot.json").read_text(encoding="utf-8"))
    coverage = json.loads((first_dir / "coverage.json").read_text(encoding="utf-8"))
    targets = [
        json.loads(line)
        for line in (first_dir / "targets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    migration = [
        json.loads(line)
        for line in (first_dir / "migration_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert snapshot["dataset_row_count"] == 2
    assert len(targets) == 2
    assert coverage["registry_dataset_row_count"] == 2
    assert coverage["represented_dataset_row_count"] == 2
    assert coverage["important_coordinate_surface_count"] == 1
    assert {row["migration_class"] for row in migration} == {
        "ambiguous_fail_closed",
        "no_change",
        "numerical_validation_required",
        "missing_or_unreadable_fail_closed",
    }
    required_migration_evidence = {
        "evidence_paths",
        "evidence_sha256",
        "validation_tool_commit",
        "validation_result",
        "values_changed",
        "previous_state",
        "previous_state_sha256",
        "result_state",
        "result_state_sha256",
    }
    assert all(required_migration_evidence <= set(row) for row in migration)
    assert all(
        row["previous_state_sha256"]
        == coordinate_audit._fingerprint(row["previous_state"])
        and row["result_state_sha256"]
        == coordinate_audit._fingerprint(row["result_state"])
        for row in migration
    )
    assert (first_dir / "issue_summary.csv").read_text(encoding="utf-8").startswith(
        "issue_code,severity,occurrence_count"
    )
    (first_dir / "targets.jsonl").write_text("mixed generation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        coordinate_audit.verify_normalized_artifact_generation(first_dir)


def test_filters_keep_full_registry_recording_and_dataset_coverage_visible(
    tmp_path: Path,
) -> None:
    selected_zarr = tmp_path / "selected.zarr"
    other_zarr = tmp_path / "other.zarr"
    null_link_zarr = tmp_path / "null-link.zarr"
    for zarr_path in (selected_zarr, other_zarr, null_link_zarr):
        _write_node(zarr_path)
    _write_contract_evidence(selected_zarr)
    track_path = "analysis/track_kinematics_runs/offline/run/tracks/id_0/positions_px"
    _write_array(
        selected_zarr,
        track_path,
        attributes=_descriptor_attrs(_complete_arena_descriptor(selected_zarr)),
        shape=[1, 2],
    )
    _write_array(
        selected_zarr,
        "analysis/track_kinematics_runs/offline/run/tracks/id_0/frame_indices",
        shape=[1],
    )
    _write_array(
        selected_zarr,
        "analysis/detect_runs/run/bbox_norm_coords",
        shape=[1, 4],
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "selected",
                "recording_id": "rec-a",
                "zarr_path": str(selected_zarr),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "other",
                "recording_id": "rec-b",
                "zarr_path": str(other_zarr),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "null-link",
                "recording_id": None,
                "zarr_path": str(null_link_zarr),
                "zarr_use": "training",
                "artifact_kind": "derived_training_merge",
                "status": "active",
            },
        ],
        recording_rows=[
            {
                "recording_id": "rec-a",
                "recording_path": "/recordings/shared",
                "session_uuid": "session-a",
            },
            {
                "recording_id": "rec-b",
                "recording_path": "/recordings/shared",
                "session_uuid": "session-b",
            },
            {
                "recording_id": "rec-orphan",
                "recording_path": "/recordings/orphan",
                "session_uuid": "session-orphan",
            },
        ],
    )
    records = audit_registry(
        registry,
        recording_ids=["rec-a"],
        recording_path_contains=["shared"],
        run_families=["track_kinematics_runs"],
    )
    assert [record["dataset_id"] for record in _dataset_rows(records)] == ["selected"]
    assert {
        record["surface_type"] for record in _surface_rows(records)
    } == {"track_positions_px"}
    assert _dataset_rows(records)[0]["discovered_surface_count"] == 2

    artifact_dir = tmp_path / "filtered-artifacts"
    coordinate_audit.write_normalized_artifacts(artifact_dir, registry, records)
    snapshot = json.loads((artifact_dir / "registry_snapshot.json").read_text(encoding="utf-8"))
    coverage = json.loads((artifact_dir / "coverage.json").read_text(encoding="utf-8"))
    assert snapshot["recording_row_count"] == 3
    assert snapshot["dataset_row_count"] == 3
    assert snapshot["selected_dataset_row_count"] == 1
    assert snapshot["recording_ids_without_dataset"] == ["rec-orphan"]
    assert snapshot["dataset_ids_without_recording_id"] == ["null-link"]
    assert snapshot["duplicate_recording_paths"] == [
        {
            "recording_ids": ["rec-a", "rec-b"],
            "recording_path": "/recordings/shared",
            "recording_row_count": 2,
        }
    ]
    assert coverage["registry_recording_row_count"] == 3
    assert coverage["registry_dataset_row_count"] == 3
    assert coverage["selected_dataset_row_count"] == 1
    assert coverage["represented_dataset_row_count"] == 1
    assert coverage["all_selected_dataset_rows_represented"] is True
    assert coverage["all_registry_rows_represented"] is False
    assert coverage["recording_ids_without_dataset"] == ["rec-orphan"]
    assert coverage["dataset_ids_without_recording_id"] == ["null-link"]


def test_metadata_traversal_and_malformed_root_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = tmp_path / "unreadable.zarr"
    _write_node(unreadable)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "unreadable",
                "recording_id": "rec",
                "zarr_path": str(unreadable),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    original_iterdir = Path.iterdir

    def _permission_denied(path: Path):
        if path == unreadable:
            raise PermissionError("denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _permission_denied)
    dataset = _dataset_rows(audit_registry(registry))[0]
    assert dataset["status"] == "missing_or_unreadable"
    assert dataset["scan_complete"] is False
    assert "ZARR_METADATA_TRAVERSAL_FAILED" in dataset["issue_codes"]

    monkeypatch.setattr(Path, "iterdir", original_iterdir)
    (unreadable / "zarr.json").write_text("{invalid", encoding="utf-8")
    dataset = _dataset_rows(audit_registry(registry))[0]
    assert dataset["status"] == "missing_or_unreadable"
    assert dataset["scan_complete"] is False
    assert "INVALID_ZARR_METADATA_INVENTORY" in dataset["issue_codes"]


def test_descriptor_digest_row_identity_and_lineage_are_resolved(tmp_path: Path) -> None:
    zarr_path = tmp_path / "contracts.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    descriptor = _complete_xy_descriptor()
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[2, 2],
        attributes={"coordinate_descriptor": descriptor},
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "contracts",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] != "compatible"
    assert "COORDINATE_DESCRIPTOR_DIGEST_MISSING" in surface["issue_codes"]
    assert "ROW_IDENTITY_LENGTH_MISMATCH" in surface["issue_codes"]

    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes={
            "coordinate_descriptor": descriptor,
            "coordinate_descriptor_sha256": "0" * 64,
        },
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "COORDINATE_DESCRIPTOR_DIGEST_MISMATCH" in surface["issue_codes"]

    broken = dict(descriptor)
    broken["lineage_refs"] = [{"ref": "/missing/source"}]
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_unvalidated_descriptor_attrs(broken),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] != "compatible"
    assert "COORDINATE_RECORD_REF_UNRESOLVED" in surface["issue_codes"]


def test_canonical_compatibility_requires_bound_lineage_and_exact_extent_authority(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "exact-evidence.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "exact-evidence",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    unbound = json.loads(json.dumps(_complete_xy_descriptor()))
    del unbound["lineage_refs"][0]["sha256"]
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_unvalidated_descriptor_attrs(unbound),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "COORDINATE_RECORD_DIGEST_MISSING" in surface["issue_codes"]

    wrong_extent = json.loads(json.dumps(_complete_xy_descriptor()))
    wrong_extent["reference_extent"]["width"] = 641
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(wrong_extent),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_EXTENT_MISMATCH" in surface["issue_codes"]

    unsupported = json.loads(json.dumps(_complete_xy_descriptor()))
    unsupported["reference_extent"]["authority"] = "/raw_video/images_full"
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(unsupported),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED" in surface["issue_codes"]


def test_invalid_directed_transform_reference_fails_descriptor_compatibility(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "transform.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    transform_path = "analysis/calibration/homography_matrix"
    _write_array(
        zarr_path,
        transform_path,
        shape=[3, 3],
        attributes={"directed_transform": {"schema_id": "broken"}},
    )
    descriptor = _complete_xy_descriptor()
    descriptor["transform_refs"] = [{"ref": f"/{transform_path}"}]
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_unvalidated_descriptor_attrs(descriptor),
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "transform",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = next(
        record
        for record in _surface_rows(audit_registry(registry))
        if record["surface_type"] == "track_positions_px"
    )
    assert surface["status"] != "compatible"
    assert "DIRECTED_TRANSFORM_REF_INVALID" in surface["issue_codes"]


def test_transform_refs_require_directed_surface_bound_chains(tmp_path: Path) -> None:
    zarr_path = tmp_path / "directed-transform.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    roi_authority = "/crop_runs/crop/roi_images.shape[-2:]"
    _write_array(zarr_path, "crop_runs/crop/roi_images", shape=[1, 256, 512])
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])

    camera_calibration_path = _write_selected_camera_authority(
        zarr_path,
        selection_name="selected-camera-1",
        camera_id="camera-1",
        width=640,
        height=480,
    )

    source_extent = {
        "width": 640,
        "height": 480,
        "units": "px",
        "authority": (
            f"/{camera_calibration_path}@native_width_px,native_height_px"
        ),
    }
    target_extent = {
        "width": 512,
        "height": 256,
        "units": "px",
        "authority": roi_authority,
    }
    forward = build_directed_homography(
        transform_id="camera_to_roi",
        matrix=np.eye(3),
        from_space_id="source_camera_image_px",
        to_space_id="roi_local_px",
        source_reference_extent=source_extent,
        target_reference_extent=target_extent,
        calibration_ref=camera_calibration_path,
        camera_id="camera-1",
    )
    forward_path = "coordinate_records/camera_to_roi"
    _write_array(
        zarr_path,
        forward_path,
        shape=[3, 3],
        attributes=directed_homography_attrs(forward),
    )

    descriptor = {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 1,
        "space_id": "roi_local_px",
        "geometry_type": "points_xy",
        "components": ["x", "y"],
        "component_units": ["px", "px"],
        "origin": "top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "reference_extent": target_extent,
        "pixel_convention": "pixel_center",
        "row_identity": {
            "mode": "track_frame_indices",
            "array_ref": "frame_indices",
        },
        "source_camera_overlay": "requires_transform",
        "lineage_refs": [
            {
                "ref": _CANONICAL_LINEAGE_REF,
                "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
            }
        ],
        "transform_refs": [{"ref": f"/{forward_path}", "sha256": forward.digest()}],
    }
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "directed-transform",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_TARGET_INVALID" in surface["issue_codes"]
    assert "ROI_CROP_PLACEMENT_LINEAGE_MISSING" in surface["issue_codes"]

    reverse = build_directed_homography(
        transform_id="roi_to_camera",
        matrix=np.eye(3),
        from_space_id="roi_local_px",
        to_space_id="source_camera_image_px",
        source_reference_extent=target_extent,
        target_reference_extent=source_extent,
        calibration_ref=camera_calibration_path,
        camera_id="camera-1",
    )
    reverse_path = "coordinate_records/roi_to_camera"
    _write_array(
        zarr_path,
        reverse_path,
        shape=[3, 3],
        attributes=directed_homography_attrs(reverse),
    )
    descriptor["transform_refs"] = [
        {"ref": f"/{reverse_path}", "sha256": reverse.digest()}
    ]
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "TRANSFORM_DIRECTION_INCOMPATIBLE_WITH_SURFACE" in surface["issue_codes"]
    assert "TRANSFORM_CHAIN_DISCONNECTED_OR_REVERSED" in surface["issue_codes"]

    mismatched_target_extent = dict(target_extent)
    mismatched_target_extent["width"] = 511
    mismatched = build_directed_homography(
        transform_id="camera_to_wrong_roi_extent",
        matrix=np.eye(3),
        from_space_id="source_camera_image_px",
        to_space_id="roi_local_px",
        source_reference_extent=source_extent,
        target_reference_extent=mismatched_target_extent,
        calibration_ref=camera_calibration_path,
        camera_id="camera-1",
    )
    mismatched_path = "coordinate_records/camera_to_wrong_roi_extent"
    _write_array(
        zarr_path,
        mismatched_path,
        shape=[3, 3],
        attributes=directed_homography_attrs(mismatched),
    )
    descriptor["transform_refs"] = [
        {"ref": f"/{mismatched_path}", "sha256": mismatched.digest()}
    ]
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "TRANSFORM_TARGET_EXTENT_MISMATCH" in surface["issue_codes"]

    legacy_record = {"scale": 2.0, "direction": "source_to_target"}
    _write_node(
        zarr_path,
        "coordinate_records/legacy_transform",
        attributes={"mapping": legacy_record},
    )
    descriptor["transform_refs"] = [
        {
            "ref": "/coordinate_records/legacy_transform.attrs[mapping]",
            "sha256": coordinate_audit._fingerprint(legacy_record),
        }
    ]
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "TRANSFORM_REF_NOT_DIRECTION_EXPLICIT" in surface["issue_codes"]


def test_physical_mm_requires_exact_digest_bound_frame_authority(tmp_path: Path) -> None:
    zarr_path = tmp_path / "physical-frame.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    physical_record = {
        "schema_id": "palette.physical_coordinate_frame",
        "schema_version": 1,
        "physical_frame_id": "arena-1-mm",
        "units": "mm",
        "origin": "physical_frame_origin",
        "positive_directions": {"x": "right", "y": "down"},
    }
    physical_ref = "/coordinate_records/physical.attrs[arena_frame]"
    _write_node(
        zarr_path,
        "coordinate_records/physical",
        attributes={"arena_frame": physical_record},
    )
    _write_node(zarr_path, "coordinate_records/arbitrary_existing_node")
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    descriptor = {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 1,
        "space_id": "physical_mm",
        "geometry_type": "points_xy",
        "components": ["x", "y"],
        "component_units": ["mm", "mm"],
        "origin": "physical_frame_origin",
        "positive_directions": {"x": "right", "y": "down"},
        "reference_extent": {
            "width": None,
            "height": None,
            "units": "not_applicable",
            "authority": physical_ref,
        },
        "pixel_convention": "continuous",
        "row_identity": {
            "mode": "track_frame_indices",
            "array_ref": "frame_indices",
        },
        "source_camera_overlay": "not_suitable",
        "physical_frame": "arena-1-mm",
        "lineage_refs": [
            {
                "ref": _CANONICAL_LINEAGE_REF,
                "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
            },
            {
                "ref": physical_ref,
                "sha256": coordinate_audit._fingerprint(physical_record),
            },
        ],
    }
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "physical-frame",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED" in surface["issue_codes"]

    calibration_path = "coordinate_records/physical_calibration"
    _write_node(
        zarr_path,
        calibration_path,
        attributes={
            "schema_id": "test.physical_scale_calibration",
            "schema_version": 1,
            "pixels_per_mm": 10.0,
        },
    )
    calibration_node = next(
        node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
        if node.relative_path == calibration_path
    )
    physical_record.update(
        {
            "source_space_id": "arena_relative_canvas_px",
            "source_reference_authority": (
                "/coordinate_records/arena_geometry.attrs"
            ),
            "calibration_ref": f"/{calibration_path}",
            "calibration_sha256": coordinate_audit._metadata_node_record_digest(
                calibration_node
            ),
            "pixels_per_mm": 10.0,
            "mm_per_pixel": 0.1,
            "reciprocal_derivation": (
                "mm_per_pixel_reciprocal_of_pixels_per_mm_v1"
            ),
        }
    )
    _write_node(
        zarr_path,
        "coordinate_records/physical",
        attributes={"arena_frame": physical_record},
    )
    descriptor["lineage_refs"][1] = {
        "ref": physical_ref,
        "sha256": coordinate_audit._fingerprint(physical_record),
    }
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "numerical_validation_required"
    assert "LEGACY_ROW_IDENTITY_REQUIRES_MIGRATION" in surface["issue_codes"]

    descriptor["lineage_refs"] = descriptor["lineage_refs"][:1]
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "PHYSICAL_FRAME_LINEAGE_MISSING" in surface["issue_codes"]

    descriptor["lineage_refs"] = [
        descriptor["lineage_refs"][0],
        {"ref": physical_ref, "sha256": "0" * 64},
    ]
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert "PHYSICAL_FRAME_LINEAGE_DIGEST_MISMATCH" in surface["issue_codes"]

    descriptor["reference_extent"]["authority"] = (
        "/coordinate_records/arbitrary_existing_node"
    )
    descriptor["lineage_refs"] = descriptor["lineage_refs"][:1]
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED" in surface["issue_codes"]

    descriptor["reference_extent"]["authority"] = physical_ref
    descriptor["lineage_refs"] = [
        descriptor["lineage_refs"][0],
        {
            "ref": physical_ref,
            "sha256": coordinate_audit._fingerprint(physical_record),
        },
    ]
    descriptor["physical_frame"] = "different-arena"
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert "PHYSICAL_FRAME_RECORD_MISMATCH" in surface["issue_codes"]


def test_canonical_physical_profile_preserves_source_camera_directions(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "canonical-physical.zarr"
    _write_node(zarr_path)
    world = _sealed_physical_frame_attrs()
    _merge_node_attrs(zarr_path, ".", dict(world["root_attrs"]))
    for path_key, attrs_key in (
        ("acquisition_path", "acquisition_attrs"),
        ("source_path", "source_attrs"),
        ("selected_path", "selected_attrs"),
        ("physical_path", "physical_attrs"),
    ):
        path = str(world[path_key])
        _ensure_groups(zarr_path, path)
        _write_node(zarr_path, path, attributes=dict(world[attrs_key]))

    rowset = "analysis/refined_online_runs/run"
    _write_canonical_row_identity(
        zarr_path,
        rowset,
        domain=STIMULUS_STATE_DOMAIN,
        row_count=2,
    )
    physical = world["physical"]
    physical_ref = (
        f"/{world['physical_path']}@{PHYSICAL_FRAME_CALIBRATION_ATTR}"
    )
    authority = DigestBoundCoordinateRecordRef(
        record_ref=physical_ref,
        record_sha256=physical.record_sha256,
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id=PHYSICAL_SOURCE_CAMERA_PROFILE_ID,
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("mm", "mm"),
        reference_width=physical.reference_width,
        reference_height=physical.reference_height,
        reference_authority=authority,
        reference_selector="record",
        pixel_convention="not_applicable",
        row_identity_contract=build_row_identity_contract(
            domain=STIMULUS_STATE_DOMAIN,
            values=np.arange(2, dtype=np.int64),
            components=("stimulus_state_index",),
        ),
        row_identity_record_ref=f"/{rowset}@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=CanonicalFrameRecord(
            kind=PHYSICAL_FRAME_CALIBRATION_KIND,
            record_ref=physical_ref,
            record_sha256=physical.record_sha256,
        ),
    )
    surface_path = f"{rowset}/positions_mm"
    _write_array(
        zarr_path,
        surface_path,
        shape=[2, 2],
        attributes=canonical_coordinate_descriptor_v2_attrs(descriptor),
    )
    registry = _make_registry(
        tmp_path / "canonical-physical.sqlite",
        [
            {
                "dataset_id": "canonical-physical",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["coordinate_descriptor"]["profile_id"] == (
        PHYSICAL_SOURCE_CAMERA_PROFILE_ID
    )
    assert surface["status"] == "numerical_validation_required"
    assert not any(
        issue["severity"] in {"error", "critical"}
        for issue in surface["issues"]
    )

    _merge_node_attrs(
        zarr_path,
        rowset,
        {
            "method": "track_kinematics_online_refined",
            "coordinate_space": "texture",
            "pixel_to_mm": 12.0,
            "pixels_per_mm_projector": 12.0,
        },
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "ONLINE_MM_CONVERSION_RECOMPUTATION_REQUIRED" in surface["issue_codes"]

    validation_path = "analysis/coordinate_validations/online_mm"
    validation_record = {
        "schema_id": "palette.coordinate_value_validation",
        "schema_version": 1,
        "validation_kind": "online_mm_conversion_values_v1",
        "producer_signature": (
            "track_kinematics_online_refined.corrected.v1"
        ),
        "surface_ref": f"/{surface_path}@array_values",
        "values_sha256": "b" * 64,
        "checks": {
            "camera_scale_reciprocal_verified": True,
            "source_pixel_values_recomputed": True,
            "output_values_match_recomputation": True,
            "row_identity_verified": True,
        },
        "result": "pass",
        "validator_commit": "unit-test-validator",
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    _ensure_groups(zarr_path, validation_path)
    _write_node(
        zarr_path,
        validation_path,
        attributes={
            "coordinate_value_validation": validation_record,
            "coordinate_value_validation_sha256": (
                coordinate_audit._fingerprint(validation_record)
            ),
        },
    )
    _merge_node_attrs(
        zarr_path,
        surface_path,
        {
            "coordinate_value_validation_ref": {
                "record_ref": (
                    f"/{validation_path}@coordinate_value_validation"
                ),
                "record_sha256": coordinate_audit._fingerprint(
                    validation_record
                ),
            }
        },
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "numerical_validation_required"
    assert "ONLINE_MM_CONVERSION_RECOMPUTATION_REQUIRED" not in surface["issue_codes"]
    assert "COORDINATE_VALUE_VALIDATION_PAYLOAD_CHECK_REQUIRED" in surface[
        "issue_codes"
    ]

    arena_descriptor = descriptor.to_dict()
    arena_descriptor["profile_id"] = "physical_mm.arena_y_down.v1"
    _write_array(
        zarr_path,
        surface_path,
        shape=[2, 2],
        attributes={
            "coordinate_descriptor": arena_descriptor,
            "coordinate_descriptor_sha256": (
                canonical_coordinate_descriptor_v2_digest(arena_descriptor)
            ),
        },
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "PHYSICAL_ARENA_TRANSFORM_AUTHORITY_REQUIRED" in surface["issue_codes"]


def test_generic_attr_suffixes_are_not_extent_authority(tmp_path: Path) -> None:
    zarr_path = tmp_path / "generic-authority.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    _write_node(
        zarr_path,
        "coordinate_records/generic_extent",
        attributes={"custom_width_px": 640, "custom_height_px": 480},
    )
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    descriptor = _complete_xy_descriptor()
    descriptor["reference_extent"]["authority"] = (
        "/coordinate_records/generic_extent.attrs[custom_width_px,custom_height_px]"
    )
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "generic-authority",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_SYNTAX_UNSUPPORTED" in surface["issue_codes"]


def test_transform_chain_rejects_branches_cycles_order_and_camera_conflicts(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "ambiguous-transform-chain.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    camera_paths: dict[str, str] = {}
    for camera_id, width, height in (
        ("camera-1", 640, 480),
        ("camera-2", 800, 600),
    ):
        camera_paths[camera_id] = _write_selected_camera_authority(
            zarr_path,
            selection_name=f"selected-{camera_id}",
            camera_id=camera_id,
            width=width,
            height=height,
        )
    _write_node(zarr_path, "coordinate_records/canvas_roi_calibration")
    _write_array(zarr_path, "coordinate_records/canvas_image", shape=[1, 300, 400])
    _write_array(zarr_path, "crop_runs/crop/roi_images", shape=[1, 256, 512])
    camera_1 = {
        "width": 640,
        "height": 480,
        "units": "px",
        "authority": (
            f"/{camera_paths['camera-1']}@native_width_px,native_height_px"
        ),
    }
    camera_2 = {
        "width": 800,
        "height": 600,
        "units": "px",
        "authority": (
            f"/{camera_paths['camera-2']}@native_width_px,native_height_px"
        ),
    }
    canvas = {
        "width": 400,
        "height": 300,
        "units": "px",
        "authority": "/coordinate_records/canvas_image.shape[-2:]",
    }
    roi = {
        "width": 512,
        "height": 256,
        "units": "px",
        "authority": "/crop_runs/crop/roi_images.shape[-2:]",
    }

    def _transform(
        transform_id: str,
        from_space: str,
        to_space: str,
        source_extent: dict[str, object],
        target_extent: dict[str, object],
        calibration_ref: str,
        camera_id: str | None = None,
    ):
        value = build_directed_homography(
            transform_id=transform_id,
            matrix=np.eye(3),
            from_space_id=from_space,
            to_space_id=to_space,
            source_reference_extent=source_extent,
            target_reference_extent=target_extent,
            calibration_ref=calibration_ref,
            camera_id=camera_id,
        )
        path = f"coordinate_records/{transform_id}"
        _write_array(
            zarr_path,
            path,
            shape=[3, 3],
            attributes=directed_homography_attrs(value),
        )
        return {"ref": f"/{path}", "sha256": value.digest()}

    camera_to_canvas = _transform(
        "camera_to_canvas",
        "source_camera_image_px",
        "stimulus_canvas_px",
        camera_1,
        canvas,
        camera_paths["camera-1"],
        "camera-1",
    )
    canvas_to_roi = _transform(
        "canvas_to_roi",
        "stimulus_canvas_px",
        "roi_local_px",
        canvas,
        roi,
        "coordinate_records/canvas_roi_calibration",
    )
    camera_to_roi = _transform(
        "camera_to_roi_branch",
        "source_camera_image_px",
        "roi_local_px",
        camera_1,
        roi,
        camera_paths["camera-1"],
        "camera-1",
    )
    canvas_to_camera = _transform(
        "canvas_to_camera_cycle",
        "stimulus_canvas_px",
        "source_camera_image_px",
        canvas,
        camera_1,
        camera_paths["camera-1"],
        "camera-1",
    )
    canvas_to_camera_2_norm = _transform(
        "canvas_to_camera_2_normalized",
        "stimulus_canvas_px",
        "source_camera_normalized_xy",
        canvas,
        camera_2,
        camera_paths["camera-2"],
        "camera-2",
    )
    camera_2_norm_to_roi = _transform(
        "camera_2_normalized_to_roi",
        "source_camera_normalized_xy",
        "roi_local_px",
        camera_2,
        roi,
        camera_paths["camera-2"],
        "camera-2",
    )

    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    descriptor = {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 1,
        "space_id": "roi_local_px",
        "geometry_type": "points_xy",
        "components": ["x", "y"],
        "component_units": ["px", "px"],
        "origin": "top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "reference_extent": roi,
        "pixel_convention": "pixel_center",
        "row_identity": {
            "mode": "explicit_array",
            "array_ref": "track_sample_key",
        },
        "source_camera_overlay": "requires_transform",
        "lineage_refs": [
            {
                "ref": _CANONICAL_LINEAGE_REF,
                "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
            }
        ],
        "transform_refs": [],
    }
    _write_array(zarr_path, f"{track}/positions_px", shape=[1, 2])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "ambiguous-transform-chain",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    for refs, expected_code in (
        (
            [camera_to_canvas, canvas_to_roi, camera_to_roi],
            "TRANSFORM_CHAIN_NOT_LINEAR",
        ),
        (
            [camera_to_canvas, canvas_to_camera, camera_to_roi],
            "TRANSFORM_CHAIN_NOT_LINEAR",
        ),
        ([canvas_to_roi, camera_to_canvas], "TRANSFORM_CHAIN_NOT_LINEAR"),
        (
            [
                camera_to_canvas,
                canvas_to_camera_2_norm,
                camera_2_norm_to_roi,
            ],
            "TRANSFORM_CHAIN_CAMERA_IDENTITY_CONFLICT",
        ),
    ):
        descriptor["transform_refs"] = refs
        _write_array(
            zarr_path,
            f"{track}/positions_px",
            shape=[1, 2],
            attributes=_descriptor_attrs(descriptor),
        )
        surface = next(
            item
            for item in _surface_rows(audit_registry(registry))
            if item["surface_type"] == "track_positions_px"
        )
        assert surface["status"] == "ambiguous_fail_closed"
        assert expected_code in surface["issue_codes"]


def test_legacy_camera_backfill_requires_resolver_evidence(tmp_path: Path) -> None:
    zarr_path = tmp_path / "legacy.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    attrs = {
        "coordinate_space": "camera",
        "units": "px",
        "origin": "top_left",
        "positive_x_direction": "right",
        "positive_y_direction": "down",
        "reference_width": 640,
        "reference_height": 480,
        "pixel_convention": "pixel_center",
        "geometry_convention": "points_xy",
        "source_ref": {
            "ref": _CANONICAL_LINEAGE_REF,
            "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
        },
        "source_camera_overlay_suitable": True,
    }
    _write_array(zarr_path, f"{track}/positions_px", shape=[1, 2], attributes=attrs)
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "legacy",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert "LEGACY_SPACE_CONTEXT_INVALID" in surface["issue_codes"]
    assert surface["status"] == "ambiguous_fail_closed"
    assert coordinate_audit._migration_class(surface) != "safe_metadata_only_backfill"

    attrs["reference_authority"] = "/raw_video/images_full.shape[-2:]"
    _write_array(zarr_path, f"{track}/positions_px", shape=[1, 2], attributes=attrs)
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "numerical_validation_required"
    assert "LEGACY_NUMERICAL_INVARIANTS_MISSING" in surface["issue_codes"]
    assert coordinate_audit._migration_class(surface) != "safe_metadata_only_backfill"

    proof = {
        "schema_id": "palette.legacy_coordinate_compatibility_evidence",
        "schema_version": 1,
        "legacy_label": "camera",
        "canonical_space_id": "source_camera_image_px",
        "surface_path": f"/{track}/positions_px",
        "row_count": 1,
        "values_sha256": "4" * 64,
        "validation_tool_commit": "fixture-commit",
        "numerical_invariants": {
            "reference_extent_verified": True,
            "row_identity_verified": True,
            "values_finite_or_declared_missing": True,
            "values_preserved": True,
        },
        "values_changed": False,
    }
    _write_node(
        zarr_path,
        "coordinate_records/legacy_compatibility",
        attributes={"proof": proof},
    )
    attrs["source_ref"] = [
        {
            "ref": _CANONICAL_LINEAGE_REF,
            "sha256": coordinate_audit._fingerprint(_CANONICAL_LINEAGE_RECORD),
        },
        {
            "ref": "/coordinate_records/legacy_compatibility.attrs[proof]",
            "sha256": coordinate_audit._fingerprint(proof),
        },
    ]
    _write_array(zarr_path, f"{track}/positions_px", shape=[1, 2], attributes=attrs)
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "numerical_validation_required"
    assert "LEGACY_COMPATIBILITY_PROOF_NOT_CANONICAL_AUTHORITY" in surface["issue_codes"]
    assert coordinate_audit._migration_class(surface) != "safe_metadata_only_backfill"


def test_run_family_no_match_is_not_compatible_or_no_change(tmp_path: Path) -> None:
    zarr_path = tmp_path / "filtered.zarr"
    _write_node(zarr_path)
    _write_array(
        zarr_path,
        "analysis/detect_runs/run/bbox_norm_coords",
        shape=[1, 4],
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "filtered",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    records = audit_registry(registry, run_families=["track_kinematics_runs"])
    dataset = _dataset_rows(records)[0]
    assert dataset["status"] == "not_applicable_unscanned"
    migration = coordinate_audit.build_migration_manifest(records)
    assert migration[0]["migration_class"] == "not_applicable_unscanned"
    assert migration[0]["migration_class"] != "no_change"


def test_checkpoint_rejects_incomplete_digest_valid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "checkpoint.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(_complete_arena_descriptor(zarr_path)),
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "checkpoint",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    checkpoint_dir = tmp_path / "checkpoints"
    audit_registry(registry, checkpoint_dir=checkpoint_dir)
    checkpoint_path = next(checkpoint_dir.glob("dataset-*.json"))
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["records"] = [payload["records"][0]]
    payload["record_count"] = 1
    payload["records_sha256"] = coordinate_audit._fingerprint(payload["records"])
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    original = coordinate_audit.audit_dataset_row
    rescanned: list[str] = []

    def _record_scan(row, **kwargs):
        rescanned.append(str(row["dataset_id"]))
        return original(row, **kwargs)

    monkeypatch.setattr(coordinate_audit, "audit_dataset_row", _record_scan)
    audit_registry(registry, checkpoint_dir=checkpoint_dir)
    assert rescanned == ["checkpoint"]

    # A self-consistent forged checkpoint still cannot redefine which
    # coordinate surfaces the current archive is expected to contain.
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    dataset_record = payload["records"][0]
    dataset_record["surface_count"] = 0
    dataset_record["expected_surface_identities"] = []
    dataset_record["expected_surface_identities_sha256"] = (
        coordinate_audit._fingerprint([])
    )
    payload["records"] = [dataset_record]
    payload["surface_count"] = 0
    payload["expected_surface_identities_sha256"] = (
        dataset_record["expected_surface_identities_sha256"]
    )
    payload["record_count"] = 1
    payload["records_sha256"] = coordinate_audit._fingerprint(payload["records"])
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    audit_registry(registry, checkpoint_dir=checkpoint_dir)
    assert rescanned == ["checkpoint", "checkpoint"]

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["audit_ruleset_version"] = coordinate_audit.AUDIT_RULESET_VERSION + 1
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    audit_registry(registry, checkpoint_dir=checkpoint_dir)
    assert rescanned == ["checkpoint", "checkpoint", "checkpoint"]


def test_audit_registry_reads_recordings_and_datasets_in_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "snapshot.zarr"
    _write_node(zarr_path)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "snapshot",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    original = coordinate_audit.read_registry_snapshot_rows
    calls: list[Path] = []

    def _record_snapshot(path: Path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(coordinate_audit, "read_registry_snapshot_rows", _record_snapshot)
    audit_registry(registry)
    assert calls == [registry]


def test_registry_drift_after_scan_invalidates_migration_manifest(tmp_path: Path) -> None:
    zarr_path = tmp_path / "drift.zarr"
    _write_node(zarr_path)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "drift",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    records = audit_registry(registry)
    conn = sqlite3.connect(registry)
    try:
        conn.execute(
            "UPDATE recordings SET session_uuid = ? WHERE recording_id = ?",
            ("changed-after-scan", "rec"),
        )
        conn.commit()
    finally:
        conn.close()
    snapshot = coordinate_audit.build_registry_snapshot(registry, records)
    assert snapshot["registry_changed_after_scan"] is True
    migration = coordinate_audit.build_migration_manifest(
        records,
        registry_snapshot=snapshot,
    )
    assert migration[0]["must_fail_closed"] is True
    assert "REGISTRY_SNAPSHOT_INVALIDATES_MIGRATION" in migration[0]["issue_codes"]


def test_expected_key_coverage_is_independent_and_duplicate_keys_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.zarr"
    second = tmp_path / "second.zarr"
    _write_node(first)
    _write_node(second)
    rows = [
        {
            "dataset_id": name,
            "recording_id": f"rec-{name}",
            "zarr_path": str(path),
            "zarr_use": "analysis",
            "artifact_kind": "source_recording",
            "status": "active",
        }
        for name, path in (("first", first), ("second", second))
    ]
    registry = _make_registry(tmp_path / "registry.sqlite", rows)
    records = audit_registry(registry)
    partial = [
        record for record in records if record.get("dataset_id") == "first"
    ]
    snapshot = coordinate_audit.build_registry_snapshot(registry, partial)
    coverage = coordinate_audit.build_coverage(partial, registry_snapshot=snapshot)
    assert coverage["all_selected_dataset_rows_represented"] is False
    assert coverage["missing_expected_dataset_keys"] == ["second"]
    migration = coordinate_audit.build_migration_manifest(
        partial,
        registry_snapshot=snapshot,
    )
    assert migration[0]["must_fail_closed"] is True
    assert "REGISTRY_SNAPSHOT_INVALIDATES_MIGRATION" in migration[0]["issue_codes"]

    duplicate_rows = [dict(rows[0]), dict(rows[0])]
    recording_rows = coordinate_audit.read_registry_recording_rows(registry)
    monkeypatch.setattr(
        coordinate_audit,
        "read_registry_snapshot_rows",
        lambda _path: (recording_rows, duplicate_rows),
    )
    with pytest.raises(ValueError, match="unique audit keys"):
        audit_registry(registry)


def test_output_and_checkpoint_paths_cannot_mutate_sources(tmp_path: Path) -> None:
    zarr_path = tmp_path / "source.zarr"
    _write_node(zarr_path)
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "source",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    forbidden_checkpoint = zarr_path / "audit-checkpoints"
    with pytest.raises(ValueError, match="inside a scanned Zarr"):
        audit_registry(registry, checkpoint_dir=forbidden_checkpoint)
    assert not forbidden_checkpoint.exists()

    before = registry.read_bytes()
    with pytest.raises(ValueError, match="replace the source registry"):
        coordinate_audit.main(
            ["--registry", str(registry), "--output-jsonl", str(registry)]
        )
    assert registry.read_bytes() == before


def test_descriptor_bearing_new_stage_and_uncontrolled_legacy_are_never_skipped(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "unknown-surfaces.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    _write_array(
        zarr_path,
        "analysis/new_stage/run/positions",
        shape=[1, 2],
        attributes={
            "coordinate_space": "bogus_frame",
            "units": "bananas",
            "origin": "somewhere",
            "positive_x_direction": "sideways",
            "positive_y_direction": "up-ish",
        },
    )
    descriptor = _complete_arena_descriptor(zarr_path)
    descriptor["row_identity"] = {
        "mode": "explicit_array",
        "array_ref": "track_sample_key",
    }
    _write_array(
        zarr_path,
        "analysis/future_coordinate_runs/run/mystery_points",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    _write_array(
        zarr_path,
        "analysis/future_coordinate_runs/run/frame_indices",
        shape=[1],
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "unknown-surfaces",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    records = audit_registry(registry)
    surfaces = _surface_rows(records)
    assert {row["surface_path"] for row in surfaces} == {
        "analysis/future_coordinate_runs/run/mystery_points",
    }
    assert all(row["surface_type"] == "unclassified_geometry_candidate" for row in surfaces)
    assert all(row["status"] == "ambiguous_fail_closed" for row in surfaces)
    assert _dataset_rows(records)[0]["coordinate_bearing_node_count"] == 1


def test_direct_descriptors_override_leaf_exclusions_and_fail_unknown_profiles(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "descriptor-precedence.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)

    bbox_descriptor = _complete_xy_descriptor(
        space_id="detector_normalized_xy",
        units="normalized",
    )
    bbox_descriptor["geometry_type"] = "bbox_xyxy"
    bbox_descriptor["components"] = ["x_min", "y_min", "x_max", "y_max"]
    bbox_descriptor["component_units"] = ["normalized"] * 4
    bbox_descriptor["source_camera_overlay"] = "requires_transform"
    bbox_path = "analysis/detect_runs/run/bbox_norm_coords"
    _write_array(
        zarr_path,
        bbox_path,
        shape=[1, 4],
        attributes={
            **_descriptor_attrs(bbox_descriptor),
            "semantic_kind": "non_spatial",
        },
    )
    _write_array(zarr_path, "analysis/detect_runs/run/frame_indices", shape=[1])

    unknown_path = "analysis/future_coordinate_runs/run/status"
    _write_array(
        zarr_path,
        unknown_path,
        shape=[1, 2],
        attributes=_descriptor_attrs(_complete_arena_descriptor(zarr_path)),
    )
    _write_array(
        zarr_path,
        "analysis/future_coordinate_runs/run/frame_indices",
        shape=[1],
    )
    registry = _make_registry(
        tmp_path / "descriptor-precedence.sqlite",
        [
            {
                "dataset_id": "descriptor-precedence",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surfaces = {
        row["surface_path"]: row
        for row in _surface_rows(audit_registry(registry))
    }
    assert surfaces[bbox_path]["surface_type"] == "detect_bbox"
    assert surfaces[unknown_path]["surface_type"] == (
        "unclassified_geometry_candidate"
    )
    assert "UNSUPPORTED_DECLARED_COORDINATE_SURFACE" in surfaces[
        unknown_path
    ]["issue_codes"]


def test_bbox_rejects_not_applicable_identity_conflicting_space_and_unrelated_lineage(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "bbox-conflict.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    descriptor = _complete_xy_descriptor(space_id="detector_normalized_xy")
    descriptor["geometry_type"] = "bbox_xyxy"
    descriptor["components"] = ["x_min", "y_min", "x_max", "y_max"]
    descriptor["component_units"] = ["normalized"] * 4
    descriptor["row_identity"] = {
        "mode": "not_applicable",
        "array_ref": None,
    }
    descriptor["source_camera_overlay"] = "requires_transform"
    _write_array(
        zarr_path,
        "analysis/detect_runs/run/bbox_norm_coords",
        shape=[1, 4],
        attributes={
            **_descriptor_attrs(descriptor),
            "coordinate_space": "texture",
        },
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "bbox-conflict",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert {
        "ROW_IDENTITY_NOT_APPLICABLE_FORBIDDEN",
        "DESCRIPTOR_DECLARATION_CONFLICT",
        "REFERENCE_AUTHORITY_LINEAGE_MISSING",
    } <= set(surface["issue_codes"])


def test_generic_two_by_two_homography_never_counts_as_direction_proof(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "bad-homography.zarr"
    _write_node(zarr_path)
    _write_array(
        zarr_path,
        "analysis/calibration/homography_matrix",
        shape=[2, 2],
        attributes={
            "from_space_id": "source_camera_image_px",
            "to_space_id": "stimulus_canvas_px",
            "calibration_ref": "/some/existing/path",
        },
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "bad-homography",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "HOMOGRAPHY_ARRAY_INVALID" in surface["issue_codes"]
    assert "DIRECTED_TRANSFORM_METADATA_MISSING" in surface["issue_codes"]


def test_identity_modes_are_semantic_and_instance_key_is_not_a_track_alias(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "identity-contract.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    descriptor = _complete_arena_descriptor(zarr_path)
    descriptor["row_identity"] = {
        "mode": "instance_key",
        "array_ref": "instance_key",
    }
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[2, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    _write_array(zarr_path, f"{track}/instance_key", shape=[2])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "identity-contract",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "SURFACE_PROFILE_ROW_IDENTITY_UNSUPPORTED" in surface["issue_codes"]
    assert surface["evidence"]["row_identity"]["descriptor_value"] == {
        "mode": "instance_key",
        "array_ref": "instance_key",
    }


def test_descriptor_free_identity_uses_exact_family_domain_candidates(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "descriptor-free-identity.zarr"
    _write_node(zarr_path)

    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(zarr_path, f"{track}/positions_px", shape=[2, 2])
    _write_array(zarr_path, f"{track}/instance_key", shape=[2])
    _write_node(
        zarr_path,
        f"{track}/track_sample_key",
        node_type="array",
        shape=[2, 2],
        data_type="int64",
    )
    _write_array(zarr_path, f"{track}/row_ids", shape=[2], data_type="int64")

    stimulus = "analysis/stimulus_runs/run/tracking_data/chaser_states"
    _write_array(zarr_path, f"{stimulus}/chaser_pos_x", shape=[2])
    _write_array(zarr_path, f"{stimulus}/stimulus_state_key", shape=[2])
    _write_array(zarr_path, f"{stimulus}/coordinate_row_identity", shape=[2])

    registry = _make_registry(
        tmp_path / "descriptor-free-identity.sqlite",
        [
            {
                "dataset_id": "descriptor-free-identity",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surfaces = {
        row["surface_path"]: row
        for row in _surface_rows(audit_registry(registry))
    }
    track_surface = surfaces[f"{track}/positions_px"]
    assert track_surface["status"] == "ambiguous_fail_closed"
    assert {
        "TRACK_SOURCE_INSTANCE_KEY_IS_LINEAGE_ONLY",
        "LEGACY_ROW_IDENTITY_UNTYPED",
    } <= set(track_surface["issue_codes"])
    assert all(
        "row_ids" not in str(issue.get("evidence", {}))
        for issue in track_surface["issues"]
    )

    stimulus_surface = surfaces[f"{stimulus}/chaser_pos_x"]
    assert stimulus_surface["status"] == "ambiguous_fail_closed"
    assert "LEGACY_ROW_IDENTITY_AMBIGUOUS" in stimulus_surface["issue_codes"]


def test_canonical_track_identity_requires_exact_acquisition_time_lineage(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "track-time-lineage.zarr"
    _write_node(zarr_path)
    (
        acquisition_path,
        acquisition_attrs,
        frame_path,
        frame_attrs,
        _source_frame,
        acquisition,
        token,
        root_attrs,
    ) = _sealed_source_camera_frame_attrs()
    _merge_node_attrs(zarr_path, ".", root_attrs)
    rowset_path = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    key_values = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
    source_indices = np.asarray([0, 1], dtype=np.int64)
    interpolation = np.zeros(2, dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE)
    interpolation["left_source_frame_index"] = source_indices
    interpolation["right_source_frame_index"] = source_indices
    source_rowset_path = "analysis/refined_detect_runs/source"
    source_rowset = _MemoryNode(source_rowset_path, token=token)
    source_key_node = _MemoryNode(
        f"{source_rowset_path}/instance_key",
        token=token,
        data=np.asarray([10, 11], dtype=np.uint64),
    )
    source_identity = stamp_and_bind_row_identity_contract(
        source_rowset,
        source_key_node,
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=source_key_node[:],
        ),
    )
    source_frame_node = _MemoryNode(
        f"{source_rowset_path}/source_acquisition_frame_index",
        token=token,
        data=source_indices,
    )
    source_temporal = stamp_source_row_temporal_authority(
        source_rowset,
        source_frame_node,
        source_row_identity=source_identity,
        acquisition_frame=acquisition,
    )

    rowset = _MemoryNode(rowset_path, token=token)
    key_node = _MemoryNode(
        f"{rowset_path}/track_sample_key",
        token=token,
        data=key_values,
    )
    source_row_index_node = _MemoryNode(
        f"{rowset_path}/source_row_index",
        token=token,
        data=np.asarray([0, 1], dtype=np.int64),
    )
    output_frame_node = _MemoryNode(
        f"{rowset_path}/source_acquisition_frame_index",
        token=token,
        data=resolve_source_acquisition_frame_indices(
            source_temporal,
            source_row_index_node[:],
        ),
    )
    interpolation_node = _MemoryNode(
        f"{rowset_path}/source_frame_interpolation",
        token=token,
        data=interpolation,
    )
    source_instance_node = _MemoryNode(
        f"{rowset_path}/source_instance_key",
        token=token,
        data=derive_track_source_instance_values(
            source_temporal,
            source_row_index_node[:],
        ),
    )
    time_lineage = stamp_track_sample_time_lineage(
        rowset,
        key_node,
        source_row_index_node,
        output_frame_node,
        interpolation_node,
        source_instance_node,
        source_temporal_authority=source_temporal,
    )
    contract = build_row_identity_contract(
        domain=TRACK_SAMPLE_DOMAIN,
        values=key_values,
        track_time_lineage=time_lineage,
    )
    stamp_and_bind_row_identity_contract(
        rowset,
        key_node,
        contract=contract,
        track_time_lineage=time_lineage,
    )

    for path, attrs in (
        (acquisition_path, acquisition_attrs),
        (frame_path, frame_attrs),
        (source_rowset_path, dict(source_rowset.attrs)),
    ):
        _ensure_groups(zarr_path, path)
        _write_node(zarr_path, path, attributes=attrs)
    _ensure_groups(zarr_path, f"{rowset_path}/track_sample_key")
    _write_node(zarr_path, rowset_path, attributes=dict(rowset.attrs))
    for node in (
        key_node,
        source_key_node,
        source_frame_node,
        source_row_index_node,
        output_frame_node,
        interpolation_node,
        source_instance_node,
    ):
        _write_node(
            zarr_path,
            node.path,
            node_type="array",
            attributes=dict(node.attrs),
            shape=list(node.shape),
            data_type=(
                node.dtype.descr
                if node.dtype.fields is not None
                else node.dtype.name
            ),
        )
    surface_path = f"{rowset_path}/positions_px"
    _write_array(zarr_path, surface_path, shape=[2, 2])
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert resolved == f"{rowset_path}/track_sample_key"
    assert [issue["code"] for issue in issues] == [
        "TRACK_TIME_LINEAGE_PAYLOAD_VALIDATION_REQUIRED"
    ]

    original_track_attrs = json.loads(json.dumps(dict(rowset.attrs)))
    tampered_track = json.loads(
        json.dumps(original_track_attrs["track_sample_time_lineage"])
    )
    tampered_track["source_row_index"]["ref"] = (
        f"/{rowset_path}/source_acquisition_frame_index"
    )
    _merge_node_attrs(
        zarr_path,
        rowset_path,
        {
            "track_sample_time_lineage": tampered_track,
            "track_sample_time_lineage_sha256": coordinate_audit._fingerprint(
                tampered_track
            ),
        },
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    _resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert "TRACK_TIME_LINEAGE_ARRAY_INVALID" in {
        issue["code"] for issue in issues
    }
    _replace_group_attrs(zarr_path, rowset_path, original_track_attrs)

    self_certified_source = json.loads(
        json.dumps(source_rowset.attrs["source_row_temporal_authority"])
    )
    self_certified_track = json.loads(
        json.dumps(original_track_attrs["track_sample_time_lineage"])
    )
    self_certified_track["source_row_temporal_authority"] = {
        "record_ref": f"/{rowset_path}@source_row_temporal_authority",
        "record_sha256": coordinate_audit._fingerprint(
            self_certified_source
        ),
    }
    _merge_node_attrs(
        zarr_path,
        rowset_path,
        {
            "source_row_temporal_authority": self_certified_source,
            "source_row_temporal_authority_sha256": (
                coordinate_audit._fingerprint(self_certified_source)
            ),
            "track_sample_time_lineage": self_certified_track,
            "track_sample_time_lineage_sha256": (
                coordinate_audit._fingerprint(self_certified_track)
            ),
        },
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    _resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert "TRACK_TIME_LINEAGE_SELF_CERTIFIED_SOURCE" in {
        issue["code"] for issue in issues
    }
    _replace_group_attrs(zarr_path, rowset_path, original_track_attrs)

    original_source_attrs = json.loads(json.dumps(dict(source_rowset.attrs)))
    tampered_source = json.loads(
        json.dumps(original_source_attrs["source_row_temporal_authority"])
    )
    tampered_source["source_row_identity"]["record_ref"] = (
        "/analysis/refined_detect_runs/other@row_identity_contract"
    )
    tampered_source["source_acquisition_frame_index"].pop("content_sha256")
    _merge_node_attrs(
        zarr_path,
        source_rowset_path,
        {
            "source_row_temporal_authority": tampered_source,
            "source_row_temporal_authority_sha256": (
                coordinate_audit._fingerprint(tampered_source)
            ),
        },
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    _resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert {
        "TRACK_TIME_LINEAGE_SOURCE_IDENTITY_INVALID",
        "TRACK_TIME_LINEAGE_SOURCE_ARRAY_INVALID",
    } <= {issue["code"] for issue in issues}
    _replace_group_attrs(zarr_path, source_rowset_path, original_source_attrs)

    _merge_node_attrs(
        zarr_path,
        acquisition_path,
        {"acquisition_import_ownership_sha256": "0" * 64},
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    _resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert "TRACK_TIME_LINEAGE_ACQUISITION_INVALID" in {
        issue["code"] for issue in issues
    }
    _replace_group_attrs(zarr_path, acquisition_path, acquisition_attrs)

    _merge_node_attrs(
        zarr_path,
        rowset_path,
        {"track_sample_time_lineage_sha256": "0" * 64},
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert resolved is None
    assert "TRACK_TIME_LINEAGE_DIGEST_MISMATCH" in {
        issue["code"] for issue in issues
    }

    _merge_node_attrs(
        zarr_path,
        rowset_path,
        {
            "track_sample_time_lineage": {
                "schema_id": "palette.track_sample_time_lineage",
                "schema_version": 1,
                "acquisition_camera_frame": {
                    "record_ref": (
                        f"/{acquisition_path}@acquisition_camera_frame"
                    ),
                    "record_sha256": acquisition_attrs[
                        "acquisition_camera_frame_sha256"
                    ],
                },
            }
        },
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    resolved, issues = coordinate_audit._legacy_row_identity_resolution(
        surface_type="track_positions_px",
        node=nodes[surface_path],
        nodes=nodes,
    )
    assert resolved is None
    assert "TRACK_TIME_LINEAGE_RETIRED_DIRECT_ACQUISITION" in {
        issue["code"] for issue in issues
    }


def test_acquisition_metadata_cross_validation_fails_closed_without_bulk_reads(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "materialized-acquisition.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }

    def issue_codes(
        candidate_nodes: dict[str, coordinate_audit.MetadataNode],
        *,
        pointer: dict[str, Any] | None = None,
    ) -> set[str]:
        _target, issues = coordinate_audit._reference_extent_binding_issues(
            pointer or fixture["pointer"],
            role="test.acquisition",
            nodes=candidate_nodes,
        )
        return {issue["code"] for issue in issues}

    assert issue_codes(nodes) == set()

    wrong_root = dict(nodes)
    wrong_root["."] = replace(
        nodes["."],
        attributes={**nodes["."].attributes, "recording_id": "other"},
    )
    assert "ACQUISITION_ROOT_METADATA_MISMATCH" in issue_codes(wrong_root)

    authority_path = str(fixture["acquisition_path"])
    wrong_authority_path = dict(nodes)
    wrong_authority_path[authority_path] = replace(
        nodes[authority_path],
        relative_path="analysis/acquisition_camera_frames/not-camera-1",
    )
    assert "ACQUISITION_AUTHORITY_PATH_MISMATCH" in issue_codes(
        wrong_authority_path
    )

    missing_frame = dict(nodes)
    missing_frame.pop(str(fixture["frame_path"]))
    assert "ACQUISITION_MATERIALIZED_NODE_UNRESOLVED" in issue_codes(
        missing_frame
    )

    bad_index = dict(nodes)
    index_path = str(fixture["index_path"])
    bad_index[index_path] = replace(
        nodes[index_path],
        data_type="float32",
    )
    assert "ACQUISITION_MATERIALIZED_NODE_METADATA_MISMATCH" in issue_codes(
        bad_index
    )

    wrong_pointer = json.loads(json.dumps(fixture["pointer"]))
    authority_attrs = json.loads(
        json.dumps(nodes[authority_path].attributes)
    )
    ownership = authority_attrs["acquisition_import_ownership"]
    ownership["mode"] = "external_video_v1"
    ownership["frame_array"] = None
    ownership["frame_index"] = None
    ownership["import_operation"] = None
    ownership_sha256 = coordinate_audit._fingerprint(ownership)
    authority_attrs["acquisition_import_ownership_sha256"] = ownership_sha256
    acquisition = authority_attrs["acquisition_camera_frame"]
    acquisition["import_ownership"]["record_sha256"] = ownership_sha256
    acquisition_sha256 = coordinate_audit._fingerprint(acquisition)
    authority_attrs["acquisition_camera_frame_sha256"] = acquisition_sha256
    wrong_pointer["record_sha256"] = acquisition_sha256
    wrong_mode = dict(nodes)
    wrong_mode[authority_path] = replace(
        nodes[authority_path],
        attributes=authority_attrs,
    )
    assert "ACQUISITION_MODE_MISMATCH" in issue_codes(
        wrong_mode,
        pointer=wrong_pointer,
    )

    manifest_path = str(fixture["manifest_path"])
    bad_manifest = dict(nodes)
    manifest_attrs = json.loads(
        json.dumps(nodes[manifest_path].attributes)
    )
    manifest = manifest_attrs["acquisition_materialization_manifest"]
    manifest["frame_map"]["record_sha256"] = "3" * 64
    manifest_attrs["acquisition_materialization_manifest_sha256"] = (
        coordinate_audit._fingerprint(manifest)
    )
    bad_manifest[manifest_path] = replace(
        nodes[manifest_path],
        attributes=manifest_attrs,
    )
    assert "ACQUISITION_MATERIALIZATION_MANIFEST_MISMATCH" in issue_codes(
        bad_manifest
    )

    bad_chunks = dict(nodes)
    chunk_attrs = json.loads(json.dumps(nodes[manifest_path].attributes))
    chunk_record = chunk_attrs["acquisition_physical_chunk_manifest"]
    chunk_record["entries"][0]["storage_key"] = "not-the-declared-key"
    chunk_attrs["acquisition_physical_chunk_manifest_sha256"] = (
        coordinate_audit._fingerprint(chunk_record)
    )
    bad_chunks[manifest_path] = replace(
        nodes[manifest_path],
        attributes=chunk_attrs,
    )
    assert "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID" in issue_codes(
        bad_chunks
    )

    bad_frame_pointer = dict(nodes)
    pointer_attrs = json.loads(
        json.dumps(nodes[authority_path].attributes)
    )
    acquisition = pointer_attrs["acquisition_camera_frame"]
    acquisition["frame_index"]["record_ref"] = (
        "/raw_video/frame_domain_maps/other@array_values"
    )
    acquisition["frame_domain"]["index_record_ref"] = acquisition[
        "frame_index"
    ]["record_ref"]
    acquisition_sha256 = coordinate_audit._fingerprint(acquisition)
    pointer_attrs["acquisition_camera_frame_sha256"] = acquisition_sha256
    pointer = dict(fixture["pointer"])
    pointer["record_sha256"] = acquisition_sha256
    bad_frame_pointer[authority_path] = replace(
        nodes[authority_path],
        attributes=pointer_attrs,
    )
    assert "ACQUISITION_MATERIALIZED_POINTER_MISMATCH" in issue_codes(
        bad_frame_pointer,
        pointer=pointer,
    )


def _audit_standalone_acquisition_fixture(
    tmp_path: Path,
    zarr_path: Path,
    *,
    dataset_id: str,
) -> dict[str, object]:
    registry = _make_registry(
        tmp_path / f"{dataset_id}.sqlite",
        [
            {
                "dataset_id": dataset_id,
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    return _dataset_rows(audit_registry(registry))[0]


def test_standalone_materialized_acquisition_authority_is_inventoried(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-materialized.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-materialized",
    )

    assert dataset["surface_count"] == 0
    assert dataset["status"] == "compatible"
    assert dataset["issue_codes"] == ["NO_COORDINATE_SURFACES_DETECTED"]
    inventory = dataset["acquisition_authority_inventory"]
    assert inventory["inventory_status"] == "compatible"
    assert inventory["publication_state"] == "published_canonical_v1"
    assert inventory["validated_authority_path"] == fixture["acquisition_path"]
    assert inventory["signal_count"] == 10
    assert inventory["validation_issue_codes"] == []


def test_standalone_external_acquisition_authority_is_mode_aware(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-external.zarr"
    (
        acquisition_path,
        acquisition_attrs,
        _frame_path,
        _frame_attrs,
        _frame,
        _acquisition,
        _token,
        root_attrs,
    ) = _sealed_source_camera_frame_attrs()
    _write_node(zarr_path, attributes=root_attrs)
    _write_node(zarr_path, "raw_video")
    _ensure_groups(zarr_path, acquisition_path)
    _write_node(
        zarr_path,
        acquisition_path,
        attributes=acquisition_attrs,
    )
    status = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
        authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path=acquisition_path,
    ).to_dict()
    for relative_path in (".", "raw_video"):
        _merge_node_attrs(
            zarr_path,
            relative_path,
            {"acquisition_authority_publication_status": status},
        )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-external",
    )

    inventory = dataset["acquisition_authority_inventory"]
    assert dataset["status"] == "compatible"
    assert inventory["inventory_status"] == "compatible"
    assert inventory["authority_mode"] == EXTERNAL_ACQUISITION_AUTHORITY_MODE
    assert inventory["materialization_manifest_node_present"] is False


def test_explicit_noncanonical_acquisition_is_recompute_inventory(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-noncanonical.zarr"
    root_attrs = _sealed_source_camera_frame_attrs()[-1]
    _write_node(zarr_path, attributes=root_attrs)
    _write_node(zarr_path, "raw_video")
    status = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_NOT_PUBLISHED,
        reason_code="images_full_not_materialized",
    ).to_dict()
    for relative_path in (".", "raw_video"):
        _merge_node_attrs(
            zarr_path,
            relative_path,
            {"acquisition_authority_publication_status": status},
        )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-noncanonical",
    )

    assert dataset["status"] == "recompute_required"
    assert "ACQUISITION_AUTHORITY_NOT_PUBLISHED" in dataset["issue_codes"]
    assert dataset["acquisition_authority_inventory"]["inventory_status"] == (
        "recompute_required"
    )


def test_source_evidence_without_acquisition_authority_is_inventoried(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-authority-missing.zarr"
    root_attrs = _sealed_source_camera_frame_attrs()[-1]
    _write_node(zarr_path, attributes=root_attrs)
    _write_node(
        zarr_path,
        "raw_video",
        attributes={"source_path": "/recordings/camera.mp4"},
    )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-authority-missing",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_AUTHORITY_MISSING" in dataset["issue_codes"]
    assert dataset["acquisition_authority_inventory"]["applicable"] is True


@pytest.mark.parametrize(
    ("node_path", "data_type"),
    [
        ("raw_video/images_full", "float32"),
        (
            "raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame",
            "int32",
        ),
    ],
)
def test_materialized_acquisition_requires_importer_exact_dtypes(
    tmp_path: Path,
    node_path: str,
    data_type: str,
) -> None:
    zarr_path = tmp_path / f"wrong-{data_type}.zarr"
    _write_node(zarr_path)
    _write_materialized_acquisition_fixture(zarr_path)
    metadata_path = zarr_path / node_path / "zarr.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["data_type"] = data_type
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id=f"wrong-{data_type}",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_MATERIALIZED_NODE_METADATA_MISMATCH" in dataset[
        "issue_codes"
    ]


def test_materialized_acquisition_rejects_rank_four_uint8_frames(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "rank-four-frames.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    metadata_path = zarr_path / str(fixture["frame_path"]) / "zarr.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["shape"] = [2, 1, 480, 640]
    payload["chunk_grid"]["configuration"]["chunk_shape"] = [1, 1, 480, 640]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="rank-four-frames",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_MATERIALIZED_NODE_METADATA_MISMATCH" in dataset[
        "issue_codes"
    ]


def test_corrupt_materialization_manifest_is_detected_without_geometry_reference(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-corrupt-manifest.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    manifest_path = zarr_path / str(fixture["manifest_path"]) / "zarr.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = payload["attributes"]["acquisition_materialization_manifest"]
    manifest["frame_map"]["record_sha256"] = "3" * 64
    payload["attributes"]["acquisition_materialization_manifest_sha256"] = (
        coordinate_audit._fingerprint(manifest)
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-corrupt-manifest",
    )

    assert dataset["surface_count"] == 0
    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_MATERIALIZATION_MANIFEST_MISMATCH" in dataset[
        "issue_codes"
    ]


def test_orphan_materialization_manifest_fails_closed_at_dataset_level(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-orphan-manifest.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    _replace_group_attrs(zarr_path, str(fixture["acquisition_path"]), {})

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-orphan-manifest",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert {
        "ACQUISITION_AUTHORITY_INCOMPLETE",
        "ACQUISITION_MATERIALIZATION_MANIFEST_ORPHAN",
        "ACQUISITION_PUBLICATION_STATUS_CONFLICT",
    } <= set(dataset["issue_codes"])


def test_multiple_misplaced_and_split_acquisition_authority_fails_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-split-authority.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    authority_path = str(fixture["acquisition_path"])
    authority_metadata_path = zarr_path / authority_path / "zarr.json"
    authority_payload = json.loads(authority_metadata_path.read_text(encoding="utf-8"))
    original_attrs = authority_payload["attributes"]
    _replace_group_attrs(
        zarr_path,
        authority_path,
        {
            "acquisition_camera_frame": original_attrs[
                "acquisition_camera_frame"
            ],
            "acquisition_camera_frame_sha256": original_attrs[
                "acquisition_camera_frame_sha256"
            ],
        },
    )
    second_path = "analysis/acquisition_camera_frames/camera-2"
    _write_node(
        zarr_path,
        second_path,
        attributes={
            "acquisition_import_ownership": original_attrs[
                "acquisition_import_ownership"
            ],
            "acquisition_import_ownership_sha256": original_attrs[
                "acquisition_import_ownership_sha256"
            ],
        },
    )
    misplaced_path = "analysis/misplaced_acquisition"
    _write_node(
        zarr_path,
        misplaced_path,
        attributes={
            "acquisition_camera_frame_sha256": original_attrs[
                "acquisition_camera_frame_sha256"
            ]
        },
    )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-split-authority",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert {
        "ACQUISITION_AUTHORITY_MULTIPLE",
        "ACQUISITION_AUTHORITY_RECORD_SPLIT",
        "ACQUISITION_AUTHORITY_SIGNAL_MISPLACED",
    } <= set(dataset["issue_codes"])


def test_published_acquisition_authority_requires_root_and_raw_status(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-status-missing.zarr"
    _write_node(zarr_path)
    _write_materialized_acquisition_fixture(zarr_path)
    for relative_path in (".", "raw_video"):
        metadata_path = zarr_path / relative_path / "zarr.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["attributes"].pop("acquisition_authority_publication_status")
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-status-missing",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_PUBLICATION_STATUS_MISSING" in dataset["issue_codes"]


def test_pending_acquisition_publication_fails_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-status-pending.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    pending = dict(fixture["publication_status"])
    pending.update(
        {
            "status": ACQUISITION_AUTHORITY_PENDING,
            "reason_code": MATERIALIZED_ACQUISITION_PENDING_REASON,
        }
    )
    _merge_node_attrs(
        zarr_path,
        ".",
        {"acquisition_authority_publication_status": pending},
    )
    _merge_node_attrs(
        zarr_path,
        "raw_video",
        {"acquisition_authority_publication_status": pending},
    )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-status-pending",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_PUBLICATION_STATUS_PENDING" in dataset["issue_codes"]


@pytest.mark.parametrize("failure_mode", ["unequal", "malformed"])
def test_unequal_or_malformed_acquisition_publication_status_fails_closed(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    zarr_path = tmp_path / f"standalone-status-{failure_mode}.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    changed = dict(fixture["publication_status"])
    if failure_mode == "unequal":
        changed["reason_code"] = "different_reason"
        expected_code = "ACQUISITION_PUBLICATION_STATUS_CONFLICT"
    else:
        changed["schema_version"] = float(changed["schema_version"])
        expected_code = "ACQUISITION_PUBLICATION_STATUS_INVALID"
    _merge_node_attrs(
        zarr_path,
        "raw_video" if failure_mode == "unequal" else ".",
        {"acquisition_authority_publication_status": changed},
    )
    if failure_mode == "malformed":
        _merge_node_attrs(
            zarr_path,
            "raw_video",
            {"acquisition_authority_publication_status": changed},
        )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id=f"standalone-status-{failure_mode}",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert expected_code in dataset["issue_codes"]


def test_noncanonical_publication_status_conflicts_with_persisted_authority(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "standalone-noncanonical-conflict.zarr"
    _write_node(zarr_path)
    _write_materialized_acquisition_fixture(zarr_path)
    noncanonical = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_NOT_PUBLISHED,
        reason_code="organized_recording_identity_absent",
    ).to_dict()
    for relative_path in (".", "raw_video"):
        _merge_node_attrs(
            zarr_path,
            relative_path,
            {"acquisition_authority_publication_status": noncanonical},
        )

    dataset = _audit_standalone_acquisition_fixture(
        tmp_path,
        zarr_path,
        dataset_id="standalone-noncanonical-conflict",
    )

    assert dataset["status"] == "ambiguous_fail_closed"
    assert "ACQUISITION_PUBLICATION_STATUS_CONFLICT" in dataset["issue_codes"]


def test_acquisition_storage_identity_distinguishes_logical_chunks_and_physical_shards(
) -> None:
    storage_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 80],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [50, 40]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [10, 10],
                    "codecs": [],
                    "index_codecs": [],
                    "index_location": "end",
                },
            }
        ],
    }
    node = coordinate_audit.MetadataNode(
        relative_path="raw_video/images_full",
        node_type="array",
        metadata_format="zarr.json",
        shape=[100, 80],
        data_type="uint8",
        chunk_shape=[50, 40],
        storage_metadata=storage_metadata,
        attributes={},
    )

    identity = coordinate_audit._metadata_array_storage_identity(node)

    assert identity is not None
    assert identity["logical_chunk_shape"] == [10, 10]
    assert identity["physical_chunk_shape"] == [50, 40]
    assert coordinate_audit._metadata_physical_chunk_indices(
        node.shape,
        identity,
    ) == [[0, 0], [0, 1], [1, 0], [1, 1]]


@pytest.mark.parametrize(
    ("mutation", "bad_value"),
    [
        ("zarr_format", 3.0),
        ("zarr_format", True),
        ("shape", [100, 81]),
        ("shape", [100.0, 80]),
        ("data_type", "float32"),
        ("fill_value", float("nan")),
    ],
)
def test_acquisition_storage_identity_rejects_metadata_interface_divergence(
    mutation: str,
    bad_value: object,
) -> None:
    storage_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [100, 80],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [50, 40]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [],
    }
    storage_metadata[mutation] = bad_value
    node = coordinate_audit.MetadataNode(
        relative_path="raw_video/images_full",
        node_type="array",
        metadata_format="zarr.json",
        shape=[100, 80],
        data_type="uint8",
        chunk_shape=[50, 40],
        storage_metadata=storage_metadata,
        attributes={},
    )

    assert coordinate_audit._metadata_array_storage_identity(node) is None


@pytest.mark.parametrize("zarr_format", [3.0, True])
def test_metadata_parser_rejects_noninteger_zarr_v3_format(
    tmp_path: Path,
    zarr_format: object,
) -> None:
    zarr_path = tmp_path / f"invalid-v3-{zarr_format!r}.zarr"
    zarr_path.mkdir()
    (zarr_path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": zarr_format,
                "node_type": "group",
                "attributes": {},
            }
        ),
        encoding="utf-8",
    )

    node = list(coordinate_audit.iter_metadata_nodes(zarr_path))[0]

    assert "zarr_format is not 3" in str(node.metadata_error)


@pytest.mark.parametrize("zarr_format", [2.0, True])
def test_metadata_parser_rejects_noninteger_zarr_v2_format(
    tmp_path: Path,
    zarr_format: object,
) -> None:
    zarr_path = tmp_path / f"invalid-v2-{zarr_format!r}.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zgroup").write_text(
        json.dumps({"zarr_format": zarr_format}),
        encoding="utf-8",
    )

    node = list(coordinate_audit.iter_metadata_nodes(zarr_path))[0]

    assert "zarr_format is not 2" in str(node.metadata_error)


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            '{"zarr_format":3,"node_type":"group","attributes":{"x":NaN}}',
            "non-finite JSON constant",
        ),
        (
            '{"zarr_format":3,"node_type":"group","attributes":{"x":Infinity}}',
            "non-finite JSON constant",
        ),
        (
            '{"zarr_format":3,"zarr_format":3,"node_type":"group","attributes":{}}',
            "duplicate JSON object key",
        ),
        (
            '{"zarr_format":3,"node_type":"group","attributes":{"x":1e309}}',
            "non-finite JSON float",
        ),
        (
            '{"zarr_format":3,"node_type":"group","attributes":{"x":"\\ud800"}}',
            "valid Unicode scalars",
        ),
    ],
)
def test_metadata_parser_rejects_nonfinite_and_duplicate_key_json_without_crashing_audit(
    tmp_path: Path,
    payload: str,
    expected_error: str,
) -> None:
    zarr_path = tmp_path / "invalid-canonical-json.zarr"
    zarr_path.mkdir()
    (zarr_path / "zarr.json").write_text(payload, encoding="utf-8")

    node = list(coordinate_audit.iter_metadata_nodes(zarr_path))[0]
    records = coordinate_audit.audit_dataset_row(
        {
            "dataset_id": "invalid-json",
            "recording_id": "rec",
            "zarr_path": str(zarr_path),
            "zarr_origin": "source",
            "zarr_use": "analysis",
            "artifact_kind": "source_recording",
            "status": "active",
        }
    )

    assert expected_error in str(node.metadata_error)
    assert _dataset_rows(records)[0]["status"] == "missing_or_unreadable"
    assert "INVALID_ZARR_METADATA_INVENTORY" in _dataset_rows(records)[0][
        "issue_codes"
    ]
    with pytest.raises(ValueError, match="Out of range float values"):
        coordinate_audit._canonical_json({"x": float("nan")})


def test_scanner_fingerprint_matches_writer_unicode_canonicalization() -> None:
    value = {"camera_id": "caméra-鱼", "label": "🐟"}
    canonical = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert coordinate_audit._canonical_json(value) == canonical
    assert coordinate_audit._fingerprint(value) == expected


@pytest.mark.parametrize(
    ("record_name", "field_path", "bad_value", "expected_code"),
    [
        (
            "chunk",
            ("schema_version",),
            float(
                coordinate_audit.ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION
            ),
            "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID",
        ),
        (
            "chunk",
            ("entry_count",),
            2.0,
            "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID",
        ),
        (
            "chunk",
            ("entries", 0, "chunk_indices", 0),
            0.0,
            "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID",
        ),
        (
            "chunk",
            ("entries", 0, "encoded_size_bytes"),
            0,
            "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_INVALID",
        ),
        (
            "manifest",
            ("schema_version",),
            float(
                coordinate_audit.ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
            ),
            "ACQUISITION_MATERIALIZATION_MANIFEST_INVALID",
        ),
    ],
)
def test_acquisition_manifest_rejects_type_coercions_after_all_digests_are_rebound(
    tmp_path: Path,
    record_name: str,
    field_path: tuple[str | int, ...],
    bad_value: object,
    expected_code: str,
) -> None:
    zarr_path = tmp_path / "redigested-malformed-acquisition.zarr"
    _write_node(zarr_path)
    fixture = _write_materialized_acquisition_fixture(zarr_path)
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }

    def mutate(record: dict[str, Any]) -> None:
        owner: Any = record
        for part in field_path[:-1]:
            owner = owner[part]
        owner[field_path[-1]] = bad_value

    candidate, pointer = _redigest_materialized_acquisition_nodes(
        nodes,
        fixture,
        mutate_chunk=mutate if record_name == "chunk" else None,
        mutate_manifest=mutate if record_name == "manifest" else None,
    )
    _target, issues = coordinate_audit._reference_extent_binding_issues(
        pointer,
        role="test.redigested_acquisition",
        nodes=candidate,
    )

    assert {issue["code"] for issue in issues} == {expected_code}


def test_miskeyed_array_owned_coordinate_descriptor_container_is_inventoried_fail_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "miskeyed-descriptor.zarr"
    _write_node(zarr_path)
    surface_path = "analysis/custom_geometry/run/points_xy"
    _write_array(
        zarr_path,
        surface_path,
        shape=[2, 2],
        attributes={
            "coordinate_descriptors": {
                "different_array": _complete_xy_descriptor()
            }
        },
    )
    registry = _make_registry(
        tmp_path / "miskeyed-descriptor.sqlite",
        [
            {
                "dataset_id": "miskeyed-descriptor",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    surfaces = _surface_rows(audit_registry(registry))

    assert len(surfaces) == 1
    assert surfaces[0]["surface_path"] == surface_path
    assert surfaces[0]["surface_type"] == "unclassified_geometry_candidate"
    assert surfaces[0]["status"] == "ambiguous_fail_closed"
    assert "ARRAY_COORDINATE_DESCRIPTORS_CONTAINER_INVALID" in surfaces[0][
        "issue_codes"
    ]


def test_array_owned_descriptor_container_rejects_valid_key_plus_unrelated_entry(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "descriptor-with-hidden-extra.zarr"
    _write_node(zarr_path)
    surface_path = "analysis/custom_geometry/run/points_xy"
    contradictory = _complete_xy_descriptor()
    contradictory["coordinate_space"] = "texture"
    _write_array(
        zarr_path,
        surface_path,
        shape=[2, 2],
        attributes={
            "coordinate_descriptors": {
                "points_xy": _complete_xy_descriptor(),
                "wrong_array": contradictory,
            }
        },
    )
    registry = _make_registry(
        tmp_path / "descriptor-with-hidden-extra.sqlite",
        [
            {
                "dataset_id": "descriptor-with-hidden-extra",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    surfaces = _surface_rows(audit_registry(registry))

    assert len(surfaces) == 1
    assert surfaces[0]["status"] == "ambiguous_fail_closed"
    assert "ARRAY_COORDINATE_DESCRIPTORS_CONTAINER_INVALID" in surfaces[0][
        "issue_codes"
    ]


def test_miskeyed_array_owned_descriptor_attr_is_inventoried_fail_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "miskeyed-descriptor-attr.zarr"
    _write_node(zarr_path)
    surface_path = "analysis/custom_geometry/run/points_xy"
    _write_array(
        zarr_path,
        surface_path,
        shape=[2, 2],
        attributes={
            "different_array_coordinate_descriptor": _complete_xy_descriptor()
        },
    )
    registry = _make_registry(
        tmp_path / "miskeyed-descriptor-attr.sqlite",
        [
            {
                "dataset_id": "miskeyed-descriptor-attr",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    surfaces = _surface_rows(audit_registry(registry))

    assert len(surfaces) == 1
    assert surfaces[0]["status"] == "ambiguous_fail_closed"
    assert "ARRAY_COORDINATE_DESCRIPTOR_ATTR_MISKEYED" in surfaces[0]["issue_codes"]


def test_stimulus_identity_is_canonical_and_coordinate_row_identity_is_legacy(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "stimulus-identity.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    rowset = "analysis/stimulus_runs/run/tracking_data/chaser_states"
    surface_path = f"{rowset}/chaser_pos_x"
    descriptor = _complete_arena_descriptor(zarr_path)
    descriptor["geometry_type"] = "coordinate_component"
    descriptor["components"] = ["x"]
    descriptor["component_units"] = ["px"]
    descriptor["row_identity"] = {
        "mode": "explicit_array",
        "array_ref": "stimulus_state_key",
    }
    _write_array(
        zarr_path,
        surface_path,
        shape=[2],
        attributes=_descriptor_attrs(descriptor),
    )
    _write_array(zarr_path, f"{rowset}/stimulus_state_key", shape=[2])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "stimulus-identity",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = next(
        row
        for row in _surface_rows(audit_registry(registry))
        if row["surface_path"] == surface_path
    )
    assert surface["status"] == "numerical_validation_required"
    assert "SURFACE_PROFILE_ROW_IDENTITY_UNSUPPORTED" not in surface["issue_codes"]
    assert "ROW_IDENTITY_KEY_PAYLOAD_VALIDATION_REQUIRED" in surface["issue_codes"]

    descriptor["row_identity"] = {
        "mode": "explicit_array",
        "array_ref": "coordinate_row_identity",
    }
    _write_array(
        zarr_path,
        surface_path,
        shape=[2],
        attributes=_descriptor_attrs(descriptor),
    )
    _write_array(zarr_path, f"{rowset}/coordinate_row_identity", shape=[2])
    _replace_group_attrs(zarr_path, rowset, {})
    surface = next(
        row
        for row in _surface_rows(audit_registry(registry))
        if row["surface_path"] == surface_path
    )
    assert surface["status"] == "numerical_validation_required"
    assert {
        "CANONICAL_ROW_IDENTITY_CONTRACT_MISSING",
        "LEGACY_COORDINATE_ROW_IDENTITY_REQUIRES_MIGRATION",
        "LEGACY_ROW_IDENTITY_REQUIRES_MIGRATION",
    } <= set(surface["issue_codes"])


def test_same_shape_array_without_controlled_authority_role_fails_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "unrelated-authority.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    _write_array(
        zarr_path,
        "analysis/unrelated_images",
        shape=[1, 480, 640],
    )
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    descriptor = _complete_xy_descriptor(space_id="roi_local_px")
    descriptor["reference_extent"]["authority"] = (
        "/analysis/unrelated_images.shape[-2:]"
    )
    descriptor["source_camera_overlay"] = "requires_transform"
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "unrelated-authority",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "REFERENCE_AUTHORITY_ROLE_INVALID" in surface["issue_codes"]


def test_track_px_mm_surfaces_require_one_exact_row_identity_contract(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "track-px-mm-identity.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    px_descriptor = _complete_arena_descriptor(zarr_path)
    mm_descriptor = json.loads(json.dumps(px_descriptor))
    mm_descriptor.update(
        {
            "space_id": "physical_mm",
            "component_units": ["mm", "mm"],
            "origin": "physical_frame_origin",
            "reference_extent": {
                "width": None,
                "height": None,
                "units": "not_applicable",
                "authority": "/coordinate_records/missing_physical_frame",
            },
            "pixel_convention": "continuous",
            "physical_frame": "arena-mm",
        }
    )
    mm_descriptor["row_identity"] = {
        "mode": "explicit_array",
        "array_ref": "frame_indices",
    }
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[2, 2],
        attributes=_descriptor_attrs(px_descriptor),
    )
    _write_array(
        zarr_path,
        f"{track}/positions_mm",
        shape=[2, 2],
        attributes=_descriptor_attrs(mm_descriptor),
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[2])
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "track-px-mm-identity",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surfaces = _surface_rows(audit_registry(registry))
    assert len(surfaces) == 2
    assert all(
        "TRACK_PX_MM_COORDINATE_CONTRACT_MISMATCH" in row["issue_codes"]
        for row in surfaces
    )


def test_canonical_v2_descriptor_resolves_exact_identity_and_authority_records(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "canonical-v2.zarr"
    _write_node(zarr_path)
    rowset = "analysis/refined_keypoints_runs/run"
    row_count = 2
    identity_values = np.arange(row_count, dtype=np.uint64)
    identity_contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=identity_values,
    )
    (
        acquisition_path,
        acquisition_attrs,
        frame_path,
        frame_attrs,
        frame_record,
        _acquisition,
        _token,
        root_attrs,
    ) = _sealed_source_camera_frame_attrs()
    _merge_node_attrs(zarr_path, ".", root_attrs)
    for path, attrs in (
        (acquisition_path, acquisition_attrs),
        (frame_path, frame_attrs),
    ):
        _ensure_groups(zarr_path, path)
        _write_node(zarr_path, path, attributes=attrs)
    frame_ref = f"/{frame_path}@{PIXEL_FRAME_AUTHORITY_ATTR}"
    authority = DigestBoundCoordinateRecordRef(
        record_ref=frame_ref,
        record_sha256=frame_record.record_sha256,
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=640,
        reference_height=480,
        reference_authority=authority,
        reference_selector="record",
        pixel_convention="pixel_center",
        row_identity_contract=identity_contract,
        row_identity_record_ref=f"/{rowset}@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame_ref,
            record_sha256=frame_record.record_sha256,
        ),
    )
    _write_canonical_row_identity(
        zarr_path,
        rowset,
        domain=OBSERVATION_INSTANCE_DOMAIN,
        row_count=row_count,
    )
    surface_path = f"{rowset}/keypoints_img"
    _write_array(
        zarr_path,
        surface_path,
        shape=[row_count, 2],
        attributes=canonical_coordinate_descriptor_v2_attrs(descriptor),
    )
    registry = _make_registry(
        tmp_path / "registry.sqlite",
        [
            {
                "dataset_id": "canonical-v2",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "numerical_validation_required"
    assert surface["coordinate_descriptor"]["schema_version"] == 2
    assert set(surface["issue_codes"]) == {
        "PIXEL_FRAME_AUTHORITY_LIVE_VALIDATION_REQUIRED",
        "ROW_IDENTITY_KEY_PAYLOAD_VALIDATION_REQUIRED",
    }

    forged = descriptor.to_dict()
    forged["row_identity"]["record_sha256"] = "0" * 64
    forged_attrs = {
        "coordinate_descriptor": forged,
        "coordinate_descriptor_sha256": canonical_coordinate_descriptor_v2_digest(
            forged
        ),
    }
    _write_array(
        zarr_path,
        surface_path,
        shape=[row_count, 2],
        attributes=forged_attrs,
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "ROW_IDENTITY_RECORD_DIGEST_MISMATCH" in surface["issue_codes"]

    forged = descriptor.to_dict()
    forged["reference_extent"]["width"] = 641
    forged_attrs = {
        "coordinate_descriptor": forged,
        "coordinate_descriptor_sha256": canonical_coordinate_descriptor_v2_digest(
            forged
        ),
    }
    _write_array(
        zarr_path,
        surface_path,
        shape=[row_count, 2],
        attributes=forged_attrs,
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "PIXEL_FRAME_AUTHORITY_EXTENT_MISMATCH" in surface["issue_codes"]

    forged = descriptor.to_dict()
    forged["reference_extent"]["authority"]["record_sha256"] = "f" * 64
    forged["lineage_refs"][0]["record_sha256"] = "f" * 64
    forged["frame_record"]["record_sha256"] = "f" * 64
    forged_attrs = {
        "coordinate_descriptor": forged,
        "coordinate_descriptor_sha256": canonical_coordinate_descriptor_v2_digest(
            forged
        ),
    }
    _write_array(
        zarr_path,
        surface_path,
        shape=[row_count, 2],
        attributes=forged_attrs,
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert "CANONICAL_RECORD_DIGEST_MISMATCH" in surface["issue_codes"]


def test_directed_transform_v2_requires_real_typed_endpoints_and_authority(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "directed-transform-v2.zarr"
    _write_node(zarr_path)
    (
        acquisition_path,
        acquisition_attrs,
        source_path,
        source_attrs,
        source_frame,
        _acquisition,
        token,
        root_attrs,
    ) = _sealed_source_camera_frame_attrs()
    _merge_node_attrs(zarr_path, ".", root_attrs)
    normalized_path = "analysis/coordinate_frames/camera_normalized"
    normalized_node = _MemoryNode(normalized_path, token=token)
    normalized_frame = stamp_normalized_pixel_frame_authority(
        normalized_node,
        frame_id="camera_1_normalized",
        pixel_frame=source_frame,
    )
    matrix_path = "analysis/transforms/normalized_to_camera"
    matrix_node = _MemoryNode(
        matrix_path,
        token=token,
        data=normalized_to_pixel_matrix(source_frame),
    )
    authority_path = "analysis/transforms/normalized_to_camera_authority"
    authority_node = _MemoryNode(authority_path, token=token)
    authority = stamp_normalized_to_pixel_transform_authority(
        authority_node,
        authority_id="camera_normalized_to_pixel",
        matrix_node=matrix_node,
        source_frame=normalized_frame,
        target_frame=source_frame,
    )
    transform = stamp_directed_transform_v2(
        matrix_node,
        transform_id="camera_normalized_to_pixel",
        authority=authority,
        source_frame=normalized_frame,
        target_frame=source_frame,
    )

    for path, attrs in (
        (acquisition_path, acquisition_attrs),
        (source_path, source_attrs),
        (normalized_path, dict(normalized_node.attrs)),
        (authority_path, dict(authority_node.attrs)),
    ):
        _ensure_groups(zarr_path, path)
        _write_node(zarr_path, path, attributes=attrs)
    _ensure_groups(zarr_path, matrix_path)
    _write_node(
        zarr_path,
        matrix_path,
        node_type="array",
        attributes=dict(matrix_node.attrs),
        shape=[3, 3],
        data_type="float64",
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    record_ref = f"/{matrix_path}@{DIRECTED_TRANSFORM_V2_ATTR}"
    parsed, issues = coordinate_audit._parse_directed_transform_v2_node(
        nodes[matrix_path],
        record_ref=record_ref,
        nodes=nodes,
    )
    assert parsed == transform.transform
    assert [issue["code"] for issue in issues] == [
        "DIRECTED_TRANSFORM_V2_LIVE_VALIDATION_REQUIRED"
    ]
    assert coordinate_audit.classify_surface(
        matrix_path,
        nodes[matrix_path],
        nodes,
    ) == "directed_affine_2d_constant"
    classified = coordinate_audit.classify_surface_contract(
        surface_type="directed_affine_2d_constant",
        node=nodes[matrix_path],
        nodes=nodes,
    )
    assert classified["status"] == "numerical_validation_required"
    assert [issue["code"] for issue in classified["issues"]] == [
        "DIRECTED_TRANSFORM_V2_LIVE_VALIDATION_REQUIRED"
    ]

    forged = transform.transform.to_dict()
    forged.pop("transform_authority")
    _write_node(
        zarr_path,
        matrix_path,
        node_type="array",
        attributes={
            DIRECTED_TRANSFORM_V2_ATTR: forged,
            DIRECTED_TRANSFORM_V2_DIGEST_ATTR: coordinate_audit._fingerprint(
                forged
            ),
        },
        shape=[3, 3],
        data_type="float64",
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    parsed, issues = coordinate_audit._parse_directed_transform_v2_node(
        nodes[matrix_path],
        record_ref=record_ref,
        nodes=nodes,
    )
    assert parsed is None
    assert [issue["code"] for issue in issues] == [
        "DIRECTED_TRANSFORM_V2_METADATA_INVALID"
    ]


def test_directed_transform_v2_inventory_uses_exact_kind_specific_roles(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "transform-kind-inventory.zarr"
    _write_node(zarr_path)
    fixtures = {
        "analysis/transforms/projective": (
            HOMOGRAPHY_KIND,
            [3, 3],
            "directed_projective_homography",
        ),
        "analysis/transforms/constant_affine": (
            AFFINE_2D_CONSTANT_KIND,
            [3, 3],
            "directed_affine_2d_constant",
        ),
        "analysis/transforms/rowwise_affine": (
            AFFINE_2D_ROWWISE_KIND,
            [7, 4],
            "directed_affine_2d_rowwise",
        ),
    }
    for path, (kind, shape, _surface_type) in fixtures.items():
        _write_array(
            zarr_path,
            path,
            shape=shape,
            data_type="float64",
            attributes={DIRECTED_TRANSFORM_V2_ATTR: {"kind": kind}},
        )
    invalid_path = "analysis/transforms/unknown"
    _write_array(
        zarr_path,
        invalid_path,
        shape=[3, 3],
        data_type="float64",
        attributes={DIRECTED_TRANSFORM_V2_ATTR: {"kind": "calibrationish"}},
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    assert {
        path: coordinate_audit.classify_surface(path, nodes[path], nodes)
        for path in fixtures
    } == {
        path: surface_type
        for path, (_kind, _shape, surface_type) in fixtures.items()
    }
    assert coordinate_audit.classify_surface(
        invalid_path,
        nodes[invalid_path],
        nodes,
    ) == "directed_transform_v2_invalid"
    identities = coordinate_audit._expected_surface_identities(
        tuple(nodes.values())
    )
    assert len(identities) == 4
    assert not any(
        row["surface_type"] == "calibration_homography"
        for row in identities
    )


def test_rowwise_transform_identity_checks_live_key_path_shape_and_dtype(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "rowwise-identity.zarr"
    _write_node(zarr_path)
    rowset = "analysis/crop_runs/run"
    _write_canonical_row_identity(
        zarr_path,
        rowset,
        domain=OBSERVATION_INSTANCE_DOMAIN,
        row_count=2,
    )
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    contract = coordinate_audit.load_row_identity_contract_attrs(
        nodes[rowset].attributes
    )
    transform = SimpleNamespace(
        row_identity=AuthorityRowIdentity(
            record_ref=f"/{rowset}@row_identity_contract",
            record_sha256=contract.digest(),
            leading_dimension=2,
        )
    )
    assert coordinate_audit._directed_transform_v2_identity_issues(
        transform,
        nodes=nodes,
    ) == []

    key_path = f"{rowset}/instance_key"
    wrong_dtype = dict(nodes)
    wrong_dtype[key_path] = replace(
        nodes[key_path],
        data_type="uint32",
    )
    issues = coordinate_audit._directed_transform_v2_identity_issues(
        transform,
        nodes=wrong_dtype,
    )
    assert [issue["code"] for issue in issues] == [
        "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_KEY_METADATA_MISMATCH"
    ]

    wrong_path = dict(nodes)
    wrong_path[key_path] = replace(
        nodes[key_path],
        relative_path="analysis/crop_runs/other/instance_key",
        shape=[3],
    )
    issues = coordinate_audit._directed_transform_v2_identity_issues(
        transform,
        nodes=wrong_path,
    )
    assert [issue["code"] for issue in issues] == [
        "DIRECTED_TRANSFORM_V2_ROW_IDENTITY_KEY_METADATA_MISMATCH"
    ]


def test_archive_root_symlink_root_array_and_invalid_v2_metadata_fail_closed(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real.zarr"
    _write_node(real_root)
    linked_root = tmp_path / "linked.zarr"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(
        coordinate_audit.MetadataTraversalError,
        match="symlinked archive root is forbidden",
    ):
        list(coordinate_audit.iter_metadata_nodes(linked_root))

    root_array = tmp_path / "root-array.zarr"
    _write_node(root_array, node_type="array", shape=[2, 2])
    registry = _make_registry(
        tmp_path / "root-array.sqlite",
        [
            {
                "dataset_id": "root-array",
                "recording_id": "rec",
                "zarr_path": str(root_array),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    root_record = _dataset_rows(audit_registry(registry))[0]
    assert root_record["status"] == "missing_or_unreadable"
    assert "ZARR_ROOT_ARRAY_FORBIDDEN" in root_record["issue_codes"]

    invalid_v2 = tmp_path / "invalid-v2.zarr"
    invalid_v2.mkdir()
    (invalid_v2 / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [1, -2],
                "chunks": [1, 1],
                "dtype": "definitely-not-a-dtype",
            }
        ),
        encoding="utf-8",
    )
    (invalid_v2 / ".zattrs").write_text(
        json.dumps({"dataset_id": "invalid-v2", "recording_id": "rec"}),
        encoding="utf-8",
    )
    registry = _make_registry(
        tmp_path / "invalid-v2.sqlite",
        [
            {
                "dataset_id": "invalid-v2",
                "recording_id": "rec",
                "zarr_path": str(invalid_v2),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    v2_record = _dataset_rows(audit_registry(registry))[0]
    assert v2_record["status"] == "missing_or_unreadable"
    assert "INVALID_ZARR_METADATA_INVENTORY" in v2_record["issue_codes"]


def test_registry_foreign_key_defects_and_archive_aliases_fail_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "shared.zarr"
    _write_node(
        zarr_path,
        attributes={"dataset_id": "orphan", "recording_id": "missing-rec"},
    )
    registry = tmp_path / "foreign-key.sqlite"
    conn = sqlite3.connect(registry)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("CREATE TABLE recordings (recording_id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE datasets (dataset_id TEXT PRIMARY KEY, recording_id TEXT REFERENCES recordings(recording_id), zarr_path TEXT, zarr_use TEXT, artifact_kind TEXT, status TEXT)"
        )
        conn.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
            ("orphan", "missing-rec", str(zarr_path), "analysis", "analysis", "active"),
        )
        conn.commit()
    finally:
        conn.close()
    dataset = _dataset_rows(audit_registry(registry))[0]
    assert dataset["status"] == "ambiguous_fail_closed"
    assert "REGISTRY_DATASET_FOREIGN_KEY_INVALID" in dataset["issue_codes"]

    alias_registry = _make_registry(
        tmp_path / "alias.sqlite",
        [
            {
                "dataset_id": "alias-a",
                "recording_id": "rec-a",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "alias-b",
                "recording_id": "rec-b",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
        ],
    )
    aliases = _dataset_rows(audit_registry(alias_registry))
    assert all("REGISTRY_ZARR_ARCHIVE_ALIAS" in row["issue_codes"] for row in aliases)
    assert all(row["status"] == "ambiguous_fail_closed" for row in aliases)


def test_run_pointer_and_completion_schema_disagreement_fails_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "run-pointer.zarr"
    _write_node(zarr_path)
    _ensure_groups(zarr_path, "analysis/detect_runs/run-a/placeholder")
    _write_node(
        zarr_path,
        "analysis/detect_runs",
        attributes={
            "palette_completion_epoch": 1,
            "latest_complete": "run-a",
        },
    )
    _write_node(
        zarr_path,
        "analysis/detect_runs/run-a",
        attributes={"palette_run_completion_status": "running"},
    )
    registry = _make_registry(
        tmp_path / "run-pointer.sqlite",
        [
            {
                "dataset_id": "run-pointer",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    dataset = _dataset_rows(audit_registry(registry))[0]
    assert dataset["status"] == "ambiguous_fail_closed"
    assert {
        "RUN_POINTER_COMPLETION_MISMATCH",
        "RUN_COMPLETION_SCHEMA_INVALID",
    } <= set(dataset["issue_codes"])


def test_partitioned_run_pointers_validate_runs_not_partition_groups(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "partitioned-runs.zarr"
    _write_node(zarr_path)
    _ensure_groups(zarr_path, "analysis/track_kinematics_runs/offline/run-a/tracks")
    _write_node(
        zarr_path,
        "analysis/track_kinematics_runs",
        attributes={
            "palette_completion_epoch": 1,
            "latest": "offline/run-a",
            "latest_complete": "offline/run-a",
            "latest_offline": "run-a",
        },
    )
    _write_node(
        zarr_path,
        "analysis/track_kinematics_runs/offline/run-a",
        attributes={
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "palette_run_name": "run-a",
        },
    )
    registry = _make_registry(
        tmp_path / "partitioned-runs.sqlite",
        [
            {
                "dataset_id": "partitioned-runs",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )

    dataset = _dataset_rows(audit_registry(registry))[0]

    assert "RUN_POINTER_UNRESOLVED" not in dataset["issue_codes"]
    assert "RUN_POINTER_COMPLETION_MISMATCH" not in dataset["issue_codes"]
    assert "RUN_COMPLETION_SCHEMA_INVALID" not in dataset["issue_codes"]


def test_discovery_uses_explicit_producer_schema_and_semantic_role_rules(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "explicit-discovery.zarr"
    _write_node(zarr_path)
    explicit = {
        "analysis/stimulus_runs/run/tracking_data/chaser_states/chaser_position_xy": (
            "stimulus_chaser_position",
            {"semantic_role": "chaser_position"},
            [2, 2],
        ),
        "analysis/stimulus_runs/run/tracking_data/chaser_states/target_position_xy": (
            "stimulus_target_position",
            {"semantic_role": "target_position"},
            [2, 2],
        ),
        "analysis/stimulus_runs/run/tracking_data/chaser_states/target_clamped_position_xy": (
            "stimulus_target_clamped_position",
            {"semantic_role": "target_clamped_position"},
            [2, 2],
        ),
        "analysis/stimulus_runs/run/tracking_data/bounding_boxes/centroid_x": (
            "stimulus_bbox_component",
            {},
            [2],
        ),
        "analysis/detection_occupancy_runs/run/spatial_occupancy/quadrants/zone_spec/bounds_xyxy": (
            "occupancy_zone_bounds",
            {},
            [4, 4],
        ),
        "analysis/keypoints_runs/run/pose_bbox_xyxy_roi": (
            "keypoint_pose_bbox",
            {},
            [2, 4],
        ),
        "analysis/subject_shape_runs/run/body_frame/origin_xy": (
            "body_frame_origin_geometry",
            {},
            [2, 2],
        ),
        "analysis/subject_shape_runs/run/body_frame/forward_axis_xy": (
            "body_frame_axis_geometry",
            {},
            [2, 2],
        ),
        "analysis/subject_shape_runs/run/centerline_spline_xy": (
            "subject_shape_geometry",
            {},
            [2, 8, 2],
        ),
        "analysis/refined_subject_masks_runs/run/masks_roi": (
            "subject_mask_raster",
            {},
            [2, 16, 16],
        ),
        "analysis/refined_subject_masks_runs/run/components/body/source_seed_masks_roi": (
            "subject_mask_seed_raster",
            {},
            [2, 16, 16],
        ),
        "analysis/refined_subject_masks_runs/run/mask_bitpacked/masks_packed": (
            "subject_mask_compact_encoding",
            {},
            [2, 32],
        ),
        "analysis/refined_subject_masks_runs/run/metrics/body/bbox_xyxy": (
            "subject_mask_metric_geometry",
            {},
            [2, 4],
        ),
        "analysis/refined_subject_masks_runs/run/metrics/body/centroid_xy": (
            "subject_mask_metric_geometry",
            {},
            [2, 2],
        ),
        "analysis/refined_subject_masks_runs/run/components/body/geometry/ellipse_params": (
            "subject_mask_component_geometry",
            {},
            [2, 5],
        ),
        "analysis/refined_subject_masks_runs/run/components/body/mask_rle/bbox_xyxy": (
            "subject_mask_metric_geometry",
            {},
            [2, 4],
        ),
        "analysis/refined_subject_masks_runs/run/components/body/contours/points_xy": (
            "subject_mask_contour",
            {},
            [8, 2],
        ),
    }
    for path, (_surface_type, attrs, shape) in explicit.items():
        _write_array(zarr_path, path, attributes=attrs, shape=shape)

    false_positives = {
        "analysis/subject_shape_runs/run/body_frame/valid": ("bool", [2]),
        "analysis/subject_shape_runs/run/body_frame/failure_reason_bytes": (
            "uint8",
            [2, 32],
        ),
        "analysis/detection_occupancy_runs/run/coverage/coverage_pct": (
            "float32",
            [2],
        ),
        "analysis/detection_occupancy_runs/run/visualizations/detection_occupancy_overview_png": (
            "uint8",
            [128],
        ),
        "analysis/calibration/homography_png_buffer": ("uint8", [3, 3]),
        "analysis/calibration/homography_png_bytes": ("uint8", [3, 3]),
        "analysis/calibration/untyped_matrix": ("float64", [3, 3]),
        "analysis/calibration/homography_media": ("uint8", [3, 3]),
        "analysis/new_stage/run/centroid_signal": ("float32", [2]),
        "analysis/new_stage/run/foo_bbox_vertices": ("float32", [2, 4]),
    }
    for path, (dtype, shape) in false_positives.items():
        attrs = (
            {
                "artifact_schema_id": "palette.visualization.png_bytes.v1",
                "media_type": "image/png",
                "storage_encoding": "png_bytes_uint8",
                **_descriptor_attrs(_complete_xy_descriptor()),
            }
            if path.endswith("_png")
            else {}
        )
        if path.endswith("homography_media"):
            attrs = {"media_type": "image/png"}
        _write_array(zarr_path, path, attributes=attrs, shape=shape, data_type=dtype)

    node_map = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    classified = {
        path: coordinate_audit.classify_surface(path, node_map[path], node_map)
        for path in [*explicit, *false_positives]
    }
    assert all(
        classified[path] == surface_type
        for path, (surface_type, _attrs, _shape) in explicit.items()
    )
    declared_media = (
        "analysis/detection_occupancy_runs/run/visualizations/"
        "detection_occupancy_overview_png"
    )
    assert classified[declared_media] == "unclassified_geometry_candidate"
    assert all(
        classified[path] is None
        for path in false_positives
        if path != declared_media
    )


def test_flattened_subject_contours_require_exact_ptr_len_lineage(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "contours-missing.zarr"
    _write_node(missing)
    points_path = (
        "analysis/refined_subject_masks_runs/run/components/body/"
        "contours/points_xy"
    )
    _write_array(missing, points_path, shape=[8, 2])
    missing_nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(missing)
    }
    missing_issues = coordinate_audit._flattened_contour_lineage_issues(
        surface_type="subject_mask_contour",
        node=missing_nodes[points_path],
        nodes=missing_nodes,
    )
    assert {issue["code"] for issue in missing_issues} == {
        "FLATTENED_CONTOUR_SCHEMA_MISSING",
        "FLATTENED_CONTOUR_INDEX_LINEAGE_MISSING",
    }

    valid = tmp_path / "contours-valid.zarr"
    _write_node(valid)
    _write_array(valid, points_path, shape=[8, 2])
    contour_group = str(Path(points_path).parent)
    _merge_node_attrs(
        valid,
        contour_group,
        {
            "schema_id": "component_contours_v1",
            "contour_schema_id": "component_contours_v1",
        },
    )
    _write_array(valid, f"{contour_group}/ptr", shape=[2], data_type="int64")
    _write_array(valid, f"{contour_group}/len", shape=[2], data_type="int32")
    valid_nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(valid)
    }
    valid_issues = coordinate_audit._flattened_contour_lineage_issues(
        surface_type="subject_mask_contour",
        node=valid_nodes[points_path],
        nodes=valid_nodes,
    )
    assert [issue["code"] for issue in valid_issues] == [
        "FLATTENED_CONTOUR_INDEX_PAYLOAD_VALIDATION_REQUIRED"
    ]

    _write_array(valid, f"{contour_group}/len", shape=[3], data_type="float32")
    invalid_nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(valid)
    }
    invalid_issues = coordinate_audit._flattened_contour_lineage_issues(
        surface_type="subject_mask_contour",
        node=invalid_nodes[points_path],
        nodes=invalid_nodes,
    )
    assert [issue["code"] for issue in invalid_issues] == [
        "FLATTENED_CONTOUR_INDEX_LINEAGE_INVALID"
    ]


def test_mask_stage_specs_inventory_legacy_modern_and_complete_rle_surfaces(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "mask-stage-inventory.zarr"
    _write_node(zarr_path)

    for stage, probability_leaf in (
        ("eye_masks_runs", "mask_probs_roi"),
        ("refined_eye_masks_runs", "mask_probs_roi_refined"),
    ):
        run = f"analysis/{stage}/run"
        for leaf, shape in (
            ("masks_roi", [2, 2, 16, 16]),
            (probability_leaf, [2, 2, 16, 16]),
            ("ellipse_params", [2, 2, 5]),
            ("contours_left", [8, 2]),
            ("contours_right", [9, 2]),
        ):
            _write_array(zarr_path, f"{run}/{leaf}", shape=shape)
        for side in ("left", "right"):
            _write_array(
                zarr_path,
                f"{run}/contour_{side}_ptr",
                shape=[2],
                data_type="int32",
            )
            _write_array(
                zarr_path,
                f"{run}/contour_{side}_len",
                shape=[2],
                data_type="int32",
            )

    run = "analysis/refined_subject_masks_runs/run"
    modern = {
        f"{run}/masks_roi": ([2, 3, 16, 16], "float32"),
        f"{run}/mask_probs_roi": ([2, 3, 16, 16], "float32"),
        f"{run}/components/body/source_seed_masks_roi": (
            [2, 16, 16],
            "uint8",
        ),
        f"{run}/components/body/geometry/ellipse_params": (
            [2, 5],
            "float32",
        ),
        f"{run}/components/body/contours/points_xy": ([8, 2], "float32"),
        f"{run}/metrics/bbox_xyxy": ([2, 3, 4], "float32"),
        f"{run}/metrics/centroid_xy": ([2, 3, 2], "float32"),
        f"{run}/mask_bitpacked/masks_packed": ([2, 3, 16, 2], "uint8"),
    }
    for path, (shape, dtype) in modern.items():
        _write_array(zarr_path, path, shape=shape, data_type=dtype)
    contour_group = f"{run}/components/body/contours"
    _merge_node_attrs(
        zarr_path,
        contour_group,
        {
            "schema_id": "component_contours_v1",
            "contour_schema_id": "component_contours_v1",
        },
    )
    _write_array(
        zarr_path,
        f"{contour_group}/ptr",
        shape=[2],
        data_type="int64",
    )
    _write_array(
        zarr_path,
        f"{contour_group}/len",
        shape=[2],
        data_type="int32",
    )

    rle = f"{run}/mask_rle"
    _write_node(
        zarr_path,
        rle,
        attributes={
            "schema_id": "palette_mask_rle_binary_v1",
            "mask_encoding": "coco_rle_fortran_v1",
            "layout": "component_groups",
            "encoded_shape_hw": [16, 16],
        },
    )
    component = f"{rle}/components/00_body"
    _write_array(
        zarr_path,
        f"{component}/counts",
        shape=[37],
        data_type="uint32",
    )
    _write_array(
        zarr_path,
        f"{component}/indptr",
        shape=[3],
        data_type="int64",
    )
    _write_array(
        zarr_path,
        f"{component}/present",
        shape=[2],
        data_type="bool",
    )
    _write_array(
        zarr_path,
        f"{component}/bbox_xyxy",
        shape=[2, 4],
        data_type="int32",
    )

    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    identities = coordinate_audit._expected_surface_identities(
        tuple(nodes.values())
    )
    counts = {
        surface_type: sum(
            row["surface_type"] == surface_type for row in identities
        )
        for surface_type in {row["surface_type"] for row in identities}
    }
    assert len(identities) == 21
    assert counts == {
        "subject_mask_raster": 6,
        "subject_mask_component_geometry": 3,
        "subject_mask_contour": 5,
        "subject_mask_seed_raster": 1,
        "subject_mask_metric_geometry": 3,
        "subject_mask_compact_encoding": 3,
    }
    for leaf in ("counts", "indptr"):
        path = f"{component}/{leaf}"
        assert [
            issue["code"]
            for issue in coordinate_audit._subject_mask_rle_lineage_issues(
                surface_type="subject_mask_compact_encoding",
                node=nodes[path],
                nodes=nodes,
            )
        ] == ["SUBJECT_MASK_RLE_PAYLOAD_VALIDATION_REQUIRED"]
    legacy_contour = (
        "analysis/refined_eye_masks_runs/run/contours_left"
    )
    assert coordinate_audit._surface_leading_dimension(
        nodes[legacy_contour],
        nodes=nodes,
        excluded_paths=set(),
    ) == (2, [])
    assert [
        issue["code"]
        for issue in coordinate_audit._flattened_contour_lineage_issues(
            surface_type="subject_mask_contour",
            node=nodes[legacy_contour],
            nodes=nodes,
        )
    ] == ["FLATTENED_CONTOUR_INDEX_PAYLOAD_VALIDATION_REQUIRED"]
    for leaf in ("counts", "indptr"):
        assert coordinate_audit._surface_leading_dimension(
            nodes[f"{component}/{leaf}"],
            nodes=nodes,
            excluded_paths=set(),
        ) == (2, [])

    _replace_group_attrs(zarr_path, rle, {})
    nodes = {
        node.relative_path: node
        for node in coordinate_audit.iter_metadata_nodes(zarr_path)
    }
    assert [
        issue["code"]
        for issue in coordinate_audit._subject_mask_rle_lineage_issues(
            surface_type="subject_mask_compact_encoding",
            node=nodes[f"{component}/counts"],
            nodes=nodes,
        )
    ] == ["SUBJECT_MASK_RLE_LINEAGE_INVALID"]


def test_refined_online_and_track_outputs_require_domain_specific_identity(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "domain-identities.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    refined = "analysis/refined_online_runs/run"
    refined_descriptor = _complete_arena_descriptor(zarr_path)
    refined_descriptor["row_identity"] = {
        "mode": "instance_key",
        "array_ref": "instance_key",
    }
    _write_array(
        zarr_path,
        f"{refined}/positions_px",
        shape=[2, 2],
        attributes=_descriptor_attrs(refined_descriptor),
    )
    _write_canonical_row_identity(
        zarr_path,
        refined,
        domain=OBSERVATION_INSTANCE_DOMAIN,
        row_count=2,
    )
    registry = _make_registry(
        tmp_path / "domain-identities.sqlite",
        [
            {
                "dataset_id": "domain-identities",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    refined_surface = _surface_rows(audit_registry(registry))[0]
    assert "SURFACE_PROFILE_ROW_IDENTITY_UNSUPPORTED" in refined_surface["issue_codes"]

    refined_descriptor["row_identity"] = {
        "mode": "explicit_array",
        "array_ref": "stimulus_state_key",
    }
    _write_array(
        zarr_path,
        f"{refined}/positions_px",
        shape=[2, 2],
        attributes=_descriptor_attrs(refined_descriptor),
    )
    _write_array(zarr_path, f"{refined}/stimulus_state_key", shape=[2])
    refined_surface = _surface_rows(audit_registry(registry))[0]
    assert "SURFACE_PROFILE_ROW_IDENTITY_UNSUPPORTED" not in refined_surface["issue_codes"]
    assert refined_surface["status"] != "compatible"

    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[2, 2],
        attributes=_descriptor_attrs(refined_descriptor),
    )
    _write_array(zarr_path, f"{track}/stimulus_state_key", shape=[2])
    track_surface = next(
        row
        for row in _surface_rows(audit_registry(registry))
        if row["surface_path"] == f"{track}/positions_px"
    )
    assert "SURFACE_PROFILE_ROW_IDENTITY_UNSUPPORTED" in track_surface["issue_codes"]


def test_run_context_is_slash_qualified_and_pointer_completion_is_specific(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "run-context.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    run = "analysis/track_kinematics_runs/offline/run-a"
    track = f"{run}/tracks/id_0"
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(_complete_arena_descriptor(zarr_path)),
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    _replace_group_attrs(
        zarr_path,
        "analysis/track_kinematics_runs",
        {"latest": "offline/run-a", "latest_offline": "run-a"},
    )
    _merge_node_attrs(
        zarr_path,
        run,
        {
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "running",
            "publication_status": "staged",
        },
    )
    registry = _make_registry(
        tmp_path / "run-context.sqlite",
        [
            {
                "dataset_id": "run-context",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    records = audit_registry(registry)
    dataset = _dataset_rows(records)[0]
    surface = _surface_rows(records)[0]
    assert "RUN_POINTER_COMPLETION_MISMATCH" not in dataset["issue_codes"]
    assert surface["run_context"] == {
        "family": "track_kinematics_runs",
        "family_path": "analysis/track_kinematics_runs",
        "partition": "offline",
        "run_path": run,
        "run_name": "offline/run-a",
        "run_leaf_name": "run-a",
        "completion_contract": "palette.zarr_run_completion.v1",
        "completion_status": "running",
        "publication_status": "staged",
        "pointer_set": {
            "selected": [],
            "latest": ["latest", "latest_offline"],
            "authoritative": [],
            "all_matching": ["latest", "latest_offline"],
            "records": [
                {
                    "pointer": "latest",
                    "value": "offline/run-a",
                    "target_path": run,
                    "completion_required": False,
                },
                {
                    "pointer": "latest_offline",
                    "value": "run-a",
                    "target_path": run,
                    "completion_required": False,
                },
            ],
        },
    }

    _merge_node_attrs(
        zarr_path,
        "analysis/track_kinematics_runs",
        {"latest_complete": "offline/run-a"},
    )
    dataset = _dataset_rows(audit_registry(registry))[0]
    assert "RUN_POINTER_COMPLETION_MISMATCH" in dataset["issue_codes"]


def test_descriptor_cannot_erase_known_producer_numerical_risk(tmp_path: Path) -> None:
    zarr_path = tmp_path / "producer-risk.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    run = "analysis/track_kinematics_runs/offline/run"
    track = f"{run}/tracks/id_0"
    descriptor = _complete_xy_descriptor()
    _replace_group_attrs(
        zarr_path,
        run,
        {
            "method": "track_kinematics_offline",
            "position_source_kind": "crop_rows",
            "coordinate_space": "camera",
            "position_source_path": "crop_runs/source",
        },
    )
    _write_array(
        zarr_path,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(descriptor),
    )
    _write_array(zarr_path, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "producer-risk.sqlite",
        [
            {
                "dataset_id": "producer-risk",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert (
        "OFFLINE_CROP_SOURCE_RECONSTRUCTION_NUMERICAL_VALIDATION_REQUIRED"
        in surface["issue_codes"]
    )
    assert surface["status"] != "compatible"


def test_conflicting_ancestor_descriptors_and_nonnumeric_geometry_fail_closed(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "descriptor-contamination.zarr"
    _write_node(zarr_path)
    _write_contract_evidence(zarr_path)
    run = "analysis/detect_runs/run"
    direct = _complete_xy_descriptor(space_id="detector_normalized_xy", units="normalized")
    direct["geometry_type"] = "bbox_xyxy"
    ancestor = _complete_xy_descriptor(space_id="source_camera_image_px")
    _ensure_groups(zarr_path, f"{run}/bbox_norm_coords")
    _replace_group_attrs(zarr_path, run, {"coordinate_descriptor": ancestor})
    _write_array(
        zarr_path,
        f"{run}/bbox_norm_coords",
        shape=[1, 4],
        data_type="U8",
        attributes=_unvalidated_descriptor_attrs(direct),
    )
    registry = _make_registry(
        tmp_path / "descriptor-contamination.sqlite",
        [
            {
                "dataset_id": "descriptor-contamination",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    surface = _surface_rows(audit_registry(registry))[0]
    assert surface["status"] == "ambiguous_fail_closed"
    assert {
        "MULTIPLE_COORDINATE_DESCRIPTORS_CONFLICT",
        "GENERIC_ANCESTOR_DESCRIPTOR_CONTAMINATION",
        "COORDINATE_ARRAY_DTYPE_NONNUMERIC",
    } <= set(surface["issue_codes"])

    _write_array(
        zarr_path,
        "analysis/calibration/homography_matrix",
        shape=[3, 3],
        data_type="U8",
    )
    homography = next(
        row
        for row in _surface_rows(audit_registry(registry))
        if row["surface_type"] == "calibration_homography"
    )
    assert "HOMOGRAPHY_DTYPE_NONNUMERIC" in homography["issue_codes"]


def test_hidden_sidecars_are_ignored_only_after_nested_metadata_is_excluded(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "nested-sidecar.zarr"
    _write_node(zarr_path)
    nested = zarr_path / "logs" / "archived" / "hidden-run"
    nested.mkdir(parents=True)
    (nested / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}),
        encoding="utf-8",
    )
    with pytest.raises(
        coordinate_audit.MetadataTraversalError,
        match="contains nested Zarr metadata",
    ):
        list(coordinate_audit.iter_metadata_nodes(zarr_path))


def test_training_without_recording_id_is_valid_and_migration_targets_are_hierarchical(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_node(analysis)
    _write_node(training)
    _write_contract_evidence(analysis)
    track = "analysis/track_kinematics_runs/offline/run/tracks/id_0"
    _write_array(
        analysis,
        f"{track}/positions_px",
        shape=[1, 2],
        attributes=_descriptor_attrs(_complete_arena_descriptor(analysis)),
    )
    _write_array(analysis, f"{track}/frame_indices", shape=[1])
    registry = _make_registry(
        tmp_path / "hierarchy.sqlite",
        [
            {
                "dataset_id": "analysis",
                "recording_id": "rec-a",
                "zarr_path": str(analysis),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "training",
                "recording_id": None,
                "zarr_path": str(training),
                "zarr_use": "training",
                "artifact_kind": "derived_training_merge",
                "status": "active",
            },
        ],
        recording_rows=[
            {
                "recording_id": "rec-a",
                "recording_path": "/recordings/duplicate",
                "session_uuid": "a",
            },
            {
                "recording_id": "rec-b",
                "recording_path": "/recordings/duplicate",
                "session_uuid": "b",
            },
        ],
    )
    records = audit_registry(registry)
    training_record = next(
        row for row in _dataset_rows(records) if row["dataset_id"] == "training"
    )
    assert "REGISTRY_DATASET_RECORDING_IDENTITY_INVALID" not in training_record["issue_codes"]
    snapshot = coordinate_audit.build_registry_snapshot(registry, records)
    manifest = coordinate_audit.build_migration_manifest(
        records,
        registry_snapshot=snapshot,
    )
    assert {"registry", "recording", "archive", "run", "coordinate_surface"} <= {
        row["target_kind"] for row in manifest
    }
    assert any(
        row["target_kind"] == "recording"
        and "REGISTRY_DUPLICATE_RECORDING_PATH" in row["issue_codes"]
        for row in manifest
    )
    assert not all(row["must_fail_closed"] for row in manifest)


def test_registry_dataset_roles_use_exact_controlled_vocabulary(
    tmp_path: Path,
) -> None:
    specifications = {
        "source": ("source", "analysis", "source_recording", "rec"),
        "imported-source": (
            "imported",
            "analysis",
            "source_recording",
            "rec-imported",
        ),
        "derived-analysis": (
            "derived",
            "analysis",
            "derived_analysis",
            None,
        ),
        "derived-training": (
            "derived",
            "training",
            "derived_training_merge",
            None,
        ),
        "model-export": (
            "derived",
            "export",
            "model_input_export",
            None,
        ),
        "unknown": ("elsewhere", "not_training", "underived", None),
        "use-conflict": (
            "derived",
            "analysis",
            "derived_training_merge",
            None,
        ),
        "source-origin-conflict": (
            "derived",
            "analysis",
            "source_recording",
            "rec-conflict",
        ),
        "derived-origin-conflict": (
            "source",
            "analysis",
            "derived_analysis",
            None,
        ),
    }
    rows: list[dict[str, object]] = []
    for dataset_id, (zarr_origin, zarr_use, artifact_kind, recording_id) in (
        specifications.items()
    ):
        zarr_path = tmp_path / f"{dataset_id}.zarr"
        _write_node(zarr_path)
        rows.append(
            {
                "dataset_id": dataset_id,
                "recording_id": recording_id,
                "zarr_path": str(zarr_path),
                "zarr_origin": zarr_origin,
                "zarr_use": zarr_use,
                "artifact_kind": artifact_kind,
                "status": "active",
            }
        )
    registry = _make_registry(tmp_path / "roles.sqlite", rows)
    records = {
        row["dataset_id"]: row for row in _dataset_rows(audit_registry(registry))
    }
    for dataset_id in (
        "source",
        "imported-source",
        "derived-analysis",
        "derived-training",
        "model-export",
    ):
        assert not {
            "REGISTRY_ZARR_USE_UNCONTROLLED",
            "REGISTRY_ZARR_ORIGIN_UNCONTROLLED",
            "REGISTRY_ARTIFACT_KIND_UNCONTROLLED",
            "REGISTRY_DATASET_ROLE_CONFLICT",
            "REGISTRY_DATASET_ORIGIN_CONFLICT",
            "REGISTRY_DATASET_RECORDING_IDENTITY_INVALID",
        } & set(records[dataset_id]["issue_codes"])
    assert {
        "REGISTRY_ZARR_USE_UNCONTROLLED",
        "REGISTRY_ZARR_ORIGIN_UNCONTROLLED",
        "REGISTRY_ARTIFACT_KIND_UNCONTROLLED",
        "REGISTRY_DATASET_RECORDING_IDENTITY_INVALID",
    } <= set(records["unknown"]["issue_codes"])
    assert records["unknown"]["status"] == "ambiguous_fail_closed"
    assert "REGISTRY_DATASET_ROLE_CONFLICT" in records["use-conflict"][
        "issue_codes"
    ]
    assert records["use-conflict"]["status"] == "ambiguous_fail_closed"
    for dataset_id in ("source-origin-conflict", "derived-origin-conflict"):
        assert "REGISTRY_DATASET_ORIGIN_CONFLICT" in records[dataset_id][
            "issue_codes"
        ]
        assert records[dataset_id]["status"] == "ambiguous_fail_closed"


def test_checkpoint_binds_scanner_ruleset_dirty_state_and_bundle_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "binding.zarr"
    _write_node(zarr_path)
    _write_array(
        zarr_path,
        "analysis/detect_runs/run/bbox_norm_coords",
        shape=[1, 4],
    )
    registry = _make_registry(
        tmp_path / "binding.sqlite",
        [
            {
                "dataset_id": "binding",
                "recording_id": "rec",
                "zarr_path": str(zarr_path),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            }
        ],
    )
    checkpoint_dir = tmp_path / "checkpoints"
    first = audit_registry(registry, checkpoint_dir=checkpoint_dir)
    dataset = _dataset_rows(first)[0]
    assert dataset["generation_complete"] is True
    assert _surface_rows(first)[0]["issue_codes"]
    assert coordinate_audit._record_bundle_is_complete("binding", first)

    original = coordinate_audit.audit_dataset_row
    calls: list[str] = []

    def _record_scan(row, **kwargs):
        calls.append(str(row["dataset_id"]))
        return original(row, **kwargs)

    monkeypatch.setattr(coordinate_audit, "audit_dataset_row", _record_scan)
    audit_registry(registry, checkpoint_dir=checkpoint_dir)
    assert calls == []

    binding = coordinate_audit._scanner_source_binding()
    changed_binding = dict(binding)
    changed_binding["ruleset_content_sha256"] = "0" * 64
    changed_binding["scanner_binding_sha256"] = coordinate_audit._fingerprint(
        {key: value for key, value in changed_binding.items() if key != "scanner_binding_sha256"}
    )
    monkeypatch.setattr(
        coordinate_audit,
        "_scanner_source_binding",
        lambda: changed_binding,
    )
    audit_registry(registry, checkpoint_dir=checkpoint_dir)
    assert calls == ["binding"]


def test_artifact_manifest_hashes_external_outputs_for_filtered_scans(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.zarr"
    other = tmp_path / "other.zarr"
    _write_node(selected)
    _write_node(other)
    registry = _make_registry(
        tmp_path / "filtered-integrity.sqlite",
        [
            {
                "dataset_id": "selected",
                "recording_id": "rec-selected",
                "zarr_path": str(selected),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
            {
                "dataset_id": "other",
                "recording_id": "rec-other",
                "zarr_path": str(other),
                "zarr_use": "analysis",
                "artifact_kind": "source_recording",
                "status": "active",
            },
        ],
    )
    records = audit_registry(registry, recording_ids=["rec-selected"])
    summary = summarize(records)
    external = {
        "inventory_jsonl": tmp_path / "inventory.jsonl",
        "inventory_csv": tmp_path / "inventory.csv",
        "report_markdown": tmp_path / "report.md",
        "summary_json": tmp_path / "summary.json",
    }
    write_jsonl(external["inventory_jsonl"], records)
    write_csv(external["inventory_csv"], records)
    write_markdown(external["report_markdown"], records, summary)
    write_summary(external["summary_json"], summary)
    artifact_dir = tmp_path / "artifacts"
    coordinate_audit.write_normalized_artifacts(
        artifact_dir,
        registry,
        records,
        external_outputs=external,
    )
    payload = coordinate_audit.verify_normalized_artifact_generation(artifact_dir)
    assert payload["complete"] is False
    assert {
        "external:inventory_jsonl",
        "external:inventory_csv",
        "external:report_markdown",
        "external:summary_json",
    } <= set(payload["files"])
    external["inventory_jsonl"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(size|digest) mismatch"):
        coordinate_audit.verify_normalized_artifact_generation(artifact_dir)
