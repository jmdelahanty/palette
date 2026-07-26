"""Strict audit reader for historical merged collection-proxy geometry.

This module is deliberately quarantined from the normal position-surface
resolver.  It proves the exact schema-v1 publications produced before Palette
separated continuous point coordinates from half-open bounding-box edges.  A
successful read establishes historical integrity only; it never mints a
current ``BoundSourceCameraPositionSurface``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import ArchiveIdentity, archive_identity
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    verify_bound_coordinate_record,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.pixel_frame_authority import (
    BoundPixelFrameAuthority,
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
    require_source_camera_pixel_frame_authority,
    require_trusted_coordinate_attrs,
)
from fisheye.shared.proof_verification import proof_verification_operation
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
)


HISTORICAL_MERGED_PROXY_SCHEMA = (
    "palette_clipped_collection_merged_proxy_crop_run_v1"
)
HISTORICAL_MERGED_PROXY_SOURCE_KIND = (
    "merged_clipped_collection_proxy_crop_run"
)
HISTORICAL_MAPPING_ATTR = "collection_proxy_acquisition_frame_mapping"
HISTORICAL_MAPPING_SCHEMA_ID = (
    "palette.collection_proxy_acquisition_frame_mapping"
)
HISTORICAL_MAPPING_OPERATION = (
    "exact_merged_proxy_rows_to_source_acquisition_frames_v1"
)
HISTORICAL_BBOX_PROJECTION_ATTR = "detection_bbox_projection"
HISTORICAL_BBOX_PROJECTION_SCHEMA_ID = "palette.detection_bbox_projection"
HISTORICAL_BBOX_PROJECTION_OPERATION = (
    "source_camera_normalized_cxcywh_to_image_xyxy_v1"
)
HISTORICAL_CENTER_DERIVATION_ATTR = "bbox_center_derivation"
HISTORICAL_CENTER_DERIVATION_SCHEMA_ID = "palette.bbox_center_derivation"
HISTORICAL_CENTER_DERIVATION_OPERATION = "xyxy_midpoint_v1"
HISTORICAL_COORDINATE_CONTRACT_LABEL = "canonical_v2"

_SEAL = object()


class HistoricalCollectionProxyV1Error(ValueError):
    """Raised when a historical merged-proxy publication is not exact."""


def _fail(message: str) -> None:
    raise HistoricalCollectionProxyV1Error(message)


def _child(root: Any, path: str) -> Any:
    node = root
    try:
        for name in path.strip("/").split("/"):
            node = node[name]
    except Exception as exc:
        _fail(f"Required historical node /{path.strip('/')} is unavailable: {exc}.")
    return node


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        values = np.array(node[:], copy=True, order="C")
        dtype = np.dtype(node.dtype)
        shape = tuple(int(item) for item in node.shape)
    except Exception as exc:
        _fail(f"Unable to read {label}: {exc}.")
    if values.dtype != dtype or values.shape != shape or values.dtype.hasobject:
        _fail(f"{label} values disagree with their declared dtype or shape.")
    return values


def _payload_array(
    rowset: Any,
    payload: Any,
    *,
    name: str,
    label: str,
) -> np.ndarray:
    if not isinstance(payload, Mapping) or set(payload) != {
        "array_ref",
        "dtype",
        "shape",
        "content_sha256",
    }:
        _fail(f"{label} payload pointer is not closed.")
    node = _child(rowset, name)
    expected_ref = f"/{canonical_node_path(rowset)}/{name}"
    if payload.get("array_ref") != expected_ref:
        _fail(f"{label} array reference is not exact.")
    values = _array(node, label=label)
    try:
        payload_dtype = np.dtype(payload.get("dtype"))
        payload_shape = tuple(int(item) for item in payload.get("shape", ()))
    except Exception as exc:
        _fail(f"{label} payload dtype or shape is invalid: {exc}.")
    if values.dtype != payload_dtype or values.shape != payload_shape:
        _fail(f"{label} payload metadata differs from the live array.")
    if payload.get("content_sha256") != array_payload_sha256(node):
        _fail(f"{label} payload digest differs from the live array.")
    return values


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return bool(np.array_equal(left, right, equal_nan=left.dtype.kind in "fc"))


def _value_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if (
        left.shape != right.shape
        or left.dtype.kind not in "iufc"
        or right.dtype.kind not in "iufc"
    ):
        return False
    return bool(np.array_equal(left, right, equal_nan=True))


def _positions_for_unique_ids(
    available_ids: np.ndarray,
    requested_ids: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    if (
        available_ids.ndim != 1
        or requested_ids.ndim != 1
        or available_ids.dtype.kind not in "iu"
        or requested_ids.dtype.kind not in "iu"
    ):
        _fail(f"{label} must use one-dimensional integer identifiers.")
    order = np.argsort(available_ids, kind="stable")
    sorted_ids = available_ids[order]
    if sorted_ids.size > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        _fail(f"{label} contains duplicate identifiers.")
    locations = np.searchsorted(sorted_ids, requested_ids)
    valid = locations < sorted_ids.size
    if np.any(valid):
        valid[valid] &= sorted_ids[locations[valid]] == requested_ids[valid]
    if not np.all(valid):
        _fail(f"{label} does not contain every requested identifier.")
    return order[locations]


def _validate_source_rows(
    root: Any,
    rowset: Any,
    *,
    source_runs: list[str],
    source_refined_paths: list[str],
) -> tuple[int, int, int]:
    merged_keys = _array(rowset["instance_key"], label="merged instance_key")
    merged_frames = _array(rowset["frame_indices"], label="merged frame_indices")
    merged_source_frames = _array(
        rowset["source_frame_indices"],
        label="merged source_frame_indices",
    )
    merged_bbox = _array(rowset["bbox_norm_coords"], label="merged bbox_norm_coords")
    run_indices = _array(
        rowset["source_proxy_crop_run_index"],
        label="merged source_proxy_crop_run_index",
    ).astype(np.int64, copy=False)
    row_indices = _array(
        rowset["source_proxy_crop_row_ids"],
        label="merged source_proxy_crop_row_ids",
    ).astype(np.int64, copy=False)
    row_count = int(merged_keys.shape[0]) if merged_keys.ndim == 1 else -1
    if (
        row_count < 0
        or merged_frames.shape != (row_count,)
        or merged_source_frames.shape != (row_count,)
        or merged_bbox.shape != (row_count, 4)
        or run_indices.shape != (row_count,)
        or row_indices.shape != (row_count,)
        or not np.array_equal(
            merged_frames.astype(np.int64),
            merged_source_frames.astype(np.int64),
        )
    ):
        _fail("Historical merged-proxy arrays are not exactly row-aligned.")

    seen_refined: list[str] = []
    verified_rows = 0
    for source_index, source_name in enumerate(source_runs):
        merged_rows = np.flatnonzero(run_indices == source_index)
        if merged_rows.size == 0:
            _fail(f"Historical source proxy {source_name!r} has no merged rows.")
        source_path = f"crop_runs/{source_name}"
        source = _child(root, source_path)
        source_rows = row_indices[merged_rows]
        source_count = int(source["instance_key"].shape[0])
        if np.any(source_rows < 0) or np.any(source_rows >= source_count):
            _fail(f"Merged source row ids exceed /{source_path}.")
        comparisons = [
            (
                merged_keys[merged_rows],
                _array(source["instance_key"], label=f"{source_path} instance_key")[
                    source_rows
                ],
            ),
            (
                merged_frames[merged_rows],
                _array(source["frame_indices"], label=f"{source_path} frame_indices")[
                    source_rows
                ],
            ),
            (
                merged_source_frames[merged_rows],
                _array(
                    source["source_frame_indices"],
                    label=f"{source_path} source_frame_indices",
                )[source_rows],
            ),
        ]
        if "bbox_norm_coords" in source:
            comparisons.append(
                (
                    merged_bbox[merged_rows],
                    _array(
                        source["bbox_norm_coords"],
                        label=f"{source_path} bbox_norm_coords",
                    )[source_rows],
                )
            )
        if any(not _equal(left, right) for left, right in comparisons):
            _fail(f"Merged values disagree with exact source proxy /{source_path}.")

        refined_paths = source.attrs.get("source_refined_run_paths")
        expected_refined_path = source_refined_paths[source_index]
        if refined_paths is None:
            refined_path = expected_refined_path
        elif isinstance(refined_paths, list) and len(refined_paths) == 1:
            refined_path = str(refined_paths[0])
            if refined_path != expected_refined_path:
                _fail(f"/{source_path} refined lineage differs from the merged rowset.")
        else:
            _fail(f"/{source_path} has ambiguous refined-row lineage.")
        refined_run = _child(root, refined_path)
        refined = _child(refined_run, "instances")
        if refined_path not in seen_refined:
            seen_refined.append(refined_path)
        source_refined_rows = _array(
            source["source_refined_row_ids"],
            label=f"{source_path} source_refined_row_ids",
        ).astype(np.int64, copy=False)[source_rows]
        refined_ids = _array(
            refined["refined_row_ids"],
            label=f"{refined_path}/instances refined_row_ids",
        )
        refined_positions = _positions_for_unique_ids(
            refined_ids,
            source_refined_rows,
            label=f"Refined row identifiers at /{refined_path}",
        )
        source_local_frames = _array(
            source["source_clip_local_frame_indices"],
            label=f"{source_path} source_clip_local_frame_indices",
        )[source_rows]
        refined_comparisons = (
            (
                merged_keys[merged_rows],
                _array(
                    refined["instance_key"],
                    label=f"{refined_path}/instances instance_key",
                )[refined_positions],
            ),
            (
                source_local_frames,
                _array(
                    refined["frame_indices"],
                    label=f"{refined_path}/instances frame_indices",
                )[refined_positions],
            ),
            (
                merged_bbox[merged_rows],
                _array(
                    refined["bbox_norm_coords"],
                    label=f"{refined_path}/instances bbox_norm_coords",
                )[refined_positions],
            ),
        )
        if any(not _value_equal(left, right) for left, right in refined_comparisons):
            _fail(f"Merged values disagree with exact refined source /{refined_path}.")
        verified_rows += int(merged_rows.size)

    if verified_rows != row_count or set(seen_refined) != set(source_refined_paths):
        _fail("Historical merged-proxy source coverage is incomplete.")
    return row_count, len(source_runs), len(seen_refined)


def _require_pointer(
    value: Any,
    bound: Any,
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "record_ref",
        "record_sha256",
    }:
        _fail(f"{label} pointer is not closed.")
    if (
        value.get("record_ref") != bound.record_ref
        or value.get("record_sha256") != bound.record_sha256
    ):
        _fail(f"{label} pointer differs from the exact bound record.")


@dataclass(frozen=True, init=False)
class BoundHistoricalMergedCollectionProxyV1:
    """Verified historical geometry that cannot impersonate a current surface."""

    archive_identity: ArchiveIdentity
    rowset_path: str
    camera_id: str
    row_count: int
    source_proxy_run_count: int
    source_refined_run_count: int
    row_identity: BoundRowIdentityContract = field(repr=False, compare=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(
        repr=False,
        compare=False,
    )
    source_camera_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    acquisition_mapping: BoundCoordinateRecord = field(repr=False, compare=False)
    bbox_projection: BoundCoordinateRecord = field(repr=False, compare=False)
    center_derivation: BoundCoordinateRecord = field(repr=False, compare=False)
    _root: Any = field(repr=False, compare=False)
    _rowset: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _SEAL:
            _fail("Historical merged-proxy bindings cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    def read_array(self, name: str) -> np.ndarray:
        """Read one exact historical array after rechecking its persisted digest."""

        payloads: dict[str, Mapping[str, Any]] = {
            "bbox_norm_coords": self.bbox_projection.record["source_bbox"],
            "bbox_img_xyxy": self.bbox_projection.record["destination_bbox"],
            "centers_img_xy": self.center_derivation.record["output_centers"],
            **{
                array_name: self.acquisition_mapping.record[array_name]
                for array_name in (
                    "frame_indices",
                    "source_frame_indices",
                    "source_acquisition_frame_index",
                    "source_proxy_crop_run_index",
                    "source_proxy_crop_row_ids",
                )
            },
        }
        if name == "instance_key":
            node = self._rowset[name]
            values = _array(node, label=name)
            key = self.row_identity.contract.key_array
            if (
                canonical_node_path(node) != self.row_identity.key_array_path
                or values.dtype != key.dtype
                or values.shape != key.shape
                or array_payload_sha256(node) != key.content_sha256
            ):
                _fail("Historical instance_key changed after identity binding.")
            return values
        if name not in payloads:
            _fail(f"Historical array {name!r} is not an exposed audited surface.")
        return _payload_array(self._rowset, payloads[name], name=name, label=name)

    def assert_verified(self) -> None:
        if archive_identity(self._rowset) != self.archive_identity:
            _fail("Historical merged-proxy rowset changed archives after binding.")
        for record in (
            self.acquisition_mapping,
            self.bbox_projection,
            self.center_derivation,
        ):
            verify_bound_coordinate_record(record)
        require_source_camera_pixel_frame_authority(self.source_camera_frame)


def require_bound_historical_merged_collection_proxy_v1(
    value: Any,
) -> BoundHistoricalMergedCollectionProxyV1:
    if (
        type(value) is not BoundHistoricalMergedCollectionProxyV1
        or value._seal is not _SEAL
    ):
        _fail("A loader-minted historical merged-proxy binding is required.")
    value.assert_verified()
    return value


@proof_verification_operation
def load_historical_merged_collection_proxy_v1(
    root: Any,
    rowset_path: str,
) -> BoundHistoricalMergedCollectionProxyV1:
    """Fully verify one exact historical schema-v1 merged-proxy publication."""

    rowset = _child(root, rowset_path)
    path = canonical_node_path(rowset)
    parts = path.split("/")
    attrs = require_trusted_coordinate_attrs(
        rowset,
        label="Historical merged collection-proxy rowset",
    )
    if (
        len(parts) != 2
        or parts[0] != "crop_runs"
        or not parts[1]
        or attrs.get("schema") != HISTORICAL_MERGED_PROXY_SCHEMA
        or attrs.get("crop_proxy_schema") != HISTORICAL_MERGED_PROXY_SCHEMA
        or attrs.get("source_kind") != HISTORICAL_MERGED_PROXY_SOURCE_KIND
        or attrs.get("coordinate_contract") != HISTORICAL_COORDINATE_CONTRACT_LABEL
        or attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != "auxiliary"
        or attrs.get("stage_selector_eligible") is not False
    ):
        _fail("Rowset is not the exact selector-ineligible historical proxy tuple.")
    source_runs = attrs.get("source_proxy_crop_runs")
    source_refined_paths = attrs.get("source_refined_run_paths")
    collection_id = attrs.get("source_collection_id")
    if (
        not isinstance(source_runs, list)
        or not source_runs
        or any(not isinstance(item, str) or not item for item in source_runs)
        or len(set(source_runs)) != len(source_runs)
        or not isinstance(source_refined_paths, list)
        or len(source_refined_paths) != len(source_runs)
        or any(not isinstance(item, str) or not item for item in source_refined_paths)
        or not isinstance(collection_id, str)
        or not collection_id
    ):
        _fail("Historical merged-proxy source identities are incomplete.")

    _, acquisition = load_persisted_acquisition_camera_authority(root)
    camera_id = acquisition.record.camera_id
    source_camera_node = _child(
        root,
        f"analysis/coordinate_frames/source_camera/{camera_id}/continuous",
    )
    source_camera_frame = load_source_camera_pixel_frame_authority(
        source_camera_node,
        acquisition_frame=acquisition,
    )
    if source_camera_frame.pixel_convention != "continuous":
        _fail("Historical source-camera authority must be the exact continuous frame.")

    row_identity = load_bound_row_identity_contract(rowset, rowset["instance_key"])
    temporal = load_bound_source_row_temporal_authority(
        rowset,
        rowset["source_acquisition_frame_index"],
        source_row_identity=row_identity,
        acquisition_frame=acquisition,
    )
    mapping = bind_persisted_coordinate_record(
        rowset,
        attr_name=HISTORICAL_MAPPING_ATTR,
    )
    projection = bind_persisted_coordinate_record(
        rowset,
        attr_name=HISTORICAL_BBOX_PROJECTION_ATTR,
    )
    center = bind_persisted_coordinate_record(
        rowset,
        attr_name=HISTORICAL_CENTER_DERIVATION_ATTR,
    )

    mapping_record = mapping.record
    expected_mapping_keys = {
        "schema_id",
        "schema_version",
        "operation",
        "direction",
        "frame_indices",
        "source_frame_indices",
        "source_acquisition_frame_index",
        "source_proxy_crop_run_index",
        "source_proxy_crop_row_ids",
        "source_proxy_crop_runs",
        "source_refined_run_paths",
        "source_collection_id",
        "acquisition_camera_frame",
        "source_total_frames",
        "proof",
    }
    if (
        set(mapping_record) != expected_mapping_keys
        or mapping_record.get("schema_id") != HISTORICAL_MAPPING_SCHEMA_ID
        or mapping_record.get("schema_version") != 1
        or mapping_record.get("operation") != HISTORICAL_MAPPING_OPERATION
        or mapping_record.get("direction")
        != "merged_proxy_row_to_source_acquisition_frame"
        or mapping_record.get("proof")
        != "exact_source_proxy_and_refined_row_value_validation_v1"
        or mapping_record.get("source_proxy_crop_runs") != source_runs
        or mapping_record.get("source_refined_run_paths") != source_refined_paths
        or mapping_record.get("source_collection_id") != collection_id
        or mapping_record.get("source_total_frames")
        != acquisition.record.source_total_frames
    ):
        _fail("Historical acquisition mapping is unsupported or not closed.")
    _require_pointer(
        mapping_record.get("acquisition_camera_frame"),
        acquisition,
        label="Historical acquisition frame",
    )
    mapping_arrays = {
        name: _payload_array(rowset, mapping_record[name], name=name, label=name)
        for name in (
            "frame_indices",
            "source_frame_indices",
            "source_acquisition_frame_index",
            "source_proxy_crop_run_index",
            "source_proxy_crop_row_ids",
        )
    }
    frame_values = mapping_arrays["frame_indices"].astype(np.int64)
    if (
        not np.array_equal(
            frame_values,
            mapping_arrays["source_frame_indices"].astype(np.int64),
        )
        or not np.array_equal(
            frame_values,
            mapping_arrays["source_acquisition_frame_index"],
        )
        or np.any(frame_values < 0)
        or np.any(frame_values >= acquisition.record.source_total_frames)
    ):
        _fail("Historical merged-proxy acquisition mapping is not exact.")

    projection_record = projection.record
    expected_projection_keys = {
        "schema_id",
        "schema_version",
        "operation",
        "source_bbox",
        "source_frame",
        "destination_bbox",
        "destination_frame",
        "direction",
        "transform_chain",
        "reference_width_px",
        "reference_height_px",
        "formula",
        "row_identity",
        "temporal_authority",
        "source_lineage",
    }
    if (
        set(projection_record) != expected_projection_keys
        or projection_record.get("schema_id") != HISTORICAL_BBOX_PROJECTION_SCHEMA_ID
        or projection_record.get("schema_version") != 1
        or projection_record.get("operation") != HISTORICAL_BBOX_PROJECTION_OPERATION
        or projection_record.get("direction")
        != "source_camera_normalized_xy_to_source_camera_image_px"
        or projection_record.get("formula")
        != "cxcywh_normalized_to_xyxy_edges_using_exact_reference_extent_v1"
        or projection_record.get("reference_width_px")
        != source_camera_frame.endpoint.width
        or projection_record.get("reference_height_px")
        != source_camera_frame.endpoint.height
    ):
        _fail("Historical bbox projection is unsupported or not closed.")
    _require_pointer(
        projection_record.get("destination_frame"),
        source_camera_frame,
        label="Historical bbox destination frame",
    )
    _require_pointer(
        projection_record.get("row_identity"),
        row_identity,
        label="Historical bbox row identity",
    )
    _require_pointer(
        projection_record.get("temporal_authority"),
        temporal,
        label="Historical bbox temporal authority",
    )
    if projection_record.get("source_lineage") != [
        {"record_ref": mapping.record_ref, "record_sha256": mapping.record_sha256}
    ]:
        _fail("Historical bbox source lineage differs from the exact mapping.")
    bbox_norm = _payload_array(
        rowset,
        projection_record["source_bbox"],
        name="bbox_norm_coords",
        label="historical normalized bbox",
    )
    bbox_img = _payload_array(
        rowset,
        projection_record["destination_bbox"],
        name="bbox_img_xyxy",
        label="historical image bbox",
    )
    dtype = bbox_norm.dtype
    half = np.asarray(0.5, dtype=dtype)
    width_px = np.asarray(source_camera_frame.endpoint.width, dtype=dtype)
    height_px = np.asarray(source_camera_frame.endpoint.height, dtype=dtype)
    expected_bbox = np.column_stack(
        (
            (bbox_norm[:, 0] - bbox_norm[:, 2] * half) * width_px,
            (bbox_norm[:, 1] - bbox_norm[:, 3] * half) * height_px,
            (bbox_norm[:, 0] + bbox_norm[:, 2] * half) * width_px,
            (bbox_norm[:, 1] + bbox_norm[:, 3] * half) * height_px,
        )
    ).astype(dtype, copy=False)
    if not _equal(bbox_img, expected_bbox):
        _fail("Historical image bboxes do not equal their persisted v1 projection.")

    center_record = center.record
    expected_center_keys = {
        "schema_id",
        "schema_version",
        "operation",
        "source_bbox",
        "output_centers",
        "coordinate_frame",
        "formula",
        "row_identity",
    }
    if (
        set(center_record) != expected_center_keys
        or center_record.get("schema_id") != HISTORICAL_CENTER_DERIVATION_SCHEMA_ID
        or center_record.get("schema_version") != 1
        or center_record.get("operation") != HISTORICAL_CENTER_DERIVATION_OPERATION
        or center_record.get("formula")
        != "center_x=(x_min+x_max)/2;center_y=(y_min+y_max)/2"
        or center_record.get("source_bbox")
        != projection_record.get("destination_bbox")
    ):
        _fail("Historical bbox-center derivation is unsupported or not closed.")
    _require_pointer(
        center_record.get("coordinate_frame"),
        source_camera_frame,
        label="Historical center coordinate frame",
    )
    _require_pointer(
        center_record.get("row_identity"),
        row_identity,
        label="Historical center row identity",
    )
    centers = _payload_array(
        rowset,
        center_record["output_centers"],
        name="centers_img_xy",
        label="historical centers",
    )
    expected_centers = np.column_stack(
        (
            (bbox_img[:, 0] + bbox_img[:, 2]) * half,
            (bbox_img[:, 1] + bbox_img[:, 3]) * half,
        )
    ).astype(dtype, copy=False)
    if not _equal(centers, expected_centers):
        _fail("Historical centers do not equal their persisted v1 midpoint derivation.")

    row_count, proxy_count, refined_count = _validate_source_rows(
        root,
        rowset,
        source_runs=source_runs,
        source_refined_paths=source_refined_paths,
    )
    if row_count != int(attrs.get("row_count", -1)):
        _fail("Historical row_count differs from the exact validated rows.")
    return BoundHistoricalMergedCollectionProxyV1(
        archive_identity=archive_identity(rowset),
        rowset_path=path,
        camera_id=camera_id,
        row_count=row_count,
        source_proxy_run_count=proxy_count,
        source_refined_run_count=refined_count,
        row_identity=row_identity,
        temporal_authority=temporal,
        source_camera_frame=source_camera_frame,
        acquisition_mapping=mapping,
        bbox_projection=projection,
        center_derivation=center,
        _root=root,
        _rowset=rowset,
        _verification_seal=_SEAL,
    )


__all__ = [
    "BoundHistoricalMergedCollectionProxyV1",
    "HISTORICAL_BBOX_PROJECTION_ATTR",
    "HISTORICAL_CENTER_DERIVATION_ATTR",
    "HISTORICAL_MAPPING_ATTR",
    "HISTORICAL_MERGED_PROXY_SCHEMA",
    "HISTORICAL_MERGED_PROXY_SOURCE_KIND",
    "HistoricalCollectionProxyV1Error",
    "load_historical_merged_collection_proxy_v1",
    "require_bound_historical_merged_collection_proxy_v1",
]
