"""Materialize immutable registered-dish gate decisions for detections.

The output is a compact keyed decision table.  It does not copy detections and
does not mutate or filter its immutable source.  Production inputs are exact,
active canonical-v3 ``detect_runs`` publications.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    PALETTE_CANDIDATE_KIND,
)
from fisheye.analysis_workflows.materializers.arena_geometry_selection import (
    MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION,
    SELECTION_RECORD_SCHEMA_VERSION,
    SELECTION_RUNS_PARENT,
    validate_arena_geometry_selection_record,
)
from fisheye.shared.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.detection_tables import (
    resolve_detection_instance_table,
    resolve_detection_source_pixel_authority,
)
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.selector_activation import (
    SelectorActivationError,
    activate_selector_eligible_run,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    canonical_detection_dimensions_from_manifest,
    require_active_coordinate_canonical_detection,
    validate_canonical_detection_run_manifest,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

GATE_RUN_SCHEMA_ID = "palette.registered_detection_gate_run"
GATE_RUN_SCHEMA_VERSION = 1
GATE_RUNS_PARENT = "detection_gate_runs"
GATE_ALGORITHM_VERSION = 1
GATE_PUBLISH_SCHEMA_ID = "palette.registered_detection_gate_publish"
GATE_POLICY = "selected_geometry_instance_key_gate_v1"
GATE_POLICY_ATTR = "registered_detection_gate_policy"
GATE_GENERATION_ATTR = "registered_detection_gate_generation"
GATE_LEASE_ATTR = "registered_detection_gate_lease"
DEFAULT_INNER_ROWS = 16_384
DEFAULT_SHARD_ROWS = 131_072
REASON_WIDTH = 40
DECISION_ENUM = {
    "schema_id": "palette.registered_detection_gate_decision_enum",
    "schema_version": 1,
    "values": {"1": "inside_or_on_boundary", "2": "outside_valid_detection_region"},
}


@dataclass(frozen=True)
class RegisteredDetectionGatePlan:
    source_zarr: Path
    source_group_path: str
    source_signature: str
    row_count: int
    frame_count: int
    width_px: int
    height_px: int
    source_pixel_frame_record_ref: str | None
    source_pixel_frame_record_sha256: str | None
    selection_run: str
    selection_record_sha256: str
    selection_record: Mapping[str, Any]
    output_run: str
    target_run_path: Path
    inner_rows: int
    shard_rows: int
    run_provenance: Mapping[str, Any]
    allow_selector_ineligible_source: bool = False


def _canonical_copy(value: Any) -> Any:
    return json.loads(strict_json_dumps(value))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _safe_name(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be one safe group name.")
    return text


def _safe_group_path(value: object) -> str:
    text = str(value or "").strip().strip("/")
    parts = text.split("/") if text else []
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("source_group_path must be a safe relative Zarr group path.")
    return "/".join(parts)


def _group_at(root: Any, path: str) -> Any:
    group = root
    for part in path.split("/"):
        group = group[part]
    return group


def _positive_int(value: object, *, label: str) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer.") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return result


def _effective_shard_rows(shard_rows: int, inner_rows: int) -> int:
    inner = _positive_int(inner_rows, label="inner_rows")
    requested = _positive_int(shard_rows, label="shard_rows")
    return max(inner, int(math.ceil(requested / inner) * inner))


def _source_dimensions(
    attrs: Mapping[str, Any],
    table: Any,
    *,
    row_count: int,
) -> tuple[int, int, int, str | None]:
    """Resolve one canonical-manifest detection extent."""

    width_value = attrs.get("source_video_width") or attrs.get("width")
    height_value = attrs.get("source_video_height") or attrs.get("height")
    frame_count_value = (
        attrs.get("recording_frame_count")
        or attrs.get("num_frames")
        or attrs.get("total_frames")
    )
    manifest_value = attrs.get("run_manifest")
    manifest_digest: str | None = None
    if isinstance(manifest_value, Mapping):
        dimensions = canonical_detection_dimensions_from_manifest(manifest_value)
        if dimensions.n_instances != row_count:
            raise ValueError(
                "Canonical detection manifest instance count differs from its "
                "persisted table."
            )
        width_value = dimensions.source_width
        height_value = dimensions.source_height
        frame_count_value = dimensions.n_frames
        manifest_digest = str(manifest_value["payload_digest"])
    elif manifest_value is not None:
        raise ValueError("Canonical detection run_manifest must be a JSON object.")

    if frame_count_value is None and "frame_counts" in table:
        frame_count_value = int(table["frame_counts"].shape[0])
    if frame_count_value is None and "frame_row_offsets" in table:
        frame_count_value = int(table["frame_row_offsets"].shape[0]) - 1
    return (
        _positive_int(width_value, label="detection source width"),
        _positive_int(height_value, label="detection source height"),
        _positive_int(frame_count_value, label="detection source frame count"),
        manifest_digest,
    )


def _selection_snapshot(root: Any, selection_run: str) -> dict[str, Any]:
    name = _safe_name(selection_run, label="selection_run")
    path = f"analysis/{SELECTION_RUNS_PARENT}/{name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise ValueError(f"Arena-geometry selection is missing: {path}") from exc
    attrs = dict(group.attrs)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not True
        or attrs.get("operational_selection_status") != "selected"
        or attrs.get("detection_gate_applied") is not False
    ):
        raise ValueError(f"Selection {name!r} is not complete and operational.")
    record = attrs.get("selection_record")
    if not isinstance(record, Mapping):
        raise ValueError(f"Selection {name!r} lacks selection_record.")
    validate_arena_geometry_selection_record(record)
    digest = _payload_sha256(record)
    if attrs.get("selection_record_sha256") != digest:
        raise ValueError(f"Selection {name!r} record digest is invalid.")
    selected = record["selected_candidate"]
    coordinate = selected["coordinate_binding"]
    gate = selected["valid_detection_region"]
    geometry = gate["geometry"]
    center = geometry["center_px"]
    width = _positive_int(coordinate.get("native_width_px"), label="selection width")
    height = _positive_int(coordinate.get("native_height_px"), label="selection height")
    circle = (float(center["x"]), float(center["y"]), float(geometry["radius_px"]))
    if not all(math.isfinite(value) for value in circle) or circle[2] <= 0:
        raise ValueError("Selected gate circle is not finite and positive.")
    return {
        "run_name": name,
        "selection_record_sha256": digest,
        "selection_record": _canonical_copy(record),
        "width_px": width,
        "height_px": height,
        "center_x_px": circle[0],
        "center_y_px": circle[1],
        "radius_px": circle[2],
        "pixel_frame_record_ref": coordinate.get("pixel_frame_record_ref"),
        "pixel_frame_record_sha256": coordinate.get("pixel_frame_record_sha256"),
    }


def _source_snapshot(
    root: Any,
    source_group_path: str,
    *,
    allow_selector_ineligible_source: bool = False,
) -> dict[str, Any]:
    path = _safe_group_path(source_group_path)
    try:
        group = _group_at(root, path)
    except KeyError as exc:
        raise ValueError(f"Detection source is missing: {path}") from exc
    if allow_selector_ineligible_source:
        active_manifest = group.attrs.get("run_manifest")
        if not isinstance(active_manifest, Mapping):
            raise ValueError(
                f"Detection source {path!r} lacks its canonical-v3 manifest."
            )
        errors = validate_canonical_detection_run_manifest(active_manifest)
        if errors:
            raise ValueError(
                f"Detection source {path!r} canonical manifest is invalid: "
                + "; ".join(errors)
            )
        payload = active_manifest["payload"]
        if (
            active_manifest.get("schema_version")
            != CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
            or payload.get("run_id") != path.split("/")[-1]
            or payload.get("publication", {}).get("stage_selector_eligible")
            is not False
            or group.attrs.get("stage_selector_eligible") is not False
            or group.attrs.get("production_candidate") is not True
        ):
            raise ValueError(
                f"Detection source {path!r} is not an exact selector-ineligible "
                "canonical-v3 production candidate."
            )
    else:
        try:
            active_manifest = require_active_coordinate_canonical_detection(
                root,
                group_path=path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Detection source {path!r} is not the active canonical-v3 "
                f"authority: {exc}"
            ) from exc
    attrs = dict(group.attrs)
    if attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(
            f"Detection source {path!r} is not complete immutable evidence."
        )
    table = resolve_detection_instance_table(group)
    required = ("bbox_norm_coords", "frame_indices", "instance_key")
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError(f"Detection source {path!r} lacks arrays: {missing}")
    rows = int(table["instance_key"].shape[0])
    if (
        tuple(table["bbox_norm_coords"].shape) != (rows, 4)
        or tuple(table["frame_indices"].shape) != (rows,)
        or tuple(table["instance_key"].shape) != (rows,)
        or np.dtype(table["instance_key"].dtype) != np.dtype(np.uint64)
    ):
        raise ValueError(f"Detection source {path!r} has an invalid row contract.")
    width, height, frame_count, manifest_digest = _source_dimensions(
        attrs,
        table,
        row_count=rows,
    )
    if manifest_digest != active_manifest["payload_digest"]:
        raise ValueError(
            "Detection source canonical manifest changed during gate preflight."
        )

    descriptor = table["bbox_norm_coords"].attrs.get("coordinate_descriptor")
    if isinstance(descriptor, Mapping):
        if (
            descriptor.get("geometry_type") != "bbox_cxcywh"
            or tuple(descriptor.get("component_units") or ()) != ("normalized",) * 4
        ):
            raise ValueError("bbox_norm_coords coordinate descriptor is incompatible.")
    elif manifest_digest is None:
        raise ValueError(
            "Canonical detection source bbox_norm_coords lacks its coordinate descriptor."
        )
    source_pixel_authority = resolve_detection_source_pixel_authority(attrs)
    signature_payload = {
        "group_path": path,
        "schema_id": attrs.get("schema_id"),
        "row_count": rows,
        "frame_count": frame_count,
        "width_px": width,
        "height_px": height,
        "run_provenance": attrs.get("run_provenance"),
        "decoded_array_sha256": attrs.get("decoded_array_sha256"),
        "instance_key_contract": attrs.get("instance_key_contract"),
        "instance_key_frame_domain": attrs.get("instance_key_frame_domain"),
        "source_slices": attrs.get("source_slices"),
        "source_pixel_authority": source_pixel_authority,
    }
    if manifest_digest is not None:
        signature_payload["canonical_run_manifest_payload_digest"] = manifest_digest
    return {
        "group": table,
        "run_group": group,
        "group_path": path,
        "row_count": rows,
        "frame_count": frame_count,
        "width_px": width,
        "height_px": height,
        "source_signature": _payload_sha256(signature_payload),
        "source_signature_payload": _canonical_copy(signature_payload),
        "source_pixel_frame_record_ref": (
            source_pixel_authority["record_ref"]
            if source_pixel_authority is not None
            else None
        ),
        "source_pixel_frame_record_sha256": (
            source_pixel_authority["record_sha256"]
            if source_pixel_authority is not None
            else None
        ),
    }


def build_registered_detection_gate_plan(
    source_zarr: str | Path,
    *,
    source_group_path: str,
    selection_run: str,
    output_run: str | None = None,
    inner_rows: int = DEFAULT_INNER_ROWS,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    allow_selector_ineligible_source: bool = False,
) -> RegisteredDetectionGatePlan:
    if type(allow_selector_ineligible_source) is not bool:
        raise TypeError("allow_selector_ineligible_source must be an exact bool.")
    archive = Path(source_zarr).expanduser().resolve()
    root = open_zarr_root(archive, mode="r")
    source = _source_snapshot(
        root,
        source_group_path,
        allow_selector_ineligible_source=allow_selector_ineligible_source,
    )
    selection = _selection_snapshot(root, selection_run)
    if (
        source["width_px"] != selection["width_px"]
        or source["height_px"] != selection["height_px"]
    ):
        raise ValueError(
            "Detection source and selected geometry native extents disagree."
        )
    if selection["selection_record"]["schema_version"] in {
        SELECTION_RECORD_SCHEMA_VERSION,
        MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION,
    }:
        if (
            source["source_pixel_frame_record_ref"]
            != selection["pixel_frame_record_ref"]
            or source["source_pixel_frame_record_sha256"]
            != selection["pixel_frame_record_sha256"]
        ):
            raise ValueError(
                "Modern boundary-aware geometry and detection source do not share "
                "the exact persisted source-camera pixel authority."
            )
    outer = _effective_shard_rows(shard_rows, inner_rows)
    identity = {
        "algorithm_version": GATE_ALGORITHM_VERSION,
        "source_group_path": source["group_path"],
        "source_signature": source["source_signature"],
        "selection_run": selection["run_name"],
        "selection_record_sha256": selection["selection_record_sha256"],
        "inner_rows": int(inner_rows),
        "shard_rows": outer,
    }
    generated_name = f"registered_detection_gate_{_payload_sha256(identity)[:20]}"
    run_name = _safe_name(output_run or generated_name, label="output_run")
    provenance = build_writer_run_provenance(
        command="fisheye.utils.materialize_registered_detection_gate",
        params={**identity, "output_run": run_name},
        input_run_ids={
            "detection_source": source["group_path"],
            "detection_source_signature": source["source_signature"],
            "arena_geometry_selection": selection["run_name"],
            "selection_record_sha256": selection["selection_record_sha256"],
        },
        cwd=Path.cwd(),
        include_system_context=False,
    )
    return RegisteredDetectionGatePlan(
        source_zarr=archive,
        source_group_path=source["group_path"],
        source_signature=source["source_signature"],
        row_count=source["row_count"],
        frame_count=source["frame_count"],
        width_px=source["width_px"],
        height_px=source["height_px"],
        source_pixel_frame_record_ref=source["source_pixel_frame_record_ref"],
        source_pixel_frame_record_sha256=source["source_pixel_frame_record_sha256"],
        selection_run=selection["run_name"],
        selection_record_sha256=selection["selection_record_sha256"],
        selection_record=selection["selection_record"],
        output_run=run_name,
        target_run_path=archive / "analysis" / GATE_RUNS_PARENT / run_name,
        inner_rows=int(inner_rows),
        shard_rows=outer,
        run_provenance=provenance,
        allow_selector_ineligible_source=allow_selector_ineligible_source,
    )


def _array_digest(array: Any, *, block_rows: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), block_rows):
        stop = min(start + block_rows, int(array.shape[0]))
        values = np.ascontiguousarray(array[start:stop])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _source_array_digests(group: Any, *, block_rows: int) -> dict[str, str]:
    return {
        name: _array_digest(group[name], block_rows=block_rows)
        for name in ("instance_key", "bbox_norm_coords", "frame_indices")
    }


def _encode_reasons(inside: np.ndarray) -> np.ndarray:
    result = np.zeros((len(inside), REASON_WIDTH), dtype=np.uint8)
    encoded = b"outside_valid_detection_region"
    rows = np.flatnonzero(~inside)
    if rows.size:
        result[rows, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def _run_attrs(plan: RegisteredDetectionGatePlan) -> dict[str, Any]:
    selected = plan.selection_record["selected_candidate"]
    selection_decision = plan.selection_record["decision"]
    comparison = selection_decision.get("comparison_binding")
    gate = selected["valid_detection_region"]["geometry"]
    return {
        "schema_id": GATE_RUN_SCHEMA_ID,
        "schema_version": GATE_RUN_SCHEMA_VERSION,
        "algorithm_version": GATE_ALGORITHM_VERSION,
        "artifact_mutability": "immutable_snapshot",
        "source_detection_group_path": plan.source_group_path,
        "source_detection_signature": plan.source_signature,
        "source_row_count": plan.row_count,
        "recording_frame_count": plan.frame_count,
        "source_video_width": plan.width_px,
        "source_video_height": plan.height_px,
        "source_pixel_frame_record_ref": plan.source_pixel_frame_record_ref,
        "source_pixel_frame_record_sha256": plan.source_pixel_frame_record_sha256,
        "selection_run": plan.selection_run,
        "selection_record_sha256": plan.selection_record_sha256,
        "selection_record_schema_version": plan.selection_record["schema_version"],
        "selection_policy": plan.selection_record["selection_policy"],
        "selection_decision_source": selection_decision["decision_source"],
        "comparison_run": (
            comparison.get("run_name") if isinstance(comparison, Mapping) else None
        ),
        "comparison_record_sha256": (
            comparison.get("comparison_record_sha256")
            if isinstance(comparison, Mapping)
            else None
        ),
        "comparison_policy_id": (
            comparison.get("policy_id") if isinstance(comparison, Mapping) else None
        ),
        "selected_candidate_run": selected["run_name"],
        "selected_candidate_record_sha256": selected["candidate_record_sha256"],
        "arena_binding": _canonical_copy(selected["arena_binding"]),
        "coordinate_binding": _canonical_copy(selected["coordinate_binding"]),
        "gate_geometry": _canonical_copy(gate),
        "gate_method": "native_camera_circle_signed_distance_inclusive_v1",
        "additional_palette_tolerance_px": 0.0,
        "row_identity": "instance_key",
        "raw_detections_preserved": True,
        "decision_enum": _canonical_copy(DECISION_ENUM),
        "rejection_reason_encoding": "null_padded_utf8_uint8_rows_v1",
        "row_chunk_rows": plan.inner_rows,
        "row_shard_rows": plan.shard_rows,
        "run_provenance": _canonical_copy(plan.run_provenance),
    }


def _materialize_local_gate(plan: RegisteredDetectionGatePlan, path: Path) -> None:
    root = open_zarr_root(plan.source_zarr, mode="r")
    source = _source_snapshot(
        root,
        plan.source_group_path,
        allow_selector_ineligible_source=plan.allow_selector_ineligible_source,
    )
    selection = _selection_snapshot(root, plan.selection_run)
    if (
        source["source_signature"] != plan.source_signature
        or selection["selection_record_sha256"] != plan.selection_record_sha256
        or selection["selection_record"] != plan.selection_record
    ):
        raise RuntimeError("Gate sources changed before local materialization.")
    if path.exists():
        raise FileExistsError(path)
    output = zarr.open_group(str(path), mode="w", zarr_format=3)
    output.attrs.update(json_attr_safe(_run_attrs(plan)))
    mark_run_started(
        output, run_name=plan.output_run, stage="registered_detection_gate"
    )
    specs = {
        "instance_key": (np.uint64, ()),
        "source_row_index": (np.int64, ()),
        "frame_indices": (np.int64, ()),
        "detection_centroid_native_px": (np.float64, (2,)),
        "signed_distance_to_gate_px": (np.float64, ()),
        "inside_registered_dish_mask": (np.bool_, ()),
        "gate_decision": (np.uint8, ()),
        "reason_bytes": (np.uint8, (REASON_WIDTH,)),
    }
    arrays: dict[str, Any] = {}
    for name, (dtype, trailing) in specs.items():
        arrays[name] = output.create_array(
            name,
            shape=(plan.row_count, *trailing),
            dtype=dtype,
            chunks=(plan.inner_rows, *trailing),
            shards=(plan.shard_rows, *trailing),
        )
    selected = plan.selection_record["selected_candidate"]
    geometry = selected["valid_detection_region"]["geometry"]
    center = geometry["center_px"]
    cx, cy, radius = (
        float(center["x"]),
        float(center["y"]),
        float(geometry["radius_px"]),
    )
    input_group = source["group"]
    key_digest = hashlib.sha256()
    bbox_digest = hashlib.sha256()
    frame_digest = hashlib.sha256()
    accepted = 0
    rejected = 0
    for start in range(0, plan.row_count, plan.shard_rows):
        stop = min(start + plan.shard_rows, plan.row_count)
        raw_bbox = np.asarray(input_group["bbox_norm_coords"][start:stop])
        raw_keys = np.asarray(input_group["instance_key"][start:stop])
        raw_frames = np.asarray(input_group["frame_indices"][start:stop])
        bbox = raw_bbox.astype(np.float64, copy=False)
        keys = raw_keys.astype(np.uint64, copy=False)
        frames = raw_frames.astype(np.int64, copy=False)
        if not np.all(np.isfinite(bbox)):
            raise RuntimeError("Detection source contains non-finite normalized boxes.")
        if np.any(frames < 0) or np.any(frames >= plan.frame_count):
            raise RuntimeError("Detection source frames escape the recording domain.")
        centers = np.column_stack(
            (bbox[:, 0] * plan.width_px, bbox[:, 1] * plan.height_px)
        )
        distance = radius - np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
        inside = distance >= 0.0
        decisions = np.where(inside, 1, 2).astype(np.uint8)
        arrays["instance_key"][start:stop] = keys
        arrays["source_row_index"][start:stop] = np.arange(start, stop, dtype=np.int64)
        arrays["frame_indices"][start:stop] = frames
        arrays["detection_centroid_native_px"][start:stop] = centers
        arrays["signed_distance_to_gate_px"][start:stop] = distance
        arrays["inside_registered_dish_mask"][start:stop] = inside
        arrays["gate_decision"][start:stop] = decisions
        arrays["reason_bytes"][start:stop] = _encode_reasons(inside)
        key_digest.update(np.ascontiguousarray(raw_keys).view(np.uint8))
        bbox_digest.update(np.ascontiguousarray(raw_bbox).view(np.uint8))
        frame_digest.update(np.ascontiguousarray(raw_frames).view(np.uint8))
        accepted += int(np.count_nonzero(inside))
        rejected += int(len(inside) - np.count_nonzero(inside))
    output.attrs["source_decoded_sha256"] = {
        "instance_key": key_digest.hexdigest(),
        "bbox_norm_coords": bbox_digest.hexdigest(),
        "frame_indices": frame_digest.hexdigest(),
    }
    output.attrs["decision_summary"] = {
        "row_count": plan.row_count,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "complete": accepted + rejected == plan.row_count,
    }
    output.attrs["decoded_array_sha256"] = {
        name: _array_digest(array, block_rows=plan.shard_rows)
        for name, array in arrays.items()
    }


def validate_registered_detection_gate_run(
    run_path: str | Path,
    *,
    expected_plan: RegisteredDetectionGatePlan,
    require_complete: bool = False,
    require_eligible: bool | None = None,
    verify_payload: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    path = Path(run_path).expanduser().resolve()
    try:
        group = open_zarr_root(path, mode="r")
        attrs = dict(group.attrs)
        for name, value in _run_attrs(expected_plan).items():
            if name != "run_provenance" and attrs.get(name) != value:
                errors.append(f"{name} mismatch")
        required_shapes = {
            "instance_key": (expected_plan.row_count,),
            "source_row_index": (expected_plan.row_count,),
            "frame_indices": (expected_plan.row_count,),
            "detection_centroid_native_px": (expected_plan.row_count, 2),
            "signed_distance_to_gate_px": (expected_plan.row_count,),
            "inside_registered_dish_mask": (expected_plan.row_count,),
            "gate_decision": (expected_plan.row_count,),
            "reason_bytes": (expected_plan.row_count, REASON_WIDTH),
        }
        for name, shape in required_shapes.items():
            if name not in group or tuple(group[name].shape) != shape:
                errors.append(f"{name} shape mismatch")
        provenance = validate_run_provenance(attrs.get("run_provenance"))
        if not provenance.valid:
            errors.extend(f"run provenance: {value}" for value in provenance.errors)
        summary = attrs.get("decision_summary")
        if not isinstance(summary, Mapping) or (
            int(summary.get("accepted_count") or 0)
            + int(summary.get("rejected_count") or 0)
            != expected_plan.row_count
        ):
            errors.append("decision summary mismatch")
        if verify_payload and not errors:
            digests = attrs.get("decoded_array_sha256")
            if not isinstance(digests, Mapping):
                errors.append("decoded array digests missing")
            else:
                for name in required_shapes:
                    if _array_digest(
                        group[name], block_rows=expected_plan.shard_rows
                    ) != digests.get(name):
                        errors.append(f"{name} decoded digest mismatch")
            keys = np.asarray(group["instance_key"][:], dtype=np.uint64)
            rows = np.asarray(group["source_row_index"][:], dtype=np.int64)
            inside = np.asarray(group["inside_registered_dish_mask"][:], dtype=bool)
            decisions = np.asarray(group["gate_decision"][:], dtype=np.uint8)
            if len(np.unique(keys)) != expected_plan.row_count:
                errors.append("instance_key values are not unique")
            if not np.array_equal(
                rows, np.arange(expected_plan.row_count, dtype=np.int64)
            ):
                errors.append("source_row_index is not dense identity")
            if not np.array_equal(decisions, np.where(inside, 1, 2).astype(np.uint8)):
                errors.append("gate decisions disagree with inside flags")
            reasons = np.asarray(group["reason_bytes"][:], dtype=np.uint8)
            if np.any(reasons[inside]) or np.any(~np.any(reasons[~inside], axis=1)):
                errors.append("gate rejection reasons disagree with inside flags")
            source_root = open_zarr_root(expected_plan.source_zarr, mode="r")
            source = _source_snapshot(
                source_root,
                expected_plan.source_group_path,
                allow_selector_ineligible_source=(
                    expected_plan.allow_selector_ineligible_source
                ),
            )
            selected = _selection_snapshot(source_root, expected_plan.selection_run)
            if source["source_signature"] != expected_plan.source_signature:
                errors.append("source detection signature changed")
            if (
                selected["selection_record_sha256"]
                != expected_plan.selection_record_sha256
                or selected["selection_record"] != expected_plan.selection_record
            ):
                errors.append("arena geometry selection changed")
            if attrs.get("source_decoded_sha256") != _source_array_digests(
                source["group"], block_rows=expected_plan.shard_rows
            ):
                errors.append("source detection payload changed")
        status = attrs.get("palette_run_completion_status")
        if require_complete and status != "complete":
            errors.append("gate run is not complete")
        elif status not in {"running", "complete"}:
            errors.append("gate run has invalid completion status")
        if (
            require_eligible is not None
            and attrs.get("stage_selector_eligible") is not require_eligible
        ):
            errors.append("gate selector eligibility mismatch")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": errors, "run_path": str(path)}


def _revalidate_sources(plan: RegisteredDetectionGatePlan) -> dict[str, Any]:
    root = open_zarr_root(plan.source_zarr, mode="r")
    source = _source_snapshot(
        root,
        plan.source_group_path,
        allow_selector_ineligible_source=plan.allow_selector_ineligible_source,
    )
    selection = _selection_snapshot(root, plan.selection_run)
    if source["source_signature"] != plan.source_signature:
        raise RuntimeError("Detection source changed during gate publication.")
    if (
        selection["selection_record_sha256"] != plan.selection_record_sha256
        or selection["selection_record"] != plan.selection_record
    ):
        raise RuntimeError("Geometry selection changed during gate publication.")
    return {
        "status": "current",
        "source_detection_signature": plan.source_signature,
        "selection_record_sha256": plan.selection_record_sha256,
    }


def validate_registered_detection_gate_consumption(
    source_zarr: str | Path,
    *,
    source_group_path: str,
    gate_run: str,
    expected_instance_keys: np.ndarray,
    require_comparison_bound_selection: bool = False,
    require_modern_operational_selection: bool = False,
    allow_selector_ineligible_source: bool = False,
) -> dict[str, Any]:
    """Validate and load one exact gate for fail-closed refinement consumption."""

    if type(require_comparison_bound_selection) is not bool:
        raise TypeError("require_comparison_bound_selection must be an exact bool.")
    if type(require_modern_operational_selection) is not bool:
        raise TypeError("require_modern_operational_selection must be an exact bool.")

    archive = Path(source_zarr).expanduser().resolve()
    run_name = _safe_name(gate_run, label="gate_run")
    root = open_zarr_root(archive, mode="r")
    path = f"analysis/{GATE_RUNS_PARENT}/{run_name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise ValueError(f"Registered detection gate is missing: {path}") from exc
    attrs = dict(group.attrs)
    if attrs.get("source_detection_group_path") != _safe_group_path(source_group_path):
        raise ValueError("Registered gate binds a different detection source path.")
    selection_run = _safe_name(attrs.get("selection_run"), label="selection_run")
    plan = build_registered_detection_gate_plan(
        archive,
        source_group_path=source_group_path,
        selection_run=selection_run,
        output_run=run_name,
        inner_rows=int(attrs.get("row_chunk_rows") or 0),
        shard_rows=int(attrs.get("row_shard_rows") or 0),
        allow_selector_ineligible_source=allow_selector_ineligible_source,
    )
    comparison = plan.selection_record["decision"].get("comparison_binding")
    if require_comparison_bound_selection and (
        plan.selection_record.get("schema_version") != SELECTION_RECORD_SCHEMA_VERSION
        or not isinstance(comparison, Mapping)
    ):
        raise ValueError(
            "Configured registered geometry requires a comparison-bound version-2 "
            "selection; legacy selection evidence is not operationally sufficient."
        )
    decision = plan.selection_record["decision"]
    selected_candidate = plan.selection_record["selected_candidate"]
    comparison_bound_v2 = (
        plan.selection_record.get("schema_version") == SELECTION_RECORD_SCHEMA_VERSION
        and isinstance(comparison, Mapping)
    )
    reviewed_palette_v3 = (
        plan.selection_record.get("schema_version")
        == MANUAL_PALETTE_SELECTION_RECORD_SCHEMA_VERSION
        and decision.get("decision_source") == "manual_review"
        and comparison is None
        and selected_candidate.get("candidate_kind") == PALETTE_CANDIDATE_KIND
    )
    if require_modern_operational_selection and not (
        comparison_bound_v2 or reviewed_palette_v3
    ):
        raise ValueError(
            "Configured registered geometry requires a modern operational selection: "
            "either a comparison-bound "
            "version-2 selection or an explicitly reviewed Palette version-3 "
            "selection; legacy selection evidence is not operationally sufficient."
        )
    report = validate_registered_detection_gate_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
        require_eligible=True,
        verify_payload=True,
    )
    if not report["valid"]:
        raise ValueError(f"Registered detection gate is invalid: {report['errors']}")
    expected = np.asarray(expected_instance_keys, dtype=np.uint64).reshape(-1)
    observed = np.asarray(group["instance_key"][:], dtype=np.uint64).reshape(-1)
    rows = np.asarray(group["source_row_index"][:], dtype=np.int64).reshape(-1)
    frames = np.asarray(group["frame_indices"][:], dtype=np.int64).reshape(-1)
    inside = np.asarray(group["inside_registered_dish_mask"][:], dtype=bool).reshape(-1)
    if len(np.unique(expected)) != len(expected):
        raise ValueError("Source detection instance_key values are not unique.")
    if not np.array_equal(observed, expected):
        raise ValueError(
            "Registered gate and source detections do not have identical ordered "
            "instance_key coverage."
        )
    if not np.array_equal(rows, np.arange(len(expected), dtype=np.int64)):
        raise ValueError("Registered gate source-row identity is missing or reordered.")
    if frames.shape != expected.shape or inside.shape != expected.shape:
        raise ValueError("Registered gate row coverage is incomplete.")
    return {
        "inside": inside,
        "gate_run": run_name,
        "gate_group_path": path,
        "gate_decoded_array_sha256": _canonical_copy(attrs["decoded_array_sha256"]),
        "source_detection_group_path": plan.source_group_path,
        "source_detection_signature": plan.source_signature,
        "selection_run": plan.selection_run,
        "selection_record_sha256": plan.selection_record_sha256,
        "selection_record_schema_version": plan.selection_record["schema_version"],
        "selection_policy": plan.selection_record["selection_policy"],
        "selection_decision_source": plan.selection_record["decision"][
            "decision_source"
        ],
        "comparison_run": (
            comparison.get("run_name") if isinstance(comparison, Mapping) else None
        ),
        "comparison_record_sha256": (
            comparison.get("comparison_record_sha256")
            if isinstance(comparison, Mapping)
            else None
        ),
        "comparison_policy_id": (
            comparison.get("policy_id") if isinstance(comparison, Mapping) else None
        ),
        "selected_candidate_run": attrs.get("selected_candidate_run"),
        "selected_candidate_record_sha256": attrs.get(
            "selected_candidate_record_sha256"
        ),
        "row_count": len(expected),
        "accepted_count": int(np.count_nonzero(inside)),
        "rejected_count": int(len(inside) - np.count_nonzero(inside)),
        "ordered_instance_key_coverage_exact": True,
        "source_row_identity_exact": True,
    }


def publish_registered_detection_gate(
    plan: RegisteredDetectionGatePlan,
    *,
    scratch_root: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    if plan.target_run_path.exists():
        existing = validate_registered_detection_gate_run(
            plan.target_run_path,
            expected_plan=plan,
            require_complete=True,
            require_eligible=True,
        )
        if not existing["valid"]:
            raise FileExistsError(f"Existing gate is not the expected run: {existing}")
        return {"published": False, "status": "already_complete", **existing}
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"palette-{plan.output_run}-", dir=scratch
    ) as temporary:
        local_run = Path(temporary) / plan.output_run
        _materialize_local_gate(plan, local_run)

        def validate(path: Path) -> dict[str, Any]:
            return validate_registered_detection_gate_run(path, expected_plan=plan)

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (require_runs_parent(analysis, GATE_RUNS_PARENT),)

        def complete(_root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
            mark_run_complete(
                run,
                parent_group=parent,
                run_name=plan.output_run,
                run_provenance=plan.run_provenance,
            )

        def verify(root: zarr.Group) -> None:
            parent = root[f"analysis/{GATE_RUNS_PARENT}"]
            run = parent[plan.output_run]
            if (
                run.attrs.get("palette_run_completion_status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
                or parent.attrs.get("latest") == plan.output_run
                or parent.attrs.get("latest_complete") == plan.output_run
            ):
                raise RuntimeError("Gate became visible before activation.")

        def activate(root: zarr.Group, parent: zarr.Group, run: zarr.Group) -> None:
            def proof() -> tuple[Any, ...]:
                current = root[f"analysis/{GATE_RUNS_PARENT}/{plan.output_run}"]
                check = validate_registered_detection_gate_run(
                    plan.target_run_path,
                    expected_plan=plan,
                    require_complete=True,
                    require_eligible=False,
                )
                if not check["valid"]:
                    raise RuntimeError(f"Gate activation proof failed: {check}")
                return (
                    current.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR),
                    current.attrs.get("palette_run_completion_status"),
                    current.attrs.get("decoded_array_sha256"),
                    current.attrs.get("selection_record_sha256"),
                    current.attrs.get("source_detection_signature"),
                )

            try:
                activate_selector_eligible_run(
                    root,
                    parent,
                    run,
                    parent_path=f"analysis/{GATE_RUNS_PARENT}",
                    run_path=f"analysis/{GATE_RUNS_PARENT}/{plan.output_run}",
                    run_name=plan.output_run,
                    owner_attr=ATOMIC_PUBLICATION_OWNER_ATTR,
                    expected_owner_uuid=str(run.attrs[ATOMIC_PUBLICATION_OWNER_ATTR]),
                    policy_attr=GATE_POLICY_ATTR,
                    generation_attr=GATE_GENERATION_ATTR,
                    lease_attr=GATE_LEASE_ATTR,
                    policy=GATE_POLICY,
                    lease_schema_id="palette.registered_detection_gate_lease",
                    proof_loader=proof,
                )
            except SelectorActivationError as exc:
                raise RuntimeError(
                    f"Registered detection gate activation failed: {exc}"
                ) from exc

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=local_run,
                target_run_path=plan.target_run_path,
                run_name=plan.output_run,
                lock_suffix="registered-detection-gate-publish",
                publish_schema_id=GATE_PUBLISH_SCHEMA_ID,
                policy="node_local_gate_then_atomic_prfs_publication_v1",
                rollback_policy="retain_failed_public_tombstone_restore_owned_selector_epoch",
                content_checksum=True,
                selector_owner_attr=GATE_LEASE_ATTR,
                selector_generation_attr=GATE_GENERATION_ATTR,
                owned_parent_attr_names=(
                    (
                        "latest_complete",
                        "latest",
                        GATE_POLICY_ATTR,
                        GATE_GENERATION_ATTR,
                        GATE_LEASE_ATTR,
                    ),
                ),
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            after_rename=lambda _root, _run: {
                "source_revision_audit": _revalidate_sources(plan)
            },
            activate_run=activate,
            payload_metadata={
                "algorithm_version": GATE_ALGORITHM_VERSION,
                "source_detection_group_path": plan.source_group_path,
                "source_detection_signature": plan.source_signature,
                "selection_run": plan.selection_run,
                "selection_record_sha256": plan.selection_record_sha256,
            },
        )
    final = validate_registered_detection_gate_run(
        plan.target_run_path,
        expected_plan=plan,
        require_complete=True,
        require_eligible=True,
    )
    if not final["valid"]:
        raise RuntimeError(f"Final registered gate validation failed: {final}")
    return {"published": True, "status": "complete", **publication, **final}


__all__ = [
    "RegisteredDetectionGatePlan",
    "build_registered_detection_gate_plan",
    "publish_registered_detection_gate",
    "validate_registered_detection_gate_consumption",
    "validate_registered_detection_gate_run",
]
