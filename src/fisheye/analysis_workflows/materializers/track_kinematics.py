"""Stage track kinematics locally and bind coordinates only after publication.

The authoritative recording remains read-only during numerical computation.
The writer emits an explicitly *unbound* node-local stage, which is copied into
Zarr-v3 indexed shards and changed to a fail-closed publishing state.  Under the
per-recording publication lock, the sharded run is renamed to its final path;
only then may the track writer bind row identity, coordinate descriptors, and
derivations against nodes in the authoritative archive.  Completion and latest
pointers follow two fresh validations of that final-path binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from ...analysis import track_kinematics as track_writer
from ...shared.json_safety import json_attr_safe
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import require_runs_parent
from ...shared.zarr_payload_receipt import (
    build_payload_integrity_receipt,
    verify_payload_integrity_receipt,
)
from ...shared.zarr_sharded_copy import (
    STRUCTURED_DTYPE_SINGLE_CHUNK_LAYOUT,
    copy_completed_run_to_sharded,
)
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group
from .runtime_telemetry import PhaseTelemetry


MATERIALIZATION_SCHEMA_ID = "palette.track_kinematics_materialization.v3"
PUBLISH_SCHEMA_ID = "palette.track_kinematics_run_publish.v3"
COORDINATE_BINDING_STATUS_ATTR = (
    track_writer.TRACK_KINEMATICS_COORDINATE_BINDING_STATUS_ATTR
)
UNBOUND_STAGE_STATUS = track_writer.TRACK_KINEMATICS_UNBOUND_STAGE_STATUS
PUBLISHING_BINDING_STATUS = (
    track_writer.TRACK_KINEMATICS_PUBLISHING_BINDING_STATUS
)
BOUND_CANONICAL_STATUS = track_writer.TRACK_KINEMATICS_BOUND_CANONICAL_STATUS
STAGING_MANIFEST_ATTR = track_writer.TRACK_KINEMATICS_STAGING_MANIFEST_ATTR
STAGING_MANIFEST_DIGEST_ATTR = (
    track_writer.TRACK_KINEMATICS_STAGING_MANIFEST_DIGEST_ATTR
)
DEFAULT_OUTPUT_SHARD_ROWS = 262_144
MANAGED_WRITER_ARGUMENTS = {
    "--keypoint-run",
    "--no-write",
    "--offline-only",
    "--offline-run-name",
    "--online-only",
    "--output-zarr-path",
}


@dataclass(frozen=True)
class TrackKinematicsMaterializationPlan:
    source_zarr: Path
    scratch_root: Path
    staging_zarr: Path
    sharded_run: Path
    keypoint_run: str
    run_name: str
    output_shard_rows: int
    shard_workers: int
    writer_arguments: tuple[str, ...]

    @property
    def local_run_path(self) -> Path:
        return (
            self.staging_zarr
            / "analysis"
            / "track_kinematics_runs"
            / "offline"
            / self.run_name
        )

    @property
    def target_run_path(self) -> Path:
        return (
            self.source_zarr
            / "analysis"
            / "track_kinematics_runs"
            / "offline"
            / self.run_name
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "staging_zarr": str(self.staging_zarr),
            "local_run_path": str(self.local_run_path),
            "sharded_run": str(self.sharded_run),
            "target_run_path": str(self.target_run_path),
            "keypoint_run": self.keypoint_run,
            "run_name": self.run_name,
            "output_shard_rows": int(self.output_shard_rows),
            "shard_workers": int(self.shard_workers),
            "writer_arguments": list(self.writer_arguments),
        }


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe track-kinematics run name: {run_name!r}.")
    return value


def build_track_kinematics_materialization_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    keypoint_run: str,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    shard_workers: int = 1,
    writer_arguments: Sequence[str] = (),
) -> TrackKinematicsMaterializationPlan:
    """Build a read-only plan; no scratch or archive paths are created."""

    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the authoritative source Zarr.")
    name = _validate_run_name(run_name)
    keypoints = str(keypoint_run).strip()
    if not keypoints:
        raise ValueError("keypoint_run is required.")
    if int(output_shard_rows) <= 0 or int(shard_workers) <= 0:
        raise ValueError("output_shard_rows and shard_workers must be positive.")
    forwarded = tuple(str(value) for value in writer_arguments)
    forbidden = sorted(
        argument.split("=", 1)[0]
        for argument in forwarded
        if argument.split("=", 1)[0] in MANAGED_WRITER_ARGUMENTS
    )
    if forbidden:
        raise ValueError(
            "Track materializer owns these writer arguments: "
            + ", ".join(forbidden)
        )
    target = source / "analysis" / "track_kinematics_runs" / "offline" / name
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing authoritative run: {target}")
    return TrackKinematicsMaterializationPlan(
        source_zarr=source,
        scratch_root=scratch,
        staging_zarr=scratch / "track-staging.zarr",
        sharded_run=scratch / "track-run-sharded",
        keypoint_run=keypoints,
        run_name=name,
        output_shard_rows=int(output_shard_rows),
        shard_workers=int(shard_workers),
        writer_arguments=forwarded,
    )


def _iter_arrays(group: zarr.Group, prefix: str = ""):
    for name, array in sorted(group.arrays(), key=lambda item: str(item[0])):
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(child, child_prefix)


def _iter_nodes(group: zarr.Group, prefix: str = ""):
    yield prefix, group
    for name, array in sorted(group.arrays(), key=lambda item: str(item[0])):
        yield f"{prefix}/{name}" if prefix else str(name), array
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_nodes(child, child_prefix)


_CANONICAL_PAYLOAD_LEAVES = frozenset(
    {
        "track_ids",
        "track_arena_ids",
        "frame_indices",
        "track_sample_key",
        "source_acquisition_frame_index",
        "source_frame_interpolation",
        "source_instance_key",
        "source_row_index",
        "positions_px",
        "positions_mm",
    }
)


def _contains_detached_coordinate_descriptor(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("schema_id") == "palette.coordinate_descriptor":
            return True
        return any(
            _contains_detached_coordinate_descriptor(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_detached_coordinate_descriptor(item) for item in value)
    return False


def _canonical_binding_attr(name: str, value: Any) -> bool:
    key = str(name)
    return (
        key in {
            "coordinate_descriptor",
            "coordinate_descriptor_sha256",
            "coordinate_descriptors",
            "track_position_derivation",
            "track_position_derivation_sha256",
            "track_sample_time_lineage",
        }
        or key.endswith("_coordinate_descriptor")
        or key.endswith("_coordinate_descriptor_sha256")
        or key.startswith("row_identity_")
        or key.startswith("track_sample_time_lineage_")
        or _contains_detached_coordinate_descriptor(value)
    )


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decoded_array_record(array: zarr.Array) -> dict[str, Any]:
    dtype = np.dtype(array.dtype)
    if dtype.hasobject:
        raise TypeError("object dtype has no stable decoded-byte contract")
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    decoded_bytes = 0
    if int(array.ndim) == 0:
        blocks = (np.asarray(array[...]),)
    else:
        chunks = getattr(array, "chunks", None)
        block_rows = (
            max(1, int(chunks[0]))
            if chunks is not None and len(chunks) >= 1
            else max(1, shape[0])
        )
        trailing = (slice(None),) * (int(array.ndim) - 1)
        blocks = (
            np.asarray(array[(slice(start, min(start + block_rows, shape[0])), *trailing)])
            for start in range(0, shape[0], block_rows)
        )
    for values in blocks:
        if values.dtype != dtype:
            raise TypeError("decoded dtype differs from declared array dtype")
        contiguous = np.ascontiguousarray(values)
        payload = contiguous.tobytes(order="C")
        digest.update(payload)
        decoded_bytes += len(payload)
    return {
        "dtype": np.lib.format.dtype_to_descr(dtype),
        "shape": list(shape),
        "decoded_bytes": int(decoded_bytes),
        "decoded_sha256": digest.hexdigest(),
    }


def _decoded_contract_payloads(group: zarr.Group) -> dict[str, Any]:
    arrays: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path, array in _iter_arrays(group):
        if path.rsplit("/", 1)[-1] not in _CANONICAL_PAYLOAD_LEAVES:
            continue
        try:
            arrays[path] = _decoded_array_record(array)
        except Exception as exc:
            errors.append(f"{path}: decoded payload validation failed: {exc}")
    digest = hashlib.sha256()
    for path, record in sorted(arrays.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "valid": not errors,
        "errors": errors,
        "scope": "track_coordinate_identity_and_position_decoded_payloads_v1",
        "array_count": len(arrays),
        "aggregate_sha256": digest.hexdigest(),
        "arrays": arrays,
    }


def _validate_track_run(
    path: Path,
    *,
    require_sharded: bool,
    expected_binding_status: str,
    require_complete: bool,
    expected_selector_eligible: bool,
    verify_decoded_payloads: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    group = open_zarr_root(path, mode="r")
    expected_completion = "complete" if require_complete else "running"
    observed_completion = str(group.attrs.get("palette_run_completion_status"))
    if observed_completion != expected_completion:
        errors.append(
            "generic completion status mismatch: "
            f"expected={expected_completion!r}, observed={observed_completion!r}"
        )
    if (
        group.attrs.get("stage_selector_eligible")
        is not expected_selector_eligible
    ):
        errors.append(
            "stage selector eligibility mismatch: "
            f"expected={expected_selector_eligible!r}, "
            f"observed={group.attrs.get('stage_selector_eligible')!r}"
        )
    try:
        track_writer._track_publication_owner_uuid(group)
    except RuntimeError as exc:
        errors.append(str(exc))
    observed_binding = str(group.attrs.get(COORDINATE_BINDING_STATUS_ATTR))
    if observed_binding != expected_binding_status:
        errors.append(
            "coordinate binding status mismatch: "
            f"expected={expected_binding_status!r}, observed={observed_binding!r}"
        )
    if str(group.attrs.get("schema_id")) != "analysis.track_kinematics_runs":
        errors.append("missing or invalid track-kinematics schema_id")
    if int(group.attrs.get("schema_version", -1)) != 1:
        errors.append("missing or invalid track-kinematics schema_version")
    if str(group.attrs.get("method_version")) != "track_kinematics.v1":
        errors.append("missing or invalid track-kinematics method_version")
    if str(group.attrs.get("row_axis")) != "track_samples":
        errors.append("missing or invalid track-kinematics row_axis")
    if not isinstance(group.attrs.get("source_refs"), dict):
        errors.append("missing track-kinematics source_refs")
    if not isinstance(group.attrs.get("parameters"), dict):
        errors.append("missing track-kinematics parameters")
    staging_manifest = group.attrs.get(STAGING_MANIFEST_ATTR)
    if not isinstance(staging_manifest, dict):
        errors.append(f"missing typed {STAGING_MANIFEST_ATTR}")
    manifest_digest = group.attrs.get(STAGING_MANIFEST_DIGEST_ATTR)
    if (
        type(manifest_digest) is not str
        or len(manifest_digest) != 64
        or any(character not in "0123456789abcdef" for character in manifest_digest)
    ):
        errors.append(f"missing or invalid {STAGING_MANIFEST_DIGEST_ATTR}")
    elif isinstance(staging_manifest, dict):
        try:
            expected_manifest_digest = _canonical_mapping_sha256(staging_manifest)
        except (TypeError, ValueError) as exc:
            errors.append(f"{STAGING_MANIFEST_ATTR} is not strict canonical JSON: {exc}")
        else:
            if manifest_digest != expected_manifest_digest:
                errors.append(f"{STAGING_MANIFEST_DIGEST_ATTR} does not match manifest")

    if expected_binding_status in {UNBOUND_STAGE_STATUS, PUBLISHING_BINDING_STATUS}:
        for node_path, node in _iter_nodes(group):
            for attr_name, value in node.attrs.items():
                if _canonical_binding_attr(str(attr_name), value):
                    errors.append(
                        f"{node_path or '/'}: unbound stage contains canonical "
                        f"binding attr {attr_name!r}"
                    )
    tracks = group.get("tracks")
    if not isinstance(tracks, zarr.Group):
        errors.append("missing tracks group")
        track_names: list[str] = []
    else:
        track_names = sorted(str(name) for name in tracks.group_keys())
    if not track_names:
        errors.append("no track groups")
    required = (
        "frame_indices",
        "track_sample_key",
        "source_acquisition_frame_index",
        "source_frame_interpolation",
        "source_instance_key",
        "source_row_index",
        "positions_px",
        "speed_raw_px",
        "speed_filtered_px",
        "speed_smoothed_px",
        "acceleration_px",
        "heading_degrees",
        "sample_valid",
        "delta_seconds",
    )
    track_rows: dict[str, int] = {}
    for name in track_names:
        track = tracks[name]
        frame_indices = track.get("frame_indices")
        if not isinstance(frame_indices, zarr.Array):
            errors.append(f"{name}: missing frame_indices")
            continue
        row_count = int(frame_indices.shape[0])
        track_rows[name] = row_count
        if int(track.attrs.get("num_samples", -1)) != row_count:
            errors.append(f"{name}: num_samples mismatch")
        for array_name in required:
            item = track.get(array_name)
            if not isinstance(item, zarr.Array):
                errors.append(f"{name}: missing {array_name}")
            elif int(item.ndim) < 1 or int(item.shape[0]) != row_count:
                errors.append(f"{name}: row mismatch for {array_name}")
        positions = track.get("positions_px")
        if isinstance(positions, zarr.Array) and tuple(positions.shape) != (row_count, 2):
            errors.append(f"{name}: positions_px must have shape (N, 2)")
        sample_key = track.get("track_sample_key")
        if isinstance(sample_key, zarr.Array) and tuple(sample_key.shape) != (row_count, 2):
            errors.append(f"{name}: track_sample_key must have shape (N, 2)")

    layout = group.attrs.get("physical_storage_layout")
    array_count = 0
    sharded_count = 0
    structured_single_chunk_count = 0
    for array_path, array in _iter_arrays(group):
        array_count += 1
        shards = getattr(array, "shards", None)
        if shards is not None:
            sharded_count += 1
            chunks = tuple(int(value) for value in array.chunks)
            outer = tuple(int(value) for value in shards)
            if any(outer[i] % chunks[i] for i in range(len(chunks))):
                errors.append(f"{array_path}: shard grid is not chunk aligned")
        elif require_sharded and int(array.ndim) >= 1:
            overrides = (
                layout.get("effective_overridden_array_layouts")
                if isinstance(layout, dict)
                else None
            )
            record = overrides.get(array_path) if isinstance(overrides, dict) else None
            expected_chunks = tuple(max(1, int(value)) for value in array.shape)
            if (
                np.dtype(array.dtype).kind == "V"
                and isinstance(record, dict)
                and record.get("layout_profile")
                == STRUCTURED_DTYPE_SINGLE_CHUNK_LAYOUT
                and record.get("effective_outer_shards") is None
                and tuple(record.get("effective_inner_chunks") or ())
                == expected_chunks
                and tuple(int(value) for value in array.chunks) == expected_chunks
            ):
                structured_single_chunk_count += 1
            else:
                errors.append(f"{array_path}: expected indexed sharding")
    if require_sharded and not isinstance(layout, dict):
        errors.append("missing physical_storage_layout")
    elif require_sharded and layout.get("exact_decoded_validation") is not True:
        errors.append("physical_storage_layout lacks exact decoded validation")
    decoded_payloads = (
        _decoded_contract_payloads(group)
        if verify_decoded_payloads
        else {
            "valid": True,
            "errors": [],
            "scope": "covered_by_bound_payload_integrity_receipt",
            "array_count": 0,
            "aggregate_sha256": None,
            "arrays": {},
        }
    )
    errors.extend(str(error) for error in decoded_payloads["errors"])
    return {
        "valid": not errors,
        "errors": errors,
        "track_rows": track_rows,
        "array_count": array_count,
        "sharded_array_count": sharded_count,
        "structured_single_chunk_array_count": structured_single_chunk_count,
        "require_sharded": bool(require_sharded),
        "require_complete": bool(require_complete),
        "expected_binding_status": expected_binding_status,
        "verify_decoded_payloads": bool(verify_decoded_payloads),
        "decoded_payload_validation": decoded_payloads,
    }


def _writer_result_summary(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must return a mapping, got {type(value).__name__}.")
    summary: dict[str, Any] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        if (
            name in {
                "valid",
                "status",
                "schema_id",
                "schema_version",
                "run_name",
                "track_count",
                "row_count",
                "duration_seconds",
            }
            or name.endswith("_sha256")
            or name.endswith("_count")
        ) and type(raw_value) in {str, int, float, bool, type(None)}:
            summary[name] = raw_value
        elif name == "errors" and isinstance(raw_value, (list, tuple)):
            summary[name] = [str(item) for item in raw_value]
        elif name == "binding_phase_seconds" and isinstance(raw_value, Mapping):
            phases: dict[str, float] = {}
            for raw_phase_name, raw_seconds in raw_value.items():
                phase_name = str(raw_phase_name)
                if (
                    not phase_name
                    or type(raw_seconds) not in {int, float}
                    or not np.isfinite(raw_seconds)
                    or float(raw_seconds) < 0.0
                ):
                    raise ValueError(
                        f"{label} returned an invalid binding phase duration."
                    )
                phases[phase_name] = float(raw_seconds)
            summary[name] = phases
    return summary


def _require_valid_writer_result(value: Any, *, label: str) -> dict[str, Any]:
    summary = _writer_result_summary(value, label=label)
    if summary.get("valid") is not True:
        raise RuntimeError(f"{label} did not report valid=true: {summary}")
    return summary


def publish_track_kinematics_run(
    plan: TrackKinematicsMaterializationPlan,
    *,
    materialization_payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    """Publish, bind final-path coordinates, validate, then update pointers."""

    transaction = {
        "binding_complete": False,
        "completion_published": False,
        "publication_owner_uuid": None,
        "payload_integrity_receipt": None,
    }
    deferred_activation: list[Any] = []

    def retain_deferred_activation(activation: Any) -> None:
        if not isinstance(
            activation,
            track_writer.DeferredTrackKinematicsSelectorActivation,
        ):
            raise RuntimeError(
                "Track completion produced an invalid deferred activation receipt."
            )
        if deferred_activation:
            raise RuntimeError(
                "Track completion attempted to retain more than one activation receipt."
            )
        deferred_activation.append(activation)

    def validate(path: Path) -> dict[str, Any]:
        if not transaction["binding_complete"]:
            return _validate_track_run(
                path,
                require_sharded=True,
                expected_binding_status=PUBLISHING_BINDING_STATUS,
                require_complete=False,
                expected_selector_eligible=False,
            )
        if path.resolve() != plan.target_run_path.resolve():
            return {
                "valid": False,
                "errors": [
                    "bound canonical validation is permitted only at the exact "
                    "authoritative target path"
                ],
                "expected_target_path": str(plan.target_run_path),
                "observed_path": str(path),
            }
        structural = _validate_track_run(
            path,
            require_sharded=True,
            expected_binding_status=BOUND_CANONICAL_STATUS,
            require_complete=transaction["completion_published"],
            expected_selector_eligible=False,
            verify_decoded_payloads=False,
        )
        if not structural["valid"]:
            return structural
        try:
            authoritative_root = open_zarr_root(plan.source_zarr, mode="r")
            run_group = authoritative_root[
                f"analysis/track_kinematics_runs/offline/{plan.run_name}"
            ]
            receipt = transaction["payload_integrity_receipt"]
            if not isinstance(receipt, Mapping):
                raise RuntimeError(
                    "Bound track publication lacks its payload integrity receipt."
                )
            if transaction["completion_published"]:
                publication_owner_uuid = transaction["publication_owner_uuid"]
                if not isinstance(publication_owner_uuid, str):
                    raise RuntimeError(
                        "Completed track publication lacks its exact owner."
                    )
                canonical = track_writer.verify_track_motion_payload_validation_receipt(
                    authoritative_root,
                    run_group,
                    expected_publication_owner_uuid=publication_owner_uuid,
                    run_path=plan.target_run_path,
                    require_complete=True,
                    verify_physical_payload=False,
                    hash_workers=max(1, int(plan.shard_workers)),
                )
                canonical_summary = _writer_result_summary(
                    canonical,
                    label="Completed receipt-bound canonical track validation",
                )
            else:
                integrity = verify_payload_integrity_receipt(
                    plan.target_run_path,
                    receipt,
                    expected_run_ref=f"/{run_group.path}",
                    hash_workers=max(1, int(plan.shard_workers)),
                    verify_physical_payload=False,
                )
                canonical_summary = {
                    "valid": True,
                    "status": "bound_payload_integrity_receipt_valid",
                    "integrity_receipt_sha256": integrity["record_sha256"],
                }
        except Exception as exc:
            canonical_summary = {"valid": False, "errors": [str(exc)]}
        structural["canonical_validation"] = canonical_summary
        if canonical_summary.get("valid") is not True:
            structural["valid"] = False
            structural["errors"].append(
                "writer-owned canonical validation did not report valid=true"
            )
        return structural

    def prepare(root: zarr.Group) -> tuple[zarr.Group, zarr.Group]:
        track_parent = require_runs_parent(
            root.require_group("analysis"),
            "track_kinematics_runs",
        )
        return track_parent, track_parent.require_group("offline")

    def validate_completed_receipt(_fresh_run: zarr.Group) -> Mapping[str, Any]:
        structural = _validate_track_run(
            plan.target_run_path,
            require_sharded=True,
            expected_binding_status=BOUND_CANONICAL_STATUS,
            require_complete=True,
            expected_selector_eligible=False,
            verify_decoded_payloads=False,
        )
        if not structural["valid"]:
            return structural
        publication_owner_uuid = transaction["publication_owner_uuid"]
        if not isinstance(publication_owner_uuid, str):
            return {
                "valid": False,
                "errors": ["completed track publication lacks its exact owner"],
            }
        authoritative_root = open_zarr_root(plan.source_zarr, mode="r")
        run_group = authoritative_root[
            f"analysis/track_kinematics_runs/offline/{plan.run_name}"
        ]
        receipt_validation = (
            track_writer.verify_track_motion_payload_validation_receipt(
                authoritative_root,
                run_group,
                expected_publication_owner_uuid=publication_owner_uuid,
                run_path=plan.target_run_path,
                require_complete=True,
                verify_physical_payload=False,
                hash_workers=max(1, int(plan.shard_workers)),
            )
        )
        structural["canonical_validation"] = dict(receipt_validation)
        if receipt_validation.get("valid") is not True:
            structural["valid"] = False
            structural["errors"].append(
                "receipt-bound canonical validation did not report valid=true"
            )
        return structural

    def after_rename(
        root: zarr.Group,
        run: zarr.Group,
    ) -> dict[str, Any]:
        if transaction["binding_complete"] or transaction["completion_published"]:
            raise RuntimeError("Track publication transaction state is inconsistent.")
        if str(run.attrs.get(COORDINATE_BINDING_STATUS_ATTR)) != PUBLISHING_BINDING_STATUS:
            raise RuntimeError(
                "Final-path binder requires exact publishing coordinate status."
            )
        if str(run.attrs.get("palette_run_completion_status")) != "running":
            raise RuntimeError(
                "Final-path binder requires a generically incomplete run."
            )
        decoded_copy_report = materialization_payload.get("sharded_copy")
        if not isinstance(decoded_copy_report, Mapping):
            raise RuntimeError(
                "Track publication lacks its exact decoded sharded-copy report."
            )
        receipt_started = time.perf_counter()
        integrity_receipt = build_payload_integrity_receipt(
            plan.target_run_path,
            run_ref=f"/{run.path}",
            decoded_copy_report=decoded_copy_report,
            hash_workers=max(1, int(plan.shard_workers)),
        )
        receipt_build_seconds = float(time.perf_counter() - receipt_started)
        transaction["publication_owner_uuid"] = (
            track_writer._track_publication_owner_uuid(run)
        )
        binding_started = time.perf_counter()
        result = track_writer.bind_staged_offline_track_kinematics_run(
            root,
            run,
            expected_keypoint_run=plan.keypoint_run,
            expected_run_name=plan.run_name,
            payload_integrity_receipt=integrity_receipt,
            payload_run_path=plan.target_run_path,
            payload_hash_workers=max(1, int(plan.shard_workers)),
        )
        canonical_binding_seconds = float(time.perf_counter() - binding_started)
        summary = _require_valid_writer_result(
            result,
            label="Final-path canonical track binding",
        )
        if str(run.attrs.get(COORDINATE_BINDING_STATUS_ATTR)) != BOUND_CANONICAL_STATUS:
            raise RuntimeError(
                "Canonical binder returned without setting exact bound status."
            )
        if str(run.attrs.get("palette_run_completion_status")) != "running":
            raise RuntimeError(
                "Canonical binder must not mark the run complete or update pointers."
            )
        post_binding_receipt_check_started = time.perf_counter()
        bound_integrity = run.attrs.get(
            track_writer.TRACK_MOTION_PAYLOAD_INTEGRITY_RECEIPT_ATTR
        )
        if bound_integrity != json_attr_safe(integrity_receipt):
            raise RuntimeError(
                "Canonical binder did not retain the exact verified payload receipt."
            )
        binding_validation = run.attrs.get(
            track_writer.TRACK_KINEMATICS_BINDING_VALIDATION_RECEIPT_ATTR
        )
        if (
            not isinstance(binding_validation, Mapping)
            or binding_validation.get("record_sha256")
            != summary.get("binding_validation_receipt_sha256")
        ):
            raise RuntimeError(
                "Canonical binder did not retain its exact validation receipt."
            )
        post_binding_receipt_check_seconds = float(
            time.perf_counter() - post_binding_receipt_check_started
        )
        transaction["payload_integrity_receipt"] = integrity_receipt
        transaction["binding_complete"] = True
        return {
            "canonical_binding": summary,
            "payload_integrity_receipt": {
                "schema_id": integrity_receipt["schema_id"],
                "schema_version": integrity_receipt["schema_version"],
                "record_sha256": integrity_receipt["record_sha256"],
                "decoded_payload_root_sha256": integrity_receipt[
                    "decoded_payload"
                ]["root_sha256"],
                "physical_payload_root_sha256": integrity_receipt[
                    "physical_payload"
                ]["root_sha256"],
                "immutable_metadata_root_sha256": integrity_receipt[
                    "immutable_metadata"
                ]["root_sha256"],
                "receipt_build_seconds": receipt_build_seconds,
                "canonical_binding_seconds": canonical_binding_seconds,
                "post_binding_receipt_check_seconds": (
                    post_binding_receipt_check_seconds
                ),
            },
        }

    def complete(
        root: zarr.Group,
        _parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        if not transaction["binding_complete"] or transaction["completion_published"]:
            raise RuntimeError(
                "Track completion requires exactly one successful final-path binding."
            )
        publication_owner_uuid = transaction["publication_owner_uuid"]
        if not isinstance(publication_owner_uuid, str):
            raise RuntimeError(
                "Track completion lacks its exact materializer publication owner."
            )
        activation = track_writer.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=plan.run_name,
            run_type="offline",
            publication_owner_uuid=publication_owner_uuid,
            validate_complete_run=validate_completed_receipt,
            payload_integrity_receipt=transaction["payload_integrity_receipt"],
            payload_run_path=plan.target_run_path,
            payload_hash_workers=max(1, int(plan.shard_workers)),
            defer_selector_eligibility=True,
            deferred_activation_sink=retain_deferred_activation,
        )
        if (
            not callable(activation)
            or len(deferred_activation) != 1
            or deferred_activation[0] is not activation
        ):
            raise RuntimeError(
                "Track completion did not retain and return one exact deferred "
                "eligibility receipt."
            )
        transaction["completion_published"] = True

    def activate(
        root: zarr.Group,
        parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        if len(deferred_activation) != 1:
            raise RuntimeError(
                "Track publication lacks one deferred eligibility commit."
            )
        final_publisher_payload = run.attrs.get("cluster_output_staging")
        if not isinstance(final_publisher_payload, Mapping):
            raise RuntimeError(
                "Track publication lacks final cluster-output metadata before "
                "eligibility commit."
            )
        deferred_activation[0](
            root,
            parent,
            run,
            validate_fresh_complete_run=validate_completed_receipt,
            expected_cluster_output_staging=json_attr_safe(
                dict(final_publisher_payload)
            ),
        )

    def rollback_activation() -> None:
        if deferred_activation:
            rollback_root = open_zarr_root(plan.source_zarr, mode="a")
            track_writer.rollback_deferred_track_kinematics_selector_activation(
                deferred_activation[0],
                root=rollback_root,
            )

    def verify(root: zarr.Group) -> None:
        pointer_parent = root["analysis/track_kinematics_runs"]
        pointer_offline = pointer_parent["offline"]
        if (
            str(pointer_parent.attrs.get("latest")) != f"offline/{plan.run_name}"
            or str(pointer_parent.attrs.get("latest_complete")) != f"offline/{plan.run_name}"
            or str(pointer_parent.attrs.get("latest_offline")) != plan.run_name
            or str(pointer_offline.attrs.get("latest")) != plan.run_name
            or str(
                pointer_offline[plan.run_name].attrs.get(
                    COORDINATE_BINDING_STATUS_ATTR
                )
            )
            != BOUND_CANONICAL_STATUS
            or str(
                pointer_offline[plan.run_name].attrs.get(
                    "palette_run_completion_status"
                )
            )
            != "complete"
            or pointer_offline[plan.run_name].attrs.get(
                "stage_selector_eligible"
            )
            is not False
        ):
            raise RuntimeError("Track-kinematics parent pointers were not updated consistently.")

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.sharded_run,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="track-kinematics-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy=(
                "read_only_compute_unbound_stage_shard_final_path_bind_then_publish"
            ),
            rollback_policy=(
                "retain_owner_bound_failed_public_tombstone_and_restore_owned_selector_attrs"
            ),
            content_checksum=True,
            publication_owner_attr=(
                track_writer.TRACK_KINEMATICS_PUBLICATION_OWNER_ATTR
            ),
        ),
        copy_backend=copy_backend,
        validate_run=validate,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
        rollback_activation=rollback_activation,
        after_rename=after_rename,
        payload_metadata={
            "local_run_path": str(plan.local_run_path),
            "sharded_run_path": str(plan.sharded_run),
            "copy_backend": copy_backend,
            "materialization": json_attr_safe(materialization_payload),
        },
    )


def materialize_track_kinematics(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    keypoint_run: str,
    run_name: str,
    output_shard_rows: int = DEFAULT_OUTPUT_SHARD_ROWS,
    shard_workers: int = 1,
    writer_arguments: Sequence[str] = (),
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    telemetry = PhaseTelemetry(
        materializer="track_kinematics",
        context={
            "requested_shard_workers": int(shard_workers),
            "output_shard_rows": int(output_shard_rows),
            "copy_backend": str(copy_backend),
        },
    )
    with telemetry.phase("materialization_plan"):
        plan = build_track_kinematics_materialization_plan(
            source_zarr,
            scratch_root=scratch_root,
            keypoint_run=keypoint_run,
            run_name=run_name,
            output_shard_rows=output_shard_rows,
            shard_workers=shard_workers,
            writer_arguments=writer_arguments,
        )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        result["runtime_telemetry"] = telemetry.to_json()
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    with telemetry.phase("scratch_prepare"):
        plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        with telemetry.phase("offline_numeric_staging"):
            stage_result = track_writer.stage_offline_track_kinematics_run(
                plan.source_zarr,
                plan.staging_zarr,
                keypoint_run=plan.keypoint_run,
                run_name=plan.run_name,
                writer_arguments=plan.writer_arguments,
            )
        compute_seconds = telemetry.duration_seconds("offline_numeric_staging") or 0.0
        with telemetry.phase("staging_result_summary"):
            stage_summary = _writer_result_summary(
                stage_result,
                label="Offline track numerical staging",
            )
        if stage_summary.get("valid") is False:
            raise RuntimeError(
                f"Offline track numerical staging reported invalid: {stage_summary}"
            )
        with telemetry.phase("local_unbound_validation"):
            regular_validation = _validate_track_run(
                plan.local_run_path,
                require_sharded=False,
                expected_binding_status=UNBOUND_STAGE_STATUS,
                require_complete=True,
                expected_selector_eligible=False,
            )
        if not regular_validation["valid"]:
            raise RuntimeError(
                f"Local unbound numerical track stage is invalid: {regular_validation}"
            )
        with telemetry.phase("shard_materialization_and_decoded_validation"):
            sharded_copy = copy_completed_run_to_sharded(
                plan.local_run_path,
                plan.sharded_run,
                row_count_array=None,
                shard_rows=plan.output_shard_rows,
                workers=plan.shard_workers,
            )
            if sharded_copy.get("exact_decoded_validation") is not True:
                raise RuntimeError(
                    "Sharded track copy did not report exact decoded validation."
                )
        local_payload = {
            "source_access": "authoritative_zarr_read_only",
            "compute_output": "node_local_unbound_numeric_stage",
            "compute_duration_seconds": compute_seconds,
            "writer_arguments": list(plan.writer_arguments),
            "stage_result": stage_summary,
            "regular_validation": regular_validation,
            "sharded_copy": sharded_copy,
        }
        with telemetry.phase("publishing_state_transition"):
            sharded = open_zarr_root(plan.sharded_run, mode="a")
            if (
                str(sharded.attrs.get(COORDINATE_BINDING_STATUS_ATTR))
                != UNBOUND_STAGE_STATUS
            ):
                raise RuntimeError(
                    "Sharded copy did not preserve exact unbound coordinate status."
                )
            if str(sharded.attrs.get("palette_run_completion_status")) != "complete":
                raise RuntimeError(
                    "Sharded copy did not preserve generic stage completion."
                )
            sharded.attrs[COORDINATE_BINDING_STATUS_ATTR] = PUBLISHING_BINDING_STATUS
            sharded.attrs["palette_run_completion_status"] = "running"
            if "palette_run_completed_at_utc" in sharded.attrs:
                del sharded.attrs["palette_run_completed_at_utc"]
            sharded.attrs["node_local_materialization"] = json_attr_safe(local_payload)
        with telemetry.phase("sharded_publishing_validation"):
            publishing_validation = _validate_track_run(
                plan.sharded_run,
                require_sharded=True,
                expected_binding_status=PUBLISHING_BINDING_STATUS,
                require_complete=False,
                expected_selector_eligible=False,
            )
        if not publishing_validation["valid"]:
            raise RuntimeError(
                "Sharded publishing-stage track run is invalid: "
                f"{publishing_validation}"
            )
        local_payload["publishing_validation"] = publishing_validation
        with telemetry.phase("local_materialization_metadata_write"):
            sharded.attrs["node_local_materialization"] = json_attr_safe(local_payload)
        with telemetry.phase("authoritative_publish"):
            publish = publish_track_kinematics_run(
                plan,
                materialization_payload=local_payload,
                copy_backend=copy_backend,
            )
        result.update(
            {
                "status": "complete",
                "local_materialization": local_payload,
                "publish": publish,
            }
        )
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            with telemetry.phase("scratch_cleanup"):
                shutil.rmtree(plan.scratch_root)
        result["runtime_telemetry"] = telemetry.to_json()


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_track_kinematics_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_track_kinematics_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute track kinematics locally, shard it, and atomically publish."
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--keypoint-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output-shard-rows", type=int, default=DEFAULT_OUTPUT_SHARD_ROWS)
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] != "--":
        parser.error(
            "unrecognized materializer arguments; place track-writer arguments after --"
        )
    writer_arguments = tuple(remaining)
    if writer_arguments[:1] == ("--",):
        writer_arguments = writer_arguments[1:]
    result = materialize_track_kinematics(
        args.zarr_path,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        keypoint_run=args.keypoint_run,
        run_name=args.run_name,
        output_shard_rows=args.output_shard_rows,
        shard_workers=args.shard_workers,
        writer_arguments=writer_arguments,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
