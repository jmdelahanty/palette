"""Versioned byte-planned storage for selector-ineligible track candidates.

The maintained v1 scientific authority stores two NumPy structured lineage
records.  This module never reinterprets those records as opaque bytes.  It
projects their five named fields into primitive v2 arrays, streams every array
through the shared Zarr-v3 planner/factory, and validates exact decoded
equivalence back to the explicit v1 source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fisheye.analysis.direct_writer_storage import (
    persist_direct_writer_storage_receipt,
    validate_direct_writer_storage_receipt,
)
from fisheye.analysis.track_kinematics_schema import (
    TRACK_KINEMATICS_FLAT_LINEAGE_PATHS,
    TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
    build_track_kinematics_flat_lineage_declarations,
)
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.analysis_array_contracts import AnalysisArrayDeclaration
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import StorageProfile
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID = (
    "palette.analysis.track_kinematics.flat_lineage_candidate"
)
TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_ATTR = "track_kinematics_flat_lineage_manifest"
TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_DIGEST_ATTR = (
    "track_kinematics_flat_lineage_manifest_sha256"
)
_STRUCTURED_SOURCE_FIELDS = {
    "source_frame_interpolation/left_source_frame_index": (
        "source_frame_interpolation",
        "left_source_frame_index",
    ),
    "source_frame_interpolation/right_source_frame_index": (
        "source_frame_interpolation",
        "right_source_frame_index",
    ),
    "source_frame_interpolation/right_weight": (
        "source_frame_interpolation",
        "right_weight",
    ),
    "source_instance_key/valid": ("source_instance_key", "valid"),
    "source_instance_key/value": ("source_instance_key", "instance_key"),
}


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _parent_and_leaf(group: Any, path: str) -> tuple[Any, str]:
    components = path.split("/")
    parent = group
    for component in components[:-1]:
        child = parent.get(component)
        if child is None:
            child = parent.create_group(component)
        parent = child
    return parent, components[-1]


def _iter_array_paths(group: Any, prefix: str = ""):
    for name, _array in sorted(group.arrays(), key=lambda item: str(item[0])):
        yield f"{prefix}/{name}" if prefix else str(name)
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_array_paths(child, child_prefix)


def _source_track_ids(source_group: Any) -> tuple[int, ...]:
    node = source_group.get("track_ids")
    if node is None:
        raise ValueError("Track source lacks track_ids.")
    values = np.asarray(node[:])
    if values.dtype != np.dtype("int32") or values.ndim != 1:
        raise ValueError("Track source track_ids must be exact int32[N].")
    result = tuple(int(value) for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(
            "Track source track_ids must be nonempty and strictly increasing."
        )
    expected_groups = {f"id_{track_id}" for track_id in result}
    tracks = source_group.get("tracks")
    if (
        tracks is None
        or set(str(name) for name in tracks.group_keys()) != expected_groups
    ):
        raise ValueError("Track source tracks inventory does not match track_ids.")
    return result


def _source_layout(source_group: Any) -> tuple[tuple[int, ...], bool, bool]:
    track_ids = _source_track_ids(source_group)
    physical = []
    for track_id in track_ids:
        track = source_group[f"tracks/id_{track_id}"]
        interpolation = track.get("source_frame_interpolation")
        instances = track.get("source_instance_key")
        if (
            interpolation is None
            or np.dtype(interpolation.dtype) != TRACK_SAMPLE_INTERPOLATION_DTYPE
            or instances is None
            or np.dtype(instances.dtype) != TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
        ):
            raise ValueError(
                f"Track {track_id} lacks the exact v1 structured lineage authority."
            )
        row_count = int(track["track_sample_key"].shape[0])
        if tuple(interpolation.shape) != (row_count,) or tuple(instances.shape) != (
            row_count,
        ):
            raise ValueError(f"Track {track_id} structured lineage row count differs.")
        instance_values = np.asarray(instances[:])
        if np.any(~instance_values["valid"] & (instance_values["instance_key"] != 0)):
            raise ValueError(
                f"Track {track_id} v1 nullable instance lineage is noncanonical."
            )
        physical.append(track.get("positions_mm") is not None)
    if len(set(physical)) != 1:
        raise ValueError(
            "Track physical peers must be present for every track or none."
        )
    arena = source_group.get("track_arena_ids")
    if arena is not None and (
        np.dtype(arena.dtype) != np.dtype("int32")
        or tuple(arena.shape) != (len(track_ids),)
    ):
        raise ValueError("track_arena_ids must be exact int32[n_tracks].")
    return track_ids, physical[0], arena is not None


def build_flat_candidate_declarations(
    source_group: Any,
) -> tuple[AnalysisArrayDeclaration, ...]:
    track_ids, include_physical, include_arena = _source_layout(source_group)
    return build_track_kinematics_flat_lineage_declarations(
        track_ids=track_ids,
        include_physical=include_physical,
        include_arena_inventory=include_arena,
    )


def _candidate_layout(group: Any) -> tuple[tuple[int, ...], bool, bool]:
    track_ids = _source_track_ids(group)
    physical: list[bool] = []
    for track_id in track_ids:
        track = group[f"tracks/id_{track_id}"]
        row_count = int(track["track_sample_key"].shape[0])
        required = {
            "source_frame_interpolation/left_source_frame_index": "int64",
            "source_frame_interpolation/right_source_frame_index": "int64",
            "source_frame_interpolation/right_weight": "float64",
            "source_instance_key/valid": "bool",
            "source_instance_key/value": "uint64",
        }
        for path, dtype in required.items():
            try:
                array = _array_at_path(track, path)
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Track {track_id} lacks flat lineage {path}."
                ) from exc
            if np.dtype(array.dtype) != np.dtype(dtype) or tuple(array.shape) != (
                row_count,
            ):
                raise ValueError(
                    f"Track {track_id} flat lineage {path} has wrong dtype/shape."
                )
        valid = np.asarray(track["source_instance_key/valid"][:], dtype=bool)
        values = np.asarray(track["source_instance_key/value"][:], dtype=np.uint64)
        if np.any(~valid & (values != 0)):
            raise ValueError(
                f"Track {track_id} flat nullable instance lineage is noncanonical."
            )
        physical.append(track.get("positions_mm") is not None)
    if len(set(physical)) != 1:
        raise ValueError("Candidate physical peers must be all present or all absent.")
    arena = group.get("track_arena_ids")
    if arena is not None and (
        np.dtype(arena.dtype) != np.dtype("int32")
        or tuple(arena.shape) != (len(track_ids),)
    ):
        raise ValueError("Candidate track_arena_ids must be exact int32[n_tracks].")
    return track_ids, physical[0], arena is not None


def _candidate_declarations(
    group: Any,
) -> tuple[AnalysisArrayDeclaration, ...]:
    track_ids, include_physical, include_arena = _candidate_layout(group)
    return build_track_kinematics_flat_lineage_declarations(
        track_ids=track_ids,
        include_physical=include_physical,
        include_arena_inventory=include_arena,
    )


def _candidate_source_path(path: str) -> tuple[str, str | None]:
    components = path.split("/")
    if len(components) >= 4 and components[0] == "tracks":
        relative = "/".join(components[2:])
        source_field = _STRUCTURED_SOURCE_FIELDS.get(relative)
        if source_field is not None:
            return "/".join(components[:2] + [source_field[0]]), source_field[1]
    return path, None


def _source_array_and_field(source_group: Any, path: str) -> tuple[Any, str | None]:
    source_path, field = _candidate_source_path(path)
    return _array_at_path(source_group, source_path), field


def _access_unit_semantics(
    declaration: AnalysisArrayDeclaration,
) -> str:
    if not declaration.contract.shape_template:
        return "whole_scalar"
    if declaration.path in {"track_ids", "track_arena_ids"}:
        return "whole_track_inventory"
    if "n_track_seconds" in " ".join(
        str(value) for value in declaration.contract.shape_template
    ):
        return "complete_track_second_record"
    return "complete_track_sample_record"


def _fill_value(declaration: AnalysisArrayDeclaration) -> object:
    semantics = declaration.fill_semantics
    if semantics == "false":
        return False
    if semantics == "minus_one":
        return -1
    if semantics in {"zero", "zero_all_rows_written"}:
        return 0
    if semantics == "nan":
        return float("nan")
    raise ValueError(
        f"No flat candidate fill mapping for {declaration.path}: {semantics!r}."
    )


def build_flat_candidate_storage_receipt(
    source_group: Any,
    *,
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    declarations = build_flat_candidate_declarations(source_group)
    facts: dict[str, AnalysisArrayStorageFacts] = {}
    for declaration in declarations:
        source, field = _source_array_and_field(source_group, declaration.path)
        dtype = np.dtype(
            source.dtype if field is None else source.dtype.fields[field][0]
        )
        facts[declaration.path] = AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=tuple(int(value) for value in source.shape),
            dtype=dtype,
            access_unit_semantics=_access_unit_semantics(declaration),
        )
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
    )


def _write_from_source(
    destination: Any,
    source: Any,
    *,
    field: str | None,
    block_rows: int,
) -> None:
    if int(source.ndim) == 0:
        values = np.asarray(source[...])
        destination[...] = values if field is None else values[field]
        return
    trailing = (slice(None),) * (int(source.ndim) - 1)
    for start in range(0, int(source.shape[0]), max(1, block_rows)):
        stop = min(start + max(1, block_rows), int(source.shape[0]))
        values = np.asarray(source[(slice(start, stop), *trailing)])
        if field is not None:
            values = values[field]
        destination[(slice(start, stop), *trailing)] = values


def rematerialize_flat_candidate(
    source_group: Any,
    destination_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
) -> None:
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    declarations = build_flat_candidate_declarations(source_group)
    if set(entries) != {declaration.path for declaration in declarations}:
        raise ValueError("Flat candidate receipt does not match source inventory.")
    for declaration in declarations:
        entry = entries[declaration.path]
        source, field = _source_array_and_field(source_group, declaration.path)
        parent, leaf = _parent_and_leaf(destination_group, declaration.path)
        destination = create_array_from_plan(
            parent,
            name=leaf,
            contract=declaration.contract,
            plan=entry.plan,
            fill_value=_fill_value(declaration),
            attributes={
                "track_kinematics_lineage_encoding": (
                    "primitive_field_v2" if field is not None else "native_primitive_v2"
                ),
                "legacy_v1_source_field": field,
            },
        )
        block_rows = (
            int(entry.plan.chunk_shape[0])
            if entry.plan.chunk_shape is not None and entry.plan.chunk_shape
            else max(1, int(source.shape[0]))
        )
        _write_from_source(
            destination,
            source,
            field=field,
            block_rows=block_rows,
        )


def _array_digest(array: Any, *, field: str | None = None) -> str:
    dtype = np.dtype(array.dtype if field is None else array.dtype.fields[field][0])
    digest = hashlib.sha256()
    digest.update(str(dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    if int(array.ndim) == 0:
        values = np.asarray(array[...])
        if field is not None:
            values = values[field]
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
        return digest.hexdigest()
    block_rows = max(1, min(int(array.shape[0]) or 1, 65_536))
    trailing = (slice(None),) * (int(array.ndim) - 1)
    for start in range(0, int(array.shape[0]), block_rows):
        values = np.asarray(
            array[
                (slice(start, min(start + block_rows, int(array.shape[0]))), *trailing)
            ]
        )
        if field is not None:
            values = values[field]
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    return digest.hexdigest()


def flat_candidate_logical_hashes(
    group: Any,
    declarations: Sequence[AnalysisArrayDeclaration],
) -> dict[str, str]:
    return {
        declaration.path: _array_digest(_array_at_path(group, declaration.path))
        for declaration in declarations
    }


def source_flat_projection_hashes(
    source_group: Any,
    declarations: Sequence[AnalysisArrayDeclaration],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in declarations:
        source, field = _source_array_and_field(source_group, declaration.path)
        result[declaration.path] = _array_digest(source, field=field)
    return result


def load_track_lineage_records(
    track_group: Any,
    *,
    lineage_schema_version: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load exact lineage records through one explicit v1/v2 compatibility gate."""

    if lineage_schema_version == 1:
        interpolation = np.asarray(track_group["source_frame_interpolation"][:])
        instances = np.asarray(track_group["source_instance_key"][:])
        if (
            interpolation.dtype != TRACK_SAMPLE_INTERPOLATION_DTYPE
            or instances.dtype != TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
            or interpolation.shape != instances.shape
        ):
            raise ValueError("Legacy v1 track lineage records are not exact.")
        return interpolation, instances
    if lineage_schema_version != TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported track lineage schema version {lineage_schema_version!r}."
        )
    left = np.asarray(
        track_group["source_frame_interpolation/left_source_frame_index"][:]
    )
    right = np.asarray(
        track_group["source_frame_interpolation/right_source_frame_index"][:]
    )
    weight = np.asarray(track_group["source_frame_interpolation/right_weight"][:])
    valid = np.asarray(track_group["source_instance_key/valid"][:])
    values = np.asarray(track_group["source_instance_key/value"][:])
    shape = left.shape
    if (
        left.dtype != np.dtype("int64")
        or right.dtype != np.dtype("int64")
        or weight.dtype != np.dtype("float64")
        or valid.dtype != np.dtype(bool)
        or values.dtype != np.dtype("uint64")
        or any(array.shape != shape for array in (right, weight, valid, values))
        or len(shape) != 1
        or np.any(~valid & (values != 0))
    ):
        raise ValueError("Flat v2 track lineage fields are not exact.")
    interpolation = np.empty(shape, dtype=TRACK_SAMPLE_INTERPOLATION_DTYPE)
    interpolation["left_source_frame_index"] = left
    interpolation["right_source_frame_index"] = right
    interpolation["right_weight"] = weight
    instances = np.empty(shape, dtype=TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE)
    instances["valid"] = valid
    instances["instance_key"] = values
    return interpolation, instances


def build_flat_candidate_manifest(
    *,
    declarations: Sequence[AnalysisArrayDeclaration],
    source_run_path: str,
    source_projection_hashes: Mapping[str, str],
) -> dict[str, object]:
    payload = {
        "schema_id": TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID,
        "schema_version": TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION,
        "status": "unpromoted_selector_ineligible",
        "source_schema": {
            "schema_id": "analysis.track_kinematics_runs",
            "schema_version": 1,
            "structured_lineage_compatibility": "explicit_read_and_reconstruct_only",
        },
        "source_run_path": source_run_path,
        "position_authority": {
            "positions_px_dtype": "float64",
            "positions_mm_dtype": "float64_or_absent_all_tracks",
            "narrowing_permitted": False,
        },
        "flat_lineage_paths": list(TRACK_KINEMATICS_FLAT_LINEAGE_PATHS),
        "arrays": [declaration.as_manifest() for declaration in declarations],
        "source_projection_hashes": dict(sorted(source_projection_hashes.items())),
    }
    return {
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def persist_flat_candidate_contract(
    run_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
    declarations: Sequence[AnalysisArrayDeclaration],
    source_run_path: str,
    source_projection_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    manifest = build_flat_candidate_manifest(
        declarations=declarations,
        source_run_path=source_run_path,
        source_projection_hashes=source_projection_hashes,
    )
    run_group.attrs[TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_ATTR] = json_attr_safe(
        manifest
    )
    run_group.attrs[TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_DIGEST_ATTR] = manifest[
        "payload_digest"
    ]
    persist_direct_writer_storage_receipt(run_group, receipt)
    return manifest


def validate_flat_candidate(
    run_group: Any,
    *,
    source_group: Any | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        declarations = _candidate_declarations(run_group)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "array_count": 0}
    declaration_paths = {declaration.path for declaration in declarations}
    observed_paths = set(_iter_array_paths(run_group))
    if observed_paths != declaration_paths:
        errors.append(
            "flat candidate array inventory differs "
            f"(missing={sorted(declaration_paths - observed_paths)!r}, "
            f"unexpected={sorted(observed_paths - declaration_paths)!r})"
        )
    if any(
        np.dtype(_array_at_path(run_group, path).dtype).fields is not None
        for path in observed_paths
    ):
        errors.append("flat candidate contains a structured dtype")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        errors.append("flat candidate is not selector-ineligible")
    if run_group.attrs.get("storage_candidate_profile_promoted") is not False:
        errors.append("flat candidate profile is not explicitly unpromoted")
    if (
        run_group.attrs.get("schema_id") != TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID
        or run_group.attrs.get("schema_version")
        != TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION
    ):
        errors.append("flat candidate run schema identity is invalid")
    if run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("flat candidate is not complete")
    manifest = run_group.attrs.get(TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "payload",
        "payload_digest",
    }:
        errors.append("flat candidate manifest is missing or inexact")
    else:
        payload = manifest.get("payload")
        expected_payload_fields = {
            "schema_id",
            "schema_version",
            "status",
            "source_schema",
            "source_run_path",
            "position_authority",
            "flat_lineage_paths",
            "arrays",
            "source_projection_hashes",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_payload_fields:
            errors.append("flat candidate manifest payload is not exact")
        elif manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("flat candidate manifest digest mismatch")
        elif run_group.attrs.get(
            TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_DIGEST_ATTR
        ) != manifest.get("payload_digest"):
            errors.append("flat candidate redundant manifest digest mismatch")
        elif list(payload.get("arrays", [])) != [
            declaration.as_manifest() for declaration in declarations
        ]:
            errors.append("flat candidate declaration manifest differs")
        elif (
            payload.get("schema_id") != TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID
            or payload.get("schema_version")
            != TRACK_KINEMATICS_FLAT_LINEAGE_SCHEMA_VERSION
            or payload.get("status") != "unpromoted_selector_ineligible"
            or payload.get("flat_lineage_paths")
            != list(TRACK_KINEMATICS_FLAT_LINEAGE_PATHS)
            or payload.get("source_schema")
            != {
                "schema_id": "analysis.track_kinematics_runs",
                "schema_version": 1,
                "structured_lineage_compatibility": (
                    "explicit_read_and_reconstruct_only"
                ),
            }
            or payload.get("position_authority")
            != {
                "positions_px_dtype": "float64",
                "positions_mm_dtype": "float64_or_absent_all_tracks",
                "narrowing_permitted": False,
            }
            or not isinstance(payload.get("source_run_path"), str)
            or not payload.get("source_run_path")
        ):
            errors.append("flat candidate semantic manifest envelope differs")
    access = {
        declaration.path: _access_unit_semantics(declaration)
        for declaration in declarations
    }
    fills = {declaration.path: _fill_value(declaration) for declaration in declarations}
    errors.extend(
        validate_direct_writer_storage_receipt(
            run_group,
            declarations=declarations,
            access_unit_semantics=access,
            fill_values=fills,
            dimensions={},
        )
    )
    hashes = flat_candidate_logical_hashes(run_group, declarations)
    if source_group is not None:
        try:
            expected = source_flat_projection_hashes(source_group, declarations)
        except Exception as exc:
            errors.append(f"source equality validation failed: {exc}")
        else:
            if hashes != expected:
                errors.append("flat candidate decoded values differ from v1 source")
            if (
                isinstance(manifest, Mapping)
                and isinstance(manifest.get("payload"), Mapping)
                and manifest["payload"].get("source_projection_hashes") != expected
            ):
                errors.append("flat candidate manifest source hashes differ")
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(declarations),
        "logical_hashes": hashes,
    }


__all__ = [
    "TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_ATTR",
    "TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_DIGEST_ATTR",
    "TRACK_KINEMATICS_FLAT_CANDIDATE_SCHEMA_ID",
    "build_flat_candidate_declarations",
    "build_flat_candidate_manifest",
    "build_flat_candidate_storage_receipt",
    "flat_candidate_logical_hashes",
    "load_track_lineage_records",
    "persist_flat_candidate_contract",
    "rematerialize_flat_candidate",
    "source_flat_projection_hashes",
    "validate_flat_candidate",
]
