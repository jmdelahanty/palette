"""Opt-in byte-planned physical storage for exact subject-shape v4 runs.

The logical full-anatomy schema remains owned by
``subject_shape_coordinate_publication``.  This module only adapts that exact
array inventory to the shared byte planner and Zarr-v3 array factory.  The
profile is deliberately family-local and selector-ineligible until a later
mounted-reader promotion gate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis.subject_shape_runs import (
    load_sealed_unbound_subject_shape_manifest,
)
from fisheye.shared.coordinate_frame_record import ARRAY_PAYLOAD_CANONICALIZATION
from fisheye.shared.coordinate_record import (
    coordinate_record_sha256,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR,
    SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR,
    SUBJECT_SHAPE_STORAGE_PLAN_ATTR,
    SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR,
    SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR,
    SUBJECT_SHAPE_STORAGE_PROFILE_ROLE,
    SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID,
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID,
    build_subject_shape_schema_inventory_record,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    AnalysisStoragePlanReceipt,
    analysis_storage_plan_receipt_from_manifest,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, DTypeContract
from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
    create_array_from_plan,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import KIB, MIB, StorageProfile


SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE = "legacy_explicit_chunks"
SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID = (
    "subject_shape_access_aware_candidate_v1"
)
SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES = (
    SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)

# Subject-shape geometry is normally resolved per displayed row.  A 128 KiB
# inner target bounds random-frame amplification, while 8 MiB indexed shards
# keep immutable full-duration object fanout low.  Tiny fixed semantic axes are
# eager and remain one object under ``eager_max_bytes``.
SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1 = StorageProfile(
    profile_id=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    target_chunk_bytes=128 * KIB,
    min_chunk_bytes=128 * KIB,
    max_chunk_bytes=128 * KIB,
    eager_max_bytes=8 * MIB,
    target_shard_bytes=8 * MIB,
    per_row_target_shard_bytes=8 * MIB,
    max_shard_bytes=8 * MIB,
    max_payload_objects=4_096,
    codec_profile_id="zstd_fast_v1",
    shard_immutable=True,
    shard_owned_appends=True,
    target_chunk_bytes_by_access=((AccessPattern.EAGER, 1 * MIB),),
)

_STATIC_AXIS_PATHS = frozenset(
    {
        "components/subject_body/tail_sample_s",
        "source_refined_subject_masks/row_revision_available",
    }
)
_BOUND_ZERO_FILL_FLOAT_PATHS = frozenset({"component_centroid_xy"})
_NEGATIVE_ONE_FILL_INT_PATHS = frozenset(
    {"components/subject_body/bspline_degree_used"}
)
_RESERVED_ARRAY_ATTRIBUTES = frozenset(
    {
        "logical_schema_id",
        "logical_schema_version",
        "storage_policy_version",
        "storage_profile_id",
        "codec_profile_id",
        "access_pattern",
        "write_mode",
    }
)


def is_subject_shape_storage_candidate(profile_id: str) -> bool:
    return str(profile_id) == SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID


def _iter_arrays(group: Any, prefix: str = ""):
    for name in sorted(str(value) for value in group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_arrays(group[name], path)


def _iter_groups(group: Any, prefix: str = ""):
    yield prefix, group
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_groups(group[name], path)


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


def _row_count(run_group: Any) -> int:
    node = run_group.get("row_index/instance_key")
    if node is None or len(node.shape) != 1:
        raise ValueError("Subject-shape storage planning requires row_index/instance_key.")
    return int(node.shape[0])


def _is_row_aligned(path: str, shape: tuple[int, ...], row_count: int) -> bool:
    return bool(path not in _STATIC_AXIS_PATHS and shape and shape[0] == row_count)


def _axis_names(path: str, shape: tuple[int, ...], row_count: int) -> tuple[str, ...]:
    if not shape:
        return ()
    first = "row" if _is_row_aligned(path, shape, row_count) else (
        "tail_sample" if path.endswith("/tail_sample_s") else "component"
    )
    return (first, *(f"record_axis_{axis}" for axis in range(1, len(shape))))


def _shape_template(
    path: str,
    shape: tuple[int, ...],
    row_count: int,
) -> tuple[str | int, ...]:
    if _is_row_aligned(path, shape, row_count):
        return ("n_rows", *shape[1:])
    return tuple(shape)


def _authority_role(role: str) -> AnalysisAuthorityRole:
    if role in {"row_identity", "source_row_identity_or_time", "source_revision_lineage"}:
        return AnalysisAuthorityRole.LINEAGE_INDEX
    if role == "compatibility_row_lineage":
        return AnalysisAuthorityRole.COMPATIBILITY_ALIAS
    if role in {"validity_or_flag", "reason_code"}:
        return AnalysisAuthorityRole.QUALITY_DIAGNOSTIC
    if role == "sample_axis":
        return AnalysisAuthorityRole.SEMANTIC_METADATA
    return AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY


def _schema_array_roles(run_group: Any, *, phase: str) -> Mapping[str, Mapping[str, str]]:
    if str(getattr(run_group, "path", "")).strip("/"):
        return build_subject_shape_schema_inventory_record(
            run_group,
            phase=phase,
        )["arrays"]
    if phase == "unbound":
        manifest = run_group.attrs.get(SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR)
    elif phase == "bound":
        manifest = run_group.attrs.get(SUBJECT_SHAPE_MANIFEST_ATTR)
    else:
        raise ValueError(f"Unsupported subject-shape storage phase {phase!r}.")
    schema_inventory = (
        manifest.get("schema_inventory") if isinstance(manifest, Mapping) else None
    )
    roles = (
        schema_inventory.get("arrays")
        if isinstance(schema_inventory, Mapping)
        else None
    )
    if (
        not isinstance(roles, Mapping)
        or schema_inventory.get("phase") != phase
        or schema_inventory.get("closed_array_inventory") is not True
    ):
        raise ValueError(
            f"Isolated subject-shape candidate lacks its exact retained {phase} inventory."
        )
    return roles


def subject_shape_fill_value(path: str, dtype: Any) -> object:
    """Return the exact candidate fill for one maintained v4 array."""

    resolved = np.dtype(dtype)
    if resolved.kind == "f":
        if path in _STATIC_AXIS_PATHS or path in _BOUND_ZERO_FILL_FLOAT_PATHS:
            return 0.0
        return float("nan")
    if resolved == np.dtype(bool):
        return False
    if resolved.kind in "iu":
        if path in _NEGATIVE_ONE_FILL_INT_PATHS:
            return -1
        return 0
    raise ValueError(f"{path}: unsupported fixed-width subject-shape dtype {resolved}.")


def _declaration(
    *,
    path: str,
    array: Any,
    role: str,
    row_count: int,
) -> AnalysisArrayDeclaration:
    shape = tuple(int(value) for value in array.shape)
    dtype = np.dtype(array.dtype)
    row_aligned = _is_row_aligned(path, shape, row_count)
    authority = _authority_role(role)
    fill_value = subject_shape_fill_value(path, dtype)
    if path in _NEGATIVE_ONE_FILL_INT_PATHS:
        fill_semantics = "minus_one_means_invalid"
    elif isinstance(fill_value, float) and math.isnan(float(fill_value)):
        fill_semantics = "nan_for_unpopulated_or_invalid_floating_records"
    else:
        fill_semantics = "zero_or_false_fixed_width_fill"
    return AnalysisArrayDeclaration(
        path=path,
        contract=ArrayContract(
            schema_id=f"analysis.subject_shape_runs.v4.array.{path.replace('/', '.')}",
            schema_version=1,
            dtype=DTypeContract(
                dtype_id=f"numpy.{dtype.name}",
                numpy_dtype=dtype.name,
                variable_length=False,
            ),
            shape_template=_shape_template(path, shape, row_count),
            axis_names=_axis_names(path, shape, row_count),
            description=f"Exact maintained subject-shape v4 array {path}.",
        ),
        required=True,
        access_pattern=(AccessPattern.PER_ROW if row_aligned else AccessPattern.EAGER),
        write_mode=WriteMode.IMMUTABLE,
        authority_role=authority,
        fill_semantics=fill_semantics,
        null_semantics=(
            "validity_arrays_and_reason_codes_define scientific missingness"
        ),
        physical_policy_owner="fisheye.analysis.subject_shape_storage",
        byte_planner_adopted=True,
    )


def build_subject_shape_storage_receipt(
    run_group: Any,
    *,
    phase: str,
    profile: StorageProfile = SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1,
) -> AnalysisStoragePlanReceipt:
    """Recompute one complete byte-derived plan from the exact v4 inventory."""

    if profile != SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1:
        raise ValueError("Subject-shape v4 accepts only its explicit candidate profile.")
    roles = _schema_array_roles(run_group, phase=phase)
    row_count = _row_count(run_group)
    arrays = dict(_iter_arrays(run_group))
    if set(arrays) != set(roles):
        raise ValueError("Subject-shape live arrays differ from the closed schema inventory.")
    declarations = tuple(
        _declaration(
            path=path,
            array=arrays[path],
            role=str(roles[path]["role"]),
            row_count=row_count,
        )
        for path in sorted(arrays)
    )
    facts = {
        declaration.path: AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=tuple(int(value) for value in arrays[declaration.path].shape),
            dtype=arrays[declaration.path].dtype,
            access_unit_semantics=(
                "one complete subject-shape row with all fixed trailing semantic "
                "axes indivisible"
                if _is_row_aligned(
                    declaration.path,
                    tuple(int(value) for value in arrays[declaration.path].shape),
                    row_count,
                )
                else "one complete eager semantic-axis record"
            ),
        )
        for declaration in declarations
    }
    return plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
        dimensions={"n_rows": row_count},
    )


def persist_subject_shape_storage_receipt(
    run_group: Any,
    receipt: AnalysisStoragePlanReceipt,
    *,
    phase: str,
) -> dict[str, object]:
    if receipt.profile != SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1:
        raise ValueError("Cannot persist an unrecognized subject-shape profile.")
    manifest = receipt.as_manifest()
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PLAN_ATTR] = manifest
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR] = manifest["payload_digest"]
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR] = receipt.profile.profile_id
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR] = (
        SUBJECT_SHAPE_STORAGE_PROFILE_ROLE
    )
    run_group.attrs[SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR] = {
        "schema_id": "palette.subject_shape_storage_candidate",
        "schema_version": 1,
        "profile_id": receipt.profile.profile_id,
        "logical_profile_id": "analysis.subject_shape.full_anatomy_v4",
        "phase": phase,
        "selector_eligible": False,
        "promotion_status": "unpromoted_candidate",
    }
    return manifest


def _copy_group_attributes(source: Any, destination: Any) -> None:
    destination.attrs.update(dict(source.attrs))
    for path, source_group in _iter_groups(source):
        if not path:
            continue
        target = destination
        for component in path.split("/"):
            target = target.require_group(component)
        target.attrs.update(dict(source_group.attrs))


def _array_digest(array: Any, *, block_rows: int) -> str:
    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
                "dtype": np.lib.format.dtype_to_descr(dtype),
                "shape": list(shape),
            }
        )
    )
    digest.update(b"\x00")
    if not array.shape:
        digest.update(np.asarray(array[...]).tobytes(order="C"))
        return digest.hexdigest()
    for start in range(0, int(array.shape[0]), max(1, int(block_rows))):
        stop = min(int(array.shape[0]), start + max(1, int(block_rows)))
        digest.update(np.ascontiguousarray(array[start:stop]).tobytes(order="C"))
    return digest.hexdigest()


def _source_manifest_link_record(sealed_manifest: Any) -> dict[str, object]:
    sealed_manifest.assert_verified()
    source_record = sealed_manifest.record
    return {
        "schema_id": SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID,
        "schema_version": 1,
        "source_manifest_attr": SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
        "source_manifest_record_ref": sealed_manifest.record_ref,
        "source_manifest_sha256": sealed_manifest.record_sha256,
        "source_manifest": source_record,
        "array_comparison_canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
        "destination_decoded_equality_required": True,
    }


def persist_subject_shape_storage_source_manifest_link(
    run_group: Any,
    sealed_manifest: Any,
) -> dict[str, object]:
    """Retain the original producer seal before any physical restamping."""

    record = _source_manifest_link_record(sealed_manifest)
    digest = coordinate_record_sha256(record)
    digest_attr = f"{SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR}_sha256"
    if (
        SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR in run_group.attrs
        or digest_attr in run_group.attrs
    ):
        raise ValueError("Subject-shape source-manifest link is already occupied.")
    run_group.attrs.update(
        {
            SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR: record,
            digest_attr: digest,
        }
    )
    if (
        run_group.attrs.get(SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR) != record
        or run_group.attrs.get(digest_attr) != digest
    ):
        raise RuntimeError("Subject-shape source-manifest link changed while stamping.")
    return {
        "record_sha256": digest,
        "source_manifest_sha256": sealed_manifest.record_sha256,
    }


def validate_subject_shape_storage_source_manifest_link(
    run_group: Any,
    *,
    phase: str,
    verify_content: bool = False,
    block_rows: int = 1_024,
) -> tuple[str, ...]:
    """Validate the retained producer seal and, optionally, every live payload."""

    errors: list[str] = []
    link = run_group.attrs.get(SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR)
    link_digest = run_group.attrs.get(
        f"{SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR}_sha256"
    )
    if not isinstance(link, Mapping) or not isinstance(link_digest, str):
        return ("subject-shape source-manifest link is absent",)
    try:
        if coordinate_record_sha256(link) != link_digest:
            return ("subject-shape source-manifest link digest is stale",)
    except Exception as exc:
        return (f"subject-shape source-manifest link is invalid: {exc}",)
    expected_fields = {
        "schema_id",
        "schema_version",
        "source_manifest_attr",
        "source_manifest_record_ref",
        "source_manifest_sha256",
        "source_manifest",
        "array_comparison_canonicalization",
        "destination_decoded_equality_required",
    }
    source_manifest = link.get("source_manifest")
    source_digest = link.get("source_manifest_sha256")
    try:
        source_digest_valid = (
            isinstance(source_manifest, Mapping)
            and coordinate_record_sha256(source_manifest) == source_digest
        )
    except Exception:
        source_digest_valid = False
    source_inventory = (
        source_manifest.get("schema_inventory")
        if isinstance(source_manifest, Mapping)
        else None
    )
    source_run_ref = (
        source_inventory.get("run_ref")
        if isinstance(source_inventory, Mapping)
        else None
    )
    expected_source_record_ref = (
        f"{source_run_ref}@{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}"
        if isinstance(source_run_ref, str) and source_run_ref.startswith("/")
        else None
    )
    if (
        set(link) != expected_fields
        or link.get("schema_id") != SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID
        or link.get("schema_version") != 1
        or link.get("source_manifest_attr") != SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR
        or not isinstance(source_manifest, Mapping)
        or link.get("source_manifest_record_ref") != expected_source_record_ref
        or not isinstance(source_digest, str)
        or not source_digest_valid
        or link.get("array_comparison_canonicalization")
        != ARRAY_PAYLOAD_CANONICALIZATION
        or link.get("destination_decoded_equality_required") is not True
    ):
        return ("subject-shape source-manifest linkage envelope is not exact",)
    source_arrays = source_manifest.get("arrays")
    inventory_arrays = (
        source_inventory.get("arrays")
        if isinstance(source_inventory, Mapping)
        else None
    )
    if (
        source_manifest.get("schema_id")
        != SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID
        or source_manifest.get("schema_version") != 1
        or set(source_manifest)
        != {
            "schema_id",
            "schema_version",
            "run_name",
            "binding_status",
            "source_refined_subject_masks_run",
            "method",
            "method_version",
            "component_names",
            "scientific_configuration",
            "schema_inventory",
            "arrays",
            "closed_array_inventory",
            "closed_group_inventory",
            "closed_attr_inventory",
            "coordinate_descriptors_present",
        }
        or source_manifest.get("closed_array_inventory") is not True
        or not isinstance(source_arrays, Mapping)
        or not isinstance(inventory_arrays, Mapping)
        or set(source_arrays) != set(inventory_arrays)
    ):
        return ("subject-shape linked producer manifest inventory is invalid",)
    live_arrays = dict(_iter_arrays(run_group))
    if phase == "unbound" and set(live_arrays) != set(source_arrays):
        errors.append("subject-shape unbound arrays differ from the producer seal")
    elif phase == "bound" and not set(source_arrays).issubset(live_arrays):
        errors.append("subject-shape bound run omits producer-sealed arrays")
    elif phase not in {"unbound", "bound"}:
        errors.append(f"unsupported subject-shape source-link phase {phase!r}")
    for path in sorted(set(source_arrays) & set(live_arrays)):
        entry = source_arrays[path]
        array = live_arrays[path]
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "relative_ref",
                "dtype",
                "shape",
                "content_sha256",
                "canonicalization",
            }
            or entry.get("relative_ref") != path
            or entry.get("dtype") != np.dtype(array.dtype).str
            or entry.get("shape") != [int(value) for value in array.shape]
            or entry.get("canonicalization") != ARRAY_PAYLOAD_CANONICALIZATION
            or not isinstance(entry.get("content_sha256"), str)
            or len(entry["content_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["content_sha256"]
            )
        ):
            errors.append(f"{path}: differs from the producer-sealed declaration")
            continue
        if verify_content and _array_digest(
            array,
            block_rows=block_rows,
        ) != entry.get("content_sha256"):
            errors.append(f"{path}: content differs from the producer seal")
    return tuple(errors)


def materialize_subject_shape_storage_candidate(
    source_run: Any,
    destination_path: str | Path,
    *,
    phase: str = "unbound",
    copy_block_rows: int = 1_024,
) -> dict[str, object]:
    """Write one complete node-local candidate through the shared factory."""

    destination = Path(destination_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing existing subject-shape candidate: {destination}")
    sealed_manifest = load_sealed_unbound_subject_shape_manifest(source_run)
    sealed_arrays = sealed_manifest.record["arrays"]
    receipt = build_subject_shape_storage_receipt(source_run, phase=phase)
    entry_by_path = {entry.declaration.path: entry for entry in receipt.entries}
    destination_run = zarr.open_group(str(destination), mode="w", zarr_format=3)
    _copy_group_attributes(source_run, destination_run)
    hashes: dict[str, str] = {}
    for path, source_array in _iter_arrays(source_run):
        entry = entry_by_path[path]
        parent, leaf = _parent_and_leaf(destination_run, path)
        destination_array = create_array_from_plan(
            parent,
            name=leaf,
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=subject_shape_fill_value(path, source_array.dtype),
            attributes=dict(source_array.attrs),
        )
        if source_array.shape:
            for start in range(
                0,
                int(source_array.shape[0]),
                max(1, int(copy_block_rows)),
            ):
                stop = min(
                    int(source_array.shape[0]),
                    start + max(1, int(copy_block_rows)),
                )
                destination_array[start:stop] = source_array[start:stop]
        else:
            destination_array[...] = source_array[...]
        destination_hash = _array_digest(
            destination_array,
            block_rows=copy_block_rows,
        )
        expected_hash = sealed_arrays[path]["content_sha256"]
        if destination_hash != expected_hash:
            raise RuntimeError(
                f"Decoded candidate differs from the producer seal for {path!r}."
            )
        hashes[path] = destination_hash
    source_manifest_link = persist_subject_shape_storage_source_manifest_link(
        destination_run,
        sealed_manifest,
    )
    persist_subject_shape_storage_receipt(
        destination_run,
        receipt,
        phase=phase,
    )
    errors = validate_subject_shape_candidate_storage(
        destination_run,
        phase=phase,
    )
    if errors:
        raise RuntimeError("Subject-shape candidate validation failed: " + "; ".join(errors))
    return {
        "schema_id": "palette.subject_shape_storage_materialization",
        "schema_version": 1,
        "phase": phase,
        "array_count": len(hashes),
        "decoded_equality": True,
        "array_content_sha256": hashes,
        "source_manifest_link": source_manifest_link,
        "storage_plan": receipt.as_manifest(),
    }


def create_bound_subject_shape_candidate_array(
    run_group: Any,
    group: Any,
    *,
    name: str,
    values: np.ndarray,
) -> Any:
    """Create one final-binding array with the candidate profile."""

    path_prefix = str(getattr(group, "path", "")).strip("/")
    run_path = str(getattr(run_group, "path", "")).strip("/")
    if path_prefix == run_path:
        relative_path = name
    elif path_prefix.startswith(f"{run_path}/"):
        relative_path = f"{path_prefix[len(run_path) + 1:]}/{name}"
    else:
        raise ValueError("Candidate binding array is outside its subject-shape run.")
    data = np.asarray(values)
    row_count = _row_count(run_group)
    declaration = _declaration(
        path=relative_path,
        array=data,
        role=(
            "row_identity"
            if relative_path == "instance_key"
            else "source_row_identity_or_time"
            if relative_path in {"source_crop_row_ids", "source_acquisition_frame_index"}
            else "validity_or_flag"
            if relative_path.endswith("valid")
            else "coordinate_geometry"
        ),
        row_count=row_count,
    )
    facts = AnalysisArrayStorageFacts(
        path=relative_path,
        shape=tuple(int(value) for value in data.shape),
        dtype=data.dtype,
        access_unit_semantics=(
            "one complete subject-shape row with all fixed trailing semantic axes indivisible"
        ),
    )
    receipt = plan_analysis_storage(
        (declaration,),
        {relative_path: facts},
        profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1,
        dimensions={"n_rows": row_count},
    )
    entry = receipt.entries[0]
    array = create_array_from_plan(
        group,
        name=name,
        contract=declaration.contract,
        plan=entry.plan,
        fill_value=subject_shape_fill_value(relative_path, data.dtype),
    )
    if data.size:
        array[...] = data
    return array


def finalize_bound_subject_shape_storage_receipt(run_group: Any) -> dict[str, object]:
    """Replace the unbound receipt with the exact final bound inventory."""

    receipt = build_subject_shape_storage_receipt(run_group, phase="bound")
    return persist_subject_shape_storage_receipt(run_group, receipt, phase="bound")


def _normalized_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_metadata(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalized_metadata(child) for child in value]
    if value == "NaN":
        return {"palette_exact_float": "nan"}
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return {"palette_exact_float": "nan"}
    return value


def _metadata_without_runtime_group_consolidation(
    value: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    """Strip a proven well-formed runtime subtree envelope from one group.

    Zarr reconstructs a non-empty inline envelope on a group reached through
    the root consolidated view.  The flattened entries are validated
    independently against the direct subtree below; this helper then removes
    only the exact envelope shape so the group's own declaration can be
    compared.
    """

    normalized = dict(value)
    envelope = normalized.pop("consolidated_metadata", None)
    if normalized.get("node_type") != "group":
        raise ValueError(f"Expected a Zarr group declaration at {path!r}.")
    if envelope is None:
        return normalized
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"kind", "must_understand", "metadata"}
        or envelope.get("kind") != "inline"
        or envelope.get("must_understand") is not False
        or not isinstance(envelope.get("metadata"), Mapping)
    ):
        raise ValueError(
            f"Runtime consolidated subtree at {path!r} is not an exact inline envelope."
        )
    return normalized


def _direct_subtree_declaration_map(run_group: Any) -> dict[str, Any]:
    declarations: dict[str, Any] = {}
    for path, group in _iter_groups(run_group):
        if not path:
            continue
        declarations[path] = metadata_without_empty_group_consolidation(
            group.metadata.to_dict(),
            path=path,
        )
    for path, array in _iter_arrays(run_group):
        declarations[path] = metadata_without_empty_group_consolidation(
            array.metadata.to_dict(),
            path=path,
        )
    return declarations


def _consolidated_subtree_declaration_map(run_group: Any) -> dict[str, Any]:
    raw = run_group.metadata.to_dict()
    envelope = raw.get("consolidated_metadata")
    if (
        not isinstance(envelope, Mapping)
        or set(envelope) != {"kind", "must_understand", "metadata"}
        or envelope.get("kind") != "inline"
        or envelope.get("must_understand") is not False
        or not isinstance(envelope.get("metadata"), Mapping)
    ):
        raise ValueError(
            "Consolidated subject-shape run lacks one exact inline subtree envelope."
        )
    declarations: dict[str, Any] = {}
    for raw_path, raw_declaration in envelope["metadata"].items():
        path = str(raw_path)
        if not path or not isinstance(raw_declaration, Mapping):
            raise ValueError(
                "Consolidated subject-shape subtree contains an invalid relative path."
            )
        declarations[path] = metadata_without_empty_group_consolidation(
            raw_declaration,
            path=path,
        )
    return declarations


def validate_subject_shape_candidate_storage(
    run_group: Any,
    *,
    phase: str,
) -> tuple[str, ...]:
    """Replan and validate every candidate array declaration fail closed."""

    errors: list[str] = []
    errors.extend(
        validate_subject_shape_storage_source_manifest_link(
            run_group,
            phase=phase,
        )
    )
    persisted = run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PLAN_ATTR)
    if not isinstance(persisted, Mapping):
        return ("subject-shape storage-plan receipt is absent",)
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(persisted)
    except Exception as exc:
        return (f"subject-shape storage-plan receipt is invalid: {exc}",)
    if parsed.profile != SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1:
        errors.append("subject-shape storage profile is not the frozen candidate")
    try:
        expected = build_subject_shape_storage_receipt(
            run_group,
            phase=phase,
        )
    except Exception as exc:
        return (*errors, f"subject-shape storage plan cannot be recomputed: {exc}")
    if dict(persisted) != expected.as_manifest():
        errors.append("subject-shape storage receipt differs from executable planning")
    if run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR) != persisted.get(
        "payload_digest"
    ):
        errors.append("subject-shape redundant storage-plan digest mismatch")
    if (
        run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR)
        != SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
        or run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR)
        != SUBJECT_SHAPE_STORAGE_PROFILE_ROLE
    ):
        errors.append("subject-shape candidate profile identity/role mismatch")
    candidate = run_group.attrs.get(SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR)
    expected_candidate = {
        "schema_id": "palette.subject_shape_storage_candidate",
        "schema_version": 1,
        "profile_id": SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        "logical_profile_id": "analysis.subject_shape.full_anatomy_v4",
        "phase": phase,
        "selector_eligible": False,
        "promotion_status": "unpromoted_candidate",
    }
    if candidate != expected_candidate:
        errors.append("subject-shape candidate envelope is not exact")
    entry_by_path = {entry.declaration.path: entry for entry in expected.entries}
    arrays = dict(_iter_arrays(run_group))
    if set(arrays) != set(entry_by_path):
        errors.append("subject-shape candidate array inventory differs from receipt")
        return tuple(errors)
    for path in sorted(arrays):
        array = arrays[path]
        entry = entry_by_path[path]
        try:
            raw = metadata_without_empty_group_consolidation(
                array.metadata.to_dict(),
                path=path,
            )
            attributes = raw.get("attributes")
            if not isinstance(attributes, Mapping):
                raise ValueError("array attributes are not an object")
            nonreserved = {
                str(key): value
                for key, value in attributes.items()
                if key not in _RESERVED_ARRAY_ATTRIBUTES
            }
            expected_metadata = array_metadata_declaration_from_plan(
                contract=entry.declaration.contract,
                plan=entry.plan,
                fill_value=subject_shape_fill_value(path, array.dtype),
                attributes=nonreserved,
            )
            observed_metadata = {
                key: value
                for key, value in raw.items()
                if key not in {"zarr_format", "node_type"}
            }
            if _normalized_metadata(observed_metadata) != _normalized_metadata(
                expected_metadata
            ):
                errors.append(f"{path}: physical metadata differs from resolved plan")
        except Exception as exc:
            errors.append(f"{path}: physical metadata validation failed: {exc}")
    return tuple(errors)


def validate_subject_shape_direct_consolidated_storage(
    direct_run: Any,
    consolidated_run: Any,
    *,
    phase: str = "bound",
) -> tuple[str, ...]:
    errors = [
        *validate_subject_shape_candidate_storage(direct_run, phase=phase),
        *validate_subject_shape_candidate_storage(consolidated_run, phase=phase),
    ]
    if dict(direct_run.attrs) != dict(consolidated_run.attrs):
        errors.append("subject-shape direct/consolidated run attrs differ")
    direct_groups = dict(_iter_groups(direct_run))
    consolidated_groups = dict(_iter_groups(consolidated_run))
    if set(direct_groups) != set(consolidated_groups):
        errors.append("subject-shape direct/consolidated group paths differ")
    for path in sorted(set(direct_groups) & set(consolidated_groups)):
        try:
            direct_declaration = metadata_without_empty_group_consolidation(
                direct_groups[path].metadata.to_dict(),
                path=path or ".",
            )
            consolidated_declaration = _metadata_without_runtime_group_consolidation(
                consolidated_groups[path].metadata.to_dict(),
                path=path or ".",
            )
        except ValueError as exc:
            errors.append(f"{path or '.'}: invalid group declaration: {exc}")
            continue
        if _normalized_metadata(direct_declaration) != _normalized_metadata(
            consolidated_declaration
        ):
            errors.append(
                f"{path or '.'}: direct/consolidated group declarations differ"
            )
    try:
        direct_subtree = _direct_subtree_declaration_map(direct_run)
        consolidated_subtree = _consolidated_subtree_declaration_map(
            consolidated_run
        )
    except ValueError as exc:
        errors.append(f"subject-shape consolidated subtree is invalid: {exc}")
    else:
        if set(direct_subtree) != set(consolidated_subtree):
            errors.append(
                "subject-shape direct/consolidated flattened subtree paths differ"
            )
        for path in sorted(set(direct_subtree) & set(consolidated_subtree)):
            if _normalized_metadata(direct_subtree[path]) != _normalized_metadata(
                consolidated_subtree[path]
            ):
                errors.append(
                    f"{path}: direct/consolidated flattened declarations differ"
                )
    direct_arrays = dict(_iter_arrays(direct_run))
    consolidated_arrays = dict(_iter_arrays(consolidated_run))
    if set(direct_arrays) != set(consolidated_arrays):
        errors.append("subject-shape direct/consolidated array paths differ")
        return tuple(errors)
    for path in sorted(direct_arrays):
        if _normalized_metadata(direct_arrays[path].metadata.to_dict()) != (
            _normalized_metadata(consolidated_arrays[path].metadata.to_dict())
        ):
            errors.append(f"{path}: direct/consolidated declarations differ")
    return tuple(errors)


def set_subject_shape_metadata_visibility_policy(
    run_group: Any,
    *,
    expected_array_count: int,
) -> None:
    run_group.attrs[SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR] = {
        "schema_id": "palette.subject_shape_metadata_visibility_policy",
        "schema_version": 1,
        "profile_id": SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        "expected_array_count": int(expected_array_count),
        "direct_consolidated_run_attrs_required": True,
        "direct_consolidated_group_declarations_required": True,
        "direct_consolidated_array_declarations_required": True,
        "direct_consolidated_exact_node_inventory_required": True,
        "root_consolidation_is_final_visibility_step": True,
    }


__all__ = [
    "SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID",
    "SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1",
    "SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE",
    "SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES",
    "build_subject_shape_storage_receipt",
    "create_bound_subject_shape_candidate_array",
    "finalize_bound_subject_shape_storage_receipt",
    "is_subject_shape_storage_candidate",
    "materialize_subject_shape_storage_candidate",
    "persist_subject_shape_storage_receipt",
    "persist_subject_shape_storage_source_manifest_link",
    "set_subject_shape_metadata_visibility_policy",
    "subject_shape_fill_value",
    "validate_subject_shape_candidate_storage",
    "validate_subject_shape_direct_consolidated_storage",
    "validate_subject_shape_storage_source_manifest_link",
]
