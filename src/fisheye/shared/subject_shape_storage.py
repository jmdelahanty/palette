"""Shared byte-planned physical storage for maintained subject-shape runs.

The logical full-anatomy schema remains owned by
``subject_shape_coordinate_publication``.  This module only adapts that exact
array inventory to the shared byte planner and Zarr-v3 array factory. The
candidate and supported publication profiles share one physical policy while
retaining distinct selector lifecycles.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.coordinate_frame_record import ARRAY_PAYLOAD_CANONICALIZATION
from fisheye.shared.coordinate_record import (
    coordinate_record_sha256,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID,
    CANONICAL_SUBJECT_SHAPE_PROFILE_ID,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR,
    SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
    SUBJECT_SHAPE_SOURCE_KIND_ATTR,
    SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR,
    SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR,
    SUBJECT_SHAPE_STORAGE_PLAN_ATTR,
    SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR,
    SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR,
    SUBJECT_SHAPE_STORAGE_PROFILE_ROLE,
    SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR,
    SUBJECT_SHAPE_SUPPORTED_STORAGE_PROFILE_ROLE,
    SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID,
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID,
    SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
    build_subject_shape_schema_inventory_record,
    load_unbound_subject_shape_manifest,
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
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import KIB, MIB, StorageProfile
from fisheye.shared.zarr_payload_receipt import (
    decoded_payload_receipt_from_copy_report,
)


SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE = "legacy_explicit_chunks"
SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID = (
    "subject_shape_access_aware_candidate_v1"
)
SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES = (
    SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE,
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID,
)

# Subject-shape geometry is normally resolved per displayed row.  A 128 KiB
# inner target bounds random-frame amplification, while 8 MiB indexed shards
# keep immutable full-duration object fanout low.  Tiny fixed semantic axes are
# eager and remain one object under ``eager_max_bytes``.
def _access_aware_profile(profile_id: str) -> StorageProfile:
    return StorageProfile(
        profile_id=profile_id,
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


SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1 = _access_aware_profile(
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
)
SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_V1 = _access_aware_profile(
    SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID
)
_ACCESS_AWARE_PROFILES = {
    profile.profile_id: profile
    for profile in (
        SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1,
        SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_V1,
    )
}

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


def is_subject_shape_access_aware_storage(profile_id: str) -> bool:
    return str(profile_id) in _ACCESS_AWARE_PROFILES


def subject_shape_access_aware_storage_profile(profile_id: str) -> StorageProfile:
    try:
        return _ACCESS_AWARE_PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported access-aware subject-shape profile {profile_id!r}."
        ) from exc


def _profile_role(profile_id: str) -> str:
    return (
        SUBJECT_SHAPE_STORAGE_PROFILE_ROLE
        if is_subject_shape_storage_candidate(profile_id)
        else SUBJECT_SHAPE_SUPPORTED_STORAGE_PROFILE_ROLE
    )


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


def _logical_profile_id(run_group: Any) -> str:
    """Return the exact logical profile that owns one physical candidate."""

    source_kind = run_group.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
    if source_kind == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND:
        return CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID
    if source_kind in {None, SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND}:
        return CANONICAL_SUBJECT_SHAPE_PROFILE_ID
    raise ValueError(f"Unsupported subject-shape source kind {source_kind!r}.")


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
    """Return the exact candidate fill for one maintained array."""

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
    logical_profile_id: str,
) -> AnalysisArrayDeclaration:
    shape = tuple(int(value) for value in array.shape)
    dtype = np.dtype(array.dtype)
    row_aligned = _is_row_aligned(path, shape, row_count)
    authority = _authority_role(role)
    fill_value = subject_shape_fill_value(path, dtype)
    contract_schema_prefix = (
        "analysis.subject_shape_runs.v4"
        if logical_profile_id == CANONICAL_SUBJECT_SHAPE_PROFILE_ID
        else "analysis.subject_shape_runs.v5"
    )
    description = (
        f"Exact maintained subject-shape v4 array {path}."
        if logical_profile_id == CANONICAL_SUBJECT_SHAPE_PROFILE_ID
        else f"Exact maintained subject-shape v5 array {path}."
    )
    if path in _NEGATIVE_ONE_FILL_INT_PATHS:
        fill_semantics = "minus_one_means_invalid"
    elif isinstance(fill_value, float) and math.isnan(float(fill_value)):
        fill_semantics = "nan_for_unpopulated_or_invalid_floating_records"
    else:
        fill_semantics = "zero_or_false_fixed_width_fill"
    return AnalysisArrayDeclaration(
        path=path,
        contract=ArrayContract(
            schema_id=f"{contract_schema_prefix}.array.{path.replace('/', '.')}",
            schema_version=1,
            dtype=DTypeContract(
                dtype_id=f"numpy.{dtype.name}",
                numpy_dtype=dtype.name,
                variable_length=False,
            ),
            shape_template=_shape_template(path, shape, row_count),
            axis_names=_axis_names(path, shape, row_count),
            description=description,
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
    """Recompute one byte-derived plan from the exact maintained inventory."""

    if profile not in _ACCESS_AWARE_PROFILES.values():
        raise ValueError(
            "Subject-shape runs accept only their explicit access-aware profiles."
        )
    roles = _schema_array_roles(run_group, phase=phase)
    row_count = _row_count(run_group)
    logical_profile_id = _logical_profile_id(run_group)
    arrays = dict(_iter_arrays(run_group))
    if set(arrays) != set(roles):
        raise ValueError("Subject-shape live arrays differ from the closed schema inventory.")
    declarations = tuple(
        _declaration(
            path=path,
            array=arrays[path],
            role=str(roles[path]["role"]),
            row_count=row_count,
            logical_profile_id=logical_profile_id,
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
    if receipt.profile not in _ACCESS_AWARE_PROFILES.values():
        raise ValueError("Cannot persist an unrecognized subject-shape profile.")
    manifest = receipt.as_manifest()
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PLAN_ATTR] = manifest
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PLAN_DIGEST_ATTR] = manifest["payload_digest"]
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR] = receipt.profile.profile_id
    run_group.attrs[SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR] = _profile_role(
        receipt.profile.profile_id
    )
    if is_subject_shape_storage_candidate(receipt.profile.profile_id):
        run_group.attrs[SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR] = {
            "schema_id": "palette.subject_shape_storage_candidate",
            "schema_version": 1,
            "profile_id": receipt.profile.profile_id,
            "logical_profile_id": _logical_profile_id(run_group),
            "phase": phase,
            "selector_eligible": False,
            "promotion_status": "unpromoted_candidate",
        }
    elif SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR in run_group.attrs:
        del run_group.attrs[SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR]
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


def _array_digest_header(*, dtype: np.dtype[Any], shape: tuple[int, ...]) -> Any:
    """Return the canonical subject-shape payload digest before decoded bytes.

    The grammar is intentionally byte-identical to
    ``coordinate_frame_record.array_payload_sha256``.  Keeping the incremental
    form here lets the immutable writer hash the exact values it writes and
    reads back without reopening the complete destination array later.
    """

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
    return digest


def _outer_write_rows(plan: Any, *, shape: tuple[int, ...]) -> int:
    """Choose one complete physical outer chunk/shard along the growth axis."""

    if not shape:
        return 1
    grid = plan.shard_shape if plan.shard_shape is not None else plan.chunk_shape
    if not isinstance(grid, (tuple, list)) or not grid:
        raise ValueError("Subject-shape storage plan lacks an outer write grid.")
    rows = int(grid[0])
    if rows <= 0:
        raise ValueError("Subject-shape outer write rows must be positive.")
    return rows


def _decoded_equal(left: np.ndarray, right: np.ndarray) -> bool:
    equal_nan = left.dtype.kind in {"f", "c"}
    return bool(np.array_equal(left, right, equal_nan=equal_nan))


def _copy_array_by_outer_units(
    source_array: Any,
    destination_array: Any,
    *,
    plan: Any,
    path: str,
) -> tuple[str, list[dict[str, object]]]:
    """Write and read back each complete physical unit exactly once.

    Zarr v3 indexed shards are single physical objects.  Writing inner logical
    chunks separately causes a read-modify-write of the same shard.  This
    routine instead owns one full outer shard per assignment.  The immediate
    decoded readback both validates the write and produces the leaf receipt
    that later publication gates can reuse.
    """

    dtype = np.dtype(source_array.dtype)
    shape = tuple(int(value) for value in source_array.shape)
    if dtype.hasobject:
        raise TypeError(f"{path}: object arrays have no immutable byte contract.")
    if np.dtype(destination_array.dtype) != dtype or tuple(
        int(value) for value in destination_array.shape
    ) != shape:
        raise RuntimeError(f"{path}: destination dtype/shape differs before copy.")

    digest = _array_digest_header(dtype=dtype, shape=shape)
    leaves: list[dict[str, object]] = []
    if not shape:
        values = np.ascontiguousarray(source_array[...])
        destination_array[...] = values
        observed = np.ascontiguousarray(destination_array[...])
        if observed.dtype != dtype or observed.shape != shape:
            raise RuntimeError(f"{path}: scalar readback metadata differs.")
        payload = observed.tobytes(order="C")
        if not _decoded_equal(values, observed):
            raise RuntimeError(f"{path}: scalar decoded readback differs.")
        digest.update(payload)
        leaves.append(
            {
                "path": path,
                "decoded_bytes": len(payload),
                "decoded_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        return digest.hexdigest(), leaves

    step = _outer_write_rows(plan, shape=shape)
    trailing = (slice(None),) * (len(shape) - 1)
    for start in range(0, shape[0], step):
        stop = min(shape[0], start + step)
        selection = (slice(start, stop), *trailing)
        values = np.ascontiguousarray(source_array[selection])
        destination_array[selection] = values
        observed = np.ascontiguousarray(destination_array[selection])
        if (
            observed.dtype != dtype
            or observed.shape != values.shape
            or not _decoded_equal(values, observed)
        ):
            raise RuntimeError(
                f"{path}[{start}:{stop}]: decoded outer-unit readback differs."
            )
        payload = observed.tobytes(order="C")
        digest.update(payload)
        leaves.append(
            {
                "path": path,
                "start_row": start,
                "stop_row": stop,
                "decoded_bytes": len(payload),
                "decoded_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return digest.hexdigest(), leaves


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
    expected_manifest_fields = {
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
    maintained_profile = (
        source_inventory.get("maintained_profile")
        if isinstance(source_inventory, Mapping)
        else None
    )
    logical_profile_id = (
        maintained_profile.get("profile_id")
        if isinstance(maintained_profile, Mapping)
        else None
    )
    bundle_source_binding = source_manifest.get("source_binding")
    if logical_profile_id == CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID:
        expected_manifest_fields.add("source_binding")
        bundle_binding_valid = (
            isinstance(bundle_source_binding, Mapping)
            and set(bundle_source_binding)
            == {
                "source_kind",
                "bundle_id",
                "bundle_active_at_derivation",
                "record_sha256",
            }
            and bundle_source_binding.get("source_kind")
            == SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
            and isinstance(bundle_source_binding.get("bundle_id"), str)
            and bool(bundle_source_binding.get("bundle_id"))
            and type(bundle_source_binding.get("bundle_active_at_derivation"))
            is bool
            and isinstance(bundle_source_binding.get("record_sha256"), str)
            and re.fullmatch(
                r"[0-9a-f]{64}",
                bundle_source_binding["record_sha256"],
            )
            is not None
        )
    else:
        bundle_binding_valid = bundle_source_binding is None
    if (
        source_manifest.get("schema_id")
        != SUBJECT_SHAPE_UNBOUND_MANIFEST_SCHEMA_ID
        or source_manifest.get("schema_version") != 1
        or set(source_manifest) != expected_manifest_fields
        or logical_profile_id
        not in {
            CANONICAL_SUBJECT_SHAPE_PROFILE_ID,
            CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID,
        }
        or not bundle_binding_valid
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


def materialize_subject_shape_access_aware_storage(
    source_run: Any,
    destination_path: str | Path,
    *,
    profile: StorageProfile,
    phase: str = "unbound",
    copy_block_rows: int = 1_024,
    workers: int = 1,
) -> dict[str, object]:
    """Write one complete node-local access-aware run through the shared factory.

    Destination arrays and their metadata are created serially.  Payload copies
    may then run concurrently across arrays; each worker owns every physical
    chunk/shard of exactly one destination array, so no physical Zarr object is
    shared between workers.
    """

    destination = Path(destination_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing existing subject-shape output: {destination}")
    if type(workers) is not int or workers <= 0:
        raise ValueError("Subject-shape storage copy workers must be positive.")
    if (
        source_run.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_UNBOUND_STAGE_STATUS
        or source_run.attrs.get("palette_run_completion_status") != "complete"
        or source_run.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Subject-shape storage conversion requires one producer-sealed, "
            "complete, unbound, selector-ineligible stage."
        )
    raw_manifest = source_run.attrs.get(SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR)
    raw_arrays = raw_manifest.get("arrays") if isinstance(raw_manifest, Mapping) else None
    if not isinstance(raw_arrays, Mapping) or not raw_arrays:
        raise ValueError("Subject-shape producer seal lacks its array inventory.")
    persisted_digests = {
        str(path): str(record.get("content_sha256"))
        for path, record in raw_arrays.items()
        if isinstance(record, Mapping)
    }
    sealed_manifest = load_unbound_subject_shape_manifest(
        source_run,
        array_content_sha256=persisted_digests,
    )
    sealed_arrays = sealed_manifest.record["arrays"]
    receipt = build_subject_shape_storage_receipt(
        source_run,
        phase=phase,
        profile=profile,
    )
    entry_by_path = {entry.declaration.path: entry for entry in receipt.entries}
    destination_run = zarr.open_group(
        str(destination),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    _copy_group_attributes(source_run, destination_run)
    copy_assignments: list[tuple[str, Any, Any, Any]] = []
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
        copy_assignments.append(
            (path, source_array, destination_array, entry.plan)
        )

    effective_workers = min(workers, max(1, len(copy_assignments)))

    def copy_assignment(assignment: tuple[str, Any, Any, Any]):
        path, source_array, destination_array, plan = assignment
        destination_hash, leaves = _copy_array_by_outer_units(
            source_array,
            destination_array,
            plan=plan,
            path=path,
        )
        expected_hash = sealed_arrays[path]["content_sha256"]
        if destination_hash != expected_hash:
            raise RuntimeError(
                f"Decoded candidate differs from the producer seal for {path!r}."
            )
        return {
            "path": path,
            "content_sha256": destination_hash,
            "leaves": leaves,
            "outer_write_rows": _outer_write_rows(
                plan,
                shape=tuple(int(value) for value in source_array.shape),
            ),
        }

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        copy_results = sorted(
            executor.map(copy_assignment, copy_assignments),
            key=lambda value: str(value["path"]),
        )
    hashes = {
        str(result["path"]): str(result["content_sha256"])
        for result in copy_results
    }
    write_leaves = {
        str(result["path"]): list(result["leaves"])
        for result in copy_results
    }
    outer_write_rows = {
        str(result["path"]): int(result["outer_write_rows"])
        for result in copy_results
    }
    source_manifest_link = persist_subject_shape_storage_source_manifest_link(
        destination_run,
        sealed_manifest,
    )
    persist_subject_shape_storage_receipt(
        destination_run,
        receipt,
        phase=phase,
    )
    errors = validate_subject_shape_access_aware_storage(
        destination_run,
        phase=phase,
        expected_profile_id=profile.profile_id,
    )
    if errors:
        raise RuntimeError(
            "Subject-shape access-aware validation failed: " + "; ".join(errors)
        )
    return {
        "schema_id": "palette.subject_shape_storage_materialization",
        "schema_version": 1,
        "phase": phase,
        "array_count": len(hashes),
        "decoded_equality": True,
        "exact_decoded_validation": True,
        "physical_write_policy": (
            "one_complete_nonoverlapping_outer_chunk_or_shard_per_assignment_v1"
        ),
        "requested_copy_block_rows": int(copy_block_rows),
        "requested_copy_workers": workers,
        "effective_copy_workers": effective_workers,
        "parallel_write_ownership": (
            "one_complete_destination_array_and_all_its_physical_objects_per_worker_v1"
        ),
        "effective_write_grid_policy": "physical_outer_grid_overrides_logical_block_rows_v1",
        "outer_write_rows": outer_write_rows,
        "decoded_write_leaves": write_leaves,
        "array_content_sha256": hashes,
        "source_manifest_link": source_manifest_link,
        "storage_plan": receipt.as_manifest(),
    }


def materialize_subject_shape_storage_candidate(
    source_run: Any,
    destination_path: str | Path,
    *,
    phase: str = "unbound",
    copy_block_rows: int = 1_024,
    workers: int = 1,
) -> dict[str, object]:
    """Preserve the explicit unpromoted-candidate materialization API."""

    return materialize_subject_shape_access_aware_storage(
        source_run,
        destination_path,
        profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_V1,
        phase=phase,
        copy_block_rows=copy_block_rows,
        workers=workers,
    )


def _scan_subject_shape_array_once(
    path: str,
    array: Any,
    *,
    row_count: int,
    block_rows: int | None,
) -> dict[str, object]:
    """Read one array once into canonical digest and decoded leaf evidence."""

    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    if dtype.hasobject:
        raise TypeError(f"{path}: object arrays have no payload receipt grammar.")
    digest = _array_digest_header(dtype=dtype, shape=shape)
    leaves: list[dict[str, object]] = []
    decoded_bytes = 0
    row_aligned = bool(shape and shape[0] == row_count)
    if row_aligned:
        if block_rows is None:
            try:
                metadata = array.metadata.to_dict()
                outer_shape = metadata["chunk_grid"]["configuration"][
                    "chunk_shape"
                ]
                step = int(outer_shape[0])
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                IndexError,
            ) as exc:
                raise ValueError(
                    f"{path}: physical outer grid is unavailable: {exc}."
                ) from exc
        else:
            step = block_rows
        if step <= 0:
            raise ValueError(f"{path}: decoded scan row step is invalid.")
        trailing = (slice(None),) * (len(shape) - 1)
        for start in range(0, shape[0], step):
            stop = min(shape[0], start + step)
            values = np.ascontiguousarray(
                array[(slice(start, stop), *trailing)]
            )
            if values.dtype != dtype or values.shape != (stop - start, *shape[1:]):
                raise RuntimeError(f"{path}: decoded scan metadata changed.")
            payload = values.tobytes(order="C")
            digest.update(payload)
            decoded_bytes += len(payload)
            leaves.append(
                {
                    "path": path,
                    "start_row": start,
                    "stop_row": stop,
                    "decoded_bytes": len(payload),
                    "decoded_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    else:
        values = np.ascontiguousarray(array[...])
        if values.dtype != dtype or values.shape != shape:
            raise RuntimeError(f"{path}: decoded static scan metadata changed.")
        payload = values.tobytes(order="C")
        digest.update(payload)
        decoded_bytes = len(payload)
        leaves.append(
            {
                "path": path,
                "decoded_bytes": len(payload),
                "decoded_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if np.dtype(array.dtype) != dtype or tuple(int(value) for value in array.shape) != shape:
        raise RuntimeError(f"{path}: array metadata changed during payload scan.")
    return {
        "path": path,
        "plan": {"path": path, "shape": list(shape), "dtype": str(dtype)},
        "row_aligned": row_aligned,
        "leaves": leaves,
        "decoded_bytes": decoded_bytes,
        "content_sha256": digest.hexdigest(),
    }


def build_subject_shape_unbound_payload_scan_receipt(
    run_group: Any,
    *,
    workers: int = 1,
    block_rows: int = 1_024,
) -> dict[str, object]:
    """Scan an exclusive local unbound payload once into reusable evidence.

    The caller owns the node-local scratch artifact exclusively and has closed
    every scientific write.  Array reads are parallelized across the closed
    inventory, while each individual array is traversed in canonical row-major
    order so its digest is byte-identical to ``array_payload_sha256``.  The
    later storage materialization re-reads every value and checks these direct
    digests, closing the boundary before authoritative publication.
    """

    if type(workers) is not int or workers <= 0:
        raise ValueError("Subject-shape payload scan workers must be positive.")
    if type(block_rows) is not int or block_rows <= 0:
        raise ValueError("Subject-shape payload scan block_rows must be positive.")
    if (
        run_group.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_UNBOUND_STAGE_STATUS
        or run_group.attrs.get("palette_run_completion_status") != "complete"
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Subject-shape unbound payload scan requires one complete, "
            "selector-ineligible unbound stage."
        )
    arrays = list(_iter_arrays(run_group))
    row_count = _row_count(run_group)
    effective_workers = min(workers, max(1, len(arrays)))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                _scan_subject_shape_array_once,
                path,
                array,
                row_count=row_count,
                block_rows=block_rows,
            )
            for path, array in arrays
        ]
        results = sorted(
            (future.result() for future in futures),
            key=lambda value: str(value["path"]),
        )
    plans = [dict(result["plan"]) for result in results]
    shard_leaves = [
        dict(leaf)
        for result in results
        if result["row_aligned"] is True
        for leaf in result["leaves"]
    ]
    static_leaves = [
        dict(leaf)
        for result in results
        if result["row_aligned"] is False
        for leaf in result["leaves"]
    ]
    content_sha256 = {
        str(result["path"]): str(result["content_sha256"])
        for result in results
    }
    decoded_bytes = sum(int(result["decoded_bytes"]) for result in results)
    copy_report = {
        "schema_id": "palette.zarr_sharded_run_copy.v1",
        "status": "complete",
        "source_run": f"/{str(run_group.path).strip('/')}",
        "destination_run": f"/{str(run_group.path).strip('/')}",
        "array_count": len(plans),
        "arrays": plans,
        "shards": shard_leaves,
        "static_arrays": static_leaves,
        "decoded_bytes_copied": decoded_bytes,
        "exact_decoded_validation": True,
        "receipt_origin": "single_exclusive_post_compute_decoded_scan_v1",
    }
    decoded_payload = decoded_payload_receipt_from_copy_report(copy_report)
    return {
        "schema_id": "palette.subject_shape_unbound_payload_scan_receipt",
        "schema_version": 1,
        "run_ref": f"/{str(run_group.path).strip('/')}",
        "array_payload_canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
        "closed_array_inventory": True,
        "mutation_exclusion": "exclusive_node_local_writer_v1",
        "requested_workers": workers,
        "effective_workers": effective_workers,
        "block_rows": block_rows,
        "duration_seconds": float(time.perf_counter() - started),
        "array_content_sha256": content_sha256,
        "decoded_payload": decoded_payload,
        "decoded_copy_report": copy_report,
    }


def build_subject_shape_bound_payload_scan_receipt(
    run_group: Any,
    *,
    workers: int = 1,
) -> dict[str, object]:
    """Scan one final bound payload once into reusable decoded digest evidence.

    The caller owns the archive publication lock and must have completed every
    array mutation.  Blocks follow the live physical outer grid, bounding
    memory while ensuring that the receipt describes exactly one immutable
    publication generation.
    """

    if (
        run_group.attrs.get("palette_run_completion_status") != "running"
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Subject-shape payload scan requires one running, ineligible child."
        )
    if type(workers) is not int or workers <= 0:
        raise ValueError("Subject-shape bound payload scan workers must be positive.")
    row_count = _row_count(run_group)
    arrays = list(_iter_arrays(run_group))
    effective_workers = min(workers, max(1, len(arrays)))
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                _scan_subject_shape_array_once,
                path,
                array,
                row_count=row_count,
                block_rows=None,
            )
            for path, array in arrays
        ]
        results = sorted(
            (future.result() for future in futures),
            key=lambda value: str(value["path"]),
        )
    plans = [dict(result["plan"]) for result in results]
    shard_leaves = [
        dict(leaf)
        for result in results
        if result["row_aligned"] is True
        for leaf in result["leaves"]
    ]
    static_leaves = [
        dict(leaf)
        for result in results
        if result["row_aligned"] is False
        for leaf in result["leaves"]
    ]
    content_sha256 = {
        str(result["path"]): str(result["content_sha256"])
        for result in results
    }
    decoded_bytes = sum(int(result["decoded_bytes"]) for result in results)

    copy_report = {
        "schema_id": "palette.zarr_sharded_run_copy.v1",
        "status": "complete",
        "source_run": f"/{str(run_group.path).strip('/')}",
        "destination_run": f"/{str(run_group.path).strip('/')}",
        "array_count": len(plans),
        "arrays": plans,
        "shards": shard_leaves,
        "static_arrays": static_leaves,
        "decoded_bytes_copied": decoded_bytes,
        "exact_decoded_validation": True,
        "receipt_origin": "single_locked_post_binding_decoded_scan_v1",
    }
    return {
        "schema_id": "palette.subject_shape_bound_payload_scan_receipt",
        "schema_version": 1,
        "run_ref": f"/{str(run_group.path).strip('/')}",
        "array_payload_canonicalization": ARRAY_PAYLOAD_CANONICALIZATION,
        "closed_array_inventory": True,
        "requested_workers": workers,
        "effective_workers": effective_workers,
        "array_content_sha256": content_sha256,
        "decoded_copy_report": copy_report,
    }


def create_bound_subject_shape_access_aware_array(
    run_group: Any,
    group: Any,
    *,
    name: str,
    values: np.ndarray,
) -> Any:
    """Create one final-binding array with the declared access-aware profile."""

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
        logical_profile_id=_logical_profile_id(run_group),
    )
    facts = AnalysisArrayStorageFacts(
        path=relative_path,
        shape=tuple(int(value) for value in data.shape),
        dtype=data.dtype,
        access_unit_semantics=(
            "one complete subject-shape row with all fixed trailing semantic axes indivisible"
        ),
    )
    profile_id = run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR)
    profile = subject_shape_access_aware_storage_profile(str(profile_id))
    receipt = plan_analysis_storage(
        (declaration,),
        {relative_path: facts},
        profile=profile,
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


def create_bound_subject_shape_candidate_array(
    run_group: Any,
    group: Any,
    *,
    name: str,
    values: np.ndarray,
) -> Any:
    """Create a binding array for the explicit unpromoted candidate profile."""

    profile_id = str(
        run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR)
    )
    if profile_id != SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID:
        raise ValueError(
            "Candidate array creation requires the candidate storage profile."
        )
    return create_bound_subject_shape_access_aware_array(
        run_group,
        group,
        name=name,
        values=values,
    )


def finalize_bound_subject_shape_storage_receipt(run_group: Any) -> dict[str, object]:
    """Replace the unbound receipt with the exact final bound inventory."""

    profile = subject_shape_access_aware_storage_profile(
        str(run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR))
    )
    receipt = build_subject_shape_storage_receipt(
        run_group,
        phase="bound",
        profile=profile,
    )
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


def validate_subject_shape_access_aware_storage(
    run_group: Any,
    *,
    phase: str,
    expected_profile_id: str,
) -> tuple[str, ...]:
    """Replan and validate every access-aware array declaration fail closed."""

    errors: list[str] = []
    try:
        expected_profile = subject_shape_access_aware_storage_profile(
            expected_profile_id
        )
    except ValueError as exc:
        return (str(exc),)
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
    if parsed.profile != expected_profile:
        errors.append("subject-shape storage profile differs from the expected profile")
    try:
        expected = build_subject_shape_storage_receipt(
            run_group,
            phase=phase,
            profile=expected_profile,
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
        != expected_profile.profile_id
        or run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ROLE_ATTR)
        != _profile_role(expected_profile.profile_id)
    ):
        errors.append("subject-shape access-aware profile identity/role mismatch")
    candidate = run_group.attrs.get(SUBJECT_SHAPE_STORAGE_CANDIDATE_ATTR)
    if is_subject_shape_storage_candidate(expected_profile.profile_id):
        expected_candidate = {
            "schema_id": "palette.subject_shape_storage_candidate",
            "schema_version": 1,
            "profile_id": SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            "logical_profile_id": _logical_profile_id(run_group),
            "phase": phase,
            "selector_eligible": False,
            "promotion_status": "unpromoted_candidate",
        }
        if candidate != expected_candidate:
            errors.append("subject-shape candidate envelope is not exact")
    elif candidate is not None:
        errors.append("supported subject-shape storage carries a candidate envelope")
    entry_by_path = {entry.declaration.path: entry for entry in expected.entries}
    arrays = dict(_iter_arrays(run_group))
    if set(arrays) != set(entry_by_path):
        errors.append("subject-shape access-aware array inventory differs from receipt")
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


def validate_subject_shape_candidate_storage(
    run_group: Any,
    *,
    phase: str,
) -> tuple[str, ...]:
    """Validate the explicit unpromoted-candidate profile."""

    return validate_subject_shape_access_aware_storage(
        run_group,
        phase=phase,
        expected_profile_id=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    )


def validate_subject_shape_direct_consolidated_storage(
    archive_path: str | Path,
    *,
    run_path: str,
    phase: str = "bound",
    expected_profile_id: str | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    archive = Path(archive_path).expanduser().resolve()
    try:
        direct_root = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=False
        )
        consolidated_root = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=True
        )
        direct_run = _array_at_path(direct_root, run_path)
        consolidated_run = _array_at_path(consolidated_root, run_path)
    except Exception as exc:
        return (f"subject-shape metadata views cannot be opened: {exc}",)
    profile_id = (
        str(expected_profile_id)
        if expected_profile_id is not None
        else str(direct_run.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR))
    )
    errors.extend(
        validate_subject_shape_access_aware_storage(
            direct_run,
            phase=phase,
            expected_profile_id=profile_id,
        )
    )
    errors.extend(
        validate_subject_shape_access_aware_storage(
            consolidated_run,
            phase=phase,
            expected_profile_id=profile_id,
        )
    )
    try:
        receipt = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
    except Exception as exc:
        errors.append(str(exc))
    else:
        visibility = direct_run.attrs.get(
            SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR
        )
        if not isinstance(visibility, Mapping):
            errors.append("subject-shape metadata visibility policy is missing")
            visibility = {}
        if visibility.get("profile_id") != profile_id:
            errors.append(
                "subject-shape metadata visibility profile differs from storage profile"
            )
        for field in (
            "direct_consolidated_run_attrs_required",
            "direct_consolidated_group_declarations_required",
            "direct_consolidated_array_declarations_required",
            "direct_consolidated_exact_node_inventory_required",
            "root_consolidation_is_final_visibility_step",
        ):
            if visibility.get(field) is not True:
                errors.append(
                    f"subject-shape metadata visibility policy requires {field!r}"
                )
        expected_array_count = (
            visibility.get("expected_array_count")
            if isinstance(visibility, Mapping)
            else None
        )
        if (
            type(expected_array_count) is not int
            or expected_array_count < 0
            or receipt.array_count != expected_array_count
        ):
            errors.append(
                "subject-shape persisted metadata array count differs from policy"
            )
    return tuple(errors)


def set_subject_shape_metadata_visibility_policy(
    run_group: Any,
    *,
    expected_array_count: int,
) -> None:
    profile_id = str(
        run_group.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR)
    )
    subject_shape_access_aware_storage_profile(profile_id)
    run_group.attrs[SUBJECT_SHAPE_STORAGE_METADATA_POLICY_ATTR] = {
        "schema_id": "palette.subject_shape_metadata_visibility_policy",
        "schema_version": 1,
        "profile_id": profile_id,
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
    "SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID",
    "SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_V1",
    "SUBJECT_SHAPE_LEGACY_EXPLICIT_STORAGE",
    "SUBJECT_SHAPE_STORAGE_PROFILE_CHOICES",
    "build_subject_shape_bound_payload_scan_receipt",
    "build_subject_shape_unbound_payload_scan_receipt",
    "build_subject_shape_storage_receipt",
    "create_bound_subject_shape_access_aware_array",
    "create_bound_subject_shape_candidate_array",
    "finalize_bound_subject_shape_storage_receipt",
    "is_subject_shape_access_aware_storage",
    "is_subject_shape_storage_candidate",
    "materialize_subject_shape_access_aware_storage",
    "materialize_subject_shape_storage_candidate",
    "persist_subject_shape_storage_receipt",
    "persist_subject_shape_storage_source_manifest_link",
    "set_subject_shape_metadata_visibility_policy",
    "subject_shape_fill_value",
    "subject_shape_access_aware_storage_profile",
    "validate_subject_shape_access_aware_storage",
    "validate_subject_shape_candidate_storage",
    "validate_subject_shape_direct_consolidated_storage",
    "validate_subject_shape_storage_source_manifest_link",
]
