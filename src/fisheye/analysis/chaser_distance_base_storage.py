"""Byte-planned storage for the exact sealed chaser-distance base projection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fisheye.analysis.chaser_distance_base_schema import (
    CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID,
    CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION,
    SEALED_CHASER_DISTANCE_BASE_PATHS,
    build_chaser_distance_base_declarations,
    validate_chaser_distance_base_semantics,
)
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    build_exact_tabular_storage_receipt,
    persist_exact_tabular_storage_receipt,
    validate_exact_tabular_storage_receipt,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.analysis_array_contracts import AnalysisArrayDeclaration
from fisheye.shared.zarr.analysis_storage_planning import AnalysisStoragePlanReceipt
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
    create_array_from_plan,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import StorageProfile
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


BASE_MANIFEST_ATTR = "chaser_distance_sealed_base_candidate_manifest"
BASE_MANIFEST_DIGEST_ATTR = "chaser_distance_sealed_base_candidate_manifest_sha256"
BASE_LOGICAL_HASHES_ATTR = "chaser_distance_sealed_base_candidate_logical_hashes"
BASE_SOURCE_BINDING_ATTR = "chaser_distance_sealed_base_source_binding"
BASE_MANIFEST_SCHEMA_ID = "palette.chaser_distance.sealed_base_candidate_manifest"
BASE_SOURCE_BINDING_SCHEMA_ID = "palette.chaser_distance.sealed_base_source_binding"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PARENT = ("analysis", "chaser_distance_runs")
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


def _array(group: Any, path: str) -> Any:
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
    for name, _node in sorted(group.arrays(), key=lambda item: str(item[0])):
        yield f"{prefix}/{name}" if prefix else str(name)
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_array_paths(child, child_prefix)


def _iter_group_paths(group: Any, prefix: str = ""):
    for name, child in sorted(group.groups(), key=lambda item: str(item[0])):
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        yield child_prefix
        yield from _iter_group_paths(child, child_prefix)


def _canonical_source_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    components = value.split("/")
    return (
        tuple(components[:2]) == _SOURCE_PARENT
        and len(components) == 3
        and components[-1] not in {"", ".", "..", "latest", "latest_complete"}
        and components[-1] == components[-1].strip()
        and "/" not in components[-1]
        and "\\" not in components[-1]
        and not any(character.isspace() for character in components[-1])
    )


def build_source_authority_binding(
    bound: Any,
    *,
    source_group: Any,
) -> dict[str, Any]:
    """Bind the exact source records that transitively protect all 30 arrays."""

    path = str(bound.run_path).strip().strip("/")
    if not _canonical_source_path(path):
        raise ValueError("Source is not one explicit canonical chaser-distance run path.")
    if canonical_node_path(source_group) != path:
        raise ValueError("Verified source group path differs from the bound run path.")
    source_attrs = source_group.attrs
    if (
        source_attrs.get("schema_id") != "palette.chaser_distance.v1"
        or source_attrs.get("schema_version") != 1
        or source_attrs.get("coordinate_publication_status")
        != "sealed_canonical_v2"
        or source_attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or source_attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError(
            "Source group is not an exact complete eligible sealed canonical "
            "palette.chaser_distance.v1 run."
        )
    protected = set(bound.publication_seal.record.get("protected_arrays", {}))
    epoch = {
        f"epoch_summary/{name}"
        for name in bound.epoch_window_identity.record.get("published_arrays", {})
    }
    measurement = set(
        bound.surface_manifest.record.get("measurement_surfaces", {})
    )
    coordinate = set(bound.surface_manifest.record.get("coordinate_surfaces", {}))
    covered = protected | epoch | measurement | coordinate
    expected = set(SEALED_CHASER_DISTANCE_BASE_PATHS)
    if covered != expected:
        raise ValueError(
            "Canonical source authorities do not cover the exact sealed base inventory: "
            f"missing={sorted(expected - covered)!r}, "
            f"unexpected={sorted(covered - expected)!r}."
        )

    def pointer(record: Any) -> dict[str, str]:
        value = {
            "record_ref": str(record.record_ref),
            "record_sha256": str(record.record_sha256),
        }
        if not _SHA256.fullmatch(value["record_sha256"]):
            raise ValueError("Source authority record has a noncanonical SHA-256.")
        return value

    return {
        "schema_id": BASE_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": 1,
        "source_run_path": path,
        "source_schema": {
            "schema_id": "palette.chaser_distance.v1",
            "schema_version": 1,
            "coordinate_publication": "sealed_canonical_v2",
        },
        "publication_seal": pointer(bound.publication_seal),
        "surface_manifest": pointer(bound.surface_manifest),
        "row_identity": pointer(bound.row_identity),
        "input_authority": pointer(bound.input_authority),
        "measurement_authority": pointer(bound.measurement_authority),
        "chaser_collection": pointer(bound.chaser_collection),
        "epoch_window_identity": pointer(bound.epoch_window_identity),
        "coverage_policy": (
            "exact_union_publication_seal_epoch_authority_surface_manifest_v1"
        ),
        "covered_array_paths": sorted(covered),
    }


def _validate_source_binding(value: Any) -> tuple[str, ...]:
    errors: list[str] = []
    expected_fields = {
        "schema_id",
        "schema_version",
        "source_run_path",
        "source_schema",
        "publication_seal",
        "surface_manifest",
        "row_identity",
        "input_authority",
        "measurement_authority",
        "chaser_collection",
        "epoch_window_identity",
        "coverage_policy",
        "covered_array_paths",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        return ("candidate source binding has an unexpected field set",)
    path = value.get("source_run_path")
    if not _canonical_source_path(path):
        errors.append("candidate source binding is not one canonical source run")
        path = ""
    if value.get("schema_id") != BASE_SOURCE_BINDING_SCHEMA_ID or value.get(
        "schema_version"
    ) != 1:
        errors.append("candidate source-binding schema identity mismatch")
    if value.get("source_schema") != {
        "schema_id": "palette.chaser_distance.v1",
        "schema_version": 1,
        "coordinate_publication": "sealed_canonical_v2",
    }:
        errors.append("candidate source schema binding is not exact v1")
    if value.get("coverage_policy") != (
        "exact_union_publication_seal_epoch_authority_surface_manifest_v1"
    ):
        errors.append("candidate source coverage policy mismatch")
    if value.get("covered_array_paths") != list(SEALED_CHASER_DISTANCE_BASE_PATHS):
        errors.append("candidate source coverage inventory mismatch")
    expected_refs = {
        "publication_seal": f"/{path}@chaser_distance_publication_seal",
        "surface_manifest": f"/{path}@chaser_distance_surface_manifest",
        "row_identity": f"/{path}@row_identity_contract",
        "input_authority": f"/{path}@chaser_distance_input_authority",
        "measurement_authority": f"/{path}@chaser_distance_measurement_authority",
        "chaser_collection": (
            f"/{path}/chasers@chaser_collection_authority"
        ),
        "epoch_window_identity": (
            f"/{path}/epoch_summary@epoch_window_identity_authority"
        ),
    }
    for field in (
        "publication_seal",
        "surface_manifest",
        "row_identity",
        "input_authority",
        "measurement_authority",
        "chaser_collection",
        "epoch_window_identity",
    ):
        pointer = value.get(field)
        if not isinstance(pointer, Mapping) or set(pointer) != {
            "record_ref",
            "record_sha256",
        }:
            errors.append(f"candidate source {field} pointer is not exact")
            continue
        if (
            pointer.get("record_ref") != expected_refs[field]
            or not _SHA256.fullmatch(str(pointer.get("record_sha256", "")))
        ):
            errors.append(f"candidate source {field} pointer is noncanonical")
    return tuple(errors)


def _validate_source_group_identity(
    source_group: Any,
    source_binding: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_path = source_binding.get("source_run_path")
    if canonical_node_path(source_group) != expected_path:
        errors.append("sealed source group path differs from candidate source binding")
    attrs = source_group.attrs
    if attrs.get("schema_id") != "palette.chaser_distance.v1":
        errors.append("sealed source schema_id is not exact chaser-distance v1")
    if attrs.get("schema_version") != 1:
        errors.append("sealed source schema_version is not exact 1")
    if attrs.get("coordinate_publication_status") != "sealed_canonical_v2":
        errors.append("sealed source coordinate publication is not canonical v2")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("sealed source is not complete")
    if attrs.get("stage_selector_eligible") is not True:
        errors.append("sealed source is not explicitly selector eligible")
    return tuple(errors)


def build_base_storage_receipt(
    source_group: Any,
    *,
    profile: StorageProfile,
) -> AnalysisStoragePlanReceipt:
    errors = validate_chaser_distance_base_semantics(source_group)
    if errors:
        raise ValueError(f"Invalid sealed chaser-distance base source: {errors!r}.")
    return build_exact_tabular_storage_receipt(
        source_group,
        declarations=build_chaser_distance_base_declarations(source_group),
        profile=profile,
    )


def _fill_value(declaration: AnalysisArrayDeclaration) -> object:
    path = declaration.path
    if path in {
        "frames/stimulus_frame_num",
        "frames/timestamp_ns",
        "frames/stimulus_epoch_window_id",
        "positions/source_detection_row_index",
        "distances/nearest_chaser_index",
    }:
        return -1
    dtype = np.dtype(declaration.contract.dtype.numpy_dtype)
    if dtype.kind == "f":
        return 0.0 if path == "epoch_distributions/hist_density" else float("nan")
    if dtype.kind == "b":
        return False
    return 0


def _write_by_units(destination: Any, source: Any, entry: Any) -> None:
    if int(source.ndim) == 0:
        destination[...] = np.asarray(source[...])
        return
    plan = entry.plan
    extent = int((plan.shard_shape or plan.chunk_shape or source.shape)[0])
    trailing = (slice(None),) * (int(source.ndim) - 1)
    for start in range(0, int(source.shape[0]), max(1, extent)):
        stop = min(start + max(1, extent), int(source.shape[0]))
        index = (slice(start, stop), *trailing)
        destination[index] = np.asarray(source[index])


def rematerialize_base_candidate(
    source_group: Any,
    destination_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
) -> None:
    if list(destination_group.array_keys()) or list(destination_group.group_keys()):
        raise ValueError("Chaser-distance candidate destination must be empty.")
    declarations = build_chaser_distance_base_declarations(source_group)
    entries = {entry.declaration.path: entry for entry in receipt.entries}
    if set(entries) != {item.path for item in declarations}:
        raise ValueError("Storage receipt differs from sealed source declarations.")
    for declaration in declarations:
        source = _array(source_group, declaration.path)
        entry = entries[declaration.path]
        parent, leaf = _parent_and_leaf(destination_group, declaration.path)
        destination = create_array_from_plan(
            parent,
            name=leaf,
            contract=declaration.contract,
            plan=entry.plan,
            fill_value=_fill_value(declaration),
            attributes={
                "authority_projection": "sealed_chaser_distance_base_v1",
                "source_relative_path": declaration.path,
            },
        )
        _write_by_units(destination, source, entry)


def _array_digest(array: Any) -> str:
    dtype = np.dtype(array.dtype)
    digest = hashlib.sha256()
    digest.update(str(dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    if int(array.ndim) == 0:
        digest.update(np.ascontiguousarray(np.asarray(array[...])).tobytes(order="C"))
        return digest.hexdigest()
    rows = max(1, min(int(array.shape[0]) or 1, 65_536))
    trailing = (slice(None),) * (int(array.ndim) - 1)
    for start in range(0, int(array.shape[0]), rows):
        index = (slice(start, min(start + rows, int(array.shape[0]))), *trailing)
        digest.update(np.ascontiguousarray(np.asarray(array[index])).tobytes(order="C"))
    return digest.hexdigest()


def base_logical_hashes(
    group: Any,
    declarations: Sequence[AnalysisArrayDeclaration] | None = None,
) -> dict[str, str]:
    declarations = declarations or build_chaser_distance_base_declarations(group)
    return {
        declaration.path: _array_digest(_array(group, declaration.path))
        for declaration in declarations
    }


def build_base_candidate_manifest(
    *,
    declarations: Sequence[AnalysisArrayDeclaration],
    source_binding: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    candidate_hashes: Mapping[str, str],
    storage_receipt: AnalysisStoragePlanReceipt,
) -> dict[str, Any]:
    paths = [declaration.path for declaration in declarations]
    payload = {
        "schema_id": CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID,
        "schema_version": CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION,
        "status": "complete_selector_ineligible_unpromoted",
        "source_binding": dict(source_binding),
        "authority_boundary": {
            "included": "exact arrays transitively sealed by canonical base authorities",
            "excluded": [
                "unsealed_protocol_behavior_and_role_intervals",
                "unsealed_raw_counts_and_threshold_fraction",
                "unsealed_visualizations_and_derived_components",
            ],
        },
        "arrays": [declaration.as_manifest() for declaration in declarations],
        "array_paths": paths,
        "source_logical_hashes": dict(sorted(source_hashes.items())),
        "candidate_logical_hashes": dict(sorted(candidate_hashes.items())),
        "storage_plan_payload_sha256": storage_receipt.as_manifest()["payload_digest"],
        "publication_state": {
            "stage_selector_eligible": False,
            "storage_candidate_profile_promoted": False,
            "selector_mutation_permitted": False,
        },
    }
    return {
        "schema_id": BASE_MANIFEST_SCHEMA_ID,
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def persist_base_candidate_contract(
    run_group: Any,
    *,
    receipt: AnalysisStoragePlanReceipt,
    declarations: Sequence[AnalysisArrayDeclaration],
    source_binding: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    candidate_hashes = base_logical_hashes(run_group, declarations)
    if dict(source_hashes) != candidate_hashes:
        raise ValueError("Candidate decoded values differ from the sealed source projection.")
    manifest = build_base_candidate_manifest(
        declarations=declarations,
        source_binding=source_binding,
        source_hashes=source_hashes,
        candidate_hashes=candidate_hashes,
        storage_receipt=receipt,
    )
    run_group.attrs[BASE_MANIFEST_ATTR] = json_attr_safe(manifest)
    run_group.attrs[BASE_MANIFEST_DIGEST_ATTR] = manifest["payload_digest"]
    run_group.attrs[BASE_LOGICAL_HASHES_ATTR] = json_attr_safe(candidate_hashes)
    run_group.attrs[BASE_SOURCE_BINDING_ATTR] = json_attr_safe(dict(source_binding))
    run_group.attrs["storage_candidate_source_run_path"] = source_binding[
        "source_run_path"
    ]
    run_group.attrs["storage_candidate_source_run"] = str(
        source_binding["source_run_path"]
    ).rsplit("/", 1)[-1]
    run_group.attrs["storage_candidate_profile_promoted"] = False
    persist_exact_tabular_storage_receipt(
        run_group,
        receipt,
    )
    return manifest


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


def validate_base_storage_receipt(
    run_group: Any,
    declarations: Sequence[AnalysisArrayDeclaration],
) -> tuple[str, ...]:
    """Validate receipt plus the base contract's path-specific physical fills."""

    errors = [
        error
        for error in validate_exact_tabular_storage_receipt(
            run_group, declarations=declarations
        )
        if "array metadata differs from resolved chunks/shards/codecs" not in error
    ]
    persisted = run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(persisted, Mapping):
        return tuple(errors)
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(persisted)
        expected = build_exact_tabular_storage_receipt(
            run_group,
            declarations=declarations,
            profile=parsed.profile,
        )
        entries = {entry.declaration.path: entry for entry in expected.entries}
        for declaration in declarations:
            array = _array(run_group, declaration.path)
            raw = array.metadata.to_dict()
            attributes = raw.get("attributes")
            if not isinstance(attributes, Mapping):
                raise ValueError(f"{declaration.path}: attributes are not an object")
            nonreserved = {
                str(key): value
                for key, value in attributes.items()
                if key not in _RESERVED_ARRAY_ATTRIBUTES
            }
            expected_metadata = array_metadata_declaration_from_plan(
                contract=declaration.contract,
                plan=entries[declaration.path].plan,
                fill_value=_fill_value(declaration),
                attributes=nonreserved,
            )
            observed_metadata = {
                key: value
                for key, value in raw.items()
                if key not in {"zarr_format", "node_type", "consolidated_metadata"}
            }
            if _normalized_metadata(observed_metadata) != _normalized_metadata(
                expected_metadata
            ):
                errors.append(
                    f"{declaration.path}: array metadata differs from the exact "
                    "path-specific chunks/shards/codecs/fill contract"
                )
    except Exception as exc:
        errors.append(f"base storage metadata could not be reconstructed: {exc}")
    return tuple(errors)


def validate_base_candidate(
    run_group: Any,
    *,
    source_group: Any | None = None,
    expected_source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        declarations = build_chaser_distance_base_declarations(run_group)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "array_count": 0}
    paths = {item.path for item in declarations}
    observed = set(_iter_array_paths(run_group))
    if observed != paths:
        errors.append(
            "candidate array inventory differs "
            f"(missing={sorted(paths-observed)!r}, unexpected={sorted(observed-paths)!r})"
        )
    expected_groups = {
        "/".join(path.split("/")[:end])
        for path in paths
        for end in range(1, len(path.split("/")))
    }
    observed_groups = set(_iter_group_paths(run_group))
    if observed_groups != expected_groups:
        errors.append(
            "candidate group inventory differs "
            f"(missing={sorted(expected_groups-observed_groups)!r}, "
            f"unexpected={sorted(observed_groups-expected_groups)!r})"
        )
    errors.extend(validate_chaser_distance_base_semantics(run_group))
    if run_group.attrs.get("schema_id") != CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_ID:
        errors.append("candidate schema_id mismatch")
    if run_group.attrs.get("schema_version") != CHASER_DISTANCE_BASE_CANDIDATE_SCHEMA_VERSION:
        errors.append("candidate schema_version mismatch")
    if run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("candidate is not complete")
    if run_group.attrs.get("stage_selector_eligible") is not False:
        errors.append("candidate is not selector-ineligible")
    if run_group.attrs.get("storage_candidate_profile_promoted") is not False:
        errors.append("candidate profile is not explicitly unpromoted")
    errors.extend(validate_base_storage_receipt(run_group, declarations))
    persisted_receipt = run_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    parsed_receipt: AnalysisStoragePlanReceipt | None = None
    if isinstance(persisted_receipt, Mapping):
        try:
            parsed_receipt = analysis_storage_plan_receipt_from_manifest(
                persisted_receipt
            )
            if parsed_receipt.profile.profile_id != "published_http_v1":
                errors.append("candidate storage profile is not published_http_v1")
        except Exception as exc:
            errors.append(f"candidate storage receipt is not executable: {exc}")

    current_hashes = base_logical_hashes(run_group, declarations)
    persisted_hashes = run_group.attrs.get(BASE_LOGICAL_HASHES_ATTR)
    if not isinstance(persisted_hashes, Mapping) or dict(persisted_hashes) != current_hashes:
        errors.append("persisted candidate logical hashes differ from decoded arrays")
    manifest = run_group.attrs.get(BASE_MANIFEST_ATTR)
    source_binding = run_group.attrs.get(BASE_SOURCE_BINDING_ATTR)
    errors.extend(_validate_source_binding(source_binding))
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id", "schema_version", "payload", "payload_digest"
    }:
        errors.append("candidate manifest is absent or has unexpected fields")
    else:
        payload = manifest.get("payload")
        expected_fields = {
            "schema_id", "schema_version", "status", "source_binding",
            "authority_boundary", "arrays", "array_paths", "source_logical_hashes",
            "candidate_logical_hashes", "storage_plan_payload_sha256",
            "publication_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            errors.append("candidate manifest payload has unexpected fields")
        elif manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("candidate manifest digest mismatch")
        else:
            if manifest.get("schema_id") != BASE_MANIFEST_SCHEMA_ID or manifest.get("schema_version") != 1:
                errors.append("candidate manifest schema identity mismatch")
            if run_group.attrs.get(BASE_MANIFEST_DIGEST_ATTR) != manifest.get("payload_digest"):
                errors.append("candidate redundant manifest digest mismatch")
            if payload.get("arrays") != [item.as_manifest() for item in declarations]:
                errors.append("candidate declaration manifest differs from executable schema")
            if payload.get("array_paths") != [item.path for item in declarations]:
                errors.append("candidate manifest array-path inventory differs")
            if payload.get("candidate_logical_hashes") != current_hashes:
                errors.append("candidate manifest hashes differ from decoded arrays")
            if payload.get("source_binding") != source_binding:
                errors.append("candidate source binding has conflicting copies")
            state = payload.get("publication_state")
            if state != {
                "stage_selector_eligible": False,
                "storage_candidate_profile_promoted": False,
                "selector_mutation_permitted": False,
            }:
                errors.append("candidate publication state is not exactly nonpromoting")
            if not isinstance(persisted_receipt, Mapping) or payload.get(
                "storage_plan_payload_sha256"
            ) != persisted_receipt.get("payload_digest"):
                errors.append("candidate manifest storage-plan binding mismatch")
            source_hash_payload = payload.get("source_logical_hashes")
            if (
                not isinstance(source_hash_payload, Mapping)
                or set(source_hash_payload) != paths
                or any(
                    not _SHA256.fullmatch(str(value))
                    for value in source_hash_payload.values()
                )
            ):
                errors.append("candidate manifest source hashes are not exact")
            elif parsed_receipt is not None and isinstance(source_binding, Mapping):
                expected_manifest = build_base_candidate_manifest(
                    declarations=declarations,
                    source_binding=source_binding,
                    source_hashes=source_hash_payload,
                    candidate_hashes=current_hashes,
                    storage_receipt=parsed_receipt,
                )
                if dict(manifest) != expected_manifest:
                    errors.append(
                        "candidate manifest differs from the complete executable contract"
                    )
    if not isinstance(source_binding, Mapping) or not _canonical_source_path(
        source_binding.get("source_run_path") if isinstance(source_binding, Mapping) else None
    ):
        errors.append("candidate source binding is not one canonical source run")
    else:
        path = source_binding["source_run_path"]
        if (
            run_group.attrs.get("storage_candidate_source_run_path") != path
            or run_group.attrs.get("storage_candidate_source_run") != path.rsplit("/", 1)[-1]
        ):
            errors.append("candidate redundant source-run binding mismatch")
    if expected_source_binding is None:
        errors.append("external verified source authority binding is required")
    elif dict(source_binding or {}) != dict(expected_source_binding):
        errors.append("candidate source authority binding differs from verified source")
    if source_group is None:
        errors.append("external sealed source group is required")
    else:
        try:
            if isinstance(source_binding, Mapping):
                errors.extend(
                    _validate_source_group_identity(source_group, source_binding)
                )
            source_declarations = build_chaser_distance_base_declarations(source_group)
            source_errors = validate_chaser_distance_base_semantics(source_group)
            if source_errors:
                raise ValueError(source_errors)
            if [item.as_manifest() for item in source_declarations] != [
                item.as_manifest() for item in declarations
            ]:
                errors.append("candidate declaration inventory differs from sealed source")
            source_hashes = base_logical_hashes(source_group, source_declarations)
            if source_hashes != current_hashes:
                errors.append("candidate decoded values differ from sealed source")
            if isinstance(manifest, Mapping) and isinstance(manifest.get("payload"), Mapping):
                if manifest["payload"].get("source_logical_hashes") != source_hashes:
                    errors.append("candidate manifest source hashes differ from sealed source")
        except Exception as exc:
            errors.append(f"sealed source validation failed: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "array_count": len(declarations),
        "logical_hashes": current_hashes,
    }


__all__ = [
    "BASE_LOGICAL_HASHES_ATTR",
    "BASE_MANIFEST_ATTR",
    "BASE_MANIFEST_DIGEST_ATTR",
    "BASE_SOURCE_BINDING_ATTR",
    "base_logical_hashes",
    "build_base_candidate_manifest",
    "build_base_storage_receipt",
    "build_source_authority_binding",
    "persist_base_candidate_contract",
    "rematerialize_base_candidate",
    "validate_base_candidate",
    "validate_base_storage_receipt",
]
