"""Both native integrity layers, without inventing an acceptance authority."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
from typing import Mapping

import h5py

from .common import (
    BLOCK_BYTES,
    MAX_DEPTH,
    MAX_OBJECTS,
    PROFILE,
    SHA256_PATTERN,
    canonical_json,
    digest,
    exact_keys,
    internal_path,
    parse_json,
    require,
    same_json,
    text,
    uint64,
)
from .hdf5_types import dataset_bytes
from .schema import contract, describe_internal_dataset, describe_table, read_json

COMPLETION = "/metadata/completion_json"
OUTCOMES = "/metadata/component_outcomes_json"
MANIFEST = "/evidence/integrity/dependency_manifest_json"
FINALIZATION = "/evidence/integrity/finalization_receipt_json"
SELF_PATHS = (MANIFEST, FINALIZATION)
COMPONENT_RECEIPTS = {
    "correspondence": "/correspondence/receipt_json",
    "geometry": "/geometry/correspondence/receipt_json",
    "recording_association": "/metadata/recording_association/receipt_json",
}


def source_identity(h5, source_h5: Path) -> dict:
    require(h5.mode == "r" and h5.driver == "sec2", "readonly_local_h5_handle_required")
    path = Path(source_h5).expanduser().resolve(strict=True)
    fd = h5.id.get_vfd_handle()
    actual, named = os.fstat(fd), path.stat()
    require(
        (actual.st_dev, actual.st_ino) == (named.st_dev, named.st_ino),
        "source_h5_handle_path_mismatch",
    )
    return {
        "path": str(path),
        "device": actual.st_dev,
        "inode": actual.st_ino,
        "size_bytes": actual.st_size,
        "mtime_ns": actual.st_mtime_ns,
        "ctime_ns": actual.st_ctime_ns,
    }


def source_file_digest(h5, identity: Mapping) -> str:
    """Hash the same open descriptor, not a path reopened after preflight."""
    hasher, offset = sha256(), 0
    fd = h5.id.get_vfd_handle()
    while offset < identity["size_bytes"]:
        data = os.pread(fd, min(BLOCK_BYTES, identity["size_bytes"] - offset), offset)
        require(bool(data), "source_h5_short_read")
        hasher.update(data)
        offset += len(data)
    require(
        source_identity(h5, Path(identity["path"])) == dict(identity),
        "source_h5_generation_changed",
    )
    return "sha256:" + hasher.hexdigest()


def validate_external_receipt(h5, *, source_h5: Path, receipt: dict) -> dict:
    identity = source_identity(h5, source_h5)
    exact_keys(
        receipt,
        (
            "schema_id",
            "schema_version",
            "canonicalization",
            "contract",
            "contract_sha256",
            "receipt_id",
        ),
        "external_receipt",
    )
    expected = "citrus.recording_observation_finalized_receipt"
    require(
        receipt["schema_id"] == expected
        and type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["canonicalization"] == "canonical_json_utf8_sort_keys_compact_v1",
        "external_receipt_schema",
    )
    body = receipt["contract"]
    exact_keys(
        body,
        (
            "schema_id",
            "schema_version",
            "observation_context_id",
            "citrus_experiment_id",
            "citrus_session_uuid",
            "request_id",
            "request_contract_sha256",
            "acceptance_id",
            "acceptance_contract_sha256",
            "target",
            "runtime_geometry_contract_sha256",
            "protocol_semantic",
            "session_status",
            "finalized_at_utc",
            "h5_artifact",
        ),
        "external_receipt_contract",
    )
    require(
        body["schema_id"] == expected
        and type(body["schema_version"]) is int
        and body["schema_version"] == 1
        and body["session_status"] == "COMPLETE",
        "external_receipt_not_complete",
    )
    contract_digest = digest(canonical_json(body))
    require(
        receipt["contract_sha256"] == contract_digest
        and receipt["receipt_id"] == "obsbindfin_" + contract_digest[7:],
        "external_receipt_identity_mismatch",
    )
    for name in (
        "observation_context_id",
        "citrus_experiment_id",
        "citrus_session_uuid",
        "finalized_at_utc",
    ):
        text(body[name], f"external_receipt:{name}")
    for prefix, marker in (("request", "obsbindreq_"), ("acceptance", "obsbindacc_")):
        value = body[prefix + "_contract_sha256"]
        require(
            isinstance(value, str) and SHA256_PATTERN.fullmatch(value),
            f"external_receipt_digest:{prefix}",
        )
        require(
            body[prefix + "_id"] == marker + value[7:],
            f"external_receipt_identity:{prefix}",
        )
    target = body["target"]
    exact_keys(
        target,
        ("arena_id", "camera_id", "canvas_name", "rig_id", "source_camera_stream_id"),
        "external_receipt_target",
    )
    for name, value in target.items():
        text(value, f"external_receipt_target:{name}")
    artifact = body["h5_artifact"]
    exact_keys(
        artifact, ("relative_path", "sha256", "size_bytes"), "external_h5_artifact"
    )
    relative = text(artifact["relative_path"], "external_h5_relative_path")
    require(
        not relative.startswith("/")
        and "\\" not in relative
        and all(part not in ("", ".", "..") for part in relative.split("/"))
        and PurePosixPath(relative).as_posix() == relative,
        "invalid_external_relative_path",
    )
    require(
        uint64(artifact["size_bytes"], "external_h5_size") == identity["size_bytes"],
        "external_h5_size_mismatch",
    )
    require(
        artifact["sha256"] == source_file_digest(h5, identity),
        "external_h5_byte_digest_mismatch",
    )
    return identity


def internal_nodes(h5) -> dict[str, str]:
    """List every hard-link path, including aliases; reject cycles before reads."""
    nodes = {"/": "group"}

    def visit(group, ancestors):
        address = h5py.h5o.get_info(group.id).addr
        require(address not in ancestors, f"cyclic_hard_link:{group.name}")
        require(len(ancestors) < MAX_DEPTH, "h5_group_depth_budget_exceeded")
        for name in group:
            require(name not in (".", "..") and "/" not in name, "invalid_h5_node_name")
            require(
                isinstance(group.get(name, getlink=True), h5py.HardLink),
                f"non_internal_h5_link:{group.name}/{name}",
            )
            child = group[name]
            require(len(nodes) < MAX_OBJECTS, "h5_object_budget_exceeded")
            if isinstance(child, h5py.Dataset):
                require(
                    not child.is_virtual and not child.external,
                    f"dataset_storage_not_internal:{child.name}",
                )
                nodes[child.name] = "dataset"
            else:
                require(
                    isinstance(child, h5py.Group), f"unsupported_h5_object:{child.name}"
                )
                nodes[child.name] = "group"
                visit(child, (*ancestors, address))

    visit(h5, ())
    return nodes


@dataclass(frozen=True)
class InternalIntegrity:
    dependency_count: int
    manifest_sha256: str
    component_outcomes: tuple[dict, ...]
    table_descriptors: Mapping[str, dict]
    nodes: Mapping[str, str]


def validate_internal_integrity(h5) -> InternalIntegrity:
    nodes = internal_nodes(h5)
    require("/metadata/session" in nodes, "unified_session_missing")
    session = h5["/metadata/session"]
    require(
        session.attrs.get("recording_artifact_profile") == PROFILE,
        "unified_profile_mismatch",
    )
    require(
        session.attrs.get("session_status") == "COMPLETE",
        "unified_session_not_complete",
    )
    require(
        h5.attrs.get("development_schema_id")
        == "citrus.experimental_h5_core_writer_test"
        and h5.attrs.get("development_scope")
        == "composed_logger_transitional_metadata_not_production_admitted"
        and "schema_id" not in h5.attrs
        and "schema_version" not in h5.attrs,
        "unsupported_unified_profile_epoch",
    )
    completion = read_json(h5, COMPLETION, canonical=True)
    require(
        same_json(
            completion,
            {
                "schema_id": "citrus.experimental_h5.completion",
                "schema_version": 1,
                "capture": "complete",
                "finalization": "complete",
            },
        ),
        "unified_finalization_incomplete",
    )
    final = read_json(h5, FINALIZATION, canonical=True)
    exact_keys(
        final,
        (
            "schema_id",
            "schema_version",
            "status",
            "reason",
            "scope",
            "dependency_count",
            "dependency_manifest_ref",
            "dependency_manifest_sha256",
            "component_outcomes_ref",
            "component_outcomes_sha256",
        ),
        "global_finalization",
    )
    require(
        final["schema_id"] == "citrus.experimental_h5.global_finalization_receipt"
        and type(final["schema_version"]) is int
        and final["schema_version"] == 1
        and final["status"] == "complete"
        and final["reason"] == ""
        and final["scope"]
        == "all_internal_datasets_except_manifest_and_receipt_self_not_file_bytes_or_scientific_acceptance"
        and final["dependency_manifest_ref"] == MANIFEST
        and final["component_outcomes_ref"] == OUTCOMES,
        "global_finalization_contract_mismatch",
    )
    manifest = read_json(h5, MANIFEST, canonical=True)
    exact_keys(
        manifest,
        (
            "schema_id",
            "schema_version",
            "status",
            "reason",
            "coverage",
            "digest_encoding",
            "excluded_self_paths",
            "dependencies",
        ),
        "global_dependency_manifest",
    )
    require(
        manifest["schema_id"] == "citrus.experimental_h5.dependency_manifest"
        and type(manifest["schema_version"]) is int
        and manifest["schema_version"] == 1
        and manifest["status"] == "complete"
        and manifest["reason"] == ""
        and manifest["coverage"]
        == "all_internal_datasets_except_manifest_and_receipt_self"
        and manifest["digest_encoding"]
        == "schema_specific_or_hdf5_closed_logical_values_v1"
        and manifest["excluded_self_paths"] == list(SELF_PATHS),
        "global_manifest_contract_mismatch",
    )
    require(
        digest(canonical_json(manifest)) == final["dependency_manifest_sha256"],
        "global_manifest_digest_mismatch",
    )
    dependencies = manifest["dependencies"]
    require(
        type(dependencies) is list
        and len(dependencies) == uint64(final["dependency_count"], "dependency_count"),
        "dependency_count_mismatch",
    )
    observed, tables, receipt_components = set(), {}, set()
    for entry in dependencies:
        require(type(entry) is dict, "invalid_dependency_entry")
        kind = entry.get("kind")
        path = internal_path(
            entry.get("table", {}).get("path")
            if kind == "canonical_table"
            else entry.get("path")
        )
        require(
            path not in observed and nodes.get(path) == "dataset",
            f"duplicate_or_missing_dependency:{path}",
        )
        observed.add(path)
        if kind == "canonical_table":
            actual = describe_table(h5, path)
            expected = dict(
                actual,
                kind="canonical_table",
                component_id=actual["table"]["component"],
            )
            require(same_json(entry, expected), f"canonical_dependency_mismatch:{path}")
            tables[path] = actual
        elif kind in ("component_receipt", "component_outcomes"):
            component = entry.get("component_id")
            expected_path = (
                OUTCOMES
                if kind == "component_outcomes" and component == "component_outcomes"
                else COMPONENT_RECEIPTS.get(component)
            )
            require(path == expected_path, f"component_receipt_path_mismatch:{path}")
            require(component not in receipt_components, "duplicate_component_receipt")
            receipt_components.add(component)
            data = dataset_bytes(h5[path])
            parsed = parse_json(data, label=path, canonical=True)
            require(
                entry
                == {
                    "kind": kind,
                    "component_id": component,
                    "path": path,
                    "content_sha256": digest(data),
                    "size_bytes": len(data),
                    "encoding": "canonical_json_utf8_sort_keys_compact_v1",
                    "reported_status": parsed.get("status", ""),
                },
                f"receipt_dependency_mismatch:{path}",
            )
            if kind == "component_receipt":
                require(
                    parsed.get("status") == "complete" and parsed.get("reason") == "",
                    f"component_receipt_incomplete:{component}",
                )
        else:
            require(
                kind == "internal_dataset"
                and entry.get("component_id") == "dataset:" + path,
                f"unknown_dependency_kind:{path}",
            )
            require(
                same_json(
                    describe_internal_dataset(h5[path], "dataset:" + path), entry
                ),
                f"internal_dependency_mismatch:{path}",
            )
    require(
        observed
        == {path for path, kind in nodes.items() if kind == "dataset"}
        - set(SELF_PATHS),
        "global_dependency_graph_not_closed",
    )
    for path, kind in nodes.items():
        if (
            kind == "dataset"
            and str(h5[path].attrs.get("schema_id", "")).startswith(
                "citrus.experimental_h5."
            )
            and path not in tables
        ):
            tables[path] = describe_table(h5, path)
    outcomes = read_json(h5, OUTCOMES, canonical=True)
    exact_keys(
        outcomes, ("schema_id", "schema_version", "components"), "component_outcomes"
    )
    require(
        outcomes["schema_id"] == "citrus.experimental_h5.component_outcomes"
        and type(outcomes["schema_version"]) is int
        and outcomes["schema_version"] == 1
        and type(outcomes["components"]) is list,
        "component_outcomes_schema",
    )
    require(
        digest(dataset_bytes(h5[OUTCOMES])) == final["component_outcomes_sha256"],
        "component_outcomes_digest_mismatch",
    )
    rules = contract("experimental_h5_core_v1.json")["component_outcomes"]
    components = {}
    for outcome in outcomes["components"]:
        exact_keys(outcome, rules["closed_entry_fields"], "component_outcome")
        component = text(outcome["component_id"], "component_id")
        require(
            component not in components
            and outcome["requirement"] in rules["requirement"]
            and outcome["status"] in rules["status"],
            "component_outcome_duplicate_or_unknown",
        )
        for count in ("expected_rows", "written_rows", "write_errors", "dropped_rows"):
            uint64(outcome[count], f"{component}:{count}")
        text(outcome["reason"], f"{component}:reason", empty=True)
        if outcome["requirement"] == "required":
            require(
                outcome["status"] == "complete"
                and outcome["write_errors"] == outcome["dropped_rows"] == 0
                and outcome["expected_rows"] == outcome["written_rows"]
                and outcome["reason"] == "",
                f"required_component_incomplete:{component}",
            )
            if component in COMPONENT_RECEIPTS:
                require(
                    component in receipt_components,
                    f"required_component_receipt_missing:{component}",
                )
        components[component] = outcome
    require(
        components.get("frames", {}).get("requirement") == "required",
        "required_frame_component_missing",
    )
    return InternalIntegrity(
        len(dependencies),
        final["dependency_manifest_sha256"],
        tuple(outcomes["components"]),
        tables,
        nodes,
    )
