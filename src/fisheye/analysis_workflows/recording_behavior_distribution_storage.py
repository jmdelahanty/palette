"""Object-free immutable Zarr storage for recording behavior distributions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.group_statistics.recording_behavior_distributions import (
    RecordingBehaviorDistributionResult,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


PARENT_PATH = "analysis/recording_behavior_distribution_runs"
SCHEMA_ID = "palette.analysis.recording_behavior_distribution_run"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA_ID = "palette.analysis.recording_behavior_distribution_manifest"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_ATTR = "recording_behavior_distribution_manifest"
MANIFEST_DIGEST_ATTR = "recording_behavior_distribution_manifest_sha256"
JSON_ROW_TABLE_SCHEMA_ID = "palette.storage.canonical_json_row_table"
JSON_ROW_TABLE_SCHEMA_VERSION = 1
JSON_ROW_TABLE_ENCODING = "canonical_utf8_offsets_uint8_row_sha256_v1"

_TABLE_ORDER = (
    "configuration",
    "scope_registry",
    "metric_registry",
    "axis_audits",
    "group_registry",
    "source_identity_registry",
    "support",
    "sparse_bins",
)
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "selected_run",
)


class RecordingBehaviorDistributionStorageError(ValueError):
    """A persisted recording-distribution run is incomplete or inconsistent."""


def _fail(message: str) -> None:
    raise RecordingBehaviorDistributionStorageError(message)


def _table_arrays(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, np.ndarray]:
    encoded = tuple(canonical_json_bytes(dict(row)) for row in rows)
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
        offsets[1:] = np.cumsum([len(value) for value in encoded], dtype=np.int64)
    payload = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    row_sha256 = np.zeros((len(encoded), 32), dtype=np.uint8)
    for index, value in enumerate(encoded):
        row_sha256[index] = np.frombuffer(hashlib.sha256(value).digest(), dtype=np.uint8)
    return MappingProxyType(
        {"row_offsets": offsets, "row_utf8": payload, "row_sha256": row_sha256}
    )


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> zarr.Array:
    array = np.ascontiguousarray(values)
    if array.ndim == 1:
        chunks = (max(1, min(65_536, array.shape[0] or 1)),)
    elif array.ndim == 2:
        chunks = (max(1, min(4_096, array.shape[0] or 1)), array.shape[1])
    else:  # pragma: no cover - closed private carrier
        raise AssertionError(array.shape)
    return group.create_array(
        name,
        data=array,
        chunks=chunks,
        overwrite=False,
    )


def _write_json_row_table(
    parent: zarr.Group,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    group = parent.create_group(name)
    arrays = _table_arrays(rows)
    declarations = []
    for array_name, values in arrays.items():
        _write_array(group, array_name, values)
        declarations.append(
            {
                "path": f"{name}/{array_name}",
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "content_sha256": sha256_array(values),
            }
        )
    group.attrs.update(
        {
            "schema_id": JSON_ROW_TABLE_SCHEMA_ID,
            "schema_version": JSON_ROW_TABLE_SCHEMA_VERSION,
            "encoding": JSON_ROW_TABLE_ENCODING,
            "row_count": len(rows),
            "array_paths": [item["path"] for item in declarations],
        }
    )
    return tuple(declarations)


def _registries(
    result: RecordingBehaviorDistributionResult,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    groups: dict[str, str] = {}
    sources: dict[str, str] = {}
    support = []
    for raw in result.support:
        row = dict(raw)
        group_sha256 = str(row.pop("group_key_sha256"))
        group_json = str(row.pop("group_key_json"))
        source_sha256 = str(row.pop("source_identity_key_sha256"))
        source_json = str(row.pop("source_identity_key_json"))
        if canonical_json_sha256(json.loads(group_json)) != group_sha256:
            _fail("In-memory group registry digest is inconsistent.")
        if canonical_json_sha256(json.loads(source_json)) != source_sha256:
            _fail("In-memory source-identity registry digest is inconsistent.")
        if groups.setdefault(group_sha256, group_json) != group_json:
            _fail("One group digest maps to multiple JSON identities.")
        if sources.setdefault(source_sha256, source_json) != source_json:
            _fail("One source digest maps to multiple JSON identities.")
        row["group_key_sha256"] = group_sha256
        row["source_identity_key_sha256"] = source_sha256
        support.append(MappingProxyType(row))
    group_rows = tuple(
        MappingProxyType({"group_key_sha256": key, "group_key_json": groups[key]})
        for key in sorted(groups)
    )
    source_rows = tuple(
        MappingProxyType(
            {
                "source_identity_key_sha256": key,
                "source_identity_key_json": sources[key],
            }
        )
        for key in sorted(sources)
    )
    return group_rows, source_rows, tuple(support)


def _logical_tables(
    result: RecordingBehaviorDistributionResult,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    groups, sources, support = _registries(result)
    scopes = tuple(
        MappingProxyType(dict(row))
        for row in result.config.record["scope_registry"]["scopes"]
    )
    return MappingProxyType(
        {
            "configuration": (result.config.record,),
            "scope_registry": scopes,
            "metric_registry": result.metric_registry,
            "axis_audits": result.axis_audits,
            "group_registry": groups,
            "source_identity_registry": sources,
            "support": support,
            "sparse_bins": result.sparse_bins,
        }
    )


def write_recording_behavior_distribution_run(
    local_zarr: str | Path,
    *,
    run_name: str,
    result: RecordingBehaviorDistributionResult,
    run_provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write one complete selector-ineligible run into a node-local Zarr."""

    if type(result) is not RecordingBehaviorDistributionResult:
        _fail("result must be one RecordingBehaviorDistributionResult.")
    if run_name != result.config.distribution_run_id:
        _fail("Run name differs from the reduced distribution identity.")
    root = open_zarr_root(local_zarr, mode="a", use_consolidated=False)
    parent = require_runs_parent(
        root.require_group("analysis"), "recording_behavior_distribution_runs"
    )
    if run_name in parent:
        _fail(f"Refusing to overwrite recording distribution run {run_name!r}.")
    run = parent.create_group(run_name)
    mark_run_started(run, run_name=run_name, stage=SCHEMA_ID)
    run.attrs["stage_selector_eligible"] = False
    tables = _logical_tables(result)
    declarations = []
    table_records = []
    for table_name in _TABLE_ORDER:
        rows = tables[table_name]
        table_declarations = _write_json_row_table(run, table_name, rows)
        declarations.extend(table_declarations)
        table_records.append(
            {
                "name": table_name,
                "row_count": len(rows),
                "array_paths": [item["path"] for item in table_declarations],
            }
        )
    unsigned_manifest = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_path": f"{PARENT_PATH}/{run_name}",
        "run_name": run_name,
        "recording_id": result.config.recording_id,
        "distribution_run_id": result.config.distribution_run_id,
        "config_sha256": result.config.record["config_sha256"],
        "result_record_sha256": result.record["record_sha256"],
        "tables": table_records,
        "array_declarations": declarations,
        "storage_encoding": JSON_ROW_TABLE_ENCODING,
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selector_activation": False,
    }
    manifest = {
        **unsigned_manifest,
        "payload_digest": canonical_json_sha256(unsigned_manifest),
    }
    run.attrs.update(
        json_attr_safe(
            {
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "run_name": run_name,
                "run_path": f"{PARENT_PATH}/{run_name}",
                "recording_id": result.config.recording_id,
                "distribution_run_id": result.config.distribution_run_id,
                MANIFEST_ATTR: manifest,
                MANIFEST_DIGEST_ATTR: canonical_json_sha256(manifest),
                "production_authority": False,
                "registry_update": False,
                "selection": "none",
                "run_provenance": dict(run_provenance),
            }
        )
    )
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_name,
        run_provenance=run_provenance,
    )
    run.attrs["stage_selector_eligible"] = False
    consolidate_metadata_capture_expected_warnings(local_zarr)
    return MappingProxyType(manifest)


def _declared_array(
    run: zarr.Group, declaration: Mapping[str, Any]
) -> np.ndarray:
    path = str(declaration["path"])
    try:
        node = run[path]
    except KeyError as exc:
        raise RecordingBehaviorDistributionStorageError(
            f"Manifest-declared array {path!r} is absent."
        ) from exc
    values = np.asarray(node[...])
    if (
        values.dtype.str != declaration.get("dtype")
        or list(values.shape) != declaration.get("shape")
        or sha256_array(values) != declaration.get("content_sha256")
    ):
        _fail(f"Manifest-declared array {path!r} changed.")
    return values


def _read_json_row_table(
    run: zarr.Group,
    table: Mapping[str, Any],
    declarations: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    name = str(table["name"])
    group = run[name]
    if (
        group.attrs.get("schema_id") != JSON_ROW_TABLE_SCHEMA_ID
        or group.attrs.get("schema_version") != JSON_ROW_TABLE_SCHEMA_VERSION
        or group.attrs.get("encoding") != JSON_ROW_TABLE_ENCODING
        or int(group.attrs.get("row_count", -1)) != int(table["row_count"])
    ):
        _fail(f"JSON row table {name!r} has invalid metadata.")
    paths = tuple(str(value) for value in table["array_paths"])
    expected_paths = tuple(f"{name}/{value}" for value in ("row_offsets", "row_utf8", "row_sha256"))
    if paths != expected_paths or tuple(group.attrs.get("array_paths", ())) != paths:
        _fail(f"JSON row table {name!r} has an invalid array roster.")
    arrays = {
        path.rsplit("/", 1)[-1]: _declared_array(run, declarations[path])
        for path in paths
    }
    offsets = arrays["row_offsets"]
    payload = arrays["row_utf8"]
    digests = arrays["row_sha256"]
    count = int(table["row_count"])
    if (
        offsets.dtype != np.dtype(np.int64)
        or offsets.shape != (count + 1,)
        or payload.dtype != np.dtype(np.uint8)
        or payload.ndim != 1
        or digests.dtype != np.dtype(np.uint8)
        or digests.shape != (count, 32)
        or offsets[0] != 0
        or offsets[-1] != payload.size
        or np.any(np.diff(offsets) < 0)
    ):
        _fail(f"JSON row table {name!r} has an invalid physical layout.")
    rows = []
    for index in range(count):
        encoded = payload[int(offsets[index]) : int(offsets[index + 1])].tobytes()
        if hashlib.sha256(encoded).digest() != digests[index].tobytes():
            _fail(f"JSON row table {name!r} has a changed row digest.")
        try:
            row = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordingBehaviorDistributionStorageError(
                f"JSON row table {name!r} has an invalid row."
            ) from exc
        if not isinstance(row, dict) or canonical_json_bytes(row) != encoded:
            _fail(f"JSON row table {name!r} row is not canonical JSON.")
        rows.append(MappingProxyType(row))
    return tuple(rows)


def _restore_result_record(
    tables: Mapping[str, tuple[Mapping[str, Any], ...]],
) -> Mapping[str, Any]:
    if len(tables["configuration"]) != 1:
        _fail("Recording distribution has no unique configuration row.")
    groups = {
        str(row["group_key_sha256"]): str(row["group_key_json"])
        for row in tables["group_registry"]
    }
    sources = {
        str(row["source_identity_key_sha256"]): str(
            row["source_identity_key_json"]
        )
        for row in tables["source_identity_registry"]
    }
    if len(groups) != len(tables["group_registry"]) or len(sources) != len(
        tables["source_identity_registry"]
    ):
        _fail("Recording distribution identity registries are duplicated.")
    support = []
    for raw in tables["support"]:
        row = dict(raw)
        try:
            row["group_key_json"] = groups[str(row["group_key_sha256"])]
            row["source_identity_key_json"] = sources[
                str(row["source_identity_key_sha256"])
            ]
        except KeyError as exc:
            raise RecordingBehaviorDistributionStorageError(
                "Support row references an absent identity registry."
            ) from exc
        support.append(row)
    body = {
        "config": dict(tables["configuration"][0]),
        "metric_registry": [dict(row) for row in tables["metric_registry"]],
        "axis_audits": [dict(row) for row in tables["axis_audits"]],
        "support": support,
        "sparse_bins": [dict(row) for row in tables["sparse_bins"]],
    }
    return MappingProxyType(
        {**body, "record_sha256": canonical_json_sha256(body)}
    )


def _validate_conservation(
    support: Sequence[Mapping[str, Any]], bins: Sequence[Mapping[str, Any]]
) -> None:
    by_key = {str(row["support_key_sha256"]): row for row in support}
    if len(by_key) != len(support):
        _fail("Recording distribution support keys are duplicated.")
    grouped: dict[str, list[Mapping[str, Any]]] = {key: [] for key in by_key}
    seen_bins: set[tuple[str, int]] = set()
    for row in bins:
        key = str(row["support_key_sha256"])
        if key not in grouped:
            _fail("Sparse bin references an absent support row.")
        bin_key = (key, int(row["grid_index"]))
        if bin_key in seen_bins:
            _fail("Sparse distribution-bin key is duplicated.")
        seen_bins.add(bin_key)
        grouped[key].append(row)
    for key, row in by_key.items():
        selected = grouped[key]
        count = sum(int(item["bin_count"]) for item in selected)
        weight = math.fsum(float(item["bin_weight"]) for item in selected)
        denominator = float(row["denominator_weight"])
        if count != int(row["valid_count"]) or not math.isclose(
            weight, denominator, rel_tol=1e-10, abs_tol=1e-10
        ):
            _fail("Sparse bins do not conserve support count or denominator.")
        fraction = math.fsum(float(item["fraction"]) for item in selected)
        expected = 1.0 if denominator > 0 else 0.0
        if not math.isclose(fraction, expected, rel_tol=1e-10, abs_tol=1e-10):
            _fail("Sparse-bin fractions do not conserve probability.")


def validate_recording_behavior_distribution_run(
    run: zarr.Group,
    *,
    expected_run_name: str | None = None,
    expected_recording_id: str | None = None,
    expected_result_record_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Deep-validate one already-open exact recording distribution run."""

    attrs = run.attrs
    manifest = attrs.get(MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        _fail("Recording distribution manifest is absent.")
    manifest = dict(manifest)
    unsigned = {key: value for key, value in manifest.items() if key != "payload_digest"}
    if (
        attrs.get("schema_id") != SCHEMA_ID
        or attrs.get("schema_version") != SCHEMA_VERSION
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("schema_id") != MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("payload_digest") != canonical_json_sha256(unsigned)
        or attrs.get(MANIFEST_DIGEST_ATTR) != canonical_json_sha256(manifest)
    ):
        _fail("Recording distribution schema, lifecycle, or manifest is invalid.")
    if expected_run_name is not None and manifest.get("run_name") != expected_run_name:
        _fail("Recording distribution run name differs from the request.")
    if expected_recording_id is not None and manifest.get("recording_id") != expected_recording_id:
        _fail("Recording distribution recording identity differs from the request.")
    table_records = tuple(manifest.get("tables", ()))
    if tuple(str(row.get("name")) for row in table_records) != _TABLE_ORDER:
        _fail("Recording distribution table roster or order is invalid.")
    raw_declarations = tuple(manifest.get("array_declarations", ()))
    declarations = {str(row.get("path")): row for row in raw_declarations}
    if len(declarations) != len(raw_declarations):
        _fail("Recording distribution array declarations are duplicated.")
    declared_paths = {
        str(path) for table in table_records for path in table.get("array_paths", ())
    }
    if set(declarations) != declared_paths:
        _fail("Recording distribution table and array manifests disagree.")
    tables = MappingProxyType(
        {
            str(table["name"]): _read_json_row_table(run, table, declarations)
            for table in table_records
        }
    )
    restored = _restore_result_record(tables)
    if restored["record_sha256"] != manifest.get("result_record_sha256"):
        _fail("Persisted tables do not reconstruct the reduced result digest.")
    if (
        expected_result_record_sha256 is not None
        and restored["record_sha256"] != expected_result_record_sha256
    ):
        _fail("Persisted result differs from the requested in-memory result.")
    configuration = tables["configuration"][0]
    if (
        configuration.get("config_sha256") != manifest.get("config_sha256")
        or configuration.get("recording_id") != manifest.get("recording_id")
        or configuration.get("distribution_run_id")
        != manifest.get("distribution_run_id")
        or tuple(configuration["scope_registry"]["scopes"])
        != tuple(dict(row) for row in tables["scope_registry"])
    ):
        _fail("Configuration, scope table, and manifest identities disagree.")
    _validate_conservation(tables["support"], tables["sparse_bins"])
    return MappingProxyType(
        {
            "manifest": MappingProxyType(manifest),
            "manifest_sha256": canonical_json_sha256(manifest),
            "tables": tables,
            "result_record": restored,
            "verification_digest": canonical_json_sha256(
                {
                    "manifest_sha256": canonical_json_sha256(manifest),
                    "result_record_sha256": restored["record_sha256"],
                    "array_content_sha256": {
                        path: declarations[path]["content_sha256"]
                        for path in sorted(declarations)
                    },
                }
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class RecordingBehaviorDistributionSourceHandle:
    analysis_zarr: Path
    run_path: str
    run_name: str
    recording_id: str
    manifest: Mapping[str, Any]
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    result_record: Mapping[str, Any]
    verification_digest: str


def load_recording_behavior_distribution_source_handle(
    analysis_zarr: str | Path,
    *,
    run_name: str,
    expected_recording_id: str | None = None,
) -> RecordingBehaviorDistributionSourceHandle:
    """Load one exact published run through consolidated metadata."""

    archive = Path(analysis_zarr).expanduser().resolve()
    run_path = f"{PARENT_PATH}/{run_name}"
    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    try:
        run = root[run_path]
    except KeyError as exc:
        raise RecordingBehaviorDistributionStorageError(
            f"Exact recording distribution run {run_path!r} is absent."
        ) from exc
    validated = validate_recording_behavior_distribution_run(
        run,
        expected_run_name=run_name,
        expected_recording_id=expected_recording_id,
    )
    manifest = validated["manifest"]
    return RecordingBehaviorDistributionSourceHandle(
        analysis_zarr=archive,
        run_path=run_path,
        run_name=run_name,
        recording_id=str(manifest["recording_id"]),
        manifest=manifest,
        tables=validated["tables"],
        result_record=validated["result_record"],
        verification_digest=str(validated["verification_digest"]),
    )


def selector_snapshot(parent: zarr.Group | None) -> Mapping[str, Any]:
    """Capture selector-like attrs so a nonpromoting publication can prove stability."""

    attrs = {} if parent is None else parent.attrs
    return MappingProxyType({name: attrs.get(name) for name in _SELECTOR_ATTRS})


__all__ = [
    "JSON_ROW_TABLE_ENCODING",
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "PARENT_PATH",
    "RecordingBehaviorDistributionSourceHandle",
    "RecordingBehaviorDistributionStorageError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "load_recording_behavior_distribution_source_handle",
    "selector_snapshot",
    "validate_recording_behavior_distribution_run",
    "write_recording_behavior_distribution_run",
]
