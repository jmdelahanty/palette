"""Benchmark one explicit stimulus-epoch v1/v2 source-candidate pair read-only.

The controller starts one fresh Python process for each role and rotates role
order between repetitions.  The source archive is opened read-only; evidence
is written only to a new benchmark-labelled directory outside that archive.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from fisheye.analysis.epoch_segments import read_epoch_segments
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis.stimulus_epoch_consumer import (
    PARENT_PATH,
    read_stimulus_epoch_snapshot,
)
from fisheye.analysis.stimulus_epoch_schema import (
    LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
    LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
    STIMULUS_EPOCH_ARRAY_MANIFEST_SCHEMA_ID,
    STIMULUS_EPOCH_FIELD_NAMES,
    STIMULUS_EPOCH_LAYOUT,
    STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_ID,
    STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_VERSION,
    STIMULUS_EPOCH_RUN_MANIFEST_ATTR,
    STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_VERSION,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
    STIMULUS_SOURCE_FINGERPRINT_ALGORITHM,
    stimulus_epoch_logical_content_document,
    stimulus_epoch_logical_content_sha256,
    validate_legacy_stimulus_epoch_source,
)
from fisheye.shared.run_lineage_fingerprint import (
    LINEAGE_PAYLOAD_SCHEMA_ID,
    LINEAGE_PAYLOAD_SCHEMA_VERSION,
    canonical_lineage_json,
    compute_run_lineage_hash,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)

FAMILY_ID = "stimulus_epoch_windows"
BENCHMARK_ID = "stimulus_epoch_source_candidate_reads_v1"
WORKLOAD_ID = "stimulus_epoch_eager_complete_table_v1"
WORKLOAD_SCHEMA_ID = "palette.stimulus_epoch.read_workload"
TRIAL_SCHEMA_ID = "palette.stimulus_epoch.read_trial"
MATRIX_SCHEMA_ID = "palette.stimulus_epoch.read_matrix"
SCHEMA_VERSION = 1
DEFAULT_SEED = 23
DEFAULT_REPETITIONS = 5
ARRAY_PATHS = tuple(f"windows/{name}" for name in STIMULUS_EPOCH_FIELD_NAMES)
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_PHYSICAL_IO_AVAILABILITY = "not_collected_requires_external_trace"
_SHA256_LENGTH = 64


def _strict_envelope(schema_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    json.dumps(normalized, allow_nan=False)
    return {
        "schema_id": schema_id,
        "schema_version": SCHEMA_VERSION,
        "payload": normalized,
        "payload_digest": canonical_json_sha256(normalized),
    }


def _require_envelope(value: Mapping[str, Any], *, schema_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError(
            "Stimulus-epoch evidence envelope has an unexpected field set."
        )
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Stimulus-epoch evidence schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Stimulus-epoch evidence payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Stimulus-epoch evidence payload digest mismatch.")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stimulus-epoch evidence is not strict JSON: {exc}") from exc
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_run_name(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    name = value.strip()
    if (
        not name
        or name != value
        or name in _ALIASES
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return name


def _safe_archive(value: Path | str) -> Path:
    archive = Path(value).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}.")
    return archive


def _safe_persisted_run_path(archive: Path, *, run_name: str) -> Path:
    run_path = archive.joinpath(*PARENT_PATH.split("/"), run_name)
    if not run_path.is_dir() or run_path.is_symlink():
        raise ValueError("Selected stimulus-epoch run must be a nonsymlink directory.")
    return run_path


def _safe_output(value: Path | str, *, archive: Path) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}.")
    if (
        output == archive
        or output.is_relative_to(archive)
        or archive.is_relative_to(output)
    ):
        raise ValueError("Benchmark output and source archive must be disjoint.")
    if not any("benchmark" in component.lower() for component in output.parts):
        raise ValueError("Output path must be explicitly benchmark-only.")
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("Benchmark output path is too broad.")
    return output


def _safe_trial_output(value: Path | str, *, benchmark_root: Path) -> Path:
    output = Path(value).expanduser().resolve()
    root = benchmark_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}.")
    if output.exists() or not output.is_relative_to(root) or output.suffix != ".json":
        raise ValueError("Trial output must be a new JSON file inside benchmark root.")
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace benchmark evidence: {path}.")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {raw}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"Strict JSON document is not one object: {path}.")
    return value


def _strict_json_copy(value: object) -> Any:
    return json.loads(
        json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True),
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {raw}")
        ),
    )


def _group(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _array(run_group: Any, path: str) -> Any:
    return _group(run_group, path)


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    value = call()
    return value, {
        "wall_seconds": float(time.perf_counter() - wall_started),
        "cpu_seconds": float(time.process_time() - cpu_started),
    }


def _require_timing(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"wall_seconds", "cpu_seconds"}:
        raise ValueError(f"{label} timing has an unexpected field set.")
    for field in ("wall_seconds", "cpu_seconds"):
        observed = value[field]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"{label} {field} is invalid.")


def _lineage_identity(group: Any) -> tuple[str, str, dict[str, Any]]:
    lineage_json = group.attrs.get("lineage_payload_json")
    if type(lineage_json) is not str:
        raise ValueError("Source lineage payload is absent.")
    payload = json.loads(lineage_json)
    if type(payload) is not dict or lineage_json != canonical_lineage_json(payload):
        raise ValueError("Source lineage payload is not canonical.")
    lineage_hash = compute_run_lineage_hash(payload)
    for attr in ("lineage_hash", "source_lineage_hash", "source_fingerprint"):
        if group.attrs.get(attr) != lineage_hash:
            raise ValueError(f"Source {attr} differs from its lineage payload.")
    return (
        lineage_hash,
        hashlib.sha256(lineage_json.encode("utf-8")).hexdigest(),
        payload,
    )


def _validate_source(group: Any, *, run_name: str) -> dict[str, Any]:
    errors = list(validate_legacy_stimulus_epoch_source(group))
    if group.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        errors.append("source completion contract is absent")
    if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("source is not explicitly complete")
    if group.attrs.get(RUN_NAME_ATTR) != run_name:
        errors.append("source completion name differs from explicit selection")
    if group.attrs.get("stage_selector_eligible") is False:
        errors.append("source is explicitly selector-ineligible")
    try:
        lineage_hash, lineage_payload_sha256, lineage_payload = _lineage_identity(group)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        lineage_hash = ""
        lineage_payload_sha256 = ""
        lineage_payload = {}
    if errors:
        raise ValueError(
            "Invalid explicit stimulus-epoch v1 source: " + "; ".join(errors)
        )
    return {
        "valid": True,
        "schema_id": "palette.stimulus_epoch_windows.v1",
        "array_count": len(ARRAY_PATHS),
        "lineage_hash": lineage_hash,
        "lineage_payload_sha256": lineage_payload_sha256,
        "lineage_payload": lineage_payload,
    }


def _metadata_receipt(archive: Path, *, run_path: str) -> dict[str, Any]:
    receipt = validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    if receipt.array_count != len(ARRAY_PATHS) or receipt.group_count != 2:
        raise ValueError(
            "Stimulus-epoch persisted metadata inventory is not exact: "
            f"{receipt.to_json()}."
        )
    return receipt.to_json()


def _segments_sha256(segments: Sequence[Any]) -> str:
    return canonical_json_sha256([asdict(segment) for segment in segments])


def _full_scan(run_group: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for path in ARRAY_PATHS:
        array = _array(run_group, path)
        values = np.ascontiguousarray(array[...])
        arrays[path] = {
            "dtype": str(np.dtype(array.dtype)),
            "shape": [int(value) for value in array.shape],
            "decoded_bytes": int(values.nbytes),
            "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        }
    return {
        "operation_count": len(arrays),
        "decoded_bytes": sum(record["decoded_bytes"] for record in arrays.values()),
        "arrays": arrays,
        "arrays_digest": canonical_json_sha256(arrays),
    }


def _storage(run_root: Path) -> dict[str, Any]:
    if run_root.is_symlink() or any(path.is_symlink() for path in run_root.rglob("*")):
        raise ValueError("Selected stimulus-epoch run cannot contain symlinks.")
    stats = storage_stats(run_root)
    return {
        **stats,
        "object_count": stats["file_count"],
        "payload_object_count": stats["payload_file_count"],
    }


def _metadata_guard(archive: Path, *, run_names: Sequence[str]) -> dict[str, Any]:
    paths = [
        archive / "zarr.json",
        archive / "analysis" / "zarr.json",
        archive.joinpath(*PARENT_PATH.split("/"), "zarr.json"),
    ]
    for run_name in run_names:
        paths.extend(
            sorted(
                archive.joinpath(*PARENT_PATH.split("/"), run_name).rglob("zarr.json")
            )
        )
    records: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=str):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required nonsymlink metadata is absent: {path}.")
        payload = path.read_bytes()
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(archive)),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "scope": "root_analysis_parent_and_complete_selected_run_zarr_json",
        "metadata_file_count": len(records),
        "metadata_tree_sha256": canonical_json_sha256(records),
        "files": records,
    }


def _require_metadata_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "scope",
        "metadata_file_count",
        "metadata_tree_sha256",
        "files",
    }:
        raise ValueError("Stimulus-epoch metadata guard has an unexpected field set.")
    if value["scope"] != "root_analysis_parent_and_complete_selected_run_zarr_json":
        raise ValueError("Stimulus-epoch metadata guard scope is unsupported.")
    files = value["files"]
    if not isinstance(files, list):
        raise ValueError("Stimulus-epoch metadata guard files must be one list.")
    paths: list[str] = []
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }:
            raise ValueError("Stimulus-epoch metadata guard record is malformed.")
        path = record["path"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(component in {"", ".", ".."} for component in path.split("/"))
            or not path.endswith("zarr.json")
        ):
            raise ValueError("Stimulus-epoch metadata guard path is unsafe.")
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or type(record["mtime_ns"]) is not int
            or record["mtime_ns"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError("Stimulus-epoch metadata guard facts are invalid.")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise ValueError(
            "Stimulus-epoch metadata guard paths are not sorted and unique."
        )
    if (
        type(value["metadata_file_count"]) is not int
        or value["metadata_file_count"] != len(files)
        or value["metadata_tree_sha256"] != canonical_json_sha256(files)
    ):
        raise ValueError("Stimulus-epoch metadata guard inventory digest mismatch.")


def _environment(*, archive: Path, cache_state: str) -> dict[str, Any]:
    git = get_git_info()
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "palette_commit": git.get("commit_hash"),
        "palette_dirty": bool(git.get("is_dirty")),
        "archive_device": int(archive.stat().st_dev),
        "cache_state": cache_state,
        "thread_environment": {
            key: os.environ.get(key) for key in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _require_environment(value: object, *, cache_state: str) -> None:
    fields = {
        "hostname",
        "system",
        "release",
        "machine",
        "python",
        "numpy",
        "zarr",
        "palette_commit",
        "palette_dirty",
        "archive_device",
        "cache_state",
        "thread_environment",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Stimulus-epoch environment receipt is malformed.")
    for field in (
        "hostname",
        "system",
        "release",
        "machine",
        "python",
        "numpy",
        "zarr",
        "palette_commit",
    ):
        if type(value[field]) is not str or not value[field]:
            raise ValueError(f"Stimulus-epoch environment {field} is invalid.")
    if (
        type(value["palette_dirty"]) is not bool
        or type(value["archive_device"]) is not int
        or value["archive_device"] < 0
        or value["cache_state"] != cache_state
    ):
        raise ValueError("Stimulus-epoch environment identity is invalid.")
    thread_environment = value["thread_environment"]
    if (
        not isinstance(thread_environment, Mapping)
        or set(thread_environment) != set(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
        or any(
            observed is not None and type(observed) is not str
            for observed in thread_environment.values()
        )
    ):
        raise ValueError("Stimulus-epoch thread environment is malformed.")


def _preflight(
    archive: Path,
    *,
    source_run_name: str,
    candidate_run_name: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    _safe_persisted_run_path(archive, run_name=source_run_name)
    _safe_persisted_run_path(archive, run_name=candidate_run_name)
    source_path = f"{PARENT_PATH}/{source_run_name}"
    candidate_path = f"{PARENT_PATH}/{candidate_run_name}"
    direct_root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    source = _group(direct_root, source_path)
    source_validation = _validate_source(source, run_name=source_run_name)
    candidate_snapshot = read_stimulus_epoch_snapshot(
        archive,
        run_name=candidate_run_name,
    )
    consolidated_root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    consolidated_source = _group(consolidated_root, source_path)
    consolidated_candidate = _group(consolidated_root, candidate_path)
    source_metadata = _metadata_receipt(archive, run_path=source_path)
    candidate_metadata = _metadata_receipt(archive, run_path=candidate_path)

    source_document = stimulus_epoch_logical_content_document(consolidated_source)
    candidate_document = stimulus_epoch_logical_content_document(consolidated_candidate)
    if source_document != candidate_document:
        raise ValueError("Stimulus-epoch source and candidate decoded arrays differ.")
    source_segments_sha256 = _segments_sha256(read_epoch_segments(consolidated_source))
    candidate_segments_sha256 = _segments_sha256(candidate_snapshot.segments)
    if source_segments_sha256 != candidate_segments_sha256:
        raise ValueError("Stimulus-epoch source and candidate decoded rows differ.")
    (
        candidate_lineage_hash,
        candidate_lineage_payload_sha256,
        candidate_lineage_payload,
    ) = _lineage_identity(consolidated_candidate)

    candidate_attrs = consolidated_candidate.attrs
    if (
        candidate_attrs.get("source_stimulus_epoch_run") != source_run_name
        or candidate_attrs.get("source_stimulus_epoch_path") != source_path
        or candidate_attrs.get("storage_candidate_source_run") != source_run_name
        or candidate_attrs.get("storage_candidate_source_run_path") != source_path
        or candidate_attrs.get("source_stimulus_epoch_lineage_hash")
        != source_validation["lineage_hash"]
        or candidate_attrs.get("source_stimulus_epoch_lineage_payload_sha256")
        != source_validation["lineage_payload_sha256"]
        or candidate_attrs.get("source_stimulus_epoch_logical_content_sha256")
        != canonical_json_sha256(source_document)
    ):
        raise ValueError("Stimulus-epoch candidate/source binding is invalid.")
    receipt = candidate_attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    manifest = candidate_attrs.get(STIMULUS_EPOCH_RUN_MANIFEST_ATTR)
    if (
        not isinstance(receipt, Mapping)
        or not _is_sha256(receipt.get("payload_digest"))
        or not isinstance(manifest, Mapping)
        or not _is_sha256(manifest.get("payload_digest"))
    ):
        raise ValueError("Stimulus-epoch candidate receipt or manifest is absent.")
    receipt_document = _strict_json_copy(receipt)
    manifest_document = _strict_json_copy(manifest)
    candidate_materializer_identity = {
        "git_commit": candidate_attrs.get("candidate_materializer_git_commit"),
        "git_dirty": candidate_attrs.get("candidate_materializer_git_dirty"),
    }
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "workload_id": WORKLOAD_ID,
        "archive_path": str(archive),
        "source_run_name": source_run_name,
        "candidate_run_name": candidate_run_name,
        "source_run_path": source_path,
        "candidate_run_path": candidate_path,
        "seed": seed,
        "repetitions": repetitions,
        "access": {
            "mode": "eager_whole_array_once",
            "array_order": list(ARRAY_PATHS),
            "operation_count": len(ARRAY_PATHS),
        },
        "expected_logical_content_sha256": canonical_json_sha256(source_document),
        "expected_logical_content": source_document,
        "expected_arrays": source_document["arrays"],
        "expected_segments_sha256": source_segments_sha256,
        "source_lineage_hash": source_validation["lineage_hash"],
        "source_lineage_payload_sha256": source_validation["lineage_payload_sha256"],
        "source_lineage_payload": source_validation["lineage_payload"],
        "candidate_lineage_hash": candidate_lineage_hash,
        "candidate_lineage_payload_sha256": candidate_lineage_payload_sha256,
        "candidate_lineage_payload": candidate_lineage_payload,
        "candidate_materializer_identity": candidate_materializer_identity,
        "candidate_storage_receipt": receipt_document,
        "candidate_storage_receipt_payload_digest": receipt["payload_digest"],
        "candidate_run_manifest": manifest_document,
        "candidate_run_manifest_payload_digest": manifest["payload_digest"],
        "metadata_equivalence": {
            "source": source_metadata,
            "candidate": candidate_metadata,
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _PHYSICAL_IO_AVAILABILITY,
        },
    }
    workload = _strict_envelope(WORKLOAD_SCHEMA_ID, payload)
    require_workload(workload)
    return {
        "workload": workload,
        "source_validation": source_validation,
        "candidate_storage_receipt_payload_digest": receipt["payload_digest"],
        "candidate_run_manifest_payload_digest": manifest["payload_digest"],
    }


def _require_metadata_receipt(value: object, *, run_path: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "subtree_path",
        "node_count",
        "group_count",
        "array_count",
        "declarations_sha256",
    }:
        raise ValueError("Stimulus-epoch metadata-equivalence receipt is malformed.")
    if (
        value["schema_id"] != "palette.zarr.metadata_equivalence"
        or value["schema_version"] != 1
        or value["subtree_path"] != run_path
        or value["array_count"] != len(ARRAY_PATHS)
        or value["group_count"] != 2
        or value["node_count"] != len(ARRAY_PATHS) + 2
        or not _is_sha256(value["declarations_sha256"])
    ):
        raise ValueError("Stimulus-epoch metadata-equivalence receipt is invalid.")


def _require_physical_io(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "request_count",
        "transferred_bytes",
        "availability",
    }:
        raise ValueError("Stimulus-epoch physical-I/O receipt is malformed.")
    if (
        value["request_count"] is not None
        or value["transferred_bytes"] is not None
        or value["availability"] != _PHYSICAL_IO_AVAILABILITY
    ):
        raise ValueError("Stimulus-epoch benchmark must not fabricate physical I/O.")


def _require_array_inventory(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(ARRAY_PATHS):
        raise ValueError("Stimulus-epoch logical array inventory is not exact.")
    for path, record in value.items():
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "shape",
            "digest_algorithm",
            "sha256",
        }:
            raise ValueError(f"Stimulus-epoch expected array {path!r} is malformed.")
        try:
            dtype = np.dtype(record["dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Stimulus-epoch expected array {path!r} dtype is invalid."
            ) from exc
        shape = record["shape"]
        if (
            str(dtype) != record["dtype"]
            or not isinstance(shape, list)
            or any(type(extent) is not int or extent < 0 for extent in shape)
            or record["digest_algorithm"] != "sha256_c_order_decoded_bytes_v1"
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError(f"Stimulus-epoch expected array {path!r} is invalid.")


def _exact_mapping(value: object, *, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"Stimulus-epoch {label} has an unexpected field set.")
    return value


def _require_logical_content(value: object) -> Mapping[str, Any]:
    document = _exact_mapping(
        value,
        fields={"schema_id", "schema_version", "array_count", "arrays"},
        label="logical-content document",
    )
    if (
        document["schema_id"] != STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_ID
        or document["schema_version"] != STIMULUS_EPOCH_LOGICAL_CONTENT_SCHEMA_VERSION
        or document["array_count"] != len(ARRAY_PATHS)
    ):
        raise ValueError("Stimulus-epoch logical-content identity is invalid.")
    _require_array_inventory(document["arrays"])
    return document


def _require_lineage_document(
    value: object,
    *,
    lineage_hash: str,
    payload_sha256: str,
    label: str,
    expected_analysis_schema: Mapping[str, Any],
) -> None:
    payload = _exact_mapping(
        value,
        fields={
            "lineage_schema_id",
            "lineage_schema_version",
            "run_family",
            "analysis_schema",
            "method",
            "method_version",
            "source_refs",
            "source_fingerprints",
            "parameters",
            "code",
        },
        label=f"{label} lineage payload",
    )
    if (
        payload["lineage_schema_id"] != LINEAGE_PAYLOAD_SCHEMA_ID
        or payload["lineage_schema_version"] != LINEAGE_PAYLOAD_SCHEMA_VERSION
        or payload["run_family"] != PARENT_PATH
        or payload["analysis_schema"] != expected_analysis_schema
        or any(
            not isinstance(payload[field], Mapping)
            for field in (
                "analysis_schema",
                "source_refs",
                "source_fingerprints",
                "parameters",
                "code",
            )
        )
    ):
        raise ValueError(f"Stimulus-epoch {label} lineage payload is invalid.")
    canonical = canonical_lineage_json(payload)
    if (
        compute_run_lineage_hash(payload) != lineage_hash
        or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != payload_sha256
    ):
        raise ValueError(f"Stimulus-epoch {label} lineage executable digest mismatch.")


def _require_storage_receipt_document(
    value: object, *, expected_arrays: Mapping[str, Any]
) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError("Stimulus-epoch complete storage receipt is absent.")
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Stimulus-epoch complete storage receipt is invalid: {exc}"
        ) from exc
    entries = {entry.facts.path: entry for entry in parsed.entries}
    if set(entries) != set(ARRAY_PATHS):
        raise ValueError("Stimulus-epoch storage receipt inventory is not exact.")
    for path, entry in entries.items():
        expected = expected_arrays[path]
        if (
            str(entry.facts.dtype) != expected["dtype"]
            or list(entry.facts.shape) != expected["shape"]
        ):
            raise ValueError(
                f"Stimulus-epoch storage receipt {path!r} differs from logical content."
            )
    return parsed


def _require_run_manifest_document(
    value: object,
    *,
    source_run_name: str,
    source_run_path: str,
    candidate_run_name: str,
    candidate_run_path: str,
    source_lineage_hash: str,
    source_lineage_payload_sha256: str,
    candidate_lineage_hash: str,
    candidate_lineage_payload_sha256: str,
    candidate_lineage_payload: Mapping[str, Any],
    candidate_materializer_identity: Mapping[str, Any],
    logical_content: Mapping[str, Any],
    storage_receipt: Mapping[str, Any],
    storage_profile_id: str,
) -> None:
    manifest = _exact_mapping(
        value,
        fields={
            "schema_id",
            "schema_version",
            "persisted_attribute",
            "digest_algorithm",
            "payload",
            "payload_digest",
        },
        label="candidate run manifest",
    )
    if (
        manifest["schema_id"] != STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_ID
        or manifest["schema_version"] != STIMULUS_EPOCH_RUN_MANIFEST_SCHEMA_VERSION
        or manifest["persisted_attribute"] != STIMULUS_EPOCH_RUN_MANIFEST_ATTR
        or manifest["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        raise ValueError("Stimulus-epoch candidate run manifest identity is invalid.")
    payload = _exact_mapping(
        manifest["payload"],
        fields={
            "run_identity",
            "dimensions",
            "source_stimulus",
            "source_epoch",
            "protocol",
            "candidate_lineage",
            "logical_content",
            "schema_bindings",
            "authoritative_child_groups",
            "publication_state",
        },
        label="candidate run-manifest payload",
    )
    if manifest["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Stimulus-epoch candidate run-manifest digest mismatch.")

    run_identity = _exact_mapping(
        payload["run_identity"],
        fields={
            "recording_id",
            "run_name",
            "run_schema_id",
            "run_schema_version",
            "layout",
            "row_axis",
        },
        label="run identity",
    )
    if (
        type(run_identity["recording_id"]) is not str
        or not run_identity["recording_id"]
        or run_identity["run_name"] != candidate_run_name
        or run_identity["run_schema_id"] != STIMULUS_EPOCH_RUN_SCHEMA_ID
        or run_identity["run_schema_version"] != STIMULUS_EPOCH_RUN_SCHEMA_VERSION
        or run_identity["layout"] != STIMULUS_EPOCH_LAYOUT
        or run_identity["row_axis"] != "epoch_windows"
    ):
        raise ValueError("Stimulus-epoch run-manifest run identity is invalid.")
    dimensions = _exact_mapping(
        payload["dimensions"],
        fields={"window_count", "total_frames", "fps"},
        label="run dimensions",
    )
    window_count = logical_content["arrays"][ARRAY_PATHS[0]]["shape"][0]
    if (
        dimensions["window_count"] != window_count
        or type(dimensions["total_frames"]) is not int
        or dimensions["total_frames"] <= 0
        or isinstance(dimensions["fps"], bool)
        or not isinstance(dimensions["fps"], (int, float))
        or not math.isfinite(float(dimensions["fps"]))
        or float(dimensions["fps"]) <= 0
    ):
        raise ValueError("Stimulus-epoch run-manifest dimensions are invalid.")
    source_epoch = _exact_mapping(
        payload["source_epoch"],
        fields={
            "run",
            "path",
            "schema_id",
            "schema_version",
            "lineage_hash",
            "lineage_payload_sha256",
            "logical_content_sha256",
        },
        label="source-epoch binding",
    )
    if source_epoch != {
        "run": source_run_name,
        "path": source_run_path,
        "schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
        "schema_version": LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
        "lineage_hash": source_lineage_hash,
        "lineage_payload_sha256": source_lineage_payload_sha256,
        "logical_content_sha256": canonical_json_sha256(logical_content),
    }:
        raise ValueError("Stimulus-epoch run-manifest source binding is invalid.")
    source_stimulus = _exact_mapping(
        payload["source_stimulus"],
        fields={"run", "path", "fingerprint_algorithm", "fingerprint", "event_schema"},
        label="source-stimulus binding",
    )
    if (
        type(source_stimulus["run"]) is not str
        or not source_stimulus["run"]
        or source_stimulus["path"] != f"analysis/stimulus_runs/{source_stimulus['run']}"
        or source_stimulus["fingerprint_algorithm"]
        != STIMULUS_SOURCE_FINGERPRINT_ALGORITHM
        or not _is_sha256(source_stimulus["fingerprint"])
    ):
        raise ValueError("Stimulus-epoch run-manifest stimulus binding is invalid.")
    event_schema = _exact_mapping(
        source_stimulus["event_schema"],
        fields={"events_path", "event_name_fields", "frame_fields"},
        label="source-event schema",
    )
    if event_schema != {
        "events_path": f"{source_stimulus['path']}/events",
        "event_name_fields": [
            "event_name",
            "event_type_name",
            "name",
            "event_type_id",
        ],
        "frame_fields": [
            "camera_frame_id",
            "camera_frame_num",
            "triggering_camera_frame_id",
        ],
    }:
        raise ValueError("Stimulus-epoch source-event schema is invalid.")
    protocol = _exact_mapping(
        payload["protocol"],
        fields={
            "method",
            "method_version",
            "epoch_policy",
            "epoch_policy_version",
            "profile",
            "source_refs",
        },
        label="protocol binding",
    )
    if protocol["source_refs"] != {
        "source_stimulus_run": source_stimulus["run"],
        "source_stimulus_path": source_stimulus["path"],
    }:
        raise ValueError(
            "Stimulus-epoch run-manifest protocol source binding is invalid."
        )
    for field in ("method", "method_version", "epoch_policy"):
        if (
            type(protocol[field]) is not str
            or not protocol[field]
            or protocol[field] != protocol[field].strip()
        ):
            raise ValueError("Stimulus-epoch protocol identity is invalid.")
    if (
        type(protocol["epoch_policy_version"]) is not int
        or protocol["epoch_policy_version"] <= 0
    ):
        raise ValueError("Stimulus-epoch protocol policy version is invalid.")
    profile = _exact_mapping(
        protocol["profile"],
        fields={
            "profile_id",
            "profile_version",
            "profile_sha256",
            "profile_source",
            "source_adapter_id",
            "source_adapter_version",
            "role_resolver_id",
            "role_resolver_version",
        },
        label="protocol profile",
    )
    for field in (
        "profile_id",
        "profile_sha256",
        "profile_source",
        "source_adapter_id",
        "role_resolver_id",
    ):
        if (
            type(profile[field]) is not str
            or not profile[field]
            or profile[field] != profile[field].strip()
        ):
            raise ValueError("Stimulus-epoch protocol profile text is invalid.")
    for field in (
        "profile_version",
        "source_adapter_version",
        "role_resolver_version",
    ):
        if type(profile[field]) is not int or profile[field] <= 0:
            raise ValueError("Stimulus-epoch protocol profile version is invalid.")
    if not _is_sha256(profile["profile_sha256"]):
        raise ValueError("Stimulus-epoch protocol profile digest is invalid.")
    candidate_lineage = _exact_mapping(
        payload["candidate_lineage"],
        fields={"lineage_hash", "lineage_payload_sha256", "fingerprint_status"},
        label="candidate lineage binding",
    )
    if candidate_lineage != {
        "lineage_hash": candidate_lineage_hash,
        "lineage_payload_sha256": candidate_lineage_payload_sha256,
        "fingerprint_status": "complete",
    }:
        raise ValueError("Stimulus-epoch candidate lineage binding is invalid.")
    if (
        candidate_lineage_payload["method"] != protocol["method"]
        or candidate_lineage_payload["method_version"] != protocol["method_version"]
        or candidate_lineage_payload["source_refs"]
        != {
            "source_stimulus_run": source_stimulus["run"],
            "source_stimulus_path": source_stimulus["path"],
            "source_stimulus_epoch_run": source_run_name,
            "source_stimulus_epoch_path": source_run_path,
        }
        or candidate_lineage_payload["source_fingerprints"]
        != {
            "source_stimulus_fingerprint_algorithm": source_stimulus[
                "fingerprint_algorithm"
            ],
            "source_stimulus_fingerprint": source_stimulus["fingerprint"],
            "source_stimulus_epoch_lineage_hash": source_lineage_hash,
            "source_stimulus_epoch_lineage_payload_sha256": (
                source_lineage_payload_sha256
            ),
            "source_stimulus_epoch_logical_content_sha256": canonical_json_sha256(
                logical_content
            ),
        }
        or candidate_lineage_payload["parameters"]
        != {
            "recording_id": run_identity["recording_id"],
            "fps": float(dimensions["fps"]),
            "total_frames": dimensions["total_frames"],
            "epoch_policy": protocol["epoch_policy"],
            "epoch_policy_version": protocol["epoch_policy_version"],
            "protocol_profile": profile,
        }
        or candidate_lineage_payload["code"] != candidate_materializer_identity
    ):
        raise ValueError(
            "Stimulus-epoch candidate lineage differs from run-manifest semantics."
        )
    if payload["logical_content"] != logical_content:
        raise ValueError("Stimulus-epoch run manifest logical content differs.")
    schema_bindings = _exact_mapping(
        payload["schema_bindings"],
        fields={
            "array_manifest_schema_id",
            "array_manifest_payload_digest",
            "storage_receipt_schema_id",
            "storage_receipt_payload_digest",
        },
        label="schema bindings",
    )
    if (
        schema_bindings["array_manifest_schema_id"]
        != STIMULUS_EPOCH_ARRAY_MANIFEST_SCHEMA_ID
        or not _is_sha256(schema_bindings["array_manifest_payload_digest"])
        or schema_bindings["storage_receipt_schema_id"] != storage_receipt["schema_id"]
        or schema_bindings["storage_receipt_payload_digest"]
        != storage_receipt["payload_digest"]
    ):
        raise ValueError("Stimulus-epoch run-manifest schema binding is invalid.")
    child_groups = _exact_mapping(
        payload["authoritative_child_groups"],
        fields={"windows"},
        label="authoritative child groups",
    )
    windows = _exact_mapping(
        child_groups["windows"],
        fields={"storage_layout", "field_names"},
        label="windows group declaration",
    )
    if windows != {
        "storage_layout": "columnar",
        "field_names": list(STIMULUS_EPOCH_FIELD_NAMES),
    }:
        raise ValueError("Stimulus-epoch windows group declaration is invalid.")
    publication = _exact_mapping(
        payload["publication_state"],
        fields={
            "stage_selector_eligible",
            "storage_candidate_profile_promoted",
            "storage_profile_id",
            "source_candidate_run",
            "source_candidate_path",
        },
        label="publication state",
    )
    if publication != {
        "stage_selector_eligible": False,
        "storage_candidate_profile_promoted": False,
        "storage_profile_id": storage_profile_id,
        "source_candidate_run": source_run_name,
        "source_candidate_path": source_run_path,
    }:
        raise ValueError("Stimulus-epoch run-manifest publication state is invalid.")
    if candidate_run_path != f"{PARENT_PATH}/{candidate_run_name}":
        raise ValueError("Stimulus-epoch candidate path binding is invalid.")


def require_workload(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=WORKLOAD_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "workload_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "source_run_path",
        "candidate_run_path",
        "seed",
        "repetitions",
        "access",
        "expected_logical_content_sha256",
        "expected_logical_content",
        "expected_arrays",
        "expected_segments_sha256",
        "source_lineage_hash",
        "source_lineage_payload_sha256",
        "source_lineage_payload",
        "candidate_lineage_hash",
        "candidate_lineage_payload_sha256",
        "candidate_lineage_payload",
        "candidate_materializer_identity",
        "candidate_storage_receipt",
        "candidate_storage_receipt_payload_digest",
        "candidate_run_manifest",
        "candidate_run_manifest_payload_digest",
        "metadata_equivalence",
        "physical_io",
    }
    if set(payload) != expected:
        raise ValueError("Stimulus-epoch workload has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID or payload["workload_id"] != WORKLOAD_ID:
        raise ValueError("Stimulus-epoch workload identity is unsupported.")
    source_name = _safe_run_name(payload["source_run_name"], label="source_run")
    candidate_name = _safe_run_name(
        payload["candidate_run_name"], label="candidate_run"
    )
    archive_path = payload["archive_path"]
    if source_name == candidate_name:
        raise ValueError("Stimulus-epoch source and candidate names must differ.")
    if (
        type(archive_path) is not str
        or not Path(archive_path).is_absolute()
        or str(Path(archive_path).resolve()) != archive_path
        or payload["source_run_path"] != f"{PARENT_PATH}/{source_name}"
        or payload["candidate_run_path"] != f"{PARENT_PATH}/{candidate_name}"
        or type(payload["seed"]) is not int
        or payload["seed"] < 0
        or type(payload["repetitions"]) is not int
        or payload["repetitions"] < 1
    ):
        raise ValueError("Stimulus-epoch workload run or matrix binding is invalid.")
    if payload["access"] != {
        "mode": "eager_whole_array_once",
        "array_order": list(ARRAY_PATHS),
        "operation_count": len(ARRAY_PATHS),
    }:
        raise ValueError("Stimulus-epoch workload access declaration is unsupported.")
    logical_content = _require_logical_content(payload["expected_logical_content"])
    if payload["expected_arrays"] != logical_content["arrays"] or payload[
        "expected_logical_content_sha256"
    ] != canonical_json_sha256(logical_content):
        raise ValueError("Stimulus-epoch workload logical-content binding is invalid.")
    for field in (
        "expected_logical_content_sha256",
        "expected_segments_sha256",
        "source_lineage_hash",
        "source_lineage_payload_sha256",
        "candidate_lineage_hash",
        "candidate_lineage_payload_sha256",
        "candidate_storage_receipt_payload_digest",
        "candidate_run_manifest_payload_digest",
    ):
        if not _is_sha256(payload[field]):
            raise ValueError(f"Stimulus-epoch workload {field} is invalid.")
    _require_lineage_document(
        payload["source_lineage_payload"],
        lineage_hash=payload["source_lineage_hash"],
        payload_sha256=payload["source_lineage_payload_sha256"],
        label="source",
        expected_analysis_schema={
            "schema_id": LEGACY_STIMULUS_EPOCH_SCHEMA_ID,
            "schema_version": LEGACY_STIMULUS_EPOCH_SCHEMA_VERSION,
            "row_axis": "epoch_windows",
        },
    )
    materializer_identity = _exact_mapping(
        payload["candidate_materializer_identity"],
        fields={"git_commit", "git_dirty"},
        label="candidate materializer identity",
    )
    if (
        type(materializer_identity["git_commit"]) is not str
        or not materializer_identity["git_commit"]
        or materializer_identity["git_commit"]
        != materializer_identity["git_commit"].strip()
        or type(materializer_identity["git_dirty"]) is not bool
    ):
        raise ValueError("Stimulus-epoch candidate materializer identity is invalid.")
    _require_lineage_document(
        payload["candidate_lineage_payload"],
        lineage_hash=payload["candidate_lineage_hash"],
        payload_sha256=payload["candidate_lineage_payload_sha256"],
        label="candidate",
        expected_analysis_schema={
            "schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
            "schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
            "layout": STIMULUS_EPOCH_LAYOUT,
            "row_axis": "epoch_windows",
        },
    )
    storage_receipt = payload["candidate_storage_receipt"]
    parsed_receipt = _require_storage_receipt_document(
        storage_receipt,
        expected_arrays=payload["expected_arrays"],
    )
    if (
        not isinstance(storage_receipt, Mapping)
        or storage_receipt.get("payload_digest")
        != payload["candidate_storage_receipt_payload_digest"]
    ):
        raise ValueError("Stimulus-epoch workload storage-receipt binding is invalid.")
    run_manifest = payload["candidate_run_manifest"]
    _require_run_manifest_document(
        run_manifest,
        source_run_name=source_name,
        source_run_path=payload["source_run_path"],
        candidate_run_name=candidate_name,
        candidate_run_path=payload["candidate_run_path"],
        source_lineage_hash=payload["source_lineage_hash"],
        source_lineage_payload_sha256=payload["source_lineage_payload_sha256"],
        candidate_lineage_hash=payload["candidate_lineage_hash"],
        candidate_lineage_payload_sha256=payload["candidate_lineage_payload_sha256"],
        candidate_lineage_payload=payload["candidate_lineage_payload"],
        candidate_materializer_identity=materializer_identity,
        logical_content=logical_content,
        storage_receipt=storage_receipt,
        storage_profile_id=parsed_receipt.profile.profile_id,
    )
    if (
        not isinstance(run_manifest, Mapping)
        or run_manifest.get("payload_digest")
        != payload["candidate_run_manifest_payload_digest"]
    ):
        raise ValueError("Stimulus-epoch workload run-manifest binding is invalid.")
    metadata = payload["metadata_equivalence"]
    if not isinstance(metadata, Mapping) or set(metadata) != {"source", "candidate"}:
        raise ValueError("Stimulus-epoch workload metadata receipts are malformed.")
    _require_metadata_receipt(metadata["source"], run_path=payload["source_run_path"])
    _require_metadata_receipt(
        metadata["candidate"], run_path=payload["candidate_run_path"]
    )
    _require_physical_io(payload["physical_io"])


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    return (
        ("candidate", "source")
        if (seed + repetition_index) % 2
        else ("source", "candidate")
    )


def _require_scan(value: object, *, expected_arrays: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "operation_count",
        "decoded_bytes",
        "arrays",
        "arrays_digest",
        "timing",
    }:
        raise ValueError("Stimulus-epoch full-scan receipt is malformed.")
    arrays = value["arrays"]
    if not isinstance(arrays, Mapping) or set(arrays) != set(ARRAY_PATHS):
        raise ValueError("Stimulus-epoch full-scan array inventory is not exact.")
    total = 0
    for path, record in arrays.items():
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "shape",
            "decoded_bytes",
            "sha256",
        }:
            raise ValueError(f"Stimulus-epoch scan record {path!r} is malformed.")
        expected = expected_arrays[path]
        expected_bytes = int(np.dtype(record["dtype"]).itemsize) * int(
            np.prod(record["shape"], dtype=np.int64)
        )
        if (
            record["dtype"] != expected["dtype"]
            or record["shape"] != expected["shape"]
            or record["sha256"] != expected["sha256"]
            or type(record["decoded_bytes"]) is not int
            or record["decoded_bytes"] != expected_bytes
        ):
            raise ValueError(
                f"Stimulus-epoch scan record {path!r} differs from workload."
            )
        total += record["decoded_bytes"]
    if (
        value["operation_count"] != len(ARRAY_PATHS)
        or value["decoded_bytes"] != total
        or value["arrays_digest"] != canonical_json_sha256(arrays)
    ):
        raise ValueError("Stimulus-epoch full-scan aggregate is invalid.")
    _require_timing(value["timing"], label="full scan")


def _require_storage(value: object) -> None:
    fields = {
        "file_count",
        "metadata_file_count",
        "payload_file_count",
        "apparent_bytes",
        "allocated_bytes",
        "object_count",
        "payload_object_count",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or any(type(value[field]) is not int or value[field] < 0 for field in fields)
        or value["object_count"] != value["file_count"]
        or value["payload_object_count"] != value["payload_file_count"]
        or value["metadata_file_count"] + value["payload_file_count"]
        != value["file_count"]
    ):
        raise ValueError("Stimulus-epoch storage receipt is invalid.")


def require_trial_result(
    value: Mapping[str, Any], *, workload: Mapping[str, Any]
) -> None:
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "seed",
        "cache_state",
        "workload_payload_digest",
        "process_id",
        "started_at_utc",
        "finished_at_utc",
        "environment",
        "validation",
        "full_scan",
        "storage",
        "runtime",
        "physical_io",
    }
    if set(payload) != expected:
        raise ValueError("Stimulus-epoch trial has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("Stimulus-epoch trial benchmark identity mismatch.")
    source_name = _safe_run_name(payload["source_run_name"], label="source_run")
    candidate_name = _safe_run_name(
        payload["candidate_run_name"], label="candidate_run"
    )
    if source_name == candidate_name:
        raise ValueError("Stimulus-epoch source and candidate names must differ.")
    role = payload["role"]
    if role not in {"source", "candidate"}:
        raise ValueError("Stimulus-epoch trial role is unsupported.")
    selected = source_name if role == "source" else candidate_name
    if (
        payload["run_name"] != selected
        or payload["run_path"] != f"{PARENT_PATH}/{selected}"
    ):
        raise ValueError("Stimulus-epoch trial role/run/path binding mismatch.")
    if (
        type(payload["archive_path"]) is not str
        or not Path(payload["archive_path"]).is_absolute()
        or type(payload["cache_state"]) is not str
        or not payload["cache_state"].strip()
        or type(payload["started_at_utc"]) is not str
        or type(payload["finished_at_utc"]) is not str
        or not payload["started_at_utc"]
        or payload["finished_at_utc"] < payload["started_at_utc"]
        or type(payload["repetition_index"]) is not int
        or payload["repetition_index"] < 0
        or payload["order_position"] not in {0, 1}
        or type(payload["seed"]) is not int
        or payload["seed"] < 0
        or _trial_order(
            seed=payload["seed"], repetition_index=payload["repetition_index"]
        )[payload["order_position"]]
        != role
        or type(payload["process_id"]) is not int
        or payload["process_id"] <= 0
        or not _is_sha256(payload["workload_payload_digest"])
    ):
        raise ValueError("Stimulus-epoch trial process/order identity is invalid.")
    _require_environment(payload["environment"], cache_state=payload["cache_state"])
    validation = payload["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "valid",
        "consumer_path",
        "logical_content_sha256",
        "segments_sha256",
        "source_lineage_hash",
        "candidate_storage_receipt_payload_digest",
        "candidate_run_manifest_payload_digest",
        "metadata_equivalence",
        "timing",
    }:
        raise ValueError("Stimulus-epoch trial validation receipt is malformed.")
    if (
        validation["valid"] is not True
        or validation["consumer_path"]
        != ("explicit_legacy_v1" if role == "source" else "strict_exact_v2")
        or not _is_sha256(validation["logical_content_sha256"])
        or not _is_sha256(validation["segments_sha256"])
        or not _is_sha256(validation["source_lineage_hash"])
    ):
        raise ValueError("Stimulus-epoch trial validation did not pass exactly.")
    for field in (
        "candidate_storage_receipt_payload_digest",
        "candidate_run_manifest_payload_digest",
    ):
        observed = validation[field]
        if role == "candidate":
            if not _is_sha256(observed):
                raise ValueError(f"Stimulus-epoch candidate {field} is invalid.")
        elif observed is not None:
            raise ValueError(f"Stimulus-epoch source {field} must be null.")
    _require_metadata_receipt(
        validation["metadata_equivalence"], run_path=payload["run_path"]
    )
    _require_timing(validation["timing"], label="validation")
    _require_storage(payload["storage"])
    _require_physical_io(payload["physical_io"])
    runtime = payload["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "initial_peak_rss_bytes",
        "final_peak_rss_bytes",
        "peak_rss_growth_bytes",
        "peak_rss_is_process_high_water_mark",
        "total_wall_seconds",
        "total_cpu_seconds",
    }:
        raise ValueError("Stimulus-epoch runtime receipt is malformed.")
    if (
        runtime["peak_rss_is_process_high_water_mark"] is not True
        or type(runtime["initial_peak_rss_bytes"]) is not int
        or type(runtime["final_peak_rss_bytes"]) is not int
        or type(runtime["peak_rss_growth_bytes"]) is not int
        or runtime["initial_peak_rss_bytes"] < 0
        or runtime["final_peak_rss_bytes"] < runtime["initial_peak_rss_bytes"]
        or runtime["peak_rss_growth_bytes"]
        != runtime["final_peak_rss_bytes"] - runtime["initial_peak_rss_bytes"]
    ):
        raise ValueError("Stimulus-epoch RSS receipt is invalid.")
    _require_timing(
        {
            "wall_seconds": runtime["total_wall_seconds"],
            "cpu_seconds": runtime["total_cpu_seconds"],
        },
        label="runtime",
    )
    require_workload(workload)
    workload_payload = workload["payload"]
    if (
        payload["archive_path"] != workload_payload["archive_path"]
        or source_name != workload_payload["source_run_name"]
        or candidate_name != workload_payload["candidate_run_name"]
        or payload["seed"] != workload_payload["seed"]
        or payload["workload_payload_digest"] != workload["payload_digest"]
        or validation["logical_content_sha256"]
        != workload_payload["expected_logical_content_sha256"]
        or validation["segments_sha256"] != workload_payload["expected_segments_sha256"]
        or validation["source_lineage_hash"] != workload_payload["source_lineage_hash"]
        or validation["metadata_equivalence"]
        != workload_payload["metadata_equivalence"][role]
    ):
        raise ValueError("Stimulus-epoch trial/workload identity binding mismatch.")
    if role == "candidate" and (
        validation["candidate_storage_receipt_payload_digest"]
        != workload_payload["candidate_storage_receipt_payload_digest"]
        or validation["candidate_run_manifest_payload_digest"]
        != workload_payload["candidate_run_manifest_payload_digest"]
    ):
        raise ValueError("Stimulus-epoch candidate trial receipt binding mismatch.")
    _require_scan(
        payload["full_scan"], expected_arrays=workload_payload["expected_arrays"]
    )


def run_single_trial(
    archive_path: Path | str,
    *,
    source_run: str,
    candidate_run: str,
    role: str,
    repetition_index: int,
    order_position: int,
    seed: int,
    cache_state: str,
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    archive = _safe_archive(archive_path)
    source_name = _safe_run_name(source_run, label="source_run")
    candidate_name = _safe_run_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("Stimulus-epoch source and candidate names must differ.")
    require_workload(workload)
    workload_payload = workload["payload"]
    if (
        workload_payload["archive_path"] != str(archive)
        or workload_payload["source_run_name"] != source_name
        or workload_payload["candidate_run_name"] != candidate_name
        or workload_payload["seed"] != seed
        or role not in {"source", "candidate"}
        or type(repetition_index) is not int
        or repetition_index < 0
        or order_position not in {0, 1}
        or repetition_index >= workload_payload["repetitions"]
        or _trial_order(seed=seed, repetition_index=repetition_index)[order_position]
        != role
    ):
        raise ValueError(
            "Stimulus-epoch trial differs from its workload/order binding."
        )
    if type(cache_state) is not str or not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    run_name = source_name if role == "source" else candidate_name
    selected_run_root = _safe_persisted_run_path(archive, run_name=run_name)
    run_path = f"{PARENT_PATH}/{run_name}"
    started = utc_now()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    initial_rss = peak_rss_bytes()

    def validate_and_open() -> tuple[Any, dict[str, Any]]:
        if role == "candidate":
            snapshot = read_stimulus_epoch_snapshot(archive, run_name=candidate_name)
            root = zarr.open_group(
                str(archive), mode="r", zarr_format=3, use_consolidated=True
            )
            group = _group(root, run_path)
            attrs = group.attrs
            receipt = attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
            manifest = attrs[STIMULUS_EPOCH_RUN_MANIFEST_ATTR]
            validation = {
                "valid": True,
                "consumer_path": "strict_exact_v2",
                "logical_content_sha256": stimulus_epoch_logical_content_sha256(group),
                "segments_sha256": _segments_sha256(snapshot.segments),
                "source_lineage_hash": attrs["source_stimulus_epoch_lineage_hash"],
                "candidate_storage_receipt_payload_digest": receipt["payload_digest"],
                "candidate_run_manifest_payload_digest": manifest["payload_digest"],
                "metadata_equivalence": snapshot.metadata_equivalence.to_json(),
            }
            return group, validation
        root = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=False
        )
        source = _group(root, run_path)
        source_validation = _validate_source(source, run_name=source_name)
        consolidated = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=True
        )
        group = _group(consolidated, run_path)
        validation = {
            "valid": True,
            "consumer_path": "explicit_legacy_v1",
            "logical_content_sha256": stimulus_epoch_logical_content_sha256(group),
            "segments_sha256": _segments_sha256(read_epoch_segments(group)),
            "source_lineage_hash": source_validation["lineage_hash"],
            "candidate_storage_receipt_payload_digest": None,
            "candidate_run_manifest_payload_digest": None,
            "metadata_equivalence": _metadata_receipt(archive, run_path=run_path),
        }
        return group, validation

    (group, validation), validation_timing = _measure(validate_and_open)
    validation["timing"] = validation_timing
    scan, scan_timing = _measure(lambda: _full_scan(group))
    scan["timing"] = scan_timing
    final_rss = peak_rss_bytes()
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "archive_path": str(archive),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "role": role,
        "run_name": run_name,
        "run_path": run_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": seed,
        "cache_state": cache_state,
        "workload_payload_digest": workload["payload_digest"],
        "process_id": os.getpid(),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "environment": _environment(archive=archive, cache_state=cache_state),
        "validation": validation,
        "full_scan": scan,
        "storage": _storage(selected_run_root),
        "runtime": {
            "initial_peak_rss_bytes": initial_rss,
            "final_peak_rss_bytes": final_rss,
            "peak_rss_growth_bytes": final_rss - initial_rss,
            "peak_rss_is_process_high_water_mark": True,
            "total_wall_seconds": float(time.perf_counter() - wall_started),
            "total_cpu_seconds": float(time.process_time() - cpu_started),
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _PHYSICAL_IO_AVAILABILITY,
        },
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, payload)
    require_trial_result(result, workload=workload)
    return result


def _median_by_role(
    trials: Sequence[Mapping[str, Any]], role: str, path: Sequence[str]
) -> float:
    values: list[float] = []
    for trial in trials:
        payload: Any = trial["payload"]
        if payload["role"] != role:
            continue
        for component in path:
            payload = payload[component]
        values.append(float(payload))
    if not values:
        raise ValueError(f"Stimulus-epoch matrix has no {role!r} trials.")
    return float(statistics.median(values))


def _matrix_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role in ("source", "candidate"):
        role_trials = [trial for trial in trials if trial["payload"]["role"] == role]
        if not role_trials:
            raise ValueError(f"Stimulus-epoch matrix has no {role!r} trials.")
        storage = role_trials[0]["payload"]["storage"]
        if any(trial["payload"]["storage"] != storage for trial in role_trials[1:]):
            raise ValueError("Stimulus-epoch storage inventory changed between trials.")
        summary[role] = {
            "median_validation_wall_seconds": _median_by_role(
                trials, role, ("validation", "timing", "wall_seconds")
            ),
            "median_validation_cpu_seconds": _median_by_role(
                trials, role, ("validation", "timing", "cpu_seconds")
            ),
            "median_full_scan_wall_seconds": _median_by_role(
                trials, role, ("full_scan", "timing", "wall_seconds")
            ),
            "median_full_scan_cpu_seconds": _median_by_role(
                trials, role, ("full_scan", "timing", "cpu_seconds")
            ),
            "median_total_wall_seconds": _median_by_role(
                trials, role, ("runtime", "total_wall_seconds")
            ),
            "median_total_cpu_seconds": _median_by_role(
                trials, role, ("runtime", "total_cpu_seconds")
            ),
            "median_peak_rss_bytes": _median_by_role(
                trials, role, ("runtime", "final_peak_rss_bytes")
            ),
            "object_count": storage["object_count"],
            "payload_object_count": storage["payload_object_count"],
            "apparent_bytes": storage["apparent_bytes"],
            "allocated_bytes": storage["allocated_bytes"],
        }
    return summary


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "seed",
        "repetitions",
        "cache_state",
        "driver_process_id",
        "started_at_utc",
        "finished_at_utc",
        "workload",
        "workload_file",
        "trial_order",
        "trial_files",
        "trials",
        "correctness",
        "performance_summary",
        "archive_read_only_metadata_guard",
        "physical_io",
        "benchmark_only",
        "balanced_fresh_process_matrix_complete",
    }
    if set(payload) != expected:
        raise ValueError("Stimulus-epoch matrix has an unexpected field set.")
    workload = payload["workload"]
    if not isinstance(workload, Mapping):
        raise ValueError("Stimulus-epoch matrix workload is absent.")
    require_workload(workload)
    workload_payload = workload["payload"]
    source_name = _safe_run_name(payload["source_run_name"], label="source_run")
    candidate_name = _safe_run_name(
        payload["candidate_run_name"], label="candidate_run"
    )
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or type(payload["archive_path"]) is not str
        or not Path(payload["archive_path"]).is_absolute()
        or source_name == candidate_name
        or type(payload["seed"]) is not int
        or payload["seed"] < 0
        or type(payload["repetitions"]) is not int
        or payload["repetitions"] < 1
        or type(payload["cache_state"]) is not str
        or not payload["cache_state"].strip()
        or type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
        or type(payload["started_at_utc"]) is not str
        or type(payload["finished_at_utc"]) is not str
        or not payload["started_at_utc"]
        or payload["finished_at_utc"] < payload["started_at_utc"]
        or payload["workload_file"] != "read_workload.json"
        or payload["archive_path"] != workload_payload["archive_path"]
        or source_name != workload_payload["source_run_name"]
        or candidate_name != workload_payload["candidate_run_name"]
        or payload["seed"] != workload_payload["seed"]
        or payload["repetitions"] != workload_payload["repetitions"]
    ):
        raise ValueError("Stimulus-epoch matrix/workload identity is invalid.")

    expected_order = [
        {
            "repetition_index": repetition_index,
            "roles": list(
                _trial_order(seed=payload["seed"], repetition_index=repetition_index)
            ),
        }
        for repetition_index in range(payload["repetitions"])
    ]
    if payload["trial_order"] != expected_order:
        raise ValueError("Stimulus-epoch trial order is not deterministic v1.")
    trials = payload["trials"]
    trial_files = payload["trial_files"]
    if (
        not isinstance(trials, list)
        or len(trials) != 2 * payload["repetitions"]
        or not isinstance(trial_files, list)
        or len(trial_files) != len(trials)
        or len(set(trial_files)) != len(trial_files)
    ):
        raise ValueError("Stimulus-epoch trial/file inventory is invalid.")
    expected_files: list[str] = []
    expected_roles: list[tuple[int, int, str]] = []
    for record in expected_order:
        repetition_index = record["repetition_index"]
        for order_position, role in enumerate(record["roles"]):
            expected_files.append(
                f"trials/rep_{repetition_index:02d}_pos_{order_position}_{role}.json"
            )
            expected_roles.append((repetition_index, order_position, role))
    if trial_files != expected_files:
        raise ValueError("Stimulus-epoch trial-file names are not exact.")

    process_ids: list[int] = []
    for expected_role, trial in zip(expected_roles, trials, strict=True):
        if not isinstance(trial, Mapping):
            raise ValueError("Stimulus-epoch matrix trial must be one object.")
        require_trial_result(trial, workload=workload)
        trial_payload = trial["payload"]
        if (
            trial_payload["repetition_index"],
            trial_payload["order_position"],
            trial_payload["role"],
        ) != expected_role or trial_payload["cache_state"] != payload["cache_state"]:
            raise ValueError("Stimulus-epoch matrix trial binding is invalid.")
        process_ids.append(trial_payload["process_id"])
    if (
        len(set(process_ids)) != len(process_ids)
        or payload["driver_process_id"] in process_ids
    ):
        raise ValueError("Stimulus-epoch trials did not use distinct fresh processes.")

    correctness = payload["correctness"]
    expected_correctness = {
        "complete_decoded_array_equality": True,
        "decoded_segment_equality": True,
        "direct_consolidated_metadata_equivalence": True,
        "strict_source_and_candidate_validation": True,
        "candidate_publication_receipt_and_manifest_bound": True,
        "archive_metadata_unchanged": True,
        "all_passed": True,
    }
    if correctness != expected_correctness:
        raise ValueError("Stimulus-epoch correctness receipt is not an exact pass.")
    if payload["performance_summary"] != _matrix_summary(trials):
        raise ValueError("Stimulus-epoch performance summary is not executable.")

    guard = payload["archive_read_only_metadata_guard"]
    if not isinstance(guard, Mapping) or set(guard) != {
        "before",
        "after",
        "unchanged",
    }:
        raise ValueError("Stimulus-epoch archive metadata guard is malformed.")
    _require_metadata_guard(guard["before"])
    _require_metadata_guard(guard["after"])
    if guard["unchanged"] is not True or guard["before"] != guard["after"]:
        raise ValueError("Stimulus-epoch archive metadata changed during benchmark.")
    _require_physical_io(payload["physical_io"])
    if payload["benchmark_only"] != {
        "selector_or_profile_change_authorized": False,
        "reason": (
            "read-only evidence cannot authorize writer, selector, registry, "
            "or physical-profile promotion"
        ),
    }:
        raise ValueError("Stimulus-epoch benchmark-only boundary is invalid.")
    if payload["balanced_fresh_process_matrix_complete"] is not (
        payload["repetitions"] == DEFAULT_REPETITIONS
    ):
        raise ValueError("Stimulus-epoch balanced-matrix classification is invalid.")


def run_benchmark_matrix(
    archive_path: Path | str,
    *,
    source_run: str,
    candidate_run: str,
    output_dir: Path | str,
    cache_state: str,
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Run an immutable, read-only, fresh-process source/candidate matrix."""

    archive = _safe_archive(archive_path)
    source_name = _safe_run_name(source_run, label="source_run")
    candidate_name = _safe_run_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("Stimulus-epoch source and candidate names must differ.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be one nonnegative exact integer.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be one positive exact integer.")
    if type(cache_state) is not str or not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    output = _safe_output(output_dir, archive=archive)
    preflight = _preflight(
        archive,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
        seed=seed,
        repetitions=repetitions,
    )
    workload = preflight["workload"]
    guard_before = _metadata_guard(archive, run_names=(source_name, candidate_name))
    output.mkdir(parents=True, exist_ok=False)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    workload_path = output / "read_workload.json"
    _write_json(workload_path, workload)

    started = utc_now()
    trials: list[Mapping[str, Any]] = []
    trial_order: list[dict[str, Any]] = []
    trial_files: list[str] = []
    environment = os.environ.copy()
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    repository = Path(__file__).resolve().parents[3]
    source_path = str(repository / "src")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not prior_pythonpath
        else os.pathsep.join((source_path, prior_pythonpath))
    )
    for repetition_index in range(repetitions):
        order = _trial_order(seed=seed, repetition_index=repetition_index)
        trial_order.append({"repetition_index": repetition_index, "roles": list(order)})
        for order_position, role in enumerate(order):
            filename = f"rep_{repetition_index:02d}_pos_{order_position}_{role}.json"
            trial_path = trials_dir / filename
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_stimulus_epoch_reads",
                "trial",
                str(archive),
                "--source-run",
                source_name,
                "--candidate-run",
                candidate_name,
                "--role",
                role,
                "--repetition-index",
                str(repetition_index),
                "--order-position",
                str(order_position),
                "--seed",
                str(seed),
                "--cache-state",
                cache_state,
                "--workload-file",
                str(workload_path),
                "--benchmark-root",
                str(output),
                "--output-file",
                str(trial_path),
            ]
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Fresh-process stimulus-epoch trial failed: "
                    f"command={command!r}, stdout={completed.stdout!r}, "
                    f"stderr={completed.stderr!r}."
                )
            trial = _read_json(trial_path)
            require_trial_result(trial, workload=workload)
            trials.append(trial)
            trial_files.append(str(trial_path.relative_to(output)))

    guard_after = _metadata_guard(archive, run_names=(source_name, candidate_name))
    if guard_before != guard_after:
        raise RuntimeError("Archive metadata changed during read-only benchmark.")
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "archive_path": str(archive),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": cache_state,
        "driver_process_id": os.getpid(),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "workload": workload,
        "workload_file": "read_workload.json",
        "trial_order": trial_order,
        "trial_files": trial_files,
        "trials": trials,
        "correctness": {
            "complete_decoded_array_equality": True,
            "decoded_segment_equality": True,
            "direct_consolidated_metadata_equivalence": True,
            "strict_source_and_candidate_validation": True,
            "candidate_publication_receipt_and_manifest_bound": True,
            "archive_metadata_unchanged": True,
            "all_passed": True,
        },
        "performance_summary": _matrix_summary(trials),
        "archive_read_only_metadata_guard": {
            "before": guard_before,
            "after": guard_after,
            "unchanged": True,
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _PHYSICAL_IO_AVAILABILITY,
        },
        "benchmark_only": {
            "selector_or_profile_change_authorized": False,
            "reason": (
                "read-only evidence cannot authorize writer, selector, registry, "
                "or physical-profile promotion"
            ),
        },
        "balanced_fresh_process_matrix_complete": (repetitions == DEFAULT_REPETITIONS),
    }
    result = _strict_envelope(MATRIX_SCHEMA_ID, payload)
    require_matrix_result(result)
    _write_json(output / "matrix_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="Run the fresh-process matrix.")
    trial = subparsers.add_parser("trial", help="Run one internal trial process.")
    for child in (matrix, trial):
        child.add_argument("zarr_path", type=Path)
        child.add_argument("--source-run", required=True)
        child.add_argument("--candidate-run", required=True)
        child.add_argument("--seed", type=int, default=DEFAULT_SEED)
        child.add_argument("--cache-state", required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, choices=(0, 1), required=True)
    trial.add_argument("--workload-file", type=Path, required=True)
    trial.add_argument("--benchmark-root", type=Path, required=True)
    trial.add_argument("--output-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        result = run_benchmark_matrix(
            args.zarr_path,
            source_run=args.source_run,
            candidate_run=args.candidate_run,
            output_dir=args.output_dir,
            cache_state=args.cache_state,
            seed=args.seed,
            repetitions=args.repetitions,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "matrix_result": str(
                        Path(args.output_dir).expanduser().resolve()
                        / "matrix_result.json"
                    ),
                    "payload_digest": result["payload_digest"],
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    benchmark_root = args.benchmark_root.expanduser().resolve()
    workload = _read_json(args.workload_file.expanduser().resolve())
    require_workload(workload)
    output = _safe_trial_output(args.output_file, benchmark_root=benchmark_root)
    result = run_single_trial(
        args.zarr_path,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        seed=args.seed,
        cache_state=args.cache_state,
        workload=workload,
    )
    _write_json(output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "DEFAULT_REPETITIONS",
    "DEFAULT_SEED",
    "MATRIX_SCHEMA_ID",
    "TRIAL_SCHEMA_ID",
    "WORKLOAD_SCHEMA_ID",
    "require_matrix_result",
    "require_trial_result",
    "require_workload",
    "run_benchmark_matrix",
    "run_single_trial",
]
