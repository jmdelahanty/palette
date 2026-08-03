"""Read-only benchmark for one explicit stimulus-response compact-v3 pair.

The source is the exact compatible compact-v3 layout and the candidate is the
selector-ineligible byte-planned layout.  Each role is read in a fresh process;
evidence is written only below a new benchmark-labelled directory outside the
archive.  This harness intentionally records no physical I/O counts unless an
external tracer supplies them in a future evidence schema.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
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
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import zarr

from fisheye.analysis.stimulus_response_io import (
    resolve_stimulus_response_v3_tables,
)
from fisheye.analysis.stimulus_response_storage import (
    STIMULUS_RESPONSE_CANDIDATE_ATTR,
    STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
    STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
    STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_ID,
    STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_VERSION,
    STIMULUS_RESPONSE_METADATA_NORMALIZATION,
    STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR,
    STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR,
    STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR,
    build_stimulus_response_storage_receipt,
    stimulus_response_fill_values,
    validate_stimulus_response_metadata_equivalence,
    validate_stimulus_response_storage_receipt,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.array_factory import array_metadata_declaration_from_plan
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.manifest_digest import (
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.metadata_equivalence import (
    METADATA_EQUIVALENCE_SCHEMA_ID,
    METADATA_EQUIVALENCE_SCHEMA_VERSION,
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from fisheye.shared.zarr.stimulus_response_schema import (
    STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR,
    STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR,
    expected_table_names,
    stimulus_response_array_declarations,
    stimulus_response_array_manifest,
    validate_stimulus_response_v3_run,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)

PARENT_PATH = "analysis/stimulus_response_runs"
FAMILY_ID = "stimulus_response_compact_v3"
BENCHMARK_ID = "stimulus_response_source_candidate_reads_v1"
WORKLOAD_ID = "stimulus_response_complete_tables_v1"
WORKLOAD_SCHEMA_ID = "palette.stimulus_response.read_workload"
TRIAL_SCHEMA_ID = "palette.stimulus_response.read_trial"
MATRIX_SCHEMA_ID = "palette.stimulus_response.read_matrix"
SCHEMA_VERSION = 1
DEFAULT_SEED = 31
DEFAULT_REPETITIONS = 5
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_PHYSICAL_IO_AVAILABILITY = "not_collected_requires_external_trace"
_SHA256_LENGTH = 64
_CANDIDATE_ATTRS = frozenset(
    {
        STIMULUS_RESPONSE_CANDIDATE_ATTR,
        STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
        STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR,
        STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR,
        STIMULUS_RESPONSE_STORAGE_PROFILE_ID_ATTR,
        STIMULUS_RESPONSE_STORAGE_PROFILE_ROLE_ATTR,
    }
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


def _strict_envelope(schema_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _strict_json_copy(dict(payload))
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
        raise ValueError("Stimulus-response evidence envelope field set is not exact.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Stimulus-response evidence schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Stimulus-response evidence payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Stimulus-response evidence payload digest mismatch.")
    _strict_json_copy(value)
    return payload


def _strict_json_copy(value: object) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stimulus-response evidence is not strict JSON: {exc}"
        ) from exc
    return json.loads(
        encoded,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token {token}")
        ),
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_run_name(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value in _ALIASES
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return value


def _safe_archive(value: Path | str) -> Path:
    archive = Path(value).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive does not exist: {archive}.")
    return archive


def _run_path(run_name: str) -> str:
    return f"{PARENT_PATH}/{run_name}"


def _safe_persisted_run_path(archive: Path, *, run_name: str) -> Path:
    path = archive.joinpath(*_run_path(run_name).split("/"))
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            "Selected stimulus-response run must be a nonsymlink directory."
        )
    return path


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
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("Benchmark output path is too broad.")
    if not any("benchmark" in part.lower() for part in output.parts):
        raise ValueError("Output path must be explicitly benchmark-only.")
    return output


def _safe_trial_output(value: Path | str, *, benchmark_root: Path) -> Path:
    output = Path(value).expanduser().resolve()
    root = benchmark_root.expanduser().resolve()
    if not root.is_dir() or output.exists():
        raise ValueError("Trial output requires an existing root and a new file.")
    if not output.is_relative_to(root) or output.suffix != ".json":
        raise ValueError(
            "Trial output must be a new JSON file below the benchmark root."
        )
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace benchmark evidence: {path}.")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"Strict JSON document is not one object: {path}.")
    return value


def _group(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _array(run_group: Any, path: str) -> Any:
    return _group(run_group, path)


def _open_root(archive: Path, *, consolidated: bool) -> Any:
    return zarr.open_group(
        str(archive),
        mode="r",
        use_consolidated=consolidated,
    )


def _measure(call: Any) -> tuple[Any, dict[str, float]]:
    started = time.perf_counter_ns()
    value = call()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return value, {"elapsed_ms": elapsed_ms}


def _require_timing(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"elapsed_ms"}:
        raise ValueError(f"{label} timing field set is not exact.")
    elapsed = value["elapsed_ms"]
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{label} timing is invalid.")


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _scan_array(array: Any) -> dict[str, Any]:
    values = np.asarray(array[:])
    return {
        "dtype": np.dtype(values.dtype).str,
        "shape": [int(value) for value in values.shape],
        "decoded_nbytes": int(values.nbytes),
        "decoded_sha256": _sha256_array(values),
    }


def _full_scan(run_group: Any, *, array_paths: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter_ns()
    arrays = {path: _scan_array(_array(run_group, path)) for path in array_paths}
    return {
        "array_count": len(arrays),
        "decoded_nbytes": sum(item["decoded_nbytes"] for item in arrays.values()),
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "arrays": arrays,
        "arrays_digest": canonical_json_sha256(arrays),
    }


def _logical_value(value: Any, *, include_attrs: bool = True) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": np.dtype(value.dtype).str,
            "shape": [int(extent) for extent in value.shape],
            "decoded_sha256": _sha256_array(value),
        }
    if isinstance(value, np.generic):
        return _logical_value(value.item(), include_attrs=include_attrs)
    if is_dataclass(value):
        return {
            item.name: _logical_value(
                getattr(value, item.name),
                include_attrs=include_attrs,
            )
            for item in fields(value)
            if include_attrs or item.name != "attrs"
        }
    if isinstance(value, Mapping):
        return {
            str(key): _logical_value(child, include_attrs=include_attrs)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_logical_value(child, include_attrs=include_attrs) for child in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            return {"palette_exact_float": "nan" if math.isnan(value) else str(value)}
        return value
    raise TypeError(f"Unsupported logical reader value {type(value).__name__}.")


def _logical_reader_document(tables: Any) -> dict[str, Any]:
    # Run attrs contain the intentionally different physical candidate envelope.
    # Step/metric attrs are also omitted so this digest covers the maintained
    # table payload rather than physical metadata already bound elsewhere.
    document = _logical_value(tables, include_attrs=False)
    if not isinstance(document, dict):
        raise TypeError("Stimulus-response logical reader did not return a document.")
    return document


def _metadata_receipt(archive: Path, *, run_path: str) -> dict[str, Any]:
    return validate_direct_consolidated_subtree(
        archive,
        subtree_path=run_path,
    ).to_json()


def _direct_metadata_declarations(
    archive: Path,
    *,
    run_path: str,
) -> dict[str, dict[str, Any]]:
    run_root = archive.joinpath(*run_path.split("/"))
    declarations: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(run_root.rglob("zarr.json")):
        if metadata_path.is_symlink():
            raise ValueError("Stimulus-response metadata declaration is a symlink.")
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("Stimulus-response Zarr declaration is not one object.")
        node_path = metadata_path.parent.relative_to(archive).as_posix()
        declarations[node_path] = metadata_without_empty_group_consolidation(
            dict(raw),
            path=node_path,
        )
    return declarations


def _normalized_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_metadata_value(child)
            for key, child in value.items()
            if key
            not in {
                "consolidated_metadata",
                STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR,
            }
        }
    if isinstance(value, (tuple, list)):
        return [_normalized_metadata_value(child) for child in value]
    if value == "NaN":
        return {"palette_exact_float": "nan"}
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return {"palette_exact_float": "nan"}
    return value


def _candidate_normalized_metadata_document(
    value: Mapping[str, Any],
    *,
    run_path: str,
    bundles: Sequence[str],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for relative_path in ("", *expected_table_names(bundles)):
        path = run_path if not relative_path else f"{run_path}/{relative_path}"
        declaration = value.get(path)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("node_type") != "group"
        ):
            raise ValueError(f"Candidate group declaration is missing {path!r}.")
        groups[relative_path] = _normalized_metadata_value(declaration)
    arrays: dict[str, Any] = {}
    for item in stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=True,
    ):
        path = f"{run_path}/{item.path}"
        declaration = value.get(path)
        if (
            not isinstance(declaration, Mapping)
            or declaration.get("node_type") != "array"
        ):
            raise ValueError(f"Candidate array declaration is missing {path!r}.")
        arrays[item.path] = _normalized_metadata_value(declaration)
    return {
        "schema_id": "palette.stimulus_response.normalized_zarr_metadata",
        "schema_version": 1,
        "normalization": STIMULUS_RESPONSE_METADATA_NORMALIZATION,
        "groups": groups,
        "arrays": arrays,
    }


def _require_run_group_bindings(
    source_value: Mapping[str, Any],
    candidate_value: Mapping[str, Any],
    *,
    source_run_path: str,
    candidate_run_path: str,
    source_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    candidate_envelope: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
    candidate_equivalence: Mapping[str, Any],
) -> None:
    source = source_value.get(source_run_path)
    candidate = candidate_value.get(candidate_run_path)
    if not isinstance(source, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("Run-group metadata declarations are missing.")
    source_attrs = source.get("attributes")
    candidate_attrs = candidate.get("attributes")
    if not isinstance(source_attrs, Mapping) or not isinstance(
        candidate_attrs, Mapping
    ):
        raise ValueError("Run-group metadata attributes are invalid.")
    if (
        source_attrs.get(STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR) != source_manifest
        or source_attrs.get("stage_selector_eligible") is not True
        or any(name in source_attrs for name in _CANDIDATE_ATTRS)
    ):
        raise ValueError("Source run-group metadata binding is invalid.")
    if (
        candidate_attrs.get(STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR) != candidate_manifest
        or candidate_attrs.get(STIMULUS_RESPONSE_CANDIDATE_ATTR) != candidate_envelope
        or candidate_attrs.get(STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR)
        != candidate_receipt
        or candidate_attrs.get(STIMULUS_RESPONSE_STORAGE_PLAN_DIGEST_ATTR)
        != candidate_receipt.get("payload_digest")
        or candidate_attrs.get(STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR)
        != candidate_equivalence
        or candidate_attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Candidate run-group metadata binding is invalid.")


def _require_metadata_declarations(
    value: object,
    *,
    receipt: Mapping[str, Any],
    run_path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Metadata declaration evidence must be nonempty.")
    prefix = f"{run_path}/"
    group_count = 0
    array_count = 0
    for path, declaration in value.items():
        if (
            type(path) is not str
            or (path != run_path and not path.startswith(prefix))
            or not isinstance(declaration, Mapping)
        ):
            raise ValueError("Metadata declaration path/document is invalid.")
        node_type = declaration.get("node_type")
        if node_type == "group":
            group_count += 1
        elif node_type == "array":
            array_count += 1
        else:
            raise ValueError("Metadata declaration node type is invalid.")
    if (
        len(value) != receipt["node_count"]
        or group_count != receipt["group_count"]
        or array_count != receipt["array_count"]
        or canonical_json_sha256(value) != receipt["declarations_sha256"]
    ):
        raise ValueError("Metadata declarations differ from their persisted receipt.")
    return value


def _require_candidate_physical_declarations(
    value: Mapping[str, Any],
    *,
    run_path: str,
    bundles: Sequence[str],
    receipt: Mapping[str, Any],
) -> None:
    parsed = analysis_storage_plan_receipt_from_manifest(receipt)
    fills = stimulus_response_fill_values(bundles=bundles)
    entries = {entry.declaration.path: entry for entry in parsed.entries}
    for relative_path, entry in entries.items():
        path = f"{run_path}/{relative_path}"
        raw = value.get(path)
        if not isinstance(raw, Mapping) or raw.get("node_type") != "array":
            raise ValueError(f"Candidate metadata declaration is missing {path!r}.")
        attributes = raw.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError(f"Candidate array attributes are invalid at {path!r}.")
        nonreserved = {
            str(key): child
            for key, child in attributes.items()
            if key not in _RESERVED_ARRAY_ATTRIBUTES
        }
        expected = array_metadata_declaration_from_plan(
            contract=entry.declaration.contract,
            plan=entry.plan,
            fill_value=fills[relative_path],
            attributes=nonreserved,
        )
        observed = {
            key: child
            for key, child in raw.items()
            if key not in {"zarr_format", "node_type", "consolidated_metadata"}
        }
        if _normalized_metadata_value(observed) != _normalized_metadata_value(expected):
            raise ValueError(
                f"Candidate metadata differs from executable storage plan at {path!r}."
            )


def _metadata_guard(archive: Path, *, run_names: Sequence[str]) -> dict[str, Any]:
    paths = {archive / "zarr.json"}
    for relative in ("analysis", PARENT_PATH):
        path = archive.joinpath(*relative.split("/"), "zarr.json")
        if path.is_file():
            paths.add(path)
    for run_name in run_names:
        run_root = _safe_persisted_run_path(archive, run_name=run_name)
        for path in run_root.rglob("zarr.json"):
            if path.is_symlink():
                raise ValueError("Stimulus-response metadata guard rejects symlinks.")
            paths.add(path)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        stat = path.stat()
        files[path.relative_to(archive).as_posix()] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"files": files, "digest": canonical_json_sha256(files)}


def _require_metadata_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"files", "digest"}:
        raise ValueError("Metadata guard field set is not exact.")
    files = value["files"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Metadata guard must contain files.")
    for path, facts in files.items():
        if (
            type(path) is not str
            or not isinstance(facts, Mapping)
            or set(facts)
            != {
                "size",
                "mtime_ns",
                "sha256",
            }
        ):
            raise ValueError("Metadata guard file entry is invalid.")
        if type(facts["size"]) is not int or facts["size"] < 0:
            raise ValueError("Metadata guard size is invalid.")
        if type(facts["mtime_ns"]) is not int or facts["mtime_ns"] < 0:
            raise ValueError("Metadata guard mtime is invalid.")
        if not _is_sha256(facts["sha256"]):
            raise ValueError("Metadata guard digest is invalid.")
    if value["digest"] != canonical_json_sha256(files):
        raise ValueError("Metadata guard digest mismatch.")


def _environment(*, archive: Path, cache_state: str) -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "archive_filesystem": str(archive.stat().st_dev),
        "cache_state": cache_state,
        "thread_environment": {
            name: os.environ.get(name) for name in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _require_environment(value: object, *, cache_state: str) -> None:
    fields = {
        "hostname",
        "system",
        "release",
        "python",
        "numpy",
        "zarr",
        "archive_filesystem",
        "cache_state",
        "thread_environment",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Benchmark environment field set is not exact.")
    if value["cache_state"] != cache_state:
        raise ValueError("Benchmark cache-state binding mismatch.")
    thread_environment = value["thread_environment"]
    if not isinstance(thread_environment, Mapping) or set(thread_environment) != set(
        STORAGE_BENCHMARK_THREAD_ENVIRONMENT
    ):
        raise ValueError("Benchmark thread environment is incomplete.")
    if any(
        item is not None and type(item) is not str
        for item in thread_environment.values()
    ):
        raise ValueError("Benchmark thread environment value is invalid.")


def _exact_candidate_envelope() -> dict[str, Any]:
    return {
        "schema_id": "palette.stimulus_response.storage_candidate",
        "schema_version": 1,
        "profile_id": STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
        "status": "unpromoted_selector_ineligible",
        "write_ownership": "serial_single_writer_whole_shard",
    }


def _require_completion(group: Any, *, run_name: str, selector_eligible: bool) -> None:
    attrs = dict(group.attrs)
    expected = {
        RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
        RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        RUN_NAME_ATTR: run_name,
        "stage_selector_eligible": selector_eligible,
    }
    for name, value in expected.items():
        if attrs.get(name) != value:
            raise ValueError(
                f"Stimulus-response run completion binding {name!r} differs."
            )


def _validate_source(group: Any, *, run_name: str) -> dict[str, Any]:
    errors = validate_stimulus_response_v3_run(group)
    if errors:
        raise ValueError("Invalid compact-v3 source: " + "; ".join(errors))
    attrs = dict(group.attrs)
    _require_completion(group, run_name=run_name, selector_eligible=True)
    present = sorted(_CANDIDATE_ATTRS.intersection(attrs))
    if present:
        raise ValueError(f"Source unexpectedly carries candidate markers: {present!r}.")
    bundles = attrs.get("stimulus_response_v3_bundles")
    if attrs.get(
        STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR
    ) != stimulus_response_array_manifest(
        bundles=bundles,
        byte_planner_adopted=False,
    ):
        raise ValueError("Source array schema manifest is not exact executable v1.")
    tables = resolve_stimulus_response_v3_tables(group)
    return {
        "consumer_path": "strict_compact_v3_source",
        "bundles": list(bundles),
        "logical_reader_digest": canonical_json_sha256(
            _logical_reader_document(tables)
        ),
    }


def _validate_candidate(group: Any, *, run_name: str) -> dict[str, Any]:
    errors = validate_stimulus_response_v3_run(group)
    errors = (
        *errors,
        *validate_stimulus_response_storage_receipt(group),
        *validate_stimulus_response_metadata_equivalence(group),
    )
    if errors:
        raise ValueError("Invalid compact-v3 candidate: " + "; ".join(errors))
    _require_completion(group, run_name=run_name, selector_eligible=False)
    attrs = dict(group.attrs)
    if attrs.get(STIMULUS_RESPONSE_CANDIDATE_ATTR) != _exact_candidate_envelope():
        raise ValueError("Candidate envelope is not exact.")
    bundles = attrs.get("stimulus_response_v3_bundles")
    if attrs.get(
        STIMULUS_RESPONSE_ARRAY_SCHEMA_ATTR
    ) != stimulus_response_array_manifest(
        bundles=bundles,
        byte_planner_adopted=True,
    ):
        raise ValueError("Candidate array schema manifest is not exact executable v2.")
    tables = resolve_stimulus_response_v3_tables(group)
    return {
        "consumer_path": "strict_byte_planned_compact_v3_candidate",
        "bundles": list(bundles),
        "logical_reader_digest": canonical_json_sha256(
            _logical_reader_document(tables)
        ),
        "storage_receipt_payload_digest": attrs[
            STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR
        ]["payload_digest"],
        "metadata_equivalence_payload_digest": attrs[
            STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR
        ]["payload_digest"],
    }


def _preflight(
    archive_value: Path | str,
    *,
    source_run_name: str,
    candidate_run_name: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    archive = _safe_archive(archive_value)
    source_name = _safe_run_name(source_run_name, label="source run")
    candidate_name = _safe_run_name(candidate_run_name, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")
    if type(seed) is not int or type(repetitions) is not int or repetitions < 1:
        raise ValueError("Seed and positive repetition count must be exact integers.")
    _safe_persisted_run_path(archive, run_name=source_name)
    _safe_persisted_run_path(archive, run_name=candidate_name)
    direct_root = _open_root(archive, consolidated=False)
    consolidated_root = _open_root(archive, consolidated=True)
    source_direct = _group(direct_root, _run_path(source_name))
    candidate_direct = _group(direct_root, _run_path(candidate_name))
    source_consolidated = _group(consolidated_root, _run_path(source_name))
    candidate_consolidated = _group(consolidated_root, _run_path(candidate_name))
    source_validation = _validate_source(source_direct, run_name=source_name)
    candidate_validation = _validate_candidate(
        candidate_direct, run_name=candidate_name
    )
    source_consolidated_validation = _validate_source(
        source_consolidated,
        run_name=source_name,
    )
    candidate_consolidated_validation = _validate_candidate(
        candidate_consolidated,
        run_name=candidate_name,
    )
    if source_validation != source_consolidated_validation:
        raise ValueError("Source direct/consolidated consumer results differ.")
    if candidate_validation != candidate_consolidated_validation:
        raise ValueError("Candidate direct/consolidated consumer results differ.")
    if source_validation["bundles"] != candidate_validation["bundles"]:
        raise ValueError("Source and candidate bundle sets differ.")
    if (
        source_validation["logical_reader_digest"]
        != candidate_validation["logical_reader_digest"]
    ):
        raise ValueError("Source and candidate maintained-reader results differ.")
    bundles = source_validation["bundles"]
    declarations = stimulus_response_array_declarations(
        bundles=bundles,
        byte_planner_adopted=False,
    )
    array_paths = tuple(item.path for item in declarations)
    source_scan = _full_scan(source_direct, array_paths=array_paths)
    candidate_scan = _full_scan(candidate_direct, array_paths=array_paths)
    if source_scan["arrays"] != candidate_scan["arrays"]:
        raise ValueError("Source and candidate decoded array values differ.")
    source_metadata = _metadata_receipt(archive, run_path=_run_path(source_name))
    candidate_metadata = _metadata_receipt(archive, run_path=_run_path(candidate_name))
    source_metadata_declarations = _direct_metadata_declarations(
        archive,
        run_path=_run_path(source_name),
    )
    candidate_metadata_declarations = _direct_metadata_declarations(
        archive,
        run_path=_run_path(candidate_name),
    )
    candidate_attrs = dict(candidate_direct.attrs)
    payload = {
        "family_id": FAMILY_ID,
        "benchmark_id": BENCHMARK_ID,
        "workload_id": WORKLOAD_ID,
        "archive": str(archive),
        "parent_path": PARENT_PATH,
        "source_run": source_name,
        "candidate_run": candidate_name,
        "source_run_path": _run_path(source_name),
        "candidate_run_path": _run_path(candidate_name),
        "seed": seed,
        "repetitions": repetitions,
        "bundles": bundles,
        "access": {
            "mode": "strict_reader_then_complete_array_scan",
            "array_order": list(array_paths),
            "operation_count": len(array_paths),
        },
        "source_array_schema_manifest": stimulus_response_array_manifest(
            bundles=bundles,
            byte_planner_adopted=False,
        ),
        "candidate_array_schema_manifest": stimulus_response_array_manifest(
            bundles=bundles,
            byte_planner_adopted=True,
        ),
        "candidate_storage_receipt": _strict_json_copy(
            candidate_attrs[STIMULUS_RESPONSE_STORAGE_PLAN_RECEIPT_ATTR]
        ),
        "candidate_metadata_equivalence_receipt": _strict_json_copy(
            candidate_attrs[STIMULUS_RESPONSE_METADATA_EQUIVALENCE_ATTR]
        ),
        "candidate_envelope": _strict_json_copy(
            candidate_attrs[STIMULUS_RESPONSE_CANDIDATE_ATTR]
        ),
        "source_metadata_equivalence": source_metadata,
        "candidate_metadata_equivalence": candidate_metadata,
        "source_metadata_declarations": source_metadata_declarations,
        "candidate_metadata_declarations": candidate_metadata_declarations,
        "expected_arrays": source_scan["arrays"],
        "expected_arrays_digest": source_scan["arrays_digest"],
        "expected_decoded_nbytes": source_scan["decoded_nbytes"],
        "logical_reader_digest": source_validation["logical_reader_digest"],
        "candidate_storage_receipt_payload_digest": candidate_validation[
            "storage_receipt_payload_digest"
        ],
        "candidate_metadata_equivalence_payload_digest": candidate_validation[
            "metadata_equivalence_payload_digest"
        ],
        "candidate_profile_id": STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID,
        "candidate_selector_eligible": False,
        "promotion_authorized": False,
        "physical_io_availability": _PHYSICAL_IO_AVAILABILITY,
    }
    workload = _strict_envelope(WORKLOAD_SCHEMA_ID, payload)
    require_workload(workload)
    return {
        "workload": workload,
        "candidate_storage_receipt_payload_digest": candidate_validation[
            "storage_receipt_payload_digest"
        ],
    }


def _require_metadata_receipt(
    value: object, *, run_path: str, array_count: int
) -> None:
    fields = {
        "schema_id",
        "schema_version",
        "subtree_path",
        "node_count",
        "group_count",
        "array_count",
        "declarations_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("Direct/consolidated metadata receipt field set is not exact.")
    if (
        value["schema_id"] != METADATA_EQUIVALENCE_SCHEMA_ID
        or value["schema_version"] != METADATA_EQUIVALENCE_SCHEMA_VERSION
        or value["subtree_path"] != run_path
        or value["array_count"] != array_count
        or type(value["node_count"]) is not int
        or type(value["group_count"]) is not int
        or value["node_count"] != value["group_count"] + value["array_count"]
        or not _is_sha256(value["declarations_sha256"])
    ):
        raise ValueError("Direct/consolidated metadata receipt is invalid.")


def _require_array_inventory(value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Expected array inventory must be nonempty.")
    for path, facts in value.items():
        if (
            type(path) is not str
            or not isinstance(facts, Mapping)
            or set(facts)
            != {
                "dtype",
                "shape",
                "decoded_nbytes",
                "decoded_sha256",
            }
        ):
            raise ValueError("Expected array inventory entry is invalid.")
        if type(facts["dtype"]) is not str or not isinstance(facts["shape"], list):
            raise ValueError("Expected array dtype/shape is invalid.")
        if any(type(extent) is not int or extent < 0 for extent in facts["shape"]):
            raise ValueError("Expected array shape extent is invalid.")
        expected_nbytes = (
            int(np.prod(facts["shape"], dtype=np.int64))
            * np.dtype(facts["dtype"]).itemsize
        )
        if facts["decoded_nbytes"] != expected_nbytes:
            raise ValueError("Expected array byte count disagrees with dtype/shape.")
        if not _is_sha256(facts["decoded_sha256"]):
            raise ValueError("Expected array digest is invalid.")


def _require_candidate_equivalence_receipt(
    value: object,
    *,
    run_path: str,
    array_count: int,
    group_count: int,
    profile_id: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("Candidate metadata-equivalence receipt is not exact.")
    if (
        value["schema_id"] != STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_ID
        or value["schema_version"]
        != STIMULUS_RESPONSE_METADATA_EQUIVALENCE_SCHEMA_VERSION
    ):
        raise ValueError("Candidate metadata-equivalence schema identity mismatch.")
    payload = value["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "run_path",
        "profile_id",
        "normalization",
        "array_declaration_count",
        "group_declaration_count",
        "normalized_metadata_sha256",
        "result",
    }:
        raise ValueError("Candidate metadata-equivalence payload is not exact.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Candidate metadata-equivalence payload digest mismatch.")
    if (
        payload["run_path"] != run_path
        or payload["profile_id"] != profile_id
        or payload["normalization"] != STIMULUS_RESPONSE_METADATA_NORMALIZATION
        or payload["array_declaration_count"] != array_count
        or payload["group_declaration_count"] != group_count
        or payload["result"] != "direct_and_consolidated_metadata_equal"
        or not _is_sha256(payload["normalized_metadata_sha256"])
    ):
        raise ValueError("Candidate metadata-equivalence binding is invalid.")


def _require_storage_receipt(
    value: object,
    *,
    bundles: Sequence[str],
    expected_arrays: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Candidate storage receipt must be one object.")
    try:
        parsed = analysis_storage_plan_receipt_from_manifest(value)
    except Exception as exc:
        raise ValueError(f"Candidate storage receipt cannot be parsed: {exc}") from exc
    if parsed.profile.as_manifest() != PUBLISHED_HTTP_V1.as_manifest():
        raise ValueError("Candidate storage receipt profile is not executable HTTP v1.")
    expected_paths = {
        item.path
        for item in stimulus_response_array_declarations(
            bundles=bundles,
            byte_planner_adopted=True,
        )
    }
    if {entry.declaration.path for entry in parsed.entries} != expected_paths:
        raise ValueError("Candidate storage receipt array path set differs.")
    facts = {
        path: SimpleNamespace(
            shape=tuple(item["shape"]),
            dtype=np.dtype(item["dtype"]),
        )
        for path, item in expected_arrays.items()
    }
    try:
        executable = build_stimulus_response_storage_receipt(
            arrays_by_path=facts,
            bundles=bundles,
            profile=PUBLISHED_HTTP_V1,
        ).as_manifest()
    except Exception as exc:
        raise ValueError(
            f"Candidate storage receipt cannot be replanned: {exc}"
        ) from exc
    if dict(value) != executable:
        raise ValueError(
            "Candidate storage receipt differs from executable byte planning."
        )


def require_workload(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=WORKLOAD_SCHEMA_ID)
    fields = {
        "family_id",
        "benchmark_id",
        "workload_id",
        "archive",
        "parent_path",
        "source_run",
        "candidate_run",
        "source_run_path",
        "candidate_run_path",
        "seed",
        "repetitions",
        "bundles",
        "access",
        "source_array_schema_manifest",
        "candidate_array_schema_manifest",
        "candidate_storage_receipt",
        "candidate_metadata_equivalence_receipt",
        "candidate_envelope",
        "source_metadata_equivalence",
        "candidate_metadata_equivalence",
        "source_metadata_declarations",
        "candidate_metadata_declarations",
        "expected_arrays",
        "expected_arrays_digest",
        "expected_decoded_nbytes",
        "logical_reader_digest",
        "candidate_storage_receipt_payload_digest",
        "candidate_metadata_equivalence_payload_digest",
        "candidate_profile_id",
        "candidate_selector_eligible",
        "promotion_authorized",
        "physical_io_availability",
    }
    if set(payload) != fields:
        raise ValueError("Stimulus-response workload field set is not exact.")
    if (
        payload["family_id"] != FAMILY_ID
        or payload["benchmark_id"] != BENCHMARK_ID
        or payload["workload_id"] != WORKLOAD_ID
        or payload["parent_path"] != PARENT_PATH
    ):
        raise ValueError("Stimulus-response workload identity mismatch.")
    if (
        type(payload["archive"]) is not str
        or not Path(payload["archive"]).is_absolute()
    ):
        raise ValueError("Workload archive must be one absolute path.")
    source_name = _safe_run_name(payload["source_run"], label="source run")
    candidate_name = _safe_run_name(payload["candidate_run"], label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Workload source and candidate names must differ.")
    if payload["source_run_path"] != _run_path(source_name) or payload[
        "candidate_run_path"
    ] != _run_path(candidate_name):
        raise ValueError("Workload run-path binding mismatch.")
    if (
        type(payload["seed"]) is not int
        or type(payload["repetitions"]) is not int
        or payload["repetitions"] < 1
    ):
        raise ValueError("Workload seed/repetitions are invalid.")
    bundles = payload["bundles"]
    if (
        not isinstance(bundles, list)
        or bundles != sorted(set(bundles))
        or any(type(item) is not str for item in bundles)
    ):
        raise ValueError("Workload bundle declaration is invalid.")
    expected_source_manifest = stimulus_response_array_manifest(
        bundles=bundles,
        byte_planner_adopted=False,
    )
    expected_candidate_manifest = stimulus_response_array_manifest(
        bundles=bundles,
        byte_planner_adopted=True,
    )
    if payload["source_array_schema_manifest"] != expected_source_manifest:
        raise ValueError("Source schema manifest differs from executable contract.")
    if payload["candidate_array_schema_manifest"] != expected_candidate_manifest:
        raise ValueError("Candidate schema manifest differs from executable contract.")
    expected_arrays = payload["expected_arrays"]
    _require_array_inventory(expected_arrays)
    expected_paths = [item["path"] for item in expected_source_manifest["arrays"]]
    if set(expected_arrays) != set(expected_paths):
        raise ValueError("Workload array inventory differs from executable schema.")
    if payload["expected_arrays_digest"] != canonical_json_sha256(expected_arrays):
        raise ValueError("Workload expected-array digest mismatch.")
    if payload["expected_decoded_nbytes"] != sum(
        item["decoded_nbytes"] for item in expected_arrays.values()
    ):
        raise ValueError("Workload decoded-byte total mismatch.")
    access = payload["access"]
    if not isinstance(access, Mapping) or access != {
        "mode": "strict_reader_then_complete_array_scan",
        "array_order": expected_paths,
        "operation_count": len(expected_paths),
    }:
        raise ValueError("Workload access declaration is not exact.")
    _require_storage_receipt(
        payload["candidate_storage_receipt"],
        bundles=bundles,
        expected_arrays=expected_arrays,
    )
    receipt = payload["candidate_storage_receipt"]
    if payload["candidate_storage_receipt_payload_digest"] != receipt["payload_digest"]:
        raise ValueError("Candidate storage receipt binding mismatch.")
    if payload["candidate_profile_id"] != STIMULUS_RESPONSE_CANDIDATE_PROFILE_ID:
        raise ValueError("Candidate profile ID binding mismatch.")
    if payload["candidate_envelope"] != _exact_candidate_envelope():
        raise ValueError("Candidate lifecycle envelope is not exact.")
    _require_candidate_equivalence_receipt(
        payload["candidate_metadata_equivalence_receipt"],
        run_path=payload["candidate_run_path"],
        array_count=len(expected_arrays),
        group_count=1 + len(expected_table_names(bundles)),
        profile_id=payload["candidate_profile_id"],
    )
    if (
        payload["candidate_metadata_equivalence_payload_digest"]
        != payload["candidate_metadata_equivalence_receipt"]["payload_digest"]
    ):
        raise ValueError("Candidate metadata-equivalence receipt binding mismatch.")
    _require_metadata_receipt(
        payload["source_metadata_equivalence"],
        run_path=payload["source_run_path"],
        array_count=len(expected_arrays),
    )
    _require_metadata_receipt(
        payload["candidate_metadata_equivalence"],
        run_path=payload["candidate_run_path"],
        array_count=len(expected_arrays),
    )
    source_declarations = _require_metadata_declarations(
        payload["source_metadata_declarations"],
        receipt=payload["source_metadata_equivalence"],
        run_path=payload["source_run_path"],
    )
    candidate_declarations = _require_metadata_declarations(
        payload["candidate_metadata_declarations"],
        receipt=payload["candidate_metadata_equivalence"],
        run_path=payload["candidate_run_path"],
    )
    _require_candidate_physical_declarations(
        candidate_declarations,
        run_path=payload["candidate_run_path"],
        bundles=bundles,
        receipt=payload["candidate_storage_receipt"],
    )
    _require_run_group_bindings(
        source_declarations,
        candidate_declarations,
        source_run_path=payload["source_run_path"],
        candidate_run_path=payload["candidate_run_path"],
        source_manifest=payload["source_array_schema_manifest"],
        candidate_manifest=payload["candidate_array_schema_manifest"],
        candidate_envelope=payload["candidate_envelope"],
        candidate_receipt=payload["candidate_storage_receipt"],
        candidate_equivalence=payload["candidate_metadata_equivalence_receipt"],
    )
    normalized_document = _candidate_normalized_metadata_document(
        candidate_declarations,
        run_path=payload["candidate_run_path"],
        bundles=bundles,
    )
    if payload["candidate_metadata_equivalence_receipt"]["payload"][
        "normalized_metadata_sha256"
    ] != canonical_json_sha256(normalized_document):
        raise ValueError(
            "Candidate metadata-equivalence normalized declaration digest mismatch."
        )
    if not _is_sha256(payload["logical_reader_digest"]):
        raise ValueError("Workload logical-reader digest is invalid.")
    if (
        payload["candidate_selector_eligible"] is not False
        or payload["promotion_authorized"] is not False
        or payload["physical_io_availability"] != _PHYSICAL_IO_AVAILABILITY
    ):
        raise ValueError("Workload violates hard nonpromotion/telemetry policy.")


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    if (
        type(seed) is not int
        or type(repetition_index) is not int
        or repetition_index < 0
    ):
        raise ValueError("Trial-order inputs must be exact nonnegative integers.")
    return (
        ("candidate", "source")
        if (seed + repetition_index) % 2
        else (
            "source",
            "candidate",
        )
    )


def _require_scan(value: object, *, expected_arrays: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "array_count",
        "decoded_nbytes",
        "elapsed_ms",
        "arrays",
        "arrays_digest",
    }:
        raise ValueError("Trial full scan field set is not exact.")
    if (
        value["array_count"] != len(expected_arrays)
        or value["arrays"] != expected_arrays
    ):
        raise ValueError("Trial decoded array scan differs from workload.")
    if value["decoded_nbytes"] != sum(
        item["decoded_nbytes"] for item in expected_arrays.values()
    ):
        raise ValueError("Trial decoded byte total mismatch.")
    if value["arrays_digest"] != canonical_json_sha256(expected_arrays):
        raise ValueError("Trial array-scan digest mismatch.")
    if (
        type(value["elapsed_ms"]) not in {int, float}
        or not math.isfinite(value["elapsed_ms"])
        or value["elapsed_ms"] < 0
    ):
        raise ValueError("Trial scan timing is invalid.")


def _physical_io() -> dict[str, Any]:
    return {
        "availability": _PHYSICAL_IO_AVAILABILITY,
        "read_operations": None,
        "transferred_bytes": None,
        "range_reads": None,
        "trace_artifact": None,
    }


def _require_physical_io(value: object) -> None:
    expected = _physical_io()
    if value != expected:
        raise ValueError(
            "Untraced benchmark must not fabricate physical I/O telemetry."
        )


def run_single_trial(
    archive_value: Path | str,
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
    require_workload(workload)
    workload_payload = workload["payload"]
    archive = _safe_archive(archive_value)
    source_name = _safe_run_name(source_run, label="source run")
    candidate_name = _safe_run_name(candidate_run, label="candidate run")
    if (
        str(archive) != workload_payload["archive"]
        or source_name != workload_payload["source_run"]
        or candidate_name != workload_payload["candidate_run"]
    ):
        raise ValueError("Trial invocation differs from its immutable workload.")
    if role not in {"source", "candidate"}:
        raise ValueError("Trial role must be source or candidate.")
    expected_order = _trial_order(seed=seed, repetition_index=repetition_index)
    if order_position not in {0, 1} or expected_order[order_position] != role:
        raise ValueError("Trial role/order differs from deterministic rotation.")
    if (
        seed != workload_payload["seed"]
        or cache_state.strip() != cache_state
        or not cache_state
    ):
        raise ValueError("Trial seed/cache state is invalid.")
    run_name = source_name if role == "source" else candidate_name
    run_path = _run_path(run_name)
    expected_metadata = workload_payload[
        (
            "source_metadata_equivalence"
            if role == "source"
            else "candidate_metadata_equivalence"
        )
    ]
    started_at = utc_now()
    root, open_timing = _measure(lambda: _open_root(archive, consolidated=True))
    run_group = _group(root, run_path)
    validation, validation_timing = _measure(
        lambda: (
            _validate_source(run_group, run_name=run_name)
            if role == "source"
            else _validate_candidate(run_group, run_name=run_name)
        )
    )
    metadata, metadata_timing = _measure(
        lambda: _metadata_receipt(archive, run_path=run_path)
    )
    if metadata != expected_metadata:
        raise ValueError("Trial metadata-equivalence receipt differs from workload.")
    scan = _full_scan(
        run_group,
        array_paths=workload_payload["access"]["array_order"],
    )
    if scan["arrays"] != workload_payload["expected_arrays"]:
        raise ValueError("Trial decoded values differ from immutable workload.")
    if validation["logical_reader_digest"] != workload_payload["logical_reader_digest"]:
        raise ValueError("Trial maintained-reader digest differs from workload.")
    result_payload = {
        "family_id": FAMILY_ID,
        "benchmark_id": BENCHMARK_ID,
        "workload_id": WORKLOAD_ID,
        "workload_payload_digest": workload["payload_digest"],
        "archive": str(archive),
        "source_run": source_name,
        "candidate_run": candidate_name,
        "role": role,
        "run_name": run_name,
        "run_path": run_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": seed,
        "cache_state": cache_state,
        "process_id": os.getpid(),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "environment": _environment(archive=archive, cache_state=cache_state),
        "timings": {
            "consolidated_open": open_timing,
            "strict_validation_and_reader": validation_timing,
            "persisted_metadata_equivalence": metadata_timing,
            "complete_array_scan": {"elapsed_ms": scan["elapsed_ms"]},
        },
        "validation": validation,
        "metadata_equivalence": metadata,
        "full_scan": scan,
        "storage": storage_stats(_safe_persisted_run_path(archive, run_name=run_name)),
        "peak_rss_bytes": peak_rss_bytes(),
        "physical_io": _physical_io(),
        "candidate_selector_eligible": False,
        "promotion_authorized": False,
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, result_payload)
    require_trial_result(result, workload=workload)
    return result


def require_trial_result(
    value: Mapping[str, Any],
    *,
    workload: Mapping[str, Any],
) -> None:
    require_workload(workload)
    expected = workload["payload"]
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    fields = {
        "family_id",
        "benchmark_id",
        "workload_id",
        "workload_payload_digest",
        "archive",
        "source_run",
        "candidate_run",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "seed",
        "cache_state",
        "process_id",
        "started_at_utc",
        "completed_at_utc",
        "environment",
        "timings",
        "validation",
        "metadata_equivalence",
        "full_scan",
        "storage",
        "peak_rss_bytes",
        "physical_io",
        "candidate_selector_eligible",
        "promotion_authorized",
    }
    if set(payload) != fields:
        raise ValueError("Stimulus-response trial field set is not exact.")
    for name in (
        "family_id",
        "benchmark_id",
        "workload_id",
        "archive",
        "source_run",
        "candidate_run",
        "seed",
    ):
        if payload[name] != expected[name]:
            raise ValueError(f"Trial {name} binding mismatch.")
    if payload["workload_payload_digest"] != workload["payload_digest"]:
        raise ValueError("Trial workload-digest binding mismatch.")
    role = payload["role"]
    if role not in {"source", "candidate"}:
        raise ValueError("Trial role is invalid.")
    expected_name = expected[f"{role}_run"]
    if (
        payload["run_name"] != expected_name
        or payload["run_path"] != expected[f"{role}_run_path"]
    ):
        raise ValueError("Trial role-specific run binding mismatch.")
    if (
        type(payload["repetition_index"]) is not int
        or not 0 <= payload["repetition_index"] < expected["repetitions"]
    ):
        raise ValueError("Trial repetition index is invalid.")
    if (
        type(payload["order_position"]) is not int
        or payload["order_position"] not in {0, 1}
        or _trial_order(
            seed=payload["seed"], repetition_index=payload["repetition_index"]
        )[payload["order_position"]]
        != role
    ):
        raise ValueError("Trial deterministic order binding mismatch.")
    if type(payload["process_id"]) is not int or payload["process_id"] <= 0:
        raise ValueError("Trial process ID is invalid.")
    if any(
        type(payload[name]) is not str or not payload[name]
        for name in ("started_at_utc", "completed_at_utc", "cache_state")
    ):
        raise ValueError("Trial timestamp/cache state is invalid.")
    _require_environment(payload["environment"], cache_state=payload["cache_state"])
    timings = payload["timings"]
    if not isinstance(timings, Mapping) or set(timings) != {
        "consolidated_open",
        "strict_validation_and_reader",
        "persisted_metadata_equivalence",
        "complete_array_scan",
    }:
        raise ValueError("Trial timing field set is not exact.")
    for label, timing in timings.items():
        _require_timing(timing, label=label)
    validation = payload["validation"]
    required_validation = {
        "consumer_path",
        "bundles",
        "logical_reader_digest",
    }
    if role == "candidate":
        required_validation.update(
            {
                "storage_receipt_payload_digest",
                "metadata_equivalence_payload_digest",
            }
        )
    if not isinstance(validation, Mapping) or set(validation) != required_validation:
        raise ValueError("Trial validation receipt field set is not exact.")
    if (
        validation["bundles"] != expected["bundles"]
        or validation["logical_reader_digest"] != expected["logical_reader_digest"]
    ):
        raise ValueError("Trial logical validation binding mismatch.")
    expected_consumer = (
        "strict_compact_v3_source"
        if role == "source"
        else "strict_byte_planned_compact_v3_candidate"
    )
    if validation["consumer_path"] != expected_consumer:
        raise ValueError("Trial consumer path is invalid.")
    if role == "candidate" and (
        validation["storage_receipt_payload_digest"]
        != expected["candidate_storage_receipt_payload_digest"]
        or validation["metadata_equivalence_payload_digest"]
        != expected["candidate_metadata_equivalence_payload_digest"]
    ):
        raise ValueError("Trial candidate receipt binding mismatch.")
    expected_metadata = expected[
        (
            "source_metadata_equivalence"
            if role == "source"
            else "candidate_metadata_equivalence"
        )
    ]
    if payload["metadata_equivalence"] != expected_metadata:
        raise ValueError("Trial persisted metadata receipt differs from workload.")
    _require_scan(payload["full_scan"], expected_arrays=expected["expected_arrays"])
    storage = payload["storage"]
    if (
        not isinstance(storage, Mapping)
        or set(storage)
        != {
            "file_count",
            "metadata_file_count",
            "payload_file_count",
            "apparent_bytes",
            "allocated_bytes",
        }
        or any(type(item) is not int or item < 0 for item in storage.values())
    ):
        raise ValueError("Trial storage facts are invalid.")
    if type(payload["peak_rss_bytes"]) is not int or payload["peak_rss_bytes"] <= 0:
        raise ValueError("Trial peak RSS is invalid.")
    _require_physical_io(payload["physical_io"])
    if (
        payload["candidate_selector_eligible"] is not False
        or payload["promotion_authorized"] is not False
    ):
        raise ValueError("Trial evidence violates hard nonpromotion policy.")


def _matrix_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role in ("source", "candidate"):
        role_trials = [
            trial["payload"] for trial in trials if trial["payload"]["role"] == role
        ]
        summary[role] = {
            "trial_count": len(role_trials),
            "median_consolidated_open_ms": statistics.median(
                trial["timings"]["consolidated_open"]["elapsed_ms"]
                for trial in role_trials
            ),
            "median_strict_reader_ms": statistics.median(
                trial["timings"]["strict_validation_and_reader"]["elapsed_ms"]
                for trial in role_trials
            ),
            "median_complete_scan_ms": statistics.median(
                trial["full_scan"]["elapsed_ms"] for trial in role_trials
            ),
            "median_peak_rss_bytes": statistics.median(
                trial["peak_rss_bytes"] for trial in role_trials
            ),
        }
    return summary


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    fields = {
        "family_id",
        "benchmark_id",
        "workload",
        "workload_payload_digest",
        "archive",
        "source_run",
        "candidate_run",
        "seed",
        "repetitions",
        "cache_state",
        "driver_process_id",
        "trials",
        "summary",
        "correctness",
        "archive_read_only_metadata_guard",
        "balanced_fresh_process_matrix_complete",
        "candidate_selector_eligible",
        "promotion_authorized",
        "physical_io_availability",
        "started_at_utc",
        "completed_at_utc",
    }
    if set(payload) != fields:
        raise ValueError("Stimulus-response matrix field set is not exact.")
    workload = payload["workload"]
    if not isinstance(workload, Mapping):
        raise ValueError("Matrix workload must be embedded.")
    require_workload(workload)
    expected = workload["payload"]
    for name in (
        "family_id",
        "benchmark_id",
        "archive",
        "source_run",
        "candidate_run",
        "seed",
        "repetitions",
    ):
        if payload[name] != expected[name]:
            raise ValueError(f"Matrix {name} binding mismatch.")
    if (
        type(payload["cache_state"]) is not str
        or not payload["cache_state"]
        or payload["cache_state"] != payload["cache_state"].strip()
    ):
        raise ValueError("Matrix cache state is invalid.")
    if any(
        type(payload[name]) is not str or not payload[name]
        for name in ("started_at_utc", "completed_at_utc")
    ):
        raise ValueError("Matrix timestamps are invalid.")
    if payload["workload_payload_digest"] != workload["payload_digest"]:
        raise ValueError("Matrix workload digest binding mismatch.")
    if (
        type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
    ):
        raise ValueError("Matrix driver process ID is invalid.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != payload["repetitions"] * 2:
        raise ValueError("Matrix trial count is invalid.")
    pids: list[int] = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            raise ValueError("Matrix trial must be one evidence envelope.")
        require_trial_result(trial, workload=workload)
        trial_payload = trial["payload"]
        repetition = index // 2
        position = index % 2
        if (
            trial_payload["repetition_index"] != repetition
            or trial_payload["order_position"] != position
            or trial_payload["role"]
            != _trial_order(seed=payload["seed"], repetition_index=repetition)[position]
        ):
            raise ValueError("Matrix trials are not in deterministic rotated order.")
        if trial_payload["cache_state"] != payload["cache_state"]:
            raise ValueError("Matrix trial cache-state binding mismatch.")
        pids.append(trial_payload["process_id"])
    if len(set(pids)) != len(pids) or payload["driver_process_id"] in pids:
        raise ValueError("Matrix trials were not executed in distinct fresh processes.")
    if payload["summary"] != _matrix_summary(trials):
        raise ValueError("Matrix summary differs from executable aggregation.")
    correctness = payload["correctness"]
    if correctness != {
        "all_passed": True,
        "decoded_arrays_equal": True,
        "logical_reader_equal": True,
        "direct_consolidated_equal": True,
        "zero_archive_metadata_mutations": True,
    }:
        raise ValueError("Matrix correctness receipt is not exact.")
    guard = payload["archive_read_only_metadata_guard"]
    if not isinstance(guard, Mapping) or set(guard) != {"before", "after", "unchanged"}:
        raise ValueError("Matrix metadata guard receipt is not exact.")
    _require_metadata_guard(guard["before"])
    _require_metadata_guard(guard["after"])
    if guard["unchanged"] is not True or guard["before"] != guard["after"]:
        raise ValueError("Benchmark modified archive metadata.")
    if payload["balanced_fresh_process_matrix_complete"] is not (
        payload["repetitions"] == DEFAULT_REPETITIONS
    ):
        raise ValueError("Matrix completion classification is invalid.")
    if (
        payload["candidate_selector_eligible"] is not False
        or payload["promotion_authorized"] is not False
        or payload["physical_io_availability"] != _PHYSICAL_IO_AVAILABILITY
    ):
        raise ValueError("Matrix violates hard nonpromotion/telemetry policy.")


def run_benchmark_matrix(
    archive_value: Path | str,
    *,
    source_run: str,
    candidate_run: str,
    output_dir: Path | str,
    cache_state: str,
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    archive = _safe_archive(archive_value)
    output = _safe_output(output_dir, archive=archive)
    source_name = _safe_run_name(source_run, label="source run")
    candidate_name = _safe_run_name(candidate_run, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")
    if (
        type(cache_state) is not str
        or not cache_state
        or cache_state != cache_state.strip()
    ):
        raise ValueError("Cache state must be one exact nonempty string.")
    preflight = _preflight(
        archive,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
        seed=seed,
        repetitions=repetitions,
    )
    workload = preflight["workload"]
    before = _metadata_guard(archive, run_names=(source_name, candidate_name))
    output.mkdir(parents=True)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    _write_json(output / "read_workload.json", workload)
    started_at = utc_now()
    trials: list[Mapping[str, Any]] = []
    environment = os.environ.copy()
    for name in STORAGE_BENCHMARK_THREAD_ENVIRONMENT:
        environment.setdefault(name, "1")
    for repetition in range(repetitions):
        for position, role in enumerate(
            _trial_order(seed=seed, repetition_index=repetition)
        ):
            result_path = trials_dir / f"rep-{repetition:02d}-{position}-{role}.json"
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_stimulus_response_reads",
                "trial",
                "--archive",
                str(archive),
                "--source-run",
                source_name,
                "--candidate-run",
                candidate_name,
                "--role",
                role,
                "--repetition-index",
                str(repetition),
                "--order-position",
                str(position),
                "--seed",
                str(seed),
                "--cache-state",
                cache_state,
                "--workload",
                str(output / "read_workload.json"),
                "--benchmark-root",
                str(output),
                "--result",
                str(result_path),
            ]
            completed = subprocess.run(
                command,
                cwd=Path.cwd(),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Fresh-process {role} trial failed: {completed.stderr or completed.stdout}"
                )
            trial = _read_json(result_path)
            require_trial_result(trial, workload=workload)
            trials.append(trial)
    after = _metadata_guard(archive, run_names=(source_name, candidate_name))
    if before != after:
        raise RuntimeError("Read benchmark modified archive metadata.")
    payload = {
        "family_id": FAMILY_ID,
        "benchmark_id": BENCHMARK_ID,
        "workload": workload,
        "workload_payload_digest": workload["payload_digest"],
        "archive": str(archive),
        "source_run": source_name,
        "candidate_run": candidate_name,
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": cache_state,
        "driver_process_id": os.getpid(),
        "trials": trials,
        "summary": _matrix_summary(trials),
        "correctness": {
            "all_passed": True,
            "decoded_arrays_equal": True,
            "logical_reader_equal": True,
            "direct_consolidated_equal": True,
            "zero_archive_metadata_mutations": True,
        },
        "archive_read_only_metadata_guard": {
            "before": before,
            "after": after,
            "unchanged": True,
        },
        "balanced_fresh_process_matrix_complete": repetitions == DEFAULT_REPETITIONS,
        "candidate_selector_eligible": False,
        "promotion_authorized": False,
        "physical_io_availability": _PHYSICAL_IO_AVAILABILITY,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
    }
    result = _strict_envelope(MATRIX_SCHEMA_ID, payload)
    require_matrix_result(result)
    _write_json(output / "matrix_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    trial = subparsers.add_parser("trial")
    for child in (matrix, trial):
        child.add_argument("--archive", type=Path, required=True)
        child.add_argument("--source-run", required=True)
        child.add_argument("--candidate-run", required=True)
        child.add_argument("--seed", type=int, default=DEFAULT_SEED)
        child.add_argument("--cache-state", required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, choices=(0, 1), required=True)
    trial.add_argument("--workload", type=Path, required=True)
    trial.add_argument("--benchmark-root", type=Path, required=True)
    trial.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        run_benchmark_matrix(
            args.archive,
            source_run=args.source_run,
            candidate_run=args.candidate_run,
            output_dir=args.output_dir,
            cache_state=args.cache_state,
            seed=args.seed,
            repetitions=args.repetitions,
        )
        return 0
    workload = _read_json(args.workload)
    result_path = _safe_trial_output(args.result, benchmark_root=args.benchmark_root)
    result = run_single_trial(
        args.archive,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        seed=args.seed,
        cache_state=args.cache_state,
        workload=workload,
    )
    _write_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "DEFAULT_REPETITIONS",
    "DEFAULT_SEED",
    "MATRIX_SCHEMA_ID",
    "TRIAL_SCHEMA_ID",
    "WORKLOAD_ID",
    "WORKLOAD_SCHEMA_ID",
    "main",
    "require_matrix_result",
    "require_trial_result",
    "require_workload",
    "run_benchmark_matrix",
    "run_single_trial",
]
