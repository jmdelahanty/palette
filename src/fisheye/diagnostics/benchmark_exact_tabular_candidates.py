"""Benchmark exact compact analysis storage candidates read-only.

The matrix controller launches one fresh Python process for every source or
candidate trial.  Trial order rotates by repetition, while every logical read
selection comes from ``palette.analysis_storage_benchmark_suite`` rebuilt from
the candidate's executable storage-plan receipt.  The archive is never opened
for mutation and all evidence is written under an exclusive benchmark-only
output directory supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import zarr

from fisheye.analysis._exact_tabular_run_schema import MANIFEST_ATTRIBUTE
from fisheye.analysis.bout_kinematics_schema import (
    build_bout_kinematics_array_declarations,
    validate_bout_kinematics_array_manifest,
)
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    ANALYSIS_STORAGE_PROFILE_ROLE,
    ANALYSIS_STORAGE_PROFILE_ROLE_ATTR,
    validate_exact_tabular_storage_receipt,
)
from fisheye.analysis.detection_occupancy_schema import (
    build_occupancy_array_declarations,
    validate_occupancy_array_manifest,
)
from fisheye.analysis.swim_bout_schema import (
    build_swim_bout_array_declarations,
    validate_swim_bout_array_manifest,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import peak_rss_bytes, storage_stats, utc_now
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


TRIAL_SCHEMA_ID = "palette.exact_tabular_candidate_read_trial"
TRIAL_SCHEMA_VERSION = 1
MATRIX_SCHEMA_ID = "palette.exact_tabular_candidate_read_matrix"
MATRIX_SCHEMA_VERSION = 2
BENCHMARK_ID = "exact_tabular_candidate_reads_v1"
DEFAULT_SEED = 17
DEFAULT_REPETITIONS = 5
FULL_SCAN_TARGET_BLOCK_BYTES = 8 * 1024 * 1024
_ALIASES = frozenset({"latest", "latest_complete"})


@dataclass(frozen=True)
class _Family:
    family_id: str
    parent_path: str
    validate_manifest: Callable[..., tuple[str, ...]]
    build_declarations: Callable[..., tuple[Any, ...]]


_FAMILIES = {
    "swim_bouts": _Family(
        family_id="swim_bouts",
        parent_path="analysis/swim_bout_runs",
        validate_manifest=validate_swim_bout_array_manifest,
        build_declarations=build_swim_bout_array_declarations,
    ),
    "bout_kinematics": _Family(
        family_id="bout_kinematics",
        parent_path="analysis/bout_kinematics_runs",
        validate_manifest=validate_bout_kinematics_array_manifest,
        build_declarations=build_bout_kinematics_array_declarations,
    ),
    "detection_occupancy": _Family(
        family_id="detection_occupancy",
        parent_path="analysis/detection_occupancy_runs",
        validate_manifest=lambda group, **kwargs: validate_occupancy_array_manifest(
            group, session=False, **kwargs
        ),
        build_declarations=lambda group, **kwargs: build_occupancy_array_declarations(
            group, session=False, **kwargs
        ),
    ),
    "session_occupancy": _Family(
        family_id="session_occupancy",
        parent_path="analysis/session_occupancy_runs",
        validate_manifest=lambda group, **kwargs: validate_occupancy_array_manifest(
            group, session=True, **kwargs
        ),
        build_declarations=lambda group, **kwargs: build_occupancy_array_declarations(
            group, session=True, **kwargs
        ),
    ),
}


def _strict_envelope(
    schema_id: str,
    schema_version: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)
    # This also rejects non-finite timing values before evidence is written.
    json.dumps(normalized, allow_nan=False)
    return {
        "schema_id": schema_id,
        "schema_version": schema_version,
        "payload": normalized,
        "payload_digest": canonical_json_sha256(normalized),
    }


def _require_envelope(
    value: Mapping[str, Any],
    *,
    schema_id: str,
    schema_version: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("Benchmark evidence envelope has an unexpected field set.")
    if value["schema_id"] != schema_id or value["schema_version"] != schema_version:
        raise ValueError("Benchmark evidence schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark evidence payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Benchmark evidence payload digest mismatch.")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Benchmark evidence is not strict JSON: {exc}") from exc
    return payload


def require_trial_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(
        value,
        schema_id=TRIAL_SCHEMA_ID,
        schema_version=TRIAL_SCHEMA_VERSION,
    )
    expected = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "seed",
        "suite_payload_digest",
        "candidate_storage_receipt_payload_digest",
        "cache_state",
        "started_at_utc",
        "finished_at_utc",
        "environment",
        "validation",
        "metadata",
        "primary_access",
        "full_scan",
        "logical_arrays",
        "storage",
        "runtime",
        "physical_io",
    }
    if set(payload) != expected:
        raise ValueError("Benchmark trial payload has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("Benchmark trial ID mismatch.")
    if payload["family_id"] not in _FAMILIES:
        raise ValueError("Benchmark trial family is unsupported.")
    if payload["role"] not in {"source", "candidate"}:
        raise ValueError("Benchmark trial role is unsupported.")
    if payload["run_name"] != payload[f"{payload['role']}_run_name"]:
        raise ValueError("Benchmark trial role/run binding mismatch.")
    if type(payload["repetition_index"]) is not int or payload["repetition_index"] < 0:
        raise ValueError("Benchmark trial repetition index is invalid.")
    if payload["order_position"] not in {0, 1}:
        raise ValueError("Benchmark trial order position is invalid.")
    validation = payload["validation"]
    metadata = payload["metadata"]
    physical = payload["physical_io"]
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("Benchmark trial validation did not pass.")
    if not isinstance(metadata, Mapping) or metadata.get("equivalent") is not True:
        raise ValueError("Benchmark trial metadata equivalence did not pass.")
    if not isinstance(physical, Mapping) or set(physical) != {
        "request_count",
        "transferred_bytes",
        "availability",
    }:
        raise ValueError("Benchmark trial physical-I/O declaration is invalid.")
    if (
        physical["request_count"] is not None
        or physical["transferred_bytes"] is not None
    ):
        raise ValueError("This runner must not fabricate physical I/O telemetry.")


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(
        value,
        schema_id=MATRIX_SCHEMA_ID,
        schema_version=MATRIX_SCHEMA_VERSION,
    )
    expected = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "source_run_name",
        "candidate_run_name",
        "seed",
        "repetitions",
        "cache_state",
        "started_at_utc",
        "finished_at_utc",
        "suite",
        "candidate_storage_receipt_payload_digest",
        "trial_order",
        "trial_files",
        "trials",
        "correctness",
        "performance_summary",
        "archive_read_only_guard",
        "physical_io",
        "balanced_read_matrix_complete",
    }
    if set(payload) != expected:
        raise ValueError("Benchmark matrix payload has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("Benchmark matrix ID mismatch.")
    if payload["family_id"] not in _FAMILIES:
        raise ValueError("Benchmark matrix family is unsupported.")
    if type(payload["repetitions"]) is not int or payload["repetitions"] < 1:
        raise ValueError("Benchmark matrix repetition count is invalid.")
    suite = payload["suite"]
    if not isinstance(suite, Mapping):
        raise ValueError("Benchmark matrix suite is missing.")
    require_analysis_benchmark_suite_manifest(suite)
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Benchmark matrix trial count is invalid.")
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("Benchmark matrix trial must be one object.")
        require_trial_result(trial)
    expected_order = [
        {
            "repetition_index": repetition_index,
            "roles": list(
                _trial_order(
                    seed=payload["seed"],
                    repetition_index=repetition_index,
                )
            ),
        }
        for repetition_index in range(payload["repetitions"])
    ]
    if payload["trial_order"] != expected_order:
        raise ValueError("Benchmark matrix trial order is not deterministic v1.")
    trial_files = payload["trial_files"]
    if (
        not isinstance(trial_files, list)
        or len(trial_files) != len(trials)
        or len(set(trial_files)) != len(trial_files)
    ):
        raise ValueError("Benchmark matrix trial-file inventory is invalid.")
    for repetition_index, order in enumerate(expected_order):
        observed = sorted(
            (
                trial["payload"]["order_position"],
                trial["payload"]["role"],
            )
            for trial in trials
            if trial["payload"]["repetition_index"] == repetition_index
        )
        if observed != list(enumerate(order["roles"])):
            raise ValueError("Benchmark matrix trial roles do not match its order.")
    for trial in trials:
        trial_payload = trial["payload"]
        if (
            trial_payload["family_id"] != payload["family_id"]
            or trial_payload["archive_path"] != payload["archive_path"]
            or trial_payload["source_run_name"] != payload["source_run_name"]
            or trial_payload["candidate_run_name"] != payload["candidate_run_name"]
            or trial_payload["seed"] != payload["seed"]
            or trial_payload["cache_state"] != payload["cache_state"]
            or trial_payload["suite_payload_digest"] != suite["payload_digest"]
            or trial_payload["candidate_storage_receipt_payload_digest"]
            != payload["candidate_storage_receipt_payload_digest"]
        ):
            raise ValueError("Benchmark matrix/trial identity binding mismatch.")
    correctness = payload["correctness"]
    guard = payload["archive_read_only_guard"]
    if (
        not isinstance(correctness, Mapping)
        or correctness.get("all_passed") is not True
    ):
        raise ValueError("Benchmark matrix correctness gates did not pass.")
    if not isinstance(guard, Mapping) or guard.get("unchanged") is not True:
        raise ValueError("Benchmark matrix archive read-only guard did not pass.")
    physical = payload["physical_io"]
    if (
        not isinstance(physical, Mapping)
        or physical.get("request_count") is not None
        or physical.get("transferred_bytes") is not None
    ):
        raise ValueError("Benchmark matrix must not fabricate physical I/O telemetry.")
    if payload["balanced_read_matrix_complete"] is not (
        payload["repetitions"] == DEFAULT_REPETITIONS
    ):
        raise ValueError(
            "Benchmark balanced-read-matrix classification is invalid."
        )


def _family(value: str) -> _Family:
    try:
        return _FAMILIES[str(value)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported family {value!r}; expected {sorted(_FAMILIES)!r}."
        ) from exc


def _safe_run_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or name in _ALIASES:
        raise ValueError(f"{label} must be one explicit immutable run name.")
    if "/" in name or "\\" in name:
        raise ValueError(f"Unsafe {label}: {value!r}.")
    return name


def _safe_archive(path: Path | str) -> Path:
    archive = Path(path).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive not found: {archive}.")
    return archive


def _safe_new_output_dir(path: Path | str, *, archive: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark output directory already exists: {output}.")
    if (
        output == archive
        or output.is_relative_to(archive)
        or archive.is_relative_to(output)
    ):
        raise ValueError("Benchmark output must be outside the source archive tree.")
    if not any("benchmark" in part.lower() for part in output.parts):
        raise ValueError(
            "Benchmark output path must contain a component identifying it as benchmark-only."
        )
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("Benchmark output path is too broad.")
    return output


def _safe_trial_output(path: Path | str, *, benchmark_root: Path) -> Path:
    output = Path(path).expanduser().resolve()
    root = benchmark_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark root does not exist: {root}.")
    if output.exists():
        raise FileExistsError(f"Trial output already exists: {output}.")
    if not output.is_relative_to(root) or output.suffix != ".json":
        raise ValueError("Trial output must be a new JSON file inside benchmark root.")
    return output


def _write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace benchmark evidence: {path}.")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _read_strict_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda raw: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {raw}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"Strict JSON document is not one object: {path}.")
    return value


def _group_at(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _array_at(run_group: Any, path: str) -> Any:
    return _group_at(run_group, path)


def _manifest_adoption(group: Any) -> bool:
    manifest = group.attrs.get(MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping) or not isinstance(
        manifest.get("payload"), Mapping
    ):
        raise ValueError("Exact run lacks its executable array-schema manifest.")
    adopted = manifest["payload"].get("byte_planner_adopted")
    if type(adopted) is not bool:
        raise ValueError(
            "Exact run manifest lacks a boolean byte-planner adoption state."
        )
    return adopted


def _validate_run(
    group: Any,
    *,
    family: _Family,
    role: str,
    source_run_name: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    errors: list[str] = []
    adopted = _manifest_adoption(group)
    if group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("run is not complete")
    eligibility = group.attrs.get("stage_selector_eligible")
    if role == "source":
        if eligibility is not True:
            errors.append("source run is not selector eligible")
    elif role == "candidate":
        if eligibility is not False:
            errors.append("candidate run is not selector-ineligible")
        if not adopted:
            errors.append("candidate exact manifest is not byte-planner adopted")
        if group.attrs.get("storage_candidate_source_run") != source_run_name:
            errors.append("candidate source-run binding mismatch")
        if (
            group.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
            != ANALYSIS_STORAGE_PROFILE_ROLE
        ):
            errors.append("candidate storage-profile role mismatch")
    else:
        raise ValueError(f"Unsupported benchmark role {role!r}.")
    errors.extend(family.validate_manifest(group, byte_planner_adopted=adopted))
    declarations: tuple[Any, ...] = ()
    try:
        declarations = family.build_declarations(
            group,
            byte_planner_adopted=adopted,
        )
    except Exception as exc:
        errors.append(f"exact declaration reconstruction failed: {exc}")
    receipt_digest: str | None = None
    if role == "candidate" and declarations:
        receipt_errors = validate_exact_tabular_storage_receipt(
            group,
            declarations=declarations,
        )
        errors.extend(receipt_errors)
        receipt = group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
        if isinstance(receipt, Mapping):
            receipt_digest = str(receipt.get("payload_digest") or "")
    result = {
        "valid": not errors,
        "errors": errors,
        "byte_planner_adopted": adopted,
        "selector_eligible": eligibility,
        "array_count": len(declarations),
        "candidate_storage_receipt_payload_digest": receipt_digest,
    }
    if errors:
        raise ValueError(f"Invalid {family.family_id} {role} run: {errors!r}.")
    return declarations, result


def _logical_inventory(group: Any, declarations: Sequence[Any]) -> dict[str, Any]:
    return {
        declaration.path: {
            "dtype": np.dtype(_array_at(group, declaration.path).dtype).str,
            "shape": [int(value) for value in _array_at(group, declaration.path).shape],
            "logical_contract": declaration.contract.as_manifest(),
        }
        for declaration in declarations
    }


def _preflight(
    archive: Path,
    *,
    family: _Family,
    source_run_name: str,
    candidate_run_name: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    root = open_zarr_root(archive, mode="r")
    parent = root.get(family.parent_path)
    if not isinstance(parent, zarr.Group):
        raise KeyError(f"Missing exact run parent {family.parent_path!r}.")
    source = parent.get(source_run_name)
    candidate = parent.get(candidate_run_name)
    if not isinstance(source, zarr.Group):
        raise KeyError(f"Explicit source run {source_run_name!r} does not exist.")
    if not isinstance(candidate, zarr.Group):
        raise KeyError(f"Explicit candidate run {candidate_run_name!r} does not exist.")
    source_declarations, source_validation = _validate_run(
        source,
        family=family,
        role="source",
        source_run_name=source_run_name,
    )
    candidate_declarations, candidate_validation = _validate_run(
        candidate,
        family=family,
        role="candidate",
        source_run_name=source_run_name,
    )
    source_inventory = _logical_inventory(source, source_declarations)
    candidate_inventory = _logical_inventory(candidate, candidate_declarations)
    if source_inventory != candidate_inventory:
        raise ValueError("Source and candidate exact logical declarations differ.")
    receipt_manifest = candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
    receipt = analysis_storage_plan_receipt_from_manifest(receipt_manifest)
    if receipt.as_manifest() != receipt_manifest:
        raise ValueError("Candidate storage receipt is not exactly executable.")
    scale = AnalysisBenchmarkScale(
        scale_id="observed_candidate",
        dimensions=tuple(
            sorted((str(key), int(value)) for key, value in receipt.dimensions)
        ),
        description=(
            f"Observed {family.family_id} source/candidate logical scale from the "
            "candidate executable storage receipt."
        ),
    )
    suite = build_analysis_benchmark_suite(
        family_id=family.family_id,
        scale=scale,
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_analysis_benchmark_suite_manifest(suite)
    return {
        "suite": suite,
        "source_validation": source_validation,
        "candidate_validation": candidate_validation,
        "logical_inventory": source_inventory,
        "candidate_storage_receipt_payload_digest": receipt_manifest["payload_digest"],
    }


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    value = call()
    return value, {
        "wall_seconds": float(time.perf_counter() - wall_started),
        "cpu_seconds": float(time.process_time() - cpu_started),
    }


def _growth_axis_by_path(suite: Mapping[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    arrays = suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]
    for record in arrays:
        path = str(record["path"])
        axis = record["observed_facts"]["growth_axis"]
        result[path] = None if axis is None else int(axis)
    return result


def _primary_cases(suite: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for record in suite["payload"]["array_cases"]:
        workload_id = record["case"]["workload"]["workload_id"]
        if workload_id.endswith(".write_materialization.v1") or workload_id.endswith(
            ".full_scan_read.v1"
        ):
            continue
        result.append(record)
    return result


def _selection_slice(
    array: Any, *, axis: int, start: int, stop: int
) -> tuple[slice, ...]:
    selection = [slice(None)] * int(array.ndim)
    selection[axis] = slice(start, stop)
    return tuple(selection)


def _selection_index(array: Any, *, axis: int, row: int) -> tuple[Any, ...]:
    selection: list[Any] = [slice(None)] * int(array.ndim)
    selection[axis] = int(row)
    return tuple(selection)


def _primary_read(
    array: Any, selection: Mapping[str, Any], *, growth_axis: int | None
) -> dict[str, Any]:
    mode = selection.get("mode")
    axis = 0 if growth_axis is None else growth_axis
    digest = hashlib.sha256()
    decoded_bytes = 0
    operations = 0

    def consume(values: Any) -> None:
        nonlocal decoded_bytes, operations
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.view(np.uint8))
        decoded_bytes += int(contiguous.nbytes)
        operations += 1

    if mode == "whole_array":
        consume(array[...])
    elif mode == "bounded_row_windows":
        for start, stop in selection["ranges"]:
            consume(
                array[
                    _selection_slice(array, axis=axis, start=int(start), stop=int(stop))
                ]
            )
    elif mode == "random_complete_rows":
        for row in selection["row_indices"]:
            consume(array[_selection_index(array, axis=axis, row=int(row))])
    elif mode == "indexed_row_resolution":
        for row in selection["index_rows"]:
            consume(array[_selection_index(array, axis=axis, row=int(row))])
    else:
        raise ValueError(f"Unsupported benchmark-suite selection mode {mode!r}.")
    result = {
        "mode": mode,
        "execution_axis": axis,
        "suite_v1_selection_extent_source": "logical_shape_axis_0",
        "operation_count": operations,
        "decoded_bytes": decoded_bytes,
        "selection_digest": digest.hexdigest(),
    }
    if mode == "indexed_row_resolution":
        result["indexed_resolution"] = (
            "deterministic_complete_table_rows; no common persisted ptr_len "
            "range index exists in exact-tabular v1"
        )
    return result


def _full_scan(array: Any, *, growth_axis: int | None) -> dict[str, Any]:
    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(dtype.str.encode("utf-8"))
    digest.update(json.dumps(list(shape), separators=(",", ":")).encode("ascii"))
    if not shape:
        block = np.ascontiguousarray(array[...])
        digest.update(block.view(np.uint8))
        return {
            "dtype": dtype.str,
            "shape": [],
            "decoded_bytes": int(block.nbytes),
            "block_count": 1,
            "logical_digest": digest.hexdigest(),
        }
    axis = 0 if growth_axis is None else growth_axis
    non_growth = max(
        1,
        int(np.prod([extent for index, extent in enumerate(shape) if index != axis])),
    )
    bytes_per_unit = max(1, dtype.itemsize * non_growth)
    block_extent = max(1, FULL_SCAN_TARGET_BLOCK_BYTES // bytes_per_unit)
    block_count = 0
    decoded_bytes = 0
    for start in range(0, shape[axis], block_extent):
        stop = min(start + block_extent, shape[axis])
        block = np.ascontiguousarray(
            array[_selection_slice(array, axis=axis, start=start, stop=stop)]
        )
        digest.update(block.view(np.uint8))
        decoded_bytes += int(block.nbytes)
        block_count += 1
    return {
        "dtype": dtype.str,
        "shape": list(shape),
        "decoded_bytes": decoded_bytes,
        "block_count": block_count,
        "logical_digest": digest.hexdigest(),
    }


def _metadata_guard(
    archive: Path, *, family: _Family, run_names: Sequence[str]
) -> dict[str, Any]:
    paths = [
        archive / "zarr.json",
        archive.joinpath(*family.parent_path.split("/"), "zarr.json"),
    ]
    for run_name in run_names:
        run_path = archive.joinpath(*family.parent_path.split("/"), run_name)
        paths.extend(sorted(run_path.rglob("zarr.json")))
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata file is missing: {path}.")
        payload = path.read_bytes()
        relative = str(path.relative_to(archive))
        item_digest = hashlib.sha256(payload).hexdigest()
        records.append({"path": relative, "size": len(payload), "sha256": item_digest})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_digest.encode("ascii"))
        digest.update(b"\0")
    return {
        "metadata_file_count": len(records),
        "metadata_tree_sha256": digest.hexdigest(),
        "files": records,
    }


def _runtime_environment(*, archive: Path, cache_state: str) -> dict[str, Any]:
    git = get_git_info()
    try:
        filesystem = os.statvfs(archive)
        filesystem_block_size: int | None = int(filesystem.f_frsize)
    except OSError:
        filesystem_block_size = None
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
        "cache_state": cache_state,
        "filesystem_block_size": filesystem_block_size,
        "thread_environment": {
            key: os.environ.get(key) for key in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def run_single_trial(
    archive_path: Path | str,
    *,
    family_id: str,
    source_run: str,
    candidate_run: str,
    role: str,
    repetition_index: int,
    order_position: int,
    seed: int,
    cache_state: str,
    suite_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one read-only source or candidate trial in the current process."""

    archive = _safe_archive(archive_path)
    family = _family(family_id)
    source_name = _safe_run_name(source_run, label="source run")
    candidate_name = _safe_run_name(candidate_run, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")
    if role not in {"source", "candidate"}:
        raise ValueError("Trial role must be source or candidate.")
    if type(repetition_index) is not int or repetition_index < 0:
        raise ValueError("repetition_index must be nonnegative.")
    if order_position not in {0, 1}:
        raise ValueError("order_position must be 0 or 1.")
    if not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    require_analysis_benchmark_suite_manifest(suite_manifest)
    suite_payload = suite_manifest["payload"]
    if suite_payload["family_id"] != family.family_id or suite_payload["seed"] != seed:
        raise ValueError("Benchmark suite family/seed differs from this trial.")
    if repetition_index >= suite_payload["repetitions"]:
        raise ValueError("Trial repetition index exceeds the benchmark suite.")
    if (
        _trial_order(seed=seed, repetition_index=repetition_index)[order_position]
        != role
    ):
        raise ValueError("Trial role differs from deterministic rotated order.")
    receipt_manifest = suite_payload["storage_plan_receipt"]
    receipt_digest = receipt_manifest["payload_digest"]
    growth_axes = _growth_axis_by_path(suite_manifest)
    paths = sorted(growth_axes)
    run_name = source_name if role == "source" else candidate_name
    run_path = f"{family.parent_path}/{run_name}"
    started = utc_now()
    initial_rss = peak_rss_bytes()

    (direct_root, direct_group), direct_open = _measure(
        lambda: (
            (root := zarr.open_group(str(archive), mode="r", use_consolidated=False)),
            _group_at(root, run_path),
        )
    )
    (consolidated_root, consolidated_group), consolidated_open = _measure(
        lambda: (
            (root := zarr.open_group(str(archive), mode="r", use_consolidated=True)),
            _group_at(root, run_path),
        )
    )
    del direct_root, consolidated_root

    def validate() -> tuple[tuple[Any, ...], dict[str, Any]]:
        declarations, result = _validate_run(
            direct_group,
            family=family,
            role=role,
            source_run_name=source_name,
        )
        if {declaration.path for declaration in declarations} != set(paths):
            raise ValueError(
                "Trial exact declaration paths differ from benchmark suite."
            )
        if role == "candidate":
            persisted = direct_group.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR]
            if (
                persisted != receipt_manifest
                or persisted.get("payload_digest") != receipt_digest
            ):
                raise ValueError(
                    "Candidate receipt differs from benchmark-suite binding."
                )
        return declarations, result

    (declarations, validation), validation_timing = _measure(validate)

    def compare_metadata() -> dict[str, Any]:
        receipt = validate_direct_consolidated_subtree(
            archive,
            subtree_path=run_path,
        )
        if receipt.array_count < len(paths):
            raise ValueError(
                "Persisted metadata subtree omits benchmark arrays: "
                f"expected at least {len(paths)}, got {receipt.array_count}."
            )
        return {
            "equivalent": True,
            "array_count": receipt.array_count,
            "group_count": receipt.group_count,
            "node_count": receipt.node_count,
            "subtree_declarations_digest": receipt.declarations_sha256,
        }

    metadata, metadata_comparison_timing = _measure(compare_metadata)

    primary_arrays: dict[str, Any] = {}
    primary_wall = 0.0
    primary_cpu = 0.0
    for record in _primary_cases(suite_manifest):
        path = str(record["array_path"])
        result, timing = _measure(
            lambda path=path, record=record: _primary_read(
                _array_at(consolidated_group, path),
                record["selection"],
                growth_axis=growth_axes[path],
            )
        )
        primary_arrays[path] = {
            **result,
            "workload_id": record["case"]["workload"]["workload_id"],
            "selection": record["selection"],
            "timing": timing,
        }
        primary_wall += timing["wall_seconds"]
        primary_cpu += timing["cpu_seconds"]

    scans: dict[str, Any] = {}
    scan_wall = 0.0
    scan_cpu = 0.0
    for path in paths:
        result, timing = _measure(
            lambda path=path: _full_scan(
                _array_at(consolidated_group, path),
                growth_axis=growth_axes[path],
            )
        )
        scans[path] = {**result, "timing": timing}
        scan_wall += timing["wall_seconds"]
        scan_cpu += timing["cpu_seconds"]

    logical_arrays = {
        path: {
            "dtype": scans[path]["dtype"],
            "shape": scans[path]["shape"],
            "logical_digest": scans[path]["logical_digest"],
        }
        for path in paths
    }
    run_storage = storage_stats(archive.joinpath(*run_path.split("/")))
    finished = utc_now()
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": family.family_id,
        "archive_path": str(archive),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "role": role,
        "run_name": run_name,
        "run_path": run_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": seed,
        "suite_payload_digest": suite_manifest["payload_digest"],
        "candidate_storage_receipt_payload_digest": receipt_digest,
        "cache_state": cache_state,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "environment": _runtime_environment(archive=archive, cache_state=cache_state),
        "validation": {**validation, "timing": validation_timing},
        "metadata": {
            **metadata,
            "direct_open": direct_open,
            "consolidated_open": consolidated_open,
            "comparison": metadata_comparison_timing,
        },
        "primary_access": {
            "arrays": primary_arrays,
            "total_wall_seconds": primary_wall,
            "total_cpu_seconds": primary_cpu,
        },
        "full_scan": {
            "arrays": scans,
            "total_wall_seconds": scan_wall,
            "total_cpu_seconds": scan_cpu,
            "total_decoded_bytes": sum(
                item["decoded_bytes"] for item in scans.values()
            ),
        },
        "logical_arrays": logical_arrays,
        "storage": {
            **run_storage,
            "payload_object_count": run_storage["payload_file_count"],
        },
        "runtime": {
            "initial_peak_rss_bytes": initial_rss,
            "final_peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_is_process_high_water_mark": True,
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": (
                "unavailable_without_os_or_filesystem_tracing; file counts and "
                "logical decoded bytes are not physical request/transfer telemetry"
            ),
        },
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, TRIAL_SCHEMA_VERSION, payload)
    require_trial_result(result)
    return result


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    return (
        ("source", "candidate")
        if (seed + repetition_index) % 2 == 0
        else ("candidate", "source")
    )


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
    return float(statistics.median(values))


def _matrix_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role in ("source", "candidate"):
        summary[role] = {
            "median_direct_open_wall_seconds": _median_by_role(
                trials, role, ("metadata", "direct_open", "wall_seconds")
            ),
            "median_consolidated_open_wall_seconds": _median_by_role(
                trials, role, ("metadata", "consolidated_open", "wall_seconds")
            ),
            "median_metadata_comparison_wall_seconds": _median_by_role(
                trials, role, ("metadata", "comparison", "wall_seconds")
            ),
            "median_manifest_validation_wall_seconds": _median_by_role(
                trials, role, ("validation", "timing", "wall_seconds")
            ),
            "median_primary_access_wall_seconds": _median_by_role(
                trials, role, ("primary_access", "total_wall_seconds")
            ),
            "median_full_scan_wall_seconds": _median_by_role(
                trials, role, ("full_scan", "total_wall_seconds")
            ),
            "median_peak_rss_bytes": _median_by_role(
                trials, role, ("runtime", "final_peak_rss_bytes")
            ),
            "payload_object_count": trials[
                next(
                    index
                    for index, item in enumerate(trials)
                    if item["payload"]["role"] == role
                )
            ]["payload"]["storage"]["payload_object_count"],
            "apparent_bytes": trials[
                next(
                    index
                    for index, item in enumerate(trials)
                    if item["payload"]["role"] == role
                )
            ]["payload"]["storage"]["apparent_bytes"],
            "allocated_bytes": trials[
                next(
                    index
                    for index, item in enumerate(trials)
                    if item["payload"]["role"] == role
                )
            ]["payload"]["storage"]["allocated_bytes"],
        }
    return summary


def run_benchmark_matrix(
    archive_path: Path | str,
    *,
    family_id: str,
    source_run: str,
    candidate_run: str,
    output_dir: Path | str,
    cache_state: str,
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Run a balanced fresh-process matrix and write immutable strict JSON."""

    archive = _safe_archive(archive_path)
    family = _family(family_id)
    source_name = _safe_run_name(source_run, label="source run")
    candidate_name = _safe_run_name(candidate_run, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be one nonnegative exact integer.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be one positive exact integer.")
    if not str(cache_state).strip():
        raise ValueError("cache_state must be explicitly declared.")
    output = _safe_new_output_dir(output_dir, archive=archive)
    preflight = _preflight(
        archive,
        family=family,
        source_run_name=source_name,
        candidate_run_name=candidate_name,
        seed=seed,
        repetitions=repetitions,
    )
    guard_before = _metadata_guard(
        archive,
        family=family,
        run_names=(source_name, candidate_name),
    )
    output.mkdir(parents=True, exist_ok=False)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    suite_path = output / "analysis_benchmark_suite.json"
    _write_strict_json(suite_path, preflight["suite"])
    started = utc_now()
    trials: list[Mapping[str, Any]] = []
    trial_order: list[dict[str, Any]] = []
    trial_files: list[str] = []
    environment = os.environ.copy()
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for repetition_index in range(repetitions):
        order = _trial_order(seed=seed, repetition_index=repetition_index)
        trial_order.append({"repetition_index": repetition_index, "roles": list(order)})
        for order_position, role in enumerate(order):
            filename = f"rep_{repetition_index:02d}_pos_{order_position}_{role}.json"
            trial_path = trials_dir / filename
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_exact_tabular_candidates",
                "trial",
                str(archive),
                "--family",
                family.family_id,
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
                str(cache_state),
                "--suite-file",
                str(suite_path),
                "--benchmark-root",
                str(output),
                "--output-file",
                str(trial_path),
            ]
            completed = subprocess.run(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Fresh-process exact-tabular benchmark trial failed: "
                    f"command={command!r}, stdout={completed.stdout!r}, "
                    f"stderr={completed.stderr!r}."
                )
            trial = _read_strict_json(trial_path)
            require_trial_result(trial)
            trials.append(trial)
            trial_files.append(str(trial_path.relative_to(output)))

    reference_logical_arrays = trials[0]["payload"]["logical_arrays"]
    logical_equality = all(
        trial["payload"]["logical_arrays"] == reference_logical_arrays
        for trial in trials
    )
    if not logical_equality:
        raise ValueError("Source and candidate full-scan logical digests differ.")
    guard_after = _metadata_guard(
        archive,
        family=family,
        run_names=(source_name, candidate_name),
    )
    unchanged = guard_before == guard_after
    if not unchanged:
        raise RuntimeError(
            "Archive metadata changed during read-only benchmark execution."
        )
    correctness = {
        "logical_equality": logical_equality,
        "direct_consolidated_metadata_equivalence": all(
            trial["payload"]["metadata"]["equivalent"] is True for trial in trials
        ),
        "manifest_and_receipt_validation": all(
            trial["payload"]["validation"]["valid"] is True for trial in trials
        ),
        "archive_metadata_unchanged": unchanged,
        "all_passed": True,
    }
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": family.family_id,
        "archive_path": str(archive),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": str(cache_state),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "suite": preflight["suite"],
        "candidate_storage_receipt_payload_digest": preflight[
            "candidate_storage_receipt_payload_digest"
        ],
        "trial_order": trial_order,
        "trial_files": trial_files,
        "trials": trials,
        "correctness": correctness,
        "performance_summary": _matrix_summary(trials),
        "archive_read_only_guard": {
            "before": guard_before,
            "after": guard_after,
            "unchanged": unchanged,
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": (
                "unavailable_without_os_or_filesystem_tracing; this matrix reports "
                "object inventory, apparent/allocated bytes, and decoded bytes only"
            ),
        },
        # Five balanced repetitions complete this read-only matrix contract.
        # This is deliberately not a profile-promotion verdict: the adapter
        # has no writer, publication, physical-I/O, representative-scale, or
        # real-consumer evidence.
        "balanced_read_matrix_complete": repetitions == DEFAULT_REPETITIONS,
    }
    result = _strict_envelope(MATRIX_SCHEMA_ID, MATRIX_SCHEMA_VERSION, payload)
    require_matrix_result(result)
    _write_strict_json(output / "matrix_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="Run the fresh-process matrix.")
    trial = subparsers.add_parser("trial", help="Run one internal fresh-process trial.")
    for child in (matrix, trial):
        child.add_argument("zarr_path", type=Path)
        child.add_argument("--family", choices=tuple(sorted(_FAMILIES)), required=True)
        child.add_argument("--source-run", required=True)
        child.add_argument("--candidate-run", required=True)
        child.add_argument("--seed", type=int, default=DEFAULT_SEED)
        child.add_argument("--cache-state", required=True)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, choices=(0, 1), required=True)
    trial.add_argument("--suite-file", type=Path, required=True)
    trial.add_argument("--benchmark-root", type=Path, required=True)
    trial.add_argument("--output-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        result = run_benchmark_matrix(
            args.zarr_path,
            family_id=args.family,
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
    suite = _read_strict_json(args.suite_file.expanduser().resolve())
    output = _safe_trial_output(
        args.output_file,
        benchmark_root=args.benchmark_root,
    )
    result = run_single_trial(
        args.zarr_path,
        family_id=args.family,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        seed=args.seed,
        cache_state=args.cache_state,
        suite_manifest=suite,
    )
    _write_strict_json(output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "DEFAULT_REPETITIONS",
    "DEFAULT_SEED",
    "MATRIX_SCHEMA_ID",
    "MATRIX_SCHEMA_VERSION",
    "TRIAL_SCHEMA_ID",
    "TRIAL_SCHEMA_VERSION",
    "require_matrix_result",
    "require_trial_result",
    "run_benchmark_matrix",
    "run_single_trial",
]
