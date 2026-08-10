"""Validate and benchmark one selector-ineligible subject-shape v5 canary.

The command is read-only with respect to the analysis archive.  It requires an
exact named v5 run and inactive recording subject-mask bundle, replays the
published coordinate and storage contracts, proves frame-to-row cardinality,
exercises the eye-geometry and tail-kinematics consumer boundaries, and writes
one strict JSON receipt to a disjoint benchmark path.

This is canary acceptance evidence, not selector or storage-profile promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from fisheye.analysis import tail_kinematics_runs as tail_consumer
from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    validate_subject_shape_candidate_storage,
    validate_subject_shape_direct_consolidated_storage,
)
from fisheye.analysis_workflows.subject_shape_candidate_execution import (
    compute_subject_shape_logical_hashes,
)
from fisheye.shared import eye_geometry_source as eye_consumer
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.subject_shape_coordinate_publication import (
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
    CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
    SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR,
    SUBJECT_SHAPE_BUNDLE_ID_ATTR,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR,
    load_completed_ineligible_subject_shape_coordinate_publication,
)
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.subject_shape_bundle_source import (
    load_subject_shape_bundle_source,
)

RUN_PARENT = "analysis/subject_shape_runs"
RESULT_SCHEMA_ID = "palette.subject_shape_v5_canary_acceptance"
RESULT_SCHEMA_VERSION = 1
DEFAULT_RANDOM_FRAME_COUNT = 128
DEFAULT_WINDOW_COUNT = 64
DEFAULT_WINDOW_ROWS = 4_096
DEFAULT_SEED = 23
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)
_HOT_ARRAY_PATHS = (
    "body_frame/heading_deg",
    "components/subject_body/tail_sample_xy",
    "relations/eye_pair/separation_px",
    "row_index/instance_key",
)


def _safe_run_name(value: str, *, label: str) -> str:
    name = str(value).strip()
    if (
        not name
        or name != value
        or name in {".", "..", "latest", "latest_complete", "latest_pending"}
        or "/" in name
        or "\\" in name
        or any(character.isspace() for character in name)
    ):
        raise ValueError(f"{label} must be one exact immutable child name.")
    return name


def _safe_archive(path: Path | str) -> Path:
    archive = Path(path).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive not found: {archive}.")
    return archive


def _safe_output(path: Path | str, *, archive: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Acceptance output already exists: {output}.")
    if output.suffix != ".json":
        raise ValueError("Acceptance output must be one JSON file.")
    if output == archive or output.is_relative_to(archive):
        raise ValueError("Acceptance output must be outside the analysis archive.")
    if not any("benchmark" in component.lower() for component in output.parts):
        raise ValueError("Acceptance output must be below a benchmark namespace.")
    return output


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace acceptance evidence: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _measure(call: Any) -> tuple[Any, dict[str, float]]:
    wall = time.perf_counter()
    cpu = time.process_time()
    result = call()
    return result, {
        "wall_seconds": float(time.perf_counter() - wall),
        "cpu_seconds": float(time.process_time() - cpu),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def validate_frame_cardinality_arrays(
    offsets: np.ndarray,
    frame_indices: np.ndarray,
    *,
    require_empty_frame: bool,
    require_multi_row_frame: bool,
) -> dict[str, Any]:
    """Prove one exact F+1 CSR index against its row-level frame column."""

    offset_values = np.asarray(offsets)
    frames = np.asarray(frame_indices)
    if offset_values.dtype != np.dtype("int64") or offset_values.ndim != 1:
        raise ValueError("frame_row_offsets must be exact rank-1 int64.")
    if frames.dtype != np.dtype("int64") or frames.ndim != 1:
        raise ValueError("source_acquisition_frame_index must be exact rank-1 int64.")
    if offset_values.size < 2:
        raise ValueError("frame_row_offsets must contain at least two entries.")
    if int(offset_values[0]) != 0 or int(offset_values[-1]) != int(frames.size):
        raise ValueError("frame_row_offsets endpoints differ from the row count.")
    counts = np.diff(offset_values)
    if np.any(counts < 0):
        raise ValueError("frame_row_offsets is not monotone.")
    n_frames = int(counts.size)
    if frames.size and (
        int(frames.min()) < 0
        or int(frames.max()) >= n_frames
        or np.any(np.diff(frames) < 0)
    ):
        raise ValueError(
            "source_acquisition_frame_index is out of range or not sorted."
        )
    observed = np.bincount(frames, minlength=n_frames).astype(np.int64, copy=False)
    if observed.shape != counts.shape or not np.array_equal(observed, counts):
        raise ValueError(
            "frame_row_offsets does not exactly index source acquisition frames."
        )
    empty = np.flatnonzero(counts == 0)
    multi = np.flatnonzero(counts > 1)
    if require_empty_frame and empty.size == 0:
        raise ValueError(
            "Canary contains no empty frame required by the acceptance gate."
        )
    if require_multi_row_frame and multi.size == 0:
        raise ValueError(
            "Canary contains no multi-row frame required by the acceptance gate."
        )

    def example(indices: np.ndarray) -> dict[str, int] | None:
        if indices.size == 0:
            return None
        frame = int(indices[0])
        return {
            "frame_index": frame,
            "row_start": int(offset_values[frame]),
            "row_stop": int(offset_values[frame + 1]),
            "row_count": int(counts[frame]),
        }

    return {
        "valid": True,
        "n_frames": n_frames,
        "n_rows": int(frames.size),
        "empty_frame_count": int(empty.size),
        "multi_row_frame_count": int(multi.size),
        "maximum_rows_per_frame": int(counts.max(initial=0)),
        "empty_frame_example": example(empty),
        "multi_row_frame_example": example(multi),
        "requirements": {
            "empty_frame_required": bool(require_empty_frame),
            "multi_row_frame_required": bool(require_multi_row_frame),
        },
    }


def require_v5_identity(
    attrs: Mapping[str, Any],
    *,
    bundle_id: str,
) -> None:
    expected = {
        "schema_id": CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
        "schema_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
        "method": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
        "method_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
        "row_axis": "recording_subject_mask_bundle_rows",
        "subject_shape_source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
        SUBJECT_SHAPE_BUNDLE_ID_ATTR: bundle_id,
        SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR: False,
        "palette_run_completion_status": "complete",
        "stage_selector_eligible": False,
    }
    differences = {
        name: {"expected": value, "observed": attrs.get(name)}
        for name, value in expected.items()
        if attrs.get(name) != value
    }
    if differences:
        raise ValueError(f"Subject-shape v5 identity differs: {differences!r}.")
    source_digest = attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR)
    if (
        type(source_digest) is not str
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise ValueError("Subject-shape v5 source-binding digest is invalid.")


def _iter_arrays(group: Any, prefix: str = ""):
    for name in sorted(str(value) for value in group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_arrays(group[name], path)


def _array_at_path(group: Any, path: str) -> Any:
    node = group
    for component in path.split("/"):
        node = node[component]
    return node


def _physical_declaration(array: Any) -> dict[str, Any]:
    metadata = array.metadata.to_dict()
    shape = [int(value) for value in array.shape]
    dtype = np.dtype(array.dtype)
    grid = metadata.get("chunk_grid", {})
    grid_configuration = (
        grid.get("configuration", {}) if isinstance(grid, Mapping) else {}
    )
    outer_shape = grid_configuration.get("chunk_shape")
    codecs = metadata.get("codecs")
    inner_shape: list[int] | None = None
    if isinstance(codecs, list):
        for codec in codecs:
            if (
                not isinstance(codec, Mapping)
                or codec.get("name") != "sharding_indexed"
            ):
                continue
            configuration = codec.get("configuration")
            if isinstance(configuration, Mapping):
                value = configuration.get("chunk_shape")
                if isinstance(value, list):
                    inner_shape = [int(item) for item in value]
            break

    def logical_bytes(chunk_shape: Any) -> int | None:
        if not isinstance(chunk_shape, list):
            return None
        return int(math.prod(int(value) for value in chunk_shape) * dtype.itemsize)

    return {
        "dtype": dtype.str,
        "shape": shape,
        "outer_chunk_or_shard_shape": outer_shape,
        "inner_chunk_shape": inner_shape,
        "outer_logical_bytes": logical_bytes(outer_shape),
        "inner_logical_bytes": logical_bytes(inner_shape),
        "codecs": codecs,
    }


def _timed_read(array: Any, selection: Any) -> tuple[int, float]:
    started = time.perf_counter()
    values = np.asarray(array[selection])
    duration = float(time.perf_counter() - started)
    return int(values.nbytes), duration


def _random_frame_workload(
    run: Any,
    *,
    offsets: np.ndarray,
    frame_count: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_frames = int(offsets.size - 1)
    selected = rng.integers(0, n_frames, size=frame_count, dtype=np.int64)
    arrays = {path: _array_at_path(run, path) for path in _HOT_ARRAY_PATHS}

    def pass_once() -> dict[str, Any]:
        durations: list[float] = []
        decoded_bytes = 0
        for frame_value in selected:
            frame = int(frame_value)
            start = int(offsets[frame])
            stop = int(offsets[frame + 1])
            frame_started = time.perf_counter()
            for array in arrays.values():
                values = np.asarray(array[start:stop])
                decoded_bytes += int(values.nbytes)
            durations.append(float(time.perf_counter() - frame_started))
        return {
            "frame_count": int(selected.size),
            "decoded_bytes": decoded_bytes,
            "median_ms": float(statistics.median(durations) * 1000.0),
            "p95_ms": float(_percentile(durations, 95.0) * 1000.0),
            "maximum_ms": float(max(durations, default=0.0) * 1000.0),
        }

    first = pass_once()
    second = pass_once()
    return {
        "seed": int(seed),
        "array_paths": list(_HOT_ARRAY_PATHS),
        "first_pass": first,
        "warm_pass": second,
    }


def _window_workload(
    run: Any,
    *,
    row_count: int,
    window_rows: int,
    window_count: int,
    seed: int,
) -> dict[str, Any]:
    width = min(max(1, int(window_rows)), max(1, row_count))
    upper = max(1, row_count - width + 1)
    rng = np.random.default_rng(seed + 1)
    starts = rng.integers(0, upper, size=window_count, dtype=np.int64)
    arrays = {path: _array_at_path(run, path) for path in _HOT_ARRAY_PATHS}
    durations: list[float] = []
    decoded_bytes = 0
    for start_value in starts:
        start = int(start_value)
        stop = min(row_count, start + width)
        window_started = time.perf_counter()
        for array in arrays.values():
            values = np.asarray(array[start:stop])
            decoded_bytes += int(values.nbytes)
        durations.append(float(time.perf_counter() - window_started))
    return {
        "window_count": int(window_count),
        "window_rows": int(width),
        "decoded_bytes": decoded_bytes,
        "median_ms": float(statistics.median(durations) * 1000.0),
        "p95_ms": float(_percentile(durations, 95.0) * 1000.0),
        "maximum_ms": float(max(durations, default=0.0) * 1000.0),
        "array_paths": list(_HOT_ARRAY_PATHS),
    }


def _consumer_evidence(
    root: Any,
    run: Any,
    *,
    run_name: str,
    publication: Any,
    row_count: int,
) -> dict[str, Any]:
    eye_authority = eye_consumer._build_staged_subject_shape_authority(
        run,
        run_name=run_name,
        publication=publication,
    )
    eye = eye_consumer.resolve_eye_geometry_source(
        root,
        subject_shape_run=run_name,
        _staged_subject_shape_authority=eye_authority,
    )
    if (
        eye.stage_group != RUN_PARENT
        or eye.run_name != run_name
        or int(eye.eye_separation.shape[0]) != row_count
    ):
        raise ValueError("Eye-geometry consumer resolved a different v5 authority.")
    eye_rows = min(row_count, 32)
    eye_sample = {
        "ellipse_params_shape": list(np.asarray(eye.ellipse_params[:eye_rows]).shape),
        "ellipse_success_shape": list(np.asarray(eye.ellipse_success[:eye_rows]).shape),
        "eye_separation_shape": list(np.asarray(eye.eye_separation[:eye_rows]).shape),
    }

    body = run["components/subject_body"]
    source_sample_count = int(body["tail_sample_s"].shape[0])
    tail_authority = tail_consumer._build_staged_source_authority(
        run,
        run_name=run_name,
        row_count=row_count,
        source_sample_count=source_sample_count,
        publication=publication,
    )
    tail_plan = tail_consumer.write_tail_kinematics_run_group(
        root,
        shape_run=run_name,
        run_name="subject_shape_v5_acceptance_tail_plan",
        block_rows=1_024,
        output_shard_rows=131_072,
        execution_backend="serial",
        num_workers=1,
        dry_run=True,
        _staged_source_authority=tail_authority,
    )
    if (
        tail_plan.get("status") != "planned"
        or tail_plan.get("source_subject_shape_run") != run_name
        or tail_plan.get("roi_count") != row_count
        or tail_plan.get("mutates_archive") is not False
    ):
        raise ValueError("Tail-kinematics dry-run resolved a different v5 authority.")
    return {
        "eye_geometry": {
            "status": "passed",
            "source_authority_mode": eye.source_authority_mode,
            "sample_rows": eye_rows,
            "sample": eye_sample,
        },
        "tail_kinematics": {
            "status": "passed",
            "source_authority_mode": tail_plan.get(
                "source_subject_shape_authority_mode"
            ),
            "dry_run": tail_plan,
        },
    }


def _publication_summary(path: Path | None, *, run_name: str) -> dict[str, Any]:
    if path is None:
        return {
            "availability": "not_supplied",
            "status": None,
            "publication_seconds": None,
            "producer_duration_seconds": None,
        }
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("Publication report must be one JSON object.")
    plan = document.get("plan")
    if (
        document.get("status") != "complete"
        or not isinstance(plan, Mapping)
        or plan.get("run_name") != run_name
    ):
        raise ValueError("Publication report does not bind the completed v5 run.")
    publish = document.get("publish")
    local = document.get("local_materialization")
    copy_seconds = None
    producer_seconds = None
    if isinstance(publish, Mapping):
        value = publish.get("copy_duration_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            copy_seconds = float(value)
    if isinstance(local, Mapping):
        compute = local.get("node_local_compute")
        if isinstance(compute, Mapping):
            value = compute.get("duration_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                producer_seconds = float(value)
    return {
        "availability": "exact_materializer_report",
        "status": "complete",
        "report_path": str(path),
        "report_sha256": canonical_json_sha256(document),
        "publication_seconds": copy_seconds,
        "producer_duration_seconds": producer_seconds,
        "peak_rss_bytes": None,
        "peak_rss_availability": (
            "not_captured_by_materializer; collect from immutable LSF accounting"
        ),
    }


def require_acceptance_result(value: Mapping[str, Any]) -> None:
    if set(value) != {"schema_id", "schema_version", "payload", "payload_digest"}:
        raise ValueError("Acceptance result envelope field set is invalid.")
    if (
        value.get("schema_id") != RESULT_SCHEMA_ID
        or value.get("schema_version") != RESULT_SCHEMA_VERSION
        or not isinstance(value.get("payload"), Mapping)
        or value.get("payload_digest") != canonical_json_sha256(value["payload"])
    ):
        raise ValueError("Acceptance result envelope identity/digest is invalid.")
    payload = value["payload"]
    if payload.get("status") != "passed":
        raise ValueError("Acceptance result did not pass.")
    json.dumps(value, allow_nan=False)


def run_acceptance(
    analysis_zarr: Path | str,
    *,
    run_name: str,
    bundle_id: str,
    publication_report: Path | None = None,
    require_empty_frame: bool = True,
    require_multi_row_frame: bool = True,
    random_frame_count: int = DEFAULT_RANDOM_FRAME_COUNT,
    window_count: int = DEFAULT_WINDOW_COUNT,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    archive = _safe_archive(analysis_zarr)
    selected_run = _safe_run_name(run_name, label="run_name")
    selected_bundle = _safe_run_name(bundle_id, label="bundle_id")
    if random_frame_count <= 0 or window_count <= 0 or window_rows <= 0 or seed < 0:
        raise ValueError("Benchmark counts must be positive and seed nonnegative.")
    run_path = f"{RUN_PARENT}/{selected_run}"
    started_at = utc_now()
    started_wall = time.perf_counter()
    initial_rss = peak_rss_bytes()

    (direct_root, direct_run), direct_open = _measure(
        lambda: (
            (
                root := zarr.open_group(
                    str(archive), mode="r", zarr_format=3, use_consolidated=False
                )
            ),
            root[run_path],
        )
    )
    (consolidated_root, consolidated_run), consolidated_open = _measure(
        lambda: (
            (
                root := zarr.open_group(
                    str(archive), mode="r", zarr_format=3, use_consolidated=True
                )
            ),
            root[run_path],
        )
    )

    def validate_contracts() -> dict[str, Any]:
        owner = direct_run.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
        if type(owner) is not str or not owner:
            raise ValueError("Subject-shape v5 canary lacks its publication owner.")
        require_v5_identity(direct_run.attrs, bundle_id=selected_bundle)
        publication = load_completed_ineligible_subject_shape_coordinate_publication(
            direct_root,
            run_path,
            expected_publication_owner=owner,
        )
        source = load_subject_shape_bundle_source(
            archive,
            bundle_id=selected_bundle,
            allow_inactive=True,
        )
        if (
            source.active is not False
            or direct_run.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR)
            != source.source_digest
            or publication.row_identity.leading_dimension != source.row_count
        ):
            raise ValueError("Subject-shape v5 source bundle binding differs.")
        parent = direct_root[RUN_PARENT]
        selected_by = [
            name for name in _SELECTOR_ATTRS if parent.attrs.get(name) == selected_run
        ]
        if selected_by:
            raise ValueError(
                f"Selector-ineligible v5 canary is selected by {selected_by!r}."
            )
        storage_errors = validate_subject_shape_candidate_storage(
            direct_run, phase="bound"
        )
        if storage_errors:
            raise ValueError(
                f"Subject-shape storage contract failed: {storage_errors!r}."
            )
        metadata_errors = validate_subject_shape_direct_consolidated_storage(
            archive,
            run_path=run_path,
            phase="bound",
        )
        if metadata_errors:
            raise ValueError(
                f"Direct/consolidated storage differs: {metadata_errors!r}."
            )
        metadata = validate_direct_consolidated_subtree(archive, subtree_path=run_path)
        manifest = direct_run.attrs.get(SUBJECT_SHAPE_MANIFEST_ATTR)
        if not isinstance(manifest, Mapping):
            raise ValueError("Subject-shape publication manifest is absent.")
        manifest_sha256 = coordinate_record_sha256(manifest)
        return {
            "owner": owner,
            "publication": publication,
            "source": source,
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
            "metadata": metadata,
            "selectors": {name: parent.attrs.get(name) for name in _SELECTOR_ATTRS},
        }

    contract, contract_timing = _measure(validate_contracts)
    source = contract["source"]
    offsets = np.asarray(source.authority.frame_row_offsets_node[:])
    frame_indices = np.asarray(source.authority.source_acquisition_frame_index_node[:])
    frame_cardinality, frame_timing = _measure(
        lambda: validate_frame_cardinality_arrays(
            offsets,
            frame_indices,
            require_empty_frame=require_empty_frame,
            require_multi_row_frame=require_multi_row_frame,
        )
    )
    if int(offsets[-1]) != int(contract["publication"].row_identity.leading_dimension):
        raise ValueError("Frame index row count differs from v5 publication identity.")

    consumers, consumer_timing = _measure(
        lambda: _consumer_evidence(
            direct_root,
            direct_run,
            run_name=selected_run,
            publication=contract["publication"],
            row_count=int(offsets[-1]),
        )
    )
    random_reads = _random_frame_workload(
        consolidated_run,
        offsets=offsets,
        frame_count=int(random_frame_count),
        seed=int(seed),
    )
    window_reads = _window_workload(
        consolidated_run,
        row_count=int(offsets[-1]),
        window_rows=int(window_rows),
        window_count=int(window_count),
        seed=int(seed),
    )
    logical_hashes, traversal_timing = _measure(
        lambda: compute_subject_shape_logical_hashes(consolidated_run)
    )
    manifest_arrays = contract["manifest"].get("arrays")
    expected_hashes = (
        {
            str(path): str(record["content_sha256"])
            for path, record in manifest_arrays.items()
        }
        if isinstance(manifest_arrays, Mapping)
        else {}
    )
    if logical_hashes != expected_hashes:
        raise ValueError("Full traversal hashes differ from the sealed v5 manifest.")

    physical_arrays = {
        path: _physical_declaration(array) for path, array in _iter_arrays(direct_run)
    }
    if set(physical_arrays) != set(logical_hashes):
        raise ValueError("Physical and logical closed array inventories differ.")
    payload = {
        "status": "passed",
        "classification": "selector_ineligible_recording_scale_canary",
        "archive_path": str(archive),
        "run_name": selected_run,
        "run_path": run_path,
        "bundle_id": selected_bundle,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "contract": {
            "valid": True,
            "profile_id": CANONICAL_SUBJECT_SHAPE_BUNDLE_PROFILE_ID,
            "schema_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
            "method": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
            "method_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
            "manifest_sha256": contract["manifest_sha256"],
            "source_binding_sha256": source.source_digest,
            "bundle_active": source.active,
            "selector_eligible": False,
            "parent_selectors": contract["selectors"],
            "validation_timing": contract_timing,
        },
        "frame_cardinality": {**frame_cardinality, "timing": frame_timing},
        "metadata": {
            "direct_open": direct_open,
            "consolidated_open": consolidated_open,
            "equivalent": True,
            "receipt": contract["metadata"].to_json(),
        },
        "storage": {
            "profile_id": SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            "tree": storage_stats(archive.joinpath(*run_path.split("/"))),
            "array_count": len(physical_arrays),
            "arrays": physical_arrays,
        },
        "read_workloads": {
            "random_frames": random_reads,
            "windowed_rows": window_reads,
            "full_traversal": {
                "array_count": len(logical_hashes),
                "logical_manifest_sha256": canonical_json_sha256(logical_hashes),
                "timing": traversal_timing,
            },
            "physical_io": {
                "request_count": None,
                "transferred_bytes": None,
                "availability": (
                    "unavailable_without_filesystem_or_tensorstore_tracing"
                ),
            },
        },
        "consumers": {**consumers, "timing": consumer_timing},
        "publication": _publication_summary(publication_report, run_name=selected_run),
        "runtime": {
            "hostname": platform.node(),
            "system": platform.system(),
            "python": platform.python_version(),
            "zarr": zarr.__version__,
            "initial_peak_rss_bytes": initial_rss,
            "final_peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_is_process_high_water_mark": True,
            "total_wall_seconds": float(time.perf_counter() - started_wall),
        },
        "promotion": {
            "selector_promoted": False,
            "physical_profile_promoted": False,
            "policy": "separate_recorded_decision_with_rollback_required",
        },
    }
    result = {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_acceptance_result(result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--publication-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-empty-frame",
        action="store_true",
        help="Diagnostic-only relaxation; a promotion receipt must not use it.",
    )
    parser.add_argument(
        "--allow-missing-multi-row-frame",
        action="store_true",
        help="Diagnostic-only relaxation; a promotion receipt must not use it.",
    )
    parser.add_argument(
        "--random-frame-count", type=int, default=DEFAULT_RANDOM_FRAME_COUNT
    )
    parser.add_argument("--window-count", type=int, default=DEFAULT_WINDOW_COUNT)
    parser.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    archive = _safe_archive(args.analysis_zarr)
    output = _safe_output(args.output, archive=archive)
    report = (
        args.publication_report.expanduser().resolve()
        if args.publication_report is not None
        else None
    )
    result = run_acceptance(
        archive,
        run_name=args.run_name,
        bundle_id=args.bundle_id,
        publication_report=report,
        require_empty_frame=not args.allow_missing_empty_frame,
        require_multi_row_frame=not args.allow_missing_multi_row_frame,
        random_frame_count=int(args.random_frame_count),
        window_count=int(args.window_count),
        window_rows=int(args.window_rows),
        seed=int(args.seed),
    )
    _write_json_atomic(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
