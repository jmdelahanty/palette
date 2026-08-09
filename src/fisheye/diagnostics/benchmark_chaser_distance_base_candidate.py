"""Benchmark one sealed chaser-distance source/candidate pair read-only.

The controller launches one fresh process per source or candidate trial and
rotates their order by repetition.  It never opens the archive for mutation;
all evidence is written to a new benchmark-only directory outside the archive.
"""

from __future__ import annotations

import argparse
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

from fisheye.analysis.chaser_distance_base_schema import (
    SEALED_CHASER_DISTANCE_BASE_PATHS,
    build_chaser_distance_base_declarations,
    validate_chaser_distance_base_semantics,
)
from fisheye.analysis.chaser_distance_base_storage import (
    BASE_MANIFEST_ATTR,
    base_logical_hashes,
    build_source_authority_binding,
    validate_base_candidate,
)
from fisheye.analysis.chaser_distance_coordinate_publication import (
    load_bound_chaser_distance_run,
)
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis_workflows.materializers.chaser_distance_base import (
    PUBLISH_SCHEMA_ID,
)
from fisheye.shared.atomic_run_publisher import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
    ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
    ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
    SERIALIZATION_POLICY,
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
from fisheye.shared.zarr.benchmark_runtime import (
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)


SOURCE_PARENT_PATH = "analysis/chaser_distance_runs"
CANDIDATE_PARENT_PATH = "analysis/chaser_distance_storage_candidates"
FAMILY_ID = "chaser_distance_sealed_base"
BENCHMARK_ID = "chaser_distance_sealed_base_read_matrix_v1"
TRIAL_SCHEMA_ID = "palette.chaser_distance.sealed_base_read_trial"
MATRIX_SCHEMA_ID = "palette.chaser_distance.sealed_base_read_matrix"
SCHEMA_VERSION = 1
DEFAULT_SEED = 17
DEFAULT_REPETITIONS = 5
FULL_SCAN_TARGET_BLOCK_BYTES = 8 * 1024 * 1024
_SHA256_LENGTH = 64
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_TRIAL_PHYSICAL_AVAILABILITY = (
    "unavailable_without_os_or_filesystem_tracing; logical decoded bytes "
    "and file counts are not physical transfer telemetry"
)
_MATRIX_PHYSICAL_AVAILABILITY = "unavailable_without_os_or_filesystem_tracing"
_PUBLISH_POLICY = "sealed_base_byte_planned_atomic_nonpromoting_publish"
_ROLLBACK_POLICY = "retain_failed_public_tombstone_leave_all_selectors_untouched"
_PUBLISH_RECEIPT_FIELDS = {
    "profile_id",
    "source_run_path",
    "source_binding",
    "source_logical_hashes",
    "local_direct_consolidated_array_count",
    "materialization_seconds",
    "schema_id",
    "publisher_contract",
    "policy",
    "serialization_policy",
    "rollback_policy",
    "published_at_utc",
    "host",
    "lsb_jobid",
    "source_zarr",
    "publication_source_run_path",
    "target_run_path",
    "publication_owner_attr",
    "publication_owner_uuid",
    "failed_public_child_policy",
    "hidden_temporary_policy",
    "copy_duration_seconds",
    "physical_copy",
    "parent_attrs_before",
    "local_validation",
    "temporary_validation",
    "pre_pointer_validation",
    "final_validation",
    "parent_attrs_after",
}


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
        raise ValueError("Benchmark evidence envelope has an unexpected field set.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_timing(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "wall_seconds",
        "cpu_seconds",
    }:
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


def require_trial_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "source_parent_path",
        "candidate_parent_path",
        "source_run_name",
        "candidate_run_name",
        "role",
        "run_name",
        "run_path",
        "repetition_index",
        "order_position",
        "seed",
        "cache_state",
        "suite_payload_digest",
        "candidate_storage_receipt_payload_digest",
        "started_at_utc",
        "finished_at_utc",
        "environment",
        "validation",
        "metadata",
        "primary_access",
        "full_scan",
        "logical_arrays",
        "storage",
        "publication_timing",
        "runtime",
        "physical_io",
    }
    if set(payload) != expected:
        raise ValueError("Chaser-distance trial payload has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID or payload["family_id"] != FAMILY_ID:
        raise ValueError("Chaser-distance trial benchmark identity mismatch.")
    if (
        payload["source_parent_path"] != SOURCE_PARENT_PATH
        or payload["candidate_parent_path"] != CANDIDATE_PARENT_PATH
    ):
        raise ValueError("Chaser-distance trial parent binding mismatch.")
    if payload["source_run_name"] == payload["candidate_run_name"]:
        raise ValueError("Chaser-distance source/candidate names must differ.")
    role = payload["role"]
    if role not in {"source", "candidate"}:
        raise ValueError("Chaser-distance trial role is unsupported.")
    run_name_field = f"{role}_run_name"
    parent = SOURCE_PARENT_PATH if role == "source" else CANDIDATE_PARENT_PATH
    if (
        payload["run_name"] != payload[run_name_field]
        or payload["run_path"] != f"{parent}/{payload['run_name']}"
    ):
        raise ValueError("Chaser-distance trial role/run/path binding mismatch.")
    if type(payload["repetition_index"]) is not int or payload["repetition_index"] < 0:
        raise ValueError("Chaser-distance trial repetition index is invalid.")
    if payload["order_position"] not in {0, 1}:
        raise ValueError("Chaser-distance trial order position is invalid.")
    for field in ("suite_payload_digest", "candidate_storage_receipt_payload_digest"):
        if not _is_sha256(payload[field]):
            raise ValueError(f"Chaser-distance trial {field} is invalid.")
    validation = payload["validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "valid",
        "role",
        "array_count",
        "source_binding_sha256",
        "candidate_manifest_payload_digest",
        "timing",
    }:
        raise ValueError("Chaser-distance trial validation receipt is malformed.")
    if (
        validation["valid"] is not True
        or validation["role"] != role
        or validation["array_count"] != len(SEALED_CHASER_DISTANCE_BASE_PATHS)
        or not _is_sha256(validation["source_binding_sha256"])
    ):
        raise ValueError("Chaser-distance trial validation did not pass exactly.")
    candidate_digest = validation["candidate_manifest_payload_digest"]
    if (role == "candidate") != _is_sha256(candidate_digest):
        raise ValueError("Candidate manifest digest role binding is invalid.")
    _require_timing(validation["timing"], label="validation")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "equivalent",
        "array_count",
        "group_count",
        "node_count",
        "subtree_declarations_digest",
        "direct_open",
        "consolidated_open",
        "comparison",
    }:
        raise ValueError("Chaser-distance trial metadata receipt is malformed.")
    if metadata["equivalent"] is not True or not _is_sha256(
        metadata["subtree_declarations_digest"]
    ):
        raise ValueError("Chaser-distance trial metadata equivalence did not pass.")
    for field in ("direct_open", "consolidated_open", "comparison"):
        _require_timing(metadata[field], label=f"metadata {field}")
    for field in ("array_count", "group_count", "node_count"):
        if type(metadata[field]) is not int or metadata[field] < 1:
            raise ValueError(f"Chaser-distance metadata {field} is invalid.")
    if (
        metadata["array_count"] < len(SEALED_CHASER_DISTANCE_BASE_PATHS)
        or (
            role == "candidate"
            and metadata["array_count"] != len(SEALED_CHASER_DISTANCE_BASE_PATHS)
        )
        or metadata["node_count"]
        != metadata["array_count"] + metadata["group_count"]
    ):
        raise ValueError("Chaser-distance metadata inventory totals are inconsistent.")
    logical = payload["logical_arrays"]
    if not isinstance(logical, Mapping) or set(logical) != set(
        SEALED_CHASER_DISTANCE_BASE_PATHS
    ):
        raise ValueError("Chaser-distance trial logical inventory is not exact.")
    for path, record in logical.items():
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "shape",
            "logical_digest",
        }:
            raise ValueError(f"Logical record {path!r} is malformed.")
        if not _is_sha256(record["logical_digest"]):
            raise ValueError(f"Logical record {path!r} digest is invalid.")
    primary = payload["primary_access"]
    full_scan = payload["full_scan"]
    if (
        not isinstance(primary, Mapping)
        or set(primary) != {"arrays", "total_wall_seconds", "total_cpu_seconds"}
        or set(primary["arrays"]) != set(SEALED_CHASER_DISTANCE_BASE_PATHS)
    ):
        raise ValueError("Chaser-distance primary-access evidence is not exact.")
    if (
        not isinstance(full_scan, Mapping)
        or set(full_scan)
        != {"arrays", "total_wall_seconds", "total_cpu_seconds", "total_decoded_bytes"}
        or set(full_scan["arrays"]) != set(SEALED_CHASER_DISTANCE_BASE_PATHS)
    ):
        raise ValueError("Chaser-distance full-scan evidence is not exact.")
    for field in ("total_wall_seconds", "total_cpu_seconds"):
        observed = primary[field]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"Chaser-distance primary {field} is invalid.")
        observed = full_scan[field]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"Chaser-distance full scan {field} is invalid.")
    if type(full_scan["total_decoded_bytes"]) is not int or full_scan[
        "total_decoded_bytes"
    ] < 0:
        raise ValueError("Chaser-distance full-scan byte count is invalid.")
    for path, record in primary["arrays"].items():
        if not isinstance(record, Mapping) or set(record) != {
            "mode",
            "execution_axis",
            "operation_count",
            "decoded_bytes",
            "selection_digest",
            "workload_id",
            "selection",
            "timing",
        }:
            raise ValueError(f"Primary access record {path!r} is malformed.")
        if (
            record["mode"] not in {"whole_array", "bounded_row_windows"}
            or type(record["execution_axis"]) is not int
            or type(record["operation_count"]) is not int
            or record["operation_count"] < 0
            or type(record["decoded_bytes"]) is not int
            or record["decoded_bytes"] < 0
            or not _is_sha256(record["selection_digest"])
            or not isinstance(record["selection"], Mapping)
            or record["selection"].get("mode") != record["mode"]
        ):
            raise ValueError(f"Primary access record {path!r} is invalid.")
        _require_timing(record["timing"], label=f"primary {path}")
    for path, record in full_scan["arrays"].items():
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "shape",
            "decoded_bytes",
            "block_count",
            "logical_digest",
            "timing",
        }:
            raise ValueError(f"Full-scan record {path!r} is malformed.")
        if (
            type(record["dtype"]) is not str
            or not isinstance(record["shape"], list)
            or type(record["decoded_bytes"]) is not int
            or record["decoded_bytes"] < 0
            or type(record["block_count"]) is not int
            or record["block_count"] < 0
            or not _is_sha256(record["logical_digest"])
        ):
            raise ValueError(f"Full-scan record {path!r} is invalid.")
        if logical[path] != {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "logical_digest": record["logical_digest"],
        }:
            raise ValueError(f"Logical/full-scan record {path!r} binding mismatch.")
        _require_timing(record["timing"], label=f"full scan {path}")
    if (
        primary["total_wall_seconds"]
        != sum(
            float(record["timing"]["wall_seconds"])
            for record in primary["arrays"].values()
        )
        or primary["total_cpu_seconds"]
        != sum(
            float(record["timing"]["cpu_seconds"])
            for record in primary["arrays"].values()
        )
        or full_scan["total_wall_seconds"]
        != sum(
            float(record["timing"]["wall_seconds"])
            for record in full_scan["arrays"].values()
        )
        or full_scan["total_cpu_seconds"]
        != sum(
            float(record["timing"]["cpu_seconds"])
            for record in full_scan["arrays"].values()
        )
        or full_scan["total_decoded_bytes"]
        != sum(
            int(record["decoded_bytes"])
            for record in full_scan["arrays"].values()
        )
    ):
        raise ValueError("Chaser-distance read aggregate totals are not reconstructed.")
    storage = payload["storage"]
    if not isinstance(storage, Mapping) or set(storage) != {
        "scope",
        "totals",
        "payload_object_count",
        "by_array",
        "whole_run_tree",
    }:
        raise ValueError("Chaser-distance storage evidence is malformed.")
    if (
        storage["scope"] != "exact_30_array_directories; group metadata excluded"
        or set(storage["by_array"]) != set(SEALED_CHASER_DISTANCE_BASE_PATHS)
    ):
        raise ValueError("Chaser-distance storage evidence scope is invalid.")
    stat_fields = {
        "file_count",
        "metadata_file_count",
        "payload_file_count",
        "apparent_bytes",
        "allocated_bytes",
    }
    for label, record in (
        ("totals", storage["totals"]),
        ("whole_run_tree", storage["whole_run_tree"]),
        *((f"by_array/{path}", record) for path, record in storage["by_array"].items()),
    ):
        if (
            not isinstance(record, Mapping)
            or set(record) != stat_fields
            or any(type(record[field]) is not int or record[field] < 0 for field in stat_fields)
        ):
            raise ValueError(f"Chaser-distance storage {label} is invalid.")
    reconstructed_totals = {
        field: sum(record[field] for record in storage["by_array"].values())
        for field in stat_fields
    }
    if (
        dict(storage["totals"]) != reconstructed_totals
        or storage["payload_object_count"] != reconstructed_totals["payload_file_count"]
    ):
        raise ValueError("Chaser-distance selected storage totals are invalid.")
    publication = payload["publication_timing"]
    if not isinstance(publication, Mapping) or set(publication) != {
        "availability",
        "publish_schema_id",
        "materialization_seconds",
        "copy_duration_seconds",
        "published_at_utc",
    }:
        raise ValueError("Chaser-distance publication timing is malformed.")
    if role == "source":
        if publication != {
            "availability": "not_applicable_source",
            "publish_schema_id": None,
            "materialization_seconds": None,
            "copy_duration_seconds": None,
            "published_at_utc": None,
        }:
            raise ValueError("Source publication timing role binding is invalid.")
    elif publication["availability"] == "recorded_in_cluster_output_staging":
        if (
            publication["publish_schema_id"] != PUBLISH_SCHEMA_ID
            or type(publication["published_at_utc"]) is not str
            or not publication["published_at_utc"].strip()
        ):
            raise ValueError("Candidate publication timing schema is invalid.")
        for field in ("materialization_seconds", "copy_duration_seconds"):
            observed = publication[field]
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) < 0
            ):
                raise ValueError(f"Candidate publication {field} is invalid.")
    elif publication != {
        "availability": "not_recorded",
        "publish_schema_id": None,
        "materialization_seconds": None,
        "copy_duration_seconds": None,
        "published_at_utc": None,
    }:
        raise ValueError("Candidate publication timing availability is invalid.")
    runtime = payload["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "initial_peak_rss_bytes",
        "final_peak_rss_bytes",
        "peak_rss_is_process_high_water_mark",
        "total_wall_seconds",
        "total_cpu_seconds",
    }:
        raise ValueError("Chaser-distance runtime evidence is malformed.")
    if (
        runtime["peak_rss_is_process_high_water_mark"] is not True
        or type(runtime["initial_peak_rss_bytes"]) is not int
        or type(runtime["final_peak_rss_bytes"]) is not int
        or runtime["initial_peak_rss_bytes"] < 0
        or runtime["final_peak_rss_bytes"] < runtime["initial_peak_rss_bytes"]
    ):
        raise ValueError("Chaser-distance RSS evidence is invalid.")
    for field in ("total_wall_seconds", "total_cpu_seconds"):
        observed = runtime[field]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"Chaser-distance runtime {field} is invalid.")
    physical = payload["physical_io"]
    if not isinstance(physical, Mapping) or set(physical) != {
        "request_count",
        "transferred_bytes",
        "availability",
    }:
        raise ValueError("Chaser-distance physical-I/O evidence is malformed.")
    if (
        physical["request_count"] is not None
        or physical["transferred_bytes"] is not None
        or physical["availability"] != _TRIAL_PHYSICAL_AVAILABILITY
    ):
        raise ValueError("This runner must not fabricate physical I/O telemetry.")


def require_matrix_result(value: Mapping[str, Any]) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    expected = {
        "benchmark_id",
        "family_id",
        "archive_path",
        "source_parent_path",
        "candidate_parent_path",
        "source_run_name",
        "candidate_run_name",
        "seed",
        "repetitions",
        "cache_state",
        "started_at_utc",
        "finished_at_utc",
        "suite",
        "candidate_storage_receipt_payload_digest",
        "source_binding_sha256",
        "candidate_manifest_payload_digest",
        "trial_order",
        "trial_files",
        "trials",
        "correctness",
        "performance_summary",
        "archive_read_only_metadata_guard",
        "physical_io",
        "promotion_decision",
    }
    if set(payload) != expected:
        raise ValueError("Chaser-distance matrix payload has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID or payload["family_id"] != FAMILY_ID:
        raise ValueError("Chaser-distance matrix benchmark identity mismatch.")
    if (
        payload["source_parent_path"] != SOURCE_PARENT_PATH
        or payload["candidate_parent_path"] != CANDIDATE_PARENT_PATH
    ):
        raise ValueError("Chaser-distance matrix parent binding mismatch.")
    if payload["source_run_name"] == payload["candidate_run_name"]:
        raise ValueError("Chaser-distance source/candidate names must differ.")
    if type(payload["repetitions"]) is not int or payload["repetitions"] < 1:
        raise ValueError("Chaser-distance matrix repetition count is invalid.")
    require_analysis_benchmark_suite_manifest(payload["suite"])
    suite = payload["suite"]
    if (
        suite["payload"]["family_id"] != FAMILY_ID
        or suite["payload"]["seed"] != payload["seed"]
        or suite["payload"]["repetitions"] != payload["repetitions"]
        or suite["payload"]["storage_plan_receipt"]["payload_digest"]
        != payload["candidate_storage_receipt_payload_digest"]
        or not _is_sha256(payload["source_binding_sha256"])
        or not _is_sha256(payload["candidate_manifest_payload_digest"])
    ):
        raise ValueError("Chaser-distance matrix suite binding mismatch.")
    expected_order = [
        {
            "repetition_index": index,
            "roles": list(_trial_order(seed=payload["seed"], repetition_index=index)),
        }
        for index in range(payload["repetitions"])
    ]
    if payload["trial_order"] != expected_order:
        raise ValueError("Chaser-distance matrix trial order is not deterministic v1.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Chaser-distance matrix trial count is invalid.")
    for trial in trials:
        require_trial_result(trial)
        trial_payload = trial["payload"]
        if any(
            (
                trial_payload["archive_path"] != payload["archive_path"],
                trial_payload["source_parent_path"] != payload["source_parent_path"],
                trial_payload["candidate_parent_path"] != payload["candidate_parent_path"],
                trial_payload["source_run_name"] != payload["source_run_name"],
                trial_payload["candidate_run_name"] != payload["candidate_run_name"],
                trial_payload["seed"] != payload["seed"],
                trial_payload["cache_state"] != payload["cache_state"],
                trial_payload["suite_payload_digest"] != suite["payload_digest"],
                trial_payload["candidate_storage_receipt_payload_digest"]
                != payload["candidate_storage_receipt_payload_digest"],
                trial_payload["validation"]["source_binding_sha256"]
                != payload["source_binding_sha256"],
                trial_payload["validation"]["candidate_manifest_payload_digest"]
                not in {None, payload["candidate_manifest_payload_digest"]},
            )
        ):
            raise ValueError("Chaser-distance matrix/trial identity binding mismatch.")
    observed_order = [
        {
            "repetition_index": index,
            "roles": [
                trial["payload"]["role"]
                for trial in sorted(
                    (
                        item
                        for item in trials
                        if item["payload"]["repetition_index"] == index
                    ),
                    key=lambda item: item["payload"]["order_position"],
                )
            ],
        }
        for index in range(payload["repetitions"])
    ]
    if observed_order != expected_order:
        raise ValueError("Chaser-distance matrix trials differ from declared order.")
    trial_files = payload["trial_files"]
    expected_files = [
        f"trials/rep_{index:02d}_pos_{position}_{role}.json"
        for index in range(payload["repetitions"])
        for position, role in enumerate(
            _trial_order(seed=payload["seed"], repetition_index=index)
        )
    ]
    if trial_files != expected_files:
        raise ValueError("Chaser-distance matrix trial-file inventory is invalid.")
    logical_reference = trials[0]["payload"]["logical_arrays"]
    if not all(
        trial["payload"]["logical_arrays"] == logical_reference for trial in trials
    ):
        raise ValueError("Chaser-distance matrix full decoded values differ.")
    primary_reference = {
        path: record["selection_digest"]
        for path, record in trials[0]["payload"]["primary_access"]["arrays"].items()
    }
    if not all(
        {
            path: record["selection_digest"]
            for path, record in trial["payload"]["primary_access"]["arrays"].items()
        }
        == primary_reference
        for trial in trials
    ):
        raise ValueError("Chaser-distance matrix primary decoded values differ.")
    binding_digests = {
        trial["payload"]["validation"]["source_binding_sha256"] for trial in trials
    }
    if len(binding_digests) != 1:
        raise ValueError("Chaser-distance trials used different source authorities.")
    candidate_manifest_digests = {
        trial["payload"]["validation"]["candidate_manifest_payload_digest"]
        for trial in trials
        if trial["payload"]["role"] == "candidate"
    }
    if candidate_manifest_digests != {payload["candidate_manifest_payload_digest"]}:
        raise ValueError("Chaser-distance trials used a different candidate manifest.")
    suite_cases = {
        str(item["array_path"]): item
        for item in payload["suite"]["payload"]["array_cases"]
        if not str(item["case"]["workload"]["workload_id"]).endswith(
            ".write_materialization.v1"
        )
        and not str(item["case"]["workload"]["workload_id"]).endswith(
            ".full_scan_read.v1"
        )
    }
    if set(suite_cases) != set(SEALED_CHASER_DISTANCE_BASE_PATHS):
        raise ValueError("Chaser-distance suite primary inventory is not exact.")
    for trial in trials:
        primary_arrays = trial["payload"]["primary_access"]["arrays"]
        for path, case in suite_cases.items():
            if (
                primary_arrays[path]["workload_id"]
                != case["case"]["workload"]["workload_id"]
                or primary_arrays[path]["selection"] != case["selection"]
            ):
                raise ValueError(
                    "Chaser-distance trial primary access differs from suite receipt."
                )
    correctness = payload["correctness"]
    if not isinstance(correctness, Mapping) or correctness != {
        "full_decoded_logical_equality": True,
        "primary_access_decoded_equality": True,
        "direct_consolidated_metadata_equivalence": True,
        "source_and_candidate_validation": True,
        "archive_metadata_unchanged": True,
        "all_passed": True,
    }:
        raise ValueError("Chaser-distance correctness gates did not all pass.")
    guard = payload["archive_read_only_metadata_guard"]
    if (
        not isinstance(guard, Mapping)
        or set(guard) != {"before", "after", "unchanged"}
        or guard.get("unchanged") is not True
        or guard.get("before") != guard.get("after")
    ):
        raise ValueError("Chaser-distance archive metadata guard did not pass.")
    _require_metadata_guard(guard["before"])
    _require_metadata_guard(guard["after"])
    physical = payload["physical_io"]
    if (
        not isinstance(physical, Mapping)
        or set(physical) != {"request_count", "transferred_bytes", "availability"}
        or physical.get("request_count") is not None
        or physical.get("transferred_bytes") is not None
        or physical.get("availability") != _MATRIX_PHYSICAL_AVAILABILITY
    ):
        raise ValueError("Chaser-distance matrix fabricates physical I/O telemetry.")
    if payload["promotion_decision"] != {
        "authorized": False,
        "reason": "benchmark_only; profile promotion requires a separate reviewed decision",
    }:
        raise ValueError("Chaser-distance matrix cannot authorize promotion.")
    if payload["performance_summary"] != _summary(trials):
        raise ValueError("Chaser-distance matrix performance summary differs from trials.")


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


def _require_parent(value: str, *, expected: str, label: str) -> str:
    if value != expected:
        raise ValueError(f"{label} must be exact {expected!r}.")
    return value


def _safe_archive(path: Path | str) -> Path:
    archive = Path(path).expanduser().resolve()
    if not archive.is_dir() or not (archive / "zarr.json").is_file():
        raise FileNotFoundError(f"Analysis Zarr archive not found: {archive}.")
    return archive


def _safe_output(path: Path | str, *, archive: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}.")
    if output == archive or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError("Benchmark output must be disjoint from the source archive.")
    if not any("benchmark" in component.lower() for component in output.parts):
        raise ValueError("Benchmark output must have a benchmark-only path component.")
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("Benchmark output path is too broad.")
    return output


def _safe_trial_output(path: Path | str, *, root: Path) -> Path:
    output = Path(path).expanduser().resolve()
    benchmark_root = root.expanduser().resolve()
    if not benchmark_root.is_dir():
        raise FileNotFoundError("Benchmark root does not exist.")
    if output.exists() or not output.is_relative_to(benchmark_root) or output.suffix != ".json":
        raise ValueError("Trial output must be a new JSON file inside benchmark root.")
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    data = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace benchmark evidence: {path}.")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError("Benchmark JSON document must be one object.")
    return value


def _group(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _array(group: Any, path: str) -> Any:
    return _group(group, path)


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    wall = time.perf_counter()
    cpu = time.process_time()
    value = call()
    return value, {
        "wall_seconds": float(time.perf_counter() - wall),
        "cpu_seconds": float(time.process_time() - cpu),
    }


def _source_binding(root: Any, source_path: str, source_group: Any) -> dict[str, Any]:
    bound = load_bound_chaser_distance_run(root, source_path)
    return build_source_authority_binding(bound, source_group=source_group)


def _validate_pair(
    root: Any,
    *,
    source_path: str,
    candidate_path: str,
) -> dict[str, Any]:
    source = _group(root, source_path)
    candidate = _group(root, candidate_path)
    source_errors = validate_chaser_distance_base_semantics(source)
    if source_errors:
        raise ValueError(f"Invalid sealed source semantics: {source_errors!r}.")
    binding = _source_binding(root, source_path, source)
    validation = validate_base_candidate(
        candidate,
        source_group=source,
        expected_source_binding=binding,
    )
    if validation["valid"] is not True:
        raise ValueError(f"Invalid sealed-base candidate: {validation['errors']!r}.")
    source_declarations = build_chaser_distance_base_declarations(source)
    candidate_declarations = build_chaser_distance_base_declarations(candidate)
    source_inventory = [item.as_manifest() for item in source_declarations]
    candidate_inventory = [item.as_manifest() for item in candidate_declarations]
    if source_inventory != candidate_inventory:
        raise ValueError("Source and candidate logical declarations differ.")
    source_hashes = base_logical_hashes(source, source_declarations)
    candidate_hashes = base_logical_hashes(candidate, candidate_declarations)
    if source_hashes != candidate_hashes:
        raise ValueError("Source and candidate decoded values differ.")
    receipt_manifest = candidate.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(receipt_manifest, Mapping):
        raise ValueError("Candidate storage receipt is absent.")
    receipt = analysis_storage_plan_receipt_from_manifest(receipt_manifest)
    if receipt.as_manifest() != receipt_manifest:
        raise ValueError("Candidate storage receipt is not exactly executable.")
    candidate_manifest = candidate.attrs.get(BASE_MANIFEST_ATTR)
    if not isinstance(candidate_manifest, Mapping) or not _is_sha256(
        candidate_manifest.get("payload_digest")
    ):
        raise ValueError("Candidate manifest digest is absent.")
    return {
        "binding": binding,
        "source_declarations": source_declarations,
        "candidate_declarations": candidate_declarations,
        "logical_hashes": source_hashes,
        "receipt": receipt,
        "receipt_manifest": receipt_manifest,
        "candidate_manifest_payload_digest": candidate_manifest["payload_digest"],
    }


def _preflight(
    archive: Path,
    *,
    source_path: str,
    candidate_path: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    pair = _validate_pair(root, source_path=source_path, candidate_path=candidate_path)
    for path in (source_path, candidate_path):
        validate_direct_consolidated_subtree(archive, subtree_path=path)
    receipt = pair["receipt"]
    scale = AnalysisBenchmarkScale(
        scale_id="observed_candidate",
        dimensions=tuple(sorted((str(key), int(value)) for key, value in receipt.dimensions)),
        description="Observed sealed chaser-distance source/candidate dimensions.",
    )
    suite = build_analysis_benchmark_suite(
        family_id=FAMILY_ID,
        scale=scale,
        storage_receipt=receipt,
        seed=seed,
        repetitions=repetitions,
    )
    require_analysis_benchmark_suite_manifest(suite)
    return {
        "suite": suite,
        "source_binding_sha256": canonical_json_sha256(pair["binding"]),
        "logical_hashes": pair["logical_hashes"],
        "candidate_storage_receipt_payload_digest": pair["receipt_manifest"][
            "payload_digest"
        ],
        "candidate_manifest_payload_digest": pair[
            "candidate_manifest_payload_digest"
        ],
    }


def _growth_axes(suite: Mapping[str, Any]) -> dict[str, int | None]:
    return {
        str(item["path"]): (
            None
            if item["observed_facts"]["growth_axis"] is None
            else int(item["observed_facts"]["growth_axis"])
        )
        for item in suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]
    }


def _read_cases(suite: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = []
    for item in suite["payload"]["array_cases"]:
        workload = str(item["case"]["workload"]["workload_id"])
        if workload.endswith(".write_materialization.v1") or workload.endswith(
            ".full_scan_read.v1"
        ):
            continue
        if item["selection"].get("mode") not in {
            "whole_array",
            "bounded_row_windows",
        }:
            raise ValueError("Sealed-base receipt produced a non-eager/windowed workload.")
        result.append(item)
    if len(result) != len(SEALED_CHASER_DISTANCE_BASE_PATHS):
        raise ValueError("Benchmark suite does not have one primary read per sealed array.")
    return result


def _slice(array: Any, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    selection = [slice(None)] * int(array.ndim)
    selection[axis] = slice(start, stop)
    return tuple(selection)


def _primary_read(
    array: Any,
    selection: Mapping[str, Any],
    *,
    growth_axis: int | None,
) -> dict[str, Any]:
    mode = selection["mode"]
    axis = 0 if growth_axis is None else growth_axis
    digest = hashlib.sha256()
    decoded_bytes = 0
    operations = 0

    def consume(value: Any) -> None:
        nonlocal decoded_bytes, operations
        block = np.ascontiguousarray(value)
        digest.update(block.view(np.uint8))
        decoded_bytes += int(block.nbytes)
        operations += 1

    if mode == "whole_array":
        consume(array[...])
    elif mode == "bounded_row_windows":
        for start, stop in selection["ranges"]:
            consume(array[_slice(array, axis, int(start), int(stop))])
    else:  # protected by _read_cases
        raise ValueError(f"Unsupported sealed-base access mode {mode!r}.")
    return {
        "mode": mode,
        "execution_axis": axis,
        "operation_count": operations,
        "decoded_bytes": decoded_bytes,
        "selection_digest": digest.hexdigest(),
    }


def _full_scan(array: Any, *, growth_axis: int | None) -> dict[str, Any]:
    dtype = np.dtype(array.dtype)
    shape = tuple(int(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(str(dtype).encode("utf-8"))
    digest.update(json.dumps(list(shape), separators=(",", ":")).encode("ascii"))
    axis = 0 if growth_axis is None else growth_axis
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
    other = max(
        1,
        int(np.prod([extent for index, extent in enumerate(shape) if index != axis])),
    )
    block_extent = max(1, FULL_SCAN_TARGET_BLOCK_BYTES // (dtype.itemsize * other))
    decoded_bytes = 0
    block_count = 0
    for start in range(0, shape[axis], block_extent):
        block = np.ascontiguousarray(
            array[_slice(array, axis, start, min(start + block_extent, shape[axis]))]
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


def _selected_storage_stats(archive: Path, run_path: str) -> dict[str, Any]:
    totals = {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }
    run_root = archive.joinpath(*run_path.split("/"))
    by_array: dict[str, Any] = {}
    for path in SEALED_CHASER_DISTANCE_BASE_PATHS:
        stats = storage_stats(run_root.joinpath(*path.split("/")))
        by_array[path] = stats
        for field in totals:
            totals[field] += stats[field]
    return {
        "scope": "exact_30_array_directories; group metadata excluded",
        "totals": totals,
        "payload_object_count": totals["payload_file_count"],
        "by_array": by_array,
        "whole_run_tree": storage_stats(run_root),
    }


def _publication_timing(
    group: Any,
    *,
    role: str,
    archive: Path,
    run_path: str,
    source_path: str,
    source_binding_sha256: str,
    logical_hashes: Mapping[str, str],
) -> dict[str, Any]:
    empty = {
        "availability": "not_applicable_source" if role == "source" else "not_recorded",
        "publish_schema_id": None,
        "materialization_seconds": None,
        "copy_duration_seconds": None,
        "published_at_utc": None,
    }
    if role == "source":
        return empty
    receipt = group.attrs.get("cluster_output_staging")
    if not isinstance(receipt, Mapping):
        return empty
    if set(receipt) != _PUBLISH_RECEIPT_FIELDS:
        raise ValueError("Candidate publication receipt has an unexpected field set.")
    if (
        receipt.get("schema_id") != PUBLISH_SCHEMA_ID
        or receipt.get("publisher_contract")
        != {
            "schema_id": ATOMIC_RUN_PUBLISHER_SCHEMA_ID,
            "schema_version": ATOMIC_RUN_PUBLISHER_SCHEMA_VERSION,
        }
        or receipt.get("policy") != _PUBLISH_POLICY
        or receipt.get("serialization_policy") != SERIALIZATION_POLICY
        or receipt.get("rollback_policy") != _ROLLBACK_POLICY
        or receipt.get("profile_id") != "published_http_v1"
        or receipt.get("source_run_path") != source_path
        or receipt.get("source_zarr") != str(archive)
        or Path(str(receipt.get("target_run_path"))).resolve()
        != archive.joinpath(*run_path.split("/"))
        or receipt.get("publication_owner_attr") != ATOMIC_PUBLICATION_OWNER_ATTR
        or receipt.get("failed_public_child_policy")
        != "retain_owner_bound_selector_ineligible_tombstone"
        or receipt.get("hidden_temporary_policy")
        != "same_parent_hidden_sibling_then_os_replace"
        or receipt.get("local_direct_consolidated_array_count")
        != len(SEALED_CHASER_DISTANCE_BASE_PATHS)
        or not isinstance(receipt.get("parent_attrs_before"), Mapping)
        or not isinstance(receipt.get("parent_attrs_after"), Mapping)
        or receipt.get("parent_attrs_before") != receipt.get("parent_attrs_after")
    ):
        raise ValueError("Candidate publication receipt contract is unsupported.")
    source_binding = receipt.get("source_binding")
    if (
        not isinstance(source_binding, Mapping)
        or canonical_json_sha256(source_binding) != source_binding_sha256
    ):
        raise ValueError("Candidate publication receipt source binding differs.")
    if receipt.get("source_logical_hashes") != dict(sorted(logical_hashes.items())):
        raise ValueError("Candidate publication receipt logical hashes differ.")
    for field in (
        "publication_source_run_path",
        "publication_owner_uuid",
        "host",
    ):
        observed = receipt.get(field)
        if type(observed) is not str or not observed.strip():
            raise ValueError(f"Candidate publication receipt {field} is invalid.")
    if (
        group.attrs.get(ATOMIC_PUBLICATION_OWNER_ATTR)
        != receipt["publication_owner_uuid"]
    ):
        raise ValueError("Candidate publication receipt owner differs from the run.")
    if not Path(str(receipt["publication_source_run_path"])).is_absolute():
        raise ValueError("Candidate publication source path is not absolute.")
    if receipt.get("lsb_jobid") is not None and (
        type(receipt["lsb_jobid"]) is not str or not receipt["lsb_jobid"].strip()
    ):
        raise ValueError("Candidate publication LSF job identity is invalid.")
    materialization = receipt.get("materialization_seconds")
    copy_duration = receipt.get("copy_duration_seconds")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in (materialization, copy_duration)
    ):
        raise ValueError("Candidate publication timing is malformed.")
    published_at = receipt.get("published_at_utc")
    if type(published_at) is not str or not published_at.strip():
        raise ValueError("Candidate publication timestamp is malformed.")
    physical_copy = receipt.get("physical_copy")
    if not isinstance(physical_copy, Mapping) or set(physical_copy) != {
        "backend",
        "verification",
        "file_count",
        "physical_bytes",
        "inventory_sha256",
        "content_sha256",
    }:
        raise ValueError("Candidate physical-copy receipt is malformed.")
    if (
        physical_copy["backend"] not in {"python", "rsync"}
        or physical_copy["verification"]
        not in {"sha256_all_physical_files", "rsync_checksum_dry_run"}
        or type(physical_copy["file_count"]) is not int
        or physical_copy["file_count"] < 1
        or type(physical_copy["physical_bytes"]) is not int
        or physical_copy["physical_bytes"] < 1
        or not _is_sha256(physical_copy["inventory_sha256"])
        or not _is_sha256(physical_copy["content_sha256"])
    ):
        raise ValueError("Candidate physical-copy receipt is invalid.")
    expected_validation = {
        "valid": True,
        "errors": [],
        "array_count": len(SEALED_CHASER_DISTANCE_BASE_PATHS),
        "logical_hashes": dict(sorted(logical_hashes.items())),
    }
    for field in (
        "local_validation",
        "temporary_validation",
        "pre_pointer_validation",
        "final_validation",
    ):
        if receipt.get(field) != expected_validation:
            raise ValueError(f"Candidate publication receipt {field} did not pass exactly.")
    return {
        "availability": "recorded_in_cluster_output_staging",
        "publish_schema_id": PUBLISH_SCHEMA_ID,
        "materialization_seconds": float(materialization),
        "copy_duration_seconds": float(copy_duration),
        "published_at_utc": published_at,
    }


def _environment(archive: Path, cache_state: str) -> dict[str, Any]:
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
        "cache_state": cache_state,
        "archive_device": int(archive.stat().st_dev),
        "thread_environment": {
            name: os.environ.get(name) for name in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _metadata_guard(archive: Path, run_paths: Sequence[str]) -> dict[str, Any]:
    paths = [archive / "zarr.json"]
    for parent in (SOURCE_PARENT_PATH, CANDIDATE_PARENT_PATH):
        paths.append(archive.joinpath(*parent.split("/"), "zarr.json"))
    for run_path in run_paths:
        paths.extend(
            sorted(archive.joinpath(*run_path.split("/")).rglob("zarr.json"))
        )
    records = []
    for path in sorted(set(paths), key=str):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required nonsymlink metadata file is absent: {path}.")
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
        "guard_scope": "root_parents_and_complete_selected_run_zarr_json",
        "metadata_file_count": len(records),
        "metadata_tree_sha256": canonical_json_sha256(records),
        "files": records,
    }


def _require_metadata_guard(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "guard_scope",
        "metadata_file_count",
        "metadata_tree_sha256",
        "files",
    }:
        raise ValueError("Chaser-distance metadata guard has an unexpected field set.")
    if value["guard_scope"] != "root_parents_and_complete_selected_run_zarr_json":
        raise ValueError("Chaser-distance metadata guard scope is unsupported.")
    files = value["files"]
    if not isinstance(files, list):
        raise ValueError("Chaser-distance metadata guard inventory must be one list.")
    observed_paths: list[str] = []
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "mtime_ns",
            "sha256",
        }:
            raise ValueError("Chaser-distance metadata guard file record is malformed.")
        path = record["path"]
        if (
            type(path) is not str
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(component in {"", ".", ".."} for component in path.split("/"))
            or not path.endswith("zarr.json")
        ):
            raise ValueError("Chaser-distance metadata guard path is unsafe.")
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or type(record["mtime_ns"]) is not int
            or record["mtime_ns"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise ValueError("Chaser-distance metadata guard file facts are invalid.")
        observed_paths.append(path)
    if observed_paths != sorted(set(observed_paths)):
        raise ValueError("Chaser-distance metadata guard paths are not unique and sorted.")
    if (
        type(value["metadata_file_count"]) is not int
        or value["metadata_file_count"] != len(files)
        or not _is_sha256(value["metadata_tree_sha256"])
        or value["metadata_tree_sha256"] != canonical_json_sha256(files)
    ):
        raise ValueError("Chaser-distance metadata guard inventory digest mismatch.")


def run_single_trial(
    archive_path: Path | str,
    *,
    source_parent: str,
    candidate_parent: str,
    source_run: str,
    candidate_run: str,
    role: str,
    repetition_index: int,
    order_position: int,
    seed: int,
    cache_state: str,
    suite_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    archive = _safe_archive(archive_path)
    source_parent = _require_parent(
        source_parent, expected=SOURCE_PARENT_PATH, label="source_parent"
    )
    candidate_parent = _require_parent(
        candidate_parent, expected=CANDIDATE_PARENT_PATH, label="candidate_parent"
    )
    source_name = _safe_run_name(source_run, label="source_run")
    candidate_name = _safe_run_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("source_run and candidate_run must be different names.")
    if role not in {"source", "candidate"}:
        raise ValueError("role must be source or candidate.")
    if type(repetition_index) is not int or repetition_index < 0:
        raise ValueError("repetition_index must be nonnegative.")
    if order_position not in {0, 1}:
        raise ValueError("order_position must be 0 or 1.")
    if not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    require_analysis_benchmark_suite_manifest(suite_manifest, require_current=True)
    suite_payload = suite_manifest["payload"]
    if (
        suite_payload["family_id"] != FAMILY_ID
        or suite_payload["seed"] != seed
        or repetition_index >= suite_payload["repetitions"]
        or _trial_order(seed=seed, repetition_index=repetition_index)[order_position]
        != role
    ):
        raise ValueError("Trial differs from its deterministic suite/order binding.")
    source_path = f"{source_parent}/{source_name}"
    candidate_path = f"{candidate_parent}/{candidate_name}"
    run_path = source_path if role == "source" else candidate_path
    receipt_manifest = suite_payload["storage_plan_receipt"]
    receipt_digest = receipt_manifest["payload_digest"]
    growth_axes = _growth_axes(suite_manifest)
    paths = tuple(sorted(growth_axes))
    if paths != SEALED_CHASER_DISTANCE_BASE_PATHS:
        raise ValueError("Suite array paths differ from the exact sealed base.")
    trial_wall = time.perf_counter()
    trial_cpu = time.process_time()
    started = utc_now()
    initial_rss = peak_rss_bytes()

    (direct_root, direct_group), direct_open = _measure(
        lambda: (
            (root := zarr.open_group(str(archive), mode="r", use_consolidated=False)),
            _group(root, run_path),
        )
    )
    (consolidated_root, consolidated_group), consolidated_open = _measure(
        lambda: (
            (root := zarr.open_group(str(archive), mode="r", use_consolidated=True)),
            _group(root, run_path),
        )
    )

    def validate() -> dict[str, Any]:
        source = _group(consolidated_root, source_path)
        binding = _source_binding(consolidated_root, source_path, source)
        candidate_manifest_digest: str | None = None
        if role == "source":
            errors = validate_chaser_distance_base_semantics(direct_group)
            if errors:
                raise ValueError(f"Invalid sealed source: {errors!r}.")
            declarations = build_chaser_distance_base_declarations(direct_group)
        else:
            result = validate_base_candidate(
                direct_group,
                source_group=source,
                expected_source_binding=binding,
            )
            if result["valid"] is not True:
                raise ValueError(f"Invalid sealed-base candidate: {result['errors']!r}.")
            persisted = direct_group.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
            if persisted != receipt_manifest:
                raise ValueError("Candidate receipt differs from benchmark suite.")
            declarations = build_chaser_distance_base_declarations(direct_group)
            manifest = direct_group.attrs.get(BASE_MANIFEST_ATTR)
            if not isinstance(manifest, Mapping) or not _is_sha256(
                manifest.get("payload_digest")
            ):
                raise ValueError("Candidate manifest digest is absent.")
            candidate_manifest_digest = str(manifest["payload_digest"])
        if tuple(item.path for item in declarations) != paths:
            raise ValueError("Trial declarations differ from benchmark suite.")
        return {
            "valid": True,
            "role": role,
            "array_count": len(declarations),
            "source_binding_sha256": canonical_json_sha256(binding),
            "candidate_manifest_payload_digest": candidate_manifest_digest,
        }

    validation, validation_timing = _measure(validate)

    def compare_metadata() -> dict[str, Any]:
        receipt = validate_direct_consolidated_subtree(archive, subtree_path=run_path)
        if receipt.array_count < len(paths):
            raise ValueError("Persisted metadata omits sealed-base arrays.")
        return {
            "equivalent": True,
            "array_count": receipt.array_count,
            "group_count": receipt.group_count,
            "node_count": receipt.node_count,
            "subtree_declarations_digest": receipt.declarations_sha256,
        }

    metadata, comparison_timing = _measure(compare_metadata)
    primary: dict[str, Any] = {}
    primary_wall = 0.0
    primary_cpu = 0.0
    for case in _read_cases(suite_manifest):
        path = str(case["array_path"])
        result, timing = _measure(
            lambda path=path, case=case: _primary_read(
                _array(consolidated_group, path),
                case["selection"],
                growth_axis=growth_axes[path],
            )
        )
        primary[path] = {
            **result,
            "workload_id": case["case"]["workload"]["workload_id"],
            "selection": case["selection"],
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
                _array(consolidated_group, path), growth_axis=growth_axes[path]
            )
        )
        scans[path] = {**result, "timing": timing}
        scan_wall += timing["wall_seconds"]
        scan_cpu += timing["cpu_seconds"]
    logical = {
        path: {
            "dtype": scans[path]["dtype"],
            "shape": scans[path]["shape"],
            "logical_digest": scans[path]["logical_digest"],
        }
        for path in paths
    }
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "source_parent_path": source_parent,
        "candidate_parent_path": candidate_parent,
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "role": role,
        "run_name": source_name if role == "source" else candidate_name,
        "run_path": run_path,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": seed,
        "cache_state": cache_state,
        "suite_payload_digest": suite_manifest["payload_digest"],
        "candidate_storage_receipt_payload_digest": receipt_digest,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "environment": _environment(archive, cache_state),
        "validation": {**validation, "timing": validation_timing},
        "metadata": {
            **metadata,
            "direct_open": direct_open,
            "consolidated_open": consolidated_open,
            "comparison": comparison_timing,
        },
        "primary_access": {
            "arrays": primary,
            "total_wall_seconds": primary_wall,
            "total_cpu_seconds": primary_cpu,
        },
        "full_scan": {
            "arrays": scans,
            "total_wall_seconds": scan_wall,
            "total_cpu_seconds": scan_cpu,
            "total_decoded_bytes": sum(item["decoded_bytes"] for item in scans.values()),
        },
        "logical_arrays": logical,
        "storage": _selected_storage_stats(archive, run_path),
        "publication_timing": _publication_timing(
            direct_group,
            role=role,
            archive=archive,
            run_path=run_path,
            source_path=source_path,
            source_binding_sha256=validation["source_binding_sha256"],
            logical_hashes={
                path: record["logical_digest"] for path, record in logical.items()
            },
        ),
        "runtime": {
            "initial_peak_rss_bytes": initial_rss,
            "final_peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_is_process_high_water_mark": True,
            "total_wall_seconds": float(time.perf_counter() - trial_wall),
            "total_cpu_seconds": float(time.process_time() - trial_cpu),
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _TRIAL_PHYSICAL_AVAILABILITY,
        },
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, payload)
    require_trial_result(result)
    return result


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    return (
        ("source", "candidate")
        if (seed + repetition_index) % 2 == 0
        else ("candidate", "source")
    )


def _median(
    trials: Sequence[Mapping[str, Any]], role: str, path: Sequence[str]
) -> float:
    values = []
    for trial in trials:
        value: Any = trial["payload"]
        if value["role"] != role:
            continue
        for component in path:
            value = value[component]
        values.append(float(value))
    return float(statistics.median(values))


def _summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("source", "candidate"):
        example = next(item["payload"] for item in trials if item["payload"]["role"] == role)
        result[role] = {
            "median_direct_open_wall_seconds": _median(
                trials, role, ("metadata", "direct_open", "wall_seconds")
            ),
            "median_consolidated_open_wall_seconds": _median(
                trials, role, ("metadata", "consolidated_open", "wall_seconds")
            ),
            "median_validation_wall_seconds": _median(
                trials, role, ("validation", "timing", "wall_seconds")
            ),
            "median_primary_access_wall_seconds": _median(
                trials, role, ("primary_access", "total_wall_seconds")
            ),
            "median_full_scan_wall_seconds": _median(
                trials, role, ("full_scan", "total_wall_seconds")
            ),
            "median_total_wall_seconds": _median(
                trials, role, ("runtime", "total_wall_seconds")
            ),
            "median_total_cpu_seconds": _median(
                trials, role, ("runtime", "total_cpu_seconds")
            ),
            "median_peak_rss_bytes": _median(
                trials, role, ("runtime", "final_peak_rss_bytes")
            ),
            "selected_base_storage": example["storage"]["totals"],
            "payload_object_count": example["storage"]["payload_object_count"],
        }
    result["candidate_publication_timing"] = next(
        item["payload"]["publication_timing"]
        for item in trials
        if item["payload"]["role"] == "candidate"
    )
    return result


def run_benchmark_matrix(
    archive_path: Path | str,
    *,
    source_parent: str,
    candidate_parent: str,
    source_run: str,
    candidate_run: str,
    output_dir: Path | str,
    cache_state: str,
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    archive = _safe_archive(archive_path)
    source_parent = _require_parent(
        source_parent, expected=SOURCE_PARENT_PATH, label="source_parent"
    )
    candidate_parent = _require_parent(
        candidate_parent, expected=CANDIDATE_PARENT_PATH, label="candidate_parent"
    )
    source_name = _safe_run_name(source_run, label="source_run")
    candidate_name = _safe_run_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("source_run and candidate_run must be different names.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative exact integer.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive exact integer.")
    if not cache_state.strip():
        raise ValueError("cache_state must be explicitly declared.")
    output = _safe_output(output_dir, archive=archive)
    source_path = f"{source_parent}/{source_name}"
    candidate_path = f"{candidate_parent}/{candidate_name}"
    guard_before = _metadata_guard(archive, (source_path, candidate_path))
    preflight = _preflight(
        archive,
        source_path=source_path,
        candidate_path=candidate_path,
        seed=seed,
        repetitions=repetitions,
    )
    output.mkdir(parents=True, exist_ok=False)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    suite_path = output / "analysis_benchmark_suite.json"
    _write_json(suite_path, preflight["suite"])
    trials: list[Mapping[str, Any]] = []
    trial_order: list[dict[str, Any]] = []
    trial_files: list[str] = []
    started = utc_now()
    environment = os.environ.copy()
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for repetition_index in range(repetitions):
        order = _trial_order(seed=seed, repetition_index=repetition_index)
        trial_order.append({"repetition_index": repetition_index, "roles": list(order)})
        for position, role in enumerate(order):
            filename = f"rep_{repetition_index:02d}_pos_{position}_{role}.json"
            trial_path = trials_dir / filename
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_chaser_distance_base_candidate",
                "trial",
                str(archive),
                "--source-parent",
                source_parent,
                "--candidate-parent",
                candidate_parent,
                "--source-run",
                source_name,
                "--candidate-run",
                candidate_name,
                "--role",
                role,
                "--repetition-index",
                str(repetition_index),
                "--order-position",
                str(position),
                "--seed",
                str(seed),
                "--cache-state",
                cache_state,
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
                    "Fresh-process chaser-distance trial failed: "
                    f"command={command!r}, stdout={completed.stdout!r}, "
                    f"stderr={completed.stderr!r}."
                )
            trial = _read_json(trial_path)
            require_trial_result(trial)
            trials.append(trial)
            trial_files.append(str(trial_path.relative_to(output)))
    logical_equality = all(
        trial["payload"]["logical_arrays"] == trials[0]["payload"]["logical_arrays"]
        for trial in trials
    )
    primary_equality = all(
        {
            path: value["selection_digest"]
            for path, value in trial["payload"]["primary_access"]["arrays"].items()
        }
        == {
            path: value["selection_digest"]
            for path, value in trials[0]["payload"]["primary_access"]["arrays"].items()
        }
        for trial in trials
    )
    if not logical_equality or not primary_equality:
        raise ValueError("Source and candidate decoded benchmark values differ.")
    guard_after = _metadata_guard(archive, (source_path, candidate_path))
    unchanged = guard_before == guard_after
    if not unchanged:
        raise RuntimeError("Archive metadata changed during read-only benchmark execution.")
    correctness = {
        "full_decoded_logical_equality": True,
        "primary_access_decoded_equality": True,
        "direct_consolidated_metadata_equivalence": True,
        "source_and_candidate_validation": True,
        "archive_metadata_unchanged": True,
        "all_passed": True,
    }
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "source_parent_path": source_parent,
        "candidate_parent_path": candidate_parent,
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": cache_state,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "suite": preflight["suite"],
        "candidate_storage_receipt_payload_digest": preflight[
            "candidate_storage_receipt_payload_digest"
        ],
        "source_binding_sha256": preflight["source_binding_sha256"],
        "candidate_manifest_payload_digest": preflight[
            "candidate_manifest_payload_digest"
        ],
        "trial_order": trial_order,
        "trial_files": trial_files,
        "trials": trials,
        "correctness": correctness,
        "performance_summary": _summary(trials),
        "archive_read_only_metadata_guard": {
            "before": guard_before,
            "after": guard_after,
            "unchanged": True,
        },
        "physical_io": {
            "request_count": None,
            "transferred_bytes": None,
            "availability": _MATRIX_PHYSICAL_AVAILABILITY,
        },
        "promotion_decision": {
            "authorized": False,
            "reason": "benchmark_only; profile promotion requires a separate reviewed decision",
        },
    }
    result = _strict_envelope(MATRIX_SCHEMA_ID, payload)
    require_matrix_result(result)
    _write_json(output / "matrix_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    matrix = commands.add_parser("matrix")
    trial = commands.add_parser("trial")
    for child in (matrix, trial):
        child.add_argument("zarr_path", type=Path)
        child.add_argument("--source-parent", required=True)
        child.add_argument("--candidate-parent", required=True)
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
            source_parent=args.source_parent,
            candidate_parent=args.candidate_parent,
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
                        Path(args.output_dir).expanduser().resolve() / "matrix_result.json"
                    ),
                    "payload_digest": result["payload_digest"],
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    suite = _read_json(args.suite_file.expanduser().resolve())
    output = _safe_trial_output(
        args.output_file, root=args.benchmark_root.expanduser().resolve()
    )
    result = run_single_trial(
        args.zarr_path,
        source_parent=args.source_parent,
        candidate_parent=args.candidate_parent,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        seed=args.seed,
        cache_state=args.cache_state,
        suite_manifest=suite,
    )
    _write_json(output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "CANDIDATE_PARENT_PATH",
    "DEFAULT_REPETITIONS",
    "DEFAULT_SEED",
    "FAMILY_ID",
    "MATRIX_SCHEMA_ID",
    "SOURCE_PARENT_PATH",
    "TRIAL_SCHEMA_ID",
    "require_matrix_result",
    "require_trial_result",
    "run_benchmark_matrix",
    "run_single_trial",
]
