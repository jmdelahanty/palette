"""Read-only source/candidate matrix for bout-classification compact v2.

The maintained source is opened through Palette's public complete-and-eligible
reader.  The byte-planned candidate deliberately has no public consumer: this
diagnostic opens it by explicit immutable name only after the production
staged-run validator accepts it.  Nothing here mutates an archive, selector,
registry, storage profile, or production default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
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
from typing import Any
import uuid

import numpy as np
import zarr

from fisheye.analysis.bout_classification_runs import (
    BOUT_CLASSIFICATION_RUN_PARENT,
    load_bout_classification_table,
    resolve_bout_classification_run,
    validate_bout_classification_run,
    validate_staged_bout_classification_run,
)
from fisheye.analysis.bout_classification_schema import (
    BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS,
    BOUT_CLASSIFICATION_ARRAY_DECLARATIONS,
    BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR,
    BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR,
    BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS,
    BOUT_CLASSIFICATION_FIELD_DTYPES,
    BOUT_CLASSIFICATION_FIELD_NAMES,
    BOUT_CLASSIFICATION_FILL_VALUES,
    BOUT_CLASSIFICATION_RUN_SCHEMA_ID,
    BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION,
    CATEGORY_LABEL_BYTES_WIDTH,
    FAILURE_REASON_BYTES_WIDTH,
    BoutClassificationDimensions,
    bout_classification_array_schema_manifest,
    validate_bout_classification_arrays,
)
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    ANALYSIS_STORAGE_PROFILE_ID_ATTR,
    ANALYSIS_STORAGE_PROFILE_ROLE,
    ANALYSIS_STORAGE_PROFILE_ROLE_ATTR,
)
from fisheye.analysis.megabouts_classifier import (
    BOUT_CLASSIFICATION_PARENT_PUBLICATION_LEASE_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR,
    BOUT_CLASSIFICATION_PUBLICATION_POLICY,
    BOUT_CLASSIFICATION_PUBLICATION_POLICY_ATTR,
)
from fisheye.shared.system_metadata import get_git_info
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
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

FAMILY_ID = "bout_classification"
BENCHMARK_ID = "bout_classification_compact_v2_source_candidate_reads_v1"
PAIR_SCHEMA_ID = "palette.bout_classification.compact_v2_read_pair"
WORKLOAD_SCHEMA_ID = "palette.bout_classification.compact_v2_read_workload"
TRIAL_SCHEMA_ID = "palette.bout_classification.compact_v2_read_trial"
MATRIX_SCHEMA_ID = "palette.bout_classification.compact_v2_read_matrix"
SCHEMA_VERSION = 1
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 17
DEFAULT_WINDOW_ROWS = 4096
DEFAULT_WINDOWS_PER_ARRAY = 3
ARRAY_COUNT = 20
PROFILE_PROMOTED = False
SOURCE_CONSUMER = "public_bout_classification_reader_v2"
CANDIDATE_CONSUMER = "private_explicit_diagnostic_staged_reader_v1"
CANDIDATE_PUBLIC_CONSUMER_IMPLEMENTED = False
CACHE_STATE = "fresh_child_post_pair_validation_os_cache_uncontrolled"
PHYSICAL_IO_REASON = (
    "unavailable_without_os_or_filesystem_tracing; logical decoded bytes and "
    "filesystem object sizes are not physical transfer telemetry"
)
_ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
_PAIR_FIELDS = {
    "benchmark_id",
    "family_id",
    "archive_path",
    "source_run_name",
    "candidate_run_name",
    "source_run_path",
    "candidate_run_path",
    "schema_contract",
    "consumers",
    "lifecycle",
    "source_validation",
    "candidate_validation",
    "logical_arrays",
    "logical_equality",
    "scientific_identity",
    "structured_table_digest",
    "candidate_storage_receipt",
    "metadata_equivalence",
    "storage",
    "workload",
    "profile_promoted",
    "physical_io",
}
_TRIAL_FIELDS = {
    "benchmark_id",
    "family_id",
    "archive_path",
    "source_run_name",
    "candidate_run_name",
    "pair_payload_digest",
    "workload_payload_digest",
    "role",
    "run_name",
    "run_path",
    "consumer",
    "public_consumer_implemented",
    "repetition_index",
    "order_position",
    "seed",
    "controller_pid",
    "child_pid",
    "child_parent_pid",
    "fresh_child_process",
    "started_at_utc",
    "finished_at_utc",
    "environment",
    "validation",
    "consumer_read",
    "array_reads",
    "aggregate_read",
    "storage",
    "runtime",
    "profile_promoted",
    "selector_eligible",
    "physical_io",
}
_ENVIRONMENT_FIELDS = {
    "hostname",
    "system",
    "release",
    "python",
    "numpy",
    "zarr",
    "palette_commit",
    "palette_dirty",
    "cache_state",
    "thread_environment",
}
_REQUIRED_SOURCE_REF_PATHS = frozenset(
    {
        "tail_posture_view_run",
        "tail_frame_indices",
        "tail_instance_key",
        "tail_angle_rad",
        "tail_valid",
        "tail_posture_source_subject_shape_run",
        "track_kinematics_run",
        "track_group",
        "track_frame_indices",
        "track_source_instance_key",
        "positions_mm",
        "heading",
        "sample_valid",
        "swim_bout_run",
        "swim_bout_level",
        "bouts",
    }
)
_REQUIRED_SOURCE_REF_MANIFESTS = frozenset(
    {
        "tail_posture_publication_manifest_ref",
        "tail_posture_source_subject_shape_publication_manifest_ref",
    }
)
_REQUIRED_SOURCE_REF_DIGESTS = frozenset(
    {
        "tail_posture_publication_manifest_sha256",
        "tail_posture_source_subject_shape_publication_manifest_sha256",
        "positions_mm_coordinate_descriptor_sha256",
        "track_motion_manifest_sha256",
        "swim_bout_source_track_motion_manifest_sha256",
    }
)
_SCIENTIFIC_ATTRS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "adapter_method",
    "adapter_method_version",
    "classifier_family",
    "classifier_name",
    "classifier_version",
    "classifier_input_mode",
    "megabouts_preprocessing",
    "megabouts_segmentation",
    "source_mode",
    "row_axis",
    "invalid_window_policy",
    "source_fps",
    "window_duration_s",
    "window_frames",
    "megabouts_time_sampling",
    "source_bout_count",
    "valid_source_window_count",
    "invalid_source_window_count",
    "classified_bout_count",
    "source_refs",
    "parameters",
    "tail_angle_conversion",
    "trajectory_conversion",
    "invalid_frame_policy",
)


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
        raise ValueError("Bout benchmark envelope has an unexpected field set.")
    if value["schema_id"] != schema_id or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Bout benchmark envelope schema identity is unsupported.")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("Bout benchmark envelope payload must be one object.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Bout benchmark envelope payload digest mismatch.")
    json.dumps(value, allow_nan=False)
    return payload


def _safe_name(value: str, *, label: str) -> str:
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


def _assert_no_symlinks(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}.")
    for parent, directories, files in os.walk(path, followlinks=False):
        parent_path = Path(parent)
        for name in (*directories, *files):
            child = parent_path / name
            if child.is_symlink():
                raise ValueError(f"{label} contains a symlink: {child}.")


def _safe_archive(value: str | Path) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or not (path / "zarr.json").is_file():
        raise ValueError("Archive must be one existing nonsymlink Zarr directory.")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("Archive path must be canonical and contain no symlink alias.")
    return resolved


def _safe_run_tree(archive: Path, run_name: str) -> Path:
    path = archive.joinpath(*BOUT_CLASSIFICATION_RUN_PARENT.split("/"), run_name)
    if not path.is_dir():
        raise FileNotFoundError(f"Bout-classification run tree not found: {path}.")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_relative_to(archive):
        raise ValueError("Bout-classification run tree escapes the archive.")
    _assert_no_symlinks(path, label="Bout-classification run tree")
    return path


def _safe_new_output_dir(value: str | Path, *, archive: Path) -> Path:
    output = Path(value).expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}.")
    if output in {Path("/"), Path.home().resolve()}:
        raise ValueError("Benchmark output path is too broad.")
    ancestor = output.parent
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise ValueError("Benchmark output has no existing parent.")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve(strict=True) != ancestor.absolute():
        raise ValueError("Benchmark output parent must not use a symlink alias.")
    resolved_output = ancestor.resolve(strict=True).joinpath(
        *output.relative_to(ancestor).parts
    )
    if (
        resolved_output == archive
        or resolved_output.is_relative_to(archive)
        or archive.is_relative_to(resolved_output)
    ):
        raise ValueError("Benchmark output must be disjoint from the archive.")
    if not any("benchmark" in part.lower() for part in resolved_output.parts):
        raise ValueError("Benchmark output must be visibly benchmark-only.")
    return resolved_output


def _safe_trial_output(value: str | Path, *, benchmark_root: Path) -> Path:
    output = Path(value).expanduser().absolute()
    root = benchmark_root.resolve(strict=True)
    if output.exists() or output.suffix != ".json":
        raise ValueError("Trial output must be one new JSON file.")
    if output.parent.resolve(strict=True) != output.parent.absolute():
        raise ValueError("Trial output parent must be canonical and nonsymlinked.")
    if not output.is_relative_to(root):
        raise ValueError("Trial output must remain inside the benchmark root.")
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
            ValueError(f"Non-finite JSON token {raw}.")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"Strict JSON document is not one object: {path}.")
    return value


def _array_at(run_group: Any, path: str) -> Any:
    node = run_group
    for component in path.split("/"):
        node = node[component]
    return node


def _normalized_fill(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return {"palette_exact_float": "nan"}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_payload(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    header = {
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(array.view(np.uint8))
    return {
        **header,
        "decoded_bytes": int(array.nbytes),
        "payload_digest": digest.hexdigest(),
    }


def _structured_payload(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    header = {
        "dtype_descr": [list(item) for item in array.dtype.descr],
        "shape": [int(value) for value in array.shape],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(array.view(np.uint8))
    return {
        **header,
        "decoded_bytes": int(array.nbytes),
        "payload_digest": digest.hexdigest(),
    }


def _decode_text_matrix(values: np.ndarray, *, width: int, label: str) -> list[str]:
    array = np.asarray(values)
    if array.dtype != np.dtype("uint8") or array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} is not exact uint8[n_bouts,{width}].")
    decoded: list[str] = []
    for row_index, row in enumerate(array):
        zero = np.flatnonzero(row == 0)
        if zero.size == 0:
            raise ValueError(f"{label} row {row_index} lacks a NUL terminator.")
        end = int(zero[0])
        if np.any(row[end:] != 0):
            raise ValueError(f"{label} row {row_index} has nonzero bytes after NUL.")
        text = bytes(row[:end]).decode("utf-8")
        if not text:
            raise ValueError(f"{label} row {row_index} is empty.")
        decoded.append(text)
    return decoded


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _require_zarr_path(value: Any, *, label: str, prefix: str | None = None) -> str:
    if type(value) is not str or not value or value != value.strip().strip("/"):
        raise ValueError(f"{label} must be one canonical archive-relative path.")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or "@" in part for part in parts):
        raise ValueError(f"{label} contains an unsafe path component.")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"{label} does not use required prefix {prefix!r}.")
    return value


def _require_manifest_ref(
    value: Any, *, label: str, run_path: str, attribute: str
) -> str:
    expected = f"{run_path}@{attribute}"
    if value != expected:
        raise ValueError(f"{label} must exactly equal {expected!r}.")
    return str(value)


def _scientific_projection(run_group: Any) -> dict[str, Any]:
    missing = [name for name in _SCIENTIFIC_ATTRS if name not in run_group.attrs]
    if missing:
        raise ValueError(
            "Bout-classification scientific identity is missing attrs: "
            + ", ".join(missing)
        )
    refs = run_group.attrs["source_refs"]
    if not isinstance(refs, Mapping) or not all(
        type(key) is str and type(value) is str for key, value in refs.items()
    ):
        raise ValueError("Bout-classification source_refs must be string-to-string.")
    required = (
        _REQUIRED_SOURCE_REF_PATHS
        | _REQUIRED_SOURCE_REF_MANIFESTS
        | _REQUIRED_SOURCE_REF_DIGESTS
    )
    missing_refs = sorted(required - set(refs))
    if missing_refs:
        raise ValueError(
            "Bout-classification source_refs omit modern dependency identities: "
            + ", ".join(missing_refs)
        )
    for name in _REQUIRED_SOURCE_REF_PATHS:
        _require_zarr_path(refs[name], label=f"source_refs.{name}")
    for name in _REQUIRED_SOURCE_REF_DIGESTS:
        _require_sha256(refs[name], label=f"source_refs.{name}")

    posture_run = _require_zarr_path(
        refs["tail_posture_view_run"],
        label="source_refs.tail_posture_view_run",
        prefix="analysis/tail_posture_view_runs/",
    )
    subject_shape_run = _require_zarr_path(
        refs["tail_posture_source_subject_shape_run"],
        label="source_refs.tail_posture_source_subject_shape_run",
        prefix="analysis/subject_shape_runs/",
    )
    _require_manifest_ref(
        refs["tail_posture_publication_manifest_ref"],
        label="source_refs.tail_posture_publication_manifest_ref",
        run_path=posture_run,
        attribute="tail_coordinate_publication_manifest",
    )
    _require_manifest_ref(
        refs["tail_posture_source_subject_shape_publication_manifest_ref"],
        label=(
            "source_refs." "tail_posture_source_subject_shape_publication_manifest_ref"
        ),
        run_path=subject_shape_run,
        attribute="subject_shape_publication_manifest",
    )
    for name, leaf in (
        ("tail_frame_indices", "source_acquisition_frame_index"),
        ("tail_instance_key", "instance_key"),
        ("tail_angle_rad", "tail_angle_rad"),
        ("tail_valid", "valid"),
    ):
        if refs[name] != f"{posture_run}/{leaf}":
            raise ValueError(f"source_refs.{name} is not bound to tail-posture run.")

    track_run = _require_zarr_path(
        refs["track_kinematics_run"],
        label="source_refs.track_kinematics_run",
        prefix="analysis/track_kinematics_runs/",
    )
    track_group = _require_zarr_path(
        refs["track_group"], label="source_refs.track_group"
    )
    if not track_group.startswith(f"{track_run}/tracks/id_"):
        raise ValueError(
            "source_refs.track_group is not inside the selected track run."
        )
    for name, leaf in (
        ("track_frame_indices", "source_acquisition_frame_index"),
        ("track_source_instance_key", "source_instance_key"),
        ("positions_mm", "positions_mm"),
        ("sample_valid", "sample_valid"),
    ):
        if refs[name] != f"{track_group}/{leaf}":
            raise ValueError(f"source_refs.{name} is not bound to track group.")
    if not refs["heading"].startswith(f"{track_group}/"):
        raise ValueError("source_refs.heading is not bound to track group.")

    swim_run = _require_zarr_path(
        refs["swim_bout_run"],
        label="source_refs.swim_bout_run",
        prefix="analysis/swim_bout_runs/",
    )
    swim_level = _require_zarr_path(
        refs["swim_bout_level"], label="source_refs.swim_bout_level"
    )
    if not swim_level.startswith(f"{swim_run}/"):
        raise ValueError("source_refs.swim_bout_level is outside selected swim run.")
    if refs["bouts"] != f"{swim_level}/bouts":
        raise ValueError("source_refs.bouts is not the selected swim-level table.")
    if (
        refs["swim_bout_source_track_motion_manifest_sha256"]
        != refs["track_motion_manifest_sha256"]
    ):
        raise ValueError(
            "Swim-bout and track-motion manifest digests do not bind the same input."
        )
    if run_group["per_bout"].attrs.get("source_swim_bout_path") != swim_level:
        raise ValueError("per_bout source_swim_bout_path differs from source_refs.")

    parameters = run_group.attrs["parameters"]
    tail_conversion = run_group.attrs["tail_angle_conversion"]
    trajectory_conversion = run_group.attrs["trajectory_conversion"]
    invalid_policy = run_group.attrs["invalid_frame_policy"]
    for value, label in (
        (parameters, "parameters"),
        (tail_conversion, "tail_angle_conversion"),
        (trajectory_conversion, "trajectory_conversion"),
        (invalid_policy, "invalid_frame_policy"),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"Bout-classification {label} must be one object.")
    required_parameters = {
        "fps",
        "bout_duration_s",
        "classifier_input_mode",
        "megabouts_preprocessing",
        "megabouts_segmentation",
        "traj_alignment",
        "traj_reference_index",
        "min_tail_valid_fraction",
        "min_traj_valid_fraction",
        "max_consecutive_invalid_frames",
        "requires_traj_reference_valid",
        "adapter_method",
        "adapter_method_version",
        "classifier_family",
        "classifier_name",
        "source_fps",
        "window_duration_s",
        "window_frames",
        "megabouts_time_sampling",
        "source_mode",
        "invalid_window_policy",
        "classified_bout_count",
        "source_bout_count",
        "valid_source_window_count",
        "invalid_source_window_count",
    }
    if not required_parameters.issubset(parameters):
        raise ValueError("Bout-classification parameters omit the modern contract.")
    attr_parameter_pairs = (
        ("classifier_input_mode", "classifier_input_mode"),
        ("megabouts_preprocessing", "megabouts_preprocessing"),
        ("megabouts_segmentation", "megabouts_segmentation"),
        ("adapter_method", "adapter_method"),
        ("adapter_method_version", "adapter_method_version"),
        ("classifier_family", "classifier_family"),
        ("classifier_name", "classifier_name"),
        ("source_fps", "source_fps"),
        ("window_duration_s", "window_duration_s"),
        ("window_frames", "window_frames"),
        ("megabouts_time_sampling", "megabouts_time_sampling"),
        ("source_mode", "source_mode"),
        ("invalid_window_policy", "invalid_window_policy"),
        ("classified_bout_count", "classified_bout_count"),
        ("source_bout_count", "source_bout_count"),
        ("valid_source_window_count", "valid_source_window_count"),
        ("invalid_source_window_count", "invalid_source_window_count"),
    )
    for attr_name, parameter_name in attr_parameter_pairs:
        if run_group.attrs[attr_name] != parameters[parameter_name]:
            raise ValueError(
                f"Bout-classification {attr_name} differs from parameters."
            )
    if (
        parameters["fps"] != run_group.attrs["source_fps"]
        or parameters["bout_duration_s"] != run_group.attrs["window_duration_s"]
    ):
        raise ValueError("Input-pack sampling parameters differ from run sampling.")
    if (
        tail_conversion.get("source_array") != refs["tail_angle_rad"]
        or tail_conversion.get("source_valid_array") != refs["tail_valid"]
        or tail_conversion.get("units") != "radians"
    ):
        raise ValueError("Tail-angle conversion is not bound to source_refs.")
    if (
        trajectory_conversion.get("source_positions_array") != refs["positions_mm"]
        or trajectory_conversion.get("source_heading_array") != refs["heading"]
        or trajectory_conversion.get("source_valid_array") != refs["sample_valid"]
    ):
        raise ValueError("Trajectory conversion is not bound to source_refs.")
    if (
        trajectory_conversion.get("alignment") != parameters["traj_alignment"]
        or trajectory_conversion.get("reference_index")
        != parameters["traj_reference_index"]
    ):
        raise ValueError("Trajectory conversion differs from parameters.")
    expected_invalid_policy = {
        "policy": run_group.attrs["invalid_window_policy"],
        "min_tail_valid_fraction": parameters["min_tail_valid_fraction"],
        "min_traj_valid_fraction": parameters["min_traj_valid_fraction"],
        "max_consecutive_invalid_frames": parameters["max_consecutive_invalid_frames"],
        "requires_traj_reference_valid": parameters["requires_traj_reference_valid"],
    }
    if dict(invalid_policy) != expected_invalid_policy:
        raise ValueError("Invalid-frame policy differs from parameters.")
    projection = {name: run_group.attrs[name] for name in _SCIENTIFIC_ATTRS}
    projection["per_bout_source_swim_bout_path"] = swim_level
    return _strict_envelope(
        "palette.bout_classification.scientific_dependency_identity",
        projection,
    )


def _semantic_validation(run_group: Any) -> dict[str, Any]:
    per_bout = run_group["per_bout"]
    arrays = {
        name: np.asarray(per_bout[name][:]) for name in BOUT_CLASSIFICATION_FIELD_NAMES
    }
    n_bouts = int(arrays["source_bout_id"].shape[0])
    classified = arrays["classified"].astype(bool, copy=False)
    valid = arrays["valid"].astype(bool, copy=False)
    source_valid = arrays["source_window_valid"].astype(bool, copy=False)
    skipped = ~classified
    labels = _decode_text_matrix(
        arrays["category_label_bytes"],
        width=CATEGORY_LABEL_BYTES_WIDTH,
        label="category_label_bytes",
    )
    reasons = _decode_text_matrix(
        arrays["failure_reason_bytes"],
        width=FAILURE_REASON_BYTES_WIDTH,
        label="failure_reason_bytes",
    )
    if len(np.unique(arrays["source_bout_id"])) != n_bouts:
        raise ValueError("source_bout_id must be unique within the run.")
    if np.any(arrays["source_bout_id"] < 0):
        raise ValueError("source_bout_id must be nonnegative.")
    for start_name, end_name in (
        ("start_frame", "end_frame"),
        ("window_start_frame", "window_end_frame"),
    ):
        if np.any(arrays[start_name] < 0) or np.any(
            arrays[end_name] < arrays[start_name]
        ):
            raise ValueError(f"{start_name}/{end_name} intervals are invalid.")
    if np.any(classified & ~source_valid) or np.any(valid & ~classified):
        raise ValueError("Classification validity bitmaps are inconsistent.")
    if np.any(skipped & (arrays["HB1_frame"] != -1)):
        raise ValueError("Unclassified HB1_frame must be -1.")
    if np.any(skipped & (arrays["HB1_offset_frames"] != -1)):
        raise ValueError("Unclassified HB1_offset_frames must be -1.")
    if np.any(skipped & (arrays["category_id"] != -1)):
        raise ValueError("Unclassified category_id must be -1.")
    if np.any(skipped & (arrays["subcategory_id"] != -1)):
        raise ValueError("Unclassified subcategory_id must be -1.")
    if np.any(skipped & (arrays["sign"] != 0)):
        raise ValueError("Unclassified sign must be zero.")
    if np.any(skipped & ~np.isnan(arrays["probability"])):
        raise ValueError("Unclassified probability must be NaN.")
    if np.any(classified & (arrays["category_id"] < 0)):
        raise ValueError("Classified category_id must be nonnegative.")
    if np.any(classified & ~np.isfinite(arrays["probability"])):
        raise ValueError("Classified probability must be finite.")
    if np.any(classified & (arrays["HB1_offset_frames"] < 0)):
        raise ValueError("Classified HB1_offset_frames must be nonnegative.")
    expected_hb1 = arrays["window_start_frame"] + arrays["HB1_offset_frames"].astype(
        np.int64
    )
    if np.any(classified & (arrays["HB1_frame"] != expected_hb1)):
        raise ValueError(
            "Classified HB1_frame must equal window_start_frame plus offset."
        )
    if np.any(
        classified
        & (
            (arrays["HB1_frame"] < arrays["window_start_frame"])
            | (arrays["HB1_frame"] > arrays["window_end_frame"])
        )
    ):
        raise ValueError("Classified HB1_frame must lie inside its inclusive window.")
    for name in ("tail_valid_fraction", "traj_valid_fraction"):
        values = arrays[name]
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name} must be finite in [0,1].")
    for name in (
        "max_consecutive_tail_invalid",
        "max_consecutive_traj_invalid",
    ):
        if np.any(arrays[name] < 0):
            raise ValueError(f"{name} must be nonnegative.")
    for index in np.flatnonzero(skipped):
        if labels[int(index)] != "skipped_invalid_window":
            raise ValueError(
                "Unclassified category labels must use the frozen sentinel."
            )
    for index in np.flatnonzero(classified):
        if reasons[int(index)] != "ok":
            raise ValueError("Classified rows must carry failure_reason 'ok'.")
    return {
        "valid": True,
        "n_bouts": n_bouts,
        "classified_count": int(np.count_nonzero(classified)),
        "valid_count": int(np.count_nonzero(valid)),
        "source_window_valid_count": int(np.count_nonzero(source_valid)),
        "category_label_width": CATEGORY_LABEL_BYTES_WIDTH,
        "failure_reason_width": FAILURE_REASON_BYTES_WIDTH,
        "semantic_fill_policy": {
            declaration.path: {
                "fill_semantics": declaration.fill_semantics,
                "null_semantics": declaration.null_semantics,
                "candidate_metadata_fill": _normalized_fill(
                    BOUT_CLASSIFICATION_FILL_VALUES[declaration.path]
                ),
            }
            for declaration in BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS
        },
    }


def _exact_inventory(run_group: Any, *, candidate: bool) -> dict[str, Any]:
    declarations = (
        BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS
        if candidate
        else BOUT_CLASSIFICATION_ARRAY_DECLARATIONS
    )
    dimensions = BoutClassificationDimensions(
        n_bouts=int(run_group.attrs["source_bout_count"])
    )
    issues = validate_bout_classification_arrays(run_group, dimensions=dimensions)
    if issues:
        raise ValueError(
            "Bout-classification exact array validation failed: "
            + "; ".join(
                f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
            )
        )
    if tuple(run_group.group_keys()) != ("per_bout",) or tuple(run_group.array_keys()):
        raise ValueError("Bout-classification run child inventory is not exact.")
    per_bout = run_group["per_bout"]
    if tuple(sorted(per_bout.array_keys())) != tuple(
        sorted(BOUT_CLASSIFICATION_FIELD_NAMES)
    ):
        raise ValueError("Bout-classification per_bout array inventory is not exact.")
    if tuple(per_bout.group_keys()):
        raise ValueError("Bout-classification per_bout contains unexpected groups.")
    if list(per_bout.attrs.get("field_names", [])) != list(
        BOUT_CLASSIFICATION_FIELD_NAMES
    ):
        raise ValueError("Bout-classification field order differs from compact v2.")
    if dict(per_bout.attrs.get("field_dtypes", {})) != BOUT_CLASSIFICATION_FIELD_DTYPES:
        raise ValueError(
            "Bout-classification logical field dtypes differ from compact v2."
        )
    manifest = run_group.attrs.get(BOUT_CLASSIFICATION_ARRAY_SCHEMA_ATTR)
    expected_manifest = bout_classification_array_schema_manifest(
        dimensions,
        byte_planner_adopted=candidate,
    )
    if manifest != expected_manifest:
        raise ValueError(
            "Bout-classification manifest differs from executable compact v2."
        )
    if run_group.attrs.get(BOUT_CLASSIFICATION_ARRAY_SCHEMA_DIGEST_ATTR) != (
        canonical_json_sha256(expected_manifest)
    ):
        raise ValueError("Bout-classification manifest digest is not executable.")
    arrays: dict[str, Any] = {}
    for declaration in declarations:
        array = _array_at(run_group, declaration.path)
        arrays[declaration.path] = {
            "dtype": np.dtype(array.dtype).str,
            "shape": [int(value) for value in array.shape],
            "access_pattern": declaration.access_pattern.value,
            "write_mode": declaration.write_mode.value,
            "authority_role": declaration.authority_role.value,
            "access_unit_semantics": BOUT_CLASSIFICATION_ACCESS_UNIT_SEMANTICS[
                declaration.path
            ],
            "payload": _array_payload(np.asarray(array[:])),
            "candidate_metadata_fill": (
                _normalized_fill(array.metadata.fill_value) if candidate else None
            ),
        }
        if candidate and arrays[declaration.path]["candidate_metadata_fill"] != (
            _normalized_fill(BOUT_CLASSIFICATION_FILL_VALUES[declaration.path])
        ):
            raise ValueError(
                f"Candidate metadata fill differs at {declaration.path!r}."
            )
    if len(arrays) != ARRAY_COUNT:
        raise ValueError("Bout-classification compact v2 must contain 20 arrays.")
    return {
        "array_count": len(arrays),
        "array_schema_manifest": manifest,
        "array_schema_payload_digest": canonical_json_sha256(manifest),
        "arrays": arrays,
        "semantic_validation": _semantic_validation(run_group),
    }


def _selector_lifecycle(
    parent: Any, source: Any, candidate: Any, *, source_name: str, candidate_name: str
) -> dict[str, Any]:
    if (
        parent.attrs.get("latest") != source_name
        or parent.attrs.get("latest_complete") != source_name
    ):
        raise ValueError("Source is not the exact selected complete authority.")
    if (
        parent.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_POLICY_ATTR)
        != BOUT_CLASSIFICATION_PUBLICATION_POLICY
    ):
        raise ValueError(
            "Bout-classification publication policy differs from guarded v1."
        )
    generation = parent.attrs.get(BOUT_CLASSIFICATION_PUBLICATION_GENERATION_ATTR)
    if type(generation) is not int or generation < 1:
        raise ValueError("Bout-classification publication generation is invalid.")
    lease = parent.attrs.get(BOUT_CLASSIFICATION_PARENT_PUBLICATION_LEASE_ATTR)
    lease_fields = {
        "schema_id",
        "schema_version",
        "policy",
        "owner_uuid",
        "publication_owner",
        "run_path",
        "run_name",
        "base_generation",
        "next_generation",
        "selector_attrs",
    }
    if not isinstance(lease, Mapping) or set(lease) != lease_fields:
        raise ValueError("Guarded publication lease has an unexpected field set.")
    source_owner = str(
        uuid.UUID(source.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR])
    )
    candidate_owner = str(
        uuid.UUID(candidate.attrs[BOUT_CLASSIFICATION_PUBLICATION_OWNER_ATTR])
    )
    if source_owner == candidate_owner:
        raise ValueError("Source and candidate publication owners must differ.")
    if lease != {
        "schema_id": "palette.bout_classification_publication_lease",
        "schema_version": 1,
        "policy": BOUT_CLASSIFICATION_PUBLICATION_POLICY,
        "owner_uuid": source_owner,
        "publication_owner": source_owner,
        "run_path": f"{BOUT_CLASSIFICATION_RUN_PARENT}/{source_name}",
        "run_name": source_name,
        "base_generation": generation - 1,
        "next_generation": generation,
        "selector_attrs": ["latest_complete", "latest"],
    }:
        raise ValueError(
            "Guarded publication lease differs from the live selector authority."
        )
    if (
        source.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or source.attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError("Source must be complete and selector eligible.")
    if (
        candidate.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or candidate.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Candidate must be complete and selector ineligible.")
    if (
        candidate.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
        != ANALYSIS_STORAGE_PROFILE_ROLE
    ):
        raise ValueError("Candidate does not carry the explicit unpromoted role.")
    return {
        "publication_mode": "guarded_direct_owner_generation_v1",
        "publication_policy": BOUT_CLASSIFICATION_PUBLICATION_POLICY,
        "publication_generation": generation,
        "publication_lease": dict(lease),
        "selectors": {"latest": source_name, "latest_complete": source_name},
        "source_owner_uuid": source_owner,
        "candidate_owner_uuid": candidate_owner,
        "source_selector_eligible": True,
        "candidate_selector_eligible": False,
        "candidate_profile_role": ANALYSIS_STORAGE_PROFILE_ROLE,
        "candidate_profile_id": candidate.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR),
        "profile_promoted": False,
    }


def _windows(
    row_count: int, *, window_rows: int, count: int, seed: int, path: str
) -> list[list[int]]:
    if row_count == 0:
        return [[0, 0]]
    width = min(row_count, window_rows)
    anchors = [0, max(0, (row_count - width) // 2), max(0, row_count - width)]
    if count > 3 and row_count > width:
        path_seed = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + path_seed)
        anchors.extend(
            int(value)
            for value in rng.integers(0, row_count - width + 1, size=count - 3)
        )
    spans: list[list[int]] = []
    for start in anchors[:count]:
        span = [int(start), int(start + width)]
        if span not in spans:
            spans.append(span)
    return spans


def _require_windows_per_array(value: Any) -> int:
    if type(value) is not int or value < 3:
        raise ValueError(
            "windows_per_array must be one exact integer greater than or equal to 3."
        )
    return value


def _parse_windows_per_array(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "windows per array must be an integer greater than or equal to 3"
        ) from error
    try:
        return _require_windows_per_array(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _build_workload(
    *,
    source_name: str,
    candidate_name: str,
    source_inventory: Mapping[str, Any],
    candidate_inventory: Mapping[str, Any],
    storage_receipt_digest: str,
    seed: int,
    window_rows: int,
    windows_per_array: int,
) -> dict[str, Any]:
    arrays: list[dict[str, Any]] = []
    for declaration in BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS:
        path = declaration.path
        source = source_inventory["arrays"][path]
        candidate = candidate_inventory["arrays"][path]
        if (
            source["dtype"] != candidate["dtype"]
            or source["shape"] != candidate["shape"]
        ):
            raise ValueError(f"Source/candidate logical type differs at {path!r}.")
        row_count = int(source["shape"][0])
        spans = _windows(
            row_count,
            window_rows=window_rows,
            count=windows_per_array,
            seed=seed,
            path=path,
        )
        arrays.append(
            {
                "path": path,
                "dtype": source["dtype"],
                "shape": source["shape"],
                "declared_access_pattern": declaration.access_pattern.value,
                "eager_read_spans": [[0, row_count]],
                "eager_operation_count": 1,
                "eager_decoded_bytes": int(np.prod(source["shape"], dtype=np.int64))
                * np.dtype(source["dtype"]).itemsize,
                "windowed_read_spans": spans,
                "windowed_operation_count": len(spans),
                "windowed_decoded_bytes": sum(
                    (stop - start)
                    * int(np.prod(source["shape"][1:], dtype=np.int64))
                    * np.dtype(source["dtype"]).itemsize
                    for start, stop in spans
                ),
            }
        )
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "seed": seed,
        "window_rows": window_rows,
        "windows_per_array": windows_per_array,
        "array_count": ARRAY_COUNT,
        "source_array_schema_payload_digest": source_inventory[
            "array_schema_payload_digest"
        ],
        "candidate_array_schema_payload_digest": candidate_inventory[
            "array_schema_payload_digest"
        ],
        "candidate_storage_receipt_payload_digest": storage_receipt_digest,
        "arrays": arrays,
    }
    return _strict_envelope(WORKLOAD_SCHEMA_ID, payload)


def _metadata_receipt(archive: Path, run_path: str) -> dict[str, Any]:
    receipt = validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    if receipt.array_count != ARRAY_COUNT or receipt.group_count != 2:
        raise ValueError("Bout-classification metadata receipt has wrong node counts.")
    return receipt.to_json()


def _storage_receipt(candidate: Any) -> dict[str, Any]:
    value = candidate.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if not isinstance(value, Mapping):
        raise ValueError("Candidate storage-plan receipt is absent.")
    parsed = analysis_storage_plan_receipt_from_manifest(value)
    expected_profile = get_storage_profile("published_http_v1")
    if candidate.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR) != "published_http_v1":
        raise ValueError("Candidate root profile must be exactly published_http_v1.")
    if (
        candidate.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
        != ANALYSIS_STORAGE_PROFILE_ROLE
    ):
        raise ValueError("Candidate root profile role is not explicitly unpromoted.")
    if parsed.profile != expected_profile:
        raise ValueError("Candidate parsed profile is not published_http_v1.")
    if parsed.as_manifest() != dict(value):
        raise ValueError(
            "Candidate parsed storage receipt does not round-trip exactly."
        )
    if len(parsed.entries) != ARRAY_COUNT:
        raise ValueError("Candidate storage-plan receipt must contain 20 arrays.")
    expected_declarations = {
        declaration.path: declaration.as_manifest()
        for declaration in BOUT_CLASSIFICATION_CANDIDATE_ARRAY_DECLARATIONS
    }
    observed_declarations = {
        entry.declaration.path: entry.declaration.as_manifest()
        for entry in parsed.entries
    }
    if observed_declarations != expected_declarations:
        raise ValueError(
            "Candidate storage receipt does not exactly cover candidate declarations."
        )
    if candidate.attrs.get(ANALYSIS_STORAGE_PLAN_DIGEST_ATTR) != value.get(
        "payload_digest"
    ):
        raise ValueError("Candidate redundant storage-plan digest differs.")
    return dict(value)


def build_pair_validation(
    archive_path: str | Path,
    *,
    source_run: str,
    candidate_run: str,
    seed: int = DEFAULT_SEED,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    windows_per_array: int = DEFAULT_WINDOWS_PER_ARRAY,
) -> dict[str, Any]:
    """Build one deterministic pair receipt from the live immutable archive."""

    archive = _safe_archive(archive_path)
    source_name = _safe_name(source_run, label="source run")
    candidate_name = _safe_name(candidate_run, label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate run names must differ.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be one nonnegative exact integer.")
    if type(window_rows) is not int or window_rows < 1:
        raise ValueError("window_rows must be one positive exact integer.")
    _require_windows_per_array(windows_per_array)
    source_tree = _safe_run_tree(archive, source_name)
    candidate_tree = _safe_run_tree(archive, candidate_name)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent = root[BOUT_CLASSIFICATION_RUN_PARENT]

    source, resolved_name, source_path = resolve_bout_classification_run(
        root, source_name
    )
    if resolved_name != source_name:
        raise ValueError("Public source reader resolved a different run.")
    source_validation = validate_bout_classification_run(root, source_name, strict=True)
    if source_validation.get("ok") is not True:
        raise ValueError(f"Public source validation failed: {source_validation!r}.")
    source_table = load_bout_classification_table(source)

    candidate_path = f"{BOUT_CLASSIFICATION_RUN_PARENT}/{candidate_name}"
    candidate_validation = validate_staged_bout_classification_run(
        root, candidate_name, strict=True
    )
    if candidate_validation.get("ok") is not True:
        raise ValueError(
            f"Staged candidate validation failed: {candidate_validation!r}."
        )
    candidate = root[candidate_path]
    candidate_table = load_bout_classification_table(candidate)
    source_table_payload = _structured_payload(source_table)
    candidate_table_payload = _structured_payload(candidate_table)
    if source_table_payload != candidate_table_payload:
        raise ValueError("Public source and private candidate tables differ logically.")

    lifecycle = _selector_lifecycle(
        parent,
        source,
        candidate,
        source_name=source_name,
        candidate_name=candidate_name,
    )
    source_scientific = _scientific_projection(source)
    candidate_scientific = _scientific_projection(candidate)
    if source_scientific != candidate_scientific:
        raise ValueError(
            "Source/candidate scientific and dependency identities differ."
        )
    source_inventory = _exact_inventory(source, candidate=False)
    candidate_inventory = _exact_inventory(candidate, candidate=True)
    logical_equality = all(
        source_inventory["arrays"][path]["payload"]
        == candidate_inventory["arrays"][path]["payload"]
        for path in source_inventory["arrays"]
    )
    if not logical_equality:
        raise ValueError("Source/candidate complete decoded arrays differ.")
    receipt = _storage_receipt(candidate)
    workload = _build_workload(
        source_name=source_name,
        candidate_name=candidate_name,
        source_inventory=source_inventory,
        candidate_inventory=candidate_inventory,
        storage_receipt_digest=str(receipt["payload_digest"]),
        seed=seed,
        window_rows=window_rows,
        windows_per_array=windows_per_array,
    )
    source_metadata = _metadata_receipt(archive, source_path)
    candidate_metadata = _metadata_receipt(archive, candidate_path)
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "source_run_name": source_name,
        "candidate_run_name": candidate_name,
        "source_run_path": source_path,
        "candidate_run_path": candidate_path,
        "schema_contract": {
            "run_schema_id": BOUT_CLASSIFICATION_RUN_SCHEMA_ID,
            "run_schema_version": BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION,
            "array_count": ARRAY_COUNT,
            "ordered_field_names": list(BOUT_CLASSIFICATION_FIELD_NAMES),
            "logical_field_dtypes": dict(BOUT_CLASSIFICATION_FIELD_DTYPES),
            "fixed_text_widths": {
                "category_label_bytes": CATEGORY_LABEL_BYTES_WIDTH,
                "failure_reason_bytes": FAILURE_REASON_BYTES_WIDTH,
            },
        },
        "consumers": {
            "source": SOURCE_CONSUMER,
            "source_public_consumer_implemented": True,
            "candidate": CANDIDATE_CONSUMER,
            "candidate_public_consumer_implemented": False,
            "candidate_diagnostic_consumer_implemented": True,
        },
        "lifecycle": lifecycle,
        "source_validation": {
            "public_validation_ok": True,
            "selector_eligible": True,
            "inventory": source_inventory,
        },
        "candidate_validation": {
            "staged_validation_ok": True,
            "selector_eligible": False,
            "inventory": candidate_inventory,
        },
        "logical_arrays": {
            path: source_inventory["arrays"][path]["payload"]
            for path in source_inventory["arrays"]
        },
        "logical_equality": True,
        "scientific_identity": {
            "source": source_scientific,
            "candidate": candidate_scientific,
            "equal": True,
            "shared_payload_digest": source_scientific["payload_digest"],
        },
        "structured_table_digest": source_table_payload,
        "candidate_storage_receipt": receipt,
        "metadata_equivalence": {
            "source": source_metadata,
            "candidate": candidate_metadata,
        },
        "storage": {
            "source": {
                "run_path": source_path,
                "filesystem_path": str(source_tree),
                **storage_stats(source_tree),
            },
            "candidate": {
                "run_path": candidate_path,
                "filesystem_path": str(candidate_tree),
                **storage_stats(candidate_tree),
            },
        },
        "workload": workload,
        "profile_promoted": False,
        "physical_io": {
            "file_reads": None,
            "range_reads": None,
            "transferred_bytes": None,
            "availability": PHYSICAL_IO_REASON,
        },
    }
    result = _strict_envelope(PAIR_SCHEMA_ID, payload)
    require_pair_validation(result, replay_archive=False)
    return result


def require_pair_validation(
    value: Mapping[str, Any], *, replay_archive: bool = True
) -> None:
    payload = _require_envelope(value, schema_id=PAIR_SCHEMA_ID)
    if set(payload) != _PAIR_FIELDS:
        raise ValueError("Bout pair payload has an unexpected field set.")
    if payload["benchmark_id"] != BENCHMARK_ID or payload["family_id"] != FAMILY_ID:
        raise ValueError("Bout pair benchmark identity mismatch.")
    source_name = _safe_name(payload["source_run_name"], label="source run")
    candidate_name = _safe_name(payload["candidate_run_name"], label="candidate run")
    if source_name == candidate_name:
        raise ValueError("Bout pair source and candidate must differ.")
    if (
        payload["source_run_path"] != f"{BOUT_CLASSIFICATION_RUN_PARENT}/{source_name}"
        or payload["candidate_run_path"]
        != f"{BOUT_CLASSIFICATION_RUN_PARENT}/{candidate_name}"
    ):
        raise ValueError("Bout pair run-path binding mismatch.")
    if payload["profile_promoted"] is not False:
        raise ValueError("Bout benchmark must not claim profile promotion.")
    if payload["schema_contract"] != {
        "run_schema_id": BOUT_CLASSIFICATION_RUN_SCHEMA_ID,
        "run_schema_version": BOUT_CLASSIFICATION_RUN_SCHEMA_VERSION,
        "array_count": ARRAY_COUNT,
        "ordered_field_names": list(BOUT_CLASSIFICATION_FIELD_NAMES),
        "logical_field_dtypes": dict(BOUT_CLASSIFICATION_FIELD_DTYPES),
        "fixed_text_widths": {
            "category_label_bytes": CATEGORY_LABEL_BYTES_WIDTH,
            "failure_reason_bytes": FAILURE_REASON_BYTES_WIDTH,
        },
    }:
        raise ValueError("Bout pair schema contract differs from compact v2.")
    consumers = payload["consumers"]
    if consumers != {
        "source": SOURCE_CONSUMER,
        "source_public_consumer_implemented": True,
        "candidate": CANDIDATE_CONSUMER,
        "candidate_public_consumer_implemented": False,
        "candidate_diagnostic_consumer_implemented": True,
    }:
        raise ValueError("Bout benchmark consumer boundary is false or incomplete.")
    lifecycle = payload["lifecycle"]
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("candidate_selector_eligible") is not False
        or lifecycle.get("profile_promoted") is not False
    ):
        raise ValueError(
            "Bout candidate lifecycle must remain ineligible and unpromoted."
        )
    if (
        payload["logical_equality"] is not True
        or len(payload["logical_arrays"]) != ARRAY_COUNT
    ):
        raise ValueError("Bout pair lacks complete decoded equality.")
    identity = payload["scientific_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {
        "source",
        "candidate",
        "equal",
        "shared_payload_digest",
    }:
        raise ValueError("Bout scientific identity receipt has unexpected fields.")
    source_identity = _require_envelope(
        identity["source"],
        schema_id="palette.bout_classification.scientific_dependency_identity",
    )
    candidate_identity = _require_envelope(
        identity["candidate"],
        schema_id="palette.bout_classification.scientific_dependency_identity",
    )
    if (
        identity["equal"] is not True
        or source_identity != candidate_identity
        or identity["source"] != identity["candidate"]
        or identity["shared_payload_digest"] != identity["source"]["payload_digest"]
    ):
        raise ValueError("Bout source/candidate scientific identities differ.")
    physical = payload["physical_io"]
    if physical != {
        "file_reads": None,
        "range_reads": None,
        "transferred_bytes": None,
        "availability": PHYSICAL_IO_REASON,
    }:
        raise ValueError("Bout pair fabricated or malformed physical-I/O evidence.")
    workload_payload = _require_envelope(
        payload["workload"], schema_id=WORKLOAD_SCHEMA_ID
    )
    if (
        workload_payload.get("array_count") != ARRAY_COUNT
        or len(workload_payload.get("arrays", [])) != ARRAY_COUNT
    ):
        raise ValueError("Bout workload does not visit all 20 arrays.")
    if replay_archive:
        expected = build_pair_validation(
            payload["archive_path"],
            source_run=source_name,
            candidate_run=candidate_name,
            seed=workload_payload["seed"],
            window_rows=workload_payload["window_rows"],
            windows_per_array=workload_payload["windows_per_array"],
        )
        if value != expected:
            raise ValueError(
                "Bout pair differs from live archive/selector/storage/workload replay."
            )


def _measure(call: Any) -> tuple[Any, dict[str, float]]:
    wall = time.perf_counter()
    cpu = time.process_time()
    result = call()
    return result, {
        "wall_seconds": float(time.perf_counter() - wall),
        "cpu_seconds": float(time.process_time() - cpu),
    }


def _require_nonnegative_timing(value: Mapping[str, Any], *, label: str) -> None:
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
            raise ValueError(f"{label}.{field} must be finite and nonnegative.")


def _require_utc(value: Any, *, label: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one UTC timestamp.")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must carry an explicit offset.")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must use UTC.")


def _aggregate_operations(operations: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256([dict(operation) for operation in operations])


def _read_array_workload(
    array: Any, specification: Mapping[str, Any]
) -> dict[str, Any]:
    eager_values, eager_timing = _measure(lambda: np.asarray(array[:]))
    eager_payload = _array_payload(eager_values)
    eager = {
        "read_spans": specification["eager_read_spans"],
        "operation_count": 1,
        "decoded_bytes": eager_payload["decoded_bytes"],
        "payload_digest": eager_payload["payload_digest"],
        "timing": eager_timing,
    }
    window_operations: list[dict[str, Any]] = []
    wall_total = 0.0
    cpu_total = 0.0
    for start, stop in specification["windowed_read_spans"]:
        values, timing = _measure(
            lambda start=start, stop=stop: np.asarray(array[start:stop])
        )
        payload = _array_payload(values)
        operation = {
            "span": [int(start), int(stop)],
            "dtype": payload["dtype"],
            "shape": payload["shape"],
            "decoded_bytes": payload["decoded_bytes"],
            "payload_digest": payload["payload_digest"],
        }
        window_operations.append(operation)
        wall_total += timing["wall_seconds"]
        cpu_total += timing["cpu_seconds"]
    windowed = {
        "read_spans": specification["windowed_read_spans"],
        "operation_count": len(window_operations),
        "decoded_bytes": sum(
            int(operation["decoded_bytes"]) for operation in window_operations
        ),
        "operations": window_operations,
        "operations_digest": _aggregate_operations(window_operations),
        "timing": {"wall_seconds": wall_total, "cpu_seconds": cpu_total},
    }
    return {
        "path": specification["path"],
        "dtype": np.dtype(array.dtype).str,
        "shape": [int(value) for value in array.shape],
        "declared_access_pattern": specification["declared_access_pattern"],
        "eager": eager,
        "windowed": windowed,
    }


def _strip_read_timings(reads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in reads:
        copied = {
            key: value
            for key, value in item.items()
            if key not in {"eager", "windowed"}
        }
        copied["eager"] = {
            key: value for key, value in item["eager"].items() if key != "timing"
        }
        copied["windowed"] = {
            key: value for key, value in item["windowed"].items() if key != "timing"
        }
        normalized.append(copied)
    return normalized


def _perform_array_reads(
    run_group: Any, workload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    reads = [
        _read_array_workload(_array_at(run_group, item["path"]), item)
        for item in workload["arrays"]
    ]
    if len(reads) != ARRAY_COUNT:
        raise ValueError("Bout benchmark did not visit all 20 arrays.")
    return reads


def _validate_array_reads(
    reads: Sequence[Mapping[str, Any]], workload: Mapping[str, Any]
) -> None:
    if not isinstance(reads, Sequence) or len(reads) != ARRAY_COUNT:
        raise ValueError("Bout trial array-read inventory is incomplete.")
    expected_paths = [str(item["path"]) for item in workload["arrays"]]
    if [str(item.get("path")) for item in reads] != expected_paths:
        raise ValueError("Bout trial array-read order differs from the workload.")
    for observed, expected in zip(reads, workload["arrays"]):
        if set(observed) != {
            "path",
            "dtype",
            "shape",
            "declared_access_pattern",
            "eager",
            "windowed",
        }:
            raise ValueError("Bout trial array receipt has an unexpected field set.")
        if (
            observed["dtype"] != expected["dtype"]
            or observed["shape"] != expected["shape"]
            or observed["declared_access_pattern"]
            != expected["declared_access_pattern"]
        ):
            raise ValueError("Bout trial array declaration differs from workload.")
        eager = observed["eager"]
        if set(eager) != {
            "read_spans",
            "operation_count",
            "decoded_bytes",
            "payload_digest",
            "timing",
        }:
            raise ValueError("Bout eager-read receipt has an unexpected field set.")
        if (
            eager["read_spans"] != expected["eager_read_spans"]
            or eager["operation_count"] != expected["eager_operation_count"]
            or eager["decoded_bytes"] != expected["eager_decoded_bytes"]
        ):
            raise ValueError("Bout eager-read accounting differs from workload.")
        _require_nonnegative_timing(eager["timing"], label="eager")
        windowed = observed["windowed"]
        if set(windowed) != {
            "read_spans",
            "operation_count",
            "decoded_bytes",
            "operations",
            "operations_digest",
            "timing",
        }:
            raise ValueError("Bout windowed-read receipt has an unexpected field set.")
        if (
            windowed["read_spans"] != expected["windowed_read_spans"]
            or windowed["operation_count"] != expected["windowed_operation_count"]
            or windowed["decoded_bytes"] != expected["windowed_decoded_bytes"]
            or windowed["operations_digest"]
            != _aggregate_operations(windowed["operations"])
        ):
            raise ValueError("Bout windowed-read accounting differs from workload.")
        _require_nonnegative_timing(windowed["timing"], label="windowed")


def _consumer_read(
    root: Any, *, role: str, run_name: str, run_path: str
) -> tuple[dict[str, Any], dict[str, float]]:
    if role == "source":

        def read_source() -> dict[str, Any]:
            group, resolved, resolved_path = resolve_bout_classification_run(
                root, run_name
            )
            if resolved != run_name or resolved_path != run_path:
                raise ValueError("Public source consumer resolved a different run.")
            return _structured_payload(load_bout_classification_table(group))

        return _measure(read_source)

    def read_candidate() -> dict[str, Any]:
        # This is deliberately not a public consumer.  The pair validator has
        # already run the production staged validation on this exact path.
        return _structured_payload(load_bout_classification_table(root[run_path]))

    return _measure(read_candidate)


def _environment() -> dict[str, Any]:
    git = get_git_info()
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "palette_commit": git.get("commit_hash"),
        "palette_dirty": bool(git.get("is_dirty")),
        "cache_state": CACHE_STATE,
        "thread_environment": {
            key: os.environ.get(key) for key in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
    }


def _aggregate_read_summary(reads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "array_count": len(reads),
        "eager_operation_count": sum(
            item["eager"]["operation_count"] for item in reads
        ),
        "windowed_operation_count": sum(
            item["windowed"]["operation_count"] for item in reads
        ),
        "eager_decoded_bytes": sum(item["eager"]["decoded_bytes"] for item in reads),
        "windowed_decoded_bytes": sum(
            item["windowed"]["decoded_bytes"] for item in reads
        ),
        "eager_wall_seconds": sum(
            item["eager"]["timing"]["wall_seconds"] for item in reads
        ),
        "eager_cpu_seconds": sum(
            item["eager"]["timing"]["cpu_seconds"] for item in reads
        ),
        "windowed_wall_seconds": sum(
            item["windowed"]["timing"]["wall_seconds"] for item in reads
        ),
        "windowed_cpu_seconds": sum(
            item["windowed"]["timing"]["cpu_seconds"] for item in reads
        ),
        "payload_receipt_digest": canonical_json_sha256(_strip_read_timings(reads)),
    }


def run_single_trial(
    pair: Mapping[str, Any],
    *,
    role: str,
    repetition_index: int,
    order_position: int,
    controller_pid: int,
) -> dict[str, Any]:
    """Run one read trial in a child process after live pair replay."""

    require_pair_validation(pair, replay_archive=True)
    pair_payload = pair["payload"]
    workload = pair_payload["workload"]["payload"]
    if role not in {"source", "candidate"}:
        raise ValueError("Bout trial role must be source or candidate.")
    if type(repetition_index) is not int or repetition_index < 0:
        raise ValueError("repetition_index must be nonnegative.")
    if order_position not in {0, 1}:
        raise ValueError("order_position must be zero or one.")
    if type(controller_pid) is not int or controller_pid < 1:
        raise ValueError("controller_pid must be a positive exact integer.")
    expected_order = _trial_order(
        seed=workload["seed"], repetition_index=repetition_index
    )
    if expected_order[order_position] != role:
        raise ValueError("Bout trial role differs from deterministic order.")
    child_pid = os.getpid()
    child_parent_pid = os.getppid()
    fresh = child_pid != controller_pid and child_parent_pid == controller_pid
    if not fresh:
        raise ValueError("Bout trial is not a distinct direct child process.")

    archive = _safe_archive(pair_payload["archive_path"])
    run_name = pair_payload[f"{role}_run_name"]
    run_path = pair_payload[f"{role}_run_path"]
    started = utc_now()
    initial_rss = peak_rss_bytes()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    consumer_payload, consumer_timing = _consumer_read(
        root, role=role, run_name=run_name, run_path=run_path
    )
    if consumer_payload != pair_payload["structured_table_digest"]:
        raise ValueError("Bout consumer decoded a different structured table.")
    reads = _perform_array_reads(root[run_path], workload)
    _validate_array_reads(reads, workload)
    aggregate = _aggregate_read_summary(reads)
    storage = pair_payload["storage"][role]
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "source_run_name": pair_payload["source_run_name"],
        "candidate_run_name": pair_payload["candidate_run_name"],
        "pair_payload_digest": pair["payload_digest"],
        "workload_payload_digest": pair_payload["workload"]["payload_digest"],
        "role": role,
        "run_name": run_name,
        "run_path": run_path,
        "consumer": SOURCE_CONSUMER if role == "source" else CANDIDATE_CONSUMER,
        "public_consumer_implemented": role == "source",
        "repetition_index": repetition_index,
        "order_position": order_position,
        "seed": workload["seed"],
        "controller_pid": controller_pid,
        "child_pid": child_pid,
        "child_parent_pid": child_parent_pid,
        "fresh_child_process": True,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "environment": _environment(),
        "validation": {
            "pair_live_replay": True,
            "exact_schema": True,
            "completion_and_eligibility": True,
            "direct_consolidated_metadata_equivalence": True,
            "storage_plan_replayed": True,
            "selector_state_replayed": True,
        },
        "consumer_read": {**consumer_payload, "timing": consumer_timing},
        "array_reads": reads,
        "aggregate_read": aggregate,
        "storage": storage,
        "runtime": {
            "initial_peak_rss_bytes": initial_rss,
            "final_peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_is_process_high_water_mark": True,
        },
        "profile_promoted": False,
        "selector_eligible": role == "source",
        "physical_io": {
            "file_reads": None,
            "range_reads": None,
            "transferred_bytes": None,
            "availability": PHYSICAL_IO_REASON,
        },
    }
    result = _strict_envelope(TRIAL_SCHEMA_ID, payload)
    require_trial_result(result, pair=pair, replay_archive=False)
    return result


def require_trial_result(
    value: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    replay_archive: bool = True,
) -> None:
    require_pair_validation(pair, replay_archive=replay_archive)
    pair_payload = pair["payload"]
    workload = pair_payload["workload"]["payload"]
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    if set(payload) != _TRIAL_FIELDS:
        raise ValueError("Bout trial payload has an unexpected field set.")
    role = payload["role"]
    if role not in {"source", "candidate"}:
        raise ValueError("Bout trial role is unsupported.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["archive_path"] != pair_payload["archive_path"]
        or payload["source_run_name"] != pair_payload["source_run_name"]
        or payload["candidate_run_name"] != pair_payload["candidate_run_name"]
        or payload["pair_payload_digest"] != pair["payload_digest"]
        or payload["workload_payload_digest"]
        != pair_payload["workload"]["payload_digest"]
        or payload["run_name"] != pair_payload[f"{role}_run_name"]
        or payload["run_path"] != pair_payload[f"{role}_run_path"]
        or payload["seed"] != workload["seed"]
    ):
        raise ValueError("Bout trial identity binding mismatch.")
    expected_consumer = SOURCE_CONSUMER if role == "source" else CANDIDATE_CONSUMER
    if payload["consumer"] != expected_consumer or payload[
        "public_consumer_implemented"
    ] is not (role == "source"):
        raise ValueError("Bout trial consumer claim is false.")
    if payload["profile_promoted"] is not False or payload["selector_eligible"] is not (
        role == "source"
    ):
        raise ValueError("Bout trial promotion/eligibility claim is false.")
    if (
        type(payload["repetition_index"]) is not int
        or payload["repetition_index"] < 0
        or type(payload["order_position"]) is not int
    ):
        raise ValueError("Bout trial repetition/order values are invalid.")
    if (
        payload["fresh_child_process"] is not True
        or payload["child_pid"] == payload["controller_pid"]
        or payload["child_parent_pid"] != payload["controller_pid"]
    ):
        raise ValueError("Bout trial fresh-child binding is invalid.")
    if (
        payload["order_position"] not in {0, 1}
        or _trial_order(
            seed=workload["seed"], repetition_index=payload["repetition_index"]
        )[payload["order_position"]]
        != role
    ):
        raise ValueError("Bout trial deterministic order is invalid.")
    _require_utc(payload["started_at_utc"], label="trial started_at_utc")
    _require_utc(payload["finished_at_utc"], label="trial finished_at_utc")
    if datetime.fromisoformat(payload["finished_at_utc"]) < datetime.fromisoformat(
        payload["started_at_utc"]
    ):
        raise ValueError("Bout trial finished before it started.")
    environment = payload["environment"]
    if not isinstance(environment, Mapping) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("Bout trial environment has an unexpected field set.")
    for field in ("hostname", "system", "release", "python", "numpy", "zarr"):
        if type(environment[field]) is not str or not environment[field]:
            raise ValueError(f"Bout trial environment {field} is invalid.")
    if type(environment["palette_dirty"]) is not bool:
        raise ValueError("Bout trial palette_dirty must be an exact bool.")
    if (
        environment["palette_commit"] is not None
        and type(environment["palette_commit"]) is not str
    ):
        raise ValueError("Bout trial Palette commit has the wrong type.")
    if environment["cache_state"] != CACHE_STATE:
        raise ValueError("Bout trial cache-state declaration differs.")
    thread_environment = environment["thread_environment"]
    if not isinstance(thread_environment, Mapping) or set(thread_environment) != set(
        STORAGE_BENCHMARK_THREAD_ENVIRONMENT
    ):
        raise ValueError("Bout trial thread environment has an unexpected field set.")
    for key, expected in STORAGE_BENCHMARK_THREAD_ENVIRONMENT.items():
        if thread_environment[key] != expected:
            raise ValueError(f"Bout trial deterministic thread setting differs: {key}.")
    if payload["validation"] != {
        "pair_live_replay": True,
        "exact_schema": True,
        "completion_and_eligibility": True,
        "direct_consolidated_metadata_equivalence": True,
        "storage_plan_replayed": True,
        "selector_state_replayed": True,
    }:
        raise ValueError("Bout trial validation claims are incomplete.")
    consumer_read = payload["consumer_read"]
    if set(consumer_read) != {
        "dtype_descr",
        "shape",
        "decoded_bytes",
        "payload_digest",
        "timing",
    }:
        raise ValueError("Bout trial consumer receipt has an unexpected field set.")
    if {key: value for key, value in consumer_read.items() if key != "timing"} != (
        pair_payload["structured_table_digest"]
    ):
        raise ValueError("Bout trial consumer digest differs from pair authority.")
    _require_nonnegative_timing(consumer_read["timing"], label="consumer")
    _validate_array_reads(payload["array_reads"], workload)
    if payload["aggregate_read"] != _aggregate_read_summary(payload["array_reads"]):
        raise ValueError("Bout trial aggregate is not recomputed from array receipts.")
    if payload["storage"] != pair_payload["storage"][role]:
        raise ValueError("Bout trial storage facts differ from live pair facts.")
    runtime = payload["runtime"]
    if (
        set(runtime)
        != {
            "initial_peak_rss_bytes",
            "final_peak_rss_bytes",
            "peak_rss_is_process_high_water_mark",
        }
        or runtime["peak_rss_is_process_high_water_mark"] is not True
    ):
        raise ValueError("Bout trial runtime receipt has an unexpected field set.")
    if (
        type(runtime["initial_peak_rss_bytes"]) is not int
        or type(runtime["final_peak_rss_bytes"]) is not int
        or runtime["initial_peak_rss_bytes"] < 0
        or runtime["final_peak_rss_bytes"] < 0
    ):
        raise ValueError("Bout trial RSS facts are invalid.")
    if payload["physical_io"] != {
        "file_reads": None,
        "range_reads": None,
        "transferred_bytes": None,
        "availability": PHYSICAL_IO_REASON,
    }:
        raise ValueError("Bout trial fabricated physical-I/O evidence.")
    if replay_archive:
        archive = _safe_archive(pair_payload["archive_path"])
        root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
        live_reads = _perform_array_reads(root[payload["run_path"]], workload)
        if _strip_read_timings(payload["array_reads"]) != _strip_read_timings(
            live_reads
        ):
            raise ValueError(
                "Bout trial array receipts differ from live workload replay."
            )
        live_consumer, _timing = _consumer_read(
            root,
            role=role,
            run_name=payload["run_name"],
            run_path=payload["run_path"],
        )
        if live_consumer != pair_payload["structured_table_digest"]:
            raise ValueError("Bout trial consumer differs from live replay.")


def _trial_order(*, seed: int, repetition_index: int) -> tuple[str, str]:
    return (
        ("source", "candidate")
        if (seed + repetition_index) % 2 == 0
        else ("candidate", "source")
    )


def _median(
    trials: Sequence[Mapping[str, Any]], role: str, path: Sequence[str]
) -> float:
    values: list[float] = []
    for trial in trials:
        node: Any = trial["payload"]
        if node["role"] != role:
            continue
        for component in path:
            node = node[component]
        values.append(float(node))
    return float(statistics.median(values))


def _performance_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("source", "candidate"):
        representative = next(
            trial["payload"] for trial in trials if trial["payload"]["role"] == role
        )
        result[role] = {
            "median_consumer_wall_seconds": _median(
                trials, role, ("consumer_read", "timing", "wall_seconds")
            ),
            "median_eager_wall_seconds": _median(
                trials, role, ("aggregate_read", "eager_wall_seconds")
            ),
            "median_windowed_wall_seconds": _median(
                trials, role, ("aggregate_read", "windowed_wall_seconds")
            ),
            "median_final_peak_rss_bytes": _median(
                trials, role, ("runtime", "final_peak_rss_bytes")
            ),
            "eager_operation_count": representative["aggregate_read"][
                "eager_operation_count"
            ],
            "windowed_operation_count": representative["aggregate_read"][
                "windowed_operation_count"
            ],
            "eager_decoded_bytes": representative["aggregate_read"][
                "eager_decoded_bytes"
            ],
            "windowed_decoded_bytes": representative["aggregate_read"][
                "windowed_decoded_bytes"
            ],
            "payload_object_count": representative["storage"]["payload_file_count"],
            "apparent_bytes": representative["storage"]["apparent_bytes"],
            "allocated_bytes": representative["storage"]["allocated_bytes"],
        }
    return result


_MATRIX_FIELDS = {
    "benchmark_id",
    "family_id",
    "archive_path",
    "source_run_name",
    "candidate_run_name",
    "controller_pid",
    "seed",
    "repetitions",
    "cache_state",
    "started_at_utc",
    "finished_at_utc",
    "pair_validation",
    "pair_file",
    "trial_order",
    "trial_files",
    "trials",
    "correctness",
    "performance_summary",
    "archive_read_only_guard",
    "evidence_boundaries",
    "consumers",
    "profile_promoted",
    "candidate_selector_eligible",
    "physical_io",
    "balanced_read_matrix_complete",
}


def require_matrix_result(
    value: Mapping[str, Any], *, replay_archive: bool = True
) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    if set(payload) != _MATRIX_FIELDS:
        raise ValueError("Bout matrix payload has an unexpected field set.")
    pair = payload["pair_validation"]
    require_pair_validation(pair, replay_archive=replay_archive)
    pair_payload = pair["payload"]
    workload = pair_payload["workload"]["payload"]
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["archive_path"] != pair_payload["archive_path"]
        or payload["source_run_name"] != pair_payload["source_run_name"]
        or payload["candidate_run_name"] != pair_payload["candidate_run_name"]
        or payload["seed"] != workload["seed"]
        or payload["cache_state"] != CACHE_STATE
    ):
        raise ValueError("Bout matrix identity binding mismatch.")
    if type(payload["repetitions"]) is not int or payload["repetitions"] < 1:
        raise ValueError("Bout matrix repetitions must be positive.")
    if type(payload["controller_pid"]) is not int or payload["controller_pid"] < 1:
        raise ValueError("Bout matrix controller PID is invalid.")
    if payload["pair_file"] != "pair_validation.json":
        raise ValueError("Bout matrix pair-file binding is invalid.")
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Bout matrix trial count is invalid.")
    expected_order = [
        {
            "repetition_index": repetition,
            "roles": list(
                _trial_order(seed=payload["seed"], repetition_index=repetition)
            ),
        }
        for repetition in range(payload["repetitions"])
    ]
    if payload["trial_order"] != expected_order:
        raise ValueError("Bout matrix deterministic order differs.")
    expected_trial_files = [
        f"trials/rep_{repetition:02d}_pos_{position}_{role}.json"
        for repetition, record in enumerate(expected_order)
        for position, role in enumerate(record["roles"])
    ]
    if payload["trial_files"] != expected_trial_files:
        raise ValueError("Bout matrix trial-file inventory is invalid.")
    child_pids: set[int] = set()
    observed_coordinates: list[tuple[int, int, str]] = []
    for trial in trials:
        require_trial_result(trial, pair=pair, replay_archive=replay_archive)
        trial_payload = trial["payload"]
        if trial_payload["controller_pid"] != payload["controller_pid"]:
            raise ValueError("Bout matrix/trial controller PID differs.")
        child_pids.add(trial_payload["child_pid"])
        observed_coordinates.append(
            (
                trial_payload["repetition_index"],
                trial_payload["order_position"],
                trial_payload["role"],
            )
        )
    expected_coordinates = [
        (repetition, position, role)
        for repetition, record in enumerate(expected_order)
        for position, role in enumerate(record["roles"])
    ]
    if observed_coordinates != expected_coordinates:
        raise ValueError("Bout matrix trial coordinates differ from balanced order.")
    if len(child_pids) != len(trials) or payload["controller_pid"] in child_pids:
        raise ValueError("Bout matrix did not use one fresh process per trial.")
    source_reads = [
        _strip_read_timings(trial["payload"]["array_reads"])
        for trial in trials
        if trial["payload"]["role"] == "source"
    ]
    candidate_reads = [
        _strip_read_timings(trial["payload"]["array_reads"])
        for trial in trials
        if trial["payload"]["role"] == "candidate"
    ]
    if (
        not source_reads
        or not candidate_reads
        or any(reads != source_reads[0] for reads in source_reads + candidate_reads)
    ):
        raise ValueError("Bout matrix source/candidate workload equality failed.")
    if payload["performance_summary"] != _performance_summary(trials):
        raise ValueError("Bout matrix performance summary is not recomputed.")
    if payload["archive_read_only_guard"] != {
        "before": pair,
        "after": pair,
        "unchanged": True,
    }:
        raise ValueError("Bout matrix archive guard is not exact.")
    if payload["correctness"] != {
        "all_twenty_arrays_visited": True,
        "complete_decoded_equality": True,
        "exact_schema_and_semantic_fills": True,
        "direct_consolidated_metadata_equivalence": True,
        "storage_plan_replayed": True,
        "selector_nonmutation": True,
        "fresh_child_processes": True,
        "all_passed": True,
    }:
        raise ValueError("Bout matrix correctness gates are incomplete.")
    if payload["evidence_boundaries"] != {
        "writer_phase_measured": False,
        "publication_phase_measured": False,
        "physical_io_measured": False,
        "representative_scale_executed": False,
        "promotion_gate_executed": False,
        "runtime_observations_attested": False,
    }:
        raise ValueError("Bout matrix overclaims evidence coverage.")
    if payload["consumers"] != pair_payload["consumers"]:
        raise ValueError("Bout matrix consumer boundary differs from the pair.")
    if (
        payload["profile_promoted"] is not False
        or payload["candidate_selector_eligible"] is not False
    ):
        raise ValueError("Bout matrix must remain unpromoted and selector ineligible.")
    if payload["physical_io"] != pair_payload["physical_io"]:
        raise ValueError("Bout matrix fabricated physical-I/O evidence.")
    if payload["balanced_read_matrix_complete"] is not (
        payload["repetitions"] == DEFAULT_REPETITIONS
    ):
        raise ValueError("Bout balanced-matrix classification is invalid.")
    _require_utc(payload["started_at_utc"], label="matrix started_at_utc")
    _require_utc(payload["finished_at_utc"], label="matrix finished_at_utc")


def run_benchmark_matrix(
    archive_path: str | Path,
    *,
    source_run: str,
    candidate_run: str,
    output_dir: str | Path,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    windows_per_array: int = DEFAULT_WINDOWS_PER_ARRAY,
) -> dict[str, Any]:
    """Run balanced fresh-process reads and emit immutable sidecar evidence."""

    archive = _safe_archive(archive_path)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be one positive exact integer.")
    output = _safe_new_output_dir(output_dir, archive=archive)
    pair = build_pair_validation(
        archive,
        source_run=source_run,
        candidate_run=candidate_run,
        seed=seed,
        window_rows=window_rows,
        windows_per_array=windows_per_array,
    )
    require_pair_validation(pair, replay_archive=True)
    output.mkdir(parents=True, exist_ok=False)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    pair_path = output / "pair_validation.json"
    _write_json(pair_path, pair)
    started = utc_now()
    controller_pid = os.getpid()
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
                "fisheye.diagnostics.benchmark_bout_classification_v2_reads",
                "trial",
                "--pair",
                str(pair_path),
                "--benchmark-root",
                str(output),
                "--output",
                str(trial_path),
                "--role",
                role,
                "--repetition-index",
                str(repetition_index),
                "--order-position",
                str(order_position),
                "--controller-pid",
                str(controller_pid),
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
                    "Fresh-process bout benchmark trial failed: "
                    f"command={command!r}, stdout={completed.stdout!r}, "
                    f"stderr={completed.stderr!r}."
                )
            trial = _read_json(trial_path)
            require_trial_result(trial, pair=pair, replay_archive=True)
            trials.append(trial)
            trial_files.append(str(trial_path.relative_to(output)))
    after = build_pair_validation(
        archive,
        source_run=source_run,
        candidate_run=candidate_run,
        seed=seed,
        window_rows=window_rows,
        windows_per_array=windows_per_array,
    )
    if after != pair:
        raise RuntimeError("Bout archive/selector/storage state changed during reads.")
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_path": str(archive),
        "source_run_name": pair["payload"]["source_run_name"],
        "candidate_run_name": pair["payload"]["candidate_run_name"],
        "controller_pid": controller_pid,
        "seed": seed,
        "repetitions": repetitions,
        "cache_state": CACHE_STATE,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "pair_validation": pair,
        "pair_file": "pair_validation.json",
        "trial_order": trial_order,
        "trial_files": trial_files,
        "trials": trials,
        "correctness": {
            "all_twenty_arrays_visited": True,
            "complete_decoded_equality": True,
            "exact_schema_and_semantic_fills": True,
            "direct_consolidated_metadata_equivalence": True,
            "storage_plan_replayed": True,
            "selector_nonmutation": True,
            "fresh_child_processes": True,
            "all_passed": True,
        },
        "performance_summary": _performance_summary(trials),
        "archive_read_only_guard": {
            "before": pair,
            "after": after,
            "unchanged": True,
        },
        "evidence_boundaries": {
            "writer_phase_measured": False,
            "publication_phase_measured": False,
            "physical_io_measured": False,
            "representative_scale_executed": False,
            "promotion_gate_executed": False,
            "runtime_observations_attested": False,
        },
        "consumers": pair["payload"]["consumers"],
        "profile_promoted": False,
        "candidate_selector_eligible": False,
        "physical_io": pair["payload"]["physical_io"],
        "balanced_read_matrix_complete": repetitions == DEFAULT_REPETITIONS,
    }
    result = _strict_envelope(MATRIX_SCHEMA_ID, payload)
    require_matrix_result(result, replay_archive=True)
    _write_json(output / "matrix_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("archive", type=Path)
    matrix.add_argument("--source-run", required=True)
    matrix.add_argument("--candidate-run", required=True)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    matrix.add_argument("--seed", type=int, default=DEFAULT_SEED)
    matrix.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
    matrix.add_argument(
        "--windows-per-array",
        type=_parse_windows_per_array,
        default=DEFAULT_WINDOWS_PER_ARRAY,
    )
    trial = subparsers.add_parser("trial")
    trial.add_argument("--pair", type=Path, required=True)
    trial.add_argument("--benchmark-root", type=Path, required=True)
    trial.add_argument("--output", type=Path, required=True)
    trial.add_argument("--role", choices=("source", "candidate"), required=True)
    trial.add_argument("--repetition-index", type=int, required=True)
    trial.add_argument("--order-position", type=int, choices=(0, 1), required=True)
    trial.add_argument("--controller-pid", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        result = run_benchmark_matrix(
            args.archive,
            source_run=args.source_run,
            candidate_run=args.candidate_run,
            output_dir=args.output,
            repetitions=args.repetitions,
            seed=args.seed,
            window_rows=args.window_rows,
            windows_per_array=args.windows_per_array,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "matrix_result": str(args.output / "matrix_result.json"),
                    "payload_digest": result["payload_digest"],
                },
                sort_keys=True,
            )
        )
        return 0
    benchmark_root = args.benchmark_root.expanduser().resolve(strict=True)
    pair_path = args.pair.expanduser().resolve(strict=True)
    if (
        not pair_path.is_relative_to(benchmark_root)
        or pair_path.name != "pair_validation.json"
    ):
        raise ValueError(
            "Pair receipt must be the benchmark root pair_validation.json."
        )
    pair = _read_json(pair_path)
    output = _safe_trial_output(args.output, benchmark_root=benchmark_root)
    result = run_single_trial(
        pair,
        role=args.role,
        repetition_index=args.repetition_index,
        order_position=args.order_position,
        controller_pid=args.controller_pid,
    )
    _write_json(output, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
