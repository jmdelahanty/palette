"""Read-only source/candidate matrix for maintained tail-posture-view v3.

Neither payload adapter in this module is a production consumer.  The source
adapter separately exercises Palette's maintained private selector boundary
and its public coordinate-publication validator.  The candidate adapter is an
explicit diagnostic validator for one completed, selector-ineligible,
byte-planned run.  The full Megabouts input-pack consumer remains a separate
gate because it also requires track-kinematics and swim-bout authorities.

The controller rotates source/candidate order through fresh child processes.
It never mutates the archive, selectors, registries, or profile state and it
cannot report promotion or physical I/O without an external trace.
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

import numpy as np
import zarr

from fisheye.analysis import megabouts_classifier_inputs as megabouts_inputs
from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    ANALYSIS_STORAGE_PROFILE_ID_ATTR,
    ANALYSIS_STORAGE_PROFILE_ROLE,
    ANALYSIS_STORAGE_PROFILE_ROLE_ATTR,
)
from fisheye.analysis.tail_posture_view_schema import (
    TAIL_POSTURE_VIEW_ARRAY_DECLARATIONS,
    TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR,
    TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR,
    TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS,
    TAIL_POSTURE_VIEW_RUN_SCHEMA_ID,
    TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION,
    TailPostureViewDimensions,
    validate_tail_posture_view_arrays,
)
from fisheye.shared import tail_coordinate_publication as tail_publication
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.analysis_storage_planning import (
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_environment import (
    STORAGE_BENCHMARK_THREAD_ENVIRONMENT,
)
from fisheye.shared.zarr.benchmark_runtime import peak_rss_bytes, utc_now
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
)


PARENT_PATH = "analysis/tail_posture_view_runs"
VIEW_FAMILY = "megabouts_compatible"
FAMILY_SELECTOR = f"latest_{VIEW_FAMILY}"
SELECTOR_NAMES = ("latest", "latest_complete", FAMILY_SELECTOR)
FAMILY_ID = "tail_posture_view_v3"
BENCHMARK_ID = "tail_posture_view_v3_source_candidate_read_matrix_v1"
PAIR_SCHEMA_ID = "palette.tail_posture_view.v3_pair_validation"
WORKLOAD_SCHEMA_ID = "palette.tail_posture_view.v3_read_workload"
TRIAL_SCHEMA_ID = "palette.tail_posture_view.v3_read_trial"
MATRIX_SCHEMA_ID = "palette.tail_posture_view.v3_read_matrix"
SCHEMA_VERSION = 1
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 43
DEFAULT_WINDOW_ROWS = 4096
DEFAULT_WINDOWS_PER_ARRAY = 4
ALIASES = frozenset({"latest", "latest_complete", "latest_pending"})
PUBLICATION_MODE = "guarded_direct_writer_v1"
SOURCE_ADAPTER = "diagnostic_only_exact_v3_payload_adapter"
CANDIDATE_ADAPTER = "diagnostic_only_byte_planned_v3_payload_adapter"
PUBLIC_SOURCE_SELECTION_EVIDENCE = (
    "maintained_private_megabouts_tail_posture_resolver"
)
PUBLIC_SOURCE_COORDINATE_EVIDENCE = (
    "public_load_tail_posture_coordinate_publication"
)
BROADER_CONSUMER_STATUS = (
    "not_run_requires_track_kinematics_and_swim_bout_authorities"
)
PHYSICAL_IO_REASON = (
    "not_collected_requires_os_or_filesystem_trace; decoded bytes are not "
    "physical transfer telemetry"
)
PROMOTION_POLICY = (
    "hard_nonpromotion_benchmark_only_no_selector_registry_or_profile_changes"
)
TAIL_ANGLE_DEG_ATOL = 1.0e-5
TAIL_ANGLE_DEG_RTOL = 1.0e-6
SCIENTIFIC_IDENTITY_FIELDS = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "row_axis",
    "view_family",
    "compatible_tool",
    "dependency_policy",
    "source_subject_shape_run",
    "source_subject_shape_path",
    "source_subject_shape_publication_manifest_sha256",
    "source_refined_subject_masks_run",
    "source_tail_kinematics_run",
    "source_tail_geometry_kind",
    "head_source",
    "keypoint_count",
    "angle_count",
    "angle_units_primary",
    "angle_convention",
    "keypoint_order",
    "tail_base_definition",
    "tail_tip_definition",
    "acquisition_frame_index_source",
    "row_lineage_copied",
    "row_lineage_missing",
    "source_refs",
    "algorithm_provenance",
    "reason_encoding",
    "reason_bytes_width",
    "reason_bytes_null_terminated",
)
_PAIR_FIELDS = frozenset(
    {
        "family_id",
        "archive_identity",
        "source_run",
        "candidate_run",
        "selectors",
        "metadata_equivalence",
        "source",
        "candidate",
        "logical_equality",
        "consumer_boundary",
        "publication_mode",
        "promotion_authorized",
    }
)
_ROLE_FIELDS = frozenset(
    {
        "role",
        "run_name",
        "run_path",
        "adapter",
        "schema_id",
        "schema_version",
        "dimensions",
        "array_count",
        "array_schema_sha256",
        "storage_plan_sha256",
        "coordinate_manifest_ref",
        "coordinate_manifest_sha256",
        "source_subject_shape_run",
        "source_subject_shape_manifest_sha256",
        "completion_status",
        "selector_eligible",
        "scientific_identity",
        "semantic_validation",
        "logical_arrays",
    }
)
_TRIAL_FIELDS = frozenset(
    {
        "benchmark_id",
        "family_id",
        "archive_identity",
        "source_run",
        "candidate_run",
        "role",
        "repetition_index",
        "order_position",
        "driver_process_id",
        "selectors_before",
        "selectors_after",
        "pair_validation",
        "workload",
        "result",
        "environment",
        "physical_io",
        "consumer_boundary",
        "promotion_authorized",
        "started_at_utc",
        "finished_at_utc",
    }
)


def _strict_json_copy(value: object) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tail-posture benchmark evidence is not strict JSON: {exc}") from exc
    return json.loads(
        encoded,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token {token}")
        ),
    )


def _envelope(schema_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
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
        raise ValueError("Tail-posture evidence envelope field set is not exact.")
    if value.get("schema_id") != schema_id or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Tail-posture evidence schema identity is unsupported.")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Tail-posture evidence payload must be one object.")
    if value.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Tail-posture evidence payload digest mismatch.")
    _strict_json_copy(value)
    return payload


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_name(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value.lower() in ALIASES
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one explicit immutable child name.")
    return value


def _safe_archive(value: str | Path) -> Path:
    raw = Path(value).expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("Archive must be one existing non-symlink directory.")
    resolved = raw.resolve(strict=True)
    if resolved != raw or not (resolved / "zarr.json").is_file():
        raise ValueError("Archive path must be canonical and contain a Zarr root.")
    return resolved


def _guard_relative_tree(archive: Path, relative: str, *, label: str) -> Path:
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError(f"{label} is not one canonical archive-relative path.")
    current = archive
    for component in relative.split("/"):
        current = current / component
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"{label} contains a missing/non-directory/symlink component.")
        try:
            current.resolve(strict=True).relative_to(archive)
        except ValueError as exc:
            raise ValueError(f"{label} escapes the archive.") from exc
    for root, directory_names, file_names in os.walk(current, followlinks=False):
        for name in (*directory_names, *file_names):
            child = Path(root) / name
            if child.is_symlink():
                raise ValueError(f"{label} contains a forbidden symlink: {child}.")
            try:
                child.resolve(strict=True).relative_to(archive)
            except ValueError as exc:
                raise ValueError(f"{label} descendant escapes the archive.") from exc
    return current


def _safe_output_root(value: str | Path, *, archive: Path) -> Path:
    raw = Path(value).expanduser().absolute()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"Benchmark output already exists: {raw}.")
    parent = raw.parent.resolve(strict=True)
    if parent.is_symlink():
        raise ValueError("Benchmark output parent must not be a symlink.")
    if not any("benchmark" in part.lower() for part in raw.parts):
        raise ValueError("Benchmark output path must visibly identify benchmark scope.")
    if raw == archive or raw.is_relative_to(archive) or archive.is_relative_to(raw):
        raise ValueError("Benchmark output must be disjoint from the source archive.")
    return raw


def _safe_trial_output(value: str | Path, *, output_root: Path) -> Path:
    raw = Path(value).expanduser().absolute()
    if raw.exists() or raw.is_symlink() or raw.suffix != ".json":
        raise ValueError("Trial output must be one new JSON file.")
    root = output_root.resolve(strict=True)
    parent = raw.parent.resolve(strict=True)
    if parent.is_symlink() or not raw.is_relative_to(root):
        raise ValueError("Trial output must remain below its benchmark root.")
    return raw


def _safe_existing_output_root(value: str | Path) -> Path:
    raw = Path(value).expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir() or raw.resolve(strict=True) != raw:
        raise ValueError("Existing benchmark root must be canonical and nonsymlinked.")
    if not any("benchmark" in part.lower() for part in raw.parts):
        raise ValueError("Existing output root does not identify benchmark scope.")
    return raw


def _safe_evidence_input(
    value: str | Path,
    *,
    output_root: Path,
    label: str,
) -> Path:
    raw = Path(value).expanduser().absolute()
    if (
        raw.is_symlink()
        or not raw.is_file()
        or raw.resolve(strict=True) != raw
        or not raw.is_relative_to(output_root)
        or raw.suffix != ".json"
    ):
        raise ValueError(f"{label} must be one canonical JSON below the benchmark root.")
    return raw


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence at {path} must be one object.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_identity(archive: Path) -> dict[str, object]:
    metadata = archive / "zarr.json"
    stat = metadata.stat()
    return {
        "resolved_path": str(archive),
        "root_metadata_sha256": _sha256_file(metadata),
        "root_metadata_device": int(stat.st_dev),
        "root_metadata_inode": int(stat.st_ino),
    }


def _selector_snapshot(parent: Any) -> dict[str, object]:
    return {
        name: {"present": name in parent.attrs, "value": parent.attrs.get(name)}
        for name in SELECTOR_NAMES
    }


def _require_expected_selectors(snapshot: Mapping[str, object], source_run: str) -> None:
    expected = {
        name: {"present": True, "value": source_run} for name in SELECTOR_NAMES
    }
    if dict(snapshot) != expected:
        raise ValueError(
            "Tail-posture latest/latest_complete/latest_megabouts_compatible must "
            "all select the frozen source run."
        )


def _run_path(name: str) -> str:
    return f"{PARENT_PATH}/{name}"


def _open_primary(archive: Path) -> zarr.Group:
    # This benchmark targets finalized immutable publications.  Consolidated
    # metadata is therefore the reader surface; direct declarations are checked
    # independently by validate_direct_consolidated_subtree.
    return zarr.open_group(archive, mode="r", use_consolidated=True)


def _require_run(root: Any, name: str) -> tuple[Any, Any]:
    parent = root.get(PARENT_PATH)
    if not isinstance(parent, zarr.Group):
        raise ValueError(f"Missing maintained tail-posture parent {PARENT_PATH}.")
    child = parent.get(name)
    if not isinstance(child, zarr.Group):
        raise ValueError(f"Missing tail-posture run {_run_path(name)}.")
    return parent, child


def _dimensions(run: Any) -> TailPostureViewDimensions:
    keys = run.get("instance_key")
    points = run.get("tail_keypoints_xy")
    angles = run.get("tail_angle_rad")
    if not isinstance(keys, zarr.Array) or not isinstance(points, zarr.Array) or not isinstance(angles, zarr.Array):
        raise ValueError("Tail-posture dimension anchors are absent or not arrays.")
    n_rows = int(keys.shape[0])
    n_keypoints = run.attrs.get("keypoint_count")
    n_angles = run.attrs.get("angle_count")
    if type(n_keypoints) is not int or type(n_angles) is not int:
        raise ValueError("Tail-posture dimension attrs must be exact integers.")
    dimensions = TailPostureViewDimensions(
        n_rows=n_rows,
        n_keypoints=n_keypoints,
        n_angles=n_angles,
    )
    if tuple(points.shape) != (n_rows, n_keypoints, 2) or tuple(angles.shape) != (
        n_rows,
        n_angles,
    ):
        raise ValueError("Tail-posture dimension attrs disagree with payload arrays.")
    return dimensions


def _array_bits_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _logical_array_inventory(run: Any) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for declaration in TAIL_POSTURE_VIEW_ARRAY_DECLARATIONS:
        array = run[declaration.path]
        values = np.asarray(array[:])
        result.append(
            {
                "path": declaration.path,
                "dtype": str(values.dtype),
                "shape": [int(value) for value in values.shape],
                "decoded_bytes": int(values.nbytes),
                "bits_sha256": _array_bits_digest(values),
            }
        )
    return result


def _decode_reason_rows(values: np.ndarray) -> tuple[str, ...]:
    reasons = np.asarray(values)
    if reasons.dtype != np.dtype("uint8") or reasons.ndim != 2 or reasons.shape[1] != 64:
        raise ValueError("failure_reason_bytes must be exact uint8[n_rows,64].")
    decoded: list[str] = []
    for row_index, row in enumerate(reasons):
        zero_indices = np.flatnonzero(row == 0)
        if zero_indices.size == 0:
            raise ValueError(
                f"failure_reason_bytes row {row_index} has no NUL terminator."
            )
        terminator = int(zero_indices[0])
        if terminator == 0 or np.any(row[terminator:] != 0):
            raise ValueError(
                f"failure_reason_bytes row {row_index} is empty or not zero padded."
            )
        payload = row[:terminator].tobytes()
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"failure_reason_bytes row {row_index} is not valid UTF-8."
            ) from exc
        if not text or text.encode("utf-8") != payload:
            raise ValueError(
                f"failure_reason_bytes row {row_index} is not canonical UTF-8."
            )
        decoded.append(text)
    return tuple(decoded)


def _semantic_validation(run: Any, dimensions: TailPostureViewDimensions) -> dict[str, object]:
    instance_keys = np.asarray(run["instance_key"][:])
    crop_rows = np.asarray(run["source_crop_row_ids"][:])
    frame_indices = np.asarray(run["source_acquisition_frame_index"][:])
    valid = np.asarray(run["valid"][:])
    reasons = _decode_reason_rows(np.asarray(run["failure_reason_bytes"][:]))
    if np.unique(instance_keys).size != dimensions.n_rows:
        raise ValueError("instance_key must be unique across the tail-posture row axis.")
    if np.any(crop_rows < 0) or np.any(frame_indices < 0):
        raise ValueError("Tail-posture crop/frame lineage must be nonnegative.")
    if valid.dtype != np.dtype("bool") or valid.shape != (dimensions.n_rows,):
        raise ValueError("Tail-posture valid must be exact bool[n_rows].")

    floating_paths = (
        "head_xy",
        "head_yaw_rad",
        "tail_keypoints_xy",
        "tail_angle_rad",
        "tail_angle_deg",
    )
    floating = {
        path: np.asarray(run[path][:], dtype=np.float32) for path in floating_paths
    }
    for row_index, is_valid in enumerate(valid.tolist()):
        reason = reasons[row_index]
        if is_valid:
            if reason != "ok":
                raise ValueError("Every valid tail-posture row must have reason 'ok'.")
            if any(
                not np.all(np.isfinite(values[row_index]))
                for values in floating.values()
            ):
                raise ValueError(
                    "Every floating payload on a valid tail-posture row must be finite."
                )
        else:
            if reason == "ok":
                raise ValueError("Every invalid tail-posture row must have a non-ok reason.")
            if any(
                not np.all(np.isnan(values[row_index]))
                for values in floating.values()
            ):
                raise ValueError(
                    "Every floating payload on an invalid tail-posture row must be NaN."
                )

    radians = floating["tail_angle_rad"][valid]
    degrees = floating["tail_angle_deg"][valid]
    expected_degrees = np.degrees(radians).astype(np.float32, copy=False)
    if not np.allclose(
        degrees,
        expected_degrees,
        rtol=TAIL_ANGLE_DEG_RTOL,
        atol=TAIL_ANGLE_DEG_ATOL,
        equal_nan=False,
    ):
        raise ValueError(
            "tail_angle_deg differs from float32 rad-to-deg conversion beyond policy."
        )
    payload = {
        "policy_id": "tail_posture_view_v3_row_semantics_v1",
        "n_rows": dimensions.n_rows,
        "valid_rows": int(np.count_nonzero(valid)),
        "invalid_rows": int(dimensions.n_rows - np.count_nonzero(valid)),
        "unique_instance_keys": True,
        "nonnegative_crop_and_frame_lineage": True,
        "reason_encoding": "utf8_nul_terminated_zero_padded_64_bytes",
        "valid_reason": "ok",
        "invalid_reason": "nonempty_and_not_ok",
        "valid_float_policy": "all_finite",
        "invalid_float_policy": "all_nan",
        "angle_deg_policy": {
            "operation": "numpy_degrees_float32",
            "rtol": TAIL_ANGLE_DEG_RTOL,
            "atol": TAIL_ANGLE_DEG_ATOL,
        },
        "reason_rows_sha256": canonical_json_sha256(list(reasons)),
    }
    return {**payload, "semantic_digest": canonical_json_sha256(payload)}


def _scientific_identity(run: Any) -> dict[str, object]:
    identity = {name: run.attrs.get(name) for name in SCIENTIFIC_IDENTITY_FIELDS}
    source_name = identity["source_subject_shape_run"]
    source_path = identity["source_subject_shape_path"]
    if (
        type(source_name) is not str
        or not source_name
        or source_path != f"analysis/subject_shape_runs/{source_name}"
        or not _is_sha256(identity["source_subject_shape_publication_manifest_sha256"])
    ):
        raise ValueError("Tail-posture subject-shape scientific identity is invalid.")
    source_tail = identity["source_tail_kinematics_run"]
    expected_refs = {
        "subject_shape_run": source_path,
        "subject_shape_body_component": f"{source_path}/components/subject_body",
    }
    if source_tail is not None:
        if type(source_tail) is not str or not source_tail or "/" in source_tail:
            raise ValueError("Tail-posture source-tail run identity is invalid.")
        expected_refs["tail_kinematics_run"] = (
            f"analysis/tail_kinematics_runs/{source_tail}"
        )
    expected_fixed = {
        "schema_id": TAIL_POSTURE_VIEW_RUN_SCHEMA_ID,
        "schema_version": TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION,
        "method": "tail_posture_view_from_subject_shape",
        "method_version": 1,
        "row_axis": "observation_instance",
        "view_family": VIEW_FAMILY,
        "compatible_tool": "megabouts",
        "dependency_policy": "no_megabouts_dependency_required",
        "source_tail_geometry_kind": "subject_shape_tail_curve_resample",
        "angle_units_primary": "rad",
        "angle_convention": "megabouts_cumulative_segment_angle",
        "keypoint_order": "tail_base_to_tail_tip",
        "tail_base_definition": (
            "subject_shape.components.subject_body.tail_sample_xy[:,0]"
        ),
        "tail_tip_definition": (
            "subject_shape.components.subject_body.tail_sample_xy[:,-1]"
        ),
        "acquisition_frame_index_source": "source_acquisition_frame_index",
        "row_lineage_copied": [
            "instance_key",
            "source_crop_row_ids",
            "source_acquisition_frame_index",
        ],
        "row_lineage_missing": [],
        "source_refs": expected_refs,
        "algorithm_provenance": {
            "implementation": "independent_palette_compatible",
            "compatible_with": (
                "megabouts.tracking_data.convert_tracking.compute_angles_from_keypoints"
            ),
            "copies_megabouts_code": False,
            "requires_megabouts_install": False,
        },
        "reason_encoding": "utf8-null-terminated",
        "reason_bytes_width": 64,
        "reason_bytes_null_terminated": True,
    }
    for name, expected in expected_fixed.items():
        if identity.get(name) != expected:
            raise ValueError(f"Tail-posture scientific identity field {name!r} is invalid.")
    if identity["head_source"] not in {"head_endpoint_xy", "snout_tip_xy"}:
        raise ValueError("Tail-posture head_source is unsupported.")
    if type(identity["keypoint_count"]) is not int or type(identity["angle_count"]) is not int:
        raise ValueError("Tail-posture scientific dimensions must be exact integers.")
    refined_source = identity["source_refined_subject_masks_run"]
    if refined_source is not None and (
        type(refined_source) is not str or not refined_source or "/" in refined_source
    ):
        raise ValueError("Tail-posture refined-mask source identity is invalid.")
    return _strict_json_copy(identity)


def _schema_summary(
    root: Any,
    *,
    run_name: str,
    role: str,
) -> dict[str, object]:
    parent, run = _require_run(root, run_name)
    dimensions = _dimensions(run)
    issues = validate_tail_posture_view_arrays(run, dimensions=dimensions)
    if issues:
        detail = "; ".join(
            f"{issue.code}:{issue.path}:{issue.message}" for issue in issues
        )
        raise ValueError(f"Tail-posture {role} exact schema failed: {detail}.")
    if set(run.array_keys()) != {
        declaration.path for declaration in TAIL_POSTURE_VIEW_ARRAY_DECLARATIONS
    }:
        raise ValueError("Tail-posture direct array inventory is not exactly ten arrays.")
    if run.attrs.get("schema_id") != TAIL_POSTURE_VIEW_RUN_SCHEMA_ID or run.attrs.get(
        "schema_version"
    ) != TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION:
        raise ValueError("Tail-posture run schema identity is unsupported.")
    if run.attrs.get("view_family") != VIEW_FAMILY:
        raise ValueError("Tail-posture benchmark requires the Megabouts-compatible family.")
    if not is_run_complete_in_parent(parent, run, legacy_default=False):
        raise ValueError(f"Tail-posture {role} run is not strictly complete.")
    expected_eligible = role == "source"
    if is_run_selector_eligible(run) is not expected_eligible or run.attrs.get(
        "stage_selector_eligible"
    ) is not expected_eligible:
        raise ValueError(f"Tail-posture {role} eligibility is not literal {expected_eligible}.")

    persisted_manifest = run.attrs.get(TAIL_POSTURE_VIEW_ARRAY_SCHEMA_ATTR)
    persisted_digest = run.attrs.get(TAIL_POSTURE_VIEW_ARRAY_SCHEMA_DIGEST_ATTR)
    if not isinstance(persisted_manifest, Mapping) or not _is_sha256(persisted_digest):
        raise ValueError("Tail-posture array schema manifest binding is absent.")
    expected_adoption = role == "candidate"
    if persisted_manifest.get("byte_planner_adopted") is not expected_adoption:
        raise ValueError("Tail-posture role disagrees with byte-planner adoption.")

    storage_plan_sha256: str | None = None
    receipt = run.attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
    if role == "source":
        if (
            receipt is not None
            or run.attrs.get(ANALYSIS_STORAGE_PLAN_DIGEST_ATTR) is not None
            or run.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR) is not None
            or run.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR) is not None
        ):
            raise ValueError("Tail-posture source must not claim a candidate storage receipt.")
    else:
        if not isinstance(receipt, Mapping):
            raise ValueError("Tail-posture candidate storage receipt is absent.")
        reconstructed = analysis_storage_plan_receipt_from_manifest(receipt)
        if len(reconstructed.entries) != 10:
            raise ValueError("Tail-posture candidate plan must contain exactly ten arrays.")
        if (
            reconstructed.profile.as_manifest() != PUBLISHED_HTTP_V1.as_manifest()
            or reconstructed.profile.profile_id != "published_http_v1"
            or run.attrs.get(ANALYSIS_STORAGE_PROFILE_ID_ATTR) != "published_http_v1"
            or run.attrs.get(ANALYSIS_STORAGE_PROFILE_ROLE_ATTR)
            != ANALYSIS_STORAGE_PROFILE_ROLE
        ):
            raise ValueError(
                "Tail-posture candidate must bind exact published_http_v1 profile/role."
            )
        expected_declarations = {
            declaration.path: declaration.as_manifest()
            for declaration in TAIL_POSTURE_VIEW_CANDIDATE_ARRAY_DECLARATIONS
        }
        observed_declarations = {
            entry.declaration.path: entry.declaration.as_manifest()
            for entry in reconstructed.entries
        }
        if observed_declarations != expected_declarations:
            raise ValueError(
                "Tail-posture candidate receipt declarations differ from exact v3 candidates."
            )
        storage_plan_sha256 = str(receipt.get("payload_digest"))
        if (
            not _is_sha256(storage_plan_sha256)
            or run.attrs.get(ANALYSIS_STORAGE_PLAN_DIGEST_ATTR) != storage_plan_sha256
        ):
            raise ValueError("Tail-posture candidate storage-plan digest binding is invalid.")

    run_path = _run_path(run_name)
    if role == "source":
        _selected, selected_name, selected_path, selected_publication = (
            megabouts_inputs._resolve_tail_posture_view_run(  # noqa: SLF001
                root,
                run_name,
                view_family=VIEW_FAMILY,
            )
        )
        if selected_name != run_name or selected_path != run_path:
            raise ValueError("Maintained private source resolver returned a different run.")
        publication = tail_publication.load_tail_posture_coordinate_publication(
            root,
            run_path,
        )
        if publication.manifest.record_sha256 != selected_publication.manifest.record_sha256:
            raise ValueError("Public coordinate validator and private resolver disagree.")
        adapter = SOURCE_ADAPTER
    else:
        # Candidate selection is intentionally unavailable.  This private
        # low-level verifier is used only to validate the completed ineligible
        # coordinate publication; it is not represented as a payload consumer.
        publication = tail_publication._load_tail_coordinate_publication(  # noqa: SLF001
            root,
            run_path,
            expected_selector_eligible=False,
            expected_kind="tail_posture_view",
            require_complete=True,
        )
        adapter = CANDIDATE_ADAPTER

    if publication.run_path != run_path or publication.kind != "tail_posture_view":
        raise ValueError("Tail-posture coordinate publication path/kind binding is invalid.")
    if publication.source.run_path != str(run.attrs.get("source_subject_shape_path")):
        raise ValueError("Tail-posture coordinate source path differs from the run attrs.")
    source_manifest = publication.source.manifest.record_sha256
    if source_manifest != run.attrs.get(
        "source_subject_shape_publication_manifest_sha256"
    ):
        raise ValueError("Tail-posture coordinate source manifest binding is stale.")

    scientific_identity = _scientific_identity(run)
    semantic_validation = _semantic_validation(run, dimensions)
    return {
        "role": role,
        "run_name": run_name,
        "run_path": run_path,
        "adapter": adapter,
        "schema_id": TAIL_POSTURE_VIEW_RUN_SCHEMA_ID,
        "schema_version": TAIL_POSTURE_VIEW_RUN_SCHEMA_VERSION,
        "dimensions": dimensions.contract_dimensions,
        "array_count": 10,
        "array_schema_sha256": persisted_digest,
        "storage_plan_sha256": storage_plan_sha256,
        "coordinate_manifest_ref": publication.manifest.record_ref,
        "coordinate_manifest_sha256": publication.manifest.record_sha256,
        "source_subject_shape_run": publication.source.run_path,
        "source_subject_shape_manifest_sha256": source_manifest,
        "completion_status": "complete",
        "selector_eligible": expected_eligible,
        "scientific_identity": scientific_identity,
        "semantic_validation": semantic_validation,
        "logical_arrays": _logical_array_inventory(run),
    }


def _consumer_boundary() -> dict[str, object]:
    return {
        "source_payload_adapter": SOURCE_ADAPTER,
        "candidate_payload_adapter": CANDIDATE_ADAPTER,
        "source_selection_evidence": PUBLIC_SOURCE_SELECTION_EVIDENCE,
        "source_coordinate_evidence": PUBLIC_SOURCE_COORDINATE_EVIDENCE,
        "candidate_coordinate_evidence": (
            "diagnostic_private_ineligible_coordinate_publication_verifier"
        ),
        "standalone_public_payload_reader": False,
        "full_megabouts_input_pack_consumer": BROADER_CONSUMER_STATUS,
    }


def validate_pair(
    archive: str | Path,
    *,
    source_run: str,
    candidate_run: str,
) -> dict[str, Any]:
    archive_path = _safe_archive(archive)
    source_name = _safe_name(source_run, label="source_run")
    candidate_name = _safe_name(candidate_run, label="candidate_run")
    if source_name == candidate_name:
        raise ValueError("Source and candidate must be distinct immutable runs.")
    _guard_relative_tree(archive_path, PARENT_PATH, label="tail-posture parent")
    root = _open_primary(archive_path)
    parent = root[PARENT_PATH]
    selectors = _selector_snapshot(parent)
    _require_expected_selectors(selectors, source_name)
    if any(item.get("value") == candidate_name for item in selectors.values()):
        raise ValueError("Selector-ineligible candidate is selected by a family pointer.")
    metadata = validate_direct_consolidated_subtree(
        archive_path,
        subtree_path=PARENT_PATH,
    ).to_json()
    source = _schema_summary(root, run_name=source_name, role="source")
    candidate = _schema_summary(root, run_name=candidate_name, role="candidate")

    source_identity = {
        field: source[field]
        for field in (
            "dimensions",
            "source_subject_shape_run",
            "source_subject_shape_manifest_sha256",
            "scientific_identity",
            "semantic_validation",
        )
    }
    candidate_identity = {
        field: candidate[field]
        for field in (
            "dimensions",
            "source_subject_shape_run",
            "source_subject_shape_manifest_sha256",
            "scientific_identity",
            "semantic_validation",
        )
    }
    if source_identity != candidate_identity:
        raise ValueError("Source/candidate logical input identity differs.")
    source_arrays = source["logical_arrays"]
    candidate_arrays = candidate["logical_arrays"]
    if source_arrays != candidate_arrays:
        raise ValueError(
            "Source/candidate arrays are not bit-exact, dtype/shape exact, and NaN-safe equal."
        )
    logical_equality = {
        "array_count": 10,
        "comparison": "dtype_shape_and_contiguous_bits_sha256_v1",
        "nan_policy": "identical_ieee_payload_bits_required",
        "all_equal": True,
        "array_digests_sha256": canonical_json_sha256(source_arrays),
    }
    payload = {
        "family_id": FAMILY_ID,
        "archive_identity": _archive_identity(archive_path),
        "source_run": source_name,
        "candidate_run": candidate_name,
        "selectors": selectors,
        "metadata_equivalence": metadata,
        "source": source,
        "candidate": candidate,
        "logical_equality": logical_equality,
        "consumer_boundary": _consumer_boundary(),
        "publication_mode": PUBLICATION_MODE,
        "promotion_authorized": False,
    }
    return _envelope(PAIR_SCHEMA_ID, payload)


def _window_start(*, seed: int, path: str, ordinal: int, n_rows: int, length: int) -> int:
    if n_rows <= length:
        return 0
    digest = hashlib.sha256(f"{seed}:{path}:{ordinal}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (n_rows - length + 1)


def build_workload(
    pair_validation: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    windows_per_array: int = DEFAULT_WINDOWS_PER_ARRAY,
) -> dict[str, Any]:
    pair = _require_envelope(pair_validation, schema_id=PAIR_SCHEMA_ID)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be one exact nonnegative integer.")
    if type(window_rows) is not int or window_rows <= 0:
        raise ValueError("window_rows must be one exact positive integer.")
    if type(windows_per_array) is not int or windows_per_array <= 0:
        raise ValueError("windows_per_array must be one exact positive integer.")
    arrays = pair["source"]["logical_arrays"]
    operations: list[dict[str, object]] = []
    for array in arrays:
        path = str(array["path"])
        shape = tuple(int(value) for value in array["shape"])
        n_rows = shape[0]
        operations.append(
            {
                "path": path,
                "mode": "eager_complete_array",
                "start": 0,
                "stop": n_rows,
                "expected_dtype": array["dtype"],
                "expected_shape": list(shape),
            }
        )
        length = min(window_rows, n_rows)
        for ordinal in range(windows_per_array):
            start = _window_start(
                seed=seed,
                path=path,
                ordinal=ordinal,
                n_rows=n_rows,
                length=length,
            )
            operations.append(
                {
                    "path": path,
                    "mode": "windowed_rows",
                    "start": start,
                    "stop": start + length,
                    "expected_dtype": array["dtype"],
                    "expected_shape": [length, *shape[1:]],
                }
            )
    payload = {
        "workload_id": "tail_posture_view_v3_all_arrays_eager_and_windowed_v1",
        "family_id": FAMILY_ID,
        "source_run": pair["source_run"],
        "candidate_run": pair["candidate_run"],
        "pair_payload_digest": pair_validation["payload_digest"],
        "seed": seed,
        "window_rows": window_rows,
        "windows_per_array": windows_per_array,
        "array_count": 10,
        "operation_count": len(operations),
        "operations": operations,
    }
    return _envelope(WORKLOAD_SCHEMA_ID, payload)


def _validate_workload(
    value: Mapping[str, Any],
    *,
    pair_validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _require_envelope(value, schema_id=WORKLOAD_SCHEMA_ID)
    expected = build_workload(
        pair_validation,
        seed=payload.get("seed"),
        window_rows=payload.get("window_rows"),
        windows_per_array=payload.get("windows_per_array"),
    )
    if dict(value) != expected:
        raise ValueError("Workload differs from deterministic live pair replay.")
    if payload.get("operation_count") != 10 * (1 + payload["windows_per_array"]):
        raise ValueError("Workload operation count does not cover every array exactly.")
    return payload


def _read_receipts(run: Any, workload: Mapping[str, Any]) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for operation_index, operation in enumerate(workload["operations"]):
        path = str(operation["path"])
        array = run[path]
        start = int(operation["start"])
        stop = int(operation["stop"])
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        values = np.asarray(array[start:stop, ...])
        wall_seconds = float(time.perf_counter() - started_wall)
        cpu_seconds = float(time.process_time() - started_cpu)
        expected_shape = tuple(int(value) for value in operation["expected_shape"])
        if str(values.dtype) != operation["expected_dtype"] or values.shape != expected_shape:
            raise ValueError(f"Read result for {path} differs from workload dtype/shape.")
        receipts.append(
            {
                "operation_index": operation_index,
                "path": path,
                "mode": operation["mode"],
                "start": start,
                "stop": stop,
                "dtype": str(values.dtype),
                "shape": [int(value) for value in values.shape],
                "element_count": int(values.size),
                "decoded_bytes": int(values.nbytes),
                "bits_sha256": _array_bits_digest(values),
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
            }
        )
    return receipts


def _logical_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in receipt.items()
            if key not in {"wall_seconds", "cpu_seconds"}
        }
        for receipt in receipts
    ]


def _physical_io_null() -> dict[str, object]:
    return {
        "available": False,
        "read_operations": None,
        "bytes_transferred": None,
        "range_reads": None,
        "cache_hits": None,
        "reason": PHYSICAL_IO_REASON,
    }


def _trial_order(repetition_index: int) -> tuple[str, str]:
    if type(repetition_index) is not int or repetition_index < 0:
        raise ValueError("repetition_index must be one exact nonnegative integer.")
    return (
        ("source", "candidate")
        if repetition_index % 2 == 0
        else ("candidate", "source")
    )


def _environment() -> dict[str, object]:
    git = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    return {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "palette_commit": git.get("commit_hash"),
        "palette_dirty": git.get("is_dirty"),
        "thread_environment": {
            key: os.environ.get(key) for key in STORAGE_BENCHMARK_THREAD_ENVIRONMENT
        },
        "cache_state": "fresh_child_os_cache_uncontrolled",
    }


def run_trial(
    archive: str | Path,
    *,
    source_run: str,
    candidate_run: str,
    role: str,
    repetition_index: int,
    order_position: int,
    driver_process_id: int,
    workload: Mapping[str, Any],
) -> dict[str, Any]:
    archive_path = _safe_archive(archive)
    if role not in {"source", "candidate"}:
        raise ValueError("role must be source or candidate.")
    order = _trial_order(repetition_index)
    if order_position not in {0, 1} or order[order_position] != role:
        raise ValueError("Trial role/order rotation binding is invalid.")
    if (
        type(driver_process_id) is not int
        or driver_process_id <= 0
        or driver_process_id != os.getppid()
        or driver_process_id == os.getpid()
    ):
        raise ValueError("Trial driver PID must equal the live parent process.")
    pair = validate_pair(
        archive_path,
        source_run=source_run,
        candidate_run=candidate_run,
    )
    workload_payload = _validate_workload(workload, pair_validation=pair)
    pair_payload = _require_envelope(pair, schema_id=PAIR_SCHEMA_ID)
    root = _open_primary(archive_path)
    parent = root[PARENT_PATH]
    before = _selector_snapshot(parent)
    _require_expected_selectors(before, source_run)
    started_at = utc_now()
    name = source_run if role == "source" else candidate_run
    run = root[_run_path(name)]
    receipts = _read_receipts(run, workload_payload)
    result = {
        "role": role,
        "run_name": name,
        "adapter": SOURCE_ADAPTER if role == "source" else CANDIDATE_ADAPTER,
        "operation_count": len(receipts),
        "decoded_bytes": sum(int(item["decoded_bytes"]) for item in receipts),
        "read_receipts": receipts,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    after = _selector_snapshot(parent)
    if after != before:
        raise ValueError("Tail-posture selectors changed during a read-only trial.")
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_identity": pair_payload["archive_identity"],
        "source_run": source_run,
        "candidate_run": candidate_run,
        "role": role,
        "repetition_index": repetition_index,
        "order_position": order_position,
        "driver_process_id": driver_process_id,
        "selectors_before": before,
        "selectors_after": after,
        "pair_validation": pair,
        "workload": workload,
        "result": result,
        "environment": _environment(),
        "physical_io": _physical_io_null(),
        "consumer_boundary": _consumer_boundary(),
        "promotion_authorized": False,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
    }
    return _envelope(TRIAL_SCHEMA_ID, payload)


def _require_nonnegative_number(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be one finite nonnegative number.")


def validate_trial_evidence(
    value: Mapping[str, Any],
    *,
    archive: str | Path,
    source_run: str,
    candidate_run: str,
) -> None:
    payload = _require_envelope(value, schema_id=TRIAL_SCHEMA_ID)
    if set(payload) != _TRIAL_FIELDS:
        raise ValueError("Tail-posture trial payload field set is not exact.")
    if payload["benchmark_id"] != BENCHMARK_ID or payload["family_id"] != FAMILY_ID:
        raise ValueError("Tail-posture trial identity is unsupported.")
    if payload["source_run"] != source_run or payload["candidate_run"] != candidate_run:
        raise ValueError("Tail-posture trial run identity differs from the request.")
    role = payload["role"]
    repetition_index = payload["repetition_index"]
    order_position = payload["order_position"]
    if (
        role not in {"source", "candidate"}
        or type(repetition_index) is not int
        or repetition_index < 0
        or order_position not in {0, 1}
        or _trial_order(repetition_index)[order_position] != role
    ):
        raise ValueError("Tail-posture trial role/order binding is invalid.")
    if (
        type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
    ):
        raise ValueError("Tail-posture trial driver PID binding is invalid.")
    if payload["physical_io"] != _physical_io_null():
        raise ValueError("Trial falsely claims physical I/O telemetry.")
    if payload["consumer_boundary"] != _consumer_boundary():
        raise ValueError("Trial consumer-boundary claim is inaccurate.")
    if payload["promotion_authorized"] is not False:
        raise ValueError("Tail-posture benchmark cannot authorize promotion.")
    claimed_pair = _require_envelope(
        payload["pair_validation"],
        schema_id=PAIR_SCHEMA_ID,
    )
    if set(claimed_pair) != _PAIR_FIELDS:
        raise ValueError("Tail-posture pair payload field set is not exact.")
    for summary_role in ("source", "candidate"):
        if set(claimed_pair[summary_role]) != _ROLE_FIELDS:
            raise ValueError(
                f"Tail-posture {summary_role} summary field set is not exact."
            )

    live_pair = validate_pair(archive, source_run=source_run, candidate_run=candidate_run)
    if payload["pair_validation"] != live_pair:
        raise ValueError("Trial pair claim differs from live schema/storage/selector replay.")

    workload_payload = _validate_workload(
        payload["workload"],
        pair_validation=live_pair,
    )
    archive_path = _safe_archive(archive)
    root = _open_primary(archive_path)
    live_selectors = _selector_snapshot(root[PARENT_PATH])
    _require_expected_selectors(live_selectors, source_run)
    if payload["selectors_before"] != live_selectors or payload["selectors_after"] != live_selectors:
        raise ValueError("Trial selector receipts differ from the live frozen selectors.")
    if payload["archive_identity"] != _archive_identity(archive_path):
        raise ValueError("Trial archive identity differs from the live archive.")
    role_result = payload["result"]
    if not isinstance(role_result, Mapping) or set(role_result) != {
        "role",
        "run_name",
        "adapter",
        "operation_count",
        "decoded_bytes",
        "read_receipts",
        "peak_rss_bytes",
    }:
        raise ValueError("Tail-posture role result field set is not exact.")
    expected_name = source_run if role == "source" else candidate_run
    expected_adapter = SOURCE_ADAPTER if role == "source" else CANDIDATE_ADAPTER
    if (
        role_result["role"] != role
        or role_result["run_name"] != expected_name
        or role_result["adapter"] != expected_adapter
    ):
        raise ValueError("Tail-posture role result is bound to the wrong adapter/run.")
    receipts = role_result["read_receipts"]
    if not isinstance(receipts, list) or len(receipts) != workload_payload["operation_count"]:
        raise ValueError("Tail-posture role receipt count is incomplete.")
    if role_result["operation_count"] != len(receipts):
        raise ValueError("Tail-posture role operation count is inconsistent.")
    for receipt in receipts:
        if set(receipt) != {
            "operation_index",
            "path",
            "mode",
            "start",
            "stop",
            "dtype",
            "shape",
            "element_count",
            "decoded_bytes",
            "bits_sha256",
            "wall_seconds",
            "cpu_seconds",
        }:
            raise ValueError("Tail-posture read receipt field set is not exact.")
        _require_nonnegative_number(receipt["wall_seconds"], label="read wall_seconds")
        _require_nonnegative_number(receipt["cpu_seconds"], label="read cpu_seconds")
        if not _is_sha256(receipt["bits_sha256"]):
            raise ValueError("Tail-posture read digest is not SHA-256.")
    expected_logical = _logical_receipts(
        _read_receipts(root[_run_path(expected_name)], workload_payload)
    )
    observed_logical = _logical_receipts(receipts)
    if observed_logical != expected_logical:
        raise ValueError("Tail-posture read receipts differ from live workload replay.")
    if role_result["decoded_bytes"] != sum(
        int(item["decoded_bytes"]) for item in receipts
    ):
        raise ValueError("Tail-posture decoded-byte total is inconsistent.")
    if type(role_result["peak_rss_bytes"]) is not int or role_result["peak_rss_bytes"] <= 0:
        raise ValueError("Tail-posture peak RSS must be a positive integer.")

    environment = payload["environment"]
    if not isinstance(environment, Mapping) or set(environment) != {
        "pid",
        "hostname",
        "system",
        "release",
        "python",
        "numpy",
        "zarr",
        "palette_commit",
        "palette_dirty",
        "thread_environment",
        "cache_state",
    }:
        raise ValueError("Tail-posture trial environment field set is not exact.")
    if type(environment["pid"]) is not int or environment["pid"] <= 0:
        raise ValueError("Tail-posture trial PID is invalid.")
    if payload["driver_process_id"] == environment["pid"]:
        raise ValueError("Tail-posture child PID must differ from its driver PID.")
    if environment["thread_environment"] != dict(STORAGE_BENCHMARK_THREAD_ENVIRONMENT):
        raise ValueError("Tail-posture child thread environment was not pinned.")
    if environment["cache_state"] != "fresh_child_os_cache_uncontrolled":
        raise ValueError("Tail-posture cache-state claim is unsupported.")


def _role_runtime_summary(trials: Sequence[Mapping[str, Any]], role: str) -> dict[str, float]:
    wall_totals: list[float] = []
    decoded: list[int] = []
    rss: list[int] = []
    for trial in trials:
        payload = _require_envelope(trial, schema_id=TRIAL_SCHEMA_ID)
        if payload["role"] != role:
            continue
        result = payload["result"]
        wall_totals.append(sum(float(item["wall_seconds"]) for item in result["read_receipts"]))
        decoded.append(int(result["decoded_bytes"]))
        rss.append(int(result["peak_rss_bytes"]))
    return {
        "median_read_wall_seconds": float(statistics.median(wall_totals)),
        "median_decoded_bytes": float(statistics.median(decoded)),
        "max_peak_rss_bytes": float(max(rss)),
    }


def _require_utc_timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one UTC timestamp string.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must carry an explicit UTC offset.")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must use UTC.")
    return parsed


def run_benchmark_matrix(
    archive: str | Path,
    *,
    source_run: str,
    candidate_run: str,
    output_root: str | Path,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    windows_per_array: int = DEFAULT_WINDOWS_PER_ARRAY,
) -> dict[str, Any]:
    archive_path = _safe_archive(archive)
    source_name = _safe_name(source_run, label="source_run")
    candidate_name = _safe_name(candidate_run, label="candidate_run")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be one exact positive integer.")
    output = _safe_output_root(output_root, archive=archive_path)
    output.mkdir(parents=False, exist_ok=False)
    pair = validate_pair(
        archive_path,
        source_run=source_name,
        candidate_run=candidate_name,
    )
    workload = build_workload(
        pair,
        seed=seed,
        window_rows=window_rows,
        windows_per_array=windows_per_array,
    )
    pair_path = output / "pair_validation.json"
    workload_path = output / "workload.json"
    _write_json(pair_path, pair)
    _write_json(workload_path, workload)
    trials: list[dict[str, Any]] = []
    pids: set[int] = set()
    driver_process_id = os.getpid()
    environment = os.environ.copy()
    environment.update(STORAGE_BENCHMARK_THREAD_ENVIRONMENT)
    for repetition in range(repetitions):
        repetition_trials: dict[str, Mapping[str, Any]] = {}
        for order_position, role in enumerate(_trial_order(repetition)):
            trial_path = output / f"trial_{repetition:03d}_{order_position}_{role}.json"
            command = [
                sys.executable,
                "-m",
                "fisheye.diagnostics.benchmark_tail_posture_view_v3_candidate",
                "--trial",
                "--archive",
                str(archive_path),
                "--source-run",
                source_name,
                "--candidate-run",
                candidate_name,
                "--role",
                role,
                "--repetition-index",
                str(repetition),
                "--order-position",
                str(order_position),
                "--driver-process-id",
                str(driver_process_id),
                "--workload-json",
                str(workload_path),
                "--output-root",
                str(output),
                "--trial-output",
                str(trial_path),
            ]
            subprocess.run(command, check=True, env=environment)
            trial = _read_json(trial_path)
            validate_trial_evidence(
                trial,
                archive=archive_path,
                source_run=source_name,
                candidate_run=candidate_name,
            )
            trial_payload = _require_envelope(trial, schema_id=TRIAL_SCHEMA_ID)
            pid = int(trial_payload["environment"]["pid"])
            if pid in pids:
                raise ValueError("Balanced trials must execute in distinct fresh processes.")
            pids.add(pid)
            repetition_trials[role] = trial_payload
            trials.append(trial)
        source_logical = _logical_receipts(
            repetition_trials["source"]["result"]["read_receipts"]
        )
        candidate_logical = _logical_receipts(
            repetition_trials["candidate"]["result"]["read_receipts"]
        )
        if source_logical != candidate_logical:
            raise ValueError("Balanced source/candidate trial payloads differ.")
    final_pair = validate_pair(
        archive_path,
        source_run=source_name,
        candidate_run=candidate_name,
    )
    if final_pair != pair:
        raise ValueError("Tail-posture archive changed during the benchmark matrix.")
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": FAMILY_ID,
        "archive_identity": _archive_identity(archive_path),
        "source_run": source_name,
        "candidate_run": candidate_name,
        "repetitions": repetitions,
        "driver_process_id": driver_process_id,
        "balanced_order": True,
        "fresh_process_count": len(pids),
        "pair_validation": pair,
        "workload": workload,
        "trials": trials,
        "source_summary": _role_runtime_summary(trials, "source"),
        "candidate_summary": _role_runtime_summary(trials, "candidate"),
        "physical_io": _physical_io_null(),
        "consumer_boundary": _consumer_boundary(),
        "publication_mode": PUBLICATION_MODE,
        "promotion_policy": PROMOTION_POLICY,
        "promotion_authorized": False,
        "selectors_unchanged": True,
        "completed_at_utc": utc_now(),
    }
    matrix = _envelope(MATRIX_SCHEMA_ID, payload)
    validate_matrix_evidence(
        matrix,
        archive=archive_path,
        source_run=source_name,
        candidate_run=candidate_name,
    )
    _write_json(output / "matrix.json", matrix)
    return matrix


def validate_matrix_evidence(
    value: Mapping[str, Any],
    *,
    archive: str | Path,
    source_run: str,
    candidate_run: str,
) -> None:
    payload = _require_envelope(value, schema_id=MATRIX_SCHEMA_ID)
    expected_fields = {
        "benchmark_id",
        "family_id",
        "archive_identity",
        "source_run",
        "candidate_run",
        "repetitions",
        "driver_process_id",
        "balanced_order",
        "fresh_process_count",
        "pair_validation",
        "workload",
        "trials",
        "source_summary",
        "candidate_summary",
        "physical_io",
        "consumer_boundary",
        "publication_mode",
        "promotion_policy",
        "promotion_authorized",
        "selectors_unchanged",
        "completed_at_utc",
    }
    if set(payload) != expected_fields:
        raise ValueError("Tail-posture matrix payload field set is not exact.")
    if type(payload["repetitions"]) is not int or payload["repetitions"] < 1:
        raise ValueError("Tail-posture matrix repetitions must be one positive integer.")
    _require_utc_timestamp(
        payload["completed_at_utc"],
        label="Tail-posture matrix completed_at_utc",
    )
    live_pair = validate_pair(archive, source_run=source_run, candidate_run=candidate_run)
    if payload["pair_validation"] != live_pair:
        raise ValueError("Tail-posture matrix pair differs from live replay.")
    _validate_workload(payload["workload"], pair_validation=live_pair)
    trials = payload["trials"]
    if not isinstance(trials, list) or len(trials) != 2 * payload["repetitions"]:
        raise ValueError("Tail-posture matrix trial inventory is incomplete.")
    pids: set[int] = set()
    for repetition in range(payload["repetitions"]):
        pair_trials = trials[2 * repetition : 2 * repetition + 2]
        pair_payloads: dict[str, Mapping[str, Any]] = {}
        for order_position, trial in enumerate(pair_trials):
            validate_trial_evidence(
                trial,
                archive=archive,
                source_run=source_run,
                candidate_run=candidate_run,
            )
            trial_payload = _require_envelope(trial, schema_id=TRIAL_SCHEMA_ID)
            expected_role = _trial_order(repetition)[order_position]
            if (
                trial_payload["role"] != expected_role
                or trial_payload["repetition_index"] != repetition
                or trial_payload["order_position"] != order_position
                or trial_payload["driver_process_id"] != payload["driver_process_id"]
            ):
                raise ValueError("Tail-posture trial rotation is not balanced/deterministic.")
            pair_payloads[expected_role] = trial_payload
            pids.add(int(trial_payload["environment"]["pid"]))
        if _logical_receipts(pair_payloads["source"]["result"]["read_receipts"]) != (
            _logical_receipts(pair_payloads["candidate"]["result"]["read_receipts"])
        ):
            raise ValueError("Tail-posture balanced trial pair is not logically equal.")
    if len(pids) != len(trials) or payload["fresh_process_count"] != len(pids):
        raise ValueError("Tail-posture matrix does not prove fresh child processes.")
    if payload["source_summary"] != _role_runtime_summary(trials, "source") or payload[
        "candidate_summary"
    ] != _role_runtime_summary(trials, "candidate"):
        raise ValueError("Tail-posture matrix aggregates differ from raw trials.")
    if (
        payload["benchmark_id"] != BENCHMARK_ID
        or payload["family_id"] != FAMILY_ID
        or payload["source_run"] != source_run
        or payload["candidate_run"] != candidate_run
        or type(payload["driver_process_id"]) is not int
        or payload["driver_process_id"] <= 0
        or payload["archive_identity"] != _archive_identity(_safe_archive(archive))
        or payload["balanced_order"] is not True
        or payload["physical_io"] != _physical_io_null()
        or payload["consumer_boundary"] != _consumer_boundary()
        or payload["publication_mode"] != PUBLICATION_MODE
        or payload["promotion_policy"] != PROMOTION_POLICY
        or payload["promotion_authorized"] is not False
        or payload["selectors_unchanged"] is not True
    ):
        raise ValueError("Tail-posture matrix contains a false identity/consumer/I/O/promotion claim.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS)
    parser.add_argument("--windows-per-array", type=int, default=DEFAULT_WINDOWS_PER_ARRAY)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--role", choices=("source", "candidate"))
    parser.add_argument("--repetition-index", type=int)
    parser.add_argument("--order-position", type=int)
    parser.add_argument("--driver-process-id", type=int)
    parser.add_argument("--workload-json", type=Path)
    parser.add_argument("--trial-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.trial:
        if any(
            value is None
            for value in (
                args.role,
                args.repetition_index,
                args.order_position,
                args.driver_process_id,
                args.workload_json,
                args.trial_output,
            )
        ):
            raise ValueError(
                "Trial mode requires role/order/process, workload-json, and trial-output."
            )
        archive = _safe_archive(args.archive)
        output_root = _safe_existing_output_root(args.output_root)
        trial_output = _safe_trial_output(args.trial_output, output_root=output_root)
        workload = _read_json(
            _safe_evidence_input(
                args.workload_json,
                output_root=output_root,
                label="Trial workload",
            )
        )
        trial = run_trial(
            archive,
            source_run=args.source_run,
            candidate_run=args.candidate_run,
            role=args.role,
            repetition_index=args.repetition_index,
            order_position=args.order_position,
            driver_process_id=args.driver_process_id,
            workload=workload,
        )
        _write_json(trial_output, trial)
        return 0
    matrix = run_benchmark_matrix(
        args.archive,
        source_run=args.source_run,
        candidate_run=args.candidate_run,
        output_root=args.output_root,
        repetitions=args.repetitions,
        seed=args.seed,
        window_rows=args.window_rows,
        windows_per_array=args.windows_per_array,
    )
    print(json.dumps(matrix, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_ID",
    "build_workload",
    "run_benchmark_matrix",
    "run_trial",
    "validate_matrix_evidence",
    "validate_pair",
    "validate_trial_evidence",
]
