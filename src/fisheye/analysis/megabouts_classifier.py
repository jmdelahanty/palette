"""Optional Megabouts bout-classifier execution over Palette-derived windows."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.analysis.bout_classification_runs import (
    BOUT_CLASSIFICATION_SCHEMA_ID,
    BOUT_CLASSIFICATION_SCHEMA_VERSION,
    PER_BOUT_SCHEMA_ID,
)
from fisheye.analysis.megabouts_classifier_inputs import (
    DEFAULT_BOUT_DURATION_S,
    DEFAULT_ALIGN_TRAJ_TO_ONSET,
    DEFAULT_HEADING_SOURCE,
    DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    DEFAULT_MIN_TAIL_VALID_FRACTION,
    DEFAULT_MIN_TRAJ_VALID_FRACTION,
    DEFAULT_TRAJ_REFERENCE_INDEX,
    MegaboutsClassifierInputPack,
    build_megabouts_classifier_input_pack,
    summarize_input_pack,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from fisheye.shared.system_metadata import get_environment_info, get_git_info
from fisheye.shared.zarr_io import open_zarr_root

SCHEMA_ID = BOUT_CLASSIFICATION_SCHEMA_ID
SCHEMA_VERSION = BOUT_CLASSIFICATION_SCHEMA_VERSION
ADAPTER_METHOD = "palette_megabouts_direct_classifier"
ADAPTER_METHOD_VERSION = 1
CLASSIFIER_FAMILY = "megabouts"
CLASSIFIER_NAME = "megabouts_transformer"
SOURCE_MODE = "palette_bouts"
INVALID_WINDOW_POLICY = "skip_invalid_windows"
PALETTE_PREPARED_INPUT_MODE = "palette_prepared_fixed_windows"
MEGABOUTS_PREPROCESSED_INPUT_MODE = "megabouts_preprocessed_full_timeseries"
CATEGORY_LABEL_BYTES_WIDTH = 64
FAILURE_REASON_BYTES_WIDTH = 128


@dataclass(frozen=True)
class MegaboutsRuntime:
    """Resolved optional Megabouts runtime objects."""

    classifier_class: object
    tracking_config_class: object
    segmentation_config_class: object
    category_names: tuple[str, ...]
    package_version: str
    package_path: str
    source_repo: Optional[str]
    git_commit: Optional[str]


@dataclass(frozen=True)
class MegaboutsClassificationResult:
    """Megabouts classification outputs for the valid-window subset."""

    classified_indices: np.ndarray
    classif_results: Mapping[str, np.ndarray]
    runtime: Optional[MegaboutsRuntime]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"megabouts_classifier_{stamp}"


_json_safe = json_attr_safe


def _resolve_megabouts_repo(megabouts_repo: Optional[str | Path]) -> Optional[Path]:
    raw = megabouts_repo if megabouts_repo is not None else os.environ.get("MEGABOUTS_REPO")
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(raw).expanduser().resolve()
    if not (path / "megabouts").is_dir():
        raise ValueError(
            f"Megabouts repo {path} does not contain a top-level 'megabouts' package directory."
        )
    return path


def _git_commit_for_repo(repo_path: Optional[Path]) -> Optional[str]:
    if repo_path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _load_megabouts_runtime(megabouts_repo: Optional[str | Path] = None) -> MegaboutsRuntime:
    repo_path = _resolve_megabouts_repo(megabouts_repo)
    if repo_path is not None and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import megabouts
        from megabouts.classification import BoutClassifier
        from megabouts.config.segmentation_config import TailSegmentationConfig
        from megabouts.tracking_data import TrackingConfig
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise RuntimeError(
            "Megabouts is required for classifier execution but is not importable. "
            "Install/configure Megabouts outside Palette, or run "
            "`fisheye.analysis.megabouts_classifier_inputs` for a dependency-free dry run."
        ) from exc

    try:
        from megabouts.utils.plots_utils import bouts_category_name

        category_names = tuple(str(value) for value in bouts_category_name)
    except Exception:  # pragma: no cover - optional label helper
        category_names = ()

    package_path = str(Path(getattr(megabouts, "__file__", "") or "").resolve())
    return MegaboutsRuntime(
        classifier_class=BoutClassifier,
        tracking_config_class=TrackingConfig,
        segmentation_config_class=TailSegmentationConfig,
        category_names=category_names,
        package_version=str(getattr(megabouts, "__version__", "unknown")),
        package_path=package_path,
        source_repo=None if repo_path is None else str(repo_path),
        git_commit=_git_commit_for_repo(repo_path),
    )


def _runtime_attrs(runtime: Optional[MegaboutsRuntime]) -> dict[str, object]:
    if runtime is None:
        return {
            "classifier_version": None,
            "megabouts_package_version": None,
            "megabouts_package_path": None,
            "megabouts_category_labels": [],
        }
    return {
        "classifier_version": runtime.package_version,
        "megabouts_package_version": runtime.package_version,
        "megabouts_package_path": runtime.package_path,
        "megabouts_source_repo": runtime.source_repo,
        "megabouts_git_commit": runtime.git_commit,
        "megabouts_category_labels": list(runtime.category_names),
    }


def _resolve_fps(pack: MegaboutsClassifierInputPack) -> int:
    fps = float(pack.parameters.get("fps", 0.0))
    rounded = int(round(fps))
    if rounded <= 0 or not math.isclose(fps, float(rounded), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "Megabouts TrackingConfig requires integer fps in [20, 700]; "
            f"Palette resolved fps={fps!r}."
        )
    if not 20 <= rounded <= 700:
        raise ValueError(
            "Megabouts TrackingConfig requires integer fps in [20, 700]; "
            f"Palette resolved fps={rounded}."
        )
    return rounded


def classify_megabouts_input_pack(
    pack: MegaboutsClassifierInputPack,
    *,
    exclude_cs: bool = False,
    device: str = "auto",
    megabouts_repo: Optional[str | Path] = None,
    runtime: Optional[MegaboutsRuntime] = None,
) -> MegaboutsClassificationResult:
    """Run Megabouts on valid source windows only.

    Invalid Palette windows are intentionally excluded from the Megabouts call.
    They are still represented in the persisted result table as skipped rows.
    """

    classified_indices = np.flatnonzero(np.asarray(pack.valid_bout, dtype=bool))
    if classified_indices.size == 0:
        return MegaboutsClassificationResult(
            classified_indices=classified_indices.astype(np.int64, copy=False),
            classif_results={
                "cat": np.asarray([], dtype=np.int32),
                "subcat": np.asarray([], dtype=np.int32),
                "sign": np.asarray([], dtype=np.int32),
                "proba": np.asarray([], dtype=np.float32),
                "first_half_beat": np.asarray([], dtype=np.int32),
            },
            runtime=runtime,
        )

    resolved_runtime = runtime if runtime is not None else _load_megabouts_runtime(megabouts_repo)
    fps = _resolve_fps(pack)
    window_frames = int(pack.tail_array.shape[2])
    # Add a tiny epsilon because Megabouts converts milliseconds with int(),
    # and we need the segmentation mask length to match our fixed window.
    bout_duration_ms = (float(window_frames) + 1e-6) * 1000.0 / float(fps)
    tracking_cfg = resolved_runtime.tracking_config_class(fps=fps, tracking="full_tracking")
    segmentation_cfg = resolved_runtime.segmentation_config_class(
        fps=fps,
        bout_duration_ms=bout_duration_ms,
    )
    if int(segmentation_cfg.bout_duration) != window_frames:
        raise ValueError(
            "Megabouts segmentation config duration does not match Palette classifier window: "
            f"{segmentation_cfg.bout_duration} != {window_frames}."
        )

    device_obj = None
    if str(device) != "auto":
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional external package path
            raise RuntimeError("A Megabouts device was requested but torch is not importable.") from exc
        device_obj = torch.device(str(device))

    classifier = resolved_runtime.classifier_class(
        tracking_cfg,
        segmentation_cfg,
        exclude_CS=bool(exclude_cs),
        device=device_obj,
    )
    classif_results = classifier.run_classification(
        tail_array=pack.tail_array[classified_indices],
        traj_array=pack.traj_array[classified_indices],
    )
    normalized_results = {
        "cat": np.asarray(classif_results["cat"], dtype=np.int32),
        "subcat": np.asarray(classif_results["subcat"], dtype=np.int32),
        "sign": np.asarray(classif_results["sign"], dtype=np.int32),
        "proba": np.asarray(classif_results["proba"], dtype=np.float32),
        "first_half_beat": np.asarray(classif_results["first_half_beat"], dtype=np.int32),
    }
    expected = int(classified_indices.size)
    for name, values in normalized_results.items():
        if int(values.shape[0]) != expected:
            raise ValueError(
                f"Megabouts returned {values.shape[0]} values for {name!r}; expected {expected}."
            )
    return MegaboutsClassificationResult(
        classified_indices=classified_indices.astype(np.int64, copy=False),
        classif_results=normalized_results,
        runtime=resolved_runtime,
    )


def _category_label(category_id: int, category_names: Sequence[str]) -> str:
    if 0 <= int(category_id) < len(category_names):
        return str(category_names[int(category_id)])
    if int(category_id) < 0:
        return "skipped_invalid_window"
    return f"category_{int(category_id)}"


def _as_bytes(value: object, *, width: int) -> bytes:
    return str(value or "").encode("utf-8", errors="replace")[: int(width)]


def build_per_bout_classification_table(
    pack: MegaboutsClassifierInputPack,
    result: MegaboutsClassificationResult,
) -> np.ndarray:
    """Build a row-aligned classification table for every source bout."""

    n_bouts = int(pack.source_bout_id.shape[0])
    dtype = np.dtype(
        [
            ("source_bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("window_start_frame", "i8"),
            ("window_end_frame", "i8"),
            ("HB1_frame", "i8"),
            ("HB1_offset_frames", "i4"),
            ("category_id", "i4"),
            ("category_label_bytes", f"S{CATEGORY_LABEL_BYTES_WIDTH}"),
            ("subcategory_id", "i4"),
            ("sign", "i4"),
            ("probability", "f4"),
            ("tail_valid_fraction", "f4"),
            ("traj_valid_fraction", "f4"),
            ("max_consecutive_tail_invalid", "i4"),
            ("max_consecutive_traj_invalid", "i4"),
            ("source_window_valid", "?"),
            ("classified", "?"),
            ("valid", "?"),
            ("failure_reason_bytes", f"S{FAILURE_REASON_BYTES_WIDTH}"),
        ]
    )
    table = np.zeros((n_bouts,), dtype=dtype)
    table["source_bout_id"] = np.asarray(pack.source_bout_id, dtype=np.int64)
    table["start_frame"] = np.asarray(pack.source_start_frame, dtype=np.int64)
    table["end_frame"] = np.asarray(pack.source_end_frame, dtype=np.int64)
    table["window_start_frame"] = np.asarray(pack.window_start_frame, dtype=np.int64)
    table["window_end_frame"] = np.asarray(pack.window_end_frame, dtype=np.int64)
    table["HB1_frame"] = -1
    table["HB1_offset_frames"] = -1
    table["category_id"] = -1
    table["category_label_bytes"] = _as_bytes("skipped_invalid_window", width=CATEGORY_LABEL_BYTES_WIDTH)
    table["subcategory_id"] = -1
    table["sign"] = 0
    table["probability"] = np.nan
    table["tail_valid_fraction"] = np.asarray(pack.tail_valid_fraction, dtype=np.float32)
    table["traj_valid_fraction"] = np.asarray(pack.traj_valid_fraction, dtype=np.float32)
    table["max_consecutive_tail_invalid"] = np.asarray(pack.max_consecutive_tail_invalid, dtype=np.int32)
    table["max_consecutive_traj_invalid"] = np.asarray(pack.max_consecutive_traj_invalid, dtype=np.int32)
    source_valid = np.asarray(pack.valid_bout, dtype=bool)
    table["source_window_valid"] = source_valid
    table["classified"] = False
    table["valid"] = False
    for idx, reason in enumerate(np.asarray(pack.failure_reason, dtype=object).tolist()):
        table["failure_reason_bytes"][idx] = _as_bytes(reason, width=FAILURE_REASON_BYTES_WIDTH)

    classified_indices = np.asarray(result.classified_indices, dtype=np.int64)
    if classified_indices.size == 0:
        return table

    category = np.asarray(result.classif_results["cat"], dtype=np.int32)
    subcategory = np.asarray(result.classif_results["subcat"], dtype=np.int32)
    sign = np.asarray(result.classif_results["sign"], dtype=np.int32)
    proba = np.asarray(result.classif_results["proba"], dtype=np.float32)
    hb1_offset = np.asarray(result.classif_results["first_half_beat"], dtype=np.int32)
    category_names = () if result.runtime is None else result.runtime.category_names

    table["category_id"][classified_indices] = category
    table["subcategory_id"][classified_indices] = subcategory
    table["sign"][classified_indices] = sign
    table["probability"][classified_indices] = proba
    table["HB1_offset_frames"][classified_indices] = hb1_offset
    table["HB1_frame"][classified_indices] = (
        np.asarray(pack.window_start_frame, dtype=np.int64)[classified_indices]
        + hb1_offset.astype(np.int64)
    )
    table["classified"][classified_indices] = True
    table["valid"][classified_indices] = True
    for source_idx, cat_id in zip(classified_indices.tolist(), category.tolist()):
        table["category_label_bytes"][source_idx] = _as_bytes(
            _category_label(int(cat_id), category_names),
            width=CATEGORY_LABEL_BYTES_WIDTH,
        )
        table["failure_reason_bytes"][source_idx] = _as_bytes("ok", width=FAILURE_REASON_BYTES_WIDTH)

    return table


def _resolve_parent(root: zarr.Group) -> zarr.Group:
    analysis = root["analysis"] if "analysis" in root else root.create_group("analysis")
    return require_runs_parent(analysis, "bout_classification_runs")


def write_megabouts_classification_run(
    root: zarr.Group,
    *,
    run_name: Optional[str],
    pack: MegaboutsClassifierInputPack,
    result: MegaboutsClassificationResult,
    overwrite: bool = False,
    exclude_cs: bool = False,
    command: Optional[str] = None,
) -> str:
    """Persist Megabouts classifier results without mutating source runs."""

    parent = _resolve_parent(root)
    resolved_run_name = str(run_name or _default_run_name())
    if resolved_run_name in parent:
        if not overwrite:
            raise ValueError(
                f"Bout classification run {resolved_run_name!r} already exists. "
                "Use --overwrite or choose another --run-name."
            )
        del parent[resolved_run_name]

    run_group = parent.create_group(resolved_run_name)
    mark_run_started(run_group, run_name=resolved_run_name, stage="bout_classification")
    created_at_utc = _utc_now()
    runtime_attrs = _runtime_attrs(result.runtime)
    table = build_per_bout_classification_table(pack, result)
    source_refs = dict(pack.source_refs)
    classifier_input_mode = str(pack.parameters.get("classifier_input_mode") or PALETTE_PREPARED_INPUT_MODE)
    megabouts_preprocessing = bool(pack.parameters.get("megabouts_preprocessing", False))
    megabouts_segmentation = bool(pack.parameters.get("megabouts_segmentation", False))
    source_fps = float(pack.parameters.get("fps", math.nan))
    window_frames = int(pack.tail_array.shape[2])
    window_duration_s = float(pack.parameters.get("bout_duration_s", math.nan))
    parameters = {
        **dict(pack.parameters),
        "adapter_method": ADAPTER_METHOD,
        "adapter_method_version": ADAPTER_METHOD_VERSION,
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_name": CLASSIFIER_NAME,
        "classifier_input_mode": classifier_input_mode,
        "megabouts_preprocessing": megabouts_preprocessing,
        "megabouts_segmentation": megabouts_segmentation,
        "source_fps": source_fps,
        "window_duration_s": window_duration_s,
        "window_frames": window_frames,
        "megabouts_time_sampling": True,
        "source_mode": SOURCE_MODE,
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "exclude_capture_swims": bool(exclude_cs),
        "calls_megabouts": bool(result.classified_indices.size > 0),
        "classified_bout_count": int(result.classified_indices.size),
        "source_bout_count": int(table.shape[0]),
        "valid_source_window_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_source_window_count": int(table.shape[0] - np.count_nonzero(pack.valid_bout)),
    }
    tail_angle_conversion = {
        "source_array": source_refs.get("tail_angle_rad"),
        "source_valid_array": source_refs.get("tail_valid"),
        "convention": "megabouts_cumulative_segment_angle",
        "channels": int(pack.tail_array.shape[1]),
        "units": "radians",
    }
    trajectory_conversion = {
        "source_positions_array": source_refs.get("positions_mm"),
        "source_heading_array": source_refs.get("heading"),
        "source_valid_array": source_refs.get("sample_valid"),
        "channels": ["x_mm", "y_mm", "heading_radians"],
        "alignment": pack.parameters.get("traj_alignment"),
        "reference_index": pack.parameters.get("traj_reference_index"),
        "heading_reference": "classifier_window_reference_sample",
    }
    invalid_frame_policy = {
        "policy": INVALID_WINDOW_POLICY,
        "min_tail_valid_fraction": pack.parameters.get("min_tail_valid_fraction"),
        "min_traj_valid_fraction": pack.parameters.get("min_traj_valid_fraction"),
        "max_consecutive_invalid_frames": pack.parameters.get("max_consecutive_invalid_frames"),
        "requires_traj_reference_valid": pack.parameters.get("requires_traj_reference_valid"),
    }
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": ADAPTER_METHOD,
        "method_version": ADAPTER_METHOD_VERSION,
        "adapter_method": ADAPTER_METHOD,
        "adapter_method_version": ADAPTER_METHOD_VERSION,
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_name": CLASSIFIER_NAME,
        "classifier_input_mode": classifier_input_mode,
        "megabouts_preprocessing": megabouts_preprocessing,
        "megabouts_segmentation": megabouts_segmentation,
        "source_fps": source_fps,
        "window_duration_s": window_duration_s,
        "window_frames": window_frames,
        "megabouts_time_sampling": True,
        "source_mode": SOURCE_MODE,
        "row_axis": "swim_bout_rows",
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "tail_angle_conversion": _json_safe(tail_angle_conversion),
        "trajectory_conversion": _json_safe(trajectory_conversion),
        "invalid_frame_policy": _json_safe(invalid_frame_policy),
        "source_refs": _json_safe(source_refs),
        "parameters": _json_safe(parameters),
        "source_bout_count": int(table.shape[0]),
        "valid_source_window_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_source_window_count": int(table.shape[0] - np.count_nonzero(pack.valid_bout)),
        "classified_bout_count": int(result.classified_indices.size),
        **runtime_attrs,
    }
    for key, value in attrs.items():
        run_group.attrs[key] = _json_safe(value)

    per_bout = write_columnar_dataset(
        run_group,
        "per_bout",
        table,
        attrs={
            "schema_id": PER_BOUT_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "storage_semantics": "one row per source swim-bout row",
            "invalid_window_policy": INVALID_WINDOW_POLICY,
            "category_label_encoding": "utf8-null-terminated",
            "category_label_bytes_width": CATEGORY_LABEL_BYTES_WIDTH,
            "failure_reason_encoding": "utf8-null-terminated",
            "failure_reason_bytes_width": FAILURE_REASON_BYTES_WIDTH,
        },
    )
    per_bout.attrs["source_swim_bout_path"] = source_refs.get("swim_bout_level")

    zarr_path = getattr(root, "_palette_fs_path", None)
    env_info = get_environment_info(
        disk_path=str(zarr_path) if zarr_path is not None else None,
        capture_env_vars=False,
    )
    provenance = build_stage_provenance(
        stage="bout_classification",
        created_at_utc=created_at_utc,
        parameters=_json_safe(parameters),
        inputs=_json_safe(source_refs),
        command=command,
        version=str(ADAPTER_METHOD_VERSION),
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "run_path": f"analysis/bout_classification_runs/{resolved_run_name}",
            "per_bout_path": f"analysis/bout_classification_runs/{resolved_run_name}/per_bout",
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="bout_classification_run")
    mark_run_complete(
        run_group,
        parent_group=parent,
        run_name=resolved_run_name,
        run_provenance=build_run_provenance_from_stage_record(provenance),
    )
    return resolved_run_name


def run_megabouts_classifier(
    zarr_path: str | Path,
    *,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    tail_posture_view_run: str = "latest",
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    swim_bout_run: str = "latest",
    speed_level: str = "default",
    heading_source: str = DEFAULT_HEADING_SOURCE,
    bout_duration_s: float = DEFAULT_BOUT_DURATION_S,
    bout_duration_frames: Optional[int] = None,
    min_tail_valid_fraction: float = DEFAULT_MIN_TAIL_VALID_FRACTION,
    min_traj_valid_fraction: float = DEFAULT_MIN_TRAJ_VALID_FRACTION,
    max_consecutive_invalid_frames: int = DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    align_traj_to_onset: bool = DEFAULT_ALIGN_TRAJ_TO_ONSET,
    traj_reference_index: int = DEFAULT_TRAJ_REFERENCE_INDEX,
    exclude_cs: bool = False,
    device: str = "auto",
    megabouts_repo: Optional[str | Path] = None,
    classifier_input_mode: str = PALETTE_PREPARED_INPUT_MODE,
    dry_run: bool = False,
    command: Optional[str] = None,
) -> dict[str, object]:
    """Run or dry-run the optional Megabouts classifier adapter."""

    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    mode = str(classifier_input_mode or PALETTE_PREPARED_INPUT_MODE)
    pack_kwargs = {
        "tail_posture_view_run": tail_posture_view_run,
        "track_kinematics_run": track_kinematics_run,
        "track_scope": track_scope,
        "track_id": track_id,
        "swim_bout_run": swim_bout_run,
        "speed_level": speed_level,
        "heading_source": heading_source,
        "bout_duration_s": bout_duration_s,
        "bout_duration_frames": bout_duration_frames,
        "min_tail_valid_fraction": min_tail_valid_fraction,
        "min_traj_valid_fraction": min_traj_valid_fraction,
        "max_consecutive_invalid_frames": max_consecutive_invalid_frames,
        "align_traj_to_onset": align_traj_to_onset,
        "traj_reference_index": traj_reference_index,
    }
    if mode == PALETTE_PREPARED_INPUT_MODE:
        pack = build_megabouts_classifier_input_pack(root, **pack_kwargs)
    elif mode == MEGABOUTS_PREPROCESSED_INPUT_MODE:
        from fisheye.analysis.megabouts_preprocessing_comparison import (
            build_megabouts_preprocessed_input_pack,
        )

        pack = build_megabouts_preprocessed_input_pack(
            root,
            megabouts_repo=megabouts_repo,
            **pack_kwargs,
        )
    else:
        raise ValueError(
            "Unsupported classifier_input_mode "
            f"{classifier_input_mode!r}; expected {PALETTE_PREPARED_INPUT_MODE!r} "
            f"or {MEGABOUTS_PREPROCESSED_INPUT_MODE!r}."
        )
    if dry_run:
        summary = summarize_input_pack(pack)
        summary_parameters = dict(summary.get("parameters", {}))
        summary_parameters.update(
            {
                "adapter_method": ADAPTER_METHOD,
                "adapter_method_version": ADAPTER_METHOD_VERSION,
                "classifier_family": CLASSIFIER_FAMILY,
                "classifier_name": CLASSIFIER_NAME,
                "classifier_input_mode": mode,
                "megabouts_preprocessing": bool(pack.parameters.get("megabouts_preprocessing", False)),
                "megabouts_segmentation": bool(pack.parameters.get("megabouts_segmentation", False)),
                "source_fps": float(pack.parameters.get("fps", math.nan)),
                "window_duration_s": float(pack.parameters.get("bout_duration_s", math.nan)),
                "window_frames": int(pack.tail_array.shape[2]),
                "megabouts_time_sampling": True,
                "calls_megabouts_classifier": False,
            }
        )
        summary["parameters"] = summary_parameters
        summary["would_write_run_family"] = "analysis/bout_classification_runs"
        summary["adapter_method"] = ADAPTER_METHOD
        summary["classifier_input_mode"] = mode
        summary["calls_megabouts_preprocessing"] = bool(pack.parameters.get("megabouts_preprocessing", False))
        summary["calls_megabouts_classifier"] = False
        return summary

    result = classify_megabouts_input_pack(
        pack,
        exclude_cs=exclude_cs,
        device=device,
        megabouts_repo=megabouts_repo,
    )
    resolved_run_name = write_megabouts_classification_run(
        root,
        run_name=run_name,
        pack=pack,
        result=result,
        overwrite=overwrite,
        exclude_cs=exclude_cs,
        command=command,
    )
    n_bouts = int(pack.valid_bout.shape[0])
    summary = {
        "status": "ok",
        "run_name": resolved_run_name,
        "run_path": f"analysis/bout_classification_runs/{resolved_run_name}",
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "adapter_method": ADAPTER_METHOD,
        "classifier_input_mode": mode,
        "source_bout_count": n_bouts,
        "valid_source_window_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_source_window_count": int(n_bouts - np.count_nonzero(pack.valid_bout)),
        "classified_bout_count": int(result.classified_indices.shape[0]),
        "invalid_window_policy": INVALID_WINDOW_POLICY,
        "source_refs": pack.source_refs,
        "parameters": {
            **pack.parameters,
            "adapter_method": ADAPTER_METHOD,
            "adapter_method_version": ADAPTER_METHOD_VERSION,
            "calls_megabouts": True,
            "classifier_family": CLASSIFIER_FAMILY,
            "classifier_name": CLASSIFIER_NAME,
            "exclude_capture_swims": bool(exclude_cs),
            "device": str(device),
            "megabouts_repo": None if megabouts_repo is None else str(megabouts_repo),
            "align_traj_to_onset": bool(align_traj_to_onset),
            "traj_reference_index": int(traj_reference_index),
        },
        **_runtime_attrs(result.runtime),
    }
    return dict(_json_safe(summary))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run optional Megabouts bout classification over Palette swim-bout windows."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument("--run-name", default=None, help="Output analysis/bout_classification_runs/<run> name.")
    parser.add_argument("--overwrite", action="store_true", help="Replace --run-name if it already exists.")
    parser.add_argument("--tail-posture-view-run", default="latest")
    parser.add_argument("--track-kinematics-run", default="latest")
    parser.add_argument("--track-scope", default="offline")
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument("--swim-bout-run", default="latest")
    parser.add_argument("--speed-level", default="default")
    parser.add_argument("--heading-source", default=DEFAULT_HEADING_SOURCE)
    parser.add_argument("--bout-duration-s", type=float, default=DEFAULT_BOUT_DURATION_S)
    parser.add_argument("--bout-duration-frames", type=int, default=None)
    parser.add_argument("--min-tail-valid-fraction", type=float, default=DEFAULT_MIN_TAIL_VALID_FRACTION)
    parser.add_argument("--min-traj-valid-fraction", type=float, default=DEFAULT_MIN_TRAJ_VALID_FRACTION)
    parser.add_argument("--max-consecutive-invalid-frames", type=int, default=DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES)
    parser.add_argument(
        "--no-align-traj-to-onset",
        action="store_true",
        help="Disable Megabouts-style onset-frame translation/rotation for trajectory windows.",
    )
    parser.add_argument("--traj-reference-index", type=int, default=DEFAULT_TRAJ_REFERENCE_INDEX)
    parser.add_argument("--exclude-CS", action="store_true", help="Pass exclude_CS=True to Megabouts.")
    parser.add_argument("--device", default="auto", help="Megabouts torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--megabouts-repo",
        default=None,
        help="Optional local Megabouts checkout to add to sys.path without installing it. Also supports MEGABOUTS_REPO.",
    )
    parser.add_argument(
        "--classifier-input-mode",
        default=PALETTE_PREPARED_INPUT_MODE,
        choices=[PALETTE_PREPARED_INPUT_MODE, MEGABOUTS_PREPROCESSED_INPUT_MODE],
        help=(
            "Input pack mode. The Megabouts-preprocessed mode runs Megabouts "
            "preprocessing before classification and records megabouts_preprocessing=true."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build inputs and report eligibility without writing. In palette-prepared mode this "
            "does not import Megabouts; in Megabouts-preprocessed mode it imports preprocessing."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]) if argv is None else None
    summary = run_megabouts_classifier(
        args.zarr_path,
        run_name=args.run_name,
        overwrite=bool(args.overwrite),
        tail_posture_view_run=args.tail_posture_view_run,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        track_id=int(args.track_id),
        swim_bout_run=args.swim_bout_run,
        speed_level=args.speed_level,
        heading_source=args.heading_source,
        bout_duration_s=float(args.bout_duration_s),
        bout_duration_frames=args.bout_duration_frames,
        min_tail_valid_fraction=float(args.min_tail_valid_fraction),
        min_traj_valid_fraction=float(args.min_traj_valid_fraction),
        max_consecutive_invalid_frames=int(args.max_consecutive_invalid_frames),
        align_traj_to_onset=not bool(args.no_align_traj_to_onset),
        traj_reference_index=int(args.traj_reference_index),
        exclude_cs=bool(args.exclude_CS),
        device=str(args.device),
        megabouts_repo=args.megabouts_repo,
        classifier_input_mode=str(args.classifier_input_mode),
        dry_run=bool(args.dry_run),
        command=command,
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
