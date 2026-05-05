"""Export cross-recording Palette analytics tables to Parquet.

The exporter treats Zarr archives as the source of truth and writes derived,
regenerable Parquet parts for cohort/population-level analysis. Extraction is
parallelized by recording. Final Parquet/manifest writes are coordinated in the
parent process so workers never append to the same file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.system import get_git_info
from fisheye.utils.zarr_io import open_zarr_root


EXPORT_SCHEMA_VERSION = 1
DEFAULT_TABLES = (
    "recording_summary",
    "stimulus_steps",
    "stimulus_step_summary",
    "stimulus_response_per_fish_step",
    "swim_bout_metrics",
)


@dataclass(frozen=True)
class StepSpan:
    step_index: int
    stimulus_mode: str | None
    step_name: str | None
    start_frame: int | None
    end_frame: int | None


@dataclass
class SourceExportResult:
    zarr_path: str
    recording_id: str
    rows_by_table: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


def _utc_now_id() -> str:
    # Prefix avoids eager hive-partition readers treating compact UTC stamps as
    # dates with incompatible parser assumptions.
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _recording_id_from_path(zarr_path: Path) -> str:
    name = zarr_path.name
    if name.endswith(".zarr"):
        name = name[:-5]
    if name.endswith("_analysis"):
        name = name[:-9]
    return name


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _scalar_for_parquet(value: Any) -> Any:
    """Convert NumPy/Zarr values into strict scalar Parquet-friendly values."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).rstrip(b"\x00").decode("utf-8", errors="ignore")
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_for_parquet(value.item())
        return _json_dumps_safe(value.tolist())
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return _json_dumps_safe(value)
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).rstrip(b"\x00").decode("utf-8", errors="ignore")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_dumps_safe(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = _json_dumps_safe(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _group_names(group: Any) -> list[str]:
    if hasattr(group, "group_keys"):
        return sorted(str(name) for name in group.group_keys())
    return sorted(str(name) for name in group.keys() if hasattr(group[name], "attrs"))


def _array_names(group: Any) -> list[str]:
    if hasattr(group, "array_keys"):
        return sorted(str(name) for name in group.array_keys())
    names: list[str] = []
    for name in group.keys():
        try:
            item = group[name]
            if not hasattr(item, "attrs") or not hasattr(item, "keys"):
                names.append(str(name))
        except Exception:
            continue
    return sorted(names)


def _has_child(group: Any, name: str) -> bool:
    try:
        return name in group
    except Exception:
        return False


def _attrs_dict(group: Any) -> dict[str, Any]:
    try:
        return {str(key): _scalar_for_parquet(value) for key, value in dict(group.attrs).items()}
    except Exception:
        return {}


def _row_count_from_group(group: Any, names: Sequence[str] | None = None) -> int:
    candidates = list(names) if names is not None else _array_names(group)
    for name in candidates:
        if not _has_child(group, name):
            continue
        try:
            arr = group[name]
            shape = tuple(arr.shape)
            if shape:
                return int(shape[0])
        except Exception:
            continue
    return 0


def _read_1d_array(group: Any, name: str) -> np.ndarray | None:
    if not _has_child(group, name):
        return None
    try:
        arr = np.asarray(group[name][:])
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def _read_table_rows(
    group: Any,
    *,
    include_arrays: Sequence[str] | None = None,
    exclude_arrays: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Read scalar 1D arrays from a Zarr group into row dictionaries."""

    names = list(include_arrays) if include_arrays is not None else _array_names(group)
    excluded = {str(name) for name in exclude_arrays}
    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    for name in names:
        if name in excluded:
            continue
        arr = _read_1d_array(group, name)
        if arr is None:
            continue
        if n_rows is None:
            n_rows = int(arr.shape[0])
        if int(arr.shape[0]) != n_rows:
            continue
        arrays[name] = arr

    if not arrays or n_rows is None:
        return []

    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        rows.append({name: _scalar_for_parquet(arr[idx]) for name, arr in arrays.items()})
    return rows


def _latest_run(root: Any, parent_path: str, requested: str | None = None) -> tuple[Any | None, str | None, str | None]:
    try:
        group, name = resolve_zarr_run(
            root,
            parent_path,
            run_name=requested,
            fallback_to_latest=True,
            fallback_to_sorted="last",
        )
        return group, name, None
    except Exception as exc:
        return None, None, str(exc)


def _common_row(
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    table: str,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": table,
        "recording_id": recording_id,
        "zarr_path": str(zarr_path),
        "source_lineage_hash": _hash_payload(lineage),
    }


def _summarize_numeric(values: np.ndarray | None, op: str) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if op == "sum":
        return float(np.sum(arr))
    if op == "mean":
        return float(np.mean(arr))
    if op == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported summary op: {op}")


def _load_stimulus_steps(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> tuple[str | None, list[StepSpan], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    spans: list[StepSpan] = []
    stim_group, stim_run, error = _latest_run(root, "analysis/stimulus_runs")
    if stim_group is None or stim_run is None:
        diagnostics.append({"table": "stimulus_steps", "status": "skipped", "reason": error})
        return None, spans, rows

    if not _has_child(stim_group, "steps"):
        diagnostics.append({"table": "stimulus_steps", "status": "skipped", "reason": "missing steps group"})
        return stim_run, spans, rows

    steps_group = stim_group["steps"]
    step_names = sorted(
        (name for name in _group_names(steps_group) if re.fullmatch(r"step_\d+", name)),
        key=lambda item: int(item.split("_")[1]),
    )

    for name in step_names:
        step_group = steps_group[name]
        attrs = _attrs_dict(step_group)
        idx = _safe_int(attrs.get("step_index"))
        if idx is None:
            idx = int(name.split("_")[1])
        start_frame = _safe_int(attrs.get("start_frame"))
        if start_frame is None:
            start_frame = _safe_int(attrs.get("start_camera_frame"))
        end_frame = _safe_int(attrs.get("end_frame"))
        if end_frame is None:
            end_frame = _safe_int(attrs.get("end_camera_frame"))
        stimulus_mode = attrs.get("stimulus_mode")
        step_name = attrs.get("step_name") or attrs.get("name")
        spans.append(StepSpan(
            step_index=idx,
            stimulus_mode=str(stimulus_mode) if stimulus_mode is not None else None,
            step_name=str(step_name) if step_name is not None else None,
            start_frame=start_frame,
            end_frame=end_frame,
        ))

        if "stimulus_steps" not in tables:
            continue

        lineage = {
            "zarr_path": str(zarr_path),
            "stimulus_run": stim_run,
            "step_index": idx,
            "stimulus_run_schema_version": stim_group.attrs.get("schema_version"),
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table="stimulus_steps",
            lineage=lineage,
        )
        row.update({
            "stimulus_run": stim_run,
            "step_index": idx,
            "step_group": name,
            "step_name": step_name,
            "stimulus_mode": stimulus_mode,
            "stimulus_mode_id": attrs.get("stimulus_mode_id"),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_camera_frame": _safe_int(attrs.get("start_camera_frame")) or start_frame,
            "end_camera_frame": _safe_int(attrs.get("end_camera_frame")) or end_frame,
            "duration_s": _safe_float(attrs.get("duration_s")),
            "stimulus_params_json": attrs.get("stimulus_params") or attrs.get("raw_protocol_params_json"),
        })
        if isinstance(row["stimulus_params_json"], str):
            # Already serialized by _attrs_dict when attrs contained a dict/list.
            pass
        elif row["stimulus_params_json"] is not None:
            row["stimulus_params_json"] = _json_dumps_safe(row["stimulus_params_json"])

        for child_name in ("moving_grating", "concentric_grating", "looming_dot"):
            if not _has_child(step_group, child_name):
                continue
            prefix = child_name
            for key, value in _attrs_dict(step_group[child_name]).items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)

    return stim_run, spans, rows


def _assign_step(start_frame: int | None, end_frame: int | None, spans: Sequence[StepSpan]) -> StepSpan | None:
    if start_frame is None and end_frame is None:
        return None
    best: tuple[int, StepSpan] | None = None
    bout_start = start_frame if start_frame is not None else end_frame
    bout_end = end_frame if end_frame is not None else start_frame
    if bout_start is None or bout_end is None:
        return None
    for span in spans:
        if span.start_frame is None or span.end_frame is None:
            continue
        overlap = max(0, min(bout_end, span.end_frame) - max(bout_start, span.start_frame))
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, span)
    if best is not None:
        return best[1]
    for span in spans:
        if span.start_frame is not None and span.end_frame is not None and span.start_frame <= bout_start < span.end_frame:
            return span
    return None


def _load_recording_summary(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    stimulus_run: str | None,
    step_count: int,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if "recording_summary" not in tables:
        return []

    stim_resp_group, stim_resp_run, stim_resp_error = _latest_run(root, "analysis/stimulus_response_runs")
    swim_group, swim_run, _swim_error = _latest_run(root, "analysis/swim_bout_runs")

    lineage = {
        "zarr_path": str(zarr_path),
        "stimulus_run": stimulus_run,
        "stimulus_response_run": stim_resp_run,
        "swim_bout_run": swim_run,
    }
    row = _common_row(
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        table="recording_summary",
        lineage=lineage,
    )
    row.update({
        "stimulus_run": stimulus_run,
        "stimulus_response_run": stim_resp_run,
        "swim_bout_run": swim_run,
        "stimulus_step_count": step_count,
    })

    if stim_resp_group is not None:
        attrs = _attrs_dict(stim_resp_group)
        row.update({
            "source_track_kinematics_run": attrs.get("source_track_kinematics_run"),
            "source_track_kinematics_type": attrs.get("source_track_kinematics_type"),
            "source_bout_run": attrs.get("source_bout_run"),
            "n_fish": attrs.get("n_fish"),
            "n_steps": attrs.get("n_steps"),
        })
        if _has_child(stim_resp_group, "global"):
            global_group = stim_resp_group["global"]
            fish_ids = _read_1d_array(global_group, "fish_id")
            row["global_fish_count"] = int(fish_ids.size) if fish_ids is not None else None
            row["total_distance_mm_sum"] = _summarize_numeric(_read_1d_array(global_group, "total_distance_mm"), "sum")
            row["mean_speed_mm_s_mean"] = _summarize_numeric(_read_1d_array(global_group, "mean_speed_mm_s"), "mean")
            row["fraction_moving_mean"] = _summarize_numeric(_read_1d_array(global_group, "fraction_moving"), "mean")
            row["total_active_s_sum"] = _summarize_numeric(_read_1d_array(global_group, "total_active_s"), "sum")
    elif stim_resp_error:
        diagnostics.append({"table": "recording_summary", "status": "partial", "reason": stim_resp_error})

    if swim_group is not None:
        default_level = str(swim_group.attrs.get("default_level", ""))
        row["swim_bout_default_level"] = default_level or None
        if default_level and _has_child(swim_group, default_level):
            level = swim_group[default_level]
            row["swim_bout_default_n_bouts"] = _safe_int(level.attrs.get("n_bouts"))
            row["swim_bout_default_mean_duration_s"] = _safe_float(level.attrs.get("mean_bout_duration_s"))
            row["swim_bout_default_total_path_length_mm"] = _safe_float(level.attrs.get("total_path_length_mm"))

    return [row]


def _load_stimulus_response_tables(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_summary_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    wanted = {"stimulus_step_summary", "stimulus_response_per_fish_step"} & tables
    if not wanted:
        return step_summary_rows, response_rows

    response_group, response_run, error = _latest_run(root, "analysis/stimulus_response_runs")
    if response_group is None or response_run is None:
        for table in wanted:
            diagnostics.append({"table": table, "status": "skipped", "reason": error})
        return step_summary_rows, response_rows

    if not _has_child(response_group, "steps"):
        for table in wanted:
            diagnostics.append({"table": table, "status": "skipped", "reason": "missing steps group"})
        return step_summary_rows, response_rows

    response_attrs = _attrs_dict(response_group)
    steps_group = response_group["steps"]
    step_names = sorted(
        (name for name in _group_names(steps_group) if re.fullmatch(r"step_\d+", name)),
        key=lambda item: int(item.split("_")[1]),
    )

    for name in step_names:
        step_group = steps_group[name]
        attrs = _attrs_dict(step_group)
        idx = _safe_int(attrs.get("step_index"))
        if idx is None:
            idx = int(name.split("_")[1])

        if not _has_child(step_group, "per_fish"):
            continue
        base_rows = _read_table_rows(step_group["per_fish"])
        for base in base_rows:
            fish_id = _safe_int(base.get("fish_id"))
            lineage = {
                "zarr_path": str(zarr_path),
                "stimulus_response_run": response_run,
                "source_stimulus_run": response_attrs.get("source_stimulus_run"),
                "source_track_kinematics_run": response_attrs.get("source_track_kinematics_run"),
                "source_bout_run": response_attrs.get("source_bout_run"),
                "step_index": idx,
                "fish_id": fish_id,
            }
            common = {
                "stimulus_response_run": response_run,
                "source_stimulus_run": response_attrs.get("source_stimulus_run"),
                "source_track_kinematics_run": response_attrs.get("source_track_kinematics_run"),
                "source_track_kinematics_type": response_attrs.get("source_track_kinematics_type"),
                "source_bout_run": response_attrs.get("source_bout_run"),
                "step_index": idx,
                "step_name": attrs.get("step_name"),
                "stimulus_mode": attrs.get("stimulus_mode"),
                "stimulus_mode_id": attrs.get("stimulus_mode_id"),
                "start_frame": _safe_int(attrs.get("start_frame")) or _safe_int(attrs.get("start_camera_frame")),
                "end_frame": _safe_int(attrs.get("end_frame")) or _safe_int(attrs.get("end_camera_frame")),
                "start_camera_frame": _safe_int(attrs.get("start_camera_frame")) or _safe_int(attrs.get("start_frame")),
                "end_camera_frame": _safe_int(attrs.get("end_camera_frame")) or _safe_int(attrs.get("end_frame")),
                "duration_s": _safe_float(attrs.get("duration_s")),
            }

            if "stimulus_step_summary" in tables:
                row = _common_row(
                    export_run_id=export_run_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    table="stimulus_step_summary",
                    lineage=lineage,
                )
                row.update(common)
                row.update(base)
                step_summary_rows.append(row)

            if "stimulus_response_per_fish_step" not in tables:
                continue

            row = _common_row(
                export_run_id=export_run_id,
                zarr_path=zarr_path,
                recording_id=recording_id,
                table="stimulus_response_per_fish_step",
                lineage=lineage,
            )
            row.update(common)
            row.update(base)
            row["omr_family"] = None

            if _has_child(step_group, "grating"):
                grating = step_group["grating"]
                if _has_child(grating, "per_fish"):
                    grating_rows = _read_table_rows(grating["per_fish"])
                    _merge_matching_fish_row(row, grating_rows, fish_id, prefix="grating_")
                if _has_child(grating, "omr"):
                    omr = grating["omr"]
                    row["omr_family"] = "moving_grating_omr"
                    for key, value in _attrs_dict(omr).items():
                        row[f"omr_attr_{key}"] = value
                    if _has_child(omr, "per_fish"):
                        omr_rows = _read_table_rows(omr["per_fish"])
                        _merge_matching_fish_row(row, omr_rows, fish_id, prefix="")

            if _has_child(step_group, "concentric_grating"):
                concentric = step_group["concentric_grating"]
                if _has_child(concentric, "per_fish"):
                    conc_rows = _read_table_rows(concentric["per_fish"])
                    _merge_matching_fish_row(row, conc_rows, fish_id, prefix="concentric_")
                if _has_child(concentric, "radial_omr"):
                    radial = concentric["radial_omr"]
                    row["omr_family"] = "concentric_radial_omr"
                    for key, value in _attrs_dict(radial).items():
                        row[f"radial_omr_attr_{key}"] = value
                    if _has_child(radial, "per_fish"):
                        radial_rows = _read_table_rows(radial["per_fish"])
                        _merge_matching_fish_row(row, radial_rows, fish_id, prefix="")

            response_rows.append(row)

    return step_summary_rows, response_rows


def _merge_matching_fish_row(
    row: dict[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    fish_id: int | None,
    *,
    prefix: str,
) -> None:
    for candidate in candidate_rows:
        if _safe_int(candidate.get("fish_id")) != fish_id:
            continue
        for key, value in candidate.items():
            if key == "fish_id":
                continue
            out_key = f"{prefix}{key}" if prefix else str(key)
            if out_key in row and row[out_key] == value:
                continue
            row[out_key] = value
        return


def _load_swim_bout_metrics(
    root: Any,
    *,
    export_run_id: str,
    zarr_path: Path,
    recording_id: str,
    stimulus_run: str | None,
    steps: Sequence[StepSpan],
    tables: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if "swim_bout_metrics" not in tables:
        return []

    swim_group, swim_run, error = _latest_run(root, "analysis/swim_bout_runs")
    if swim_group is None or swim_run is None:
        diagnostics.append({"table": "swim_bout_metrics", "status": "skipped", "reason": error})
        return []

    swim_attrs = _attrs_dict(swim_group)
    default_level = str(swim_attrs.get("default_level") or "speed_smoothed")
    if not _has_child(swim_group, default_level):
        diagnostics.append({
            "table": "swim_bout_metrics",
            "status": "skipped",
            "reason": f"missing default level {default_level!r}",
            "swim_bout_run": swim_run,
        })
        return []
    level_group = swim_group[default_level]
    if not _has_child(level_group, "bouts"):
        diagnostics.append({
            "table": "swim_bout_metrics",
            "status": "skipped",
            "reason": "missing bouts group",
            "swim_bout_run": swim_run,
            "speed_level": default_level,
        })
        return []

    level_attrs = _attrs_dict(level_group)
    bout_rows = _read_table_rows(level_group["bouts"])
    rows: list[dict[str, Any]] = []
    for bout in bout_rows:
        bout_id = _safe_int(bout.get("bout_id"))
        start_frame = _safe_int(bout.get("start_frame"))
        end_frame = _safe_int(bout.get("end_frame"))
        step = _assign_step(start_frame, end_frame, steps)
        lineage = {
            "zarr_path": str(zarr_path),
            "swim_bout_run": swim_run,
            "speed_level": default_level,
            "source_track_kinematics_run": swim_attrs.get("source_track_kinematics_run"),
            "track_id": swim_attrs.get("track_id"),
            "bout_id": bout_id,
        }
        row = _common_row(
            export_run_id=export_run_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            table="swim_bout_metrics",
            lineage=lineage,
        )
        row.update({
            "stimulus_run": stimulus_run,
            "swim_bout_run": swim_run,
            "source_track_kinematics_run": swim_attrs.get("source_track_kinematics_run"),
            "source_track_kinematics_type": swim_attrs.get("source_track_kinematics_type"),
            "track_id": _safe_int(swim_attrs.get("track_id")),
            "speed_level": default_level,
            "detection_method": swim_attrs.get("detection_method") or level_attrs.get("detection_method"),
            "detection_signal_transform_type": level_attrs.get("detection_signal_transform_type"),
            "detection_signal_source_level": level_attrs.get("detection_signal_source_level"),
            "movement_metric_source_level": level_attrs.get("movement_metric_source_level"),
            "threshold_mm_s": _safe_float(level_attrs.get("threshold_mm_s") or swim_attrs.get("threshold_mm_s")),
            "peak_prominence_mm_s": _safe_float(level_attrs.get("peak_prominence_mm_s") or swim_attrs.get("peak_prominence_mm_s")),
            "step_index": step.step_index if step else None,
            "step_name": step.step_name if step else None,
            "stimulus_mode": step.stimulus_mode if step else None,
        })
        row.update(bout)
        rows.append(row)
    return rows


def export_one_zarr(zarr_path: str | Path, *, tables: Sequence[str], export_run_id: str) -> SourceExportResult:
    zarr_path = Path(zarr_path).expanduser().resolve()
    recording_id = _recording_id_from_path(zarr_path)
    result = SourceExportResult(zarr_path=str(zarr_path), recording_id=recording_id)
    table_set = set(tables)

    try:
        root = open_zarr_root(zarr_path, mode="r")
    except Exception as exc:
        result.diagnostics.append({"table": "*", "status": "failed", "reason": f"open_failed: {exc}"})
        return result

    stimulus_run, steps, step_rows = _load_stimulus_steps(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if step_rows:
        result.rows_by_table.setdefault("stimulus_steps", []).extend(step_rows)

    recording_rows = _load_recording_summary(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        stimulus_run=stimulus_run,
        step_count=len(steps),
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if recording_rows:
        result.rows_by_table.setdefault("recording_summary", []).extend(recording_rows)

    step_summary_rows, response_rows = _load_stimulus_response_tables(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if step_summary_rows:
        result.rows_by_table.setdefault("stimulus_step_summary", []).extend(step_summary_rows)
    if response_rows:
        result.rows_by_table.setdefault("stimulus_response_per_fish_step", []).extend(response_rows)

    bout_rows = _load_swim_bout_metrics(
        root,
        export_run_id=export_run_id,
        zarr_path=zarr_path,
        recording_id=recording_id,
        stimulus_run=stimulus_run,
        steps=steps,
        tables=table_set,
        diagnostics=result.diagnostics,
    )
    if bout_rows:
        result.rows_by_table.setdefault("swim_bout_metrics", []).extend(bout_rows)

    for table in table_set:
        if table not in result.rows_by_table:
            result.rows_by_table[table] = []
    return result


def discover_analysis_zarrs(recordings_root: Path) -> list[Path]:
    recordings_root = recordings_root.expanduser().resolve()
    if not recordings_root.exists():
        raise FileNotFoundError(f"recordings root does not exist: {recordings_root}")
    return sorted(path for path in recordings_root.rglob("*_analysis.zarr") if path.is_dir())


def _parse_tables(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TABLES
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    else:
        raw = []
        for item in value:
            raw.extend(part.strip() for part in str(item).split(","))
    tables = tuple(item for item in raw if item)
    unknown = sorted(set(tables) - set(DEFAULT_TABLES))
    if unknown:
        expected = ", ".join(DEFAULT_TABLES)
        raise ValueError(f"Unknown table(s): {', '.join(unknown)}. Expected subset of: {expected}")
    return tables or DEFAULT_TABLES


def _collect_sources(args: argparse.Namespace) -> list[Path]:
    sources: list[Path] = []
    for path in args.zarr or []:
        sources.append(Path(path).expanduser().resolve())
    if args.recordings_root is not None:
        sources.extend(discover_analysis_zarrs(Path(args.recordings_root)))
    deduped: dict[str, Path] = {}
    for path in sources:
        deduped[str(path)] = path
    out = sorted(deduped.values())
    if args.limit is not None:
        out = out[: int(args.limit)]
    return out


def _normalize_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in columns} for row in rows]


def _infer_schema(rows: Sequence[Mapping[str, Any]]):
    import pyarrow as pa

    return pa.Table.from_pylist([dict(row) for row in rows]).schema


def _write_table_parts(
    *,
    output_root: Path,
    export_run_id: str,
    table: str,
    rows_by_source: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    overwrite: bool,
) -> tuple[int, list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_dir = output_root / "v1" / table / f"export_run_id={export_run_id}"
    if table_dir.exists() and any(table_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Export table directory already exists: {table_dir}")
    table_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[Mapping[str, Any]] = []
    for _source, rows in rows_by_source:
        all_rows.extend(rows)
    if not all_rows:
        return 0, []

    columns = sorted({key for row in all_rows for key in row.keys()})
    normalized_all = _normalize_rows(all_rows, columns)
    schema = _infer_schema(normalized_all)

    row_count = 0
    part_paths: list[str] = []
    part_index = 0
    for source_name, rows in rows_by_source:
        if not rows:
            continue
        part_rows = _normalize_rows(rows, columns)
        arrow_table = pa.Table.from_pylist(part_rows, schema=schema)
        source_hash = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:10]
        part_path = table_dir / f"part-{part_index:05d}-{source_hash}.parquet"
        tmp_path = table_dir / f".{part_path.name}.tmp"
        if tmp_path.exists():
            tmp_path.unlink()
        pq.write_table(arrow_table, tmp_path)
        os.replace(tmp_path, part_path)
        row_count += len(rows)
        part_paths.append(str(part_path))
        part_index += 1

    return row_count, part_paths


def export_sources(
    zarr_paths: Sequence[Path],
    *,
    output_root: Path,
    tables: Sequence[str] = DEFAULT_TABLES,
    jobs: int = 1,
    export_run_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    tables = _parse_tables(tables)
    output_root = Path(output_root).expanduser().resolve()
    export_run_id = export_run_id or _utc_now_id()
    zarr_paths = [Path(path).expanduser().resolve() for path in zarr_paths]
    if not zarr_paths:
        raise ValueError("No analysis Zarr sources were provided or discovered.")

    results: list[SourceExportResult] = []
    if jobs <= 1:
        for path in zarr_paths:
            results.append(export_one_zarr(path, tables=tables, export_run_id=export_run_id))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(export_one_zarr, path, tables=tables, export_run_id=export_run_id)
                for path in zarr_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item.zarr_path)

    rows_by_table_source: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        table: [] for table in tables
    }
    diagnostics: list[dict[str, Any]] = []
    for result in results:
        diagnostics.extend(
            {"zarr_path": result.zarr_path, "recording_id": result.recording_id, **diag}
            for diag in result.diagnostics
        )
        for table in tables:
            rows_by_table_source[table].append((result.zarr_path, result.rows_by_table.get(table, [])))

    output_root.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    part_files: dict[str, list[str]] = {}
    for table in tables:
        count, parts = _write_table_parts(
            output_root=output_root,
            export_run_id=export_run_id,
            table=table,
            rows_by_source=rows_by_table_source[table],
            overwrite=overwrite,
        )
        row_counts[table] = count
        part_files[table] = parts

    git = get_git_info(Path(__file__).resolve().parents[3])
    manifest = {
        "export_run_id": export_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": EXPORT_SCHEMA_VERSION,
        "tool": "fisheye.utils.export_cross_recording_analytics",
        "hostname": socket.gethostname(),
        "palette_git_commit": git.get("commit_hash"),
        "palette_git_dirty": git.get("is_dirty"),
        "source_recording_count": len(zarr_paths),
        "source_zarrs": [str(path) for path in zarr_paths],
        "tables_requested": list(tables),
        "row_counts_by_table": row_counts,
        "part_files_by_table": part_files,
        "diagnostics": diagnostics,
        "export_parameters": {
            "jobs": jobs,
            "overwrite": overwrite,
        },
    }
    manifest_dir = output_root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"export_run_id={export_run_id}.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Export manifest already exists: {manifest_path}")
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    os.replace(tmp_manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export cross-recording Palette analytics tables to Parquet.",
    )
    parser.add_argument("--zarr", action="append", type=Path, help="Analysis Zarr path. May be repeated.")
    parser.add_argument("--recordings-root", type=Path, help="Root to scan recursively for *_analysis.zarr archives.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root, e.g. /nvme1/exports/palette_analytics.",
    )
    parser.add_argument(
        "--tables",
        default=",".join(DEFAULT_TABLES),
        help=f"Comma-separated table list. Available: {', '.join(DEFAULT_TABLES)}.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel extraction workers by recording.")
    parser.add_argument("--limit", type=int, help="Limit discovered sources, useful for canaries.")
    parser.add_argument("--export-run-id", help="Explicit export run id. Defaults to current UTC timestamp.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing export_run_id directory/manifest.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tables = _parse_tables(args.tables)
    sources = _collect_sources(args)
    manifest = export_sources(
        sources,
        output_root=args.output_root,
        tables=tables,
        jobs=max(1, int(args.jobs)),
        export_run_id=args.export_run_id,
        overwrite=bool(args.overwrite),
    )
    print(f"export_run_id\t{manifest['export_run_id']}")
    print(f"manifest\t{manifest['manifest_path']}")
    for table, count in manifest["row_counts_by_table"].items():
        print(f"rows\t{table}\t{count}")
    if manifest["diagnostics"]:
        print(f"diagnostics\t{len(manifest['diagnostics'])}")


if __name__ == "__main__":
    main()
