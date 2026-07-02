"""Materialize reusable stimulus epoch windows inside analysis zarrs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_lineage_fingerprint import (
    build_run_lineage_payload,
    write_run_lineage_attrs,
)
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_pending,
    mark_run_started,
    require_runs_parent,
    resolve_authoritative_run_name,
)
from fisheye.utils.system import get_git_info
from fisheye.visualization.plot_detection_epoch_heatmaps import (
    _event_names_from_columns,
    _first_column,
    _load_structured_group,
)


SCHEMA_ID = "palette.stimulus_epoch_windows.v1"
SCHEMA_VERSION = 1
METHOD = "goodcopbadcop_chaser_epochs"
METHOD_VERSION = "1"
PARENT_NAME = "stimulus_epoch_runs"


@dataclass(frozen=True)
class StimulusEpochWindow:
    window_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    source_start_event_name: str
    source_end_event_name: str
    source_start_event_frame: int
    source_end_event_frame: int
    source_policy: str


@dataclass(frozen=True)
class StimulusEpochResult:
    zarr_path: str
    recording_id: str
    run_name: str
    stimulus_run_name: str
    stimulus_path: str
    fps: float
    total_frames: int
    windows: tuple[StimulusEpochWindow, ...]


def utc_run_name(prefix: str = "goodcopbadcop_epochs") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _attr_text(attrs: Any, *keys: str) -> Optional[str]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _attr_int(attrs: Any, *keys: str) -> Optional[int]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _attr_float(attrs: Any, *keys: str) -> Optional[float]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _resolve_dimensions(root: zarr.Group) -> tuple[float, int]:
    raw_video = root.get("raw_video")
    fps = (
        _attr_float(root.attrs, "fps", "video_fps")
        or (_attr_float(raw_video.attrs, "fps", "video_fps") if raw_video is not None else None)
        or 30.0
    )
    total_frames = (
        _attr_int(root.attrs, "total_frames", "n_frames", "source_video_total_frames")
        or (
            _attr_int(raw_video.attrs, "total_frames", "source_video_total_frames", "original_video_length")
            if raw_video is not None
            else None
        )
        or 0
    )
    return float(fps), int(total_frames)


def _resolve_stimulus_run(
    root: zarr.Group,
    stimulus_run: Optional[str],
) -> tuple[zarr.Group, str, str]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise ValueError("Archive has no analysis/stimulus_runs group.")
    parent = analysis["stimulus_runs"]
    resolved = str(stimulus_run).strip() if stimulus_run else None
    if not resolved:
        resolved = resolve_authoritative_run_name(parent)
    if not resolved:
        latest = parent.attrs.get("latest")
        resolved = str(latest).strip() if latest else None
    if not resolved or resolved not in parent:
        raise ValueError("No usable stimulus run found; pass --stimulus-run.")
    return parent[resolved], resolved, f"analysis/stimulus_runs/{resolved}"


def _event_frames_from_stimulus(root: zarr.Group, stimulus_group: zarr.Group) -> dict[str, int]:
    events_node = stimulus_group.get("events")
    if events_node is None:
        raise ValueError("Stimulus run has no events group.")
    columns = _load_structured_group(events_node)
    event_names = _event_names_from_columns(root, columns)
    camera_frames = _first_column(columns, "camera_frame_id", "camera_frame_num", "triggering_camera_frame_id")
    if camera_frames is None:
        raise ValueError("Stimulus events lack camera frame column.")
    if len(event_names) != len(camera_frames):
        raise ValueError("Stimulus event names and frame columns disagree on length.")

    event_frames: dict[str, int] = {}
    for event_name, frame_value in zip(event_names, camera_frames):
        name = str(event_name).strip()
        if not name:
            continue
        frame = int(np.asarray(frame_value).item())
        if frame >= 0:
            event_frames.setdefault(name, frame)
    return event_frames


def _first_event(event_frames: Mapping[str, int], names: Sequence[str]) -> tuple[str, Optional[int]]:
    for name in names:
        if name in event_frames:
            return name, int(event_frames[name])
    return "", None


def build_goodcopbadcop_windows(
    event_frames: Mapping[str, int],
    *,
    fps: float,
    total_frames: int,
) -> tuple[StimulusEpochWindow, ...]:
    """Resolve GoodCopBadCop pre/training/post windows from stimulus events."""

    pre_name, pre_start = _first_event(event_frames, ("CHASER_PRE_PERIOD_START", "PROTOCOL_START"))
    training_name, training_start = _first_event(event_frames, ("CHASER_TRAINING_START",))
    post_name, post_start = _first_event(event_frames, ("CHASER_POST_PERIOD_START",))
    finish_name, finish = _first_event(
        event_frames,
        ("PROTOCOL_FINISH", "PROTOCOL_STOP", "STEP_END", "CHASER_PRESENTATION_END"),
    )

    if training_start is None:
        raise ValueError("Stimulus events do not include CHASER_TRAINING_START.")
    if post_start is None:
        raise ValueError("Stimulus events do not include CHASER_POST_PERIOD_START.")
    policy_notes: list[str] = []
    if pre_start is None:
        pre_name = "RECORDING_START_FALLBACK"
        pre_start = 0
        policy_notes.append("missing_pre_start_used_frame_0")
    if finish is None:
        finish_name = "RECORDING_END_FALLBACK"
        finish = int(total_frames) if total_frames > 0 else int(post_start)
        policy_notes.append("missing_finish_used_total_frames")

    max_frame = int(total_frames) - 1 if total_frames > 0 else max(int(finish) - 1, int(post_start))

    def make(
        window_id: int,
        label: str,
        start_frame: int,
        end_boundary_frame: int,
        start_event_name: str,
        end_event_name: str,
    ) -> StimulusEpochWindow:
        start = max(0, int(start_frame))
        end = min(max_frame, max(start, int(end_boundary_frame) - 1))
        start_s = float(start) / float(fps)
        end_s = float(end + 1) / float(fps)
        policy = "inclusive_start_exclusive_end_event_boundary"
        if policy_notes:
            policy += ";" + ";".join(policy_notes)
        return StimulusEpochWindow(
            window_id=int(window_id),
            label=label,
            start_frame=start,
            end_frame=end,
            start_time_s=start_s,
            end_time_s=end_s,
            duration_s=max(0.0, end_s - start_s),
            source_start_event_name=start_event_name,
            source_end_event_name=end_event_name,
            source_start_event_frame=int(start_frame),
            source_end_event_frame=int(end_boundary_frame),
            source_policy=policy,
        )

    return (
        make(0, "pre_event", int(pre_start), int(training_start), pre_name, training_name),
        make(1, "training_event", int(training_start), int(post_start), training_name, post_name),
        make(2, "post_event", int(post_start), int(finish), post_name, finish_name),
    )


def build_stimulus_epoch_result(
    zarr_path: Path,
    *,
    run_name: str,
    stimulus_run: Optional[str] = None,
) -> StimulusEpochResult:
    root = _open_root(zarr_path, mode="r")
    stimulus_group, stimulus_run_name, stimulus_path = _resolve_stimulus_run(root, stimulus_run)
    fps, total_frames = _resolve_dimensions(root)
    event_frames = _event_frames_from_stimulus(root, stimulus_group)
    windows = build_goodcopbadcop_windows(event_frames, fps=fps, total_frames=total_frames)
    recording_id = _attr_text(root.attrs, "recording_id", "recording_name") or Path(zarr_path).stem
    return StimulusEpochResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        run_name=run_name,
        stimulus_run_name=stimulus_run_name,
        stimulus_path=stimulus_path,
        fps=fps,
        total_frames=total_frames,
        windows=windows,
    )


def _bytes_array(values: Sequence[str], *, width: int = 96) -> np.ndarray:
    out = np.zeros((len(values), int(width)), dtype=np.uint8)
    for row_idx, value in enumerate(values):
        payload = str(value).encode("utf-8", "ignore")[: max(0, int(width) - 1)]
        if payload:
            out[row_idx, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return out


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(data)
    if arr.ndim == 0:
        chunks = (1,)
    elif arr.ndim == 1:
        chunks = (max(1, min(int(arr.shape[0]), 8192)),)
    else:
        chunks = (max(1, min(int(arr.shape[0]), 1024)), *arr.shape[1:])
    group.create_array(name, data=arr, chunks=chunks, overwrite=True)


def write_stimulus_epoch_run(
    zarr_path: Path,
    result: StimulusEpochResult,
    *,
    overwrite: bool = False,
) -> str:
    root = _open_root(zarr_path, mode="a")
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, PARENT_NAME)
    run_name = result.run_name
    if run_name in parent:
        if not overwrite:
            raise ValueError(f"Stimulus epoch run already exists: analysis/{PARENT_NAME}/{run_name}")
        del parent[run_name]
    run = parent.create_group(run_name)
    mark_run_pending(parent, run_name)
    mark_run_started(run, run_name=run_name, stage="stimulus_epoch")
    try:
        windows = run.require_group("windows")
        items = list(result.windows)
        _write_array(windows, "window_id", np.asarray([w.window_id for w in items], dtype=np.int32))
        _write_array(windows, "label_bytes", _bytes_array([w.label for w in items]))
        _write_array(windows, "start_frame", np.asarray([w.start_frame for w in items], dtype=np.int64))
        _write_array(windows, "end_frame", np.asarray([w.end_frame for w in items], dtype=np.int64))
        _write_array(windows, "start_time_s", np.asarray([w.start_time_s for w in items], dtype=np.float64))
        _write_array(windows, "end_time_s", np.asarray([w.end_time_s for w in items], dtype=np.float64))
        _write_array(windows, "duration_s", np.asarray([w.duration_s for w in items], dtype=np.float64))
        _write_array(
            windows,
            "source_start_event_name_bytes",
            _bytes_array([w.source_start_event_name for w in items]),
        )
        _write_array(
            windows,
            "source_end_event_name_bytes",
            _bytes_array([w.source_end_event_name for w in items]),
        )
        _write_array(
            windows,
            "source_start_event_frame",
            np.asarray([w.source_start_event_frame for w in items], dtype=np.int64),
        )
        _write_array(
            windows,
            "source_end_event_frame",
            np.asarray([w.source_end_event_frame for w in items], dtype=np.int64),
        )
        _write_array(windows, "source_policy_bytes", _bytes_array([w.source_policy for w in items], width=160))
        windows.attrs.update(
            {
                "storage_layout": "columnar",
                "field_names": [
                    "window_id",
                    "label_bytes",
                    "start_frame",
                    "end_frame",
                    "start_time_s",
                    "end_time_s",
                    "duration_s",
                    "source_start_event_name_bytes",
                    "source_end_event_name_bytes",
                    "source_start_event_frame",
                    "source_end_event_frame",
                    "source_policy_bytes",
                ],
            }
        )

        git = get_git_info(Path(__file__).resolve().parents[3])
        source_refs = {
            "source_stimulus_run": result.stimulus_run_name,
            "source_stimulus_path": result.stimulus_path,
        }
        parameters = {"epoch_policy": METHOD, "fps": result.fps, "total_frames": result.total_frames}
        attrs = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_axis": "epoch_windows",
            "run_name": run_name,
            "recording_id": result.recording_id,
            "source_stimulus_run": result.stimulus_run_name,
            "source_stimulus_path": result.stimulus_path,
            "source_event_schema": {
                "events_path": f"{result.stimulus_path}/events",
                "event_name_fields": ["event_name", "event_type_name", "name", "event_type_id"],
                "frame_fields": ["camera_frame_id", "camera_frame_num", "triggering_camera_frame_id"],
            },
            "epoch_policy": METHOD,
            "source_refs": source_refs,
            "parameters": parameters,
            "fps": float(result.fps),
            "total_frames": int(result.total_frames),
            "window_count": len(items),
            "git_commit": git.get("commit_hash"),
            "git_branch": git.get("branch"),
            "git_dirty": git.get("is_dirty"),
            "provenance": {
                "stage": "stimulus_epoch",
                "created_by": "fisheye.analysis.stimulus_epoch_runs",
                "inputs": source_refs,
                "parameters": parameters,
            },
        }
        run.attrs.update(json_attr_safe(attrs))
        lineage_payload = build_run_lineage_payload(
            run_family="analysis/stimulus_epoch_runs",
            analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "epoch_windows"},
            method=METHOD,
            method_version=METHOD_VERSION,
            source_refs=source_refs,
            parameters=parameters,
            code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
        )
        write_run_lineage_attrs(run, lineage_payload, fingerprint_status="best_effort", overwrite=True)
        mark_run_complete(run, parent_group=parent, run_name=run_name)
    except Exception as exc:
        mark_run_failed(run, error=str(exc))
        raise
    return f"analysis/{PARENT_NAME}/{run_name}"


def _result_payload(result: StimulusEpochResult) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "run_name": result.run_name,
        "stimulus_run_name": result.stimulus_run_name,
        "stimulus_path": result.stimulus_path,
        "fps": result.fps,
        "total_frames": result.total_frames,
        "windows": [asdict(window) for window in result.windows],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--run-name", default=utc_run_name(), help="Stimulus epoch run name.")
    parser.add_argument("--stimulus-run", help="Explicit source stimulus run name.")
    parser.add_argument("--apply", action="store_true", help="Write analysis/stimulus_epoch_runs/<run>.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run with the same name.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_stimulus_epoch_result(
        Path(args.zarr_path),
        run_name=str(args.run_name),
        stimulus_run=args.stimulus_run,
    )
    path = None
    if args.apply:
        path = write_stimulus_epoch_run(Path(args.zarr_path), result, overwrite=bool(args.overwrite))
    payload = _result_payload(result)
    payload["applied_path"] = path
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id: {result.recording_id}")
        print(f"stimulus_run: {result.stimulus_run_name}")
        for window in result.windows:
            print(
                f"  {window.label}: frames={window.start_frame}-{window.end_frame} "
                f"duration_s={window.duration_s:.3f}"
            )
        if path:
            print(f"wrote: {path}")
        else:
            print("dry_run: pass --apply to write analysis/stimulus_epoch_runs/<run>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
