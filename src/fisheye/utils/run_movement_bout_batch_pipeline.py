"""Batch-run movement, swim-bout, and bout-kinematics canary parameters.

The utility is intentionally conservative:

* default mode is dry-run;
* archives are processed serially by default, with opt-in archive-level parallelism;
* run names and parameters mirror the vetted feeding canary;
* each child stage remains responsible for its own detailed provenance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import zarr

from fisheye.analysis.eye_angle_io import (
    EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
    EyeAngleIOError,
    load_eye_angle_run_tables,
)
from fisheye.utils.zarr_io import open_zarr_root


DEFAULT_TRACK_RUN = "tk_hyst4_low2_latch_s005"
DEFAULT_EYE_ANGLE_LAYOUT = EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
DEFAULT_EYE_ANGLE_RUN = "eye_angle_compact_dense_v2_batch_20260511"
DEFAULT_SWIM_BOUT_RUN = "bouts_tk_hyst4_low2_latch_s005_peak_event_exp_tau025_prom4_dist010_w098"
DEFAULT_BOUT_KINEMATICS_RUN = "bk_tk_hyst4_low2_latch_s005_peak_event_prom4_w098_interbout"


@dataclass(frozen=True)
class ArchivePlan:
    zarr_path: str
    crop_run: Optional[str]
    refined_keypoint_run: Optional[str]
    refined_subject_run: Optional[str]
    track_run: str
    eye_angle_run: str
    swim_bout_run: str
    bout_kinematics_run: str
    run_arena_assignment: bool
    run_track_kinematics: bool
    run_track_visualization: bool
    run_eye_angles: bool
    run_swim_bouts: bool
    run_bout_kinematics: bool
    include_eye_gaze: bool
    skip_reason: str = ""


@dataclass
class ArchiveResult:
    zarr_path: str
    track_run: str
    eye_angle_run: str
    swim_bout_run: str
    bout_kinematics_run: str
    planned_eye_angles: bool
    planned_eye_gaze: bool
    arena_assignment_status: str = "not_requested"
    track_kinematics_status: str = "not_requested"
    track_visualization_status: str = "not_requested"
    eye_angle_status: str = "not_requested"
    swim_bout_status: str = "not_requested"
    bout_kinematics_status: str = "not_requested"
    validation_status: str = "not_run"
    log_path: str = ""
    error: str = ""


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _attrs(group_path: Path) -> dict[str, Any]:
    payload = _read_json(group_path / "zarr.json")
    raw = payload.get("attributes") if isinstance(payload, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _child_groups(group_path: Path) -> list[str]:
    if not group_path.is_dir():
        return []
    return sorted(path.name for path in group_path.iterdir() if path.is_dir())


def _latest_group_name(parent_path: Path) -> Optional[str]:
    attrs = _attrs(parent_path)
    for key in ("latest", "latest_materialized", "detect_review_status_latest"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (parent_path / value).is_dir():
            return value
    children = _child_groups(parent_path)
    return children[-1] if children else None


def _zarr_paths_from_report(report_path: Path) -> list[Path]:
    payload = _read_json(report_path)
    if payload is None:
        raise ValueError(f"Could not read JSON report: {report_path}")

    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = payload.get("plans")
    if not isinstance(rows, list):
        raise ValueError(f"Report {report_path} does not contain a results or plans list.")

    seen: set[str] = set()
    zarr_paths: list[Path] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("zarr_path")
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        zarr_paths.append(path)
    if not zarr_paths:
        raise ValueError(f"Report {report_path} did not contain any zarr_path entries.")
    return zarr_paths


def _discover_analysis_zarrs(roots: Sequence[Path], *, include_smoke: bool) -> list[Path]:
    seen: set[str] = set()
    zarrs: list[Path] = []
    for root in roots:
        root = root.expanduser()
        candidates = [root] if root.name.endswith("_analysis.zarr") else sorted(root.rglob("*_analysis.zarr"))
        for candidate in candidates:
            if not include_smoke and "/smoke/" in str(candidate):
                continue
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            zarrs.append(candidate)
    return sorted(zarrs)


def _exists(zarr_path: Path, rel_path: str) -> bool:
    path = zarr_path / rel_path
    return (path / "zarr.json").exists() or path.exists()


def _swim_bout_logical_bouts_exist(zarr_path: Path, run_name: str) -> bool:
    """Return whether a swim-bout run exposes logical bout rows.

    This intentionally uses metadata-file checks instead of opening the Zarr
    store so the batch status path remains safe in Codex/sandbox contexts.
    """

    run_rel = f"analysis/swim_bout_runs/{run_name}"
    run_path = zarr_path / run_rel
    if not _exists(zarr_path, run_rel):
        return False

    attrs = _attrs(run_path)
    if attrs.get("layout") == "compact_tabular_v2":
        return _exists(zarr_path, f"{run_rel}/tables/bouts")

    default_level = attrs.get("default_level")
    candidate_levels: list[str] = []
    if isinstance(default_level, str) and default_level:
        candidate_levels.append(default_level)
    candidate_levels.append("speed_exponential")
    candidate_levels.extend(
        child for child in _child_groups(run_path) if child.startswith("speed_")
    )
    seen: set[str] = set()
    for level in candidate_levels:
        if level in seen:
            continue
        seen.add(level)
        if _exists(zarr_path, f"{run_rel}/{level}/bouts"):
            return True

    return _exists(zarr_path, f"{run_rel}/bouts")


def _bout_kinematics_logical_level_exists(zarr_path: Path, run_name: str, level_name: str) -> bool:
    """Return whether a bout-kinematics run exposes a logical measurement level."""

    run_rel = f"analysis/bout_kinematics_runs/{run_name}"
    run_path = zarr_path / run_rel
    if not _exists(zarr_path, run_rel):
        return False

    attrs = _attrs(run_path)
    if attrs.get("layout") == "compact_tabular_v2":
        if level_name == "movement":
            return _exists(zarr_path, f"{run_rel}/movement_metrics")
        if level_name == "eye_gaze":
            return _exists(zarr_path, f"{run_rel}/eye_gaze_metrics")
        if str(level_name).startswith("heading_"):
            return _exists(zarr_path, f"{run_rel}/heading_metrics")
        return False

    return _exists(zarr_path, f"{run_rel}/{level_name}/per_bout_metrics")


def _eye_angle_frame_outputs_exist(zarr_path: Path, eye_angle_run: str) -> bool:
    base = zarr_path / "analysis" / "eye_angle_runs" / eye_angle_run / "angles" / "frame"
    required = ("left_gaze_deg", "right_gaze_deg", "vergence_gaze_deg")
    if all((base / name / "zarr.json").exists() for name in required):
        return True

    # Compact-dense eye-angle runs do not materialize angles/frame/<name>
    # groups. Use the logical resolver so skip-existing planning treats compact
    # and hierarchical runs equivalently.
    try:
        root = open_zarr_root(zarr_path, mode="r")
        tables = load_eye_angle_run_tables(root, run_name=eye_angle_run)
    except (EyeAngleIOError, FileNotFoundError, KeyError, ValueError, OSError):
        return False
    return all(name in tables.frame for name in required)


def build_archive_plan(
    zarr_path: Path,
    *,
    track_run: str,
    eye_angle_run: str,
    swim_bout_run: str,
    bout_kinematics_run: str,
    include_eye_gaze: bool,
    skip_existing: bool,
    force_bout_kinematics: bool,
) -> ArchivePlan:
    crop_run = _latest_group_name(zarr_path / "crop_runs")
    refined_keypoint_run = _latest_group_name(zarr_path / "refined_keypoints_runs")
    refined_subject_run = _latest_group_name(zarr_path / "refined_subject_masks_runs")

    target_track_exists = _exists(zarr_path, f"analysis/track_kinematics_runs/offline/{track_run}")
    target_eye_exists = _exists(zarr_path, f"analysis/eye_angle_runs/{eye_angle_run}")
    target_swim_bout_exists = _exists(zarr_path, f"analysis/swim_bout_runs/{swim_bout_run}")
    target_bout_kin_exists = _exists(zarr_path, f"analysis/bout_kinematics_runs/{bout_kinematics_run}")
    eye_outputs_exist = target_eye_exists and _eye_angle_frame_outputs_exist(zarr_path, eye_angle_run)

    skip_reasons: list[str] = []
    if crop_run is None:
        skip_reasons.append("missing_crop_run")
    if refined_keypoint_run is None:
        skip_reasons.append("missing_refined_keypoints")
    if include_eye_gaze and refined_subject_run is None and not eye_outputs_exist:
        skip_reasons.append("missing_refined_subject_masks_for_eye_angles")

    prereqs_ok = crop_run is not None and refined_keypoint_run is not None
    can_eye = bool(include_eye_gaze and prereqs_ok and (refined_subject_run is not None or eye_outputs_exist))
    should_run_eye = bool(can_eye and not (skip_existing and target_eye_exists))
    planned_eye_outputs = bool(can_eye and (eye_outputs_exist or should_run_eye))

    should_run_track = bool(prereqs_ok and not (skip_existing and target_track_exists))
    should_run_swim = bool(prereqs_ok and not (skip_existing and target_swim_bout_exists))
    should_run_bout = bool(
        prereqs_ok
        and (planned_eye_outputs or not include_eye_gaze)
        and (force_bout_kinematics or not (skip_existing and target_bout_kin_exists))
    )

    if not should_run_bout and include_eye_gaze and not planned_eye_outputs:
        skip_reasons.append("eye_gaze_requested_but_no_eye_angle_outputs")

    return ArchivePlan(
        zarr_path=str(zarr_path),
        crop_run=crop_run,
        refined_keypoint_run=refined_keypoint_run,
        refined_subject_run=refined_subject_run,
        track_run=track_run,
        eye_angle_run=eye_angle_run,
        swim_bout_run=swim_bout_run,
        bout_kinematics_run=bout_kinematics_run,
        run_arena_assignment=should_run_track,
        run_track_kinematics=should_run_track,
        run_track_visualization=should_run_track,
        run_eye_angles=should_run_eye,
        run_swim_bouts=should_run_swim,
        run_bout_kinematics=should_run_bout,
        include_eye_gaze=bool(include_eye_gaze and planned_eye_outputs),
        skip_reason=", ".join(skip_reasons),
    )


def _write_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _run_command(cmd: Sequence[str], *, dry_run: bool, log_path: Optional[Path] = None) -> str:
    command_line = "+ " + " ".join(str(part) for part in cmd)
    if log_path is None:
        print(command_line, flush=True)
    else:
        _write_log(log_path, command_line)
    if dry_run:
        return "planned"
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    if log_path is None:
        completed = subprocess.run(list(cmd), check=False, env=env)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(list(cmd), check=False, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return "ok" if completed.returncode == 0 else f"failed_exit_{completed.returncode}"


def _arena_assignment_command(plan: ArchivePlan) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fisheye.tracking.arena_assignment",
        plan.zarr_path,
        "--source-rowset",
        f"crop_runs/{plan.crop_run}",
    ]


def _track_kinematics_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fisheye.analysis.track_kinematics",
        plan.zarr_path,
        "--offline-only",
        "--keypoint-run",
        f"refined/{plan.refined_keypoint_run}",
        "--hysteresis-high-px",
        str(args.hysteresis_high_px),
        "--hysteresis-low-px",
        str(args.hysteresis_low_px),
        "--hysteresis-min-frames",
        str(args.hysteresis_min_frames),
        "--hysteresis-band-policy",
        args.hysteresis_band_policy,
        "--smooth-seconds",
        str(args.smooth_seconds),
        "--smoothing-alignment",
        args.smoothing_alignment,
        "--offline-run-name",
        plan.track_run,
    ]


def _track_visualization_command(plan: ArchivePlan) -> list[str]:
    return [
        sys.executable,
        "-m",
        "fisheye.analysis.plot_track_kinematics",
        plan.zarr_path,
        "--track-kinematics-run",
        plan.track_run,
        "--track-id",
        "0",
        "--offline-only",
        "--write-zarr-artifacts",
        "--swim-bout-run",
        "none",
    ]


def _eye_angle_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.eye_angle_analysis",
        plan.zarr_path,
        "--refined-subject-run",
        str(plan.refined_subject_run),
        "--keypoint-run",
        str(plan.refined_keypoint_run),
        "--run-name",
        plan.eye_angle_run,
        "--chunk-size",
        str(args.eye_angle_chunk_size),
        "--layout",
        DEFAULT_EYE_ANGLE_LAYOUT,
        "--quiet",
    ]
    if args.eye_angle_execution_backend:
        cmd.extend(["--execution-backend", args.eye_angle_execution_backend])
        cmd.extend(["--scheduler", args.eye_angle_scheduler])
        cmd.extend(["--num-workers", str(args.eye_angle_num_workers)])
    return cmd


def _swim_bout_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.detect_bouts_multi_level",
        plan.zarr_path,
        "--run-name",
        plan.swim_bout_run,
        "--track-kinematics-run",
        plan.track_run,
        "--track-id",
        "0",
        "--method",
        "peak_event",
        "--default-level",
        "exponential",
        "--exponential-source-level",
        "filtered",
        "--exponential-tau-s",
        str(args.exponential_tau_s),
        "--min-peak-prominence-mm-s",
        str(args.min_peak_prominence_mm_s),
        "--min-peak-distance-s",
        str(args.min_peak_distance_s),
        "--peak-width-rel-height",
        str(args.peak_width_rel_height),
        "--peak-event-boundary-mode",
        "relative_prominence_width",
        "--shape-split-policy",
        "none",
        "--min-bout-duration",
        str(args.min_bout_duration_s),
        "--min-gap-duration",
        str(args.min_gap_duration_s),
        "--gap-merge-policy",
        "sampled_frame_gap",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _bout_kinematics_command(plan: ArchivePlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.bout_kinematics",
        plan.zarr_path,
        "--run-name",
        plan.bout_kinematics_run,
        "--track-kinematics-run",
        plan.track_run,
        "--track-scope",
        "offline",
        "--track-id",
        "0",
        "--swim-bout-run",
        plan.swim_bout_run,
        "--speed-level",
        "exponential",
        "--pre-post-mode",
        "interbout_epoch",
        "--pre-window-s",
        str(args.pre_window_s),
        "--post-window-s",
        str(args.post_window_s),
        "--within-window",
        "bout_start_end",
        "--physical-active-signal-level",
        "filtered",
        "--physical-active-threshold-mm-s",
        str(args.physical_active_threshold_mm_s),
        "--physical-active-boundary-constraint",
        "search_with_margin",
        "--physical-active-boundary-margin-s",
        str(args.physical_active_boundary_margin_s),
        "--write-zarr-artifacts",
    ]
    if plan.include_eye_gaze:
        cmd.extend(["--include-eye-gaze", "--eye-angle-run", plan.eye_angle_run, "--eye-angle-family", "gaze"])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _validate_plan_outputs(plan: ArchivePlan) -> tuple[str, str]:
    zarr_path = Path(plan.zarr_path)
    checks = [
        f"analysis/track_kinematics_runs/offline/{plan.track_run}",
    ]
    if plan.include_eye_gaze:
        checks.append(f"analysis/eye_angle_runs/{plan.eye_angle_run}")
    missing = [rel for rel in checks if not _exists(zarr_path, rel)]
    if not _swim_bout_logical_bouts_exist(zarr_path, plan.swim_bout_run):
        missing.append(f"analysis/swim_bout_runs/{plan.swim_bout_run} logical bouts")
    if not _bout_kinematics_logical_level_exists(zarr_path, plan.bout_kinematics_run, "movement"):
        missing.append(f"analysis/bout_kinematics_runs/{plan.bout_kinematics_run} logical movement metrics")
    if plan.include_eye_gaze and not _bout_kinematics_logical_level_exists(
        zarr_path, plan.bout_kinematics_run, "eye_gaze"
    ):
        missing.append(f"analysis/bout_kinematics_runs/{plan.bout_kinematics_run} logical eye_gaze metrics")
    if missing:
        return "failed", "missing " + ", ".join(missing)
    return "ok", ""


def _safe_log_stem(zarr_path: str) -> str:
    raw = Path(zarr_path).name
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)


def _run_archive_plan(
    plan: ArchivePlan,
    *,
    args: argparse.Namespace,
    dry_run: bool,
    index: int,
    total: int,
    log_path: Optional[Path] = None,
) -> ArchiveResult:
    if log_path is None:
        print(f"\n[{index}/{total}] {plan.zarr_path}", flush=True)
    else:
        _write_log(log_path, f"[{index}/{total}] {plan.zarr_path}")

    result = ArchiveResult(
        zarr_path=plan.zarr_path,
        track_run=plan.track_run,
        eye_angle_run=plan.eye_angle_run,
        swim_bout_run=plan.swim_bout_run,
        bout_kinematics_run=plan.bout_kinematics_run,
        planned_eye_angles=plan.run_eye_angles,
        planned_eye_gaze=plan.include_eye_gaze,
        log_path=str(log_path) if log_path is not None else "",
    )
    try:
        if plan.skip_reason:
            if log_path is None:
                print(f"plan_notes: {plan.skip_reason}", flush=True)
            else:
                _write_log(log_path, f"plan_notes: {plan.skip_reason}")
        if plan.run_arena_assignment:
            result.arena_assignment_status = _run_command(_arena_assignment_command(plan), dry_run=dry_run, log_path=log_path)
            if result.arena_assignment_status != "ok" and not dry_run:
                raise RuntimeError(f"arena_assignment {result.arena_assignment_status}")
        if plan.run_track_kinematics:
            result.track_kinematics_status = _run_command(_track_kinematics_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.track_kinematics_status != "ok" and not dry_run:
                raise RuntimeError(f"track_kinematics {result.track_kinematics_status}")
        if plan.run_track_visualization:
            result.track_visualization_status = _run_command(_track_visualization_command(plan), dry_run=dry_run, log_path=log_path)
            if result.track_visualization_status != "ok" and not dry_run:
                raise RuntimeError(f"track_visualization {result.track_visualization_status}")
        if plan.run_eye_angles:
            result.eye_angle_status = _run_command(_eye_angle_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.eye_angle_status != "ok" and not dry_run:
                raise RuntimeError(f"eye_angle {result.eye_angle_status}")
        if plan.run_swim_bouts:
            result.swim_bout_status = _run_command(_swim_bout_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.swim_bout_status != "ok" and not dry_run:
                raise RuntimeError(f"swim_bout {result.swim_bout_status}")
        if plan.run_bout_kinematics:
            result.bout_kinematics_status = _run_command(_bout_kinematics_command(plan, args), dry_run=dry_run, log_path=log_path)
            if result.bout_kinematics_status != "ok" and not dry_run:
                raise RuntimeError(f"bout_kinematics {result.bout_kinematics_status}")
        if dry_run:
            result.validation_status = "planned"
        else:
            status, detail = _validate_plan_outputs(plan)
            result.validation_status = status
            if status != "ok":
                raise RuntimeError(detail)
            if args.consolidate_metadata:
                zarr.consolidate_metadata(plan.zarr_path)
    except Exception as exc:
        result.error = str(exc)
        if log_path is None:
            print(f"error: {exc}", file=sys.stderr, flush=True)
        else:
            _write_log(log_path, f"error: {exc}")
    return result


def _write_json_report(path: Path, *, plans: Sequence[ArchivePlan], results: Sequence[ArchiveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plans": [asdict(plan) for plan in plans],
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown_report(path: Path, *, plans: Sequence[ArchivePlan], results: Sequence[ArchiveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Movement / Bout Batch Pipeline Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Planned archives: {len(plans)}",
        f"- Results: {len(results)}",
        f"- Eye-angle requested: {sum(1 for plan in plans if plan.run_eye_angles)}",
        f"- Eye-gaze bout metrics planned: {sum(1 for plan in plans if plan.include_eye_gaze)}",
        f"- Validation ok: {sum(1 for result in results if result.validation_status == 'ok')}",
        f"- Errors: {sum(1 for result in results if result.error)}",
        "",
        "## Results",
        "",
        "| Zarr | Track | Eye angle | Swim bouts | Bout kinematics | Validation | Log | Error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                Path(result.zarr_path).name,
                result.track_kinematics_status,
                result.eye_angle_status,
                result.swim_bout_status,
                result.bout_kinematics_status,
                result.validation_status,
                result.log_path,
                (result.error or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("/nvme1/recordings")])
    parser.add_argument("--roots-from-report", type=Path)
    parser.add_argument("--apply", action="store_true", help="Run commands. Default is dry-run planning.")
    parser.add_argument("--pilot-size", type=int, default=None)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent analysis archives to process concurrently. Default preserves serial execution.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-archive subprocess logs when --jobs > 1.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite swim-bout and bout-kinematics run names if present.")
    parser.add_argument(
        "--force-bout-kinematics",
        action="store_true",
        help="Run bout_kinematics even when the target run group already exists. Use with --overwrite to replace it.",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Plan existing outputs too.")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--track-run", default=DEFAULT_TRACK_RUN)
    parser.add_argument("--eye-angle-run", default=DEFAULT_EYE_ANGLE_RUN)
    parser.add_argument("--swim-bout-run", default=DEFAULT_SWIM_BOUT_RUN)
    parser.add_argument("--bout-kinematics-run", default=DEFAULT_BOUT_KINEMATICS_RUN)
    parser.add_argument("--include-eye-gaze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hysteresis-high-px", type=float, default=4.0)
    parser.add_argument("--hysteresis-low-px", type=float, default=2.0)
    parser.add_argument("--hysteresis-min-frames", type=int, default=3)
    parser.add_argument("--hysteresis-band-policy", choices=("reset", "latch"), default="latch")
    parser.add_argument("--smooth-seconds", type=float, default=0.05)
    parser.add_argument("--smoothing-alignment", choices=("centered", "causal"), default="causal")
    parser.add_argument("--exponential-tau-s", type=float, default=0.025)
    parser.add_argument("--min-peak-prominence-mm-s", type=float, default=4.0)
    parser.add_argument("--min-peak-distance-s", type=float, default=0.10)
    parser.add_argument("--peak-width-rel-height", type=float, default=0.98)
    parser.add_argument("--min-bout-duration-s", type=float, default=0.05)
    parser.add_argument("--min-gap-duration-s", type=float, default=0.1)
    parser.add_argument("--pre-window-s", type=float, default=0.05)
    parser.add_argument("--post-window-s", type=float, default=0.05)
    parser.add_argument("--physical-active-threshold-mm-s", type=float, default=0.01)
    parser.add_argument("--physical-active-boundary-margin-s", type=float, default=0.05)
    parser.add_argument("--eye-angle-chunk-size", type=int, default=8192)
    parser.add_argument("--eye-angle-execution-backend", choices=("serial_driver", "dask_worker_chunks"), default=None)
    parser.add_argument("--eye-angle-scheduler", choices=("single-threaded", "threads", "processes", "distributed"), default="processes")
    parser.add_argument("--eye-angle-num-workers", type=int, default=24)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--consolidate-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run = not bool(args.apply)

    root_inputs = _zarr_paths_from_report(args.roots_from_report) if args.roots_from_report else args.roots
    all_zarrs = _discover_analysis_zarrs(root_inputs, include_smoke=bool(args.include_smoke))
    all_plans = [
        build_archive_plan(
            zarr_path,
            track_run=args.track_run,
            eye_angle_run=args.eye_angle_run,
            swim_bout_run=args.swim_bout_run,
            bout_kinematics_run=args.bout_kinematics_run,
            include_eye_gaze=bool(args.include_eye_gaze),
            skip_existing=bool(args.skip_existing),
            force_bout_kinematics=bool(args.force_bout_kinematics),
        )
        for zarr_path in all_zarrs
    ]
    plans = [
        plan
        for plan in all_plans
        if any(
            (
                plan.run_arena_assignment,
                plan.run_track_kinematics,
                plan.run_eye_angles,
                plan.run_swim_bouts,
                plan.run_bout_kinematics,
            )
        )
    ]
    if args.pilot_size is not None:
        plans = plans[: max(0, int(args.pilot_size))]

    print(f"analysis_archives_discovered: {len(all_plans)}")
    print(f"archives_selected: {len(plans)}")
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"track_run: {args.track_run}")
    print(f"eye_angle_run: {args.eye_angle_run}")
    print(f"swim_bout_run: {args.swim_bout_run}")
    print(f"bout_kinematics_run: {args.bout_kinematics_run}")
    print(f"jobs: {max(1, int(args.jobs))}")

    results: list[ArchiveResult] = []
    exit_code = 0

    jobs = max(1, int(args.jobs))
    if jobs == 1 or dry_run:
        for idx, plan in enumerate(plans, start=1):
            result = _run_archive_plan(plan, args=args, dry_run=dry_run, index=idx, total=len(plans))
            results.append(result)
            if result.error:
                exit_code = 1
                if not bool(args.continue_on_error):
                    break
    else:
        log_dir = args.log_dir or Path("/tmp") / f"palette_movement_bout_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"log_dir: {log_dir}")
        if not bool(args.continue_on_error):
            print("note: --jobs > 1 collects all submitted archive results before exiting on errors.", flush=True)

        future_to_index = {}
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            for idx, plan in enumerate(plans, start=1):
                log_path = log_dir / f"{idx:03d}_{_safe_log_stem(plan.zarr_path)}.log"
                future = executor.submit(
                    _run_archive_plan,
                    plan,
                    args=args,
                    dry_run=dry_run,
                    index=idx,
                    total=len(plans),
                    log_path=log_path,
                )
                future_to_index[future] = idx

            ordered_results: dict[int, ArchiveResult] = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                ordered_results[idx] = result
                status = "ERROR" if result.error else "ok"
                print(f"[{idx}/{len(plans)}] {status} {Path(result.zarr_path).name} log={result.log_path}", flush=True)
                if result.error:
                    exit_code = 1
        results = [ordered_results[idx] for idx in sorted(ordered_results)]

    if args.json_report:
        _write_json_report(args.json_report, plans=plans, results=results)
    if args.markdown_report:
        _write_markdown_report(args.markdown_report, plans=plans, results=results)

    print("\nsummary:")
    print(json.dumps({"plans": len(plans), "results": len(results), "exit_code": exit_code}, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
