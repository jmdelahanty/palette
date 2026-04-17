#!/usr/bin/env python3
"""Single-recording analysis import helpers (create archive + metadata/stimulus import).

This module intentionally excludes detect/refine orchestration. Use
`fisheye.utils.run_recording_analysis_pipeline` for the full pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import zarr

from fisheye.utils.import_video_metadata import probe_video_metadata, write_video_metadata
from fisheye.utils.recording_preflight import preflight_gate_reason


@dataclass
class RecordingAnalysisPlan:
    recording_dir: Path
    h5_path: Path
    cam_video: Path
    zarr_path: Path


@dataclass
class RecordingImportOptions:
    import_video_metadata: bool
    video_metadata_overwrite: bool
    import_stimulus: bool
    stimulus_always: bool
    stimulus_run_name: Optional[str]
    stimulus_overwrite: bool
    stimulus_quiet: bool
    allow_preflight_failures: bool = False


@dataclass
class RecordingImportResult:
    ok: bool
    failed_step: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None


# Backward-compatible aliases during refactor window.
RecordingAnalysisOptions = RecordingImportOptions
RecordingAnalysisResult = RecordingImportResult


EventLogger = Callable[[str], None]


def stimulus_runs_present(zarr_path: Path) -> bool:
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return False
    analysis = root.get("analysis")
    if analysis is None:
        return False
    stim = analysis.get("stimulus_runs")
    if stim is None:
        return False
    try:
        return len(list(stim.group_keys())) > 0
    except Exception:
        return False


def ensure_analysis_archive(plan: RecordingAnalysisPlan) -> None:
    plan.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if plan.zarr_path.exists() else "w"
    root = zarr.open_group(str(plan.zarr_path), mode=mode)
    attrs = dict(root.attrs)
    attrs["zarr_purpose"] = "analysis"
    attrs.setdefault("source_video", plan.cam_video.name)
    attrs.setdefault("source_video_path", str(plan.cam_video))
    attrs.setdefault("source_path", str(plan.cam_video))
    root.attrs.put(attrs)


def apply_video_metadata(plan: RecordingAnalysisPlan, *, overwrite: bool) -> dict[str, int]:
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    meta = probe_video_metadata(plan.cam_video)
    updates = write_video_metadata(root, meta, overwrite=overwrite, import_purpose="analysis")
    return {
        "root_attrs_updated": len(updates.get("root", {})),
        "raw_video_attrs_updated": len(updates.get("raw_video", {})),
    }


def _log(logger: Optional[Callable[..., None]], event: str, **fields: object) -> None:
    if logger is not None:
        logger(event, **fields)


def run_stimulus_import(plan: RecordingAnalysisPlan, opts: RecordingImportOptions) -> tuple[bool, int, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.import_stimulus_to_zarr",
        str(plan.h5_path),
        str(plan.zarr_path),
    ]
    if opts.stimulus_run_name:
        cmd.extend(["--run-name", opts.stimulus_run_name])
    if opts.stimulus_overwrite:
        cmd.append("--overwrite")
    if opts.stimulus_quiet:
        cmd.append("--quiet")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def process_recording_import(
    plan: RecordingAnalysisPlan,
    opts: RecordingImportOptions,
    *,
    logger: Optional[Callable[..., None]] = None,
) -> RecordingImportResult:
    gate_reason = preflight_gate_reason(
        plan.recording_dir,
        allow_failures=bool(opts.allow_preflight_failures),
    )
    if gate_reason is not None:
        _log(
            logger,
            "preflight_gate_blocked",
            recording_dir=str(plan.recording_dir),
            reason=gate_reason,
        )
        return RecordingImportResult(ok=False, failed_step="preflight_gate", error=gate_reason)

    try:
        ensure_analysis_archive(plan)
    except Exception as exc:
        return RecordingImportResult(ok=False, failed_step="ensure_analysis_archive", error=str(exc))

    if opts.import_video_metadata:
        try:
            updates = apply_video_metadata(plan, overwrite=bool(opts.video_metadata_overwrite))
        except Exception as exc:
            return RecordingImportResult(ok=False, failed_step="import_video_metadata", error=str(exc))
        _log(
            logger,
            "video_metadata_imported",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            root_attrs_updated=updates["root_attrs_updated"],
            raw_video_attrs_updated=updates["raw_video_attrs_updated"],
            overwrite=bool(opts.video_metadata_overwrite),
        )

    if opts.import_stimulus:
        stim_present = stimulus_runs_present(plan.zarr_path)
        if stim_present and not opts.stimulus_always:
            _log(
                logger,
                "stimulus_skipped",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
                reason="stimulus_runs already present",
            )
        else:
            stim_ok, stim_rc, stim_cmd = run_stimulus_import(plan, opts)
            _log(
                logger,
                "stimulus_result",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
                returncode=int(stim_rc),
                cmd=stim_cmd,
            )
            if not stim_ok:
                return RecordingImportResult(
                    ok=False,
                    failed_step="import_stimulus_to_zarr",
                    returncode=int(stim_rc),
                    error="stimulus import failed",
                )

    return RecordingImportResult(ok=True)


# Backward-compatible alias name.
process_recording_analysis = process_recording_import


def resolve_single_recording_plan(
    *,
    recording_dir: Path,
    video: Optional[Path] = None,
    h5: Optional[Path] = None,
    output: Optional[Path] = None,
) -> RecordingAnalysisPlan:
    rec_dir = recording_dir.expanduser().resolve()
    if not rec_dir.exists() or not rec_dir.is_dir():
        raise ValueError(f"recording_dir not found: {rec_dir}")

    if video is None:
        cams_dir = rec_dir / "cams"
        mp4s = sorted(cams_dir.glob("*.mp4"))
        if not mp4s:
            raise ValueError(f"no .mp4 files found under {cams_dir}")
        if len(mp4s) > 1:
            raise ValueError(
                f"multiple .mp4 files found under {cams_dir}; pass --video explicitly for single-recording mode"
            )
        cam_video = mp4s[0].resolve()
    else:
        cam_video = video.expanduser().resolve()
        if not cam_video.exists() or not cam_video.is_file():
            raise ValueError(f"video not found: {cam_video}")

    if h5 is None:
        raw_dir = rec_dir / "raw"
        h5s = sorted(raw_dir.glob("*.h5"))
        if not h5s:
            raise ValueError(f"no .h5 files found under {raw_dir}")
        if len(h5s) > 1:
            raise ValueError(
                f"multiple .h5 files found under {raw_dir}; pass --h5 explicitly for single-recording mode"
            )
        h5_path = h5s[0].resolve()
    else:
        h5_path = h5.expanduser().resolve()
        if not h5_path.exists() or not h5_path.is_file():
            raise ValueError(f"h5 not found: {h5_path}")

    if output is None:
        zarr_path = (rec_dir / "zarr" / f"{rec_dir.name}_analysis.zarr").resolve()
    else:
        zarr_path = output.expanduser().resolve()

    return RecordingAnalysisPlan(
        recording_dir=rec_dir,
        h5_path=h5_path,
        cam_video=cam_video,
        zarr_path=zarr_path,
    )


def _build_options(args: argparse.Namespace) -> RecordingImportOptions:
    return RecordingImportOptions(
        import_video_metadata=bool(args.import_video_metadata),
        video_metadata_overwrite=bool(args.video_metadata_overwrite),
        import_stimulus=bool(args.import_stimulus),
        stimulus_always=bool(args.stimulus_always),
        stimulus_run_name=args.stimulus_run_name,
        stimulus_overwrite=bool(args.stimulus_overwrite),
        stimulus_quiet=bool(args.stimulus_quiet),
        allow_preflight_failures=bool(args.allow_preflight_failures),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import/create analysis zarr for one recording (archive + metadata + optional stimulus).",
    )
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--video", type=Path, help="Optional explicit camera video path.")
    parser.add_argument("--h5", type=Path, help="Optional explicit stimulus H5 path.")
    parser.add_argument("--output", type=Path, help="Optional explicit analysis zarr output path.")

    parser.add_argument("--apply", action="store_true", help="Execute import steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan only.")

    parser.add_argument(
        "--import-video-metadata",
        dest="import_video_metadata",
        action="store_true",
        help="Import source-video metadata into root/raw_video attrs (default).",
    )
    parser.add_argument(
        "--no-import-video-metadata",
        dest="import_video_metadata",
        action="store_false",
        help="Skip source-video metadata import.",
    )
    parser.add_argument(
        "--video-metadata-overwrite",
        action="store_true",
        help="Overwrite existing source-video metadata attrs when importing.",
    )

    parser.add_argument(
        "--import-stimulus",
        dest="import_stimulus",
        action="store_true",
        help="Import H5 stimulus metadata into analysis/stimulus_runs (default).",
    )
    parser.add_argument(
        "--no-import-stimulus",
        dest="import_stimulus",
        action="store_false",
        help="Skip stimulus import.",
    )
    parser.set_defaults(import_stimulus=True, import_video_metadata=True)
    parser.add_argument("--stimulus-always", action="store_true", help="Run stimulus import even if runs exist.")
    parser.add_argument("--stimulus-run-name", type=str, help="Optional stimulus run name.")
    parser.add_argument("--stimulus-overwrite", action="store_true", help="Overwrite existing stimulus run name.")
    parser.add_argument("--stimulus-quiet", action="store_true", help="Suppress verbose stimulus import output.")
    parser.add_argument(
        "--allow-preflight-failures",
        action="store_true",
        help="Proceed even if recording_manifest.json marks preflight.status=fail.",
    )

    args = parser.parse_args(argv)
    if not args.apply:
        args.dry_run = True

    try:
        plan = resolve_single_recording_plan(
            recording_dir=args.recording_dir,
            video=args.video,
            h5=args.h5,
            output=args.output,
        )
    except ValueError as exc:
        print(f"Plan resolution failed: {exc}")
        return 1

    print("Single recording import plan")
    print(f"  recording_dir: {plan.recording_dir}")
    print(f"  video: {plan.cam_video}")
    print(f"  h5: {plan.h5_path}")
    print(f"  output: {plan.zarr_path}")
    print(f"  import_video_metadata: {bool(args.import_video_metadata)}")
    print(f"  import_stimulus: {bool(args.import_stimulus)}")
    print(f"  allow_preflight_failures: {bool(args.allow_preflight_failures)}")
    if args.dry_run:
        print("Dry run: no changes were made.")
        return 0

    opts = _build_options(args)
    result = process_recording_import(plan, opts, logger=None)

    if not result.ok:
        print(
            f"Failed: step={result.failed_step or 'unknown'}"
            + (f" returncode={result.returncode}" if result.returncode is not None else "")
            + (f" error={result.error}" if result.error else "")
        )
        return 1

    print("Success: analysis import completed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
