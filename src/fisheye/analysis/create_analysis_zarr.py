#!/usr/bin/env python3
"""Create/validate analysis Zarr archives independently from inference stages."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.utils.import_video_metadata import probe_video_metadata, write_video_metadata

from .import_stimulus_to_zarr import import_stimulus_to_zarr


_utc_now = utc_now
JsonLogger = SharedJsonLogger


@dataclass
class CreationPlan:
    recording_dir: Optional[Path]
    video_path: Optional[Path]
    h5_path: Optional[Path]
    output_path: Optional[Path]
    import_stimulus: bool
    status: str
    reason: Optional[str] = None


def _normalize_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path.expanduser().resolve()


def _infer_recording_dir(
    *,
    recording_dir: Optional[Path],
    video_path: Optional[Path],
    h5_path: Optional[Path],
    output_path: Optional[Path],
) -> Optional[Path]:
    if recording_dir is not None:
        return _normalize_path(recording_dir)

    if h5_path is not None:
        h5 = _normalize_path(h5_path)
        if h5 is not None and h5.parent.name == "raw":
            return h5.parent.parent

    if video_path is not None:
        vid = _normalize_path(video_path)
        if vid is not None and vid.parent.name == "cams":
            return vid.parent.parent

    if output_path is not None:
        out = _normalize_path(output_path)
        if out is not None and out.parent.name == "zarr":
            return out.parent.parent

    return None


def _resolve_video(recording_dir: Optional[Path], explicit_video: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    if explicit_video is not None:
        video = _normalize_path(explicit_video)
        if video is None or not video.exists():
            return None, f"video does not exist: {explicit_video}"
        if not video.is_file():
            return None, f"video path is not a file: {video}"
        return video, None

    if recording_dir is None:
        return None, "recording_dir required to auto-resolve video path"

    cams_dir = recording_dir / "cams"
    if not cams_dir.exists():
        return None, f"missing cams/ directory: {cams_dir}"

    mp4s = sorted(cams_dir.glob("*.mp4"))
    if not mp4s:
        return None, "no .mp4 files found under cams/"
    if len(mp4s) > 1:
        return None, "multiple camera videos found; specify --video explicitly"
    return mp4s[0].resolve(), None


def _resolve_h5(recording_dir: Optional[Path], explicit_h5: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    if explicit_h5 is not None:
        h5 = _normalize_path(explicit_h5)
        if h5 is None or not h5.exists():
            return None, f"h5 does not exist: {explicit_h5}"
        if not h5.is_file():
            return None, f"h5 path is not a file: {h5}"
        return h5, None

    if recording_dir is None:
        return None, "recording_dir required to auto-resolve h5 path"

    raw_dir = recording_dir / "raw"
    if not raw_dir.exists():
        return None, f"missing raw/ directory: {raw_dir}"

    h5s = sorted(raw_dir.glob("*.h5")) + sorted(raw_dir.glob("*.hdf5"))
    if not h5s:
        return None, "no .h5/.hdf5 files found under raw/"
    if len(h5s) > 1:
        return None, "multiple h5 files found; specify --h5 explicitly"
    return h5s[0].resolve(), None


def _resolve_output(recording_dir: Optional[Path], explicit_output: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    if explicit_output is not None:
        return _normalize_path(explicit_output), None
    if recording_dir is None:
        return None, "recording_dir required to derive default output path"
    return (recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr").resolve(), None


def _build_plan(args: argparse.Namespace) -> CreationPlan:
    recording_dir = _infer_recording_dir(
        recording_dir=args.recording_dir,
        video_path=args.video,
        h5_path=args.h5,
        output_path=args.output,
    )

    reasons: List[str] = []
    if args.recording_dir is None and (args.video is None or args.output is None):
        reasons.append("provide --recording-dir or both --video and --output")

    if recording_dir is not None and not recording_dir.exists():
        reasons.append(f"recording_dir does not exist: {recording_dir}")

    video_path, video_reason = _resolve_video(recording_dir, args.video)
    if video_reason:
        reasons.append(video_reason)

    h5_path, h5_reason = _resolve_h5(recording_dir, args.h5)
    import_stimulus = args.import_stimulus
    if import_stimulus is None:
        import_stimulus = h5_reason is None
    if import_stimulus and h5_reason:
        reasons.append("stimulus import requested but h5 is unresolved")
        reasons.append(h5_reason)
    if not import_stimulus:
        h5_path = None

    output_path, output_reason = _resolve_output(recording_dir, args.output)
    if output_reason:
        reasons.append(output_reason)

    if reasons:
        return CreationPlan(
            recording_dir=recording_dir,
            video_path=video_path,
            h5_path=h5_path,
            output_path=output_path,
            import_stimulus=bool(import_stimulus),
            status="missing",
            reason="; ".join(reasons),
        )

    return CreationPlan(
        recording_dir=recording_dir,
        video_path=video_path,
        h5_path=h5_path,
        output_path=output_path,
        import_stimulus=bool(import_stimulus),
        status="ok",
    )


def _ensure_archive(plan: CreationPlan) -> None:
    assert plan.output_path is not None
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if plan.output_path.exists() else "w"
    root = zarr.open_group(str(plan.output_path), mode=mode)
    attrs = dict(root.attrs)
    attrs["zarr_purpose"] = "analysis"
    attrs["analysis_archive_created_utc"] = attrs.get("analysis_archive_created_utc") or _utc_now()
    attrs["analysis_archive_updated_utc"] = _utc_now()
    if plan.video_path is not None:
        attrs.setdefault("source_video", plan.video_path.name)
        attrs.setdefault("source_video_path", str(plan.video_path))
        attrs.setdefault("source_path", str(plan.video_path))
    root.attrs.put(attrs)


def _apply_video_metadata(plan: CreationPlan, *, overwrite: bool) -> dict[str, int]:
    assert plan.output_path is not None
    assert plan.video_path is not None
    root = zarr.open_group(str(plan.output_path), mode="r+")
    meta = probe_video_metadata(plan.video_path)
    updates = write_video_metadata(root, meta, overwrite=overwrite, import_purpose="analysis")
    return {
        "root_attrs_updated": len(updates.get("root", {})),
        "raw_video_attrs_updated": len(updates.get("raw_video", {})),
    }


_run_id = make_run_id


def _resolve_log_dir(arg_log_dir: Optional[Path], recording_dir: Optional[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "create_analysis_zarr"
    if recording_dir is not None:
        return recording_dir / "logs" / "create_analysis_zarr"
    return Path.cwd() / "logs" / "create_analysis_zarr"


def _print_plan(plan: CreationPlan) -> None:
    print("Analysis Zarr plan")
    print(f"  status: {plan.status}")
    if plan.reason:
        print(f"  reason: {plan.reason}")
    print(f"  recording_dir: {plan.recording_dir}")
    print(f"  video: {plan.video_path}")
    print(f"  h5: {plan.h5_path}")
    print(f"  output: {plan.output_path}")
    print(f"  import_stimulus: {plan.import_stimulus}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or validate analysis Zarr archive independent of inference stages.",
    )
    parser.add_argument("--recording-dir", type=Path, help="Recording directory containing raw/ and cams/.")
    parser.add_argument("--video", type=Path, help="Explicit video path.")
    parser.add_argument("--h5", type=Path, help="Explicit H5 path.")
    parser.add_argument("--output", type=Path, help="Explicit output analysis zarr path.")
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
        help="Overwrite existing source-video metadata attrs when importing video metadata.",
    )

    parser.add_argument(
        "--import-stimulus",
        dest="import_stimulus",
        action="store_true",
        help="Import stimulus metadata into analysis/stimulus_runs.",
    )
    parser.add_argument(
        "--no-import-stimulus",
        dest="import_stimulus",
        action="store_false",
        help="Skip stimulus import.",
    )
    parser.set_defaults(import_stimulus=None, import_video_metadata=True)
    parser.add_argument("--stimulus-run-name", type=str, help="Optional stimulus run name.")
    parser.add_argument("--stimulus-overwrite", action="store_true", help="Overwrite stimulus run if it exists.")
    parser.add_argument("--stimulus-quiet", action="store_true", help="Suppress verbose stimulus output.")
    parser.add_argument(
        "--skip-chaser-repair",
        action="store_true",
        help="Skip chaser interpolation repair during stimulus import.",
    )

    parser.add_argument("--register", action="store_true", help="Scan resulting archive into registry.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")

    parser.add_argument("--apply", action="store_true", help="Apply create/update actions.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writes.")
    parser.add_argument("--log-dir", type=Path, help="Optional JSONL log directory.")
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL log output.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if not args.apply:
        args.dry_run = True

    plan = _build_plan(args)
    _print_plan(plan)
    print(f"  import_video_metadata: {bool(args.import_video_metadata)}")

    logger: Optional[JsonLogger] = None
    if not args.no_log:
        run_id = _run_id()
        log_dir = _resolve_log_dir(args.log_dir, plan.recording_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"create_analysis_zarr_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            apply=bool(args.apply),
            dry_run=bool(args.dry_run),
            recording_dir=str(plan.recording_dir) if plan.recording_dir else None,
            output_path=str(plan.output_path) if plan.output_path else None,
            import_stimulus=plan.import_stimulus,
            import_video_metadata=bool(args.import_video_metadata),
            video_metadata_overwrite=bool(args.video_metadata_overwrite),
            register=bool(args.register),
        )
        logger.log(
            "plan",
            status=plan.status,
            reason=plan.reason,
            recording_dir=str(plan.recording_dir) if plan.recording_dir else None,
            video_path=str(plan.video_path) if plan.video_path else None,
            h5_path=str(plan.h5_path) if plan.h5_path else None,
            output_path=str(plan.output_path) if plan.output_path else None,
            import_stimulus=plan.import_stimulus,
        )

    if plan.status != "ok":
        if logger is not None:
            logger.log("run_end", ok=False, reason=plan.reason)
            logger.close()
        return 1

    if args.dry_run:
        if logger is not None:
            logger.log("run_end", ok=True, dry_run=True)
            logger.close()
        return 0

    try:
        _ensure_archive(plan)
        if logger is not None:
            logger.log("archive_updated", output_path=str(plan.output_path))

        if args.import_video_metadata and plan.video_path is not None:
            metadata_updates = _apply_video_metadata(
                plan,
                overwrite=bool(args.video_metadata_overwrite),
            )
            print(
                "Source video metadata imported: "
                f"root_attrs_updated={metadata_updates['root_attrs_updated']} "
                f"raw_video_attrs_updated={metadata_updates['raw_video_attrs_updated']}"
            )
            if logger is not None:
                logger.log(
                    "video_metadata_imported",
                    output_path=str(plan.output_path),
                    video_path=str(plan.video_path),
                    overwrite=bool(args.video_metadata_overwrite),
                    root_attrs_updated=metadata_updates["root_attrs_updated"],
                    raw_video_attrs_updated=metadata_updates["raw_video_attrs_updated"],
                )

        if plan.import_stimulus and plan.h5_path is not None:
            run_name = import_stimulus_to_zarr(
                stimulus_h5=plan.h5_path,
                zarr_path=plan.output_path,
                run_name=args.stimulus_run_name,
                overwrite=bool(args.stimulus_overwrite),
                verbose=not bool(args.stimulus_quiet),
                repair_chaser_gaps=not bool(args.skip_chaser_repair),
            )
            if logger is not None:
                logger.log("stimulus_imported", output_path=str(plan.output_path), run_name=run_name)

        if args.register:
            registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
            registry = Registry(registry_path)
            try:
                dataset_id = registry.scan_zarr(plan.output_path)
            finally:
                registry.close()
            print(f"Registry: {registry_path}")
            print(f"Registered dataset_id: {dataset_id}")
            if logger is not None:
                logger.log(
                    "registry_scanned",
                    registry_path=str(registry_path),
                    output_path=str(plan.output_path),
                    dataset_id=dataset_id,
                )
    except Exception as exc:
        print(f"Error: {exc}")
        if logger is not None:
            logger.log("run_end", ok=False, error=str(exc))
            logger.close()
        return 1

    if logger is not None:
        logger.log("run_end", ok=True, dry_run=False)
        logger.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
