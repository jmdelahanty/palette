#!/usr/bin/env python3
"""Single-recording analysis pipeline wrapper.

Pipeline order:
1) import and receipt-bound current-source finalization, or verified bound replay
2) registry-resolved detect inference with node-local atomic publication
3) detect_quality (required before refine_detect)
4) refine_detect (optional)
5) keypoints (optional)
6) refine_keypoints (optional)
7) registry metadata refresh or compatibility scan (optional)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from fisheye.registry.db import RegistryPaths
from fisheye.registry.shadow_publish import (
    shadow_synchronize_recording_import,
)
from fisheye.shared.recording_import_receipt import (
    RecordingImportReceiptError,
    recording_import_receipt_paths,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    SourceRecordingIdentityError,
    load_source_recording_identity_profile,
    load_strict_json_object,
)
from fisheye.refinement.refine_keypoints import (
    require_future_normal_refined_keypoint_publication,
)
from fisheye.utils.import_recording_analysis import (
    RecordingAnalysisPlan,
    RecordingImportOptions,
    RecordingImportResult,
    process_recording_import,
    resolve_single_recording_plan,
)


@dataclass
class RecordingPipelineOptions:
    detect_config: Optional[Path]
    conf: Optional[float]
    iou: Optional[float]
    max_det: Optional[int]
    batch_size: Optional[int]
    cpu: bool
    set_id: Optional[str]
    require_unique: bool
    include_non_success: bool
    top_k: int
    expected_subject_count: Optional[int]
    refine_detect: bool
    refine_config: Optional[Path]
    register: bool
    registry_path: Path
    import_opts: RecordingImportOptions
    run_keypoints: bool = False
    refine_keypoints: bool = False
    keypoints_config: Optional[Path] = None


@dataclass
class RecordingPipelineResult:
    ok: bool
    failed_step: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None
    dataset_id: Optional[str] = None


EventLogger = Callable[[str], None]


def _log(logger: Optional[Callable[..., None]], event: str, **fields: object) -> None:
    if logger is not None:
        logger(event, **fields)


def _sync_pipeline_registry(
    *,
    registry_path: Path,
    plan: RecordingAnalysisPlan,
    receipt: object | None,
) -> str:
    publication = shadow_synchronize_recording_import(
        canonical_registry=registry_path,
        zarr_path=plan.zarr_path,
        receipt=receipt,
        decided_by="fisheye.utils.run_recording_analysis_pipeline",
    )
    return str(publication.mutation_result["dataset_id"])


def run_detect_registry_model(plan: RecordingAnalysisPlan, opts: RecordingPipelineOptions) -> tuple[bool, int, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.utils.run_detect_with_registry_model",
        "--recording-dir",
        str(plan.recording_dir),
        "--output",
        str(plan.zarr_path),
        "--registry",
        str(opts.registry_path),
    ]
    if opts.detect_config is not None:
        cmd.extend(["--config", str(opts.detect_config)])
    if opts.conf is not None:
        cmd.extend(["--conf", str(opts.conf)])
    if opts.iou is not None:
        cmd.extend(["--iou", str(opts.iou)])
    if opts.max_det is not None:
        cmd.extend(["--max-det", str(opts.max_det)])
    if opts.batch_size is not None:
        cmd.extend(["--batch-size", str(opts.batch_size)])
    if opts.cpu:
        cmd.append("--cpu")
    if opts.set_id is not None:
        cmd.extend(["--set-id", str(opts.set_id)])
    if opts.require_unique:
        cmd.append("--require-unique")
    if opts.include_non_success:
        cmd.append("--include-non-success")
    cmd.extend(["--top-k", str(opts.top_k)])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def run_refine_detect(plan: RecordingAnalysisPlan, opts: RecordingPipelineOptions) -> tuple[bool, int, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.refine_detect",
        str(plan.zarr_path),
    ]
    if opts.refine_config is not None:
        cmd.extend(["--config", str(opts.refine_config)])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def run_detect_quality(plan: RecordingAnalysisPlan, opts: RecordingPipelineOptions) -> tuple[bool, int, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.detect_quality",
        str(plan.zarr_path),
    ]
    if opts.expected_subject_count is not None:
        cmd.extend(["--expected-subject-count", str(opts.expected_subject_count)])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def run_keypoints_batch(plan: RecordingAnalysisPlan, opts: RecordingPipelineOptions) -> tuple[bool, int, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.utils.run_keypoints_batch",
        "--apply",
        "--quiet",
        "--no-log",
        str(plan.zarr_path),
    ]
    if opts.keypoints_config is not None:
        cmd.extend(["--config", str(opts.keypoints_config)])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def run_refine_keypoints(plan: RecordingAnalysisPlan, opts: RecordingPipelineOptions) -> tuple[bool, int, List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.refine_keypoints",
        str(plan.zarr_path),
    ]
    if opts.keypoints_config is not None:
        cmd.extend(["--config", str(opts.keypoints_config)])
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def process_recording_analysis_pipeline(
    plan: RecordingAnalysisPlan,
    opts: RecordingPipelineOptions,
    *,
    registry: object | None = None,
    logger: Optional[Callable[..., None]] = None,
) -> RecordingPipelineResult:
    if opts.refine_keypoints:
        require_future_normal_refined_keypoint_publication()
    dataset_id: str | None = None
    manifest_path = plan.recording_dir / "recording_manifest.json"
    if manifest_path.is_file():
        try:
            manifest_profile = load_strict_json_object(manifest_path).get(
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR
            )
        except SourceRecordingIdentityError as exc:
            return RecordingPipelineResult(
                ok=False,
                failed_step="recording_import_preflight",
                error=str(exc),
            )
        if (
            manifest_profile == SOURCE_RECORDING_IDENTITY_PROFILE
            and not opts.register
        ):
            return RecordingPipelineResult(
                ok=False,
                failed_step="recording_import_preflight",
                error="current source pipelines require receipt-bound registry publication",
            )
    current_profile = (
        plan.zarr_path.exists()
        and load_source_recording_identity_profile(plan.zarr_path)
        == SOURCE_RECORDING_IDENTITY_PROFILE
    )
    sealed_replay = False
    if opts.register and current_profile:
        try:
            sealed_replay = bool(recording_import_receipt_paths(plan.zarr_path))
        except RecordingImportReceiptError as exc:
            return RecordingPipelineResult(
                ok=False,
                failed_step="recording_import_preflight",
                error=str(exc),
            )
    if not sealed_replay:
        import_result = process_recording_import(plan, opts.import_opts, logger=logger)
    else:
        import_result = RecordingImportResult(ok=True)
    if not import_result.ok:
        return RecordingPipelineResult(
            ok=False,
            failed_step=import_result.failed_step,
            error=import_result.error,
            returncode=import_result.returncode,
        )

    current_profile = (
        (opts.register or plan.zarr_path.exists())
        and load_source_recording_identity_profile(plan.zarr_path)
        == SOURCE_RECORDING_IDENTITY_PROFILE
    )
    if opts.register and current_profile:
        dataset_id = _sync_pipeline_registry(
            registry_path=opts.registry_path,
            plan=plan,
            receipt=import_result.receipt,
        )

    detect_ok, detect_rc, detect_cmd = run_detect_registry_model(plan, opts)
    _log(
        logger,
        "detect_result",
        recording_dir=str(plan.recording_dir),
        zarr_path=str(plan.zarr_path),
        returncode=int(detect_rc),
        cmd=detect_cmd,
        model_source="registry",
    )
    if not detect_ok:
        return RecordingPipelineResult(
            ok=False,
            failed_step="detect_yolo",
            returncode=int(detect_rc),
            error="detect step failed",
        )

    if opts.refine_detect:
        quality_ok, quality_rc, quality_cmd = run_detect_quality(plan, opts)
        _log(
            logger,
            "detect_quality_result",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            returncode=int(quality_rc),
            cmd=quality_cmd,
        )
        if not quality_ok:
            return RecordingPipelineResult(
                ok=False,
                failed_step="detect_quality",
                returncode=int(quality_rc),
                error="detect quality failed",
            )

        refine_ok, refine_rc, refine_cmd = run_refine_detect(plan, opts)
        _log(
            logger,
            "refine_result",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            returncode=int(refine_rc),
            cmd=refine_cmd,
        )
        if not refine_ok:
            return RecordingPipelineResult(
                ok=False,
                failed_step="refine_detect",
                returncode=int(refine_rc),
                error="refine detect failed",
            )

    if opts.run_keypoints:
        keypoints_ok, keypoints_rc, keypoints_cmd = run_keypoints_batch(plan, opts)
        _log(
            logger,
            "keypoints_result",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            returncode=int(keypoints_rc),
            cmd=keypoints_cmd,
        )
        if not keypoints_ok:
            return RecordingPipelineResult(
                ok=False,
                failed_step="keypoints",
                returncode=int(keypoints_rc),
                error="keypoints step failed",
            )

    if opts.refine_keypoints:
        refine_kp_ok, refine_kp_rc, refine_kp_cmd = run_refine_keypoints(plan, opts)
        _log(
            logger,
            "refine_keypoints_result",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            returncode=int(refine_kp_rc),
            cmd=refine_kp_cmd,
        )
        if not refine_kp_ok:
            return RecordingPipelineResult(
                ok=False,
                failed_step="refine_keypoints",
                returncode=int(refine_kp_rc),
                error="refine keypoints failed",
            )

    if opts.register:
        dataset_id = _sync_pipeline_registry(
            registry_path=opts.registry_path,
            plan=plan,
            receipt=None if current_profile else import_result.receipt,
        )

    return RecordingPipelineResult(ok=True, dataset_id=dataset_id)


def _build_import_options(args: argparse.Namespace) -> RecordingImportOptions:
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


def _build_pipeline_options(args: argparse.Namespace, registry_path: Path) -> RecordingPipelineOptions:
    return RecordingPipelineOptions(
        detect_config=args.detect_config,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch_size=args.batch_size,
        cpu=bool(args.cpu),
        set_id=args.set_id,
        require_unique=bool(args.require_unique),
        include_non_success=bool(args.include_non_success),
        top_k=int(args.top_k),
        expected_subject_count=args.expected_subject_count,
        refine_detect=bool(args.refine_detect),
        refine_config=args.refine_config,
        register=bool(args.register),
        registry_path=registry_path,
        import_opts=_build_import_options(args),
        run_keypoints=bool(args.keypoints),
        refine_keypoints=bool(args.refine_keypoints),
        keypoints_config=args.keypoints_config,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one-recording analysis pipeline: import -> detect -> detect_quality -> refine_detect -> keypoints -> refine_keypoints -> register.",
    )
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--video", type=Path, help="Optional explicit camera video path.")
    parser.add_argument("--h5", type=Path, help="Optional explicit stimulus H5 path.")
    parser.add_argument("--output", type=Path, help="Optional explicit analysis zarr output path.")

    parser.add_argument("--apply", action="store_true", help="Execute pipeline steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan only.")

    parser.add_argument("--detect-config", type=Path, help="YOLO detect config YAML.")
    parser.add_argument("--conf", type=float, help="YOLO confidence threshold override.")
    parser.add_argument("--iou", type=float, help="YOLO IoU threshold override.")
    parser.add_argument("--max-det", type=int, help="YOLO max detections per frame override.")
    parser.add_argument("--batch-size", type=int, help="YOLO batch size override.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for YOLO detect.")
    parser.add_argument("--set-id", type=str, help="Optional registered detect model-set filter.")
    parser.add_argument("--require-unique", action="store_true", help="Fail detect step when top scores tie.")
    parser.add_argument(
        "--include-non-success",
        action="store_true",
        help="Allow non-success training runs as candidates for registry model resolution.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of detect model candidates to persist.")
    parser.add_argument(
        "--expected-subject-count",
        type=int,
        default=None,
        help=(
            "Expected total subjects per frame for detect_quality. For example, "
            "use 4 for four one-fish square sub-arenas in one camera view."
        ),
    )

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
    parser.add_argument(
        "--recording-only",
        action="store_true",
        help="Process a camera-video-only recording without requiring an H5/protocol source.",
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

    parser.add_argument(
        "--refine-detect",
        dest="refine_detect",
        action="store_true",
        help="Run refine_detect after detect (default).",
    )
    parser.add_argument(
        "--no-refine-detect",
        dest="refine_detect",
        action="store_false",
        help="Skip refine_detect.",
    )
    parser.set_defaults(refine_detect=True)
    parser.add_argument(
        "--refine-config",
        type=Path,
        default=Path("configs/fisheye/default.yaml"),
        help="Config passed to refine_detect.",
    )
    parser.add_argument("--refine-max-gap", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--keypoints", action="store_true", help="Run keypoints after detect/refine_detect.")
    parser.add_argument(
        "--refine-keypoints",
        action="store_true",
        help=(
            "Request refine_keypoints after keypoints; future-normal refined "
            "publication currently fails closed before opening the archive."
        ),
    )
    parser.add_argument(
        "--keypoints-config",
        type=Path,
        default=Path("configs/fisheye/default.yaml"),
        help="Config passed to keypoints/refine_keypoints stages.",
    )

    parser.add_argument(
        "--register",
        action="store_true",
        help="Receipt-finalize the current source import in the registry.",
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")

    args = parser.parse_args(argv)
    if args.recording_only:
        args.import_stimulus = False
    if args.refine_max_gap is not None:
        raise SystemExit(
            "Interpolation overrides are deprecated and unsupported for the recording analysis pipeline. "
            "Remove --refine-max-gap; refine_detect now always runs with interpolation disabled."
        )
    if not args.apply:
        args.dry_run = True

    try:
        plan = resolve_single_recording_plan(
            recording_dir=args.recording_dir,
            video=args.video,
            h5=args.h5,
            output=args.output,
            require_h5=bool(args.import_stimulus),
        )
    except ValueError as exc:
        print(f"Plan resolution failed: {exc}")
        return 1

    registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    if args.apply and not registry_path.exists():
        print(f"Registry not found for detection publication: {registry_path}")
        return 1

    print("Single recording analysis pipeline plan")
    print(f"  recording_dir: {plan.recording_dir}")
    print(f"  video: {plan.cam_video}")
    print(f"  h5: {plan.h5_path if plan.h5_path is not None else 'none (recording-only)'}")
    print(f"  output: {plan.zarr_path}")
    print("  model_source: registry")
    print(f"  import_stimulus: {bool(args.import_stimulus)}")
    print(f"  allow_preflight_failures: {bool(args.allow_preflight_failures)}")
    print(f"  refine_detect: {bool(args.refine_detect)}")
    print(f"  keypoints: {bool(args.keypoints)}")
    print(f"  refine_keypoints: {bool(args.refine_keypoints)}")
    print(f"  register: {bool(args.register)}")
    if args.dry_run:
        print("Dry run: no changes were made.")
        return 0

    if args.register:
        print(f"Registry: {registry_path}")

    opts = _build_pipeline_options(args, registry_path)
    result = process_recording_analysis_pipeline(
        plan,
        opts,
        registry=None,
        logger=None,
    )

    if not result.ok:
        print(
            f"Failed: step={result.failed_step or 'unknown'}"
            + (f" returncode={result.returncode}" if result.returncode is not None else "")
            + (f" error={result.error}" if result.error else "")
        )
        return 1

    print("Success: analysis pipeline completed.")
    if result.dataset_id:
        print(f"  dataset_id: {result.dataset_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
