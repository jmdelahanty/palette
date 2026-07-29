"""Plan and submit selector-ineligible native canonical detection campaigns."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from fisheye.cluster.clipped_inference import (
    DEFAULT_REGISTRY,
    CampaignTarget,
    ModelBinding,
    load_target_manifest,
    resolve_detection_model_binding,
)
from fisheye.cluster.keypoints.common import (
    safe_component,
    validate_registered_analysis_zarr,
)
from fisheye.cluster.lsf import (
    CommandRunner,
    LsfWorkflow,
    build_ssh_bsub_runner,
    submit_lsf_workflow,
    write_json_snapshot,
)
from fisheye.cluster.native_detection import (
    NativeDetectionAuthoritySpec,
    NativeDetectionClipSpec,
    NativeDetectionFragmentInputs,
    NativeDetectionModelSpec,
    build_native_detection_fragment,
    compose_native_detection_workflow,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.plan_clipped_detect_refine_workflow import (
    build_plan as build_detection_plan,
)


PLAN_SCHEMA = "palette.native_detection_campaign_plan.v1"
FAMILY = "native_canonical_detection"
DEFAULT_SUBMIT_HOST = "login1-citrus-poller"


@dataclass(frozen=True)
class NativeArchiveAuthority:
    recording_identity: str
    camera_serial: str
    n_frames: int
    source_width: int
    source_height: int
    frame: NativeDetectionAuthoritySpec
    pixel: NativeDetectionAuthoritySpec

    def to_json(self) -> dict[str, object]:
        return {
            "recording_identity": self.recording_identity,
            "camera_serial": self.camera_serial,
            "n_frames": self.n_frames,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_frame_authority": {
                "record_ref": self.frame.record_ref,
                "record_sha256": self.frame.record_sha256,
            },
            "source_pixel_authority": {
                "record_ref": self.pixel.record_ref,
                "record_sha256": self.pixel.record_sha256,
            },
            "pixel_authority_basis": (
                "acquisition_camera_frame.width_px_height_px"
            ),
        }


@dataclass(frozen=True)
class NativeDetectionCampaignPlan:
    run_label: str
    workflow_id: str
    repo: Path
    repo_commit: str
    registry: Path
    run_root: Path
    targets: tuple[CampaignTarget, ...]
    target_plans: tuple[Mapping[str, Any], ...]
    detection_plans: tuple[Mapping[str, Any], ...]
    model_bindings: tuple[ModelBinding, ...]
    workflow: LsfWorkflow

    def to_json(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "run_label": self.run_label,
            "workflow_id": self.workflow_id,
            "repo": str(self.repo),
            "repo_commit": self.repo_commit,
            "registry": str(self.registry),
            "run_root": str(self.run_root),
            "target_count": len(self.targets),
            "targets": [dict(item) for item in self.target_plans],
            "model_bindings": [item.to_json() for item in self.model_bindings],
            "selector_activation": "deferred",
            "registry_update": False,
            "lsf_workflow": self.workflow.to_json(),
        }


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _zarr_attrs(path: Path) -> Mapping[str, Any]:
    payload = _read_strict_json(path / "zarr.json")
    attrs = payload.get("attributes")
    if not isinstance(attrs, Mapping):
        raise ValueError(f"Zarr node has no attributes mapping: {path}")
    return attrs


def load_native_archive_authority(target: CampaignTarget) -> NativeArchiveAuthority:
    """Resolve the immutable acquisition record used for time and pixel extent."""

    archive = target.analysis_zarr.expanduser().resolve()
    root_attrs = _zarr_attrs(archive)
    recording_identity = str(root_attrs.get("recording_id") or "").strip()
    if not recording_identity or recording_identity != target.recording_id:
        raise ValueError(
            "Target recording_id differs from the analysis archive identity: "
            f"target={target.recording_id!r}, archive={recording_identity!r}."
        )
    raw_attrs = _zarr_attrs(archive / "raw_video")
    status = raw_attrs.get("acquisition_authority_publication_status")
    if not isinstance(status, Mapping):
        raise ValueError("Analysis archive lacks acquisition publication status.")
    if status.get("status") != "published_canonical_v1":
        raise ValueError("Acquisition authority is not published canonical v1.")
    authority_path = str(status.get("authority_path") or "").strip().strip("/")
    expected_prefix = "analysis/acquisition_camera_frames/"
    if not authority_path.startswith(expected_prefix):
        raise ValueError("Acquisition authority path is not canonical.")
    camera_serial = authority_path.removeprefix(expected_prefix)
    if not camera_serial or "/" in camera_serial:
        raise ValueError("Acquisition authority camera serial is invalid.")
    authority_attrs = _zarr_attrs(archive / authority_path)
    record = authority_attrs.get("acquisition_camera_frame")
    if not isinstance(record, Mapping):
        raise ValueError("Canonical acquisition-camera record is missing.")
    digest = str(authority_attrs.get("acquisition_camera_frame_sha256") or "")
    observed = canonical_json_sha256(record)
    if digest != observed:
        raise ValueError("Acquisition-camera record digest is stale.")
    if record.get("recording_id") != recording_identity:
        raise ValueError("Acquisition-camera record has the wrong recording identity.")
    if str(record.get("camera_id") or "") != camera_serial:
        raise ValueError("Acquisition-camera record has the wrong camera serial.")
    n_frames = int(record.get("source_total_frames") or 0)
    width = int(record.get("width_px") or 0)
    height = int(record.get("height_px") or 0)
    if min(n_frames, width, height) <= 0:
        raise ValueError("Acquisition-camera record has invalid source dimensions.")
    if int(record.get("frame_count") or 0) != n_frames:
        raise ValueError("Acquisition-camera frame count disagrees with source extent.")
    record_ref = f"/{authority_path}@acquisition_camera_frame"
    pointer = NativeDetectionAuthoritySpec(
        record_ref=record_ref,
        record_sha256=digest,
    )
    return NativeArchiveAuthority(
        recording_identity=recording_identity,
        camera_serial=camera_serial,
        n_frames=n_frames,
        source_width=width,
        source_height=height,
        frame=pointer,
        pixel=pointer,
    )


def _repo_commit(repo: Path) -> str:
    resolved = repo.expanduser().resolve()
    status = subprocess.run(
        ["git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        capture_output=True,
    )
    if status.stdout.strip():
        raise ValueError(f"Palette repo must be clean: {resolved}")
    commit = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("Palette repo did not resolve one full commit SHA.")
    return commit


def _validate_recording_frame_index(path: Path, *, n_frames: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"recording_frame_index.parquet not found: {path}")
    parquet = pq.ParquetFile(path)
    required = {
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "parent_frame_index",
    }
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"recording frame index is missing columns: {missing}")
    if parquet.metadata.num_rows != int(n_frames):
        raise ValueError(
            "recording frame index row count differs from acquisition authority: "
            f"{parquet.metadata.num_rows} != {n_frames}."
        )


def _refuse_outputs(
    *,
    analysis_zarr: Path,
    canonical_run_id: str,
    clips: Sequence[NativeDetectionClipSpec],
) -> None:
    outputs = [analysis_zarr / "detect_runs" / canonical_run_id]
    outputs.extend(analysis_zarr / clip.artifact_group_path for clip in clips)
    collisions = [path for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(
            "Planned immutable native-detection outputs already exist: "
            + ", ".join(str(path) for path in collisions)
        )


def build_plan(
    *,
    targets: Sequence[CampaignTarget],
    run_label: str,
    repo: Path,
    registry_path: Path,
    run_root: Path,
    detection_set_id: str,
    detection_run_id: str,
    detect_array_concurrency: int = 8,
) -> NativeDetectionCampaignPlan:
    if not targets:
        raise ValueError("Native detection campaign requires at least one target.")
    label = safe_component(run_label, default="native_detection", max_length=80)
    workflow_id = f"native_detection_{label}"
    resolved_repo = repo.expanduser().resolve()
    resolved_registry = registry_path.expanduser().resolve()
    resolved_run_root = run_root.expanduser().resolve()
    commit = _repo_commit(resolved_repo)
    modules = []
    target_plans: list[dict[str, Any]] = []
    detection_plans: list[Mapping[str, Any]] = []
    bindings: list[ModelBinding] = []

    for index, target in enumerate(targets):
        validate_registered_analysis_zarr(
            registry_path=resolved_registry,
            recording_id=target.recording_id,
            analysis_zarr=target.analysis_zarr,
        )
        binding = resolve_detection_model_binding(
            registry_path=resolved_registry,
            target=target,
            set_id=detection_set_id,
            run_id=detection_run_id,
        )
        authority = load_native_archive_authority(target)
        frame_index = target.recording_dir / "recording_frame_index.parquet"
        _validate_recording_frame_index(frame_index, n_frames=authority.n_frames)
        target_safe = safe_component(
            target.target_id,
            default=f"target_{index:03d}",
            max_length=56,
        )
        target_label = safe_component(
            f"{label}_{target_safe}",
            default=target_safe,
            max_length=90,
        )
        target_dir = resolved_run_root / "targets" / target_safe
        detection_plan_path = target_dir / "detection_plan.json"
        detection_plan = build_detection_plan(
            target.recording_dir,
            analysis_zarr=target.analysis_zarr,
            model=binding.path,
            config=resolved_repo / "configs" / "fisheye" / "yolo_detect_config.yaml",
            workflow_id=target_label,
            output_dir=target_dir / "detection_artifacts",
        )
        clips = tuple(
            NativeDetectionClipSpec.from_plan_work_unit(
                unit,
                report_path=(
                    target_dir
                    / "detection_reports"
                    / f"{str(unit['clip_id'])}.json"
                ),
            )
            for unit in detection_plan["work_units"]
        )
        if {clip.camera_serial for clip in clips} != {authority.camera_serial}:
            raise ValueError("Detection plan camera differs from acquisition authority.")
        canonical_run_id = safe_component(
            f"detect_native_{target_label}",
            default=f"detect_native_{target_safe}",
            max_length=120,
        )
        _refuse_outputs(
            analysis_zarr=target.analysis_zarr,
            canonical_run_id=canonical_run_id,
            clips=clips,
        )
        module = build_native_detection_fragment(
            NativeDetectionFragmentInputs(
                workflow_id=workflow_id,
                family=FAMILY,
                target_id=target.target_id,
                recording_identity=authority.recording_identity,
                recording_dir=target.recording_dir,
                analysis_zarr=target.analysis_zarr,
                repo=resolved_repo,
                run_root=resolved_run_root,
                canonical_run_id=canonical_run_id,
                n_frames=authority.n_frames,
                source_width=authority.source_width,
                source_height=authority.source_height,
                source_frame_authority=authority.frame,
                source_pixel_authority=authority.pixel,
                producer_version=commit,
                clips=clips,
                model=NativeDetectionModelSpec(
                    set_id=binding.set_id,
                    run_id=binding.run_id,
                    path=binding.path,
                    sha256=binding.sha256,
                ),
                detect_array_concurrency=int(detect_array_concurrency),
            )
        )
        modules.append(module)
        bindings.append(binding)
        detection_plans.append(detection_plan)
        target_plans.append(
            {
                **target.to_json(),
                "target_label": target_label,
                "detection_plan_path": str(detection_plan_path),
                "authority": authority.to_json(),
                "model": binding.to_json(),
                "outputs": module.outputs.to_json(),
            }
        )

    workflow = compose_native_detection_workflow(
        workflow_id=workflow_id,
        family=FAMILY,
        modules=tuple(modules),
    )
    return NativeDetectionCampaignPlan(
        run_label=label,
        workflow_id=workflow_id,
        repo=resolved_repo,
        repo_commit=commit,
        registry=resolved_registry,
        run_root=resolved_run_root,
        targets=tuple(targets),
        target_plans=tuple(target_plans),
        detection_plans=tuple(detection_plans),
        model_bindings=tuple(bindings),
        workflow=workflow,
    )


def materialize_plan_bundle(plan: NativeDetectionCampaignPlan) -> dict[str, Any]:
    payload = plan.to_json()
    plan_path = plan.run_root / "plan.json"
    lsf_path = plan.run_root / "lsf_plan.json"
    if plan_path.exists():
        existing = _read_strict_json(plan_path)
        if existing != payload:
            raise FileExistsError(f"Run root contains a different plan: {plan_path}")
        if not lsf_path.is_file() or _read_strict_json(lsf_path) != plan.workflow.to_json():
            raise FileExistsError(f"Run root has mismatched LSF evidence: {lsf_path}")
        return existing
    for name in ("logs", "status", "progress", "targets", "native_detection"):
        (plan.run_root / name).mkdir(parents=True, exist_ok=True)
    for target, detection_plan in zip(
        plan.target_plans,
        plan.detection_plans,
        strict=True,
    ):
        detection_plan_path = Path(str(target["detection_plan_path"]))
        write_json_snapshot(detection_plan_path, detection_plan)
    write_json_snapshot(plan_path, payload)
    write_json_snapshot(lsf_path, plan.workflow.to_json())
    return payload


def apply_plan(
    plan: NativeDetectionCampaignPlan,
    *,
    runner: CommandRunner,
) -> dict[str, Any]:
    submission_path = plan.run_root / "lsf_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"Submission evidence already exists: {submission_path}")
    materialize_plan_bundle(plan)
    return submit_lsf_workflow(
        plan.workflow,
        cwd=plan.repo,
        plan_path=plan.run_root / "lsf_plan.json",
        submission_path=submission_path,
        runner=runner,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--detection-set-id", required=True)
    parser.add_argument("--detection-run-id", required=True)
    parser.add_argument("--detect-array-concurrency", type=int, default=8)
    parser.add_argument(
        "--submit-host",
        default=os.environ.get("PALETTE_LSF_SUBMIT_HOST", DEFAULT_SUBMIT_HOST),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = load_target_manifest(args.manifest)
    plan = build_plan(
        targets=targets,
        run_label=args.run_label,
        repo=args.repo,
        registry_path=args.registry,
        run_root=args.run_root,
        detection_set_id=args.detection_set_id,
        detection_run_id=args.detection_run_id,
        detect_array_concurrency=args.detect_array_concurrency,
    )
    result = (
        apply_plan(plan, runner=build_ssh_bsub_runner(args.submit_host))
        if args.apply
        else materialize_plan_bundle(plan)
    )
    summary = {
        "status": "submitted" if args.apply else "dry_run",
        "plan_path": str(plan.run_root / "plan.json"),
        "lsf_plan_path": str(plan.run_root / "lsf_plan.json"),
        "submission_path": (
            str(plan.run_root / "lsf_submission.json") if args.apply else None
        ),
        "target_count": len(plan.targets),
        "work_unit_count": sum(
            len(item["outputs"]["artifact_group_paths"])
            for item in plan.target_plans
        ),
        "job_count": len(plan.workflow.jobs),
        "selector_activation": "deferred",
        "registry_update": False,
        "result": result if args.apply else None,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"{summary['status']}: {summary['target_count']} targets, "
            f"{summary['work_unit_count']} work units, {summary['job_count']} jobs"
        )
        print(f"Plan: {summary['plan_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
