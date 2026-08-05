"""Freeze or apply a crop-v2 production-candidate cohort plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.crop_defaults import (
    DEFAULT_ZEBRAFISH_CROP_PURPOSE,
    DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
    crop_geometry_policy_from_manifest,
)
from fisheye.shared.zarr.crop_snapshot_publication import (
    publish_crop_geometry_production_candidate,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_authority_activation import (
    inspect_active_refined_detection_authority,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from fisheye.utils.activate_refined_detection_authority_batch import (
    validate_plan as validate_activation_plan,
)
from fisheye.utils.preflight_refined_detection_crops import (
    inspect_refined_detection_crop_preflight,
)


PLAN_SCHEMA_ID = "palette.crop_geometry.production_candidate_batch_plan"
PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.crop_geometry.production_candidate_batch_result"
RESULT_SCHEMA_VERSION = 1
_PLAN_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "source_activation_plan_digest",
        "crop_run_id",
        "policy",
        "storage_profile_id",
        "candidate_count",
        "candidates",
        "digest_algorithm",
        "plan_digest",
    }
)


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _plan_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"digest_algorithm", "plan_digest"}
    }


def _camera_identity(preflight: Mapping[str, Any]) -> str:
    pixel = preflight.get("pixel_authority")
    authority = pixel.get("authority") if isinstance(pixel, Mapping) else None
    camera = authority.get("camera_identity") if isinstance(authority, Mapping) else None
    if not isinstance(camera, str) or not camera.strip() or camera != camera.strip():
        raise ValueError("Crop preflight lacks one exact camera identity.")
    return camera


def _require_candidate_matches_activation(
    active: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    expected = {
        "analysis_zarr": frozen["analysis_zarr"],
        "recording_identity": frozen["recording_identity"],
        "run_id": frozen["run_id"],
        "manifest_digest": frozen["activation_manifest_digest"],
        "logical_content_digest": frozen["logical_content_digest"],
        "publication_owner_uuid": frozen["publication_owner_uuid"],
        "storage_profile_id": frozen["storage_profile_id"],
    }
    observed = {key: active.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            "Active refined authority differs from the frozen activation plan."
        )


def validate_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if set(plan) != _PLAN_FIELDS:
        errors.append("crop candidate plan field set is not exact")
    if plan.get("schema_id") != PLAN_SCHEMA_ID:
        errors.append("crop candidate plan schema_id is invalid")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        errors.append("crop candidate plan schema_version is invalid")
    if plan.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("crop candidate plan digest_algorithm is invalid")
    try:
        policy = crop_geometry_policy_from_manifest(plan.get("policy"))
    except (TypeError, ValueError) as exc:
        errors.append(f"crop candidate policy is invalid: {exc}")
        policy = None
    if plan.get("storage_profile_id") != PUBLISHED_HTTP_V1.profile_id:
        errors.append("crop candidate storage profile is invalid")
    run_id = plan.get("crop_run_id")
    if not isinstance(run_id, str) or not run_id or "/" in run_id:
        errors.append("crop candidate run id is invalid")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        errors.append("crop candidate candidates must be a list")
        candidates = []
    if plan.get("candidate_count") != len(candidates):
        errors.append("crop candidate_count differs from candidates")
    paths: list[str] = []
    identities: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "analysis_zarr",
            "recording_identity",
            "camera_identity",
            "active_authority",
            "preflight",
        }:
            errors.append(f"crop candidate {index} field set is not exact")
            continue
        active = candidate.get("active_authority")
        preflight = candidate.get("preflight")
        if not isinstance(active, Mapping) or not isinstance(preflight, Mapping):
            errors.append(f"crop candidate {index} evidence is invalid")
            continue
        path = str(candidate.get("analysis_zarr") or "")
        identity = str(candidate.get("recording_identity") or "")
        camera = str(candidate.get("camera_identity") or "")
        if active.get("analysis_zarr") != path or preflight.get("analysis_zarr") != path:
            errors.append(f"crop candidate {index} archive binding differs")
        if active.get("recording_identity") != identity:
            errors.append(f"crop candidate {index} recording binding differs")
        if preflight.get("selection_mode") != "approved_authoritative_refined_v1":
            errors.append(f"crop candidate {index} selection mode is invalid")
        try:
            if _camera_identity(preflight) != camera:
                errors.append(f"crop candidate {index} camera binding differs")
        except ValueError as exc:
            errors.append(f"crop candidate {index} {exc}")
        if policy is not None and preflight.get("policy") != policy.as_manifest():
            errors.append(f"crop candidate {index} policy binding differs")
        paths.append(path)
        identities.append(identity)
    if len(paths) != len(set(paths)):
        errors.append("crop candidate paths must be unique")
    if len(identities) != len(set(identities)):
        errors.append("crop candidate recording identities must be unique")
    if plan.get("plan_digest") != canonical_json_sha256(_plan_payload(plan)):
        errors.append("crop candidate plan_digest differs from its payload")
    return tuple(errors)


def build_plan(
    *,
    activation_plan: Mapping[str, Any],
    crop_run_id: str,
    policy: CropGeometryPolicy,
) -> dict[str, Any]:
    activation_errors = validate_activation_plan(activation_plan)
    if activation_errors:
        raise ValueError(
            "Refined activation plan is invalid: " + "; ".join(activation_errors)
        )
    run_id = str(crop_run_id).strip()
    if not run_id or "/" in run_id:
        raise ValueError("crop_run_id must be one safe child-group name.")
    candidates: list[dict[str, Any]] = []
    for source in activation_plan["candidates"]:
        archive = Path(str(source["analysis_zarr"])).expanduser().resolve()
        if (archive / "crop_runs" / run_id).exists():
            raise FileExistsError(f"Immutable crop target already exists: {archive}")
        active = inspect_active_refined_detection_authority(
            analysis_zarr=archive,
            run_id=str(activation_plan["run_id"]),
        )
        _require_candidate_matches_activation(active, source["inspection"])
        preflight = inspect_refined_detection_crop_preflight(
            analysis_zarr=archive,
            policy=policy,
            max_examples=0,
        )
        if preflight["refined_run_id"] != activation_plan["run_id"]:
            raise RuntimeError("Crop preflight selected a different refined authority.")
        camera = _camera_identity(preflight)
        candidates.append(
            {
                "analysis_zarr": str(archive),
                "recording_identity": active["recording_identity"],
                "camera_identity": camera,
                "active_authority": active,
                "preflight": preflight,
            }
        )
    candidates.sort(key=lambda item: str(item["analysis_zarr"]))
    payload: dict[str, Any] = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_activation_plan_digest": activation_plan["plan_digest"],
        "crop_run_id": run_id,
        "policy": policy.as_manifest(),
        "storage_profile_id": PUBLISHED_HTTP_V1.profile_id,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    plan = {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }
    errors = validate_plan(plan)
    if errors:
        raise RuntimeError("Generated crop candidate plan is invalid: " + "; ".join(errors))
    return plan


def apply_plan(
    plan: Mapping[str, Any],
    *,
    scratch_root: Path,
    receipt_root: Path,
    only_recording_identities: frozenset[str] = frozenset(),
    exclude_recording_identities: frozenset[str] = frozenset(),
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    errors = validate_plan(plan)
    if errors:
        raise ValueError("Refusing invalid crop candidate plan: " + "; ".join(errors))
    overlap = only_recording_identities & exclude_recording_identities
    if overlap:
        raise ValueError(f"Recordings cannot be selected and excluded: {overlap!r}")
    selected = []
    for candidate in plan["candidates"]:
        identity = str(candidate["recording_identity"])
        if only_recording_identities and identity not in only_recording_identities:
            continue
        if identity in exclude_recording_identities:
            continue
        selected.append(candidate)
    if only_recording_identities:
        observed = {str(item["recording_identity"]) for item in selected}
        missing = sorted(only_recording_identities - observed)
        if missing:
            raise ValueError(f"Requested recordings are absent: {missing!r}")
    if not selected:
        raise ValueError("Crop candidate plan selection contains no candidates.")

    policy = crop_geometry_policy_from_manifest(plan["policy"])
    receipt_root = receipt_root.expanduser().resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for candidate in selected:
        archive = Path(str(candidate["analysis_zarr"]))
        active = inspect_active_refined_detection_authority(
            analysis_zarr=archive,
            run_id=str(candidate["active_authority"]["run_id"]),
        )
        if active != candidate["active_authority"]:
            raise RuntimeError(
                f"Active refined authority drifted for {candidate['recording_identity']}."
            )
        preflight = inspect_refined_detection_crop_preflight(
            analysis_zarr=archive,
            policy=policy,
            max_examples=0,
        )
        if preflight != candidate["preflight"]:
            raise RuntimeError(
                f"Crop preflight drifted for {candidate['recording_identity']}."
            )
        result = publish_crop_geometry_production_candidate(
            analysis_zarr=archive,
            run_id=str(plan["crop_run_id"]),
            policy=policy,
            expected_camera_identity=str(candidate["camera_identity"]),
            scratch_root=scratch_root,
            profile=PUBLISHED_HTTP_V1,
            copy_backend=copy_backend,
            keep_scratch=keep_scratch,
        )
        write_json_atomic(
            receipt_root / f"{candidate['recording_identity']}.json",
            result,
        )
        results.append(result)
    return {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "plan_digest": plan["plan_digest"],
        "requested_candidate_count": len(selected),
        "completed_candidate_count": len(results),
        "receipts": [
            str(receipt_root / f"{item['recording_identity']}.json")
            for item in selected
        ],
        "selector_activation": "none",
        "registry_updated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--activation-plan", type=Path)
    parser.add_argument("--crop-run")
    parser.add_argument("--purpose", default=DEFAULT_ZEBRAFISH_CROP_PURPOSE)
    parser.add_argument("--roi-width", type=int, default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX)
    parser.add_argument("--roi-height", type=int, default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--only-recording", action="append", default=[])
    parser.add_argument("--exclude-recording", action="append", default=[])
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply:
            if args.scratch_root is None or args.receipt_root is None:
                raise ValueError("--apply requires --scratch-root and --receipt-root.")
            plan = _strict_json_load(args.plan_json.expanduser().resolve())
            result = apply_plan(
                plan,
                scratch_root=args.scratch_root,
                receipt_root=args.receipt_root,
                only_recording_identities=frozenset(args.only_recording),
                exclude_recording_identities=frozenset(args.exclude_recording),
                copy_backend=args.copy_backend,
                keep_scratch=bool(args.keep_scratch),
            )
        else:
            if args.activation_plan is None or not args.crop_run:
                raise ValueError(
                    "Plan construction requires --activation-plan and --crop-run."
                )
            activation_plan = _strict_json_load(
                args.activation_plan.expanduser().resolve()
            )
            policy = CropGeometryPolicy(
                purpose=args.purpose,
                size_mode=CropSizeMode.FIXED_PER_RUN,
                fixed_size_wh=(int(args.roi_width), int(args.roi_height)),
                padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
            )
            plan = build_plan(
                activation_plan=activation_plan,
                crop_run_id=args.crop_run,
                policy=policy,
            )
            write_json_atomic(args.plan_json.expanduser().resolve(), plan)
            result = {
                "schema_id": RESULT_SCHEMA_ID,
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "plan_frozen",
                "completed_at_utc": utc_now(),
                "plan_digest": plan["plan_digest"],
                "candidate_count": plan["candidate_count"],
                "selector_activation": "none",
                "registry_updated": False,
            }
        if args.result_json is not None:
            write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

