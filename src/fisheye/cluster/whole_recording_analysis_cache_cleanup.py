"""Safely remove ephemeral flat ROI caches after a completed analysis DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.flat_roi_cache import DEFAULT_SHARED_CACHE_ROOT
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.flat_roi_cache import (
    FLAT_ROI_CACHE_LAYOUT,
    FLAT_ROI_CACHE_SCHEMA,
)


ANALYSIS_PLAN_SCHEMA = "palette.whole_recording_analysis_bsub_plan.v1"
KEYPOINT_PLAN_SCHEMA = "palette.whole_recording_keypoint_bsub_plan.v2"
CLEANUP_REPORT_SCHEMA = "palette.whole_recording_analysis_cache_cleanup.v1"
DEFAULT_ALLOWED_ROOT = DEFAULT_SHARED_CACHE_ROOT


@dataclass(frozen=True)
class CacheArtifact:
    target_id: str
    manifest_path: Path
    payload_path: Path
    manifest_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "manifest_path": str(self.manifest_path),
            "payload_path": str(self.payload_path),
            "manifest_sha256": self.manifest_sha256,
        }


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return value


def _resolve_under_root(value: object, *, allowed_root: Path, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required.")
    path = Path(text).expanduser().resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"Refusing cache cleanup outside allowed root {allowed_root}: {path}"
        ) from exc
    if path == allowed_root:
        raise ValueError(f"Refusing to treat the allowed root itself as a cache artifact: {path}")
    return path


def _manifest_payload_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    array = _require_mapping(manifest.get("array"), field="cache manifest array")
    raw_path = str(array.get("bin_path") or "").strip()
    if not raw_path:
        raise ValueError(f"Cache manifest is missing array.bin_path: {manifest_path}")
    payload = Path(raw_path).expanduser()
    if not payload.is_absolute():
        payload = manifest_path.parent / payload
    return payload.resolve()


def _canonical_identity(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def repair_cleanup_plan_from_keypoints(
    plan_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Backfill cache cleanup bindings from the immutable keypoint plan.

    Early combined analysis plans omitted cache payload and digest fields even
    though their nested keypoint plans retained the complete reviewed binding.
    This writes a separate repaired plan and never mutates the original.
    """

    source_path = plan_path.expanduser().resolve()
    repaired_path = output_path.expanduser().resolve()
    if repaired_path == source_path:
        raise ValueError("Cleanup plan repair must not overwrite the source plan.")
    analysis = dict(
        _require_mapping(
            json.loads(source_path.read_text(encoding="utf-8")),
            field="analysis plan",
        )
    )
    if analysis.get("schema") != ANALYSIS_PLAN_SCHEMA:
        raise ValueError(f"Unsupported analysis plan: {source_path}")
    keypoint_plan_value = str(analysis.get("keypoint_plan_path") or "").strip()
    if not keypoint_plan_value:
        raise ValueError("Analysis plan is missing keypoint_plan_path.")
    keypoint_plan_path = Path(keypoint_plan_value).expanduser()
    if not keypoint_plan_path.is_absolute():
        keypoint_plan_path = source_path.parent / keypoint_plan_path
    keypoint_plan_path = keypoint_plan_path.resolve()
    keypoint_plan = _require_mapping(
        json.loads(keypoint_plan_path.read_text(encoding="utf-8")),
        field="keypoint plan",
    )
    if keypoint_plan.get("schema") != KEYPOINT_PLAN_SCHEMA:
        raise ValueError(f"Unsupported keypoint plan: {keypoint_plan_path}")
    raw_keypoint_targets = keypoint_plan.get("targets")
    if not isinstance(raw_keypoint_targets, list) or not raw_keypoint_targets:
        raise ValueError("Keypoint plan requires a non-empty targets array.")
    keypoint_targets: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_keypoint_targets, start=1):
        planned = _require_mapping(raw, field=f"keypoint targets[{index}]")
        target = _require_mapping(
            planned.get("target"), field=f"keypoint targets[{index}].target"
        )
        target_id = str(target.get("target_id") or "").strip()
        if not target_id or target_id in keypoint_targets:
            raise ValueError("Keypoint plan target ids must be present and unique.")
        keypoint_targets[target_id] = planned

    raw_analysis_targets = analysis.get("targets")
    if not isinstance(raw_analysis_targets, list) or not raw_analysis_targets:
        raise ValueError("Analysis plan requires a non-empty targets array.")
    repaired_targets: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_analysis_targets, start=1):
        target = dict(_require_mapping(raw, field=f"targets[{index}]"))
        target_id = str(target.get("target_id") or "").strip()
        planned = keypoint_targets.get(target_id)
        if planned is None:
            raise ValueError(
                f"Analysis target {target_id!r} is absent from the keypoint plan."
            )
        keypoint_target = _require_mapping(
            planned.get("target"), field=f"keypoint target {target_id}"
        )
        cache = _require_mapping(
            planned.get("cache"), field=f"keypoint cache {target_id}"
        )
        identity_fields = (
            ("analysis_zarr", keypoint_target.get("analysis_zarr")),
            ("roi_cache_manifest", cache.get("manifest_path")),
            ("crop_run", cache.get("crop_run")),
        )
        for field, expected in identity_fields:
            if str(target.get(field) or "").strip() != str(expected or "").strip():
                raise ValueError(
                    f"Analysis/keypoint plan {field} mismatch for {target_id}."
                )
        target.update(
            {
                "roi_cache_manifest_sha256": cache.get("manifest_sha256"),
                "roi_cache_payload": cache.get("payload_path"),
                "roi_cache_availability": cache.get("availability") or "existing",
                "roi_cache_producer_job_key": cache.get("producer_job_key"),
                "roi_cache_contract": {
                    "crop_signature": cache.get("crop_signature"),
                    "crop_revision": cache.get("crop_revision"),
                    "shape": cache.get("shape"),
                    "total_bytes": cache.get("total_bytes"),
                },
            }
        )
        repaired_targets.append(target)
    if set(keypoint_targets) != {
        str(target.get("target_id") or "") for target in repaired_targets
    }:
        raise ValueError("Analysis and keypoint plans do not contain the same targets.")
    analysis["targets"] = repaired_targets
    analysis["cleanup_plan_repair"] = {
        "source_analysis_plan": str(source_path),
        "source_keypoint_plan": str(keypoint_plan_path),
        "repaired_at_utc": utc_now(),
        "repair_policy": "backfill_exact_cache_binding_from_immutable_keypoint_plan",
    }
    return write_json_snapshot(repaired_path, analysis)


def load_cleanup_artifacts(
    plan_path: Path,
    *,
    allowed_root: Path = DEFAULT_ALLOWED_ROOT,
) -> tuple[CacheArtifact, ...]:
    """Preflight every planned cache before permitting any deletion."""

    resolved_plan = plan_path.expanduser().resolve()
    resolved_root = allowed_root.expanduser().resolve()
    payload = _require_mapping(
        json.loads(resolved_plan.read_text(encoding="utf-8")),
        field="analysis plan",
    )
    if payload.get("schema") != ANALYSIS_PLAN_SCHEMA:
        raise ValueError(
            f"Unsupported analysis plan schema {payload.get('schema')!r}; "
            f"expected {ANALYSIS_PLAN_SCHEMA!r}."
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Analysis plan requires a non-empty targets array.")
    if int(payload.get("target_count") or -1) != len(raw_targets):
        raise ValueError("Analysis plan target_count does not match targets.")

    artifacts: list[CacheArtifact] = []
    seen_manifests: set[Path] = set()
    seen_payloads: set[Path] = set()
    for index, raw_target in enumerate(raw_targets, start=1):
        target = _require_mapping(raw_target, field=f"targets[{index}]")
        target_id = str(target.get("target_id") or "").strip()
        if not target_id:
            raise ValueError(f"targets[{index}].target_id is required.")
        manifest_path = _resolve_under_root(
            target.get("roi_cache_manifest"),
            allowed_root=resolved_root,
            field=f"targets[{index}].roi_cache_manifest",
        )
        planned_payload_path = _resolve_under_root(
            target.get("roi_cache_payload"),
            allowed_root=resolved_root,
            field=f"targets[{index}].roi_cache_payload",
        )
        expected_sha256 = str(target.get("roi_cache_manifest_sha256") or "").strip()
        if expected_sha256 and len(expected_sha256) != 64:
            raise ValueError(
                f"targets[{index}].roi_cache_manifest_sha256 must be null or a SHA-256 digest."
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Planned cache manifest is missing: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                f"Cache manifest digest changed for {target_id}: expected "
                f"{expected_sha256}, got {actual_sha256}."
            )
        manifest = _require_mapping(
            json.loads(manifest_bytes),
            field=f"cache manifest for {target_id}",
        )
        if manifest.get("schema") != FLAT_ROI_CACHE_SCHEMA:
            raise ValueError(f"Unsupported cache schema for {target_id}.")
        if manifest.get("layout") != FLAT_ROI_CACHE_LAYOUT:
            raise ValueError(f"Unsupported cache layout for {target_id}.")
        if not bool(manifest.get("cache_complete")):
            raise ValueError(f"Cache is not marked complete for {target_id}.")
        planned_zarr = str(target.get("analysis_zarr") or "").strip()
        planned_crop = str(target.get("crop_run") or "").strip()
        raw_contract = target.get("roi_cache_contract")
        contract = (
            _require_mapping(
                raw_contract,
                field=f"targets[{index}].roi_cache_contract",
            )
            if raw_contract is not None
            else None
        )
        if not expected_sha256 and (
            not planned_zarr or not planned_crop or contract is None
        ):
            raise ValueError(
                f"Late-bound cache target {target_id} requires analysis_zarr, "
                "crop_run, and roi_cache_contract."
            )
        source = (
            _require_mapping(
                manifest.get("source"),
                field=f"cache manifest source for {target_id}",
            )
            if planned_zarr or planned_crop
            else {}
        )
        if planned_zarr:
            actual_zarr = str(source.get("archive_path") or "").strip()
            if not actual_zarr or Path(actual_zarr).expanduser().resolve() != Path(
                planned_zarr
            ).expanduser().resolve():
                raise ValueError(
                    f"Cache source archive does not match the analysis plan for {target_id}."
                )
        if planned_crop and str(source.get("crop_run_name") or "").strip() != planned_crop:
            raise ValueError(
                f"Cache crop run does not match the analysis plan for {target_id}."
            )
        array = _require_mapping(
            manifest.get("array"), field=f"cache manifest array for {target_id}"
        )
        if contract is not None:
            for field in ("crop_signature", "crop_revision"):
                if _canonical_identity(source.get(field)) != _canonical_identity(
                    contract.get(field)
                ):
                    raise ValueError(
                        f"Cache {field} does not match the analysis plan for {target_id}."
                    )
            planned_shape = contract.get("shape")
            if not isinstance(planned_shape, list) or len(planned_shape) != 3:
                raise ValueError(f"Invalid planned cache shape for {target_id}.")
            if [int(value) for value in array.get("shape") or []] != [
                int(value) for value in planned_shape
            ]:
                raise ValueError(
                    f"Cache shape does not match the analysis plan for {target_id}."
                )
            if int(array.get("total_bytes") or -1) != int(
                contract.get("total_bytes") or -2
            ):
                raise ValueError(
                    f"Cache size contract does not match the analysis plan for {target_id}."
                )
        manifest_payload_path = _resolve_under_root(
            _manifest_payload_path(manifest_path, manifest),
            allowed_root=resolved_root,
            field=f"cache manifest payload for {target_id}",
        )
        if manifest_payload_path != planned_payload_path:
            raise ValueError(
                f"Planned payload does not match cache manifest for {target_id}: "
                f"{planned_payload_path} != {manifest_payload_path}"
            )
        if not planned_payload_path.is_file():
            raise FileNotFoundError(f"Planned cache payload is missing: {planned_payload_path}")
        manifest_total_bytes = int(array.get("total_bytes") or -1)
        if planned_payload_path.stat().st_size != manifest_total_bytes:
            raise ValueError(
                f"Cache payload size does not match its manifest for {target_id}."
            )
        if manifest_path in seen_manifests or planned_payload_path in seen_payloads:
            raise ValueError(f"Analysis targets do not own unique cache artifacts: {target_id}")
        seen_manifests.add(manifest_path)
        seen_payloads.add(planned_payload_path)
        artifacts.append(
            CacheArtifact(
                target_id=target_id,
                manifest_path=manifest_path,
                payload_path=planned_payload_path,
                manifest_sha256=actual_sha256,
            )
        )
    return tuple(artifacts)


def cleanup_roi_caches(
    plan_path: Path,
    *,
    allowed_root: Path = DEFAULT_ALLOWED_ROOT,
    apply: bool = False,
) -> dict[str, Any]:
    """Plan or execute exact-file cleanup, deleting each manifest last."""

    resolved_plan = plan_path.expanduser().resolve()
    resolved_root = allowed_root.expanduser().resolve()
    artifacts = load_cleanup_artifacts(resolved_plan, allowed_root=resolved_root)
    payload_bytes = sum(item.payload_path.stat().st_size for item in artifacts)
    manifest_bytes = sum(item.manifest_path.stat().st_size for item in artifacts)
    deleted: list[str] = []
    if apply:
        for artifact in artifacts:
            artifact.payload_path.unlink()
            deleted.append(str(artifact.payload_path))
        for artifact in artifacts:
            artifact.manifest_path.unlink()
            deleted.append(str(artifact.manifest_path))
        for parent in sorted(
            {item.manifest_path.parent for item in artifacts},
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            current = parent
            while current != resolved_root:
                try:
                    current.rmdir()
                except OSError:
                    break
                current = current.parent

    return {
        "schema": CLEANUP_REPORT_SCHEMA,
        "status": "deleted" if apply else "planned",
        "completed_at_utc": utc_now(),
        "analysis_plan": str(resolved_plan),
        "allowed_root": str(resolved_root),
        "cache_count": len(artifacts),
        "payload_bytes": int(payload_bytes),
        "manifest_bytes": int(manifest_bytes),
        "total_bytes": int(payload_bytes + manifest_bytes),
        "artifacts": [item.to_json() for item in artifacts],
        "deleted_paths": deleted,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--allowed-root", type=Path, default=DEFAULT_ALLOWED_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--repair-plan-output",
        type=Path,
        help=(
            "Write a separate cleanup plan with cache bindings recovered from "
            "the immutable keypoint plan, then clean using that repaired plan."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    effective_plan = args.plan
    if args.repair_plan_output is not None:
        repair_cleanup_plan_from_keypoints(
            args.plan,
            output_path=args.repair_plan_output,
        )
        effective_plan = args.repair_plan_output
    report = cleanup_roi_caches(
        effective_plan,
        allowed_root=args.allowed_root,
        apply=bool(args.apply),
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_PLAN_SCHEMA",
    "CLEANUP_REPORT_SCHEMA",
    "DEFAULT_ALLOWED_ROOT",
    "CacheArtifact",
    "cleanup_roi_caches",
    "load_cleanup_artifacts",
    "main",
    "repair_cleanup_plan_from_keypoints",
]
