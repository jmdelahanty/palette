"""Validate whole-recording keypoint-v2 production candidates.

This terminal DAG stage deliberately does not update the registry or any Zarr
selector.  Candidate activation is a separate reviewed operation after the
first production-shaped canary has passed its consumer gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import zarr

from fisheye.cluster.keypoints.whole_recording import PLAN_SCHEMA
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.registry.db import Registry
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.zarr.body_frame_manifest import (
    BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
    validate_body_frame_run_manifest,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_quality_manifest import (
    KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE,
    validate_keypoint_quality_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent
from fisheye.utils.finalize_whole_recording_keypoint_v2 import (
    WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_ID,
    WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_VERSION,
)


REPORT_SCHEMA = "palette.whole_recording_keypoint_candidate_validator.v2"
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "approved_run",
)
_OWNER_RE = re.compile(r"^[0-9a-f]{32}$")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Plan field {field!r} must be an object.")
    return dict(value)


def _check_registry_integrity(registry_path: Path) -> str:
    registry = Registry(registry_path)
    registry.conn.execute("PRAGMA busy_timeout=60000;")
    try:
        return str(registry.conn.execute("PRAGMA integrity_check;").fetchone()[0])
    finally:
        registry.close()


def _target_rows(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_targets = plan.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Whole-recording plan has no targets.")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(raw_targets, start=1):
        planned = _mapping(raw, field=f"targets[{index}]")
        target = _mapping(planned.get("target"), field=f"targets[{index}].target")
        run_names = _mapping(
            planned.get("run_names"), field=f"targets[{index}].run_names"
        )
        model = _mapping(planned.get("model"), field=f"targets[{index}].model")
        cache = _mapping(planned.get("cache"), field=f"targets[{index}].cache")
        row = {
            "target_id": str(target.get("target_id") or ""),
            "recording_id": str(target.get("recording_id") or ""),
            "analysis_zarr": str(target.get("analysis_zarr") or ""),
            "crop_run": str(cache.get("crop_run") or ""),
            "model_set_id": str(model.get("set_id") or ""),
            "model_run_id": str(model.get("run_id") or ""),
            "model_sha256": str(model.get("model_sha256") or ""),
            "raw_run": str(run_names.get("keypoint_run") or ""),
            "quality_run": str(run_names.get("keypoint_quality_run") or ""),
            "refined_run": str(run_names.get("refined_keypoint_run") or ""),
            "body_frame_run": str(run_names.get("body_frame_run") or ""),
            "finalization_result": str(planned.get("finalization_result") or ""),
        }
        if any(not value for value in row.values()):
            raise ValueError(f"Whole-recording plan target {index} is incomplete: {row}")
        rows.append(row)
    return rows


def _require_candidate(
    root: Any,
    *,
    archive: Path,
    parent_path: str,
    run_id: str,
    manifest_attr: str,
    validate_manifest: Callable[[Mapping[str, Any]], tuple[str, ...]],
) -> Mapping[str, Any]:
    parent = root[parent_path]
    if run_id not in parent:
        raise RuntimeError(f"Missing planned candidate {parent_path}/{run_id}")
    run = parent[run_id]
    if not is_run_complete_in_parent(parent, run):
        raise RuntimeError(f"Candidate is not strictly complete: {parent_path}/{run_id}")
    expected_attrs = {
        "status": "complete",
        "stage_selector_eligible": False,
        "shadow_only": False,
        "production_candidate": True,
        "production_selector_activation": "deferred_separate_reviewed_change",
    }
    mismatches = {
        name: (run.attrs.get(name), expected)
        for name, expected in expected_attrs.items()
        if run.attrs.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Candidate lifecycle mismatch at {parent_path}/{run_id}: {mismatches}"
        )
    owner = run.attrs.get("atomic_publication_owner_uuid")
    if not isinstance(owner, str) or _OWNER_RE.fullmatch(owner) is None:
        raise RuntimeError(
            f"Candidate lacks exact atomic publication ownership: {parent_path}/{run_id}"
        )
    selected = [name for name in _SELECTOR_ATTRS if parent.attrs.get(name) == run_id]
    if selected:
        raise RuntimeError(
            f"Selector-ineligible candidate {parent_path}/{run_id} is selected by {selected}."
        )
    manifest = run.attrs.get(manifest_attr)
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"Candidate lacks {manifest_attr}: {parent_path}/{run_id}")
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError(
            f"Candidate manifest is invalid at {parent_path}/{run_id}: "
            + "; ".join(errors)
        )
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping) or payload.get("run_id") != run_id:
        raise RuntimeError(f"Candidate manifest binds a different run at {parent_path}/{run_id}")
    validate_direct_consolidated_subtree(
        archive, subtree_path=f"{parent_path}/{run_id}"
    )
    return manifest


def _require_finalization_receipt(
    target: Mapping[str, str], *, expected_runs: Mapping[str, str]
) -> Mapping[str, Any]:
    path = Path(target["finalization_result"]).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Finalization receipt is not an object: {path}")
    expected = {
        "schema_id": WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_ID,
        "schema_version": WHOLE_RECORDING_KEYPOINT_FINALIZATION_SCHEMA_VERSION,
        "status": "complete",
        "analysis_zarr": str(Path(target["analysis_zarr"]).expanduser().resolve()),
        "runs": dict(expected_runs),
        "selector_eligible": False,
        "selector_activation": "deferred_separate_reviewed_change",
        "registry_updated": False,
    }
    mismatches = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Finalization receipt mismatch at {path}: {mismatches}")
    publication = value.get("publication")
    if not isinstance(publication, Mapping) or publication.get(
        "public_validation_errors"
    ) != []:
        raise RuntimeError(f"Finalization receipt lacks a clean public gate: {path}")
    return value


def _validate_target(target: Mapping[str, str]) -> dict[str, Any]:
    archive = Path(target["analysis_zarr"]).expanduser().resolve()
    root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    raw = _require_candidate(
        root,
        archive=archive,
        parent_path="keypoints_runs",
        run_id=target["raw_run"],
        manifest_attr=KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        validate_manifest=validate_keypoint_run_manifest,
    )
    quality = _require_candidate(
        root,
        archive=archive,
        parent_path="keypoint_quality_runs",
        run_id=target["quality_run"],
        manifest_attr=KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE,
        validate_manifest=validate_keypoint_quality_run_manifest,
    )
    refined = _require_candidate(
        root,
        archive=archive,
        parent_path="refined_keypoints_runs",
        run_id=target["refined_run"],
        manifest_attr=REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        validate_manifest=validate_refined_keypoint_run_manifest,
    )
    body = _require_candidate(
        root,
        archive=archive,
        parent_path="analysis/body_frame_runs",
        run_id=target["body_frame_run"],
        manifest_attr=BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
        validate_manifest=validate_body_frame_run_manifest,
    )
    raw_payload = raw["payload"]
    quality_payload = quality["payload"]
    refined_payload = refined["payload"]
    body_payload = body["payload"]
    pose = raw_payload["pose_model_schema_binding"]
    model = pose["model"]
    if (
        model.get("registry_set_id") != target["model_set_id"]
        or model.get("registry_run_id") != target["model_run_id"]
        or model.get("sha256") != target["model_sha256"]
    ):
        raise RuntimeError("Raw keypoint manifest binds a different exact model.")
    crop_source = raw_payload["source_crop_snapshot"]
    if crop_source.get("run_id") != target["crop_run"]:
        raise RuntimeError("Raw keypoint manifest binds a different crop run.")
    raw_digest = canonical_json_sha256(raw)
    quality_digest = canonical_json_sha256(quality)
    refined_digest = canonical_json_sha256(refined)
    quality_source = quality_payload["source_keypoint_snapshot"]
    if (
        quality_source.get("run_name") != target["raw_run"]
        or quality_source.get("manifest_digest") != raw_digest
    ):
        raise RuntimeError("Quality candidate does not bind the exact raw candidate.")
    sources = refined_payload["source_bindings"]
    if (
        sources["raw_keypoint_snapshot"].get("run_id") != target["raw_run"]
        or sources["raw_keypoint_snapshot"].get("manifest_digest") != raw_digest
        or sources["quality_snapshot"].get("run_id") != target["quality_run"]
        or sources["quality_snapshot"].get("manifest_digest") != quality_digest
        or sources["crop_snapshot"].get("run_id") != target["crop_run"]
    ):
        raise RuntimeError("Refined candidate source bindings do not match its chain.")
    body_source = body_payload["source_keypoint_snapshot"]
    if (
        body_source.get("run_name") != target["refined_run"]
        or body_source.get("manifest_digest") != refined_digest
    ):
        raise RuntimeError("Body-frame candidate does not bind the exact refined candidate.")
    runs = {
        "raw_keypoints": target["raw_run"],
        "keypoint_quality": target["quality_run"],
        "refined_keypoints": target["refined_run"],
        "body_frame": target["body_frame_run"],
    }
    receipt = _require_finalization_receipt(target, expected_runs=runs)
    return {
        **dict(target),
        "runs": runs,
        "manifest_digests": {
            "raw_keypoints": raw_digest,
            "keypoint_quality": quality_digest,
            "refined_keypoints": refined_digest,
            "body_frame": canonical_json_sha256(body),
        },
        "finalization_receipt": str(target["finalization_result"]),
        "terminal_receipt_digest": receipt.get("terminal_receipt_digest"),
        "registry_status": "unchanged_candidate_validated",
        "selector_status": "unchanged_candidate_ineligible",
    }


def finalize_registry(
    run_root: Path,
    *,
    registry_path: Path,
    apply: bool,
) -> dict[str, Any]:
    if apply:
        raise RuntimeError(
            "Whole-recording keypoint-v2 candidates cannot be activated by the DAG. "
            "Run the separate reviewed selector/registry activation gate after canary approval."
        )
    resolved_run_root = run_root.expanduser().resolve()
    resolved_registry = registry_path.expanduser().resolve()
    plan_path = resolved_run_root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported whole-recording plan schema in {plan_path}.")
    targets = _target_rows(plan)
    integrity_before = _check_registry_integrity(resolved_registry)
    if integrity_before != "ok":
        raise RuntimeError(f"Registry integrity_check failed: {integrity_before}")
    validated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for target in targets:
        try:
            validated.append(_validate_target(target))
        except Exception as exc:
            errors.append(
                {
                    "target_id": target.get("target_id"),
                    "analysis_zarr": target.get("analysis_zarr"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    integrity_after = _check_registry_integrity(resolved_registry)
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok" if not errors and len(validated) == len(targets) else "error",
        "apply": False,
        "activation_performed": False,
        "run_root": str(resolved_run_root),
        "plan_path": str(plan_path),
        "registry_path": str(resolved_registry),
        "target_count": len(targets),
        "validated_count": len(validated),
        "registry_integrity_before": integrity_before,
        "registry_integrity_after": integrity_after,
        "finished_at_utc": utc_now(),
        "errors": errors,
        "validated": validated,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Forbidden for v2 candidates; activation is a separate reviewed gate.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = finalize_registry(
        args.run_root, registry_path=args.registry, apply=bool(args.apply)
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json.expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "finalize_registry", "main"]
