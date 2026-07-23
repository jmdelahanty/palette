#!/usr/bin/env python3
"""Finalize keypoint refinement artifacts for approved refined-keypoint runs.

Default behavior is dry-run. Use --apply to write visualization artifacts.
"""

from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use
from fisheye.shared.json_safety import write_json_atomic as _write_json_report
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.visualization.visualize_keypoint_quality import (
    QUALITY_ARTIFACT_NAME,
    REFINEMENT_PIPELINE_ARTIFACT_NAME,
    render_keypoint_quality_png,
    render_keypoint_refinement_pipeline_png,
)


REFINED_PARENT_NAMES = ("refined_keypoints_runs", "keypoints_refined_runs")
SIGNATURE_VERSION = 1

ARTIFACT_SPECS = {
    QUALITY_ARTIFACT_NAME: {
        "path": f"visualizations/{QUALITY_ARTIFACT_NAME}",
        "description": "Keypoint quality overview PNG",
    },
    REFINEMENT_PIPELINE_ARTIFACT_NAME: {
        "path": f"visualizations/{REFINEMENT_PIPELINE_ARTIFACT_NAME}",
        "description": "Keypoint refinement pipeline overview PNG",
    },
}


@dataclass
class FinalizeRow:
    zarr_path: str
    zarr_use: str
    parent_name: Optional[str]
    refined_run: Optional[str]
    review_state: Optional[str]
    review_intended_use: Optional[str]
    review_method: Optional[str]
    review_timestamp_utc: Optional[str]
    review_reviewer: Optional[str]
    review_notes: Optional[str]
    source_keypoints_run: Optional[str]
    source_crop_run: Optional[str]
    source_detect_run: Optional[str]
    artifact_signature: Optional[str]
    status: str
    reason: str
    action: str


_utc_now = utc_now


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _decode_text_lower(value: object) -> Optional[str]:
    text = _decode_text(value)
    return text.lower() if text else None


def _coerce_mapping(value: object) -> Optional[Dict[str, object]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    if hasattr(value, "items"):
        try:
            return dict(value)  # type: ignore[arg-type]
        except Exception:
            return None
    return None


def _resolve_roots(paths: List[Path]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _read_zarr_attrs(zarr_json_path: Path) -> Dict[str, object]:
    if not zarr_json_path.exists():
        return {}
    try:
        payload = json.loads(zarr_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _resolve_refined_parent(zarr_path: Path) -> tuple[Optional[Path], Optional[str]]:
    for parent_name in REFINED_PARENT_NAMES:
        parent_path = zarr_path / parent_name
        if parent_path.exists() and (parent_path / "zarr.json").exists():
            return parent_path, parent_name
    return None, None


def _resolve_review_timestamp(review_status: Dict[str, object]) -> Optional[str]:
    for key in ("timestamp_utc", "timestamp", "reviewed_at_utc", "reviewed_at", "updated_utc"):
        text = _decode_text(review_status.get(key))
        if text:
            return text
    return None


def _build_artifact_signature(row: FinalizeRow, *, visuals_dpi: int) -> str:
    payload = {
        "signature_version": SIGNATURE_VERSION,
        "artifact_names": sorted(ARTIFACT_SPECS.keys()),
        "refined_run": row.refined_run,
        "source_keypoints_run": row.source_keypoints_run,
        "source_crop_run": row.source_crop_run,
        "source_detect_run": row.source_detect_run,
        "review_state": row.review_state,
        "review_intended_use": row.review_intended_use,
        "review_method": row.review_method,
        "review_timestamp_utc": row.review_timestamp_utc,
        "visuals_dpi": int(visuals_dpi),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_existing_artifact_signature(run_path: Path) -> Optional[str]:
    signatures: List[str] = []
    for artifact_name in ARTIFACT_SPECS:
        artifact_attrs = _read_zarr_attrs(run_path / "visualizations" / artifact_name / "zarr.json")
        signature = _decode_text(artifact_attrs.get("artifact_signature"))
        if signature is None:
            return None
        signatures.append(signature)
    if not signatures:
        return None
    if len(set(signatures)) != 1:
        return None
    return signatures[0]


def _build_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    approved_state: str,
    required_intended_use: Optional[str],
    force: bool,
    visuals_dpi: int,
    refined_run: Optional[str],
    allow_legacy_latest_fallback: bool,
) -> List[FinalizeRow]:
    rows: List[FinalizeRow] = []
    approved_state_norm = approved_state.strip().lower()
    required_use_norm = required_intended_use.strip().lower() if required_intended_use else None

    for zarr_path in _iter_zarr(roots, recursive=recursive):
        root_attrs = _read_zarr_attrs(zarr_path / "zarr.json")
        zarr_use = infer_zarr_use(root_attrs, zarr_path, default="unknown")
        if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
            continue

        row = FinalizeRow(
            zarr_path=str(zarr_path),
            zarr_use=zarr_use,
            parent_name=None,
            refined_run=None,
            review_state=None,
            review_intended_use=None,
            review_method=None,
            review_timestamp_utc=None,
            review_reviewer=None,
            review_notes=None,
            source_keypoints_run=None,
            source_crop_run=None,
            source_detect_run=None,
            artifact_signature=None,
            status="skip",
            reason="unknown",
            action="skip",
        )

        parent_path, parent_name = _resolve_refined_parent(zarr_path)
        if parent_path is None or parent_name is None:
            row.reason = "no_refined_keypoint_runs"
            rows.append(row)
            continue
        row.parent_name = parent_name

        if refined_run is not None:
            row.refined_run = refined_run
        elif allow_legacy_latest_fallback:
            parent_attrs = _read_zarr_attrs(parent_path / "zarr.json")
            latest_refined_run = _decode_text(parent_attrs.get("latest"))
            if latest_refined_run is None:
                row.reason = "no_latest_refined_run"
                rows.append(row)
                continue
            row.refined_run = latest_refined_run
        else:
            row.reason = "exact_refined_run_required"
            rows.append(row)
            continue

        assert row.refined_run is not None
        run_path = parent_path / row.refined_run
        run_attrs = _read_zarr_attrs(run_path / "zarr.json")
        if not run_attrs:
            row.reason = "missing_refined_run_attrs"
            rows.append(row)
            continue

        review_status = _coerce_mapping(run_attrs.get("keypoint_review_status"))
        if review_status is None:
            row.reason = "no_keypoint_review_status"
            rows.append(row)
            continue

        row.review_state = _decode_text(review_status.get("state"))
        row.review_intended_use = _decode_text(review_status.get("intended_use"))
        row.review_method = _decode_text(review_status.get("method"))
        row.review_reviewer = _decode_text(review_status.get("reviewer"))
        row.review_notes = _decode_text(review_status.get("notes"))
        row.review_timestamp_utc = _resolve_review_timestamp(review_status)

        review_state_norm = row.review_state.strip().lower() if row.review_state else None
        if review_state_norm is None:
            row.reason = "review_state_missing"
            rows.append(row)
            continue
        if review_state_norm != approved_state_norm:
            row.reason = "review_state_not_approved"
            rows.append(row)
            continue

        if required_use_norm is not None:
            intended_use_norm = row.review_intended_use.strip().lower() if row.review_intended_use else None
            if intended_use_norm is None:
                row.reason = "review_intended_use_missing"
                rows.append(row)
                continue
            if intended_use_norm != required_use_norm:
                row.reason = "review_intended_use_mismatch"
                rows.append(row)
                continue

        row.source_keypoints_run = (
            _decode_text(run_attrs.get("source_keypoints_run"))
            or _decode_text(run_attrs.get("source_keypoint_run"))
        )
        row.source_crop_run = _decode_text(run_attrs.get("source_crop_run"))
        row.source_detect_run = _decode_text(run_attrs.get("source_detect_run"))

        row.status = "ok"
        row.reason = "eligible"
        row.artifact_signature = _build_artifact_signature(row, visuals_dpi=visuals_dpi)
        existing_signature = _read_existing_artifact_signature(run_path)
        if (not force) and existing_signature and existing_signature == row.artifact_signature:
            row.action = "unchanged"
        else:
            row.action = "render"
        rows.append(row)

    rows.sort(key=lambda row: row.zarr_path)
    return rows


def _write_png_artifact(
    *,
    vis_group: zarr.Group,
    artifact_name: str,
    png_bytes: bytes,
    base_attrs: Dict[str, object],
    extra_attrs: Dict[str, object],
) -> None:
    if artifact_name in vis_group:
        del vis_group[artifact_name]

    data = np.frombuffer(png_bytes, dtype=np.uint8)
    chunk = max(1, min(int(data.shape[0]), 1_048_576))
    dataset = vis_group.create_array(
        artifact_name,
        data=data,
        chunks=(chunk,),
        overwrite=True,
    )
    dataset.attrs.update(base_attrs)
    dataset.attrs.update(extra_attrs)


def _finalize_row(row: FinalizeRow, *, visuals_dpi: int, force: bool) -> str:
    if row.parent_name is None or row.refined_run is None:
        raise RuntimeError("missing refined run context")

    root = zarr.open_group(row.zarr_path, mode="a")
    if row.parent_name not in root:
        raise RuntimeError(f"missing parent group: {row.parent_name}")
    parent_group = root[row.parent_name]
    if row.refined_run not in parent_group:
        raise RuntimeError(f"missing refined run: {row.refined_run}")
    refined_group = parent_group[row.refined_run]

    if "visualizations" in refined_group:
        existing_signature = _read_existing_artifact_signature(
            Path(row.zarr_path) / row.parent_name / row.refined_run
        )
    else:
        existing_signature = None

    if (not force) and row.artifact_signature and existing_signature == row.artifact_signature:
        return "unchanged"

    quality_png_bytes, quality_meta = render_keypoint_quality_png(
        row.zarr_path,
        refined_run=row.refined_run,
        dpi=visuals_dpi,
        show=False,
    )
    refinement_png_bytes, refinement_meta = render_keypoint_refinement_pipeline_png(
        row.zarr_path,
        refined_run=row.refined_run,
        dpi=visuals_dpi,
        show=False,
    )
    expected_verification = "legacy_unverified_diagnostic"
    if (
        quality_meta.get("coordinate_verification") != expected_verification
        or refinement_meta.get("coordinate_verification") != expected_verification
        or quality_meta.get("source_camera_overlay_authority") is not False
        or refinement_meta.get("source_camera_overlay_authority") is not False
    ):
        raise RuntimeError(
            "Refusing to persist refined-keypoint diagnostics without explicit "
            "unverified-coordinate and no-overlay-authority metadata."
        )

    vis_group = refined_group.require_group("visualizations")
    finalized_at = _utc_now()
    summary_stats = quality_meta.get("summary_statistics")
    usable_keypoints = None
    if isinstance(summary_stats, dict):
        usable_keypoints = summary_stats.get("usable_keypoints")

    base_attrs = {
        "mime": "image/png",
        "artifact_signature": row.artifact_signature,
        "artifact_signature_version": SIGNATURE_VERSION,
        "finalized_at_utc": finalized_at,
        "finalized_by": "finalize_keypoint_refinement_artifacts",
        "review_state": row.review_state,
        "review_intended_use": row.review_intended_use,
        "review_timestamp_utc": row.review_timestamp_utc,
        "source_keypoints_run": row.source_keypoints_run,
        "source_crop_run": row.source_crop_run,
        "source_detect_run": row.source_detect_run,
        "coordinate_verification": expected_verification,
        "source_camera_overlay_authority": False,
    }

    _write_png_artifact(
        vis_group=vis_group,
        artifact_name=QUALITY_ARTIFACT_NAME,
        png_bytes=quality_png_bytes,
        base_attrs=base_attrs,
        extra_attrs={
            "description": str(ARTIFACT_SPECS[QUALITY_ARTIFACT_NAME]["description"]),
            "usable_keypoints": usable_keypoints,
        },
    )
    _write_png_artifact(
        vis_group=vis_group,
        artifact_name=REFINEMENT_PIPELINE_ARTIFACT_NAME,
        png_bytes=refinement_png_bytes,
        base_attrs=base_attrs,
        extra_attrs={
            "description": str(ARTIFACT_SPECS[REFINEMENT_PIPELINE_ARTIFACT_NAME]["description"]),
            "render_refined_run": _decode_text(refinement_meta.get("refined_run")),
        },
    )

    manifest = _coerce_mapping(refined_group.attrs.get("visualizations")) or {}
    for artifact_name, spec in ARTIFACT_SPECS.items():
        manifest[artifact_name] = {
            "path": spec["path"],
            "description": spec["description"],
            "artifact_signature": row.artifact_signature,
            "finalized_at_utc": finalized_at,
            "coordinate_verification": expected_verification,
            "source_camera_overlay_authority": False,
        }
    refined_group.attrs["visualizations"] = manifest
    return "rendered"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize keypoint refinement artifacts (quality + refinement-pipeline visualizations) "
            "for approved refined-keypoint runs."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter archives by zarr use (default: any).",
    )
    parser.add_argument(
        "--approved-state",
        default="approved",
        help="Keypoint review state treated as approved (default: approved).",
    )
    parser.add_argument(
        "--required-intended-use",
        choices=["training", "full_recording"],
        help="Optional keypoint review intended_use requirement.",
    )
    parser.add_argument(
        "--visuals-dpi",
        type=int,
        default=150,
        help="DPI for rendered finalized PNG artifacts (default: 150).",
    )
    parser.add_argument("--force", action="store_true", help="Re-render even when artifact signature is unchanged.")
    parser.add_argument(
        "--allow-legacy-latest-fallback",
        action="store_true",
        help=(
            "Explicitly permit parent-latest selection for historical unverified "
            "refined-keypoint diagnostics."
        ),
    )
    parser.add_argument(
        "--refined-run",
        help=(
            "Exact refined-keypoint diagnostic child to finalize. This is the "
            "default-safe selection mode and does not consult a parent selector."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only (default behavior).")
    parser.add_argument("--apply", action="store_true", help="Write/update artifacts.")
    parser.add_argument("--json-report", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        raise SystemExit("Choose either --apply or --dry-run, not both.")
    refined_run = _decode_text(args.refined_run)
    if refined_run is not None and (
        refined_run in {".", ".."}
        or "/" in refined_run
        or "\\" in refined_run
    ):
        parser.error("--refined-run must be one exact child name, not a path")
    if refined_run is not None and args.allow_legacy_latest_fallback:
        parser.error(
            "--refined-run and --allow-legacy-latest-fallback are mutually exclusive"
        )
    apply = bool(args.apply)

    roots = _resolve_roots(list(args.paths))
    rows = _build_rows(
        roots,
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        approved_state=str(args.approved_state),
        required_intended_use=args.required_intended_use,
        force=bool(args.force),
        visuals_dpi=int(args.visuals_dpi),
        refined_run=refined_run,
        allow_legacy_latest_fallback=bool(args.allow_legacy_latest_fallback),
    )

    scanned = len(rows)
    eligible = sum(1 for row in rows if row.status == "ok")
    skipped = scanned - eligible
    rendered = 0
    unchanged = 0
    would_finalize = 0
    errors = 0

    for row in rows:
        if row.status != "ok":
            print(f"skip: {row.zarr_path}: {row.reason}")
            continue

        if not apply:
            if row.action == "render":
                would_finalize += 1
                print(
                    f"would_finalize: {row.zarr_path}: "
                    f"{row.parent_name}/{row.refined_run} ({QUALITY_ARTIFACT_NAME}, {REFINEMENT_PIPELINE_ARTIFACT_NAME})"
                )
            else:
                unchanged += 1
                print(
                    f"unchanged: {row.zarr_path}: "
                    f"{row.parent_name}/{row.refined_run} ({QUALITY_ARTIFACT_NAME}, {REFINEMENT_PIPELINE_ARTIFACT_NAME})"
                )
            continue

        try:
            action = _finalize_row(row, visuals_dpi=int(args.visuals_dpi), force=bool(args.force))
        except Exception as exc:
            errors += 1
            row.status = "error"
            row.reason = str(exc)
            row.action = "error"
            print(f"error: {row.zarr_path}: {exc}")
            continue

        row.action = action
        if action == "rendered":
            rendered += 1
            print(
                f"finalized: {row.zarr_path}: "
                f"{row.parent_name}/{row.refined_run} ({QUALITY_ARTIFACT_NAME}, {REFINEMENT_PIPELINE_ARTIFACT_NAME})"
            )
        else:
            unchanged += 1
            print(
                f"unchanged: {row.zarr_path}: "
                f"{row.parent_name}/{row.refined_run} ({QUALITY_ARTIFACT_NAME}, {REFINEMENT_PIPELINE_ARTIFACT_NAME})"
            )

    if apply:
        print(
            "Keypoint refinement artifact finalize: "
            f"scanned={scanned} eligible={eligible} skipped={skipped} "
            f"rendered={rendered} unchanged={unchanged} errors={errors}"
        )
    else:
        print(
            "Keypoint refinement artifact finalize: "
            f"scanned={scanned} eligible={eligible} skipped={skipped} "
            f"would_finalize={would_finalize} unchanged={unchanged} errors={errors}"
        )

    if args.json_report:
        report_payload: Dict[str, object] = {
            "mode": "apply" if apply else "dry-run",
            "summary": {
                "scanned": scanned,
                "eligible": eligible,
                "skipped": skipped,
                "rendered": rendered if apply else 0,
                "would_finalize": would_finalize if not apply else 0,
                "unchanged": unchanged,
                "errors": errors,
            },
            "rows": [asdict(row) for row in rows],
        }
        _write_json_report(args.json_report.expanduser(), report_payload)

    if errors:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
