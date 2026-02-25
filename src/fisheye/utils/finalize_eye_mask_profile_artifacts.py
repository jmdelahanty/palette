#!/usr/bin/env python3
"""Finalize eye-mask profile overview artifacts in profile runs.

Default behavior is dry-run. Use --apply to write visualization artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import zarr

from fisheye.utils.export_eye_mask_quality_overview import (
    ARTIFACT_NAME,
    ARTIFACT_PATH,
    render_eye_mask_profile_overview_png,
)


PROFILE_PARENT_PATH = "analysis/eye_mask_profile_runs"
SIGNATURE_VERSION = 1

ARTIFACT_SPECS = {
    ARTIFACT_NAME: {
        "path": ARTIFACT_PATH,
        "description": "Eye-mask profile overview PNG",
    }
}


@dataclass
class FinalizeRow:
    zarr_path: str
    zarr_use: str
    profile_run: Optional[str]
    review_state: Optional[str]
    review_intended_use: Optional[str]
    review_method: Optional[str]
    review_timestamp_utc: Optional[str]
    source_stage_group: Optional[str]
    source_eye_mask_method: Optional[str]
    artifact_signature: Optional[str]
    status: str
    reason: str
    action: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: List[Path] = []
        if root.suffix == ".zarr" and (root.is_dir() or root.is_file()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _read_zarr_attrs(zarr_json_path: Path) -> Dict[str, object]:
    if not zarr_json_path.exists():
        return {}
    try:
        payload = json.loads(zarr_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _infer_zarr_use(zarr_path: Path, root_attrs: Dict[str, object]) -> str:
    for key in ("zarr_use", "zarr_purpose"):
        value = _decode_text_lower(root_attrs.get(key))
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _resolve_profile_parent_path(zarr_path: Path) -> Optional[Path]:
    parent_path = zarr_path / "analysis" / "eye_mask_profile_runs"
    if parent_path.exists() and (parent_path / "zarr.json").exists():
        return parent_path
    return None


def _resolve_review_timestamp(review_map: Dict[str, object]) -> Optional[str]:
    for key in ("timestamp_utc", "timestamp", "reviewed_at_utc", "reviewed_at", "updated_utc"):
        text = _decode_text(review_map.get(key))
        if text:
            return text
    return None


def _build_artifact_signature(
    row: FinalizeRow,
    *,
    visuals_dpi: int,
    profile_summary: Dict[str, object],
) -> str:
    payload = {
        "signature_version": SIGNATURE_VERSION,
        "artifact_names": sorted(ARTIFACT_SPECS.keys()),
        "profile_run": row.profile_run,
        "review_state": row.review_state,
        "review_intended_use": row.review_intended_use,
        "review_method": row.review_method,
        "review_timestamp_utc": row.review_timestamp_utc,
        "source_stage_group": row.source_stage_group,
        "source_eye_mask_method": row.source_eye_mask_method,
        "visuals_dpi": int(visuals_dpi),
        "profile_summary_hash": hashlib.sha256(
            json.dumps(profile_summary, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
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
    required_review_state: Optional[str],
    required_intended_use: Optional[str],
    force: bool,
    visuals_dpi: int,
) -> List[FinalizeRow]:
    rows: List[FinalizeRow] = []
    required_review_state_norm = required_review_state.strip().lower() if required_review_state else None
    required_use_norm = required_intended_use.strip().lower() if required_intended_use else None

    for zarr_path in _iter_zarr(roots, recursive=recursive):
        root_attrs = _read_zarr_attrs(zarr_path / "zarr.json")
        zarr_use = _infer_zarr_use(zarr_path, root_attrs)
        if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
            continue

        row = FinalizeRow(
            zarr_path=str(zarr_path),
            zarr_use=zarr_use,
            profile_run=None,
            review_state=None,
            review_intended_use=None,
            review_method=None,
            review_timestamp_utc=None,
            source_stage_group=None,
            source_eye_mask_method=None,
            artifact_signature=None,
            status="skip",
            reason="unknown",
            action="skip",
        )

        parent_path = _resolve_profile_parent_path(zarr_path)
        if parent_path is None:
            row.reason = "no_eye_mask_profile_runs"
            rows.append(row)
            continue

        parent_attrs = _read_zarr_attrs(parent_path / "zarr.json")
        latest_run = _decode_text(parent_attrs.get("latest"))
        if latest_run is None:
            row.reason = "no_latest_profile_run"
            rows.append(row)
            continue
        row.profile_run = latest_run

        run_path = parent_path / latest_run
        run_attrs = _read_zarr_attrs(run_path / "zarr.json")
        if not run_attrs:
            row.reason = "missing_profile_run_attrs"
            rows.append(row)
            continue

        profile_summary = _coerce_mapping(run_attrs.get("profile_summary"))
        if profile_summary is None:
            row.reason = "profile_summary_missing"
            rows.append(row)
            continue

        source_map = _coerce_mapping(profile_summary.get("source")) or {}
        row.review_state = _decode_text(source_map.get("review_state"))
        row.review_intended_use = _decode_text(source_map.get("review_intended_use"))
        row.review_method = _decode_text(source_map.get("review_method"))
        row.review_timestamp_utc = _decode_text(source_map.get("review_timestamp_utc"))
        row.source_stage_group = _decode_text(source_map.get("stage_group"))
        row.source_eye_mask_method = _decode_text(source_map.get("eye_mask_method")) or _decode_text(
            source_map.get("method")
        )

        review_payload = _coerce_mapping(source_map.get("review"))
        if review_payload:
            row.review_state = row.review_state or _decode_text(review_payload.get("state"))
            row.review_intended_use = row.review_intended_use or _decode_text(review_payload.get("intended_use"))
            row.review_method = row.review_method or _decode_text(review_payload.get("method"))
            row.review_timestamp_utc = row.review_timestamp_utc or _resolve_review_timestamp(review_payload)

        if required_review_state_norm is not None:
            row_state_norm = row.review_state.strip().lower() if row.review_state else None
            if row_state_norm is None:
                row.reason = "review_state_missing"
                rows.append(row)
                continue
            if row_state_norm != required_review_state_norm:
                row.reason = "review_state_mismatch"
                rows.append(row)
                continue

        if required_use_norm is not None:
            row_use_norm = row.review_intended_use.strip().lower() if row.review_intended_use else None
            if row_use_norm is None:
                row.reason = "review_intended_use_missing"
                rows.append(row)
                continue
            if row_use_norm != required_use_norm:
                row.reason = "review_intended_use_mismatch"
                rows.append(row)
                continue

        row.status = "ok"
        row.reason = "eligible"
        row.artifact_signature = _build_artifact_signature(
            row,
            visuals_dpi=visuals_dpi,
            profile_summary=profile_summary,
        )
        existing_signature = _read_existing_artifact_signature(run_path)
        if (not force) and existing_signature and existing_signature == row.artifact_signature:
            row.action = "unchanged"
        else:
            row.action = "render"
        rows.append(row)

    rows.sort(key=lambda item: item.zarr_path)
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
    if row.profile_run is None:
        raise RuntimeError("missing profile run context")

    root = zarr.open_group(row.zarr_path, mode="a")
    if PROFILE_PARENT_PATH not in root:
        raise RuntimeError(f"missing profile parent group: {PROFILE_PARENT_PATH}")
    parent_group = root[PROFILE_PARENT_PATH]
    if row.profile_run not in parent_group:
        raise RuntimeError(f"missing profile run: {row.profile_run}")
    run_group = parent_group[row.profile_run]

    if "visualizations" in run_group:
        existing_signature = _read_existing_artifact_signature(Path(row.zarr_path) / PROFILE_PARENT_PATH / row.profile_run)
    else:
        existing_signature = None

    if (not force) and row.artifact_signature and existing_signature == row.artifact_signature:
        return "unchanged"

    summary = _coerce_mapping(run_group.attrs.get("profile_summary"))
    if summary is None:
        raise RuntimeError("profile_summary_missing")
    png_bytes = render_eye_mask_profile_overview_png(
        summary,
        zarr_name=Path(row.zarr_path).name,
        profile_run=row.profile_run,
    )

    vis_group = run_group.require_group("visualizations")
    finalized_at = _utc_now()
    base_attrs = {
        "mime": "image/png",
        "artifact_signature": row.artifact_signature,
        "artifact_signature_version": SIGNATURE_VERSION,
        "finalized_at_utc": finalized_at,
        "finalized_by": "finalize_eye_mask_profile_artifacts",
        "review_state": row.review_state,
        "review_intended_use": row.review_intended_use,
        "review_timestamp_utc": row.review_timestamp_utc,
        "source_stage_group": row.source_stage_group,
        "source_eye_mask_method": row.source_eye_mask_method,
    }
    _write_png_artifact(
        vis_group=vis_group,
        artifact_name=ARTIFACT_NAME,
        png_bytes=png_bytes,
        base_attrs=base_attrs,
        extra_attrs={
            "description": str(ARTIFACT_SPECS[ARTIFACT_NAME]["description"]),
            "profile_run": row.profile_run,
        },
    )

    manifest = _coerce_mapping(run_group.attrs.get("visualizations")) or {}
    manifest[ARTIFACT_NAME] = {
        "path": ARTIFACT_PATH,
        "description": ARTIFACT_SPECS[ARTIFACT_NAME]["description"],
        "artifact_signature": row.artifact_signature,
        "finalized_at_utc": finalized_at,
    }
    run_group.attrs["visualizations"] = manifest
    return "rendered"


def _write_json_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize eye-mask profile overview artifacts for profile runs "
            "(analysis/eye_mask_profile_runs/<run>/visualizations)."
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
        "--required-review-state",
        default="approved",
        help="Optional required profile source.review_state (default: approved). Use '' to disable.",
    )
    parser.add_argument(
        "--required-intended-use",
        choices=["training", "full_recording"],
        help="Optional required profile source.review_intended_use.",
    )
    parser.add_argument(
        "--visuals-dpi",
        type=int,
        default=150,
        help="DPI for rendered finalized PNG artifacts (default: 150).",
    )
    parser.add_argument("--force", action="store_true", help="Re-render even when artifact signature is unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only (default behavior).")
    parser.add_argument("--apply", action="store_true", help="Write/update artifacts.")
    parser.add_argument("--json-report", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        raise SystemExit("Choose either --apply or --dry-run, not both.")
    apply = bool(args.apply)
    required_review_state = str(args.required_review_state).strip() if args.required_review_state is not None else ""
    if required_review_state == "":
        required_review_state = None

    roots = _resolve_roots(list(args.paths))
    rows = _build_rows(
        roots,
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        required_review_state=required_review_state,
        required_intended_use=args.required_intended_use,
        force=bool(args.force),
        visuals_dpi=int(args.visuals_dpi),
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
                    f"{PROFILE_PARENT_PATH}/{row.profile_run} ({ARTIFACT_NAME})"
                )
            else:
                unchanged += 1
                print(
                    f"unchanged: {row.zarr_path}: "
                    f"{PROFILE_PARENT_PATH}/{row.profile_run} ({ARTIFACT_NAME})"
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
                f"{PROFILE_PARENT_PATH}/{row.profile_run} ({ARTIFACT_NAME})"
            )
        else:
            unchanged += 1
            print(
                f"unchanged: {row.zarr_path}: "
                f"{PROFILE_PARENT_PATH}/{row.profile_run} ({ARTIFACT_NAME})"
            )

    if apply:
        print(
            "Eye-mask profile artifact finalize: "
            f"scanned={scanned} eligible={eligible} skipped={skipped} "
            f"rendered={rendered} unchanged={unchanged} errors={errors}"
        )
    else:
        print(
            "Eye-mask profile artifact finalize: "
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
