#!/usr/bin/env python3
"""Prune legacy eye-mask profile runs missing stage/review profile metadata.

Default behavior is dry-run. Use --apply to delete selected run directories.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")
PROFILE_PARENT_PATH = Path("analysis") / "eye_mask_profile_runs"
EXPECTED_STAGE_GROUPS = {"eye_masks_runs", "refined_eye_masks_runs"}


@dataclass(frozen=True)
class ProfileRunMeta:
    run_name: str
    run_path: Path
    stage_group: Optional[str]
    review_state: Optional[str]
    review_intended_use: Optional[str]
    eye_mask_method: Optional[str]
    source_eye_mask_run: Optional[str]
    source_keypoint_run: Optional[str]
    source_crop_run: Optional[str]
    rows_total: Optional[int]
    rows_usable: Optional[int]
    usable_rate: Optional[float]
    profile_created_utc: Optional[str]


@dataclass
class PruneRow:
    zarr_path: str
    zarr_use: str
    profile_run: Optional[str]
    stage_group: Optional[str]
    review_state: Optional[str]
    review_intended_use: Optional[str]
    replacement_run: Optional[str]
    status: str
    reason: str
    action: str
    run_path: Optional[str] = None


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


def _as_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


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


def _parse_iso_utc(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_roots(paths: List[Path]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


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


def _extract_source_stage_group(source_map: Dict[str, object]) -> Optional[str]:
    return (
        _decode_text(source_map.get("stage_group"))
        or _decode_text(source_map.get("eye_stage_group"))
        or _decode_text(source_map.get("source_eye_stage"))
    )


def _extract_run_meta(run_path: Path) -> Optional[ProfileRunMeta]:
    attrs = _read_zarr_attrs(run_path / "zarr.json")
    summary = _coerce_mapping(attrs.get("profile_summary"))
    if not summary:
        return None
    source_map = _coerce_mapping(summary.get("source")) or {}
    quality_map = _coerce_mapping(summary.get("quality")) or {}
    return ProfileRunMeta(
        run_name=run_path.name,
        run_path=run_path,
        stage_group=_extract_source_stage_group(source_map),
        review_state=_decode_text(source_map.get("review_state")),
        review_intended_use=(
            _decode_text(source_map.get("review_intended_use"))
            or _decode_text(source_map.get("intended_use"))
        ),
        eye_mask_method=_decode_text(source_map.get("eye_mask_method")) or _decode_text(source_map.get("method")),
        source_eye_mask_run=(
            _decode_text(source_map.get("eye_mask_run"))
            or _decode_text(source_map.get("source_eye_mask_run"))
        ),
        source_keypoint_run=(
            _decode_text(source_map.get("keypoint_run"))
            or _decode_text(source_map.get("source_keypoint_run"))
        ),
        source_crop_run=_decode_text(source_map.get("source_crop_run")),
        rows_total=_as_int(quality_map.get("rows_total")),
        rows_usable=_as_int(
            quality_map.get("rows_usable")
            if quality_map.get("rows_usable") is not None
            else quality_map.get("usable_rows")
        ),
        usable_rate=_as_float(
            quality_map.get("usable_rate")
            if quality_map.get("usable_rate") is not None
            else quality_map.get("rows_usable_rate")
        ),
        profile_created_utc=_decode_text(summary.get("created_at_utc")),
    )


def _is_legacy(meta: ProfileRunMeta) -> bool:
    if meta.stage_group is None:
        return True
    if meta.review_state is None or meta.review_intended_use is None:
        return True
    return False


def _matches_replacement(candidate: ProfileRunMeta, peer: ProfileRunMeta) -> bool:
    if peer.run_name == candidate.run_name:
        return False
    if peer.stage_group not in EXPECTED_STAGE_GROUPS:
        return False

    checks: List[Tuple[str, Optional[str], Optional[str]]] = [
        ("eye_mask_method", candidate.eye_mask_method, peer.eye_mask_method),
        ("source_eye_mask_run", candidate.source_eye_mask_run, peer.source_eye_mask_run),
        ("source_keypoint_run", candidate.source_keypoint_run, peer.source_keypoint_run),
        ("source_crop_run", candidate.source_crop_run, peer.source_crop_run),
    ]
    constrained = [(left, right) for _, left, right in checks if left]
    if constrained:
        return all(left == right for left, right in constrained)

    fallback = [
        (candidate.rows_total, peer.rows_total),
        (candidate.rows_usable, peer.rows_usable),
        (candidate.usable_rate, peer.usable_rate),
    ]
    constrained_fallback = [(left, right) for left, right in fallback if left is not None]
    if not constrained_fallback:
        return False
    return all(left == right for left, right in constrained_fallback)


def _select_replacement(candidate: ProfileRunMeta, peers: List[ProfileRunMeta]) -> Optional[ProfileRunMeta]:
    matches = [peer for peer in peers if _matches_replacement(candidate, peer)]
    if not matches:
        return None
    matches.sort(
        key=lambda row: (
            _parse_iso_utc(row.profile_created_utc),
            row.run_name,
        ),
        reverse=True,
    )
    return matches[0]


def _collect_profile_runs(parent_path: Path) -> List[ProfileRunMeta]:
    rows: List[ProfileRunMeta] = []
    for run_path in sorted(parent_path.iterdir()):
        if not run_path.is_dir():
            continue
        if not (run_path / "zarr.json").exists():
            continue
        meta = _extract_run_meta(run_path)
        if meta is not None:
            rows.append(meta)
    return rows


def _plan_zarr(
    zarr_path: Path,
    *,
    zarr_use_filter: str,
    allow_unpaired: bool,
) -> List[PruneRow]:
    rows: List[PruneRow] = []
    root_attrs = _read_zarr_attrs(zarr_path / "zarr.json")
    zarr_use = _infer_zarr_use(zarr_path, root_attrs)
    if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
        rows.append(
            PruneRow(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                profile_run=None,
                stage_group=None,
                review_state=None,
                review_intended_use=None,
                replacement_run=None,
                status="filtered_zarr_use",
                reason=f"wanted={zarr_use_filter} found={zarr_use}",
                action="skip",
            )
        )
        return rows

    parent_path = zarr_path / PROFILE_PARENT_PATH
    if not parent_path.exists() or not (parent_path / "zarr.json").exists():
        rows.append(
            PruneRow(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                profile_run=None,
                stage_group=None,
                review_state=None,
                review_intended_use=None,
                replacement_run=None,
                status="no_eye_mask_profile_runs",
                reason="analysis/eye_mask_profile_runs missing",
                action="skip",
            )
        )
        return rows

    parent_attrs = _read_zarr_attrs(parent_path / "zarr.json")
    latest_run = _decode_text(parent_attrs.get("latest"))
    metas = _collect_profile_runs(parent_path)
    if not metas:
        rows.append(
            PruneRow(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                profile_run=None,
                stage_group=None,
                review_state=None,
                review_intended_use=None,
                replacement_run=None,
                status="no_profile_runs",
                reason="analysis/eye_mask_profile_runs has no valid profile_summary runs",
                action="skip",
            )
        )
        return rows

    peers = [meta for meta in metas if not _is_legacy(meta)]
    for meta in metas:
        if not _is_legacy(meta):
            continue
        replacement = _select_replacement(meta, peers)
        replacement_run = replacement.run_name if replacement else None

        if latest_run and meta.run_name == latest_run:
            rows.append(
                PruneRow(
                    zarr_path=str(zarr_path),
                    zarr_use=zarr_use,
                    profile_run=meta.run_name,
                    stage_group=meta.stage_group,
                    review_state=meta.review_state,
                    review_intended_use=meta.review_intended_use,
                    replacement_run=replacement_run,
                    status="legacy_latest",
                    reason="run is parent latest; skipped for safety",
                    action="skip",
                    run_path=str(meta.run_path),
                )
            )
            continue

        if replacement is None and not allow_unpaired:
            rows.append(
                PruneRow(
                    zarr_path=str(zarr_path),
                    zarr_use=zarr_use,
                    profile_run=meta.run_name,
                    stage_group=meta.stage_group,
                    review_state=meta.review_state,
                    review_intended_use=meta.review_intended_use,
                    replacement_run=None,
                    status="legacy_no_replacement",
                    reason="no normalized replacement run found",
                    action="skip",
                    run_path=str(meta.run_path),
                )
            )
            continue

        rows.append(
            PruneRow(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                profile_run=meta.run_name,
                stage_group=meta.stage_group,
                review_state=meta.review_state,
                review_intended_use=meta.review_intended_use,
                replacement_run=replacement_run,
                status="legacy_candidate",
                reason="missing stage/review metadata",
                action="delete",
                run_path=str(meta.run_path),
            )
        )
    return rows


def _build_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    allow_unpaired: bool,
) -> List[PruneRow]:
    rows: List[PruneRow] = []
    for zarr_path in _iter_zarr(roots, recursive=recursive):
        rows.extend(
            _plan_zarr(
                zarr_path,
                zarr_use_filter=zarr_use_filter,
                allow_unpaired=allow_unpaired,
            )
        )
    return rows


def _apply_delete(row: PruneRow) -> str:
    if row.action != "delete":
        return "skipped"
    if not row.run_path:
        return "error:no_run_path"
    run_path = Path(row.run_path)
    if not run_path.exists():
        return "missing"
    shutil.rmtree(run_path)
    return "deleted"


def _summarize(rows: List[PruneRow], *, apply: bool) -> Dict[str, int]:
    summary = {
        "scanned": len(rows),
        "legacy_candidates": sum(1 for row in rows if row.status == "legacy_candidate"),
        "legacy_latest": sum(1 for row in rows if row.status == "legacy_latest"),
        "legacy_no_replacement": sum(1 for row in rows if row.status == "legacy_no_replacement"),
        "filtered_zarr_use": sum(1 for row in rows if row.status == "filtered_zarr_use"),
        "deleted": 0,
        "would_delete": 0,
        "errors": 0,
    }
    if apply:
        return summary
    summary["would_delete"] = summary["legacy_candidates"]
    return summary


def _write_json_report(path: Path, *, rows: List[PruneRow], summary: Dict[str, int], mode: str) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prune legacy eye-mask profile runs from analysis/eye_mask_profile_runs. "
            "Legacy runs are runs missing stage_group or review_state/review_intended_use in profile_summary.source."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by purpose (default: any).",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="Allow deleting legacy runs even when no normalized replacement run is found.",
    )
    parser.add_argument("--apply", action="store_true", help="Delete selected legacy profile runs.")
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args(argv)

    roots = _resolve_roots(list(args.paths))
    rows = _build_rows(
        roots,
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        allow_unpaired=bool(args.allow_unpaired),
    )

    summary = _summarize(rows, apply=bool(args.apply))

    if not args.apply:
        for row in rows:
            if row.status.startswith("legacy_"):
                replacement = row.replacement_run or "-"
                run_name = row.profile_run or "-"
                print(
                    f"{row.status}\t{row.zarr_path}\t{run_name}\t{row.action}\t"
                    f"{row.reason}\treplacement={replacement}"
                )
        print(
            "Legacy eye-mask profile prune: "
            f"mode=dry-run scanned={summary['scanned']} "
            f"legacy_candidates={summary['legacy_candidates']} "
            f"legacy_latest={summary['legacy_latest']} "
            f"legacy_no_replacement={summary['legacy_no_replacement']} "
            f"would_delete={summary['would_delete']} "
            f"filtered_zarr_use={summary['filtered_zarr_use']}"
        )
        print("Dry run: re-run with --apply to delete legacy profile runs.")
        if args.json_report is not None:
            _write_json_report(args.json_report, rows=rows, summary=summary, mode="dry-run")
        return 0

    deleted = 0
    errors = 0
    for row in rows:
        if row.action != "delete":
            continue
        result = _apply_delete(row)
        if result == "deleted":
            deleted += 1
        else:
            errors += 1
            print(
                f"error\t{row.zarr_path}\t{row.profile_run or '-'}\t"
                f"{row.run_path or '-'}\t{result}"
            )

    summary["deleted"] = deleted
    summary["errors"] = errors
    print(
        "Legacy eye-mask profile prune: "
        f"mode=apply scanned={summary['scanned']} "
        f"legacy_candidates={summary['legacy_candidates']} "
        f"deleted={summary['deleted']} "
        f"errors={summary['errors']} "
        f"legacy_latest={summary['legacy_latest']} "
        f"legacy_no_replacement={summary['legacy_no_replacement']} "
        f"filtered_zarr_use={summary['filtered_zarr_use']}"
    )
    if args.json_report is not None:
        _write_json_report(args.json_report, rows=rows, summary=summary, mode="apply")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
