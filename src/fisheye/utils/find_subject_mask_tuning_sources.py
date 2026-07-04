#!/usr/bin/env python3
"""Find zarr archives whose subject_mask_tuning includes selected components."""

from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from fisheye.utils.apply_tuning_by_camera import (
    _camera_id_for_zarr,
    _normalize_subject_mask_tuning_payload,
    _parse_subject_mask_components,
)
from fisheye.utils.zarr_io import open_zarr_root


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")


@dataclass(frozen=True)
class ScanRow:
    zarr_path: Path
    status: str
    observed_use: Optional[str] = None
    camera_id: Optional[str] = None
    component_names: tuple[str, ...] = ()
    matched_components: tuple[str, ...] = ()
    latest_component: Optional[str] = None
    latest_timestamp: Optional[str] = None
    component_details: tuple[str, ...] = ()
    reason: Optional[str] = None


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _format_component_detail(name: str, payload: object) -> str:
    if not isinstance(payload, Mapping):
        return f"component[{name}]: payload_type={type(payload).__name__}"
    method = payload.get("subject_method_family") or payload.get("method") or "unknown"
    tuned_timestamp = payload.get("tuned_timestamp") or "unknown"
    return f"component[{name}]: method={method} tuned_timestamp={tuned_timestamp}"


def _scan_zarr_path(
    zarr_path: Path,
    *,
    zarr_use_filter: str,
    subject_mask_components: Sequence[str],
) -> ScanRow:
    try:
        root = open_zarr_root(zarr_path, mode="r")
    except Exception as exc:
        return ScanRow(zarr_path=zarr_path, status="error", reason=f"failed to open archive ({exc})")

    observed_use = _infer_zarr_use(root, zarr_path)
    if zarr_use_filter != "any" and observed_use != zarr_use_filter:
        return ScanRow(
            zarr_path=zarr_path,
            status="filtered_zarr_use",
            observed_use=observed_use,
            reason=f"wanted={zarr_use_filter} found={observed_use or 'unknown'}",
        )

    analysis = root.get("analysis_metadata")
    if analysis is None:
        return ScanRow(
            zarr_path=zarr_path,
            status="missing_analysis_metadata",
            observed_use=observed_use,
        )

    attrs = dict(getattr(analysis, "attrs", {}) or {})
    subject_tuning = attrs.get("subject_mask_tuning")
    if not isinstance(subject_tuning, Mapping):
        return ScanRow(
            zarr_path=zarr_path,
            status="missing_subject_mask_tuning",
            observed_use=observed_use,
            camera_id=_camera_id_for_zarr(zarr_path, root),
        )

    payload = _normalize_subject_mask_tuning_payload(subject_tuning)
    components_raw = payload.get("components", {})
    if not isinstance(components_raw, Mapping) or not components_raw:
        return ScanRow(
            zarr_path=zarr_path,
            status="missing_subject_mask_tuning",
            observed_use=observed_use,
            camera_id=_camera_id_for_zarr(zarr_path, root),
        )

    component_names = tuple(sorted(str(name) for name in components_raw))
    if subject_mask_components:
        matched_components = tuple(name for name in subject_mask_components if name in components_raw)
    else:
        matched_components = component_names

    camera_id = _camera_id_for_zarr(zarr_path, root)
    latest_component = payload.get("latest_component")
    latest_component_text = str(latest_component) if latest_component is not None else None
    latest_timestamp = payload.get("latest_timestamp")
    latest_timestamp_text = str(latest_timestamp) if latest_timestamp is not None else None

    if not matched_components:
        return ScanRow(
            zarr_path=zarr_path,
            status="missing_components",
            observed_use=observed_use,
            camera_id=camera_id,
            component_names=component_names,
            latest_component=latest_component_text,
            latest_timestamp=latest_timestamp_text,
            reason=", ".join(subject_mask_components),
        )

    component_details = tuple(
        _format_component_detail(name, components_raw.get(name))
        for name in matched_components
    )
    return ScanRow(
        zarr_path=zarr_path,
        status="match",
        observed_use=observed_use,
        camera_id=camera_id,
        component_names=component_names,
        matched_components=matched_components,
        latest_component=latest_component_text,
        latest_timestamp=latest_timestamp_text,
        component_details=component_details,
    )


def _print_match(row: ScanRow) -> None:
    print(f"MATCH {row.zarr_path}")
    print(f"  camera_id: {row.camera_id or 'unknown'}")
    print(f"  use: {row.observed_use or 'unknown'}")
    print(f"  components: {', '.join(row.component_names) if row.component_names else 'none'}")
    if row.matched_components and row.matched_components != row.component_names:
        print(f"  matched_components: {', '.join(row.matched_components)}")
    print(f"  latest_component: {row.latest_component or '—'}")
    print(f"  latest_timestamp: {row.latest_timestamp or '—'}")
    for detail in row.component_details:
        print(f"  {detail}")


def _print_skip(row: ScanRow) -> None:
    reason = row.reason or "-"
    print(f"{row.status.upper()} {row.zarr_path} ({reason})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="List zarr archives whose subject_mask_tuning contains selected components."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan roots for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="training",
        help="Filter zarr archives by purpose (default: training).",
    )
    parser.add_argument(
        "--subject-mask-components",
        action="append",
        help=(
            "Optional comma-separated subject_mask_tuning component names to require "
            "(for example swim_bladder). If omitted, any subject_mask_tuning payload matches."
        ),
    )
    parser.add_argument(
        "--show-skips",
        action="store_true",
        help="Print filtered/missing/error rows in addition to matches.",
    )
    args = parser.parse_args(argv)

    try:
        subject_mask_components = _parse_subject_mask_components(args.subject_mask_components)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    roots = _resolve_roots(list(args.paths))
    rows: list[ScanRow] = []
    for zarr_path in _iter_zarr(roots, bool(args.recursive)):
        rows.append(
            _scan_zarr_path(
                zarr_path,
                zarr_use_filter=str(args.zarr_use),
                subject_mask_components=subject_mask_components,
            )
        )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        if row.status == "match":
            _print_match(row)
        elif args.show_skips:
            _print_skip(row)

    print(
        "Subject-mask tuning scan: "
        f"scope={args.zarr_use} scanned={len(rows)} "
        f"matched={counts.get('match', 0)} "
        f"filtered_zarr_use={counts.get('filtered_zarr_use', 0)} "
        f"missing_analysis_metadata={counts.get('missing_analysis_metadata', 0)} "
        f"missing_subject_mask_tuning={counts.get('missing_subject_mask_tuning', 0)} "
        f"missing_components={counts.get('missing_components', 0)} "
        f"errors={counts.get('error', 0)}"
    )
    if counts.get("match", 0) == 0:
        print("No matching subject_mask_tuning sources found.")
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
