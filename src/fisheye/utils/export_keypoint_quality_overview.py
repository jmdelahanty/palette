#!/usr/bin/env python3
"""Export saved keypoint quality overview PNG artifacts from refined keypoint runs."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr

from fisheye.visualization.visualize_keypoint_quality import (
    QUALITY_ARTIFACT_NAME,
    REFINEMENT_PIPELINE_ARTIFACT_NAME as REFINEMENT_ARTIFACT_NAME,
)


REFINED_PARENT_NAMES = ("refined_keypoints_runs", "keypoints_refined_runs")


@dataclass
class ExportRow:
    zarr_path: str
    zarr_use: str
    artifact_name: str
    parent_name: Optional[str]
    refined_run: Optional[str]
    status: str
    reason: str
    output_path: Optional[str]
    bytes_written: Optional[int]
    heading_temporal_outlier: int
    heading_temporal_evaluable: int
    heading_temporal_outlier_rate_percent: Optional[float]


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


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _infer_zarr_use(zarr_path: Path, root_attrs: Dict[str, object]) -> str:
    for key in ("zarr_use", "zarr_purpose"):
        value = _decode_text(root_attrs.get(key))
        if value and value.lower() in {"analysis", "training"}:
            return value.lower()
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _resolve_refined_parent(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    for parent_name in REFINED_PARENT_NAMES:
        if parent_name in root:
            return parent_name, root[parent_name]
    return None, None


def _resolve_refined_run(parent_group: zarr.Group, requested_refined_run: Optional[str]) -> Optional[str]:
    if requested_refined_run:
        return requested_refined_run if requested_refined_run in parent_group else None
    latest = _decode_text(parent_group.attrs.get("latest"))
    if latest and latest in parent_group:
        return latest
    try:
        names = sorted(parent_group.group_keys())
    except Exception:
        names = sorted(parent_group.keys())
    if not names:
        return None
    return str(names[-1])


def _extract_temporal_heading_summary(run_group: zarr.Group) -> tuple[int, int, Optional[float]]:
    summary_raw = run_group.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        return 0, 0, None
    postprocess = summary_raw.get("postprocess")
    source = postprocess if isinstance(postprocess, dict) else summary_raw
    try:
        outlier = int(source.get("heading_temporal_outlier", 0))
    except Exception:
        outlier = 0
    try:
        evaluable = int(source.get("heading_temporal_evaluable", 0))
    except Exception:
        evaluable = 0
    try:
        rate_raw = source.get("heading_temporal_outlier_rate_percent")
        rate = float(rate_raw) if rate_raw is not None else None
    except Exception:
        rate = None
    return outlier, evaluable, rate


def _collect_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    refined_run: Optional[str],
    output_dir: Path,
    artifact_name: str,
    sort_by: str,
) -> List[ExportRow]:
    rows: List[ExportRow] = []
    for zarr_path in _iter_zarr(roots, recursive=recursive):
        root_attrs = _read_zarr_attrs(zarr_path / "zarr.json")
        zarr_use = _infer_zarr_use(zarr_path, root_attrs)
        if zarr_use_filter != "any" and zarr_use_filter != zarr_use:
            continue

        row = ExportRow(
            zarr_path=str(zarr_path),
            zarr_use=zarr_use,
            artifact_name=artifact_name,
            parent_name=None,
            refined_run=None,
            status="skip",
            reason="unknown",
            output_path=None,
            bytes_written=None,
            heading_temporal_outlier=0,
            heading_temporal_evaluable=0,
            heading_temporal_outlier_rate_percent=None,
        )

        try:
            root = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            row.status = "error"
            row.reason = f"zarr_open_failed: {exc}"
            rows.append(row)
            continue

        parent_name, parent_group = _resolve_refined_parent(root)
        if parent_name is None or parent_group is None:
            row.reason = "no_refined_keypoint_runs"
            rows.append(row)
            continue
        row.parent_name = parent_name

        selected_run = _resolve_refined_run(parent_group, refined_run)
        if selected_run is None:
            row.reason = "refined_run_not_found"
            rows.append(row)
            continue
        row.refined_run = selected_run

        run_group = parent_group[selected_run]
        outlier_count, evaluable_count, outlier_rate = _extract_temporal_heading_summary(run_group)
        row.heading_temporal_outlier = outlier_count
        row.heading_temporal_evaluable = evaluable_count
        row.heading_temporal_outlier_rate_percent = outlier_rate
        if "visualizations" not in run_group:
            row.reason = "no_visualizations_group"
            rows.append(row)
            continue
        vis_group = run_group["visualizations"]
        if artifact_name not in vis_group:
            row.reason = "artifact_missing"
            rows.append(row)
            continue

        row.status = "ready"
        row.reason = "ok"
        safe_name = f"{zarr_path.stem}__{selected_run}__{artifact_name}.png"
        row.output_path = str(output_dir / safe_name)
        rows.append(row)

    if sort_by == "temporal-outliers":
        rows.sort(key=lambda row: (-int(row.heading_temporal_outlier), row.zarr_path))
    elif sort_by == "temporal-outlier-rate":
        rows.sort(
            key=lambda row: (
                -(float(row.heading_temporal_outlier_rate_percent) if row.heading_temporal_outlier_rate_percent is not None else -1.0),
                row.zarr_path,
            )
        )
    else:
        rows.sort(key=lambda row: row.zarr_path)
    return rows


def _write_json_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _view_png_bytes(png_bytes: bytes, *, title: str) -> None:
    image = plt.imread(BytesIO(png_bytes), format="png")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)
    plt.show()
    plt.close(fig)


def _load_artifact_bytes(row: ExportRow) -> bytes:
    if row.parent_name is None or row.refined_run is None:
        raise RuntimeError("missing refined run context")
    root = zarr.open_group(row.zarr_path, mode="r")
    vis_array = root[row.parent_name][row.refined_run]["visualizations"][row.artifact_name]
    return np.asarray(vis_array[:], dtype=np.uint8).tobytes()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export keypoint quality overview PNG artifacts stored in refined keypoint runs."
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
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--refined-run",
        help="Specific refined keypoint run to export from (default: refined parent latest).",
    )
    parser.add_argument(
        "--artifact",
        choices=[QUALITY_ARTIFACT_NAME, REFINEMENT_ARTIFACT_NAME],
        default=QUALITY_ARTIFACT_NAME,
        help=f"Artifact to export/view (default: {QUALITY_ARTIFACT_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/keypoint_quality_overview_exports"),
        help="Directory to write exported PNG files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output-dir.")
    parser.add_argument("--list", action="store_true", help="List export candidates without writing files.")
    parser.add_argument(
        "--view",
        action="store_true",
        help="View artifacts directly from zarr without exporting files.",
    )
    parser.add_argument("--open", action="store_true", help="Open exported files with xdg-open.")
    parser.add_argument(
        "--sort-by",
        choices=["path", "temporal-outliers", "temporal-outlier-rate"],
        default="path",
        help="Sort rows by path or temporal-heading outlier severity.",
    )
    parser.add_argument("--json-report", type=Path, help="Optional path to write a JSON report.")
    args = parser.parse_args(argv)

    if args.view and args.list:
        parser.error("Choose either --view or --list, not both.")
    if args.view and args.open:
        parser.error("--open is only valid for exported files (not with --view).")
    roots = _resolve_roots(list(args.paths))
    output_dir = args.output_dir.expanduser()
    rows = _collect_rows(
        roots,
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        refined_run=args.refined_run,
        output_dir=output_dir,
        artifact_name=str(args.artifact),
        sort_by=str(args.sort_by),
    )

    scanned = len(rows)
    ready = sum(1 for row in rows if row.status == "ready")
    skipped = sum(1 for row in rows if row.status == "skip")
    errors = sum(1 for row in rows if row.status == "error")
    exported = 0
    existing = 0
    listed = 0
    viewed = 0
    open_errors = 0
    exported_paths: List[Path] = []

    for row in rows:
        if row.status != "ready":
            print(f"{row.status}: {row.zarr_path}: {row.reason}")
            continue
        assert row.refined_run is not None
        assert row.output_path is not None

        if args.list:
            listed += 1
            row.status = "listed"
            row.reason = "list_mode"
            print(
                "list: "
                f"{row.zarr_path}: {row.output_path} "
                f"(temporal_outliers={row.heading_temporal_outlier}/{row.heading_temporal_evaluable})"
            )
            continue

        if args.view:
            try:
                png_bytes = _load_artifact_bytes(row)
                _view_png_bytes(
                    png_bytes,
                    title=f"{Path(row.zarr_path).name} | {row.refined_run} | {row.artifact_name}",
                )
                row.status = "viewed"
                row.reason = "shown"
                row.bytes_written = len(png_bytes)
                viewed += 1
                print(f"viewed: {row.zarr_path}: {row.refined_run}")
            except Exception as exc:
                row.status = "error"
                row.reason = f"view_failed: {exc}"
                errors += 1
                print(f"error: {row.zarr_path}: {exc}")
            continue

        out_path = Path(row.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            existing += 1
            row.status = "existing"
            row.reason = "output_exists"
            print(f"existing: {row.zarr_path}: {out_path}")
            exported_paths.append(out_path)
            continue

        try:
            png_bytes = _load_artifact_bytes(row)
            out_path.write_bytes(png_bytes)
            row.status = "exported"
            row.reason = "written"
            row.bytes_written = len(png_bytes)
            exported += 1
            exported_paths.append(out_path)
            print(f"exported: {row.zarr_path}: {out_path}")
        except Exception as exc:
            row.status = "error"
            row.reason = f"export_failed: {exc}"
            errors += 1
            print(f"error: {row.zarr_path}: {exc}")

    if args.open and not args.list and not args.view:
        for path in exported_paths:
            try:
                subprocess.run(["xdg-open", str(path)], check=False)
            except Exception:
                open_errors += 1

    if args.list:
        print(
            "Keypoint-quality artifact export: "
            f"scanned={scanned} ready={ready} listed={listed} skipped={skipped} errors={errors}"
        )
    elif args.view:
        print(
            "Keypoint-quality artifact export: "
            f"scanned={scanned} ready={ready} viewed={viewed} skipped={skipped} errors={errors}"
        )
    else:
        print(
            "Keypoint-quality artifact export: "
            f"scanned={scanned} ready={ready} exported={exported} existing={existing} "
            f"skipped={skipped} errors={errors} open_errors={open_errors}"
        )

    if args.json_report:
        payload: Dict[str, object] = {
            "mode": "list" if args.list else ("view" if args.view else "export"),
            "summary": {
                "scanned": scanned,
                "ready": ready,
                "listed": listed,
                "viewed": viewed,
                "exported": exported,
                "existing": existing,
                "skipped": skipped,
                "errors": errors,
                "open_errors": open_errors,
            },
            "rows": [asdict(row) for row in rows],
        }
        _write_json_report(args.json_report.expanduser(), payload)

    if errors:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
