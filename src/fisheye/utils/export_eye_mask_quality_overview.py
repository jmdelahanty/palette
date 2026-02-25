#!/usr/bin/env python3
"""Export saved eye-mask profile overview PNG artifacts from profile runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr


PROFILE_PARENT_PATH = "analysis/eye_mask_profile_runs"
ARTIFACT_NAME = "eye_mask_profile_overview_png"
ARTIFACT_PATH = f"visualizations/{ARTIFACT_NAME}"


@dataclass
class ExportRow:
    zarr_path: str
    zarr_use: str
    artifact_name: str
    profile_run: Optional[str]
    status: str
    reason: str
    output_path: Optional[str]
    bytes_written: Optional[int]


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


def _resolve_profile_parent(root: zarr.Group) -> Optional[zarr.Group]:
    parent = root.get(PROFILE_PARENT_PATH)
    if isinstance(parent, zarr.Group):
        return parent
    analysis = root.get("analysis")
    if not isinstance(analysis, zarr.Group):
        return None
    parent = analysis.get("eye_mask_profile_runs")
    if isinstance(parent, zarr.Group):
        return parent
    return None


def _resolve_profile_run(parent_group: zarr.Group, requested_profile_run: Optional[str]) -> Optional[str]:
    if requested_profile_run:
        return requested_profile_run if requested_profile_run in parent_group else None
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


def _as_mapping(value: object) -> Optional[Mapping[str, Any]]:
    return value if isinstance(value, Mapping) else None


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _extract_p50(metric_payload: Mapping[str, Any]) -> Optional[float]:
    candidates: List[Mapping[str, Any]] = [metric_payload]
    nested_stats = _as_mapping(metric_payload.get("stats"))
    if nested_stats is not None:
        candidates.append(nested_stats)
    for candidate in candidates:
        for key in ("p50", "median", "mean"):
            value = _as_float(candidate.get(key))
            if value is not None:
                return value
    return None


def _extract_reason_counts(quality: Mapping[str, Any]) -> Mapping[str, Any]:
    reason_counts = _as_mapping(quality.get("exclusion_reasons"))
    if reason_counts is not None:
        return reason_counts
    reason_counts = _as_mapping(quality.get("excluded_reasons"))
    if reason_counts is not None:
        return reason_counts

    reason_json = quality.get("exclusion_reasons_json")
    if isinstance(reason_json, str):
        text = reason_json.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return parsed
    reason_json_map = _as_mapping(reason_json)
    if reason_json_map is not None:
        return reason_json_map
    return {}


def _summary_text(summary: Mapping[str, Any]) -> str:
    dataset = _as_mapping(summary.get("dataset")) or {}
    source = _as_mapping(summary.get("source")) or {}
    quality = _as_mapping(summary.get("quality")) or {}
    created = _decode_text(summary.get("created_at_utc")) or "unknown"

    lines = [
        f"Dataset: {_decode_text(dataset.get('dataset_id')) or 'unknown'}",
        f"Recording: {_decode_text(dataset.get('recording_id')) or 'unknown'}",
        f"Use: {_decode_text(dataset.get('zarr_use')) or 'unknown'}",
        f"Stage: {_decode_text(source.get('stage_group')) or 'unknown'}",
        f"Method: {_decode_text(source.get('eye_mask_method')) or _decode_text(source.get('method')) or 'unknown'}",
        (
            "Review: "
            f"{_decode_text(source.get('review_state')) or 'missing'}/"
            f"{_decode_text(source.get('review_intended_use')) or 'missing'}"
        ),
        f"Created: {created}",
    ]

    rows_total = _as_float(quality.get("rows_total"))
    rows_usable = _as_float(quality.get("rows_usable"))
    if rows_total is not None and rows_usable is not None:
        lines.append(f"Rows usable: {int(rows_usable)}/{int(rows_total)}")
    return "\n".join(lines)


def render_eye_mask_profile_overview_png(
    summary: Mapping[str, Any],
    *,
    zarr_name: str,
    profile_run: str,
) -> bytes:
    quality = _as_mapping(summary.get("quality")) or {}
    geometry = _as_mapping(summary.get("geometry")) or {}

    usable_rate = _as_float(quality.get("successful_roi_pair_rate"))
    if usable_rate is None:
        usable_rate = _as_float(quality.get("usable_rate"))
    reviewed_rate = _as_float(quality.get("reviewed_rate"))
    excluded_rate = _as_float(quality.get("excluded_rate"))

    metric_values = [usable_rate, reviewed_rate, excluded_rate]
    metric_labels = ["usable", "reviewed", "excluded"]
    filtered_rates = [
        (label, float(value))
        for label, value in zip(metric_labels, metric_values)
        if value is not None
    ]

    geometry_specs = [
        ("eye sep p50", ("eye_separation", "interocular_px")),
        ("major p50", ("ellipse_major", "major_axis")),
        ("minor p50", ("ellipse_minor", "minor_axis")),
        ("union area p50", ("union_area", "area_union")),
        ("lr ratio p50", ("area_lr_ratio", "left_right_area_ratio")),
    ]
    geometry_rows: List[tuple[str, float]] = []
    for label, keys in geometry_specs:
        payload = None
        for key in keys:
            value = _as_mapping(geometry.get(key))
            if value is not None:
                payload = value
                break
        if payload is None:
            continue
        p50 = _extract_p50(payload)
        if p50 is None:
            continue
        geometry_rows.append((label, p50))

    reason_counts = _extract_reason_counts(quality)
    reason_rows: List[tuple[str, float]] = []
    for name, count in reason_counts.items():
        label = _decode_text(name)
        value = _as_float(count)
        if label is None or value is None or value <= 0:
            continue
        reason_rows.append((label, value))
    reason_rows.sort(key=lambda item: (-item[1], item[0]))
    reason_rows = reason_rows[:6]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_summary, ax_quality, ax_geometry, ax_reasons = axes.flatten()

    ax_summary.axis("off")
    ax_summary.text(
        0.01,
        0.98,
        _summary_text(summary),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    if filtered_rates:
        q_labels = [item[0] for item in filtered_rates]
        q_values = [item[1] for item in filtered_rates]
        q_bars = ax_quality.bar(q_labels, q_values, color="#2E6F95", edgecolor="#123146", linewidth=0.8)
        ax_quality.set_ylim(0.0, 1.0)
        ax_quality.set_title("Quality Rates")
        for bar, value in zip(q_bars, q_values):
            ax_quality.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(0.99, value + 0.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax_quality.axis("off")
        ax_quality.text(0.5, 0.5, "No rate metrics", ha="center", va="center")

    if geometry_rows:
        g_labels = [item[0] for item in geometry_rows]
        g_values = [item[1] for item in geometry_rows]
        y_pos = np.arange(len(g_labels))
        ax_geometry.barh(y_pos, g_values, color="#4A7C59", edgecolor="#264A38", linewidth=0.8)
        ax_geometry.set_yticks(y_pos)
        ax_geometry.set_yticklabels(g_labels)
        ax_geometry.invert_yaxis()
        ax_geometry.set_title("Geometry P50")
        ax_geometry.grid(axis="x", alpha=0.25)
    else:
        ax_geometry.axis("off")
        ax_geometry.text(0.5, 0.5, "No geometry metrics", ha="center", va="center")

    if reason_rows:
        r_labels = [item[0] for item in reason_rows]
        r_values = [item[1] for item in reason_rows]
        y_pos = np.arange(len(r_labels))
        ax_reasons.barh(y_pos, r_values, color="#B85C38", edgecolor="#6A2C1C", linewidth=0.8)
        ax_reasons.set_yticks(y_pos)
        ax_reasons.set_yticklabels(r_labels)
        ax_reasons.invert_yaxis()
        ax_reasons.set_title("Top Exclusion Reasons")
        ax_reasons.grid(axis="x", alpha=0.25)
    else:
        ax_reasons.axis("off")
        ax_reasons.text(0.5, 0.5, "No exclusion reasons", ha="center", va="center")

    fig.suptitle(f"{zarr_name} | {profile_run}", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def _collect_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    profile_run: Optional[str],
    output_dir: Path,
    artifact_name: str,
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
            profile_run=None,
            status="skip",
            reason="unknown",
            output_path=None,
            bytes_written=None,
        )

        try:
            root = zarr.open_group(str(zarr_path), mode="r")
        except Exception as exc:
            row.status = "error"
            row.reason = f"zarr_open_failed: {exc}"
            rows.append(row)
            continue

        parent_group = _resolve_profile_parent(root)
        if parent_group is None:
            row.reason = "no_eye_mask_profile_runs"
            rows.append(row)
            continue

        selected_run = _resolve_profile_run(parent_group, profile_run)
        if selected_run is None:
            row.reason = "profile_run_not_found"
            rows.append(row)
            continue
        row.profile_run = selected_run

        run_group = parent_group[selected_run]
        if "visualizations" not in run_group:
            row.reason = "no_visualizations_group"
            rows.append(row)
            continue
        vis_group = run_group["visualizations"]
        if ARTIFACT_NAME not in vis_group:
            row.reason = "artifact_missing"
            rows.append(row)
            continue

        row.status = "ready"
        row.reason = "ok"
        safe_name = f"{zarr_path.stem}__{selected_run}__{artifact_name}.png"
        row.output_path = str(output_dir / safe_name)
        rows.append(row)

    rows.sort(key=lambda item: item.zarr_path)
    return rows


def _write_json_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_artifact_bytes(row: ExportRow) -> bytes:
    if row.profile_run is None:
        raise RuntimeError("missing profile run context")
    root = zarr.open_group(row.zarr_path, mode="r")
    vis_array = root[PROFILE_PARENT_PATH][row.profile_run]["visualizations"][row.artifact_name]
    return np.asarray(vis_array[:], dtype=np.uint8).tobytes()


def _view_png_bytes(png_bytes: bytes, *, title: str) -> None:
    image = plt.imread(BytesIO(png_bytes), format="png")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)
    plt.show()
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export eye-mask profile overview PNG artifacts stored in profile runs."
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
        "--profile-run",
        help="Specific eye-mask profile run to export from (default: latest profile run).",
    )
    parser.add_argument(
        "--artifact",
        choices=[ARTIFACT_NAME],
        default=ARTIFACT_NAME,
        help=f"Artifact to export/view (default: {ARTIFACT_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/eye_mask_quality_overview_exports"),
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
        profile_run=args.profile_run,
        output_dir=output_dir,
        artifact_name=str(args.artifact),
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
        assert row.profile_run is not None
        assert row.output_path is not None

        if args.list:
            listed += 1
            row.status = "listed"
            row.reason = "list_mode"
            print(f"list: {row.zarr_path}: {row.output_path}")
            continue

        if args.view:
            try:
                png_bytes = _load_artifact_bytes(row)
                _view_png_bytes(
                    png_bytes,
                    title=f"{Path(row.zarr_path).name} | {row.profile_run} | {row.artifact_name}",
                )
                row.status = "viewed"
                row.reason = "shown"
                row.bytes_written = len(png_bytes)
                viewed += 1
                print(f"viewed: {row.zarr_path}: {row.profile_run}")
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
            "Eye-mask quality artifact export: "
            f"scanned={scanned} ready={ready} listed={listed} skipped={skipped} errors={errors}"
        )
    elif args.view:
        print(
            "Eye-mask quality artifact export: "
            f"scanned={scanned} ready={ready} viewed={viewed} skipped={skipped} errors={errors}"
        )
    else:
        print(
            "Eye-mask quality artifact export: "
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
