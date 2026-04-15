#!/usr/bin/env python3
"""Inspect the sparse refined-detect authoring surface for a single archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_KIND_CODE_MAP,
    build_source_detection_decision_summary,
    extract_present_curated_rows,
    extract_source_detection_rows,
    has_curated_refined_source_detections_projection,
    has_sparse_curated_refined_detect_instances_arrays,
    present_curated_row_mask,
    resolve_curated_refined_detect_run,
)


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _source_kind_labels_for_present_rows(refined_run: zarr.Group) -> np.ndarray:
    reverse = {int(value): str(key) for key, value in REFINED_SOURCE_KIND_CODE_MAP.items()}
    if has_sparse_curated_refined_detect_instances_arrays(refined_run):
        codes = np.asarray(refined_run["instances"]["source_kind_codes"][:], dtype=np.int8).reshape(-1)
    else:
        mask = present_curated_row_mask(refined_run)
        codes = np.asarray(refined_run["source_kind_codes"][:], dtype=np.int8).reshape(-1)[mask]
    return np.asarray([reverse.get(int(code), "unknown") for code in codes.tolist()], dtype=object)


def _count_labels(labels: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in np.asarray(labels, dtype=object).reshape(-1).tolist():
        text = str(label)
        counts[text] = counts.get(text, 0) + 1
    return counts


def _preview_instance_rows(refined_run: zarr.Group, *, limit: int) -> List[Dict[str, Any]]:
    payload = extract_present_curated_rows(refined_run)
    source_kind_labels = _source_kind_labels_for_present_rows(refined_run)
    row_count = int(np.asarray(payload["frame_indices"]).shape[0])
    reason = np.asarray(payload.get("reason", np.full(row_count, "", dtype=object)), dtype=object).reshape(-1)
    manual_flags = np.asarray(
        payload.get("manual_edit_flags", np.zeros(row_count, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    source_detect_row_index = np.asarray(
        payload.get("source_detect_row_index", np.full(row_count, -1, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    bbox_norm_coords = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    refined_row_ids = np.asarray(payload["refined_row_ids"], dtype=np.int64).reshape(-1)

    rows: List[Dict[str, Any]] = []
    for idx in range(min(max(int(limit), 0), row_count)):
        rows.append(
            {
                "refined_row_id": int(refined_row_ids[idx]),
                "frame_index": int(frame_indices[idx]),
                "source_kind": str(source_kind_labels[idx]),
                "manual_edit": bool(manual_flags[idx]),
                "source_detect_row_index": int(source_detect_row_index[idx]),
                "reason": str(reason[idx]),
                "bbox_norm_coords": [float(value) for value in bbox_norm_coords[idx].tolist()],
            }
        )
    return rows


def _collect_instances_summary(refined_run: zarr.Group, *, limit: int) -> Dict[str, Any]:
    payload = extract_present_curated_rows(refined_run)
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    source_kind_labels = _source_kind_labels_for_present_rows(refined_run)
    manual_flags = np.asarray(
        payload.get("manual_edit_flags", np.zeros(frame_indices.shape[0], dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    return {
        "total_instances": int(frame_indices.shape[0]),
        "frame_count": int(np.unique(frame_indices).shape[0]) if frame_indices.size else 0,
        "manual_edited_instances": int(np.count_nonzero(manual_flags)),
        "source_kind_counts": _count_labels(source_kind_labels),
        "preview": _preview_instance_rows(refined_run, limit=limit),
    }


def _collect_source_detections_summary(
    refined_run: zarr.Group,
    *,
    limit: int,
    decision_labels: Optional[Sequence[str]],
) -> Dict[str, Any]:
    if not has_curated_refined_source_detections_projection(refined_run):
        return {"available": False, "summary": {}, "preview": []}

    summary = build_source_detection_decision_summary(refined_run)
    payload = extract_source_detection_rows(refined_run, decision_labels=decision_labels)
    row_count = int(np.asarray(payload["frame_indices"]).shape[0])
    reason = np.asarray(payload.get("reason", np.full(row_count, "", dtype=object)), dtype=object).reshape(-1)
    bbox_norm_coords = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    source_detect_row_index = np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1)
    decision_names = np.asarray(payload["decision_labels"], dtype=object).reshape(-1)
    resolved_refined_row_id = np.asarray(payload["resolved_refined_row_id"], dtype=np.int64).reshape(-1)

    preview: List[Dict[str, Any]] = []
    for idx in range(min(max(int(limit), 0), row_count)):
        preview.append(
            {
                "source_detect_row_index": int(source_detect_row_index[idx]),
                "frame_index": int(frame_indices[idx]),
                "decision": str(decision_names[idx]),
                "resolved_refined_row_id": int(resolved_refined_row_id[idx]),
                "reason": str(reason[idx]),
                "bbox_norm_coords": [float(value) for value in bbox_norm_coords[idx].tolist()],
            }
        )

    return {
        "available": True,
        "summary": summary,
        "preview_decision_filter": [str(label) for label in (decision_labels or [])],
        "preview": preview,
    }


def collect_refined_detect_report(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    refined_run: Optional[str] = None,
    instance_limit: int = 10,
    source_limit: int = 10,
    source_decisions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    refined_group, resolved_run = resolve_curated_refined_detect_run(root, run_name=refined_run)
    summary_statistics = refined_group.attrs.get("summary_statistics")
    review_status = refined_group.attrs.get("detect_review_status")
    return {
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        "refined_run": resolved_run,
        "source_detect_run": _normalize_text(refined_group.attrs.get("source_detect_run")),
        "detect_review_status": dict(review_status) if isinstance(review_status, Mapping) else None,
        "summary_statistics": dict(summary_statistics) if isinstance(summary_statistics, Mapping) else None,
        "instances": _collect_instances_summary(refined_group, limit=instance_limit),
        "source_detections": _collect_source_detections_summary(
            refined_group,
            limit=source_limit,
            decision_labels=source_decisions,
        ),
    }


def inspect_refined_detect_run(
    zarr_path: Path,
    *,
    refined_run: Optional[str] = None,
    instance_limit: int = 10,
    source_limit: int = 10,
    source_decisions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    root = _open_root(zarr_path.expanduser())
    return collect_refined_detect_report(
        root,
        zarr_path=zarr_path.expanduser(),
        refined_run=refined_run,
        instance_limit=instance_limit,
        source_limit=source_limit,
        source_decisions=source_decisions,
    )


def _print_text_report(payload: Dict[str, Any]) -> None:
    print(f"Archive: {payload.get('zarr_path') or '-'}")
    print(f"Refined run: {payload.get('refined_run') or '-'}")
    print(f"Source detect run: {payload.get('source_detect_run') or '-'}")
    review_status = payload.get("detect_review_status")
    if isinstance(review_status, Mapping):
        print(f"Detect review status: {json.dumps(dict(review_status), sort_keys=True)}")
    else:
        print("Detect review status: none")
    print()

    summary_statistics = payload.get("summary_statistics")
    print("Run summary:")
    if isinstance(summary_statistics, Mapping) and summary_statistics:
        for key in sorted(summary_statistics):
            print(f"- {key}: {summary_statistics[key]}")
    else:
        print("- none")
    print()

    instances = payload.get("instances", {})
    print("Instances:")
    print(f"- total_instances: {instances.get('total_instances', 0)}")
    print(f"- frame_count: {instances.get('frame_count', 0)}")
    print(f"- manual_edited_instances: {instances.get('manual_edited_instances', 0)}")
    source_kind_counts = instances.get("source_kind_counts", {})
    if source_kind_counts:
        print(f"- source_kind_counts: {json.dumps(source_kind_counts, sort_keys=True)}")
    else:
        print("- source_kind_counts: {}")
    preview_rows = instances.get("preview", [])
    print("- preview:")
    if preview_rows:
        for row in preview_rows:
            print(
                f"  - row_id={row.get('refined_row_id')} frame={row.get('frame_index')} "
                f"source_kind={row.get('source_kind')} manual={row.get('manual_edit')} "
                f"raw_row={row.get('source_detect_row_index')} reason={row.get('reason')}"
            )
    else:
        print("  - none")
    print()

    source = payload.get("source_detections", {})
    print("Source detections:")
    print(f"- available: {bool(source.get('available', False))}")
    summary = source.get("summary", {})
    if summary:
        print(f"- summary: {json.dumps(summary, sort_keys=True)}")
    else:
        print("- summary: {}")
    decision_filter = source.get("preview_decision_filter") or []
    if decision_filter:
        print(f"- preview_decision_filter: {decision_filter}")
    preview_rows = source.get("preview", [])
    print("- preview:")
    if preview_rows:
        for row in preview_rows:
            print(
                f"  - raw_row={row.get('source_detect_row_index')} frame={row.get('frame_index')} "
                f"decision={row.get('decision')} resolved_row={row.get('resolved_refined_row_id')} "
                f"reason={row.get('reason')}"
            )
    else:
        print("  - none")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to a .zarr archive.")
    parser.add_argument("--refined-run", help="Specific refined detect run to inspect (default: latest).")
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=10,
        help="Number of instance preview rows to print (default: 10).",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=10,
        help="Number of source-detection preview rows to print (default: 10).",
    )
    parser.add_argument(
        "--source-decision",
        dest="source_decisions",
        action="append",
        help="Restrict source-detection preview rows to a decision label. Repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    payload = inspect_refined_detect_run(
        args.zarr_path,
        refined_run=args.refined_run,
        instance_limit=max(0, int(args.instance_limit)),
        source_limit=max(0, int(args.source_limit)),
        source_decisions=args.source_decisions,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_text_report(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
