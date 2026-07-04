from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr

from ..shared.zarr_helpers import _direct_group_names, _group_names, _open_group_direct, _open_mode, _root_fs_path
from .backfill_keypoint_label_names import _canonicalize_label_seq
from .zarr_io import open_zarr_root


SWIM_LABEL_ALIASES = ("swim_bladder", "bladder")
LEFT_EYE_LABEL_ALIASES = ("eye_left", "left_eye")
RIGHT_EYE_LABEL_ALIASES = ("eye_right", "right_eye")


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _select_runs(
    root: zarr.Group,
    all_runs: bool,
    *,
    zarr_path: Optional[Path] = None,
    open_mode: Optional[str] = None,
) -> list[zarr.Group]:
    groups: list[zarr.Group] = []
    root_fs_path = zarr_path.expanduser().resolve() if zarr_path is not None else _root_fs_path(root)
    resolved_open_mode = open_mode or _open_mode(root)
    for parent_name in ("keypoints_runs", "refined_keypoints_runs", "keypoints_refined_runs"):
        parent_fs_path = root_fs_path
        if parent_fs_path is not None:
            parent_fs_path = parent_fs_path / parent_name

        parent = root.get(parent_name)
        if parent is None and parent_fs_path is not None and parent_fs_path.is_dir():
            try:
                parent = _open_group_direct(parent_fs_path, mode=resolved_open_mode)
            except Exception:
                parent = None
        if parent is None:
            continue

        names = sorted(set(_group_names(parent)) | set(_direct_group_names(parent_fs_path)))
        if all_runs:
            selected_names = names
        else:
            latest = parent.attrs.get("latest")
            latest_name = str(latest) if latest else ""
            if latest_name and latest_name in names:
                selected_names = [latest_name]
            else:
                selected_names = [names[-1]] if names else []
        for name in selected_names:
            direct_path = (parent_fs_path / name) if parent_fs_path is not None else None
            if direct_path is not None and name in names:
                try:
                    groups.append(_open_group_direct(direct_path, mode=resolved_open_mode))
                except Exception:
                    if name in parent:
                        groups.append(parent[name])
                    continue
            elif name in parent:
                groups.append(parent[name])
    return groups


def _normalize_label_sequence(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    labels, _ = _canonicalize_label_seq(values)
    return [label for label in labels if label]


def _extract_pose_schema_labels(pose_schema: dict[str, object]) -> list[str]:
    keypoint_labels = _normalize_label_sequence(pose_schema.get("keypoint_labels"))
    if keypoint_labels:
        return keypoint_labels

    nodes = pose_schema.get("nodes")
    labels: list[str] = []
    if isinstance(nodes, list) and nodes:
        for node in nodes:
            if isinstance(node, dict):
                raw_name = node.get("name")
                if raw_name is not None:
                    labels.append(str(raw_name).strip())
            elif node is not None:
                labels.append(str(node).strip())
    return _normalize_label_sequence(labels)


def _resolve_effective_labels(pose_schema: dict[str, object], run_group: zarr.Group) -> list[str]:
    run_labels = _normalize_label_sequence(run_group.attrs.get("keypoint_labels"))
    if run_labels:
        return run_labels
    return _extract_pose_schema_labels(pose_schema)


def _find_first_label(labels: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    normalized = {str(label).strip().lower(): str(label).strip() for label in labels if str(label).strip()}
    for alias in aliases:
        match = normalized.get(alias.lower())
        if match:
            return match
    return None


def _build_heading_computation_spec(labels: Sequence[str]) -> Optional[dict[str, object]]:
    swim_label = _find_first_label(labels, SWIM_LABEL_ALIASES)
    left_eye_label = _find_first_label(labels, LEFT_EYE_LABEL_ALIASES)
    right_eye_label = _find_first_label(labels, RIGHT_EYE_LABEL_ALIASES)
    if not swim_label or not left_eye_label or not right_eye_label:
        return None

    return {
        "version": 1,
        "enabled": True,
        "origin": {"op": "midpoint", "labels": [left_eye_label, right_eye_label]},
        "direction_from": {"op": "keypoint", "label": swim_label},
        "direction_to": {"op": "midpoint", "labels": [left_eye_label, right_eye_label]},
        "dependent_keypoints": [swim_label, left_eye_label, right_eye_label],
    }


def _backfill_run_group(run_group: zarr.Group, *, apply: bool) -> BackfillResult:
    raw_pose_schema = run_group.attrs.get("pose_schema")
    if not isinstance(raw_pose_schema, dict):
        return BackfillResult(status="no_pose_schema", reason="pose_schema attr missing or invalid")

    pose_schema = deepcopy(raw_pose_schema)
    labels = _resolve_effective_labels(pose_schema, run_group)
    heading_spec = _build_heading_computation_spec(labels)
    if heading_spec is None:
        return BackfillResult(status="unsupported_labels", reason="heading keypoint labels not found")

    metadata = pose_schema.get("metadata")
    if isinstance(metadata, dict):
        metadata_out = deepcopy(metadata)
    else:
        metadata_out = {}

    existing = metadata_out.get("heading_computation")
    if existing == heading_spec:
        return BackfillResult(status="skipped_existing")

    metadata_out["heading_computation"] = heading_spec
    pose_schema["metadata"] = metadata_out

    if apply:
        run_group.attrs["pose_schema"] = pose_schema
    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill canonical pose_schema.metadata.heading_computation into "
            "keypoint and refined-keypoint run attrs."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all run groups (default: latest only).")
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")

    args = parser.parse_args(argv)
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]

    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_pose_schema": 0,
        "unsupported_labels": 0,
        "missing_runs": 0,
        "filtered_zarr_use": 0,
        "errors": 0,
    }

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = open_zarr_root(zarr_path, mode="a" if args.apply else "r")
            if args.zarr_use != "any":
                observed_use = _infer_zarr_use(root, zarr_path)
                if observed_use != args.zarr_use:
                    counts["filtered_zarr_use"] += 1
                    continue
            run_groups = _select_runs(
                root,
                all_runs=bool(args.all_runs),
                zarr_path=zarr_path,
                open_mode="a" if args.apply else "r",
            )
            if not run_groups:
                counts["missing_runs"] += 1
                continue
            for run_group in run_groups:
                counts["runs_considered"] += 1
                result = _backfill_run_group(run_group, apply=bool(args.apply))
                counts[result.status] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint heading-computation backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_pose_schema={counts['no_pose_schema']} unsupported_labels={counts['unsupported_labels']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
