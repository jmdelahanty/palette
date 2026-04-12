from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr


SWIM_LABEL_ALIASES = ("swim_bladder", "bladder")
LEFT_EYE_LABEL_ALIASES = ("eye_left", "left_eye")
RIGHT_EYE_LABEL_ALIASES = ("eye_right", "right_eye")


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _select_runs(root: zarr.Group, all_runs: bool) -> list[zarr.Group]:
    groups: list[zarr.Group] = []
    for parent_name in ("keypoints_runs", "refined_keypoints_runs", "keypoints_refined_runs"):
        parent = root.get(parent_name)
        if parent is None:
            continue
        if all_runs:
            try:
                names = sorted(list(parent.group_keys()))
            except Exception:
                names = sorted(list(parent.keys()))
        else:
            latest = parent.attrs.get("latest")
            if latest and latest in parent:
                names = [str(latest)]
            else:
                try:
                    all_names = sorted(list(parent.group_keys()))
                except Exception:
                    all_names = sorted(list(parent.keys()))
                names = [all_names[-1]] if all_names else []
        for name in names:
            if name in parent:
                groups.append(parent[name])
    return groups


def _extract_pose_schema_labels(pose_schema: dict[str, object], run_group: zarr.Group) -> list[str]:
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
    if labels:
        return [label for label in labels if label]

    keypoint_labels = pose_schema.get("keypoint_labels")
    if isinstance(keypoint_labels, (list, tuple)):
        labels = [str(value).strip() for value in keypoint_labels if str(value).strip()]
    if labels:
        return labels

    run_labels = run_group.attrs.get("keypoint_labels")
    if isinstance(run_labels, (list, tuple)):
        labels = [str(value).strip() for value in run_labels if str(value).strip()]
    return labels


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
    labels = _extract_pose_schema_labels(pose_schema, run_group)
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
            root = zarr.open_group(str(zarr_path), mode="a" if args.apply else "r")
            if args.zarr_use != "any":
                observed_use = _infer_zarr_use(root, zarr_path)
                if observed_use != args.zarr_use:
                    counts["filtered_zarr_use"] += 1
                    continue
            run_groups = _select_runs(root, all_runs=bool(args.all_runs))
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
