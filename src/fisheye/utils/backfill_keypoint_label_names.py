from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import zarr


CANONICAL_LABELS = ("swim_bladder", "eye_left", "eye_right")
CANONICAL_TRADITIONAL_EDGES = ((0, 1), (0, 2), (1, 2))


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


def _canonicalize_label(value: object) -> str:
    text = str(value).strip()
    return "swim_bladder" if text.lower() == "bladder" else text


def _canonicalize_label_seq(values: Sequence[object]) -> tuple[list[str], bool]:
    out = [_canonicalize_label(v) for v in values]
    changed = any(str(v).strip() != new_v for v, new_v in zip(values, out))
    return out, changed


def _is_traditional_three_point_schema(schema: dict[str, object]) -> bool:
    name = str(schema.get("name", "")).strip().lower()
    if name == "traditional_v1":
        return True

    nodes = schema.get("nodes")
    if not isinstance(nodes, list):
        return False

    expected = {
        "swim_bladder": 0,
        "eye_left": 1,
        "eye_right": 2,
    }
    node_name_to_id: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        raw_name = node.get("name")
        raw_id = node.get("id")
        if raw_name is None or raw_id is None:
            continue
        try:
            node_id = int(raw_id)
        except Exception:
            continue
        node_name = _canonicalize_label(raw_name)
        node_name_to_id[node_name] = node_id
    return all(node_name_to_id.get(label) == idx for label, idx in expected.items())


def _canonicalize_edge_list(raw_edges: object) -> tuple[object, bool]:
    if not isinstance(raw_edges, (list, tuple)):
        return raw_edges, False

    out: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    changed = False
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            out.append(edge if isinstance(edge, list) else list(edge) if isinstance(edge, tuple) else edge)
            continue
        try:
            a = int(edge[0])
            b = int(edge[1])
        except Exception:
            out.append(list(edge))
            continue
        pair = (a, b) if a <= b else (b, a)
        if pair in seen:
            changed = True
            continue
        seen.add(pair)
        out.append([pair[0], pair[1]])
        if list(edge) != [pair[0], pair[1]]:
            changed = True
    return out, changed


def _ensure_triangle_edge(raw_edges: object) -> tuple[object, bool]:
    edges_out, changed = _canonicalize_edge_list(raw_edges)
    if not isinstance(edges_out, list):
        return edges_out, changed
    existing_pairs = set()
    for edge in edges_out:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        try:
            existing_pairs.add((int(edge[0]), int(edge[1])))
        except Exception:
            continue
    for pair in CANONICAL_TRADITIONAL_EDGES:
        if pair not in existing_pairs:
            edges_out.append([pair[0], pair[1]])
            existing_pairs.add(pair)
            changed = True
    return edges_out, changed


def _canonicalize_pose_schema(raw_schema: object) -> tuple[object, bool]:
    if not isinstance(raw_schema, dict):
        return raw_schema, False
    changed = False
    schema = dict(raw_schema)

    nodes = schema.get("nodes")
    if isinstance(nodes, list):
        nodes_out = []
        for node in nodes:
            if isinstance(node, dict):
                node_copy = dict(node)
                old_name = node_copy.get("name")
                if old_name is not None:
                    new_name = _canonicalize_label(old_name)
                    if str(old_name).strip() != new_name:
                        node_copy["name"] = new_name
                        changed = True
                nodes_out.append(node_copy)
            else:
                nodes_out.append(node)
        schema["nodes"] = nodes_out

    labels = schema.get("keypoint_labels")
    if isinstance(labels, (list, tuple)):
        labels_out, labels_changed = _canonicalize_label_seq(labels)
        if labels_changed:
            schema["keypoint_labels"] = labels_out
            changed = True

    if _is_traditional_three_point_schema(schema):
        edges = schema.get("edges")
        edges_out, edges_changed = _ensure_triangle_edge(edges)
        if edges_changed:
            schema["edges"] = edges_out
            changed = True

    return schema, changed


def _canonicalize_triangle_angle_order(values: object) -> tuple[object, bool]:
    if not isinstance(values, (list, tuple)):
        return values, False
    out: list[str] = []
    changed = False
    for item in values:
        text = str(item)
        replaced = text.replace("angle at bladder", "angle at swim_bladder")
        if replaced != text:
            changed = True
        out.append(replaced)
    return out, changed


def _backfill_run_group(run_group: zarr.Group, *, apply: bool) -> BackfillResult:
    updates: dict[str, object] = {}
    changed = False

    labels = run_group.attrs.get("keypoint_labels")
    if isinstance(labels, (list, tuple)):
        labels_out, labels_changed = _canonicalize_label_seq(labels)
        if labels_changed:
            updates["keypoint_labels"] = labels_out
            changed = True

    conf_labels = run_group.attrs.get("keypoint_confidence_labels")
    if isinstance(conf_labels, (list, tuple)):
        conf_labels_out, conf_changed = _canonicalize_label_seq(conf_labels)
        if conf_changed:
            updates["keypoint_confidence_labels"] = conf_labels_out
            changed = True

    angle_order = run_group.attrs.get("triangle_angle_order")
    angle_order_out, angle_changed = _canonicalize_triangle_angle_order(angle_order)
    if angle_changed:
        updates["triangle_angle_order"] = angle_order_out
        changed = True

    pose_schema = run_group.attrs.get("pose_schema")
    pose_schema_out, pose_changed = _canonicalize_pose_schema(pose_schema)
    if pose_changed:
        updates["pose_schema"] = pose_schema_out
        changed = True

    if not changed:
        return BackfillResult(status="skipped_existing")

    if apply:
        for key, value in updates.items():
            run_group.attrs[key] = value
    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill keypoint metadata labels from 'bladder' to "
            "'swim_bladder' in keypoint/refined-keypoint run attrs."
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
        "Keypoint label-name backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']}")
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
