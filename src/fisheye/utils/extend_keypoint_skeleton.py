from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.pose.schema import PoseSchema, schema_from_package
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.zarr_io import open_zarr_root


KEYPOINT_PARENT_CHOICES = ("auto", "keypoints_runs", "refined_keypoints_runs")
TARGET_SCHEMA_DEFAULT = "traditional_v2"
MIGRATION_REASON_LABEL = "needs_skeleton_extension"
EXPANDED_ARRAY_NAMES = frozenset(("keypoints_roi", "keypoints_img", "keypoints_norm", "keypoint_confidences"))
REFINED_BOOL_RESET_ARRAYS = frozenset(("refined_success", "confidence_valid", "geometry_valid", "usable_keypoints", "heading_usable"))
RUNTIME_COORD_DIMS = 2


@dataclass(frozen=True)
class ResolvedSourceRun:
    parent_name: str
    run_name: str
    group: zarr.Group


def _normalize_kpt_shape(value: object) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            first = int(value[0])
            second = int(value[1])
        except Exception:
            return None
        if first > 0 and second > 0:
            return [first, second]
    return None


def _infer_runtime_kpt_shape(source_group: zarr.Group) -> list[int]:
    attr_shape = _normalize_kpt_shape(source_group.attrs.get("kpt_shape"))
    if attr_shape is not None:
        return attr_shape

    pose_schema = source_group.attrs.get("pose_schema")
    if isinstance(pose_schema, dict):
        schema_shape = _normalize_kpt_shape(pose_schema.get("kpt_shape"))
        if schema_shape is not None:
            return schema_shape

    keypoints = source_group.get("keypoints_roi")
    if keypoints is not None and hasattr(keypoints, "shape") and int(len(keypoints.shape)) == 3:
        return [int(keypoints.shape[1]), int(keypoints.shape[2])]
    raise ValueError("Could not infer source kpt_shape from attrs or keypoints_roi.")


def _resolve_source_skeleton_id(source_group: zarr.Group) -> str | None:
    value = source_group.attrs.get("skeleton_id")
    if value:
        return str(value)
    pose_schema = source_group.attrs.get("pose_schema")
    if isinstance(pose_schema, dict):
        schema_id = pose_schema.get("skeleton_id")
        if schema_id:
            return str(schema_id)
        schema_name = pose_schema.get("name")
        if schema_name:
            return f"pose_schema:{schema_name}"
    return None


def _schema_to_attr_payload(schema_name: str) -> tuple[PoseSchema, dict[str, object]]:
    schema = schema_from_package(schema_name)
    metadata = dict(schema.metadata)
    skeleton_id = str(metadata.get("skeleton_id") or f"pose_schema:{schema.name}")
    labels = list(schema.node_names)
    payload = {
        "name": schema.name,
        "skeleton_id": skeleton_id,
        "kpt_shape": [int(schema.num_keypoints), int(RUNTIME_COORD_DIMS)],
        "keypoint_labels": labels,
        "nodes": [{"id": int(node.id), "name": str(node.name)} for node in schema.nodes],
        "edges": [[int(src), int(dst)] for src, dst in schema.edges],
        "metadata": metadata,
        "source": f"configs/fisheye/pose_schemas/{schema.name}.json",
    }
    return schema, payload


def _default_target_run_name(source_run: str, target_schema_name: str) -> str:
    return f"{source_run}_{target_schema_name}_seed"


def _resolve_source_run(
    root: zarr.Group,
    *,
    source_run: str,
    source_parent: str,
) -> ResolvedSourceRun:
    parent_names: Sequence[str]
    if source_parent == "auto":
        parent_names = ("refined_keypoints_runs", "keypoints_runs")
    else:
        parent_names = (source_parent,)

    errors: list[str] = []
    for parent_name in parent_names:
        try:
            group, run_name = resolve_zarr_run(
                root,
                parent_name,
                source_run,
                fallback_to_latest=False,
                run_label="Source keypoint run",
            )
            return ResolvedSourceRun(parent_name=parent_name, run_name=run_name, group=group)
        except Exception as exc:
            errors.append(f"{parent_name}: {exc}")
    raise ValueError("; ".join(errors) if errors else "Could not resolve source keypoint run.")


def _extend_coordinate_array(data: np.ndarray, *, target_keypoint_count: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 3 or int(arr.shape[2]) != int(RUNTIME_COORD_DIMS):
        raise ValueError(f"Expected coordinate array with shape (N,K,2), got {arr.shape}.")
    out = np.full((int(arr.shape[0]), int(target_keypoint_count), int(arr.shape[2])), np.nan, dtype=arr.dtype)
    copy_count = min(int(arr.shape[1]), int(target_keypoint_count))
    out[:, :copy_count, :] = arr[:, :copy_count, :]
    return out


def _extend_confidence_array(data: np.ndarray, *, target_keypoint_count: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Expected confidence array with shape (N,K), got {arr.shape}.")
    out = np.full((int(arr.shape[0]), int(target_keypoint_count)), np.nan, dtype=arr.dtype)
    copy_count = min(int(arr.shape[1]), int(target_keypoint_count))
    out[:, :copy_count] = arr[:, :copy_count]
    return out


def _copy_group_recursive(
    source_group: zarr.Group,
    target_group: zarr.Group,
    *,
    target_keypoint_count: int,
) -> None:
    target_group.attrs.update(dict(source_group.attrs))
    if hasattr(source_group, "array_keys"):
        array_names = sorted(str(name) for name in source_group.array_keys())
    else:
        array_names = [str(name) for name, item in source_group.items() if hasattr(item, "shape")]
    for name in array_names:
        source_arr = source_group[name]
        data = np.asarray(source_arr[:])
        if name in {"keypoints_roi", "keypoints_img", "keypoints_norm"}:
            data = _extend_coordinate_array(data, target_keypoint_count=target_keypoint_count)
        elif name == "keypoint_confidences":
            data = _extend_confidence_array(data, target_keypoint_count=target_keypoint_count)
        if data.dtype.kind in {"O", "U", "S"}:
            labels = np.asarray(
                ["" if value is None else str(value) for value in data.reshape(-1).tolist()],
                dtype=object,
            ).reshape(data.shape)
            target_arr = target_group.create_array(
                name,
                shape=labels.shape,
                dtype=VariableLengthUTF8(),
                fill_value="",
                overwrite=True,
            )
            target_arr[:] = labels
        else:
            target_group.create_array(name, data=data, overwrite=True)

    if hasattr(source_group, "group_keys"):
        subgroup_names = sorted(str(name) for name in source_group.group_keys())
    else:
        subgroup_names = [str(name) for name, item in source_group.items() if not hasattr(item, "shape")]
    for name in subgroup_names:
        child_target = target_group.create_group(name)
        _copy_group_recursive(
            source_group[name],
            child_target,
            target_keypoint_count=target_keypoint_count,
        )


def _reset_refined_seed_state(target_group: zarr.Group) -> None:
    row_count = int(target_group["frame_indices"].shape[0])
    if "refined_success" in target_group:
        target_group["refined_success"][:] = np.zeros((row_count,), dtype=bool)
    for name in REFINED_BOOL_RESET_ARRAYS:
        if name in target_group:
            target_group[name][:] = np.zeros((row_count,), dtype=bool)
    if "failure_indices" in target_group:
        del target_group["failure_indices"]
        target_group.create_array(
            "failure_indices",
            data=np.arange(row_count, dtype=np.int32),
            overwrite=True,
        )

    labels = read_reason_labels(target_group)
    if labels is None or int(labels.shape[0]) != row_count:
        labels = np.full((row_count,), MIGRATION_REASON_LABEL, dtype=object)
    else:
        labels = np.asarray(
            [
                MIGRATION_REASON_LABEL if not str(label).strip() else f"{str(label).strip()}|{MIGRATION_REASON_LABEL}"
                for label in labels.tolist()
            ],
            dtype=object,
        )
    if "reason_bytes" in target_group:
        del target_group["reason_bytes"]
    if "reason" in target_group:
        del target_group["reason"]
    write_reason_columns(target_group, labels, max(1, row_count), overwrite=True)


def _apply_target_metadata(
    target_group: zarr.Group,
    *,
    source: ResolvedSourceRun,
    source_kpt_shape: list[int],
    source_skeleton_id: str | None,
    target_schema_name: str,
    target_schema_payload: dict[str, object],
) -> None:
    target_labels = list(target_schema_payload["keypoint_labels"])
    target_group.attrs["keypoint_labels"] = target_labels
    if "keypoint_confidence_labels" in target_group.attrs or "keypoint_confidences" in target_group:
        target_group.attrs["keypoint_confidence_labels"] = target_labels
    target_group.attrs["kpt_shape"] = list(target_schema_payload["kpt_shape"])
    target_group.attrs["skeleton_id"] = str(target_schema_payload["skeleton_id"])
    target_group.attrs["pose_schema"] = dict(target_schema_payload)
    target_group.attrs["source_skeleton_id"] = str(source_skeleton_id) if source_skeleton_id else None
    target_group.attrs["source_kpt_shape"] = list(source_kpt_shape)
    source_pose_schema = source.group.attrs.get("pose_schema")
    if isinstance(source_pose_schema, dict):
        target_group.attrs["source_pose_schema"] = dict(source_pose_schema)
    target_group.attrs["migration_source_run"] = source.run_name
    target_group.attrs["migration_source_group"] = source.parent_name
    target_group.attrs["migration_source_keypoint_labels"] = list(
        source.group.attrs.get("keypoint_labels", [])
    )
    target_group.attrs["migration_target_schema"] = target_schema_name
    target_group.attrs["migration_status"] = "needs_keypoint_completion"
    target_group.attrs["migration_completion_required_keypoints"] = target_labels[int(source_kpt_shape[0]) :]
    target_group.attrs["migration_index_mapping"] = {
        str(idx): int(idx) for idx in range(min(int(source_kpt_shape[0]), len(target_labels)))
    }
    target_group.attrs["migration_created_at_utc"] = _utc_now()


def extend_keypoint_skeleton_run(
    root: zarr.Group,
    *,
    source_run: str,
    source_parent: str = "auto",
    target_run: str | None = None,
    target_schema: str = TARGET_SCHEMA_DEFAULT,
    overwrite: bool = False,
    set_latest: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    source = _resolve_source_run(root, source_run=source_run, source_parent=source_parent)
    source_kpt_shape = _infer_runtime_kpt_shape(source.group)
    source_skeleton_id = _resolve_source_skeleton_id(source.group)
    target_schema_obj, target_schema_payload = _schema_to_attr_payload(target_schema)
    target_run_name = target_run or _default_target_run_name(source.run_name, target_schema)
    target_parent_name = source.parent_name
    target_labels = list(target_schema_payload["keypoint_labels"])

    summary = {
        "status": "planned" if not apply else "updated",
        "source_parent": source.parent_name,
        "source_run": source.run_name,
        "target_parent": target_parent_name,
        "target_run": target_run_name,
        "source_skeleton_id": source_skeleton_id,
        "source_kpt_shape": list(source_kpt_shape),
        "target_skeleton_id": str(target_schema_payload["skeleton_id"]),
        "target_kpt_shape": list(target_schema_payload["kpt_shape"]),
        "target_schema": target_schema_obj.name,
        "target_labels": target_labels,
        "row_count": int(source.group["frame_indices"].shape[0]),
        "completion_required_keypoints": target_labels[int(source_kpt_shape[0]) :],
        "index_mapping": {
            str(idx): int(idx) for idx in range(min(int(source_kpt_shape[0]), len(target_labels)))
        },
        "set_latest": bool(set_latest),
    }
    if not apply:
        return summary

    parent = root.require_group(target_parent_name)
    if target_run_name in parent and not overwrite:
        raise ValueError(
            f"{target_parent_name}/{target_run_name} already exists. Pass --overwrite to replace it."
        )
    previous_latest = parent.attrs.get("latest")
    if target_run_name in parent and overwrite:
        del parent[target_run_name]

    target_group = parent.create_group(target_run_name)
    _copy_group_recursive(
        source.group,
        target_group,
        target_keypoint_count=int(target_schema_obj.num_keypoints),
    )
    target_group.attrs.update(dict(source.group.attrs))
    _apply_target_metadata(
        target_group,
        source=source,
        source_kpt_shape=source_kpt_shape,
        source_skeleton_id=source_skeleton_id,
        target_schema_name=target_schema,
        target_schema_payload=target_schema_payload,
    )
    if source.parent_name == "refined_keypoints_runs":
        _reset_refined_seed_state(target_group)

    if set_latest:
        parent.attrs["latest"] = target_run_name
    elif previous_latest is not None:
        parent.attrs["latest"] = previous_latest

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Recording zarr containing keypoint runs.")
    parser.add_argument("--source-run", required=True, help="Existing source keypoint run to extend.")
    parser.add_argument(
        "--source-parent",
        choices=list(KEYPOINT_PARENT_CHOICES),
        default="auto",
        help="Parent group containing the source run (default: auto).",
    )
    parser.add_argument("--target-run", help="Optional explicit target run name.")
    parser.add_argument(
        "--target-schema",
        default=TARGET_SCHEMA_DEFAULT,
        help=f"Target pose schema name from configs/fisheye/pose_schemas (default: {TARGET_SCHEMA_DEFAULT}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target run.")
    parser.add_argument("--set-latest", action="store_true", help="Update parent latest to the new run.")
    parser.add_argument("--apply", action="store_true", help="Write the extended seed run.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = open_zarr_root(args.zarr_path, mode="r+")
    summary = extend_keypoint_skeleton_run(
        root,
        source_run=str(args.source_run),
        source_parent=str(args.source_parent),
        target_run=str(args.target_run) if args.target_run else None,
        target_schema=str(args.target_schema),
        overwrite=bool(args.overwrite),
        set_latest=bool(args.set_latest),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"{summary['status'].upper()} {Path(args.zarr_path).expanduser().resolve()} "
            f"source={summary['source_parent']}/{summary['source_run']} "
            f"target={summary['target_parent']}/{summary['target_run']} "
            f"schema={summary['target_schema']}"
        )
        print(
            f"  source_skeleton_id={summary['source_skeleton_id']} "
            f"source_kpt_shape={summary['source_kpt_shape']}"
        )
        print(
            f"  target_skeleton_id={summary['target_skeleton_id']} "
            f"target_kpt_shape={summary['target_kpt_shape']}"
        )
        print(f"  completion_required_keypoints={summary['completion_required_keypoints']}")
        print(f"  set_latest={summary['set_latest']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
