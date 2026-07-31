"""Materialize a current-v2 coordinate successor for one historical proxy rowset.

The historical source is fully revalidated while the authoritative archive is
read-only.  Compact geometry and lineage arrays are staged on node-local
storage, copied to a hidden sibling, and atomically published.  Current
coordinate records are minted only after the copied arrays have their final
authoritative paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from ...shared.historical_collection_proxy_v1 import (
    load_historical_merged_collection_proxy_v1,
)
from ...shared.json_safety import json_attr_safe
from ...shared.observation_coordinate_publication import (
    COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA,
    COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND,
    OBSERVATION_ROW_COUNT_ATTR,
    _load_persisted_collection_proxy_successor_geometry,
    publish_collection_proxy_successor_mapping,
    publish_detection_observation_geometry,
)
from ...shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from ...shared.proof_verification import proof_verification_scope
from ...shared.run_provenance import build_writer_run_provenance
from ...shared.zarr.chunk_profiles import create_geometry_preload_array
from ...shared.zarr_io import open_zarr_root
from ...shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from .atomic_run_publisher import AtomicRunPublishSpec, atomic_publish_run_group

MATERIALIZATION_SCHEMA_ID = "palette.collection_proxy_coordinate_materialization.v1"
PUBLISH_SCHEMA_ID = "palette.collection_proxy_coordinate_publish.v1"
COORDINATE_CONTRACT = "canonical_v2"
COPIED_ARRAY_NAMES = (
    "instance_key",
    "frame_indices",
    "source_frame_indices",
    "source_acquisition_frame_index",
    "source_proxy_crop_run_index",
    "source_proxy_crop_row_ids",
    "bbox_norm_coords",
)
GEOMETRY_ARRAY_NAMES = ("bbox_img_xyxy", "centers_img_xy")


@dataclass(frozen=True)
class CollectionProxyCoordinateMaterializationPlan:
    source_zarr: Path
    historical_rowset: str
    scratch_root: Path
    local_zarr: Path
    run_name: str

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr / "crop_runs" / self.run_name

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr / "crop_runs" / self.run_name

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": MATERIALIZATION_SCHEMA_ID,
            "source_zarr": str(self.source_zarr),
            "historical_rowset": self.historical_rowset,
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "local_run_path": str(self.local_run_path),
            "target_run_path": str(self.target_run_path),
            "run_name": self.run_name,
        }


def _validate_run_name(run_name: str) -> str:
    value = str(run_name).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Unsafe coordinate-successor run name: {run_name!r}.")
    return value


def _canonical_rowset_path(value: str) -> str:
    path = str(value).strip().strip("/")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "crop_runs" or not parts[1]:
        raise ValueError("historical_rowset must be an exact crop_runs/<run> path.")
    return path


def build_collection_proxy_coordinate_materialization_plan(
    source_zarr: str | Path,
    *,
    historical_rowset: str,
    scratch_root: str | Path,
    run_name: str,
) -> CollectionProxyCoordinateMaterializationPlan:
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source analysis Zarr not found: {source}")
    try:
        scratch.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Scratch root must not be inside the authoritative Zarr.")
    name = _validate_run_name(run_name)
    target = source / "crop_runs" / name
    if target.exists():
        raise FileExistsError(
            f"Refusing to replace existing authoritative run: {target}"
        )
    return CollectionProxyCoordinateMaterializationPlan(
        source_zarr=source,
        historical_rowset=_canonical_rowset_path(historical_rowset),
        scratch_root=scratch,
        local_zarr=scratch / "collection-proxy-coordinate-output.zarr",
        run_name=name,
    )


def derive_current_geometry(
    bbox_norm_coords: Any,
    *,
    width_px: int,
    height_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact current-v2 normalized-box projection and midpoint rules."""

    normalized = np.asarray(bbox_norm_coords)
    if (
        normalized.dtype.kind != "f"
        or normalized.ndim != 2
        or normalized.shape[1:] != (4,)
        or not np.isfinite(normalized).all()
        or type(width_px) is not int
        or type(height_px) is not int
        or width_px <= 0
        or height_px <= 0
    ):
        raise ValueError(
            "bbox_norm_coords must be finite floating (N,4), with positive "
            "integer camera dimensions."
        )
    dtype = normalized.dtype
    half = np.asarray(0.5, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)
    cx, cy, width, height = (normalized[:, index] for index in range(4))
    x_min = cx - width * half
    y_min = cy - height * half
    x_max = cx + width * half
    y_max = cy + height * half
    if normalized.shape[0] and (
        np.any(width <= 0)
        or np.any(height <= 0)
        or np.any(x_min < 0)
        or np.any(y_min < 0)
        or np.any(x_max > one)
        or np.any(y_max > one)
    ):
        raise ValueError(
            "Canonical normalized boxes must have positive extents and remain "
            "inside the source-camera extent."
        )
    width_value = np.asarray(width_px, dtype=dtype)
    height_value = np.asarray(height_px, dtype=dtype)
    bbox_img = np.column_stack(
        (
            x_min * width_value,
            y_min * height_value,
            x_max * width_value,
            y_max * height_value,
        )
    ).astype(dtype, copy=False)
    centers = np.column_stack(
        (
            (bbox_img[:, 0] + bbox_img[:, 2]) * half,
            (bbox_img[:, 1] + bbox_img[:, 3]) * half,
        )
    ).astype(dtype, copy=False)
    return (
        np.array(bbox_img, copy=True, order="C"),
        np.array(centers, copy=True, order="C"),
    )


def _validate_materialized_run(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        group = open_zarr_root(path, mode="r")
        attrs = dict(group.attrs)
    except Exception as exc:
        return {"valid": False, "errors": [f"cannot open run: {exc}"]}
    if attrs.get("schema") != COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA:
        errors.append("invalid schema")
    if attrs.get("source_kind") != COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND:
        errors.append("invalid source_kind")
    if attrs.get("coordinate_contract") != COORDINATE_CONTRACT:
        errors.append("invalid coordinate_contract")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append("run must remain selector-ineligible during validation")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) not in {"running", "complete"}:
        errors.append("run is neither running nor complete")
    try:
        arrays = {
            name: np.asarray(group[name][:])
            for name in (*COPIED_ARRAY_NAMES, *GEOMETRY_ARRAY_NAMES)
        }
    except Exception as exc:
        errors.append(f"cannot read required arrays: {exc}")
        arrays = {}
    if arrays:
        rows = int(arrays["instance_key"].shape[0])
        expected_shapes = {
            "instance_key": (rows,),
            "frame_indices": (rows,),
            "source_frame_indices": (rows,),
            "source_acquisition_frame_index": (rows,),
            "source_proxy_crop_run_index": (rows,),
            "source_proxy_crop_row_ids": (rows,),
            "bbox_norm_coords": (rows, 4),
            "bbox_img_xyxy": (rows, 4),
            "centers_img_xy": (rows, 2),
        }
        if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
            errors.append("required arrays are not exactly row-aligned")
        if arrays["instance_key"].dtype != np.dtype("<u8"):
            errors.append("instance_key is not uint64")
        if arrays["source_acquisition_frame_index"].dtype != np.dtype("<i8"):
            errors.append("source_acquisition_frame_index is not int64")
        if attrs.get(OBSERVATION_ROW_COUNT_ATTR) != rows:
            errors.append("declared observation row count is not exact")
        try:
            expected_bbox, expected_centers = derive_current_geometry(
                arrays["bbox_norm_coords"],
                width_px=int(attrs["source_camera_width_px"]),
                height_px=int(attrs["source_camera_height_px"]),
            )
            if not np.array_equal(
                arrays["bbox_img_xyxy"], expected_bbox, equal_nan=True
            ):
                errors.append("bbox_img_xyxy differs from current-v2 projection")
            if not np.array_equal(
                arrays["centers_img_xy"], expected_centers, equal_nan=True
            ):
                errors.append("centers_img_xy differs from current-v2 midpoint")
        except Exception as exc:
            errors.append(f"current-v2 geometry validation failed: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": attrs.get(OBSERVATION_ROW_COUNT_ATTR),
        "completion_status": attrs.get(RUN_COMPLETION_STATUS_ATTR),
    }


def _stage_local_run(
    plan: CollectionProxyCoordinateMaterializationPlan,
) -> dict[str, Any]:
    source_root = open_zarr_root(plan.source_zarr, mode="r")
    with proof_verification_scope():
        historical = load_historical_merged_collection_proxy_v1(
            source_root,
            plan.historical_rowset,
        )
        _, acquisition = load_persisted_acquisition_camera_authority(source_root)
        copied = {name: historical.read_array(name) for name in COPIED_ARRAY_NAMES}
        bbox_img, centers = derive_current_geometry(
            copied["bbox_norm_coords"],
            width_px=int(acquisition.record.width_px),
            height_px=int(acquisition.record.height_px),
        )
        source_summary = {
            "historical_rowset": historical.rowset_path,
            "historical_row_count": historical.row_count,
            "historical_proxy_run_count": historical.source_proxy_run_count,
            "historical_refined_run_count": historical.source_refined_run_count,
            "camera_id": historical.camera_id,
            "acquisition_camera_frame": {
                "record_ref": acquisition.record_ref,
                "record_sha256": acquisition.record_sha256,
            },
        }

    local_root = open_zarr_root(plan.local_zarr, mode="w")
    parent = require_runs_parent(local_root, "crop_runs")
    run = parent.require_group(plan.run_name)
    run.attrs.update(
        {
            "schema": COLLECTION_PROXY_SUCCESSOR_RUN_SCHEMA,
            "source_kind": COLLECTION_PROXY_SUCCESSOR_SOURCE_KIND,
            "coordinate_contract": COORDINATE_CONTRACT,
            "historical_source_rowset_path": plan.historical_rowset,
            OBSERVATION_ROW_COUNT_ATTR: int(copied["instance_key"].shape[0]),
            "row_count": int(copied["instance_key"].shape[0]),
            "source_camera_width_px": int(acquisition.record.width_px),
            "source_camera_height_px": int(acquisition.record.height_px),
            "stage_selector_eligible": False,
            "materialization_schema_id": MATERIALIZATION_SCHEMA_ID,
            "historical_source_summary": json_attr_safe(source_summary),
        }
    )
    mark_run_started(
        run,
        run_name=plan.run_name,
        stage="collection_proxy_coordinate_successor",
    )
    for name, values in copied.items():
        create_geometry_preload_array(run, name, data=values, overwrite=True)
    create_geometry_preload_array(
        run,
        "bbox_img_xyxy",
        data=bbox_img,
        overwrite=True,
    )
    create_geometry_preload_array(
        run,
        "centers_img_xy",
        data=centers,
        overwrite=True,
    )
    validation = _validate_materialized_run(plan.local_run_path)
    if not validation["valid"]:
        raise RuntimeError(f"Local coordinate successor is invalid: {validation}")
    return {"source": source_summary, "local_validation": validation}


def _publish_run(
    plan: CollectionProxyCoordinateMaterializationPlan,
    *,
    payload: dict[str, Any],
    copy_backend: str,
) -> dict[str, Any]:
    def prepare(root: zarr.Group) -> tuple[zarr.Group]:
        return (require_runs_parent(root, "crop_runs"),)

    def after_rename(root: zarr.Group, run: zarr.Group) -> dict[str, Any]:
        # Keep the heavyweight detector stack out of planning and node-local
        # geometry computation; only its shared publication helper is needed.
        from ...detection.detect_yolo import _publish_detection_frame_evidence

        with proof_verification_scope():
            historical = load_historical_merged_collection_proxy_v1(
                root,
                plan.historical_rowset,
            )
            _, acquisition = load_persisted_acquisition_camera_authority(root)
            evidence, _checkpoints = _publish_detection_frame_evidence(
                root,
                run,
                acquisition_frame=acquisition,
            )
            mapping = publish_collection_proxy_successor_mapping(
                run,
                historical_source=historical,
                acquisition_frame=acquisition,
            )
            publish_detection_observation_geometry(
                run,
                run["instance_key"],
                run["source_acquisition_frame_index"],
                run["bbox_norm_coords"],
                run["bbox_img_xyxy"],
                run["centers_img_xy"],
                frame_evidence=evidence,
                source_lineage_records=(mapping,),
            )
        return {
            "coordinate_binding": {
                "status": "bound_current_v2_at_authoritative_path",
                "historical_rowset": plan.historical_rowset,
                "mapping_record_ref": mapping.record_ref,
                "mapping_record_sha256": mapping.record_sha256,
            }
        }

    def complete(
        _root: zarr.Group,
        _parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        mark_run_complete(
            run,
            run_name=plan.run_name,
            run_provenance=build_writer_run_provenance(
                command="collection_proxy_coordinate_successor_materializer",
                params={
                    "coordinate_contract": COORDINATE_CONTRACT,
                    "copy_backend": copy_backend,
                },
                input_run_ids={"historical_rowset": plan.historical_rowset},
            ),
        )

    def verify(root: zarr.Group) -> None:
        with proof_verification_scope():
            _load_persisted_collection_proxy_successor_geometry(
                root,
                f"crop_runs/{plan.run_name}",
                require_selector_eligible=False,
            )

    def activate(
        _root: zarr.Group,
        _parent: zarr.Group,
        run: zarr.Group,
    ) -> None:
        if (
            run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "complete"
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                "Coordinate-successor activation requires one complete, "
                "selector-ineligible run."
            )
        run.attrs["stage_selector_eligible"] = True

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=plan.source_zarr,
            local_run_path=plan.local_run_path,
            target_run_path=plan.target_run_path,
            run_name=plan.run_name,
            lock_suffix="collection-proxy-coordinate-publish",
            publish_schema_id=PUBLISH_SCHEMA_ID,
            policy="node_local_geometry_compute_atomic_run_group_publish",
            rollback_policy=(
                "retain_failed_public_tombstone_leave_parent_selectors_untouched"
            ),
        ),
        copy_backend=copy_backend,
        validate_run=_validate_materialized_run,
        prepare_parents=prepare,
        after_rename=after_rename,
        complete_run=complete,
        verify_pointers=verify,
        activate_run=activate,
        payload_metadata={
            "copy_backend": copy_backend,
            "materialization": json_attr_safe(payload),
            "activation_policy": "selector_eligibility_literal_final_write",
        },
    )


def materialize_collection_proxy_coordinates(
    source_zarr: str | Path,
    *,
    historical_rowset: str,
    scratch_root: str | Path,
    run_name: str,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = build_collection_proxy_coordinate_materialization_plan(
        source_zarr,
        historical_rowset=historical_rowset,
        scratch_root=scratch_root,
        run_name=run_name,
    )
    result: dict[str, Any] = {
        "schema_id": MATERIALIZATION_SCHEMA_ID,
        "status": "planned" if not apply else "running",
        "mutates_archive": bool(apply),
        "plan": plan.to_json(),
    }
    if not apply:
        return result
    if plan.scratch_root.exists():
        raise FileExistsError(f"Refusing existing scratch root: {plan.scratch_root}")
    plan.scratch_root.mkdir(parents=True)
    succeeded = False
    try:
        started = time.perf_counter()
        payload = _stage_local_run(plan)
        payload["compute_duration_seconds"] = float(time.perf_counter() - started)
        local = open_zarr_root(plan.local_run_path, mode="a")
        local.attrs["node_local_materialization"] = json_attr_safe(payload)
        publish = _publish_run(plan, payload=payload, copy_backend=copy_backend)
        result.update(status="complete", local_materialization=payload, publish=publish)
        succeeded = True
        return result
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch_root(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job_id = os.environ.get("LSB_JOBID") or "manual"
    scratch_user = Path("/scratch") / user
    if scratch_user.is_dir() and os.access(scratch_user, os.W_OK | os.X_OK):
        return scratch_user / job_id / f"palette_coordinate_successor_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_coordinate_successor_{job_id}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--historical-rowset", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_collection_proxy_coordinates(
        args.zarr_path,
        historical_rowset=args.historical_rowset,
        scratch_root=args.scratch_root or _default_scratch_root(args.run_name),
        run_name=args.run_name,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CollectionProxyCoordinateMaterializationPlan",
    "build_collection_proxy_coordinate_materialization_plan",
    "derive_current_geometry",
    "materialize_collection_proxy_coordinates",
]
