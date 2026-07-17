"""Audit a finalized clipped refined-detection collection against a dish mask.

This tool is intentionally read-only with respect to the analysis Zarr.  It
tests whether the *selected refined instance rowset* would lose any rows if the
current dish-mask bbox-center gate were applied.  It does not assert that the
source refinement originally enabled the gate, and it does not claim bitwise
equivalence for the source-detection decision/reason surfaces.
"""

from __future__ import annotations

from fisheye.refinement.refine_detect import (
    _dish_mask_inside_bbox_centers,
    _read_dish_mask_spec,
)
from fisheye.shared.dish_mask_boundary import (
    DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM,
    apply_dish_mask_boundary_tolerance,
    resolve_dish_mask_boundary_tolerance,
)
from fisheye.shared.batch_logging import utc_now as _utc_now
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.utils.finalize_clipped_detect_refine_workflow import COLLECTION_SCHEMA
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr


AUDIT_SCHEMA = "palette.dish_mask_posthoc_equivalence_audit.v2"
AUDIT_SCOPE = "selected_refined_instance_rowset"

_AFFECTED_SCHEMA = pa.schema(
    [
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("camera_serial", pa.string(), nullable=False),
        pa.field("refined_group_path", pa.string(), nullable=False),
        pa.field("member_row_index", pa.int64(), nullable=False),
        pa.field("clip_local_frame_index", pa.int64(), nullable=False),
        pa.field("instance_key", pa.uint64(), nullable=False),
        pa.field("bbox_norm_cx", pa.float64(), nullable=False),
        pa.field("bbox_norm_cy", pa.float64(), nullable=False),
        pa.field("bbox_norm_w", pa.float64(), nullable=False),
        pa.field("bbox_norm_h", pa.float64(), nullable=False),
        pa.field("dish_circle_radial_ratio", pa.float64(), nullable=True),
    ],
    metadata={
        b"palette.schema_version": AUDIT_SCHEMA.encode("utf-8"),
        b"palette.table_role": b"selected_refined_instances_outside_dish_mask",
    },
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(digest: Any, values: np.ndarray, *, dtype: np.dtype[Any]) -> None:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest.update(canonical.tobytes(order="C"))


def _resolve_collection(
    root: zarr.Group,
    *,
    collection_path: str | None,
) -> tuple[str, str, zarr.Group, list[dict[str, Any]]]:
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise ValueError("Analysis Zarr has no refined_detect_runs group")

    explicit_path = str(collection_path or "").strip("/")
    latest_path = str(refined_parent.attrs.get("latest_collection_path") or "").strip("/")
    latest_id = str(refined_parent.attrs.get("latest_collection") or "").strip()
    resolved_path = explicit_path or latest_path
    if not resolved_path and latest_id:
        resolved_path = f"experiment_index/finalized_runs/{latest_id}"
    if not resolved_path:
        raise ValueError(
            "No --collection-path was provided and refined_detect_runs has no "
            "latest_collection_path/latest_collection pointer"
        )
    try:
        collection = root[resolved_path]
    except Exception as exc:
        raise ValueError(f"Finalized refined-detection collection not found: {resolved_path}") from exc

    attrs = dict(collection.attrs)
    if attrs.get("schema_version") != COLLECTION_SCHEMA:
        raise ValueError(
            f"Expected collection schema {COLLECTION_SCHEMA!r}, got "
            f"{attrs.get('schema_version')!r}: {resolved_path}"
        )
    collection_id = str(attrs.get("collection_id") or resolved_path.rsplit("/", 1)[-1]).strip()
    if not collection_id:
        raise ValueError(f"Finalized collection has no collection_id: {resolved_path}")
    if latest_id and not explicit_path and collection_id != latest_id:
        raise ValueError(
            "refined_detect_runs latest_collection and latest_collection_path disagree: "
            f"{latest_id!r} != {collection_id!r}"
        )

    raw_selected = attrs.get("selected_runs")
    if not isinstance(raw_selected, list) or not raw_selected:
        raise ValueError(f"Finalized collection has no selected_runs: {resolved_path}")
    selected = [dict(row) for row in raw_selected if isinstance(row, Mapping)]
    if len(selected) != len(raw_selected):
        raise ValueError(f"Finalized collection selected_runs contains non-object entries: {resolved_path}")
    declared_count = attrs.get("selected_run_count")
    if declared_count is None or int(declared_count) != len(selected):
        raise ValueError(
            f"Finalized collection selected_run_count {declared_count!r} does not match "
            f"selected_runs length {len(selected)}"
        )

    seen_pairs: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    for row in selected:
        clip_id = str(row.get("clip_id") or "")
        camera_serial = str(row.get("camera_serial") or "")
        refined_path = str(row.get("refined_group_path") or "").strip("/")
        if not clip_id or not camera_serial or not refined_path:
            raise ValueError(
                "Every selected run must declare clip_id, camera_serial, and refined_group_path"
            )
        pair = (camera_serial, clip_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate selected camera/clip pair: {pair}")
        if refined_path in seen_paths:
            raise ValueError(f"Duplicate selected refined_group_path: {refined_path}")
        seen_pairs.add(pair)
        seen_paths.add(refined_path)

    selected.sort(
        key=lambda row: (
            int(row.get("clip_index")) if row.get("clip_index") is not None else 2**31 - 1,
            str(row.get("camera_serial") or ""),
            str(row.get("clip_id") or ""),
            str(row.get("refined_group_path") or ""),
        )
    )
    return collection_id, resolved_path, collection, selected


def _inspect_selected_runs(
    root: zarr.Group,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[dict[str, Any], zarr.Group, int]], int]:
    inspected: list[tuple[dict[str, Any], zarr.Group, int]] = []
    total_rows = 0
    for selected_row in selected:
        row = dict(selected_row)
        path = str(row["refined_group_path"]).strip("/")
        try:
            refined = root[path]
        except Exception as exc:
            raise ValueError(f"Selected refined run not found: {path}") from exc
        if not is_run_complete(refined, legacy_default=False):
            raise ValueError(f"Selected refined run is not explicitly complete: {path}")
        if "instances" not in refined:
            raise ValueError(f"Selected refined run has no instances group: {path}")
        instances = refined["instances"]
        required = ("bbox_norm_coords", "instance_key", "frame_indices")
        missing = [name for name in required if name not in instances]
        if missing:
            raise ValueError(f"Selected refined run is missing modern arrays {missing}: {path}")
        bbox = instances["bbox_norm_coords"]
        keys = instances["instance_key"]
        frames = instances["frame_indices"]
        if len(bbox.shape) != 2 or int(bbox.shape[1]) != 4:
            raise ValueError(f"bbox_norm_coords must have shape (n, 4): {path}")
        row_count = int(bbox.shape[0])
        if int(keys.shape[0]) != row_count or int(frames.shape[0]) != row_count:
            raise ValueError(f"Selected refined instance arrays disagree on row count: {path}")
        if np.dtype(keys.dtype) != np.dtype(np.uint64):
            raise ValueError(f"instances/instance_key must be uint64: {path}")
        inspected.append((row, instances, row_count))
        total_rows += row_count
    return inspected, total_rows


def _atomic_affected_writer(path: Path, *, overwrite: bool) -> tuple[pq.ParquetWriter, Path]:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    tmp_path = Path(raw_tmp)
    try:
        writer = pq.ParquetWriter(tmp_path, _AFFECTED_SCHEMA, compression="zstd")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return writer, tmp_path


def audit_clipped_dish_mask_equivalence(
    zarr_path: str | Path,
    *,
    collection_path: str | None = None,
    output_parquet: str | Path,
    chunk_rows: int = 131_072,
    dish_mask_boundary_tolerance_mm: float = DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM,
    pixels_per_mm_camera: float | None = None,
    overwrite_evidence: bool = False,
    root: zarr.Group | None = None,
) -> dict[str, Any]:
    """Audit the selected collection rowset and write outside-mask keys.

    The full collection is never materialized in RAM. Arrays are scanned in
    ``chunk_rows`` batches, and collection-wide key uniqueness is checked using
    a temporary disk-backed uint64 memmap.
    """

    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    archive_path = Path(zarr_path).expanduser().resolve()
    opened_root = root or open_zarr_group_direct(archive_path, mode="r")
    collection_id, resolved_path, collection, selected = _resolve_collection(
        opened_root,
        collection_path=collection_path,
    )
    inspected, total_rows = _inspect_selected_runs(opened_root, selected)
    if not inspected:
        raise ValueError(f"Finalized collection has no inspectable selected runs: {resolved_path}")
    if total_rows <= 0:
        raise ValueError(f"Finalized collection has zero selected instance rows: {resolved_path}")

    mask_spec = _read_dish_mask_spec(opened_root, inspected[0][1])
    if not mask_spec or not bool(mask_spec.get("enabled")):
        raise ValueError("analysis_metadata.dish_mask cannot be resolved to an enabled gate")
    tolerance = resolve_dish_mask_boundary_tolerance(
        opened_root,
        source_group=inspected[0][1],
        tolerance_mm=dish_mask_boundary_tolerance_mm,
        pixels_per_mm_camera=pixels_per_mm_camera,
    )
    mask_spec = apply_dish_mask_boundary_tolerance(mask_spec, tolerance)

    affected_path = Path(output_parquet).expanduser().resolve()
    writer, tmp_parquet = _atomic_affected_writer(
        affected_path,
        overwrite=overwrite_evidence,
    )
    key_digest = hashlib.sha256()
    bbox_digest = hashlib.sha256()
    outside_key_digest = hashlib.sha256()
    outside_count = 0
    outside_circle_ratio_min = float("inf")
    outside_circle_ratio_max = float("-inf")
    outside_circle_ratio_sum = 0.0
    outside_circle_ratio_count = 0
    run_reports: list[dict[str, Any]] = []
    key_memmap_path: Path | None = None
    try:
        descriptor, raw_memmap = tempfile.mkstemp(prefix="palette_dish_audit_keys_", suffix=".u64")
        os.close(descriptor)
        key_memmap_path = Path(raw_memmap)
        key_memmap = np.memmap(key_memmap_path, mode="w+", dtype=np.uint64, shape=(total_rows,))
        key_cursor = 0
        for selected_row, instances, row_count in inspected:
            run_outside = 0
            path = str(selected_row["refined_group_path"]).strip("/")
            clip_id = str(selected_row["clip_id"])
            camera_serial = str(selected_row["camera_serial"])
            marker = f"{path}\n".encode("utf-8")
            key_digest.update(marker)
            bbox_digest.update(marker)
            for start in range(0, row_count, int(chunk_rows)):
                stop = min(start + int(chunk_rows), row_count)
                bboxes = np.asarray(
                    instances["bbox_norm_coords"][start:stop],
                    dtype=np.float64,
                ).reshape(-1, 4)
                keys = np.asarray(
                    instances["instance_key"][start:stop],
                    dtype=np.uint64,
                ).reshape(-1)
                frames = np.asarray(
                    instances["frame_indices"][start:stop],
                    dtype=np.int64,
                ).reshape(-1)
                expected = stop - start
                if bboxes.shape[0] != expected or keys.shape[0] != expected or frames.shape[0] != expected:
                    raise ValueError(f"Short array read while auditing {path} rows {start}:{stop}")
                _hash_array(key_digest, keys, dtype=np.dtype("<u8"))
                _hash_array(bbox_digest, bboxes, dtype=np.dtype("<f8"))
                key_memmap[key_cursor : key_cursor + expected] = keys
                key_cursor += expected

                inside = _dish_mask_inside_bbox_centers(bboxes, mask_spec)
                outside_indices = np.flatnonzero(~inside)
                if outside_indices.size:
                    circle_ratios: np.ndarray | None = None
                    if str(mask_spec.get("shape") or "") == "circle":
                        center = np.asarray(mask_spec["center_norm"], dtype=np.float64).reshape(2)
                        radius_x = float(mask_spec["radius_norm_x"])
                        radius_y = float(mask_spec["radius_norm_y"])
                        circle_ratios = np.sqrt(
                            ((bboxes[outside_indices, 0] - center[0]) / radius_x) ** 2
                            + ((bboxes[outside_indices, 1] - center[1]) / radius_y) ** 2
                        )
                        finite_ratios = circle_ratios[np.isfinite(circle_ratios)]
                        if finite_ratios.size:
                            outside_circle_ratio_min = min(
                                outside_circle_ratio_min,
                                float(np.min(finite_ratios)),
                            )
                            outside_circle_ratio_max = max(
                                outside_circle_ratio_max,
                                float(np.max(finite_ratios)),
                            )
                            outside_circle_ratio_sum += float(np.sum(finite_ratios))
                            outside_circle_ratio_count += int(finite_ratios.size)
                    outside_keys = keys[outside_indices]
                    _hash_array(outside_key_digest, outside_keys, dtype=np.dtype("<u8"))
                    count = int(outside_indices.size)
                    run_outside += count
                    outside_count += count
                    rows = pa.table(
                        {
                            "collection_id": pa.array([collection_id] * count, type=pa.string()),
                            "clip_id": pa.array([clip_id] * count, type=pa.string()),
                            "camera_serial": pa.array([camera_serial] * count, type=pa.string()),
                            "refined_group_path": pa.array([path] * count, type=pa.string()),
                            "member_row_index": pa.array(
                                (outside_indices + start).astype(np.int64, copy=False),
                                type=pa.int64(),
                            ),
                            "clip_local_frame_index": pa.array(
                                frames[outside_indices],
                                type=pa.int64(),
                            ),
                            "instance_key": pa.array(outside_keys, type=pa.uint64()),
                            "bbox_norm_cx": pa.array(bboxes[outside_indices, 0], type=pa.float64()),
                            "bbox_norm_cy": pa.array(bboxes[outside_indices, 1], type=pa.float64()),
                            "bbox_norm_w": pa.array(bboxes[outside_indices, 2], type=pa.float64()),
                            "bbox_norm_h": pa.array(bboxes[outside_indices, 3], type=pa.float64()),
                            "dish_circle_radial_ratio": pa.array(
                                circle_ratios
                                if circle_ratios is not None
                                else np.full(count, np.nan, dtype=np.float64),
                                type=pa.float64(),
                                from_pandas=True,
                            ),
                        },
                        schema=_AFFECTED_SCHEMA,
                    )
                    writer.write_table(rows)
            run_reports.append(
                {
                    "clip_id": clip_id,
                    "clip_index": selected_row.get("clip_index"),
                    "camera_serial": camera_serial,
                    "refined_group_path": path,
                    "selected_row_count": row_count,
                    "outside_dish_mask_count": run_outside,
                }
            )

        if key_cursor != total_rows:
            raise ValueError(f"Audited key count {key_cursor} does not match expected rows {total_rows}")
        key_memmap.flush()
        key_memmap.sort()
        duplicate_count = int(np.count_nonzero(key_memmap[1:] == key_memmap[:-1])) if total_rows > 1 else 0
        del key_memmap
        if duplicate_count:
            raise ValueError(
                f"Finalized collection contains {duplicate_count} adjacent duplicate instance_key values"
            )

        writer.close()
        os.replace(tmp_parquet, affected_path)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        tmp_parquet.unlink(missing_ok=True)
        raise
    finally:
        if key_memmap_path is not None:
            key_memmap_path.unlink(missing_ok=True)

    attrs = dict(collection.attrs)
    equivalence = "equivalent" if outside_count == 0 else "not_equivalent"
    return {
        "status": "ok",
        "schema_version": AUDIT_SCHEMA,
        "created_at_utc": _utc_now(),
        "analysis_zarr": str(archive_path),
        "collection_id": collection_id,
        "collection_path": resolved_path,
        "collection_schema_version": attrs.get("schema_version"),
        "audit_scope": AUDIT_SCOPE,
        "equivalence_status": equivalence,
        "interpretation": (
            "Post-hoc counterfactual validation of the selected refined instance rowset only; "
            "this does not assert that the original refinement enabled the dish-mask gate and "
            "does not attest source-detection decision/reason surfaces."
        ),
        "decision": (
            "retain_existing_downstream_outputs"
            if outside_count == 0
            else "rerun_mask_aware_refinement_and_reconcile_downstream_instance_keys"
        ),
        "dish_mask_gate": dict(mask_spec),
        "dish_mask_boundary_tolerance": dict(tolerance),
        "selected_run_count": len(inspected),
        "selected_row_count": total_rows,
        "outside_dish_mask_count": outside_count,
        "outside_geometry": (
            {
                "metric": "dish_circle_radial_ratio",
                "boundary": 1.0,
                "minimum": outside_circle_ratio_min,
                "maximum": outside_circle_ratio_max,
                "mean": outside_circle_ratio_sum / outside_circle_ratio_count,
                "finite_row_count": outside_circle_ratio_count,
            }
            if outside_count > 0
            and outside_circle_ratio_count > 0
            and np.isfinite(outside_circle_ratio_min)
            and str(mask_spec.get("shape") or "") == "circle"
            else None
        ),
        "instance_key_uniqueness": {
            "status": "unique",
            "checked_count": total_rows,
            "method": "disk_backed_uint64_sort",
        },
        "digests": {
            "selected_runs_canonical_sha256": _canonical_json_sha256(selected),
            "dish_mask_gate_canonical_sha256": _canonical_json_sha256(mask_spec),
            "ordered_instance_key_sha256": key_digest.hexdigest(),
            "ordered_bbox_norm_coords_sha256": bbox_digest.hexdigest(),
            "outside_instance_key_sha256": outside_key_digest.hexdigest(),
        },
        "affected_rows_parquet": {
            "path": str(affected_path),
            "row_count": outside_count,
            "sha256": _sha256_file(affected_path),
            "storage_role": "sparse_audit_evidence_only",
            "canonical_identity_authority": False,
            "canonical_instance_keys_remain_in": "selected refined-detection Zarr arrays",
        },
        "streaming": {
            "requested_chunk_rows": int(chunk_rows),
            "full_collection_materialized_in_ram": False,
            "collection_key_index": "temporary_disk_backed_memmap",
        },
        "runs": run_reports,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of a finalized clipped refined-detection collection against "
            "analysis_metadata.dish_mask."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Canonical analysis Zarr")
    parser.add_argument(
        "--collection-path",
        default=None,
        help="Explicit finalized collection group path; default follows latest_collection_path",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Immutable audit receipt")
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Rows outside the dish mask, keyed by instance_key (may be empty)",
    )
    parser.add_argument("--chunk-rows", type=int, default=131_072)
    parser.add_argument(
        "--dish-mask-boundary-tolerance-mm",
        type=float,
        default=DEFAULT_DISH_MASK_BOUNDARY_TOLERANCE_MM,
        help="Physical expansion around the fitted dish boundary (default: 0.5 mm)",
    )
    parser.add_argument(
        "--pixels-per-mm-camera",
        type=float,
        default=None,
        help=(
            "Explicit raw camera pixels/mm override. Required when the Zarr lacks "
            "analysis/calibration.attrs.pixels_per_mm_camera."
        ),
    )
    parser.add_argument(
        "--overwrite-evidence",
        action="store_true",
        help="Allow replacing existing report/parquet paths",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_json = args.output_json.expanduser().resolve()
    if output_json.exists() and not args.overwrite_evidence:
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output_json}")
    report = audit_clipped_dish_mask_equivalence(
        args.zarr_path,
        collection_path=args.collection_path,
        output_parquet=args.output_parquet,
        chunk_rows=args.chunk_rows,
        dish_mask_boundary_tolerance_mm=args.dish_mask_boundary_tolerance_mm,
        pixels_per_mm_camera=args.pixels_per_mm_camera,
        overwrite_evidence=args.overwrite_evidence,
    )
    write_json_atomic(output_json, report, overwrite=args.overwrite_evidence)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["equivalence_status"] == "equivalent" else 2


if __name__ == "__main__":
    raise SystemExit(main())
