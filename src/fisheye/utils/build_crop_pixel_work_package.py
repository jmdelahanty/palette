"""Plan or build one keyed subset ROI pixel work package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    build_crop_pixel_work_package_from_source,
)
from fisheye.shared.flat_roi_cache import open_flat_roi_cache
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative_path(value: object, *, base: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _column(table: Any, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"Clipped ROI-cache row index lacks {name!r}.")
    return np.asarray(table[name].combine_chunks().to_numpy(), dtype=dtype)


class _GeometryBoundFlatCacheSource:
    """Authenticated clip pixels addressed by canonical crop-v2 row ids."""

    def __init__(
        self,
        *,
        root: Any,
        crop_group: Any,
        crop_run_name: str,
        cache: Any,
        cache_manifest_path: Path,
        target_rows: np.ndarray,
        cache_rows: np.ndarray,
        frame_indices: np.ndarray,
        roi_coordinates_full: np.ndarray,
        roi_pixel_contract: Mapping[str, Any],
        source_identity: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.crop_group = crop_group
        self.crop_run_name = crop_run_name
        self.storage_mode = "geometry_only"
        self.roi_shape = tuple(int(value) for value in cache.shape[1:])
        self.roi_pixel_contract = dict(roi_pixel_contract)
        self.roi_read_mode = "canonical_geometry_bound_flat_roi_cache"
        self.roi_cache_used = True
        self.frame_source_kind = "authenticated_clipped_flat_roi_cache"
        self.pixel_materialization_id = None
        self.total_rois = int(crop_group["instance_key"].shape[0])
        self.frame_indices = frame_indices
        self.roi_coordinates_full = roi_coordinates_full
        self.bound_crop_rows = target_rows
        self._cache = cache
        self._cache_manifest_path = cache_manifest_path
        self._target_rows = target_rows
        self._cache_rows = cache_rows
        self._source_identity = dict(source_identity)

    def read_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        rows = np.asarray(indices, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(self._target_rows, rows)
        if (
            np.any(positions >= self._target_rows.size)
            or not np.array_equal(self._target_rows[positions], rows)
        ):
            raise ValueError(
                "Requested crop rows are outside the authenticated clip cache binding."
            )
        return np.asarray(self._cache[self._cache_rows[positions]], dtype=np.uint8)

    def _build_frame_source_identity(self) -> dict[str, Any]:
        return dict(self._source_identity)

    def close(self) -> None:
        self._cache.close()


def _open_geometry_bound_flat_cache_source(
    *,
    root: Any,
    zarr_path: Path,
    crop_run: str,
    cache_manifest_path: Path,
    clip_id: str,
    clip_index: int,
    frame_start: int,
    frame_stop: int,
) -> _GeometryBoundFlatCacheSource:
    """Bind one authenticated clip cache directly to strict crop-v2 rows."""

    if not clip_id or "/" in clip_id:
        raise ValueError("source clip id must be one nonempty path-safe component.")
    if type(clip_index) is not int or clip_index < 0:
        raise ValueError("source clip index must be a nonnegative exact integer.")
    if type(frame_start) is not int or type(frame_stop) is not int:
        raise ValueError("source frame bounds must be exact integers.")
    if frame_start < 0 or frame_stop <= frame_start:
        raise ValueError("source frame bounds must define one nonempty interval.")

    open_persisted_crop_geometry_publication(zarr_path, run_id=crop_run)
    crop_group = root[f"crop_runs/{crop_run}"]
    required_crop_arrays = (
        "instance_key",
        "frame_indices",
        "roi_coordinates_full",
    )
    missing_crop = [name for name in required_crop_arrays if name not in crop_group]
    if missing_crop:
        raise ValueError(
            "Strict crop-v2 lacks clipped row binding arrays: "
            + ", ".join(missing_crop)
        )

    manifest_path = cache_manifest_path.expanduser().resolve()
    cache = open_flat_roi_cache(
        manifest_path,
        expected_archive_path=zarr_path,
        require_payload_sha256=True,
    )
    try:
        manifest = cache.manifest
        row_declaration = manifest.get("row_index")
        if not isinstance(row_declaration, Mapping):
            raise ValueError("Clipped ROI-cache manifest lacks row_index metadata.")
        row_path = _relative_path(
            row_declaration.get("path"), base=manifest_path.parent
        )
        table = pq.read_table(row_path)
        if int(table.num_rows) != int(cache.shape[0]):
            raise ValueError(
                "Clipped ROI-cache row index and pixel payload row counts differ."
            )
        cache_rows = np.arange(int(table.num_rows), dtype=np.int64)
        if "roi_row_index" in table.column_names:
            declared_cache_rows = _column(table, "roi_row_index", np.dtype(np.int64))
            if not np.array_equal(declared_cache_rows, cache_rows):
                raise ValueError(
                    "Clipped ROI-cache rows are not in exact payload-row order."
                )
        clip_ids = [str(value) for value in table["clip_id"].to_pylist()]
        if set(clip_ids) != {clip_id}:
            raise ValueError("Clipped ROI-cache rows bind a different source clip id.")
        observed_clip_indices = _column(
            table, "clip_index", np.dtype(np.int64)
        )
        if not np.all(observed_clip_indices == clip_index):
            raise ValueError(
                "Clipped ROI-cache rows bind a different source clip index."
            )
        cache_keys = _column(table, "instance_key", np.dtype(np.uint64))
        if np.unique(cache_keys).size != cache_keys.size:
            raise ValueError("Clipped ROI-cache instance keys are not unique.")

        crop_keys = np.asarray(crop_group["instance_key"][:], dtype=np.uint64)
        if np.unique(crop_keys).size != crop_keys.size:
            raise ValueError("Strict crop-v2 instance keys are not unique.")
        order = np.argsort(crop_keys, kind="stable")
        sorted_keys = crop_keys[order]
        positions = np.searchsorted(sorted_keys, cache_keys)
        if (
            np.any(positions >= sorted_keys.size)
            or not np.array_equal(sorted_keys[positions], cache_keys)
        ):
            raise ValueError(
                "Clipped ROI-cache contains rows absent from strict crop-v2 geometry."
            )
        target_rows_unsorted = np.asarray(order[positions], dtype=np.int64)
        canonical_order = np.argsort(target_rows_unsorted, kind="stable")
        target_rows = target_rows_unsorted[canonical_order]
        cache_rows = cache_rows[canonical_order]

        crop_frames = np.asarray(crop_group["frame_indices"][:], dtype=np.int64)
        crop_coordinates = np.asarray(
            crop_group["roi_coordinates_full"][:], dtype=np.int32
        )
        cache_frames = _column(table, "parent_frame_index", np.dtype(np.int64))[
            canonical_order
        ]
        cache_local_frames = _column(
            table, "clip_local_frame_index", np.dtype(np.int64)
        )[canonical_order]
        cache_coordinates = np.column_stack(
            (
                _column(table, "roi_x", np.dtype(np.int32)),
                _column(table, "roi_y", np.dtype(np.int32)),
            )
        )[canonical_order]
        if not np.array_equal(crop_frames[target_rows], cache_frames):
            raise ValueError("Clipped ROI-cache frame identities differ from crop-v2.")
        if not np.array_equal(crop_coordinates[target_rows], cache_coordinates):
            raise ValueError("Clipped ROI-cache ROI origins differ from crop-v2.")
        if not np.array_equal(cache_local_frames, cache_frames - frame_start):
            raise ValueError(
                "Clipped ROI-cache local-frame identities differ from the planned "
                "recording-frame interval."
            )
        if np.any(cache_frames < frame_start) or np.any(cache_frames >= frame_stop):
            raise ValueError(
                "Clipped ROI-cache frames fall outside the planned recording interval."
            )
        roi_shape = crop_group.attrs.get("roi_shape") or crop_group.attrs.get(
            "roi_size"
        )
        if (
            not isinstance(roi_shape, (list, tuple))
            or len(roi_shape) != 2
            or tuple(int(value) for value in roi_shape) != tuple(cache.shape[1:])
        ):
            raise ValueError("Clipped ROI-cache shape differs from strict crop-v2.")
        builder = manifest.get("builder")
        pixel_contract = (
            builder.get("pixel_contract") if isinstance(builder, Mapping) else None
        )
        if not isinstance(pixel_contract, Mapping):
            raise ValueError(
                "Authenticated clipped ROI-cache lacks its pixel contract."
            )
        source_identity = {
            "schema_id": "palette.canonical_crop_flat_cache_binding",
            "schema_version": 1,
            "crop_run": crop_run,
            "clip_id": clip_id,
            "clip_index": clip_index,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "cache_manifest_path": str(manifest_path),
            "cache_manifest_sha256": _sha256_file(manifest_path),
            "cache_payload_sha256": str(manifest["array"]["sha256"]),
            "cache_row_index_path": str(row_path),
            "cache_row_index_sha256": _sha256_file(row_path),
            "row_count": int(target_rows.size),
        }
        return _GeometryBoundFlatCacheSource(
            root=root,
            crop_group=crop_group,
            crop_run_name=crop_run,
            cache=cache,
            cache_manifest_path=manifest_path,
            target_rows=target_rows,
            cache_rows=cache_rows,
            frame_indices=crop_frames,
            roi_coordinates_full=crop_coordinates,
            roi_pixel_contract=pixel_contract,
            source_identity=source_identity,
        )
    except BaseException:
        cache.close()
        raise


def _load_rows(
    args: argparse.Namespace,
    *,
    total_rows: int | None = None,
    bound_rows: np.ndarray | None = None,
) -> np.ndarray:
    explicit_selection = bool(
        args.all_crop_rows
        or args.crop_row
        or args.crop_rows_json is not None
        or args.crop_rows_npy is not None
    )
    if bound_rows is not None:
        if explicit_selection:
            raise ValueError(
                "Geometry-bound clipped caches define their exact crop rows; "
                "explicit crop-row selection is forbidden."
            )
        return np.asarray(bound_rows, dtype=np.int64).reshape(-1)
    if bool(args.all_crop_rows):
        if total_rows is None or total_rows < 0:
            raise ValueError("--all-crop-rows requires a resolved crop row count.")
        return np.arange(int(total_rows), dtype=np.int64)
    values = [int(value) for value in (args.crop_row or [])]
    if args.crop_rows_json is not None:
        payload = json.loads(args.crop_rows_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--crop-rows-json must contain one JSON integer list.")
        values.extend(int(value) for value in payload)
    if args.crop_rows_npy is not None:
        values.extend(
            np.asarray(
                np.load(args.crop_rows_npy, allow_pickle=False), dtype=np.int64
            )
            .reshape(-1)
            .tolist()
        )
    rows = np.asarray(values, dtype=np.int64)
    if rows.size == 0:
        raise ValueError(
            "Provide at least one --crop-row, --crop-rows-json, or --crop-rows-npy."
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or persist only selected logical crop rows for shared keypoint/"
            "subject-mask delta inference. Dry-run is the default."
        )
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-row", type=int, action="append")
    parser.add_argument(
        "--all-crop-rows",
        action="store_true",
        help="Select the complete crop row domain in canonical ascending order.",
    )
    parser.add_argument(
        "--crop-rows-json",
        type=Path,
        help="JSON file containing source crop row integers.",
    )
    parser.add_argument(
        "--crop-rows-npy",
        type=Path,
        help="NumPy file containing source crop row integers.",
    )
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument(
        "--roi-cache-manifest",
        type=Path,
        help=(
            "Optional authenticated flat ROI-cache manifest used as the pixel "
            "provider while the package remains bound to --crop-run geometry."
        ),
    )
    parser.add_argument(
        "--roi-cache-expected-archive-path",
        type=Path,
        help=(
            "Optional canonical archive identity expected by a staged/cache "
            "manifest. Defaults to zarr_path when a cache is supplied."
        ),
    )
    parser.add_argument(
        "--bind-clipped-cache-to-crop-geometry",
        action="store_true",
        help=(
            "Bind an authenticated clipped collection cache directly to the "
            "matching rows of strict --crop-run geometry."
        ),
    )
    parser.add_argument("--source-clip-id")
    parser.add_argument("--source-clip-index", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write and validate the package. Without this flag, only print the plan.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    zarr_path = args.zarr_path.expanduser().resolve()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    if args.bind_clipped_cache_to_crop_geometry:
        if args.roi_cache_manifest is None:
            raise ValueError(
                "--bind-clipped-cache-to-crop-geometry requires "
                "--roi-cache-manifest."
            )
        required_binding = {
            "--source-clip-id": args.source_clip_id,
            "--source-clip-index": args.source_clip_index,
            "--frame-start": args.frame_start,
            "--frame-stop": args.frame_stop,
        }
        missing = [name for name, value in required_binding.items() if value is None]
        if missing:
            raise ValueError(
                "Clipped cache geometry binding requires: " + ", ".join(missing)
            )
        source = _open_geometry_bound_flat_cache_source(
            root=root,
            zarr_path=zarr_path,
            crop_run=args.crop_run,
            cache_manifest_path=args.roi_cache_manifest,
            clip_id=str(args.source_clip_id),
            clip_index=int(args.source_clip_index),
            frame_start=int(args.frame_start),
            frame_stop=int(args.frame_stop),
        )
    else:
        source = CropImageSource.open(
            root,
            crop_run=args.crop_run,
            zarr_path=zarr_path,
            roi_cache_policy="never",
            roi_cache_manifest=args.roi_cache_manifest,
            roi_cache_expected_archive_path=(
                args.roi_cache_expected_archive_path or zarr_path
                if args.roi_cache_manifest is not None
                else None
            ),
        )
    try:
        rows = _load_rows(
            args,
            total_rows=int(source.total_rois),
            bound_rows=getattr(source, "bound_crop_rows", None),
        )
        unique = np.unique(rows)
        in_bounds = bool(
            rows.size
            and int(rows.min()) >= 0
            and int(rows.max()) < int(source.total_rois)
        )
        canonical_order = bool(np.array_equal(rows, np.sort(rows, kind="stable")))
        plan = {
            "action": "apply" if args.apply else "dry_run",
            "zarr_path": str(zarr_path),
            "crop_run": str(source.crop_run_name),
            "manifest_path": str(args.manifest.expanduser().resolve()),
            "source_crop_rows": int(source.total_rois),
            "selected_rows": int(rows.shape[0]),
            "selected_rows_unique": int(unique.shape[0]),
            "selection_in_bounds": in_bounds,
            "selection_canonical_ascending": canonical_order,
            "roi_shape": [int(value) for value in source.roi_shape],
            "source_roi_read_mode": str(source.roi_read_mode),
            "source_roi_cache_used": bool(source.roi_cache_used),
            "estimated_pixel_payload_bytes": int(
                rows.shape[0] * source.roi_shape[0] * source.roi_shape[1]
            ),
            "downstream_contract": {
                "keypoints_output_parent": "keypoint_shard_runs",
                "subject_masks_output_parent": "subject_mask_shard_runs",
                "canonical_publication": "finalizer_only",
            },
        }
        if not in_bounds or int(unique.shape[0]) != int(rows.shape[0]) or not canonical_order:
            raise ValueError(
                "Selected crop rows must be unique, ascending, and within the crop run."
            )
        if args.apply:
            plan["package"] = build_crop_pixel_work_package_from_source(
                source,
                target_crop_rows=rows,
                manifest_path=args.manifest,
                archive_path=zarr_path,
                batch_rows=int(args.batch_rows),
                overwrite=bool(args.overwrite),
            )
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    finally:
        source.close()


if __name__ == "__main__":
    main()
