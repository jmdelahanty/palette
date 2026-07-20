"""Dry-run or apply one keyed incremental crop materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.tracking.incremental_crop import (
    DEFAULT_SIGNATURE_BATCH_ROWS,
    DEFAULT_TABULAR_SHARD_ROWS,
    HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE,
    IncrementalCropError,
    build_incremental_crop_plan,
    capture_crop_source_snapshot,
    materialize_composite_incremental_crop_run,
    materialize_incremental_crop_run,
    resolve_historical_composite_coordinate_context,
    resolve_incremental_crop_coordinate_context,
)


REPORT_SCHEMA_ID = "palette.incremental_crop_cli_report"
REPORT_SCHEMA_VERSION = 2
PAYLOAD_STRATEGIES = {"standalone", "composite"}


def _normalize_group_path(value: str) -> str:
    path = str(value).strip().strip("/")
    if not path or any(part == ".." for part in path.split("/")):
        raise IncrementalCropError(f"Invalid Zarr group path: {value!r}.")
    return path


def _open_root(path: Path, *, apply: bool) -> Any:
    return zarr.open_group(
        str(path),
        mode="a" if apply else "r",
        use_consolidated=False,
    )


def plan_or_materialize_incremental_crop(
    zarr_path: Path,
    *,
    source_rowset_path: str,
    source_pixel_fingerprint: str,
    roi_size: tuple[int, int],
    output_run: str,
    base_crop_run: str | None,
    apply: bool,
    roi_chunk_rows: int,
    signature_batch_rows: int,
    tabular_shard_rows: int,
    command: str,
    payload_strategy: str = "standalone",
    promote_composite: bool = False,
    historical_composite: bool = False,
    coordinate_contract_mode: str | None = None,
) -> dict[str, Any]:
    archive_path = zarr_path.expanduser().resolve()
    source_path = _normalize_group_path(source_rowset_path)
    root = _open_root(archive_path, apply=apply)
    if source_path not in root:
        raise IncrementalCropError(f"Source rowset {source_path!r} does not exist.")
    if "raw_video/images_full" not in root:
        raise IncrementalCropError("Archive is missing raw_video/images_full.")
    frame_source = root["raw_video/images_full"]
    strategy = str(payload_strategy).strip().lower()
    if strategy not in PAYLOAD_STRATEGIES:
        raise IncrementalCropError(
            "payload_strategy must be one of: " + ", ".join(sorted(PAYLOAD_STRATEGIES))
        )
    if strategy == "composite" and not base_crop_run:
        raise IncrementalCropError("Composite crop output requires --base-crop-run.")
    if promote_composite:
        raise IncrementalCropError(
            "Historical composite crops cannot be promoted or update selectors."
        )
    if strategy == "composite" and not historical_composite:
        raise IncrementalCropError(
            "Composite crop output requires explicit --historical-composite."
        )
    if strategy != "composite" and historical_composite:
        raise IncrementalCropError(
            "--historical-composite requires --payload-strategy=composite."
        )
    expected_coordinate_mode = (
        HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE
        if strategy == "composite"
        else "canonical"
    )
    if coordinate_contract_mode not in {None, expected_coordinate_mode}:
        raise IncrementalCropError(
            f"{strategy} output requires coordinate mode "
            f"{expected_coordinate_mode!r}; got {coordinate_contract_mode!r}."
        )
    coordinate_contract_mode = expected_coordinate_mode
    resolver = (
        resolve_historical_composite_coordinate_context
        if strategy == "composite"
        else resolve_incremental_crop_coordinate_context
    )
    coordinate_context = resolver(
        root,
        source_group=root[source_path],
        source_path=source_path,
        frame_source=frame_source,
        source_pixel_fingerprint=source_pixel_fingerprint,
        coordinate_contract_mode=coordinate_contract_mode,
    )
    source_group = coordinate_context.source_group
    frame_source = coordinate_context.frame_source
    source_pixel_fingerprint = coordinate_context.source_pixel_fingerprint
    frame_shape = tuple(int(value) for value in frame_source.shape[1:])
    snapshot = capture_crop_source_snapshot(
        source_group,
        source_path=source_path,
        source_pixel_fingerprint=source_pixel_fingerprint,
        frame_shape=frame_shape,
        roi_size=roi_size,
        signature_batch_rows=signature_batch_rows,
    )
    base_group = None
    if base_crop_run is not None:
        if "crop_runs" not in root or base_crop_run not in root["crop_runs"]:
            raise IncrementalCropError(
                f"Reuse source crop run {base_crop_run!r} does not exist."
            )
        base_group = root[f"crop_runs/{base_crop_run}"]
    plan = build_incremental_crop_plan(
        snapshot,
        base_group=base_group,
        roi_size=roi_size,
    )
    report: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "would_materialize" if not apply else "running",
        "apply": bool(apply),
        "zarr_path": str(archive_path),
        "source_rowset_path": source_path,
        "source_rowset_fingerprint": snapshot.rowset_fingerprint.fingerprint,
        "source_signature_spec_digest": snapshot.signature_spec.spec_digest,
        "source_pixel_fingerprint": str(source_pixel_fingerprint),
        "base_crop_run": base_crop_run,
        "output_run": output_run,
        "payload_strategy": strategy,
        "promote_composite": bool(promote_composite),
        "historical_composite": bool(historical_composite),
        "coordinate_contract_mode": coordinate_contract_mode,
        "roi_size": list(roi_size),
        "roi_chunk_rows": int(roi_chunk_rows),
        "signature_batch_rows": int(signature_batch_rows),
        "tabular_shard_rows": int(tabular_shard_rows),
        "plan": plan.summary(),
        "estimated_roi_payload_bytes": int(
            (
                plan.summary()["action_counts"]["compute"]
                if strategy == "composite"
                else snapshot.row_count
            )
            * int(roi_size[0])
            * int(roi_size[1])
        ),
    }
    if not apply:
        return report

    ds_shape = None
    if "raw_video/images_ds" in root:
        raw_ds_shape = tuple(int(value) for value in root["raw_video/images_ds"].shape[1:])
        if len(raw_ds_shape) == 2:
            ds_shape = raw_ds_shape
    provenance = build_writer_run_provenance(
        command=command,
        params={
            "source_rowset_path": source_path,
            "source_pixel_fingerprint": source_pixel_fingerprint,
            "roi_size": list(roi_size),
            "base_crop_run": base_crop_run,
            "output_run": output_run,
            "roi_chunk_rows": int(roi_chunk_rows),
            "signature_batch_rows": int(signature_batch_rows),
            "tabular_shard_rows": int(tabular_shard_rows),
            "payload_strategy": strategy,
            "promote_composite": bool(promote_composite),
            "historical_composite": bool(historical_composite),
            "coordinate_contract_mode": coordinate_contract_mode,
        },
        input_run_ids={
            "source_rowset": source_path,
            "reuse_crop_run": base_crop_run,
        },
    )
    common_kwargs = {
        "source_group": source_group,
        "source_path": source_path,
        "frame_source": frame_source,
        "source_pixel_fingerprint": source_pixel_fingerprint,
        "roi_size": roi_size,
        "run_name": output_run,
        "run_provenance": provenance,
        "downsampled_frame_shape": ds_shape,
        "roi_chunk_rows": roi_chunk_rows,
        "signature_batch_rows": signature_batch_rows,
        "tabular_shard_rows": tabular_shard_rows,
        "coordinate_contract_mode": coordinate_contract_mode,
    }
    if strategy == "composite":
        assert base_crop_run is not None
        result = materialize_composite_incremental_crop_run(
            root,
            base_run_name=base_crop_run,
            promote=bool(promote_composite),
            **common_kwargs,
        )
    else:
        result = materialize_incremental_crop_run(
            root,
            base_run_name=base_crop_run,
            **common_kwargs,
        )
    report["status"] = "materialized"
    report["result"] = result.to_dict()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or write a complete crop run by keyed copy-forward plus localized "
            "crop computation. The default is read-only dry-run."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Recording analysis Zarr path")
    parser.add_argument(
        "--source-rowset-path",
        required=True,
        help="Exact modern detection/refined-detection row group path",
    )
    parser.add_argument(
        "--source-pixel-fingerprint",
        required=True,
        help=(
            "Exact acquisition materialization id for canonical output; historical "
            "fingerprints are accepted only with --historical-composite"
        ),
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        required=True,
    )
    parser.add_argument("--output-run", required=True, help="New crop_runs child name")
    parser.add_argument(
        "--base-crop-run",
        default=None,
        help="Explicit complete Phase-1 crop run eligible for keyed reuse",
    )
    parser.add_argument(
        "--payload-strategy",
        choices=sorted(PAYLOAD_STRATEGIES),
        default="standalone",
        help=(
            "standalone writes every target ROI; composite writes only computed rows "
            "and retains a depth-one immutable base (default: standalone)"
        ),
    )
    parser.add_argument(
        "--historical-composite",
        action="store_true",
        help=(
            "Explicitly write a complete but non-selector historical composite. "
            "It can never update latest or authoritative crop pointers."
        ),
    )
    parser.add_argument(
        "--roi-chunk-rows",
        type=int,
        default=32,
        help="Dense ROI logical chunk rows (default: 32)",
    )
    parser.add_argument(
        "--signature-batch-rows",
        type=int,
        default=DEFAULT_SIGNATURE_BATCH_ROWS,
    )
    parser.add_argument(
        "--tabular-shard-rows",
        type=int,
        default=DEFAULT_TABULAR_SHARD_ROWS,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write, validate, and publish. Without this flag no archive is modified.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = " ".join(sys.argv if argv is None else ["materialize_incremental_crop", *argv])
    try:
        report = plan_or_materialize_incremental_crop(
            args.zarr_path,
            source_rowset_path=args.source_rowset_path,
            source_pixel_fingerprint=args.source_pixel_fingerprint,
            roi_size=tuple(args.roi_size),
            output_run=args.output_run,
            base_crop_run=args.base_crop_run,
            apply=args.apply,
            roi_chunk_rows=args.roi_chunk_rows,
            signature_batch_rows=args.signature_batch_rows,
            tabular_shard_rows=args.tabular_shard_rows,
            command=command,
            payload_strategy=args.payload_strategy,
            promote_composite=False,
            historical_composite=args.historical_composite,
        )
    except Exception as exc:
        parser.error(str(exc))
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
