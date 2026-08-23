"""Publish one selector-ineligible immutable crop-geometry candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.analysis_workflows.materializers.registered_detection_gate import (
    validate_registered_detection_gate_consumption,
)
from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_snapshot_publication import (
    CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
    publish_crop_geometry_production_candidate,
)
from fisheye.shared.hybrid_crop_provider import (
    validate_hybrid_crop_signed_identity,
)


def _fixed_signed_provider_roi_size(
    archive: Path,
    *,
    run_id: str,
) -> tuple[int, int]:
    root = zarr.open_group(
        str(archive.expanduser().resolve()), mode="r", use_consolidated=False
    )
    provider = root[f"crop_runs/{run_id}"]
    validate_hybrid_crop_signed_identity(
        provider,
        expected_provider_record_sha256=str(
            provider.attrs.get("provider_record_sha256") or ""
        ),
    )
    sizes = np.asarray(provider["roi_sizes_full"][:])
    if (
        sizes.dtype != np.dtype(np.int32)
        or sizes.ndim != 2
        or sizes.shape[0] <= 0
        or sizes.shape[1] != 2
        or np.any(sizes <= 0)
        or not np.all(sizes == sizes[0])
    ):
        raise ValueError(
            "Signed hybrid origin provider must declare one fixed positive int32 "
            "ROI size."
        )
    return int(sizes[0, 0]), int(sizes[0, 1])


def _registered_gate_from_source(
    archive: Path,
    *,
    run_id: str,
) -> tuple[str, str | None]:
    root = zarr.open_group(
        str(archive.expanduser().resolve()), mode="r", use_consolidated=False
    )
    source = root[f"refined_detect_runs/{run_id}"]
    requirement = str(
        source.attrs.get("registered_detection_gate_requirement") or ""
    ).strip()
    if requirement not in {"off", "if_available", "required"}:
        raise ValueError(
            "Finalized refined source has an invalid registered-gate requirement."
        )
    evidence: Any = source.attrs.get("registered_detection_gate")
    if not isinstance(evidence, Mapping):
        raise ValueError("Finalized refined source lacks registered-gate evidence.")
    gate_run = str(evidence.get("gate_run") or "").strip() or None
    if requirement == "required" and (
        evidence.get("status") != "applied"
        or evidence.get("applied") is not True
        or gate_run is None
    ):
        raise ValueError(
            "Required finalized refined source lacks one applied registered gate."
        )
    return requirement, gate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument(
        "--roi-width",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    )
    parser.add_argument(
        "--roi-height",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    )
    parser.add_argument(
        "--padding-mode",
        choices=tuple(mode.value for mode in CropPaddingMode),
        default=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME.value,
    )
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--source-refined-run")
    parser.add_argument(
        "--registered-gate-requirement",
        choices=("off", "if_available", "required", "from_source"),
        default="off",
    )
    parser.add_argument("--registered-gate-run")
    parser.add_argument(
        "--geometry-origin-provider-run",
        help=(
            "Exact signed hybrid crop run whose verified per-row integer origins "
            "become the strict crop geometry."
        ),
    )
    parser.add_argument(
        "--roi-size-from-geometry-origin-provider",
        action="store_true",
        help=(
            "Derive the fixed width/height from the exact signed hybrid provider; "
            "the crop-v2 publisher revalidates every provider row."
        ),
    )
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        roi_width = int(args.roi_width)
        roi_height = int(args.roi_height)
        if args.roi_size_from_geometry_origin_provider:
            provider_run = str(args.geometry_origin_provider_run or "").strip()
            if not provider_run:
                raise ValueError(
                    "--roi-size-from-geometry-origin-provider requires "
                    "--geometry-origin-provider-run."
                )
            roi_width, roi_height = _fixed_signed_provider_roi_size(
                args.analysis_zarr,
                run_id=provider_run,
            )
        gate_requirement = str(args.registered_gate_requirement)
        gate_run = args.registered_gate_run
        if gate_requirement == "from_source":
            source_run = str(args.source_refined_run or "").strip()
            if not source_run:
                raise ValueError(
                    "--registered-gate-requirement from_source requires "
                    "--source-refined-run."
                )
            gate_requirement, gate_run = _registered_gate_from_source(
                args.analysis_zarr,
                run_id=source_run,
            )
        policy = CropGeometryPolicy(
            purpose=args.purpose,
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(roi_width, roi_height),
            padding_mode=CropPaddingMode(args.padding_mode),
        )
        result = publish_crop_geometry_production_candidate(
            analysis_zarr=args.analysis_zarr,
            run_id=args.run_id,
            policy=policy,
            expected_camera_identity=args.camera_id,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
            keep_scratch=bool(args.keep_scratch),
            source_refined_run_id=args.source_refined_run,
            registered_gate_requirement=gate_requirement,
            registered_gate_run=gate_run,
            registered_gate_validator=validate_registered_detection_gate_consumption,
            geometry_origin_provider_run_id=args.geometry_origin_provider_run,
        )
    except Exception as exc:
        result = {
            "schema_id": CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
