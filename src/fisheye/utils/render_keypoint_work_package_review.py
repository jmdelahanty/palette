"""Publish bounded visual-review evidence for one keypoint work package.

This utility is deliberately diagnostic.  It reads one exact crop-pixel work
package and one exact keypoint run, verifies their ordered row identities, and
publishes an immutable file bundle outside the analysis Zarr.  It never changes
selectors or scientific arrays.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zarr

from fisheye.shared.crop_pixel_work_package import (
    CropPixelWorkPackage,
    open_crop_pixel_work_package,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.keypoint_work_package_visual_review"
SCHEMA_VERSION = 1
KEYPOINT_EVIDENCE_ARRAYS = (
    "confidence",
    "detection_success",
    "frame_indices",
    "instance_key",
    "keypoint_confidences",
    "keypoints_roi",
    "pose_bbox_xyxy_roi",
    "pose_failure_codes",
    "source_crop_row_ids",
    "source_row_signature",
)
KEYPOINT_COLORS_RGB = (
    (255, 0, 255),
    (0, 170, 255),
    (255, 140, 0),
    (0, 255, 80),
    (255, 230, 0),
)
PROVIDER_BORDER_COLORS_RGB = {
    "acquisition_crop_video": (0, 220, 80),
    "offline_full_frame_supplemental_flat_cache": (255, 150, 0),
}
FAILURE_BORDER_COLOR_RGB = (255, 40, 40)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path) -> dict[str, Any]:
    def _reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _normalize_group_path(value: str) -> str:
    path = "/".join(part for part in str(value).strip("/").split("/") if part)
    if not path:
        raise ValueError("Keypoint run path must not be empty.")
    return path


def _provider_name_map(pixel_contract: Mapping[str, Any]) -> dict[int, str]:
    raw = pixel_contract.get("source_pixel_kind_map")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("Work-package pixel contract lacks source_pixel_kind_map.")
    result: dict[int, str] = {}
    for name, code in raw.items():
        if not isinstance(name, str) or type(code) is not int:
            raise ValueError("source_pixel_kind_map must map names to integer codes.")
        if int(code) in result:
            raise ValueError("source_pixel_kind_map contains duplicate codes.")
        result[int(code)] = name
    return result


def select_review_rows(
    *,
    detection_success: np.ndarray,
    confidence: np.ndarray,
    provider_codes: np.ndarray,
    low_confidence_per_provider: int,
    spread_per_provider: int,
) -> dict[str, list[int]]:
    """Select deterministic, nonduplicated focus rows by review purpose."""

    success = np.asarray(detection_success, dtype=bool).reshape(-1)
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    providers = np.asarray(provider_codes, dtype=np.int16).reshape(-1)
    if not (success.shape == scores.shape == providers.shape):
        raise ValueError("Review-selection arrays must have identical lengths.")
    if int(low_confidence_per_provider) < 0 or int(spread_per_provider) < 0:
        raise ValueError("Review sample counts must be nonnegative.")

    result: dict[str, list[int]] = {
        "failures": np.flatnonzero(~success).astype(int).tolist()
    }
    used = set(result["failures"])
    for provider_code in sorted(int(value) for value in np.unique(providers)):
        eligible = np.flatnonzero(success & (providers == provider_code))
        finite_rank = np.where(np.isfinite(scores[eligible]), scores[eligible], np.inf)
        order = np.argsort(finite_rank, kind="stable")
        low = eligible[order[: int(low_confidence_per_provider)]].astype(int).tolist()
        result[f"provider_{provider_code}_low_confidence"] = low
        used.update(low)

        remaining = np.asarray(
            [int(row) for row in eligible.tolist() if int(row) not in used],
            dtype=np.int64,
        )
        count = min(int(spread_per_provider), int(remaining.shape[0]))
        if count:
            positions = np.rint(
                np.linspace(0, int(remaining.shape[0]) - 1, count)
            ).astype(np.int64)
            spread = remaining[positions].astype(int).tolist()
        else:
            spread = []
        result[f"provider_{provider_code}_spread"] = spread
        used.update(spread)
    return result


def _finite_xy(value: np.ndarray) -> tuple[int, int] | None:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape[0] < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return int(round(float(point[0]))), int(round(float(point[1])))


def _annotated_panel(
    *,
    pixel: np.ndarray,
    output_row: int,
    crop_row: int,
    frame_index: int,
    provider_name: str,
    success: bool,
    confidence: float,
    failure_code: int,
    keypoints_roi: np.ndarray,
    pose_bbox_xyxy_roi: np.ndarray,
) -> Image.Image:
    gray = np.asarray(pixel, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError("Review pixels must be two-dimensional grayscale arrays.")
    image = Image.fromarray(gray, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    bbox = np.asarray(pose_bbox_xyxy_roi, dtype=np.float64).reshape(-1)
    if bbox.shape[0] >= 4 and np.all(np.isfinite(bbox[:4])):
        draw.rectangle(tuple(float(value) for value in bbox[:4]), outline=(80, 255, 80), width=2)

    points = np.asarray(keypoints_roi, dtype=np.float64)
    edges = ((0, 1), (0, 2), (1, 2), (3, 1), (3, 2), (0, 4))
    if success and points.ndim == 2:
        for source, target in edges:
            if source >= points.shape[0] or target >= points.shape[0]:
                continue
            p1 = _finite_xy(points[source])
            p2 = _finite_xy(points[target])
            if p1 is not None and p2 is not None:
                draw.line((p1, p2), fill=(245, 245, 245), width=2)
        for index, point in enumerate(points):
            xy = _finite_xy(point)
            if xy is None:
                continue
            color = KEYPOINT_COLORS_RGB[index % len(KEYPOINT_COLORS_RGB)]
            radius = 5
            draw.ellipse(
                (xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius),
                fill=color,
                outline=(0, 0, 0),
                width=1,
            )

    header_height = 52
    panel = Image.new("RGB", (image.width, image.height + header_height), (0, 0, 0))
    panel.paste(image, (0, header_height))
    panel_draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    status = f"OK conf={confidence:.3f}" if success else f"FAIL code={failure_code}"
    panel_draw.text(
        (6, 5),
        f"out={output_row} crop={crop_row} frame={frame_index}",
        fill=(255, 255, 255),
        font=font,
    )
    panel_draw.text(
        (6, 25),
        f"{provider_name} | {status}",
        fill=(220, 220, 220),
        font=font,
    )
    border = (
        FAILURE_BORDER_COLOR_RGB
        if not success
        else PROVIDER_BORDER_COLORS_RGB.get(provider_name, (180, 180, 180))
    )
    panel_draw.rectangle((1, 1, panel.width - 2, panel.height - 2), outline=border, width=4)
    return panel


def render_review_montage(
    panels: Sequence[Image.Image],
    *,
    columns: int,
    panel_width: int,
) -> Image.Image:
    if not panels:
        placeholder = Image.new("RGB", (max(320, int(panel_width)), 120), (0, 0, 0))
        ImageDraw.Draw(placeholder).text(
            (10, 45), "No rows in this review category", fill=(255, 255, 255)
        )
        return placeholder
    columns = max(1, int(columns))
    panel_width = max(64, int(panel_width))
    resized: list[Image.Image] = []
    for panel in panels:
        height = max(1, int(round(panel.height * panel_width / panel.width)))
        resized.append(panel.resize((panel_width, height), Image.Resampling.LANCZOS))
    panel_height = max(panel.height for panel in resized)
    rows = int(math.ceil(len(resized) / columns))
    montage = Image.new(
        "RGB", (columns * panel_width, rows * panel_height), (20, 20, 20)
    )
    for index, panel in enumerate(resized):
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        montage.paste(panel, (x, y))
    return montage


def _array_record(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "canonical_c_order_sha256": hashlib.sha256(
            array.tobytes(order="C")
        ).hexdigest(),
    }


def _copy_work_package(package: CropPixelWorkPackage, target: Path) -> dict[str, Any]:
    target.mkdir(parents=True)
    manifest = dict(package.manifest)
    array_meta = manifest.get("array")
    rows_meta = manifest.get("rows")
    if not isinstance(array_meta, Mapping) or not isinstance(rows_meta, Mapping):
        raise ValueError("Validated work package lacks array or rows metadata.")
    source_manifest = package.manifest_path
    def resolve_artifact(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (source_manifest.parent / path).resolve()
        )

    source_payload = resolve_artifact(array_meta["bin_path"])
    source_rows = resolve_artifact(rows_meta["path"])
    copied = {
        "package.json": source_manifest,
        source_payload.name: source_payload,
        source_rows.name: source_rows,
    }
    for name, source in copied.items():
        shutil.copy2(source, target / name)

    reopened = open_crop_pixel_work_package(
        target / "package.json", verify_payload=True, verify_pixel_rows=True
    )
    try:
        if reopened.package_id != package.package_id:
            raise ValueError("Copied work package changed logical package identity.")
    finally:
        reopened.close()
    return {
        name: {"sha256": _sha256_file(target / name), "bytes": (target / name).stat().st_size}
        for name in sorted(copied)
    }


def build_review_bundle(
    *,
    analysis_zarr: Path,
    work_package_manifest: Path,
    keypoint_run_path: str,
    output_dir: Path,
    metadata_mode: str,
    low_confidence_per_provider: int,
    spread_per_provider: int,
    source_result_json: Path | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Review output already exists: {output_dir}")
    if metadata_mode not in {"consolidated", "unconsolidated"}:
        raise ValueError("metadata_mode must be consolidated or unconsolidated.")

    root = zarr.open_group(
        str(analysis_zarr),
        mode="r",
        use_consolidated=(metadata_mode == "consolidated"),
    )
    run_path = _normalize_group_path(keypoint_run_path)
    keypoint = root[run_path]
    if keypoint.attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(f"Keypoint run is not complete: {run_path}")
    source_crop_run = str(keypoint.attrs.get("source_crop_run") or "").strip()
    if not source_crop_run:
        raise ValueError("Keypoint run lacks exact source_crop_run provenance.")

    package = open_crop_pixel_work_package(
        work_package_manifest,
        expected_archive_path=analysis_zarr,
        expected_crop_run=source_crop_run,
        root=root,
        verify_payload=True,
        verify_pixel_rows=True,
    )
    temporary = output_dir.with_name(
        f".{output_dir.name}.{os.getpid()}_{uuid.uuid4().hex}.moving"
    )
    try:
        expected_package_id = str(
            keypoint.attrs.get("source_crop_pixel_work_package_id") or ""
        )
        if expected_package_id != package.package_id:
            raise ValueError("Keypoint run is bound to a different work package.")
        total_rows = package.row_count
        arrays: dict[str, np.ndarray] = {}
        for name in KEYPOINT_EVIDENCE_ARRAYS:
            if name not in keypoint:
                raise ValueError(f"Keypoint run is missing evidence array {name!r}.")
            values = np.asarray(keypoint[name][:])
            if values.shape[0] != total_rows:
                raise ValueError(
                    f"Keypoint array {name!r} length differs from work package."
                )
            arrays[name] = values

        if not np.array_equal(arrays["source_crop_row_ids"], package.crop_row_indices):
            raise ValueError("Keypoint source_crop_row_ids differ from the work package.")
        if not np.array_equal(arrays["instance_key"], package.instance_keys):
            raise ValueError("Keypoint instance_key order differs from the work package.")
        if not np.array_equal(
            arrays["source_row_signature"], package.source_row_signatures
        ):
            raise ValueError("Keypoint row signatures differ from the work package.")
        if not np.array_equal(arrays["frame_indices"], package.frame_indices):
            raise ValueError("Keypoint frame_indices differ from the work package.")

        crop_group = root[f"crop_runs/{source_crop_run}"]
        if "source_pixel_kind_codes" not in crop_group:
            raise ValueError("Hybrid crop run lacks source_pixel_kind_codes.")
        provider_codes = np.asarray(
            crop_group["source_pixel_kind_codes"][package.crop_row_indices],
            dtype=np.int16,
        )
        provider_names = _provider_name_map(package.pixel_contract)
        unknown_codes = sorted(set(int(value) for value in provider_codes) - set(provider_names))
        if unknown_codes:
            raise ValueError(f"Unknown source-pixel kind codes: {unknown_codes}")

        success = np.asarray(arrays["detection_success"], dtype=bool)
        confidence = np.asarray(arrays["confidence"], dtype=np.float64)
        selection = select_review_rows(
            detection_success=success,
            confidence=confidence,
            provider_codes=provider_codes,
            low_confidence_per_provider=low_confidence_per_provider,
            spread_per_provider=spread_per_provider,
        )
        pixels = np.asarray(package.pixels[:], dtype=np.uint8)
        labels_raw = keypoint.attrs.get("keypoint_labels")
        if not isinstance(labels_raw, list) or not labels_raw:
            raise ValueError("Keypoint run lacks keypoint_labels.")
        labels = [str(value) for value in labels_raw]

        def panels_for(rows: Sequence[int]) -> list[Image.Image]:
            return [
                _annotated_panel(
                    pixel=pixels[row],
                    output_row=int(row),
                    crop_row=int(package.crop_row_indices[row]),
                    frame_index=int(package.frame_indices[row]),
                    provider_name=provider_names[int(provider_codes[row])],
                    success=bool(success[row]),
                    confidence=float(confidence[row]),
                    failure_code=int(arrays["pose_failure_codes"][row]),
                    keypoints_roi=arrays["keypoints_roi"][row],
                    pose_bbox_xyxy_roi=arrays["pose_bbox_xyxy_roi"][row],
                )
                for row in rows
            ]

        temporary.mkdir(parents=True)
        overview_path = temporary / "all_rows_overview.png"
        attention_path = temporary / "failures_and_low_confidence.png"
        spread_path = temporary / "provider_spread.png"
        render_review_montage(
            panels_for(range(total_rows)), columns=16, panel_width=192
        ).save(overview_path, format="PNG", optimize=False)

        attention_rows = list(selection["failures"])
        spread_rows: list[int] = []
        for code in sorted(provider_names):
            attention_rows.extend(selection.get(f"provider_{code}_low_confidence", []))
            spread_rows.extend(selection.get(f"provider_{code}_spread", []))
        render_review_montage(
            panels_for(attention_rows), columns=4, panel_width=384
        ).save(attention_path, format="PNG", optimize=False)
        render_review_montage(
            panels_for(spread_rows), columns=4, panel_width=384
        ).save(spread_path, format="PNG", optimize=False)

        evidence_arrays = {
            **arrays,
            "provider_codes": provider_codes,
            "package_pixel_sha256": np.asarray(package.pixel_sha256, dtype=np.uint8),
            "package_roi_coordinates_full": np.asarray(
                package.roi_coordinates_full, dtype=np.int32
            ),
        }
        evidence_path = temporary / "keypoint_row_evidence.npz"
        np.savez_compressed(evidence_path, **evidence_arrays)
        attrs_path = temporary / "keypoint_run_attrs.json"
        _write_json(attrs_path, dict(keypoint.attrs))
        package_files = _copy_work_package(package, temporary / "pixel_package")

        source_result_record: dict[str, Any] | None = None
        if source_result_json is not None:
            source_result = _strict_json_object(source_result_json)
            if source_result.get("status") != "succeeded":
                raise ValueError("Source canary receipt is not succeeded.")
            source_work_package = source_result.get("work_package")
            inference = source_result.get("inference")
            if (
                not isinstance(source_work_package, Mapping)
                or source_work_package.get("package_id") != package.package_id
                or not isinstance(inference, Mapping)
                or inference.get("run_path") != run_path
            ):
                raise ValueError("Source canary receipt does not bind these exact inputs.")
            copied_result = temporary / "source_canary_result.json"
            shutil.copy2(source_result_json, copied_result)
            source_result_record = {
                "path": copied_result.name,
                "sha256": _sha256_file(copied_result),
                "bytes": copied_result.stat().st_size,
            }

        artifacts: dict[str, Any] = {}
        for artifact_id, path in (
            ("all_rows_overview", overview_path),
            ("failures_and_low_confidence", attention_path),
            ("provider_spread", spread_path),
            ("keypoint_row_evidence", evidence_path),
            ("keypoint_run_attrs", attrs_path),
        ):
            artifacts[artifact_id] = {
                "path": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        if source_result_record is not None:
            artifacts["source_canary_result"] = source_result_record

        provider_summary: dict[str, Any] = {}
        for code, name in sorted(provider_names.items()):
            mask = provider_codes == int(code)
            provider_summary[name] = {
                "code": int(code),
                "rows": int(np.count_nonzero(mask)),
                "successful_rows": int(np.count_nonzero(mask & success)),
                "failed_rows": int(np.count_nonzero(mask & ~success)),
            }

        body: dict[str, Any] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "pending_visual_review",
            "created_at_utc": _utc_now(),
            "exporter": {
                "module": "fisheye.utils.render_keypoint_work_package_review",
                "git": get_git_info(),
            },
            "source": {
                "analysis_zarr": str(analysis_zarr),
                "metadata_mode": metadata_mode,
                "keypoint_run_path": run_path,
                "source_crop_run": source_crop_run,
                "work_package_manifest": str(work_package_manifest),
                "work_package_id": package.package_id,
                "provider_record_sha256": package.manifest["source"][
                    "crop_signature"
                ]["provider_record_sha256"],
                "stage_selector_eligible": bool(
                    keypoint.attrs.get("stage_selector_eligible")
                ),
            },
            "row_alignment": {
                "row_count": total_rows,
                "source_crop_row_ids_exact": True,
                "instance_keys_exact": True,
                "source_row_signatures_exact": True,
                "frame_indices_exact": True,
            },
            "outcomes": {
                "successful_rows": int(np.count_nonzero(success)),
                "failed_rows": int(np.count_nonzero(~success)),
                "providers": provider_summary,
            },
            "keypoint_labels": labels,
            "keypoint_colors_rgb": {
                label: list(KEYPOINT_COLORS_RGB[index % len(KEYPOINT_COLORS_RGB)])
                for index, label in enumerate(labels)
            },
            "provider_border_colors_rgb": {
                name: list(PROVIDER_BORDER_COLORS_RGB.get(name, (180, 180, 180)))
                for name in provider_names.values()
            },
            "failure_border_color_rgb": list(FAILURE_BORDER_COLOR_RGB),
            "selection": selection,
            "array_evidence": {
                name: _array_record(values)
                for name, values in sorted(evidence_arrays.items())
            },
            "artifacts": artifacts,
            "copied_work_package": package_files,
            "review_instructions": [
                "Inspect all_rows_overview.png for crop centering and gross landmark errors.",
                "Inspect failures_and_low_confidence.png for threshold or source-specific defects.",
                "Inspect provider_spread.png for acquisition-versus-fallback consistency over time.",
                "Record acceptance or rejection in a separate immutable decision bound to this record_sha256.",
            ],
        }
        record = {**body, "record_sha256": canonical_json_sha256(body)}
        record_path = temporary / "review_record.json"
        _write_json(record_path, record)
        temporary.replace(output_dir)
        return {**record, "output_dir": str(output_dir)}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        package.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish exact bounded keypoint work-package visual-review evidence."
    )
    parser.add_argument("analysis_zarr", type=Path)
    parser.add_argument("--work-package-manifest", type=Path, required=True)
    parser.add_argument("--keypoint-run-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-mode",
        choices=("consolidated", "unconsolidated"),
        default="consolidated",
    )
    parser.add_argument("--low-confidence-per-provider", type=int, default=8)
    parser.add_argument("--spread-per-provider", type=int, default=12)
    parser.add_argument("--source-result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    plan = {
        "action": "apply" if args.apply else "dry_run",
        "analysis_zarr": str(args.analysis_zarr.expanduser().resolve()),
        "work_package_manifest": str(
            args.work_package_manifest.expanduser().resolve()
        ),
        "keypoint_run_path": _normalize_group_path(args.keypoint_run_path),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "metadata_mode": args.metadata_mode,
        "low_confidence_per_provider": int(args.low_confidence_per_provider),
        "spread_per_provider": int(args.spread_per_provider),
        "source_result_json": (
            str(args.source_result_json.expanduser().resolve())
            if args.source_result_json is not None
            else None
        ),
    }
    if args.apply:
        plan["result"] = build_review_bundle(
            analysis_zarr=args.analysis_zarr.expanduser().resolve(),
            work_package_manifest=args.work_package_manifest.expanduser().resolve(),
            keypoint_run_path=args.keypoint_run_path,
            output_dir=args.output_dir.expanduser().resolve(),
            metadata_mode=args.metadata_mode,
            low_confidence_per_provider=int(args.low_confidence_per_provider),
            spread_per_provider=int(args.spread_per_provider),
            source_result_json=(
                args.source_result_json.expanduser().resolve()
                if args.source_result_json is not None
                else None
            ),
        )
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
