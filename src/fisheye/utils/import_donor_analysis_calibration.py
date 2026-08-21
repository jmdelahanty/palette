#!/usr/bin/env python3
"""Import one explicitly approved, camera-matched analysis calibration.

The command is dry-run by default.  ``--apply`` copies the donor's complete
``analysis/calibration`` subtree into a target analysis Zarr and writes a
sidecar receipt.  It does not infer a donor: both Zarrs and the expected camera
serial must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

MODULE_NAME = "fisheye.utils.import_donor_analysis_calibration"
RECEIPT_SCHEMA = "palette.donor_analysis_calibration_import.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_group(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
    ]


def _tree_digest(files: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(list(files), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _positive_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return parsed


def _target_recording_dir(target_zarr: Path) -> Path:
    if target_zarr.parent.name != "zarr":
        raise ValueError("target Zarr must be located below a recording/zarr directory")
    return target_zarr.parent.parent


def _source_geometry(recording_dir: Path, camera_serial: str) -> dict[str, Any]:
    snapshot_path = (
        recording_dir / "raw" / "recording_geometry_bundle" / "recording_snapshot.json"
    )
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"recording geometry snapshot not found: {snapshot_path}"
        )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"recording geometry snapshot is not an object: {snapshot_path}"
        )
    runtime = payload.get("camera_runtime")
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get(camera_serial), Mapping
    ):
        raise ValueError(f"camera {camera_serial} is absent from {snapshot_path}")
    camera = runtime[camera_serial]
    width = int(camera.get("width") or 0)
    height = int(camera.get("height") or 0)
    if width <= 0 or height <= 0:
        coordinate_frame = camera.get("coordinate_frame")
        extent = (
            coordinate_frame.get("extent")
            if isinstance(coordinate_frame, Mapping)
            else None
        )
        if isinstance(extent, Mapping):
            width = int(extent.get("width_px") or 0)
            height = int(extent.get("height_px") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"camera dimensions are absent for {camera_serial} in {snapshot_path}"
        )
    return {"path": str(snapshot_path), "width": width, "height": height}


def _preflight(
    target_zarr: Path,
    donor_zarr: Path,
    *,
    expected_camera: str,
) -> dict[str, Any]:
    target_zarr = target_zarr.expanduser().resolve()
    donor_zarr = donor_zarr.expanduser().resolve()
    if not target_zarr.is_dir():
        raise FileNotFoundError(f"target Zarr not found: {target_zarr}")
    if not donor_zarr.is_dir():
        raise FileNotFoundError(f"donor Zarr not found: {donor_zarr}")
    if not expected_camera.isdigit():
        raise ValueError(f"invalid camera serial: {expected_camera!r}")

    target_root = _open_group(target_zarr, mode="r")
    cameras = [str(value) for value in target_root.attrs.get("camera_serials", [])]
    if cameras != [expected_camera]:
        raise ValueError(
            f"target camera mismatch: expected {[expected_camera]!r}, observed {cameras!r}"
        )
    target_recording_id = str(target_root.attrs.get("recording_id") or "")
    if not target_recording_id:
        raise ValueError("target Zarr has no recording_id")
    target_group_path = target_zarr / "analysis" / "calibration"
    if target_group_path.exists():
        raise FileExistsError(
            f"target analysis/calibration already exists: {target_group_path}"
        )

    donor_root = _open_group(donor_zarr, mode="r")
    analysis = donor_root.get("analysis")
    calibration = (
        analysis.get("calibration") if isinstance(analysis, zarr.Group) else None
    )
    if not isinstance(calibration, zarr.Group):
        raise ValueError(f"donor has no analysis/calibration group: {donor_zarr}")
    attrs = dict(calibration.attrs)
    for field in ("active_camera_id", "primary_camera_id"):
        if str(attrs.get(field) or "") != expected_camera:
            raise ValueError(
                f"donor {field} mismatch: expected {expected_camera}, observed {attrs.get(field)!r}"
            )

    pixels_per_mm = _positive_float(
        attrs.get("pixels_per_mm_camera"), field="pixels_per_mm_camera"
    )
    pixel_to_mm = _positive_float(attrs.get("pixel_to_mm"), field="pixel_to_mm")
    if not math.isclose(pixels_per_mm * pixel_to_mm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "donor pixels_per_mm_camera and pixel_to_mm are not reciprocal"
        )

    native_width = int(attrs.get("native_width_px") or 0)
    native_height = int(attrs.get("native_height_px") or 0)
    geometry = _source_geometry(_target_recording_dir(target_zarr), expected_camera)
    if [native_width, native_height] != [geometry["width"], geometry["height"]]:
        raise ValueError(
            "donor native geometry does not match target recording geometry: "
            f"donor={[native_width, native_height]}, "
            f"target={[geometry['width'], geometry['height']]}"
        )

    homography = calibration.get("homography_matrix")
    if not isinstance(homography, zarr.Array):
        raise ValueError("donor analysis/calibration/homography_matrix is missing")
    matrix = np.asarray(homography[:], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("donor homography_matrix must be finite with shape (3, 3)")

    physical = calibration.get("coordinate_frames/source_camera_physical_mm")
    if not isinstance(physical, zarr.Group):
        raise ValueError("donor source-camera physical coordinate record is missing")
    physical_record = physical.attrs.get("physical_frame_calibration")
    if not isinstance(physical_record, Mapping):
        raise ValueError("donor physical_frame_calibration record is missing")
    if str(physical_record.get("camera_id") or "") != expected_camera:
        raise ValueError("donor physical-frame camera does not match expected camera")

    donor_path = donor_zarr / "analysis" / "calibration"
    donor_files = _tree_manifest(donor_path)
    if not donor_files:
        raise ValueError(f"donor calibration tree is empty: {donor_path}")
    return {
        "target_zarr": str(target_zarr),
        "target_recording_id": target_recording_id,
        "donor_zarr": str(donor_zarr),
        "donor_recording_id": str(donor_root.attrs.get("recording_id") or ""),
        "expected_camera": expected_camera,
        "target_geometry": geometry,
        "pixels_per_mm_camera": pixels_per_mm,
        "pixel_to_mm": pixel_to_mm,
        "pixels_per_mm_projector": _positive_float(
            attrs.get("pixels_per_mm_projector"), field="pixels_per_mm_projector"
        ),
        "native_width_px": native_width,
        "native_height_px": native_height,
        "homography_matrix": matrix.tolist(),
        "donor_file_count": len(donor_files),
        "donor_tree_sha256": _tree_digest(donor_files),
        "donor_files": donor_files,
    }


def import_donor_calibration(
    target_zarr: str | Path,
    donor_zarr: str | Path,
    *,
    expected_camera: str,
    operator_note: str,
    apply: bool = False,
) -> dict[str, Any]:
    if not operator_note.strip():
        raise ValueError("operator_note must describe why donor reuse is valid")
    target_path = Path(target_zarr).expanduser().resolve()
    donor_path = Path(donor_zarr).expanduser().resolve()
    preflight = _preflight(
        target_path, donor_path, expected_camera=str(expected_camera).strip()
    )
    receipt_path = (
        target_path.parent / f"{target_path.name}_calibration_import_receipt.json"
    )
    result = {
        "status": "planned" if not apply else "in_progress",
        "schema_id": RECEIPT_SCHEMA,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "operator_note": operator_note.strip(),
        "receipt_path": str(receipt_path),
        **preflight,
    }
    if not apply:
        return result

    source_group_path = donor_path / "analysis" / "calibration"
    target_group_path = target_path / "analysis" / "calibration"
    temporary = target_group_path.with_name(
        f".{target_group_path.name}.donor-import-{os.getpid()}.incomplete"
    )
    if temporary.exists():
        raise FileExistsError(f"temporary import path already exists: {temporary}")
    shutil.copytree(source_group_path, temporary, copy_function=shutil.copy2)
    os.replace(temporary, target_group_path)

    target_root = _open_group(target_path, mode="r+")
    calibration = target_root["analysis/calibration"]
    calibration.attrs.update(
        {
            "immediate_donor_zarr": str(donor_path),
            "immediate_donor_calibration_path": "analysis/calibration",
            "immediate_donor_recording_id": preflight["donor_recording_id"],
            "immediate_donor_calibration_tree_sha256": preflight["donor_tree_sha256"],
            "target_recording_id": preflight["target_recording_id"],
            "operator_configuration_verified": True,
            "operator_configuration_verification_note": operator_note.strip(),
            "imported_at_utc": _utc_now(),
            "imported_by": MODULE_NAME,
        }
    )

    imported_matrix = np.asarray(calibration["homography_matrix"][:], dtype=np.float64)
    if not np.array_equal(imported_matrix, np.asarray(preflight["homography_matrix"])):
        raise RuntimeError("imported homography does not match donor")
    if str(calibration.attrs.get("active_camera_id") or "") != expected_camera:
        raise RuntimeError("imported calibration camera identity changed")
    result.update(
        {
            "status": "pass",
            "completed_at_utc": _utc_now(),
            "target_calibration_path": str(target_group_path),
            "target_file_count": len(_tree_manifest(target_group_path)),
        }
    )
    receipt_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_zarr", type=Path)
    parser.add_argument("donor_zarr", type=Path)
    parser.add_argument("--expected-camera", required=True)
    parser.add_argument("--operator-note", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = import_donor_calibration(
            args.target_zarr,
            args.donor_zarr,
            expected_camera=args.expected_camera,
            operator_note=args.operator_note,
            apply=bool(args.apply),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: camera={result['expected_camera']} "
            f"target={result['target_zarr']} donor={result['donor_zarr']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
