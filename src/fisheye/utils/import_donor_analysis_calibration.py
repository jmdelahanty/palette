#!/usr/bin/env python3
"""Import or repair one explicitly approved, camera-matched calibration.

The command is dry-run by default.  ``--apply`` copies the donor's complete
``analysis/calibration`` subtree into a target analysis Zarr and writes a
sidecar receipt.  Modern coordinate records are rebuilt from the exact source
H5 evidence so they bind the target recording's acquisition authority instead
of retaining the donor recording's binding.  ``--repair-existing`` performs
that rebinding for a calibration previously copied by this tool.  It does not
infer a donor: both Zarrs and the expected camera serial must be supplied
explicitly.
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

import h5py
import numpy as np
import zarr

from fisheye.shared.coordinate_frame_record import (
    PHYSICAL_FRAME_CALIBRATION_ATTR,
    SELECTED_CAMERA_FRAME_EVIDENCE_ATTR,
    parse_physical_frame_calibration_record,
    parse_selected_camera_frame_evidence_record,
)
from fisheye.shared.selected_calibration import (
    SOURCE_ARENA_CONFIG_DATASET_PATH,
    VerifiedSelectedCameraSourceEvidence,
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.source_camera_physical_authority import (
    load_source_camera_physical_authority,
    rebind_source_camera_physical_authority,
    validate_source_camera_physical_authority_rebind,
)

MODULE_NAME = "fisheye.utils.import_donor_analysis_calibration"
RECEIPT_SCHEMA = "palette.donor_analysis_calibration_import.v2"
SOURCE_KIND = "operator_verified_donor_calibration"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_group(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _open_consolidated(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=True)
    except TypeError:
        return zarr.open_group(str(path), mode="r", consolidated=True)


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


def _verified_donor_camera_evidence(
    calibration: zarr.Group,
    *,
    expected_camera: str,
) -> VerifiedSelectedCameraSourceEvidence:
    selected = calibration.get("coordinate_frames/selected_camera_evidence")
    if not isinstance(selected, zarr.Group):
        raise ValueError("donor selected-camera coordinate evidence is missing")
    raw_record = selected.attrs.get(SELECTED_CAMERA_FRAME_EVIDENCE_ATTR)
    parsed_record = parse_selected_camera_frame_evidence_record(raw_record)
    if parsed_record.camera_id != expected_camera:
        raise ValueError("donor selected-camera evidence names another camera")

    source_h5 = Path(
        str(parsed_record.source_camera.get("source_h5_path") or "")
    ).expanduser()
    if not source_h5.is_absolute() or not source_h5.is_file():
        raise FileNotFoundError(
            f"donor selected-camera source H5 is unavailable: {source_h5}"
        )
    source_h5 = source_h5.resolve()
    with h5py.File(source_h5, "r") as handle:
        actual = Path(str(handle.filename)).expanduser().resolve()
        try:
            descriptor = handle.id.get_vfd_handle()
            if isinstance(descriptor, tuple):
                descriptor = descriptor[0]
            open_stat = os.fstat(int(descriptor))
            path_stat = os.stat(source_h5)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "unable to bind the donor source H5 handle to its exact file"
            ) from exc
        if actual != source_h5 or (
            open_stat.st_dev,
            open_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError("donor source H5 changed while it was opened")
        if SOURCE_ARENA_CONFIG_DATASET_PATH not in handle or not isinstance(
            handle[SOURCE_ARENA_CONFIG_DATASET_PATH], h5py.Dataset
        ):
            raise ValueError(
                f"donor source H5 lacks {SOURCE_ARENA_CONFIG_DATASET_PATH}"
            )
        camera_group_path = f"/calibration_snapshot/{expected_camera}"
        if camera_group_path not in handle or not isinstance(
            handle[camera_group_path], h5py.Group
        ):
            raise ValueError(
                f"donor source H5 lacks exact camera group {camera_group_path}"
            )
        camera_group = handle[camera_group_path]
        verified = build_selected_camera_source_evidence_from_h5_values(
            source_h5_path=str(source_h5),
            arena_config_raw=handle[SOURCE_ARENA_CONFIG_DATASET_PATH][()],
            camera_group_path=camera_group_path,
            camera_group_attrs=dict(camera_group.attrs),
            expected_camera_id=expected_camera,
        )
    if verified.to_dict() != parsed_record.source_camera:
        raise ValueError(
            "donor persisted selected-camera evidence differs from the exact "
            "source H5 nodes"
        )
    return verified


def _validate_existing_import(
    target_root: zarr.Group,
    *,
    donor_path: Path,
    expected_camera: str,
    preflight: Mapping[str, Any],
    evidence: VerifiedSelectedCameraSourceEvidence,
) -> dict[str, Any]:
    calibration = target_root.get("analysis/calibration")
    if not isinstance(calibration, zarr.Group):
        raise ValueError("target has no existing analysis/calibration to repair")
    attrs = dict(calibration.attrs)
    if attrs.get("imported_by") != MODULE_NAME:
        raise ValueError("existing target calibration was not imported by this tool")
    if Path(str(attrs.get("immediate_donor_zarr") or "")).resolve() != donor_path:
        raise ValueError("existing target calibration names another immediate donor")
    if attrs.get("operator_configuration_verified") is not True:
        raise ValueError(
            "existing target donor configuration was not operator verified"
        )
    for field in ("active_camera_id", "primary_camera_id"):
        if str(attrs.get(field) or "") != expected_camera:
            raise ValueError(f"existing target {field} names another camera")
    for field in ("pixels_per_mm_camera", "pixel_to_mm"):
        if _positive_float(attrs.get(field), field=field) != float(preflight[field]):
            raise ValueError(f"existing target {field} differs from the donor")
    matrix = np.asarray(calibration["homography_matrix"][:], dtype=np.float64)
    if not np.array_equal(matrix, np.asarray(preflight["homography_matrix"])):
        raise ValueError("existing target homography differs from the donor")

    selected = calibration.get("coordinate_frames/selected_camera_evidence")
    physical = calibration.get("coordinate_frames/source_camera_physical_mm")
    if not isinstance(selected, zarr.Group) or not isinstance(physical, zarr.Group):
        raise ValueError("existing target coordinate calibration records are missing")
    selected_record = parse_selected_camera_frame_evidence_record(
        selected.attrs.get(SELECTED_CAMERA_FRAME_EVIDENCE_ATTR)
    )
    if selected_record.source_camera != evidence.to_dict():
        raise ValueError(
            "existing target selected-camera evidence differs from the exact donor H5"
        )
    physical_record = parse_physical_frame_calibration_record(
        physical.attrs.get(PHYSICAL_FRAME_CALIBRATION_ATTR)
    )
    if (
        physical_record.camera_id != expected_camera
        or physical_record.pixels_per_mm_camera != evidence.pixels_per_mm_camera
    ):
        raise ValueError("existing target physical record differs from donor evidence")
    try:
        bound = load_source_camera_physical_authority(target_root)
    except (KeyError, ValueError) as exc:
        validate_source_camera_physical_authority_rebind(
            target_root,
            source_camera_evidence=evidence,
        )
        return {
            "target_authority_status": "requires_rebind",
            "target_authority_error": str(exc),
        }
    return {
        "target_authority_status": "already_valid",
        "target_authority_record_sha256": bound.manifest.record_sha256,
    }


def _preflight(
    target_zarr: Path,
    donor_zarr: Path,
    *,
    expected_camera: str,
    repair_existing: bool,
) -> tuple[dict[str, Any], VerifiedSelectedCameraSourceEvidence]:
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
    if target_group_path.exists() and not repair_existing:
        raise FileExistsError(
            f"target analysis/calibration already exists: {target_group_path}"
        )
    if repair_existing and not target_group_path.exists():
        raise FileNotFoundError(
            f"target analysis/calibration does not exist: {target_group_path}"
        )

    donor_root = _open_group(donor_zarr, mode="r")
    analysis = donor_root.get("analysis")
    calibration = (
        analysis.get("calibration") if isinstance(analysis, zarr.Group) else None
    )
    if not isinstance(calibration, zarr.Group):
        raise ValueError(f"donor has no analysis/calibration group: {donor_zarr}")
    donor_authority = load_source_camera_physical_authority(donor_root)
    if donor_authority.camera_id != expected_camera:
        raise ValueError("donor physical authority names another camera")
    evidence = _verified_donor_camera_evidence(
        calibration,
        expected_camera=expected_camera,
    )
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
    if (
        evidence.pixels_per_mm_camera != pixels_per_mm
        or donor_authority.mm_per_pixel != pixel_to_mm
    ):
        raise ValueError(
            "donor calibration attrs, selected-camera evidence, and physical "
            "authority do not share one exact camera scale"
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
    result = {
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
    if repair_existing:
        result.update(
            _validate_existing_import(
                target_root,
                donor_path=donor_zarr,
                expected_camera=expected_camera,
                preflight=result,
                evidence=evidence,
            )
        )
    return result, evidence


def import_donor_calibration(
    target_zarr: str | Path,
    donor_zarr: str | Path,
    *,
    expected_camera: str,
    operator_note: str,
    apply: bool = False,
    repair_existing: bool = False,
) -> dict[str, Any]:
    if not operator_note.strip():
        raise ValueError("operator_note must describe why donor reuse is valid")
    target_path = Path(target_zarr).expanduser().resolve()
    donor_path = Path(donor_zarr).expanduser().resolve()
    preflight, evidence = _preflight(
        target_path,
        donor_path,
        expected_camera=str(expected_camera).strip(),
        repair_existing=bool(repair_existing),
    )
    receipt_path = target_path.parent / (
        f"{target_path.name}_calibration_authority_repair_receipt.json"
        if repair_existing
        else f"{target_path.name}_calibration_import_receipt.json"
    )
    result = {
        "status": "planned" if not apply else "in_progress",
        "schema_id": RECEIPT_SCHEMA,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "operator_note": operator_note.strip(),
        "operation": "repair_existing" if repair_existing else "import",
        "receipt_path": str(receipt_path),
        **preflight,
    }
    if not apply:
        return result

    target_group_path = target_path / "analysis" / "calibration"
    if not repair_existing:
        source_group_path = donor_path / "analysis" / "calibration"
        temporary = target_group_path.with_name(
            f".{target_group_path.name}.donor-import-{os.getpid()}.incomplete"
        )
        if temporary.exists():
            raise FileExistsError(f"temporary import path already exists: {temporary}")
        shutil.copytree(source_group_path, temporary, copy_function=shutil.copy2)
        os.replace(temporary, target_group_path)

    target_root = _open_group(target_path, mode="r+")
    calibration = target_root["analysis/calibration"]
    if preflight.get("target_authority_status") == "already_valid":
        authority = load_source_camera_physical_authority(target_root)
    else:
        authority = rebind_source_camera_physical_authority(
            target_root,
            source_camera_evidence=evidence,
            source_kind=SOURCE_KIND,
            provenance={
                "generated_by": MODULE_NAME,
                "immediate_donor_zarr": str(donor_path),
                "immediate_donor_recording_id": preflight["donor_recording_id"],
                "source_h5": evidence.source_h5_path,
                "operator_configuration_verified": True,
                "operator_configuration_verification_note": operator_note.strip(),
                "target_recording_id": preflight["target_recording_id"],
            },
        )

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
    if repair_existing:
        calibration.attrs.update(
            {
                "physical_authority_repaired_at_utc": _utc_now(),
                "physical_authority_repaired_by": MODULE_NAME,
                "physical_authority_repair_note": operator_note.strip(),
            }
        )

    imported_matrix = np.asarray(calibration["homography_matrix"][:], dtype=np.float64)
    if not np.array_equal(imported_matrix, np.asarray(preflight["homography_matrix"])):
        raise RuntimeError("imported homography does not match donor")
    if str(calibration.attrs.get("active_camera_id") or "") != expected_camera:
        raise RuntimeError("imported calibration camera identity changed")
    authority.assert_verified()
    zarr.consolidate_metadata(str(target_path))
    consolidated_root = _open_consolidated(target_path)
    consolidated_authority = load_source_camera_physical_authority(consolidated_root)
    if (
        consolidated_authority.manifest.record_sha256
        != authority.manifest.record_sha256
    ):
        raise RuntimeError(
            "consolidated source-camera authority differs from direct metadata"
        )
    result.update(
        {
            "status": "pass",
            "completed_at_utc": _utc_now(),
            "target_calibration_path": str(target_group_path),
            "target_file_count": len(_tree_manifest(target_group_path)),
            "target_authority_record_ref": authority.manifest.record_ref,
            "target_authority_record_sha256": authority.manifest.record_sha256,
            "target_source_camera_frame_sha256": (
                authority.physical_frame.source_camera_pixels.record_sha256
            ),
            "consolidated_metadata_validated": True,
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
    parser.add_argument("--repair-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = import_donor_calibration(
            args.target_zarr,
            args.donor_zarr,
            expected_camera=args.expected_camera,
            operator_note=args.operator_note,
            apply=bool(args.apply),
            repair_existing=bool(args.repair_existing),
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
