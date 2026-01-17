#!/usr/bin/env python3
"""
Prepare a YOLO detection training config + manifest from Palette Zarr archives.

Validates crop provenance, bbox integrity, and downsample frame availability.
Optionally captures experiment provenance (arena/camera/calibration) for auditing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
import zarr
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..utils.zarr_metadata import get_downsample_array_path, get_downsample_shape


class CameraParameters(BaseModel):
    camera_id: Optional[str] = None
    native_width_px: Optional[int] = None
    native_height_px: Optional[int] = None
    pixels_per_mm_camera: Optional[float] = None
    pixels_per_mm_projector: Optional[float] = None
    homography_matrix: Optional[List[List[float]]] = None
    stimulus_offset_x: Optional[float] = None
    stimulus_offset_y: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class ArenaParameters(BaseModel):
    arena_id: Optional[str] = None
    arena_number: Optional[str] = None
    dish_design: Optional[str] = None
    arena_config_name: Optional[str] = None
    shape: Optional[str] = None
    center_x_px: Optional[float] = None
    center_y_px: Optional[float] = None
    radius_px: Optional[float] = None
    width_px: Optional[float] = None
    height_px: Optional[float] = None
    corner_radius_px: Optional[float] = None
    diameter_mm: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class CalibrationSummary(BaseModel):
    pixel_to_mm: Optional[float] = None
    pixels_per_mm: Optional[float] = None
    pixels_per_mm_camera: Optional[float] = None
    pixels_per_mm_projector: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class ProvenanceInfo(BaseModel):
    source_h5: Optional[str] = None
    stimulus_run: Optional[str] = None
    arena: Optional[ArenaParameters] = None
    camera: Optional[CameraParameters] = None
    calibration: Optional[CalibrationSummary] = None
    rig_info: Optional[Dict[str, Any]] = None
    arena_config: Optional[Dict[str, Any]] = None
    provenance_source: str = "missing"
    missing_fields: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    override_fields: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class DatasetManifest(BaseModel):
    name: str
    zarr_path: str
    crop_run: Optional[str] = None
    bbox_array_path: str
    detection_source_type: str
    detection_source_path: Optional[str] = None
    includes_interpolated: bool = False
    input_format: str
    images_ds_shape: List[int]
    total_bboxes: int
    invalid_bboxes: int
    invalid_bbox_sample: List[int] = Field(default_factory=list)
    detection_source_counts: Dict[str, int] = Field(default_factory=dict)
    frame_indices_present: bool = False
    detection_source_present: bool = False
    provenance: Optional[ProvenanceInfo] = None

    model_config = ConfigDict(extra="allow")


class TrainingManifest(BaseModel):
    created_at_utc: str
    task: str
    source_type: str
    input_format: str
    imgsz: List[int]
    datasets: List[DatasetManifest]
    base_config_path: Optional[str] = None
    output_config_path: Optional[str] = None
    output_manifest_path: Optional[str] = None
    project: Optional[str] = None
    run_name: Optional[str] = None
    provenance_policy: str

    model_config = ConfigDict(extra="allow")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def _parse_arena_config(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _build_arena_info(arena_config: Dict[str, Any], root: zarr.Group) -> ArenaParameters:
    arena_id = arena_config.get("arena_id") or root.attrs.get("arena_id")
    arena_number = (
        arena_config.get("arena_number")
        or root.attrs.get("arena_number")
        or root.attrs.get("arena_id")
    )
    arena_config_name = arena_config.get("arena_config_name")
    dish_design = (
        arena_config.get("dish_design")
        or arena_config.get("arena_design")
        or arena_config_name
        or root.attrs.get("dish_design")
    )
    shape = arena_config.get("experimental_area_shape") or arena_config.get("swimmable_area_shape")
    center_x = arena_config.get("experimental_area_center_x_px") or arena_config.get(
        "swimmable_area_center_x_px"
    )
    center_y = arena_config.get("experimental_area_center_y_px") or arena_config.get(
        "swimmable_area_center_y_px"
    )
    radius = arena_config.get("experimental_area_radius_px") or arena_config.get("swimmable_area_radius_px")
    width = arena_config.get("experimental_area_width_px") or arena_config.get("swimmable_area_width_px")
    height = arena_config.get("experimental_area_height_px") or arena_config.get("swimmable_area_height_px")
    corner_radius = arena_config.get("experimental_area_corner_radius_px")

    diameter_mm = None
    if "calibration" in root and "arena" in root["calibration"]:
        arena_group = root["calibration"]["arena"]
        diameter_mm = _as_float(arena_group.attrs.get("diameter_mm"))

    return ArenaParameters(
        arena_id=str(arena_id) if arena_id is not None else None,
        arena_number=str(arena_number) if arena_number is not None else None,
        dish_design=str(dish_design) if dish_design is not None else None,
        arena_config_name=str(arena_config_name) if arena_config_name is not None else None,
        shape=str(shape) if shape is not None else None,
        center_x_px=_as_float(center_x),
        center_y_px=_as_float(center_y),
        radius_px=_as_float(radius),
        width_px=_as_float(width),
        height_px=_as_float(height),
        corner_radius_px=_as_float(corner_radius),
        diameter_mm=_as_float(diameter_mm),
    )


def _load_camera_from_stimulus(stim_group: zarr.Group) -> Optional[CameraParameters]:
    calib_group = stim_group.get("calibration")
    if calib_group is None:
        return None
    camera_ids = list(calib_group.keys())
    if not camera_ids:
        return None
    camera_id = camera_ids[0]
    cam_group = calib_group.get(camera_id)
    if cam_group is None:
        return None
    homography = None
    if "homography_matrix" in cam_group:
        try:
            homography = np.asarray(cam_group["homography_matrix"][:], dtype=float).tolist()
        except Exception:
            homography = None
    return CameraParameters(
        camera_id=str(camera_id),
        native_width_px=_as_int(cam_group.attrs.get("camera_width_px")),
        native_height_px=_as_int(cam_group.attrs.get("camera_height_px")),
        pixels_per_mm_camera=_as_float(cam_group.attrs.get("pixels_per_mm_camera")),
        pixels_per_mm_projector=_as_float(cam_group.attrs.get("pixels_per_mm_projector")),
        stimulus_offset_x=_as_float(cam_group.attrs.get("stimulus_offset_x")),
        stimulus_offset_y=_as_float(cam_group.attrs.get("stimulus_offset_y")),
        homography_matrix=homography,
    )


def _load_camera_from_root(root: zarr.Group) -> Optional[CameraParameters]:
    calib_group = root.get("calibration")
    if calib_group is None:
        return None
    camera_group = calib_group.get("cameras")
    if camera_group is None:
        return None
    camera_ids = list(camera_group.keys())
    if not camera_ids:
        return None
    camera_id = calib_group.attrs.get("primary_camera_id") or camera_ids[0]
    cam_group = camera_group.get(str(camera_id))
    if cam_group is None:
        return None
    homography = None
    if "homography_matrix" in cam_group:
        try:
            homography = np.asarray(cam_group["homography_matrix"][:], dtype=float).tolist()
        except Exception:
            homography = None
    return CameraParameters(
        camera_id=str(camera_id),
        native_width_px=_as_int(cam_group.attrs.get("native_width_px")),
        native_height_px=_as_int(cam_group.attrs.get("native_height_px")),
        pixels_per_mm_camera=_as_float(cam_group.attrs.get("pixels_per_mm_camera")),
        pixels_per_mm_projector=_as_float(cam_group.attrs.get("pixels_per_mm_projector")),
        stimulus_offset_x=_as_float(cam_group.attrs.get("stimulus_offset_x")),
        stimulus_offset_y=_as_float(cam_group.attrs.get("stimulus_offset_y")),
        homography_matrix=homography,
    )


def _load_calibration_summary(root: zarr.Group) -> CalibrationSummary:
    summary = CalibrationSummary()
    calib_group = root.get("calibration")
    if calib_group is None:
        return summary
    summary.pixel_to_mm = _as_float(calib_group.attrs.get("pixel_to_mm"))
    summary.pixels_per_mm = _as_float(calib_group.attrs.get("pixels_per_mm"))
    summary.pixels_per_mm_camera = _as_float(calib_group.attrs.get("pixels_per_mm_camera"))
    summary.pixels_per_mm_projector = _as_float(calib_group.attrs.get("pixels_per_mm_projector"))
    return summary


def _load_rig_info(root: zarr.Group) -> Optional[Dict[str, Any]]:
    calib_group = root.get("calibration")
    if calib_group is None:
        return None
    rig_group = calib_group.get("rig_info")
    if rig_group is None:
        return None
    return {k: rig_group.attrs.get(k) for k in rig_group.attrs.keys()}


def _extract_provenance(
    root: zarr.Group,
    override: Optional[Dict[str, Any]],
    provenance_policy: str,
) -> ProvenanceInfo:
    provenance = ProvenanceInfo()

    stim_group = None
    stimulus_run = None
    if "analysis" in root and "stimulus_runs" in root["analysis"]:
        stim_parent = root["analysis"]["stimulus_runs"]
        stimulus_run = stim_parent.attrs.get("latest")
        if stimulus_run and stimulus_run in stim_parent:
            stim_group = stim_parent[stimulus_run]
            provenance.stimulus_run = str(stimulus_run)
            provenance.source_h5 = stim_group.attrs.get("source_h5")

    arena_config = {}
    if stim_group is not None:
        arena_config = _parse_arena_config(stim_group.attrs.get("arena_config_json"))
        if arena_config:
            provenance.arena_config = arena_config
            provenance.arena = _build_arena_info(arena_config, root)
        provenance.camera = _load_camera_from_stimulus(stim_group)

    if provenance.arena is None:
        provenance.arena = _build_arena_info({}, root)
    if provenance.camera is None:
        provenance.camera = _load_camera_from_root(root)

    provenance.calibration = _load_calibration_summary(root)
    provenance.rig_info = _load_rig_info(root)

    if stim_group is not None:
        provenance.provenance_source = "stimulus_import"
    elif "calibration" in root:
        provenance.provenance_source = "zarr"

    if override:
        override_fields = []
        try:
            override_model = ProvenanceInfo.model_validate(override)
        except ValidationError:
            override_model = ProvenanceInfo.model_validate(override, strict=False)
        for field in ("arena", "camera", "calibration", "rig_info", "arena_config", "source_h5", "stimulus_run"):
            override_value = getattr(override_model, field)
            if override_value is not None:
                setattr(provenance, field, override_value)
                override_fields.append(field)
        provenance.override_fields = override_fields
        provenance.provenance_source = "user_override" if provenance.provenance_source == "missing" else "mixed"

    missing = []
    arena = provenance.arena
    camera = provenance.camera
    if not arena or not arena.dish_design:
        missing.append("dish_design")
    if not arena or not arena.arena_number:
        missing.append("arena_number")
    if not camera or not camera.camera_id:
        missing.append("camera_id")
    if not camera or camera.pixels_per_mm_camera is None:
        missing.append("pixels_per_mm_camera")

    provenance.missing_fields = missing

    if missing and provenance_policy == "warn":
        provenance.warnings.append(
            "Missing provenance fields: " + ", ".join(missing)
        )
    if missing and provenance_policy == "strict":
        raise ValueError(
            "Missing required provenance fields: "
            + ", ".join(missing)
            + ". Provide --metadata-json or run import_stimulus_to_zarr."
        )

    return provenance


def _load_override(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Metadata override file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _validate_bboxes(bbox: np.ndarray) -> Tuple[int, List[int]]:
    if bbox.size == 0:
        return 0, []
    nan_mask = np.isnan(bbox).any(axis=1)
    out_of_range = (bbox[:, 0] < 0) | (bbox[:, 0] > 1) | (bbox[:, 1] < 0) | (bbox[:, 1] > 1)
    invalid_dims = (bbox[:, 2] <= 0) | (bbox[:, 3] <= 0)
    invalid = nan_mask | out_of_range | invalid_dims
    invalid_indices = np.where(invalid)[0].astype(int).tolist()
    return int(np.sum(invalid)), invalid_indices[:10]


def _ensure_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Base config is empty or invalid.")
    if "datasets" not in config:
        raise ValueError("Base config missing 'datasets' section.")
    if "training_params" not in config:
        raise ValueError("Base config missing 'training_params' section.")
    return config


def _choose_dataset_name(existing: set, path: Path, index: int) -> str:
    stem = path.stem
    name = stem
    if name in existing:
        name = f"{stem}_{index}"
    existing.add(name)
    return name


def _print_summary(manifest: TrainingManifest) -> None:
    print("\nDetection Training Preflight")
    print(f"  Task: {manifest.task}")
    print(f"  Source type: {manifest.source_type}")
    print(f"  Input format: {manifest.input_format}")
    print(f"  imgsz: {manifest.imgsz}")
    for dataset in manifest.datasets:
        print(f"\nDataset: {dataset.name}")
        print(f"  Zarr: {dataset.zarr_path}")
        print(f"  Crop run: {dataset.crop_run}")
        print(f"  Detection source: {dataset.detection_source_type}")
        print(f"  Detection source path: {dataset.detection_source_path}")
        print(f"  Total bboxes: {dataset.total_bboxes}")
        print(f"  Invalid bboxes: {dataset.invalid_bboxes}")
        if dataset.invalid_bbox_sample:
            print(f"  Invalid bbox sample: {dataset.invalid_bbox_sample}")
        if dataset.detection_source_counts:
            print(f"  Detection source counts: {dataset.detection_source_counts}")
        if dataset.provenance and dataset.provenance.warnings:
            for warning in dataset.provenance.warnings:
                print(f"  ⚠ {warning}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a YOLO detection training config and manifest from Palette Zarr data."
    )
    parser.add_argument("zarr_paths", nargs="+", type=Path, help="One or more Palette Zarr paths.")
    parser.add_argument(
        "--source-type",
        choices=["detect", "filtered", "interpolated"],
        default="filtered",
        help="Detection source type to train on.",
    )
    parser.add_argument(
        "--input-format",
        choices=["gray", "rgb"],
        default="gray",
        help="Downsample frame format (images_ds or images_ds_rgb).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/fisheye/detect_config.yaml"),
        help="Base detect training config to use for hyperparameters.",
    )
    parser.add_argument("--out-config", type=Path, help="Output path for the generated config YAML.")
    parser.add_argument("--out-manifest", type=Path, help="Output path for the manifest JSON.")
    parser.add_argument("--project", type=str, help="Ultralytics project directory for outputs.")
    parser.add_argument("--run-name", type=str, help="Suggested run name for training.")
    parser.add_argument("--imgsz", type=int, help="Override training image size.")
    parser.add_argument(
        "--provenance-policy",
        choices=["warn", "strict", "ignore"],
        default="warn",
        help="How to handle missing provenance fields.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional JSON/YAML file with provenance overrides.",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Allow crop source type to differ from requested source_type.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config + manifest without writing files.",
    )

    args = parser.parse_args(argv)

    override = _load_override(args.metadata_json)

    if not args.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    base_config = _ensure_config(yaml.safe_load(args.base_config.read_text(encoding="utf-8")))

    dataset_entries: Dict[str, Dict[str, Any]] = {}
    manifests: List[DatasetManifest] = []
    seen_names: set = set()
    reference_shape: Optional[Tuple[int, int]] = None

    for idx, zarr_path in enumerate(args.zarr_paths, start=1):
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr path not found: {zarr_path}")
        root = zarr.open_group(str(zarr_path), mode="r")

        array_path = get_downsample_array_path(root, format_hint=args.input_format)
        if array_path is None:
            raise ValueError(
                f"Downsampled frames for format '{args.input_format}' not found in {zarr_path.name}."
            )
        ds_shape = get_downsample_shape(root, format_hint=args.input_format)
        if ds_shape is None or len(ds_shape) < 2:
            raise ValueError(f"Downsampled frame shape missing for {zarr_path.name}.")
        height, width = int(ds_shape[0]), int(ds_shape[1])
        if reference_shape is None:
            reference_shape = (height, width)
        elif reference_shape != (height, width) and args.imgsz is None:
            raise ValueError(
                f"Downsample sizes differ across datasets ({reference_shape} vs {(height, width)}). "
                "Use --imgsz to override."
            )

        crop_run = None
        crop_group = None
        bbox_array_path = ""
        detection_source_type = "detect"
        detection_source_path = None
        includes_interpolated = False
        detection_source_counts: Dict[str, int] = {}
        frame_indices_present = False
        detection_source_present = False

        if "crop_runs" in root and root["crop_runs"].attrs.get("latest"):
            crop_run = root["crop_runs"].attrs.get("latest")
            crop_group = root["crop_runs"][crop_run]
            if "bbox_norm_coords" not in crop_group:
                raise ValueError(f"crop_runs/{crop_run} missing bbox_norm_coords in {zarr_path.name}.")
            bbox_array_path = f"crop_runs/{crop_run}/bbox_norm_coords"
            detection_source_type = crop_group.attrs.get("detection_source_type", "detect")
            detection_source_path = crop_group.attrs.get("detection_source_path")
            includes_interpolated = bool(crop_group.attrs.get("includes_interpolated", False))
            frame_indices_present = "frame_indices" in crop_group
            detection_source_present = "detection_source" in crop_group
            if detection_source_present:
                detection_source = np.asarray(crop_group["detection_source"][:], dtype=np.int8)
                unique, counts = np.unique(detection_source, return_counts=True)
                detection_source_counts = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        elif "detect_runs" in root and root["detect_runs"].attrs.get("latest"):
            detect_run = root["detect_runs"].attrs.get("latest")
            detect_group = root["detect_runs"][detect_run]
            if "bbox_norm_coords" not in detect_group:
                raise ValueError(f"detect_runs/{detect_run} missing bbox_norm_coords in {zarr_path.name}.")
            bbox_array_path = f"detect_runs/{detect_run}/bbox_norm_coords"
            frame_indices_present = "frame_indices" in detect_group
        else:
            raise ValueError(f"No crop_runs or detect_runs found in {zarr_path.name}.")

        if not args.allow_source_mismatch and crop_group is not None:
            if detection_source_type != args.source_type:
                raise ValueError(
                    f"{zarr_path.name}: crop source type is '{detection_source_type}', "
                    f"requested '{args.source_type}'. Re-run crop or pass --allow-source-mismatch."
                )

        if detection_source_type in {"filtered", "interpolated"} and detection_source_path:
            if detection_source_path not in root:
                raise ValueError(
                    f"{zarr_path.name}: detection_source_path '{detection_source_path}' not found in Zarr."
                )
        if args.source_type in {"filtered", "detect"} and includes_interpolated and not detection_source_present:
            raise ValueError(
                f"{zarr_path.name}: crop run includes interpolated ROIs but detection_source is missing."
            )

        bboxes = np.asarray(root[bbox_array_path][:], dtype=np.float32)
        invalid_count, invalid_sample = _validate_bboxes(bboxes)

        dataset_name = _choose_dataset_name(seen_names, zarr_path, idx)
        dataset_entries[dataset_name] = {
            "zarr_path": str(zarr_path),
            "source_type": args.source_type,
            "input_format": args.input_format,
        }

        provenance = _extract_provenance(root, override, args.provenance_policy)
        if args.provenance_policy == "ignore":
            provenance.warnings = []

        manifests.append(
            DatasetManifest(
                name=dataset_name,
                zarr_path=str(zarr_path),
                crop_run=crop_run,
                bbox_array_path=bbox_array_path,
                detection_source_type=detection_source_type,
                detection_source_path=detection_source_path,
                includes_interpolated=includes_interpolated,
                input_format=args.input_format,
                images_ds_shape=[height, width],
                total_bboxes=int(bboxes.shape[0]),
                invalid_bboxes=invalid_count,
                invalid_bbox_sample=invalid_sample,
                detection_source_counts=detection_source_counts,
                frame_indices_present=frame_indices_present,
                detection_source_present=detection_source_present,
                provenance=provenance,
            )
        )

    imgsz = args.imgsz if args.imgsz is not None else int(reference_shape[0])
    if args.imgsz is None and reference_shape and reference_shape[0] != reference_shape[1]:
        imgsz = list(reference_shape)
    elif args.imgsz is not None:
        imgsz = int(args.imgsz)
    manifest_imgsz = (
        list(reference_shape)
        if reference_shape is not None
        else [int(imgsz), int(imgsz)]
    )
    if isinstance(imgsz, int):
        manifest_imgsz = [int(imgsz), int(imgsz)]
    elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        manifest_imgsz = [int(imgsz[0]), int(imgsz[1])]

    base_config["datasets"] = dataset_entries
    base_config["task"] = "detect"
    if "training_params" not in base_config:
        base_config["training_params"] = {}
    base_config["training_params"]["imgsz"] = imgsz
    if args.project:
        base_config["training_params"]["project"] = args.project

    manifest = TrainingManifest(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        task="detect",
        source_type=args.source_type,
        input_format=args.input_format,
        imgsz=manifest_imgsz,
        datasets=manifests,
        base_config_path=str(args.base_config),
        output_config_path=str(args.out_config) if args.out_config else None,
        output_manifest_path=str(args.out_manifest) if args.out_manifest else None,
        project=args.project,
        run_name=args.run_name,
        provenance_policy=args.provenance_policy,
    )

    _print_summary(manifest)

    config_yaml = yaml.safe_dump(base_config, sort_keys=False)
    manifest_json = json.dumps(manifest.model_dump(exclude_none=True), indent=2)

    if args.dry_run:
        print("\n--- Generated Config (YAML) ---")
        print(config_yaml.strip())
        print("\n--- Training Manifest (JSON) ---")
        print(manifest_json)
        return

    if args.out_config is None:
        raise ValueError("--out-config is required unless --dry-run is set.")
    out_manifest = args.out_manifest
    if out_manifest is None:
        out_manifest = args.out_config.with_suffix(".manifest.json")

    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    args.out_config.write_text(config_yaml, encoding="utf-8")
    out_manifest.write_text(manifest_json, encoding="utf-8")

    print(f"\nWrote config: {args.out_config}")
    print(f"Wrote manifest: {out_manifest}")
    if args.run_name:
        print(
            "Next: python -m fisheye.training.train_detection "
            f"{args.out_config} --run-name {args.run_name}"
        )
    else:
        print(f"Next: python -m fisheye.training.train_detection {args.out_config}")


if __name__ == "__main__":  # pragma: no cover
    main()
