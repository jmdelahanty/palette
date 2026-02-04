import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import h5py
import zarr


DEFAULT_KEYS = [
    "dish_mask",
    "detection_tuning",
    "keypoint_tuning",
    "eye_mask_tuning",
    "subdish_mask_tuning",
]


@dataclass
class TargetPlan:
    recording_dir: Path
    h5_path: Path
    zarr_path: Path
    camera_id: Optional[str]
    status: str
    reason: Optional[str] = None


def _normalize_attr(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    match = re.search(r"cam_(\d+)", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else None


def _read_camera_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        if "camera_id" in root:
            cam = _normalize_attr(root.get("camera_id"))
            if cam:
                return cam
        ipc = _normalize_attr(root.get("ipc_source_name"))
        return _derive_camera_id(ipc)


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_h5(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            if path.suffix.lower() in {".h5", ".hdf5"}:
                yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("raw/*.h5")
            yield from path.rglob("raw/*.hdf5")
        else:
            yield from path.glob("*/raw/*.h5")
            yield from path.glob("*/raw/*.hdf5")


def _build_plans(roots: List[Path], recursive: bool, camera_id: str) -> List[TargetPlan]:
    plans: List[TargetPlan] = []
    for h5_path in _iter_h5(roots, recursive):
        recording_dir = h5_path.parent.parent
        cam = _read_camera_id(h5_path)
        if cam != camera_id:
            continue
        zarr_path = recording_dir / "zarr" / f"{h5_path.stem}.zarr"
        if not zarr_path.exists():
            plans.append(
                TargetPlan(
                    recording_dir=recording_dir,
                    h5_path=h5_path,
                    zarr_path=zarr_path,
                    camera_id=cam,
                    status="missing",
                    reason="zarr missing",
                )
            )
            continue
        plans.append(
            TargetPlan(
                recording_dir=recording_dir,
                h5_path=h5_path,
                zarr_path=zarr_path,
                camera_id=cam,
                status="ok",
            )
        )
    return plans


def _source_camera_id(source_zarr: Path) -> Optional[str]:
    recording_dir = source_zarr.parent.parent
    h5_path = recording_dir / "raw" / f"{source_zarr.stem}.h5"
    if h5_path.exists():
        return _read_camera_id(h5_path)
    return None


def _parse_keys(keys: Optional[str]) -> List[str]:
    if keys is None or keys.strip() == "":
        return list(DEFAULT_KEYS)
    if keys.strip().lower() == "all":
        return list(DEFAULT_KEYS)
    return [key.strip() for key in keys.split(",") if key.strip()]


def _load_source_tuning(source_zarr: Path, keys: Sequence[str]) -> Dict[str, object]:
    root = zarr.open(str(source_zarr), mode="r")
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return {}
    attrs = dict(analysis.attrs)
    return {key: attrs[key] for key in keys if key in attrs}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply tuning parameters from a source Zarr to recordings with the same camera_id.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording root(s) to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source Zarr with tuned analysis_metadata attrs.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        help="Camera ID to apply to (defaults to camera_id inferred from source).",
    )
    parser.add_argument(
        "--keys",
        type=str,
        help=f"Comma-separated tuning keys to copy (default: {', '.join(DEFAULT_KEYS)}; use 'all').",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for recordings under each root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default: dry-run).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tuning keys in target Zarrs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines for each target plan.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    source = args.source.expanduser()
    if not source.exists():
        print(f"Source Zarr not found: {source}")
        return 1

    camera_id = args.camera_id
    if not camera_id:
        camera_id = _source_camera_id(source)
    if not camera_id:
        print("Unable to infer camera_id from source; pass --camera-id explicitly.")
        return 1

    keys = _parse_keys(args.keys)
    tuning = _load_source_tuning(source, keys)
    if not tuning:
        print("No tuning keys found in source analysis_metadata.")
        return 1

    plans = _build_plans(roots, args.recursive, camera_id)
    if not plans:
        print(f"No recordings found for camera_id {camera_id}.")
        return 1

    applied = 0
    skipped = 0
    missing = 0

    for plan in plans:
        if plan.status == "missing":
            missing += 1
            if args.json:
                print(json.dumps({"status": "missing", "zarr": str(plan.zarr_path)}))
            else:
                print(f"Skipping (missing zarr): {plan.zarr_path}")
            continue

        if not args.apply:
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": "plan",
                            "zarr": str(plan.zarr_path),
                            "camera_id": plan.camera_id,
                            "keys": list(tuning.keys()),
                        }
                    )
                )
            else:
                print(f"Would apply to {plan.zarr_path} (camera_id={plan.camera_id})")
            continue

        root = zarr.open(str(plan.zarr_path), mode="a")
        analysis = root.require_group("analysis_metadata")
        target_attrs = analysis.attrs
        updated = []
        skipped_keys = []
        for key, value in tuning.items():
            if not args.overwrite and key in target_attrs:
                skipped_keys.append(key)
                continue
            target_attrs[key] = value
            updated.append(key)

        if updated:
            applied += 1
        else:
            skipped += 1

        if args.json:
            print(
                json.dumps(
                    {
                        "status": "applied" if updated else "skipped",
                        "zarr": str(plan.zarr_path),
                        "updated": updated,
                        "skipped_keys": skipped_keys,
                    }
                )
            )
        else:
            if updated:
                print(f"Applied {updated} to {plan.zarr_path}")
            if skipped_keys:
                print(f"  Skipped existing keys: {skipped_keys}")

    if not args.apply:
        print(f"\nDry-run complete. Targets: {len(plans)} (missing zarrs: {missing})")
        return 0

    print("\nSummary")
    print(f"  camera_id: {camera_id}")
    print(f"  updated: {applied}")
    print(f"  unchanged: {skipped}")
    print(f"  missing zarrs: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
