import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py


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
            yield from path.rglob("*.h5")
            yield from path.rglob("*.hdf5")
        else:
            yield from path.glob("*.h5")
            yield from path.glob("*.hdf5")


def _dataset_info(group: Optional[h5py.Group], name: str) -> Dict[str, object]:
    if group is None or name not in group:
        return {"present": False}
    ds = group[name]
    shape = ds.shape
    size = ds.size
    nonempty = bool(size) and (shape[0] > 0 if shape else True)
    return {
        "present": True,
        "shape": list(shape),
        "size": int(size),
        "nonempty": bool(nonempty),
        "dtype": str(ds.dtype),
    }


def _inspect_h5(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as hf:
        tracking = hf.get("tracking_data")
        info = {
            "path": str(path),
            "tracking_data": bool(tracking is not None),
        }
        info["chaser_states"] = _dataset_info(tracking, "chaser_states")
        info["bounding_boxes"] = _dataset_info(tracking, "bounding_boxes")
    return info


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check H5 files for tracking datasets (chaser_states, bounding_boxes)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="H5 file(s) or directories to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for H5 files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines for each H5 file.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)

    total = 0
    has_tracking = 0
    chaser_present = 0
    chaser_nonempty = 0
    bboxes_present = 0
    bboxes_nonempty = 0

    for h5_path in _iter_h5(roots, args.recursive):
        total += 1
        info = _inspect_h5(h5_path)
        if info["tracking_data"]:
            has_tracking += 1
        chaser = info["chaser_states"]
        bboxes = info["bounding_boxes"]
        if chaser.get("present"):
            chaser_present += 1
        if chaser.get("nonempty"):
            chaser_nonempty += 1
        if bboxes.get("present"):
            bboxes_present += 1
        if bboxes.get("nonempty"):
            bboxes_nonempty += 1

        if args.json:
            print(json.dumps(info, sort_keys=True))
        else:
            print(f"{h5_path.name}")
            print(f"  tracking_data: {info['tracking_data']}")
            print(
                "  chaser_states: "
                f"present={chaser.get('present', False)} "
                f"nonempty={chaser.get('nonempty', False)} "
                f"shape={chaser.get('shape')}"
            )
            print(
                "  bounding_boxes: "
                f"present={bboxes.get('present', False)} "
                f"nonempty={bboxes.get('nonempty', False)} "
                f"shape={bboxes.get('shape')}"
            )

    if total == 0:
        print("No H5 files found.")
        return 1

    if not args.json:
        print("\nSummary")
        print(f"  files: {total}")
        print(f"  tracking_data present: {has_tracking}")
        print(f"  chaser_states present: {chaser_present} (nonempty: {chaser_nonempty})")
        print(f"  bounding_boxes present: {bboxes_present} (nonempty: {bboxes_nonempty})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
