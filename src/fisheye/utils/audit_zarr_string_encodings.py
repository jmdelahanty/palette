from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import zarr


Category = str


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _iter_arrays(group: zarr.Group, prefix: str = "") -> Iterable[Tuple[str, zarr.Array]]:
    try:
        array_names = sorted(list(group.array_keys()))
    except Exception:
        array_names = []
    for name in array_names:
        path = f"{prefix}/{name}" if prefix else str(name)
        yield path, group[name]

    try:
        group_names = sorted(list(group.group_keys()))
    except Exception:
        group_names = []
    for name in group_names:
        path = f"{prefix}/{name}" if prefix else str(name)
        child = group[name]
        yield from _iter_arrays(child, prefix=path)


def _classify_array(path: str, arr: zarr.Array) -> Optional[Category]:
    dtype = arr.dtype
    kind = getattr(dtype, "kind", None)
    dtype_text = str(dtype).lower()
    dtype_type = type(dtype).__name__.lower()
    dtype_repr = repr(dtype).lower()
    metadata = getattr(arr, "metadata", None)
    data_type = getattr(metadata, "data_type", None)
    data_type_text = str(data_type).lower()
    data_type_type = type(data_type).__name__.lower()
    data_type_repr = repr(data_type).lower()

    if path.endswith("reason_bytes") and kind == "u" and int(arr.ndim) == 2:
        return "reason_bytes"
    if (
        "variablelengthutf8" in dtype_text
        or "variablelengthutf8" in dtype_type
        or "variablelengthutf8" in dtype_repr
        or "vlenutf8" in dtype_text
        or "vlenutf8" in dtype_type
        or "vlenutf8" in dtype_repr
        or "utf8" in dtype_text and "length" not in dtype_text
        or "variablelengthutf8" in data_type_text
        or "variablelengthutf8" in data_type_type
        or "variablelengthutf8" in data_type_repr
        or "vlenutf8" in data_type_text
        or "vlenutf8" in data_type_type
        or "vlenutf8" in data_type_repr
        or "utf8" in data_type_text and "length" not in data_type_text
    ):
        return "vlen_utf8"
    if kind == "U":
        return "fixed_unicode"
    if kind == "S":
        return "fixed_bytes"
    if kind == "O":
        return "object"
    return None


def audit_archive(zarr_path: Path, *, zarr_use_filter: str = "any") -> Dict[str, Any]:
    root = zarr.open_group(str(zarr_path), mode="r")
    observed_use = _infer_zarr_use(root, zarr_path)
    if zarr_use_filter != "any" and observed_use != zarr_use_filter:
        return {
            "zarr_path": str(zarr_path),
            "zarr_use": observed_use,
            "filtered_zarr_use": True,
            "arrays_scanned": 0,
            "counts": {},
            "samples": {},
        }

    counts: Dict[str, int] = {
        "reason_bytes": 0,
        "vlen_utf8": 0,
        "fixed_unicode": 0,
        "fixed_bytes": 0,
        "object": 0,
    }
    samples: Dict[str, List[str]] = {k: [] for k in counts}
    arrays_scanned = 0

    for path, arr in _iter_arrays(root):
        arrays_scanned += 1
        category = _classify_array(path, arr)
        if category is None:
            continue
        counts[category] += 1
        if len(samples[category]) < 20:
            samples[category].append(path)

    return {
        "zarr_path": str(zarr_path),
        "zarr_use": observed_use,
        "filtered_zarr_use": False,
        "arrays_scanned": arrays_scanned,
        "counts": counts,
        "samples": samples,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Zarr string encodings (reason_bytes, VariableLengthUTF8, "
            "fixed-width unicode/bytes) across archives."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or .zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for Zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter archives by inferred zarr purpose.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--list-limit", type=int, default=10, help="Max sample paths per category in text mode.")
    args = parser.parse_args(argv)

    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    aggregate_counts = {
        "reason_bytes": 0,
        "vlen_utf8": 0,
        "fixed_unicode": 0,
        "fixed_bytes": 0,
        "object": 0,
    }
    archives: List[Dict[str, Any]] = []
    zarr_scanned = 0
    filtered_zarr_use = 0
    errors = 0
    arrays_scanned_total = 0
    any_zarr = False

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        zarr_scanned += 1
        try:
            report = audit_archive(zarr_path, zarr_use_filter=str(args.zarr_use))
        except Exception as exc:
            errors += 1
            archives.append(
                {
                    "zarr_path": str(zarr_path),
                    "error": str(exc),
                }
            )
            continue

        if report.get("filtered_zarr_use"):
            filtered_zarr_use += 1
            continue

        archives.append(report)
        arrays_scanned_total += int(report.get("arrays_scanned", 0))
        counts = report.get("counts", {})
        for key in aggregate_counts:
            aggregate_counts[key] += int(counts.get(key, 0))

    if not any_zarr:
        print("No zarr files found.")
        return 1

    payload = {
        "scope": args.zarr_use,
        "zarr_scanned": zarr_scanned,
        "filtered_zarr_use": filtered_zarr_use,
        "arrays_scanned": arrays_scanned_total,
        "errors": errors,
        "counts": aggregate_counts,
        "archives": archives,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Zarr string-encoding audit: "
            f"scope={args.zarr_use} zarr_scanned={zarr_scanned} "
            f"filtered_zarr_use={filtered_zarr_use} arrays_scanned={arrays_scanned_total} errors={errors}"
        )
        print(
            "Counts: "
            f"reason_bytes={aggregate_counts['reason_bytes']} "
            f"vlen_utf8={aggregate_counts['vlen_utf8']} "
            f"fixed_unicode={aggregate_counts['fixed_unicode']} "
            f"fixed_bytes={aggregate_counts['fixed_bytes']} "
            f"object={aggregate_counts['object']}"
        )
        if archives:
            print("Sample paths by category:")
            seen: Dict[str, int] = {k: 0 for k in aggregate_counts}
            for archive in archives:
                samples = archive.get("samples")
                if not isinstance(samples, dict):
                    continue
                for category in aggregate_counts:
                    if seen[category] >= args.list_limit:
                        continue
                    for rel_path in samples.get(category, []):
                        if seen[category] >= args.list_limit:
                            break
                        print(f"  [{category}] {archive['zarr_path']}::{rel_path}")
                        seen[category] += 1

    return 0 if errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
