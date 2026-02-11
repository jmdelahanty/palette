from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8


# Rewrite scope is intentionally narrow/operator-safe.
DEFAULT_ALLOWLIST_PATHS: tuple[str, ...] = (
    "source_index/source_dataset_id",
    "source_index/source_zarr_path",
    "source_dataset_id",
    "source_zarr_path",
)


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.suffix == ".zarr" and (root.is_file() or root.is_dir()):
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


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


def _resolve_array(root: zarr.Group, rel_path: str) -> Optional[Tuple[zarr.Group, str, zarr.Array]]:
    parts = [part for part in str(rel_path).split("/") if part]
    if not parts:
        return None
    parent: zarr.Group = root
    for part in parts[:-1]:
        if part not in parent:
            return None
        child = parent[part]
        if not isinstance(child, zarr.Group):
            return None
        parent = child
    name = parts[-1]
    if name not in parent:
        return None
    arr = parent[name]
    if not isinstance(arr, zarr.Array):
        return None
    return parent, name, arr


def _is_fixed_unicode(arr: zarr.Array) -> bool:
    kind = getattr(arr.dtype, "kind", None)
    return kind == "U"


def _rewrite_array(parent: zarr.Group, name: str, arr: zarr.Array) -> None:
    if int(arr.ndim) != 1:
        raise ValueError(f"Only 1D arrays are supported, got shape={arr.shape}")

    attrs = dict(arr.attrs)
    labels = np.asarray(arr[:], dtype=object).reshape(-1)
    chunks = arr.chunks if arr.chunks and len(arr.chunks) == 1 else None
    if not chunks:
        chunks = (max(1, min(1024, int(labels.shape[0]) if labels.ndim > 0 else 1)),)

    rewritten = parent.create_array(
        name,
        shape=(int(labels.shape[0]),),
        dtype=VariableLengthUTF8(),
        fill_value="",
        chunks=chunks,
        overwrite=True,
    )
    rewritten[:] = labels
    rewritten.attrs.update(attrs)


def backfill_archive(
    zarr_path: Path,
    *,
    apply: bool,
    allowlist_paths: tuple[str, ...] = DEFAULT_ALLOWLIST_PATHS,
) -> Dict[str, int]:
    root = zarr.open_group(str(zarr_path), mode="a" if apply else "r")
    counts = {
        "allowlisted_paths": 0,
        "paths_missing": 0,
        "paths_found": 0,
        "rewritable": 0,
        "rewritten": 0,
        "skipped_not_fixed_unicode": 0,
        "skipped_unsupported_shape": 0,
    }
    for rel_path in allowlist_paths:
        counts["allowlisted_paths"] += 1
        resolved = _resolve_array(root, rel_path)
        if resolved is None:
            counts["paths_missing"] += 1
            continue
        parent, name, arr = resolved
        counts["paths_found"] += 1
        if not _is_fixed_unicode(arr):
            counts["skipped_not_fixed_unicode"] += 1
            continue
        if int(arr.ndim) != 1:
            counts["skipped_unsupported_shape"] += 1
            continue
        counts["rewritable"] += 1
        if apply:
            _rewrite_array(parent, name, arr)
            counts["rewritten"] += 1
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill legacy fixed-unicode string arrays on a strict allowlist "
            "to VariableLengthUTF8."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or .zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter archives by inferred zarr purpose.",
    )
    parser.add_argument(
        "--allowlist-path",
        action="append",
        default=[],
        help="Additional array path to allow for rewrite (repeatable).",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    allowlist = tuple(dict.fromkeys([*DEFAULT_ALLOWLIST_PATHS, *[str(item) for item in args.allowlist_path]]))
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]

    aggregate = {
        "allowlisted_paths": 0,
        "paths_missing": 0,
        "paths_found": 0,
        "rewritable": 0,
        "rewritten": 0,
        "skipped_not_fixed_unicode": 0,
        "skipped_unsupported_shape": 0,
        "errors": 0,
    }
    archives: List[Dict[str, Any]] = []
    zarr_scanned = 0
    filtered_zarr_use = 0
    any_zarr = False

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        zarr_scanned += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="r")
            observed_use = _infer_zarr_use(root, zarr_path)
            if args.zarr_use != "any" and observed_use != args.zarr_use:
                filtered_zarr_use += 1
                continue
            report = backfill_archive(zarr_path, apply=bool(args.apply), allowlist_paths=allowlist)
            report_payload: Dict[str, Any] = {
                "zarr_path": str(zarr_path),
                "zarr_use": observed_use,
                **report,
            }
            archives.append(report_payload)
            for key in aggregate:
                if key == "errors":
                    continue
                aggregate[key] += int(report.get(key, 0))
        except Exception as exc:
            aggregate["errors"] += 1
            archives.append({"zarr_path": str(zarr_path), "error": str(exc)})

    if not any_zarr:
        print("No zarr files found.")
        return 1

    payload = {
        "scope": args.zarr_use,
        "zarr_scanned": zarr_scanned,
        "filtered_zarr_use": filtered_zarr_use,
        "allowlist_paths": list(allowlist),
        "counts": aggregate,
        "archives": archives,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        mode = "Applied" if args.apply else "Dry run"
        print(
            "Zarr string-encoding backfill: "
            f"scope={args.zarr_use} zarr_scanned={zarr_scanned} "
            f"filtered_zarr_use={filtered_zarr_use} errors={aggregate['errors']}"
        )
        print(
            f"{mode}: rewritable={aggregate['rewritable']} rewritten={aggregate['rewritten']} "
            f"paths_found={aggregate['paths_found']} paths_missing={aggregate['paths_missing']} "
            f"skipped_not_fixed_unicode={aggregate['skipped_not_fixed_unicode']} "
            f"skipped_unsupported_shape={aggregate['skipped_unsupported_shape']}"
        )

    return 0 if aggregate["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
