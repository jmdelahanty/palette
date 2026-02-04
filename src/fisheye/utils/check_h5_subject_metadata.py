#!/usr/bin/env python3
"""
Scan H5 files for subject metadata fields (fish counts, dish counts, etc.).

Usage:
  python src/fisheye/utils/check_h5_subject_metadata.py /path/to/recordings --recursive
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py


KEY_PATTERN = re.compile(r"(fish|dish|subject|num|count)", re.IGNORECASE)


def _decode(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return value


def _find_attrs(attrs: h5py.AttributeManager) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for key, value in attrs.items():
        if KEY_PATTERN.search(str(key)):
            results[str(key)] = _decode(value)
    return results


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


def _check_h5(path: Path) -> Dict[str, object]:
    result: Dict[str, object] = {
        "path": str(path),
        "root_attrs": {},
        "subject_metadata_attrs": {},
        "subject_metadata_present": False,
    }
    try:
        with h5py.File(path, "r") as h5:
            result["root_attrs"] = _find_attrs(h5.attrs)
            if "/subject_metadata" in h5:
                result["subject_metadata_present"] = True
                result["subject_metadata_attrs"] = _find_attrs(h5["/subject_metadata"].attrs)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan H5 files for subject metadata fields (fish/dish counts).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="H5 file(s) or directories to scan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for H5 files under each path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines.",
    )
    args = parser.parse_args(argv)

    if not args.paths:
        print("No paths provided.")
        return 2

    total = 0
    with_subject = 0
    with_hits = 0

    for h5_path in _iter_h5(args.paths, args.recursive):
        total += 1
        info = _check_h5(h5_path)
        if info.get("subject_metadata_present"):
            with_subject += 1
        has_hits = bool(info.get("root_attrs")) or bool(info.get("subject_metadata_attrs"))
        if has_hits:
            with_hits += 1

        if args.json:
            print(json.dumps(info, sort_keys=True))
        else:
            print(h5_path.name)
            print(f"  subject_metadata: {info.get('subject_metadata_present')}")
            if info.get("error"):
                print(f"  error: {info['error']}")
                continue
            root_attrs = info.get("root_attrs") or {}
            subject_attrs = info.get("subject_metadata_attrs") or {}
            print(f"  root attrs matches: {root_attrs or 'none'}")
            print(f"  subject_metadata matches: {subject_attrs or 'none'}")

    print("\nSummary")
    print(f"  files: {total}")
    print(f"  subject_metadata present: {with_subject}")
    print(f"  files with matching keys: {with_hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
