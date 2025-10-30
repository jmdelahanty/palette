#!/usr/bin/env python3
"""
Quick inspector for the /enums table inside a stimulus H5 file.

Lists every enum dataset (or a user-specified subset), prints the number of
entries, and optionally dumps the full mapping for verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import h5py


def _decode_name(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore").rstrip("\x00")
    return str(raw).rstrip("\x00")


def _load_enum_dataset(dataset: h5py.Dataset) -> Sequence[Tuple[int, str]]:
    data = dataset[:]
    ids = data["id"]
    names = data["name"]
    return [(int(enum_id), _decode_name(name)) for enum_id, name in zip(ids, names)]


def list_enum_datasets(h5: h5py.File) -> list[str]:
    enums = h5.get("enums")
    if enums is None:
        return []
    return sorted(
        name
        for name, node in enums.items()
        if isinstance(node, h5py.Dataset)
    )


def inspect_enums(
    h5_path: Path,
    datasets: Optional[Iterable[str]],
    max_rows: Optional[int],
) -> None:
    with h5py.File(h5_path, "r") as h5:
        available = list_enum_datasets(h5)
        if not available:
            print(f"{h5_path}: no /enums datasets found.")
            return

        print(f"Found {len(available)} enum datasets in {h5_path}:\n  - " + "\n  - ".join(available))

        target = list(datasets) if datasets else available
        enums_group = h5["enums"]

        for name in target:
            if name not in available:
                print(f"\n[name={name}] skipping (not present in file)")
                continue

            dataset = enums_group[name]
            entries = _load_enum_dataset(dataset)
            print(f"\n[name={name}] {len(entries)} entries")

            if not entries:
                continue

            preview = entries if max_rows is None else entries[: max_rows]
            for enum_id, enum_name in preview:
                print(f"  {enum_id:>5d} -> {enum_name}")

            if max_rows is not None and len(entries) > max_rows:
                print(f"  … ({len(entries) - max_rows} more)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect /enums datasets inside a stimulus H5 file.",
    )
    parser.add_argument("h5_path", type=Path, help="Path to the stimulus H5 file.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Limit inspection to the given enum dataset (can be repeated).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum rows to print per dataset (default: 20, use 0 for no limit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_rows is not None and args.max_rows <= 0:
        max_rows = None
    else:
        max_rows = args.max_rows

    inspect_enums(args.h5_path, args.datasets, max_rows)


if __name__ == "__main__":
    main()
