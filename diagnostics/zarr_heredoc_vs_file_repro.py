#!/usr/bin/env python3
"""Compare sync zarr.open_group behavior when run from a file script."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True, help="Path to zarr root")
    parser.add_argument(
        "--mode",
        default="r",
        choices=["r", "r+", "a", "w", "w-"],
        help="open_group mode",
    )
    args = parser.parse_args()

    print("file_script: before import", flush=True)
    import zarr

    print("file_script: before open", flush=True)
    root = zarr.open_group(args.zarr_path, mode=args.mode)
    print("file_script: after open", flush=True)
    print("file_script: keys_count", len(list(root.keys())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
