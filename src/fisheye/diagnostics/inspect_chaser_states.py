#!/usr/bin/env python3
import argparse
from pathlib import Path
import zarr

from fisheye.shared.zarr_helpers import resolve_zarr_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", type=Path, help="Path to the Zarr archive")
    parser.add_argument(
        "--stimulus-run",
        default=None,
        help="Name under analysis/stimulus_runs/ (defaults to the latest run)",
    )
    args = parser.parse_args()

    root = zarr.open(str(args.zarr_path), mode="r")

    run_group, run_name = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        args.stimulus_run,
        run_label="Stimulus run",
    )
    chaser_states = run_group["tracking_data"]["chaser_states"]
    print(f"Run: {run_name}")
    print("Fields:")
    preview_rows = min(10, chaser_states.shape[0])
    preview = chaser_states[:preview_rows]
    for name in chaser_states.dtype.names:
        sample = preview[name]
        print(f"  {name:>24s}  dtype={sample.dtype}  values={sample}")

if __name__ == "__main__":
    main()
