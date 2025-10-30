#!/usr/bin/env python3
import argparse
from pathlib import Path
import zarr

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

    stim_parent = root["analysis"]["stimulus_runs"]
    run_name = args.stimulus_run or stim_parent.attrs.get("latest")
    if run_name not in stim_parent:
        raise ValueError(f"Stimulus run '{run_name}' not found; available: {list(stim_parent.array_keys())}")

    chaser_states = stim_parent[run_name]["tracking_data"]["chaser_states"]
    print(f"Run: {run_name}")
    print("Fields:")
    preview_rows = min(10, chaser_states.shape[0])
    preview = chaser_states[:preview_rows]
    for name in chaser_states.dtype.names:
        sample = preview[name]
        print(f"  {name:>24s}  dtype={sample.dtype}  values={sample}")

if __name__ == "__main__":
    main()
