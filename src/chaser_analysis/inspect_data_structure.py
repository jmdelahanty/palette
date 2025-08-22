#!/usr/bin/env python3
"""
Script to inspect Zarr and H5 file structures to understand the data format
"""

import zarr
import h5py
import numpy as np
import sys
from pathlib import Path

def inspect_zarr(zarr_path):
    """Inspect a Zarr file structure, including nested groups."""
    print(f"\n{'='*60}")
    print(f"ZARR FILE: {zarr_path}")
    print(f"{'='*60}")

    try:
        store = zarr.open(zarr_path, mode='r')

        # Helper function to recursively print the contents of a group
        def print_group_contents(group, indent=""):
            for key in group.keys():
                item = group[key]
                if isinstance(item, zarr.Array):
                    print(f"{indent}  {key}: Array - shape={item.shape}, dtype={item.dtype}, chunks={item.chunks}")
                    if item.shape[0] > 0:
                        # Show a small sample of the data to understand its format
                        sample_size = min(2, item.shape[0])
                        print(f"{indent}    Sample data: {item[:sample_size]}")
                elif isinstance(item, zarr.Group):
                    print(f"{indent}  {key}: Group")
                    # Recurse into the subgroup
                    print_group_contents(item, indent + "    ")

        print("\nFile Contents:")
        print_group_contents(store)

    except Exception as e:
        print(f"Error reading Zarr file: {e}")

def inspect_h5_bounding_boxes(h5_path):
    """Inspect H5 file bounding box structure"""
    print(f"\n{'='*60}")
    print(f"H5 FILE: {h5_path}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            def find_datasets_with_keyword(group, keyword, path=""):
                results = []
                for key in group.keys():
                    item = group[key]
                    current_path = f"{path}/{key}"
                    if keyword.lower() in key.lower():
                        if isinstance(item, h5py.Dataset):
                            results.append((current_path, item))
                    if isinstance(item, h5py.Group):
                        results.extend(find_datasets_with_keyword(item, keyword, current_path))
                return results
            
            print("\nSearching for datasets containing 'box':")
            box_datasets = find_datasets_with_keyword(f, "box")
            
            for path, dataset in box_datasets:
                print(f"\nFound: {path}")
                if isinstance(dataset, h5py.Dataset):
                    print(f"  Shape: {dataset.shape}")
                    print(f"  Dtype: {dataset.dtype}")
                    if dataset.dtype.names:
                        print(f"  Field names: {list(dataset.dtype.names)}")
                        if dataset.shape[0] > 0:
                            first = dataset[0]
                            print(f"\n  First record:")
                            for field in dataset.dtype.names:
                                value = first[field]
                                print(f"    {field}: {value}")
                                
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def main():
    # --- Update these paths to your actual file locations ---
    zarr_path = "/home/delahantyj@hhmi.org/Desktop/escape_3/2025-08-12T20-25-51Z_arena_4_chaser_detections.zarr"
    h5_path = "/home/delahantyj@hhmi.org/Desktop/escape_3/2025-08-12T20-25-51Z_arena_4_chaser_analysis.h5"
    
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    if len(sys.argv) > 2:
        h5_path = sys.argv[2]
    
    if Path(zarr_path).exists():
        inspect_zarr(zarr_path)
    else:
        print(f"Zarr file not found: {zarr_path}")
    
    if Path(h5_path).exists():
        inspect_h5_bounding_boxes(h5_path)
    else:
        print(f"H5 file not found: {h5_path}")

if __name__ == "__main__":
    main()