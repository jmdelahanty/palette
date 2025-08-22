#!/usr/bin/env python3
"""Quick script to verify filtered data was saved correctly"""

import zarr
import sys

def verify_filtered_data(zarr_path):
    """Check that filtered data exists and print its structure"""
    
    root = zarr.open(zarr_path, mode='r')
    
    print("Root groups and datasets:")
    for key in root.keys():
        print(f"  - {key}")
    
    if 'filtered_runs' in root:
        print("\nFiltered runs found!")
        filtered_group = root['filtered_runs']
        
        # Show all runs
        print("Available runs:")
        for run_name in filtered_group.keys():
            print(f"  - {run_name}")
            run = filtered_group[run_name]
            if 'frames_dropped' in run.attrs:
                print(f"    Frames dropped: {run.attrs['frames_dropped']}")
            if 'threshold' in run.attrs:
                print(f"    Threshold used: {run.attrs['threshold']}")
        
        # Show latest
        if 'latest' in filtered_group.attrs:
            print(f"\nLatest run: {filtered_group.attrs['latest']}")
            
            # Load latest and compare with original
            latest_run = filtered_group[filtered_group.attrs['latest']]
            original_n_detections = root['n_detections'][:]
            filtered_n_detections = latest_run['n_detections'][:]
            
            print(f"\nComparison:")
            print(f"  Original frames with detections: {(original_n_detections > 0).sum()}")
            print(f"  Filtered frames with detections: {(filtered_n_detections > 0).sum()}")
            print(f"  Difference: {(original_n_detections > 0).sum() - (filtered_n_detections > 0).sum()}")
            
            if 'drop_mask' in latest_run:
                drop_mask = latest_run['drop_mask'][:]
                dropped_frames = [i for i, dropped in enumerate(drop_mask) if dropped]
                print(f"  Dropped frame indices: {dropped_frames}")
    else:
        print("\nNo filtered_runs group found - data may not have been saved")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python verify_filtered.py <zarr_path>")
        sys.exit(1)
    
    verify_filtered_data(sys.argv[1])