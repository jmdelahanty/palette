#!/usr/bin/env python3
"""
Frame ID Monotonicity Checker

Checks if camera frame IDs are monotonically increasing in H5 files.
Detects jumps, resets, and other anomalies that might indicate shared memory issues.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def check_frame_monotonicity(h5_path: str, plot: bool = False, verbose: bool = True):
    """
    Check if frame IDs are monotonically increasing.
    
    Args:
        h5_path: Path to H5 file
        plot: Whether to plot frame IDs
        verbose: Print detailed output
    
    Returns:
        Dictionary with analysis results
    """
    
    print(f"\n{'='*70}")
    print("FRAME ID MONOTONICITY CHECK")
    print(f"{'='*70}")
    print(f"File: {Path(h5_path).name}")
    print()
    
    results = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Check frame metadata
        if '/video_metadata/frame_metadata' in f:
            print("📊 Analyzing frame_metadata...")
            metadata = f['/video_metadata/frame_metadata'][:]
            frame_ids = metadata['triggering_camera_frame_id']
            
            results['frame_metadata'] = analyze_sequence(
                frame_ids, 
                "triggering_camera_frame_id",
                verbose=verbose
            )
            
            # Also check stimulus frame numbers
            if 'stimulus_frame_num' in metadata.dtype.names:
                stim_frames = metadata['stimulus_frame_num']
                print("\n📊 Analyzing stimulus_frame_num...")
                results['stimulus_frames'] = analyze_sequence(
                    stim_frames,
                    "stimulus_frame_num",
                    verbose=verbose
                )
        
        # Check events
        if '/events' in f:
            print("\n📊 Analyzing events...")
            events = f['/events'][:]
            
            # Check camera_frame_id in events
            if 'camera_frame_id' in events.dtype.names:
                event_frame_ids = events['camera_frame_id']
                # Filter out zeros (events without frame IDs)
                valid_frame_ids = event_frame_ids[event_frame_ids > 0]
                
                if len(valid_frame_ids) > 0:
                    results['event_frames'] = analyze_sequence(
                        valid_frame_ids,
                        "camera_frame_id (events)",
                        verbose=verbose
                    )
                    
                    # Check specific event types
                    print("\n🎯 Checking key event frame IDs...")
                    check_event_frames(events, verbose)
        
        # Check bounding boxes
        if '/tracking_data/bounding_boxes' in f:
            print("\n📊 Analyzing bounding_boxes...")
            bboxes = f['/tracking_data/bounding_boxes'][:]
            bbox_frame_ids = bboxes['payload_frame_id']
            
            results['bbox_frames'] = analyze_sequence(
                bbox_frame_ids,
                "payload_frame_id",
                verbose=verbose
            )
    
    # Plot if requested
    if plot and results:
        plot_frame_sequences(results, h5_path)
    
    # Summary
    print_summary(results)
    
    return results

def analyze_sequence(values: np.ndarray, name: str, verbose: bool = True) -> Dict:
    """
    Analyze a sequence of values for monotonicity.
    
    Args:
        values: Array of values to check
        name: Name of the sequence
        verbose: Print detailed output
    
    Returns:
        Dictionary with analysis results
    """
    
    if len(values) == 0:
        print(f"  ⚠️  No values found for {name}")
        return {'error': 'No values'}
    
    # Calculate differences
    diffs = np.diff(values)
    
    # Find issues
    decreases = np.where(diffs < 0)[0]
    zeros = np.where(diffs == 0)[0]
    jumps = np.where(diffs > np.median(diffs[diffs > 0]) * 10)[0] if len(diffs[diffs > 0]) > 0 else []
    
    # Statistics
    is_monotonic = len(decreases) == 0
    is_strictly_monotonic = len(decreases) == 0 and len(zeros) == 0
    
    results = {
        'name': name,
        'total_values': len(values),
        'min': int(values.min()),
        'max': int(values.max()),
        'range': int(values.max() - values.min()),
        'is_monotonic': is_monotonic,
        'is_strictly_monotonic': is_strictly_monotonic,
        'decreases': len(decreases),
        'duplicates': len(zeros),
        'large_jumps': len(jumps),
        'values': values  # Store for plotting
    }
    
    if verbose:
        print(f"\n  {name}:")
        print(f"    Range: {results['min']} to {results['max']}")
        print(f"    Total values: {results['total_values']}")
        print(f"    Monotonic: {'✅ Yes' if is_monotonic else '❌ No'}")
        print(f"    Strictly monotonic: {'✅ Yes' if is_strictly_monotonic else '❌ No'}")
        
        if len(decreases) > 0:
            print(f"    ⚠️  Found {len(decreases)} DECREASES!")
            # Show first few decreases
            for i in decreases[:5]:
                print(f"       Index {i}: {values[i]} → {values[i+1]} (decrease of {values[i+1]-values[i]})")
            if len(decreases) > 5:
                print(f"       ... and {len(decreases)-5} more")
        
        if len(jumps) > 0:
            print(f"    ⚠️  Found {len(jumps)} large jumps")
            for i in jumps[:3]:
                print(f"       Index {i}: {values[i]} → {values[i+1]} (jump of {values[i+1]-values[i]})")
        
        if len(zeros) > 0:
            print(f"    ℹ️  Found {len(zeros)} duplicate values")
    
    # Store problem indices
    results['decrease_indices'] = decreases
    results['jump_indices'] = jumps
    results['duplicate_indices'] = zeros
    
    return results

def check_event_frames(events: np.ndarray, verbose: bool = True):
    """
    Check frame IDs for specific event types.
    
    Args:
        events: Event array
        verbose: Print detailed output
    """
    
    # Event types to check
    key_events = {
        0: "PROTOCOL_START",
        24: "CHASER_PRE_PERIOD_START",
        25: "CHASER_TRAINING_START",
        26: "CHASER_POST_PERIOD_START",
        27: "CHASER_CHASE_SEQUENCE_START",
        28: "CHASER_CHASE_SEQUENCE_END",
        4: "PROTOCOL_FINISH"
    }
    
    for event_id, event_name in key_events.items():
        event_mask = events['event_type_id'] == event_id
        if np.any(event_mask):
            event_subset = events[event_mask]
            if 'camera_frame_id' in events.dtype.names:
                frame_ids = event_subset['camera_frame_id']
                valid_frames = frame_ids[frame_ids > 0]
                
                if len(valid_frames) > 0:
                    if verbose:
                        print(f"\n  {event_name}:")
                        print(f"    Count: {len(valid_frames)}")
                        print(f"    Frame IDs: {valid_frames[:5].tolist()}")
                        if len(valid_frames) > 5:
                            print(f"    ... and {len(valid_frames)-5} more")
                    
                    # Check for anomalies
                    if len(valid_frames) > 1:
                        diffs = np.diff(valid_frames)
                        if np.any(diffs < 0):
                            print(f"    ⚠️  NON-MONOTONIC! Decreases found")
                            for i in np.where(diffs < 0)[0]:
                                print(f"       {valid_frames[i]} → {valid_frames[i+1]}")

def plot_frame_sequences(results: Dict, h5_path: str):
    """
    Plot frame ID sequences to visualize monotonicity issues.
    
    Args:
        results: Analysis results
        h5_path: Path to H5 file for title
    """
    
    # Count how many valid results we have
    valid_results = [r for r in results.values() if 'error' not in r]
    n_plots = len(valid_results)
    
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 4*n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, 2)
    
    for idx, (key, result) in enumerate([(k, v) for k, v in results.items() if 'error' not in v]):
        values = result['values']
        
        # Left plot: Frame IDs over index
        ax1 = axes[idx, 0]
        ax1.plot(values, 'b-', linewidth=0.5, alpha=0.7)
        
        # Mark problems
        if len(result['decrease_indices']) > 0:
            for i in result['decrease_indices']:
                ax1.axvline(x=i, color='red', alpha=0.5, linewidth=1)
                ax1.plot(i, values[i], 'ro', markersize=4)
        
        if len(result['jump_indices']) > 0:
            for i in result['jump_indices']:
                ax1.axvline(x=i, color='orange', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Frame ID')
        ax1.set_title(f"{result['name']} - Sequential View")
        ax1.grid(True, alpha=0.3)
        
        # Add text annotation for issues
        if not result['is_monotonic']:
            ax1.text(0.02, 0.98, f"⚠️ {result['decreases']} decreases found!",
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # Right plot: Differences
        ax2 = axes[idx, 1]
        diffs = np.diff(values)
        ax2.plot(diffs, 'g-', linewidth=0.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Mark negative differences
        neg_diffs = diffs < 0
        if np.any(neg_diffs):
            neg_indices = np.where(neg_diffs)[0]
            ax2.scatter(neg_indices, diffs[neg_indices], c='red', s=20, zorder=5)
        
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Frame ID Difference')
        ax2.set_title(f"{result['name']} - Differences")
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Median diff: {np.median(diffs[diffs > 0]):.0f}\n" if len(diffs[diffs > 0]) > 0 else ""
        stats_text += f"Max diff: {np.max(diffs):.0f}\n"
        if np.any(diffs < 0):
            stats_text += f"Min diff: {np.min(diffs):.0f}"
        
        ax2.text(0.98, 0.98, stats_text,
                transform=ax2.transAxes, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Frame ID Monotonicity Analysis\n{Path(h5_path).name}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_summary(results: Dict):
    """
    Print a summary of all checks.
    
    Args:
        results: Analysis results
    """
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    all_good = True
    
    for key, result in results.items():
        if 'error' in result:
            continue
        
        status = "✅" if result['is_monotonic'] else "❌"
        print(f"\n{status} {result['name']}:")
        print(f"   Range: {result['min']:,} to {result['max']:,}")
        
        if not result['is_monotonic']:
            all_good = False
            print(f"   ⚠️  {result['decreases']} decreases detected!")
            
            # Try to identify the likely cause
            if result['decreases'] == 1 and result['min'] > 10000:
                print("   💡 Possible cause: Shared memory leftover from previous experiment")
            elif result['decreases'] > 10:
                print("   💡 Possible cause: Frame drops or camera restarts")
        
        if result['large_jumps'] > 0:
            print(f"   ⚠️  {result['large_jumps']} large jumps detected")
    
    print(f"\n{'='*70}")
    if all_good:
        print("✅ All frame sequences are monotonically increasing!")
    else:
        print("⚠️  Issues detected - see details above")
        print("\nPossible solutions:")
        print("  1. Check if shared memory was cleared between experiments")
        print("  2. Look for camera disconnections or restarts")
        print("  3. Verify synchronization between camera and stimulus systems")
    print(f"{'='*70}\n")

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Check frame ID monotonicity in H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool checks if frame IDs are monotonically increasing, which they should be.
Non-monotonic frame IDs can indicate:
- Shared memory issues from previous experiments
- Camera restarts during recording
- Synchronization problems

Examples:
  %(prog)s analysis.h5
  %(prog)s analysis.h5 --plot
  %(prog)s analysis.h5 --quiet
        """
    )
    
    parser.add_argument('h5_path', help='Path to H5 file')
    parser.add_argument('--plot', action='store_true',
                       help='Plot frame ID sequences')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.h5_path).exists():
        print(f"❌ File not found: {args.h5_path}")
        return 1
    
    # Run analysis
    results = check_frame_monotonicity(
        h5_path=args.h5_path,
        plot=args.plot,
        verbose=not args.quiet
    )
    
    # Return non-zero if issues found
    for result in results.values():
        if 'is_monotonic' in result and not result['is_monotonic']:
            return 1
    
    return 0

if __name__ == '__main__':
    exit(main())