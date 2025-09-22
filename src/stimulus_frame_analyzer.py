#!/usr/bin/env python3
"""
Stimulus Frame Timing Analyzer
Analyzes stimulus frame timing from H5 files to identify frame drops, duplicates, and timing issues.
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
from datetime import datetime
import seaborn as sns
from collections import Counter

def load_stimulus_metadata(h5_path):
    """Load stimulus frame metadata from H5 file."""
    print(f"Loading stimulus metadata from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as h5f:
        # Get basic H5 attributes
        attrs = dict(h5f.attrs)
        print(f"  Session: {attrs.get('session_uuid', 'unknown')}")
        print(f"  Rig: {attrs.get('rig_id', 'unknown')}")
        print(f"  Arena: {attrs.get('arena_id', 'unknown')}")
        
        # List all available datasets for debugging
        print(f"\nAvailable datasets in H5 file:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name}: {obj.shape} {obj.dtype}")
        h5f.visititems(print_structure)
        
        # Check for frame metadata
        if '/video_metadata/frame_metadata' not in h5f:
            print("\n  WARNING: No /video_metadata/frame_metadata found in H5 file")
            print("  This file may not have stimulus frame logging enabled.")
            
            # Check if there's any video metadata at all
            if '/video_metadata' in h5f:
                print(f"  Found /video_metadata group with: {list(h5f['/video_metadata'].keys())}")
            
            # Return empty DataFrame to allow analysis to continue
            return pd.DataFrame(), attrs
        
        # Load frame metadata
        frame_metadata = h5f['/video_metadata/frame_metadata'][:]
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(frame_metadata)
        
        # Display columns and basic info
        print(f"\nFrame metadata info:")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Total records: {len(df)}")
        
        if len(df) > 0:
            print(f"  First record: {df.iloc[0].to_dict()}")
            print(f"  Last record: {df.iloc[-1].to_dict()}")
        
        return df, attrs

def analyze_stimulus_timing(df):
    """Analyze stimulus frame timing patterns."""
    
    # Check if DataFrame is empty
    if len(df) == 0:
        print("  WARNING: No frame metadata found in file!")
        stats = {
            'total_frames': 0,
            'unique_stimulus_frames': 0,
            'unique_camera_frames': 0,
            'duration_s': 0,
            'mean_interval_ms': 0,
            'median_interval_ms': 0,
            'std_interval_ms': 0,
            'min_interval_ms': 0,
            'max_interval_ms': 0,
            'expected_interval_ms': 8.333,
            'missing_frames_count': 0,
            'duplicate_frames_count': 0,
            'timing_issues_count': 0,
            'avg_stimulus_fps': 0
        }
        return stats, [], pd.Series(), pd.DataFrame(), df
    
    # Sort by stimulus frame number to ensure proper ordering
    df = df.sort_values('stimulus_frame_num').reset_index(drop=True)
    
    # Calculate inter-frame intervals
    df['timestamp_diff_ns'] = df['timestamp_ns'].diff()
    df['timestamp_diff_ms'] = df['timestamp_diff_ns'] / 1e6
    
    # Calculate expected timing (120Hz = 8.333ms per frame)
    expected_interval_ms = 1000 / 120  # 8.333ms
    
    # Analyze stimulus frame continuity
    df['stim_frame_diff'] = df['stimulus_frame_num'].diff()
    
    # Detect issues
    missing_frames = []
    duplicate_frames = []
    timing_issues = []
    
    # Check for missing stimulus frames
    if len(df) > 0:
        min_frame = int(df['stimulus_frame_num'].min())
        max_frame = int(df['stimulus_frame_num'].max())
        expected_frames = set(range(min_frame, max_frame + 1))
        actual_frames = set(df['stimulus_frame_num'].unique())
        missing_frames = sorted(expected_frames - actual_frames)
    
    # Check for duplicate stimulus frames (multiple records for same frame)
    frame_counts = df['stimulus_frame_num'].value_counts()
    duplicates = frame_counts[frame_counts > 1]
    
    # Check for timing irregularities (>20% deviation from expected)
    timing_threshold = expected_interval_ms * 0.2
    timing_issues_mask = abs(df['timestamp_diff_ms'] - expected_interval_ms) > timing_threshold
    timing_issues = df[timing_issues_mask & df['timestamp_diff_ms'].notna()]
    
    # Calculate statistics
    intervals_ms = df['timestamp_diff_ms'].dropna()
    
    stats = {
        'total_frames': len(df),
        'unique_stimulus_frames': df['stimulus_frame_num'].nunique(),
        'unique_camera_frames': df['triggering_camera_frame_id'].nunique(),
        'duration_s': (df['timestamp_ns'].max() - df['timestamp_ns'].min()) / 1e9,
        'mean_interval_ms': intervals_ms.mean(),
        'median_interval_ms': intervals_ms.median(),
        'std_interval_ms': intervals_ms.std(),
        'min_interval_ms': intervals_ms.min(),
        'max_interval_ms': intervals_ms.max(),
        'expected_interval_ms': expected_interval_ms,
        'missing_frames_count': len(missing_frames),
        'duplicate_frames_count': len(duplicates),
        'timing_issues_count': len(timing_issues),
        'avg_stimulus_fps': 1000 / intervals_ms.mean() if intervals_ms.mean() > 0 else 0
    }
    
    return stats, missing_frames, duplicates, timing_issues, df

def analyze_camera_stimulus_mapping(df):
    """Analyze the mapping between camera frames and stimulus frames."""
    
    # Expected ratio (60fps camera / 120Hz stimulus = 0.5)
    expected_ratio = 60 / 120  # 0.5
    
    # Group by camera frame
    camera_grouped = df.groupby('triggering_camera_frame_id')
    
    # How many stimulus frames per camera frame?
    stim_per_camera = camera_grouped['stimulus_frame_num'].nunique()
    
    # Group by stimulus frame
    stim_grouped = df.groupby('stimulus_frame_num')
    
    # How many camera frames per stimulus frame?
    camera_per_stim = stim_grouped['triggering_camera_frame_id'].nunique()
    
    mapping_stats = {
        'avg_stim_frames_per_camera': stim_per_camera.mean(),
        'expected_stim_frames_per_camera': 1 / expected_ratio,  # Should be ~2
        'camera_frames_with_1_stim': (stim_per_camera == 1).sum(),
        'camera_frames_with_2_stim': (stim_per_camera == 2).sum(),
        'camera_frames_with_3plus_stim': (stim_per_camera >= 3).sum(),
        'stimulus_frames_with_multiple_cameras': (camera_per_stim > 1).sum()
    }
    
    return mapping_stats, stim_per_camera

def plot_stimulus_analysis(df, stats, missing_frames, duplicates, timing_issues, output_prefix):
    """Create visualization of stimulus frame timing analysis."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle(f'Stimulus Frame Timing Analysis - {output_prefix}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of inter-frame intervals
    ax1 = fig.add_subplot(gs[0, 0])
    intervals_ms = df['timestamp_diff_ms'].dropna()
    if len(intervals_ms) > 0:
        ax1.hist(intervals_ms, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(stats['expected_interval_ms'], color='green', linestyle='--', 
                   label=f'Expected: {stats["expected_interval_ms"]:.2f}ms')
        ax1.axvline(stats['median_interval_ms'], color='red', linestyle='--',
                   label=f'Median: {stats["median_interval_ms"]:.2f}ms')
    ax1.set_xlabel('Inter-frame Interval (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title('Stimulus Frame Interval Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Stimulus frame rate over time
    ax2 = fig.add_subplot(gs[0, 1])
    # Calculate rolling frame rate
    window = 100
    df['fps_instant'] = 1000 / df['timestamp_diff_ms']
    df['fps_rolling'] = df['fps_instant'].rolling(window=window, center=True).mean()
    df['elapsed_s'] = (df['timestamp_ns'] - df['timestamp_ns'].min()) / 1e9
    
    ax2.plot(df['elapsed_s'], df['fps_rolling'], linewidth=1, alpha=0.8)
    ax2.axhline(120, color='green', linestyle='--', alpha=0.5, label='Expected 120Hz')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Stimulus Rate (Hz)')
    ax2.set_title(f'Stimulus Rate Over Time (rolling avg, window={window})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Camera-Stimulus Frame Mapping
    ax3 = fig.add_subplot(gs[0, 2])
    # Show distribution of stimulus frames per camera frame
    camera_grouped = df.groupby('triggering_camera_frame_id').size()
    counts = Counter(camera_grouped.values)
    
    bars = ax3.bar(counts.keys(), counts.values(), edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Stimulus Frames per Camera Frame')
    ax3.set_ylabel('Count')
    ax3.set_title('Camera→Stimulus Frame Mapping')
    ax3.grid(True, alpha=0.3)
    
    # Color the expected bar (should be 2 for 60fps→120Hz)
    if 2 in counts:
        bars[list(counts.keys()).index(2)].set_color('green')
    
    # 4. Stimulus frame continuity
    ax4 = fig.add_subplot(gs[1, 0])
    frame_diffs = df['stim_frame_diff'].dropna()
    diff_counts = Counter(frame_diffs)
    
    ax4.bar(diff_counts.keys(), diff_counts.values(), edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Stimulus Frame Number Difference')
    ax4.set_ylabel('Count')
    ax4.set_title('Stimulus Frame Continuity')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Timing issues over time
    ax5 = fig.add_subplot(gs[1, 1])
    if len(timing_issues) > 0:
        ax5.scatter(timing_issues['elapsed_s'], timing_issues['timestamp_diff_ms'], 
                   color='red', s=20, alpha=0.6)
    ax5.axhline(stats['expected_interval_ms'], color='green', linestyle='--', alpha=0.5)
    ax5.scatter(df['elapsed_s'], df['timestamp_diff_ms'], s=1, alpha=0.3)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Inter-frame Interval (ms)')
    ax5.set_title(f'Timing Issues ({len(timing_issues)} detected)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative frame count
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(df['elapsed_s'], df['stimulus_frame_num'], linewidth=2, label='Actual')
    
    # Expected line (perfect 120Hz)
    expected_frames = df['elapsed_s'] * 120 + df['stimulus_frame_num'].min()
    ax6.plot(df['elapsed_s'], expected_frames, 'r--', alpha=0.5, label='Expected (120Hz)')
    
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Stimulus Frame Number')
    ax6.set_title('Cumulative Stimulus Frames')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Summary text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = "Stimulus Frame Analysis Summary\n" + "="*40 + "\n"
    summary_text += f"Total Records: {stats['total_frames']:,}\n"
    summary_text += f"Unique Stimulus Frames: {stats['unique_stimulus_frames']:,}\n"
    summary_text += f"Unique Camera Frames: {stats['unique_camera_frames']:,}\n"
    summary_text += f"Duration: {stats['duration_s']:.2f}s\n"
    summary_text += f"Average Rate: {stats['avg_stimulus_fps']:.2f} Hz (expected: 120 Hz)\n"
    summary_text += f"\nTiming Statistics (ms):\n"
    summary_text += f"  Expected: {stats['expected_interval_ms']:.3f}\n"
    summary_text += f"  Median: {stats['median_interval_ms']:.3f}\n"
    summary_text += f"  Mean: {stats['mean_interval_ms']:.3f}\n"
    summary_text += f"  Std Dev: {stats['std_interval_ms']:.3f}\n"
    summary_text += f"  Range: [{stats['min_interval_ms']:.3f}, {stats['max_interval_ms']:.3f}]\n"
    summary_text += f"\nIssues Detected:\n"
    summary_text += f"  Missing Frames: {stats['missing_frames_count']}\n"
    summary_text += f"  Duplicate Frames: {stats['duplicate_frames_count']}\n"
    summary_text += f"  Timing Issues: {stats['timing_issues_count']}\n"
    
    if len(missing_frames) > 0 and len(missing_frames) <= 10:
        summary_text += f"  Missing frame numbers: {missing_frames}\n"
    elif len(missing_frames) > 10:
        summary_text += f"  Missing frames: {missing_frames[:5]} ... {missing_frames[-5:]}\n"
    
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{output_prefix}_stimulus_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved stimulus analysis plot to {output_path}")
    
    return fig

def print_summary(stats, mapping_stats, missing_frames, duplicates):
    """Print summary to console."""
    print("\n" + "="*50)
    print("STIMULUS FRAME TIMING ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nFrame Statistics:")
    print(f"  Total Records: {stats['total_frames']:,}")
    print(f"  Unique Stimulus Frames: {stats['unique_stimulus_frames']:,}")
    print(f"  Unique Camera Frames: {stats['unique_camera_frames']:,}")
    print(f"  Duration: {stats['duration_s']:.2f} seconds")
    print(f"  Average Stimulus Rate: {stats['avg_stimulus_fps']:.2f} Hz")
    
    print(f"\nTiming Analysis (ms):")
    print(f"  Expected Interval: {stats['expected_interval_ms']:.3f} (120Hz)")
    print(f"  Median Interval: {stats['median_interval_ms']:.3f}")
    print(f"  Mean Interval: {stats['mean_interval_ms']:.3f}")
    print(f"  Std Dev: {stats['std_interval_ms']:.3f}")
    print(f"  Min: {stats['min_interval_ms']:.3f}")
    print(f"  Max: {stats['max_interval_ms']:.3f}")
    
    print(f"\nCamera-Stimulus Mapping:")
    print(f"  Avg stimulus frames per camera frame: {mapping_stats['avg_stim_frames_per_camera']:.2f}")
    print(f"  Expected ratio: {mapping_stats['expected_stim_frames_per_camera']:.1f}")
    print(f"  Camera frames with 1 stimulus frame: {mapping_stats['camera_frames_with_1_stim']}")
    print(f"  Camera frames with 2 stimulus frames: {mapping_stats['camera_frames_with_2_stim']}")
    print(f"  Camera frames with 3+ stimulus frames: {mapping_stats['camera_frames_with_3plus_stim']}")
    
    print(f"\nIssues Detected:")
    print(f"  Missing Stimulus Frames: {stats['missing_frames_count']}")
    if stats['missing_frames_count'] > 0 and stats['missing_frames_count'] <= 20:
        print(f"    Missing: {missing_frames}")
    print(f"  Duplicate Stimulus Frames: {stats['duplicate_frames_count']}")
    if len(duplicates) > 0 and len(duplicates) <= 20:
        print(f"    Duplicates: {list(duplicates.index)}")
    print(f"  Timing Issues (>20% deviation): {stats['timing_issues_count']}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if stats['missing_frames_count'] == 0 and stats['duplicate_frames_count'] == 0:
        if abs(stats['avg_stimulus_fps'] - 120) < 1:
            print("  ✅ EXCELLENT - Stimulus presentation appears stable at 120Hz")
        else:
            print(f"  ⚠️ RATE ISSUE - Stimulus rate {stats['avg_stimulus_fps']:.2f}Hz deviates from expected 120Hz")
    else:
        print("  ⚠️ FRAME ISSUES - Missing or duplicate frames detected")
    
    print("\n" + "="*50)

def analyze_h5_file(h5_path):
    """Main analysis function for a single H5 file."""
    # Load data
    result = load_stimulus_metadata(h5_path)
    if result is None:
        return None
    
    df, attrs = result
    
    # Check if we have data to analyze
    if len(df) == 0:
        print("\n  ⚠️ No stimulus frame metadata to analyze in this file.")
        print("  This could mean:")
        print("    - Stimulus frame logging was not enabled")
        print("    - The recording session had issues")
        print("    - This is an incomplete or test file")
        return None
    
    # Analyze timing
    stats, missing_frames, duplicates, timing_issues, df_analyzed = analyze_stimulus_timing(df)
    
    # Analyze camera-stimulus mapping
    mapping_stats, stim_per_camera = analyze_camera_stimulus_mapping(df)
    
    # Print summary
    print_summary(stats, mapping_stats, missing_frames, duplicates)
    
    # Create visualizations
    output_prefix = Path(h5_path).stem
    plot_stimulus_analysis(df_analyzed, stats, missing_frames, duplicates, 
                          timing_issues, output_prefix)
    
    # Save detailed reports if there are issues
    if stats['missing_frames_count'] > 0:
        missing_report_path = f"{output_prefix}_missing_frames.txt"
        with open(missing_report_path, 'w') as f:
            f.write("Missing Stimulus Frame Numbers\n")
            f.write("="*30 + "\n")
            for frame_num in missing_frames:
                f.write(f"{frame_num}\n")
        print(f"Saved missing frames report to {missing_report_path}")
    
    if stats['duplicate_frames_count'] > 0:
        dup_report_path = f"{output_prefix}_duplicate_frames.csv"
        duplicates.to_csv(dup_report_path)
        print(f"Saved duplicate frames report to {dup_report_path}")
    
    return df_analyzed, stats, mapping_stats

def main():
    """Main function to run stimulus frame analysis on H5 files."""
    if len(sys.argv) < 2:
        print("Usage: python stimulus_frame_analyzer.py <file.h5> [file2.h5 ...]")
        print("\nThis script analyzes stimulus frame timing from H5 files")
        print("to detect frame drops, duplicates, and timing issues.")
        sys.exit(1)
    
    # Process each file
    results = {}
    for h5_path in sys.argv[1:]:
        if not Path(h5_path).exists():
            print(f"Error: File {h5_path} not found")
            continue
        
        print(f"\nProcessing {h5_path}...")
        try:
            result = analyze_h5_file(h5_path)
            if result:
                df, stats, mapping_stats = result
                results[h5_path] = {'df': df, 'stats': stats, 'mapping': mapping_stats}
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # If multiple files, create comparison
    if len(results) > 1:
        print("\n" + "="*50)
        print("COMPARISON ACROSS H5 FILES")
        print("="*50)
        
        comparison_data = []
        for h5_path, data in results.items():
            stats = data['stats']
            mapping = data['mapping']
            comparison_data.append({
                'File': Path(h5_path).stem,
                'Stimulus_Frames': stats['unique_stimulus_frames'],
                'Duration(s)': f"{stats['duration_s']:.2f}",
                'Avg_Hz': f"{stats['avg_stimulus_fps']:.2f}",
                'Median_Interval(ms)': f"{stats['median_interval_ms']:.3f}",
                'Missing': stats['missing_frames_count'],
                'Duplicates': stats['duplicate_frames_count'],
                'Stim/Camera': f"{mapping['avg_stim_frames_per_camera']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    plt.show()

if __name__ == "__main__":
    main()