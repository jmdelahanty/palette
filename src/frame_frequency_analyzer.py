#!/usr/bin/env python3
"""
Camera Frame Frequency Analyzer
Analyzes frame timing from camera metadata CSV files to identify recording patterns and gaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats

def load_camera_metadata(filepath):
    """Load camera metadata CSV file."""
    print(f"Loading {filepath}...")
    
    # Try to read CSV - first attempt with comma separator (most common)
    try:
        df = pd.read_csv(filepath)
        print(f"Detected columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading with comma separator, trying whitespace: {e}")
        # Fallback to whitespace separator
        df = pd.read_csv(filepath, sep='\s+')
        print(f"Detected columns: {list(df.columns)}")
    
    # Strip any whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['timestamp']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns not found. Found columns: {list(df.columns)}")
        print(f"Looking for: {required_cols}")
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")
    
    # Convert timestamps from scientific notation to integers (nanoseconds)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(np.int64)
    if 'timestamp_sys' in df.columns:
        df['timestamp_sys'] = pd.to_numeric(df['timestamp_sys'], errors='coerce').astype(np.int64)
    
    # Remove any rows with invalid timestamps
    initial_len = len(df)
    df = df.dropna(subset=['timestamp'])
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows with invalid timestamps")
    
    print(f"Loaded {len(df)} frames")
    print(f"Time range: {df['timestamp'].min() / 1e9:.3f}s to {df['timestamp'].max() / 1e9:.3f}s")
    
    return df

def calculate_frame_intervals(df):
    """Calculate inter-frame intervals and frame rates."""
    # Calculate time differences between consecutive frames (in nanoseconds)
    df['interval_ns'] = df['timestamp'].diff()
    
    # Convert to milliseconds for easier interpretation
    df['interval_ms'] = df['interval_ns'] / 1e6
    
    # Calculate instantaneous FPS (avoiding division by zero)
    df['fps'] = np.where(df['interval_ns'] > 0, 1e9 / df['interval_ns'], np.nan)
    
    # Calculate elapsed time from start (in seconds)
    df['elapsed_s'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    
    return df

def calculate_statistics(df):
    """Calculate comprehensive statistics for frame intervals."""
    intervals_ms = df['interval_ms'].dropna()
    
    stats_dict = {
        'total_frames': len(df),
        'total_duration_s': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1e9,
        'mean_interval_ms': intervals_ms.mean(),
        'median_interval_ms': intervals_ms.median(),
        'std_interval_ms': intervals_ms.std(),
        'min_interval_ms': intervals_ms.min(),
        'max_interval_ms': intervals_ms.max(),
        'p25_interval_ms': intervals_ms.quantile(0.25),
        'p75_interval_ms': intervals_ms.quantile(0.75),
        'p95_interval_ms': intervals_ms.quantile(0.95),
        'p99_interval_ms': intervals_ms.quantile(0.99),
    }
    
    # Calculate average FPS
    stats_dict['avg_fps'] = len(df) / stats_dict['total_duration_s']
    
    # Calculate expected FPS based on median interval
    stats_dict['median_fps'] = 1000 / stats_dict['median_interval_ms'] if stats_dict['median_interval_ms'] > 0 else 0
    
    return stats_dict

def detect_gaps(df, threshold_multiplier=5):
    """
    Detect significant gaps in recording.
    Gaps are defined as intervals > threshold_multiplier * median interval.
    """
    median_interval = df['interval_ms'].median()
    threshold = median_interval * threshold_multiplier
    
    gaps = df[df['interval_ms'] > threshold].copy()
    
    if len(gaps) > 0:
        gaps['gap_duration_s'] = gaps['interval_ms'] / 1000
        gaps['multiplier'] = gaps['interval_ms'] / median_interval
        
        # Add human-readable timestamp
        gaps['time_at_gap'] = gaps['elapsed_s'].apply(lambda x: f"{x:.2f}s")
        
        # Return only the columns we added plus the key identifiers
        return gaps[['frame_id', 'elapsed_s', 'interval_ms', 'gap_duration_s', 'multiplier', 'time_at_gap']]
    else:
        # Return empty DataFrame with the expected columns
        return pd.DataFrame(columns=['frame_id', 'elapsed_s', 'interval_ms', 'gap_duration_s', 'multiplier', 'time_at_gap'])

def plot_analysis(df, gaps, output_prefix):
    """Create comprehensive visualization of frame timing analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Camera Frame Timing Analysis - {output_prefix}', fontsize=16, fontweight='bold')
    
    # 1. Histogram of inter-frame intervals
    ax = axes[0, 0]
    intervals_ms = df['interval_ms'].dropna()
    # Cap the histogram at 99th percentile to avoid outliers dominating
    cap_value = intervals_ms.quantile(0.99)
    intervals_capped = intervals_ms[intervals_ms <= cap_value]
    ax.hist(intervals_capped, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(intervals_ms.median(), color='red', linestyle='--', label=f'Median: {intervals_ms.median():.2f}ms')
    ax.set_xlabel('Inter-frame Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Frame Intervals (capped at P99)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Frame rate over time
    ax = axes[0, 1]
    # Calculate rolling average FPS over windows
    window_size = 100
    df['fps_rolling'] = df['fps'].rolling(window=window_size, center=True).mean()
    ax.plot(df['elapsed_s'], df['fps_rolling'], linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frame Rate (fps)')
    ax.set_title(f'Frame Rate Over Time (rolling avg, window={window_size})')
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative frame count
    ax = axes[0, 2]
    ax.plot(df['elapsed_s'], df.index, linewidth=2)
    expected_frames = df['elapsed_s'] * df['fps'].median()
    ax.plot(df['elapsed_s'], expected_frames, 'r--', alpha=0.5, label='Expected (constant FPS)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frame Count')
    ax.set_title('Cumulative Frame Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Inter-frame intervals over time
    ax = axes[1, 0]
    ax.scatter(df['elapsed_s'], df['interval_ms'], s=1, alpha=0.5)
    if len(gaps) > 0:
        ax.scatter(gaps['elapsed_s'], gaps['interval_ms'], color='red', s=20, 
                  label=f'Gaps (n={len(gaps)})', zorder=5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Inter-frame Interval (ms)')
    ax.set_title('Frame Intervals Over Time')
    ax.set_yscale('log')
    if len(gaps) > 0:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Box plot of intervals by time segments
    ax = axes[1, 1]
    # Divide recording into segments
    n_segments = min(10, len(df) // 1000)  # Max 10 segments, min 1000 frames per segment
    if n_segments > 1:
        df['segment'] = pd.cut(df['elapsed_s'], bins=n_segments, labels=[f'S{i+1}' for i in range(n_segments)])
        segment_data = [group['interval_ms'].dropna() for _, group in df.groupby('segment')]
        bp = ax.boxplot(segment_data, labels=[f'S{i+1}' for i in range(n_segments)])
        ax.set_xlabel('Time Segment')
        ax.set_ylabel('Inter-frame Interval (ms)')
        ax.set_title('Interval Distribution by Time Segment')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data for segmentation', 
               ha='center', va='center', transform=ax.transAxes)
    
    # 6. Gap analysis summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create summary text
    summary_text = "Recording Summary\n" + "="*30 + "\n"
    stats = calculate_statistics(df)
    
    summary_text += f"Total Frames: {stats['total_frames']:,}\n"
    summary_text += f"Duration: {stats['total_duration_s']:.2f}s\n"
    summary_text += f"Avg FPS: {stats['avg_fps']:.2f}\n"
    summary_text += f"Median FPS: {stats['median_fps']:.2f}\n"
    summary_text += f"\nInterval Statistics (ms):\n"
    summary_text += f"  Median: {stats['median_interval_ms']:.2f}\n"
    summary_text += f"  Mean: {stats['mean_interval_ms']:.2f}\n"
    summary_text += f"  Std Dev: {stats['std_interval_ms']:.2f}\n"
    summary_text += f"  Range: [{stats['min_interval_ms']:.2f}, {stats['max_interval_ms']:.2f}]\n"
    summary_text += f"  P95: {stats['p95_interval_ms']:.2f}\n"
    summary_text += f"  P99: {stats['p99_interval_ms']:.2f}\n"
    
    if len(gaps) > 0:
        summary_text += f"\nGaps Detected: {len(gaps)}\n"
        summary_text += f"Total Gap Time: {gaps['gap_duration_s'].sum():.2f}s\n"
        summary_text += f"Largest Gap: {gaps['gap_duration_s'].max():.2f}s\n"
    else:
        summary_text += f"\nNo significant gaps detected\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
           fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{output_prefix}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis plot to {output_path}")
    
    return fig

def print_summary(stats, gaps):
    """Print summary statistics to console."""
    print("\n" + "="*50)
    print("FRAME TIMING ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nRecording Statistics:")
    print(f"  Total Frames: {stats['total_frames']:,}")
    print(f"  Total Duration: {stats['total_duration_s']:.2f} seconds")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    print(f"  Median-based FPS: {stats['median_fps']:.2f}")
    
    print(f"\nInter-frame Intervals (ms):")
    print(f"  Median: {stats['median_interval_ms']:.3f}")
    print(f"  Mean: {stats['mean_interval_ms']:.3f}")
    print(f"  Std Dev: {stats['std_interval_ms']:.3f}")
    print(f"  Min: {stats['min_interval_ms']:.3f}")
    print(f"  Max: {stats['max_interval_ms']:.3f}")
    print(f"  P25-P75: [{stats['p25_interval_ms']:.3f}, {stats['p75_interval_ms']:.3f}]")
    print(f"  P95: {stats['p95_interval_ms']:.3f}")
    print(f"  P99: {stats['p99_interval_ms']:.3f}")
    
    if len(gaps) > 0:
        print(f"\nGap Analysis (gaps > 5x median):")
        print(f"  Number of gaps: {len(gaps)}")
        print(f"  Total gap duration: {gaps['gap_duration_s'].sum():.2f} seconds")
        print(f"  Largest gap: {gaps['gap_duration_s'].max():.2f} seconds")
        print(f"\n  Top 5 gaps:")
        print(gaps.nlargest(5, 'interval_ms').to_string(index=False))
    else:
        print(f"\nNo significant gaps detected - recording appears continuous")
    
    print("\n" + "="*50)

def analyze_camera_file(filepath):
    """Main analysis function for a single camera file."""
    # Load data
    df = load_camera_metadata(filepath)
    
    # Calculate intervals and frame rates
    df = calculate_frame_intervals(df)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Detect gaps
    gaps = detect_gaps(df)
    
    # Print summary
    print_summary(stats, gaps)
    
    # Create visualizations
    output_prefix = Path(filepath).stem
    plot_analysis(df, gaps, output_prefix)
    
    # Save detailed gap report if gaps exist
    if len(gaps) > 0:
        gap_report_path = f"{output_prefix}_gaps.csv"
        gaps.to_csv(gap_report_path, index=False)
        print(f"Saved gap report to {gap_report_path}")
    
    return df, stats, gaps

def main():
    """Main function to run analysis on camera metadata files."""
    if len(sys.argv) < 2:
        print("Usage: python frame_frequency_analyzer.py <camera_meta.csv> [camera_meta2.csv ...]")
        print("\nThis script analyzes camera metadata files to determine frame frequencies")
        print("and identify gaps that might indicate recording stops/starts.")
        sys.exit(1)
    
    # Process each file
    results = {}
    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            print(f"Error: File {filepath} not found")
            continue
        
        print(f"\nProcessing {filepath}...")
        try:
            df, stats, gaps = analyze_camera_file(filepath)
            results[filepath] = {'df': df, 'stats': stats, 'gaps': gaps}
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # If multiple files, create comparison
    if len(results) > 1:
        print("\n" + "="*50)
        print("COMPARISON ACROSS CAMERAS")
        print("="*50)
        
        comparison_data = []
        for filepath, data in results.items():
            stats = data['stats']
            comparison_data.append({
                'Camera': Path(filepath).stem,
                'Frames': stats['total_frames'],
                'Duration(s)': f"{stats['total_duration_s']:.2f}",
                'Avg_FPS': f"{stats['avg_fps']:.2f}",
                'Median_Interval(ms)': f"{stats['median_interval_ms']:.3f}",
                'Gaps': len(data['gaps'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    plt.show()

if __name__ == "__main__":
    main()