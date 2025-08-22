#!/usr/bin/env python3
"""
Analyze camera hardware tick frequency from timestamp data.
Determines the GevTimestampTickFrequency parameter value.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_camera_tick_frequency(csv_path):
    """
    Determine the camera's tick frequency by comparing hardware timestamps 
    with system timestamps.
    """
    print(f"\n{'='*60}")
    print("CAMERA TICK FREQUENCY ANALYSIS")
    print(f"{'='*60}\n")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} frames from {csv_path}")
    
    if 'timestamp' not in df.columns or 'timestamp_sys' not in df.columns:
        print("Error: Required columns 'timestamp' and 'timestamp_sys' not found")
        return
    
    # Get the data
    hw_ticks = df['timestamp'].values
    sys_ns = df['timestamp_sys'].values
    
    # Calculate time spans
    hw_span = hw_ticks[-1] - hw_ticks[0]
    sys_span_ns = sys_ns[-1] - sys_ns[0]
    sys_span_s = sys_span_ns / 1e9
    
    print(f"Recording duration (system time): {sys_span_s:.2f} seconds")
    print(f"Hardware tick span: {hw_span:,} ticks")
    print(f"Hardware tick range: {hw_ticks[0]:,} to {hw_ticks[-1]:,}")
    
    # Calculate tick frequency
    tick_frequency = hw_span / sys_span_s
    
    print(f"\n{'='*60}")
    print(f"CALCULATED TICK FREQUENCY: {tick_frequency:,.0f} Hz")
    print(f"                          ({tick_frequency/1e9:.3f} GHz)")
    print(f"{'='*60}")
    
    # Common camera tick frequencies to check
    common_frequencies = {
        '100 MHz': 100e6,
        '125 MHz': 125e6,
        '1 GHz': 1e9,
        '2.5 GHz': 2.5e9,
        '10 GHz': 10e9,
    }
    
    print("\nComparison with common frequencies:")
    for name, freq in common_frequencies.items():
        ratio = tick_frequency / freq
        if 0.95 < ratio < 1.05:  # Within 5%
            print(f"  ✓ {name}: ratio = {ratio:.4f} (MATCH!)")
        else:
            print(f"    {name}: ratio = {ratio:.4f}")
    
    # Verify by converting hardware timestamps to seconds
    print("\n--- Verification ---")
    hw_seconds = (hw_ticks - hw_ticks[0]) / tick_frequency
    sys_seconds = (sys_ns - sys_ns[0]) / 1e9
    
    # Compare frame intervals
    hw_intervals = np.diff(hw_seconds) * 1000  # to ms
    sys_intervals = np.diff(sys_seconds) * 1000  # to ms
    
    print(f"Average frame interval (hardware): {np.mean(hw_intervals):.2f} ms")
    print(f"Average frame interval (system):   {np.mean(sys_intervals):.2f} ms")
    
    # Calculate drift
    time_diff = hw_seconds - sys_seconds
    drift_rate = (time_diff[-1] - time_diff[0]) / len(time_diff)
    print(f"\nClock drift: {drift_rate*1000:.6f} ms/frame")
    print(f"Total drift over recording: {(time_diff[-1] - time_diff[0])*1000:.2f} ms")
    
    # Check for timestamp wrapping
    hw_diffs = np.diff(hw_ticks)
    negative_jumps = hw_diffs < 0
    if np.any(negative_jumps):
        print(f"\n⚠️  WARNING: Detected {np.sum(negative_jumps)} timestamp wraparounds!")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Hardware ticks vs system time
    ax = axes[0, 0]
    ax.plot(sys_seconds, hw_ticks - hw_ticks[0], 'b-', alpha=0.7)
    ax.plot(sys_seconds, sys_seconds * tick_frequency, 'r--', 
            label=f'Expected @ {tick_frequency/1e9:.3f} GHz')
    ax.set_xlabel('System Time (seconds)')
    ax.set_ylabel('Hardware Ticks (relative)')
    ax.set_title('Hardware Ticks vs System Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Clock drift
    ax = axes[0, 1]
    ax.plot(df.index, time_diff * 1000, 'g-', alpha=0.7)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Time Difference (ms)')
    ax.set_title('Clock Drift (Hardware - System)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Frame intervals comparison
    ax = axes[1, 0]
    ax.hist(hw_intervals, bins=50, alpha=0.5, label='Hardware', color='blue')
    ax.hist(sys_intervals, bins=50, alpha=0.5, label='System', color='red')
    ax.set_xlabel('Frame Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('Frame Interval Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Tick frequency stability
    ax = axes[1, 1]
    # Calculate instantaneous frequency for each frame
    instant_freq = hw_diffs / np.diff(sys_ns) * 1e9  # Hz
    valid = instant_freq[(instant_freq > 0) & (instant_freq < tick_frequency * 2)]
    ax.plot(valid, 'b.', markersize=1, alpha=0.5)
    ax.axhline(y=tick_frequency, color='r', linestyle='--', 
               label=f'Mean: {tick_frequency/1e9:.3f} GHz')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Instantaneous Frequency (Hz)')
    ax.set_title('Tick Frequency Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Camera Tick Frequency Analysis: {Path(csv_path).name}', fontsize=14)
    plt.tight_layout()
    
    # Generate conversion functions
    print("\n--- Python Conversion Functions ---")
    print(f"""
# To convert camera timestamp to seconds:
def camera_ticks_to_seconds(ticks, ref_tick=0):
    return (ticks - ref_tick) / {tick_frequency:.0f}

# To convert camera timestamp to system time (nanoseconds):
def camera_ticks_to_system_ns(ticks, ref_tick, ref_sys_ns):
    seconds_elapsed = (ticks - ref_tick) / {tick_frequency:.0f}
    return ref_sys_ns + int(seconds_elapsed * 1e9)
    
# Example usage with your data:
tick_freq = {tick_frequency:.0f}  # Hz
first_tick = {hw_ticks[0]}
first_sys_ns = {sys_ns[0]}
""")
    
    return tick_frequency, fig

def main():
    parser = argparse.ArgumentParser(
        description='Analyze camera hardware tick frequency'
    )
    parser.add_argument('csv_file', help='Camera metadata CSV file')
    parser.add_argument('--save-plot', help='Save analysis plot')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return
    
    tick_freq, fig = analyze_camera_tick_frequency(args.csv_file)
    
    if args.save_plot and fig:
        fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {args.save_plot}")
    
    plt.show()

if __name__ == '__main__':
    main()