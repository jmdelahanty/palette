#!/usr/bin/env python3
"""
Swimming Bout Analysis

Detects and analyzes individual swimming bouts from tracking data:
- Bout detection using speed thresholds
- Inter-bout intervals (IBI)
- Bout distance, duration, and speed
- Temporal patterns and bout statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import gaussian_kde
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Bout:
    """Data class for individual swimming bout."""
    bout_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_s: float
    distance_px: float
    mean_speed: float
    peak_speed: float
    start_time_s: float
    end_time_s: float
    positions_x: np.ndarray
    positions_y: np.ndarray
    speeds: np.ndarray


class BoutAnalyzer:
    """Analyze swimming bouts from tracking data."""
    
    def __init__(self, zarr_path: str, source: str = 'latest',
                 speed_threshold: float = 50.0,
                 min_bout_duration_s: float = 0.05,
                 min_gap_duration_s: float = 0.1,
                 verbose: bool = True):
        """
        Initialize bout analyzer.
        
        Args:
            zarr_path: Path to zarr file
            source: Data source to use
            speed_threshold: Speed threshold for bout detection (px/s)
            min_bout_duration_s: Minimum bout duration in seconds
            min_gap_duration_s: Minimum gap between bouts in seconds
            verbose: Print progress
        """
        self.zarr_path = Path(zarr_path)
        self.speed_threshold = speed_threshold
        self.min_bout_duration_s = min_bout_duration_s
        self.min_gap_duration_s = min_gap_duration_s
        self.verbose = verbose
        
        # Load tracking data
        self._load_tracking_data(source)
        
        # Detect bouts
        self.bouts = self._detect_bouts()
        
        if verbose:
            print(f"Detected {len(self.bouts)} swimming bouts")
    
    def _load_tracking_data(self, source: str):
        """Load tracking data from zarr."""
        import zarr
        root = zarr.open(str(self.zarr_path), mode='r')
        
        # Get FPS
        self.fps = root.attrs.get('fps', 60.0)
        
        # Determine data source
        if source == 'latest':
            if 'preprocessing' in root and root['preprocessing'].attrs.get('latest'):
                source_path = 'preprocessing/' + root['preprocessing'].attrs['latest']
            elif 'filtered_runs' in root and root['filtered_runs'].attrs.get('latest'):
                source_path = 'filtered_runs/' + root['filtered_runs'].attrs['latest']
            else:
                source_path = 'detect_runs/' + root['detect_runs'].attrs['latest']
        else:
            source_path = source
        
        if self.verbose:
            print(f"Loading data from: {source_path}")
        
        # Load detection data
        data_group = root[source_path]
        n_detections = data_group['n_detections'][:]
        bbox_coords = data_group['bbox_norm_coords'][:]
        
        # Get dimensions
        if 'raw_video' in root:
            self.width = root['raw_video/images_ds'].shape[2]
            self.height = root['raw_video/images_ds'].shape[1]
        else:
            self.width = 640
            self.height = 640
        
        # Extract positions frame by frame
        self.positions_x = np.full(len(n_detections), np.nan)
        self.positions_y = np.full(len(n_detections), np.nan)
        
        cumulative = np.cumsum(np.insert(n_detections, 0, 0))
        for frame_idx in range(len(n_detections)):
            if n_detections[frame_idx] > 0:
                bbox_idx = cumulative[frame_idx]
                self.positions_x[frame_idx] = bbox_coords[bbox_idx][0] * self.width
                self.positions_y[frame_idx] = bbox_coords[bbox_idx][1] * self.height
        
        self.total_frames = len(n_detections)
        self.time_axis = np.arange(self.total_frames) / self.fps
        
        # Calculate speed
        self._calculate_speed()
        
        # Get calibration if available
        self.pixel_to_mm = None
        if 'calibration' in root:
            self.pixel_to_mm = root['calibration'].attrs.get('pixel_to_mm')
    
    def _calculate_speed(self):
        """Calculate instantaneous speed from positions."""
        # Calculate frame-to-frame displacement
        dx = np.diff(self.positions_x)
        dy = np.diff(self.positions_y)
        
        # Handle gaps in tracking
        frame_gaps = np.where(np.isnan(self.positions_x[:-1]) | 
                              np.isnan(self.positions_x[1:]))[0]
        dx[frame_gaps] = np.nan
        dy[frame_gaps] = np.nan
        
        # Calculate speed
        displacement = np.sqrt(dx**2 + dy**2)
        self.speed = np.full(self.total_frames, np.nan)
        self.speed[1:] = displacement * self.fps
        
        # Smooth speed to reduce noise
        valid_mask = ~np.isnan(self.speed)
        if np.sum(valid_mask) > 5:
            self.speed[valid_mask] = savgol_filter(
                self.speed[valid_mask],
                window_length=min(5, np.sum(valid_mask)),
                polyorder=2
            )
    
    def _detect_bouts(self) -> List[Bout]:
        """Detect swimming bouts based on speed threshold."""
        bouts = []
        
        # Find periods above speed threshold
        above_threshold = self.speed > self.speed_threshold
        
        # Find transitions
        transitions = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        bout_starts = np.where(transitions == 1)[0]
        bout_ends = np.where(transitions == -1)[0]
        
        # Minimum bout duration in frames
        min_bout_frames = int(self.min_bout_duration_s * self.fps)
        min_gap_frames = int(self.min_gap_duration_s * self.fps)
        
        # Filter and merge bouts
        filtered_bouts = []
        for start, end in zip(bout_starts, bout_ends):
            duration = end - start
            if duration >= min_bout_frames:
                filtered_bouts.append((start, end))
        
        # Merge bouts separated by short gaps
        if filtered_bouts:
            merged_bouts = [filtered_bouts[0]]
            for start, end in filtered_bouts[1:]:
                last_end = merged_bouts[-1][1]
                if start - last_end < min_gap_frames:
                    # Merge with previous bout
                    merged_bouts[-1] = (merged_bouts[-1][0], end)
                else:
                    merged_bouts.append((start, end))
            filtered_bouts = merged_bouts
        
        # Create Bout objects
        for i, (start, end) in enumerate(filtered_bouts):
            # Skip if positions are missing
            pos_x = self.positions_x[start:end]
            pos_y = self.positions_y[start:end]
            speeds = self.speed[start:end]
            
            if np.all(np.isnan(pos_x)) or np.all(np.isnan(pos_y)):
                continue
            
            # Calculate bout metrics
            duration_frames = end - start
            duration_s = duration_frames / self.fps
            
            # Calculate distance traveled during bout
            valid_mask = ~(np.isnan(pos_x[:-1]) | np.isnan(pos_x[1:]))
            if np.any(valid_mask):
                dx = np.diff(pos_x)
                dy = np.diff(pos_y)
                distances = np.sqrt(dx[valid_mask]**2 + dy[valid_mask]**2)
                distance_px = np.sum(distances)
            else:
                distance_px = 0
            
            # Speed statistics
            valid_speeds = speeds[~np.isnan(speeds)]
            if len(valid_speeds) > 0:
                mean_speed = np.mean(valid_speeds)
                peak_speed = np.max(valid_speeds)
            else:
                mean_speed = 0
                peak_speed = 0
            
            bout = Bout(
                bout_id=i + 1,
                start_frame=start,
                end_frame=end,
                duration_frames=duration_frames,
                duration_s=duration_s,
                distance_px=distance_px,
                mean_speed=mean_speed,
                peak_speed=peak_speed,
                start_time_s=start / self.fps,
                end_time_s=end / self.fps,
                positions_x=pos_x,
                positions_y=pos_y,
                speeds=speeds
            )
            bouts.append(bout)
        
        return bouts
    
    def calculate_inter_bout_intervals(self) -> np.ndarray:
        """Calculate inter-bout intervals (IBI)."""
        if len(self.bouts) < 2:
            return np.array([])
        
        ibis = []
        for i in range(1, len(self.bouts)):
            ibi = self.bouts[i].start_time_s - self.bouts[i-1].end_time_s
            ibis.append(ibi)
        
        return np.array(ibis)
    
    def calculate_bout_statistics(self) -> Dict:
        """Calculate summary statistics for all bouts."""
        if not self.bouts:
            return {}
        
        durations = np.array([b.duration_s for b in self.bouts])
        distances = np.array([b.distance_px for b in self.bouts])
        mean_speeds = np.array([b.mean_speed for b in self.bouts])
        peak_speeds = np.array([b.peak_speed for b in self.bouts])
        ibis = self.calculate_inter_bout_intervals()
        
        stats = {
            'n_bouts': len(self.bouts),
            'total_active_time_s': np.sum(durations),
            'total_distance_px': np.sum(distances),
            'bout_rate_per_min': len(self.bouts) / (self.total_frames / self.fps) * 60,
            
            # Duration stats
            'duration_mean_s': np.mean(durations),
            'duration_median_s': np.median(durations),
            'duration_std_s': np.std(durations),
            'duration_min_s': np.min(durations),
            'duration_max_s': np.max(durations),
            
            # Distance stats
            'distance_mean_px': np.mean(distances),
            'distance_median_px': np.median(distances),
            'distance_std_px': np.std(distances),
            
            # Speed stats
            'mean_speed_mean': np.mean(mean_speeds),
            'peak_speed_mean': np.mean(peak_speeds),
            'peak_speed_max': np.max(peak_speeds),
            
            # IBI stats
            'ibi_mean_s': np.mean(ibis) if len(ibis) > 0 else np.nan,
            'ibi_median_s': np.median(ibis) if len(ibis) > 0 else np.nan,
            'ibi_std_s': np.std(ibis) if len(ibis) > 0 else np.nan,
            'ibi_min_s': np.min(ibis) if len(ibis) > 0 else np.nan,
            'ibi_max_s': np.max(ibis) if len(ibis) > 0 else np.nan,
        }
        
        # Add mm conversions if calibrated
        if self.pixel_to_mm:
            stats['total_distance_mm'] = stats['total_distance_px'] * self.pixel_to_mm
            stats['distance_mean_mm'] = stats['distance_mean_px'] * self.pixel_to_mm
            stats['mean_speed_mm_s'] = stats['mean_speed_mean'] * self.pixel_to_mm
        
        return stats
    
    def plot_bout_analysis(self, save_path: Optional[str] = None):
        """Create comprehensive bout analysis visualization."""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Speed trace with bout markers
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.time_axis, self.speed, 'k-', alpha=0.5, linewidth=0.5, label='Speed')
        ax1.axhline(y=self.speed_threshold, color='r', linestyle='--', 
                   alpha=0.5, label=f'Threshold ({self.speed_threshold} px/s)')
        
        # Mark bouts
        for bout in self.bouts:
            ax1.axvspan(bout.start_time_s, bout.end_time_s, 
                       alpha=0.3, color='green')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Speed (pixels/second)')
        ax1.set_title(f'Swimming Speed and Bout Detection (n={len(self.bouts)} bouts)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bout duration distribution
        ax2 = fig.add_subplot(gs[1, 0])
        durations = [b.duration_s for b in self.bouts]
        if durations:
            ax2.hist(durations, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(np.mean(durations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(durations):.2f}s')
            ax2.set_xlabel('Bout Duration (seconds)')
            ax2.set_ylabel('Count')
            ax2.set_title('Bout Duration Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Inter-bout interval distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ibis = self.calculate_inter_bout_intervals()
        if len(ibis) > 0:
            ax3.hist(ibis, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax3.axvline(np.mean(ibis), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ibis):.2f}s')
            ax3.set_xlabel('Inter-Bout Interval (seconds)')
            ax3.set_ylabel('Count')
            ax3.set_title('Inter-Bout Interval Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Distance per bout
        ax4 = fig.add_subplot(gs[1, 2])
        distances = [b.distance_px for b in self.bouts]
        if distances:
            ax4.hist(distances, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(np.mean(distances), color='red', linestyle='--',
                       label=f'Mean: {np.mean(distances):.1f}px')
            ax4.set_xlabel('Distance per Bout (pixels)')
            ax4.set_ylabel('Count')
            ax4.set_title('Bout Distance Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Bout characteristics over time
        ax5 = fig.add_subplot(gs[2, 0])
        bout_times = [b.start_time_s for b in self.bouts]
        bout_distances = [b.distance_px for b in self.bouts]
        if bout_times:
            scatter = ax5.scatter(bout_times, bout_distances, 
                                c=range(len(bout_times)), cmap='viridis',
                                s=20, alpha=0.6)
            ax5.set_xlabel('Time (seconds)')
            ax5.set_ylabel('Bout Distance (pixels)')
            ax5.set_title('Bout Distance Over Time')
            plt.colorbar(scatter, ax=ax5, label='Bout Number')
            ax5.grid(True, alpha=0.3)
        
        # 6. Duration vs Distance relationship
        ax6 = fig.add_subplot(gs[2, 1])
        if durations and distances:
            ax6.scatter(durations, distances, alpha=0.6, s=30)
            
            # Add trend line
            z = np.polyfit(durations, distances, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(durations), max(durations), 100)
            ax6.plot(x_line, p(x_line), 'r--', alpha=0.5, 
                    label=f'Trend: {z[0]:.1f}px/s')
            
            ax6.set_xlabel('Bout Duration (seconds)')
            ax6.set_ylabel('Bout Distance (pixels)')
            ax6.set_title('Duration vs Distance Relationship')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats = self.calculate_bout_statistics()
        summary_text = f"""BOUT STATISTICS
        
Total bouts: {stats['n_bouts']}
Bout rate: {stats['bout_rate_per_min']:.1f}/min
Active time: {stats['total_active_time_s']:.1f}s
Total distance: {stats['total_distance_px']:.1f}px

Duration (s):
  Mean: {stats['duration_mean_s']:.3f}
  Median: {stats['duration_median_s']:.3f}
  Range: {stats['duration_min_s']:.3f} - {stats['duration_max_s']:.3f}

IBI (s):
  Mean: {stats['ibi_mean_s']:.3f}
  Median: {stats['ibi_median_s']:.3f}
  
Speed (px/s):
  Mean: {stats['mean_speed_mean']:.1f}
  Peak: {stats['peak_speed_max']:.1f}
"""
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        fig.suptitle('Swimming Bout Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Bout analysis plot saved to: {save_path}")
        
        plt.show()
    
    def export_bout_data(self, csv_path: str):
        """Export bout data to CSV for further analysis."""
        bout_data = []
        for bout in self.bouts:
            bout_data.append({
                'bout_id': bout.bout_id,
                'start_frame': bout.start_frame,
                'end_frame': bout.end_frame,
                'start_time_s': bout.start_time_s,
                'end_time_s': bout.end_time_s,
                'duration_s': bout.duration_s,
                'distance_px': bout.distance_px,
                'mean_speed': bout.mean_speed,
                'peak_speed': bout.peak_speed
            })
        
        df = pd.DataFrame(bout_data)
        
        # Add IBI column
        ibis = [np.nan]  # First bout has no previous IBI
        for i in range(1, len(self.bouts)):
            ibi = self.bouts[i].start_time_s - self.bouts[i-1].end_time_s
            ibis.append(ibi)
        df['inter_bout_interval_s'] = ibis
        
        df.to_csv(csv_path, index=False)
        print(f"Bout data exported to: {csv_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze swimming bouts from tracking data'
    )
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--source', type=str, default='latest',
                       help='Data source to use')
    parser.add_argument('--speed-threshold', type=float, default=50.0,
                       help='Speed threshold for bout detection (px/s)')
    parser.add_argument('--min-bout-duration', type=float, default=0.05,
                       help='Minimum bout duration (seconds)')
    parser.add_argument('--min-gap', type=float, default=0.1,
                       help='Minimum gap between bouts (seconds)')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save analysis plot')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Path to save bout data CSV')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SWIMMING BOUT ANALYSIS")
    print("=" * 60)
    print(f"Zarr file: {args.zarr_path}")
    print(f"Speed threshold: {args.speed_threshold} px/s")
    print(f"Min bout duration: {args.min_bout_duration} s")
    print(f"Min gap duration: {args.min_gap} s")
    
    # Initialize analyzer
    analyzer = BoutAnalyzer(
        args.zarr_path,
        source=args.source,
        speed_threshold=args.speed_threshold,
        min_bout_duration_s=args.min_bout_duration,
        min_gap_duration_s=args.min_gap,
        verbose=True
    )
    
    # Calculate statistics
    stats = analyzer.calculate_bout_statistics()
    
    print("\n" + "=" * 60)
    print("BOUT SUMMARY")
    print("=" * 60)
    print(f"Total bouts detected: {stats['n_bouts']}")
    print(f"Bout rate: {stats['bout_rate_per_min']:.2f} bouts/minute")
    print(f"Total active time: {stats['total_active_time_s']:.1f} seconds")
    print(f"Total distance: {stats['total_distance_px']:.1f} pixels")
    
    if analyzer.pixel_to_mm:
        print(f"  ({stats['total_distance_mm']:.1f} mm)")
    
    print(f"\nBout duration: {stats['duration_mean_s']:.3f} ± {stats['duration_std_s']:.3f} s")
    print(f"Inter-bout interval: {stats['ibi_mean_s']:.3f} ± {stats['ibi_std_s']:.3f} s")
    print(f"Distance per bout: {stats['distance_mean_px']:.1f} ± {stats['distance_std_px']:.1f} pixels")
    print(f"Mean bout speed: {stats['mean_speed_mean']:.1f} px/s")
    
    # Create visualization
    analyzer.plot_bout_analysis(save_path=args.output_plot)
    
    # Export data if requested
    if args.output_csv:
        df = analyzer.export_bout_data(args.output_csv)
        print(f"\nExported {len(df)} bouts to CSV")
    
    return 0


if __name__ == '__main__':
    exit(main())