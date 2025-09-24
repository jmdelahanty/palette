#!/usr/bin/env python3
"""
Enhanced Swimming Bout Analyzer with Calibration Support

Detects and analyzes swimming bouts with full calibration support:
- Automatic unit conversion (pixels to mm/cm/body lengths)
- Dual unit display throughout analysis
- Calibration-aware statistics
- Enhanced visualizations with real-world units
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import gaussian_kde
import zarr
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field


@dataclass
class CalibrationData:
    """Calibration information for unit conversion."""
    pixel_to_mm: float
    pixels_per_mm: float
    arena_diameter_mm: Optional[float] = None
    fish_length_mm: float = 4.0  # Default larval zebrafish
    camera_info: Dict = field(default_factory=dict)
    
    @property
    def pixel_to_cm(self) -> float:
        return self.pixel_to_mm / 10.0
    
    @property
    def pixel_to_body_length(self) -> float:
        return self.pixel_to_mm / self.fish_length_mm


@dataclass
class BoutWithUnits:
    """Enhanced bout data with multi-unit support."""
    # Required fields (no defaults)
    bout_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_s: float
    distance_px: float
    mean_speed_px_s: float
    peak_speed_px_s: float
    start_time_s: float
    end_time_s: float
    
    # Raw data arrays (with field configuration)
    positions_x: np.ndarray = field(repr=False)
    positions_y: np.ndarray = field(repr=False)
    speeds_px_s: np.ndarray = field(repr=False)
    
    # Optional calibrated measurements (with defaults)
    distance_mm: Optional[float] = None
    distance_bl: Optional[float] = None  # Body lengths
    mean_speed_mm_s: Optional[float] = None
    mean_speed_bl_s: Optional[float] = None
    peak_speed_mm_s: Optional[float] = None
    peak_speed_bl_s: Optional[float] = None


class EnhancedBoutAnalyzer:
    """Analyze swimming bouts with full calibration support."""
    
    def __init__(
        self,
        zarr_path: str,
        source: str = 'latest',
        speed_threshold_px_s: Optional[float] = None,
        speed_threshold_mm_s: Optional[float] = None,
        speed_threshold_bl_s: Optional[float] = None,
        min_bout_duration_s: float = 0.05,
        min_gap_duration_s: float = 0.1,
        display_units: Literal['auto', 'pixels', 'mm', 'cm', 'bl'] = 'auto',
        verbose: bool = True
    ):
        """
        Initialize enhanced bout analyzer with calibration support.
        
        Args:
            zarr_path: Path to zarr file
            source: Data source to use
            speed_threshold_px_s: Speed threshold in pixels/second
            speed_threshold_mm_s: Speed threshold in mm/second
            speed_threshold_bl_s: Speed threshold in body lengths/second
            min_bout_duration_s: Minimum bout duration
            min_gap_duration_s: Minimum gap between bouts
            display_units: Preferred display units
            verbose: Print progress
        """
        self.zarr_path = Path(zarr_path)
        self.min_bout_duration_s = min_bout_duration_s
        self.min_gap_duration_s = min_gap_duration_s
        self.verbose = verbose
        
        # Load zarr data
        self.root = zarr.open(str(self.zarr_path), mode='r')
        
        # Load calibration
        self.calibration = self._load_calibration()
        
        # Determine display units
        if display_units == 'auto':
            self.display_units = 'mm' if self.calibration else 'pixels'
        else:
            self.display_units = display_units
            
        # Set speed threshold with unit conversion
        self._set_speed_threshold(speed_threshold_px_s, speed_threshold_mm_s, speed_threshold_bl_s)
        
        # Load tracking data
        self._load_tracking_data(source)
        
        # Detect bouts
        self.bouts = self._detect_bouts()
        
        if verbose:
            self._print_summary()
    
    def _load_calibration(self) -> Optional[CalibrationData]:
        """Load calibration data from zarr."""
        if 'calibration' not in self.root:
            if self.verbose:
                print("No calibration found - measurements will be in pixels only")
            return None
        
        calib_group = self.root['calibration']
        pixel_to_mm = calib_group.attrs.get('pixel_to_mm')
        
        if not pixel_to_mm:
            return None
        
        calibration = CalibrationData(
            pixel_to_mm=pixel_to_mm,
            pixels_per_mm=1.0 / pixel_to_mm,
            arena_diameter_mm=calib_group.attrs.get('arena_diameter_mm'),
            fish_length_mm=calib_group.attrs.get('fish_length_mm', 4.0)
        )
        
        # Load camera info if available
        if 'camera_info' in calib_group.attrs:
            calibration.camera_info = dict(calib_group.attrs['camera_info'])
        
        if self.verbose:
            print(f"Calibration loaded: 1 pixel = {pixel_to_mm:.4f} mm")
            print(f"Fish length: {calibration.fish_length_mm} mm")
            if calibration.arena_diameter_mm:
                print(f"Arena diameter: {calibration.arena_diameter_mm:.1f} mm")
        
        return calibration
    
    def _set_speed_threshold(
        self,
        px_s: Optional[float],
        mm_s: Optional[float],
        bl_s: Optional[float]
    ):
        """Set speed threshold with proper unit conversion."""
        if px_s is not None:
            self.speed_threshold_px = px_s
        elif mm_s is not None and self.calibration:
            self.speed_threshold_px = mm_s / self.calibration.pixel_to_mm
        elif bl_s is not None and self.calibration:
            self.speed_threshold_px = (bl_s * self.calibration.fish_length_mm) / self.calibration.pixel_to_mm
        else:
            # Default threshold
            self.speed_threshold_px = 50.0  # pixels/second
        
        if self.verbose:
            print(f"\nSpeed threshold: {self.speed_threshold_px:.1f} px/s", end='')
            if self.calibration:
                mm_s = self.speed_threshold_px * self.calibration.pixel_to_mm
                bl_s = mm_s / self.calibration.fish_length_mm
                print(f" ({mm_s:.2f} mm/s, {bl_s:.2f} BL/s)")
            else:
                print()
    
    def _load_tracking_data(self, source: str):
        """Load tracking data from zarr."""
        # Store source info
        self.source_info = {'name': source, 'type': 'original'}
        
        # Determine which data to load
        if source == 'latest':
            # Try different preprocessing stages in order of preference
            if 'preprocessing' in self.root:
                if 'interpolated_runs' in self.root['preprocessing']:
                    runs = self.root['preprocessing']['interpolated_runs']
                    run_names = sorted([name for name in runs.keys() if name.startswith('run_')])
                    if run_names:
                        data_group = self.root['preprocessing']['interpolated_runs'][run_names[-1]]
                        self.source_info = {'name': f"interpolated/{run_names[-1]}", 'type': 'interpolated'}
                    else:
                        data_group = self.root
                elif 'latest' in self.root['preprocessing'].attrs:
                    latest = self.root['preprocessing'].attrs['latest']
                    data_group = self.root['preprocessing'][latest]
                    self.source_info = {'name': f"preprocessing/{latest}", 'type': 'interpolated'}
                else:
                    data_group = self.root
            elif 'filtered_runs' in self.root:
                if 'latest' in self.root['filtered_runs'].attrs:
                    latest = self.root['filtered_runs'].attrs['latest']
                    data_group = self.root['filtered_runs'][latest]
                    self.source_info = {'name': f"filtered/{latest}", 'type': 'filtered'}
                else:
                    runs = self.root['filtered_runs']
                    run_names = sorted([name for name in runs.keys() if name.startswith('run_')])
                    if run_names:
                        data_group = self.root['filtered_runs'][run_names[-1]]
                        self.source_info = {'name': f"filtered/{run_names[-1]}", 'type': 'filtered'}
                    else:
                        data_group = self.root
            else:
                data_group = self.root
        elif source == 'original':
            data_group = self.root
        else:
            # Try to access the specified source directly
            try:
                data_group = self.root[source]
                self.source_info = {'name': source, 'type': 'custom'}
            except KeyError:
                if self.verbose:
                    print(f"Source '{source}' not found, using root data")
                data_group = self.root
        
        # Load data arrays
        if 'bboxes' in data_group:
            self.bboxes = data_group['bboxes'][:]
            self.n_detections = data_group['n_detections'][:]
        else:
            # Handle different zarr structures (e.g., detect_runs)
            if 'detect_runs' in self.root:
                detect_group = self.root['detect_runs']
                if 'latest' in detect_group.attrs:
                    latest_detect = detect_group.attrs['latest']
                    data_group = detect_group[latest_detect]
                    
                    # Handle normalized bbox coordinates
                    if 'bbox_norm_coords' in data_group:
                        self._load_from_normalized_coords(data_group)
                        return
            
            raise ValueError(f"Could not find 'bboxes' data in {self.source_info['name']}")
        
        # Get metadata
        self.fps = self.root.attrs.get('fps', 60.0)
        self.total_frames = len(self.n_detections)
        
        # Extract positions from bounding boxes
        self.positions_x = np.full(self.total_frames, np.nan)
        self.positions_y = np.full(self.total_frames, np.nan)
        
        for frame_idx in range(self.total_frames):
            if self.n_detections[frame_idx] > 0:
                bbox = self.bboxes[frame_idx, 0]  # First detection
                # Calculate centroid from bbox [x1, y1, x2, y2]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                self.positions_x[frame_idx] = cx
                self.positions_y[frame_idx] = cy
        
        # Create time axis
        self.time_axis = np.arange(self.total_frames) / self.fps
        
        # Calculate speed
        self._calculate_speed()
        
        if self.verbose:
            coverage = np.sum(~np.isnan(self.positions_x)) / self.total_frames
            print(f"Loaded {self.source_info['name']} data: {coverage*100:.1f}% coverage")
    
    def _load_from_normalized_coords(self, data_group):
        """Load data from normalized coordinate format."""
        n_detections = data_group['n_detections'][:]
        bbox_coords = data_group['bbox_norm_coords'][:]
        
        # Get metadata first (before using fps)
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Get dimensions
        width = self.root.attrs.get('width', 640)
        height = self.root.attrs.get('height', 640)
        
        self.total_frames = len(n_detections)
        self.positions_x = np.full(self.total_frames, np.nan)
        self.positions_y = np.full(self.total_frames, np.nan)
        
        # Extract positions
        cumulative = np.cumsum(np.insert(n_detections, 0, 0))
        for frame_idx in range(self.total_frames):
            if n_detections[frame_idx] > 0:
                bbox_idx = cumulative[frame_idx]
                # Normalized coords are [center_x, center_y, width, height]
                self.positions_x[frame_idx] = bbox_coords[bbox_idx][0] * width
                self.positions_y[frame_idx] = bbox_coords[bbox_idx][1] * height
        
        self.time_axis = np.arange(self.total_frames) / self.fps
        self._calculate_speed()
        
        if self.verbose:
            coverage = np.sum(~np.isnan(self.positions_x)) / self.total_frames
            print(f"Loaded normalized coord data: {coverage*100:.1f}% coverage")
    
    def _calculate_speed(self):
        """Calculate instantaneous speed from positions."""
        dx = np.diff(self.positions_x)
        dy = np.diff(self.positions_y)
        
        # Handle gaps
        frame_gaps = np.where(np.isnan(self.positions_x[:-1]) | 
                              np.isnan(self.positions_x[1:]))[0]
        dx[frame_gaps] = np.nan
        dy[frame_gaps] = np.nan
        
        # Calculate speed in pixels/second
        displacement = np.sqrt(dx**2 + dy**2)
        self.speed_px = np.full(self.total_frames, np.nan)
        self.speed_px[1:] = displacement * self.fps
        
        # Smooth to reduce noise
        valid_mask = ~np.isnan(self.speed_px)
        if np.sum(valid_mask) > 5:
            self.speed_px[valid_mask] = savgol_filter(
                self.speed_px[valid_mask],
                window_length=min(5, np.sum(valid_mask)),
                polyorder=2
            )
    
    def _detect_bouts(self) -> List[BoutWithUnits]:
        """Detect and characterize swimming bouts."""
        bouts = []
        
        # Find periods above threshold
        above_threshold = self.speed_px > self.speed_threshold_px
        
        # Find transitions
        transitions = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        bout_starts = np.where(transitions == 1)[0]
        bout_ends = np.where(transitions == -1)[0]
        
        # Filter and merge bouts
        min_bout_frames = int(self.min_bout_duration_s * self.fps)
        min_gap_frames = int(self.min_gap_duration_s * self.fps)
        
        filtered_bouts = []
        for start, end in zip(bout_starts, bout_ends):
            if end - start >= min_bout_frames:
                filtered_bouts.append((start, end))
        
        # Merge close bouts
        if filtered_bouts:
            merged_bouts = [filtered_bouts[0]]
            for start, end in filtered_bouts[1:]:
                if start - merged_bouts[-1][1] < min_gap_frames:
                    merged_bouts[-1] = (merged_bouts[-1][0], end)
                else:
                    merged_bouts.append((start, end))
            filtered_bouts = merged_bouts
        
        # Create bout objects with units
        for i, (start, end) in enumerate(filtered_bouts):
            pos_x = self.positions_x[start:end]
            pos_y = self.positions_y[start:end]
            speeds_px = self.speed_px[start:end]
            
            if np.all(np.isnan(pos_x)) or np.all(np.isnan(pos_y)):
                continue
            
            # Calculate metrics
            duration_frames = end - start
            duration_s = duration_frames / self.fps
            
            # Distance calculation
            valid_mask = ~(np.isnan(pos_x[:-1]) | np.isnan(pos_x[1:]))
            if np.any(valid_mask):
                dx = np.diff(pos_x)
                dy = np.diff(pos_y)
                distances = np.sqrt(dx[valid_mask]**2 + dy[valid_mask]**2)
                distance_px = np.sum(distances)
            else:
                distance_px = 0
            
            # Speed statistics
            valid_speeds = speeds_px[~np.isnan(speeds_px)]
            if len(valid_speeds) > 0:
                mean_speed_px = np.mean(valid_speeds)
                peak_speed_px = np.max(valid_speeds)
            else:
                mean_speed_px = peak_speed_px = 0
            
            # Create bout with units
            bout = BoutWithUnits(
                bout_id=i + 1,
                start_frame=start,
                end_frame=end,
                duration_frames=duration_frames,
                duration_s=duration_s,
                distance_px=distance_px,
                mean_speed_px_s=mean_speed_px,
                peak_speed_px_s=peak_speed_px,
                start_time_s=start / self.fps,
                end_time_s=end / self.fps,
                positions_x=pos_x,
                positions_y=pos_y,
                speeds_px_s=speeds_px
            )
            
            # Add calibrated units if available
            if self.calibration:
                bout.distance_mm = distance_px * self.calibration.pixel_to_mm
                bout.distance_bl = bout.distance_mm / self.calibration.fish_length_mm
                bout.mean_speed_mm_s = mean_speed_px * self.calibration.pixel_to_mm
                bout.mean_speed_bl_s = bout.mean_speed_mm_s / self.calibration.fish_length_mm
                bout.peak_speed_mm_s = peak_speed_px * self.calibration.pixel_to_mm
                bout.peak_speed_bl_s = bout.peak_speed_mm_s / self.calibration.fish_length_mm
            
            bouts.append(bout)
        
        return bouts
    
    def _print_summary(self):
        """Print analysis summary with appropriate units."""
        print("\n" + "="*60)
        print("BOUT ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total bouts detected: {len(self.bouts)}")
        
        if not self.bouts:
            return
        
        # Calculate statistics
        durations = [b.duration_s for b in self.bouts]
        distances_px = [b.distance_px for b in self.bouts]
        mean_speeds_px = [b.mean_speed_px_s for b in self.bouts]
        peak_speeds_px = [b.peak_speed_px_s for b in self.bouts]
        
        total_time = self.total_frames / self.fps
        active_time = sum(durations)
        total_distance_px = sum(distances_px)
        
        print(f"Bout rate: {len(self.bouts)/(total_time/60):.1f} bouts/min")
        print(f"Active time: {active_time:.1f}s ({active_time/total_time*100:.1f}%)")
        print(f"\nBout duration: {np.mean(durations):.3f} ± {np.std(durations):.3f} s")
        
        # Display with appropriate units
        if self.calibration and self.display_units != 'pixels':
            distances_mm = [b.distance_mm for b in self.bouts]
            distances_bl = [b.distance_bl for b in self.bouts]
            mean_speeds_mm = [b.mean_speed_mm_s for b in self.bouts]
            mean_speeds_bl = [b.mean_speed_bl_s for b in self.bouts]
            
            print(f"\nDistance per bout:")
            print(f"  Pixels: {np.mean(distances_px):.1f} ± {np.std(distances_px):.1f} px")
            print(f"  Millimeters: {np.mean(distances_mm):.2f} ± {np.std(distances_mm):.2f} mm")
            print(f"  Body lengths: {np.mean(distances_bl):.2f} ± {np.std(distances_bl):.2f} BL")
            
            print(f"\nMean bout speed:")
            print(f"  Pixels: {np.mean(mean_speeds_px):.1f} ± {np.std(mean_speeds_px):.1f} px/s")
            print(f"  Millimeters: {np.mean(mean_speeds_mm):.2f} ± {np.std(mean_speeds_mm):.2f} mm/s")
            print(f"  Body lengths: {np.mean(mean_speeds_bl):.2f} ± {np.std(mean_speeds_bl):.2f} BL/s")
            
            print(f"\nTotal distance traveled:")
            print(f"  {total_distance_px:.1f} pixels")
            print(f"  {total_distance_px * self.calibration.pixel_to_mm:.1f} mm")
            print(f"  {total_distance_px * self.calibration.pixel_to_mm / self.calibration.fish_length_mm:.1f} body lengths")
        else:
            print(f"\nDistance per bout: {np.mean(distances_px):.1f} ± {np.std(distances_px):.1f} pixels")
            print(f"Mean speed: {np.mean(mean_speeds_px):.1f} ± {np.std(mean_speeds_px):.1f} px/s")
            print(f"Peak speed: {np.max(peak_speeds_px):.1f} px/s")
            print(f"Total distance: {total_distance_px:.1f} pixels")
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Create comprehensive bout analysis visualization."""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Determine unit labels
        if self.calibration and self.display_units != 'pixels':
            speed_unit = 'mm/s' if self.display_units == 'mm' else 'BL/s'
            distance_unit = 'mm' if self.display_units == 'mm' else 'BL'
            speed_values = [b.mean_speed_mm_s if self.display_units == 'mm' else b.mean_speed_bl_s 
                           for b in self.bouts]
            distance_values = [b.distance_mm if self.display_units == 'mm' else b.distance_bl 
                              for b in self.bouts]
            threshold_display = (self.speed_threshold_px * self.calibration.pixel_to_mm 
                               if self.display_units == 'mm' 
                               else self.speed_threshold_px * self.calibration.pixel_to_body_length)
        else:
            speed_unit = 'px/s'
            distance_unit = 'pixels'
            speed_values = [b.mean_speed_px_s for b in self.bouts]
            distance_values = [b.distance_px for b in self.bouts]
            threshold_display = self.speed_threshold_px
        
        # 1. Speed trace with bouts (spanning top row)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot speed in selected units
        if self.calibration and self.display_units != 'pixels':
            if self.display_units == 'mm':
                speed_plot = self.speed_px * self.calibration.pixel_to_mm
            else:
                speed_plot = self.speed_px * self.calibration.pixel_to_body_length
        else:
            speed_plot = self.speed_px
        
        ax1.plot(self.time_axis, speed_plot, 'k-', alpha=0.5, linewidth=0.5, label='Speed')
        ax1.axhline(y=threshold_display, color='r', linestyle='--', 
                   alpha=0.5, label=f'Threshold ({threshold_display:.1f} {speed_unit})')
        
        # Mark bouts
        for bout in self.bouts:
            ax1.axvspan(bout.start_time_s, bout.end_time_s, alpha=0.3, color='green')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel(f'Speed ({speed_unit})')
        ax1.set_title(f'Swimming Speed and Bout Detection (n={len(self.bouts)} bouts)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Duration distribution
        ax2 = fig.add_subplot(gs[1, 0])
        durations = [b.duration_s for b in self.bouts]
        if durations:
            ax2.hist(durations, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(np.mean(durations), color='red', linestyle='--',
                       label=f'Mean: {np.mean(durations):.3f}s')
            ax2.set_xlabel('Bout Duration (seconds)')
            ax2.set_ylabel('Count')
            ax2.set_title('Bout Duration Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Inter-bout intervals
        ax3 = fig.add_subplot(gs[1, 1])
        if len(self.bouts) > 1:
            ibis = []
            for i in range(1, len(self.bouts)):
                ibi = self.bouts[i].start_time_s - self.bouts[i-1].end_time_s
                ibis.append(ibi)
            
            ax3.hist(ibis, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax3.axvline(np.mean(ibis), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ibis):.3f}s')
            ax3.set_xlabel('Inter-Bout Interval (seconds)')
            ax3.set_ylabel('Count')
            ax3.set_title('IBI Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Distance per bout
        ax4 = fig.add_subplot(gs[1, 2])
        if distance_values:
            ax4.hist(distance_values, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(np.mean(distance_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(distance_values):.2f}')
            ax4.set_xlabel(f'Distance per Bout ({distance_unit})')
            ax4.set_ylabel('Count')
            ax4.set_title('Bout Distance Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Speed distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if speed_values:
            ax5.hist(speed_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax5.axvline(np.mean(speed_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(speed_values):.2f}')
            ax5.set_xlabel(f'Mean Bout Speed ({speed_unit})')
            ax5.set_ylabel('Count')
            ax5.set_title('Speed Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Duration vs Distance
        ax6 = fig.add_subplot(gs[2, 1])
        if durations and distance_values:
            scatter = ax6.scatter(durations, distance_values, 
                                c=range(len(durations)), cmap='viridis',
                                s=30, alpha=0.6)
            
            # Add trend line
            z = np.polyfit(durations, distance_values, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(durations), max(durations), 100)
            ax6.plot(x_line, p(x_line), 'r--', alpha=0.5,
                    label=f'Slope: {z[0]:.1f} {distance_unit}/s')
            
            ax6.set_xlabel('Duration (seconds)')
            ax6.set_ylabel(f'Distance ({distance_unit})')
            ax6.set_title('Duration vs Distance')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax6, label='Bout #')
        
        # 7. Temporal pattern
        ax7 = fig.add_subplot(gs[2, 2])
        bout_times = [b.start_time_s for b in self.bouts]
        if bout_times and distance_values:
            ax7.scatter(bout_times, distance_values, alpha=0.6, s=20)
            ax7.set_xlabel('Time (seconds)')
            ax7.set_ylabel(f'Distance ({distance_unit})')
            ax7.set_title('Bout Pattern Over Time')
            ax7.grid(True, alpha=0.3)
        
        # 8. Summary statistics box
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create summary text with units
        stats_text = self._create_summary_text()
        
        ax8.text(0.5, 0.5, stats_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='center',
                horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        fig.suptitle('Enhanced Swimming Bout Analysis with Calibration', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def _create_summary_text(self) -> str:
        """Create formatted summary text with units."""
        if not self.bouts:
            return "No bouts detected"
        
        # Calculate statistics
        durations = [b.duration_s for b in self.bouts]
        total_time = self.total_frames / self.fps
        active_time = sum(durations)
        
        lines = ["SUMMARY STATISTICS", "="*50]
        
        # Basic stats
        lines.append(f"Total bouts: {len(self.bouts)}")
        lines.append(f"Bout rate: {len(self.bouts)/(total_time/60):.1f} bouts/min")
        lines.append(f"Active time: {active_time:.1f}s ({active_time/total_time*100:.1f}%)")
        lines.append("")
        
        # Duration stats
        lines.append(f"Duration (s): {np.mean(durations):.3f} ± {np.std(durations):.3f}")
        lines.append(f"  Range: {np.min(durations):.3f} - {np.max(durations):.3f}")
        lines.append("")
        
        # Distance and speed with units
        if self.calibration:
            lines.append("MEASUREMENTS (with calibration):")
            lines.append("-"*40)
            
            # Create table format
            headers = ["Metric", "Pixels", "Millimeters", "Body Lengths"]
            
            # Distance
            dist_px = [b.distance_px for b in self.bouts]
            dist_mm = [b.distance_mm for b in self.bouts]
            dist_bl = [b.distance_bl for b in self.bouts]
            
            lines.append("Distance per bout (mean ± std):")
            lines.append(f"  {np.mean(dist_px):.1f} ± {np.std(dist_px):.1f} px | "
                        f"{np.mean(dist_mm):.2f} ± {np.std(dist_mm):.2f} mm | "
                        f"{np.mean(dist_bl):.2f} ± {np.std(dist_bl):.2f} BL")
            
            # Speed
            speed_px = [b.mean_speed_px_s for b in self.bouts]
            speed_mm = [b.mean_speed_mm_s for b in self.bouts]
            speed_bl = [b.mean_speed_bl_s for b in self.bouts]
            
            lines.append("Mean speed:")
            lines.append(f"  {np.mean(speed_px):.1f} ± {np.std(speed_px):.1f} px/s | "
                        f"{np.mean(speed_mm):.2f} ± {np.std(speed_mm):.2f} mm/s | "
                        f"{np.mean(speed_bl):.2f} ± {np.std(speed_bl):.2f} BL/s")
            
            # Peak speeds
            peak_px = [b.peak_speed_px_s for b in self.bouts]
            peak_mm = [b.peak_speed_mm_s for b in self.bouts]
            peak_bl = [b.peak_speed_bl_s for b in self.bouts]
            
            lines.append("Peak speed (max):")
            lines.append(f"  {np.max(peak_px):.1f} px/s | "
                        f"{np.max(peak_mm):.2f} mm/s | "
                        f"{np.max(peak_bl):.2f} BL/s")
            
            # Total distance
            total_px = sum(dist_px)
            lines.append(f"\nTotal distance: {total_px:.1f} px | "
                        f"{total_px * self.calibration.pixel_to_mm:.1f} mm | "
                        f"{total_px * self.calibration.pixel_to_body_length:.1f} BL")
        else:
            lines.append("MEASUREMENTS (no calibration):")
            lines.append("-"*40)
            dist_px = [b.distance_px for b in self.bouts]
            speed_px = [b.mean_speed_px_s for b in self.bouts]
            
            lines.append(f"Distance: {np.mean(dist_px):.1f} ± {np.std(dist_px):.1f} pixels")
            lines.append(f"Speed: {np.mean(speed_px):.1f} ± {np.std(speed_px):.1f} px/s")
            lines.append(f"Total: {sum(dist_px):.1f} pixels")
        
        return "\n".join(lines)
    
    def export_to_csv(self, csv_path: str):
        """Export bout data to CSV with all units."""
        data = []
        
        for bout in self.bouts:
            row = {
                'bout_id': bout.bout_id,
                'start_frame': bout.start_frame,
                'end_frame': bout.end_frame,
                'start_time_s': bout.start_time_s,
                'end_time_s': bout.end_time_s,
                'duration_s': bout.duration_s,
                'distance_px': bout.distance_px,
                'mean_speed_px_s': bout.mean_speed_px_s,
                'peak_speed_px_s': bout.peak_speed_px_s
            }
            
            if self.calibration:
                row.update({
                    'distance_mm': bout.distance_mm,
                    'distance_bl': bout.distance_bl,
                    'mean_speed_mm_s': bout.mean_speed_mm_s,
                    'mean_speed_bl_s': bout.mean_speed_bl_s,
                    'peak_speed_mm_s': bout.peak_speed_mm_s,
                    'peak_speed_bl_s': bout.peak_speed_bl_s
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add IBI
        ibis = [np.nan]
        for i in range(1, len(self.bouts)):
            ibi = self.bouts[i].start_time_s - self.bouts[i-1].end_time_s
            ibis.append(ibi)
        df['inter_bout_interval_s'] = ibis
        
        df.to_csv(csv_path, index=False)
        print(f"Data exported to: {csv_path}")
        
        return df


    def save_to_zarr(self, group_name: str = 'bout_analysis'):
        """Save bout analysis results to zarr file."""
        if 'r+' not in str(self.root.mode):
            print("Warning: Zarr file opened in read-only mode. Cannot save results.")
            return
        
        # Create or get bout analysis group
        if group_name in self.root:
            bout_group = self.root[group_name]
            print(f"Updating existing {group_name} group...")
        else:
            bout_group = self.root.create_group(group_name)
            print(f"Creating new {group_name} group...")
        
        # Save metadata
        bout_group.attrs['analysis_date'] = datetime.now().isoformat()
        bout_group.attrs['speed_threshold_px'] = self.speed_threshold_px
        bout_group.attrs['min_bout_duration_s'] = self.min_bout_duration_s
        bout_group.attrs['min_gap_duration_s'] = self.min_gap_duration_s
        bout_group.attrs['n_bouts'] = len(self.bouts)
        bout_group.attrs['source_data'] = self.source_info['name'] if hasattr(self, 'source_info') else 'unknown'
        
        if self.calibration:
            bout_group.attrs['speed_threshold_mm'] = self.speed_threshold_px * self.calibration.pixel_to_mm
            bout_group.attrs['speed_threshold_bl'] = (self.speed_threshold_px * self.calibration.pixel_to_mm / 
                                                       self.calibration.fish_length_mm)
        
        # Save bout data as arrays
        if self.bouts:
            bout_ids = np.array([b.bout_id for b in self.bouts])
            start_frames = np.array([b.start_frame for b in self.bouts])
            end_frames = np.array([b.end_frame for b in self.bouts])
            durations = np.array([b.duration_s for b in self.bouts])
            distances_px = np.array([b.distance_px for b in self.bouts])
            mean_speeds_px = np.array([b.mean_speed_px_s for b in self.bouts])
            peak_speeds_px = np.array([b.peak_speed_px_s for b in self.bouts])
            
            # Save arrays
            bout_group.array('bout_ids', bout_ids, overwrite=True)
            bout_group.array('start_frames', start_frames, overwrite=True)
            bout_group.array('end_frames', end_frames, overwrite=True)
            bout_group.array('durations_s', durations, overwrite=True)
            bout_group.array('distances_px', distances_px, overwrite=True)
            bout_group.array('mean_speeds_px_s', mean_speeds_px, overwrite=True)
            bout_group.array('peak_speeds_px_s', peak_speeds_px, overwrite=True)
            
            # Save calibrated measurements if available
            if self.calibration:
                distances_mm = np.array([b.distance_mm for b in self.bouts])
                distances_bl = np.array([b.distance_bl for b in self.bouts])
                mean_speeds_mm = np.array([b.mean_speed_mm_s for b in self.bouts])
                mean_speeds_bl = np.array([b.mean_speed_bl_s for b in self.bouts])
                peak_speeds_mm = np.array([b.peak_speed_mm_s for b in self.bouts])
                peak_speeds_bl = np.array([b.peak_speed_bl_s for b in self.bouts])
                
                bout_group.array('distances_mm', distances_mm, overwrite=True)
                bout_group.array('distances_bl', distances_bl, overwrite=True)
                bout_group.array('mean_speeds_mm_s', mean_speeds_mm, overwrite=True)
                bout_group.array('mean_speeds_bl_s', mean_speeds_bl, overwrite=True)
                bout_group.array('peak_speeds_mm_s', peak_speeds_mm, overwrite=True)
                bout_group.array('peak_speeds_bl_s', peak_speeds_bl, overwrite=True)
            
            # Calculate and save summary statistics
            stats = self.calculate_summary_statistics()
            bout_group.attrs.update(stats)
            
            print(f"Saved {len(self.bouts)} bouts to zarr")
    
    def calculate_summary_statistics(self) -> Dict:
        """Calculate comprehensive summary statistics."""
        if not self.bouts:
            return {'n_bouts': 0}
        
        durations = [b.duration_s for b in self.bouts]
        distances_px = [b.distance_px for b in self.bouts]
        mean_speeds_px = [b.mean_speed_px_s for b in self.bouts]
        peak_speeds_px = [b.peak_speed_px_s for b in self.bouts]
        
        total_time = self.total_frames / self.fps
        active_time = sum(durations)
        
        stats = {
            'n_bouts': len(self.bouts),
            'bout_rate_per_min': len(self.bouts) / (total_time / 60),
            'total_active_time_s': active_time,
            'percent_active': (active_time / total_time) * 100,
            'total_distance_px': sum(distances_px),
            
            'duration_mean_s': np.mean(durations),
            'duration_std_s': np.std(durations),
            'duration_median_s': np.median(durations),
            'duration_min_s': np.min(durations),
            'duration_max_s': np.max(durations),
            
            'distance_mean_px': np.mean(distances_px),
            'distance_std_px': np.std(distances_px),
            'distance_median_px': np.median(distances_px),
            
            'mean_speed_mean_px_s': np.mean(mean_speeds_px),
            'mean_speed_std_px_s': np.std(mean_speeds_px),
            'peak_speed_max_px_s': np.max(peak_speeds_px),
        }
        
        # Add calibrated statistics if available
        if self.calibration:
            distances_mm = [b.distance_mm for b in self.bouts]
            distances_bl = [b.distance_bl for b in self.bouts]
            mean_speeds_mm = [b.mean_speed_mm_s for b in self.bouts]
            mean_speeds_bl = [b.mean_speed_bl_s for b in self.bouts]
            peak_speeds_mm = [b.peak_speed_mm_s for b in self.bouts]
            peak_speeds_bl = [b.peak_speed_bl_s for b in self.bouts]
            
            stats.update({
                'total_distance_mm': sum(distances_mm),
                'total_distance_bl': sum(distances_bl),
                
                'distance_mean_mm': np.mean(distances_mm),
                'distance_std_mm': np.std(distances_mm),
                'distance_mean_bl': np.mean(distances_bl),
                'distance_std_bl': np.std(distances_bl),
                
                'mean_speed_mean_mm_s': np.mean(mean_speeds_mm),
                'mean_speed_std_mm_s': np.std(mean_speeds_mm),
                'mean_speed_mean_bl_s': np.mean(mean_speeds_bl),
                'mean_speed_std_bl_s': np.std(mean_speeds_bl),
                
                'peak_speed_max_mm_s': np.max(peak_speeds_mm),
                'peak_speed_max_bl_s': np.max(peak_speeds_bl),
            })
        
        # Calculate inter-bout intervals
        if len(self.bouts) > 1:
            ibis = []
            for i in range(1, len(self.bouts)):
                ibi = self.bouts[i].start_time_s - self.bouts[i-1].end_time_s
                ibis.append(ibi)
            
            stats.update({
                'ibi_mean_s': np.mean(ibis),
                'ibi_std_s': np.std(ibis),
                'ibi_median_s': np.median(ibis),
                'ibi_min_s': np.min(ibis),
                'ibi_max_s': np.max(ibis),
            })
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced bout analyzer with calibration support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with automatic calibration detection
  %(prog)s detections.zarr
  
  # Set threshold in mm/s (requires calibration)
  %(prog)s detections.zarr --threshold-mm 2.0
  
  # Set threshold in body lengths per second
  %(prog)s detections.zarr --threshold-bl 0.5
  
  # Force display in specific units
  %(prog)s detections.zarr --display-units mm
  
  # Full analysis with exports
  %(prog)s detections.zarr --threshold-mm 2.0 --output-plot bouts.png --output-csv bouts.csv --save-zarr
        """
    )
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--source', default='latest',
                       help='Data source (latest/filtered/interpolated/original)')
    
    # Threshold options
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument('--threshold-px', type=float,
                                help='Speed threshold in pixels/second')
    threshold_group.add_argument('--threshold-mm', type=float,
                                help='Speed threshold in mm/second')
    threshold_group.add_argument('--threshold-bl', type=float, default=0.5,
                                help='Speed threshold in body lengths/second (default: 0.5)')
    
    parser.add_argument('--min-bout-duration', type=float, default=0.05,
                       help='Minimum bout duration in seconds (default: 0.05)')
    parser.add_argument('--min-gap', type=float, default=0.1,
                       help='Minimum gap between bouts in seconds (default: 0.1)')
    parser.add_argument('--display-units', choices=['auto', 'pixels', 'mm', 'cm', 'bl'],
                       default='auto', help='Display units for output (default: auto)')
    parser.add_argument('--output-plot', type=str,
                       help='Path to save analysis plot')
    parser.add_argument('--output-csv', type=str,
                       help='Path to save CSV data')
    parser.add_argument('--save-zarr', action='store_true',
                       help='Save bout analysis results to zarr file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedBoutAnalyzer(
        zarr_path=args.zarr_path,
        source=args.source,
        speed_threshold_px_s=args.threshold_px,
        speed_threshold_mm_s=args.threshold_mm,
        speed_threshold_bl_s=args.threshold_bl,
        min_bout_duration_s=args.min_bout_duration,
        min_gap_duration_s=args.min_gap,
        display_units=args.display_units,
        verbose=not args.quiet
    )
    
    # Save to zarr if requested
    if args.save_zarr:
        analyzer.save_to_zarr()
    
    # Create visualization
    if args.output_plot or not args.quiet:
        analyzer.plot_analysis(save_path=args.output_plot)
    
    # Export data if requested
    if args.output_csv:
        analyzer.export_to_csv(args.output_csv)
    
    return 0


if __name__ == '__main__':
    import sys
    from datetime import datetime
    sys.exit(main())