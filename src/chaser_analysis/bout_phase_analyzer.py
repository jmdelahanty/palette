#!/usr/bin/env python3
"""
Training Phase Bout Analyzer

Analyzes swimming bouts separately for pre-training, training, and post-training phases.
Creates comprehensive comparisons of bout metrics across experimental phases.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import zarr
import h5py
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import argparse
import sys
import os

# Import the enhanced bout analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_analysis_environment_info():
    """
    Get comprehensive environment and code version information.
    Tracks git commit, package versions, and analysis context.
    """
    import sys
    import subprocess
    from datetime import datetime
    import platform
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'python_executable': sys.executable,
    }
    
    # Get git information
    try:
        # Get current commit hash
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode('ascii').strip()
        info['git_commit'] = git_hash
        info['git_commit_short'] = git_hash[:8]
        
        # Get branch name
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode('ascii').strip()
        info['git_branch'] = git_branch
        
        # Check if working directory has uncommitted changes
        git_status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode('ascii').strip()
        info['git_dirty'] = len(git_status) > 0
        
        # Get last commit date
        git_date = subprocess.check_output(
            ['git', 'log', '-1', '--format=%ai'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode('ascii').strip()
        info['git_commit_date'] = git_date
        
    except Exception as e:
        info['git_commit'] = 'unknown'
        info['git_error'] = str(e)
    
    # Track package versions that affect analysis
    packages = {
        'numpy': 'np',
        'scipy': 'scipy',
        'pandas': 'pd',
        'zarr': 'zarr',
        'matplotlib': 'matplotlib',
        'h5py': 'h5py'
    }
    
    info['package_versions'] = {}
    for package_name, import_name in packages.items():
        try:
            if import_name == 'np':
                import numpy as np
                info['package_versions'][package_name] = np.__version__
            elif import_name == 'scipy':
                import scipy
                info['package_versions'][package_name] = scipy.__version__
            elif import_name == 'pd':
                import pandas as pd
                info['package_versions'][package_name] = pd.__version__
            elif import_name == 'zarr':
                import zarr
                info['package_versions'][package_name] = zarr.__version__
            elif import_name == 'matplotlib':
                import matplotlib
                info['package_versions'][package_name] = matplotlib.__version__
            elif import_name == 'h5py':
                import h5py
                info['package_versions'][package_name] = h5py.__version__
        except ImportError:
            info['package_versions'][package_name] = 'not installed'
    
    # Add script information
    info['script_name'] = os.path.basename(__file__)
    info['script_path'] = os.path.abspath(__file__)
    
    return info


@dataclass
class PhaseMetrics:
    """Metrics for a specific experimental phase."""
    phase_name: str
    start_frame: int
    end_frame: int
    duration_s: float
    n_bouts: int
    bout_rate_per_min: float
    total_distance_px: float
    total_distance_mm: Optional[float]
    mean_bout_duration_s: float
    std_bout_duration_s: float
    mean_bout_distance_px: float
    mean_bout_distance_mm: Optional[float]
    mean_speed_px_s: float
    mean_speed_mm_s: Optional[float]
    mean_ibi_s: float
    percent_active: float
    bout_durations: List[float] = field(default_factory=list)
    bout_distances: List[float] = field(default_factory=list)
    bout_speeds: List[float] = field(default_factory=list)
    ibis: List[float] = field(default_factory=list)


class TrainingPhaseBoutAnalyzer:
    """Analyze bouts across different training phases."""
    
    def __init__(
        self,
        zarr_path: str,
        h5_path: Optional[str] = None,
        phase_durations: Optional[Dict[str, float]] = None,
        speed_threshold_bl_s: float = 0.5,
        min_bout_duration_s: float = 0.05,
        min_gap_duration_s: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize phase-based bout analyzer.
        
        Args:
            zarr_path: Path to zarr file with tracking data
            h5_path: Optional path to H5 file with event markers
            phase_durations: Dictionary with phase durations in seconds
                           Default: {'pre': 300, 'training': 150, 'post': 300}
            speed_threshold_bl_s: Speed threshold in body lengths/second
            min_bout_duration_s: Minimum bout duration
            min_gap_duration_s: Minimum gap between bouts
            verbose: Print progress
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path) if h5_path else None
        self.verbose = verbose
        
        # Set default phase durations if not provided
        if phase_durations is None:
            self.phase_durations = {
                'pre_training': 300,  # 5 minutes
                'training': 150,      # 2.5 minutes
                'post_training': 300  # 5 minutes
            }
        else:
            self.phase_durations = phase_durations
        
        # Load zarr data
        self.root = zarr.open(str(self.zarr_path), mode='r')
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Load calibration
        self.calibration = self._load_calibration()
        
        # Bout detection parameters
        self.speed_threshold_bl_s = speed_threshold_bl_s
        self.min_bout_duration_s = min_bout_duration_s
        self.min_gap_duration_s = min_gap_duration_s
        
        # Identify phase boundaries
        self.phases = self._identify_phases()
        
        # Load and process tracking data
        self._load_tracking_data()
        
        # Analyze bouts for each phase
        self.phase_metrics = self._analyze_phases()
        
        if verbose:
            self._print_phase_summary()
    
    def _load_calibration(self) -> Optional[Dict]:
        """Load calibration data from zarr."""
        if 'calibration' not in self.root:
            return None
        
        calib_group = self.root['calibration']
        pixel_to_mm = calib_group.attrs.get('pixel_to_mm')
        
        if not pixel_to_mm:
            return None
        
        return {
            'pixel_to_mm': pixel_to_mm,
            'pixels_per_mm': 1.0 / pixel_to_mm,
            'fish_length_mm': calib_group.attrs.get('fish_length_mm', 4.0)
        }
    
    def _identify_phases(self) -> Dict:
        """Identify phase boundaries from H5 events or use timing."""
        phases = {}
        
        if self.h5_path and self.h5_path.exists():
            # Try to load phase markers from H5 events
            phases = self._load_phases_from_h5()
        
        if not phases:
            # Use default timing-based phases
            cumulative_time = 0
            for phase_name, duration_s in self.phase_durations.items():
                start_frame = int(cumulative_time * self.fps)
                end_frame = int((cumulative_time + duration_s) * self.fps)
                phases[phase_name] = {
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'duration_s': duration_s
                }
                cumulative_time += duration_s
        
        # Adjust to actual data length
        # Try different possible locations for frame count
        if 'frame_count' in self.root.attrs:
            total_frames = self.root.attrs['frame_count']
        elif 'n_detections' in self.root:
            total_frames = len(self.root['n_detections'])
        elif 'bboxes' in self.root:
            total_frames = self.root['bboxes'].shape[0]
        else:
            # Try to find it in detect_runs or other groups
            total_frames = None
            if 'detect_runs' in self.root and 'latest' in self.root['detect_runs'].attrs:
                latest = self.root['detect_runs'].attrs['latest']
                if 'n_detections' in self.root['detect_runs'][latest]:
                    total_frames = len(self.root['detect_runs'][latest]['n_detections'])
            
            if total_frames is None:
                # Default to expected duration
                total_frames = int(sum(self.phase_durations.values()) * self.fps)
        
        last_phase = list(phases.keys())[-1]
        phases[last_phase]['end_frame'] = min(phases[last_phase]['end_frame'], total_frames)
        
        return phases
    
    def _load_phases_from_h5(self) -> Dict:
        """Load phase boundaries from H5 event markers."""
        phases = {}
        
        try:
            with h5py.File(self.h5_path, 'r') as hf:
                if '/events' not in hf:
                    return phases
                
                events = hf['/events'][:]
                
                # Look for specific phase markers
                phase_markers = {
                    24: 'pre_training',    # CHASER_PRE_PERIOD_START
                    25: 'training',        # CHASER_TRAINING_START
                    26: 'post_training'    # CHASER_POST_PERIOD_START
                }
                
                # Find phase start times
                phase_starts = {}
                for event in events:
                    # Access structured array fields directly
                    event_type = event['event_type_id']
                    if event_type in phase_markers:
                        phase_name = phase_markers[event_type]
                        # Use frame number if available, otherwise estimate from timestamp
                        if 'frame_number' in event.dtype.names and event['frame_number'] > 0:
                            frame = int(event['frame_number'])
                        else:
                            # Estimate frame from timestamp
                            # Assuming first event is near frame 0
                            first_timestamp = events[0]['timestamp_ns_session']
                            timestamp_diff = event['timestamp_ns_session'] - first_timestamp
                            frame = int(timestamp_diff / 1e9 * self.fps)
                        phase_starts[phase_name] = frame
                
                # Create phase dictionary with durations
                phase_names = ['pre_training', 'training', 'post_training']
                for i, phase_name in enumerate(phase_names):
                    if phase_name in phase_starts:
                        start_frame = phase_starts[phase_name]
                        # End frame is start of next phase or use duration
                        if i < len(phase_names) - 1 and phase_names[i+1] in phase_starts:
                            end_frame = phase_starts[phase_names[i+1]]
                        else:
                            duration_s = self.phase_durations.get(phase_name, 300)
                            end_frame = start_frame + int(duration_s * self.fps)
                        
                        phases[phase_name] = {
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'duration_s': (end_frame - start_frame) / self.fps
                        }
        
        except Exception as e:
            if self.verbose:
                print(f"Could not load phases from H5: {e}")
        
        return phases
    
    def _load_tracking_data(self):
        """Load tracking data from zarr."""
        # Try to find the data in various possible locations
        data_loaded = False
        
        # Try preprocessed data first
        if 'preprocessing' in self.root:
            if 'interpolated_runs' in self.root['preprocessing']:
                runs = self.root['preprocessing']['interpolated_runs']
                run_names = sorted([name for name in runs.keys() if name.startswith('run_')])
                if run_names:
                    data_group = runs[run_names[-1]]
                    data_loaded = True
            elif 'latest' in self.root['preprocessing'].attrs:
                latest = self.root['preprocessing'].attrs['latest']
                data_group = self.root['preprocessing'][latest]
                data_loaded = True
        
        # Try filtered data
        if not data_loaded and 'filtered_runs' in self.root:
            if 'latest' in self.root['filtered_runs'].attrs:
                latest = self.root['filtered_runs'].attrs['latest']
                data_group = self.root['filtered_runs'][latest]
                data_loaded = True
            else:
                runs = self.root['filtered_runs']
                run_names = sorted([name for name in runs.keys() if name.startswith('run_')])
                if run_names:
                    data_group = runs[run_names[-1]]
                    data_loaded = True
        
        # Try detect_runs
        if not data_loaded and 'detect_runs' in self.root:
            if 'latest' in self.root['detect_runs'].attrs:
                latest = self.root['detect_runs'].attrs['latest']
                data_group = self.root['detect_runs'][latest]
                data_loaded = True
        
        # Fall back to root level
        if not data_loaded:
            data_group = self.root
        
        # Load arrays - handle different formats
        if 'bboxes' in data_group:
            # Standard format
            self.bboxes = data_group['bboxes'][:]
            self.n_detections = data_group['n_detections'][:]
            self.total_frames = len(self.n_detections)
            
            # Extract positions
            self.positions_x = np.full(self.total_frames, np.nan)
            self.positions_y = np.full(self.total_frames, np.nan)
            
            for frame_idx in range(self.total_frames):
                if self.n_detections[frame_idx] > 0:
                    bbox = self.bboxes[frame_idx, 0]
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    self.positions_x[frame_idx] = cx
                    self.positions_y[frame_idx] = cy
        
        elif 'bbox_norm_coords' in data_group:
            # Normalized coordinate format
            n_detections = data_group['n_detections'][:]
            bbox_coords = data_group['bbox_norm_coords'][:]
            
            width = self.root.attrs.get('width', 640)
            height = self.root.attrs.get('height', 640)
            
            self.total_frames = len(n_detections)
            self.positions_x = np.full(self.total_frames, np.nan)
            self.positions_y = np.full(self.total_frames, np.nan)
            
            cumulative = np.cumsum(np.insert(n_detections, 0, 0))
            for frame_idx in range(self.total_frames):
                if n_detections[frame_idx] > 0:
                    bbox_idx = cumulative[frame_idx]
                    # Normalized coords are [center_x, center_y, width, height]
                    self.positions_x[frame_idx] = bbox_coords[bbox_idx][0] * width
                    self.positions_y[frame_idx] = bbox_coords[bbox_idx][1] * height
        else:
            raise ValueError("Could not find tracking data (bboxes or bbox_norm_coords)")
        
        # Calculate speed
        self._calculate_speed()
    
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
    
    def _detect_bouts_in_phase(self, start_frame: int, end_frame: int) -> List:
        """Detect bouts within a specific phase."""
        # Get phase data
        phase_speed = self.speed_px[start_frame:end_frame]
        phase_positions_x = self.positions_x[start_frame:end_frame]
        phase_positions_y = self.positions_y[start_frame:end_frame]
        
        # Convert threshold to pixels/second
        if self.calibration:
            speed_threshold_px = (self.speed_threshold_bl_s * 
                                 self.calibration['fish_length_mm'] / 
                                 self.calibration['pixel_to_mm'])
        else:
            speed_threshold_px = 50.0  # Default threshold
        
        # Find bouts
        above_threshold = phase_speed > speed_threshold_px
        transitions = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        bout_starts = np.where(transitions == 1)[0]
        bout_ends = np.where(transitions == -1)[0]
        
        # Filter by minimum duration
        min_bout_frames = int(self.min_bout_duration_s * self.fps)
        min_gap_frames = int(self.min_gap_duration_s * self.fps)
        
        bouts = []
        for start, end in zip(bout_starts, bout_ends):
            if end - start >= min_bout_frames:
                # Calculate bout metrics
                bout_speeds = phase_speed[start:end]
                bout_pos_x = phase_positions_x[start:end]
                bout_pos_y = phase_positions_y[start:end]
                
                # Calculate distance
                valid_mask = ~(np.isnan(bout_pos_x[:-1]) | np.isnan(bout_pos_x[1:]))
                if np.any(valid_mask):
                    dx = np.diff(bout_pos_x)
                    dy = np.diff(bout_pos_y)
                    distances = np.sqrt(dx[valid_mask]**2 + dy[valid_mask]**2)
                    distance_px = np.sum(distances)
                else:
                    distance_px = 0
                
                bouts.append({
                    'start_frame': start,
                    'end_frame': end,
                    'duration_s': (end - start) / self.fps,
                    'distance_px': distance_px,
                    'mean_speed_px': np.nanmean(bout_speeds),
                    'peak_speed_px': np.nanmax(bout_speeds)
                })
        
        return bouts
    
    def _analyze_phases(self) -> Dict[str, PhaseMetrics]:
        """Analyze bouts for each phase."""
        phase_metrics = {}
        
        for phase_name, phase_info in self.phases.items():
            bouts = self._detect_bouts_in_phase(
                phase_info['start_frame'],
                phase_info['end_frame']
            )
            
            # Calculate metrics
            if bouts:
                durations = [b['duration_s'] for b in bouts]
                distances = [b['distance_px'] for b in bouts]
                speeds = [b['mean_speed_px'] for b in bouts]
                
                # Calculate IBIs
                ibis = []
                for i in range(1, len(bouts)):
                    ibi = (bouts[i]['start_frame'] - bouts[i-1]['end_frame']) / self.fps
                    ibis.append(ibi)
                
                total_distance_px = sum(distances)
                active_time = sum(durations)
                phase_duration = phase_info['duration_s']
                
                metrics = PhaseMetrics(
                    phase_name=phase_name,
                    start_frame=phase_info['start_frame'],
                    end_frame=phase_info['end_frame'],
                    duration_s=phase_duration,
                    n_bouts=len(bouts),
                    bout_rate_per_min=len(bouts) / (phase_duration / 60),
                    total_distance_px=total_distance_px,
                    total_distance_mm=total_distance_px * self.calibration['pixel_to_mm'] if self.calibration else None,
                    mean_bout_duration_s=np.mean(durations),
                    std_bout_duration_s=np.std(durations),
                    mean_bout_distance_px=np.mean(distances),
                    mean_bout_distance_mm=np.mean(distances) * self.calibration['pixel_to_mm'] if self.calibration else None,
                    mean_speed_px_s=np.mean(speeds),
                    mean_speed_mm_s=np.mean(speeds) * self.calibration['pixel_to_mm'] if self.calibration else None,
                    mean_ibi_s=np.mean(ibis) if ibis else 0,
                    percent_active=(active_time / phase_duration) * 100,
                    bout_durations=durations,
                    bout_distances=distances,
                    bout_speeds=speeds,
                    ibis=ibis
                )
            else:
                # No bouts detected in this phase
                metrics = PhaseMetrics(
                    phase_name=phase_name,
                    start_frame=phase_info['start_frame'],
                    end_frame=phase_info['end_frame'],
                    duration_s=phase_info['duration_s'],
                    n_bouts=0,
                    bout_rate_per_min=0,
                    total_distance_px=0,
                    total_distance_mm=0,
                    mean_bout_duration_s=0,
                    std_bout_duration_s=0,
                    mean_bout_distance_px=0,
                    mean_bout_distance_mm=0,
                    mean_speed_px_s=0,
                    mean_speed_mm_s=0,
                    mean_ibi_s=0,
                    percent_active=0
                )
            
            phase_metrics[phase_name] = metrics
        
        return phase_metrics
    
    def _print_phase_summary(self):
        """Print summary of bout metrics for each phase."""
        print("\n" + "="*70)
        print("TRAINING PHASE BOUT ANALYSIS")
        print("="*70)
        
        for phase_name, metrics in self.phase_metrics.items():
            print(f"\n{phase_name.upper().replace('_', ' ')} PHASE")
            print("-"*40)
            print(f"Duration: {metrics.duration_s:.1f} seconds")
            print(f"Frames: {metrics.start_frame}-{metrics.end_frame}")
            print(f"Bouts detected: {metrics.n_bouts}")
            
            if metrics.n_bouts > 0:
                print(f"Bout rate: {metrics.bout_rate_per_min:.1f} bouts/min")
                print(f"Active time: {metrics.percent_active:.1f}%")
                print(f"Mean bout duration: {metrics.mean_bout_duration_s:.3f} ± {metrics.std_bout_duration_s:.3f} s")
                
                if self.calibration:
                    print(f"Total distance: {metrics.total_distance_px:.1f} px ({metrics.total_distance_mm:.1f} mm)")
                    print(f"Mean bout distance: {metrics.mean_bout_distance_mm:.2f} mm")
                    print(f"Mean speed: {metrics.mean_speed_mm_s:.2f} mm/s")
                else:
                    print(f"Total distance: {metrics.total_distance_px:.1f} pixels")
                    print(f"Mean bout distance: {metrics.mean_bout_distance_px:.1f} pixels")
                    print(f"Mean speed: {metrics.mean_speed_px_s:.1f} px/s")
                
                if metrics.ibis:
                    print(f"Mean IBI: {metrics.mean_ibi_s:.3f} s")
    
    def plot_phase_comparison(self, save_path: Optional[str] = None):
        """Create comprehensive visualization comparing phases."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        phase_colors = {
            'pre_training': '#3498db',
            'training': '#e74c3c',
            'post_training': '#2ecc71'
        }
        
        phase_labels = {
            'pre_training': 'Pre-Training',
            'training': 'Training',
            'post_training': 'Post-Training'
        }
        
        # 1. Speed trace with phase markers (spanning full width)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot speed
        time_axis = np.arange(self.total_frames) / self.fps / 60  # Convert to minutes
        ax1.plot(time_axis, self.speed_px, 'k-', alpha=0.3, linewidth=0.5)
        
        # Mark phases
        for phase_name, phase_info in self.phases.items():
            start_min = phase_info['start_frame'] / self.fps / 60
            end_min = phase_info['end_frame'] / self.fps / 60
            ax1.axvspan(start_min, end_min, alpha=0.2, color=phase_colors[phase_name],
                       label=phase_labels[phase_name])
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Speed (px/s)')
        ax1.set_title('Swimming Activity Across Experimental Phases')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Bout count comparison
        ax2 = fig.add_subplot(gs[1, 0])
        phase_names = list(self.phase_metrics.keys())
        bout_counts = [self.phase_metrics[p].n_bouts for p in phase_names]
        colors = [phase_colors[p] for p in phase_names]
        
        bars = ax2.bar(range(len(phase_names)), bout_counts, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(phase_names)))
        ax2.set_xticklabels([phase_labels[p] for p in phase_names], rotation=45)
        ax2.set_ylabel('Number of Bouts')
        ax2.set_title('Bout Count by Phase')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, bout_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        # 3. Bout rate comparison
        ax3 = fig.add_subplot(gs[1, 1])
        bout_rates = [self.phase_metrics[p].bout_rate_per_min for p in phase_names]
        
        bars = ax3.bar(range(len(phase_names)), bout_rates, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(phase_names)))
        ax3.set_xticklabels([phase_labels[p] for p in phase_names], rotation=45)
        ax3.set_ylabel('Bouts per Minute')
        ax3.set_title('Bout Rate by Phase')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, bout_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{rate:.1f}', ha='center', va='bottom')
        
        # 4. Mean bout duration
        ax4 = fig.add_subplot(gs[1, 2])
        durations = [self.phase_metrics[p].mean_bout_duration_s for p in phase_names]
        errors = [self.phase_metrics[p].std_bout_duration_s for p in phase_names]
        
        bars = ax4.bar(range(len(phase_names)), durations, yerr=errors,
                      color=colors, alpha=0.7, capsize=5)
        ax4.set_xticks(range(len(phase_names)))
        ax4.set_xticklabels([phase_labels[p] for p in phase_names], rotation=45)
        ax4.set_ylabel('Duration (seconds)')
        ax4.set_title('Mean Bout Duration by Phase')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Total distance traveled
        ax5 = fig.add_subplot(gs[1, 3])
        if self.calibration:
            distances = [self.phase_metrics[p].total_distance_mm for p in phase_names]
            ylabel = 'Distance (mm)'
        else:
            distances = [self.phase_metrics[p].total_distance_px for p in phase_names]
            ylabel = 'Distance (pixels)'
        
        bars = ax5.bar(range(len(phase_names)), distances, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(phase_names)))
        ax5.set_xticklabels([phase_labels[p] for p in phase_names], rotation=45)
        ax5.set_ylabel(ylabel)
        ax5.set_title('Total Distance by Phase')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Bout duration distributions
        ax6 = fig.add_subplot(gs[2, 0:2])
        for phase_name in phase_names:
            if self.phase_metrics[phase_name].bout_durations:
                ax6.hist(self.phase_metrics[phase_name].bout_durations,
                        bins=20, alpha=0.5, label=phase_labels[phase_name],
                        color=phase_colors[phase_name], density=True)
        
        ax6.set_xlabel('Bout Duration (seconds)')
        ax6.set_ylabel('Probability Density')
        ax6.set_title('Bout Duration Distributions')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Speed distributions
        ax7 = fig.add_subplot(gs[2, 2:])
        for phase_name in phase_names:
            if self.phase_metrics[phase_name].bout_speeds:
                speeds = self.phase_metrics[phase_name].bout_speeds
                if self.calibration:
                    speeds = [s * self.calibration['pixel_to_mm'] for s in speeds]
                    xlabel = 'Mean Bout Speed (mm/s)'
                else:
                    xlabel = 'Mean Bout Speed (px/s)'
                
                ax7.hist(speeds, bins=20, alpha=0.5, label=phase_labels[phase_name],
                        color=phase_colors[phase_name], density=True)
        
        ax7.set_xlabel(xlabel)
        ax7.set_ylabel('Probability Density')
        ax7.set_title('Bout Speed Distributions')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle('Swimming Bout Analysis: Training Phase Comparison', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    def statistical_comparison(self) -> pd.DataFrame:
        """Perform statistical comparisons between phases."""
        comparisons = []
        
        phase_names = list(self.phase_metrics.keys())
        
        # Compare each pair of phases
        for i, phase1 in enumerate(phase_names):
            for phase2 in phase_names[i+1:]:
                metrics1 = self.phase_metrics[phase1]
                metrics2 = self.phase_metrics[phase2]
                
                # Compare bout durations
                if metrics1.bout_durations and metrics2.bout_durations:
                    t_stat, p_value = stats.ttest_ind(
                        metrics1.bout_durations,
                        metrics2.bout_durations
                    )
                    comparisons.append({
                        'comparison': f'{phase1} vs {phase2}',
                        'metric': 'bout_duration',
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                
                # Compare bout distances
                if metrics1.bout_distances and metrics2.bout_distances:
                    t_stat, p_value = stats.ttest_ind(
                        metrics1.bout_distances,
                        metrics2.bout_distances
                    )
                    comparisons.append({
                        'comparison': f'{phase1} vs {phase2}',
                        'metric': 'bout_distance',
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
        
        return pd.DataFrame(comparisons)
    
    def save_to_zarr(self, group_name: str = 'bout_phase_analysis'):
        """
        Save phase bout analysis to zarr with full provenance tracking.
        
        Following project conventions:
        - Timestamped runs for versioning
        - Complete parameter tracking
        - Hierarchical data organization
        - 'latest' attribute for easy access
        """
        from datetime import datetime
        
        # Reopen in write mode if needed
        try:
            # Try to create a test group to check write access
            test_group_name = f'_test_{datetime.now().timestamp()}'
            self.root.create_group(test_group_name)
            del self.root[test_group_name]
        except:
            # Reopen in write mode
            self.root = zarr.open(str(self.zarr_path), mode='r+')
            if self.verbose:
                print("\nReopened zarr in write mode for saving results")
        
        # Create or get main group
        if group_name not in self.root:
            main_group = self.root.create_group(group_name)
            if self.verbose:
                print(f"Created new group: /{group_name}")
        else:
            main_group = self.root[group_name]
            if self.verbose:
                print(f"Using existing group: /{group_name}")
        
        # Create timestamped run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        
        # Check if runs subgroup exists, create if not
        if 'runs' not in main_group:
            runs_group = main_group.create_group('runs')
        else:
            runs_group = main_group['runs']
        
        run_group = runs_group.create_group(run_name)
        
        # Store parameters as attributes
        run_group.attrs['created_at'] = datetime.now().isoformat()
        run_group.attrs['parameters'] = {
            'speed_threshold_bl_s': self.speed_threshold_bl_s,
            'min_bout_duration_s': self.min_bout_duration_s,
            'min_gap_duration_s': self.min_gap_duration_s,
            'phase_durations': self.phase_durations
        }
        
        # Store data source info
        run_group.attrs['data_source'] = {
            'zarr_path': str(self.zarr_path),
            'h5_path': str(self.h5_path) if self.h5_path else None,
            'fps': self.fps,
            'total_frames': self.total_frames
        }
        
        # Store calibration info if available
        if self.calibration:
            run_group.attrs['calibration'] = {
                'pixel_to_mm': self.calibration['pixel_to_mm'],
                'fish_length_mm': self.calibration['fish_length_mm'],
                'has_calibration': True
            }
        else:
            run_group.attrs['calibration'] = {'has_calibration': False}
        
        # Store phase boundaries
        phases_group = run_group.create_group('phases')
        for phase_name, phase_info in self.phases.items():
            phase_group = phases_group.create_group(phase_name)
            phase_group.attrs['start_frame'] = phase_info['start_frame']
            phase_group.attrs['end_frame'] = phase_info['end_frame']
            phase_group.attrs['duration_s'] = phase_info['duration_s']
        
        # Store phase metrics
        metrics_group = run_group.create_group('phase_metrics')
        
        for phase_name, metrics in self.phase_metrics.items():
            phase_metrics_group = metrics_group.create_group(phase_name)
            
            # Store scalar metrics as attributes
            phase_metrics_group.attrs['n_bouts'] = metrics.n_bouts
            phase_metrics_group.attrs['bout_rate_per_min'] = metrics.bout_rate_per_min
            phase_metrics_group.attrs['total_distance_px'] = metrics.total_distance_px
            phase_metrics_group.attrs['total_distance_mm'] = metrics.total_distance_mm if metrics.total_distance_mm else 0
            phase_metrics_group.attrs['mean_bout_duration_s'] = metrics.mean_bout_duration_s
            phase_metrics_group.attrs['std_bout_duration_s'] = metrics.std_bout_duration_s
            phase_metrics_group.attrs['mean_bout_distance_px'] = metrics.mean_bout_distance_px
            phase_metrics_group.attrs['mean_bout_distance_mm'] = metrics.mean_bout_distance_mm if metrics.mean_bout_distance_mm else 0
            phase_metrics_group.attrs['mean_speed_px_s'] = metrics.mean_speed_px_s
            phase_metrics_group.attrs['mean_speed_mm_s'] = metrics.mean_speed_mm_s if metrics.mean_speed_mm_s else 0
            phase_metrics_group.attrs['mean_ibi_s'] = metrics.mean_ibi_s
            phase_metrics_group.attrs['percent_active'] = metrics.percent_active
            
            # Store array data
            if metrics.bout_durations:
                phase_metrics_group.array('bout_durations', 
                                         np.array(metrics.bout_durations), 
                                         overwrite=True)
            if metrics.bout_distances:
                phase_metrics_group.array('bout_distances_px', 
                                         np.array(metrics.bout_distances), 
                                         overwrite=True)
                if self.calibration:
                    phase_metrics_group.array('bout_distances_mm', 
                                             np.array(metrics.bout_distances) * self.calibration['pixel_to_mm'], 
                                             overwrite=True)
            if metrics.bout_speeds:
                phase_metrics_group.array('bout_speeds_px_s', 
                                         np.array(metrics.bout_speeds), 
                                         overwrite=True)
                if self.calibration:
                    phase_metrics_group.array('bout_speeds_mm_s', 
                                             np.array(metrics.bout_speeds) * self.calibration['pixel_to_mm'], 
                                             overwrite=True)
            if metrics.ibis:
                phase_metrics_group.array('inter_bout_intervals_s', 
                                         np.array(metrics.ibis), 
                                         overwrite=True)
        
        # Store summary statistics across phases
        summary_stats = self._calculate_summary_statistics()
        run_group.attrs['summary_statistics'] = summary_stats
        
        # Update 'latest' attribute to point to this run
        main_group.attrs['latest'] = f'runs/{run_name}'
        
        # Also store 'best' if this is the first run or update criteria
        if 'best' not in main_group.attrs:
            main_group.attrs['best'] = {
                'run_name': f'runs/{run_name}',
                'criteria': 'most_recent',
                'timestamp': timestamp
            }
        
        run_group.attrs['environment'] = get_analysis_environment_info()
        
        if self.verbose:
            print(f"\nResults saved to zarr:")
            print(f"  Group: /{group_name}/runs/{run_name}")
            print(f"  Latest run: {run_name}")
            print(f"  Parameters preserved: {list(run_group.attrs['parameters'].keys())}")
            print(f"  Phases analyzed: {list(self.phase_metrics.keys())}")
        
        return f"{group_name}/runs/{run_name}"
    
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics across all phases."""
        stats = {
            'total_bouts_all_phases': sum(m.n_bouts for m in self.phase_metrics.values()),
            'phase_comparison': {}
        }
        
        # Compare phases
        phase_names = list(self.phase_metrics.keys())
        for i, phase1 in enumerate(phase_names):
            metrics1 = self.phase_metrics[phase1]
            stats['phase_comparison'][phase1] = {
                'n_bouts': metrics1.n_bouts,
                'bout_rate_per_min': metrics1.bout_rate_per_min,
                'percent_active': metrics1.percent_active
            }
            
            # Calculate relative changes from pre-training
            if i > 0 and 'pre_training' in self.phase_metrics:
                pre_metrics = self.phase_metrics['pre_training']
                if pre_metrics.n_bouts > 0:
                    stats['phase_comparison'][phase1]['bout_rate_change_from_pre'] = (
                        (metrics1.bout_rate_per_min - pre_metrics.bout_rate_per_min) / 
                        pre_metrics.bout_rate_per_min * 100
                    )
                    stats['phase_comparison'][phase1]['activity_change_from_pre'] = (
                        metrics1.percent_active - pre_metrics.percent_active
                    )
        
        return stats
    
    @staticmethod
    def load_from_zarr(zarr_path: str, run_name: Optional[str] = None):
        """
        Load previously saved phase bout analysis from zarr.
        
        Args:
            zarr_path: Path to zarr file
            run_name: Specific run to load (default: latest)
        
        Returns:
            Dictionary with loaded analysis results
        """
        root = zarr.open(str(zarr_path), mode='r')
        
        if 'bout_phase_analysis' not in root:
            raise ValueError("No bout_phase_analysis found in zarr")
        
        analysis_group = root['bout_phase_analysis']
        
        if run_name is None:
            if 'latest' in analysis_group.attrs:
                run_name = analysis_group.attrs['latest']
            else:
                raise ValueError("No 'latest' run found")
        
        run_group = analysis_group[run_name]
        
        # Load all data
        results = {
            'parameters': dict(run_group.attrs['parameters']),
            'data_source': dict(run_group.attrs['data_source']),
            'calibration': dict(run_group.attrs['calibration']),
            'summary_statistics': dict(run_group.attrs['summary_statistics']),
            'phases': {},
            'phase_metrics': {}
        }
        
        # Load phase boundaries
        for phase_name in run_group['phases'].keys():
            phase_group = run_group['phases'][phase_name]
            results['phases'][phase_name] = dict(phase_group.attrs)
        
        # Load phase metrics
        for phase_name in run_group['phase_metrics'].keys():
            metrics_group = run_group['phase_metrics'][phase_name]
            phase_data = dict(metrics_group.attrs)
            
            # Load arrays if present
            for array_name in metrics_group.arrays():
                phase_data[array_name] = metrics_group[array_name][:]
            
            results['phase_metrics'][phase_name] = phase_data
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze swimming bouts across training phases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default phases (5min pre, 2.5min training, 5min post)
  %(prog)s detections.zarr
  
  # With H5 file for precise phase markers
  %(prog)s detections.zarr --h5 experiment.h5
  
  # Custom phase durations
  %(prog)s detections.zarr --pre-duration 240 --training-duration 120 --post-duration 240
  
  # Save to zarr with full provenance
  %(prog)s detections.zarr --save-zarr
  
  # Complete analysis with all outputs
  %(prog)s detections.zarr --h5 experiment.h5 --save-zarr --output-plot phases.png --output-csv phases.csv --stats
  
  # Load previous analysis from zarr
  python -c "from bout_phase_analyzer import TrainingPhaseBoutAnalyzer; results = TrainingPhaseBoutAnalyzer.load_from_zarr('detections.zarr'); print(results['summary_statistics'])"
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with tracking data')
    parser.add_argument('--h5', dest='h5_path', help='Path to H5 file with event markers')
    parser.add_argument('--pre-duration', type=float, default=300,
                       help='Pre-training duration in seconds (default: 300)')
    parser.add_argument('--training-duration', type=float, default=150,
                       help='Training duration in seconds (default: 150)')
    parser.add_argument('--post-duration', type=float, default=300,
                       help='Post-training duration in seconds (default: 300)')
    parser.add_argument('--threshold-bl', type=float, default=0.5,
                       help='Speed threshold in body lengths/second (default: 0.5)')
    parser.add_argument('--min-bout-duration', type=float, default=0.05,
                       help='Minimum bout duration in seconds (default: 0.05)')
    parser.add_argument('--output-plot', help='Path to save comparison plot')
    parser.add_argument('--output-csv', help='Path to save metrics CSV')
    parser.add_argument('--save-zarr', action='store_true',
                       help='Save results to zarr with full provenance tracking')
    parser.add_argument('--stats', action='store_true',
                       help='Perform statistical comparisons between phases')
    parser.add_argument('--load-run', help='Load specific previous run from zarr instead of analyzing')
    
    args = parser.parse_args()
    
    # Check if loading previous results
    if args.load_run:
        print("Loading previous analysis from zarr...")
        results = TrainingPhaseBoutAnalyzer.load_from_zarr(args.zarr_path, args.load_run)
        print(f"Loaded run: {args.load_run}")
        print(f"Parameters: {results['parameters']}")
        print(f"Summary: {results['summary_statistics']}")
        return 0
    
    # Set up phase durations
    phase_durations = {
        'pre_training': args.pre_duration,
        'training': args.training_duration,
        'post_training': args.post_duration
    }
    
    # Initialize analyzer
    analyzer = TrainingPhaseBoutAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        phase_durations=phase_durations,
        speed_threshold_bl_s=args.threshold_bl,
        min_bout_duration_s=args.min_bout_duration,
        verbose=True
    )
    
    # Save to zarr if requested
    if args.save_zarr:
        saved_path = analyzer.save_to_zarr()
        print(f"\n✓ Analysis saved to zarr: {saved_path}")
        print(f"  To reload: TrainingPhaseBoutAnalyzer.load_from_zarr('{args.zarr_path}')")
    
    # Create comparison plot
    analyzer.plot_phase_comparison(save_path=args.output_plot)
    
    # Export metrics if requested
    if args.output_csv:
        analyzer.export_metrics(args.output_csv)
    
    # Perform statistical comparisons if requested
    if args.stats:
        stats_df = analyzer.statistical_comparison()
        print("\n" + "="*70)
        print("STATISTICAL COMPARISONS")
        print("="*70)
        print(stats_df.to_string())
    
    return 0


if __name__ == '__main__':
    sys.exit(main())