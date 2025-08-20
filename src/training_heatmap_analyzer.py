#!/usr/bin/env python3
"""
Training Period Heatmap Analyzer

Generates spatial heatmaps of fish positions before and after training periods,
using event markers from H5 files to identify training phases.
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Event type mappings from your stimulus program
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START", 1: "PROTOCOL_STOP", 2: "PROTOCOL_PAUSE", 3: "PROTOCOL_RESUME", 4: "PROTOCOL_FINISH",
    5: "PROTOCOL_CLEAR", 6: "PROTOCOL_LOAD", 7: "STEP_ADD", 8: "STEP_REMOVE", 9: "STEP_MOVE_UP",
    10: "STEP_MOVE_DOWN", 11: "STEP_START", 12: "STEP_END", 13: "ITI_START", 14: "ITI_END",
    15: "PARAMS_APPLIED", 16: "MANAGER_REINIT", 17: "MANAGER_REINIT_FAIL", 18: "LOOM_AUTO_REPEAT_TRIGGER",
    19: "LOOM_MANUAL_START", 20: "USER_INTERVENTION", 21: "ERROR_RUNTIME", 22: "LOG_MESSAGE",
    23: "IPC_BOUNDING_BOX_RECEIVED", 24: "CHASER_PRE_PERIOD_START", 25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START", 27: "CHASER_CHASE_SEQUENCE_START", 28: "CHASER_CHASE_SEQUENCE_END",
    29: "CHASER_RANDOM_TARGET_SET"
}

# Reverse mapping for easy lookup
EVENT_NAME_TO_ID = {v: k for k, v in EXPERIMENT_EVENT_TYPE.items()}


class TrainingHeatmapAnalyzer:
    """
    Analyzes fish behavior before and after training periods using spatial heatmaps.
    """
    
    def __init__(self, 
                 zarr_path: str, 
                 h5_path: str,
                 interpolation_run: Optional[str] = None,
                 bin_size: int = 50,
                 verbose: bool = True):
        """
        Initialize the heatmap analyzer.
        
        Args:
            zarr_path: Path to zarr file with fish positions
            h5_path: Path to H5 file with events and metadata
            interpolation_run: Specific interpolation run to use
            bin_size: Size of bins for heatmap (pixels)
            verbose: Enable verbose logging
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.interpolation_run = interpolation_run
        self.bin_size = bin_size
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Load data
        self.zarr_root = zarr.open(str(zarr_path), mode='r')
        
        # Get video dimensions
        self.video_width = self.zarr_root.attrs.get('width', 4512)
        self.video_height = self.zarr_root.attrs.get('height', 4512)
        self.fps = self.zarr_root.attrs.get('fps', 60.0)
        
        # Data containers
        self.events = None
        self.frame_metadata = None
        self.training_periods = []
        self.fish_positions = None
        self.chaser_positions = None
        
    def _setup_logging(self):
        """Configure logging."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_events(self):
        """Load and parse events from H5 file."""
        self.logger.info("Loading events from H5 file...")
        
        with h5py.File(self.h5_path, 'r') as h5f:
            if '/events' not in h5f:
                raise ValueError("No /events dataset found in H5 file")
            
            events_data = h5f['/events'][:]
            
            # Parse events
            self.events = []
            for event in events_data:
                self.events.append({
                    'timestamp_ns_epoch': int(event['timestamp_ns_epoch']),
                    'timestamp_ns_session': int(event['timestamp_ns_session']),
                    'event_type_id': int(event['event_type_id']),
                    'event_type': EXPERIMENT_EVENT_TYPE.get(event['event_type_id'], 'UNKNOWN'),
                    'stimulus_frame': int(event['current_step_index']),
                    'name_or_context': event['name_or_context'].decode('utf-8').strip('\x00'),
                    'details_json': event['details_json'].decode('utf-8').strip('\x00')
                })
            
            self.logger.info(f"  Loaded {len(self.events)} events")
            
            # Load frame metadata for timing alignment
            if '/video_metadata/frame_metadata' in h5f:
                self.frame_metadata = h5f['/video_metadata/frame_metadata'][:]
                self.logger.info(f"  Loaded {len(self.frame_metadata)} frame metadata records")
    
    def identify_training_periods(self):
        """Identify pre-training, training, and post-training periods from events."""
        self.logger.info("Identifying training periods...")
        
        # Find key events
        protocol_start = None
        pre_period_start = None
        training_start = None
        post_period_start = None
        protocol_finish = None
        
        for event in self.events:
            event_type = event['event_type']
            timestamp = event['timestamp_ns_session']
            
            if event_type == 'PROTOCOL_START':
                protocol_start = timestamp
                self.logger.info(f"  Protocol start: {timestamp/1e9:.2f}s")
            elif event_type == 'CHASER_PRE_PERIOD_START':
                pre_period_start = timestamp
                self.logger.info(f"  Pre-period start: {timestamp/1e9:.2f}s")
            elif event_type == 'CHASER_TRAINING_START':
                training_start = timestamp
                self.logger.info(f"  Training start: {timestamp/1e9:.2f}s")
            elif event_type == 'CHASER_POST_PERIOD_START':
                post_period_start = timestamp
                self.logger.info(f"  Post-period start: {timestamp/1e9:.2f}s")
            elif event_type == 'PROTOCOL_FINISH':
                protocol_finish = timestamp
                self.logger.info(f"  Protocol finish: {timestamp/1e9:.2f}s")
        
        # Define periods
        self.training_periods = {}
        
        if protocol_start is not None and training_start is not None:
            # Pre-training: from protocol start to training start
            self.training_periods['pre_training'] = {
                'start_ns': protocol_start,
                'end_ns': training_start,
                'duration_s': (training_start - protocol_start) / 1e9
            }
            self.logger.info(f"  Pre-training period: {self.training_periods['pre_training']['duration_s']:.1f}s")
        
        if training_start is not None and post_period_start is not None:
            # Training: from training start to post-period start
            self.training_periods['training'] = {
                'start_ns': training_start,
                'end_ns': post_period_start,
                'duration_s': (post_period_start - training_start) / 1e9
            }
            self.logger.info(f"  Training period: {self.training_periods['training']['duration_s']:.1f}s")
        
        if post_period_start is not None:
            # Post-training: from post-period start to end
            end_time = protocol_finish if protocol_finish else self.events[-1]['timestamp_ns_session']
            self.training_periods['post_training'] = {
                'start_ns': post_period_start,
                'end_ns': end_time,
                'duration_s': (end_time - post_period_start) / 1e9
            }
            self.logger.info(f"  Post-training period: {self.training_periods['post_training']['duration_s']:.1f}s")
        
        if not self.training_periods:
            self.logger.warning("  Could not identify clear training periods!")
            # Fallback: split data in half
            total_duration = self.events[-1]['timestamp_ns_session'] - self.events[0]['timestamp_ns_session']
            mid_point = self.events[0]['timestamp_ns_session'] + total_duration // 2
            
            self.training_periods['pre_training'] = {
                'start_ns': self.events[0]['timestamp_ns_session'],
                'end_ns': mid_point,
                'duration_s': total_duration / 2e9
            }
            self.training_periods['post_training'] = {
                'start_ns': mid_point,
                'end_ns': self.events[-1]['timestamp_ns_session'],
                'duration_s': total_duration / 2e9
            }
            self.logger.info("  Using fallback: splitting data at midpoint")
    
    def load_positions(self):
        """Load fish and chaser positions from zarr file."""
        self.logger.info("Loading position data...")
        
        # Check if we have chaser comparison data
        if 'chaser_comparison' in self.zarr_root:
            # Use the analyzed positions
            run_name = self.interpolation_run or self.zarr_root['chaser_comparison'].attrs.get('latest', 'original')
            
            if run_name in self.zarr_root['chaser_comparison']:
                analysis = self.zarr_root[f'chaser_comparison/{run_name}']
                self.fish_positions = analysis['fish_position_camera'][:]
                self.chaser_positions = analysis['chaser_position_camera'][:]
                self.logger.info(f"  Loaded positions from chaser_comparison/{run_name}")
            else:
                self.logger.warning(f"  Run {run_name} not found in chaser_comparison")
                self._load_raw_positions()
        else:
            self._load_raw_positions()
    
    def _load_raw_positions(self):
        """Load raw fish positions from detection data."""
        self.logger.info("  Loading raw detection positions...")
        
        # Get detection data
        if self.interpolation_run and f'interpolation_runs/{self.interpolation_run}' in self.zarr_root:
            data_path = f'interpolation_runs/{self.interpolation_run}'
        else:
            data_path = ''
        
        if data_path:
            bboxes = self.zarr_root[f'{data_path}/bboxes'][:]
            n_detections = self.zarr_root[f'{data_path}/n_detections'][:]
        else:
            bboxes = self.zarr_root['bboxes'][:]
            n_detections = self.zarr_root['n_detections'][:]
        
        # Calculate centers
        self.fish_positions = np.full((len(bboxes), 2), np.nan)
        for i in range(len(bboxes)):
            if n_detections[i] > 0:
                bbox = bboxes[i, 0]
                self.fish_positions[i, 0] = (bbox[0] + bbox[2]) / 2
                self.fish_positions[i, 1] = (bbox[1] + bbox[3]) / 2
        
        self.logger.info(f"  Calculated fish positions from bounding boxes")
    
    def map_frames_to_periods(self):
        """Map frame indices to training periods based on timestamps."""
        self.logger.info("Mapping frames to training periods...")
        
        # Use proportional mapping based on durations
        # This is more robust when exact timestamp alignment is tricky
        
        n_frames = len(self.fish_positions)
        total_duration_s = sum(p['duration_s'] for p in self.training_periods.values())
        fps = self.fps
        
        self.logger.info(f"  Total frames: {n_frames}")
        self.logger.info(f"  Total duration: {total_duration_s:.1f}s")
        self.logger.info(f"  Estimated FPS: {fps}")
        
        frame_mapping = {}
        current_frame = 0
        
        # Map frames proportionally based on period durations
        for period_name, period_info in self.training_periods.items():
            # Calculate how many frames should be in this period
            period_duration = period_info['duration_s']
            period_frames = int((period_duration / total_duration_s) * n_frames)
            
            # Handle last period to include any remaining frames
            if period_name == list(self.training_periods.keys())[-1]:
                period_frames = n_frames - current_frame
            
            # Assign frame indices
            frames_in_period = list(range(current_frame, min(current_frame + period_frames, n_frames)))
            frame_mapping[period_name] = frames_in_period
            
            self.logger.info(f"  {period_name}: {len(frames_in_period)} frames "
                           f"({period_duration:.1f}s @ {fps} fps)")
            
            current_frame += period_frames
        
        return frame_mapping
    
    def create_heatmap(self, positions: np.ndarray, title: str = "Position Heatmap") -> np.ndarray:
        """
        Create a 2D heatmap from position data.
        
        Args:
            positions: Nx2 array of (x, y) positions
            title: Title for the heatmap
            
        Returns:
            2D heatmap array
        """
        # Remove NaN values
        valid_positions = positions[~np.isnan(positions[:, 0])]
        
        if len(valid_positions) == 0:
            self.logger.warning(f"  No valid positions for {title}")
            return np.zeros((self.video_height // self.bin_size, 
                           self.video_width // self.bin_size))
        
        # Create 2D histogram
        x_bins = np.arange(0, self.video_width + self.bin_size, self.bin_size)
        y_bins = np.arange(0, self.video_height + self.bin_size, self.bin_size)
        
        heatmap, _, _ = np.histogram2d(
            valid_positions[:, 0], 
            valid_positions[:, 1],
            bins=[x_bins, y_bins]
        )
        
        # Apply Gaussian smoothing for better visualization
        heatmap = gaussian_filter(heatmap, sigma=1.5)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap.T  # Transpose for correct orientation
    
    def plot_heatmaps(self, save_path: Optional[str] = None):
        """Generate and plot heatmaps for different training periods."""
        self.logger.info("Generating heatmaps...")
        
        # Map frames to periods
        frame_mapping = self.map_frames_to_periods()
        
        # Determine which periods to plot
        periods_to_plot = []
        if 'pre_training' in frame_mapping and len(frame_mapping['pre_training']) > 0:
            periods_to_plot.append('pre_training')
        if 'training' in frame_mapping and len(frame_mapping['training']) > 0:
            periods_to_plot.append('training')
        if 'post_training' in frame_mapping and len(frame_mapping['post_training']) > 0:
            periods_to_plot.append('post_training')
        
        if len(periods_to_plot) < 2:
            self.logger.warning("Not enough periods with data to create comparison")
            periods_to_plot = list(frame_mapping.keys())[:2]
        
        # Create figure
        n_periods = len(periods_to_plot)
        fig, axes = plt.subplots(2, n_periods, figsize=(6*n_periods, 12))
        
        if n_periods == 1:
            axes = axes.reshape(2, 1)
        
        # Color map
        cmap = plt.cm.hot
        
        for col, period_name in enumerate(periods_to_plot):
            frames = frame_mapping.get(period_name, [])
            
            if not frames:
                continue
            
            # Get positions for this period
            period_fish_positions = self.fish_positions[frames]
            
            # Create fish heatmap
            fish_heatmap = self.create_heatmap(
                period_fish_positions, 
                f"Fish - {period_name.replace('_', ' ').title()}"
            )
            
            # Plot fish heatmap
            ax1 = axes[0, col]
            im1 = ax1.imshow(fish_heatmap, cmap=cmap, aspect='equal', 
                            extent=[0, self.video_width, self.video_height, 0])
            ax1.set_title(f"Fish - {period_name.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1, label='Occupancy')
            
            # Add period info
            period_info = self.training_periods.get(period_name, {})
            duration = period_info.get('duration_s', 0)
            ax1.text(0.02, 0.98, f"Duration: {duration:.1f}s\nFrames: {len(frames)}", 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Create chaser heatmap if available
            if self.chaser_positions is not None:
                period_chaser_positions = self.chaser_positions[frames]
                chaser_heatmap = self.create_heatmap(
                    period_chaser_positions,
                    f"Chaser - {period_name.replace('_', ' ').title()}"
                )
                
                ax2 = axes[1, col]
                im2 = ax2.imshow(chaser_heatmap, cmap=plt.cm.cool, aspect='equal',
                               extent=[0, self.video_width, self.video_height, 0])
                ax2.set_title(f"Chaser - {period_name.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                ax2.set_xlabel('X (pixels)')
                ax2.set_ylabel('Y (pixels)')
                plt.colorbar(im2, ax=ax2, label='Occupancy')
            else:
                # Create difference heatmap if comparing pre/post
                if col == 1 and periods_to_plot[0] in ['pre_training', 'training']:
                    # Calculate difference between periods
                    prev_frames = frame_mapping[periods_to_plot[0]]
                    prev_positions = self.fish_positions[prev_frames]
                    prev_heatmap = self.create_heatmap(prev_positions, "Previous")
                    
                    diff_heatmap = fish_heatmap - prev_heatmap
                    
                    ax2 = axes[1, col]
                    im2 = ax2.imshow(diff_heatmap, cmap='RdBu_r', aspect='equal',
                                   extent=[0, self.video_width, self.video_height, 0],
                                   vmin=-1, vmax=1)
                    ax2.set_title(f"Change from {periods_to_plot[0].replace('_', ' ').title()}", 
                                fontsize=12, fontweight='bold')
                    ax2.set_xlabel('X (pixels)')
                    ax2.set_ylabel('Y (pixels)')
                    plt.colorbar(im2, ax=ax2, label='Change in Occupancy')
                else:
                    axes[1, col].axis('off')
        
        plt.suptitle('Spatial Occupancy Analysis Across Training Periods', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"  Saved heatmap to: {save_path}")
        
        plt.show()
    
    def calculate_spatial_metrics(self) -> Dict:
        """Calculate spatial metrics for each training period."""
        self.logger.info("Calculating spatial metrics...")
        
        frame_mapping = self.map_frames_to_periods()
        metrics = {}
        
        for period_name, frames in frame_mapping.items():
            if not frames:
                continue
            
            period_positions = self.fish_positions[frames]
            valid_positions = period_positions[~np.isnan(period_positions[:, 0])]
            
            if len(valid_positions) == 0:
                continue
            
            # Calculate metrics
            period_metrics = {
                'mean_x': float(np.mean(valid_positions[:, 0])),
                'mean_y': float(np.mean(valid_positions[:, 1])),
                'std_x': float(np.std(valid_positions[:, 0])),
                'std_y': float(np.std(valid_positions[:, 1])),
                'min_x': float(np.min(valid_positions[:, 0])),
                'max_x': float(np.max(valid_positions[:, 0])),
                'min_y': float(np.min(valid_positions[:, 1])),
                'max_y': float(np.max(valid_positions[:, 1])),
                'total_frames': len(frames),
                'valid_frames': len(valid_positions),
                'coverage': len(valid_positions) / len(frames) if len(frames) > 0 else 0
            }
            
            # Calculate area explored (using convex hull)
            from scipy.spatial import ConvexHull
            if len(valid_positions) > 3:
                try:
                    hull = ConvexHull(valid_positions)
                    period_metrics['explored_area'] = float(hull.volume)  # In 2D, volume is area
                except:
                    period_metrics['explored_area'] = 0
            
            # Calculate total distance traveled
            if len(valid_positions) > 1:
                distances = np.sqrt(np.sum(np.diff(valid_positions, axis=0)**2, axis=1))
                period_metrics['total_distance'] = float(np.sum(distances))
                period_metrics['mean_speed'] = float(np.mean(distances) * self.fps)
            
            metrics[period_name] = period_metrics
            
            # Log summary
            self.logger.info(f"  {period_name}:")
            self.logger.info(f"    Mean position: ({period_metrics['mean_x']:.1f}, {period_metrics['mean_y']:.1f})")
            self.logger.info(f"    Spread (std): ({period_metrics['std_x']:.1f}, {period_metrics['std_y']:.1f})")
            if 'total_distance' in period_metrics:
                self.logger.info(f"    Total distance: {period_metrics['total_distance']:.1f} pixels")
            if 'explored_area' in period_metrics:
                self.logger.info(f"    Explored area: {period_metrics['explored_area']:.1f} sq pixels")
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate a text report of the spatial analysis."""
        report = []
        report.append("=" * 80)
        report.append("TRAINING PERIOD SPATIAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # File information
        report.append("DATA SOURCES")
        report.append("-" * 40)
        report.append(f"Zarr file: {self.zarr_path}")
        report.append(f"H5 file: {self.h5_path}")
        report.append(f"Video dimensions: {self.video_width}x{self.video_height}")
        report.append(f"Frame rate: {self.fps} FPS")
        report.append("")
        
        # Training periods
        report.append("TRAINING PERIODS")
        report.append("-" * 40)
        for period_name, period_info in self.training_periods.items():
            report.append(f"{period_name.replace('_', ' ').title()}:")
            report.append(f"  Duration: {period_info['duration_s']:.1f} seconds")
            report.append(f"  Start: {period_info['start_ns']/1e9:.1f}s")
            report.append(f"  End: {period_info['end_ns']/1e9:.1f}s")
        report.append("")
        
        # Spatial metrics
        metrics = self.calculate_spatial_metrics()
        if metrics:
            report.append("SPATIAL METRICS")
            report.append("-" * 40)
            
            for period_name, period_metrics in metrics.items():
                report.append(f"\n{period_name.replace('_', ' ').title()}:")
                report.append(f"  Frames analyzed: {period_metrics.get('valid_frames', 0)}/{period_metrics.get('total_frames', 0)}")
                report.append(f"  Mean position: ({period_metrics.get('mean_x', 0):.1f}, {period_metrics.get('mean_y', 0):.1f})")
                report.append(f"  Position spread (std): ({period_metrics.get('std_x', 0):.1f}, {period_metrics.get('std_y', 0):.1f})")
                
                if 'total_distance' in period_metrics:
                    report.append(f"  Total distance traveled: {period_metrics['total_distance']:.1f} pixels")
                if 'mean_speed' in period_metrics:
                    report.append(f"  Mean speed: {period_metrics['mean_speed']:.1f} pixels/second")
                if 'explored_area' in period_metrics:
                    report.append(f"  Explored area: {period_metrics['explored_area']:.1f} square pixels")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Command-line interface for the heatmap analyzer."""
    parser = argparse.ArgumentParser(
        description='Generate spatial heatmaps for pre/post training periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detections.zarr out_analysis.h5
  %(prog)s detections.zarr out_analysis.h5 --save-plot heatmaps.png
  %(prog)s detections.zarr out_analysis.h5 --bin-size 100
  %(prog)s detections.zarr out_analysis.h5 --interpolation-run interp_linear_20240120
        """
    )
    
    parser.add_argument(
        'zarr_path',
        help='Path to zarr file with detection/position data'
    )
    parser.add_argument(
        'h5_path',
        help='Path to H5 file with events and metadata'
    )
    parser.add_argument(
        '--interpolation-run',
        help='Specific interpolation run to use'
    )
    parser.add_argument(
        '--bin-size',
        type=int,
        default=50,
        help='Bin size for heatmap in pixels (default: 50)'
    )
    parser.add_argument(
        '--save-plot',
        help='Path to save heatmap figure'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Print detailed report'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TrainingHeatmapAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        interpolation_run=args.interpolation_run,
        bin_size=args.bin_size,
        verbose=not args.quiet
    )
    
    # Load data
    analyzer.load_events()
    analyzer.identify_training_periods()
    analyzer.load_positions()
    
    # Generate heatmaps
    analyzer.plot_heatmaps(save_path=args.save_plot)
    
    # Generate report if requested
    if args.report:
        print("\n" + analyzer.generate_report())
    
    return 0


if __name__ == '__main__':
    exit(main())