#!/usr/bin/env python3
"""
Unified Chaser-Fish Distance Analyzer

Combines data from:
- Multi-fish tracker zarr (fish positions from video analysis)
- H5 experiment file (chaser positions and chase events)

Handles:
- Frame alignment between 60Hz camera and 120Hz stimulus
- Coordinate transformation from texture space (358×358) to camera space (4512×4512)
- Distance calculations and behavioral metrics
- Comprehensive visualization
"""

import zarr
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ChaserFishData:
    """Container for aligned chaser and fish position data."""
    frame_numbers: np.ndarray  # Camera frame numbers
    timestamps: np.ndarray  # Time in seconds
    fish_x: np.ndarray  # Fish X positions (camera space)
    fish_y: np.ndarray  # Fish Y positions (camera space)
    chaser_x: np.ndarray  # Chaser X positions (camera space)
    chaser_y: np.ndarray  # Chaser Y positions (camera space)
    distances: np.ndarray  # Frame-by-frame distances
    fish_interpolated: np.ndarray  # Boolean mask for interpolated fish positions
    chase_events: List[Dict]  # Chase event markers
    metadata: Dict  # Additional metadata


class UnifiedDistanceAnalyzer:
    """Analyze fish-chaser distances combining zarr and H5 data."""
    
    def __init__(self, zarr_path: str, h5_path: str, 
                 source: str = 'latest', verbose: bool = True):
        """
        Initialize analyzer with both data sources.
        
        Args:
            zarr_path: Path to multi-fish tracker zarr
            h5_path: Path to experiment H5 file
            source: Which zarr data to use ('latest', 'preprocessing', etc.)
            verbose: Print progress messages
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.verbose = verbose
        
        # Coordinate transformation parameters
        self.texture_width = 358
        self.texture_height = 358
        self.camera_width = 4512
        self.camera_height = 4512
        self.texture_to_camera_scale = self.camera_width / self.texture_width  # ~12.604
        
        # Load data from both sources
        self._load_zarr_data(source)
        self._load_h5_data()
        
        # Align and calculate distances
        self.aligned_data = self._align_and_calculate()
        
        if verbose:
            self._print_summary()
    
    def _load_zarr_data(self, source: str):
        """Load fish tracking data from zarr."""
        self.zarr_root = zarr.open(str(self.zarr_path), mode='r')
        
        # Get FPS and calibration
        self.fps = self.zarr_root.attrs.get('fps', 60.0)
        self.pixel_to_mm = None
        if 'calibration' in self.zarr_root:
            self.pixel_to_mm = self.zarr_root['calibration'].attrs.get('pixel_to_mm')
        
        # Determine data source
        if source == 'latest':
            if 'preprocessing' in self.zarr_root and self.zarr_root['preprocessing'].attrs.get('latest'):
                source_path = 'preprocessing/' + self.zarr_root['preprocessing'].attrs['latest']
            elif 'filtered_runs' in self.zarr_root and self.zarr_root['filtered_runs'].attrs.get('latest'):
                source_path = 'filtered_runs/' + self.zarr_root['filtered_runs'].attrs['latest']
            else:
                source_path = 'detect_runs/' + self.zarr_root['detect_runs'].attrs['latest']
        else:
            source_path = source
        
        if self.verbose:
            print(f"Loading fish data from: {source_path}")
        
        # Load detection data
        data_group = self.zarr_root[source_path]
        self.n_detections = data_group['n_detections'][:]
        self.bbox_coords = data_group['bbox_norm_coords'][:]
        
        # Check for interpolation mask
        self.interp_mask = None
        if 'interpolation_mask' in data_group:
            self.interp_mask = data_group['interpolation_mask'][:]
        
        # Extract fish positions frame by frame
        self.fish_positions = []
        cumulative = np.cumsum(np.insert(self.n_detections, 0, 0))
        
        for frame_idx in range(len(self.n_detections)):
            if self.n_detections[frame_idx] > 0:
                bbox_idx = cumulative[frame_idx]
                bbox = self.bbox_coords[bbox_idx]
                # Convert normalized coords to camera pixels
                center_x = bbox[0] * self.camera_width
                center_y = bbox[1] * self.camera_height
                is_interpolated = self.interp_mask[bbox_idx] if self.interp_mask is not None else False
                self.fish_positions.append({
                    'frame': frame_idx,
                    'x': center_x,
                    'y': center_y,
                    'interpolated': is_interpolated
                })
    
    def _load_h5_data(self):
        """Load chaser positions and events from H5."""
        with h5py.File(self.h5_path, 'r') as hf:
            # Try multiple possible locations for chaser states
            chaser_paths = [
                '/tracking_data/chaser_states',
                '/analysis/chaser_states',
                '/chaser_states'
            ]
            
            chaser_data = None
            for path in chaser_paths:
                if path in hf:
                    chaser_data = hf[path][:]
                    if self.verbose:
                        print(f"Found chaser states at: {path}")
                    break
            
            if chaser_data is None:
                # List what's actually in the file for debugging
                print("\nAvailable H5 groups/datasets:")
                def print_structure(name, obj):
                    print(f"  {name}")
                hf.visititems(print_structure)
                raise ValueError("No chaser states found in H5 file")
            
            self.chaser_states = pd.DataFrame(chaser_data)
            
            # Load frame metadata for alignment
            frame_paths = [
                '/video_metadata/frame_metadata',
                '/frame_metadata',
                '/metadata/frame_metadata'
            ]
            
            self.frame_metadata = None
            for path in frame_paths:
                if path in hf:
                    self.frame_metadata = pd.DataFrame(hf[path][:])
                    if self.verbose:
                        print(f"Found frame metadata at: {path}")
                    break
            
            # Load chase events
            self.chase_events = []
            event_paths = [
                '/events',
                '/analysis/experiment_events',
                '/experiment_events'
            ]
            
            for path in event_paths:
                if path in hf:
                    events = hf[path][:]
                    if self.verbose:
                        print(f"Found events at: {path}")
                    
                    # Check for different event structures
                    if 'event_type_id' in events.dtype.names:
                        # New structure
                        for event in events:
                            if event['event_type_id'] in [27, 28]:  # Chase start/end
                                self.chase_events.append({
                                    'frame': event['camera_frame_id'] if 'camera_frame_id' in event.dtype.names else -1,
                                    'type': 'start' if event['event_type_id'] == 27 else 'end',
                                    'timestamp': event['timestamp_ns_epoch'] / 1e9 if 'timestamp_ns_epoch' in event.dtype.names else event['timestamp_ns_session'] / 1e9
                                })
                    elif 'event_id' in events.dtype.names:
                        # Old structure
                        for event in events:
                            if event['event_id'] in [27, 28]:
                                self.chase_events.append({
                                    'frame': event['camera_frame_num'] if 'camera_frame_num' in event.dtype.names else -1,
                                    'type': 'start' if event['event_id'] == 27 else 'end',
                                    'timestamp': event['timestamp_ns'] / 1e9 if 'timestamp_ns' in event.dtype.names else 0
                                })
                    break
            
        if self.verbose:
            print(f"Loaded {len(self.chaser_states)} chaser states")
            print(f"Found {len(self.chase_events)} chase events")
            
            # Print sample of chaser state columns to debug
            if len(self.chaser_states) > 0:
                print(f"Chaser state columns: {list(self.chaser_states.columns)[:10]}")
    
    def _align_and_calculate(self) -> ChaserFishData:
        """Align fish and chaser data, calculate distances."""
        # Prepare arrays for all camera frames
        total_frames = len(self.n_detections)
        frame_numbers = np.arange(total_frames)
        timestamps = frame_numbers / self.fps
        
        fish_x = np.full(total_frames, np.nan)
        fish_y = np.full(total_frames, np.nan)
        chaser_x = np.full(total_frames, np.nan)
        chaser_y = np.full(total_frames, np.nan)
        fish_interpolated = np.zeros(total_frames, dtype=bool)
        
        # Fill fish positions
        for pos in self.fish_positions:
            frame_idx = pos['frame']
            fish_x[frame_idx] = pos['x']
            fish_y[frame_idx] = pos['y']
            fish_interpolated[frame_idx] = pos['interpolated']
        
        # Fill chaser positions using frame metadata alignment
        if self.frame_metadata is not None:
            if self.verbose:
                print(f"Aligning {len(self.chaser_states)} chaser states with camera frames...")
            
            # For this H5 structure, use triggering_camera_frame_id
            for idx, row in self.frame_metadata.iterrows():
                stim_frame = int(row['stimulus_frame_num'])
                cam_frame = int(row['triggering_camera_frame_id'])  # Convert to int
                
                if cam_frame >= total_frames:
                    continue
                
                # Get chaser position for this stimulus frame
                chaser_mask = self.chaser_states['stimulus_frame_num'] == stim_frame
                if np.any(chaser_mask):
                    chaser = self.chaser_states[chaser_mask].iloc[0]
                    # Transform from texture to camera space
                    chaser_x[cam_frame] = chaser['chaser_pos_x'] * self.texture_to_camera_scale
                    chaser_y[cam_frame] = chaser['chaser_pos_y'] * self.texture_to_camera_scale
        else:
            if self.verbose:
                print("Warning: No frame metadata found, attempting direct stimulus frame mapping...")
            # Fallback: try to use stimulus frames directly (less accurate)
            for idx, row in self.chaser_states.iterrows():
                stim_frame = int(row['stimulus_frame_num'])
                # Approximate camera frame (assuming 2:1 ratio for 120Hz:60Hz)
                cam_frame = stim_frame // 2
                if cam_frame < total_frames:
                    chaser_x[cam_frame] = row['chaser_pos_x'] * self.texture_to_camera_scale
                    chaser_y[cam_frame] = row['chaser_pos_y'] * self.texture_to_camera_scale
        
        # Calculate distances
        distances = np.sqrt((fish_x - chaser_x)**2 + (fish_y - chaser_y)**2)
        
        metadata = {
            'zarr_source': str(self.zarr_path),
            'h5_source': str(self.h5_path),
            'fps': self.fps,
            'pixel_to_mm': self.pixel_to_mm,
            'texture_to_camera_scale': self.texture_to_camera_scale,
            'total_frames': total_frames,
            'valid_frames': np.sum(~np.isnan(distances))
        }
        
        return ChaserFishData(
            frame_numbers=frame_numbers,
            timestamps=timestamps,
            fish_x=fish_x,
            fish_y=fish_y,
            chaser_x=chaser_x,
            chaser_y=chaser_y,
            distances=distances,
            fish_interpolated=fish_interpolated,
            chase_events=self.chase_events,
            metadata=metadata
        )
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\n" + "=" * 60)
        print("CHASER-FISH DISTANCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        valid_distances = self.aligned_data.distances[~np.isnan(self.aligned_data.distances)]
        if len(valid_distances) > 0:
            print(f"\nDistance Statistics:")
            print(f"  Mean: {np.mean(valid_distances):.1f} pixels", end="")
            if self.pixel_to_mm:
                print(f" ({np.mean(valid_distances) * self.pixel_to_mm:.1f} mm)")
            else:
                print()
            
            print(f"  Median: {np.median(valid_distances):.1f} pixels")
            print(f"  Min: {np.min(valid_distances):.1f} pixels")
            print(f"  Max: {np.max(valid_distances):.1f} pixels")
            
            print(f"\nCoverage:")
            print(f"  Valid frames: {len(valid_distances)} / {len(self.aligned_data.distances)}")
            print(f"  Coverage: {len(valid_distances)/len(self.aligned_data.distances)*100:.1f}%")
            
            n_interp = np.sum(self.aligned_data.fish_interpolated)
            if n_interp > 0:
                print(f"  Interpolated: {n_interp} frames ({n_interp/len(valid_distances)*100:.1f}%)")
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive distance and movement metrics."""
        data = self.aligned_data
        valid_mask = ~np.isnan(data.distances)
        
        metrics = {
            'mean_distance': np.mean(data.distances[valid_mask]) if np.any(valid_mask) else np.nan,
            'median_distance': np.median(data.distances[valid_mask]) if np.any(valid_mask) else np.nan,
            'min_distance': np.min(data.distances[valid_mask]) if np.any(valid_mask) else np.nan,
            'max_distance': np.max(data.distances[valid_mask]) if np.any(valid_mask) else np.nan,
            'std_distance': np.std(data.distances[valid_mask]) if np.any(valid_mask) else np.nan,
        }
        
        # Calculate velocities
        if np.sum(valid_mask) > 1:
            # Fish velocity
            fish_dx = np.diff(data.fish_x)
            fish_dy = np.diff(data.fish_y)
            fish_speed = np.sqrt(fish_dx**2 + fish_dy**2) * self.fps
            
            # Relative velocity (negative = approaching)
            distance_diff = np.diff(data.distances)
            relative_velocity = distance_diff * self.fps
            
            metrics.update({
                'mean_fish_speed': np.nanmean(fish_speed),
                'max_fish_speed': np.nanmax(fish_speed),
                'mean_relative_velocity': np.nanmean(relative_velocity),
                'approach_events': np.sum(relative_velocity < -50)  # Rapid approaches
            })
        
        # Add chase event metrics
        if self.chase_events:
            chase_distances = []
            for event in self.chase_events:
                if event['type'] == 'start':
                    frame = event['frame']
                    if frame < len(data.distances):
                        chase_distances.append(data.distances[frame])
            
            if chase_distances:
                metrics['mean_distance_at_chase_start'] = np.nanmean(chase_distances)
        
        return metrics
    
    def plot_analysis(self, save_path: Optional[str] = None):
        """Create comprehensive visualization of chaser-fish distances."""
        data = self.aligned_data
        
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Distance over time
        ax1 = fig.add_subplot(gs[0, :])
        valid_mask = ~np.isnan(data.distances)
        ax1.plot(data.timestamps[valid_mask], data.distances[valid_mask], 
                'b-', alpha=0.6, linewidth=0.5, label='Distance')
        
        # Mark chase events
        for event in data.chase_events:
            frame = event['frame']
            if frame < len(data.timestamps):
                color = 'green' if event['type'] == 'start' else 'red'
                ax1.axvline(x=data.timestamps[frame], color=color, alpha=0.5, 
                           linestyle='--', label=f"Chase {event['type']}")
        
        # Mark interpolated regions
        if np.any(data.fish_interpolated):
            interp_times = data.timestamps[data.fish_interpolated]
            interp_dists = data.distances[data.fish_interpolated]
            ax1.scatter(interp_times, interp_dists, color='orange', s=2, 
                       alpha=0.5, label='Interpolated')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Distance (pixels)')
        ax1.set_title('Fish-Chaser Distance Over Time')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectories
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Plot fish trajectory
        valid_fish = ~(np.isnan(data.fish_x) | np.isnan(data.fish_y))
        scatter = ax2.scatter(data.fish_x[valid_fish], data.fish_y[valid_fish],
                            c=data.frame_numbers[valid_fish], cmap='viridis',
                            s=1, alpha=0.5, label='Fish')
        
        # Plot chaser trajectory
        valid_chaser = ~(np.isnan(data.chaser_x) | np.isnan(data.chaser_y))
        ax2.plot(data.chaser_x[valid_chaser], data.chaser_y[valid_chaser],
                'r-', alpha=0.3, linewidth=0.5, label='Chaser')
        
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_title('Spatial Trajectories')
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distance distribution
        ax3 = fig.add_subplot(gs[1, 1])
        valid_distances = data.distances[~np.isnan(data.distances)]
        if len(valid_distances) > 0:
            ax3.hist(valid_distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(np.mean(valid_distances), color='red', linestyle='--',
                       label=f'Mean: {np.mean(valid_distances):.1f}')
            ax3.axvline(np.median(valid_distances), color='green', linestyle='--',
                       label=f'Median: {np.median(valid_distances):.1f}')
            ax3.set_xlabel('Distance (pixels)')
            ax3.set_ylabel('Count')
            ax3.set_title('Distance Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Speed analysis
        ax4 = fig.add_subplot(gs[1, 2])
        if len(data.distances) > 1:
            # Calculate speeds
            fish_dx = np.diff(data.fish_x)
            fish_dy = np.diff(data.fish_y)
            fish_speed = np.sqrt(fish_dx**2 + fish_dy**2) * self.fps
            
            valid_speed = ~np.isnan(fish_speed)
            ax4.plot(data.timestamps[1:][valid_speed], fish_speed[valid_speed],
                    'b-', alpha=0.5, linewidth=0.5)
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Fish Speed (pixels/s)')
            ax4.set_title('Fish Swimming Speed')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Relative velocity
        ax5 = fig.add_subplot(gs[2, 0])
        if len(data.distances) > 1:
            relative_velocity = np.diff(data.distances) * self.fps
            valid_vel = ~np.isnan(relative_velocity)
            
            ax5.plot(data.timestamps[1:][valid_vel], relative_velocity[valid_vel],
                    'purple', alpha=0.5, linewidth=0.5)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.fill_between(data.timestamps[1:][valid_vel], 
                            relative_velocity[valid_vel], 0,
                            where=(relative_velocity[valid_vel] < 0),
                            color='red', alpha=0.3, label='Approaching')
            ax5.fill_between(data.timestamps[1:][valid_vel],
                            relative_velocity[valid_vel], 0,
                            where=(relative_velocity[valid_vel] > 0),
                            color='green', alpha=0.3, label='Escaping')
            ax5.set_xlabel('Time (seconds)')
            ax5.set_ylabel('Relative Velocity (pixels/s)')
            ax5.set_title('Approach/Escape Dynamics')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('off')
        
        metrics = self.calculate_metrics()
        
        summary_text = "ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
        summary_text += f"Distance Statistics:\n"
        summary_text += f"  Mean: {metrics['mean_distance']:.1f} px"
        if self.pixel_to_mm:
            summary_text += f" ({metrics['mean_distance']*self.pixel_to_mm:.1f} mm)"
        summary_text += f"\n  Median: {metrics['median_distance']:.1f} px\n"
        summary_text += f"  Range: {metrics['min_distance']:.1f} - {metrics['max_distance']:.1f} px\n\n"
        
        summary_text += f"Movement Statistics:\n"
        if 'mean_fish_speed' in metrics:
            summary_text += f"  Mean fish speed: {metrics['mean_fish_speed']:.1f} px/s\n"
            summary_text += f"  Max fish speed: {metrics['max_fish_speed']:.1f} px/s\n"
            summary_text += f"  Approach events: {metrics.get('approach_events', 0)}\n\n"
        
        summary_text += f"Coverage:\n"
        summary_text += f"  Valid frames: {data.metadata['valid_frames']}/{data.metadata['total_frames']}\n"
        summary_text += f"  Coverage: {data.metadata['valid_frames']/data.metadata['total_frames']*100:.1f}%\n"
        
        if np.any(data.fish_interpolated):
            n_interp = np.sum(data.fish_interpolated)
            summary_text += f"  Interpolated: {n_interp} frames\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Main title
        fig.suptitle('Chaser-Fish Distance Analysis (Zarr + H5 Integration)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def save_to_zarr(self, overwrite: bool = False):
        """Save distance metrics to zarr file."""
        zarr_root = zarr.open(str(self.zarr_path), mode='r+')
        
        # Check if group exists
        if 'chaser_distance_analysis' in zarr_root and not overwrite:
            print("Error: chaser_distance_analysis already exists. Use --overwrite to replace.")
            return False
        
        # Create or overwrite group
        if 'chaser_distance_analysis' in zarr_root:
            del zarr_root['chaser_distance_analysis']
        
        analysis_group = zarr_root.create_group('chaser_distance_analysis')
        
        # Save metadata
        analysis_group.attrs['created_at'] = datetime.now().isoformat()
        analysis_group.attrs['h5_source'] = str(self.h5_path)
        analysis_group.attrs['texture_to_camera_scale'] = self.texture_to_camera_scale
        
        # Save data arrays
        data = self.aligned_data
        analysis_group.create_dataset('frame_numbers', data=data.frame_numbers)
        analysis_group.create_dataset('timestamps', data=data.timestamps)
        analysis_group.create_dataset('fish_x', data=data.fish_x, chunks=True, compression='gzip')
        analysis_group.create_dataset('fish_y', data=data.fish_y, chunks=True, compression='gzip')
        analysis_group.create_dataset('chaser_x', data=data.chaser_x, chunks=True, compression='gzip')
        analysis_group.create_dataset('chaser_y', data=data.chaser_y, chunks=True, compression='gzip')
        analysis_group.create_dataset('distances', data=data.distances, chunks=True, compression='gzip')
        analysis_group.create_dataset('fish_interpolated', data=data.fish_interpolated)
        
        # Save chase events
        if data.chase_events:
            events_data = np.array([(e['frame'], e['timestamp'], 1 if e['type']=='start' else 0) 
                                   for e in data.chase_events],
                                  dtype=[('frame', 'i4'), ('timestamp', 'f8'), ('is_start', 'i1')])
            analysis_group.create_dataset('chase_events', data=events_data)
        
        # Save summary metrics
        metrics = self.calculate_metrics()
        for key, value in metrics.items():
            if not np.isnan(value):
                analysis_group.attrs[key] = float(value)
        
        if self.verbose:
            print(f"✓ Analysis saved to {self.zarr_path}/chaser_distance_analysis")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Analyze fish-chaser distances using zarr and H5 data'
    )
    parser.add_argument('zarr_path', help='Path to multi-fish tracker zarr')
    parser.add_argument('h5_path', help='Path to experiment H5 file')
    parser.add_argument('--source', type=str, default='latest',
                       help='Zarr data source (latest, preprocessing, etc.)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate analysis plots')
    parser.add_argument('--save', action='store_true',
                       help='Save analysis to zarr')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing analysis')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNIFIED CHASER-FISH DISTANCE ANALYZER")
    print("=" * 60)
    print(f"Zarr: {args.zarr_path}")
    print(f"H5: {args.h5_path}")
    
    try:
        # Initialize analyzer
        analyzer = UnifiedDistanceAnalyzer(
            args.zarr_path,
            args.h5_path,
            source=args.source,
            verbose=True
        )
        
        # Save to zarr if requested
        if args.save:
            analyzer.save_to_zarr(overwrite=args.overwrite)
        
        # Generate plots if requested
        if args.plot or args.output_plot:
            analyzer.plot_analysis(save_path=args.output_plot)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())