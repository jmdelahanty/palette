# #!/usr/bin/env python3
# """
# Training Period Heatmap Analyzer

# Generates spatial heatmaps of fish positions before and after training periods,
# using event markers from H5 files to identify training phases.

# Updated to work with the preprocessing pipeline (filtered/interpolated data).
# """

# import zarr
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from scipy.ndimage import gaussian_filter
# import argparse
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Tuple, Optional
# import json
# import logging

# # Event type mappings from your stimulus program
# EXPERIMENT_EVENT_TYPE = {
#     0: "PROTOCOL_START", 1: "PROTOCOL_STOP", 2: "PROTOCOL_PAUSE", 3: "PROTOCOL_RESUME", 4: "PROTOCOL_FINISH",
#     5: "PROTOCOL_CLEAR", 6: "PROTOCOL_LOAD", 7: "STEP_ADD", 8: "STEP_REMOVE", 9: "STEP_MOVE_UP",
#     10: "STEP_MOVE_DOWN", 11: "STEP_START", 12: "STEP_END", 13: "ITI_START", 14: "ITI_END",
#     15: "PARAMS_APPLIED", 16: "MANAGER_REINIT", 17: "MANAGER_REINIT_FAIL", 18: "LOOM_AUTO_REPEAT_TRIGGER",
#     19: "LOOM_MANUAL_START", 20: "USER_INTERVENTION", 21: "ERROR_RUNTIME", 22: "LOG_MESSAGE",
#     23: "IPC_BOUNDING_BOX_RECEIVED", 24: "CHASER_PRE_PERIOD_START", 25: "CHASER_TRAINING_START",
#     26: "CHASER_POST_PERIOD_START", 27: "CHASER_CHASE_SEQUENCE_START", 28: "CHASER_CHASE_SEQUENCE_END",
#     29: "CHASER_RANDOM_TARGET_SET"
# }

# # Reverse mapping for easy lookup
# EVENT_NAME_TO_ID = {v: k for k, v in EXPERIMENT_EVENT_TYPE.items()}


# class TrainingHeatmapAnalyzer:
#     """
#     Analyzes fish behavior before and after training periods using spatial heatmaps.
#     """
    
#     def __init__(self, 
#                  zarr_path: str, 
#                  h5_path: str,
#                  use_preprocessed: bool = True,
#                  bin_size: int = 50,
#                  verbose: bool = True):
#         """
#         Initialize the heatmap analyzer.
        
#         Args:
#             zarr_path: Path to zarr file with fish positions
#             h5_path: Path to H5 file with events and metadata
#             use_preprocessed: Use preprocessed data if available (filtered/interpolated)
#             bin_size: Size of bins for heatmap (pixels)
#             verbose: Enable verbose logging
#         """
#         self.zarr_path = Path(zarr_path)
#         self.h5_path = Path(h5_path)
#         self.use_preprocessed = use_preprocessed
#         self.bin_size = bin_size
#         self.verbose = verbose
        
#         # Setup logging
#         self._setup_logging()
        
#         # Load zarr data
#         self.zarr_root = zarr.open(str(zarr_path), mode='r')
        
#         # Get video dimensions
#         self.video_width = self.zarr_root.attrs.get('width', 4512)
#         self.video_height = self.zarr_root.attrs.get('height', 4512)
#         self.fps = self.zarr_root.attrs.get('fps', 60.0)
        
#         # Data containers
#         self.events = None
#         self.frame_metadata = None
#         self.training_periods = {}
#         self.fish_positions = None
#         self.interpolation_mask = None
#         self.data_source = "original"
        
#     def _setup_logging(self):
#         """Configure logging."""
#         level = logging.INFO if self.verbose else logging.WARNING
#         logging.basicConfig(
#             level=level,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             datefmt='%H:%M:%S'
#         )
#         self.logger = logging.getLogger(__name__)
    
#     def load_events(self):
#         """Load and parse events from H5 file."""
#         self.logger.info("Loading events from H5 file...")
        
#         with h5py.File(self.h5_path, 'r') as h5f:
#             if '/events' not in h5f:
#                 raise ValueError("No /events dataset found in H5 file")
            
#             events_data = h5f['/events'][:]
            
#             # Parse events
#             self.events = []
#             for event in events_data:
#                 parsed_event = {
#                     'timestamp_ns_epoch': int(event['timestamp_ns_epoch']),
#                     'timestamp_ns_session': int(event['timestamp_ns_session']),
#                     'event_type_id': int(event['event_type_id']),
#                     'event_type': EXPERIMENT_EVENT_TYPE.get(event['event_type_id'], 'UNKNOWN'),
#                     'stimulus_frame': int(event['stimulus_frame_num']) if 'stimulus_frame_num' in event.dtype.names else 0,
#                     'name_or_context': event['name_or_context'].decode('utf-8', errors='ignore').strip('\x00'),
#                     'details_json': event['details_json'].decode('utf-8', errors='ignore').strip('\x00')
#                 }
                
#                 # Add camera_frame_id if available
#                 if 'camera_frame_id' in event.dtype.names:
#                     parsed_event['camera_frame_id'] = int(event['camera_frame_id'])
                
#                 self.events.append(parsed_event)
            
#             self.logger.info(f"  Loaded {len(self.events)} events")
            
#             # Load frame metadata for timing alignment
#             if '/video_metadata/frame_metadata' in h5f:
#                 self.frame_metadata = h5f['/video_metadata/frame_metadata'][:]
#                 self.logger.info(f"  Loaded {len(self.frame_metadata)} frame metadata records")
    
#     def identify_training_periods(self):
#         """Identify pre-training, training, and post-training periods from events."""
#         self.logger.info("Identifying training periods...")
        
#         # Find key events
#         protocol_start = None
#         pre_period_start = None
#         training_start = None
#         post_period_start = None
#         protocol_finish = None
        
#         # Also track camera frames if available
#         protocol_start_frame = None
#         training_start_frame = None
#         post_period_start_frame = None
#         protocol_finish_frame = None
        
#         for event in self.events:
#             event_type = event['event_type']
#             timestamp = event['timestamp_ns_session']
#             camera_frame = event.get('camera_frame_id', None)
            
#             if event_type == 'PROTOCOL_START':
#                 protocol_start = timestamp
#                 protocol_start_frame = camera_frame
#                 self.logger.info(f"  Protocol start: {timestamp/1e9:.2f}s (frame {camera_frame})")
#             elif event_type == 'CHASER_PRE_PERIOD_START':
#                 pre_period_start = timestamp
#                 self.logger.info(f"  Pre-period start: {timestamp/1e9:.2f}s")
#             elif event_type == 'CHASER_TRAINING_START':
#                 training_start = timestamp
#                 training_start_frame = camera_frame
#                 self.logger.info(f"  Training start: {timestamp/1e9:.2f}s (frame {camera_frame})")
#             elif event_type == 'CHASER_POST_PERIOD_START':
#                 post_period_start = timestamp
#                 post_period_start_frame = camera_frame
#                 self.logger.info(f"  Post-period start: {timestamp/1e9:.2f}s (frame {camera_frame})")
#             elif event_type == 'PROTOCOL_FINISH':
#                 protocol_finish = timestamp
#                 protocol_finish_frame = camera_frame
#                 self.logger.info(f"  Protocol finish: {timestamp/1e9:.2f}s (frame {camera_frame})")
        
#         # Define periods
#         self.training_periods = {}
        
#         # Use camera frames if available, otherwise fall back to timestamps
#         use_frames = all(f is not None for f in [protocol_start_frame, training_start_frame, post_period_start_frame])
        
#         if use_frames and protocol_start_frame is not None and training_start_frame is not None:
#             # Pre-training period
#             self.training_periods['pre_training'] = {
#                 'start_frame': protocol_start_frame - 1,  # Convert to 0-based
#                 'end_frame': training_start_frame - 1,
#                 'start_ns': protocol_start,
#                 'end_ns': training_start,
#                 'duration_s': (training_start - protocol_start) / 1e9
#             }
#             self.logger.info(f"  Pre-training: frames {protocol_start_frame}-{training_start_frame}")
        
#         if use_frames and training_start_frame is not None and post_period_start_frame is not None:
#             # Training period (actual chase trials)
#             self.training_periods['training'] = {
#                 'start_frame': training_start_frame - 1,
#                 'end_frame': post_period_start_frame - 1,
#                 'start_ns': training_start,
#                 'end_ns': post_period_start,
#                 'duration_s': (post_period_start - training_start) / 1e9
#             }
#             self.logger.info(f"  Training: frames {training_start_frame}-{post_period_start_frame}")
        
#         if use_frames and post_period_start_frame is not None:
#             # Post-training period
#             end_frame = protocol_finish_frame if protocol_finish_frame else len(self.fish_positions)
#             self.training_periods['post_training'] = {
#                 'start_frame': post_period_start_frame - 1,
#                 'end_frame': end_frame - 1 if end_frame else len(self.fish_positions) - 1,
#                 'start_ns': post_period_start,
#                 'end_ns': protocol_finish if protocol_finish else self.events[-1]['timestamp_ns_session'],
#                 'duration_s': (protocol_finish - post_period_start) / 1e9 if protocol_finish else 0
#             }
#             self.logger.info(f"  Post-training: frames {post_period_start_frame}-{end_frame}")
        
#         if not self.training_periods:
#             self.logger.warning("  Could not identify clear training periods!")
#             # Fallback: use proportional splitting
#             self._fallback_period_identification()
    
#     def _fallback_period_identification(self):
#         """Fallback method to identify periods when events are unclear."""
#         self.logger.info("  Using fallback period identification...")
        
#         # Look for chase sequences to identify training period
#         chase_starts = [e for e in self.events if e['event_type'] == 'CHASER_CHASE_SEQUENCE_START']
#         chase_ends = [e for e in self.events if e['event_type'] == 'CHASER_CHASE_SEQUENCE_END']
        
#         if chase_starts and chase_ends:
#             # Training period = first chase to last chase
#             first_chase_frame = chase_starts[0].get('camera_frame_id', None)
#             last_chase_frame = chase_ends[-1].get('camera_frame_id', None)
            
#             if first_chase_frame and last_chase_frame:
#                 total_frames = len(self.fish_positions) if self.fish_positions is not None else 15000
                
#                 # Pre-training: start to first chase
#                 self.training_periods['pre_training'] = {
#                     'start_frame': 0,
#                     'end_frame': first_chase_frame - 1,
#                     'duration_s': (first_chase_frame / self.fps)
#                 }
                
#                 # Training: first chase to last chase
#                 self.training_periods['training'] = {
#                     'start_frame': first_chase_frame - 1,
#                     'end_frame': last_chase_frame - 1,
#                     'duration_s': ((last_chase_frame - first_chase_frame) / self.fps)
#                 }
                
#                 # Post-training: last chase to end
#                 self.training_periods['post_training'] = {
#                     'start_frame': last_chase_frame - 1,
#                     'end_frame': total_frames - 1,
#                     'duration_s': ((total_frames - last_chase_frame) / self.fps)
#                 }
                
#                 self.logger.info(f"  Identified periods from chase sequences")
#                 return
        
#         # Ultimate fallback: split in thirds
#         total_frames = len(self.fish_positions) if self.fish_positions is not None else 15000
#         third = total_frames // 3
        
#         self.training_periods = {
#             'pre_training': {'start_frame': 0, 'end_frame': third},
#             'training': {'start_frame': third, 'end_frame': 2*third},
#             'post_training': {'start_frame': 2*third, 'end_frame': total_frames-1}
#         }
#         self.logger.info("  Split data into thirds as last resort")
    
#     def load_positions(self):
#         """Load fish positions from zarr file using preprocessing pipeline."""
#         self.logger.info("Loading position data from zarr...")
        
#         # Determine which data to use
#         if self.use_preprocessed:
#             # Try preprocessed (interpolated) first
#             if 'preprocessing' in self.zarr_root and 'latest' in self.zarr_root['preprocessing'].attrs:
#                 source_path = self.zarr_root['preprocessing'].attrs['latest']
#                 source_group = self.zarr_root['preprocessing'][source_path]
#                 self.data_source = f"preprocessing/{source_path}"
#                 self.logger.info(f"  Using interpolated data: {source_path}")
                
#                 # Get interpolation mask if available
#                 if 'interpolation_mask' in source_group:
#                     self.interpolation_mask = source_group['interpolation_mask'][:]
                
#             # Try filtered data
#             elif 'filtered_runs' in self.zarr_root and 'latest' in self.zarr_root['filtered_runs'].attrs:
#                 source_name = self.zarr_root['filtered_runs'].attrs['latest']
#                 source_group = self.zarr_root['filtered_runs'][source_name]
#                 self.data_source = f"filtered_runs/{source_name}"
#                 self.logger.info(f"  Using filtered data: {source_name}")
                
#             else:
#                 source_group = self.zarr_root
#                 self.data_source = "original"
#                 self.logger.info("  Using original data")
#         else:
#             source_group = self.zarr_root
#             self.data_source = "original"
#             self.logger.info("  Using original data (as requested)")
        
#         # Load arrays
#         bboxes = source_group['bboxes'][:]
#         n_detections = source_group['n_detections'][:]
        
#         # Calculate centroids
#         self.fish_positions = np.full((len(bboxes), 2), np.nan)
#         for i in range(len(bboxes)):
#             if n_detections[i] > 0:
#                 bbox = bboxes[i, 0]
#                 self.fish_positions[i, 0] = (bbox[0] + bbox[2]) / 2  # X center
#                 self.fish_positions[i, 1] = (bbox[1] + bbox[3]) / 2  # Y center
        
#         # Calculate coverage
#         valid_frames = np.sum(~np.isnan(self.fish_positions[:, 0]))
#         coverage = valid_frames / len(self.fish_positions) * 100
#         self.logger.info(f"  Loaded {len(self.fish_positions)} frames")
#         self.logger.info(f"  Coverage: {coverage:.1f}% ({valid_frames}/{len(self.fish_positions)})")
        
#         if self.interpolation_mask is not None:
#             interp_count = np.sum(self.interpolation_mask)
#             self.logger.info(f"  Interpolated frames: {interp_count}")
    
#     def create_heatmap(self, positions: np.ndarray, title: str = "Position Heatmap") -> np.ndarray:
#         """
#         Create a 2D heatmap from position data.
        
#         Args:
#             positions: Nx2 array of (x, y) positions
#             title: Title for the heatmap
            
#         Returns:
#             2D heatmap array
#         """
#         # Remove NaN values
#         valid_positions = positions[~np.isnan(positions[:, 0])]
        
#         if len(valid_positions) == 0:
#             self.logger.warning(f"  No valid positions for {title}")
#             return np.zeros((self.video_height // self.bin_size, 
#                            self.video_width // self.bin_size))
        
#         # Create 2D histogram
#         x_bins = np.arange(0, self.video_width + self.bin_size, self.bin_size)
#         y_bins = np.arange(0, self.video_height + self.bin_size, self.bin_size)
        
#         heatmap, _, _ = np.histogram2d(
#             valid_positions[:, 0], 
#             valid_positions[:, 1],
#             bins=[x_bins, y_bins]
#         )
        
#         # Apply Gaussian smoothing for better visualization
#         heatmap = gaussian_filter(heatmap, sigma=1.5)
        
#         # Normalize
#         if heatmap.max() > 0:
#             heatmap = heatmap / heatmap.max()
        
#         return heatmap.T  # Transpose for correct orientation
    
#     def plot_heatmaps(self, save_path: Optional[str] = None):
#         """Generate and plot heatmaps for different training periods."""
#         self.logger.info("Generating heatmaps...")
        
#         # Create figure - show all three periods if available
#         periods = ['pre_training', 'training', 'post_training']
#         available_periods = [p for p in periods if p in self.training_periods]
        
#         if len(available_periods) < 2:
#             self.logger.warning("Not enough periods identified for comparison")
#             return
        
#         n_periods = len(available_periods)
#         fig, axes = plt.subplots(2, n_periods, figsize=(6*n_periods, 12))
        
#         if n_periods == 1:
#             axes = axes.reshape(2, 1)
        
#         # Color maps
#         cmap = plt.cm.hot
        
#         # Store heatmaps for difference calculation
#         heatmaps = {}
        
#         for col, period_name in enumerate(available_periods):
#             period_info = self.training_periods[period_name]
            
#             # Get frames for this period
#             start_frame = period_info.get('start_frame', 0)
#             end_frame = period_info.get('end_frame', len(self.fish_positions))
            
#             # Ensure valid range
#             start_frame = max(0, min(start_frame, len(self.fish_positions)-1))
#             end_frame = max(0, min(end_frame, len(self.fish_positions)-1))
            
#             if start_frame >= end_frame:
#                 self.logger.warning(f"  Invalid frame range for {period_name}")
#                 continue
            
#             # Get positions for this period
#             period_positions = self.fish_positions[start_frame:end_frame+1]
            
#             # Create heatmap
#             heatmap = self.create_heatmap(
#                 period_positions, 
#                 f"{period_name.replace('_', ' ').title()}"
#             )
#             heatmaps[period_name] = heatmap
            
#             # Plot main heatmap
#             ax1 = axes[0, col]
#             im1 = ax1.imshow(heatmap, cmap=cmap, aspect='equal', 
#                             extent=[0, self.video_width, self.video_height, 0],
#                             vmin=0, vmax=1)
#             ax1.set_title(f"{period_name.replace('_', ' ').title()}", 
#                          fontsize=12, fontweight='bold')
#             ax1.set_xlabel('X (pixels)')
#             ax1.set_ylabel('Y (pixels)')
#             plt.colorbar(im1, ax=ax1, label='Occupancy')
            
#             # Add period statistics
#             valid_positions = period_positions[~np.isnan(period_positions[:, 0])]
#             n_frames = end_frame - start_frame + 1
#             coverage = len(valid_positions) / n_frames * 100 if n_frames > 0 else 0
            
#             # Check interpolation in this period
#             interp_text = ""
#             if self.interpolation_mask is not None:
#                 period_interp = self.interpolation_mask[start_frame:end_frame+1]
#                 n_interp = np.sum(period_interp)
#                 if n_interp > 0:
#                     interp_text = f"\nInterpolated: {n_interp} frames"
            
#             info_text = (f"Frames: {n_frames}\n"
#                         f"Coverage: {coverage:.1f}%"
#                         f"{interp_text}")
            
#             ax1.text(0.02, 0.98, info_text, 
#                     transform=ax1.transAxes, verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#                     fontsize=9)
            
#             # Plot difference heatmap (compared to pre-training)
#             ax2 = axes[1, col]
#             if period_name == 'pre_training':
#                 # For pre-training, show the same heatmap
#                 im2 = ax2.imshow(heatmap, cmap=cmap, aspect='equal',
#                                extent=[0, self.video_width, self.video_height, 0],
#                                vmin=0, vmax=1)
#                 ax2.set_title("Baseline Period", fontsize=12)
#                 plt.colorbar(im2, ax=ax2, label='Occupancy')
#             else:
#                 # Show difference from pre-training
#                 if 'pre_training' in heatmaps:
#                     diff_heatmap = heatmap - heatmaps['pre_training']
                    
#                     # Use diverging colormap for differences
#                     max_diff = np.max(np.abs(diff_heatmap))
#                     im2 = ax2.imshow(diff_heatmap, cmap='RdBu_r', aspect='equal',
#                                    extent=[0, self.video_width, self.video_height, 0],
#                                    vmin=-max_diff, vmax=max_diff)
#                     ax2.set_title(f"Change from Pre-training", fontsize=12)
#                     plt.colorbar(im2, ax=ax2, label='Change in Occupancy')
                    
#                     # Add statistics about the change
#                     increased = np.sum(diff_heatmap > 0.1)
#                     decreased = np.sum(diff_heatmap < -0.1)
#                     change_text = f"↑ {increased} bins\n↓ {decreased} bins"
#                     ax2.text(0.02, 0.98, change_text,
#                             transform=ax2.transAxes, verticalalignment='top',
#                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
#                             fontsize=9)
#                 else:
#                     ax2.axis('off')
            
#             ax2.set_xlabel('X (pixels)')
#             ax2.set_ylabel('Y (pixels)')
        
#         # Add main title with data source info
#         title = f'Spatial Occupancy Analysis - {self.data_source}'
#         plt.suptitle(title, fontsize=14, fontweight='bold')
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             self.logger.info(f"  Saved heatmap to: {save_path}")
        
#         plt.show()
    
#     def calculate_spatial_metrics(self) -> Dict:
#         """Calculate spatial metrics for each training period."""
#         self.logger.info("Calculating spatial metrics...")
        
#         metrics = {}
        
#         for period_name, period_info in self.training_periods.items():
#             start_frame = period_info.get('start_frame', 0)
#             end_frame = period_info.get('end_frame', len(self.fish_positions))
            
#             # Ensure valid range
#             start_frame = max(0, min(start_frame, len(self.fish_positions)-1))
#             end_frame = max(0, min(end_frame, len(self.fish_positions)-1))
            
#             if start_frame >= end_frame:
#                 continue
            
#             period_positions = self.fish_positions[start_frame:end_frame+1]
#             valid_positions = period_positions[~np.isnan(period_positions[:, 0])]
            
#             if len(valid_positions) == 0:
#                 continue
            
#             # Calculate metrics
#             period_metrics = {
#                 'mean_x': float(np.mean(valid_positions[:, 0])),
#                 'mean_y': float(np.mean(valid_positions[:, 1])),
#                 'std_x': float(np.std(valid_positions[:, 0])),
#                 'std_y': float(np.std(valid_positions[:, 1])),
#                 'min_x': float(np.min(valid_positions[:, 0])),
#                 'max_x': float(np.max(valid_positions[:, 0])),
#                 'min_y': float(np.min(valid_positions[:, 1])),
#                 'max_y': float(np.max(valid_positions[:, 1])),
#                 'total_frames': end_frame - start_frame + 1,
#                 'valid_frames': len(valid_positions),
#                 'coverage': len(valid_positions) / (end_frame - start_frame + 1) * 100
#             }
            
#             # Calculate area explored (using convex hull)
#             from scipy.spatial import ConvexHull
#             if len(valid_positions) > 3:
#                 try:
#                     hull = ConvexHull(valid_positions)
#                     period_metrics['explored_area'] = float(hull.volume)  # In 2D, volume is area
#                     period_metrics['explored_area_pct'] = float(hull.volume / (self.video_width * self.video_height) * 100)
#                 except:
#                     period_metrics['explored_area'] = 0
#                     period_metrics['explored_area_pct'] = 0
            
#             # Calculate total distance traveled
#             if len(valid_positions) > 1:
#                 distances = np.sqrt(np.sum(np.diff(valid_positions, axis=0)**2, axis=1))
#                 period_metrics['total_distance'] = float(np.sum(distances))
#                 period_metrics['mean_speed'] = float(np.mean(distances) * self.fps)
#                 period_metrics['max_speed'] = float(np.max(distances) * self.fps)
            
#             # Center preference (distance from arena center)
#             center_x, center_y = self.video_width / 2, self.video_height / 2
#             distances_from_center = np.sqrt((valid_positions[:, 0] - center_x)**2 + 
#                                            (valid_positions[:, 1] - center_y)**2)
#             period_metrics['mean_distance_from_center'] = float(np.mean(distances_from_center))
#             period_metrics['thigmotaxis_index'] = float(np.mean(distances_from_center) / (self.video_width / 2))
            
#             metrics[period_name] = period_metrics
            
#             # Log summary
#             self.logger.info(f"  {period_name}:")
#             self.logger.info(f"    Coverage: {period_metrics['coverage']:.1f}%")
#             self.logger.info(f"    Mean position: ({period_metrics['mean_x']:.1f}, {period_metrics['mean_y']:.1f})")
#             if 'total_distance' in period_metrics:
#                 self.logger.info(f"    Total distance: {period_metrics['total_distance']:.1f} pixels")
#                 self.logger.info(f"    Mean speed: {period_metrics['mean_speed']:.1f} px/s")
#             self.logger.info(f"    Thigmotaxis index: {period_metrics['thigmotaxis_index']:.2f}")
        
#         return metrics


# def main():
#     """Command-line interface for the heatmap analyzer."""
#     parser = argparse.ArgumentParser(
#         description='Generate spatial heatmaps for training periods',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# This tool creates spatial occupancy heatmaps for pre-training, training, and post-training periods.
# It works with the preprocessed data from your filtering/interpolation pipeline.

# Examples:
#   %(prog)s detections.zarr analysis.h5
#   %(prog)s detections.zarr analysis.h5 --save-plot heatmaps.png
#   %(prog)s detections.zarr analysis.h5 --use-original  # Skip preprocessing
#   %(prog)s detections.zarr analysis.h5 --bin-size 100
#         """
#     )
    
#     parser.add_argument('zarr_path', help='Path to zarr file with detection data')
#     parser.add_argument('h5_path', help='Path to H5 file with events')
    
#     parser.add_argument('--use-original', action='store_true',
#                        help='Use original data instead of preprocessed')
#     parser.add_argument('--bin-size', type=int, default=50,
#                        help='Bin size for heatmap in pixels (default: 50)')
#     parser.add_argument('--save-plot', help='Path to save heatmap figure')
#     parser.add_argument('--export-metrics', help='Export metrics to JSON file')
#     parser.add_argument('-q', '--quiet', action='store_true',
#                        help='Suppress verbose output')
    
#     args = parser.parse_args()
    
#     # Create analyzer
#     analyzer = TrainingHeatmapAnalyzer(
#         zarr_path=args.zarr_path,
#         h5_path=args.h5_path,
#         use_preprocessed=not args.use_original,
#         bin_size=args.bin_size,
#         verbose=not args.quiet
#     )
    
#     # Load data
#     analyzer.load_events()
#     analyzer.load_positions()
#     analyzer.identify_training_periods()
    
#     # Generate heatmaps
#     analyzer.plot_heatmaps(save_path=args.save_plot)
    
#     # Calculate and export metrics if requested
#     metrics = analyzer.calculate_spatial_metrics()
    
#     if args.export_metrics and metrics:
#         with open(args.export_metrics, 'w') as f:
#             json.dump({
#                 'analysis_date': datetime.now().isoformat(),
#                 'zarr_file': str(analyzer.zarr_path),
#                 'h5_file': str(analyzer.h5_path),
#                 'data_source': analyzer.data_source,
#                 'periods': metrics
#             }, f, indent=2)
#         print(f"\n✅ Exported metrics to: {args.export_metrics}")
    
#     return 0


# if __name__ == '__main__':
#     exit(main())

#!/usr/bin/env python3
"""
Training Period Heatmap Analyzer

Generates spatial heatmaps of fish positions before and after training periods,
using event markers from H5 files to identify training phases.

Updated to work with the preprocessing pipeline (filtered/interpolated data).
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
import json
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
                 use_preprocessed: bool = True,
                 bin_size: int = 50,
                 verbose: bool = True):
        """
        Initialize the heatmap analyzer.
        
        Args:
            zarr_path: Path to zarr file with fish positions
            h5_path: Path to H5 file with events and metadata
            use_preprocessed: Use preprocessed data if available (filtered/interpolated)
            bin_size: Size of bins for heatmap (pixels)
            verbose: Enable verbose logging
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.use_preprocessed = use_preprocessed
        self.bin_size = bin_size
        self.verbose = verbose
        
        # Setup logging
        self._setup_logging()
        
        # Load zarr data
        self.zarr_root = zarr.open(str(zarr_path), mode='r')
        
        # Get video dimensions
        self.video_width = self.zarr_root.attrs.get('width', 4512)
        self.video_height = self.zarr_root.attrs.get('height', 4512)
        self.fps = self.zarr_root.attrs.get('fps', 60.0)
        
        # Data containers
        self.events = None
        self.frame_metadata = None
        self.training_periods = {}
        self.fish_positions = None
        self.interpolation_mask = None
        self.data_source = "original"
        
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
                parsed_event = {
                    'timestamp_ns_epoch': int(event['timestamp_ns_epoch']),
                    'timestamp_ns_session': int(event['timestamp_ns_session']),
                    'event_type_id': int(event['event_type_id']),
                    'event_type': EXPERIMENT_EVENT_TYPE.get(event['event_type_id'], 'UNKNOWN'),
                    'stimulus_frame': int(event['stimulus_frame_num']) if 'stimulus_frame_num' in event.dtype.names else 0,
                    'name_or_context': event['name_or_context'].decode('utf-8', errors='ignore').strip('\x00'),
                    'details_json': event['details_json'].decode('utf-8', errors='ignore').strip('\x00')
                }
                
                # Add camera_frame_id if available
                if 'camera_frame_id' in event.dtype.names:
                    parsed_event['camera_frame_id'] = int(event['camera_frame_id'])
                
                self.events.append(parsed_event)
            
            self.logger.info(f"  Loaded {len(self.events)} events")
            
            # Load frame metadata for timing alignment
            if '/video_metadata/frame_metadata' in h5f:
                self.frame_metadata = h5f['/video_metadata/frame_metadata'][:]
                self.logger.info(f"  Loaded {len(self.frame_metadata)} frame metadata records")
    
    def identify_training_periods(self):
        """Identify pre-training, training, and post-training periods from events."""
        self.logger.info("Identifying training periods...")
        
        # Find key events with timestamps
        protocol_start = None
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
                # Pre-period start is often same as protocol start
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
        
        # Calculate frame indices based on time proportions
        # This is more reliable than using camera_frame_id which may not align
        total_frames = len(self.fish_positions) if self.fish_positions is not None else 15048
        
        if protocol_start is not None and protocol_finish is not None:
            total_duration = (protocol_finish - protocol_start) / 1e9  # in seconds
            
            # Calculate frame indices based on time offsets
            if training_start is not None:
                pre_duration = (training_start - protocol_start) / 1e9
                pre_frames = int((pre_duration / total_duration) * total_frames)
                
                self.training_periods['pre_training'] = {
                    'start_frame': 0,
                    'end_frame': pre_frames,
                    'start_ns': protocol_start,
                    'end_ns': training_start,
                    'duration_s': pre_duration
                }
                self.logger.info(f"  Pre-training: frames 0-{pre_frames} ({pre_duration:.1f}s)")
            
            if training_start is not None and post_period_start is not None:
                training_duration = (post_period_start - training_start) / 1e9
                training_start_frame = self.training_periods['pre_training']['end_frame'] if 'pre_training' in self.training_periods else 0
                training_frames = int((training_duration / total_duration) * total_frames)
                training_end_frame = training_start_frame + training_frames
                
                self.training_periods['training'] = {
                    'start_frame': training_start_frame,
                    'end_frame': training_end_frame,
                    'start_ns': training_start,
                    'end_ns': post_period_start,
                    'duration_s': training_duration
                }
                self.logger.info(f"  Training: frames {training_start_frame}-{training_end_frame} ({training_duration:.1f}s)")
            
            if post_period_start is not None:
                post_duration = (protocol_finish - post_period_start) / 1e9
                post_start_frame = self.training_periods['training']['end_frame'] if 'training' in self.training_periods else int(total_frames * 0.67)
                
                self.training_periods['post_training'] = {
                    'start_frame': post_start_frame,
                    'end_frame': total_frames - 1,
                    'start_ns': post_period_start,
                    'end_ns': protocol_finish,
                    'duration_s': post_duration
                }
                self.logger.info(f"  Post-training: frames {post_start_frame}-{total_frames-1} ({post_duration:.1f}s)")
        
        elif training_start is not None and post_period_start is not None:
            # Fallback if protocol start/finish not available
            # Assume recording starts at protocol start
            self.logger.info("  Using timestamps to estimate frame ranges...")
            
            # Estimate based on 60fps
            pre_frames = int(training_start / 1e9 * self.fps)
            training_frames = int((post_period_start - training_start) / 1e9 * self.fps)
            
            self.training_periods['pre_training'] = {
                'start_frame': 0,
                'end_frame': pre_frames,
                'duration_s': training_start / 1e9
            }
            
            self.training_periods['training'] = {
                'start_frame': pre_frames,
                'end_frame': pre_frames + training_frames,
                'duration_s': (post_period_start - training_start) / 1e9
            }
            
            self.training_periods['post_training'] = {
                'start_frame': pre_frames + training_frames,
                'end_frame': total_frames - 1,
                'duration_s': (total_frames - pre_frames - training_frames) / self.fps
            }
            
        if not self.training_periods:
            self.logger.warning("  Could not identify clear training periods!")
            self._fallback_period_identification()
    
    def _fallback_period_identification(self):
        """Fallback method to identify periods when events are unclear."""
        self.logger.info("  Using fallback period identification...")
        
        # Look for chase sequences to identify training period
        chase_starts = [e for e in self.events if e['event_type'] == 'CHASER_CHASE_SEQUENCE_START']
        chase_ends = [e for e in self.events if e['event_type'] == 'CHASER_CHASE_SEQUENCE_END']
        
        if chase_starts and chase_ends:
            # Training period = first chase to last chase
            first_chase_frame = chase_starts[0].get('camera_frame_id', None)
            last_chase_frame = chase_ends[-1].get('camera_frame_id', None)
            
            if first_chase_frame and last_chase_frame:
                total_frames = len(self.fish_positions) if self.fish_positions is not None else 15000
                
                # Pre-training: start to first chase
                self.training_periods['pre_training'] = {
                    'start_frame': 0,
                    'end_frame': first_chase_frame - 1,
                    'duration_s': (first_chase_frame / self.fps)
                }
                
                # Training: first chase to last chase
                self.training_periods['training'] = {
                    'start_frame': first_chase_frame - 1,
                    'end_frame': last_chase_frame - 1,
                    'duration_s': ((last_chase_frame - first_chase_frame) / self.fps)
                }
                
                # Post-training: last chase to end
                self.training_periods['post_training'] = {
                    'start_frame': last_chase_frame - 1,
                    'end_frame': total_frames - 1,
                    'duration_s': ((total_frames - last_chase_frame) / self.fps)
                }
                
                self.logger.info(f"  Identified periods from chase sequences")
                return
        
        # Ultimate fallback: split in thirds
        total_frames = len(self.fish_positions) if self.fish_positions is not None else 15000
        third = total_frames // 3
        
        self.training_periods = {
            'pre_training': {'start_frame': 0, 'end_frame': third},
            'training': {'start_frame': third, 'end_frame': 2*third},
            'post_training': {'start_frame': 2*third, 'end_frame': total_frames-1}
        }
        self.logger.info("  Split data into thirds as last resort")
    
    def load_positions(self):
        """Load fish positions from zarr file using preprocessing pipeline."""
        self.logger.info("Loading position data from zarr...")
        
        # Determine which data to use
        if self.use_preprocessed:
            # Try preprocessed (interpolated) first
            if 'preprocessing' in self.zarr_root and 'latest' in self.zarr_root['preprocessing'].attrs:
                source_path = self.zarr_root['preprocessing'].attrs['latest']
                source_group = self.zarr_root['preprocessing'][source_path]
                self.data_source = f"preprocessing/{source_path}"
                self.logger.info(f"  Using interpolated data: {source_path}")
                
                # Get interpolation mask if available
                if 'interpolation_mask' in source_group:
                    self.interpolation_mask = source_group['interpolation_mask'][:]
                
            # Try filtered data
            elif 'filtered_runs' in self.zarr_root and 'latest' in self.zarr_root['filtered_runs'].attrs:
                source_name = self.zarr_root['filtered_runs'].attrs['latest']
                source_group = self.zarr_root['filtered_runs'][source_name]
                self.data_source = f"filtered_runs/{source_name}"
                self.logger.info(f"  Using filtered data: {source_name}")
                
            else:
                source_group = self.zarr_root
                self.data_source = "original"
                self.logger.info("  Using original data")
        else:
            source_group = self.zarr_root
            self.data_source = "original"
            self.logger.info("  Using original data (as requested)")
        
        # Load arrays
        bboxes = source_group['bboxes'][:]
        n_detections = source_group['n_detections'][:]
        
        # Calculate centroids
        self.fish_positions = np.full((len(bboxes), 2), np.nan)
        for i in range(len(bboxes)):
            if n_detections[i] > 0:
                bbox = bboxes[i, 0]
                self.fish_positions[i, 0] = (bbox[0] + bbox[2]) / 2  # X center
                self.fish_positions[i, 1] = (bbox[1] + bbox[3]) / 2  # Y center
        
        # Calculate coverage
        valid_frames = np.sum(~np.isnan(self.fish_positions[:, 0]))
        coverage = valid_frames / len(self.fish_positions) * 100
        self.logger.info(f"  Loaded {len(self.fish_positions)} frames")
        self.logger.info(f"  Coverage: {coverage:.1f}% ({valid_frames}/{len(self.fish_positions)})")
        
        if self.interpolation_mask is not None:
            interp_count = np.sum(self.interpolation_mask)
            self.logger.info(f"  Interpolated frames: {interp_count}")
    
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
        
        # Create figure - show all three periods if available
        periods = ['pre_training', 'training', 'post_training']
        available_periods = [p for p in periods if p in self.training_periods]
        
        if len(available_periods) < 2:
            self.logger.warning("Not enough periods identified for comparison")
            return
        
        n_periods = len(available_periods)
        fig, axes = plt.subplots(2, n_periods, figsize=(6*n_periods, 12))
        
        if n_periods == 1:
            axes = axes.reshape(2, 1)
        
        # Color maps
        cmap = plt.cm.hot
        
        # Store heatmaps for difference calculation
        heatmaps = {}
        
        for col, period_name in enumerate(available_periods):
            period_info = self.training_periods[period_name]
            
            # Get frames for this period
            start_frame = period_info.get('start_frame', 0)
            end_frame = period_info.get('end_frame', len(self.fish_positions))
            
            # Ensure valid range
            start_frame = max(0, min(start_frame, len(self.fish_positions)-1))
            end_frame = max(0, min(end_frame, len(self.fish_positions)-1))
            
            if start_frame >= end_frame:
                self.logger.warning(f"  Invalid frame range for {period_name}")
                continue
            
            # Get positions for this period
            period_positions = self.fish_positions[start_frame:end_frame+1]
            
            # Create heatmap
            heatmap = self.create_heatmap(
                period_positions, 
                f"{period_name.replace('_', ' ').title()}"
            )
            heatmaps[period_name] = heatmap
            
            # Plot main heatmap
            ax1 = axes[0, col]
            im1 = ax1.imshow(heatmap, cmap=cmap, aspect='equal', 
                            extent=[0, self.video_width, self.video_height, 0],
                            vmin=0, vmax=1)
            ax1.set_title(f"{period_name.replace('_', ' ').title()}", 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1, label='Occupancy')
            
            # Add period statistics
            valid_positions = period_positions[~np.isnan(period_positions[:, 0])]
            n_frames = end_frame - start_frame + 1
            coverage = len(valid_positions) / n_frames * 100 if n_frames > 0 else 0
            
            # Check interpolation in this period
            interp_text = ""
            if self.interpolation_mask is not None:
                period_interp = self.interpolation_mask[start_frame:end_frame+1]
                n_interp = np.sum(period_interp)
                if n_interp > 0:
                    interp_text = f"\nInterpolated: {n_interp} frames"
            
            info_text = (f"Frames: {n_frames}\n"
                        f"Coverage: {coverage:.1f}%"
                        f"{interp_text}")
            
            ax1.text(0.02, 0.98, info_text, 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            # Plot difference heatmap (compared to pre-training)
            ax2 = axes[1, col]
            if period_name == 'pre_training':
                # For pre-training, show the same heatmap
                im2 = ax2.imshow(heatmap, cmap=cmap, aspect='equal',
                               extent=[0, self.video_width, self.video_height, 0],
                               vmin=0, vmax=1)
                ax2.set_title("Baseline Period", fontsize=12)
                plt.colorbar(im2, ax=ax2, label='Occupancy')
            else:
                # Show difference from pre-training
                if 'pre_training' in heatmaps:
                    diff_heatmap = heatmap - heatmaps['pre_training']
                    
                    # Use diverging colormap for differences
                    max_diff = np.max(np.abs(diff_heatmap))
                    im2 = ax2.imshow(diff_heatmap, cmap='RdBu_r', aspect='equal',
                                   extent=[0, self.video_width, self.video_height, 0],
                                   vmin=-max_diff, vmax=max_diff)
                    ax2.set_title(f"Change from Pre-training", fontsize=12)
                    plt.colorbar(im2, ax=ax2, label='Change in Occupancy')
                    
                    # Add statistics about the change
                    increased = np.sum(diff_heatmap > 0.1)
                    decreased = np.sum(diff_heatmap < -0.1)
                    change_text = f"↑ {increased} bins\n↓ {decreased} bins"
                    ax2.text(0.02, 0.98, change_text,
                            transform=ax2.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=9)
                else:
                    ax2.axis('off')
            
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
        
        # Add main title with data source info
        title = f'Spatial Occupancy Analysis - {self.data_source}'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"  Saved heatmap to: {save_path}")
        
        plt.show()
    
    def calculate_spatial_metrics(self) -> Dict:
        """Calculate spatial metrics for each training period."""
        self.logger.info("Calculating spatial metrics...")
        
        metrics = {}
        
        for period_name, period_info in self.training_periods.items():
            start_frame = period_info.get('start_frame', 0)
            end_frame = period_info.get('end_frame', len(self.fish_positions))
            
            # Ensure valid range
            start_frame = max(0, min(start_frame, len(self.fish_positions)-1))
            end_frame = max(0, min(end_frame, len(self.fish_positions)-1))
            
            if start_frame >= end_frame:
                continue
            
            period_positions = self.fish_positions[start_frame:end_frame+1]
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
                'total_frames': end_frame - start_frame + 1,
                'valid_frames': len(valid_positions),
                'coverage': len(valid_positions) / (end_frame - start_frame + 1) * 100
            }
            
            # Calculate area explored (using convex hull)
            from scipy.spatial import ConvexHull
            if len(valid_positions) > 3:
                try:
                    hull = ConvexHull(valid_positions)
                    period_metrics['explored_area'] = float(hull.volume)  # In 2D, volume is area
                    period_metrics['explored_area_pct'] = float(hull.volume / (self.video_width * self.video_height) * 100)
                except:
                    period_metrics['explored_area'] = 0
                    period_metrics['explored_area_pct'] = 0
            
            # Calculate total distance traveled
            if len(valid_positions) > 1:
                distances = np.sqrt(np.sum(np.diff(valid_positions, axis=0)**2, axis=1))
                period_metrics['total_distance'] = float(np.sum(distances))
                period_metrics['mean_speed'] = float(np.mean(distances) * self.fps)
                period_metrics['max_speed'] = float(np.max(distances) * self.fps)
            
            # Center preference (distance from arena center)
            center_x, center_y = self.video_width / 2, self.video_height / 2
            distances_from_center = np.sqrt((valid_positions[:, 0] - center_x)**2 + 
                                           (valid_positions[:, 1] - center_y)**2)
            period_metrics['mean_distance_from_center'] = float(np.mean(distances_from_center))
            period_metrics['thigmotaxis_index'] = float(np.mean(distances_from_center) / (self.video_width / 2))
            
            metrics[period_name] = period_metrics
            
            # Log summary
            self.logger.info(f"  {period_name}:")
            self.logger.info(f"    Coverage: {period_metrics['coverage']:.1f}%")
            self.logger.info(f"    Mean position: ({period_metrics['mean_x']:.1f}, {period_metrics['mean_y']:.1f})")
            if 'total_distance' in period_metrics:
                self.logger.info(f"    Total distance: {period_metrics['total_distance']:.1f} pixels")
                self.logger.info(f"    Mean speed: {period_metrics['mean_speed']:.1f} px/s")
            self.logger.info(f"    Thigmotaxis index: {period_metrics['thigmotaxis_index']:.2f}")
        
        return metrics


def main():
    """Command-line interface for the heatmap analyzer."""
    parser = argparse.ArgumentParser(
        description='Generate spatial heatmaps for training periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool creates spatial occupancy heatmaps for pre-training, training, and post-training periods.
It works with the preprocessed data from your filtering/interpolation pipeline.

Examples:
  %(prog)s detections.zarr analysis.h5
  %(prog)s detections.zarr analysis.h5 --save-plot heatmaps.png
  %(prog)s detections.zarr analysis.h5 --use-original  # Skip preprocessing
  %(prog)s detections.zarr analysis.h5 --bin-size 100
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with detection data')
    parser.add_argument('h5_path', help='Path to H5 file with events')
    
    parser.add_argument('--use-original', action='store_true',
                       help='Use original data instead of preprocessed')
    parser.add_argument('--bin-size', type=int, default=50,
                       help='Bin size for heatmap in pixels (default: 50)')
    parser.add_argument('--save-plot', help='Path to save heatmap figure')
    parser.add_argument('--export-metrics', help='Export metrics to JSON file')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TrainingHeatmapAnalyzer(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        use_preprocessed=not args.use_original,
        bin_size=args.bin_size,
        verbose=not args.quiet
    )
    
    # Load data
    analyzer.load_events()
    analyzer.load_positions()
    analyzer.identify_training_periods()
    
    # Generate heatmaps
    analyzer.plot_heatmaps(save_path=args.save_plot)
    
    # Calculate and export metrics if requested
    metrics = analyzer.calculate_spatial_metrics()
    
    if args.export_metrics and metrics:
        with open(args.export_metrics, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'zarr_file': str(analyzer.zarr_path),
                'h5_file': str(analyzer.h5_path),
                'data_source': analyzer.data_source,
                'periods': metrics
            }, f, indent=2)
        print(f"\n✅ Exported metrics to: {args.export_metrics}")
    
    return 0


if __name__ == '__main__':
    exit(main())