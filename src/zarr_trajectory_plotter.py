#!/usr/bin/env python3
"""
Zarr Trajectory Plotter

Plots fish trajectories directly from zarr detection data.
Supports single ROI, multiple ROIs, and various visualization options.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
from datetime import datetime
from rich.console import Console
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

console = Console()


class ZarrTrajectoryPlotter:
    """Plot trajectories from zarr detection data."""
    
    def __init__(self, zarr_path: str, verbose: bool = True, use_downsampled: bool = True):
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.root = zarr.open_group(self.zarr_path, mode='r')
        
        # Determine which coordinate system to use
        self.use_downsampled = use_downsampled
        
        if use_downsampled:
            # Use downsampled dimensions
            if 'raw_video/images_ds' in self.root:
                ds_shape = self.root['raw_video/images_ds'].shape
                self.img_width = ds_shape[2]
                self.img_height = ds_shape[1]
            else:
                self.img_width = 640
                self.img_height = 640
            console.print(f"[cyan]Using downsampled coordinates: {self.img_width}x{self.img_height}[/cyan]")
        else:
            # Use full resolution
            self.img_width = self.root.attrs.get('width', 4512)
            self.img_height = self.root.attrs.get('height', 4512)
            console.print(f"[cyan]Using full resolution: {self.img_width}x{self.img_height}[/cyan]")
        
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Try to load appropriate background image
        self.background = None
        if 'background_runs' in self.root:
            latest_bg = self.root['background_runs'].attrs.get('latest')
            if latest_bg:
                if use_downsampled:
                    # Load downsampled background
                    if f'background_runs/{latest_bg}/background_ds' in self.root:
                        self.background = self.root[f'background_runs/{latest_bg}/background_ds'][:]
                else:
                    # Load full resolution background
                    if f'background_runs/{latest_bg}/background_full' in self.root:
                        self.background = self.root[f'background_runs/{latest_bg}/background_full'][:]
                    elif f'background_runs/{latest_bg}/background_ds' in self.root:
                        bg_ds = self.root[f'background_runs/{latest_bg}/background_ds'][:]
                        import cv2
                        self.background = cv2.resize(bg_ds, (self.img_width, self.img_height))
                    
        if self.verbose and self.background is not None:
            console.print(f"[green]Background image loaded[/green]")
    
    def get_roi_positions(self, roi_id: int, use_interpolated: bool = True) -> Dict:
        """Get all positions for a specific ROI."""
        # Load detection data
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        n_detections = detect_group[latest_detect]['n_detections'][:]
        bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        detection_ids = id_group[latest_id]['detection_ids'][:]
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
        # IMPORTANT: The normalized coordinates are ALWAYS relative to the full 4512x4512 image
        # When displaying in downsampled space, we need to account for this
        full_width = self.root.attrs.get('width', 4512)
        full_height = self.root.attrs.get('height', 4512)
        
        # Extract positions
        positions = {}
        cumulative_idx = 0
        
        for frame_idx in range(len(n_detections)):
            frame_det_count = int(n_detections[frame_idx])
            
            if frame_det_count > 0 and n_detections_per_roi[frame_idx, roi_id] > 0:
                frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    bbox = bbox_coords[cumulative_idx + roi_idx]
                    
                    # Calculate centroid
                    # First convert to full resolution pixels
                    centroid_x_full = ((bbox[0] + bbox[2]) / 2) * full_width
                    centroid_y_full = ((bbox[1] + bbox[3]) / 2) * full_height
                    
                    if self.use_downsampled:
                        # Then scale down to 640x640 space
                        scale_down = 640.0 / full_width
                        centroid_x = centroid_x_full * scale_down
                        centroid_y = centroid_y_full * scale_down
                    else:
                        centroid_x = centroid_x_full
                        centroid_y = centroid_y_full
                    
                    positions[frame_idx] = np.array([centroid_x, centroid_y])
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions if available
        if use_interpolated and 'interpolated_detections' in self.root:
            interp_det_group = self.root['interpolated_detections']
            if 'latest' in interp_det_group.attrs:
                latest_det = interp_det_group.attrs['latest']
                det_group = interp_det_group[latest_det]
                
                frame_indices = det_group['frame_indices'][:]
                roi_ids = det_group['roi_ids'][:]
                bboxes = det_group['bboxes'][:]
                
                for i in range(len(frame_indices)):
                    if int(roi_ids[i]) == roi_id:
                        frame_idx = int(frame_indices[i])
                        bbox = bboxes[i]
                        
                        # Same conversion for interpolated positions
                        centroid_x_full = ((bbox[0] + bbox[2]) / 2) * full_width
                        centroid_y_full = ((bbox[1] + bbox[3]) / 2) * full_height
                        
                        if self.use_downsampled:
                            scale_down = 640.0 / full_width
                            centroid_x = centroid_x_full * scale_down
                            centroid_y = centroid_y_full * scale_down
                        else:
                            centroid_x = centroid_x_full
                            centroid_y = centroid_y_full
                        
                        positions[frame_idx] = np.array([centroid_x, centroid_y])
        
        return positions
    
    def plot_single_trajectory(self, roi_id: int, use_interpolated: bool = True,
                             show_points: bool = False, save_path: Optional[Path] = None):
        """Plot trajectory for a single ROI."""
        positions = self.get_roi_positions(roi_id, use_interpolated)
        
        if not positions:
            console.print(f"[yellow]No positions found for ROI {roi_id}[/yellow]")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Show background if available
        if self.background is not None:
            ax.imshow(self.background, cmap='gray', extent=[0, self.img_width, self.img_height, 0])
        
        # Extract coordinates
        sorted_frames = sorted(positions.keys())
        x_coords = [positions[f][0] for f in sorted_frames]
        y_coords = [positions[f][1] for f in sorted_frames]
        
        # Plot trajectory
        if show_points:
            # Color by time
            colors = sorted_frames
            scatter = ax.scatter(x_coords, y_coords, c=colors, cmap='viridis', 
                                s=2, alpha=0.6, edgecolors='none')
            plt.colorbar(scatter, ax=ax, label='Frame')
        else:
            ax.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1)
        
        # Mark start and end
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, 
               markeredgecolor='white', markeredgewidth=2, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10,
               markeredgecolor='white', markeredgewidth=2, label='End')
        
        # Set limits and labels
        ax.set_xlim(0, self.img_width)
        ax.set_ylim(self.img_height, 0)  # Invert y-axis
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        
        coverage = len(positions) / (max(sorted_frames) + 1) * 100
        ax.set_title(f'ROI {roi_id} Trajectory (Coverage: {coverage:.1f}%)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Plot saved to:[/green] {save_path}")
        
        plt.show()
    
    def plot_all_trajectories(self, use_interpolated: bool = True, 
                            separate_colors: bool = True,
                            show_subdish: bool = False,
                            save_path: Optional[Path] = None):
        """Plot all ROI trajectories on one figure."""
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Show background if available
        if self.background is not None:
            ax.imshow(self.background, cmap='gray', extent=[0, self.img_width, self.img_height, 0],
                     alpha=0.5)
        
        # Get colors for each ROI
        if separate_colors:
            colors = plt.cm.tab20(np.linspace(0, 1, num_rois))
        else:
            colors = ['blue'] * num_rois
        
        # Plot each ROI
        for roi_id in range(num_rois):
            positions = self.get_roi_positions(roi_id, use_interpolated)
            
            if positions:
                sorted_frames = sorted(positions.keys())
                x_coords = [positions[f][0] for f in sorted_frames]
                y_coords = [positions[f][1] for f in sorted_frames]
                
                ax.plot(x_coords, y_coords, color=colors[roi_id], 
                       alpha=0.6, linewidth=0.8, label=f'ROI {roi_id}')
                
                # Mark start and end
                ax.plot(x_coords[0], y_coords[0], 'o', color=colors[roi_id],
                       markersize=8, markeredgecolor='white', markeredgewidth=1)
                ax.plot(x_coords[-1], y_coords[-1], 's', color=colors[roi_id],
                       markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Add sub-dish boundaries if requested
        if show_subdish:
            self.add_subdish_boundaries(ax)
        
        # Add legend markers for start/end
        ax.plot([], [], 'ko', markersize=8, markeredgecolor='white', 
               markeredgewidth=1, label='Start points')
        ax.plot([], [], 'ks', markersize=8, markeredgecolor='white',
               markeredgewidth=1, label='End points')
        
        ax.set_xlim(0, self.img_width)
        ax.set_ylim(self.img_height, 0)  # Invert y-axis
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'All ROI Trajectories ({num_rois} ROIs)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Plot saved to:[/green] {save_path}")
        
        plt.show()
    
    def plot_trajectory_grid(self, use_interpolated: bool = True,
                            save_path: Optional[Path] = None):
        """Plot each ROI trajectory in a separate subplot."""
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(num_rois)))
        n_rows = int(np.ceil(num_rois / n_cols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each ROI
        for roi_id in range(num_rois):
            row = roi_id // n_cols
            col = roi_id % n_cols
            ax = axes[row, col]
            
            positions = self.get_roi_positions(roi_id, use_interpolated)
            
            if positions:
                sorted_frames = sorted(positions.keys())
                x_coords = [positions[f][0] for f in sorted_frames]
                y_coords = [positions[f][1] for f in sorted_frames]
                
                # Show background if available
                if self.background is not None:
                    ax.imshow(self.background, cmap='gray', 
                             extent=[0, self.img_width, self.img_height, 0],
                             alpha=0.3)
                
                # Plot trajectory
                ax.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=1)
                
                # Mark start and end
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=6)
                ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=6)
                
                coverage = len(positions) / (max(sorted_frames) + 1) * 100
                ax.set_title(f'ROI {roi_id} ({coverage:.1f}%)')
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'ROI {roi_id} (0.0%)')
            
            ax.set_xlim(0, self.img_width)
            ax.set_ylim(self.img_height, 0)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide empty subplots
        for roi_id in range(num_rois, n_rows * n_cols):
            row = roi_id // n_cols
            col = roi_id % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('Individual ROI Trajectories', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Plot saved to:[/green] {save_path}")
        
        plt.show()
    
    def add_subdish_boundaries(self, ax):
        """Add sub-dish boundary rectangles to plot."""
        # Try to load from zarr attributes first
        subdish_rois = []
        
        if 'id_assignments_runs' in self.root or 'id_assignments' in self.root:
            id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
            id_group = self.root[id_key]
            if 'latest' in id_group.attrs:
                latest = id_group.attrs['latest']
                params = id_group[latest].attrs.get('parameters', {})
                if 'sub_dish_rois' in params:
                    subdish_rois = params['sub_dish_rois']
        
        if subdish_rois:
            for roi in subdish_rois:
                x, y, w, h = roi['roi_pixels']
                
                # If using downsampled, no scaling needed
                if self.use_downsampled:
                    x_plot = x
                    y_plot = y
                    w_plot = w
                    h_plot = h
                else:
                    # Scale from downsampled to full coordinates
                    scale = 4512 / 640
                    x_plot = x * scale
                    y_plot = y * scale
                    w_plot = w * scale
                    h_plot = h * scale
                
                rect = Rectangle((x_plot, y_plot), w_plot, h_plot,
                               linewidth=2, edgecolor='cyan',
                               facecolor='none', linestyle='--')
                ax.add_patch(rect)
                
                # Add ID label
                ax.text(x_plot + w_plot/2, y_plot + h_plot/2, f"Dish {roi['id']}",
                       ha='center', va='center', color='cyan',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    def plot_time_colored_trajectory(self, roi_id: int, use_interpolated: bool = True,
                                    save_path: Optional[Path] = None):
        """Plot trajectory colored by time progression."""
        positions = self.get_roi_positions(roi_id, use_interpolated)
        
        if not positions:
            console.print(f"[yellow]No positions found for ROI {roi_id}[/yellow]")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        sorted_frames = sorted(positions.keys())
        x_coords = np.array([positions[f][0] for f in sorted_frames])
        y_coords = np.array([positions[f][1] for f in sorted_frames])
        time_seconds = np.array(sorted_frames) / self.fps
        
        # Left plot: trajectory colored by time
        if self.background is not None:
            ax1.imshow(self.background, cmap='gray', 
                      extent=[0, self.img_width, self.img_height, 0], alpha=0.5)
        
        # Create line segments colored by time
        for i in range(1, len(sorted_frames)):
            ax1.plot(x_coords[i-1:i+1], y_coords[i-1:i+1], 
                    color=plt.cm.viridis(i/len(sorted_frames)),
                    linewidth=2, alpha=0.8)
        
        ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, 
                markeredgecolor='white', markeredgewidth=2, label='Start')
        ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10,
                markeredgecolor='white', markeredgewidth=2, label='End')
        
        ax1.set_xlim(0, self.img_width)
        ax1.set_ylim(self.img_height, 0)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title(f'ROI {roi_id} - Time-Colored Trajectory')
        ax1.legend()
        
        # Right plot: X and Y over time
        ax2_y = ax2.twinx()
        
        line1 = ax2.plot(time_seconds, x_coords, 'b-', alpha=0.7, label='X position')
        line2 = ax2_y.plot(time_seconds, y_coords, 'r-', alpha=0.7, label='Y position')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('X Position (pixels)', color='b')
        ax2_y.set_ylabel('Y Position (pixels)', color='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_y.tick_params(axis='y', labelcolor='r')
        ax2.set_title('Position Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        plt.suptitle(f'ROI {roi_id} Temporal Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Plot saved to:[/green] {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot fish trajectories from zarr detection data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single ROI trajectory
  %(prog)s detections.zarr --roi 3
  
  # Plot all trajectories together
  %(prog)s detections.zarr --all
  
  # Plot grid of individual trajectories
  %(prog)s detections.zarr --grid
  
  # Time-colored trajectory
  %(prog)s detections.zarr --roi 3 --time-color
  
  # Show sub-dish boundaries
  %(prog)s detections.zarr --all --show-subdish
  
  # Save output
  %(prog)s detections.zarr --all --save trajectories.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--roi', type=int, help='Specific ROI to plot')
    parser.add_argument('--all', action='store_true', help='Plot all ROIs together')
    parser.add_argument('--grid', action='store_true', help='Plot ROIs in grid layout')
    parser.add_argument('--time-color', action='store_true', 
                       help='Color trajectory by time (requires --roi)')
    parser.add_argument('--show-points', action='store_true',
                       help='Show individual detection points')
    parser.add_argument('--show-subdish', action='store_true',
                       help='Show sub-dish boundaries')
    parser.add_argument('--no-interpolated', action='store_true',
                       help='Do not use interpolated positions')
    parser.add_argument('--full-res', action='store_true',
                       help='Use full resolution coordinates instead of downsampled')
    parser.add_argument('--save', type=str, help='Path to save figure')
    
    args = parser.parse_args()
    
    # Initialize plotter with appropriate resolution
    use_downsampled = not args.full_res
    plotter = ZarrTrajectoryPlotter(args.zarr_path, use_downsampled=use_downsampled)
    
    console.print(f"\n[bold cyan]Trajectory Plotter[/bold cyan]")
    console.print(f"Zarr file: {args.zarr_path}")
    
    save_path = Path(args.save) if args.save else None
    use_interpolated = not args.no_interpolated
    
    # Execute requested plot
    if args.roi is not None:
        if args.time_color:
            plotter.plot_time_colored_trajectory(args.roi, use_interpolated, save_path)
        else:
            plotter.plot_single_trajectory(args.roi, use_interpolated, 
                                          args.show_points, save_path)
    elif args.all:
        plotter.plot_all_trajectories(use_interpolated, True,
                                     args.show_subdish, save_path)
    elif args.grid:
        plotter.plot_trajectory_grid(use_interpolated, save_path)
    else:
        console.print("[yellow]Please specify --roi, --all, or --grid[/yellow]")


if __name__ == "__main__":
    main()