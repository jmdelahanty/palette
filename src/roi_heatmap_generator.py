#!/usr/bin/env python3
"""
ROI Heatmap Generator

Creates spatial heatmaps showing where each ROI (fish) spends time.
Supports individual and batch processing with various visualization options.
CORRECTED: Fixed coordinate system transformations for accurate positioning.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm, LogNorm
from scipy.ndimage import gaussian_filter
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

console = Console()


class ROIHeatmapGenerator:
    """Generate spatial heatmaps for ROI positions."""
    
    def __init__(self, zarr_path: str, verbose: bool = True, use_downsampled: bool = False):
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.use_downsampled = use_downsampled
        self.root = zarr.open_group(self.zarr_path, mode='r+')
        
        # Get dimensions
        self.img_width = self.root.attrs.get('width', 4512)
        self.img_height = self.root.attrs.get('height', 4512)
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Downsampled dimensions (for detections)
        self.ds_width = 640
        self.ds_height = 640
        
        # Set working dimensions based on mode
        if use_downsampled:
            self.working_width = self.ds_width
            self.working_height = self.ds_height
        else:
            self.working_width = self.img_width
            self.working_height = self.img_height
        
        # Check for calibration
        self.pixel_to_mm = None
        if 'calibration' in self.root:
            self.pixel_to_mm = self.root['calibration'].attrs.get('pixel_to_mm', None)
            if self.verbose and self.pixel_to_mm:
                console.print(f"[green]Calibration found:[/green] 1 pixel = {self.pixel_to_mm:.4f} mm")
    
    def get_roi_positions(self, roi_id: int, use_interpolated: bool = True) -> Dict:
        """Get all positions for a specific ROI with CORRECTED coordinate transformations."""
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
                    
                    # CORRECTED: bbox format is [center_x_norm, center_y_norm, width_norm, height_norm]
                    # Normalized coordinates are relative to 640x640 downsampled space
                    center_x_norm = bbox[0]  # Already the center, not corner!
                    center_y_norm = bbox[1]  # Already the center, not corner!
                    
                    # Convert to pixel coordinates
                    if self.use_downsampled:
                        # Stay in 640x640 space
                        centroid_x = center_x_norm * self.ds_width
                        centroid_y = center_y_norm * self.ds_height
                    else:
                        # Convert to full resolution
                        centroid_x_ds = center_x_norm * self.ds_width
                        centroid_y_ds = center_y_norm * self.ds_height
                        scale_factor = self.img_width / self.ds_width
                        centroid_x = centroid_x_ds * scale_factor
                        centroid_y = centroid_y_ds * scale_factor
                    
                    positions[frame_idx] = np.array([centroid_x, centroid_y])
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions if available and requested
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
                        
                        # CORRECTED: Same fix for interpolated positions
                        center_x_norm = bbox[0]
                        center_y_norm = bbox[1]
                        
                        if self.use_downsampled:
                            centroid_x = center_x_norm * self.ds_width
                            centroid_y = center_y_norm * self.ds_height
                        else:
                            centroid_x_ds = center_x_norm * self.ds_width
                            centroid_y_ds = center_y_norm * self.ds_height
                            scale_factor = self.img_width / self.ds_width
                            centroid_x = centroid_x_ds * scale_factor
                            centroid_y = centroid_y_ds * scale_factor
                        
                        positions[frame_idx] = np.array([centroid_x, centroid_y])
        
        return positions
    
    def create_heatmap(self, positions: Dict, bin_size: int = 50, 
                      sigma: float = 1.5, normalize: bool = True) -> np.ndarray:
        """
        Create a 2D heatmap from positions.
        
        Args:
            positions: Dictionary of frame_idx -> [x, y] positions
            bin_size: Size of bins in pixels (scaled based on resolution)
            sigma: Gaussian smoothing sigma
            normalize: Normalize to probability density
            
        Returns:
            2D heatmap array
        """
        if not positions:
            return None
        
        # Scale bin size if using downsampled
        if self.use_downsampled:
            effective_bin_size = bin_size * (self.ds_width / self.img_width)
        else:
            effective_bin_size = bin_size
        
        # Calculate number of bins
        n_bins_x = int(np.ceil(self.working_width / effective_bin_size))
        n_bins_y = int(np.ceil(self.working_height / effective_bin_size))
        
        # Create 2D histogram
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=[n_bins_x, n_bins_y],
            range=[[0, self.working_width], [0, self.working_height]]
        )
        
        # Apply Gaussian smoothing
        if sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if normalize and heatmap.sum() > 0:
            heatmap = heatmap / heatmap.sum()
        
        return heatmap.T  # Transpose for proper orientation
    
    def visualize_roi_heatmap(self, roi_id: int, bin_size: int = 50,
                             sigma: float = 1.5, cmap: str = 'hot',
                             save_path: Optional[Path] = None,
                             use_interpolated: bool = True,
                             show_background: bool = False):
        """Create visualization for a single ROI's heatmap."""
        # Get positions
        positions = self.get_roi_positions(roi_id, use_interpolated)
        
        if not positions:
            console.print(f"[yellow]No positions found for ROI {roi_id}[/yellow]")
            return
        
        # Create heatmap
        heatmap = self.create_heatmap(positions, bin_size, sigma)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main heatmap
        ax1 = fig.add_subplot(gs[:, 0:2])
        
        # Load background if requested
        if show_background:
            try:
                latest_bg_run = self.root['background_runs'].attrs['latest']
                if self.use_downsampled:
                    background = self.root[f'background_runs/{latest_bg_run}/background_ds'][:]
                else:
                    background = self.root[f'background_runs/{latest_bg_run}/background_full'][:]
                ax1.imshow(background, cmap='gray', alpha=0.3, 
                          extent=[0, self.working_width, self.working_height, 0])
            except:
                pass
        
        extent = [0, self.working_width, self.working_height, 0]
        im = ax1.imshow(heatmap, cmap=cmap, extent=extent, 
                       aspect='equal', interpolation='bilinear', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Occupancy Density', rotation=270, labelpad=20)
        
        # Add trajectory overlay (subsample for clarity)
        sorted_frames = sorted(positions.keys())
        step = max(1, len(sorted_frames) // 1000)  # Show max 1000 points
        sample_frames = sorted_frames[::step]
        
        x_traj = [positions[f][0] for f in sample_frames]
        y_traj = [positions[f][1] for f in sample_frames]
        
        ax1.plot(x_traj, y_traj, 'cyan', alpha=0.3, linewidth=0.5)
        ax1.scatter(x_traj[0], y_traj[0], c='green', s=100, marker='o', 
                   edgecolors='white', linewidth=2, label='Start', zorder=5)
        ax1.scatter(x_traj[-1], y_traj[-1], c='red', s=100, marker='s', 
                   edgecolors='white', linewidth=2, label='End', zorder=5)
        
        resolution_text = "640×640" if self.use_downsampled else "4512×4512"
        ax1.set_xlabel(f'X Position (pixels) [{resolution_text}]')
        ax1.set_ylabel(f'Y Position (pixels) [{resolution_text}]')
        ax1.set_title(f'ROI {roi_id} - Spatial Heatmap (Coverage: {len(positions)/len(sorted_frames)*100:.1f}%)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.2)
        
        # 2. X-axis marginal distribution
        ax2 = fig.add_subplot(gs[0, 2])
        
        x_hist, x_bins = np.histogram([p[0] for p in positions.values()], 
                                     bins=50, range=(0, self.working_width))
        ax2.bar(x_bins[:-1], x_hist, width=np.diff(x_bins), color='steelblue', alpha=0.7)
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('X-axis Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Y-axis marginal distribution
        ax3 = fig.add_subplot(gs[1, 2])
        
        y_hist, y_bins = np.histogram([p[1] for p in positions.values()], 
                                     bins=50, range=(0, self.working_height))
        ax3.bar(y_bins[:-1], y_hist, width=np.diff(y_bins), color='coral', alpha=0.7)
        ax3.set_xlabel('Y Position (pixels)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Y-axis Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        x_coords = np.array([p[0] for p in positions.values()])
        y_coords = np.array([p[1] for p in positions.values()])
        
        stats_text = f"""
Statistics ({resolution_text}):
━━━━━━━━━━━━━━━━━━━━
Frames tracked: {len(positions)}
Total frames: {max(positions.keys()) + 1}
Coverage: {len(positions)/(max(positions.keys()) + 1)*100:.1f}%

X-axis:
  Mean: {np.mean(x_coords):.1f} px
  Std: {np.std(x_coords):.1f} px
  Range: [{np.min(x_coords):.0f}, {np.max(x_coords):.0f}]

Y-axis:
  Mean: {np.mean(y_coords):.1f} px
  Std: {np.std(y_coords):.1f} px
  Range: [{np.min(y_coords):.0f}, {np.max(y_coords):.0f}]
"""
        
        if self.pixel_to_mm and not self.use_downsampled:
            stats_text += f"""
Physical units:
  X range: {(np.max(x_coords) - np.min(x_coords)) * self.pixel_to_mm:.1f} mm
  Y range: {(np.max(y_coords) - np.min(y_coords)) * self.pixel_to_mm:.1f} mm
"""
        
        fig.text(0.98, 0.02, stats_text, transform=fig.transFigure,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'ROI {roi_id} - Spatial Activity Analysis', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def create_batch_heatmaps(self, bin_size: int = 50, sigma: float = 1.5,
                             use_interpolated: bool = True) -> Dict:
        """Generate heatmaps for all ROIs."""
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        all_heatmaps = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"[cyan]Generating heatmaps for {num_rois} ROIs...", total=num_rois)
            
            for roi_id in range(num_rois):
                progress.update(task, description=f"[cyan]Processing ROI {roi_id}/{num_rois-1}")
                
                positions = self.get_roi_positions(roi_id, use_interpolated)
                if positions:
                    heatmap = self.create_heatmap(positions, bin_size, sigma)
                    all_heatmaps[roi_id] = {
                        'heatmap': heatmap,
                        'positions': positions,
                        'coverage': len(positions) / (max(positions.keys()) + 1) * 100
                    }
                
                progress.advance(task)
        
        return all_heatmaps
    
    def visualize_all_heatmaps(self, heatmaps: Dict, cmap: str = 'hot',
                               save_path: Optional[Path] = None):
        """Create grid visualization of all ROI heatmaps."""
        num_rois = len(heatmaps)
        
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(num_rois)))
        n_rows = int(np.ceil(num_rois / n_cols))
        
        # Create figure
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        for idx, (roi_id, data) in enumerate(sorted(heatmaps.items())):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            # Plot heatmap
            extent = [0, self.working_width, self.working_height, 0]
            im = ax.imshow(data['heatmap'], cmap=cmap, extent=extent,
                          aspect='equal', interpolation='bilinear')
            
            # Add title with coverage
            ax.set_title(f'ROI {roi_id} ({data["coverage"]:.1f}% coverage)')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add single colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Occupancy Density')
        
        resolution = "640×640" if self.use_downsampled else "4512×4512"
        plt.suptitle(f'Spatial Heatmaps - All ROIs [{resolution}]', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Generate spatial heatmaps for ROI positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single ROI heatmap (full resolution)
  %(prog)s detections.zarr --roi 3
  
  # Single ROI heatmap (downsampled 640x640)
  %(prog)s detections.zarr --roi 3 --downsampled
  
  # All ROIs in grid
  %(prog)s detections.zarr --all
  
  # Custom parameters
  %(prog)s detections.zarr --roi 3 --bin-size 100 --sigma 2.0 --cmap viridis
  
  # With background image
  %(prog)s detections.zarr --roi 3 --show-background
  
  # Save outputs
  %(prog)s detections.zarr --all --save-fig all_heatmaps.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--roi', type=int, help='Specific ROI to visualize')
    parser.add_argument('--all', action='store_true', help='Generate heatmaps for all ROIs')
    parser.add_argument('--bin-size', type=int, default=50,
                       help='Bin size in pixels (default: 50)')
    parser.add_argument('--sigma', type=float, default=1.5,
                       help='Gaussian smoothing sigma (default: 1.5)')
    parser.add_argument('--cmap', default='hot',
                       help='Colormap for heatmaps (default: hot)')
    parser.add_argument('--no-interpolated', action='store_true',
                       help='Do not use interpolated positions')
    parser.add_argument('--downsampled', action='store_true',
                       help='Use 640x640 downsampled coordinates instead of full resolution')
    parser.add_argument('--show-background', action='store_true',
                       help='Show background image under heatmap')
    parser.add_argument('--save-fig', type=str,
                       help='Path to save figure')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ROIHeatmapGenerator(args.zarr_path, use_downsampled=args.downsampled)
    
    resolution = "640×640" if args.downsampled else "4512×4512"
    console.print(f"\n[bold cyan]ROI Heatmap Generator[/bold cyan]")
    console.print(f"File: {args.zarr_path}")
    console.print(f"Resolution: {resolution}")
    console.print(f"Bin size: {args.bin_size} pixels")
    console.print(f"Smoothing sigma: {args.sigma}")
    
    # Process based on mode
    if args.roi is not None:
        # Single ROI
        save_path = Path(args.save_fig) if args.save_fig else None
        generator.visualize_roi_heatmap(
            args.roi, args.bin_size, args.sigma, args.cmap,
            save_path, not args.no_interpolated, args.show_background
        )
    
    elif args.all:
        # All ROIs
        heatmaps = generator.create_batch_heatmaps(
            args.bin_size, args.sigma, not args.no_interpolated
        )
        
        if heatmaps:
            save_path = Path(args.save_fig) if args.save_fig else None
            generator.visualize_all_heatmaps(heatmaps, args.cmap, save_path)
    
    else:
        console.print("[yellow]Please specify --roi or --all[/yellow]")


if __name__ == "__main__":
    main()