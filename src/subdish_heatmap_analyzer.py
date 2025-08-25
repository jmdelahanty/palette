#!/usr/bin/env python3
"""
Sub-dish Heatmap Analyzer

Creates spatial heatmaps for each sub-dish ROI, showing activity patterns
within individual dishes/wells. Useful for multi-well plate experiments.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter
import yaml
from pathlib import Path
import argparse
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

console = Console()


class SubdishHeatmapAnalyzer:
    """Analyze spatial patterns within sub-dish ROIs."""
    
    def __init__(self, zarr_path: str, config_path: str = "src/pipeline_config.yaml", 
                 verbose: bool = True):
        self.zarr_path = Path(zarr_path)
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.root = zarr.open_group(self.zarr_path, mode='r+')
        
        # Get dimensions
        self.img_width = self.root.attrs.get('width', 4512)
        self.img_height = self.root.attrs.get('height', 4512)
        self.fps = self.root.attrs.get('fps', 60.0)
        
        # Get downsampled dimensions for sub-dish ROIs
        if 'raw_video/images_ds' in self.root:
            self.ds_shape = self.root['raw_video/images_ds'].shape[1:]
        else:
            self.ds_shape = (640, 640)  # Default
        
        # Load sub-dish ROIs from config
        self.subdish_rois = self.load_subdish_rois()
        
        # Check for calibration
        self.pixel_to_mm = None
        if 'calibration' in self.root:
            self.pixel_to_mm = self.root['calibration'].attrs.get('pixel_to_mm', None)
            if self.verbose and self.pixel_to_mm:
                console.print(f"[green]Calibration found:[/green] 1 pixel = {self.pixel_to_mm:.4f} mm")
    
    def load_subdish_rois(self) -> List[Dict]:
        """Load sub-dish ROI definitions from config or zarr."""
        subdish_rois = []
        
        # First try to load from zarr if it was saved there
        if 'id_assignments_runs' in self.root or 'id_assignments' in self.root:
            id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
            id_group = self.root[id_key]
            if 'latest' in id_group.attrs:
                latest = id_group.attrs['latest']
                params = id_group[latest].attrs.get('parameters', {})
                if 'sub_dish_rois' in params:
                    subdish_rois = params['sub_dish_rois']
                    console.print(f"[green]Loaded {len(subdish_rois)} sub-dish ROIs from zarr[/green]")
        
        # If not found in zarr, load from config file
        if not subdish_rois and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'assign_ids' in config and 'sub_dish_rois' in config['assign_ids']:
                    subdish_rois = config['assign_ids']['sub_dish_rois']
                    console.print(f"[green]Loaded {len(subdish_rois)} sub-dish ROIs from config[/green]")
        
        if not subdish_rois:
            console.print("[yellow]Warning: No sub-dish ROIs found![/yellow]")
            console.print("Please define sub-dish ROIs using subdish_roi_mask_creator.py")
        
        return subdish_rois
    
    def scale_roi_to_full(self, roi_pixels: List[int]) -> Tuple[int, int, int, int]:
        """Scale sub-dish ROI from downsampled to full resolution."""
        x, y, w, h = roi_pixels
        scale_x = self.img_width / self.ds_shape[1]
        scale_y = self.img_height / self.ds_shape[0]
        
        x_full = int(x * scale_x)
        y_full = int(y * scale_y)
        w_full = int(w * scale_x)
        h_full = int(h * scale_y)
        
        return x_full, y_full, w_full, h_full
    
    def get_positions_in_subdish(self, roi_id: int, subdish_id: int,
                                 use_interpolated: bool = True) -> Dict:
        """Get positions for a specific ROI within a specific sub-dish."""
        if subdish_id >= len(self.subdish_rois):
            return {}
        
        # Get sub-dish boundaries (in full resolution)
        subdish = self.subdish_rois[subdish_id]
        x_full, y_full, w_full, h_full = self.scale_roi_to_full(subdish['roi_pixels'])
        
        # Get all positions for this ROI
        positions = self.get_roi_positions(roi_id, use_interpolated)
        
        # Filter positions within sub-dish boundaries
        filtered_positions = {}
        for frame_idx, pos in positions.items():
            if (x_full <= pos[0] < x_full + w_full and 
                y_full <= pos[1] < y_full + h_full):
                # Translate to sub-dish local coordinates
                local_pos = np.array([pos[0] - x_full, pos[1] - y_full])
                filtered_positions[frame_idx] = local_pos
        
        return filtered_positions
    
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
                    # Calculate centroid in pixels
                    centroid_x = ((bbox[0] + bbox[2]) / 2) * self.img_width
                    centroid_y = ((bbox[1] + bbox[3]) / 2) * self.img_height
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
                        centroid_x = ((bbox[0] + bbox[2]) / 2) * self.img_width
                        centroid_y = ((bbox[1] + bbox[3]) / 2) * self.img_height
                        positions[frame_idx] = np.array([centroid_x, centroid_y])
        
        return positions
    
    def create_subdish_heatmap(self, positions: Dict, subdish_dims: Tuple[int, int],
                              bin_size: int = 20, sigma: float = 1.0,
                              normalize: bool = True) -> np.ndarray:
        """Create heatmap for positions within a sub-dish."""
        if not positions:
            return None
        
        w, h = subdish_dims
        
        # Calculate number of bins
        n_bins_x = int(np.ceil(w / bin_size))
        n_bins_y = int(np.ceil(h / bin_size))
        
        # Create 2D histogram
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=[n_bins_x, n_bins_y],
            range=[[0, w], [0, h]]
        )
        
        # Apply Gaussian smoothing
        if sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if normalize and heatmap.sum() > 0:
            heatmap = heatmap / heatmap.sum()
        
        return heatmap.T  # Transpose for proper orientation
    
    def visualize_subdish_heatmaps(self, bin_size: int = 20, sigma: float = 1.0,
                                   cmap: str = 'hot', save_path: Optional[Path] = None):
        """Create visualization of all sub-dish heatmaps."""
        if not self.subdish_rois:
            console.print("[red]No sub-dish ROIs defined![/red]")
            return
        
        num_subdishes = len(self.subdish_rois)
        
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        # Check if ROIs match sub-dishes
        if num_rois != num_subdishes:
            console.print(f"[yellow]Warning: {num_rois} ROIs but {num_subdishes} sub-dishes defined[/yellow]")
        
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(num_subdishes)))
        n_rows = int(np.ceil(num_subdishes / n_cols))
        
        # Create figure
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        # Process each sub-dish
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"[cyan]Processing {num_subdishes} sub-dishes...", 
                                    total=num_subdishes)
            
            for subdish_idx, subdish in enumerate(self.subdish_rois):
                ax = fig.add_subplot(n_rows, n_cols, subdish_idx + 1)
                
                # Get sub-dish dimensions
                x_full, y_full, w_full, h_full = self.scale_roi_to_full(subdish['roi_pixels'])
                
                # Get positions for the corresponding ROI
                # Since IDs are assigned based on sub-dish, ROI ID == subdish ID
                roi_id = subdish['id']
                all_positions = self.get_roi_positions(roi_id, True)
                
                # Translate positions to sub-dish local coordinates
                positions = {}
                for frame_idx, pos in all_positions.items():
                    # Translate to sub-dish local coordinates (origin at sub-dish corner)
                    local_pos = np.array([pos[0] - x_full, pos[1] - y_full])
                    # Only keep if within bounds (shouldn't be necessary but good to check)
                    if 0 <= local_pos[0] < w_full and 0 <= local_pos[1] < h_full:
                        positions[frame_idx] = local_pos
                
                if positions:
                    # Create heatmap
                    heatmap = self.create_subdish_heatmap(positions, (w_full, h_full),
                                                         bin_size, sigma, normalize=True)
                    
                    if heatmap is not None:
                        # Plot heatmap
                        im = ax.imshow(heatmap, cmap=cmap, aspect='equal',
                                      interpolation='bilinear', extent=[0, w_full, h_full, 0])
                        
                        # Add colorbar for each subplot
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    else:
                        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No detections', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12)
                
                # Calculate coverage
                total_frames = max(positions.keys()) + 1 if positions else 1
                coverage = len(positions) / total_frames * 100 if positions else 0
                
                # Set title and labels
                ax.set_title(f'Sub-dish {subdish_idx} (ROI {roi_id})\n{coverage:.1f}% coverage')
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                
                progress.advance(task)
        
        plt.suptitle('Sub-dish Spatial Heatmaps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def visualize_full_arena_with_subdishes(self, bin_size: int = 50, sigma: float = 1.5,
                                           cmap: str = 'hot', save_path: Optional[Path] = None):
        """Visualize full arena with sub-dish boundaries overlaid."""
        if not self.subdish_rois:
            console.print("[red]No sub-dish ROIs defined![/red]")
            return
        
        # Create full arena heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get all positions across all ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        # Collect all positions
        all_positions = []
        roi_positions = {}
        
        for roi_id in range(num_rois):
            positions = self.get_roi_positions(roi_id, True)
            roi_positions[roi_id] = positions
            for pos in positions.values():
                all_positions.append(pos)
        
        if all_positions:
            all_positions = np.array(all_positions)
            
            # Create full arena heatmap
            heatmap, xedges, yedges = np.histogram2d(
                all_positions[:, 0], all_positions[:, 1],
                bins=[self.img_width // bin_size, self.img_height // bin_size],
                range=[[0, self.img_width], [0, self.img_height]]
            )
            
            # Apply smoothing
            if sigma > 0:
                heatmap = gaussian_filter(heatmap, sigma=sigma)
            
            # Plot full arena heatmap
            im1 = ax1.imshow(heatmap.T, cmap=cmap, extent=[0, self.img_width, self.img_height, 0],
                            aspect='equal', interpolation='bilinear')
            
            # Add sub-dish boundaries
            for subdish in self.subdish_rois:
                x, y, w, h = self.scale_roi_to_full(subdish['roi_pixels'])
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan',
                               facecolor='none', linestyle='--')
                ax1.add_patch(rect)
                ax1.text(x + w/2, y + h/2, f"ID: {subdish['id']}", 
                        ha='center', va='center', color='white',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            ax1.set_title('Full Arena Heatmap with Sub-dish Boundaries')
            ax1.set_xlabel('X Position (pixels)')
            ax1.set_ylabel('Y Position (pixels)')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot trajectories colored by ROI
        colors = plt.cm.tab20(np.linspace(0, 1, num_rois))
        
        for roi_id, positions in roi_positions.items():
            if positions:
                sorted_frames = sorted(positions.keys())
                x_coords = [positions[f][0] for f in sorted_frames]
                y_coords = [positions[f][1] for f in sorted_frames]
                ax2.plot(x_coords, y_coords, alpha=0.5, linewidth=0.5,
                        color=colors[roi_id], label=f'ROI {roi_id}')
        
        # Add sub-dish boundaries to trajectory plot
        for subdish in self.subdish_rois:
            x, y, w, h = self.scale_roi_to_full(subdish['roi_pixels'])
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='black',
                           facecolor='none', linestyle='-')
            ax2.add_patch(rect)
        
        ax2.set_xlim(0, self.img_width)
        ax2.set_ylim(self.img_height, 0)  # Invert y-axis for image coordinates
        ax2.set_aspect('equal')
        ax2.set_title('All Trajectories with Sub-dish Boundaries')
        ax2.set_xlabel('X Position (pixels)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        plt.suptitle('Arena Overview with Sub-dish Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Figure saved to:[/green] {save_path}")
        
        plt.show()
    
    def print_subdish_summary(self):
        """Print summary of sub-dish ROI analysis."""
        if not self.subdish_rois:
            console.print("[red]No sub-dish ROIs defined![/red]")
            return
        
        table = Table(title="Sub-dish ROI Summary")
        
        table.add_column("Sub-dish", style="cyan", no_wrap=True)
        table.add_column("ROI ID", style="yellow")
        table.add_column("Position (DS)", style="green")
        table.add_column("Size (DS)", style="blue")
        table.add_column("Position (Full)", style="magenta")
        table.add_column("Size (Full)", style="red")
        
        for idx, subdish in enumerate(self.subdish_rois):
            x_ds, y_ds, w_ds, h_ds = subdish['roi_pixels']
            x_full, y_full, w_full, h_full = self.scale_roi_to_full(subdish['roi_pixels'])
            
            table.add_row(
                str(idx),
                str(subdish['id']),
                f"({x_ds}, {y_ds})",
                f"{w_ds}×{h_ds}",
                f"({x_full}, {y_full})",
                f"{w_full}×{h_full}"
            )
        
        console.print(table)
        
        if self.pixel_to_mm:
            console.print(f"\n[cyan]Physical dimensions (with calibration):[/cyan]")
            for idx, subdish in enumerate(self.subdish_rois):
                x_full, y_full, w_full, h_full = self.scale_roi_to_full(subdish['roi_pixels'])
                w_mm = w_full * self.pixel_to_mm
                h_mm = h_full * self.pixel_to_mm
                console.print(f"  Sub-dish {idx}: {w_mm:.1f} × {h_mm:.1f} mm")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze spatial patterns within sub-dish ROIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all sub-dish heatmaps
  %(prog)s detections.zarr
  
  # Full arena overview with sub-dish boundaries
  %(prog)s detections.zarr --full-arena
  
  # Custom parameters
  %(prog)s detections.zarr --bin-size 30 --sigma 2.0 --cmap viridis
  
  # Use custom config file
  %(prog)s detections.zarr --config my_config.yaml
  
  # Save outputs
  %(prog)s detections.zarr --save-fig subdish_heatmaps.png
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--config', default='src/pipeline_config.yaml',
                       help='Path to config file with sub-dish ROI definitions')
    parser.add_argument('--full-arena', action='store_true',
                       help='Show full arena view with sub-dish boundaries')
    parser.add_argument('--bin-size', type=int, default=20,
                       help='Bin size in pixels (default: 20)')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Gaussian smoothing sigma (default: 1.0)')
    parser.add_argument('--cmap', default='hot',
                       help='Colormap for heatmaps (default: hot)')
    parser.add_argument('--save-fig', type=str,
                       help='Path to save figure')
    parser.add_argument('--summary', action='store_true',
                       help='Print sub-dish ROI summary')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SubdishHeatmapAnalyzer(args.zarr_path, args.config)
    
    console.print(f"\n[bold cyan]Sub-dish Heatmap Analysis[/bold cyan]")
    console.print(f"Zarr file: {args.zarr_path}")
    console.print(f"Config file: {args.config}")
    
    # Print summary if requested
    if args.summary:
        analyzer.print_subdish_summary()
    
    # Create visualizations
    save_path = Path(args.save_fig) if args.save_fig else None
    
    if args.full_arena:
        analyzer.visualize_full_arena_with_subdishes(
            args.bin_size, args.sigma, args.cmap, save_path
        )
    else:
        analyzer.visualize_subdish_heatmaps(
            args.bin_size, args.sigma, args.cmap, save_path
        )


if __name__ == "__main__":
    main()