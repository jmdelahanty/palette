#!/usr/bin/env python3
"""
First 5 Minutes Trajectory Plotter

Visualizes fish swimming trajectories for the first 5 minutes of recording.
Creates publication-quality plots with multiple visualization options.
"""

import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
import argparse
from rich.console import Console
from typing import Optional, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

console = Console()


class First5MinTrajectoryPlotter:
    """Plot trajectories for the first 5 minutes from zarr detection data."""
    
    def __init__(self, zarr_path: str, duration_minutes: float = 5.0,
                 h5_path: Optional[str] = None, bg_image_path: Optional[str] = None):
        """
        Initialize trajectory plotter.
        
        Args:
            zarr_path: Path to zarr file with detection data
            duration_minutes: Duration to plot in minutes (default: 5.0)
            h5_path: Optional path to H5 file containing background image
            bg_image_path: Optional direct path to background image
        """
        self.zarr_path = Path(zarr_path)
        self.duration_minutes = duration_minutes
        self.h5_path = Path(h5_path) if h5_path else None
        self.bg_image_path = Path(bg_image_path) if bg_image_path else None
        
        # Open zarr file
        self.root = zarr.open_group(str(self.zarr_path), mode='r')
        
        # Get metadata
        self.fps = self.root.attrs.get('fps', 60.0)
        self.width = self.root.attrs.get('width', 4512)
        self.height = self.root.attrs.get('height', 4512)
        
        # Calculate frame limits
        self.max_frames = int(duration_minutes * 60 * self.fps)
        
        console.print(f"[cyan]Loading data from: {self.zarr_path}[/cyan]")
        console.print(f"[cyan]FPS: {self.fps}, Duration: {duration_minutes} minutes ({self.max_frames} frames)[/cyan]")
        console.print(f"[cyan]Arena size: {self.width}x{self.height} pixels[/cyan]")
        
        # Load background image if available
        self.background_image = self.load_background_image()
        
        # Load detection data
        self.load_detections()
    
    def load_background_image(self) -> Optional[np.ndarray]:
        """Load background image from zarr, H5 file, or image file."""
        bg_image = None
        
        # Try loading from zarr first (most common case)
        if 'background_runs' in self.root:
            latest_bg = self.root['background_runs'].attrs.get('latest')
            if latest_bg:
                # Try to load background from zarr
                bg_path_ds = f'background_runs/{latest_bg}/background_ds'
                bg_path_full = f'background_runs/{latest_bg}/background_full'
                
                # Prefer full resolution if available
                if bg_path_full in self.root:
                    bg_image = self.root[bg_path_full][:]
                    console.print(f"[green]✓ Loaded full resolution background from zarr[/green]")
                elif bg_path_ds in self.root:
                    bg_image_ds = self.root[bg_path_ds][:]
                    # Upscale to match arena dimensions if needed
                    if bg_image_ds.shape != (self.height, self.width):
                        try:
                            import cv2
                            bg_image = cv2.resize(bg_image_ds, (self.width, self.height), 
                                                interpolation=cv2.INTER_LINEAR)
                            console.print(f"[green]✓ Loaded and upscaled background from zarr (640x640 → {self.width}x{self.height})[/green]")
                        except ImportError:
                            from PIL import Image
                            bg_pil = Image.fromarray(bg_image_ds)
                            bg_pil_resized = bg_pil.resize((self.width, self.height), Image.BILINEAR)
                            bg_image = np.array(bg_pil_resized)
                            console.print(f"[green]✓ Loaded and upscaled background from zarr[/green]")
                    else:
                        bg_image = bg_image_ds
                        console.print(f"[green]✓ Loaded downsampled background from zarr[/green]")
        
        # If not found in zarr, try loading from direct image path
        if bg_image is None and self.bg_image_path and self.bg_image_path.exists():
            try:
                import cv2
                bg_image = cv2.imread(str(self.bg_image_path), cv2.IMREAD_GRAYSCALE)
                if bg_image is not None:
                    console.print(f"[green]✓ Loaded background image from {self.bg_image_path}[/green]")
            except ImportError:
                from PIL import Image
                bg_image = np.array(Image.open(self.bg_image_path).convert('L'))
                console.print(f"[green]✓ Loaded background image from {self.bg_image_path}[/green]")
        
        # Try loading from H5 file
        if bg_image is None and self.h5_path and self.h5_path.exists():
            try:
                with h5py.File(self.h5_path, 'r') as hf:
                    # Common locations for background image in H5
                    possible_paths = [
                        '/background',
                        '/background_image',
                        '/background_model',
                        '/images/background',
                        '/calibration_snapshot/background_image'
                    ]
                    
                    for path in possible_paths:
                        if path in hf:
                            bg_image = hf[path][:]
                            console.print(f"[green]✓ Loaded background image from {self.h5_path}:{path}[/green]")
                            break
                    
                    if bg_image is None:
                        # Try to get first frame as background
                        if '/images/full_images' in hf:
                            bg_image = hf['/images/full_images'][0]
                            console.print(f"[green]✓ Using first frame as background from {self.h5_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load background from H5: {e}[/yellow]")
        
        # Try to find background_model.png in the same directory as zarr
        if bg_image is None:
            bg_path_auto = self.zarr_path.parent / "background_model.png"
            if bg_path_auto.exists():
                try:
                    import cv2
                    bg_image = cv2.imread(str(bg_path_auto), cv2.IMREAD_GRAYSCALE)
                    console.print(f"[green]✓ Found background_model.png in zarr directory[/green]")
                except ImportError:
                    from PIL import Image
                    bg_image = np.array(Image.open(bg_path_auto).convert('L'))
                    console.print(f"[green]✓ Found background_model.png in zarr directory[/green]")
        
        if bg_image is None:
            console.print("[yellow]No background image found - will use white background[/yellow]")
        
        return bg_image
    
    def load_detections(self):
        """Load detection data from zarr."""
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        
        # Load only first 5 minutes of data
        n_detections_full = detect_group[latest_detect]['n_detections'][:]
        total_frames = min(len(n_detections_full), self.max_frames)
        
        self.n_detections = n_detections_full[:total_frames]
        self.bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        self.detection_ids = id_group[latest_id]['detection_ids'][:]
        
        # Load interpolated if available
        self.interpolated_data = None
        if 'interpolated_detections' in self.root:
            interp_group = self.root['interpolated_detections']
            if 'latest' in interp_group.attrs:
                latest_interp = interp_group.attrs['latest']
                interp_data = interp_group[latest_interp]
                self.interpolated_data = {
                    'frame_indices': interp_data['frame_indices'][:],
                    'roi_ids': interp_data['roi_ids'][:],
                    'bboxes': interp_data['bboxes'][:]
                }
        
        self.total_frames = total_frames
        console.print(f"[green]✓ Loaded {total_frames} frames of detection data[/green]")
    
    def get_roi_trajectory(self, roi_id: int) -> Dict:
        """
        Extract trajectory for a specific ROI.
        
        Returns:
            Dictionary with frames, x_coords, y_coords arrays
        """
        frames = []
        x_coords = []
        y_coords = []
        
        cumulative_idx = 0
        for frame_idx in range(self.total_frames):
            frame_det_count = int(self.n_detections[frame_idx])
            
            if frame_det_count > 0:
                frame_detection_ids = self.detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    bbox = self.bbox_coords[cumulative_idx + roi_idx]
                    
                    # Convert normalized coords to pixels
                    center_x = bbox[0] * self.width
                    center_y = bbox[1] * self.height
                    
                    frames.append(frame_idx)
                    x_coords.append(center_x)
                    y_coords.append(center_y)
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions if available
        if self.interpolated_data is not None:
            interp_frames = set()
            for j in range(len(self.interpolated_data['frame_indices'])):
                frame_idx = int(self.interpolated_data['frame_indices'][j])
                if frame_idx < self.total_frames and int(self.interpolated_data['roi_ids'][j]) == roi_id:
                    if frame_idx not in frames:
                        bbox = self.interpolated_data['bboxes'][j]
                        center_x = bbox[0] * self.width
                        center_y = bbox[1] * self.height
                        
                        frames.append(frame_idx)
                        x_coords.append(center_x)
                        y_coords.append(center_y)
                        interp_frames.add(frame_idx)
            
            # Sort by frame number if we added interpolated data
            if interp_frames:
                sorted_indices = np.argsort(frames)
                frames = [frames[i] for i in sorted_indices]
                x_coords = [x_coords[i] for i in sorted_indices]
                y_coords = [y_coords[i] for i in sorted_indices]
        
        return {
            'frames': np.array(frames),
            'x': np.array(x_coords),
            'y': np.array(y_coords),
            'roi_id': roi_id
        }
    
    def plot_all_trajectories(self, save_path: Optional[str] = None, 
                            exclude_fish: List[int] = None,
                            show_start_end: bool = True,
                            show_background: bool = True):
        """
        Plot all fish trajectories on a single figure.
        
        Args:
            save_path: Path to save figure
            exclude_fish: List of fish IDs to exclude
            show_start_end: Whether to mark start and end positions
            show_background: Whether to show background image
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Show background image if available
        if show_background and self.background_image is not None:
            ax.imshow(self.background_image, cmap='gray', alpha=0.7,
                     extent=[0, self.width, self.height, 0])
        
        # Set up colors for different fish/groups
        colors_group1 = plt.cm.Blues(np.linspace(0.4, 0.9, 6))
        colors_group2 = plt.cm.Reds(np.linspace(0.4, 0.9, 6))
        
        # Track which fish we've plotted
        plotted_fish = []
        
        for roi_id in range(12):  # Assuming 12 fish
            # Skip excluded fish
            if exclude_fish and roi_id in exclude_fish:
                continue
                
            trajectory = self.get_roi_trajectory(roi_id)
            
            if len(trajectory['frames']) > 100:  # Only plot if we have enough data
                # Determine color based on group
                if roi_id <= 5:
                    color = colors_group1[roi_id]
                    group_label = 'Group 1'
                else:
                    color = colors_group2[roi_id - 6]
                    group_label = 'Group 2'
                
                # Plot trajectory with higher contrast if background is shown
                linewidth = 1.2 if show_background else 0.8
                alpha = 0.8 if show_background else 0.6
                
                ax.plot(trajectory['x'], trajectory['y'], 
                       color=color, alpha=alpha, linewidth=linewidth,
                       label=f'Fish {roi_id} ({group_label})')
                
                if show_start_end:
                    # Mark start position with white edge for contrast
                    ax.scatter(trajectory['x'][0], trajectory['y'][0],
                             color=color, s=100, marker='o', 
                             edgecolor='white', linewidth=2, zorder=5)
                    
                    # Mark end position with white edge for contrast
                    ax.scatter(trajectory['x'][-1], trajectory['y'][-1],
                             color=color, s=100, marker='s',
                             edgecolor='white', linewidth=2, zorder=5)
                
                plotted_fish.append(roi_id)
        
        # Set arena boundaries
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinates (y increases downward)
        
        # Add arena outline with white color for visibility
        arena_rect = Rectangle((0, 0), self.width, self.height, 
                              fill=False, edgecolor='white', linewidth=2)
        ax.add_patch(arena_rect)
        
        # Labels and title
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        ax.set_title(f'Fish Trajectories - First {self.duration_minutes} Minutes\n'
                    f'({len(plotted_fish)} fish tracked)', 
                    fontsize=14, fontweight='bold')
        
        # Add legend with semi-transparent background
        if len(plotted_fish) <= 6:
            legend = ax.legend(loc='upper right', fontsize=9, 
                              framealpha=0.95, facecolor='white')
            legend.get_frame().set_edgecolor('gray')
        
        # Add scale bar (500 pixels) with white background for visibility
        scale_bar_x = self.width - 700
        scale_bar_y = self.height - 200
        
        # Add white background rectangle for scale bar
        scale_bg = Rectangle((scale_bar_x - 50, scale_bar_y - 100), 
                            600, 150, fill=True, facecolor='white', 
                            alpha=0.8, edgecolor='gray')
        ax.add_patch(scale_bg)
        
        ax.plot([scale_bar_x, scale_bar_x + 500], [scale_bar_y, scale_bar_y],
               'k-', linewidth=3)
        ax.text(scale_bar_x + 250, scale_bar_y - 50, '500 px', 
               ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension
            ext = save_file.suffix.lower()
            if ext == '.svg':
                plt.savefig(save_file, format='svg', bbox_inches='tight')
            elif ext == '.pdf':
                plt.savefig(save_file, format='pdf', bbox_inches='tight')
            else:
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
            
            console.print(f"[green]✓ Saved trajectory plot to {save_file}[/green]")
        
        plt.show()
    
    def plot_grid_trajectories(self, save_path: Optional[str] = None,
                              exclude_fish: List[int] = None):
        """
        Plot individual trajectories in a grid layout.
        
        Args:
            save_path: Path to save figure
            exclude_fish: List of fish IDs to exclude
        """
        # Determine which fish to plot
        fish_to_plot = []
        for roi_id in range(12):
            if exclude_fish and roi_id in exclude_fish:
                continue
            trajectory = self.get_roi_trajectory(roi_id)
            if len(trajectory['frames']) > 100:
                fish_to_plot.append(roi_id)
        
        n_fish = len(fish_to_plot)
        if n_fish == 0:
            console.print("[red]No fish with sufficient data to plot[/red]")
            return
        
        # Create grid layout
        n_cols = 4
        n_rows = (n_fish + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for idx, roi_id in enumerate(fish_to_plot):
            ax = axes_flat[idx]
            trajectory = self.get_roi_trajectory(roi_id)
            
            # Create time-based coloring
            time_seconds = trajectory['frames'] / self.fps
            
            # Plot with time gradient
            scatter = ax.scatter(trajectory['x'], trajectory['y'],
                               c=time_seconds, cmap='viridis',
                               s=1, alpha=0.7)
            
            # Add start and end markers
            ax.plot(trajectory['x'][0], trajectory['y'][0],
                   'go', markersize=6, label='Start')
            ax.plot(trajectory['x'][-1], trajectory['y'][-1],
                   'ro', markersize=6, label='End')
            
            # Set limits and aspect
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Title and labels
            group = 1 if roi_id <= 5 else 2
            ax.set_title(f'Fish {roi_id} (Group {group})', fontsize=10)
            ax.set_xlabel('X (px)', fontsize=8)
            ax.set_ylabel('Y (px)', fontsize=8)
            ax.tick_params(labelsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time (s)', fontsize=8)
            cbar.ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(n_fish, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        fig.suptitle(f'Individual Fish Trajectories - First {self.duration_minutes} Minutes',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            console.print(f"[green]✓ Saved grid plot to {save_file}[/green]")
        
        plt.show()
    
    def calculate_trajectory_stats(self, exclude_fish: List[int] = None) -> Dict:
        """Calculate statistics for trajectories."""
        stats = {}
        
        for roi_id in range(12):
            if exclude_fish and roi_id in exclude_fish:
                continue
                
            trajectory = self.get_roi_trajectory(roi_id)
            
            if len(trajectory['frames']) > 100:
                # Calculate total distance
                dx = np.diff(trajectory['x'])
                dy = np.diff(trajectory['y'])
                distances = np.sqrt(dx**2 + dy**2)
                total_distance = np.sum(distances)
                
                # Calculate area covered (convex hull would be better but this is simpler)
                x_range = trajectory['x'].max() - trajectory['x'].min()
                y_range = trajectory['y'].max() - trajectory['y'].min()
                area_covered = x_range * y_range
                
                # Detection rate
                detection_rate = len(trajectory['frames']) / self.total_frames
                
                stats[roi_id] = {
                    'total_distance_px': total_distance,
                    'area_covered_px2': area_covered,
                    'detection_rate': detection_rate,
                    'n_detections': len(trajectory['frames']),
                    'group': 1 if roi_id <= 5 else 2
                }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Plot fish trajectories for the first 5 minutes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all trajectories with background from H5
  python first_5min_trajectory.py detections.zarr --h5 experiment.h5 --save trajectories_5min.svg
  
  # Use specific background image
  python first_5min_trajectory.py detections.zarr --bg-image background_model.png
  
  # Exclude specific fish
  python first_5min_trajectory.py detections.zarr --exclude 3 4 5 11 --h5 experiment.h5
  
  # Plot without background
  python first_5min_trajectory.py detections.zarr --no-background
  
  # Custom duration (e.g., first 3 minutes)
  python first_5min_trajectory.py detections.zarr --minutes 3 --h5 experiment.h5
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with detections')
    parser.add_argument('--h5', type=str, help='Path to H5 file containing background image')
    parser.add_argument('--bg-image', type=str, help='Direct path to background image')
    parser.add_argument('--minutes', type=float, default=5.0,
                       help='Duration to plot in minutes (default: 5.0)')
    parser.add_argument('--exclude', nargs='+', type=int,
                       help='Fish IDs to exclude from plotting')
    parser.add_argument('--save', type=str,
                       help='Path to save plot (format determined by extension)')
    parser.add_argument('--grid', action='store_true',
                       help='Plot trajectories in grid layout')
    parser.add_argument('--no-markers', action='store_true',
                       help='Hide start/end markers')
    parser.add_argument('--no-background', action='store_true',
                       help='Do not show background image')
    parser.add_argument('--stats', action='store_true',
                       help='Print trajectory statistics')
    
    args = parser.parse_args()
    
    # Create plotter
    console.print("[bold]First 5 Minutes Trajectory Plotter[/bold]\n")
    plotter = First5MinTrajectoryPlotter(
        zarr_path=args.zarr_path,
        duration_minutes=args.minutes,
        h5_path=args.h5,
        bg_image_path=args.bg_image
    )
    
    # Print statistics if requested
    if args.stats:
        console.print("\n[cyan]Calculating trajectory statistics...[/cyan]")
        stats = plotter.calculate_trajectory_stats(exclude_fish=args.exclude)
        
        console.print("\n[bold]Trajectory Statistics:[/bold]")
        for roi_id, stat in sorted(stats.items()):
            console.print(f"\nFish {roi_id} (Group {stat['group']}):")
            console.print(f"  Total distance: {stat['total_distance_px']:.1f} pixels")
            console.print(f"  Area covered: {stat['area_covered_px2']:.0f} px²")
            console.print(f"  Detection rate: {stat['detection_rate']:.1%}")
    
    # Create plots
    if args.grid:
        console.print("\n[cyan]Creating grid plot...[/cyan]")
        plotter.plot_grid_trajectories(
            save_path=args.save,
            exclude_fish=args.exclude
        )
    else:
        console.print("\n[cyan]Creating trajectory plot...[/cyan]")
        plotter.plot_all_trajectories(
            save_path=args.save,
            exclude_fish=args.exclude,
            show_start_end=not args.no_markers,
            show_background=not args.no_background
        )
    
    console.print("\n[green]✓ Complete![/green]")


if __name__ == '__main__':
    main()