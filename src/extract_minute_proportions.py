#!/usr/bin/env python3
"""
Minute-by-Minute Quadrant Proportions Extractor

Extracts minute-binned top/bottom proportions for each fish from zarr/h5 files.
Outputs CSV with minute-level resolution instead of trial-level.
"""

import zarr
import h5py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
warnings.filterwarnings('ignore')

console = Console()

@dataclass
class MinuteData:
    """Data for one fish during one minute bin."""
    roi_id: int
    minute: float  # Center of minute bin
    minute_start: float  # Start of bin
    minute_end: float  # End of bin
    top_proportion: float  # Proportion of valid detections in top half
    bottom_proportion: float  # Proportion in bottom half
    detection_rate: float  # Proportion of frames with valid detection
    n_frames: int  # Total frames in bin
    n_valid: int  # Frames with valid detection
    n_top: int  # Frames in top quadrant
    n_bottom: int  # Frames in bottom quadrant


class MinuteProportionExtractor:
    """Extract minute-by-minute quadrant proportions from zarr/h5 files."""
    
    def __init__(self, zarr_path: str, h5_path: str, bin_minutes: float = 1.0):
        """
        Initialize extractor.
        
        Args:
            zarr_path: Path to zarr file with detections
            h5_path: Path to H5 file (for metadata/fps if needed)
            bin_minutes: Size of time bins in minutes
        """
        self.zarr_path = Path(zarr_path)
        self.h5_path = Path(h5_path)
        self.bin_minutes = bin_minutes
        
        # Load zarr data
        self.root = zarr.open_group(self.zarr_path, mode='r')
        
        # Get dimensions and fps
        self.camera_width = self.root.attrs.get('width', 4512)
        self.camera_height = self.root.attrs.get('height', 4512)
        self.fps = self.root.attrs.get('fps', 60.0)
        
        console.print(f"[cyan]Camera: {self.camera_width}x{self.camera_height}, {self.fps} fps[/cyan]")
        console.print(f"[cyan]Bin size: {bin_minutes} minutes[/cyan]")
        
        # Load ROI boundaries
        self.load_roi_boundaries()
        
        # Load detection data
        self.load_detections()
        
    def load_roi_boundaries(self):
        """Load predefined ROI boundaries from config."""
        # Hardcoded from config file - these are in 640x640 space
        sub_dish_rois = [
            {'id': 0, 'roi_pixels': [59, 73, 71, 180]},
            {'id': 1, 'roi_pixels': [139, 74, 71, 178]},
            {'id': 2, 'roi_pixels': [241, 76, 70, 180]},
            {'id': 3, 'roi_pixels': [320, 73, 70, 183]},
            {'id': 4, 'roi_pixels': [425, 76, 71, 183]},
            {'id': 5, 'roi_pixels': [503, 70, 73, 188]},
            {'id': 6, 'roi_pixels': [53, 272, 75, 183]},
            {'id': 7, 'roi_pixels': [137, 275, 71, 181]},
            {'id': 8, 'roi_pixels': [236, 271, 72, 185]},
            {'id': 9, 'roi_pixels': [317, 272, 70, 185]},
            {'id': 10, 'roi_pixels': [421, 273, 73, 185]},
            {'id': 11, 'roi_pixels': [502, 275, 70, 184]}
        ]
        
        # Convert from 640x640 to camera resolution
        scale = self.camera_width / 640
        
        self.roi_boundaries = {}
        for roi_info in sub_dish_rois:
            roi_id = roi_info['id']
            x, y, width, height = roi_info['roi_pixels']
            
            # Scale to camera resolution
            x_min = x * scale
            y_min = y * scale
            x_max = (x + width) * scale
            y_max = (y + height) * scale
            
            self.roi_boundaries[roi_id] = {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'y_center': (y_min + y_max) / 2
            }
        
        self.num_rois = len(self.roi_boundaries)
        console.print(f"[cyan]Loaded {self.num_rois} ROI boundaries[/cyan]")
    
    def load_detections(self):
        """Load detection data from zarr."""
        # Get detection data
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        self.n_detections = detect_group[latest_detect]['n_detections'][:]
        self.bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Get ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        self.detection_ids = id_group[latest_id]['detection_ids'][:]
        self.n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
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
                console.print(f"[cyan]Loaded interpolated detections[/cyan]")
        
        self.total_frames = len(self.n_detections)
        self.total_minutes = self.total_frames / (self.fps * 60)
        console.print(f"[cyan]Total: {self.total_frames} frames ({self.total_minutes:.1f} minutes)[/cyan]")
    
    def get_roi_positions(self, roi_id: int) -> np.ndarray:
        """
        Get all positions for a specific ROI.
        
        Returns:
            Array of [x, y] positions in camera pixels (NaN for missing frames)
        """
        positions = np.full((self.total_frames, 2), np.nan)
        
        # Get original detections
        cumulative_idx = 0
        for frame_idx in range(self.total_frames):
            frame_det_count = int(self.n_detections[frame_idx])
            
            if frame_det_count > 0 and self.n_detections_per_roi[frame_idx, roi_id] > 0:
                frame_detection_ids = self.detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    bbox = self.bbox_coords[cumulative_idx + roi_idx]
                    
                    # bbox[0] and bbox[1] are centers
                    center_x_norm = bbox[0]
                    center_y_norm = bbox[1]
                    
                    # Convert to camera pixels
                    centroid_x_ds = center_x_norm * 640
                    centroid_y_ds = center_y_norm * 640
                    scale = self.camera_width / 640
                    
                    positions[frame_idx, 0] = centroid_x_ds * scale
                    positions[frame_idx, 1] = centroid_y_ds * scale
            
            cumulative_idx += frame_det_count
        
        # Add interpolated positions
        if self.interpolated_data is not None:
            for j in range(len(self.interpolated_data['frame_indices'])):
                frame_idx = int(self.interpolated_data['frame_indices'][j])
                if frame_idx < self.total_frames and int(self.interpolated_data['roi_ids'][j]) == roi_id:
                    if np.isnan(positions[frame_idx, 0]):  # Only use if no original detection
                        bbox = self.interpolated_data['bboxes'][j]
                        center_x_norm = bbox[0]
                        center_y_norm = bbox[1]
                        
                        centroid_x_ds = center_x_norm * 640
                        centroid_y_ds = center_y_norm * 640
                        scale = self.camera_width / 640
                        
                        positions[frame_idx, 0] = centroid_x_ds * scale
                        positions[frame_idx, 1] = centroid_y_ds * scale
        
        return positions
    
    def assign_quadrant(self, y_position: float, roi_boundaries: Dict[str, float]) -> int:
        """
        Assign quadrant based on y-position within ROI.
        
        Returns:
            1 for top half, 2 for bottom half, -1 for invalid/outside
        """
        if np.isnan(y_position):
            return -1
        
        # Check if within ROI bounds
        if y_position < roi_boundaries['y_min'] or y_position > roi_boundaries['y_max']:
            return -1
        
        # Check top vs bottom
        if y_position < roi_boundaries['y_center']:
            return 1  # Top half
        else:
            return 2  # Bottom half
    
    def extract_minute_proportions(self) -> pd.DataFrame:
        """
        Extract minute-by-minute proportions for all fish.
        
        Returns:
            DataFrame with minute-binned quadrant proportions
        """
        all_data = []
        
        # Create time bins
        time_bins = np.arange(0, self.total_minutes + self.bin_minutes, self.bin_minutes)
        n_bins = len(time_bins) - 1
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            total_analyses = self.num_rois * n_bins
            task = progress.add_task(
                f"[cyan]Extracting minute proportions for {self.num_rois} fish...", 
                total=total_analyses
            )
            
            for roi_id in range(self.num_rois):
                # Get all positions for this fish
                positions = self.get_roi_positions(roi_id)
                roi_bounds = self.roi_boundaries[roi_id]
                
                # Assign quadrants
                quadrants = np.array([self.assign_quadrant(y, roi_bounds) 
                                    for y in positions[:, 1]])
                
                # Process each minute bin
                for i in range(n_bins):
                    minute_start = time_bins[i]
                    minute_end = time_bins[i + 1]
                    
                    # Get frames in this minute
                    frame_start = int(minute_start * 60 * self.fps)
                    frame_end = min(int(minute_end * 60 * self.fps), self.total_frames)
                    
                    if frame_start < frame_end:
                        bin_quadrants = quadrants[frame_start:frame_end]
                        
                        # Count quadrant occupancy
                        n_frames = len(bin_quadrants)
                        valid_mask = bin_quadrants != -1
                        n_valid = np.sum(valid_mask)
                        
                        if n_valid > 0:
                            n_top = np.sum(bin_quadrants[valid_mask] == 1)
                            n_bottom = np.sum(bin_quadrants[valid_mask] == 2)
                            
                            top_prop = n_top / n_valid
                            bottom_prop = n_bottom / n_valid
                            detection_rate = n_valid / n_frames
                        else:
                            n_top = n_bottom = 0
                            top_prop = bottom_prop = np.nan
                            detection_rate = 0
                        
                        # Use end of bin instead of center for cleaner alignment
                        minute_data = MinuteData(
                            roi_id=roi_id,
                            minute=minute_end,  # Use end of bin (1.0, 2.0, etc.)
                            minute_start=minute_start,
                            minute_end=minute_end,
                            top_proportion=top_prop,
                            bottom_proportion=bottom_prop,
                            detection_rate=detection_rate,
                            n_frames=n_frames,
                            n_valid=n_valid,
                            n_top=n_top,
                            n_bottom=n_bottom
                        )
                        
                        all_data.append(minute_data.__dict__)
                    
                    progress.advance(task)
        
        df = pd.DataFrame(all_data)
        
        # Add group labels
        df['group'] = df['roi_id'].apply(lambda x: 1 if x <= 5 else 2)
        
        # Add preference score
        df['preference_score'] = df['top_proportion'] - df['bottom_proportion']
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
        
        # Overall stats
        valid_df = df[df['detection_rate'] > 0]
        console.print(f"\nTotal minute bins: {len(df)}")
        console.print(f"Bins with valid data: {len(valid_df)} ({100*len(valid_df)/len(df):.1f}%)")
        
        # Per-fish summary
        console.print("\n[bold]Per-fish averages:[/bold]")
        fish_summary = valid_df.groupby('roi_id').agg({
            'top_proportion': 'mean',
            'detection_rate': 'mean'
        })
        
        for roi_id, row in fish_summary.iterrows():
            console.print(f"  Fish {roi_id}: {row['top_proportion']:.1%} top (detection: {row['detection_rate']:.1%})")
        
        # Group comparison
        console.print("\n[bold]Group comparison:[/bold]")
        group_summary = valid_df.groupby('group').agg({
            'top_proportion': ['mean', 'std'],
            'detection_rate': 'mean'
        })
        
        for group in [1, 2]:
            if group in group_summary.index:
                row = group_summary.loc[group]
                console.print(f"  Group {group}: {row['top_proportion']['mean']:.1%} ± {row['top_proportion']['std']:.1%}")
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save results to CSV."""
        df.to_csv(output_path, index=False)
        console.print(f"[green]✓ Saved {len(df)} minute bins to {output_path}[/green]")
        
        # Also save a summary version
        summary_path = Path(output_path).with_suffix('.summary.csv')
        summary_df = df.groupby(['roi_id', 'group']).agg({
            'top_proportion': ['mean', 'std', 'count'],
            'detection_rate': 'mean',
            'n_valid': 'sum',
            'n_top': 'sum',
            'n_bottom': 'sum'
        }).reset_index()
        
        # Flatten column names
        summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
        summary_df.to_csv(summary_path, index=False)
        console.print(f"[green]✓ Saved summary to {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description='Extract minute-by-minute quadrant proportions from zarr/h5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file with detections')
    parser.add_argument('h5_path', help='Path to H5 file with metadata')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--bin-minutes', type=float, default=1.0,
                       help='Bin size in minutes (default: 1.0)')
    parser.add_argument('--min-detection', type=float, default=0.0,
                       help='Minimum detection rate to include bin (0-1, default: 0)')
    
    args = parser.parse_args()
    
    # Create extractor
    console.print("[bold]Minute-by-Minute Proportion Extractor[/bold]")
    console.print(f"Zarr: {args.zarr_path}")
    console.print(f"H5: {args.h5_path}")
    console.print(f"Bin size: {args.bin_minutes} minutes\n")
    
    extractor = MinuteProportionExtractor(
        zarr_path=args.zarr_path,
        h5_path=args.h5_path,
        bin_minutes=args.bin_minutes
    )
    
    # Extract proportions
    console.print("\n[cyan]Extracting proportions...[/cyan]")
    df = extractor.extract_minute_proportions()
    
    # Filter by detection rate if specified
    if args.min_detection > 0:
        original_len = len(df)
        df = df[df['detection_rate'] >= args.min_detection]
        console.print(f"[yellow]Filtered from {original_len} to {len(df)} bins (detection >= {args.min_detection:.0%})[/yellow]")
    
    # Print summary
    extractor.print_summary(df)
    
    # Save results
    console.print("\n[cyan]Saving results...[/cyan]")
    extractor.save_results(df, args.output)
    
    # Print sample data
    console.print("\n[bold]Sample output (first 10 rows):[/bold]")
    sample_cols = ['roi_id', 'minute', 'top_proportion', 'detection_rate', 'group']
    console.print(df[sample_cols].head(10).to_string(index=False))
    
    console.print("\n[green]✓ Extraction complete![/green]")


if __name__ == '__main__':
    main()