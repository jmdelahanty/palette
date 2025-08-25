#!/usr/bin/env python3
"""
Batch ROI Interpolator

Performs interpolation for all ROIs in a detection zarr file.
Handles multiple ROIs efficiently with configurable parameters.
"""

import zarr
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from scipy import interpolate
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import warnings
warnings.filterwarnings('ignore')

console = Console()


class BatchROIInterpolator:
    """Batch interpolation for multiple ROIs."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.root = zarr.open_group(self.zarr_path, mode='r+')
        
        # Get metadata
        self.img_width = self.root.attrs.get('width', 4512)
        self.img_height = self.root.attrs.get('height', 4512)
        self.fps = self.root.attrs.get('fps', 60.0)
    
    def get_roi_detections(self, roi_id: int):
        """Extract all detections for a specific ROI."""
        # Load detection data
        detect_group = self.root['detect_runs']
        latest_detect = detect_group.attrs['latest']
        n_detections = detect_group[latest_detect]['n_detections'][:]
        bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
        
        # Check if scores exist, otherwise use default
        if 'scores' in detect_group[latest_detect]:
            scores = detect_group[latest_detect]['scores'][:]
        else:
            # Use confidence scores if available, otherwise default to 1.0
            if 'confidence_scores' in detect_group[latest_detect]:
                scores = detect_group[latest_detect]['confidence_scores'][:]
            else:
                # Create default scores
                scores = np.ones(len(bbox_coords), dtype=np.float32)
        
        # Check if class_ids exist, otherwise use default
        if 'class_ids' in detect_group[latest_detect]:
            class_ids = detect_group[latest_detect]['class_ids'][:]
        else:
            # Default to class 0 (fish)
            class_ids = np.zeros(len(bbox_coords), dtype=np.int32)
        
        # Load ID assignments
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        detection_ids = id_group[latest_id]['detection_ids'][:]
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        
        # Build detection map for this ROI
        detections = {}
        cumulative_idx = 0
        
        for frame_idx in range(len(n_detections)):
            frame_det_count = int(n_detections[frame_idx])
            
            if frame_det_count > 0 and n_detections_per_roi[frame_idx, roi_id] > 0:
                frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
                roi_mask = frame_detection_ids == roi_id
                
                if np.any(roi_mask):
                    roi_idx = np.where(roi_mask)[0][0]
                    detections[frame_idx] = {
                        'bbox': bbox_coords[cumulative_idx + roi_idx],
                        'score': scores[cumulative_idx + roi_idx] if cumulative_idx + roi_idx < len(scores) else 1.0,
                        'class_id': class_ids[cumulative_idx + roi_idx] if cumulative_idx + roi_idx < len(class_ids) else 0
                    }
            
            cumulative_idx += frame_det_count
        
        return detections, len(n_detections)
    
    def find_gaps(self, detections: dict, total_frames: int, max_gap: int = 600):
        """Find gaps in detections that can be interpolated."""
        detected_frames = sorted(detections.keys())
        gaps = []
        
        if not detected_frames:
            return gaps
        
        # Find gaps between consecutive detections
        for i in range(len(detected_frames) - 1):
            start_frame = detected_frames[i]
            end_frame = detected_frames[i + 1]
            gap_size = end_frame - start_frame - 1
            
            if 0 < gap_size <= max_gap:
                gaps.append({
                    'start': start_frame,
                    'end': end_frame,
                    'size': gap_size
                })
        
        return gaps
    
    def interpolate_roi(self, roi_id: int, max_gap: int = 600, method: str = 'linear'):
        """
        Interpolate missing detections for a single ROI.
        
        Args:
            roi_id: ROI to interpolate
            max_gap: Maximum gap size to interpolate
            method: Interpolation method ('linear', 'cubic', 'nearest')
            
        Returns:
            Dict with interpolation results
        """
        # Get existing detections
        detections, total_frames = self.get_roi_detections(roi_id)
        
        if len(detections) < 2:
            return {
                'roi_id': roi_id,
                'original_count': len(detections),
                'interpolated_count': 0,
                'gaps_filled': 0,
                'coverage_before': 0,
                'coverage_after': 0,
                'interpolated_frames': []
            }
        
        # Find gaps
        gaps = self.find_gaps(detections, total_frames, max_gap)
        
        # Prepare interpolation data
        detected_frames = sorted(detections.keys())
        bbox_coords = np.array([detections[f]['bbox'] for f in detected_frames])
        scores = np.array([detections[f]['score'] for f in detected_frames])
        
        # Get a valid class_id (use the most common one or 0)
        class_ids_in_detections = [detections[f]['class_id'] for f in detected_frames]
        if class_ids_in_detections:
            # Use the first detection's class_id (should be consistent for same ROI)
            default_class_id = class_ids_in_detections[0]
        else:
            default_class_id = 0
        
        # Create interpolation functions for each bbox coordinate
        interp_funcs = []
        for coord_idx in range(4):
            if method == 'cubic' and len(detected_frames) >= 4:
                f = interpolate.interp1d(detected_frames, bbox_coords[:, coord_idx], 
                                        kind='cubic', fill_value='extrapolate')
            else:
                f = interpolate.interp1d(detected_frames, bbox_coords[:, coord_idx], 
                                        kind='linear', fill_value='extrapolate')
            interp_funcs.append(f)
        
        # Score interpolation (always linear)
        score_interp = interpolate.interp1d(detected_frames, scores, 
                                           kind='linear', fill_value='extrapolate')
        
        # Perform interpolation
        interpolated_frames = []
        
        for gap in gaps:
            gap_frames = list(range(gap['start'] + 1, gap['end']))
            
            for frame_idx in gap_frames:
                # Interpolate bbox
                interp_bbox = np.array([f(frame_idx) for f in interp_funcs])
                
                # Ensure bbox is within bounds
                interp_bbox[0] = np.clip(interp_bbox[0], 0, 1)  # x_min
                interp_bbox[1] = np.clip(interp_bbox[1], 0, 1)  # y_min
                interp_bbox[2] = np.clip(interp_bbox[2], 0, 1)  # x_max
                interp_bbox[3] = np.clip(interp_bbox[3], 0, 1)  # y_max
                
                # Interpolate score
                interp_score = np.clip(score_interp(frame_idx), 0, 1)
                
                interpolated_frames.append({
                    'frame_idx': frame_idx,
                    'roi_id': roi_id,
                    'bbox': interp_bbox,
                    'score': float(interp_score),
                    'class_id': default_class_id
                })
        
        # Calculate statistics
        coverage_before = len(detections) / total_frames * 100
        coverage_after = (len(detections) + len(interpolated_frames)) / total_frames * 100
        
        return {
            'roi_id': roi_id,
            'original_count': len(detections),
            'interpolated_count': len(interpolated_frames),
            'gaps_filled': len(gaps),
            'coverage_before': coverage_before,
            'coverage_after': coverage_after,
            'interpolated_frames': interpolated_frames,
            'total_frames': total_frames
        }
    
    def interpolate_all_rois(self, max_gap: int = 600, method: str = 'linear',
                            skip_existing: bool = True):
        """
        Interpolate all ROIs in the dataset.
        
        Args:
            max_gap: Maximum gap size to interpolate
            method: Interpolation method
            skip_existing: Skip if interpolation already exists
            
        Returns:
            Dict with results for all ROIs
        """
        # Check for existing interpolation
        if skip_existing and 'interpolated_detections' in self.root:
            if 'latest' in self.root['interpolated_detections'].attrs:
                console.print("[yellow]Interpolation already exists. Use --force to overwrite.[/yellow]")
                return None
        
        # Get number of ROIs
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        id_group = self.root[id_key]
        latest_id = id_group.attrs['latest']
        n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
        num_rois = n_detections_per_roi.shape[1]
        
        all_results = {}
        all_interpolated = []
        
        # Process each ROI with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"[cyan]Interpolating {num_rois} ROIs...", total=num_rois)
            
            for roi_id in range(num_rois):
                progress.update(task, description=f"[cyan]Processing ROI {roi_id}/{num_rois-1}")
                
                results = self.interpolate_roi(roi_id, max_gap, method)
                all_results[roi_id] = results
                
                # Collect all interpolated frames
                all_interpolated.extend(results['interpolated_frames'])
                
                progress.advance(task)
        
        # Sort interpolated frames by frame index
        all_interpolated.sort(key=lambda x: (x['frame_idx'], x['roi_id']))
        
        return {
            'roi_results': all_results,
            'all_interpolated': all_interpolated,
            'total_interpolated': len(all_interpolated),
            'num_rois': num_rois,
            'max_gap': max_gap,
            'method': method
        }
    
    def save_interpolation(self, results: dict):
        """Save interpolation results to zarr."""
        if not results or not results['all_interpolated']:
            console.print("[yellow]No interpolations to save[/yellow]")
            return
        
        # Create interpolated_detections group
        if 'interpolated_detections' not in self.root:
            self.root.create_group('interpolated_detections')
        
        interp_group = self.root['interpolated_detections']
        
        # Create timestamped run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'interp_{timestamp}'
        run_group = interp_group.create_group(run_name)
        
        # Prepare arrays
        interpolated = results['all_interpolated']
        n_interp = len(interpolated)
        
        frame_indices = np.array([d['frame_idx'] for d in interpolated], dtype=np.int32)
        roi_ids = np.array([d['roi_id'] for d in interpolated], dtype=np.int32)
        bboxes = np.array([d['bbox'] for d in interpolated], dtype=np.float32)
        scores = np.array([d['score'] for d in interpolated], dtype=np.float32)
        class_ids = np.array([d['class_id'] for d in interpolated], dtype=np.int32)
        
        # Save arrays
        run_group.create_dataset('frame_indices', data=frame_indices, dtype='int32')
        run_group.create_dataset('roi_ids', data=roi_ids, dtype='int32')
        run_group.create_dataset('bboxes', data=bboxes, dtype='float32')
        run_group.create_dataset('scores', data=scores, dtype='float32')
        run_group.create_dataset('class_ids', data=class_ids, dtype='int32')
        
        # Save metadata
        run_group.attrs.update({
            'created_at': datetime.now().isoformat(),
            'total_interpolated': n_interp,
            'num_rois': results['num_rois'],
            'max_gap': results['max_gap'],
            'method': results['method']
        })
        
        # Save per-ROI statistics
        roi_stats = {}
        for roi_id, roi_result in results['roi_results'].items():
            roi_stats[str(roi_id)] = {
                'original_count': roi_result['original_count'],
                'interpolated_count': roi_result['interpolated_count'],
                'gaps_filled': roi_result['gaps_filled'],
                'coverage_before': roi_result['coverage_before'],
                'coverage_after': roi_result['coverage_after']
            }
        
        run_group.attrs['roi_statistics'] = str(roi_stats)
        
        # Update latest
        interp_group.attrs['latest'] = run_name
        
        console.print(f"[green]✓ Interpolation saved to:[/green] interpolated_detections/{run_name}")
        console.print(f"[green]  Total interpolated frames:[/green] {n_interp}")
    
    def print_summary(self, results: dict):
        """Print summary table of interpolation results."""
        if not results:
            return
        
        table = Table(title="Interpolation Results Summary")
        
        table.add_column("ROI", style="cyan", no_wrap=True)
        table.add_column("Original", style="yellow")
        table.add_column("Added", style="green")
        table.add_column("Gaps", style="magenta")
        table.add_column("Before %", style="red")
        table.add_column("After %", style="blue")
        table.add_column("Improvement", style="bold green")
        
        total_original = 0
        total_added = 0
        total_gaps = 0
        
        for roi_id in sorted(results['roi_results'].keys()):
            roi_result = results['roi_results'][roi_id]
            
            improvement = roi_result['coverage_after'] - roi_result['coverage_before']
            
            table.add_row(
                str(roi_id),
                str(roi_result['original_count']),
                str(roi_result['interpolated_count']),
                str(roi_result['gaps_filled']),
                f"{roi_result['coverage_before']:.1f}%",
                f"{roi_result['coverage_after']:.1f}%",
                f"+{improvement:.1f}%"
            )
            
            total_original += roi_result['original_count']
            total_added += roi_result['interpolated_count']
            total_gaps += roi_result['gaps_filled']
        
        # Add totals row
        table.add_row(
            "[bold]TOTAL",
            f"[bold]{total_original}",
            f"[bold]{total_added}",
            f"[bold]{total_gaps}",
            "",
            "",
            "",
            style="bold yellow"
        )
        
        console.print(table)
        
        # Print overall statistics
        console.print(f"\n[bold cyan]Overall Statistics:[/bold cyan]")
        console.print(f"  Total ROIs processed: {results['num_rois']}")
        console.print(f"  Total frames added: {results['total_interpolated']}")
        console.print(f"  Max gap size: {results['max_gap']} frames")
        console.print(f"  Interpolation method: {results['method']}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch interpolation for all ROIs in detection data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic interpolation with defaults
  %(prog)s detections.zarr
  
  # Custom gap size and method
  %(prog)s detections.zarr --max-gap 300 --method cubic
  
  # Force overwrite existing interpolation
  %(prog)s detections.zarr --force
  
  # Dry run without saving
  %(prog)s detections.zarr --dry-run
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--max-gap', type=int, default=600,
                       help='Maximum gap size to interpolate (default: 600 frames)')
    parser.add_argument('--method', choices=['linear', 'cubic', 'nearest'],
                       default='linear',
                       help='Interpolation method (default: linear)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing interpolation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Initialize interpolator
    interpolator = BatchROIInterpolator(args.zarr_path, verbose=not args.quiet)
    
    console.print(f"\n[bold cyan]Batch ROI Interpolation[/bold cyan]")
    console.print(f"File: {args.zarr_path}")
    console.print(f"Max gap: {args.max_gap} frames")
    console.print(f"Method: {args.method}")
    
    # Perform interpolation
    results = interpolator.interpolate_all_rois(
        max_gap=args.max_gap,
        method=args.method,
        skip_existing=not args.force
    )
    
    if results:
        # Print summary
        interpolator.print_summary(results)
        
        # Save if not dry run
        if not args.dry_run:
            interpolator.save_interpolation(results)
            console.print("\n[green]✓ Interpolation complete![/green]")
        else:
            console.print("\n[yellow]Dry run - no data saved[/yellow]")
    else:
        console.print("[yellow]No interpolation performed[/yellow]")


if __name__ == "__main__":
    main()