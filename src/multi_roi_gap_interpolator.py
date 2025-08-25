#!/usr/bin/env python3
"""
Multi-ROI Gap Interpolator for ID-Assigned Detection Data

Fills gaps in detection data for multiple ROIs using intelligent interpolation.
Works with the output from tracker.py's ID assignment stage.

This tool:
- Handles multiple ROIs independently
- Preserves detection IDs during interpolation
- Creates proper interpolation masks per ROI
- Maintains compatibility with existing zarr structure
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import interpolate
import json
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()


def find_roi_gaps(roi_detections: np.ndarray, max_gap: Optional[int] = None) -> List[Dict]:
    """
    Find gaps in detection data for a single ROI.
    
    Args:
        roi_detections: Array of detection counts for one ROI
        max_gap: Maximum gap size to consider for interpolation
    
    Returns:
        List of gap dictionaries
    """
    gaps = []
    in_gap = False
    gap_start = None
    
    for frame_idx in range(len(roi_detections)):
        has_detection = roi_detections[frame_idx] > 0
        
        if not has_detection and not in_gap:
            in_gap = True
            gap_start = frame_idx
        elif has_detection and in_gap:
            gap_size = frame_idx - gap_start
            if max_gap is None or gap_size <= max_gap:
                gaps.append({
                    'start': gap_start,
                    'end': frame_idx - 1,
                    'size': gap_size,
                    'before_frame': gap_start - 1 if gap_start > 0 else None,
                    'after_frame': frame_idx
                })
            in_gap = False
    
    # Handle gap at the end
    if in_gap:
        gap_size = len(roi_detections) - gap_start
        if max_gap is None or gap_size <= max_gap:
            gaps.append({
                'start': gap_start,
                'end': len(roi_detections) - 1,
                'size': gap_size,
                'before_frame': gap_start - 1 if gap_start > 0 else None,
                'after_frame': None
            })
    
    return gaps


def interpolate_roi_gap(detection_positions: Dict[int, np.ndarray], 
                        gap: Dict, roi_id: int, 
                        method: str = 'linear',
                        confidence_decay: float = 0.95) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Interpolate detections for a gap in a single ROI.
    
    Args:
        detection_positions: Dict mapping frame_idx to [x, y] positions for this ROI
        gap: Gap dictionary
        roi_id: ROI identifier
        method: Interpolation method
        confidence_decay: Confidence decay factor
        
    Returns:
        Tuple of (interpolated_positions, interpolated_confidences)
    """
    if gap['before_frame'] is None or gap['after_frame'] is None:
        return {}, np.array([])
    
    # Get positions before and after gap
    pos_before = detection_positions[gap['before_frame']]
    pos_after = detection_positions[gap['after_frame']]
    
    gap_frames = range(gap['start'], gap['end'] + 1)
    n_frames = len(gap_frames)
    
    interpolated_positions = {}
    interpolated_confidences = np.zeros(n_frames)
    
    if method == 'linear':
        # Linear interpolation for x and y
        for i, frame_idx in enumerate(gap_frames):
            t = (i + 1) / (n_frames + 1)  # Normalized position
            interp_pos = pos_before * (1 - t) + pos_after * t
            interpolated_positions[frame_idx] = interp_pos
            
            # Decay confidence based on distance from nearest real detection
            distance_to_nearest = min(i + 1, n_frames - i)
            interpolated_confidences[i] = confidence_decay ** distance_to_nearest
            
    elif method == 'cubic' and n_frames >= 2:
        # Need at least 4 points for cubic, so get more context if available
        context_frames = []
        context_positions = []
        
        # Get 2 frames before if possible
        for f in range(max(0, gap['before_frame'] - 1), gap['before_frame'] + 1):
            if f in detection_positions:
                context_frames.append(f)
                context_positions.append(detection_positions[f])
        
        # Get 2 frames after if possible
        for f in range(gap['after_frame'], min(gap['after_frame'] + 2, max(detection_positions.keys()) + 1)):
            if f in detection_positions:
                context_frames.append(f)
                context_positions.append(detection_positions[f])
        
        if len(context_frames) >= 4:
            context_positions = np.array(context_positions)
            # Interpolate x and y separately
            interp_x = interpolate.interp1d(context_frames, context_positions[:, 0], 
                                           kind='cubic', fill_value='extrapolate')
            interp_y = interpolate.interp1d(context_frames, context_positions[:, 1], 
                                           kind='cubic', fill_value='extrapolate')
            
            for i, frame_idx in enumerate(gap_frames):
                interpolated_positions[frame_idx] = np.array([interp_x(frame_idx), interp_y(frame_idx)])
                distance_to_nearest = min(i + 1, n_frames - i)
                interpolated_confidences[i] = confidence_decay ** distance_to_nearest
        else:
            # Fall back to linear if not enough points
            return interpolate_roi_gap(detection_positions, gap, roi_id, 'linear', confidence_decay)
    
    return interpolated_positions, interpolated_confidences


class MultiROIGapInterpolator:
    """Interpolator for multi-ROI detection data from tracker.py."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        """
        Initialize the interpolator.
        
        Args:
            zarr_path: Path to zarr file
            verbose: Whether to print progress
        """
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.console = console if verbose else None
        
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")
        
        # Open in read-write mode
        self.root = zarr.open_group(self.zarr_path, mode='r+')
        self.fps = self.root.attrs.get('fps', 60.0)
        
    def load_detection_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Load detection data from zarr.
        
        Returns:
            Tuple of (detection_ids, n_detections_per_roi, bbox_coords, source_name)
        """
        # Handle both naming conventions
        id_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        
        if id_key not in self.root:
            raise ValueError("No ID assignments found. Run tracker.py with ID assignment first.")
        
        id_group = self.root[id_key]
        
        # Get latest run
        if 'latest' in id_group.attrs:
            latest_run = id_group.attrs['latest']
        else:
            runs = sorted(id_group.group_keys())
            if not runs:
                raise ValueError("No ID assignment runs found")
            latest_run = runs[-1]
        
        run_group = id_group[latest_run]
        
        # Load detection data
        detection_ids = run_group['detection_ids'][:]
        n_detections_per_roi = run_group['n_detections_per_roi'][:]
        
        # Get bbox coordinates from detect_runs
        detect_key = 'detect_runs'
        if detect_key in self.root:
            detect_group = self.root[detect_key]
            if 'latest' in detect_group.attrs:
                latest_detect = detect_group.attrs['latest']
            else:
                detects = sorted(detect_group.group_keys())
                latest_detect = detects[-1] if detects else None
            
            if latest_detect:
                bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
            else:
                raise ValueError("No detection data found")
        else:
            raise ValueError("No detect_runs found")
        
        source = f"{id_key}/{latest_run}"
        
        if self.verbose:
            self.console.print(f"\n[green]Loaded data from:[/green] {source}")
            self.console.print(f"  • Total frames: {n_detections_per_roi.shape[0]}")
            self.console.print(f"  • Number of ROIs: {n_detections_per_roi.shape[1]}")
            self.console.print(f"  • Total detections: {len(detection_ids)}")
        
        return detection_ids, n_detections_per_roi, bbox_coords, source
    
    def extract_roi_positions(self, roi_id: int, detection_ids: np.ndarray, 
                             bbox_coords: np.ndarray, 
                             n_detections_per_roi: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract position data for a specific ROI.
        
        Returns:
            Dict mapping frame_idx to [x, y] centroid position
        """
        positions = {}
        cumulative_idx = 0
        
        # Get n_detections from detect_runs to properly index
        detect_group = self.root['detect_runs']
        if 'latest' in detect_group.attrs:
            latest_detect = detect_group.attrs['latest']
        else:
            detects = sorted(detect_group.group_keys())
            latest_detect = detects[-1] if detects else None
        
        n_detections = detect_group[latest_detect]['n_detections'][:]
        
        for frame_idx in range(len(n_detections_per_roi)):
            frame_detection_count = int(n_detections[frame_idx])
            
            if frame_detection_count > 0 and n_detections_per_roi[frame_idx, roi_id] > 0:
                # Find detection for this ROI in this frame
                if cumulative_idx + frame_detection_count <= len(detection_ids):
                    frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_detection_count]
                    roi_mask = frame_detection_ids == roi_id
                    
                    if np.any(roi_mask):
                        roi_idx = np.where(roi_mask)[0][0]
                        if cumulative_idx + roi_idx < len(bbox_coords):
                            bbox = bbox_coords[cumulative_idx + roi_idx]
                            # Calculate centroid (bbox is [center_x, center_y, width, height])
                            positions[frame_idx] = bbox[:2]  # Just take center coordinates
            
            cumulative_idx += frame_detection_count
        
        return positions
    
    def interpolate_all_rois(self, max_gap: int = 20, method: str = 'linear',
                             confidence_decay: float = 0.95,
                             roi_ids: Optional[List[int]] = None) -> Dict:
        """
        Interpolate gaps for all or specified ROIs.
        
        Args:
            max_gap: Maximum gap size to interpolate
            method: Interpolation method ('linear' or 'cubic')
            confidence_decay: Confidence decay factor
            roi_ids: Optional list of specific ROI IDs to process
            
        Returns:
            Dictionary with interpolation results
        """
        # Load data
        detection_ids, n_detections_per_roi, bbox_coords, source = self.load_detection_data()
        
        total_frames, num_rois = n_detections_per_roi.shape
        
        # Determine which ROIs to process
        if roi_ids is None:
            roi_ids = list(range(num_rois))
        
        # Initialize results
        results = {
            'roi_results': {},
            'total_gaps_filled': 0,
            'total_frames_added': 0,
            'interpolation_masks': {}
        }
        
        # Process each ROI
        for roi_id in track(roi_ids, description="Processing ROIs", disable=not self.verbose):
            try:
                roi_detections = n_detections_per_roi[:, roi_id]
                
                # Find gaps
                all_gaps = find_roi_gaps(roi_detections, max_gap=None)
                fillable_gaps = find_roi_gaps(roi_detections, max_gap=max_gap)
                
                # Get existing positions
                positions = self.extract_roi_positions(roi_id, detection_ids, 
                                                      bbox_coords, n_detections_per_roi)
                
                # Initialize interpolation mask
                interpolation_mask = np.zeros(total_frames, dtype=bool)
                frames_added = 0
                
                # Fill gaps
                new_positions = positions.copy()
                new_confidences = {}
                
                for gap in fillable_gaps:
                    interp_positions, interp_confidences = interpolate_roi_gap(
                        positions, gap, roi_id, method, confidence_decay
                    )
                    
                    for frame_idx, pos in interp_positions.items():
                        new_positions[frame_idx] = pos
                        new_confidences[frame_idx] = interp_confidences[frame_idx - gap['start']]
                        interpolation_mask[frame_idx] = True
                        frames_added += 1
                
                # Calculate coverage
                coverage_before = len(positions) / total_frames * 100 if total_frames > 0 else 0
                coverage_after = len(new_positions) / total_frames * 100 if total_frames > 0 else 0
                
                results['roi_results'][roi_id] = {
                    'gaps_found': len(all_gaps),
                    'gaps_filled': len(fillable_gaps),
                    'frames_added': frames_added,
                    'coverage_before': coverage_before,
                    'coverage_after': coverage_after,
                    'coverage_gain': coverage_after - coverage_before,
                    'interpolated_positions': new_positions,
                    'interpolated_confidences': new_confidences,
                    'original_positions': positions
                }
                
                results['interpolation_masks'][roi_id] = interpolation_mask
                results['total_gaps_filled'] += len(fillable_gaps)
                results['total_frames_added'] += frames_added
                
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]Warning: Error processing ROI {roi_id}: {e}[/yellow]")
                # Continue with next ROI
                results['roi_results'][roi_id] = {
                    'gaps_found': 0,
                    'gaps_filled': 0,
                    'frames_added': 0,
                    'coverage_before': 0,
                    'coverage_after': 0,
                    'coverage_gain': 0,
                    'interpolated_positions': {},
                    'interpolated_confidences': {},
                    'original_positions': {},
                    'error': str(e)
                }
                results['interpolation_masks'][roi_id] = np.zeros(total_frames, dtype=bool)
        
        return results
    
    def save_interpolated_data(self, results: Dict, max_gap: int, 
                               method: str, confidence_decay: float):
        """Save interpolated data back to zarr with proper metadata."""
        
        if self.verbose:
            self.console.print("\n[bold cyan]SAVING INTERPOLATED DATA[/bold cyan]")
        
        # Create interpolation group if needed
        if 'interpolation_runs' not in self.root:
            interp_group = self.root.create_group('interpolation_runs')
            interp_group.attrs['created_at'] = datetime.now().isoformat()
        else:
            interp_group = self.root['interpolation_runs']
        
        # Generate run name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'interpolation_{timestamp}'
        
        # Create run group
        run_group = interp_group.create_group(run_name)
        run_group.attrs.update({
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'max_gap': max_gap,
                'method': method,
                'confidence_decay': confidence_decay
            },
            'total_gaps_filled': results['total_gaps_filled'],
            'total_frames_added': results['total_frames_added']
        })
        
        # Determine total number of ROIs from the data
        _, n_detections_per_roi, _, _ = self.load_detection_data()
        total_rois = n_detections_per_roi.shape[1]
        total_frames = n_detections_per_roi.shape[0]
        
        # Create full-size masks array
        masks_array = np.zeros((total_rois, total_frames), dtype=bool)
        
        # Fill in the masks for processed ROIs
        for roi_id, mask in results['interpolation_masks'].items():
            masks_array[roi_id] = mask
        
        run_group.create_dataset('interpolation_masks', data=masks_array, dtype=bool)
        
        # Save per-ROI statistics
        roi_stats = {}
        for roi_id, roi_result in results['roi_results'].items():
            roi_stats[str(roi_id)] = {
                'gaps_filled': roi_result['gaps_filled'],
                'frames_added': roi_result['frames_added'],
                'coverage_before': roi_result['coverage_before'],
                'coverage_after': roi_result['coverage_after']
            }
        
        run_group.attrs['roi_statistics'] = json.dumps(roi_stats)
        run_group.attrs['processed_rois'] = list(results['roi_results'].keys())
        
        # Update latest pointer
        interp_group.attrs['latest'] = run_name
        
        if self.verbose:
            self.console.print(f"[green]✓ Saved as:[/green] interpolation_runs/{run_name}")
            self.console.print(f"[green]✓ Total frames interpolated:[/green] {results['total_frames_added']}")
            self.console.print(f"[green]✓ ROIs processed:[/green] {list(results['roi_results'].keys())}")
    
    def visualize_results(self, results: Dict, save_path: Optional[Path] = None):
        """Create visualization of interpolation results."""
        
        num_rois = len(results['roi_results'])
        fig, axes = plt.subplots(num_rois, 2, figsize=(15, 3 * num_rois))
        
        if num_rois == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (roi_id, roi_result) in enumerate(sorted(results['roi_results'].items())):
            # Left plot: Before/after trajectory
            ax = axes[idx, 0]
            
            # Original positions
            if roi_result['original_positions']:
                orig_frames = sorted(roi_result['original_positions'].keys())
                orig_positions = np.array([roi_result['original_positions'][f] for f in orig_frames])
                ax.scatter(orig_positions[:, 0], orig_positions[:, 1], 
                          c=orig_frames, cmap='viridis', s=2, alpha=0.6, label='Original')
            
            # Interpolated positions (only the new ones)
            new_frames = [f for f in roi_result['interpolated_positions'] 
                         if f not in roi_result['original_positions']]
            if new_frames:
                new_positions = np.array([roi_result['interpolated_positions'][f] for f in new_frames])
                ax.scatter(new_positions[:, 0], new_positions[:, 1],
                          c='red', s=4, alpha=0.5, label='Interpolated')
            
            ax.set_title(f'ROI {roi_id}: Trajectory (Added {roi_result["frames_added"]} frames)')
            ax.set_xlabel('X (normalized)')
            ax.set_ylabel('Y (normalized)')
            ax.legend()
            ax.set_aspect('equal')
            
            # Right plot: Coverage improvement
            ax = axes[idx, 1]
            categories = ['Before', 'After']
            coverages = [roi_result['coverage_before'], roi_result['coverage_after']]
            colors = ['blue', 'green']
            
            bars = ax.bar(categories, coverages, color=colors)
            ax.set_ylabel('Coverage (%)')
            ax.set_title(f'ROI {roi_id}: Coverage Improvement (+{roi_result["coverage_gain"]:.1f}%)')
            ax.set_ylim(0, 105)
            
            # Add value labels on bars
            for bar, coverage in zip(bars, coverages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{coverage:.1f}%', ha='center', va='bottom')
            
            # Add gap information
            gap_text = f"Gaps filled: {roi_result['gaps_filled']}/{roi_result['gaps_found']}"
            ax.text(0.5, 0.5, gap_text, transform=ax.transAxes,
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.suptitle(f'Multi-ROI Gap Interpolation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                self.console.print(f"\n[green]Visualization saved to:[/green] {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print a formatted summary of interpolation results."""
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]INTERPOLATION SUMMARY[/bold cyan]")
        self.console.print("="*60)
        
        # Create summary table
        table = Table(title="Per-ROI Interpolation Results")
        table.add_column("ROI ID", style="cyan", no_wrap=True)
        table.add_column("Gaps Found", style="yellow")
        table.add_column("Gaps Filled", style="green")
        table.add_column("Frames Added", style="green")
        table.add_column("Coverage Before", style="red")
        table.add_column("Coverage After", style="green")
        table.add_column("Gain", style="magenta")
        
        for roi_id, roi_result in sorted(results['roi_results'].items()):
            table.add_row(
                str(roi_id),
                str(roi_result['gaps_found']),
                str(roi_result['gaps_filled']),
                str(roi_result['frames_added']),
                f"{roi_result['coverage_before']:.1f}%",
                f"{roi_result['coverage_after']:.1f}%",
                f"+{roi_result['coverage_gain']:.1f}%"
            )
        
        self.console.print(table)
        
        # Overall statistics
        self.console.print(f"\n[bold]Overall Statistics:[/bold]")
        self.console.print(f"  Total gaps filled: {results['total_gaps_filled']}")
        self.console.print(f"  Total frames added: {results['total_frames_added']}")
        
        avg_gain = np.mean([r['coverage_gain'] for r in results['roi_results'].values()])
        self.console.print(f"  Average coverage gain: {avg_gain:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Fill gaps in multi-ROI detection data using interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview gap filling for all ROIs
  %(prog)s detections.zarr --visualize
  
  # Fill small gaps only
  %(prog)s detections.zarr --max-gap 5 --visualize
  
  # Process specific ROIs
  %(prog)s detections.zarr --roi-ids 3 8 --max-gap 20 --visualize
  
  # Use cubic interpolation for smoother trajectories
  %(prog)s detections.zarr --method cubic --visualize
  
  # Save after reviewing
  %(prog)s detections.zarr --max-gap 10 --save --visualize
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--max-gap', type=int, default=20,
                       help='Maximum gap size to interpolate (default: 20 frames)')
    
    parser.add_argument('--method', choices=['linear', 'cubic'], default='linear',
                       help='Interpolation method (default: linear)')
    
    parser.add_argument('--confidence-decay', type=float, default=0.95,
                       help='Confidence decay per frame (default: 0.95)')
    
    parser.add_argument('--roi-ids', nargs='+', type=int,
                       help='Specific ROI IDs to process (default: all)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization of results')
    
    parser.add_argument('--save', action='store_true',
                       help='Save interpolated data to zarr')
    
    parser.add_argument('--save-viz', type=str,
                       help='Path to save visualization')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    try:
        # Initialize interpolator
        interpolator = MultiROIGapInterpolator(args.zarr_path, verbose=not args.quiet)
        
        # Run interpolation
        results = interpolator.interpolate_all_rois(
            max_gap=args.max_gap,
            method=args.method,
            confidence_decay=args.confidence_decay,
            roi_ids=args.roi_ids
        )
        
        # Print summary
        if not args.quiet:
            interpolator.print_summary(results)
        
        # Visualize if requested
        if args.visualize:
            save_path = Path(args.save_viz) if args.save_viz else None
            interpolator.visualize_results(results, save_path)
        
        # Save if requested
        if args.save:
            interpolator.save_interpolated_data(results, args.max_gap, 
                                               args.method, args.confidence_decay)
            if not args.quiet:
                console.print("\n[green]✓ Interpolation data saved successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())