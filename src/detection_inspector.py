#!/usr/bin/env python3
"""
Detection Inspector for Zarr Tracking Data

Analyzes detection coverage and identifies gaps in tracked fish data from tracker.py.
Provides comprehensive inspection of detection IDs, coverage statistics, and gap analysis.

Usage:
    python detection_inspector.py detections.zarr
    python detection_inspector.py detections.zarr --roi-ids 0 1 2
    python detection_inspector.py detections.zarr --save-report
"""

import zarr
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

console = Console()

@dataclass
class DetectionStats:
    """Statistics for a single detection ID."""
    roi_id: int
    total_frames: int
    detected_frames: int
    missing_frames: int
    coverage_percentage: float
    gaps: List[Tuple[int, int]]  # List of (start, end) gap indices
    gap_sizes: List[int]
    max_gap_size: int
    mean_gap_size: float
    num_gaps: int
    longest_continuous_detection: int
    first_detection_frame: int
    last_detection_frame: int


@dataclass
class InspectionReport:
    """Complete inspection report for all ROIs."""
    zarr_path: str
    timestamp: str
    total_frames: int
    num_rois: int
    roi_stats: Dict[int, DetectionStats]
    overall_coverage: float
    detection_source: str  # Which run was used
    has_interpolation: bool
    has_filtering: bool
    metadata: Dict[str, Any]


class DetectionInspector:
    """Inspector for analyzing detection coverage and gaps in zarr tracking data."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        """
        Initialize the detection inspector.
        
        Args:
            zarr_path: Path to the zarr file
            verbose: Whether to print detailed output
        """
        self.zarr_path = Path(zarr_path)
        self.verbose = verbose
        self.console = console if verbose else None
        
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {self.zarr_path}")
        
        self.root = zarr.open_group(self.zarr_path, mode='r')
        self._detect_available_data()
        
    def _detect_available_data(self):
        """Detect what data is available in the zarr file."""
        # Check for both naming conventions
        self.has_id_assignments = 'id_assignments' in self.root or 'id_assignments_runs' in self.root
        self.id_assignments_key = 'id_assignments_runs' if 'id_assignments_runs' in self.root else 'id_assignments'
        
        self.has_preprocessing = 'preprocessing' in self.root
        self.has_filtered = 'filtered_runs' in self.root
        self.has_detect = 'detect_runs' in self.root
        
        if self.verbose:
            self.console.print("\n[bold cyan]Available Data:[/bold cyan]")
            self.console.print(f"  • Detect Runs: {'✓' if self.has_detect else '✗'}")
            self.console.print(f"  • ID Assignments: {'✓' if self.has_id_assignments else '✗'}")
            self.console.print(f"  • Preprocessing: {'✓' if self.has_preprocessing else '✗'}")
            self.console.print(f"  • Filtered Runs: {'✓' if self.has_filtered else '✗'}")
    
    def load_detection_data(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load detection data from the best available source.
        
        Returns:
            Tuple of (detection_ids, n_detections_per_roi, source_name)
        """
        if not self.has_id_assignments:
            raise ValueError("No ID assignments found. Run tracker.py with ID assignment first.")
        
        # Get the ID assignment group (handle both naming conventions)
        id_assign_group = self.root[self.id_assignments_key]
        
        # Check for 'latest' attribute or find the most recent run
        if 'latest' in id_assign_group.attrs:
            latest_run = id_assign_group.attrs['latest']
            run_group = id_assign_group[latest_run]
        else:
            # Find the most recent run by name
            runs = sorted(id_assign_group.group_keys())
            if not runs:
                raise ValueError("No ID assignment runs found")
            latest_run = runs[-1]
            run_group = id_assign_group[latest_run]
        
        detection_ids = run_group['detection_ids'][:]
        n_detections_per_roi = run_group['n_detections_per_roi'][:]
        
        if self.verbose:
            self.console.print(f"\n[green]Loaded data from:[/green] {self.id_assignments_key}/{latest_run}")
            self.console.print(f"  • Detection IDs shape: {detection_ids.shape}")
            self.console.print(f"  • Detections per ROI shape: {n_detections_per_roi.shape}")
            self.console.print(f"  • Number of ROIs: {n_detections_per_roi.shape[1]}")
            self.console.print(f"  • Total frames: {n_detections_per_roi.shape[0]}")
        
        return detection_ids, n_detections_per_roi, f"{self.id_assignments_key}/{latest_run}"
    
    def analyze_gaps(self, detection_mask: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Analyze ALL missing detections, including at the beginning and end.
        
        Args:
            detection_mask: Boolean array indicating detection presence
            
        Returns:
            Tuple of (gap_ranges, gap_sizes)
        """
        gaps = []
        gap_sizes = []
        
        # Find all continuous segments of missing detections
        in_gap = False
        gap_start = None
        
        for i, detected in enumerate(detection_mask):
            if not detected and not in_gap:
                # Start of a gap
                gap_start = i
                in_gap = True
            elif detected and in_gap:
                # End of a gap
                gaps.append((gap_start, i - 1))
                gap_sizes.append(i - gap_start)
                in_gap = False
        
        # Handle gap that extends to the end
        if in_gap:
            gaps.append((gap_start, len(detection_mask) - 1))
            gap_sizes.append(len(detection_mask) - gap_start)
        
        return gaps, gap_sizes
    
    def find_longest_continuous(self, detection_mask: np.ndarray) -> int:
        """Find the longest continuous detection sequence."""
        if not np.any(detection_mask):
            return 0
        
        # Find transitions
        diff = np.diff(np.concatenate(([False], detection_mask, [False])).astype(int))
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]
        
        if len(run_starts) == 0:
            return 0
        
        run_lengths = run_ends - run_starts
        return int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
    
    def calculate_roi_stats(self, roi_id: int, n_detections_per_roi: np.ndarray) -> DetectionStats:
        """
        Calculate statistics for a single ROI.
        
        Args:
            roi_id: The ROI ID to analyze
            n_detections_per_roi: Array of detection counts per ROI per frame
            
        Returns:
            DetectionStats object
        """
        roi_detections = n_detections_per_roi[:, roi_id]
        detection_mask = roi_detections > 0
        
        total_frames = len(roi_detections)
        detected_frames = np.sum(detection_mask)
        missing_frames = total_frames - detected_frames
        
        # Find gaps
        gaps, gap_sizes = self.analyze_gaps(detection_mask)
        
        # Find first and last detection
        detected_indices = np.where(detection_mask)[0]
        first_detection = int(detected_indices[0]) if len(detected_indices) > 0 else -1
        last_detection = int(detected_indices[-1]) if len(detected_indices) > 0 else -1
        
        # Calculate longest continuous detection
        longest_continuous = self.find_longest_continuous(detection_mask)
        
        return DetectionStats(
            roi_id=roi_id,
            total_frames=total_frames,
            detected_frames=detected_frames,
            missing_frames=missing_frames,
            coverage_percentage=(detected_frames / total_frames * 100) if total_frames > 0 else 0,
            gaps=gaps,
            gap_sizes=gap_sizes,
            max_gap_size=max(gap_sizes) if gap_sizes else 0,
            mean_gap_size=np.mean(gap_sizes) if gap_sizes else 0,
            num_gaps=len(gaps),
            longest_continuous_detection=longest_continuous,
            first_detection_frame=first_detection,
            last_detection_frame=last_detection
        )
    
    def generate_report(self, roi_ids: Optional[List[int]] = None) -> InspectionReport:
        """
        Generate a complete inspection report.
        
        Args:
            roi_ids: Optional list of specific ROI IDs to analyze. If None, analyze all.
            
        Returns:
            InspectionReport object
        """
        # Load detection data
        detection_ids, n_detections_per_roi, source = self.load_detection_data()
        
        total_frames, num_rois = n_detections_per_roi.shape
        
        # Determine which ROIs to analyze
        if roi_ids is None:
            roi_ids = list(range(num_rois))
        else:
            # Validate ROI IDs
            roi_ids = [rid for rid in roi_ids if 0 <= rid < num_rois]
        
        if self.verbose:
            self.console.print(f"\n[bold]Analyzing {len(roi_ids)} ROI(s)...[/bold]")
        
        # Calculate stats for each ROI
        roi_stats = {}
        for roi_id in track(roi_ids, description="Processing ROIs", disable=not self.verbose):
            roi_stats[roi_id] = self.calculate_roi_stats(roi_id, n_detections_per_roi)
        
        # Calculate overall coverage
        total_possible_detections = total_frames * len(roi_ids)
        total_actual_detections = sum(stats.detected_frames for stats in roi_stats.values())
        overall_coverage = (total_actual_detections / total_possible_detections * 100) if total_possible_detections > 0 else 0
        
        # Gather metadata
        metadata = dict(self.root.attrs)
        
        return InspectionReport(
            zarr_path=str(self.zarr_path),
            timestamp=datetime.now().isoformat(),
            total_frames=total_frames,
            num_rois=num_rois,
            roi_stats=roi_stats,
            overall_coverage=overall_coverage,
            detection_source=source,
            has_interpolation=self.has_preprocessing,
            has_filtering=self.has_filtered,
            metadata=metadata
        )
    
    def print_summary(self, report: InspectionReport):
        """Print a formatted summary of the inspection report."""
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]DETECTION INSPECTION SUMMARY[/bold cyan]")
        self.console.print("="*60)
        
        self.console.print(f"\n[bold]File:[/bold] {Path(report.zarr_path).name}")
        self.console.print(f"[bold]Total Frames:[/bold] {report.total_frames}")
        self.console.print(f"[bold]Number of ROIs:[/bold] {report.num_rois}")
        self.console.print(f"[bold]Overall Coverage:[/bold] {report.overall_coverage:.1f}%")
        
        # Create a table for ROI statistics
        table = Table(title="\nPer-ROI Detection Statistics")
        table.add_column("ROI ID", style="cyan", no_wrap=True)
        table.add_column("Coverage", style="green")
        table.add_column("Detected", style="yellow")
        table.add_column("Missing", style="red")
        table.add_column("Gaps", style="magenta")
        table.add_column("Max Gap", style="red")
        table.add_column("Longest Run", style="green")
        
        for roi_id, stats in sorted(report.roi_stats.items()):
            table.add_row(
                str(roi_id),
                f"{stats.coverage_percentage:.1f}%",
                f"{stats.detected_frames}/{stats.total_frames}",
                str(stats.missing_frames),
                str(stats.num_gaps),
                str(stats.max_gap_size),
                str(stats.longest_continuous_detection)
            )
        
        self.console.print(table)
        
        # Print gap analysis
        self.console.print("\n[bold]Gap Analysis:[/bold]")
        for roi_id, stats in sorted(report.roi_stats.items()):
            if stats.num_gaps > 0:
                self.console.print(f"\n  ROI {roi_id}:")
                self.console.print(f"    • Number of gaps: {stats.num_gaps}")
                self.console.print(f"    • Average gap size: {stats.mean_gap_size:.1f} frames")
                self.console.print(f"    • Maximum gap size: {stats.max_gap_size} frames")
                
                # Show gap distribution
                gap_distribution = {}
                for size in stats.gap_sizes:
                    if size <= 5:
                        key = "1-5 frames"
                    elif size <= 10:
                        key = "6-10 frames"
                    elif size <= 20:
                        key = "11-20 frames"
                    else:
                        key = ">20 frames"
                    gap_distribution[key] = gap_distribution.get(key, 0) + 1
                
                self.console.print("    • Gap distribution:")
                for key, count in sorted(gap_distribution.items()):
                    self.console.print(f"      - {key}: {count} gaps")
    
    def save_report(self, report: InspectionReport, output_path: Optional[Path] = None):
        """
        Save the inspection report to a JSON file.
        
        Args:
            report: The inspection report to save
            output_path: Optional output path. If None, saves next to zarr file.
        """
        if output_path is None:
            output_path = self.zarr_path.with_suffix('.inspection.json')
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        report_dict = convert_types(report_dict)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        if self.verbose:
            self.console.print(f"\n[green]Report saved to:[/green] {output_path}")
    
    def visualize_gaps(self, report: InspectionReport, save_path: Optional[Path] = None, 
                       roi_ids: Optional[List[int]] = None, separate_plots: bool = False):
        """
        Create visualizations of detection gaps.
        
        Args:
            report: The inspection report to visualize
            save_path: Optional path to save the figure
            roi_ids: Optional list of specific ROI IDs to visualize
            separate_plots: If True, create individual plots for each ROI
        """
        # Load the actual detection data to create proper visualization
        _, n_detections_per_roi, _ = self.load_detection_data()
        
        # Determine which ROIs to visualize
        if roi_ids is None:
            roi_ids = sorted(report.roi_stats.keys())
        else:
            roi_ids = [rid for rid in roi_ids if rid in report.roi_stats]
        
        if separate_plots:
            # Create individual plots for each ROI
            output_dir = Path(save_path).parent if save_path else Path.cwd() / "detection_plots"
            output_dir.mkdir(exist_ok=True)
            
            for roi_id in roi_ids:
                self._visualize_single_roi(roi_id, report.roi_stats[roi_id], 
                                          n_detections_per_roi[:, roi_id], 
                                          output_dir, report.zarr_path)
            
            if self.verbose:
                self.console.print(f"\n[green]Individual plots saved to:[/green] {output_dir}")
        
        # Also create combined plot if not too many ROIs
        if len(roi_ids) <= 6 or not separate_plots:
            self._visualize_all_rois(roi_ids, report, n_detections_per_roi, save_path)
    
    def _visualize_single_roi(self, roi_id: int, stats: DetectionStats, 
                              roi_detections: np.ndarray, output_dir: Path, zarr_path: str):
        """Create a detailed visualization for a single ROI."""
        detection_mask = roi_detections > 0
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
        
        # 1. Detection timeline heatmap
        ax1 = fig.add_subplot(gs[0, :])
        frame_chunks = 500  # Wider chunks for individual view
        num_rows = (stats.total_frames + frame_chunks - 1) // frame_chunks
        padded_size = num_rows * frame_chunks
        padded_array = np.ones(padded_size) * np.nan
        padded_array[:stats.total_frames] = detection_mask.astype(float)
        detection_matrix = padded_array.reshape(num_rows, frame_chunks)
        
        im = ax1.imshow(detection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_title(f'ROI {roi_id}: Detection Timeline (Coverage: {stats.coverage_percentage:.1f}%)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frame (within chunk)')
        ax1.set_ylabel('Frame chunk (×500)')
        plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02, label='Detection Present')
        
        # 2. Detection rate over time
        ax2 = fig.add_subplot(gs[1, :])
        window_size = 100  # Calculate detection rate in windows
        num_windows = stats.total_frames // window_size
        detection_rates = []
        window_centers = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            rate = np.mean(detection_mask[start:end]) * 100
            detection_rates.append(rate)
            window_centers.append(start + window_size // 2)
        
        ax2.plot(window_centers, detection_rates, linewidth=1.5, color='darkgreen')
        ax2.fill_between(window_centers, detection_rates, alpha=0.3, color='green')
        ax2.axhline(y=stats.coverage_percentage, color='red', linestyle='--', 
                   label=f'Overall: {stats.coverage_percentage:.1f}%', alpha=0.7)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Detection Rate (%)')
        ax2.set_title('Detection Rate Over Time (100-frame windows)')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Missing segment analysis
        ax3 = fig.add_subplot(gs[2, 0])
        missing_segments = []
        in_missing = False
        start = None
        
        for i, detected in enumerate(detection_mask):
            if not detected and not in_missing:
                start = i
                in_missing = True
            elif detected and in_missing:
                missing_segments.append((start, i-1))
                in_missing = False
        
        if in_missing:
            missing_segments.append((start, len(detection_mask)-1))
        
        if missing_segments:
            segment_sizes = [end - start + 1 for start, end in missing_segments]
            
            # Create histogram with better bins
            if max(segment_sizes) > 100:
                bins = np.logspace(0, np.log10(max(segment_sizes)), 30)
                ax3.set_xscale('log')
            else:
                bins = np.arange(0, max(segment_sizes) + 2, max(1, max(segment_sizes) // 20))
            
            n, bins, patches = ax3.hist(segment_sizes, bins=bins, edgecolor='black', alpha=0.7, color='red')
            
            # Color code by size
            for i, patch in enumerate(patches):
                if bins[i] <= 10:
                    patch.set_facecolor('yellow')
                elif bins[i] <= 50:
                    patch.set_facecolor('orange')
                else:
                    patch.set_facecolor('red')
            
            ax3.axvline(x=np.mean(segment_sizes), color='darkred', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(segment_sizes):.1f}')
            ax3.axvline(x=np.median(segment_sizes), color='darkblue', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(segment_sizes):.1f}')
            ax3.set_xlabel('Missing Segment Size (frames)')
            ax3.set_ylabel('Count')
            ax3.set_title(f'Missing Segment Distribution ({len(missing_segments)} segments)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '✓ Perfect Coverage!\nNo missing segments', 
                    ha='center', va='center', fontsize=14, color='green', weight='bold')
            ax3.set_title('Missing Segment Analysis')
            ax3.axis('off')
        
        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis('off')
        
        summary_text = f"""
        Detection Statistics for ROI {roi_id}
        {'='*40}
        
        Coverage:           {stats.coverage_percentage:.2f}%
        Total Frames:       {stats.total_frames:,}
        Detected Frames:    {stats.detected_frames:,}
        Missing Frames:     {stats.missing_frames:,}
        
        Gap Analysis:
        Number of Gaps:     {stats.num_gaps}
        Mean Gap Size:      {stats.mean_gap_size:.1f} frames
        Max Gap Size:       {stats.max_gap_size} frames
        
        Detection Span:
        First Detection:    Frame {stats.first_detection_frame}
        Last Detection:     Frame {stats.last_detection_frame}
        Longest Run:        {stats.longest_continuous_detection} frames
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, pad=1))
        
        # Add recommendation
        if stats.missing_frames > 0:
            if stats.mean_gap_size <= 20:
                recommendation = "✓ Good candidate for interpolation"
                color = 'green'
            elif stats.mean_gap_size <= 50:
                recommendation = "⚠ Moderate gaps - review before interpolation"
                color = 'orange'
            else:
                recommendation = "✗ Large gaps - consider re-tracking"
                color = 'red'
            
            ax4.text(0.5, 0.1, recommendation, transform=ax4.transAxes,
                    fontsize=12, color=color, weight='bold', 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
        
        plt.suptitle(f'Detection Analysis: {Path(zarr_path).name} - ROI {roi_id}', 
                    fontsize=16, fontweight='bold')
        
        # Save the figure
        output_path = output_dir / f"roi_{roi_id:02d}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_all_rois(self, roi_ids: List[int], report: InspectionReport, 
                           n_detections_per_roi: np.ndarray, save_path: Optional[Path]):
        """Create the combined visualization for all ROIs."""
        num_rois = len(roi_ids)
        fig, axes = plt.subplots(num_rois, 2, figsize=(15, 3 * num_rois))
        
        if num_rois == 1:
            axes = axes.reshape(1, -1)
        
        for idx, roi_id in enumerate(roi_ids):
            stats = report.roi_stats[roi_id]
            roi_detections = n_detections_per_roi[:, roi_id]
            detection_mask = roi_detections > 0
            
            # Left plot: Detection timeline
            ax = axes[idx, 0]
            frame_chunks = 100
            num_rows = (stats.total_frames + frame_chunks - 1) // frame_chunks
            padded_size = num_rows * frame_chunks
            padded_array = np.ones(padded_size) * np.nan
            padded_array[:stats.total_frames] = detection_mask.astype(float)
            detection_matrix = padded_array.reshape(num_rows, frame_chunks)
            
            im = ax.imshow(detection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'ROI {roi_id}: Detection Timeline (Coverage: {stats.coverage_percentage:.1f}%)')
            ax.set_xlabel('Frame (within chunk)')
            ax.set_ylabel('Frame chunk (×100)')
            
            # Right plot: Gap analysis
            ax = axes[idx, 1]
            if stats.missing_frames > 0:
                missing_segments = []
                in_missing = False
                start = None
                
                for i, detected in enumerate(detection_mask):
                    if not detected and not in_missing:
                        start = i
                        in_missing = True
                    elif detected and in_missing:
                        missing_segments.append((start, i-1))
                        in_missing = False
                
                if in_missing:
                    missing_segments.append((start, len(detection_mask)-1))
                
                if missing_segments:
                    segment_sizes = [end - start + 1 for start, end in missing_segments]
                    max_size = max(segment_sizes)
                    
                    if max_size > 100:
                        bins = np.logspace(0, np.log10(max_size), 20)
                        ax.set_xscale('log')
                    else:
                        bins = np.arange(0, max_size + 2, max(1, max_size // 20))
                    
                    ax.hist(segment_sizes, bins=bins, edgecolor='black', alpha=0.7, color='red')
                    ax.axvline(x=np.mean(segment_sizes), color='darkred', linestyle='--', 
                              label=f'Mean: {np.mean(segment_sizes):.1f}')
                    ax.set_xlabel('Missing Segment Size (frames)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'ROI {roi_id}: {len(missing_segments)} segments')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '✓ Perfect Coverage!', 
                       ha='center', va='center', fontsize=12, color='green', weight='bold')
                ax.set_title(f'ROI {roi_id}: Gap Analysis')
                ax.axis('off')
        
        plt.suptitle(f'Detection Coverage Analysis: {Path(report.zarr_path).name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                self.console.print(f"\n[green]Combined visualization saved to:[/green] {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect detection coverage and gaps in zarr tracking data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection of all ROIs
  %(prog)s detections.zarr
  
  # Inspect specific ROIs
  %(prog)s detections.zarr --roi-ids 0 1 2
  
  # Save report and visualizations
  %(prog)s detections.zarr --save-report --visualize
  
  # Create individual plots for each ROI
  %(prog)s detections.zarr --visualize --separate-plots
  
  # Focus on problematic ROIs with individual plots
  %(prog)s detections.zarr --roi-ids 3 8 --visualize --separate-plots
  
  # Quiet mode (no console output)
  %(prog)s detections.zarr --quiet --save-report
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--roi-ids', nargs='+', type=int,
                       help='Specific ROI IDs to analyze (default: all)')
    
    parser.add_argument('--save-report', action='store_true',
                       help='Save inspection report to JSON file')
    
    parser.add_argument('--output', type=str,
                       help='Output path for report (default: <zarr_name>.inspection.json)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Create gap visualization plots')
    
    parser.add_argument('--separate-plots', action='store_true',
                       help='Create individual plots for each ROI (saves to detection_plots/)')
    
    parser.add_argument('--save-viz', type=str,
                       help='Path to save visualization (default: show only)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    try:
        # Initialize inspector
        inspector = DetectionInspector(args.zarr_path, verbose=not args.quiet)
        
        # Generate report
        report = inspector.generate_report(roi_ids=args.roi_ids)
        
        # Print summary
        if not args.quiet:
            inspector.print_summary(report)
        
        # Save report if requested
        if args.save_report:
            output_path = Path(args.output) if args.output else None
            inspector.save_report(report, output_path)
        
        # Create visualizations if requested
        if args.visualize:
            save_path = Path(args.save_viz) if args.save_viz else None
            inspector.visualize_gaps(report, save_path, 
                                   roi_ids=args.roi_ids,
                                   separate_plots=args.separate_plots)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())