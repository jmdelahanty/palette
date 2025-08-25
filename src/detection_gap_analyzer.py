#!/usr/bin/env python3
"""
Advanced Gap Analysis for Detection Data

Provides detailed analysis of detection gaps including pattern detection,
temporal distribution, and cross-ID comparisons.

Usage:
    python detection_gap_analyzer.py detections.zarr
    python detection_gap_analyzer.py detections.zarr --max-gap 20 --pattern-analysis
    python detection_gap_analyzer.py detections.zarr --export-gaps gaps.csv
"""

import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import signal, stats

console = Console()

@dataclass
class GapPattern:
    """Describes a pattern in gap occurrences."""
    pattern_type: str  # 'periodic', 'random', 'clustered', 'increasing', 'decreasing'
    confidence: float  # 0-1 confidence in pattern detection
    period: Optional[float] = None  # For periodic patterns
    cluster_centers: Optional[List[int]] = None  # For clustered patterns
    trend_slope: Optional[float] = None  # For increasing/decreasing patterns
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GapAnalysis:
    """Comprehensive gap analysis for a single ROI."""
    roi_id: int
    total_gaps: int
    gap_segments: List[Tuple[int, int]]  # (start, end) for each gap
    gap_sizes: np.ndarray
    gap_positions: np.ndarray  # Frame indices where gaps start
    
    # Statistical measures
    mean_gap_size: float
    median_gap_size: float
    std_gap_size: float
    min_gap_size: int
    max_gap_size: int
    
    # Temporal distribution
    gap_density_over_time: np.ndarray  # Gap density in time windows
    inter_gap_distances: np.ndarray  # Distance between consecutive gaps
    
    # Pattern analysis
    detected_patterns: List[GapPattern]
    periodicity_score: float  # 0-1 score for how periodic gaps are
    clustering_score: float  # 0-1 score for how clustered gaps are
    
    # Gap categories
    single_frame_gaps: int
    short_gaps: int  # 2-5 frames
    medium_gaps: int  # 6-20 frames
    long_gaps: int  # >20 frames
    
    # Interpolation potential
    interpolatable_gaps: int  # Gaps that could be interpolated
    interpolation_coverage_gain: float  # % coverage gain if interpolated


class DetectionGapAnalyzer:
    """Advanced gap analysis for detection data."""
    
    def __init__(self, zarr_path: str, verbose: bool = True):
        """
        Initialize the gap analyzer.
        
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
        self.fps = self.root.attrs.get('fps', 60.0)
        
    def load_detection_data(self) -> Tuple[np.ndarray, Dict]:
        """Load detection data and metadata."""
        if 'id_assignments' not in self.root:
            raise ValueError("No ID assignments found. Run tracker.py with ID assignment first.")
        
        id_assign_group = self.root['id_assignments']
        latest_run = id_assign_group.attrs.get('latest', sorted(id_assign_group.group_keys())[-1])
        run_group = id_assign_group[latest_run]
        
        n_detections_per_roi = run_group['n_detections_per_roi'][:]
        
        metadata = {
            'source': f"id_assignments/{latest_run}",
            'total_frames': n_detections_per_roi.shape[0],
            'num_rois': n_detections_per_roi.shape[1],
            'fps': self.fps
        }
        
        return n_detections_per_roi, metadata
    
    def detect_patterns(self, gap_positions: np.ndarray, gap_sizes: np.ndarray) -> List[GapPattern]:
        """
        Detect patterns in gap occurrences.
        
        Args:
            gap_positions: Frame indices where gaps start
            gap_sizes: Size of each gap
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if len(gap_positions) < 3:
            return patterns
        
        # 1. Check for periodicity
        inter_gap_distances = np.diff(gap_positions)
        if len(inter_gap_distances) > 1:
            # Use autocorrelation to detect periodicity
            if len(inter_gap_distances) > 10:
                autocorr = np.correlate(inter_gap_distances, inter_gap_distances, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                # Find peaks in autocorrelation
                peaks, properties = signal.find_peaks(autocorr[1:], height=0.5)
                if len(peaks) > 0:
                    period = peaks[0] + 1
                    confidence = properties['peak_heights'][0]
                    patterns.append(GapPattern(
                        pattern_type='periodic',
                        confidence=float(confidence),
                        period=float(period),
                        details={'mean_interval': float(np.mean(inter_gap_distances))}
                    ))
        
        # 2. Check for clustering
        if len(gap_positions) > 5:
            # Use KDE to find density peaks
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(gap_positions)
                x_range = np.linspace(gap_positions.min(), gap_positions.max(), 100)
                density = kde(x_range)
                
                # Find peaks in density
                peaks, _ = signal.find_peaks(density, height=np.mean(density))
                if len(peaks) > 1:
                    cluster_centers = x_range[peaks].tolist()
                    clustering_score = (np.max(density) - np.mean(density)) / np.std(density) if np.std(density) > 0 else 0
                    patterns.append(GapPattern(
                        pattern_type='clustered',
                        confidence=min(1.0, clustering_score / 3),
                        cluster_centers=cluster_centers,
                        details={'num_clusters': len(peaks)}
                    ))
            except:
                pass  # KDE can fail with too few points
        
        # 3. Check for trends (increasing/decreasing gap frequency)
        if len(gap_positions) > 5:
            # Linear regression on gap positions
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.arange(len(gap_positions)), gap_positions
            )
            
            if p_value < 0.05:  # Significant trend
                trend_type = 'increasing' if slope > 0 else 'decreasing'
                patterns.append(GapPattern(
                    pattern_type=trend_type,
                    confidence=abs(r_value),
                    trend_slope=float(slope),
                    details={'p_value': float(p_value), 'r_squared': float(r_value**2)}
                ))
        
        # 4. Check for randomness
        if len(inter_gap_distances) > 5:
            # Use variance-to-mean ratio as a test for randomness
            vmr = np.var(inter_gap_distances) / np.mean(inter_gap_distances) if np.mean(inter_gap_distances) > 0 else 0
            if 0.5 < vmr < 1.5:  # Close to 1 suggests Poisson (random) process
                patterns.append(GapPattern(
                    pattern_type='random',
                    confidence=1.0 - abs(vmr - 1.0),
                    details={'variance_mean_ratio': float(vmr)}
                ))
        
        return patterns
    
    def analyze_roi_gaps(self, roi_id: int, n_detections_per_roi: np.ndarray, 
                         max_interpolatable_gap: int = 20) -> GapAnalysis:
        """
        Perform comprehensive gap analysis for a single ROI.
        
        Args:
            roi_id: The ROI ID to analyze
            n_detections_per_roi: Detection counts array
            max_interpolatable_gap: Maximum gap size considered interpolatable
            
        Returns:
            GapAnalysis object
        """
        roi_detections = n_detections_per_roi[:, roi_id]
        detection_mask = roi_detections > 0
        total_frames = len(roi_detections)
        
        # Find gaps
        gaps = []
        gap_sizes = []
        gap_positions = []
        
        # Find transitions
        diff = np.diff(np.concatenate(([False], detection_mask, [False])).astype(int))
        gap_starts = np.where(diff == -1)[0]
        gap_ends = np.where(diff == 1)[0]
        
        for start, end in zip(gap_starts, gap_ends):
            if end > start:
                gaps.append((start, end - 1))
                gap_sizes.append(end - start)
                gap_positions.append(start)
        
        gap_sizes = np.array(gap_sizes) if gap_sizes else np.array([])
        gap_positions = np.array(gap_positions) if gap_positions else np.array([])
        
        # Calculate temporal distribution
        window_size = total_frames // 20  # Divide timeline into 20 windows
        gap_density_over_time = np.zeros(20)
        if len(gap_positions) > 0:
            for pos in gap_positions:
                window_idx = min(int(pos / window_size), 19)
                gap_density_over_time[window_idx] += 1
        
        # Inter-gap distances
        inter_gap_distances = np.diff(gap_positions) if len(gap_positions) > 1 else np.array([])
        
        # Detect patterns
        patterns = self.detect_patterns(gap_positions, gap_sizes)
        
        # Calculate pattern scores
        periodicity_score = max([p.confidence for p in patterns if p.pattern_type == 'periodic'], default=0.0)
        clustering_score = max([p.confidence for p in patterns if p.pattern_type == 'clustered'], default=0.0)
        
        # Categorize gaps
        single_frame = np.sum(gap_sizes == 1) if len(gap_sizes) > 0 else 0
        short = np.sum((gap_sizes > 1) & (gap_sizes <= 5)) if len(gap_sizes) > 0 else 0
        medium = np.sum((gap_sizes > 5) & (gap_sizes <= 20)) if len(gap_sizes) > 0 else 0
        long = np.sum(gap_sizes > 20) if len(gap_sizes) > 0 else 0
        
        # Calculate interpolation potential
        interpolatable = np.sum(gap_sizes <= max_interpolatable_gap) if len(gap_sizes) > 0 else 0
        interpolatable_frames = np.sum(gap_sizes[gap_sizes <= max_interpolatable_gap]) if len(gap_sizes) > 0 else 0
        current_coverage = np.sum(detection_mask) / total_frames
        potential_coverage = (np.sum(detection_mask) + interpolatable_frames) / total_frames
        coverage_gain = (potential_coverage - current_coverage) * 100
        
        return GapAnalysis(
            roi_id=roi_id,
            total_gaps=len(gaps),
            gap_segments=gaps,
            gap_sizes=gap_sizes,
            gap_positions=gap_positions,
            mean_gap_size=float(np.mean(gap_sizes)) if len(gap_sizes) > 0 else 0,
            median_gap_size=float(np.median(gap_sizes)) if len(gap_sizes) > 0 else 0,
            std_gap_size=float(np.std(gap_sizes)) if len(gap_sizes) > 0 else 0,
            min_gap_size=int(np.min(gap_sizes)) if len(gap_sizes) > 0 else 0,
            max_gap_size=int(np.max(gap_sizes)) if len(gap_sizes) > 0 else 0,
            gap_density_over_time=gap_density_over_time,
            inter_gap_distances=inter_gap_distances,
            detected_patterns=patterns,
            periodicity_score=periodicity_score,
            clustering_score=clustering_score,
            single_frame_gaps=int(single_frame),
            short_gaps=int(short),
            medium_gaps=int(medium),
            long_gaps=int(long),
            interpolatable_gaps=int(interpolatable),
            interpolation_coverage_gain=coverage_gain
        )
    
    def create_gap_heatmap(self, analyses: Dict[int, GapAnalysis], save_path: Optional[Path] = None):
        """
        Create a heatmap showing gap distribution across ROIs and time.
        
        Args:
            analyses: Dictionary of GapAnalysis objects by ROI ID
            save_path: Optional path to save the figure
        """
        if not analyses:
            if self.verbose:
                self.console.print("[yellow]No gap data to visualize[/yellow]")
            return
        
        # Create gap density matrix
        roi_ids = sorted(analyses.keys())
        num_rois = len(roi_ids)
        time_windows = 20
        
        gap_matrix = np.zeros((num_rois, time_windows))
        for i, roi_id in enumerate(roi_ids):
            gap_matrix[i, :] = analyses[roi_id].gap_density_over_time
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Gap density heatmap
        ax = axes[0, 0]
        im = ax.imshow(gap_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_yticks(range(num_rois))
        ax.set_yticklabels([f'ROI {id}' for id in roi_ids])
        ax.set_xlabel('Time Window')
        ax.set_ylabel('ROI')
        ax.set_title('Gap Density Distribution Over Time')
        plt.colorbar(im, ax=ax, label='Number of Gaps')
        
        # 2. Gap size distribution
        ax = axes[0, 1]
        all_gap_sizes = []
        for roi_id, analysis in analyses.items():
            if len(analysis.gap_sizes) > 0:
                all_gap_sizes.extend([(roi_id, size) for size in analysis.gap_sizes])
        
        if all_gap_sizes:
            df = pd.DataFrame(all_gap_sizes, columns=['ROI', 'Gap Size'])
            sns.violinplot(data=df, x='ROI', y='Gap Size', ax=ax)
            ax.set_title('Gap Size Distribution by ROI')
            ax.set_ylabel('Gap Size (frames)')
        
        # 3. Pattern detection summary
        ax = axes[1, 0]
        pattern_data = []
        for roi_id, analysis in analyses.items():
            for pattern in analysis.detected_patterns:
                pattern_data.append({
                    'ROI': roi_id,
                    'Pattern': pattern.pattern_type,
                    'Confidence': pattern.confidence
                })
        
        if pattern_data:
            df = pd.DataFrame(pattern_data)
            pivot = df.pivot_table(values='Confidence', index='ROI', columns='Pattern', fill_value=0)
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, vmin=0, vmax=1)
            ax.set_title('Detected Gap Patterns (Confidence Scores)')
        else:
            ax.text(0.5, 0.5, 'No patterns detected', ha='center', va='center')
            ax.set_title('Gap Pattern Detection')
        
        # 4. Interpolation potential
        ax = axes[1, 1]
        roi_ids_plot = []
        coverage_gains = []
        colors = []
        
        for roi_id, analysis in sorted(analyses.items()):
            roi_ids_plot.append(f'ROI {roi_id}')
            coverage_gains.append(analysis.interpolation_coverage_gain)
            
            # Color based on gain potential
            if analysis.interpolation_coverage_gain > 5:
                colors.append('green')
            elif analysis.interpolation_coverage_gain > 2:
                colors.append('yellow')
            else:
                colors.append('red')
        
        bars = ax.bar(roi_ids_plot, coverage_gains, color=colors)
        ax.set_ylabel('Coverage Gain (%)')
        ax.set_title('Potential Coverage Improvement with Interpolation')
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='High gain (>5%)')
        ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Medium gain (2-5%)')
        ax.legend()
        
        # Add value labels on bars
        for bar, gain in zip(bars, coverage_gains):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{gain:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.suptitle(f'Gap Analysis Dashboard: {self.zarr_path.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                self.console.print(f"\n[green]Heatmap saved to:[/green] {save_path}")
        
        plt.show()
    
    def export_gaps_to_csv(self, analyses: Dict[int, GapAnalysis], output_path: Path):
        """
        Export gap information to CSV for manual review or further analysis.
        
        Args:
            analyses: Dictionary of GapAnalysis objects
            output_path: Path to save CSV file
        """
        gap_records = []
        
        for roi_id, analysis in analyses.items():
            for gap_start, gap_end in analysis.gap_segments:
                gap_size = gap_end - gap_start + 1
                gap_records.append({
                    'roi_id': roi_id,
                    'gap_start_frame': gap_start,
                    'gap_end_frame': gap_end,
                    'gap_size': gap_size,
                    'time_start_sec': gap_start / self.fps,
                    'time_end_sec': gap_end / self.fps,
                    'duration_sec': gap_size / self.fps,
                    'interpolatable': gap_size <= 20  # Default threshold
                })
        
        df = pd.DataFrame(gap_records)
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            self.console.print(f"\n[green]Gap data exported to:[/green] {output_path}")
            self.console.print(f"  Total gaps exported: {len(gap_records)}")
    
    def print_pattern_summary(self, analyses: Dict[int, GapAnalysis]):
        """Print a summary of detected patterns."""
        self.console.print("\n[bold cyan]GAP PATTERN ANALYSIS[/bold cyan]")
        self.console.print("="*60)
        
        for roi_id, analysis in sorted(analyses.items()):
            self.console.print(f"\n[bold]ROI {roi_id}:[/bold]")
            
            if not analysis.detected_patterns:
                self.console.print("  No clear patterns detected")
            else:
                for pattern in analysis.detected_patterns:
                    if pattern.pattern_type == 'periodic':
                        self.console.print(f"  • [yellow]Periodic[/yellow]: Period ≈ {pattern.period:.1f} frames "
                                         f"(confidence: {pattern.confidence:.2f})")
                    elif pattern.pattern_type == 'clustered':
                        self.console.print(f"  • [magenta]Clustered[/magenta]: {len(pattern.cluster_centers)} clusters "
                                         f"(confidence: {pattern.confidence:.2f})")
                    elif pattern.pattern_type == 'random':
                        self.console.print(f"  • [cyan]Random[/cyan]: Uniformly distributed "
                                         f"(confidence: {pattern.confidence:.2f})")
                    elif pattern.pattern_type in ['increasing', 'decreasing']:
                        self.console.print(f"  • [{'red' if pattern.pattern_type == 'increasing' else 'green'}]"
                                         f"{pattern.pattern_type.capitalize()}[/]: "
                                         f"Gap frequency {pattern.pattern_type} over time "
                                         f"(confidence: {pattern.confidence:.2f})")
            
            # Add interpolation recommendation
            if analysis.interpolation_coverage_gain > 5:
                self.console.print(f"  💡 [green]High interpolation potential: "
                                  f"+{analysis.interpolation_coverage_gain:.1f}% coverage[/green]")
            elif analysis.interpolation_coverage_gain > 2:
                self.console.print(f"  💡 [yellow]Moderate interpolation potential: "
                                  f"+{analysis.interpolation_coverage_gain:.1f}% coverage[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description='Advanced gap analysis for detection data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic gap analysis
  %(prog)s detections.zarr
  
  # Analyze specific ROIs with pattern detection
  %(prog)s detections.zarr --roi-ids 0 1 2 --pattern-analysis
  
  # Export gaps to CSV for manual review
  %(prog)s detections.zarr --export-gaps gaps.csv
  
  # Create comprehensive visualizations
  %(prog)s detections.zarr --visualize --save-viz gap_analysis.png
  
  # Set custom interpolation threshold
  %(prog)s detections.zarr --max-gap 30
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--roi-ids', nargs='+', type=int,
                       help='Specific ROI IDs to analyze (default: all)')
    
    parser.add_argument('--max-gap', type=int, default=20,
                       help='Maximum gap size considered interpolatable (default: 20)')
    
    parser.add_argument('--pattern-analysis', action='store_true',
                       help='Enable detailed pattern analysis')
    
    parser.add_argument('--export-gaps', type=str,
                       help='Export gap data to CSV file')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Create gap heatmap and visualizations')
    
    parser.add_argument('--save-viz', type=str,
                       help='Path to save visualization')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = DetectionGapAnalyzer(args.zarr_path, verbose=not args.quiet)
        
        # Load data
        n_detections_per_roi, metadata = analyzer.load_detection_data()
        
        if not args.quiet:
            console.print(f"\n[bold]Analyzing gaps in {metadata['num_rois']} ROIs "
                         f"across {metadata['total_frames']} frames...[/bold]")
        
        # Determine which ROIs to analyze
        if args.roi_ids:
            roi_ids = [rid for rid in args.roi_ids if 0 <= rid < metadata['num_rois']]
        else:
            roi_ids = list(range(metadata['num_rois']))
        
        # Perform analysis
        analyses = {}
        for roi_id in track(roi_ids, description="Analyzing ROIs", disable=args.quiet):
            analyses[roi_id] = analyzer.analyze_roi_gaps(
                roi_id, n_detections_per_roi, args.max_gap
            )
        
        # Print pattern summary if requested
        if args.pattern_analysis and not args.quiet:
            analyzer.print_pattern_summary(analyses)
        
        # Export gaps if requested
        if args.export_gaps:
            output_path = Path(args.export_gaps)
            analyzer.export_gaps_to_csv(analyses, output_path)
        
        # Create visualizations if requested
        if args.visualize:
            save_path = Path(args.save_viz) if args.save_viz else None
            analyzer.create_gap_heatmap(analyses, save_path)
        
        # Print summary statistics
        if not args.quiet:
            console.print("\n[bold cyan]SUMMARY STATISTICS[/bold cyan]")
            console.print("="*60)
            
            total_gaps = sum(a.total_gaps for a in analyses.values())
            avg_gap_size = np.mean([a.mean_gap_size for a in analyses.values() if a.total_gaps > 0])
            max_gap = max([a.max_gap_size for a in analyses.values()], default=0)
            
            console.print(f"Total gaps across all ROIs: {total_gaps}")
            console.print(f"Average gap size: {avg_gap_size:.1f} frames")
            console.print(f"Largest gap: {max_gap} frames")
            
            # Interpolation potential
            total_gain = sum(a.interpolation_coverage_gain for a in analyses.values()) / len(analyses)
            console.print(f"\nAverage coverage gain with interpolation: {total_gain:.1f}%")
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())