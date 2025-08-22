#!/usr/bin/env python3
"""
Trial Analyzer with Zarr Detection Data

Combines trial information from H5 files with cleaned detection data from zarr files.
This allows you to analyze chase trials using the preprocessed detection data
from your frame_distance_analyzer and gap_interpolator pipeline.
"""

import h5py
import zarr
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

# Event type mappings
EXPERIMENT_EVENT_TYPE = {
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END"
}

class TrialZarrAnalyzer:
    """Analyze trials using H5 metadata and zarr detection data."""
    
    def __init__(self, h5_path: str, zarr_path: str, use_interpolated: bool = True):
        self.h5_path = Path(h5_path)
        self.zarr_path = Path(zarr_path)
        self.use_interpolated = use_interpolated
        
        # Load zarr data
        self.load_zarr_data()
        
    def load_zarr_data(self):
        """Load detection data from zarr file."""
        print(f"\n📂 Loading zarr data from: {self.zarr_path}")
        
        root = zarr.open(str(self.zarr_path), mode='r')
        
        # Determine which data to use
        if self.use_interpolated:
            # Try to use preprocessed data
            if 'preprocessing' in root and 'latest' in root['preprocessing'].attrs:
                source_path = root['preprocessing'].attrs['latest']
                source_group = root['preprocessing'][source_path]
                print(f"  Using interpolated data: {source_path}")
                
                # Check if this has interpolation mask
                if 'interpolation_mask' in source_group:
                    self.interpolation_mask = source_group['interpolation_mask'][:]
                else:
                    self.interpolation_mask = None
                    
            elif 'filtered_runs' in root and 'latest' in root['filtered_runs'].attrs:
                source_name = root['filtered_runs'].attrs['latest']
                source_group = root['filtered_runs'][source_name]
                print(f"  Using filtered data: {source_name}")
                self.interpolation_mask = None
                
            else:
                source_group = root
                print("  Using original data")
                self.interpolation_mask = None
        else:
            source_group = root
            print("  Using original data (as requested)")
            self.interpolation_mask = None
        
        # Load the arrays
        self.bboxes = source_group['bboxes'][:]
        self.scores = source_group['scores'][:]
        self.n_detections = source_group['n_detections'][:]
        
        # Get metadata
        self.fps = root.attrs.get('fps', 60.0)
        self.img_width = root.attrs.get('width', 4512)
        self.img_height = root.attrs.get('height', 4512)
        self.total_frames = len(self.n_detections)
        
        # Calculate coverage
        frames_with_detections = np.sum(self.n_detections > 0)
        self.coverage = frames_with_detections / self.total_frames * 100
        
        print(f"  Loaded {self.total_frames} frames")
        print(f"  Coverage: {self.coverage:.1f}% ({frames_with_detections}/{self.total_frames})")
        
        if self.interpolation_mask is not None:
            interpolated_count = np.sum(self.interpolation_mask)
            print(f"  Interpolated frames: {interpolated_count}")
    
    def calculate_centroid(self, bbox):
        """Calculate the centroid of a bounding box."""
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    
    def analyze_trials(self, visualize: bool = False):
        """Analyze chase trials using zarr detection data."""
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get events
            if '/events' not in f:
                print("❌ No events dataset found in H5 file")
                return
            
            events = f['/events'][:]
            
            # Find chase sequences
            chase_starts = events[events['event_type_id'] == 27]
            chase_ends = events[events['event_type_id'] == 28]
            
            print(f"\n🎯 Analyzing {len(chase_starts)} chase trials")
            print("=" * 70)
            
            trial_results = []
            
            for i, (start_event, end_event) in enumerate(zip(chase_starts, chase_ends)):
                print(f"\n📊 Trial {i+1}:")
                print("-" * 40)
                
                # Get frame range from events
                if 'camera_frame_id' in start_event.dtype.names:
                    start_frame = start_event['camera_frame_id']
                    end_frame = end_event['camera_frame_id']
                    
                    if start_frame <= 0 or end_frame <= 0:
                        print("  ⚠️  No valid camera frame IDs in events")
                        continue
                    
                    # Adjust frame indices for zarr (0-based indexing)
                    # Assuming camera_frame_id starts at 1
                    start_idx = start_frame - 1
                    end_idx = end_frame - 1
                    
                    # Ensure indices are within bounds
                    start_idx = max(0, min(start_idx, self.total_frames - 1))
                    end_idx = max(0, min(end_idx, self.total_frames - 1))
                    
                    trial_frames = end_idx - start_idx + 1
                    duration_s = trial_frames / self.fps
                    
                    print(f"  Frame range: {start_frame} to {end_frame} (camera frame IDs)")
                    print(f"  Zarr indices: {start_idx} to {end_idx}")
                    print(f"  Duration: {duration_s:.2f} seconds ({trial_frames} frames)")
                    
                    # Get detection data for this trial
                    trial_n_detections = self.n_detections[start_idx:end_idx+1]
                    trial_bboxes = self.bboxes[start_idx:end_idx+1]
                    trial_scores = self.scores[start_idx:end_idx+1]
                    
                    # Calculate trial statistics
                    frames_with_detection = np.sum(trial_n_detections > 0)
                    trial_coverage = frames_with_detection / trial_frames * 100
                    
                    print(f"  Detection coverage: {trial_coverage:.1f}% ({frames_with_detection}/{trial_frames} frames)")
                    
                    # Check interpolation if available
                    if self.interpolation_mask is not None:
                        trial_interp_mask = self.interpolation_mask[start_idx:end_idx+1]
                        interpolated_frames = np.sum(trial_interp_mask)
                        real_frames = frames_with_detection - interpolated_frames
                        print(f"  Real detections: {real_frames} frames")
                        print(f"  Interpolated: {interpolated_frames} frames")
                    
                    # Calculate trajectory statistics
                    centroids = []
                    for j in range(len(trial_n_detections)):
                        if trial_n_detections[j] > 0:
                            centroid = self.calculate_centroid(trial_bboxes[j, 0])
                            centroids.append(centroid)
                    
                    if len(centroids) > 1:
                        centroids = np.array(centroids)
                        
                        # Calculate path length
                        distances = []
                        for j in range(1, len(centroids)):
                            dist = np.linalg.norm(centroids[j] - centroids[j-1])
                            distances.append(dist)
                        
                        total_distance = np.sum(distances)
                        avg_speed = total_distance / duration_s  # pixels per second
                        
                        print(f"  Trajectory length: {total_distance:.1f} pixels")
                        print(f"  Average speed: {avg_speed:.1f} pixels/second")
                        
                        # Store results
                        trial_results.append({
                            'trial_num': i + 1,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'duration_s': duration_s,
                            'coverage': trial_coverage,
                            'total_distance': total_distance,
                            'avg_speed': avg_speed,
                            'centroids': centroids,
                            'n_detections': trial_n_detections,
                            'interpolated_frames': interpolated_frames if self.interpolation_mask is not None else 0
                        })
                    else:
                        print("  ⚠️  Not enough detections for trajectory analysis")
                        
                else:
                    print("  ⚠️  No camera_frame_id field in events")
            
            # Visualize if requested
            if visualize and trial_results:
                self.visualize_trials(trial_results)
            
            return trial_results
    
    def visualize_trials(self, trial_results: List[Dict]):
        """Visualize trial trajectories and statistics."""
        
        n_trials = len(trial_results)
        n_cols = min(3, n_trials)
        n_rows = (n_trials + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 8*n_rows))
        fig.suptitle('Chase Trial Analysis with Zarr Detection Data', fontsize=16, fontweight='bold')
        
        for i, trial in enumerate(trial_results):
            # Trajectory plot
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            centroids = trial['centroids']
            
            # Plot trajectory
            ax.plot(centroids[:, 0], centroids[:, 1], 'b-', alpha=0.5, linewidth=1)
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c=np.arange(len(centroids)), cmap='viridis', 
                      s=10, alpha=0.7, zorder=2)
            
            # Mark start and stop
            ax.plot(centroids[0, 0], centroids[0, 1], 'g^', 
                   markersize=12, markeredgecolor='darkgreen', 
                   markeredgewidth=1.5, zorder=3, label='START')
            ax.plot(centroids[-1, 0], centroids[-1, 1], 'rs', 
                   markersize=12, markeredgecolor='darkred', 
                   markeredgewidth=1.5, zorder=3, label='STOP')
            
            # Set full image bounds
            ax.set_xlim(0, self.img_width)
            ax.set_ylim(0, self.img_height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            # Add trial info
            title = f"Trial {trial['trial_num']}"
            subtitle = (f"Duration: {trial['duration_s']:.1f}s | "
                       f"Coverage: {trial['coverage']:.1f}% | "
                       f"Speed: {trial['avg_speed']:.0f} px/s")
            
            if trial['interpolated_frames'] > 0:
                subtitle += f"\nInterpolated: {trial['interpolated_frames']} frames"
            
            ax.set_title(f"{title}\n{subtitle}", fontsize=10)
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
        
        plt.tight_layout()
        
        # Add summary statistics below
        fig.text(0.5, 0.02, self.generate_summary(trial_results), 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.show()
    
    def generate_summary(self, trial_results: List[Dict]) -> str:
        """Generate summary statistics for all trials."""
        
        coverages = [t['coverage'] for t in trial_results]
        speeds = [t['avg_speed'] for t in trial_results]
        distances = [t['total_distance'] for t in trial_results]
        
        summary = (f"Summary: {len(trial_results)} trials | "
                  f"Avg Coverage: {np.mean(coverages):.1f}% | "
                  f"Avg Speed: {np.mean(speeds):.0f} px/s | "
                  f"Avg Distance: {np.mean(distances):.0f} px")
        
        if self.use_interpolated:
            total_interp = sum(t['interpolated_frames'] for t in trial_results)
            if total_interp > 0:
                summary += f" | Total Interpolated: {total_interp} frames"
        
        return summary
    
    def export_trial_data(self, trial_results: List[Dict], output_path: str):
        """Export trial analysis results to JSON."""
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = []
        for trial in trial_results:
            trial_copy = trial.copy()
            trial_copy['centroids'] = trial_copy['centroids'].tolist()
            trial_copy['n_detections'] = trial_copy['n_detections'].tolist()
            export_data.append(trial_copy)
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'h5_file': str(self.h5_path),
                'zarr_file': str(self.zarr_path),
                'use_interpolated': self.use_interpolated,
                'total_coverage': self.coverage,
                'fps': self.fps,
                'trials': export_data
            }, f, indent=2)
        
        print(f"\n✅ Exported trial data to: {output_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Analyze trials using H5 metadata and zarr detection data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool combines:
- Trial structure and timing from H5 files
- Cleaned/interpolated detection data from zarr files

The zarr data should be the output from:
1. frame_distance_analyzer.py (filtering)
2. gap_interpolator.py (interpolation)

Examples:
  # Analyze using interpolated data
  %(prog)s analysis.h5 detections.zarr --visualize
  
  # Use original (unprocessed) detections
  %(prog)s analysis.h5 detections.zarr --use-original
  
  # Export results to JSON
  %(prog)s analysis.h5 detections.zarr --export trial_results.json
        """
    )
    
    parser.add_argument('h5_path', help='Path to H5 analysis file')
    parser.add_argument('zarr_path', help='Path to zarr detection file')
    
    parser.add_argument('--use-original', action='store_true',
                       help='Use original detections instead of preprocessed')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize trial trajectories')
    
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.h5_path).exists():
        print(f"❌ H5 file not found: {args.h5_path}")
        return 1
    
    if not Path(args.zarr_path).exists():
        print(f"❌ Zarr file not found: {args.zarr_path}")
        return 1
    
    # Create analyzer
    analyzer = TrialZarrAnalyzer(
        h5_path=args.h5_path,
        zarr_path=args.zarr_path,
        use_interpolated=not args.use_original
    )
    
    # Analyze trials
    results = analyzer.analyze_trials(visualize=args.visualize)
    
    # Export if requested
    if args.export and results:
        analyzer.export_trial_data(results, args.export)
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"✅ Successfully analyzed {len(results)} trials")
        print(f"{'='*70}")
    else:
        print(f"\n⚠️  No trials could be analyzed")
    
    return 0


if __name__ == '__main__':
    exit(main())