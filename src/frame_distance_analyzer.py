#!/usr/bin/env python3
"""
Frame-to-Frame Distance Analyzer

Simple script to calculate distances between detection centroids in consecutive frames.
Helps identify unrealistic jumps, short noisy segments, and understand detection quality.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def calculate_distance(centroid1, centroid2):
    """Calculate Euclidean distance between two centroids."""
    return np.linalg.norm(centroid2 - centroid1)


def analyze_frame_distances(zarr_path, threshold=100.0, visualize=False,
                           drop_jumps=False, save_cleaned=False, min_segment_length=10):
    """
    Analyze frame-to-frame movement distances and filter outliers.

    Args:
        zarr_path: Path to YOLO detection zarr file.
        threshold: Distance threshold for flagging jumps (pixels).
        visualize: If True, plot detailed before/after comparison graphs.
        drop_jumps: If True, perform the filtering of jumps and short segments.
        save_cleaned: If True, save cleaned data to a new group in the zarr file.
        min_segment_length: Minimum number of frames for a trajectory segment to be kept.
    
    Returns:
        Dictionary with analysis results.
    """
    print(f"\n{'='*60}")
    print("FRAME-TO-FRAME DISTANCE ANALYZER")
    print(f"{'='*60}")
    print(f"Zarr file: {zarr_path}")
    print(f"Jump threshold: {threshold} pixels")
    if drop_jumps:
        print(f"Drop mode: ENABLED")
        print(f"  - Removing blips > {threshold} pixels")
        print(f"  - Removing segments < {min_segment_length} frames")
    print()

    zarr_path = str(zarr_path).rstrip('/')
    root = zarr.open(zarr_path, mode='r+' if save_cleaned else 'r')

    bboxes = root['bboxes'][:]
    scores = root['scores'][:]
    class_ids = root['class_ids'][:]
    n_detections = root['n_detections'][:]
    
    fps = root.attrs.get('fps', 30.0)
    total_frames = len(n_detections)
    frames_with_detections = np.sum(n_detections > 0)
    
    print(f"Video info: {total_frames} frames @ {fps} FPS")
    print(f"Frames with detections: {frames_with_detections}/{total_frames} "
          f"({frames_with_detections/total_frames*100:.1f}%)")
    print()
    
    print("Calculating centroids...")
    centroids = []
    for frame_idx in range(total_frames):
        if n_detections[frame_idx] > 0:
            bbox = bboxes[frame_idx, 0]
            centroids.append({
                'frame': frame_idx,
                'centroid': calculate_centroid(bbox),
                'score': scores[frame_idx, 0]
            })
        else:
            centroids.append({
                'frame': frame_idx,
                'centroid': np.array([np.nan, np.nan]),
                'score': 0.0
            })
    
    print("Analyzing movements...")
    consecutive_distances, gap_distances, all_transitions = [], [], []
    prev_valid_idx = None
    
    for frame_idx in tqdm(range(total_frames), desc="Analyzing"):
        if n_detections[frame_idx] == 0:
            continue
        
        if prev_valid_idx is not None:
            prev_centroid = centroids[prev_valid_idx]['centroid']
            current_centroid = centroids[frame_idx]['centroid']
            distance = calculate_distance(prev_centroid, current_centroid)
            frame_gap = frame_idx - prev_valid_idx
            
            transition = {
                'from_frame': prev_valid_idx, 'to_frame': frame_idx,
                'distance': distance, 'frame_gap': frame_gap
            }
            all_transitions.append(transition)
            
            if frame_gap == 1:
                consecutive_distances.append(distance)
            else:
                gap_distances.append(transition)
        
        prev_valid_idx = frame_idx
    
    consecutive_distances = np.array(consecutive_distances)
    consecutive_jumps = [t for t in all_transitions if t['distance'] > threshold and t['frame_gap'] == 1]
    gap_jumps = [t for t in all_transitions if t['distance'] > threshold and t['frame_gap'] > 1]
    
    cleaned_bboxes, cleaned_scores, cleaned_n_detections = None, None, None
    drop_reasons = np.zeros(total_frames, dtype=np.int8)
    frames_from_short_segments = set()

    if drop_jumps:
        print(f"\n{'='*60}")
        print("FILTERING DATA")
        print(f"{'='*60}")
        
        # 0=kept, 1=jump, 2=short_segment
        
        # --- Stage 1: Iteratively remove blips ---
        max_iterations, iteration = 10, 0
        while iteration < max_iterations:
            iteration += 1
            new_drops_found = False
            
            surviving_indices = [
                i for i in range(total_frames)
                if n_detections[i] > 0 and drop_reasons[i] == 0
            ]
            
            if len(surviving_indices) < 3:
                break
                
            for i in range(1, len(surviving_indices) - 1):
                prev_idx = surviving_indices[i-1]
                current_idx = surviving_indices[i]
                next_idx = surviving_indices[i+1]
                
                centroid_A = centroids[prev_idx]['centroid']
                centroid_B = centroids[current_idx]['centroid']
                centroid_C = centroids[next_idx]['centroid']
                
                dist_AB = calculate_distance(centroid_A, centroid_B)
                dist_BC = calculate_distance(centroid_B, centroid_C)
                dist_AC = calculate_distance(centroid_A, centroid_C)
                
                if dist_AB > threshold and dist_BC > threshold and dist_AC < threshold:
                    if drop_reasons[current_idx] == 0:
                        drop_reasons[current_idx] = 1
                        new_drops_found = True
                        print(f"  Iter {iteration}: Dropping frame {current_idx} (blip detected between {prev_idx} and {next_idx})")

            if not new_drops_found:
                print(f"  Converged after {iteration} iterations.")
                break
        
        # --- Stage 2: Filter by minimum segment length (using temporal AND spatial continuity) ---
        if min_segment_length > 1:
            print(f"\nFiltering for segments shorter than {min_segment_length} frames...")
            surviving_frames = sorted([
                f for f in range(total_frames)
                if n_detections[f] > 0 and drop_reasons[f] == 0
            ])
            
            if surviving_frames:
                segments, current_segment = [], [surviving_frames[0]]
                for i in range(1, len(surviving_frames)):
                    prev_frame_idx = surviving_frames[i-1]
                    curr_frame_idx = surviving_frames[i]

                    # A segment is continuous if frames are consecutive AND the jump is below threshold
                    is_temporally_continuous = (curr_frame_idx == prev_frame_idx + 1)
                    distance = calculate_distance(centroids[prev_frame_idx]['centroid'], centroids[curr_frame_idx]['centroid'])
                    is_spatially_continuous = (distance <= threshold)

                    if is_temporally_continuous and is_spatially_continuous:
                        current_segment.append(curr_frame_idx)
                    else:
                        segments.append(current_segment)
                        current_segment = [curr_frame_idx]
                segments.append(current_segment)
                
                for segment in segments:
                    if len(segment) < min_segment_length:
                        frames_from_short_segments.update(segment)
                
                if frames_from_short_segments:
                    print(f"  Found {len(frames_from_short_segments)} frames in short segments. Marking for removal.")
                    for frame_idx in frames_from_short_segments:
                        # Only mark as short_segment if not already marked as a blip
                        if drop_reasons[frame_idx] == 0:
                            drop_reasons[frame_idx] = 2
                else:
                    print("  No segments found shorter than the minimum.")

        frames_dropped_count = np.count_nonzero(drop_reasons)
        print(f"\nTotal: Dropping {frames_dropped_count} detections.")
        
        cleaned_bboxes = bboxes.copy()
        cleaned_scores = scores.copy()
        cleaned_n_detections = n_detections.copy()
        cleaned_n_detections[drop_reasons > 0] = 0
        
        cleaned_frames_with_detections = np.sum(cleaned_n_detections > 0)
        print(f"Original frames with detections: {frames_with_detections}")
        print(f"Cleaned frames with detections: {cleaned_frames_with_detections}")
        print(f"Detections removed: {frames_with_detections - cleaned_frames_with_detections}")
        
    def save_data_to_zarr(run_name_suffix=''):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'filtered_{timestamp}{run_name_suffix}'
        print(f"\n{'='*60}\nSAVING FILTERED DATA...\n{'='*60}")
        print(f"Run name: {run_name}")
        
        filtered_group = root.require_group('filtered_runs')
        run_group = filtered_group.create_group(run_name)
        
        frames_dropped_count = np.count_nonzero(drop_reasons)
        dropped_indices = np.where(drop_reasons > 0)[0].tolist()
        
        run_group.attrs['created_at'] = datetime.now().isoformat()
        run_group.attrs['pipeline_step'] = 'remove_jumps_and_segments'
        run_group.attrs['frames_dropped'] = frames_dropped_count
        run_group.attrs['dropped_frame_indices'] = dropped_indices
        run_group.attrs['parameters'] = {
            'threshold_pixels': threshold,
            'min_segment_length_frames': min_segment_length,
            'initial_consecutive_jumps': len(consecutive_jumps),
            'initial_gap_jumps': len(gap_jumps),
            'frames_removed_total': frames_dropped_count
        }
        
        run_group.create_dataset('bboxes', data=cleaned_bboxes, dtype='float32')
        run_group.create_dataset('scores', data=cleaned_scores, dtype='float32')
        run_group.create_dataset('class_ids', data=class_ids, dtype='int32')
        run_group.create_dataset('n_detections', data=cleaned_n_detections, dtype='int32')
        run_group.create_dataset('drop_reasons', data=drop_reasons, dtype='int8')
        
        filtered_group.attrs['latest'] = run_name
        print(f"✓ Filtered data saved to: {zarr_path}/filtered_runs/{run_name}")
        print(f"✓ Frames removed: {frames_dropped_count}")
        print("="*60)
        
    if save_cleaned and not visualize and drop_jumps:
        save_data_to_zarr()
    
    # --- STATISTICS & VISUALIZATION ---
    # (Analysis summary is printed below visualization block)
    
    if visualize and drop_jumps:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), gridspec_kw={'width_ratios': [2, 3, 2]})
        fig.suptitle('Before vs After Filtering Comparison', fontsize=16, fontweight='bold')
        
        # === BEFORE (Top Row) ===
        ax1 = axes[0, 0]
        ax1.plot(consecutive_distances, 'b-', alpha=0.6, lw=1)
        ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold} px)')
        ax1.set_title('Movement Distances - BEFORE')
        ax1.set(xlabel='Consecutive Transition Index', ylabel='Distance (pixels)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        valid_centroids_before = np.array([c['centroid'] for c in centroids if not np.isnan(c['centroid']).any()])
        valid_frames_before = [c['frame'] for c in centroids if not np.isnan(c['centroid']).any()]
        scatter = ax2.scatter(valid_centroids_before[:, 0], valid_centroids_before[:, 1], c=valid_frames_before, cmap='viridis', s=5, alpha=0.5, zorder=2)
        
        # Plot segments for context
        valid_indices = [i for i, c in enumerate(centroids) if not np.isnan(c['centroid']).any()]
        if valid_indices:
            segments, current_segment = [], [centroids[valid_indices[0]]['centroid']]
            for i in range(1, len(valid_indices)):
                if valid_indices[i] == valid_indices[i-1] + 1:
                    current_segment.append(centroids[valid_indices[i]]['centroid'])
                else:
                    if len(current_segment) > 1: segments.append(np.array(current_segment))
                    current_segment = [centroids[valid_indices[i]]['centroid']]
            if len(current_segment) > 1: segments.append(np.array(current_segment))
            
            for segment in segments:
                ax2.plot(segment[:, 0], segment[:, 1], '-', color='gray', alpha=0.4, lw=1, zorder=1)

        all_jump_frames = {t['to_frame'] for t in consecutive_jumps + gap_jumps}
        # Mark frames from short segments in Cyan (with a smaller size)
        for frame_idx in frames_from_short_segments:
            if not np.isnan(centroids[frame_idx]['centroid']).any():
                ax2.plot(centroids[frame_idx]['centroid'][0], centroids[frame_idx]['centroid'][1],
                        'o', color='c', markersize=4,
                        markeredgecolor='black', alpha=0.8, mew=0.5, zorder=3)

        # Mark jumps in Red (with a smaller size)
        for frame_idx in all_jump_frames:
            if not np.isnan(centroids[frame_idx]['centroid']).any():
                ax2.plot(centroids[frame_idx]['centroid'][0], centroids[frame_idx]['centroid'][1],
                        'ro', markersize=6,
                        markeredgecolor='white', zorder=4)

        ax2.set_title(f'Trajectory - BEFORE (Red=Jumps, Cyan=Short Segments)')
        ax2.set(xlabel='X Position (pixels)', ylabel='Y Position (pixels)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[0, 2]
        ax3.axis('off')
        stats_text = f"""BEFORE FILTERING:
        \nFrames with detections: {frames_with_detections}
        \nTotal initial jumps: {len(all_jump_frames)}
        \nMovement (consecutive):
        - Mean: {np.mean(consecutive_distances):.2f} px
        - Median: {np.median(consecutive_distances):.2f} px
        - 95th percentile: {np.percentile(consecutive_distances, 95):.2f} px"""
        ax3.text(0.05, 0.5, stats_text, transform=ax3.transAxes, fontsize=11, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # === AFTER (Bottom Row) ===
        cleaned_consecutive_distances = []
        prev_valid_idx_after = None
        for i in range(total_frames):
            if cleaned_n_detections[i] > 0:
                if prev_valid_idx_after is not None:
                    # Only calculate distance for truly consecutive frames
                    if i == prev_valid_idx_after + 1:
                        dist = calculate_distance(centroids[prev_valid_idx_after]['centroid'], centroids[i]['centroid'])
                        cleaned_consecutive_distances.append(dist)
                prev_valid_idx_after = i
        cleaned_consecutive_distances = np.array(cleaned_consecutive_distances)

        ax4 = axes[1, 0]
        if len(cleaned_consecutive_distances) > 0:
            ax4.plot(cleaned_consecutive_distances, 'g-', alpha=0.6, lw=1)
            ax4.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold} px)')
        ax4.set_title('Movement Distances - AFTER')
        ax4.set(xlabel='Consecutive Transition Index', ylabel='Distance (pixels)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = axes[1, 1]
        valid_frames_after = np.where(cleaned_n_detections > 0)[0]
        if len(valid_frames_after) > 0:
            valid_centroids_after = np.array([centroids[i]['centroid'] for i in valid_frames_after])
            ax5.scatter(valid_centroids_after[:, 0], valid_centroids_after[:, 1], c=valid_frames_after, cmap='viridis', s=5, alpha=0.8, zorder=2)
            
            # Segmented plotting for better visualization of trajectory breaks
            segments_after, current_segment_after = [], []
            if valid_frames_after.size > 0:
                current_segment_after.append(valid_centroids_after[0])
                for i in range(1, len(valid_frames_after)):
                    if valid_frames_after[i] == valid_frames_after[i-1] + 1:
                        current_segment_after.append(valid_centroids_after[i])
                    else:
                        if len(current_segment_after) > 1: segments_after.append(np.array(current_segment_after))
                        current_segment_after = [valid_centroids_after[i]]
                if len(current_segment_after) > 1: segments_after.append(np.array(current_segment_after))
            
            for segment in segments_after:
                ax5.plot(segment[:, 0], segment[:, 1], '-', color='#228B22', alpha=0.6, lw=1.5, zorder=1) # ForestGreen

        ax5.set_title('Trajectory - AFTER (Filtered)')
        ax5.set(xlabel='X Position (pixels)', ylabel='Y Position (pixels)')
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)

        ax6 = axes[1, 2]
        ax6.axis('off')
        frames_dropped_count = np.count_nonzero(drop_reasons)
        stats_text_after = f"""AFTER FILTERING:
        \nFrames with detections: {cleaned_frames_with_detections}
        \nFrames removed: {frames_dropped_count} ({frames_dropped_count/frames_with_detections*100:.1f}% of detected)
        - Blips: {np.sum(drop_reasons == 1)}
        - Short Segments: {np.sum(drop_reasons == 2)}
        \nMovement (consecutive):
        - Mean: {np.mean(cleaned_consecutive_distances):.2f} px
        - Median: {np.median(cleaned_consecutive_distances):.2f} px
        - Max: {np.max(cleaned_consecutive_distances):.2f} px""" if len(cleaned_consecutive_distances) > 0 else "AFTER FILTERING:\n\nNo consecutive frames remaining."
        ax6.text(0.05, 0.5, stats_text_after, transform=ax6.transAxes, fontsize=11, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig.text(0.5, 0.02, "Press 'W' to save filtered data | 'Q' to quit | Close window to cancel",
                 ha='center', fontsize=12, weight='bold', bbox=dict(facecolor='yellow', alpha=0.7))
        
        def on_key(event):
            if event.key.lower() == 'w':
                save_data_to_zarr('_interactive')
                plt.close(fig)
            elif event.key.lower() == 'q':
                print("\n✗ Cancelled - no data saved.")
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("\n" + "="*60 + "\nINTERACTIVE MODE\n" + "="*60)
        print("Press 'W' in the plot window to save.")
        print("Press 'Q' to quit without saving.")
        plt.show()

    # --- Print Final Analysis Summary ---
    print(f"\n{'='*60}\nANALYSIS RESULTS\n{'='*60}")
    print(f"\nMOVEMENT STATISTICS (Initial):")
    if len(consecutive_distances) > 0:
        print(f"  - Mean: {np.mean(consecutive_distances):.2f} px, Median: {np.median(consecutive_distances):.2f} px, "
              f"95th percentile: {np.percentile(consecutive_distances, 95):.2f} px")
    print(f"  - Total initial jumps found (> {threshold} px): {len(consecutive_jumps) + len(gap_jumps)}")
    if drop_jumps:
        frames_dropped_count = np.count_nonzero(drop_reasons)
        print(f"\nFILTERING RESULTS:")
        print(f"  - Total frames removed: {frames_dropped_count}")
        print(f"    - Due to blips: {np.sum(drop_reasons == 1)}")
        print(f"    - Due to short segments: {np.sum(drop_reasons == 2)}")
        print(f"  - Remaining frames with detections: {np.sum(cleaned_n_detections > 0)}")

    return {
        'frames_dropped': np.count_nonzero(drop_reasons) if drop_jumps else 0,
        'drop_reasons': drop_reasons if drop_jumps else None,
        'statistics': { 'jump_rate': len(consecutive_jumps)/len(consecutive_distances) if len(consecutive_distances) > 0 else 0 },
        'consecutive_distances': consecutive_distances
    }

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and filter frame-to-frame distances in YOLO detections.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis and visualization
  %(prog)s detections.zarr --visualize

  # Filter jumps and short segments, then visualize
  %(prog)s detections.zarr --drop --threshold 75 --min-segment-length 15 --visualize
  
  # Filter and save the cleaned data without opening a plot
  %(prog)s detections.zarr --drop --save --threshold 75
        """
    )
    parser.add_argument('zarr_path', help='Path to YOLO detection zarr file.')
    parser.add_argument('--threshold', type=float, default=100.0,
                        help='Distance threshold for flagging jumps (pixels). Default: 100')
    parser.add_argument('--visualize', action='store_true', help='Show detailed before/after comparison plots.')
    parser.add_argument('--drop', action='store_true', help='Enable filtering of jumps and short segments.')
    parser.add_argument('--save', action='store_true', help='Save cleaned data to zarr (requires --drop).')
    parser.add_argument('--min-segment-length', type=int, default=10,
                        help='(Requires --drop) Minimum frames for a trajectory segment to be kept. Default: 10')
    args = parser.parse_args()
    
    if args.save and not args.drop:
        parser.error("--save requires --drop to be enabled.")
    
    results = analyze_frame_distances(
        zarr_path=args.zarr_path,
        threshold=args.threshold,
        visualize=args.visualize,
        drop_jumps=args.drop,
        save_cleaned=args.save,
        min_segment_length=args.min_segment_length
    )
    
    print(f"\n{'='*60}\nAnalysis complete!\n{'='*60}")
    
    if not args.drop and results['statistics']['jump_rate'] > 0.1:
        p95 = np.percentile(results['consecutive_distances'], 95)
        print(f"💡 Suggestion: A high jump rate was detected. "
              f"Consider running with --drop --threshold {p95:.1f}")

if __name__ == '__main__':
    main()