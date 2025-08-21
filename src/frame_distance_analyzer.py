#!/usr/bin/env python3
"""
Frame-to-Frame Distance Analyzer

Simple script to calculate distances between detection centroids in consecutive frames.
Helps identify unrealistic jumps and understand detection quality.
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
                           drop_jumps=False, save_cleaned=False):
    """
    Analyze frame-to-frame movement distances.
    
    Args:
        zarr_path: Path to YOLO detection zarr file
        threshold: Distance threshold for flagging jumps (pixels)
        visualize: If True, plot distance graph
        drop_jumps: If True, remove detections that are jumps
        save_cleaned: If True, save cleaned data to zarr (requires drop_jumps)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print("FRAME-TO-FRAME DISTANCE ANALYZER")
    print(f"{'='*60}")
    print(f"Zarr file: {zarr_path}")
    print(f"Jump threshold: {threshold} pixels")
    if drop_jumps:
        print(f"Drop mode: ENABLED - will remove jumps > {threshold} pixels")
    print()
    
    # Load zarr data
    root = zarr.open(str(zarr_path), mode='r+' if save_cleaned else 'r')
    
    # Load detection data
    bboxes = root['bboxes'][:]
    scores = root['scores'][:]
    class_ids = root['class_ids'][:]
    n_detections = root['n_detections'][:]
    
    # Get metadata
    fps = root.attrs.get('fps', 30.0)
    total_frames = len(n_detections)
    frames_with_detections = np.sum(n_detections > 0)
    
    print(f"Video info: {total_frames} frames @ {fps} FPS")
    print(f"Frames with detections: {frames_with_detections}/{total_frames} "
          f"({frames_with_detections/total_frames*100:.1f}%)")
    print()
    
    # Calculate all centroids first
    print("Calculating centroids...")
    centroids = []
    for frame_idx in range(total_frames):
        if n_detections[frame_idx] > 0:
            bbox = bboxes[frame_idx, 0]  # First detection
            centroid = calculate_centroid(bbox)
            centroids.append({
                'frame': frame_idx,
                'centroid': centroid,
                'score': scores[frame_idx, 0]
            })
        else:
            centroids.append({
                'frame': frame_idx,
                'centroid': np.array([np.nan, np.nan]),
                'score': 0.0
            })
    
    # Calculate frame-to-frame distances
    print("Analyzing movements...")
    consecutive_distances = []  # Truly consecutive frames (no gaps)
    gap_distances = []         # Distances across gaps
    all_transitions = []       # All transitions for reference
    
    prev_valid_idx = None
    
    for frame_idx in tqdm(range(total_frames)):
        if n_detections[frame_idx] == 0:
            continue
            
        current_centroid = centroids[frame_idx]['centroid']
        
        if prev_valid_idx is not None:
            prev_centroid = centroids[prev_valid_idx]['centroid']
            distance = calculate_distance(prev_centroid, current_centroid)
            frame_gap = frame_idx - prev_valid_idx
            
            transition = {
                'from_frame': prev_valid_idx,
                'to_frame': frame_idx,
                'distance': distance,
                'frame_gap': frame_gap,
                'time_gap': frame_gap / fps,
                'velocity': distance * fps / frame_gap  # pixels per second
            }
            
            all_transitions.append(transition)
            
            if frame_gap == 1:
                # True consecutive frames
                consecutive_distances.append(distance)
            else:
                # Gap between detections
                gap_distances.append({
                    'distance': distance,
                    'gap_size': frame_gap,
                    'from_frame': prev_valid_idx,
                    'to_frame': frame_idx
                })
        
        prev_valid_idx = frame_idx
    
    # Analyze results
    consecutive_distances = np.array(consecutive_distances)
    
    # Find jumps (both consecutive and across gaps)
    consecutive_jumps = []
    gap_jumps = []
    
    for trans in all_transitions:
        if trans['distance'] > threshold:
            if trans['frame_gap'] == 1:
                consecutive_jumps.append(trans)
            else:
                gap_jumps.append(trans)
    
    # Drop jumps if requested
    frames_to_drop = set()
    cleaned_bboxes = None
    cleaned_scores = None
    cleaned_n_detections = None
    
    if drop_jumps:
        print(f"\n{'='*60}")
        print("FILTERING DATA")
        print(f"{'='*60}")
        
        # Collect all frames that are jump destinations
        for jump in consecutive_jumps:
            frames_to_drop.add(jump['to_frame'])
        
        for jump in gap_jumps:
            frames_to_drop.add(jump['to_frame'])
        
        # Also check for isolated detections (optional)
        # An isolated detection is one that has large distances to both neighbors
        for i, trans in enumerate(all_transitions):
            frame = trans['to_frame']
            
            # Check if this frame is isolated (large distance from previous)
            if trans['distance'] > threshold:
                # Already handled above
                pass
            
            # Check if next transition from this frame is also large
            next_trans = None
            for t in all_transitions:
                if t['from_frame'] == frame:
                    next_trans = t
                    break
            
            if next_trans and next_trans['distance'] > threshold:
                # This frame has large jumps both TO and FROM it - it's isolated
                frames_to_drop.add(frame)
                print(f"  Found isolated detection at frame {frame}")
        
        # Also check for suspicious first/last detections
        if len(all_transitions) > 0:
            # Check if first detection is far from second
            first_detection_frame = None
            second_detection_frame = None
            
            for frame_idx in range(total_frames):
                if n_detections[frame_idx] > 0:
                    if first_detection_frame is None:
                        first_detection_frame = frame_idx
                    elif second_detection_frame is None:
                        second_detection_frame = frame_idx
                        break
            
            if first_detection_frame is not None and second_detection_frame is not None:
                first_centroid = centroids[first_detection_frame]['centroid']
                second_centroid = centroids[second_detection_frame]['centroid']
                dist = calculate_distance(first_centroid, second_centroid)
                
                if dist > threshold:
                    frames_to_drop.add(first_detection_frame)
                    print(f"  First detection (frame {first_detection_frame}) is {dist:.1f} pixels from second - dropping")
        
        print(f"Total: Dropping {len(frames_to_drop)} detections (jumps, isolated points, and outliers)")
        
        # Create cleaned versions
        cleaned_bboxes = bboxes.copy()
        cleaned_scores = scores.copy()
        cleaned_n_detections = n_detections.copy()
        
        # Zero out the dropped frames
        for frame_idx in frames_to_drop:
            cleaned_n_detections[frame_idx] = 0
            # Note: We don't need to clear bbox/score arrays since n_detections=0 
            # tells downstream code to ignore them
        
        cleaned_frames_with_detections = np.sum(cleaned_n_detections > 0)
        print(f"Original frames with detections: {frames_with_detections}")
        print(f"Cleaned frames with detections: {cleaned_frames_with_detections}")
        print(f"Detections removed: {frames_with_detections - cleaned_frames_with_detections}")
        
    # Auto-save without visualization (only when --save is used without --visualize)
    if save_cleaned and not visualize and drop_jumps:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'filtered_{timestamp}'
        
        print(f"\n" + "="*60)
        print("SAVING FILTERED DATA...")
        print(f"="*60)
        print(f"Run name: {run_name}")
        
        # Create filtered_runs group
        if 'filtered_runs' not in root:
            filtered_group = root.create_group('filtered_runs')
            filtered_group.attrs['created_at'] = datetime.now().isoformat()
            filtered_group.attrs['description'] = 'Filtered detection data with jumps/outliers removed'
        else:
            filtered_group = root['filtered_runs']
        
        # Create run group
        run_group = filtered_group.create_group(run_name)
        run_group.attrs['created_at'] = datetime.now().isoformat()
        run_group.attrs['pipeline_step'] = 'remove_jumps'
        run_group.attrs['source'] = 'original'
        run_group.attrs['threshold'] = threshold
        run_group.attrs['frames_dropped'] = len(frames_to_drop)
        run_group.attrs['dropped_frame_indices'] = sorted(list(frames_to_drop))
        run_group.attrs['parameters'] = {
            'threshold': threshold,
            'frames_removed': len(frames_to_drop),
            'consecutive_jumps_found': len(consecutive_jumps),
            'gap_jumps_found': len(gap_jumps)
        }
        
        # Save filtered data
        run_group.create_dataset('bboxes', data=cleaned_bboxes, dtype='float32')
        run_group.create_dataset('scores', data=cleaned_scores, dtype='float32')
        run_group.create_dataset('class_ids', data=class_ids, dtype='int32')
        run_group.create_dataset('n_detections', data=cleaned_n_detections, dtype='int32')
        
        # Store which frames were dropped
        drop_mask = np.zeros(total_frames, dtype=bool)
        for frame_idx in frames_to_drop:
            drop_mask[frame_idx] = True
        run_group.create_dataset('drop_mask', data=drop_mask, dtype=bool)
        
        # Update latest
        filtered_group.attrs['latest'] = run_name
        
        print(f"✓ Filtered data saved successfully")
        print(f"✓ Saved to: {zarr_path}/filtered_runs/{run_name}")
        print(f"✓ Set as latest filtered run")
        print(f"✓ Frames removed: {len(frames_to_drop)}")
        print("="*60)
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print("\nMOVEMENT STATISTICS:")
    print(f"  Total transitions analyzed: {len(all_transitions)}")
    print(f"  Consecutive frame transitions: {len(consecutive_distances)}")
    print(f"  Transitions across gaps: {len(gap_distances)}")
    
    if len(consecutive_distances) > 0:
        print(f"\n  Consecutive frame distances:")
        print(f"    Mean: {np.mean(consecutive_distances):.2f} pixels")
        print(f"    Median: {np.median(consecutive_distances):.2f} pixels")
        print(f"    Std: {np.std(consecutive_distances):.2f} pixels")
        print(f"    Min: {np.min(consecutive_distances):.2f} pixels")
        print(f"    Max: {np.max(consecutive_distances):.2f} pixels")
        print(f"    95th percentile: {np.percentile(consecutive_distances, 95):.2f} pixels")
    
    print(f"\n  JUMPS DETECTED (>{threshold} pixels):")
    
    if len(consecutive_jumps) > 0:
        print(f"    Consecutive frame jumps: {len(consecutive_jumps)}")
        print(f"    Jump rate: {len(consecutive_jumps)/len(consecutive_distances)*100:.2f}%")
        
        print(f"\n    Top 3 consecutive jumps:")
        sorted_jumps = sorted(consecutive_jumps, key=lambda x: x['distance'], reverse=True)[:3]
        for i, jump in enumerate(sorted_jumps, 1):
            print(f"      {i}. Frame {jump['from_frame']} → {jump['to_frame']}: "
                  f"{jump['distance']:.1f} pixels ({jump['velocity']:.1f} px/sec)")
    
    if len(gap_jumps) > 0:
        print(f"\n    Gap-crossing jumps: {len(gap_jumps)}")
        print(f"    (Jumps after detection gaps)")
        
        print(f"\n    Top 3 gap-crossing jumps:")
        sorted_gap_jumps = sorted(gap_jumps, key=lambda x: x['distance'], reverse=True)[:3]
        for i, jump in enumerate(sorted_gap_jumps, 1):
            print(f"      {i}. Frame {jump['from_frame']} → {jump['to_frame']} "
                  f"(gap: {jump['frame_gap']} frames): {jump['distance']:.1f} pixels")
    
    total_jumps = len(consecutive_jumps) + len(gap_jumps)
    if total_jumps > 0:
        print(f"\n    TOTAL JUMPS: {total_jumps}")
        all_jumps = consecutive_jumps + gap_jumps
        print(f"    Problem frames: {total_jumps}/{len(all_transitions)} transitions "
              f"({total_jumps/len(all_transitions)*100:.1f}%)")
    
    if len(gap_distances) > 0:
        print(f"\n  Gap statistics:")
        gap_sizes = [g['gap_size'] for g in gap_distances]
        print(f"    Number of gaps: {len(gap_distances)}")
        print(f"    Average gap size: {np.mean(gap_sizes):.1f} frames")
        print(f"    Max gap size: {np.max(gap_sizes)} frames")
    
    # Visualize if requested
    if visualize and len(consecutive_distances) > 0:
        if drop_jumps:
            # Create before/after comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Before vs After Filtering Comparison', fontsize=16, fontweight='bold')
            
            # === BEFORE (Top Row) ===
            
            # Plot 1: Distance plot (before)
            ax1 = axes[0, 0]
            ax1.plot(consecutive_distances, 'b-', alpha=0.6, linewidth=1)
            ax1.axhline(y=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold} px)')
            ax1.set_xlabel('Consecutive Transition Index')
            ax1.set_ylabel('Distance (pixels)')
            ax1.set_title('Movement Distances - BEFORE')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Trajectory (before)
            ax2 = axes[0, 1]
            valid_centroids_before = [c['centroid'] for c in centroids if not np.isnan(c['centroid'][0])]
            valid_frames_before = [c['frame'] for c in centroids if not np.isnan(c['centroid'][0])]
            
            if valid_centroids_before:
                valid_centroids_before = np.array(valid_centroids_before)
                scatter = ax2.scatter(valid_centroids_before[:, 0], valid_centroids_before[:, 1], 
                                     c=valid_frames_before, cmap='viridis', s=2, alpha=0.6)
                ax2.plot(valid_centroids_before[:, 0], valid_centroids_before[:, 1], 
                        'b-', alpha=0.2, linewidth=0.5)
                
                # Mark jumps
                all_jump_frames = []
                for jump in consecutive_jumps:
                    if jump['to_frame'] < len(centroids):
                        all_jump_frames.append(jump['to_frame'])
                for jump in gap_jumps:
                    if jump['to_frame'] < len(centroids):
                        all_jump_frames.append(jump['to_frame'])
                
                for jump_frame in all_jump_frames:
                    if jump_frame < len(centroids):
                        jump_centroid = centroids[jump_frame]['centroid']
                        if not np.isnan(jump_centroid[0]):
                            ax2.plot(jump_centroid[0], jump_centroid[1], 
                                    'ro', markersize=8, markeredgecolor='white')
                
                ax2.set_xlabel('X Position (pixels)')
                ax2.set_ylabel('Y Position (pixels)')
                ax2.set_title(f'Trajectory - BEFORE (Red = Jumps)')
                ax2.set_aspect('equal')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Statistics (before)
            ax3 = axes[0, 2]
            ax3.axis('off')
            stats_text = f"""BEFORE FILTERING:
            
Frames with detections: {frames_with_detections}
Total jumps found: {len(consecutive_jumps) + len(gap_jumps)}
  • Consecutive jumps: {len(consecutive_jumps)}
  • Gap-crossing jumps: {len(gap_jumps)}

Movement stats:
  • Mean: {np.mean(consecutive_distances):.2f} px
  • Median: {np.median(consecutive_distances):.2f} px
  • Max: {np.max(consecutive_distances):.2f} px
  • 95th percentile: {np.percentile(consecutive_distances, 95):.2f} px"""
            ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # === AFTER (Bottom Row) ===
            
            # Recalculate distances for cleaned data
            cleaned_transitions = []
            cleaned_consecutive_distances = []
            prev_valid_idx = None
            
            for frame_idx in range(total_frames):
                if cleaned_n_detections[frame_idx] == 0:
                    continue
                
                if prev_valid_idx is not None:
                    prev_centroid = centroids[prev_valid_idx]['centroid']
                    curr_centroid = centroids[frame_idx]['centroid']
                    distance = calculate_distance(prev_centroid, curr_centroid)
                    frame_gap = frame_idx - prev_valid_idx
                    
                    if frame_gap == 1:
                        cleaned_consecutive_distances.append(distance)
                
                prev_valid_idx = frame_idx
            
            cleaned_consecutive_distances = np.array(cleaned_consecutive_distances)
            
            # Plot 4: Distance plot (after)
            ax4 = axes[1, 0]
            if len(cleaned_consecutive_distances) > 0:
                ax4.plot(cleaned_consecutive_distances, 'g-', alpha=0.6, linewidth=1)
                ax4.axhline(y=threshold, color='r', linestyle='--', 
                           label=f'Threshold ({threshold} px)')
            ax4.set_xlabel('Consecutive Transition Index')
            ax4.set_ylabel('Distance (pixels)')
            ax4.set_title('Movement Distances - AFTER')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Trajectory (after)
            ax5 = axes[1, 1]
            viz_centroids = []
            for frame_idx in range(total_frames):
                if cleaned_n_detections[frame_idx] > 0:
                    bbox = bboxes[frame_idx, 0]
                    centroid = calculate_centroid(bbox)
                    viz_centroids.append({
                        'frame': frame_idx,
                        'centroid': centroid,
                        'score': scores[frame_idx, 0]
                    })
            
            valid_centroids_after = [c['centroid'] for c in viz_centroids if not np.isnan(c['centroid'][0])]
            valid_frames_after = [c['frame'] for c in viz_centroids if not np.isnan(c['centroid'][0])]
            
            if valid_centroids_after:
                valid_centroids_after = np.array(valid_centroids_after)
                scatter = ax5.scatter(valid_centroids_after[:, 0], valid_centroids_after[:, 1], 
                                     c=valid_frames_after, cmap='viridis', s=2, alpha=0.6)
                ax5.plot(valid_centroids_after[:, 0], valid_centroids_after[:, 1], 
                        'g-', alpha=0.2, linewidth=0.5)
                
                ax5.set_xlabel('X Position (pixels)')
                ax5.set_ylabel('Y Position (pixels)')
                ax5.set_title('Trajectory - AFTER (Filtered)')
                ax5.set_aspect('equal')
                ax5.grid(True, alpha=0.3)
            
            # Plot 6: Statistics (after)
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            cleaned_frames_with_detections = np.sum(cleaned_n_detections > 0)
            if len(cleaned_consecutive_distances) > 0:
                stats_text_after = f"""AFTER FILTERING:

Frames with detections: {cleaned_frames_with_detections}
Frames removed: {len(frames_to_drop)}
Removal rate: {len(frames_to_drop)/frames_with_detections*100:.1f}%

Movement stats:
  • Mean: {np.mean(cleaned_consecutive_distances):.2f} px
  • Median: {np.median(cleaned_consecutive_distances):.2f} px
  • Max: {np.max(cleaned_consecutive_distances):.2f} px
  • 95th percentile: {np.percentile(cleaned_consecutive_distances, 95):.2f} px

✓ All jumps >{threshold}px removed"""
            else:
                stats_text_after = f"""AFTER CLEANING:

Frames with detections: {cleaned_frames_with_detections}
Frames removed: {len(frames_to_drop)}

⚠ No consecutive frames remaining"""
            
            ax6.text(0.1, 0.5, stats_text_after, transform=ax6.transAxes, 
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            plt.tight_layout()
            
            # Add instructions
            fig.text(0.5, 0.01, 
                    "Press 'W' to write/save filtered data | Press 'Q' to quit without saving | Close window to cancel", 
                    ha='center', fontsize=12, weight='bold', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Interactive key handling
            save_completed = [False]  # Use list to modify in nested function
            
            def on_key(event):
                if event.key == 'w':  # Always allow saving when visualizing
                    print("\n" + "="*60)
                    print("SAVING FILTERED DATA...")
                    print("="*60)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    run_name = f'filtered_{timestamp}'
                    
                    # Create filtered_runs group
                    if 'filtered_runs' not in root:
                        filtered_group = root.create_group('filtered_runs')
                        filtered_group.attrs['created_at'] = datetime.now().isoformat()
                        filtered_group.attrs['description'] = 'Filtered detection data with jumps/outliers removed'
                    else:
                        filtered_group = root['filtered_runs']
                    
                    # Create run group
                    run_group = filtered_group.create_group(run_name)
                    run_group.attrs['created_at'] = datetime.now().isoformat()
                    run_group.attrs['pipeline_step'] = 'remove_jumps'
                    run_group.attrs['source'] = 'original'
                    run_group.attrs['threshold'] = threshold
                    run_group.attrs['frames_dropped'] = len(frames_to_drop)
                    run_group.attrs['dropped_frame_indices'] = sorted(list(frames_to_drop))
                    run_group.attrs['parameters'] = {
                        'threshold': threshold,
                        'frames_removed': len(frames_to_drop),
                        'consecutive_jumps_found': len(consecutive_jumps),
                        'gap_jumps_found': len(gap_jumps)
                    }
                    
                    # Save filtered data
                    run_group.create_dataset('bboxes', data=cleaned_bboxes, dtype='float32')
                    run_group.create_dataset('scores', data=cleaned_scores, dtype='float32')
                    run_group.create_dataset('class_ids', data=class_ids, dtype='int32')
                    run_group.create_dataset('n_detections', data=cleaned_n_detections, dtype='int32')
                    
                    # Store which frames were dropped
                    drop_mask = np.zeros(total_frames, dtype=bool)
                    for frame_idx in frames_to_drop:
                        drop_mask[frame_idx] = True
                    run_group.create_dataset('drop_mask', data=drop_mask, dtype=bool)
                    
                    # Update latest
                    filtered_group.attrs['latest'] = run_name
                    
                    print(f"✓ Filtered data saved as: {run_name}")
                    print(f"✓ Saved to: {zarr_path}/filtered_runs/{run_name}")
                    print(f"✓ Set as latest filtered run")
                    print(f"✓ Frames removed: {len(frames_to_drop)}")
                    print("="*60)
                    save_completed[0] = True
                    plt.close('all')
                
                elif event.key == 'q':
                    print("\n✗ Cancelled - no data saved")
                    plt.close('all')
            
            # Always connect key handler when visualizing
            fig.canvas.mpl_connect('key_press_event', on_key)
            print("\n" + "="*60)
            print("INTERACTIVE MODE")
            print("Press 'W' to write/save filtered data to disk")
            print("Press 'Q' to quit without saving")
            print("(Note: 'S' opens matplotlib's save dialog)")
            print("="*60)
            
            plt.show()
            
        else:
            # Original visualization when not dropping
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Consecutive frame distances over time
            ax1 = axes[0, 0]
            ax1.plot(consecutive_distances, 'b-', alpha=0.6, linewidth=1)
            ax1.axhline(y=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold} px)')
            ax1.set_xlabel('Consecutive Transition Index')
            ax1.set_ylabel('Distance (pixels)')
            ax1.set_title('Frame-to-Frame Movement Distances (Consecutive Only)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Distance histogram
            ax2 = axes[0, 1]
            ax2.hist(consecutive_distances, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(x=threshold, color='r', linestyle='--', 
                       label=f'Threshold ({threshold} px)')
            ax2.set_xlabel('Distance (pixels)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Frame-to-Frame Distances')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Trajectory
            ax3 = axes[1, 0]
            valid_centroids = [c['centroid'] for c in centroids if not np.isnan(c['centroid'][0])]
            valid_frames = [c['frame'] for c in centroids if not np.isnan(c['centroid'][0])]
            
            if valid_centroids:
                valid_centroids = np.array(valid_centroids)
                scatter = ax3.scatter(valid_centroids[:, 0], valid_centroids[:, 1], 
                                     c=valid_frames, cmap='viridis', s=2, alpha=0.6)
                ax3.plot(valid_centroids[:, 0], valid_centroids[:, 1], 
                        'b-', alpha=0.2, linewidth=0.5)
                
                # Mark ALL jump locations
                all_jump_frames = []
                for jump in consecutive_jumps:
                    if jump['to_frame'] < len(centroids):
                        all_jump_frames.append(jump['to_frame'])
                for jump in gap_jumps:
                    if jump['to_frame'] < len(centroids):
                        all_jump_frames.append(jump['to_frame'])
                
                for jump_frame in all_jump_frames:
                    if jump_frame < len(centroids):
                        jump_centroid = centroids[jump_frame]['centroid']
                        if not np.isnan(jump_centroid[0]):
                            ax3.plot(jump_centroid[0], jump_centroid[1], 
                                    'ro', markersize=8, markeredgecolor='white')
                
                plt.colorbar(scatter, ax=ax3, label='Frame Number')
                ax3.set_xlabel('X Position (pixels)')
                ax3.set_ylabel('Y Position (pixels)')
                ax3.set_title('Detection Trajectory (Red = Jumps > Threshold)')
                ax3.set_aspect('equal')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Detection gaps
            ax4 = axes[1, 1]
            detection_mask = n_detections > 0
            gap_starts = []
            gap_lengths = []
            
            in_gap = False
            gap_start = None
            
            for i, has_detection in enumerate(detection_mask):
                if not has_detection and not in_gap:
                    in_gap = True
                    gap_start = i
                elif has_detection and in_gap:
                    gap_starts.append(gap_start)
                    gap_lengths.append(i - gap_start)
                    in_gap = False
            
            if gap_lengths:
                ax4.bar(range(len(gap_lengths)), gap_lengths, color='red', alpha=0.7)
                ax4.set_xlabel('Gap Index')
                ax4.set_ylabel('Gap Length (frames)')
                ax4.set_title(f'Detection Gaps ({len(gap_lengths)} total gaps)')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No gaps detected', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Detection Gaps')
            
            plt.tight_layout()
            plt.show()
    
    # Return results
    all_jumps_combined = consecutive_jumps + gap_jumps
    
    return {
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'consecutive_distances': consecutive_distances,
        'gap_distances': gap_distances,
        'consecutive_jumps': consecutive_jumps,
        'gap_jumps': gap_jumps,
        'all_jumps': all_jumps_combined,
        'all_transitions': all_transitions,
        'centroids': centroids,
        'frames_dropped': len(frames_to_drop) if drop_jumps else 0,
        'dropped_frames': sorted(list(frames_to_drop)) if drop_jumps else [],
        'cleaned_data': {
            'bboxes': cleaned_bboxes,
            'scores': cleaned_scores,
            'n_detections': cleaned_n_detections
        } if drop_jumps else None,
        'statistics': {
            'mean_distance': float(np.mean(consecutive_distances)) if len(consecutive_distances) > 0 else 0,
            'median_distance': float(np.median(consecutive_distances)) if len(consecutive_distances) > 0 else 0,
            'std_distance': float(np.std(consecutive_distances)) if len(consecutive_distances) > 0 else 0,
            'max_distance': float(np.max(consecutive_distances)) if len(consecutive_distances) > 0 else 0,
            'consecutive_jump_count': len(consecutive_jumps),
            'gap_jump_count': len(gap_jumps),
            'total_jump_count': len(all_jumps_combined),
            'jump_rate': len(consecutive_jumps)/len(consecutive_distances) if len(consecutive_distances) > 0 else 0
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze frame-to-frame distances in YOLO detections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  %(prog)s detections.zarr
  
  # With custom threshold
  %(prog)s detections.zarr --threshold 50
  
  # With visualization
  %(prog)s detections.zarr --visualize
  
  # Drop detections that are jumps (preview)
  %(prog)s detections.zarr --threshold 75 --drop
  
  # Drop jumps and save cleaned data
  %(prog)s detections.zarr --threshold 75 --drop --save
  
  # Suggested for fish tracking
  %(prog)s detections.zarr --threshold 75 --visualize --drop
        """
    )
    
    parser.add_argument('zarr_path', help='Path to YOLO detection zarr file')
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=100.0,
        help='Distance threshold for flagging jumps (pixels, default: 100)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization plots'
    )
    
    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop detections that are jumps (remove outliers)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save cleaned data to zarr (requires --drop)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.save and not args.drop:
        parser.error("--save requires --drop")
    
    # Run analysis
    results = analyze_frame_distances(
        zarr_path=args.zarr_path,
        threshold=args.threshold,
        visualize=args.visualize,
        drop_jumps=args.drop,
        save_cleaned=args.save
    )
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    
    # Suggest threshold if many jumps detected
    if results['statistics']['jump_rate'] > 0.1:  # More than 10% jumps
        suggested_threshold = np.percentile(results['consecutive_distances'], 95)
        print(f"\n💡 Suggestion: You have a {results['statistics']['jump_rate']*100:.1f}% jump rate.")
        print(f"   Consider using --threshold {suggested_threshold:.1f} (95th percentile)")
    
    return 0


if __name__ == '__main__':
    exit(main())