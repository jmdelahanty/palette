#!/usr/bin/env python3
"""
ROI Position Debugger

Diagnoses position tracking issues for specific ROIs.
Helps understand why interpolation positions are incorrect.
"""

import zarr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import argparse

console = Console()


def analyze_roi_positions(zarr_path: str, roi_id: int, sample_frames: int = 10):
    """
    Analyze position data for a specific ROI to diagnose tracking issues.
    """
    console.print(f"\n[bold cyan]ROI {roi_id} Position Analysis[/bold cyan]")
    console.print("="*60)
    
    # Open zarr
    root = zarr.open_group(zarr_path, mode='r')
    
    # Load detection data
    detect_group = root['detect_runs']
    latest_detect = detect_group.attrs['latest']
    n_detections = detect_group[latest_detect]['n_detections'][:]
    bbox_coords = detect_group[latest_detect]['bbox_norm_coords'][:]
    
    # Load ID assignments
    id_key = 'id_assignments_runs' if 'id_assignments_runs' in root else 'id_assignments'
    id_group = root[id_key]
    latest_id = id_group.attrs['latest']
    detection_ids = id_group[latest_id]['detection_ids'][:]
    n_detections_per_roi = id_group[latest_id]['n_detections_per_roi'][:]
    
    # Get video dimensions
    img_width = root.attrs.get('width', 4512)
    img_height = root.attrs.get('height', 4512)
    
    console.print(f"Video dimensions: {img_width}x{img_height}")
    console.print(f"Total frames: {len(n_detections)}")
    console.print(f"Total detections: {len(detection_ids)}")
    
    # Analyze ROI coverage
    roi_detections = n_detections_per_roi[:, roi_id]
    frames_with_detection = np.where(roi_detections > 0)[0]
    coverage = len(frames_with_detection) / len(roi_detections) * 100
    
    console.print(f"\n[bold]ROI {roi_id} Statistics:[/bold]")
    console.print(f"  Coverage: {coverage:.1f}% ({len(frames_with_detection)}/{len(roi_detections)} frames)")
    console.print(f"  Total detections: {np.sum(roi_detections)}")
    
    # Find sample frames with detections
    if len(frames_with_detection) == 0:
        console.print(f"[red]No detections found for ROI {roi_id}![/red]")
        return
    
    # Sample some frames
    sample_indices = np.linspace(0, len(frames_with_detection)-1, 
                                min(sample_frames, len(frames_with_detection)), 
                                dtype=int)
    sample_frame_numbers = frames_with_detection[sample_indices]
    
    console.print(f"\n[bold]Sample Frame Analysis:[/bold]")
    
    # Create table for sample positions
    table = Table(title=f"ROI {roi_id} Sample Positions")
    table.add_column("Frame", style="cyan")
    table.add_column("Center X", style="yellow")
    table.add_column("Center Y", style="yellow")
    table.add_column("Width", style="green")
    table.add_column("Height", style="green")
    table.add_column("Det Index", style="magenta")
    
    positions = []
    cumulative_idx = 0
    
    for frame_idx in range(len(n_detections)):
        frame_det_count = int(n_detections[frame_idx])
        
        if frame_det_count > 0 and roi_detections[frame_idx] > 0:
            # Find detection for this ROI
            frame_detection_ids = detection_ids[cumulative_idx:cumulative_idx + frame_det_count]
            roi_mask = frame_detection_ids == roi_id
            
            if np.any(roi_mask):
                roi_indices = np.where(roi_mask)[0]
                for roi_idx in roi_indices:
                    global_idx = cumulative_idx + roi_idx
                    bbox = bbox_coords[global_idx]
                    
                    # Convert to pixel coordinates
                    center_x_px = bbox[0] * img_width
                    center_y_px = bbox[1] * img_height
                    width_px = bbox[2] * img_width
                    height_px = bbox[3] * img_height
                    
                    positions.append({
                        'frame': frame_idx,
                        'center_x': center_x_px,
                        'center_y': center_y_px,
                        'width': width_px,
                        'height': height_px,
                        'global_idx': global_idx
                    })
                    
                    # Add to table if it's a sample frame
                    if frame_idx in sample_frame_numbers:
                        table.add_row(
                            str(frame_idx),
                            f"{center_x_px:.1f}",
                            f"{center_y_px:.1f}",
                            f"{width_px:.1f}",
                            f"{height_px:.1f}",
                            str(global_idx)
                        )
        
        cumulative_idx += frame_det_count
    
    console.print(table)
    
    # Analyze position distribution
    if positions:
        pos_array = np.array([[p['center_x'], p['center_y']] for p in positions])
        
        console.print(f"\n[bold]Position Statistics:[/bold]")
        console.print(f"  X range: {pos_array[:, 0].min():.1f} - {pos_array[:, 0].max():.1f}")
        console.print(f"  Y range: {pos_array[:, 1].min():.1f} - {pos_array[:, 1].max():.1f}")
        console.print(f"  Mean position: ({pos_array[:, 0].mean():.1f}, {pos_array[:, 1].mean():.1f})")
        console.print(f"  Std deviation: ({pos_array[:, 0].std():.1f}, {pos_array[:, 1].std():.1f})")
        
        # Check for position jumps
        if len(positions) > 1:
            frames = np.array([p['frame'] for p in positions])
            frame_diffs = np.diff(frames)
            pos_diffs = np.diff(pos_array, axis=0)
            distances = np.sqrt(pos_diffs[:, 0]**2 + pos_diffs[:, 1]**2)
            
            # Calculate speed (pixels per frame)
            speeds = distances / frame_diffs
            
            console.print(f"\n[bold]Movement Analysis:[/bold]")
            console.print(f"  Max jump: {distances.max():.1f} pixels")
            console.print(f"  Mean speed: {speeds.mean():.1f} pixels/frame")
            console.print(f"  Max speed: {speeds.max():.1f} pixels/frame")
            
            # Find largest gaps
            gap_indices = np.where(frame_diffs > 1)[0]
            if len(gap_indices) > 0:
                console.print(f"\n[bold]Gaps in Detection:[/bold]")
                largest_gaps = sorted(zip(gap_indices, frame_diffs[gap_indices]), 
                                    key=lambda x: x[1], reverse=True)[:5]
                for gap_idx, gap_size in largest_gaps:
                    frame_before = frames[gap_idx]
                    frame_after = frames[gap_idx + 1]
                    pos_before = pos_array[gap_idx]
                    pos_after = pos_array[gap_idx + 1]
                    distance = distances[gap_idx]
                    
                    console.print(f"  Frames {frame_before}-{frame_after} (gap: {gap_size} frames)")
                    console.print(f"    Position jump: {distance:.1f} pixels")
                    console.print(f"    From: ({pos_before[0]:.1f}, {pos_before[1]:.1f})")
                    console.print(f"    To: ({pos_after[0]:.1f}, {pos_after[1]:.1f})")
        
        return positions
    
    return []


def visualize_roi_trajectory(zarr_path: str, roi_id: int, max_frames: int = 5000):
    """
    Visualize the trajectory of a specific ROI.
    """
    positions = analyze_roi_positions(zarr_path, roi_id, sample_frames=20)
    
    if not positions:
        console.print("[red]No positions to visualize[/red]")
        return
    
    # Limit to max_frames for visualization
    positions = positions[:max_frames]
    
    # Extract data
    frames = np.array([p['frame'] for p in positions])
    x_coords = np.array([p['center_x'] for p in positions])
    y_coords = np.array([p['center_y'] for p in positions])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Trajectory plot
    ax = axes[0, 0]
    scatter = ax.scatter(x_coords, y_coords, c=frames, cmap='viridis', 
                        s=2, alpha=0.6)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'ROI {roi_id} Trajectory (colored by time)')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Frame')
    
    # Add start and end markers
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    ax.legend()
    
    # 2. X position over time
    ax = axes[0, 1]
    ax.plot(frames, x_coords, 'b-', alpha=0.6, linewidth=0.5)
    ax.scatter(frames[::max(1, len(frames)//100)], 
              x_coords[::max(1, len(frames)//100)], 
              c='blue', s=10, alpha=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('X Position (pixels)')
    ax.set_title('X Position Over Time')
    ax.grid(True, alpha=0.3)
    
    # 3. Y position over time
    ax = axes[1, 0]
    ax.plot(frames, y_coords, 'r-', alpha=0.6, linewidth=0.5)
    ax.scatter(frames[::max(1, len(frames)//100)], 
              y_coords[::max(1, len(frames)//100)], 
              c='red', s=10, alpha=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Y Position Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. Gap analysis
    ax = axes[1, 1]
    frame_diffs = np.diff(frames)
    gap_frames = frames[:-1][frame_diffs > 1]
    gap_sizes = frame_diffs[frame_diffs > 1]
    
    if len(gap_sizes) > 0:
        ax.scatter(gap_frames, gap_sizes, alpha=0.6)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Gap Size (frames)')
        ax.set_title(f'Detection Gaps (Total: {len(gap_sizes)})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Max gap: {gap_sizes.max()} frames\n"
        stats_text += f"Mean gap: {gap_sizes.mean():.1f} frames\n"
        stats_text += f"Gaps >50 frames: {np.sum(gap_sizes > 50)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No gaps detected', ha='center', va='center')
        ax.set_title('Detection Gaps')
    
    plt.suptitle(f'ROI {roi_id} Position Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Debug ROI position tracking issues',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    parser.add_argument('--roi', type=int, default=3,
                       help='ROI ID to analyze (default: 3)')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of sample frames to display (default: 20)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show trajectory visualization')
    parser.add_argument('--max-frames', type=int, default=5000,
                       help='Maximum frames to visualize (default: 5000)')
    
    args = parser.parse_args()
    
    # Analyze positions
    positions = analyze_roi_positions(args.zarr_path, args.roi, args.samples)
    
    # Visualize if requested
    if args.visualize and positions:
        visualize_roi_trajectory(args.zarr_path, args.roi, args.max_frames)


if __name__ == "__main__":
    main()