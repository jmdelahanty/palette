#!/usr/bin/env python3
"""
Diagnostic script to investigate crop hang issue.

This will check:
1. Total frames in zarr metadata vs actual video
2. Maximum frame index in detection data
3. Whether the last batch requests frames beyond video length
4. Detection distribution in the problematic batch
"""

import sys
import zarr
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

def diagnose_crop_hang(zarr_path: str, video_path: str = None):
    """Diagnose why cropping hangs on the last batch."""
    console = Console()
    console.rule("[bold red]Crop Hang Diagnostics[/bold red]")
    
    # Open zarr
    root = zarr.open(zarr_path, mode='r')
    
    # Get metadata
    console.print("\n[bold]1. Checking Metadata[/bold]")
    metadata_table = Table(show_header=False)
    metadata_table.add_column("Property", style="cyan")
    metadata_table.add_column("Value", style="yellow")
    
    total_frames_meta = root.attrs.get('total_frames', None)
    video_width = root.attrs.get('width', None)
    video_height = root.attrs.get('height', None)
    source_video_path = root.attrs.get('source_video_path', None)
    
    metadata_table.add_row("total_frames (metadata)", str(total_frames_meta))
    metadata_table.add_row("Video dimensions", f"{video_width}x{video_height}")
    metadata_table.add_row("Source video path", str(source_video_path))
    console.print(metadata_table)
    
    # Check video if path provided
    if video_path or source_video_path:
        console.print("\n[bold]2. Checking Video File[/bold]")
        actual_video_path = video_path or source_video_path
        
        try:
            import cv2
            cap = cv2.VideoCapture(str(actual_video_path))
            if cap.isOpened():
                actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                video_table = Table(show_header=False)
                video_table.add_column("Property", style="cyan")
                video_table.add_column("Value", style="yellow")
                video_table.add_row("Actual video frames", str(actual_frame_count))
                video_table.add_row("Metadata total_frames", str(total_frames_meta))
                
                if actual_frame_count != total_frames_meta:
                    video_table.add_row("⚠️  MISMATCH", f"Difference: {actual_frame_count - total_frames_meta}", style="bold red")
                else:
                    video_table.add_row("✓ Match", "OK", style="bold green")
                
                console.print(video_table)
            else:
                console.print(f"[red]Could not open video: {actual_video_path}[/red]")
        except Exception as e:
            console.print(f"[yellow]Could not check video: {e}[/yellow]")
    else:
        console.print("\n[yellow]No video path provided, skipping video check[/yellow]")
        console.print("[dim]Use: python diagnose_crop_hang.py /path/to/data.zarr --video /path/to/video.mp4[/dim]")
    
    # Get detection source
    console.print("\n[bold]3. Checking Detection Data[/bold]")
    
    # Find latest detect run
    if 'detect_runs' in root:
        latest_run = root['detect_runs'].attrs.get('latest')
        detect_group = root[f'detect_runs/{latest_run}']
        console.print(f"Using detect run: [cyan]{latest_run}[/cyan]")
    else:
        console.print("[red]No detect_runs found![/red]")
        return
    
    # Load detection arrays
    frame_indices = detect_group['frame_indices'][:]
    bbox_coords = detect_group['bbox_norm_coords'][:]
    total_detections = len(frame_indices)
    
    detection_table = Table(show_header=False)
    detection_table.add_column("Property", style="cyan")
    detection_table.add_column("Value", style="yellow")
    
    detection_table.add_row("Total detections", f"{total_detections:,}")
    detection_table.add_row("Min frame index", str(frame_indices.min()))
    detection_table.add_row("Max frame index", str(frame_indices.max()))
    detection_table.add_row("Frame range span", str(frame_indices.max() - frame_indices.min() + 1))
    
    # Check if max frame exceeds video length
    max_frame_idx = frame_indices.max()
    if total_frames_meta is not None and max_frame_idx >= total_frames_meta:
        detection_table.add_row(
            "⚠️  PROBLEM FOUND",
            f"Max frame index ({max_frame_idx}) >= total_frames ({total_frames_meta})",
            style="bold red"
        )
    else:
        detection_table.add_row("✓ Frame indices OK", "All within range", style="bold green")
    
    console.print(detection_table)
    
    # Simulate batching
    console.print("\n[bold]4. Simulating Batch Creation[/bold]")
    max_frames_per_batch = 96  # GPU default
    
    # Sort by frame
    sorted_idx = np.argsort(frame_indices)
    
    batches = []
    current_batch = []
    unique_frames_in_batch = set()
    
    for det_idx in sorted_idx:
        frame = frame_indices[det_idx]
        
        if len(unique_frames_in_batch) >= max_frames_per_batch:
            batches.append(np.array(current_batch))
            current_batch = []
            unique_frames_in_batch = set()
        
        current_batch.append(det_idx)
        unique_frames_in_batch.add(frame)
    
    if current_batch:
        batches.append(np.array(current_batch))
    
    console.print(f"Total batches: {len(batches)}")
    
    # Analyze last few batches
    console.print("\n[bold]5. Last 3 Batches Analysis[/bold]")
    
    for i in range(max(0, len(batches) - 3), len(batches)):
        det_indices = batches[i]
        batch_frames = frame_indices[det_indices]
        unique_frames = np.unique(batch_frames)
        
        batch_table = Table(title=f"Batch {i+1}/{len(batches)}")
        batch_table.add_column("Property", style="cyan")
        batch_table.add_column("Value", style="yellow")
        
        batch_table.add_row("Detection count", str(len(det_indices)))
        batch_table.add_row("Unique frames", str(len(unique_frames)))
        batch_table.add_row("Frame range", f"{unique_frames.min()} - {unique_frames.max()}")
        batch_table.add_row("First frame", str(unique_frames[0]))
        batch_table.add_row("Last frame", str(unique_frames[-1]))
        
        # Check if this batch requests frames beyond video
        if total_frames_meta is not None and unique_frames.max() >= total_frames_meta:
            batch_table.add_row(
                "⚠️  PROBLEM",
                f"Requests frame {unique_frames.max()} but video only has {total_frames_meta} frames (0-{total_frames_meta-1})",
                style="bold red"
            )
        
        console.print(batch_table)
        
        # Show frame distribution for last batch
        if i == len(batches) - 1:
            console.print("\n[dim]Frame-by-frame breakdown of LAST batch:[/dim]")
            frame_counts = {}
            for frame in batch_frames:
                frame_counts[int(frame)] = frame_counts.get(int(frame), 0) + 1
            
            for frame, count in sorted(frame_counts.items())[:10]:  # Show first 10
                console.print(f"  Frame {frame}: {count} detections")
            if len(frame_counts) > 10:
                console.print(f"  ... and {len(frame_counts) - 10} more frames")
    
    # Summary
    console.print("\n[bold]Summary[/bold]")
    issues_found = []
    
    if total_frames_meta is not None and max_frame_idx >= total_frames_meta:
        issues_found.append(f"Max frame index ({max_frame_idx}) exceeds video length ({total_frames_meta})")
    
    if video_path or source_video_path:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path or source_video_path))
            actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if actual_frame_count != total_frames_meta:
                issues_found.append(f"Metadata mismatch: metadata says {total_frames_meta} frames, video has {actual_frame_count}")
        except:
            pass
    
    if issues_found:
        console.print("[bold red]Issues Found:[/bold red]")
        for issue in issues_found:
            console.print(f"  • {issue}")
        
        console.print("\n[bold yellow]Recommended Fix:[/bold yellow]")
        console.print("  1. Check if detection data includes frames beyond video length")
        console.print("  2. Filter frame_indices to only include valid frames (0 to total_frames-1)")
        console.print("  3. Verify video file is complete and not corrupted")
        console.print("  4. Update metadata to match actual video length")
    else:
        console.print("[bold green]✓ No obvious issues found[/bold green]")
        console.print("The hang might be due to:")
        console.print("  • GPU memory issues")
        console.print("  • Zarr write contention")
        console.print("  • Decord internal issues")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose crop hang issue")
    parser.add_argument("zarr_path", help="Path to zarr file")
    parser.add_argument("--video", help="Path to video file (optional)")
    
    args = parser.parse_args()
    
    diagnose_crop_hang(args.zarr_path, args.video)