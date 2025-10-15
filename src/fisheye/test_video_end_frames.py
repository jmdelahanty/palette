#!/usr/bin/env python3
"""
Test if video file can decode frames near the end.
Tests both GPU (decord) and CPU (OpenCV) decoders.
"""

import sys
import time
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

def test_video_end_frames(video_path: str, test_gpu: bool = True, test_cpu: bool = True):
    """Test decoding frames near the end of the video."""
    console = Console()
    console.rule("[bold]Video End Frame Decode Test[/bold]")
    
    video_path = Path(video_path)
    if not video_path.exists():
        console.print(f"[red]Video file not found: {video_path}[/red]")
        return
    
    # Get video info with OpenCV first
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[red]Failed to open video with OpenCV[/red]")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    console.print(f"[cyan]Video Info:[/cyan]")
    console.print(f"  Path: {video_path}")
    console.print(f"  Frames: {total_frames}")
    console.print(f"  Resolution: {width}x{height}")
    console.print(f"  FPS: {fps}")
    
    # Define test frames - focus on the end
    test_frames = [
        0,                          # First frame
        total_frames // 2,          # Middle frame
        total_frames - 100,         # 100 from end
        total_frames - 50,          # 50 from end
        total_frames - 20,          # 20 from end
        total_frames - 10,          # 10 from end
        total_frames - 5,           # 5 from end
        total_frames - 2,           # 2nd to last
        total_frames - 1,           # Last frame (0-indexed)
    ]
    
    # Also test the exact batch that's failing: frames 45690-45752
    if total_frames >= 45753:
        test_frames.extend([45690, 45720, 45752])
    
    test_frames = sorted(set(test_frames))
    
    console.print(f"\n[bold]Testing {len(test_frames)} individual frames[/bold]")
    console.print(f"[dim]Frames to test: {test_frames}[/dim]")
    
    # Test GPU decoding with decord
    if test_gpu:
        console.print("\n[bold cyan]GPU Decoding Test (Decord)[/bold cyan]")
        try:
            import torch
            import decord
            from decord import VideoReader, gpu
            
            if not torch.cuda.is_available():
                console.print("[yellow]CUDA not available, skipping GPU test[/yellow]")
            else:
                console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")
                
                # Initialize GPU reader with timeout
                init_result = {'reader': None, 'error': None}
                
                def init_reader():
                    try:
                        console.print("[dim]Setting decord bridge to torch...[/dim]")
                        decord.bridge.set_bridge('torch')
                        console.print("[dim]Creating VideoReader with GPU context...[/dim]")
                        init_result['reader'] = VideoReader(str(video_path), ctx=gpu(0))
                    except Exception as e:
                        init_result['error'] = e
                
                thread = threading.Thread(target=init_reader, daemon=True)
                start = time.perf_counter()
                thread.start()
                thread.join(timeout=5.0)  # 5 second timeout for initialization
                
                if thread.is_alive():
                    elapsed = (time.perf_counter() - start) * 1000
                    console.print(f"[red]VideoReader initialization HUNG after {elapsed:.0f}ms[/red]")
                    console.print("[yellow]Cannot initialize GPU decoder - skipping GPU tests[/yellow]")
                    console.print("[yellow]This might be a CUDA/driver issue or video codec incompatibility[/yellow]")
                    return  # Skip GPU tests
                
                if init_result['error']:
                    console.print(f"[red]VideoReader initialization failed: {init_result['error']}[/red]")
                    return
                
                video_reader = init_result['reader']
                console.print(f"[green]✓ VideoReader initialized: {len(video_reader)} frames[/green]\n")
                
                gpu_results = Table()
                gpu_results.add_column("Frame", style="cyan")
                gpu_results.add_column("Status", style="yellow")
                gpu_results.add_column("Time (ms)", style="green")
                gpu_results.add_column("Notes", style="dim")
                
                import threading
                
                for frame_idx in test_frames:
                    if frame_idx >= len(video_reader):
                        gpu_results.add_row(
                            str(frame_idx),
                            "❌ SKIP",
                            "-",
                            f"Beyond video length ({len(video_reader)})"
                        )
                        continue
                    
                    # Use a timeout to detect hangs
                    result = {'frame': None, 'error': None, 'elapsed': None}
                    
                    def decode_frame():
                        try:
                            start = time.perf_counter()
                            result['frame'] = video_reader[frame_idx]
                            result['elapsed'] = (time.perf_counter() - start) * 1000
                        except Exception as e:
                            result['error'] = e
                    
                    console.print(f"[dim]Decoding frame {frame_idx}...[/dim]", end="")
                    thread = threading.Thread(target=decode_frame, daemon=True)
                    start = time.perf_counter()
                    thread.start()
                    thread.join(timeout=3.0)  # 3 second timeout
                    
                    if thread.is_alive():
                        # Thread still running = hung
                        elapsed = (time.perf_counter() - start) * 1000
                        console.print(f" [red]HUNG after {elapsed:.0f}ms[/red]")
                        gpu_results.add_row(
                            str(frame_idx),
                            "⏱️ TIMEOUT",
                            f">{elapsed:.0f}",
                            "Decode hung (>3s timeout)"
                        )
                        console.print(f"[yellow]Frame {frame_idx} is hanging, skipping remaining GPU tests[/yellow]")
                        break
                    else:
                        console.print(" [green]done[/green]")
                        
                        if result['error']:
                            gpu_results.add_row(
                                str(frame_idx),
                                "❌ ERROR",
                                "-",
                                str(result['error'])[:50]
                            )
                        elif result['frame'] is None or result['frame'].numel() == 0:
                            gpu_results.add_row(
                                str(frame_idx),
                                "❌ FAIL",
                                f"{result['elapsed']:.1f}",
                                "Frame is empty/None"
                            )
                        else:
                            gpu_results.add_row(
                                str(frame_idx),
                                "✓ OK",
                                f"{result['elapsed']:.1f}",
                                f"shape={tuple(result['frame'].shape)}"
                            )
                
                console.print(gpu_results)
                
                # Now test batch reading the problematic range
                if total_frames >= 45753:
                    console.print("\n[bold]Testing batch decode of frames 45690-45752 (the hanging batch)[/bold]")
                    batch_frames = list(range(45690, 45753))  # 45690-45752 inclusive
                    
                    # Use timeout for batch decode too
                    batch_result = {'frames': None, 'error': None, 'elapsed': None}
                    
                    def decode_batch():
                        try:
                            start = time.perf_counter()
                            batch_result['frames'] = video_reader.get_batch(batch_frames)
                            batch_result['elapsed'] = (time.perf_counter() - start) * 1000
                        except Exception as e:
                            batch_result['error'] = e
                    
                    console.print(f"Requesting batch of {len(batch_frames)} frames (3s timeout)...")
                    thread = threading.Thread(target=decode_batch, daemon=True)
                    start = time.perf_counter()
                    thread.start()
                    thread.join(timeout=3.0)
                    
                    if thread.is_alive():
                        elapsed = (time.perf_counter() - start) * 1000
                        console.print(f"[red]❌ Batch decode HUNG after {elapsed:.0f}ms[/red]")
                        console.print(f"[yellow]This confirms the hang! Decord can't batch-decode frames near EOF.[/yellow]")
                    elif batch_result['error']:
                        console.print(f"[red]❌ Batch decode FAILED: {batch_result['error']}[/red]")
                        console.print(f"[yellow]This explains the hang![/yellow]")
                    elif batch_result['frames'] is not None:
                        console.print(f"[green]✓ Batch decode successful![/green]")
                        console.print(f"  Time: {batch_result['elapsed']:.1f} ms")
                        console.print(f"  Shape: {tuple(batch_result['frames'].shape)}")
                        console.print(f"  Avg per frame: {batch_result['elapsed']/len(batch_frames):.1f} ms")
                
        except ImportError as e:
            console.print(f"[yellow]Decord not available: {e}[/yellow]")
        except Exception as e:
            console.print(f"[red]GPU test failed: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    # Test CPU decoding with OpenCV
    if test_cpu:
        console.print("\n[bold cyan]CPU Decoding Test (OpenCV)[/bold cyan]")
        
        cpu_results = Table()
        cpu_results.add_column("Frame", style="cyan")
        cpu_results.add_column("Status", style="yellow")
        cpu_results.add_column("Time (ms)", style="green")
        cpu_results.add_column("Notes", style="dim")
        
        cap = cv2.VideoCapture(str(video_path))
        
        for frame_idx in test_frames:
            if frame_idx >= total_frames:
                cpu_results.add_row(
                    str(frame_idx),
                    "❌ SKIP",
                    "-",
                    f"Beyond video length ({total_frames})"
                )
                continue
            
            try:
                start = time.perf_counter()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                elapsed = (time.perf_counter() - start) * 1000
                
                if not ret or frame is None:
                    cpu_results.add_row(
                        str(frame_idx),
                        "❌ FAIL",
                        f"{elapsed:.1f}",
                        "Failed to read frame"
                    )
                else:
                    cpu_results.add_row(
                        str(frame_idx),
                        "✓ OK",
                        f"{elapsed:.1f}",
                        f"shape={frame.shape}"
                    )
            except Exception as e:
                cpu_results.add_row(
                    str(frame_idx),
                    "❌ ERROR",
                    "-",
                    str(e)[:50]
                )
        
        cap.release()
        console.print(cpu_results)
    
    console.print("\n[bold]Summary[/bold]")
    console.print("If GPU batch decode fails but individual frames succeed:")
    console.print("  → This is a known decord limitation with batch reading near EOF")
    console.print("  → Solution: Use CPU decoder for final batch, or decode frame-by-frame")
    console.print("\nIf CPU succeeds where GPU fails:")
    console.print("  → Confirms decord GPU decoder issue")
    console.print("  → Safe to fall back to CPU for problematic batches")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video end frame decoding")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU test")
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU test")
    
    args = parser.parse_args()
    
    test_video_end_frames(
        args.video_path,
        test_gpu=not args.no_gpu,
        test_cpu=not args.no_cpu
    )