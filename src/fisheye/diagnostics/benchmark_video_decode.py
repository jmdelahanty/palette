#!/usr/bin/env python3
"""
Benchmark different video decoding methods for YOLO inference.

Compares:
- OpenCV (cv2) CPU decoding
- Decord CPU decoding
- Decord GPU decoding
- Decord GPU + batch loading
"""

import os
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.markup import escape

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

console = Console()

# Cache Decord import status so we only attempt it once.
try:
    import decord  # type: ignore
    from decord import VideoReader, cpu, gpu  # type: ignore
    _decord_import_error = None
except Exception as exc:  # pragma: no cover - depends on local environment
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    gpu = None  # type: ignore
    _decord_import_error = exc


def _decord_unavailable_message(prefix: str) -> str:
    """Format a Decord warning message."""
    prefix = escape(prefix)
    if _decord_import_error:
        return f"{prefix}: {escape(str(_decord_import_error))}"
    return f"{prefix}"


def benchmark_opencv(video_path: str, n_frames: int = 1000, resize: tuple = None):
    """Benchmark OpenCV video decoding."""
    import cv2
    console.print("\n[bold cyan]OpenCV (CPU)[/bold cyan]")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, total_frames)
    
    frame_count = 0
    first_shape = None
    start = time.time()
    
    with Progress(TextColumn("[cyan]Decoding"), BarColumn(), TextColumn("{task.completed}/{task.total}"), console=console) as progress:
        task = progress.add_task("frames", total=n_frames)
        
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if resize:
                frame = cv2.resize(frame, resize)
            
            # Convert BGR to RGB (what YOLO needs)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if first_shape is None:
                first_shape = frame_rgb.shape
            frame_count += 1
            progress.update(task, advance=1)
    
    cap.release()
    duration = time.time() - start
    
    return {
        'method': 'OpenCV (CPU)',
        'frames': frame_count,
        'duration': duration,
        'fps': frame_count / duration if duration > 0 else 0.0,
        'frame_shape': first_shape
    }


def benchmark_decord_cpu(video_path: str, n_frames: int = 1000, resize: tuple = None):
    """Benchmark Decord CPU decoding."""
    if decord is None or VideoReader is None or cpu is None:
        console.print(f"[yellow]{_decord_unavailable_message('Decord not available (skipping)')}[/yellow]")
        return None
    
    console.print("\n[bold cyan]Decord (CPU)[/bold cyan]")
    
    vr: Optional[VideoReader] = None
    try:
        decord.bridge.set_bridge('native')
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        n_frames = min(n_frames, total_frames)
        
        frame_count = 0
        first_shape = None
        start = time.time()
        
        with Progress(TextColumn("[cyan]Decoding"), BarColumn(), TextColumn("{task.completed}/{task.total}"), console=console) as progress:
            task = progress.add_task("frames", total=n_frames)
            
            for i in range(n_frames):
                frame = vr[i]
                if hasattr(frame, "asnumpy"):  # Fallback if bridge did not return ndarray
                    frame = frame.asnumpy()
                else:
                    frame = np.asarray(frame)
                
                if resize:
                    import cv2
                    frame = cv2.resize(frame, resize)
                
                if first_shape is None:
                    first_shape = frame.shape
                frame_count += 1
                progress.update(task, advance=1)
        
        duration = time.time() - start
        
        return {
            'method': 'Decord (CPU)',
            'frames': frame_count,
            'duration': duration,
            'fps': frame_count / duration if duration > 0 else 0.0,
            'frame_shape': first_shape
        }
    except Exception as e:
        message = str(e)
        console.print(f"[yellow]Decord CPU test failed: {escape(message)}[/yellow]")
        if "decord_eof_retry_max" in message.lower():
            console.print("[yellow]Tip: try increasing the DECORD_EOF_RETRY_MAX environment variable if the video has slow-to-read trailing frames.[/yellow]")
        return None
    finally:
        del vr


def benchmark_decord_gpu(video_path: str, n_frames: int = 1000, resize: tuple = None):
    """Benchmark Decord GPU decoding (single frame at a time)."""
    if decord is None or VideoReader is None or gpu is None:
        console.print(f"[yellow]{_decord_unavailable_message('Decord GPU not available (skipping)')}[/yellow]")
        return None
    
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, skipping GPU test[/yellow]")
        return None
    
    console.print("\n[bold cyan]Decord (GPU)[/bold cyan]")
    
    vr: Optional[VideoReader] = None
    try:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(video_path, ctx=gpu(0))
        total_frames = len(vr)
        n_frames = min(n_frames, total_frames)
        
        frame_count = 0
        first_shape = None
        start = time.time()
        
        with Progress(TextColumn("[cyan]Decoding"), BarColumn(), TextColumn("{task.completed}/{task.total}"), console=console) as progress:
            task = progress.add_task("frames", total=n_frames)
            
            for i in range(n_frames):
                frame = vr[i]  # Already on GPU as torch tensor
                
                if resize:
                    import torch.nn.functional as F
                    # Add batch and channel dims for resize
                    frame = frame.permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]
                    frame = F.interpolate(frame, size=resize, mode='bilinear', align_corners=False)
                    frame = frame.squeeze(0).permute(1, 2, 0).to(torch.uint8)  # Back to [H, W, C]
                
                if first_shape is None:
                    first_shape = tuple(frame.shape)
                frame_count += 1
                progress.update(task, advance=1)
        
        duration = time.time() - start
        
        return {
            'method': 'Decord (GPU, single)',
            'frames': frame_count,
            'duration': duration,
            'fps': frame_count / duration if duration > 0 else 0.0,
            'frame_shape': first_shape
        }
    except Exception as e:
        message = str(e)
        console.print(f"[yellow]Decord GPU test failed: {escape(message)}[/yellow]")
        if "DECORD_EOF_RETRY_MAX" in message:
            console.print("[yellow]Tip: increase DECORD_EOF_RETRY_MAX to allow more retries near EOF.[/yellow]")
        return None
    finally:
        del vr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def benchmark_decord_gpu_batch(video_path: str, n_frames: int = 1000, batch_size: int = 32, resize: tuple = None) -> Tuple[Optional[dict], bool]:
    """Benchmark Decord GPU batch decoding.

    Returns (result_dict, oom_flag).
    """
    if decord is None or VideoReader is None or gpu is None:
        console.print(f"[yellow]{_decord_unavailable_message('Decord GPU batch not available (skipping)')}[/yellow]")
        return None, False
    
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, skipping GPU test[/yellow]")
        return None, False
    
    console.print(f"\n[bold cyan]Decord (GPU, batch={batch_size})[/bold cyan]")
    
    vr: Optional[VideoReader] = None
    try:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(video_path, ctx=gpu(0))
        total_frames = len(vr)
        n_frames = min(n_frames, total_frames)
        
        frame_count = 0
        first_shape = None
        start = time.time()
        
        with Progress(TextColumn("[cyan]Decoding"), BarColumn(), TextColumn("{task.completed}/{task.total}"), console=console) as progress:
            task = progress.add_task("frames", total=n_frames)
            
            for batch_start in range(0, n_frames, batch_size):
                batch_end = min(batch_start + batch_size, n_frames)
                indices = list(range(batch_start, batch_end))
                
                # Batch decode - MUCH faster!
                batch = vr.get_batch(indices)  # [B, H, W, C] on GPU
                
                if resize:
                    import torch.nn.functional as F
                    # Batch resize on GPU
                    batch = batch.permute(0, 3, 1, 2).float()  # [B, C, H, W]
                    batch = F.interpolate(batch, size=resize, mode='bilinear', align_corners=False)
                    batch = batch.permute(0, 2, 3, 1).to(torch.uint8)  # Back to [B, H, W, C]
                
                if first_shape is None:
                    first_shape = tuple(batch[0].shape)
                frame_count += batch.shape[0]
                progress.update(task, advance=len(indices))
                del batch
        
        duration = time.time() - start
        
        return {
            'method': f'Decord (GPU, batch={batch_size})',
            'frames': frame_count,
            'duration': duration,
            'fps': frame_count / duration if duration > 0 else 0.0,
            'frame_shape': first_shape
        }, False
    except Exception as e:
        raw_message = str(e)
        console.print(f"[yellow]Decord GPU batch test failed: {escape(raw_message)}[/yellow]")
        message = raw_message.lower()
        if "decord_eof_retry_max" in message:
            console.print("[yellow]Tip: increase DECORD_EOF_RETRY_MAX to allow more retries near EOF.[/yellow]")
        is_oom = "out of memory" in message or "cudaerrormemoryallocation" in message
        return None, is_oom
    finally:
        del vr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark video decoding methods")
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--frames', type=int, default=1000, help='Number of frames to test (default: 1000)')
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='Resize to [width, height] (e.g., --resize 640 640)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 64, 128], help='Batch sizes to test for GPU (default: 32 64 128)')
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/red]")
        return
    
    resize = tuple(args.resize) if args.resize else None
    
    console.print(Panel(
        f"[cyan]Video:[/cyan] {video_path.name}\n"
        f"[cyan]Test frames:[/cyan] {args.frames}\n"
        f"[cyan]Resize:[/cyan] {resize if resize else 'None (original size)'}",
        title="[bold]Video Decode Benchmark[/bold]"
    ))
    
    # Run benchmarks
    results = []
    
    # OpenCV
    result = benchmark_opencv(str(video_path), args.frames, resize)
    if result:
        results.append(result)
    
    # Decord-based benchmarks
    if decord is None or VideoReader is None or cpu is None:
        console.print(f"[yellow]{_decord_unavailable_message('Skipping Decord benchmarks')}[/yellow]")
    else:
        result = benchmark_decord_cpu(str(video_path), args.frames, resize)
        if result:
            results.append(result)
        
        result = benchmark_decord_gpu(str(video_path), args.frames, resize)
        if result:
            results.append(result)
        
        oom_encountered = False
        for batch_size in args.batch_sizes:
            res, hit_oom = benchmark_decord_gpu_batch(str(video_path), args.frames, batch_size, resize)
            if res:
                results.append(res)
            if hit_oom:
                console.print("[yellow]Skipping remaining batch sizes due to GPU OOM[/yellow]")
                oom_encountered = True
                break
    
    # Display results
    console.print("\n")
    table = Table(title="📊 Benchmark Results", show_header=True)
    table.add_column("Method", style="cyan")
    table.add_column("Frames", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="yellow")
    
    baseline_fps = results[0]['fps'] if results else 1.0
    
    for r in results:
        speedup = r['fps'] / baseline_fps
        table.add_row(
            r['method'],
            str(r['frames']),
            f"{r['duration']:.2f}",
            f"{r['fps']:.1f}",
            f"{speedup:.2f}×"
        )
    
    console.print(table)
    
    # Show best method
    if results:
        best = max(results, key=lambda x: x['fps'])
        console.print(f"\n[bold green]🏆 Winner: {best['method']} at {best['fps']:.1f} fps[/bold green]")
        console.print(f"[bold green]   {best['fps']/baseline_fps:.1f}× faster than OpenCV![/bold green]")


if __name__ == '__main__':
    main()
