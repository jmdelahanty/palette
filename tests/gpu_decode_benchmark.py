#!/usr/bin/env python3
"""
GPU Video Decoding Performance Benchmark
Tests various aspects of GPU-accelerated video decoding with Decord.
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import decord
from decord import VideoReader, gpu, cpu
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)


class GPUDecodeBenchmark:
    """Comprehensive GPU video decoding benchmark."""
    
    def __init__(self, video_path, device='gpu', console=None):
        self.video_path = Path(video_path)
        self.device = device
        self.console = console or Console()
        
        # Initialize video reader
        decord.bridge.set_bridge('torch')
        ctx = gpu(0) if device == 'gpu' else cpu(0)
        self.vr = VideoReader(str(video_path), ctx=ctx)
        
        # Video properties
        self.n_frames = len(self.vr)
        self.height = self.vr[0].shape[0]
        self.width = self.vr[0].shape[1]
        self.fps = self.vr.get_avg_fps()
        
        # Results storage
        self.results = {
            'video_info': {
                'path': str(self.video_path),
                'frames': self.n_frames,
                'resolution': f"{self.width}x{self.height}",
                'fps': self.fps,
                'duration_seconds': self.n_frames / self.fps
            },
            'device': 'GPU' if device == 'gpu' else 'CPU',
            'benchmarks': {}
        }
        
        self.console.print(Panel(
            f"[bold cyan]Video:[/bold cyan] {self.video_path.name}\n"
            f"[bold cyan]Frames:[/bold cyan] {self.n_frames}\n"
            f"[bold cyan]Resolution:[/bold cyan] {self.width}x{self.height}\n"
            f"[bold cyan]FPS:[/bold cyan] {self.fps:.2f}\n"
            f"[bold cyan]Device:[/bold cyan] {'GPU (CUDA)' if device == 'gpu' else 'CPU'}",
            title="Benchmark Configuration",
            box=box.ROUNDED
        ))
    
    def benchmark_sequential_decode(self, batch_sizes=[1, 4, 8, 16, 32, 64], max_frames=1000):
        """Benchmark sequential decoding with various batch sizes."""
        self.console.rule("[bold yellow]Sequential Decode Benchmark[/bold yellow]")
        
        results = {}
        frames_to_test = min(max_frames, self.n_frames)
        
        for batch_size in batch_sizes:
            self.console.print(f"\nTesting batch size: [cyan]{batch_size}[/cyan]")
            
            # Warmup
            _ = self.vr.get_batch([0])
            if self.device == 'gpu':
                torch.cuda.synchronize()
            
            decode_times = []
            memory_usage = []
            
            for i in tqdm(range(0, frames_to_test, batch_size), 
                         desc=f"Batch {batch_size}", 
                         disable=False):
                batch_end = min(i + batch_size, frames_to_test)
                indices = list(range(i, batch_end))
                
                # Track memory before
                if self.device == 'gpu':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated()
                
                # Time the decode
                start = time.perf_counter()
                frames = self.vr.get_batch(indices)
                if self.device == 'gpu':
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - start
                
                decode_times.append(decode_time)
                
                # Track memory after
                if self.device == 'gpu':
                    mem_after = torch.cuda.memory_allocated()
                    peak_mem = torch.cuda.max_memory_allocated()
                    memory_usage.append({
                        'allocated': mem_after - mem_before,
                        'peak': peak_mem
                    })
                
                # Cleanup
                del frames
            
            # Calculate statistics
            total_time = sum(decode_times)
            avg_fps = frames_to_test / total_time
            avg_time_per_batch = np.mean(decode_times)
            
            results[batch_size] = {
                'total_frames': frames_to_test,
                'total_time': total_time,
                'avg_fps': avg_fps,
                'avg_time_per_batch': avg_time_per_batch,
                'decode_times': decode_times,
                'memory_usage': memory_usage if self.device == 'gpu' else None
            }
            
            self.console.print(
                f"  → Average FPS: [green]{avg_fps:.1f}[/green]\n"
                f"  → Avg time/batch: [yellow]{avg_time_per_batch*1000:.2f}ms[/yellow]"
            )
        
        self.results['benchmarks']['sequential_decode'] = results
        return results
    
    def benchmark_random_access(self, n_samples=100):
        """Benchmark random frame access pattern."""
        self.console.rule("[bold yellow]Random Access Benchmark[/bold yellow]")
        
        # Generate random indices
        np.random.seed(42)
        random_indices = np.random.randint(0, self.n_frames, n_samples)
        
        # Warmup
        _ = self.vr[0]
        if self.device == 'gpu':
            torch.cuda.synchronize()
        
        access_times = []
        
        for idx in tqdm(random_indices, desc="Random access"):
            start = time.perf_counter()
            frame = self.vr[idx]
            if self.device == 'gpu':
                torch.cuda.synchronize()
            access_time = time.perf_counter() - start
            access_times.append(access_time)
            del frame
        
        results = {
            'n_samples': n_samples,
            'avg_access_time': np.mean(access_times),
            'min_access_time': np.min(access_times),
            'max_access_time': np.max(access_times),
            'std_access_time': np.std(access_times),
            'p50_access_time': np.percentile(access_times, 50),
            'p95_access_time': np.percentile(access_times, 95),
            'p99_access_time': np.percentile(access_times, 99)
        }
        
        self.console.print(
            f"  → Avg access time: [yellow]{results['avg_access_time']*1000:.2f}ms[/yellow]\n"
            f"  → P95 access time: [yellow]{results['p95_access_time']*1000:.2f}ms[/yellow]\n"
            f"  → P99 access time: [yellow]{results['p99_access_time']*1000:.2f}ms[/yellow]"
        )
        
        self.results['benchmarks']['random_access'] = results
        return results
    
    def benchmark_processing_pipeline(self, batch_size=32, max_frames=500):
        """Benchmark full processing pipeline (decode + grayscale + downsample)."""
        self.console.rule("[bold yellow]Processing Pipeline Benchmark[/bold yellow]")
        
        frames_to_test = min(max_frames, self.n_frames)
        device = 'cuda:0' if self.device == 'gpu' else 'cpu'
        
        # Setup processing
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
        target_size = (540, 960)  # Common downsampling target
        
        # Warmup
        _ = self.vr.get_batch([0])
        if self.device == 'gpu':
            torch.cuda.synchronize()
        
        stage_times = {
            'decode': [],
            'grayscale': [],
            'downsample': [],
            'transfer': []
        }
        
        for i in tqdm(range(0, frames_to_test, batch_size), 
                     desc="Pipeline benchmark"):
            batch_end = min(i + batch_size, frames_to_test)
            indices = list(range(i, batch_end))
            
            # Decode
            t0 = time.perf_counter()
            frames = self.vr.get_batch(indices)
            if self.device == 'gpu':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            stage_times['decode'].append(t1 - t0)
            
            # Grayscale conversion
            gray = torch.matmul(frames.float(), gray_weights).unsqueeze(1)
            if self.device == 'gpu':
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            stage_times['grayscale'].append(t2 - t1)
            
            # Downsample
            downsampled = F.interpolate(gray, size=target_size, 
                                       mode='bilinear', align_corners=False)
            if self.device == 'gpu':
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            stage_times['downsample'].append(t3 - t2)
            
            # Transfer to CPU (if GPU)
            if self.device == 'gpu':
                cpu_data = downsampled.cpu().numpy()
                t4 = time.perf_counter()
                stage_times['transfer'].append(t4 - t3)
            else:
                stage_times['transfer'].append(0)
            
            # Cleanup
            del frames, gray, downsampled
            if self.device == 'gpu':
                del cpu_data
        
        # Calculate statistics
        results = {}
        total_time = 0
        
        for stage, times in stage_times.items():
            stage_total = sum(times)
            total_time += stage_total
            results[stage] = {
                'total_time': stage_total,
                'avg_time': np.mean(times),
                'percentage': 0  # Will be filled
            }
        
        # Calculate percentages
        for stage in results:
            results[stage]['percentage'] = (results[stage]['total_time'] / total_time) * 100
        
        # Overall metrics
        results['overall'] = {
            'total_time': total_time,
            'fps': frames_to_test / total_time,
            'frames_processed': frames_to_test
        }
        
        # Display results
        self.console.print("\n[bold]Pipeline Stage Breakdown:[/bold]")
        for stage, metrics in results.items():
            if stage != 'overall':
                self.console.print(
                    f"  {stage:12s}: [yellow]{metrics['avg_time']*1000:6.2f}ms[/yellow] "
                    f"({metrics['percentage']:5.1f}%)"
                )
        
        self.console.print(f"\n  → Overall FPS: [green]{results['overall']['fps']:.1f}[/green]")
        
        self.results['benchmarks']['pipeline'] = results
        return results
    
    def benchmark_memory_scaling(self, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]):
        """Test memory usage scaling with batch size (GPU only)."""
        if self.device != 'gpu':
            self.console.print("[yellow]Memory scaling benchmark requires GPU[/yellow]")
            return None
        
        self.console.rule("[bold yellow]Memory Scaling Benchmark[/bold yellow]")
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > self.n_frames:
                continue
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                # Allocate batch
                indices = list(range(min(batch_size, self.n_frames)))
                frames = self.vr.get_batch(indices)
                torch.cuda.synchronize()
                
                # Measure memory
                allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                
                results[batch_size] = {
                    'allocated_mb': allocated,
                    'peak_mb': peak,
                    'mb_per_frame': allocated / batch_size
                }
                
                self.console.print(
                    f"  Batch {batch_size:3d}: "
                    f"[cyan]{allocated:7.1f}MB[/cyan] "
                    f"({results[batch_size]['mb_per_frame']:.1f}MB/frame)"
                )
                
                del frames
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                results[batch_size] = {'error': 'OOM'}
                self.console.print(f"  Batch {batch_size:3d}: [red]Out of Memory[/red]")
                break
        
        self.results['benchmarks']['memory_scaling'] = results
        return results
    
    def generate_report(self, save_path=None):
        """Generate comprehensive benchmark report."""
        self.console.rule("[bold green]Benchmark Report[/bold green]")
        
        # Create summary table
        table = Table(title="Performance Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Add key metrics
        if 'sequential_decode' in self.results['benchmarks']:
            best_batch = max(
                self.results['benchmarks']['sequential_decode'].items(),
                key=lambda x: x[1]['avg_fps']
            )
            table.add_row("Best Batch Size", str(best_batch[0]))
            table.add_row("Max Decode FPS", f"{best_batch[1]['avg_fps']:.1f}")
        
        if 'random_access' in self.results['benchmarks']:
            ra = self.results['benchmarks']['random_access']
            table.add_row("Avg Random Access", f"{ra['avg_access_time']*1000:.2f}ms")
        
        if 'pipeline' in self.results['benchmarks']:
            pl = self.results['benchmarks']['pipeline']['overall']
            table.add_row("Pipeline FPS", f"{pl['fps']:.1f}")
        
        self.console.print(table)
        
        # Add timestamp
        self.results['timestamp'] = datetime.now().isoformat()
        
        # Save results if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.console.print(f"\n[green]Results saved to:[/green] {save_path}")
        
        return self.results
    
    def plot_results(self, save_dir=None):
        """Generate visualization plots."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Batch size vs FPS
        if 'sequential_decode' in self.results['benchmarks']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            data = self.results['benchmarks']['sequential_decode']
            batch_sizes = list(data.keys())
            fps_values = [data[bs]['avg_fps'] for bs in batch_sizes]
            time_values = [data[bs]['avg_time_per_batch']*1000 for bs in batch_sizes]
            
            # FPS plot
            ax1.plot(batch_sizes, fps_values, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Batch Size', fontsize=12)
            ax1.set_ylabel('Frames Per Second', fontsize=12)
            ax1.set_title('Decode Performance vs Batch Size', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)
            
            # Time per batch plot
            ax2.plot(batch_sizes, time_values, 'o-', color='orange', linewidth=2, markersize=8)
            ax2.set_xlabel('Batch Size', fontsize=12)
            ax2.set_ylabel('Time per Batch (ms)', fontsize=12)
            ax2.set_title('Batch Processing Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(save_dir / 'batch_performance.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        # Plot 2: Pipeline breakdown
        if 'pipeline' in self.results['benchmarks']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pipeline_data = self.results['benchmarks']['pipeline']
            stages = ['decode', 'grayscale', 'downsample', 'transfer']
            percentages = [pipeline_data[s]['percentage'] for s in stages]
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            
            wedges, texts, autotexts = ax.pie(
                percentages, 
                labels=stages,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12}
            )
            
            ax.set_title('Processing Pipeline Time Distribution', 
                        fontsize=14, fontweight='bold', pad=20)
            
            if save_dir:
                plt.savefig(save_dir / 'pipeline_breakdown.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        # Plot 3: Memory scaling (GPU only)
        if 'memory_scaling' in self.results['benchmarks'] and self.results['benchmarks']['memory_scaling']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            mem_data = self.results['benchmarks']['memory_scaling']
            valid_data = {k: v for k, v in mem_data.items() if 'error' not in v}
            
            if valid_data:
                batch_sizes = list(valid_data.keys())
                memory_mb = [valid_data[bs]['allocated_mb'] for bs in batch_sizes]
                
                ax.plot(batch_sizes, memory_mb, 'o-', linewidth=2, markersize=8, color='purple')
                ax.set_xlabel('Batch Size', fontsize=12)
                ax.set_ylabel('Memory Usage (MB)', fontsize=12)
                ax.set_title('GPU Memory Usage vs Batch Size', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if save_dir:
                    plt.savefig(save_dir / 'memory_scaling.png', dpi=150, bbox_inches='tight')
                plt.show()


def main():
    parser = argparse.ArgumentParser(description='GPU Video Decoding Benchmark')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu',
                       help='Device to use for decoding')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum frames to test')
    parser.add_argument('--output', '-o', help='Path to save results JSON')
    parser.add_argument('--plot-dir', help='Directory to save plots')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with fewer tests')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = GPUDecodeBenchmark(args.video_path, device=args.device)
    
    # Run benchmarks
    if args.quick:
        # Quick mode - fewer batch sizes, fewer frames
        benchmark.benchmark_sequential_decode(
            batch_sizes=[1, 8, 32],
            max_frames=min(args.max_frames, 200)
        )
        benchmark.benchmark_processing_pipeline(
            batch_size=32,
            max_frames=min(args.max_frames, 200)
        )
    else:
        # Full benchmark suite
        benchmark.benchmark_sequential_decode(
            batch_sizes=[1, 2, 4, 8, 16, 32, 64],
            max_frames=args.max_frames
        )
        benchmark.benchmark_random_access(n_samples=100)
        benchmark.benchmark_processing_pipeline(
            batch_size=32,
            max_frames=min(args.max_frames, 500)
        )
        
        if args.device == 'gpu':
            benchmark.benchmark_memory_scaling()
    
    # Generate report
    benchmark.generate_report(save_path=args.output)
    
    # Generate plots
    if args.plot_dir:
        benchmark.plot_results(save_dir=args.plot_dir)
    
    return benchmark.results


if __name__ == '__main__':
    results = main()