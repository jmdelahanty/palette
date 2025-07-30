#!/usr/bin/env python3
"""
Decord Video Decoding Profiler
Comprehensive tool to profile video decoding performance using the decord library.
Tests different configurations, formats, and provides detailed performance metrics.
"""

import os
import time
import argparse
import subprocess
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

try:
    import decord
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("❌ Decord not available. Install with: pip install decord")

def get_video_info(video_path):
    """Get detailed video information using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
            
        # Parse frame rate
        fps_str = video_stream.get('avg_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) > 0 else 0
        else:
            fps = float(fps_str)
        
        return {
            'format': info.get('format', {}).get('format_name', 'unknown'),
            'duration': float(info.get('format', {}).get('duration', 0)),
            'size_bytes': int(info.get('format', {}).get('size', 0)),
            'codec': video_stream.get('codec_name', 'unknown'),
            'profile': video_stream.get('profile', 'unknown'),
            'level': video_stream.get('level', 'unknown'),
            'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': fps,
            'bit_rate': int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else 0,
            'color_space': video_stream.get('color_space', 'unknown'),
            'color_transfer': video_stream.get('color_transfer', 'unknown'),
        }
    except Exception as e:
        print(f"⚠️  Could not get video info: {e}")
        return None

def profile_decord_loading(video_path, device_type='cpu'):
    """Profile the initial video loading time."""
    print(f"📂 Profiling video loading ({device_type})...")
    
    ctx = cpu(0) if device_type == 'cpu' else gpu(0)
    
    start_time = time.perf_counter()
    try:
        vr = VideoReader(str(video_path), ctx=ctx)
        load_time = time.perf_counter() - start_time
        
        return {
            'load_time': load_time,
            'total_frames': len(vr),
            'fps': vr.get_avg_fps(),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'load_time': time.perf_counter() - start_time,
            'total_frames': 0,
            'fps': 0,
            'success': False,
            'error': str(e)
        }

def profile_sequential_decoding(video_path, device_type='cpu', max_frames=1000, batch_size=32):
    """Profile sequential frame decoding performance."""
    print(f"⏯️  Profiling sequential decoding ({device_type}, batch_size={batch_size})...")
    
    ctx = cpu(0) if device_type == 'cpu' else gpu(0)
    
    try:
        vr = VideoReader(str(video_path), ctx=ctx)
        total_frames = min(len(vr), max_frames)
        
        decode_times = []
        frame_counts = []
        memory_usage = []
        
        print(f"   Testing {total_frames} frames in batches of {batch_size}")
        
        for i in tqdm(range(0, total_frames, batch_size), desc="Decoding"):
            batch_end = min(i + batch_size, total_frames)
            batch_indices = list(range(i, batch_end))
            actual_batch_size = len(batch_indices)
            
            # Memory before
            if device_type == 'gpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            else:
                mem_before = 0
            
            # Decode batch
            start_time = time.perf_counter()
            batch_frames = vr.get_batch(batch_indices)
            
            if device_type == 'gpu' and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure GPU operations complete
            
            decode_time = time.perf_counter() - start_time
            
            # Memory after
            if device_type == 'gpu' and torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append(mem_after - mem_before)
            else:
                memory_usage.append(0)
            
            decode_times.append(decode_time)
            frame_counts.append(actual_batch_size)
            
            # Clean up
            del batch_frames
            if device_type == 'gpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {
            'success': True,
            'total_frames_decoded': sum(frame_counts),
            'total_decode_time': sum(decode_times),
            'avg_fps': sum(frame_counts) / sum(decode_times) if sum(decode_times) > 0 else 0,
            'decode_times': decode_times,
            'frame_counts': frame_counts,
            'memory_usage': memory_usage,
            'batch_size': batch_size,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def profile_random_access(video_path, device_type='cpu', num_samples=100):
    """Profile random frame access performance."""
    print(f"🎲 Profiling random access ({device_type}, {num_samples} samples)...")
    
    ctx = cpu(0) if device_type == 'cpu' else gpu(0)
    
    try:
        vr = VideoReader(str(video_path), ctx=ctx)
        total_frames = len(vr)
        
        # Generate random frame indices
        np.random.seed(42)  # Reproducible results
        random_indices = np.random.randint(0, total_frames, num_samples)
        
        access_times = []
        
        for frame_idx in tqdm(random_indices, desc="Random access"):
            start_time = time.perf_counter()
            frame = vr[frame_idx]
            
            if device_type == 'gpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            access_time = time.perf_counter() - start_time
            access_times.append(access_time)
            
            del frame
        
        return {
            'success': True,
            'num_samples': num_samples,
            'total_time': sum(access_times),
            'avg_access_time': np.mean(access_times),
            'min_access_time': np.min(access_times),
            'max_access_time': np.max(access_times),
            'std_access_time': np.std(access_times),
            'access_times': access_times,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def profile_grayscale_conversion(video_path, device_type='cpu', batch_size=32, max_frames=500):
    """Profile grayscale conversion performance (like in your pipeline)."""
    print(f"🎨 Profiling grayscale conversion ({device_type})...")
    
    ctx = cpu(0) if device_type == 'cpu' else gpu(0)
    device = 'cpu' if device_type == 'cpu' else 'cuda:0'
    
    try:
        # Set bridge to match your pipeline
        decord.bridge.set_bridge('torch')
        vr = VideoReader(str(video_path), ctx=ctx)
        
        total_frames = min(len(vr), max_frames)
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
        
        conversion_times = []
        decode_times = []
        
        for i in tqdm(range(0, total_frames, batch_size), desc="Grayscale conversion"):
            batch_end = min(i + batch_size, total_frames)
            batch_indices = list(range(i, batch_end))
            
            # Decode
            decode_start = time.perf_counter()
            batch_frames = vr.get_batch(batch_indices)
            if device_type == 'gpu':
                torch.cuda.synchronize()
            decode_time = time.perf_counter() - decode_start
            decode_times.append(decode_time)
            
            # Convert to grayscale (your pipeline's method)
            convert_start = time.perf_counter()
            gray_batch = (batch_frames.float() @ gray_weights)
            if device_type == 'gpu':
                torch.cuda.synchronize()
            convert_time = time.perf_counter() - convert_start
            conversion_times.append(convert_time)
            
            del batch_frames, gray_batch
            if device_type == 'gpu':
                torch.cuda.empty_cache()
        
        return {
            'success': True,
            'total_decode_time': sum(decode_times),
            'total_conversion_time': sum(conversion_times),
            'avg_decode_fps': sum([len(range(i, min(i + batch_size, total_frames))) 
                                 for i in range(0, total_frames, batch_size)]) / sum(decode_times),
            'avg_conversion_fps': sum([len(range(i, min(i + batch_size, total_frames))) 
                                     for i in range(0, total_frames, batch_size)]) / sum(conversion_times),
            'decode_times': decode_times,
            'conversion_times': conversion_times,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def profile_downsampling(video_path, device_type='cpu', target_sizes=[(640, 640), (320, 320)], 
                        batch_size=16, max_frames=200):
    """Profile downsampling performance."""
    print(f"📐 Profiling downsampling ({device_type})...")
    
    ctx = cpu(0) if device_type == 'cpu' else gpu(0)
    device = 'cpu' if device_type == 'cpu' else 'cuda:0'
    
    try:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(str(video_path), ctx=ctx)
        
        total_frames = min(len(vr), max_frames)
        gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
        
        results = {}
        
        for target_size in target_sizes:
            print(f"   Testing {target_size[0]}x{target_size[1]} downsampling...")
            
            downsample_times = []
            
            for i in tqdm(range(0, total_frames, batch_size), 
                         desc=f"Downsampling to {target_size}", leave=False):
                batch_end = min(i + batch_size, total_frames)
                batch_indices = list(range(i, batch_end))
                
                # Decode and convert to grayscale
                batch_frames = vr.get_batch(batch_indices)
                gray_batch = (batch_frames.float() @ gray_weights).unsqueeze(1)
                
                # Downsample
                downsample_start = time.perf_counter()
                downsampled = F.interpolate(gray_batch, size=target_size, 
                                          mode='bilinear', align_corners=False)
                if device_type == 'gpu':
                    torch.cuda.synchronize()
                downsample_time = time.perf_counter() - downsample_start
                downsample_times.append(downsample_time)
                
                del batch_frames, gray_batch, downsampled
                if device_type == 'gpu':
                    torch.cuda.empty_cache()
            
            results[f"{target_size[0]}x{target_size[1]}"] = {
                'total_time': sum(downsample_times),
                'avg_fps': sum([len(range(i, min(i + batch_size, total_frames))) 
                              for i in range(0, total_frames, batch_size)]) / sum(downsample_times),
                'times': downsample_times
            }
        
        return {
            'success': True,
            'results': results,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_performance_report(video_path, results):
    """Create a comprehensive performance report."""
    print("\n" + "="*80)
    print("📊 DECORD PERFORMANCE REPORT")
    print("="*80)
    
    # Video information
    video_info = get_video_info(video_path)
    if video_info:
        print(f"\n🎬 Video Information:")
        print(f"   File: {Path(video_path).name}")
        print(f"   Format: {video_info['format']}")
        print(f"   Codec: {video_info['codec']} ({video_info['profile']})")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   Frame Rate: {video_info['fps']:.2f} fps")
        print(f"   Duration: {video_info['duration']:.2f} seconds")
        print(f"   File Size: {video_info['size_bytes']/1024/1024:.1f} MB")
        print(f"   Bitrate: {video_info['bit_rate']/1000:.0f} kbps")
        print(f"   Pixel Format: {video_info['pixel_format']}")
    
    # Device comparison
    print(f"\n⚡ Performance Summary:")
    for device in ['cpu', 'gpu']:
        if device in results:
            print(f"\n   {device.upper()} Performance:")
            
            # Loading
            load_result = results[device].get('loading', {})
            if load_result.get('success'):
                print(f"     Loading Time: {load_result['load_time']:.3f}s")
                print(f"     Total Frames: {load_result['total_frames']}")
            
            # Sequential decoding
            seq_result = results[device].get('sequential', {})
            if seq_result.get('success'):
                print(f"     Sequential Decode: {seq_result['avg_fps']:.1f} fps")
                print(f"     Total Decode Time: {seq_result['total_decode_time']:.2f}s")
            
            # Random access
            rand_result = results[device].get('random_access', {})
            if rand_result.get('success'):
                print(f"     Random Access: {rand_result['avg_access_time']*1000:.2f}ms avg")
            
            # Grayscale conversion
            gray_result = results[device].get('grayscale', {})
            if gray_result.get('success'):
                print(f"     Grayscale Decode: {gray_result['avg_decode_fps']:.1f} fps")
                print(f"     Grayscale Convert: {gray_result['avg_conversion_fps']:.1f} fps")
    
    # Performance recommendations
    print(f"\n💡 Recommendations:")
    
    # Compare CPU vs GPU if both available
    if 'cpu' in results and 'gpu' in results:
        cpu_fps = results['cpu'].get('sequential', {}).get('avg_fps', 0)
        gpu_fps = results['gpu'].get('sequential', {}).get('avg_fps', 0)
        
        if gpu_fps > cpu_fps * 1.5:
            print(f"   ✅ Use GPU decoding - {gpu_fps/cpu_fps:.1f}x faster than CPU")
        elif cpu_fps > gpu_fps:
            print(f"   ✅ Use CPU decoding - GPU shows no advantage")
        else:
            print(f"   ⚖️  CPU and GPU performance similar - use CPU for simplicity")
    
    # Check if video is suitable for batch processing
    if video_info and 'cpu' in results:
        seq_result = results['cpu'].get('sequential', {})
        if seq_result.get('success') and seq_result['avg_fps'] > video_info['fps'] * 2:
            print(f"   ✅ Video decodes faster than real-time - suitable for batch processing")
        elif seq_result.get('success'):
            print(f"   ⚠️  Video decoding is slower than real-time - consider optimization")
    
    # Check memory usage
    for device in ['cpu', 'gpu']:
        if device in results:
            seq_result = results[device].get('sequential', {})
            if seq_result.get('success') and seq_result.get('memory_usage'):
                max_mem = max(seq_result['memory_usage']) / 1024 / 1024  # MB
                if max_mem > 1000:  # > 1GB
                    print(f"   ⚠️  High {device.upper()} memory usage: {max_mem:.0f}MB - consider smaller batch sizes")

def create_performance_plots(results, output_dir=None):
    """Create performance visualization plots."""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Decord Performance Analysis', fontsize=16)
    
    # Plot 1: Decode FPS comparison
    ax1 = axes[0, 0]
    devices = []
    fps_values = []
    
    for device in ['cpu', 'gpu']:
        if device in results:
            seq_result = results[device].get('sequential', {})
            if seq_result.get('success'):
                devices.append(device.upper())
                fps_values.append(seq_result['avg_fps'])
    
    if devices:
        bars = ax1.bar(devices, fps_values, color=['skyblue', 'lightcoral'][:len(devices)])
        ax1.set_ylabel('Frames Per Second')
        ax1.set_title('Sequential Decoding Performance')
        for bar, fps in zip(bars, fps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.01,
                    f'{fps:.1f}', ha='center', va='bottom')
    
    # Plot 2: Random access time distribution
    ax2 = axes[0, 1]
    for i, device in enumerate(['cpu', 'gpu']):
        if device in results:
            rand_result = results[device].get('random_access', {})
            if rand_result.get('success'):
                times = np.array(rand_result['access_times']) * 1000  # Convert to ms
                ax2.hist(times, bins=20, alpha=0.7, label=device.upper(), 
                        color=['skyblue', 'lightcoral'][i])
    
    ax2.set_xlabel('Access Time (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Random Access Time Distribution')
    ax2.legend()
    
    # Plot 3: Batch decode time progression
    ax3 = axes[1, 0]
    for i, device in enumerate(['cpu', 'gpu']):
        if device in results:
            seq_result = results[device].get('sequential', {})
            if seq_result.get('success') and 'decode_times' in seq_result:
                batch_nums = range(len(seq_result['decode_times']))
                fps_per_batch = [fc/dt for fc, dt in zip(seq_result['frame_counts'], seq_result['decode_times'])]
                ax3.plot(batch_nums, fps_per_batch, label=f'{device.upper()}', 
                        color=['skyblue', 'lightcoral'][i], linewidth=2)
    
    ax3.set_xlabel('Batch Number')
    ax3.set_ylabel('FPS')
    ax3.set_title('Decoding Performance Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pipeline component breakdown
    ax4 = axes[1, 1]
    
    # Show breakdown for first available device
    for device in ['gpu', 'cpu']:  # Prefer GPU if available
        if device in results:
            gray_result = results[device].get('grayscale', {})
            if gray_result.get('success'):
                decode_fps = gray_result['avg_decode_fps']
                convert_fps = gray_result['avg_conversion_fps']
                
                components = ['Decode', 'Grayscale\nConversion']
                fps_values = [decode_fps, convert_fps]
                
                bars = ax4.bar(components, fps_values, color=['lightblue', 'lightgreen'])
                ax4.set_ylabel('Frames Per Second')
                ax4.set_title(f'Pipeline Components ({device.upper()})')
                
                for bar, fps in zip(bars, fps_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.01,
                            f'{fps:.1f}', ha='center', va='bottom')
                break
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'decord_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Profile video decoding performance using decord",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic profiling
  python decord_profiler.py video.mp4
  
  # Full profiling with plots
  python decord_profiler.py video.mp4 --full-profile --create-plots
  
  # CPU only profiling
  python decord_profiler.py video.mp4 --devices cpu
  
  # Custom batch sizes and frame limits
  python decord_profiler.py video.mp4 --batch-sizes 16,32,64 --max-frames 1000
        """
    )
    
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--devices", type=str, default="cpu,gpu", 
                       help="Devices to test (cpu,gpu)")
    parser.add_argument("--batch-sizes", type=str, default="32",
                       help="Batch sizes to test (comma-separated)")
    parser.add_argument("--max-frames", type=int, default=1000,
                       help="Maximum frames for sequential tests")
    parser.add_argument("--random-samples", type=int, default=100,
                       help="Number of random access samples")
    parser.add_argument("--full-profile", action="store_true",
                       help="Run all profiling tests")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create performance visualization plots")
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save results and plots")
    
    args = parser.parse_args()
    
    if not DECORD_AVAILABLE:
        return
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return
    
    # Parse devices and batch sizes
    devices = [d.strip() for d in args.devices.split(',')]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]
    
    # Validate GPU availability
    if 'gpu' in devices:
        try:
            if not torch.cuda.is_available():
                print("⚠️  GPU requested but CUDA not available, skipping GPU tests")
                devices = [d for d in devices if d != 'gpu']
            else:
                # Test GPU decord access
                ctx = gpu(0)
                print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"⚠️  GPU decord not available: {e}")
            devices = [d for d in devices if d != 'gpu']
    
    if not devices:
        print("❌ No valid devices available for testing")
        return
    
    print(f"🚀 Starting decord profiling...")
    print(f"📁 Video: {video_path}")
    print(f"🖥️  Devices: {', '.join(devices)}")
    print(f"📦 Batch sizes: {batch_sizes}")
    print()
    
    results = {}
    
    for device in devices:
        print(f"\n🔧 Testing {device.upper()} device...")
        results[device] = {}
        
        # 1. Video loading
        results[device]['loading'] = profile_decord_loading(video_path, device)
        
        # 2. Sequential decoding with different batch sizes
        best_fps = 0
        best_batch_size = batch_sizes[0]
        
        for batch_size in batch_sizes:
            seq_result = profile_sequential_decoding(
                video_path, device, args.max_frames, batch_size
            )
            
            if seq_result.get('success') and seq_result['avg_fps'] > best_fps:
                best_fps = seq_result['avg_fps']
                best_batch_size = batch_size
                results[device]['sequential'] = seq_result
        
        print(f"   ✅ Best batch size for {device}: {best_batch_size} ({best_fps:.1f} fps)")
        
        # 3. Random access (if full profile requested)
        if args.full_profile:
            results[device]['random_access'] = profile_random_access(
                video_path, device, args.random_samples
            )
        
        # 4. Grayscale conversion
        results[device]['grayscale'] = profile_grayscale_conversion(
            video_path, device, best_batch_size, args.max_frames
        )
        
        # 5. Downsampling (if full profile requested)
        if args.full_profile:
            results[device]['downsampling'] = profile_downsampling(
                video_path, device, [(640, 640), (320, 320)], best_batch_size
            )
    
    # Generate report
    create_performance_report(video_path, results)
    
    # Create plots if requested
    if args.create_plots:
        create_performance_plots(results, args.output_dir)
    
    # Save detailed results if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (excluding non-serializable data)
        import json
        serializable_results = {}
        for device, device_results in results.items():
            serializable_results[device] = {}
            for test, test_results in device_results.items():
                serializable_results[device][test] = {
                    k: v for k, v in test_results.items() 
                    if isinstance(v, (int, float, str, bool, list)) and k != 'access_times'
                }
        
        with open(output_dir / 'decord_profile_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {output_dir}")
    
    print(f"\n🎉 Profiling complete!")

if __name__ == "__main__":
    main()