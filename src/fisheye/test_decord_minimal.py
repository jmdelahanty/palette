#!/usr/bin/env python3
"""
Minimal test to isolate exactly where decord fails.
"""

import sys
import time
from pathlib import Path

def test_stage(stage_name, func, timeout=5.0):
    """Run a test stage with timeout."""
    import threading
    
    result = {'success': False, 'error': None, 'elapsed': 0}
    
    def run():
        try:
            start = time.perf_counter()
            func()
            result['elapsed'] = time.perf_counter() - start
            result['success'] = True
        except Exception as e:
            result['error'] = e
    
    print(f"\n{'='*60}")
    print(f"Testing: {stage_name}")
    print(f"{'='*60}")
    
    thread = threading.Thread(target=run, daemon=True)
    start = time.perf_counter()
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        elapsed = time.perf_counter() - start
        print(f"❌ HUNG after {elapsed:.1f}s")
        return False
    elif result['error']:
        print(f"❌ ERROR: {result['error']}")
        return False
    else:
        print(f"✓ OK ({result['elapsed']*1000:.1f}ms)")
        return True


def main(video_path):
    video_path = Path(video_path)
    
    print("="*60)
    print("DECORD MINIMAL TEST")
    print("="*60)
    print(f"Video: {video_path}")
    
    # Stage 1: Import decord
    if not test_stage("Import decord", lambda: __import__('decord')):
        return 1
    
    import decord
    
    # Stage 2: Import GPU components
    def import_gpu():
        from decord import gpu, VideoReader
        import torch
        assert torch.cuda.is_available()
    
    if not test_stage("Import GPU components", import_gpu):
        return 1
    
    from decord import gpu, VideoReader
    import torch
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    # Stage 3: Set bridge
    if not test_stage("Set torch bridge", lambda: decord.bridge.set_bridge('torch')):
        return 1
    
    # Stage 4: Create VideoReader with CPU context first
    def create_cpu_reader():
        from decord import cpu
        reader = VideoReader(str(video_path), ctx=cpu(0))
        return reader
    
    if test_stage("Create CPU VideoReader", create_cpu_reader, timeout=10.0):
        print("  CPU reader works!")
    else:
        print("  CPU reader failed - video file may be corrupted")
        return 1
    
    # Stage 5: Create VideoReader with GPU context
    gpu_reader = [None]
    
    def create_gpu_reader():
        gpu_reader[0] = VideoReader(str(video_path), ctx=gpu(0))
    
    if not test_stage("Create GPU VideoReader", create_gpu_reader, timeout=10.0):
        print("\n⚠️  GPU VideoReader creation failed/hung")
        print("This is the problem! Decord GPU context cannot be initialized.")
        return 1
    
    # Stage 6: Access reader properties
    def check_properties():
        length = len(gpu_reader[0])
        print(f"  Video length: {length} frames")
    
    if not test_stage("Check VideoReader properties", check_properties):
        return 1
    
    # Stage 7: Decode first frame
    def decode_first():
        frame = gpu_reader[0][0]
        print(f"  Frame shape: {frame.shape}")
    
    if not test_stage("Decode first frame", decode_first, timeout=10.0):
        print("\n⚠️  Frame decoding failed/hung")
        print("This is the problem! GPU decoder cannot decode frames.")
        return 1
    
    # Stage 8: Decode middle frame
    def decode_middle():
        mid = len(gpu_reader[0]) // 2
        frame = gpu_reader[0][mid]
        print(f"  Frame {mid} shape: {frame.shape}")
    
    if not test_stage("Decode middle frame", decode_middle, timeout=10.0):
        return 1
    
    # Stage 9: Decode last frame
    def decode_last():
        last = len(gpu_reader[0]) - 1
        frame = gpu_reader[0][last]
        print(f"  Frame {last} shape: {frame.shape}")
    
    if not test_stage("Decode last frame", decode_last, timeout=10.0):
        return 1
    
    # Stage 10: Batch decode
    def decode_batch():
        frames = gpu_reader[0].get_batch([0, 1, 2, 3, 4])
        print(f"  Batch shape: {frames.shape}")
    
    if not test_stage("Decode batch (5 frames)", decode_batch, timeout=10.0):
        print("\n⚠️  Batch decoding failed/hung")
        print("This is the problem! GPU batch decoder has issues.")
        return 1
    
    # Stage 11: Batch decode near end
    def decode_batch_end():
        last = len(gpu_reader[0]) - 1
        frames = gpu_reader[0].get_batch([last-4, last-3, last-2, last-1, last])
        print(f"  Batch shape: {frames.shape}")
    
    if not test_stage("Decode batch near EOF", decode_batch_end, timeout=10.0):
        print("\n⚠️  Batch decoding near EOF failed/hung")
        print("This is a known decord issue with EOF!")
        return 1
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("GPU decoding should work for your crop operation.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_decord_minimal.py /path/to/video.mp4")
        sys.exit(1)
    
    sys.exit(main(sys.argv[1]))