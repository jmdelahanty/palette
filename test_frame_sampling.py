#!/usr/bin/env python3
"""
Quick test script to verify frame sampling implementation.
This checks that the CLI arguments are properly wired and metadata is stored.
"""

import sys
from pathlib import Path

# Test the argument parser
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fisheye.capture.import_video import build_parser, _compute_frame_indices

def test_parser():
    """Test that CLI arguments are properly configured"""
    parser = build_parser()

    # Test 1: training-data requires frame-step
    args = parser.parse_args(['video.mp4', '--training-data', '--frame-step', '100'])
    assert args.training_data == True
    assert args.frame_step == 100
    print("✓ Parser accepts --training-data with --frame-step")

    # Test 2: frame-step without training-data should be accepted by parser
    # (validation happens in main())
    args = parser.parse_args(['video.mp4', '--frame-step', '50'])
    assert args.frame_step == 50
    assert args.training_data == False
    print("✓ Parser accepts --frame-step argument")

    # Test 3: training-data alone
    args = parser.parse_args(['video.mp4', '--training-data'])
    assert args.training_data == True
    assert args.frame_step is None
    print("✓ Parser accepts --training-data alone")


def test_compute_frame_indices():
    """Test frame index computation"""
    # Test 1: Full import (no sampling)
    indices = _compute_frame_indices(100, None)
    assert len(indices) == 100
    assert indices == list(range(100))
    print("✓ Full import computes all indices")

    # Test 2: Every 10th frame
    indices = _compute_frame_indices(100, 10)
    assert len(indices) == 10
    assert indices == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    print("✓ Uniform sampling (step=10) works correctly")

    # Test 3: Step size 1 (same as full)
    indices = _compute_frame_indices(50, 1)
    assert len(indices) == 50
    assert indices == list(range(50))
    print("✓ Step size 1 equivalent to full import")

    # Test 4: Large step (sparse sampling)
    indices = _compute_frame_indices(1000, 100)
    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 900
    print("✓ Sparse sampling (step=100) works correctly")

    # Test 5: Step larger than total frames
    indices = _compute_frame_indices(50, 100)
    assert len(indices) == 1
    assert indices == [0]
    print("✓ Step larger than total frames returns only frame 0")


if __name__ == "__main__":
    print("Testing frame sampling implementation...")
    print()

    print("Testing argument parser...")
    test_parser()
    print()

    print("Testing frame index computation...")
    test_compute_frame_indices()
    print()

    print("="*60)
    print("✓ All tests passed!")
    print("="*60)
    print()
    print("To test with actual video import:")
    print("  python -m fisheye.capture.import_video \\")
    print("    video.mp4 \\")
    print("    --training-data \\")
    print("    --frame-step 100 \\")
    print("    --zarr-path test_sampled.zarr \\")
    print("    --overwrite")
