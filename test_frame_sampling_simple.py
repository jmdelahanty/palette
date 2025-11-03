#!/usr/bin/env python3
"""
Simple standalone test for frame index computation logic.
"""

def _compute_frame_indices(total_frames: int, frame_step: int | None) -> list[int]:
    """
    Compute which frame indices to import based on sampling strategy.

    Args:
        total_frames: Total number of frames in the video
        frame_step: If provided, sample every Nth frame. If None, import all frames.

    Returns:
        List of frame indices to import
    """
    if frame_step is None or frame_step == 1:
        # Import all frames
        return list(range(total_frames))
    else:
        # Uniform sampling: [0, step, 2*step, ...]
        return list(range(0, total_frames, frame_step))


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

    # Test 6: Edge case - frame_step = total_frames
    indices = _compute_frame_indices(100, 100)
    assert len(indices) == 1
    assert indices == [0]
    print("✓ Step equal to total frames returns only frame 0")


if __name__ == "__main__":
    print("Testing frame index computation logic...")
    print()
    test_compute_frame_indices()
    print()
    print("="*60)
    print("✓ All logic tests passed!")
    print("="*60)
