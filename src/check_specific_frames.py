#!/usr/bin/env python3
"""
Check specific camera frames 816, 817, 818 in the H5 file
to understand why they're not showing chaser/target positions.
"""

import h5py
import numpy as np
import sys

def check_specific_frames(h5_path, frames_to_check=[816, 817, 818]):
    """
    Check if specific camera frames have all necessary data for visualization.
    """
    print(f"\n🔍 Checking frames {frames_to_check} in: {h5_path}")
    print("=" * 70)
    
    with h5py.File(h5_path, 'r') as f:
        # Load all datasets
        metadata = f['/video_metadata/frame_metadata'][:]
        chaser = f['/tracking_data/chaser_states'][:]
        bboxes = f['/tracking_data/bounding_boxes'][:]
        
        # Load interpolation mask if present
        mask = None
        if '/analysis/interpolation_mask' in f:
            mask = f['/analysis/interpolation_mask'][:]
        
        for cam_frame in frames_to_check:
            print(f"\n📷 CAMERA FRAME {cam_frame}:")
            print("-" * 50)
            
            # 1. Check if this camera frame exists in metadata
            meta_matches = metadata[metadata['triggering_camera_frame_id'] == cam_frame]
            
            if len(meta_matches) == 0:
                print(f"  ❌ NO METADATA ENTRY for camera frame {cam_frame}")
                print(f"     This frame cannot be displayed!")
            else:
                print(f"  ✅ Found {len(meta_matches)} metadata record(s)")
                
                for i, meta_record in enumerate(meta_matches):
                    stim_frame = meta_record['stimulus_frame_num']
                    timestamp = meta_record['timestamp_ns']
                    
                    # Find index in full metadata array
                    meta_idx = np.where((metadata['triggering_camera_frame_id'] == cam_frame) & 
                                       (metadata['stimulus_frame_num'] == stim_frame))[0]
                    
                    if len(meta_idx) > 0:
                        meta_idx = meta_idx[0]
                        is_interpolated = not mask[meta_idx] if mask is not None else "Unknown"
                    else:
                        is_interpolated = "Unknown"
                    
                    print(f"\n     Record {i+1}:")
                    print(f"       Stimulus frame: {stim_frame}")
                    print(f"       Timestamp: {timestamp}")
                    print(f"       Interpolated: {is_interpolated}")
                    
                    # 2. Check if this stimulus frame has chaser state
                    chaser_matches = chaser[chaser['stimulus_frame_num'] == stim_frame]
                    
                    if len(chaser_matches) == 0:
                        print(f"       ❌ NO CHASER STATE for stimulus frame {stim_frame}")
                    else:
                        chaser_record = chaser_matches[0]
                        print(f"       ✅ Chaser state found:")
                        print(f"          Chaser pos: ({chaser_record['chaser_pos_x']:.2f}, {chaser_record['chaser_pos_y']:.2f})")
                        print(f"          Target pos: ({chaser_record['target_pos_x']:.2f}, {chaser_record['target_pos_y']:.2f})")
                        print(f"          Is chasing: {chaser_record['is_chasing']}")
            
            # 3. Check if this camera frame has bounding boxes
            bbox_matches = bboxes[bboxes['payload_frame_id'] == cam_frame]
            
            if len(bbox_matches) == 0:
                print(f"\n  ⚠️  No bounding boxes for camera frame {cam_frame}")
            else:
                print(f"\n  ✅ Found {len(bbox_matches)} bounding box(es)")
                for j, bbox in enumerate(bbox_matches[:3]):  # Show first 3
                    print(f"     Box {j+1}: ({bbox['x_min']:.1f}, {bbox['y_min']:.1f}) "
                          f"size: {bbox['width']:.1f}x{bbox['height']:.1f}")
        
        # Additional analysis: Check nearby frames
        print("\n" + "=" * 70)
        print("📊 CONTEXT: Checking surrounding frames")
        print("-" * 50)
        
        # Check range around these frames
        check_range = range(814, 821)  # 814-820
        
        print(f"\nCamera frames {min(check_range)}-{max(check_range)} presence in metadata:")
        for cf in check_range:
            has_meta = len(metadata[metadata['triggering_camera_frame_id'] == cf]) > 0
            has_bbox = len(bboxes[bboxes['payload_frame_id'] == cf]) > 0
            
            symbol = "✅" if has_meta else "❌"
            bbox_symbol = "📦" if has_bbox else "⚠️"
            
            if cf in frames_to_check:
                print(f"  {symbol} Frame {cf}: {'Present' if has_meta else 'MISSING'} in metadata "
                      f"{bbox_symbol} {'Has' if has_bbox else 'No'} bbox  <-- PROBLEM FRAME")
            else:
                print(f"  {symbol} Frame {cf}: {'Present' if has_meta else 'MISSING'} in metadata "
                      f"{bbox_symbol} {'Has' if has_bbox else 'No'} bbox")
        
        # Show gap pattern
        all_camera_frames = np.unique(metadata['triggering_camera_frame_id'])
        
        # Find where these frames should be in the sequence
        nearest_before = all_camera_frames[all_camera_frames < min(frames_to_check)]
        nearest_after = all_camera_frames[all_camera_frames > max(frames_to_check)]
        
        if len(nearest_before) > 0 and len(nearest_after) > 0:
            print(f"\n🔗 Gap Analysis:")
            print(f"  Last frame before gap: {nearest_before[-1]}")
            print(f"  First frame after gap: {nearest_after[0]}")
            print(f"  Gap size: {nearest_after[0] - nearest_before[-1] - 1} frames")
            print(f"  Missing frames: {list(range(nearest_before[-1] + 1, nearest_after[0]))}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_specific_frames.py <h5_file> [frame1 frame2 ...]")
        print("Default: checks frames 816, 817, 818")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        frames = [int(f) for f in sys.argv[2:]]
    else:
        frames = [816, 817, 818]
    
    check_specific_frames(h5_path, frames)

if __name__ == '__main__':
    main()