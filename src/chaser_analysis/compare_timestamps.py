#!/usr/bin/env python3
"""
Timing analyzer for dual-rate system:
- Camera: 60 FPS (triggers bounding box detection)
- Stimulus: 120 FPS (renders visual output)
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from pathlib import Path

class DualRateTimingAnalyzer:
    def __init__(self, h5_path):
        self.h5_path = Path(h5_path)
        self.camera_fps = 60.0
        self.stimulus_fps = 120.0
        
    def analyze(self):
        """Perform complete dual-rate timing analysis."""
        print(f"\n{'='*70}")
        print(f"DUAL-RATE TIMING ANALYSIS")
        print(f"Camera: {self.camera_fps} FPS | Stimulus: {self.stimulus_fps} FPS")
        print(f"{'='*70}")
        
        with h5py.File(self.h5_path, 'r') as f:
            results = {}
            
            # Analyze stimulus frames (120 fps)
            if '/video_metadata/frame_metadata' in f:
                print("\n--- STIMULUS FRAMES (120 FPS) ---")
                frame_data = f['/video_metadata/frame_metadata']
                stim_timestamps = frame_data['timestamp_ns'][:]
                stim_frame_nums = frame_data['stimulus_frame_num'][:]
                camera_frame_ids = frame_data['triggering_camera_frame_id'][:]
                
                print(f"Total stimulus frames: {len(stim_timestamps)}")
                print(f"Stimulus frame range: {stim_frame_nums.min()}-{stim_frame_nums.max()}")
                print(f"Triggering camera frames: {np.unique(camera_frame_ids).size}")
                
                # Calculate actual stimulus frame rate
                stim_intervals = np.diff(stim_timestamps) / 1e6  # to ms
                valid_stim = stim_intervals[(stim_intervals > 0) & (stim_intervals < 50)]
                if len(valid_stim) > 0:
                    measured_stim_fps = 1000.0 / np.median(valid_stim)
                    print(f"Measured stimulus FPS: {measured_stim_fps:.1f}")
                    print(f"Expected interval: {1000/self.stimulus_fps:.2f} ms")
                    print(f"Actual interval: {np.median(valid_stim):.2f} ± {np.std(valid_stim):.2f} ms")
                    
                    # Check for frame drops
                    expected_interval = 1000.0 / self.stimulus_fps
                    dropped = np.sum(valid_stim > expected_interval * 1.5)
                    if dropped > 0:
                        print(f"⚠️  Potential dropped frames: {dropped}")
                
                # Analyze camera trigger pattern
                print(f"\n--- CAMERA TRIGGER PATTERN ---")
                trigger_changes = np.diff(camera_frame_ids)
                triggers_per_camera_frame = []
                current_count = 1
                for change in trigger_changes:
                    if change == 0:
                        current_count += 1
                    else:
                        triggers_per_camera_frame.append(current_count)
                        current_count = 1
                triggers_per_camera_frame.append(current_count)
                
                avg_triggers = np.mean(triggers_per_camera_frame)
                print(f"Stimulus frames per camera frame: {avg_triggers:.2f}")
                print(f"Expected ratio (120/60): {self.stimulus_fps/self.camera_fps:.1f}")
                
                if abs(avg_triggers - 2.0) > 0.1:
                    print(f"⚠️  Unexpected trigger ratio! Should be ~2.0")
                else:
                    print(f"✓ Trigger ratio correct")
                
                results['stimulus'] = {
                    'total_frames': len(stim_timestamps),
                    'measured_fps': measured_stim_fps,
                    'frames_per_camera': avg_triggers
                }
            
            # Analyze camera frames (60 fps via bounding boxes)
            if '/tracking_data/bounding_boxes' in f:
                print("\n--- CAMERA FRAMES (60 FPS) ---")
                bbox_data = f['/tracking_data/bounding_boxes']
                bbox_frame_ids = bbox_data['payload_frame_id'][:]
                bbox_payload_ts = bbox_data['payload_timestamp_ns_epoch'][:]
                bbox_received_ts = bbox_data['received_timestamp_ns_epoch'][:]
                
                unique_frames = np.unique(bbox_frame_ids)
                print(f"Unique camera frames: {len(unique_frames)}")
                print(f"Camera frame range: {unique_frames.min()}-{unique_frames.max()}")
                
                # Get timestamps for unique frames
                frame_times = {}
                frame_latencies = {}
                for i in range(len(bbox_frame_ids)):
                    fid = bbox_frame_ids[i]
                    if fid not in frame_times:
                        frame_times[fid] = bbox_payload_ts[i]
                        # Note: This "latency" will be wrong due to epoch mismatch
                        # but the relative values might still be informative
                        frame_latencies[fid] = bbox_received_ts[i] - bbox_payload_ts[i]
                
                # Calculate camera frame rate
                sorted_frames = sorted(frame_times.keys())
                frame_timestamps = [frame_times[f] for f in sorted_frames]
                cam_intervals = np.diff(frame_timestamps) / 1e6  # to ms
                
                # Filter out the huge jumps that indicate epoch issues
                valid_cam = cam_intervals[(cam_intervals > 0) & (cam_intervals < 50)]
                if len(valid_cam) > 0:
                    measured_cam_fps = 1000.0 / np.median(valid_cam)
                    print(f"Measured camera FPS: {measured_cam_fps:.1f}")
                    print(f"Expected interval: {1000/self.camera_fps:.2f} ms")
                    print(f"Actual interval: {np.median(valid_cam):.2f} ± {np.std(valid_cam):.2f} ms")
                else:
                    # Try with received timestamps instead
                    frame_times_recv = {}
                    for i in range(len(bbox_frame_ids)):
                        fid = bbox_frame_ids[i]
                        if fid not in frame_times_recv:
                            frame_times_recv[fid] = bbox_received_ts[i]
                    
                    sorted_frames = sorted(frame_times_recv.keys())
                    frame_timestamps_recv = [frame_times_recv[f] for f in sorted_frames]
                    cam_intervals_recv = np.diff(frame_timestamps_recv) / 1e6
                    valid_cam_recv = cam_intervals_recv[(cam_intervals_recv > 0) & (cam_intervals_recv < 50)]
                    
                    if len(valid_cam_recv) > 0:
                        measured_cam_fps = 1000.0 / np.median(valid_cam_recv)
                        print(f"Measured camera FPS (from received ts): {measured_cam_fps:.1f}")
                        print(f"Actual interval: {np.median(valid_cam_recv):.2f} ± {np.std(valid_cam_recv):.2f} ms")
                
                results['camera'] = {
                    'unique_frames': len(unique_frames),
                    'measured_fps': measured_cam_fps
                }
            
            # Analyze synchronization
            print("\n--- SYNCHRONIZATION ANALYSIS ---")
            if 'stimulus' in results and 'camera' in results:
                # Calculate expected totals
                duration_sec = results['camera']['unique_frames'] / self.camera_fps
                expected_stim_frames = duration_sec * self.stimulus_fps
                
                print(f"Recording duration: {duration_sec:.1f} seconds")
                print(f"Expected stimulus frames: {expected_stim_frames:.0f}")
                print(f"Actual stimulus frames: {results['stimulus']['total_frames']}")
                
                frame_diff = results['stimulus']['total_frames'] - expected_stim_frames
                pct_diff = 100 * frame_diff / expected_stim_frames
                
                if abs(pct_diff) < 1:
                    print(f"✓ Frame counts match expected (diff: {pct_diff:.2f}%)")
                else:
                    print(f"⚠️  Frame count mismatch: {frame_diff:+.0f} frames ({pct_diff:+.2f}%)")
                
                # Check timing consistency
                actual_ratio = results['stimulus']['total_frames'] / results['camera']['unique_frames']
                expected_ratio = self.stimulus_fps / self.camera_fps
                print(f"\nFrame ratio (stimulus/camera):")
                print(f"  Expected: {expected_ratio:.2f}")
                print(f"  Actual: {actual_ratio:.2f}")
                
                if abs(actual_ratio - expected_ratio) < 0.05:
                    print(f"  ✓ Rates are properly synchronized")
                else:
                    print(f"  ⚠️  Possible synchronization issue")
            
            return results
    
    def plot_timing(self):
        """Create visualization of dual-rate timing."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        with h5py.File(self.h5_path, 'r') as f:
            # Plot 1: Stimulus frame intervals
            if '/video_metadata/frame_metadata' in f:
                ax = axes[0, 0]
                frame_data = f['/video_metadata/frame_metadata']
                stim_timestamps = frame_data['timestamp_ns'][:]
                
                intervals = np.diff(stim_timestamps) / 1e6
                valid = intervals[(intervals > 0) & (intervals < 50)]
                
                ax.hist(valid, bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(x=1000/self.stimulus_fps, color='r', linestyle='--', 
                          label=f'Expected: {1000/self.stimulus_fps:.2f}ms')
                ax.set_xlabel('Inter-frame interval (ms)')
                ax.set_ylabel('Count')
                ax.set_title(f'Stimulus Frame Timing (Target: {self.stimulus_fps} FPS)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 2: Camera frame intervals
            if '/tracking_data/bounding_boxes' in f:
                ax = axes[0, 1]
                bbox_data = f['/tracking_data/bounding_boxes']
                bbox_frame_ids = bbox_data['payload_frame_id'][:]
                bbox_received_ts = bbox_data['received_timestamp_ns_epoch'][:]
                
                # Get unique frame timestamps
                frame_times = {}
                for i in range(len(bbox_frame_ids)):
                    fid = bbox_frame_ids[i]
                    if fid not in frame_times:
                        frame_times[fid] = bbox_received_ts[i]
                
                sorted_frames = sorted(frame_times.keys())
                timestamps = [frame_times[f] for f in sorted_frames]
                intervals = np.diff(timestamps) / 1e6
                valid = intervals[(intervals > 0) & (intervals < 50)]
                
                ax.hist(valid, bins=50, edgecolor='black', alpha=0.7)
                ax.axvline(x=1000/self.camera_fps, color='r', linestyle='--',
                          label=f'Expected: {1000/self.camera_fps:.2f}ms')
                ax.set_xlabel('Inter-frame interval (ms)')
                ax.set_ylabel('Count')
                ax.set_title(f'Camera Frame Timing (Target: {self.camera_fps} FPS)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 3: Camera trigger pattern in stimulus
            if '/video_metadata/frame_metadata' in f:
                ax = axes[1, 0]
                frame_data = f['/video_metadata/frame_metadata']
                camera_frame_ids = frame_data['triggering_camera_frame_id'][:]
                
                # Count stimulus frames per camera frame
                trigger_counts = {}
                current_cam_id = camera_frame_ids[0]
                count = 1
                
                for i in range(1, len(camera_frame_ids)):
                    if camera_frame_ids[i] == current_cam_id:
                        count += 1
                    else:
                        if current_cam_id not in trigger_counts:
                            trigger_counts[current_cam_id] = []
                        trigger_counts[current_cam_id].append(count)
                        current_cam_id = camera_frame_ids[i]
                        count = 1
                
                all_counts = []
                for counts in trigger_counts.values():
                    all_counts.extend(counts)
                
                ax.hist(all_counts, bins=range(0, 6), edgecolor='black', alpha=0.7)
                ax.axvline(x=2, color='r', linestyle='--', label='Expected: 2')
                ax.set_xlabel('Stimulus frames per camera frame')
                ax.set_ylabel('Count')
                ax.set_title('Camera Trigger Pattern')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Timeline
            if '/video_metadata/frame_metadata' in f and '/tracking_data/bounding_boxes' in f:
                ax = axes[1, 1]
                
                # Get stimulus timestamps
                frame_data = f['/video_metadata/frame_metadata']
                stim_timestamps = frame_data['timestamp_ns'][:1000]  # First 1000 for visibility
                stim_times = (stim_timestamps - stim_timestamps[0]) / 1e9  # to seconds
                
                # Get camera frame timestamps
                bbox_data = f['/tracking_data/bounding_boxes']
                bbox_frame_ids = bbox_data['payload_frame_id'][:500]  # First 500
                bbox_received_ts = bbox_data['received_timestamp_ns_epoch'][:500]
                
                # Get unique camera frames
                seen = set()
                cam_times = []
                for i in range(len(bbox_frame_ids)):
                    if bbox_frame_ids[i] not in seen:
                        seen.add(bbox_frame_ids[i])
                        cam_times.append((bbox_received_ts[i] - stim_timestamps[0]) / 1e9)
                
                ax.scatter(stim_times, np.ones_like(stim_times), s=1, alpha=0.5, label='Stimulus frames')
                ax.scatter(cam_times[:len(cam_times)//2], np.ones(len(cam_times)//2) * 0.5, 
                          s=10, alpha=0.7, label='Camera frames', color='red')
                
                ax.set_xlabel('Time (seconds)')
                ax.set_ylim(0, 1.5)
                ax.set_title('Frame Timeline (first second)')
                ax.set_xlim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yticks([0.5, 1])
                ax.set_yticklabels(['Camera\n60 FPS', 'Stimulus\n120 FPS'])
        
        plt.suptitle(f'Dual-Rate Timing Analysis: {self.h5_path.name}', fontsize=14)
        plt.tight_layout()
        return fig

def main():
    parser = argparse.ArgumentParser(
        description='Analyze dual-rate timing (60fps camera + 120fps stimulus)'
    )
    parser.add_argument('h5_file', help='Path to H5 file')
    parser.add_argument('--save-plot', help='Save plot to file')
    
    args = parser.parse_args()
    
    analyzer = DualRateTimingAnalyzer(args.h5_file)
    results = analyzer.analyze()
    
    fig = analyzer.plot_timing()
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {args.save_plot}")
    
    plt.show()

if __name__ == '__main__':
    main()