#!/usr/bin/env python3
"""
Video integrity checker to detect missing frames, packet loss, and timestamp issues.
Specifically designed for videos encoded with low-latency settings (no B-frames).
"""

import subprocess
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class VideoIntegrityChecker:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.frame_data = []
        self.issues = defaultdict(list)
        
    def extract_frame_info(self, max_frames=None):
        """Extract detailed frame information using ffprobe."""
        print(f"\n{'='*60}")
        print(f"EXTRACTING FRAME INFORMATION")
        print(f"{'='*60}")
        
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=pkt_pts,pkt_pts_time,pkt_dts,pkt_dts_time,pict_type,key_frame,pkt_size',
            '-of', 'json', str(self.video_path)
        ]
        
        if max_frames:
            cmd.extend(['-read_intervals', f'%+{max_frames}'])
        
        try:
            print("Extracting frame data (this may take a moment)...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            self.frame_data = data.get('frames', [])
            print(f"Extracted {len(self.frame_data)} frames")
            
            return self.frame_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame info: {e}")
            return []
    
    def analyze_timestamps(self):
        """Analyze timestamp continuity and detect gaps."""
        if not self.frame_data:
            print("No frame data available")
            return [], []
        
        print(f"\n{'='*60}")
        print("TIMESTAMP ANALYSIS")
        print(f"{'='*60}")
        
        pts_times = []
        dts_times = []
        frame_types = []
        gaps = []  # Initialize gaps list
        
        for i, frame in enumerate(self.frame_data):
            pts_time = frame.get('pkt_pts_time')
            dts_time = frame.get('pkt_dts_time')
            pict_type = frame.get('pict_type', 'U')
            
            if pts_time and pts_time != 'N/A':
                pts_times.append(float(pts_time))
            if dts_time and dts_time != 'N/A':
                dts_times.append(float(dts_time))
            frame_types.append(pict_type)
        
        # Check for timestamp gaps
        if len(pts_times) > 1:
            pts_diffs = np.diff(pts_times)
            expected_interval = np.median(pts_diffs)
            
            print(f"\nPresentation Timestamps (PTS):")
            print(f"  Total frames: {len(pts_times)}")
            print(f"  Expected interval: {expected_interval*1000:.2f} ms")
            print(f"  Actual interval: {np.mean(pts_diffs)*1000:.2f} ± {np.std(pts_diffs)*1000:.2f} ms")
            
            # Detect gaps (frames that are missing)
            gap_threshold = expected_interval * 1.5
            gaps = []
            for i, diff in enumerate(pts_diffs):
                if diff > gap_threshold:
                    missing_frames = int(diff / expected_interval) - 1
                    gaps.append({
                        'position': i,
                        'time': pts_times[i],
                        'gap_duration': diff,
                        'missing_frames': missing_frames
                    })
                    self.issues['timestamp_gaps'].append(f"Gap at {pts_times[i]:.2f}s: {missing_frames} frames missing")
            
            if gaps:
                print(f"\n⚠️  DETECTED {len(gaps)} TIMESTAMP GAPS:")
                for gap in gaps[:10]:  # Show first 10
                    print(f"  At {gap['time']:.2f}s: ~{gap['missing_frames']} frames missing ({gap['gap_duration']*1000:.1f}ms gap)")
                if len(gaps) > 10:
                    print(f"  ... and {len(gaps)-10} more gaps")
                    
                total_missing = sum(g['missing_frames'] for g in gaps)
                print(f"\n  Total estimated missing frames: {total_missing}")
            else:
                print("  ✓ No timestamp gaps detected")
        
        # Analyze frame types
        frame_type_counts = defaultdict(int)
        for ft in frame_types:
            frame_type_counts[ft] += 1
        
        print(f"\nFrame Type Distribution:")
        for ft, count in sorted(frame_type_counts.items()):
            print(f"  {ft}-frames: {count}")
        
        # Check for B-frames
        if frame_type_counts.get('B', 0) > 0:
            print("\n⚠️  B-frames detected! This contradicts the encoder configuration.")
            self.issues['unexpected_b_frames'] = frame_type_counts['B']
        else:
            print("  ✓ No B-frames (as expected for low-latency encoding)")
        
        return pts_times, gaps
    
    def analyze_gop_structure(self):
        """Analyze Group of Pictures structure."""
        print(f"\n{'='*60}")
        print("GOP STRUCTURE ANALYSIS")
        print(f"{'='*60}")
        
        if not self.frame_data:
            return
        
        keyframe_positions = []
        gop_sizes = []
        current_gop_start = 0
        
        for i, frame in enumerate(self.frame_data):
            if frame.get('key_frame') == 1 or frame.get('pict_type') == 'I':
                if i > 0:
                    gop_sizes.append(i - current_gop_start)
                current_gop_start = i
                keyframe_positions.append(i)
        
        if keyframe_positions:
            print(f"Keyframes found: {len(keyframe_positions)}")
            
            if gop_sizes:
                print(f"GOP sizes:")
                print(f"  Average: {np.mean(gop_sizes):.1f} frames")
                print(f"  Min: {min(gop_sizes)} frames")
                print(f"  Max: {max(gop_sizes)} frames")
                
                if max(gop_sizes) > 300:  # More than 5 seconds at 60fps
                    print("\n⚠️  Very large GOPs detected! This can cause decoder issues.")
                    self.issues['large_gops'] = max(gop_sizes)
        
        return keyframe_positions
    
    def check_sequential_integrity(self):
        """Check if frames can be read sequentially without errors."""
        print(f"\n{'='*60}")
        print("SEQUENTIAL READ TEST")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print("❌ Failed to open video")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Testing sequential read of {min(1000, total_frames)} frames...")
        
        failed_frames = []
        last_valid_frame = None
        
        for i in range(min(1000, total_frames)):
            ret, frame = cap.read()
            
            if not ret or frame is None:
                failed_frames.append(i)
                # Try to recover by seeking to next frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)
            else:
                last_valid_frame = frame
            
            if i % 100 == 0:
                print(f"  Checked {i} frames...")
        
        cap.release()
        
        if failed_frames:
            print(f"\n⚠️  Failed to read {len(failed_frames)} frames:")
            print(f"  Failed frame indices: {failed_frames[:20]}")
            if len(failed_frames) > 20:
                print(f"  ... and {len(failed_frames)-20} more")
            self.issues['unreadable_frames'] = failed_frames
        else:
            print("  ✓ All frames readable")
        
        return failed_frames
    
    def check_random_access(self, test_positions=None):
        """Test random access to frames."""
        print(f"\n{'='*60}")
        print("RANDOM ACCESS TEST")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print("❌ Failed to open video")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if test_positions is None:
            # Test 20 random positions
            test_positions = np.linspace(0, total_frames-1, min(20, total_frames), dtype=int)
        
        print(f"Testing random access to {len(test_positions)} positions...")
        
        seek_failures = []
        for pos in test_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            
            if not ret or frame is None:
                seek_failures.append(pos)
                print(f"  ❌ Failed to seek to frame {pos}")
            elif abs(actual_pos - pos) > 1:
                print(f"  ⚠️  Seek inaccuracy: requested {pos}, got {actual_pos}")
        
        cap.release()
        
        if seek_failures:
            print(f"\n⚠️  Random access failed for {len(seek_failures)} positions")
            self.issues['seek_failures'] = seek_failures
        else:
            print("  ✓ Random access working correctly")
    
    def visualize_analysis(self, pts_times, gaps):
        """Create visualization of the analysis."""
        if not pts_times:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Frame intervals
        ax = axes[0]
        if len(pts_times) > 1:
            intervals = np.diff(pts_times) * 1000  # Convert to ms
            ax.plot(intervals, alpha=0.7)
            ax.axhline(y=np.median(intervals), color='r', linestyle='--', 
                      label=f'Expected: {np.median(intervals):.2f}ms')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Interval (ms)')
            ax.set_title('Frame-to-Frame Intervals')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Timestamp progression
        ax = axes[1]
        expected_times = np.linspace(0, pts_times[-1], len(pts_times))
        ax.plot(pts_times, label='Actual', alpha=0.7)
        ax.plot(expected_times, '--', label='Expected (linear)', alpha=0.7)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Timestamp Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Gap visualization
        ax = axes[2]
        if gaps:
            gap_times = [g['time'] for g in gaps]
            gap_sizes = [g['missing_frames'] for g in gaps]
            ax.scatter(gap_times, gap_sizes, color='red', s=50)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Missing Frames')
            ax.set_title(f'Detected Gaps ({len(gaps)} total)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No gaps detected', ha='center', va='center', fontsize=14)
            ax.set_title('Gap Analysis')
        
        plt.suptitle(f'Video Integrity Analysis: {self.video_path.name}', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive report."""
        print(f"\n{'='*60}")
        print("INTEGRITY CHECK SUMMARY")
        print(f"{'='*60}")
        
        if not self.issues:
            print("✅ No integrity issues detected!")
            print("   Video appears to be properly encoded without packet loss.")
        else:
            print("⚠️  ISSUES DETECTED:")
            
            if 'timestamp_gaps' in self.issues:
                print(f"\n1. Missing Frames/Packet Loss:")
                print(f"   - {len(self.issues['timestamp_gaps'])} gaps detected")
                print(f"   - Likely cause: Frames dropped in encoding/writing pipeline")
                
            if 'unreadable_frames' in self.issues:
                print(f"\n2. Unreadable Frames:")
                print(f"   - {len(self.issues['unreadable_frames'])} frames cannot be decoded")
                print(f"   - Likely cause: Missing reference frames due to packet loss")
                
            if 'seek_failures' in self.issues:
                print(f"\n3. Seek Failures:")
                print(f"   - {len(self.issues['seek_failures'])} positions cannot be seeked to")
                print(f"   - Likely cause: Missing keyframes or corrupted GOP structure")
                
            if 'large_gops' in self.issues:
                print(f"\n4. Large GOPs:")
                print(f"   - Maximum GOP size: {self.issues['large_gops']} frames")
                print(f"   - This can cause decoder issues and seek problems")
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if 'timestamp_gaps' in self.issues:
            print("\nFor packet loss in encoder/writer:")
            print("  1. Increase queue size in FFmpegWriter")
            print("  2. Check CPU/disk I/O during encoding")
            print("  3. Consider using a separate SSD for video output")
            print("  4. Monitor queue depth during recording")
            
        if self.issues:
            print("\nTo salvage this video:")
            print("  ffmpeg -i input.mp4 -c:v libx264 -preset medium -crf 23 output_fixed.mp4")
            print("\nFor future recordings:")
            print("  1. Add queue overflow detection in FFmpegWriter")
            print("  2. Log dropped frame statistics")
            print("  3. Consider reducing resolution/framerate if system is overloaded")

def main():
    parser = argparse.ArgumentParser(
        description='Check video integrity and detect missing frames'
    )
    parser.add_argument('video_file', help='Path to video file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to analyze')
    parser.add_argument('--save-plot', help='Save analysis plots')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick check (first 1000 frames only)')
    
    args = parser.parse_args()
    
    if not Path(args.video_file).exists():
        print(f"Error: File not found: {args.video_file}")
        return
    
    checker = VideoIntegrityChecker(args.video_file)
    
    # Extract frame information
    max_frames = 1000 if args.quick else args.max_frames
    checker.extract_frame_info(max_frames)
    
    # Analyze timestamps
    pts_times, gaps = checker.analyze_timestamps()
    
    # Analyze GOP structure
    checker.analyze_gop_structure()
    
    # Check sequential reading
    checker.check_sequential_integrity()
    
    # Check random access
    checker.check_random_access()
    
    # Generate report
    checker.generate_report()
    
    # Create visualization
    if pts_times:
        fig = checker.visualize_analysis(pts_times, gaps)
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"\nPlots saved to: {args.save_plot}")
        else:
            plt.show()

if __name__ == '__main__':
    main()