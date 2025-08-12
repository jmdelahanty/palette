#!/usr/bin/env python3
"""
Diagnostic tool for video files with HEVC/H.265 encoding issues.
Checks for POC errors, keyframe issues, and provides repair options.
"""

import subprocess
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys

try:
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Note: decord not available, using OpenCV only")

class VideoDiagnostic:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.diagnostics = {}
        
    def get_ffprobe_info(self):
        """Get detailed video information using ffprobe."""
        print(f"\n{'='*60}")
        print(f"VIDEO DIAGNOSTICS: {self.video_path.name}")
        print(f"{'='*60}")
        
        try:
            # Get stream info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if video_stream:
                print("\n--- Video Stream Info ---")
                print(f"Codec: {video_stream.get('codec_name', 'unknown')}")
                print(f"Profile: {video_stream.get('profile', 'unknown')}")
                print(f"Resolution: {video_stream.get('width')}x{video_stream.get('height')}")
                print(f"Frame rate: {eval(video_stream.get('r_frame_rate', '0/1')):.2f} fps")
                print(f"Duration: {float(video_stream.get('duration', 0)):.2f} seconds")
                print(f"Total frames: {video_stream.get('nb_frames', 'unknown')}")
                print(f"Pixel format: {video_stream.get('pix_fmt', 'unknown')}")
                print(f"Has B-frames: {video_stream.get('has_b_frames', 'unknown')}")
                
                self.diagnostics['codec'] = video_stream.get('codec_name', 'unknown')
                self.diagnostics['has_b_frames'] = video_stream.get('has_b_frames', 0)
                
                # Check for HEVC specific issues
                if video_stream.get('codec_name') == 'hevc':
                    print("\n⚠️  HEVC/H.265 codec detected")
                    print("   POC errors are common with HEVC when:")
                    print("   - Keyframes are too sparse")
                    print("   - B-frames reference missing frames")
                    print("   - Seeking without proper keyframe alignment")
            
            return info
            
        except Exception as e:
            print(f"Error running ffprobe: {e}")
            return None
    
    def check_keyframes(self):
        """Check keyframe distribution."""
        print("\n--- Keyframe Analysis ---")
        
        try:
            # Get keyframe information
            cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                '-show_entries', 'frame=pict_type,pts_time',
                '-of', 'csv=p=0', str(self.video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            frames = result.stdout.strip().split('\n')[:1000]  # Check first 1000 frames
            
            keyframes = []
            frame_types = {'I': 0, 'P': 0, 'B': 0}
            
            for i, frame in enumerate(frames):
                if frame:
                    parts = frame.split(',')
                    if len(parts) >= 2:
                        frame_type = parts[0]
                        timestamp = float(parts[1]) if parts[1] != 'N/A' else i/60.0
                        
                        if frame_type in frame_types:
                            frame_types[frame_type] += 1
                        
                        if frame_type == 'I':
                            keyframes.append((i, timestamp))
            
            print(f"Frame type distribution (first 1000 frames):")
            print(f"  I-frames (keyframes): {frame_types['I']}")
            print(f"  P-frames: {frame_types['P']}")
            print(f"  B-frames: {frame_types['B']}")
            
            if keyframes:
                intervals = [keyframes[i+1][1] - keyframes[i][1] 
                            for i in range(len(keyframes)-1)]
                if intervals:
                    print(f"\nKeyframe intervals:")
                    print(f"  Average: {np.mean(intervals):.2f} seconds")
                    print(f"  Max: {max(intervals):.2f} seconds")
                    print(f"  Min: {min(intervals):.2f} seconds")
                    
                    if max(intervals) > 10:
                        print("\n⚠️  WARNING: Very sparse keyframes detected!")
                        print("   This can cause POC errors during seeking")
                        self.diagnostics['sparse_keyframes'] = True
            
            return keyframes
            
        except Exception as e:
            print(f"Error analyzing keyframes: {e}")
            return []
    
    def test_opencv_reading(self):
        """Test reading with OpenCV."""
        print("\n--- OpenCV Reading Test ---")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print("❌ Failed to open with OpenCV")
            return False
        
        # Try reading frames
        success_count = 0
        error_count = 0
        
        for i in range(min(100, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
            else:
                error_count += 1
        
        cap.release()
        
        print(f"✓ Successfully read {success_count}/100 frames")
        if error_count > 0:
            print(f"⚠️  Failed to read {error_count} frames")
        
        return error_count == 0
    
    def test_decord_reading(self):
        """Test reading with decord if available."""
        if not DECORD_AVAILABLE:
            print("\n--- Decord Test ---")
            print("Decord not available")
            return None
        
        print("\n--- Decord Reading Test ---")
        
        try:
            # Try CPU first
            vr = VideoReader(str(self.video_path), ctx=cpu(0))
            print(f"✓ Opened with decord (CPU)")
            print(f"  Total frames: {len(vr)}")
            
            # Test reading
            test_frames = [0, len(vr)//2, len(vr)-1]
            for idx in test_frames:
                try:
                    frame = vr[idx]
                    print(f"  ✓ Read frame {idx}")
                except Exception as e:
                    print(f"  ❌ Failed to read frame {idx}: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Decord failed: {e}")
            return False
    
    def suggest_fixes(self):
        """Suggest fixes based on diagnostics."""
        print("\n" + "="*60)
        print("RECOMMENDED FIXES")
        print("="*60)
        
        fixes = []
        
        if self.diagnostics.get('codec') == 'hevc':
            if self.diagnostics.get('sparse_keyframes'):
                fixes.append({
                    'issue': 'Sparse keyframes in HEVC',
                    'command': f'ffmpeg -i {self.video_path} -c:v libx264 -preset slow -crf 22 {self.video_path.stem}_h264.mp4',
                    'description': 'Re-encode to H.264 with regular keyframes'
                })
                
                fixes.append({
                    'issue': 'Sparse keyframes in HEVC',
                    'command': f'ffmpeg -i {self.video_path} -c:v hevc_nvenc -preset slow -rc vbr -cq 22 -g 60 {self.video_path.stem}_hevc_fixed.mp4',
                    'description': 'Re-encode with NVENC HEVC with keyframe every 60 frames'
                })
            
            if self.diagnostics.get('has_b_frames', 0) > 0:
                fixes.append({
                    'issue': 'B-frames causing POC errors',
                    'command': f'ffmpeg -i {self.video_path} -c:v copy -bsf:v hevc_mp4toannexb {self.video_path.stem}_annexb.mp4',
                    'description': 'Convert to Annex B format (may help with some players)'
                })
        
        # General fix
        fixes.append({
            'issue': 'General compatibility',
            'command': f'ffmpeg -i {self.video_path} -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p {self.video_path.stem}_compatible.mp4',
            'description': 'Create a widely compatible H.264 version'
        })
        
        if fixes:
            for i, fix in enumerate(fixes, 1):
                print(f"\n{i}. {fix['issue']}")
                print(f"   Description: {fix['description']}")
                print(f"   Command:")
                print(f"   {fix['command']}")
        
        return fixes
    
    def create_test_segment(self, output_path=None, start_sec=0, duration=10):
        """Create a small test segment for debugging."""
        if output_path is None:
            output_path = self.video_path.stem + "_test_segment.mp4"
        
        print(f"\n--- Creating Test Segment ---")
        print(f"Extracting {duration} seconds starting at {start_sec}s")
        
        cmd = [
            'ffmpeg', '-ss', str(start_sec), '-i', str(self.video_path),
            '-t', str(duration), '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', str(output_path), '-y'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ Test segment saved to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create test segment: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose and fix video encoding issues'
    )
    parser.add_argument('video_file', help='Path to video file')
    parser.add_argument('--fix', action='store_true', 
                       help='Apply recommended fix automatically')
    parser.add_argument('--test-segment', action='store_true',
                       help='Create a test segment for debugging')
    parser.add_argument('--output-format', choices=['h264', 'hevc_nvenc'],
                       default='h264', help='Output format for fixes')
    
    args = parser.parse_args()
    
    if not Path(args.video_file).exists():
        print(f"Error: File not found: {args.video_file}")
        sys.exit(1)
    
    # Run diagnostics
    diag = VideoDiagnostic(args.video_file)
    
    # Get video info
    diag.get_ffprobe_info()
    
    # Check keyframes
    diag.check_keyframes()
    
    # Test reading
    diag.test_opencv_reading()
    diag.test_decord_reading()
    
    # Suggest fixes
    fixes = diag.suggest_fixes()
    
    # Create test segment if requested
    if args.test_segment:
        diag.create_test_segment()
    
    # Apply fix if requested
    if args.fix and fixes:
        print("\n" + "="*60)
        print("APPLYING FIX")
        print("="*60)
        
        if args.output_format == 'h264':
            # Use the H.264 fix
            fix_cmd = f'ffmpeg -i {args.video_file} -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -g 60 {Path(args.video_file).stem}_fixed.mp4'
        else:
            # Use NVENC HEVC
            fix_cmd = f'ffmpeg -i {args.video_file} -c:v hevc_nvenc -preset p4 -rc vbr -cq 23 -g 60 -bf 0 {Path(args.video_file).stem}_fixed.mp4'
        
        print(f"Running: {fix_cmd}")
        
        try:
            subprocess.run(fix_cmd.split(), check=True)
            print("\n✓ Fixed video created successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Fix failed: {e}")

if __name__ == '__main__':
    main()