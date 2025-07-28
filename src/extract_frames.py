#!/usr/bin/env python3
"""
Extract Test Frames Script
Uses ffmpeg to extract frames from the source video for testing the trained YOLO model.
"""

import subprocess
import argparse
from pathlib import Path
import os

def extract_frames_with_ffmpeg(video_path, output_dir, num_frames=50, start_time=None, frame_interval=None):
    """
    Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to source video
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        start_time: Start time in seconds (optional)
        frame_interval: Extract every Nth frame (optional)
    """
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Extracting frames from: {video_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔢 Number of frames: {num_frames}")
    
    try:
        # First, get video info
        info_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(video_path)
        ]
        
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        
        # Parse duration and frame count if possible
        import json
        info = json.loads(result.stdout)
        
        duration = None
        total_frames = None
        fps = None
        
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                duration = float(stream.get('duration', 0))
                total_frames = int(stream.get('nb_frames', 0))
                fps_str = stream.get('avg_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) > 0 else 0
                break
        
        print(f"📊 Video info: {duration:.1f}s, {total_frames} frames, {fps:.2f} fps")
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-i', str(video_path)]
        
        # Add start time if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        
        # Method 1: Extract evenly spaced frames
        if frame_interval is None and duration:
            # Calculate interval to get approximately num_frames
            if start_time:
                remaining_duration = duration - start_time
            else:
                remaining_duration = duration
            
            frame_interval = max(1, int(remaining_duration * fps / num_frames))
        
        # Add frame selection
        if frame_interval:
            cmd.extend(['-vf', f'select=not(mod(n\\,{frame_interval}))'])
            cmd.extend(['-frames:v', str(num_frames)])
        else:
            cmd.extend(['-frames:v', str(num_frames)])
        
        # Output settings
        cmd.extend([
            '-q:v', '2',  # High quality
            '-f', 'image2',
            str(output_dir / 'frame_%04d.jpg')
        ])
        
        print(f"🔧 Running ffmpeg command...")
        print(f"   {' '.join(cmd)}")
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count extracted frames
        extracted_frames = list(output_dir.glob('frame_*.jpg'))
        print(f"✅ Successfully extracted {len(extracted_frames)} frames")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg error: {e}")
        print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def extract_specific_frames(video_path, output_dir, frame_numbers):
    """
    Extract specific frame numbers from video.
    
    Args:
        video_path: Path to source video
        output_dir: Directory to save frames
        frame_numbers: List of frame numbers to extract
    """
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Extracting specific frames: {frame_numbers}")
    
    for i, frame_num in enumerate(frame_numbers):
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', f'select=eq(n\\,{frame_num})',
                '-frames:v', '1',
                '-q:v', '2',
                str(output_dir / f'frame_{frame_num:06d}.jpg')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Extracted frame {frame_num}")
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to extract frame {frame_num}: {e}")
    
    extracted_frames = list(output_dir.glob('frame_*.jpg'))
    print(f"✅ Total extracted: {len(extracted_frames)} frames")

def extract_time_based_frames(video_path, output_dir, times):
    """
    Extract frames at specific times.
    
    Args:
        video_path: Path to source video
        output_dir: Directory to save frames
        times: List of times in seconds
    """
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⏰ Extracting frames at times: {times}")
    
    for i, time_sec in enumerate(times):
        try:
            cmd = [
                'ffmpeg', '-ss', str(time_sec), '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',
                str(output_dir / f'frame_t{time_sec:06.1f}s.jpg')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"   ✅ Extracted frame at {time_sec}s")
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to extract frame at {time_sec}s: {e}")
    
    extracted_frames = list(output_dir.glob('frame_*.jpg'))
    print(f"✅ Total extracted: {len(extracted_frames)} frames")

def main():
    parser = argparse.ArgumentParser(
        description="Extract test frames from video for YOLO model testing",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Extract 50 evenly spaced frames
  python extract_test_frames.py /path/to/video.mp4 test_frames --num-frames 50
  
  # Extract frames starting at 30 seconds
  python extract_test_frames.py /path/to/video.mp4 test_frames --start-time 30 --num-frames 20
  
  # Extract specific frame numbers
  python extract_test_frames.py /path/to/video.mp4 test_frames --frame-numbers 100,200,300,400,500
  
  # Extract frames at specific times
  python extract_test_frames.py /path/to/video.mp4 test_frames --times 10.5,25.0,45.2,60.0
        """
    )
    
    parser.add_argument("video_path", type=str, help="Path to the source video file")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted frames")
    
    # Different extraction methods
    parser.add_argument("--num-frames", type=int, default=50, 
                       help="Number of frames to extract (evenly spaced)")
    parser.add_argument("--start-time", type=float, 
                       help="Start time in seconds")
    parser.add_argument("--frame-interval", type=int,
                       help="Extract every Nth frame")
    parser.add_argument("--frame-numbers", type=str,
                       help="Comma-separated list of specific frame numbers to extract")
    parser.add_argument("--times", type=str,
                       help="Comma-separated list of times (in seconds) to extract frames")
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Please install ffmpeg:")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/")
        return
    
    # Choose extraction method
    if args.frame_numbers:
        # Extract specific frame numbers
        frame_numbers = [int(x.strip()) for x in args.frame_numbers.split(',')]
        extract_specific_frames(args.video_path, args.output_dir, frame_numbers)
        
    elif args.times:
        # Extract at specific times
        times = [float(x.strip()) for x in args.times.split(',')]
        extract_time_based_frames(args.video_path, args.output_dir, times)
        
    else:
        # Extract evenly spaced frames
        success = extract_frames_with_ffmpeg(
            args.video_path, 
            args.output_dir, 
            args.num_frames,
            args.start_time,
            args.frame_interval
        )
        
        if not success:
            print("❌ Frame extraction failed")
            return
    
    # Show results
    output_dir = Path(args.output_dir)
    frames = list(output_dir.glob('*.jpg'))
    
    print(f"\n🎉 Frame extraction complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Extracted frames: {len(frames)}")
    print(f"📝 Example frames:")
    for frame in sorted(frames)[:5]:
        print(f"   {frame.name}")
    if len(frames) > 5:
        print(f"   ... and {len(frames) - 5} more")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Test your YOLO model on these frames:")
    print(f"      cd {output_dir}")
    print(f"      yolo predict model=../runs/detect/train16/weights/best.pt source=. save=True")
    print(f"   2. Or use your custom prediction script:")
    print(f"      python yolo_video_predictor.py {args.video_path} runs/detect/train16/weights/best.pt output_video.mp4")

if __name__ == "__main__":
    main()