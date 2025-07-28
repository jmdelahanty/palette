#!/usr/bin/env python3
"""
Extract frames that exactly match your training data format and characteristics.
"""

import subprocess
import numpy as np
import cv2
import zarr
import argparse
from pathlib import Path
import json

def get_training_data_info(zarr_path):
    """
    Extract information about the training data format from the Zarr file.
    """
    print("📊 Analyzing training data format...")
    
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Get training image properties
        if 'raw_video/images_ds' in root:
            training_images = root['raw_video/images_ds']
            
            # Sample a few images to understand the data
            sample_img = training_images[0]
            
            info = {
                'shape': sample_img.shape,  # Should be (640, 640)
                'dtype': str(sample_img.dtype),
                'pixel_range': (int(sample_img.min()), int(sample_img.max())),
                'mean_intensity': float(sample_img.mean()),
                'total_frames': training_images.shape[0]
            }
            
            print(f"✅ Training data info:")
            print(f"   📐 Shape: {info['shape']}")
            print(f"   🔢 Dtype: {info['dtype']}")
            print(f"   📈 Pixel range: {info['pixel_range']}")
            print(f"   📊 Mean intensity: {info['mean_intensity']:.1f}")
            print(f"   🎬 Total frames: {info['total_frames']}")
            
            return info
        else:
            print("❌ No training images found in Zarr file")
            return None
            
    except Exception as e:
        print(f"❌ Error reading Zarr file: {e}")
        return None

def get_valid_frame_indices(zarr_path, max_frames=50):
    """
    Get frame indices that had successful tracking (these are most likely to have fish).
    """
    print("🎯 Finding frames with successful tracking...")
    
    try:
        root = zarr.open(zarr_path, mode='r')
        tracking_results = root['tracking/tracking_results']
        
        # Find frames with valid tracking (non-NaN heading)
        valid_mask = ~np.isnan(tracking_results[:, 0])
        valid_indices = np.where(valid_mask)[0]
        
        print(f"✅ Found {len(valid_indices)} frames with successful tracking")
        
        # Select a subset, spread across the video
        if len(valid_indices) > max_frames:
            # Sample evenly across the valid frames
            step = len(valid_indices) // max_frames
            selected_indices = valid_indices[::step][:max_frames]
        else:
            selected_indices = valid_indices
        
        print(f"📊 Selected {len(selected_indices)} frames for extraction")
        print(f"🔢 Frame indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        
        return selected_indices.tolist()
        
    except Exception as e:
        print(f"❌ Error finding valid frames: {e}")
        return None

def extract_frames(video_path, output_dir, frame_indices=None, target_size=(640, 640)):
    """
    Extract frames from video at specific indices, matching training data format.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    print(f"🎬 Extracting frames from: {video_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # First, get video info
        info_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', str(video_path)
        ]
        
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            print("❌ No video stream found")
            return False
        
        fps = eval(video_stream.get('avg_frame_rate', '30/1'))  # Convert fraction to float
        duration = float(video_stream.get('duration', 0))
        total_frames = int(video_stream.get('nb_frames', 0)) or int(fps * duration)
        
        print(f"📊 Video info: {fps:.2f} fps, {duration:.1f}s, ~{total_frames} frames")
        
        # Extract frames
        if frame_indices:
            # Extract specific frame indices
            print(f"🎯 Extracting {len(frame_indices)} specific frames...")
            
            for i, frame_idx in enumerate(frame_indices):
                if frame_idx >= total_frames:
                    print(f"⚠️  Skipping frame {frame_idx} (beyond video length)")
                    continue
                
                # Calculate time for this frame
                time_sec = frame_idx / fps
                
                output_file = output_dir / f"frame_{frame_idx:06d}.jpg"
                
                cmd = [
                    'ffmpeg', '-y',  # Overwrite existing files
                    '-ss', str(time_sec),  # Seek to specific time
                    '-i', str(video_path),
                    '-vframes', '1',  # Extract 1 frame
                    '-vf', f'scale={target_size[0]}:{target_size[1]}',  # Resize to 640x640
                    '-q:v', '2',  # High quality
                    str(output_file)
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    if i < 5 or i % 10 == 0:  # Show progress
                        print(f"   ✅ Extracted frame {frame_idx} -> {output_file.name}")
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Failed to extract frame {frame_idx}: {e}")
                    continue
        
        else:
            # Extract evenly spaced frames
            num_frames = min(50, total_frames // 10)  # Extract ~50 frames
            print(f"📊 Extracting {num_frames} evenly spaced frames...")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f'scale={target_size[0]}:{target_size[1]},fps=1/{total_frames//num_frames}',
                '-q:v', '2',
                str(output_dir / 'frame_%04d.jpg')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count extracted frames
        extracted_frames = list(output_dir.glob('frame_*.jpg'))
        print(f"✅ Successfully extracted {len(extracted_frames)} frames")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg error: {e}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_extracted_frames(frames_dir, training_info):
    """
    Validate that extracted frames match the training data format.
    """
    print("🔍 Validating extracted frames...")
    
    frames_dir = Path(frames_dir)
    frame_files = list(frames_dir.glob('frame_*.jpg'))
    
    if not frame_files:
        print("❌ No extracted frames found")
        return False
    
    print(f"📊 Found {len(frame_files)} frames to validate")
    
    # Check a few sample frames
    sample_frames = frame_files[:3]
    all_valid = True
    
    for frame_path in sample_frames:
        try:
            # Load as grayscale (single channel like training data)
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"   ❌ Could not load {frame_path.name}")
                all_valid = False
                continue
            
            # Check properties
            shape_ok = img.shape == training_info['shape']
            dtype_ok = str(img.dtype) == training_info['dtype']
            
            # Check pixel range (should be similar)
            pixel_range = (int(img.min()), int(img.max()))
            range_ok = (abs(pixel_range[0] - training_info['pixel_range'][0]) < 50 and
                       abs(pixel_range[1] - training_info['pixel_range'][1]) < 50)
            
            print(f"   {'✅' if shape_ok else '❌'} {frame_path.name}: shape {img.shape} {'==' if shape_ok else '!='} {training_info['shape']}")
            print(f"   {'✅' if dtype_ok else '❌'} dtype: {img.dtype} {'==' if dtype_ok else '!='} {training_info['dtype']}")
            print(f"   {'✅' if range_ok else '❌'} range: {pixel_range} {'~=' if range_ok else '!='} {training_info['pixel_range']}")
            
            if not (shape_ok and dtype_ok):
                all_valid = False
                
        except Exception as e:
            print(f"   ❌ Error validating {frame_path.name}: {e}")
            all_valid = False
    
    if all_valid:
        print("✅ All frames match training data format!")
    else:
        print("⚠️  Some frames don't match training format")
    
    return all_valid

def test_model_on_extracted_frames(model_path, frames_dir, num_test=5):
    """
    Test the YOLO model on extracted frames with various confidence levels.
    """
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"🤖 Testing model on extracted frames...")
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        frames_dir = Path(frames_dir)
        frame_files = sorted(list(frames_dir.glob('frame_*.jpg')))[:num_test]
        
        if not frame_files:
            print("❌ No frames found for testing")
            return
        
        print(f"🎯 Testing on {len(frame_files)} frames...")
        
        confidence_levels = [0.5, 0.25, 0.1, 0.05, 0.01]
        
        for conf in confidence_levels:
            print(f"\n🔍 Testing with confidence = {conf}")
            detections_found = 0
            
            for frame_path in frame_files:
                try:
                    results = model.predict(str(frame_path), conf=conf, verbose=False)
                    
                    if len(results) > 0 and results[0].boxes is not None:
                        num_detections = len(results[0].boxes)
                        if num_detections > 0:
                            detections_found += 1
                            highest_conf = max(float(box.conf[0]) for box in results[0].boxes)
                            print(f"   ✅ {frame_path.name}: {num_detections} detections (max conf: {highest_conf:.3f})")
                        
                except Exception as e:
                    print(f"   ❌ Error testing {frame_path.name}: {e}")
            
            print(f"   📊 Found detections in {detections_found}/{len(frame_files)} frames")
            
            if detections_found > 0:
                print(f"   🎉 SUCCESS! Your model works with confidence = {conf}")
                break
        else:
            print("   😞 No detections found at any confidence level")
            print("   💡 This suggests either:")
            print("      1. No fish visible in these frames")
            print("      2. Model needs retraining")
            print("      3. Frame format still doesn't match training")
        
    except ImportError:
        print("❌ ultralytics not available - skipping model test")
    except Exception as e:
        print(f"❌ Model test error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames matching your YOLO training data format",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Extract frames from successful tracking indices
  python extract_matching_frames.py video.zarr ~/Desktop/video.mp4 extracted_frames/ --model best.pt
  
  # Extract specific frame numbers
  python extract_matching_frames.py video.zarr ~/Desktop/video.mp4 extracted_frames/ --frame-indices 100,500,1000,2000
        """
    )
    
    parser.add_argument("zarr_path", type=str, help="Path to training data Zarr file")
    parser.add_argument("video_path", type=str, help="Path to source video file")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted frames")
    
    parser.add_argument("--model-path", type=str, help="Path to trained YOLO model for testing")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum frames to extract")
    parser.add_argument("--frame-indices", type=str, help="Comma-separated frame indices to extract")
    parser.add_argument("--target-size", type=str, default="640,640", help="Target frame size (width,height)")
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    print("🎬 FRAME EXTRACTION FOR YOLO PREDICTION")
    print("🎯 Extracting frames that match your training data exactly")
    print("=" * 60)
    
    # 1. Analyze training data
    training_info = get_training_data_info(args.zarr_path)
    if not training_info:
        print("❌ Could not analyze training data")
        return
    
    # 2. Get frame indices to extract
    if args.frame_indices:
        frame_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
        print(f"🎯 Using specified frame indices: {frame_indices}")
    else:
        frame_indices = get_valid_frame_indices(args.zarr_path, args.max_frames)
        if not frame_indices:
            print("❌ Could not find valid frame indices")
            return
    
    # 3. Extract frames
    success = extract_frames(args.video_path, args.output_dir, frame_indices, target_size)
    if not success:
        print("❌ Frame extraction failed")
        return
    
    # 4. Validate extracted frames
    validate_extracted_frames(args.output_dir, training_info)
    
    # 5. Test model if provided
    if args.model_path:
        test_model_on_extracted_frames(args.model_path, args.output_dir)
    
    print(f"\n🎉 EXTRACTION COMPLETE!")
    print(f"📁 Frames saved to: {args.output_dir}")
    print(f"🚀 Next steps:")
    print(f"   1. Test with your model: yolo predict model={args.model_path or 'best.pt'} source={args.output_dir} conf=0.1")
    print(f"   2. Visualize results: yolo predict model={args.model_path or 'best.pt'} source={args.output_dir} show=True")

if __name__ == "__main__":
    main()