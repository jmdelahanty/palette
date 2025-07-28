#!/usr/bin/env python3
"""
Comprehensive Frame Extraction for Model Testing
Extract frames using multiple methods to match training data exactly.
"""

import subprocess
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

def extract_frames_all_methods(video_path, output_dir, frame_numbers=None, max_frames=50):
    """
    Extract frames using all possible methods to find the one that matches training.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 COMPREHENSIVE FRAME EXTRACTION")
    print(f"📁 Video: {video_path}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)
    
    # Get video info
    try:
        info_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        fps = 30  # default
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                fps_str = stream.get('avg_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) > 0 else 30
                break
        
        print(f"📊 Video FPS: {fps:.2f}")
    except:
        fps = 30
        print("⚠️  Could not determine FPS, using 30")
    
    # Define extraction methods that might match decord
    extraction_methods = {
        'bilinear': {
            'description': 'Bilinear scaling (most likely to match decord)',
            'cmd_template': [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', 'scale=640:640:flags=bilinear',
                '-q:v', '2'
            ]
        },
        'lanczos': {
            'description': 'Lanczos scaling (high quality)',
            'cmd_template': [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', 'scale=640:640:flags=lanczos',
                '-q:v', '2'
            ]
        },
        'bicubic': {
            'description': 'Bicubic scaling',
            'cmd_template': [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', 'scale=640:640:flags=bicubic',
                '-q:v', '2'
            ]
        },
        'neighbor': {
            'description': 'Nearest neighbor (fast)',
            'cmd_template': [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', 'scale=640:640:flags=neighbor',
                '-q:v', '2'
            ]
        },
        'exact_frame': {
            'description': 'Exact frame extraction by number',
            'cmd_template': [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', 'select=eq(n\\,FRAME_NUM),scale=640:640:flags=bilinear',
                '-frames:v', '1', '-q:v', '2'
            ]
        },
        'grayscale_bilinear': {
            'description': 'Grayscale with bilinear (closest to training)',
            'cmd_template': [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', 'scale=640:640:flags=bilinear,format=gray',
                '-q:v', '2'
            ]
        }
    }
    
    extracted_results = {}
    
    # If specific frame numbers provided, extract those
    if frame_numbers:
        print(f"🎯 Extracting specific frames: {frame_numbers}")
        
        for method_name, method_info in extraction_methods.items():
            method_dir = output_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            print(f"\n🔧 Method: {method_name} - {method_info['description']}")
            
            extracted_files = []
            
            if method_name == 'exact_frame':
                # Extract exact frame numbers
                for frame_num in frame_numbers:
                    try:
                        cmd = method_info['cmd_template'].copy()
                        # Replace FRAME_NUM placeholder
                        for i, arg in enumerate(cmd):
                            if 'FRAME_NUM' in str(arg):
                                cmd[i] = arg.replace('FRAME_NUM', str(frame_num))
                        
                        output_file = method_dir / f'frame_{frame_num:06d}.jpg'
                        cmd.append(str(output_file))
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        if output_file.exists():
                            extracted_files.append(output_file)
                            print(f"   ✅ Frame {frame_num}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ Frame {frame_num}: {e}")
            else:
                # Extract using time-based seeking
                for frame_num in frame_numbers:
                    try:
                        time_sec = frame_num / fps
                        cmd = [
                            'ffmpeg', '-y', '-ss', str(time_sec), '-i', str(video_path),
                            '-vframes', '1'
                        ] + method_info['cmd_template'][4:]  # Skip the base ffmpeg part
                        
                        output_file = method_dir / f'frame_{frame_num:06d}.jpg'
                        cmd.append(str(output_file))
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        if output_file.exists():
                            extracted_files.append(output_file)
                            print(f"   ✅ Frame {frame_num}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ Frame {frame_num}: {e}")
            
            extracted_results[method_name] = {
                'files': extracted_files,
                'method_dir': method_dir,
                'description': method_info['description']
            }
    
    else:
        # Extract evenly spaced frames
        print(f"📊 Extracting {max_frames} evenly spaced frames")
        
        for method_name, method_info in extraction_methods.items():
            if method_name == 'exact_frame':
                continue  # Skip exact frame method for bulk extraction
                
            method_dir = output_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            print(f"\n🔧 Method: {method_name} - {method_info['description']}")
            
            try:
                # Calculate frame interval for even spacing
                cmd = method_info['cmd_template'].copy()
                cmd.extend(['-frames:v', str(max_frames)])
                cmd.append(str(method_dir / 'frame_%06d.jpg'))
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Count extracted files
                extracted_files = list(method_dir.glob('frame_*.jpg'))
                extracted_results[method_name] = {
                    'files': extracted_files,
                    'method_dir': method_dir,
                    'description': method_info['description']
                }
                
                print(f"   ✅ Extracted {len(extracted_files)} frames")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed: {e}")
                extracted_results[method_name] = {
                    'files': [],
                    'method_dir': method_dir,
                    'description': method_info['description'],
                    'error': str(e)
                }
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    for method_name, result in extracted_results.items():
        if 'error' in result:
            print(f"   ❌ {method_name}: Failed - {result['error']}")
        else:
            print(f"   ✅ {method_name}: {len(result['files'])} frames")
    
    return extracted_results

def test_all_methods_with_model(model_path, extracted_results, confidence_levels=[0.5, 0.25, 0.1, 0.05]):
    """
    Test the model on all extracted frame methods to find which works best.
    """
    print(f"\n🤖 TESTING MODEL ON ALL EXTRACTION METHODS")
    print(f"🎯 Model: {model_path}")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return None
    
    method_results = {}
    
    for method_name, extraction_result in extracted_results.items():
        if 'error' in extraction_result or not extraction_result['files']:
            print(f"\n❌ Skipping {method_name} - no frames available")
            continue
        
        print(f"\n🔧 Testing method: {method_name}")
        print(f"📋 Description: {extraction_result['description']}")
        print(f"📊 Files: {len(extraction_result['files'])}")
        
        method_results[method_name] = {}
        
        # Test each confidence level
        for conf in confidence_levels:
            print(f"   🎯 Confidence {conf}:")
            
            detections_found = 0
            total_tested = len(extraction_result['files'])
            all_confidences = []
            
            for frame_file in extraction_result['files'][:10]:  # Test first 10 frames
                try:
                    results = model.predict(str(frame_file), conf=conf, verbose=False)
                    
                    if len(results) > 0 and results[0].boxes is not None:
                        num_detections = len(results[0].boxes)
                        if num_detections > 0:
                            detections_found += 1
                            # Get confidence scores
                            for box in results[0].boxes:
                                conf_score = float(box.conf[0].cpu().numpy())
                                all_confidences.append(conf_score)
                
                except Exception as e:
                    print(f"      ❌ Error testing {frame_file.name}: {e}")
            
            detection_rate = detections_found / min(10, total_tested)
            method_results[method_name][conf] = {
                'detection_rate': detection_rate,
                'detections_found': detections_found,
                'total_tested': min(10, total_tested),
                'confidence_scores': all_confidences
            }
            
            print(f"      📊 Detection rate: {detections_found}/{min(10, total_tested)} ({detection_rate*100:.1f}%)")
            if all_confidences:
                print(f"      📈 Avg confidence: {np.mean(all_confidences):.3f}")
    
    # Find best method
    print(f"\n🏆 BEST METHODS RANKING:")
    
    best_methods = []
    for method_name, conf_results in method_results.items():
        for conf, results in conf_results.items():
            if results['detection_rate'] > 0:
                best_methods.append({
                    'method': method_name,
                    'confidence': conf,
                    'detection_rate': results['detection_rate'],
                    'avg_confidence': np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
                })
    
    # Sort by detection rate
    best_methods.sort(key=lambda x: x['detection_rate'], reverse=True)
    
    for i, result in enumerate(best_methods[:5]):  # Top 5
        print(f"   {i+1}. {result['method']} (conf={result['confidence']}): "
              f"{result['detection_rate']*100:.1f}% detection rate")
    
    if best_methods:
        best = best_methods[0]
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   Use method: {best['method']}")
        print(f"   With confidence: {best['confidence']}")
        print(f"   Expected detection rate: {best['detection_rate']*100:.1f}%")
        
        # Give specific ffmpeg command
        method_info = extracted_results[best['method']]
        print(f"\n🔧 FFMPEG COMMAND:")
        if best['method'] == 'bilinear':
            print(f"   ffmpeg -i video.mp4 -vf 'scale=640:640:flags=bilinear' -q:v 2 output_%06d.jpg")
        elif best['method'] == 'grayscale_bilinear':
            print(f"   ffmpeg -i video.mp4 -vf 'scale=640:640:flags=bilinear,format=gray' -q:v 2 output_%06d.jpg")
        # Add more method-specific commands as needed
    else:
        print(f"\n😞 NO WORKING METHOD FOUND")
        print(f"   This suggests a fundamental difference between training and inference")
        print(f"   Your model works on Zarr data but not extracted frames")
    
    return method_results

def main():
    parser = argparse.ArgumentParser(description="Comprehensive frame extraction and model testing")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("model_path", type=str, help="Path to YOLO model")
    parser.add_argument("output_dir", type=str, help="Output directory")
    
    parser.add_argument("--frame-numbers", type=str, 
                       help="Comma-separated frame numbers (e.g., 1000,2000,3000)")
    parser.add_argument("--max-frames", type=int, default=20,
                       help="Max frames to extract if no specific numbers given")
    parser.add_argument("--confidence-levels", type=str, default="0.5,0.25,0.1,0.05",
                       help="Comma-separated confidence levels to test")
    
    args = parser.parse_args()
    
    # Parse frame numbers
    frame_numbers = None
    if args.frame_numbers:
        frame_numbers = [int(x.strip()) for x in args.frame_numbers.split(',')]
    
    # Parse confidence levels
    confidence_levels = [float(x.strip()) for x in args.confidence_levels.split(',')]
    
    print("🎬 COMPREHENSIVE FRAME EXTRACTION & MODEL TESTING")
    print("🎯 Goal: Find the extraction method that matches your training data")
    print()
    
    # Extract frames using all methods
    extracted_results = extract_frames_all_methods(
        args.video_path, args.output_dir, frame_numbers, args.max_frames
    )
    
    # Test model on all methods
    model_results = test_all_methods_with_model(
        args.model_path, extracted_results, confidence_levels
    )
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Extracted frames in: {args.output_dir}")

if __name__ == "__main__":
    main()