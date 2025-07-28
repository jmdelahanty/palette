#!/usr/bin/env python3
"""
Decord vs FFmpeg Frame Decoder Comparison
Compare how decord and ffmpeg decode the same video frames to identify differences.
"""

import cv2
import numpy as np
import subprocess
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
import decord
import torch
import torch.nn.functional as F

def analyze_video_properties(video_path):
    """
    Get detailed video properties using ffprobe.
    """
    print("📊 Analyzing video properties...")
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', '-show_frames', '-select_streams', 'v:0',
            '-read_intervals', '%+#1',  # Just read first frame for detailed info
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream:
            properties = {
                'codec': video_stream.get('codec_name'),
                'profile': video_stream.get('profile'),
                'level': video_stream.get('level'),
                'pix_fmt': video_stream.get('pix_fmt'),
                'color_space': video_stream.get('color_space'),
                'color_transfer': video_stream.get('color_transfer'),
                'color_primaries': video_stream.get('color_primaries'),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('avg_frame_rate', '0/1')),
                'bit_depth': video_stream.get('bits_per_raw_sample'),
                'chroma_location': video_stream.get('chroma_location')
            }
            
            print("✅ Video properties:")
            for key, value in properties.items():
                print(f"   {key}: {value}")
            
            return properties
        else:
            print("❌ No video stream found")
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing video: {e}")
        return None

def extract_frame_with_ffmpeg(video_path, frame_number, output_path, target_size=(640, 640)):
    """
    Extract a specific frame using ffmpeg with various processing options.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # Get video info first
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
    except:
        fps = 30
    
    # Calculate time for the frame
    time_sec = frame_number / fps
    
    extraction_methods = {
        'default': [
            'ffmpeg', '-y', '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', f'scale={target_size[0]}:{target_size[1]}',
            '-q:v', '2', str(output_path / 'ffmpeg_default.jpg')
        ],
        'lanczos': [
            'ffmpeg', '-y', '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', f'scale={target_size[0]}:{target_size[1]}:flags=lanczos',
            '-q:v', '2', str(output_path / 'ffmpeg_lanczos.jpg')
        ],
        'bilinear': [
            'ffmpeg', '-y', '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', f'scale={target_size[0]}:{target_size[1]}:flags=bilinear',
            '-q:v', '2', str(output_path / 'ffmpeg_bilinear.jpg')
        ],
        'neighbor': [
            'ffmpeg', '-y', '-ss', str(time_sec), '-i', str(video_path),
            '-vframes', '1', '-vf', f'scale={target_size[0]}:{target_size[1]}:flags=neighbor',
            '-q:v', '2', str(output_path / 'ffmpeg_neighbor.jpg')
        ],
        'full_decode': [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', f'select=eq(n\\,{frame_number}),scale={target_size[0]}:{target_size[1]}',
            '-frames:v', '1', '-q:v', '2', str(output_path / 'ffmpeg_full_decode.jpg')
        ]
    }
    
    print(f"🎬 Extracting frame {frame_number} using FFmpeg methods...")
    
    extracted_files = {}
    for method_name, cmd in extraction_methods.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_file = output_path / f'ffmpeg_{method_name}.jpg'
            if output_file.exists():
                extracted_files[method_name] = output_file
                print(f"   ✅ {method_name}: {output_file.name}")
            else:
                print(f"   ❌ {method_name}: Failed to create output file")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {method_name}: {e}")
    
    return extracted_files

def extract_frame_with_decord(video_path, frame_number, output_path, target_size=(640, 640)):
    """
    Extract the same frame using decord (matching your training pipeline).
    """
    print(f"🎯 Extracting frame {frame_number} using Decord (training method)...")
    
    try:
        # Try GPU first, fallback to CPU if memory issues
        try:
            # Set up decord exactly like your training pipeline
            decord.bridge.set_bridge('torch')
            vr = decord.VideoReader(str(video_path), ctx=decord.gpu(0))
            device = 'cuda:0'
            print("   ✅ Using GPU for decord extraction")
        except Exception as gpu_error:
            print(f"   ⚠️  GPU failed ({gpu_error}), falling back to CPU")
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            device = 'cpu'
        
        # Get the frame
        frame = vr[frame_number]  # Shape: (H, W, C)
        
        # Move to device and convert to grayscale exactly like training pipeline
        if device == 'cuda:0':
            frame_device = frame
            gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
        else:
            frame_device = torch.from_numpy(frame.asnumpy()) if hasattr(frame, 'asnumpy') else frame
            gray_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        
        gray_frame = (frame_device.float() @ gray_weights)  # Shape: (H, W)
        
        # Downsample exactly like training pipeline
        gray_frame_4d = gray_frame.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        ds_frame_tensor = F.interpolate(gray_frame_4d, size=target_size, mode='bilinear', align_corners=False)
        ds_frame = ds_frame_tensor.squeeze().byte().cpu().numpy()  # Shape: (640, 640)
        
        # Save as grayscale
        output_file = output_path / 'decord_training_method.jpg'
        # Convert grayscale to RGB for saving as JPEG
        ds_frame_rgb = np.stack([ds_frame, ds_frame, ds_frame], axis=-1)
        cv2.imwrite(str(output_file), cv2.cvtColor(ds_frame_rgb, cv2.COLOR_RGB2BGR))
        
        print(f"   ✅ decord training method: {output_file.name}")
        
        # Also save the original color frame for comparison
        if device == 'cuda:0':
            original_frame = frame_device.cpu().numpy()
        else:
            original_frame = frame_device.numpy() if hasattr(frame_device, 'numpy') else frame_device
        
        original_resized = cv2.resize(original_frame, target_size)
        original_file = output_path / 'decord_original_color.jpg'
        cv2.imwrite(str(original_file), cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR))
        
        print(f"   ✅ decord original color: {original_file.name}")
        
        return {
            'training_method': output_file,
            'original_color': original_file,
            'raw_gray_array': ds_frame,
            'raw_color_array': original_resized
        }
        
    except Exception as e:
        print(f"   ❌ Decord extraction failed: {e}")
        return None

def compare_extracted_frames(ffmpeg_files, decord_files, output_path):
    """
    Compare the extracted frames pixel by pixel and visually.
    """
    print("\n🔍 Comparing extracted frames...")
    
    output_path = Path(output_path)
    
    # Load decord reference (training method)
    if 'training_method' not in decord_files:
        print("❌ No decord training method frame available")
        return
    
    decord_img = cv2.imread(str(decord_files['training_method']), cv2.IMREAD_GRAYSCALE)
    if decord_img is None:
        print("❌ Could not load decord reference image")
        return
    
    print(f"📊 Decord reference: shape={decord_img.shape}, dtype={decord_img.dtype}, range=[{decord_img.min()}, {decord_img.max()}]")
    
    # Compare with each ffmpeg method
    comparison_results = {}
    
    for method_name, ffmpeg_file in ffmpeg_files.items():
        try:
            ffmpeg_img = cv2.imread(str(ffmpeg_file), cv2.IMREAD_GRAYSCALE)
            if ffmpeg_img is None:
                print(f"   ❌ Could not load {method_name} image")
                continue
            
            # Ensure same size
            if ffmpeg_img.shape != decord_img.shape:
                ffmpeg_img = cv2.resize(ffmpeg_img, (decord_img.shape[1], decord_img.shape[0]))
            
            # Calculate differences
            pixel_diff = np.abs(ffmpeg_img.astype(np.float32) - decord_img.astype(np.float32))
            
            comparison_results[method_name] = {
                'mean_absolute_error': float(np.mean(pixel_diff)),
                'max_absolute_error': float(np.max(pixel_diff)),
                'pixel_range': [int(ffmpeg_img.min()), int(ffmpeg_img.max())],
                'identical_pixels': float(np.mean(pixel_diff == 0)),
                'ssim_score': calculate_ssim(decord_img, ffmpeg_img)
            }
            
            print(f"   📊 {method_name}:")
            print(f"      MAE: {comparison_results[method_name]['mean_absolute_error']:.3f}")
            print(f"      Max diff: {comparison_results[method_name]['max_absolute_error']:.0f}")
            print(f"      Identical pixels: {comparison_results[method_name]['identical_pixels']*100:.1f}%")
            print(f"      SSIM: {comparison_results[method_name]['ssim_score']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error comparing {method_name}: {e}")
    
    # Create visual comparison
    create_visual_comparison(decord_files, ffmpeg_files, comparison_results, output_path)
    
    return comparison_results

def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    """
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2)
    except ImportError:
        # Simple correlation-based similarity if skimage not available
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Normalize
        img1_norm = (img1_f - np.mean(img1_f)) / (np.std(img1_f) + 1e-8)
        img2_norm = (img2_f - np.mean(img2_f)) / (np.std(img2_f) + 1e-8)
        
        # Calculate correlation
        correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

def create_visual_comparison(decord_files, ffmpeg_files, comparison_results, output_path):
    """
    Create a visual comparison plot of all extraction methods.
    """
    print("\n🎨 Creating visual comparison...")
    
    # Determine grid size
    total_images = 1 + len(ffmpeg_files)  # decord + ffmpeg methods
    cols = min(4, total_images)
    rows = (total_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot decord reference
    decord_img = cv2.imread(str(decord_files['training_method']), cv2.IMREAD_GRAYSCALE)
    axes[0].imshow(decord_img, cmap='gray')
    axes[0].set_title('Decord (Training Method)\nReference Image', fontweight='bold', color='blue')
    axes[0].axis('off')
    
    # Plot ffmpeg methods
    plot_idx = 1
    for method_name, ffmpeg_file in ffmpeg_files.items():
        if plot_idx >= len(axes):
            break
            
        try:
            ffmpeg_img = cv2.imread(str(ffmpeg_file), cv2.IMREAD_GRAYSCALE)
            if ffmpeg_img is not None:
                if ffmpeg_img.shape != decord_img.shape:
                    ffmpeg_img = cv2.resize(ffmpeg_img, (decord_img.shape[1], decord_img.shape[0]))
                
                axes[plot_idx].imshow(ffmpeg_img, cmap='gray')
                
                # Add comparison stats to title
                if method_name in comparison_results:
                    stats = comparison_results[method_name]
                    title = f'FFmpeg ({method_name})\nMAE: {stats["mean_absolute_error"]:.1f}, SSIM: {stats["ssim_score"]:.3f}'
                    color = 'green' if stats["mean_absolute_error"] < 5 else 'orange' if stats["mean_absolute_error"] < 20 else 'red'
                else:
                    title = f'FFmpeg ({method_name})'
                    color = 'black'
                
                axes[plot_idx].set_title(title, color=color)
                axes[plot_idx].axis('off')
        except Exception as e:
            axes[plot_idx].text(0.5, 0.5, f'Error loading\n{method_name}', 
                               ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].set_title(f'FFmpeg ({method_name}) - Error')
            axes[plot_idx].axis('off')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Decord vs FFmpeg Frame Extraction Comparison', fontsize=16, y=0.98)
    
    # Save comparison plot
    comparison_plot_path = output_path / 'decoder_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visual comparison saved to: {comparison_plot_path}")
    plt.show()

def recommend_best_ffmpeg_method(comparison_results):
    """
    Recommend the best FFmpeg method based on comparison results.
    """
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if not comparison_results:
        print("❌ No comparison results available")
        return
    
    # Find the method with lowest mean absolute error
    best_method = min(comparison_results.keys(), 
                     key=lambda x: comparison_results[x]['mean_absolute_error'])
    best_mae = comparison_results[best_method]['mean_absolute_error']
    best_ssim = comparison_results[best_method]['ssim_score']
    
    print(f"🏆 BEST MATCHING FFmpeg method: {best_method}")
    print(f"   Mean Absolute Error: {best_mae:.3f}")
    print(f"   SSIM Score: {best_ssim:.4f}")
    print(f"   Identical pixels: {comparison_results[best_method]['identical_pixels']*100:.1f}%")
    
    # Provide specific FFmpeg command
    if best_mae < 5:
        print(f"✅ EXCELLENT match! Use this FFmpeg command:")
        if best_method == 'default':
            print(f"   ffmpeg -ss TIME -i video.mp4 -vframes 1 -vf 'scale=640:640' -q:v 2 output.jpg")
        elif best_method == 'bilinear':
            print(f"   ffmpeg -ss TIME -i video.mp4 -vframes 1 -vf 'scale=640:640:flags=bilinear' -q:v 2 output.jpg")
        elif best_method == 'lanczos':
            print(f"   ffmpeg -ss TIME -i video.mp4 -vframes 1 -vf 'scale=640:640:flags=lanczos' -q:v 2 output.jpg")
        elif best_method == 'full_decode':
            print(f"   ffmpeg -i video.mp4 -vf 'select=eq(n\\,FRAME_NUM),scale=640:640' -frames:v 1 -q:v 2 output.jpg")
    elif best_mae < 20:
        print(f"⚠️  GOOD match, but some differences remain")
        print(f"   This might explain why your extracted frames don't work perfectly")
    else:
        print(f"❌ POOR match - significant differences found")
        print(f"   The decoding differences are likely causing your prediction issues")
    
    print(f"\n🔧 For your use case:")
    print(f"   1. Use the '{best_method}' FFmpeg method")
    print(f"   2. Convert to GRAYSCALE after extraction (your model was trained on grayscale)")
    print(f"   3. Use confidence threshold 0.25 (what we found works)")

def main():
    parser = argparse.ArgumentParser(
        description="Compare decord vs ffmpeg frame extraction methods",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--frame-number", type=int, default=1000, 
                       help="Frame number to extract for comparison (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="decoder_comparison",
                       help="Output directory for comparison files")
    parser.add_argument("--target-size", type=str, default="640,640",
                       help="Target frame size (default: 640,640)")
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎬 DECORD vs FFMPEG FRAME EXTRACTION COMPARISON")
    print(f"📁 Video: {args.video_path}")
    print(f"🎯 Frame: {args.frame_number}")
    print(f"📐 Target size: {target_size}")
    print("=" * 60)
    
    # Analyze video properties
    video_props = analyze_video_properties(args.video_path)
    
    # Extract frame with both methods
    ffmpeg_files = extract_frame_with_ffmpeg(args.video_path, args.frame_number, output_dir, target_size)
    decord_files = extract_frame_with_decord(args.video_path, args.frame_number, output_dir, target_size)
    
    if not ffmpeg_files or not decord_files:
        print("❌ Frame extraction failed")
        return
    
    # Compare the results
    comparison_results = compare_extracted_frames(ffmpeg_files, decord_files, output_dir)
    
    # Provide recommendations
    recommend_best_ffmpeg_method(comparison_results)
    
    print(f"\n🎉 COMPARISON COMPLETE!")
    print(f"📁 Results saved in: {output_dir}")

if __name__ == "__main__":
    main()