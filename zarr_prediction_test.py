#!/usr/bin/env python3
"""
YOLO Prediction from Zarr Data
Run YOLO predictions directly on the training data stored in Zarr format.
This eliminates format conversion issues and tests on the exact training data.
"""

import zarr
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from tqdm import tqdm
import tempfile
import os

def load_zarr_data(zarr_path):
    """
    Load training data and tracking results from Zarr file.
    """
    print("📊 Loading Zarr data...")
    
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Check what data is available
        data_info = {}
        
        if 'raw_video/images_ds' in root:
            images_ds = root['raw_video/images_ds']
            data_info['images_ds'] = {
                'shape': images_ds.shape,
                'dtype': images_ds.dtype,
                'description': '640x640 downsampled training images'
            }
        
        if 'raw_video/images_full' in root:
            images_full = root['raw_video/images_full']
            data_info['images_full'] = {
                'shape': images_full.shape,
                'dtype': images_full.dtype,
                'description': '4512x4512 full resolution images'
            }
        
        if 'crop_data/roi_images' in root:
            roi_images = root['crop_data/roi_images']
            data_info['roi_images'] = {
                'shape': roi_images.shape,
                'dtype': roi_images.dtype,
                'description': '320x320 ROI crop images'
            }
        
        if 'tracking/tracking_results' in root:
            tracking_results = root['tracking/tracking_results']
            column_names = tracking_results.attrs.get('column_names', [])
            data_info['tracking_results'] = {
                'shape': tracking_results.shape,
                'columns': len(column_names),
                'column_names': column_names[:10] if len(column_names) > 10 else column_names
            }
        
        print("✅ Available data:")
        for key, info in data_info.items():
            print(f"   📊 {key}: {info}")
        
        return root, data_info
        
    except Exception as e:
        print(f"❌ Error loading Zarr data: {e}")
        return None, None

def get_sample_frames(root, data_source='images_ds', max_samples=20):
    """
    Get sample frames for prediction testing.
    Prioritizes frames that had successful tracking.
    """
    print(f"🎯 Getting sample frames from {data_source}...")
    
    try:
        # Get tracking info to find frames with fish
        if 'tracking/tracking_results' in root:
            tracking_results = root['tracking/tracking_results']
            valid_mask = ~np.isnan(tracking_results[:, 0])  # Valid heading
            valid_indices = np.where(valid_mask)[0]
            
            print(f"✅ Found {len(valid_indices)} frames with successful tracking")
            
            # Sample evenly from valid frames
            if len(valid_indices) > max_samples:
                step = len(valid_indices) // max_samples
                sample_indices = valid_indices[::step][:max_samples]
            else:
                sample_indices = valid_indices
        else:
            # No tracking data, sample evenly from all frames
            total_frames = root[data_source].shape[0]
            step = max(1, total_frames // max_samples)
            sample_indices = np.arange(0, total_frames, step)[:max_samples]
        
        print(f"📊 Selected {len(sample_indices)} frames for testing")
        print(f"🔢 Frame indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
        
        return sample_indices.tolist()
        
    except Exception as e:
        print(f"❌ Error selecting sample frames: {e}")
        return None

def prepare_image_for_yolo(zarr_image, target_size=(640, 640)):
    """
    Convert Zarr image to format suitable for YOLO prediction.
    """
    # Handle different input formats
    if zarr_image.ndim == 2:  # Grayscale
        # Convert to 3-channel RGB for YOLO
        image_rgb = np.stack([zarr_image, zarr_image, zarr_image], axis=-1)
    elif zarr_image.ndim == 3 and zarr_image.shape[-1] == 1:  # Single channel with explicit dim
        zarr_image = zarr_image.squeeze(-1)
        image_rgb = np.stack([zarr_image, zarr_image, zarr_image], axis=-1)
    elif zarr_image.ndim == 3 and zarr_image.shape[-1] == 3:  # Already RGB
        image_rgb = zarr_image
    else:
        raise ValueError(f"Unsupported image shape: {zarr_image.shape}")
    
    # Ensure uint8 dtype
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    
    # Resize if needed
    if image_rgb.shape[:2] != target_size:
        image_rgb = cv2.resize(image_rgb, target_size)
    
    return image_rgb

def run_predictions_on_zarr_data(model_path, root, data_source='images_ds', 
                                sample_indices=None, confidence_levels=[0.5, 0.25, 0.1, 0.05, 0.01]):
    """
    Run YOLO predictions on Zarr data using temporary files.
    """
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"🤖 Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    if sample_indices is None:
        sample_indices = list(range(min(10, root[data_source].shape[0])))
    
    print(f"🎯 Running predictions on {len(sample_indices)} frames from {data_source}")
    
    results_summary = {}
    
    # Create temporary directory for images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for conf_level in confidence_levels:
            print(f"\n🔍 Testing confidence level: {conf_level}")
            results_summary[conf_level] = {
                'detections_found': 0,
                'total_tested': 0,
                'frame_results': []
            }
            
            for i, frame_idx in enumerate(tqdm(sample_indices, desc=f"Conf {conf_level}")):
                try:
                    # Load image from Zarr
                    zarr_image = root[data_source][frame_idx]
                    
                    # Prepare for YOLO
                    yolo_image = prepare_image_for_yolo(zarr_image)
                    
                    # Save temporary image file
                    temp_image_path = temp_path / f"temp_frame_{frame_idx}.jpg"
                    cv2.imwrite(str(temp_image_path), cv2.cvtColor(yolo_image, cv2.COLOR_RGB2BGR))
                    
                    # Run prediction
                    pred_results = model.predict(str(temp_image_path), conf=conf_level, verbose=False)
                    
                    # Process results
                    num_detections = 0
                    detections = []
                    
                    if len(pred_results) > 0 and pred_results[0].boxes is not None:
                        num_detections = len(pred_results[0].boxes)
                        
                        for box in pred_results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class': cls_id
                            })
                    
                    results_summary[conf_level]['frame_results'].append({
                        'frame_idx': frame_idx,
                        'detections': num_detections,
                        'boxes': detections
                    })
                    
                    if num_detections > 0:
                        results_summary[conf_level]['detections_found'] += 1
                        if i < 5:  # Show first few detections
                            max_conf = max(d['confidence'] for d in detections)
                            print(f"   ✅ Frame {frame_idx}: {num_detections} detections (max conf: {max_conf:.3f})")
                    
                    results_summary[conf_level]['total_tested'] += 1
                    
                    # Clean up temp file
                    temp_image_path.unlink()
                    
                except Exception as e:
                    print(f"   ❌ Error processing frame {frame_idx}: {e}")
                    continue
            
            # Summary for this confidence level
            found = results_summary[conf_level]['detections_found']
            total = results_summary[conf_level]['total_tested']
            print(f"   📊 Found detections in {found}/{total} frames ({found/total*100:.1f}%)")
            
            # Stop if we found good results
            if found >= total * 0.5:  # Found detections in >50% of frames
                print(f"   🎉 Great results with confidence {conf_level}!")
                break
    
    return results_summary

def analyze_prediction_results(results_summary):
    """
    Analyze and display prediction results.
    """
    print("\n📊 PREDICTION RESULTS ANALYSIS")
    print("=" * 40)
    
    best_conf = None
    best_detection_rate = 0
    
    for conf_level, results in results_summary.items():
        found = results['detections_found']
        total = results['total_tested']
        detection_rate = found / total if total > 0 else 0
        
        print(f"🎯 Confidence {conf_level}:")
        print(f"   Detections found: {found}/{total} frames ({detection_rate*100:.1f}%)")
        
        if detection_rate > best_detection_rate:
            best_detection_rate = detection_rate
            best_conf = conf_level
        
        # Show some example detections
        if found > 0:
            print(f"   📋 Example detections:")
            shown = 0
            for frame_result in results['frame_results']:
                if frame_result['detections'] > 0 and shown < 3:
                    frame_idx = frame_result['frame_idx']
                    num_det = frame_result['detections']
                    max_conf = max(box['confidence'] for box in frame_result['boxes'])
                    print(f"      Frame {frame_idx}: {num_det} detections (max conf {max_conf:.3f})")
                    shown += 1
        print()
    
    # Overall assessment
    print("🎯 OVERALL ASSESSMENT:")
    if best_detection_rate > 0.5:
        print(f"🎉 SUCCESS! Your model works well with confidence = {best_conf}")
        print(f"   Best detection rate: {best_detection_rate*100:.1f}%")
        print("   This means your model is properly trained and working!")
    elif best_detection_rate > 0.1:
        print(f"⚠️  PARTIAL SUCCESS: Model works with confidence = {best_conf}")
        print(f"   Detection rate: {best_detection_rate*100:.1f}%")
        print("   Model is working but may need tuning or more training")
    else:
        print("😞 NO DETECTIONS FOUND at any confidence level")
        print("   This suggests:")
        print("   1. Model wasn't trained properly")
        print("   2. Severe overfitting to training data")
        print("   3. Data format mismatch (less likely since we're using exact training data)")
    
    return best_conf, best_detection_rate

def create_prediction_visualization(root, model_path, results_summary, 
                                  data_source='images_ds', best_conf=0.25, num_examples=6):
    """
    Create visualization showing predictions on Zarr images.
    """
    print(f"\n🎨 Creating prediction visualization...")
    
    try:
        model = YOLO(model_path)
        
        # Find frames with detections at the best confidence level
        best_results = results_summary.get(best_conf, {})
        frames_with_detections = [
            r for r in best_results.get('frame_results', []) 
            if r['detections'] > 0
        ]
        
        if not frames_with_detections:
            print("❌ No detections found for visualization")
            return
        
        # Select examples
        examples = frames_with_detections[:num_examples]
        
        # Create figure
        rows = (len(examples) + 2) // 3
        cols = min(3, len(examples))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        
        if len(examples) == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, frame_result in enumerate(examples):
                frame_idx = frame_result['frame_idx'] 
                detections = frame_result['boxes']
                
                # Load and prepare image
                zarr_image = root[data_source][frame_idx]
                yolo_image = prepare_image_for_yolo(zarr_image)
                
                # Save and predict
                temp_image_path = temp_path / f"viz_frame_{frame_idx}.jpg"
                cv2.imwrite(str(temp_image_path), cv2.cvtColor(yolo_image, cv2.COLOR_RGB2BGR))
                
                # Run prediction for visualization
                pred_results = model.predict(str(temp_image_path), conf=best_conf, verbose=False)
                
                # Display image
                axes[i].imshow(yolo_image)
                axes[i].set_title(f'Frame {frame_idx}\n{len(detections)} detections', fontsize=12)
                axes[i].axis('off')
                
                # Draw bounding boxes
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    conf = detection['confidence']
                    
                    # Create rectangle
                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                           edgecolor='lime', facecolor='none')
                    axes[i].add_patch(rect)
                    
                    # Add confidence label
                    axes[i].text(x1, y1-5, f'Fish {conf:.3f}', fontsize=10, 
                               color='lime', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(examples), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'YOLO Predictions on Zarr Data (confidence = {best_conf})', 
                     fontsize=16, y=0.98)
        
        # Save visualization
        output_path = 'zarr_predictions_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO predictions directly on Zarr training data",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Test model on downsampled training images
  python zarr_prediction_test.py video.zarr runs/detect/train16/weights/best.pt
  
  # Test on ROI crops instead
  python zarr_prediction_test.py video.zarr best.pt --data-source roi_images
  
  # Test specific frames
  python zarr_prediction_test.py video.zarr best.pt --frame-indices 100,500,1000
        """
    )
    
    parser.add_argument("zarr_path", type=str, help="Path to Zarr training data file")
    parser.add_argument("model_path", type=str, help="Path to trained YOLO model")
    
    parser.add_argument("--data-source", type=str, default="images_ds",
                       choices=["images_ds", "images_full", "roi_images"],
                       help="Which images to use from Zarr (default: images_ds)")
    parser.add_argument("--max-samples", type=int, default=20,
                       help="Maximum number of frames to test")
    parser.add_argument("--frame-indices", type=str,
                       help="Comma-separated specific frame indices to test")
    parser.add_argument("--confidence-levels", type=str, default="0.5,0.25,0.1,0.05,0.01",
                       help="Comma-separated confidence levels to test")
    parser.add_argument("--create-visualization", action='store_true',
                       help="Create visualization of predictions")
    
    args = parser.parse_args()
    
    print("🤖 YOLO PREDICTION ON ZARR DATA")
    print("🎯 Testing your model on the exact training data")
    print("=" * 50)
    
    # Load Zarr data
    root, data_info = load_zarr_data(args.zarr_path)
    if root is None:
        return
    
    # Map data source names to actual Zarr paths
    data_source_mapping = {
        'images_ds': 'raw_video/images_ds',
        'images_full': 'raw_video/images_full', 
        'roi_images': 'crop_data/roi_images'
    }
    
    # Check if requested data source exists
    zarr_path = data_source_mapping.get(args.data_source)
    if zarr_path is None or zarr_path not in root:
        print(f"❌ Data source '{args.data_source}' not found in Zarr file")
        print(f"Available sources: {list(data_source_mapping.keys())}")
        print(f"Zarr paths: {list(data_source_mapping.values())}")
        return
    
    # Get sample frames
    if args.frame_indices:
        sample_indices = [int(x.strip()) for x in args.frame_indices.split(',')]
        print(f"🎯 Using specified frame indices: {sample_indices}")
    else:
        sample_indices = get_sample_frames(root, zarr_path, args.max_samples)
        if sample_indices is None:
            return
    
    # Parse confidence levels
    confidence_levels = [float(x.strip()) for x in args.confidence_levels.split(',')]
    
    # Run predictions
    results = run_predictions_on_zarr_data(
        args.model_path, root, zarr_path, 
        sample_indices, confidence_levels
    )
    
    if results is None:
        return
    
    # Analyze results
    best_conf, best_rate = analyze_prediction_results(results)
    
    # Create visualization if requested
    if args.create_visualization and best_rate > 0:
        create_prediction_visualization(
            root, args.model_path, results, zarr_path, best_conf
        )
    
    print(f"\n🎉 ZARR PREDICTION TEST COMPLETE!")
    if best_rate > 0:
        print(f"✅ Your model IS working with confidence = {best_conf}")
        print(f"💡 The issue with extracted frames was likely format conversion")
        print(f"🔧 Use confidence = {best_conf} for your extracted frame predictions")
    else:
        print(f"❌ Model not detecting anything even on training data")
        print(f"💡 This suggests a fundamental training or model issue")

if __name__ == "__main__":
    main()