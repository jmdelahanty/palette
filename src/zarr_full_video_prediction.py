#!/usr/bin/env python3
"""
Full Video YOLO Prediction from Zarr
Run YOLO predictions on ALL frames in the Zarr data and save comprehensive results.
"""

import zarr
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import tempfile
import os
import json

def run_full_video_predictions(model_path, zarr_path, data_source='images_ds', 
                              confidence=0.25, batch_size=32, output_dir='predictions_output'):
    """
    Run YOLO predictions on ALL frames in the Zarr data.
    """
    print(f"🎬 FULL VIDEO PREDICTION")
    print(f"📁 Zarr: {zarr_path}")
    print(f"🤖 Model: {model_path}")
    print(f"🎯 Confidence: {confidence}")
    print(f"📊 Data source: {data_source}")
    print("=" * 60)
    
    # Load Zarr data
    try:
        root = zarr.open(zarr_path, mode='r')
        print("✅ Zarr data loaded")
    except Exception as e:
        print(f"❌ Error loading Zarr: {e}")
        return None
    
    # Map data source to actual path
    data_source_mapping = {
        'images_ds': 'raw_video/images_ds',
        'images_full': 'raw_video/images_full', 
        'roi_images': 'crop_data/roi_images'
    }
    
    zarr_data_path = data_source_mapping.get(data_source)
    if zarr_data_path is None or zarr_data_path not in root:
        print(f"❌ Data source '{data_source}' not found")
        return None
    
    images_array = root[zarr_data_path]
    total_frames = images_array.shape[0]
    print(f"📊 Total frames to process: {total_frames}")
    
    # Get tracking data for context
    tracking_results = None
    if 'tracking/tracking_results' in root:
        tracking_results = root['tracking/tracking_results']
        valid_tracking_mask = ~np.isnan(tracking_results[:, 0])
        valid_tracking_count = np.sum(valid_tracking_mask)
        print(f"📈 Frames with valid tracking: {valid_tracking_count}/{total_frames} ({valid_tracking_count/total_frames*100:.1f}%)")
    
    # Load YOLO model
    try:
        model = YOLO(model_path)
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize results storage
    all_results = []
    detection_stats = {
        'total_frames': total_frames,
        'frames_with_detections': 0,
        'total_detections': 0,
        'confidence_scores': [],
        'detection_by_frame': np.zeros(total_frames, dtype=bool)
    }
    
    # Process in batches to manage memory
    print(f"🔄 Processing {total_frames} frames in batches of {batch_size}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for batch_start in tqdm(range(0, total_frames, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_indices = range(batch_start, batch_end)
            
            # Process batch
            batch_results = []
            
            for frame_idx in batch_indices:
                try:
                    # Load image from Zarr
                    zarr_image = images_array[frame_idx]
                    
                    # Prepare for YOLO (convert grayscale to RGB)
                    if zarr_image.ndim == 2:  # Grayscale
                        yolo_image = np.stack([zarr_image, zarr_image, zarr_image], axis=-1)
                    else:
                        yolo_image = zarr_image
                    
                    # Ensure uint8 and correct size
                    if yolo_image.dtype != np.uint8:
                        yolo_image = yolo_image.astype(np.uint8)
                    
                    if yolo_image.shape[:2] != (640, 640):
                        yolo_image = cv2.resize(yolo_image, (640, 640))
                    
                    # Save temporary image
                    temp_image_path = temp_path / f"temp_{frame_idx}.jpg"
                    cv2.imwrite(str(temp_image_path), cv2.cvtColor(yolo_image, cv2.COLOR_RGB2BGR))
                    
                    # Run prediction
                    pred_results = model.predict(str(temp_image_path), conf=confidence, verbose=False)
                    
                    # Process results
                    frame_detections = []
                    if len(pred_results) > 0 and pred_results[0].boxes is not None:
                        for box in pred_results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            
                            frame_detections.append({
                                'frame_idx': frame_idx,
                                'x1': float(x1), 'y1': float(y1), 
                                'x2': float(x2), 'y2': float(y2),
                                'confidence': conf,
                                'class_id': cls_id,
                                'center_x': float((x1 + x2) / 2),
                                'center_y': float((y1 + y2) / 2),
                                'width': float(x2 - x1),
                                'height': float(y2 - y1)
                            })
                    
                    # Store frame result
                    frame_result = {
                        'frame_idx': frame_idx,
                        'num_detections': len(frame_detections),
                        'detections': frame_detections,
                        'has_valid_tracking': False
                    }
                    
                    # Add tracking context if available
                    if tracking_results is not None:
                        frame_result['has_valid_tracking'] = bool(valid_tracking_mask[frame_idx])
                        if frame_result['has_valid_tracking']:
                            tracking_data = tracking_results[frame_idx]
                            frame_result['tracking_heading'] = float(tracking_data[0])
                            # Add more tracking data as needed
                    
                    batch_results.append(frame_result)
                    
                    # Update statistics
                    if len(frame_detections) > 0:
                        detection_stats['frames_with_detections'] += 1
                        detection_stats['total_detections'] += len(frame_detections)
                        detection_stats['detection_by_frame'][frame_idx] = True
                        detection_stats['confidence_scores'].extend([d['confidence'] for d in frame_detections])
                    
                    # Clean up temp file
                    temp_image_path.unlink()
                    
                except Exception as e:
                    print(f"   ❌ Error processing frame {frame_idx}: {e}")
                    # Add empty result for failed frame
                    batch_results.append({
                        'frame_idx': frame_idx,
                        'num_detections': 0,
                        'detections': [],
                        'has_valid_tracking': False,
                        'error': str(e)
                    })
            
            # Add batch results to main results
            all_results.extend(batch_results)
    
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Total frames processed: {detection_stats['total_frames']}")
    print(f"   Frames with detections: {detection_stats['frames_with_detections']}")
    print(f"   Detection rate: {detection_stats['frames_with_detections']/detection_stats['total_frames']*100:.2f}%")
    print(f"   Total detections: {detection_stats['total_detections']}")
    print(f"   Average detections per frame: {detection_stats['total_detections']/detection_stats['total_frames']:.3f}")
    
    if detection_stats['confidence_scores']:
        conf_scores = np.array(detection_stats['confidence_scores'])
        print(f"   Confidence - Mean: {np.mean(conf_scores):.3f}, Min: {np.min(conf_scores):.3f}, Max: {np.max(conf_scores):.3f}")
    
    # Save detailed results
    save_results(all_results, detection_stats, output_dir, confidence)
    
    return all_results, detection_stats

def save_results(all_results, detection_stats, output_dir, confidence):
    """
    Save prediction results in multiple formats.
    """
    print(f"\n💾 Saving results to {output_dir}...")
    
    # 1. Save summary statistics
    summary = {
        'model_confidence': confidence,
        'total_frames': detection_stats['total_frames'],
        'frames_with_detections': detection_stats['frames_with_detections'],
        'detection_rate': detection_stats['frames_with_detections'] / detection_stats['total_frames'],
        'total_detections': detection_stats['total_detections'],
        'average_detections_per_frame': detection_stats['total_detections'] / detection_stats['total_frames'],
        'confidence_stats': {
            'mean': float(np.mean(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0,
            'min': float(np.min(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0,
            'max': float(np.max(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0,
            'std': float(np.std(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0
        }
    }
    
    with open(output_dir / 'prediction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 2. Save frame-by-frame results as CSV
    frame_data = []
    for result in all_results:
        frame_data.append({
            'frame_idx': result['frame_idx'],
            'num_detections': result['num_detections'],
            'has_valid_tracking': result.get('has_valid_tracking', False),
            'tracking_heading': result.get('tracking_heading', np.nan),
            'max_confidence': max([d['confidence'] for d in result['detections']]) if result['detections'] else 0,
            'mean_confidence': np.mean([d['confidence'] for d in result['detections']]) if result['detections'] else 0
        })
    
    df_frames = pd.DataFrame(frame_data)
    df_frames.to_csv(output_dir / 'frame_results.csv', index=False)
    
    # 3. Save all detections as CSV
    detection_data = []
    for result in all_results:
        for detection in result['detections']:
            detection_data.append(detection)
    
    if detection_data:
        df_detections = pd.DataFrame(detection_data)
        df_detections.to_csv(output_dir / 'all_detections.csv', index=False)
    
    # 4. Save detection timeline (for plotting)
    detection_timeline = detection_stats['detection_by_frame'].astype(int)
    np.save(output_dir / 'detection_timeline.npy', detection_timeline)
    
    print(f"✅ Results saved:")
    print(f"   📊 prediction_summary.json - Overall statistics")
    print(f"   📋 frame_results.csv - Frame-by-frame results")
    print(f"   📋 all_detections.csv - All individual detections")
    print(f"   📈 detection_timeline.npy - Binary detection timeline")

def create_analysis_plots(output_dir):
    """
    Create analysis plots from the saved results.
    """
    print(f"\n📈 Creating analysis plots...")
    
    output_dir = Path(output_dir)
    
    # Load data
    with open(output_dir / 'prediction_summary.json', 'r') as f:
        summary = json.load(f)
    
    df_frames = pd.read_csv(output_dir / 'frame_results.csv')
    detection_timeline = np.load(output_dir / 'detection_timeline.npy')
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Detection timeline
    ax1 = axes[0, 0]
    ax1.plot(detection_timeline, linewidth=0.5, alpha=0.7)
    ax1.fill_between(range(len(detection_timeline)), detection_timeline, alpha=0.3)
    ax1.set_title(f'Detection Timeline\n({summary["frames_with_detections"]} detections in {summary["total_frames"]} frames)')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Detection (0/1)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detection rate over time (rolling window)
    ax2 = axes[0, 1]
    window_size = max(100, len(detection_timeline) // 50)  # Adaptive window size
    rolling_rate = pd.Series(detection_timeline).rolling(window_size, center=True).mean()
    ax2.plot(rolling_rate, linewidth=2, color='orange')
    ax2.set_title(f'Detection Rate (Rolling Window = {window_size})')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Detection Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Confidence distribution
    ax3 = axes[1, 0]
    if len(df_frames[df_frames['max_confidence'] > 0]) > 0:
        conf_values = df_frames[df_frames['max_confidence'] > 0]['max_confidence']
        ax3.hist(conf_values, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title(f'Confidence Score Distribution\n(Mean: {summary["confidence_stats"]["mean"]:.3f})')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Number of Detections')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No detections found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Confidence Score Distribution')
    
    # Plot 4: Tracking vs Detection comparison
    ax4 = axes[1, 1]
    if 'has_valid_tracking' in df_frames.columns:
        # Create 2x2 contingency table
        tracking_detection = pd.crosstab(df_frames['has_valid_tracking'], 
                                       df_frames['num_detections'] > 0,
                                       margins=True)
        
        # Calculate agreement rate
        both_positive = len(df_frames[(df_frames['has_valid_tracking'] == True) & (df_frames['num_detections'] > 0)])
        both_negative = len(df_frames[(df_frames['has_valid_tracking'] == False) & (df_frames['num_detections'] == 0)])
        total = len(df_frames)
        agreement_rate = (both_positive + both_negative) / total
        
        ax4.text(0.1, 0.9, f'Tracking vs YOLO Detection Agreement', transform=ax4.transAxes, fontsize=12, weight='bold')
        ax4.text(0.1, 0.8, f'Agreement rate: {agreement_rate:.2%}', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Both detected: {both_positive}', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Both missed: {both_negative}', transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'YOLO only: {len(df_frames[(df_frames["has_valid_tracking"] == False) & (df_frames["num_detections"] > 0)])}', transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f'Tracking only: {len(df_frames[(df_frames["has_valid_tracking"] == True) & (df_frames["num_detections"] == 0)])}', transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, 'No tracking data available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.suptitle(f'Full Video YOLO Prediction Analysis (conf={summary["model_confidence"]})', fontsize=16, y=0.98)
    
    # Save plot
    plot_path = output_dir / 'prediction_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Analysis plots saved to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO predictions on ALL frames in Zarr data",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Run on all frames with confidence 0.25
  python zarr_full_prediction.py video.zarr runs/detect/train4/weights/best.pt --confidence 0.25
  
  # Process in smaller batches (if memory issues)
  python zarr_full_prediction.py video.zarr model.pt --batch-size 16
  
  # Use ROI images instead of downsampled
  python zarr_full_prediction.py video.zarr model.pt --data-source roi_images
        """
    )
    
    parser.add_argument("zarr_path", type=str, help="Path to Zarr data file")
    parser.add_argument("model_path", type=str, help="Path to trained YOLO model")
    
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--data-source", type=str, default="images_ds",
                       choices=["images_ds", "images_full", "roi_images"],
                       help="Which images to use (default: images_ds)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing (default: 32)")
    parser.add_argument("--output-dir", type=str, default="full_video_predictions",
                       help="Output directory for results")
    parser.add_argument("--create-plots", action='store_true',
                       help="Create analysis plots")
    
    args = parser.parse_args()
    
    print("🎬 FULL VIDEO YOLO PREDICTION")
    print("🎯 Running predictions on ALL frames in the video")
    print("⚠️  This may take a while for large videos!")
    print()
    
    # Run predictions
    results, stats = run_full_video_predictions(
        args.model_path, args.zarr_path, args.data_source,
        args.confidence, args.batch_size, args.output_dir
    )
    
    if results is None:
        print("❌ Prediction failed")
        return
    
    # Create analysis plots
    if args.create_plots:
        create_analysis_plots(args.output_dir)
    
    print(f"\n🎉 FULL VIDEO PREDICTION COMPLETE!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"📊 Detection rate: {stats['frames_with_detections']}/{stats['total_frames']} ({stats['frames_with_detections']/stats['total_frames']*100:.1f}%)")

if __name__ == "__main__":
    main()