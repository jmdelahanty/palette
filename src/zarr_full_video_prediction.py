#!/usr/bin/env python3
"""
Full Video YOLO Prediction from Zarr (Optimized Version)
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
import json
import os

def draw_yolo_bbox(image, detections, color=(0, 255, 0), thickness=2):
    """Draw YOLO format bounding boxes on an image."""
    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        conf = det['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Add label with confidence score
        label = f"Fish {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

def yolo_image_generator(images_array, batch_size):
    """A generator that yields batches of images preprocessed for YOLO."""
    total_frames = images_array.shape[0]
    for i in range(0, total_frames, batch_size):
        batch_images = images_array[i:i + batch_size]
        processed_batch = []
        for zarr_image in batch_images:
            if zarr_image.ndim == 2:
                yolo_image = np.stack([zarr_image] * 3, axis=-1)
            else:
                yolo_image = zarr_image
            
            if yolo_image.dtype != np.uint8:
                yolo_image = yolo_image.astype(np.uint8)
            if yolo_image.shape[:2] != (640, 640):
                yolo_image = cv2.resize(yolo_image, (640, 640))
            
            processed_batch.append(yolo_image)
        yield processed_batch

def run_full_video_predictions(model_path, zarr_path, data_source='images_ds', 
                              confidence=0.25, batch_size=32, output_dir='predictions_output',
                              save_annotated=False):
    """Run YOLO predictions on ALL frames in the Zarr data. (Optimized)"""
    print(f"🎬 FULL VIDEO PREDICTION (Optimized)")
    print(f"📁 Zarr: {zarr_path}")
    print(f"🤖 Model: {model_path}")
    print("=" * 60)
    
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"❌ Error loading Zarr: {e}")
        return None
    
    data_source_mapping = {'images_ds': 'raw_video/images_ds', 'images_full': 'raw_video/images_full', 'roi_images': 'crop_data/roi_images'}
    zarr_data_path = data_source_mapping.get(data_source)
    if not zarr_data_path or zarr_data_path not in root:
        print(f"❌ Data source '{data_source}' not found")
        return None

    images_array = root[zarr_data_path]
    total_frames = images_array.shape[0]
    print(f"📊 Total frames to process: {total_frames}")

    try:
        model = YOLO(model_path)
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated_frames" if save_annotated else None
    if save_annotated:
        annotated_dir.mkdir(exist_ok=True)
        print(f"🖼️  Saving annotated images to: {annotated_dir}")
    
    all_results = []
    detection_stats = {'total_frames': total_frames, 'frames_with_detections': 0, 'total_detections': 0, 'confidence_scores': [], 'detection_by_frame': np.zeros(total_frames, dtype=bool)}
    
    image_gen = yolo_image_generator(images_array, batch_size)
    num_batches = (total_frames + batch_size - 1) // batch_size
    
    print(f"🔄 Processing {total_frames} frames...")
    
    current_frame_idx = 0
    for image_batch in tqdm(image_gen, total=num_batches, desc="Predicting Batches"):
        # Pass the list of images directly to the model
        results_list = model.predict(image_batch, verbose=False, conf=confidence)
        
        for i, result in enumerate(results_list):
            frame_idx = current_frame_idx + i
            frame_detections = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    frame_detections.append({'frame_idx': frame_idx, 'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2), 'confidence': conf, 'class_id': cls_id})
            
            if save_annotated and len(frame_detections) > 0:
                annotated_image = draw_yolo_bbox(result.orig_img.copy(), frame_detections)
                save_path = annotated_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            all_results.append({'frame_idx': frame_idx, 'num_detections': len(frame_detections), 'detections': frame_detections})
            
            if len(frame_detections) > 0:
                detection_stats['frames_with_detections'] += 1
                detection_stats['total_detections'] += len(frame_detections)
                detection_stats['detection_by_frame'][frame_idx] = True
                detection_stats['confidence_scores'].extend([d['confidence'] for d in frame_detections])
        
        current_frame_idx += len(image_batch)

    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Detection rate: {detection_stats['frames_with_detections']/total_frames*100:.2f}%")
    
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
    parser.add_argument("--save-annotated", action='store_true',
                        help="Save images with bounding boxes drawn on them.")
    
    args = parser.parse_args()
    
    print("🎬 FULL VIDEO YOLO PREDICTION")
    print("🎯 Running predictions on ALL frames in the video")
    print("⚠️  This may take a while for large videos!")
    print()
    
    # Run predictions
    results, stats = run_full_video_predictions(
        args.model_path, args.zarr_path, args.data_source,
        args.confidence, args.batch_size, args.output_dir,
        args.save_annotated
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