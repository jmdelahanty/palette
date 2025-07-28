#!/usr/bin/env python3
"""
YOLO Prediction with Top-K=1 Filtering
Modify YOLO predictions to return only the highest confidence detection per frame.
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
import json

def run_topk1_predictions(model_path, zarr_path, data_source='images_ds', 
                         confidence=0.25, batch_size=32, output_dir='topk1_predictions'):
    """
    Run YOLO predictions with top-k=1 filtering (only highest confidence detection per frame).
    """
    print(f"🎯 TOP-K=1 YOLO PREDICTION")
    print(f"📁 Zarr: {zarr_path}")
    print(f"🤖 Model: {model_path}")
    print(f"🎯 Confidence: {confidence}")
    print(f"🏆 Top-K: 1 (only best detection per frame)")
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
        'total_detections': 0,  # This will equal frames_with_detections since top-k=1
        'confidence_scores': [],
        'detection_by_frame': np.zeros(total_frames, dtype=bool)
    }
    
    # Process in batches
    print(f"🔄 Processing {total_frames} frames with TOP-K=1 filtering...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for batch_start in tqdm(range(0, total_frames, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_indices = range(batch_start, batch_end)
            
            batch_results = []
            
            for frame_idx in batch_indices:
                try:
                    # Load and prepare image
                    zarr_image = images_array[frame_idx]
                    
                    if zarr_image.ndim == 2:  # Grayscale
                        yolo_image = np.stack([zarr_image, zarr_image, zarr_image], axis=-1)
                    else:
                        yolo_image = zarr_image
                    
                    if yolo_image.dtype != np.uint8:
                        yolo_image = yolo_image.astype(np.uint8)
                    
                    if yolo_image.shape[:2] != (640, 640):
                        yolo_image = cv2.resize(yolo_image, (640, 640))
                    
                    # Save temporary image
                    temp_image_path = temp_path / f"temp_{frame_idx}.jpg"
                    cv2.imwrite(str(temp_image_path), cv2.cvtColor(yolo_image, cv2.COLOR_RGB2BGR))
                    
                    # Run prediction
                    pred_results = model.predict(str(temp_image_path), conf=confidence, verbose=False)
                    
                    # Apply TOP-K=1 filtering
                    best_detection = None
                    if len(pred_results) > 0 and pred_results[0].boxes is not None:
                        boxes = pred_results[0].boxes
                        if len(boxes) > 0:
                            # Find the detection with highest confidence
                            confidences = [float(box.conf[0].cpu().numpy()) for box in boxes]
                            best_idx = np.argmax(confidences)
                            best_box = boxes[best_idx]
                            
                            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                            conf = float(best_box.conf[0].cpu().numpy())
                            cls_id = int(best_box.cls[0].cpu().numpy())
                            
                            best_detection = {
                                'frame_idx': frame_idx,
                                'x1': float(x1), 'y1': float(y1), 
                                'x2': float(x2), 'y2': float(y2),
                                'confidence': conf,
                                'class_id': cls_id,
                                'center_x': float((x1 + x2) / 2),
                                'center_y': float((y1 + y2) / 2),
                                'width': float(x2 - x1),
                                'height': float(y2 - y1),
                                'is_best': True  # Mark as the best detection
                            }
                    
                    # Store frame result (only best detection or none)
                    frame_detections = [best_detection] if best_detection else []
                    
                    frame_result = {
                        'frame_idx': frame_idx,
                        'num_detections': len(frame_detections),
                        'detections': frame_detections,
                        'best_confidence': best_detection['confidence'] if best_detection else 0.0
                    }
                    
                    batch_results.append(frame_result)
                    
                    # Update statistics
                    if best_detection:
                        detection_stats['frames_with_detections'] += 1
                        detection_stats['total_detections'] += 1  # Always 1 per frame with detection
                        detection_stats['detection_by_frame'][frame_idx] = True
                        detection_stats['confidence_scores'].append(best_detection['confidence'])
                    
                    # Clean up temp file
                    temp_image_path.unlink()
                    
                except Exception as e:
                    print(f"   ❌ Error processing frame {frame_idx}: {e}")
                    batch_results.append({
                        'frame_idx': frame_idx,
                        'num_detections': 0,
                        'detections': [],
                        'best_confidence': 0.0,
                        'error': str(e)
                    })
            
            all_results.extend(batch_results)
    
    print(f"\n📊 TOP-K=1 FINAL STATISTICS:")
    print(f"   Total frames processed: {detection_stats['total_frames']}")
    print(f"   Frames with detections: {detection_stats['frames_with_detections']}")
    print(f"   Detection rate: {detection_stats['frames_with_detections']/detection_stats['total_frames']*100:.2f}%")
    print(f"   Total detections: {detection_stats['total_detections']} (= frames with detections)")
    print(f"   Average detections per frame: {detection_stats['total_detections']/detection_stats['total_frames']:.3f}")
    
    if detection_stats['confidence_scores']:
        conf_scores = np.array(detection_stats['confidence_scores'])
        print(f"   Confidence - Mean: {np.mean(conf_scores):.3f}, Min: {np.min(conf_scores):.3f}, Max: {np.max(conf_scores):.3f}")
    
    # Save results
    save_topk1_results(all_results, detection_stats, output_dir, confidence)
    
    return all_results, detection_stats

def save_topk1_results(all_results, detection_stats, output_dir, confidence):
    """Save top-k=1 results in multiple formats."""
    print(f"\n💾 Saving TOP-K=1 results to {output_dir}...")
    
    # Summary statistics
    summary = {
        'model_confidence': confidence,
        'top_k': 1,
        'total_frames': detection_stats['total_frames'],
        'frames_with_detections': detection_stats['frames_with_detections'],
        'detection_rate': detection_stats['frames_with_detections'] / detection_stats['total_frames'],
        'total_detections': detection_stats['total_detections'],
        'detections_per_frame': detection_stats['total_detections'] / detection_stats['total_frames'],
        'confidence_stats': {
            'mean': float(np.mean(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0,
            'min': float(np.min(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0,
            'max': float(np.max(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0,
            'std': float(np.std(detection_stats['confidence_scores'])) if detection_stats['confidence_scores'] else 0
        }
    }
    
    with open(output_dir / 'topk1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Frame-by-frame results
    frame_data = []
    for result in all_results:
        frame_data.append({
            'frame_idx': result['frame_idx'],
            'has_detection': result['num_detections'] > 0,
            'confidence': result['best_confidence'],
            'center_x': result['detections'][0]['center_x'] if result['detections'] else np.nan,
            'center_y': result['detections'][0]['center_y'] if result['detections'] else np.nan,
            'width': result['detections'][0]['width'] if result['detections'] else np.nan,
            'height': result['detections'][0]['height'] if result['detections'] else np.nan
        })
    
    df_frames = pd.DataFrame(frame_data)
    df_frames.to_csv(output_dir / 'topk1_frame_results.csv', index=False)
    
    # All detections (only best ones)
    detection_data = []
    for result in all_results:
        for detection in result['detections']:
            detection_data.append(detection)
    
    if detection_data:
        df_detections = pd.DataFrame(detection_data)
        df_detections.to_csv(output_dir / 'topk1_detections.csv', index=False)
    
    # Detection timeline
    detection_timeline = detection_stats['detection_by_frame'].astype(int)
    np.save(output_dir / 'topk1_detection_timeline.npy', detection_timeline)
    
    print(f"✅ TOP-K=1 results saved:")
    print(f"   📊 topk1_summary.json - Statistics")
    print(f"   📋 topk1_frame_results.csv - Frame-by-frame results")
    print(f"   📋 topk1_detections.csv - Best detections only")
    print(f"   📈 topk1_detection_timeline.npy - Detection timeline")

def create_topk1_analysis(output_dir):
    """Create analysis plots for top-k=1 results."""
    print(f"\n📈 Creating TOP-K=1 analysis plots...")
    
    output_dir = Path(output_dir)
    
    # Load data
    with open(output_dir / 'topk1_summary.json', 'r') as f:
        summary = json.load(f)
    
    df_frames = pd.read_csv(output_dir / 'topk1_frame_results.csv')
    detection_timeline = np.load(output_dir / 'topk1_detection_timeline.npy')
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Detection timeline
    ax1 = axes[0, 0]
    ax1.plot(detection_timeline, linewidth=0.8, alpha=0.8, color='blue')
    ax1.fill_between(range(len(detection_timeline)), detection_timeline, alpha=0.3, color='blue')
    ax1.set_title(f'TOP-K=1 Detection Timeline\n({summary["frames_with_detections"]} detections in {summary["total_frames"]} frames)')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Detection (0/1)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence distribution (only best detections)
    ax2 = axes[0, 1]
    if len(df_frames[df_frames['confidence'] > 0]) > 0:
        conf_values = df_frames[df_frames['confidence'] > 0]['confidence']
        ax2.hist(conf_values, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax2.set_title(f'Best Detection Confidence Distribution\n(Mean: {summary["confidence_stats"]["mean"]:.3f})')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Number of Best Detections')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No detections found', ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Fish position heatmap
    ax3 = axes[1, 0]
    detected_frames = df_frames[df_frames['has_detection'] == True]
    if len(detected_frames) > 0:
        x_coords = detected_frames['center_x'].dropna()
        y_coords = detected_frames['center_y'].dropna()
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            ax3.hexbin(x_coords, y_coords, gridsize=20, cmap='Blues', alpha=0.7)
            ax3.set_title(f'Fish Position Heatmap\n({len(x_coords)} detections)')
            ax3.set_xlabel('X Position (pixels)')
            ax3.set_ylabel('Y Position (pixels)')
            ax3.set_xlim(0, 640)
            ax3.set_ylim(640, 0)  # Flip Y axis for image coordinates
        else:
            ax3.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Fish size distribution
    ax4 = axes[1, 1]
    if len(detected_frames) > 0:
        widths = detected_frames['width'].dropna()
        heights = detected_frames['height'].dropna()
        
        if len(widths) > 0:
            ax4.scatter(widths, heights, alpha=0.6, s=20)
            ax4.set_title(f'Fish Size Distribution\n({len(widths)} detections)')
            ax4.set_xlabel('Width (pixels)')
            ax4.set_ylabel('Height (pixels)')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics
            ax4.text(0.02, 0.98, f'Mean size: {np.mean(widths):.1f}×{np.mean(heights):.1f}', 
                    transform=ax4.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No size data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.suptitle(f'TOP-K=1 YOLO Analysis (conf={summary["model_confidence"]})', fontsize=16, y=0.98)
    
    # Save plot
    plot_path = output_dir / 'topk1_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ TOP-K=1 analysis saved to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run TOP-K=1 YOLO predictions on Zarr data")
    
    parser.add_argument("zarr_path", type=str, help="Path to Zarr data file")
    parser.add_argument("model_path", type=str, help="Path to trained YOLO model")
    
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--data-source", type=str, default="images_ds",
                       choices=["images_ds", "images_full", "roi_images"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="topk1_predictions")
    parser.add_argument("--create-plots", action='store_true')
    
    args = parser.parse_args()
    
    print("🏆 TOP-K=1 YOLO PREDICTION")
    print("🎯 Only the highest confidence detection per frame")
    print()
    
    # Run predictions
    results, stats = run_topk1_predictions(
        args.model_path, args.zarr_path, args.data_source,
        args.confidence, args.batch_size, args.output_dir
    )
    
    if results is None:
        return
    
    # Create analysis plots
    if args.create_plots:
        create_topk1_analysis(args.output_dir)
    
    print(f"\n🎉 TOP-K=1 PREDICTION COMPLETE!")
    print(f"📁 Results: {args.output_dir}")
    print(f"🏆 Detection rate: {stats['frames_with_detections']}/{stats['total_frames']} ({stats['frames_with_detections']/stats['total_frames']*100:.1f}%)")
    print(f"📊 Exactly 1 detection per positive frame")

if __name__ == "__main__":
    main()