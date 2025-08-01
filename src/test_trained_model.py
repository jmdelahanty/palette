#!/usr/bin/env python3
"""
Test Trained Model on Zarr Dataset
Run your trained YOLO model on frames from a Zarr dataset and create annotated results.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import zarr

def test_model_on_zarr(model_path, zarr_path, output_dir, confidence=0.25, save_annotated=True, max_frames=None):
    """
    Test trained YOLO model on frames from a Zarr dataset.
    
    Args:
        model_path: Path to trained model (best.pt or last.pt)
        zarr_path: Path to the Zarr file
        output_dir: Directory to save results
        confidence: Confidence threshold for detections
        save_annotated: Whether to save annotated images
        max_frames: Optional limit on the number of frames to test
    """
    
    model_path = Path(model_path)
    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False
    
    try:
        zarr_root = zarr.open(str(zarr_path), mode='r')
        images_array = zarr_root['raw_video/images_ds']
    except Exception as e:
        print(f"Error opening Zarr file or finding images_ds: {e}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading trained model: {model_path}")
    try:
        model = YOLO(str(model_path))
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
    num_frames_to_process = min(max_frames, images_array.shape[0]) if max_frames else images_array.shape[0]
    print(f"Found {images_array.shape[0]} frames. Processing the first {num_frames_to_process}.")
    
    # Process each image
    results_summary = []
    
    for frame_idx in tqdm(range(num_frames_to_process), desc="Processing frames"):
        try:
            # Load image from Zarr and prepare for YOLO
            image_gray = images_array[frame_idx]
            image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

            # Run inference
            results = model.predict(image_bgr, conf=confidence, verbose=False)
            result = results[0]
            
            num_detections = len(result.boxes) if result.boxes is not None else 0
            
            # Get detection info
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': cls_id
                    })
            
            results_summary.append({
                'frame_index': frame_idx,
                'detections': num_detections,
                'boxes': detections
            })
            
            # Save annotated image if requested
            if save_annotated and num_detections > 0:
                annotated_image = result.plot() # Use ultralytics' plotting function
                annotated_path = output_dir / f"annotated_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(annotated_path), annotated_image)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
    
    # Print summary
    print(f"\nDetection Summary:")
    print(f"Processed: {num_frames_to_process} frames")

    total_detections = sum(r['detections'] for r in results_summary)
    images_with_detections = sum(1 for r in results_summary if r['detections'] > 0)

    print(f"Total detections: {total_detections}")
    print(f"Images with detections: {images_with_detections}/{num_frames_to_process} ({images_with_detections/num_frames_to_process*100:.1f}%)")
    
    if total_detections > 0:
        confidences = [box['confidence'] for r in results_summary for box in r['boxes']]
        print(f"Confidence scores - Avg: {np.mean(confidences):.3f}, Min: {np.min(confidences):.3f}, Max: {np.max(confidences):.3f}")

    # Save detailed results
    results_file = output_dir / 'detection_results.txt'
    with open(results_file, 'w') as f:
        f.write("Detection Results Summary\n" + "=" * 50 + "\n\n")
        for result in results_summary:
            f.write(f"Frame: {result['frame_index']}\n Detections: {result['detections']}\n")
            if result['boxes']:
                for i, box in enumerate(result['boxes']):
                    x1, y1, x2, y2 = box['bbox']
                    f.write(f"  - Detection {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={box['confidence']:.3f}\n")
            f.write("\n")

    print(f"Detailed results saved to: {results_file}")

    return True

def create_detection_grid(output_dir, max_images=16):
    """
    Create a grid showing annotated images.
    """
    output_dir = Path(output_dir)
    annotated_images = sorted(output_dir.glob('annotated_*.jpg'))
    
    if not annotated_images:
        print("No annotated images found for grid creation")
        return
    
    selected_annotated = annotated_images[:max_images]
    
    n_images = len(selected_annotated)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = np.array(axes).flatten()
    
    for i, annotated_path in enumerate(selected_annotated):
        img = cv2.imread(str(annotated_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f'{annotated_path.name}', fontsize=10)
        axes[i].axis('off')
    
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Fish Detection Results', fontsize=16, y=0.98)
    
    grid_path = output_dir / 'detection_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Detection grid saved to: {grid_path}")

def main():
    parser = argparse.ArgumentParser(description="Test trained YOLO model on a Zarr dataset")
    
    parser.add_argument("model_path", type=str, help="Path to trained model (e.g., runs/detect/train/weights/best.pt)")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr dataset file")
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--no-annotated", action='store_true', help="Don't save annotated images")
    parser.add_argument("--create-grid", action='store_true', help="Create detection grid visualization")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to test from the start of the video")
    
    args = parser.parse_args()

    print(f"Testing Trained Fish Detection Model on Zarr")
    print(f"Model: {args.model_path}")
    print(f"Zarr Dataset: {args.zarr_path}")
    print(f"Output: {args.output_dir}")
    print(f"Confidence: {args.confidence}")

    success = test_model_on_zarr(
        args.model_path,
        args.zarr_path, 
        args.output_dir,
        args.confidence,
        not args.no_annotated,
        args.max_frames
    )
    
    if success:
        print(f"\nModel testing completed!")
        if args.create_grid:
            print(f"Creating detection grid...")
            create_detection_grid(args.output_dir)
        print(f"\nResults available in: {args.output_dir}")
    else:
        print(f"\nModel testing failed")

if __name__ == "__main__":
    main()