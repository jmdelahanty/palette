#!/usr/bin/env python3
"""
Test Trained Model on Extracted Frames
Run your trained YOLO model on extracted frames and create annotated results.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

def test_model_on_frames(model_path, frames_dir, output_dir, confidence=0.25, save_annotated=True):
    """
    Test trained YOLO model on extracted frames.
    
    Args:
        model_path: Path to trained model (best.pt or last.pt)
        frames_dir: Directory containing extracted frames
        output_dir: Directory to save results
        confidence: Confidence threshold for detections
        save_annotated: Whether to save annotated images
    """
    
    model_path = Path(model_path)
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"🤖 Loading trained model: {model_path}")
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(frames_dir.glob(f'*{ext}'))
        image_files.extend(frames_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No image files found in {frames_dir}")
        return False
    
    print(f"🖼️  Found {len(image_files)} images to process")
    
    # Process each image
    results_summary = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Run inference
            results = model.predict(str(image_path), conf=confidence, verbose=False)
            
            if len(results) == 0:
                print(f"⚠️  No results for {image_path.name}")
                continue
            
            result = results[0]
            
            # Count detections
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
                'image': image_path.name,
                'detections': num_detections,
                'boxes': detections
            })
            
            # Save annotated image if requested
            if save_annotated and num_detections > 0:
                # Read original image
                image = cv2.imread(str(image_path))
                annotated_image = image.copy()
                
                # Draw detections
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    conf = detection['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add confidence label
                    label = f"Fish {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(annotated_image, 
                                (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)), 
                                (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(annotated_image, label, 
                              (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Save annotated image
                annotated_path = output_dir / f"annotated_{image_path.name}"
                cv2.imwrite(str(annotated_path), annotated_image)
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
    
    # Print summary
    print(f"\n📊 Detection Summary:")
    print(f"📁 Processed: {len(image_files)} images")
    
    total_detections = sum(r['detections'] for r in results_summary)
    images_with_detections = sum(1 for r in results_summary if r['detections'] > 0)
    
    print(f"🎯 Total detections: {total_detections}")
    print(f"🖼️  Images with detections: {images_with_detections}/{len(image_files)} ({images_with_detections/len(image_files)*100:.1f}%)")
    
    if total_detections > 0:
        confidences = []
        for r in results_summary:
            for box in r['boxes']:
                confidences.append(box['confidence'])
        
        avg_conf = np.mean(confidences)
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        
        print(f"📈 Confidence scores - Avg: {avg_conf:.3f}, Min: {min_conf:.3f}, Max: {max_conf:.3f}")
    
    # Save detailed results
    results_file = output_dir / 'detection_results.txt'
    with open(results_file, 'w') as f:
        f.write("Detection Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results_summary:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Detections: {result['detections']}\n")
            
            if result['boxes']:
                for i, box in enumerate(result['boxes']):
                    x1, y1, x2, y2 = box['bbox']
                    f.write(f"  Detection {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={box['confidence']:.3f}\n")
            
            f.write("\n")
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Show some example results
    print(f"\n🖼️  Example results:")
    for result in results_summary[:5]:
        if result['detections'] > 0:
            print(f"   {result['image']}: {result['detections']} fish detected")
    
    return True

def create_detection_grid(frames_dir, output_dir, max_images=16):
    """
    Create a grid showing original and annotated images side by side.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # Find original and annotated images
    original_images = sorted(frames_dir.glob('frame_*.jpg'))
    annotated_images = sorted(output_dir.glob('annotated_*.jpg'))
    
    if not annotated_images:
        print("⚠️  No annotated images found for grid creation")
        return
    
    # Select subset for grid
    selected_annotated = annotated_images[:max_images]
    
    # Calculate grid dimensions
    n_images = len(selected_annotated)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, annotated_path in enumerate(selected_annotated):
        if i >= len(axes):
            break
        
        # Load annotated image
        img = cv2.imread(str(annotated_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img_rgb)
        axes[i].set_title(f'{annotated_path.name}', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Fish Detection Results', fontsize=16, y=0.98)
    
    grid_path = output_dir / 'detection_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Detection grid saved to: {grid_path}")

def main():
    parser = argparse.ArgumentParser(description="Test trained YOLO model on extracted frames")
    
    parser.add_argument("model_path", type=str, help="Path to trained model (e.g., runs/detect/train16/weights/best.pt)")
    parser.add_argument("frames_dir", type=str, help="Directory containing extracted frames")
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    
    parser.add_argument("--confidence", type=float, default=0.25, 
                       help="Confidence threshold for detections")
    parser.add_argument("--no-annotated", action='store_true',
                       help="Don't save annotated images")
    parser.add_argument("--create-grid", action='store_true',
                       help="Create detection grid visualization")
    
    args = parser.parse_args()
    
    print(f"🚀 Testing Trained Fish Detection Model")
    print(f"🤖 Model: {args.model_path}")
    print(f"🖼️  Frames: {args.frames_dir}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🎯 Confidence: {args.confidence}")
    
    # Test model
    success = test_model_on_frames(
        args.model_path,
        args.frames_dir, 
        args.output_dir,
        args.confidence,
        not args.no_annotated
    )
    
    if success:
        print(f"\n✅ Model testing completed!")
        
        if args.create_grid:
            print(f"📊 Creating detection grid...")
            create_detection_grid(args.frames_dir, args.output_dir)
        
        print(f"\n🎉 Results available in: {args.output_dir}")
    else:
        print(f"\n❌ Model testing failed")

if __name__ == "__main__":
    main()