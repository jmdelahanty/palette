#!/usr/bin/env python3
"""
Detection Coverage Analysis Script
Analyzes the train/val split to ensure we're not accidentally including frames without detections.
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def analyze_detection_coverage(zarr_path, split_ratio=0.8, random_seed=42):
    """
    Analyze detection coverage in the dataset and train/val splits.
    """
    print(f"🔍 Analyzing detection coverage in: {zarr_path}")
    print("=" * 60)
    
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"❌ Error opening Zarr file: {e}")
        return
    
    # Check if tracking results exist
    if 'tracking/tracking_results' not in root:
        print("❌ No tracking results found in the Zarr file")
        return
    
    tracking_results = root['tracking/tracking_results']
    column_names = tracking_results.attrs.get('column_names', [])
    
    print(f"📊 Total frames in dataset: {tracking_results.shape[0]}")
    print(f"📋 Tracking columns: {len(column_names)}")
    print()
    
    # Load all tracking data
    data = tracking_results[:]
    total_frames = data.shape[0]
    
    # Different ways to detect valid frames
    print("🎯 Detection Analysis:")
    print("-" * 40)
    
    # Method 1: Check heading (first column) - our current method
    valid_heading = ~np.isnan(data[:, 0])
    heading_count = np.sum(valid_heading)
    print(f"✅ Frames with valid heading: {heading_count}/{total_frames} ({heading_count/total_frames*100:.1f}%)")
    
    # Method 2: Check bbox center coordinates
    col_map = {name: i for i, name in enumerate(column_names)}
    
    if 'bbox_x_norm_ds' in col_map and 'bbox_y_norm_ds' in col_map:
        bbox_x_col = col_map['bbox_x_norm_ds']
        bbox_y_col = col_map['bbox_y_norm_ds']
    elif 'bbox_x_norm' in col_map and 'bbox_y_norm' in col_map:
        bbox_x_col = col_map['bbox_x_norm']
        bbox_y_col = col_map['bbox_y_norm']
    else:
        print("❌ No bbox coordinate columns found")
        return
    
    valid_bbox = ~np.isnan(data[:, bbox_x_col]) & ~np.isnan(data[:, bbox_y_col])
    bbox_count = np.sum(valid_bbox)
    print(f"✅ Frames with valid bbox coords: {bbox_count}/{total_frames} ({bbox_count/total_frames*100:.1f}%)")
    
    # Method 3: Check keypoint data (if available)
    keypoint_cols = []
    for kp_name in ['bladder_x_roi_norm', 'bladder_y_roi_norm', 'eye_l_x_roi_norm', 'eye_l_y_roi_norm', 'eye_r_x_roi_norm', 'eye_r_y_roi_norm']:
        if kp_name in col_map:
            keypoint_cols.append(col_map[kp_name])
    
    if keypoint_cols:
        valid_keypoints = np.all(~np.isnan(data[:, keypoint_cols]), axis=1)
        keypoint_count = np.sum(valid_keypoints)
        print(f"✅ Frames with all keypoints valid: {keypoint_count}/{total_frames} ({keypoint_count/total_frames*100:.1f}%)")
    else:
        valid_keypoints = valid_heading  # Fallback
        keypoint_count = heading_count
    
    # Method 4: Check confidence scores (if available)
    if 'confidence_score' in col_map:
        conf_col = col_map['confidence_score']
        valid_confidence = ~np.isnan(data[:, conf_col]) & (data[:, conf_col] > 0)
        conf_count = np.sum(valid_confidence)
        print(f"✅ Frames with valid confidence: {conf_count}/{total_frames} ({conf_count/total_frames*100:.1f}%)")
    else:
        valid_confidence = valid_heading
        conf_count = heading_count
    
    # Find the intersection of all validation methods
    comprehensive_valid = valid_heading & valid_bbox & valid_keypoints & valid_confidence
    comprehensive_count = np.sum(comprehensive_valid)
    print(f"🎯 Frames passing ALL checks: {comprehensive_count}/{total_frames} ({comprehensive_count/total_frames*100:.1f}%)")
    print()
    
    # Analyze what our current dataset selection is doing
    print("🔬 Current Dataset Selection Analysis:")
    print("-" * 45)
    
    # This mimics the logic in ZarrYOLODataset
    current_valid_mask = ~np.isnan(data[:, 0])  # Valid heading
    
    # For enhanced format, also check bbox data
    if len(column_names) >= 20 and 'bbox_x_norm_ds' in column_names:
        bbox_valid = ~np.isnan(data[:, bbox_x_col]) & ~np.isnan(data[:, bbox_y_col])
        current_valid_mask = current_valid_mask & bbox_valid
        print("📊 Using enhanced format validation (heading + bbox)")
    else:
        print("📊 Using original format validation (heading only)")
    
    current_valid_indices = np.where(current_valid_mask)[0]
    current_count = len(current_valid_indices)
    print(f"✅ Frames selected by current logic: {current_count}/{total_frames} ({current_count/total_frames*100:.1f}%)")
    
    # Analyze the train/val split
    print()
    print("📊 Train/Val Split Analysis:")
    print("-" * 30)
    
    if current_count == 0:
        print("❌ No valid frames found for splitting!")
        return
    
    # Perform the same split as the dataset
    train_indices, val_indices = train_test_split(
        current_valid_indices,
        train_size=split_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"🚂 Training samples: {len(train_indices)}")
    print(f"✅ Validation samples: {len(val_indices)}")
    print(f"📊 Split ratio: {len(train_indices)/(len(train_indices)+len(val_indices)):.3f}")
    print()
    
    # Check if any of our selected samples are actually invalid for YOLO
    print("🎯 YOLO Training Validity Check:")
    print("-" * 35)
    
    def check_sample_validity(indices, split_name):
        print(f"\n{split_name} Set Analysis:")
        invalid_samples = []
        
        for i, idx in enumerate(indices[:10]):  # Check first 10 samples
            sample_data = data[idx]
            
            # Check for NaN values in critical fields
            heading = sample_data[0]
            bbox_x = sample_data[bbox_x_col]
            bbox_y = sample_data[bbox_y_col]
            
            issues = []
            if np.isnan(heading):
                issues.append("heading")
            if np.isnan(bbox_x):
                issues.append("bbox_x")
            if np.isnan(bbox_y):
                issues.append("bbox_y")
            
            # Check keypoints if available
            if keypoint_cols:
                keypoint_values = sample_data[keypoint_cols]
                if np.any(np.isnan(keypoint_values)):
                    issues.append("keypoints")
            
            if issues:
                invalid_samples.append((idx, i, issues))
                print(f"  ❌ Sample {i} (frame {idx}): Missing {', '.join(issues)}")
            elif i < 5:  # Show first 5 valid ones
                print(f"  ✅ Sample {i} (frame {idx}): Valid - heading={heading:.2f}, bbox=({bbox_x:.3f}, {bbox_y:.3f})")
        
        if not invalid_samples:
            print(f"  🎉 All checked {split_name.lower()} samples are valid!")
        else:
            print(f"  ⚠️  Found {len(invalid_samples)} invalid samples in {split_name.lower()} set!")
        
        return invalid_samples
    
    train_invalid = check_sample_validity(train_indices, "Training")
    val_invalid = check_sample_validity(val_indices, "Validation")
    
    # Create visualization
    print()
    print("📈 Creating detection coverage visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Detection coverage over time
    ax1 = axes[0, 0]
    frame_indices = np.arange(total_frames)
    ax1.scatter(frame_indices[current_valid_mask], np.ones(current_count), alpha=0.6, s=1, label='Valid detections')
    ax1.scatter(frame_indices[~current_valid_mask], np.zeros(total_frames - current_count), alpha=0.3, s=1, label='No detection')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Detection Status')
    ax1.set_title('Detection Coverage Over Time')
    ax1.legend()
    
    # Plot 2: Train/val distribution
    ax2 = axes[0, 1]
    ax2.scatter(train_indices, np.ones(len(train_indices)), alpha=0.6, s=2, label=f'Training ({len(train_indices)})', color='blue')
    ax2.scatter(val_indices, np.ones(len(val_indices)) * 1.1, alpha=0.6, s=2, label=f'Validation ({len(val_indices)})', color='orange')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Dataset Split')
    ax2.set_title('Train/Val Split Distribution')
    ax2.legend()
    
    # Plot 3: Detection rate histogram
    ax3 = axes[1, 0]
    detection_rates = []
    window_size = total_frames // 20  # 20 windows
    for i in range(0, total_frames - window_size, window_size):
        window_data = current_valid_mask[i:i+window_size]
        detection_rate = np.sum(window_data) / len(window_data)
        detection_rates.append(detection_rate)
    
    ax3.hist(detection_rates, bins=10, alpha=0.7)
    ax3.set_xlabel('Detection Rate per Window')
    ax3.set_ylabel('Number of Windows')
    ax3.set_title('Distribution of Detection Rates')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    categories = ['Total\nFrames', 'Valid\nHeading', 'Valid\nBbox', 'Valid\nKeypoints', 'Comprehensive\nValid']
    counts = [total_frames, heading_count, bbox_count, keypoint_count, comprehensive_count]
    colors = ['gray', 'lightblue', 'lightgreen', 'orange', 'red']
    
    bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
    ax4.set_ylabel('Number of Frames')
    ax4.set_title('Detection Validation Summary')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + total_frames*0.01,
                f'{count}\n({count/total_frames*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('detection_coverage_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'detection_coverage_analysis.png'")
    plt.show()
    
    # Final recommendations
    print()
    print("💡 Recommendations:")
    print("-" * 20)
    
    if len(train_invalid) > 0 or len(val_invalid) > 0:
        print("⚠️  ISSUE FOUND: Some selected samples have missing data!")
        print("   - This could cause the 0-d tensor error during training")
        print("   - Consider using more strict validation criteria")
        print("   - Check the dataset filtering logic in ZarrYOLODataset")
    else:
        print("✅ All checked samples appear valid for YOLO training")
    
    if comprehensive_count < current_count:
        print(f"⚠️  Gap between current selection ({current_count}) and comprehensive valid ({comprehensive_count})")
        print("   - Consider using stricter validation criteria")
    
    if current_count / total_frames < 0.5:
        print(f"⚠️  Low detection rate: only {current_count/total_frames*100:.1f}% of frames have valid detections")
        print("   - This is normal for tracking data, but verify your tracking pipeline")
    
    return {
        'total_frames': total_frames,
        'valid_detections': current_count,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'train_invalid': len(train_invalid),
        'val_invalid': len(val_invalid)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze detection coverage in Zarr dataset")
    parser.add_argument("zarr_path", type=str, help="Path to the video.zarr file")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/validation split ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate paths
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"❌ Error: Zarr file not found: {zarr_path}")
        return
    
    analyze_detection_coverage(args.zarr_path, args.split_ratio, args.random_seed)

if __name__ == "__main__":
    main()