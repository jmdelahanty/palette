#!/usr/bin/env python3
"""
H5 File Visualization Format Validator

This script validates that H5 files contain all necessary fields and structures
for proper visualization of chaser/target states, including interpolation support.

Usage:
    python validate_h5_viz.py <h5_file_path>
    python validate_h5_viz.py /path/to/out_analysis.h5
"""

import h5py
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Store validation results for a specific check"""
    passed: bool
    message: str
    details: Optional[Dict] = None


class H5VisualizationValidator:
    """Validator for H5 files used in visualization"""
    
    # Required fields for each data structure
    BOUNDING_BOX_FIELDS = [
        'payload_timestamp_ns_epoch',
        'received_timestamp_ns_epoch', 
        'payload_frame_id',
        'payload_camera_id',
        'box_index_in_payload',
        'x_min',
        'y_min',
        'width',     # Note: width, not x_max
        'height',    # Note: height, not y_max
        'class_id',
        'confidence'
    ]
    
    CHASER_STATE_FIELDS = [
        'stimulus_frame_num',
        'timestamp_ns_session',
        'chaser_index',
        'is_chasing',
        'chaser_pos_x',
        'chaser_pos_y',
        'target_pos_x',
        'target_pos_y'
    ]
    
    FRAME_METADATA_FIELDS = [
        'stimulus_frame_num',
        'triggering_camera_frame_id',
        'timestamp_ns'
    ]
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.results = []
        self.is_analysis_file = False
        
    def validate(self) -> bool:
        """Run all validation checks"""
        print(f"\n{'='*60}")
        print(f"H5 VISUALIZATION FORMAT VALIDATOR")
        print(f"{'='*60}")
        print(f"File: {self.filepath}")
        print(f"{'='*60}\n")
        
        # Check file exists
        if not self.filepath.exists():
            print(f"❌ Error: File does not exist: {self.filepath}")
            return False
            
        try:
            with h5py.File(self.filepath, 'r') as f:
                # Detect if this is an analysis file
                self.is_analysis_file = self._detect_analysis_file(f)
                
                # Run all validation checks
                self._validate_structure(f)
                self._validate_tracking_data(f)
                self._validate_video_metadata(f)
                self._validate_analysis_data(f)
                self._validate_calibration_data(f)
                self._validate_data_consistency(f)
                self._check_interpolation_coverage(f)
                
        except Exception as e:
            print(f"❌ Error opening file: {e}")
            return False
        
        # Print summary
        self._print_summary()
        
        # Return overall success
        return all(r.passed for r in self.results if "Warning" not in r.message)
    
    def _detect_analysis_file(self, f: h5py.File) -> bool:
        """Detect if this is an analysis file"""
        filename = self.filepath.name.lower()
        is_analysis = ('analysis' in filename or 
                      'out_analysis' in filename or
                      '/analysis' in f)
        
        if is_analysis:
            print(f"📊 Detected as ANALYSIS file (contains interpolation data)\n")
        else:
            print(f"📊 Detected as STANDARD file (no interpolation expected)\n")
            
        return is_analysis
    
    def _validate_structure(self, f: h5py.File):
        """Validate basic HDF5 structure"""
        print("1. VALIDATING FILE STRUCTURE")
        print("-" * 40)
        
        # Check required groups
        required_groups = ['/tracking_data', '/video_metadata']
        for group in required_groups:
            if group in f:
                self.results.append(ValidationResult(
                    True, f"✅ Found required group: {group}",
                    {"keys": list(f[group].keys()) if group in f else []}
                ))
                print(f"  ✅ Found {group}")
                if group in f:
                    print(f"     Contains: {list(f[group].keys())}")
            else:
                self.results.append(ValidationResult(
                    False, f"❌ Missing required group: {group}"
                ))
                print(f"  ❌ Missing {group}")
        
        # Check optional but important groups
        optional_groups = ['/analysis', '/calibration_snapshot', '/protocol_snapshot']
        for group in optional_groups:
            if group in f:
                print(f"  ✅ Found optional group: {group}")
            else:
                print(f"  ⚠️  Missing optional group: {group}")
        print()
    
    def _validate_tracking_data(self, f: h5py.File):
        """Validate tracking data structures"""
        print("2. VALIDATING TRACKING DATA")
        print("-" * 40)
        
        if '/tracking_data' not in f:
            print("  ⚠️  No tracking data group found")
            return
            
        tracking = f['/tracking_data']
        
        # Validate bounding boxes
        if 'bounding_boxes' in tracking:
            dataset = tracking['bounding_boxes']
            print(f"  📦 Bounding Boxes:")
            print(f"     Shape: {dataset.shape}")
            print(f"     Dtype: {dataset.dtype}")
            
            if dataset.dtype.names:
                missing_fields = set(self.BOUNDING_BOX_FIELDS) - set(dataset.dtype.names)
                extra_fields = set(dataset.dtype.names) - set(self.BOUNDING_BOX_FIELDS)
                
                if missing_fields:
                    self.results.append(ValidationResult(
                        False, f"❌ Missing bounding box fields: {missing_fields}"
                    ))
                    print(f"     ❌ Missing fields: {missing_fields}")
                else:
                    self.results.append(ValidationResult(
                        True, "✅ All bounding box fields present"
                    ))
                    print(f"     ✅ All required fields present")
                    
                if extra_fields:
                    print(f"     ℹ️  Extra fields: {extra_fields}")
                    
                # Check for data
                if dataset.shape[0] > 0:
                    print(f"     ✅ Contains {dataset.shape[0]} bounding boxes")
                    # Sample first box
                    first_box = dataset[0]
                    print(f"     Sample box: frame_id={first_box['payload_frame_id']}, "
                          f"pos=({first_box['x_min']:.1f}, {first_box['y_min']:.1f})")
                else:
                    print(f"     ⚠️  No bounding box data")
        else:
            print(f"  ⚠️  No bounding_boxes dataset")
            
        # Validate chaser states
        if 'chaser_states' in tracking:
            dataset = tracking['chaser_states']
            print(f"\n  🎯 Chaser States:")
            print(f"     Shape: {dataset.shape}")
            print(f"     Dtype: {dataset.dtype}")
            
            if dataset.dtype.names:
                missing_fields = set(self.CHASER_STATE_FIELDS) - set(dataset.dtype.names)
                
                if missing_fields:
                    self.results.append(ValidationResult(
                        False, f"❌ Missing chaser state fields: {missing_fields}"
                    ))
                    print(f"     ❌ Missing fields: {missing_fields}")
                else:
                    self.results.append(ValidationResult(
                        True, "✅ All chaser state fields present"
                    ))
                    print(f"     ✅ All required fields present")
                    
                if dataset.shape[0] > 0:
                    print(f"     ✅ Contains {dataset.shape[0]} chaser states")
                    # Sample first state
                    first_state = dataset[0]
                    print(f"     Sample state: frame={first_state['stimulus_frame_num']}, "
                          f"chaser=({first_state['chaser_pos_x']:.1f}, {first_state['chaser_pos_y']:.1f}), "
                          f"target=({first_state['target_pos_x']:.1f}, {first_state['target_pos_y']:.1f})")
                else:
                    print(f"     ⚠️  No chaser state data")
        else:
            self.results.append(ValidationResult(
                False, "❌ Missing chaser_states dataset (required for visualization)"
            ))
            print(f"  ❌ Missing chaser_states dataset")
        print()
    
    def _validate_video_metadata(self, f: h5py.File):
        """Validate video metadata"""
        print("3. VALIDATING VIDEO METADATA")
        print("-" * 40)
        
        if '/video_metadata' not in f:
            print("  ⚠️  No video metadata group")
            return
            
        video = f['/video_metadata']
        
        if 'frame_metadata' in video:
            dataset = video['frame_metadata']
            print(f"  🎬 Frame Metadata:")
            print(f"     Shape: {dataset.shape}")
            
            if dataset.dtype.names:
                missing_fields = set(self.FRAME_METADATA_FIELDS) - set(dataset.dtype.names)
                
                if missing_fields:
                    self.results.append(ValidationResult(
                        False, f"❌ Missing frame metadata fields: {missing_fields}"
                    ))
                    print(f"     ❌ Missing fields: {missing_fields}")
                else:
                    self.results.append(ValidationResult(
                        True, "✅ All frame metadata fields present"
                    ))
                    print(f"     ✅ All required fields present")
                    
                # Check for continuity
                if dataset.shape[0] > 0:
                    frame_nums = dataset['stimulus_frame_num'][:]
                    gaps = []
                    for i in range(1, len(frame_nums)):
                        if frame_nums[i] != frame_nums[i-1] + 1:
                            gap_size = frame_nums[i] - frame_nums[i-1] - 1
                            gaps.append((frame_nums[i-1], frame_nums[i], gap_size))
                    
                    if gaps:
                        print(f"     ⚠️  Found {len(gaps)} gaps in frame sequence")
                        for start, end, size in gaps[:3]:  # Show first 3 gaps
                            print(f"        Gap: frames {start} -> {end} (missing {size} frames)")
                    else:
                        print(f"     ✅ Continuous frame sequence (no gaps)")
                        
                    # Calculate FPS
                    if len(dataset) > 10:
                        time_diff = (dataset[-1]['timestamp_ns'] - dataset[0]['timestamp_ns']) / 1e9
                        fps = (len(dataset) - 1) / time_diff
                        print(f"     📊 Calculated FPS: {fps:.2f}")
        else:
            print(f"  ❌ Missing frame_metadata dataset")
        print()
    
    def _validate_analysis_data(self, f: h5py.File):
        """Validate analysis-specific data (interpolation)"""
        print("4. VALIDATING ANALYSIS DATA")
        print("-" * 40)
        
        if not self.is_analysis_file:
            print("  ℹ️  Not an analysis file - skipping interpolation checks")
            print()
            return
            
        if '/analysis' not in f:
            self.results.append(ValidationResult(
                False, "❌ Analysis file missing /analysis group"
            ))
            print("  ❌ Missing /analysis group in analysis file!")
            print()
            return
            
        analysis = f['/analysis']
        
        # Check interpolation mask
        if 'interpolation_mask' in analysis:
            mask = analysis['interpolation_mask']
            print(f"  🎭 Interpolation Mask:")
            print(f"     Shape: {mask.shape}")
            print(f"     Dtype: {mask.dtype}")
            
            if mask.shape[0] > 0:
                mask_data = mask[:]
                original_frames = np.sum(mask_data)
                interpolated_frames = len(mask_data) - original_frames
                coverage = (original_frames / len(mask_data)) * 100
                
                self.results.append(ValidationResult(
                    True, "✅ Interpolation mask present",
                    {"original": int(original_frames), 
                     "interpolated": int(interpolated_frames),
                     "coverage": coverage}
                ))
                
                print(f"     ✅ Original frames: {original_frames} ({coverage:.1f}%)")
                print(f"     🔄 Interpolated frames: {interpolated_frames} ({100-coverage:.1f}%)")
                
                # Find interpolation runs
                runs = []
                in_run = False
                run_start = 0
                for i, is_original in enumerate(mask_data):
                    if not is_original and not in_run:
                        in_run = True
                        run_start = i
                    elif is_original and in_run:
                        runs.append((run_start, i-1, i-run_start))
                        in_run = False
                if in_run:
                    runs.append((run_start, len(mask_data)-1, len(mask_data)-run_start))
                    
                if runs:
                    print(f"     📈 Found {len(runs)} interpolated segments")
                    longest_run = max(runs, key=lambda x: x[2])
                    print(f"     Longest interpolation: {longest_run[2]} frames "
                          f"(frames {longest_run[0]}-{longest_run[1]})")
            else:
                print(f"     ⚠️  Empty interpolation mask")
        else:
            self.results.append(ValidationResult(
                False, "❌ Missing interpolation_mask in analysis file"
            ))
            print(f"  ❌ Missing interpolation_mask")
            
        # Check gap info
        if 'gap_info' in analysis:
            print(f"  📋 Gap Info: Present")
            try:
                gap_info = analysis['gap_info'][()].decode('utf-8')
                gap_data = json.loads(gap_info)
                print(f"     Contains: {len(gap_info)} characters of JSON data")
            except:
                print(f"     ⚠️  Could not parse gap info JSON")
        else:
            print(f"  ⚠️  Missing gap_info")
        print()
    
    def _validate_calibration_data(self, f: h5py.File):
        """Validate calibration data"""
        print("5. VALIDATING CALIBRATION DATA")
        print("-" * 40)
        
        if '/calibration_snapshot' in f:
            calib = f['/calibration_snapshot']
            print(f"  🎯 Calibration Snapshot:")
            
            # Check for arena config
            if 'arena_config_json' in calib:
                print(f"     ✅ Arena config present")
            else:
                print(f"     ⚠️  Missing arena config")
                
            # Check for camera calibrations
            camera_dirs = [k for k in calib.keys() if k not in ['arena_config_json']]
            if camera_dirs:
                print(f"     📷 Found {len(camera_dirs)} camera calibrations: {camera_dirs}")
                
                # Check first camera for homography
                first_cam = camera_dirs[0]
                if f'{first_cam}/homography_matrix_yml' in calib:
                    print(f"     ✅ Homography matrix found for {first_cam}")
                else:
                    print(f"     ⚠️  No homography matrix for {first_cam}")
            else:
                print(f"     ⚠️  No camera calibrations found")
        else:
            print("  ⚠️  No calibration snapshot")
        print()
    
    def _validate_data_consistency(self, f: h5py.File):
        """Check consistency between different data sources"""
        print("6. VALIDATING DATA CONSISTENCY")
        print("-" * 40)
        
        # Check if chaser states align with frame metadata
        if '/tracking_data/chaser_states' in f and '/video_metadata/frame_metadata' in f:
            chaser_states = f['/tracking_data/chaser_states']
            frame_metadata = f['/video_metadata/frame_metadata']
            
            if chaser_states.shape[0] > 0 and frame_metadata.shape[0] > 0:
                chaser_frames = set(chaser_states['stimulus_frame_num'][:])
                metadata_frames = set(frame_metadata['stimulus_frame_num'][:])
                
                # Check overlap
                overlap = chaser_frames & metadata_frames
                only_chaser = chaser_frames - metadata_frames
                only_metadata = metadata_frames - chaser_frames
                
                print(f"  🔄 Frame Alignment:")
                print(f"     Frames with both chaser & metadata: {len(overlap)}")
                
                if only_chaser:
                    print(f"     ⚠️  Frames with only chaser data: {len(only_chaser)}")
                if only_metadata:
                    print(f"     ⚠️  Frames with only metadata: {len(only_metadata)}")
                    
                if len(overlap) == len(chaser_frames) == len(metadata_frames):
                    self.results.append(ValidationResult(
                        True, "✅ Perfect alignment between chaser states and frame metadata"
                    ))
                    print(f"     ✅ Perfect alignment!")
                else:
                    coverage = len(overlap) / max(len(chaser_frames), len(metadata_frames)) * 100
                    print(f"     📊 Coverage: {coverage:.1f}%")
        print()
    
    def _check_interpolation_coverage(self, f: h5py.File):
        """Check interpolation coverage if analysis file"""
        if not self.is_analysis_file or '/analysis/interpolation_mask' not in f:
            return
            
        print("7. INTERPOLATION COVERAGE ANALYSIS")
        print("-" * 40)
        
        mask = f['/analysis/interpolation_mask'][:]
        total_frames = len(mask)
        
        # Calculate coverage metrics
        original_count = np.sum(mask)
        interpolated_count = total_frames - original_count
        
        # Find gaps (consecutive interpolated frames)
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_original in enumerate(mask):
            if not is_original and not in_gap:
                in_gap = True
                gap_start = i
            elif is_original and in_gap:
                gap_length = i - gap_start
                gaps.append((gap_start, i-1, gap_length))
                in_gap = False
        
        if in_gap:
            gaps.append((gap_start, total_frames-1, total_frames-gap_start))
        
        # Statistics
        if gaps:
            gap_lengths = [g[2] for g in gaps]
            print(f"  📊 Gap Statistics:")
            print(f"     Total gaps: {len(gaps)}")
            print(f"     Average gap length: {np.mean(gap_lengths):.1f} frames")
            print(f"     Median gap length: {np.median(gap_lengths):.1f} frames")
            print(f"     Max gap length: {max(gap_lengths)} frames")
            print(f"     Min gap length: {min(gap_lengths)} frames")
            
            # Show largest gaps
            largest_gaps = sorted(gaps, key=lambda x: x[2], reverse=True)[:5]
            print(f"\n  🔍 Largest Gaps:")
            for start, end, length in largest_gaps:
                print(f"     Frames {start:6d} - {end:6d}: {length:4d} frames")
        else:
            print(f"  ✅ No gaps found - all frames are original!")
        print()
    
    def _print_summary(self):
        """Print validation summary"""
        print("="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]
        
        print(f"\n✅ Passed: {len(passed)} checks")
        for result in passed:
            print(f"   {result.message}")
            
        if failed:
            print(f"\n❌ Failed: {len(failed)} checks")
            for result in failed:
                print(f"   {result.message}")
        
        print("\n" + "="*60)
        
        if failed:
            print("⚠️  FILE NEEDS ATTENTION for proper visualization")
            print("\nRecommendations:")
            for result in failed:
                if "chaser_states" in result.message:
                    print("  • Ensure chaser tracking data is properly exported")
                elif "interpolation" in result.message:
                    print("  • Run interpolation pipeline on analysis files")
                elif "frame_metadata" in result.message:
                    print("  • Check video metadata export pipeline")
        else:
            print("✅ FILE IS PROPERLY FORMATTED for visualization!")
            
            # Additional info for analysis files
            if self.is_analysis_file:
                for result in self.results:
                    if result.details and 'coverage' in result.details:
                        print(f"\n📊 Data Coverage: {result.details['coverage']:.1f}% original frames")
                        print(f"   {result.details['original']} original / {result.details['interpolated']} interpolated")
        
        print("="*60)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python validate_h5_viz.py <h5_file_path>")
        print("\nExample:")
        print("  python validate_h5_viz.py /path/to/out_analysis.h5")
        sys.exit(1)
    
    filepath = sys.argv[1]
    validator = H5VisualizationValidator(filepath)
    
    success = validator.validate()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()