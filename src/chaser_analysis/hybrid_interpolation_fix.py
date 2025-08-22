#!/usr/bin/env python3
"""
Hybrid H5 Interpolation Fix

Combines the best aspects of both implementations:
1. Proper stimulus frame calculation (no duplicates)
2. Interpolates BOTH metadata AND chaser states
3. Ensures every camera frame has positions for visualization
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Set


class HybridH5Interpolator:
    """
    Fixes H5 files by properly interpolating both metadata and chaser states.
    """
    
    def __init__(self, input_h5: str, output_h5: str = None, verbose: bool = True):
        self.input_path = Path(input_h5)
        self.output_path = Path(output_h5) if output_h5 else self.input_path.with_suffix('.fixed.h5')
        self.verbose = verbose
        self.gaps_info = {
            'camera_frame_gaps': [],
            'interpolated_frames': [],
            'statistics': {}
        }
        
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def analyze_gaps(self) -> bool:
        """Analyze gaps in camera frames."""
        self.log(f"\n📊 Analyzing gaps in: {self.input_path}")
        
        with h5py.File(self.input_path, 'r') as f:
            if '/video_metadata/frame_metadata' not in f:
                raise ValueError("Dataset '/video_metadata/frame_metadata' not found!")
            
            metadata_ds = f['/video_metadata/frame_metadata']
            self.original_metadata = metadata_ds[:]
            
            # Also load chaser states
            if '/tracking_data/chaser_states' in f:
                self.original_chaser = f['/tracking_data/chaser_states'][:]
            else:
                raise ValueError("Dataset '/tracking_data/chaser_states' not found!")
            
            # Get camera frame IDs and analyze gaps
            camera_frame_ids = metadata_ds['triggering_camera_frame_id'][:]
            unique_camera_ids = np.unique(camera_frame_ids)
            sorted_camera_ids = np.sort(unique_camera_ids)
            
            # Find gaps
            expected_range = np.arange(sorted_camera_ids[0], sorted_camera_ids[-1] + 1)
            missing_frames = np.setdiff1d(expected_range, sorted_camera_ids)
            
            if len(missing_frames) == 0:
                self.log("  ✅ No gaps found in camera frame IDs!")
                return False
            
            # Group consecutive missing frames
            gaps = []
            if len(missing_frames) > 0:
                gap_start = missing_frames[0]
                gap_end = missing_frames[0]
                
                for i in range(1, len(missing_frames)):
                    if missing_frames[i] == missing_frames[i-1] + 1:
                        gap_end = missing_frames[i]
                    else:
                        gaps.append((gap_start, gap_end))
                        gap_start = missing_frames[i]
                        gap_end = missing_frames[i]
                gaps.append((gap_start, gap_end))
            
            self.gaps_info['camera_frame_gaps'] = gaps
            self.gaps_info['missing_frames'] = missing_frames.tolist()
            
            self.log(f"  🔍 Found {len(missing_frames)} missing camera frame IDs")
            self.log(f"  📍 Grouped into {len(gaps)} gaps")
            
            # Show first few gaps
            for i, (start, end) in enumerate(gaps[:5]):
                if start == end:
                    self.log(f"    Gap {i+1}: Frame {start}")
                else:
                    self.log(f"    Gap {i+1}: Frames {start}-{end} ({end-start+1} frames)")
            
            if len(gaps) > 5:
                self.log(f"    ... and {len(gaps)-5} more gaps")
            
            return True
    
    def interpolate_both(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate BOTH metadata and chaser states properly.
        Returns: (interpolated_metadata, interpolated_chaser, interpolation_mask)
        """
        self.log("\n🔧 Interpolating metadata AND chaser states...")
        
        # Build mappings from original data
        camera_to_metadata = {}
        stim_to_chaser = {}
        
        # Map camera frames to metadata records
        for record in self.original_metadata:
            cam_frame = int(record['triggering_camera_frame_id'])
            if cam_frame not in camera_to_metadata:
                camera_to_metadata[cam_frame] = []
            camera_to_metadata[cam_frame].append(record)
        
        # Map stimulus frames to chaser states
        for record in self.original_chaser:
            stim_frame = int(record['stimulus_frame_num'])
            stim_to_chaser[stim_frame] = record  # Assuming one chaser state per stimulus frame
        
        # Prepare lists for combined data
        all_metadata = list(self.original_metadata)
        all_chaser = list(self.original_chaser)
        original_pairs = set()  # Track original (stim, cam) pairs
        
        # Record original pairs for mask creation
        for record in self.original_metadata:
            pair = (int(record['stimulus_frame_num']), int(record['triggering_camera_frame_id']))
            original_pairs.add(pair)
        
        # Interpolate missing camera frames
        interpolated_count = 0
        
        for missing_frame in self.gaps_info['missing_frames']:
            missing_frame = int(missing_frame)
            
            # Find surrounding frames
            existing_frames = sorted(camera_to_metadata.keys())
            
            prev_frame = None
            next_frame = None
            
            for frame in existing_frames:
                if frame < missing_frame:
                    prev_frame = frame
                elif frame > missing_frame:
                    next_frame = frame
                    break
            
            if prev_frame is None or next_frame is None:
                self.log(f"  ⚠️  Cannot interpolate frame {missing_frame} - boundary")
                continue
            
            # Get stimulus frames from surrounding camera frames
            prev_records = camera_to_metadata[prev_frame]
            next_records = camera_to_metadata[next_frame]
            
            prev_stim_frames = [int(r['stimulus_frame_num']) for r in prev_records]
            next_stim_frames = [int(r['stimulus_frame_num']) for r in next_records]
            
            prev_max_stim = max(prev_stim_frames)
            next_min_stim = min(next_stim_frames)
            
            # Calculate interpolation (from old script's approach)
            frame_distance = next_frame - prev_frame
            interpolation_factor = (missing_frame - prev_frame) / frame_distance
            
            # Calculate stimulus frames for this camera frame
            stim_diff = next_min_stim - prev_max_stim - 1
            expected_stim_per_camera = stim_diff / frame_distance if frame_distance > 0 else 2
            
            stim_offset = int((missing_frame - prev_frame) * expected_stim_per_camera)
            estimated_stim_base = prev_max_stim + stim_offset + 1
            
            # Determine number of stimulus frames (usually 2 for 120Hz/60Hz)
            num_stim_frames = 2 if expected_stim_per_camera >= 1.5 else 1
            
            # Interpolate timestamp
            prev_time = int(prev_records[-1]['timestamp_ns'])
            next_time = int(next_records[0]['timestamp_ns'])
            interp_time = int(prev_time + (next_time - prev_time) * interpolation_factor)
            
            # Create interpolated records
            for offset in range(num_stim_frames):
                stim_frame_num = estimated_stim_base + offset
                
                # Create metadata record
                meta_record = np.zeros(1, dtype=self.original_metadata.dtype)[0]
                meta_record['stimulus_frame_num'] = stim_frame_num
                meta_record['triggering_camera_frame_id'] = missing_frame
                meta_record['timestamp_ns'] = interp_time + (offset * 8333333)  # ~8.33ms for 120Hz
                
                all_metadata.append(meta_record)
                
                # Create chaser state if it doesn't exist
                if stim_frame_num not in stim_to_chaser:
                    # Find nearest chaser states for interpolation
                    prev_chaser = None
                    next_chaser = None
                    
                    # Search backwards for previous chaser
                    for s in range(stim_frame_num - 1, 0, -1):
                        if s in stim_to_chaser:
                            prev_chaser = stim_to_chaser[s]
                            break
                    
                    # Search forwards for next chaser
                    for s in range(stim_frame_num + 1, max(stim_to_chaser.keys()) + 1):
                        if s in stim_to_chaser:
                            next_chaser = stim_to_chaser[s]
                            break
                    
                    if prev_chaser is not None and next_chaser is not None:
                        # Create interpolated chaser state
                        chaser_record = np.zeros(1, dtype=self.original_chaser.dtype)[0]
                        
                        chaser_record['stimulus_frame_num'] = stim_frame_num
                        chaser_record['timestamp_ns_session'] = interp_time + (offset * 8333333)
                        chaser_record['chaser_index'] = prev_chaser['chaser_index']
                        chaser_record['is_chasing'] = prev_chaser['is_chasing'] or next_chaser['is_chasing']
                        
                        # Interpolate positions
                        weight = interpolation_factor  # Use same weight as metadata
                        
                        for field in ['chaser_pos_x', 'chaser_pos_y', 'target_pos_x', 'target_pos_y']:
                            prev_val = float(prev_chaser[field])
                            next_val = float(next_chaser[field])
                            
                            # Handle -1 (not visible) specially
                            if prev_val < 0 or next_val < 0:
                                chaser_record[field] = -1
                            else:
                                chaser_record[field] = prev_val + (next_val - prev_val) * weight
                        
                        all_chaser.append(chaser_record)
                        stim_to_chaser[stim_frame_num] = chaser_record  # Add to mapping
                        interpolated_count += 1
            
            self.gaps_info['interpolated_frames'].append({
                'camera_frame': int(missing_frame),
                'stimulus_frames': [estimated_stim_base + i for i in range(num_stim_frames)],
                'method': 'linear_interpolation'
            })
        
        self.log(f"  ✅ Interpolated {len(self.gaps_info['interpolated_frames'])} camera frames")
        self.log(f"  ✅ Created {interpolated_count} new chaser states")
        
        # Convert to numpy arrays and sort
        final_metadata = np.array(all_metadata, dtype=self.original_metadata.dtype)
        final_chaser = np.array(all_chaser, dtype=self.original_chaser.dtype)
        
        # Sort by stimulus frame number
        meta_sort = np.argsort(final_metadata['stimulus_frame_num'])
        final_metadata = final_metadata[meta_sort]
        
        chaser_sort = np.argsort(final_chaser['stimulus_frame_num'])
        final_chaser = final_chaser[chaser_sort]
        
        # Create interpolation mask
        mask = np.ones(len(final_metadata), dtype=bool)
        for i, record in enumerate(final_metadata):
            pair = (int(record['stimulus_frame_num']), int(record['triggering_camera_frame_id']))
            if pair not in original_pairs:
                mask[i] = False
        
        return final_metadata, final_chaser, mask
    
    def create_output_file(self, metadata: np.ndarray, chaser: np.ndarray, mask: np.ndarray):
        """Create the output H5 file with all data."""
        self.log(f"\n💾 Creating output file: {self.output_path}")
        
        with h5py.File(self.input_path, 'r') as src:
            with h5py.File(self.output_path, 'w') as dst:
                
                # Copy root attributes
                for attr_name, attr_value in src.attrs.items():
                    dst.attrs[attr_name] = attr_value
                
                dst.attrs['interpolation_fixed'] = datetime.now().isoformat()
                dst.attrs['fix_version'] = 'hybrid_1.0'
                
                # Create video_metadata group
                video_group = dst.create_group('video_metadata')
                
                frame_meta_ds = video_group.create_dataset(
                    'frame_metadata',
                    data=metadata,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Copy original attributes
                if '/video_metadata/frame_metadata' in src:
                    for attr, val in src['/video_metadata/frame_metadata'].attrs.items():
                        frame_meta_ds.attrs[attr] = val
                
                frame_meta_ds.attrs['interpolated'] = True
                frame_meta_ds.attrs['original_records'] = np.sum(mask)
                frame_meta_ds.attrs['total_records'] = len(metadata)
                
                # Create tracking_data group
                tracking_group = dst.create_group('tracking_data')
                
                # Copy bounding boxes as-is
                if '/tracking_data/bounding_boxes' in src:
                    bbox_data = src['/tracking_data/bounding_boxes'][:]
                    bbox_ds = tracking_group.create_dataset(
                        'bounding_boxes',
                        data=bbox_data,
                        compression='gzip',
                        compression_opts=4
                    )
                    for attr, val in src['/tracking_data/bounding_boxes'].attrs.items():
                        bbox_ds.attrs[attr] = val
                
                # Write interpolated chaser states
                chaser_ds = tracking_group.create_dataset(
                    'chaser_states',
                    data=chaser,
                    compression='gzip',
                    compression_opts=4
                )
                
                if '/tracking_data/chaser_states' in src:
                    for attr, val in src['/tracking_data/chaser_states'].attrs.items():
                        chaser_ds.attrs[attr] = val
                
                chaser_ds.attrs['interpolated'] = True
                chaser_ds.attrs['original_records'] = len(self.original_chaser)
                chaser_ds.attrs['total_records'] = len(chaser)
                
                # Copy other essential groups
                for group in ['calibration_snapshot', 'protocol_snapshot', 'events', 'stimulus_coordinates']:
                    if f'/{group}' in src:
                        self.log(f"  📁 Copying {group}...")
                        src.copy(f'/{group}', dst)
                
                # Create analysis group
                analysis_group = dst.create_group('analysis')
                
                # Add interpolation mask
                mask_ds = analysis_group.create_dataset(
                    'interpolation_mask',
                    data=mask,
                    compression='gzip',
                    compression_opts=4
                )
                mask_ds.attrs['description'] = 'True for original data, False for interpolated'
                
                # Add gap information
                gap_ds = analysis_group.create_dataset(
                    'gap_info',
                    data=json.dumps(self.gaps_info, indent=2)
                )
                gap_ds.attrs['description'] = 'JSON string with gap analysis'
                
                # Add statistics
                analysis_group.attrs['creation_time'] = datetime.now().isoformat()
                analysis_group.attrs['interpolation_method'] = 'linear'
                analysis_group.attrs['original_metadata_records'] = len(self.original_metadata)
                analysis_group.attrs['original_chaser_records'] = len(self.original_chaser)
                analysis_group.attrs['final_metadata_records'] = len(metadata)
                analysis_group.attrs['final_chaser_records'] = len(chaser)
                analysis_group.attrs['interpolated_frames'] = np.sum(~mask)
        
        self.log("  ✅ Output file created successfully!")
    
    def verify_output(self):
        """Verify the output file has no gaps."""
        self.log("\n🔍 Verifying output...")
        
        with h5py.File(self.output_path, 'r') as f:
            # Check metadata gaps
            metadata = f['/video_metadata/frame_metadata'][:]
            camera_ids = np.unique(metadata['triggering_camera_frame_id'])
            
            expected = np.arange(camera_ids.min(), camera_ids.max() + 1)
            gaps = np.setdiff1d(expected, camera_ids)
            
            if len(gaps) == 0:
                self.log("  ✅ No gaps in camera frames!")
            else:
                self.log(f"  ⚠️  Still have {len(gaps)} gaps")
            
            # Check chaser coverage
            chaser = f['/tracking_data/chaser_states'][:]
            stim_frames_meta = set(metadata['stimulus_frame_num'])
            stim_frames_chaser = set(chaser['stimulus_frame_num'])
            
            missing = stim_frames_meta - stim_frames_chaser
            if len(missing) == 0:
                self.log("  ✅ All stimulus frames have chaser states!")
            else:
                self.log(f"  ⚠️  {len(missing)} stimulus frames missing chaser states")
            
            # Check against bounding boxes
            if '/tracking_data/bounding_boxes' in f:
                bboxes = f['/tracking_data/bounding_boxes'][:]
                bbox_frames = set(bboxes['payload_frame_id'])
                meta_frames = set(metadata['triggering_camera_frame_id'])
                
                missing_bbox = bbox_frames - meta_frames
                if len(missing_bbox) == 0:
                    self.log("  ✅ All bbox frames have metadata!")
                else:
                    self.log(f"  ⚠️  {len(missing_bbox)} bbox frames not in metadata")
    
    def run(self):
        """Execute the complete fix."""
        self.log("🚀 Starting hybrid interpolation fix...")
        
        # Analyze gaps
        has_gaps = self.analyze_gaps()
        
        if not has_gaps:
            self.log("\n✨ No gaps found - file is complete!")
            return False
        
        # Interpolate both metadata and chaser
        metadata, chaser, mask = self.interpolate_both()
        
        # Create output file
        self.create_output_file(metadata, chaser, mask)
        
        # Verify
        self.verify_output()
        
        self.log("\n✨ Fix complete!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid fix for H5 files - interpolates both metadata and chaser states properly'
    )
    
    parser.add_argument('input_h5', help='Input H5 file')
    parser.add_argument('-o', '--output', help='Output H5 file (default: input.fixed.h5)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    interpolator = HybridH5Interpolator(
        input_h5=args.input_h5,
        output_h5=args.output,
        verbose=not args.quiet
    )
    
    success = interpolator.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())