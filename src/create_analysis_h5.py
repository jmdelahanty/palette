#!/usr/bin/env python3
"""
Create Complete Analysis H5 File with Frame Interpolation

This script creates a properly structured analysis H5 file from raw stimulus
output, including:
1. Interpolation of missing camera frames in metadata
2. Copying all essential datasets
3. Ensuring correct structure for downstream analysis
4. Adding analysis metadata and interpolation masks

Author: Your Name
Date: 2024
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class InterpolationStats:
    """Statistics about the interpolation process."""
    total_camera_frames: int
    original_frames: int
    missing_frames: int
    interpolated_frames: int
    gap_ranges: List[Tuple[int, int]]
    largest_gap: int
    interpolation_method: str = "linear"
    timestamp: str = ""


class AnalysisH5Creator:
    """
    Creates a complete analysis H5 file with frame interpolation and proper structure.
    """
    
    def __init__(self, source_h5_path: str, output_h5_path: str, verbose: bool = True):
        """
        Initialize the analysis H5 creator.
        
        Args:
            source_h5_path: Path to source stimulus H5 file
            output_h5_path: Path for output analysis H5 file
            verbose: Print detailed progress
        """
        self.source_path = Path(source_h5_path)
        self.output_path = Path(output_h5_path)
        self.verbose = verbose
        
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source H5 file not found: {self.source_path}")
        
        # Storage for interpolation info
        self.original_metadata = None
        self.interpolated_metadata = None
        self.interpolation_stats = None
        self.interpolation_mask = None
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def analyze_frame_gaps(self) -> InterpolationStats:
        """
        Analyze gaps in camera frame IDs.
        
        Returns:
            InterpolationStats object with gap analysis
        """
        self.log("\n📊 Analyzing frame gaps...")
        
        with h5py.File(self.source_path, 'r') as f:
            # Check for frame metadata
            if '/video_metadata/frame_metadata' not in f:
                self.log("  ⚠️  No frame metadata found - will create minimal structure")
                return None
            
            metadata = f['/video_metadata/frame_metadata'][:]
            self.original_metadata = metadata
            
            # Analyze camera frame IDs
            camera_ids = metadata['triggering_camera_frame_id'][:]
            unique_ids = np.unique(camera_ids)
            sorted_ids = np.sort(unique_ids)
            
            # Find missing frames
            full_range = np.arange(sorted_ids[0], sorted_ids[-1] + 1)
            missing_frames = np.setdiff1d(full_range, sorted_ids)
            
            # Find gap ranges
            gap_ranges = []
            if len(missing_frames) > 0:
                # Group consecutive missing frames
                gaps = np.split(missing_frames, np.where(np.diff(missing_frames) != 1)[0] + 1)
                gap_ranges = [(int(gap[0]), int(gap[-1])) for gap in gaps if len(gap) > 0]
            
            stats = InterpolationStats(
                total_camera_frames=int(sorted_ids[-1] - sorted_ids[0] + 1),
                original_frames=len(unique_ids),
                missing_frames=len(missing_frames),
                interpolated_frames=0,  # Will be updated after interpolation
                gap_ranges=gap_ranges,
                largest_gap=max([end - start + 1 for start, end in gap_ranges]) if gap_ranges else 0,
                timestamp=datetime.now().isoformat()
            )
            
            self.log(f"  📈 Frame range: {sorted_ids[0]} to {sorted_ids[-1]}")
            self.log(f"  ✅ Original frames: {stats.original_frames}")
            self.log(f"  ⚠️  Missing frames: {stats.missing_frames}")
            if gap_ranges:
                self.log(f"  🔍 Number of gaps: {len(gap_ranges)}")
                self.log(f"  📏 Largest gap: {stats.largest_gap} frames")
            
            return stats
    
    def interpolate_missing_frames(self) -> np.ndarray:
        """
        Interpolate missing camera frames in metadata.
        
        Returns:
            Combined metadata array with interpolated entries
        """
        if self.original_metadata is None:
            return None
        
        self.log("\n🔧 Interpolating missing frames...")
        
        # Build mappings
        camera_to_stim = {}
        stim_to_camera = {}
        camera_to_timestamp = {}
        
        for record in self.original_metadata:
            cam_id = int(record['triggering_camera_frame_id'])
            stim_id = int(record['stimulus_frame_num'])
            timestamp = int(record['timestamp_ns'])
            
            if cam_id not in camera_to_stim:
                camera_to_stim[cam_id] = []
                camera_to_timestamp[cam_id] = []
            
            camera_to_stim[cam_id].append(stim_id)
            camera_to_timestamp[cam_id].append(timestamp)
            stim_to_camera[stim_id] = cam_id
        
        # Determine stimulus frames per camera frame (usually 2 for 120Hz/60Hz)
        stim_per_camera = np.mean([len(stims) for stims in camera_to_stim.values()])
        self.log(f"  📊 Stimulus frames per camera frame: {stim_per_camera:.2f}")
        
        # Get missing frames
        camera_ids = sorted(camera_to_stim.keys())
        full_range = range(min(camera_ids), max(camera_ids) + 1)
        missing_frames = [f for f in full_range if f not in camera_to_stim]
        
        if not missing_frames:
            self.log("  ✅ No missing frames to interpolate")
            return self.original_metadata
        
        # Interpolate each missing frame
        interpolated_records = []
        
        for missing_frame in missing_frames:
            # Find surrounding frames
            prev_frame = max([f for f in camera_ids if f < missing_frame], default=None)
            next_frame = min([f for f in camera_ids if f > missing_frame], default=None)
            
            if prev_frame is None or next_frame is None:
                self.log(f"  ⚠️  Cannot interpolate frame {missing_frame} (boundary)")
                continue
            
            # Calculate interpolation weights
            total_gap = next_frame - prev_frame
            weight = (missing_frame - prev_frame) / total_gap
            
            # Interpolate timestamp
            prev_time = camera_to_timestamp[prev_frame][-1]  # Last timestamp
            next_time = camera_to_timestamp[next_frame][0]   # First timestamp
            interp_time = int(prev_time + (next_time - prev_time) * weight)
            
            # Calculate stimulus frame numbers
            prev_max_stim = max(camera_to_stim[prev_frame])
            next_min_stim = min(camera_to_stim[next_frame])
            
            # Estimate stimulus frames for this camera frame
            stim_gap = next_min_stim - prev_max_stim - 1
            stim_per_gap_frame = stim_gap / total_gap
            stim_offset = int((missing_frame - prev_frame) * stim_per_gap_frame)
            base_stim = prev_max_stim + stim_offset + 1
            
            # Create records (typically 2 per camera frame)
            num_stim_frames = int(round(stim_per_camera))
            
            for i in range(num_stim_frames):
                record = np.zeros(1, dtype=self.original_metadata.dtype)
                record['stimulus_frame_num'] = base_stim + i
                record['triggering_camera_frame_id'] = missing_frame
                # Add time offset for each stimulus frame (~8.33ms for 120Hz)
                record['timestamp_ns'] = interp_time + (i * 8333333)
                
                interpolated_records.append(record[0])
        
        self.log(f"  ✅ Created {len(interpolated_records)} interpolated records")
        self.log(f"     for {len(missing_frames)} missing camera frames")
        
        # Update stats
        if self.interpolation_stats:
            self.interpolation_stats.interpolated_frames = len(interpolated_records)
        
        # Combine and sort
        if interpolated_records:
            combined = np.concatenate([self.original_metadata, np.array(interpolated_records)])
            # Sort by stimulus frame number
            combined = combined[np.argsort(combined['stimulus_frame_num'])]
            
            # Create interpolation mask
            self.interpolation_mask = np.ones(len(combined), dtype=bool)
            
            # Mark interpolated entries
            original_pairs = set()
            for rec in self.original_metadata:
                pair = (int(rec['stimulus_frame_num']), int(rec['triggering_camera_frame_id']))
                original_pairs.add(pair)
            
            for i, rec in enumerate(combined):
                pair = (int(rec['stimulus_frame_num']), int(rec['triggering_camera_frame_id']))
                if pair not in original_pairs:
                    self.interpolation_mask[i] = False
            
            return combined
        
        return self.original_metadata
    
    def create_analysis_h5(self, 
                          copy_calibration: bool = True,
                          copy_events: bool = True,
                          copy_protocol: bool = True,
                          copy_tracking: bool = True):
        """
        Create the complete analysis H5 file.
        
        Args:
            copy_calibration: Copy calibration data including homography
            copy_events: Copy experimental events
            copy_protocol: Copy protocol snapshot
            copy_tracking: Copy tracking data (bounding boxes, chaser states)
        """
        self.log(f"\n💾 Creating analysis H5: {self.output_path}")
        
        # Analyze and interpolate if needed
        self.interpolation_stats = self.analyze_frame_gaps()
        
        if self.interpolation_stats and self.interpolation_stats.missing_frames > 0:
            self.interpolated_metadata = self.interpolate_missing_frames()
        elif self.original_metadata is not None:
            self.interpolated_metadata = self.original_metadata
            self.interpolation_mask = np.ones(len(self.original_metadata), dtype=bool)
        
        # Create output file
        with h5py.File(self.source_path, 'r') as src:
            with h5py.File(self.output_path, 'w') as dst:
                
                # Copy root attributes
                self.log("\n📋 Copying root attributes...")
                for attr_name, attr_value in src.attrs.items():
                    dst.attrs[attr_name] = attr_value
                
                # Add analysis metadata
                dst.attrs['analysis_created'] = datetime.now().isoformat()
                dst.attrs['analysis_source'] = str(self.source_path)
                dst.attrs['analysis_version'] = "1.0.0"
                
                # 1. Create video_metadata group with interpolated data
                if self.interpolated_metadata is not None:
                    self.log("📹 Creating video_metadata with interpolated frame data...")
                    video_group = dst.create_group('video_metadata')
                    
                    frame_meta_ds = video_group.create_dataset(
                        'frame_metadata',
                        data=self.interpolated_metadata,
                        compression='gzip',
                        compression_opts=4
                    )
                    
                    # Copy original attributes
                    if '/video_metadata/frame_metadata' in src:
                        for attr, val in src['/video_metadata/frame_metadata'].attrs.items():
                            frame_meta_ds.attrs[attr] = val
                    
                    frame_meta_ds.attrs['interpolated'] = True
                    frame_meta_ds.attrs['original_records'] = len(self.original_metadata) if self.original_metadata is not None else 0
                    frame_meta_ds.attrs['total_records'] = len(self.interpolated_metadata)
                
                # 2. Copy tracking_data group
                if copy_tracking and '/tracking_data' in src:
                    self.log("🎯 Copying tracking data...")
                    self._copy_group(src, dst, '/tracking_data')
                
                # 3. Copy calibration_snapshot
                if copy_calibration and '/calibration_snapshot' in src:
                    self.log("📐 Copying calibration data (including homography)...")
                    self._copy_group(src, dst, '/calibration_snapshot')
                elif copy_calibration:
                    self.log("  ⚠️  No calibration data found in source")
                
                # 4. Copy events
                if copy_events and '/events' in src:
                    self.log("⏰ Copying events data...")
                    self._copy_dataset(src, dst, '/events')
                elif copy_events:
                    self.log("  ⚠️  No events data found in source")
                
                # 5. Copy protocol_snapshot
                if copy_protocol and '/protocol_snapshot' in src:
                    self.log("📝 Copying protocol snapshot...")
                    self._copy_group(src, dst, '/protocol_snapshot')
                
                # 6. Create analysis group with interpolation info
                self.log("📊 Creating analysis group...")
                analysis_group = dst.create_group('analysis')
                
                # Add coordinate transformation metadata
                if '/calibration_snapshot/arena_config_json' in src:
                    arena_config_str = src['/calibration_snapshot/arena_config_json'][()].decode('utf-8')
                    arena_config = json.loads(arena_config_str)
                    
                    # Store coordinate system info
                    coord_info = {
                        'texture_dimensions': [358, 358],  # Standard texture size
                        'camera_dimensions': [4512, 4512],  # From camera calibration
                        'texture_to_camera_scale': 4512 / 358,  # 12.604
                        'coordinate_note': 'Chaser positions are in texture space (358x358), fish in camera space (4512x4512)'
                    }
                    
                    # Try to get actual dimensions from config
                    if 'camera_calibrations' in arena_config:
                        cam_calib = arena_config['camera_calibrations'][0]
                        coord_info['camera_dimensions'] = [
                            cam_calib['native_width_px'],
                            cam_calib['native_height_px']
                        ]
                        coord_info['texture_to_camera_scale'] = cam_calib['native_width_px'] / 358
                    
                    analysis_group.attrs['coordinate_transform'] = json.dumps(coord_info)
                    self.log("  ✅ Added coordinate transformation metadata")
                
                if self.interpolation_stats:
                    # Store interpolation statistics
                    analysis_group.attrs.update(asdict(self.interpolation_stats))
                    
                    # Store gap information as JSON
                    gap_info = {
                        'missing_frames': list(set(range(min(self.interpolation_stats.gap_ranges[0]) if self.interpolation_stats.gap_ranges else 0,
                                                        max(self.interpolation_stats.gap_ranges[-1]) if self.interpolation_stats.gap_ranges else 0 + 1))
                                                if self.interpolation_stats.gap_ranges else []),
                        'gap_ranges': self.interpolation_stats.gap_ranges,
                        'interpolation_method': self.interpolation_stats.interpolation_method
                    }
                    
                    gap_ds = analysis_group.create_dataset(
                        'gap_info',
                        data=json.dumps(gap_info, indent=2)
                    )
                    gap_ds.attrs['description'] = 'JSON string with gap analysis details'
                
                # Store interpolation mask
                if self.interpolation_mask is not None:
                    mask_ds = analysis_group.create_dataset(
                        'interpolation_mask',
                        data=self.interpolation_mask,
                        compression='gzip',
                        compression_opts=4
                    )
                    mask_ds.attrs['description'] = 'True for original data, False for interpolated'
                
                self.log("\n✅ Analysis H5 created successfully!")
                
                # Print summary
                if self.interpolation_stats:
                    self.log("\n📈 Summary:")
                    self.log(f"  - Original frames: {self.interpolation_stats.original_frames}")
                    self.log(f"  - Interpolated frames: {self.interpolation_stats.interpolated_frames}")
                    self.log(f"  - Total frames: {self.interpolation_stats.original_frames + self.interpolation_stats.interpolated_frames}")
                    if self.interpolation_stats.gap_ranges:
                        self.log(f"  - Gaps filled: {len(self.interpolation_stats.gap_ranges)}")
    
    def _copy_group(self, src_file, dst_file, group_path: str):
        """Recursively copy a group and all its contents."""
        if group_path not in src_file:
            return
        
        src_group = src_file[group_path]
        dst_group = dst_file.create_group(group_path)
        
        # Copy group attributes
        for attr_name, attr_value in src_group.attrs.items():
            dst_group.attrs[attr_name] = attr_value
        
        # Copy datasets in this group
        for name, obj in src_group.items():
            if isinstance(obj, h5py.Dataset):
                self._copy_dataset(src_file, dst_file, f"{group_path}/{name}")
            elif isinstance(obj, h5py.Group):
                self._copy_group(src_file, dst_file, f"{group_path}/{name}")
    
    def _copy_dataset(self, src_file, dst_file, dataset_path: str):
        """Copy a dataset with attributes."""
        if dataset_path not in src_file:
            return
        
        src_ds = src_file[dataset_path]
        
        # Handle scalar datasets differently
        if src_ds.shape == ():
            # Scalar dataset - read without slicing
            data = src_ds[()]
        else:
            # Array dataset - read with slicing
            data = src_ds[:]
        
        # Create parent groups if needed
        parent_path = '/'.join(dataset_path.split('/')[:-1])
        if parent_path and parent_path not in dst_file:
            dst_file.create_group(parent_path)
        
        # Create dataset with compression (only for non-scalar arrays)
        if np.isscalar(data) or (hasattr(data, 'shape') and data.shape == ()):
            # Scalar - no compression
            dst_ds = dst_file.create_dataset(dataset_path, data=data)
            self.log(f"    ✅ Copied {dataset_path}: scalar value, dtype={src_ds.dtype}")
        else:
            # Array - apply compression if large
            dst_ds = dst_file.create_dataset(
                dataset_path,
                data=data,
                compression='gzip' if data.size > 1000 else None,
                compression_opts=4 if data.size > 1000 else None
            )
            self.log(f"    ✅ Copied {dataset_path}: shape={data.shape}, dtype={data.dtype}")
        
        # Copy attributes
        for attr_name, attr_value in src_ds.attrs.items():
            dst_ds.attrs[attr_name] = attr_value


def main():
    parser = argparse.ArgumentParser(
        description='Create complete analysis H5 with frame interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates a complete analysis H5 file by:
1. Interpolating missing camera frames in metadata
2. Copying all essential datasets (tracking, calibration, events)
3. Ensuring proper structure for downstream analysis
4. Adding interpolation masks and statistics

Examples:
  # Create analysis H5 with all data
  %(prog)s stimulus.h5 out_analysis.h5
  
  # Skip certain data types
  %(prog)s stimulus.h5 out_analysis.h5 --no-events --no-protocol
  
  # Quiet mode
  %(prog)s stimulus.h5 out_analysis.h5 -q
        """
    )
    
    parser.add_argument(
        'source_h5',
        help='Path to source stimulus H5 file'
    )
    parser.add_argument(
        'output_h5',
        help='Path for output analysis H5 file'
    )
    parser.add_argument(
        '--no-calibration',
        action='store_true',
        help="Don't copy calibration data"
    )
    parser.add_argument(
        '--no-events',
        action='store_true',
        help="Don't copy events data"
    )
    parser.add_argument(
        '--no-protocol',
        action='store_true',
        help="Don't copy protocol snapshot"
    )
    parser.add_argument(
        '--no-tracking',
        action='store_true',
        help="Don't copy tracking data"
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Create analysis H5
    creator = AnalysisH5Creator(
        source_h5_path=args.source_h5,
        output_h5_path=args.output_h5,
        verbose=not args.quiet
    )
    
    creator.create_analysis_h5(
        copy_calibration=not args.no_calibration,
        copy_events=not args.no_events,
        copy_protocol=not args.no_protocol,
        copy_tracking=not args.no_tracking
    )
    
    return 0


if __name__ == '__main__':
    exit(main())