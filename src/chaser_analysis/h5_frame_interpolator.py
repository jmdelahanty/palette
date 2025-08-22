#!/usr/bin/env python3
"""
Script to create an analysis version of H5 files with interpolated frame metadata.
This specifically handles gaps in the triggering_camera_frame_id dataset.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json


class H5FrameMetadataInterpolator:
    """Interpolates missing frame metadata in H5 files."""
    
    def __init__(self, input_h5_path, output_h5_path=None, verbose=True):
        self.input_path = Path(input_h5_path)
        self.output_path = output_h5_path or self.input_path.with_suffix('.analysis.h5')
        self.verbose = verbose
        self.gaps_info = {
            'camera_frame_gaps': [],
            'interpolated_frames': [],
            'statistics': {}
        }
        
    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def analyze_gaps(self):
        """Analyze gaps in triggering_camera_frame_id."""
        self.log(f"\n📊 Analyzing gaps in: {self.input_path}")
        
        with h5py.File(self.input_path, 'r') as f:
            if '/video_metadata/frame_metadata' not in f:
                raise ValueError("Dataset '/video_metadata/frame_metadata' not found!")
            
            metadata_ds = f['/video_metadata/frame_metadata']
            self.original_metadata = metadata_ds[:]
            
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
            
            # Analyze gap patterns
            self.log(f"  🔍 Found {len(missing_frames)} missing camera frame IDs")
            
            # Group consecutive missing frames into gap ranges
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
            
            # Print gap summary
            self.log(f"\n  📍 Gap Summary:")
            for i, (start, end) in enumerate(gaps[:10]):  # Show first 10 gaps
                if start == end:
                    self.log(f"    Gap {i+1}: Frame {start}")
                else:
                    self.log(f"    Gap {i+1}: Frames {start}-{end} ({end-start+1} frames)")
            
            if len(gaps) > 10:
                self.log(f"    ... and {len(gaps)-10} more gaps")
            
            # Analyze gap periodicity
            if len(gaps) > 1:
                gap_positions = [g[0] for g in gaps]
                gap_intervals = np.diff(gap_positions)
                
                self.gaps_info['statistics'] = {
                    'total_gaps': len(gaps),
                    'total_missing_frames': len(missing_frames),
                    'avg_gap_interval': float(np.mean(gap_intervals)) if len(gap_intervals) > 0 else 0,
                    'std_gap_interval': float(np.std(gap_intervals)) if len(gap_intervals) > 0 else 0,
                    'min_camera_frame': int(sorted_camera_ids[0]),
                    'max_camera_frame': int(sorted_camera_ids[-1])
                }
                
                # Check for periodic pattern (e.g., every ~60 frames)
                if len(gap_intervals) > 0:
                    avg_interval = np.mean(gap_intervals)
                    if 58 <= avg_interval <= 62:
                        self.log(f"  🎯 Detected periodic pattern: gaps every ~60 frames")
            
            return True
    
    def interpolate_metadata(self):
        """Create interpolated frame metadata for missing camera frames."""
        self.log("\n🔧 Interpolating frame metadata...")
        
        # Create mapping of stimulus_frame -> camera_frame from existing data
        stim_to_camera = {}
        camera_to_stim = {}
        
        for record in self.original_metadata:
            stim_frame = int(record['stimulus_frame_num'])
            camera_frame = int(record['triggering_camera_frame_id'])
            timestamp = int(record['timestamp_ns'])
            
            if stim_frame not in stim_to_camera:
                stim_to_camera[stim_frame] = []
            stim_to_camera[stim_frame].append((camera_frame, timestamp))
            
            if camera_frame not in camera_to_stim:
                camera_to_stim[camera_frame] = []
            camera_to_stim[camera_frame].append((stim_frame, timestamp))
        
        # Prepare interpolated records
        interpolated_records = []
        
        for missing_frame in self.gaps_info['missing_frames']:
            missing_frame = int(missing_frame)
            
            # Find surrounding frames for interpolation
            existing_frames = sorted(camera_to_stim.keys())
            
            # Find previous and next existing frames
            prev_frame = None
            next_frame = None
            
            for frame in existing_frames:
                if frame < missing_frame:
                    prev_frame = frame
                elif frame > missing_frame:
                    next_frame = frame
                    break
            
            if prev_frame is None or next_frame is None:
                self.log(f"  ⚠️  Cannot interpolate frame {missing_frame} - missing boundary data")
                continue
            
            # Get stimulus frames and timestamps from surrounding frames
            prev_stim_frames = [s for s, _ in camera_to_stim[prev_frame]]
            next_stim_frames = [s for s, _ in camera_to_stim[next_frame]]
            prev_timestamps = [t for _, t in camera_to_stim[prev_frame]]
            next_timestamps = [t for _, t in camera_to_stim[next_frame]]
            
            # Calculate interpolation factor
            frame_distance = next_frame - prev_frame
            interpolation_factor = (missing_frame - prev_frame) / frame_distance
            
            # Interpolate timestamp
            if prev_timestamps and next_timestamps:
                # Use the last timestamp from previous frame and first from next frame
                time_diff = next_timestamps[0] - prev_timestamps[-1]
                interp_timestamp = int(prev_timestamps[-1] + time_diff * interpolation_factor)
            else:
                interp_timestamp = 0
            
            # Estimate stimulus frame numbers for this camera frame
            # Find the pattern - usually 2 stimulus frames per camera frame
            prev_max_stim = max(prev_stim_frames)
            next_min_stim = min(next_stim_frames)
            
            # Calculate expected stimulus frame range
            stim_diff = next_min_stim - prev_max_stim - 1  # Number of stimulus frames in between
            expected_stim_per_camera = stim_diff / frame_distance
            
            # Calculate base stimulus frame for this missing camera frame
            stim_offset = int((missing_frame - prev_frame) * expected_stim_per_camera)
            estimated_stim_base = prev_max_stim + stim_offset + 1
            
            # Create interpolated records (typically 2 per camera frame for 120Hz/60Hz)
            num_stim_frames = 2 if expected_stim_per_camera >= 1.5 else 1
            
            for offset in range(num_stim_frames):
                new_record = np.zeros(1, dtype=self.original_metadata.dtype)
                new_record['stimulus_frame_num'] = estimated_stim_base + offset
                new_record['triggering_camera_frame_id'] = missing_frame
                # Add time offset for each stimulus frame (~8.33ms for 120Hz)
                new_record['timestamp_ns'] = interp_timestamp + (offset * 8333333)
                
                interpolated_records.append(new_record[0])
            
            self.gaps_info['interpolated_frames'].append({
                'camera_frame': int(missing_frame),
                'stimulus_frames': [estimated_stim_base + i for i in range(num_stim_frames)],
                'method': 'linear_interpolation',
                'prev_frame': int(prev_frame),
                'next_frame': int(next_frame)
            })
        
        self.log(f"  ✅ Created {len(interpolated_records)} interpolated metadata records")
        self.log(f"     for {len(self.gaps_info['interpolated_frames'])} missing camera frames")
        
        # Combine original and interpolated data
        if interpolated_records:
            self.combined_metadata = np.concatenate([self.original_metadata, 
                                                    np.array(interpolated_records)])
            # Sort by stimulus frame number
            sort_indices = np.argsort(self.combined_metadata['stimulus_frame_num'])
            self.combined_metadata = self.combined_metadata[sort_indices]
        else:
            self.combined_metadata = self.original_metadata
        
        return len(interpolated_records) > 0
    
    def create_analysis_file(self):
        """Create the output H5 file with interpolated frame metadata and tracking data."""
        self.log(f"\n💾 Creating analysis file: {self.output_path}")
        
        with h5py.File(self.output_path, 'w') as dst:
            # Create the video_metadata group
            video_meta_group = dst.create_group('video_metadata')
            
            # Create the interpolated frame_metadata dataset
            frame_meta_ds = video_meta_group.create_dataset(
                'frame_metadata', 
                data=self.combined_metadata,
                compression='gzip', 
                compression_opts=4
            )
            
            # Open source file to copy data
            with h5py.File(self.input_path, 'r') as src:
                # Copy attributes from original frame_metadata
                if '/video_metadata/frame_metadata' in src:
                    orig_ds = src['/video_metadata/frame_metadata']
                    for attr_name, attr_value in orig_ds.attrs.items():
                        frame_meta_ds.attrs[attr_name] = attr_value
                
                # Copy tracking_data group with bounding_boxes and chaser_states
                if '/tracking_data' in src:
                    self.log("  📦 Copying tracking data...")
                    tracking_group = dst.create_group('tracking_data')
                    
                    # Copy bounding_boxes dataset
                    if 'bounding_boxes' in src['/tracking_data']:
                        bbox_data = src['/tracking_data/bounding_boxes'][:]
                        bbox_ds = tracking_group.create_dataset(
                            'bounding_boxes',
                            data=bbox_data,
                            compression='gzip',
                            compression_opts=4
                        )
                        # Copy attributes
                        for attr_name, attr_value in src['/tracking_data/bounding_boxes'].attrs.items():
                            bbox_ds.attrs[attr_name] = attr_value
                        self.log(f"    ✅ Copied {len(bbox_data)} bounding box records")
                    
                    # Copy chaser_states dataset
                    if 'chaser_states' in src['/tracking_data']:
                        chaser_data = src['/tracking_data/chaser_states'][:]
                        chaser_ds = tracking_group.create_dataset(
                            'chaser_states',
                            data=chaser_data,
                            compression='gzip',
                            compression_opts=4
                        )
                        # Copy attributes
                        for attr_name, attr_value in src['/tracking_data/chaser_states'].attrs.items():
                            chaser_ds.attrs[attr_name] = attr_value
                        self.log(f"    ✅ Copied {len(chaser_data)} chaser state records")
                    
                    # Copy any other datasets in tracking_data
                    for dataset_name in src['/tracking_data'].keys():
                        if dataset_name not in ['bounding_boxes', 'chaser_states']:
                            try:
                                data = src[f'/tracking_data/{dataset_name}'][:]
                                ds = tracking_group.create_dataset(
                                    dataset_name,
                                    data=data,
                                    compression='gzip',
                                    compression_opts=4
                                )
                                for attr_name, attr_value in src[f'/tracking_data/{dataset_name}'].attrs.items():
                                    ds.attrs[attr_name] = attr_value
                                self.log(f"    ✅ Copied {dataset_name} dataset")
                            except:
                                self.log(f"    ⚠️  Could not copy {dataset_name}")
                
                # Copy calibration_snapshot group with homography
                if '/calibration_snapshot' in src:
                    self.log("  🔄 Copying calibration data...")
                    calib_src = src['/calibration_snapshot']
                    calib_dst = dst.create_group('calibration_snapshot')
                    
                    # Copy arena_config_json if it exists
                    if 'arena_config_json' in calib_src:
                        arena_config = calib_src['arena_config_json'][()]
                        calib_dst.create_dataset('arena_config_json', data=arena_config)
                        self.log("    ✅ Copied arena configuration")
                    
                    # Copy camera-specific calibration data
                    for cam_id in calib_src.keys():
                        if isinstance(calib_src[cam_id], h5py.Group):
                            self.log(f"    📷 Copying calibration for camera: {cam_id}")
                            cam_group = calib_dst.create_group(cam_id)
                            
                            # Copy homography matrix YAML
                            if 'homography_matrix_yml' in calib_src[cam_id]:
                                homography_yml = calib_src[cam_id]['homography_matrix_yml'][()]
                                cam_group.create_dataset('homography_matrix_yml', data=homography_yml)
                                self.log(f"      ✅ Copied homography matrix for {cam_id}")
                            
                            # Copy any attributes
                            for attr_name, attr_value in calib_src[cam_id].attrs.items():
                                cam_group.attrs[attr_name] = attr_value
            
            # Add analysis metadata group
            analysis_group = dst.create_group('analysis')
            analysis_group.attrs['creation_time'] = datetime.now().isoformat()
            analysis_group.attrs['interpolation_method'] = 'linear'
            analysis_group.attrs['source_file'] = str(self.input_path)
            analysis_group.attrs['original_records'] = len(self.original_metadata)
            analysis_group.attrs['interpolated_records'] = len(self.combined_metadata) - len(self.original_metadata)
            analysis_group.attrs['total_records'] = len(self.combined_metadata)
            
            # Create interpolation mask
            mask = np.ones(len(self.combined_metadata), dtype=bool)
            
            # Mark interpolated entries as False
            original_stim_camera_pairs = set()
            for record in self.original_metadata:
                pair = (int(record['stimulus_frame_num']), int(record['triggering_camera_frame_id']))
                original_stim_camera_pairs.add(pair)
            
            for i, record in enumerate(self.combined_metadata):
                pair = (int(record['stimulus_frame_num']), int(record['triggering_camera_frame_id']))
                if pair not in original_stim_camera_pairs:
                    mask[i] = False
            
            dst.create_dataset('analysis/interpolation_mask', 
                             data=mask,
                             compression='gzip', 
                             compression_opts=4)
            dst['analysis/interpolation_mask'].attrs['description'] = \
                'True for original data, False for interpolated data'
            
            # Save gap information as JSON string
            dst.create_dataset('analysis/gap_info', 
                             data=json.dumps(self.gaps_info, indent=2))
            dst['analysis/gap_info'].attrs['description'] = \
                'JSON string containing gap analysis and interpolation details'
                
        self.log(f"  ✅ Analysis file created successfully!")
    
    def generate_report(self, output_file=None):
        """Generate a detailed report of the interpolation process."""
        report_lines = [
            "=" * 70,
            "FRAME METADATA INTERPOLATION REPORT",
            "=" * 70,
            f"Source file: {self.input_path}",
            f"Output file: {self.output_path}",
            f"Processing time: {datetime.now().isoformat()}",
            "",
            "SUMMARY STATISTICS:",
            "-" * 40,
        ]
        
        if self.gaps_info['statistics']:
            stats = self.gaps_info['statistics']
            report_lines.extend([
                f"Total gaps found: {stats['total_gaps']}",
                f"Total missing frames: {stats['total_missing_frames']}",
                f"Camera frame range: {stats['min_camera_frame']} - {stats['max_camera_frame']}",
                f"Average gap interval: {stats['avg_gap_interval']:.1f} frames",
                f"Std dev of gap interval: {stats['std_gap_interval']:.1f} frames",
            ])
        
        report_lines.extend([
            "",
            "INTERPOLATION DETAILS:",
            "-" * 40,
            f"Interpolated frames: {len(self.gaps_info['interpolated_frames'])}",
            f"Original metadata records: {len(self.original_metadata)}",
            f"Final metadata records: {len(self.combined_metadata)}",
            f"New records created: {len(self.combined_metadata) - len(self.original_metadata)}",
        ])
        
        # Add first few interpolation examples
        if self.gaps_info['interpolated_frames']:
            report_lines.extend([
                "",
                "INTERPOLATION EXAMPLES (first 5):",
                "-" * 40,
            ])
            for entry in self.gaps_info['interpolated_frames'][:5]:
                report_lines.append(
                    f"Camera frame {entry['camera_frame']}: "
                    f"stimulus frames {entry['stimulus_frames']} "
                    f"(between camera frames {entry['prev_frame']} and {entry['next_frame']})"
                )
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.log(f"\n📄 Report saved to: {output_file}")
        else:
            print("\n" + report_text)
        
        return report_text
    
    def validate_output(self):
        """Validate the output file."""
        self.log("\n🔍 Validating output file...")
        
        with h5py.File(self.output_path, 'r') as f:
            if '/video_metadata/frame_metadata' not in f:
                self.log("  ❌ frame_metadata dataset not found in output!")
                return False
            
            metadata = f['/video_metadata/frame_metadata']
            camera_ids = np.sort(np.unique(metadata['triggering_camera_frame_id'][:]))
            
            # Check for gaps
            expected = np.arange(camera_ids[0], camera_ids[-1] + 1)
            remaining_gaps = np.setdiff1d(expected, camera_ids)
            
            if len(remaining_gaps) == 0:
                self.log("  ✅ Validation passed - no gaps in output file!")
                self.log(f"     Total camera frames: {len(camera_ids)}")
                self.log(f"     Frame range: {camera_ids[0]} to {camera_ids[-1]}")
                return True
            else:
                self.log(f"  ⚠️  Validation warning - {len(remaining_gaps)} gaps remain")
                return False
    
    def run(self):
        """Execute the complete interpolation pipeline."""
        self.log("🚀 Starting frame metadata interpolation pipeline...")
        
        # Step 1: Analyze gaps
        has_gaps = self.analyze_gaps()
        
        if not has_gaps:
            self.log("\n✨ No interpolation needed - file is complete!")
            return False
        
        # Step 2: Interpolate metadata
        interpolated = self.interpolate_metadata()
        
        if not interpolated:
            self.log("\n⚠️  No interpolation performed")
            return False
        
        # Step 3: Create output file
        self.create_analysis_file()
        
        self.log("\n✨ Interpolation complete!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Create analysis H5 file with interpolated frame metadata'
    )
    parser.add_argument('input_h5', help='Path to input H5 file')
    parser.add_argument('-o', '--output', help='Output H5 file path (default: input.analysis.h5)')
    parser.add_argument('-r', '--report', help='Save report to text file')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--validate', action='store_true', help='Validate output file after creation')
    
    args = parser.parse_args()
    
    # Create interpolator
    interpolator = H5FrameMetadataInterpolator(
        args.input_h5,
        args.output,
        verbose=not args.quiet
    )
    
    # Run interpolation
    success = interpolator.run()
    
    # Generate report
    if args.report or not args.quiet:
        interpolator.generate_report(args.report)
    
    # Validate if requested
    if args.validate and success:
        interpolator.validate_output()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())