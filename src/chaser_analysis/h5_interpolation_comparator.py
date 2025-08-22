#!/usr/bin/env python3
"""
H5 Interpolation Fields Comparator

Compare only the interpolation-related fields between two H5 files
to diagnose why interpolated frames aren't being visualized.
Focuses on the specific fields used by the visualization tool.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Set, List
import json


class InterpolationComparator:
    """Compare interpolation-specific fields between two H5 files."""
    
    def __init__(self, old_file: str, new_file: str):
        self.old_path = Path(old_file)
        self.new_path = Path(new_file)
        self.results = {}
    
    def compare(self):
        """Run focused comparison on interpolation-related fields."""
        print("=" * 80)
        print("🔍 INTERPOLATION FIELDS COMPARISON")
        print("=" * 80)
        print(f"OLD (working): {self.old_path.name}")
        print(f"NEW (not showing interpolated): {self.new_path.name}")
        print("=" * 80)
        
        with h5py.File(self.old_path, 'r') as old_f:
            with h5py.File(self.new_path, 'r') as new_f:
                # 1. Check interpolation mask
                self.compare_interpolation_mask(old_f, new_f)
                
                # 2. Check frame metadata structure
                self.compare_frame_metadata(old_f, new_f)
                
                # 3. Check chaser states
                self.compare_chaser_states(old_f, new_f)
                
                # 4. Check alignment between datasets
                self.check_data_alignment(old_f, new_f)
                
                # 5. Check analysis group attributes
                self.compare_analysis_attributes(old_f, new_f)
        
        # Print diagnosis
        self.print_diagnosis()
    
    def compare_interpolation_mask(self, old_f, new_f):
        """Compare the interpolation mask structure and content."""
        print("\n" + "=" * 60)
        print("1️⃣  INTERPOLATION MASK (/analysis/interpolation_mask)")
        print("-" * 60)
        
        mask_path = '/analysis/interpolation_mask'
        
        # Check existence
        old_has = mask_path in old_f
        new_has = mask_path in new_f
        
        print(f"Exists in OLD: {old_has}")
        print(f"Exists in NEW: {new_has}")
        
        if not (old_has and new_has):
            print("⚠️  Cannot compare - mask missing in one file")
            return
        
        old_mask = old_f[mask_path][:]
        new_mask = new_f[mask_path][:]
        
        # Basic stats
        print(f"\nMask length:")
        print(f"  OLD: {len(old_mask)}")
        print(f"  NEW: {len(new_mask)}")
        
        print(f"\nInterpolated frames (False values):")
        old_interp = np.sum(~old_mask)
        new_interp = np.sum(~new_mask)
        print(f"  OLD: {old_interp} ({old_interp/len(old_mask)*100:.1f}%)")
        print(f"  NEW: {new_interp} ({new_interp/len(new_mask)*100:.1f}%)")
        
        # Check attributes
        print(f"\nMask attributes:")
        old_attrs = dict(old_f[mask_path].attrs)
        new_attrs = dict(new_f[mask_path].attrs)
        print(f"  OLD: {old_attrs}")
        print(f"  NEW: {new_attrs}")
        
        self.results['mask'] = {
            'old_length': len(old_mask),
            'new_length': len(new_mask),
            'old_interpolated': old_interp,
            'new_interpolated': new_interp
        }
    
    def compare_frame_metadata(self, old_f, new_f):
        """Compare frame metadata structure."""
        print("\n" + "=" * 60)
        print("2️⃣  FRAME METADATA (/video_metadata/frame_metadata)")
        print("-" * 60)
        
        meta_path = '/video_metadata/frame_metadata'
        
        if meta_path not in old_f or meta_path not in new_f:
            print("⚠️  frame_metadata missing")
            return
        
        old_meta = old_f[meta_path]
        new_meta = new_f[meta_path]
        
        # Check dtypes
        print(f"Dtype fields:")
        print(f"  OLD: {old_meta.dtype.names}")
        print(f"  NEW: {new_meta.dtype.names}")
        
        # Check shape
        print(f"\nShape:")
        print(f"  OLD: {old_meta.shape}")
        print(f"  NEW: {new_meta.shape}")
        
        # Check attributes related to interpolation
        print(f"\nInterpolation-related attributes:")
        for attr in ['interpolated', 'original_records', 'total_records']:
            old_val = old_meta.attrs.get(attr, 'NOT PRESENT')
            new_val = new_meta.attrs.get(attr, 'NOT PRESENT')
            print(f"  {attr}:")
            print(f"    OLD: {old_val}")
            print(f"    NEW: {new_val}")
        
        # Sample stimulus frame numbers
        old_data = old_meta[:]
        new_data = new_meta[:]
        
        print(f"\nStimulus frame range:")
        print(f"  OLD: {old_data['stimulus_frame_num'].min()} - {old_data['stimulus_frame_num'].max()}")
        print(f"  NEW: {new_data['stimulus_frame_num'].min()} - {new_data['stimulus_frame_num'].max()}")
        
        self.results['metadata'] = {
            'old_shape': old_meta.shape,
            'new_shape': new_meta.shape,
            'old_stim_range': (int(old_data['stimulus_frame_num'].min()), 
                              int(old_data['stimulus_frame_num'].max())),
            'new_stim_range': (int(new_data['stimulus_frame_num'].min()),
                              int(new_data['stimulus_frame_num'].max()))
        }
    
    def compare_chaser_states(self, old_f, new_f):
        """Compare chaser states structure and coverage."""
        print("\n" + "=" * 60)
        print("3️⃣  CHASER STATES (/tracking_data/chaser_states)")
        print("-" * 60)
        
        chaser_path = '/tracking_data/chaser_states'
        
        if chaser_path not in old_f or chaser_path not in new_f:
            print("⚠️  chaser_states missing")
            return
        
        old_chaser = old_f[chaser_path]
        new_chaser = new_f[chaser_path]
        
        # Check dtypes - these should match exactly
        print(f"Required fields present:")
        required_fields = ['stimulus_frame_num', 'timestamp_ns_session', 'chaser_index',
                          'is_chasing', 'chaser_pos_x', 'chaser_pos_y', 
                          'target_pos_x', 'target_pos_y']
        
        for field in required_fields:
            old_has = field in old_chaser.dtype.names if old_chaser.dtype.names else False
            new_has = field in new_chaser.dtype.names if new_chaser.dtype.names else False
            symbol = "✅" if (old_has and new_has) else "❌"
            print(f"  {symbol} {field}: OLD={old_has}, NEW={new_has}")
        
        # Check shape
        print(f"\nTotal chaser state records:")
        print(f"  OLD: {old_chaser.shape[0]}")
        print(f"  NEW: {new_chaser.shape[0]}")
        
        # Get unique stimulus frames
        old_chaser_data = old_chaser[:]
        new_chaser_data = new_chaser[:]
        
        old_unique_frames = set(old_chaser_data['stimulus_frame_num'])
        new_unique_frames = set(new_chaser_data['stimulus_frame_num'])
        
        print(f"\nUnique stimulus frames with chaser data:")
        print(f"  OLD: {len(old_unique_frames)}")
        print(f"  NEW: {len(new_unique_frames)}")
        
        print(f"\nStimulus frame range in chaser states:")
        print(f"  OLD: {min(old_unique_frames)} - {max(old_unique_frames)}")
        print(f"  NEW: {min(new_unique_frames)} - {max(new_unique_frames)}")
        
        self.results['chaser'] = {
            'old_records': old_chaser.shape[0],
            'new_records': new_chaser.shape[0],
            'old_unique_frames': len(old_unique_frames),
            'new_unique_frames': len(new_unique_frames),
            'old_unique_frames_set': old_unique_frames,
            'new_unique_frames_set': new_unique_frames
        }
    
    def check_data_alignment(self, old_f, new_f):
        """Check alignment between mask, metadata, and chaser states."""
        print("\n" + "=" * 60)
        print("4️⃣  DATA ALIGNMENT CHECK")
        print("-" * 60)
        
        # OLD FILE
        print("\n📁 OLD FILE:")
        if '/analysis/interpolation_mask' in old_f and '/video_metadata/frame_metadata' in old_f:
            old_mask = old_f['/analysis/interpolation_mask'][:]
            old_meta = old_f['/video_metadata/frame_metadata'][:]
            
            print(f"  Mask length == Metadata length? {len(old_mask) == len(old_meta)} "
                  f"({len(old_mask)} vs {len(old_meta)})")
            
            # Check if interpolated frames have chaser states
            if '/tracking_data/chaser_states' in old_f:
                old_chaser = old_f['/tracking_data/chaser_states'][:]
                interpolated_indices = np.where(~old_mask)[0]
                
                if len(interpolated_indices) > 0:
                    # Get stimulus frames for interpolated entries
                    interpolated_stim_frames = old_meta[interpolated_indices]['stimulus_frame_num']
                    chaser_stim_frames = set(old_chaser['stimulus_frame_num'])
                    
                    # Check coverage
                    covered = sum(1 for f in interpolated_stim_frames if f in chaser_stim_frames)
                    print(f"  Interpolated frames with chaser states: {covered}/{len(interpolated_stim_frames)} "
                          f"({covered/len(interpolated_stim_frames)*100:.1f}%)")
                    
                    # Show some examples
                    missing = [f for f in interpolated_stim_frames if f not in chaser_stim_frames]
                    if missing:
                        print(f"  Example missing stimulus frames: {missing[:5]}")
        
        # NEW FILE
        print("\n📁 NEW FILE:")
        if '/analysis/interpolation_mask' in new_f and '/video_metadata/frame_metadata' in new_f:
            new_mask = new_f['/analysis/interpolation_mask'][:]
            new_meta = new_f['/video_metadata/frame_metadata'][:]
            
            print(f"  Mask length == Metadata length? {len(new_mask) == len(new_meta)} "
                  f"({len(new_mask)} vs {len(new_meta)})")
            
            # Check if interpolated frames have chaser states
            if '/tracking_data/chaser_states' in new_f:
                new_chaser = new_f['/tracking_data/chaser_states'][:]
                interpolated_indices = np.where(~new_mask)[0]
                
                if len(interpolated_indices) > 0:
                    # Get stimulus frames for interpolated entries
                    interpolated_stim_frames = new_meta[interpolated_indices]['stimulus_frame_num']
                    chaser_stim_frames = set(new_chaser['stimulus_frame_num'])
                    
                    # Check coverage
                    covered = sum(1 for f in interpolated_stim_frames if f in chaser_stim_frames)
                    print(f"  Interpolated frames with chaser states: {covered}/{len(interpolated_stim_frames)} "
                          f"({covered/len(interpolated_stim_frames)*100:.1f}%)")
                    
                    # Show some examples
                    missing = [f for f in interpolated_stim_frames if f not in chaser_stim_frames]
                    if missing:
                        print(f"  Example missing stimulus frames: {missing[:5]}")
                    
                    self.results['alignment_issue'] = {
                        'interpolated_without_chaser': len(missing),
                        'total_interpolated': len(interpolated_stim_frames)
                    }
    
    def compare_analysis_attributes(self, old_f, new_f):
        """Compare analysis group attributes."""
        print("\n" + "=" * 60)
        print("5️⃣  ANALYSIS GROUP ATTRIBUTES")
        print("-" * 60)
        
        if '/analysis' not in old_f or '/analysis' not in new_f:
            print("⚠️  Analysis group missing")
            return
        
        old_analysis = old_f['/analysis']
        new_analysis = new_f['/analysis']
        
        # Check interpolation-related attributes
        interp_attrs = ['interpolated_frames', 'missing_frames', 'total_camera_frames',
                       'original_frames', 'interpolation_method', 'has_camera_frame_sync']
        
        print("Interpolation attributes:")
        for attr in interp_attrs:
            old_val = old_analysis.attrs.get(attr, 'NOT PRESENT')
            new_val = new_analysis.attrs.get(attr, 'NOT PRESENT')
            if old_val != 'NOT PRESENT' or new_val != 'NOT PRESENT':
                print(f"  {attr}:")
                print(f"    OLD: {old_val}")
                print(f"    NEW: {new_val}")
    
    def print_diagnosis(self):
        """Print diagnostic summary."""
        print("\n" + "=" * 80)
        print("🔬 DIAGNOSIS")
        print("=" * 80)
        
        issues = []
        
        # Check if NEW file has interpolated frames but no chaser states for them
        if 'alignment_issue' in self.results:
            issue = self.results['alignment_issue']
            if issue['interpolated_without_chaser'] > 0:
                issues.append(f"❌ CRITICAL: {issue['interpolated_without_chaser']}/{issue['total_interpolated']} "
                            f"interpolated frames have NO chaser states!")
                issues.append("   This is why interpolated frames aren't being drawn.")
                issues.append("   The visualization needs chaser position data to draw anything.")
        
        # Check mask/metadata alignment
        if 'mask' in self.results and 'metadata' in self.results:
            if self.results['mask']['new_length'] != self.results['metadata']['new_shape'][0]:
                issues.append(f"⚠️  Mask length doesn't match metadata length in NEW file")
        
        # Check if there are any interpolated frames at all
        if 'mask' in self.results:
            if self.results['mask']['new_interpolated'] == 0:
                issues.append("⚠️  NEW file has NO interpolated frames in mask")
            elif self.results['mask']['old_interpolated'] > 0 and self.results['mask']['new_interpolated'] > 0:
                issues.append(f"✅ Both files have interpolated frames "
                            f"(OLD: {self.results['mask']['old_interpolated']}, "
                            f"NEW: {self.results['mask']['new_interpolated']})")
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("✅ No obvious issues found")
        
        print("\n💡 SOLUTION:")
        if 'alignment_issue' in self.results and self.results['alignment_issue']['interpolated_without_chaser'] > 0:
            print("The NEW file only interpolated frame_metadata but NOT chaser_states.")
            print("The visualization can't draw chaser/target positions without the actual position data.")
            print("\nTo fix this, you need to:")
            print("1. Interpolate chaser states for the missing frames, OR")
            print("2. Use a flag to skip drawing when chaser data is missing, OR")
            print("3. Modify the create_analysis_h5.py script to also interpolate chaser states")


def main():
    parser = argparse.ArgumentParser(
        description='Compare interpolation-specific fields between H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Focuses only on fields relevant to interpolated frame visualization:
- Interpolation mask
- Frame metadata  
- Chaser states
- Alignment between these datasets

This tool helps diagnose why interpolated frames might not be
visualized correctly in one file vs another.

Example:
  %(prog)s old_working.h5 new_broken.h5
        """
    )
    
    parser.add_argument('old_file', help='Path to old/working H5 file')
    parser.add_argument('new_file', help='Path to new/problematic H5 file')
    
    args = parser.parse_args()
    
    comparator = InterpolationComparator(
        old_file=args.old_file,
        new_file=args.new_file
    )
    
    comparator.compare()
    
    return 0


if __name__ == '__main__':
    exit(main())