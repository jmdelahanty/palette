#!/usr/bin/env python3
"""
Zarr File Inspector

Comprehensive tool to inspect the structure and contents of zarr files
created by the fish tracking preprocessing pipeline.

Shows:
- Complete hierarchy
- Dataset shapes and types
- Attributes and metadata
- Processing history
- Coverage statistics
- Memory usage
"""

import zarr
import numpy as np
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List


class ZarrInspector:
    """Inspect and report on zarr file structure and contents."""
    
    def __init__(self, zarr_path: str):
        """Initialize inspector with zarr file."""
        self.zarr_path = Path(zarr_path)
        self.root = zarr.open(str(self.zarr_path), mode='r')
        self.total_size = 0
        
    def format_bytes(self, bytes_val: int) -> str:
        """Format bytes into human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"
    
    def format_timestamp(self, timestamp: str) -> str:
        """Format ISO timestamp to readable format."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp
    
    def inspect_array(self, array: zarr.Array, indent: int = 0) -> Dict:
        """Inspect a zarr array and return info."""
        info = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'chunks': array.chunks,
            'size_bytes': array.nbytes,
            'size_stored': array.nbytes_stored,
            'compression': array.compressor.codec_id if array.compressor else 'none'
        }
        
        # Sample statistics for numeric arrays
        if array.size > 0 and np.issubdtype(array.dtype, np.number):
            try:
                data_sample = array[:]
                if data_sample.size < 1000000:  # Only for reasonably sized arrays
                    info['stats'] = {
                        'min': float(np.nanmin(data_sample)),
                        'max': float(np.nanmax(data_sample)),
                        'mean': float(np.nanmean(data_sample)),
                        'coverage': float((~np.isnan(data_sample)).mean()) if data_sample.ndim == 1 else None
                    }
            except:
                pass
        
        return info
    
    def inspect_group(self, group: zarr.Group, name: str = '/', level: int = 0) -> Dict:
        """Recursively inspect a zarr group."""
        indent = "  " * level
        info = {
            'name': name,
            'type': 'group',
            'attrs': dict(group.attrs),
            'children': {}
        }
        
        # Process subgroups
        for key in sorted(group.group_keys()):
            try:
                subgroup = group[key]
                info['children'][key] = self.inspect_group(subgroup, key, level + 1)
            except Exception as e:
                # Handle any issues with accessing subgroups
                info['children'][key] = {
                    'name': key,
                    'type': 'group',
                    'error': str(e),
                    'children': {}
                }
        
        # Process arrays
        for key in sorted(group.array_keys()):
            try:
                array = group[key]
                info['children'][key] = {
                    'name': key,
                    'type': 'array',
                    'info': self.inspect_array(array, level + 1)
                }
                self.total_size += array.nbytes_stored
            except Exception as e:
                # Handle any issues with accessing arrays
                info['children'][key] = {
                    'name': key,
                    'type': 'array',
                    'error': str(e)
                }
        
        return info
    
    def print_tree(self, info: Dict, level: int = 0):
        """Print tree structure of zarr file."""
        indent = "  " * level
        prefix = "├─" if level > 0 else ""
        
        if 'error' in info:
            # Handle error cases
            print(f"{indent}{prefix}❌ {info['name']} (Error: {info['error']})")
            return
        
        if info['type'] == 'group':
            # Print group name
            group_name = info['name'] if info['name'] != '/' else 'ROOT'
            print(f"{indent}{prefix}📁 {group_name}/")
            
            # Print attributes if any
            if info['attrs'] and level < 2:  # Only show attrs for top levels
                for key, value in info['attrs'].items():
                    if key in ['created_at', 'updated_at']:
                        value = self.format_timestamp(str(value))
                    elif isinstance(value, (int, float)):
                        value = f"{value}"
                    elif isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    print(f"{indent}  │ @{key}: {value}")
            
            # Print children
            for key, child in info['children'].items():
                self.print_tree(child, level + 1)
        
        elif info['type'] == 'array':
            # Print array info
            if 'error' in info:
                print(f"{indent}{prefix}❌ {info['name']} (Error: {info['error']})")
            else:
                arr_info = info['info']
                size_str = self.format_bytes(arr_info['size_stored'])
                print(f"{indent}{prefix}📊 {info['name']} {arr_info['shape']} {arr_info['dtype']} ({size_str})")
                
                # Print statistics if available
                if 'stats' in arr_info and level < 3:
                    stats = arr_info['stats']
                    if stats.get('coverage') is not None and stats['coverage'] < 1.0:
                        print(f"{indent}  │ coverage: {stats['coverage']*100:.1f}%")
    
    def generate_report(self) -> str:
        """Generate comprehensive report of zarr contents."""
        report = []
        report.append("=" * 80)
        report.append("ZARR FILE INSPECTION REPORT")
        report.append("=" * 80)
        report.append(f"File: {self.zarr_path}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic info
        root_info = self.inspect_group(self.root)
        
        # Dataset counts
        n_arrays = 0
        n_groups = 0
        
        def count_items(info):
            nonlocal n_arrays, n_groups
            if info['type'] == 'group':
                n_groups += 1
                for child in info['children'].values():
                    count_items(child)
            else:
                n_arrays += 1
        
        count_items(root_info)
        
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total size on disk: {self.format_bytes(self.total_size)}")
        report.append(f"Number of groups: {n_groups}")
        report.append(f"Number of arrays: {n_arrays}")
        
        # Root attributes
        if self.root.attrs:
            report.append("")
            report.append("ROOT METADATA")
            report.append("-" * 40)
            for key, value in self.root.attrs.items():
                if isinstance(value, (int, float)):
                    report.append(f"{key}: {value}")
                else:
                    report.append(f"{key}: {str(value)[:100]}")
        
        # Check for calibration data
        report.append("")
        report.append("CALIBRATION STATUS")
        report.append("-" * 40)
        
        has_calibration = False
        if 'calibration' in self.root:
            has_calibration = True
            cal = self.root['calibration']
            report.append("✓ Calibration data found:")
            for key in cal.attrs.keys():
                report.append(f"  - {key}: {cal.attrs[key]}")
        else:
            report.append("✗ No calibration data found")
            report.append("  Suggested calibration metrics to add:")
            report.append("  - pixel_to_mm: Conversion factor from pixels to millimeters")
            report.append("  - arena_diameter_mm: Physical arena size")
            report.append("  - water_depth_mm: Water depth in the arena")
            report.append("  - camera_fps: Actual measured frame rate")
            report.append("  - camera_model: Camera model and settings")
        
        # Processing pipeline status
        report.append("")
        report.append("PREPROCESSING PIPELINE STATUS")
        report.append("-" * 40)
        
        # Original data
        if 'n_detections' in self.root:
            n_det = self.root['n_detections'][:]
            coverage = (n_det > 0).sum() / len(n_det) * 100
            report.append(f"1. Original data: {coverage:.1f}% coverage")
        
        # Filtered data
        if 'filtered_runs' in self.root:
            if 'latest' in self.root['filtered_runs'].attrs:
                latest = self.root['filtered_runs'].attrs['latest']
                n_det = self.root['filtered_runs'][latest]['n_detections'][:]
                coverage = (n_det > 0).sum() / len(n_det) * 100
                report.append(f"2. Filtered data ({latest}): {coverage:.1f}% coverage")
                
                # Show what was filtered
                if 'frames_dropped' in self.root['filtered_runs'][latest].attrs:
                    n_dropped = self.root['filtered_runs'][latest].attrs['frames_dropped']
                    report.append(f"   - Frames removed: {n_dropped}")
        else:
            report.append("2. Filtered data: Not found")
        
        # Interpolated data
        if 'preprocessing' in self.root:
            if 'latest' in self.root['preprocessing'].attrs:
                latest = self.root['preprocessing'].attrs['latest']
                n_det = self.root['preprocessing'][latest]['n_detections'][:]
                coverage = (n_det > 0).sum() / len(n_det) * 100
                report.append(f"3. Interpolated data ({latest}): {coverage:.1f}% coverage")
                
                # Show interpolation info
                if 'history' in self.root['preprocessing'][latest].attrs:
                    history = json.loads(self.root['preprocessing'][latest].attrs['history'])
                    for step in history:
                        if step['step'] == 'interpolate_gaps':
                            report.append(f"   - Max gap filled: {step['params']['max_gap']} frames")
                            report.append(f"   - Gaps filled: {step.get('gaps_filled', '?')}")
        else:
            report.append("3. Interpolated data: Not found")
        
        # Analysis results
        report.append("")
        report.append("ANALYSIS RESULTS")
        report.append("-" * 40)
        
        if 'behavior_metrics' in self.root:
            metrics = self.root['behavior_metrics']
            report.append("✓ Behavior metrics found:")
            report.append(f"  - Source: {metrics.attrs.get('source_name', 'unknown')}")
            report.append(f"  - Created: {self.format_timestamp(metrics.attrs.get('created_at', 'unknown'))}")
            
            if 'distance' in metrics:
                dist = metrics['distance']
                report.append(f"  - Total distance: {dist.attrs.get('total_distance', 0):.1f} pixels")
                report.append(f"  - Mean movement: {dist.attrs.get('mean_distance_per_frame', 0):.2f} px/frame")
            
            if 'speed' in metrics:
                speed = metrics['speed']
                report.append(f"  - Mean speed: {speed.attrs.get('mean_speed', 0):.1f} px/s")
                report.append(f"  - Max speed: {speed.attrs.get('max_speed', 0):.1f} px/s")
        else:
            report.append("✗ No behavior metrics found")
            report.append("  Run: python fish_behavior_metrics.py <zarr_file>")
        
        # Data quality warnings
        report.append("")
        report.append("DATA QUALITY NOTES")
        report.append("-" * 40)
        
        if not has_calibration:
            report.append("⚠ No calibration data - all metrics in pixels, not physical units")
        
        if 'preprocessing' in self.root:
            latest = self.root['preprocessing'].attrs.get('latest', '')
            if latest:
                n_det = self.root['preprocessing'][latest]['n_detections'][:]
                coverage = (n_det > 0).sum() / len(n_det) * 100
                if coverage < 95:
                    report.append(f"⚠ Coverage below 95% ({coverage:.1f}%) - consider adjusting parameters")
        
        return "\n".join(report)
    
    def print_full_tree(self):
        """Print complete tree structure."""
        print("\nFILE STRUCTURE")
        print("=" * 80)
        root_info = self.inspect_group(self.root)
        self.print_tree(root_info)
        print(f"\nTotal size: {self.format_bytes(self.total_size)}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect zarr file structure and contents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection with tree view
  %(prog)s detections.zarr
  
  # Generate detailed report
  %(prog)s detections.zarr --report
  
  # Save report to file
  %(prog)s detections.zarr --report --output report.txt
  
  # Show only summary
  %(prog)s detections.zarr --summary
        """
    )
    
    parser.add_argument('zarr_path', help='Path to zarr file')
    
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed text report')
    
    parser.add_argument('--output', '-o',
                       help='Save report to file')
    
    parser.add_argument('--summary', action='store_true',
                       help='Show only summary statistics')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = ZarrInspector(args.zarr_path)
    
    if args.summary:
        # Just show basic stats
        root_info = inspector.inspect_group(inspector.root)
        print(f"File: {args.zarr_path}")
        print(f"Total size: {inspector.format_bytes(inspector.total_size)}")
        
        # Count items
        n_arrays = 0
        n_groups = 0
        
        def count_items(info):
            nonlocal n_arrays, n_groups
            if info['type'] == 'group':
                n_groups += 1
                for child in info['children'].values():
                    count_items(child)
            else:
                n_arrays += 1
        
        count_items(root_info)
        print(f"Groups: {n_groups}, Arrays: {n_arrays}")
        
    elif args.report:
        # Generate full report
        report = inspector.generate_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)
    
    else:
        # Default: show tree structure
        inspector.print_full_tree()
        print("\n" + "=" * 80)
        print("Use --report for detailed analysis")
        print("Use --summary for quick statistics")
    
    return 0


if __name__ == '__main__':
    exit(main())