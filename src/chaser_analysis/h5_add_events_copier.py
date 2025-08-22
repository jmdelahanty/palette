#!/usr/bin/env python3
"""
Script to add missing datasets (like /events) to an existing analysis H5 file.
This can be run on already-created analysis files to add the events data.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


def copy_events_to_analysis(original_h5_path: str, analysis_h5_path: str, verbose: bool = True):
    """
    Copy events dataset from original H5 to analysis H5 file.
    
    Args:
        original_h5_path: Path to original H5 file with events
        analysis_h5_path: Path to analysis H5 file to update
        verbose: Print progress messages
    """
    original_path = Path(original_h5_path)
    analysis_path = Path(analysis_h5_path)
    
    if not original_path.exists():
        raise FileNotFoundError(f"Original H5 file not found: {original_path}")
    
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis H5 file not found: {analysis_path}")
    
    if verbose:
        print(f"📂 Original file: {original_path}")
        print(f"📂 Analysis file: {analysis_path}")
        print()
    
    # Open both files
    with h5py.File(original_path, 'r') as src:
        with h5py.File(analysis_path, 'r+') as dst:  # r+ for read/write
            
            # Check if events already exists in destination
            if '/events' in dst:
                if verbose:
                    print("⚠️  /events already exists in analysis file")
                    response = input("Overwrite? (y/n): ").lower()
                    if response != 'y':
                        print("Skipping events copy")
                        return
                    del dst['/events']
            
            # Copy events dataset
            if '/events' in src:
                if verbose:
                    print("📋 Copying /events dataset...")
                
                events_data = src['/events'][:]
                events_ds = dst.create_dataset(
                    'events',
                    data=events_data,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Copy attributes
                for attr_name, attr_value in src['/events'].attrs.items():
                    events_ds.attrs[attr_name] = attr_value
                
                if verbose:
                    print(f"  ✅ Copied {len(events_data)} event records")
            else:
                if verbose:
                    print("  ❌ No /events dataset found in original file")
            
            # Also copy subject_metadata if it exists and is missing
            if '/subject_metadata' not in dst and '/subject_metadata' in src:
                if verbose:
                    print("📋 Copying /subject_metadata...")
                
                subject_group = dst.create_group('subject_metadata')
                
                # Copy all attributes
                for attr_name, attr_value in src['/subject_metadata'].attrs.items():
                    subject_group.attrs[attr_name] = attr_value
                
                if verbose:
                    print(f"  ✅ Copied subject metadata")
            
            # Copy protocol_snapshot if missing
            if '/protocol_snapshot' not in dst and '/protocol_snapshot' in src:
                if verbose:
                    print("📋 Copying /protocol_snapshot...")
                
                protocol_group = dst.create_group('protocol_snapshot')
                
                # Copy protocol_definition_json dataset
                if 'protocol_definition_json' in src['/protocol_snapshot']:
                    protocol_data = src['/protocol_snapshot/protocol_definition_json'][()]
                    protocol_group.create_dataset(
                        'protocol_definition_json',
                        data=protocol_data
                    )
                    
                    # Copy attributes
                    for attr_name, attr_value in src['/protocol_snapshot/protocol_definition_json'].attrs.items():
                        protocol_group['protocol_definition_json'].attrs[attr_name] = attr_value
                
                if verbose:
                    print(f"  ✅ Copied protocol snapshot")
            
            # Update metadata to indicate modification
            if '/analysis' in dst:
                dst['/analysis'].attrs['events_added_from'] = str(original_path)
                dst['/analysis'].attrs['events_added_at'] = datetime.now().isoformat()
            
            if verbose:
                print("\n✨ Successfully updated analysis file with events data!")


def list_datasets(h5_path: str):
    """List all datasets and groups in an H5 file."""
    print(f"\n📊 Structure of {Path(h5_path).name}:")
    print("-" * 50)
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  📄 {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  📁 {name}/")
    
    with h5py.File(h5_path, 'r') as f:
        f.visititems(print_structure)


def main():
    parser = argparse.ArgumentParser(
        description='Add events data from original H5 to analysis H5 file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add events to existing analysis file
  %(prog)s original.h5 analysis.h5
  
  # List contents of H5 files
  %(prog)s original.h5 analysis.h5 --list
  
  # Quiet mode
  %(prog)s original.h5 analysis.h5 -q
        """
    )
    
    parser.add_argument(
        'original_h5',
        help='Path to original H5 file containing events'
    )
    parser.add_argument(
        'analysis_h5',
        help='Path to analysis H5 file to update'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List contents of both files'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n🔍 Original file contents:")
        list_datasets(args.original_h5)
        print("\n🔍 Analysis file contents:")
        list_datasets(args.analysis_h5)
        print()
    
    # Copy events
    copy_events_to_analysis(
        args.original_h5,
        args.analysis_h5,
        verbose=not args.quiet
    )
    
    return 0


if __name__ == '__main__':
    exit(main())