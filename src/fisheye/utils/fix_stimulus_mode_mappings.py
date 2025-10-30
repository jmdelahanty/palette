#!/usr/bin/env python3
"""
Fix incorrect stimulus_modes enum mappings in Citrus HDF5 files.

Problem: Some HDF5 files have sequential IDs (0, 1, 2, ...) instead of the 
actual C++ enum values (-1, 2, 3, 4, ..., 12, 13, 14, ...).

This script replaces the /enums/stimulus_modes table with the correct mappings
from the C++ enum definition.
"""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


# Correct stimulus mode mappings from stimulus_globals.h
# NOTE: These MUST match the actual C++ enum values, NOT sequential indices!
# From: enum class Type : int in src/core/stimulus_globals.h
CORRECT_STIMULUS_MODES = [
    (-1, "UNDEFINED"),                  # Enum value: -1 (not 0!)
    (2, "COHERENT_DOTS"),               # Enum value: 2 (not 1!)
    (3, "MOVING_GRATING"),              # Enum value: 3 (not 2!)
    (4, "SOLID_BLACK"),                 # Enum value: 4
    (5, "SOLID_WHITE"),                 # Enum value: 5
    (6, "CONCENTRIC_GRATING"),          # Enum value: 6
    (7, "LOOMING_DOT"),                 # Enum value: 7
    (8, "STATIC_IMAGE"),                # Enum value: 8
    (9, "CALIBRATION_GRID"),            # Enum value: 9
    (10, "ARENA_DEFINITION_SQUARE"),    # Enum value: 10
    (11, "SPOTLIGHT"),                  # Enum value: 11
    (12, "CHASER"),                     # Enum value: 12 ← THE KEY ONE!
    (13, "CALIBRATION_TEST_SHAPE"),     # Enum value: 13
    (14, "SCROLLING_GRID"),             # Enum value: 14
    (15, "INDEPENDENT_MOTION_GRID"),    # Enum value: 15
    (16, "MOVING_DOTS"),                # Enum value: 16
    (99, "NONE"),                       # Enum value: 99 (not 16!)
]

# Also provide the INCORRECT mappings that might exist in legacy files
# so we can detect and report them clearly
LEGACY_INCORRECT_MAPPINGS = {
    0: "UNDEFINED",           # Wrong! Should be -1
    1: "COHERENT_DOTS",       # Wrong! Should be 2
    11: "CHASER",             # Wrong! Should be 12
    12: "CALIBRATION_TEST_SHAPE",  # Wrong! Should be 13
    16: "NONE",               # Wrong! Should be 99
}


def verify_needs_fix(h5_path: Path) -> tuple[bool, str]:
    """
    Check if the file needs fixing by verifying CHASER is at ID 12.
    
    Returns:
        (needs_fix, reason)
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'enums' not in f or 'stimulus_modes' not in f['enums']:
                return (False, "No /enums/stimulus_modes dataset found")
            
            modes = f['enums/stimulus_modes'][:]
            
            # Check if CHASER is at the correct ID
            chaser_records = [m for m in modes if b'CHASER' in m['name']]
            if not chaser_records:
                return (True, "CHASER not found in mappings")
            
            chaser_id = chaser_records[0]['id']
            if chaser_id != 12:
                return (True, f"CHASER at wrong ID ({chaser_id} instead of 12)")
            
            # Check a few other key IDs
            for expected_id, expected_name in [(13, "CALIBRATION_TEST_SHAPE"), 
                                               (14, "SCROLLING_GRID")]:
                records = [m for m in modes if m['id'] == expected_id]
                if records:
                    actual_name = records[0]['name'].decode('utf-8').rstrip('\x00')
                    if actual_name != expected_name:
                        return (True, f"ID {expected_id} is '{actual_name}' instead of '{expected_name}'")
            
            return (False, "Mappings are correct")
    
    except Exception as e:
        return (False, f"Error checking file: {e}")


def fix_stimulus_modes(h5_path: Path, backup: bool = True, dry_run: bool = False):
    """
    Fix the stimulus_modes enum mappings in an HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 file
        backup: Whether to create a backup before modifying
        dry_run: If True, only report what would be done
    """
    print(f"\nProcessing: {h5_path}")
    
    # Check if fix is needed
    needs_fix, reason = verify_needs_fix(h5_path)
    
    if not needs_fix:
        print(f"  ✓ {reason}")
        return True
    
    print(f"  ⚠ Needs fix: {reason}")
    
    if dry_run:
        print("  → [DRY RUN] Would fix this file")
        return True
    
    # Create backup
    if backup:
        backup_path = h5_path.with_suffix(h5_path.suffix + '.backup')
        if backup_path.exists():
            print(f"  ⚠ Backup already exists: {backup_path}")
            response = input("  Overwrite backup? [y/N]: ")
            if response.lower() != 'y':
                print("  ✗ Skipped")
                return False
        
        print(f"  📦 Creating backup: {backup_path.name}")
        shutil.copy2(h5_path, backup_path)
    
    # Fix the file
    try:
        with h5py.File(h5_path, 'r+') as f:
            # Read old mappings for comparison
            old_modes = None
            if 'enums' in f and 'stimulus_modes' in f['enums']:
                old_modes = f['enums/stimulus_modes'][:]
                print(f"  📋 Old mappings had {len(old_modes)} entries")
            
            # Create or replace the stimulus_modes dataset
            if 'enums' not in f:
                f.create_group('enums')
            
            enums_group = f['enums']
            
            # Delete old dataset if it exists
            if 'stimulus_modes' in enums_group:
                del enums_group['stimulus_modes']
            
            # Create dtype for the mapping records
            dtype = np.dtype([
                ('id', np.int32),
                ('name', 'S64')  # 64-byte string
            ])
            
            # Create new data
            new_data = np.array(
                [(id_val, name.encode('utf-8')) for id_val, name in CORRECT_STIMULUS_MODES],
                dtype=dtype
            )
            
            # Write new dataset
            enums_group.create_dataset('stimulus_modes', data=new_data)
            
            print(f"  ✅ Fixed! Wrote {len(CORRECT_STIMULUS_MODES)} correct mappings")
            
            # Verify the fix
            needs_fix_after, reason_after = verify_needs_fix(h5_path)
            if needs_fix_after:
                print(f"  ⚠ WARNING: Verification failed after fix: {reason_after}")
                return False
            else:
                print(f"  ✓ Verification passed: {reason_after}")
                return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix incorrect stimulus_modes enum mappings in Citrus HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single file (dry run)
  python fix_stimulus_mode_mappings.py file.h5 --dry-run
  
  # Fix a single file with backup
  python fix_stimulus_mode_mappings.py file.h5
  
  # Fix multiple files
  python fix_stimulus_mode_mappings.py file1.h5 file2.h5 file3.h5
  
  # Fix all HDF5 files in a directory
  python fix_stimulus_mode_mappings.py /path/to/experiments/*.h5
        """
    )
    
    parser.add_argument(
        'files',
        type=Path,
        nargs='+',
        help='HDF5 file(s) to fix'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files (.backup)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only check files without modifying them'
    )
    
    args = parser.parse_args()
    
    # Process each file
    print("=" * 80)
    print("Citrus HDF5 Stimulus Mode Mapping Fixer")
    print("=" * 80)
    
    if args.dry_run:
        print("\n⚠ DRY RUN MODE - No files will be modified\n")
    
    files_to_process = []
    for file_pattern in args.files:
        # Handle glob patterns
        if '*' in str(file_pattern) or '?' in str(file_pattern):
            files_to_process.extend(Path().glob(str(file_pattern)))
        else:
            files_to_process.append(file_pattern)
    
    if not files_to_process:
        print("No files found to process")
        return
    
    print(f"Found {len(files_to_process)} file(s) to process\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for h5_path in files_to_process:
        if not h5_path.exists():
            print(f"\n✗ File not found: {h5_path}")
            error_count += 1
            continue
        
        result = fix_stimulus_modes(
            h5_path, 
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        if result is True:
            if args.dry_run:
                skip_count += 1
            else:
                success_count += 1
        else:
            if result is False:
                skip_count += 1
            else:
                error_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(files_to_process)}")
    if not args.dry_run:
        print(f"  ✅ Successfully fixed: {success_count}")
    print(f"  ⏭ Skipped (already correct): {skip_count}")
    print(f"  ✗ Errors: {error_count}")
    print("=" * 80)


if __name__ == '__main__':
    main()