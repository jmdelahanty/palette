#!/usr/bin/env python3
"""
Patch legacy HDF5 files to add /enums group with event type mappings.

This script adds the enum mapping tables that newer versions of Citrus
automatically create, allowing legacy datasets to work with updated analysis scripts.

CORRECTED VERSION: Uses actual C++ enum values, not sequential indices!
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# Complete event type mappings from Citrus (0-54+)
EVENT_TYPE_MAPPINGS = {
    # Core Protocol Events (0-23)
    0: "PROTOCOL_START",
    1: "PROTOCOL_STOP",
    2: "PROTOCOL_PAUSE",
    3: "PROTOCOL_RESUME",
    4: "PROTOCOL_FINISH",
    5: "PROTOCOL_CLEAR",
    6: "PROTOCOL_LOAD",
    7: "STEP_ADD",
    8: "STEP_REMOVE",
    9: "STEP_MOVE_UP",
    10: "STEP_MOVE_DOWN",
    11: "STEP_START",
    12: "STEP_END",
    13: "ITI_START",
    14: "ITI_END",
    15: "PARAMS_APPLIED",
    16: "MANAGER_REINIT",
    17: "MANAGER_REINIT_FAIL",
    18: "LOOM_AUTO_REPEAT_TRIGGER",
    19: "ERROR_RUNTIME",
    20: "ERROR_CUDA",
    21: "ERROR_OPENGL",
    22: "WARNING_GENERAL",
    23: "INFO_GENERAL",
    
    # Chaser Events (24-48)
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END",
    29: "CHASER_ITI_START",
    30: "CHASER_ITI_END",
    31: "CHASER_TARGET_VISIBLE",
    32: "CHASER_TARGET_HIDDEN",
    33: "CHASER_POSITIONING_START",
    34: "CHASER_POSITIONING_END",
    35: "CHASER_APPROACHING",
    36: "CHASER_LOOM_START",
    37: "CHASER_LOOM_MAX_SIZE",
    38: "CHASER_ESCAPE_TRIGGERED",
    39: "CHASER_RETREAT_START",
    40: "CHASER_RETREAT_END",
    41: "CHASER_RANDOM_TARGET_SET",
    42: "CHASER_RANDOM_TARGET_REACHED",
    43: "CHASER_CAVE_DEFENSE_START",
    44: "CHASER_CAVE_DEFENSE_END",
    45: "CHASER_CAVE_EMERGE_START",
    46: "CHASER_CAVE_APPROACHING",
    47: "CHASER_CAVE_RETURN_START",
    48: "CHASER_CAVE_RETURN_END",
    
    # Grid Events (49-54)
    49: "GRID_PHASE_START",
    50: "GRID_TRANSITION",
    51: "GRID_MOTION_START",
    52: "GRID_MOTION_STOP",
    53: "GRID_IMAGE_SWAP",
    54: "GRID_SPEED_CHANGED",
    55: "GRID_DIRECTION_CHANGED",
}

# CORRECTED: Stimulus mode mappings using ACTUAL C++ enum values!
# From: src/core/stimulus_globals.h
# enum class Type : int {
#     UNDEFINED = -1,
#     COHERENT_DOTS = 2,    // NOTE: Starts at 2, not 0!
#     ...
#     CHASER = 12,          // NOT 11!
#     CALIBRATION_TEST_SHAPE = 13,  // NOT 12!
#     ...
#     NONE = 99             // NOT 16!
# };
STIMULUS_MODE_MAPPINGS = {
    -1: "UNDEFINED",                 # ← Negative value!
    2: "COHERENT_DOTS",              # ← Starts at 2, not 0!
    3: "MOVING_GRATING",
    4: "SOLID_BLACK",
    5: "SOLID_WHITE",
    6: "CONCENTRIC_GRATING",
    7: "LOOMING_DOT",
    8: "STATIC_IMAGE",
    9: "CALIBRATION_GRID",
    10: "ARENA_DEFINITION_SQUARE",
    11: "SPOTLIGHT",
    12: "CHASER",                    # ← The critical one!
    13: "CALIBRATION_TEST_SHAPE",
    14: "SCROLLING_GRID",
    15: "INDEPENDENT_MOTION_GRID",
    16: "MOVING_DOTS",
    99: "NONE",                      # ← Jump to 99!
}

# Chaser trial state mappings (these are correct - sequential 0,1,2)
CHASER_TRIAL_STATE_MAPPINGS = {
    0: "PRE_PERIOD",
    1: "TRAINING",
    2: "POST_PERIOD",
}


def create_enum_mapping_dtype(max_name_len: int = 128):
    """Create numpy dtype for enum mapping records."""
    return np.dtype([
        ('id', 'i4'),
        ('name', f'S{max_name_len}')
    ])


def patch_hdf5_file(
    file_path: Path, 
    backup: bool = True,
    overwrite: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Add /enums group to an HDF5 file.
    
    Args:
        file_path: Path to HDF5 file to patch
        backup: Create backup before modifying
        overwrite: Overwrite existing /enums group if present
        dry_run: Just check, don't actually modify
    
    Returns:
        True if successful (or would be successful in dry_run)
    """
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    if dry_run:
        print(f"🔍 DRY RUN: Would patch {file_path}")
    else:
        print(f"📝 Patching {file_path}")
    
    # Create backup if requested
    if backup and not dry_run:
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if backup_path.exists():
            print(f"   ⚠️  Backup already exists: {backup_path}")
            print(f"   ℹ️  Using existing backup, not creating new one")
        else:
            print(f"   💾 Creating backup: {backup_path}")
            shutil.copy2(file_path, backup_path)
    
    try:
        # Open file in read-only mode for dry run
        mode = 'r' if dry_run else 'a'
        with h5py.File(file_path, mode) as f:
            # Check if /enums already exists
            if 'enums' in f:
                if not overwrite:
                    print(f"   ✓ /enums already exists, skipping")
                    print(f"   ℹ️  Use --overwrite to replace existing /enums")
                    return True
                else:
                    if dry_run:
                        print(f"   ⚠️  Would overwrite existing /enums group")
                    else:
                        print(f"   ⚠️  Overwriting existing /enums group")
                        del f['enums']
            
            if dry_run:
                print(f"   ✓ Would add /enums group with:")
                print(f"      - {len(EVENT_TYPE_MAPPINGS)} event types")
                print(f"      - {len(STIMULUS_MODE_MAPPINGS)} stimulus modes (CORRECTED VALUES)")
                print(f"      - {len(CHASER_TRIAL_STATE_MAPPINGS)} chaser trial states")
                return True
            
            # Create /enums group
            enums_group = f.create_group('enums')
            
            # Create dtype for enum mappings
            dtype = create_enum_mapping_dtype()
            
            # Add events dataset
            events_data = np.zeros(len(EVENT_TYPE_MAPPINGS), dtype=dtype)
            for i, (event_id, event_name) in enumerate(sorted(EVENT_TYPE_MAPPINGS.items())):
                events_data[i]['id'] = event_id
                events_data[i]['name'] = event_name.encode('utf-8')
            
            enums_group.create_dataset('events', data=events_data, dtype=dtype)
            print(f"   ✓ Added {len(EVENT_TYPE_MAPPINGS)} event type mappings")
            
            # Add stimulus_modes dataset WITH CORRECT ENUM VALUES
            modes_data = np.zeros(len(STIMULUS_MODE_MAPPINGS), dtype=dtype)
            for i, (mode_id, mode_name) in enumerate(sorted(STIMULUS_MODE_MAPPINGS.items())):
                modes_data[i]['id'] = mode_id
                modes_data[i]['name'] = mode_name.encode('utf-8')
            
            enums_group.create_dataset('stimulus_modes', data=modes_data, dtype=dtype)
            print(f"   ✓ Added {len(STIMULUS_MODE_MAPPINGS)} stimulus mode mappings (CORRECTED)")
            print(f"      Key correction: CHASER is ID 12 (not 11)")
            
            # Add chaser_trial_states dataset
            states_data = np.zeros(len(CHASER_TRIAL_STATE_MAPPINGS), dtype=dtype)
            for i, (state_id, state_name) in enumerate(sorted(CHASER_TRIAL_STATE_MAPPINGS.items())):
                states_data[i]['id'] = state_id
                states_data[i]['name'] = state_name.encode('utf-8')
            
            enums_group.create_dataset('chaser_trial_states', data=states_data, dtype=dtype)
            print(f"   ✓ Added {len(CHASER_TRIAL_STATE_MAPPINGS)} chaser trial state mappings")
            
            print(f"   ✅ Successfully patched {file_path}")
            return True
            
    except Exception as e:
        print(f"   ❌ Error patching file: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_patch_directory(
    directory: Path,
    pattern: str = "*.h5",
    backup: bool = True,
    overwrite: bool = False,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Patch all HDF5 files in a directory.
    
    Returns:
        (success_count, total_count)
    """
    files = list(directory.glob(pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in {directory}")
        return 0, 0
    
    print(f"\nFound {len(files)} files to patch")
    print("=" * 60)
    
    success_count = 0
    for file_path in files:
        if patch_hdf5_file(file_path, backup=backup, overwrite=overwrite, dry_run=dry_run):
            success_count += 1
        print()  # Blank line between files
    
    return success_count, len(files)


def inspect_file(file_path: Path):
    """Show what's currently in a file's /enums group (if any)."""
    print(f"\n📊 Inspecting {file_path}")
    print("=" * 60)
    
    try:
        with h5py.File(file_path, 'r') as f:
            if 'enums' not in f:
                print("❌ No /enums group found in file")
                print("   This file needs to be patched.")
                return
            
            enums_group = f['enums']
            print(f"✓ /enums group exists with {len(enums_group.keys())} datasets:")
            
            for dataset_name in enums_group.keys():
                dataset = enums_group[dataset_name]
                print(f"\n  📋 {dataset_name}:")
                print(f"     Entries: {len(dataset)}")
                
                # Show first few entries
                data = dataset[:]
                for i, record in enumerate(data[:5]):
                    enum_id = record['id']
                    enum_name = record['name'].decode('utf-8').rstrip('\x00')
                    print(f"       {enum_id}: {enum_name}")
                
                if len(data) > 5:
                    print(f"       ... and {len(data) - 5} more")
                
                # Check for the critical CHASER mapping
                if dataset_name == 'stimulus_modes':
                    print(f"\n     🔍 Checking CHASER mapping:")
                    chaser_records = [r for r in data if b'CHASER' in r['name'] and b'CALIBRATION' not in r['name']]
                    if chaser_records:
                        chaser_id = chaser_records[0]['id']
                        if chaser_id == 12:
                            print(f"        ✅ CORRECT: CHASER is at ID 12")
                        else:
                            print(f"        ❌ WRONG: CHASER is at ID {chaser_id} (should be 12)")
                            print(f"           This file needs --overwrite to fix")
                    
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch legacy HDF5 files to add /enums group (CORRECTED VERSION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a file to see if it needs patching
  python patch_legacy_h5_CORRECTED.py experiment.h5 --inspect
  
  # Patch a single file (creates backup)
  python patch_legacy_h5_CORRECTED.py experiment.h5
  
  # Dry run to see what would happen
  python patch_legacy_h5_CORRECTED.py experiment.h5 --dry-run
  
  # Fix files that were patched with the OLD INCORRECT script
  python patch_legacy_h5_CORRECTED.py experiment.h5 --overwrite
  
  # Patch all .h5 files in a directory
  python patch_legacy_h5_CORRECTED.py /path/to/data --batch
  
  # Patch without creating backups (not recommended!)
  python patch_legacy_h5_CORRECTED.py experiment.h5 --no-backup
  
  # Force overwrite existing /enums group
  python patch_legacy_h5_CORRECTED.py experiment.h5 --overwrite
  
NOTE: This corrected version uses the actual C++ enum values where:
  - CHASER = 12 (not 11)
  - CALIBRATION_TEST_SHAPE = 13 (not 12)
  - NONE = 99 (not 16)
        """
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="Path to HDF5 file or directory"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all .h5 files in directory"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.h5",
        help="File pattern for batch mode (default: *.h5)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create .backup files (not recommended)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing /enums group if present (use this to fix old incorrect mappings!)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files"
    )
    
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Just inspect the file, don't modify"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("CORRECTED Legacy HDF5 Patcher")
    print("=" * 80)
    print("This version uses the correct C++ enum values from stimulus_globals.h")
    print("Key corrections:")
    print("  • CHASER = 12 (was incorrectly 11 in old script)")
    print("  • CALIBRATION_TEST_SHAPE = 13 (was incorrectly 12 in old script)")
    print("  • NONE = 99 (was incorrectly 16 in old script)")
    print("=" * 80)
    
    # Inspect mode
    if args.inspect:
        if args.path.is_file():
            inspect_file(args.path)
        else:
            print(f"Error: {args.path} is not a file")
        return
    
    # Batch mode
    if args.batch:
        if not args.path.is_dir():
            print(f"Error: {args.path} is not a directory")
            return
        
        success, total = batch_patch_directory(
            args.path,
            pattern=args.pattern,
            backup=not args.no_backup,
            overwrite=args.overwrite,
            dry_run=args.dry_run
        )
        
        print("=" * 60)
        if args.dry_run:
            print(f"DRY RUN: Would successfully patch {success}/{total} files")
        else:
            print(f"Successfully patched {success}/{total} files")
        
        if success < total:
            print(f"⚠️  {total - success} files failed")
    
    # Single file mode
    else:
        if not args.path.is_file():
            print(f"Error: {args.path} is not a file")
            return
        
        success = patch_hdf5_file(
            args.path,
            backup=not args.no_backup,
            overwrite=args.overwrite,
            dry_run=args.dry_run
        )
        
        if not success:
            exit(1)


if __name__ == "__main__":
    main()