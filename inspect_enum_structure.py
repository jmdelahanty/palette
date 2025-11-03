#!/usr/bin/env python3
"""Quick script to show what the current enum structure looks like in memory."""

import numpy as np
from src.fisheye.utils.patch_legacy_h5 import (
    EVENT_TYPE_MAPPINGS,
    STIMULUS_MODE_MAPPINGS,
    CHASER_TRIAL_STATE_MAPPINGS,
    create_enum_mapping_dtype
)

print("=" * 80)
print("CURRENT ENUM STRUCTURE (as stored in H5/Zarr)")
print("=" * 80)

# Create example enum data as it's currently stored
dtype = create_enum_mapping_dtype()
print(f"\nDtype: {dtype}")
print(f"Field names: {dtype.names}")
print(f"Field types: {[dtype.fields[name][0] for name in dtype.names]}")

# Create a small sample
events_data = np.zeros(5, dtype=dtype)
for i, (event_id, event_name) in enumerate(list(EVENT_TYPE_MAPPINGS.items())[:5]):
    events_data[i]['id'] = event_id
    events_data[i]['name'] = event_name.encode('utf-8')

print("\n" + "=" * 80)
print("SAMPLE: First 5 event enum entries")
print("=" * 80)
print(f"\nStructured array shape: {events_data.shape}")
print(f"Structured array dtype: {events_data.dtype}")
print("\nData (as numpy sees it):")
print(events_data)

print("\n" + "-" * 80)
print("Accessing fields:")
print("-" * 80)
print(f"IDs only: {events_data['id']}")
print(f"Names only: {events_data['name']}")

print("\n" + "=" * 80)
print("WHAT COLUMNAR FORMAT WOULD LOOK LIKE")
print("=" * 80)
print("\nInstead of:")
print("  analysis/enums/events/events          <- structured array [('id', 'i4'), ('name', 'S128')]")
print("\nWould be:")
print("  analysis/enums/events/events/         <- GROUP")
print("    ├── id                               <- int32 array [0, 1, 2, 3, 4, ...]")
print("    └── name                             <- string array ['PROTOCOL_START', 'PROTOCOL_STOP', ...]")

print("\n" + "=" * 80)
print("COMPARISON: Current vs Proposed")
print("=" * 80)

print("\n📦 CURRENT (Structured Array):")
print(f"   Type: Single Zarr array with compound dtype")
print(f"   Access pattern: zarr_group['events'][:]  -> full structured array")
print(f"   Field access: data['id'] or data['name']  (after loading)")

print("\n📦 PROPOSED (Columnar):")
print(f"   Type: Group containing separate arrays")
print(f"   Access pattern: zarr_group['events/id'][:] or zarr_group['events/name'][:]")
print(f"   Field access: Direct, no reconstruction needed")

print("\n" + "=" * 80)
print("ACTUAL DATA COUNTS")
print("=" * 80)
print(f"Event types: {len(EVENT_TYPE_MAPPINGS)} entries")
print(f"Stimulus modes: {len(STIMULUS_MODE_MAPPINGS)} entries")
print(f"Chaser trial states: {len(CHASER_TRIAL_STATE_MAPPINGS)} entries")

print("\n" + "=" * 80)
print("SAMPLE OF ALL ENUM TABLES")
print("=" * 80)

print("\n📋 Event Types (first 10):")
for i, (id, name) in enumerate(list(EVENT_TYPE_MAPPINGS.items())[:10]):
    print(f"   {id:3d} → {name}")
print(f"   ... ({len(EVENT_TYPE_MAPPINGS) - 10} more)")

print("\n📋 Stimulus Modes (all):")
for id, name in sorted(STIMULUS_MODE_MAPPINGS.items())[:10]:
    print(f"   {id:3d} → {name}")
if len(STIMULUS_MODE_MAPPINGS) > 10:
    print(f"   ... ({len(STIMULUS_MODE_MAPPINGS) - 10} more)")

print("\n📋 Chaser Trial States (all):")
for id, name in CHASER_TRIAL_STATE_MAPPINGS.items():
    print(f"   {id:3d} → {name}")

print("\n" + "=" * 80)
