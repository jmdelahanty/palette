# Experiment Types & Protocol Reference

Purpose: catalog the experiment types, protocols, stimulus modes, and hardware
configurations present in the Palette system as of 2026-02-28.

## Recording Types

Defined in registry schema (`src/fisheye/registry/db.py`, vocab tables).

| Recording Type | Subtypes | Behavior Mode | Description |
|---------------|----------|---------------|-------------|
| `behavior` | `free`, `embedded` | `free`, `embedded`, `none` | Primary recording type. Free-swimming fish in arena. |
| `microscopy` | `lightsheet`, `confocal`, `2p` | — | Imaging recordings. |
| `histology` | `section`, `wholemount` | — | Fixed tissue recordings. |

Current data is overwhelmingly `behavior` / `free` / `free`.

## Protocols (Current)

Only 2 protocols are populated in the registry (52 of 114 recordings):

| Protocol Name | Count | Description |
|--------------|-------|-------------|
| `Feeding` | 26 | Feeding behavior with no programmed visual stimulus. |
| `DefaultScreen` | 26 | Baseline/default screen protocol. |

62 recordings have no protocol data (60 training zarrs + 2 others).
Protocol extraction is blocked on a key-name mismatch in the H5 import path
(`protocol_json` vs `protocol_definition_json`). See `docs/protocol_parameter_registry_todo.md`.

## Stimulus Modes

17 stimulus types defined in the Citrus C++ enum `StimulusMode::Type`.
Source: `src/fisheye/utils/read_h5_data.py`, `src/fisheye/docs/citrus_data_structure_documentation.md`.

| ID | Name | Category | Description |
|----|------|----------|-------------|
| -1 | UNDEFINED | — | Uninitialized/error state |
| 2 | COHERENT_DOTS | Motion | Moving dots with coherent direction |
| 3 | MOVING_GRATING | Motion | Drifting sinusoidal grating (spatial freq, orientation, speed) |
| 4 | SOLID_BLACK | Static | Uniform black screen (baseline/ITI) |
| 5 | SOLID_WHITE | Static | Uniform white screen (contrast control) |
| 6 | CONCENTRIC_GRATING | Looming | Radial expanding/contracting grating |
| 7 | LOOMING_DOT | Looming | Simple expanding circle |
| 8 | STATIC_IMAGE | Static | Display a static image file |
| 9 | CALIBRATION_GRID | Calibration | Dot pattern for projector calibration |
| 10 | ARENA_DEFINITION_SQUARE | Calibration | Sub-arena boundary marker |
| 11 | SPOTLIGHT | Reactive | Reactive spotlight following a target |
| 12 | CHASER | Reactive/Looming | Complex looming/chasing agent (most parameterized) |
| 13 | CALIBRATION_TEST_SHAPE | Calibration | Test shape at specific mm size |
| 14 | SCROLLING_GRID | Motion | Grid of images that scroll |
| 15 | INDEPENDENT_MOTION_GRID | Motion | Grid segments with independent motion |
| 16 | MOVING_DOTS | Motion | Dots spawning and moving (prey-like) |
| 99 | NONE | — | No stimulus/blank |

### Chaser Loom Modes

The CHASER stimulus (ID 12) supports 6 looming behavior sub-modes:

| Mode | Name | Description |
|------|------|-------------|
| 0 | FIXED | Constant radius, no scaling |
| 1 | PROXIMITY | Size scales with proximity to target |
| 2 | VA_LOOM | Visual-angle looming (l/v ratio) with movement |
| 3 | STATIONARY | Biologically accurate visual-angle looming, stationary |
| 4 | CAVE_DEF | Cave dweller defensive (hides, then looms) |
| 5 | CAVE_AGG | Cave dweller aggressive (hides, then chases) |

## Hardware Configurations

### Rigs
- `omnifin0` — primary rig
- `omnifin1` — secondary rig

### Arenas
- `arena_1`, `arena_2` — arena positions on each rig

### Dish Designs
- `cedar` — cedar dish variant
- `cedar_shadow` — cedar dish with shadow/looming stimulus

### Cameras
- Identified by serial number (e.g., `2010093`, `2010096`)
- `camera_id` stored in provenance

### Canvas Names
- `DefaultScreen` — default stimulus display
- `Feeding` — feeding protocol display
- Maps to protocol names; represents the display/projection configuration

## Protocol Data Model

Protocols are structured hierarchically (see `docs/protocol_parameter_registry_todo.md`
for full architecture):

```
protocol
├── protocol_hash (SHA256, deduplication key)
├── protocol_name ("Feeding", "DefaultScreen", ...)
├── num_steps
└── steps[]
    ├── step_index
    ├── step_name
    ├── stimulus_mode ("CHASER", "MOVING_GRATING", etc.)
    ├── stimulus_mode_id
    ├── duration_s
    ├── iti_s (inter-trial interval)
    └── parameters (stimulus-specific params)
```

Protocol data lives in:
1. **Source H5 files**: `protocol_snapshot/protocol_definition_json`
2. **Zarr attrs**: `analysis/stimulus_runs/<run>/protocol_json` (currently empty due to import bug)
3. **Registry**: `provenance.protocol_name`, `provenance.protocol_hash` (currently NULL)

## Biological Variables

### Subject Metadata
- `cross_id` — genetic cross identifier
- `genotype` — genotype string
- `dpf_at_acquisition` — days post-fertilization at recording time
- `dish_id` — which dish the fish came from
- `fish_id` — individual fish identifier
- `line_strain` — transgenic line or strain

### Provenance Grouping
Datasets are typically grouped for analysis by:
- `(rig_id, arena_id, camera_id)` — hardware context
- `(dish_design, canvas_name, protocol_name)` — experimental context
- `(genotype, cross_id, dpf_at_acquisition)` — biological context

## Status of Protocol Registry Implementation

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | Partial | Fix H5 key mismatch, backfill protocol_json into zarrs. Script exists (`scripts/backfill_protocol_json.py`), 55 runs patched, but zarr attrs still empty. |
| 1 | Not started | Create `protocols` + `protocol_steps` tables, populate from zarr attrs. |
| 2 | Deferred | Per-stimulus-type parameter tables (chaser_step_params, grating_step_params, etc.). Waiting for real analysis workflow to drive requirements. |
| 3 | Deferred | Analysis result tables (chase_sequences, etc.). Requires analysis pipeline. |
| 4-5 | Deferred | Query CLI integration for protocol parameter filtering. |

The `raw_json` escape hatch in the `protocols` table (Phase 1) will cover ad-hoc
`json_extract()` queries until per-stimulus tables are needed.

## Key Files

- `docs/protocol_parameter_registry_todo.md` — full protocol registry architecture
- `src/fisheye/docs/citrus_data_structure_documentation.md` — Citrus protocol parameter reference
- `src/fisheye/utils/read_h5_data.py` — stimulus mode enum mappings
- `src/fisheye/analysis/import_stimulus_to_zarr.py` — stimulus import (has key mismatch)
- `scripts/backfill_protocol_json.py` — protocol data repair script
- `src/fisheye/registry/db.py` — recording type/subtype vocab tables (lines 2096-2121)
- `docs/session_context.md` — session metadata fields
- `docs/recording_manifest_contract.md` — recording manifest structure
