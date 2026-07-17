# Protocol Parameter Registry TODO

## Goal
Make every protocol parameter for every stimulus type queryable through the
registry so that datasets can be selected by experimental conditions, not just
protocol name. Eventually support joining protocol parameters against per-event
analysis results (e.g. per-chase escape metrics) to answer questions like:
"aggregate all fish that went through a chaser protocol with >5% chase
probability, 120 s training, visual-angle looming, and had a successful escape."

## Current state (2026-02-27)
- `protocol_json` is **NOT** present in any zarr stimulus run attrs (0/64 runs
  checked). The import function (`import_stimulus_to_zarr.py:752`) looks for
  `protocol_snapshot/protocol_json` in the source H5, but the actual H5 key is
  `protocol_snapshot/protocol_definition_json`.
- The protocol data **does exist** in the source H5 files under
  `protocol_snapshot/protocol_definition_json` and contains structured data:
  `protocol_name`, `steps[]` with `stimulus_mode_str`, `duration_seconds`,
  `parameters`, etc.
- The registry's `provenance.protocol_name` and `provenance.protocol_hash` are
  NULL for all 114 rows because extraction depends on the missing zarr attr.
- `query.py` supports `--protocol` (exact name) and `--protocol-hash`, but
  both are non-functional since no rows have values.

## Critical prerequisite: fix the data pipeline

**None of the architecture below is useful until protocol data actually flows
into the zarrs.** The import path has a key name mismatch that must be fixed
first. See "Phase 0" in the implementation plan.

## Deferral note (2026-02-27)

Layers 2-3 (per-stimulus parameter tables, analysis result tables) and their
query CLI integration (Phase 4-5) are **deferred** until:
1. Phase 0 is complete and protocol data is in the zarrs + registry.
2. An analysis pipeline exists that would actually consume these queries.
3. At least one manual notebook-based analysis has been done to validate which
   parameters and joins are actually needed.

Building 10+ per-stimulus-type tables and a complex query CLI before having a
real analysis workflow to drive requirements is premature. The `raw_json` escape
hatch in the `protocols` table (Phase 1) covers ad-hoc `json_extract()` queries
in the interim.

## Architecture overview

Three layers, built in order.

### Layer 1 -- Protocol skeleton (shared, deduplicated)

Protocols are shared across many datasets. Deduplicate by `protocol_hash`.

```
protocols
---------
protocol_hash    TEXT PK      -- SHA256 (already computed)
protocol_name    TEXT
num_steps        INTEGER
raw_json         TEXT          -- full protocol_json escape hatch

protocol_steps
--------------
protocol_hash    TEXT   \
step_index       INTEGER > composite PK,  FK -> protocols
step_name        TEXT
stimulus_mode    TEXT          -- "CHASER", "MOVING_GRATING", etc.
stimulus_mode_id INTEGER
duration_s       REAL
iti_s            REAL
parameters_json  TEXT          -- full step params blob escape hatch
```

The existing `provenance.protocol_hash` column is the FK that links every
dataset to its protocol rows.

### Layer 2 -- Per-stimulus-type parameter tables

One table per stimulus type. Each row is keyed to a protocol step. Store only
the scientifically authoritative values (mm, degrees, seconds); skip
pixel-derived/computational duplicates and runtime state fields.

#### `chaser_step_params`
| Column                | Type    | Notes |
|-----------------------|---------|-------|
| protocol_hash         | TEXT    | PK/FK |
| step_index            | INTEGER | PK/FK |
| pre_period_s          | REAL    | |
| training_period_s     | REAL    | |
| post_period_s         | REAL    | |
| chase_probability_ps  | REAL    | per-second probability |
| chase_duration_s      | REAL    | |
| danger_zone_enabled   | INTEGER | bool |
| danger_zone_w_mm      | REAL    | |
| danger_zone_h_mm      | REAL    | |
| num_chasers           | INTEGER | len(chasers) |
| proximity_feedback    | INTEGER | bool |
| proximity_thresh_mm   | REAL    | |

#### `chaser_agents`
| Column                | Type    | Notes |
|-----------------------|---------|-------|
| protocol_hash         | TEXT    | PK/FK |
| step_index            | INTEGER | PK/FK |
| chaser_index          | INTEGER | PK |
| loom_mode             | INTEGER | 0=FIXED 1=PROXIMITY 2=VA_LOOM 3=STATIONARY 4=CAVE_DEF 5=CAVE_AGG |
| l_over_v_ms           | REAL    | |
| initial_distance_mm   | REAL    | |
| trigger_angle_deg     | REAL    | |
| max_angle_deg         | REAL    | |
| speed_mm_s            | REAL    | |
| radius_mm             | REAL    | |
| retreat_duration_s    | REAL    | |
| retreat_distance_mm   | REAL    | |
| random_movement       | INTEGER | bool |
| random_jump_interval  | REAL    | seconds |
| random_jump_min_mm    | REAL    | |
| random_jump_max_mm    | REAL    | |
| cave_trigger_radius_px| REAL    | no mm equivalent in C++ struct |
| cave_emerge_duration_s| REAL    | |
| cave_emerge_speed_mult| REAL    | |

#### `grating_step_params`
| Column             | Type | Notes |
|--------------------|------|-------|
| protocol_hash      | TEXT | PK/FK |
| step_index         | INTEGER | PK/FK |
| spatial_freq_cpmm  | REAL | cycles/mm |
| speed_mm_s         | REAL | |
| orientation_deg    | REAL | |
| duty_cycle         | REAL | |
| reactive_module    | TEXT | |

#### `looming_dot_step_params`
| Column              | Type    | Notes |
|---------------------|---------|-------|
| protocol_hash       | TEXT    | PK/FK |
| step_index          | INTEGER | PK/FK |
| start_radius_px     | REAL    | no mm equivalent in docs |
| end_radius_px       | REAL    | |
| loom_duration_s     | REAL    | |
| target_side         | INTEGER | 0=center 1=left 2=right |
| auto_repeat         | INTEGER | bool |
| inter_loom_interval_s | REAL  | |

#### `concentric_grating_step_params`
| Column            | Type    | Notes |
|-------------------|---------|-------|
| protocol_hash     | TEXT    | PK/FK |
| step_index        | INTEGER | PK/FK |
| spatial_freq_cpmm | REAL    | |
| speed_mm_s        | REAL    | |
| is_expanding      | INTEGER | bool |
| duty_cycle        | REAL    | |

#### `coherent_dots_step_params`
| Column          | Type    | Notes |
|-----------------|---------|-------|
| protocol_hash   | TEXT    | PK/FK |
| step_index      | INTEGER | PK/FK |
| num_dots        | INTEGER | |
| orientation_deg | REAL    | |
| speed_mm_s      | REAL    | |
| dot_radius_mm   | REAL    | |

#### `moving_dots_step_params`
| Column             | Type    | Notes |
|--------------------|---------|-------|
| protocol_hash      | TEXT    | PK/FK |
| step_index         | INTEGER | PK/FK |
| dot_radius_mm      | REAL    | |
| dot_speed_mm_s     | REAL    | |
| uniform_direction  | INTEGER | bool |
| direction_angle_deg| REAL    | |
| num_simultaneous   | INTEGER | |
| spawn_interval_s   | REAL    | |
| spawn_side         | TEXT    | |

#### `spotlight_step_params`
| Column          | Type | Notes |
|-----------------|------|-------|
| protocol_hash   | TEXT | PK/FK |
| step_index      | INTEGER | PK/FK |
| radius_mm       | REAL | |
| center_x_mm     | REAL | |
| center_y_mm     | REAL | |
| reactive_module | TEXT | |

#### `scrolling_grid_step_params`
| Column        | Type    | Notes |
|---------------|---------|-------|
| protocol_hash | TEXT    | PK/FK |
| step_index    | INTEGER | PK/FK |
| grid_rows     | INTEGER | |
| grid_cols     | INTEGER | |
| speed_mm_s    | REAL    | |
| direction_deg | REAL    | |

#### `independent_motion_grid_step_params`
| Column               | Type    | Notes |
|----------------------|---------|-------|
| protocol_hash        | TEXT    | PK/FK |
| step_index           | INTEGER | PK/FK |
| grid_rows            | INTEGER | |
| grid_cols            | INTEGER | |
| speed_mm_s           | REAL    | |
| direction_deg        | REAL    | |
| moving_segments_json | TEXT    | JSON array of segment indices |

#### `solid_color_step_params`
| Column        | Type    | Notes |
|---------------|---------|-------|
| protocol_hash | TEXT    | PK/FK |
| step_index    | INTEGER | PK/FK |
| color_type    | TEXT    | "black" or "white" |

#### `static_image_step_params`
| Column        | Type    | Notes |
|---------------|---------|-------|
| protocol_hash | TEXT    | PK/FK |
| step_index    | INTEGER | PK/FK |
| image_path    | TEXT    | |
| brightness    | REAL    | |

### Layer 3 -- Analysis result tables (future)

These would be populated by analysis pipelines, one row per event occurrence
per dataset. They reference `dataset_id` + `step_index` so they join naturally
to both protocol parameters and biological metadata in `provenance`.

#### `chase_sequences` (future)
| Column                | Type    | Notes |
|-----------------------|---------|-------|
| dataset_id            | TEXT    | FK -> datasets |
| step_index            | INTEGER | |
| chase_index           | INTEGER | Nth chase in this step |
| chaser_index          | INTEGER | which agent |
| start_time_ns         | INTEGER | |
| end_time_ns           | INTEGER | |
| duration_s            | REAL    | |
| escape_triggered      | INTEGER | bool |
| escape_latency_s      | REAL    | NULL if no escape |
| pre_chase_distance_mm | REAL    | |
| max_speed_mm_s        | REAL    | |
| mean_speed_mm_s       | REAL    | |
| end_reason            | TEXT    | "DURATION", "ESCAPE", etc. |
| in_danger_zone        | INTEGER | bool |

## Example queries

### All chaser recordings with >5% chase probability and 120 s training
```sql
SELECT d.dataset_id, d.zarr_path, csp.chase_probability_ps, csp.training_period_s
FROM datasets d
JOIN provenance p ON d.dataset_id = p.dataset_id
JOIN chaser_step_params csp ON p.protocol_hash = csp.protocol_hash
WHERE csp.chase_probability_ps > 0.05
  AND csp.training_period_s = 120.0;
```

### All visual-angle-loom recordings with l/v = 90 ms
```sql
SELECT d.dataset_id, ca.l_over_v_ms, ca.initial_distance_mm
FROM datasets d
JOIN provenance p ON d.dataset_id = p.dataset_id
JOIN chaser_agents ca ON p.protocol_hash = ca.protocol_hash
WHERE ca.loom_mode = 2
  AND ca.l_over_v_ms = 90.0;
```

### Dream query: per-chase escape metrics filtered by protocol params
```sql
SELECT
    p.fish_id,
    p.dpf_at_acquisition,
    cs.escape_latency_s,
    cs.max_speed_mm_s,
    ca.l_over_v_ms
FROM datasets d
JOIN provenance p      ON d.dataset_id = p.dataset_id
JOIN chaser_step_params csp ON p.protocol_hash = csp.protocol_hash
JOIN chaser_agents ca  ON csp.protocol_hash = ca.protocol_hash
                      AND csp.step_index = ca.step_index
JOIN chase_sequences cs ON d.dataset_id = cs.dataset_id
                       AND csp.step_index = cs.step_index
WHERE csp.chase_probability_ps > 0.05
  AND csp.training_period_s = 120.0
  AND ca.loom_mode = 2
  AND cs.escape_triggered = 1
ORDER BY p.fish_id, cs.chase_index;
```

## Implementation plan

### Phase 0 -- Fix protocol data pipeline (DO THIS FIRST)

The import function looks for `protocol_snapshot/protocol_json` but the H5
files store it under `protocol_snapshot/protocol_definition_json`. Until this
is fixed, no protocol data reaches the zarrs or registry.

- [x] Fix key name in `import_stimulus_to_zarr.py` (~line 752): read
      `protocol_definition_json` instead of (or in addition to) `protocol_json`.
- [x] Re-import stimulus data for existing recordings to populate the zarr attr.
      Used `scripts/backfill_protocol_json.py` to patch 55 stimulus runs (4
      already had the attr, 5 had missing H5 files).
- [x] Run registry rescan to populate `provenance.protocol_name` and
      `provenance.protocol_hash`.
- [x] Verify non-zero `protocol_name`/`protocol_hash` in registry after rescan.
      Result: 52/114 rows populated (all analysis zarrs with stimulus data).
      2 distinct protocols: `Feeding` (26), `DefaultScreen` (26).
      62 missing are training zarrs (60) + 1 analysis w/ missing H5 + 1 unclassified.

### Phase 1 -- Protocol skeleton tables
- [ ] Add `protocols` and `protocol_steps` table creation to `db.py` schema migration.
- [ ] Extend `_extract_protocol` (or add a sibling function) to parse the steps
      array from `protocol_json` and insert rows.
- [ ] Deduplicate on `protocol_hash`: only insert if the hash is new.
- [ ] Populate during `register_from_root` / scan.
- [ ] Backfill: one-time scan of existing datasets to populate the new tables.
- [ ] Tests: round-trip a sample protocol JSON through extraction and verify rows.

**Stop here.** Use `protocols.raw_json` + `json_extract()` for any ad-hoc
parameter queries. Only proceed to Phase 2+ once a real analysis workflow
demonstrates repeated need for denormalized parameter columns.

### Phase 2 -- Chaser parameter tables (DEFERRED)
- [ ] Add `chaser_step_params` and `chaser_agents` table creation.
- [ ] Write shredding logic that maps `ProtocolChaserParams` JSON fields to columns.
- [ ] Handle nested `chasers[]` array -> `chaser_agents` rows.
- [ ] Tests: verify extraction for single-chaser and multi-chaser protocols.

### Phase 3 -- Other stimulus type tables (DEFERRED)
- [ ] `grating_step_params`
- [ ] `looming_dot_step_params`
- [ ] `concentric_grating_step_params`
- [ ] `coherent_dots_step_params`
- [ ] `moving_dots_step_params`
- [ ] `spotlight_step_params`
- [ ] `scrolling_grid_step_params`
- [ ] `independent_motion_grid_step_params`
- [ ] `solid_color_step_params`
- [ ] `static_image_step_params`
- [ ] Tests for each.

### Phase 4 -- Query CLI integration (DEFERRED)
- [ ] Add query flags to `query.py`: `--stimulus-mode`, `--chase-probability-min`,
      `--training-duration`, `--loom-mode`, `--l-over-v`, etc.
- [ ] Support joining across protocol param tables in the query builder.
- [ ] Add `--group-by protocol` aggregation option.

### Phase 5 -- Analysis result tables (future, blocked on analysis pipeline)
- [ ] Define `chase_sequences` table schema.
- [ ] Build per-chase extraction from zarr event + chaser_states data.
- [ ] Populate during analysis pipeline or as a standalone backfill tool.
- [ ] Add query flags for escape filtering, speed thresholds, etc.

## Design notes

- **Protocol deduplication**: Many datasets share the same protocol. The
  `protocol_hash` (SHA256 of canonical JSON) is the natural deduplication key.
  Protocol parameter tables have one row per unique protocol+step, not per
  dataset.
- **Authoritative units**: Store mm, degrees, and seconds. Skip pixel-derived
  computational values that depend on calibration. Exception: fields that only
  exist in pixels in the C++ struct (e.g. `start_radius_px` for looming dot,
  `cave_trigger_radius_px` for cave dweller).
- **Escape hatch**: `protocols.raw_json` and `protocol_steps.parameters_json`
  store the full JSON blobs. Any parameter not yet denormalized can still be
  reached via `json_extract()`.
- **Schema migration**: Use the existing `_SCHEMA_VERSION` / `_ensure_schema`
  pattern in `db.py` to add tables non-destructively.
- **Parameter source**: All values come from `protocol_definition_json` in the
  source H5 (key name: `protocol_snapshot/protocol_definition_json`), imported
  to the zarr as `protocol_json` on stimulus run attrs. Field names in the
  tables should match the Citrus JSON keys where possible, abbreviated for
  readability.
- **H5 key mismatch**: The import function currently looks for
  `protocol_snapshot/protocol_json` but the actual key is
  `protocol_snapshot/protocol_definition_json`. This is the root cause of all
  missing protocol data.

## Related docs
- `src/fisheye/docs/citrus_data_structure_documentation.md` -- full parameter
  reference for all stimulus types
- `src/fisheye/docs/zarr_structure.md` -- where protocol_json lives in zarr
- `docs/registry_query_todo.md` -- existing query CLI status
- `docs/zarr_run_completion_contract.md` -- active provenance/completion contract
- `src/fisheye/registry/db.py` -- registry schema and `_extract_protocol`
- `src/fisheye/registry/query.py` -- query CLI
- `src/fisheye/analysis/import_stimulus_to_zarr.py` -- h5 -> zarr import
