# Provenance Backfill TODO

Purpose: track investigation and implementation of missing provenance fields
in the registry (`protocol_hash`, `protocol_name`, `snapshot_status`).

## Current State (2026-02-27)

Registry has 114 provenance rows. Audit results:

| Field | Present | Missing | Notes |
|-------|---------|---------|-------|
| `protocol_name` | 0/114 | 114 | `analysis/stimulus_runs/<latest>` groups exist but contain no `protocol_json` attr |
| `protocol_hash` | 0/114 | 114 | Derived from `protocol_json`; blocked by above |
| `snapshot_status` | 0/114 | 114 | `subject_metadata` snapshot exists but `status`/`missing` fields are `None` |
| `dpf_at_acquisition` | 105/114 | 9 | Missing 9 are merged training datasets + 1 recording (expected) |
| `dish_id` / `cross_id` / etc. | 105/114 | 9 | Same 9 rows as above |

### The 9 rows missing most fields (expected)
These are derived/merged training datasets that don't carry subject metadata:
- 3 detection training datasets (`detect_cedar_shadow_*`)
- 2 pose training datasets (`pose_cedar_shadow_*`)
- 3 eye mask training datasets (`eye_mask_cedar_shadow_*`)
- 1 raw recording (`2026-01-28T22-50-39Z_arena_2_Feeding`)

## Investigation Items

### Protocol metadata
- [ ] Determine where protocol definitions live upstream (Zebrobot? stimulus config files?).
- [ ] Determine whether stimulus import should write `protocol_json` into
      `analysis/stimulus_runs/<run>/attrs` at import time.
- [ ] If protocol data exists outside zarrs, design an injection/backfill path
      (e.g. a script that reads protocol configs and patches zarr attrs).
- [ ] Decide: should `protocol_name` be extracted from `arena_config_json` or
      from the recording folder name (e.g. `Feeding`, `DefaultScreen`) as a
      fallback?

### Snapshot status
- [ ] Decide: compute `snapshot_status` at registration time from field presence
      rather than relying on upstream to populate it.
  - Proposed logic: `complete` if `dish_id`, `cross_id`, `dpf_at_acquisition`,
    `line_strain`, and `genotype` are all non-null; `partial` if some present;
    `missing` if none.
  - `snapshot_missing_json`: list of null field names for `partial` status.
- [ ] Alternatively: fix the upstream snapshot writer to populate `status`/`missing`.
- [ ] Determine which approach is more maintainable long-term.

### Merged training datasets (the 9 missing rows)
- [ ] Decide if merged training datasets should inherit provenance from their
      source recordings (multi-source lineage), or if missing provenance is
      acceptable for derived artifacts.
- [ ] If inheriting: design the lineage lookup (manifest -> source dataset IDs
      -> provenance aggregation).

## Implementation (after investigation)

- [ ] Implement chosen `snapshot_status` approach.
- [ ] Implement chosen `protocol_name`/`protocol_hash` approach.
- [ ] Run rescan or backfill against live registry.
- [ ] Verify coverage metrics post-backfill.
- [ ] Update `docs/detection_registry_curation_todo.md` backfill checkbox.
