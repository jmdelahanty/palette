# Movement Analysis: Split Online / Offline Runs

## Overview

We now compute two flavours of movement metrics:

- **Online**: detection/keypoint driven tracks (existing `movement_runs` pipeline).
- **Offline**: synthetic tracks derived from `analysis/chaser_fish_metrics` (via the new bundle).

Both products currently land in the flat `analysis/movement_runs/` namespace differentiated only by naming conventions and per-run metadata. The next step is to formalise two subtrees so consumers can browse predictable slots and the CLI can surface truly separate `latest` pointers.

## Proposed Zarr Layout

```
analysis/
  movement_runs/
    online/
      <run>/ ... existing track arrays, attrs, provenance
      latest -> <run>
    offline/
      <run>/ ... same schema, but sourced from offline metrics
      latest -> <run>
```

Key points:

- `analysis/movement_runs/online` and `.../offline` are required groups.
- Each subgroup manages its own `latest` attribute (no more `latest_online` vs `latest_offline` juggling).
- Individual run groups keep exactly the same internal layout and provenance attrs we write today (inputs, summary tables, etc.).
- The top-level `analysis/movement_runs` parent no longer stores run groups directly; it only contains the two children and optional compatibility pointers.

## Implementation Plan

1. **Migration Helper (optional now, future proof later)**
   - Add a utility to detect legacy flat runs and move/rename them into the new subgroups (can be invoked manually during upgrade).

2. **movement_analysis.py**
   - Update run creation to write under `online/<run>` or `offline/<run>`.
   - Ensure provenance attrs include the subgroup path for clarity.
   - Accept optional `--output-group` or similar if we ever need more than two categories (keep extensibility in mind).

3. **plot_movement.py**
   - Update discovery logic to enumerate `analysis/movement_runs/online` and `.../offline`.
   - Provide `--online-only` / `--offline-only` to mirror existing UX.
   - When the user targets a specific run, allow shorthand like `online/<run>` while preserving backwards compatibility (e.g., translate old names to new paths when possible).

4. **Other Consumers**
   - Audit any scripts/notebooks referencing `analysis/movement_runs/<run>` directly (search for `movement_runs/`). Patch them to use the new helper.
   - Update docs (README, CLI help, pipeline diagrams) to showcase the new layout.

5. **Testing / Validation**
   - Unit tests (or integration scripts) that execute `movement_analysis` with both online and offline enabled, then assert runs land in the correct subgroup and `latest` pointers update correctly.
   - Run `plot_movement` on a freshly generated archive to confirm both plots appear by default.
   - Verify provenance metadata still contains all expected keys (detection path, metrics run, etc.).

## Open Questions / Follow Ups

- Do we want a compatibility shim that exposes the old flat runs via symbolic references for third-party notebooks? (Probably not urgent if we coordinate the cutover.)
- Should we introduce a common loader (similar to `load_chaser_metrics`) for movement runs so visualisers can pull both online/offline data through one API?
- How should we expose combined summaries (e.g., CLI command that prints both online/offline metrics side-by-side)?

## Next Steps

1. Update `movement_analysis.py` to create runs inside `online/` and `offline/` subgroups.
2. Patch `plot_movement.py` (and any other tooling) to list the new structure.
3. Refresh documentation and sample commands.
4. Once the new layout is stable, decide whether to write a migration script for existing archives or continue supporting the legacy structure for a short window.

