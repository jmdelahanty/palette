<!-- ARCHIVED 2026-07-04: dated point-in-time snapshot / spent work ticket, retained for history only. -->

# Documentation Staleness Audit: 2026-05-18

## Scope

This pass reviewed the active Palette docs and the external contracts repo at
`~/gitrepos/contracts`, focusing on the current Palette/Crimson boundary for:

- refined-detect `instances/` reading;
- clipped-analysis finalized collections and parent-frame mapping;
- bbox array chunking;
- analysis-to-training promotion and web review status.

## Palette Docs

Active Palette docs are mostly aligned with the current implementation.

Corrections applied in this pass:

- `docs/crimson_detect_bbox_read_contract.md` now states that the bbox leaf
  contract applies to selected clipped `instances/` groups after a resolver has
  chosen the clip-local run.
- `docs/clipped_recording_consumer_mapping_contract.md` now reflects the local
  Crimson prototype status: finalized collection resolution and read-only
  clipped detection loading are working, but clipped edit/write routing remains
  intentionally disabled.
- `docs/cluster_pipeline_migration_checklist.md` no longer describes
  clip-local model import/finalize as design-only; the sleepyfish all-clips
  detect/refine/finalize smoke passed.

Still valid but important:

- `recording_frame_index.parquet` is a frame-map artifact, not review state.
- Clipped analysis shells are not traditional top-level analysis Zarrs.
- Readers must consume finalized collection metadata and concrete clip-local
  run paths instead of scanning `clips/` by sorted directory order.
- Logical bbox arrays are `[N, 4]`; readers must not assume physical chunks
  keep each row contiguous in raw memory.

## Contracts Repo Findings

The contracts repo is clean, but several Palette/Crimson contracts are stale
relative to Palette's current docs and implementation:

- `palette-crimson/detect_bbox_read.md` still describes refined detection as
  `refined_detect_runs/<run>/<group>` with `manual/interpolated/filtered`
  subgroup preference. It should be updated to prefer
  `refined_detect_runs/<run>/instances`, `bbox_img_xyxy`, and
  `source_kind_codes`, with legacy subgroup fallback only.
- `palette-crimson/refined_detect_manual.md` is now a historical/manual
  subgroup write contract. It should either be marked legacy or superseded by a
  current refined-detect `instances` write/promotion contract.
- `palette-crimson/zarr_alignment.md` still says manual refined detection
  writes should target manual subgroups and does not describe clipped finalized
  collections, `recording_frame_index.parquet`, or bbox chunking rules.

Severity: high if future Crimson agents rely on the contracts repo as their
source of truth. The active Palette docs are ahead of those contracts.

## Recommended Contracts Update

Use the contracts repo workflow from `~/gitrepos/contracts/AGENTS.md`:

1. Create a branch from `main`.
2. Update `palette-crimson/detect_bbox_read.md` to mirror
   `docs/crimson_detect_bbox_read_contract.md`.
3. Add or update a clipped-analysis read contract covering finalized
   collections, `recording_frame_index.parquet`, Arrow/Parquet consumption,
   parent-frame to clip-local mapping, and read-only clipped edit policy.
4. Update `palette-crimson/zarr_alignment.md` to point at the current
   `instances` and clipped resolver contracts.
5. Mark `palette-crimson/refined_detect_manual.md` as legacy unless a current
   `instances` write contract replaces it in the same pass.

Because `~/gitrepos/contracts` is outside this workspace's writable roots, that
contract update needs an explicit outside-sandbox edit/commit workflow.
