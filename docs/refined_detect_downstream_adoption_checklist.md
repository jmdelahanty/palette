# Refined Detect Downstream Adoption Checklist

<!-- design-meta
status: active
last_updated: 2026-04-15
-->

## Goal

Track what is still left after the sparse-first refined-detect contract was
adopted in Palette.

The stable target contract is:

- `refined_detect_runs/<run>/instances` is the canonical curated detect surface
- `refined_detect_runs/<run>/source_detections` is the candidate-audit surface
- `detect_review_status.resolved_group = "refined"` is the normal current-run
  review state
- legacy `manual` / `interpolated` / `filtered` subgroups are
  compatibility-only

## Current State

### Palette runtime and docs

- [x] Preferred detect/crop storage was removed from the active workflow.
- [x] Active refined-detect writes use sparse `instances/` and
      `source_detections/`.
- [x] Palette runtime readers now prefer `instances/` for current refined runs,
      with subgroup fallback only for historical archives.
- [x] Palette operator help, contracts, and local Crimson-facing docs now teach
      the sparse-first refined-detect model.
- [x] Palette registry/query surfaces now prefer
      `refined_detect_review_current` and treat `detect_quality_current` as a
      compatibility alias.
- [x] Interpolation-era refine controls were deprecated from the active detect
      workflow.

### Archive normalization

- [x] Existing training and analysis archives were migrated to the sparse
      refined-detect structure.
- [x] Crop/runtime provenance now records
      `refined_detect_runs/<run>/instances` as the canonical current refined
      source path.
- [x] A latest-run audit on 2026-04-15 found that all current `latest` refined
      runs are already sparse.

Audit result:

- `105` archives scanned
- `105` skipped because the current `latest` refined run was already sparse
- `0` latest-run legacy conflicts
- `0` latest runs left to normalize

### Migration tooling

- [x] `migrate_refined_detect_sparse` now materializes successor sparse runs
      instead of rewriting source runs in place.
- [x] Legacy subgroup presence fails closed by default and requires
      `--ignore-legacy-groups`.
- [x] Promotion to `refined_detect_runs.attrs["latest"]` is allowed by default
      only when migrating the current parent `latest` refined run.
- [x] Non-latest migration requires either `--no-promote-latest` or explicit
      `--force-promote-nonlatest`.

## What Is Left

### 1. Cross-repo contracts

These are no longer Palette-runtime blockers, but they are still useful to keep
aligned with the current contract.

- [ ] Update `~/gitrepos/contracts/palette-crimson/` detect docs so they teach:
  - `instances/` as canonical refined reads
  - `source_detections/` as audit-only
  - `resolved_group = "refined"` as the normal current review state
  - sparse refined write expectations instead of subgroup-era manual writes

### 2. Crimson runtime and write path

Palette-side docs and local contracts are aligned, but Crimson runtime/write
adoption still needs to be confirmed or completed in `~/gitrepos/crimson`.

- [ ] Confirm Crimson detect read-path selection uses `instances/` as the
      primary refined surface.
- [ ] Keep subgroup fallback only for historical archives.
- [ ] Decide whether Crimson refined-detect writing remains read-only for now
      or lands a true sparse refined write path.

### 3. Optional historical hygiene

This is not required for the active canonical surface. It is only historical
cleanup.

- [ ] Optionally inventory non-latest refined runs that still use legacy
      subgroup-only storage.
- [ ] If desired, migrate those older runs to successor sparse runs with
      `--no-promote-latest` so archive history is cleaner without changing the
      canonical active run.

## Explicitly Done

The following older checklist items are complete and should not be treated as
active work anymore:

- Palette runtime consumer alignment for sparse refined detect
- Palette help text/docstring cleanup for `instances/` and
  `source_detections/`
- Palette-owned Crimson-facing detect contract cleanup
- Detect training/profile/runtime adoption of sparse refined surfaces
- Detect review status normalization around `resolved_group = "refined"`

## Recommended Next Steps

1. Leave latest-run batch migration alone; there is nothing left to normalize
   in the active canonical surface.
2. If cross-repo alignment matters for the next pass, update the contracts repo
   and Crimson docs/runtime next.
3. If historical cleanup matters, add a read-only report for non-latest legacy
   refined runs and only migrate those with `--no-promote-latest`.
