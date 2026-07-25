# Zarr Spec vs Runtime Drift TODO

Purpose: track concrete mismatches between the "authoritative" Zarr spec and what current code paths actually write, then drive them to closure.

Date anchored: 2026-02-09.

Cross-cutting string/text encoding policy and migration is tracked separately in:
`docs/zarr_string_encoding_todo.md`.

## Why this exists

`src/fisheye/docs/zarr_structure.md` declares an authoritative layout, but runtime entry points still create/update archives through different code paths (`import_recording_analysis`, `detect_yolo`, metadata-only updates). This creates drift risk for operators and downstream consumers.

## Confirmed Drift Items

1. Analysis archive bootstrap does not populate full root schema.
- Spec expects root attrs like `schema_version`, `zarr_format`, `created_at`, etc., and a standard set of immediate child groups.
- `import_recording_analysis` currently ensures a minimal archive and sets a bounded attrs subset.
- Sources:
  - `src/fisheye/docs/zarr_structure.md`
  - `src/fisheye/utils/import_recording_analysis.py`

2. Analysis archive purpose can be overwritten to `production` during detect metadata writes.
- `import_recording_analysis` enforces `zarr_purpose=analysis`.
- `run_detect_with_registry_model` passes `--write-raw-video-metadata` into detect.
- detect metadata writer uses `import_purpose="production"` and can set `zarr_purpose=production` if overwrite is enabled.
- Sources:
  - `src/fisheye/utils/import_recording_analysis.py`
  - `src/fisheye/utils/run_detect_with_registry_model.py`
  - `src/fisheye/utils/import_video_metadata.py`

3. No explicit profile concept in spec for "metadata-only analysis archive".
- Runtime intentionally supports split archives (`training` with imported frames vs analysis/prod with metadata-only `raw_video`).
- Spec does not explicitly define required-vs-optional groups/attrs by archive profile.
- Sources:
  - `docs/zarr_split_policy.md`
  - `src/fisheye/docs/zarr_structure.md`

4. Archive creation paths are not consolidated on one schema helper.
- `create_palette_zarr` builds a fuller schema shape.
- `import_recording_analysis` and `detect_yolo` create/update archives through separate logic.
- This makes schema drift likely whenever one path evolves.
- Sources:
  - `src/fisheye/shared/zarr/schema.py`
  - `src/fisheye/utils/import_recording_analysis.py`
  - `src/fisheye/detection/detect_yolo.py`

## TODO Plan

## Phase 1: Define profiles in spec

- [ ] Add explicit archive profiles to `src/fisheye/docs/zarr_structure.md`:
  - `training_import` (with `raw_video/images_*`)
  - `analysis_metadata_only` (no `raw_video/images_*`, metadata-only raw_video)
  - optional `detect_only_legacy` compatibility profile
- [ ] For each profile, mark:
  - required root attrs
  - required groups
  - allowed missing groups
  - frame-universe semantics for `total_frames`/`n_frames`

## Phase 2: Unify creation/bootstrap behavior

- [ ] Add shared helper for archive bootstrap with profile input.
- [x] Retire the duplicate `create_analysis_zarr` entry point; use
  `import_recording_analysis` as the canonical analysis bootstrap (2026-07-24).
- [ ] Migrate detect-created archives to shared bootstrap helper (or explicit compatibility mode).

## Phase 3: Fix purpose semantics

- [ ] Prevent detect metadata writes from downgrading analysis archives to `zarr_purpose=production`.
- [ ] In registry-driven detect wrapper, use analysis-safe import purpose when output is analysis archive.
- [ ] Add explicit tests for purpose stability across:
  - create analysis archive
  - detect append with metadata writes
  - overwrite vs non-overwrite modes

## Phase 4: Validation and guardrails

- [ ] Add `validate_zarr_profile` diagnostic:
  - detects profile
  - validates required attrs/groups
  - reports drift with actionable messages
- [ ] Integrate validation in CI for representative fixtures.

## Phase 5: Doc alignment and downstream handoff

- [ ] Update all docs that imply a single canonical root shape:
  - `docs/recording_analysis_pipeline_contract.md`
  - `docs/zarr_split_policy.md`
  - any operator runbooks referencing required groups
- [ ] Publish a short migration note for external consumers (`crimson`, etc.) describing profile-aware expectations.

## Exit Criteria

- [ ] Spec explicitly defines archive profiles and required fields.
- [ ] Analysis archive creation and detect append paths produce profile-consistent outputs.
- [ ] `zarr_purpose` remains stable and intentional across workflow steps.
- [ ] Validation tooling can detect drift automatically before operator-facing failures.
