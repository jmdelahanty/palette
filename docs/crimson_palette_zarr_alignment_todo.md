# Crimson <> Palette Zarr Alignment TODO (Agent Handoff)

As of 2026-02-09, this note captures the concrete differences between:

- Palette's current archive/registry design (training + analysis split)
- What `crimson` currently discovers/loads, and what its `zarr_structure.md` documents

Goal: give another agent a precise implementation map to update `crimson` without guesswork.

## Scope

- In scope:
  - Archive discovery/open behavior in `crimson`
  - Compatibility of `crimson` loader with current Palette analysis archives
  - Doc/spec sync between repos
- Out of scope:
  - Palette DB schema changes
  - Reworking detection/crop/keypoint algorithms
  - Multi-camera 3D schema redesign (tracked separately)

## Primary References

Palette repo:
- `src/fisheye/docs/zarr_structure.md`
- `docs/zarr_split_policy.md`
- `docs/analysis_zarr_creation_contract.md`
- `docs/crimson_detect_bbox_read_contract.md`
- `docs/crimson_refined_detect_manual_contract.md`
- `docs/crimson_detect_review_acceptance_contract.md`
- `src/fisheye/analysis/create_analysis_zarr.py`
- `src/fisheye/utils/import_video_metadata.py`
- `src/fisheye/detection/detect_yolo.py`

Crimson repo:
- `/home/delahantyj@hhmi.org/gitrepos/crimson/zarr_structure.md`
- `/home/delahantyj@hhmi.org/gitrepos/crimson/src/zarr_loader.cpp`
- `/home/delahantyj@hhmi.org/gitrepos/crimson/src/zarr_loader.h`
- `/home/delahantyj@hhmi.org/gitrepos/crimson/src/red.cpp`

## Current Palette Reality (not just spec)

1. Training and analysis archives are intentionally split.
- Training archives may contain imported frame arrays.
- Analysis archives can be metadata-only for `raw_video` and carry inference/refinement runs.
- Source: `docs/zarr_split_policy.md`.

2. `create_analysis_zarr` creates a minimal archive first, then enriches.
- Initial attrs include `zarr_purpose=analysis` and archive timestamps.
- Source-video attrs are added via metadata import.
- Stimulus import and registry scan are optional steps.
- Source: `src/fisheye/analysis/create_analysis_zarr.py`.

3. Analysis archive naming convention is `<recording>_analysis.zarr`.
- Detect then appends `detect_runs/<run>` into that archive.
- Source: `src/fisheye/analysis/create_analysis_zarr.py`, `src/fisheye/utils/run_detect_with_registry_model.py`.

4. Palette detect data uses normalized center-width-height boxes.
- `bbox_norm_coords` are `[cx, cy, w, h]` normalized to inference frame size.
- Source: `src/fisheye/detection/detect_yolo.py`.

5. Registry model is "multiple datasets per recording".
- A recording can have both `training` and `analysis` datasets simultaneously.
- Semantics are in registry columns (`zarr_use`, `zarr_origin`, etc.), not filename-only.

## What Crimson Currently Assumes

1. Discovery is filename-filtered.
- `findZarrDetectionFile()` currently picks only `.zarr/.zr3` directories containing `"detection"` or `"chaser"` in the name.
- It will miss `<recording>_analysis.zarr` by default.
- Source: `crimson/src/zarr_loader.cpp` (`findZarrDetectionFile`, `loadZarrDetectionFromDirectory`).

2. Loader requires `detect_runs` layout for load success.
- If `detect_runs` is absent, load fails with "layout not found".
- Source: `crimson/src/zarr_loader.cpp` (`loadZarrFile`).

3. Crimson `zarr_structure.md` is older than Palette's current authoritative spec.
- Diff exists vs `src/fisheye/docs/zarr_structure.md` (newer fields/stages/attrs in Palette doc).

4. Loader does support metadata-only `raw_video` attrs and external video playback path.
- This is good and should be preserved.

5. Refined detect reason labels now include a TensorStore-safe encoding.
- Preferred array: `reason_bytes` (`uint8[N,width]`, null-terminated UTF-8).
- Fallbacks: `reason` (string) then `detection_source` (0=clean, 1=interpolated).
- Read contract: `docs/crimson_detect_bbox_read_contract.md`.
- Write contract: `docs/crimson_refined_detect_manual_contract.md`.

6. Refined keypoint reason labels follow the same compatibility pattern.
- Preferred array: `reason_bytes` (`uint8[N,width]`, null-terminated UTF-8).
- Fallbacks: `reason` (string) then `detection_source` (0=clean, 1=interpolated).
- Authoritative schema docs:
  - `src/fisheye/docs/zarr_structure.md`
  - `src/fisheye/docs/provenance_workflow.md`
  - `docs/keypoint_review_policy.md`

7. Analysis stimulus provenance now includes optional rendered-video path.
- Analysis stimulus runs may include:
  - `analysis/stimulus_runs/<run>.attrs["source_h5"]`
  - `analysis/stimulus_runs/<run>.attrs["source_stimulus_video_path"]` (when `<source_h5>.mp4` exists)
- Training Zarrs are not expected to carry this stimulus-run attr.

## Gap Matrix

| Area | Palette current behavior | Crimson current behavior | Impact | Required change |
|---|---|---|---|---|
| Archive discovery | Uses `*_analysis.zarr` and `*_training.zarr` naming | Finds only names containing `detection`/`chaser` | Analysis archives not auto-loaded | Replace filename substring heuristic with structural probe |
| Multi-archive recording | Recording can have both training + analysis datasets | Single auto-picked archive from dir scan | Wrong archive may be chosen | Add deterministic selection policy or explicit `--zarr` input |
| Pre-detect analysis archives | Valid archive may exist before `detect_runs` exists | Load path expects `detect_runs` | Early archive cannot open in Crimson loader mode | Graceful "video/stimulus-only" mode when `detect_runs` missing |
| Doc/spec drift | Palette spec evolved | Crimson spec copy is stale | Implementers can code against wrong schema | Sync `crimson/zarr_structure.md` from Palette authoritative doc |
| Registry-aware selection | Dataset intent stored as `zarr_use` etc. | No registry integration | Cannot choose best archive per recording intent | Optional phase: registry-backed archive choice |
| Manual refined detect edits | Palette expects `manual_review_latest` + `detect_review_status` + manual subgroup arrays | No explicit Crimson write contract yet | Manual edits may not become active source for crop/registry | Implement manual-write contract in `docs/crimson_refined_detect_manual_contract.md` |
| Review acceptance after inspection | Palette status consumers expect structured `detect_review_status` + latest pointer | Acceptance can be ad hoc/manual | Inconsistent review state, poor auditability | Implement acceptance flow per `docs/crimson_detect_review_acceptance_contract.md` |
| Stimulus rendered-video provenance | Analysis stimulus runs can expose `source_stimulus_video_path` | Loader/spec may only read `source_h5` or ignore stimulus attrs | Missed provenance link to rendered stimulus MP4 | Treat `source_stimulus_video_path` as optional analysis-only hint; do not expect it on training archives |

## Implementation Plan For Crimson Agent

### Phase 1: Discovery + Open Contract

1. Update detection archive discovery.
- File: `crimson/src/zarr_loader.cpp`
- Replace name-based check with:
  - candidate is `.zarr` or `.zr3`
  - candidate contains `detect_runs` group metadata (`detect_runs/zarr.json`) or is explicitly passed.

2. Add explicit CLI/UI archive path override.
- File: `crimson/src/red.cpp`
- Ensure user can pick exact archive path without auto-discovery heuristics.

3. If scanning a recording directory with multiple candidates, rank deterministically.
- Preferred order:
  - archive with `zarr_purpose=analysis` and `detect_runs` present
  - else archive with `detect_runs` present
  - else prompt user.

### Phase 2: Loader Tolerance

1. Keep hard requirement for `detect_runs` only when detection overlays are requested.
- If no `detect_runs`, still allow metadata/stimulus inspection mode when possible.
- File: `crimson/src/zarr_loader.cpp` (`loadZarrFile` flow).

2. Preserve current support for metadata-only `raw_video`.
- Do not require `raw_video/images_full` arrays.
- Continue using `raw_video` attrs (`source_path`, fps, width/height) and external decode.

### Phase 3: Doc Sync

1. Replace/refresh:
- `crimson/zarr_structure.md` from Palette `src/fisheye/docs/zarr_structure.md`.
- Add a short "compatibility notes" section for optional/missing groups during staged pipelines.

### Phase 4: Optional Registry-Aware Selection

1. Add optional resolver that can use registry metadata (`zarr_use=analysis`).
- This can be a separate utility first; not required for initial compatibility fix.

### Phase 5: Manual Refined-Detect Writes

1. Implement Crimson-side writer for manual refined detections.
- Use contract in `docs/crimson_refined_detect_manual_contract.md`.
- Write/overwrite `refined_detect_runs/<latest>/<manual_group>` arrays + attrs.

2. Update run-level pointers/status after manual writes.
- Set `manual_review_latest` on refined run.
- Set `detect_review_status` on refined run.
- Set parent `detect_review_status_latest`.

3. Keep manual edits isolated.
- Never overwrite `detect_runs/<run>` raw outputs.
- Manual/retune edits must live only under `refined_detect_runs/<run>/<manual_group>`.

## Validation Checklist

Use one real recording directory containing both training + analysis archives:

1. Discovery
- Crimson can locate `<recording>_analysis.zarr` without filename containing `detection`.

2. Analysis archive with detect run
- Loads `detect_runs/<latest>` and overlays detections.
- Reads metadata-only `raw_video` attrs and decodes external video path.

3. Analysis archive without detect run (pre-inference)
- Opens archive in non-crash mode.
- Shows explicit status: detect data unavailable.

4. Training archive
- Still loads as before; no regressions in existing behavior.

5. Spec/docs
- `crimson/zarr_structure.md` matches current Palette authoritative spec snapshot.
- Include analysis-only note for `analysis/stimulus_runs/<run>.attrs["source_stimulus_video_path"]`.
- Include explicit note that training Zarrs are not expected to have this attr.

6. Manual edit activation
- After Crimson manual edit write, Palette resolution picks manual subgroup as active.
- Crop with preferred/auto resolves to manual source (not interpolated/filtered).

## Notes For Safe Rollout

- This is an application-side compatibility update only; no DB migration needed.
- Keep behavior operator-first:
  - explicit error messages on ambiguous archive selection
  - no silent fallback to wrong archive when both training and analysis exist
  - do not mutate archives during discovery/load.
