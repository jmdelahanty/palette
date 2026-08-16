# Provenance Contract (v1)
<!-- contract-meta
version: 1
status: draft
implementation: partial
last_verified: 2026-05-20
-->

Purpose: define the active, queryable stage-wide provenance contract for
refinement stages so detect/keypoints/eye-masks expose consistent metadata,
especially git commit/hash and contract identity, through shared helpers.

## Scope

In scope:

- `refined_detect_runs/<run>`
- `refined_keypoints_runs/<run>`
- `refined_eye_masks_runs/<run>`

Adoption note:

- this contract defines a stage-wide contract family name so the same object can be
  adopted in non-refinement stages later without renaming.

Out of scope:

- raw detect/keypoint/eye-mask inference runs (tracked separately)
- training/export contracts
- `refine_online_detect` migration (deferred while offline dataset provenance
  work is prioritized)

## Related Docs

- `docs/provenance_checks.md`
  - operational diagnostics and CLI usage.
- `docs/archive/provenance_todo.md`
  - backlog and migration items.
- `docs/pipeline_metadata_boundaries.md`
  - repository-level ownership boundaries for provenance, contracts, code, and workflow docs.
- `src/fisheye/docs/provenance_workflow.md`
  - end-to-end workflow sequencing and lineage expectations.
- `docs/recording_analysis_pipeline_contract.md`
  - stage orchestration contract that consumes provenance.
- `docs/archive/eye_mask_row_mapping_contract.md`
  - eye-mask lineage array contract referenced by eye-mask refinement inputs.

## Canonical Run-Level Contract

Each refinement run must contain `attrs["provenance"]` with this shape:

```json
{
  "contract": {
    "name": "palette_stage_provenance",
    "version": 1
  },
  "stage": "refine_detect|refine_keypoints|refine_eye_masks",
  "command": "<argv string>",
  "created_at_utc": "<ISO-8601 UTC timestamp>",
  "version": "<short git hash or pipeline version>",
  "git": {
    "commit": "<full hash>",
    "short": "<short hash>",
    "branch": "<branch>",
    "is_dirty": true,
    "remote": "<remote url>"
  },
  "environment": {
    "...": "runtime/package summary"
  },
  "platform": {
    "...": "host/python/system summary"
  },
  "scheduler": {
    "...": "optional scheduler/dask metadata"
  },
  "parameters": {
    "...": "stage parameters"
  },
  "inputs": {
    "...": "source run lineage"
  },
  "artifacts": {
    "...": "model/checkpoint/source artifacts"
  }
}
```

Required keys: `contract`, `stage`, `created_at_utc`, `parameters`, `inputs`.

Optional keys: `command`, `version`, `git`, `environment`, `platform`,
`scheduler`, `artifacts`.

## Stage-Specific `inputs` Contract

- `refine_detect`: `detect_run`, `quality_run` (or `N/A`), `frame_source`, `source_video_path`
- `refine_keypoints`: `keypoints_run`, `source_crop_run`, `frame_source`, `source_video_path`
- `refine_eye_masks`: `eye_masks_run`, `keypoints_run`, `crop_run`

## Top-Level Convenience Attrs (Compatibility)

Refinement runs may also expose top-level attrs for easy shell inspection.
Canonical provenance remains `attrs["provenance"]`.

Write policy for new/updated runs:

- always write canonical `attrs["provenance"]`
- also write `git_commit` and `git_branch` at top level
- do not require top-level `git_commit_hash` (legacy read-only fallback)

Read compatibility order for git commit:

1. `provenance.git.commit`
2. `provenance.git.commit_hash` (legacy payloads)
3. top-level `git_commit`
4. top-level `git_commit_hash`

## Unified Helper API

Add shared helpers in `src/fisheye/shared/stage_provenance.py`.

Reader helpers:

- `get_stage_provenance(attrs: Mapping[str, Any]) -> Dict[str, Any]`
  - returns normalized provenance dict (never mutates attrs)
- `get_stage_git(attrs: Mapping[str, Any]) -> Dict[str, Any]`
  - returns normalized git payload with `commit`, `short`, `branch`, `is_dirty`, `remote`
- `get_stage_contract(attrs: Mapping[str, Any]) -> Dict[str, Any]`
  - returns `{name, version}` with compatibility defaults

Writer helper:

- `build_stage_provenance(...) -> Dict[str, Any]`
  - produces canonical payload including `contract`
- `write_stage_provenance(run_group, payload, include_top_level_git=True) -> None`
  - writes canonical payload and optional top-level git convenience attrs

## Diagnostics Contract

`check_provenance_capture` should treat refinement stages as first-class:

- `refined_detect`
- `refined_keypoints`
- `refined_eye_masks`

In strict mode, diagnostics should additionally require:

- `provenance.contract.name == "palette_stage_provenance"`
- `provenance.contract.version >= 1`

## Migration/Backfill Plan

1. Introduce helper module + unit tests (read/write normalization).
2. Migrate writers:
   - `src/fisheye/refinement/refine_detect.py`
   - `src/fisheye/refinement/refine_keypoints.py`
   - `src/fisheye/refinement/refine_eye_masks.py`
3. Update diagnostics (`check_provenance_capture`) to include `refined_eye_masks`.
4. Add optional backfill utility for legacy refinement runs:
   - inject missing `provenance.contract`
   - normalize git fields into `provenance.git.commit` when derivable
   - preserve original attrs; no destructive rewrites
5. Deferred follow-up:
   - migrate `src/fisheye/refinement/refine_online_detect.py` from ad-hoc
     provenance payload to `palette_stage_provenance` helpers.

## Acceptance Criteria

- A single helper call can read git commit/branch consistently from any refinement run.
- All new refinement runs write `provenance.contract` + canonical `provenance.git`.
- Provenance diagnostics report refinement-stage contract compliance uniformly.
- Legacy runs remain readable through helper fallbacks.
