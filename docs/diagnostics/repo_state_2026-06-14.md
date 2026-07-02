# Palette repo state handoff: 2026-06-14

Purpose: durable handoff for the current engineering state. This is an index
and decision log, not a replacement for the detailed contract docs.

## Current Load-Bearing Changes

### Zarr run completion is now strict for active non-eye scopes

The completion-epoch work has moved from design into archive state for the
active non-eye StageSpec-backed scopes.

Implemented pieces:

- Parent-group strict completion epochs are implemented in
  `src/fisheye/shared/zarr_run_completion.py`.
- The archive backfill tool is implemented in
  `src/fisheye/utils/backfill_completion_epoch.py`.
- Existing parent groups under `/nvme1/recordings` have been backfilled in
  stages.
- Active non-eye compatibility blockers have been resolved by policy: invalid
  non-latest legacy children are left unmarked and ignored under strict mode;
  invalid `latest` / `latest_complete` children still block.

Latest guarded apply:

```bash
scripts/py -m fisheye.utils.backfill_completion_epoch \
  --recordings-root /nvme1/recordings \
  --stage crop --stage detect --stage keypoints \
  --stage refined_detect --stage refined_keypoints --stage refined_subject_masks \
  --apply \
  --expect-store-count 138 \
  --expect-non-ok-store-count 0 \
  --expect-blocked-parent-count 0 \
  --expect-would-stamp-parent-count 65 \
  --expect-would-mark-child-count 120 \
  --expect-ignored-legacy-child-count 74 \
  --expect-write-failed-parent-count 0 \
  --expect-applied-stamped-parent-count 65 \
  --expect-applied-marked-child-count 120
```

Result:

- 65 parent strict epochs written.
- 120 verified legacy children marked complete.
- 74 invalid non-latest legacy children left unmarked and ignored.
- 0 blockers.
- 0 write failures.

Detailed source of truth:

- `docs/zarr_run_completion_strict_mode_todo.md`
- `/tmp/completion_epoch_active_non_eye_after_ignore_policy_apply.json`
- `/tmp/completion_epoch_backfill_post_active_non_eye_apply_summary.json`

### Detection/refinement surface is clarified

Current detection contract:

- `detect_runs/<run>` is still a raw candidate-detection surface with top-level
  arrays such as `frame_indices`, `bbox_norm_coords`, `scores`, `class_ids`,
  and `frame_counts`.
- `refined_detect_runs/<run>/source_detections` is the candidate/provenance
  projection.
- `refined_detect_runs/<run>/instances` is the accepted refined instance
  surface and can represent multiple accepted instances per frame.
- One-fish-per-frame workflows commonly use `--per-frame-top-k 1`; excluded
  candidates remain in `source_detections` rather than becoming accepted
  `instances`.

Implication: old refined-detect runs missing `source_detections` / `instances`
are legacy surfaces, not valid current-contract refined detections.

### Pixel/decode contract remains mid-migration

The pixel/decode contract is understood but not fully enforced everywhere.

Current position:

- Orange-style detection uses NV12/RGB-style preprocessing.
- Crop/keypoint/mask paths can use `pynvvc_luma` replicated inputs.
- Current training zarrs and source videos do not consistently carry enough
  decode backend, colorimetry, or encoded source-video metadata to prove pixel
  equivalence after the fact.

Detailed docs:

- `docs/video_pixel_model_input_contract.md`
- `docs/diagnostics/pixel_contract_audit_2026-06-05.md`
- `docs/import_step_design_review_2026-06-04.md`
- `docs/pipeline_stage_review_2026-06-04.md`

## Remaining Blockers

Post-apply full verification:

```bash
scripts/py -m fisheye.utils.backfill_completion_epoch \
  --recordings-root /nvme1/recordings \
  --summary-only \
  --output-json /tmp/completion_epoch_backfill_post_active_non_eye_apply_summary.json \
  --blocked-jsonl /tmp/completion_epoch_backfill_post_active_non_eye_apply_blocked.jsonl \
  --no-stdout
```

Result:

- 138 stores scanned.
- 138 ok stores.
- 0 non-ok stores.
- 210 blocked parents remain.
- 8 stampable parents remain, all deprecated eye-mask scopes.
- 0 write failures.

Remaining blocked scopes:

- `analysis/stimulus_response_runs`: 52 no-spec parents.
- `analysis/swim_bout_runs`: 52 no-spec parents.
- `eye_masks`: 53 deprecated/latest-invalid parents.
- `refined_eye_masks`: 53 deprecated/latest-invalid parents.

Decision already made: eye-mask scopes are deprecated and should not drive
current active-data validity. They may remain deferred unless a deletion or
archive policy requires stricter cleanup.

Decision still open: whether `analysis/stimulus_response_runs` and
`analysis/swim_bout_runs` should get layout-specific validators, be migrated to
a common surface, or be explicitly deferred long term.

## Active Engineering Risks

### Runtime contracts are improving, but not done

Completion epochs are now load-bearing for active non-eye archive scopes. The
remaining runtime-hardening work is mainly enforcement coverage and stale
surface retirement, not the core strict-mode mechanism.

Risk: adding new writers that bypass `require_runs_parent` or completion
markers would reintroduce untrusted run groups. Existing ratchet tests are the
guardrail.

### Pixel provenance is still the biggest cross-model uncertainty

The repo needs a clear metadata and enforcement path for:

- decode backend
- source video codec/colorimetry/range metadata
- model input representation
- persisted training image representation
- export refusal when merged datasets mix incompatible pixel contracts

Until this is enforced, model comparisons can still confound training data,
offline inference, and Orange deployment preprocessing.

### Native TensorRT inference path is still TODO

TensorRT export/build artifacts are tracked conceptually, but native TensorRT
inference is not yet the canonical runtime path in Palette.

Detailed TODO:

- `docs/native_tensorrt_inference_todo.md`
- `docs/orange_a16_tensorrt_engine_handoff.md`

### Eye-mask severance was a policy cleanup

Eye masks were deprecated and the standalone writer surface was severed on
2026-07-01. Historical notes remain archived for provenance; current eye-capable
mask work routes through subject-mask components.

Archived docs:

- `docs/archive/eye_mask_severance_plan_2026-05-28.md`
- `docs/archive/eye_mask_severance_phase0_coverage_audit_2026-05-28.md`
- `docs/archive/eye_mask_severance_phase1_verification_delta_2026-05-30.md`

## Recommended Next Slices

1. Commit the strict completion-epoch implementation and docs together.
   Include `src/fisheye/utils/backfill_completion_epoch.py`,
   `src/fisheye/utils/triage_completion_epoch_blockers.py`,
   `src/fisheye/shared/zarr_run_completion.py`,
   `src/fisheye/shared/zarr/stage_arrays.py`, tests, and this doc if they are
   part of the same working-tree slice.

2. Decide the policy for `analysis/stimulus_response_runs` and
   `analysis/swim_bout_runs`.
   Options: add layout-specific validators, migrate old runs to a common
   surface, or explicitly defer those parent paths in the completion-epoch
   plan.

3. Do the pixel-contract Phase 0 metadata pass.
   Add/stabilize metadata for decode backend, source-video encoded properties,
   colorimetry/range, and model-input representation before changing pixels.

4. Convert the native TensorRT TODO into an implementation slice.
   Keep export/build provenance separate from runtime inference, and avoid
   forcing Ultralytics runtime semantics into Palette inference code.

5. Continue eye-mask severance as a cleanup/deprecation track.
   Do not let deprecated eye-mask blockers hold active dataset validity hostage.

## Do Not Regress

- Do not reintroduce `legacy_default=True` in production readers.
- Do not stamp a strict parent when its `latest` or `latest_complete` child is
  invalid.
- Do not mark invalid legacy children complete just to silence blockers.
- Do not silently mix detection/refined-detection surfaces in training exports.
- Do not make pixel-contract decisions implicit; record backend and
  color/range metadata before relying on parity claims.
