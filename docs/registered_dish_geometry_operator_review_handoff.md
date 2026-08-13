# Registered dish geometry operator-review handoff

Status: implemented but not automatically promoted as of 2026-08-13.

This is the operator handoff for recording-bound dish geometry. The governing
Palette decision record and implementation checklist is
`docs/registered_dish_geometry_production_implementation_checklist_2026-08-12.md`.
Acquisition agents should conform to Orange's
`docs/recording_geometry_contract.md` and
`docs/palette_stimulus_v5_producer_compliance.md`; Palette consumes the exact
checksummed recording contract, not the rounded compatibility mask.

## Safety boundary

- The acquisition physical inner rim and outward centroid gate are different
  circles and remain different artifacts.
- The independent fit is frozen before acquisition geometry is revealed.
- There is no acquisition-only fallback. A failed or unavailable offline fit
  cannot make acquisition geometry operational.
- `corroborated_acquisition_v1` is unpromoted. It cannot automatically select
  geometry until the frozen derivation/holdout canary and a timestamped policy
  promotion are complete.
- A failed comparison cannot be overridden by manually selecting one of the
  same candidates. Corrected evidence requires a new candidate and comparison.
- No step rewrites the recording, producer geometry, raw detections, or an
  existing immutable candidate/comparison/selection/gate.

## Review states and action

| State | Operator action |
| --- | --- |
| `corroborated_pass` | Review until the numerical policy is separately promoted. Automatic selection is currently blocked. |
| `review_required` | Inspect the frozen early/middle/late candidate family, image support, stability, and exact detection disagreement. Publish an explicit reviewed selection only if justified. |
| `projected_edges_unresolved` | Do not claim top-rim/inner-water-edge identity. Review center, stable rim family, acquisition-boundary image support, and operational disagreement; do not average radii. |
| `semantic_feature_incompatible` | Keep same-feature radius metrics inapplicable. Review operational consequences separately. |
| `offline_fit_failed_but_acquisition_geometry_valid` | Stop selection. Diagnose decode, imagery, keyframes, layout, or fitting and rerun independent evidence. |
| `producer_geometry_invalid` | Stop. Return the exact contract/checksum/scope failure to acquisition. |
| `coordinate_or_extent_mismatch` | Stop. Resolve camera, arena, native extent, or pixel-authority identity; do not flip or transform coordinates. |
| `comparison_failed` | Stop and preserve both inputs for diagnosis. |

## Artifact sequence

1. Publish the producer-native acquisition candidate from the recording folder
   or exact recording-bound Citrus H5.
2. Run the blind early/middle/late probe. When the acquisition observation is
   supplied, the reveal measures fixed-circle image support only after the fit
   report is frozen.
3. Review the montage and publish the Palette candidate without selecting it.
4. Publish the immutable comparison, optionally bound to the exact raw
   detection rowset.
5. For a reviewable, non-failing comparison, publish an explicit
   comparison-bound selection.
6. Materialize the keyed gate from that exact selection and raw detection
   source.
7. Run the whole-video or clipped recipe with `gate_requirement` set to
   `if_available` or explicitly opted-in `required`. Downstream work accepts
   only the immutable finalized refined-detection authority.

The producer-native acquisition dry run is:

```bash
scripts/py -m fisheye.utils.publish_acquisition_geometry_candidates \
  --recording /absolute/recording/root \
  --geometry-source producer-folder \
  --camera-serial CAMERA_SERIAL \
  --arena-id ARENA_ID
```

Add `--apply` only after the dry-run identity and target are correct. For the
H5 representation, use `--geometry-source citrus-h5 --citrus-h5 /exact/file.h5`.
Use `recovery-receipt` only for an approved historical recovery.

The comparison and reviewed selection dry runs are:

```bash
scripts/py -m fisheye.utils.publish_arena_geometry_comparison ANALYSIS_ZARR \
  --acquisition-candidate-run ACQUISITION_RUN \
  --palette-candidate-run PALETTE_RUN \
  --semantic-compatibility projected_edges_unresolved \
  --policy-id manual_review_only_v1 \
  --detect-source-group detect_runs/RAW_RUN

scripts/py -m fisheye.utils.publish_arena_geometry_selection ANALYSIS_ZARR \
  --candidate-run ACQUISITION_OR_REVIEWED_PALETTE_RUN \
  --comparison-run COMPARISON_RUN \
  --selected-by REVIEWER \
  --decision-reason 'specific evidence-based reason'
```

Both commands are read-only plans unless `--apply` is supplied. Record the
printed candidate/comparison/selection IDs and digests; never substitute a
mutable `latest` pointer in a planned workload.

## Evidence still required for automation

- Checksummed August 10 derivation and August 11 locked-holdout manifests.
- Per-camera and per-registration distributions for center, rim-family image
  support/residual, temporal stability, acquisition-boundary support, and
  exact-row gate disagreement.
- Operator-adjudicated false-pass and false-review outcomes, including all
  injected fail-closed controls.
- A bounded whole-video versus split-clipped canonical-lineage parity canary.
- A new policy version and timestamped promotion/activation decision after the
  holdout passes unchanged thresholds.
