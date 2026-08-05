# Whole-recording keypoint-v2 production adapter — 2026-08-05

Status: implementation and source-worktree dry-run complete; commit-pinned
deployment and first selector-ineligible Batman canary pending.

## Goal

Promote cache-backed whole-recording pose inference to the strict keypoint-v2
publication path without letting inference workers write authoritative arrays.
The workflow consumes one complete, verified crop-v2 pixel cache and produces a
manifest-bound four-surface candidate chain:

1. `keypoints_runs/<raw-run>`
2. `keypoint_quality_runs/<quality-run>`
3. `refined_keypoints_runs/<refined-run>`
4. `analysis/body_frame_runs/<body-frame-run>`

All four runs remain selector-ineligible and unregistered until a separate,
reviewed activation operation.

## Implemented boundary

- [x] Require an exact registered model set, run, model digest, training
  manifest, and pose-schema binding.
- [x] Require one complete crop-v2 geometry run and an external flat ROI cache
  bound to that run and analysis archive.
- [x] Copy and SHA-256-authenticate the cache in the same sequential pass used
  to stage it to node-local scratch.
- [x] Run inference only into a noncanonical `keypoint_shard_runs` terminal
  group in a private node-local Zarr shell.
- [x] Seal terminal arrays, model identity, preprocessing, cache manifest,
  payload digest, staging proof, and terminal success/failure semantics into an
  immutable workflow artifact outside the analysis archive.
- [x] Reopen terminal evidence and the exact persisted crop-v2 authority in the
  finalizer.
- [x] Reconstruct the recording-level rowset with stable `instance_key`,
  `source_crop_row_ids`, `frame_indices`, and `frame_row_offsets` contracts.
- [x] Derive raw, quality, refined, and body-frame snapshots using the existing
  strict schema builders.
- [x] Plan each array from uncompressed bytes, dtype, per-row shape, and access
  class through `published_http_v1`; legacy row-count shard flags have no
  effect on v2 publication.
- [x] Write complete candidates on node-local scratch, validate them, atomically
  import each immutable run, consolidate the archive once, and reopen the
  public paths through consolidated metadata.
- [x] Refuse direct selector changes, registry writes, raw writer shortcuts,
  model fallback, crop mismatches, output collisions, malformed manifests, and
  incomplete chains.
- [x] Make the terminal DAG job a read-only consumer of the analysis archive.
- [x] Make the terminal DAG fan-in a candidate validator rather than a registry
  mutator.

The direct-writer fence does not prohibit publication. It prohibits the model
inference process from writing `keypoints_runs` directly. Only the strict
finalizer may publish canonical candidates after complete terminal evidence has
been validated.

## Storage and execution policy

The canonical writers do not use one row constant for every dtype. Their inner
chunks and outer shards are derived by the shared byte planner. The current
finalizer writes whole, non-overlapping physical units serially. Historical
`--keypoint-*-shard-rows` and Dask refinement controls remain accepted only so
old command templates fail predictably; the v2 plan records that they have no
effect.

The flat cache remains an ephemeral pixel-materialization input, not an
analysis-array authority. The cache payload is not reread merely to hash it:
the node-local staging copy computes SHA-256 while transferring the bytes and
compares it with the cache manifest before inference begins.

## Frozen first canary

The initial canary uses the smallest Batman crop-v2 recording:

- recording: `2026-07-21T19-38-32Z_arena_2_Batman`
- crop run: `crop_geometry_v2_348_20260805`
- rows: 126,214
- ROI shape: 348×348 `uint8`
- cache manifest:
  `/nrs/johnson/palette_staging/flat_roi_cache/batman_crop_geometry_v2_348_20260805/roi_cache/2026-07-21T19-38-32Z_arena_2_Batman.flat_roi_cache.json`
- next run label: `batman_kpt5_v2_canary_20260805_v003`

Exact model:

- set: `pose_all_registry_reviewed_v2_keypoints_20260520_v001`
- run: `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`
- model SHA-256:
  `cce63d534a8f1491db1e2c71cb9236768c445722013dc39faeaf62a9d0a9a377`
- training-manifest SHA-256:
  `5cfe9cefdeb5adde2eb35e26e469c1898cd31b007274b259272a42a6c1cdc317`
- labels: `swim_bladder`, `eye_left`, `eye_right`, `snout_tip`, `tail_tip`

As of the 2026-08-05 registry census, this is the most recent successful pose
training run with that exact ordered five-keypoint skeleton. The selected model
is the immutable `/groups` copy at
`models/pose/<set>/<run>/weights/best.pt`; inference must never resolve model
weights from `/nvme1`.

The registry's historical training-manifest path is workstation-local under
`/nvme1` and is not visible to cluster jobs. The deployed model packages the
same manifest at `models/pose/<set>/<run>/inputs/<manifest-name>`. Resolution
may use that packaged manifest only when the registered path is absent and its
SHA-256 exactly equals the registered training-manifest digest. This is a
provenance-path repair, not a model fallback.

The target manifest is
`docs/diagnostics/batman_keypoint_v2_candidate_20260805/targets.canary.json`.

The 2026-08-05 source-worktree dry-run resolved the exact registered model,
validated the real crop-v2 archive and NRS cache, confirmed 126,214 rows and a
15,285,020,256-byte payload, and rendered exactly three jobs: terminal
prediction, strict four-surface finalization, and candidate validation. It
submitted no jobs. The same dry-run must be repeated from the immutable cluster
deployment before submission.

## Execution history

- `v001` attempted submission from the workstation, where `bsub` is
  unavailable. It submitted zero jobs and changed no Zarr.
- `v002` submitted from the Citrus poller and failed closed before inference
  because the registered `/nvme1` training-manifest path was unavailable on
  the cluster. Its dependent jobs exited without publication.
- `v003` is reserved for the digest-verified packaged-manifest repair. It must
  use the same exact `/groups` model path and digest and remain
  selector-ineligible.

## Canary checklist

- [ ] Commit this implementation with a clean worktree.
- [ ] Deploy that exact commit with
  `scripts/deploy_palette_cluster_worktree.sh` and retain the printed
  `PALETTE_GROUPS_REPO`.
- [ ] Render the LSF plan using the commit-pinned deployment and `--dry-run`.
- [ ] Verify the plan contains exactly prediction → strict finalization →
  candidate validation, with no activation command.
- [ ] Review requested queue, GPU, CPU, memory, batch size, and run paths.
- [ ] Submit only the single canary after review.
- [ ] Confirm cache SHA-256 staging proof and exact model binding.
- [ ] Confirm all four candidate manifests, dtypes, logical hashes, storage
  plans, direct/consolidated equivalence, and source-chain digests.
- [ ] Confirm no production selector or registry row changed.
- [ ] Give Crimson the explicit refined-keypoint and crop run IDs for exact
  typed open, traversal, cancellation, identity, and rendering validation.
- [ ] Activate selectors only through a later reviewed gate if both Palette and
  Crimson approve the canary.
- [ ] Expand to the remaining 35 Batman recordings only after that checkpoint.

## Dry-run template

```bash
"${PALETTE_GROUPS_REPO}/scripts/submit_whole_recording_keypoints_bsub.sh" \
  --manifest "${PALETTE_GROUPS_REPO}/docs/diagnostics/batman_keypoint_v2_candidate_20260805/targets.canary.json" \
  --run-label batman_kpt5_v2_canary_20260805_v003 \
  --run-root /groups/johnson/johnsonlab/jeremy/logs/whole_recording_keypoints/batman_kpt5_v2_canary_20260805_v003 \
  --repo "${PALETTE_GROUPS_REPO}" \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --model-set-id pose_all_registry_reviewed_v2_keypoints_20260520_v001 \
  --model-run-id pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2 \
  --pose-schema traditional_v2 \
  --min-roi-size 348 \
  --input-mode tensor \
  --dry-run
```

Dry-run materializes only plan evidence. It does not submit LSF jobs or mutate
any analysis archive, selector, or registry row.
