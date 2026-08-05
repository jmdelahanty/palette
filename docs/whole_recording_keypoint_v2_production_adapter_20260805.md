# Whole-recording keypoint-v2 production adapter — 2026-08-05

Status: implementation complete and first selector-ineligible Batman canary
passed Palette publication validation and Crimson interoperability review.
Scientific acceptance of the observed pose yield and any separate selector
activation remain pending.

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
old command templates fail predictably; the v3 execution plan records that they have no
effect.

The flat cache remains an ephemeral pixel-materialization input, not an
analysis-array authority. The cache payload is not reread merely to hash it:
the node-local staging copy computes SHA-256 while transferring the bytes and
compares it with the cache manifest before inference begins.

The inference tensor shape is derived rather than hardcoded. For tensor and
auto input modes, the planner applies the existing reversible
`ModelInputTransform` contract and selects the smallest centered square extent
divisible by the planned maximum model stride. The current five-keypoint model
therefore maps native 348x348 pixels to a zero-padded 352x352 tensor with two
pixels on every edge. Predictions are inverse-mapped into native ROI
coordinates before terminal arrays are written. The worker loads the exact
model, reads its declared maximum stride, and fails before ROI inference unless
that value matches the planned stride. Both the transform and verified stride
are persisted in preprocessing provenance.

## Frozen first canary

The initial canary uses the smallest Batman crop-v2 recording:

- recording: `2026-07-21T19-38-32Z_arena_2_Batman`
- crop run: `crop_geometry_v2_348_20260805`
- rows: 126,214
- ROI shape: 348×348 `uint8`
- cache manifest:
  `/nrs/johnson/palette_staging/flat_roi_cache/batman_crop_geometry_v2_348_20260805/roi_cache/2026-07-21T19-38-32Z_arena_2_Batman.flat_roi_cache.json`
- next run label: `batman_kpt5_v2_canary_20260805_v005`

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
- `v003` proved the digest-verified packaged-manifest repair from the cluster
  and submitted no jobs. Its plan lacked an explicit full Palette commit.
- `v004` used the exact deployment commit and model binding, then failed safely
  before inference because tensor mode received an identity 348x348 transform
  that was not divisible by the model's maximum stride of 32. Prediction job
  `153273489` failed; dependent finalization job `153273490` and validation job
  `153273491` did not run. Scratch cleanup completed and no terminal arrays,
  canonical runs, selectors, or registry records were created.
- `v005` retains the same exact cache, model, and publication boundary. Its only
  scientific-input change is enforcing the existing reversible preprocessing
  contract as derived 348x348 to 352x352 centered padding, bound to the stride
  reported by the loaded model. It ran from Palette commit
  `9598f402e27c18b5ff2dfc390811cc0472a5eaec` in the detached cluster worktree
  `keypoint-whole-recording-production-20260805-9598f402`.

## v005 result

All jobs completed successfully on 2026-08-05:

- prediction: `153273676`, L4 node `h08u02`;
- strict four-surface finalization: `153273677`, node `h07u29`;
- selector-ineligible candidate validation: `153273678`, node `h07u31`.

Inference processed all 126,214 cache rows on `cuda:0` in 226.19 inference
seconds. It produced 89,527 successful poses and 36,687 explicit failures
(70.93% success), reporting 395.81 poses/s for the complete inference phase.
The sealed terminal receipt records the exact model and training-manifest
digests, cache manifest and payload digests, tensor input mode, maximum stride
32, and the centered 348x348 to 352x352 padding transform. Job-local cache and
terminal scratch were removed after the durable terminal artifact was sealed.

The finalizer atomically published and independently validated:

- `keypoints_runs/keypoints_batman_kpt5_v2_canary_20260805_v005`;
- `keypoint_quality_runs/keypoint_quality_batman_kpt5_v2_canary_20260805_v005`;
- `refined_keypoints_runs/refined_keypoints_batman_kpt5_v2_canary_20260805_v005`;
- `analysis/body_frame_runs/body_frame_batman_kpt5_v2_canary_20260805_v005`.

All four runs are complete and `stage_selector_eligible=false`. Direct parent
metadata retains no `latest`, `latest_complete`, or `pending` value for these
runs. The final candidate validator reported zero errors, registry integrity
`ok` before and after, `activation_performed=false`, and no registry mutation.

The only emitted warning was Zarr 3.1.6's standard notice that consolidated
metadata is not yet part of the final Zarr v3 specification. Direct and public
validation otherwise passed with no metadata errors. Updating the warning
classifier for this library wording is a non-scientific follow-up; it does not
authorize selector activation.

## Post-canary pose-yield investigation

The 70.93% success rate is a model/inference outcome, not a storage or
publication failure. All 126,214 requested observations reached an explicit
terminal state, and all four recording-level surfaces were published and
validated. Read-only investigation found:

- the immediate failed-row outcome was that postprocessing returned no usable
  pose at the configured `conf=0.25` threshold;
- successful rows reached as low as approximately `0.250004`, so some outcomes
  lie close to that threshold;
- sampled successful and failed crops had nearly identical brightness and
  contrast distributions, providing no evidence for blank or corrupt cache
  rows;
- 1,436 rows used explicit padding and 620 of those rows failed; padded rows
  therefore account for only 1.69% of all 36,687 failures;
- padded rows failed more often than unpadded rows (43.18% versus 28.90%), so
  edge geometry is a contributing covariate but cannot explain most failures;
- failures were temporally and row-clustered rather than distributed like
  independent storage corruption; and
- failure rate varied strongly with detection confidence, from 77.78% in the
  `[0.4, 0.5)` stratum to 14.31% in `[0.8, 0.9)`.

The strongest current hypothesis is model-domain mismatch. The selected model
was trained from 512x512 source crops resized to a 256x256 model input, whereas
the Batman canary uses 348x348 source crops mapped by centered zero padding to
352x352 with no scale change. The canary proves that the latter transform is
exactly bound and reversible; it does not prove that the trained model is
scientifically adequate under the changed apparent scale and context.

## Terminal failure provenance

Palette commit `1327fe9f833c12d6d5e197426ab8f8fb4a430b11` adds an exact
terminal-only `pose_failure_codes: uint8[N]` contract. Code zero is valid only
for a successful pose row. Declared nonzero outcomes distinguish no detection
above threshold, missing keypoint payload, empty payload, and insufficient
keypoint cardinality. The immutable terminal receipt binds the exact code map,
complete histogram, and array digest. Unknown codes and disagreement with the
success mask fail closed.

This evidence remains outside the already frozen public raw-keypoint-v2 array
set. Strict v2 preparation validates it but does not add an optional public
array. Any public adoption requires an explicit schema revision.

The v005 canary predates this implementation. Its rows cannot be assigned to
the new subclasses retrospectively; collecting an exact histogram requires a
new terminal inference run. Job-wide model/skeleton or array-cardinality
incompatibilities remain hard job failures rather than row-level failure codes.

## Mixed crop-size training decision

Training data may contain both 348x348 and 512x512 source crops, but source
geometry must not be erased. Each example must retain its crop policy, native
shape, source coordinate contract, and exact source-to-model transform.
Batching should normalize or shape-bucket pixels at the model boundary, while
labels are transformed through the same reversible geometry.

The current canary should not be rerun merely to populate the new failure-code
histogram. First, inference must fail closed unless the selected model package
declares—and the worker applies—the exact source-pixel-to-model transform used
during training. Matching only the final tensor dimensions is insufficient.
For the current model, `512x512 -> 256x256` is the relevant training transform;
naively resizing `348x348 -> 256x256` changes anatomical scale and field of
view, while the current `348x348 -> 352x352` padding changes both the input
extent and apparent model scale.

The safest diagnostic rerun for the current model is therefore to materialize
the same 512x512 source-camera window and apply its exact historical 256x256
preprocessing. The resulting landmarks should be inverse-mapped into
source-camera pixels and then projected into the canonical 348x348 crop
geometry by stable `instance_key`; the inference pixel window need not replace
the published downstream crop authority. If the historical transform cannot
be reconstructed exactly from digest-bound model metadata, the job must stop
rather than infer a resize policy.

The next training checkpoint should:

1. add manually reviewed Batman 348x348 examples rather than treating failed
   inference as ground truth;
2. split by recording or session so nearby frames cannot leak across train and
   validation sets;
3. retain a Batman-domain holdout and report metrics separately for 348x348
   and 512x512 source profiles;
4. compare the current 348-to-352 transform at multiple diagnostic confidence
   thresholds with the exact historical 512-to-256 inference path; a
   348-to-256 resize may be studied as a separate candidate but must not be
   described as reproducing training scale;
5. train a successor against the intended production crop/input policy before
   deciding whether the current 70.93% candidate is scientifically acceptable.

Multi-scale augmentation may improve robustness, but it does not by itself
resolve a change in source field of view, anatomical scale, or acquisition
domain.

## Crimson interoperability verdict

Crimson reported `PASS` for refined source-binding v2 using exact,
digest-validated skeleton semantics. It performed no dtype probing, selector
fallback, or archive writes. Across five fresh processes it reported:

- median readiness 353 ms, with a 3.98 s worst uncontrolled-cache trial;
- warm random-frame p95 0.137 ms;
- forward 70-frame-page p95 4.36 ms;
- zero deadline misses and zero stale frames;
- 32.23 MB transferred in 109 reads per process;
- approximately 104.6 MB peak RSS;
- zero quality-payload reads; and
- exactly one read of each retained offset index.

The Metal visual gate passed at frame 1,000 with 11 keypoint primitives, one
body-frame heading, and exact live-crop placement. It confirms visible
alignment for the inspected frame, not an exhaustive search of every padded or
ROI-edge case. Crimson also reported 79/79 passing macOS headless tests.

The supplied immutable evidence reference is Crimson commit
`f4edbff7b5c3e6d341395f35092b2a8997d5c3d5`, with evidence under
`docs/diagnostics/keypoint_v2_batman_canary_2026-08-05/`. That commit was
reported clean but remains local and unpushed. It must be pushed or otherwise
made durably retrievable before Palette uses it as activation provenance.

## Activation boundary

Crimson's result completes the consumer interoperability gate. It does not
alone activate production selectors. Activation still requires an explicit
review that separates:

- schema, identity, storage, and publication correctness, which passed;
- consumer correctness and performance, which passed for the tested canary;
- scientific acceptance of the current pose-yield profile, which remains a
  project decision; and
- one atomic, fail-closed activation operation for the raw, quality, refined,
  and body-frame authority chain, followed by final consolidated-metadata
  validation and registry update.

## Canary checklist

- [x] Commit this implementation with a clean worktree.
- [x] Deploy that exact commit with
  `scripts/deploy_palette_cluster_worktree.sh` and retain the printed
  `PALETTE_GROUPS_REPO`.
- [x] Render the LSF plan using the commit-pinned deployment and `--dry-run`.
- [x] Verify the plan contains exactly prediction → strict finalization →
  candidate validation, with no activation command.
- [x] Review requested queue, GPU, CPU, memory, batch size, and run paths.
- [x] Submit only the single canary after review.
- [x] Confirm cache SHA-256 staging proof and exact model binding.
- [x] Confirm all four candidate manifests, dtypes, logical hashes, storage
  plans, direct/consolidated equivalence, and source-chain digests.
- [x] Confirm no production selector or registry row changed.
- [x] Give Crimson the explicit refined-keypoint and crop run IDs for exact
  typed open, traversal, cancellation, identity, and rendering validation.
- [x] Pass Crimson's exact-schema, traversal, cancellation, performance, and
  visual-alignment canary gate.
- [x] Add exact terminal-only failure codes and bind them into v2 preparation
  receipts without changing the frozen public raw-v2 array set.
- [ ] Push or otherwise durably publish Crimson evidence commit
  `f4edbff7b5c3e6d341395f35092b2a8997d5c3d5`.
- [ ] Decide whether 70.93% current-model yield is acceptable for initial
  production use or requires a diagnostic matrix, new 348-domain labels, and a
  successor model first.
- [ ] Make the model package declare the training source geometry and complete
  source-to-model preprocessing transform, and make inference fail closed when
  that transform cannot be reproduced exactly.
- [ ] Rerun a selector-ineligible terminal canary using the enforced
  model-trained scale when an exact failure-code histogram is needed; v005
  cannot be retroactively classified.
- [ ] Implement and review the atomic four-surface selector activation gate.
- [ ] Activate selectors only through that later reviewed gate after final
  direct/consolidated metadata and registry validation.
- [ ] Expand to the remaining 35 Batman recordings only after that checkpoint.

## Dry-run template

```bash
"${PALETTE_GROUPS_REPO}/scripts/submit_whole_recording_keypoints_bsub.sh" \
  --manifest "${PALETTE_GROUPS_REPO}/docs/diagnostics/batman_keypoint_v2_candidate_20260805/targets.canary.json" \
  --run-label batman_kpt5_v2_canary_20260805_v005 \
  --run-root /groups/johnson/johnsonlab/jeremy/logs/whole_recording_keypoints/batman_kpt5_v2_canary_20260805_v005 \
  --palette-repo "${PALETTE_GROUPS_REPO}" \
  --palette-commit "${PALETTE_COMMIT}" \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --model-set-id pose_all_registry_reviewed_v2_keypoints_20260520_v001 \
  --model-run-id pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2 \
  --pose-schema traditional_v2 \
  --min-roi-size 348 \
  --input-mode tensor \
  --model-input-stride 32 \
  --dry-run
```

Dry-run materializes only plan evidence. It does not submit LSF jobs or mutate
any analysis archive, selector, or registry row.
