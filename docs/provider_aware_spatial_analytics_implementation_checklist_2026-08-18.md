# Provider-Aware Spatial Analytics Implementation Checklist

<!-- contract-meta
status: accepted-implementation-checklist
decision_date: 2026-08-18
implementation: selector-ineligible-materializers-and-cross-stage-adapter
promotion_status: selector-ineligible-only
-->

Purpose: implement a reusable stimulus-selection, trajectory, occupancy,
contrast, cohort, and plotting foundation over explicit subject-position
providers. The first canary compares detection- and keypoint-derived position
without making either provider a new scientific default.

This checklist narrows the next implementation slice of:

- [Composable Stimulus Analysis and Plot Recipe Design](composable_stimulus_analysis_and_plot_recipes_design.md);
- [Position, Body-Frame, and Motion Provider Design](position_body_frame_and_motion_provider_design.md);
- [Subject Position Storage Contract v1](subject_position_storage_contract_v1.md); and
- [Derived Analysis Run Contract](derived_analysis_run_contract.md).
- [Compact Array-Backed Provenance Contract](compact_array_backed_provenance_contract.md).

Existing immutable stimulus, detection, position, tracking, occupancy, motion,
and visualization runs remain unchanged. New publications are immutable
successors or new run families. No production selector, registry authority, or
provider default may change during this checklist's canary phases.

## Implementation checkpoint: selector-ineligible materializers and cross-stage adapter (2026-08-18)

The current implementation checkpoint extends the pure foundation with
immutable, selector-ineligible Zarr materializers and an exact published-run
cross-stage adapter. It remains a development checkpoint, not a production
rollout:

- `analysis_workflows.composable_stimulus_selection` implements authority-bound
  atomic step/annotation references, exact member/union/intersection/difference
  expressions, directional trim policy, source-membership-preserving overlap,
  occurrence and pooled interval resolution, and independent request/resolved
  digests.
- `analysis.provider_spatial_trajectory` validates Palette's canonical integer
  `[track_id, acquisition_frame]` keys, complete selected-frame membership,
  single-subject frame uniqueness, explicit camera-to-arena-mm transforms,
  multi-occurrence frame membership, independent validity states, and
  deterministic coverage evidence.
- `analysis.provider_occupancy_v2` computes provider-neutral per-occurrence and
  pooled occupancy over exact float64 grids. Expected exposure comes from the
  complete resolved selection, not merely from provider rows that exist.
  Overlapping frames count once in the pooled product and once in each exact
  contributing occurrence.
- `analysis.provider_occupancy_contrast` implements strict difference-only
  contrasts. It requires exact digested estimator, track-policy, coordinate,
  transform, geometry, timing, grid, selection, occurrence, and source-manifest
  identities and rejects cross-provider ordinary contrasts.
- `analysis_workflows.materializers.composable_stimulus_selection` publishes
  exact named selection runs with immutable arrays, manifests, direct versus
  consolidated validation, and unchanged parent selector state.
- `analysis_workflows.materializers.provider_spatial_trajectory` publishes
  unsmoothed selector-ineligible trajectories with exact row lineage,
  `track_sample_policy_id`, source-camera extent validity, and separate reason
  and coverage states.
- `analysis_workflows.materializers.provider_occupancy_v2` publishes the
  provider-neutral per-occurrence and pooled occupancy arrays with shared
  storage declarations, exact source bindings, conservation checks, and final
  consolidated visibility validation.
- `analysis_workflows.materializers.provider_occupancy_contrast` consumes the
  actual occupancy-v2 source manifest and publishes strict pooled differences
  with exact source-arm references.
- `analysis_workflows.provider_spatial_pipeline` reads exact published,
  complete, selector-ineligible selection and trajectory runs, validates their
  manifests, arrays, direct/consolidated equivalence, and recomputes the
  occupancy result from the complete selection denominator. Its E2E tests cover
  distinct selection/trajectory/occupancy lineages through contrast publication,
  tampered source arrays, stale consolidated metadata, and mismatched results.

The complete new focused suite passes (`106 passed`). The maintained
provider-offer, binding, and resolved-epoch baseline also passes (`47 passed`,
with three existing Zarr-v3 consolidated-metadata warnings). Both suites ran
outside the Codex sandbox as required by `AGENTS.md`. Static compile, Ruff, and
`git diff --check` validation also pass.

At that checkpoint, the canonical source adapters and millimetre grid were
still open. The 2026-08-19 checkpoints below implement them and complete the
first real recording publication. Physical `ArrayContract` migration for
selection/trajectory/contrast, cohort products, plot recipes, Marimo
discovery, registry projection, required CI, and any provider promotion remain
open. No production selector, registry authority, or provider default has been
written or changed, and this branch is not merge-ready.

## Implementation checkpoint: exact canary adapters and frozen grid (2026-08-19)

This checkpoint implements the remaining read boundary for the first real
GoodBatBadBat provider comparison. It is still selector-ineligible and is not
a provider-promotion decision:

- `analysis_workflows.composable_epoch_selection_adapter` accepts only a
  loader-minted exact epoch-v2 selection plus explicit caller bindings from
  source window IDs to `black_before`, `chaser`, and `black_after`. It binds the
  source video, acquisition clock, recording timing, source timeline, epoch
  manifest, and interval evidence without inferring roles from labels or
  order. `all_black` is available only as an explicit composition.
- `analysis_workflows.provider_spatial_track_source` joins one exact
  observation-position run to its exact tracking projection exclusively by
  `uint64 instance_key`. It revalidates both loader-minted handles, preserves
  source failure evidence, excludes explicitly unassigned tracking rows, and
  rejects partial rowsets, frame disagreement, duplicate assigned frames, and
  cross-recording composition.
- `analysis_workflows.provider_spatial_grid_policy` freezes
  `goodbatbadbat_arena_mm_grid_v1`: 1 mm bins and a symmetric extent rounded
  outward from the exact selected circular arena boundary. Observed positions
  cannot influence the extent. A physical inner rim and a manually reviewed
  `visible_dish_top_rim_edge` remain distinct allowed boundary roles; an
  outward detection centroid gate is rejected as grid geometry.
- `utils.materialize_provider_spatial_canary` consumes one explicit task
  document, creates a provider-specific detection tracking successor when
  needed, and publishes three selection, six trajectory, six occupancy, and
  four contrast runs. Every output is immutable and selector-ineligible. The
  utility does not update the registry, any selector, or any source payload.

The canonical arena-2 read-only preflight passed with recording timing
authority `7b96148686e885648e20aa8d19f6cfa45c3b902ed7a2d78673b2760be7c2c3c8`,
100 Hz and 152,035 acquisition frames. The selected reviewed Palette circle
has radius 40.93503226842414 mm under the exact physical scale, yielding
float64 grid edges from -41 through +41 mm. The detection tracking plan binds
151,052 rows. The exact keypoint position/tracking join binds 150,788 rows and
does not absorb the 264 detection-only observations.

The new focused suite passes (`39 passed`), the read-only canonical authority,
epoch, geometry, detection-tracking-plan, and keypoint-join preflights pass,
and static compile, Ruff, and `git diff --check` pass. The real canary
publication evidence is recorded below. Required CI, cohort products, plots,
Marimo discovery, registry projection, and any provider promotion remain
separate gates.

### Read-only canary coordinate preflight

The canonical arena-2 canary currently selects reviewed Palette geometry
`arena_geometry_selection_06b5cd2c35c04917004e` with selection-record digest
`06b5cd2c35c04917004e52a897b3bae60cbbcfa8f2f97a2b80735826e0677026`.
Its reviewed circle is centred at
`(2286.7729648010045, 2307.6434917690376)` native-camera pixels with radius
`2152.594087583115` pixels. The exact source-camera physical authority records
`mm_per_pixel = 0.019016605362130807` and digest
`47758cca2a336a848300b92ebc77d953e74d417b0634915ada7421b63a401d69`.

Existing provider-motion `positions_mm` retain the source-camera physical
origin; they are not an arena-centred coordinate frame. The canary adapter
must therefore bind the reviewed geometry and apply an explicit translation
from source-camera pixels to arena-centred millimetres. It must preserve +X
right and +Y down and must not apply a presentation reflection or heuristic
Y flip.

The 2026-08-19 checkpoint froze the cohort grid explicitly. It uses a declared
1 mm scientific bin width and derives the outer extent from the selected arena
boundary and exact camera scale. For this recording that produces `[-41, 41]`
mm edges; it does not hardcode the nominal `[-40, 40]` mm dish or derive bins
from observed position minima and maxima.

## Real GoodBatBadBat canary publication (2026-08-19)

Campaign `goodbatbadbat_position_spatial_canary_20260819_v2` completed at
`2026-08-19T21:25:54.708229+00:00` from Palette commit
`1db19d7ba024192daa1354fd2e714d4a1f327ec0` on branch
`agent/palette/provider-comparison-canary-20260818`. The exact executed task
bytes had SHA-256
`b197691d8292bdd3501ef59b7d3cafa07cd6545a0c4936770aca97af5c4c141e`.
The parsed task is reproduced exactly in
[`diagnostics/provider_spatial_canary_2026-08-19/task.v2.json`](diagnostics/provider_spatial_canary_2026-08-19/task.v2.json),
whose formatted file SHA-256 is
`2538ce6dd421ab9fa45c6d88c5a71d420aab66ad957019f1bb2525da76a1e3a2`.
The result document had SHA-256
`a5f96e3546eff5d848a70be62de786ef97c29bbd066448ed57644f06548323f2`.

The executed command was:

```text
scripts/py -m fisheye.utils.materialize_provider_spatial_canary --task-json /tmp/goodbatbadbat_provider_spatial_canary_20260819_v1.task.json --scratch-root /tmp/goodbatbadbat-provider-spatial-canary-20260819-v2-scratch --result-json /tmp/goodbatbadbat-provider-spatial-canary-20260819-v2.result.json --apply --json
```

The immutable epoch source was
`analysis/stimulus_epoch_runs/stimulus_epochs_goodbatbadbat_canary_20260818_v2`
with manifest digest
`6232cb9f35635e942c525cc6ccb38721aec8953c949dd8a527d31aa9bddd96d7`.
The three selector-ineligible selection publications are under
`analysis/stimulus_selection_runs/`:

| Selection | Frames | Request digest | Resolved digest | Logical array manifest |
| --- | ---: | --- | --- | --- |
| `black_before` | 60,000 | `9717a598da1c3056e1acd06e9fee50ffad7aa23dd7b9c76d0255a1a9f2784a26` | `b00bc6b84ca79eec4a0e77eee59427cf9d87a033618eb40c0cc2a4700fc00634` | `190eed401edf701fadae33517b564ca3e90758f8f97c087f6d9414fa44529734` |
| `chaser` | 30,001 | `c5fb1d4a672a787643cf1cb6af86780deb4647204c6f601f454c61ef00f707f5` | `c6182fe2e43840d5659d385bdc7ef76b3ee97c9ce08feb1e13213d223ade41c4` | `9d6890d4d2b57d168553ba58657db480705903e5662ec4846050d61133f7a15f` |
| `black_after` | 59,999 | `5f7b03fa245356c4d0d54a38642a29c4994c64bae39d4088217da21b085e7c21` | `f37739f10063fe9a67a3235aa2e9775bc9211044236f41314985c91f66c476db` | `5b5b99c9aa4a10ad54b464e44bed31b959f7228a387fddc1e897079f2c8ba16e` |

The detection source binds 151,052 exact position/tracking rows with position
manifest `953aa1e19e7db52e5621333e232959b669e7090b85966551d5ead64718b44ad5`
and tracking manifest
`8f4d7b4390a6aab147a95cc43f0d6a4dcce4e4a6a6dc65ac6dbd93a2ad38a52b`.
The keypoint source binds 150,788 exact rows with position manifest
`3e47c00354477945b191685d8dc8dcd934f382a85b7fdf280c01f20169986d88`
and tracking manifest
`8c547bb482f7597080925718c550ab4faa9921a57aa6aa4fff1712cbdf473aa7`.
Both providers use the exact same resolved selections, 1 mm grid edges from
`-41` through `+41` mm, and camera-to-arena transform digest
`3d273f76e028c4f9847e6f33fb6f60cce69af51b1d41c86e2ef12d4ec311fb9d`.

Trajectory runs are under `analysis/provider_spatial_trajectory_runs/` and
occupancy runs are under `analysis/provider_occupancy_runs/`. The table gives
the exact child run name, exact publication manifest digests, and selected-row
coverage. Prefix each listed suffix with `provider_spatial_trajectory_` for the
trajectory child and `provider_occupancy_` for the occupancy child:

| Provider / selection | Child run suffix | Valid / expected frames | Missing frames | Provider-invalid rows | Trajectory manifest | Occupancy manifest |
| --- | --- | ---: | ---: | ---: | --- | --- |
| detection / `black_before` | `detection_black_before_goodbatbadbat_position_spatial_canary_20260819_v2` | 59,601 / 60,000 | 399 | 0 | `c8a6b2d6f7fffc3773fa4605d2778720e5233f9068ec8b9c440352caa1add0d9` | `eeb477ab5b9cb4833443e69b4498f3e7676af4908acd8322e841fdacd2cd1001` |
| detection / `chaser` | `detection_chaser_goodbatbadbat_position_spatial_canary_20260819_v2` | 29,688 / 30,001 | 313 | 0 | `ecff00c61780ce2d9c899f17d22c6ef61400347d0edae94dfc7ec187807c2d29` | `6b2a956114bbd975f6f6d011310292d0cad3f8bc1b04c57e56bb961a57706990` |
| detection / `black_after` | `detection_black_after_goodbatbadbat_position_spatial_canary_20260819_v2` | 59,728 / 59,999 | 271 | 0 | `0b730087ae59b0d71a9607b0a3a68d7293c574e5ad83dddc963d86445fc27329` | `e5b17f527133fa17803ba32e3e58f98b2d306af0652a6cb7025abe6d01e94960` |
| keypoint / `black_before` | `keypoint_black_before_goodbatbadbat_position_spatial_canary_20260819_v2` | 58,414 / 60,000 | 508 | 1,078 | `7572e445be7bd870ac95ab0dc71cfd6f5f6a0da586b659f1ab37ef18e21bf325` | `c9632a437630f0cfe78d31e974e6e8c4e6f1e711d6f4204946bce5db87e55a28` |
| keypoint / `chaser` | `keypoint_chaser_goodbatbadbat_position_spatial_canary_20260819_v2` | 29,034 / 30,001 | 381 | 586 | `b63fc08f623ce1c4b3dfc5b3af086ba4fcebdde03cc13109ba15e8de609e99e4` | `99a67b7b0a69c25783f0afeb65c6536b1a61eba31c2c87bd3c8c8a57a84697c2` |
| keypoint / `black_after` | `keypoint_black_after_goodbatbadbat_position_spatial_canary_20260819_v2` | 59,016 / 59,999 | 358 | 625 | `c6576f78d1fc0b8eff5208d2a979e3ea58bd72cd5d5220222a7812dbe7a6e654` | `7f5b78d07646dc74c0e7734aa7e466b745291ba19d1c8bb0b93f28d88a269eb7` |

All six products reported zero selected out-of-grid rows. Missing selected
frames and provider-invalid rows remain separate evidence; neither was
interpolated or filled from the other provider.

The four exact contrast children under
`analysis/provider_occupancy_contrast_runs/` are:

| Provider / contrast | Child run name | Manifest digest |
| --- | --- | --- |
| detection / `chaser - black_before` | `provider_occupancy_contrast_detection_chaser_minus_black_before_goodbatbadbat_position_spatial_canary_20260819_v2` | `7ee816cd3b24f5073bad96815d81041852f72ed8b16b712561083990363f70d2` |
| detection / `black_after - black_before` | `provider_occupancy_contrast_detection_black_after_minus_black_before_goodbatbadbat_position_spatial_canary_20260819_v2` | `2c5a0f095d4c6b27fb38015fc25cf4e435973f7c48fac081465f91641dcfd2b4` |
| keypoint / `chaser - black_before` | `provider_occupancy_contrast_keypoint_chaser_minus_black_before_goodbatbadbat_position_spatial_canary_20260819_v2` | `83c3f02356b144a91d9d884a9683127c96e16942b599c10e6a45db9793fa0d22` |
| keypoint / `black_after - black_before` | `provider_occupancy_contrast_keypoint_black_after_minus_black_before_goodbatbadbat_position_spatial_canary_20260819_v2` | `e0b79d9565862e70dc2f97ac8c01305dfbdde6f5b018455a9d8ce9ae4a8a361e` |

The post-publication metadata audit found exactly 19 expected artifacts: 3
selections, 6 trajectories, 6 occupancies, and 4 contrasts. Every root reports
complete and selector-ineligible, every staging record reports a successful
final validation and selector-ineligible policy, and every publisher recorded
identical parent selector attributes before and after publication. The result
also explicitly records `selector_updates: false`, `registry_updates: false`,
and `source_payloads_rewritten: false`.

The final aggregate focused suite passed `147` tests with `12` expected Zarr-v3
consolidated-metadata warnings in 95.09 seconds outside the sandbox:

```text
scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_composable_epoch_selection_adapter.py tests/unit/fisheye/test_composable_stimulus_selection.py tests/unit/fisheye/test_composable_stimulus_selection_materializer.py tests/unit/fisheye/test_materialize_provider_spatial_canary.py tests/unit/fisheye/test_provider_occupancy_contrast.py tests/unit/fisheye/test_provider_occupancy_contrast_materializer.py tests/unit/fisheye/test_provider_occupancy_v2.py tests/unit/fisheye/test_provider_occupancy_v2_materializer.py tests/unit/fisheye/test_provider_spatial_grid_policy.py tests/unit/fisheye/test_provider_spatial_pipeline.py tests/unit/fisheye/test_provider_spatial_track_source.py tests/unit/fisheye/test_provider_spatial_trajectory.py tests/unit/fisheye/test_provider_spatial_trajectory_materializer.py -q
```

An earlier mechanical V1 attempt published one valid selector-ineligible
diagnostic trajectory before a coordinate-frame compatibility check stopped
the run:
`analysis/provider_spatial_trajectory_runs/provider_spatial_trajectory_detection_black_before_goodbatbadbat_position_spatial_canary_20260819_v1`,
manifest
`7d2daee46c7ee70cffac06cc42e4e9ff8797c4984bbdcb1df5db0e148d5b1d63`.
It is not selected and was not reused as V2 evidence.

The deliberately serial canary took about 49 minutes. Independent immutable
provider/selection products remain the correct scientific publication model,
but production execution can fan out the six trajectory/occupancy branches
and fan in only for contrasts. The canary also exposed repeated full parent
validation/consolidation during contrast publication as a performance target;
a trusted-parent receipt fast path can be evaluated separately without
changing scientific semantics. The timestamped decision for this canary is to
retain both providers as comparison-only offers: successful publication is not
evidence to promote either provider as the GoodBatBadBat default.

## Implementation checkpoint: compact readable provenance v2 (2026-08-20)

The first real canary exposed a publication-shape defect rather than a loss of
scientific evidence. Its root consolidated `zarr.json` reached approximately
1.45 GB because row/occurrence-sized attribute values and complete upstream
manifests were copied recursively through selections, trajectories,
occupancies, and contrasts. The same exact values already existed in typed
arrays.

The accepted correction is the
[Compact Array-Backed Provenance Contract](compact_array_backed_provenance_contract.md):

- [x] Keep explicit policy IDs, formulas, parameters, units, authorities,
      source paths, codebooks, counts, and array declarations readable.
- [x] Treat digests as verification evidence, never as a replacement for the
      scientific description.
- [x] Store frame-, row-, observation-, occurrence-, and grid-cardinality
      values in typed arrays, referenced by path, dtype, shape, and digest.
- [x] Stop recursively embedding upstream manifests in downstream source
      bindings.
- [x] Store exact requested-selection and timeline-authority JSON bytes in
      arrays while retaining readable fixed selection summaries in metadata.
- [x] Replace row-sized provider reason lists with array references, a fixed
      reason codebook, and per-code counts.
- [x] Replace trajectory selection payload copies with exact compact selection
      authority and selection-array references.
- [x] Replace per-occurrence occupancy conservation lists with named arrays and
      explicit invariant formulas.
- [x] Replace contrast source-manifest, occurrence, and grid-edge copies with
      readable source-run/arm summaries and exact source/edge array references.
- [x] Add a reusable fail-closed structural metadata-cardinality guard and
      focused tests.
- [ ] Publish clean v2 canary successors and measure direct and consolidated
      metadata size before deciding whether to extract the old canary into a
      clean archive generation.
- [ ] Add a separate operational consolidated-metadata size budget after the
      legacy root no longer makes such a gate fail by construction.

Existing immutable canary runs and the existing large root metadata file are
not rewritten by this implementation. The branch remains selector-ineligible
and requires required CI before integration. The focused and adjacent
provider-spatial suite passed `152` tests with `12` expected Zarr-v3 warnings
outside the sandbox; Ruff, static compilation, and `git diff --check` also
passed.

## Accepted first-slice decisions

- [x] Start with `detection_bbox_centroid.v1` and
      `keypoint_anatomical_triad_mean.v1` as two separate explicit offers.
- [x] Keep component-mask and subject-body-mask position providers out of the
      first analytics canary while new mask labels and model evidence are being
      developed. Preserve their existing runs and comparison evidence.
- [x] Use one valid tracked subject sample per acquisition frame as the
      scientific occupancy sample unit. Never count arbitrary raw detection
      rows as independent exposure samples.
- [x] Require an exact observation-to-track-sample projection. Duplicate valid
      subjects in a single-subject recording fail closed; they are not reduced
      with first-row, highest-confidence, or mean-position heuristics.
- [x] Keep position and heading independent. Detection-derived trajectory,
      occupancy, speed, acceleration, and bouts may be computed without a
      heading provider.
- [x] Do not join the 264 detection-only canary observations to the keypoint
      body-frame rowset by cardinality or nearest row. Each position provider
      owns its exact tracking and linear-motion lineage.
- [x] Use source-camera pixels only as an explicitly labeled diagnostic
      coordinate product. Scientific cross-recording occupancy uses an exact
      selected camera/arena transform and a persisted arena-millimetre grid.
- [x] Fail closed for the scientific millimetre product when scale, extent,
      coordinate, selected geometry, or transform authority is missing or
      stale. Do not infer scale from raster dimensions or dish diameter.
- [x] Treat every stimulus step as an atomic state. `SOLID_BLACK` does not
      imply `baseline`, `pre`, or `post`; saved compositions assign those roles.
- [x] Materialize exact frame membership before calculating a metric. Metrics
      and renderers do not independently re-resolve stimulus steps.
- [x] Keep scientific normalization separate from display normalization.
      Per-panel maximum scaling is never an input to a contrast.
- [x] Make the recording-balanced cohort view primary for cohort comparison.
      Pooled-frame products remain separately labeled descriptive outputs.
- [x] Keep mask-derived heading, body-mask heading, gaze, turn-toward,
      circling, and provider promotion outside this implementation slice.

## Phase 0: freeze the implementation contracts

### Selection and frame-set contract

- [x] Name and version the immutable selection-expression and resolved-frame-
      set schemas without changing `stimulus_epoch_runs` v2.
- [x] Bind every selection to one exact recording, stimulus run, acquisition
      frame domain, source-video metadata record, acquisition-clock authority,
      and source metadata digest.
- [x] Support exact atomic-step references and exact interval-annotation
      references. Persist predicate text only together with its concrete
      resolved members.
- [x] Support the narrow v1 expression vocabulary: exact member,
      `union`, `intersection`, and `difference`.
- [x] Represent all intervals as ordered, de-duplicated, half-open acquisition
      frame intervals `[start_frame, end_frame)`.
- [x] Define overlap behavior explicitly: a resolved frame contributes at most
      once to a pooled metric while all source-membership evidence is retained.
- [x] Support `keep_occurrences` and `pool_intervals` as distinct aggregation
      policies. Preserve occurrence identity in either case.
- [x] Support directional leading and trailing trims. For the existing
      nominal-frame-clock v1 policy, remove `ceil(seconds * fps)` frames and
      persist requested seconds, effective frame count, and rounding policy.
- [x] Reject negative trims, trims that invert an interval, incompatible
      timelines, unresolved predicates, and unsupported expression operators.
- [x] Make roles such as `baseline`, `treatment`, and `control` explicit saved
      metadata. Never infer them from step mode, order, or display label.
- [x] Canonicalize and digest the requested expression independently from the
      resolved frame set so stale stimulus resolution is detectable.

### Provider-track binding contract

- [x] Define a typed input handle for one exact
      `analysis/subject_position_runs/track_sample/<run>` or for an exact
      observation-position run plus its immutable observation-to-track
      projection.
- [x] Require exact `track_sample_key`, acquisition frame, subject/track
      identity, coordinate descriptor, provider ID, estimator digest, source
      manifest, and recording-timing authority from a canonical published
      detection or keypoint source handle.
- [x] Require uniqueness of `(subject_track_identity, acquisition_frame)` for
      the first single-subject profile.
- [x] Preserve provider-present, provider-valid, in-selection, transform-valid,
      and in-grid states separately.
- [x] Publish or bind selector-ineligible track successors for both detection
      and keypoint position. Do not reuse a body-frame source as position
      evidence.
- [x] Reject implicit provider fallback, selector lookup, same-length joins,
      reordered keys, duplicate keys, stale manifests, and cross-recording
      composition.

### Scientific spatial-grid contract

- [x] Freeze one versioned GoodBatBadBat arena-millimetre grid profile before
      writing a canary. Record exact x/y edges as float64 arrays.
- [x] Choose the fixed grid extent and bin width from declared arena geometry
      and bounded canary evidence, not the observed position minima/maxima.
- [x] Define bin membership as left-closed/right-open, with the final outer
      edge inclusive. Persist the edge policy.
- [x] Record the selected arena geometry, physical scale, camera-to-arena
      transform, coordinate descriptor, and every authority digest used to
      project source positions.
- [x] Record out-of-grid finite samples separately. Do not clip them into edge
      bins or silently expand the grid.
- [ ] Give any camera-pixel diagnostic grid a different policy ID and prevent
      it from entering millimetre-grid cohort contrasts.

## Phase 1: pure composable selection compiler

- [x] Implement pure schema models, canonical JSON, and digest helpers for
      atomic references, annotations, expressions, resolved intervals, source
      memberships, occurrences, and assigned roles.
- [x] Generalize `resolved_epoch_selection` through the new compiler while
      retaining its current compatibility behavior for maintained epoch-v2
      runs.
- [ ] Resolve exact canonical stimulus steps rather than relying only on
      `pre_event`, `training_event`, and `post_event` aliases.
- [ ] Migrate the selection materializer's array declarations to the shared
      physical `ArrayContract` authority; its current manifest is immutable
      and content-digested but remains a local declaration surface.
- [x] Make compilation deterministic under equivalent input mapping order.
- [x] Persist both requested and resolved selection representations.
- [x] Materialize immutable named selection runs with content manifests,
      direct/consolidated validation, and permanently selector-ineligible
      publication semantics.
- [x] Add a mixed `SOLID_BLACK -> CHASER_PRESENTATION -> SOLID_BLACK` fixture
      with distinct `black_before`, `chaser`, `black_after`, and `all_black`
      compositions.
- [x] Prove that the two black steps remain separate occurrences unless a
      saved expression explicitly pools them.
- [x] Add tests for boundaries, gaps, trims, empty results, overlap,
      de-duplication, set algebra, role preservation, and stale-source
      rejection.

## Phase 2: generic trajectory product

- [x] Define one versioned selector-ineligible trajectory schema over the
      `track_sample` row axis.
- [x] Materialize exact source track-sample indices/keys, acquisition frames,
      occurrence and selection membership, provider position, arena-mm
      position, and validity/reason evidence.
- [x] Keep trajectory position unsmoothed in this profile. Any smoothed trajectory is a
      separately identified derived method.
- [x] Preserve selected expected-frame count, source-row count, valid-position
      count, transform-valid count, in-grid count, and missing/invalid counts.
- [x] Bind declared position-provider, track-sample policy, selection, timing,
      geometry, transform, and software identities in the immutable trajectory
      content manifest, including `track_sample_policy_id`, source-row digest,
      and source-camera extent validity.
- [x] Bind those identities to actual canonical detection/keypoint source
      handles and published source manifests through the exact keyed adapter.
- [ ] Migrate trajectory arrays to the shared physical `ArrayContract`
      authority; occupancy-v2 has this declaration surface, but selection and
      trajectory do not yet share it.
- [x] Publish one detection-position and one keypoint-position trajectory for
      the same GoodBatBadBat selections without selecting a winner.
- [x] Validate source preservation and direct/consolidated metadata equality.

## Phase 3: provider-aware occupancy v2

- [x] Move scientific histogram computation out of
      `visualization/plot_detection_epoch_heatmaps.py` into a provider-neutral
      analysis module. Keep existing occupancy-v1 readers and runs unchanged.
- [x] Define a new provider-neutral run family and schema; do not label it
      `detection_occupancy` when it accepts other position providers.
- [x] Consume only an exact validated trajectory/track-sample input and exact
      resolved frame set.
- [x] Materialize, per occurrence and pooled selection:
      - raw bin counts;
      - valid in-grid sample count;
      - occupancy fraction of valid in-grid samples;
      - occupancy time in seconds under the bound timing policy;
      - expected selected frames;
      - provider-present and provider-valid counts;
      - transform-invalid and out-of-grid counts; and
      - exact x/y grid edges.
- [x] Exclude invalid or non-finite provider positions from spatial bins and
      the occupancy-fraction denominator, while reporting their coverage
      against all expected selected frames.
- [x] Do not interpolate missing positions, substitute another provider, clip
      finite out-of-grid points, or normalize each panel by its maximum.
- [x] Require count conservation:
      `sum(bin_counts) == valid_in_grid_sample_count`.
- [x] Require fraction conservation within floating tolerance when the
      denominator is nonzero; define all-zero/NaN behavior for an empty valid
      selection explicitly.
- [x] Bind every declared position, tracking, selection, geometry, transform,
      timing, grid, validity, and configuration source by exact path/digest in
      the occupancy-v2 source-binding manifest and cross-stage adapter.
- [x] Connect those exact bindings to canonical stimulus and detection/keypoint
      source adapters rather than fixture/in-memory provider inputs.
- [x] Publish detection and keypoint occupancy canaries for identical saved
      `black_before`, `chaser`, and `black_after` selections and the same
      millimetre grid.

## Phase 4: strict recording-level occupancy contrasts

- [x] Implement a narrow v1 contrast algebra with `difference` as the first
      operation: `treatment occupancy_fraction - baseline occupancy_fraction`.
- [x] Require named arms and preserve every contributing selection, role,
      occurrence, source step, and source occupancy manifest.
- [x] Require both arms to agree on provider run and estimator, track-sample
      policy, coordinate frame, transform, geometry, grid edges, denominator,
      normalization, recording, subject, and timing authority.
- [x] Reject ordinary scientific contrasts between detection and keypoint
      providers. Cross-provider sensitivity belongs in an explicitly labeled
      comparison product.
- [x] Publish `chaser - black_before` and `black_after - black_before` canary
      contrasts separately for detection and keypoint position.
- [x] Store result arrays and exact references; do not overwrite, average, or
      duplicate the immutable source occupancy runs.
- [x] Test rejection for mismatched provider, source run, grid, extent,
      coordinate frame, geometry, sample unit, denominator, overlap policy,
      timing, and stale lineage.
- [ ] Migrate contrast arrays to the shared physical `ArrayContract` authority;
      the current contrast materializer preserves exact source references but
      does not yet provide that shared declaration surface.

## Phase 5: cohort products

- [ ] Define one recording-level scalar/vector summary contract suitable for
      cohort concatenation without reopening plot artifacts.
- [ ] Freeze the cohort input manifest before campaign submission. Bind every
      recording, subject-track unit, provider, metric run, selection, contrast,
      and manifest digest.
- [ ] Treat `(recording_identity, subject_track_identity)` as the first
      recording-balanced experimental unit. A repeated or erroneous subject
      UUID must not silently collapse distinct recordings.
- [ ] Publish pooled-frame descriptive occupancy separately from the primary
      recording-balanced cohort product.
- [ ] Require a stated aggregation policy for unequal valid-frame coverage.
      Report coverage and do not silently reweight recordings by frame count.
- [ ] Keep detection and keypoint cohort products separate and comparable by
      exact provider label. Do not average providers.
- [ ] Export tidy per-recording/per-subject tables with metric values,
      validity, coverage, selection roles, provider identity, and source
      digests.
- [ ] Add campaign accounting for expected, succeeded, failed, blocked,
      missing, and stale recording products before cohort publication.

## Phase 6: plot recipes and Marimo inspection

- [ ] Define canonical plot-recipe JSON with independent scientific-analysis
      and render signatures.
- [ ] Implement generic trajectory, occupancy-panel, and occupancy-contrast
      recipes over exact immutable products.
- [ ] Keep labels, colormap, facets, panel order, and display scaling in the
      recipe rather than the scientific arrays.
- [ ] Generalize chaser markers into an optional annotation provider over the
      resolved frame set. Bind semantic behavior labels independently from
      display color.
- [ ] Recreate the current pre/chaser/post occupancy presentation as a recipe
      over the generic products rather than a second occupancy computation.
- [ ] Publish immutable plot-artifact attempts with recipe, PNG/spec/media
      hashes, source product digests, and consolidated metadata validation.
- [ ] Add recording-level Marimo discovery for exact selector-ineligible
      detection and keypoint offers, trajectories, occupancy panels, and
      contrasts.
- [ ] Show expected frames, valid coverage, out-of-grid count, provider ID,
      selection membership, grid policy, and selector/promotion status beside
      every plot.
- [ ] Prevent a read-only visualization or annotation failure from mutating a
      scientific product or production selector.

## Phase 7: provider sensitivity for motion and swim bouts

- [ ] Materialize or bind detection- and keypoint-position track successors
      with identical tracking policy where scientifically compatible.
- [ ] Compute speed, path length, and acceleration from each provider's own
      position track and exact timing authority.
- [ ] Run the same versioned bout-segmentation policy independently over each
      compatible provider-motion run.
- [ ] Compare coverage, speed and acceleration distributions, bout count,
      duration, path length, peak speed, inter-bout intervals, and
      pre/chaser/post contrasts without selecting a provider.
- [ ] Preserve algorithm identity separately from provider identity so a
      changed position source is not mistaken for a changed bout algorithm.
- [ ] Do not require or synthesize heading for these linear-motion analyses.
- [ ] Block rather than join keypoint body-frame rows onto detection-only
      observations. Heading-dependent comparisons remain deferred.
- [ ] Add recording- and camera-stratified checks for provider-dependent bias
      before any future default-promotion discussion.

## Canary acceptance

Use `2026-08-10T17-20-55Z_arena_2_goodbatbadbat` first because its four
position providers and immutable provider-comparison evidence already exist.
This checklist's canary uses only detection and keypoint position.

- [x] Preflight exact source manifests, selector-ineligible state, coordinate
      graph, geometry, transform, acquisition clock, stimulus run, and frame
      domain without writing.
- [x] Freeze the selection specs and millimetre-grid profile in the canary
      plan before materialization.
- [x] Prove detection and keypoint calculations use identical resolved frame
      sets and grid edges while retaining their own validity and row lineage.
- [ ] Review trajectories and occupancy panels against recording playback for
      representative pre-, chaser-, and post-period frames.
- [ ] Quantify coverage and position-provider sensitivity for occupancy,
      contrasts, speed, acceleration, and bouts.
- [ ] Check for systematic step/state-dependent disagreement; do not replace
      this with review of weak mask-model disagreements.
- [x] Record exact canary run IDs, manifests, Palette commit, commands, test
      results, and a timestamped decision.
- [x] Keep all canary runs selector-ineligible and leave production/default
      provider policy unchanged.

## Production and integration gates

- [x] Add focused pure/in-memory and materializer/adapter tests before real
      recording integration tests.
- [x] Run the new real-Zarr materializer and cross-stage tests outside the
      Codex sandbox according to `AGENTS.md`. Marimo checks remain open because
      Marimo discovery and plot recipes are not implemented in this slice.
- [x] Preserve and rerun the maintained provider-offer, binding, and
      resolved-epoch baseline. Broader occupancy-v1, GoodCopBadCop, chaser, and
      provider-canary CI coverage remains required before integration.
- [x] Validate immutable retries, manifest tampering, stale lineage, direct vs
      consolidated reads, and final consolidated visibility in the focused
      materializer/adapter suites.
- [ ] Update the storage-contract catalog, analysis-offer capability registry,
      recording-local discovery, and registry projection only after the
      scientific run contracts are stable.
- [ ] Keep parallel workers away from SQLite. If registry projection is added,
      use immutable receipts and one dependent serial finalizer.
- [ ] Pass every required CI check before integration, shared-checkout update,
      production selector activation, campaign publication, or any claim of
      merge readiness.
- [ ] Require a separate timestamped provider-promotion decision after
      multi-recording evidence. Successful materialization alone cannot make
      keypoints or detections the GoodBatBadBat default.

## Explicitly deferred

- Retraining, promoting, or scientifically adjudicating the current subject-
  mask position providers.
- Visual disagreement review for weak current mask predictions.
- Component-mask-triad or whole-body-mask production analytics campaigns.
- Mask-derived anatomical heading and full-body-mask heading.
- Gaze, turn-toward, circling, predicted-miss, and other heading-dependent
  escape analyses.
- Arbitrary user-defined formulas, weighted anatomical points, arbitrary
  contrast expressions, and implicit provider fallback.
- Production provider selection or an 84-recording provider campaign before
  the canary, required CI, and a separate promotion decision are complete.

## Completion evidence to record

- Exact branch and commit.
- Exact source and output run paths and manifest digests.
- Selection-expression and resolved-frame-set digests.
- Position provider, tracking, timing, geometry, transform, and grid policy
  identities.
- Expected, present, valid, transformed, in-grid, and missing sample counts.
- Count/fraction conservation results.
- Detection-versus-keypoint occupancy, contrast, motion, and bout sensitivity.
- Focused, adjacent, integration, Marimo, and required-CI commands/results.
- Confirmation that source recordings, immutable inputs, existing analysis
  runs, selectors, registry authority, and provider defaults were not changed.
