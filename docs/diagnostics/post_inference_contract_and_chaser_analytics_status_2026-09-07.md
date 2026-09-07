# Post-inference contracts and chaser analytics — September 7, 2026

## Scope and reading guide

This is a documentation-only evidence snapshot, requested after the successful
GoodBatBadBat core/chaser canary. It reconciles contracts, runtime code, existing
audits, and the Claude chaser analytics branches. It does not authorize or
implement a new admission interface, scientific recipe, merge, deployment,
historical repair, or production activation.

The [authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md)
continues to own cross-cutting admission/authority status. This report is linked
evidence, not another active queue. Scientific proposals remain in the
[chaser roadmap](chaser_analytics_roadmap_2026-08-10.md); historical strategy
results remain in the [strategy analysis](strategy_state_analysis_2026-09-01.md).

### Bottom line

Validated components satisfy a consumer's input contract **when they are the
specific suppliers that contract requires and their identities/relationships
also validate**. There is no additional universal requirement to obtain every
neural-network modality or human approval. An individually valid artifact from
a different recording, row surface, source generation, or permitted use is not
interchangeable with the requested supplier.

The exact full/single-recording GoodBatBadBat core/chaser chain is computation-
and presentation-complete as selector-ineligible evidence. That does not prove
that every future recording can run unattended after inference, nor that the
Claude cohort analyses ran against this new export. Those are separate claims.

## Audited versions and ownership

Observed on September 7, 2026. Source findings below refer to this fixed snapshot,
not an unspecified moving `main`.

| Surface | Exact version / state | Interpretation |
| --- | --- | --- |
| Published Palette `main` and clean review checkout | `fc6fea5fd937759f5ed21c79d13279b25c309e13`; `/tmp/palette-chaser-presentation-v2-deploy-20260907` | Integrated code baseline. GitHub main was rechecked during this inspection. |
| Documentation worktree | `/tmp/palette-chaser-contract-analytics-status-20260907`; branch `agent/palette/chaser-contract-analytics-status-20260907`; base `fc6fea5fd937759f5ed21c79d13279b25c309e13` | Parent Codex agent owns these documentation edits. Review agents were read-only. |
| Primary workstation checkout | `/home/delahantyj@hhmi.org/gitrepos/palette`; branch `agent/palette/refined-assignment-rebinding-gaze-20260831`; `73bee0d5194c662e3b7e535be3de92db7ef53f63` | Older checkout with modified `AGENTS.md`, untracked `.claude/`, and untracked audit documents. Preserved; not used as current-main implementation evidence. |
| Published agent-contracts `main` | `35efd44c50e73892dc4b3e20fd0d686dd4130d3c`; read-only clone `/tmp/palette-contracts-inspection-20260907` | Published external contract baseline; distinguish it from Palette's bundled contracts and runtime enforcement. |
| Existing local contracts checkout | `de08f3685c3b5d11b28651c03bad1a69101d8485`, branch `agent/palette/coordinate-contract-remediation-20260719` | Four modified contract files and three untracked contract files; preserved. Its local changes are not published-main evidence. |
| Parent clipped intake | `/tmp/palette-parent-clipped-intake-20260906`; `0653e75e5faad6b05a4ec715621cc6d9ee727f5d` | Clean branch, 11 commits ahead of audited Palette main and zero behind. Stronger branch-only work exists; do not recreate it or count it as integrated. |

Main's exact post-integration [CI run 34131195011](https://github.com/jmdelahanty/palette/actions/runs/34131195011)
completed successfully, with all 24 required jobs passing. This evidence applies
to `fc6fea5f`, not to the new documentation edits or other branches.

The parent-intake branch's exact head `0653e75e` also now has all 24 checks
successful in [CI run 34134814331](https://github.com/jmdelahanty/palette/actions/runs/34134814331).
This fresh query supersedes older handoff text calling its combined checks
unrun. It remains branch-only at this snapshot; a future integration would
need its own exact combined-commit CI. Synthetic intake receipts on that
branch are not live acquisition or current-main deployment evidence.

## What “input contract satisfied” means

A consumer needs a particular valid dependency closure, not an indiscriminate
all-modalities bundle. The necessary checks can include recording/source
identity, exact row/frame correspondence, coordinate system, validity, content
digests, source generation, and the authority or candidate-use rules declared
by that consumer. A sealed derivative can discharge upstream obligations that
it already binds; reopening unrelated source authorities is not an extra gate.

In particular, a body-frame supplier is sufficient for consumers reading only
anatomical origin, forward/left axes, heading, and validity. Such a consumer
does not acquire a landmark or pixel dependency merely because keypoints or
masks were used upstream. Conversely, a consumer that actually reads landmarks
or pixels must validate that concrete surface. `keypoint_authority=false`
prevents a derived body cache from impersonating canonical keypoints; it does
not negate its body-frame authority.

Canonical schema/lineage validation, accepted authority for a declared use,
scientific review, and successful execution are distinct claims. Human review
is required only where an applicable contract defines and validates that
reviewed claim. Missing optional eye/gaze/tail capability must not invalidate
an independent valid motion or chaser product.

### Bundles and subset validation are complementary

The user's architectural preference is not a choice between “bundles” and
“consumers validate what they need.” A receipt-bound bundle can be a small
manifest over independently governed products. It establishes one coherent
selection and the relationships among its members without copying scientific
arrays. A consumer then validates the exact subset required for its work.

For example, a bout-response consumer must not combine motion from one source
with bouts from an unrelated source merely because both artifacts separately
validate. A shared selection/relationship contract prevents that error. But an
occupancy consumer need not acquire an eye-angle dependency just because eyes
are another capability listed in the same recording bundle.

An all-required bundle is appropriate for a specifically declared deliverable
that needs every member. It is not a universal post-inference entry gate.
Missing optional members should remain explicitly unavailable and block only
dependent outputs; conflicting identities or a malformed bundle seal are
different defects and must fail the applicable composition validation.

The architectural direction supported by this inspection is **coherent sealed
composition plus consumer-specific requirements**, not a mandatory bundle of
all neural-network modalities. Current roster/subset receipts implement part
of this pattern; this statement does not claim every generic DAG consumer has
already migrated to it or create a new persisted bundle schema.

### Concrete consumer boundaries

| Consumer / interface | Existing supplier requirement | Coverage boundary |
| --- | --- | --- |
| Core roster consumption receipt | Explicit unique capability subset, mandatory cross-grain join authority, exactly resolved selected track, sealed capability digests | `core_authority_roster.py:574`; a real executable subset contract, not just terminology. |
| Maintained core-bound chaser relative-frame path | Cross-grain identity + selected kinematics + subject body frame | `core_paradigm_authority.py:158`; this is this consumer's minimum, not a minimum for every analysis. |
| Core/chaser composite | Child receipt lineage must match the selected core roster | `core_chaser_composite_bundle.py:324`; separately valid but competing motion/bout identities are not silently admitted. |
| Shared stimulus response | Exact track/bout lineage and source hashes, supplied independently | `execution.py:590`, `stimulus_response_candidate_execution.py:345`; not yet the shared core-roster consumption interface. |
| Generic activity/spatial export | Explicit track run and complete per-track bout mapping | `analytics_exports/activity_spatial_time_bins.py:557`; exact generic inputs do not imply core-bound provenance. |
| Bout kinematics / classifier command paths | Independent track/bout inputs; some command paths still pass literal track 0 | `execution.py:317` and `:561`; underlying `analysis/bout_kinematics.py:2861` still defaults the bout run to `latest`. These paths are outside the completed chaser migration. |

The source paths in this table are below `src/fisheye/analysis_workflows/`
unless another subtree is stated. The current static enforcement script
`scripts/check_paradigm_core_authority_access.py:12` covers the maintained
composable chaser planner and eight named descendants, not all those other
analysis consumers. The remaining migrations are already identified in
`docs/core_behavior_paradigm_composition_authority_design_2026-09-03.md:1160`.
Generic independent-source exports must not be relabeled as core-bound merely
because both use validated arrays.

### Published external contracts versus current runtime

The inspected `agent-contracts/palette-crimson` read/write documents still say
they were reconciled against Palette `9f514aca` on July 23. Published contract
main is stronger than the dirty older local proposal, but neither timestamp
establishes synchronization with the September 7 runtime. Specific differences:

| Contract claim | Runtime at `fc6fea5f` | Disposition |
| --- | --- | --- |
| Keypoint ordered labels and model pose-schema binding are implemented | `shared/keypoint_coordinate_publication.py` binds label authority, validates the pose schema, and checks the model artifact's pose binding | Supported. Do not restore the dirty local document's older “pending” claim. |
| `detect_bbox_read.md:70` lists bbox geometry as float64 only | `shared/observation_coordinate_publication.py:2860` validates versioned float64/cardinality-v1 and float32/cardinality-v2, with exact dtype/version agreement | External contract omits the current v2 profile. Record the version distinction; do not weaken or change the validator in a documentation pass. |
| `track_motion_read.md:84` specifies manifest/commit v1 only | `analysis/track_kinematics.py:11641` accepts bound v1 and v2 manifests and validates their different lineage grammar | External contract needs explicit v2 reconciliation; the runtime does not indiscriminately accept mixed-version fields. |
| Subject-shape description names `latest` | Strict runtime resolution also requires the matching complete selector and full source/manifest validation | Description is incomplete; this is stronger runtime enforcement, not permission to ignore the second selector. |
| Canonical Crimson manual keypoint/refined-detection submission is not implemented | Existing legacy edit/refinement code does not supply those specified immutable successor write paths | Still unavailable under those canonical write contracts; do not infer implementation from a legacy editor's existence. |

Runtime paths in this table are relative to `src/fisheye/`. No external contract
file or local contract proposal was changed by this audit. This list is a
bounded reconciliation, not a certification of every cross-repository contract.

## Execution coverage on current main

The generic analysis DAG and the specialized core/chaser workflow should not
be collapsed into one completion claim. Current main contains executable
planning/materialization machinery and strict, exact-source consumer paths;
repository-wide profile-neutral admission remains incomplete.

`profiles/core_behavior_v1.yaml:37` declares the core path from refined
keypoints through tracks, motion, and swim bouts, plus a mask → shape →
eye/tail branch. The planner computes the requested target's dependency closure
and returns `reuse`/`run`/`blocked` actions (`dag.py:111`). It does not yet
implement the queue's general producer-declaration / `admitted` /
`pending_receipt` model. Existing lifecycle profiles are production and
selector-ineligible canary (`execution_profiles.py:15`); neither is a general
modern-only origin/profile declaration. Target selection and capability
availability should not be confused with the fixed dependencies of an
individual node: current `bout_kinematics` explicitly requests eye gaze.

Static inspection identified these additional boundaries in the generic DAG.
They were **not exercised against an archive in this audit** and do not negate
the different, exact-source paths proven by the completed canary:

| Boundary | Code evidence | What remains unproven / incomplete |
| --- | --- | --- |
| Production tracking pin propagation | `execution.py:156` supplies `--tracking-run` for canary only; `runtime_verification.py:435` discards `dependency_runs` in motion verification | The production command/verification pair does not enforce the planner's exact tracking choice end to end. |
| Subject-shape canary command lifecycle | `execution.py:440` chooses supported-production or explicit legacy storage and Dask settings, without a candidate-profile branch | Generic canary generation does not encode the candidate lifecycle required by its verifier. |
| Eye-angle canary command lifecycle | `execution.py:373` uses the common Dask command without candidate storage/owner flags; `materializers/eye_angles.py:1322` requires serial execution for candidate storage | Direct candidate materializers exist, but this generated DAG command is not shown to compose them correctly. |
| Bout-kinematics canary policy | `execution.py:317` has no canary profile flag; `materializers/bout_kinematics.py:721` uses a publishing/promoting compute path | Valid ineligible eye inputs are expected to be rejected by the ordinary eligible-only eye reader before publication (`analysis/eye_angle_io.py:611`, `:1357`). This is an uncomposed canary path, not evidence of observed production mutation. |
| Payload verification coverage | `runtime_verification.py:550` lists tracks, motion, swim bouts, eyes, shape, and tail, but not bout kinematics | Generic metadata/lifecycle fallback is not equivalent to dedicated bout payload verification. |
| Modern bundle selection in generic canary discovery | `availability.py:1032` uses root subject-mask authority resolution for selector-eligible discovery; ineligible discovery uses exact child runs | The production modern-bundle route exists, but inactive-bundle canary dispatch is not equivalent. |
| CLI availability override | `utils/plan_analysis_workflow.py:38` records an artifact path without a concrete run name | `--available-stage STAGE=PATH` is not by itself an executable exact-run dependency binding. |

Paths in this section are relative to `src/fisheye/analysis_workflows/`, except
the explicitly named `utils/` path, which is relative to `src/fisheye/`.
The applicable next validation is a public planner → actual producer → strict
unpatched consumer boundary test for each supported lifecycle, not just a
command-string assertion. This report does not implement those tests or fixes.

The most concrete modern clipped boundary is already fail-closed:
`src/fisheye/cluster/clipped_inference.py:1976` rejects strict-v2 keypoints in
the `full` scope because downstream subject-mask/assignment consumers are not
yet wired to that profile. It does not silently substitute the ordinary shard
finalizer. The `keypoints` scope is canonical strict-v2 only; the full scope's
alternative is explicitly noncanonical compatibility, not modern-full success.

The autonomous transfer-triggered supervisor is still marked “to build” in
`docs/interface_and_execution_strategy.md:45`, with separate open transfer
watcher and failure-surfacing work at lines 85 and 89. This is broader than a
manually submitted, receipt-bound post-inference workflow. It must not be made
an extra prerequisite for reporting the already completed canary.

### Inference producer handoffs are not interchangeable

| Producer surface | What is implemented | Consequence for post-inference admission |
| --- | --- | --- |
| Whole-recording keypoint campaign | Terminal candidate validation, explicitly no selector activation or registry mutation (`cluster/keypoints/whole_recording.py:643`, `registry_finalize.py:108`) | Successful inference is not automatic production authority activation. An exact candidate can still be used where the consumer's candidate profile permits it; no human review gate is inferred from its status. |
| Clipped downstream campaign | Exact finalized-detection input; staged candidate products/bundles and deferred activation (`cluster/clipped_inference.py:4021`) | Validate the actual required terminal products and bindings, not merely the job exit state. |
| Subject-mask inference versus bundle recipe | Raw inference consumes exact crop/cache; the whole-recording finalization recipe also joins refined keypoints (`cluster/whole_recording_analysis.py:1040`) | A recipe-specific assignment dependency is not a universal prerequisite for raw mask inference or unrelated consumers. |
| Modern subject-mask publication | Recording bundle publisher defaults to inactive unless explicitly activated (`cluster/subject_masks/publish_recording_bundle.py:1200`) | Modern consumers requiring mask authority need the validated raw/refined/quality bundle envelope; a raw mask family selector is not an equivalent claim. |
| Existing mask batch planner | Default canonical output planning still has selector/last-child fallback helpers (`utils/run_subject_mask_batch_pipeline.py:259`, `:354`) | Modern-only behavior is not globally enforced across maintained entry points. Explicit-input safe paths do not erase this remaining fallback surface. |
| Legacy keypoint auto-approval | Opt-in algorithmic usable-rate rule and selector updates (`utils/run_keypoints_batch.py:412`, `:1479`) | Algorithmic approval is not evidence of human scientific review; retain the distinction from canonical validation and use-scoped activation. |

Paths here are relative to `src/fisheye/`. The existing batch-mask staged copy
and finalization sequence (`utils/run_subject_mask_batch_pipeline.py:902`)
also should not be treated as equivalent to the modern atomic bundle publisher:
it copies groups before later provenance/status/registry/consolidation work.
This static observation does not assert a failure occurred in the completed
canary, which used its separately recorded paths and state checks.

Recording-only post-inference execution and transfer-parent orchestration are
different scopes. Current importers already reject legacy source-analysis
intake through the scoped enforcement work; the stronger parent branch adds
transfer-v2/clipped intake and synthetic recovery evidence. Neither the missing
watcher nor an unrelated optional modality should be invented as an input
requirement for a consumer that already has its admitted recording suppliers.

## Exact core/chaser canary evidence

Recording: `2026-08-10T17-20-55Z_arena_2_goodbatbadbat` (full/single).

For the paths below, `OP` denotes the exact operation directory
`/groups/johnson/johnsonlab/jeremy/operations/core_chaser_positive_canary_20260906_1bf9d919`.
It is an evidence-path abbreviation, not a mutable selector.

| Claim | Exact evidence | Bounded conclusion |
| --- | --- | --- |
| Numerical export is readable through the supported validator | `OP/evidence/validated_core_chaser_reader_v4.json`; record SHA-256 `0a1bbdd7808f27fb742792e396232935933f6a2329b6ab0d86da8cb01e78401e`; full validation, 30 tables | `validated_core_behavior_chaser_v1` export computation is complete for this recording. No new cohort or all-profile claim. |
| Immutable numerical publication | `OP/publication_v4`; run `goodbatbadbat-core-chaser-positive-canary-20260907-19fca24a-v1`; manifest record SHA-256 `61b417d019185cf20b5100b943aefac626a941c70828db993c72ca9c4dc2f2d2` | Numerical products retain their producing identities; presentation did not recompute them. |
| Presentation task completed | `OP/chaser/composable_chaser_successors_presentation_v2_20260907_fc6fea5f/receipts/task_0001/receipt.json`; payload SHA-256 `f97aa05239bd1596c8c1829d7d62c3149871525ec07bd57ee77594a827ade04f` | 34 recorded stages; `complete_selector_ineligible`; exact commit `fc6fea5fd937759f5ed21c79d13279b25c309e13`. |
| Task/deployment binding | `OP/chaser/cohort_task_presentation_v6.json`; task SHA-256 `c91cfef1453cb06285f0e884c522ddab2ae8608ede16993cdd8d9e3b05fa7e20`; deployment `/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/core-chaser-positive-canary-presentation-v2-20260907-fc6fea5f` | Fresh commit-pinned deployment; no moving shared-checkout dependency. |
| Required plots validated | Six plot receipts; 17 PDFs and 17 PNGs, all output hashes verified during the canary; all 17 one-page PDFs rendered with Poppler and individually inspected | Computation and presentation complete for this chosen workflow. Five versioned bundles newly rendered; body-alignment plot reused unchanged. |
| Existing state preserved | `OP/evidence/chaser_state_comparison_presentation_v2.json`; record SHA-256 `027d0d701359e88693fd9227df19bd2642c80e3fb018f3ddc3be19ae9f5ba75a`; all eight checks true | Root/selector attrs, 16 reused scientific groups, and registry state unchanged; registry physical files also compared equal in the canary QA. |
| Palette registry gate passed | `OP/evidence/registry_integrity_after_presentation_v2.json` | Palette `scripts/py` SQLite runtime `3.52.0`: integrity `ok`, zero foreign-key issues. No system SQLite substitution. |

The figure set is dashboard (1), detailed epoch/bout views (9), near-field
views (4), spatial occupancy (2), and body alignment (1). Readable display labels
replace opaque IDs only in presentation; receipts retain exact provider/chaser
identities and explicit keypoint/detection roles. Presentation recipes are
dashboard v3, detailed v6, spatial v6, near-field v2; body alignment is unchanged.

No selector, production authority, registry registration, or scientific
acceptance was activated. Historical outputs were retained. The operation's
earlier failed attempts remain failure evidence and are not relabeled as
successful merely because a later fresh version completed.

## Claude chaser extensions: included in this account, not yet integrated

The four inspected Claude worktrees are clean, their local heads match their
remote branches, and none of their unique commits is ancestor- or patch-
equivalent to audited main. Exact-head GitHub inspection found **no open or
closed PR for these branch names and no check runs/status contexts**. These
are pushed branches, not currently reviewed/CI-green PRs.

All worktree paths below are under
`/home/delahantyj@hhmi.org/gitrepos/palette/.claude/worktrees/`; all branches
start with `agent/palette/`. Existing Claude work and ownership were preserved.

| Extension | Worktree / branch | Exact head | What exists |
| --- | --- | --- | --- |
| Strategy states / Sankey-related transitions | `agent-a4952768e724061c4`; `validated-behavior-strategy-states-20260901` | `abba9d3f2139bafb510aa41f11a08726b2e4b45d` | New numerical module, CLI, and tests; pooled pre/post clustering, posterior-weighted transitions, responder relationships, held-out recording decoder, and time-resolved analyses. |
| Chaser-relative twin nulls | `agent-a7c58d1a0842b2c80`; `chaser-relative-twin-nulls-20260901` | `bbc109237ce8275aebecf796fd656b1181af4ce6` | New rotated virtual-twin numerical module, CLI, and tests; compares actual chaser-relative observations against rotated controls and writes sidecar outputs. |
| Role contrasts | `agent-a9b9aedb33aefcd10`; `validated-behavior-role-contrasts-20260901` | `b6e97ed3f949e1b1653b640118853b91f445f3fb` (also contains `983d3771`) | New module/CLI/tests for role/epoch contrasts, distance bins, bout associations, quantile shape, and censoring-aware inter-bout intervals (IBIs). |
| Escape/freeze signal provenance | `agent-adfb9d0f3a11a1824`; `escape-freeze-signal-provenance-20260901` | `305fb4b1cddaeeb178de73216a8b95f0791abd5c` | Changes existing producer/planner/export code to retain signal/threshold provenance and gate raw speed; adds tests. This is not just a render-only extension. |

The first three modules/CLIs are absent from audited main. The escape/freeze
branch touches existing code that has since changed; its historical diff needs
reconciliation with current owners/contracts, not blind replay. None of these
branches has a maintained core/chaser DAG registration, installed export
profile, cohort-publisher integration, or selector authority established by
this inspection. Branch-local exploratory manifests are not automatically
equivalent to the core/chaser publication contract.

### Connection to the new validated core/chaser export

The new `validated_core_behavior_chaser_v1` export has 30 tables, including
`kinematics_samples` and `canonical_swim_bouts`. It intentionally does **not**
duplicate the historical `provider_motion_samples` or `provider_swim_bouts`
tables as competing core facts. Similar subject matter is not table-contract
equivalence.

The following is static input compatibility, **not a successful execution
claim**. No extra-analysis CLI was run against the new canary in this pass.

| Extension | Existing input connection | Remaining compatibility/evidence work |
| --- | --- | --- |
| Twin nulls | Required `chaser_relative_samples`, `semantic_epochs`, and `radial_near_field_summary` tables exist in the core export | Verify the current reader/profile against the actual module, admitted-member filtering, and exact denominator/geometry/null-policy parity. Available tables alone do not prove accepted publication integration. |
| Role contrasts | Base, distance-bin, bout-association, and quantile analyses have corresponding core tables | The IBI extension still reads `provider_motion_samples` and its `linear_sample_valid` contract. It cannot be treated as a direct core-compatible IBI consumer. |
| Strategy states | Reads validated-behavior tables plus explicit twin-excess and IBI-cell Parquet sidecars | `group_statistics/validated_behavior_strategy_states.py:1223` still reads provider-motion FPS. An explicit core input adapter/profile is needed; renamed tables or guessed validity/FPS are not an acceptable bridge. |
| Escape/freeze provenance | Extends `trial_escape_freeze_summaries` with `speed_level`, freeze window/threshold, escape threshold, and provenance-status fields | Those branch fields are absent from the current canary schema. Adoption needs a deliberate version/compatibility decision and fresh output; do not rewrite the sealed historical export. |

Strategy sidecar loaders validate required columns, filtering, and row
uniqueness (`validated_behavior_strategy_states.py:200`, `:237`), while its CLI
records input file hashes. That is useful explicit provenance but does not
by itself prove the sidecars were derived from the exact same admitted cohort
and source generation as the selected export. Preserve those checks and add
the applicable relationship binding if this becomes a maintained consumer.

### What exists for the Sankey diagrams

The strategy pipeline is **cohort-level analysis over exported datasets**, not
video inference or another recording-local Zarr producer. Its existing stages
are separable (`group_statistics/validated_behavior_strategy_states.py`):

1. Join export tables and explicit twin/IBI sidecars into one row per recording
   × pre/post epoch. Seven features describe twin-corrected near-zone fraction
   and distance, wall distance, occupancy entropy, bout rate, mean absolute
   bout heading change, and long-IBI fraction. Only the IBI feature is allowed
   median imputation, with an explicit flag (`:38`, `:51`).
2. Standardize pooled pre/post features, apply three-component PCA, and fit
   full-covariance Gaussian mixtures with BIC selecting one to six clusters.
   Retain hard assignments and posterior probabilities; numeric labels are
   ordered by cluster size, not fixed behavioral meaning (`:385`).
3. Compare each recording's pre/post assignments, relate them to measured
   trial responder classes, and assess transition/stability statistics.
4. Separately decode pre versus post from time-window bout/IBI features using
   held-out recordings, and decompose pre/post feature displacement. This
   decoder is not the same model as the strategy clustering.

Outputs include feature rows, cluster assignments/means, BIC results,
transitions, decoder scores, and direction decomposition, plus input/config/
output-hash provenance. The writer does not export a fitted PCA/GMM inference
artifact for prospective classification (`:1435`). Arena strata are currently
parsed from recording names (`:140`), and rows are recording-keyed: those are
explicit modernization topics, not proof of registry-bound subject identity.

The strategy module computes per-recording transitions and a posterior flow
matrix (`validated_behavior_strategy_states.py:571`, `:626`), persists
`strategy_transitions.parquet`, and records the flow matrix in its manifest
(`:1459`, `:1501`). These are the numerical inputs for a Sankey view.

The historical strategy report describes rendered Sankey results and scratch
figures. The inspected committed strategy module/CLI is a numerical pipeline,
not a maintained Sankey renderer or receipt-bound static-presentation bundle.
Do not count those historical scratch figures among the 17 validated canary
PDFs. A maintained Sankey presentation would need to bind its exact transition
output, model/seed/feature recipe, cohort identity, responder labels, and output
hashes, with separate computation and presentation status.

Historical output location `/tmp/strategy-states-v001` and the report's
`strategy_clustering/` scratch reference are recorded leads, not durable
receipt-verified evidence in this inspection. No claim is made that those
scratch artifacts remain available or reproducible at current main.

There **is** a different, already integrated Sankey implementation:
`apps/marimo/components/training_response.py:120`, mounted by
`apps/marimo/baseline_strategy_explorer.py:836`. It joins complete baseline
`primary_strategy` rows to complete whole-training `primary_training_profile`
rows and counts matched recording-level focal-fish sessions. It is a two-tier
descriptive correspondence, not the Claude branch's pre/post soft-cluster and
responder analysis. Its training-response reader enforces a manifest/Arrow
contract (`src/fisheye/training_response/query.py:314`).

The existing [baseline strategy analytics](../baseline_behavior_strategy_analytics.md)
also operate downstream of immutable exports, but consume the baseline summary,
time-bin, and optional kinematic-sample contracts. They are not aliases for the
new core export or the Claude GMM recipe. Review this existing owner when
modernizing Sankey presentation so shared rendering mechanics can be reused
without conflating category meanings, hard counts, and posterior-weighted
flows. This audit inspected its code, not a live Marimo rendering or a fresh
training-response canary.

### Scientific interpretation stays scoped

The September 1 phase-B cohort analysis concerned 80 admitted recordings,
not the new single-recording core canary. Its reported unit was recording,
without subject/batch clustering adjustment. Preserve the September 6
interpretation correction in the strategy report: causal claims, “innate
avoidance,” and “passive-coping state” are hypotheses; the first three post
minutes are a candidate window, not an established standard readout.

Other recorded constraints also remain material: corner-fair features; a pooled
pre/post model rather than incomparable separately fitted labels; the training
interval represented by measured responder class rather than clustered as an
equivalent exposure; held-out recording validation rather than window-level
leakage; explicit censoring; and the documented side-confounding of quadrant
role contrasts in this cohort. This report does not rerun, endorse causally,
or convert those exploratory findings into scientific acceptance.

The historical suggested sequence was escape/freeze provenance → twin nulls →
role contrasts → strategy states. Strategy consumes sidecar data rather than
importing code from the other branches, so runtime data dependencies and git
integration prerequisites are not the same. Reconcile that sequence against
current schemas before any authorized implementation; avoid merging changes
solely to reproduce an old branch order.

## Existing audits: current disposition and remaining decisions

The active queue's `ADM-001`–`ADM-003`, broader producer/resolver work, and
`ACC-001`–`ACC-003` are not completed by this documentation or by one canary.
Its governing `admitted`/`pending_receipt` prose describes the target contract;
it is not proof that the generic planner implements those states today.
The current strict-v2 full-clipped refusal is stronger than an older audit
describing no guard at all, while still short of complete downstream wiring.

The [core/paradigm composition design](../core_behavior_paradigm_composition_authority_design_2026-09-03.md)
and [exact chaser checklist](chaser_exact_full_gap_closure_implementation_checklist_2026-08-30.md)
predate the final canary. Their earlier unchecked canary/presentation statements
must be read with the exact evidence above; unrelated unchecked migrations or
scientific proposals are not thereby complete.

No new product/science decision is needed to state the facts in this report.
Before implementing the next slice, clarify the requested outputs/profile,
candidate versus production lifecycle, and which additional cohort analyses
should become maintained deliverables. Existing consumer contracts—not a new
universal all-inference bundle—determine their dependencies. Decisions about
new statistical endpoints, censoring, cohort identity, or persisted schema
changes need explicit reconciliation; ordinary sealed-supplier validation
does not require an invented human-review step.

### User-directed modernization: one component at a time

The user wants to modernize and eventually merge selected extensions, with an
item-by-item examination of what to keep. This is not blanket approval to merge
the four historical branches. Start with the purpose of individual strategy
classification and its Sankey presentation, then review the required twin-null,
IBI, and escape/freeze dependencies on their own merits.

For each item, record a proposed **keep / change / defer** disposition covering
the scientific question, input contract, computation, output identity,
presentation, and validation. Resolve meaningful scientific choices with the
user before implementation. Only then modernize that bounded component in an
owned worktree, preserve/version its contracts explicitly, run its boundary
tests and applicable canary, obtain required CI, and integrate with fresh
combined-commit CI. Production activation is a separate claim and scope.

For strategy states, the first unresolved choice is what a label should mean:
a cohort-relative exploratory grouping, a comparable descriptive behavior
label, or a classifier intended to assign new recordings to a fixed model.
The existing pooled GMM/soft-transition analysis is evidence for the first
kind; it does not establish stable biological types or a validated prospective
classifier. No keep/change/defer scientific disposition has yet been approved.

## Documentation handoff

This pass changes documentation only. No executable interface, serialization,
scientific parameter, persisted status, source data, or authority changed. No
dependencies were installed and no new numerical analysis or cluster job was
run for this inspection. Existing canary receipts were read without mutation.

Local documentation validation: `git diff --check` passed;
`scripts/py scripts/check_contract_freshness.py --json` passed (48 managed
contracts, zero issues); a read-only check of all six changed/new Markdown
files found no trailing whitespace or missing relative-link targets/anchors
(24 links).
No pytest run was needed for this source-unchanged documentation pass; full
required CI remains mandatory before any later integration.

The user authorized a documentation-only commit and remote branch push for
preservation, without opening a PR. No integration, deployment, or activation
is included. Required CI is unrun; the current-main CI result above does not
make this documentation branch merge-ready. Historical branch-local test
counts are evidence at their stated versions, not present-day integration
permission. At this snapshot, `.github/workflows/ci.yml` triggers on pull
requests and pushes to `main`; a feature-branch push with no PR does not trigger
that workflow. Opening or updating a PR would require the full CI gate.

For each of the four Claude heads, all required checks are unrun:
`import boundaries`, `generated artifacts`, `file-size ratchet`,
`zarr open metadata modes`, `observed metadata literals`,
`active contract freshness`, `package and collection`,
`non-gpu tests (shard 0)` through `non-gpu tests (shard 15)`, and `ci-required`.
The same full required-CI suite is unrun for this documentation change.
Local static documentation checks, reported separately, are not a
substitute for that integration gate.
