# Clipped Eye-Assignment Authority Workload Failure

**Date:** 2026-08-25

**Scope:** read-only inspection of the four Sleepyfish clipped-recording
operations, their selected keypoint artifacts, the active workflow commands,
and the producer/consumer boundary in Palette, followed by fifteen parallel
Luna xhigh audits of producers, resolvers, tests, reporting, and real operation
evidence. No recording, selector, publication, or registry state was changed
for this audit.

## Decision

The required policy is fail closed:

> A production workload may enter planning only when every required input
> authority resolves through a supported, full-strength profile and the plan
> seals the exact resolved artifact and proof. Completion, selector
> eligibility, a familiar path, or an authority-sounding run name is not
> sufficient.

If a boundary cannot be satisfied, planning must return a typed admission
failure before expensive payload validation, scratch creation, cluster
submission, or output publication. A workflow must not try another profile,
infer authority from arrays, or fall back to a legacy run.

Before repairing the eye-angle consumer, Palette must inventory the producers
used by active processing workloads. Otherwise a stricter consumer can expose
the next noncanonical producer only after another expensive run.

## Parallel audit verdict

The strict machinery is substantially present, but it is not composed into one
production path. This is not a missing activation flag. The active command
generators, shared resolvers, planner admission, lifecycle profiles, and
boundary tests disagree about which artifacts are production-consumable.

The audit found these highest-priority facts:

| Boundary | Implemented machinery | Active-production finding |
|---|---|---|
| Detection | Native artifact-first detection, recording-level canonical publication, registered geometry gate, quality/refinement receipts | The real Sleepyfish detection operation used this strong path. However, the default full clipped CLI still falls back to the legacy detection fragment when the registered gate is `off`. |
| Crop and pixels | Sealed geometry-only crop-v2, signed hybrid provider, materialized canonical crop, strict per-profile validators | The shared position resolver still lacks the sealed crop-v2 branch; the process-global geometry-only adapter is still live; active clipped keypoint/mask production still uses proxy crops. |
| Keypoints | Whole-recording and clipped strict-v2 candidate chains, bundle activation evidence, coordinate-successor machinery | Both active clipped planners still terminate with `finalize_keypoint_shards`. That writer produces an ordinary selector-visible run, not a strict coordinate-v2 authority. Current shard commands also omit the required `legacy_noncanonical` mode, and the following legacy refinement command is no longer a production path. |
| Subject masks | Strong receipt-composed recording bundle publisher, dense refined-mask authority, component review gates | Assignment-keypoint ingress can still select by path/completion, fall back to ordinary `keypoints_runs`, and preserve legacy assignments into an otherwise well-sealed bundle. The bundle publisher preserves evidence; it does not upgrade it. |
| Subject shape | Historical v4 and recording-bundle v5 publication machinery with strict loaders | Maintained v4 is normal selector-visible production. Bundle v5 is currently candidate-only and selector-ineligible. The real v003 Sleepyfish attempt failed at this authorization boundary before eye angles ran. |
| Eye angles | Strong runtime row/time/keypoint proof, staged integrity receipts, atomic publication, strict output validation | The bundle assignment resolver is absent. Staging permits a null assignment pointer, while the later reader assumes the historical `.context` object shape. Planner admission does not resolve this indirect authority. |
| Core workflow | Correct scientific DAG and strong stage-local materializers | Availability/reuse is based on run names, completion, and selector eligibility rather than producer profile and authority proof. Forced availability can bypass discovery entirely. Under the new policy, every executable node is unknown/blocking until it receives typed admission. |
| Downstream and reporting | Strict source checks exist in many writers and exports | Bout-kinematics storage promotion can start from a completion/layout-only source. Reporting, registry readiness, visualization discovery, reuse, and campaign handoff contain additional path/`latest`/completion-only decisions. |
| Tests and CI | Many strong isolated producer and reader tests | No qualifying real production writer -> unpatched authority resolver/consumer test covers the keypoint or bundle-to-eye boundary. Several suites monkeypatch the exact boundary under test. CI has no dedicated boundary-test gate. |

The real Sleepyfish corpus reinforces the distinction between bytes and
authority. Detection, refined detection, crop, masks, keypoints, tracking,
track kinematics, swim bouts, and subject-shape payloads exist. Several are
complete and selector eligible. No successful downstream registry-finalization
receipt exists for the latest authority-recovery attempt, so the evidence does
not establish one canonically published downstream chain.

## Exact failure chain

### 1. The recovery workload selected the ordinary shard finalizer

The operation manifest at:

```text
/groups/johnson/johnsonlab/jeremy/operations/sleepyfish_2026_08_06_keypoint_geometry_authority_recovery_20260824_v001/recovery_manifest.json
```

contains four commands of this shape:

```text
scripts/py -m fisheye.utils.finalize_keypoint_shards \
  <analysis.zarr> \
  --target-crop-run crop_v2_... \
  --output-run keypoints_geometry_authority_... \
  --shard-run <clip-000000> ... --shard-run <clip-000054>
```

It did not invoke the clipped canonical-v2 finalization/publication chain.
`finalize_keypoint_shards()` describes its output as a normal keypoint run,
writes `stage_selector_eligible=True`, marks it complete, and advances the
ordinary keypoint selector. It validates row alignment and applies the crop
rebase, but it does not publish the canonical coordinate manifest, coordinate
context, coordinate derivation, or successor proof required by downstream
canonical consumers.

This is the first missed workload boundary: command generation selected a
scientifically useful merger, but no producer-admission gate checked whether
that command produced the authority profile required by the downstream DAG.

### 2. All four selected artifacts confirm the mismatch

Metadata-file inspection of the four selected
`keypoints_geometry_authority_20260824_v001_sleepyfish_cam*` runs produced the
same result:

| Property | cam2010093 | cam2010094 | cam2010095 | cam2010096 |
|---|---:|---:|---:|---:|
| `palette_run_completion_status` | complete | complete | complete | complete |
| `stage_selector_eligible` | true | true | true | true |
| `source_crop_mapping_mode` | identity_rebase | identity_rebase | identity_rebase | identity_rebase |
| `source_keypoint_shard_count` | 55 | 55 | 55 | 55 |
| `run_manifest` present | no | no | no | no |
| `keypoint_coordinate_context` present | no | no | no | no |
| `keypoint_coordinate_derivation` present | no | no | no | no |
| `coordinate_contract` present | no | no | no | no |
| `production_candidate` present | no | no | no | no |

The name `geometry_authority` did not make these artifacts canonical. The
ordinary lifecycle fields proved that the merger completed successfully under
its own contract; they did not prove the distinct coordinate-authority claim.

### 3. Workflow planning propagated the run without admitting its authority

The downstream operation pinned each artifact as:

```text
--stage-run refined_keypoints=keypoints_geometry_authority_...
```

The workflow then constructed subject shape and eye angles. The eye-angle
command in `analysis_workflows/execution.py` receives the resolved
`--subject-shape-run`, but it does not receive an exact canonical
`--keypoint-run` or a resolved assignment-authority digest.

The DAG edge `subject_shape -> eye_angles` is correct: keypoint authority is
supposed to travel through the subject-shape assignment proof. The defect is
that workflow admission checked stage availability, completion, and selection,
but did not resolve this full chain:

```text
subject-shape publication
  -> assignment-keypoint collection
  -> exact canonical keypoint publication or canonical successor
  -> ordered row and acquisition-frame equivalence
```

This is the second missed workload boundary. The planner rendered the
eye-angle node as part of the runnable downstream plan without proving that its
indirect dependency was consumable. The real v003 execution stopped at
subject-shape authorization before this node ran.

### 4. Staging preserved the omission

`shared/eye_geometry_source.py::_build_staged_subject_shape_authority()` looks
for assignment authority through `publication.source.context`. Historical
refined-mask-backed publications expose that shape. Recording-bundle-backed
subject-shape publications expose their authority through the bundle binding
instead.

The staging code uses nested `getattr(..., None)` and permits the resulting
`assignment_keypoint_authority` pointer to be null. Thus bundle-backed geometry
can be staged without an assignment-keypoint proof even though normal eye-angle
analysis requires one.

The staging schema therefore recorded absence rather than rejecting an
unsatisfied required boundary.

### 5. The observed v003 workload stopped at subject-shape authorization

The latest real operation inspected was:

```text
/groups/johnson/johnsonlab/jeremy/operations/sleepyfish_2026_08_06_authority_downstream_recovery_20260824_v003
```

All four core-behavior jobs failed at the `subject_shape` node with:

```text
Recording-bundle subject shape is initially authorized only as the
selector-ineligible access-aware storage candidate.
```

This is a correct fail-closed stage-local decision. The problem is that the
generic workflow planner admitted and submitted the node without knowing that
the selected producer/source combination was candidate-only. Eye angles and
dependent nodes remained pending in this attempt. The planned serial registry
finalization record was not written.

Older `subject_shape_sleepyfish_2026_08_06_core_behavior_v007_cam*` runs are
physically complete and selector eligible, but no matching operation-status or
registry-finalization receipt was found. Their presence cannot be used as an
authority oracle.

### 6. The eye-angle `.context` failure is the next reachable defect

The canonical eye-angle reader later assumes the historical profile directly:

```text
materialize_eye_angles
  -> build_eye_angle_materialization_plan
  -> _resolve_source_plan
  -> _resolve_eye_angle_inputs
  -> _resolve_canonical_eye_keypoints
  -> publication.source.context
```

At `analysis/eye_angle_analysis.py:3248`, a bundle source reaches an
unconditional `.context` dereference and raises:

```text
AttributeError: 'BoundSubjectShapeBundleSource' object has no attribute 'context'
```

The current v003 workload did not reach this line; subject-shape authorization
failed first. The raw exception is nevertheless reachable if a bundle-backed
subject-shape publication enters eye-angle resolution. It would occur before
eye-angle scratch staging and publication. Changing only that line would leave
the earlier producer, assignment, lifecycle, and workflow-admission defects
intact.

## Active-chain inventory synthesis

This table combines real-operation evidence with the parallel source audit.

| Active chain surface | Observed producer/profile | Current disposition |
|---|---|---|
| Geometry-only crop | Sealed crop-v2 geometry profile with signed hybrid pixel authority | Producer and strict profile reader exist, but the shared position resolver branch and real tracking boundary test do not. The monkeypatch adapter remains active. Block shared downstream consumption. |
| Clipped keypoint shard merge | `fisheye.utils.finalize_keypoint_shards`; ordinary selector-visible keypoint run | Valid legacy merger output, but **not admitted** as a canonical coordinate producer. Block for canonical downstream use. |
| Four selected recovered keypoint runs | Complete, eligible, identity-rebased, 55 shards; no canonical coordinate records; `source_detect_run=unknown` | The existing coordinate-successor publisher cannot consume these ordinary runs. Require a new immutable migration bridge and an explicit production-consumption lifecycle. Do not mutate or reinterpret in place. |
| Refined subject-mask assignment | Bundle assignment collection retains exact worker/keypoint references | Upstream selection is legacy-capable and per-worker collections can name different runs. Require one normalized assignment resolver and republish/rebind immutable dependents after migration. |
| Bundle-backed subject shape | Bundle v5 is a strong, distinct profile but currently candidate-only and selector-ineligible | The v003 workload correctly rejected normal publication. Eye-angle use remains blocked until both lifecycle admission and assignment-keypoint authority are implemented. |
| Eye-angle materializer | Canonical consumer and atomic publisher after successful input resolution | Blocked before rerun. It must consume the shared resolver result and seal that proof into staging and output receipts. |
| Track, motion, swim-bout, subject-shape, and visualization artifacts on the four recordings | Multiple physical runs are complete and some are selector eligible | No successful latest-operation or registry-finalization receipt proves one coherent downstream chain. Quarantine from authority/reuse decisions until admitted individually. |
| Bout-kinematics storage candidate/promotion | Atomic candidate copy and promotion machinery | Source planning checks completion/layout only and promotion does not re-prove source admission. Block promotion until fixed. |
| Reporting, registry readiness, visualization, and handoff | Multiple discovery and publication surfaces | Several paths still infer availability/readiness from group presence, names, `latest`, or completion. They must consume the same typed admission result and digest. |

## Producer inventory contract

The parallel audit established the initial evidence baseline. The first code
deliverable remains a generated, reviewable inventory/CI ratchet covering every
producer reachable from active production workflow catalogs, operation
builders, submission scripts, and maintained CLIs. For the clipped core-
behavior chain, it must cover at least:

```text
detect -> quality -> refinement/finalization
       -> crop geometry and pixel provider
       -> keypoints and subject masks
       -> refined subject masks and assignment collection
       -> tracks and track kinematics
       -> subject shape
       -> eye angles, tail kinematics, swim bouts, and bout kinematics
       -> downstream immutable exports
```

Each inventory row must record:

| Field | Required evidence |
|---|---|
| Workload entry point | Exact planner, operation builder, CLI/module, and command template. |
| Producer identity | Exact writer/finalizer/publisher symbol and deployment commit binding. |
| Output profile | Parent/path, schema/profile ID and version, lifecycle state, selector policy, and canonical claim. |
| Authority evidence | Required manifests, coordinate/temporal records, assignment or successor proof, and their digest algorithms. |
| Consumer admission | Shared resolver branch that accepts it, with no fallback or adapter. |
| Boundary coverage | Real writer -> unpatched resolver/consumer test and required CI check. |
| Disposition | Canonical producer, supported profile, compatible legacy producer requiring successor, diagnostic-only, tombstoned, or unknown/blocking. |

Unknown is a blocking result. The inventory must inspect code-generated command
templates and real operation plans; module names, output names, status, and
schema markers cannot stand in for producer proof.

## Updated implementation order

### Phase 0 — Freeze unsafe admission

- Do not rerun eye angles or dependent eye-trace/bout products from the four
  current subject-shape assignments.
- Do not mutate, rename, or promote the four recovered keypoint runs.
- Treat the current artifacts as valid legacy evidence, not canonical inputs.

**Gate:** no new work plan can label an input canonical solely from completion,
eligibility, path, or run name.

### Phase 1 — Ratify and automate the active-producer inventory

- Preserve this parallel audit as the human-reviewed baseline.
- Generate the maintained entry-point/producer/profile matrix from workflow
  catalogs, command builders, CLIs, submitters, and recovery planners.
- Classify every produced profile against its full-strength loader/resolver and
  real boundary test.
- Record unknown, adapter-only, compatibility-default, and legacy-only
  producers as blocking.
- Add a CI ratchet so a new production command generator cannot appear without
  a profile, resolver branch, and boundary-test disposition.

**Gate:** every node required by the requested workload has one named admitted
producer profile or a named migration. No unknown row remains, and the
generated inventory agrees with this audited baseline.

### Phase 2 — Add producer admission to workflow planning

- Add a profile-neutral admission result for each required stage input.
- Require exact producer/profile/manifest or successor evidence before a node
  is runnable.
- Persist proof mode, exact run/path, authority digest, row count, recording,
  camera, temporal identity, selector snapshot, and consolidated metadata
  generation in the dry-run plan and handoff.
- Make plan reuse depend on those identities and digests.
- Remove forced availability from production mode or require it to carry an
  independently validated admission receipt.
- Revalidate admitted inputs immediately before worker launch.

**Gate:** the current four keypoint runs fail planning with a typed
`unsupported_canonical_keypoint_authority` result before payload scans or job
submission.

### Phase 3 — Wire maintained producers into active command generation

- Make the native artifact-first recording-level detection path the explicit
  production default; isolate the legacy detection fragment behind a named
  compatibility mode.
- Add the sealed geometry-only crop-v2 branch to the shared position resolver
  and retire the process-global adapter only after keypoint, mask, and tracking
  consumers migrate.
- Wire the existing clipped strict-v2 keypoint fragment into the active clipped
  planner. Do not treat `finalize_keypoint_shards` as its terminal authority.
- Pass the required `legacy_noncanonical` mode to inference shards and stop the
  stale legacy refined-keypoint command from masquerading as production.
- Make subject-mask assignment selection call the canonical assignment
  resolver rather than path/completion or latest/sorted fallback.
- Close the bout-kinematics source-admission and promotion gap.

**Gate:** generated production plans contain only admitted producer commands;
all compatibility commands are explicitly labeled and cannot satisfy a
canonical dependency.

### Phase 4 — Define and publish a migration profile for the four keypoint runs

- Do not reuse the existing keypoint coordinate-successor publisher: it
  requires an already-sealed selector-ineligible production candidate and
  cannot consume these ordinary selector-visible shard-finalizer runs.
- Build a CPU-only legacy-clipped-keypoint migration bridge; inference must not
  be rerun when scientific equivalence can be proved.
- Pin source metadata, selectors, and direct/consolidated generations; prove
  full row/order/time/crop equivalence; compare every scientific keypoint
  array; and record byte identity, exact numeric equality, or bounded
  canonicalizing conversion accurately.
- Publish a new immutable strict-v2 candidate while leaving the legacy source
  untouched.
- Decide and implement the normal-consumption lifecycle explicitly. The
  current eye reader accepts selector-visible canonical surfaces, while strict
  v2 bundle/successor members are selector-ineligible. Either create a reviewed
  selector-visible publication bound to the candidate or admit exact
  bundle-authority members as a supported normal profile. Do not flip a boolean
  without a contract.

**Gate:** real legacy writer -> migration publisher -> chosen normal resolver
succeeds under the declared lifecycle; tamper, row reorder, crop mismatch,
missing authority, partial copy, consolidated-metadata drift, and selector
movement fail closed.

### Phase 5 — Republish immutable assignment dependents

- A new keypoint run cannot silently replace the exact run paths and digests
  already sealed into refined subject masks, worker assignment collections,
  recording bundles, and subject-shape publications.
- Rebuild the assignment-keypoint collection against the admitted keypoint
  publication, then publish new refined subject-mask/bundle/subject-shape
  generations through their maintained immutable writers.
- Do not mutate or restamp old manifests in place.
- Require one recording-wide exact `keypoints_runs` group/run for the initial
  eye-angle contract; mixed worker runs are blocking.

**Gate:** every new dependent publication names the exact migrated keypoint
authority and reproduces ordered crop rows, instance keys, acquisition frames,
labels, success, and coordinate digests.

### Phase 6 — Add one assignment-keypoint resolver

- Return one sealed profile-neutral result for historical refined-mask and
  recording-bundle subject-shape publications.
- Reuse each profile's strict loader; do not add a fallback, mode bypass,
  consumer-local grammar, or module-global adapter.
- Require an explicit `used` or `not_used` assignment state. Normal eye-angle
  analysis rejects `not_used`.
- Initially require one recording-wide canonical keypoint run for eye angles;
  mixed worker runs fail closed until a deliberate segmented-reader contract is
  implemented.
- Keep the resolver in a low-level shared leaf module so workflow/materializer
  imports cannot form a cycle.

**Gate:** both supported subject-shape profiles resolve through the same public
interface, and a failure in the selected profile does not try another profile.

### Phase 7 — Bind the proof through planning, staging, and publication

- Resolve assignment keypoints during metadata preflight, before exhaustive
  subject-shape payload validation.
- Pass the exact `--keypoint-run` as an assertion, not as an alternate selector.
- Version the staged subject-shape, keypoint-authority, and input-integrity
  receipts so the assignment proof is tagged and mandatory.
- Require the staged subject-shape and keypoint receipts to match the normalized
  resolver result field for field.
- Seal the authority and source-contract digest into the eye-angle publication,
  downstream export identity, availability, and reuse decisions.
- Make published immutable eye outputs and their readers use and validate the
  consolidated metadata generation.
- Propagate the same admission identity through registry readiness, reporting,
  visualization, exports, and campaign handoff.

**Gate:** missing authority fails in seconds, before scratch/output creation;
stale selectors, mixed runs, gaps, overlaps, row/time reorder, or digest drift
cannot reach a worker.

### Phase 8 — Close the test and reporting gaps

- Extend the real bundle writer fixture through the unpatched eye-angle input
  resolver.
- Add positive historical and bundle+migrated-authority round trips, including
  immutable assignment rebinding.
- Add negative `not_used`, missing, mixed-run, gap/overlap, stale-selector,
  row/time-reorder, payload-tamper, and successor-mismatch cases.
- Update reporting and registry readiness code that still assumes a direct
  refined-keypoint dependency.
- Add a test over generated production plans that rejects
  `finalize_keypoint_shards` as the terminal authority producer for a canonical
  clipped keypoint dependency.
- Add real writer -> unpatched reader tests for sealed crop-v2 tracking,
  materialized crop, keypoint normal consumption, track/swim/bout boundaries,
  and report/reuse admission.
- Add a required CI inventory/boundary gate rather than relying only on test
  file discovery.

**Gate:** focused outside-sandbox tests pass, followed by every required CI
check. Until then the implementation remains incomplete, not merge-ready, and
must not activate production selectors or update the shared checkout.

### Phase 9 — Rerun the four-recording chain

- Use a clean commit-pinned cluster deployment.
- Run an admission-only dry run first and preserve the resolved authority
  inventory with the campaign plan.
- Run eye angles and dependent analytics only after all proofs resolve.
- Preserve per-component scientific failure annotations without confusing them
  with authority-admission failures.
- Require successful serial registry finalization and preserve its receipt;
  physical complete/eligible outputs are not sufficient.

**Gate:** the campaign report identifies the exact producer and authority proof
for every consumed stage, and all requested outputs publish atomically under
their maintained contracts.

## Tests that allowed this to escape

The existing real bundle publication test reaches strict subject-shape loading
and eye-geometry resolution, but it does not call the unpatched eye-angle input
resolver. Eye-angle unit fixtures use context-bearing synthetic sources, so the
bundle source shape never reaches `_resolve_canonical_eye_keypoints()`.

The minimum regression is:

```text
real recording-bundle writer
  -> real subject-shape publisher
  -> strict publication loader
  -> unpatched assignment-keypoint resolver
  -> unpatched eye-angle input resolver
```

The negative fixture with assignment mode `not_used` should produce a semantic
fail-closed error rather than a raw `AttributeError`. The positive fixture must
use the exact admitted migrated keypoint publication plus republished assignment
dependents, and prove that its authority survives planning, staging, and
publication.

## Source map

- `src/fisheye/utils/finalize_keypoint_shards.py:627-810`
- `src/fisheye/analysis_workflows/execution.py:274-311`
- `src/fisheye/analysis_workflows/availability.py:345-411`
- `src/fisheye/shared/eye_geometry_source.py:240-355`
- `src/fisheye/analysis/eye_angle_analysis.py:3205-3425`
- `src/fisheye/shared/zarr/keypoint_coordinate_successor.py:537-661`
- `src/fisheye/shared/subject_mask_worker_receipt.py:944-1081`
- `src/fisheye/cluster/clipped_inference.py:1357-2101`
- `tests/unit/fisheye/test_subject_mask_recording_bundle_publication.py`
- `docs/diagnostics/crop_contract_split_audit_2026-08-24.md`
- `docs/diagnostics/source_of_truth_consolidation_plan_2026-08-25.md`
