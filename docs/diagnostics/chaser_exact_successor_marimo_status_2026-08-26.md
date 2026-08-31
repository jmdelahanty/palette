# Exact chaser successor Marimo status — 2026-08-26

<!-- contract-meta
status: active
implementation: first-class-body-bearing-and-gaze-local-validation-complete-ci-pending
last_verified: 2026-08-30
-->

## Purpose

This note records the boundary between persisted chaser publications and the
read-only Marimo recording explorer after the protocol-semantic successor work.
Persisted PNG/PDF bundles and Marimo are independent consumers of the same
immutable analysis products. External images do not replace, disable, or alter
interactive exploration.

The implementation plan and acceptance gates for the findings below are in
[`chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md`](chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md).

## Implemented in code

The explorer contains a `stimulus_chaser_exact_successors` capability. It is
intended to discover a recording only when one complete, selector-ineligible
spatial-occupancy successor seals an ordered keypoint/detection bundle and each
referenced relative-frame and radial child still has the exact bound manifest
digest and provider identity. The 2026-08-27 live-artifact audit below found a
reader compatibility defect that prevented receipt-bound v4 bundles from
reaching this capability. PR 70 corrected and merged that reader on 2026-08-29.

The declared interactive views are:

1. paired-provider distance, moving-reference radial selection, and exact-time
   near-field summaries;
2. full-session and exact protocol-epoch fish–chaser distance traces;
3. exact protocol-epoch fish positions with logged chaser-position overlays in
   the reviewed circular arena; and
4. exact protocol-epoch paired-provider spatial occupancy and the sealed-array
   detection-minus-keypoint display;
5. producer-logged exact controller-trial membership and trigger-aligned
   distance;
6. generalized bout-rate, kinematic, separation, and optional body-frame
   response;
7. exact-trial escape/freeze outcomes, event facts, trace-validity reasons, and
   persisted threshold sensitivity; and
8. conditional keypoint-authority body-bearing polar distributions;
9. conditional schema-v3 exact eye-gaze tracking, dynamic summaries, lock
   events, and rotated spatial controls; and
10. exact manifest, child identity, provider authority, and display-projection
   provenance.

The adapter resolves no `latest`, default, promoted, or fallback selector. It
deep-audits the small spatial/radial successor arrays. For frame-scale panels it
reads and content-hashes only the relative-frame arrays required by the selected
view. This is a bounded replacement for rescanning unrelated archive payloads.
Its cache/reactive identity binds the exact archive, run path, bundle manifest,
renderer, analysis, display-version string, and display-parameter digest. A
completed projection is rejected if any of those identities changes before
rendering.

Plotly point reduction is display-only and is recorded in figure metadata. The
distance projection preserves source order, local first/last/minimum/maximum
values, and all observed missing-data breaks. The position projection preserves
source order and coordinate extrema. Neither projection is a scientific input,
and neither writes a derived product to Zarr.

The discovery specification is currently synthesized in memory. The
`artifact_path` ending in `interactive` is a capability address, not a
persisted Zarr child: the audited spatial successor contains scientific arrays
and its sealed manifest, but no `interactive/zarr.json` or `visualizations`
subgroup. This can remain a valid read-only architecture if the runtime adapter
and its display semantics are versioned and tested. A separately persisted
interactive descriptor would be an additive visualization publication, not a
reason to mutate an immutable scientific successor.

## Live-artifact audit correction — 2026-08-27

### Static publication is complete and independently reproducible

The selector-ineligible execution completed for 80 eligible GoodBatBadBat
recordings: task 1 was the controlled canary and task indices `2-76,81-84`
completed in LSF array job `153756073`. Indices `77-80` remain excluded by the
known protocol-semantic ordering/overlap failure and were not silently treated
as members of the successful cohort.

The final read-only audit found:

- 80 unique cohort receipts, each bound to Palette commit
  `65b06a2f6ab4c4c30a92a8248a7ffb1742d70b0c` and frozen task digest
  `bbbcc9710d38c6bd5c0a611bc68b40c24ef908f1faaf93913b26b914a0256509`;
- 240 canonical plot receipts and 720 source-validation receipts;
- 720 PNG and 720 PDF files, exactly 18 outputs for every eligible recording;
- successful independent rehashing of all 1,440 plot outputs; and
- `selector_eligible=false`, `production_authority=false`, and
  `registry_update=false` throughout.

The Zarr publications contain the scientific arrays needed by an interactive
viewer. For example, the spatial successor persists exact epoch/provider
occupancy grids, bin edges, candidate and validity denominators, and arena
membership evidence. External plot-receipt JSON persists exact source
bindings, source-validation receipt digests, rendering and binning parameters,
and output path/size/SHA-256 records. Those JSON files are safe static
publication receipts; they are not standalone interactive datasets and the
current Marimo exact-successor adapter does not consume them.

### Consolidated visibility is correct

The root consolidated metadata for the audited recording contains the exact v4
spatial successor and all of its arrays. Direct metadata and consolidated
discovery both expose the run. The parent has no `latest`, `selected`,
`authoritative`, `default`, or other forbidden selector attribute. Therefore
the zero-option result is not caused by missing or stale consolidated metadata.

### Pre-fix exact Marimo discovery returned zero live v4 bundles

A read-only smoke against recording
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` and its exact spatial run
`goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_body_frame_projection_receipt_bound_v4`
returned zero `palette-chaser-exact-successor-explorer-v1` options.

All scientific identities pass individually:

- the spatial manifest is complete, canonical, selector-ineligible, and
  digest-valid;
- both relative-frame children validate against their exact run path, manifest
  digest, recording identity, provider identity, and completion state; and
- both radial children validate against their exact run path, manifest digest,
  recording identity, provider identity, and completion state.

Discovery fails only because it compares two versioned representations of the
same relative child by literal dictionary equality. A radial successor records
the minimal immutable child identity:

```json
{
  "run_path": "analysis/chaser_relative_frame_runs/<exact-run>",
  "manifest_sha256": "<exact-manifest-digest>"
}
```

The receipt-bound spatial bundle correctly enriches that identity:

```json
{
  "run_path": "analysis/chaser_relative_frame_runs/<exact-run>",
  "manifest_sha256": "<exact-manifest-digest>",
  "validation_receipt_sha256": "<exact-receipt-digest>",
  "verification_mode": "receipt_bound_targeted_array_rehash_v1"
}
```

The `run_path` and `manifest_sha256` agree exactly. The additional fields are
stronger publication evidence, not a different child. Both discovery in
`apps/marimo/components/registry.py` and the projection loader in
`apps/marimo/components/chaser_exact_successors.py` currently require whole-
object equality and therefore fail closed for the enriched binding.

The unit fixture uses the minimal two-field binding in both parent and child,
so it does not exercise the real receipt-bound shape. This is a test-boundary
gap. The fix must not loosen exact identity: it must validate a closed
receipt-bound binding schema, compare normalized child identity
`(run_path, manifest_sha256)`, retain and validate the receipt digest and
verification mode, and make discovery and loading use the same helper.

This defect affects reader admission only. It does not invalidate or require
rewriting the scientific Zarr publications, static images, plot receipts, or
source-validation receipts. No cohort recomputation or Zarr migration is
needed for the compatibility fix.

### Reader correction and acceptance evidence — verified 2026-08-29

The reader-only correction at commit
`559c08fdea42f0e4de3985033f95e99917a67a5f` passed all 23 required CI checks
and merged through PR 70 as merge commit
`33bc1c5b7f4d91348fc18ff2a8683e72761ea185`. It was not deployed or used to
activate a selector.

The correction adds one Marimo-independent exact-relative-child binding
validator. It accepts only the closed minimal and receipt-bound profiles,
normalizes scientific identity to exact `run_path` plus `manifest_sha256`,
and retains the validation-receipt digest and mode in an immutable proof. Both
registry discovery and projection loading use that validator. Neither path
reopens the external receipt, and the recorded validation behavior says so.
The synthesized explorer spec is now schema version 2.

Local fail-closed validation passed:

- 25 focused unit tests, including the production asymmetric binding shape,
  wrong child identity, malformed receipt evidence, unsupported mode,
  unexpected fields, provider mismatch/order, recording mismatch, incomplete
  child, forbidden selector state, and no unconsolidated retry;
- Ruff check, Python compilation, and `git diff --check`; and
- `scripts/py -m marimo check apps/marimo/palette_explorer.py` outside the
  sandbox.

Read-only acceptance against the frozen smoke recording found exactly one
matching receipt-bound v4 schema-v2 option in 0.359 seconds. Every analysis
declared by that reader revision loaded successfully:

- `radial_near_field`: 3.265 seconds, deep-auditing the exact spatial and two
  radial children without loading relative arrays;
- `provenance`: 3.209 seconds;
- `distance_traces`: 9.491 seconds; and
- `trajectory_overlays`: 8.350 seconds.

Each relative-array load content-hashed the 13 required array names across
both 149,936-frame, two-chaser providers, representing 26,988,480 decoded
array bytes. Both receipt digests remain visible in projection provenance, but
are described as sealed manifest evidence that was not independently reopened.

Metadata-only consolidated discovery over the frozen cohort task manifest
passed 80/80 eligible recordings in 31.079 seconds. Task indices `77-80`
remained explicitly excluded, and the task digest was
`bbbcc9710d38c6bd5c0a611bc68b40c24ef908f1faaf93913b26b914a0256509`.
No Zarr, selector, registry, authority, receipt, or static plot was written or
changed. A commit-pinned representative deep-load report remains an acceptance
follow-up for broader interactive parity; it does not erase the green required
CI evidence for the reader correction.

The remaining exact views will use the modular provider/package plan in
[`marimo_explorer_architecture.md`](../marimo_explorer_architecture.md). Palette
will retain one supported recording-explorer entrypoint while moving shared
projection, radial, distance, trajectory, spatial, controller-trial,
bout-response, escape/freeze, full-profile, and provenance behavior into
focused components behind one closed exact-chaser adapter. New exact behavior
will not be added to the legacy GoodCopBadCop component.

## Modular spatial-view implementation and smoke — 2026-08-29

PR 70 merged as `33bc1c5b7f4d91348fc18ff2a8683e72761ea185`, and
the modular architecture documentation merged through PR 68 as
`b1495ff7664fc80169aab583db3a765da6439660`. The next implementation branch
keeps two reviewable commits:

- `345083fb922316767a1b58f105f26c85943d2ef7` mechanically extracts the exact
  provider, projection, provenance, radial, distance, and trajectory modules,
  retains the old module as a compatibility facade, and replaces three
  analysis-specific notebook render branches with one closed adapter call; and
- `d215eb686045b59aacf2a201a99fdfdcf12c88ec` adds the persisted
  spatial-occupancy route and schema-v3 display parameters.

The spatial renderer reads the deep-audited `occupancy_count`,
`occupancy_density_valid_in_arena`, `occupancy_fraction_candidate_epoch`, exact
bin edges, bin-center arena mask, provider/epoch registries, and every persisted
candidate/validity denominator. It verifies row conservation and the two
persisted normalizations before displaying the source density as percent per
bin. Detection-minus-keypoint is display-only. The reviewed circle, `+y_down`
orientation, coverage counts, shared density scale, symmetric difference
scale, exact bin edges, mask-display policy, interpolation prohibition, and
scientific-recomputation prohibition are recorded in Plotly figure metadata.

The commit-pinned read-only smoke used:

- Palette path: `/tmp/palette-authority-supplier-clarity-20260827`;
- recording: `2026-08-10T17-20-55Z_arena_1_goodbatbadbat`;
- exact run:
  `goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_body_frame_projection_receipt_bound_v4`;
- bundle manifest:
  `7ab19bf9fa888f12cbb4fc06e01c847024584a3cc00f9df2a7be5fd9d57950f0`;
- display-parameter digest:
  `be992c84cc076f376c2b31a1237688b5e1a690a555d1625a1686eff8b2872af3`;
- providers: `keypoint_anatomical_triad_mean.v1` then
  `detection_bbox_centroid.v1`; and
- grid shape: two providers by three epochs by 42 by 42 bins.

Metadata discovery took 0.304 seconds and returned two explicit immutable exact
sources. Exactly one matched the receipt-bound v4 identity above; the other was
the older `20260825_exact_epochs_v1` bundle. No source was inferred or silently
selected. The v4 spatial projection loaded in 3.758 seconds, deep-auditing 19
spatial and 61+61 radial arrays totaling 324,573 decoded bytes. It loaded no
relative-frame arrays. The renderer exposed nine heatmaps totaling 15,876
cells. The first behavior smoke rendered in 0.116 seconds.

The modular/spatial work and explicit multi-source choice passed all 23
required checks and merged through PR 71 as merge commit
`13318a8b7a16399a70f290ea3bd5ad466f309ae9`. It remains undeployed and did not
change the shared `/groups` checkout or any selector.

## Exact controller-trial implementation and smoke — 2026-08-29

The current controller-trial branch adds a focused renderer in
`chaser_exact/controller_trials.py`, a focused audited binding/loader in
`chaser_exact/controller_trial_projection.py`, and one closed provider route. At
metadata discovery, it joins a spatial bundle to a controller-trial successor
only when exactly one complete selector-ineligible controller run has the same
recording ID, keypoint relative-frame run path/digest, semantic-selection run
path/digest, dimensions, exact logged-membership method, and no-fallback
policy. It never uses a shared run name, modification time, selector, or sorted
default. Zero or multiple matches leave only this capability unavailable. The
chosen controller outer manifest, scientific payload, relative source, and
semantic source are copied into schema-v4 synthesized analysis bindings and
their digest participates in the reactive selection identity.

The projection deep-audits all controller arrays and verifies the stored trial
table against dense source-row membership, envelope membership, gap membership
and reason codes, trigger timestamp/acquisition identity, per-trial counts,
chaser identity, and the paired relative-frame dimensions. Any `fallback_used`
row, unresolved active trial identity, changed digest, incompatible source, or
membership/gap overlap makes the panel unavailable.

The live read-only smoke selected receipt-bound v4 spatial run
`goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_body_frame_projection_receipt_bound_v4`
and uniquely joined controller run
`goodbatbadbat_chaser_successors_20260827_body_frame_projection_v4`. The exact
controller outer manifest digest was
`ba886b62cc99ffa8bd2272d06a051ba37779352b5aec6731abbe92333ee31b00`;
its scientific payload digest was
`3bd888e3da2e74fbccfde484b4c9474b05466712ad95c2f81a94a15145c9f9da`.
Discovery returned the two explicit spatial bundles in 0.314 seconds. The v4
smoke observed one exact controller match for each bundle and selected the
receipt-bound v4 bundle explicitly; no controller was cross-joined or chosen
as a default. The v4
controller projection loaded and deep-audited 299,872 source rows in 11.368
seconds. It found four trials with 2,024, 2,037, 1,907, and 1,993 exact active
members, zero gaps in this recording, and `fallback_used=false` throughout.
Synthetic tests retain and label explicit nonmember gaps so the zero-gap live
case does not erase that behavior. Real Plotly construction produced eight
full-session traces, eight trigger-aligned trial traces, and four exact trial
table rows. Gap markers require a valid source timestamp while the table
retains all gap counts, including untimed gaps. Display-only reduction limits,
missing-line breaks, gap-marker placement, interpolation prohibition, and
legacy-reconstruction prohibition are recorded in figure metadata.

Focused validation passed 48 tests, and all 23 required checks passed. The
controller-trial view merged through PR 72 as merge commit
`d7b1cde38efce05a106c20882f7cd3b8452cc1d2`.

The next modular package adds `generalized_bout_response`. Discovery joins the
spatial keypoint relative frame to exactly one immutable generalized successor
by recording identity, normalized relative-frame child, semantic-selection
digest, exact controller scientific payload, motion source/projection, swim-
bout lineage and signal, dimensions, method, and no-fallback policy. Ambiguity
or any mismatch hides only this capability. The loader deep-audits the chosen
successor and rechecks every synthesized binding before exposing persisted
arrays.

The view reads the already persisted semantic-role × chaser × onset-distance
band rates and medians without re-binning or re-aggregation. Bout-level plots
show the persisted bout start-to-end separation change and, when declared,
the persisted body-frame bearing/turn rows. The successor does not define an
independent post-bout response window; the viewer therefore does not invent
one. It does not resegment bouts, reconstruct trial membership, substitute
motion heading, interpolate rows, or turn envelope gaps into events. Raw rows
may undergo deterministic endpoint-preserving display thinning capped at 6,000
points per role/chaser series; persisted summaries are not thinned.

The live receipt-bound v4 smoke discovered two explicit spatial bundles and
selected the v4 bundle by its exact run identity. It uniquely joined
`analysis/generalized_chaser_bout_response_runs/goodbatbadbat_chaser_successors_20260827_body_frame_projection_v4`,
whose outer manifest digest is
`81a01f070e274ab546814e36380b335a84e6116718052678977a6bc85a23dc80`
and scientific payload digest is
`bd989380194427405b0a42387defa1e084ce5a738dda8ae109de523aea2cb30f`.
The deep audit loaded 1,445 bouts, 2,890 bout-by-chaser rows, and 30 persisted
summary rows in 13.361 seconds. Of 2,466 base-valid rows, all 2,466 carried
valid body-frame axes. Eight rows began during exact active controller-trial
membership; the remaining rows retained their explicit non-trial attachment
state. Focused validation passed 57 tests and all 23 required checks passed.
The generalized view merged through PR 74 as merge commit
`267553f548d0590958938d1d2227e5c87f19ec8a`. It was not deployed and did not
activate a selector.

The next modular package adds conditional `escape_freeze` availability only
when exactly one immutable version-2 escape/freeze successor binds the selected
recording's exact controller scientific payload, generalized bout-response
scientific payload, and identical motion source/projection. Discovery also
validates the complete classifier parameter record, dimensions, method,
policies, response-class registry, trace-reason registry, and selector-
ineligible state. Ambiguity or any mismatch hides only this capability. The
loader deep-audits every persisted array and rechecks the payload joins and
classifier contract.

The viewer uses the persisted trial response classes, event outcomes,
recapture facts, reason codes, and threshold-sweep rows. It never derives a
class from displayed points, resegments a bout, reconstructs a trial, changes a
threshold, substitutes orientation, or interpolates evidence. Event rows may
undergo source-order, endpoint-preserving display thinning capped at 6,000
points; event tables are capped at 1,000 rows and all recording totals remain
persisted values. Version 2 records whether an event has a usable recapture
trace and its outcome, but does not persist aligned distance samples. The
viewer therefore states that trace trajectories are unavailable and does not
reconstruct them.

The receipt-bound v4 live smoke explicitly selected the v4 spatial bundle from
two visible immutable choices and joined
`analysis/chaser_escape_freeze_runs/goodbatbadbat_chaser_successors_20260827_body_frame_projection_v4`.
Its outer manifest digest is
`2fcc8807b7c362ea98a0639f57c302b5ee03fcabb81276a51b8551316f999c70`
and its scientific payload digest is
`839db255e9f9ede7a46a08f1d706c6032f332bf0f5754a022179ebba581a2774`.
The original 12.971-second deep load validated four trials, five events, twenty
persisted sweep rows, one escape trial, three freeze candidates, zero
high-turn events, and five trace-usable events. All five event trace-reason
codes were `valid`. The implementation passes 59 focused tests, the 163-test
Marimo suite (148 passed and 15 expected xfails), and Marimo check. It passed
all 23 required checks and merged through PR 75 as merge commit
`8c6b2d7d1a1b1491098b1f82680c2fd1596edddd`.

### Receipt-bound exact projection reader — 2026-08-30

The current follow-up adds one external composition receipt over the seven
exact immutable children and two relative-frame children used by the exact
viewer. Each producer receipt remains an independent lineage record. The
composition is a closed consumer choice and is explicitly selector-ineligible,
production-ineligible, registry-ineligible, and not a cache authority.

With `--exact-chaser-receipt`, each composable child is opened through its
direct receipt rather than reparsing the 16.5 MB archive-root consolidated
document. The typed loader rehashes only the selected renderer's declared array
roster. Bout and escape/freeze routes validate relative-frame receipt metadata,
dimensions, authorities, and exact joins without reading relative arrays they
do not render. No-receipt launch retains the exhaustive deep-audit path; a
supplied stale or mismatched receipt fails closed and never falls back.

On the same live receipt-bound v4 target, escape/freeze loaded in 4.151 seconds
and renderer validation completed in 0.001 seconds. The load rehashed 53
escape/freeze arrays and read zero spatial, radial, controller, bout, or
relative arrays. Figure provenance records the projection receipt digest,
child receipt digests, verification modes, and exact verified array rosters.
The smoke composition resides only under `/tmp`; no recording, registry,
selector, existing receipt, or shared checkout was modified. Required CI for
this new branch is pending, so it is not merge-ready or deployed.

The receipt-bound reader subsequently passed required CI and merged through
PR 76 as merge commit `e0521c6bcf39859634c8d52bc4a8bc98f73cf721`.
The temporary smoke receipt remains non-durable and no selector or shared
checkout was changed by that merge.

## Gaze, bearing, distance-distribution, and heatmap audit — 2026-08-30

### Gaze was not lost by the explorer

The frozen 84-entry body-frame cohort task never requested
`chaser_gaze_tracking_v2`, never selected an exact eye-angle child, and never
supplied the required gaze-convention review receipt. A direct metadata-file
census of those 84 archive paths found zero `analysis/eye_angle_runs`, zero
`analysis/subject_shape_runs`, and zero modern subject/refined-subject mask
families. The audited smoke archive consequently contains no
`analysis/chaser_gaze_tracking_runs` child.

The absence occurs at three separate boundaries:

1. the cohort producer requested controller, generalized bout-response, and
   escape/freeze successors but not gaze;
2. no modern reviewed eye-orientation source was available to that frozen
   cohort; and
3. exact projection-receipt schema v1 and the exact Marimo provider contain no
   gaze child or gaze route.

This is an upstream production and composition gap, not a hidden panel and not
authorization to use the legacy `analysis/chaser_distance_runs/.../gaze_tracking`
component. The modern successor already fails closed on an exact compact-v7
eye-angle payload, its exact biological-direction convention receipt, the
protocol-semantic selection, and the anatomical body-bearing relative-frame
extension. It prohibits world-frame gaze, nasal-positive eye-angle
substitution, motion-heading substitution, and selector resolution.

### Lock fraction and sustained lock events

The modern successor defines one row per acquisition frame x eye x chaser. A
row is valid only when it has a nonzero semantic role, valid reviewed eye gaze,
valid anatomical chaser bearing, valid physical distance, and distance no
greater than the configured 50 mm default. Each eye's accessible bearing range
is the configured empirical gaze interval; the current successor default is
the 2.5th through 97.5th percentiles of that eye's valid gaze values.

For a semantic role x eye x chaser stratum:

```text
lock_fraction = count(valid and accessible and abs(wrapped_gaze_error) <= 10 deg)
                / count(valid and accessible)
```

The 10-degree threshold, 50 mm distance limit, empirical quantiles, numerator,
and denominator are successor provenance. A lock fraction is descriptive
alignment occupancy; it is not alone evidence that the eye actively followed
the chaser.

A lock event is one contiguous run of lock rows for the same semantic role,
eye, and chaser whose timestamp span is at least the configured 0.10 seconds.
The successor persists start/end acquisition-frame identities, timestamp-derived
duration, sample count, and median absolute gaze error. Invalid or non-lock
rows split events; the current contract does not bridge gaps.

### Rotated spatial null controls are missing from the modern successor

The legacy gaze component constructed virtual references by rotating each real
chaser trajectory around the reviewed arena center by 60, 120, 180, 240, and
300 degrees. This preserves the reference trajectory's arena radius, motion,
and wall proximity while moving it to an unoccupied location. A virtual
reference was rejected when it came within 8 mm of a real chaser for more than
5 percent of finite samples. The legacy recording summary compared the real
chaser with the mean of its retained virtual twins for lock fraction, static
tracking gain, zero/best-lag dynamic gain, and median absolute gaze error.

This is a spatial arena/wall-geometry null, not a temporal shuffle. The modern
`palette.analysis.chaser_gaze_tracking` schema version 2 persists gaze error,
lock occupancy, static regression, and sustained events but does not persist
the rotated controls or dynamic-lag summaries. The older catalog description
still promises rotated controls, so the catalog and modern payload currently
diverge. First-class gaze publication must close that divergence in a new
successor schema before the cohort is materialized; Marimo must not recreate a
null from displayed rows.

### First-class exact gaze and bearing boundary

"First-class gaze" means a conditional analysis capability of the exact chaser
provider with its own immutable successor lineage, source bindings, validation
receipt, composition-receipt child, discovery proof, projection loader, array
roster, renderer, and explicit unavailable state. It is not a third position
provider. Keypoint anatomical-triad and detection bbox-centroid remain paired
first-class position providers. Detection can support position, distance,
radial, and spatial analytics, but it cannot provide anatomical bearing without
an independently bound body axis.

The anatomical keypoint relative-frame child already persists body-frame
chaser bearing. A keypoint-only polar bearing view can therefore be added now,
independently of eye-angle cohort readiness. It must display the provider/body-
frame asymmetry explicitly rather than invent detection bearing. The gaze view
then compares reviewed left/right eye direction with that same sealed
anatomical bearing and adds error, tracking, lock, event, and rotated-control
panels.

Existing projection-receipt schema v1 remains valid for the seven exact
children it closes. A gaze-capable composition must use a new closed receipt
schema/version that requires an exact gaze child; a supplied v1 receipt cannot
silently deep-audit or discover gaze. No-receipt diagnostic mode may retain its
explicit exhaustive-audit behavior.

### Distance distributions and exact spatial heatmaps

Both radial successors already persist exact epoch x behavior x chaser CDFs,
observed radial probability mass, moving-reference geometric expected mass,
selection indices, and 5 mm wall-excluded twins. The current exact Marimo
radial roster loads only medians/interquartile ranges, near-field summaries,
and selection indices. The missing per-epoch distributions are therefore a
reader parity gap, not a scientific-publication gap.

The safe interactive addition is a receipt-bound `distance_distributions`
route that reads those persisted arrays without framewise reconstruction or
rebinning. It should display paired-provider CDFs, observed versus geometrically
expected radial probability mass, and explicitly separate wall-excluded twins.
Every exact bin edge and denominator remains successor evidence and every
display parameter remains in figure metadata.

The current exact spatial heatmap is already on the latest scientific surface:
the sealed 2 mm reviewed-arena grid, pre/training/post epochs, paired providers,
valid-in-arena conditional density, candidate-epoch fraction, complete coverage
denominators, shared provider scale, and no interpolation. Static and Marimo
renderers duplicate their validation and display-parameter derivation. They
should share one scientific display projection. Conditional valid-in-arena
occupancy remains primary; the already-persisted candidate-epoch-normalized
surface should be available as an explicitly coverage-sensitive companion.

### First-class gaze and body-bearing implementation addendum — 2026-08-30

The implementation branch now mounts the accepted keypoint body-bearing
extension as a conditional `body_bearing_polar` capability. It uses fixed
10-degree whole-circle bins for the full recording and exact semantic epochs,
normalizes within panel and chaser, and records the display recipe in figure
metadata. Detection bbox centroid remains a first-class position and distance
provider, but neither detection nor motion heading is permitted to supply an
anatomical body axis.

The scientific gaze successor is advanced to schema version 3. It binds the
exact reviewed eye-orientation payload and convention receipt, keypoint
relative-frame body extension, protocol-semantic selection, and the exact
radial successor that seals reviewed arena geometry and physical scale. The
payload now retains every rotated candidate and its collision decision,
accepted virtual-reference summaries, finite per-metric null denominators,
real-minus-virtual gain/lock contrasts, virtual-minus-real error improvement,
zero-lag dynamic gain, and causal nonnegative best-lag gain. Sustained lock
events and their exact-frame/timestamp evidence remain persisted scientific
outputs rather than viewer reconstructions.

The exact explorer now has a closed gaze array roster, metadata-only discovery,
strict projection loader, typed unavailable state, and persisted-array panels
for gaze versus bearing, gaze-error distributions, lock/static summaries,
dynamic tracking, real-versus-rotated controls, and sustained events. A
projection receipt containing the original seven exact children remains schema
v1. Adding the gaze child produces the closed schema-v2 roster. A supplied v1
receipt cannot authorize gaze and cannot silently fall back to deep audit.

This makes the code path ready for validated schema-v3 gaze publications, but
does not manufacture gaze for the frozen body-frame cohort: those recordings
still need explicit reviewed eye-angle sources, biological-direction convention
receipts, and newly materialized immutable gaze successors. Required CI and a
live gaze-capable artifact smoke remain release evidence, not assumptions.

A read-only live smoke on
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` validates the independent bearing
path. Metadata discovery exposes `body_bearing_polar`; the existing projection
receipt permits targeted rehash of exactly `body/body_bearing_deg` and
`body/body_bearing_valid`; and the renderer returns eight polar traces for two
chasers across the full recording and three exact semantic epochs. The same
recording truthfully omits gaze because it has no schema-v3 gaze successor.

### Historical refined-keypoint rebinding and cohort execution addendum — 2026-08-31

The earlier census statement that these archives had no modern subject-mask or
keypoint families was incorrect: that check searched below `analysis/`, while
`subject_mask_bundle_runs`, `refined_subject_masks_runs`, `keypoints_runs`, and
`refined_keypoints_runs` are root-level families. The scientific conclusion
that the frozen cohort has no subject-shape, eye-angle, or schema-v3 gaze
successor remains correct.

The remaining source mismatch is narrower. Each historical subject-mask bundle
records the refined-keypoint rows that were actually used during assignment,
while the current coordinate authority is a later immutable
`keypoints_runs` coordinate successor. That relationship must not be inferred
from matching dimensions or labels. The rebinding contract now admits exactly
two historical assignment profiles: raw `keypoints_runs/detection_success` and
refined `refined_keypoints_runs/usable_keypoints`. The refined profile must be
the exact active refined-keypoint bundle member, retain its valid manifest and
source bindings to the active raw member, match recording and skeleton
semantics, and compare row-for-row with the selected canonical coordinate
successor. Coordinates, crop-row identity, instance identity, acquisition-frame
identity, and the profile-specific success flag are hashed and compared without
interpolation or row dropping.

For `2026-08-10T17-20-55Z_arena_1_goodbatbadbat`, the exhaustive read-only
preflight compared all 151,478 assignment rows and passed every identity,
coordinate, and `usable_keypoints`-to-`pose_success` equivalence. Its proposed
rebinding manifest payload digest is
`54679865067e150f46ad7dac958d07833f227c75861fb801ce28ba2180c5b5ec`.
Nothing was written to the recording by that preflight.

A metadata-only census of the frozen 84-recording chaser cohort found all 84
ready for the same exhaustive proof: each has exactly one compatible inactive
coordinate-bound subject-mask bundle, exactly one canonical coordinate
successor bound to the active keypoint authority, and no pre-existing target
rebinding, subject-shape, or eye-angle run. The frozen prerequisite task is
`palette.eye_gaze_prerequisite_cohort_task` version 1 with task digest
`e919319073170d30c33ba0cae06f551f173c43b4f179c28feed327b345ec6178`.
It seals every relevant metadata and validation-receipt file digest, exact
source and output identities, and a one-writer-per-recording policy.

Execution is deliberately split into two gates. First, one read-only LSF worker
per recording reruns the exhaustive comparison and writes an exclusive proof
receipt bound to the frozen task entry and full clean Palette commit. Second,
materialization revalidates the frozen inputs, reruns the proof, and requires
the exact receipt before publishing the immutable selector-ineligible rebinding,
an access-aware subject-shape candidate, an access-aware compact-v7 eye-angle
candidate, numeric convention validation, and a bounded review PNG. It changes
no selector, registry row, production authority, or human review decision.
Materialization begins with one recording; indices 2 through 84 are eligible
only after that smoke succeeds. Any partial write requires new immutable run
names rather than an in-place retry.

The numeric candidate validator now has one explicit admission path for a
complete `stage_selector_eligible=false` access-aware eye-angle child. It
requires an exact child name, validates compact-v7 payload and candidate storage
contracts, and proves direct/consolidated subtree equivalence. Normal canonical
eye-angle resolution is unchanged. Numeric validation is not biological
direction acceptance: a reviewer must still accept the convention before a
convention receipt, gaze successor, or gaze-capable composition receipt can be
created.

This repair is historical, not a new-ingestion requirement. Future subject-mask
assignment publications should bind the canonical coordinate source they use;
when that identity is already exact, no rebinding successor is needed. At the
time of this addendum, the implementation and frozen cohort task are locally
validated, but no cohort Zarr writes or LSF submissions have occurred and
required CI remains pending.

### Exact body-bearing × distance parity addendum — 2026-08-30

The earlier GoodCopBadCop viewer retained two related products that had not
been migrated into the exact-successor provider: a polar point cloud whose
angle is anatomical chaser bearing and whose radius is fish--chaser distance,
and a joint distance-by-bearing polar density. The exact provider exposed only
the one-dimensional `body_bearing_polar` marginal, so neither older product was
discoverable for the live recording even though its exact keypoint relative
frame already contained the required evidence.

The implementation branch now adds a distinct conditional
`body_bearing_distance` analysis. It is available only through the same closed
accepted-keypoint body-bearing binding as `body_bearing_polar`; detection
centroids remain a first-class distance provider but are not substituted for
an anatomical body-axis supplier. The view intersects exact panel membership,
`chaser_occurrence_member`, `relative_physical_valid`, and
`body_bearing_valid`. Declared-valid non-finite, negative-distance, or
out-of-circle bearing values fail closed. Missing or invalid rows remain
excluded evidence and no interpolation occurs.

Both Marimo and static publication use the shared plotting-library-neutral
recipe `accepted_body_axis_bearing_distance_display_v1`. Density uses fixed
5 mm distance bins and 30-degree whole-circle bearing bins, normalizes within
each panel and chaser, and records realized bin edges and exact denominators.
Its color scale uses the recorded 0.98 positive-bin probability quantile shared
across panels. The interactive point cloud is deterministically limited to
4,000 source-ordered rows per panel/chaser; static publication is limited to
20,000. The density always consumes every jointly valid exact row.

The detailed static receipt is advanced to schema/recipe v4 and grows from
seven to nine figure families (14 to 18 PNG/PDF files). The two added families
are `keypoint_body_bearing_distance_point_cloud` and
`keypoint_body_bearing_distance_density`. Receipt-bound publication now
target-rehashes the two body arrays in addition to its base-array roster and
seals body authority, source paths, bin edges, panel denominators, sampling
limits, normalization, and color scaling. Cohort recipe names advance to the
non-colliding detailed bundle v4 / receipt-bound recipe v6 identities; earlier
bundles remain immutable.

This closes single-recording interactive and receipt-sealed static parity for
the keypoint-authority bearing--distance views. It does not yet create a
persisted cross-recording joint-histogram successor, so group/cohort pooling of
this exact product remains a separate versioned publication step.

A read-only receipt-bound smoke against
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` discovers the new analysis only
on the body-frame-authorized v4 bundle. The targeted load completed in 4.042 s
and rendering in 0.174 s. Both figures contain eight panel/chaser traces (full
recording plus three semantic epochs for two chasers). Their eight denominators
sum to 592,362 jointly valid panel rows; density retained all of them and the
display-only point clouds retained the configured 32,000-row total. Focused
validation passed 79 tests, the neighboring receipt/materialization surface
passed 67 tests, and Marimo check passed. Required CI remains pending release
evidence.

### Protocol-authored chaser appearance and exact location overlays — 2026-08-30

The earlier exact trajectory reader assigned green, purple, and later palette
colors by chaser column. That was a display defect: experimental color is a
protocol datum and is not a synonym for behavior role. A blue rendered object
may be aggressive, random non-chasing, or inert. The spatial occupancy reader
also omitted the pre/post logged chaser locations even though its bound
relative-frame children retained them.

The implementation branch now resolves one shared fail-closed appearance
projection. The relative-frame manifest already seals a self-digested chaser
occurrence record containing the exact stimulus-run path, complete protocol
SHA-256, protocol chaser index, stimulus-run-scoped identity, and behavior
role. The reader opens that exact run through consolidated metadata, rehashes
the complete `protocol_json`, resolves its chaser list, and requires exact
identity/role/cardinality agreement with both paired relative-frame manifests
and their stable frame-column codes. Missing explicit RGBA, stale digest,
mismatched role, or selector-like path makes the panel unavailable; no
role-color, index-palette, or default-black substitution is allowed.

Marker fill now uses the protocol-authored RGBA. Role is a separate glyph and
text layer: aggressive is a star, random non-chasing a diamond, inert a circle,
and unknown an X. Exact identity and protocol index remain in hover/legend;
the contrast outline is display-only. This contract is used by the interactive
trajectory view and by new spatial pre/post overlays. Spatial locations are
medians of all exact valid logged rows within the sealed pre or post interval,
with sample count and median/p95/maximum drift retained in figure metadata and
hover. The moving training epoch is deliberately not collapsed to one static
marker; its logged path remains in the trajectory view.

The live receipt-bound smoke target proves the distinction: both configured
objects are experimentally black, while their roles are aggressive and inert.
Both views render black fills with star/circle glyphs. The complete exact-chaser
Marimo unit surface passed 82 tests, including an adversarial two-blue-chaser
case, and `marimo check` passed outside the sandbox. Required CI remains a
release gate.

### Occupancy display range and static appearance parity — 2026-08-30

The canonical spatial-occupancy evidence remains the persisted 2 mm count,
density, candidate-fraction, edge, mask, and denominator arrays. The viewer now
defaults to one shared p98 color limit computed over strictly positive bins
across both providers and all three epochs. This is a display transform only:
bins above the limit are color-saturated, while their exact persisted density
and count remain available in hover. A full-range reference is selectable from
the same figure, and both robust and full limits, quantile method, positive-bin
count, and saturated-bin count are sealed in figure metadata.

The same menu offers an aligned 4 mm display. It sums complete 2 by 2 blocks of
the canonical 2 mm integer counts, proves count conservation independently for
every provider/epoch, and normalizes with the original persisted in-arena and
candidate denominators. It does not interpolate, alter the source Zarr, or
become a scientific authority. The four explicit modes are 2 mm robust, 2 mm
full range, 4 mm robust, and 4 mm full range; the 2 mm robust view is the
default.

The receipt-sealed static trajectory publication now uses the same fail-closed
appearance projection as the interactive viewer. Every valid exact row remains
in the dense path and uses protocol RGBA. An independent, bounded set of exact
source-order rows carries the behavior-role glyph, including endpoints; it is
display sampling, not scientific downsampling. The detailed receipt advances
to schema/recipe v5 and records the appearance projection, glyph limit and
sampling policy, color source, and prohibited fallbacks. Cohort output names
advance to detailed bundle v5 / recipe v7 so existing immutable publications
are never overwritten. The publisher resolves and validates this appearance
projection once, then supplies that same object to both the trajectory renderer
and receipt-parameter builder so the receipt cannot describe a separately
reloaded appearance snapshot.

The receipt-bound live smoke selected the exact v4 occupancy child from two
discoverable bundles. Its canonical 42 by 42 grid became a 21 by 21 aligned
display, with all six provider/epoch count totals conserved exactly. The 2 mm
density scale was 0.636491%/bin at p98 versus a 59.106231% full maximum; 53 of
2,621 positive bins exceeded the robust limit. The 4 mm scale was 1.715864%
versus 59.466676%; 20 of 964 positive bins exceeded the robust limit. Both
chasers remained protocol black and rendered as aggressive/star versus
inert/circle. A separate receipt-bound static smoke produced non-empty PNG and
PDF trajectory files using appearance projection
`966103d63f0b1081b43c97ea1068cb9adc2ea7e1f05cbc6c1d3b4156453b45e6`.
The 107-test affected surface and Marimo check passed outside the sandbox.
Required CI remains the release gate.

### Independent relative-child receipt admission — 2026-08-30

The completed body-alignment cohort exposed one reader-only mismatch between
scientific child identity and validation-receipt lineage. All 80 eligible
recordings persist a complete anatomical alignment-by-distance successor, and
each successor names the same exact keypoint relative-frame `run_path` and
manifest SHA-256 as its spatial-occupancy anchor. In every recording, however,
the spatial and alignment publications correctly bind different validation
receipt digests because each producer invocation audited the immutable child
for its own lineage.

The shared exact-relative binding contract already states that receipt evidence
does not create a different scientific child, but its comparator still rejected
two receipt-bound records when their receipt digests differed. Metadata-only
Marimo discovery therefore omitted `body_alignment_by_distance` for 80/80
eligible recordings even though the scientific Zarrs and static receipt-bound
plots were valid.

The reader correction validates both closed binding profiles independently and
compares only the immutable scientific identity `(run_path, manifest_sha256)`.
It retains the expected and observed receipt digests, verification modes, and
their `independent_receipts_same_exact_child` relationship in proof provenance.
Different child paths or manifest digests still fail closed; malformed receipt
digests, unsupported verification modes, selector-like paths, and unrecognized
field sets remain rejected.

A read-only live smoke against
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` now discovers
`body_alignment_by_distance` on the receipt-bound v4 exact bundle in 0.379
seconds. Its v7 composition receipt target-rehashed the authorized arrays in
21.610 seconds, and the renderer produced four interactive figures in 0.174
seconds with 6, 6, 12, and 12 traces. No Zarr, scientific manifest, validation
receipt, static plot, registry record, selector, or authority was rewritten.
Required CI remains pending for this correction.

## Persisted but not yet mounted in this capability

These products are safe candidates for later read-only panels, but were kept
out of the initial component:

- gaze/controller-trial views where a complete gaze successor is present; and
- the full-profile readiness and module-binding envelope.

Each addition should anchor to the same exact bundle or to an exact full-profile
binding, validate child manifest and payload identities, and render persisted
arrays without recomputing trial membership, event classification, timing, or
geometry.

The currently declared exact-successor analyses are
`radial_near_field`, `distance_traces`, conditionally `body_bearing_polar`,
conditionally `body_bearing_distance`, `trajectory_overlays`,
`spatial_occupancy`, conditionally `controller_trials`, conditionally
`generalized_bout_response`, conditionally `escape_freeze`, conditionally
`gaze_tracking`, and `provenance`. They do not provide interactive
equivalents for all nine static figure families. In particular,
event-aligned escape distance trajectories and the composed full dashboard
remain absent from this exact reader. Older GoodCopBadCop components or
candidate views must not be used as an implicit fallback for these
receipt-bound successors.

## Still evidence-blocked or deferred

- Ring-entry video clips remain deferred. A safe implementation needs explicit
  frame-to-video identity, clip-boundary policy, and an immutable media receipt.
- Camera-exposure or physical-stimulus synchronization must not be inferred from
  session timestamps. Current exact-time panels retain their declared session-
  time semantics.
- Promotion or production selector activation remains outside the explorer. The
  current successors are intentionally selector-ineligible.
- Cohort comparison remains a separate product. This recording explorer does
  not silently aggregate recordings or choose among multiple bundles.

## Live recording smoke target and current result

The required implementation smoke target is:

`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`

Its spatial bundle binds the first-class providers
`keypoint_anatomical_triad_mean.v1` and `detection_bbox_centroid.v1`, three exact
chaser epochs, their paired radial successors, and their paired relative-frame
runs. The pre-fix reader returned zero options despite correct direct and
consolidated metadata. The merged correction now discovers exactly one option
matching the receipt-bound v4 identity without a selector, legacy, candidate,
or unconsolidated fallback. The recording also retains an older explicit exact
bundle, so current discovery presents two source choices without treating
either as scientific authority. When more than one immutable exact bundle is
visible, the source dropdown starts unselected and no analysis projection loads
until the operator chooses one. A sole exact bundle remains the unambiguous
default. The reader and modular spatial implementation are present on `main`;
the controller-trial addition above is merged. The generalized bout-response
addition is also merged. The escape/freeze addition remains on its locally
validated, CI-pending branch.
