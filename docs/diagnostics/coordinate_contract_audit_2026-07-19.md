# Coordinate contract audit and remediation status

Date: 2026-07-19; remediation status updated 2026-07-19

This report records the read-only registry/archive audit and the resulting
future-facing contract direction. Implementation changes are tracked separately
and do not alter production archives. Fleet counts come from the completed
read-only scan of the canonical live registry at
`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`. The
previous `/nvme1/palette_registry.sqlite` scan is explicitly retained only as a
superseded historical snapshot.

## Conclusion

Palette is converging on one metadata framework, not one universal coordinate
space. Every geometry array owns a compact descriptor and digest; that
descriptor binds exact typed frame, extent, transform, derivation, row-identity,
and time evidence stored elsewhere in the same archive.

The audit does **not** show that most existing numerical results are wrong.
It shows that many existing results cannot be independently proved from their
persisted metadata. Correct mathematical kernels can remain unchanged after a
reader or producer boundary validates and, when necessary, transforms their
inputs. Code must change where it currently chooses a dimension, crop,
calibration, or coordinate conversion by convention.

## Scientific-computation impact

This remediation is more than documentation, but it is not a general rewrite
of Palette's mathematical algorithms. Distance, speed, angle, interpolation,
body-frame, and smoothing kernels should continue to receive ordinary NumPy
arrays. Their producer/reader boundary must first prove the arrays' frame,
extent, identity, time mapping, and calibration or apply one exact persisted
directed transform. Output publication must then bind the result to its exact
inputs and row identity.

Most already-canonical contiguous inputs therefore retain identical numerical
results. One intentional behavioral correction is that refined time-series
smoothing now breaks at nonconsecutive acquisition-frame indices: adjacent
stored rows separated by an unrecorded frame are no longer treated as adjacent
time samples. A second bounded correction is required at the subject-shape
publication boundary: current origins and landmarks are computed and stored in
ROI-local pixels, while canonical body-frame v1 permits source-camera pixels or
physical millimetres. Translation-only crops preserve axes, angles, and
distances and translate only points/origins; scale or padding requires applying
the exact persisted ROI-to-source transform. Historical results otherwise
remain `plausible_unproven` unless a value-level comparison demonstrates an
error. Missing metadata alone is not evidence that their computations were
inaccurate.

## Coordinate flow

```text
acquisition camera frame (source_camera_image_px)
  |
  +-- exact resize/letterbox/crop transform
  v
detector model input (detector_model_input_px)
  |
  +-- detector-normalized bbox + inverse preprocessing transform
  v
detection source-image bbox/center (source_camera_image_px)
  |
  +-- exact selected detection rows + source_crop_xywh placement
  +-------------------------------> crop/ROI local pixels (roi_local_px)
  |                                      |
  |                                      +-- keypoint model-input transform
  |                                      v
  |                                keypoint ROI/model coordinates
  |                                      |
  |                                      +-- ROI-to-source placement
  |                                      v
  +-------------------------------> keypoint source-image coordinates
  |
  +-- exact subset/reorder by track_sample_key
  v
track positions_px in the selected native frame
  |
  +-- exact typed physical calibration, when compatible
  v
track positions_mm (physical_mm)

stimulus texture -> canvas -> arena-relative canvas -> projector
  |                  direction-labelled transform chain
  +------------------------------------------------------+
                                                         v
                                          source-camera overlay frame

subject ROI/source geometry -> fish-anatomical body frame

Palette source-camera overlay frame
  -> Crimson viewport/display transform (ephemeral renderer state only)
```

## Important persisted geometry surfaces

| Surface | Producer | Required space and axes | Extent/units | Required lineage | Current consumer |
|---|---|---|---|---|---|
| acquisition image arrays | acquisition/import | `source_camera_image_px`, top-left, +X right, +Y down | exact camera W/H, px | acquisition camera/frame ownership | detection, crop, Crimson video |
| detection `bbox_norm_coords` | detector/import | `detector_normalized_xy` or explicitly transformed source-normalized profile | exact detector-input frame, unitless | model-input preprocessing and selected acquisition stream | refinement/audit only |
| detection `bbox_img_xyxy`, `centers_img_xy` | detection publication | `source_camera_image_px`, top-left, +X right, +Y down | exact source-camera W/H, px | inverse model-input transform, observation identity/time | crop, tracking, Crimson bbox |
| crop `source_crop_xywh` and top-left alias | crop | `source_camera_image_px`, top-left, +X right, +Y down | exact source-camera W/H, px | exact selected detection rows and placement derivation | ROI reader, keypoints, masks |
| crop `bbox_roi_xyxy`, `roi_images` | crop | `roi_local_px`, top-left, +X right, +Y down | exact ROI W/H, px | direction-labelled ROI-to-source placement | keypoints, masks, review |
| keypoint model-input geometry | keypoint inference | `detector_model_input_px`, top-left, +X right, +Y down | exact submitted tensor W/H, px | exact resize/pad matrix and model artifact | keypoint inverse transform |
| keypoint `keypoints_roi` and ROI pose bbox | keypoint publication | `roi_local_px`, top-left, +X right, +Y down | exact ROI W/H, px | model-input-to-ROI transform and `instance_key` | review, subject shape |
| keypoint `keypoints_img` and image pose bbox | keypoint publication | `source_camera_image_px`, top-left, +X right, +Y down | exact source-camera W/H, px | ROI placement plus same `instance_key` | Crimson, shape, analysis |
| keypoint normalized mirrors | keypoint publication | `source_camera_normalized_xy` | exact source-camera W/H authority, unitless | exact image-coordinate derivation | interchange/validation |
| stimulus target/chaser positions | stimulus import | selected texture/canvas/arena-relative profile | exact texture/canvas W/H, px | stimulus-state identity, acquisition-frame map, directed transform chain | refined online, chaser metrics, Crimson |
| track `positions_px` | track kinematics | exact selected source profile; often source camera, sometimes arena-relative canvas | copied source extent, px | `track_sample_key`, source row selection/time, source descriptor | bout/scientific readers, Crimson |
| track `positions_mm` | track kinematics | typed `physical_mm` profile | mm | exact compatible calibration and px-to-mm derivation | speed/distance/bout analysis |
| subject-mask raster/contours/metrics | mask finalizers | usually ROI-local; source-image only after explicit placement | exact ROI or source W/H, px | observation identity, component, encoding and placement | review/training/subject shape |
| subject-shape landmarks/axes/splines | subject shape | ROI-local, source-image, or fish-anatomical body frame as array-specific profiles | exact ROI/source extent or anatomical units | keypoint/mask row identity and body-frame derivation | pose/shape analysis, Crimson |
| chaser-distance position/distance surfaces | chaser analysis | both inputs in one validated frame; output px or typed mm | bound input extent/physical frame | exact stimulus/track identity and transform/calibration | behavioral analysis |

Array names in this table are discovery hints only. They are not authority.

## Live registry inventory

The final schema-v12 inventory opened the canonical live registry read-only at:

```text
/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

The initial and final registry fingerprints are both
`83cfd79dda4b79c8f2efe8eaf7e096fa85adbd7b838ee05d5e7d3572759678f9`;
the registry did not drift during the scan. All 300 dataset rows and all 188
recording rows were selected and represented. The 300 dataset rows name 300
distinct Zarr paths: there are no duplicate dataset keys, recording keys, or
Zarr paths. Eighteen portable/global dataset rows have no `recording_id`, no
dataset row names an unknown recording ID, and no recording row is omitted from
the dataset inventory.

The live registry does contain 20 duplicate recording-path groups involving 40
recording identities. Each is a July 1 or July 2 acquisition registered once
with a short arena ID and once with the experiment suffix such as
`_DefaultScreen` or `_GoodCopBadCop`. These require reviewed registry identity
reconciliation; they are not duplicate Zarr dataset rows.

Metadata inspection found 17,678 important coordinate surfaces, with zero
unclassified geometry candidates and no missing or unexpected selected row.
Under the strict future-normal contract:

- 7,660 surfaces are `ambiguous_fail_closed`;
- 10,018 surfaces are `missing_or_unreadable`;
- 173 archives are inspectable but remain ambiguous/fail-closed; and
- 127 archives are missing/unreadable.

The last category is explicit rather than silently partial: 126 reachable
archives across 81 recordings contain malformed Zarr JSON metadata, and one
registered source path is absent. The malformed-node evidence contains 460
instances where `attributes` is not a JSON object, 163 forbidden `NaN` values,
and 24 forbidden `Infinity` values. Because one malformed node makes a mixed
archive snapshot unsafe, the audit inventories the discovered geometry but
invalidates its migration classification. Coverage therefore correctly reports
`all_selected_dataset_scans_complete == false`; it also proves that all selected
rows are represented and identifies every reason the 127 scans cannot be
complete.

The most prevalent archive-level gaps are not proof of bad arithmetic. For
example, 299 of 300 dataset rows lack an exact persisted archive-to-registry
identity binding, 281 lack the required acquisition authority, and all
historical geometry predates the array-specific future contract. One surface
can emit several missing-field issues, so issue occurrence totals for space,
units, axes, extent, identity, and lineage must not be added as independent
scientific errors.

The highest-priority value-validation set is 203 historical offline
`positions_px` surfaces across 114 archives/recordings. All carry the explicit
`OFFLINE_CROP_SOURCE_RECONSTRUCTION_NUMERICAL_VALIDATION_REQUIRED` flag. There
are 209 `positions_px` and 209 `positions_mm` surfaces total; the six other
pixel surfaces do not match that historical reconstruction signature. The
exact 114-recording list and every surface path are in
`migration_manifest.jsonl` and `issue_summary.csv`. No coordinate surface is
eligible for automatic metadata-only backfill from metadata alone.

The normalized dry-run migration manifest has 36,671 targets:

- 300 archive targets;
- 17,678 coordinate-surface targets;
- 4,074 run targets;
- 14,430 dependent derived-surface targets;
- 188 recording targets; and
- one registry target.

Its most restrictive migration classes are 16,859
`ambiguous_fail_closed`, 19,623 `missing_or_unreadable_fail_closed`, 41
`registry_reconciliation_required`, and 148 hierarchy-only `no_change` records.
There are zero automatically applicable targets, zero proven safe metadata-only
backfills, and zero metadata-proven recomputation targets. A total of 14,616
targets carry `requires_numerical_validation`, mostly through dependency-risk
propagation; the direct scientific source set is the 203 pixel-position
surfaces above. A fail-closed class can therefore retain a numerical-validation
flag without implying that the stored numbers are known wrong.

The complete, integrity-bound artifacts are outside Git at:

```text
/tmp/palette-coordinate-audit-live-20260719-v12-y481r6/artifacts
```

They include `coverage.json`, `registry_snapshot.json`, `issues.jsonl`,
`issue_summary.csv`, `archive_summary.csv`, `targets.jsonl`, and
`migration_manifest.jsonl`. `artifact_manifest.json` reports complete
generation and binds the exact outputs with generation digest
`11e758c46bec71e96eeb027fc1c4d302eb964193e53baaa430b4b0d04dc575f9`.
The audit made no registry or archive changes.

A second metadata-only pass adds the aggregate views requested for fleet
triage without modifying the normalized artifact generation. The hardened
aggregate covers all 17,678 surfaces, the 4,074 persisted run contexts, and
232 archive-level/no-run contexts across 16 run-family buckets. Producer
method, method version, commit, software version, and run schema are copied
only from exact declared run metadata paths; conflicting declarations are
retained rather than resolved by precedence.

Independent review then found that the first supplemental writer did not
sufficiently validate its source manifest, input schema, output path, portable
recording counts, legacy `git_commit_hash` declarations, zero-surface archive
roots, or verifier arithmetic. Those defects are fixed in the working tree;
the old aggregate digest `f414f950...` is explicitly obsolete and must not be
cited. The regenerated aggregate has payload digest
`1f18de20c7e17f5562504146e0dd77d534ceacedf6a574ec6f22838121d82100`
and file SHA-256
`ce30d535e212b022a2dfa36ff509d25c5d2d17560bf53a5908741bf804ae29e3`.
The hardened verifier returns true, an independent rebuild reproduced the
payload digest, and all 300 selected dataset Zarr roots are retained in the
output guard. Focused pytest remains deferred by the execution-approval gate;
the live regeneration itself completed atomically. The primary schema-v12
inventory and its generation digest above are unaffected.

Across all 4,306 contexts, 3,324 have one resolved producer identity, 331
contain conflicting or invalid producer declarations, and 651 are unavailable.
The unavailable total comprises 312 readable persisted runs with neither a
method nor commit declaration, 107 invalid persisted run metadata records, and
232 archive-level/no-run contexts. At surface level those categories are
12,561 resolved, 2,469 conflicting, and 2,648 unavailable surfaces. The
hardened extraction conflict-checks 234 legacy `git_commit_hash` declarations;
that is why it exposes substantially more producer conflicts than the obsolete
candidate while reducing unavailable status by two contexts and six surfaces.
The aggregate contains 297 exact producer keys and issue-by-family, producer,
and recording lists for 183 surface-bearing registered recordings plus 37
portable/unbound surfaces. It retains 290 surface-bearing dataset roots and all
10 selected zero-surface dataset roots. The five registered recordings with no
discovered coordinate surface are:

- `2026-03-27T22-37-24Z_arena_1_Blindfish_Flash_OMR_Loom`;
- `2026-03-28T00-57-03Z_arena_1_Blindfish_Flash_OMR_Loom`;
- `2026-03-28T03-17-16Z_arena_1_Blindfish_Flash_OMR_Loom`;
- `2026_03_27_23_16_52_cam2010095_Blindfish_recording_only`; and
- `2026_03_27_23_16_52_cam2010096_Blindfish_recording_only`.

Their absence from producer aggregates is explicit; they remain represented
by the primary dataset/recording inventory.

The pose-model registry was also checked read-only because an array's
keypoint collection axis is not useful unless the exact model artifact is
bound to the same ordered pose schema. Six pose rows have materialized model
artifacts. This is a coordinate-identity compatibility result, not a judgment
of model accuracy:

| Registry run | Future-normal binding result | Exact reason |
|---|---|---|
| `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2` | compatible | model and manifest digests are present; manifest and registry agree on the ordered five-point schema |
| `omnifin0_cedar_shadow_v001_pose_20260304-010906_ac34f114` | compatible | exact three-point manifest and registry schema agree |
| `omnifin0_cedar_shadow_v001_pose_20260227-173634_26e4eb25` | compatible | exact three-point manifest and registry schema agree |
| `pose_cedar_shadow_filtered_gray_latest_traditional_a4c30ae1_v001_r001` | compatible | exact three-point manifest and registry schema agree |
| `omnifin0_cedar_shadow_v002_pose_20260330-113338_7eb0b4ed` | reject | registry shape is five keypoints but its populated ordered-label field contains only three labels |
| `omnifin0_cedar_shadow_v004_pose_20260208-163716_c9dc72f5` | reject | the hash-verified training manifest lacks an explicit skeleton identity |

Two additional pose training-run rows have no materialized model path and are
therefore not model-artifact candidates. No registry row, manifest, or model
was changed. Package defaults, a matching numeric `K`, and familiar label
order are deliberately insufficient compatibility evidence.

The exact inventory command was:

```bash
scripts/py -m fisheye.utils.audit_coordinate_contracts \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --output-jsonl /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/inventory.jsonl \
  --output-csv /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/inventory.csv \
  --output-markdown /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/report.md \
  --summary-json /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/summary.json \
  --checkpoint-dir /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/checkpoints \
  --artifact-dir /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/artifacts
```

The hardened supplemental aggregate was regenerated with:

```bash
scripts/py -m fisheye.utils.summarize_coordinate_audit \
  --inventory-jsonl /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/inventory.jsonl \
  --artifact-manifest /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/artifacts/artifact_manifest.json \
  --output-json /tmp/palette-coordinate-audit-live-20260719-v12-y481r6/artifacts/producer_run_family_aggregate.json
```

`verify_normalized_artifact_generation(...)` returned complete/integrity true
with the digest above. The deterministic audit suite passed 129/129 focused
tests outside the sandbox. The pre-hardening supplemental aggregate is
superseded as described above. The hardened implementation and deterministic
unit test file pass static compilation, focused Ruff, and diff checks; focused
pytest remains deferred because the execution approval gate is usage-limited.

The earlier `/nvme1/palette_registry.sqlite` figures were from a stale local
registry snapshot and are superseded by this section. They remain reproducible
only as historical audit evidence under
`/tmp/palette-coordinate-audit-20260719-v12-final-a`.

## Prioritized fleet remediation queue

1. **Stop creating new ambiguity.** Complete future-normal producer lifecycle,
   descriptors, identity/time lineage, and strict reader gates before applying
   any historical metadata plan.
2. **Reconcile archive/registry identity.** Review the 20 duplicate
   recording-path groups involving 40 IDs and establish exact persisted
   archive ownership before coordinate migration. A coordinate descriptor
   cannot safely bind an archive whose recording ownership is disputed.
3. **Numerically validate historical track positions.** Compare the 203
   `positions_px` surfaces in 114 archives to the exact selected persisted
   source-image centers and row mapping; then verify physical outputs against
   their exact calibration evidence. Classify each result as value-validated
   metadata backfill, recomputation, or fail-closed ambiguity.
4. **Triage malformed historical metadata.** Review the 126 archives with
   invalid JSON metadata and the one absent registered path. Repair or
   migration, if later authorized, must retain the exact malformed evidence;
   normal readers continue to fail closed.
5. **Migrate subject geometry as a vertical slice.** Bind raw masks, refined
   masks/contours, and subject shape to the same observation/crop lineage;
   convert ROI-local body origins/landmarks through the exact placement instead
   of relabelling them.
6. **Review registry hygiene separately.** Reconcile the 20 duplicate
   recording paths and the one absent dataset path only after human review of
   proposed registry operations. The live registry has no duplicate Zarr path
   or unknown recording-ID rows.

Each step produces a new immutable run or reviewed registry operation. None
authorizes in-place payload rewriting.

## Baseline strict reader cutover order

The baseline audit found one shared motion choke point and several bypasses.
This retained ordering explains the remediation sequence; the working-tree
status is recorded separately after final review so this table is not mistaken
for a description of the post-fix reader:

| Priority | Boundary | Current behavior | Required cutover |
|---|---|---|---|
| 1 | `analysis/track_kinematics_io.py` | Resolves `latest`, prefers grouped speed arrays, and silently falls back to flat legacy names; it does not validate an all-array motion seal | Resolve only selector-eligible canonical runs, fresh-validate the exact motion seal, and return typed bindings for every requested surface |
| 2 | Central-loader consumers: bout detection, stimulus response, chaser bearing/epoch summaries, plotting, and cross-recording export | Receive ordinary arrays through `load_track_kinematics_track`, so one permissive loader affects many scientific outputs | Keep kernels unchanged; require the typed loader and copy its authority/seal digest into each derived output lineage |
| 3 | Direct scientific bypasses: `megabouts_classifier_inputs.py`, `bout_kinematics.py`, `goodcopbadcop_common.py`, and `chaser_response_regimes.py` | Read `frame_indices`, positions, headings, validity, or speed directly; some invent all-valid rows or catch every exception and fall back to another signal | Replace direct reads with typed bindings; reject missing validity/identity evidence; make any alternative signal an explicit, provenance-bound mode |
| 4 | Diagnostic and visualization bypasses: `measure_noise_floor.py`, `interactive_track_kinematics.py`, and heatmap/inspection tools | Discover groups and arrays by name and often choose a latest/sorted run | Use the same strict loader by default; expose legacy inspection only through an explicitly labelled read-only compatibility command |
| 5 | Crimson | The external contract currently permits motion discovery without one exact all-array seal | Require a supported canonical space plus validated position/motion seal; transform only canonical source coordinates into ephemeral viewport coordinates |

Using `latest` is acceptable only as a selector lookup: the selected child must
still be eligible and freshly validate. A selector, path, array name, or grouped
layout is never coordinate or derivation authority. Historical adapters remain
available only to explicit migration/inspection tooling; future-normal readers
do not negotiate legacy layouts.

## Initial confirmed gaps, ordered by severity

The following findings describe the audited baseline that motivated the
working-tree remediation. They are retained as the evidence ledger, not as a
claim that every item remains unfixed. Current implementation and validation
status is recorded separately below after each independent review completes.

1. **Future publication could expose unvalidated coordinate claims.** Several
   writers historically marked a run complete or selector-eligible before a
   fresh complete-path validation. Normal producers must keep the child
   selector-ineligible, validate staging, mark complete, reopen and validate,
   write selectors while still ineligible, and flip eligibility last. All
   rollback paths must catch `BaseException` and restore exact prior selectors.

2. **Historical track positions can be numerically plausible but unprovable.**
   Commit `72f2e7f90860ebbd3ded12f94734004a677ebf75` selected crop
   `bbox_norm_coords` and multiplied centers by `resolve_dimensions(root)`.
   It did not bind the exact selected source-image center array, model-input
   transform, or crop placement. Current canonical publication instead requires
   an exact dtype-preserving subset/reorder of the selected persisted
   source-camera center surface.

3. **The normal chaser-distance writer still accepts an unlabelled transform
   fallback.** `analysis/chaser_distance_runs.py:656-677` obtains a generic
   homography through `shared/coordinate_transform.py:138-169`, applies it as
   camera-to-canvas despite that helper's projector-to-camera contract, and may
   obtain projector pixels/mm from an unbound median embedded in state rows
   (`chaser_distance_runs.py:279-291`). Group attrs declare the intended output
   after the calculation, but no bound input descriptor, active-camera
   authority, directed transform, or array-owned output descriptor proves it.
   This path must fail closed until it loads the exact canonical detection and
   stimulus evidence, applies the verified camera-to-arena direction, and seals
   the distance derivation. This is a new-write risk even when historical
   values happen to be correct.

4. **Normal readers still contain permissive legacy resolution.** Core motion,
   bbox, keypoint, and plot/analysis readers may select `latest`, infer from
   names, or accept flat legacy arrays without validating descriptors and row
   identity. Scientific kernels may remain array-based, but their input boundary
   must accept only sealed canonical evidence. Historical compatibility must be
   an explicit mode or migration tool, never the future default. The new typed
   track-position loader intentionally authorizes only `positions_px` and
   optional `positions_mm`; `track_kinematics_io.py` still exposes speed, path,
   heading, acceleration, time, and validity arrays without one exact persisted
   all-array payload/derivation seal. Strict scientific-reader cutover is
   blocked until that separate seal exists and freshly validates.

5. **Modern subject geometry is not yet a sealed coordinate-bound surface.**
   Raw and refined subject-mask rasters, centroids, boxes, contours, and
   subject-shape arrays are published without array-owned descriptors and may
   omit `instance_key`. Their normal readers still accept unmarked runs and
   infer layouts from array names and dimensions. In addition,
   `subject_shape_runs.py` labels body-frame geometry as `roi_pixels`, which
   conflicts with canonical body-frame v1 source-camera/mm publication. These
   values must be converted through exact crop placement, not merely relabelled.

6. **Registry identity and path hygiene obscure the true fleet.** Duplicate
   aliases, stale temp rows, missing paths, recording-ID disagreement, and
   uncontrolled roles must be resolved before using issue-row counts as a
   migration queue. Registry cleanup remains a separately reviewed operation;
   none was applied during this audit.

7. **Many downstream geometry arrays discard upstream semantics.** Masks,
   contours, subject-shape geometry, chaser-distance results, and historical
   refined surfaces often preserve numbers while dropping exact extent,
   identity, or transform evidence. Correct declarations cannot be copied from
   a nearby run or reconstructed from root dimensions.

8. **Two maintenance/training paths can bypass normal publication semantics.**
   `predict_training_detections.py` writes training-sample/model-image normalized
   boxes into selectable `detect_runs` without canonical observation identity or
   source-camera authority. `patch_crops_from_refined.py` rewrites ROI placement,
   normalized boxes, and pixels in an existing crop run in place, which can
   invalidate already-published descriptors and derivation digests. Training
   predictions must remain a typed nonselector artifact until explicit
   promotion, and canonical crop corrections must create a new derived run.

9. **Manual-edit contracts can invalidate an otherwise correct declaration.**
   The shared Crimson keypoint write contract currently requires row-wise
   in-place overwrites of a completed refined run
   (`contracts/palette-crimson/keypoint_manual_write.md:15-29`). Palette's
   in-tree review tool performs the same style of mutation in
   `tune/keypoint_failure_review.py:1172-1460` and refreshes additional heading
   fields after the UI exits. A descriptor or publication digest stamped before
   that edit would become stale. Normal readers fail safely when they detect the
   mismatch, but the edited run would cease to be a usable canonical source.
   Future manual editing therefore needs a controlled edit transaction: make the
   child ineligible, mutate only the declared editable authority, recompute every
   derived mirror, reseal, freshly validate, and reactivate—or publish an
   immutable successor when the stage contract permits it. Crimson must not
   independently mutate Palette scientific authority under a stale write
   contract.

## Branch-local remediation ledger

Nothing in this table is deployed, selected by production archives, or applied
to the live registry. `GO` means the scoped Palette producer/reader boundary
has passed its independent review and focused validation on this branch; it
does not mean the cross-repository contract or Crimson implementation is
released.

| Priority slice | Branch-local result | Remaining gate |
|---|---|---|
| canonical descriptor, frame, transform, identity/time, collection-axis, measurement, and atomic-publication primitives | implemented as one shared framework; future writers stamp compact array records that reference exact digest-bound evidence | combined final regression and release/version coordination |
| detection and crop point/bbox boundary | `GO`: separate continuous point and half-open pixel-edge camera/ROI authorities; projection and center records are v2; ordinary and incremental crop publication preserve both directed chains | outside-sandbox real-Zarr/full focused suite remains deferred; merge bbox contract and add Crimson fixtures |
| stimulus/refined-online boundary | `GO`: canonical source/texture/canvas evidence is preserved; physical outputs require typed calibration; unsupported future-normal inputs fail closed | authoritative consumer contract and cross-repository fixtures |
| track motion | `GO`: canonical source subset/reorder, physical authority, full-motion seal, strict reader, lifecycle activation, and scientific/visualization reader cutover reviewed; 144 focused tests passed | merge authoritative contract, implement Crimson reader, then run shared fixtures |
| raw/refined subject masks and subject shape | `GO` for the scoped publishers/readers: dense refined masks remain edit authority, collection identity is explicit, shape points are translated through exact placement, and vectors are not translated | merge subject-shape contract and implement external consumer; no production migration yet |
| raw keypoints and pose-model identity | canonical raw publisher and strict loader bind observation identity, acquisition time, ordered `keypoint` collection axis, exact model digest, and exact pose schema; same-cardinality reordered labels fail closed | final combined keypoint suite is deferred; refined-keypoint successor publication and Crimson remain pending |
| chaser distance | canonical writer and exact reader bind fish/chaser inputs, transform/calibration lineage, positions, and distances; normal scientific consumers were cut over | component/dashboard/export semantics are intentionally unavailable until independently sealed; deferred combined suite remains |
| fleet inventory and migration plan | complete read-only schema-v12 inventory and dry-run classification produced; the hardened supplemental producer aggregate was regenerated deterministically and protects all 300 selected dataset roots; no automatic backfill candidate and no write authorization | focused aggregate pytest remains approval-gated; perform human registry review and value validation before any migration |
| cross-repository contracts | branch-local track-motion and subject-shape contracts exist; bbox/keypoint/manual-edit contracts have partial edits | external contracts workspace write is currently approval-gated; reconcile final keypoint point/edge and model-binding text before release |
| Crimson | implementation work package drafted only | no Crimson code change is authorized; each family remains unavailable until its contract and hostile fixtures pass |

The principal numerical kernels were not rewritten. Changes concentrate at
selection, validation, transformation, publication, and reader-preflight
boundaries. Deliberate value changes remain limited to computations whose old
frame assumptions cannot be proved, canonical half-open bbox handling, exact
ROI-to-source point placement, and acquisition-frame gap handling described
above.

## Documentation, writer, metadata, and test conflicts

| Surface | Conflict confirmed by the audit | Required resolution |
|---|---|---|
| GoodCopBadCop homography | `docs/goodcopbadcop_coordinate_frame_workflow.md:198-224` says the historical helper name is not direction authority; `shared/coordinate_transform.py:4-9,172-198` still describes `projector_to_camera_px`, while `analysis/chaser_distance_runs.py:297-303` passes the active camera-to-canvas matrix through it | Use a direction-neutral numerical primitive behind a persisted `from_space_id`/`to_space_id` record; normal consumers validate the record and never infer direction from the helper |
| Historical offline motion | The production canary declares only run-level `coordinate_space = "camera"`; its `positions_px` array has no descriptor. Historical writer behavior reconstructed normalized crop centers using root dimensions | Keep the run historical/unverified until exact value and row-map validation; never relabel it from the run attr alone |
| Future-normal track writer and contract | The reviewed working tree initially emitted legacy `"camera"`, adapted it during sealing, and used `points_xy` in strict consumers even though the canonical observation producer publishes one row point as `point_xy` | New writers emit `source_camera_image_px` directly; normal sealing rejects legacy labels; one point per sample is `point_xy` end to end |
| Track motion tests and manifest | A fake IO descriptor used `points_xy`, masking the real producer-to-reader failure. `detection_indices` was declared public but could be forged and then resealed because its true ordinal source was not reconstructed | Require a real producer→seal→reader fixture; remove the unused ordinal from the future public schema or bind and validate an exact ordinal authority |
| Branch-local shared bbox read draft | The current contracts-worktree draft still declares source-image `bbox_img_xyxy` as `pixel_convention == "continuous"` (`contracts/palette-crimson/detect_bbox_read.md:119-127`), while the finalized Palette point/edge topology publishes boxes in the distinct `pixel_edge_half_open` frame | Before merge, change the contract and hostile fixtures to require `pixel_edge_half_open`, allow `x_max == width`/`y_max == height`, and forbid resolving bbox lineage through the continuous point frame |
| Shared keypoint read contract | `contracts/palette-crimson/keypoint_read.md:34-46,166-185,255-265` permits lexicographic fallback, normalization/display scaling, and ROI-to-image reconstruction from nearby dimensions | Normal Crimson reads require the selected array's descriptor, exact row identity, extent authority, and directed crop placement; keep fallback only in visibly historical inspection |
| Shared keypoint write contract | `contracts/palette-crimson/keypoint_manual_write.md:15-29,35-39` requires in-place mutation and reconstructs image/normalized mirrors with unsealed crop/dimension assumptions | Replace with a Palette-owned edit/reseal or immutable-successor protocol using the exact selected crop authority and transactional activation |
| Shared refined-detect write contract | `contracts/palette-crimson/refined_detect_manual.md:24-35,62-77` publishes selectable normalized boxes and pointers without source-camera boxes, row identity, descriptors, or a publication seal | Manual detections remain nonselector staging until Palette validates exact observation identity, source-camera half-open boxes, normalized mirrors, and complete publication |
| Subject shape | Historical `subject_shape_runs.py` published ROI-local point values while body-frame documentation described source-camera/mm publication; bbox maxima were also historically ambiguous between inclusive and half-open conventions | Transform points through exact persisted ROI placement, publish half-open edge boxes, and bind body origins/vectors to array-specific source-camera descriptors |

Line references above identify the reviewed 2026-07-19 snapshots. Working-tree
remediation is not considered complete until focused tests and a second
independent review confirm that the corresponding conflict is gone.

## Production canary

Metadata-only inspection was performed on:

```text
/groups/johnson/johnsonlab/jeremy/recordings/
  2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/
  2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr
```

Run:

```text
analysis/track_kinematics_runs/offline/
  goodcopbadcop_tk_hyst4_low2_latch_s005_defaults_20260708
```

Confirmed metadata facts:

- the run is complete and declares `coordinate_space = "camera"` only in
  provenance/parameters;
- its selected position source is
  `crop_runs/crop_2026-06-17_19-37-49`, kind `crop_rows`;
- track `positions_px` is float32 `[120221, 2]` and has no array-owned
  coordinate descriptor;
- the crop has 120,221 rows and declares a 4512×4512 external source frame;
- the analysis root also declares 4512×4512;
- crop `bbox_norm_coords` is `[120221, 4]` with no array attrs;
- its upstream refined instance group labels normalized boxes against a
  640×640 inference image while `bbox_img_xyxy` is labelled source-image
  4512×4512;
- the detector provenance records an explicit 640×640 pre-resize, but no
  digest-bound directed preprocessing transform;
- `positions_mm` uses the historical scalar
  `0.018788045498284132` mm/px rather than an array-bound typed physical frame;
- track, crop, and refined detection row counts agree.

The matching square source/inference aspect and exact row counts make the
existing numerical result plausible. Metadata alone does not prove equality to
the persisted source-image centers or prove row mapping. Per audit policy, no
production array payload was read. This run is therefore
`numerical_validation_required`, not `known_incorrect` and not safe for a blind
metadata-only relabel.

The same archive also exposes a migration incompatibility: its recording
manifest lacks full-stream width/height, and its manifest recording ID omits the
`_GoodCopBadCop` suffix present in the analysis-root ID. A hardened acquisition
import must fail closed until those authorities are reconciled; it must not copy
dimensions or identity from the analysis root by assumption.

## Canonical descriptor and controlled vocabulary

The normative framework is
[`docs/coordinate_metadata_framework.md`](../coordinate_metadata_framework.md).
Each array owns `coordinate_descriptor` and its SHA-256 digest. The descriptor
contains a controlled profile/space, geometry/components/units, origin and axes,
reference extent, pixel convention, row-identity reference, source-camera
overlay status, and typed lineage/frame references. Large records are stored
once and referenced by canonical path plus digest.

Legacy `camera` and `texture` labels remain migration inputs. Compatibility is
allowed only when an explicit rule binds exact dimensions, identity, and
lineage; it does not mint canonical descriptors or enter normal selectors.
Future recordings and writers emit canonical-v2 evidence only.

## Palette/Crimson boundary

Palette must persist and validate:

- array-owned descriptor and digest;
- exact frame and reference-extent records;
- exact row identity and acquisition-time mapping;
- direction-labelled crop/model/calibration/transform lineage;
- exact subset/reorder or derivation evidence;
- direct, transform-required, or unsupported source-camera overlay status.

Crimson may validate supported profiles, reach the source-camera overlay frame
using only the persisted ordered transform chain, and then create an ephemeral
viewport/display transform. Crimson must not infer a frame from `positions_px`,
run type, dimensions, numeric ranges, or historical labels, and must not create
missing millimetres with a scalar or resolution ratio.

The separately scoped Crimson implementation sequence and release gates are in
[`crimson_coordinate_implementation_work_package_2026-07-19.md`](crimson_coordinate_implementation_work_package_2026-07-19.md).
That handoff does not authorize Crimson changes; each reader remains unavailable
until its Palette producer, authoritative contract, and cross-repository hostile
fixtures pass independently.

## Focused tests required at every boundary

- swap ROI-local and source-image arrays with numerically plausible values and
  require rejection;
- reverse a homography/transform chain and require direction mismatch;
- alter reference W/H or the bound extent digest and require rejection;
- delete or stale one lineage record and require rejection;
- remove/mutate row keys or acquisition-frame mapping and require rejection;
- present an unsupported profile/space and require fail-closed behavior;
- interrupt with literal `KeyboardInterrupt` and `SystemExit` before and during
  activation and require exact selector restoration;
- validate zero-row outputs against a full-domain decode/no-observation proof;
- for historical tracks, compare positions to the exact selected source-image
  centers and compare physical outputs to the exact typed calibration before
  classifying a migration.

## Migration strategy

1. **Registry hygiene, read-only plan first.** Review duplicate-path groups,
   stale temp/test rows, role conflicts, and recording identities. Applying
   registry changes is separate from coordinate migration.
2. **Metadata-only backfill.** Allowed only when existing payloads and persisted
   records already prove one unique frame, extent, direction, identity, and
   lineage. The current strict fleet scan found no automatic candidates.
3. **Value validation followed by metadata backfill.** Where exact source arrays
   and transforms still exist, compare payloads and row mappings first. Only a
   passing, recorded validation may authorize descriptors on unchanged values.
4. **Recomputation.** Create a new derived run when values used guessed root
   dimensions, missing crop/model transforms, an incompatible calibration, or
   an unprovable row selection. Never rewrite the old array in place.
5. **Ambiguous/fail closed.** If the archive cannot prove a unique frame or
   lineage, retain it as a historical artifact and exclude it from normal
   selectors. Do not guess.

## Implemented `track_motion_read.md` contract changes

The branch-local cross-repository contract replaces permissive run/array
fallback with these exact requirements:

1. Require explicit canonical completion, `coordinate_binding_status ==
   "bound_canonical_v2"`, and `stage_selector_eligible == true` before normal
   selection. `latest*` never overrides validation.
2. Make `track_sample_key = (track_id,
   source_acquisition_frame_index)` primary row identity. Treat `frame_indices`
   only as an exact compatibility alias and `source_instance_key` as nullable
   observation lineage.
3. Require each `positions_px`/`positions_mm` array descriptor and every
   referenced digest. Define direct overlay, ordered transform-required overlay,
   and unsupported behavior.
4. Permit `positions_mm` and `*_mm` motion fields only with the exact compatible
   typed physical-frame authority. Forbid run-level scalar and resolution-ratio
   synthesis.
5. State that canonical offline positions are an exact source subset/reorder,
   not reconstructed normalized centers.
6. Require a digest-bound exact live payload/derivation inventory before
   exposing speed, path, heading, acceleration, time, or validity arrays. The
   position-binding gates alone authorize only `positions_px` and optional
   `positions_mm`.
7. Treat renderer viewport/display coordinates as ephemeral Crimson state.
8. Move `camera`/`texture`, flat-array, and `movement_runs` handling into an
   explicitly invoked historical compatibility section. It must never be the
   normal path for future recordings.
9. Fail closed on missing identity, extent, lineage, transform direction,
   unsupported profile, stale digest, or incomplete/ineligible publication.

The exact contract is now present at
`contracts/palette-crimson/track_motion_read.md` on the dedicated contracts
branch. Palette's producer and strict-reader review is green, including 144/144
focused tests. The contract has not been merged or pushed, and Crimson has not
been modified; Crimson implementation must still pass the shared hostile
fixtures before advertising future-normal motion support.
