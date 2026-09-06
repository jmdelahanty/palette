# Canonical Representation Propagation Trace: Detection to Exported Datasets — 2026-09-03

> Disposition added 2026-09-06: the [second opinion](review_wave_second_opinion_2026-09-04.md) supersedes claims that sealed
> tracking is unused or the legacy path is the only motion writer. Exact-path
> supplier loading need not use a family selector; missing `latest` does not
> authorize backfill. Maintained callers still need scoped admission checks,
> not a second upstream/human gate when their validated supplier is sufficient.
> This is a historical trace, not an all-stage acceptance result; see the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-03
implementation: specified-only
-->

**Date:** 2026-09-03
**Snapshot:** `d730470d` on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`
**Method:** four parallel read-only tracer agents, one per DAG segment (detection → refined detection → crop; crop → keypoints → tracking → track kinematics; crop → subject masks → successors → subject shape → chaser components; every training and analytics exporter). Each hop was classified by how the consumer locates its upstream, whether it binds the upstream digest into its own envelope, whether it verifies that digest before reading arrays, and whether it instead copies raw arrays by run name. Synthesis re-read the lines marked **[VERIFIED]**. No tests run, no live stores opened. Other line numbers are from agent reading at this snapshot and **must be re-verified before implementation**.
**Question asked:** are unified canonical representations (canonical detection manifest, crop-v2 manifest, coordinate frame records, digest-bound record refs, subject-mask core manifests) propagated by digest through the DAG all the way to exported datasets, or does some stage re-derive from raw arrays by name?
**Governing contract:** [`../detection_artifact_and_canonical_publication_boundary.md`](../detection_artifact_and_canonical_publication_boundary.md) §Consumer rule: every modern consumer binds one exact `detect_runs/<run>` and its manifest payload digest; "latest" lookups are not a production policy; the rule is transitive through crops, pose, masks, tracking, analytics, and training/export publication. Its implementation checklist shows 23 of 23 items ticked.
**Companion evidence:** [`receipt_builder_census_2026-09-02.md`](receipt_builder_census_2026-09-02.md) (what each envelope binds); [`provenance_chain_review_2026-09-01.md`](provenance_chain_review_2026-09-01.md) (digest grammars; chain root and figure ends); [`training_data_and_model_provenance_review_2026-09-01.md`](training_data_and_model_provenance_review_2026-09-01.md).

**Queue disposition:** audit evidence and checklist source. Every item here touches a resolver or a publisher and therefore adopts into [`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md); none should be started before wave-1 Package A (fail-closed gate) lands, because a digest-bound consumer that refuses is only useful if the refusal is not swallowed. This document does not track status.

---

## 1. Verdict

**The canonical chain is digest-bound and fails closed from detection through crop-v2, and again through the subject-mask stack to subject shape. It breaks at four places, and every one of them is a `latest` or sorted-last lookup that the boundary contract says does not exist.** The contract's checklist is fully ticked; the code does not meet its consumer rule.

The four breaks, in DAG order:

1. **Refine-detect.** `refinement/refine_detect.py:1485` resolves the detection run from the family's `latest` attr when no run is passed, and `require_active_canonical_source` defaults to `False` at `:1355` **[VERIFIED]**. The refined working run records its source by name; the digest it did verify is buried in a provenance parameters blob. Finalization re-binds by name.
2. **Keypoints → tracking → track kinematics.** Two lineages coexist. The digest-bound lineage (keypoint publication, refined keypoint manifest, body frame, subject position, single-subject tracking) is correct hop by hop and then terminates at the tracking run. The only writer of track kinematics is the legacy lineage: keypoint inference resolves its crop via `latest_any`/`latest`/`latest_materialized` (`shared/crop_image_source.py:164` **[VERIFIED]**), refine-keypoints and arena assignment read `latest`, and track kinematics picks its tracking run by attr scan with a sorted-last tie-break (`tracking/single_subject_per_arena.py:595` **[VERIFIED]**). The sealed tracking manifest and its verifier exist; track kinematics and its materializer reference neither (zero hits **[VERIFIED]**).
3. **Subject-mask refinement input and chaser components.** Refinement resolves the raw mask run and its crop via `latest` (`tune/refined_subject_mask_review.py:1343,1368-1370` **[VERIFIED]**). Chaser bout response and escape events cite the distance run by name and path only (`analysis/chaser_bout_response.py:1249-1257` **[VERIFIED]**) and fall back to `resolve_authoritative_run_name` then sorted-last.
4. **Every training exporter and every non-chaser cross-recording table.** None of the four training exporters opens a run manifest. Detect export hashes the prepare-manifest *file* (`utils/export_detect_training_zarr.py:301` **[VERIFIED]**). Keypoint export falls to `latest` (`:1298-1305` **[VERIFIED]**). Subject-mask export falls to `latest` then sorted-last (`:144-153` **[VERIFIED]**). Cross-recording analytics resolves non-chaser tables with `fallback_to_latest=True, fallback_to_sorted="last"` (`utils/export_cross_recording_analytics.py:973-979` **[VERIFIED]**). The trainers then record only file hashes, so the training zarr's own attrs never reach the model row.

What is right, and should be protected: the artifact-first native detection path, canonical publication, refined-detection authority activation, crop-from-refined, the whole subject-mask core/bundle/successor/shape stack, subject-position sources, single-subject tracking materialization, and the kinematics/spatial-bins/tail-trace/chaser analytics exports. Those hops recompute array digests before use and refuse when the upstream manifest is absent.

**The shape of the defect.** Wherever a stage was rewritten under the coordinate framework, it is digest-bound. Wherever the old writer survived as "the thing that actually produces the output," the chain runs through it by name. The digest-bound lineage and the name-bound lineage produce different representations of the same object, and the downstream consumer reads whichever one exists.

---

## 2. What "canonical" means here

| Object | Produced by | Digest | Consumers must bind |
|---|---|---|---|
| canonical detection run `detect_runs/<run>` | native canonical publication | `run_manifest.payload_digest`; parent attr `CANONICAL_DETECTION_AUTHORITY_DIGEST` | manifest digest + `logical_content.digest` |
| coordinate frame records (pixel-frame authority, observation geometry, calibration, anatomical body frame) | stamped by producers | `<attr>_sha256` twin; `DigestBoundFrameRecordRef{record_ref, record_sha256}` | record pointer with sha |
| crop-v2 run manifest | crop snapshot publication | `payload_digest`, `logical_content.digest`, `coordinate_catalog_digest` | `source_crop_snapshot` block |
| keypoint / refined keypoint / body frame manifests | keypoint bundle production | `payload_digest`, `metadata_declarations_digest` | `source_manifest_digests` |
| subject-position manifest | subject position sources | `manifest_sha256`, decoded content sha | tracking manifest `source` record |
| tracking run manifest | single-subject tracking | `tracking_run_manifest_sha256` | (nobody, today) |
| subject-mask core / bundle / successor manifests | mask stack | `manifest_payload_digest`, `logical_content_digest`, bundle `authority_digest` | `coordinate_dependencies` |
| chaser component manifest | component publication | `component_manifest_sha256`; base `publication_seal_sha256` | handle or selector |
| track-kinematics publication | track kinematics | `payload_integrity_receipt`, manifest digest attr | analytics exports |

A hop is **bound** when the consumer stores the upstream digest in its own envelope. It is **verified** when it recomputes that digest from the upstream artifact before reading arrays. It is **broken** when it locates the upstream by `latest`, sorted order, or an attr-copied run name and binds no digest.

---

## 3. Hop table

Legend: locate = how the upstream is found · bind = upstream digest stored in consumer's envelope · verify = recomputed before use · **B** = break.

### 3.1 Detection → refined detection → crop

| Hop | Consumer | Locate | Bind | Verify | Legacy path |
|---|---|---|---|---|---|
| artifact runs → canonical candidate | `utils/assemble_clipped_native_detection.py:117-144` → `detection/native_canonical_candidate.py:166-219` | explicit artifact paths from plan | ✓ members digest, model sha, frame/pixel authority refs | ✓ one model digest across artifacts | artifact rows never selector-eligible |
| candidate → `detect_runs/<run>` | `analysis_workflows/native_canonical_detection_publication.py:286-358` | explicit + receipt cross-check | ✓ parent authority digest attr | ✓ array digests `:693`; authority records rehashed `:379-401` | none |
| `detect_runs` → detect quality | `refinement/detect_quality_collection.py:841-846` | explicit + exactly-one-of digest/receipt | ✓ `source_detect_run_manifest_digest` | manifest-level only ("without reopening arrays") | only when flag off (default off) |
| `detect_runs` → refine-detect **B** | `refinement/refine_detect.py:1485` | **`attrs['latest']`** when no run passed; flag default `False` `:1355` | ~ name attrs `:1823-1825`; digest only inside provenance params `:1802` | manifest-level `:1496` | yes by default |
| `detect_runs` → registered detection gate | `analysis_workflows/materializers/registered_detection_gate.py:276-340` | explicit | ✓ `canonical_run_manifest_payload_digest` in signature | manifest-level + changed-during-preflight | tolerated: `manifest_digest=None` fallback `:163-186` |
| working refined → finalized refined **B** | `utils/finalize_recording_refined_detection_v1.py:370` | explicit names; canonical by digest | ✓ `source_detection{run_manifest_digest, logical_content_digest}` | canonical: array-level `:124`; working run: **name equality only** | no |
| finalized refined → authority activation | `shared/zarr/refined_detection_authority_activation.py:195,325` | explicit; proof compares digest before/after | ✓ | ✓ recomputes logical content `:188-197` | no |
| refined authority → crop-v2 | `shared/zarr/refined_detection_crop_source.py:274-300` via `crop_snapshot_publication.py:581` | parent authority attr + envelope | ✓ `CropRefinedSourceIdentity` | ✓ arrays + evidence paths | no |
| `detect_runs` → ordinary materialized crop | `tracking/crop.py:697,806,890,2898` | `'auto'/'refined'/latest` resolver `:3401-3414`, then preflight requires exact run `:682` | ✓ `source_detection_manifest_digest` + name triple in `crop_signature` | manifest-level; frame records re-verified `:806-815` | legacy `crop_detections` resolver still lists filtered/interpolated/manual surfaces |
| clip evidence, snapshot successor, shadow, sampled training seed | `clipped_detection_evidence.py:92-121`, `detection_snapshot_publication.py:360-430`, `canonical_detection_shadow.py:423-490`, `sampled_training_detection_publication.py:750-816` | explicit | ✓ | ✓ array-level | shadow is the only legal legacy reader |

Refined runs carry **no** coordinate frame records; they bind source only through the refined manifest's `source_detection` block.

### 3.2 Crop → keypoints → tracking → track kinematics

Two lineages. **A** (legacy): keypoint inference → refine-keypoints → arena assignment → track kinematics. **B** (digest-bound): keypoint publication → refined keypoint manifest → body frame → subject position → single-subject tracking. B stops at tracking. A is the only track-kinematics writer.

| Hop | Consumer | Locate | Bind | Verify | Legacy path |
|---|---|---|---|---|---|
| crop → keypoint inference (A) **B** | `detection/detect_keypoints_yolo.py:2023` via `crop_image_source.py:164` | **`latest_any` → `latest` → `latest_materialized`** unless `--crop-run` | ✗ `source_crop_run` name + `crop_signature`/`crop_revision` (`provenance_attrs.py:116-120`) | canonical mode only (`:2069,:2087`) | yes, silently: snapshot keys omitted when absent |
| crop → keypoint publication (B) | `shared/zarr/keypoint_publication.py:109,553-557`; bundle `:175-214` | explicit crop run id | ✓ `source_crop_snapshot` (5 digests) | ✓ recompute + equality with persisted crop manifest | raises |
| keypoints → refine-keypoints (A) **B** | `refinement/refine_keypoints.py:1162` | **`keypoints_runs.attrs["latest"]`** | ✗ name + copied crop attrs `:1250,:1257` | ✗ | this is the only path |
| keypoints → refined keypoint manifest (B) | `shared/zarr/refined_keypoint_manifest.py:768-772,921` | chain-supplied manifests | ✓ raw/quality/crop digests | ✓ three recomputes + cross-binding | raises |
| refined → body frame (B) | `shared/zarr/body_frame_manifest.py:247-262` | chain-supplied | ✓ | ✓ | raises |
| keypoints / `detect_runs` → subject position (B) | `shared/subject_position_keypoint_source.py:292-376`; `subject_position_detection_source.py:171-247` | explicit path **and** must equal `latest`/`latest_complete`; latest forbidden as input | ✓ four digests | ✓ | raises |
| subject position → tracking (B) | `analysis_workflows/materializers/single_subject_tracking.py:185-343` via `subject_position_source_handle.py` | explicit ("never resolves latest") | ✓ manifest + decoded content sha | ✓ | raises |
| detection rowset → tracking (A) **B** | `tracking/arena_assignment.py:985-1015` | **`refined_detect_runs.attrs['latest']` else `detect_runs.attrs['latest']`** | ✗ names + rowset fingerprint | fingerprint only | yes |
| tracking → track kinematics **B** | `analysis/track_kinematics.py:14079` → `single_subject_per_arena.py:531-596` | **attr scan on source names; tie → `latest` else `matches[-1]`** `:595` | ✗ `tracking_path` string `:14323` | ✗ manifest never referenced; `tracking_source_handle.py:459` has no caller here | yes: needs only `track_ids` |
| keypoints → track kinematics heading **B** | `track_kinematics.py:1144,1152` | **`refined_keypoints_runs.attrs["latest"]` else `keypoints_runs.attrs["latest"]`** | ✗ names `:14316-14322` | ✗ no completion or manifest check | yes |
| crop position surface → track kinematics | `track_kinematics.py:1594-1652` | crop run from keypoint attr | ~ path strings | ✓ record `content_sha256` equality | raises |
| track kinematics own payload | `:12005,:12484`; materializer `:647-775` | — | ✓ integrity + validation receipts | ✓ | — |

### 3.3 Crop → subject masks → successors → subject shape → chaser

| Hop | Consumer | Locate | Bind | Verify | Legacy path |
|---|---|---|---|---|---|
| refined detection → crop-v2 | `shared/zarr/crop_manifest.py:118-241` | explicit | ✓ `source_refined_snapshot` | ✓ logical content recompute `:1231` | rejected |
| crop → raw mask inference | `segmentation/infer_unet_subject_masks.py:3494-3597` | `--crop-run` else `latest_any`/`latest`/`latest_materialized` | ✓ scientific identity: crop manifest ref, ROI sha, decoded pixel sha, row shas | canonical path ✓ (`:3573`); **non-canonical path copies manifest ref unverified** `:1838-1847` | non-canonical branch accepts crop with no manifest; copies crop arrays and `detection_source` by name `:3712-3742` |
| crop + camera frame → raw coordinate publication | `shared/subject_mask_coordinate_publication.py:1376-1450` | producer-passed path | ✓ ROI pixel-frame authority, crop placement, inference authority, model fingerprint | ✓ stamp-then-reload | canonical only |
| raw mask → refinement input **B** | `refinement/finalize_subject_masks.py:5949-6036`; `tune/refined_subject_mask_review.py:1343,1367-1370` | **`latest` for raw run; crop by attr name then `latest`** | ✓ `input_binding` + `source_crop_snapshot` | validates identity record; **does not rehash raw arrays**; `require_production_proof=False` → identity v1 fallback `:6026-6036` | yes |
| raw surfaces → refined coordinate publication | `shared/refined_subject_mask_coordinate_publication.py:1419-1471` | passed path; attr must equal | ✓ record pointers for context, inventory, derivation, inference authority, row identity | ✓ `bind_persisted_coordinate_record` | canonical raw required |
| raw + refined + crop-v2 + receipt → core | `shared/zarr/subject_mask_core_publication.py:380-586` | explicit; crop path must equal manifest run id | ✓ `coordinate_dependencies` (crop, assembly receipt, raw core, assignment keypoints) | ✓ rehashes live row arrays vs crop logical content `:428-451` | crop-v2 required |
| cores → bundle → authority | `subject_mask_bundle_publication.py:660-690,1866-1885`; `subject_mask_bundle_coordinate_authority.py:241-432` | explicit run ids; authority by activated `bundle_id` only | ✓ | ✓ live bundle before activation | none |
| cores → successor | `subject_mask_coordinate_successor.py:571-700` | explicit | ✓ source manifest digests | ✓ persisted core; model bytes rehash | ~ falls back to producer `run_provenance` attrs for model artifact `:571-586` |
| bundle authority → subject shape | `shared/zarr/subject_shape_bundle_source.py:137-262`; materializer `subject_shape.py:314-405` | `bundle_id` (bundle path) or `resolve_refined_subject_masks_run` (historical path `:377`) | ✓ authorities, camera record shas, assignment keypoints digest | ✓ | historical path accepts un-bundled refined run; both then read masks by run name `:401-408` after the check |
| `detect_runs` → chaser distance | `analysis/chaser_distance_coordinate_publication.py:597-680` | `--detection-path` explicit; legacy `detect_runs.latest` | ✓ manifest digest, row identity sha, calibration sha | ✓ rehashes centers/frame index/scores | legacy writer stores `source_detection_path` name only `chaser_distance_runs.py:1567-1574` |
| chaser distance → bout response / escape events **B** | `chaser_bout_response.py:942,1249-1257`; `chaser_escape_events.py:388,1170-1174` | `load_chaser_distance_run("latest")` → `authoritative_run`/`latest_complete` | ✗ **name and path only** | distance loader verifies its own seal; digest not written into lineage | n/a |
| component → component **B** | bout response `:324-360`; escape events `:229-270` | dependency handle, **or `resolve_authoritative_run_name` → `latest` → `sorted(keys)[-1]`** `:261-266` | ✓ with handle; `None` otherwise | handle path only | yes |

### 3.4 Exports

| Exporter | Locate | Bind | Verify | Absent manifest |
|---|---|---|---|---|
| `utils/export_detect_training_zarr.py` **B** | prepare manifest → `resolve_authoritative_run_name` (selector) | ✗ `source_manifest_sha256` = sha of the **prepare-manifest file** `:301`; zarr paths | ✗ (`run_manifest`: 0 hits) | proceeds |
| merged detect export (frame decisions) | selected refined run name | ~ `source_frame_decisions[].digest` **minted** from arrays read live (`detection_frame_supervision.py:118-130`) | ✗ minted, not compared | falls to non-strict branch |
| `sampled_training_detection_publication.py` | explicit artifact run | ✓ seal sha, lineage sha, model run | ✓ | fails — **but only caller is the canary; detect export never uses it** |
| `utils/export_keypoint_training_zarr.py` **B** | manifest `keypoint_run` else **`kp_parent.attrs["latest"]`** `:1298-1305`; refined run from `source_bindings` | ✗ only crop `source_binding_digest` copied from attrs `:295`; envelope = manifest file sha + paths | ✗ refined `payload_digest` never bound | proceeds ("Normalize legacy flat") |
| `utils/compose_task_keypoint_training_manifest.py` | explicit | ~ copies `payload_digest` as `reviewed_artifact_receipt_digest` `:161` | ✗ never recomputed | fails |
| `utils/export_acquisition_crop_pose_training_zarr.py` **B** | explicit → `resolve_authoritative_run_name` → `latest` `:334-337` | ✗ paths | ✗ | proceeds |
| `utils/export_subject_mask_training_zarr.py` **B** | explicit → **`latest` → sorted-last** `:144-153`; crop likewise `:168-181` | ✗ paths + run names; approval gate reads attr, bypassable `:345-347` | ✗ | proceeds |
| trainers → `training_runs` row | CLI paths | ✗ config/manifest/model/metrics **file** hashes only (`registry/db.py:6844-6847`); `training_export` attrs never reach the row | ✗ | n/a |
| `analytics_exports/kinematics_samples.py` | explicit dependency run from materializer plan | ✓ `source_manifest_sha256` | ✓ digest attr == canonical hash `:645-647` | fails |
| `activity_spatial_time_bins.py` | explicit per-track map | ✓ track + swim-bout manifest shas, cross-bound | ✓ | fails |
| `tail_trace_samples.py` | explicit child names | ✓ tail + shape record shas, per-array content sha | ✓ | fails |
| `eye_trace_samples.py` | explicit; `"latest"` → `latest_complete` | ~ `manifest_digests` **minted** from live attrs `:267-276` | ✗ minted | fails on layout |
| cross-recording chaser tables | `chaser_export_authority_set` file | ✓ seal, component manifest sha, handle sha | ✓ | fails |
| cross-recording **all other tables** **B** | `_latest_run` → `resolve_zarr_run(fallback_to_latest=True, fallback_to_sorted="last")` `:973-979` | ✗ `source_zarrs` paths; lineage rows carry run names | ✗ | proceeds |
| frozen cohort → release | registry rows | ✓ member rows + query sha; **no zarr content digest** | path-set equality with chaser file only | n/a |

---

## 4. Findings across segments

**F-1. The contract's consumer rule is unmet at four hops it explicitly names.** Refinement, crops (the ordinary publisher), pose, tracking, and training/export publication are listed; refine-detect, keypoint inference, arena assignment, track kinematics, mask refinement input, and all four training exporters resolve by `latest`. The 23-of-23 ticked checklist measures publisher-side work, not consumer-side compliance.

**F-2. Two lineages per object, and the consumer reads whichever exists.** Raw keypoints: v2 manifest vs attr-only. Refined keypoints: v2 manifest vs v1 name attrs. Crop identity: `source_crop_snapshot` digest block vs `crop_signature`/`crop_revision`. Tracking source: subject-position manifest sha vs detection-rowset fingerprint (`resolve_tracking_run` matches only the fingerprint fields and ignores the manifest sha). Chaser lineage: dependency handle vs `resolve_authoritative_run_name`. Detection source in masks: canonical manifest vs copied `detection_source` array. Every pair is a place where a verified representation and an unverified one describe the same bytes, and the unverified one is on the production path.

**F-3. The digest-bound lineage terminates before the science.** Lineage B is correct through single-subject tracking and then nothing reads it. Track kinematics, the root of every kinematics-derived analysis and export, is fed by lineage A. Analytics exports downstream of track kinematics are digest-bound to it, so the chain from figure back to track kinematics is sound, and from track kinematics back to detection it is names.

**F-4. Refined runs carry no coordinate frame records.** Canonical `detect_runs` stamp `DigestBoundFrameRecordRef`s and crops re-verify them. Refined detection binds source only through its manifest's `source_detection` block, so the frame-authority chain skips a hop.

**F-5. Verifiers are written and then not called.** `tracking_source_handle.py:459` (tracking manifest), `sampled_training_detection_publication.py` (detect training seed, canary-only), `validate_frozen_cohort` (not called by release). Three complete verifiers with zero production callers. This is the same pattern as the completion gate: the primitive exists, the call site does not.

**F-6. Minted digests.** Merged detect export and eye-trace export compute a digest from what they just read and store it as if it were an upstream identity. A minted digest describes the read, not the publication, and cannot detect substitution.

**F-7. Exports are the weakest layer, and training exports are weaker than analytics exports.** Four of five analytics exporters are digest-bound; zero of four training exporters are. A model's provenance therefore ends at a file hash of a prepare manifest, two hops short of any zarr content.

**F-8. "Latest" is not one thing.** The tracers met `latest`, `latest_any`, `latest_materialized`, `latest_complete`, `authoritative_run`, `resolve_authoritative_run_name`, `resolve_zarr_run(fallback_to_sorted="last")`, `_resolve_latest_run`, `resolve_tracking_run` attr scan, and `sorted(keys)[-1]`. The architecture review counted nine resolvers; this trace found the same nine on the canonical path itself.

---

## 5. Checklists

Every item touches a resolver or publisher: adopt into the authority queue, start after Package A.

### 5.1 D: detection → crop

- [ ] **D-1** `refine_detect.py`: default `require_active_canonical_source=True` (`:1355`); delete the `attrs['latest']` fallback (`:1485`); write `source_detect_run_manifest_digest` as a first-class attr beside `:1823`, mirroring `detect_quality_collection.py:1026`.
- [ ] **D-2** `finalize_recording_refined_detection_v1.py:370`: require `working.attrs["source_detect_run_manifest_digest"] == canonical["payload_digest"]`, not name equality.
- [ ] **D-3** `registered_detection_gate.py:163-186`: remove the manifest-less branch; the gate signature always carries the canonical digest.
- [ ] **D-4** `crop_signature.py`: make `source_detection_manifest_digest` required and drop the name triple, or retire `build_crop_signature` in favour of `crop_manifest` on the ordinary publisher (`tracking/crop.py:2898`). Record which.
- [ ] **D-5** Stamp coordinate frame records (or bound refs to the source's) on refined detection runs so the frame-authority chain does not skip the refined hop.
- [ ] **D-6** `detect_quality_collection.py:841`: same default flip as D-1.

### 5.2 K: keypoints → tracking → track kinematics

- [ ] **K-1** `single_subject_per_arena.py:613` (`load_tracking_ids`): replace `resolve_tracking_run` with `load_tracking_source_handle(..., expected_manifest_sha256=...)` (`tracking_source_handle.py:459`); `track_kinematics.py:14323` persists `source_tracking_manifest_sha256` beside `tracking_path`; provenance recompute at `:5859` and `_lineage_refs` at `:995` compare the digest. **This is the single highest-value edit in the doc**: it closes the only hop whose sealed digest already exists and is never read, and it puts every kinematics-derived export on a digest chain back to subject position.
- [ ] **K-2** `track_kinematics.py:1096-1134` (`resolve_keypoint_group`): require the keypoint or refined-keypoint run manifest attr, run `validate_keypoint_run_manifest`, record `manifest_digest` in `offline_inputs`; delete the `latest` fallbacks at `:1144,:1152`.
- [ ] **K-3** `detect_keypoints_yolo.py:2023`: require `--crop-run` or a crop manifest ref; delete the `latest_any`/`latest`/`latest_materialized` chain at `crop_image_source.py:164` from the canonical path; bind `source_crop_snapshot` (digest block) instead of `crop_signature`.
- [ ] **K-4** `refine_keypoints.py:1162`, `arena_assignment.py:985-1015`: these are labelled diagnostic/non-selector writers (`refine_keypoints.py:1141-1144`). Either retire them as canonical producers, leaving `keypoint_bundle_production_publication.py` + `single_subject_tracking.py` as the only writers, or bind digests. Record the decision; do not do both.
- [ ] **K-5** `resolve_tracking_run` (`:551-585`): match on `source_subject_position_manifest_sha256` when present, never on name fields alone; remove `matches[-1]`.
- [ ] **K-6** Ratchet: no new `attrs["latest"]` / `attrs["latest_any"]` read in `analysis/`, `tracking/`, `refinement/`, `detection/` outside the resolver module the authority queue designates.

### 5.3 M: masks → shape → chaser

- [ ] **M-1** `finalize_subject_masks.py:6026-6036`: `require_production_proof=True` becomes the only path; delete the identity-v1 fallback.
- [ ] **M-2** `refined_subject_mask_review.py:1343,1367-1370`: resolve the raw run explicitly and the crop from `canonical_surfaces.context.source.crop_path`; delete both `_resolve_latest_run` calls.
- [ ] **M-3** `infer_unet_subject_masks.py:3712-3742`: gate the non-canonical branch behind an explicit `legacy_noncanonical` flag so `crop_manifest_reference` can never be `None` on a normal run.
- [ ] **M-4** `chaser_bout_response.py:1249-1257`, `chaser_escape_events.py:1170-1174`: add `source_chaser_distance_publication_seal_sha256` from the loaded distance run (`chaser_distance_io.py:309-310`); make the dependency-handle path mandatory and delete the `resolve_authoritative_run_name` / `sorted(keys)[-1]` fallbacks at `chaser_bout_response.py:344`, `chaser_escape_events.py:261-266`.
- [ ] **M-5** `subject_mask_coordinate_successor.py:571-586`: read the sealed inference-authority record, not `run_provenance` attrs.
- [ ] **M-6** `chaser_distance_runs.py:1560-1575`: legacy writer path binds the detection manifest digest or is removed.
- [ ] **M-7** `materializers/subject_shape.py:377-408`: historical path reads masks by run name after the digest check; bind the run to the verified surface record so the read cannot drift from the check.

### 5.4 X: exports

- [ ] **X-1** `export_detect_training_zarr.py:209`: require the refined run's `run_manifest`, validate with the canonical-v3 validator, write `payload_digest` as `training_export.source_detection_manifest_sha256`; refuse when absent. Merged path: bind the stored refined digest, not the minted decision digest.
- [ ] **X-2** `export_keypoint_training_zarr.py:1298`: delete the `latest` fallback; read the refined manifest via `refined_keypoint_snapshot_identity_from_manifest` (`refined_keypoint_manifest.py:557`), verify `payload_digest`, store `source_refined_keypoint_manifest_sha256` per spec in `training_export` (`:2656`).
- [ ] **X-3** `compose_task_keypoint_training_manifest.py:30-33`: recompute `canonical_json_sha256(publication["payload"])` and compare before copying at `:161`.
- [ ] **X-4** `export_acquisition_crop_pose_training_zarr.py:334-337`: drop the `latest` fallback; bind the keypoint manifest digest in root attrs.
- [ ] **X-5** `export_subject_mask_training_zarr.py:144-153,168-181`: drop `latest` and sorted-last; require the core manifest; bind `payload_digest` in `training_export` and `input_run_ids`; remove the approval-gate bypass at `:345-347`.
- [ ] **X-6** `training_run_shared.py:57` + `registry/db.py:6807`: persist `training_zarr_export_sha256 = canonical_json_sha256(root.attrs["training_export"])` on the `training_runs` row. Pairs with M-4 in the training review.
- [ ] **X-7** `eye_trace_samples.py:267-276`: bind a stored publication digest from the eye-angle stage (add one at publish time if absent); stop minting.
- [ ] **X-8** `export_cross_recording_analytics.py:931-979`: `_latest_run` refuses `fallback_to_sorted`; non-chaser tables go through the same authority-set pattern as chaser tables; write per-source manifest digests beside `source_zarrs` (`:6246`).
- [ ] **X-9** `cohorts/registry.py:716-740`: members carry a per-recording content digest (track-kinematics publication digest is the natural one) so `manifest_sha256` depends on content, not path. `release.py` calls `validate_frozen_cohort`.
- [ ] **X-10** Route `export_detect_training_zarr` through `sampled_training_detection_publication.py`, which is already digest-bound and fails closed, instead of leaving it canary-only. Pairs with X-1.

### 5.5 Sequencing

| Order | Items | Why |
|---|---|---|
| first, after Package A | **K-1**, X-6, X-3 | K-1 is one call-site swap that closes the largest gap; X-6 and X-3 are pure additions |
| second | D-1, D-2, D-6, M-1, M-2, M-4 | flag flips and fallback deletions on the production path; each needs a store sweep to confirm no live run depends on the fallback |
| third | X-1, X-2, X-4, X-5, X-8, X-10 | exporters, once their upstreams (second row) are digest-bound |
| fourth | K-2, K-3, K-5, M-3, M-5, M-6, M-7, D-3, D-4, D-5, X-7, X-9 | remaining dual representations |
| last | K-4 decision, K-6 ratchet | after the writers are down to one per object |

Before any second-row item: run a read-only store sweep counting, per family, how many runs on the production path were located via a fallback (the registry's `run_name` vs each resolver's answer, P-10 in the architecture review). That number is what the flag flips will break.

---

## 6. What was not assessed

- Whether any live run on the store was actually produced through a fallback path. Code-only; the sweep above answers it.
- The stimulus-epoch and swim-bout families' own upstream binding (they appear here only as export sources).
- The realtime path.
- Whether the two representations in F-2 have ever disagreed on a live archive. If they never have, the fixes are cheap; if they have, some published results rest on the unverified copy.
