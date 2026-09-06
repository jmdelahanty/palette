# Receipt Builder and Verifier Census — 2026-09-02

> Disposition added 2026-09-06: the universal envelope module and mass builder
> migration below are historical proposals, not an agreed interface or executable
> dependency. The [second opinion](review_wave_second_opinion_2026-09-04.md) documents existing digest-bound review and
> stronger publication paths omitted by universal absence claims here. Share
> mechanics while preserving each family's scientific validators, authority
> rules, and stable digest grammar. See the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-02
implementation: specified-only
-->

**Date:** 2026-09-02
**Snapshot:** `ea03eb76` on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`
**Method:** four parallel read-only census agents, split by stage family (import + detection/tracking runs; subject mask + successor + shape; chaser components → export → release → figures; training exports + models + the promotion gate). Each agent reported per builder which envelope parts it binds, whether a verifier exists, who calls it, and whether it runs before promotion. Synthesis re-read the lines marked **[VERIFIED]**. No tests run, no live stores opened. All other line numbers are from agent reading at this snapshot and **must be re-verified before implementation**.
**Question asked:** do the stage receipts share a shape, is there a shared builder/verifier worth extracting, and what should it look like?
**Companion evidence:** [`provenance_chain_review_2026-09-01.md`](provenance_chain_review_2026-09-01.md) §2 (digest primitives census; this doc is the layer above it); [`architecture_review_five_lens_2026-09-01.md`](architecture_review_five_lens_2026-09-01.md) §4; `validation_receipt_audit_2026-08-17.md` (historical reference, absent from this draft; link qualified 2026-09-06).

**Queue disposition:** audit evidence and checklist source only. The envelope-module work (E-*) is a new item for the wave-1 brief's successor; gate items (G-*) fold into Package A of [`../../agents_todo/brief_architecture_review_wave_1.md`](../../agents_todo/brief_architecture_review_wave_1.md); builder migrations (B-*) adopt into [`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md). This document does not track status.

---

## 1. Verdict

**Roughly 35 builders exist. About 80 % of them already share one header shape by accident. None of them is a complete envelope, and the promotion path checks none of them.**

Three findings carry the weight:

1. **The subject/producer split.** Content manifests (keypoint, crop, tracking, subject-mask core, chaser component, frame-clock) bind produced-content digests and no code identity. Provenance sidecars (`run_provenance`, `stage_provenance`, lineage) bind code identity and no content digests. Only two builders bind both: the native canonical-detection source evidence and the refined subject-mask activation receipt. Everywhere else the "what" and the "who/how" live in separate attrs that never reference each other by digest, so a verifier can confirm either half but not that they describe the same run.
2. **Review is unbound everywhere.** No builder in any family carries a human-acceptance receipt digest. Approval writes `{approved_by, approved_at, git_sha, note}` with no digest of what was approved (`cli/palette.py:1521-1526`) **[VERIFIED]**. The chaser selector's `approval_state: "approved"` is a literal. Training exporters enforce review as a precondition and then bind nothing. The keypoint-review registry row is written `ok` directly from the labeling web app with no receipt.
3. **Nothing recomputes a digest before promotion.** `mark_run_complete` checks that `git_sha` and `config_hash` are present, then moves `latest` (`zarr_run_completion.py:271` → `:300-301`) **[VERIFIED]**. `stage_complete` repeats the presence check plus array shapes and swallows every failure at `:488`. Reconcile writes `ok` from marker presence alone. The real digest verifiers run either on the consumer side (inference model pinning, track-kinematics offline bind) or inside four family-specific activators (subject-mask bundle, raw mask, refined mask, subject shape). The keypoint publisher calls `mark_run_complete` at `:541` and builds and validates its manifest at `:553-573`, after `latest` has moved **[VERIFIED]**.

The 80 % shared shape is the good news. A shared envelope module is a consolidation of what exists, not a new design.

---

## 2. Vocabulary

A **digest** fingerprints one concrete thing: chunk bytes, decoded array values, a canonicalized config, a commit. An **envelope** (receipt, manifest, attestation) is a statement that binds digests together with a claim. A complete envelope has four parts:

| Part | Binds | Answers |
|---|---|---|
| **S** subjects | content digests of what was produced | is this the output the producer wrote? |
| **P** producer | git commit + dirty, effective config digest, method/schema version | what code and parameters made it? |
| **I** inputs | digests of upstream envelopes (or upstream subjects) | what did it read? |
| **R** review | digest of the human-acceptance record, where a human touched it | who signed off, on exactly which bytes? |

An envelope's own digest is what the next stage cites in its **I** block. That is the chain. A **verifier** recomputes S from the artifact and the envelope digest from the body and compares. **Gate wiring** is whether the verifier runs before `latest`, a selector, or a registry `ok` moves.

This matches the in-toto Statement shape (subject digests + predicate) and W3C PROV (entity / activity / agent). The repo has all four parts somewhere; no single builder has them together.

---

## 3. Census by stage family

Legend: ✓ bound by digest · ~ partial (named in the table) · ✗ absent · **path** = bound by path or run name only. "Gate" = verifier runs before promotion of that family's output.

### 3.1 Import, ownership, frame clock, crop

| Builder | Envelope | S | P | I | R | Verifier | Gate |
|---|---|---|---|---|---|---|---|
| `shared/recording_import_receipt.py:117,183` | digest-named sidecar | ✗ | ✓ git, `config_sha256`; dirty forced `false` | ✓ ownership + frame-record sha | ✗ | `:154`, `:193`; `recording_identity_authority.py:569` | on registry read only |
| `shared/pixel_frame_authority.py:2337` | `acquisition_import_ownership` | ✓ video metadata sha | ✗ constant string | ~ materialization receipt | ✗ | `:2556` rebuild-compare | on load only |
| `shared/acquisition_frame_clock.py:686` | `acquisition_frame_clock_record` | ✓ per-array | ✗ schema only | **stat** (size/mtime) | ✗ | `:712`, `:770` rehash | ✓ own `latest` at `:874` |
| `shared/crop_signature.py:26` | `crop_signature` dict | ✗ | ~ lossy `default=str` hash | **names** + copied digest string | ✗ | NONE | ✗ |
| `shared/import_source_fingerprint.py:35` | `*_fingerprint` | **stat** | ✗ | — | — | NONE | ✗ |
| `shared/coordinate_frame_record.py:1343,1717,1966,3903` | frame records + `_sha256` twin | ✓ headered array sha | ✗ | ✓ `DigestBoundFrameRecordRef` | ✗ | `:4261`; stamp reload-verifies `:1587` | ✓ at stamp (transactional) |

### 3.2 Detection, tracking, keypoints, stimulus

| Builder | Envelope | S | P | I | R | Verifier | Gate |
|---|---|---|---|---|---|---|---|
| `shared/run_provenance.py:241` | `run_provenance` attr | ✗ | ✓ git+dirty, `config_hash`, version | **names** + model fingerprint | ✗ | `:405` presence only; recompute only in `track_kinematics.py:5862`, `refined_subject_mask_coordinate_publication.py:1562` | presence check before `latest` |
| `shared/stage_provenance.py:136` | `provenance` attr | ✗ | ✓ git; raw params, no digest | **names** | ✗ | NONE | ✗ |
| `shared/run_lineage_fingerprint.py:300` | `lineage_hash` + payload | ✗ | ✓ commit/dirty/method | ~ refs + fingerprints (chaser callers pass `{}`) | ✗ | only `stimulus_epoch_schema.py:692` | stimulus yes; track_kinematics no |
| `shared/zarr_payload_receipt.py:519,632` | integrity + validation receipts | ✓ 3-root Merkle | ✗ validator schema only | ✓ integrity sha, scientific manifest sha | ✗ | `:601`, `:694` rehash | ✓ track_kinematics offline bind; unused by detect/keypoints |
| `tracking/run_manifest.py:142` | `tracking_run_manifest` | ✓ per-array + decoded | ~ raw `provenance` dict embedded | **names** + subject_position sha | ✗ | `:187`, `:219`; content recompute in consumers | ✗ built then `mark_run_complete` without re-read |
| `shared/zarr/keypoint_manifest.py:458` and six siblings (`refined_keypoint`, `keypoint_quality`, `body_frame`, `crop`, `refined_detection`, `canonical_detection`) | `<stage>_run_manifest` | ✓ array docs + metadata digest | ✗ | ~ crop snapshot ref | ✗ | `validate_*_run_manifest` / `validate_*_publication` per module | **after** `latest` (`keypoint_publication.py:541` vs `:573`) **[VERIFIED]** |
| `canonical_detection_manifest.py:328` source evidence | block in run manifest | ✓ dims | ✓ producer id/version, model sha, `run_provenance` digest | ✓ frame/pixel authority records | ✗ | `:392`, `:1108` | ✓ native path only; legacy `detect_yolo.py` writes no manifest |
| `analysis/stimulus_epoch_schema.py:748` | `stimulus_epoch_run_manifest` | ✓ | ✓ via lineage | ✓ source stimulus shas | ✗ | `:951`; selection recheck `resolved_epoch_selection.py:126` | ✓ |
| `shared/observation_coordinate_publication.py:306` | geometry records | ✓ | ✗ | ✓ acquisition mapping sha | ✗ | `load_*` | stamp-time only |

### 3.3 Subject mask, successor, subject shape, review

| Builder | Envelope | S | P | I | R | Verifier | Gate |
|---|---|---|---|---|---|---|---|
| `shared/subject_mask_attempt.py:34` | `scientific_identity` | ✗ | ~ model artifact sha; no git | ✓ crop/pixel/row mappings | ✗ | `:941` | consumed by worker receipts only |
| `shared/subject_mask_worker_receipt.py:483` | worker semantic receipt | ✓ raw-bytes array sha | ✗ | ✓ identity + attempt digests | ✗ | `:563` | assembly, not promotion |
| `shared/subject_mask_worker_receipt.py:1136` → `subject_mask_validation_receipt.py:202,562` | source run manifest + validation receipt | ✓ | ~ assembly-identity digest; no git | ✓ attempt digest, producer evidence | ✗ | `:260`, `:654`, `:638` | core publish + bundle admission |
| `subject_mask_sampled_contour_worker_receipt.py:243` | contour worker receipt | ✓ | ~ `producer_commit` **free string from CLI**, unchecked | ✓ dense worker receipt digest | ✗ | `:351`, `:648` | pre-bundle |
| `shared/zarr/subject_mask_core_publication.py:2105,842` | core run manifest (+ successor) | ✓ logical content + metadata digest | ✗ | ✓ source manifest + validation receipt digests, crop deps | ✗ | `:1640`, `:1032`, `:1056` | ✓ bundle; ✗ stage selectors |
| `refined_subject_mask_extensions.py:515,596` | cache receipt | ✓ | ~ generator id/version | ✓ core manifest digest | ✗ | `:629` | bundle member |
| `subject_mask_quality_manifest.py`, `subject_mask_cache_publication.py` | quality / cache run manifests | ✓ (third raw-bytes hash copy) | ~ profile/policy digests | ✓ | ✗ | cache `validate_*` | via bundle |
| `subject_mask_bundle_publication.py:660` | bundle manifest | ✓ member digests | ✗ | ✓ | ✗ | `:716`, `:1655`, `:1749` | ✓ `activate_subject_mask_bundle:1866` re-validates before root commit `:1961` |
| `subject_mask_bundle_publication.py:1961` | `subject_mask_authority` | ✓ bundle digest | ✗ | — | ✗ | `subject_mask_bundle_coordinate_authority.py:310` | **is** the authority commit |
| `shared/zarr/coordinate_successor_files.py:102` | payload-file equivalence | **inventory + samefile, no bytes** (H1) | ✗ | — | ✗ | self | ✓ successor stamping |
| `coordinate_successor_authority.py:75` | successor authority + `_sha256` | ✓ record pointers | ✗ | ✓ source digests, equivalence receipt | ✗ | `:188`, `:362` | read/activation time; selector stays deferred |
| `subject_mask_coordinate_validation_receipt.py:509` | surface validation receipt | ✓ pointers | ~ package version, not commit | ✓ | ✗ | `:564`, `:649` | same |
| `shared/subject_mask_coordinate_publication.py:797` | `inference_authority` | ✓ model sha | ~ transform/threshold, no git | — | ✗ | `:818`; reload at `:3604` | ✓ raw activate `:3555` |
| `refined_subject_mask_coordinate_publication.py:1419,1471,3991,4128` | refined coordinate records | ✓ | ✓ `stage_provenance` + sha | ✓ pointers | ✗ string-state check only (`refined_subject_mask_mutation.py:107`) | `:5285` | ✓ activate `:5941` |
| `refined_subject_mask_coordinate_publication.py:1513` | activation receipt | ✗ | ✓ **only P-complete envelope in scope** (`run_provenance` + sha, cross-checked to stage provenance) | ✓ refinement authority | ✗ | inline | ✓ |
| `cli/palette.py:1476` approve → `zarr_run_completion.py:705` | `authoritative_run_provenance` | ✗ | ~ approver git_sha | ✗ | **R without S**: no digest of what was approved **[VERIFIED]** | NONE | writes selector directly; nothing downstream cites it |
| `tune/refined_subject_mask_review.py:633` | review status attrs | ✗ | ✗ | ✗ | free-text state | string equality | gates sealing only |
| `subject_shape_coordinate_publication.py:2264,4382,5316` | unbound numeric manifest, manifest, pending receipts | ✓ **headered** array sha | ~ method/version; git listed as operational, excluded | ✓ bundle/refined digests | ✗ | `:2384`, `:4563`, `:5686` | ✓ activate `:5570` |
| `subject_shape_storage.py:191,492,728` | storage / link receipts | ~ dtype/shape only | ✗ | ✓ manifest ref/sha | ✗ | `:774` | partial |
| `shared/zarr/subject_shape_bundle_source.py` | bundle source record | — | ✗ | ✓ camera authority pointers | ✗ | `:381`, `:431` | ✓ |
| `shared/subject_mask_component_provenance.py:23` | component provenance | ✗ | ✗ | **names** | ✗ | NONE | ✗ |

### 3.4 Chaser components, analytics export, release, figures

| Builder | Envelope | S | P | I | R | Verifier | Gate |
|---|---|---|---|---|---|---|---|
| `analysis/chaser_component_publication.py:443` | `component_manifest` | ✓ strongest: every array + all attrs | ~ method/schema ids; **lineage `code` block dropped** by `chaser_component_writer.py:186-190` **[VERIFIED]** | ✓ base seal, surface manifest, lineage sha | ✗ | `:559` rebuild + byte-compare; arrays rehashed on read `:863-904` | ✓ selector cannot be written without it (`:663-698`) |
| `chaser_component_publication.py:630` | `component_selector` | ✓ manifest sha | schema only | ✓ base seal | `approval_state: "approved"` **literal** | `:700` | ✓ (is the family `latest`) |
| `chaser_component_publication.py:750` | portable handle | ✓ manifest sha | copied ids | ✓ | ✗ | `:795` self-digest only | no selector authority by design |
| lineage in `chaser_distance_runs.py:1652`, `chaser_bout_response.py:1461`, `chaser_response_regimes.py:1139`, `chaser_escape_events.py:1180` | `lineage_hash` | ✗ | ✓ commit/dirty/method | **paths**; `source_fingerprints={}`; `fingerprint_status="best_effort"` | ✗ | NONE | ✗ |
| `analytics_exports/activity_spatial_time_bins.py:1974` + 3 twins + `export_cross_recording_analytics.py:6236` | `export_run_id=<id>.json` | ✓ per-parquet sha/size/rows | ✓ git commit+dirty, tool, schema; no config digest | `source_zarrs` **paths**; cross-recording adds registry identity + chaser authority set | ✗ | `publication.py:399`, `:483` pre-commit; `validation.py:118,573` post | ✓ CAS on file sha inside `manifest_commit_lock:781` |
| `analytics_exports/derived_publication.py:288` | derived manifest, self-digest | ✓ | ~ schema; git only if caller passes | caller-supplied | ✗ | `:170`, `:440` | ✓ pre-commit |
| `group_statistics/goodcopbadcop.py:2717,2933` | stats export manifest | ✓ | ✓ git commit/branch/dirty | ✓ `source_export_manifest_sha256` | ✗ | `validate_staged_publication` `:2997` | ✓ before `os.replace :3006`; **re-implements** the commit sequence |
| `reporting/export.py:75`, `montage_report.py:53` | `report_manifest.json` | ✓ per-artifact sha | ✗ **no git at all** | ✓ export manifest file sha, collection sha | ✗ | `:265` self; `report_registry.py:199` re-hashes | at registry indexing, after write |
| `analytics_exports/chaser_authority.py:175,235` | authority set | ✓ handle + manifest shas | ✗ | ✓ base seals | ✗ | `:203`, `:252`, `:297` | consumed, not gated |
| `cohorts/registry.py:699` `freeze_cohort` | frozen cohort manifest | ✓ member list, query sha | ✗ | ✓ query digest | ✗ | `validate_frozen_cohort :572` | **not called by `release.py`** (only `cohorts/cli.py:80,95`) **[VERIFIED]** |
| `cohorts/release.py:364` | release submission | ✗ `expected_artifacts` are **paths** **[VERIFIED]** | ~ commit; dirty checked in preflight, not recorded | ✓ cohort/zarr-list/authority shas | ✗ | NONE; no reader anywhere | ✗ status mutated in place |
| `shared/plot_artifacts.py:60,184` | `visualizations` attr manifest | ✓ bytes sha | ~ `created_by`, renderer; no git | `source_paths` **paths** | ✗ | none shared; two ad-hoc readers | ✗ `overwrite=True` default deletes prior |
| `analysis/goodcopbadcop_common.py:133,172` | `.exploratory.json` sidecar | ✗ name only | ✗ | ✗ | ✗ explicitly ineligible | NONE | ✗ |

### 3.5 Training exports, models, inference pinning

| Builder | Envelope | S | P | I | R | Verifier | Gate |
|---|---|---|---|---|---|---|---|
| `utils/export_keypoint_training_zarr.py:2656,2948` | `training_export` + immutable publication receipt | ✗ **`tree_inventory(hash_content=True)` computed at `:2981` and discarded** **[VERIFIED]**; only `manifest_sha256` | ✓ invocation git+dirty, `config_hash` (= sha of the export attr itself) | **paths**; compose-v3 binds `source_manifest_sha256` | ✗ precondition only (`:1932`) | `:3064` schema/skeleton only; no digest recompute | ✗ `stage_selector_eligible=False`; registry upsert without digest columns |
| `utils/export_detect_training_zarr.py:295` | `training_export` | ~ `source_manifest_sha256` only | ✓ via run_provenance | **paths** | ✗ | `:586` no digests | `mark_run_complete` → `latest` immediately |
| `utils/export_subject_mask_training_zarr.py:1107` | `training_export` | ✗ | ✓ via run_provenance | **paths** | ✗ precondition only (`:422`) | `:1208` no digests | same |
| `training/training_run_shared.py:57` → `registry/db.py:6807` | `training_runs` row | ✓ model + metrics file sha | ~ raw config-file sha; invocation git; **no base-weights sha**, seed pose-only, no version column | ~ `manifest_sha256` + set id; no dataset content digest | ✗ | `:75-81` expected-manifest check (pose only) | **fail-open** `:121-122` |
| `train_pose.py:2165`, `train_detection.py:2040`, `train_unet_subject_masks.py:1149` | run-dir report | ~ pose runtime receipt sha | ~ commit/branch, **no dirty** | **path** | ✗ | NONE | ✗ |
| `training/export_shared.py:190` → `db.py:6967` | onnx / tensorrt rows | ✓ artifact sha + manifest sha | ~ build env; no git | ✓ skeleton, parent run | ✗ | `model_resolution.py:97` | ✓ cluster inference refuses to start on mismatch (`whole_video_detection.py:347`, `clipped_inference.py:398`) |
| `shared/artifact_fingerprint.py:108,181` | fingerprint dict | ✓ (stat shortcut, mismatch **warning only** `:168`); `:181` fail-closed twin | — | — | — | itself | consumer side |
| inference pinning: `detect_yolo.py:213,2426`, `detect_keypoints_yolo.py:556,1920`, `infer_unet_subject_masks.py:3454` | run attrs `model_sha256` | ✓ | — | model sha vs registry | — | inline, raises | consumer side; `run_sam_subject_masks.py:1593` **not pinned** |

---

## 4. Promotion paths: what is actually checked

| Path | Where | Verifier invoked | Failure blocks? | Relative to pointer move |
|---|---|---|---|---|
| A. zarr `latest` | `shared/zarr_run_completion.py:258` `mark_run_complete`; check `:271`; writes `:300-301` **[VERIFIED]** | `validate_run_provenance` = **presence** of `git_sha`, `config_hash` | yes, only if parent epoch ≥ 2; bypass attr `run_provenance_enforcement_bypass` honoured `:222` | before |
| B. registry `ok` | `registry/stage_complete.py:346`; `_validate_completion_run_group :212` | presence check + `stage_arrays.validate_run :599` (shape/dtype; **no sha anywhere in `stage_arrays.py`**) | **no**: every raise funnels to `:488 except Exception` → warn → `return False` | after (A already moved `latest`) |
| C. direct registry `ok` writers | `labeling/web.py:1052` **[VERIFIED]**, `dish_mask_registry_sync.py:129`, `finalize_crop_flat_roi_cache_batch_registry.py:293` | NONE | — | independent |
| D. reconcile | `registry/maintenance.py:7032` → `:5463`; `_resolve_latest_group :4113` trusts `attrs["latest"]` | marker **presence** | — | after |
| E. selector activation | `shared/selector_activation.py:309`; proof `:463,:467,:498` | caller `proof_loader` = unchanged-since-proof equality | yes | before |
| F. family activators | subject-mask bundle `:1866`, raw mask `:3555`, refined mask `:5941`, subject shape `:5570`, chaser selector `:663` | family-specific full verifiers | yes | before |
| G. stale-run reconcile | `maintenance.py:2029` | NONE (age only) | — | — |
| H. inline refresh | `inline_refresh.py:69` | NONE | — | after |

**Conclusion.** Paths A-D, which cover detection, tracking, keypoints, training exports, and every registry `ok`, recompute no digest. Paths E-F are real gates but exist only where a family author wrote one. The verifiers in §3 that do rehash are never reachable from `stage_complete` (its imports at `:26-30` bring in only `validate_run_provenance`).

---

## 5. Shared shape and copy-paste

**Header.** About 80 % of digest-bearing builders emit exactly `{schema_id, schema_version, digest_algorithm: "sha256_canonical_json_v1", payload_digest, payload}` and their verifiers all open with a five-key set-equality check followed by `canonical_json_sha256(payload)`. Three variants: `record_sha256` over the whole body with the digest excluded (payload receipt, subject-shape pending, storage), digest over header + payload (coordinate validation receipt), and a sibling `<attr>_sha256` twin written at stamp time (successor authority, frame records, coordinate validation). Export-family manifests share a different 80 % (`export_run_id`, git, `table_contracts`, `publication.parts_by_table[]`).

**Array records.** Two grammars: `{shape, dtype, digest_algorithm, sha256}` over raw C-order bytes (all subject-mask families, three separate copies of the streaming hash) vs `{array_ref, dtype, shape, content_sha256, canonicalization}` with a header prefix (coordinate frame, subject shape, chaser). Digests do not compare across the mask → shape boundary. Five more private array-hash implementations exist in frame clock, frame record, tracking manifest, observation geometry, and the payload receipt.

**Producer identity** appears in five incompatible shapes: `producer.{id,git_sha,git_dirty,config_sha256}`, bare `producer` string, flat `git_sha/git_dirty/config_hash`, `code.{git_commit,git_dirty}`, and `git.commit`. Two git-identity implementations with different key names (`system_metadata.get_git_info` vs `run_provenance.git_identity`).

**Input pointers** share `{record_ref, record_sha256}` in the coordinate families and `{ref, sha256}` in receipts, but chaser lineage, exports, plot artifacts, training exports, and provenance sidecars all cite inputs by path or run name.

**Copy-paste clusters worth naming:** seven `build_/validate_*_run_manifest/_publication` triples each with a private `_require_sha256`; four analytics export writers byte-identical except one key; group statistics re-implementing `commit_staged_publication`; two report manifest builders; three chaser per-component attr blocks; four write-reload-verify stamp idioms; `_sha256_file` in five modules; `_register_merged_dataset_in_registry` three times; onnx/engine metadata dicts twice.

---

## 6. What the shared module should be

Not a class per stage. One envelope shape, one verifier, and small stage-specific collectors. The envelope is the same at every stage; only the subject vocabulary changes.

```
fisheye/shared/envelope.py            (new; sits on shared/zarr/manifest_digest.py)

build_envelope(schema_id, schema_version, *, subjects, producer, inputs, review=None,
               digest_scope="payload") -> {schema_id, schema_version, digest_algorithm,
                                           payload_digest, payload}
verify_envelope(envelope, *, expected_schema, recompute=<subject digester>) -> Verification
producer_block(cwd=None, config=None, method=None) -> {git_sha, git_dirty, fisheye_version,
                                           config_sha256, method_id, method_version}
array_content_record(node, grammar="headered_v1") -> {array_ref, dtype, shape,
                                           content_sha256, canonicalization}
file_content_record(path) -> {path, size_bytes, sha256}
inputs_block({name: envelope_or_attrs}) -> {name: {ref, envelope_sha256}}
review_block(approval_doc) -> {receipt_sha256, approved_by, approved_at}
stamp_record(attrs, name, envelope) -> writes attr + `<name>_sha256`, reload-compares,
                                           refuses to overwrite an occupied slot
```

Rules the module enforces so the four parts stop drifting: `producer_block` is required (a builder cannot omit P); `inputs_block` refuses bare paths; `review_block` refuses a state string without a digest; `verify_envelope` never trusts a stat shortcut. The digest grammar is exactly `manifest_digest.canonical_json_bytes` **[VERIFIED at `:18-26`]**, labelled per the D-1 item in the provenance review.

Adapters keep existing readers working during migration: `envelope_to_zarr_attrs`, `envelope_to_training_runs_row`, `envelope_to_export_manifest`. Existing digests keep their `canonicalization` label so old verifiers still pass.

**What already exists and should be the seed:** `zarr_payload_receipt.build_payload_validation_receipt` (S + I, real verifier), `refined_subject_mask_coordinate_publication._activation_receipt_record` (the only P-complete envelope), `chaser_component_publication.validate_chaser_component_manifest` (rebuild-and-byte-compare verifier shape), `analytics_exports/publication.commit_validated_immutable_generation` (validate-then-CAS commit).

---

## 7. Checklists

### 7.1 E: envelope module

- [ ] **E-1** Create `shared/envelope.py` with the surface in §6, built on `manifest_digest.canonical_json_bytes`. Unit tests: round-trip, tamper detection per part, refusal of path-only inputs, refusal of string-only review.
- [ ] **E-2** `producer_block` merges `run_provenance.git_identity` and `system_metadata.get_git_info` into one implementation with one key set; delete the other. `git_dirty` becomes value-required.
- [ ] **E-3** `array_content_record` exports both grammars under explicit labels (`raw_c_order_bytes_v1`, `numpy_dtype_shape_c_order_bytes_v1`); new writers use headered only. Fold the eight private array-hash copies onto it.
- [ ] **E-4** `file_content_record` replaces the five `_sha256_file` copies.
- [ ] **E-5** `stamp_record` replaces the four write-reload-verify idioms.
- [ ] **E-6** Ratchet: forbid new modules defining `build_*_manifest` / `validate_*_manifest` that do not import `shared.envelope`; forbid new `{schema_id, schema_version, digest_algorithm, payload_digest, payload}` literals outside it. Baseline the current count.
- [ ] **E-7** Contract doc for the envelope written **after** E-1 lands and the gate (G-1) consumes it, not before.

### 7.2 G: gate consumption (folds into wave-1 Package A)

- [ ] **G-1** In `stage_complete.py`, move `_validate_completion_run_group` (`:384-402`) outside the `try` at `:374` so validation failures propagate; keep registry-I/O failures inside. This is the zero-new-code fix.
- [ ] **G-2** Add to `_validate_completion_run_group` a call to `require_artifact_content_identity` on every `run_provenance.input_artifacts[*].sha256`; raise on mismatch. First point where S/I digests are recomputed before a registry `ok`.
- [ ] **G-3** `mark_run_complete`: accept an optional `envelope` argument; when present, run `verify_envelope` before moving `latest`. Publishers that build manifests after completion (keypoint `:541/:553`, tracking `:470`) reorder to build → verify → complete.
- [ ] **G-4** Close the three direct registry-`ok` writers (labeling web, dish-mask sync, ROI cache finalize) by routing them through `emit_stage_completion` with an envelope or a review receipt.
- [ ] **G-5** Reconcile (`maintenance.py:7032`) must not write `ok` for a run whose envelope fails `verify_envelope`; write `pending` with a reason instead.
- [ ] **G-6** Remove `run_provenance_enforcement_bypass` acceptance at `zarr_run_completion.py:222` once G-3 lands; grep for remaining writers of that attr first.

### 7.3 R: bind review

- [ ] **R-1** `set_authoritative_run` (`zarr_run_completion.py:705`) records the digest of the approved run's envelope in the approval doc. Approve becomes `review_block(...)`.
- [ ] **R-2** Refined subject-mask seal gate (`refined_subject_mask_mutation.py:107`) checks the review receipt digest against the sealed manifest, not `state == "approved"`.
- [ ] **R-3** Chaser selector `approval_state` becomes a review-receipt digest or is removed; a literal is misleading.
- [ ] **R-4** Training exporters bind `source_review_receipts` (matches T/R items in the training review) instead of enforcing review as a precondition and discarding it.
- [ ] **R-5** Labeling web `keypoints_review` writes a review receipt; the registry `ok` row cites it.

### 7.4 B: builder migration, by family, in order

- [ ] **B-1** Payload receipt + subject shape (already closest): rebase on `build_envelope`, add `producer_block`. Verifies the module against the strictest existing verifier.
- [ ] **B-2** Seven keypoint/crop/detection run manifests: replace the seven triples with collectors + shared verifier; add P; reorder publishers per G-3.
- [ ] **B-3** Chaser components: carry the lineage `code` block into `component_manifest`; pass `source_fingerprints`; replace path `source_refs` with `inputs_block`.
- [ ] **B-4** Subject-mask families: one streaming hash; add P to core manifests; `producer_commit` free string becomes `producer_block`.
- [ ] **B-5** Analytics exports: four twins become one writer; group statistics calls `commit_validated_immutable_generation`; `source_zarrs` become `inputs_block` over run envelopes; report manifests gain P; pick one file-identity rule (canonical bytes with in-file digest, or file sha) and apply it to every cross-reference.
- [ ] **B-6** Release: `release_record` gains S (artifact digests after render) and a self-digest; `release.py` calls `validate_frozen_cohort` before rendering.
- [ ] **B-7** Training: exporters persist the inventory they already compute (`export_keypoint_training_zarr.py:2981`); `training_runs` row built from an envelope with base-weights sha, dataset content sha, seed; registry write fail-closed.
- [ ] **B-8** Figures: `save_figure` primitive (F-1 in the provenance review) is a thin `build_envelope` caller with `inputs_block` over the analysis run; exploratory sidecar gains a content sha even while ineligible.
- [ ] **B-9** Import root: `import_source_fingerprint` and `crop_signature` gain content digests (U-1, U-3 in the provenance review); `recording_import_receipt` stops forcing `git_dirty=false`.

### 7.5 Sequencing

| Order | Items | Depends on |
|---|---|---|
| now, parallel with wave 1 | E-1..E-5, B-1, D-1..D-4 (provenance review) | nothing |
| with Package A | G-1, G-2 | Package A owns `stage_complete.py` |
| after E-1 | E-6, R-1, R-3, B-3, B-5, B-6, B-7, B-8, B-9 | envelope module |
| after G-1 + E-1 | G-3, G-4, G-5, B-2, B-4, R-2, R-4, R-5 | gate consumes envelopes |
| last | G-6, E-7 | everything above |

---

## 8. What was not assessed

- Whether any stored digest on the live store currently fails its own verifier. The census is code-only. A store-wide `verify_envelope` sweep is the natural first use of E-1.
- The realtime/TensorRT path's receipts.
- Registry-side row digests beyond `training_runs` / `training_models`.
- Whether the four analytics export twins produce byte-identical manifests in practice, or have drifted.
