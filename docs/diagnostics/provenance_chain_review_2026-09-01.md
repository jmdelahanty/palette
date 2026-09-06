# Provenance Chain Review: Acquisition to Figure — 2026-09-01

> Disposition added 2026-09-06: retain this snapshot as evidence, not a mandate
> for repository-wide digest replacement. The [second opinion](review_wave_second_opinion_2026-09-04.md) corrects universal
> absence claims and shows that the decoded Merkle root depends on leaf
> partitioning; arbitrary rechunking does not preserve it. Keep logical content,
> physical layout, manifests, execution, and review identities distinct.
> Current disposition and limits: [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-01
implementation: specified-only
-->

**Date:** 2026-09-01
**Snapshot:** `dcdcc081` on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`
**Method:** three parallel read-only review agents (digest/receipt mechanics census; upstream chain trace acquisition→chaser component; downstream chain trace analysis run→figure) plus a synthesis pass that read `zarr_payload_receipt.py` and `run_provenance.py` directly. No tests run, no live stores opened. Findings marked **[VERIFIED]** were re-read by the synthesizer at the cited line. All other line numbers are from agent code reading at this snapshot and **must be re-verified before implementation**.
**Question asked:** is the receipt/manifest digest design consistent with best practice, and does the chain from acquisition to plot actually close?
**Companion evidence:** [`architecture_review_five_lens_2026-09-01.md`](architecture_review_five_lens_2026-09-01.md) §4; `validation_receipt_audit_2026-08-17.md` (historical reference, absent from this draft; link qualified 2026-09-06); [`artifact_identity_merkle_receipt_review_2026-08-29.md`](artifact_identity_merkle_receipt_review_2026-08-29.md); [`publication_receipt_hashing_lifecycle_2026-08-29.md`](publication_receipt_hashing_lifecycle_2026-08-29.md).

**Queue disposition:** audit evidence and checklist source only. Items overlapping resolver, admission, or authority work belong in [`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md). The new items here (video content hash, digest-grammar unification, figure provenance primitive, frozen-cohort science scripts) should be adopted there or into the wave-1 brief's successor. This document does not track status.

---

## 1. Verdict

**The design is above the norm for a first attempt, and the chain does not close.** The middle of the chain is stronger than either end.

What is genuinely right:

- `zarr_payload_receipt.py` is a real three-root Merkle receipt (physical chunk bytes, decoded values, immutable metadata), self-digested, with a verifier that rehashes **[VERIFIED]**.
- The subject-mask bundle envelope, keypoint-v2 bundle, and chaser component manifests bind member digests and are verified on read.
- Cohort release freezes the registry query, every consulted row hash, and the zarr list, and refuses a dirty checkout.
- Group-statistics exports stamp `source_export_manifest_sha256` on every result row and the viewer rechecks it.

What is wrong, in order of consequence:

1. **The root is unbound.** The source video is fingerprinted by path, size, and mtime only (`shared/import_source_fingerprint.py:53-63`) **[VERIFIED]**. Nothing downstream can prove which pixels were analyzed.
2. **The science figures are outside the chain.** The standalone GoodCopBadCop scripts re-query the live registry and take the lexicographically last run (`analysis/goodcopbadcop_common.py:253-255`) **[VERIFIED]**, record thresholds only as module constants, and write a disclaimer sidecar. Two figures a week apart cannot be shown to share inputs. The n=12→32 stale-registry incident is what an unrecorded selection looks like.
3. **Four canonical-JSON grammars and two array-digest grammars coexist.** Same dict, different digest depending on which helper the writer imported.
4. **`config_hash` does not identify configuration.** It hashes the dict the writer passed, which includes video path, output path, and run name; it excludes defaults, model weights, and algorithm version; nothing recomputes it.
5. **Selection is never digested.** Consumers record which run they read, never why it was selected. With nine "latest" resolvers in the tree, that omission is load-bearing.
6. **H1 from the 2026-08-17 audit is still open.** Successor payload equivalence is proven by size and `os.path.samefile` (`shared/zarr/coordinate_successor_files.py:131-162`) **[VERIFIED: samefile at :140]**, then the source digest is copied rather than recomputed.

---

## 2. Digest and receipt census

| # | Family | Bytes hashed | Canonicalization | Stored | Verified by a reader? | Volatile fields |
|---|---|---|---|---|---|---|
| 1 | `zarr_payload_integrity_receipt` (3 roots + `record_sha256`) | per-file sha256 of chunk files; per-shard decoded sha256; `zarr.json` minus `attributes`; roots = sha256 of canonical JSON of sorted leaves | `sort_keys`, compact, `ensure_ascii=False`, `allow_nan=False` (`zarr_payload_receipt.py:37-48`) **[VERIFIED]** | run attrs | yes, `verify_payload_integrity_receipt` (`:601-629`); callers: `track_kinematics`, `subject_shape_coordinate_publication` only | excluded |
| 2 | `zarr_payload_validation_receipt` | validator id/version + integrity root + scientific manifest sha + numerical policy | as #1 | run attrs | yes (`:694-770`), digest-of-digests | excluded |
| 3 | `run_provenance.config_hash` | `stable_json(params)` | `sort_keys`, compact, `ensure_ascii=True`, NaN allowed (`run_provenance.py:63-68`) **[VERIFIED]**; `json_ready` falls back to `str(value)` | run attrs | **no reader recomputes** | includes `video_path`, `output_zarr`, `run_name` (`detect_yolo.py:3895-3910`) |
| 4 | run lineage `source_fingerprint` / `lineage_hash` | source refs + fingerprints + parameter attrs + code attrs | grammar of #1; strips `TRANSIENT_LINEAGE_KEYS` (`run_lineage_fingerprint.py:48-70`) | run attrs, 3 names | partial (`stimulus_epochs.py:777-796`) | excluded |
| 5 | `recording_import_receipt.receipt_sha256` | producer git sha/dirty + config sha + ownership/frame-clock record refs + shas | `manifest_digest.canonical_json_sha256` (grammar of #1) | digest-named file under zarr | yes on load (`:173`, `:216`) | excluded |
| 6 | tracking run manifest `payload_digest` | config + provenance attrs + source manifest sha + per-array records | `manifest_digest` | run attr | JSON only (`run_manifest.py:211-212, 389`) | includes `provenance` attr → hostname, timestamps |
| 7 | subject-mask core/source manifests | nested JSON-of-digests; array leaf = `array.view(uint8)` bytes **without** dtype/shape header (`subject_mask_core_publication.py:195`) | `manifest_digest` | run attrs | yes | mixed |
| 8 | `array_values_sha256` (`coordinate_frame_record.py:600-611`) | canonical JSON header `{canonicalization,dtype,shape}` + `\x00` + C-order `tobytes()` | `_canonical_json` | manifests as `content_sha256` | yes where used (chaser components, subject shape) | n/a |
| 9 | chaser component manifest / selector / handle | JSON with `created_at_utc` redacted; array leaves via #8 | `ensure_ascii=True`, `allow_nan=False` (`chaser_component_publication.py:307-313`) | run attrs | yes (`:559, :700, :795`); arrays rehashed on read (`:863-904`) | redacted |
| 10 | analytics export manifest; cohort release | file bytes and JSON; release writes `indent=2` then hashes the file (`release.py:69-78`) | mixed | manifest JSON / registry | export commit re-hashes parts (`publication.py:611`); readers load parts by path only | includes paths, commit |
| 11 | `fingerprint_artifact` (models), `source_stat_fingerprint` (videos) | **stat_v1 = path + size + mtime_ns**, cached sha256 when size+mtime match (`artifact_fingerprint.py:27-68`; `import_source_fingerprint.py:53-63`) | `sort_keys`, compact | attrs + registry | `require_artifact_content_identity` rehashes, not universal | path included |
| 12 | successor payload-file equivalence | inventory `{path,size}` + `samefile` (`shared/zarr/coordinate_successor_files.py:131-162`) | `manifest_digest` | successor attrs | structural only, **no byte rehash** | — |
| 13 | `crop_signature._hash_parameters` | `json.dumps(params, sort_keys, default=str)` with `str(params)` fallback (`crop_signature.py:15-18`) | loose | attrs | no | — |

Plus roughly 50 ad-hoc `_array_digest` functions, roughly 75 private `_canonical_json*` definitions, and 17 distinct `canonicalization` label strings.

### 2.1 Assessment against practice

- **Canonical JSON.** Four incompatible hashed grammars: (i) `manifest_digest` / receipt / lineage: `ensure_ascii=False, allow_nan=False`; (ii) `run_provenance.stable_json` and chaser components: `ensure_ascii=True`, and (ii-a) `stable_json` admits NaN; (iii) `default=str` loose forms; (iv) indented files hashed as bytes. None is RFC 8785. Grammar (i) is closest. See §6 for what RFC 8785 is and why it matters here.
- **Array identity.** Grammars #7 (no header) and #8 (header) yield different digests for identical logical content. Already flagged in the 2026-08-29 Merkle review at `:255-268`; still unresolved.
- **Re-chunk / re-compress.** The physical root intentionally changes; the decoded root does not. This is documented and correct.
- **Config vs data identity.** Separated in name only. `config_hash` covers a hand-picked dict, includes run-specific paths, excludes effective defaults and model weights, and is never recomputed. It cannot detect duplicate runs and cannot be used to compare re-runs.
- **Merkle.** Real: #1 roots → #2 binds `record_sha256` → downstream manifests carry `source_*_manifest_sha256`. The gap is that roots are not content-addressed names; receipts live in the same mutable attrs as the data. Only #5 (digest-named receipt file) uses the right pattern.
- **Comparison.** Content-addressed storage / Merkle DAG: present in the receipt, absent in addressing. in-toto/SLSA attestation: #2 is structurally an attestation (subject digest + predicate) but with no envelope and no separation from the attested store. W3C PROV: entity/activity/agent exist informally, never as one graph. DVC-style input hashing: stat_v1 matches DVC's fast path, but DVC rehashes on commit boundaries and never trusts mtime for the artifact of record.

### 2.2 Checklist: digest mechanics

- [ ] **D-1** Make `shared/zarr/manifest_digest.canonical_json_sha256` the only JSON digest grammar. Adopt `ensure_ascii=False, allow_nan=False, sort_keys, separators=(",", ":")`. Document it as "JCS-like, not JCS" (see §6).
- [ ] **D-2** Delete `zarr_payload_receipt._canonical_json`, `run_provenance.stable_json`, and the chaser `canonical_component_json_bytes`; route them through D-1. Add a compatibility note for any stored digest whose grammar changes (a re-verify would fail; record `canonicalization` label per stored digest so verifiers pick the right grammar).
- [ ] **D-3** One `array_content_sha256` with the dtype/shape/`\x00` header (grammar #8). Migrate #7 writers; readers accept both labels during transition.
- [ ] **D-4** Ratchet test forbidding new `hashlib.sha256(json.dumps(` and new private `_canonical_json` / `_array_digest` definitions outside `shared/zarr/manifest_digest.py`.
- [ ] **D-5** Close H1: in `validate_payload_file_equivalence`, rehash every linked file and compare to the source integrity receipt's `physical_payload.files[].sha256`; drop `samefile` as sufficient evidence; bump the `hardlink_validation` label.
- [ ] **D-6** Redefine `config_hash` → `effective_config_sha256`: hash the defaults-resolved config plus model `content_sha256` plus algorithm/schema version; exclude paths, run names, output locations. Keep the old attr for compatibility. Have `stage_complete` recompute once and compare.
- [ ] **D-7** `json_ready` must raise, not `str()`, on unknown types; `str(object)` includes memory addresses and silently poisons digests.
- [ ] **D-8** Store a `canonicalization` label alongside every persisted digest that lacks one (audit: 17 label strings today; several digests carry none).

---

## 3. Upstream chain: acquisition → chaser component

Path traced: import → detect → refined detect → crop → keypoints → subject masks → body frame → track kinematics → chaser distance → chaser escape events. Note that `chaser_escape_events` consumes `chaser_distance_runs` and `chaser_bout_response`, not `track_kinematics` directly; `track_kinematics` sits on a sibling branch via swim bouts.

| Hop | Output | Input identity | Code identity | Params | Model | Own payload digest | Downstream verifies? |
|---|---|---|---|---|---|---|---|
| 1 | import receipt + root attrs | identity = sha256(session_uuid, camera_id) (`source_recording_identity.py:128-140`); frame-clock and ownership record sha256; **video = stat_v1 only** **[VERIFIED]** | `producer.git_sha`; `git_dirty` must be literal `false` (`recording_import_receipt.py:161-163`) | `config_sha256` | n/a | self-digested, digest-named | frame-clock record re-verified by detect (`detect_yolo.py:1146-1160`) and crop (`crop.py:811-816`) |
| 2 | `detect_runs/<ts>` | frame-clock ref + sha; video locator + stat fingerprint | `provenance.git` + `run_provenance.git_sha/git_dirty`; dirty recorded, never rejected | `params` → `config_hash` | `model_sha256` cross-checked to registry pin (`detect_yolo.py:215-243`) | only if canonical-detection publication ran (`canonical_detection_manifest.py:148-214`); legacy runs none | refine checks quality-run digest (`refine_detect.py:571-574`) |
| 3 | `refined_detect_runs/<ts>` | `source_detect_run` **name** (`refine_detect.py:1823-1825`); `expected_source_manifest_digest` only when planner-bound (`:1489-1502`) | stage git incl. dirty | effective post-default params (`:1311-1323`) | copies detect's model path strings only | none | crop resolves by name |
| 4 | `crop_runs/<run>` | source run **names** (`crop.py:2820-2824`) + `source_detection_manifest_digest` when available (`:825`) | stage git; `run_provenance` validated at finalize (`:1062-1071`) | explicit incl. `crop_geometry_policy_digest`; `crop_signature` = policy digest + params hash + source names | n/a | `crop_signature` covers attrs, **not ROI pixels** | keypoints snapshot `source_crop_signature` — attrs, not bytes |
| 5 | `keypoints_runs/<ts>` | `source_crop_run` name + crop signature + ROI pixel contract | stage git incl. dirty | requested vs effective recorded (`detect_keypoints_yolo.py:3089-3098`) | `model_sha256`, pose-schema binding sha, training-manifest sha (`:2798-2813`) | shard write records per-array sha (`:1257-1259`) | v2 bundle activation binds four sealed manifests + model sha |
| 6 | subject-mask runs + `subject_mask_authority` | `source_crop_run` + snapshot | stage git | `parameters` | checkpoint `artifact_sha256`, registry-pinned `model_sha256` required (`infer_unet_subject_masks.py:3446-3457`) | **strongest hop**: envelope binds every member manifest digest + per-array sha (`subject_mask_bundle_publication.py:323-475`) | planner verifies bundle |
| 7 | `body_frame_runs/<run>` | v2 manifest with array digests (`body_frame_manifest.py:48`); legacy = keypoint run names | via bundle | schema/estimator version | n/a | yes (v2) | bundle activation |
| 8 | `track_kinematics_runs/<type>/<ts>` | source arrays hashed: `content_sha256` + descriptor + row-identity + temporal record sha (`track_kinematics.py:6294-6318`) | `run_provenance` validated; `config_hash` re-checked (`:5862`) | stage params; lineage fingerprint | n/a | **yes**: integrity + validation receipts (`:11909`) | verified by materializer and subject_shape, **not by chaser code** |
| 9a | `chaser_distance_runs/<run>` | `source_detection_path`, stimulus run **names** (`chaser_distance_runs.py:1568-1573`) | git attr | explicit | n/a | surface manifest `record_sha256` | io refuses `latest` fallback (`chaser_distance_io.py:73-95`) |
| 9b | `chaser_escape_events/<component>` | bout_response by **manifest sha256** (`chaser_escape_events.py:246-263`); chaser_distance by **name** | git + `lineage.code.git_commit/dirty` | 9 explicit params; profile sha computed (`chaser_profiles.py:292, 487`) but **never written** | n/a | **yes**: component manifest, digest-named, selector binds `component_manifest_sha256` + base seal | arrays rehashed on read; cross-recording export uses it |

### 3.1 Findings

- **Name-only hops:** video→import; detect→refined (unless planner-bound); refined→crop; crop pixels→keypoints (legacy path); detection/stimulus→chaser_distance; chaser_distance→escape_events.
- **Digest-bound hops:** import receipt; frame-clock; subject-mask bundle; keypoint-v2 bundle; track_kinematics; bout_response→escape_events; escape_events→exports.
- **Parameter gaps:** model-resolution *policy* not hashed; `quality_detection_params` only if present; chaser profile sha computed and discarded; `config_hash` is of the passed dict, not the on-disk config, except at import.
- **Selection not digested:** `latest`, `latest_complete`, `authoritative_run` are parent attrs outside every receipt; `zarr_payload_receipt` excludes `zarr.json` by design; `TRANSIENT_LINEAGE_KEYS` strips `latest` from lineage hashes. Only chaser components digest their selector.
- **Re-run breakage:** every stage names runs by wall-clock, so `source_*_run` attrs differ on identical re-runs; `crop_signature` includes source run names; stat_v1 changes on any copy; chaser manifests redact `generated_at` for retry equivalence but the persisted digest still includes it.

### 3.2 Checklist: upstream chain

- [ ] **U-1** Content-hash the source video at import (whole-file sha256, plus optional per-GOP digests for partial verification) into the frame-clock record that detect and crop already re-verify. Backfill for existing recordings as a maintenance job with a `video_content_sha256_backfilled_at` attr.
- [ ] **U-2** Make canonical-detection publication mandatory for new detect runs after a cutover date, and require `expected_source_manifest_digest` in refine (`refine_detect.py:1489-1502`) and crop (`crop.py:825`) for post-cutover sources.
- [ ] **U-3** Add ROI pixel array sha256 (grammar #8) into `crop_signature` so keypoints and masks bind to bytes, not attrs.
- [ ] **U-4** `chaser_distance_runs` → `chaser_escape_events`: bind by `source_manifest_sha256`, mirroring the bout_response pattern at `chaser_escape_events.py:246-263`.
- [ ] **U-5** Write the chaser analysis-profile sha (computed at `chaser_profiles.py:292, 487`) into component lineage.
- [ ] **U-6** Record selector state in every consumer: at each `resolve_authoritative_run_name` / `resolve_latest_complete_run_name` call site, write `{parent, selector_attr, value, selection_branch, parent_attrs_sha256}` into the consumer's `source_refs`. Depends on resolver consolidation in the authority queue; do not implement against nine resolvers.
- [ ] **U-7** Refuse `git_dirty=True` for production selectors after a cutover date (import already does; extend to every stage).
- [ ] **U-8** Give chaser component digests a `generated_at`-free preimage so identical re-runs produce identical manifest digests; keep the timestamp outside the digested body.
- [ ] **U-9** Extend `verify_track_motion_payload_validation_receipt` calls to chaser consumers of track kinematics (currently only materializer and subject_shape).

---

## 4. Downstream chain: analysis run → figure

| Hop | Output | Input identity | Code identity | Params / exclusions | Own digest | Consumer verifies? |
|---|---|---|---|---|---|---|
| 1 | per-recording run binding in export | run name + `source_manifest_sha256`, `source_publication_commit_sha256`, `physical_authority_sha256`, `selection_snapshot` (parent `latest`, `latest_complete`, epochs), `completion_snapshot.selector_eligible` (`kinematics_samples.py:760-794`) | attrs only | stage-specific | `payload_sha256` (`:797`) | exporter fails closed on mismatch (`activity_spatial_time_bins.py:340-469`) |
| 2 | analytics export generation (`v1/manifests/export_run_id=<id>.json` + parts) | `source_zarrs` + per-source binding; self-digested `registry_identity` receipt (`registry_identity.py:366-392`); `chaser_authority` receipt with `base_run_name` (never `latest`) + seal sha (`chaser_authority.py:175-200`) | `palette_git_commit`, `palette_git_dirty` (`kinematics_samples.py:1617-1618`) | `export_parameters` (`:1642-1651`) | per-part sha256 + row count; manifest-last CAS rename (`publication.py:135-207, 486-702`) | commit re-hashes staged parts (`:611`) and registry identity (`:671-683`); **readers load parts by path, no digest check** |
| 3 | cohort release (`release_record.json`, `frozen_cohort_manifest.json`) | registry UUID, hash of every consulted row, `cohort_query_sha256`, `zarr_list_sha256`, chaser authority sha (`release.py:363-389`) | `palette_commit`; requires clean checkout, workstation==cluster HEAD (`:273-286`) | cohort query incl. `--missing-selected-metadata` | `frozen_cohort_manifest_sha256` (`:377`) | export checks registry UUID vs frozen identity; release only **names** expected export/stats/report ids (`:385-389`), never hashes them |
| 4 | group-statistics export | `source_export_run_id` + `source_export_manifest_sha256` in manifest (`goodcopbadcop.py:2729-2736`) **and every result row** (`:2263`, `:2399`) | commit/branch/dirty (`:2725-2727`) | full: bootstrap/permutation iterations, CI, `minimum_recordings`, `minimum_acquisition_batches`, seed, FDR method+family, contrasts, `excluded_unit_count` per row (`:2282, :2762-2777`) | part inventory (`:2957`); manifest validated on load | viewer re-hashes source export manifest (`group_analytics_viewer/query.py:473-476`); parquet parts by path only (`:772-783`) |
| 5a | cohort report (`report_manifest.json` + PNGs) | export id + collection manifest sha (`report_registry.py:67-123`); per artifact `zarr_path`, `source_run`, `declared_content_sha256`, renderer+version (`export.py:160-178`) | `renderer_version` only; **no git commit** (`export.py:223-248`) | `report_plan` + its sha; no analysis params | `manifest_sha256` (`:249`); PNG bytes re-hashed (`:189-194`) | viewer verifies (`artifacts.py:131-164`) |
| 5b | group-analytics viewer figures (browser) | `export_run_id` context | none | none | none written | n/a |
| 6 | standalone science figures (`figures/*.png` + `.exploratory.json`) | live registry query (`goodcopbadcop_common.py:222-235`); run = lexicographically last child (`:253-255`) **[VERIFIED]**; `track_kinematics_run="latest"` (`:343`) | none | none; thresholds are module constants (`analyze_goodcopbadcop_escape.py:34`); skipped recordings only `print("skip", ...)` (`:65`) | none; sidecar records `analysis_id`, tier, artifact name only (`:62-86, :133-169`) | nobody |

### 4.1 Findings

- **Where the chain ends.** The last verifiable binding is the group-statistics export. The cohort report manifest is self-verifiable but binds to export id, not export manifest digest, and carries no commit. The viewer persists nothing. The standalone science figures, which produced the claims in `goodcopbadcop_behavior_synthesis_handoff_2026-07-17.md`, are outside the chain.
- **No figure provenance primitive.** Five private `_artifact_signature` copies (`chaser_distance_runs.py:99`, `bout_kinematics.py:857`, `plot_track_kinematics.py:686`, `swim_bout_visualization.py:181`, `visualize_eye_angles.py:440`), one `_save_figure_with_metadata`, and `save_standalone_exploratory_figure` whose sidecar is a disclaimer, not provenance. Over 100 `savefig` sites; none pass `metadata=`. `shared/plot_artifacts.py` exists and should be checked as the natural home.
- **Exclusion provenance.** Recorded only at hop 4 (minimums, excluded unit counts, FDR rule) and hop 3 (missing-metadata policy). Speed thresholds, immobility thresholds, `latest()` picks, try/except skips, and `require_on_disk` drops are code-only.
- **Two figures a week apart.** Through the release pipeline: yes for statistics tables and cohort-report PNGs. For standalone science figures: no, by construction.

### 4.2 Checklist: downstream chain

- [ ] **F-1** One `save_figure(fig, path, *, sources, parameters, exclusions)` primitive (candidate home: `shared/plot_artifacts.py`) writing `<name>.provenance.json` with: `palette_git_commit`, `palette_git_dirty`, `fisheye_version`; either `export_run_id` + `source_export_manifest_sha256` or per-recording `{zarr_path, run_name, payload_sha256}`; parameter dict; explicit exclusion list with reasons; PNG/SVG `content_sha256`; `canonicalization` label; self `record_sha256`. Embed the record sha in the image metadata via `savefig(metadata=...)`.
- [ ] **F-2** Replace the five `_artifact_signature` copies and `save_standalone_exploratory_figure` with F-1.
- [ ] **F-3** Add `source_export_manifest_sha256` and `palette_git_commit` to the cohort report manifest (`reporting/export.py:223`, `montage_report.py:172`); have `report_registry.py:212-231` verify the digest, not just the id.
- [ ] **F-4** Group-statistics and viewer readers verify parquet part digests against the manifest inventory on load, not only at commit.
- [ ] **F-5** Cohort release records the export, stats, and report manifest digests after they are produced (a `release_closure` record), not only their expected ids.
- [ ] **F-6** Ban `latest()` in `analysis/goodcopbadcop_common.py` and siblings: `resolve_cohort` takes a frozen-cohort manifest path and returns exact run names; scripts refuse to run without `--frozen-cohort`.
- [ ] **F-7** Move analysis thresholds out of module constants into a parameter dict that F-1 records; skipped recordings become entries in the exclusion list, not print statements.
- [ ] **F-8** Regenerate the figures in the 2026-07-17 handoff through F-1 + F-6 once landed, and record their provenance sidecars next to the handoff.

---

## 5. Sequencing relative to the architecture wave

Independent of wave 1 and safe to run in parallel:

- D-1 through D-5, D-7, D-8 (digest hygiene; touches `shared/zarr/manifest_digest.py`, receipt module, successor files).
- U-1 (video content hash; touches import only).
- F-1, F-2, F-7 (figure primitive; touches visualization and analysis scripts only).

Blocked on wave-1 Package A (gate fail-closed) or the authority queue's resolver consolidation:

- D-6 (`config_hash` recompute in `stage_complete`).
- U-2, U-6, U-7 (mandatory canonical detection, selector recording, dirty refusal all run through the gate and the resolvers).
- F-6 depends on cohort release semantics being the only selection path for science scripts.

Recommended first package after wave 1: **U-1 + D-5 + D-1..D-4** (root binding and one grammar), then **F-1 + F-2 + F-7** (figure primitive). Together they turn "I hope these figures match" into "here is the digest" without waiting for the enforcement work.

---

## 6. What RFC 8785 is and why it matters here

RFC 8785, the JSON Canonicalization Scheme (JCS), defines exactly one byte sequence for any given JSON value so that hashing or signing JSON is deterministic across languages and libraries. Its rules: object keys sorted by UTF-16 code units; no whitespace; strings serialized with the minimal escaping JSON requires and non-ASCII emitted as UTF-8, not `\uXXXX`; numbers serialized using the ECMAScript `Number.prototype.toString` algorithm (shortest round-trip, specific exponent format); no NaN or Infinity; UTF-8 output.

Why it matters for palette: the whole chain rests on "hash of canonical JSON". Every place a different helper serializes the same dict differently produces a different digest for the same content, and a verifier written against one helper will reject a digest produced by another. Today four grammars coexist (§2.1). Two of them differ only on `ensure_ascii`, which changes bytes for any non-ASCII string; one admits NaN, which JCS forbids and which breaks round-tripping.

Python's `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` is close to JCS but not identical: Python sorts keys by code point rather than UTF-16 code unit (differs only for astral-plane characters in keys), and Python's float repr agrees with ECMAScript for most values but not the exponent formatting of very large or very small magnitudes. For palette's purposes, adopting that call as the single grammar and labeling it `canonical_json_python_sorted_compact_v1` is sufficient. Full JCS compliance is a later step if signatures or cross-language verification are ever needed; a small library (`rfc8785` on PyPI, or a 40-line implementation) exists if so.

---

## 7. What was not assessed

- Training-dataset exports and model-training provenance are under separate review (same date).
- No live zarr, registry, or export was opened; all binding claims are from code reading.
- Labeling web app receipts, acquisition-side (Citrus) provenance before import, and TensorRT engine provenance were not traced.
