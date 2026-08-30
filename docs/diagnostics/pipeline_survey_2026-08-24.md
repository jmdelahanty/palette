# End-to-End Pipeline Survey: Import → Plots

**Date:** 2026-08-24
**Method:** 8 parallel read-only survey agents (ingestion/import, cluster orchestration, registry spine, provenance/receipts, reporting/export, `visualization/` module, diagnostics + video renderers, immobility raw-vs-smoothed sweep), synthesized. No files modified.
**Repo state:** branch `agent/palette/clipped-geometry-acquisition-authority-20260821`, HEAD `a2859cb0`.
**Companion docs:** `crop_contract_split_audit_2026-08-24.md` (the crop→keypoint→tracking boundary, audited separately and not repeated here), `validation_receipt_audit_2026-08-17.md`, `contract_enforcement_divergence_review_2026-08-21.md`.
**Coverage gaps of this survey:** detection/refinement stage internals, the mask/shape segment, and the chaser component family were surveyed only via adjacent lenses (import, orchestration, provenance, immobility) — a dedicated pass on those segments was cut short by a session limit and can be rerun on request.

**Plan disposition (2026-08-25):** end-to-end evidence and finding catalog.
Overlapping authority/admission/resolver work is tracked only in
[`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md).
Scientific and performance findings remain scoped evidence for their own work
queues; the numbered queue below is not an independent authority-work status
source.

---

## 0. Verdict

The pipeline's *interior* stages (import → detection → publication) are in the best shape they have
ever been: authority stamping is disciplined, atomic publication with content hashing is the norm,
and the 2026-08 acquisition-authority wave (`9a17a946`/`a2859cb0`) turned geometry fits from
self-rooted trust into leaves of a verified authority tree. The health gradient runs **downhill
toward both ends**:

- **Upstream anchor:** every cryptographic chain roots in a `stat_v1` fingerprint
  (path+size+mtime, never content) for the video pixels themselves.
- **Downstream terminals:** plots, dashboards, and reports — the surfaces the error-budget policy
  declares Tier boundaries — are the least contract-bound layer in the system: unchecked
  `latest` resolution, one still-raw speed metric on a dashboard stat tile and a published cohort
  figure, visualizers that write to archives by default, and a reporting discovery layer whose
  chaser branch unconditionally returns nothing.
- **Cross-cutting:** orchestration trusts markers without payloads (13 catalogued trust
  boundaries), the registry has no scheduled reconciler (the known 7-week-freeze shape), and the
  strongest evidence in the system (`cluster_output_staging` full-tree content hashes) is
  persisted but excluded from every manifest digest — retained but unbound.

Same disease as every prior wave: **capture outruns enforcement, and enforcement thins at
boundaries** — now mapped end-to-end.

---

## 1. The pipeline map

### Stage 1 — Import / ingestion (healthy; best-documented layer)

Two writer families per recording (`import_profile_contract.py:20-34` classifies post hoc):

- **`<rec>_analysis.zarr`** — metadata-only (no pixels). `import_recording_analysis.py:449`
  (single), `import_organized_recordings_analysis.py:371` (canonical batch, what the `palette` CLI
  emits), `create_clipped_analysis_zarr.py:520` (clipped shell). Completion marker =
  `acquisition_authority_publication_status == "published_canonical_v1"` (mirrored root +
  raw_video, `acquisition_publication_status.py:19-65`), with a pending→published resumable
  contract. Stamps: acquisition camera authority (`analysis/acquisition_camera_frames/<cam>` with
  record + sha256), frame clock runs (digest-idempotent), colorimetry
  (`video_color_range/space/transfer/primaries` from ffprobe), encoder tags, source-video
  metadata v2. Metadata precedence: recording_manifest.json > ffprobe (disagreement fatal) >
  OpenCV (cross-check only).
- **`<rec>_training.zarr`** — materialized sampled pixels, only via
  `publish_sampled_training_base` → `import_sampled_training_pynvvc.py:431` (atomic temp+rename,
  not resumable). Deliberately publishes **no** acquisition authority
  (non-canonical by definition); stamps the full pixel contract incl.
  `color_range = "source_full_range_0_255_container_observed_<pc|tv|unknown>"`.

Weak spots: `--allow-preflight-failures` bypasses; colorimetry probe swallows exceptions to `{}`;
training importer accepts `container_observed_unknown` instead of the fail-closed normalizer
(`import_sampled_training_pynvvc.py:259` vs `roi_pixel_contract.py:31-47`);
`create_clipped_training_zarr.py` stamps **no** import profile, authority, or fingerprints
(classifies as `unknown_raw_video_profile`); `finalize_organized_staging.py:194` gates teardown on
path existence of the analysis zarr, not on its publication status.

**Root-of-trust caveat (systemic):** `import_source_fingerprint.py:35-70` is `stat_v1` by design
("intentionally does not hash full MP4/H5 contents"); clipped collection members likewise
(`clipped_video_collection.py:386`, `relocation_stable: False`). Everything above it is now
cryptographically tight; its anchor is not.

### Stage 2 — Orchestration (pure DAG compiler; no runtime brain)

Every planner compiles an immutable LSF workflow, submits once via `bsub -w done(...)`, and exits.
**No scheduler, no poller (zero `bjobs`/`bkill` in the repo), no retry.** Three non-equivalent
completion mechanisms: LSF exit codes, `Path.exists()` on expected outputs
(`lsf/runtime.py:162-177` — no payload inspection), and zarr completion attrs (fail-open by
default, below). Registry rows gate nothing; `registry_finalize` is a terminal reconcile.

13 trust boundaries catalogued (agent report, T1–T13). The sharpest:

- **T11 — silent registry skip then cleanup.**
  `whole_recording_analysis_registry_finalize.py:123-176`: a non-authoritative bundle skips all
  registry writes yet returns top-level `"status": "ok"`; the job exits 0 and
  `roi_cache_cleanup` proceeds to **delete the ROI caches**. (`clipped_inference_cleanup.py:52-60`
  does re-read the report; the whole-recording path does not — T13.)
- **T2 — directory-exists completion.** Whole-recording expected outputs are *directories*
  (`whole_recording_analysis.py:396-402,493-499,600-606`); an empty group or a `running` run
  satisfies the gate.
- **T4/T5 — fail-open completion defaults.** `zarr_run_completion.py:84-97,397-409,438-443`:
  no `palette_completion_epoch` and no `palette_store_epoch` ⇒ unmarked run counts complete;
  absent `stage_selector_eligible` ⇒ eligible; `:606-613` reverse-lexical child-name scan as the
  legacy "latest". Epoch stamping only happens on **newly created empty** parents
  (`:159`) — populated parents are permanently grandfathered (the consolidated-metadata
  split-brain's sibling). Accepted fail-opens surface only as an `atexit` RuntimeWarning
  (`:375-394`) — invisible in LSF stderr.
- **T12 — consolidated metadata.** Cluster readers bypass it (`open_zarr_group_direct`);
  only `keypoints/registry_finalize.py:161,202` both trusts and verifies it
  (`validate_direct_consolidated_subtree`).
- **T9 — plan re-read at execution.** Array elements re-read `lsf_plan.json` from disk at run
  time with no digest binding to what was submitted (`lsf/task_group.py:30-53`).

`core/pipeline.py` is confirmed dead for orchestration (reachable only via `python -m fisheye`).

### Stage 3 — Registry spine (index, not truth; no actuator)

Authority policy is explicit and structurally enforced (`registry_data_governance_policy.md:11-35`):
zarr wins for scientific content (all scientific tables are DELETE-then-INSERT projections via
extractors), registry wins for identity/locators/minted entities (batches, models, analytics
manifests, `registry_identity` — triple-trigger immutable singleton).
`reconcile_dataset_from_root` (`db.py:6787`) is the strict-superset engine; the 2026-06 collapse
landed and the profile sync scripts are deleted.

- **No scheduled reconciler.** All reconciliation is manual; the only automation is a backup cron,
  a nightly smoke probe, and CI schema-drift checks. The error-budget SLO
  (`error_budget_policy_2026-08-11.md:46` — reconcile divergence <1%, ledger freshness ≤7 days)
  has **no watcher**, the exact shape that let the 2026-06 ledger freeze run 7 weeks.
  `manual_add_row_propagation_design.md:226-243`: "Only the actuator is missing. Build the
  actuator before anything else."
- **Weakest write path:** `scan.py` / `registry_rescan` without `--safe-shadow-publish` —
  apply-by-default, no backup, no integrity gate, direct NFS SQLite writes. (Contrast:
  `prune_stale_datasets` and `dedupe`, the destructive tools, have the strongest guards.)
- **Canonical path is hardcoded in ≥6 places** with separate constants; `RegistryPaths.from_env`
  tier-3 fallback silently hands a **repo-local** registry to a caller with no env/config.
  No code-level guard against the stale `/nvme1` copy (comments only).
- Commit `26883c1a` is a runtime-provenance fix (one validator: full `integrity_check` +
  `foreign_key_check` via Python's sqlite3; system `sqlite3` demoted to supplementary;
  `quick_check` banned after the 08-13 incident; backup via the sqlite backup API with
  double-validated receipts). WAL stays off (NFS multi-host), enforced by absence + a sidecar
  tripwire that refuses publication if `-wal`/`-shm`/`-journal` exist.

### Stage 4 — Provenance / receipts (strong capture, weak binding)

The 2026-08-17 audit needs two revisions and keeps one finding:

- `zarr_payload_receipt` is **still single-consumer** (track_kinematics only; module untouched
  since 07-25). Three materializers (`subject_shape.py:1010`, `bout_kinematics.py:451`,
  `eye_angles.py:69`) already produce the exact `exact_decoded_validation` copy report the
  receipt builder consumes — and build no receipt. Hot-path verification also passes
  `verify_physical_payload=False` (`materializers/track_kinematics.py:646,652`).
- "Evidence evaporates" is now only **partly** true: `atomic_run_publisher` persists a full
  publication receipt (full-tree `content_sha256` + four validation passes) to
  `run.attrs["cluster_output_staging"]` by default (~30 families), and the mask coordinate
  validation receipt landed 08-17 (`1e7b2eb9`). **The remaining defect is
  persisted-but-unbound**: `cluster_output_staging` has no self-digest and is on every manifest's
  exclusion list (`chaser_component_publication.py:127`, `subject_mask_cache_publication.py:130`,
  `subject_shape_coordinate_publication.py:1238`, `subject_mask_quality_manifest.py:72`,
  `track_kinematics.py:4617`) — the strongest evidence in the system sits in a mutable attr
  nothing seals or re-verifies.
- **H1 OPEN** (and introduced by the 08-17 fix itself): mask-successor payload identity is
  `os.path.samefile` + size — inode identity, not content
  (`coordinate_successor_files.py:102-164`; `receipt_digest` hashes path+size only). Doesn't
  survive relocation; certifies agreement even if the shared inode is corrupted. Fix: add
  per-file sha256 to `_payload_inventory` (`:90-92`), keep `samefile` as a fast extra check.
- **H2 OPEN**: `chaser_escape_events._resolve_bout_response` bypass
  (`chaser_escape_events.py:254-268`) — no schema check, `sorted(parent.keys())[-1]` final
  fallback, emits `source_bout_response_manifest_sha256: null` into persisted provenance
  (`:859,:1174,:1193`). Uniquely in its family, **no CLI flag exists to supply a dependency
  handle** (`:1226-1249`), so direct invocation can only take the unverified path.
- The acquisition-authority binding pattern (`load_persisted_acquisition_camera_authority` +
  `assert_verified()`) is now used consistently across 13 consumer modules — the one provenance
  pattern that has genuinely spread.

### Stage 5 — Analysis components (spot-checked via immobility + provenance lenses)

The immobility/smoothed-speed policy fix (post-artifact-discovery) **mostly held**:
`chaser_response_regimes` and `chaser_near_field_occupancy` default to
`verified_track_motion` smoothed speed with QC warnings and provenance attrs; no production
config selects `raw_centroid_explicit`; status page / registry browser / group viewer expose no
immobility at all. **One computation was missed** — see Finding 1 below.

### Stage 6 — Reporting / export (strong manifests, silent-skip discovery)

Reports are a static import-validated catalog (8 families, 15 visualizations, 6 providers,
`reporting/catalog.py`); reporting never plots — it collects pre-rendered PNG artifacts from the
zarr. Export manifests are exemplary: plan sha256 embedded, per-artifact declared-vs-actual hash
raise, atomic temp+rename, immutable registry indexing with re-verification
(`report_registry.py:199-294`). But discovery erases causes:

- **The chaser-distance family always resolves to zero runs** — both branches of
  `discovery.py:101-109` `return ()` (the canonical load is paid as a preflight and discarded), so
  all three chaser visualizations permanently plan as `NEEDS_ANALYSIS` even when complete
  contracted artifacts exist. Locked in by tests (`test_reporting.py:369-413`).
- **An unopenable recording vanishes from `nonready[]`** — `planner.py:292-305` records the error
  but `export.py:50-60` iterates items only, so a bundle can export "clean" while silently
  omitting a whole recording; `fail_on_nonready` cannot trip.
- Failed/running runs are dropped indistinguishably from never-existed
  (`discovery.py:122-124`, denylist so unknown statuses pass); six bare `return ()` exits in the
  track-kinematics artifact path collapse authority-mismatch and never-rendered into one generic
  `NEEDS_RENDER` reason. No logging anywhere in the package.
- `query_indexed_reports(latest=True)` is an unverified `LIMIT 1` (`report_registry.py:328`)
  while its sibling `resolve_latest_export_table` verifies every candidate — the reporting-side
  "latest" can return a report whose artifacts are gone.
- The manifest's `source_backends` can claim `"parquet"` but the package never reads parquet
  (`export.py:230-232`); `source_tables` is declaration-only.

### Stage 7 — Visualization / plotting (the least contract-bound layer)

30 modules in `visualization/`, none CLI-registered. Split verdict:

- **Properly gated:** `interactive_track_kinematics` (completion + eligibility + authority
  cross-check), `visualize_keypoints` (strictest selector), `goodcopbadcop_interactive`
  (fail-closed on unsealed components, `speed_source` carried), `plot_detection_epoch_heatmaps`
  (best provenance: descriptor/manifest/profile sha256s), `visualize_subject_shape_overlays`,
  `visualize_eye_angles` (richest stamping: git/env + artifact signature + PNG metadata).
- **Unchecked `latest` / `sorted(keys)[-1]`:** `visualize_detect_quality.py:104,114`,
  `detection_visualizer.py:1023`, `visualize_refined_detections.py:400`,
  `overlay_arena_mask.py:144`, `visualize_chaser_vs_fish.py:25-34`,
  `visualize_experiment_timeline*.py` (ignores `latest` entirely), and —
  most consequentially — `_latest_unsealed_inspection_child` (`chaser_analysis_figures.py:49-55`,
  self-documented "never use for a normal scientific read") is the read path for **all five**
  derived components in both cohort-figure modules.
- **Visualizers that WRITE:** `visualize_eye_angles.py` mutates the archive **by default**
  (`--write-zarr-artifact` defaults True, `:1508-1513`); `visualize_crops.py:134` sets review
  attrs from a "quick viewer"; `visualize_swim_bladder_mask_patches.py:704` is an editor living
  in `visualization/`.
- **Wrong-circle pattern still alive:** `overlay_arena_mask.py:74-108` rebuilds the mask from
  nominal `experimental_area_*` (the exact pattern that inverted thigmotaxis);
  `visualize_chaser_vs_fish.py:136-138` draws a user-supplied circle at `(r, r)`. Both untested.
  The four chaser figure modules correctly use `require_dish_mask_arena_geometry`.
- Fail-closed stubs (`collect_visits`, `collect_ring_entries`) fall off the end returning `None`
  if the authority call ever stops raising.

### Stage 8 — Diagnostics + video renderers

Only 5 of 129 diagnostics render pixels. Two are model-quality provenance
(`compare_realtime_offline_detections` — schema'd, hashed, zarr-artifact;
`probe_recording_dish_rim_fit` — blind-fit protocol with per-PNG sha256); two are weak
(`preview_eye_mask_background_subtraction` — zero provenance, raw `roi_images` indexing;
`plot_sampled_component_contours` — raw-zarr fallback with provenance only on stdout).

**Color-range reality:** the contract (`roi_pixel_contract.py:23-47`) is exact and fail-closed,
and its two constants are the only `"tv"`/`"pc"` literals in the tree — but **no renderer honors
it**:

- `build_review_proxy_videos.py:328` — `-pix_fmt yuv420p`, no `-color_range`/`-colorspace`;
  `probe_video()` `:247` never requests color fields so the manifest cannot record the source's
  claim; encoder auto-select (`h264_nvenc` default on the sharded bsub path, `:522`) makes VUI
  tagging host-dependent. Feeds every reviewer's browser via the labeling/review web UIs.
- `playgrounds/heartrate_stabilization/` — 14 renderers decode via `cv2.VideoCapture`, whose
  FFmpeg backend assumes limited range: an implicit `(Y-16)*255/219` expansion that grep for
  `"tv"` can never find, in scripts doing **absolute-intensity photometry**. The detector for
  this exact lattice already exists (`pixel_decode_exposure_census.py:175,463-464`) — it was
  pointed at training surfaces, never at these.
- `materialize_orange_style_clips.py` is safe (`-c:v copy` preserves VUI exactly).

---

## 2. Ranked findings (new this wave)

1. **`chaser_escape_freeze_summary` still thresholds raw centroid speed at 1.0 mm/s** — below the
   ~1.6 mm/s noise floor (`chaser_escape_freeze_summary.py:432-442,792`; threshold `:55`). Its
   `freeze_low_speed_fraction` metrics reach a **marimo headline stat tile**
   (`apps/marimo/components/goodcopbadcop_chaser.py:3796-3799`) and the habituation cohort figure
   panel titled *"…and freezing replaces it"* (`chaser_habituation_figures.py:145-152,408`).
   This is the immobility artifact class that already killed one result, live on a published
   claim. Port the `immobility_signal_mode`/`load_verified_smoothed_frame_speed` fix.
2. **Reporting's chaser discovery returns `()` on both branches** (`discovery.py:101-109`) —
   chaser reports can never include chaser artifacts. One-line-shaped bug, test-locked, so fix
   tests with it.
3. **T11+T13: whole-recording registry finalize can silently skip registry writes, report ok, and
   let cleanup delete ROI caches** (`whole_recording_analysis_registry_finalize.py:123-176,188`).
   Make cleanup re-read the registry report as the clipped path does
   (`clipped_inference_cleanup.py:52-60`), and make the skip a non-ok status.
4. **Review proxy color range** (`build_review_proxy_videos.py:328,247`): probe
   `color_range/space/pix_fmt`, set `-color_range` explicitly from the probe, record both in the
   proxy manifest. The store's tv/pc split makes this concretely wrong today, host-dependently.
5. **H2** — add a dependency-handle CLI flag to `chaser_escape_events`, delete the
   `sorted(keys)[-1]` fallback, refuse to publish `manifest_sha256: null`.
6. **H1** — per-file sha256 in the mask-successor payload inventory
   (`coordinate_successor_files.py:90-92`).
7. **Build the reconcile actuator** — a cron/`/loop` job running `reconcile_sweep --dry-run` +
   the ledger-freshness check, alerting on divergence. Already designed
   (`manual_add_row_propagation_design.md:226-243`); the SLO exists with no watcher.
8. **Point `pixel_decode_exposure_census` at the heartrate playground renderers** before trusting
   any photometry from them; fix decodes (PyNvVC luma path or explicit range handling).
9. **Arena-geometry stragglers**: `overlay_arena_mask.py`, `visualize_chaser_vs_fish.py` — dish
   mask or a loud UNVERIFIED banner; both currently untested.
10. **`visualize_eye_angles` write-by-default** — flip `--write-zarr-artifact` to opt-in, or move
    persistence to the finalize utilities where its siblings keep it.
11. **Export nonready hole** — an unopenable recording must appear in `nonready[]`
    (`export.py:50-60` vs `planner.py:292-305`); plus verify `query_indexed_reports --latest`
    like its analytics sibling.
12. **Seal the atomic receipt** — give `cluster_output_staging` a `record_sha256` and stop
    excluding it from manifest digests (or bind its digest into them); extend
    `zarr_payload_receipt` to the three materializers already producing its input.
13. **Fail-open completion residue** — epoch-stamp populated run parents (currently grandfathered
    forever, `zarr_run_completion.py:159`), and turn the atexit fail-open counter into a hard
    signal orchestration can see.
14. **Registry path unification** — one constant for the canonical path; make the
    `RegistryPaths.from_env` tier-3 repo-local fallback loud or fatal outside tests.
15. Dead code: `analyze_goodcopbadcop_immobility_artifact.load()` returns `None`
    (`:45-49`) — the module cannot run as written.

## 3. What is genuinely healthy (keep, and copy from)

- Import-stage stamping and the pending→published resumable completion contract.
- The acquisition-authority spine and its 13-consumer `assert_verified()` adoption (the one
  provenance pattern that spread) — the template for spreading the payload receipt.
- Atomic publication with content checksums as the default posture (~30 families).
- Registry: single-validator integrity binding (`26883c1a`), backup receipts, dry-run-first
  tooling with the strongest guards on the most destructive tools.
- Reporting/export manifest hashing and immutable registry indexing.
- The two exemplary diagnostics (`compare_realtime_offline_detections`,
  `probe_recording_dish_rim_fit`) — the provenance pattern to require of new figure code.
- The immobility fix held everywhere except one module; the fix pattern
  (`immobility_signal_mode` + `speed_source` provenance) is proven and portable.

---

## 4. Redundancy consolidation plan (added 2026-08-24)

Follow-up to the overengineering assessment: the repo's overbuild is **redundancy and unretired
generations, not rigor** — the same guarantee implemented 2–4 times, depth where it's cheap,
absence where it's needed. Five clusters, enumerated with call sites, ranked by what each
consolidation actually buys. Rule of thumb throughout: **one implementation per guarantee**, and
when a receipt covers something, delete the rescan it replaces.

### Cluster 2 first — full-tree rescans alongside an unbound receipt (the only expensive one)

This is where the "2–7 full rescans/publication" cost lives (validation/receipt audit 2026-08-17).

- Producer: `shared/atomic_run_publisher.py` runs four full `validate_run` tree-scans per
  publication (local, temporary, pre-pointer `:1013`, final `:1028`) plus a full-tree
  `content_sha256` (`:216-239`), persisting everything to `run.attrs["cluster_output_staging"]`
  (`:1006,:1016,:1040`).
- The waste: that receipt has **no self-digest** (`:975-1002`) and is on every manifest's
  exclusion list — `analysis/chaser_component_publication.py:127`,
  `shared/zarr/subject_mask_cache_publication.py:130`,
  `shared/subject_shape_coordinate_publication.py:1238`,
  `shared/zarr/subject_mask_quality_manifest.py:72`, `analysis/track_kinematics.py:4617` — so
  consumers cannot trust it and re-scan instead.
- The fix pattern is already proven in-repo: the mask coordinate validation receipt
  (`1e7b2eb9`) gave three consumers a receipt fast path replacing their rescans
  (`shared/subject_mask_coordinate_publication.py:1616,:3144`,
  `shared/refined_subject_mask_coordinate_publication.py:5408`).

**Action:** seal `cluster_output_staging` (add `record_sha256`), bind that digest INTO the
manifests instead of excluding it, and convert downstream deep-revalidation sites to receipt
verification. End state: exactly one full scan per artifact (at publication); everything after
verifies a hash. This single change removes most redundant *compute* in the system.

### Cluster 1 — manifest validator fan-out (cheap per call; fix ownership, not cost)

`validate_*_run_manifest` calls are microsecond dict checks; the redundancy signal is that every
caller re-validates because no type records "already validated." Production call sites
(imports/`__all__`/tests/`diagnostics/benchmark_*` excluded):

| Validator | Sites | Locations |
|---|---|---|
| `validate_subject_mask_core_run_manifest` | 16 | `subject_mask_core_publication.py:770,820,854,2481,2579` · `subject_mask_coordinate_publication.py:1659,2807,2845` · `subject_position_mask_source.py:412,494,511` · `subject_mask_cache_publication.py:373,952` · `refined_subject_mask_coordinate_publication.py:5422` · `subject_mask_coordinate_successor.py:326` · `subject_mask_bundle_publication.py:226` |
| `validate_refined_detection_run_manifest` | 12 | `refined_detection_manifest.py:1678,1853,1886,2073,2085,2103,2175` · `crop_manifest.py:241,1203` · `clipped_refined_detection_finalization.py:280` · `refined_detection_compaction.py:81` |
| `validate_crop_run_manifest` | 11 | `crop_manifest.py:1125` · `crop_consumer.py:63` · `crop_image_source.py:1422` · `keypoint_manifest.py:175` · `historical_geometry_only_crop_adapter.py:971` · `subject_mask_core_publication.py:315` · `subject_mask_bundle_coordinate_authority.py:392` · `keypoint_bundle_activation.py:478` · `publish_recording_bundle.py:717` · `full_duration_canary.py:351` |
| `validate_canonical_detection_run_manifest` | 8 | `canonical_detection_manifest.py:819,1106,1186,1235` · `registered_detection_gate.py:251` · `subject_position_detection_source.py:305` · `registry/maintenance.py:107` · `finalize_crimson_canonical_v3_companion.py:84` |
| `validate_keypoint_run_manifest` | 10 | `keypoint_manifest.py:546,571,836` · `keypoint_coordinate_publication.py:2943` · `keypoint_coordinate_successor.py:565` · `subject_position_keypoint_source.py:509` · `refined_keypoint_manifest.py:736` · `keypoint_bundle_activation.py:84` · `registry_finalize.py:210` |

**Action: parse-don't-validate.** Loader boundaries (`open_persisted_*`/`load_*`) validate once
and return a `ValidatedManifest` wrapper; delete downstream re-calls that receive it. Keep the
legitimate write-then-verify pairs (e.g. `subject_mask_core_publication.py:2481` pre-import +
`:2579` on the persisted copy). Collapses roughly half of these ~57 sites and makes forgetting
validation impossible — which is what each call was defensively guarding against.

### Cluster 3 — track_kinematics three merkle roots vs. siblings zero (level UP, not down)

- Producers: `materializers/track_kinematics.py:741,1076`; `analysis/track_kinematics.py:11875,
  12296`, persisted `:11894,:11897,:12468,:12472`. Verifiers: `materializers/…:652` (note
  `verify_physical_payload=False`), `analysis/…:11971,12013,12450,12457`.
- Siblings already producing the receipt's required input (`exact_decoded_validation` copy
  report) with no receipt built: `materializers/subject_shape.py:1010`,
  `materializers/bout_kinematics.py:451`, `materializers/eye_angles.py:69`.

**Action:** the three roots answer three different questions (logical drift / bitrot / metadata
tamper) at the cost of one hash pass — keep them; the asymmetry is under-adoption. Add
`build_payload_integrity_receipt` to the three siblings (a few lines each; input exists). One
honest trim: decide what the `physical_payload` root is FOR — the hot path never re-verifies it,
so either point a scheduled job at it or document it as write-once forensic evidence, not an
active guarantee.

### Cluster 4 — four crop contracts (rides the already-decided resolver work)

Enumerated in `crop_contract_split_audit_2026-08-24.md` §1/§4. Deletions that fall out once the
resolver branches land: the adapter's monkey-patch installation path
(`historical_geometry_only_crop_adapter.py:1217-1314`) plus the
`CROP_PLACEMENT_PADDED_PRODUCER` special-casing (`pixel_frame_authority.py:174-175`), and
profile D's new-publication entry points (D stays readable via the proxy-successor bridge).

### Cluster 5 — pure deletions / dedups (no guarantee lost)

- Duplicate `build_ssh_bsub_runner`: `lsf/backend.py:201-227` vs
  `cluster/clipped_inference.py:2498-2524` — keep one.
- `montage/registry.py:25-154` re-implements `reporting/selection.py`'s registry query (and
  `reporting/report_registry.py:17` imports the private `_connect_read_only`) — extract one
  shared selection module.
- Never-used speculative machinery: `ALL_ENDED` (`lsf/models.py:20`), `EntityScope.CHASER` /
  `STIMULUS_STEP` (`reporting/models.py:25-31`).
  `build_clipped_storage_keypoint_chain_fragments`
  (`clipped_storage_finalization.py:260-298`, zero callers) becomes live wiring if the
  keypoint→geometry rebase lands — TAG it with that dependency rather than deleting.
- Dead: `load_offline_position_source` (`analysis/track_kinematics.py:1391`, no callers);
  `analyze_goodcopbadcop_immobility_artifact.load()` (`:45-49`, returns `None` — module cannot
  run as written).

### Order of work

1. Cluster 2 (seal + bind the atomic receipt; convert rescans to receipt checks) — the only
   cluster where redundant compute and unbound evidence coincide; pattern proven in-repo.
2. Cluster 1 (parse-don't-validate) — mechanical, agent-friendly, large site count.
3. Cluster 3 (sibling receipts) — small, high leverage.
4. Cluster 5 (deletions) — whenever an agent has spare capacity; pair with the subtraction queue.
5. Cluster 4 — no separate work; rides the crop resolver implementation.
