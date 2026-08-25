# Redundancy & Multiple-Sources-of-Truth Campaign

**Date:** 2026-08-24
**Method:** 4 parallel read-only hunters (constants/config, logic/helpers, runtime data, scripts/CLIs), each primed with the existing docs so this catalogues only NEW findings plus status updates.
**Companions:** `pipeline_survey_2026-08-24.md` §4 (validator fan-out, receipt-vs-rescan, crop contracts, known dedups), `crop_contract_split_audit_2026-08-24.md`, the 2026-08-21 subtraction queue.
**Campaign rules:** one definition per fact; one implementation per guarantee; when a receipt/engine covers something, delete the rescan/script it replaces; superseded generations are deleted on sight, not kept alongside their guarded replacements.

## Meta-findings (read first)

- **`docs/diagnostics/subtraction_queue_2026-08-21.md` is NOT in this branch's working tree** —
  it exists only in git history (commit `d065360c`). Restore/merge it before the campaign starts;
  three of four hunters hit this wall independently.
- The "six definitions of latest" from the 2026-08-21 enforcement review is now
  **~12 distinguishable semantics** at HEAD (strict/legacy/authoritative trio in
  `zarr_run_completion.py:597-672`; `registry/maintenance.py:4097-4134`; `registry/db.py:1142`;
  `registry/extractors/masks.py:232`; `shared/run_resolution.py:21-27`;
  `canonical_detection_manifest.py:1202-1261`; `keypoint_bundle_activation.py:145-262`;
  `incremental_crop.py:386-393`; the crop `latest`/`latest_materialized`/`latest_any` triple;
  bare `attrs['latest']` in production writers `refine_detect.py:1471`, `detect_quality.py:865`,
  `arena_assignment.py:753`, `refine_keypoints.py:1162`). The problem is growing, not shrinking.
- Canonical registry path literals: ≥8 sites at HEAD (two hunters counted 8 and 11 with slightly
  different inclusion rules; either way it grew from ~6). Only 2–3 honor
  `$PALETTE_REGISTRY_PATH`; none consult `RegistryPaths.from_env`.
- Docs-as-truth: confirmed negative — nothing parses docs/ at runtime. The inverse risk is real:
  the 1.6 mm/s noise floor exists only as prose while the thresholds it should constrain are set
  independently per module (Wave 1.3).

---

## Wave 1 — Facts that already diverge or are already wrong (science first)

1. **`NEAR_MM`: same name, two values.** `analysis/analyze_goodcopbadcop_per_fish.py:74` = 25.0;
   `analysis/plot_goodcopbadcop_occupancy_heatmaps.py:42` = 15.0. Two figures both claiming
   "near the chaser." Decide the value (or rename one), define once in
   `goodcopbadcop_common.py`.
2. **Raw centroid speed re-derived in the production analytics export.**
   `utils/export_cross_recording_analytics.py:3262-3282` (`_frame_speed_mm_s`) recomputes
   px-diff speed — the ~1.6 mm/s-noise-floor class — with no mode switch, no `speed_source`
   label, feeding the `goodcopbadcop_speed_distance_bins` export table (`:3323`) that dashboards
   read. Found independently by two hunters. Fix alongside the already-queued
   `chaser_escape_freeze_summary` item (pipeline survey Finding 1).
3. **Speed thresholds retyped per module** — `1.0` under two names
   (`chaser_near_field_occupancy.py:77`, `chaser_response_regimes.py:89`;
   `chaser_escape_freeze_summary.py:55` as `DEFAULT_LOW_SPEED_THRESHOLD_MM_S`), `2.0` twice
   (`chaser_bout_response.py:112`, `chaser_response_regimes.py:92`). One
   `analysis/speed_policy.py` owning thresholds AND the 1.6 noise-floor constant
   (currently comment-only in 4 files). Do inside the Wave 1.2 fix.
4. **Frame-count resolvers with opposite precedence.** `shared/frame_domains.py:493-503` is
   source-preferring; `analysis/stimulus_epoch_runs.py:139-143` is clip-preferring over the same
   six attr names — different integers on any materialized archive. Also:
   `duration_seconds` vs `n_frames` disagree by construction wherever raw arrays exist
   (`import_video_metadata.py:516-522`); `frame_domains.py:372-375` records conflicts nothing
   reads; `_first_positive_int:182` accepts 0; `stimulus_epoch_runs.py:137` fabricates
   `fps or 30.0`; fps fallback `60.0` hardcoded in `visualize_refined_detections.py:386`,
   `visualize_refined_online.py:109`, `analysis/compute_speed.py:106`. **Lever:** one
   FrameDomains accessor with one precedence; conflicts raise.
5. **Acquisition authority seeded from the wrong count.**
   `utils/preflight_source_video_metadata_backfill.py:176-201` seeds
   `source_video_metadata.total_frames` from plain `root.attrs["total_frames"]` — which is the
   *clip* count on materialized archives — and `assert_verified()` then confirms it because it
   re-derives from the same blob (`pixel_frame_authority.py:2167-2179`). Guard the seeding;
   add one authority-vs-plain-attr cross-check.
6. **Registry `recordings` row: two writers, opposite precedence, asymmetric SQL.**
   `registry/db.py:748-761` prefers zarr>manifest; `registry/maintenance.py:824-960` reads
   manifest only; `maintenance.py:902` hard-overwrites `session_uuid` while `db.py:2618` does
   the opposite. Last-writer-wins on identity. **Lever:** one recordings-row writer with one
   precedence.
7. **`backfill_stage_provenance.py:29` (`OFFLINE_RUN_GROUPS`) omits both subject-mask families**
   while `report_acquisition_crop_video_roi_readiness.py:22` counts them — a provenance backfill
   silently skips subject masks today. Both lists replaced by `shared/stage_run_groups.py:8`
   (`STAGE_RUN_PARENTS` — the canonical mapping with only 4 importers).
8. **Epoch windows: 7 copies, digest-free binding.**
   `chaser_distance_runs.py:528-546,729-730` copies stimulus epoch windows with name+path
   provenance only, and `except ValueError → ()` silently yields zero windows; consumers read
   the copy (`chaser_radial_occupancy.py:254`, `chaser_analysis_figures.py:88`,
   `goodcopbadcop_interactive.py:2160,2572`; a 7th copy in
   `chaser_epoch_behavior_summary.py:1782-1840`). Bind like
   `resolved_epoch_selection.py:506,580` does.
9. **Nominal-circle stragglers (production):** `analysis/swim_bout_statistics.py:719` feeds
   `CalibrationData.arena_diameter_mm` from nominal geometry;
   `analytics_exports/baseline.py:323,464,568` hardcodes `"circle"` into exported tables;
   `diagnostics/prepare_detect_training.py:392-402` gates training selection on it. (Adds to the
   two visualization offenders in the pipeline survey.)

## Wave 2 — Delete superseded generations (these are enforcement bypasses, not clutter)

1. `utils/set_detect_review_status.py` (174L) and `utils/set_keypoint_review_status.py` (156L) —
   pre-guardrail writers of the same attrs as `accept_detect_review` (612L, 08-08) /
   `accept_keypoint_review` (213L), with `latest` defaults and no strictness. Delete after doc
   repoint. (`set_crop_review_status`, untouched since 2026-04-14, is already on the dead list.)
2. `utils/import_video_metadata.py` + `import_video_metadata_batch.py` +
   `utils/backfill_h5_metadata.py` — write overlapping attrs without the preflight/rollback
   machinery of the current `preflight_source_video_metadata_backfill` /
   `apply_source_video_metadata_backfill` pair. Superseded twins; delete or reduce to wrappers.
3. `utils/registry_rescan.py` — a second implementation of `reconcile_sweep`'s job that still
   calls the old `scan_zarr` path (`:46-48`) and is the registry's weakest write path
   (apply-by-default, no backup). Collapse into a `reconcile_sweep` mode, don't just harden.
4. `utils/dish_mask_registry_sync.py` (156L, untouched 2026-06-16) — the one profile-sync
   survivor the 2026-07 reconcile collapse missed; 2 importers. Repoint to
   `reconcile_dataset_from_root`, delete.
5. Alias shims: 5 goodcopbadcop `run_*` re-export stubs + `run_pose_training_pipeline.py` /
   `prepare_pose_training_from_registry.py`. Delete after doc grep.
6. Dead: `utils/training_image_profile.py` (1050L, 0 importers);
   `analyze_goodcopbadcop_immobility_artifact.load()` returns None (already queued).
7. Review-mutating tools that fall back to `sorted(keys)[-1]` and can write review state onto
   the wrong run (`accept_detect_review.py:91` and ~15 siblings) — replace the fallback with the
   canonical resolver as part of Wave 3.1.

## Wave 3 — One implementation per guarantee (helpers)

1. **Run resolution: adopt or kill `shared/run_resolution.py`.** Three competing shared
   resolvers exist (`zarr_run_completion.resolve_authoritative_run_name` 27 importers;
   `zarr_helpers.resolve_zarr_run` 19 importers, different semantics; `run_resolution.resolve_run`
   — the built-but-unadopted unifier, **3 importers**), plus ~25 local `_resolve_run` defs,
   **41 `sorted(...)[-1]` run-pick sites** (hotspots: `check_recording_steps.py` ×10,
   review/backfill CLIs ~20, `goodcopbadcop_common.py:255` feeding cohort analysis,
   `registry/maintenance.py:4520`), and `_pick_refined_parent`/`_resolve_refined_parent`
   defined **30×** with the family-alias list retyped in each. Decide the unifier, then burn the
   41 sites down against it.
2. **Digest unification — prerequisite for the receipt-spine work (survey §4 Cluster 2).**
   `canonical_json_sha256`: 20 definitions, ≥3 incompatible preimages
   (`manifest_digest.py:30`/`zarr_payload_receipt.py:47` agree; `legacy_arrow.py:67` forces
   ASCII; `chaser_profiles.py:106` allows NaN; `selected_calibration.py:1167` salts).
   Array content-sha256: 20+ definitions, 3 incompatible schemes (`coordinate_identity.py:712`
   self-describing header — the keeper; `track_kinematics_storage.py:380`;
   `occupancy_candidate_execution.py:92` bytes-only, shape-collision-prone). Also the
   algorithm-id constant `"sha256_c_contiguous_bytes_v1"` exists as 6+ per-family constants
   agreeing by luck, and the hash idiom is retyped in ~112 files with no shared
   `array_content_sha256()`. **Receipts across families are currently mutually incomparable.**
   Migration note: never re-verify a persisted digest with a different variant than wrote it.
3. **SQLite read-only connect:** no shared helper exists; ≥8 private `_connect_read_only` defs,
   three URI-quoting styles, **11 unquoted `f"file:{path}?mode=ro"` sites** (latent bugs), some
   missing `PRAGMA query_only`. Promote one helper (as_uri + query_only) into `registry/`.
   Complements the known "21 writable ro-connects."
4. **ffprobe:** two "shared" modules (`shared/import_video_metadata.py` vs
   `diagnostics/video/ffprobe.py`) plus 7 local wrappers; the
   `build_review_proxy_videos.py:235` twin that omits colorimetry **is the direct cause** of the
   color-range finding — the fix is calling the shared colorimetry probe that already exists.
5. **pixels-per-mm resolution:** divergent fallback chains
   (`dish_mask_boundary.py:33-43` with conflict detection vs
   `plot_sampled_component_contours.py:283-300` different second key, no check; plus projector-
   space variants). Same hazard shape as the arena incident. One resolver per coordinate space.
6. **Atomic small-file writer:** 13 local temp+`os.replace` implementations (fsync behavior
   diverges); one `shared/atomic_file.py`.
7. Mechanical: `_mirror_authoritative_approval` ×4 verbatim (`tune/*backend*.py`), git/env
   collectors ×12 (some omit the dirty flag — their provenance can't distinguish dirty-tree
   runs), `_selector_snapshot` ×20 with 3 incompatible schemas (blocks queryable selector
   audits), micro-helper long tail (`_utc_now` ×72 with mixed timestamp formats,
   `_sha256_file` ×51, `_read_json` ×55, …) — handle via a **lint ratchet** (ban new local defs
   of a blessed-name list), not a mass rewrite.

## Wave 4 — Single-source constants modules

1. **`shared/paths.py`**: recordings root (`/nvme1/recordings` retyped **77×**, 14 independent
   `DEFAULT_RECORDINGS_ROOT` constants, ~30 retyped env-fallback idioms, ~40 bare literals, and
   two files defaulting to the `/groups` store under the same parameter name); registry path
   (≥8 literals); repo path (5 constants); figures dir (3, two bypassing the existing
   `goodcopbadcop_common.py:209` helper). Fold survey finding #14 in here.
2. **`cluster/lsf/site.py`**: submit host `login1-citrus-poller` (15 py + 30 shell sites; only
   4 py sites honor `$PALETTE_LSF_SUBMIT_HOST` — the env var works from bash but not most
   Python), queue names (`gpu_l4` ~15 sites + one hardcoded allowed-set), GPU request string ×2.
3. **`shared/coordinate_contract.py`** — opening commit of the crop-resolver work: the
   `"coordinate_contract"` attr name (116 raw lines, only constant lives in a leaf), the
   `"canonical_v2"` value (**6 differently-named constants** + ~36 raw lines), and the digest
   algorithm id (Wave 3.2).
4. **Public attr-name constants + ratchets**: `"stage_selector_eligible"` — 728 raw lines, only
   constant is *private* (`detection_producer_lifecycle.py:750`); `"latest_complete"` 284 raw
   lines vs 40 constant uses; `"palette_run_completion_status"` 179 raw (+ a duplicate private
   constant `inventory_analysis_components.py:33`, + 6 raw `"complete"` comparisons);
   `"authoritative_run"` 95; `"crop_storage_mode"` 89 (zero constants); `"cluster_output_staging"`
   32 (zero constants — extract as the first commit of the receipt-sealing work);
   `"run_manifest"` as **12 per-family constants**; `"verified_track_motion"` 9 raw.
   Pair each ratchet with the wave that touches its subsystem.
5. **`is_network_path()`**: the `("/groups/", "/nrs/")` guard copy-pasted 13× with two variant
   vocabularies (`/misc/public` in 2 files, `/groups/`-only in 1) — a new mount is invisible to
   whichever copy wasn't updated.
6. **Vocabulary dedups**: labeling operator-gate status values verbatim-triplicated
   (`web_policy.py:1014` = canonical, `admin_dashboard.py:137`, `web.py:5640`) — the
   target-production surface; `REVIEW_STATES` duplicated (2 review CLIs + derived subset in
   `web_runtimes.py:984`); `reporting/discovery.py:34` `_INCOMPLETE_STATUSES` hand-rolled with
   two members no producer emits — derive from `RUN_STATUS_*` (root cause of the known
   denylist fail-open); schema-id `"palette.stimulus_epoch_windows.v1"` as 2 constants + 4
   literals.
7. Adopt `STAGE_RUN_PARENTS` (Wave 1.7): the legacy-alias tuples
   `("refined_detect_runs","refined_runs")` ×7 and
   `("refined_keypoints_runs","keypoints_refined_runs")` ×6 all replaced by imports;
   `"refined_detect_runs"` as a raw literal in 98 files can ratchet.
8. Encoder-default split: `build_review_proxy_videos.py:31` (`libx264`) vs
   `submit_review_proxy_videos_sharded_bsub.py:522` (`h264_nvenc`) — unify as part of the
   color-range fix.

## Wave 5 — Data-mirror policy (same fact, two stores)

1. **Mirror-equality or no mirror.** The exemplary pattern exists:
   `acquisition_publication_status.py:304-307` (transactional dual-stamp + exact-equality on
   load). No other root↔raw_video mirror pair (colorimetry, encoder tags, counts, codec — full
   inventory `import_video_metadata.py:450-532`) has ANY comparator, and readers split between
   root-only, raw-only, and two opposite fallback orders (registry fps is raw_video-only,
   `registry/db.py:886`). Either compare on load or stop dual-writing. One bypass reader to fix
   regardless: `cluster/native_detection_authority.py:76-81` reads only the raw_video copy of
   the publication status with no root comparison.
2. **Digest-coverage parity across manifest families.** Canonical + refined detection strip
   `attributes` from `metadata_declarations_digest`
   (`canonical_detection_manifest.py:568-577`, `refined_detection_manifest.py:320-329`) so
   consumers read unsealed attrs (`canonical_detection_benchmark_input.py:210-220`,
   `detection_coverage_dashboard.py:296`); subject-mask families redact `status`
   (`subject_mask_core_publication.py:699-710`) so failed-vs-complete divergence is
   undetectable. Gold standards to copy: track motion rebuild-and-compare
   (`track_kinematics.py:7514-7613`), chaser byte-compare
   (`chaser_component_publication.py:559-609`), and the tree's only status==manifest check
   (`materializers/subject_position.py:720-752`).
3. **`raw_video.total_frames` means source count in metadata-only archives but sampled count in
   PyNvVC training zarrs** (`import_video_metadata.py:459` vs
   `import_sampled_training_pynvvc.py:387`); blind readers `tune/mask_tuner.py:191`,
   `refined_detect_curation.py:913`, `migrate_legacy_detect_labels.py:173`. PyNvVC zarrs also
   omit root width/height/fps (silent 640×640 fallback in `detect_quality.py:216-217`; registry
   fps None). Disambiguate the attr or stamp the domain.
4. Clipped shells synthesize identity from different sidecars per writer, and
   `create_clipped_training_zarr.py:323-327` copies 16 identity attrs from a donor zarr **with
   no same-recording check**; `repair_recording_identities.py` fixes `recording_id` only.
5. Write-only mirrors: `cluster_output_staging.parent_attrs_after`
   (`atomic_run_publisher.py:1034-1037`) is guaranteed stale post-activation and never read —
   drop it or read it. `publish_acquisition_frame_clock` disables its own rows-vs-video check
   (`expected_frame_count=None`, `acquisition_frame_clock.py:856-859`) — enable it.
6. Detection completeness lives in ~7 systems with no closing loop; the registry finalizer's
   triple-check **excludes detection** (`storage_contract_catalog.py:447-451`), and
   `registry/maintenance.py:4127-4133` can select a run every zarr resolver rejects (checks
   completeness, not eligibility). Fold detection into the finalizer's coverage.
7. Remaining `zarr.consolidate_metadata` bypass sites (split-brain feeders), cheap to close:
   `stimulus_response_storage.py:591`, `tail_kinematics_storage.py:274`,
   `migrate_refined_subject_mask_editable_draft.py:64`,
   `canonical_detection_benchmark.py:225`.

## Wave 6 — Script/CLI collapses (see hunter report for full clusters)

1. **Cohort-plan lib** (`utils/cohort_plan_lib.py`): `publish_accept_all_refined_detection_batch`
   vs `publish_canonical_detection_successor_batch` are **78% line-identical** with a third ~50%
   copy (`publish_crop_geometry_candidate_batch`) — an actively multiplying family; extract
   before the fourth copy.
2. **Batch-review loop lib**: 7 interactive review wrappers hand-roll the same resumable loop
   (~4.5k LOC; delta = which visualizer + which status attr).
3. **Keypoint backfill family**: 11 one-shot scripts, one CLI shape, two of them split pairs of
   single repairs — merge to one `--field` CLI or verify-complete-and-delete.
4. **Acquisition-authority CLI**: `seal|repair|audit` × `external|clipped` replacing the
   migrate/repair pair + clipped repair.
5. **Merged analysis importer**: `--from-organize-log | --scan-root` collapses
   `import_organized_recordings_analysis` + `import_recordings_analysis` (stamping lives in the
   shared library; nothing is lost). Training/intake/clipped importers stay distinct.
6. **Submit-script dedup**: delete spent benchmark/canary submitters first (shrinks the
   bsub_common extraction by a third); collapse the 3 flat-roi-cache wrappers and other
   same-module multi-wrappers; `submit_import_recordings_training_bsub.sh` (450L hand-rolled
   submission manifest) is the largest single duplicate of `lsf/`.
7. **Reports of the same fact**: three zarr-size reporters (keep `audit_zarr_array_sizes`,
   delete `zarr_size_report` 2025-10, `report_zarr_storage` 2026-02); six pre-epoch
   `check_*_runs.py` diagnostics folded into `check_zarr_run_completion --stage`;
   `list_unapproved_*` pair → `--family` flag; the six browse-the-store entry points +
   Datasette + TUI need an explicit "which three survive" decision.
8. Serial-finalizer twins (`finalize_crop_flat_roi_cache_batch_registry` vs
   `finalize_refine_keypoints_batch_registry`, 30% similar = divergent twins): flag "no third
   copy"; parametrize when next touched.

## Suggested execution order

1. **Wave 1** (facts already wrong): 1.1 NEAR_MM, 1.2+1.3 speed policy (with the freeze-summary
   fix), 1.7 run-group list, 1.4 frame-count accessor, 1.6 recordings-row writer, 1.5 authority
   seeding guard, 1.8 epoch binding, 1.9 nominal-circle stragglers.
2. **Wave 2** (bypass deletions) — cheap, pure risk reduction.
3. **Wave 3.2** (digest unification) — gates the receipt-spine work in survey §4.
4. **Wave 3.1** (run-resolver adopt-or-kill) — gates the ~12-latest cleanup and Wave 2.7.
5. **Waves 4–5** paired with whichever subsystem wave touches them (ratchets, not churn).
6. **Wave 6** as agent-capacity filler; cohort-plan lib first (actively multiplying).
