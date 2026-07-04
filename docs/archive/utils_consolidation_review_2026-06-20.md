<!-- ARCHIVED 2026-07-04: dated point-in-time diagnostic snapshot, retained for history only. -->

# `src/fisheye/utils/` Consolidation Review — what can be merged / shared (2026-06-20)

## Method

`utils/` is 268 files / ~151K LOC — the application layer the whole-repo review flagged as
P2-7. Six read-only agents each took a functional family (mutation/maintenance, batch runners,
training export, audit/validate/check, query/discovery, import/materialize/finalize/review/plot)
and hunted specifically for merge/share opportunities, cross-referencing what already exists in
`shared/` and `cli/`. Claims below were verified by reading function bodies, AST body-equality
diffs, and grep counts. Nothing was modified.

## Headline

The dominant finding is **not** family-specific — it is a layer of ~8 micro-helpers copy-pasted
dozens of times each, with canonical homes that already exist and are almost entirely unused.
Worse than bloat: several copies have **drifted**, so "discover the recordings" or "slug a name"
means different things depending on which tool you run — a correctness hazard.

Realistic reduction: **~6,000–8,000 LOC of dedup** plus **~4,500 LOC of eye-mask deletion**
(gated), without losing any capability — and, more valuably, collapsing dozens of independent
maintenance surfaces into a handful. The work also naturally produces the `shared < apps` split
the architecture review wants.

---

## Tier 0 — Cross-cutting micro-helpers (do FIRST; biggest aggregate win, lowest risk)

These span the whole directory, appeared in **every** agent's report, and de-risk every Tier-1/2
merge (a base class over un-deduped leaves just hides copy-paste). All are behavior-identical
clones; the home mostly already exists.

| Helper | Copies in utils | Canonical home | Status |
|---|---|---|---|
| `_iter_zarr` / `_iter_h5` filesystem walk | ~72–75 | `shared/zarr_discovery.py::iter_filesystem_zarrs` | exists; **only 6 files use it** |
| `_infer_zarr_use` (analysis/training) | ~55 | none | **add** `infer_zarr_use()` to `shared/zarr_helpers.py` |
| `_resolve_roots` / recording-root + `PALETTE_RECORDINGS_ROOT` default | ~30–45 | `shared/environment.py::resolve_recording_roots` | exists; under-used |
| `_load_paths_file` / `_read_file_list` | ~7 | `shared/zarr_discovery.py::load_path_list` | exists; under-used |
| `_utc_now` | 33 | `shared/batch_logging.py::utc_now` | exists; **unused by the 33** |
| `_write_json` / `_read_json` / `_json_default` / atomic `os.replace` write | 27 / 19 / 11 / 14 | none (json_safety has only `json_attr_safe`/`strict_json_dumps`) | **add** `write_json_atomic`/`read_json`/`json_default` to `shared/json_safety.py` |
| `_norm_text` / `_as_float` / `_as_int` / `_coerce_int` | 37 / 43 / … | `shared/type_conversions.py` (partial), `zarr_helpers.safe_int` | consolidate as `norm_text`/`as_float`/`as_int` |
| `_safe_component` / group-name slug | 6 | none | add to `shared/zarr_helpers.py` |
| `_progress` rich-Progress factory + import guard | 6 byte-identical | none | add `shared/progress.py::make_progress()` |
| `_prompt_continue` (review backends) | 7 | none | fold into the review driver (Tier 1) |

**Drift hazards to fix while consolidating (not cosmetic):**
- `_iter_zarr` globs disagree — `zarr/*.zarr` vs `*/zarr/*.zarr` vs `rglob("*.zarr")` (some dedup,
  some don't). Generalize the shared helper with an optional `glob_pattern`/`dedupe` and confirm
  each call site's intended depth (so this is a per-file migration, not a blind sed).
- The skip-existing flag is spelled three ways across runners — `--no-skip-existing` / `--overwrite`
  / `--force-new`. Standardize on one (keep aliases).

**Also add to `cli/shared_args.py`** (it currently only has `add_registry_discovery_args`): a
`add_recordings_args` (the `paths`/`--recursive`/`--zarr-use` triad, present in ~12 files) and
`add_report_output_args` (the `--json`/`--strict` triad, in ~21 files). The existing
`add_apply_dry_run_args`/`add_log_args` are barely adopted — sweep them in.

**Saving:** conservatively **~2,000–3,000 LOC** across the directory, near-zero risk, and it
removes the discovery/slug drift. This is the best impact/risk ratio in the codebase.

---

## Tier 1 — Whole near-duplicate scripts → one parameterized command

Ranked by saving × (inverse) risk. Each is a family of scripts differing only by a stage/component
constant and one or two hooks.

1. **`review_*_batch` cluster → `shared/subject_mask_review_batch.py` driver.**
   `review_subject_body_masks_batch.py` (1038 L) and `review_swim_bladder_masks_batch.py` (936 L)
   are **~90% byte-identical** (`diff` = ~51 differing lines: `COMPONENT_NAME`, the viewer module,
   and a body-only run-semantics/`--sam` block that is a strict superset). `review_keypoints_batch.py`
   shares the same `ReviewPlan`/`_build_plans_from_registry`/`_iter_zarr`/scope skeleton. A
   `ReviewBatchDriver(component_name, viewer_module, build_viewer_cmd, extra_args)` collapses them;
   bladder gains run-semantics support for free. **~830 LOC** (×3–4 files: ~1,200–1,400). Low risk,
   well-bounded. *Flagged independently by two agents — highest-confidence merge.*
2. **3 `backfill_*_profiles.py` → `backfill_profile.py --kind`.** `backfill_detection_profiles.py`
   (306), `backfill_eye_mask_profiles.py` (344), `backfill_keypoint_profiles.py` (342) are ~90%
   identical (same argparse, `_iter_zarr`, `_resolve_roots`, `_run_exists`, `_process_zarr_path`);
   differ by run-parent key, writer/summary fns, error type. Parameterize by a `ProfileKind`. Drop
   the eye_mask one (deprecated). **~590 LOC.** Medium risk (write paths; dry-run-default mitigates).
3. **2 `finalize_*_refinement_artifacts.py` → `shared/refinement_artifact_finalize.py`.**
   `finalize_refinement_artifacts.py` (684) and `finalize_keypoint_refinement_artifacts.py` (593)
   are detect-vs-keypoint twins: ~200 L of identical helpers + same `FinalizeRow` + same `_build_rows`
   state machine (differs only by review-status attr key). **~350–400 LOC.** Medium risk (idempotency
   signatures — needs both unit suites).
4. **`sync_*_profile_registry` trio shared layer.** `sync_{detection,eye_mask,keypoint}_profile_registry.py`
   share line-for-line `_normalize_text/_as_int/_as_float/_coerce_mapping/_to_json_text`, zarr-open,
   run-selection, argparse; only `_build_profile_payload` (the column mapping) differs. Extract the
   shared layer; keep three thin payload callbacks. **~350 LOC.** Medium risk (registry writes).
5. **`run_*_with_registry_model` trio → `run_registry_model_common.py`.**
   `run_{detect,keypoints,eye_masks}_with_registry_model.py` share identical `_pick_best_candidate`,
   `_resolve_output`, `_candidate_payload`, `_write_model_resolution_provenance` (modulo parent
   group name) and arg block. **~350–450 LOC.** Low–moderate risk.
6. **3 `set_*_review_status.py` → `set_review_status.py --target`.** Identical argparse + payload
   dict; **also fixes real divergence** (keypoint uses atomic `.attrs.put()` + writes a duplicate
   `timestamp`/`timestamp_utc`; crop/detect use direct assignment + bare `zarr.open_group`).
   **~200 LOC** + normalizes three write semantics. Low risk.
7. **4 `validate_*_training_zarr.py` → `validate_training_zarr.py --stage`.** Byte-identical thin
   CLI shims over `export_*.validate_merged_*`; differ by one import line. Drop eye_mask. **~110 LOC.**
   Lowest-risk quick win.
8. **2 `list_unapproved_*` → one `--stage`.** `list_unapproved_analysis_zarrs.py` (372) and
   `list_unapproved_keypoint_analysis_zarrs.py` (240) are ~75% identical (parent group + status key
   differ). **~180–220 LOC.** Low risk.

---

## Tier 2 — Extract reusable cores into new/existing shared modules

- **`_register_merged_dataset_in_registry` (4-way) → `shared/training_export.py`.** All four
  `export_*_training_zarr.py` carry a 55–65 L copy; detect-vs-keypoint differ by **2 lines**
  (`task_type`, `producer`). One `register_merged_dataset_in_registry(*, task_type, producer,
  invocation_merge=None)`. **~170 net LOC** and kills provenance-stamping drift (correctness-
  sensitive). Highest value-per-line in the export family.
- **Training-export stateless helpers → `shared/training_export.py` (new).** `_json_dict`/`_json_list`
  (byte-identical ×3), `_sha256`, `_compute_split_indices`, `_normalize_chunks`, `_copy_progress`,
  `_write_string_array` (reconcile the 2 drifting variants), `_clean_slug` (2 drifting variants —
  a real slug-drift bug). This also gives `export_subject_mask` a real home for the 5 helpers it
  currently imports **sideways from `export_eye_mask`** (the anti-pattern blocking eye-mask deletion).
  **~470–550 LOC** across export+prepare.
- **Training-data-card render → `shared/training_data_card_render.py` (new).** `plot_*_training_data_card.py`
  share ~15 near-identical helpers (histogram/heatmap parse+plot, Agg guard, argparse/dry-run `main`).
  They write loose PNGs (can't use `shared/plot_artifacts.py` directly) — bridge with a
  `render_fig_to_png_bytes` + `run_card_plot_cli(domain, ...)`. **~700–850 LOC.**
- **Audit/report framework → `shared/audit_report.py` (new).** 14 verification files hand-roll a
  per-item dataclass + `{valid, errors[], warnings[]}` aggregate + json/text dual render + exit-code
  logic, with an inconsistent status vocabulary and inconsistent exit codes. Provide `Severity`,
  `Finding`, `Report.exit_code(strict=)`, `emit(report, json=)`. **~400–700 LOC** + consistent
  severity/exit semantics (preserve each tool's current code to avoid breaking callers).
- **LSF/bsub + scratch staging → `cli/lsf.py` (new).** The only real LSF code lives duplicated in
  `submit_clipped_detect_refine_plan_bsub.py` and `submit_review_proxy_videos_sharded_bsub.py`
  (`_parse_bsub_job_id`, `_bsub_args`, `_submit_bsub`, LSB_* provenance, job-cache shell prelude).
  **~150–200 LOC** and establishes the shared LSF home.
- **Promote `resolve_detect_model._load_candidates/_load_target_profile/_resolve_recording_id` →
  `registry/model_resolution.py` (public).** These underscore-private helpers are imported by **6+
  batch runners** — a CLI's internals are the de-facto model-resolution library. Public-ify, then
  put `resolve_subject_mask_model` on the same skeleton. Fixes the "private-as-public-API" smell
  the whole-repo review also flagged (P3-14).
- **Recording/h5 discovery → `shared/recording_discovery.py` (new).** `_read_h5_meta`/`_derive_camera_id`/
  `_find_h5_files`/`_select_cam_video` copied verbatim between training & analysis importers (~60 L each).

---

## Tier 3 — Deletions (eye-mask deprecation; delete-don't-refactor)

Per project direction eye masks are legacy. These are **deletion**, not consolidation, targets —
but **gated**: `export_subject_mask` currently imports 5 helpers from `export_eye_mask`, and
`core/pipeline.py:800,812` / `run_eye_mask_training_pipeline.py` / `validate_eye_mask_training_zarr.py`
still call eye-mask export. **Sequence: land the Tier-2 `training_export.py` extraction first
(severs the subject→eye import), confirm the pipeline no longer requests eye-mask exports, then
delete in one pass.**

Candidates (~4,500+ LOC): `export_eye_mask_training_zarr.py` (2,673), `prepare_eye_mask_training_from_registry.py`
(1,159), `aggregate_eye_mask_training_data_card.py` (1,918), `plot_eye_mask_training_data_card.py`
(661), `review_eye_masks_batch.py` (635), `materialize_refined_eye_masks_compat.py`,
`resolve_eye_mask_stale.py` (218, self-described deprecated shim), `run_eye_mask_training_pipeline.py`,
`validate_eye_mask_training_zarr.py`, `finalize_eye_mask_profile_artifacts.py`,
`inspect_eye_mask_source_areas.py`. (Note the eye-mask *inference/refinement* deletion is the
separate, larger migration tracked in the whole-repo review's P2-11 — that needs the
`infer_unet_subject_masks` rewire first.)

---

## The "one batch-runner framework?" question

Partially yes — but as **three small drivers, not one mega-framework**, and only after Tier 0:
1. **Stage-inference batch driver** (detect/keypoints/sam/swim + data-processing batch group):
   `discover → build_plans(ok/skip/missing) → dry-run print/JSON → [resolve models] → per-item run
   → record → summary`. Hooks: `discover_fn`, `plan_classifier`, `skip_check`, `run_item`,
   `record_provenance`.
2. **Review batch driver** (Tier 1 #1) — dispatches an interactive viewer, no inference.
3. **Single-recording resolve-and-run** (Tier 1 #5) — no plan/dry-run partitioning.

`cli/batch_runner.py` already exists but is **dead/legacy** (shells out to `python -m fisheye`,
used by nothing) — it is not the framework; retire it.

---

## Things to explicitly NOT merge (already correctly factored)

- `migrate_refined_detect_sparse{,_batch}.py` — batch already imports the single-run plan/apply.
- `build_flat_roi_cache` / `crop_batch` vs their `*_flat_roi_cache_*` / `_clipped_*` supersets —
  logic already in `shared/flat_roi_cache.py`; the second is a superset workflow, not a dup.
- `materialize_refined_subject_mask_store.py` — already thin, delegates to `shared/mask_store.py`.
- `plan_orange_style_clips` / `materialize_orange_style_clips` — correct plan/materialize split.
- The big multi-mode dashboards (`check_training_registry.py` 4.8k, `check_recording_steps.py` 3.5k,
  `audit_zarr_pixel_contracts.py` 1.7k, `registry_query.py` 2.9k) — should *consume* the shared
  discovery/report helpers but not be folded into generic tools. `registry_query.py` is a separate
  god-module concern: it is unreferenced by the import graph; its reusable query primitives
  (`_query_*_quality_map`, `_metric_summary`, `_percentile`, group-summary builders) should be
  extracted to `registry/` so the small `list_`/`aggregate_` tools call them — a larger structural
  project for later.

---

## Recommended execution order

1. **Tier 0 micro-helpers** (one PR per helper or a small batch). ~2–3k LOC, near-zero risk,
   builds the shared homes everything else depends on. Start with `_iter_zarr` → `zarr_discovery`
   (highest reach) and the json/utc/coercion sweep.
2. **Tier 1 quick wins** (#7 validate, #6 set-status, #8 list-unapproved) — small, low-risk, also
   fix divergence bugs.
3. **Tier 1 #1 review driver** — biggest single merge; do once Tier 0 helpers exist.
4. **Tier 2 `training_export.py` extraction** — unblocks Tier 3.
5. **Tier 3 eye-mask deletions** — after #4 + pipeline confirmation.
6. **Remaining Tier 1/2 merges** (backfill profiles, sync trio, finalizers, model-resolution public-ify).
7. **The three batch drivers**, then the `registry_query.py` split (largest, last).

## Verification (per merge)

- `scripts/py -m pytest -m "not gpu and not slow" -q` green before/after each PR.
- For merged CLIs: keep old entry points as thin shims or argparse aliases for one release; diff
  the JSON output of old vs new on a sample zarr to prove behavior parity.
- For registry-write merges (backfill/sync/finalize/set-status): run with default dry-run and diff
  the planned actions against the pre-merge tool.

---

*Companion reports: `docs/diagnostics/codebase_review_2026-06-20.md` (whole-repo; this expands its
P2-7), `docs/diagnostics/code_review_goodcopbadcop_dashboard_2026-06-20.md` (sun-branch diff).*
