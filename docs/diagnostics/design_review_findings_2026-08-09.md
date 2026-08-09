# Design review findings and fix queue — 2026-08-09

<!-- contract-meta
version: 1
status: active
last_verified: 2026-08-09
implementation: specified-only
-->

Source: five parallel read-only design reviews (architecture, provenance, auditability,
scientific rigor, data contracts) at `386e4278`, plus an empirical import-enforcement
pass on 2026-08-09. This document is the implementation queue. Findings marked
**[VERIFIED]** were confirmed by executing tools this session (lint-imports, grimp,
gh CLI, live-registry reads); everything else is from code reading by review agents —
**implementing agents must re-verify a finding against current code before changing it.**

Known-good patterns to copy are named per item. Do not invent new mechanisms where a
proven one exists in-repo.

---

## Wave 0 — Resurrect CI (do first; nothing else is verifiable until this lands)

**W0.1 — Un-break the layers contract. [VERIFIED]**
`pyproject.toml:96` lists `capture` un-parenthesized; commit `35a872bf` (2026-07-25)
deleted the last files in `src/fisheye/capture/`, so `lint-imports` has errored
("Missing layer ... fisheye.capture does not exist") on every run since. Because the
"Lint import boundaries" step fails, CI **skips** the file-size ratchet and
`pytest --collect-only` steps. All 50 most recent CI runs on all branches are
failure/cancelled.
Fix: remove `capture : ` from the layers string. Also delete the local stray
`src/fisheye/capture/` dir (holds only `__pycache__`).
Acceptance: `lint-imports --config pyproject.toml` evaluates contracts (pass or fail,
but no "missing layer" abort).

**W0.2 — Triage the 8 shared-layer violations that accumulated while the linter was dead. [VERIFIED]**
Confirmed by running lint-imports with a patched config:
- `fisheye.shared.derived_analysis_registry_status -> fisheye.registry.stage_complete` (l.10)
- `fisheye.shared.derived_analysis_registry_status -> fisheye.registry.stage_catalog` (l.11)
- `fisheye.shared.detection_candidate -> fisheye.detection.detect_yolo` (l.41)
- `fisheye.shared.zarr.native_canonical_detection_publication -> fisheye.detection.native_canonical_candidate` (l.26)
- `fisheye.shared.zarr.training_crop_materialization_publication -> fisheye.utils.regenerate_training_crops_pynvvc` (l.47)
- `fisheye.shared.zarr.training_base_publication -> fisheye.utils.import_sampled_training_pynvvc` (l.29)
- `fisheye.shared.zarr.clipped_binding_builder -> fisheye.utils.plan_clipped_detect_refine_workflow` (l.22)
- `fisheye.shared.zarr.sampled_training_crop_materialization -> fisheye.training.detection_frame_supervision` (l.64)
Fix: per import, either move the imported code down into `shared` or add a commented
`RATCHET:` ignore entry (existing entries at `pyproject.toml:99-114` show the format).
Prefer moving code down where the import target is small.
Acceptance: `lint-imports` green; every new ratchet has a why-comment.

**W0.3 — Make the three quality gates independent.**
In `.github/workflows/ci.yml` quality job, lint failure currently skips the file-size
ratchet and collect-only steps. Either split into separate jobs or add
`if: always()` semantics so each gate reports independently.
Acceptance: forcing a lint failure on a test branch still runs ratchet + collect steps.

**W0.4 — Diagnose the 6 failing non-gpu test shards.**
Shards 0, 6, 7, 8, 9, 10 fail (run 31288102204). Root cause NOT diagnosed this
session — do not assume it is related to W0.1. Investigate, fix or quarantine with
linked issue.
Acceptance: full CI green on the working branch.

**W0.5 — Process gate: no agent merges on red CI.**
Two weeks of red CI normalized landing unverified work (8 boundary violations
accumulated in exactly that window). Encode in `AGENTS.md`: agent branches must be
green before handoff/merge.

---

## Wave 1 — Science correctness (highest wrong-science risk; touches figures)

**W1.1 — Kill the raw-centroid `immobile_fraction` shipping to dashboards.**
`src/fisheye/analysis/chaser_near_field_occupancy.py:695-725`
(`_speed_state_for_phase`) frame-differences raw centroids at the 1 mm/s threshold —
the exact artifact class that produced the retracted "learned avoidance" result
(see `analysis/analyze_goodcopbadcop_immobility_artifact.py:1-22`). Output is written
as `thigmotaxis/immobile_fraction`, `mean_speed_mm_s`, `median_speed_mm_s`
(`:1996-2000`) and consumed by `visualization/goodcopbadcop_interactive.py:1665-1685`
and `apps/marimo/components/goodcopbadcop_chaser.py:3440-3444`. It also diffs across
tracking gaps (no `transition_valid` check → teleport-magnitude speeds).
Copy the proven pattern: `chaser_response_regimes.py:551-614`
(`_load_smoothed_immobility_speed`, requires `required_speed_levels=("smoothed",)`,
`sample_valid & transition_valid`) and its `immobility_signal_mode` contract
(`:634-650`, raw reachable only via `"raw_centroid_explicit"`). Port the two guard
tests from `tests/unit/fisheye/test_chaser_response_regimes.py:215,243`.
Emit a `speed_source` field alongside every published speed-derived array.
Acceptance: no code path computes a published speed metric from raw centroid diffs
without explicit opt-in; `speed_source` present in the zarr group and Arrow export;
tests ported.

**W1.2 — Make arena-geometry fallback impossible to ignore; protect the virtual-control nulls.**
`chaser_radial_occupancy.py:232-249` defines a `_resolve_arena_geometry` that discards
the QC notes. Three modules import it and use `geometry.center_*` as the **virtual
reference control rotation center**: `chaser_escape_events.py:105,420`,
`chaser_bout_response.py:79,966`, `chaser_gaze_tracking.py:58,935`. Also discarding
notes: `chaser_habituation_figures.py:199`, `chaser_analysis_figures.py:74`.
Fix: delete the notes-discarding helper; repoint all consumers at
`resolve_arena_geometry_with_notes` (canonical: `shared/arena_geometry.py`); fold
notes into each component's `qc_warnings` (pattern: `chaser_radial_occupancy.py:1058`).
Modules using geometry as a virtual-control rotation center must **hard-fail** on
nominal-circle fallback, not warn.
Also fix `chaser_epoch_behavior_summary.py:360`: `pixels_per_mm_projector ... or 1.0`
— a missing/zero attr silently forces dish-mask rejection (residual_px misread as
residual_mm) → silent fallback to the known-3mm-off nominal circle. Must raise.
Acceptance: grep shows zero imports of the deleted helper; a synthetic recording with
a forced nominal fallback raises in escape/bout/gaze components and surfaces a QC
warning everywhere else; `or 1.0` gone.

**W1.3 — Audit whether any escape-cohort recording actually fell back to nominal geometry.**
Before/alongside W1.2: run the exporter-style geometry resolution
(`utils/export_cross_recording_analytics.py:2400-2461` already computes
`arena_geometry_status`/`arena_geometry_source`) across the GoodCopBadCop cohort and
report which recordings resolved `circle` vs `dish_mask`. This determines whether the
published escape/pursuit numbers need recomputation.
Acceptance: a table (recording_id → geometry source) in a diagnostics doc.

**W1.4 — Real FDR families + session clustering in group_statistics.**
`group_statistics/goodcopbadcop.py:1707-1713` defines
`multiple_comparison_family = metric_family|metric_name|contrast` → 24 of 37 specs sit
in size-1 families → `q_value == p_value` by construction across a ~111-test battery.
The `primary`/`exploratory` flags are stored (`:1731-1732`) and never read by
`apply_fdr` (`:1862-1872`).
Fix: family = `{primary|exploratory}|{metric_family}` (or per pre-registered unit);
update the manifest validator (`:1147-1170`) to match intent, not implementation.
Add `session_id` (and `subject_id`) to `analytics_exports/contracts.py` and
`baseline_strategy/contracts.py:45-49` identity columns, sourced from the registry —
not regex-parsed (current hack: `analyze_goodcopbadcop_learning_mixed_model.py:47`).
Promote the session random-intercept + ICC fit from that script into
`group_statistics/` as `cluster="session"` mode with a `cluster_status` output field.
Acceptance: no size-1 families for exploratory metrics; `session_id` a real exported
column; clustered and naive estimates both reported.

**W1.5 — Denominator and default hygiene in epoch summaries.**
- `chaser_epoch_behavior_summary.py:295-297`: silent `"filtered"` speed-level default
  → require explicit level or fail.
- `chaser_epoch_behavior_summary.py:663`: `bout_rate` uses wall-clock window duration;
  dropout is behavior-correlated (freezing fish are hardest to track). Use
  valid-tracked time (pattern: `chaser_escape_events.py:489,790` +
  `test_rate_is_normalized_by_valid_tracked_time_not_wall_clock`).
- Two incompatible wall/thigmotaxis denominators
  (`chaser_near_field_occupancy.py:688-690` vs `chaser_epoch_behavior_summary.py:630-638`):
  declare denominators as data (pattern: `baseline_strategy/contracts.py:81-82`
  `*_denominator` columns).
- Add the raw-speed noise-floor pitfall to `docs/analytics_math_primer.md` pitfalls
  table (~:464-477).
Acceptance: no silent speed-level default; bout_rate denominator is valid-tracked
time; exported wall metrics carry denominator declarations.

---

## Wave 2 — Enforcement gates (fail-open → fail-closed)

**W2.1 — Close the consolidated-metadata split-brain.**
Three parts:
(a) `utils/backfill_completion_epoch.py` opens with `use_consolidated=False` (:813)
and never reconsolidates — stamping the epoch is what CREATES the stale cache that
hides it. Call `reconsolidate_zarr_metadata()` at the end.
(b) Add a lint/AST check (CI pattern: `scripts/check_file_size_ratchet.py`) failing any
`zarr.open_group(` in `src/fisheye/` without explicit `use_consolidated`. ~404 bare
call sites exist, including the `analysis/analyze_goodcopbadcop_*` family. Route
mutable-state reads through `shared/zarr_helpers.py:409` (`open_zarr_group_direct`).
Expect a large mechanical sweep; allowlist-then-ratchet is acceptable.
(c) Invert the legacy default: `zarr_run_completion.py:56-60`
(`effective_legacy_default` → True on missing epoch = absence reads as completeness).
Add a store-level `palette_store_epoch` root attr: post-cutover stores treat a missing
parent epoch as an ERROR. Replace the per-process warning dedup (`:40,303-314`) with a
process-exit counter ("accepted N legacy-default runs").
Acceptance: backfill reconsolidates; CI fails on new bare opens; a store stamped
post-cutover raises on missing parent epochs.

**W2.2 — Promote git_dirty + environment identity into the finalization gate.**
`shared/run_provenance.py`: gate value-requires only `git_sha` + `config_hash`;
`git_dirty` is recorded but permitted. Add `COMPLETION_EPOCH_REQUIRE_CLEAN_CODE = 3`
(mechanism precedent: epochs 1/2 in `zarr_run_completion.py:31-33`), requiring
`git_dirty is False` (reuse the explicit-reason bypass machinery) and a new
`environment_digest` = sha256 of the sorted full package list from
`get_software_versions()` (`shared/system_metadata.py` — currently computed then
discarded in favor of the ~40-pattern `key_packages` allowlist at `:653-690`).
Roll out shadow-first per stage, as stage-array validation did.
Acceptance: epoch-3 stores refuse dirty-tree finalization without a written bypass;
every new run_provenance payload carries environment_digest.

**W2.3 — Fix the color-range hardcode before the transcode campaign.**
`utils/regenerate_training_crops_pynvvc.py:982` hardcodes
`"container_color_range_observed": "tv"`, bypassing
`normalize_observed_container_color_range()` (landed `844b6254`, docstring: "must
never be guessed"). This tool is the engine of
`utils/batch_migrate_training_crop_pixel_contract.py` — the wrong value propagates at
scale, and post-cutover (2026-07-02) recordings get a factually wrong `pc`→`tv` stamp.
Correct pattern already in: `utils/append_acquisition_crop_video_training.py:280`,
`utils/export_acquisition_crop_pose_training_zarr.py:791`,
`utils/import_sampled_training_pynvvc.py:259`.
Also: registry migration 66 adding `color_range` (+ `color_space`, `color_transfer`,
`color_primaries`) to `acquisition_video_streams`, backfilled via the ffprobe fields
`shared/import_video_metadata.py:92` already requests.
Also: sweep for other hardcoded literals assigned to `*_observed` / `*_measured` /
`*_detected` fields.
Acceptance: no literal color-range in any writer; the "which recordings are
pre-cutover mislabeled" question is one SQL query.

**W2.4 — One queryable provenance index.**
Add registry table `run_provenance_index(dataset_id, stage, run_name, git_sha,
git_dirty, config_hash, environment_digest, input_artifact_sha256s_json,
bypass_reason, finalized_utc)` populated from `registry/stage_complete.py`
(`emit_stage_completion` already resolves the run group at the right moment).
Add real `git_sha`/`git_dirty` columns to `training_runs`/`training_models` (currently
buried in `invocation_json`, unqueryable). Wire the existing-but-uncalled
`registry/model_resolution.py:143` (`verify_deployment_artifact_content`) into
`palette status`; surface bypass counts.
Acceptance: "which runs finalized from dirty trees" and "which deployed weights drift
from their registered hash" are single queries/status lines.

---

## Wave 3 — Audit integrity

**W3.1 — Registry instance identity.**
No `registry_id`/UUID/`PRAGMA application_id` exists; a frozen cohort manifest records
path + schema_version only (`cohorts/registry.py:498-503`) — cannot distinguish the
canonical `/groups` registry from a stale copy (the class of bug that caused the
"12 fish" under-selection). Mint an immutable identity row on `_init_schema`; bind
`registry_uuid` into frozen manifests; make `legacy_bootstrap` (`db.py:1443-1452`)
mint `legacy_bootstrap_unverified` instead of self-certifying as fully migrated.
Fix the two docs still pointing at stale `/nvme1`:
`docs/registry_data_governance_policy.md:173`, `docs/registry_browser/README.md:36`.
Acceptance: manifests carry registry_uuid; docs point at the canonical path.

**W3.2 — Wire the existing doc/schema checkers into CI.**
Two additions next to the ratchet step:
`python scripts/generate_registry_schema_reference.py && git diff --exit-code -- docs/registry_schema_reference.md`
(reference is ~5 migrations stale: missing `recording_chasers`, `stimulus_protocols`,
`analytics_reports`, `recording_subject_traits`, `strain_trait_expectations`,
`strain_label_mappings`) and `python scripts/check_contract_freshness.py` (currently
fails 99/115 docs — fix the checker's status vocabulary and glob first, add an
`implementation: implemented|partial|specified-only` field; honest-form model:
`docs/mutable_review_runs_contract.md:15-24`).
Acceptance: both run in CI; freshness failures are triaged (fixed or explicitly
waived), not ignored.

**W3.3 — Packaging + import-graph truth. [VERIFIED]**
`fisheye/analysis` (119 files) and `fisheye/pose` have no `__init__.py`;
`find_packages` excludes them from any non-editable wheel; grimp cannot see them
(graph = 1,149 modules, both absent), and the `(analysis)`/`(pose)` parens in
`pyproject.toml:96` silently tolerate that.
Fix: add `__init__.py` to both; de-parenthesize; add the 8 missing packages to the
layers list (`analysis_workflows`, `analytics_exports`, `baseline_strategy`,
`cluster`, `cohorts`, `montage`, `reporting`, `training_response`); delete phantom
`(apps)`/`(docs)`/`(group_analytics)`; set `exhaustive = true` on the contract so a
future package retirement/addition fails loudly instead of crashing or escaping.
Add a wheel smoke to CI: `python -m build && pip install dist/*.whl &&
python -c "import fisheye.analysis, fisheye.pose"`.
Expect a violation triage wave when analysis enters the graph (known:
`shared/zarr/stage_arrays.py:694 -> analysis.eye_angle_schema`; 7×
`shared/zarr/* -> analysis_workflows.materializers.atomic_run_publisher` — resolve by
moving `atomic_run_publisher` down into `shared/zarr/`).
Acceptance: wheel installs and imports; grimp sees analysis/pose; contract exhaustive.

**W3.4 — Wire or delete the orphaned audit machinery.**
Zero production callers today: `shared/tabular_deltas.py` (per-row `editor` — wire
into the labeling apply boundary `labeling/web.py:2440-2760`, which already has
`apply_id`, `edit_revision_before/after`, touched rows, and the authenticated user;
start with subject_mask_component), `utils/audit_analysis_staleness.py` (compares a
fingerprint no stage writes), `verify_deployment_artifact_content` (→ W2.4).
Per the July audit's own words: deleting is better than leaving unwired.
Acceptance: each of the three is either on a production path or deleted.

**W3.5 — Review-state history + ledger revival.**
(a) The four `*_quality` review tables are upsert-overwrite — an approval erases a
prior rejection with no trace. Add history tables or an event log (pattern:
`recording_step_status_history`, the only `_history` table in 65 migrations).
(b) `recording_step_status` is frozen: max `updated_utc` = 2026-06-18 vs datasets
`last_seen_utc` = 2026-08-09. Root cause NOT diagnosed — suspect unapplied
deferred-finalizer receipts (`PALETTE_DISABLE_REGISTRY_WRITES` path). Investigate;
add a receipt-backlog count to `palette status`. This also unblocks the stage-array
validation graduation telemetry (18 rows total; 18/25 stages stuck in shadow mode).
(c) `prune_stale_datasets.py:46` + `ON DELETE CASCADE` (`migration_bodies.py:3155`)
delete step-status history with the dataset — violates the governance policy's own
"provenance survives cleanup" intent (`registry_data_governance_policy.md:147`).
Acceptance: review transitions are append-only; ledger receiving rows again; history
survives dataset pruning.

---

## Wave 4 — Structure & storage (important, not urgent; schedule after Waves 0-2)

**W4.1** Split `utils/` (353 modules, 204k lines, 334 argparse mains): `fisheye/stages/`
(the ~40 `run_*`/`export_*`/`finalize_*` the CLI and cluster actually invoke, promoted
to a declared dispatch), `fisheye/maintenance/` (69 `backfill_*`/`audit_*`/`check_*`
one-shots, marked retirable), helpers → `shared/`. Delete the function-local imports
at `cli/palette.py:1636,1727,1861`.

**W4.2** Execute the eye-mask deletion the policy already authorizes (measured: one
archive's `refined_eye_masks_runs` = 27,440 files for 216 MB; 105 registry rows assert
deprecated stages ok). Remove `EYE_MASKS_SPEC`/`REFINED_EYE_MASKS_SPEC` from
`stage_arrays.py` (which has no deprecation concept — `stage_catalog.py` does);
reconcile the two catalogs; drop eye-mask tables (migrations 15/17/25/26) in a new
migration.

**W4.3** Adopt sharding on immutable run families (17 shard sites vs 707 chunk sites;
measured 23× copy speedup from packing; `storage_contract_catalog.py`'s
`byte_planner_adopted: false` is the worklist). Leave actively-edited surfaces chunked.

**W4.4** Route `emit_stage_completion` validation through `ArrayContract`
(`shared/zarr/array_contracts.py`) instead of legacy `ArraySpec`
(`stage_arrays.py`). Start by binding the coordinate-critical ambiguous names —
`bbox_norm_coords` (5 signatures), `bbox_img_xyxy` (3), `bbox_xyxy` (3, incl. int32) —
per `docs/shared_zarr_schema_catalog_design.md`. Stamp `contract_id`/`contract_version`
in array attrs at write time.

**W4.5** Orphan cleanup: 44 loose `src/*.py` (zero inbound imports, shadowing-capable
via `scripts/py` PYTHONPATH), `src/chaser_analysis/` (self-bootstraps `sys.path`),
6 stray `src/fisheye/test_*.py`, `src/fisheye/core/pipeline.py` still targeted by
`__main__.py:8`. Extend the file-size ratchet from 4 files to top ~25
(`utils/audit_coordinate_contracts.py` 16,856 lines and `analysis/track_kinematics.py`
14,305 are unguarded).

---

## Not diagnosed this session (do not assume)

- Root cause of the 6 failing CI test shards (W0.4).
- Root cause of the step-status ledger freeze since 2026-06-18 (W3.5b).
- Whether any escape-cohort recording actually resolved nominal geometry (W1.3 is the
  audit that answers it).

## Ordering constraints

- Wave 0 strictly first: fixes landed on red CI are unverified fixes.
- W1.3 (geometry audit) before or with W1.2, so the science impact is known.
- W2.3 before any master-transcode campaign starts.
- W3.3 will surface new lint violations; land after W0.2 so the triage waves don't mix.
