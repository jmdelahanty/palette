# Contract Enforcement, Divergence & Sources-of-Truth Review — 2026-08-21

**Date:** 2026-08-21 (compiled from a four-agent parallel review run 2026-08-20/21 at HEAD ~6e7caa07, branch sun)
**Lenses:** (1) sources-of-truth census, (2) contract enforcement classification, (3) registry write-path sprawl, (4) divergence detection/repair coverage.
**Live evidence feeding this review:** the sleepyfish registry lag (37/37 step rows stale, reconciled 2026-08-20), the duplicate bare-vs-`:zXXXX` step-status rows, the `--reconcile-dataset --dry-run` write incident (`docs/diagnostics/dry_run_audit_2026-08-20.md`), and the legacy-epoch RuntimeWarning during the cam2010095 reconcile.

**Plan disposition (2026-08-25):** audit evidence. Overlapping authority,
admission, resolver, and enforcement work is tracked only in
[`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md).
The waves below are not an independent active status source.

---

## Unified diagnosis

The sprawl is real, but it is not primarily *missing* checks. Nearly every contract has a validator. The failure is structural, in four reinforcing layers:

1. **Enforcement is non-blocking by construction.** `emit_stage_completion` wraps every hard gate in a blanket `except Exception` that degrades refusal to a console warning (`stage_complete.py:488-495`), and its `False` return is discarded at almost every call site. The zarr run is already published either way — the gates only ever withhold bookkeeping.
2. **Every enforced decision has a second, unenforced path that can overwrite it.** The reconcile/backfill machinery re-creates `ok` rows with no eligibility, provenance, or array validation (`maintenance.py:6886-6982`, `_resolve_latest_group` at `:4033-4070`). Whatever the gate refuses, reconcile writes anyway.
3. **Parallel policies where there should be one.** Six mutually incompatible definitions of "latest run"; two step-status write primitives with different history/validation/NULL semantics; two live semantics for `stage_selector_eligible` (writers: absent⇒eligible; readers: absent⇒ineligible); two staleness models (`ok`+details vs demote-to-`missing`); three ways to mint a `dataset_id`; five overlapping refresh layers for the quality tables.
4. **Nothing runs the reconcilers.** Zero scheduled automation exists — no cron, no CI `schedule:`, no pipeline hook. Every detector/repairer is a human-remembered CLI, and the highest-traffic one (`--reconcile-dataset --dry-run`) writes during its dry run.

The registry-lag incident is the system working exactly as built: analysis jobs set `PALETTE_DISABLE_REGISTRY_WRITES=1` (`execute_analysis_workflow.py:231`), the emitters silently no-op, the compensating `registry_finalize` is wired into only two bsub scripts, and no sweep exists to notice. Same story for the split-brain: 596 of 852 `zarr.open_group` sites read the stale consolidated snapshot, no tool detects the disagreement, and the flag-day fix (W2.1) has been open since 2026-08-09.

---

## Top cross-lens findings (ranked by blast radius)

### F1. The two-edit neutralization of the whole gate system
- `stage_complete.py:488-495` swallows every enforcement raise → gates never block.
- `maintenance.py:6886` re-writes whatever was refused, with no validation.
Two edits flip roughly half the enforcement table from SHADOW back to HARD. Everything else in this review is downstream of these.

### F2. Six definitions of "latest", one of them correct
Strict `resolve_latest_complete_run_name` (`zarr_run_completion.py:509-516`) vs registry `_resolve_latest_group` reverse-lexical fallback (`maintenance.py:4033-4070`) vs bare `attrs['latest']` in the stimulus/protocol/arena extractors (`db.py:661,804`; `extractors/stimulus_metadata.py:152`) vs `inline_refresh._select_latest_profile_run` lexical-max (`inline_refresh.py:28-36`) vs timestamp-ranked SQL views (`migration_bodies.py:4360-4364`; `db.py:1645-1700`) vs `manual_review_latest` (`extractors/quality.py:276`). Only the first checks eligibility or the `latest==latest_complete` pairing. Consequence: registry `run_name`, `is_latest`, and `*_latest` views can all name runs that scientific readers refuse — including runs mid-publication (the two-attr activation window at `zarr_run_completion.py:243-244`).

### F3. `stage_selector_eligible` has two incompatible semantics
Contract + writer helper: absent ⇒ eligible (`zarr_run_completion.py:361-368`). Every strict reader (reporting/discovery, chaser_metrics_loader, crimson read contracts): absent ⇒ ineligible. ~40 of ~100 `mark_run_complete` callers never touch the attr (all of `analysis/`, tracking, background, segmentation); `subject_shape_runs.py:1107` writes `False` at creation and **never writes `True`** — its complete runs silently vanish from strict readers (observed on sleepyfish as empty step status). The two canonical False→True activators are `eye_angle_analysis.py:4893→6658` and `materializers/bout_kinematics.py:609,868`.

### F4. Derived-analysis publications are registry-invisible by design, with no net
`PALETTE_DISABLE_REGISTRY_WRITES=1` set unconditionally by `cluster/keypoints/common.py:992,1108` and `execute_analysis_workflow.py:231`; `registry_finalize --apply` is the only repair and is wired into exactly two bsub scripts. Any derived analysis launched any other way publishes a sealed, valid zarr run with zero registry notification, indefinitely. Core-pipeline stages have independent finalizers and don't exhibit the lag; only `analysis/*` families do — matching the observed month-long sleepyfish lag exactly.

### F5. Step-status ledger: two primitives, crashing vocabulary, duplicate-key steady state
- P1 `status_ledger.upsert_recording_step_status:105` (validates enum, COALESCEs recording_id, always appends history) vs P2 `maintenance._apply_recording_step_status_rows:6787` (no validation, can NULL recording_id, signature-diffed history).
- `_tracking_source_freshness` (`maintenance.py:5239`) returns status `"stale"`, which is not in the enum or either CHECK constraint → `IntegrityError` that aborts the whole backfill batch at `maintenance.py:6798`.
- Bare vs `:zXXXX` dataset ids each get an independent 30+-row ledger (`resolve_effective_dataset_id`, `db.py:6521`; public unresolved `upsert_dataset`, `db.py:2493`); the bare row's ledger freezes forever (backfill skips `status='missing'` rows) → the observed simultaneous `ok`/`missing` pairs. `recording_step_overview` already counts `DISTINCT dataset_id` per recording — a duplicate detector nobody acts on.
- Cascade demotes to `missing` (destroying `na`/`absent` classifications) while backfill expresses the same staleness as `ok`+`details.source_freshness_state` — last writer determines which model you see.

### F6. Consolidated-metadata split-brain: still zero coverage
988/1404 run parents disagree with their snapshot; `palette_completion_epoch` invisible on 884 groups; ~596 open sites read the snapshot. No detector exists; reconsolidation has 2 production callers; `backfill_completion_epoch.py:813` still writes where consolidated readers can't see and doesn't reconsolidate (W2.1(a) open). The whole `analysis/` namespace additionally publishes into **unstamped parents** (~96 of 129 `require_runs_parent` sites omit `completion_epoch`), so the epoch-2 provenance gate has never fired for any analysis family.

### F7. Staleness/provenance is declared, never resolved
`source_refs` emitted by 24 families, presence-validated by 6, expected-vs-actual compared by **0**, checked at consume time by **0**. `audit_analysis_staleness.py` is read-only with one non-test consumer. The sleepyfish bout_kinematics-stale-vs-exponential-swim-bouts finding came from a human, not a tool — and `palette plan`'s `_find_stale` would have flagged it per-recording if anyone ran it.

### F8. Identity/metadata COALESCE-freeze
`recordings` merges N zarrs per recording with last-non-NULL-wins and no conflict record (`db.py:2645-2678`); `provenance` context fields freeze at first write (`db.py:2950-2958`) and silently back `dataset_context_current` — i.e., cohort selection. `provenance` has exactly one writer, called only from `register_from_root`, so stores first seen by a pipeline stage have no provenance row at all. Three subject-metadata CLIs share a publish→register→verify tail with three different identity assertions (one absent); `--backfill-subjects` can resurrect placeholder identities that `migrate_count_only_subject_context` deliberately deleted.

### F9. Geometry: modern chain forbidden from updating what production reads
The immutable `arena_geometry_*` chain never writes `analysis_metadata.attrs['dish_mask']` (by contract, `arena_geometry_selection.py:408`), yet detect/crop/refine/segment read only that legacy attr. Nothing verifies the two circles agree. The `dish_mask` step row has two colliding writers (tuner sync with provenance vs reconcile's presence-check, which erases it). `pixel_to_mm` has six candidate locations resolved first-non-null in two places, against a declared single-authority policy — and the registry's permissive answer is what gates cohort inclusion.

### F10. The repair tools themselves
`--reconcile-dataset --dry-run` writes (confirmed); `Registry(path)` is never read-only (creates files, applies DDL from any "plan" path); registry has no instance-identity (W3.1 open) while 158 `/nvme1` literals persist and the backup script's documented default source is the stale copy.

---

## Fix queue

Ordering principle: make refusal real and durable first (A), then collapse duplicate policies (B), then add the missing automation (C), then burn down legacy surface (D). A and B items are small, high-leverage edits; C is operational; D is gradual.

### Wave A — make enforcement actually block (small edits, do first)
- **A1.** Narrow `stage_complete.py:488` so contract violations (completion/eligibility/provenance/array-spec raises) propagate or produce a durable `error` step row; keep the blanket catch only for registry-unavailable I/O. *Accept: a run failing validation cannot end with a silent `False`; test asserts an `error` row or raise.*
- **A2.** Route `reconcile_recording_step_status_for_dataset` / `_build_recording_step_rows_from_root` through the strict resolver (eligibility + `latest==latest_complete`), replacing `_resolve_latest_group`'s reverse-lexical fallback. *Accept: reconcile can no longer mark `ok` any run the strict resolver refuses; sleepyfish cam095 subject_shape reconciles to a non-ok state with a reason, not `ok`.*
- **A3.** Fix the `"stale"` vocabulary crash: either add `stale` to the enum + both CHECK constraints + views, or map to `ok`+details like every other family (`maintenance.py:5239,5850,6539`). *Accept: a store with stale tracks completes a full backfill batch.*
- **A4.** Thread `dry_run` through the reconcile-dataset path and add the hash-sandwich test (already specified in `docs/diagnostics/dry_run_audit_2026-08-20.md` fixes 1–2, including `Registry.open_read_only()`). *Accept: dry-run leaves the registry byte-identical.*
- **A5.** Pick one eligibility semantic — the readers' strict side (absent ⇒ ineligible) — and update `is_run_selector_eligible_attrs` + the contract doc; fix `subject_shape_runs.py` to activate `True` on successful completion; backfill eligibility on the existing complete analysis runs (sleepyfish first). *Accept: strict readers and the helper agree; subject_shape runs become visible to strict readers.*

### Wave B — one policy per decision (consolidation)
- **B1.** Merge P1/P2 into a single step-status primitive: enum validation + COALESCE recording_id + signature-diffed history, used by both runtime and reconcile paths (`status_ledger.py:105` absorbs `maintenance.py:6787`). *Accept: one INSERT site for each of the two tables.*
- **B2.** Make `dataset_id` minting a private invariant of `upsert_dataset` (fold in `resolve_effective_dataset_id`); delete the five duplicated mint-then-upsert prologues and `refine_keypoints._resolve_status_dataset_id`. Then run `dedupe --apply` once to collapse existing bare/suffixed pairs. *Accept: no code path can create a bare-id row for a `/recordings/` path; sleepyfish duplicate rows gone.*
- **B3.** One "latest" for registry projection: extractors (`stimulus_metadata.py:152`, `db.py:661,804`), `inline_refresh._select_latest_profile_run`, and the `*_latest` SQL views either call/replicate the strict resolver or carry an explicit `selector_eligible` column so consumers can filter. *Accept: no registry surface reports a run the strict resolver refuses without labeling it.*
- **B4.** Retire `inline_refresh.sync_latest_detection_profile_for_zarr` (self-declared superseded) and the `refresh_*_from_root` family (redundant after B2); make `scan` either call the full reconcile or print a loud "profiles NOT refreshed — run --reconcile-dataset" notice. *Accept: `scan` cannot silently leave profile tables frozen.*
- **B5.** Reconcile `RECORDING_STATUS_STAGE_IDS` with the emitter set (chaser_distance backfillable; `track_kinematics_visualization`/`keypoint_quality` either get writers or leave the cascade graph; `keypoints_review` becomes a canonical id or stops being written). Derive one list from the other. *Accept: no permanent-`missing`-with-no-path-to-ok stages; no non-catalog step names in the ledger.*
- **B6.** Merge the three subject-metadata apply tails into one helper with the strictest identity assertion; give `_backfill_subjects`/`_backfill_subject_dish_cross_entities` the placeholder-scope guard from `upsert_subject_snapshot_entities`. *Accept: backfill cannot resurrect deleted placeholder identities.*

### Wave C — close the automation gap
- **C1.** Wire `registry_finalize` into `execute_analysis_workflow` itself (or write a `receipt_pending` step row at submit time so un-finalized analyses are queryable). *Accept: a derived-analysis publication is either registered or visibly pending — never invisible.*
- **C2.** Schedule the sweep: a weekly `reconcile_sweep --dry-run` + step-status compare over `/groups/.../recordings`, reporting divergence counts (registry vs zarr) somewhere visible (status page or a dated report in the registries `audits/` drop-box). Depends on A4 so the sweep is safe. *Accept: the 2026-08-20 lag class is detected within a week without a human remembering.*
- **C3.** Split-brain: land W2.1 (writers reconsolidate or a lint bans bare `zarr.open_group` in `src/`; backfill reconsolidates; then flip `legacy_default=False`). Add a one-shot census tool that diffs snapshot vs direct metadata so progress is measurable. *Accept: `use_consolidated` idiom can no longer change what an attr read returns.*
- **C4.** Stamp `palette_completion_epoch` on `analysis/` parents (writer default in the materializers + one backfill pass), eliminating the legacy-open namespace and the reconcile RuntimeWarning. *Accept: `check_zarr_run_completion --fail-on-unsafe` clean on the sleepyfish stores with strict semantics.*
- **C5.** Staleness as a check, not a doc: run `palette plan`-style `_find_stale` (or `audit_analysis_staleness`) inside the C2 sweep and surface expected-vs-actual `source_refs` mismatches. *Accept: the bout_kinematics-vs-exponential-swim-bouts class is machine-reported.*

### Wave D — legacy surface burn-down (opportunistic)
- **D1.** Geometry unification: either the selection chain projects to the legacy `dish_mask` attr under a guarded writer, or production readers migrate to the selection run; single writer for the `dish_mask` step row (tuner sync wins, reconcile preserves method).
- **D2.** Calibration: replace the two first-non-null `pixel_to_mm` ladders with `selected_calibration`; registry `calibration=ok` requires the single authority.
- **D3.** Registry instance identity (W3.1): `registry_uuid` + refuse-on-mismatch, killing the `/nvme1` stale-copy class; fix the backup script's default source; sweep the 158 `/nvme1` literals behind one config point.
- **D4.** Delete the retired eye-mask `replace_*` primitives; adopt `_backfill_dataset_lineage` as the reference backfill shape.

---

## Appendix: full agent reports

The four underlying reports (with complete tables, every file:line, and the SAFE inventories) are in the session transcript of 2026-08-20/21. Key prior docs this review extends: `docs/diagnostics/design_review_findings_2026-08-09.md` (Wave-status diff: W0 done, W2.1–W2.4 open, W3.1/W3.2/W3.5 open, W3.3 partial), `docs/diagnostics/validation_receipt_audit_2026-08-17.md` (receipts: still single-adopter), `docs/diagnostics/provenance_chain_audit_2026-07-24.md` (split-brain census), `docs/diagnostics/dry_run_audit_2026-08-20.md` (repair-tool safety), `docs/error_budget_policy_2026-08-11.md` (still specified-only; none of its indicators run on any cadence).

One correction to the record: the import-linter CI outage reported in the 2026-08 import pass is **fixed** — `import-boundaries` is a live, required CI job with exhaustive layer contracts (`.github/workflows/ci.yml:33-36`, `pyproject.toml:88-133`); only the 7 named RATCHET exemptions remain.
