# Handoff: architecture / provenance review wave, 2026-09-01 → 2026-09-04

> Historical handoff, preserved as review evidence on 2026-09-05. Its commit
> range and claims describe the original wave, not the current checkout. Some
> conclusions are disputed in [the second opinion](../docs/diagnostics/review_wave_second_opinion_2026-09-04.md).
> The instruction below not to re-derive findings does not constrain fresh
> verification. See [the landing handoff](../docs/diagnostics/review_docs_rules_landing_handoff_2026-09-05.md)
> for the imported source revision, current base, and remaining validation.

> Correction added 2026-09-06: the body below preserves the original handoff,
> including disputed claims; it is not an executable work order. Blanket
> `latest` backfill, registry-only authority, and universal receipt/digest
> replacement remain rejected by the second opinion. Correct historical facts:
> 32 selector mismatches = 28 RedScare + four GoodCopBadCop; four backups, not
> five; `/nvme1` references remain. Five recording/model overlaps across four
> cameras do not prove exact-example overlap, evaluation leakage, or that all
> other recordings are clean. Six runtime verifiers and dynamic reuse
> revalidation now exist at main `1bf9d919`; see the
> [September 6 reconciliation](../docs/diagnostics/review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation)
> for exact versions and the remaining boundary defects.

You are picking up after a multi-day read-only review of the `palette` repository (package `fisheye`)
plus a small set of measured, backed-up changes to the live registry and storage. Everything below is
committed on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`, commits `ea03eb76..254ff700`
(15 commits, all docs and two scripts; **no changes under `src/`**). Nothing has been pushed. Start by
reading the documents in §2 in the order given. Do not re-derive what they already establish; verify
line citations against current code before acting on any of them.

## 1. What the reviews concluded

One diagnosis, reached independently by every lens: **the primitives of an auditable pipeline exist
and are optional at the point of use.** Completion epochs, selector activation, atomic publish,
digest-bound manifests, a real three-root Merkle receipt, commit-pinned worktrees, and an exhaustive
import-linter are all present. Each has bypasses on the production path.

Load-bearing defects, in the order they should be fixed:

1. **The completion gate is fail-open.** `registry/stage_complete.py:488` catches every exception and
   returns False after the run is already `latest`; `zarr_run_completion_contract.md` calls it
   fail-closed. Same pattern in `training/training_run_shared.py:121`. No promotion path
   (`mark_run_complete`, `stage_complete`, direct registry-ok writers, reconcile) recomputes any digest.
2. **Nine independent "latest" resolvers**, and the canonical chain breaks at four of them: refine-detect
   (`refine_detect.py:1485`, strict flag defaults off at `:1355`); the legacy keypoint→tracking lineage
   that is the *only* writer of `track_kinematics` (sealed tracking manifest and its verifier exist and
   are never called); subject-mask refinement input and chaser bout/escape (`chaser_bout_response.py:1249`);
   and all four training exporters plus every non-chaser cross-recording table.
3. **Envelopes are incomplete.** ~35 receipt/manifest builders; ~80 % share a header by accident; none
   binds all four parts (subjects, producer, inputs, review). Content manifests carry subjects without
   code identity; provenance sidecars carry code identity without subjects. Review acceptance is never
   digest-bound anywhere (`cli/palette.py:1521` records no digest of what was approved).
4. **Four canonical-JSON grammars and two array-digest grammars** coexist; `run_provenance.stable_json`
   admits NaN and `ensure_ascii=True`; `config_hash` includes paths and is never recomputed; source video
   is fingerprinted by size+mtime only; H1 hardlink-as-identity hole still open.
5. **Training provenance**: `registry/model_resolution.py` scores by feature similarity and never excludes
   sets containing the target recording; no base-weights sha, no `parent_run_id`, YOLO splits not
   persisted; training exporters enforce review as a precondition then discard it.
6. **Data-model conflict**: mutable-in-place store with pointers inside the data (`latest`, selectors,
   authority envelopes live in zarr attrs and are mirrored to the registry). Proposed rule: run group
   immutable at `mark_run_complete`; registry is the only authoritative pointer.
7. **Correction loop is not closed**: corrections do not flow into training with a receipt; no
   per-stage error rate is measured (error-budget policy has zero measurement code); no uncertainty
   routing to the labeling queue.

## 2. Documents (read in this order)

| Doc | What it is | Items |
|---|---|---|
| `docs/diagnostics/architecture_review_five_lens_2026-09-01.md` | five-lens review, verdict + per-component checklists | 46 (O/P/S/C/G) |
| `agents_todo/brief_architecture_review_wave_1.md` | **the implementation brief**: Package A (strict gate flag, split registry-I/O vs validation exceptions), B (LSF runtime commit assertion), C (widen ratchets) | 3 packages |
| `docs/diagnostics/provenance_chain_review_2026-09-01.md` | digest primitives census; chain acquisition→figure; RFC 8785 explainer §6 | 25 (D/U/F) |
| `docs/diagnostics/training_data_and_model_provenance_review_2026-09-01.md` | dataset export + model training provenance | 23 (T/R/M/X) |
| `docs/diagnostics/receipt_builder_census_2026-09-02.md` | every builder/verifier, promotion-path table, proposed `shared/envelope.py` surface | 27 (E/G/R/B) |
| `docs/diagnostics/pipeline_model_and_correction_loop_2026-09-02.md` | Nextflow boundary, immutability rule, correction loop, PROV/RO-Crate | 21 (N/I/L/X) |
| `docs/diagnostics/canonical_propagation_trace_2026-09-03.md` | hop-by-hop trace detection→exports, ~45 hops, 4 breaks | 29 (D/K/M/X) |
| `docs/diagnostics/store_measurements_selectors_and_training_membership_2026-09-03.md` | **measured** on the live store (not code reading); S-checklist; records every state change made | 9 (S) |
| `docs/artifact_storage_map.md` §Storage Roots | the `/groups` roots and the `models/<task>/<set>/<run>/` layout | — |
| `docs/workflow_provenance_prior_art.md` (2026-07-24) | earlier prior-art doc; reaches the same conclusions; was unread until rediscovered | — |

Scripts landed: `scripts/sweep_run_selectors.py` (read-only selector/model sweep, ~95 s for the store,
prints a per-family disagreement table) and `scripts/check_training_membership.py` (train/eval
contamination join). Run with `scripts/py`.

**Queue discipline:** these docs do not open a new fix queue. New items adopt into
`docs/diagnostics/authority_consolidation_work_queue_2026-08-25.md` or the wave-1 brief's successor.

## 3. What was measured on the live store (facts, not code claims)

- Registry `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` is canonical;
  408 registered archives, 296 exist. The 112 missing were `/nvme1` archives the operator deleted
  deliberately (old data); rows are now `status='missing'`.
- **Train/eval contamination is exactly the four cameras of `sleepyfish_2026_05_05_17_45_30`**:
  keypoint runs on the production pose model (trained on them) plus one detect run; all are the
  registry-selected `ok` run and are analyzed through `swim_bouts`. Every other production recording is
  clean against its model. One pose model is in production everywhere.
- `latest` is unset on 191/248 crop, 124/225 refined-detect, 103/176 refined-mask, 85/205 keypoint,
  84/153 tracking families; sorted-last picks a different detect run than `latest` in 120/122 multi-run
  detect families; 32 RedScare refined-keypoint families have `latest` ≠ `latest_complete`.
  **Consequence: a `latest` backfill from registry rows (S-5) must precede any fallback deletion.**

## 4. State changes made (all backed up, all recorded in the measurements doc)

Five validated registry backups in `registries/backups/palette_registry_20260903_*.sqlite`.
- 112 `/nvme1` recording dataset rows → `status='missing'` (rows kept; lineage references them).
- Merged training zarrs (30.16 GB, 19 zarrs) copied `/nvme1/training/datasets` → `/groups/.../training/datasets`,
  verified by per-tree content digest (0 mismatches; digests at `training/datasets/_index/…json`);
  17 `derived_training_merge` rows + 13 `training_runs.manifest_path` + 13 `config_path` re-pointed.
- Eye-mask UNet model rows re-pointed to the identical file under `models/eye_masks/`.
- Deleted from registry: 2 failed pose runs, 2 set-less runs (and their orphan weight dirs on disk).
- Deleted `/groups/.../palette_models/` (redundant May snapshot; manifest kept in
  `operations/palette_models_snapshot_20260514/`); 3 docs' example commands re-pointed to `models/`.
- Result: 12 `training_runs` all `success` with `set_id`, 12 `training_models`, zero `/nvme1`
  references anywhere in the registry; `/groups` is the only recording and model store.

## 5. Recommended next work, in order

1. **Wave 1 from the brief** (Packages A, B, C in parallel; A first if serial). Fold census item
   G-1 (move `_validate_completion_run_group` outside the swallowing `try`) into Package A.
2. Independent of wave 1: D-1..D-4 digest-grammar unification; U-1 video content hash; F-1
   `save_figure` primitive; E-1 envelope module; M-1/M-2/M-3 training-row fields; S-3 training-membership
   registry view + flag the four sleepyfish recordings; S-4 decision on that session.
3. After Package A: S-5 `latest` backfill, then propagation-trace K-1 (swap `resolve_tracking_run` for
   `load_tracking_source_handle` in `load_tracking_ids`; single highest-value edit), then D-1, M-2, M-4,
   then the exporters (X-*).
4. Do not: author new contract docs before G-1 lands; add review tooling before L-1/L-3 (correction
   loop) exist; open a fourth queue.

## 6. Constraints in force

Use `scripts/py`; never `pip`/`conda install`; pytest runs outside the sandbox; never plain `git push`;
commit only when asked; eye-mask stages are legacy-compat, not delete-on-sight; registry writes require
a validated backup first (`scripts/py -m fisheye.utils.registry_backup`) and one transaction with
before/after counts; the user wants unflinching findings, not reassurance.
