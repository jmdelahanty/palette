# Brief: migration ledger — make Layer 3 answerable, then subtract it

**From:** commander session, 2026-08-04
**Status: READY after `brief_gate_restoration_2026-08-04.md` checkpoint 2.**
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Read first:** `docs/utils_reorganization_strategy.md` §"Layer 3 — the sediment" and
Phase 3 (this brief **executes and extends** that phase — where the strategy doc and this
brief disagree, the strategy doc wins and you report the disagreement),
`docs/legacy_archive_migration_policy.md`, `src/fisheye/registry/migrations.py` (the
pattern you are copying).

Ground rules: local `sun` is ground truth; fresh worktree from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python`.

---

## The gap this brief closes

The strategy doc prescribes `RETIRES_AFTER` markers on migrations. That is an **expiry**,
and it is necessary but not sufficient. The missing thing is an **applied-ledger**.

Verified 2026-08-04 across all 67 `backfill_*`/`migrate_*`/`repair_*`/`fix_*`/`clean_*`
modules in `src/fisheye/utils/` (28,309 LOC):

- `grep -lE 'backfill_version|_backfill_applied|migration_marker|palette_backfill|palette_migrations'`
  over `src/fisheye` returns **zero hits**. Not one writes a marker.
- **Zero** register a row in the registry. The SQLite side has 62 numbered migrations
  under `PRAGMA user_version`; the zarr side that holds the actual science has no version
  concept at all.
- 2 of 67 write any receipt; 8 support `--json-report` that is collected nowhere.
- 50 docs mention backfills, but every one describes *intent or policy*, never execution.

**So the question "has this backfill been applied to the whole store?" is unanswerable
today.** That is why Phase 3's "verify-then-delete the spent ones" has not moved: nobody
can verify. Build the ledger and the phase unblocks itself.

---

## Scope — four checkpoints, stop and report after each

### Checkpoint 1: quarantine the loaded guns (do this before anything else)

These are dangerous to keep, independent of any reorg. Do not wait on the ledger.

1. **`utils/repair_keypoint_offset_corruption.py`** — escalate it. The strategy doc lists
   it as merely "verify-then-delete." It is worse than that: it does not verify a known
   defect, it **searches** for one, iterating `n` in `1..max_n` over
   `roi' = roi − n·offset` and selecting whichever `n` puts the most rows in bounds
   (`:289-297`). It writes no repaired-marker. Running it twice shifts keypoint
   coordinates twice. Move it out of the importable package into `scripts/archive/` with
   a header stating the defect epoch it applies to and an explicit "re-running this on
   repaired data corrupts it" warning — or delete it (git history is the archive). Your
   call; justify it in the report.
2. **The 8 modules with no dry-run gate at all** — `repair_keypoint_review_status`,
   `repair_keypoint_training_refined_run_ties`, `repair_refined_subject_mask_frame_counts`
   (`--execute` instead of `--apply`), `backfill_refined_subject_mask_metrics`,
   `backfill_detect_review_authoritative_run`, `backfill_pose_onnx_registry_metadata`,
   `fix_stimulus_mode_mappings`, `migrate_training_label_runs_identity`. Recount yourself.
   Give each a dry-run default and an `--apply` gate matching the house pattern. Two of
   them rewrite **human review state** — those are the priority.
3. **`utils/migrate_refined_detect_sparse.py`** promotes its output run to `latest` by
   default. On an archive where a human has since curated `latest`, that demotes curated
   work. Make promotion opt-in (`--promote`), not default.
4. **The stale-store default.** **71 modules** under `src/fisheye/utils/` hardcode
   `/nvme1/recordings`, which is the known-stale copy (it caused the GoodCopBadCop
   under-selection bug). 11 of them define a byte-identical `_resolve_roots()`. Change the
   default to a **required explicit argument** — no migration should have a silent target.
   Do this before checkpoint 3 or the sweep audits the wrong store.
5. **STOP and report.**

### Checkpoint 2: the ledger and the base class

6. **Store-side ledger.** One attr on each zarr root:
   `palette_migrations: {"<name>": {"applied_at_utc", "commit", "counts", "scope"}}`.
   Follow the conventions in `shared/run_provenance.py` (canonical JSON, `stable_json`)
   and `shared/zarr_run_completion.py` (attr write discipline, `use_consolidated=False`).
   Reader + writer + a "has this been applied here?" predicate.
7. **One `Migration` ABC** in `shared/` (or wherever the strategy doc's target structure
   puts it — check before inventing a home): `plan(root) -> list[Change]` /
   `apply(changes)`, with traversal, root resolution, argparse, dry-run gating, the
   summary counter, and the ledger write all inherited. Each migration then becomes
   50–150 lines of actual logic.
8. Port **three** migrations onto it as proof — pick one attr-stamper, one metric
   re-deriver, one registry-column populator. Do not port all 67 in this checkpoint.
9. **STOP and report** with the before/after LOC on the three ported modules.

### Checkpoint 3: the sweep that converts unknowns into decisions

10. Run **every** remaining one-shot in dry-run against the **canonical** registry store
    (`/groups/...`, per `project_canonical_registry` — NOT `/nvme1`). Capture the pending
    counts into a single table committed as `docs/migration_applied_census_2026-08-04.md`.
11. That table is the deliverable. Zero-pending ⇒ deletable with evidence. Nonzero-pending
    ⇒ you have just found un-backfilled data; **report it, do not apply it** — applying is
    a commander decision.
12. **STOP and report.** This is the checkpoint the whole brief exists for.

### Checkpoint 4: subtract, gated on evidence

13. Delete the modules the census proves spent, module + paired test together, then
    `grep -rn "<basename>" src/ tests/ scripts/ tools/ apps/ configs/ docs/` and fix
    dangling doc references **in the same commit** (standing rule: no doc landmines).
14. Free deletions needing no census — **verified zero references anywhere including
    docs**: `migrate_video_keyframes_to_cams`, `clear_detection_flags`,
    `clean_refined_keypoint_runs`. Recount before deleting.
15. Survivors move to the strategy doc's `apps/migrations/` home **with both** a
    `RETIRES_AFTER` marker (strategy doc) **and** ledger participation (this brief).

---

## Explicitly OUT of scope — read this carefully

- **The H5 cluster.** `backfill_h5_metadata`, `fix_stimulus_mode_mappings`,
  `inspect_zarr_events`, `read_h5_data`, `check_h5_tracking_data`,
  `check_h5_subject_metadata` are **operator-gated, not dead** per the 2026-07-08 H5
  audit recorded in the strategy doc, even though several show zero import references.
  Deletion requires operator sign-off per file. My census flagged some of these as
  unreferenced; **the strategy doc's operator gate overrides the import graph.** Do not
  delete them. (`fix_stimulus_mode_mappings` still gets its dry-run gate in checkpoint 1 —
  gating is not deleting.)
- **Not one-shots despite the prefix.** `apply_tuning_by_camera` (12 refs) is a recurring
  config-application tool. `patch_keypoints_from_crops` / `patch_crops_from_refined` are
  the most-referenced modules in the cluster and may be recurring operations. Classify by
  *actually one-shot?* before touching them by prefix.
- Mass-deleting `utils/` on import-graph silence. The strategy doc's guardrail holds:
  **orphan-in-code ≠ dead**; ~44 code-orphans are live human CLIs. This brief only
  authorizes deleting what the **census** proves spent, plus the three verified-zero-ref
  modules above.
- Renaming `utils/`, creating `fisheye/apps/`, moving runners — strategy doc Phases 2/5/6/7.
- Applying any pending backfill discovered by the sweep.

## Constraints

- **Never run a migration with `--apply` in this brief.** Dry-run only, every time,
  including the three ported ones. Verify against a synthetic fixture store for
  correctness; verify against the canonical store read-only for the census.
- The ledger must be additive and idempotent: writing it twice for the same migration
  must not duplicate or lose prior entries. Test this explicitly.
- Ledger writes go through the existing atomic/attr discipline. Do **not** invent a fourth
  publish mechanism — the repo already has three incompatible ones.
- Net line count must go **down** by checkpoint 4. Report the delta at every checkpoint.

## Validation bar

- Focused tests: ledger round-trip, ledger idempotency, `Migration` dry-run produces zero
  writes (assert via a read-only store or a write-spy), each ported migration's plan
  matches its pre-port plan on the same fixture (**equality proof before deletion**).
- Full non-GPU suite green, rebased on current `sun`. Establish your own baseline.
- `lint-imports` and `check_file_size_ratchet.py` exit 0 (these work again only after
  `brief_gate_restoration_2026-08-04.md` lands — do not start before it).
- `git diff --check` + `py_compile` clean.

## Reporting

Branch `agent/palette/migration-ledger` from current `sun`. Per checkpoint: what landed,
the quarantine decision and its justification, ledger schema, the three ported migrations'
LOC delta, and — the headline — the applied-census table with pending counts per
migration. Flag every disagreement you find between the strategy doc's Layer 3 estimate
(~1–1.5k LOC of dead tooling) and what the census actually shows.
