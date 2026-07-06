# Brief: detect-review legacy pointer cleanup — census + decision memo (`agent/detect-review-pointer-census`)

**From:** commander session, 2026-07-06
**Status: READY.** One agent, two phases, **mandatory CHECKPOINT after Phase 2**
(census + decision memo). Phase 3 exists but is NOT cleared by this brief — the
commander clears it at the checkpoint, or doesn't.
**Do NOT push or merge — the commander verifies and merges.**
**This brief authorizes ZERO writes to real recording stores.** Phase 1–2 are
read-only against `/nvme1/recordings` and the registry; Phase 3 (if cleared)
ships a dry-run-default tool that only the maintainer executes.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `HANDOFF_2026-07-05.md` operating-notes section. Ground rules:
local `sun` is ground truth; fresh worktree on `agent/detect-review-pointer-census`
from CURRENT `sun`; env `~/miniconda3/envs/palette-py311/bin/python` (conda; never
uv, never a `.venv`); sync code only; re-locate cited line numbers by content.
Local gates before the checkpoint: import-linter via
`scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`,
focused tests, then full suite
`PYTHONPATH=src ... -m pytest tests -m "not gpu" -q -n 16`.
Baseline: **3,451 passed / 2 skipped** at merge `c57e810`; recount on your branch.

## State (commander-verified 2026-07-06 — do not re-derive, do re-verify cheaply)

The WRITER side is at its sanctioned end state (verified 2026-07-05, brief
`brief_promotion_authority.md`): the only `detect_review_status_latest` writers are
the two annotated migrations (`utils/migrate_legacy_detect_labels.py`,
`utils/migrate_refined_detect_sparse.py`) plus the batch wrapper's `--no-pointer-update`
help text (`utils/migrate_refined_detect_sparse_batch.py:278`). **These stay. Touching
them is out of scope.**

The remaining READER surface (grep `detect_review_status_latest` across `src/`,
12 hits in 9 files at time of writing) is:

1. `shared/run_resolution.py` — the central legacy fallback
   (`LEGACY_DETECT_REVIEW_AUTHORITY_ATTR`): in `AUTHORITATIVE` mode, when the parent
   has no `authoritative_run` attr AND the parent is a refined-detect parent
   (`refined_detect_runs`/`refined_runs`) AND the legacy attr names a COMPLETE run,
   it resolves to the legacy pointer with `fallback_used=True` and
   `source_attr="detect_review_status_latest"` recorded in the result.
2. `labeling/task_generation.py::_detect_review_status_for_zarr` — candidate-name
   ordering `("detect_review_status_latest", "latest")`, then directory listing.
3. Three batch pipelines (`utils/run_megabouts_batch_pipeline.py`,
   `utils/run_movement_bout_batch_pipeline.py`,
   `utils/run_subject_mask_batch_pipeline.py`) — pointer preference
   `("latest", "latest_materialized", "detect_review_status_latest")`.
4. `shared/zarr/schema.py:80` — schema description of the optional attr
   (documentation; retires WITH the reader retirement, not before).

Retiring 1–3 is safe only if no real store still resolves through the legacy attr.
Nobody has measured that. That measurement is this brief.

## Phase 1 — read-only census of real stores

Scan every recording store and report where the legacy pointer still matters.

- **Enumeration, two independent modalities (report both and diff them):**
  (a) filesystem walk of `/nvme1/recordings` (`PALETTE_RECORDINGS_ROOT` env override
  respected — reuse the pattern in `utils/audit_swim_bladder_tuning_metadata.py`),
  finding every `*.zarr` under each recording (layout: per-recording `zarr/` dirs
  holding training + analysis stores); (b) registry enumeration via read-only
  `RegistryPaths` defaults. Stores in one modality but not the other are themselves
  a finding.
- **Per store, attrs-only reads** (open read-only; never load arrays; tolerate
  OSError/missing/consolidated-metadata quirks per store and keep scanning — a
  store that cannot be read is a table row, not a crash): for each refined-detect
  parent (`refined_detect_runs`, `refined_runs`) record:
  - presence + value of `authoritative_run`, `detect_review_status_latest`,
    `latest`, `latest_materialized`;
  - whether each named run exists as a child and is complete
    (`is_run_complete_in_parent`, same `legacy_default` the resolver uses);
  - **would the `run_resolution` legacy fallback fire TODAY** (authoritative
    missing AND legacy present AND legacy run complete);
  - **would removing the legacy key change the winner** in the batch pipelines'
    preference order and in `task_generation`'s candidate order (compute the
    winner with and without the key — pure attr logic, no pipeline execution).
- **Deliverable:** `docs/diagnostics/detect_review_pointer_census_2026-07-06.md` —
  per-store table plus summary counts (stores scanned / unreadable / fallback-would-fire /
  winner-would-change / clean), and the census script committed under
  `src/fisheye/diagnostics/` (read-only, rerunnable, `--recordings-root` +
  `--registry` args) with a unit test on a fixture store.

## Phase 2 — decision memo, then CHECKPOINT

Classify every store from Phase 1 into exactly one bucket:

- **SAFE** — `authoritative_run` present, or no legacy attr at all: legacy-reader
  removal changes nothing for this store.
- **BACKFILLABLE** — `authoritative_run` missing; legacy attr names a complete,
  existing run; no conflict with `latest` semantics: stamping
  `authoritative_run = <legacy value>` is mechanical and unambiguous.
- **AMBIGUOUS** — legacy attr names a missing/incomplete run, or disagrees with
  `latest`/`latest_materialized` in a way that changes any winner computed in
  Phase 1: enumerate each with the specific conflict. These need per-store
  maintainer eyes, not a script.

End the memo with a recommendation: retire-now / backfill-then-retire / keep, with
the bucket counts as the argument. **CHECKPOINT: stop and report** — census doc,
memo, suite counts, premise discrepancies. The commander reviews with the
maintainer and decides whether Phase 3 opens.

## Phase 3 — backfill tool (NOT cleared by this brief; specified to prevent scope drift)

If and only if cleared at the checkpoint: a `utils/` backfill tool that stamps
`authoritative_run` from the legacy pointer for BACKFILLABLE stores only —
`--dry-run` is the default and prints the exact per-store mutations; `--execute`
requires an explicit store list (no implicit "all"); unit tests on fixture stores;
**never executed against `/nvme1/recordings` by the agent** — the maintainer runs
`--execute` from their own session. Reader retirement (deleting the fallback in
`run_resolution.py`, tightening the key tuples in the batch pipelines and
`task_generation.py`, updating `schema.py`) is a SEPARATE later slice, gated on the
backfill having actually been executed and re-censused clean.

## Out of scope — hard boundaries
- Any write to any real store or the registry (Phases 1–2 are read-only; Phase 3
  writes only via maintainer-run `--execute`).
- The two sanctioned pointer writers and the batch wrapper's help text.
- Retiring or altering ANY of the reader sites 1–4 in this brief — including
  "harmless" reorderings of the key tuples. Measurement first.
- Eye-mask paths (deprecated), keypoint review status logic in
  `task_generation.py` (only the detect-review pointer ordering is in scope for
  the census computation).

## Reporting (checkpoint)
Branch + commit SHAs, census doc path, bucket counts, the two-modality store-list
diff, any unreadable stores with errors verbatim, whether the writer-side premise
still holds (re-grep the writers — cheap), suite counts (recounted baseline →
final), premise discrepancies.
