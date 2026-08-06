# Wave index — 2026-08-04

**From:** commander session, 2026-08-04.
Five briefs from a six-axis read-only audit of the repository (architecture, schemas,
provenance, testing/CI, automation, docs/hygiene). Read this before dispatching any of
them.

---

## The one-line finding

The design is good and almost none of it is enforced. Every guardrail this repo built for
itself — the import-linter contract, the file-size ratchet, the test suite, the type
annotations — is currently off, stale, or scoped so narrowly it cannot fail. **CI has
failed on every run since 2026-07-24.**

This wave is enforcement repair plus the science-integrity fixes that shouldn't wait for it.

---

## Dispatch order

```
brief_gate_restoration_2026-08-04.md          ← BLOCKS the three below. Dispatch first, alone.
    │
    ├── brief_migration_ledger_2026-08-04.md
    ├── brief_diagnostics_exit_contract_2026-08-04.md
    └── brief_enforcement_wiring_2026-08-04.md        (authored separately, same day)

brief_failclosed_science_defaults_2026-08-04.md   ← parallel, no dependency
brief_publication_readiness_2026-08-04.md         ← parallel, no dependency
```

Two briefs can start immediately and in parallel with gate restoration. The rest need
working gates, because their own validation bar calls for `lint-imports` and the file-size
ratchet to pass — which is impossible today.

**`brief_enforcement_wiring_2026-08-04.md` was authored in a separate session on the same
day and is complementary, not conflicting** — it wires 13 existing but unconnected
`scripts/check_*` gates into CI (only `check_file_size_ratchet.py` is currently wired).
Two notes on the seam:

- It inherits the same blocker: its mandatory pre-checkpoint gate list includes
  import-linter, which aborts on a config error today. Land gate restoration first.
- Both briefs modify `.github/workflows/ci.yml`. Gate restoration **splits** the quality
  job so a red shard cannot mask a red boundary check; enforcement wiring **adds** gate
  invocations. Land gate restoration first and enforcement wiring rebases onto the split
  job — the reverse order produces a merge conflict in the workflow and re-buries the
  new gates inside the same maskable job.
- Its governing rule ("a retroactive gate ships as a ratchet, never a hard pass/fail") is
  the same rule gate restoration applies to the import contract. They agree; keep it that
  way and do not invent a second ratchet idiom.

| Brief | Depends on | Touches | Conflicts with |
|---|---|---|---|
| gate restoration | — | `pyproject.toml`, `ci.yml`, ratchet baseline, test markers | none (touches no `src/` logic) |
| migration ledger | gate ckpt 2 | `utils/` one-shots, new `shared/` migration base | subtraction wave (see below) |
| fail-closed defaults | — | `analysis/chaser_epoch_behavior_summary.py`, config models | publication readiness (`src/config_models.py`) |
| diagnostics exit contract | gate ckpt 2 | `utils/`+`diagnostics/` checkers | migration ledger (both in `utils/`) |
| publication readiness | — | repo root, `src/*.py`, `src/chaser_analysis/` | fail-closed (`src/config_models.py`) |

**Known collision:** `src/config_models.py` is deleted by *fail-closed* ckpt 3 and also
listed in *publication readiness* ckpt 3. Whichever lands first takes it; the other reports
it as already gone.

**`utils/` collision:** migration-ledger and diagnostics-exit-contract both work inside
`utils/` but on disjoint sets (one-shot `backfill_*`/`migrate_*` vs read-only
`audit_*`/`check_*`). Dispatch both only if you can merge checkpoints promptly; otherwise
serialize, ledger first.

---

## Reconciliation with work already in flight

**`docs/utils_reorganization_strategy.md` (2026-07-09) is the authority on the reorg, and
this wave does not replace it.** That document already specifies what an independent audit
independently re-derived: retire the name `utils`, an `apps/` layer where the presence of
`__main__` is the load-bearing predicate, a `forbidden: fisheye.utils` contract as the
keystone, and `RETIRES_AFTER` markers on migrations. It is a good plan. It has not been
executed, and its Phase 0 gate is broken.

Where this wave extends it:

1. **Phase 0's gate is dead.** `lint-imports` aborts on a config error before evaluating
   any contract, and has for ten days. The keystone forcing function does not exist in
   practice. → `brief_gate_restoration`.
2. **The layers contract enforces one rule.** All 20 subpackages sit on one `:`-joined
   line, and `:` means *non-independent* in import-linter. Even repaired, it would only
   assert "`shared` must not import upward." → `brief_gate_restoration` ckpt 3.
3. **`RETIRES_AFTER` is an expiry, not a ledger.** Nothing records whether a backfill was
   ever *applied*; zero of 67 write a marker, and the 50 docs mentioning them describe
   intent only. Phase 3's "verify-then-delete" has not moved because verification is
   impossible. → `brief_migration_ledger`.
4. **`repair_keypoint_offset_corruption` is under-classified.** The strategy doc lists it
   as ordinary verify-then-delete. It heuristically *searches* for an offset and writes no
   repaired-marker, so a second run corrupts. → escalated to quarantine.
5. **Layer 3 is larger than estimated.** The strategy doc puts the dead surface at
   ~1–1.5k LOC. Census: 154 modules with no importer, **53 with no reference anywhere
   including docs**, and 28,309 LOC in the one-shot cluster. The doc's own guardrail
   (*gate deletion on docs + operator knowledge, not the import graph*) is correct and is
   preserved in the briefs — the 53 are precisely the subset that passes that gate.

**`agents_todo/brief_subtraction_wave_2026-08-03.md` is in flight and takes precedence
inside `src/fisheye/`.** Two hard constraints inherited from it:

- Its **Tier D** rules that `utils/` and `diagnostics/` modules unreachable from a pipeline
  entry point are human-invoked CLIs **by design**. No brief in this wave authorizes
  mass-deleting `utils/` on import-graph silence. Deletion is gated on the applied-census
  (ledger brief) or on verified-zero-references-including-docs.
- Its validation bar lists `lint-imports` as a mandatory gate. **That gate cannot pass
  today.** Any agent on that brief is currently blocked or has learned to ignore a red
  gate — a second reason to land gate restoration first.

**`agents_todo/brief_utils_phase2.md`** (strategy doc Phase 2, moving ~8 misfiled libraries
down) is unaffected by this wave and can proceed independently.

---

## Standing rules for every brief in this wave

- Local `sun` is ground truth. Fresh worktree from **current** `sun`; rebase at each
  checkpoint.
- Env: `~/miniconda3/envs/palette-py311/bin/python` (conda — never uv, never a `.venv`).
- **Establish your own test baseline.** The suite is currently red; no previously recorded
  pass count is trustworthy.
- Mandatory checkpoint stops. Do not push, do not merge — the commander verifies each.
- Every suppression, ratchet entry, and quarantine carries a dated comment naming the
  reason. Silent suppression is the failure mode this wave exists to end.
- Recount every number in these briefs before acting on it. They were verified on
  2026-08-04 against `agent/palette/derived-analytics-storage-contracts-20260803`;
  branches move.
- **If a test asserts the buggy behavior is correct, the test is part of the defect.** This
  repo has precedent (`tests/unit/fisheye/test_registry_dedupe.py:250-280`). An agent
  optimizing for green will preserve a bug and conclude it was intentional. Report it.
- Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Anti-goals — state these back in your first checkpoint

1. **Do not wire unwired modules.** The default action for something unreachable is delete
   or explicitly mark staged, never connect.
2. **Do not add modules.** Except where a brief explicitly requires promoting an existing
   implementation to a shared home (the migration base class is the one sanctioned
   addition). Net line count goes **down**; report the delta at every checkpoint.
3. **Do not fix violations you were asked to make visible.** Gate restoration ratchets and
   records; it does not repair.
4. **Do not confuse eye masks with eye angles.** Masks are deprecated and nearly severed.
   `analysis/eye_angle_*.py` was committed 2026-08-03 and is live.
5. **Do not default to `/nvme1/recordings`.** It is the stale store copy. The canonical
   registry lives under `/groups/`.
