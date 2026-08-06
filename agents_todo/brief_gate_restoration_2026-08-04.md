# Brief: gate restoration — turn the forcing function back on

**From:** commander session, 2026-08-04
**Status: READY. This brief BLOCKS every other brief in the 2026-08-04 wave.**
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Read first:** `docs/utils_reorganization_strategy.md` §"The forcing function" (this
brief repairs the Phase 0 gate that document specifies), `.github/workflows/ci.yml`,
`pyproject.toml` `[tool.importlinter]`.

Ground rules: local `sun` is ground truth; fresh worktree from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python` (conda — do NOT use uv or create a `.venv`).

---

## Why this is first

The strategy doc's Phase 0 gate landed and then died. Verified 2026-08-04:

- `lint-imports --config pyproject.toml` exits **1** on a config-validation error before
  evaluating a single contract:
  `Missing layer in container 'fisheye': module fisheye.capture does not exist.`
  `fisheye.capture` was deleted by `35a872bf` (2026-07-25); the layer entry at
  `pyproject.toml` was not updated. `src/fisheye/capture/` is now an empty directory
  containing only a stale `__pycache__`.
- CI on `sun` has failed on **every run since 2026-07-24** (last green
  2026-07-24T14:40). The most recent run had **9 of 12 test shards red plus the quality
  job**.
- The quality job houses `lint-imports`, `scripts/check_file_size_ratchet.py`,
  `pytest --collect-only`, and the `palette --help` smoke test. All four have gated
  nothing for ten days.

**Consequence for other agents:** `agents_todo/brief_subtraction_wave_2026-08-03.md`
lists import-linter as a mandatory pre-checkpoint gate. That gate cannot pass today
regardless of the diff. Until this brief lands, any agent following that instruction is
either blocked or is learning to ignore a red gate.

---

## Scope — four checkpoints, stop and report after each

### Checkpoint 1: make the contract evaluate at all

1. Remove `capture` from the `layers` list in `pyproject.toml` `[tool.importlinter]`.
   Delete the empty `src/fisheye/capture/` directory if it is not already gitignored.
2. Audit **every** name in that layers list against `src/fisheye/` on disk. `capture`
   will not be the only casualty of the last ten days — report what you find.
3. Run `lint-imports`. It will now evaluate contracts and is **expected to fail** with
   real violations. Capture the full violation list verbatim into the checkpoint report.
   **Do not fix the violations in this checkpoint.**
4. **STOP and report.** The violation list is the input to checkpoint 2 and is a commander
   decision, not an agent decision.

### Checkpoint 2: ratchet the real violations, do not paper over them

Expect roughly **15 `shared → non-shared` module-level imports** beyond the 7 already in
`ignore_imports` (verified by AST scan; recount yourself — branches move). Known members
include `shared/pose_model_schema_binding.py`, `shared/zarr/body_frame_producer.py`,
`shared/zarr/stage_arrays.py`, `shared/subject_shape_coordinate_publication.py`, and four
`shared/zarr/*_publication.py` modules reaching
`analysis_workflows/materializers/atomic_run_publisher`.

5. Add each to `ignore_imports` **with a dated `# RATCHET:` comment in the existing
   house style**, naming the reason and the intended exit. The debt becomes written
   down, not invisible. `unmatched_ignore_imports_alerting = "error"` stays on.
6. CI quality job goes green. **STOP and report.**

### Checkpoint 3: make the layers contract mean something

The current contract enforces exactly one rule. In import-linter 2.x, `:` between
siblings means **non-independent** (they may import each other freely); `|` means
independent. All 20 subpackages currently sit on one `:`-joined line above `shared`, so
the whole contract reduces to *"`shared` must not import upward."* `analysis ↔
analysis_workflows`, `refinement ↔ tune`, `cli ↔ utils`, `visualization ↔ analysis` are
all unconstrained.

7. Replace the single line with real strata. Minimum viable stratification, consistent
   with the strategy doc's target (`shared < domain < cli < apps`):
   - top: `cli : status_page : group_analytics_viewer : labeling : diagnostics : tune`
   - then: `utils : analysis_workflows : refinement : training`
   - then: `analysis : detection : tracking : segmentation : pose : registry : cluster`
   - bottom: `shared`
   Use `|` where you actually want sibling independence.
8. Expect new violations. Ratchet them the same way, with comments. If a stratum is
   unachievable even with ratchets, **report it rather than flattening the contract back**
   — a contract that cannot fail is the thing being fixed here.
9. **STOP and report.**

### Checkpoint 4: close the coverage holes in the other gates

10. **File-size ratchet.** `scripts/file_size_ratchet_baseline.json` guards exactly four
    files and **omits the two largest in the repo**:
    `utils/audit_coordinate_contracts.py` (**16,856 lines**) and
    `analysis/track_kinematics.py` (14,305). Add every file over ~4,000 lines to the
    baseline at its current size. This is a ceiling, not a mandate to split — splitting is
    a separate brief. Also add `utils/export_cross_recording_analytics.py` (5,916).
11. **Split the CI quality job** so `lint-imports`, the ratchet, and the smoke test each
    fail independently of the test shards. Today one red shard hides a red boundary check.
12. **Triage the test shards.** Do not mass-`xfail`. Separate genuine regressions from
    infrastructure flake — at least one failure class (`array_changed_during_binding`,
    shard 4, 1 failed / 576 passed) has an mtime-race signature. Genuine regressions get
    fixed or reported; flake gets an explicit `@pytest.mark.flaky` / `xfail(strict=False)`
    **with a tracking comment naming the race**. Report the split with counts.
13. Recommend (do not implement) branch protection on `sun` in your report.

---

## Explicitly OUT of scope

- Fixing the underlying `shared → upward` violations. Ratchet and record; the strategy
  doc's Phases 1–2 own the actual repair.
- Splitting `audit_coordinate_contracts.py` or any other large file.
- Renaming `utils/` or creating `fisheye/apps/` — strategy doc Phases 0/7 own that, and
  they depend on this gate working first.
- Any `src/fisheye/` deletion. `brief_subtraction_wave_2026-08-03.md` owns deletions.
- Adding a `forbidden: fisheye.utils` contract — that is strategy doc Phase 7 and is
  gated on Phases 1+2, not on this brief.

## Constraints

- **Zero behavior change to `src/`.** This brief touches `pyproject.toml`,
  `.github/workflows/ci.yml`, `scripts/file_size_ratchet_baseline.json`, and test markers
  only. If you find a live bug while triaging shards, **report it, do not fix it here** —
  a mixed diff makes the gate restoration unrevertible.
- Every ratcheted import and every quarantined flaky test carries a dated comment naming
  the reason. Silent suppression is the failure mode this brief exists to end.
- One commit per checkpoint, independently revertible.

## Validation bar

- `lint-imports --config pyproject.toml` exits **0** and reports a nonzero number of
  contracts *evaluated* (paste the output — "0 contracts" is a failure, not a pass).
- `scripts/check_file_size_ratchet.py` exits 0.
- `git diff --check` clean; `py_compile` clean.
- Full non-GPU suite green:
  `PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python -m pytest tests -m "not gpu" -q -n 16`.
  **Establish your own baseline first** — the suite is currently red, so no prior count is
  trustworthy. Report before/after counts.
- A CI run on your branch that is green, or a written account of exactly what remains red
  and why.

## Reporting

Branch `agent/palette/gate-restoration` from current `sun`. Per checkpoint: what changed,
the verbatim `lint-imports` output, the ratchet additions with reasons, the shard triage
split (genuine vs flake, with counts), and the list of stale layer names you found in
step 2. State back the anti-goal: **you did not fix any violation, you made them
visible.**
