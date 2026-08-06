# Brief: diagnostics exit contract — make the checkers gateable

**From:** commander session, 2026-08-04
**Status: READY after `brief_gate_restoration_2026-08-04.md` checkpoint 2.**
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Read first:** `src/fisheye/cli/envelope.py` (the frozen contract you are propagating),
`src/fisheye/utils/audit_analysis_staleness.py:707` (the one member that already does this
right), `docs/utils_reorganization_strategy.md` Phase 6 ("graduate 2–3 contract-checkers
into `palette verify`") — this brief produces the shortlist that phase consumes.

Ground rules: local `sun` is ground truth; fresh worktree from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python`.

---

## The defect

The ~55 `audit_*`/`check_*`/`validate_*`/`inspect_*` modules in `utils/` are a write-only
tool fleet: nothing runs them automatically (verified — **zero** invocations in
`.github/workflows/ci.yml`, no cron anywhere, `palette status` builds its own payload and
calls none of them), and the ones most likely to be automated **have inverted exit codes**.

Verified 2026-08-04, `utils/check_recording_steps.py`:
- `:3393` — `print("No recordings found."); return 1`
- `:3396` — `print("No status mismatches found."); return 0`

Empty result set ⇒ failure. Clean bill of health ⇒ success. `utils/check_training_registry.py`
has **eleven** consecutive `print("No X rows found."); return 1` sites (`:3197-3215`).
Any CI gate or cron wrapped around these fires backwards today.

The verbs carry no meaning either. `validate_imported_run_group.py:465` and
`validate_refined_detect_run.py:279` both end in
`print(json.dumps(result)); return 0 if status == "ok" else 1` — a real machine contract.
But `validate_detect_training_zarr.py:37` prints "Validation passed." and **never** returns
nonzero. `audit_coordinate_contracts.py:16852` returns 0 unconditionally.

Meanwhile the frozen envelope at `cli/envelope.py` (`SCHEMA =
"palette.cli.workflow_oracle.v1"`, `EXIT_OK/FAILED/BLOCKED/USAGE`) has **exactly one**
production importer: `cli/palette.py`. Sixty-odd tools each invented their own output shape
next to one frozen standard nothing else speaks.

---

## Scope — three checkpoints, stop and report after each

### Checkpoint 1: fix the inversion, then census the exit contracts

1. Fix the inverted exit codes in `check_recording_steps.py` and
   `check_training_registry.py`. The rule: **exit code reports whether the check found a
   problem, never whether it found rows.** Empty result set with no problem ⇒ 0. Make
   "nothing to check" distinguishable from "checked, all clean" in the *output*, not the
   exit code.
2. Census every `audit_*`/`check_*`/`validate_*`/`inspect_*`/`list_*`/`report_*` module in
   `utils/` **and** `diagnostics/` into a table: name, LOC, exit-code behavior
   (always-0 / real predicate / inverted / raises), output format (prose / ad-hoc JSON /
   envelope), and whether anything references it. Commit as
   `docs/diagnostics_exit_contract_census_2026-08-04.md`.
3. **STOP and report.** The census drives checkpoint 2's shortlist, which is a commander
   decision.

### Checkpoint 2: give the gateable few a real contract

4. From the census, the commander picks **5–8 tools whose answer would actually change an
   action.** Propose the shortlist with reasoning; do not pick unilaterally. Strong
   candidates on current evidence: `audit_analysis_staleness` (already correct — use it as
   the reference), `validate_imported_run_group`, `validate_refined_detect_run`,
   `check_recording_steps` (post-fix), and whatever the census shows actually reads
   `palette_completion_epoch` (today **only two** of 338 utils modules do).
5. Give each shortlisted tool: envelope JSON output via `cli/envelope.py` (do **not**
   invent a parallel schema — `validate_refined_detect_run.py:22` declares its own
   `VALIDATION_SCHEMA`; fold it into the envelope or justify why it must stay separate),
   an explicit `--fail-on <severity>` knob modeled on `audit_analysis_staleness.py:707`,
   and honest exit codes.
6. Everything **not** shortlisted gets stripped of exit-code pretension: rename to
   `inspect_*` per the strategy doc's `apps/tools/` category, or leave the name and
   document that its exit code is not a signal. Nobody should be able to mistake a
   reporter for a gate.
7. **STOP and report.**

### Checkpoint 3: collapse the overlaps

Four clusters answer the same question more than once, and two of them can **disagree**.
Collapse each to one tool with a `--kind`/`--stage` flag. This is also strategy doc Phase 4
work — coordinate, do not duplicate.

8. **Refined-detect readiness — 5 tools, ~1,490 LOC, genuinely divergent predicates.**
   `inspect_refined_detect_run.py:21` uses `has_sparse_curated_refined_detect_instances_arrays`;
   `validate_refined_detect_run.py:13-14` uses `has_curated_refined_detect_surface`. **A run
   can be "curated" to one and not the other.** Determine which predicate is correct — this
   is a correctness finding, not a merge — report it, then collapse.
9. **Zarr storage size — 5 tools, ~3,790 LOC, across two packages.**
   `utils/audit_zarr_array_sizes.py`, `utils/audit_zarr_group_counts.py`,
   `utils/report_zarr_storage.py` (cold since 2026-02-04), `diagnostics/zarr_size_report.py`,
   `diagnostics/zarr_storage_census.py` (2,162 LOC, 2026-07-23). The new census **accreted
   onto** the old report rather than replacing it. Given the storage crunch, disagreement
   here has real cost. Keep the census; retire the rest.
10. **Training-zarr validation trio** — `validate_detect_/keypoint_/subject_mask_training_zarr.py`
    at 42/42/49 lines differ only in which `export_*` validator they import. This is the
    strategy doc's named `--stage` merge. Do it.
11. **Approval listers** — `list_unapproved_analysis_zarrs.py` (356, raw SQL) vs
    `list_unapproved_keypoint_analysis_zarrs.py` (221). A fork, not a parameter.
12. While here: 8 modules in this cluster call `sqlite3.connect` **directly**, bypassing
    `registry/db.py`; three files do both. Report them. Retarget only the ones inside tools
    you are already touching — a full retarget is a separate brief.

---

## Explicitly OUT of scope

- **Splitting `audit_coordinate_contracts.py`** (16,856 lines, ~50 tools in one file). It
  gets a ratchet ceiling in `brief_gate_restoration_2026-08-04.md`; decomposition is its
  own brief. Do not start it here.
- Wiring any of these into CI or a cron. This brief makes them *gateable*; the commander
  decides what actually gates. State this back in your report.
- Merging `utils/` and `diagnostics/` as packages. The split is historical, not principled
  — but the strategy doc's target structure (`apps/verify/`, `apps/tools/`) owns the
  resolution, not this brief.
- Deleting anything the census shows unreferenced. `brief_subtraction_wave_2026-08-03.md`
  Tier D is explicit: unreachable-from-a-pipeline is **by design** for human-invoked CLIs.
  Report candidates; do not act.
- `diagnostics/check_provenance_consistency.py` and `diagnostics/check_eye_mask_lineage.py`
  — both are load-bearing per the subtraction wave's Tier D despite appearances.

## Constraints

- **Changing an exit code is a behavior change with downstream reach.** Before each fix,
  grep `scripts/`, `.github/`, `*.sh`, and docs for anything invoking that module and
  branching on its status. Report what you find; fix callers in the same commit.
- Do not add a new output schema. Either the envelope or a documented, tested reason not to.
- Collapses must be proven behavior-preserving: run old and new on the same fixture and
  compare output **before** deleting the old tool.
- Net line count must go **down** by checkpoint 3. Report the delta.

## Validation bar

- Focused tests per changed tool: clean input ⇒ exit 0; seeded problem ⇒ nonzero; empty
  input ⇒ 0 with a distinguishable message; envelope output validates against
  `cli/envelope.py`'s schema.
- Output-equality proof for every collapse, before the delete commit.
- Full non-GPU suite green, rebased on current `sun`. Establish your own baseline.
- `lint-imports` + `check_file_size_ratchet.py` exit 0 (requires the gate-restoration
  brief to have landed).
- `git diff --check` + `py_compile` clean.

## Reporting

Branch `agent/palette/diagnostics-exit-contract` from current `sun`. Per checkpoint: the
exit-contract census table, the proposed shortlist with reasoning, the refined-detect
predicate finding (**which one is correct, and how many runs the two disagree on**), the
collapse equality proofs, LOC delta, and the list of direct-`sqlite3` bypass sites.
