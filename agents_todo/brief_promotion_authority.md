# Brief: route promotion authority through approve; delete the last live legacy pointer write

**From:** commander session, 2026-07-05
**Status: HOLD — dispatch after the registry reconcile agent's work is merged** (no file
overlap; hold is merge-window hygiene only). Small slice: one source file + tests.
**Read first:** `agents_todo/brief_detect_pointer_retirement.md` (the completed
predecessor whose out-of-scope carve-out this closes) and the promotion flow in
`src/fisheye/tune/detect_training_promotion_backend.py` (~lines 637-660 and 930-960;
re-locate by content).

## Context and decision (maintainer-approved 2026-07-05)

The detect-pointer retirement retired every legacy `detect_review_status_latest` writer
except promotion, which uses the pointer as a PENDING-marker: `_resolve_training_refined_run`
writes it at run creation right after `mark_run_pending` (~651-652), and the completion
path later stamps `detect_review_status = {state: "approved", method: "promotion", ...}`
and calls `mark_run_complete` (~937-958). Maintainer decision:

1. **Delete the creation-time pointer write** (~652). Its "run being built" role is
   redundant with `latest_pending`, and it currently exposes mid-promotion pending runs
   to readers that don't check completeness (task_generation, batch pipelines).
2. **At the completion site** (immediately after `mark_run_complete`), route authority
   through **`palette approve`** (stage `refined_detect`, apply=True) with
   promotion-actor provenance, then mirror `set_authoritative_run` — the SAME pattern
   the retirement slice used in `accept_detect_review.py`/`set_detect_review_status.py`
   (`_approve_refined_detect_authority` helpers — copy that shape). Promotion must not
   become a second direct authority-writing path outside the approve funnel.
3. Keep the `{method: "promotion"}` review-status stamp as-is — it honestly records
   how approval happened.

**Escape hatch (report, don't improvise):** `approve`'s envelope was built against
analysis stores. If it cannot resolve training-store datasets (dataset-resolution or
registry checks fail structurally, not incidentally), STOP that item and report the
exact failure. The documented fallback is direct `set_authoritative_run` with promotion
provenance — but that is a maintainer-confirmed deviation, not your call to make
silently.

## Constraints

- Fail-closed: if `approve` returns a non-ok envelope, promotion must surface the
  failure — no silent skip of the authority stamp on an otherwise-"successful"
  promotion. Decide (and defend in the report) whether that fails the whole promotion
  or returns a partial-status payload; precedent favors failing loudly.
- Do NOT touch `crop.attrs["canonical_refined_detect_run"]` / `canonical_refined_detect_path`
  (~950-951) — noted as a bespoke pointer for a future run-resolution pass, out of scope.
- Do not touch the two annotated migrations or anything eye-mask.
- After this slice, the grep proof for live `detect_review_status_latest` writers in
  src/ must be EXACTLY the two annotated migrations.

## Validation bar

- Tests: promotion completion sets `authoritative_run` with promotion provenance;
  creation-time pointer absent; approve-failure path fails loudly; legacy readers
  (run_resolution AUTHORITATIVE) resolve the promoted run.
- Focused: promotion backend tests + detect-review family + run_resolution.
- Full non-GPU suite: `PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python -m
  pytest tests -m "not gpu" -q -n 16`. Recount baseline from current `sun` (grew past
  3364 with the registry reconcile merge).
- `git diff --check` + `py_compile` clean.

## Reporting

Branch `agent/promotion-authority` from current `sun`. Commits: pointer-write deletion;
approve routing + tests (separate if clean). Report: conversion notes, the
fail-loudly-vs-partial decision, grep proof, whether the escape hatch was needed,
validation counts + PYTHONPATH=src confirmation.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
