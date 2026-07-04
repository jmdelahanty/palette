# Brief: mint instance keys for manual curation rows + close the analysis-writer key gap

**From:** commander session, 2026-07-04
**Status: READY** — no holds. Touches only `shared/refined_detect_curation.py`, the three
analysis run writers, and `shared/instance_keys.py`; disjoint from utils Phase 2 and the
detect-pointer retirement briefs. Parallel-safe.
**Read first:** `src/fisheye/shared/instance_keys.py` (whole file, it's small),
`docs/identity_lineage_staleness_review.md` Rec 2 status, and the audit findings recap
below. Env: `~/miniconda3/envs/palette-py311/bin/python`; if testing from a worktree,
`PYTHONPATH=src` or imports resolve to the installed main checkout.

## Context — the two residual instance-key gaps (2026-07-04 audit)

Instance keys are minted once at detect (`mint_detection_instance_keys`: BLAKE2b over
`recording | frame | class | quantized bbox`, duplicate-ordinal disambiguation, in-call
uniqueness check) and copied verbatim downstream. Two copy-through gaps remain:

**Gap A — curation all-or-nothing drop.** `shared/refined_detect_curation.py:~2839-2845`:
when building the instances rowset, keys are gathered from the source detect run only if
`np.all(valid_instance_rows)` — i.e. every row has an in-range
`instance_source_detect_row_index`. One manual/out-of-range row (index `-1`) and the
ENTIRE rowset silently gets no `instance_key`. The guard exists for a good reason:
uint64 fancy-indexing with `-1` wraps to the last element and would assign another
detection's key to a manual row — a false identity. But the all-or-nothing silence
means one hand-drawn box degrades a whole curated run to positional fallback, with no
signal why.

**Gap B — analysis writers exclude keys.** `analysis/subject_shape_runs.py:~977`,
`analysis/tail_kinematics_runs.py:~467`, `analysis/tail_posture_view_runs.py:~357`: the
`copy_row_lineage_arrays` `names=` allowlist omits `instance_key` (and
`source_crop_row_ids`), so analysis rows are positional-only.

## Design decision (maintainer-approved direction)

Manual curation rows are NEW instances whose point of origin is curation — so minting
for them is legitimate ("mint at the point of origin", not "never mint downstream").
Requirements:

1. **Namespaced payload.** Manual-row keys must be minted with an origin discriminator
   in the hash payload (e.g. an `origin="manual_curation"` component), so a manual box
   that quantizes identically to an existing detection CANNOT collide with the copied
   key. Do this by extending `mint_detection_instance_keys` (e.g. optional
   `payload_context: str | None`) rather than forking a second minting function — one
   implementation, one hash discipline. Keys stay deterministic: same curation decisions
   → same keys; no run-ids or timestamps in the payload.
2. **Per-row origin provenance.** The curated rowset becomes a MIX of copied and minted
   keys. Record which is which — e.g. a parallel `instance_key_origin` array (or a
   compact encoding you justify) with values like `copied_from_detect` /
   `minted_at_curation`. The existing rowset-level policy attr pattern
   (`instance_key_policy` in crop) is precedent but is not granular enough here.
3. **Merged-rowset uniqueness assertion.** After combining copied + minted keys, assert
   uniqueness across the full rowset and fail loudly on violation. The existing check
   inside `mint_detection_instance_keys` only covers a single minting call.
4. **Kill the silence either way.** Rows that STILL end up without keys (if any path
   remains) must be logged/stamped, never silently omitted.
5. **Gap B:** add `"instance_key"` and `"source_crop_row_ids"` to the three analysis
   writers' `names=` tuples. `copy_row_lineage_arrays` handles absence gracefully and
   the writers already stamp `row_lineage_copied`/`row_lineage_missing`.

## Constraints

- `mint_detection_instance_keys` default behavior must be BIT-IDENTICAL for existing
  callers — detect-minted keys on re-run must not change. Add a test locking a known
  payload → known digest for the no-context path.
- Legacy curated runs are untouched; this changes writes going forward only.
- Verification semantics in `row_lineage.py` need no change (mixed-origin keys verify by
  equality like any others) — but confirm, don't assume; if the verifier special-cases
  anything about key provenance, report before proceeding.
- One commit per concern: minting extension, curation integration, analysis writers,
  tests may be folded into their respective commits.

## Validation bar

- New tests: manual-row minting (namespaced, deterministic, no cross-origin collision on
  an identical-bbox fixture), mixed-rowset uniqueness failure case, per-row origin
  array correctness, analysis writers copying keys through (and `missing` for legacy).
- Determinism lock test for the existing no-context minting path.
- Focused suites: instance_keys, row_lineage, refined_detect_curation, the three
  analysis writers' test files.
- Full non-GPU suite green on branch tip rebased on current `sun` (baseline ~3328
  passed — recount, it grows).
- `git diff --check` + `py_compile` clean.

## Reporting

Branch `agent/curation-instance-keys`, push the branch only (never `sun`). Report:
commits, the payload/namespace format chosen, the per-row origin encoding chosen and
why, uniqueness-assertion behavior, validation results, and anything that contradicts
this brief's line numbers (they drift).
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
