# Brief: architecture review wave 1 — enforcement closure

> Historical implementation proposal, copied into an uncommitted main-based
> draft on 2026-09-05; it was not imported or integrated onto main.
> The READY status below records the original author's assessment, not current
> assignment or implementation authorization. Do not execute these packages
> unchanged: default-off strictness is not admission/publication closure,
> `sun` is a stale starting ref, and shared queue edits require one owner.
> Reconcile the packages and cited premises with
> [the second opinion](../docs/diagnostics/review_wave_second_opinion_2026-09-04.md)
> and [the landing handoff](../docs/diagnostics/review_docs_rules_landing_handoff_2026-09-05.md)
> against the final integration base before assigning implementation work.
> Dated corrections and current scope are in the
> [September 6 reconciliation](../docs/diagnostics/review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

Three independent implementation packages derived from
`docs/diagnostics/architecture_review_five_lens_2026-09-01.md`. Each package
is designed for one agent in its own isolated worktree. The packages touch
disjoint files and can run in parallel. Do not take more than one package per
agent.

**Date:** 2026-09-01
**Historical status:** READY (original assessment; not current authorization)
**Production effect:** none authorized by this brief. No selector, registry
authority, cohort release, or canonical-archive changes. Package A changes
runtime behavior of the registry completion gate and must ship behind the
strictness switch described below, default-off until the canary passes.

## Rules for every package

- Start from current `sun` in a clean isolated worktree. Follow `AGENTS.md`.
  Use `scripts/py` for every Python command. No `pip install` / `conda install`.
- Run pytest outside the sandbox (`scripts/py -m pytest ...`). In-sandbox
  validation is limited to `scripts/py -m py_compile` and `git diff --check`.
- Re-verify every cited line number against the code you check out. The
  review was read at `dcdcc081`; line numbers drift. If a cited claim is no
  longer true, stop and record that in your handoff instead of implementing
  against a stale premise.
- Before writing code, read the "Read first" list for your package in full,
  then write a short "Verified premises" section at the top of your handoff
  listing each claim you confirmed with the command you used.
- Keep each commit reviewable and single-purpose. Do not push. Do not merge.
  Do not fast-forward `/groups`. Hand off with the exact commit, CI status,
  and any unrun validation, per the Required CI rule in `AGENTS.md`.
- Do not author or edit any `docs/*contract*.md`. If your change makes an
  existing contract doc wrong, note the exact line in the handoff; the human
  will reconcile it.
- Adopt your package's tracking line into
  `docs/diagnostics/authority_consolidation_work_queue_2026-08-25.md` under
  "Stage 11 — Cross-cutting tests, CI, and production execution" as a single
  row with an ID of the form `ARCH-<letter>-001`. Do not create a new queue.

---

## Package A — Make the registry completion gate fail closed (O-1, O-2, G-1)

### Read first

1. `docs/diagnostics/architecture_review_five_lens_2026-09-01.md` §1, §3.3
   rows O1 and O4, §3.4 items O-1 and O-2, §7.4 row G2.
2. `docs/diagnostics/contract_enforcement_divergence_review_2026-08-21.md`
   "Unified diagnosis" and finding F1. This is the earlier statement of the
   same defect; note what it says about `maintenance.py` re-writing refused
   rows, because that is the second half of the neutralization.
3. `docs/zarr_run_completion_contract.md` in full, especially the paragraph
   near line 91 that describes `emit_stage_completion` as fail-closed.
4. `src/fisheye/registry/stage_complete.py` in full (it is ~500 lines).
5. `src/fisheye/shared/zarr_run_completion.py` lines 1-320: the completion
   epoch model and `mark_run_complete`.
6. `src/fisheye/registry/status_ledger.py` (`upsert_recording_step_status`)
   and `src/fisheye/registry/step_cascade.py`.
7. `docs/error_budget_policy_2026-08-11.md` Tier 0 and Tier 1 definitions.
8. `tests/unit/fisheye/test_detect_keypoints_step_status.py` for the existing
   test style around step status.

### Verify before implementing

- Confirm `emit_stage_completion` wraps its body in `except Exception` that
  prints a warning and returns `False`. Record the current line range.
- Enumerate every caller. At `dcdcc081` there were 19 call sites across 15
  files (`detection/detect_keypoints_traditional.py`,
  `detection/detect_keypoints_yolo.py`, `inference/predict_detections.py`,
  `inference/predict_pose.py`, `refinement/detect_quality.py`,
  `refinement/refine_detect.py`, `refinement/refine_keypoints.py`,
  `registry/derived_analysis_status.py` ×3,
  `shared/subject_mask_registry_status.py` ×2,
  `tracking/arena_assignment.py` ×2, `tracking/crop.py`,
  `tune/keypoint_failure_review.py`, `utils/keypoint_retry.py`,
  `utils/run_detection_local_publish.py`, `utils/run_keypoints_batch.py`).
  For each, record whether the return value is checked and whether the call
  happens before or after `mark_run_complete` for that stage.
- Confirm which exception classes the body can raise and classify each as
  (a) registry I/O or connectivity, (b) contract validation (array spec,
  provenance, eligibility, invalidation), or (c) other. Put the table in the
  handoff.
- Confirm `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` and which stages it covers.
- Check whether `PALETTE_DISABLE_REGISTRY_WRITES=1` short-circuits before or
  after validation. If validation is skipped under that flag, say so; it is a
  finding, not something to fix in this package.

### Implement

1. Add a module-level strictness switch, `PALETTE_STEP_STATUS_STRICT`
   (env var, values `0`/`1`, default `0` for this package). Document it in
   the module docstring.
2. Split the single `except Exception` into two paths:
   - Registry I/O and connectivity errors keep the current behavior in both
     modes (warn, return `False`), but now also record a structured reason
     (`gate_outcome="registry_io_error"`, exception class name) via the
     console message and, where a registry handle exists, an `error` status
     row with `details_json` carrying the reason.
   - Contract validation failures (array spec, provenance, eligibility,
     downstream invalidation) in strict mode re-raise after writing an
     `error` status row with `details_json` naming the failed check. In
     non-strict mode they behave as today but additionally write the `error`
     row and log `gate_outcome="contract_refused_nonstrict"`.
3. Add `gate_outcome` and `strict_mode` to `details_json` on every row the
   function writes, including the `ok` path, so a reconcile can count how
   many rows were admitted under each mode.
4. Do **not** reorder the call relative to `mark_run_complete` in this
   package. Instead, for each of the callers, add a comment-free, test-backed
   note in the handoff of whether the run is already `latest` at the time of
   the call. O-2 (running validation before promotion) is a follow-up package
   because it changes 15 callers' control flow.
5. Tests, all in-memory or `tmp_path` zarr, no live stores:
   - a contract failure in strict mode raises and leaves an `error` row;
   - the same failure in non-strict mode returns `False` and leaves an
     `error` row with `gate_outcome="contract_refused_nonstrict"`;
   - a registry I/O failure in strict mode returns `False` and does not
     raise;
   - the `ok` path carries `gate_outcome="admitted"` and the mode.
6. Add a one-paragraph note to `docs/zarr_run_completion_contract.md`'s
   maintainer, in the handoff only, identifying the line that must change
   from "fail-closed" to "fail-closed when `PALETTE_STEP_STATUS_STRICT=1`".

### Acceptance

- `scripts/py -m pytest tests/unit/fisheye -k "step_status or stage_complete"`
  green outside the sandbox.
- Full CI green on the branch.
- Handoff includes the caller table, exception classification table, and a
  proposed canary: one recording, one stage, `PALETTE_STEP_STATUS_STRICT=1`,
  on a commit-pinned deployment, with the exact command.

### Do not

- Do not flip the default to strict.
- Do not touch `registry/maintenance.py` reconcile paths (F1 second half);
  that belongs to the authority queue's existing items.
- Do not change `mark_run_complete`.

---

## Package B — Assert executing commit at LSF job start (O-3, O-5)

### Read first

1. `docs/diagnostics/architecture_review_five_lens_2026-09-01.md` §3.3 row
   O5, §3.4 item O-3, §4.3 row P4.
2. `AGENTS.md` "Git Push Rule" paragraphs on
   `scripts/deploy_palette_cluster_worktree.sh`, `scripts/py`, and
   "Cluster submissions from a dedicated deployment must record and pass its
   absolute `--palette-repo` path and full commit."
3. `scripts/deploy_palette_cluster_worktree.sh` in full.
4. `scripts/py` in full.
5. `src/fisheye/cluster/lsf/models.py` (find where `palette_commit` is
   recorded on the plan; at `dcdcc081` it was near line 256),
   `src/fisheye/cluster/lsf/runtime.py` in full (`run_with_status`),
   `src/fisheye/cluster/lsf/submission.py`.
6. `src/fisheye/shared/run_provenance.py` lines 1-80 for how `git_sha` and
   `git_dirty` are captured today; reuse that helper rather than shelling out
   again.
7. One planner that emits plans, e.g. `src/fisheye/cluster/clipped_inference.py`
   near the line that sets `palette_commit`, and
   `tests/unit/fisheye/test_cluster_lsf.py` for the existing test fixtures.

### Verify before implementing

- Confirm `runtime.py` and `models.py` contain no commit verification
  (`grep -n commit` on both).
- Confirm how `run_with_status` learns which plan/job it is executing (argv,
  env, status path) and whether the plan JSON is reachable from inside the
  job.
- Confirm what `scripts/py` does to `sys.path` and whether the executing
  `fisheye` package location can be resolved to a git checkout from inside
  the job (`fisheye.__file__` → walk up to `.git` or `.git` file for
  worktrees).
- Confirm whether any existing job already runs from a non-git location
  (wheel install, `/tmp` copy) that would make the assertion impossible. If
  so, list them; the assertion must degrade to a recorded warning there, not
  a refusal.

### Implement

1. Add `fisheye.cluster.lsf.code_identity.resolve_executing_commit()` that
   returns `(sha, dirty, source_path, method)` using the existing
   `run_provenance` git helper on the directory containing the imported
   `fisheye` package. Handle detached worktrees (`.git` is a file).
2. In `run_with_status`, before executing the task: if the plan records
   `palette_commit`, compare. On mismatch, write the status JSON with
   `status="refused"`, `reason="commit_mismatch"`, both shas, and exit
   non-zero without running the task. On match, record
   `executing_commit`, `executing_dirty`, and `commit_check="passed"` in the
   status JSON. If the plan has no `palette_commit`, record
   `commit_check="plan_unpinned"` and continue.
3. Add an env override `PALETTE_LSF_ALLOW_COMMIT_MISMATCH=1` that downgrades
   refusal to a recorded warning. Log its use prominently in the status
   JSON so the reconcile can count it.
4. Tests: mismatch refuses; match passes and records; unpinned plan
   continues and records; override downgrades and records. Use a `tmp_path`
   git repo fixture (`git init`, one commit) rather than the real repo.
5. Update `docs/cluster_batching_guide.md` (or whichever doc describes
   `run_with_status` status JSON) with the new fields, in a single short
   section.

### Acceptance

- `scripts/py -m pytest tests/unit/fisheye/test_cluster_lsf.py` plus the new
  tests green outside the sandbox. Full CI green.
- Handoff includes the list of any job paths that cannot be asserted and why.

### Do not

- Do not modify the bash submitters in `scripts/submit_*_bsub.sh`.
- Do not change how `palette_commit` is recorded on the plan.
- Do not touch `run_provenance` attr grammars (P-6 is a separate package).

---

## Package C — Widen the ratchets (C-3, S-2 baseline only)

### Read first

1. `docs/diagnostics/architecture_review_five_lens_2026-09-01.md` §5.3 row
   S1, §5.4 item S-2, §6.3 row C3, §6.4 item C-3.
2. `scripts/check_file_size_ratchet.py` and
   `scripts/file_size_ratchet_baseline.json`.
3. `scripts/check_zarr_open_group_modes.py` and
   `scripts/zarr_open_group_mode_ratchet_baseline.json`, as the model for a
   new count ratchet (CI job `zarr-open-mode-ratchet` in `ci.yml`).
4. `.github/workflows/ci.yml`: the `file-size-ratchet` and
   `zarr-open-mode-ratchet` jobs, including the `git diff --exit-code` step
   that pins each baseline.
5. `pyproject.toml` `[tool.importlinter]` section, to see whether a
   forbidden-import contract is the better mechanism for S-2 than a script.

### Verify before implementing

- Reproduce the counts: modules over 1,500 lines under `src/fisheye`
  (review found 153), and direct `zarr.open_group` / `zarr.open` /
  `zarr.open_consolidated` call sites outside `fisheye.shared.zarr_io`
  (review found 839 across 352 files, including 219 with no consolidated
  flag). Put your commands and numbers in the handoff.
- Confirm the file-size ratchet's tolerance (`+200`) and that it currently
  guards exactly four files.

### Implement

1. Regenerate `scripts/file_size_ratchet_baseline.json` to include every
   module over 1,500 lines with its current line count. Keep the tolerance.
   Do not change the script's semantics.
2. Add `scripts/check_zarr_open_site_ratchet.py` mirroring the existing
   ratchet style: count bare zarr open call sites outside
   `src/fisheye/shared/zarr_io.py`, compare to a committed baseline JSON
   (total count and per-file counts), fail if any file's count rises or a
   new file appears. Include a `--write-baseline` flag.
3. Wire it into CI as its own job mirroring `zarr-open-mode-ratchet`,
   including the `git diff --exit-code` baseline pin.
4. Do **not** convert any call sites. This package establishes the ceiling
   only.
5. Tests: a small fixture tree where the script passes at baseline, fails on
   a new site, and passes when a site is removed.

### Acceptance

- Both ratchet scripts pass on the branch at baseline. Full CI green.
- Handoff lists the ten files with the highest bare-open counts, as the
  starting point for the S-3 codemod package.

### Do not

- Do not edit import-linter contracts in this package unless the verify step
  shows the script approach is unworkable; if so, stop and hand off with the
  reason.
- Do not delete files, even ones the subtraction queue marks dead.

---

## Handoff format (all packages)

Write `HANDOFF_<package>_<date>.md` in the worktree root containing: exact
commit; branch name; CI run URL and per-check status; "Verified premises"
table; the package-specific tables named above; tests run with the exact
command and result; anything skipped or deferred and why; the
`authority_consolidation_work_queue` row you added. Per `AGENTS.md`, a
handoff with failing or unrun required checks is explicitly incomplete work.
