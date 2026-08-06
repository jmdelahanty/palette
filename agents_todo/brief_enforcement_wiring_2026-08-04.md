# Brief: enforcement wiring (`agent/palette/enforcement-wiring`)

**From:** commander session, 2026-08-04
**Status: READY.** One agent, five stages **in order**, **mandatory CHECKPOINT
after each stage.** Stages 1-2 are the whole value; 3-5 are optional if time runs
short.
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/diagnostics/provenance_chain_audit_2026-07-24.md` (Verdict +
Root Cause 1 + Fix). Ground rules: local `sun` is ground truth; fresh worktree
from CURRENT `sun`; env `~/miniconda3/envs/palette-py311/bin/python` (conda;
never uv, never a `.venv`). Gates before every checkpoint: import-linter via
`scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`,
full suite.

---

## The problem this brief fixes

`scripts/` contains **13 gate scripts. Exactly one is wired into
`.github/workflows/ci.yml`** (`check_file_size_ratchet.py`).

```
check_file_size_ratchet.py                    in CI: 1
check_contract_freshness.py                   in CI: 0   <-- validates 139 docs' headers
check_labeling_production_decision_record.py  in CI: 0   <-- a decision-record gate
check_detect_compute_smoke.py                 in CI: 0
check_legacy_source_dataset_ids.py            in CI: 0
check_mask_probabilities.py / _values.py      in CI: 0
check_keypoint_confidences.py                 in CI: 0
check_detect_{export_inputs,decode_backend_parity}.py  in CI: 0
check_labeling_web_{readiness,static,unit}.sh in CI: 0
```

The mechanisms were all built and none were connected. **This brief is wiring,
not authoring.** See the anti-goals.

### The one rule that governs every stage

**A retroactive gate must ship as a ratchet, never as a hard pass/fail.** A gate
that reds the build on day one gets commented out within a week. A ratchet
baselines today's violation count and fails only on *increase* — nothing has to
be fixed up front, and the number drifts down as people touch things.

`scripts/check_file_size_ratchet.py` is the in-repo model: JSON baseline dict,
per-entry threshold, fails on growth, and *reports* when a value tightens so the
baseline can be lowered. **Copy that shape. Do not invent a second ratchet
idiom.**

---

## Stage 1 — Ratchet + wire `check_contract_freshness.py`. **CHECKPOINT.**

Highest value in the brief. 139 docs already carry the `<!-- contract-meta -->`
header this validates, and nothing enforces it.

**Measured current state** (`python scripts/check_contract_freshness.py --json`,
2026-08-04): **exit 1, 103 issues** —

| code | count |
|---|---:|
| `missing_meta` | 57 |
| `missing_field` | 24 |
| `stale` | 12 |
| `invalid_status` | 10 |

Do:

1. Add `--baseline <path>` and `--update-baseline` to
   `scripts/check_contract_freshness.py`, mirroring `check_file_size_ratchet.py`
   (`_load_baseline` / `_write_baseline` / threshold / tightened-report).
   Baseline granularity is **per issue `code`**, not per file — per-file churns
   on every doc rename.
2. Commit `scripts/contract_freshness_baseline.json` with today's counts above.
3. Exit non-zero only when a count **exceeds** its baseline. Emit the tightened
   report when a count drops.
4. Add a CI step to `.github/workflows/ci.yml` in the existing
   **"package, boundaries, and collection"** job, immediately after the
   `Check file-size ratchet` step. Same shape as that step.
5. Do **not** fix the 103 issues. That is a separate wave.

Verify: the new step passes on `sun` as-is; then hand-add a doc with a bad
`status:` value locally and confirm CI-equivalent failure; then revert it.

**Note for the follow-up wave, not this one:** the gate's `VALID_STATUSES` is
`{active, draft, superseded}`. Earlier commander guidance suggested
`implemented | partial | specified-only`. **The existing vocabulary wins** —
extend it if needed, do not introduce a parallel one.

---

## Stage 2 — Unreachable-module ratchet. **CHECKPOINT.**

The direct counter-pressure to the sprawl: `src/` gained 227 files and lost 3 in
the ten days after the audit, and ~116 new modules have no production entry
point.

This is the **only new script** the brief authorizes. Write
`scripts/check_module_reachability_ratchet.py`:

1. Build the import graph with `ast` over `src/`, `scripts/`, `tools/`, `apps/`
   (imports only — exclude `tests/`).
2. **Also resolve string-dispatch edges.** This repo launches bsub jobs by
   module-name string; a pure import graph produces false orphans. Grep
   `*.sh`, `*.yaml`, `*.yml`, `*.toml`, `*.json` under `scripts/`, `tools/`,
   `configs/`, plus `pyproject.toml` and `runnables.yaml`, for each module's
   dotted path and basename. **`cluster/arena_geometry.py` is the regression
   test for this** — zero Python importers, referenced only by
   `scripts/submit_arena_geometry_detection_gate_audit_bsub.sh`. If your tool
   reports it unreachable, the tool is wrong.
3. **Exempt modules with a CLI entry point** (`def main(`, `if __name__ ==
   "__main__"`, `argparse`/`ArgumentParser`). 45 of 45 new `utils/` files and 26
   of 27 new `diagnostics/` files are human-invoked `python -m` tools;
   unreachability is correct for them. Report them in a separate,
   non-failing bucket.
4. Roots: `[project.scripts]` in `pyproject.toml`, `runnables.yaml`, and every
   `-m fisheye.X` in `scripts/`.
5. Baseline the current count per top-level package
   (`shared/zarr`, `cluster`, `analysis_workflows`, …). Fail on increase.
6. Wire into the same CI job.

Include the counts you measure in the checkpoint report. Expect roughly 40-45
genuine library orphans concentrated in `shared/zarr/`; if you get a number near
116 you have not implemented step 3.

---

## Stage 3 — Provenance smoke as a required check

`scripts/check_detect_compute_smoke.py` already exists and is unwired. **Read it
first** and decide, in the checkpoint report, whether it can be extended or
whether a sibling is genuinely needed. Prefer extending.

The assertion set is Stage 1 of
`docs/archive/detect_provenance_activation_brief_2026-07-24.md` — run detect on a
tiny fixture and assert the provenance chain closes: `run_provenance` present;
`input_artifacts` carries `role == "detect_model"` with 64-hex `sha256`; digest
matches the registry; `inference_precision` recorded. **Read every attr with
`use_consolidated=False`** (Root Cause 1).

If it cannot run in CI without a GPU or real recording data, say so plainly in
the checkpoint and stop — do not build a fixture pipeline inside this brief. A
correct "this needs a GPU runner, here is what it would take" is a successful
outcome for this stage.

---

## Stage 4 — Digest-helper duplication gate

Prevent regrowth of what the subtraction wave collapses. Current counts:
8 private `_array_digest`, 4 `_payload_sha256`, 38 `_canonical_json`, and
`sha256_c_contiguous_bytes_v1` declared verbatim in 9 modules — against
`shared/zarr/manifest_digest.py`, which already has **109 importers** and is the
winning implementation.

Implement as a **count ratchet in the Stage 2 script or a sibling** — baseline
today's numbers, fail on increase. Do not write a bespoke third ratchet idiom.

Prefer an `import-linter` contract if one can express it (there are currently 2
contracts in `pyproject.toml`); fall back to a grep ratchet if not.

---

## Stage 5 — ADR-0001 and the agent-facing error convention

**5a — Create `docs/adr/` and transcribe ADR-0001: consolidated metadata.**
Content already exists — the "Fix" section of
`docs/diagnostics/provenance_chain_audit_2026-07-24.md` (measurements: 8.6×
slower on NVMe; ~660 ms walk cost; the three structural reasons per-stage
reconsolidation cannot work; the sealed-store end state). **Transcribe, do not
re-reason.** Use Nygard format: numbered, immutable, superseded-rather-than-edited.

Then add a **Status: PROPOSED — commander ratifies** line. This is a live
contradiction: four modules currently *raise* when the consolidated snapshot is
absent (`shared/zarr/refined_detection_snapshot.py:174`,
`subject_mask_bundle_publication.py:498`, `subject_mask_cache_publication.py:534`,
`subject_mask_core_publication.py:166`), wired into production validate/finalize
stages, with four contract docs specifying reconsolidation as required.
**Change no code in either direction.** Record both positions in the ADR and stop.

**5b — Add the error-message convention to `AGENTS.md`.** `AGENTS.md` is loaded
every session and is the only reliable channel to a fleet. Add a short section:

> When a check enforces an architectural decision, the failure message must name
> the decision and cite the ADR path. Agents read assertion output with far
> higher reliability than they read documentation, so the error message *is* the
> documentation.
>
> ```
> AssertionError: consolidate_metadata() called on a live archive.
>   Ruled out by docs/adr/0001-consolidated-metadata.md
>   (8.6x slower measured; walk-then-write is not atomic).
>   Consolidation is permitted only on sealed stores at cohort release.
> ```

Apply that message style to every gate added in Stages 1-4.

---

## Anti-goals — state these back in the checkpoint

1. **Do not fix the violations the gates find.** 103 contract-freshness issues and
   ~40 orphan modules stay exactly as they are. This brief installs the ratchets;
   other waves lower them. Mixing the two makes the diff unreviewable.
2. **Do not author new gate scripts beyond Stage 2's one file.** The repo has 13
   gates and 1 wired. The deficit is wiring, not authoring. If you find yourself
   writing a fourteenth, stop and report instead.
3. **Do not hard-gate anything retroactively.** Every check added here ships as a
   ratchet. No exceptions.
4. **Do not change `src/fisheye/**` in this brief** except where Stage 4's
   import-linter contract requires an import rewrite. This is CI and tooling
   work. Net `src/` line delta should be ~0; report it.
5. **Do not touch either side of the consolidated-metadata contradiction.**
   Stage 5a documents it. The commander decides.
