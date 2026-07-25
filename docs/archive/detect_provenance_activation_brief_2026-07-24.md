<!-- ARCHIVED 2026-07-24: completed and superseded by docs/detection_publication_contract.md. Historical filesystem census claims below were not adopted as current repository truth. -->

# Brief: detect provenance activation (`agent/detect-provenance`)

**From:** commander session, 2026-07-24
**Status: COMPLETED AND SUPERSEDED.** This is the historical execution brief;
do not use its branch, environment, or operator commands as a current runbook.
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/diagnostics/provenance_chain_audit_2026-07-24.md` — Root
Cause 1, Lane A, and Lane F. Ground rules: local `sun` is ground truth; fresh
worktree on `agent/detect-provenance` from CURRENT `sun`; env
`~/miniconda3/envs/palette-py311/bin/python` (conda; never uv, never a `.venv`).
Local gates before every checkpoint: import-linter via
`scripts/py -m importlinter.cli --config pyproject.toml`,
`python scripts/check_file_size_ratchet.py`, `git diff --check`, `py_compile`,
focused tests, then the full suite.

---

## Why this brief exists

**The detect provenance code is written, correct, and has never executed.**

- `detection/detect_yolo.py:3666-3669` writes `run_provenance` **unconditionally**
  (not gated on `palette_completion_epoch`), with the model artifact appended via
  `append_input_artifacts` and fingerprinted at `:2208` by `fingerprint_artifact`.
- That code landed **2026-07-05** in `f557b998` ("provenance: record model
  artifacts in run provenance").
- The **newest detect run on disk is `detect_2026-02-09_14-28-54`** — five months
  older.
- Census: **474** detect run directories exist under `/nvme1/recordings`;
  `grep -rl --include=zarr.json 'run_provenance' /nvme1/recordings` returns **0**.

So the audit's stated cause for the missing records ("the completion gate never
fires") is **wrong for detect**. The simpler truth: no detect run has been
produced since the provenance work landed. The chain has never been exercised
end to end, and nobody knows whether it closes.

This also blocks in-flight work: `utils/run_detection_local_publish.py:237-247`
(untracked) verifies that a run's `run_provenance.input_artifacts` contains a
`detect_model` entry whose `sha256` matches a pinned digest, and fails the
publish otherwise. That verifier has no run on disk it could pass against.

---

## Stage 1 — Exercise the chain once, end to end. **CHECKPOINT.**

**This is the whole point of the brief. Do not skip to the code changes.**

Produce **one** detect run on **one** recording with current `sun` code, then
assert the chain closes. Write the result to
`docs/diagnostics/detect_provenance_smoke_2026-07-24.md`.

Assertions, in order — report each as PASS/FAIL with the observed value:

1. `run_provenance` attr exists on the new detect run group.
   **Read it with `zarr.open_group(<run_path>, use_consolidated=False)`** — see
   Root Cause 1; a read through the store root may serve a stale snapshot.
2. `run_provenance["input_artifacts"]` contains an entry with
   `role == "detect_model"` and a 64-hex `sha256`.
3. That `sha256` equals the registry's `model_sha256` for the resolved model.
   Note whether `fingerprint_artifact` **computed** the digest or returned a
   cached `<weights>.content_v1.json` sidecar — `shared/artifact_fingerprint.py:150-156`
   short-circuits on `(size_bytes, mtime_ns)` match and **skips the registry
   cross-check entirely** on that path. If it took the sidecar path, delete the
   sidecar and re-run to get a genuinely computed digest.
4. `run_provenance["git_dirty"]` — record the value. If `true`, note that the run
   is not reproducible from the SHA alone.
5. The verifier at `utils/run_detection_local_publish.py:237-247` passes against
   this run.
6. Re-read the same attrs **through the store root** (default `open_group`). If
   the store carries `consolidated_metadata`, confirm whether `run_provenance` is
   visible that way. Report the answer — it determines whether every consumer
   needs `use_consolidated=False`.

**Checkpoint here.** If assertions 1-3 fail, stop and report; the remaining
stages assume the writer works.

---

## Stage 2 — Record inference precision (confirmed gap, highest value)

`detection/detect_yolo.py:2248-2278` — verified 2026-07-24:

```python
model.fuse()                                          # :2248  layer fusion, unrecorded
torch.backends.cudnn.benchmark = True                 # :2262  unrecorded
model.model = model.model.to(memory_format=torch.channels_last)  # :2264  unrecorded
model.half()                                          # :2265  unrecorded
model_fp16 = True                                     # :2265
...
"half": model_fp16,                                   # :2278  used, never persisted
```

`model_fp16` is consumed at `:2278`, `:2889`, `:3002` and **written to no attr**.
`grep -n "precision"` in this file returns zero hits.

Consequence: the same video and the same weights, run on a GPU node vs a CPU
node, produce numerically different boxes with byte-identical provenance.

**Do:** add these to the `parameters` block at `:3478` (which is otherwise
already rich — it carries imgsz, resize dims, decode backend, and pynvvc surface
mode; precision is the notable hole, not general sparsity):

- `inference_precision`: `"fp16"` / `"fp32"`
- `inference_device`: `"cuda"` / `"cpu"`
- `torch_cudnn_benchmark`: bool
- `torch_memory_format`: `"channels_last"` / `"contiguous"`
- `model_fused`: bool (whether `model.fuse()` succeeded at `:2248`)

Add a regression test asserting these are present and that
`inference_precision == "fp16"` implies `parameters["half"] is True`.

---

## Stage 3 — Flat `model_sha256`, and close two small recording holes

**3a — promote the digest.** `detection_attrs` at `:3446-3456` carries
`model_path` and `model_name` but no digest; the digest exists only nested at
`run_provenance.input_artifacts[].sha256`. Add `model_sha256` (and
`model_fingerprint_scheme`) as flat attrs alongside `model_path`. A human or a
SQL extractor reading `detect_runs/<run>` should not have to traverse a nested
provenance payload to learn which weights produced the boxes.

**3b — the `created_new_root` gate.** `:2481` gates the environment capture
block on `if created_new_root:`, so a detect run written into an **existing**
zarr captures no environment. Ungate it.

**3c — naive datetime mislabelled UTC.** `refinement/detect_quality.py:705`
writes `'created_at_utc': datetime.now().isoformat()` — machine-local time in a
field named UTC, inside `provenance`. Every other writer uses
`datetime.now(timezone.utc)`. Fix and add a test.

---

## Stage 4 — An explicit behaviour discriminator

`b14dc8e3` ("detect: sanitize backend boxes at image bounds") changed detection
geometry **with no parameter change**. The only discriminator between pre- and
post-sanitize output is the git SHA — and `git_dirty` is `true` for ~75% of
sampled runs repo-wide, so two runs on the same SHA with different uncommitted
edits are indistinguishable.

**Do:** add a module-level `DETECT_ALGORITHM_VERSION` integer to
`detection/detect_yolo.py`, record it in the `parameters` block, and document in
a docstring that it **must be bumped on any change to detection geometry,
filtering, or box post-processing that is not expressible as a parameter**. Set
it to `2` (1 = pre-`b14dc8e3`, 2 = with bounds sanitization).

This is the poor-man's version of Nextflow hashing the script text (see
`docs/workflow_provenance_prior_art.md`). It is not automatic and it depends on
discipline — but it is strictly better than a git SHA that is dirty three runs
out of four.

---

## Explicitly OUT of scope

Do not take these on in this brief; they are repo-wide decisions:

- **The `consolidated_metadata` fix.** Repo-wide (430 `open_group` sites).
  This brief only *reads* with `use_consolidated=False`; it does not change the
  policy.
- **Content-addressed run names.** `_next_run_name` (`:543-551`) is a
  timestamp + check-then-create collision loop — a genuine TOCTOU, and the right
  fix is a content digest. But the naming scheme must be repo-wide, and **detect
  is the correct pilot** precisely because its inputs (video fingerprint, model
  sha256, resolved params, algorithm version) are all knowable up front and do
  not depend on an upstream run. Stages 2-4 exist partly to make those inputs
  complete enough that the digest can be defined. Flag readiness in the
  checkpoint; do not implement.
- Registry authority, dedupe, or reconcile changes.

## Corrections to the audit to fold back

Both belong in `docs/diagnostics/provenance_chain_audit_2026-07-24.md` once
Stage 1 lands:

1. Lane A characterizes detect's recorded `parameters` as a thin subset. It is
   not — `:3478` records ~15 keys covering imgsz, resize, decode backend, and
   surface materialization. **Precision is the specific hole.**
2. Root Cause 1 attributes the 0/4052 `run_provenance` census to the completion
   gate never firing. For detect that is wrong: the writer is unconditional
   (`:3666-3669`) and simply postdates every run on disk.
