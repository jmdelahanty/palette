# AGENTS

## Python Environment Rule

- Use `scripts/py` for all Python commands in this repository.
- Do not run `conda activate` in this repository.
- Prefer `scripts/py -m <module>` over bare `python -m <module>`.
- `scripts/py` is expected to resolve to the `palette-py311` conda environment; verify with `scripts/py -c 'import sys; print(sys.executable)'` when needed.
- Do not run install or dependency mutation commands unless the user explicitly approves in chat first.
- Blocked without approval: `pip install`, `conda install`, `mamba install`, `poetry add`, `uv pip install`.

## Git Push Rule

- Pushes from this repository require the Palette workstation SSH key and should run outside the Codex sandbox because sandbox DNS/network access can fail.
- When a workflow intentionally uses the single shared `/groups` checkout,
  prefer the tracked helper, which pushes the current branch and then
  fast-forwards that checkout:
  `scripts/push_and_update_groups_checkout.sh`
- When concurrent agents or cluster jobs need different Palette commits, do
  not switch or fast-forward the shared checkout. Use
  `scripts/deploy_palette_cluster_worktree.sh` from the clean source worktree.
  It pushes only that branch, creates a detached commit-pinned worktree below
  the shared deployment root, leaves the shared checkout unchanged, and prints
  the exact `PALETTE_GROUPS_REPO` value for submission.
- Cluster submissions from a dedicated deployment must record and pass its
  absolute `--palette-repo` path and full commit. Never point an already-planned
  job at a mutable shared checkout or move a deployment path to a newer commit.
- If you intentionally need to push without updating `/groups`, use:
  `GIT_SSH_COMMAND='ssh -i /home/delahantyj@hhmi.org/.ssh/delahantyj-ws1-git-id_ed25519 -o IdentitiesOnly=yes' git -C /home/delahantyj@hhmi.org/gitrepos/palette push`
- Do not rely on plain `git push` for Palette; it may fail with `Permission denied (publickey)` or sandbox DNS errors.
- The `/groups` fast-forward must fail closed: do not use merge commits, rebase, reset, or dirty-checkout workarounds unless the user explicitly approves.

## Sandbox Zarr Fallback Rule

- If sync `zarr.open_group(...)` hangs in Codex sandbox, use metadata-file checks from `docs/sandbox_zarr_fallback.md`.
- For keypoint review status checks, prefer `zarr.json` + `jq` fallback over Python `zarr` reads when sandbox hangs are observed.

## Consolidated Metadata Read Policy

- Treat metadata mode as a lifecycle decision, not a universal reader option.
- Writers, edit tools, and readers inspecting an actively mutable or incomplete
  Palette Zarr must use `use_consolidated=False` so newly created groups are
  visible during mutation.
- For a selector-visible immutable publication, finish and validate all payload,
  attrs, and provenance writes; update the direct `latest`/manifest selection
  metadata; then consolidate the root as the final published visibility step.
  Validate that the consolidated generation contains the intended selector
  state before declaring publication complete.
- New readers of published immutable artifacts should use consolidated metadata
  by default and validate the published metadata generation/schema contract.
- Missing or stale consolidated metadata on a published immutable artifact is a
  publication defect. Do not silently normalize that state by making
  unconsolidated traversal the permanent reader default.
- Diagnostics and benchmarks that compare metadata paths must select
  consolidated or unconsolidated mode explicitly and record the selected mode
  in their result.

## Sandbox Zarr Test Policy

- Run pytest-based validation outside the Codex sandbox by default for this repository; tests run normally there and sandbox execution can hang on zarr paths.
- Use `scripts/py -m pytest ...` with an outside-sandbox/escalated command when running focused or full test suites.
- Keep in-sandbox validation to static/non-zarr checks such as `scripts/py -m py_compile`, `git diff --check`, or explicitly safe in-memory tests.
- In Codex sandbox, prefer in-memory or fake-group test harnesses (with monkeypatch) for zarr-related unit tests.
- Do not rely on sync real-zarr integration tests in sandbox when equivalent logic can be validated with in-memory tests.
- If a real-zarr test hangs or is known to hang in sandbox:
  - stop that test path,
  - run non-hanging validation (`scripts/py -m py_compile` and relevant fast unit tests),
  - if the real-zarr test is important to the current change, rerun that exact focused test outside the sandbox with escalation,
  - if outside-sandbox execution is unavailable or still fails, report the skipped test as deferred local validation.
- For deferred local validation, provide exact commands for the user to run in their terminal.
- For new zarr-heavy tests, default to deterministic in-memory coverage first; add real-zarr integration checks only when required and mark them for local execution if sandbox stability is an issue.

## Dask / Parallel Zarr Write Rule

- Parallel writes to Zarr are safe only when each worker owns whole, non-overlapping physical Zarr chunks for every array it writes.
- Do not assume disjoint logical row slices are safe. If two workers write different row ranges inside the same physical chunk, Zarr chunk-level read-modify-write behavior can cause stale overwrites.
- When adding or changing Dask writes, align worker chunks to the physical chunk grid of the written arrays, serialize writes that cannot be chunk-aligned, or write per-worker temporary outputs and merge deterministically.
- Record both requested and effective worker chunking in provenance when Dask chunk sizes are adjusted for Zarr write safety.
- See `docs/dask_zarr_write_safety.md` before modifying Dask-backed Zarr writers.

## Subject Mask Direction

- Treat eye-mask-specific stages as legacy compatibility surfaces.
- For new mask work, prefer `subject_mask_runs` and `refined_subject_masks_runs`.
- For modern editable refined subject-mask outputs, dense `masks_roi` is the
  authoritative pixel surface and must be physically present. Compact
  `mask_bitpacked` and `mask_rle` stores are derived display/archive caches, not
  edit or training authorities.
- Review/edit paths should mutate dense `masks_roi` only. After accepted dense
  edits, mark derived bitpacked/RLE/metrics/contours stale and refresh them only
  during explicit validation, promotion, or maintenance steps.
- Historical compact-only refined subject-mask runs may be read through
  `MaskStore` for compatibility, but must be materialized to dense `masks_roi`
  before review/editing or training export.
- Do not add new workflows, docs, or model paths that make `eye_masks_runs` or `refined_eye_masks_runs` the primary source of truth.
- It is acceptable to read, migrate, validate, or materialize eye-mask compatibility data when supporting historical archives or legacy consumers.
- Canonical manual review/editing for body, swim bladder, and eye components should route through unified refined subject-mask tooling and component review state.
- Training/export paths should prefer subject-mask contracts; eye-mask training artifacts remain legacy or compatibility-specific unless the user explicitly asks for them.
- Subject-mask training zarrs are dense `uint8` export artifacts: compact analysis sources (`mask_bitpacked` or `mask_rle`) must be materialized through `MaskStore` into dense `subject_mask_runs/<run>/masks_roi`, with source encoding recorded in training provenance.

## Outside-Sandbox Validation Notes

- CUDA/GPU visibility may be unavailable in Codex sandbox even when available outside it.
- Run `scripts/py -m marimo check ...` outside the Codex sandbox by default. In the sandbox, marimo's checker can hang before diagnostics because its async filesystem path uses `asyncio.to_thread`; outside-sandbox execution has been observed to return normally.
- For CUDA checks, run outside the sandbox with `scripts/py` rather than `conda activate`, for example:
  `scripts/py -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'`
- Real-zarr training/export/inference smokes should run outside the sandbox with escalation, especially when they use CUDA or touch `/nvme1`.
- For U-Net subject-mask smoke validation, prefer the CUDA-capable outside-sandbox path over CPU sandbox execution. If the sandbox prints the startup banner but does not reach the artifact summary promptly, stop it and rerun outside the sandbox.
- When newly written Zarr groups are hidden during an active mutation, follow
  the Consolidated Metadata Read Policy above; do not use a stale consolidated
  view of a mutable archive.

## Examples

- `scripts/py --version`
- `scripts/py -m pytest`
- `scripts/py src/test_fisheye.py`
