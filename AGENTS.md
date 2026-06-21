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
- Use:
  `GIT_SSH_COMMAND='ssh -i /home/delahantyj@hhmi.org/.ssh/delahantyj-ws1-git-id_ed25519 -o IdentitiesOnly=yes' git -C /home/delahantyj@hhmi.org/gitrepos/palette push`
- Do not rely on plain `git push` for Palette; it may fail with `Permission denied (publickey)` or sandbox DNS errors.

## Sandbox Zarr Fallback Rule

- If sync `zarr.open_group(...)` hangs in Codex sandbox, use metadata-file checks from `docs/sandbox_zarr_fallback.md`.
- For keypoint review status checks, prefer `zarr.json` + `jq` fallback over Python `zarr` reads when sandbox hangs are observed.

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
- For refined subject-mask pixels, treat the logical store as `MaskStore` over
  dense `masks_roi`, compact editable `mask_bitpacked`, or compact final/read-mostly
  `mask_rle`; do not assume `masks_roi` is always physically present.
- Review/edit paths should materialize or mutate dense `masks_roi`, refresh
  touched `mask_bitpacked` rows/components when present, and mark compact RLE
  stale until explicitly refreshed.
- Do not add new workflows, docs, or model paths that make `eye_masks_runs` or `refined_eye_masks_runs` the primary source of truth.
- It is acceptable to read, migrate, validate, or materialize eye-mask compatibility data when supporting historical archives or legacy consumers.
- Canonical manual review/editing for body, swim bladder, and eye components should route through unified refined subject-mask tooling and component review state.
- Training/export paths should prefer subject-mask contracts; eye-mask training artifacts remain legacy or compatibility-specific unless the user explicitly asks for them.

## Outside-Sandbox Validation Notes

- CUDA/GPU visibility may be unavailable in Codex sandbox even when available outside it.
- Run `scripts/py -m marimo check ...` outside the Codex sandbox by default. In the sandbox, marimo's checker can hang before diagnostics because its async filesystem path uses `asyncio.to_thread`; outside-sandbox execution has been observed to return normally.
- For CUDA checks, run outside the sandbox with `scripts/py` rather than `conda activate`, for example:
  `scripts/py -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'`
- Real-zarr training/export/inference smokes should run outside the sandbox with escalation, especially when they use CUDA or touch `/nvme1`.
- For U-Net subject-mask smoke validation, prefer the CUDA-capable outside-sandbox path over CPU sandbox execution. If the sandbox prints the startup banner but does not reach the artifact summary promptly, stop it and rerun outside the sandbox.
- When newly written zarr groups are hidden by stale consolidated metadata, open mutable Palette zarrs with `use_consolidated=False`.

## Examples

- `scripts/py --version`
- `scripts/py -m pytest`
- `scripts/py src/test_fisheye.py`
