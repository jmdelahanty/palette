# AGENTS

## Python Environment Rule

- Use `scripts/py` for all Python commands in this repository.
- Do not run `conda activate` in this repository.
- Prefer `scripts/py -m <module>` over bare `python -m <module>`.
- Do not run install or dependency mutation commands unless the user explicitly approves in chat first.
- Blocked without approval: `pip install`, `conda install`, `mamba install`, `poetry add`, `uv pip install`.

## Sandbox Zarr Fallback Rule

- If sync `zarr.open_group(...)` hangs in Codex sandbox, use metadata-file checks from `docs/sandbox_zarr_fallback.md`.
- For keypoint review status checks, prefer `zarr.json` + `jq` fallback over Python `zarr` reads when sandbox hangs are observed.

## Sandbox Zarr Test Policy

- In Codex sandbox, prefer in-memory or fake-group test harnesses (with monkeypatch) for zarr-related unit tests.
- Do not rely on sync real-zarr integration tests in sandbox when equivalent logic can be validated with in-memory tests.
- If a real-zarr test hangs or is known to hang in sandbox:
  - stop that test path,
  - run non-hanging validation (`scripts/py -m py_compile` and relevant fast unit tests),
  - report the skipped test as deferred local validation.
- For deferred local validation, provide exact commands for the user to run in their terminal.
- For new zarr-heavy tests, default to deterministic in-memory coverage first; add real-zarr integration checks only when required and mark them for local execution if sandbox stability is an issue.

## Examples

- `scripts/py --version`
- `scripts/py -m pytest`
- `scripts/py src/test_fisheye.py`
