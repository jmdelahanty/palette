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

## Examples

- `scripts/py --version`
- `scripts/py -m pytest`
- `scripts/py src/test_fisheye.py`
