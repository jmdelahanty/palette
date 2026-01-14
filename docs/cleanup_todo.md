# Cleanup TODO

## Decide Artifact Policy
- Decide whether model weights, wheels, zips, and plots should live in git, Git LFS, or external storage.
- If using Git LFS, define which extensions are LFS tracked (e.g., .pt, .whl, .zip, .png).
- Document the policy in README or a short CONTRIBUTING note.

## Remove/Relocate Tracked Artifacts
- Inventory large binaries and generated outputs currently tracked (examples: .pt weights, .whl, .zip, .png, runs/).
- Remove tracked artifacts from git history or move them to LFS/external storage.
- Add a small sample/placeholder data set for tests if needed.

## Test Layout and Discovery
- Decide on a single test location (prefer `tests/` or move package tests under `tests/`).
- If keeping tests outside `tests/`, update `pytest.ini` to include them.
- Add or update test markers for GPU/slow tests to prevent accidental CI failures.

## Packaging and Dependencies
- Define runtime and optional dependencies (e.g., extras for GPU, dev tools).
- Add console entry points if CLI is intended to be installed.
- Consider moving to `pyproject.toml` for modern packaging (or expand `setup.py`).

## Decord Source of Truth
- Decide whether to vendor `decord/` or rely on a wheel or external dependency.
- Remove the unused path to reduce confusion.
- Document any build/setup steps if vendoring is required.

## Repo Hygiene
- Remove tracked `__pycache__`, logs, and other build artifacts.
- Ensure `.gitignore` covers all generated directories and logs in actual use.
- Add a brief section in README for reproducible setup and data locations.
