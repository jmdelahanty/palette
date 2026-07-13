# Palette

[![CI](https://github.com/jmdelahanty/palette/actions/workflows/ci.yml/badge.svg?branch=sun)](https://github.com/jmdelahanty/palette/actions/workflows/ci.yml)

Palette is the FishEye/Crimson data-processing repository for zebrafish behavior
recordings. It imports acquisition and stimulus data, runs detection/keypoint/
segmentation stages, records Zarr provenance, exports training datasets, and
provides analysis and review tooling for downstream scientific workflows.

The original preprocessing scripts were started from work by Dr. Jinyao Yan and
Dr. Ratan Othayoth. Current development centers on the `fisheye` Python package
under `src/`.

## Environment

Use the repository wrapper for Python commands:

```bash
scripts/py --version
scripts/py -c 'import sys; print(sys.executable)'
```

`scripts/py` should resolve to the `palette-py311` conda environment. Do not
create a local `.venv` for normal development in this repository.

For editable development installs, use the existing environment:

```bash
scripts/py -m pip install -e ".[dev]"
```

GPU workflows use workstation-specific CUDA/PyTorch setup. See
`environment.yml`, the operator docs, and the relevant stage scripts before
changing GPU dependencies.

## Command-Line Entry Point

The primary command surface is:

```bash
palette status <recording-or-zarr>
palette plan <recording-or-zarr>
palette detect <recording-or-zarr> --dry-run
palette crop <recording-or-zarr> --dry-run
palette keypoints <recording-or-zarr> --dry-run
palette approve <recording-or-zarr> <stage> <run> --dry-run
```

Run `palette --help` for the current command list. For interactive recording
inspection, use the marimo launcher:

```bash
scripts/run_palette_explorer.sh /path/to/analysis.zarr
```

For read-only exploration of an exported cross-recording analytics dataset,
use the locked lightweight Pixi application:

```bash
pixi run app -- --export-root /groups/johnson/johnsonlab/palette_analytics
```

## Documentation

Useful starting points:

- `AGENTS.md` - repository rules for coding agents and local validation.
- `docs/operator_guide/pipeline_workflow.md` - current processing workflow.
- `docs/palette_cli_narrow_waist_design.md` - CLI/API contract.
- `docs/run_resolution_semantics.md` - run-selection semantics.
- `docs/provenance_finalization_enforcement_design.md` - completion and provenance gate.
- `docs/marimo_explorer_architecture.md` - interactive viewer structure.
- `docs/palette_analytics_app_deployment.md` - Pixi and FileGlancer deployment.

Historical investigations and one-off plans are under `docs/archive/` and
`docs/diagnostics/`.

## Validation

Run focused tests during development, then the non-GPU suite before merging:

```bash
PYTHONPATH=src scripts/py -m pytest -q tests/unit/fisheye/test_zarr_run_completion.py
PYTHONPATH=src scripts/py -m pytest -q -m "not gpu" tests/
```

Some Zarr-heavy or GPU-related tests should run outside the Codex sandbox; see
`AGENTS.md` for the local validation policy.

## License

No repository LICENSE file is currently declared. Licensing and public-sharing
terms remain a maintainer/HHMI decision.
