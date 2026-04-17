# Shared Test Data

This guide defines the shared real-data fixture area used for Palette smoke
checks, diagnostics runs, and operator verification without touching the main
`/nvme1/recordings` tree.

## Root convention

Use these paths consistently:

```bash
export PALETTE_TEST_DATA_ROOT=/nvme1/palette_test_data
export PALETTE_TEST_FIXTURES_ROOT="$PALETTE_TEST_DATA_ROOT/fixtures"
export PALETTE_TEST_RUNS_ROOT="$PALETTE_TEST_DATA_ROOT/runs"
```

Recommended meaning:

- `PALETTE_TEST_DATA_ROOT`: shared top-level test-data area on the NVMe volume
- `PALETTE_TEST_FIXTURES_ROOT`: stable read-only inputs copied from real recordings
- `PALETTE_TEST_RUNS_ROOT`: writable outputs from smoke tests, diagnostics, and experiments

## Layout

The shared layout is:

```text
/nvme1/palette_test_data/
  fixtures/
    recordings/
      <recording>/
  runs/
    <tool-or-workflow>/
      <date_or_label>/
```

Conventions:

- Treat `fixtures/` as read-only.
- Put generated outputs only under `runs/`.
- Group outputs first by tool or workflow, then by date or human-readable label.
- If a test needs to mutate an input recording, copy the fixture into a fresh run
  directory first instead of editing the shared fixture in place.

## Current shared fixture

Current curated recording fixture:

- `/nvme1/palette_test_data/fixtures/recordings/2026-01-28T19-36-18Z_arena_1_Feeding`

This fixture is a curated copy of an organized recording. It keeps the canonical
files needed for organize/import/diagnostics work while excluding backup and
repair variants such as `.bak*` and `*_fixed.mp4`.

Included artifacts:

- `cams/*.mp4`
- `cams/*_meta.csv`
- `raw/*.h5`
- `raw/*.mp4`
- `raw/*_update_timing.csv`
- `derived/recording_snapshot.json`
- `recording_manifest.json`

## Typical usage

### Run diagnostics against the shared fixture

```bash
scripts/py -m fisheye.diagnostics.video batch \
  "$PALETTE_TEST_FIXTURES_ROOT/recordings/2026-01-28T19-36-18Z_arena_1_Feeding"

scripts/py -m fisheye.diagnostics.h5 report \
  "$PALETTE_TEST_FIXTURES_ROOT/recordings/2026-01-28T19-36-18Z_arena_1_Feeding"
```

### Run the shared diagnostics smoke runner

```bash
scripts/run_shared_diagnostics_smoke.sh

scripts/run_shared_diagnostics_smoke.sh --label feeding_fixture
```

This writes text and JSON/JSONL diagnostics artifacts under
`$PALETTE_TEST_RUNS_ROOT/diagnostics/...`.

### Write smoke outputs into a run directory

```bash
run_dir="$PALETTE_TEST_RUNS_ROOT/organize_diagnostics/20260417_smoke"
mkdir -p "$run_dir"
```

### Clone a fixture before destructive testing

```bash
run_dir="$PALETTE_TEST_RUNS_ROOT/organize_apply/20260417_apply_smoke"
mkdir -p "$run_dir"
cp -a --reflink=auto \
  "$PALETTE_TEST_FIXTURES_ROOT/recordings/2026-01-28T19-36-18Z_arena_1_Feeding" \
  "$run_dir/"
```

Using `--reflink=auto` is preferred on the NVMe volume because it keeps copies
cheap when the filesystem supports copy-on-write cloning.
