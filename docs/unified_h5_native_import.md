# Unified experimental H5 import

The stimulus importer admits a Citrus `unified_experimental_h5_v1` file and
records a **sealed reference** to it. The H5 is not copied: the raw file in the
recording's `raw/` tree is the single primary source, and its external
finalization receipt binds its exact bytes. Design and decisions:
[reference storage plan](design/2026-09-24-unified-h5-reference-storage/README.md)
(D13, D14 in the [adapter design](design/2026-09-23-unified-h5-legacy-adapter/README.md)).

```bash
scripts/py -m fisheye.analysis.import_stimulus_to_zarr \
  /path/to/recording/raw/acquisition/arena.h5 /path/to/recording/zarr/rec_analysis.zarr \
  --source-profile unified_experimental_h5_v1 \
  --finalization-receipt /path/to/finalization-receipt.json \
  --run-name unified_native_candidate
```

The session importer (`fisheye.utils.import_recording_analysis`) routes a
unified H5 here automatically when its plan carries the receipt
(`--finalization-receipt`). A unified H5 without explicit profile selection, an
unknown profile, a missing receipt, `--overwrite` or metadata-only mode are
refused before anything is written. Runs are immutable: a retry needs a new run
name. The run is never selector-eligible and never changes `latest`,
`latest_complete` or `authoritative_run`.

## What admission checks

- Both integrity domains: the file's exact bytes against the external receipt,
  and the internal dependency manifest against logical types, shapes,
  references and payload digests.
- Component accounting, exact uint64 identities, recording/camera
  correspondence, authored and executed protocol, geometry and presentation,
  and optional appearance replay.

Recorded source claims (geometry readiness, acceptance) stay source claims;
admission is not scientific acceptance.

## What the run stores

Under `analysis/stimulus_runs/<run>/`:

- `source_reference_json_utf8`: the H5 path relative to the Zarr store
  (confined to the recording directory), its size and mtime, the admission
  claims, a snapshot of every attribute, and per-dataset layout.
- `block_digests`: SHA-256 of each block of every fixed-width dataset, on
  Palette's own grid (about 1 MiB of whole rows, at most 4,096 rows). The grid
  does not depend on Citrus's HDF5 chunking.
- Attributes `source_profile`, `unified_reference_schema`/`_version` and
  `source_reference_sha256`, plus normal run completion and provenance.

The block digests come from admission's own streaming pass, so sealing adds no
extra read. Admission hashes each table once per import.

## Reading

`open_unified_source` is the only reader. Analyses never open the H5 path
themselves.

```python
import zarr
from fisheye.shared.unified_h5.reference import open_unified_source

root = zarr.open_group("/path/to/rec_analysis.zarr", mode="r", use_consolidated=True)
source = open_unified_source(root, run_name="unified_native_candidate")
frames = source.read_table("/frames/stimulus", start=0, stop=1000)
execution = source.read_json("/protocol/executed/execution_index_json")
subject = source.typed_attributes("/metadata/subject")
```

- **Open** checks the reference digest, that the direct and consolidated
  metadata agree, and that the H5 exists with the recorded size and mtime. A
  moved or replaced file fails loudly (`unified_source_changed`,
  `unified_source_missing`). Moving the whole recording directory is fine,
  because the path is relative.
- **Reads** verify only the blocks a row range touches, so memory is bounded by
  the range plus one block. JSON and variable-length strings (at most 8 MiB)
  are checked whole.
- **`source.verify()`** streams every block. It is for integrity sweeps and is
  not run on every open. A changed byte in a block nobody reads is caught by a
  sweep or by the first read that touches it.

## Not yet

Contract v2 capacity limits (after agent-contracts PR 52), the legacy-layout
adapter, and a unified end-to-end canary are separate work.
