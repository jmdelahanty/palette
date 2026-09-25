# Unified H5 reference storage (D14 implementation plan)

- **Status:** draft plan, not started.
- **Owner:** Jeremy Delahanty.
- **Last reviewed:** 2026-09-24.
- **Builds on:** decisions D13 (capacity) and D14 (reference, do not copy) in
  the [unified H5 adapter design](../2026-09-23-unified-h5-legacy-adapter/README.md).
- **Replaces:** PR 169's byte-copy native storage
  (`palette.unified_h5_native_storage@1`). No stored candidates exist
  (0 unified H5s in the store), so nothing migrates.

## Rule for this work: net subtraction

The change must leave `src/` smaller than it found it. Acceptance requires the
`src/` line count to fall by at least 400. If the implementation cannot meet
that, stop and simplify the design before writing more code.

## What is deleted

| File (on main) | Lines | Why it goes |
|---|---:|---|
| `shared/unified_h5/storage.py` | 473 | Byte-copy writer, Zarr payload validation, and the candidate reader over copied bytes |
| `shared/unified_h5/storage_schema.py` | 61 | Zarr payload/manifest array contracts for the copy (`new_native_run_name` moves) |
| `analysis/unified_stimulus_import.py` | 239 | Rewritten around the reference (about half its size) |
| `hdf5_types.dtype_from_descriptor` and helpers | ~100 | Only used to decode copied bytes; the reader returns h5py's own dtype |
| `tests/.../test_unified_stimulus_import.py` | 411 | Copy round-trip and copy-failure tests; replaced below |
| Generated census entries for the native payload writers | — | The writers no longer exist |

Kept unchanged: admission and validation (`admission`, `integrity`,
`correspondence`, `schema`, `rows`, `protocol`, `geometry`, `identity`,
`appearance`, `presentation`, the vendored evaluator), `metadata.py`, the
routing from PR 188 and the registry guard from PR 181.

## What is added (budget: about 350 lines)

One module, `shared/unified_h5/reference.py`:

1. **`seal_reference(...)`**: writes the sealed reference into the stimulus
   run group. It is called by the rewritten importer after admission.
2. **`open_unified_source(root, run_name)`**: the one and only reader.

### Sealed reference record

Stored at `analysis/stimulus_runs/<run>/` (same location; the run keeps
`source_profile` and `stage_selector_eligible=False`, so the PR 181 guard
still applies). Schema `palette.unified_h5_source_reference`, version 1.

- `source_relative_path`: the H5 path relative to the analysis Zarr store,
  confined to the recording directory; never absolute.
- `source_size_bytes`, `source_mtime_ns`: the cheap open check.
- `source_sha256` and the finalization receipt id and contract digest, as
  admission already computes and checks them.
- The admission summary.
- A typed-attribute snapshot (existing `native_attributes`, bounded as today),
  so metadata such as `/metadata/subject` is served from verified values.
- Per dataset: path, h5py dtype descriptor, shape, whole-dataset logical
  digest, rows per block, and an offset into one shared digest array.
- One Zarr array `block_digests` (uint8, `[n_blocks, 32]`): SHA-256 of each
  block's logical bytes.

**The block grid is Palette's own.** It reuses admission's existing
`iter_blocks` grid (about 1 MiB of whole rows, capped at 4,096 rows), not
Citrus's HDF5 chunking, which Citrus says is readable but not frozen. The
digests are collected inside admission's existing per-dataset hashing
(`describe_internal_dataset` / `describe_table` already stream every block
through SHA-256), so sealing adds no extra read. A 26 GiB 24-hour H5 gives
tens of thousands of digests (roughly 1–2 MB).

### Reader

`open_unified_source` returns an object with the same surface consumers use
today: `admission`, `read_table(path, start, stop)`, `read_json(path)`,
`typed_attributes(path)`. `read_payload` and `read_dataset` over copied bytes
are removed.

- **Open:** resolve the relative path and require the file to exist with the
  recorded size and mtime. Otherwise refuse with `unified_source_changed` or
  `unified_source_missing` and the exact re-admit command. The receipt is not
  re-hashed on open.
- **Read:** map the row range to its blocks, read each block as one hyperslab,
  compare its digest, then slice. A mismatch refuses
  (`unified_block_digest_mismatch`). Memory is bounded by the requested range
  plus one block.
- **JSON and variable strings:** read whole (still bounded at 8 MiB) and
  checked against the whole-dataset digest.
- **Full re-verification:** `verify_unified_source(root, run_name)` streams
  every block. It is for integrity sweeps; it is not run on every open.

Stated weakness: a changed byte in a block nobody reads is not detected until
it is read or a sweep runs. Size and mtime catch ordinary replacement; the
sweep catches the rest.

### Import flow (rewritten importer)

The same entry point and flags as PR 169 and PR 188:

1. Admission, with a hook that also records per-block digests. Today
   `describe_table` is called separately from integrity, correspondence and
   appearance validation, so some tables are hashed two or three times per
   admission. Memoize it per admission, keyed by path and file identity. That
   is a small deletion of repeated work, and the measured read volume must drop
   (asserted in a test that counts block reads).
2. `seal_reference` inside the existing stimulus-run creation, ownership and
   completion guard.
3. Register the H5 in `recording_artifacts` (existing table: `relpath`,
   `size_bytes`, `metadata_json` with the source digest and run name), so
   integrity sweeps and the registry know the H5 is load-bearing.

No payload is copied; the run group holds only the reference record and
digest array.

## Consumers switched

- `import_recording_analysis.project_unified_subject_metadata` (PR 190):
  `load_unified_stimulus_candidate` becomes `open_unified_source`; the
  provenance digest becomes the reference record digest.
- Tests that import `storage` switch to `reference`.

Nothing else consumes native candidates.

## Out of scope (separate PRs, in order)

- **PR B: contract v2 and capacity.** Pin the Citrus admission v2,
  correspondence v2 and pose v1 contracts once agent-contracts PR 52 is
  reviewed and pinned. Replace the 64 MiB / 2,000,000-row constants with the
  contract's ceilings (1 TiB / 2^40 rows), and add a test past 2,000,000 rows.
  This depends on PR 52, not on this plan.
- **Adapter v1** (complete sessions only), reading through
  `open_unified_source`.
- **The unified end-to-end canary**, which needs Citrus's transfer-v2
  fixtures.

## Tests (target: no more test lines than those removed)

- **Equivalence:** for every table in the base and appearance fixtures,
  `read_table` equals a direct h5py read, over full ranges and ranges
  straddling block boundaries.
- **Tamper and failure:**
  - a flipped byte in one block refuses reads touching that block only;
  - a replaced file (size or mtime changed) refuses on open;
  - a deleted H5 refuses loudly;
  - moving the whole recording directory still works (relative path);
  - a path escaping the recording refuses.
- **Memory:** a synthetic long table (bounded by today's constants until
  PR B) is read in ranges with peak memory under range plus one block.
- **Unchanged behaviour:** the PR 188 routing and PR 190 projection tests pass
  with only the import swap. The subject record digest must be identical to
  legacy (the existing equivalence test).
- **Registry:** a scan registers the H5 artifact, and the PR 181 guard still
  excludes the run from legacy protocol extraction.

## Acceptance

- `src/` lines fall by at least 400; tests do not grow.
- The generated Zarr census drops the native payload writers.
- Static gates pass, the parent-intake canaries pass on the exact commit, and
  all 24 required checks pass.
- `docs/unified_h5_native_import.md` is rewritten for the reference design
  (shorter), and the #169 handoff is marked superseded.

## Open questions

1. **The open check uses size and mtime, not a full hash.** Is that
   acceptable, given the full hash runs at import and in integrity sweeps?
   (Recommended: yes; a full hash of a 26 GiB file on every open is not.)
2. **Should the H5 artifact row be written by the importer (above) or by the
   registry scan?** Recommended: the importer, so the reference and the
   artifact row are created together.
