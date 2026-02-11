# Zarr String Encoding Standardization TODO

Purpose: unify how text-like data is stored in Zarr across Palette so runtime tools, C++/TensorStore consumers, and registry backfills behave consistently.

Date anchored: 2026-02-11.

## Why This Exists

Current code uses mixed encodings for text:

- fixed-length Unicode arrays (`<U...`) in some export paths,
- variable-length UTF-8 (`VariableLengthUTF8`) in several runtime writers,
- byte-encoded UTF-8 matrices (`uint8[N,width]`, null-terminated) for cross-tool reason labels.

This mixture causes avoidable drift, warning noise, and consumer compatibility issues.

## Design Goals

- migration-safe (additive-first, no destructive rewrites by default)
- operator-first (audit first, dry-run/apply tooling)
- cross-tool compatible (Python + C++ TensorStore readers)
- explicit and documented per-field storage contracts

## Canonical Encoding Policy (Target)

1. Reason/status labels that must be C++/TensorStore-safe:
- Primary: `uint8[N,width]` null-terminated UTF-8 (e.g. `reason_bytes`).
- Optional mirror: `reason` as `VariableLengthUTF8()` for Python ergonomics.
- Required attrs:
  - `reason_encoding="utf8-null-terminated"`
  - `reason_bytes_width=<int>`
  - `reason_bytes_null_terminated=true`
  - `reason_fallback_order=["reason_bytes","reason","detection_source"]`

2. General string columns/arrays (metadata tables, source indices, names):
- Canonical: `VariableLengthUTF8()`.
- Avoid new fixed-width Unicode (`<U...`) writes in runtime code.

3. Legacy compatibility:
- Reads should tolerate legacy `reason`/`<U...` arrays where present.
- Writes should emit canonical encoding going forward.

## Known Runtime Hotspots

Current fixed-width Unicode writes to migrate:

- `src/fisheye/utils/export_detect_training_zarr.py` (`_write_string_array`)
- `src/fisheye/utils/export_keypoint_training_zarr.py` (`_write_string_array`)

Current canonical reason codec (already aligned):

- `src/fisheye/shared/detect_reason_codec.py`

Current `VariableLengthUTF8` writers (already aligned):

- `src/fisheye/shared/detect_reason_codec.py` (`reason` mirror)
- `src/fisheye/segmentation/eye_segmentation.py` (reason array)
- `src/fisheye/refinement/refine_eye_masks.py` (reason array)

## Rollout Plan

### Phase 1: Policy + Guardrails

- [x] Add this policy link to relevant contracts/TODO docs.
- [ ] Add lint/CI grep guard to block new runtime writes using `<U...` for string arrays.

### Phase 2: Runtime Writer Convergence

- [x] Update export helpers to write string arrays using `VariableLengthUTF8()`.
  - `export_detect_training_zarr.py`
  - `export_keypoint_training_zarr.py`
- [x] Keep read paths backward-compatible with existing `<U...` archives.

### Phase 3: Audit + Backfill

- [x] Add an audit utility to scan archives and report text-encoding usage:
  - count arrays by encoding class (`reason_bytes`, vlen utf8, fixed-width unicode, other).
  - output candidate rewrite paths.
- [ ] Add optional backfill utility for safe rewrites (opt-in, scoped):
  - dry-run default, `--apply` required.
  - field/path allowlist only; no broad blind rewrites.

### Phase 4: Contract/Spec Sync

- [x] Update `src/fisheye/docs/zarr_structure.md` with explicit text-encoding conventions.
- [x] Ensure Crimson-facing contracts reference canonical encoding and fallback rules.

## Acceptance Criteria

- [x] No runtime code paths create new fixed-width Unicode (`<U...`) arrays for string data (for updated export writers).
- [x] Reason-label flows consistently emit `reason_bytes` with documented fallback attrs.
- [x] Audit report shows zero unsupported/legacy string encodings for newly created archives (validated on `/nvme1/recordings` snapshot).
- [ ] Existing archives remain readable without forced migration.

## Non-Goals (for this TODO)

- Forcing immediate rewrite of all historical archives.
- Changing numeric array layouts/chunking unrelated to string encoding.
- Altering registry biological normalization semantics.
