# Protocol Hash Stability TODO

## Goal

Ensure that the `protocol_hash` used to deduplicate protocols in the Palette
registry is stable across rigs, software versions, and serialization runs. Today
the hash covers the **entire** `protocol_definition_json` blob from Citrus, which
includes fields that are not experimentally meaningful. This causes "hash
fragmentation" — the same logical experiment gets multiple hashes.

Date anchored: 2026-03-02.

## Background

### How the hash is computed today

In `src/fisheye/registry/db.py` (`_extract_protocol`, ~line 1340):

```python
proto_hash = sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode("utf-8")
).hexdigest()
```

The `payload` is the full protocol JSON blob, read from the zarr attr
`protocol_json`, which in turn was imported verbatim from the H5 key
`protocol_snapshot/protocol_definition_json`.

`sort_keys=True` makes the hash insensitive to key ordering on the Python side,
but the actual field **values** come from C++ serialization and are never
normalized.

### Why the current approach is fragile

Three categories of instability:

#### 1. Pixel-derived / calibration-dependent fields

Some parameter structs include both physical units (mm, degrees, seconds) and
pixel-derived computational values (e.g. `spatial_freq_cpp` = cycles per pixel).
The pixel values depend on the arena calibration (mm-per-pixel) which can differ
between rigs or even between sessions on the same rig after recalibration.

**Impact**: Same physical protocol run on two rigs → different JSON → different
hash. These look like "different protocols" in the registry but are
experimentally identical.

#### 2. Software version drift

When a new Citrus version adds a field to a parameter struct (even with a
default value), the serialized JSON includes the new field. Every protocol
exported after the update gets a new hash, even if the experimental parameters
are unchanged.

**Impact**: Upgrading Citrus causes all protocols to get new hashes. Historical
data and new data appear to use "different protocols" even when the experiment
didn't change.

#### 3. Floating-point serialization

The protocol JSON originates from C++ serialization (nlohmann/json or similar).
Floating-point representation may vary across compilers, platforms, or library
versions. A value like `90.0` could serialize as `90`, `90.0`, or
`9e1` depending on the serializer configuration.

Python's `json.dumps` normalizes on the Python side, but the original C++ values
are baked into the blob before Python ever sees them. If C++ serializes `90.0`
as `90` on one machine and `90.0` on another, the strings differ and so does the
hash.

**Impact**: Same protocol, same parameters, different machines → potentially
different hash due to float formatting.

## Current state assessment

### What we know

- The hash deduplication strategy is fundamentally correct: hash-everything never
  conflates two different protocols. It errs on the side of splitting, not
  merging. This is the safe direction for scientific data.
- The `protocols.raw_json` escape hatch (planned for Phase 1 of the protocol
  parameter registry) means any parameter can always be reached via
  `json_extract()` regardless of hash fragmentation.
- Fragmentation is bounded by `(protocol files) × (software versions) × (rig
  calibrations)`. For a single-lab setup this is likely manageable but could grow
  over time.

### What we don't know yet

- **Which fields in each parameter struct are calibration-dependent vs
  experimentally meaningful?** This needs a Citrus-side audit.
- **How stable is the C++ float serialization in practice?** Are we already
  seeing different representations for the same value across machines?
- **Has Citrus ever added fields to parameter structs in a release?** If so, did
  it cause visible hash fragmentation in existing data?

---

## Recommendations

### Short term: Palette-side semantic hash (low risk, high value)

Add a second hash — `protocol_semantic_hash` — computed over only the
experimentally meaningful fields, stored alongside the existing full-content
`protocol_hash`.

- `protocol_hash` remains the exact-content fingerprint (never changes).
- `protocol_semantic_hash` groups protocols that are experimentally equivalent
  despite calibration, version, or serialization differences.
- Queries can use either: exact match (`protocol_hash`) or semantic grouping
  (`protocol_semantic_hash`).

This requires knowing which fields are "signal" vs "noise." That knowledge comes
from the Citrus-side audit below.

### Medium term: Citrus-side canonical serialization

Have Citrus produce a deterministic, normalized protocol JSON:

1. **Exclude pixel-derived fields** from the protocol definition JSON (or move
   them to a separate `calibration_context` block). The protocol definition
   should contain only physical units and experimental settings.
2. **Pin floating-point formatting** in the C++ serializer (e.g.,
   `nlohmann::json` default is `dump()` with `-1` precision which uses shortest
   round-trip representation — verify this is stable or pin to a specific format
   like `%.17g`).
3. **Version the protocol schema** — add a `protocol_schema_version` field to
   the JSON. When new fields are added, bump the version. Palette can then
   normalize across versions before hashing.

### Long term: Protocol family grouping

If semantic hashing isn't sufficient, add a `protocol_family` concept to the
registry — a human-assigned label that groups related protocol variants. But this
is likely unnecessary if the Citrus serialization is cleaned up.

---

## Citrus agent investigation tasks

The following tasks should be performed in the Citrus codebase to assess and fix
the serialization issues. These are ordered by priority.

### Task 1: Audit parameter structs for noise fields

For each protocol parameter struct in Citrus, classify every field as:

- **Experimental** — directly set by the experimenter, defines the experiment
  (e.g., `chase_probability`, `training_period_duration_s`, `loom_mode`,
  `l_over_v_ms`)
- **Calibration-derived** — computed from experimental values + arena calibration
  (e.g., `spatial_freq_cpp`, anything in pixels that has a mm counterpart)
- **Runtime/internal** — serializer artifacts, random seeds, display state, frame
  counters

Structs to audit (from `citrus_data_structure_documentation.md` Section 3):

- [ ] `ProtocolMovingGratingParams` (3.1)
- [ ] `ProtocolLoomingDotParams` (3.2)
- [ ] `ProtocolConcentricGratingParams` (3.3)
- [ ] `ProtocolCoherentDotsParams` (3.4)
- [ ] `ProtocolMovingDotsParams` (3.5)
- [ ] `ProtocolSpotlightParams` (3.6)
- [ ] `ProtocolScrollingGridParams` / `ProtocolIndependentMotionGridParams` (3.7)
- [ ] `ProtocolChaserParams` + `ChaserProperties` (3.8)
- [ ] `ProtocolSolidColorParams` (if exists)
- [ ] `ProtocolStaticImageParams` (if exists)

Deliverable: a table per struct with columns `field_name | type | category
(experimental / calibration / runtime) | physical_unit | notes`.

### Task 2: Check float serialization stability

1. Find where protocol JSON is serialized in Citrus (likely in the protocol
   save/export path, wherever `protocol_definition_json` is written to the H5
   file).
2. Check what JSON library is used and what float formatting it applies.
3. Verify whether the same protocol file produces byte-identical JSON across:
   - Different machines / compilers (if applicable)
   - Different runs on the same machine
4. If the serialization is not deterministic, recommend a fix (e.g., pin
   precision, use `dump(4)` or shortest round-trip mode consistently).

Deliverable: description of the serialization path, the library used, the float
format, and whether it is deterministic. If not, a recommended fix.

### Task 3: Check for version-drift field additions

1. Review git history for additions of new fields to any protocol parameter
   struct.
2. For each addition, check whether the field has a default value and whether
   old protocol files (without the field) would serialize differently after
   a round-trip through the new code.
3. Determine if Citrus has ever bumped a protocol schema version or if all
   versions produce the same JSON structure.

Deliverable: list of field additions with dates and whether they could cause
hash fragmentation on existing data.

### Task 4: Implement canonical serialization (if warranted)

Based on findings from Tasks 1-3, the Citrus agent should consider:

- [ ] Add a `protocol_schema_version` field to the top-level protocol JSON.
- [ ] Move calibration-derived fields to a separate `calibration_context` block
      (or simply exclude them from `protocol_definition_json` and keep them in a
      sibling key like `protocol_calibration_json`).
- [ ] Pin float formatting in the JSON serializer for protocol output.
- [ ] Ensure that re-serializing an old protocol file through new code produces
      the same JSON (or document where it won't and why).

### Task 5: Palette-side semantic hash implementation

Once the Citrus audit (Task 1) identifies which fields are experimental vs
noise, implement on the Palette side:

- [ ] Define a `_canonical_protocol_payload()` function that strips noise fields
      and normalizes values before hashing.
- [ ] Compute `protocol_semantic_hash` alongside `protocol_hash` in
      `_extract_protocol()`.
- [ ] Add `protocol_semantic_hash` column to `provenance` table (new migration).
- [ ] Add `semantic_hash` column to the planned `protocols` table (Phase 1 of
      `protocol_parameter_registry_todo.md`).
- [ ] Tests: verify that two protocol blobs differing only in calibration fields
      produce the same semantic hash but different full hashes.

---

## Blocking relationships

- **Task 1 (Citrus audit) blocks Task 4 and Task 5.** We cannot build the
  semantic hash or fix the serialization without knowing which fields are noise.
- **Tasks 2-3 (Citrus investigation) are non-blocking** — they inform priority
  but don't gate implementation.
- **Phase 1 of `protocol_parameter_registry_todo.md` is NOT blocked by this
  work.** The `protocols` and `protocol_steps` tables can use the current
  `protocol_hash` now, and `protocol_semantic_hash` can be added later as an
  additional column. The `raw_json` escape hatch covers ad-hoc queries in the
  interim.

## Related docs

- `docs/protocol_parameter_registry_todo.md` — parent TODO for protocol
  queryability
- `src/fisheye/docs/citrus_data_structure_documentation.md` — full parameter
  struct reference (Section 3)
- `src/fisheye/registry/db.py` — `_extract_protocol()` at ~line 1340
- `src/fisheye/analysis/import_stimulus_to_zarr.py` — H5 → zarr import (~line
  752)
