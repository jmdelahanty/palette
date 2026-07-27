# Refined-Detection Physical-Profile Canary Plan

Date: 2026-07-27

Status: Palette full-duration publication gate passed; Crimson physical
measurement pending

## Preconditions

Crimson's backend-independent refined-v1 consumer passed the unchanged Palette
shadow handoff. The immutable consumer evidence is:

- implementation commit:
  `28537f64bcae765b062374b17dd879c0a9614ade`;
- evidence commit:
  `57693f8cf0ebb18072bf031f1306ad84cc6bbf0c`;
- evidence file:
  `docs/diagnostics/refined_detection_v1_shadow_gate_2026-07-27/result.json`;
- evidence SHA-256:
  `b0221cfbce74824e31e8b557a4dde6b0357bab5ece004437b65b100c4a7fd643`;
  and
- result: 59/59 macOS tests and the real shadow gate passed.

That gate proved refined semantics, exact handles, one retained offset read,
lazy source-audit opening, complete traversal, residency, cancellation, and no
stale publication. It did not promote a physical profile.

## Representative Physical Input

The 23,287-frame real refined shadow is intentionally retained as the semantic
fixture, but it is too small for the required payload-object reduction: most
arrays collapse to one physical object under either profile. The physical
canary therefore uses the existing immutable full-duration canonical fixture:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/fixtures/sleepyfish_cam2010095_detect_20260724_v1/source.zarr`

Its frozen dimensions are 1,188,000 frames, 1,187,087 canonical rows, and
4512x4512 source pixels. The fixture manifest is the sibling
`fixture_manifest.json`, SHA-256
`ae1b65b1e5255168bed320cf0d099b16ef9966255c6aed098182e33bf653062a`.

Palette initializes a genuine refined-v1 root snapshot by accepting every
canonical row. It preserves the canonical frame, acquisition-frame,
`instance_key`, geometry, score, class, and offset arrays; assigns contiguous
root `refined_row_ids`; marks every score valid and every row raw-backed; and
creates the exact source-audit projection. This is benchmark-only root-snapshot
initialization. It does not relabel or discard the ten lineage arrays in the
existing clipped refined aggregate.

## Compared Profiles

- Regular control: exact 1 MiB uncompressed chunks, no sharding.
- Access-aware candidate: 128 KiB `WINDOWED`/`INDEXED` inner chunks, 1 MiB
  `EAGER` offset chunks, and 8 MiB outer shards.

For the declared full-duration refined dimensions, the byte planner estimates
240 regular payload objects and 48 candidate payload objects: exactly a 5x
reduction before fill-value elision. Publication also counts actual payload
objects and fails if either the planned or observed candidate/control ratio is
greater than 0.25.

## Publication Workflow

`publish_refined_detection_profile_canary()` and its diagnostic CLI:

1. require an immutable, selector-ineligible benchmark source and exact fixture
   inventory;
2. copy that source to node-local scratch and verify path/size/content identity;
3. build a canonical source plus logically identical regular and access-aware
   refined stores on scratch;
4. validate every logical array, manifest, physical plan, codec, CRC, direct
   and consolidated declaration, source-audit binding, identity, and offset;
5. copy each candidate through an exclusive incomplete sibling and atomic
   rename into a fresh `.palette_benchmarks` workflow;
6. reopen and rerun the complete validators from shared storage;
7. write a strict canary manifest binding Palette, Crimson, source, logical
   hashes, paths, plans, object counts, timings, and peak RSS; and
8. freeze the completed workflow read-only without creating selectors or
   registry state.

The intended destination is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/profile_canary/sleepyfish_accept_all_regular_vs_access_aware_20260727_v1`

## Promotion Boundary

Palette publication can prove correctness, equality, codec declarations, and
the 4x object gate. Crimson must still measure its frozen workload on the
mounted macOS path. Promotion additionally requires zero stale publications,
one retained offset read, zero deadline misses, no meaningful readiness,
current-frame, or RSS regression, and at least 20% lower traversal transfer.
Until that evidence is committed and reviewed, both profiles remain benchmark
artifacts and `profile_promoted=false`.

## Palette Publication Result

LSF job `153190577` completed on `h07u04` from the isolated, clean Palette
worktree pinned to
`a94abaea1206ae232b65185d1c98f250499af45b`. The shared `sun` checkout was not
changed. The immutable workflow is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/profile_canary/sleepyfish_accept_all_regular_vs_access_aware_20260727_v1`

The corresponding macOS path is:

`/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/profile_canary/sleepyfish_accept_all_regular_vs_access_aware_20260727_v1`

The strict `canary_manifest.json` has canonical payload digest
`2c00649c378c7a33f5621c4cd91ad787db46dcd0a512d6391ece92b383cd5609`
and file SHA-256
`8d9215aa29bf4b0787e50114e9fb429f959194ee3a7f1bdea6fd1d04ae1424b6`.

Results:

- dimensions: 1,188,000 frames, 1,187,087 instance rows, and 1,188,001
  retained offset boundaries;
- planned payload objects: 240 regular versus 48 access-aware, ratio 0.20;
- observed payload objects after fill elision: 220 regular versus 42
  access-aware, ratio 0.1909 or a 5.24x reduction;
- apparent payload: 59,867,920 bytes regular versus 59,769,380 bytes
  access-aware;
- exact decoded logical hashes: equal;
- canonical source-audit projection: equal in every array;
- direct/consolidated declarations, codec/CRC declarations, offsets, and
  selector-safety checks: passed;
- production-state changes: none;
- node-local source copy: 6.41 seconds;
- complete workflow: 42.08 seconds inside 57.94 seconds process wall time;
  and
- peak RSS: 702,332,928 bytes.

A separate fresh process reopened the shared canonical, regular, and
access-aware stores and reran the complete canonical/refined validators. It
again proved the 28-array schemas, 1,188,001-entry offsets, decoded equality,
source-audit equality, direct/consolidated equivalence, and selector absence.

Palette's publication and 4x object gate therefore pass. The access-aware
profile remains an unpromoted candidate until Crimson completes its frozen
mounted-macOS correctness, transfer, latency, readiness, deadline, and RSS
gate.
