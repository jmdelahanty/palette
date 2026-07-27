# Refined-Detection V1 Transition And Shadow Checkpoint

Status: implementation checkpoint; selector-ineligible and not profile
promotion evidence

Date: 2026-07-27

## Outcome

Palette can now convert a current full-acquisition refined run into the exact
28-array refined-detection v1 logical schema without mutating its source, then
publish that result into a fresh standalone shadow store. The shadow path is
fail-closed: it cannot live inside a recording archive, cannot update a
selector or registry, and must pass decoded-schema, logical-hash,
direct/consolidated-metadata, manifest, snapshot-identity, and publication
validation before it returns success.

This does not authorize a production writer or storage-profile promotion.

## Crimson And DAG Re-Review

Crimson accepted the hardened Palette publication gate for recomputed-digest
tampering, root/successor identity, bound clipped evidence, and multiple rows
per frame. Its remaining required changes are consumer work:

- add a dedicated refined-v1 repository and fail-closed refined-first
  selection;
- retain `instance_key`, `refined_row_id`, and source kind through presentation
  and editing;
- validate offsets against `frame_indices` and enforce key uniqueness, or bind
  those checks to an accepted publication manifest;
- keep the source-audit table lazy; and
- keep within-frame ordinals and the legacy `uint8` box index outside the v1
  identity model.

The parallel Palette DAG review also accepted the hardened logical and
publication foundation for planning. Delta-schema, compactor, downstream
invalidation, and selector work remain deliberately deferred.

## Read-Only Real-Run Census

No source archive, selector, registry, or training artifact was changed.

### Full-acquisition historical run

Source:

`/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr/refined_detect_runs/refined_detect_2026-02-09_16-41-59`

Result:

- 23,287 frames;
- 22,926 refined instance rows;
- 22,938 source-candidate rows;
- default transition correctly blocked because the historical source audit has
  no durable `instance_key`;
- explicit historical initialization minted the frozen
  recording/frame/bbox/class keys for all 22,938 source rows;
- all 22,938 source keys and all 22,926 presented instance keys were unique;
- the instance and source `frame_row_offsets` ended at 22,926 and 22,938;
- exact v1 validation passed with no lossy conversion; and
- eight compatibility arrays were excluded explicitly rather than carried as
  aliases.

The initial logical transition was blocked on a canonical raw-source run
manifest and logical-content digest. The follow-up checkpoint below closes
that blocker by materializing a separate canonical raw shadow and binding the
refined source-audit table to it exactly. The historical source archive itself
remains unchanged and is never relabeled as canonical.

### Clipped Sleepyfish recording aggregate

Source:

`/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr/refined_detect_runs/refined_detect_sleepyfish_allclips_sharded_20260715_01`

Result: the full-acquisition adapter rejected the run before conversion because
it contains ten recording/clip/refined lineage arrays across `instances` and
`source_detections`. This is intentional. A clipped snapshot must be rebuilt
through the ordered source-collection and media/frame-map binding; treating it
as a full-acquisition run would silently lose scientific lineage.

## Shadow Publisher Guarantees

`publish_refined_detection_shadow()`:

- accepts only a contract-ready full-acquisition transition;
- accepts only a fresh `.zarr` child below `/tmp` or a
  `.palette_benchmarks` namespace;
- refuses any destination nested inside `<recording>_analysis.zarr`;
- creates no `latest`, `latest_complete`, or `authoritative_run` pointer;
- writes every array in whole non-overlapping shard or chunk ownership units;
- writes the exact planned Zarr v3 codecs and physical declarations;
- proves source/destination decoded logical hashes are identical;
- consolidates metadata, builds the exact run manifest, reconsolidates, and
  passes the complete named publication validator;
- writes strict JSON evidence with `production_state_changes: []`; and
- marks a failed run ineligible before re-raising the failure.

The real-Zarr test uses a frame containing one raw-backed and one manual row.
It verifies both rows survive the CSR range, keys remain unique, all exact
arrays are visible through consolidated metadata, and no selector exists.

## Canonical Source And Real Shadow Pair

Palette now has an executable `palette.canonical_detection.run_manifest` v1.
It binds:

- the exact canonical nine-array logical schema and decoded array hashes;
- the byte-planned `published_http_v1` storage declaration;
- exact direct and consolidated Zarr-v3 metadata declarations;
- an exact cross-check from every array's emitted chunk grid, shard codec,
  inner codec, checksum, fill, dtype, axes, and reserved attributes back to the
  resolved byte-based storage plan;
- an immutable logical-content digest used by refined snapshots;
- the source run's direct `zarr.json` digest and four legacy source-array
  hashes; and
- the read-only legacy-to-canonical conversion declaration.

The refined shadow publisher no longer accepts caller-supplied source digests.
It requires a validated canonical source publication and compares all nine
canonical arrays against the refined `source_detections` projection, including
stable keys and both `F+1` offset arrays, before creating its destination.

The full 23,287-frame historical pair was published under `/tmp`:

- canonical source:
  `/tmp/palette-canonical-detection-shadows/20260128_full_v1.zarr`;
- refined snapshot:
  `/tmp/palette-refined-detection-shadows/20260128_full_v1.zarr`;
- canonical rows: 22,938;
- refined presented rows: 22,926;
- refined source-audit rows: 22,938;
- canonical logical-content digest:
  `10c825c4d4605ebcc296f8b9a35da6581c5fc393a1b9232d753dd87bb3d60156`;
- canonical manifest digest:
  `0a8681605cacc69c91a7b8f7494de26e337df14db674a1d3bcf14fc138631211`;
- refined manifest digest:
  `d1e1501bae817b8c55ba9cec757daee9e4208919cd42eefa1bc3b87b5b5fb797`;
- canonical/refined apparent local sizes: approximately 920 KiB / 2.3 MiB;
  and
- canonical/refined file counts: 22 / 58.

A fresh-process consolidated reopen found no manifest errors, no selectors,
canonical offsets `[0, 22938]`, refined offsets `[0, 22926]`, and refined
source offsets `[0, 22938]`. Both strict receipts record
`production_state_changes: []`. A second fresh-process run through the complete
canonical and refined publication validators, including the physical-metadata
cross-check, returned no errors.

Adversarial coverage rejects a recomputed-digest nested storage mutation,
mutation of the historical source after evidence capture, and drift between a
refined source-audit value and its canonical source. The last failure occurs
before the refined destination is created.

These paths are ephemeral Palette-local integration evidence. They are not
Crimson handoff paths, production authorities, or physical-profile promotion
evidence.

## Shared Crimson Handoff

Crimson reported that its backend-independent refined-v1 consumer is ready for
a real selector-ineligible artifact. Palette therefore published a new,
immutable integration pair at:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/integration/20260128_refined_v1_crimson_20260727_v1`

The exact artifacts are:

- canonical source:
  `canonical_source.zarr`, run
  `detect_canonical_shadow_crimson_20260727_v1`;
- refined snapshot:
  `refined.zarr`, run
  `refined_detect_shadow_crimson_20260727_v1`; and
- strict pair handoff:
  `handoff_manifest.json`, payload digest
  `1fce7956f7f24bf6588900263366ca1c15d060bfdf3bddd4ac1f6c609bdaaf82`.

The corresponding macOS mount prefix is:

`/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/integration/20260128_refined_v1_crimson_20260727_v1`

The pair was produced from Palette commit
`2ecb7271f9f9d439b99c44e25141f1abfc11c16c` and re-opened in a separate
process. The complete canonical and refined publication validators returned no
errors; every one of the nine canonical arrays exactly equals its refined
`source_detections` projection; direct and consolidated declarations agree;
and neither store has a selector or production-state change. The canonical
manifest digest is
`70566e8d375cb1485ee4125ce09e3ac387832a912c1a24b6bb4748f8c68edf1d`.
The refined manifest digest is
`acdcf8209b1f329246c09d8aa2826682d3109963f864a2d87d50a92b2105de7b`.

This real historical snapshot has 361 empty frames, but no manual rows and no
frame with more than one refined row. It therefore tests complete real-store
metadata, offsets, filtering, stable identity, paging, and cancellation, while
the existing deterministic `[2,0,1,3]` raw/manual tests remain the evidence for
multi-instance frame behavior. Crimson must run both coverage classes; this
real fixture alone is not evidence for manual additions or multiple subjects.
No video was copied. The first handoff is consequently a headless consumer
gate; any GUI/media smoke must reference a separately accessible source video.

## Remaining Gates

- Implement the separate clipped transition with ordered per-clip refined/raw
  evidence and full media/frame-map validation.
- Have Crimson open the shared shadow through the dedicated refined-v1
  consumer, retain observation identity through presentation, and publish the
  exact consumer commit and results alongside the handoff digest.
- Only then run the paired regular-versus-access-aware canary gate and consider
  a versioned physical-profile promotion.
- Keep delta v2, compaction, manual additions, downstream regeneration, and
  production selection deferred until this checkpoint is accepted.
