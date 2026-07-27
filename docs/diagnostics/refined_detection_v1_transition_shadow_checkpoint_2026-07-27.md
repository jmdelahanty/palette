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

This is logical transition evidence only. A real v1 shadow from this run is
still blocked on a canonical raw-source run manifest and logical-content
digest. Those identities must be derived from a validated source artifact, not
fabricated by the transition or shadow publisher.

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

## Remaining Gates

- Define or migrate canonical raw-source manifests so a historical transition
  can be bound to real source identity and content digests.
- Implement the separate clipped transition with ordered per-clip refined/raw
  evidence and full media/frame-map validation.
- Publish a bounded standalone refined shadow from validated real evidence.
- Have Crimson open that shadow through its dedicated refined-v1 consumer.
- Only then run the paired regular-versus-access-aware canary gate and consider
  a versioned physical-profile promotion.
- Keep delta v2, compaction, manual additions, downstream regeneration, and
  production selection deferred until this checkpoint is accepted.
