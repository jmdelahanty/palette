<!-- ARCHIVED 2026-07-24: historical migration record. Its 262,144-row values describe that campaign; current raw detection and keypoint defaults are 131,072 rows. -->

# Immutable YOLO sharding migration (2026-07-13)

## Scope and decision

At the time of this migration, Palette's serial YOLO writers used Zarr v3
indexed sharding for raw, immutable inference outputs:

- detections: `262144` outer rows for detection- and frame-domain arrays;
- keypoints: `262144` outer ROI rows and `262144` outer frame rows.

The July 2026 maintenance migration applies that layout to selected historical
`detect_runs` and `keypoints_runs` without rerunning inference. It does not
rename runs, change `latest`/`latest_complete`, or rewrite refined detection or
keypoint outputs.

Refined outputs remain ordinary-chunk working surfaces. Complete-shard creation
is fast, but a small edit inside a physical shard can require read-modify-write
of that shard and makes logically independent edits share one write object. A
future frozen refined snapshot may be compacted separately after it is declared
immutable; the live review/edit authority should not be sharded by this tool.

## New-run completion enforcement

Sharding is now the default throughout the raw YOLO production surface:

- direct detection and keypoint CLIs;
- registry-model wrappers;
- batch wrappers;
- whole-recording and clipped-collection keypoint planners; and
- their LSF submit scripts.

Ordinary chunks remain available only through the explicit
`--no-detect-sharding` or `--no-keypoint-sharding` compatibility overrides.
The analysis-workflow DAG consumes already-produced refined artifacts and does
not launch raw YOLO inference, so it does not define a competing storage
default.

Before a new raw detection or keypoint run can be marked complete or selected
as `latest_complete`, the writer now applies
`palette.immutable_yolo_storage_completion.v1`. It verifies:

1. the declared layout, policy, and requested outer row counts;
2. the physical shard grid of every direct fixed-width numeric/bool array;
3. required schema arrays and row/frame-domain lengths;
4. `uint64`, row-aligned, unique `instance_key` identity;
5. frame-count totals and paired count-array equality; and
6. the writer's completed decoded-byte hash comparison.

The report is stored in `immutable_yolo_storage_validation`. Any error records
that report, marks the run failed, and refuses pointer publication. This makes
a missing modern identity array, a lone ordinary array inside a nominally
sharded run, or an incomplete direct-shard write a production completion error
rather than a later audit discovery. Explicit ordinary-chunk runs pass the same
schema and identity checks while requiring that all eligible arrays are
actually unsharded.

## Identity prerequisite

`instance_key` is the modern stable observation identity. Historical
GoodCopBadCop lineages lacked it in detections, crops, raw/refined keypoints,
and refined detections. `backfill_legacy_instance_keys` now inherits the storage
grid of an already-sharded raw run:

- a new raw-keypoint `instance_key` uses the run's
  `keypoint_roi_shard_rows`;
- a new raw-detection `instance_key` uses `detect_row_shard_rows` when the
  selected detection run is already sharded;
- editable refined/crop/tracking identity arrays remain ordinarily chunked.

This lets identity be repaired before compaction without leaving a lone
ordinary array in an otherwise sharded raw-keypoint run.

## Migration safety contract

`fisheye.utils.migrate_immutable_yolo_sharding` is dry-run by default. On
`--apply`, one process owns the complete selected run and writes complete,
non-overlapping physical shards. It does not use Dask.

For every ordinary numeric/bool array selected for migration, the tool:

1. preserves shape, dtype, fill value, inner chunks, codecs, and array attrs;
2. creates a same-directory temporary sharded array;
3. copies complete outer shards while hashing decoded bytes;
4. rereads the staged array and requires exact SHA-256 equality;
5. stages and validates every array before publishing any array;
6. renames each original to a retained backup and publishes the staged array;
7. reopens every published array, verifies shard geometry, and hashes it again;
8. preserves run names, selectors, completion state, and historical provenance;
9. removes backups only after the entire selected run passes validation; and
10. refreshes consolidated metadata for the migrated selected-run subtree.

Stages with zero arrays to rewrite are metadata no-ops: their writer policy,
writer hash summary, and run provenance are preserved byte-for-byte. An early
GoodCopBadCop canary exposed and repaired a bug that relabeled such a stage as
`migrated_indexed_sharding_v1` with an empty migration summary. The repair
restored `default_indexed_sharding_v1` and the original
`palette.keypoint_double_buffered_shards.v1` summary from the run's preserved
inference provenance. A regression test now covers both no-op preservation and
that provenance recovery path.

An archive-level file lock prevents two migrators from running concurrently.
Ordinary Python failures trigger array and attribute rollback. Retained
`_palette_shard_*` artifacts are treated as an explicit recovery condition;
another migration refuses to proceed until they are inspected.

The cluster submitter always submits through the Citrus poller. The login host
only runs `bsub`; all Zarr planning, reads, hashes, writes, and validation occur
inside the LSF allocation:

```bash
scripts/submit_immutable_yolo_sharding_migration_bsub.sh \
  --run-id example_dry \
  --zarr-path /groups/.../recording_analysis.zarr \
  --stage both \
  --submit

scripts/submit_immutable_yolo_sharding_migration_bsub.sh \
  --run-id example_apply \
  --zarr-path /groups/.../recording_analysis.zarr \
  --stage both \
  --apply \
  --submit
```

## Canary result

Canonical canaries:

- RedScare: `2026-06-23T16-01-09Z_arena_1_RedScare`;
- GoodCopBadCop: `2026-05-29T18-11-16Z_arena_1_GoodCopBadCop`.

LSF job `153085179` completed both archive migrations in 20 seconds with
116 MB peak RSS. Every rewritten array passed staged and published decoded-byte
SHA-256 equality. A post-migration compute-node plan (`153085184`) found zero
arrays requiring another rewrite.

Payload-object changes in the selected runs were:

| archive/stage | before | after |
| --- | ---: | ---: |
| RedScare detection | 452 | 20 |
| RedScare keypoints | 623 | 56 |
| GoodCopBadCop detection | 717 | 12 |
| GoodCopBadCop keypoints | 51 | 51 (already sharded) |

For the RedScare canary, complete-shard copying took about 0.43 seconds for all
seven detection arrays and 1.57 seconds for all 23 raw-keypoint arrays. The
GoodCopBadCop detection copy took about 0.55 seconds. The remainder was setup,
repeat validation, atomic publication, and metadata maintenance.

The live registry also contains an active GoodCopBadCop example-copy dataset
under `jlcrsi/example_heartrate_recording`. Campaign selection must restrict
paths to the canonical `/groups/johnson/johnsonlab/jeremy/recordings/` tree;
the example override is not a migration target.

## Validation

Focused local validation (outside the Codex sandbox, per repository policy):

```text
28 passed, 4 Zarr-v3 consolidated-metadata warnings
```

The warnings state that consolidated metadata is not yet part of the Zarr v3
specification. Both canary root `zarr.json` files have no root-level
`consolidated_metadata`, so no stale root storage description remains; direct
array `zarr.json` is authoritative.

## Completed campaign

The production campaign completed all canonical eligible archives:

- 36 GoodCopBadCop recordings;
- 28 RedScare recordings;
- 64 total analysis Zarrs.

The four July 2 GoodCopBadCop recordings without selected detection/keypoint
runs were not eligible. The active `jlcrsi/example_heartrate_recording` copy was
also excluded because it is not under the canonical recordings tree.

Before storage migration, all 36 GoodCopBadCop selected lineages received and
validated modern `instance_key` identity. The canary plus remaining identity
jobs validated every archive; the 35-recording apply wrote 277 identity/lineage
arrays and reported 35/35 successful rereads.

The remaining 62-archive storage job was LSF job `153085200`. It completed in
379 seconds with 155 MB peak RSS. Together with the two canaries, the campaign
reduced selected raw detection/keypoint payload objects as follows:

| campaign | payload files before | payload files after |
| --- | ---: | ---: |
| GoodCopBadCop | 26,651 | 2,580 |
| RedScare | 29,187 | 2,032 |
| **Total** | **55,838** | **4,612** |

Apparent selected-run bytes changed from 1,516,949,616 to 1,521,577,397, an
increase of only 4,627,781 bytes (about 0.31%) for shard indexes and metadata.
Complete-shard copying across the campaign took about 71.1 seconds; staged
reread validation took about 33.1 seconds.

Final compute-node audit job `153085208` inspected all 64 archives and found:

- 64/64 archives readable;
- zero arrays remaining to migrate;
- zero no-op provenance repairs pending;
- all 1,830 selected raw named arrays on the default physical shard grids.

The 1,830 figure is a schema count across recordings, not a physical object
count: each GoodCopBadCop archive has seven detection and 22 keypoint arrays;
RedScare has seven detection and generally 21–23 keypoint arrays. Physical
payload objects are represented by the 4,612-file total above.

Registry projection job `153085214` then backed up the registry and refreshed
all 64 canonical Zarrs. It reported `missing=0`, `errors=0`, and
`no_quality=0`, updated 64 detect-quality rows, and passed SQLite foreign-key
and integrity checks. Its backup is:

```text
/groups/johnson/johnsonlab/jeremy/registries/audits/
registry_zarr_projection_refresh_bsub/
red_good_sharding_all64_registry_apply_20260713/registry.before_refresh.sqlite
```
