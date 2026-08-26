# Registry Stimulus Metadata Backfill

Palette normalizes recording stimulus metadata into:

```text
stimulus_protocols
stimulus_protocol_steps
recording_stimulus_runs
recording_stimulus_steps
recording_stimulus_modes
recording_stimulus_mode_counts
```

These registry tables are discovery indexes over authoritative recording
analysis Zarr metadata. They enable protocol-independent cohort queries such as
`stimulus_mode = 'CHASER'`.

Registry migration 72 also projects the exact producer protocol-semantic
status/hash, a bounded ordered recipe, versioned trial-index integrity, and
per-step semantic family/display context. Snapshot-v1 rows retain Palette's
explicitly local trial-index byte digest. Snapshot-v2 rows instead expose the
producer trial-index hash, exact execution hash/status and stimulus-frame
intervals, plus the sealed correspondence-proxy status/manifest digest. The
existing `protocol_hash` remains the distinct Palette-derived authored-protocol
hash.

## Safety model

`fisheye.registry.stimulus_metadata_backfill` is census-only by default. It:

- selects active, recording-owned analysis datasets through a read-only SQLite
  connection;
- opens recording Zarrs read-only with unconsolidated metadata;
- reports normalized protocols, runs, steps, and modes;
- reloads and validates exact semantic snapshot arrays and every stored step
  binding before reporting a run as `verified`;
- for snapshot v2, reloads the exact execution document, revalidates every
  half-open step/chaser-phase interval, and recomputes the correspondence-proxy
  arrays and manifest before indexing its seal;
- flags unreadable archives, missing/multiple latest run pointers, and
  `UNKNOWN` modes;
- does not run unrelated registry extractors.

Apply mode requires `--backup`. It creates a full SQLite backup immediately
before one sequential transaction and replaces only stimulus metadata tables
for successfully read datasets. A census containing issues is refused unless
the operator explicitly passes `--allow-issues`. Failed archive reads are never
applied or cleared, even with that override.

## Cluster workflow

Do not scan recording Zarrs or perform the backfill on an LSF login node. The
submitter renders locally and sends only `bsub` to
`login1-citrus-poller`; all archive reads happen inside the allocation.

First submit a census:

```bash
scripts/submit_stimulus_metadata_backfill_bsub.sh \
  --run-id stimulus_census_<date>_v001 \
  --all-recordings \
  --submit
```

Inspect the resulting JSON, especially:

```text
dataset_count
datasets_with_stimulus_count
latest_mode_dataset_counts
latest_protocol_semantic_status_run_counts
latest_protocol_semantic_hash_dataset_counts
latest_protocol_recipe_run_counts
issue_count and issues
```

Then submit a new apply run ID:

```bash
scripts/submit_stimulus_metadata_backfill_bsub.sh \
  --run-id stimulus_backfill_<date>_v001 \
  --all-recordings \
  --apply \
  --submit
```

The apply output root contains the census JSON, pre-write registry backup, job
script, LSF logs, submission record, and completion status. After completion,
verify SQLite integrity and compare `recording_stimulus_mode_counts` against the
census before building a virtual collection.

## All-chaser cohort

After a clean backfill, build the immutable collection with:

```bash
scripts/py -m fisheye.utils.build_virtual_collection_manifest \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --stimulus-mode CHASER \
  --collection-id all_chaser_<date>_v001 \
  --collection-name "All normalized chaser recordings" \
  --output /groups/johnson/johnsonlab/palette_analytics/v1/manifests/collections/all_chaser_<date>_v001.manifest.json
```

Protocol names remain metadata for confound checks; they are not cohort
membership predicates.

To distinguish the currently known GoodBatBadBat recipes, query the full
producer hash or bounded recipe instead of names:

```sql
SELECT protocol_semantic_hash, protocol_recipe_label, COUNT(*) AS recordings
FROM recording_stimulus_runs
WHERE is_latest = 1 AND protocol_semantic_status = 'verified'
GROUP BY protocol_semantic_hash, protocol_recipe_label;
```

Rows with `NULL` semantic status have not yet been inspected through this
contract. `legacy_missing` is written only after complete absence was observed
in the authoritative source; corrupt or partial modern state is an extraction
issue and never falls back to legacy.

The registry's proxy fields are discovery evidence, not an acquisition join.
`protocol_acquisition_containment_status` remains
`unavailable_without_sealed_stimulus_to_acquisition_mapping` until the live
frame-bound `recording_frame_id` identity chain is available. Camera frame IDs
and timestamps must not be promoted to acquisition-row authority by a query.
