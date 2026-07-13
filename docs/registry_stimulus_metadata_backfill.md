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

## Safety model

`fisheye.registry.stimulus_metadata_backfill` is census-only by default. It:

- selects active, recording-owned analysis datasets through a read-only SQLite
  connection;
- opens recording Zarrs read-only with unconsolidated metadata;
- reports normalized protocols, runs, steps, and modes;
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
