# Chaser behavior metadata contract

## Canonical vocabulary

Palette uses the acquisition/runtime `chaser_behavior_classes` vocabulary:

| ID | Canonical label |
|---:|---|
| 0 | `unknown` |
| 1 | `aggressive` |
| 2 | `random_non_chasing` |
| 3 | `inert` |

`benign` is a deprecated Palette alias for `inert`. Readers may accept it for
historical CRA v1 artifacts and exports, but new analysis outputs and user-facing
labels use `inert`.

`static` is not a chaser behavior class. It describes an experimental phase or
activity state, such as `pre_static` or `post_static`, and must be stored
separately from chaser behavior identity.

## Variable cardinality

Chaser metadata is long-form and keyed by `chaser_index`. No contract assumes
that a recording contains exactly one or two chasers. A recording or stimulus
run may contain zero, one, two, or more rows:

```text
recording_id
stimulus_run_id
chaser_index
behavior_class_id
behavior_class
enable_chase
enable_random_movement
behavior_mode
raw_color_rgba
start_position_preset
end_position_preset
source_path
```

The logical key is `(recording_id, stimulus_run_id, chaser_index)`. Consumers
must join on that key rather than creating fixed columns such as
`chaser_0_role` or `chaser_1_role`.

## Current materialized surface

New chaser-distance runs materialize the configured classification at:

```text
analysis/chaser_distance_runs/<run>/chasers/
  chaser_index
  behavior_class_id
  behavior_class_label_bytes
```

This table has one row per chaser and aligns with the chaser axis used by
distance, position, epoch-summary, and histogram arrays.

The source authority remains the stimulus metadata:

```text
analysis/stimulus_runs/<run>/tracking_data/chaser_states/
  chaser_index
  chaser_behavior_class_id

analysis/enums/chaser_behavior_classes/
  id
  name
```

Configured behavior can also be resolved from `protocol_json` chaser entries.
The deterministic fallback mapping is:

```text
enable_chase=true                              -> aggressive
enable_chase=false, enable_random_movement=true  -> random_non_chasing
enable_chase=false, enable_random_movement=false -> inert
missing or unresolved runtime metadata          -> unknown
```

## Configured behavior, runtime behavior, and phase

These are separate dimensions:

- **Configured behavior**: the intended class for a chaser in the protocol.
- **Runtime behavior**: framewise `chaser_behavior_class_id`; it may briefly be
  `unknown` during startup or change during more complex protocols.
- **Phase/activity state**: `pre_static`, `training`, `post_static`, or another
  protocol-defined epoch.

Analyses that need a single label per chaser should use configured behavior and
retain runtime-class counts as QC. Analyses of transitions should use the
framewise runtime class rather than flattening it into one recording-level role.

## Registry normalization

Per-chaser behavior should not be added to the `recordings` table because it is
one-to-many and stimulus-run scoped. Palette indexes configured behavior in
`recording_chasers`, with one row per
`(dataset_id, stimulus_run_id, chaser_index)`, plus recording identity, source
lineage, and extraction timestamps. `recording_chaser_runs` provides per-run
counts without flattening chasers into fixed recording columns. The registry
remains a discovery index; the Zarr/H5 stimulus metadata remains authoritative.

Preview a protocol census without modifying the registry:

```bash
scripts/py -m fisheye.registry.chaser_metadata \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name RedScare \
  --output /tmp/redscare_chaser_census.json
```

After reviewing a zero-issue census, repeat with `--apply` to replace the child
rows for the selected datasets. Applied backfills fail closed when the census
contains extraction issues unless `--allow-issues` is explicitly supplied.

Pair-specific analyses such as the current CRA endpoint may require exactly one
`aggressive` and one `inert` chaser. That is an analysis precondition, not a
restriction on the shared behavior metadata contract.
