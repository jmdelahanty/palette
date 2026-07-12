# Registry-driven visualization montages

`fisheye.montage` selects analysis Zarrs
from the Palette registry and combines curated visualization artifacts into one
PNG per requested plot type. Registry access is SQLite read-only; the command
does not modify source Zarrs or the registry.

The package separates extension points by responsibility:

- `profiles.py`: curated plot types and their artifact path contracts
- `registry.py`: read-only cohort selection
- `render.py`: artifact loading, placeholders, resizing, and composition
- `workflow.py`: multi-plot orchestration and provenance manifests
- `cli.py`: command-line arguments and reporting

The command requires an explicit cohort selector, such as `--protocol-name`, or
an explicit `--all-recordings`. It also requires exact run names for plot types
whose artifact paths are run-scoped. This keeps a montage from silently mixing
analysis runs across recordings.

List the curated plot types and their required arguments:

```bash
scripts/py -m fisheye.montage \
  --list-plot-types
```

Create the RedScare distance-distribution montage from the production registry:

```bash
scripts/py -m fisheye.montage \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --protocol-name RedScare \
  --plot-type chaser-distance-distribution \
  --chaser-distance-run chaser_distance_redscare_v1_20260708 \
  --output-dir /nvme1/exports/palette_analytics/v1/artifacts/registry_montages/redscare_distance_20260710
```

Multiple `--plot-type` arguments produce separate montage PNGs with one shared
query and provenance manifest. Useful filters can be repeated or combined:

```bash
scripts/py -m fisheye.montage \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name RedScare \
  --chaser-behavior aggressive \
  --chaser-behavior inert \
  --chaser-count 2 \
  --arena-id arena_1 \
  --arena-id arena_2 \
  --recording-id-contains 2026-06-23 \
  --plot-type chaser-distance-median \
  --plot-type egocentric-bearing-polar \
  --chaser-distance-run chaser_distance_redscare_v1_20260708 \
  --egocentric-component egocentric_bearing_redscare_v1_20260708 \
  --output-dir /tmp/redscare_selected_montages
```

By default, a missing artifact fails the export. `--allow-missing` draws a
clearly marked placeholder and records the error in `montage_manifest.json`.
Existing outputs are never replaced unless `--overwrite` is passed.

Repeated `--chaser-behavior` filters and `--chaser-count` must match the same
stimulus run. They query the normalized `recording_chasers` child table. Tile
labels and the manifest include the indexed behavior vocabulary for each
selected recording.

Profiles may declare a required `visualization_contract_id`. Artifact loading
fails closed when a PNG is missing that contract or carries a different
renderer generation. Static PNG metadata records the contract ID, renderer,
and renderer version in both the artifact `zarr.json` and the parent
visualization manifest; those values also participate in the artifact
signature.

The `fish-escape-outcome-timeline` profile uses the escape/freeze component's
trial timeline. Red triangles represent candidate successful fish escapes and
gray circles represent failed/no fish escapes; classification remains governed
by the component's recorded full-trial path-length threshold.
