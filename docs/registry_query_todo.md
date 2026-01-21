# Registry Query TODO

## Goal
Provide a simple query interface over the registry to support dataset selection
and downstream analysis (training, movement stats, protocol aggregates).

## Example Queries
- All fish at 6 dpf:
  - dpf_at_acquisition = 6
- All fish at 6 dpf from a given strain:
  - dpf_at_acquisition = 6 AND line_strain LIKE "%HHMI%"
- All fish from a cross with a given protocol:
  - cross_id = "17257" AND protocol_name = "DefaultScreen"
- All datasets with complete provenance:
  - snapshot_status = "complete" AND protocol_hash IS NOT NULL
- All datasets with missing provenance:
  - snapshot_status != "complete" OR snapshot_status IS NULL

## CLI Ideas
- python -m fisheye.registry.query --where 'dpf_at_acquisition = 6'
- python -m fisheye.registry.query --strain HHMI --protocol DefaultScreen
- python -m fisheye.registry.query --missing
- python -m fisheye.registry.query --list-ids

## Stub CLI (available now)
- python -m fisheye.registry.query --dpf 6
- python -m fisheye.registry.query --strain HHMI

## Output Formats
- Table summary (default)
- JSON list (for pipelines)
- CSV export

## Schema Coverage
- datasets: dataset_id, zarr_path, status
- provenance: dish_id, cross_id, line_strain, genotype, parents_json, species, sex,
  dpf_at_acquisition, protocol_name, protocol_hash, snapshot_status, snapshot_missing_json
- training_runs (optional join): run_id, manifest_path, model_path

## Query Helpers
- `--dpf <int>`
- `--strain <substring>`
- `--protocol <name>`
- `--cross-id <id>`
- `--dish-id <id>`
- `--status <active|missing>`
- `--provenance <complete|partial|missing>`
- `--since <YYYY-MM-DD>`

## Future Extensions
- Join with training_runs to pull best models for a given subset.
- Join with analysis outputs (swim bouts, movement metrics).
- Materialized views for common filters (e.g., complete provenance only).
