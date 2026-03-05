# Registry Query TODO

## Goal
Provide a simple query interface over the registry to support dataset selection
and downstream analysis (training, movement stats, protocol aggregates).

## Status (2026-02-25)

The core query CLI (`src/fisheye/registry/query.py`) is **fully implemented** and
far exceeds the original stub scope described below. Current capabilities include:

### Implemented query helpers
- [x] `--dpf <int>`, `--dpf-min`, `--dpf-max`
- [x] `--strain <substring>` (via `--genotype`)
- [x] `--protocol <name>`
- [x] `--cross-id <id>`
- [x] `--dish-id <id>`
- [x] `--status <active|missing>`
- [x] `--provenance <complete|partial|missing>`
- [x] `--rig-id`, `--arena-id`, `--camera-id`
- [x] `--where <raw SQL>`
- [x] `--list-ids`, `--output-file-list`
- [x] `--detect-coverage-min`, `--detect-fps-min`, `--detect-read-ms-max`
- [x] `--detect-method`, `--detect-model-like`, `--detect-model-only`
- [x] `--group-by model|rig|camera|arena|dish`
- [x] `--include-training`, `--trained-only`, `--set-id`

### Implemented output formats
- [x] Table summary (default)
- [x] JSON list
- [x] CSV export

## Remaining items

- [ ] Add `--since <YYYY-MM-DD>` date filter.
- [ ] Materialized views for common filters (evaluate need based on query performance).
