# Provenance, Registry Export, and Strict-JSON Audit - 2026-05-11

<!-- contract-meta
status: audit
last_verified: 2026-05-11
purpose: Snapshot of current provenance/staleness, analytics-export registry, and strict-JSON readiness.
-->

## Scope

This audit covers three operational surfaces:

- Derived-analysis provenance and source-staleness checks.
- Registry-indexed analytics export consistency.
- Strict JSON validity of `zarr.json` metadata under `/nvme1/recordings`.

It does not certify scientific correctness of any specific run. It checks whether the current tooling can identify stale/unverifiable lineage, whether indexed exports resolve to files, and whether metadata is parseable by strict JSON readers such as Crimson.

## Code Changes From This Audit

- `audit_analysis_staleness` now ignores absolute external filesystem paths unless they explicitly include a `.zarr/<internal>` suffix. This prevents raw video paths and archive-root paths from being misread as in-archive Zarr nodes.
- `audit_analysis_staleness` now strips compact-table query suffixes such as `tables/bouts?candidate_id=0&signal_id=4` before resolving the Zarr node. The query remains provenance, but the table path is the resolvable source node.
- `audit_analysis_staleness` now falls back to direct `zarr.json` / `.zattrs` metadata reads when parent Zarr metadata fails to enumerate a child that exists on disk. This matches the known stale-consolidated-metadata behavior seen in mutable local stores.
- `backfill_run_lineage_fingerprints` now covers the same run parent paths as `audit_analysis_staleness`, including online track-kinematics and stimulus runs.

## Real-Zarr Staleness Summary

Command shape:

```bash
scripts/py -c '... audit_zarr_analysis_staleness(...) summary ...'
```

Archives checked:

- `/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr`
- `/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr`
- `/nvme1/recordings/2026-01-28T23-15-10Z_arena_3_Feeding/zarr/2026-01-28T23-15-10Z_arena_3_Feeding_analysis.zarr`

Results after the auditor fixes:

| Archive | Runs audited | Fresh | Warning | No sources | Stale |
| --- | ---: | ---: | ---: | ---: | ---: |
| arena 2 Feeding | 87 | 0 | 86 | 1 | 0 |
| arena 1 DefaultScreen | 16 | 0 | 13 | 3 | 0 |
| arena 3 Feeding | 10 | 0 | 9 | 1 | 0 |

The remaining warnings are expected for the current canaries:

- `unverifiable_missing_expected_fingerprint`: older or best-effort runs refer to a source that has a current fingerprint, but the downstream run did not persist the expected source fingerprint at creation time.
- `source_not_latest`: historical exploratory runs still point at older source runs. This is useful information, not necessarily a stale failure unless `--require-latest-sources` is used.
- `no_sources`: some stimulus runs are source roots rather than downstream derived products, so they do not carry same-archive upstream source refs.

The practical gap is provenance completeness, not source resolution. New writers should persist source refs plus expected source fingerprints at creation time so future audits can move from "warning/unverifiable" to "fresh/stale".

## Registry Export Consistency

Current indexed export queried:

```bash
scripts/py -m fisheye.utils.query_analytics_exports \
  --registry /nvme1/palette_registry.sqlite \
  --collection-id movement_bouts_20260128_all_analysis_v002 \
  --latest --format json
```

Latest active export:

- `collection_id`: `movement_bouts_20260128_all_analysis_v002`
- `collection_manifest_sha256`: `03e69e88b06809bde50104378c0bd1c7f6b737a07a65103de76abf34019d2eb7`
- `export_run_id`: `run_20260507T_manifest_v002`
- `source_recording_count`: 52
- `table_count`: 6
- `diagnostics_count`: 0
- `output_root`: `/nvme1/exports/palette_analytics`

Row counts:

| Table | Rows |
| --- | ---: |
| `recording_summary` | 52 |
| `stimulus_steps` | 156 |
| `stimulus_step_summary` | 156 |
| `stimulus_response_per_fish_step` | 156 |
| `swim_bout_metrics` | 19,662 |
| `bout_kinematics_metrics` | 77,282 |

File validation command:

```bash
scripts/py -m fisheye.utils.check_analytics_exports \
  --registry /nvme1/palette_registry.sqlite \
  --check-files --format json --limit 20
```

The checked active export tables resolved with `check_status="ok"` and no missing listed Parquet parts. This validates the registry-to-manifest-to-file path for the current movement-bouts export.

Important distinction: cross-recording Parquet exports under `/nvme1/exports/palette_analytics` are implemented and registry-indexed. Archive-local Parquet sidecars inside individual `.zarr` stores remain a separate deferred layout option.

## Strict JSON Metadata

Command shape:

```bash
scripts/py -c '... json.loads(each zarr.json under /nvme1/recordings) ...'
```

Result:

```text
zarr_count 105
zarr_json_files 168895
bad_json_files 0
bad_zarrs 0
```

This means current recording/training/analysis archives under `/nvme1/recordings` are strict-JSON-clean at the `zarr.json` file level. The previous bare `NaN` metadata issue is not present in this scan.

## Remaining Risks

- Many existing runs still lack expected source fingerprints. The backfill tool can add best-effort fingerprints to run groups, but it cannot reconstruct the exact expected source fingerprint that should have been captured when a downstream run was first created.
- Source-not-latest warnings are common because the canaries intentionally retain exploratory historical runs. Production selection should use explicit run ids or virtual manifests rather than blindly selecting every historical run.
- Registry export checks validate file presence and indexed metadata. They do not prove every Parquet row is semantically fresh relative to upstream Zarr changes; that remains the responsibility of virtual collection selection plus source-freshness policy.
