# Virtual Collection Manifest Schema

<!-- design-meta
status: draft
last_updated: 2026-05-07
-->

## Purpose

A virtual collection manifest is Palette's reproducible project/cohort
boundary. It records which recording archives and exact analysis runs were
selected for a cross-recording analysis or export.

It is not a SLEAP-style mutable labeling project. Labels and curated instances
live in each recording archive's `refined_*` authoring surfaces. The manifest
only points to those archives and freezes the source selections used for a
query, export, plot, report, or training-set build.

## Design Goals

- Keep one recording analysis Zarr as the canonical per-recording provenance
  archive.
- Make cross-recording analysis reproducible even when `latest` run pointers
  later change.
- Let registry queries produce durable, reviewable source selections.
- Let Parquet/DuckDB exports point back to the exact collection that produced
  them.
- Keep the manifest compact enough to diff, archive, and review.

## Non-Goals

- Do not duplicate full analysis arrays into the manifest.
- Do not make the registry or Parquet export lake authoritative.
- Do not use fuzzy protocol hashes. Similar-trial search is a future
  `protocol_trial_index_json` / registry indexing layer.
- Do not replace training manifests; training manifests can reference virtual
  collections or copy their resolved source rows.

## Required Top-Level Fields

```json
{
  "schema_id": "palette.virtual_collection_manifest",
  "schema_version": 1,
  "collection_id": "default_screen_6dpf_20260507_v001",
  "collection_name": "DefaultScreen 6 dpf May 2026",
  "created_utc": "2026-05-07T00:00:00Z",
  "created_by": "user_or_agent",
  "purpose": "cross_recording_analytics_export",
  "selection_policy": {
    "latest_allowed_during_selection": true,
    "latest_resolved_before_export": true,
    "missing_optional_runs": "warn",
    "missing_required_runs": "exclude"
  },
  "query": {},
  "records": [],
  "export_artifacts": []
}
```

## Query Block

The `query` block records how the collection was assembled. It should be
human-readable and machine-readable, but it is not the authority after
resolution. The `records` block is the frozen source selection.

Recommended fields:

```json
{
  "query": {
    "registry_path": "/nvme1/palette_registry.sqlite",
    "registry_snapshot_sha256": "optional",
    "filters": {
      "protocol_signature_hash": "optional",
      "protocol_semantic_hash": "optional",
      "canvas_name": "DefaultScreen",
      "dpf": 6,
      "strain": "optional",
      "recording_date_range": ["2026-01-01", "2026-12-31"]
    },
    "trial_descriptor_filters": {
      "status": "deferred",
      "note": "Future protocol_trial_index_json-backed search goes here."
    },
    "ordering": ["recording_start_utc", "recording_id"]
  }
}
```

## Record Entries

Each record entry resolves one recording archive and the exact source runs used
by downstream work.

```json
{
  "recording_id": "2026-01-28T19-22-28Z_arena_1_DefaultScreen",
  "analysis_zarr_path": "/nvme1/recordings/.../_analysis.zarr",
  "training_zarr_path": "/nvme1/recordings/.../_training.zarr",
  "recording_attrs": {
    "recording_start_utc": "2026-01-28T19:22:28Z",
    "arena_id": "arena_1",
    "canvas_name": "DefaultScreen",
    "dpf": 6,
    "strain": null,
    "clutch_id": null
  },
  "protocol": {
    "stimulus_run_id": "stimulus_20260209_084518",
    "protocol_signature_hash": "strict_or_derived_hash",
    "protocol_semantic_hash": "strict_semantic_hash",
    "protocol_snapshot_sha256": "optional"
  },
  "source_runs": {
    "detect_run": {
      "run_id": "detect_...",
      "path": "detect_runs/detect_...",
      "required": true,
      "selection": "resolved_latest",
      "schema_id": "optional",
      "schema_version": 1,
      "source_revision": null,
      "source_fingerprint": "optional"
    },
    "refined_detect_run": {},
    "refined_keypoints_run": {},
    "refined_subject_mask_run": {},
    "subject_shape_run": {},
    "tail_kinematics_run": {},
    "track_kinematics_run": {},
    "swim_bout_run": {},
    "bout_kinematics_run": {},
    "eye_angle_run": {},
    "stimulus_response_run": {}
  },
  "status": {
    "included": true,
    "warnings": [],
    "exclusions": []
  }
}
```

Run objects should use the same shape across run families:

```json
{
  "run_id": "string",
  "path": "analysis/track_kinematics_runs/offline/tk_hyst4_low2_latch_s005",
  "required": true,
  "selection": "explicit | resolved_latest | query_default",
  "schema_id": "optional",
  "schema_version": 1,
  "method": "optional",
  "method_version": "optional",
  "source_revision": "optional",
  "source_fingerprint": "optional",
  "lineage_hash": "optional"
}
```

## `latest` Policy

`latest` is acceptable while constructing a collection interactively, but it
must not remain implicit in exported rows.

Required behavior:

- The manifest may record that a run was selected by `latest`.
- The manifest must also record the concrete run ID resolved at build time.
- Exported Parquet rows must include the concrete source run IDs and manifest
  ID or manifest hash.
- Rebuilding an export from the same manifest must use the recorded run IDs,
  not re-resolve `latest`.

## Missing Run Policy

The manifest must distinguish missing required and optional runs.

Recommended defaults:

- Required for movement/bout analytics: `track_kinematics_run`,
  `swim_bout_run`, `bout_kinematics_run`.
- Required for stimulus response analytics: `stimulus_run` plus the selected
  response-specific source runs.
- Optional by default: `eye_angle_run`, `tail_kinematics_run`,
  `stimulus_response_run`, unless the export explicitly requests fields from
  them.

Records excluded by missing required sources should remain in the manifest with
`included: false` and an explicit exclusion reason. This preserves auditability
of the original query result.

## Manifest Hash

Writers should compute a stable SHA-256 over a canonical JSON representation of
the manifest excluding fields that are generated from the hash itself.

Recommended derived fields:

```json
{
  "manifest_sha256": "hex",
  "manifest_canonicalization": "json_sorted_keys_no_hash_fields_v1"
}
```

Exports should store both `collection_id` and `manifest_sha256`.

## Export Artifact Entries

When a collection is exported, append or write an export artifact entry:

```json
{
  "export_id": "palette_analytics_20260507T120000Z",
  "export_schema_id": "palette.cross_recording_analytics",
  "export_schema_version": 1,
  "output_root": "/nvme1/exports/palette_analytics",
  "created_utc": "2026-05-07T12:00:00Z",
  "tables": [
    {
      "name": "swim_bouts",
      "path": "swim_bouts/part-000.parquet",
      "row_count": 18800
    }
  ],
  "diagnostics": []
}
```

If the manifest is treated as immutable after creation, export artifacts should
live in a sibling export manifest instead of mutating the source collection
manifest.

## Relationship To Other Artifacts

- Recording Zarrs remain authoritative for per-recording raw/refined/derived
  data.
- Registry rows are indexes that help build manifests.
- Virtual collection manifests freeze cross-recording source selection.
- Parquet/DuckDB exports are rebuildable analytics products derived from a
  manifest.
- Training manifests may reference a virtual collection but should still carry
  task-specific source rows and quality filters.

## Open Decisions

1. Should collection manifests be immutable after creation, with separate
   export manifests, or can export artifact entries be appended?
2. Should manifests store absolute paths, registry dataset IDs, or both?
3. Which run families are required for the first production analytics export?
4. Should `latest_allowed_during_selection` default to false for automated
   production exports?
5. Where should manifests live by default: inside the export root, in a
   registry-managed directory, or next to analysis notebooks?
