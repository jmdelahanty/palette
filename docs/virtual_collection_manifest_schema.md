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
- Use the registry as the fast searchable index for available datasets,
  protocols, run status, and quality/state filters.
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

## Registry Role

The registry is expected to be the fast query layer. It should answer questions
such as "which active recordings have DefaultScreen data, 6 dpf metadata,
track kinematics, swim bouts, and bout kinematics available?" without scanning
every Zarr archive on demand.

That does not make the registry scientifically authoritative. Registry rows
are an index/cache over canonical recording archives and derived sidecars. A
virtual collection should use the registry to find candidate datasets, then
freeze the resolved recording IDs, paths, source run IDs, revisions, and
fingerprints in the manifest. If the registry is rebuilt or corrected later,
the manifest still records what was actually selected for the analysis.

Practical rule:

- Registry: fast discovery, filtering, freshness/status tracking, and
  operational state.
- Recording Zarrs: authoritative raw/refined/derived source data for one
  recording.
- Virtual collection manifest: immutable cross-recording source selection.
- Export lake: rebuildable analytics product generated from a manifest.

## Identity Versus Location

Manifests must not treat absolute filesystem paths as scientific identity.
Datasets can move between hot NVMe, network storage, and cold object/archive
storage while remaining the same logical source dataset.

V1 uses this split:

- Stable identity: `recording_id`, optional registry `dataset_id`, artifact
  kind, source run IDs, source revisions/fingerprints, and protocol hashes.
- Location audit snapshot: `locator_at_selection`, recording where the data
  was found when the manifest was built.
- Current/alternate locations: registry-owned mutable locator state, not
  duplicated into immutable manifests.

Recommended record shape:

```json
{
  "recording_id": "2026-01-28T19-22-28Z_arena_1_DefaultScreen",
  "dataset_id": "analysis_2026-01-28T19-22-28Z_arena_1_DefaultScreen",
  "artifact_kind": "analysis_zarr",
  "locator_at_selection": {
    "uri": "/nvme1/recordings/.../_analysis.zarr",
    "storage_tier": "hot_nvme",
    "last_verified_utc": "2026-05-07T12:00:00Z"
  }
}
```

Do not add `alternate_locators` to v1 manifests. The registry/catalog should
own current and alternate storage locators so stable manifests do not need to
change when data is archived or restored. A manifest remains scientifically
valid if its source archive moves; recomputation only requires resolving the
current locator through the registry.

For v1, `manifest_sha256` hashes the manifest as written, including
`locator_at_selection`, because it identifies that exact manifest file. If we
later need to compare scientific source selections independent of storage
movement, add a separate `source_selection_hash` that excludes locator fields.

## Required Top-Level Fields

See [examples/virtual_collection_manifest_v1.example.json](examples/virtual_collection_manifest_v1.example.json)
for a concrete v1 example.

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
    "production_requires_explicit_runs": true,
    "missing_optional_runs": "warn",
    "missing_required_runs": "exclude"
  },
  "query": {},
  "records": [],
  "manifest_canonicalization": "json_sorted_keys_no_hash_fields_v1",
  "manifest_sha256": "computed_after_canonicalization"
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
  "dataset_id": "analysis_2026-01-28T19-22-28Z_arena_1_DefaultScreen",
  "artifact_kind": "analysis_zarr",
  "locator_at_selection": {
    "uri": "/nvme1/recordings/.../_analysis.zarr",
    "storage_tier": "hot_nvme",
    "last_verified_utc": "2026-05-07T12:00:00Z"
  },
  "training_dataset_id": "training_2026-01-28T19-22-28Z_arena_1_DefaultScreen",
  "training_locator_at_selection": {
    "uri": "/nvme1/recordings/.../_training.zarr",
    "storage_tier": "hot_nvme",
    "last_verified_utc": "2026-05-07T12:00:00Z"
  },
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
  "present": true,
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

Absent runs should also be structured objects, not empty objects and not bare
`null` values:

```json
{
  "present": false,
  "required": false,
  "reason": "not_generated",
  "run_id": null,
  "path": null
}
```

This keeps diagnostics explicit and avoids ambiguity between "not generated",
"not selected", "not required", and "writer forgot to fill this object".

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
- Automated/production exports should default to explicit/concrete run IDs and
  require opt-in before resolving `latest`.

V1 selection examples:

```json
{
  "present": true,
  "selection": "resolved_latest",
  "run_id": "tk_hyst4_low2_latch_s005",
  "source_fingerprint": "example"
}
```

V2 tabular/refined-authoring surfaces should resolve `latest` to the most
precise stable snapshot available, not just a run group:

```json
{
  "present": true,
  "selection": "resolved_latest",
  "run_id": "refined_keypoints_main",
  "authoring_revision": 7,
  "table_snapshot_id": "rev_000007",
  "rowset_fingerprint": "example"
}
```

The policy is the same in both layouts: `latest` is a convenience for
selection, never a persisted unresolved dependency. If a refined surface is
edited, split, swapped, or repredicted later, `latest` may point to a newer
authoring revision, but existing manifests continue to point to the concrete
revision/snapshot they originally resolved.

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

V1 hash meaning:

- `manifest_sha256` identifies the exact immutable manifest document.
- Exclude only the `manifest_sha256` field itself.
- Include `created_utc`, query metadata, `locator_at_selection`, records,
  warnings, exclusions, and other manifest content.
- Do not include sibling export manifests in the collection manifest hash.

Canonicalization rules:

- Encode as UTF-8.
- Sort object keys recursively.
- Emit compact JSON with no insignificant whitespace.
- Preserve array order exactly as written.
- Normalize strings to Unicode NFC before hashing.
- Reject non-JSON numeric values (`NaN`, `Infinity`, `-Infinity`).
- Do not normalize or rewrite path/URI strings during hashing; hash the
  manifest exactly as written after JSON canonicalization.

Recommended derived fields:

```json
{
  "manifest_sha256": "hex",
  "manifest_canonicalization": "json_sorted_keys_no_hash_fields_v1"
}
```

Exports should store both `collection_id` and `manifest_sha256`.

This hash intentionally answers "which manifest file was used?" not "is this
the same scientific source selection after storage moves or timestamp changes?"
If location/time-independent source-selection equivalence becomes necessary,
add a separate `source_selection_hash` that excludes locator, creator, and
timestamp fields and hashes only stable source identities, run IDs, revisions,
fingerprints, and protocol hashes.

## Pre-Implementation Checklist

Work through these before implementing manifest writers/export integration:

- [x] Define canonical JSON hashing exactly: excluded fields, key ordering,
      whitespace, Unicode normalization, null handling, numeric/float
      formatting, path normalization, and whether sibling export manifests are
      excluded.
- [x] Define absent-run representation. Use structured objects with
      `present: false`, `required`, and `reason`; avoid ambiguous empty
      objects and bare `null` values.
- [x] Define path identity policy. Store stable recording/dataset IDs plus
      `locator_at_selection`; keep current/alternate locators in the registry,
      and do not treat absolute filesystem paths as scientific identity.
- [ ] Define `source_fingerprint` and `lineage_hash` semantics per run family,
      especially for refined authoring revisions and derived runs.
- [ ] Define immutable-manifest writer behavior: refuse overwrite by default,
      create new collection IDs/version suffixes for changed selections, and
      keep export artifacts in sibling export manifests.
- [x] Define production `latest` policy. Interactive selection may use
      `latest`, but production exporters should resolve and record concrete run
      IDs before writing rows, and may forbid `latest` by default.
- [ ] Decide how to record registry query provenance: registry path, registry
      backup/snapshot hash, query filters, result ordering, and excluded
      candidate records.
- [ ] Decide how to validate calibration/protocol metadata required by
      stimulus-response analyses, including direction mapping, homography,
      projector/camera scale, and protocol semantic hash status.
- [ ] Audit which refined surfaces currently expose authoring revisions or
      row-level revisions and which need backfilled revision metadata.
- [ ] Keep the v1 schema narrow enough to support current cross-recording
      analytics without prematurely locking in all future trial-search or
      compact-Zarr migration choices.

## Immutability Policy

Collection manifests are immutable after creation. Do not append export
artifact entries to the collection manifest. If the source selection changes,
write a new collection manifest with a new `collection_id` or version suffix.
If the same source selection is exported again with different export settings,
write a sibling export manifest that points back to the immutable collection.

## Export Artifact Manifests

When a collection is exported, write a sibling export manifest:

```json
{
  "schema_id": "palette.virtual_collection_export_manifest",
  "schema_version": 1,
  "export_id": "palette_analytics_20260507T120000Z",
  "collection_id": "default_screen_6dpf_20260507_v001",
  "collection_manifest_sha256": "hex",
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

## Relationship To Other Artifacts

- Recording Zarrs remain authoritative for per-recording raw/refined/derived
  data.
- Registry rows are indexes that help build manifests.
- Virtual collection manifests freeze cross-recording source selection.
- Parquet/DuckDB exports are rebuildable analytics products derived from an
  immutable collection manifest and recorded in sibling export manifests.
- Training manifests may reference a virtual collection but should still carry
  task-specific source rows and quality filters.

## Open Decisions

1. Should manifests store absolute paths, registry dataset IDs, or both?
2. Which run families are required for the first production analytics export?
3. Should `latest_allowed_during_selection` default to false for automated
   production exports?
4. Where should manifests live by default: inside the export root, in a
   registry-managed directory, or next to analysis notebooks?
