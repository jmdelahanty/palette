# MetaZebrobot H5 Snapshot Integration

This document describes how the acquisition agent should query the
MetaZebrobot read-only API and store a minimal, no-PII snapshot in an H5 file.

## API access

- If using the SSH tunnel, the base URL is:
  - http://127.0.0.1:18000
- If running on the DB host directly:
  - http://127.0.0.1:8000

No auth is required when using the SSH tunnel; the API is bound to localhost on
the DB host.

## Endpoints used

- List active dishes for dropdown:
  - GET /dishes?status=active&limit=200&offset=0
- Fetch a specific dish:
  - GET /dishes/{dish_id}
- Fetch its cross:
  - GET /crosses/{cross_id}

Note: GET /crosses/{cross_id} returns parents as a JSON-encoded string.
Parse it before storing.

## Snapshot JSON schema (required fields)

Store a single JSON object with the following fields:

```json
{
  "schema_version": 1,
  "queried_at_utc": "2026-01-16T23:45:12Z",
  "status": "complete",
  "missing": [],
  "dish_id": "15238_1",
  "dish": {
    "dish_id": "15238_1",
    "cross_id": "15238",
    "genotype": "Tg(gfap:TRPV1-T2A-GFP)",
    "dof": "20250106",
    "fish_count": 25,
    "species": "Danio rerio",
    "sex": "unknown"
  },
  "cross": {
    "cross_id": "15238",
    "line_strain": "Tg(gfap:TRPV1-T2A-GFP); Tg(elavl3:jRGECO1b)",
    "parents": [
      { "identifier": "M11:E5 (5187)", "sex": "M" },
      { "identifier": "M11:E6 (5188)", "sex": "F" }
    ]
  }
}
```

## Snapshot source priority

Prefer in-file subject metadata when available:

1. /subject_metadata (if present in H5)
2. MetaZebrobot API (dish + cross endpoints)
3. Partial snapshot (fill what is available, mark missing)

The subject_metadata block should be treated as capture-time provenance and
mirrored as-is (after PII filtering and normalization).

Normalization rules for /subject_metadata:
- Drop any PII fields (if present) that match the API exclusion list.
- Normalize parents into a list of objects: {identifier, sex}.
  - If parents is a single string, split on ";" and trim whitespace.
  - If parsing fails, set parents=[] and add an error if desired.
- Normalize date fields to strings (YYYYMMDD or ISO 8601 where applicable).

## Derived fields (optional, recommended)

These values are computed at acquisition time and are not fetched from the API:

- dpf_at_acquisition (int)
  - Compute as: session_start_utc.date() - dof_date (UTC dates).
  - If dof is missing, set null and add "dpf" to missing.

If present, include dpf_at_acquisition at the top level of the snapshot JSON.

## Session UUID mirror

If present in H5 root attributes, mirror session_uuid into Zarr analysis metadata
so it can be used as a stable dataset_id:

- analysis_metadata.session_uuid

If missing, the registry should fall back to a path-derived ID.

## Partial snapshot rules

Partial snapshots must not block acquisition. Use:

- status="partial"
- missing list with one or more of: "dish", "cross", "dpf"
- cross=null if the cross fetch fails or is missing
- dish_id is always recorded at the root if known

Optional provenance:

```json
"errors": [
  { "source": "cross", "message": "not found" }
]
```

## H5 storage layout

Write the JSON blob to a single dataset:

- Group: /zebrobot_snapshot
- Dataset: snapshot_json (UTF-8 JSON string)

Optional root attributes (for fast indexing):

- dish_id
- cross_id
- zebrobot_snapshot_utc
- zebrobot_schema_version

## PII rules (do not store)

Do not store the following fields:

- responsible, responsible_requestor
- notes, termination_reason
- quality_checks (including any notes)
- full data JSON from endpoints

Keep only the explicit fields listed in the snapshot schema above.

## Zarr mirror (post-import)

When H5 data is imported into Zarr, mirror the snapshot JSON into:

- analysis_metadata.zebrobot_snapshot

If /subject_metadata exists in H5, store it separately as:

- analysis_metadata.subject_metadata

This keeps downstream tooling independent of the H5 file while preserving the
H5 as the source of truth.

## Implementation notes

- Parse cross.parents from a JSON string to a list of objects.
  - If parsing fails, set parents=[] and add an error if desired.
- species and sex are only available from GET /dishes/{dish_id}.
- dish_id is provided by the acquisition UI dropdown (populated from active
  dishes).
