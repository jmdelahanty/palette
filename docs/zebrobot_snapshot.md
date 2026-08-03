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

Recommended subject_metadata fields (capture-time provenance):
- fish_id (UUID string for the subject, when applicable; format `8-4-4-4-12` hex)
- subject_count (int; number of subjects in the recording)
- subject_type (e.g., individual, group)
- fish_count (int; if provided by acquisition UI)
- dish_id, cross_id, genotype, line_strain
- species, sex

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

Historical imports mirrored the snapshot JSON into:

- analysis_metadata.zebrobot_snapshot

and stored `/subject_metadata` as:

- analysis_metadata.subject_metadata

For new imports, the canonical acquisition snapshot is a versioned
`analysis/subject_metadata_runs/<run>` authority, and the declared recording
population is a versioned `analysis/experiment_setup_runs/<run>` authority. See
`docs/experiment_setup_contract.md`. The historical attrs above remain read
compatibility surfaces only. This keeps downstream tooling independent of the
H5 file while preserving the H5 as the source of truth.

Capture-time subject metadata is immutable. If acquisition recorded an
incorrect subject UUID, do not rewrite the H5, either completed canonical run,
or registry identities directly. The future correction publisher must create a
new lineage-bound subject/setup authority pair after the replacement UUID is
registered in MetaZebrobot. See
[`subject_metadata_identity_corrections.md`](subject_metadata_identity_corrections.md).

## Implementation notes

- Parse cross.parents from a JSON string to a list of objects.
  - If parsing fails, set parents=[] and add an error if desired.
- species and sex are only available from GET /dishes/{dish_id}.
- dish_id is provided by the acquisition UI dropdown (populated from active
  dishes).

## Migration note (2026-03-31): /crosses endpoint now backed by PyRAT

The local `crosses` table has been removed from MetaZebrobot. The
`GET /crosses/{cross_id}` endpoint now fetches crossing data live from the
PyRAT API and returns the same JSON shape.

Key difference: **parent sex is always "unknown"**. PyRAT stores fish as
`number_of_unknown` rather than `number_of_male`/`number_of_female`. The old
local crosses table had manually-entered sex assignments that are no longer
available. Parent identifiers (rack:position + tank ID) remain correct.

Consumers should treat `sex: "unknown"` as a valid value and not fail on it.
