# Experimental Session Registry Contract

Status: implementation foundation, selector/export adoption not yet activated

Schema migration: 67 (`explicit_experimental_session_identity`)

Session schema: `palette.registry.experimental_session.v1`

Assignment schema: `palette.registry.experimental_session_assignment.v1`

## Purpose

Palette needs an explicit identity for the experimental unit shared by recordings
that were acquired together, such as simultaneous recordings from four arenas.
That identity is `experimental_session_id`.

It is deliberately distinct from:

- `session_uuid`, which identifies one acquisition/recording surface and may be
  arena-specific;
- `recording_started_utc`, which is descriptive acquisition time and may differ by
  a second or more across simultaneous recorders; and
- `recording_id` and `dataset_id`, which identify one recording and one concrete
  registered artifact, respectively.

No Palette registry migration, scan, or API may infer `experimental_session_id`
from timestamps, path names, recording names, arena numbers, or `session_uuid`.
Historical rows remain explicitly unassigned until an operator or an upstream
authoritative manifest supplies an assignment.

## Persisted model

`experimental_sessions` stores one immutable session entity:

- exact `experimental_session_id`;
- unique `session_snapshot_id`;
- exact schema ID;
- `creation_method`, `created_by`, and `created_at_utc`;
- canonical strict-JSON evidence; and
- the registry schema version at creation.

`recording_experimental_session_assignments` stores append-only assignment
revisions:

- exact `recording_id` and `experimental_session_id`;
- monotonically increasing `assignment_revision` per recording;
- the prior `supersedes_assignment_snapshot_id` for corrections;
- unique `assignment_snapshot_id` and shared `assignment_batch_id`;
- exact schema ID;
- `assignment_method`, `assigned_by`, and `assigned_at_utc`;
- canonical strict-JSON evidence; and
- the registry schema version at assignment.

`recording_experimental_session_current` is the one explicit current-authority
pointer per recording. It references exactly one append-only assignment snapshot.
Foreign keys require both the recording and session entity to exist. Recording,
session, current-pointer, and assignment-history foreign keys use `ON DELETE
RESTRICT`; deleting a recording or session cannot erase its assignment audit
history.

Initial assignment cannot overwrite a current pointer, even with the same target
session. Corrections use a separate API that appends a new revision and atomically
moves the pointer only when the caller supplies the exact expected current snapshot
ID. This compare-and-swap requirement prevents a stale operator or process from
overwriting a newer correction. Prior snapshot IDs, evidence, actors, timestamps,
and target sessions remain queryable.

The current registry architecture does not persist a stable registry-instance UUID.
This contract records the provenance that architecture can currently guarantee:
schema version, UTC timestamps, actor/method/evidence, entity and assignment
snapshot IDs, and an atomic assignment-batch ID. A future registry-instance
identity can extend the envelope in a new schema version.

## Public API

`Registry` exposes:

- `create_experimental_session(...)`;
- `get_experimental_session(...)`;
- `assign_recordings_to_experimental_session(...)`;
- `correct_recording_experimental_session_assignment(...)`;
- `get_recording_experimental_session_assignment(...)`;
- `list_recording_experimental_session_assignment_history(...)`; and
- `resolve_dataset_experimental_session_assignment(...)`.

Creation and multi-recording assignment are separate transactions. Assignment of
the supplied recordings is atomic: an unknown recording, duplicate input, unknown
session, or existing assignment aborts the entire batch. Read APIs raise on missing
identity by default. Callers must opt into `require_assigned=False` when missing
identity is an expected inspection result rather than a correctness failure.

Correction is also atomic. It requires the current assignment snapshot ID, appends
exactly one revision, and updates exactly one current pointer. A stale snapshot ID,
unknown target session, or duplicate revision/supersession aborts without changing
current authority or history.

Identifiers, method names, actors, UTC timestamps, UUID receipts, and evidence JSON
are validated before persistence. Non-finite JSON is rejected.

## Query surface

`dataset_context_current` and `Registry.query_datasets()` expose:

- `experimental_session_id`;
- session entity schema/snapshot IDs and creation provenance;
- `experimental_session_identity_status` (`explicit` or `missing`);
- assignment snapshot and batch IDs;
- assignment schema ID, revision, and superseded snapshot ID;
- assignment method, actor, and UTC timestamp; and
- the current-pointer update timestamp; and
- session-creation and assignment-time registry schema versions.

`Registry.query_datasets()` accepts exact `experimental_session_id` and
`require_experimental_session` filters. The registry query CLI accepts
`--experimental-session-id` and `--experimental-session-status`.

Multiple datasets for one recording resolve through the same normalized recording
assignment. A dataset without a registered recording context remains `missing` and
the strict resolver fails closed.

## Migration behavior

Migration 67 creates the entity and assignment tables and refreshes
`dataset_context_current` and `recording_step_status_latest`. It performs no data
backfill. Existing recordings retain `session_uuid` and `recording_started_utc`, but
their new identity fields read as:

```text
experimental_session_id = NULL
experimental_session_identity_status = missing
```

This is intentional. Timestamp equality and near-equality are not evidence of
experimental-session membership.

## Analytics/export integration seam

An immutable analytics exporter should resolve each source dataset through
`resolve_dataset_experimental_session_assignment(..., require_assigned=True)` and
persist at least:

- `experimental_session_id` as the statistical `session_id`;
- `experimental_session_snapshot_id`;
- `experimental_session_schema_id`;
- `experimental_session_assignment_snapshot_id`;
- `experimental_session_assignment_batch_id`;
- `experimental_session_assignment_schema_id`;
- `experimental_session_assignment_revision`;
- `experimental_session_supersedes_assignment_snapshot_id`;
- `experimental_session_creation_registry_schema_version`; and
- `experimental_session_assignment_registry_schema_version`.

The export receipt should bind those fields to the exact source dataset and registry
snapshot/evidence used for selection. It must not fall back to
`recording_started_utc`, `session_uuid`, or name parsing. Export adoption and any
live-registry assignment/backfill are intentionally outside this foundation
checkpoint.

## Acceptance coverage

Focused tests prove:

- simultaneous arena recordings with a one-second start skew share one explicitly
  assigned session;
- identical timestamps and related names do not create implicit membership;
- missing and recording-only contexts remain explicit and fail closed under strict
  resolution;
- multi-recording assignment is atomic;
- duplicate, unknown, invalid, and already-assigned inputs are rejected;
- corrections preserve history and reject stale compare-and-swap inputs;
- recording and session deletion cannot erase assignment history;
- persisted provenance receipts are exposed through dataset queries; and
- migration preserves all pre-existing rows as unassigned.
