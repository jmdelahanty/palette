# Acquisition Batch Registry Contract

Status: implemented; production batch assignments and cohort rollout not yet activated

Schema migration: 67 (`explicit_acquisition_batch_identity`)

Batch schema: `palette.registry.acquisition_batch.v1`

Assignment schema: `palette.registry.acquisition_batch_assignment.v1`

## Purpose

Palette needs an optional explicit identity for recordings that were acquired
together under shared technical conditions, such as simultaneous recordings from
four arenas. That nuisance/blocking identity is `acquisition_batch_id`.

The biological subject remains the experimental and inferential unit. An
acquisition batch does not merge arena-local state machines, recordings, or subjects
into one experiment, and it is not part of the subject identity. Analyses use it only
when their predeclared statistical design calls for batch adjustment or blocking.

It is deliberately distinct from:

- `session_uuid`, which identifies one acquisition/recording surface and may be
  arena-specific;
- `recording_started_utc`, which is descriptive acquisition time and may differ by
  a second or more across simultaneous recorders; and
- `recording_id` and `dataset_id`, which identify one recording and one concrete
  registered artifact, respectively.

No Palette registry migration, scan, or API may infer `acquisition_batch_id`
from timestamps, path names, recording names, arena numbers, or `session_uuid`.
Historical rows remain explicitly unassigned until an operator or an upstream
authoritative manifest supplies an assignment.

## Persisted model

`acquisition_batches` stores one immutable acquisition-batch entity:

- exact `acquisition_batch_id`;
- unique `batch_snapshot_id`;
- exact schema ID;
- `creation_method`, `created_by`, and `created_at_utc`;
- canonical strict-JSON evidence; and
- the registry schema version at creation.

`recording_acquisition_batch_assignments` stores append-only assignment
revisions:

- exact `recording_id` and `acquisition_batch_id`;
- monotonically increasing `assignment_revision` per recording;
- the prior `supersedes_assignment_snapshot_id` for corrections;
- unique `assignment_snapshot_id` and shared `assignment_batch_id`;
- exact schema ID;
- `assignment_method`, `assigned_by`, and `assigned_at_utc`;
- canonical strict-JSON evidence; and
- the registry schema version at assignment.

`recording_acquisition_batch_current` is the one explicit current-authority
pointer per recording. It references exactly one append-only assignment snapshot.
Foreign keys require both the recording and acquisition-batch entity to exist.
Recording, batch, current-pointer, and assignment-history foreign keys use
`ON DELETE RESTRICT`; deleting a recording or batch cannot erase its assignment
audit history.

Initial assignment cannot overwrite a current pointer, even with the same target
batch. Corrections use a separate API that appends a new revision and atomically
moves the pointer only when the caller supplies the exact expected current snapshot
ID. This compare-and-swap requirement prevents a stale operator or process from
overwriting a newer correction. Prior snapshot IDs, evidence, actors, timestamps,
and target acquisition batches remain queryable.

The current registry architecture does not persist a stable registry-instance UUID.
This contract records the provenance that architecture can currently guarantee:
schema version, UTC timestamps, actor/method/evidence, entity and assignment
snapshot IDs, and an atomic assignment-batch ID. A future registry-instance
identity can extend the envelope in a new schema version.

## Public API

`Registry` exposes:

- `create_acquisition_batch(...)`;
- `get_acquisition_batch(...)`;
- `assign_recordings_to_acquisition_batch(...)`;
- `correct_recording_acquisition_batch_assignment(...)`;
- `get_recording_acquisition_batch_assignment(...)`;
- `list_recording_acquisition_batch_assignment_history(...)`; and
- `resolve_dataset_acquisition_batch_assignment(...)`.

Creation and multi-recording assignment are separate transactions. Assignment of
the supplied recordings is atomic: an unknown recording, duplicate input, unknown
acquisition batch, or existing assignment aborts the entire operation. Read APIs
raise on missing
identity by default. Callers must opt into `require_assigned=False` when missing
identity is an expected inspection result rather than a correctness failure.

Correction is also atomic. It requires the current assignment snapshot ID, appends
exactly one revision, and updates exactly one current pointer. A stale snapshot ID,
unknown target acquisition batch, or duplicate revision/supersession aborts without changing
current authority or history.

Identifiers, method names, actors, UTC timestamps, UUID receipts, and evidence JSON
are validated before persistence. Non-finite JSON is rejected.

## Query surface

`dataset_context_current` and `Registry.query_datasets()` expose:

- `acquisition_batch_id`;
- acquisition-batch entity schema/snapshot IDs and creation provenance;
- `acquisition_batch_identity_status` (`explicit` or `missing`);
- assignment snapshot and batch IDs;
- assignment schema ID, revision, and superseded snapshot ID;
- assignment method, actor, and UTC timestamp; and
- the current-pointer update timestamp; and
- batch-creation and assignment-time registry schema versions.

`Registry.query_datasets()` accepts exact `acquisition_batch_id` and
`require_acquisition_batch` filters. The registry query CLI accepts
`--acquisition-batch-id` and `--acquisition-batch-status`.

Multiple datasets for one recording resolve through the same normalized recording
assignment. A dataset without a registered recording context remains `missing` and
the strict resolver fails closed.

## Migration behavior

Migration 67 creates the entity and assignment tables and refreshes
`dataset_context_current` and `recording_step_status_latest`. It performs no data
backfill. Existing recordings retain `session_uuid` and `recording_started_utc`, but
their new identity fields read as:

```text
acquisition_batch_id = NULL
acquisition_batch_identity_status = missing
```

This is intentional. Timestamp equality and near-equality are not evidence of
acquisition-batch membership.

## Analytics/export integration seam

An immutable analytics exporter resolves each source dataset through
`resolve_dataset_acquisition_batch_assignment(..., require_assigned=False)` and
always persists the explicit identity status. When the status is `explicit`, it
also persists:

- `acquisition_batch_id` as an optional nuisance/blocking variable;
- `acquisition_batch_snapshot_id`;
- `acquisition_batch_schema_id`;
- `acquisition_batch_assignment_snapshot_id`;
- `acquisition_batch_assignment_batch_id`;
- `acquisition_batch_assignment_schema_id`;
- `acquisition_batch_assignment_revision`;
- `acquisition_batch_supersedes_assignment_snapshot_id`;
- `acquisition_batch_creation_registry_schema_version`; and
- `acquisition_batch_assignment_registry_schema_version`.

The export receipt binds those fields—or the explicit `missing` status—to the exact
source dataset and registry snapshot/evidence used for selection. It must not fall
back to `recording_started_utc`, `session_uuid`, or name parsing. A missing batch
does not block a valid single-subject export. It does block an analysis that
explicitly requests acquisition-batch adjustment. Subject identity remains required
for both paths.

## Acceptance coverage

Focused tests prove:

- simultaneous arena recordings with a one-second start skew share one explicitly
  assigned acquisition batch;
- identical timestamps and related names do not create implicit membership;
- missing and recording-only contexts remain explicit and fail closed under strict
  resolution;
- multi-recording assignment is atomic;
- duplicate, unknown, invalid, and already-assigned inputs are rejected;
- corrections preserve history and reject stale compare-and-swap inputs;
- recording and acquisition-batch deletion cannot erase assignment history;
- persisted provenance receipts are exposed through dataset queries; and
- migration preserves all pre-existing rows as unassigned.
