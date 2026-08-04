# Experiment setup and subject identity contract

Status: canonical for new acquisition imports. Historical root attributes and
stimulus-run snapshots remain read compatibility surfaces.

## Scope

Experiment setup records what the acquisition intended to contain. It is not a
measurement of what detection, tracking, or later review found.

| Concept | Canonical field/surface | Meaning |
|---|---|---|
| Expected subjects | `expected_subject_count` | Subjects intended to be visible in the recording |
| Assigned subjects | `assigned_subject_count` and registry `recording_subjects` | Explicit biological identities attached to the recording |
| Expected arenas | `expected_arena_count` | Arena/dish regions represented by this recording |
| Source population | `source_dish_population_count` | Upstream holding/source-dish population; never the recording subject count |
| Observed objects | detection, arena-assignment, and tracking runs | Empirical output; never acquisition setup |

This distinction is important for current Batman H5 files: `subject_count=1`
describes the recorded individual, while `fish_count=35` or `45` describes the
source-dish population.

## Zarr authority

New imports publish separate immutable, versioned subject and setup records at:

```text
analysis/
  subject_metadata_runs/
    subject_metadata_<digest-prefix>/ # subject UUID(s) + biological metadata
  experiment_setup_runs/
    experiment_setup_<digest-prefix>/ # expected counts/arenas + subject ref
```

The schemas are `palette.subject_metadata.v1` and
`palette.experiment_setup.v2`. Each run stores a strict-JSON SHA-256, normal
Palette completion/provenance attributes, and is selected only through matching
`latest` and `latest_complete` pointers. The selected setup binds the exact
selected subject run and digest.
Consumers must resolve the selected complete run and validate its digest. A
present but incomplete or contradictory modern authority fails closed.

The root `experiment_setup` and `subject_count` attrs are compatibility
projections. They are not the modern authority. Historical archives without
`analysis/experiment_setup_runs` may still be read from the legacy root attr.

## H5 import mapping

`/subject_metadata/subject_count` is required to build a canonical setup and
must be a positive integer. `fish_id` or `subject_ids` supply explicit subject
membership. Missing identities do not cause synthetic biological IDs to be
invented. `fish_count` is copied as `source_dish_population_count` and is never
used to derive `expected_subject_count`.

Current Batman H5s store the biological subject UUID in the historical field
name `fish_id`. The source field is preserved verbatim, while each canonical
subject run also publishes normalized `subject_ids`, `subject_identity_kind`,
and `subject_identity_source_field` attrs. Registry projection uses those IDs
as `subjects.subject_id` and `recording_subjects.subject_id`. The H5/root
`session_uuid` identifies the acquisition session and must not be substituted
for a subject UUID.

The source H5 path and cheap stat fingerprint are recorded in provenance. A
second import of identical metadata is idempotent. Changed source metadata is
published as a new immutable version; an existing digest-addressed run whose
stored content does not match its digest fails closed.

## Count-only recordings

A recording can have a known expected subject count without any known
biological subject identities. This is a first-class state, including when the
expected count is greater than one:

- the subject metadata record has normalized `subject_ids=[]`,
  `subject_identity_kind="none"`, and no `fish_id`/`subject_id` fields;
- the setup has `expected_subject_count=N`, `assigned_subject_count=null`, and
  `subject_assignment_status="count_only"`;
- no `subjects` or `recording_subjects` rows are manufactured; and
- instances and run-local `track_id` values remain observations and temporal
  associations. They do not fill the missing biological identity.

Species, DPF, genotype, and similar cohort facts may still be asserted as
recording context. `dataset_context_current` exposes those values through the
`*_effective` columns and labels the identity state `count_only`, while the
explicit identity columns remain null. Cohort filtering may use effective DPF;
identity-coverage checks must continue to require explicit subject membership.

For a manual assertion, the selected immutable record contains the reviewer,
reason, assertion kind, and digest-bound evidence. The setup source is
`manual_operator_assertion`, never a fictitious H5 source.

## Stimulus runs

Stimulus runs own protocol definitions, stimulus/chaser timing, frame metadata,
alignment, events, enums, calibration snapshots, and H5 provenance. New runs
refer to the canonical setup and subject snapshot by path and SHA-256; they do
not own another authoritative subject count. Historical direct-CLI imports may
retain their old run-local snapshots as a labeled compatibility path.

## Registry projection

Registry scans prefer the selected `analysis/subject_metadata_runs` child over historical
`analysis_metadata` attrs. The declared count is projected to the provenance
subject-count snapshot. Explicit subject IDs are also upserted into `subjects`
and `recording_subjects`, with dish/cross entities when those IDs are present.
A count without identity remains useful setup metadata but does not manufacture
membership rows.

## Subject identity corrections

Capture-time subject metadata is immutable provenance. An incorrect biological
subject ID must be corrected by publishing a new, lineage-bound immutable
subject/setup authority pair, not by rewriting the H5, editing a completed run,
adding an unconsumed compatibility overlay, or directly mutating registry
identity rows.

The pair cannot be activated until Palette has a single pair selector or a
recovery-safe offline transition protocol; the current independent parent
selectors could otherwise expose a temporary subject/setup mismatch. Registry
state must then be refreshed from the selected pair. The complete design
boundary and implementation checklist are in
[Subject identity correction boundary](subject_metadata_identity_corrections.md).

## Historical H5 backfill

`fisheye.utils.backfill_subject_experiment_setup` is the canonical migration
command. It selects active `source_recording` analysis datasets from the
registry, requires exactly one colocated `raw/*.h5`, and accepts subject count
and identity only from `/subject_metadata`. Filesystem names, detections,
tracks, and source-dish population counts are never used as substitutes.

The default mode is read-only and reports `publish`, `verify_existing`,
`skipped`, and fail-closed `blocked` dispositions. Apply mode re-runs that
preflight immediately before mutation, publishes digest-addressed immutable
runs, validates both selected authorities, and refreshes the registry from the
Zarr. Registry-refreshing applies require a SQLite backup path. Reapplying an
identical H5 snapshot verifies the existing runs and does not add children.

```bash
scripts/py -m fisheye.utils.backfill_subject_experiment_setup \
  --registry /path/to/palette_registry.sqlite \
  --path-contains Batman \
  --output /tmp/batman-subject-setup-dry-run.json

scripts/py -m fisheye.utils.backfill_subject_experiment_setup \
  --registry /path/to/palette_registry.sqlite \
  --path-contains Batman \
  --apply \
  --backup /path/to/palette_registry.pre-subject-setup.sqlite \
  --output /tmp/batman-subject-setup-apply.json
```

This H5 command deliberately skips missing explicit identity. A missing
`/subject_metadata` group is not silently converted to manual evidence.

## Recording-local placeholder migration

The retired `backfill_subject_context` command historically converted a count
into names such as `<recording_id>:subject_0`. Those values have
`identity_scope=recording_local_placeholder`; they are not biological subjects.
Its CLI no longer permits apply.

`fisheye.utils.migrate_count_only_subject_context` converts that exact legacy
case. It compares the scientific fields across all active source-recording
analysis/training siblings, blocks disagreement, and produces a digest-bound
dry-run plan. Apply requires the unchanged plan and a new registry backup. It
publishes the identity-free authority on the canonical analysis Zarr, refreshes
all sibling registry projections, and deletes only placeholder rows whose
metadata digests still match the reviewed plan. A failure after the first
publication is recoverable by replaying the same plan; the tool never edits a
completed run.

```bash
scripts/py -m fisheye.utils.migrate_count_only_subject_context \
  --registry /path/to/palette_registry.sqlite \
  --all-placeholders \
  --reviewer jeremy \
  --reason "Known recording count; biological identities were never captured" \
  --output /tmp/count-only-plan.json

scripts/py -m fisheye.utils.migrate_count_only_subject_context \
  --registry /path/to/palette_registry.sqlite \
  --apply-plan /tmp/count-only-plan.json \
  --backup /path/to/palette_registry.pre-count-only.sqlite \
  --output /tmp/count-only-apply.json
```

This workflow is not a biological-ID correction. It removes an identity claim
that was never supported. Replacing one known subject UUID with another remains
governed by `subject_metadata_identity_corrections.md` and its stricter paired
activation boundary.

## Production gates

Detection quality resolves the expected count through the shared setup
resolver. An explicit CLI or DAG count is an assertion and must equal the
setup—it is not an override. Saved quality runs bind both the setup path and
digest. Collection quality uses the same rule.

Detection refinement verifies the quality run's expected count and setup
digest against the current authority before publishing a refined run. New
canonical setup plus an unbound or stale quality run fails closed. The explicit
`--allow-missing-experiment-setup` quality option exists only for controlled
historical compatibility and should not be used for production.

## Future evolution

Changes to the acquisition plan create a new immutable setup run and select it
only after validation; they do not edit a published run. Biological identity
updates must follow the versioned correction boundary above rather than
silently changing the setup.
Tracking identity (`track_id`) and observation identity (`instance_key`) remain
separate contracts.
