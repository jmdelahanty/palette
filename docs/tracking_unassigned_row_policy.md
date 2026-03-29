# Tracking Unassigned-Row Policy

Date anchored: 2026-03-06

Purpose: define how unassigned rows in `arena_assignment_runs` and
`tracking_runs` should be represented, monitored, and eventually enforced in
status-driven workflows.

## Executive Summary

Unassigned rows are a first-class data-quality signal.

They mean:

- a source row exists
- the row could not be attributed to a valid `arena_id`
- the derived tracking row therefore has `track_id == -1`

They do not mean:

- a missing artifact
- a missing tracking run
- a real biological/public track

Current policy direction:

- keep unassigned rows explicit in `arena_assignment_runs` and `tracking_runs`
- exclude them from public `track_kinematics` outputs by default
- surface them in registry/status tooling as structured QA state plus readable
  warning/block text
- keep workflow blocking separate from basic step presence

## Scope

This policy applies to:

- `arena_assignment_runs/<run>`
- `tracking_runs/<run>`
- `track_kinematics` inputs and outputs
- registry `recording_step_status` rows for `tracks`
- `recording_step_status_wide`
- `check_recording_steps`

This policy is written for the current `single_subject_per_arena` operating
mode, but the same separation of concerns should hold for future multi-subject
tracking methods.

## Core Principles

### 1. Unassigned rows must remain representable

The archive must keep enough information to answer:

- how many rows were unassigned
- what fraction of tracked rows were unassigned
- which runs produced those rows

That means unassigned rows stay visible in raw tracking artifacts and step
details.

### 2. Unassigned rows must not become public tracks

Unassigned rows are QA/provenance signals, not subject identities.

Therefore:

- `track_id == -1` is allowed inside `tracking_runs`
- `track_id == -1` must not produce public `tracks/id_-1` outputs in normal
  downstream analysis
- `track_kinematics` should exclude unassigned rows by default and only keep
  them for explicit diagnostic runs

### 3. Artifact presence and quality state are different things

The `tracks` step can be present and still have a quality problem.

So the policy should not overload step presence status with QA state.

Keep these concepts separate:

- step status: `ok | missing | absent | na | error`
- tracking QA state: current runtime `ok | warn`; `block` is deferred policy

The step can still be `ok` while QA state is `warn`.

## Required Metrics

The canonical tracking summary should expose:

- `n_rows`
- `n_assigned_rows`
- `n_unassigned_rows`
- `unassigned_row_rate_percent`
- `n_tracks`

These values should flow from `tracking_runs` into registry/status details
without recomputing different semantics in each consumer.

## QA State Model

### `ok`

Use `ok` when:

- the tracking artifact is present
- `n_unassigned_rows == 0`

### `warn`

Use `warn` when:

- the tracking artifact is present
- `n_unassigned_rows > 0`
- the configured blocking threshold is not reached

This is the current active UI/status behavior.

### `block` (deferred)

Use `block` when:

- the tracking artifact is present
- unassigned rows exceed a workflow-defined unacceptable threshold

`block` means:

- the run should remain inspectable
- the run should not be treated as missing
- selected downstream workflows may refuse to proceed until the issue is
  resolved or explicitly overridden

## Recommended Thresholds For `single_subject_per_arena`

Recommended defaults:

- `ok`: `n_unassigned_rows == 0`
- `warn`: `n_unassigned_rows > 0`
- future `block`: `n_unassigned_rows >= 10` or `unassigned_row_rate_percent >= 1.0`

Rationale:

- any unassigned row is operationally important in one-subject-per-arena data
- a few rows may still be diagnosable without invalidating the entire run
- a sustained fraction or a larger absolute count should be treated as a real
  workflow gate

These defaults should remain configurable per workflow.

## Status Surface Behavior

### Registry / Wide Views

The `tracks` row should carry the summary metrics in `details_json`.

The wide view currently renders:

- `OK`
- `WARN (<n> unassigned, <pct>%)`
- later, if blocking is added, a distinct blocking rendering such as
  `BLOCK (<n> unassigned, <pct>%)`

### `check_recording_steps`

`check_recording_steps` should show the same QA interpretation as registry
status:

- `MISS` when the step is absent
- `OK` when present with no unassigned rows
- `WARN (...)` when present with nonzero unassigned rows

If blocking is introduced, it should render as a distinct tracking QA state,
not by pretending the step is missing.

## Workflow Gating

Recommended rollout:

### Phase 1: informational warning

Current behavior:

- expose warning text in registry and CLI status surfaces
- do not block downstream workflows automatically

### Phase 2: explicit QA state

Current behavior:

- `tracking_qc_state`
- optional threshold metadata:
  - `tracking_warn_threshold_rows`
  - `tracking_warn_threshold_percent`
  - `tracking_block_threshold_rows`
  - `tracking_block_threshold_percent`

These values now exist on the canonical tracking summary and are propagated into
registry/status details so status tooling does not need to infer semantics from
free-form display text alone.

### Phase 3: workflow-specific blocking (deferred)

Use `tracking_qc_state == block` as a gate only in workflows that care about
track completeness, for example:

- final downstream analysis exports
- derived summary bundles
- release-like reporting flows

Avoid making all developer diagnostics fail automatically just because a run is
blocked for production use.

## Non-Goals

This policy does not currently define:

- how to recover or relabel unassigned rows
- how multi-subject tracking should handle unresolved identity
- how online target/chaser identity should express QA thresholds

## Related Docs

- [`single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)
- [`tracking_runs_contract_status.md`](./tracking_runs_contract_status.md)
- [`track_identity_target_architecture.md`](./track_identity_target_architecture.md)
