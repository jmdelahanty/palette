# Refined Detect Collapse V2

<!-- design-meta
status: active
last_updated: 2026-04-09
-->

## Purpose

Record the current detect contract after collapsing the detect-specific
preferred-layer experiment into `refined_detect_runs`.

## Contract

`refined_detect_runs/<run>` is the canonical curated detect surface.
The run root is metadata-only for current runs. Canonical data lives in:

- `instances/`
- `source_detections/`

## Sparse Projections

Current refined runs carry sparse curated projections under the run:

- `instances/`: active curated accepted rows
- `source_detections/`: raw-candidate audit rows

Shared readers should prefer `instances/` when it exists. `source_detections/`
is an audit surface, not the primary bbox render surface.

## Semantics

The slot-based edit vocabulary is:

- `present`
- `missing`
- `filtered_out`
- `ambiguous`

`source_kind_codes` remains the primary machine-readable provenance state:

- `none`
- `raw_detect`
- `interpolated`
- `manual`

`manual_edit_flags` is a sticky human-touch marker:

- `False`: the slot has not been explicitly changed by a human in this run
- `True`: the slot was manually corrected, manually cleared, or manually retuned

`reason` is explanatory only. It should not be the field consumers parse to
decide whether a row is present or filtered.

## Write Rules

- `refine_detect` writes sparse `instances/` and `source_detections/` for
  current runs.
- `detect_review` edits the canonical sparse surfaces directly.
- `detect_review` supports two refined edit modes today:
  - one slot per frame for the legacy single-subject workflow
  - one slot per `(frame, arena_id)` when fixed ROI definitions are available
    from subdish masks or arena-assignment metadata
- unconstrained multiple curated detections inside the same arena/ROI are still
  out of scope for the current `detect_review` UI.
- Small manual or retune edits should patch only the touched sparse rows plus
  run-level metadata; full rematerialization is for initialization or explicit
  rematerialize flows.
- `accept_detect_review` / `set_detect_review_status` should normally resolve
  current runs to `resolved_group="refined"`.

## Sparse Legacy Groups

Older archives may still contain sparse subgroups such as:

- `filtered`
- `interpolated`
- `manual`
- `manual_*`

These remain compatibility or provenance artifacts for legacy runs. They are not
the primary detect contract for new runs.

## Consumer Rule

Consumers that want the active curated detect surface should read:

1. `refined_detect_runs/<run>/instances` when present
2. otherwise legacy sparse refined groups for historical archives only

Consumers should only read sparse subgroups when they explicitly need legacy
provenance behavior.

## Removed Experiment

The earlier preferred-layer detect experiment was removed from the active
schema. Current archives should use only `detect_runs` and
`refined_detect_runs` for detect storage.

## Related Design Note

This document records the active short-term detect contract.

The longer-term multi-subject target is documented separately in:

- `docs/refined_detect_multisubject_goal.md`
