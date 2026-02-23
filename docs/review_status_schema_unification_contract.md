# Review Status Schema Unification Contract

## Purpose

Define a single review-status contract shared by detect and keypoint workflows,
then align writers, registry storage, and query consumers around that contract.

This contract is the source of truth for the open TODO item in
`docs/pose_detect_parity_todo.md`:
"Unify review-status schema shape across detect/keypoint where practical."

## Scope

In scope:
- Detect/keypoint review payload shape and key naming.
- Registry quality extraction/upsert/view alignment for shared review fields.
- Query/consumer alignment so detect/keypoint review fields are accessed
  uniformly.

Out of scope:
- Eye-mask review schema changes (tracked separately).
- Destructive historical rewrites of old payloads.

## Current Gaps

- Detect/keypoint writers commonly emit `timestamp` rather than a canonical
  `timestamp_utc` key.
- Registry quality tables/views are not fully symmetric:
  - `detect_quality` has detect-specific `review_resolved_group`.
  - `keypoint_quality` retains keypoint-specific signature semantics via
    run attrs, but both modalities do not consistently expose the same shared
    review fields at the table/view level.
- Consumer code frequently uses per-modality field handling and fallback logic.

## Canonical Shared Review Payload

Writers must emit this shared payload shape for both detect and keypoint review
status attrs.

Required keys:
- `state`: `approved | pending | rejected | needs_review`
- `method`: `manual | algorithmic | hybrid | spotcheck`
- `intended_use`: `training | full_recording`
- `timestamp_utc`: ISO-8601 UTC timestamp (canonical key)

Optional keys:
- `reviewer`: string
- `notes`: string

Example:

```json
{
  "state": "approved",
  "method": "manual",
  "intended_use": "training",
  "reviewer": "alice",
  "notes": "spot-checked 100 frames",
  "timestamp_utc": "2026-02-23T03:12:45+00:00"
}
```

Domain-specific extensions remain allowed:
- Detect may include `resolved_group`, `target_group`, `preference_chain`.
- Keypoint may continue writing `keypoint_review_signature`.

## Timestamp Policy

- Canonical write key is `timestamp_utc`.
- Writers may optionally mirror to legacy `timestamp` during compatibility
  transition, but `timestamp_utc` must always be present for new writes.
- Readers must continue accepting legacy keys:
  - `timestamp_utc`
  - `timestamp`
  - `reviewed_at_utc`
  - `reviewed_at`

## Registry Alignment Requirements

Shared review fields should be symmetric between detect and keypoint quality
surfaces wherever practical:
- `review_state`
- `review_method`
- `review_intended_use`
- `review_reviewer`
- `review_notes`
- `review_timestamp_utc`

Modality-specific fields remain modality-specific:
- Detect: `review_resolved_group`
- Keypoint: signature fields remain on run attrs (and can be surfaced later if
  needed).

## Implementation Plan (Agent-Friendly)

### Workstream A: Writer Normalization

Target files:
- `src/fisheye/utils/accept_detect_review.py`
- `src/fisheye/utils/accept_keypoint_review.py`
- `src/fisheye/utils/set_keypoint_review_status.py`

Deliverables:
- Canonical payload keys from this contract.
- `timestamp_utc` always written.
- Existing strict-mode behavior preserved.

### Workstream B: Registry Schema/View Alignment

Target area:
- `src/fisheye/registry/db.py`

Deliverables:
- Align detect/keypoint quality extraction + upsert + current views for shared
  review columns.
- Keep existing detect/keypoint modality-specific columns.
- Maintain compatibility for existing rows and legacy attr keys.

### Workstream C: Consumer/Query Alignment

Target area:
- `src/fisheye/utils/registry_query.py`
- Any related reporting/selection utilities that currently special-case
  detect/keypoint review field access.

Deliverables:
- Consumers access aligned shared fields consistently.
- Reduced modality-specific parsing branches where not required.

## Validation Gates

Required:
- Unit tests for canonical payload writing in detect/keypoint review writers.
- Registry tests confirming detect/keypoint quality views expose aligned shared
  review columns.
- Query/consumer tests verifying aligned field access paths.

Recommended:
- One end-to-end recording validation run after merge:
  - write review status
  - refresh/scan registry
  - query detect/keypoint quality surfaces

## Compatibility Policy

- Non-breaking migration strategy: additive columns + fallback readers.
- Do not require immediate historical backfill to mark task complete.
- New writes should converge immediately on canonical payload keys.
