# Keypoint Merged Row-Gate Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-02-27
-->

Purpose: define row-level inclusion semantics for merged pose training exports.

## Contract

1. Row inclusion is finalized at merged-export time.
2. Preferred mask is refined `usable_keypoints` from the refined run linked to the selected keypoint run.
3. If refined usable mask is unavailable, fallback is raw `detection_success`.
4. Pose loader must not apply an additional row drop for merged exports that already applied row gating.

## Policies

- `auto` (default): use refined usable mask when available, else raw success.
- `refined_usable`: require refined usable mask; fail if unavailable.
- `raw_success`: use raw `detection_success` only.

## Provenance requirements

Merged keypoint run attrs must include:

- `method = "merged_export"`
- `row_gate_applied = true`
- `row_gate_policy = <policy|mixed>`
- `row_gate_counts = { ... }`

Merged manifest/summary should include:

- requested and applied row-gate policy
- per-policy counts
- per-source row-gate details (selected/total/refined run when used)

## Rationale

- Keeps training rows aligned with manual review intent (`usable_keypoints`).
- Avoids hidden post-export row drops in loader.
- Makes row accounting deterministic and auditable.
