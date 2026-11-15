# Recovered mask / tail18 implementation handoff

Owner: root. Worktree: `/tmp/palette-recovered-mask-tail18-20260918`.
Branch: `agent/palette/recovered-mask-tail18-20260918`.
Prerequisite: freshly fetched `origin/main` at
`46d8dce8fed3bb82134c6d3fe69ae03b94106b43`.
The original workstation checkout and other workers' worktrees are untouched.

The maintained implementation and compatibility contract are documented in
[the recovery workflow](../training/recovered_mask_tail_review.md). This adds
a new scientific recipe/schema and crop-only training review. It does not
change acquisition-camera defaults, existing skeletons, or existing analytics
sampling. The missing-point browser correction and out-of-crop save refusal
are explicit enforcement corrections.

## Evidence and first-recording scope

The read-only cohort audit joined all 3,153 mask rows across 15 recordings to
existing recovered pose crops. Pixel comparisons were exact; boxes matched
at absolute tolerance `1e-7`. There were no missing or ambiguous frame keys.
There are repeated images, so images alone are not a valid join key.

The local pilot is a disposable copy of
`2026-01-28T21-27-20Z_arena_4_Feeding_recovered_training.zarr`, at
`/tmp/palette-tail18-pilot-20260918/recording_training.zarr`.
Its first experimental version produced 155 valid tail seeds from 184 rows,
retaining 29 failures (27 fragmented body masks, two snout-extension failures).
All four fins are missing and all 184 initial rows are training-ineligible.
The JSON report retains exact failed source/crop row IDs and task definitions;
the montage includes both successes and failures. Local task-import validation
accepted both generated tasks; the isolated test store initially has no owner
assignment, so importing alone does not make tasks visible to a user.

Evidence directory: `/tmp/palette-tail18-pilot-20260918`.
The initial `pilot_v001` predates final lineage-column and frontend fixes; it is
local experimental evidence, not an activation or a production deployment.
Use a fresh version for final-code deployment; do not rewrite this evidence.

Local validation includes 59 focused recovery/schema/geometry/editor tests and
43 browser-script/web-route/delta-review tests. Tests cover identity mismatch,
byte tampering, ambiguous joins, immutable seed preservation, all-visibility
save gating, point-specific manual provenance, corrected-mask regeneration,
failed-publication retry, and unchanged selectors. The browser regression
executes the shipped JavaScript, including load/reset/place/save of null fins.

## Delivery and remaining gates

At this handoff's creation, implementation and local experimental computation
are validated. Required remote CI is still unrun; the branch is **not
merge-ready**. Required gates are generated artifacts, import boundaries,
file-size ratchet, Zarr open metadata modes, observed metadata literals,
active contract freshness, package and collection, all 16 non-GPU test shards,
and the aggregate `ci-required` gate. The exact pushed commit and later results
must be recorded in the PR and execution receipt.

The generated storage inventories are updated through their existing census
owner. No registry authority or production selector is activated. A shared
archive trial, if performed before integration, must remain selector-ineligible
and bind the exact committed code. Do not fast-forward the shared checkout or
deploy the new viewer over an existing service without its own green integration.

Manual fin annotation and resolution of the failed mask rows remain user work.
Finish mask corrections before fin labeling: regeneration creates another
version and preserves previous pose edits in their original version, but does
not yet carry them forward. A future label-based mapping can reuse the four
manual fin points and compatible head landmarks from `traditional_v3`; it must
not confuse its `mid_tail` or `tail_tip` indices with the new arc-length stations.
No tail18 merged-export adapter or new model training is included in this step.
