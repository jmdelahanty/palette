# Doc/Code Staleness Pass - 2026-05-10
<!-- design-meta
status: audit
last_verified: 2026-05-10
-->

## Scope

This pass checked docs and code after the first derived-analysis registry/status
slice. The focus was stale claims caused by:

- canonical stage-catalog adoption in registry, pipeline, and launcher code
- derived-analysis `recording_step_status` coverage
- Megabouts classifier and bout-classification run implementation

## Confirmed Current State

- `src/fisheye/registry/stage_catalog.py` is now the canonical stage catalog
  for registry/status stages and includes `category="derived_analysis"` specs
  for:
  - `track_kinematics`
  - `swim_bouts`
  - `bout_kinematics`
  - `eye_angles`
  - `subject_shape`
  - `tail_kinematics`
  - `tail_posture_view`
  - `bout_classification`
  - `stimulus_response`
- `src/fisheye/registry/step_cascade.py` derives invalidation edges from the
  catalog.
- `src/fisheye/core/pipeline.py` exposes `Pipeline.STAGE_CANONICAL_IDS` so
  runtime command-stage names have an explicit registry-stage mapping.
- `src/fisheye/cli/interactive_launcher.py` exposes canonical IDs for launcher
  stage rows while preserving UI command names.
- `src/fisheye/registry/maintenance.py` backfills presence-level status rows
  for the derived-analysis families above.
- Tail behavior backfill now includes source-ref freshness checks for:
  - `tail_kinematics_runs` against current `subject_shape`
  - `tail_posture_view_runs` against current `subject_shape` and, when
    declared, current `tail_kinematics`
  - `bout_classification_runs` against current `tail_posture_view`,
    `track_kinematics`, and `swim_bouts`
- `recording_step_status_wide` and `src/fisheye/status_page/query.py` expose
  those derived-analysis stages.
- Tail behavior freshness is display-visible in `recording_step_status_wide`:
  stale source refs render as `STALE`; unverifiable or missing source refs
  render as `UNVER`.
- `analysis/bout_classification_runs` is implemented and documented by
  `docs/bout_classification_runs_contract.md`.

## Docs Updated In This Pass

- `docs/stimulus_response_implementation_plan.md`
  - Replaced stale wording that implied derived-analysis registry stages still
    needed to be added from scratch.
  - Clarified remaining work: formal runner integration, writer-side status
    emission, and source-ref freshness.
- `docs/doc_code_divergence_inventory_2026-05-01.md`
  - Added an inline 2026-05-10 status note to the original stale item about
    derived-analysis registry absence.
- `docs/stage_catalog_design.md`
  - Clarified that tail/posture/classification families were deliberately not
    part of the first follow-up slice.
- `docs/tail_kinematics_tool_interop_design.md`
  - Marked classifier-only Megabouts integration, bout-classification contract,
    label/config/source-ref storage, and independent classifier outputs as done.
- `docs/tail_kinematics_run_design.md`
  - Marked the Palette-owned optional Megabouts classifier adapter and
    `analysis/bout_classification_runs` contract as done.

## Remaining Stale Or Incomplete Areas

These are not simple doc corrections; they need design or code decisions.

- Most derived-analysis registry coverage is still presence-level. The tail
  behavior slice compares stored source refs against current upstream run
  selections, but track kinematics, swim bouts, bout kinematics, eye angles,
  subject shape, stimulus response, and lineage fingerprints still need
  equivalent freshness semantics.
- Derived-analysis writers mostly do not upsert their own
  `recording_step_status` rows. Registry backfill can discover their presence,
  but live writer-side status emission is still uneven.
- `analysis/speed_runs` from `compute_speed.py` remains a legacy parallel
  surface. Current docs correctly direct new work to
  `analysis/track_kinematics_runs`; no code migration was attempted here.
- Several schema-version fields remain metadata-only rather than enforced
  reader gates. This is intentional until each reader has a compatibility
  policy.

## Recommended Next Slice

Broaden semantic freshness for derived runs:

1. Extend source-ref freshness to the remaining derived-analysis families.
2. Add source revision or lineage-fingerprint comparison where writers expose
   those fields.
3. Extend `STALE`/`UNVER` display semantics beyond tail behavior once other
   derived families gain source freshness checks.
4. Add writer-side `recording_step_status` upserts after the freshness
   semantics are settled.
