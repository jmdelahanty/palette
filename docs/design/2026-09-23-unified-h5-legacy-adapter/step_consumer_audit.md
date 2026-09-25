# Interrupted-run step consumer audit (D5), 2026-09-23

_Supporting evidence for design decision D5 in [README.md](README.md). Agent-generated read-only audit, 2026-09-23; the interrupted GoodCopBadCop run and the `steps[0]` reads were spot-checked, the rest is as reported._

Scope: read-only audit of main @ b525e040 (PR 186 head) plus read-only
queries of the canonical registry (`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`)
and one analysis zarr. Nothing was modified.

Tags: **[V]** verified by reading code or by querying the registry/store. **[I]** inferred (not executed).

## 0. What the store holds today (verified)

- No stimulus run in the registry carries `protocol_execution_status` or `protocol_semantic_status`: every
  latest run is **legacy** (no execution index, no semantic snapshot). So the v2 and interrupted paths have
  never run against real data. [V]
- Every latest run has materialized step count equal to the recipe count. The only mismatch is 12
  **non-latest** GoodCopBadCop runs with 0 steps against a 1-step recipe. [V]
- **The store already holds interrupted legacy sessions, but they are not flagged.**
  `2026-07-02T15-06-50Z_arena_{1..4}_GoodCopBadCop` has events `PROTOCOL_STOP@85607` with no
  `PROTOCOL_FINISH` and no `STEP_END`. `CHASER_POST_PERIOD_START@78742`, so the post period stopped about
  69 s in. The legacy materializer turned this into `step_0 = [741, 742)` (1 frame) with
  `duration_s = 1380` (the recipe duration). These zarrs have no `stimulus_epoch_runs` yet. [V]
- A related existing failure, not caused by interruption: 7 Blindfish recordings (2026-03-27/28) have all
  111 steps collapsed to 1-frame spans at camera frames 0 to 3 (the camera-frame column is broken). Consumers
  treat them as valid. [V]

## Summary counts (27 consumers / consumer sites)

| Class | Count |
|---|---|
| SAFE | 7 |
| SILENT-WRONG | 11 |
| CRASH | 7 |
| UNKNOWN | 2 |

Several sites fall into more than one class; each is counted under its dominant class.

## Table

Index/name column: **IDX** = keyed on step index or recipe position, **NAME** = keyed on mode, role or label,
**ITER** = iterates whatever was materialized. **RECIPE** marks sites that read `protocol_json` or snapshot
recipe steps instead of materialized steps.

| # | Consumer (file:line) | Class | Key | Behaviour on an interrupted run |
|---|---|---|---|---|
| 1 | `analysis/import_stimulus_to_zarr.py:1777-1784` (`_materialize_stimulus_steps` gate) | CRASH (by design) | — | Refuses a non-`complete` or prefix execution. This gate is what D5 removes. [V] |
| 2 | `import_stimulus_to_zarr.py:1830-1860,1839-1854` legacy path (no execution index) | SILENT-WRONG | ITER | Materializes `sorted(starts)` only, so a prefix is silently allowed. A missing `STEP_END` becomes `end = start+1` (a 1-frame step). No completion flag is written. Already present in the store (see section 0). [V] |
| 3 | `import_stimulus_to_zarr.py:1862-1866,1934-1948` `duration_s` | SILENT-WRONG | IDX→RECIPE | `duration_s` on a step always comes from the recipe (`protocol_step.duration_seconds` or `semantic_identity.duration_s`), and the semantic check *forces* it to equal the recipe. An interrupted step therefore claims its full recipe duration, and every downstream reader of `duration_s` inherits that. [V] |
| 4 | `import_stimulus_to_zarr.py:1152-1160` `_validate_protocol_semantic_steps` / `_bind_protocol_semantic_steps:1220` | CRASH | IDX | `step_names != {step_i for every recipe step}` raises. The D5 change has to relax this at the same time as the gate. [V] |
| 5 | `shared/protocol_semantic_contract.py:756-812` `read_materialized_protocol_semantic_snapshot` | CRASH | IDX (recipe) | Requires the exact recipe step set and iterates `snapshot.steps`, then `steps_group[f"step_{i}"]`. This is the hub reader, so every semantic consumer fails. [V] |
| 6 | `shared/protocol_execution_contract.py:269-368` `validate_protocol_execution_index` | SAFE | IDX (prefix-checked) | Accepts a prefix plus `interrupted`, and enforces a prefix of the recipe. **But** an interrupted CHASER step still carries all three `chaser_phases` partitioning the truncated interval, so `chaser_post` can be short or empty (`is_empty`) and is not flagged per phase. The truncation is pushed down to consumers. [V] |
| 7 | `registry/extractors/stimulus_metadata.py:299-340` `_materialized_semantic_state` | CRASH | IDX (recipe) | Duplicate of #5 (exact step set). The whole recording's stimulus extraction fails, so the registry gets no run, step or mode rows for it. [V] |
| 8 | `stimulus_metadata.py:412-470` `_materialized_execution_state` | SAFE | ITER (realized) | Validates only the realized steps and checks `execution_completion_status` per step. [V] |
| 9 | `stimulus_metadata.py:549-660` `_materialized_steps` → `recording_stimulus_steps` | SAFE (rows) | ITER | Emits rows for the materialized steps only and carries `execution_completion_status`. The `duration_s` column inherits the recipe duration from #3. [V] |
| 10 | `stimulus_metadata.py:869-885` → `recording_stimulus_mode_counts.total_duration_s` | SILENT-WRONG | NAME (mode) | Sums the per-step `duration_s`, which is the recipe duration, so an interrupted CHASER reports 1380 s. It already does this for the 4 July-2 sessions. [V] |
| 11 | `stimulus_metadata.py:692-713,852` protocol hash and step_count | SILENT-WRONG | RECIPE | `protocol_hash` is computed from `protocol_json` (the recipe), so interrupted and complete runs share one hash. `stimulus_protocols.step_count` is the recipe count while `recording_stimulus_runs.step_count` is the materialized count, and nothing compares them. [V] |
| 12 | `cohorts/registry.py:160-199` `_stimulus_context` | SILENT-WRONG | NAME (mode/hash) | Cohort membership by mode or protocol_hash puts interrupted runs in with complete runs. A run whose CHASER never started drops out with no recorded reason. [V code; I impact] |
| 13 | `utils/build_virtual_collection_manifest.py:570-640` | SILENT-WRONG | NAME (mode) | Selects "has mode X" and publishes `selected_mode_total_duration_s` from #10 (the recipe duration). [V] |
| 14 | `analysis_workflows/protocol_semantic_chaser_selection.py:690,752-786,412-422,823-846` | CRASH | IDX (recipe) | Goes through #5. It also has an explicit `raw_execution.status != "complete"` check (:780) and `step_bounds must bind every semantic recipe index` (:417). In the v1 branch, `steps_group[f"step_{i}"]` at :824 raises a bare KeyError. `_semantic_steps:1007` picks the CHASER step from **recipe** `snapshot.steps` by mode. `capability_assessments:1289-1310` always returns `READY` for chaser windows, so it cannot express exclusion. [V] |
| 15 | `analysis_workflows/historical_protocol_semantic_stimulus_successor.py:385,426` | CRASH | IDX | Bind plus reload through #4 and #5. [V] |
| 16 | `utils/export_provider_epoch_behavior_cohort.py:620-720` | CRASH (upstream) | NAME (role) | Needs a semantic selection manifest, which #14 cannot produce. If #14 were relaxed without role dispositions, truncated roles would be carried through silently (bounds are compared exactly, not checked for completeness). [V code; I relaxed behaviour] |
| 17 | `analysis/stimulus_epoch_runs.py:170-240` plus `chaser_profiles.py:580-650` `resolve_profile_windows` (the real epoch source for legacy runs) | SILENT-WRONG / CRASH | NAME (event alias) | If `CHASER_TRAINING_START` or `CHASER_POST_PERIOD_START` is missing, it raises ValueError. If the session stopped in post, the `finish` alias chain `PROTOCOL_FINISH, PROTOCOL_STOP, STEP_END, …` resolves to `PROTOCOL_STOP` **with no note in `source_policy`** (notes appear only for configured fallbacks), and a truncated `post_event` is written as a normal window. For July-2 arena_1 this would be post = [78742, 84866], about 61 s of a much longer recipe post. `event_frames.setdefault` keeps the *first* `STEP_END`, which in a multi-step protocol can come before `post_start` and collapse post to 1 frame [I]. |
| 18 | Window-label consumers: `detection_occupancy_runs.py:184`, `chaser_distance_runs.py:432-460`, `chaser_escape_events.py:273-312,732`, `chaser_bout_response.py`, `chaser_epoch_behavior_summary.py:156-170`, `epoch_segments.py:50-200` | SILENT-WRONG | NAME (label), ITER | They iterate whatever windows exist. A truncated `post_event` or `training_event` looks exactly like a full one. Rates use realized frames (arithmetically correct), but pre/post contrasts compare unequal exposure, and a truncated training period has fewer chases. None of them normalize by recipe duration (grep found none). [V code; I impact] |
| 19 | `analysis/chaser_quadrant_occupancy.py:500-545` | SAFE-ish / CRASH | NAME | Raises if `pre_event` or `post_event` is missing, and raises if post is shorter than `post_settle_duration_s`. Otherwise it silently uses a truncated post. [V] |
| 20 | `analysis/chaser_near_field_occupancy.py:426-445` | CRASH | NAME | Missing `PHASE_LABELS` raises. Truncated phases pass silently. [V] |
| 21 | `analysis/goodcopbadcop_common.py:274-300` `load_epochs` | CRASH / SILENT-WRONG | NAME (substring) | The dict simply lacks a missing epoch, so callers doing `ep["post"]` raise KeyError. Truncated epochs pass. The docstring says: "On these recordings all three windows are present." [V] |
| 22 | `analysis/analyze_goodcopbadcop_habituation.py:71-95` | SILENT-WRONG (unrecorded exclusion) | NAME (trial ordinal) | Sessions with no late ordinals are dropped from the paired test with no reason recorded, and `except Exception: print("skip")` swallows loader errors. [V] |
| 23 | `group_statistics/paired.py:205-230` (`compute_paired_contrast`), `goodcopbadcop.py:1355` | SAFE (partial) | NAME | Records `excluded_unit_count` under `missing_policy="paired_complete_recordings"`: a count, with no per-recording reason. Truncated phases are included. [V] |
| 24 | `analysis/stimulus_response.py:615-648` `_parse_canonical_stimulus_steps` (+ `:1048` IBI) | SILENT-WRONG | ITER | For missing steps it iterates only what was materialized (fine). **Existing bug:** v2 steps have no `start_camera_frame`/`end_camera_frame`, so defaults of 0 and 1 are used and **every v2 step window becomes [0,1)**, complete or not. A legacy missing `STEP_END` also gives a 1-frame step. `mean_ibi` scales by `step.duration_s/step_frames` using the recipe duration, so IBIs on a truncated step are inflated. The event fallback at `:700-760` uses frame-derived duration and does not have the IBI problem. [V] |
| 25 | `utils/export_cross_recording_analytics.py:1055-1190` | SILENT-WRONG (minor) | ITER | The signature is built from materialized steps, so interrupted runs get a distinct `derived_protocol_hash` (good). But `protocol_duration_sequence_s` holds recipe durations and `protocol_step_count` holds the *executed* count under a "protocol" name. v2 steps have `None` frames, so `_assign_step` drops every bout. [V] |
| 26 | `reporting/discovery.py:234-270`, `stimulus_response_io.py:280-360`, `plot_stimulus_response_omr.py:163-204`, `shared/unified_h5/appearance.py:55-120`, `shared/unified_h5/protocol.py:15-65`, `analysis_workflows/controller_trial_successor.py:337-474` | SAFE | ITER / realized | They iterate only materialized or realized steps, or observed trials. appearance and protocol explicitly accept `interrupted`. The discovery display inherits the recipe `duration_s`. Controller trials count observed trials, so later-ordinal group stats (`validated_behavior_specs.py:700`) get unbalanced n with no flag [I]. [V] |
| 27a | Recipe-param readers: `chaser_behavior.py:71`, `chaser_egocentric_bearing.py:176`, `chaser_escape_freeze_summary.py:338`, `visualization/goodcopbadcop_interactive.py:418`, `visualization/chaser_appearance.py:206`, `chaser_quadrant_occupancy.py:262`, `goodcopbadcop_common.py:314`, `utils/export_protocol_mermaid.py:254` | UNKNOWN (benign if the step ran) | RECIPE, NAME (first step with `chasers`) | They read static chaser config from `protocol_json`. This is fine when the chaser step executed. When it did not, they still resolve roles and colours and fail or misbehave further downstream. Mermaid draws the full recipe. [V] |
| 27b | `chaser_radial_occupancy.py:309` and `chaser_response_regimes.py:200` | UNKNOWN (existing IDX bug) | **IDX `steps[0]`** | `steps[0]["parameters"]`. goodbatbadbat's step 0 is SOLID_BLACK (registry-verified), so it silently returns a 0.0 settle time and NaN radii today. This is independent of interruption. [V] |
| — | `shared/unified_h5/admission.py:118-124` `declared_appearance_missing` | (counted in UNKNOWN 27a) | RECIPE | If the authored recipe declares appearance for a step that never executed, admission depends on whether Citrus writes the appearance witness for unexecuted steps. [I] |
| — | viz: `plot_detection_epoch_heatmaps.py:185-225`, `plot_training_heatmaps_zarr.py:169` | (in #17 family) | event alias | Same `PROTOCOL_STOP`/`STEP_END` or `max_cam` fallback, so post is truncated silently. Visualization only. [V] |

## Top risks

1. **Truncated chaser phases are not flagged per phase.** Citrus always emits all three `chaser_phases`
   that partition the truncated CHASER interval (#6). The legacy event path also silently picks
   `PROTOCOL_STOP` (#17). Every window-label consumer (#18) then computes "post" or "training" on partial
   exposure and cannot tell. The protocols that matter most (GoodCopBadCop, RedScare, Batman, goodbatbadbat)
   have **1 or 2 steps**, so pre/train/post are phases *inside* one step. The realistic interrupted case is
   therefore "same step count, truncated phase", not "fewer steps". A step-count check alone misses it.
2. **`duration_s` on a step is always the recipe duration** (#3). It feeds registry
   `total_duration_s` (#10), virtual manifests (#13), cross-recording signatures (#25) and stimulus_response
   IBI scaling (#24). This already happens for the 4 July-2 sessions.
3. **Cohort identity comes from the recipe** (#11, #12). The protocol_hash from `protocol_json` puts
   interrupted and complete runs in the same cohort, and "mode present" selection admits a CHASER step that
   ran for 1 frame.
4. **The hub validators crash** (#4, #5, #7, #14, #15). Relaxing only the importer gate would make every
   interrupted v2 run unreadable by the registry and by semantic chaser selection. That fails closed, which is
   better than silent, but the registry would lose the run entirely instead of recording why it was excluded.
5. **stimulus_response reads v2 steps as [0,1)** (#24). This is an existing bug for *all* execution-indexed
   runs. Relatedly, the 7 Blindfish recordings already have collapsed 1-frame steps that are consumed without
   complaint.

## Existing helpers to reuse (question 1)

No helper exists for "require steps X". The nearest pieces:
- `analysis_workflows/chaser_profile_applicability.py`: `CapabilityAssessment(capability_id, state:
  CapabilityState{READY, NOT_APPLICABLE, MISSING, INVALID, REVIEW_REQUIRED, STALE}, reason_code, evidence)`,
  `plan_chaser_profile_applicability(...)` and `ModuleApplicabilityDecision` (records
  `BLOCKED_MISSING_CAPABILITY` / `INAPPLICABLE` together with a reason code). Profile YAML modules already
  declare `required_capabilities:` (`profiles/chaser_behavior_full_v4.yaml:28-70`, parsed at
  `chaser_profiles.py:306-374`). **This is the exclusion-with-recorded-reason mechanism D5 asks for.**
- `ProtocolSemanticChaserSelections.capability_assessments()` (`protocol_semantic_chaser_selection.py:1289`)
  is where chaser-window capabilities are produced today. They are unconditionally `READY`.
- Parallel pattern: `validated_recording_behavior_bundle.py:48-75` has `{state, reason_code, detail}`
  dispositions.
- Constraint: `shared` must not import `analysis_workflows` (`.importlinter` layers contract, line 114).
  The pure coverage computation therefore goes in `shared`, and the projection to `CapabilityAssessment` goes
  in `analysis_workflows`.

## Proposed interface (question 2)

Pure core, in `fisheye/shared/protocol_execution_contract.py`. Extend the existing module rather than adding
a new one:

```python
@dataclass(frozen=True)
class ProtocolStepRequirement:
    requirement_id: str                 # "chaser_pre" | "chaser_training" | "chaser_post" | "solid_black_baseline" ...
    stimulus_mode: str                  # match by mode/family/display_context, never by index
    stimulus_family: str | None = None
    display_context: str | None = None
    chaser_phase: str | None = None     # one of CHASER_PHASE_NAMES
    min_realized_fraction: float = 1.0  # realized/recipe frames; 1.0 == must complete

@dataclass(frozen=True)
class ProtocolStepCoverage:
    requirement_id: str
    status: Literal["satisfied", "not_executed", "interrupted", "truncated",
                    "empty", "not_in_recipe", "legacy_unverifiable"]
    step_index: int | None
    recipe_duration_s: float | None
    realized_interval: Mapping[str, int] | None
    run_execution_status: str | None    # complete | interrupted | None (legacy)

def assess_protocol_step_requirements(
    run_group, requirements: Sequence[ProtocolStepRequirement]
) -> tuple[ProtocolStepCoverage, ...]: ...
```

For legacy runs, `legacy_unverifiable` or `interrupted` is derived from a missing `STEP_END`, or from
`PROTOCOL_STOP` without `PROTOCOL_FINISH`.

Projection, in `analysis_workflows`. `capability_assessments()` maps each coverage to
`CapabilityAssessment(capability_id=f"protocol_step.{requirement_id}", state=READY|MISSING|NOT_APPLICABLE,
reason_code=status, evidence=...)`. Profile modules then declare, for example,
`required_capabilities: [protocol_step.chaser_pre, protocol_step.chaser_post]`, and the existing planner
records `BLOCKED_MISSING_CAPABILITY` with the reason. Non-profile scripts (goodcopbadcop_common, habituation,
group stats) call `assess_protocol_step_requirements` directly and write an exclusion row
`{recording_id, requirement_id, status}`.

Also: stop writing the recipe duration into a step's `duration_s`. Split it into `recipe_duration_s` and a
realized duration or interval.

## Files needing change

Required for D5 correctness:
- `src/fisheye/analysis/import_stimulus_to_zarr.py` (gate :1777, :1152-1160 validator, `duration_s` :1862-1866/:1934, legacy `end=start+1` :1854)
- `src/fisheye/shared/protocol_semantic_contract.py` (:756-812, allow a realized prefix bound to the execution index)
- `src/fisheye/shared/protocol_execution_contract.py` (add the coverage helper, per-phase truncation status)
- `src/fisheye/registry/extractors/stimulus_metadata.py` (:299-340 relax, :869-885 realized duration, :692-713 add an execution-status dimension to cohort identity)
- `src/fisheye/analysis_workflows/protocol_semantic_chaser_selection.py` (:417, :780, :823, `_semantic_steps` :1007, `capability_assessments` :1289)
- `src/fisheye/analysis_workflows/historical_protocol_semantic_stimulus_successor.py`
- `src/fisheye/analysis/chaser_profiles.py` (`resolve_profile_windows`: record `PROTOCOL_STOP`/`STEP_END` use as a truncation note, fix first-`STEP_END` selection)
- `src/fisheye/analysis/stimulus_epoch_runs.py` (persist per-window completeness)
- `src/fisheye/analysis/profiles/*.yaml` (declare `protocol_step.*` required capabilities)
- `src/fisheye/cohorts/registry.py`, `src/fisheye/utils/build_virtual_collection_manifest.py` (filter or flag interrupted runs)
- `src/fisheye/analysis/goodcopbadcop_common.py`, `analyze_goodcopbadcop_habituation.py`, `chaser_quadrant_occupancy.py`, `chaser_near_field_occupancy.py` (explicit required epochs plus a recorded exclusion)
- `src/fisheye/utils/export_provider_epoch_behavior_cohort.py` (consume role dispositions)

Existing bugs found along the way (independent of D5):
- `src/fisheye/analysis/stimulus_response.py:615-648` (v2 steps read as [0,1); recipe `duration_s` in IBI)
- `src/fisheye/utils/export_cross_recording_analytics.py:1076-1085` (v2 steps have no frames, so bouts are unassigned)
- `src/fisheye/analysis/chaser_radial_occupancy.py:309`, `chaser_response_regimes.py:200` (`steps[0]` index bug on goodbatbadbat)
- `src/fisheye/visualization/plot_detection_epoch_heatmaps.py:185-225`, `analysis/plot_training_heatmaps_zarr.py:169` (silent finish fallback)
