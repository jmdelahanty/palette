# Zarr run-completion: retire legacy_default=True (strict-mode convergence)

Status: in progress
Date: 2026-06-10
Source: codebase review (docs/diagnostics/codebase_review_2026-06-10.md, Provenance & contracts)

## 2026-08-16 metadata-lifecycle enforcement

- New analysis stores are stamped with root `palette_store_epoch=1`. In those
  stores, a run parent without `palette_completion_epoch` is an error; absence
  can no longer be interpreted as completeness.
- Existing unstamped stores retain explicit legacy compatibility until they are
  verified and migrated. Accepted unmarked runs are counted once per process
  and reported together at process exit.
- Completion backfill apply mode requires `--metadata-lifecycle mutable` or
  `--metadata-lifecycle published-immutable`. Mutable stores remain on direct
  metadata. Published immutable stores reconsolidate only after all attr writes
  and validate direct/consolidated equivalence for every stamped parent.
- CI ratchets all `zarr.open_group` calls under `src/fisheye`: existing implicit
  calls are grandfathered, but every new or modified call must select direct or
  consolidated metadata explicitly.

## The problem

`is_run_complete` (src/fisheye/shared/zarr_run_completion.py:102-114) answers
"is this run complete?" with `legacy_default` when a run group carries **no
completion markers at all**. At plan creation every one of the **84 call
sites** in src/fisheye passed `legacy_default=True` (verified by grep
2026-06-10; zero passed False). Phase 1/3's first implementation slice removed
the registry completion gate's explicit override, lowering the count to 83.
The strict mode promised by the module docstring ("once all active writers
emit completion attrs, callers can switch") has not yet been fully engaged,
even though 19 writer modules now call
mark_run_started/mark_run_complete.

Consequences:

- **The fail-closed gate fails open.** `stage_complete.py:183` refuses to mark
  a registry step "ok" unless `is_run_complete(..., legacy_default=True)` — so
  an *unmarked* group passes the gate. The gate only protects against writers
  that adopted the contract and didn't finish.
- **Unmarked groups can win latest-resolution.** The fallback scan in
  `resolve_latest_complete_run_name` walks child names reverse-sorted
  (zarr_run_completion.py:161-164); any attr-less group (hand-made debug
  group, agent-written one-off, or hand-edited metadata that bypasses the
  completion helper) that sorts last becomes
  "latest complete" and is silently consumed by predict_pose, training
  exporters, and review tools.
- **Trust is unrecorded.** A consumer cannot distinguish "complete because
  verified" from "complete because old." This directly undercuts the
  data-sharing goal.

Principles (see also the review's cross-cutting theme #2):

1. On a trust question, "unknown" must not collapse into "yes" — fail closed.
2. The trust level must be a property of the **data**, not a parameter of the
   **reader**. 84 hardcoded call-site decisions is the reader deciding.
3. A compatibility default with no forcing function attached becomes
   permanent. This doc is the forcing function; each phase has a finite
   checklist.

## Design: parent-group completion epoch

Stamp the strictness level on each runs-parent group (e.g. `detect_runs`,
`crop_runs`, `speed_runs`) — not the store root, because resolvers already
hold `parent_group`, fake-group unit tests have no root, and both
training.zarr and analysis.zarr get covered without root-walking.

New attrs/constants in `zarr_run_completion.py`:

```python
COMPLETION_EPOCH_ATTR = "palette_completion_epoch"
COMPLETION_EPOCH_STRICT = 1   # epoch >= 1: unmarked child runs are NOT complete
```

Terminology: a **schema** says what a run's data must look like. In this
checklist that is the `StageSpec`/validator surface: required arrays, required
attrs, and exact attr values where needed. A **completion epoch** says what a
runs-parent means by "done." It is a versioned trust marker on the parent
group. Epoch 0/unstamped means legacy compatibility, where unmarked child runs
may still be treated as complete. Epoch 1 means strict mode, where only child
runs with explicit completion markers can be consumed as complete. These are
related but separate: schema validation is how the backfill tool decides
whether a legacy child is safe to mark complete; the epoch is the reader-facing
policy that prevents future unmarked children from being trusted.

New choke-point helper — the only sanctioned way to create/obtain a
runs-parent:

```python
def require_runs_parent(root: Any, name: str) -> Any:
    parent = root.require_group(name)
    if parent.attrs.get(COMPLETION_EPOCH_ATTR) is None and not _has_children(parent):
        # Stamp ONLY brand-new parents; pre-existing parents with unmarked
        # children must be verified by the backfill tool before upgrade.
        parent.attrs[COMPLETION_EPOCH_ATTR] = COMPLETION_EPOCH_STRICT
    return parent
```

Resolution change — `legacy_default` becomes three-valued:

```python
def effective_legacy_default(parent_group: Any) -> bool:
    epoch = parent_group.attrs.get(COMPLETION_EPOCH_ATTR)
    return not (isinstance(epoch, int) and epoch >= COMPLETION_EPOCH_STRICT)

def resolve_latest_complete_run_name(parent_group, *, latest_attr="latest",
                                     legacy_default: bool | None = None):
    if legacy_default is None:
        legacy_default = effective_legacy_default(parent_group)
    ...
```

`None` (the new default) means "ask the data." Explicit True/False remains
for tools that must override (the backfill verifier reads legacy stores with
True; nothing in production should pass True explicitly once migration ends).
When the legacy path actually decides an outcome (unmarked group treated as
complete), log one warning per parent so the exemption is visible while it
still exists.

Important API boundary: data-driven strictness is a property of the
**runs-parent**, not of an isolated child run. Parent-aware helpers such as
`resolve_latest_complete_run_name(parent_group, legacy_default=None)` can infer
the effective default from `parent_group.attrs`. Child-only calls to
`is_run_complete(run_group, legacy_default=None)` cannot infer the parent epoch
unless the caller also passes the parent or precomputes
`effective_legacy_default(parent_group)`. Do not hide that with a best-effort
parent lookup; make the parent context explicit in the call path.

In strict-epoch parents the lexicographic-scan hazard disappears
automatically: unmarked groups are excluded, so a stray `test_run` group can
never become latest.

## Migration plan

### Phase 0 — visibility + ratchet (hours; do immediately)

- [x] Add the legacy-path warning log to `is_run_complete` /
      `resolve_latest_complete_run_name` (one per parent per process).
- [x] Add a ratchet test: grep `src/fisheye` for `legacy_default=True`, assert
      count <= 84, and lower the literal as call sites migrate. New code can
      never add an explicit True without failing the test.

### Phase 1 — epoch plumbing (about a day)

- [x] Add COMPLETION_EPOCH_ATTR, effective_legacy_default,
      require_runs_parent to zarr_run_completion.py.
- [x] Change parent-aware resolver signatures to
      `legacy_default: bool | None = None` with data-driven resolution.
- [x] Keep child-only `is_run_complete` explicit, or add a parent-aware wrapper
      such as `is_run_complete_in_parent(parent_group, run_group,
      legacy_default=None)`. Do not let a child-only helper silently guess the
      parent epoch.
- [x] Unit tests: unmarked child under epoch-stamped parent is NOT complete;
      under unstamped parent IS complete (legacy); explicit override wins;
      stray unmarked group cannot win the fallback scan under strict epoch.

### Phase 2 — writers create strict parents (about a day, mechanical)

- [x] Replace every `require_group("*_runs")` in the 19 writer modules with
      `require_runs_parent(root, ...)` (grep inventory in this doc's source
      review; ~20+ sites across detection, segmentation, analysis, tune).
      Progress 2026-06-10: migrated the central `get_run_group` /
      `add_processing_run` helpers and the primary runtime writers for detect,
      background, keypoints, refined_detect, detect_quality, refined_keypoints,
      eye_masks, subject_masks, refined_eye_masks, refined_subject_masks,
      profile/export/review utilities, and the analysis-stage writers
      (`stimulus_runs`, `speed_runs`, `track_kinematics_runs`,
      `swim_bout_runs`, `bout_kinematics_runs`, `eye_angle_runs`,
      `subject_shape_runs`, `tail_kinematics_runs`,
      `tail_posture_view_runs`, `bout_classification_runs`, and
      `stimulus_response_runs`). Read-only resolvers that previously used
      `require_group` were changed to non-mutating `get` checks. Remaining raw
      grep hits are only `experiment_index/finalized_runs`, which is collection
      index metadata, not a stage run parent.
- [x] Add a second ratchet test: no raw `require_group("..._runs")` outside
      zarr_run_completion.py, with an explicit `finalized_runs` collection-index
      exception.
- [x] From this point, every NEW runs-parent on every store is strict.
- [x] Make `_has_children(parent)` conservative: if child listing fails or is
      ambiguous, do **not** stamp the parent strict. A false negative merely
      keeps a new parent legacy until touched again; a false positive can make
      legacy children disappear from strict readers.

### Phase 3 — call-site migration (mechanical; good parallel-agent task)

- [x] Replace `legacy_default=True` with omission (-> None/data-driven) at all
      production sites. stage_complete.py was migrated first, then resolver
      call sites across shared readers, training/detection/review utilities,
      diagnostics, segmentation, inference, registry maintenance, and
      visualization. Child-only `is_run_complete(...)` checks were converted to
      `is_run_complete_in_parent(parent, child)` before removing the explicit
      override.
- [x] For stage_complete specifically, return or resolve the runs-parent along
      with the child run group so the completion check can use
      `effective_legacy_default(parent_group)` instead of a child-only guess.
- [x] Ratchet the Phase-0 test to 0 production occurrences. Unit tests still
      exercise explicit override behavior.

### Phase 4 — verify-and-stamp backfill of existing stores (batch job)

- [x] New tool `utils/backfill_completion_epoch.py` (read-mostly, dry-run/apply
      per repo idiom): for each recording store and each runs-parent, verify
      every child run group either carries the contract or is positively
      identified as a legacy-complete run (arrays validate against
      stage_arrays spec where one exists); for every verified legacy-complete
      child, write explicit `palette_run_completion_*` attrs first; stamp the
      parent epoch only after all trusted children have explicit completion
      markers; emit a JSON report of parents that cannot be auto-verified.
      Stamping the parent without first marking verified legacy children would
      make those children incomplete under strict readers.
- [x] Compatibility policy for old non-current children: if a child has a
      known StageSpec but fails the current schema, and that child is neither
      the parent `latest` nor `latest_complete`, the backfill tool may stamp
      the parent strict without marking that child complete. The child is
      reported with `ignored_for_parent_epoch=true` and remains unmarked, so
      strict readers ignore it. This is intentionally **not** a validator
      relaxation: an invalid `latest` / `latest_complete` child still blocks,
      and no-spec families still block or remain explicitly deferred.
- [x] All attr writes go through the zarr API (`group.attrs[...] = ...`), NOT
      raw zarr.json/.zattrs JSON edits. Do not copy the
      `audit_zarr_pixel_contracts._set_node_attrs` pattern: its unlocked
      read-modify-write of the JSON file can lose concurrent attr updates and
      desync consolidated metadata. (Its merge-only/no-overwrite behavior is
      fine; the raw-file transport is the hazard.) Do not run the backfill
      concurrently with pipeline writers on the same store. The apply path is
      conservative per parent: if writing any verified child completion marker
      fails, the parent epoch must not be stamped. A rerun should remain
      idempotent and self-healing for parents whose children were marked but
      whose parent epoch was not stamped. Apply reports now include
      `write_failed_parent_count`; the CLI exits nonzero after writing the
      structured JSON report if any parent write fails. Summary reports also
      group write failures by stage/path and include compact write-failure
      examples, so `--summary-only` remains triageable. Operators can pass
      `--write-failed-jsonl` to persist one compact row per write-failed
      parent, separate from validation blockers in `--blocked-jsonl`.
- [ ] Run across /nvme1/recordings; triage the unverifiable remainder by hand
      (mark stray groups failed, or delete debris).
- [ ] Record completion stats here after the apply/triage pass.
      Current status as of 2026-06-14: staged applies have modified archive
      attrs under `/nvme1/recordings`. Active non-eye StageSpec scopes have now
      been strict-epoch backfilled, including the compatibility-policy pass
      that ignores invalid non-latest legacy children. Remaining unresolved
      scopes are the deferred no-spec analysis families and deprecated eye-mask
      families. Unless a command below explicitly says apply, numbers are
      read-only dry-runs against `/nvme1/recordings`.

      Compatibility policy update recorded 2026-06-14: active blockers were
      reviewed and none of the active compatibility failures were the current
      `latest` run for their parent. The backfill tool now treats invalid
      non-latest children as ignored legacy debris for epoch-stamping
      purposes, reports `ignored_legacy_child_count`, and supports
      `--expect-ignored-legacy-child-count` for guarded apply runs.

      Guarded active non-eye apply recorded 2026-06-14:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage crop --stage detect --stage keypoints --stage refined_detect --stage refined_keypoints --stage refined_subject_masks --apply --expect-store-count 138 --expect-non-ok-store-count 0 --expect-blocked-parent-count 0 --expect-would-stamp-parent-count 65 --expect-would-mark-child-count 120 --expect-ignored-legacy-child-count 74 --expect-write-failed-parent-count 0 --expect-applied-stamped-parent-count 65 --expect-applied-marked-child-count 120 --output-json /tmp/completion_epoch_active_non_eye_after_ignore_policy_apply.json --blocked-jsonl /tmp/completion_epoch_active_non_eye_after_ignore_policy_apply_blocked.jsonl --write-failed-jsonl /tmp/completion_epoch_active_non_eye_after_ignore_policy_apply_write_failed.jsonl --no-stdout`
      completed successfully. It scanned 138 stores, wrote 65 strict parent
      epochs, marked 120 verified legacy children complete, left 74 invalid
      non-latest legacy children unmarked/ignored, and reported 0 blocked
      parents and 0 write failures. Stamped parent counts were
      `refined_detect=55`, `detect=5`, `crop=2`, `keypoints=1`,
      `refined_keypoints=1`, and `refined_subject_masks=1`.

      Post-apply full verification recorded 2026-06-14:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --summary-only --output-json /tmp/completion_epoch_backfill_post_active_non_eye_apply_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_post_active_non_eye_apply_blocked.jsonl --no-stdout`
      scanned 138 stores with 138 ok stores and 0 non-ok stores. Remaining
      summary: `blocked_parent_count=210`, `would_stamp_parent_count=8`,
      `would_mark_child_count=0`, `ignored_legacy_child_count=217`,
      `ignored_legacy_parent_count=171`, and `write_failed_parent_count=0`.
      Remaining blockers are only `analysis/stimulus_response_runs` (52
      no-spec parents), `analysis/swim_bout_runs` (52 no-spec parents),
      `eye_masks` (53 deprecated/latest-invalid parents), and
      `refined_eye_masks` (53 deprecated/latest-invalid parents). The only
      remaining stampable parents are deprecated eye-mask scopes:
      `eye_masks=4` and `refined_eye_masks=4`.

      Initial dry-run stats recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --output-json /tmp/completion_epoch_backfill_dry_run.json --no-stdout`
      scanned 138 zarr stores successfully. It found 658 parent groups that
      could be auto-stamped and 1174 blocked parent groups. This baseline
      exposed real spec/writer drift rather than bad epoch logic.

      StageSpec compatibility pass recorded 2026-06-10: demoted
      compatibility/legacy-only arrays that current writers do not always
      produce (`background.frame_indices`, keypoint triangle diagnostics,
      copied/legacy refined-keypoint fields, and `arena_assignment.confidence`).
      This changed only verifier expectations; it did not mutate archive data.

      Stimulus import contract added 2026-06-10: `STIMULUS_SPEC` now validates
      the minimal invariant import substrate under
      `analysis/stimulus_runs/<run>`: `interpolation_mask`,
      `video_metadata/frame_metadata/stimulus_frame_num`, and
      `frame_alignment/camera_to_metadata_index`. This intentionally does not
      attempt to validate every optional protocol/tracking/calibration subtree.
      A read-only representative probe validated this contract against all 119
      blocked legacy stimulus child runs present in the blocker sidecar.

      Keypoint profile contract added 2026-06-10: `KEYPOINT_PROFILE_SPEC` now
      validates attrs-only profile runs under `analysis/keypoint_profile_runs`
      using the documented required attrs `schema_name`, `schema_version`,
      `created_at_utc`, and `profile_summary`. The keypoint profile writer now
      creates the parent through `require_runs_parent` and marks new profile
      runs started/complete. A read-only representative probe validated the
      attr contract against all 110 legacy keypoint profile child runs present
      in the blocker sidecar.

      Detection profile contract added 2026-06-10:
      `DETECTION_PROFILE_SPEC` now validates attrs-only profile runs under
      `analysis/detection_profile_runs` using the documented required attrs
      `schema_name`, `schema_version`, `created_at_utc`, and
      `profile_summary`. The detection profile writer now creates the parent
      through `require_runs_parent` and marks new profile runs
      started/complete. A read-only real-zarr probe validated the attr contract
      against all 169 legacy detection profile child runs present in the
      blocker sidecar; all carried `schema_name = detection_dataset_profile`,
      `schema_version = v1`, and dict `profile_summary` payloads.

      Eye mask profile contract added 2026-06-10:
      `EYE_MASK_PROFILE_SPEC` now validates attrs-only profile runs under
      `analysis/eye_mask_profile_runs` using the documented required attrs
      `schema_name`, `schema_version`, `created_at_utc`, and
      `profile_summary`. The eye mask profile writer now creates the parent
      through `require_runs_parent` and marks new profile runs
      started/complete. A read-only real-zarr probe validated the attr contract
      against all 167 legacy eye mask profile child runs present in the
      blocker sidecar; all carried `schema_name = eye_mask_dataset_profile`,
      `schema_version = v1`, and dict `profile_summary` payloads.

      Subject shape contract added 2026-06-10: `SUBJECT_SHAPE_SPEC` now
      validates the minimal invariant surface under
      `analysis/subject_shape_runs`: required provenance attrs
      (`schema_id`, `schema_version`, `method`, `method_version`,
      `created_at_utc`, `row_axis`, `source_refined_subject_masks_run`,
      `source_refs`), `row_index/frame_indices`,
      `row_index/source_refined_row_ids`, and core
      `components/subject_body` row-aligned geometry arrays. Optional
      body-frame and source-row-revision arrays are intentionally not required
      because older legacy children omit them. A read-only real-zarr probe
      validated this contract against all 56 legacy subject-shape child runs
      present in the blocker sidecar with zero validation errors.

      Tail posture view contract added 2026-06-10:
      `TAIL_POSTURE_VIEW_SPEC` now validates the tool-compatible posture view
      surface under `analysis/tail_posture_view_runs`: required provenance and
      convention attrs, `valid`, `failure_reason_bytes`, `frame_index`,
      `head_xy`, `head_yaw_rad`, `tail_keypoints_xy`, `tail_angle_rad`,
      `tail_angle_deg`, and required row-lineage arrays under `row_index`.
      A read-only real-zarr probe validated this contract against all 48
      legacy tail-posture-view child runs present in the blocker sidecar with
      zero missing attrs and one consistent array layout.

      Bout classification contract added 2026-06-10:
      `BOUT_CLASSIFICATION_SPEC` now validates the Megabouts classifier output
      surface under `analysis/bout_classification_runs`: required classifier
      attrs and the `per_bout` columnar table fields used by
      `bout_classification_runs.validate_bout_classification_run`. A read-only
      real-zarr probe validated this contract against all 49 legacy
      bout-classification child runs present in the blocker sidecar with zero
      missing attrs and one consistent `per_bout` array layout.

      Tail kinematics contract added 2026-06-10:
      `TAIL_KINEMATICS_SPEC` now validates the tail metrics surface under
      `analysis/tail_kinematics_runs`: required provenance attrs,
      `valid`, `failure_reason_bytes`, `frame_index`,
      tail-angle sample and summary arrays, and required row-lineage arrays
      under `row_index`. A read-only real-zarr probe validated this contract
      against the one legacy tail-kinematics child run present in the blocker
      sidecar.

      Track kinematics contract added 2026-06-10:
      `TRACK_KINEMATICS_SPEC` now validates nested track kinematics runs under
      `analysis/track_kinematics_runs/<run_type>/<run>`. The completion
      resolver and backfill tool now handle slash-qualified child names such as
      `offline/<run>` explicitly. The minimal invariant surface is
      `track_ids`, `track_arena_ids`, and core provenance/summary attrs
      observed across the real stores. A read-only real-zarr probe validated
      this contract against the 52 legacy track-kinematics parents that were
      previously no-spec blocked.

      Eye angle contract added 2026-06-10:
      `EYE_ANGLE_SPEC` now validates the legacy-compatible eye-angle surface
      under `analysis/eye_angle_runs`. The invariant is intentionally narrower
      than the newest writer surface: exact `status == "complete"`, common
      provenance/source attrs, and row-aligned `support` arrays
      (`frame_indices`, `time_seconds`, `ellipse_major`, `ellipse_minor`,
      `ellipse_ratio`). A first draft also required `support/frame_time_seconds`,
      but a read-only dry-run showed one valid legacy parent where that array
      is frame-aligned rather than detection-row-aligned, so it is excluded from
      the completion verifier contract. A guarded read-only dry-run validated
      all 52 legacy eye-angle parents with zero blocked parents.

      Bout kinematics contract added 2026-06-10:
      `BOUT_KINEMATICS_SPEC` now validates the attrs-only invariant shared by
      the legacy layout generations under `analysis/bout_kinematics_runs`.
      The verifier requires exact `schema_id`, `method`, and `row_axis` values,
      plus shared provenance/source attrs (`schema_version`, `method_version`,
      `parameters`, `provenance`, `source_refs`, source swim-bout/track refs,
      and heading-level metadata). It intentionally does not require layout
      groups or `status` because older valid runs predate the compact layout and
      do not all carry `status`. A guarded read-only dry-run validated all 52
      legacy bout-kinematics parents with zero blocked parents.

      Historical post-compatibility summary before staged applies:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --output-json /tmp/completion_epoch_backfill_dry_run_summary_after_bout_kinematics_spec.json --blocked-jsonl /tmp/completion_epoch_backfill_blocked_parents_after_bout_kinematics_spec.jsonl --no-stdout`
      scanned 138 zarr stores, found 1557 parent groups that would stamp, and
      left 275 blocked parent groups. The blocked-parent JSONL contains 275
      rows. Compared with the initial dry-run, 899 parent groups moved from
      blocked to stampable; `BOUT_KINEMATICS_SPEC` accounts for the latest
      52-parent reduction.

      Historical post-compatibility triage before staged applies:
      `scripts/py -m fisheye.utils.triage_completion_epoch_blockers /tmp/completion_epoch_backfill_blocked_parents_after_bout_kinematics_spec.jsonl --output-json /tmp/completion_epoch_blocker_triage_after_bout_kinematics_spec.json --no-stdout`
      groups the 275 blocked parents into 10 scopes. The largest remaining
      blockers are analysis run families with no current StageSpec:
      `analysis/stimulus_response_runs` and `analysis/swim_bout_runs` each
      block 52 parents. These need either a StageSpec or an explicit decision
      to defer that scope. The remaining
      current-spec/surface-compatibility bucket is led by `refined_detect`
      (55), `eye_masks` (53), and `refined_eye_masks` (53).

      Decision 2026-06-10: defer both remaining no-spec analysis families
      from the first strict-epoch backfill. `analysis/swim_bout_runs` spans
      several historical layouts (`schema_version` 3-7, compact and
      pre-compact group structures), and only 122/138 legacy child runs carry
      `schema_id = palette.swim_bout_runs`. `analysis/stimulus_response_runs`
      is more heterogeneous: only 1/56 legacy child runs carries
      `schema_id = palette.stimulus_response` or the compact layout marker.
      Both families have useful common provenance/source attrs, but no stable
      array/table surface that all legacy children share. An attrs-only
      StageSpec here would certify weak metadata presence rather than a real
      data contract, so these scopes remain blocked until their legacy layouts
      are either migrated to a common surface or given explicit
      layout-specific validators. The blocker triage tool now reports these
      paths as `defer_scope_until_layout_specific_validator` instead of a
      generic no-spec recommendation. Verification command:
      `scripts/py -m fisheye.utils.triage_completion_epoch_blockers /tmp/completion_epoch_backfill_blocked_parents_after_bout_kinematics_spec.jsonl --output-json /tmp/completion_epoch_blocker_triage_after_deferred_scope_patch.json --no-stdout`.
      The updated triage report classifies
      `analysis/stimulus_response_runs` (52 parents, 56 no-spec children) and
      `analysis/swim_bout_runs` (52 parents, 138 no-spec children) with that
      deferred recommendation.

      Before the active non-eye compatibility-policy apply, remaining
      current-spec/surface blockers after the compatibility pass were
      much narrower: `refined_detect` (55 blocked parents), `eye_masks` (53),
      `refined_eye_masks` (53), `detect` (5), `crop` (2),
      `keypoints` (1), `refined_keypoints` (1), and
      `refined_subject_masks` (1). Highest
      concrete first-error blockers are `refined_eye_masks: missing required
      array 'edit_applied'` (115 children), `refined_detect: missing required
      subgroup 'source_detections'` (58), `eye_masks: missing required array
      'masks_roi'` (25), `refined_subject_masks/metrics: missing required array
      'centroid_xy'` (6), `detect: missing required array 'frame_indices'` (5),
      and `crop: missing required array 'bbox_norm_coords'` (2).

      The summary includes `backfill_scope_plan`, which classifies each
      stage/path scope as `ready_to_apply_if_approved`, `partially_blocked`, or
      `blocked_triage_required` and provides the matching filter hint. Current
      clean staged apply candidates are `stimulus` (109 would-stamp parents),
      `subject_masks` (109), `keypoint_profile` (105),
      `detection_profile` (60), `arena_assignment` (56), `background` (56),
      `tracking` (56), `bout_kinematics` (52), `eye_angle` (52),
      `eye_mask_profile` (52), `track_kinematics` (52),
      `bout_classification` (48), `subject_shape` (48),
      `tail_posture_view` (48), `tail_kinematics` (1),
      `analysis/detection_comparison_runs` (4), and
      `refined_online_runs` (4).
      Partially blocked but mostly stampable scopes include `crop` (119
      would-stamp, 2 blocked), `detect` (117 would-stamp, 5 blocked),
      `keypoints` (116 would-stamp, 1 blocked), `refined_keypoints` (112
      would-stamp, 1 blocked), `refined_subject_masks` (107 would-stamp,
      1 blocked), and `refined_detect` (66 would-stamp, 55 blocked).

      The backfill tool supports staged dry-runs/applies with `--stage` and
      `--parent-path` filters. Filters are ANDed; non-matching parents are
      reported as `filtered` and are not validated or mutated. Read-only crop
      filter validation:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage crop --summary-only --output-json /tmp/completion_epoch_backfill_crop_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_crop_blocked_parents.jsonl --no-stdout`
      scanned 138 stores, filtered 1711 non-crop parents, found 119 crop
      parents that would stamp, and found 2 blocked crop parents
      (`crop: missing required array 'bbox_norm_coords'`). This is the safer
      operational pattern for approval: dry-run one stage, inspect its blocked
      JSONL, then apply only that same stage if acceptable.

      Read-only `subject_masks` filter validation:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage subject_masks --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1723 --expect-would-stamp-parent-count 109 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_subject_masks_with_non_ok_expect_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_subject_masks_with_non_ok_expect_blocked_parents.jsonl --no-stdout`
      scanned 138 stores, found 138 ok stores and 0 non-ok stores, filtered
      1723 non-subject-mask parents, found 109 subject-mask parents that would
      stamp, and found 0 blocked subject-mask parents.

      Post child-marker-count `subject_masks` dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage subject_masks --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1723 --expect-would-stamp-parent-count 109 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_subject_masks_with_would_mark_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_subject_masks_with_would_mark_blocked_parents.jsonl --no-stdout`
      confirmed the same 109 would-stamp parents and 0 blockers, with
      `would_mark_child_count=340`. The blocked-parent JSONL remained empty.

      Read-only combined clean-stage validation:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage stimulus --stage subject_masks --stage keypoint_profile --stage detection_profile --stage eye_mask_profile --stage subject_shape --stage tail_posture_view --stage bout_classification --stage tail_kinematics --stage track_kinematics --stage eye_angle --stage bout_kinematics --stage arena_assignment --stage background --stage tracking --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 928 --expect-would-stamp-parent-count 904 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_blocked_parents.jsonl --no-stdout`
      scanned 138 stores, found 138 ok stores and 0 non-ok stores, filtered
      928 non-matching parents, found 904 parents that would stamp, and found
      0 blocked parents. The blocked-parent JSONL is empty. This is now the
      best first staged apply candidate because it covers every currently
      clean StageSpec-backed stage scope in one guarded operation while still
      excluding blocked or no-spec analysis scopes.

      Post write-failure-reporting dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage stimulus --stage subject_masks --stage keypoint_profile --stage detection_profile --stage eye_mask_profile --stage subject_shape --stage tail_posture_view --stage bout_classification --stage tail_kinematics --stage track_kinematics --stage eye_angle --stage bout_kinematics --stage arena_assignment --stage background --stage tracking --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 928 --expect-would-stamp-parent-count 904 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_stages_post_write_failure_hardening_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_stages_post_write_failure_hardening_blocked_parents.jsonl --no-stdout`
      revalidated the same clean-stage scope after adding structured
      write-failure reporting. Counts remained stable: 138 stores, 138 ok
      stores, 0 non-ok stores, 928 filtered parents, 904 would-stamp parents,
      0 blocked parents, and 0 write-failed parents. The blocked-parent JSONL
      remained empty.

      Post expected-write-failure-count dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage stimulus --stage subject_masks --stage keypoint_profile --stage detection_profile --stage eye_mask_profile --stage subject_shape --stage tail_posture_view --stage bout_classification --stage tail_kinematics --stage track_kinematics --stage eye_angle --stage bout_kinematics --stage arena_assignment --stage background --stage tracking --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 928 --expect-would-stamp-parent-count 904 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_stages_with_write_failed_expect_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_stages_with_write_failed_expect_blocked_parents.jsonl --no-stdout`
      revalidated the final clean-stage dry-run command shape with
      `--expect-write-failed-parent-count 0`: 138 stores, 138 ok stores, 0
      non-ok stores, 928 filtered parents, 904 would-stamp parents, 0 blocked
      parents, 0 write-failed parents, and an empty blocked-parent JSONL.

      Post child-marker-count clean-stage dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage stimulus --stage subject_masks --stage keypoint_profile --stage detection_profile --stage eye_mask_profile --stage subject_shape --stage tail_posture_view --stage bout_classification --stage tail_kinematics --stage track_kinematics --stage eye_angle --stage bout_kinematics --stage arena_assignment --stage background --stage tracking --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 928 --expect-would-stamp-parent-count 904 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_stages_with_would_mark_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_stages_with_would_mark_blocked_parents.jsonl --no-stdout`
      confirmed the same 904 would-stamp parents and 0 blockers, with
      `would_mark_child_count=1748`. The blocked-parent JSONL remained empty.

      Read-only combined clean parent-path validation:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 8 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_blocked_parents.jsonl --no-stdout`
      scanned 138 stores, found 138 ok stores and 0 non-ok stores, filtered
      1824 non-matching parents, found 8 parents that would stamp, and found 0
      blocked parents. The blocked-parent JSONL is empty. These two
      path-filtered scopes cannot be combined with the clean stage-filtered
      scopes in one command because `--stage` and `--parent-path` filters are
      intentionally ANDed.

      Post write-failure-reporting clean parent-path dry-run recorded
      2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 8 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_parent_paths_post_write_failure_hardening_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_post_write_failure_hardening_blocked_parents.jsonl --no-stdout`
      revalidated the two clean parent-path scopes after adding structured
      write-failure reporting. Counts remained stable: 138 stores, 138 ok
      stores, 0 non-ok stores, 1824 filtered parents, 8 would-stamp parents,
      0 blocked parents, and 0 write-failed parents. The blocked-parent JSONL
      remained empty.

      Post expected-write-failure-count clean parent-path dry-run recorded
      2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 8 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_parent_paths_with_write_failed_expect_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_with_write_failed_expect_blocked_parents.jsonl --no-stdout`
      revalidated the final clean parent-path dry-run command shape with
      `--expect-write-failed-parent-count 0`: 138 stores, 138 ok stores, 0
      non-ok stores, 1824 filtered parents, 8 would-stamp parents, 0 blocked
      parents, 0 write-failed parents, and an empty blocked-parent JSONL.

      Post child-marker-count clean parent-path dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 8 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_parent_paths_with_would_mark_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_with_would_mark_blocked_parents.jsonl --no-stdout`
      confirmed the same 8 would-stamp parents and 0 blockers, with
      `would_mark_child_count=0`. The blocked-parent JSONL remained empty.

      Apply guardrails added 2026-06-10: `--apply` now requires at least one
      `--stage` or `--parent-path` filter unless `--allow-broad-apply` is
      passed. It also runs a dry-run preflight and aborts before writes if the
      selected scope has blocked parents, unless `--allow-blocked-apply` is
      passed. These flags keep broad or partially blocked writes possible but
      make them explicit. Unit coverage pins the CLI-level behavior: blocked
      preflights and expected-count drift exit before any apply call, while
      `--allow-blocked-apply` plus matching expected counts proceeds through
      the explicit partial-apply path. Apply preflight is now unconditional:
      missing or open-failed zarr stores are reported as non-ok stores and
      abort before any write-mode pass, even when blocked-parent partial apply
      is explicitly allowed.

      Drift guardrails added 2026-06-10: `--expect-store-count`,
      `--expect-non-ok-store-count`,
      `--expect-blocked-parent-count`, `--expect-filtered-parent-count`,
      `--expect-would-stamp-parent-count`,
      `--expect-would-mark-child-count`, and
      `--expect-write-failed-parent-count` make dry-runs/apply preflights abort
      if counts differ from the reviewed dry-run. Successful apply reports now
      preserve both `expected_counts` and `preflight_counts`, so the report
      itself records which reviewed counts were checked before mutation.
      Apply-only guardrails `--expect-applied-stamped-parent-count` and
      `--expect-applied-marked-child-count` can also fail the command after
      writing if the actual apply result differs from the expected mutation
      count. These flags are rejected without `--apply`, so dry-runs cannot
      accidentally carry mutation-result assertions that are silently ignored.
      That validation runs before `--recordings-root` discovery, so invalid
      commands fail before scanning archive paths.
      The staged apply commands below pin stamped parent counts and marked
      child counts because the reviewed clean scopes should apply every
      preflight `would_stamp` parent and `would_mark` child.

      Broad auto-apply command, pending explicit operator approval because it
      mutates real archive attrs and now requires `--allow-broad-apply`:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --output-json /tmp/completion_epoch_backfill_apply.json --no-stdout --apply --allow-broad-apply`.

      First staged auto-apply completed 2026-06-10 after explicit operator
      approval:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage stimulus --stage subject_masks --stage keypoint_profile --stage detection_profile --stage eye_mask_profile --stage subject_shape --stage tail_posture_view --stage bout_classification --stage tail_kinematics --stage track_kinematics --stage eye_angle --stage bout_kinematics --stage arena_assignment --stage background --stage tracking --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 928 --expect-would-stamp-parent-count 904 --expect-would-mark-child-count 1748 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --expect-applied-stamped-parent-count 904 --expect-applied-marked-child-count 1748 --output-json /tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_apply.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_blocked_parents_apply.jsonl --write-failed-jsonl /tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_write_failed_parents_apply.jsonl --no-stdout --apply`
      succeeded with 138 ok stores, 0 non-ok stores, 928 filtered parents,
      904 stamped parents, 1748 marked child runs, 0 blocked parents, and 0
      write-failed parents. Both sidecars were empty:
      `/tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_blocked_parents_apply.jsonl`
      and
      `/tmp/completion_epoch_backfill_clean_stages_with_non_ok_expect_write_failed_parents_apply.jsonl`.

      Post-first-apply idempotence validation recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage stimulus --stage subject_masks --stage keypoint_profile --stage detection_profile --stage eye_mask_profile --stage subject_shape --stage tail_posture_view --stage bout_classification --stage tail_kinematics --stage track_kinematics --stage eye_angle --stage bout_kinematics --stage arena_assignment --stage background --stage tracking --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 928 --expect-would-stamp-parent-count 0 --expect-would-mark-child-count 0 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_stages_post_apply_verify_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_stages_post_apply_verify_blocked_parents.jsonl --no-stdout`
      confirmed the same scope now has 0 remaining would-stamp parents, 0
      would-mark child runs, 0 blocked parents, and 0 write-failed parents.
      The blocked-parent JSONL was empty.

      Full post-first-apply dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --summary-only --output-json /tmp/completion_epoch_backfill_full_post_first_apply_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_full_post_first_apply_blocked_parents.jsonl --no-stdout`
      scanned 138 ok stores and 0 non-ok stores. Remaining unstamped surface:
      653 would-stamp parents, 1341 would-mark child runs, 275 blocked
      parents, and 0 write-failed parents. The blocker triage sidecar
      `/tmp/completion_epoch_blocker_triage_full_post_first_apply.json`
      still groups the 275 blocked parents into 10 scopes: deferred no-spec
      analysis scopes (`analysis/stimulus_response_runs`, 52 parents, and
      `analysis/swim_bout_runs`, 52 parents), plus compatibility/backfill
      blockers for `refined_detect` (55), `eye_masks` (53),
      `refined_eye_masks` (53), `detect` (5), `crop` (2), `keypoints` (1),
      `refined_keypoints` (1), and `refined_subject_masks` (1).

      Post-first-apply clean parent-path dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 8 --expect-would-mark-child-count 0 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_parent_paths_post_first_apply_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_post_first_apply_blocked_parents.jsonl --no-stdout`
      confirmed this second clean scope still has 8 would-stamp parents, 0
      would-mark child runs, 0 blocked parents, and 0 write-failed parents.
      The blocked-parent JSONL was empty.

      Second staged auto-apply completed 2026-06-10 after explicit operator
      approval:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 8 --expect-would-mark-child-count 0 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --expect-applied-stamped-parent-count 8 --expect-applied-marked-child-count 0 --output-json /tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_apply.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_blocked_parents_apply.jsonl --write-failed-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_write_failed_parents_apply.jsonl --no-stdout --apply`
      succeeded with 138 ok stores, 0 non-ok stores, 1824 filtered parents,
      8 stamped parents, 0 marked child runs, 0 blocked parents, and 0
      write-failed parents. Both sidecars were empty:
      `/tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_blocked_parents_apply.jsonl`
      and
      `/tmp/completion_epoch_backfill_clean_parent_paths_with_non_ok_expect_write_failed_parents_apply.jsonl`.

      Post-second-apply idempotence validation recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --parent-path analysis/detection_comparison_runs --parent-path refined_online_runs --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1824 --expect-would-stamp-parent-count 0 --expect-would-mark-child-count 0 --expect-blocked-parent-count 0 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_clean_parent_paths_post_apply_verify_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_clean_parent_paths_post_apply_verify_blocked_parents.jsonl --no-stdout`
      confirmed these parent-path scopes now have 0 remaining would-stamp
      parents, 0 would-mark child runs, 0 blocked parents, and 0 write-failed
      parents. The blocked-parent JSONL was empty.

      Full post-second-apply dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --summary-only --output-json /tmp/completion_epoch_backfill_full_post_second_apply_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_full_post_second_apply_blocked_parents.jsonl --no-stdout`
      scanned 138 ok stores and 0 non-ok stores. Remaining unstamped surface:
      645 would-stamp parents, 1341 would-mark child runs, 275 blocked
      parents, and 0 write-failed parents. All remaining would-stamp parents
      are inside partially blocked stage scopes: `crop` (119), `detect`
      (117), `keypoints` (116), `refined_keypoints` (112),
      `refined_subject_masks` (107), `refined_detect` (66), `eye_masks` (4),
      and `refined_eye_masks` (4). The blocked scopes are unchanged:
      deferred no-spec analysis scopes (`analysis/stimulus_response_runs`,
      52 parents, and `analysis/swim_bout_runs`, 52 parents), plus
      compatibility/backfill blockers for `refined_detect` (55),
      `eye_masks` (53), `refined_eye_masks` (53), `detect` (5), `crop` (2),
      `keypoints` (1), `refined_keypoints` (1), and
      `refined_subject_masks` (1).

      Eye-mask deprecation decision recorded 2026-06-12: `eye_masks` and
      `refined_eye_masks` are now deprecated completion-backfill scopes. The
      rollout should not spend more effort inventorying or backfilling their
      legacy surfaces. They remain readable if present, but strict-epoch
      convergence excludes them from staged apply commands. The blocker triage
      tool now reports them as `defer_deprecated_scope`, and the backfill
      summary plan reports them as `deprecated_scope_not_backfilled`.

      Full post-second-apply deprecation-aware dry-run recorded 2026-06-12:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --summary-only --output-json /tmp/completion_epoch_backfill_full_post_second_apply_with_eye_deprecation_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_full_post_second_apply_with_eye_deprecation_blocked_parents.jsonl --no-stdout`
      scanned 138 ok stores and 0 non-ok stores. Top-level counts are still
      645 would-stamp parents, 1341 would-mark child runs, 275 blocked
      parents, and 0 write-failed parents, but `eye_masks` and
      `refined_eye_masks` are now classified as deprecated rather than active
      partial blockers. The corresponding triage report is
      `/tmp/completion_epoch_blocker_triage_full_post_second_apply_with_eye_deprecation.json`.

      Read-only partially blocked stage dry-run recorded 2026-06-10:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage crop --stage detect --stage keypoints --stage refined_keypoints --stage refined_subject_masks --stage refined_detect --stage eye_masks --stage refined_eye_masks --summary-only --output-json /tmp/completion_epoch_backfill_partially_blocked_stages_post_second_apply_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_partially_blocked_stages_post_second_apply_blocked_parents.jsonl --no-stdout`
      scanned 138 ok stores and 0 non-ok stores, filtered 1016
      non-matching parents, found 645 would-stamp parents, 1341 would-mark
      child runs, 171 blocked parents, and 0 write-failed parents. The
      matching blocker triage
      `/tmp/completion_epoch_blocker_triage_partially_blocked_stages_post_second_apply.json`
      groups these blockers as `refined_detect` (55 parents; missing
      `source_detections` subgroup), `eye_masks` (53; missing `masks_roi` or
      contour length mismatches), `refined_eye_masks` (53; mostly missing
      `edit_applied`), `detect` (5; missing `frame_indices`), `crop` (2;
      missing `bbox_norm_coords`), `keypoints` (1; missing `n_keypoints`),
      `refined_keypoints` (1; missing `heading_finite`), and
      `refined_subject_masks` (1; missing `metrics/centroid_xy`).

      Read-only non-eye partially blocked stage dry-run recorded 2026-06-12:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage crop --stage detect --stage keypoints --stage refined_keypoints --stage refined_subject_masks --stage refined_detect --summary-only --output-json /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_dry_run_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_blocked_parents.jsonl --no-stdout`
      scanned 138 ok stores and 0 non-ok stores, filtered 1130
      non-matching parents, found 637 would-stamp parents, 1341 would-mark
      child runs, 65 blocked parents, and 0 write-failed parents. The matching
      blocker triage
      `/tmp/completion_epoch_blocker_triage_non_eye_partially_blocked_stages.json`
      groups the active blockers as `refined_detect` (55 parents; missing
      `source_detections` subgroup), `detect` (5; missing `frame_indices`),
      `crop` (2; missing `bbox_norm_coords`), `keypoints` (1; missing
      `n_keypoints`), `refined_keypoints` (1; missing `heading_finite`), and
      `refined_subject_masks` (1; missing `metrics/centroid_xy`).

      Third staged partial apply completed 2026-06-12 after explicit operator
      approval, intentionally leaving the 65 active blocked parents unresolved
      and excluding deprecated eye-mask scopes:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage crop --stage detect --stage keypoints --stage refined_keypoints --stage refined_subject_masks --stage refined_detect --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1130 --expect-would-stamp-parent-count 637 --expect-would-mark-child-count 1341 --expect-blocked-parent-count 65 --expect-write-failed-parent-count 0 --expect-applied-stamped-parent-count 637 --expect-applied-marked-child-count 1341 --output-json /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_apply.json --blocked-jsonl /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_blocked_parents_apply.jsonl --write-failed-jsonl /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_write_failed_parents_apply.jsonl --no-stdout --apply --allow-blocked-apply`.
      It succeeded with 138 ok stores, 0 non-ok stores, 1130 filtered
      parents, 637 stamped parents, 1341 marked child runs, 65 blocked
      parents, and 0 write-failed parents. The write-failed sidecar was empty;
      the blocked-parent sidecar contains the expected 65 unresolved blocked
      parents.

      Post-third-apply non-eye idempotence validation recorded 2026-06-12:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --stage crop --stage detect --stage keypoints --stage refined_keypoints --stage refined_subject_masks --stage refined_detect --expect-store-count 138 --expect-non-ok-store-count 0 --expect-filtered-parent-count 1130 --expect-would-stamp-parent-count 0 --expect-would-mark-child-count 0 --expect-blocked-parent-count 65 --expect-write-failed-parent-count 0 --summary-only --output-json /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_post_apply_verify_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_non_eye_partially_blocked_stages_post_apply_verify_blocked_parents.jsonl --no-stdout`
      confirmed these selected non-eye stages now have 0 remaining would-stamp
      parents, 0 would-mark child runs, 65 blocked parents, and 0 write-failed
      parents.

      Full post-third-apply dry-run recorded 2026-06-12:
      `scripts/py -m fisheye.utils.backfill_completion_epoch --recordings-root /nvme1/recordings --summary-only --output-json /tmp/completion_epoch_backfill_full_post_third_apply_summary.json --blocked-jsonl /tmp/completion_epoch_backfill_full_post_third_apply_blocked_parents.jsonl --no-stdout`
      scanned 138 ok stores and 0 non-ok stores. Remaining surface: 8
      would-stamp parents, 0 would-mark child runs, 275 blocked parents, and
      0 write-failed parents. The only remaining would-stamp parents are in
      deprecated eye-mask scopes (`eye_masks`: 4, `refined_eye_masks`: 4).
      Active non-eye scopes have no remaining stampable parents; they are
      blocked-only. The corresponding triage report is
      `/tmp/completion_epoch_blocker_triage_full_post_third_apply.json`.

      The tool is conservative but not transactional across attr writes. It
      only attempts mutation when every child in that parent is already
      contracted or validates as legacy-complete. Blocked parents are reported
      but not mutated. Verified legacy child runs are marked complete before
      the parent epoch is stamped; if a write fails, the parent reports
      `write_failed` and is not considered stamped. Already-written child
      markers are intentionally left in place so a rerun can finish the parent
      without rewriting completion timestamps. Unit coverage pins that
      idempotence: once a parent is strict and its legacy children have
      explicit completion attrs, a second apply reports `already_strict` and
      does not rewrite child completion timestamps.

### Phase 5 — retire

- [ ] When all known stores are stamped: flip the module-level fallback so an
      *unstamped* parent logs a warning unconditionally (external/unknown
      archives stay readable but loudly provisional).
- [ ] Delete this doc to docs/archive/ with a closing note.

## Interaction with other convergences

Same playbook applies to `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` (currently
`{'detect_quality'}` only, stage_complete.py): once Phase 2 lands, grow that
allowlist stage-by-stage with the same ratchet-test pattern, since "marker
present" and "required arrays valid" are the two halves of trustworthy
completion. Track that as its own checklist when this one reaches Phase 3.
