# Zarr Run Completion Contract

Status: active  
Last verified: 2026-06-25

Palette run parents such as `detect_runs`, `crop_runs`,
`refined_detect_runs`, `keypoints_runs`, and nested `quality_reports` must not
publish an incomplete run as the preferred input for downstream stages.

## Attributes

Run groups that opt into this contract set:

- `palette_run_completion_contract = "palette.zarr_run_completion.v1"`
- `palette_run_completion_status = "running" | "complete" | "failed"`
- `palette_run_started_at_utc`
- `palette_run_completed_at_utc` when complete
- `palette_run_name`
- `palette_run_stage`

Run parent groups may set:

- `latest`: backward-compatible pointer to the latest complete run.
- `latest_complete`: explicit latest complete run pointer.
- `latest_pending`: newest started run that is not complete yet.

New contract-aware writers should create the run group, immediately mark the
run `running`, and set `latest_pending`. They should not move `latest` to the
new run until the run is complete. On success, writers call
`mark_run_complete(..., parent_group=<parent>, run_name=<name>)`, which updates
both `latest_complete` and `latest`.

Writers that create refined outputs in multiple phases must call
`mark_run_complete` only after every required array, component subgroup,
review-status attr, and provenance payload has been written. For example,
`finalize_subject_masks` creates `refined_subject_masks_runs/<run>`, writes the
component masks/metrics/geometry, and only then stamps the refined run complete
before registry stage completion is emitted. Directly assigning
`parent.attrs["latest"]` is not sufficient and will be rejected by registry
completion validation.

## Registry Completion Rule

Zarr stage writers should report successful stage completion through
`fisheye.registry.stage_complete.emit_stage_completion`.

`emit_stage_completion` resolves the registry's effective dataset ID before
writing `recording_step_status`. For source-recording paths under
`/recordings/`, this means live stage-completion rows use the same canonical
path-disambiguated IDs as full registry scans, e.g.
`<session_uuid>:z<path_hash_prefix>`, instead of reintroducing legacy
`dataset_id == session_uuid` rows for training zarrs.

For `status="ok"` with a `run_name`, that helper is fail-closed on this
contract:

- the caller must pass a readable Zarr root;
- the helper must resolve the named run group under the expected run parent;
- the run group must satisfy `is_run_complete(..., legacy_default=True)`.

If any of those checks fail, the helper refuses to write an `ok`
`recording_step_status` row. Non-`ok` statuses may bypass run-group validation
when prebuilt dataset metadata is supplied.

Nested writers whose run group is not under the default top-level parent should
pass `completion_group_path`, for example
`clips/clip_000000/cameras/2010093/detect_runs/<detect_run>/quality_reports/<quality_run>`.
This lets the validator address clip-local run groups directly instead of
depending on top-level parent scanning or consolidated metadata freshness.

After the completion-marker check passes, `emit_stage_completion` also runs the
stage-array validator. Array validation is currently hard-enforced for
`detect_quality`; all other stages remain shadow-mode until real-run telemetry
shows their writers are contract-clean. Shadowed stages write validation
status/errors into `details_json`, but array-contract failures only block
completion for stages explicitly added to
`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`.

Inspect shadow telemetry with:

```bash
scripts/py -m fisheye.utils.report_stage_array_validation_shadow \
  --registry /nvme1/palette_registry.sqlite \
  --include-no-spec
```

## Read Rule

Contract-aware readers should resolve default inputs with
`resolve_latest_complete_run_name(parent, legacy_default=True)`.

Legacy runs without completion attrs are treated as complete for compatibility.
Runs that opt into `palette.zarr_run_completion.v1` are considered usable only
when `palette_run_completion_status == "complete"`.

## Safety Check

Scan an archive for unsafe latest pointers:

```bash
scripts/py -m fisheye.utils.check_zarr_run_completion /path/to/archive.zarr --fail-on-unsafe
```

Useful outputs:

- `unsafe_parent_count > 0`: at least one run parent has `latest` or
  `latest_complete` pointing at a missing or incomplete opted-in run.
- `pending_parent_count > 0`: one or more parents have `latest_pending` or
  incomplete opted-in runs. This is acceptable while a job is running, but
  should be investigated if the job is finished.

## Physical Publication

The completion attributes remain the generic compatibility contract for every
run family. Hardened analysis materializers additionally use
`palette.atomic_run_group_publisher` version 1: compute in node-local storage,
copy to a hidden same-parent sibling, verify the physical inventory, atomically
rename the sibling to the final run name, mark completion, update pointers, and
roll back the target and parent attributes after any failure.

Readers must still enforce completion because historical and non-materialized
writers may use only the attr contract. Atomic installation prevents a new
production analysis run from becoming visible under its final path while it is
being populated; completion controls whether default resolution may select it.
