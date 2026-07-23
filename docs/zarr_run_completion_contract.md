# Zarr Run Completion Contract

Status: active  
Last verified: 2026-07-20

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
- `stage_selector_eligible = false | true`

Future-normal public writers also mint an immutable publication-owner UUID.
The exact attribute name is run-family specific until a shared publication
schema is adopted, but it must be written in the same first metadata operation
as `stage_selector_eligible = false`. A writer may mutate the public child only
while that freshly resolved child still has the UUID minted by the current
attempt.

Run parent groups may set:

- `latest`: backward-compatible pointer to the latest complete run.
- `latest_complete`: explicit latest complete run pointer.
- `latest_pending`: newest started run that is not complete yet.

For future-normal publication, a public run name is immutable from its first
successful creation. Writers must use this order:

1. Choose a new, unoccupied public child name.
2. Atomically create it with its owner UUID and literal
   `stage_selector_eligible = false`. Include the `running` completion state in
   that operation when supported; otherwise stamp it immediately inside the
   same failure guard, before any payload write.
3. Write the payload, coordinate/measurement descriptors, row identity, and
   exact lineage.
4. Freshly re-resolve the canonical path and owner, validate the exact child
   inventory and payload, then mark the child `complete` while it remains
   selector-ineligible.
5. Acquire the run parent's owner/generation lease and advance the complete
   selectors while the child remains ineligible. Revalidate the fresh child
   after the selector writes.
6. Make `stage_selector_eligible = true` the literal commit write. No fallible
   validation, metadata write, registry update, or console/logging operation
   may follow that commit.

`mark_run_complete` is therefore a completion marker, not sufficient authority
to publish a future-normal run. A hardened writer uses a family validator and
guarded selector activation after completion. Post-commit reporting must be
nonthrowing and registry publication must independently preflight the committed
run.

Each transaction has exactly one selector-rollback authority. A run-family
activation receipt snapshots the live predecessor state when it acquires its
lease, after any long copy or validation interval. When such a receipt exists,
the generic atomic publisher's pre-copy parent snapshot must be disabled for
selector rollback; the two mechanisms must never be composed as fallbacks. If
the exact receipt rollback itself cannot be verified, the transaction fails
loudly and leaves the owned child ineligible. It must not restore an older
pre-copy snapshot that could erase an intervening publication.

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
- the run group must satisfy the parent-scoped completion rule; and
- a present `stage_selector_eligible` marker must be literal `true`.

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

Future-normal readers resolve default inputs with
`resolve_latest_complete_run_name(parent, legacy_default=False)`. Canonical
resolution requires:

- both `latest` and `latest_complete` are present and name the same exact
  child;
- that child is complete; and
- that child has literal `stage_selector_eligible = true`.

Any selector disagreement is an in-progress or invalid handoff, not permission
to resurrect whichever older pointer still resolves. Readers return a
retry/fail-closed result. Explicit run paths still require the selected child's
completion, eligibility, schema, lineage, and consumer-specific contract.

Legacy runs without completion attrs are treated as complete for compatibility.
That behavior is available only through an explicit, testable compatibility
path such as `legacy_default=True`; it is not the normal policy for recordings
created after the canonical cutover. Runs that opt into
`palette.zarr_run_completion.v1` are usable only when completion and literal
selector eligibility both pass.

## Failed Publication Tombstones

A failed public publication is retained as an immutable tombstone. It is not a
new scientific array or a legacy-coordinate adapter. It is the same public run
child, marked so that no scientific reader may select it. A verified tombstone
has at least:

- the originally minted publication-owner UUID;
- `palette_run_completion_status = "failed"`;
- literal `stage_selector_eligible = false`;
- no completed timestamp;
- failure time/reason and a versioned tombstone record; and
- a retry policy requiring a new immutable run name.

The child may retain partial payload arrays for diagnosis. Readers ignore or
reject it, and an operator can inspect it only by exact path. Writers must not
delete it, rename another child over it, or reinterpret it as a new attempt.
Cleanup must freshly resolve the canonical path and prove the original owner
before every mutation; a stale in-memory group handle is not ownership proof.

Only hidden same-parent staging children may be deleted after failed physical
installation. Once a canonical public child exists, including a failed child,
its name is permanently occupied. This policy means healthy future archives
can contain occasional failed-attempt tombstones without requiring any legacy
reader adapter.

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
install a newly owner-bound and selector-ineligible public child, validate and
mark it complete, perform a guarded selector handoff, and commit eligibility
last. Failure before public installation may delete only the hidden staging
child. Failure after public installation retains an owner-bound, ineligible
tombstone and conditionally restores only parent attributes still proven to
belong to the failed publication epoch.

Readers must still enforce completion because historical and non-materialized
writers may use only the attr contract. Atomic installation prevents a new
production analysis run from appearing partially copied under its final path.
Completion, exact contract validation, consistent selectors, and the final
eligibility bit jointly control whether a default reader may select it.
