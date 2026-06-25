# Web Labeling Multi-User Execution Checklist

Use this checklist before sharing the browser labeling workflow with additional labelers.

The expected contract is:

- Labelers use browser-only entry points such as `/identity?expected_user=<user>` and `/my-datasets?expected_user=<user>`.
- One active user owns a recording at a time.
- Browser label/mutation requests are authorized server-side.
- Actual label mutations target assigned task-scoped training Zarrs.
- CSV, handoff, roster, HTML, and JSON files are metadata/control-plane artifacts only.
- Browsers never receive direct Zarr write authority, raw Zarr paths, storage credentials, or filesystem write targets.

## 1. Static syntax checks

- [ ] Run Python syntax compilation for the touched web route and test files.

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m py_compile \
  src/fisheye/labeling/web.py \
  tests/unit/fisheye/test_labeling_web_routes.py \
  tests/unit/fisheye/test_labeling_assignment_store.py
```

## 2. Focused unit tests

- [ ] Run the focused route-contract subset that exercises the personalized queue APIs, signed-link handoff, session-owned completion, stale-session rejection, unknown-user rejection, identity diagnostic contract, and editor completion route.

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_labeling_assignment_store.py::test_personalized_dataset_queue_http_routes_scope_to_expected_user \
  tests/unit/fisheye/test_labeling_assignment_store.py::test_keypoint_editor_exposes_copy_mutation_support_reference_button \
  tests/unit/fisheye/test_labeling_assignment_store.py::test_identity_probe_points_matched_users_to_dataset_queue_first \
  -q --tb=short
```

This focused subset should be rerun after any change to assigned-user page-shell routing, unknown-user page-shell denial, failed-identity launch CTA suppression, successful-identity launch CTA retention, signed-link start gates, or session-owned completion/mutation guards.

- [ ] Run the web route tests that cover personalized queues, expected-user guards, task open/complete, promotion retry, session completion, and support metadata.

```bash
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_labeling_web_routes.py -q
```

- [ ] Run the assignment/handoff store tests that cover roster CSV flattening and readiness reconstruction.

```bash
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_labeling_assignment_store.py -q
```

## 3. Focused invariant assertions

Confirm the tests cover these fields on successful personal APIs and fail-closed error payloads:

- [ ] `personalized_launch_readiness.schema == "palette.web_labeling_personalized_launch_readiness.v1"`
- [ ] `personalized_launch_readiness.personalized_labeler_entry_url == "/my-datasets?expected_user=<user>"`
- [ ] `personalized_launch_readiness.browser_label_write_target == "training_zarr"`
- [ ] `personalized_launch_readiness.browser_writes_csv_or_handoff_files is False`
- [ ] `personalized_launch_readiness.browser_has_direct_zarr_write_authority is False`
- [ ] `dataset_queue_direct_start_policy.browser_label_write_target == "training_zarr"`
- [ ] `dataset_queue_direct_start_policy.browser_writes_csv_or_handoff_files is False`
- [ ] `dataset_queue_direct_start_policy.browser_has_direct_zarr_write_authority is False`
- [ ] `csv_handoff_artifact_role == "metadata_only_control_plane"`

Confirm these payload classes expose the same contract:

- [ ] `/api/me/identity`
- [ ] `/api/me/tasks`
- [ ] `/api/me/datasets`
- [ ] Unknown-labeler responses
- [ ] Expected-user mismatch responses
- [ ] Signed task-link mismatch responses
- [ ] Signed task-link runtime operator-validation start-gate denials
- [ ] Task-open success and denial responses
- [ ] Task-completion success and denial responses
- [ ] Session-completion success and denial responses
- [ ] Labeler failed-promotion retry denial responses
- [ ] Browser mutation success and denial responses

Confirm these additional multi-user handoff invariants:

- [ ] `/identity` and `/api/me/identity` are diagnostic only and do not authorize work.
- [ ] Successful `/identity?expected_user=<user>` pages expose the normal personal queue, landing, personal dashboard, and canonical fallback launch CTAs.
- [ ] `/identity` and `/api/me/identity` without an `expected_user` guard return `identity_expected_user_required`, suppress all launch CTAs, and keep support URLs diagnostic-only.
- [ ] Unknown identity probes return `unknown_labeling_user` and the rendered/nested identity `operator_action` starts with `Stop before labeling`.
- [ ] Expected-user mismatch probes return `identity_user_mismatch` in both `/identity` HTML and `/api/me/identity` JSON, and the nested identity action starts with `Stop before labeling`.
- [ ] Identity probe payloads expose `identity_probe_expected_user_guard_required`, `identity_probe_launch_ctas_rendered`, `identity_probe_launch_ctas_suppressed`, and `identity_probe_failed_support_urls_diagnostic_only`; successful expected-user guarded probes render CTAs, while missing-expected-user, unknown, or mismatched probes suppress CTAs and mark support URLs diagnostic-only.
- [ ] Unknown or mismatched identity probes do not tell the labeler to open the guarded dataset queue first.
- [ ] Unknown or mismatched `/identity` pages suppress all normal launch CTAs, including personal queue, landing, personal dashboard, and canonical fallback links, and instead show a stop-before-labeling support/copy path.
- [ ] Failed identity probe support details may include guarded URLs for operator diagnostics, but those URLs are not rendered as labeler launch CTAs and do not authorize work.
- [ ] Unknown labelers are blocked from `/`, `/me`, `/my-datasets`, `/labeling`, `/my-work`, canonical `/datasets`/`/work`, `/api/me/tasks`, and `/api/me/datasets` even when `expected_user` matches the browser identity.
- [ ] Direct task-open mismatch responses include `expected_user`, `resolved_user`, and `task_open_authorization_contract.server_authorizes_open is False`.
- [ ] Signed links are treated as entry hints only and re-run identity, assignment, task-state, session, and runtime operator-validation start-gate checks before session creation.
- [ ] Editor completion uses `/api/sessions/<session_id>/complete`, not `/api/tasks/<task_id>/complete`, so completion remains session-owned.
- [ ] Stale, superseded, wrong-user, or reassigned sessions cannot complete or mutate labels.

## 4. Generated artifact checks

- [ ] Generate or refresh the launch/handoff artifacts using the current workflow command.

Record the exact command used here:

```text
<paste command>
```

- [ ] Inspect generated `labeler-roster.csv` for these flattened fields:

```text
personalized_launch_readiness_schema
personalized_launch_readiness_personalized_labeler_entry_url
personalized_launch_readiness_browser_label_write_target
personalized_launch_readiness_browser_writes_csv_or_handoff_files
personalized_launch_readiness_browser_has_direct_zarr_write_authority
labeler_safety_identity_probe_expected_user_guard_required
labeler_safety_identity_probe_success_launch_ctas_rendered
labeler_safety_identity_probe_failed_launch_ctas_suppressed
labeler_safety_identity_probe_failed_support_urls_diagnostic_only
```

- [ ] Confirm each generated user row has:

```text
personalized_launch_readiness_browser_label_write_target=training_zarr
personalized_launch_readiness_browser_writes_csv_or_handoff_files=false
personalized_launch_readiness_browser_has_direct_zarr_write_authority=false
labeler_safety_identity_probe_expected_user_guard_required=true
labeler_safety_identity_probe_success_launch_ctas_rendered=true
labeler_safety_identity_probe_failed_launch_ctas_suppressed=true
labeler_safety_identity_probe_failed_support_urls_diagnostic_only=true
```

- [ ] Confirm handoff/roster/CSV artifacts do not claim to be label-write targets.

- [ ] Confirm generated `inspection-targets.json` advertises required values for:

```text
browser_label_write_target=training_zarr
browser_writes_csv_or_handoff_files=false
browser_has_direct_zarr_write_authority=false
labeler_safety_identity_probe_expected_user_guard_required=true
labeler_safety_identity_probe_success_launch_ctas_rendered=true
labeler_safety_identity_probe_failed_launch_ctas_suppressed=true
labeler_safety_identity_probe_failed_support_urls_diagnostic_only=true
```

## 5. Runtime API smoke with one assigned labeler

Start the web labeling server in the same mode you plan to use for sharing.

Record the exact server command here:

```text
<paste command>
```

With one assigned test labeler, check:

- [ ] `/identity?expected_user=<user>` reports the resolved user and expected user match.
- [ ] `/identity?expected_user=<other-user>` fails closed.
- [ ] `/?expected_user=<user>` and `/me?expected_user=<user>` route to the queue-first labeler landing, not an admin or unscoped dashboard.
- [ ] `/my-datasets?expected_user=<user>` shows only that user's assigned datasets/recordings.
- [ ] `/my-datasets?expected_user=<other-user>` fails closed.
- [ ] Unknown/unassigned users fail closed on every labeler page shell: `/`, `/me`, `/my-datasets`, `/labeling`, `/my-work`, `/datasets`, and `/work`.
- [ ] `/labeling?expected_user=<user>` lands on the queue-first labeling page.
- [ ] `/my-work?expected_user=<user>` opens the full dashboard fallback.
- [ ] Canonical fallback pages `/datasets?expected_user=<user>` and `/work?expected_user=<user>` preserve the same resolved-user/expected-user guard and assigned-only work visibility as `/my-datasets` and `/my-work`.
- [ ] A task can be opened only by the assigned user.
- [ ] A copied/wrong-user task-open POST returns `task_open_user_mismatch`.
- [ ] A task-complete POST requires the current guarded session.
- [ ] A stale, superseded, or reassigned session cannot save.
- [ ] A completed task cannot be reopened by a labeler without operator action.

## 6. Browser mutation smoke

Use a disposable, restorable, or backed-up training Zarr.

- [ ] Open one representative assigned task in the browser.
- [ ] Make a minimal label edit.
- [ ] Confirm the browser save succeeds.
- [ ] Confirm the response includes a `mutation.audit_event_id`.
- [ ] Confirm the response says the write target is server-owned task/training Zarr scope.
- [ ] Confirm the response does not expose raw Zarr paths.
- [ ] Confirm the browser response says no CSV/handoff browser writes.
- [ ] Confirm the browser response says no direct browser Zarr authority.
- [ ] Confirm the audit event exists in the operator/admin event lookup.
- [ ] Confirm the changed label data is present in the intended training Zarr.
- [ ] Confirm no intermediate CSV/handoff file was used as the label-write target.

## 7. Multi-user safety smoke

Use two test users and two recordings.

- [ ] Assign recording A to user A.
- [ ] Assign recording B to user B.
- [ ] Confirm user A cannot see or open recording B.
- [ ] Confirm user B cannot see or open recording A.
- [ ] Reassign recording A from user A to user B.
- [ ] Confirm old user A sessions for recording A are closed or rejected.
- [ ] Confirm user A stale browser tabs cannot save after reassignment.
- [ ] Confirm user B can open the reassigned recording only after the assignment is active.
- [ ] Confirm assignment integrity reports zero duplicate active owners.

## 8. Operator evidence gates before sharing

- [ ] Browser identity-source evidence has been captured from the deployed environment.
- [ ] Browser smoke evidence has been captured for at least one representative labeler identity.
- [ ] Response-security header evidence has been captured for deployed `/my-datasets?expected_user=<user>`.
- [ ] Disposable/restorable Zarr mutation smoke evidence has been captured for launched workflow kinds.
- [ ] Mutable Zarr backup evidence is complete where required.
- [ ] Operator validation checklist is applied and refreshed after evidence updates.
- [ ] Handoff checksums are refreshed after evidence/checklist updates.
- [ ] `inspect-handoff --require-shareable` or the equivalent launch gate reports shareable before links are sent.

## 9. Final launch criteria

- [ ] Tests in sections 1 and 2 pass.
- [ ] All required generated artifacts expose `personalized_launch_readiness`.
- [ ] All checked browser/API denial paths fail closed with guarded personal queue links.
- [ ] All checked browser/API denial paths preserve training-Zarr/no-CSV/no-direct-Zarr assertions.
- [ ] Labelers need only a browser and do not need local Palette, Crimson, Conda, or project dependencies.
- [ ] Operators have a recovery route for reassignment, reopen, session cleanup, audit lookup, failed-promotion retry, backup, and rollback.
- [ ] No labeler-facing payload exposes raw Zarr paths, direct task scope, filesystem paths, storage credentials, or direct write authority.
