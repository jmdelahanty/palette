# Web Labeling Workflow Implementation Checklist

<!-- design-meta
status: clean implementation checklist
last_updated: 2026-06-24
scope: browser-based assigned labeling workflow for mutable Palette/Zarr training data
-->

## Purpose

This document consolidates the implementation decisions from the recent web-labeling workflow work.

For a short operator-facing status summary, see `docs/web_labeling_implementation_status.md`.

The goal is to let collaborators label, review, or correct assigned training data from a browser without installing Palette, Crimson, Conda environments, or local project dependencies.

The core operating rule is:

> One recording has one active assigned user at a time.

The browser is only the editing surface. Palette remains the server-side authority for identity, assignment, task state, session state, Zarr reads, Zarr writes, audit history, validation evidence, and operator recovery.

## Target Experience

A labeler should be able to receive one safe link, open it in a browser, and see only the datasets, recordings, and tasks assigned to them.

Preferred labeler entry points:

| Entry point | Purpose |
| --- | --- |
| `/` | Personalized datasets-waiting queue-first landing page for the authenticated user. |
| `/me` | Alias for the same personalized datasets-waiting landing page. |
| `/labeling?expected_user=<user>` | Human-readable guarded alias for the same datasets-waiting queue. |
| `/my-datasets?expected_user=<user>` | Preferred personalized dataset and recording queue for one intended assignee. |
| `/datasets?expected_user=<user>` | Canonical dataset queue fallback for the same assigned work. |
| `/my-work?expected_user=<user>` | Preferred personalized full assigned-work dashboard fallback. |
| `/work?expected_user=<user>` | Canonical full assigned-work dashboard fallback. |
| `/identity?expected_user=<user>` | Identity probe to confirm the browser session resolves to the intended assignee. |

Preferred operator flow:

1. Create or import labeling tasks.
2. Assign each recording to exactly one active user.
3. Generate launch or handoff artifacts.
4. Confirm identity, assignment, backup, queue, audit, session, and validation-readiness gates.
5. Send only ready-row draft text or guarded queue links after safe-share inspection passes.
6. Monitor progress, blocked states, stale sessions, and audit events from admin views.
7. Reopen, reassign, repair, or retry work only through operator-controlled routes.

## Architecture

The workflow is intentionally split into four layers.

| Layer | Responsibility |
| --- | --- |
| Assignment store | Records users, recordings, assignments, tasks, sessions, audit events, and one-owner transitions. |
| Labeler web surface | Serves queue pages, dashboards, task sessions, read APIs, and save APIs. |
| Operator web surface | Serves admin, roster, recording, repair, completion, reopen, launch, and inspection views. |
| Handoff and validation artifacts | Package ready-row draft links, dataset queues, safety policy metadata, backup plans, validation logs, and readiness checklists. |

Important boundary decisions:

- A URL is never sufficient authorization by itself.
- The browser never receives direct Zarr write authority.
- The browser never receives raw Zarr filesystem paths, storage credentials, or direct task scope.
- The mutable browser-labeling data plane is the assigned task-scoped training Zarr; CSVs, handoff files, roster files, and intermediate exports are metadata/control-plane artifacts, not label-write targets.
- Every browser read or write is rechecked server-side against resolved identity, expected user, active assignment, task state, session state, session expiry, and workflow scope.
- Palette server code performs all Zarr and filesystem mutation.

## Required Safety Invariants

These invariants define the minimum safe behavior for multi-user browser labeling.

- [x] A labeler sees only work assigned to the resolved browser user.
- [x] A labeler mutates only work assigned to the resolved browser user.
- [x] A recording cannot have two active owners.
- [x] Reassignment closes or invalidates stale sessions from the previous owner.
- [x] Reassignment closes previous-owner sessions before committing the owner/status update.
- [x] Reassignment session closure and owner/status update commit as one state transition.
- [x] Invalid reassignment requests fail before any previous-owner session is closed.
- [x] The assignment store exposes a read-only `single_owner_assignment_contract()` helper so wrappers can assert one-owner/current-assignee-only browser mutation, server-resolved training-Zarr targets, and metadata-only CSV artifacts without mutating assignment state.
- [x] Store-backed control-plane reports expose that nested single-owner assignment-store proof and mirror its presence/readiness/met status inside `assignment_ownership_contract` so lightweight wrappers can require the store API evidence rather than only static policy text.
- [x] Dashboard and launch roster CSV exports include the flattened store-backed single-owner proof fields, and `inspection-targets.json` advertises `shareability_single_owner_store_contract_fields` plus required values for wrapper discovery.
- [x] Live `/api/me/identity`, `/api/me/tasks`, and `/api/me/datasets` responses expose the nested store-backed `single_owner_assignment_contract` plus flat `assignment_ownership_contract_store_single_owner_assignment_contract_*` fields so browser wrappers can gate on store evidence before opening or mutating work.
- [x] Live identity/work/dataset API store proofs include `assignment_ownership_integrity`, `assignment_ownership_contract_assignment_ownership_integrity_ok`, and `assignment_ownership_contract_duplicate_active_owner_count`, so wrappers can reject duplicate-owner stores even if the structural schema contract is present.
- [x] Live labeler-route authorization checklists include `single_owner_store_contract_required`, `single_owner_store_proof_ready`, `assignment_ownership_integrity_ok`, and `duplicate_active_owner_count`, and fold the store proof into checklist readiness when the contract is supplied.
- [x] Labeler-route authorization policy and validation-checklist contracts declare the store-proof requirement up front: browser work requires single-owner store proof, assignment-integrity OK, zero duplicate active owners, server-resolved training-Zarr targets, and no intermediate CSV mutation.
- [x] Handoff roster CSV rows flatten the labeler-route authorization store-proof requirement fields, including required integrity OK, zero duplicate active owners, training-Zarr targets, and rejection of intermediate CSV mutation.
- [x] `inspection-targets.json` advertises `shareability_labeler_route_authorization_store_proof_fields` and required values alongside the single-owner store-contract field discovery block.
- [x] Route-authorization failure actions and sendability repair guidance explicitly name single-owner store proof, zero duplicate active owners, server-resolved training-Zarr targets, and no intermediate CSV mutation.
- [x] Handoff sendability fails closed when route policy is present but runtime route-authorization checklist evidence is missing `single_owner_store_proof_ready=true`, `assignment_ownership_integrity_ok=true`, server-resolved training-Zarr proof, or `labelers_mutate_intermediate_csvs=false`; roster CSV rows flatten the runtime checklist present/ready/store-proof/integrity/data-plane fields.
- [x] `inspection-targets.json` advertises `shareability_labeler_route_authorization_runtime_checklist_fields` and required values for runtime checklist present/ready, `single_owner_store_proof_ready=true`, `assignment_ownership_integrity_ok=true`, `duplicate_active_owner_count=0`, `browser_mutation_target_resolved_server_side=true`, `labelers_mutate_assigned_training_zarrs=true`, and `labelers_mutate_intermediate_csvs=false`.
- [x] `inspection-targets.json` also advertises `shareability_labeler_route_authorization_runtime_checklist_gate_contract`, including the gate field, compact/nested contract paths, required value, mismatch fields, fail-closed reason, repair command ID, required fields, and required values for wrapper discovery.
- [x] Dashboard roster/preflight/queue-readiness summaries aggregate the runtime route-checklist gate as `labeler_route_authorization_runtime_checklist_gate_all_users_met`, not-met users, and total mismatch count fields so live link handoff decisions fail closed on missing store-proof, training-Zarr target, or no-CSV evidence.
- [x] CLI `work-summary` JSON exports and stdout summaries expose the same nested/flat store-backed single-owner assignment contract fields as live labeler APIs.
- [x] `/work` and `/datasets` copyable support details include the store-backed single-owner proof fields, including server-resolved training-Zarr targets and no intermediate CSV mutation, so labeler-provided diagnostics retain the same gating evidence as API payloads.
- [x] Task completion prevents ordinary labeler writes until operator reopen.
- [x] A stale browser tab cannot save after reassignment, completion, expiration, or session supersession.
- [x] A forwarded queue, dashboard, or task link cannot bypass identity and assignment checks.
- [x] Signed links remain short-lived convenience links, not authorization grants.
- [x] Admin routes have a separate operator authorization boundary.
- [x] Browser clients never receive direct Zarr handles, filesystem paths, storage credentials, or write authority.
- [x] Browser mutations target the assigned task-scoped training Zarr through server-owned Palette code, not handoff CSVs, intermediate CSVs, or roster files; generated contracts explicitly mark both handoff CSVs and intermediate CSVs as non-label-write targets.
- [x] All successful mutations are auditable.
- [x] Mutable Zarr rollback is possible from an operator-approved backup or known-good source.

## Current Implementation Status

### Focused validation completed

- [x] Focused web-route validation passed: `scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_web_routes.py -q` returned `38 passed in 34.71s`.
- [x] Current web-route validation passed after multi-user identity/handoff hardening and updated route-contract expectations: `PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_web_routes.py -q --tb=short` returned `38 passed in 34.88s`.
- [x] Focused multi-user handoff route-contract baseline validation passed before later route-shell and identity-CTA hardening assertions were added: `PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_assignment_store.py::test_personalized_dataset_queue_http_routes_scope_to_expected_user tests/unit/fisheye/test_labeling_assignment_store.py::test_keypoint_editor_exposes_copy_mutation_support_reference_button tests/unit/fisheye/test_labeling_assignment_store.py::test_identity_probe_points_matched_users_to_dataset_queue_first -q --tb=short` returned `3 passed in 3.35s`.
- [x] Runtime identity probes suppress all normal launch CTAs when the probe is not OK; failed `/identity` pages now show stop-before-labeling support/copy guidance instead of personal queue, landing, personal dashboard, or canonical fallback launch links, while successful identity probes still expose those launch CTAs.
- [x] Identity probes require an explicit `expected_user` guard before reporting `ok` or rendering launch CTAs; missing-expected-user probes return `identity_expected_user_required`, suppress CTAs, and keep support URLs diagnostic-only.
- [x] Failed identity probe support JSON can preserve guarded URL diagnostics for operators, but those diagnostic URLs are not rendered as labeler launch CTAs and do not authorize work.
- [x] Identity probe payloads expose machine-readable expected-user guard and CTA state with `identity_probe_expected_user_guard_required`, `identity_probe_launch_ctas_rendered`, `identity_probe_launch_ctas_suppressed`, and `identity_probe_failed_support_urls_diagnostic_only`, so wrappers do not need to scrape HTML links.
- [x] Labeler-safety policy payloads and generated roster rows expose static identity CTA/guard policy fields: `identity_probe_expected_user_guard_required`, `identity_probe_success_launch_ctas_rendered`, `identity_probe_failed_launch_ctas_suppressed`, and `identity_probe_failed_support_urls_diagnostic_only`.
- [x] Launch-bundle `inspection-targets.json` advertises the static labeler-safety identity expected-user guard and CTA policy required values so wrapper tooling can verify generated artifacts without sampling roster CSVs.
- [x] Handoff sendability treats unsafe static identity CTA policy fields as `labeler_safety_policy_not_ready`, so handoffs cannot be marked ready when successful identity launch CTAs are disabled, failed identity launch CTAs are not suppressed, or failed-probe support URLs are not diagnostic-only.
- [x] Handoff sendability repair guidance names failed-identity launch CTA suppression and diagnostic-only support URLs when labeler-safety metadata is not ready.
- [x] Focused route-contract assertions cover assigned-user page shells, unknown-user page-shell denials, missing-expected-user and unknown/mismatched identity stop guidance in both `/identity` HTML and `/api/me/identity` JSON, absence of all normal launch CTAs on failed identity pages, retention of normal launch CTAs on successful identity pages, and the machine-readable identity CTA contract fields.
- [x] Focused multi-user handoff route-contract validation passed after the added assigned-user page-shell, unknown-user page-shell, missing-expected-user and unknown/mismatched identity HTML/JSON, failed-identity launch-CTA suppression, successful-identity launch-CTA retention, runtime identity expected-user guard/CTA contract field assertions, static labeler-safety expected-user guard/CTA policy field assertions, inspection-target required-value assertions, and unsafe static identity policy sendability assertions: `PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_assignment_store.py::test_personalized_dataset_queue_http_routes_scope_to_expected_user tests/unit/fisheye/test_labeling_assignment_store.py::test_keypoint_editor_exposes_copy_mutation_support_reference_button tests/unit/fisheye/test_labeling_assignment_store.py::test_identity_probe_points_matched_users_to_dataset_queue_first -q --tb=short` returned `3 passed in 6.28s`.
- [x] Full assignment-store validation passed after the multi-user handoff and identity-probe hardening: `PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_assignment_store.py -q --tb=short` returned `134 passed in 13.45s`.
- [x] Focused launch-bundle/checklist inspection validation passed: `scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_assignment_store.py::test_inspect_handoff_launch_evidence_execution_checklist_reports_directory_and_zip tests/unit/fisheye/test_labeling_assignment_store.py::test_export_launch_bundle_cli_writes_plan_readiness_handoffs_and_zip -q` returned `2 passed, 1 warning in 0.68s`.
- [x] Browser-smoke evidence templates include the guarded `/labeling?expected_user=<user>` URL base used by correct-user and wrong-user smoke rows.
- [x] Top-level launch-bundle manifests include `browser_mutation_write_policy` and `browser_mutation_write_checklist`, matching the handoff-level training-Zarr/no-CSV mutation contract exposed elsewhere.
- [ ] Deployment-specific browser/proxy evidence, identity-source evidence, mutable-Zarr backup evidence, representative browser smoke evidence, and disposable-Zarr mutation smoke evidence have not been executed in this working session.

### Launch-bundle implementation-status diagnostics

- [x] Launch bundles include `implementation-status.txt` as advisory status metadata, not launch approval.
- [x] Launch README, HTML index, implementation-status file, and operator evidence command sheet all state that stale packages missing the complete `implementation_status_artifact` contract fail closed.
- [x] Those generated operator surfaces also state that safe sharing requires `implementation_status_checklist_artifact_complete=true` in `shareability.safe_to_share_requires`.
- [x] Those generated operator surfaces also point wrappers to `implementation_status_checklist_artifact_gate_contract` in `inspection-targets.json`.
- [x] Those generated operator surfaces also point wrappers to `shareability_repair_command_contracts` for repair UI over browser-mutation target, direct Start/Open, single-owner, runtime route-checklist, and implementation-status regeneration failures.
- [x] Those generated operator surfaces also point wrappers to compact `shareability_contract` / `shareability.contract` for one-object safe-share gating.
- [x] Those generated operator surfaces explicitly require `shareability_contract.safe_to_share=true` before sharing labeler links when wrappers use the compact contract.
- [x] The compact contract exposes `safe_to_share_required_value=true` and `safe_to_share_matches_required_value` so wrappers can gate without hardcoding the expected boolean.
- [x] `inspection-targets.json` advertises compact-contract safe-share observed field, required value, and match field metadata at root and per-target levels.
- [x] Those generated operator surfaces also tell wrappers to use `fields`, `field_count`, `source_fields`, and `source_field_count` to detect malformed or truncated compact contract payloads.
- [x] `validation-checklist.json`, dry-run payloads, and `inspect-handoff` expose the nested `implementation_status_artifact`, its required field list/count, and flat `implementation_status_*` companion fields so wrappers do not need to scrape README or HTML files.
- [x] `inspect-handoff` now projects validation-checklist artifact completeness into top-level and nested `shareability` fields: `implementation_status_checklist_artifact_present`, `implementation_status_checklist_artifact_complete`, `implementation_status_checklist_artifact_missing_fields`, and `implementation_status_checklist_artifact_missing_field_count`.
- [x] `inspect-handoff` also exposes `implementation_status_checklist_artifact_gate`, a compact summary object with schema, observed value, required value, match status, missing fields, fail-closed reason, mismatch reason, and repair command ID.
- [x] `inspection-targets.json` advertises a compact `implementation_status_checklist_artifact_gate_contract` exemplar so wrappers can interpret the gate object without scraping README or HTML.
- [x] Top-level `inspect-handoff`, nested `shareability`, and `inspection-targets.json` all expose `implementation_status_checklist_artifact_complete_required_value=true` plus `implementation_status_checklist_artifact_complete_matches_required_value` so wrappers can compare observed and required values without scraping prose.
- [x] `shareability.safe_to_share_requires` explicitly includes `implementation_status_checklist_artifact_complete` and `implementation_status_checklist_artifact_complete_matches_required_value`, so wrapper launch decisions treat the nested artifact contract as a required safe-share gate.
- [x] `inspection-targets.json` advertises those checklist completeness field names, the required `implementation_status_checklist_artifact_complete=true` value, the required-value mismatch blocking reason, plus the stale-package fail-closed reason and repair command ID for wrapper discovery.
- [x] Stale packages generated before the nested artifact contract fail closed with `implementation_status_artifact_incomplete`, emit a structured `regenerate_package_with_implementation_status_artifact` repair command, and can be diagnosed from those missing-field diagnostics without treating `implementation-status.txt` as safe-share approval.
- [x] Stale packages also expose `implementation_status_checklist_artifact_complete_required_value_mismatch` in shareability blocking reasons, giving wrappers a direct required-value mismatch reason.
- [x] Stale-package failure actions list the exact missing `implementation_status_artifact` required fields so operators can confirm the package needs regeneration rather than treating the advisory status file as launch approval.
- [x] The `regenerate_package_with_implementation_status_artifact` repair command carries structured `missing_fields`, `missing_field_count`, `repair_mode=regenerate_package`, `artifact_contract=implementation_status_artifact`, and the safe-share blocker ID for wrapper UI repair guidance.

### Queue-first labeler entry

- [x] `/`, `/me`, `/labeling`, and `/datasets` serve the simple personalized datasets-waiting queue-first landing experience.
- [x] Queue-first contract metadata records `/`, `/me`, `/labeling`, and `/datasets` as datasets-waiting aliases, with `/work` as the full-dashboard fallback.
- [x] Live labeler APIs, identity probes, validation/inspection artifacts, generated manifests, generated dataset queues, and roster CSVs expose `expected_user_labeling_home_url` so operators and wrappers can offer the human-readable guarded `/labeling?expected_user=<user>` alias without making it the preferred mutation/start URL.
- [x] `/work` remains the full dashboard and task-opening surface.
- [x] `/work` task-opening buttons are shown only when the task state is present in serialized `startable_task_states`; non-startable rows show operator-action guidance instead.
- [x] `/api/me/tasks`, `/api/me/datasets`, and `/api/me/identity` expose expected-user guarded self-links.
- [x] Dataset queue rows include recording counts, task counts, open-task counts, workflow counts, and guarded `/work` links.
- [x] Dataset queue `open_task_count`, waiting dataset counts, and labeler-start readiness count only startable tasks in `pending` or `in_progress`; non-startable incomplete tasks remain visible for operator diagnostics but block labeler start.
- [x] Labeler-facing dashboard and dataset queue labels say `startable` where counts reflect tasks that can actually open a browser labeling session.
- [x] The live personalized datasets-waiting page states the one-active-owner rule in labeler language: each recording has one active assigned owner, and only that current assignee can open or save browser labeling work.
- [x] The live personalized datasets-waiting page states that browser saves run through server-side assigned task/training Zarr writers while CSV/HTML/JSON/handoff files remain metadata-only.
- [x] The live personalized datasets-waiting page gives explicit labeler-facing status copy for assigned datasets still waiting for browser labeling, all assigned work complete, blocked/no-startable work requiring operator action, and no active assignment states.
- [x] Copied dataset, recording, and queue support details expose `non_startable_task_count` so blocked/future task states are visible to operators.
- [x] Recording workflow summaries distinguish startable, non-startable, incomplete, and complete task counts instead of treating all incomplete tasks as open.
- [x] Task rows include task IDs, workflow metadata, guarded task-filtered `/work` links, and redacted support details.
- [x] Roster, manifest, and dataset-queue `dataset_queue_preview_url` fields point at the guarded `/my-datasets?expected_user=<user>` queue page, while `canonical_dataset_queue_preview_url` keeps the guarded `/datasets?expected_user=<user>` fallback; per-dataset `/work` links remain task-opening/dashboard fallbacks.
- [x] Roster and handoff artifacts expose link-role metadata: guarded `/my-datasets` is the preferred queue link, guarded `/datasets` is the canonical queue fallback, guarded `/` and `/labeling` are queue-first start pages, guarded `/my-work` and `/work` are dashboard fallbacks, identity links are preflight checks, and task links are convenience entry hints.
- [x] `/datasets` task rows expose a direct browser-start action backed by the serialized `direct_browser_start_endpoint` and existing guarded task-open API, while keeping the guarded `/work` dashboard link as a fallback.
- [x] `/datasets` direct browser-start endpoints are advertised only for explicit startable task states: `pending` and `in_progress`.
- [x] `/datasets` direct browser-start buttons use only the serialized server-provided `direct_browser_start_endpoint`; the browser does not reconstruct task-open endpoints from `task_id`.
- [x] `/datasets` direct browser-start buttons also fail closed unless the row state is present in serialized `startable_task_states`.
- [x] The task-open API and session creator reject guessed or stale opens for task states outside `pending` and `in_progress`.
- [x] `/datasets` direct browser-start actions refuse endpoints outside the exact same-origin `/api/tasks/{task_id}/open` route shape before POSTing.
- [x] `/datasets` direct browser-start actions require the serialized endpoint's decoded task segment to match the `task_id` on the same queue row before POSTing.
- [x] Non-startable/completed dataset queue task rows do not advertise or reconstruct direct browser-start endpoints.
- [x] Dataset, recording, and task support details are copyable without exposing raw Zarr paths or direct storage scope.
- [x] Dataset, recording, and task support details include preferred-entry and guarded-link role metadata so operators can diagnose copied-link issues without opening bundle CSVs.
- [x] Live and exported dataset queues expose `dataset_queue_state` so lightweight clients can distinguish open work, no assignments, all-complete work, and operator-blocked/no-open recordings.
- [x] Live and exported dataset queues expose top-level `labeler_start_ready`, `labeler_start_status`, `labeler_action`, and start message/action fields for minimal clients.
- [x] Live `/api/me/tasks` and `/api/me/datasets` payloads expose `single_owner_policy` plus flat `single_owner_policy_*` fields so lightweight clients can assert recording-scoped one-owner/current-assignee-only labeling directly from the personalized queue APIs.
- [x] Live and exported work summaries, dataset queues, manifests, handoff indexes, dashboard status rows, and `labeler-roster.csv` rows expose `labeler_work_completion` plus flat `labeler_work_completion_*` fields so lightweight clients can classify each user as `waiting`, `complete`, `blocked`, `unassigned`, or `idle` without deriving state from separate progress and queue fields.
- [x] `/datasets` and `/work` copied support blocks include labeler work completion fields (`status`, `completed`, `has_waiting_work`, `ready_for_more_labeling`, and `operator_action_required`) so labelers can send clear completion/waiting/blocker evidence to operators.
- [x] Browser task-completion responses expose `post_completion_queue` plus top-level `labeler_work_completion_*` and `post_completion_return_*` fields so lightweight editors know whether to send the user back to guarded `/my-datasets?expected_user=<user>`, show all assigned work complete, or surface an operator-blocked state after a mutation.
- [x] `post_completion_queue` carries flat guarded recovery-link fields (`return_expected_user`, `return_personal_dataset_queue_url`, `return_personal_dataset_queue_expected_user_guarded`, `return_personal_work_url`, and `return_personal_work_expected_user_guarded`) alongside `next_labeler_url`.
- [x] Post-completion queue contracts repeat the mutation-target assertions: browser edits target server-owned task-scoped training Zarrs, CSV/handoff artifacts remain metadata-only control-plane files, and browsers do not receive direct Zarr write authority.
- [x] Keypoint, detection, detection-analysis, and subject-mask browser editors use the post-completion queue contract to return labelers to their personalized guarded dataset queue after a successful task-complete POST.
- [x] Browser editor chrome, session banners, stale/superseded support text, and authorization-context-aware access error pages return labelers to expected-user-guarded personalized `/my-datasets?expected_user=<user>` and `/my-work?expected_user=<user>` links when the intended user is known; context-free access error fallbacks still return to the personalized aliases instead of generic `/` or `/work` links.
- [x] Access-error copied support details include `return_expected_user`, `return_personal_dataset_queue_url`, `return_personal_dataset_queue_expected_user_guarded`, `return_personal_work_url`, and `return_personal_work_expected_user_guarded` so pasted diagnostics preserve and prove the same guarded personalized recovery links shown on the page.
- [x] JSON authorization/error payloads include the same guarded recovery-link metadata inside `authorization_context`, so browser wrappers can send labelers back to `/my-datasets?expected_user=<user>` without scraping HTML.
- [x] Dataset queue datasets, recordings, and tasks expose `labeler_start_ready`, `labeler_action`, and metadata-only mutation-target assertions for lightweight browser clients.
- [x] Live `/api/me/tasks` and `/api/me/datasets` payloads, generated manifests, aggregate handoff indexes, `inspection-targets.json`, and `labeler-roster.csv` rows expose a versioned `personalized_launch_readiness` summary object that combines the guarded personal queue link, labeler start state, work completion state, training-Zarr/no-CSV mutation target, safe-share gate, and external launch-evidence gap fields for lightweight clients.
- [x] `personalized_launch_readiness` is self-described with `fields` and `field_count`; roster CSV rows flatten the guarded personal queue URL, labeler state, external launch-evidence gap summary/todos/template paths/record command IDs, `browser_label_write_target=training_zarr`, `browser_writes_csv_or_handoff_files=false`, and `browser_has_direct_zarr_write_authority=false`.
- [x] `personalized_launch_readiness` derives its browser write-target assertions from existing nested browser-mutation/direct-start policy and checklist objects plus their flat CSV/status projections, preserving `training_zarr` and no-CSV/no-direct-Zarr facts in generated manifests without creating a separate writable CSV or handoff data path.
- [x] `personalized_launch_readiness` reconstructs external launch-evidence gap IDs, statuses, counts, template paths, record-command IDs, and todo field metadata from todo-only artifacts, including CSV-style JSON strings, so roster/spreadsheet consumers do not silently collapse pending evidence to zero gaps.
- [x] `personalized_launch_readiness` normalizes CSV/string boolean projections such as `"False"` and `"0"` before computing start, queue-match, safe-share, no-CSV, and no-direct-Zarr readiness fields.
- [x] Live personal APIs, refreshed handoff work summaries, and standalone `work-summary` exports recompute `personalized_launch_readiness` after operator-validation/safe-share fields are present, so the nested and top-level readiness objects carry the same pending-evidence counts as the source safe-share contract.
- [x] Live `/api/me/identity`, `/api/me/tasks`, and `/api/me/datasets` selected-key responses emit `personalized_launch_readiness` explicitly at top level, not only inside nested identity/work objects, so lightweight clients can read the launch contract without reconstructing the full dashboard payload.
- [x] Fail-closed unknown-labeler responses from `/api/me/tasks`, `/api/me/datasets`, and `/api/me/identity` preserve the same top-level `personalized_launch_readiness` shape, guarded personal queue URL, and training-Zarr/no-CSV/no-direct-Zarr assertions so wrappers can render safe next steps without special-casing missing assignment users.
- [x] Fail-closed expected-user mismatch responses from `/api/me/tasks` and `/api/me/datasets` also preserve top-level `personalized_launch_readiness` for the guarded expected user plus training-Zarr/no-CSV/no-direct-Zarr assertions, while still returning no assigned work or dataset queue payload.
- [x] Fail-closed signed task-link expected-user mismatches preserve `personalized_launch_readiness` for the guarded expected user plus training-Zarr/no-CSV/no-direct-Zarr assertions in the access-problem support payload, while still refusing to open a session.
- [x] Direct task-open API success and denial payloads preserve top-level `personalized_launch_readiness`, guarded personal queue URLs, and direct-start write-policy assertions, so expected-user mismatch responses still prove training-Zarr-only browser writes, no CSV/handoff browser writes, and no direct browser Zarr authority while refusing session creation.
- [x] Task-completion denial payloads preserve top-level `personalized_launch_readiness`, guarded personal queue URLs, and direct-start write-policy assertions across task-scoped and session-scoped completion endpoints, so expected-user/session failures still prove no CSV/handoff browser writes and no direct browser Zarr authority while refusing task completion.
- [x] Labeler failed-promotion retry denial payloads preserve top-level `personalized_launch_readiness`, guarded personal queue URLs, and no-CSV/no-direct-Zarr assertions while still returning `operator_support_required` instead of giving labelers a browser mutation path.
- [x] `/work` and `/datasets` copied support details include compact `personalized_launch_readiness_*` lines plus the full readiness JSON, keeping operator diagnostics aligned with the live API contract for guarded personal queue links, external launch-evidence gaps, and training-Zarr/no-CSV/no-direct-Zarr assertions.
- [x] `/work` and `/datasets` copied support details also emit explicit `operator_support_*` lines for row-level guarded personal queue URL, personalized entry URL, training-Zarr write target, no CSV/handoff browser writes, no direct browser Zarr authority, and metadata-only CSV/handoff role, preferring row `operator_support` values when present.
- [x] Full `/work` task support plus `/datasets` dataset, recording, and task `operator_support` rows carry the guarded personal dataset-queue URL, personal/canonical queue roles, `browser_label_write_target=training_zarr`, no browser CSV/handoff writes, no direct browser Zarr authority, and metadata-only CSV/handoff role so copied row-level diagnostics remain self-contained.
- [x] `/my-datasets` and `/my-work` are guarded aliases for the personalized dataset queue and full work dashboard, using the same expected-user mismatch checks as `/datasets` and `/work`, and the labeler-route/expected-user guard contracts fail closed if those alias guards disappear.
- [x] `queue_first_entry_contract` exposes `personalized_labeler_entry_url`, requires it to match the expected-user-guarded `/my-datasets` URL for user-specific handoffs, and separately reports canonical `/datasets` as fallback.
- [x] `queue_first_entry_contract` exposes `personalized_entry_required`, `personal_dataset_queue_ready`, `personal_work_ready`, `personalized_labeler_entry_url_matches_personal_dataset_queue`, and `personalized_labeler_entry_url_is_expected_user_guarded` for wrapper gating.
- [x] Live admin/dashboard roster rows, nested status reports, manifests, aggregate handoff indexes, and roster CSVs expose `preferred_labeler_entry_url_matches_personal_dataset_queue=true`, so wrappers can assert the sendable start URL is the guarded `/my-datasets?expected_user=<user>` link rather than the canonical `/datasets` fallback.
- [x] CSV/readiness exports include explicit `personal_dataset_queue_link_role=preferred_queue`, `canonical_dataset_queue_link_role=canonical_queue_fallback`, and `preferred_labeler_entry_url_matches_personal_dataset_queue` fields alongside the legacy generic dataset-queue match field, so spreadsheet-based review can reject canonical-fallback-only launches.
- [x] Handoff entry-field helpers, aggregate indexes, and roster CSV rows also expose `personalized_labeler_entry_url_matches_personal_dataset_queue`, keeping preferred and personalized `/my-datasets` match diagnostics aligned across generated artifacts.
- [x] Validation checklist artifacts expose top-level `/labeling`, personalized `/my-datasets`, and `/my-work` paths/base URLs plus per-user guarded personalized URLs when available; aggregate checklist queue-first contracts prefer the base `/my-datasets` entry instead of falling back to canonical `/datasets`.
- [x] Dataset queues suppress direct-start endpoints and report `reassignment_session_safety_failed` when stale previous-owner sessions would block safe task opening or mutation.
- [x] The `/datasets` page displays a copyable reassignment-session safety panel when stale previous-owner sessions block labeler start.
- [x] Live APIs, exported work summaries, exported dataset queues, manifests, handoff indexes, and roster CSV rows expose `dataset_queue_direct_start_policy` so lightweight clients and operators can apply the same POST-only, same-origin, task-row-matched direct-start rules.
- [x] `dataset_queue_direct_start_policy` also declares that direct Start POST bodies must include the `expected_user` field, failed task-open responses return `task_open_authorization_contract`, denied-start support blocks preserve that contract plus authorization context, denied starts create no browser session, and the server reports `server_authorizes_open=false`.
- [x] `dataset_queue_direct_start_policy` and flattened roster/status fields explicitly name `task_scoped_training_zarr` as the label mutation target kind, `training_zarr` as `browser_label_write_target`, and `metadata_only_control_plane` as the CSV/handoff artifact role, with combined and split handoff/intermediate CSV artifacts marked as non-label-write targets and `browser_writes_handoff_csv=false`/`browser_writes_intermediate_csv=false`.
- [x] Flattened dashboard status rows, handoff indexes, and `labeler-roster.csv` rows expose `dataset_queue_direct_start_post_body_expected_user_*` and `dataset_queue_direct_start_denied_start_*` fields so wrappers and spreadsheets can assert guarded Start POST bodies and contract-rich denied-start support without parsing nested policy JSON.
- [x] Dataset queue task rows expose `direct_browser_start_authorization_contract`, plus flat readiness/expected-user/server-recheck fields, so each row declares whether direct start is safe and confirms that `/api/tasks/{task_id}/open` will recheck expected user, active assignment, task ownership, startable state, reassignment-session safety, and no browser Zarr/CSV write authority before creating a session.
- [x] Dataset queue task rows expose `direct_browser_start_not_ready_reason` and `direct_browser_start_not_ready_reasons`, and `/datasets` renders/copies the primary reason so labelers can distinguish completed tasks, non-startable tasks, missing task IDs, stale previous-owner sessions, and generic blocked starts.
- [x] Dataset queue and enriched `/work` task rows expose `direct_browser_start_operator_action`, and both pages render/copy it when direct start is not ready, including stale previous-owner session repair guidance and task-state reopen/move guidance.
- [x] The `/datasets` page only renders an active "Start browser task" button when the server-provided row-level `direct_browser_start_authorization_contract_ready` flag is true, in addition to the existing endpoint, task ID, labeler-start, and startable-state checks.
- [x] The `/datasets` and `/work` Start actions POST the current `expected_user` guard to `/api/tasks/{task_id}/open`, so personalized links are rechecked server-side at session creation time rather than trusted client-side.
- [x] Live and exported `/work` task rows are enriched with the same `direct_browser_start_authorization_contract` fields used by `/datasets`, so the dashboard fallback carries server-provided expected-user, active-assignment, task-ownership, startable-state, reassignment-safety, and no-direct-Zarr-authority evidence.
- [x] Enriched `/work` task-row `operator_support` payloads include the same direct-start readiness, not-ready reason, operator action, expected-user enforcement, server-recheck, `browser_label_write_target=training_zarr`, CSV/handoff no-write, and no-direct-Zarr-authority fields as dataset queue rows.
- [x] Live and exported `/work`, `manifest.json`, and `dataset-queue.json` payloads include `direct_browser_start_contract_summary`, aggregating ready/not-ready task counts, not-ready reason counts, operator-action counts, expected-user/server-recheck flags, task-scoped training-Zarr target metadata, CSV/handoff no-write flags, and no-direct-Zarr-authority assertions for wrapper-level gating.
- [x] Dashboard-roster rows/status reports, multi-user handoff indexes, and `labeler-roster.csv` rows expose nested and flat `direct_browser_start_contract_summary_*` fields so spreadsheet and wrapper tooling can gate direct browser start readiness without parsing every task row.
- [x] Dashboard status reports aggregate direct browser start readiness across users, including ready/not-ready user lists, task counts, not-ready reason counts, operator-action counts, missing-summary users, and a fail-closed dataset-queue start-readiness gate when any user cannot be authorized for direct browser start.
- [x] The `/work` dashboard fallback honors `direct_browser_start_authorization_contract_ready` when present on task rows, while retaining startable-state fallback behavior for legacy/raw task rows that do not yet carry the row contract.
- [x] The `/datasets` page renders and copies the server-provided labeler start/action state and mutation-target assertions instead of inferring readiness only from counts.
- [x] The `/work` dashboard dataset queue panel renders and copies the same server-provided labeler start/action and mutation-target assertions.
- [x] `/datasets` and `/work` copyable support blocks include `browser_mutation_write_checklist` ready/target/CSV-role fields, `browser_label_write_target=training_zarr`, split handoff/intermediate CSV non-label-target and no-write fields, and no-direct-Zarr-authority fields for lightweight operator diagnostics.
- [x] `/datasets` and `/work` copyable support blocks include flat `single_owner_policy_*` fields so operators can confirm recording-scoped one-owner/current-assignee-only mutation expectations from pasted labeler diagnostics.
- [x] Successful and denied browser-mutation responses expose the same server-owned task-scoped-training-Zarr target assertions, metadata-only CSV/handoff assertions, no-direct-Zarr-authority assertions, and flat runtime mutation-gate diagnostics so wrappers can assert that CSVs are never label-write targets even on failure paths.
- [x] `/datasets` and `/work` copyable support blocks include friendly personalized alias fields (`expected_user_personal_dataset_queue_url`, `expected_user_personal_work_url`, and `personalized_labeler_entry_url`) alongside canonical fallback links, with explicit `personal_dataset_queue_link_role`, `canonical_dataset_queue_link_role`, `dataset_queue_preview_url`, and `canonical_dataset_queue_preview_url` diagnostics.
- [x] Failed Start/Open copied support details in `/datasets` and `/work` include `authorization_return_*` guarded recovery-link fields copied from the denied response `authorization_context`.
- [x] `/datasets` and `/work` copied support blocks include operator-validation source, gate IDs, gate-count fields, and visibility-policy field lists/boundary flags so labeler-provided diagnostics can show exactly which launch evidence is still pending, needs review, or passed without exposing local checklist paths.
- [x] `/api/me/tasks`, `/api/me/datasets`, exported handoff JSON artifacts, and copied `/datasets`/`/work` support blocks expose flat per-gate `operator_validation_gate_<gate_id>_*` launch-evidence fields plus gate IDs, suffixes, and status vocabulary for wrapper-safe backup, response-security, identity-source, browser-smoke, disposable-Zarr smoke, and operator-recovery gating without parsing JSON gate lists.
- [x] `/work` copied dataset-queue support blocks expose the same direct-start policy diagnostics as `/datasets`, including route template, startable states, explicit handoff/intermediate CSV no-write fields, and no direct Zarr authority.
- [x] Dataset queue and dashboard copied support blocks include page context plus guarded preferred `/my-datasets`, canonical `/datasets`, and dashboard URLs so operators can reconstruct where a labeler was stuck.
- [x] Dashboard-roster counts and nested status reports expose aggregate preferred-entry readiness fields: `personalized_dataset_queue_preview_users`, `canonical_dataset_queue_preview_users`, `missing_personalized_dataset_queue_preview_users`, `all_users_have_personalized_dataset_queue_preview`, `dataset_queue_preferred_entrypoint_counts`, and `dataset_queue_link_role_counts`.
- [x] Dashboard-roster counts, nested status reports, and live admin summaries expose aggregate strict match fields: `preferred_personal_queue_match_users`, `missing_preferred_personal_queue_match_users`, `all_users_have_preferred_personal_queue_match`, `personalized_personal_queue_match_users`, `missing_personalized_personal_queue_match_users`, and `all_users_have_personalized_personal_queue_match`.
- [x] Aggregate queue-preview readiness is distinct from aggregate preferred-URL match readiness: a user may have a valid guarded `/my-datasets?expected_user=<user>` start URL even when their current queue is complete or blocked and therefore has no waiting preview.
- [x] Aggregate strict match readiness derives from preferred/personalized URL equality to `expected_user_personal_dataset_queue_url`, not only from precomputed row booleans, so legacy or partial rows with correct guarded URLs do not falsely fail launch gating.
- [x] Dashboard-roster ready-row draft readiness fails closed with `preferred_personal_queue_mismatch` when an otherwise-sendable row cannot prove `Start here` is the guarded `/my-datasets?expected_user=<user>` preferred queue; copy intent remains diagnostic until links are regenerated.
- [x] Dashboard-roster aggregate invite-reason counts and invite actions surface `preferred_personal_queue_mismatch`, so spreadsheet and wrapper flows can fail closed without inspecting every row.
- [x] Handoff package sendability fails closed with `preferred_personal_queue_mismatch` when a manifest can only prove canonical `/datasets?expected_user=<user>` fallback and cannot prove the guarded `/my-datasets?expected_user=<user>` preferred queue start URL.
- [x] Handoff sendability policy fixtures treat guarded `/my-datasets?expected_user=<user>` as required safe-link evidence; canonical `/datasets` alone remains a fallback and is not enough for a sendable handoff.
- [x] Handoff package inspection promotes `preferred_personal_queue_mismatch` from per-handoff sendability into package-level failure reasons and operator actions, so wrappers can distinguish preferred-link repair from generic `handoff_not_ready`.
- [x] Handoff package inspection emits a structured `handoff_regeneration` repair command with reason ID `preferred_personal_queue_mismatch`, pointing operators to regenerate handoffs with `export-user-handoffs --base-url ...` so preferred `/my-datasets` links are rebuilt.
- [x] Handoff package inspection shareability declares that safe sharing requires inspection success, no pending operator action, the browser-mutation target contract, the direct browser Start/Open contract, the single-owner package contract, and the runtime route-checklist gate; focused assertions pin the nested `shareability.safe_to_share_requires` list and contract summaries for wrappers.
- [x] Browser-mutation target, direct browser Start/Open, single-owner, and runtime route-checklist repair commands carry structured `contract`, `repair_mode`, `safe_share_blocker`, mismatch counts, mismatch users/recordings, and required-value details so wrappers can render precise regeneration guidance.
- [x] Top-level `operator_repair_commands` is accompanied by `operator_repair_command_detail_fields`, `operator_repair_command_detail_fields_by_id`, and `operator_repair_command_contracts`, keeping top-level repair rows self-describing.
- [x] Nested `shareability.repair_commands` carries the same enriched repair command rows as top-level `operator_repair_commands`, so wrappers that consume only the shareability object still receive structured repair diagnostics.
- [x] Nested `shareability` also carries `repair_command_detail_fields`, `repair_command_detail_fields_by_id`, and `repair_command_contracts`, so wrappers can interpret repair rows from one `inspect-handoff` response without opening `inspection-targets.json`.
- [x] `inspect-handoff` exposes a compact self-described `shareability_contract` object, mirrored at `shareability.contract`, that includes its own field list/count and source-field provenance/count, and bundles the current safe-share decision, blockers, repair IDs/count, safe-share gate, required field/value, required gates, implementation-status artifact gate, core package contracts, the runtime route-checklist gate, and repair-command contracts for one-object wrapper gating.
- [x] `inspection-targets.json` advertises the compact contract field names (`shareability_contract`, `shareability.contract`), `palette.web_labeling_handoff_shareability_contract.v1` schema, expected compact-contract field list/count, and source-field provenance map/count at root and per-target levels.
- [x] `inspection-targets.json` advertises `shareability_repair_commands_field`, `shareability_repair_command_detail_fields`, and `shareability_repair_command_detail_fields_by_id` at root and per-target levels so wrappers can discover enriched repair-command row shapes.
- [x] `inspection-targets.json` also advertises `shareability_repair_command_contracts`, including required training-Zarr targets, metadata-only CSV/handoff artifact roles, one-active-owner policy, runtime route-checklist proof, and implementation-status artifact regeneration expectations.
- [x] Admin summary payloads and the live admin UI surface the same aggregate preferred-entry readiness fields, so operators can confirm every invited user has a guarded `/my-datasets?expected_user=<user>` preferred queue before sending links.
- [x] The live admin UI surfaces runtime operator-validation start and mutation gates, including required/configured/ready/blocking status plus pending and missing-evidence gate IDs, so operators can see whether browser task opening and already-open-session mutations are currently allowed.
- [x] Live admin per-user rows display preferred/personalized queue match diagnostics (`preferred_matches_personal_queue` and `personalized_matches_personal_queue`) alongside entry-role text, so operators can spot canonical-fallback-only links before copying ready-row draft text.
- [x] Dashboard-roster HTML rows print explicit entry-role text for `entry=personal_datasets_waiting_queue`, `personal_queue=preferred_queue`, `queue=canonical_queue_fallback`, and `canonical_queue=canonical_queue_fallback`.
- [x] Labeler-facing API safety metadata states that task scope and raw Zarr paths are not returned to browsers.
- [x] Labeler-facing runtime state payloads redact raw Zarr paths, promotion training Zarr paths, and backend path fields.
- [x] Labeler-facing current-frame, save, action, review-status, and promotion responses redact raw Zarr paths and backend path fields.
- [x] Labeler-facing promotion retry responses redact raw Zarr paths and backend path fields before returning browser-visible results.
- [x] Labeler-facing error details redact absolute paths and Zarr-path tokens before returning browser-visible failures.
- [x] Labeler-facing browser payload redaction also removes path-like string values, not just path-like keys.
- [x] Labeler-facing work summaries and failed-promotion support rows redact path-like string values in nested error/details fields.
- [x] Labeler-facing redaction preserves server-generated app-local media URLs needed by browser editors.
- [x] Admin/operator diagnostics remain unredacted for repair workflows; redaction is scoped to labeler-facing browser payloads.
- [x] Labeler safety metadata records that the runtime surface is browser-only and requires no local Palette, Crimson, Conda, or project dependency installation.
- [x] Live labeler APIs expose safe `zarr_backup_policy`, `mutation_audit_policy`, and `session_guard_policy` metadata.
- [x] Live `/api/me/tasks` and `/api/me/datasets` expose top-level strict preferred/personal queue match booleans (`preferred_labeler_entry_url_matches_dataset_queue`, `preferred_labeler_entry_url_matches_personal_dataset_queue`, and `personalized_labeler_entry_url_matches_personal_dataset_queue`) so lightweight browser wrappers can gate `/my-datasets` launch readiness without reconstructing URLs.
- [x] `/datasets` and `/work` copied support blocks include the same strict preferred/personal queue match booleans, so labeler-provided diagnostics can show whether the copied start path is really the guarded `/my-datasets?expected_user=<user>` preferred queue.
- [x] Live labeler APIs and handoff artifacts expose `browser_mutation_write_policy`, making task-scoped training Zarrs the mutable label data plane while handoff CSV/HTML/JSON files remain metadata-only.
- [x] Live labeler APIs and handoff artifacts expose `labeler_route_authorization_policy`, making known-user, expected-user, active-assignment, startable task-state, session, signed-link, and forwarded-link recheck requirements explicit.
- [x] Live labeler APIs expose `labeler_route_authorization_checklist` so `/api/me/tasks` and `/api/me/datasets` report the active runtime authorization contract for the current browser user.
- [x] Live `/api/me/tasks` and `/api/me/datasets` payloads expose top-level `reassignment_session_safety` and flat `reassignment_session_safety_*` fields so browser wrappers can gate stale previous-owner sessions without parsing nested work.
- [x] Live `/api/me/tasks` and `/api/me/datasets` payloads expose safe public operator-validation status/source/gate-count diagnostics and `operator_validation_visibility_policy`, defaulting to the explicit fail-closed launch evidence gates without exposing local checklist paths.
- [x] Live `/api/me/tasks`, `/api/me/datasets`, and copied `/datasets`/`/work` support blocks expose the redacted runtime operator-validation mutation gate, so labelers and lightweight wrappers can distinguish task-open/start blocking from already-open-session mutation blocking without exposing local checklist paths.
- [x] Live `/api/me/tasks`, `/api/me/datasets`, standalone `work-summary` JSON/stdout summaries, dashboard roster exports/status rows, static handoff artifacts, aggregate handoff indexes, roster CSV rows, and copied `/datasets`/`/work` support blocks expose the safe `runtime_operator_validation_gate_cli_policy`, so lightweight wrappers can show the preferred browser-work gate flag and verify that the configured runtime gate protects Start/Open, mutation writes, target-token checks, Zarr writes, and audit-event creation.
- [x] Operator-validation public-field extraction and visibility policy classify identity personal-queue proof aggregates (`identity_personal_queue_evidence_*` and `identity_all_users_have_personal_queue_evidence`) as safe support metadata, not labeler instructions or local operator-only paths.
- [x] Public identity personal-queue evidence fields include `identity_personal_queue_evidence_status` so wrappers can distinguish no recorded deployment proof (`missing`) from partially recorded but incomplete proof (`incomplete`) and approved aggregate proof (`ready`).
- [x] Public identity personal-queue evidence status is derived from aggregate proof fields and fails closed; `ready` requires `identity_all_users_have_personal_queue_evidence=True`, positive ready-user/count evidence, and no missing count/user/missing-field evidence.
- [x] Validation-checklist and package inspection derive identity personal-queue evidence status from aggregate proof fields as well, so stale checklist/package status strings cannot make archived handoffs appear ready.
- [x] Live operator-validation visibility policy and roster CSV exports advertise the allowed identity personal-queue evidence status values (`missing`, `incomplete`, `ready`), matching the archived inspection-target contract.
- [x] Identity personal-queue evidence status values and fail-closed derivation are centralized in the web-labeling implementation so live payloads, roster CSVs, validation-checklist inspection, package inspection, shareability, and inspection-target metadata cannot drift independently.
- [x] Browser route assertions for `/api/me/tasks` and `/api/me/datasets` use the centralized identity personal-queue evidence status values, matching roster and inspection-target assertions.
- [x] Live `/api/me/tasks`, `/api/me/datasets`, and the personalized dataset-queue HTML support block expose identity personal-queue proof status and allowed status values for browser-wrapper gating.
- [x] Admin-page fallback ready-row draft copy, generated labeler messages, quickstarts, and labeler HTML consistently direct users to the guarded personalized dataset queue, expose `/labeling?expected_user=<user>` as a human-readable alias, and state that browser saves are server-side assigned task/training-Zarr mutations while CSV/HTML/JSON/handoff files are metadata only.
- [x] Live `/api/me/tasks` work payloads, `/api/me/datasets`, standalone `work-summary`, per-user `work-summary.json`, `dataset-queue.json`, `manifest.json`, validation checklists, and aggregate handoff indexes expose the same `queue_first_entry_contract` for wrapper-safe `/my-datasets?expected_user=<user>` gating.
- [x] Fail-closed `/api/me/tasks` and `/api/me/datasets` unknown-labeler responses include guarded queue-first route metadata, link roles, and `queue_first_entry_contract`, so wrappers can explain bad-login/no-assignment states without falling back to dashboard URLs.
- [x] `labeler_route_authorization_checklist.ready` fails closed when the resolved user has no active assignments, even if they are a known paused/inactive labeler, because task opens and browser mutations require an active assignment.
- [x] Flat route-authorization diagnostics expose `labeler_route_authorization_active_assignment_required`, `labeler_route_authorization_active_assignment_count`, and `labeler_route_authorization_has_active_assignment` in copied support blocks and `work-summary --output` summaries, so wrappers can fail closed on no-assignment states without parsing the nested checklist.
- [x] Identity probe API/HTML support payloads expose the same `queue_first_entry_contract`, so the first page labelers are told to open also carries the personalized queue readiness contract.
- [x] Per-user `queue_first_entry_contract.ready` requires expected-user-guarded preferred and personalized queue entry URLs when expected-user queue URLs are present; unguarded `/my-datasets` links fail closed even if the path itself matches the personal queue.
- [x] Queue-first contract guard booleans distinguish path matching from expected-user guarding: a generic `/my-datasets` URL can match the personal queue path but does not satisfy `*_is_expected_user_guarded`.
- [x] Browser copied support details expose flat `queue_first_entry_contract_*` fields, including ready status, preferred/personalized entrypoint URLs, personal-queue match booleans, expected-user-guard booleans, and queue-first readiness preconditions.
- [x] Handoff roster CSV exports flat `queue_first_entry_contract_*` fields, including ready status, preferred/personalized guarded `/my-datasets` entry URLs, personal-queue match booleans, expected-user-guard booleans, and queue-first readiness preconditions for spreadsheet-wrapper gating.
- [x] Live `/api/me/datasets` exposes the same browser response-security policy as `/api/me/tasks`, so `/datasets` support diagnostics can report deployed no-store/clickjacking/header expectations.
- [x] Browser response-security policy metadata lists protected personalized labeler routes (`/labeling`, `/my-datasets`, `/my-work`), canonical fallbacks (`/datasets`, `/work`), and personal APIs (`/api/me/tasks`, `/api/me/datasets`) so wrappers can require hardened headers on the actual queue-first entrypoints.
- [x] Browser response-security static readiness fails when the protected route set omits `/labeling`, personalized `/my-datasets`, or `/my-work` aliases, even if the legacy canonical/header fields are otherwise present.
- [x] Handoff response-security flattened fields expose protected-route lists and per-group readiness booleans so roster/index consumers can see personalized alias header coverage without parsing nested policy JSON.
- [x] Live admin/preflight UI text displays browser response-security protected routes, personalized aliases, and the alias/canonical header parity flag for quick operator review.
- [x] Per-user `work-summary` embedded `work` payloads and dashboard-roster user/status rows expose only public operator-validation diagnostics, while operator-report top-level metadata may retain the local checklist path for traceability.
- [x] Work-summary and dashboard-roster reports expose `operator_validation_visibility_policy`, marking `operator_validation_checklist_path` as operator-only and declaring that per-user/labeler-visible payloads use public validation fields only.
- [x] Dashboard-roster nested `status_report.operator_validation` exposes public validation fields only and carries `operator_validation_visibility_policy`; top-level operator metadata remains the place for local checklist paths.
- [x] Public `operator_validation` projections preserve exact filtered safe-share blocker fields and compact `safe_share_next_action_summary` when present, so nested live/status payloads do not lose checklist-side safe-share diagnostics while still omitting operator-only checklist paths.
- [x] Dashboard-roster user rows, nested `status_report.operator_validation`, and nested `status_report.user_statuses` expose flat `operator_validation_gate_<gate_id>_*` launch-evidence fields while preserving the operator-only checklist path boundary.
- [x] Live admin summaries, preflight payloads, dashboard-roster top-level JSON, dashboard-roster user rows, dashboard-roster CSV rows, and nested dashboard status reports expose the centralized `safe_share_gate`/flat `safe_share_*` contract so live operator tooling does not treat `ready_to_invite` or `ready_to_send` as sufficient for link sharing.
- [x] Live admin summaries, preflight payloads, dashboard-roster top-level JSON, dashboard-roster command summaries, dashboard-roster user/CSV rows, and nested dashboard status reports expose filtered safe-share blocker fields plus compact `safe_share_next_action_summary` derived from operator-validation diagnostics, including explicit `missing_evidence` blockers when no approved launch checklist evidence is configured.
- [x] Focused admin summary and preflight CLI coverage asserts `safe_share_next_action_summary` in live readiness payloads and archived preflight reports, so operators can see compact launch blockers before exporting handoffs.
- [x] Dashboard-roster CSV coverage asserts the filtered safe-share blocker counts and missing/unsatisfied gate lists, so ready-row draft spreadsheet consumers cannot silently lose fail-closed launch evidence diagnostics.
- [x] Safe-share blocker diagnostics include structured `safe_share_launch_blocking_next_actions`, `safe_share_launch_blocking_next_action_count`, and compact `safe_share_next_action_summary` across JSON/CSV wrapper surfaces, so operators can render per-gate evidence todos without parsing prose.
- [x] Operator-facing multi-user handoff `index.html` and `README.txt` render a fail-closed safe-share next-action summary with blocker count and gate statuses, while per-labeler pages remain focused on browser work links.
- [x] The deployment runbook now states the safe-share launch rule up front: do not send links on `ready_to_send` alone, require `labeler_links_safe_to_share=true`, use guarded `/my-datasets?expected_user=<user>` as the preferred labeler entry, keep CSV/handoff artifacts metadata-only, and confirm one active owner per recording.
- [x] First-batch, production-decision, validation-log, and assignment-implementation operator docs now repeat that `ready_to_invite`/`ready_to_send` are readiness signals only; final link sharing requires `labeler_links_safe_to_share=true`, guarded personalized `/my-datasets` entry, assigned training-Zarr browser writes, and metadata-only CSV/handoff artifacts.
- [x] First-batch operator guidance now gives a copyable `inspect-handoff --require-shareable` command, and the validation log template records both the safe-share inspection command and report path as launch evidence.
- [x] Admin/dashboard copy controls and generated operator command sheets now label copied ready-row text as draft material and require safe-share inspection before sharing, preventing `ready_to_invite` rows from being presented as final send-approved invitations.
- [x] Dashboard/admin roster `copy_intent` now uses `ready_row_draft` instead of `send_invitation` for ready rows, so wrappers cannot mistake row readiness for final approval to contact labelers.
- [x] Dashboard roster JSON keeps legacy `ready_invitations*` compatibility fields but marks them as `draft_text_only_safe_share_required`, and adds `ready_row_drafts`, `ready_row_draft_text`, and `ready_row_draft_share_rule` so new wrappers can avoid treating copied text as share-approved invitations.
- [x] Runtime admin/dashboard HTML, validation-log wording, and focused assertions now use ready-row draft terminology for copied work text and reserve final sharing for the safe-share inspection result.
- [x] Dashboard/admin roster count payloads keep legacy `ready_to_invite*` compatibility fields but mark their semantics as row readiness only and add `ready_row_draft_count`, `diagnostic_note_count`, `ready_row_draft_users`, and `diagnostic_note_users` aliases keyed to the safer copy-intent contract.
- [x] Live admin `dashboard_user_counts` exposes the same draft-safe count aliases and `ready_to_invite_legacy_semantics` marker as exported dashboard-roster payloads, keeping runtime and archived wrapper semantics aligned.
- [x] Dashboard roster HTML rendering now prefers `ready_row_draft_text` and only falls back to legacy `ready_invitations_text`, so the rendered operator copy block follows the draft-safe payload contract.
- [x] Ready-row draft semantics and share-rule strings are centralized in the dashboard roster bundle helper, so legacy `ready_invitations*` compatibility fields and draft-safe aliases cannot drift independently.
- [x] Dashboard roster payloads expose versioned ready-row draft bundle metadata (`ready_row_draft_bundle_schema`, `ready_row_draft_bundle_kind`, and `ready_invitations_legacy_field_names`) so wrappers can identify legacy invitation-named fields as draft-only compatibility aliases.
- [x] Dashboard roster user rows and nested `status_report.user_statuses` include per-row ready-row draft bundle metadata plus safe-share required-field/value flags, so CSV-only and row-only wrappers can enforce `labeler_links_safe_to_share=true` without parsing top-level JSON.
- [x] Launch-bundle `inspection-targets.json` advertises ready-row draft top-level fields, row fields, bundle schema/kind, and the required safe-share field/value for wrapper discovery.
- [x] Launch-bundle `operator-evidence-commands.txt` advertises the ready-row draft bundle/row fields and repeats that wrappers must still require `labeler_links_safe_to_share=true`.
- [x] Ready-row state allowed values (`ready_row_draft`, `diagnostic_note`) are centralized and advertised in `inspection-targets.json` plus operator command-sheet guidance for wrapper validation.
- [x] Live admin counts, dashboard-roster top-level payloads, and nested status reports expose `ready_row_state_values` so clients can validate row states without opening inspection-target metadata.
- [x] Dashboard roster user rows and nested `status_report.user_statuses` now carry `ready_row_state_values`, so CSV-only and row-only wrappers can validate `ready_row_state` without top-level metadata.
- [x] `copy_intent_values` is centralized with the ready-row vocabulary and exposed in live admin counts, dashboard-roster top-level payloads, nested status reports, user rows, and nested status rows for wrapper validation.
- [x] Launch-bundle `inspection-targets.json` and `operator-evidence-commands.txt` advertise `copy_intent_values`, matching the ready-row vocabulary for wrapper validation.
- [x] Legacy ready-row draft compatibility field names (`ready_invitations`, `ready_invitations_text`) are centralized and advertised in inspection targets plus operator command-sheet guidance.
- [x] Launch-bundle `inspection-targets.json` and `operator-evidence-commands.txt` advertise the flat browser-mutation target CSV fields and required values, and `inspection-targets.json` also advertises the server-owned target-selector rejection policy plus CSV/Zarr/write-target field names, so wrappers can assert `browser_label_write_target=training_zarr`, no handoff/intermediate CSV writes, and no browser-selected CSV/Zarr mutation targets before links are shared.
- [x] `inspect-handoff` now validates those browser-mutation target required values and fails closed with `browser_mutation_target_contract_mismatch` if a handoff would imply browser writes target anything other than server-owned task-scoped training Zarrs or if handoff/intermediate CSV artifacts are write targets.
- [x] Handoff inspection repair diagnostics include a dedicated handoff-regeneration command with reason id `browser_mutation_target_contract_mismatch`, so wrappers can route wrong mutation-target packages to regeneration instead of treating the reason as opaque.
- [x] `inspect-handoff`, `inspection-targets.json`, and operator command sheets now validate and advertise the direct browser Start/Open contract: policy metadata must be present, Start/Open must use POST same-origin expected-user guarded task-open requests, denied starts must report authorization contracts, task targets stay task-scoped training Zarrs, and handoff/intermediate CSV artifacts remain non-write targets.
- [x] Direct Start/Open contract mismatches fail closed with `direct_browser_start_contract_mismatch` and a dedicated handoff-regeneration repair command, keeping copied start links from being shared when the server-recheck/no-CSV/no-direct-Zarr contract is stale.
- [x] `inspect-handoff` validates the package-level one-active-owner rule by deriving recording owners from each handoff assignment snapshot and failing closed with `single_owner_package_contract_mismatch` when a recording appears under multiple active labelers.
- [x] Single-owner package mismatches expose duplicate recording/user maps plus a dedicated handoff-regeneration repair command, so wrappers can block sharing stale/corrupt packages without parsing per-user manifests.
- [x] Generated handoff README/operator guidance now names the compact inspection fields `browser_mutation_target_contract_met`, `direct_browser_start_contract_met`, `single_owner_package_contract_met`, and `labeler_route_authorization_runtime_checklist_gate_met`, so humans and wrappers can see the extra fail-closed contract gates alongside safe-share evidence gates.
- [x] Row-level ready-row draft metadata defaults are centralized in one helper and reused by roster row construction plus nested status projection, preventing CSV-only safe-share fields from drifting.
- [x] Ready-row draft required safe-share field/value constants are centralized and reused by the row metadata helper and share-rule text.
- [x] Nested dashboard status reports expose the same ready-row draft count/user aliases and legacy `ready_to_invite` semantics marker as top-level roster/admin counts, so JSON wrappers do not need special-case status-report payloads.
- [x] Dashboard roster user rows and nested `status_report.user_statuses` include row-level `ready_to_invite_legacy_semantics` and `copy_intent`, making row readiness versus draft-copy intent explicit in JSON, CSV, and nested status projections.
- [x] Dashboard roster user rows and nested `status_report.user_statuses` expose `ready_row_state` (`ready_row_draft` or `diagnostic_note`) as a draft-safe alias for legacy `ready_state` values.
- [x] Admin and dashboard roster HTML count labels now say ready rows / ready-row draft users and explicitly state that ready rows remain draft text until safe-share inspection passes.
- [x] Admin and dashboard roster per-user row status text now says `ready row draft; safe-share review required` instead of plain `ready`, keeping row-level display semantics aligned with the safe-share contract.
- [x] Remaining human-facing `ready to invite` prose in admin user display, dashboard warning details, and reassignment runbook guidance now uses ready-row draft plus safe-share inspection wording.
- [x] `dashboard-roster` CLI help now describes the output as a ready-row draft roster and states that safe-share inspection is still required before sharing links.
- [x] Dashboard roster HTML titles, parser option help, not-ready dashboard messages, first-batch guidance, and deployment-runbook copy guidance now describe copied text as ready-row drafts rather than send-approved invitations.
- [x] Operator evidence guidance, validation-gate text, and deployment-runbook recovery/dry-run steps now say labeler link sharing or ready-row drafts instead of treating invitations as the launch decision.
- [x] Operator-validation payloads now expose `operator_launch_approved_legacy_semantics`, `operator_launch_approved_is_safe_share_approval=false`, and required safe-share field/value metadata so `--operator-launch-approved` cannot be mistaken for final link-sharing approval.
- [x] Operator-validation payloads also expose legacy-semantics metadata for `operator_validation_required_before_invite`, marking it as a ready-row-draft validation gate rather than safe-share approval and pointing wrappers to `labeler_links_safe_to_share`.
- [x] Invalid configured validation-checklist fallbacks preserve the same operator-validation/safe-share interpretation metadata, so error payloads cannot accidentally imply safe-share approval.
- [x] Operator-validation approval-scope metadata is centralized in one helper and reused by checklist-backed, manual-approved, default-pending, and invalid-checklist fallback paths.
- [x] Validation gates preserve legacy `blocks_invitation` but now include `blocks_invitation_legacy_semantics`, `blocks_invitation_is_safe_share_approval=false`, and `blocks_invitation_safe_share_field=labeler_links_safe_to_share` in generated gates and inspection/update summaries.
- [x] Validation-gate `blocks_invitation` interpretation metadata is centralized in one helper and reused by generated gates plus older-checklist projection fallbacks.
- [x] Safe-share next-action summaries fail closed when only generic operator-validation fields are available: listed blockers keep their status and unlisted launch blockers remain visible as `unknown` rather than disappearing.
- [x] Safe-share exact-field projection preserves partial artifacts that contain only `safe_share_launch_blocking_next_actions`, reconstructing status maps, missing/unsatisfied lists, counts, and the compact summary instead of discarding the action array.
- [x] Safe-share exact-field projection also preserves partial artifacts that contain only `safe_share_external_launch_evidence_gap_todos`, reconstructing gate statuses, pending/missing/unsatisfied lists, counts, and template/command metadata instead of discarding the per-gate evidence todo rows.
- [x] Safe-share projection preserves summary-only artifacts that contain `safe_share_next_action_summary` without inventing detailed gate lists, so compact wrapper diagnostics survive partial upgrades.
- [x] Operator-only command sheets and `inspection-targets.json` advertise `safe_share_launch_blocking_next_actions`, `safe_share_launch_blocking_next_action_count`, and `safe_share_next_action_summary`, so wrapper authors can discover both machine-readable and compact per-gate evidence todo contracts.
- [x] Launch-bundle `inspection-targets.json` advertises `shareability_safe_to_share_requires`, matching the nested `shareability.safe_to_share_requires` contract (`inspection_ok`, no pending operator action, browser-mutation target contract, direct browser Start/Open contract, single-owner package contract, and runtime route-checklist gate), so wrappers can enforce the current safe-share rule without parsing implementation code.
- [x] Live admin summaries and preflight payloads derive top-level operator-validation and filtered safe-share blocker diagnostics from the configured validation checklist when one is supplied, while invalid checklists fail closed instead of falling back to default missing-evidence state.
- [x] Top-level dashboard-roster JSON/stdout summaries and live admin/preflight operator-validation reports expose the same flat `operator_validation_gate_<gate_id>_*` launch-evidence fields, gate IDs, suffix metadata, and status vocabulary so wrappers do not need to parse nested gate lists.
- [x] Standalone `work-summary` CLI exports `known_user_status`, `labeler_route_authorization_policy`, and computed `labeler_route_authorization_checklist` at top level and inside the embedded `work` payload.
- [x] Standalone `work-summary` CLI exports the same labeler-safety, operator-authorization, operator-recovery, Zarr-backup, mutation-audit, browser-response-security, and session-guard policy blocks used by handoff work summaries.
- [x] Standalone `work-summary` CLI exports single-owner policy, assignment ownership integrity, schema primary-key evidence, duplicate-owner counts, and store consistency so wrappers can confirm one owner per recording before previewing or exporting labeler work.
- [x] Standalone `work-summary` CLI accepts `--operator-validation-checklist` or `--operator-launch-approved` and exports the same operator-validation gate fields used by ready-row draft rosters, so per-user work summaries can fail closed until launch evidence is approved.
- [x] When no operator-validation checklist is supplied, standalone `work-summary` and roster exports fail closed with explicit pending launch gate IDs for mutable-Zarr backup evidence, browser response-security headers, identity-source verification, browser smoke, disposable-Zarr mutation smoke, and operator recovery contract evidence.
- [x] Standalone `work-summary` JSON, stdout summaries, and embedded `work` payloads expose the centralized `safe_share_gate` plus filtered `safe_share_launch_blocking_*` blocker fields derived from operator-validation state, so one-user wrappers can fail closed without parsing dashboard-roster or package inspection artifacts.
- [x] Legacy or partial pending operator-validation payloads are normalized in public diagnostics to `operator_validation_source=none` with the same explicit pending launch gate IDs and operator action.
- [x] Partial legacy pending operator-validation payloads with only pending gate IDs or only missing-evidence gate IDs are normalized so public diagnostics keep the two gate lists and counts aligned.
- [x] Standalone `work-summary --output` stdout summary exposes store consistency ok/status counters, issue/warning counts, issue/warning code lists, blocking warning counts, and blocking warning codes for wrapper fail-closed checks.
- [x] Standalone `work-summary --output` stdout summary exposes `reassignment_session_safety_*` fields so wrappers can fail closed on stale previous-owner sessions without reopening the JSON artifact.
- [x] Exported user handoffs include `store_consistency`/`reassignment_session_safety` evidence and emit a dedicated `reassignment_session_safety_failed` sendability reason/action when stale previous-owner sessions would block safe labeling.
- [x] Exported user handoffs preserve full store consistency evidence but apply reassignment-session sendability per user, so unaffected labelers are not blocked by another recording's stale previous-owner session.
- [x] Standalone `work-summary --output` stdout summary includes personalized `/my-datasets` entry URL, canonical `/datasets` fallback URL, labeler start message/operator action, recordings-without-open-tasks counts, reason buckets, and operator actions so wrappers can explain blocked or empty queues without reopening the JSON artifact.
- [x] Standalone `work-summary --output` stdout summary includes known-user, active-assignment count, queue-start readiness/status/action, waiting/open dataset-work counts, route-authorization readiness, operator-boundary/recovery readiness, operator-validation status/gate IDs/counts/action/visibility-boundary flags, runtime operator-validation gate CLI fields, flat per-gate launch-evidence fields/status vocabulary, browser-mutation-write readiness, direct-start enabled, no-local-install, backup, audit, response-security, session-guard, task-state, and signed-link flags for wrapper gating without reopening the JSON artifact.
- [x] Standalone `work-summary --output` stdout summary explicitly reports task-scoped training-Zarr mutation target kind, `browser_label_write_target=training_zarr`, metadata-only CSV/handoff role, CSV/handoff non-label-write-target status, no browser handoff/intermediate CSV writes, and no direct browser Zarr authority for both browser mutation and direct-start contracts.
- [x] Standalone `work-summary --output` stdout summary now exposes compact gate booleans and mismatch counters for `browser_mutation_target_contract_met`, `direct_browser_start_contract_met`, and `single_owner_policy_contract_met`, so one-user wrappers can fail closed without re-comparing every raw field.
- [x] Live work payloads, dashboard-roster user/status rows, refreshed handoff work summaries, generated `dataset-queue.json`, and handoff manifests now carry the same compact contract booleans, keeping browser wrappers, operator dashboard wrappers, and static package wrappers aligned on mutation-target, direct-start, and one-owner gates.
- [x] Dashboard-roster top-level `counts` and nested `status_report` aggregate those compact gates into all-users-met booleans, not-met user lists, and mismatch totals so operator wrappers can fail closed without iterating every user row.
- [x] Dashboard-roster CSV output rows and printed CSV-output summaries expose the same compact gate booleans, mismatch counters, all-users-met values, and `dataset_queue_start_readiness`, so CSV-only operator handoffs can assert training-Zarr/no-CSV mutation targets, guarded direct Start/Open, and one-owner assignment safety.
- [x] Dashboard-roster HTML exposes the same compact contract aggregate gates and per-user gate values, so a human operator can see training-Zarr/no-CSV mutation target safety, guarded direct Start/Open safety, and one-owner assignment safety without opening JSON or CSV artifacts.
- [x] Dashboard-roster HTML output summaries now mirror the CSV output summaries for compact contract all-users-met values and `dataset_queue_start_readiness`, so automation launched from the HTML export path can fail closed without scraping the rendered page.
- [x] `dataset_queue_start_readiness` now fails closed on compact mutation-target, direct-start, and one-owner contract aggregate failures, so the queue-start gate cannot pass when browser writes would imply CSV/handoff targets, direct Start/Open contract drift, or multiple active labelers per recording.
- [x] Generated handoff manifests attach `assignment_ownership_integrity` and `single_owner_policy` before computing `single_owner_policy_contract_met`, so the compact one-owner gate is evidence-backed instead of inferred from defaults.
- [x] Aggregate handoff index rows and `labeler-roster.csv` exports include `browser_mutation_target_contract_met`, `direct_browser_start_contract_met`, `single_owner_policy_contract_met`, and mismatch counts so spreadsheet/package wrappers can gate without opening each per-user JSON file.
- [x] `inspection-targets.json` and operator command sheets advertise the compact per-payload contract gate field names, so wrappers can discover `browser_mutation_target_contract_met`, `direct_browser_start_contract_met`, `single_owner_policy_contract_met`, and mismatch counters without parsing README prose.
- [x] `/datasets` and `/work` copied support blocks include the compact contract booleans and mismatch counters, so pasted labeler diagnostics preserve mutation-target, direct-start, and one-owner gate state without requiring JSON artifact access.
- [x] `/datasets` and `/work` copied support blocks include `labeler_route_authorization_checklist` fields for expected-user match, known-user/active-assignment required/count/presence, startable task-state, current-session mutation, current target-token, signed-link entry-hint, and forwarded-link identity recheck diagnostics.
- [x] Handoff `manifest.json`, `work-summary.json`, and `dataset-queue.json` include the computed `labeler_route_authorization_checklist`, and `/datasets` plus `/work` copied support blocks include handoff readiness, sendability reasons/actions, operator-validation status/action, and link expiration diagnostics when present.
- [x] Static handoff `manifest.json`, `work-summary.json`, and `dataset-queue.json` include `operator_validation_visibility_policy` so archived bundles preserve the same public/operator-only validation field boundary as live APIs and CLI reports.
- [x] Static handoff `manifest.json`, `work-summary.json`, embedded `work`, and `dataset-queue.json` include the centralized `safe_share_gate`, flat `safe_share_*` contract, and filtered `safe_share_launch_blocking_*` gate-status fields so offline per-user bundles can fail closed before aggregate inspection.
- [x] Static `validation-checklist.json` files and `inspect-handoff` validation-checklist summaries include `operator_validation_visibility_policy` plus the centralized `safe_share_gate`/flat `safe_share_*` contract, so operator launch evidence, archive inspection, and wrapper tooling preserve the same public/operator-only boundary and fail-closed link-sharing rule.
- [x] Aggregate multi-user handoff indexes and launch-bundle manifests include `operator_validation_visibility_policy`, centralized safe-share blocker fields, and compact `safe_share_next_action_summary`; per-user rows copy the user manifest visibility policy for wrapper-safe roster/index consumption.
- [x] Aggregate multi-user handoff index rows expose flat per-gate `operator_validation_gate_<gate_id>_*` launch-evidence fields for the required backup, response-security, identity-source, browser-smoke, disposable-Zarr smoke, and operator-recovery gates, matching per-user handoff JSON and roster CSV rows.
- [x] Handoff roster CSV exports include operator-validation visibility-boundary columns so spreadsheet wrappers can see that checklist paths are operator-only without parsing JSON indexes.
- [x] Handoff roster CSV exports flat per-gate `operator_validation_gate_<gate_id>_*` launch-evidence columns for the required backup, response-security, identity-source, browser-smoke, disposable-Zarr smoke, and operator-recovery gates, so spreadsheet wrappers can fail closed without parsing JSON gate lists.
- [x] Handoff roster CSV exports filtered `safe_share_launch_blocking_*` gate-status/count columns plus `safe_share_checklist_gate_evidence_complete`, so spreadsheet wrappers can distinguish missing, pending, review, unknown, satisfied, and unsatisfied safe-share blockers without parsing JSON manifests.
- [x] Operator-validation visibility policy and roster CSV exports declare allowed flat per-gate launch status values (`unknown`, `pending`, `missing_evidence`, `needs_review`, `passed`) so wrappers can validate gate status strings without hardcoded undocumented assumptions.
- [x] Handoff roster CSV visibility columns include `operator_validation_gate_ids` and `operator_validation_gate_flat_field_suffixes` alongside status values, so spreadsheet wrappers can discover the full flat gate-field contract without parsing JSON artifacts.
- [x] Handoff roster CSV exports include identity personal-queue proof aggregate/status columns as metadata-only launch support fields, alongside assertions that browser label writes target task-scoped training Zarrs rather than handoff/intermediate CSVs.
- [x] Applying operator evidence templates backfills/refreshed each touched `manifest.json`, `work-summary.json`, and `dataset-queue.json` with the computed `labeler_route_authorization_checklist` before regenerating visible handoff artifacts.
- [x] Applying operator evidence templates backfills/refreshed `manifest.json`, `work-summary.json`, embedded `work`, `dataset-queue.json`, and aggregate handoff index rows with flat `operator_validation_gate_<gate_id>_*` launch-evidence fields so approved evidence refreshes cannot leave stale fail-closed wrapper metadata behind.
- [x] Applying operator evidence templates backfills refreshed `work-summary.json` and embedded `work` payloads with browser mutation write checklist and dataset-queue direct-start policy metadata used by copied support blocks.
- [x] Applying operator evidence templates reports refreshed visible JSON artifacts (`manifest.json`, `work-summary.json`, `dataset-queue.json`) and counts at both top level and in nested `handoff_refresh`, so wrappers can require checksum refresh without parsing prose.
- [x] Applying operator evidence templates exposes stable zero-count refreshed-file fields when no handoff manifests are found or handoff refresh is disabled, so wrappers do not need special-case schemas.
- [x] Applying operator evidence templates exposes stable skipped refresh counts/lists at both top level and in nested `handoff_refresh`, so wrappers can surface missing or malformed handoff artifacts without parsing prose.
- [x] Refreshed `dataset-queue.json` preserves generated-handoff entry metadata including base URL, page paths, dataset queue URL, expected user, include-completed flag, assignment snapshot, known-user status, and operator-authorization policy.
- [x] Live labeler APIs expose `browser_workflow_scope_checklist` so `/api/me/tasks` and `/api/me/datasets` report supported workflow kinds, server-owned targets, target-token requirements, and out-of-scope navigation rejection.
- [x] Each advertised mutable `browser_workflows[*].write_contract` records the server-owned Zarr target class, training-Zarr write mode, `browser_label_write_target=training_zarr`, CSV/handoff metadata-only role, and split handoff/intermediate CSV no-write fields; `detect_analysis` explicitly reports analysis-Zarr source mutation plus `task_scoped_training_zarr` promotion when configured.
- [x] Live labeler APIs, exported work summaries, and exported handoff dataset queues expose `browser_mutation_write_checklist` so lightweight browser clients can check one ready flag for server-owned task/training Zarr writes and metadata-only CSV/handoff artifacts.
- [x] Labeler-facing task/session authorization failures include redacted `authorization_context` support details with task, recording, assignee, assignment, state, session, and current-session IDs when available.
- [x] Live labeler safety metadata reports failed-promotion retry as `operator_support_only` and points recovery to the admin retry route.
- [x] Labeler-facing failed-promotion retry requests remain guarded by expected-user/current-session checks but return `operator_support_required` without claiming or retrying the promotion; operators use `/api/admin/events/{event_id}/retry-promotion`.
- [x] Operator failed-promotion retry rejects non-`promotion_failed` audit events before claiming a retry, so invalid admin retry requests do not create `promotion_retry_started` events.
- [x] `/datasets` displays copyable safe backup-policy, audit-policy, and session-guard-policy blocks for operator support.
- [x] `/datasets` displays an always-visible copyable queue-state panel showing whether labeler start is allowed or blocked.

### Assignment ownership

- [x] One active assignee is allowed per recording.
- [x] The assignment store schema enforces the recording ownership key with a `recording_id` primary key.
- [x] `single_owner_policy` explicitly declares `assignment_scope=recording`, `recording_assignment_key=recording_id`, `one_current_assignment_row_per_recording=true`, `multiple_labelers_per_recording_allowed=false`, assignment manifests as control-plane metadata, and browser mutations as requiring the current assignment owner.
- [x] Standalone work summaries, dashboard roster rows, aggregate handoff index rows, and handoff roster CSV rows expose flat `single_owner_policy_*` and `assignment_ownership_contract_*` fields so wrappers and spreadsheets can enforce the one-user-per-recording rule and validation-gate ownership contract without parsing nested policy JSON.
- [x] Reassignment reports the previous assignment, ownership transition metadata, `assignment_single_owner_contract`, and store schema-integrity evidence, so callers can assert the one-active-owner-per-recording invariant directly from the assignment operation result.
- [x] Reassignment closes stale sessions belonging to the previous owner.
- [x] Low-level raw assignment changes fail closed when open browser sessions would be left stale.
- [x] Admin assignment POST uses the single-owner transition path.
- [x] CLI assignment and import paths use the same single-owner transition path.
- [x] Assignment exports, imports, batch plans, launch bundles, and admin payloads include ownership policy metadata.
- [x] Assignment ownership integrity reports store-backed schema evidence from `recording_assignments` PRAGMA inspection, including `recording_id` primary-key columns and one-row-per-recording enforcement.
- [x] Assignment and CSV/batch import reports include the same live assignment ownership integrity, assignment ownership contract, and browser mutation write contract used by handoffs, so CSV manifests are explicitly metadata/control-plane inputs while label mutations remain server-owned task/training Zarr writes.
- [x] Assignment and CSV/batch import reports assert assignment manifests are not label write targets, do not write label data, and browser saves still report `browser_label_write_target=training_zarr`, no CSV/handoff writes, and no direct browser Zarr authority.

### Expected-user guard rails

- [x] Root, `/me`, `/labeling`, `/datasets`, and `/work` entry points support expected-user mismatch checks.
- [x] Friendly aliases `/labeling`, `/my-datasets`, and `/my-work` use the same expected-user mismatch checks as their canonical queue/dashboard routes and are listed in `expected_user_guard_contract.guarded_labeler_entrypoints`.
- [x] `/identity?expected_user=<user>` and `/api/me/identity?expected_user=<user>` expose the same single-owner, `personalized_launch_readiness`, and browser-mutation write contract metadata as the personalized queue, and the identity page states the guarded personal queue URL plus training-Zarr/no-CSV/no-direct-Zarr and current-assignee-only rules before labelers open work.
- [x] Identity probe API/HTML payloads expose compact `browser_mutation_target_contract_met` and `direct_browser_start_contract_met` gates, so preflight wrappers can verify training-Zarr/no-CSV mutation and expected-user-guarded Start/Open contracts before opening assigned work.
- [x] The identity page exposes copyable identity support details so labelers can paste resolved user, guarded queue links, ownership policy, and mutation-target contract metadata to the operator before starting work.
- [x] Personal API reads enforce expected-user checks.
- [x] Identity probe payloads expose preferred `/my-datasets?expected_user=<user>` entry URLs, queue link roles, and strict preferred/personal queue match booleans, so the identity-check step can verify the same launch contract before labelers open work.
- [x] Identity probe HTML visibly renders the preferred entry URL, personal/canonical queue roles, and strict personal-queue match diagnostics before the work links, so a labeler can copy the page content to an operator without parsing raw JSON.
- [x] Task-open, task-complete, and labeler promotion retry requests carry expected-user guard state.
- [x] Signed task links are expected-user bound.
- [x] Identity probe links are included in ready-row draft text, roster entries, queue artifacts, handoff artifacts, and inspection artifacts.

Expected-user links are guard rails, not authentication. They catch copied, forwarded, or wrong-browser links after the service has resolved the authenticated user.

### Session and mutation safety

- [x] Browser editing happens through guarded sessions.
- [x] Save routes are session-scoped.
- [x] Browser task completion is session-scoped; direct task-complete compatibility requests must include the current guarded session ID.
- [x] Completed work is read-only to ordinary labeler save routes unless reopened by an operator.
- [x] Completed-task browser save attempts return `task_complete` with `mutation_authorization_contract.server_authorizes_mutation=false`, `task_open_for_mutation=false`, `current_target_token_result=not_checked`, closure-event support, and no browser CSV/direct-Zarr authority before runtime backend or Zarr mutation work can run.
- [x] Completed tasks reject new direct or signed-link browser sessions with `task_complete` until operator reopen.
- [x] Expected-user-bound signed task links remain entry hints, not authorization grants: completed tasks reject with `task_complete`, and the same link only opens after an operator state transition reopens the task to a startable state.
- [x] Operator reopen restores ordinary labeler task-open startability only after the admin state transition records `task_reopened`; focused route assertions confirm the reopened labeler session is server-authorized, expected-user guarded, assignment-current, and still gives the browser no CSV/direct-Zarr authority.
- [x] Existing guarded sessions reject saves/actions when the current task state is no longer `pending` or `in_progress`, returning `task_not_startable` before mutation.
- [x] Task-open/session creation and existing guarded sessions fail closed with `reassignment_session_safety_failed` when stale previous-owner sessions remain open for the recording.
- [x] Signed-link manifests mark task states outside `pending` and `in_progress` as not ready to share with `task_not_startable`.
- [x] Labeler promotion retries require an open task and current guarded session; completed tasks reject with `task_complete`, while operator retry remains available for explicit recovery.
- [x] Save routes are expected to check identity, assignment, task state, session ownership, session expiry, and workflow payload scope.
- [x] Stale session API errors include the latest safe closure event metadata when a session was closed by reassignment, completion, supersession, stale cleanup, or manual close.
- [x] Session-guard recovery entrypoints advertise the personalized `/my-datasets` and `/my-work` aliases ahead of canonical `/datasets` and `/work` fallbacks.
- [x] Superseded sessions report `session_superseded`; stale-cleaned expired sessions report `session_expired`.
- [x] Expired but not-yet-cleaned sessions report `session_expired` before newer-session supersession checks.
- [x] Operators can repair stale previous-owner session blockers with `repair-reassignment-sessions`, which closes only assignment-mismatched sessions and records repair audit events.
- [x] Operators can repair stale previous-owner session blockers from the live admin API with `POST /api/admin/recordings/{recording_id}/repair-reassignment-sessions`.
- [x] Task-open responses report superseded session IDs and safe closure-event summaries when a new browser session invalidates an older tab.
- [x] Successful task-open responses include `task_open_authorization_contract`, proving expected-user guard checking, active assignment, task ownership, startable task state, reassignment-session safety, server-created browser session, server-side authorization, task-scoped training-Zarr target metadata, metadata-only CSV/handoff artifacts, and no browser Zarr write authority.
- [x] Failed task-open responses include `task_open_authorization_contract` with `ready=false`, the exact not-ready reason, no server-created session, server authorization denied, precise guard/check-pass fields, task-scoped training-Zarr target metadata, metadata-only CSV/handoff artifact metadata, and no browser Zarr write authority.
- [x] Runtime browser Start/Open can be configured with `--validation-checklist` plus `--require-operator-validation-for-browser-work` to fail closed before session creation when required operator-validation evidence is missing, pending, invalid, or marked `needs_review`; denied responses include a redacted `operator_validation_start_gate` contract and keep checklist paths out of labeler-visible payloads.
- [x] Live `/api/me/datasets` and `/api/me/tasks` expose the redacted runtime `operator_validation_start_gate`, and both the preferred dataset queue and `/work` dashboard fallback disable visible Start/Open buttons when that gate blocks task open, so pending launch evidence is visible before a click as well as enforced server-side.
- [x] Labeler-facing `/datasets`, `/work`, mutation response/failure metadata, and promotion-retry support payloads intentionally omit `safe_share_next_action_summary`; compact launch-sharing todos remain on operator/wrapper surfaces while labeler pages expose only redacted start/mutation gate diagnostics.
- [x] Browser save/action/review-status mutation routes reuse the runtime validation gate and fail closed before target-token checks, Zarr writes, or audit-event creation when required operator evidence becomes pending after a session is already open; denied responses include a redacted `operator_validation_mutation_gate` contract with `server_authorizes_mutation=false`.
- [x] Browser save/action/review-status mutation routes share one runtime preflight for client target-selector rejection, operator mutation gate enforcement, and current `target_token` checks; denied selector/token responses include `mutation_authorization_contract` with `server_authorizes_mutation=false`, no CSV/Zarr target authority, task-scoped training-Zarr target metadata, and explicit selector/token result fields.
- [x] Pre-handler session/task guard failures on browser mutation POST routes also return `mutation_authorization_contract`, so stale/superseded sessions, inactive assignments, completed/non-startable tasks, and reassignment-session blockers prove `server_authorizes_mutation=false` before target-token checks or backend writes.
- [x] Missing-session and wrong-session-user browser mutation POSTs also return mutation-denial metadata with `server_authorizes_mutation=false`, no browser CSV/Zarr target authority, and explicit session-lookup/session-owner failure fields; focused route assertions cover both early denial cases.
- [x] Successful browser save responses for every supported workflow kind (`keypoints`, `detect_training`, `detect_analysis`, and `subject_mask_component`) expose `mutation.audit_event_id`, `mutation.audit_event_type`, `mutation.audit_event_store`, and the matching redacted audit event summary, while shared route assertions prove assignment/session/target-token authorization passed, the mutable label data plane is the server-owned task-scoped training Zarr, CSV/handoff artifacts are metadata-only non-label-write targets, and the browser has no CSV/direct-Zarr write authority.
- [x] `--require-operator-validation-for-browser-work` is the clearer preferred runtime gate flag for operators; the existing `--require-operator-validation-for-start` flag name is retained for compatibility, and CLI help states that the configured checklist gate protects both Start/Open session creation and browser mutation writes.
- [x] `/datasets` and `/work` task-open failure support blocks preserve `task_open_authorization_contract` fields, authorization context, server-created-session/server-authorized flags, and training-Zarr/no-CSV/no-direct-Zarr assertions so labelers can copy actionable denied-start evidence to operators.
- [x] Signed task-link denial pages include the same `task_open_authorization_contract` evidence as direct Start/Open denials, so copied task links prove `server_authorizes_open=false` on revoked-link, wrong-user, unassigned-user, completed-task, and other preflight failures instead of looking like authorization grants.
- [x] Signed task-link session-creation failures after preflight also include `task_open_authorization_contract` and authorization context, so race-condition assignment/session failures still prove no session was authorized or returned.
- [x] Invalid or tampered signed task-link failures include `signed_link_policy` and `signed_link_contract` support metadata, proving signed links are entry hints rather than authorization grants even when no trusted task payload is available for a task-open contract.
- [x] Missing-token, disabled, invalid, and tampered signed-link failures also include `signed_link_policy`, `signed_link_contract`, `browser_mutation_write_policy`, and `browser_mutation_write_contract` support metadata, so no-task-context signed-link errors still prove signed links are entry hints rather than authorization grants and do not give the browser CSV/direct-Zarr mutation authority.
- [x] `/datasets` and `/work` copied support blocks include the corresponding `direct_start_policy_post_body_expected_user_*` and `direct_start_policy_denied_start_*` policy fields, keeping browser support text aligned with exported handoff policy metadata.
- [x] Expected-user mismatch denials on labeler read routes (`/`, `/me`, `/labeling`, `/work`, `/datasets`, `/my-work`, `/my-datasets`, `/api/me/tasks`, and `/api/me/datasets`) include `labeler_read_authorization_contract` evidence proving the server rejected the read before returning assigned work, did not authorize task open or mutation, and gave the browser no CSV/direct-Zarr write authority.
- [x] Labeler completion routes return closed-session IDs and safe closure-event summaries so stale-tab invalidation is visible from API responses.
- [x] `/api/tasks/{task_id}/complete` and `/api/sessions/{session_id}/complete` responses include `task_completion_authorization_contract` proving expected-user checks, active assignment, current guarded session, server-side completion authorization, labeling-task-store state mutation target, and no browser CSV/Zarr write authority for both allowed and denied completion attempts.
- [x] Early completion denials from missing or wrong-owner sessions also include `task_completion_authorization_contract`, so provided-but-invalid session IDs fail closed with `server_authorizes_completion=false` and no browser CSV/Zarr write authority.
- [x] Stale-session cleanup responses report closed-session IDs and safe closure-event summaries for operator audit/support workflows.
- [x] Browser absolute navigation requests outside the task queue reject with `nav_error` instead of silently clamping to an in-scope target.
- [x] Browser mutation targets are server-owned; save/action/status payloads reject client-supplied target selectors such as `position`, `roi_idx`, `row_idx`, `frame_idx`, component identifiers, explicit Zarr targets (`target_zarr`, `zarr_path`, `training_zarr`, `analysis_zarr`), CSV targets (`csv_path`, `target_csv`, `handoff_csv`, `intermediate_csv`), and generic write-target selectors (`data_plane_write_target`, `label_write_target`, `browser_label_write_target`); focused save-route assertions cover keypoints, detection-training, detection-analysis, and subject-mask workflows with explicit CSV/Zarr selector-smuggling payloads.
- [x] Browser mutation payloads must include the current opaque `target_token` issued for the server-held target, so stale same-session tabs cannot save after another tab navigates the session.
- [x] Mutations stay server-side in Palette-controlled workflow code.
- [x] Successful mutations are expected to append audit or provenance events.
- [x] Successful browser mutation responses include a safe `mutation` block proving server-side assignment/task/session/target-token checks, active assignment presence, task ownership by the resolved user, current-session ownership, reassignment-session safety, server-owned task/training Zarr write scope, `browser_label_write_target=training_zarr`, metadata-only CSV/handoff artifacts, no browser handoff/intermediate CSV writes, no browser Zarr write authority, and the recorded audit event ID.
- [x] Successful browser mutation responses include `mutation_authorization_contract`, proving session lookup, session ownership, server-side task reload, active assignment, task ownership, open task state, current-session status, reassignment-session safety, current target-token match, server-side authorization, and rejection of browser-supplied Zarr/CSV target selectors.
- [x] Successful browser mutation responses include a redacted `authorization_context` with resolved user, session ID, task ID, recording ID, and guarded personalized recovery-link support details, matching the authorization-failure context shape without exposing local Zarr paths.
- [x] Successful browser mutation `mutation` blocks expose flat guarded recovery-link fields (`return_expected_user`, `return_personal_dataset_queue_url`, `return_personal_dataset_queue_expected_user_guarded`, `return_personal_work_url`, and `return_personal_work_expected_user_guarded`) so wrappers do not need to parse nested authorization context.
- [x] Failed browser mutation payloads expose the same nested and flat guarded recovery-link fields as successful mutation payloads, alongside no-CSV/no-direct-Zarr write assertions.
- [x] Labeler promotion-retry denial payloads remain operator-support-only and include the same nested and flat guarded recovery-link fields alongside no-CSV/no-direct-Zarr write assertions.
- [x] Successful browser mutation responses include `browser_mutation_write_checklist` so labeler-provided save/action support details carry the same ready/target/CSV-role contract as the labeler APIs and handoff artifacts.
- [x] Successful browser mutation status messages tell labelers to give operators the returned audit event ID and server-owned Zarr target scope when they need support.
- [x] Successful save/action editor support references include audit ID, task/recording IDs, server target, `browser_label_write_target`, CSV/no-direct-Zarr assertions, and guarded personalized recovery-link fields when available, so pasted save reports do not require opening raw JSON.
- [x] Browser editors expose a copyable mutation support reference containing audit event ID, event type, task ID, recording ID, and server-owned target label.

### Operator visibility

- [x] Admin views expose users, recordings, assignments, sessions, audit events, blocked states, and progress summaries.
- [x] Admin routes require a resolved operator user listed in configured `--admin-user` values.
- [x] Runtime preflight and admin summary expose `operator_authorization_policy` for operator-boundary readiness.
- [x] Runtime preflight and admin summary expose `operator_recovery_policy` and `operator_recovery_contract`, including target validation before session closure, atomic closure/update transition, and pre-update previous-owner session closure, for recovery-route readiness.
- [x] Runtime preflight and admin summary expose `zarr_backup_policy` for backup-readiness checks.
- [x] Runtime preflight and admin summary expose `mutation_audit_policy` for audit-readiness checks.
- [x] Runtime preflight and admin summary expose `browser_response_security_policy` for header-readiness checks.
- [x] Runtime preflight and admin summary expose `session_guard_policy` for stale-tab and session-safety checks.
- [x] Runtime preflight and admin summary expose `dataset_queue_direct_start_policy` for queue direct-start safety checks.
- [x] Store consistency reports include mutation-blocking flags, issue counts, and concrete operator actions for missing assignments or stale assignment/session mismatches.
- [x] Store consistency reports expose `reassignment_session_safety` so wrappers can gate stale previous-owner sessions without parsing generic issue rows.
- [x] Handoff manifests and roster CSV sendability fields surface stale previous-owner sessions as `reassignment_session_safety_failed`, with an operator action to close stale sessions or re-run assignment through `assign_recording_with_session_closure`.
- [x] The live `/admin` preflight card displays operator-boundary, operator-recovery, reassignment target-validation/atomic closure ordering, backup, audit, direct-start, browser-response-security, and session-guard readiness.
- [x] Personalized labeler/admin HTTP responses send no-store/no-cache headers and proxies should preserve them.
- [x] Personalized labeler/admin HTTP responses send clickjacking, MIME-sniffing, and referrer-leakage protection headers.
- [x] Personalized labeler/admin HTTP responses send a narrow CSP and permissions policy compatible with the existing inline UI.
- [x] Operator assignment and completion routes return safe closure-event summaries for closed browser sessions.
- [x] Failed-promotion retry is exposed as an operator recovery route under `/api/admin/events/{event_id}/retry-promotion`; labeler dashboards provide support context instead of a direct retry button.
- [x] The operator retry route preflights event type before claim/audit mutation and only creates `promotion_retry_started` after the target event is confirmed as `promotion_failed`.
- [x] Task repair/unblock is exposed as an operator recovery route under `/api/admin/tasks/{task_id}/repair`; it closes current task sessions, reopens completed tasks to pending by default, and records `task_operator_repaired` without mutating label data.
- [x] Stale or closed session closure events are inspectable through the operator-only `/api/admin/sessions/{session_id}/closure` route.
- [x] Labeler-provided mutation audit event IDs are inspectable through the operator-only `/api/admin/events/{event_id}` route.
- [x] Operator recovery contract and roster exports report disposable bad-mutation recovery readiness before launch.
- [x] The live `/admin` page includes an audit-event lookup form so operators can paste labeler-provided save event IDs and inspect the matching server-side mutation event.
- [x] Operators can resolve labeler-provided audit event IDs from the terminal with `lookup-event --event-id <event-id>`.
- [x] Disposable-Zarr mutation smoke evidence can archive `lookup-event --output` JSON reports and treat reports covering every mutation event ID as operator lookup verification.
- [x] Disposable-Zarr mutation smoke evidence rejects archived lookup reports whose event task, recording, supplied labeler user, or reported workflow context does not match the smoke run.
- [x] Static handoff artifacts mark operator-boundary readiness as runtime-preflight-required when no server config is available.
- [x] Per-user admin views expose guarded dashboard, dataset queue, and identity-probe links.
- [x] Recording admin payloads and HTML views expose the current owner's guarded personalized `/my-datasets` and `/my-work` links alongside canonical dataset queue/dashboard links.
- [x] Admin preflight and summary payloads include ownership integrity, readiness state, personalized `/my-datasets` and `/my-work` entry route metadata, and canonical `/datasets`/`/work` fallback metadata.
- [x] Admin summaries, dashboard roster rows, status reports, CSV/HTML exports, and per-user admin views expose `dataset_queue_state` for operator triage of complete or blocked labeler queues.
- [x] Per-user admin payloads and HTML views expose `reassignment_session_safety`/flat `reassignment_session_safety_*` blockers so operators can identify stale previous-owner sessions from the live admin surface.
- [x] Per-user and per-recording admin HTML views expose scoped reassignment-session repair controls for affected recordings.
- [x] Runtime `/api/admin/summary` exposes `dataset_queue_start_readiness` so operators can see queue-start blockers before exporting handoff artifacts.
- [x] Runtime `/api/admin/summary` exposes top-level reassignment-session safety blocked users, mismatch counts, and affected recording IDs for lightweight operator clients.
- [x] Runtime admin and dashboard-roster status reports expose `reassignment_session_safety_blocked_users`, mismatch counts, and affected recording IDs so operators can distinguish stale previous-owner session blockers from ordinary no-work queue states.
- [x] Runtime `/api/admin/summary`, admin HTML, and preflight reports expose `browser_mutation_write_checklist` so operators can confirm live server-owned task/training Zarr writes, `browser_label_write_target=training_zarr`, metadata-only CSV/handoff artifacts, and split handoff/intermediate CSV no-write fields before exporting handoffs.
- [x] Server preflight reports the runtime operator-validation Start/Open and mutation gates and warns when production is launched without the runtime browser-work validation gate, making it visible that both task opening and already-open browser mutations are not held behind the checklist gate.
- [x] Runtime preflight, admin summary, standalone work summaries, dashboard roster JSON/CSV/status reports, generated handoffs, aggregate indexes, and roster CSV rows expose `runtime_operator_validation_gate_cli_policy`, including the preferred browser-work gate flag, legacy-compatible flag, validation-checklist flag, and the protected Start/Open, mutation, target-token, Zarr-write, and audit-event boundaries.
- [x] Dashboard-roster JSON/status/HTML exports expose the same `dataset_queue_start_readiness` object for pre-share launch review.
- [x] Dashboard-roster status and CSV exports expose operator-validation source, gate counts, pending/needs-review/missing-evidence counts, visibility-boundary fields, and operator action so spreadsheet-driven ready-row draft wrappers can fail closed on launch evidence without parsing the full checklist.
- [x] Dashboard-roster HTML exposes the personalized `/my-datasets` entry URL first, canonical `/datasets` fallback, and link-role summary alongside landing, queue, dashboard, and identity links.
- [x] Dashboard-roster JSON/CSV exports include friendly personalized alias links (`/labeling`, `/my-datasets`, and `/my-work`) alongside canonical queue/dashboard fallback URLs, with top-level personalized page paths/base URLs plus row-level expected-user guarded URLs.
- [x] Dashboard-roster rows expose flat `dataset_queue_start_ready`, `dataset_queue_start_status`, and `dataset_queue_start_operator_action` fields for CSV/spreadsheet launch review.
- [x] Dashboard-roster CSV exports preserve flat `assignment_ownership_contract_*` fields, including recording-key, primary-key-column, one-active-owner, no-multiple-labelers, current-owner-only mutation, and duplicate-owner-count checks for spreadsheet launch gating.
- [x] Dashboard-roster rows and status reports expose flat `dataset_queue_direct_start_*` policy fields for CSV/spreadsheet/JSON review of the direct browser-start contract.
- [x] Direct-start status/CSV fields explicitly assert browsers write labels to training Zarrs and do not write handoff or intermediate CSV files, matching the server-owned task/training Zarr mutation contract.
- [x] Dashboard-roster CSV rows expose flat `reassignment_session_safety_*` fields so spreadsheet wrappers can distinguish stale previous-owner session blockers from ordinary no-open-work states.
- [x] Dashboard-roster CSV rows expose flat labeler-route authorization assertions showing copied links still require known-user, expected-user, active-assignment, and startable task-state checks before task open or mutation.
- [x] Handoff manifests, handoff indexes, and roster CSV rows expose flat task-state/session-guard fields proving only `pending` and `in_progress` tasks can open labeler sessions, while other non-startable states reject with `task_not_startable`.
- [x] Assignment transitions validate the reassignment target, then close previous-owner sessions and write the new owner/status in one committed state transition; handoff/operator-recovery metadata exposes that ordering for launch review.
- [x] Dashboard-roster JSON/status/CSV/HTML exports expose operator-recovery fields, including target validation before session closure, atomic closure/update transition, and pre-update previous-owner session closure, so safe-share wrappers and browser operators can gate reassignment safety without opening handoff bundles.
- [x] Dashboard-roster rows, nested status reports, and bundle roster CSVs expose operator recovery route strings for task-state changes, task repair, reassignment-session repair, audit lookup, and failed-promotion retry.
- [x] Dashboard-roster rows, nested status reports, and bundle roster CSVs expose that labeler promotion retry mutation is disabled and rejected with `operator_support_required`.
- [x] Batch readiness treats assignments with only non-startable task states as not launch-ready and reports `non_startable_task_state`.
- [x] Dashboard-roster CSV rows expose mutation-plane assertions showing browser saves target server-owned assigned task/training Zarr scope, task-scoped training Zarrs are the mutable label data plane, `browser_label_write_target=training_zarr`, and CSV/handoff files remain metadata-only with split no-write flags.
- [x] Dashboard-roster CSV rows explicitly assert that server code mutates task-scoped Zarr targets, training-Zarr promotion requires task scope, and browsers receive or have no direct Zarr write authority.
- [x] Dashboard-roster CSV rows explicitly name `task_scoped_training_zarr` as the label mutation target kind, `training_zarr` as `browser_label_write_target`, and `metadata_only_control_plane` as the CSV/handoff artifact role, with CSV/handoff artifacts marked as non-label-write targets and handoff/intermediate CSV browser writes explicitly false.
- [x] Dashboard-roster CSV rows expose operator-validation-before-invite fields for spreadsheet review of safe-to-send status.
- [x] Dashboard-roster invitation copy is diagnostic/preview-only by default and emits sendable invitation text only with an explicit operator launch-approval assertion.
- [x] Dashboard-roster can consume `validation-checklist.json` and unlock sendable invitation copy only when `all_validation_complete` is true.
- [x] Dashboard-roster fails closed if `all_validation_complete` is stale, has no gates, or is contradicted by pending/needs-review required gates.
- [x] Dashboard-roster fails closed if operators provide both checklist-backed approval and manual launch approval in the same command.
- [x] Dashboard-roster status reports expose the operator-validation source/status used to decide whether invitations are copyable.
- [x] Launch bundle operator command sheets include the checklist-backed `dashboard-roster --operator-validation-checklist` command for approved invitation export.
- [x] Dashboard-roster launch reports emit `dataset_queue_blocks_labeler_start` warnings when any user's queue state should block sending a start link.
- [x] Dashboard-roster JSON/status/CSV/HTML ready-row draft surfaces fail closed with `reassignment_session_safety_failed` only for users whose assigned recordings have stale previous-owner sessions blocking safe queue start or mutation.
- [x] Dashboard-roster ready-row draft text is queue-first: the guarded `/my-datasets` preferred queue link is the `Start here` URL, guarded `/datasets` is listed as the canonical queue fallback, guarded `/` is listed as the queue-first start page, and full `/work` is labeled as the dashboard fallback.
- [x] Handoff inspection reports validation, backup, queue, identity, invitation, and ownership status.
- [x] Handoff inspection reports per-user and aggregate `reassignment_session_safety_*` fields so archived packages can be audited for stale previous-owner session blockers.
- [x] Handoff inspection reports whether launch bundles include `operator-evidence-commands.txt` before operator validation.
- [x] Handoff inspection reports operator evidence-template readiness and flags gates marked passed while their evidence templates remain unapproved.

### Handoff and launch artifacts

- [x] Launch and handoff bundles include guarded root landing links.
- [x] Bundles include guarded dataset queue links and guarded dashboard links.
- [x] Bundles include per-user `dataset-queue.json` artifacts.
- [x] Per-user handoff `work-summary.json` and `dataset-queue.json` artifacts expose top-level `reassignment_session_safety` objects and flat `reassignment_session_safety_*` fields so wrappers can gate stale previous-owner sessions without parsing nested work or manifests.
- [x] Bundle handoff indexes and roster CSVs expose `dataset_queue_state_code`, whether the queue blocks labeler start, and flat queue-start ready/status/action fields for spreadsheet triage.
- [x] Bundle handoff manifests, dataset queues, indexes, and roster CSVs include friendly personalized alias links (`expected_user_personal_dataset_queue_url`, `expected_user_personal_work_url`, `expected_user_labeling_home_url`, and `personalized_labeler_entry_url`) alongside canonical guarded URLs, and aggregate indexes expose top-level `/my-datasets`, `/my-work`, and `/labeling` page paths/base URLs for wrappers.
- [x] Handoff inspection/shareability reports expose a top-level `labeler_entrypoint_summary` with personalized `/my-datasets` page metadata, per-user personalized entry URLs, canonical `/datasets` fallback URLs, and an `all_handoffs_have_personalized_entry_url` gate for operator wrappers.
- [x] Bundle handoff roster CSVs expose flat `reassignment_session_safety_*` fields so spreadsheet wrappers can fail closed only for users whose assigned recordings have stale previous-owner sessions.
- [x] Bundle handoff roster CSVs expose mutation-plane assertions showing browser saves target server-owned assigned task/training Zarr scope, task-scoped training Zarrs are the mutable label data plane, and CSV/handoff files remain metadata-only.
- [x] Multi-user handoff tests assert the same task-scoped training-Zarr, `browser_label_write_target=training_zarr`, split no-write handoff/intermediate CSV flags, split non-label-write-target handoff/intermediate CSV flags, and metadata-only CSV/handoff contract for every generated per-user handoff manifest/dataset-queue pair and roster CSV row.
- [x] Bundle handoff roster CSVs expose labeler-route authorization assertions showing copied dashboard, queue, and signed task links remain server-authorized entry hints gated by known-user, active-assignment, and startable task-state checks.
- [x] Bundle handoff roster CSVs explicitly assert that server code mutates task-scoped Zarr targets, training-Zarr promotion requires task scope, and browsers receive or have no direct Zarr write authority.
- [x] Bundle handoff roster CSVs explicitly name `task_scoped_training_zarr` as the label mutation target kind, `training_zarr` as `browser_label_write_target`, and `metadata_only_control_plane` as the CSV/handoff artifact role, with CSV/handoff artifacts marked as non-label-write targets and handoff/intermediate CSV browser writes explicitly false.
- [x] Batch handoff and launch counts expose aggregate `dataset_queue_states` and `dataset_queue_blocked_start_users` for validation-checklist queue-start gates.
- [x] Bundles include a safe per-user assignment snapshot so copied packages can be checked against current ownership.
- [x] Bundles include roster, README, HTML, progress, empty-state, and summary artifacts.
- [x] Bundles separate ready-row draft text from not-ready diagnostics.
- [x] Bundles include safety policy, operator-authorization policy, task-state policy, signed-link policy, and expected-user guard metadata.
- [x] Bundles include operator-recovery policy metadata for reassignment, reopen, closure inspection, failed-promotion retry, and backup/rollback controls.
- [x] Bundles include safe Zarr backup and rollback policy metadata without exposing backup paths to labelers.
- [x] Bundles include mutation-audit policy metadata for operator validation evidence.
- [x] Bundles include browser response security policy metadata for proxy/header validation evidence.
- [x] Bundles include session-guard policy metadata for stale-tab and session-safety evidence.
- [x] Bundles include backup-plan artifacts for mutable Zarr scopes.
- [x] Launch bundles include `operator-evidence-commands.txt` with exact operator evidence, audit-event lookup, checklist-apply, checksum-refresh, and inspection command templates.
- [x] Launch bundle readmes include preflight and serve command templates that pass `--validation-checklist` and the preferred `--require-operator-validation-for-browser-work` alias, so deployed browser Start/Open and browser mutations can be held behind approved launch evidence rather than only static shareability checks.
- [x] Launch bundles include an operator-only `zarr-backup-evidence-template.json` for backup destination, manifest, verification, restore-test result, and approval evidence.
- [x] Launch bundles include an operator-only `browser-response-security-evidence-template.json` for deployed header capture and approval evidence.
- [x] Launch bundles include an operator-only `identity-source-evidence-template.json` for deployed identity probe capture and approval evidence.
- [x] Launch bundles include an operator-only `browser-smoke-evidence-template.json` for queue-first browser smoke capture and approval evidence.
- [x] Browser smoke evidence templates and `record-browser-smoke-evidence` require `personalized_dataset_queue_verified`, `preferred_labeler_entry_url_matches_personal_dataset_queue`, and `personalized_labeler_entry_url_matches_personal_dataset_queue` checks, with generated smoke rows carrying guarded `/my-datasets?expected_user=<user>`, `/labeling?expected_user=<user>`, and `/my-work?expected_user=<user>` URLs for the representative labeler.
- [x] Launch bundles include an operator-only `disposable-zarr-mutation-smoke-evidence-template.json` for disposable-Zarr mutation smoke capture and approval evidence.
- [x] Handoff HTML, message, and quickstart files name the guarded `/my-datasets` personalized queue as the preferred queue-first entry point, keep canonical `/datasets` as a fallback, and print whether the queue state allows or blocks labeler start.
- [x] Operator launch-bundle and multi-user handoff README/HTML summaries show personalized `/my-datasets`, `/my-work`, and `/labeling` links beside canonical `/datasets`/`/work` fallbacks, and multi-user handoff HTML rows show each labeler's exact guarded `/my-datasets?expected_user=<user>` entry URL plus the human-readable guarded `/labeling?expected_user=<user>` alias.
- [x] Handoff message text uses the guarded `/my-datasets` queue as `Start here`, lists guarded `/datasets` as the canonical queue fallback, lists guarded `/` as the queue-first start page, and labels `/work` as the full-dashboard fallback when applicable.
- [x] Handoff manifests and roster CSVs expose no-local-install assertions for Palette, Crimson, Conda, and project dependencies.
- [x] Handoff HTML, message, quickstart, and dashboard invitation text state that browser saves are server-side to assigned task/training Zarr scope while CSV/HTML/JSON/handoff files remain metadata-only.
- [x] Handoff HTML, message, quickstart, and dashboard invitation text state the one-active-owner rule in labeler language: each recording has one active assigned owner, and only that current assignee can open or save browser labeling work.
- [x] Handoff HTML, message, quickstart, and dashboard invitation text state that labelers should not run operator evidence, repair, checksum, or validation commands; those commands remain operator-only launch controls.
- [x] Handoff sendability treats `dataset_queue_blocks_labeler_start` as a not-ready reason before start links are shared.
- [x] Handoff sendability blocks missing or unsafe labeler-safety metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe labeler-route authorization policy metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe signed-link policy metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe session-guard policy metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe task-state policy metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe Zarr backup policy metadata, and requires backup-plan artifacts when mutable backup-required targets are reported.
- [x] Handoff sendability blocks missing or unsafe mutation-audit policy metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe browser response-security policy metadata before start links are shared.
- [x] Handoff sendability blocks missing or unsafe browser mutation-write policy metadata before start links are shared.
- [x] Handoff sendability repair guidance explicitly names `task_scoped_training_zarr` as the label mutation target kind, `training_zarr` as the browser label write target, `metadata_only_control_plane` CSV/handoff artifacts as non-label-write targets, and handoff/intermediate CSV browser writes as false.

### Validation evidence

- [x] `validation-checklist.json` records queue-first entry metadata.
- [x] `validation-checklist.json` records queue-first entry contract metadata proving guarded landing, preferred guarded `/my-datasets` queue entry, canonical `/datasets` fallback, datasets-waiting aliases, dashboard fallback, and identity-check entry points are present.
- [x] `validation-checklist.json` records preferred-entry link roles proving `/my-datasets` is the preferred queue, `/datasets` is the canonical queue fallback, `/work` is the fallback dashboard, and copied task links are convenience entry hints.
- [x] `validation-checklist.json` records identity-probe link contract metadata proving expected-user identity probe entrypoints are present while deployed identity verification remains operator evidence.
- [x] `validation-checklist.json` points deployed identity verification gates at the operator-only identity-source evidence template when a launch bundle includes one.
- [x] `validation-checklist.json` records single-owner policy metadata.
- [x] `validation-checklist.json` records operator-authorization policy metadata.
- [x] `validation-checklist.json` records operator authorization contract metadata proving admin routes are separate, labelers are not operators by default, and operator authorization does not grant labeler mutation authority.
- [x] `validation-checklist.json` records operator recovery policy metadata.
- [x] `validation-checklist.json` records operator recovery contract metadata proving reassignment, reopen, session-closure inspection, failed-promotion retry, and backup/rollback recovery surfaces are operator-owned.
- [x] `validation-checklist.json` records the operator-only audit event lookup route used to resolve labeler-provided save event IDs during support.
- [x] `validation-checklist.json` records labeler-route authorization contract metadata proving copied queue, dashboard, and signed task links remain server-authorized entry hints and must pass known-user, active-assignment, and startable task-state checks.
- [x] `validation-checklist.json` records signed-link contract metadata proving signed links are expected-user-bound entry hints, not authorization grants.
- [x] `validation-checklist.json` records expected-user guard contract metadata proving copied landing, queue, dashboard, task, promotion-retry support, and signed links stop on expected-user mismatch, with promotion retry guarded as operator-support-only.
- [x] `validation-checklist.json` records browser payload redaction contract metadata proving labeler APIs do not expose raw Zarr paths, direct task scopes, storage credentials, or filesystem write authority.
- [x] `validation-checklist.json` records Zarr backup and rollback policy metadata.
- [x] `validation-checklist.json` records Zarr backup contract metadata proving backup/rollback is operator-owned and backup paths are not labeler-facing.
- [x] `validation-checklist.json` points mutable-Zarr backup confirmation gates at the operator-only backup evidence template when a launch bundle includes one.
- [x] Operators can execute `zarr-backup-plan` output with `execute-zarr-backup-plan`, producing copied Zarr backups and an execution manifest.
- [x] Operators can record `execute-zarr-backup-plan` output into `zarr-backup-evidence-template.json` with `record-zarr-backup-evidence`.
- [x] Operators can restore from a backup execution manifest with `restore-zarr-backup`; restore blocks active recording assignments unless explicitly overridden.
- [x] `validation-checklist.json` records mutation-audit policy metadata.
- [x] `validation-checklist.json` records mutation-audit contract metadata proving successful mutations are server-audited and append-only, while browsers cannot write audit records directly.
- [x] Browser mutation response metadata reports the append-only `labeling_task_events` audit event created by the server-side mutation route.
- [x] Validation logs prompt operators to record `/admin` audit-event lookup results for labeler-reported mutation event IDs before approving mutation smoke evidence.
- [x] Disposable-Zarr mutation smoke evidence requires `operator_event_lookup_verified` before the template can be operator-approved.
- [x] Disposable-Zarr mutation smoke evidence stores archived lookup report paths and resolved event IDs for auditability.
- [x] Disposable-Zarr mutation smoke evidence requires `bad_mutation_recovery_verified` and records the recovery mode/report before operator approval.
- [x] `validation-checklist.json` records browser response security policy metadata.
- [x] `validation-checklist.json` records browser response security contract metadata proving the application intends no-store, anti-framing, no-sniff, no-referrer, CSP, and permissions-policy headers.
- [x] `validation-checklist.json` includes a required `browser_response_security_headers` gate for deployed proxy/header evidence.
- [x] `validation-checklist.json` points deployed response-security header gates at the operator-only browser response security evidence template when a launch bundle includes one.
- [x] `validation-checklist.json` points browser smoke gates at the operator-only browser smoke evidence template when a launch bundle includes one.
- [x] `validation-checklist.json` points disposable-Zarr mutation smoke gates at the operator-only mutation smoke evidence template when a launch bundle includes one.
- [x] `validation-checklist.json` records session-guard policy metadata.
- [x] `validation-checklist.json` records task-state policy and browser mutation target-token contract metadata.
- [x] `validation-checklist.json` records browser workflow scope contract metadata proving only supported workflows are exposed, target selectors are server-owned, `target_selector_fields_rejected` includes CSV/Zarr/write-target selector names, and out-of-scope absolute navigation rejects.
- [x] `validation-checklist.json` records browser mutation write policy metadata proving browser saves target server-owned assigned task Zarr scope, task-scoped training Zarrs are the mutable label data plane, `task_scoped_training_zarr` is the label mutation target kind, `training_zarr` is the browser label write target, and handoff CSV/HTML/JSON artifacts are `metadata_only_control_plane` non-label-write targets with no browser writes to handoff or intermediate CSVs.
- [x] `validation-checklist.json` records session-guard contract metadata proving stale or superseded tabs must reject before mutation.
- [x] `validation-checklist.json` records task-state contract metadata proving completed tasks are read-only to ordinary labelers until operator reopen and failed-promotion retry remains an `operator_support_required` labeler support path.
- [x] `validation-checklist.json` records assignment ownership integrity metadata.
- [x] `validation-checklist.json` records assignment ownership contract metadata proving recording-scoped ownership, `recording_id` as the assignment key, one current assignment row per recording, no multiple labelers per recording, current-owner-only browser mutation, and no duplicate active owners before invitations.
- [x] Handoff sendability fails closed with `assignment_ownership_contract_not_ready` when the normalized ownership contract is not ready, so missing store schema primary-key evidence, reassignment-session-closure failures, and current-owner-only mutation failures cannot pass launch review just because duplicate-owner counts are zero.
- [x] Static readiness depends on ownership integrity where that signal is available.
- [x] Validation logs include paired checklist paths and update command templates.
- [x] Validation logs include audit-event lookup prompts for task, recording, user, workflow, target, and mutation outcome reconciliation.
- [x] Operators can append structured Markdown evidence with `update-validation-checklist --append-log`.
- [x] Operators can apply approved operator evidence templates to `validation-checklist.json` with `apply-operator-evidence-templates`.
- [x] `update-validation-checklist` and `apply-operator-evidence-templates` normalize older checklist files to include the centralized `safe_share_gate`/flat `safe_share_*` contract and echo those fields in JSON reports, so evidence-recording workflows cannot silently drop fail-closed link-sharing metadata.
- [x] Handoff inspection reports evidence counts, evidence files, evidence timestamps, and missing required evidence gates.
- [x] Handoff inspection reports per-workflow missing disposable-Zarr mutation smoke evidence fields, including `operator_event_lookup_verified`, `client_target_selector_rejection_verified`, and `bad_mutation_recovery_verified`.
- [x] Handoff inspection failure actions name incomplete disposable-Zarr smoke workflow fields and point operators to `/admin` audit-event lookup before recording `--operator-event-lookup-verified`.
- [x] Handoff inspection reports per-user and aggregate `dataset_queue_state` fields for copied-package triage.
- [x] Handoff inspection reports `dataset_queue_blocks_labeler_start` as a package-level failure reason before blocked packages are re-shared.
- [x] Handoff inspection compares archived assignment snapshots to an explicit current store and marks reassigned or incomplete packages stale.
- [x] Handoff inspection without explicit `--store` remains static and does not initialize a default assignment database.
- [x] Missing evidence is reported only for required operator gates, not generated passed gates.
- [x] Handoff inspection reports missing, pending, and operator-approved evidence-template status for required operator evidence gates.
- [x] Handoff inspection blocks gates marked passed when their linked operator evidence template is still missing, invalid, or pending approval.
- [x] `validation-checklist.json` and handoff inspection summaries classify gates as generated/static contracts versus operator evidence gates, with explicit pending/needs-review/failed ID lists.
- [x] Approved operator evidence templates can be synced into checklist gate statuses without hand-editing `validation-checklist.json`.
- [x] Operators can refresh directory bundle checksums after evidence/checklist/log updates with `refresh-handoff-checksums`, recording an auditable refresh log.

## Authorization Checklist For Labeler Routes

Every labeler-facing read or write route should verify the following before returning data or accepting mutation.

- [x] Browser identity resolves to a known assignment-store user.
- [x] If `expected_user` is present, it matches the resolved user.
- [x] The recording is actively assigned to the resolved user.
- [x] The task belongs to that assigned recording.
- [x] The task is open for ordinary labeler mutation.
- [x] Existing sessions recheck the current task state before mutation and reject non-startable states.
- [x] No stale previous-owner session remains open for the recording.
- [x] The session is active, unexpired, and owned by the resolved user.
- [x] The session has not been superseded by reassignment, completion, reopen, or operator repair.
- [x] The requested workflow kind is browser-supported.
- [x] The requested target indices, components, labels, or frames are inside the task scope.
- [x] Absolute browser navigation positions outside the task scope reject instead of resolving to a different target.
- [x] Browser mutation payloads cannot select a target directly; the guarded server session supplies the current target.
- [x] Browser mutation payloads include the current opaque `target_token`; missing or stale tokens reject before mutation.
- [x] The mutation path appends audit or provenance data.

## Supported Browser Workflow Scope

Initial browser-supported workflow kinds:

- [x] `keypoints`
- [x] `detect_training`
- [x] `detect_analysis`
- [x] `subject_mask_component`

Do not expand the browser surface until the authorization and mutation contract is explicit for the new workflow kind.

## Handoff Readiness Checklist

A handoff should be considered sendable only when all of these are true.

- [x] Every included user resolves to a known assignment-store user.
- [x] Every included recording has exactly one active owner.
- [x] No included recording has ownership-integrity conflicts.
- [x] Queue artifacts exist for every included user.
- [x] Guarded root, dataset queue, dashboard, and identity-probe links exist for every included user.
- [x] Handoff ready-row draft text distinguishes preview/static browser-entry readiness from safe-share readiness.
- [x] Fresh handoffs remain not-ready-to-send until required operator-validation evidence is approved.
- [x] Not-ready diagnostics include concrete repair actions.
- [x] Public operator-validation metadata distinguishes template-backed external evidence gates from checklist-only gates with `operator_validation_external_evidence_required_*` and `operator_validation_checklist_only_required_*` fields, so wrappers can tell that evidence tooling exists while actual deployment/operator approval is still missing.
- [x] Operator-validation command templates expose a machine-readable `launch_evidence_collection_plan` with one operator-only step per launch-blocking evidence gate, required record/apply command IDs, template paths when applicable, checksum-refresh requirements, and the final `inspect-handoff --require-shareable` / `labeler_links_safe_to_share=true` shareability requirement.
- [x] `inspection-targets.json`, `inspect-handoff` top-level/shareability output, launch roster CSV rows, and labeler copied-support diagnostics expose the launch-evidence collection plan contract/flat fields so wrappers can discover remaining operator-only evidence work without scraping prose.
- [x] Live work-summary/dashboard compact JSON summaries preserve the same `operator_validation_command_template_launch_evidence_collection_*` fields, including empty-plan summaries when no operator-validation commands are required.
- [x] `safe_share_launch_blocking_next_actions` entries carry operator-validation record/apply command IDs plus evidence-template field/path metadata for each blocking gate, so wrapper UIs can route operators directly to the right evidence workflow without exposing runnable commands to labelers.
- [x] `inspection-targets.json` advertises the enriched safe-share next-action row detail/command fields at root and per target, so wrappers can validate blocker-row shape without scraping README prose or sampling an inspection result.
- [x] Launch-bundle `operator-evidence-commands.txt` repeats the enriched safe-share next-action row fields, including record/apply command IDs and evidence-template path metadata, for operator/wrapper discovery from the text command sheet.
- [x] Generated launch bundles include `launch-evidence-execution-checklist.txt`, an operator-only one-by-one runbook for backup, response-security, identity-source, browser-smoke, disposable-Zarr smoke, evidence-apply, and final shareability-inspection gates.
- [x] `inspect-handoff` reports `launch-evidence-execution-checklist.txt` presence/contract status and fails launch-bundle shareability when the operator-only checklist is missing or stale.
- [x] Launch-bundle `inspection-targets.json` advertises the generated launch-evidence checklist artifact name, top-level inspection fields, required phrase contract, and blocking reason IDs so wrappers can discover the checklist gate without sampling an inspection result.
- [x] Compact `shareability_contract` payloads include launch-evidence checklist summary/presence/validity fields and source-field provenance so one-object wrapper gating can detect stale or missing checklist artifacts.
- [x] Inspection repair-command metadata includes `regenerate_package_with_launch_evidence_execution_checklist` with missing phrase counts and artifact contract fields for launch bundles missing or carrying stale checklist artifacts.
- [x] `regenerate_package_with_launch_evidence_execution_checklist` repair rows and contracts include the required checklist file, required phrase-contract field, and required phrase list so wrappers can render stale-checklist repairs without opening `inspection-targets.json`.
- [x] Generated implementation-status, README, HTML index, and operator command-sheet text name `regenerate_package_with_launch_evidence_execution_checklist` so human operators see the same stale-checklist repair path exposed in JSON.
- [x] Generated launch-evidence checklist text is self-describing: it names its artifact file, inspection field, summary field, required phrase contract, blocking reasons, repair command, and repair metadata fields.
- [x] Generated validation-log templates list `launch-evidence-execution-checklist.txt` as a required review file and name `regenerate_package_with_launch_evidence_execution_checklist` for stale-checklist inspection failures.
- [x] `inspect-handoff` top-level and nested `shareability` safe-share projections include `safe_share_launch_blocking_next_action_detail_fields` and `safe_share_launch_blocking_next_action_command_fields`, so wrappers can validate blocker-row shape from a single inspection response.
- [x] `validation-checklist.json` payloads expose the same safe-share next-action detail/command field lists, so checklist-only wrappers can validate enriched blocker rows before applying operator evidence templates.
- [x] Operator-facing dashboard roster HTML renders the safe-share next-action summary plus enriched blocker action detail/command field lists, so human review sees the same blocker-row contract as JSON/CSV wrappers.
- [x] Launch roster CSV exports preserve `safe_share_launch_blocking_next_action_detail_fields` and `safe_share_launch_blocking_next_action_command_fields`, keeping enriched blocker-row shape discoverable for spreadsheet-based wrappers.
- [x] Compact `shareability_contract` payloads include `safe_share_launch_blocking_next_action_detail_fields` and `safe_share_launch_blocking_next_action_command_fields` with source-field provenance, so one-object wrapper gating can discover the enriched blocker-row shape.
- [x] `inspection-targets.json` advertises compact-contract field paths for the enriched next-action detail/command field lists at root and per target, tying sidecar discovery to `shareability_contract` consumers.
- [x] Compact `shareability_contract`, nested `shareability`, live `/work` and `/datasets` copied support diagnostics, launch roster CSVs, dashboard operator HTML, operator command sheets, and `inspection-targets.json` expose/discover consolidated `safe_share_external_launch_evidence_gap_*` fields, so wrappers can show the exact remaining external launch-evidence gates, statuses, templates, and record-command IDs from one safe-share contract.
- [x] Safe-share external launch-evidence diagnostics also expose `safe_share_external_launch_evidence_gap_todos`, `safe_share_external_launch_evidence_gap_todo_count`, and `safe_share_external_launch_evidence_gap_todo_fields`, giving wrappers one operator-only row per remaining evidence gate with status, action text, evidence-template field/path, record command IDs, apply command ID, and apply-after-approval flag.
- [ ] Mutable Zarr backup-plan evidence exists and is operator-approved.
- [x] Mutable Zarr backup evidence template exists in launch bundles for operator approval capture.
- [x] Operator CLI can copy backup-plan targets and produce backup execution manifests for approval evidence.
- [x] Operator CLI can update backup evidence templates from execution manifests without hand-editing JSON.
- [x] Operator CLI can restore backup targets after pausing or unassigning affected recordings.
- [ ] Browser/proxy response-security-header evidence exists and is operator-approved.
- [x] Browser/proxy response-security-header evidence template exists in launch bundles for operator approval capture.
- [x] Browser response-security policy protects the actual labeler entry and probe surfaces (`/`, `/me`, `/labeling`, `/my-datasets`, `/my-work`, `/identity`, `/api/me/identity`) in addition to canonical `/datasets`/`/work` and personal queue APIs, and route-header assertions cover the preferred personalized aliases.
- [x] Browser/proxy response-security-header evidence templates prefer authenticated `/my-datasets?expected_user=<user>` captures, list `/labeling`, `/my-work`, canonical `/datasets`/`/work`, and personal API fallbacks, and require personalized alias headers to match canonical route headers.
- [x] Browser/proxy response-security-header evidence templates include a structured `required_capture_contract` naming the preferred `/my-datasets` entrypoint, required expected-user query, authenticated-test-user requirement, identity agreement, declared-path matching, and personalized/canonical header parity.
- [x] Fresh browser/proxy response-security-header evidence templates declare route-context checks for expected-user query presence, authenticated test-user presence, authenticated-user/expected-user agreement, preferred-path matching, declared sample-path matching, and overall capture-URL/user contract readiness before any operator capture is recorded.
- [x] Browser/proxy response-security-header evidence recording refuses operator approval for new templates when the capture URL omits the `expected_user` guarded route context, omits the authenticated test user, the authenticated test user does not match the captured expected user, or the route falls outside the declared labeler capture paths.
- [x] Browser/proxy response-security-header evidence recording compares capture URLs by normalized route path, so path-only templates and full deployed URLs validate against the same `/my-datasets`/`/labeling`/`/my-work` route contract.
- [x] Browser/proxy response-security-header evidence readiness inspects recorded `capture.url` and `capture.authenticated_test_user`, not only boolean checks, before applying approved evidence to the validation gate.
- [x] Browser/proxy response-security-header evidence recording and checklist readiness use the same expected-user query parser so the approval command and apply-template gate cannot disagree about the captured user.
- [x] Browser/proxy response-security-header generated command guidance tells operators to pass the deployed `/my-datasets?...expected_user=...` URL and the same authenticated test user named by that query.
- [x] Browser/proxy response-security-header evidence update reports echo the structured `required_capture_contract` so CLI wrappers can display the enforced personalized-route capture rules without reopening the template.
- [x] Browser/proxy response-security-header evidence readiness summaries echo the structured `required_capture_contract` so inspection/apply tooling can display the enforced personalized-route capture rules without reopening the raw evidence file.
- [x] Operator CLI can update browser response-security evidence templates from captured deployed headers without hand-editing JSON.
- [ ] Required validation gates have evidence.
- [ ] The deployment identity source has been checked in the target environment.
- [x] Deployment identity-source evidence template exists in launch bundles for operator approval capture and carries the guarded `/labeling?expected_user=<user>` human-readable queue alias as non-preferred metadata.
- [x] Operator CLI can update identity-source evidence templates from deployed identity-probe results without hand-editing JSON.
- [x] Identity-source evidence readiness requires exact `resolved_user == expected_user`, a named operator, and an approval timestamp before guarded links are treated as deployment-verified.
- [ ] A browser smoke test has been run with at least one representative labeler identity.
- [x] Browser smoke evidence template exists in launch bundles for operator approval capture.
- [x] Browser smoke evidence templates include a structured personalized-route smoke contract naming `/my-datasets` as the preferred queue entry, `/labeling` as the human-readable queue alias, `/my-work` as the personalized dashboard fallback, canonical fallbacks, required expected-user query context, and the two personalized-route verification flags.
- [x] Browser smoke personalized-route contracts name the per-user personalized URL fields, canonical fallback URL fields, and identity-probe URL field so wrappers can render the exact smoke route set without inferring field names.
- [x] Browser smoke personalized-route contracts mark link roles explicitly: `/my-datasets` as preferred queue, `/labeling` as human-readable queue alias, `/my-work` as fallback dashboard, canonical `/datasets`/`/work` as fallbacks, identity probe as identity check, and wrong-user aliases as expected-user mismatch checks.
- [x] Browser smoke evidence templates include wrong-user personalized `/my-datasets`, `/labeling`, and `/my-work` URLs so operators can verify expected-user mismatch behavior on the actual labeler entrypoints.
- [x] Browser smoke personalized-route contracts name the wrong-user personalized URL fields and `expected_user_mismatch_rejected` check so wrappers can distinguish required mismatch evidence from ordinary support links.
- [x] Browser smoke evidence readiness requires the structured personalized-route smoke contract to match the generated contract, and reports missing/mismatched contract fields when stale templates omit `/my-work` or other personalized-route metadata.
- [x] Browser smoke evidence recording fails closed and update reports expose expected/actual personalized-route smoke contract metadata, contract readiness, and missing/mismatched contract fields when stale templates omit `/my-work` or other personalized-route metadata.
- [x] Browser smoke evidence readiness summaries expose both actual and expected personalized-route smoke contracts so inspection/apply tooling can explain stale `/my-datasets`/`/labeling`/`/my-work` route metadata without reopening the raw evidence template.
- [x] Browser smoke evidence update reports and readiness summaries use the same actual/expected personalized-route smoke contract fields so wrapper diagnostics are symmetric before and after apply-template inspection.
- [x] Browser smoke stale-route-contract recorder errors and readiness summaries include an operator action to regenerate `browser-smoke-evidence-template.json` from the current launch bundle before approval/apply.
- [x] Handoff inspection failure actions surface stale browser-smoke personalized-route contracts with the regenerate-template action and the required `/my-datasets`, `/labeling`, and `/my-work` smoke routes.
- [x] Handoff inspection structured repair commands map stale browser-smoke personalized-route-contract failures to `record-browser-smoke-evidence`, `apply-operator-evidence-templates`, and checksum-refresh follow-up for the `browser_smoke` gate, with a `browser_smoke_personalized_route_contract_stale` reason ID.
- [x] Handoff inspection response-security repair commands use the guarded deployed `/my-datasets?...expected_user=...` capture placeholder and matching authenticated-test-user placeholder.
- [x] Browser smoke evidence update reports and readiness summaries echo the structured personalized-route smoke contract so wrappers can display required `/my-datasets`, `/labeling`, and `/my-work` checks without reopening raw evidence files.
- [x] Browser smoke readiness user summaries expose per-user personalized `/my-datasets`, `/labeling`, `/my-work`, and wrong-user personalized alias URLs so inspection/apply tooling can display the exact routes operators were expected to smoke.
- [x] Browser smoke validation-gate operator evidence text names personalized `/my-datasets` queue entry, human-readable `/labeling` alias, and personalized `/my-work` dashboard fallback.
- [x] Operator CLI can update browser smoke evidence templates from a representative smoke run without hand-editing JSON.
- [x] Browser smoke evidence readiness requires exact `resolved_user == expected_user`, browser-only/no-local-install checks, personalized `/my-datasets`, `/labeling`, and `/my-work` route checks, queue visibility checks, stale-tab/completion checks, a named operator, and an approval timestamp.
- [ ] A disposable-Zarr mutation smoke has been run before broad launch.
- [x] Disposable-Zarr mutation smoke evidence template exists in launch bundles for operator approval capture.
- [x] Operator CLI can update disposable-Zarr mutation smoke evidence templates from workflow smoke runs without hand-editing JSON, including explicit proof that browser-supplied CSV/Zarr/write-target selectors were rejected.
- [x] Disposable-Zarr mutation smoke evidence rows include per-workflow Zarr target class, training-Zarr write mode, `browser_label_write_target=training_zarr`, CSV/handoff metadata-only role, split handoff/intermediate CSV no-write fields, and no-direct-browser-Zarr-authority contract fields; inspection refuses approval when these fields are missing or unsafe.
- [x] Applying approved operator evidence templates refreshes sibling handoff manifests, aggregate handoff indexes/rosters when present, and labeler-facing HTML/message/quickstart files with the approved operator-validation state, then reports that package checksums must be refreshed before sharing.
- [x] Inspection reports evidence-template readiness so operators can see which templates remain missing, pending, or approved before launch.
- [x] Identity-source evidence inspection exposes aggregate top-level personal-queue proof fields (`identity_personal_queue_evidence_status`, `identity_personal_queue_evidence_ready_users`, `identity_personal_queue_evidence_missing_users`, and `identity_all_users_have_personal_queue_evidence`) so wrappers can block launch without parsing per-user missing-field arrays.
- [x] Full handoff/package inspection propagates identity personal-queue proof status/count/user fields to the top-level report and the consolidated `shareability.identity_personal_queue_evidence` object.
- [x] Launch-bundle `inspection-targets.json` advertises `identity_personal_queue_evidence_status`, its allowed values (`missing`, `incomplete`, `ready`), and the identity personal-queue proof count/user fields so archive wrappers can require the same gate used by live and CSV payloads.
- [x] Launch-bundle `inspection-targets.json`, top-level `inspect-handoff` output, and nested `shareability` inspection output advertise the flat operator-validation gate IDs, field suffixes, and allowed status values (`unknown`, `pending`, `missing_evidence`, `needs_review`, `passed`) so archive wrappers can validate per-gate launch-evidence fields consistently.
- [x] Inspection repair guidance identifies incomplete mutable-Zarr backup evidence targets and prints the `record-zarr-backup-evidence` command shape.
- [x] Inspection repair guidance identifies missing browser response-security headers/checks and prints the `record-browser-response-security-evidence --header ...` command shape.
- [x] Inspection repair guidance identifies unresolved, mismatched, or personal-queue-incomplete identity-source evidence rows, explains that preferred and personalized entry URLs must equal the guarded `/my-datasets?expected_user=<user>` queue while `/labeling?expected_user=<user>` remains a human-readable alias, and prints the `record-identity-source-evidence` command shape.
- [x] Inspection repair guidance prints the exact browser-smoke flags needed to confirm browser-only/no-local-install runtime, personalized `/my-datasets` queue entry, human-readable `/labeling` alias route, personalized `/my-work` dashboard fallback, assigned-only visibility, redaction, stale-tab rejection, and operator reopen behavior.
- [x] Generated operator evidence command sheets use the deployed `/my-datasets?...expected_user=...` response-security capture URL placeholder and matching authenticated-test-user placeholder instead of a generic `--capture-url URL`.
- [x] Generated operator evidence command sheets and `inspection-targets.json` enumerate the protected response-security route set for wrapper checks: `/me`, guarded `/labeling`, guarded `/identity` and `/api/me/identity`, guarded `/my-work`, canonical `/datasets`/`/work` fallbacks, and personal queue APIs in addition to the preferred guarded `/my-datasets` capture.
- [x] Inspection repair guidance prints the exact disposable-Zarr smoke flags needed to confirm task-scoped training-Zarr writes, no direct browser Zarr authority, metadata-only handoff/CSV artifacts, no browser CSV/handoff writes, and rejected browser-supplied CSV/Zarr target selectors.
- [x] Inspection repair guidance prints the `repair-reassignment-sessions --user OPERATOR --recording-id RECORDING_ID` command when stale previous-owner sessions block handoff shareability.
- [x] Inspection JSON exposes structured `operator_repair_commands` entries plus command counts, categories, linked validation gate IDs, reason IDs, and checksum-refresh requirements so operator tooling can surface the next evidence/apply-template/checksum commands without parsing prose.
- [x] Operator repair-command generation merges duplicate command rows from template-derived and prose-derived paths, preserving all gate IDs and reason IDs for wrapper tooling.
- [x] Identity-source repair commands carry `personal_dataset_queue_link_evidence_incomplete` when aggregate identity evidence status shows any user missing guarded personal `/my-datasets` proof.
- [x] Operator repair commands include `apply-operator-evidence-templates` for approved operator evidence templates before checksum refresh, so handoff manifests, indexes, quickstarts, messages, HTML, and roster CSVs refresh sendability from the updated validation checklist.
- [x] Operator-validation command-template metadata exposes machine-readable gate-to-evidence-template maps (`template_backed_gate_ids`, `apply_required_gate_ids`, `evidence_template_fields_by_gate_id`, and `evidence_template_paths_by_gate_id`) in live support blocks, roster CSVs, validation checklists, manifests, and inspection reports so wrapper tooling can route each pending launch gate to the correct evidence template without parsing prose.
- [x] Inspection JSON exposes explicit `labeler_links_safe_to_share`, `shareability_status`, `shareability_operator_action`, the centralized `safe_share_gate`/flat `safe_share_*` contract, structured `safe_share_launch_blocking_next_actions`, compact `safe_share_next_action_summary`, and a versioned consolidated `shareability` contract with decision source, blocking reason IDs, blocking gate IDs, and repair command IDs/categories/reason IDs so launch tooling can block link sharing without inferring from low-level gate arrays.
- [x] Inspection payloads record whether shareability was required/met, and `inspect-handoff --require-shareable` returns nonzero when labeler links are not safe to share.
- [x] Launch-bundle operator overviews and readmes explicitly warn operators not to share links solely because per-user handoffs say `ready_to_send`; `inspect-handoff --require-shareable` must report `labeler_links_safe_to_share=true`, and unapproved backup, response-security, identity, browser-smoke, disposable-Zarr, or operator-recovery evidence gates remain launch blockers.
- [x] Launch indexes, `labeler-roster.csv` rows, `validation-checklist.json`, and `inspection-targets.json` expose a machine-readable `safe_share_gate` plus flat `safe_share_*` fields from one centralized safe-share policy helper, declaring `ready_to_send` insufficient, `labeler_links_safe_to_share=true` required, and the backup, response-security, identity, browser-smoke, disposable-Zarr, and operator-recovery evidence gates as launch blockers.
- [x] Validation checklists, validation-checklist inspection summaries, full `inspect-handoff` payloads, nested `shareability` contracts, `update-validation-checklist` reports, and `apply-operator-evidence-templates` reports expose filtered safe-share blocker status fields (`safe_share_launch_blocking_*` and `safe_share_checklist_gate_evidence_complete`) so wrappers can identify pending, needs-review, missing, unknown, satisfied, and unsatisfied launch-blocking evidence gates without re-parsing generic gate arrays.
- [x] Live admin/preflight payloads plus dashboard-roster top-level JSON, command summaries, CSV/user rows, and nested status reports derive the same filtered safe-share blocker status from operator-validation fields, including explicit `missing_evidence` blockers when no validation checklist evidence has been approved.
- [x] Safe-share projections and live labeler support-copy diagnostics also expose consolidated `safe_share_external_launch_evidence_gap_*` fields, including gap gate IDs, statuses, counts, action-required state, summaries, evidence-template paths, and record-command IDs, so wrappers can show the remaining operator/deployment evidence work without joining multiple blocker arrays.
- [x] Generated inspection command templates include `--require-shareable` so copied launch-package checks default to the safe link-sharing contract.
- [x] Generated `inspection-targets.json` is schema-versioned and records `shareability_required`, `shareability_gate`, shareability contract schema, decision source, and full identity personal-queue evidence fields, including counts and missing-field maps, for each package target so operator tooling knows inspection commands enforce safe link sharing and guarded `/my-datasets` proof.
- [x] Launch bundles include a single operator evidence command sheet so operators do not have to reconstruct command paths manually.
- [x] Operator evidence command sheets start with an operator-only boundary warning that commands require operator authorization, are not labeler instructions, and must not be sent to labelers.
- [x] Generated launch-bundle README files include a concise implementation/evidence status summary, naming the implemented browser-labeling contracts, the remaining external evidence gates, and `docs/web_labeling_implementation_status.md` as the repository status reference.
- [x] Generated launch-bundle README files also name the `inspect-handoff` implementation-status fields (`implementation_status_artifact`, `shareability.implementation_status_artifact`, and flat `implementation_status_*`) so the README and command sheet expose the same wrapper-facing status contract.
- [x] Generated launch bundles include `implementation-status.txt`, a bundle-local copy of the concise implementation/evidence status summary; launch HTML indexes link it from the Review-first section, and checksums/ZIP coverage assert the file is included with copied packages.
- [x] Generated `implementation-status.txt` explicitly says it is advisory status metadata, not launch approval, names the machine-readable `inspect-handoff` implementation-status fields, lists the required `implementation_status_artifact` fields/count plus separate payload-specific and inspect-handoff flat `implementation_status_*` fields/counts, and states it does not replace `inspect-handoff --require-shareable` or `labeler_links_safe_to_share=true`.
- [x] Launch-bundle manifests expose a nested `implementation_status_artifact` plus flat implementation-status advisory metadata (`implementation_status_artifact_schema`, status file/role, not-launch-approval flag, operator-evidence-required flag, safe-share gate, required safe-share field/value, and require-shareable-inspection flag), so manifest-only wrappers cannot mistake the status summary for launch approval.
- [x] Launch-bundle manifests, validation checklists, dry-run reports, and inspection outputs expose `implementation_status_artifact_required_fields` plus `implementation_status_artifact_required_field_count` beside the nested artifact, so wrappers can validate artifact completeness without opening `inspection-targets.json`.
- [x] Launch-bundle manifests, validation checklists, and dry-run reports expose payload-specific `implementation_status_flat_fields` plus `implementation_status_flat_field_count` beside the nested artifact, including the status-file path and advisory fields, while inspection outputs expose the broader inspect-handoff flat field list with path/declaration/presence fields; wrappers can validate the relevant flat companion fields without opening `inspection-targets.json`.
- [x] Launch-bundle dry-run reports expose the same nested `implementation_status_artifact` and flat implementation-status advisory metadata at top level, and include the planned `implementation-status.txt` path/count in the embedded dry-run validation checklist, so wrappers can reject unsafe planned launches before writing package files.
- [x] Generated launch-bundle HTML indexes name the `inspect-handoff` implementation-status fields (`implementation_status_artifact`, `shareability.implementation_status_artifact`, and flat `implementation_status_*`) so browser-visible operator package reviews expose the same wrapper-facing status contract as README and command sheet.
- [x] Generated operator evidence command sheets point operators to `implementation-status.txt` before the smoke/evidence command sequence, so copied bundles expose the implementation/evidence split in every operator launch entry point.
- [x] Generated operator evidence command sheets also name the `inspect-handoff` implementation-status fields (`implementation_status_artifact`, `shareability.implementation_status_artifact`, and flat `implementation_status_*`) so wrapper authors can find the machine-readable status contract without inspecting code.
- [x] Launch-bundle `inspection-targets.json` advertises `implementation-status.txt` as the bundle-local implementation/evidence status summary and marks it checksum-covered, so wrappers can discover the status artifact without scraping README or HTML.
- [x] Launch-bundle `inspection-targets.json` advertises the versioned `implementation_status_artifact` schema, an exemplar artifact contract object, the artifact required-field list/count, top-level/shareability artifact field names, payload-side status-path/artifact/schema field names, payload-side artifact-required-field companion names, payload-specific flat-field companion names/list/count, and inspect-handoff flat-field companion names/list/count, including non-approval fields plus exact safe-share required field/value dependencies, so wrappers can discover the status-artifact contract without sampling package output.
- [x] Implementation-status artifacts, inspection-target metadata, README, HTML, and operator command sheets explicitly state the status summary is not launch approval; wrappers must still require `inspect-handoff --require-shareable` and `labeler_links_safe_to_share=true` before sharing links.
- [x] Launch-bundle `validation-checklist.json` exposes `implementation-status.txt` as a first-class artifact path with an `implementation_status_present` count, so wrappers can discover the implementation/evidence split from the checklist without scraping README, HTML, or inspection metadata.
- [x] Launch-bundle `validation-checklist.json` and embedded dry-run validation checklists expose the versioned `implementation_status_artifact` plus flat advisory/non-approval fields, including exact safe-share gate and required field/value, so checklist-only wrappers cannot mistake `implementation-status.txt` for launch approval.
- [x] Handoff/package inspection validation-checklist summaries expose the checklist-declared implementation-status path/count, expected package location, and packaged `implementation-status.txt` match, so archive wrappers can audit the status artifact without opening package contents directly.
- [x] Handoff/package inspection top-level payloads and nested `shareability` summaries expose a versioned `implementation_status_artifact` plus flat `implementation_status_*` convenience fields, so wrappers can discover the implementation/evidence split without knowing the nested validation-checklist schema.
- [x] Handoff inspection validates operator evidence command sheets contain the operator-only boundary warning and treats stale command sheets without that warning as invalid.
- [x] Launch-bundle inspection reports a distinct command-sheet boundary failure/action when `operator-evidence-commands.txt` exists but lacks the operator-only warning.
- [x] Handoff inspection and shareability summaries expose wrapper-safe operator command-sheet summary fields, including boundary-present, missing-phrase, validity, blocking-reason IDs, and repair-command reason IDs.
- [x] Live admin summary, preflight payloads, dashboard-roster JSON/HTML, and nested dashboard status reports expose versioned operator-validation command templates, including evidence-recording commands, operator-recovery contract gate recording, apply-template refresh, gate IDs, checksum-refresh flags, and missing-command diagnostics.
- [x] Dashboard-roster per-user JSON rows and CSV exports expose flattened operator-validation command-template fields, preserving operator-only/not-labeler-instruction boundary flags for spreadsheet wrappers.
- [x] Static handoff manifests, work summaries, dataset queues, aggregate indexes, per-handoff index rows, and roster CSVs expose the same operator-validation command-template contract so offline launch bundles preserve operator next-step guidance.
- [x] Standalone `work-summary` JSON output and stdout summary expose operator-validation command-template metadata, including operator-only/not-labeler-instruction boundary flags.
- [x] Applying approved operator evidence templates refreshes operator-validation command templates in manifests, work summaries, dataset queues, aggregate index rows, and roster CSVs so stale evidence-recording commands disappear once gates pass.
- [x] `validation-checklist.json` and handoff inspection summaries expose operator-validation command templates derived from pending, needs-review, missing, or unapproved operator-evidence gates, with unsupported manual gates listed as missing-command diagnostics.
- [x] Handoff inspection repair-command output is seeded from locally regenerated operator-validation command templates using inspected gate IDs, so operator tooling gets structured next-step commands without trusting package-provided command text.
- [x] Handoff inspection top-level payloads and versioned shareability summaries expose compact operator-validation command-template summaries, command IDs, gate IDs, and missing-command gate diagnostics for wrapper-safe launch gating.
- [x] Operator-validation command-template payloads and flattened CSV/inspection summaries explicitly mark commands as operator-only, not labeler instructions, requiring operator authorization, and not something labelers should run.
- [x] Operator-validation visibility policy distinguishes strict operator-only fields from operator-action support fields, so command templates can be carried for operator support without reclassifying them as labeler instructions.
- [x] `/datasets` and `/work` copyable support blocks expose only non-runnable operator-validation command-template diagnostics (schema, counts, IDs, missing-command gates, and boundary flags), with tests asserting runnable operator command strings are absent from labeler pages.
- [x] Static labeler-facing handoff HTML/message/quickstart artifacts, including refreshed-ready artifacts after evidence application, are tested to exclude runnable operator command strings.

## Operator Mutable-Zarr Backup Command Sequence

Use `docs/web_labeling_implementation_status.md` for the condensed launch status, remaining evidence blockers, and command summary. The sections below remain the detailed evidence procedures.

Before broad launch, operators should use command-owned evidence rather than hand-editing backup JSON.

1. Generate `zarr-backup-plan.json` from the assignment store.
2. Run `execute-zarr-backup-plan --plan zarr-backup-plan.json --backup-dir <operator-backup-dir> --operator <operator>`.
3. Record the execution manifest into `zarr-backup-evidence-template.json` with `record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json --execution-manifest <manifest> --target-index <index> --restore-test-result <result> --operator <operator>`.
4. Run `apply-operator-evidence-templates --path validation-checklist.json --operator <operator>` after the backup evidence template is approved.
5. Run `refresh-handoff-checksums --path <bundle-dir> --operator <operator>` after updating evidence/checklist/log files in the directory bundle.
6. If rollback is needed, pause or unassign affected recordings, then run `restore-zarr-backup --manifest <manifest> --target-index <index> --operator <operator> --replace-current`.

## Operator Identity Evidence Command Sequence

Before sending links to a labeler, operators should record the deployed identity-probe result and guarded personal dataset queue proof for that expected user.

1. Open `/identity?expected_user=<user>` in the deployed browser/auth context.
2. Copy the resolved user reported by Palette and confirm the probe reports the preferred and personalized entry URLs as the guarded `/my-datasets?expected_user=<user>` queue.
3. Run `record-identity-source-evidence --evidence identity-source-evidence-template.json --expected-user <user> --resolved-user <resolved-user> --operator <operator> --authenticated-session-context DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED`.
4. The command derives the strict personal-queue match booleans from the generated identity evidence URLs; operators should not hand-edit CSV or evidence files into label-write state.
5. If the resolved user does not exactly match the expected user, or the evidence row cannot prove the guarded personal dataset queue match, the command records the mismatch but leaves the evidence unapproved.
6. Run `apply-operator-evidence-templates --path validation-checklist.json --operator <operator>` after identity evidence is approved.
7. Run `refresh-handoff-checksums --path <bundle-dir> --operator <operator>` after updating evidence/checklist/log files in the directory bundle.

## Operator Browser Response-Security Evidence Command Sequence

Before broad launch through a proxy, operators should record deployed response headers from an authenticated labeler-facing request.

1. Capture headers from `/my-datasets?expected_user=<user>` first as an authenticated labeler; spot-check `/labeling?expected_user=<user>` and `/my-work?expected_user=<user>` when the proxy has route-specific header rules.
2. Use `/datasets?expected_user=<user>`, `/work?expected_user=<user>`, `/api/me/tasks?expected_user=<user>`, or `/api/me/datasets?expected_user=<user>` only as canonical/API fallbacks; personalized alias headers must match canonical route headers.
3. Run `record-browser-response-security-evidence --evidence browser-response-security-evidence-template.json --header "Cache-Control=<value>" --header "X-Frame-Options=<value>" --operator <operator> --capture-url <deployed-my-datasets-url-with-expected-user> --authenticated-test-user <same-user-as-expected-user>`.
4. Include every expected header from the evidence template as a repeated `--header NAME=VALUE` argument.
5. If any header is missing or weakened, the command records the capture but leaves the evidence unapproved.
6. Run `apply-operator-evidence-templates --path validation-checklist.json --operator <operator>` after response-security evidence is approved.
7. Run `refresh-handoff-checksums --path <bundle-dir> --operator <operator>` after updating evidence/checklist/log files in the directory bundle.

## Operator Browser Smoke Evidence Command Sequence

Before broad launch, operators should run one representative queue-first browser smoke with a real assigned labeler identity.

For the full local execution checklist covering syntax checks, focused tests,
generated artifact inspection, runtime API smoke, browser mutation smoke,
multi-user safety smoke, and final launch criteria, use
`docs/web_labeling_execution_checklist.md`.

1. Open `/identity?expected_user=<user>`, `/my-datasets?expected_user=<user>`, `/labeling?expected_user=<user>`, `/my-work?expected_user=<user>`, and a guarded task link in the deployed browser/auth context.
2. Confirm identity match, browser-only runtime, no local Palette/Crimson install requirement, no local Conda/project dependency requirement, personalized queue visibility, personalized work-dashboard fallback, assigned-only visibility, redacted support text, expected-user mismatch rejection, task open, completion/read-only behavior, stale-tab rejection, and operator reopen behavior.
3. Run `record-browser-smoke-evidence --evidence browser-smoke-evidence-template.json --expected-user <user> --resolved-user <resolved-user> --operator <operator>` with the explicit boolean flags for every confirmed check, including `--browser-only-runtime-verified`, `--no-local-palette-install-verified`, `--no-local-crimson-install-verified`, `--no-local-conda-or-project-dependencies-verified`, `--personalized-dataset-queue-verified`, `--preferred-labeler-entry-url-matches-personal-dataset-queue`, `--personalized-labeler-entry-url-matches-personal-dataset-queue`, and `--personalized-work-dashboard-verified`; also open the template's `/labeling?expected_user=<user>` alias as a human-readable queue-home spot check.
4. If any required check flag is omitted or identity mismatches, the command records the partial run but leaves the evidence unapproved.
5. Run `apply-operator-evidence-templates --path validation-checklist.json --operator <operator>` after browser-smoke evidence is approved.
6. Run `refresh-handoff-checksums --path <bundle-dir> --operator <operator>` after updating evidence/checklist/log files in the directory bundle.

## Operator Disposable-Zarr Mutation Smoke Evidence Command Sequence

Before broad launch, operators should run one representative disposable or restorable Zarr mutation smoke for each launched workflow kind.

1. Select a disposable Zarr or known-good regeneration source for the workflow kind.
2. Complete one browser save through the guarded queue/task flow and capture the mutation event ID.
3. Paste the event ID into the live `/admin` audit-event lookup, call `GET /api/admin/events/<event-id>` as an operator, or run `lookup-event --event-id <event-id> --output <event-id>-lookup.json`, and confirm task, recording, user, workflow, target, and mutation outcome.
4. Confirm backup/regeneration, server write scope, task-scoped training-Zarr write target, no direct browser Zarr write authority, metadata-only handoff/CSV artifacts, audit event, completion, stale-tab rejection, bad-mutation recovery mode/report, and restore/discard behavior.
5. Run `record-disposable-zarr-mutation-smoke-evidence --evidence disposable-zarr-mutation-smoke-evidence-template.json --workflow-kind <kind> --mutation-event-id <event-id> --event-lookup-report <event-id>-lookup.json --operator <operator> --labeler-user <user> --task-scoped-training-zarr-write-verified --browser-no-direct-zarr-write-authority-verified --handoff-artifacts-metadata-only-verified --browser-no-csv-or-handoff-write-verified --client-target-selector-rejection-verified --operator-event-lookup-verified --bad-mutation-recovery-verified --bad-mutation-recovery-mode <restore_backup|regenerate_known_good|discard_disposable> --bad-mutation-recovery-report <path-or-note>` with the explicit boolean flags for every confirmed check.
6. If event IDs or any required check flag are missing, the command records the partial run but leaves the evidence unapproved.
7. Run `apply-operator-evidence-templates --path validation-checklist.json --operator <operator>` after disposable-Zarr smoke evidence is approved.
8. Run `refresh-handoff-checksums --path <bundle-dir> --operator <operator>` after updating evidence/checklist/log files in the directory bundle.

## Validation Gates

These gates should be tracked in `validation-checklist.json` and summarized by handoff inspection.

| Gate | Evidence |
| --- | --- |
| Assignment integrity | Export or inspection result showing one owner per recording. |
| Identity resolution | Browser identity probe result from the deployment environment. |
| Operator authorization boundary | Runtime preflight or `/api/admin/preflight` evidence that admin users are configured and non-admin labelers receive `admin_required`. |
| Operator recovery static contract | Generated checklist evidence that reassignment, reopen, closure inspection, failed-promotion retry, and backup/rollback recovery surfaces are operator-owned. |
| Browser response security headers | Captured deployed `/my-datasets?expected_user=<user>` response headers, with `/labeling` and `/my-work` route-specific spot checks when needed and canonical/API fallbacks, showing the proxy preserved no-store, anti-framing, MIME-sniffing, referrer, CSP, and permissions headers. |
| Dataset queue start readiness | Generated checklist showing no invite-ready labeler has `dataset_queue_blocks_labeler_start=true`; if blocked, operator evidence records the repair or decision to stop labeling. |
| Queue visibility | Screenshot or log showing the labeler sees only assigned work. |
| Expected-user mismatch | Evidence that a wrong expected user is rejected or warned. |
| Session safety | Evidence that stale, completed, or reassigned sessions cannot save. |
| Disposable mutation | Evidence that a representative save mutates only intended disposable data. |
| Completion behavior | Evidence that completed tasks become read-only to ordinary labelers. |
| Backup posture | Backup path, restore plan, or known-good source for mutable Zarrs. |
| Audit trail | Evidence that representative mutations append audit or provenance events. |
| Operator recovery | Evidence that reopen, repair, retry, or reassignment paths are available. |

## Queue-First Browser Smoke Procedure

Run this before sending real links from a new deployment environment.

1. Pick one representative assignee with at least one open assigned task.
2. Open `/identity?expected_user=<user>` in the same browser session the labeler will use.
3. Confirm the resolved user matches the expected user.
4. Open `/?expected_user=<user>`.
5. Confirm the page loads the queue-first landing, not an admin-only surface.
6. Open `/my-datasets?expected_user=<user>`.
7. Confirm only that user's assigned datasets, recordings, and startable tasks are shown as labeler-ready.
8. Open `/my-work?expected_user=<user>` and confirm the personalized work-dashboard fallback resolves to the same authenticated user and assigned work.
9. Use the copy button for one dataset, one recording, and one task.
10. Confirm copied support details include safe IDs, counts, workflow state, guarded work URLs, and the direct browser-start endpoint.
11. Confirm copied support details do not include raw Zarr paths, direct task scopes, storage credentials, or filesystem write targets.
12. Start one task directly from the `/my-datasets` queue.
13. Confirm the browser editor session opens without local Palette or Crimson installation.
14. Open one guarded task-filtered `/my-work` fallback link from the dataset queue.
15. Confirm `/my-work` shows only matching assigned work and provides copyable filter support details if no matching task is available.
16. Repeat `/my-datasets?expected_user=<wrong-user>` from the same browser session.
17. Confirm the mismatch is rejected or clearly warned before task access.
18. Append the observed result to the validation checklist with `update-validation-checklist --append-log`.

## Disposable-Zarr Mutation Smoke Procedure

Run this against disposable or restorable data before broad launch.

1. Create or select a disposable Zarr target that represents the workflow kind being launched.
2. Confirm the target is covered by a backup, restore, known-good regeneration, or disposable discard plan.
3. Assign the recording to exactly one smoke-test user.
4. Open the queue through `/my-datasets?expected_user=<user>`.
5. Start one task directly from the guarded `/my-datasets` queue.
6. Make one minimal representative edit.
7. Confirm the server writes only the intended workflow scope.
8. Confirm an audit or provenance event records the mutation.
9. Complete the task or close the session.
10. Reopen the old browser tab and attempt another save.
11. Confirm the stale save is rejected and includes safe closure context where available.
12. Restore, regenerate, or discard the disposable data and archive the recovery report/path used for smoke evidence.
13. Append the observed result to the validation checklist with `update-validation-checklist --append-log`.

## Operator Recovery Checklist

Operators need these recovery paths before broader use.

- [x] Reassign a recording to a new user and confirm old sessions are closed.
- [x] Reopen a completed task for the same assigned user.
- [x] Execute a mutable-Zarr backup plan into operator-controlled backup storage.
- [x] Restore a mutable-Zarr backup only through operator CLI after affected recordings are paused or unassigned.
- [x] Retry a failed promotion or mutation operation from an admin route.
- [x] Repair or unblock a task with clear operator-only authority.
- [x] Inspect the latest closure event for a stale or blocked labeler session.
- [x] Recover from backup, regenerate from known-good source, or discard disposable data after a bad disposable mutation.
- [x] Confirm labeler support text gives enough redacted IDs for the operator to locate the problem.

## Implementation Order For Remaining Work

Use this order when finishing hardening work. Earlier items are more fundamental than later convenience work.

1. Verify all labeler read routes enforce resolved-user and expected-user checks.
2. Verify all labeler save routes enforce assignment, task, workflow-scope, active-session, and session-expiry checks.
3. Verify raw Zarr paths and direct task scopes are absent from browser API payloads and copied support text.
4. Verify reassignment closes previous-owner sessions and stale tabs cannot save.
5. Verify completion makes ordinary labeler mutation read-only until operator reopen.
6. Verify mutation audit/provenance events are appended for each supported workflow kind.
7. Verify backup and rollback evidence exists before real mutable Zarr handoff.
8. Verify handoff inspection blocks or clearly marks not-ready handoffs when required evidence is missing.
9. Run a representative queue-first browser smoke in the deployment environment.
10. Run a disposable-Zarr mutation smoke before broad launch.

## Notes For Future Extensions

- Prefer adding new workflow kinds behind explicit authorization and mutation contracts.
- Keep queue pages simple and labeler-oriented; advanced recovery belongs in operator views.
- Keep generated handoff artifacts redacted by default.
- Treat expected-user links as guard rails only; authentication and authorization remain server-side.
- Treat signed links as convenience links only; they should not bypass identity, assignment, task, or session checks.
- Preserve the one-active-owner-per-recording rule unless a future design explicitly introduces conflict-free multi-user merge semantics.
