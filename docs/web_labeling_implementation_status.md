# Web labeling implementation status

This is the short operational status for the multi-user browser labeling handoff work. The detailed checklist remains `docs/web_labeling_implementation_checklist_clean.md`.

## Current implementation checkpoint

- Step 2 focused web-route validation is passing: `scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_web_routes.py -q` returned `38 passed in 34.71s`.
- Focused launch-bundle/checklist inspection validation is passing: `scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_assignment_store.py::test_inspect_handoff_launch_evidence_execution_checklist_reports_directory_and_zip tests/unit/fisheye/test_labeling_assignment_store.py::test_export_launch_bundle_cli_writes_plan_readiness_handoffs_and_zip -q` returned `2 passed, 1 warning in 0.68s`.
- That focused assignment-store validation required two narrow implementation fixes: generated browser-smoke evidence now defines the guarded `/labeling?expected_user=<user>` base URL, and top-level launch-bundle manifests now include `browser_mutation_write_policy` plus `browser_mutation_write_checklist`.
- The implemented browser-labeling route contract now treats pending operator-validation evidence as a launch blocker: users with assigned work receive diagnostic-note readiness rows until the required external evidence is approved.
- The remaining unchecked checklist items are operator/deployment evidence gates, not missing browser-route implementation: mutable-Zarr backup approval, deployed browser/proxy response-security capture, deployment identity-source check, representative browser smoke, and disposable-Zarr mutation smoke.
- A concise repo operator runbook for those remaining gates is `docs/web_labeling_launch_evidence_execution_checklist.md`.
- Newly generated launch bundles now also include `launch-evidence-execution-checklist.txt` beside `operator-evidence-commands.txt`, `implementation-status.txt`, and `inspect-command.txt`.
- `inspect-handoff` reports the generated checklist artifact and blocks launch-bundle shareability if it is missing or stale.
- `inspection-targets.json` advertises the checklist artifact, required fields, required phrases, and blocking reason IDs for wrapper discovery.
- Compact `shareability_contract` payloads now include checklist summary, presence, validity, contract-present, and blocking-reason fields for one-object wrapper gating.
- Missing or stale checklist artifacts now produce structured `regenerate_package_with_launch_evidence_execution_checklist` repair-command metadata, including missing phrase counts when available.
- That repair command also carries the required checklist file, required phrase-contract field, and required phrase list for wrapper display.
- Generated implementation-status, README, HTML index, and operator command-sheet text also name `regenerate_package_with_launch_evidence_execution_checklist` for human operator repair guidance.
- Generated launch-evidence checklist text now names its own inspection fields, blocking reasons, repair command, and repair metadata fields.
- Generated validation-log templates now list the launch-evidence checklist artifact and stale-checklist repair command.
- Do not share labeler links broadly until the generated evidence templates have been recorded, approved, applied, checksums refreshed if needed, and `inspect-handoff --require-shareable` reports `labeler_links_safe_to_share=true`.

## Implemented code contracts

- Personalized labeler entrypoints exist for queue-first browser work:
  `/my-datasets?expected_user=<user>`, `/labeling?expected_user=<user>`,
  `/my-work?expected_user=<user>`, plus canonical `/datasets` and `/work`
  fallbacks.
- Live personalized `/api/me/tasks` and `/api/me/datasets` payloads, generated
  manifests, aggregate handoff indexes, inspection-target metadata, and
  `labeler-roster.csv` rows expose a versioned
  `personalized_launch_readiness` object summarizing the guarded personal queue
  link, labeler start/work-completion state, training-Zarr/no-CSV mutation
  contract, safe-share gate, and remaining external launch-evidence gaps for
  lightweight wrappers.
  Roster rows flatten the personalized readiness gap todos, template paths, and
  record command IDs so spreadsheet clients can show the same remaining
  operator evidence work without parsing nested manifests.
- Labelers do not need local Palette, Crimson, Conda, or project dependency
  installs for browser labeling.
- Browser saves are server-side mutations against assigned task/training Zarr
  scope. CSV, HTML, JSON, handoff, roster, and intermediate files are metadata
  or control-plane artifacts, not label-write targets.
- Browser mutation payloads reject client-selected target fields, including
  row/frame/component selectors, explicit Zarr targets, CSV targets, and generic
  write-target selectors.
- One active owner per recording is enforced through the assignment store
  schema and reassignment transition path. Reassignment closes stale browser
  sessions before ownership changes and reports single-owner transition
  evidence.
- The assignment store exposes a read-only
  `single_owner_assignment_contract()` helper for wrappers and web routes to
  assert the recording-scoped one-owner/current-assignee-only mutation contract
  without performing an assignment mutation. The contract also states that
  browser saves resolve the target server-side from active assignments, mutate
  assigned training Zarrs, and never mutate intermediate CSVs.
- Store-backed control-plane reports now include the nested
  `single_owner_assignment_contract`; the normalized
  `assignment_ownership_contract` records that this store proof is present,
  ready, met, and aligned with the server-resolved training-Zarr/no-CSV mutation
  boundary.
- Dashboard/launch roster CSVs and `inspection-targets.json` now preserve and
  advertise the single-owner store-proof fields plus required values, so
  spreadsheet or wrapper flows can fail closed if the store-backed proof is
  absent even when static single-owner policy text is present.
- Dashboard/launch roster CSVs and `inspection-targets.json` also advertise the
  self-described `personalized_launch_readiness` contract and flatten the
  training-Zarr-only browser write target / no CSV-handoff write assertions.
  The readiness summary derives those assertions from the existing nested
  browser-mutation/direct-start policy and checklist contracts plus their flat
  CSV/status projections, so generated artifacts do not need a separate writable
  CSV or handoff data path.
- The personalized readiness summary also derives external launch-evidence gap
  IDs, statuses, counts, template paths, record-command IDs, and todo field
  metadata from todo-only artifacts, including CSV-style JSON strings.
- The readiness summary normalizes CSV/string boolean projections such as
  `"False"` and `"0"` before computing queue-match, safe-share, no-CSV, and
  no-direct-Zarr fields.
- Live personal APIs, refreshed handoff work summaries, and standalone
  `work-summary` exports recompute personalized readiness after safe-share and
  operator-validation fields are present, keeping readiness gap counts aligned
  with the source safe-share contract.
- The selected-key live `/api/me/identity`, `/api/me/tasks`, and
  `/api/me/datasets` responses also emit top-level
  `personalized_launch_readiness`, so thin clients do not need to parse nested
  identity or dashboard work objects to find the guarded launch contract.
- The visible `/identity?expected_user=<user>` page now summarizes the same
  readiness schema, guarded personal queue URL, training-Zarr target, no
  CSV/handoff browser writes, and no direct browser Zarr authority before a
  labeler enters their queue.
- Fail-closed unknown-labeler responses from identity, task, and dataset APIs
  preserve the same readiness object shape and write-target assertions, keeping
  wrapper error handling aligned with successful personal-queue responses.
- Fail-closed expected-user mismatch responses from task and dataset APIs carry
  the same guarded expected-user readiness/write-target contract while still
  withholding assigned work and dataset queues.
- Fail-closed signed task-link expected-user mismatches now carry that same
  guarded expected-user readiness/write-target contract in the access-problem
  support payload while still refusing session creation.
- Direct task-open API success and denial payloads now carry top-level
  `personalized_launch_readiness` plus the guarded personal queue URL and
  direct-start write policy, so expected-user mismatch wrappers get the same
  training-Zarr/no-CSV/no-direct-Zarr contract as queue-first APIs while the
  server still refuses session creation.
- Task-completion denial payloads now carry the same top-level readiness,
  guarded personal queue URL, and direct-start write-policy assertions, so
  expected-user/session failures prove the browser still has no CSV/handoff
  write path or direct Zarr authority while the server refuses completion.
  This includes both task-scoped and session-scoped completion endpoints.
- Labeler failed-promotion retry denials now carry the same readiness, guarded
  personal queue URL, and no-CSV/no-direct-Zarr assertions while preserving the
  `operator_support_required` boundary; labelers get support diagnostics, not a
  browser mutation path.
- `/work` and `/datasets` copied support details now include the consolidated
  readiness JSON plus compact `personalized_launch_readiness_*` lines for the
  personal queue URL, pending evidence gaps, and training-Zarr/no-CSV/no-direct
  Zarr assertions.
- Those copied support details also include explicit `operator_support_*` lines
  that prefer row-level support values for guarded personal queue URLs,
  personalized entry URLs, training-Zarr targets, metadata-only CSV/handoff
  role, no CSV/handoff browser writes, and no direct browser Zarr authority.
- Full `/work` task support plus `/datasets` dataset, recording, and task
  `operator_support` rows now repeat the guarded personal dataset-queue URL,
  queue-link roles, and training-Zarr/no-CSV/no-direct-Zarr safety facts so
  row-level copied diagnostics stand alone.
- `docs/web_labeling_execution_checklist.md` provides the operator-run checklist
  for syntax checks, focused route tests, generated artifact inspection,
  browser/API smoke, multi-user safety smoke, and launch gates.
- Safe-share external launch-evidence diagnostics now include per-gate
  `safe_share_external_launch_evidence_gap_todos` rows with gate status,
  operator-only action text, evidence-template field/path, record command IDs,
  apply command ID, and apply-after-approval flag, so wrappers do not need to
  join separate maps before showing the remaining operator evidence work.
  Projection fallbacks also preserve todo-only stale artifacts and reconstruct
  the gate status/count fields from those rows.
- Live labeler identity and personal work/dataset APIs now expose the same
  nested `single_owner_assignment_contract` plus flattened
  `assignment_ownership_contract_store_single_owner_assignment_contract_*`
  fields, so browser wrappers can require store-backed assignment ownership
  evidence before opening or mutating assigned training-Zarr work.
- Those live API contracts include the `assignment_ownership_integrity` object
  and duplicate-active-owner count used to derive readiness, so the proof checks
  current store state in addition to the structural assignment-store schema.
- Live route authorization checklists now include store-proof readiness fields
  (`single_owner_store_contract_required`, `single_owner_store_proof_ready`, and
  duplicate-owner count) and require them when store-backed assignment evidence
  is supplied.
- Labeler-route authorization policy and generated validation-checklist
  contracts now declare that browser work requires single-owner store proof,
  assignment-integrity OK, zero duplicate active owners, training-Zarr mutation
  targets, and no intermediate CSV mutation.
- Handoff roster CSVs now flatten those route-authorization store-proof
  requirements, so spreadsheet workflows can confirm the policy requires
  integrity, zero duplicate owners, training-Zarr targets, and no intermediate
  CSV mutation without parsing nested JSON.
- `inspection-targets.json` advertises both the single-owner store-contract
  fields and the labeler-route authorization store-proof policy fields with
  required values for wrapper discovery.
- Route-authorization failure actions now name the single-owner store proof,
  zero duplicate active owners, server-resolved training-Zarr targets, and no
  intermediate CSV mutation requirements directly.
- Handoff sendability now requires runtime route-authorization checklist
  evidence for `single_owner_store_proof_ready=true` and
  `assignment_ownership_integrity_ok=true`, plus server-resolved
  training-Zarr/no-intermediate-CSV proof, when route policy declares browser
  work depends on single-owner store proof; static route policy text alone is
  not sufficient.
- `inspection-targets.json` advertises the observed runtime route-checklist
  fields and required values (`checklist_present=true`, `checklist_ready=true`,
  `single_owner_store_proof_ready=true`, `assignment_ownership_integrity_ok=true`,
  `duplicate_active_owner_count=0`,
  `browser_mutation_target_resolved_server_side=true`,
  `labelers_mutate_assigned_training_zarrs=true`, and
  `labelers_mutate_intermediate_csvs=false`) for wrapper discovery.
- `inspection-targets.json` also advertises
  `shareability_labeler_route_authorization_runtime_checklist_gate_contract`,
  including compact/nested gate paths, required value, mismatch fields,
  fail-closed reason, repair command ID, required fields, and required values.
- Dashboard roster, preflight, and queue-readiness summaries aggregate the same
  runtime route-checklist gate with
  `labeler_route_authorization_runtime_checklist_gate_all_users_met`,
  not-met users, and total mismatch count fields, so live share/handoff
  decisions fail closed before labelers receive links.
- CLI `work-summary` JSON exports and stdout summaries now use the same
  store-backed single-owner assignment contract as the live APIs, keeping
  offline wrapper inputs aligned with browser payloads.
- `/work` and `/datasets` copyable support details include the same
  store-backed single-owner/training-Zarr/no-intermediate-CSV field names, so
  pasted labeler diagnostics preserve the proof needed for operator triage.
- Task open, browser mutation, task completion, and signed-link denial paths
  expose contract metadata proving server-side authorization checks, expected
  user guards, active assignment checks, task-state checks, no browser CSV write
  authority, and no direct browser Zarr write authority.
- Signed task links are short-lived entry hints, not authorization grants.
  Verified signed-link denials expose task-open authorization contracts; invalid
  or no-task-context signed-link failures expose signed-link and browser
  mutation write contracts.
- Launch bundles, handoff manifests, roster CSVs, inspection targets, and
  operator command sheets expose wrapper-readable safe-share gates for preferred
  `/my-datasets` links, direct Start/Open safety, browser mutation targets,
  single-owner policy, and operator evidence status.
- Operator-validation command templates now include a nested
  `launch_evidence_collection_plan` plus flat plan fields, with one
  operator-only step per launch-blocking evidence gate and the final
  `inspect-handoff --require-shareable` / `labeler_links_safe_to_share=true`
  requirement, so wrappers can guide evidence collection without treating
  external evidence as already approved.
- `inspection-targets.json`, `inspect-handoff` top-level/shareability output,
  launch roster CSVs, and copied support diagnostics expose the same
  launch-evidence collection plan contract/flat fields, keeping wrapper and
  operator UIs aligned on the remaining external evidence work.
- Live work-summary/dashboard compact summaries also project the
  `operator_validation_command_template_launch_evidence_collection_*` fields,
  so lightweight wrappers do not lose the evidence plan when consuming summary
  JSON instead of full manifests.
- Per-gate `safe_share_launch_blocking_next_actions` now include
  operator-validation record/apply command IDs and evidence-template field/path
  metadata, so operator UIs can jump from a blocker row to the correct
  evidence-capture workflow without parsing the full command sheet.
- `inspection-targets.json` advertises both the safe-share next-action top-level
  fields and the enriched per-action detail/command field lists at root and
  per target, so wrappers can validate blocker rows without sampling an
  inspection payload first.
- The operator evidence command sheet repeats the enriched
  `safe_share_launch_blocking_next_actions` row fields, including record/apply
  command IDs and evidence-template path metadata, for human and wrapper
  authors working from the bundle text artifacts.
- `inspect-handoff` safe-share projections now include
  `safe_share_launch_blocking_next_action_detail_fields` and
  `safe_share_launch_blocking_next_action_command_fields` alongside the action
  rows, so a wrapper can validate blocker-row shape from one inspection response
  without reopening `inspection-targets.json`.
- Safe-share projections now include consolidated
  `safe_share_external_launch_evidence_gap_*` fields for the remaining
  operator/deployment evidence gaps, including gate IDs, statuses, counts,
  action-required state, summaries, evidence-template paths, and record-command
  IDs.
- Compact `shareability_contract`, nested `shareability`, live `/work` and
  `/datasets` copied support diagnostics, launch roster CSVs, dashboard operator
  HTML, operator command sheets, and `inspection-targets.json` now expose or
  discover the same external-evidence gap fields, so wrappers can drive
  evidence collection from one safe-share contract instead of joining generic
  blocker arrays.
- `validation-checklist.json` payloads also expose the same safe-share
  next-action detail/command field lists, so checklist-only wrappers can
  validate enriched blocker rows before applying operator evidence templates.
- Operator-facing dashboard roster HTML renders the safe-share next-action
  summary plus enriched blocker action detail/command field lists, keeping the
  human review surface aligned with JSON/CSV wrapper contracts.
- Launch roster CSV exports preserve the same safe-share next-action
  detail/command field-list columns, so spreadsheet wrappers do not drop the
  enriched blocker-row contract when flattening handoff readiness.
- The compact `shareability_contract` also includes the enriched safe-share
  next-action detail/command field lists and source provenance, so one-object
  wrapper gating can discover blocker-row shape without reading sidecar targets.
- `inspection-targets.json` now advertises compact-contract field paths for
  those enriched next-action field lists at root and per target, bridging
  sidecar discovery to `shareability_contract` one-object consumers.
- Browser-mutation target, direct browser Start/Open, single-owner, and runtime
  route-checklist repair commands include structured contract diagnostics for
  wrapper UI repair guidance.
- Top-level `operator_repair_commands` includes adjacent detail-field and
  contract metadata for wrappers that consume the top-level repair rows.
- Nested `shareability.repair_commands` includes the same enriched repair rows
  as top-level `operator_repair_commands`.
- Nested `shareability` also includes repair-command detail fields and contract
  maps so a single `inspect-handoff` response can drive wrapper repair UI.
- `inspect-handoff` exposes a compact self-described `shareability_contract`
  object, mirrored at `shareability.contract`, with its own field list/count and
  source-field provenance/count for one-object wrapper gating over the current
  safe-share decision, blockers, repair IDs/count, safe-share gate,
  implementation-status gate, core package contracts, the runtime route-checklist
  gate, and repair-command contracts.
- `inspection-targets.json` advertises the compact contract field names, schema,
  expected compact-contract field list/count, and source-field provenance
  map/count at root and per-target levels.
- `inspection-targets.json` advertises the `shareability.repair_commands` field
  name and per-command detail fields for wrapper discovery.
- `inspection-targets.json` also advertises `shareability_repair_command_contracts`
  for required training-Zarr targets, metadata-only CSV/handoff artifacts,
  one-active-owner policy, runtime route-checklist proof, and
  implementation-status regeneration expectations.

## Remaining launch blockers

These are not code-only items. They require evidence from the target deployment
or a disposable/restorable Zarr smoke environment before links should be shared.

- Mutable Zarr backup-plan evidence exists and is operator-approved.
- Browser/proxy response-security-header evidence exists and is operator-approved.
- Required validation gates have approved evidence.
- The deployment identity source has been checked in the target environment.
- A representative browser smoke test has been run with at least one assigned
  labeler identity.
- A disposable-Zarr mutation smoke has been run before broad launch.

## Launch rule

Do not share labeler links because a row says `ready_to_send` or
`ready_to_invite`. Share only after inspection reports:

```text
labeler_links_safe_to_share=true
```

The preferred labeler link is the guarded personalized queue:

```text
/my-datasets?expected_user=<user>
```

## Generated launch-bundle status

Launch bundles include a bundle-local `implementation-status.txt` summary with
the same operational split: implemented browser-labeling contracts versus
operator evidence still required before sharing links.
The generated status file explicitly says it is advisory metadata, not launch
approval, names the machine-readable `inspect-handoff` implementation-status
fields, lists the required `implementation_status_artifact` fields/count plus
separate payload-specific and inspect-handoff flat `implementation_status_*`
fields/counts, and states it does not replace `inspect-handoff --require-shareable` or
`labeler_links_safe_to_share=true`.
The generated launch README, HTML index, manifest, checksums, ZIP archive, and
operator evidence command sheet all reference or include this file.
Launch-bundle manifests also expose the advisory status metadata directly:
a nested `implementation_status_artifact`, artifact schema, status file/role, not-launch-approval flag,
operator-evidence-required flag, safe-share gate, exact required safe-share
field/value, and require-shareable-inspection flag.
Manifests, validation checklists, dry-run reports, and inspection outputs include
`implementation_status_artifact_required_fields` and
`implementation_status_artifact_required_field_count` beside the nested artifact,
so wrappers can validate artifact completeness without opening
`inspection-targets.json`.
Manifests, validation checklists, and dry-run reports include payload-specific
`implementation_status_flat_fields` and `implementation_status_flat_field_count`
beside the nested artifact, including the status-file path and advisory fields.
Inspection outputs include the broader inspect-handoff flat field list with
path/declaration/presence fields. Wrappers can validate the relevant flat
companion fields without opening `inspection-targets.json`.
Launch-bundle dry-run reports expose the same nested `implementation_status_artifact`
and flat advisory fields at top level, and include the planned
`implementation-status.txt` path/count in the embedded dry-run validation
checklist.
Launch-bundle `validation-checklist.json` files and embedded dry-run validation
checklists also expose `implementation_status_artifact` plus flat advisory
fields, including the exact safe-share gate and required field/value.
The launch README, HTML index, and operator evidence command sheet also name the `inspect-handoff`
implementation-status fields: `implementation_status_artifact`,
`shareability.implementation_status_artifact`, and flat
`implementation_status_*` fields.
Those generated operator-facing surfaces also state that stale packages missing
the complete `implementation_status_artifact` contract fail closed with
`implementation_status_artifact_incomplete` and emit
`regenerate_package_with_implementation_status_artifact`.
They also state the exact safe-share prerequisite:
`implementation_status_checklist_artifact_complete=true` in
`shareability.safe_to_share_requires`.
They also point wrappers to
`implementation_status_checklist_artifact_gate_contract` in
`inspection-targets.json`.
They also point wrappers to `shareability_repair_command_contracts` for repair
UI over browser-mutation target, direct Start/Open, single-owner, runtime
route-checklist, and implementation-status regeneration failures.
They also point wrappers to compact `shareability_contract` / `shareability.contract`
for one-object safe-share gating.
When wrappers use the compact contract, the generated operator surfaces
explicitly require `shareability_contract.safe_to_share=true` before sharing
labeler links.
The compact contract also exposes `safe_to_share_required_value=true` and
`safe_to_share_matches_required_value` for direct machine checks.
`inspection-targets.json` advertises the compact-contract safe-share observed
field, required value, and match field at root and per-target levels.
They also tell wrappers to use `fields`, `field_count`, `source_fields`, and
`source_field_count` to detect malformed or truncated compact contract payloads.
These fields are advisory status metadata, not launch approval. Wrappers must
still require `inspect-handoff --require-shareable` and
`labeler_links_safe_to_share=true` before sharing links.
`inspection-targets.json` also advertises the file as a checksum-covered
bundle-local implementation/evidence status summary for wrapper tooling, along
with the versioned `implementation_status_artifact` schema, an exemplar artifact
contract object, its required-field list/count, payload-side artifact/schema
field names, payload-side status-path field name, payload-specific flat-field
companion names/list/count, payload-side artifact-required-field companion field
names, inspect-handoff flat-field companion names/list/count, and the stale
package fail-closed reason plus repair command ID, including the
`implementation_status_checklist_artifact_complete=true` requirement, the
required-value mismatch blocking reason, the non-approval fields, and exact
safe-share required field/value dependencies.
`validation-checklist.json` exposes the same status artifact through
`implementation_status` and `counts.implementation_status_present`, and
`inspect-handoff` validation-checklist summaries report both the checklist
declaration, expected package location, and packaged `implementation-status.txt`
match. Top-level `inspect-handoff` payloads and nested `shareability` summaries
also expose `implementation_status_artifact` plus flat `implementation_status_*`
fields, so wrappers do not need to know the nested validation-checklist schema.
They also expose validation-checklist artifact completeness diagnostics:
`implementation_status_checklist_artifact_present`,
`implementation_status_checklist_artifact_complete`,
`implementation_status_checklist_artifact_missing_fields`, and
`implementation_status_checklist_artifact_missing_field_count`, so stale bundles
generated before the nested artifact contract fail closed with
`implementation_status_artifact_incomplete`, emit a structured
`regenerate_package_with_implementation_status_artifact` repair command, and can
be rejected or regenerated.
When the observed completeness value does not satisfy the required value,
`inspect-handoff` also adds
`implementation_status_checklist_artifact_complete_required_value_mismatch` to
shareability blocking reasons.
Failure actions include the exact missing `implementation_status_artifact`
required fields so operators can identify stale bundles that must be regenerated.
The structured `regenerate_package_with_implementation_status_artifact` repair
command also carries `missing_fields`, `missing_field_count`,
`repair_mode=regenerate_package`, `artifact_contract=implementation_status_artifact`,
and the safe-share blocker ID for wrapper UI repair guidance.
The nested `shareability.safe_to_share_requires` list includes
`implementation_status_checklist_artifact_complete` and
`implementation_status_checklist_artifact_complete_matches_required_value`, so
wrappers can treat this as an explicit safe-share prerequisite rather than a
prose-only warning.
The same `shareability.safe_to_share_requires` contract includes
`labeler_route_authorization_runtime_checklist_gate_met`, so packages missing
runtime route-checklist proof fail closed before labeler links are shared.
Top-level `inspect-handoff`, nested `shareability`, and `inspection-targets.json`
also expose `implementation_status_checklist_artifact_complete_required_value=true`
plus `implementation_status_checklist_artifact_complete_matches_required_value`
for direct observed-versus-required comparisons.
`inspect-handoff` also includes a compact
`implementation_status_checklist_artifact_gate` object with schema, observed
value, required value, match status, missing fields, fail-closed reason,
mismatch reason, and repair command ID.
`inspection-targets.json` advertises the matching
`implementation_status_checklist_artifact_gate_contract` exemplar so wrappers
can interpret the gate object without scraping human-readable files.
Wrappers should prefer these machine-readable fields over scraping README or
HTML content.

## Evidence commands to complete before sharing

Use the generated launch bundle paths where available.

```bash
scripts/py -m fisheye.utils.labeling_work record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json --operator <operator>
scripts/py -m fisheye.utils.labeling_work record-browser-response-security-evidence --evidence browser-response-security-evidence-template.json --operator <operator> --capture-url <deployed-my-datasets-url-with-expected-user> --authenticated-test-user <same-user-as-expected-user>
scripts/py -m fisheye.utils.labeling_work record-identity-source-evidence --evidence identity-source-evidence-template.json --operator <operator> --expected-user <user> --resolved-user <user>
scripts/py -m fisheye.utils.labeling_work record-browser-smoke-evidence --evidence browser-smoke-evidence-template.json --expected-user <user> --resolved-user <user> --operator <operator> --browser-only-runtime-verified --no-local-palette-install-verified --no-local-crimson-install-verified --no-local-conda-or-project-dependencies-verified --personalized-dataset-queue-verified --preferred-labeler-entry-url-matches-personal-dataset-queue --personalized-labeler-entry-url-matches-personal-dataset-queue --personalized-work-dashboard-verified
scripts/py -m fisheye.utils.labeling_work record-disposable-zarr-mutation-smoke-evidence --evidence disposable-zarr-mutation-smoke-evidence-template.json --workflow-kind <kind> --mutation-event-id <event-id> --event-lookup-report <event-id>-lookup.json --operator <operator> --labeler-user <user> --task-scoped-training-zarr-write-verified --browser-no-direct-zarr-write-authority-verified --handoff-artifacts-metadata-only-verified --browser-no-csv-or-handoff-write-verified --client-target-selector-rejection-verified --operator-event-lookup-verified --bad-mutation-recovery-verified --bad-mutation-recovery-mode <restore_backup|regenerate_known_good|discard_disposable> --bad-mutation-recovery-report <path-or-note>
scripts/py -m fisheye.utils.labeling_work apply-operator-evidence-templates --path validation-checklist.json --operator <operator>
scripts/py -m fisheye.utils.labeling_work inspect-handoff --path <handoff-or-launch-bundle-path> --require-shareable
```

## Validation status

Current focused validation evidence:

- Route suite: `38 passed in 34.71s`.
- Launch-bundle/checklist inspection focused tests: `2 passed, 1 warning in 0.68s`.

This validation covers the route contract and generated launch-bundle checklist,
inspection, repair, and manifest metadata paths exercised by those tests. It
does not prove deployment-specific browser/proxy behavior, identity-source
resolution, backup execution, or real/disposable training-Zarr mutation safety.
Those remain operator evidence gates and must be executed before treating
labeler links as safe to share.
