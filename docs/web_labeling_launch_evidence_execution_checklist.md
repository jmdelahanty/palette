# Web Labeling Launch Evidence Execution Checklist

This checklist is for the operator-only evidence collection step after the
browser-labeling implementation and focused route tests are in place.

Current implementation checkpoint:

- Focused web-route validation passed with `38 passed in 34.71s` for `tests/unit/fisheye/test_labeling_web_routes.py`.
- Focused launch-bundle/checklist inspection validation passed with `2 passed, 1 warning in 0.68s` for `test_inspect_handoff_launch_evidence_execution_checklist_reports_directory_and_zip` and `test_export_launch_bundle_cli_writes_plan_readiness_handoffs_and_zip`.
- Generated browser-smoke evidence templates include the guarded `/labeling?expected_user=<user>` alias for correct-user and wrong-user smoke rows.
- Top-level launch-bundle manifests include the same `browser_mutation_write_policy` and `browser_mutation_write_checklist` contract as generated handoff artifacts.
- Labeler links are still not broadly shareable until launch evidence is recorded, approved, applied, and inspected.
- Browser labeling mutations must remain server-side writes to assigned task/training Zarrs.
- CSV, HTML, JSON, launch bundles, handoff files, and intermediate CSVs are metadata or packaging artifacts only.
- Each recording must have exactly one active assigned owner.

Validated implementation baseline commands:

```bash
scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_web_routes.py -q
scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_labeling_assignment_store.py::test_inspect_handoff_launch_evidence_execution_checklist_reports_directory_and_zip tests/unit/fisheye/test_labeling_assignment_store.py::test_export_launch_bundle_cli_writes_plan_readiness_handoffs_and_zip -q
```

These tests validate route and generated-artifact contracts only. They do not
replace any operator evidence gate below.

Inspection and repair contract:

- Artifact file: `launch-evidence-execution-checklist.txt`.
- Inspection field: `launch_evidence_execution_checklist`.
- Summary field: `launch_evidence_execution_checklist_summary`.
- Required phrase contract: `shareability_launch_evidence_execution_checklist_required_phrases`.
- Required final phrase: `labeler_links_safe_to_share=true`.
- Blocking reasons: `launch_evidence_execution_checklist_missing`, `launch_evidence_execution_checklist_incomplete`, and `launch_evidence_execution_checklist_invalid`.
- Repair command: `regenerate_package_with_launch_evidence_execution_checklist`.
- Repair metadata fields: `required_file`, `required_phrase_contract`, `required_phrases`, `missing_phrases`, and `missing_phrase_count`.

## Inputs to fill in

- `PACKAGE_PATH`: launch bundle or handoff directory/zip to inspect.
- `VALIDATION_CHECKLIST`: usually `validation-checklist.json` in the launch bundle.
- `OPERATOR`: human operator name or ID.
- `USER`: representative expected labeler user.
- `DEPLOYED_MY_DATASETS_URL`: deployed `/my-datasets?expected_user=USER` URL.
- `MUTATION_EVENT_ID`: event ID produced by the disposable-Zarr browser mutation smoke.
- `EVENT_LOOKUP_REPORT`: JSON report proving the mutation event was inspected.

## Gate 1: mutable Zarr backup evidence

Status before execution: required, operator-only, not satisfied by route tests.

Required proof:

- Mutable Zarr backup targets were identified from the generated plan.
- Backup execution manifest exists.
- Restore test result is recorded.
- Operator approved the evidence.

Command shape:

```bash
scripts/py -m fisheye.utils.labeling_work record-zarr-backup-evidence \
  --evidence zarr-backup-evidence-template.json \
  --execution-manifest <manifest> \
  --target-index <index> \
  --restore-test-result <result> \
  --operator <operator>
```

## Gate 2: browser/proxy response-security evidence

Status before execution: required, operator-only, not satisfied by route tests.

Required proof:

- Capture URL is the deployed guarded personalized queue, not an unguarded fallback.
- `expected_user` query is present.
- Authenticated test user equals the expected user.
- Captured headers satisfy the response-security policy.
- Personalized alias headers match canonical route expectations.

Command shape:

```bash
scripts/py -m fisheye.utils.labeling_work record-browser-response-security-evidence \
  --evidence browser-response-security-evidence-template.json \
  --header "Cache-Control=<value>" \
  --header "X-Frame-Options=<value>" \
  --operator <operator> \
  --capture-url <deployed-my-datasets-url-with-expected-user> \
  --authenticated-test-user <same-user-as-expected-user>
```

## Gate 3: deployment identity-source evidence

Status before execution: required, operator-only, not satisfied by route tests.

Required proof:

- Deployed identity probe resolves the expected user.
- Resolved user exactly equals expected user.
- Preferred and personalized entry URLs resolve to guarded `/my-datasets?expected_user=<user>`.
- `/labeling?expected_user=<user>` remains only the human-readable alias.

Command shape:

```bash
scripts/py -m fisheye.utils.labeling_work record-identity-source-evidence \
  --evidence identity-source-evidence-template.json \
  --expected-user <user> \
  --resolved-user <resolved-user> \
  --operator <operator> \
  --authenticated-session-context DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED
```

## Gate 4: representative browser smoke evidence

Status before execution: required, operator-only, not satisfied by route tests.

Required proof:

- Browser-only runtime works without local Palette, Crimson, Conda, or project dependencies for the labeler.
- Guarded `/my-datasets?expected_user=<user>` shows only assigned work.
- `/labeling?expected_user=<user>` works as a human-readable queue alias.
- `/my-work?expected_user=<user>` works as the personalized dashboard fallback.
- Wrong-user personalized aliases reject expected-user mismatches.
- Stale-tab, completion/read-only, redacted support, and operator-reopen behavior were checked.

Command shape:

```bash
scripts/py -m fisheye.utils.labeling_work record-browser-smoke-evidence \
  --evidence browser-smoke-evidence-template.json \
  --expected-user <user> \
  --resolved-user <resolved-user> \
  --operator <operator> \
  --browser-only-runtime-verified \
  --no-local-palette-install-verified \
  --no-local-crimson-install-verified \
  --no-local-conda-or-project-dependencies-verified \
  --personalized-dataset-queue-verified \
  --preferred-labeler-entry-url-matches-personal-dataset-queue \
  --personalized-labeler-entry-url-matches-personal-dataset-queue \
  --personalized-work-dashboard-verified
```

## Gate 5: disposable-Zarr mutation smoke evidence

Status before execution: required, operator-only, not satisfied by route tests.

Required proof:

- Smoke uses disposable or recoverable Zarr data.
- Mutation event proves task-scoped training-Zarr write behavior.
- Browser did not write CSV, handoff, HTML, JSON, or intermediate CSV artifacts.
- Browser did not receive direct Zarr write authority.
- Browser-supplied CSV/Zarr/write-target selectors were rejected.
- Operator inspected the event lookup report.
- Bad-mutation recovery path is recorded.

Command shape:

```bash
scripts/py -m fisheye.utils.labeling_work record-disposable-zarr-mutation-smoke-evidence \
  --evidence disposable-zarr-mutation-smoke-evidence-template.json \
  --workflow-kind <kind> \
  --mutation-event-id <event-id> \
  --event-lookup-report <event-id>-lookup.json \
  --operator <operator> \
  --labeler-user <user> \
  --task-scoped-training-zarr-write-verified \
  --browser-no-direct-zarr-write-authority-verified \
  --handoff-artifacts-metadata-only-verified \
  --browser-no-csv-or-handoff-write-verified \
  --client-target-selector-rejection-verified \
  --operator-event-lookup-verified \
  --bad-mutation-recovery-verified \
  --bad-mutation-recovery-mode <restore_backup|regenerate_known_good|discard_disposable> \
  --bad-mutation-recovery-report <path-or-note>
```

## Apply approved evidence

Run after the evidence templates above are approved.

```bash
scripts/py -m fisheye.utils.labeling_work apply-operator-evidence-templates \
  --path validation-checklist.json \
  --operator <operator>
```

If the apply report says package checksums or manifests must be refreshed, do
that before sharing labeler links.

## Final shareability inspection

This is the only passing condition for broad link sharing.

```bash
scripts/py -m fisheye.utils.labeling_work inspect-handoff \
  --path <handoff-or-launch-bundle-path> \
  --require-shareable
```

Share labeler links only when the inspection succeeds and reports
`labeler_links_safe_to_share=true`.

If inspection fails, use the structured repair commands in the inspection JSON
before repeating evidence application and inspection.
