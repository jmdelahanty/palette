# Pipeline Metadata Boundaries

Purpose: define where metadata belongs in Palette pipeline stages so we get
auditable outputs without duplicating semantics across code and docs.

## Decision Rule

Use this test:

- "What happened in this specific run?" -> put it in run `provenance` attrs.
- "What does this field mean?" -> put it in a contract/schema doc.
- "How is the value computed?" -> put it in code.
- "How do operators execute/review it?" -> put it in workflow docs.

## Canonical Ownership

| Concern | Canonical owner | Example |
| --- | --- | --- |
| Resolved parameter values | Run attrs (`provenance.parameters`) | `min_circularity=0.77` used by one run |
| Lineage/source pointers | Run attrs (`provenance.inputs` + `source_*`) | `source_crop_run`, `source_keypoints_run` |
| Software identity | Run attrs (`provenance.git`, `contract`) | commit hash, branch, contract version |
| Field definitions (type/units/range/default) | Contract doc | "`min_circularity` is in [0,1], 0 disables filter" |
| Algorithm details | Code + code comments | ellipse canonicalization, overlap rejection logic |
| Operator sequence and flags | Workflow docs | batch run, review, backfill commands |
| Compliance checks | Diagnostics + tests | `check_provenance_capture`, unit tests |

## Anti-Drift Policy

- Do not duplicate parameter semantics in run attrs.
- Do not treat CLI help text as the canonical field-definition source.
- Keep one canonical contract per schema family (for example
  `palette_stage_provenance`), with `provenance.stage` as discriminator.
- Provenance should reference the contract by `name` + `version`; it should not
  embed long prose definitions.

## Minimum Provenance Contract Pattern

For stage runs, write:

- `attrs["provenance"]["contract"] = {name, version}`
- `attrs["provenance"]["stage"]`
- `attrs["provenance"]["created_at_utc"]`
- `attrs["provenance"]["parameters"]`
- `attrs["provenance"]["inputs"]`

Recommended:

- `attrs["provenance"]["git"]`
- `attrs["provenance"]["environment"]`
- `attrs["provenance"]["platform"]`
- top-level compatibility attrs only when needed by legacy readers.

## When Adding a New Parameter

1. Add parameter value capture in run provenance.
2. Add parameter definition to the stage contract doc:
   type, units, valid range, default, disable behavior, and edge-case notes.
3. Add/adjust diagnostics and tests for required presence/shape.
4. If compatibility fallback exists, document fallback order once in the
   contract (not in multiple workflow docs).
5. Update workflow docs only for operator-facing behavior changes.

## References

- `docs/provenance_contract_draft.md`
- `docs/provenance_checks.md`
- `docs/zarr_run_completion_contract.md`
- `src/fisheye/docs/provenance_workflow.md`
