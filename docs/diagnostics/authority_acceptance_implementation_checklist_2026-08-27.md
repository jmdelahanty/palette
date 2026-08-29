# Authority And Scientific Acceptance Implementation Checklist

<!-- contract-meta
status: accepted-design
implementation: planned
last_updated: 2026-08-27
-->

## Purpose

Converge Palette's detection, keypoint, subject-mask, derived-analysis, and
cohort acceptance gates without collapsing their distinct scientific lineages
or atomic publication boundaries.

The implementation must keep four claims independent:

1. technical completion;
2. canonical schema, identity, lineage, and content validity;
3. use-scoped scientific acceptance; and
4. selector or authority activation.

The common addition is an immutable scientific acceptance receipt and a
use-aware resolver. Existing product-specific manifests, bundles, selectors,
and review-state payloads remain separate inputs to that receipt.

## Accepted architecture

### Keep product bundles separate

Detection, keypoint, subject-mask, and chaser bundles retain their own exact
manifests and payload digests. They have different row domains, source
lineages, member families, edit/finalization lifecycles, and atomic commit
requirements. A refactor must not replace them with one cross-product mutable
bundle.

Shared machinery should instead provide:

- canonical manifest serialization and digest validation;
- candidate inspection and frozen proof records;
- publication leases, generations, rollback, and one atomic visibility write;
- direct and consolidated metadata agreement;
- acceptance-receipt construction and validation;
- use-scoped resolution with normalized result and failure reasons; and
- common tamper, stale-generation, selector-movement, and retry tests.

Each product keeps a small adapter that declares its exact member schema,
logical-content validator, source lineage, selector mechanism, and acceptance
policy. Cross-product workflows and cohorts compose exact references to those
independent bundle digests; they do not copy or redefine the member payloads.

### Terminology

- `canonical`: contract-valid representation and lineage; no activation or
  review implied.
- `complete`: writer lifecycle completed; no review implied.
- `selector_eligible`: technically visible to the corresponding selector; no
  review implied.
- `authoritative`: exact selected source for a declared fact or consumer scope.
- `accepted`: exact artifact or bundle approved for a declared intended use by
  a digest-bound decision and policy.
- `reviewed_authority`: accepted authority whose policy requires and records
  the applicable human, algorithmic, hybrid, or spot-check review.
- `stale`: source lineage no longer matches; independent of all states above.

### Current product matrix

| Surface | Technical authority | Current review evidence | Required convergence |
| --- | --- | --- | --- |
| `detect_runs/<run>` | matching `latest` / `latest_complete`, canonical-v3 contract and manifest digest | none required for exact model output | preserve review-free canonical model-observation authority |
| `refined_detect_runs/<run>` | family `authoritative_run` or typed immutable refined activation | `detect_review_status` and, on the strict path, typed approval provenance | make reviewed-use resolution require one exact acceptance receipt; converge generic and typed activation |
| legacy `refined_keypoints_runs/<run>` | manual backend may set family `authoritative_run`; other writers only move review-status pointers | `keypoint_review_status` | separate selected authority from reviewed acceptance and remove path-dependent semantics |
| keypoint v2 bundle | root `keypoint_bundle_authority` | reviewed plan/status is not part of the root technical envelope | bind optional use-scoped receipt without weakening atomic bundle validation |
| `subject_mask_runs/<run>` | raw member of technical mask bundle | raw review attrs are compatibility/operator evidence | preserve exact model-probability authority without universal manual review |
| `refined_subject_masks_runs/<run>` | sealed family run may use `authoritative_run` | per-component statuses plus aggregate run state | bind dense payload, component decision snapshot, lifecycle, QC, and intended use in the receipt |
| subject-mask bundle | root `subject_mask_authority` atomically binds raw, refined, and quality members | bundle activation currently does not require review | retain root atomicity and add receipt digest/use-aware resolution alongside it |
| chaser component | immutable manifest, component authority envelope, and exact dependency handles | component-specific scientific review only when declared | do not invent upstream manual-review gates; compose accepted sources by exact digest |
| cohort/release | frozen cohort and source-set receipts | no shared collection acceptance decision | add one collection receipt with member, sampling, exception, coverage, and policy evidence |

## Safety and compatibility rules

- [x] Do not infer reviewed acceptance from `authoritative_run`,
      `keypoint_bundle_authority`, `subject_mask_authority`, `latest_complete`,
      `step_status=ok`, or selector eligibility.
- [x] Do not infer technical authority from `*_review_status_latest` or an
      approved review payload.
- [x] Do not require manual review for raw model-output consumers or derived
      suppliers unless their declared intended-use policy requires it.
- [x] Keep a validated body-frame supplier sufficient for consumers of exact
      anatomical axes and validity; do not reopen landmark review merely
      because keypoints are sealed upstream lineage.
- [ ] Never fabricate historical reviewer identity, intended use, acceptance
      state, policy, sampling coverage, or receipt digest.
- [ ] Keep existing arrays, manifests, selectors, and review attrs immutable;
      migrate through classification, additive receipts, or successors.
- [ ] Preserve explicit technical/diagnostic readers. Add strict use-aware
      readers rather than globally changing every loader to require acceptance.
- [ ] Preserve `*_review_status_latest` as compatibility/task pointers until
      their callers migrate; never retain them as scientific selectors.

## Phase 0 — Freeze the divergence inventory

- [x] Audit generic approval, run resolution, stage catalog, and authority
      primitives.
- [x] Audit raw/refined detection, legacy/v2 keypoints, refined/bundled subject
      masks, registry consumers, training exports, migration paths, and chaser
      cohort release.
- [x] Record the normative terminology in `docs/current_pipeline_contract.md`
      and the agent rules in `AGENTS.md`.
- [ ] Add a read-only divergence diagnostic reporting at least:
  - authority pointer without acceptance receipt;
  - approved review without selected authority;
  - review-status pointer and selected run disagreement;
  - generic versus typed authority-provenance shape;
  - technical root bundle without use-scoped acceptance;
  - receipt/run, receipt/manifest, receipt/generation, or intended-use mismatch;
  - source-matching or `latest` fallback that bypasses completion or eligibility.
- [ ] Run the diagnostic over representative legacy, current family-scoped,
      keypoint-v2, subject-mask-bundle, clipped, and whole-recording archives.
- [ ] Freeze counts and exact example identities before changing resolution.

Primary locations:

- `src/fisheye/shared/zarr_run_completion.py`
- `src/fisheye/shared/run_resolution.py`
- `src/fisheye/cli/palette.py`
- `src/fisheye/registry/stage_catalog.py`
- `src/fisheye/shared/recording_artifact_inventory.py`
- `src/fisheye/registry/maintenance.py`

## Phase 1 — Shared scientific acceptance receipt

- [ ] Add one pure schema/validator module, provisionally
      `src/fisheye/shared/zarr/scientific_acceptance.py`.
- [ ] Freeze a closed schema ID/version and canonical JSON/digest grammar.
- [ ] Bind the exact recording/dataset identity and product scope.
- [ ] Bind an exact run, root bundle, component, or collection identity plus
      its manifest/logical-content/member digests.
- [ ] Record decision state, intended use, method, actor/evaluator, timestamp,
      notes, and policy ID/version/digest.
- [ ] Record automated QC inputs, thresholds, outputs, and result digest.
- [ ] Record review coverage or sampling plan, exclusions, exceptions, and
      quarantined members without treating missing evidence as acceptance.
- [ ] Bind the expected authority selector or root-envelope generation when the
      use policy requires an active selection.
- [ ] Bind source freshness and any component-review snapshot digest.
- [ ] Support multiple receipts for one technical bundle when different uses
      require different decisions; never overwrite one use with another.
- [ ] Define exact failure codes for missing, malformed, pending, rejected,
      stale, wrong-use, wrong-run, wrong-digest, wrong-generation, and
      direct/consolidated disagreement.
- [ ] Add pure in-memory tests for deterministic serialization, digest
      reproduction, field validation, use mismatch, tampering, and extension
      rejection.

The intended-use vocabulary must be reconciled before implementation. Existing
surfaces use at least `training`, `full_recording`, and `analysis`; no adapter
may silently equate them or invent `analysis_and_training` without a versioned
policy decision.

## Phase 2 — Shared bundle and activation mechanics

- [ ] Inventory duplicated lease, generation, frozen-inspection, rollback,
      selector-guard, direct/consolidated, and final-visibility code in:
  - `src/fisheye/shared/selector_activation.py`;
  - `src/fisheye/shared/zarr/refined_detection_authority_activation.py`;
  - `src/fisheye/shared/zarr/keypoint_bundle_activation.py`;
  - `src/fisheye/shared/zarr/subject_mask_bundle_publication.py`.
- [ ] Freeze behavioral parity tests before extracting shared mechanics.
- [ ] Define a narrow product adapter/protocol for member enumeration, logical
      validation, authority-envelope construction, and final commit fields.
- [ ] Keep every product's schema and manifest validator product-specific.
- [ ] Preserve exactly one product-appropriate atomic visibility write.
- [ ] Prove rollback and stale-lease recovery preserve both technical authority
      and any acceptance-receipt binding.
- [ ] Prove a cross-product workflow composes independent bundle digests rather
      than copying scientific payloads.
- [ ] Do not make this refactor a prerequisite for the initial receipt schema;
      extract mechanics only after stage adapters demonstrate common behavior.

## Phase 3 — Authority and use-aware resolution

- [ ] Keep `resolve_authoritative_run_name()` explicitly selection-only and
      retain its compatibility contract.
- [ ] Add a normalized result that reports separately:
  - selected product and selection mechanism;
  - technical validation state;
  - acceptance-required policy;
  - acceptance receipt/state/intended use;
  - source and receipt digests;
  - stale or blocked reason.
- [ ] Add a strict use-aware resolver adjacent to
      `src/fisheye/shared/run_resolution.py`; it must never fall back to a newer
      `latest` child after an explicit authority fails.
- [ ] Add product adapters for family `authoritative_run`, canonical raw
      detection, keypoint root bundle, subject-mask root bundle, and exact
      derived dependency handles.
- [ ] Add stage acceptance policy declarations to `StageSpec` or a separate
      versioned policy catalog. Do not encode review requirements in run names.
- [ ] Make generic `palette approve` either route through the declared
      stage-specific activation policy or clearly create only a technical
      selection. It must not mint a reviewed-authority claim by itself.
- [ ] Keep explicit exact-path diagnostic reads available, labeled as
      non-accepted when the requested use requires acceptance.

Primary tests:

- `tests/unit/fisheye/test_zarr_run_completion.py`
- `tests/unit/fisheye/test_run_resolution.py`
- `tests/unit/fisheye/test_selector_activation.py`
- `tests/unit/fisheye/test_palette_run_verbs.py`

## Phase 4 — Detection convergence

- [ ] Preserve canonical raw detection activation without human review:
      matching `latest` / `latest_complete`, canonical-v3 contract and digest,
      completion, eligibility, and deep manifest validation remain sufficient.
- [ ] Make refined-detection reviewed-use authority require the typed receipt
      bound to the exact run manifest and logical-content digest.
- [ ] Reconcile generic `authoritative_run_provenance` with the stricter typed
      refined-detection envelope; strict consumers must not accept the weaker
      shape as reviewed evidence.
- [ ] Route `accept_detect_review.py`, `set_detect_review_status.py`, interactive
      review, and immutable refined activation through one acceptance builder.
- [ ] Reject review/receipt intended-use mismatch.
- [ ] Retire or narrow backfills that synthesize approved scientific status;
      retain only exact legacy-evidence classification when acceptance cannot
      be proven.
- [ ] Preserve selector-ineligible reviewed training candidates without
      mislabeling them production authority.

Primary tests:

- `test_native_canonical_detection_publication.py`
- `test_canonical_detection_source_authority.py`
- `test_accept_detect_review.py`
- `test_set_detect_review_status.py`
- `test_detect_review_backend.py`
- `test_refined_detection_manifest.py`
- `test_refined_detection_crop_source.py`
- `test_activate_refined_detection_authority_batch.py`
- `test_backfill_detect_review_authoritative_run.py`

## Phase 5 — Keypoint convergence

- [ ] Treat raw keypoint model observations, legacy refined runs, and keypoint
      v2 bundle authority as distinct technical profiles.
- [ ] Make legacy approved review and selected authority converge through one
      receipt-aware path; status-only tools must remain explicitly status-only.
- [ ] Bind keypoint-v2 acceptance to the exact bundle manifest, members,
      generation, and relevant refined/keypoint logical digests.
- [ ] Add an acceptance-aware option to the keypoint source resolver without
      changing technical body-frame consumers to require landmark review.
- [ ] Clarify or replace metadata that calls a product a
      `reviewed_keypoint_authority_candidate` while declaring
      `keypoint_authority=false`.
- [ ] Make pose training and landmark-coordinate exports request their declared
      accepted use explicitly.
- [ ] Preserve historical review attrs and selectors; classify them rather than
      rewriting arrays or manifests.

Primary locations/tests:

- `src/fisheye/shared/zarr/keypoint_publication.py`
- `src/fisheye/shared/zarr/keypoint_quality_publication.py`
- `src/fisheye/shared/zarr/refined_keypoint_publication.py`
- `src/fisheye/shared/zarr/keypoint_bundle_activation.py`
- `src/fisheye/shared/subject_position_keypoint_source.py`
- `src/fisheye/tune/keypoint_review_backend.py`
- `tests/unit/fisheye/test_keypoint_bundle_activation.py`
- `tests/unit/fisheye/test_refined_keypoint_publication.py`
- `tests/unit/fisheye/test_keypoint_review_backend.py`
- `tests/unit/fisheye/test_accept_keypoint_review.py`
- `tests/unit/fisheye/test_set_keypoint_review_status.py`

## Phase 6 — Subject-mask convergence

- [ ] Keep raw probability output and low-level technical coordinate authority
      review-free unless the consumer requests an accepted use.
- [ ] Require dense canonical `masks_roi`, complete component approvals,
      aggregate approval, sealed lifecycle, QC/freshness evidence, and exact
      source binding before minting an accepted refined-mask receipt.
- [ ] Construct the receipt before immutable bundle import or from the exact
      source evidence before the current attribute whitelist can discard review
      state.
- [ ] Bind the receipt digest into the root `subject_mask_authority` envelope
      for an accepted-use activation, while retaining an explicitly technical
      activation mode if still required.
- [ ] Add a use-aware root bundle resolver; retain the technical coordinate
      loader for diagnostic/model-output consumers.
- [ ] Make registry technical `ok` distinct from accepted-use readiness.
- [ ] Invalidate or supersede acceptance after dense edits, source drift, stale
      derived metrics, or component-review changes.
- [ ] Preserve the single root atomic commit and lease recovery semantics.

Primary locations/tests:

- `src/fisheye/shared/refined_subject_mask_mutation.py`
- `src/fisheye/tune/refined_subject_mask_review.py`
- `src/fisheye/refinement/finalize_subject_masks.py`
- `src/fisheye/cluster/subject_masks/publish_recording_bundle.py`
- `src/fisheye/shared/zarr/subject_mask_bundle_publication.py`
- `src/fisheye/shared/zarr/subject_mask_bundle_coordinate_authority.py`
- `src/fisheye/shared/subject_mask_registry_status.py`
- `tests/unit/fisheye/test_refined_subject_mask_review.py`
- `tests/unit/fisheye/test_subject_mask_bundle_publication.py`
- `tests/unit/fisheye/test_subject_mask_registry_status.py`
- `tests/unit/fisheye/test_finalize_subject_masks.py`

## Phase 7 — Registry, inventory, training, and consumer projection

- [ ] Add a normalized acceptance table and history rather than extending
      modality-specific opaque review JSON indefinitely.
- [ ] Key rows by dataset/recording, product scope, exact run or bundle,
      intended use, receipt schema/version, and receipt digest.
- [ ] Project technical selection identity, authority generation, acceptance
      decision/method/use, policy digest, source/manifest digest, and exception
      state independently.
- [ ] Preserve `review_status_json` for compatibility and operator display.
- [ ] Make `recording_step_status=ok` remain technical stage evidence, not
      accepted-use evidence.
- [ ] Make registry reconciliation resolve the root `subject_mask_authority`
      instead of independently choosing raw/refined mask children.
- [ ] Remove completion/eligibility bypasses in `_resolve_latest_group()` and
      source-matching fallback paths used for current canonical state.
- [ ] Make detection, pose, and subject-mask training preparation use the
      strict use-aware resolver and then reopen the exact on-disk receipt.
- [ ] Make inventory report selection, technical authority, and acceptance as
      separate fields; never convert its latest-complete fallback into
      scientific authority.

Primary locations/tests:

- `src/fisheye/registry/maintenance.py`
- `src/fisheye/registry/extractors/quality.py`
- `src/fisheye/registry/extractors/masks.py`
- `src/fisheye/registry/migration_bodies.py`
- `src/fisheye/registry/migrations.py`
- `src/fisheye/registry/status_ledger.py`
- `src/fisheye/shared/recording_artifact_inventory.py`
- `src/fisheye/utils/prepare_detect_training_from_registry.py`
- `src/fisheye/utils/prepare_keypoint_training_from_registry.py`
- `src/fisheye/utils/prepare_subject_mask_training_from_registry.py`
- `src/fisheye/utils/export_subject_mask_training_zarr.py`
- `tests/unit/fisheye/test_registry_maintenance.py`
- `tests/unit/fisheye/test_reconcile_dataset_from_root.py`
- `tests/unit/fisheye/test_registry_query.py`
- the three corresponding `test_prepare_*_training_from_registry.py` modules

The migration number must be allocated from the then-current registry head; it
must not be hard-coded from this checklist. Registry acceptance must use the
SQLite runtime loaded by `scripts/py`.

## Phase 8 — Historical classification and migration

- [ ] Implement a read-only census first, with exact outcomes:
      `safe_already_accepted`, `backfillable_unambiguous`, `pending_receipt`,
      `ambiguous_conflict`, and `unmigratable`.
- [ ] Require exact recording, run/bundle, manifest/logical digest, review
      payload, intended use, selector, and freshness agreement before calling a
      historical case unambiguous.
- [ ] Make any backfill dry-run by default and explicit-apply only.
- [ ] Preserve old pointers/statuses and write additive receipts or immutable
      successors; never restamp historical arrays.
- [ ] Revalidate under the archive lock immediately before writes.
- [ ] Consolidate as the final visibility step and verify direct/consolidated
      receipt equality afterward.
- [ ] Quarantine ambiguity; do not choose by newest name, insertion order,
      mtime, nearby path, or apparent semantic equivalence.
- [ ] Emit durable accepted/rejected operation evidence without moving a
      production selector unless a separate activation is authorized.

## Phase 9 — Scalable collection acceptance

- [ ] Run deterministic canonical validation and automated QC for every member.
- [ ] Freeze the QC policy, thresholds, software commit, and result digest.
- [ ] Freeze a reproducible stratified human sample across recording, camera,
      biological subject, provider, protocol role, coverage, and QC strata.
- [ ] Require targeted human review of every automated exception and outlier.
- [ ] Keep excluded and quarantined members explicit; absence is not success.
- [ ] Publish one immutable collection receipt binding:
  - frozen cohort manifest digest and ordered recording IDs;
  - each member's selected bundle/run and acceptance-receipt digest;
  - intended use and policy digest;
  - sampling plan and realized sample;
  - exception, exclusion, and quarantine identities/reasons;
  - expected, accepted, failed, blocked, missing, stale, and inapplicable counts;
  - exact source-authority set digest;
  - actor/reviewer and decision timestamp.
- [ ] Use the collection receipt to gate release/submission/export. Do not make
      it another per-recording Zarr selector.
- [ ] Preserve recording-balanced primary summaries and label pooled
      frame/event summaries separately.

Primary locations/tests:

- `src/fisheye/cohorts/release.py`
- `src/fisheye/analytics_exports/chaser_authority.py`
- `tests/unit/fisheye/test_cohort_registry.py`
- `tests/unit/fisheye/test_chaser_export_authority.py`
- `tests/unit/fisheye/test_chaser_export_authority_integration.py`

## Phase 10 — Chaser cohort execution boundary

The acceptance refactor must not delay safe evidence generation that does not
claim accepted production authority.

- [x] The provider-aware GoodBatBadBat plan freezes exact recording, provider,
      semantic-selection, timing, geometry, relative-frame, body-frame, and
      successor identities.
- [x] Detection bbox centroids and keypoint-triad positions remain separate
      first-class providers; neither is silently substituted for the other.
- [x] A timestamped provider decision retains detection bbox centroid as the
      current position default while preserving the paired-provider products.
- [x] A single-recording body-frame successor canary completed with exact
      receipts and visual inspection; no keypoint-review receipt is required
      for its body-axis-only consumers.
- [x] The existing cohort plot operation completed for 80 eligible recordings
      with exact child and plot receipts.
- [x] Four recordings remain blocked by overlapping/non-ordered semantic steps;
      no legacy interval convention or inferred ordering is permitted.
- [ ] Materialize a durable commit-pinned copy of the frozen body-frame cohort
      task before submission; `/tmp` planning evidence is not the operational
      source of truth.
- [ ] Run the selector-ineligible full successor cohort for eligible recordings
      with no selector, registry, or production-authority writes.
- [ ] Preserve exact per-recording and per-module complete, blocked, missing,
      invalid, review-required, stale, and inapplicable outcomes.
- [ ] Rehash worker, component, and plot receipts and inspect representative
      outputs across cameras/providers before calling the evidence campaign
      complete.
- [ ] Keep gaze blocked where eye-orientation acceptance is missing, and keep
      response-regime or visit-level products blocked where their scientific
      successor contract is still absent. Independent distance, occupancy,
      radial/near-field, trial, bout, escape/freeze, and body-bearing products
      may still complete.
- [ ] Do not call the selector-ineligible evidence campaign a production or
      fully accepted cohort release.
- [ ] Require the Phase 9 collection receipt before later production release,
      selector activation, or a claim that all accepted full-profile modules
      are cohort-ready.

## Phase 11 — Validation and integration gates

- [ ] Add `tests/unit/fisheye/test_scientific_acceptance.py` for the common
      schema, serializer, validator, and use-aware resolution.
- [ ] Add real writer -> publisher -> resolver -> unpatched consumer tests for
      detection, keypoint, subject-mask, and one composed chaser path.
- [ ] Test missing/malformed/stale receipts, content tampering, selector
      movement, generation drift, intended-use mismatch, aggregate/component
      disagreement, exception coverage, retry, rollback, and consolidation.
- [ ] Test that technical raw/model consumers remain valid without manual
      review and that acceptance-required consumers fail closed.
- [ ] Test that body-frame-only chaser consumers do not acquire an upstream
      landmark-review dependency.
- [ ] Use `scripts/py` for all Python commands and run Zarr-heavy pytest outside
      the sandbox.
- [ ] Pass formatting/static checks and every repository-required CI check.
- [ ] Do not merge, integrate, update the shared checkout, activate selectors,
      or describe an implementation branch as complete while any required CI
      check is failed, cancelled, skipped unexpectedly, timed out, or pending.

## Recommended implementation order

1. Phases 0–1: divergence census and pure receipt schema.
2. Phase 3: normalized use-aware resolver and stage policy declaration.
3. Phases 4–6 in parallel after the common schema freezes.
4. Phase 7: registry and consumer convergence against the real stage adapters.
5. Phase 8: read-only historical census, then narrowly authorized migration.
6. Phase 9: collection acceptance and release gate.
7. Phase 2 extraction only after parity tests prove which activation mechanics
   are genuinely common.

Phase 10 selector-ineligible chaser evidence may proceed independently after
its current commit-pinned deployment and task checks. Its production release
remains gated by Phase 9.

## Completion criteria

This work is complete only when:

- every maintained product declares its technical authority mechanism and
  intended-use acceptance policy;
- no maintained consumer infers review from a pointer, name, completion, or
  bundle activation;
- reviewed-use consumers validate one exact receipt and source digest;
- technical model-output and diagnostic consumers remain explicit and usable;
- registry and inventory expose technical and accepted state independently;
- historical ambiguity remains visible rather than normalized;
- collection release binds every accepted/excluded member and exception; and
- all required CI and controlled canary evidence pass.
