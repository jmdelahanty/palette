# Shared-helper consolidation: preserve the contracts, reduce competing implementations

> Update added 2026-09-06: all seven primary characterization cases below still
> reproduce at the rechecked main-equivalent code; those passes confirm defects,
> not fixes. The single-runtime-hook statement is superseded: main `1bf9d919`
> registers six stage-specific verifiers and revalidates reused inputs before
> execution. This closes specific gaps, not all admission boundaries. See the
> [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation) for exact versions and evidence; this report does not authorize broad consolidation.

Date: 2026-09-04. Reviewed commit: `73bee0d5194c662e3b7e535be3de92db7ef53f63`.

Six parallel reviewers examined publication/coordinates, registry/model selection, workflows/cluster execution, readers/exports, inference/storage, and enforcement/tests. The primary reviewer independently checked representative code, ran a repository-wide function-clone census, and reproduced seven cases from the highest-impact findings. This is a read-only code review, not authorization to refactor. Only review documents were changed; temporary probes used synthetic data. No source changes, live registry/store mutations, installations, commits, pushes, deployments, or `/groups` writes occurred.

This report extends the [independent design assessment](independent_codebase_design_assessment_2026-09-04.md) and [handoff second opinion](review_wave_second_opinion_2026-09-04.md). Findings are for adoption into the existing authority-consolidation work queue, not a new parallel queue. Source anchors below refer to `src/fisheye/` unless another root is shown.

## Bottom line

There is real duplication worth removing. More importantly, several duplicated implementations have already drifted in validation, ownership, memory bounds, or recovery behavior. The useful target is **fewer independent implementations of each invariant**, not fewer scientific checks.

The strongest existing implementations should become small, tested owners of reusable mechanics. Stage-specific scientific schemas, coordinate systems, validity rules, supplier sufficiency, and authority policies should remain explicit adapters. Making every caller use a weaker generic helper would reduce code while damaging the features that make Palette valuable.

The first changes should be narrow and testable: close the forwarded-argument ownership hole, adopt a strict metadata comparator, align model-task vocabulary, and migrate genuinely identical helper copies. Then consolidate the asynchronous writer lifecycle and publication validation/recovery machinery under shared adversarial tests. Broad file splitting, a new workflow framework, and a universal publisher/authority schema are not prerequisites.

## User-confirmed design constraints

The scientific defaults in command builders are intentional defaults for this camera configuration and rig. Preserve their values and intended applicability. Represent them as a named/versioned, validated recipe with explicit effective values and permitted overrides; command rendering translates that recipe, while resource configuration remains separate. The existing command already records values, so this is not a diagnosis of universally missing parameter provenance.

The user endorsed three distinct product claims:

- **Computation complete:** the declared numerical products and their scientific contracts are validated.
- **Presentation complete:** the associated required plots/specifications are validated against the numerical identities they depict.
- **Deliverable complete:** all products required by the selected workflow are ready.

A failed required plot keeps the deliverable incomplete without making valid numerical bytes invalid. Presentation retry should not silently alter numerical identity. This direction is documented, not implemented, and does not waive scientific acceptance, explicit human-review contracts, or required CI gates. Details and caveats are now in section 6 of the independent assessment.

## Highest-value consolidation findings

### 1. High: copied argument guards do not preserve materializer output ownership

`analysis_workflows/materializers/bout_kinematics.py:240` rejects exact managed option names, then appends accepted arguments after its own output-path/run arguments (`:675`). The scientific writer uses an abbreviating `ArgumentParser` (`analysis/bout_kinematics.py:3555`). Consequently, the materializer accepts:

```text
--output-zarr-p /temporary/source.zarr --overw
```

The writer parses these as an output-path override and `overwrite=True`. The primary reviewer reproduced both the separated-value and abbreviated `--output-zarr-p=...` forms, intercepting execution immediately after parsing. In both cases the parsed output became the input archive rather than the planned scratch output. No writer ran and no source archive was modified. Full managed option names and full `--flag=value` forms are already rejected.

The same exact-token guard pattern exists in swim-bout and track materializers (`swim_bouts.py:131`, `track_kinematics.py:157`); those other writers were not dynamically exercised here.

**Consolidation boundary:** a typed scientific-options interface shared by recipes, direct CLI parsing, and materializers. Output location, run identity, overwrite, and publication ownership are not scientific overrides. No existing general argument-ownership helper was found that can simply be adopted. A compatibility adapter should disable abbreviations and validate the resulting parsed ownership fields, not merely compare input tokens.

**Required protection:** test full names, prefixes, aliases, repeated options, equals forms, and `--` forwarding at every maintained materializer. Rejection must occur before output creation. Preserve the rig defaults while making their scientific override surface explicit.

### 2. High: strict JSON comparison is duplicated, while the shared metadata gate uses weaker equality

`shared/zarr/metadata_equivalence.py:185` uses ordinary dictionary equality for its exact direct/consolidated declaration comparison. Python considers `True` equal to `1`, and `1` equal to `1.0`. The primary review reproduced both mismatches using temporary JSON metadata files; the real gate returned a receipt.

Type-strict recursive comparators already exist in `shared/pixel_frame_authority.py:638`, `coordinate_descriptor.py:377`, `selected_calibration.py:1173`, and related transform modules. The gate has active source-handle and registry-finalizer callers, so this is a maintained-boundary weakness, not just unused utility duplication. No mismatched live store was inspected.

**Consolidation boundary:** one small strict contract-value comparator, extracted from existing semantics, used by metadata equivalence and applicable contract validators. Do not substitute the permissive report/display conversion in `json_safety`.

**Required protection:** nested bool/int and int/float mismatches, container distinctions, and nonfinite-input policy. Preserve the deliberately allowed omitted/null/exact-empty leaf-consolidation representation equivalence. Static detection of new local comparator copies can supplement, not replace, behavioral tests.

### 3. High: model selection has competing task, admission, and cache semantics

The registry writer normalizes `segmentation` to `subject_masks` and recognizes explicit `subject_masks` (`registry/db.py:296`). The resolver normalizes `segmentation` to `eye_masks` and does not recognize `subject_masks` (`registry/model_resolution.py:167`). The primary reviewer reproduced this disagreement. A parallel reviewer also demonstrated that `load_candidates(task="pose")` can admit an explicitly declared subject-mask row with a pose-looking name because an unrecognized declaration falls back to filename inference (`:454`). This is candidate misclassification, not demonstrated successful wrong-model inference; downstream binding checks may reject it.

Admission order is also duplicated. Single-run keypoint selection validates candidates before choosing (`utils/run_keypoints_with_registry_model.py:265`, `:674`); batch selection picks the highest-ranked candidate before validating only that candidate (`utils/run_keypoints_batch.py:1158`). The latter can fail despite a lower-ranked valid candidate. It does not imply the batch skips its selected-model validation.

Finally, the existing shared batch helper caches too much. `utils/batch_registry_model_resolution.py:24` caches the entire `ResolvedModel` and mutable provenance payload by recording directory. Payload builders include per-archive `inputs.output_zarr` (`run_keypoints_batch.py:1187`) and sometimes input-video fields. A synthetic two-plan reviewer probe returned the same object and first archive's path for the second archive. The existing cache test (`test_run_keypoints_batch.py:755`) intentionally covers analysis/training stores sharing a recording but omits those provenance fields.

**Consolidation boundary:** a registry-domain task vocabulary and a model-selection service returning the exact selected binding, policy, and rejection evidence. Cache only context-independent selection by an explicit equivalence key; construct input/output/invocation provenance separately for every plan. Preserve modality-specific binding checks. Historical migration vocabulary must remain frozen or versioned rather than silently following an evolving normalizer.

**Required protection:** explicit task declarations outrank name heuristics; distinguish missing legacy declarations from unsupported declarations. Require single/batch parity for invalid leading candidates and ties, and correct per-plan provenance after cache reuse.

### 4. Medium-high: the two-buffer writer copies have different failure cleanup

`detection/detect_keypoints_yolo.py:998` and `segmentation/infer_unet_subject_masks.py:2239` each implement a sequential, two-buffer, asynchronous physical-shard writer. The keypoint implementation has bounded shutdown, idempotent abort/finish, and inference-failure cleanup (`:392`, `:1155`). The probability implementation calls `_raise_error()` before sending its worker sentinel (`infer_unet_subject_masks.py:2425`) and has no abort contract. Its normal caller finishes it only after successful processing (`:2939`).

A reviewer executed the actual probability class extracted into an isolated namespace with a fake destination raising `OSError`. `finish()` raised, but its daemon worker remained alive. The reviewer then manually stopped the synthetic worker. This proves cleanup drift, not a production corruption incident.

**Consolidation boundary:** extract the stronger bounded worker lifecycle into the existing shared Zarr subsystem. Retain stage adapters: keypoints write heterogeneous arrays; probability buffers are channel-first internally and hash/write by channel. Keep full destination validation distinct from explicitly deferred validation that a later mandatory stage must perform.

**Required protection:** model/write failures before finish, blocked writes, shutdown timeouts, repeat abort/finish, partial last shard, mixed dtypes, and actual destination layout versus requested layout. Preserve the two-buffer memory bound and whole physical write ownership; do not reduce these receipts to a generic success boolean.

### 5. Medium-high: export validation copies differ before and after manifest selection

Eye and kinematics exports use a reduced staged hash check, select the new manifest, then invoke fuller decoded validators (`analytics_exports/eye_trace_samples.py:884`, `kinematics_samples.py:1656`). The fuller checks additionally validate constant lineage, frame/time ordering, and—in kinematics—physical Parquet policy (`eye_trace_samples.py:650`, `kinematics_samples.py:1327`). Generic staging checks do not replace these scientific row checks. A later failure does not undo the already selected manifest.

The better pattern exists in tail exports: `_decoded_part_validation` accepts explicit parts and is called both before publication and when validating the selected publication (`tail_trace_samples.py:1535`, `:1663`, `:1869`). This is a code-path ordering finding, not evidence that current live exports are corrupt.

Physical-policy validation also differs. Kinematics checks row-group sizes, compression codec, and dictionary encodings (`:1355`). Eye and tail validate declared policy receipts but omit those corresponding physical checks in their decoded validators. A declared compression level is not necessarily independently recoverable from the Parquet codec field; distinguish configured from observed properties.

**Consolidation boundary:** each scientific family supplies one complete path-independent decoded-part validator, used for both staged and selected inventories. Share a small physical-policy verifier. Keep distinct frame/sample/geometry schemas and source, row-group, and part-boundary controls.

**Required protection:** tamper with a lineage column or layout, recompute superficial receipts, and assert rejection before the previous selecting manifest changes. Reuse existing tail failed-replacement and rehashed-tampering tests across these adapters.

### 6. High-impact primitive drift, limited reachability: rollback needs ownership, not just equal values

Refined-mask `_restore_owned_parent_state` (`shared/refined_subject_mask_coordinate_publication.py:5906`) restores attributes when their values equal the attempted values, explicitly discarding owner/run arguments. Another owner can legitimately have the same policy and generation values. The primary reviewer reproduced a takeover where the helper preserved the successor lease but reset its generation and deleted its policy before raising an incomplete-rollback error.

Subject-shape rollback (`shared/subject_shape_coordinate_publication.py:5417`) and the generic atomic publisher use stronger fresh lease/generation checks. The subject-shape same-valued-takeover test at `tests/unit/fisheye/test_subject_shape_coordinate_publication.py:1791` passes.

**Reachability limitation:** the refined activation helper was found only in its module and test callers, not another current source caller. This is a demonstrated primitive inconsistency to fix before adoption, not a claimed production incident.

**Consolidation boundary:** owner-guarded mutation/rollback mechanics with explicit field maps and mutation receipts. Preserve stage-specific proofs, pending formats, and activation contracts. Do not blindly replace every stage activation with today's generic callback interface.

**Required protection:** fail or transfer ownership before/after each mutation, including equal-valued successor epochs and writes that persist before raising. Ownership loss must preserve successor state; lease release/restoration must follow the correct ordering.

### 7. Medium-high: registry connection and transaction policy is duplicated

Safe read-only opening exists in several copies with different URI escaping and timeout policy, including `registry/prune_stale_datasets.py:112`, `chaser_metadata.py:33`, `shadow_publish.py:72`, `shared/recording.py:87`, and `shared/run_resolution.py:82`. Meanwhile query/status/model-resolution CLIs still instantiate `Registry`, whose constructor opens writable SQLite and initializes/migrates schema (`registry/db.py:1239`; callers include `registry/query.py:482` and `registry/status.py:53`).

Current/history persistence is duplicated in `registry/status_ledger.py:105` and `registry/maintenance.py:6933`. The former participates in the registry transaction; the latter uses `with registry.conn`, which can commit its caller's transaction. A reviewer demonstrated the premature commit using in-memory SQLite and the real transaction context. The ordinary reconcile call currently invokes this refresh after its main transaction, so the demonstrated limitation is unsafe composition, not a proven nested production incident. Runtime append-every-event versus reconcile-append-on-change is a legitimate difference to preserve.

Four finalizers also hand-roll `integrity_check`/`fetchone` without foreign-key validation: `cluster/keypoints/registry_finalize.py:66`, `cluster/whole_recording_analysis_registry_finalize.py:178`, `cluster/clipped_inference_registry_finalize.py:65`, and `analysis_workflows/registry_finalize.py:654`. Existing `registry.shadow_publish.validate_registry_sqlite:79` already performs the full integrity and FK checks and records the runtime.

**Consolidation boundary:** explicit reader/writer/schema-upgrader capabilities in a neutral registry module; normalized persistence in `status_ledger`; acceptance through the existing integrity helper. Keep event policy separate from persistence mechanics and preserve registry projection versus scientific authority roles.

**Required protection:** no mutation/migration from observation commands, special characters in URI paths, missing databases, outer rollback, idempotent reconciliation, and FK-invalid fixtures that still pass `integrity_check`. Use the Palette Python SQLite runtime, not a separate SQLite executable.

### 8. Medium: copy/window helpers promise different memory bounds

Detection training export's `_copy_indexed_frames` accepts a batch limit but bypasses it for a wholly contiguous selection (`utils/export_detect_training_zarr.py:1168`). A fake-array reviewer probe requested batch size 128 and observed one 10,000-row read. Subject-mask export similarly bulk-reads ROI images (`export_subject_mask_training_zarr.py:941`) while correctly streaming mask pixels through `MaskStore` (`:956`). Keypoint export has separate bounded sequential/indexed/padding copy variants (`export_keypoint_training_zarr.py:691`).

Eye's irregular/ROI time selection reads the entire coordinate axis before enforcing the selected-row limit (`analysis/eye_angle_io.py:928`). A fake 100,000-coordinate probe requested one row and observed a full-axis read. Modern frame-axis reads with positive FPS take an arithmetic branch and are not implicated. Tail's existing binary search (`tail_kinematics_io.py:390`) located the row in 17 scalar reads; a generalized version must preserve floating timestamps and may need bounded block caching for remote stores.

**Consolidation boundary:** bounded row copying in the existing training-helper owner and bounded monotonic-coordinate lookup reused by domain readers. Keep contiguous fast paths inside the declared budget. A row budget should account for bytes when rows contain large images.

**Required protection:** sentinel arrays rejecting oversized reads, repeated/noncontiguous rows, endpoint semantics, empty ranges, coordinate transforms, and real destination chunk ownership. Do not replace grouped training splits with row-random splits, hide crop-coordinate changes, or parallelize disjoint logical slices sharing a physical chunk. No OOM or throughput improvement was measured.

## Smaller helper migrations and bloat reduction

These are concrete adoption candidates, not a mandate to move everything into `shared`.

| Repeated work | Existing owner or extraction direction | Important preservation condition |
|---|---|---|
| Detect/keypoint `_json_list` and `_json_dict` | `shared/training_zarr_helpers.py:18`, `:38`; subject-mask export already imports them. | Preserve current permissive metadata-decoding behavior; do not reuse it for strict contract validation. |
| Export split, chunk, and string-array utilities | Extend the existing training helpers rather than add another exporter utility module. | Shared and copied split error/normalization behavior differs; string chunks differ (65,536 versus 1,024) and fill policy differs. Freeze outputs before migration. |
| Atomic JSON receipt writers in eye/shape/tail materializers | `shared/json_safety.write_json_atomic:129` or the existing snapshot wrapper. | Preserve serialized/digest bytes and error semantics. Existing helper has unique temporary names, flush/fsync, and cleanup; PID-only local names are weaker. |
| Registry direct-Zarr openers | `registry/zarr_open.py:26`, with compatible shared direct-open mechanics. | `inline_refresh.py:21` falls back from an unsupported keyword to implicit mode. Preserve explicit direct metadata during mutation inspection, not a universal unconsolidated reader policy. |
| Physical-unit write loops in at least nine publishers | Existing `shared/zarr/array_factory.py`, storage planner, and a promoted physical-write kernel. | Keep lazy/incremental hashing, channel layouts, exact inventory, bounded dispatch, and scientific adapters. Array creation already checks contract/plan identity. |
| Strict publication JSON readers/declaration loading/shadow-path checks | Small helpers beside array creation and metadata equivalence. | A supplied metadata snapshot pins a generation; exact whole-subtree inventory is stronger than checking only listed plan paths. |
| Two analytics rename/CAS/cleanup implementations | `analytics_exports/publication.commit_validated_immutable_generation:135`; retain `commit_staged_publication:712` as a validating adapter. | Preserve analytics telemetry, containment, namespace rules, and prior selection on failure; do not unify distinct scientific envelopes. |
| Selected-file staging/inventory/copy machinery in eye/tail/other materializers | Extend the `TreeFile`/`TreeInventory` machinery near `shared/atomic_run_publisher.py:104`. | Whole-tree inventory is not a drop-in replacement for selected-file staging. Preserve source-closing checks, capacity policy, and bounded access. |
| Execution-candidate tombstone workflows in roughly ten materializers | One owner beside existing candidate-execution machinery, using shared lock/failure/equivalence helpers. | Post-execution rejection is not interrupted-publication rollback. Preserve extra unpromoted-candidate and selector-snapshot checks. |
| Per-column streaming export hashers and decoded-array byte traversal | Existing Arrow contract layer and shared Zarr receipt traversal, with explicit grammar adapters. | UTF-8 versus ASCII-escaped JSON, column headers/order, dtype/shape headers, NaN, endianness, and per-channel aggregation change identity. Do not silently change persisted digests. |

The primary review's AST census also found easy-to-locate batch/profile duplication:

- Seven chaser/GoodCopBadCop batch CLIs contain identical 35-line `_query_targets` bodies; examples are `utils/run_chaser_near_field_occupancy.py:38` and `run_goodcopbadcop_detection_analysis.py:28`. Four repeat the same filesystem-target discovery. Extract a query/target service with a true read-only registry capability, preserving ordering, limits, active-analysis filtering, and explicit source-selection policy. Do not make a CLI wrapper the new shared dependency.
- Three profile builders contain the same 45-line `_extract_composition` body: `shared/subject_mask_profile.py:154`, `utils/detection_profile.py:340`, and `utils/keypoint_profile.py:273`. A shared recording-composition extractor can own precedence rules while each profile owns its modality metrics. Function-body identity alone does not prove identical globals or coercion dependencies; audit those before migration.

Across 1,514 tracked Python files, the census examined 24,373 function definitions, including 7,660 spanning at least 30 lines. Normalizing only function name, docstring, formatting, and source positions produced **16 cross-file groups / 41 function occurrences**. The repeated source spans totaled 898 lines, but that is **not a deletion estimate**: globals, callers, nested spans, and contracts can differ. The conservative search misses renamed-variable and more divergent clones. It supports candidate discovery, not a claim that most of the million-line codebase is copy-paste.

In particular, the two 6,164-line refined-mask and subject-shape coordinate modules are not interchangeable copies. They own different raster/point domains, lineage, editability, body-frame derivation, and lifecycle contracts. Extract repeated mechanics; retain their distinct scientific validation.

## How to enforce the good design after consolidation

### Give each invariant one implementation owner and test all maintained routes

For each migrated family, identify the supported CLI, Python, batch, maintenance, and compatibility entry points. Route them through the shared implementation, then remove the old executable copies or reduce them to thin compatibility adapters. A helper existing somewhere in the tree is not adoption.

The existing `analysis_workflows/storage_contract_catalog.py` already centralizes ownership and path declarations. Its tests cross-check stage catalogs and reject unclassified maintained stages. Extend it; do not build a competing catalog. Today some checks prove an imported function's identity or a callable's existence (`:201`, `:215`), not that every execution path invokes it. Runtime verification also still has only the subject-shape-specific hook.

Require behavioral conformance for each maintained adapter: invalid/missing payload, wrong source or recipe, validator exception, failed publication, concurrent ownership transfer, and successful retry. Tests should assert absence of illegal state changes, not only that a helper was called. A valid sufficient supplier can close a dependency branch; this must not become an extra upstream/human-review gate.

### Extend the existing ratchets after migrating callers

The import rules already forbid three migrated utility shims (`pyproject.toml:141`). Apply that pattern incrementally: leave old user-facing commands available but forbid production imports back into the wrappers. Scientific logic, application orchestration, and presentation adapters need different owners; `shared` should not become a new miscellaneous directory.

The open-mode ratchet currently recognizes only literal `zarr.open_group` spelling and keyword presence (`scripts/check_zarr_open_group_modes.py:55`). A reviewer probe found aliases, `zarr.open`, and explicit `use_consolidated=None` unflagged. The check passes with 220 grandfathered bare calls. A separate census found 711 literal `zarr.open_group` and 123 literal `zarr.open` calls; these are not defect counts. Extend alias/API handling and lifecycle declarations, with explicit diagnostic/edit compatibility. Then test stale/missing consolidated generations behaviorally: spelling an option does not prove metadata freshness.

The size ratchet guards four files with 200-line growth budgets (`scripts/file_size_ratchet_baseline.json`; checker `:55`). Moving or copying code into unguarded files can satisfy it without establishing ownership. Keep size as a maintenance signal, not proof of architectural improvement. Clone analysis should be advisory until a specific helper family is intentionally consolidated.

### Define completion of a consolidation by evidence, not net line count

A consolidation is finished only when the canonical owner is identified, behavior/identity fixtures are frozen, maintained callers delegate, old copies are gone or explicitly isolated, bypass tests pass, and required CI is green. Preserve scientific results and current persisted digest bytes for extraction-only work. Any intentional semantic change belongs in a separately reviewed contract/version migration.

A practical sequence is to capture counterexamples first, extract the smallest common mechanics, migrate one caller and compare results, migrate the remaining maintained callers, and then tighten the scoped import/AST rules. The decisive simplification is that the next bug fix lands once and its conformance tests exercise every adapter.

These measures enforce supported software paths, not a security boundary against arbitrary code with filesystem write permission. If the requirement is to prevent any independently written process from mutating authority, filesystem/service ownership must enforce that separately. This review does not propose deploying a new service or changing permissions now.

## Validation, reproduction, and limits

All Python used `scripts/py -B`. Pytest runs were outside the sandbox with cache disabled. Completed focused runs were:

| Run | Result | Scope |
|---|---|---|
| Primary review probes + metadata equivalence + exact in-memory takeover test | **19 passed in 3.01 s** | Includes seven cases characterizing undesirable behavior; nine Zarr consolidated-metadata warnings. |
| Enforcement/catalog/import-boundary tests | **75 passed in 3.93 s** | Existing static/temporary-fixture coverage; does not prove all runtime routes are safe. |
| Array factory and four bounded physical-write/hash tests | **9 passed in 0.21 s** | Existing strong storage machinery, not a new implementation. |

An earlier broader run was interrupted after **17 passes / 88.82 seconds** when the full subject-shape publication file stopped advancing in its real-Zarr fixture, even outside the sandbox. It is incomplete validation, not a passing run. The relevant in-memory takeover cases subsequently passed in the focused run. Deferred command for a normal local terminal:

```text
scripts/py -B -m pytest -q -p no:cacheprovider \
  tests/unit/fisheye/test_subject_shape_coordinate_publication.py
```

Successful commands:

```text
scripts/py -B -m pytest -q -p no:cacheprovider \
  /tmp/palette-consolidation-review-20260904-L0DOux/test_consolidation_probes.py \
  tests/unit/fisheye/test_zarr_metadata_equivalence.py \
  tests/unit/fisheye/test_subject_shape_coordinate_publication.py::test_subject_shape_activation_rollback_preserves_takeover_after_epoch_write

scripts/py -B -m pytest -q -p no:cacheprovider \
  tests/unit/fisheye/test_check_zarr_open_group_modes.py \
  tests/unit/fisheye/test_derived_analysis_storage_contract_catalog.py \
  tests/unit/fisheye/test_analytics_storage_coverage.py \
  tests/unit/fisheye/test_stage_catalog_drift.py \
  tests/unit/fisheye/test_stage_catalog.py \
  tests/unit/fisheye/test_cli_import_boundary.py

scripts/py -B -m pytest -q -p no:cacheprovider \
  tests/unit/fisheye/test_zarr_array_factory.py \
  tests/unit/fisheye/test_subject_mask_core_publication.py::test_physical_row_band_writes_are_bounded_disjoint_and_parallel \
  tests/unit/fisheye/test_subject_mask_core_publication.py::test_physical_row_band_write_failure_propagates \
  tests/unit/fisheye/test_subject_mask_core_publication.py::test_incremental_row_hashes_ignore_execution_batch_boundaries \
  tests/unit/fisheye/test_subject_mask_core_publication.py::test_incremental_row_hashes_reject_gaps_and_dtype_changes
```

The primary clone-census script is temporarily at `/tmp/palette-consolidation-review-20260904-L0DOux/exact_function_clones.py`. No permanent source/test changes were made. Parallel reviewers additionally ran the isolated fake-array, worker, cache, transaction, and parser probes described with their findings. Not every proposed migration was dynamically tested. No full CI, performance benchmark, GPU workload, live registry acceptance, or historical digest migration was performed; no merge-readiness claim is made.

For durable reproduction, the primary seven-case characterization suite follows. Save it as a temporary test file and substitute its path above. These assertions document the reviewed behavior, not desired fixed behavior.

<details>
<summary>Primary characterization probes</summary>

```python
import argparse
import copy
import json

import pytest


@pytest.mark.parametrize('equal_form', [False, True])
def test_managed_output_and_overwrite_accept_abbreviations(tmp_path, monkeypatch, equal_form):
    from fisheye.analysis_workflows.materializers.bout_kinematics import (
        build_bout_kinematics_compute_plan,
    )
    from fisheye.analysis import bout_kinematics

    source = tmp_path / 'source.zarr'
    source.mkdir()
    forwarded = (f'--output-zarr-p={source}', '--overw') if equal_form else (
        '--output-zarr-p', str(source), '--overw',
    )
    plan = build_bout_kinematics_compute_plan(
        source, scratch_root=tmp_path / 'scratch',
        run_name='candidate', writer_arguments=forwarded,
    )
    real_parse = argparse.ArgumentParser.parse_args
    observed = {}

    class ParsedOnly(Exception):
        pass

    def parse_only(parser, args=None, namespace=None):
        observed.update(vars(real_parse(parser, args, namespace)))
        raise ParsedOnly()

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', parse_only)
    with pytest.raises(ParsedOnly):
        bout_kinematics.main([
            str(source), '--output-zarr-path', str(plan.local_zarr),
            '--run-name', plan.run_name, *plan.writer_arguments,
        ])
    assert observed['output_zarr_path'] == source
    assert observed['output_zarr_path'] != plan.local_zarr
    assert observed['overwrite'] is True
    assert not plan.scratch_root.exists()
    assert list(source.iterdir()) == []


@pytest.mark.parametrize('direct_value,inline_value', [(True, 1), (1, 1.0)])
def test_metadata_equivalence_accepts_distinct_json_scalar_types(tmp_path, direct_value, inline_value):
    from fisheye.shared.zarr.metadata_equivalence import validate_direct_consolidated_subtree

    run = tmp_path / 'runs' / 'example'
    run.mkdir(parents=True)
    direct = {'zarr_format': 3, 'node_type': 'group',
              'attributes': {'stage_selector_eligible': direct_value}}
    inline = copy.deepcopy(direct)
    inline['attributes']['stage_selector_eligible'] = inline_value
    root = {'zarr_format': 3, 'node_type': 'group', 'attributes': {},
            'consolidated_metadata': {'kind': 'inline', 'must_understand': False,
                                      'metadata': {'runs/example': inline}}}
    (run / 'zarr.json').write_text(json.dumps(direct))
    (tmp_path / 'zarr.json').write_text(json.dumps(root))
    assert type(direct_value) is not type(inline_value)
    receipt = validate_direct_consolidated_subtree(tmp_path, subtree_path='runs/example')
    assert receipt.node_count == 1


@pytest.mark.parametrize('value,stored,resolved', [
    ('segmentation', 'subject_masks', 'eye_masks'),
    ('subject_masks', 'subject_masks', None),
])
def test_task_normalizers_disagree(value, stored, resolved):
    from fisheye.registry.db import _normalize_task_type as database_normalize
    from fisheye.registry.model_resolution import _normalize_task_type as resolver_normalize

    assert database_normalize(value) == stored
    assert resolver_normalize(value) == resolved


def test_refined_mask_rollback_changes_shared_values_after_owner_takeover():
    from types import SimpleNamespace
    from fisheye.shared import refined_subject_mask_coordinate_publication as publication

    generation = publication.REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR
    policy = publication.REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR
    lease = publication.REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR
    original = {lease: (False, None), policy: (False, None), generation: (True, 0)}
    attempted = {lease: (True, {'owner': 'first'}),
                 policy: (True, 'same_policy'), generation: (True, 1)}
    parent = SimpleNamespace(attrs={lease: {'owner': 'second'},
                                   policy: 'same_policy', generation: 1})
    with pytest.raises(RuntimeError, match='rollback was incomplete'):
        publication._restore_owned_parent_state(
            parent, original, run_name='first_run', owner='first',
            attempt_owned_states=attempted,
        )
    assert parent.attrs[lease] == {'owner': 'second'}
    assert parent.attrs[generation] == 0
    assert policy not in parent.attrs
```

</details>
