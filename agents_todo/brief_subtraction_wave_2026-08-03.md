# Brief: subtraction wave (`agent/palette/subtraction-wave`)

**From:** commander session, 2026-08-03
**Status: READY.** One agent, four tiers, **mandatory CHECKPOINT after each tier.**
**Do NOT push or merge — the commander verifies and merges each checkpoint.**
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Read first:** `docs/diagnostics/provenance_chain_audit_2026-07-24.md` (verdict +
Root Cause 2). Ground rules: local `sun` is ground truth; fresh worktree from
CURRENT `sun`; env `~/miniconda3/envs/palette-py311/bin/python`. Gates before
every checkpoint: import-linter, `check_file_size_ratchet.py`, `git diff --check`,
`py_compile`, full suite.

---

## Purpose

The 2026-07-24 audit prescribed **subtract**. In the ten days since, `src/`
gained **227 files and lost 3** — a 76:1 add-to-delete ratio. This brief is the
counter-pressure. It is a deletion and consolidation worklist, not a feature.

**Every candidate below was verified by AST import census over `src/`, `tests/`,
`scripts/`, `tools/`, `apps/`, plus a string-reference grep over `*.sh`, `*.yaml`,
`*.toml`, `*.json`** (this repo dispatches some bsub jobs by module-name string,
so import-graph analysis alone is not sufficient). Census run at
`agent/palette/derived-analytics-storage-contracts-20260803`, which is a strict
superset of `sun` — so zero callers there implies zero callers on `sun`.

**Re-run the census yourself before deleting anything.** Branches move.

---

## Tier D — DO NOT DELETE (read this first)

These look orphaned and are not. The census caught them; a grep-only sweep would
not have.

| Module | Why it survives |
|---|---|
| `cluster/arena_geometry.py` | 0 Python importers, but referenced by **`scripts/submit_arena_geometry_detection_gate_audit_bsub.sh`**. String dispatch. |
| `diagnostics/check_provenance_consistency.py` | Imported by `utils/check_recording_steps.py`. **Load-bearing.** (It has a real defect — `main()` returns `None`, so it can never fail a build — but that is a *fix*, not a delete.) |
| `diagnostics/check_eye_mask_lineage.py` | Imported by `utils/prune_zarr_runs.py`. Untangle before removing. |
| Everything in `utils/` and `diagnostics/` not named below | 45 of 45 new `utils/` files and 26 of 27 new `diagnostics/` files have a `main()`/argparse entry. They are human-invoked `python -m` tools. Being unreachable from a pipeline entry point is **by design**, not rot. Do not mass-delete these. |

---

## Tier A — SAFE DELETE (verified zero production callers)

**~5,570 lines including paired tests.** Delete the module and its test together.

| Module | LOC | src importers | Why |
|---|---:|---:|---|
| `shared/tabular_deltas.py` (+ test, 122) | 391 | **0** | Append-only edit log. Well designed, never wired. `docs/tabular_delta_compaction_contract.md:206-210` admits review writers were never routed through it. Audit's flagship example of built-but-unwired. |
| `utils/audit_analysis_staleness.py` (+ test, 369) | 711 | 1 (only `inspect_run_lineage_graph`) | **Structurally cannot return `stale`** — compares `source_fingerprints` against `lineage_hash`, and no pipeline stage writes either. Delete as a pair with the next row. |
| `utils/inspect_run_lineage_graph.py` (+ test, 363) | 579 | **0** | Sole consumer of the above; itself has zero importers. The pair is a closed orphan loop. |
| `utils/resolve_latest_registered_model.py` (+ test, 107) | 148 | **0** | Correct digest-drift resolver, only ever self-called. Its logic already exists in production form at `utils/run_detection_local_publish.py:337-344`. |
| `analysis_workflows/storage_benchmark_catalog.py` | 307 | **0** | Worse than dead: `benchmark_coverage_complete()` requires four fields that neither construction path populates, so it **can never return True**, and it validates `evidence_receipt_sha256` for hex shape without ever hashing the file. Validation theater. |
| `analysis_workflows/materializers/runtime_telemetry.py` | 203 | **0** | Orphan from the storage-contracts push. |
| `shared/zarr/storage_report.py` | 97 | **0** | Orphan. |
| `cluster/crop_snapshot.py` | 159 | **0** | Orphan; also hardcodes `selector_eligible: False, registry_updated: False`. |
| `cluster/detection_snapshots.py` | 210 | **0** | Orphan. |
| `diagnostics/check_full_provenance.py` | 207 | **0** | Effectively dead: hardcodes `keypoints_runs` (not `refined_keypoints_runs`) and the deprecated `eye_masks_runs`; its assertions flag a historically-correct pinned run as an error. |
| `diagnostics/check_provenance_capture.py` (+ test, 451) | 548 | **0** | Verifies *presence*, not correctness — `_has_inputs` passes if any attr starts with `source_`, so a run pointing at a nonexistent upstream passes. False assurance with a test suite. |
| `shared/keypoint_stale.py` | 189 | **0** | `mark_downstream_eye_mask_runs_stale` targets the deprecated eye-mask stage and has no callers anywhere. Eye masks are legacy — subject masks subsume them. |
| `diagnostics/preview_eye_mask_background_subtraction.py` (+ test, 95) | 311 | **0** | Eye-mask legacy. |

**Note on eye masks:** the surface is already nearly gone — only 2 `.py` modules
remain (the 42-file count in earlier sweeps was `.pyc` and docs). This is close
to finished; do not go hunting for more.

**Procedure per deletion:** delete module + paired test, run the full suite, then
`grep -rn "<module_basename>" src/ tests/ scripts/ tools/ apps/ configs/ docs/`
and fix any doc references left dangling. Docs referencing a deleted module are
part of the deletion, not a follow-up.

---

## Tier B — CONSOLIDATE (N implementations → 1)

Do not delete these; collapse them. Each is a duplication the audit counted and
that has since grown.

**B1 — array digests: 8 private `_array_digest` implementations → 1.**
`analysis/subject_shape_storage.py:381` is the only one done correctly (canonical
JSON header + declared `ARRAY_PAYLOAD_CANONICALIZATION` tag + `\x00` separator).
Promote it to a shared module and delete the other seven. Start with
`analysis_workflows/materializers/registered_detection_gate.py:329-335`, which
hashes `values.view(np.uint8)` with **no dtype, no shape, no separator** — an
`int64` and a `uint64` array with identical bytes hash identically, and
`array.shape[0]` raises on a 0-d array. That one is a correctness bug, not just
duplication.

**B2 — `sha256_c_contiguous_bytes_v1` declared verbatim in 9 modules.**
`detection/clipped_native_binding.py`, `shared/zarr/{refined_keypoint_manifest,
subject_mask_validation_receipt, subject_mask_quality_manifest, keypoint_manifest,
body_frame_manifest, canonical_detection_manifest, crop_manifest,
keypoint_quality_manifest}.py`. One constant, one owner, nine imports.

**B3 — `_payload_sha256` copy-pasted in 4 modules** with no schema id or version
pin in the output. Contrast `shared/rowset_fingerprint.py:19-24`, which exports
and writes all three. Collapse to that pattern.

**B4 — 38 private `_canonical_json` definitions.** `shared/zarr/manifest_digest.py`
already has **109 importers** — it is the winning implementation and the one real
de-duplication this repo has achieved. Migrate the 38 stragglers onto it.

**B5 — review front-ends: 4 surfaces, 1 audit log.** Only `labeling/web.py`,
`labeling/web_admin_api.py`, and `labeling/assignment_store.py` call
`record_event`. `tune/keypoint_review_web.py`, `tune/detect_review_web.py`, and
the **new** `cluster/arena_geometry_review.py` write reviewer decisions with
**zero** audit events. Either route them through `record_event` or retire them.
Do not add a fifth.

**B6 — 4 registry finalizers** (`analysis_workflows/registry_finalize.py`,
`cluster/clipped_inference_registry_finalize.py`,
`cluster/keypoints/registry_finalize.py`,
`cluster/whole_recording_analysis_registry_finalize.py`), four schema ids, no
shared type. `analysis_workflows/registry_finalize.py` is the best of the four
(re-hashes receipts, raises on evidence mismatch, post-write readback). Make it
the base; do not write a fifth. **Lower priority than B1-B4 — this one is a
refactor with real risk, not a mechanical collapse.**

---

## Tier C — DECIDE, DO NOT DELETE

**C1 — the `shared/zarr/` storage stack: 87 new modules, zero CLI entries.**
Roughly half have no production caller; the rest are reached only by benchmarks
and canaries. Meanwhile the production writers are untouched
(`refinement/refine_detect.py`, `refine_keypoints.py`, `tracking/crop_extraction.py`
have **zero diff** since `b14dc8e3`).

This is **declared** staged work —
`docs/canonical_detection_storage_implementation_checklist.md:3-5` says adoption
is blocked and lists 92 open items — which makes it a legitimate fork in the
road, not false assurance. But it is a fork that gets more expensive every week.

**This is a commander decision, not an agent decision.** Two options: finish the
migration and retire the old writers, or freeze it and stamp every module with a
`status: staged` header. Do not delete. Do not wire it opportunistically.

**C2 — consolidated metadata.** Unresolved and contradictory. The audit ruled
*bypass, never reconsolidate live archives* (measured 8.6× slower; three
structural reasons in the Fix section). Since then, 4 new modules **raise** when
the consolidated snapshot is absent (`shared/zarr/refined_detection_snapshot.py:174`,
`subject_mask_bundle_publication.py:498`, `subject_mask_cache_publication.py:534`,
`subject_mask_core_publication.py:166`), wired into production validate/finalize
stages, and 4 contract docs now specify reconsolidation as required.

**Do not touch either side in this brief.** Flag it in the checkpoint. It needs a
written decision record before any code moves, or an agent will "fix" it in
whichever direction it read first.

---

## Tier E — non-code

`docs/diagnostics/` holds **~7.7 MB of raw JSON census dumps**:
`zarr_array_schema_census.json` (3.3 MB), `zarr_production_writer_census.json`
(3.2 MB), `zarr_detection_schema_inventory.json` (0.5 MB), plus the
`registry_stale_rows_*` dry-run/execute pairs (~1.2 MB combined).

These are point-in-time machine output committed to a docs tree. They are already
stale and cannot be reviewed. Delete them, or move generation to a script that
writes outside the repo. If any is load-bearing evidence for a decision, keep the
*summary* and cite the regenerating command.

---

## Anti-goals (state these back in the checkpoint so they are on record)

1. **Do not wire the unwired modules.** The obvious reading of "116 unreachable
   modules" is "connect them." That is backwards and would be the worst available
   outcome. Default action is delete or explicitly mark staged.
2. **Do not touch the registry destructive paths in this brief.**
   `registry/dedupe.py`, `prune_stale_datasets.py`, `reconcile_missing_datasets`
   are byte-identical to the audited baseline and **the tests codify the bug** —
   `tests/unit/fisheye/test_registry_dedupe.py:250-280` asserts that a
   `detect='ok'` row is deleted in favour of `detect='missing'`. An agent working
   to keep tests green will preserve the destructive behaviour and conclude it is
   intentional. Separate brief, tests first.
3. **Do not add a new module in this brief.** Not a helper, not a catalog, not a
   shared base class — except where Tier B explicitly requires promoting an
   existing implementation to a shared location. Net line count must go **down**
   at every checkpoint. Report the delta.
