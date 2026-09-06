# Second opinion: review wave, live evidence, and workload architecture

> Follow-up added 2026-09-06: this report remains the correction owner for the
> unsafe original selector/authority/receipt proposals and overstated historical
> measurements. The [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation) records newer code/CI dispositions, including six
> runtime verifiers and execution-time reuse validation at main `1bf9d919`.
> Historical live measurements below were not rerun or rewritten; neither this
> report nor its follow-up is a separate implementation/status queue.

Date: 2026-09-04. Code reviewed: `73bee0d5194c662e3b7e535be3de92db7ef53f63`, branch `agent/palette/refined-assignment-rebinding-gaze-20260831`.

Scope: the handoff, its seven diagnostics documents, the wave-1 implementation brief, the storage map, the existing authority work queue and acceptance checklist, relevant prior-art/benchmark documents, current code, and read-only live measurements. Source references below are relative to `src/fisheye/`, unless prefixed otherwise. Line anchors refer to this commit.

This is a review, not an implementation plan or a new fix queue. During the original review, no source, registry, store, existing document, or `/groups` file was changed. The deliverable was this new document; temporary probes and their outputs are under `/tmp/palette-second-opinion-20260904-LlmmLz/`. The user subsequently requested the documentation-only design clarification below. No commits, pushes, installations, deployments, or production activations were performed. The pre-existing untracked `.claude/` directory was left alone.

## Summary

The wave correctly identifies weak boundaries between completion, selection, provenance verification, and registry projection. Its handoff overstates both the absence of existing machinery and the sufficiency of fixing one gate. **Do not implement the brief and subsequent selector migration unchanged.**

- The generic completion-status writer does swallow validation failures. However, changing this writer alone cannot prevent earlier publication, bypassing writers, stale projections, or incorrect consumer selection.
- **S-5's blanket `latest` backfill is unsafe.** Every child of all 84 tracking families without `latest` is explicitly selector-ineligible; none has an `ok` registry run. The sealed tracking loader takes an exact run path and does not require `latest`.
- The selector table reproduces numerically, but its interpretation does not. The 32 selector-pair mismatches comprise **28 RedScare and four GoodCopBadCop** archives. Run names do not establish which was reviewed. Three registry `ok` projections select explicitly ineligible runs.
- Membership reproduces as **five recording/model overlaps on four Sleepyfish cameras**. That is not proof that every downstream frame was a training example, that these were held-out evaluations, or that all other recordings are clean. Five detection and seven mask entries remain unresolved. The current pose training artifact also contains rows from each camera in both persisted train and validation splits.
- The 19 relocated Zarr trees pass an independent per-file comparison: **11,542 files / 30,147,352,423 bytes, zero differences**. All 12 retained model files match their registered hashes. But there are **four matching backups, not five**, many `/nvme1` references remain, and four active frame-index references point to missing files.
- Sealed tracking consumers, a provider track-motion publisher, digest-bound reviewed-keypoint compaction, persisted splits, pose seed forwarding, and export-manifest verification already exist. Several proposed additions duplicate these surfaces or contradict the repository's use-scoped authority rules.
- For efficient, repeatable workloads, prioritize typed admission, immutable input identity, recoverable submission/publication, bounded memory, and explicit registry ownership. A generic envelope rewrite, engine migration, or Crimson integration is not the first dependency.

### Follow-up: rig defaults and distinct completion claims

After the independent code-first assessment, the user confirmed that the scientific values embedded in workflow command builders are intentional defaults for their camera configuration and rig. Preserve those values and their applicability; the recommendation is to represent them as an explicit validated, named/versioned recipe, not to remove defaults or imply they are scientifically wrong or unrecorded. Record effective values and permitted overrides, keep scientific configuration separate from execution resources, and bind reuse to the applicable recipe and input identities.

The user endorsed separating **computation complete** (validated numerical products), **presentation complete** (validated associated plots/specifications), and **deliverable complete** (all products required by the selected workflow are ready). An independent plotting failure should remain visible and keep a plot-required deliverable incomplete without invalidating reusable numerical products. This is a proposed product-state distinction, not an implemented lifecycle change or a waiver of scientific authority, human-review contracts, or required CI gates. See section 6 of the [independent design assessment](independent_codebase_design_assessment_2026-09-04.md) for the precise scope and implementation caveats.

The subsequent [parallel shared-helper consolidation review](shared_helper_consolidation_review_2026-09-04.md) records concrete duplication, observed behavioral drift, and safe helper-adoption/enforcement boundaries. It supplements this historical audit rather than replacing its measured findings or opening another fix queue.

## 1. Verified premises

“Supported” is deliberately scoped to the cited path. It does not mean every producer or consumer uses it.

| Claim from the wave | Verdict | Current evidence and correction |
|---|---|---|
| Review commits change docs and two scripts, not `src/` | Supported | `git diff --stat ea03eb76^..73bee0d5`: 15 files, documentation plus the two diagnostic scripts. This says nothing about historical live-store mutations, which are separately documented. |
| `emit_stage_completion` catches validation errors | Supported | `registry/stage_complete.py:374–488`: validation is inside `try`; `except Exception` returns `False`. Logging requires a console. Registry-disabled or unresolved-registry paths return before validation. |
| `mark_run_complete` promotes before full content validation | Conditional | `shared/zarr_run_completion.py:258–306` checks provenance structure, then marks complete. It changes `latest`/`latest_complete` only with a parent, run name, and eligible run. It is not a universal content verifier. |
| Every cited keypoint publication is already `latest` before validation | False | `shared/zarr/keypoint_publication.py:541` supplies **no parent** to `mark_run_complete`; its manifest and publication validation follow at `:553–573`. The cited snapshot path does not thereby activate `latest`. |
| Refine-detect's strict source gate defaults off | Supported | `refinement/refine_detect.py:1355`, `:1485–1505`: default `require_active_canonical_source=False`; an omitted run uses the family's `latest`. Supplying digest expectations without the strict flag is refused. |
| Legacy tracking uses source matching and sorted fallback | Supported, incomplete description | `tracking/single_subject_per_arena.py:531–613`: source-match filtering, latest preference, then `matches[-1]`. The loader also validates keyed alignment/uniqueness, not merely run-name equality. |
| Sealed tracking verification is never called; legacy tracking is the only motion writer | False | Calls exist in `analysis_workflows/materializers/single_subject_tracking.py:405`, `analysis_workflows/provider_spatial_track_source.py:257`, and `utils/materialize_provider_behavior_chain.py:235`. The latter publishes provider track motion at `:263–315`. |
| Chaser bout response does not bind every parent by digest | Supported for the cited block | `analysis/chaser_bout_response.py:1249–1257` records distance and swim-bout names/paths, plus an egocentric-component manifest digest. The block is neither “no hashes” nor a complete digest-bound parent chain. |
| Ordinary keypoint export can select `latest` | Supported | `utils/export_keypoint_training_zarr.py:1298–1310` falls back to `keypoints_runs.latest`. Do not generalize this one path to every reviewed-artifact exporter. |
| Model resolution has no target-training-membership exclusion | Supported | `registry/model_resolution.py:15`, `:358–407`, `:411–493`: feature-weighted matching, without an evaluation-membership gate. Inclusion of a target does **not** guarantee the highest score: scores average feature agreement across source rows. |
| Generic approval binds the approved payload digest | False | `cli/palette.py:1521–1526` records approver/time/git/note, not a subject content digest. The generic authority primitive is not scientific-review evidence. |
| Review acceptance is never digest-bound anywhere | False | `training/training_review_compaction_publication.py:499–517`, `:616–640` validate/bind a review receipt, compaction manifest, and payload inventories. The training review itself acknowledges detection decision digests and compose-v3 keypoint receipts. |
| `config_hash` is never recomputed | False as a universal claim | The generic completion gate checks required fields, but `analysis/track_kinematics.py:5862` compares it with `sha256_payload(parameters)`. Domain-specific stronger validators must not be erased by the census summary. |
| JSON digest grammars differ | Supported | `shared/run_provenance.py:37–69` differs from `shared/zarr/manifest_digest.py`. It also converts sets to unsorted lists and unknown objects to strings. Those are additional determinism/type-identity risks. |
| Decoded Merkle identity survives arbitrary rechunking | False | `shared/zarr_payload_receipt.py:144–277` includes logical leaf bounds and leaf count. Repartitioning equal values into different leaves changes the root; see §3. |
| Pose does not pass Ultralytics a seed | False | `training/train_pose.py:1891` sets `declared_training_params["seed"]`; the effective training arguments are checked later. This does not establish full deterministic execution. |
| Training export's computed inventory is simply discarded | Too broad | `utils/export_keypoint_training_zarr.py:2981`, `:3016` uses the full inventory to verify the copied tree before publication. Not persisting the inventory is a different, valid durability criticism. |
| The cited Dask mask path has concurrent overlapping Zarr writers | Not supported | `refinement/refine_subject_masks.py:342` workers read/compute; `:497` gathers; `:514–527` writes serially. Its evident risk is unbounded gathered results, not demonstrated concurrent chunk corruption. |
| Five backups; no `/nvme1` references anywhere | False | Four matching backup files were found. Read-only all-column SQLite scans and path checks find historical references and four missing active frame-index locators (§5). |

## 2. Ranked disagreements

Ranking reflects the risk of acting on the review, followed by runtime impact. These are findings, not new tracking IDs.

### 1. High — the recommended selector migration can promote the wrong artifacts

The absence of `latest` is not evidence of a missing migration. The store contains complete, deliberately selector-ineligible candidates, snapshots, and provider artifacts. Their exact-path consumers can be valid without a family selector. S-5's required “latest unset = zero” invariant conflicts with that lifecycle.

The registry is not a safe unvalidated backfill oracle either: the fresh scan finds `ok` rows pointing at explicitly ineligible detection, crop, and refined-detection runs (§4). `registry/maintenance.py:4113–4149` accepts complete direct-latest or sorted-fallback runs without consistently enforcing eligibility. A valid fix must distinguish compatibility discovery, exact diagnostic consumption, canonical publication, and accepted selection for a declared use.

Do not migrate pointers before determining the consumer contract and verifying the selected artifact. In particular, `analysis_workflows/tracking_source_handle.py:459` explicitly loads a run path **without selector lookup**. The proposed tracking backfill is not a prerequisite of this API.

### 2. High — “completion gate is the single load-bearing defect” is not a sufficient diagnosis

The fail-open status writer is real. But the writer checks an already-complete, eligible run; some callers have already exposed selectors. Package A intentionally leaves ordering, registry-disabled bypass, direct writers, and reconcile unchanged. Default-off strictness adds observability, not enforcement closure.

Independent defects remain even with perfect exception propagation: an incorrect source can be selected; an unverified acceptance string can be copied; a parent can mutate after validation; a consolidated generation can disagree with direct attrs; a scheduler can rerun an ambiguously submitted task; a registry query can reconstruct an incompatible profile. None is repaired by making this one helper raise.

The stronger diagnosis is **inconsistent admission and publication boundaries across supported producer/consumer paths**. Completion, content/schema validation, use-scoped acceptance, authority activation, and registry projection are distinct transitions. Existing `ADM-001–003`, `ACC-001–003`, `REP-001`, and `TEST-001` already express much of this. A coherent publication transition is needed, not another globally optional check alone.

### 3. High — the handoff erases existing strict paths and would create parallel infrastructure

The provider behavior chain composes position and body-frame authorities, binds tracking, and publishes `analysis/track_kinematics_runs/provider/<run>` (`utils/materialize_provider_behavior_chain.py:263–315`). It uses verified handles. It is not proof that all ordinary motion workflows have migrated, but it disproves “only legacy writer” and “verifier never called.” The actual task is to converge maintained entry points on the existing machinery under `RES-TRACK-001`.

Likewise, reviewed-keypoint compaction carries a review receipt and inventory binding. Failure-review/task-generation surfaces exist in `tune/keypoint_failure_review.py`, `labeling/task_generation.py`, and `labeling/work_queue.py`. They do not establish a universal closed learning loop, but “nothing routes corrections” is incorrect. The real remaining audit is whether a particular exported row and trained model retain and validate the exact correction lineage.

Do not require a second upstream keypoint authority or human approval when a validated body-frame supplier already satisfies the consumer. Do not replace subject-mask bundle authority with a generic segmentation authority. Those changes would contradict current `AGENTS.md` and the acceptance checklist.

### 4. High — registry/store cleanup is materially less complete than the handoff says

The successful copy and model checks do not erase stale locators, path-derived identity hazards, or experiment history. Four active Sleepyfish dataset rows retain nonexistent `/nvme1/.../recording_frame_index.parquet` paths. The three path-derived merged-dataset IDs still need relocation-safe re-registration. Deleting failed and set-less model rows removes evidence of unsuccessful or unresolved experiments; “all remaining runs succeeded” is not a quality metric.

Prefer unavailable/retired/tombstoned experiment records with immutable identity over deleting provenance to make the catalog clean. Backups can recover registry history; they do not alone recover deleted weight directories. Do not globally replace old paths inside historical invocation or receipt payloads: those may be truthful provenance and digest-covered bytes.

### 5. Medium-high — the contamination statement exceeds the measurement

The five hits are reproducible recording-level overlap among resolvable candidates. The scan does not reconstruct exact training examples, evaluation exclusions, acquisition-frame mappings, or downstream source manifests. A full-recording inference can include many frames never labeled. Downstream `ok` status does not prove that its current motion/bout run consumes the exact matched inference run.

The present pose artifact provides stronger evidence of a **non-recording-held-out split**, with each camera in both train and validation (§4). That still does not prove historical model consumption of these exact bytes without a bound dataset-content identity. Keep the distinction among example overlap, recording/session leakage, and ordinary in-distribution inference.

Membership should be a use-aware eligibility/annotation result, including `unknown`, not a universal ban on inferring on training recordings. Evaluation claims need held-out identity and grouping rules; ordinary correction or behavioral workflows can legitimately process previously labeled recordings. The review provides no measured magnitude or direction of behavioral bias.

### 6. Medium — a universal envelope and registry-only authority would conflict with existing contracts

Reuse canonical serialization, typed input references, inventories, and verification mechanics. Do not require every scientific manifest to include all execution, environment, and human-review fields. A content-equivalence identity may intentionally remain stable across producer hosts, times, and physical layouts; producer/review attestations can bind that identity without changing it.

The acceptance checklist expressly preserves product-specific manifests and separates canonical validity from accepted selection. `analytics_exports/publication.py:135` already provides validated immutable-generation commit machinery. The proposed rule “registry is the only authoritative pointer” is a major cross-system authority migration, not a cleanup: it reverses current reconciliation and must specify ownership, atomic visibility, recovery, offline reads, and generation binding first.

Similarly, “immutable at `mark_run_complete`” conflates a technical marker with sealing. Review/edit surfaces intentionally have their own mutable lifecycle. Seal payload generations before publication; keep mutable work products visibly separate, rather than declaring current edits illegal by changing terminology.

### 7. Medium — hash recommendations need domain and immutability semantics, not just another helper

Multiple serialization grammars are hazardous when mixed without a version, but different labeled digest domains are legitimate. Raw file SHA, logical-array identity, and a canonical JSON receipt are not interchangeable. Switching every existing helper in place would invalidate persisted manifests and caches.

Hardlink identity proves that two paths refer to the same current inode. It does not prove equality to an earlier sealed digest or prevent later shared-inode mutation. Rehashing at one instant closes the stale-receipt comparison, not the subsequent mutation window. H1 needs immutable/protected backing or copy-on-write/new-generation behavior as well as digest verification.

The proposed figure recipe also contains a circular dependency: the receipt hashes the image, while the image embeds the receipt hash. Embed a provenance-core identity that excludes image bytes, or retain an external receipt that hashes the final rendered image. Do not implement a hash fixed point.

### 8. Medium — the scheduler review misses recovery and overstates what an engine fixes

`cluster/lsf/submission.py:95–142` resets the submission snapshot, submits with `bsub`, parses the returned ID, then persists it. A crash after scheduler acceptance but before persistence leaves an unknown submission. Blind retry can duplicate work. A commit check does not fix this. Use a stable attempt identity, durable submission intent/journal, scheduler reconciliation, and output ownership/fencing before automatic retries.

`cluster/lsf/runtime.py:175–302` records lifecycle and checks expected-output existence, not necessarily generation/content identity. Existing output files can satisfy existence checks. SIGKILL/OOM cannot be repaired by the dying wrapper; external scheduler reconciliation is needed. Separate successful scientific publication from later cleanup/operational failure so retries do not overwrite valid outputs.

LSF `bsub -r` addresses host/system-failure reruns, not arbitrary unsuccessful job exits; exit-code requeue policy is separate. Nextflow adds useful execution caching, but requires cache identity and immutable input semantics; a mutable Zarr directory is not made safe simply by wrapping its CLI. A bounded engine experiment is reasonable after those semantics are declared. [IBM rerunnable jobs](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-r-1), [IBM automatic job requeue](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=queuing-configuring-automatic-job-re), [Nextflow cache and resume](https://docs.seqera.io/nextflow/cache-and-resume).

## 3. Digest counterexample and narrower positive evidence

An in-memory probe of the receipt construction used array path `a`, shape `[4]`, dtype `uint8`, and identical values `[1, 2, 3, 4]`, once as leaf `[0,4]` and once as leaves `[0,2]` and `[2,4]`. The decoded roots were respectively:

```text
87cec596cd8c069f1d0bcb1839e614305631786bffcd686e7e6ee79b45fbd103
eb32e56c4a9074cce4a0f88ba774158908457796e5bd84df1f292f34f2edad52
```

This is a counterexample to arbitrary repartition-invariance, not a rejection of the receipt. Recompression with unchanged logical leaves can preserve decoded identity. A layout-independent root needs a fixed logical partition independent of physical shards, with its own versioned domain.

`verify_payload_integrity_receipt` checks the receipt and physical payload/metadata by default; do not describe every verification as independently decoding and recomputing every logical leaf. Conversely, do not describe every publisher as presence-only: the keypoint publisher validates arrays/declarations; reviewed compaction verifies inventories; analytics immutable publication has stronger gates. The census needs call-path and argument-level evidence, not counts of function names.

Additional concrete corrections:

- Pose seed forwarding is already implemented; remaining determinism work must distinguish requested parameters, effective Ultralytics parameters, device/runtime settings, and realized split identity.
- `reporting/report_registry.py:118`, `:237–238`, `:367` already bind/check the export manifest's content hash. A missing renderer-code identity is a different task from adding the parent manifest digest again.
- `utils/compose_task_keypoint_training_manifest.py` propagates `reviewed_artifact_receipt_digest`; the inspected receipt-reading path does not itself recompute that receipt before copying it. Audit that boundary rather than claiming the receipt does not exist.
- `training/training_run_shared.py:72–78` checks an expected manifest hash **before** its registry exception handler at `:121`. The registry write is fail-open; not every validation in the function is swallowed.
- RFC 8785 is stricter than Python sorted compact JSON. Adopt explicit supported types and versioned canonicalization; do not call a short untested reimplementation JCS-compliant. [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785).

## 4. Re-measured selectors and membership

### 4.1 The numeric selector table holds

The landed script opened 296/296 currently active source-recording archives with `use_consolidated=False`, with no archive errors. The old 408 count precedes S-1's 112 status changes. These are **archive/family counts**, not counts of production consumers actually taking each fallback.

| Family | Present | Multi | Latest set | Unset | Latest ≠ sorted | Latest ≠ complete | Pending | Registry ok | Registry ≠ latest | Registry = sorted without latest | Registry selected by sort |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| detect_runs | 246 | 122 | 245 | 1 | 120 | 0 | 0 | 214 | 0 | 1 | 1 |
| refined_detect_runs | 225 | 97 | 101 | 124 | 1 | 0 | 0 | 97 | 3 | 1 | 1 |
| crop_runs | 248 | 113 | 57 | 191 | 0 | 0 | 0 | 100 | 3 | 63 | 40 |
| keypoints_runs | 205 | 129 | 120 | 85 | 28 | 0 | 0 | 116 | 0 | 0 | 0 |
| refined_keypoints_runs | 193 | 70 | 104 | 89 | 29 | 32 | 0 | 102 | 4 | 0 | 0 |
| subject_mask_runs | 173 | 96 | 85 | 88 | 4 | 0 | 0 | 85 | 1 | 0 | 0 |
| refined_subject_masks_runs | 176 | 93 | 73 | 103 | 5 | 0 | 16 | 71 | 0 | 0 | 0 |
| tracking_runs | 153 | 40 | 69 | 84 | 4 | 0 | 0 | 65 | 0 | 0 | 0 |

There are no dangling registry targets in this table. The broader 46 sorted-selected registry rows also reproduce: crop 40, detection 1, refined detection 1, stimulus 4. The stimulus family is outside this eight-family table. The 40 and 63 crop measures overlap; they must not be added as independent affected populations.

### 4.2 Lifecycle changes the interpretation

A separate direct-`zarr.json` pass inspected every child group, used the actual completion helper with legacy-acceptance recording disabled, and checked explicit eligibility.

| Family | Latest unset | All children explicitly ineligible | At least one complete eligible child | Unset family has registry ok run |
|---|---:|---:|---:|---:|
| detect_runs | 1 | 1 | 0 | 1 |
| refined_detect_runs | 124 | 88 | 36 | 1 |
| crop_runs | 191 | 124 | 67 | 68 |
| keypoints_runs | 85 | 85 | 0 | 0 |
| refined_keypoints_runs | 89 | 89 | 0 | 0 |
| subject_mask_runs | 88 | 88 | 0 | 0 |
| refined_subject_masks_runs | 103 | 103 | 0 | 0 |
| tracking_runs | 84 | 84 | 0 | 0 |

The third column is explicit `stage_selector_eligible=false`, not an inference from names such as “canary.” Completion/eligibility here does not itself establish a full canonical contract or scientific acceptance. Fifteen of the unset refined-mask families are training archives; the remaining unset families in this pass are analysis archives.

Three current `ok` projections point to explicitly ineligible runs:

| Recording / archive use | Family | Registry run | Projection selector |
|---|---|---|---|
| Sleepyfish May 5 cam2010094 / analysis | detect | `detect_native_sleepyfish_cam2010094_native_canary_20260728_v003_sleepyfish_cam2010094` | sorted_fallback |
| 2026-08-10T17-20-55Z arena 1 goodbatbadbat / analysis | crop | `crop_goodbatbadbat_geometry_production_400fce8f` | sorted_fallback |
| 2026-08-10T17-20-55Z arena 2 goodbatbadbat / analysis | refined detect | `refined_detect_geometry_review_eae1d724a7018802ce30_working` | source_match_sorted_fallback |

These establish unsafe projection semantics, not proof that every downstream consumer executes those runs.

The 32 refined-keypoint selector-pair mismatches are **28 RedScare and four GoodCopBadCop training archives**. For the 28 RedScare `latest` runs, five have `keypoint_review_status.state=approved`, 23 have `needs_review`; none of the 28 `latest_complete` targets has that review-status field. The four GoodCopBadCop pairs also lack that field on both targets. Absence of this field is not proof of absence of all possible review evidence, but directly defeats “the reviewed run, presumably” based on a suffix.

A direct/consolidated selector comparison found one mismatch among 108 refined-mask parents represented in consolidated metadata: `2026-07-01T14-32-13Z_arena_1_DefaultScreen_training.zarr` has direct `latest=refined_subject_masks_danionella_strong_single_inframe_review_20260804`, absent in the consolidated parent. Determine its lifecycle before deciding whether to republish. This was a selector comparison, not a full metadata-generation audit.

### 4.3 Scanner limitations that matter before using it as a gate

`scripts/sweep_run_selectors.py` is useful reconnaissance, but:

- Its `complete` field checks a run-level `palette_completion_epoch` or generic `status`, not the real completion-status/parent-epoch contract. The printed selector table does not use this field, so that bug does not invalidate the reproduced counts.
- It probes selector targets, sorted-last, and the registry target, not every candidate run's model attrs. It does not validate selector eligibility, bundle authority, receipts, intended use, or publication generations.
- Mismatched names prove disagreement, not which resolver is correct or which execution path was used. Candidate discovery is not admission.
- It catches archive errors and still returns success. Before scheduling as an error-budget gate, define coverage, unknown/error handling, exit behavior, and lifecycle-specific tolerances. “Zero unset latest” is not an appropriate universal SLO.
- Its registry reads and live-store traversal do not form one frozen cross-system snapshot. Repeated observations need registry snapshot identity and per-artifact generation binding, not only a date.

### 4.4 Membership holds with a narrower conclusion

The same five hits reproduce; all are registry-selected in the script's comparison:

| Recording suffix under `sleepyfish_2026_05_05_17_45_30_` | Matched inference/model |
|---|---|
| cam2010093 | keypoints / pose retry2 |
| cam2010094 | keypoints / pose retry2 |
| cam2010095 | keypoints / pose retry2; detection / detect v002 |
| cam2010096 | keypoints / pose retry2 |

Full model IDs are `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2` and `detect_all_available_detect_training_v002_yolo11n_trt_20260513_tmux`. The registry also reports `tracks`, `track_kinematics`, and `swim_bouts` as `ok` for all four analysis datasets; their exact transitive source identities were not reconstructed in this review.

Resolution coverage is detection 241 resolved / 5 unresolved; keypoints 205 / 0; subject masks 166 / 7. The three other identified model populations reproduce as detection v004 127 entries / no detected overlap, cedar-shadow v007 113 / none, and subject-mask union 166 / none. Membership expansion finds 59 training recordings for pose, 60 for each all-available detection model, 52 for cedar-shadow, and 15 for the mask model. These are script-defined populations, not a proof of universal production use or cleanliness. Eye-mask membership was not checked.

I also read only the small `source_index` and `splits` arrays of the relocated pose merged artifact, explicitly unconsolidated. It declares `zarr_purpose=training`, 12,292 source rows, 9,834 train indices, and 2,458 validation indices:

| Source camera | Artifact rows | Train entries | Validation entries |
|---|---:|---:|---:|
| cam2010093 | 236 | 188 | 48 |
| cam2010094 | 235 | 193 | 42 |
| cam2010095 | 237 | 189 | 48 |
| cam2010096 | 238 | 177 | 61 |

Artifact: `/groups/johnson/johnsonlab/jeremy/training/datasets/pose_all_registry_reviewed_v2_keypoints_20260520_v001/zarr/pose_all_registry_reviewed_v2_keypoints_20260520_v001_merged.zarr`.

This demonstrates that stored splits exist and are not held out by camera/recording for this session. It does not prove exact historical trainer input bytes or that any specified downstream evaluation used them. A future membership projection should bind model artifact identity, immutable dataset/split identity, source recording and acquisition-frame identities, and session/grouping policy; recursively expand merges, retain unknowns, and cross-check model ID/path/hash instead of taking the first plausible nested identity. A `set_id` alone is insufficient: one retained set-backed pose model has an empty set membership list.

## 5. Registry and storage change audit: S-1, S-2b, S-9

All SQLite acceptance checks used `scripts/py` and Palette's `registry_integrity` module, not the system SQLite CLI. Runtime: Python 3.11.14, SQLite 3.52.0, executable `/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311/bin/python`. The live registry and each of the four backups pass complete `integrity_check` and `foreign_key_check` with zero issues.

Actual backup directory: `/groups/johnson/johnsonlab/jeremy/registries/backups/`, not a backup directory in this checkout. Matching files:

| Backup filename | State visible before the next change |
|---|---|
| palette_registry_20260903_024848.sqlite | 16 training runs/models; before 112 dataset status changes |
| palette_registry_20260903_025814.sqlite | Those 112 statuses changed; before merged-path relocation |
| palette_registry_20260903_030504.sqlite | 17 dataset locators/path hashes and 13 training manifest locators relocated |
| palette_registry_20260903_031808.sqlite | 14 runs/models after failed-pose deletion, model/config relocation; before deleting two set-less runs |

Live observed file SHA-256: `bc38e56a9350acb424321bdaa22bc0248e2f17e7d2e57f90a40e2062ae3203e0`. This identifies the file observed during the audit, not an atomic hash of the entire registry-plus-stores state. Read-only comparison queries used explicit read transactions. All five database files report `user_version=73` and 63 non-internal tables; integrity is not proof of full application-contract conformance.

### S-1: supported, with scope corrections

The backup transition shows exactly 112 dataset status changes, with IDs retained. The current registry has 428 dataset rows: 296 active source-recording archives, 113 missing source-recording rows (one was already missing), one inactive-smoke source row, 17 derived-training merges, and one derived-analysis row. All 296 active source archive paths exist. There are 779 dataset-lineage rows and 21 training sets.

Keeping missing sources and their lineage was appropriate. “Missing archive” must not erase their contribution to an existing merged training dataset or imply their labels are gone everywhere. Conversely, the existing merged copy cannot reconstruct every original review surface or source recording.

### S-2b: copy verified; relocation identity still open

I independently compared sorted relative-file inventories and streamed SHA-256 of every file on both sides for all 19 Zarr trees named in the copy receipt. All matched; all tree file counts and sizes also match the historical receipt. This covered 11,542 Zarr files / 30,147,352,423 bytes in approximately 298 seconds. It did **not** re-audit every non-Zarr file in the original 11,771-file directory copy.

The 17 registered merged dataset paths exist, and no duplicate `datasets.zarr_path` values currently exist. The known three `path-<hash>` IDs remain a credible duplicate-registration hazard: `registry/db.py:603` derives identity from the current path when no session UUID exists; `registry/recording_identity_authority.py:783–826` does not generally resolve a non-source artifact by its already-registered relocated path. Require an idempotent re-registration test before broad rescans, without rewriting immutable archived identity to hide the mismatch.

The external `_index/copy_verification_from_nvme1_2026-09-03.json` is useful but is not a complete long-lived publication receipt: it lacks a self-contained schema/algorithm declaration, persisted per-file inventories, and a bound immutable generation. An external receipt is acceptable; it need not be inserted into a tree and create self-hashing problems. Keep the verified `/groups` copy as the protected copy before treating `/nvme1` as disposable scratch. This review deleted nothing.

### S-9: retained model bytes are coherent; catalog history and locators are not fully reconciled

All 12 retained model paths in both training tables exist and their actual SHA-256 values match the registry. All 24 remaining training-run config/manifest references exist and match their registered hashes. Backup comparisons support the eye-mask model relocation and the four training-run/model deletions. Intermediate values in deleted rows cannot all be reconstructed from the later live state; “13 config changes” is a historical transaction claim, not 13 surviving live rows.

The all-column scan disproves “zero `/nvme1` references” even in the training tables: `training_runs.invocation_json` has 12 affected rows and `final_metrics_json` has three; `training_models.final_metrics_json` has three. Other examples include `datasets.zarr_path` 115, `training_sets.query_filter` 17, `recordings.recording_path` 56, `recording_artifacts.path` 262, and three analytics export manifests. The three inspected `/nvme1` analytics manifests still exist, so `/nvme1` is not simply an absent historic string everywhere.

More importantly, four **active** dataset `source_recording_frame_index_path` values point to missing `/nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam20100{93,94,95,96}/recording_frame_index.parquet` files. Their current Zarr paths existing does not repair those auxiliary locators. Classify each old path as historical provenance, an active locator needing relocation, or retired evidence; avoid indiscriminate replacement.

The old model snapshot and two removed weight directories cannot be independently byte-compared after deletion. Preserved inventories and registry backups support forensic reconstruction of references, not proof that every possible external consumer was unreferenced or that deleted model bytes are recoverable. Future retirement should preserve immutable model/attempt identity and a disposition receipt even when storage is reclaimed.

## 6. Wave-1 brief: what is underspecified before implementation

### Package A — exception propagation and registry status

The package is implementable as limited instrumentation only after these choices are made. It must not be called complete enforcement closure:

1. Define typed contract refusals separately from transport failures, corruption, SQL constraints/schema errors, programming errors, and run-group resolution errors. `sqlite3.OperationalError` alone is not a safe “connectivity” category; the current resolver also catches read errors and turns them into missing-group results.
2. Define an **attempt failure** versus the last accepted stage result. Writing a new `error` row must not unintentionally erase a still-valid prior selection. If validation fails before dataset identity is resolved, where does the refusal belong?
3. Specify the error-write transaction and fallback when that write fails. An invalidation failure rolls back the successful-status transaction; the refusal record needs separate handling that preserves the original exception. A broken connection cannot guarantee a durable error row in itself.
4. The validator requires already-complete and eligible state. Moving it before `mark_run_complete` is not a mechanically valid follow-up. Split candidate validation from selector visibility and declare the publication/registry-recovery ordering. Avoid making selector eligibility a prerequisite for legitimate diagnostic validation.
5. Decide how strictness propagates through caller-level `except` blocks, registry-disabled runs, missing run names, and optional invalidation enforcement. Moving one call outside one `try` does not close those paths. `require_complete_invalidation` is presently optional.
6. Test preserved prior authority, failure during error recording/close, malformed provenance, disabled registry, strict caller behavior, invalidation rollback, and a candidate failing after some output bytes exist. Follow with a real producer→publisher→resolver→consumer boundary test, not only a mocked exception test.

The handoff's suggestion to move validation outside the swallowing `try` must be reconciled with the brief's deliberate non-strict compatibility behavior. The contract cannot honestly become “fail-closed when flag=1” while activation ordering and bypassing publication paths remain unchanged.

### Package B — code identity at execution

`run_with_status` does not currently accept the plan or an expected commit; `LsfWorkflow` has metadata, not the presumed typed `palette_commit` field at the cited location. Specify the plan/job identity schema, serialization, command-builder/CLI propagation, and task-group behavior. An assertion cannot read an expectation that never reaches it.

Checking the imported wrapper package is useful but does not prove that an arbitrary child command imports the same checkout. Bind repository path plus full commit to the actual stage process. Specify dirty checkout, unavailable Git, wheel execution, environment/dependency identity, and override policy. A warning-only mismatch/unknown path cannot qualify as verified production code identity. Record diagnostic exceptions as explicitly ineligible for the stronger claim.

Include a real generated-plan→runtime test, detached-worktree tests, and the crash/retry semantics in finding 8. Keep commit checking distinct from output-content verification.

### Package C — ratchets

Regenerating the size baseline while keeping the existing semantics does not detect a future new oversized module unless the script also discovers unlisted files. Resetting existing baselines to current sizes grants a new growth allowance. A deliberate ceiling can be useful, but it is not a monotonic subtraction ratchet.

Specify AST-level matching/alias handling for open-site counts and a narrow maintained-production scope. An import-linter rule controls imports, not individual function calls. Counting a keyword is also not semantic validation: `registry/inline_refresh.py` uses `consolidated=False` then falls back, which is not the intended `use_consolidated=False` lifecycle policy. Direct opens in diagnostics and low-level storage utilities are not automatically wrong.

`git diff --exit-code` after CI runs catches a generated change relative to the PR checkout; it does not prevent the PR itself from raising the committed baseline. Define comparison against the accepted base and an explicit exception review policy if ratcheting is the goal.

### Brief-wide corrections

“Start from current `sun`” needs an exact commit/ref; no local `refs/heads/sun` was available here. All packages also edit the same authority queue, so they are not completely disjoint. Assign integration ownership for that document. Preserve the no-contract-edit instruction during implementation, but allow necessary contract decisions to be resolved before implementation rather than deferring contradictions until handoff. Full required CI remains a separate acceptance gate; focused local tests do not satisfy it.

## 7. Additional architecture and efficiency findings

These are extensions of existing queue concerns, not reasons to start a platform rewrite.

### Registry reads, ownership, and query correctness

`registry/db.py:1239` opens a writable connection and initializes/migrates schema in `Registry.__init__`. A caller merely intending to query can therefore enter a write-capable initialization path. Read-only helpers already exist in `cohorts/registry.py:34` and `cli/palette.py:320`; converge on a shared explicit read-only/snapshot API and separate migrations/writer ownership.

For shared `/groups` SQLite, choose one write authority or a serialized writer service, short transactions, bounded busy/retry behavior, and explicit reconciliation. Do not casually enable WAL for a multi-host network-filesystem database. SQLite's own guidance warns about network locking and recommends moving database access behind a single host/service when appropriate. This is a deployment/ownership decision, not evidence that an immediate database replacement is required. [SQLite over a network](https://www.sqlite.org/useovernet.html), [SQLite appropriate uses](https://www.sqlite.org/whentouse.html).

`model_resolution.load_target_profile` aggregates independent fields with `MAX`, which can synthesize a combination that never existed if source rows disagree. Define a consistency/refusal rule. Candidate scoring repeatedly loads source rows per model/set and does not recursively resolve all merge lineage in that scoring path. Batch/cache by immutable set identity, and make missing/retired-source semantics explicit. Existing dataset-lineage parent/child indexes are present; choose additional indexes only after representative `EXPLAIN QUERY PLAN` and timing, not from file-size counts.

Dataset-level lineage is insufficient for runtime proof/reuse. The useful projection is run/artifact→input artifact→profile/version→payload or manifest digest→publication generation, with reverse edges for invalidation. Generate it from admitted publications, rather than creating a second source of truth by rescanning guessed names. Existing `ADM-003` is the right home for the executable inventory/proof graph.

### Bounded work and physical ownership

The cited mask apply path materializes all result batches before its serial write loop. Worker count alone does not bound driver memory. Use bounded in-flight results, deterministic ownership, and incremental commit/merge aligned with physical storage units. This is a concrete efficiency issue visible in code; peak memory was not benchmarked in this review.

For actual parallel writers, the ownership unit must be the physical object being updated: a shard when inner Zarr chunks share a shard, not just a logical row range. `shared/zarr/storage_planner.py:138` already represents write ownership and should be reused. Do not apply a universal row-multiple assertion to read-only compute tasks or serialize independent units unnecessarily. [Zarr performance and sharding guidance](https://zarr.readthedocs.io/en/stable/user-guide/performance/).

### Verification cost, immutable generations, and streaming

Strong handles perform substantial payload/identity checks; tracking handle `assert_current` can re-read arrays and compare direct/consolidated snapshots. Those checks are valuable, but should not accidentally become repeated full-archive work for every frame viewed or query served. Define verification lifetimes: full validation when sealing/publishing, a pinned immutable generation at session/query open, appropriate verification of fetched chunks, and periodic integrity scrubs. Cheap checks are justified only by actual immutability and trusted bindings, not path/mtime alone.

Likewise, replacing every mask stale flag with a full dense-mask hash on every open could defeat interactive access. Tie derived caches to an edit revision and sealed content/chunk identities, invalidate on edits, and refresh explicitly. A content-addressed name does not itself prevent mutable-in-place corruption. Lock timestamps do not safely authorize stealing a live lock; any lease takeover needs ownership/fencing semantics.

Keep consolidated metadata as a publication-generation contract. Mutable tools inspect direct metadata; published immutable readers use the validated consolidated generation. An unconsolidated-everywhere codemod would hide publication defects and increase metadata traffic.

### Measure the workload, not just module counts

The repo already has storage planning and benchmark work. `docs/diagnostics/canonical_detection_storage_access_aware_result_2026-07-24.md` records promising physical-layout measurements but explicitly does not claim the frozen eager-consumer gate passed or authorize promotion. Do not relabel a prototype as a production win. The analytics query/export benchmark record likewise contains incomplete evidence, not a finished universal performance baseline.

For the next workload, collect cold/warm open latency, metadata/request counts, bytes read per useful frame, decode throughput, GPU idle time, peak worker/driver memory, queue wait, stage duration, publication/verification cost, and resume behavior against the same pinned scientific inputs. Separate local scratch throughput from shared-filesystem and HTTP access. These measurements can guide chunk/shard size, batching, prefetching, and concurrency without making Crimson integration a prerequisite. Crimson itself was not audited here.

## 8. Checklist disposition and sequencing

Use these dispositions when adopting work into the existing [authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md). This report intentionally changes no queue status and creates no additional tracking IDs.

| Existing review items | Disposition | Existing queue/contract home |
|---|---|---|
| S-5 backfill; propagation K-1/K-5, M-2, X-5/X-8 | Replace blanket migration with lifecycle/use-scoped admission and exact-path migration. Preserve ineligible candidates. | ADM-001/002, ACC-002, RES-TRACK-001, PROD-MASK-001, REP-001 |
| S-6 “32 RedScare reviewed runs” | Correct population to 28+4; inspect actual acceptance/content before any activation. | ACC-001/002 |
| O-1/O-2, census G-1/G-2, Package A, training M-1 | Overlap in error/activation semantics; split observability from actual boundary closure and transactional projection. | ADM-001/002, REP-001, TEST-001; wave brief's designated Stage 11 tracking |
| “sealed tracking unused,” K-1 one-call swap | Reuse existing provider/handle path; preserve instance-key, position/body-frame, coordinate and temporal alignment. | RES-TRACK-001, TEST-001 |
| Census R-*; training R-*; correction L-3 | Reconcile with already implemented review/compaction contracts; add missing boundary validation rather than a parallel receipt schema. | ACC-001/002 and the acceptance checklist |
| Training T-1, correction L-2, measurements S-3 | One membership projection with merge expansion, split/group identity and unknown coverage, not three independent features. A persisted SQLite view is a schema change even if no new table is needed. | ACC-003, REP-001 |
| Pose seed M-3; export inventory B-7; report manifest F-3 | Partly done. Limit remaining work to determinism/effective args, durable inventory binding, and renderer identity respectively. | Relevant producer/reporting boundary work; TEST-001 |
| Digest D-1..D-4; envelope E-* | Versioned shared mechanics after domain compatibility tests; no unversioned mass rehash or mandatory universal review field. | ADM-001, ACC-001, existing payload/manifest contracts |
| H1 D-5; mask-cache S-5 | Pair byte verification with immutable backing/edit-generation policy; account for interactive read cost. | Existing storage/publication contracts and boundary tests |
| Dask S-6 | Correct worker/write census first; bound gather memory; enforce shard ownership only on actual writers. | Existing Dask write-safety/storage-planning contracts |
| Figure F-1/F-2; census B-8 | Same feature; resolve circular digest recipe and reuse existing export-manifest linkage. | REP-001 |
| Engine N-1/N-2; LSF O-* and Package B | Require attempt identity/recovery and input/output identity before treating resume/retry as safe. Evaluate engine in a bounded experiment. | ADM-002, TEST-001, Stage 11 |
| File/open ratchets C-*; deletion/refactor campaigns | Diagnostic ceiling, not proof of runtime safety. Subtract only after supported callers and boundary coverage are established. | ADM-003, SUB-001, TEST-001 |

Recommended order: first correct the unsafe premises and classify the three ineligible projections plus stale active locators; next define and test admission/publication/projection semantics on one maintained end-to-end path; then converge tracking and other consumers on existing verified suppliers. In parallel, bind model dataset/split identity and harden submission recovery, followed by measured storage/memory improvements. Only then expand migrations and remove compatibility fallbacks under explicit deletion gates.

Do not make new review tooling wait for a universal retraining loop: review evidence is itself needed to build that loop. Do not let unresolved unrelated catalog entries block an already admitted dependency closure. Do not activate a selector, release a cohort, or call an implementation merge-ready without its required CI and applicable acceptance evidence.

## 9. Reproduction, validation, and limits

The two landed diagnostics were rerun read-only, outside the sandbox for real Zarr access:

```bash
scripts/py -B scripts/sweep_run_selectors.py --out /tmp/palette-second-opinion-20260904-LlmmLz/selectors.jsonl
scripts/py -B scripts/check_training_membership.py --sweep /tmp/palette-second-opinion-20260904-LlmmLz/selectors.jsonl --json /tmp/palette-second-opinion-20260904-LlmmLz/membership.json
```

SQLite acceptance command, run on the live path and separately on each exact backup listed in §5:

```bash
scripts/py -B -m fisheye.utils.registry_integrity --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

Additional temporary read-only probes: `audit_registry.py` (snapshot comparisons, hashes, locator checks), `audit_selectors.py` (direct-metadata lifecycle census), `audit_copy.py` (per-file comparison), `final_probe.py` (ineligible projections, consolidated mismatch, auxiliary locators, pose split counts), and `digest_counterexample.py` (the in-memory partition test in §3). Durable summary evidence is included above; raw JSON outputs are temporary and are not promised as long-term archived receipts. These probes opened SQLite with `mode=ro`/`query_only`; metadata probes read files directly; the small split-array probe used Zarr `mode=r, use_consolidated=False`.

Focused validation, outside the sandbox:

```bash
scripts/py -B -m pytest -q -p no:cacheprovider \
  tests/unit/fisheye/test_zarr_run_completion.py \
  tests/unit/fisheye/test_detect_keypoints_step_status.py \
  tests/unit/fisheye/test_tracking_source_handle.py \
  tests/unit/fisheye/test_provider_track_motion_publication.py
```

Result: **115 passed in 19.22 seconds**, with one expected legacy-completion compatibility warning. This supports the existence and tested behavior of these primitives; it does not prove every production entry point routes through them. No full CI or cluster/GPU workload was run, no current CI status was established, and this report makes no branch merge-readiness claim. No performance improvement, historical frame-level contamination rate, universal acceptance coverage, or complete deletion recoverability is claimed.
