# Pipeline Model, Orchestration Boundary, and the Correction Loop — 2026-09-02

> Disposition added 2026-09-06: the registry-only authority/pointer rule and
> universal envelope dependency below are not adopted. The [second opinion](review_wave_second_opinion_2026-09-04.md)
> preserves stage/use-scoped authority and distinguishes registry projection
> from publication and acceptance. Neither an engine migration nor a blanket
> selector backfill is a prerequisite for narrow enforcement corrections.
> Historical proposals remain below; see the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation) before assigning work.

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-02
implementation: specified-only
-->

**Date:** 2026-09-02
**Snapshot:** `d0681f0a` on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`
**Method:** synthesis from the four review docs of 2026-09-01/02 and a conversation about whether palette's data model and workloads are sound against scalable pipelining (Nextflow-class engines), and how to think about the human-correction burden of a solo lab. No new code reading beyond what those docs cite. Line numbers are theirs and must be re-verified before implementation.
**Questions asked:** (1) Is the receipt work a reinvention of Nextflow, and how would the two coexist? (2) Is the data model good for scalable pipelining? (3) Is needing to edit masks and keypoints a sign the approach is wrong? (4) What are W3C PROV and RO-Crate?
**Companion evidence:** [`../workflow_provenance_prior_art.md`](../workflow_provenance_prior_art.md) (2026-07-24; Nextflow, Snakemake, DVC, Pachyderm, DataLad, MLflow, PROV in depth; **read it first, this doc does not repeat it**); [`receipt_builder_census_2026-09-02.md`](receipt_builder_census_2026-09-02.md); [`training_data_and_model_provenance_review_2026-09-01.md`](training_data_and_model_provenance_review_2026-09-01.md); [`architecture_review_five_lens_2026-09-01.md`](architecture_review_five_lens_2026-09-01.md); [`../error_budget_policy_2026-08-11.md`](../error_budget_policy_2026-08-11.md).

**Queue disposition:** audit evidence and checklist source. Orchestration items (N-*) are a future decision, not a queue item now. Immutability items (I-*) adopt into [`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md). Correction-loop items (L-*) extend the T/R items of the training review. Export items (X-*) follow the envelope module (E-1 in the census). This document does not track status.

**A note on the prior-art doc.** It was written by an agent on 2026-07-24 and reaches the same conclusions this conversation reached independently six weeks later. That it had to be rediscovered is a governance finding in its own right: documents produced by agents are not being read back by the person they were written for. Item G-3 (findings ledger) in the architecture review is the fix.

---

## 1. Verdict

1. **The orchestration layer is a Nextflow reimplementation; the evidence layer is not.** The 69 bash submitters, completion-marker polling, subprocess runner, and commit-pinning script do what a workflow engine does. The envelope/receipt layer is a genuine attestation design with no engine equivalent. The two are separable today, and adopting an engine later would delete the first without touching the second.
2. **The data model is conventional except for one structural conflict: mutable-in-place storage with pointers inside the data.** Every scalable pipeline keeps outputs immutable and pointers in a catalog. Palette keeps both in the zarr and mirrors pointers to the registry, which is the root of the split-brain, the nine resolvers, and reconcile-by-presence. The fix is a rule, not a rewrite.
3. **Needing to correct model output is the normal condition of a novel rig, not a defect.** Every pose/segmentation tool in real use ships a correct-and-retrain loop as its core. The defect is that palette's loop is not closed: corrections do not flow back into training with a receipt, and no per-stage error rate is measured, so review effort is set by anxiety rather than by a number.
4. **PROV and RO-Crate are export formats for the envelopes, not replacements.** Once the envelope module exists, both are small serializers. The cohort release is the natural RO-Crate unit.

---

## 2. Two layers, and where the boundary is

| Layer | Owns | Palette today | Engine equivalent |
|---|---|---|---|
| **Scheduling** | which stage runs next, on which node, with which inputs; retry; resume; skip-if-done | 69 bash bsub submitters; `cluster/lsf` typed framework used by a minority; zarr completion markers polled by subprocess runner; `scripts/deploy_palette_cluster_worktree.sh`; wave-1 Package B commit assertion | Nextflow process + LSF executor + `-resume` + container/worktree pin |
| **Evidence** | what was produced, by what code and config, from what inputs, accepted by whom; verified before promotion | envelopes/receipts (census §3); `mark_run_complete`; `stage_complete`; selector activation; registry projection; review receipts | none; the engine sees exit code and output paths only |

**Coexistence.** Each stage entrypoint already has the shape an engine wants: one command, inputs in, outputs out, nonzero exit on failure. Adoption is one Nextflow process per stage calling the existing entrypoint. Nothing under `shared/`, the envelope module, the gate, or the registry changes. The gate matters more under an engine, not less, because an engine marks a task complete on exit code zero and will cache a bad output forever if the stage did not refuse to write it.

**Why engines do not do the evidence layer.** Their unit of identity is the file, and their hash exists to answer one question, "can I skip this task," so it must be cheap and generic. They cannot know what correct means inside a zarr run group, have no model of a human approving a subset of rows, and no rule for choosing among competing outputs for one input. Those are domain decisions. Engines also grew up in bioinformatics, where inputs are immutable files, tools are deterministic, and there is one output per input. Palette's setting is mutable review, several runs per recording, and a store edited in place. The generic pieces got built by others; the pieces that depend on this data model could not be.

**What has no engine equivalent and should be kept:** sub-file identity inside a zarr with physical and decoded roots; human review as a first-class input; several competing runs per recording with a promotion rule; the registry as a queryable projection; the realtime TensorRT path.

---

## 3. The data-model conflict: mutability and pointer placement

| Practice | Nextflow / DVC / Rubin butler / Delta | Palette |
|---|---|---|
| Output artifacts | immutable once written; a rerun writes a new artifact | run groups are mutable (review runs, add-a-row, successor hardlinks, attr rewrites) |
| "Current" pointer | lives in a catalog separate from the data | `latest`, `latest_complete`, selector, `authoritative_run`, `subject_mask_authority` all live in the zarr attrs next to the data, then mirrored to the registry |
| Resolving "current" | one catalog query | nine resolvers (architecture review §2) |
| Staleness | catalog compares content hashes | reconcile trusts `attrs["latest"]` then marker presence (census §4 path D) |

Consequences the reviews already found, restated as one cause: split-brain between consolidated and live metadata; reconcile-by-presence; verifiers that cannot be trusted because the artifact they verified may have changed; and the H1 hardlink hole, which exists only because a successor can share bytes with a still-mutable source.

**The rule.** A run group becomes immutable at `mark_run_complete`. Anything that needs to change afterward is a new run group with the old one in its inputs block. The registry is the only authoritative pointer; the zarr keeps `latest` as a cache that reconcile overwrites from the registry, never the reverse. Review runs already follow the "new run" shape; the remaining offenders are attr rewrites on completed runs and in-place selector flips.

---

## 4. The correction loop

**Why editing is not a failure.** A model trained on another lab's animals, dish, and lighting fails on yours. Labels for your distribution come only from correcting outputs on your distribution. DeepLabCut, SLEAP, and CVAT each ship a labeling interface as the core product for this reason. Labs that appear to run without correction either have annotators or have accepted a fixed error rate and carry it as uncertainty. A solo lab with a novel rig has neither option for free.

**How practitioners think about it, in payoff order.**

| Idea | Meaning | Palette status |
|---|---|---|
| Edits are data, not fixes | a corrected mask or keypoint set is a training row with a human agent; it must reach the next training set with a receipt | review is enforced as an export precondition and then discarded (training review R-1..R-4; census R-4); `model_resolution` never excludes sets containing the target recording (T-2) |
| Measure the error rate before sizing review | per-stage model failure rate on held-out reviewed frames decides whether 2 % or 40 % of frames need eyes | error budget policy exists; zero measurement code (architecture review G3); no held-out evaluation bound to a model row |
| Review the uncertain frames | confidence, model disagreement, tracker loss, and geometry checks nominate frames; 300 hard frames beat 3000 random ones | signals exist (tracker-loss finding, chaser escape geometry notes, `subject_mask_quality_manifest`) but nothing routes them to the labeling queue |
| Carry residual error forward | flag, exclude from sensitive analyses, report the exclusion | done well in places (valid-tracked-time denominator for escape rate); not systematic |

**The caution.** The labeling web app, mutable review runs, add-a-row propagation, and successor hardlinks are a large fraction of the codebase in service of corrections that do not yet flow into training with a receipt. Closing the loop and measuring how fast the error rate falls should decide how much more review tooling is worth building. The manual add-row design doc (2026-07-08) should not be implemented until L-1 and L-2 below exist.

---

## 5. W3C PROV and RO-Crate

**W3C PROV** is a standard vocabulary for how something came to be: three node types and a few edges. An **entity** is a thing (a run group, a figure). An **activity** is something that happened over an interval (a detection run, a review session). An **agent** is who was responsible (a person; a script at a commit). Edges: *wasGeneratedBy* (entity ← activity), *used* (activity → entity), *wasAssociatedWith* (activity → agent), *wasDerivedFrom* (entity → entity). Serialized as JSON-LD or RDF. The prior-art doc §"W3C PROV" has the fuller treatment.

Mapping to the envelope parts of the census: subjects are *wasGeneratedBy*, inputs are *used*, producer is *wasAssociatedWith* with a software agent, review is a second activity with a human agent. An envelope is the local neighbourhood of one activity node. Exporting PROV is therefore a traversal over envelopes, not a new data model.

**RO-Crate** is a packaging convention. A directory of files plus one `ro-crate-metadata.json` at the top describing every file and its relationships in schema.org vocabulary, with PROV terms for provenance. The result is a self-describing folder a repository or reviewer can open without knowing the pipeline. **Workflow Run RO-Crate** is the profile for one execution: which workflow, inputs, outputs, tool versions, order. Nextflow's nf-prov plugin emits it.

PROV answers "how did this come to be." RO-Crate answers "how do I hand this to someone else." The cohort release is the natural crate: parquet parts, manifests, figures, frozen cohort, one metadata file. The current release record binds inputs by digest but outputs by path and has no reader (census §3.4), which is the inverse of what a crate needs.

---

## 6. Checklists

### 6.1 N: orchestration boundary (decision items, not queue items)

- [ ] **N-1** Write one Nextflow process for one existing stage entrypoint (detection is the obvious candidate) against the LSF executor, in a scratch directory outside the repo, and record: does the stage run unmodified; does `-resume` skip correctly; does the commit pin survive. This is a one-day experiment that replaces speculation.
- [ ] **N-2** Decision record: on the next occasion the bsub submitter sprawl reaches a queue, the options are (a) consolidate 69 scripts onto `cluster/lsf`, (b) replace with engine processes. Record which, and why. Do not do (a) by default.
- [ ] **N-3** Wave-1 Package B (runtime commit assertion) stays as specified; note in its handoff that under an engine it becomes a container/worktree pin and the assertion moves to the process definition.
- [ ] **N-4** Confirm each stage entrypoint fails with nonzero exit when it refuses to publish (after G-1 lands). An engine caches on exit zero.

### 6.2 I: immutability rule

- [ ] **I-1** Contract statement, one paragraph, added to `zarr_run_completion_contract.md` after G-1 lands: a run group is immutable after `mark_run_complete`; changes are new run groups citing the old in `inputs`.
- [ ] **I-2** Census of writers that mutate attrs or arrays on a completed run group (grep for attr writes after `is_run_complete_in_parent`). Classify each as "becomes a new run" or "moves to registry."
- [ ] **I-3** Registry is the authoritative pointer: reconcile writes zarr `latest` from the registry row, never the reverse; `_resolve_latest_group` fallback ordering (`maintenance.py:4113`) is deleted once the direction flips.
- [ ] **I-4** Selector flips (`selector_activation.py`, family activators) record the flip as a registry write first, zarr attr second; on partial failure the registry wins.
- [ ] **I-5** Successor hardlinks are only created from completed, immutable sources; H1 rehash (D-5 in the provenance review) becomes redundant once I-1 holds but stays until I-2 is empty.
- [ ] **I-6** Ratchet: no new `attrs[...] =` write targeting a group whose completion marker is set, outside the registry-driven reconcile path.

### 6.3 L: close the correction loop

- [ ] **L-1** Per-stage held-out evaluation bound to every `training_models` row: a reviewed frame set the model never trained on, an error metric per stage (mask IoU, keypoint PCK, detection recall), stored as a digest-bound receipt in the model's envelope. This is the first measurement behind the error budget policy.
- [ ] **L-2** Training-membership view (T-1 in the training review) run against the live registry, result recorded. No claim about a production model's error rate is valid before this.
- [ ] **L-3** Review edits flow to training with a receipt: every reviewed row carries a review-receipt digest, every training export binds `source_review_receipts` (R-4 census, R-1..R-4 training review). Verify by exporting one training set and confirming a corrected frame appears with its receipt.
- [ ] **L-4** Uncertainty routing: one queue (a registry table) fed by existing signals: detector confidence below threshold, tracker loss intervals, mask quality manifest flags, model disagreement where two models exist. The labeling web app reads from this queue before random frames.
- [ ] **L-5** Retrain cadence tied to L-1: a retrain is triggered when the queue holds N reviewed frames, and its evaluation delta is recorded against the parent model (`parent_run_id`, M-2 in the training review).
- [ ] **L-6** Residual-error policy per analysis: each analytics export declares which frame-level flags it excludes and reports the excluded fraction in its manifest. The valid-tracked-time pattern from escape rate becomes the template.
- [ ] **L-7** Freeze on new review tooling (add-a-row propagation, multi-instance add) until L-1 and L-3 are done and one retrain has shown a measured error-rate drop.

### 6.4 X: provenance exports (after E-1)

- [ ] **X-1** `prov_export(run_ref) -> PROV-JSON` traversing envelopes' inputs blocks recursively to the video root; emits entity/activity/agent per §5 mapping. Validate with an off-the-shelf PROV validator.
- [ ] **X-2** `release_crate(release_id)` writes `ro-crate-metadata.json` beside the cohort release, describing parts, manifests, figures, frozen cohort, and the PROV graph from X-1. Validate against the Workflow Run RO-Crate profile.
- [ ] **X-3** The release record (B-6 census) binds output digests so the crate can reference them.
- [ ] **X-4** One crate deposited to a test Zenodo sandbox or equivalent to confirm a stranger can open it.

### 6.5 Sequencing

| Order | Items | Depends on |
|---|---|---|
| now | L-2, N-1, I-2 | nothing; all read-only or scratch |
| with wave 1 | N-3, N-4, L-1 design | Package A/B |
| after G-1 | I-1, I-3, I-4, I-6 | gate fail-closed |
| after E-1 | L-3, L-4, X-1, X-3 | envelope module; review binding |
| after L-1 + L-3 | L-5, L-6, L-7 lifts | measurement exists |
| last | I-5 retirement, X-2, X-4, N-2 | everything above |

---

## 7. What was not assessed

- Whether Nextflow's LSF executor handles this cluster's queue and GPU reservation conventions; N-1 answers it.
- Actual per-stage error rates; none are measured (L-1).
- Whether SLEAP or DeepLabCut labeling GUIs could replace the labeling web app for the correction step. They do not read palette's zarr layout; the trade is a converter versus continued maintenance of a bespoke UI. Worth a separate look after L-3.
