# AGENTS

## Python Environment Rule

- Use `scripts/py` for all Python commands in this repository.
- Do not run `conda activate` in this repository.
- Prefer `scripts/py -m <module>` over bare `python -m <module>`.
- `scripts/py` is expected to resolve to the `palette-py311` conda environment; verify with `scripts/py -c 'import sys; print(sys.executable)'` when needed.
- Do not run install or dependency mutation commands unless the user explicitly approves in chat first.
- Blocked without approval: `pip install`, `conda install`, `mamba install`, `poetry add`, `uv pip install`.

## Registry SQLite Runtime Rule

- Palette registry acceptance must use the SQLite library loaded by
  `scripts/py`, never a separately installed `sqlite3` command-line binary.
- Use `scripts/py -m fisheye.utils.registry_integrity --registry PATH` for
  complete `PRAGMA integrity_check` and `PRAGMA foreign_key_check` validation.
- Use `scripts/backup_palette_registry.sh` for backups. It delegates backup and
  validation to the same Palette Python SQLite runtime and records that runtime
  in its receipt.
- A system `sqlite3` result may be collected as supplementary diagnostic
  evidence, but it cannot override or satisfy Palette's acceptance gate.

## Git Push Rule

- Pushes from this repository require the Palette workstation SSH key and should run outside the Codex sandbox because sandbox DNS/network access can fail.
- When a workflow intentionally uses the single shared `/groups` checkout,
  prefer the tracked helper, which pushes the current branch and then
  fast-forwards that checkout:
  `scripts/push_and_update_groups_checkout.sh`
- When concurrent agents or cluster jobs need different Palette commits, do
  not switch or fast-forward the shared checkout. Use
  `scripts/deploy_palette_cluster_worktree.sh` from the clean source worktree.
  It pushes only that branch, creates a detached commit-pinned worktree below
  the shared deployment root, leaves the shared checkout unchanged, and prints
  the exact `PALETTE_GROUPS_REPO` value for submission.
- `scripts/py` prepends the `src` tree beside that exact script. The deployment
  helper verifies the imported `fisheye` path through Citrus before reporting
  success, so another checkout's editable install cannot supply job code.
- Cluster submissions from a dedicated deployment must record and pass its
  absolute `--palette-repo` path and full commit. Never point an already-planned
  job at a mutable shared checkout or move a deployment path to a newer commit.
- If you intentionally need to push without updating `/groups`, use:
  `GIT_SSH_COMMAND='ssh -i /home/delahantyj@hhmi.org/.ssh/delahantyj-ws1-git-id_ed25519 -o IdentitiesOnly=yes' git -C /home/delahantyj@hhmi.org/gitrepos/palette push`
- Do not rely on plain `git push` for Palette; it may fail with `Permission denied (publickey)` or sandbox DNS errors.
- The `/groups` fast-forward must fail closed: do not use merge commits, rebase, reset, or dirty-checkout workarounds unless the user explicitly approves.

## Required CI and Integration Rule

<!-- required-ci-integration-contract:v1 -->

- A branch must not be merged, integrated into another merge candidate,
  fast-forwarded into the shared `/groups` checkout, used to activate a
  production selector/publication, or described as complete or merge-ready
  unless every required CI check for that change has completed successfully.
- Failed, cancelled, timed-out, or accidentally skipped required checks are
  blocking. A check skipped because an earlier gate failed is unrun evidence,
  not a successful result. Do not dismiss a blocking result as pre-existing or
  unrelated while still describing the branch as safe to merge.
- An intentionally inapplicable check is acceptable only when the workflow
  contract makes that condition explicit. Record the reason in the handoff.
- A commit or branch with failing or unrun required checks may be pushed or
  handed off only as explicitly incomplete work. The handoff must identify the
  exact commit, every failing or unrun check, and the validation still needed.
  An incomplete handoff is not authorization to merge, integrate, promote, or
  update the shared checkout.
- A commit-pinned experimental deployment may be used to obtain missing
  evidence when the task requires it, but it must remain selector-ineligible,
  must not alter production authority, and must be reported as not merge-ready
  until the required CI is green.

## Collaborative Development and Contract Preservation

<!-- collaborative-development-contract:v1 -->

### Worktree and interface ownership

- Before overlapping implementation work, reconcile the relevant worktrees'
  exact commits, uncommitted changes, owners, prerequisites, and handoffs.
  Recheck audit findings against those versions; an older checkout or review
  must not overwrite stronger work already developed elsewhere.
- Use separate worktrees for concurrent implementation. Do not switch, reset,
  rebase, clean, or remove another worker's checkout or changes. Read-only
  reviewers may share a checkout; coordinate ownership of review documents.
- Agree one owner for each shared interface, receipt/digest grammar, and
  integration branch. Parallel consumers should use that agreed interface;
  resolve conflicting contract changes before integration rather than adding
  competing helpers. Unrelated performance/scientific work need not stop.
- Use `docs/diagnostics/authority_consolidation_work_queue_2026-08-25.md`
  for overlapping authority/admission/consolidation status. Preserve separate
  performance and scientific work queues. Audits and subtraction censuses are
  evidence to reconcile, not additional independent status authorities or
  permission to delete code without current caller/compatibility checks.

### Preserve contracts before simplifying implementations

- Classify a change as behavior-preserving extraction, performance/storage
  optimization, enforcement correction, or scientific/schema/identity change.
  State what must remain identical and what may change before implementation.
  Do not hide semantic changes inside a cleanup or performance patch.
- Capture preservation tests first: valid artifacts and golden identities,
  plus malformed, tampered, stale, incomplete, and wrong-source/use cases.
  Preserve valid workflows; deliberately tightened rejection behavior must be
  identified and tested as an enforcement correction.
- Preserve product-specific scientific validators, row/frame identity,
  coordinates, validity, supplier sufficiency, and authority/acceptance rules.
  Share repeated mechanics, not a weaker generic contract. Inventory existing
  helpers and adopt or extend the appropriate owner instead of adding another
  implementation or a catch-all `shared.primitives` module.
- Preserve existing serialization and digest bytes wherever their contracts
  promise stability. Do not conflate logical content, physical layout,
  manifests, or execution attestations. New commits/timestamps/layouts can
  legitimately change receipts; compare declared invariant fields, not whole
  receipts indiscriminately. Never assume a digest survives rechunking.
  Intentional persisted-grammar changes require an explicit version and
  compatibility decision; do not rewrite historical evidence to fit them.
- Preserve the camera/rig's scientific defaults. Represent changes through a
  named/versioned applicable recipe, with effective parameters and permitted
  overrides recorded through existing provenance machinery. Keep scientific
  parameters separate from execution resources and physical storage settings;
  preserve required source, model, producing-code, and environment bindings.
- Receipt-based reuse or faster validation must establish the same required
  claim under explicit immutable-identity/generation assumptions. Reject
  stale/conflicting evidence; never invent a future digest or weaken a
  validator to obtain a speedup. Require the concrete admission appropriate
  to the consumer before reuse, execution, or submission.
- Preserve bounded memory/I/O and physical write ownership under the rules
  below. Benchmark comparable inputs and separate computation, data movement,
  validation, and publication costs; faster runtime alone is not acceptance.

### Stage adoption, integration, and removal separately

- Prefer small changes: establish tests/interface, migrate one caller, prove
  behavior, migrate remaining maintained callers, add scoped enforcement,
  then remove executable copies or retain named compatibility adapters.
  Do not combine extraction, digest-format migration, historical backfill,
  production activation, and broad deletion in one change.
- Test supported public paths through the real producer, publisher, resolver,
  and unpatched consumer where applicable. Include refusal, ownership loss,
  failed publication, and retry behavior; import identity or a helper-call
  assertion alone does not prove enforcement.
- Follow the Required CI and Integration Rule above for each incoming exact
  commit. After integration, validate the resulting combined commit as well;
  independently green branches do not establish a green combination.
- When authorized canary evidence is needed, use commit-pinned deployments
  and fresh selector-ineligible outputs under the existing deployment rules.
  Keep historical mutation, migration, and production activation separately
  scoped and authorized; never perform them as incidental cleanup.
- Finish consolidation by migrating maintained callers and preventing new
  bypasses with scoped import/AST checks and behavioral conformance tests.
  Net line reduction, file splitting, or adding a helper is not completion.
  Extend existing catalogs/ratchets rather than creating competing ones.

### Report product completion precisely

- Distinguish computation complete (validated numerical products), presentation
  complete (validated required plots/specifications bound to those numerical
  identities), and deliverable complete (all products required by the chosen
  workflow are ready).
- An independent presentation failure must remain visible and keep a
  presentation-required deliverable incomplete without invalidating already
  validated numerical products. Retry presentation independently where safe.
  None of these claims alone establishes scientific acceptance or activation.
- This distinction is not permission to rename persisted status fields,
  bypass an existing stage contract, or manufacture completion evidence.
  Change lifecycle schemas only through an explicit compatible/versioned plan.

### Handoff requirements

- Record the worktree/branch and exact commit, dirty/uncommitted scope, owner,
  prerequisite commits, changed interfaces/contracts, preserved invariants,
  intentional behavior/identity changes, tests/benchmarks/canary evidence,
  every failing or unrun required check, and compatibility/removal work left.
- Identify separately what is implemented, validated, integrated, deployed,
  and activated. Keep status in the owning queue and link the handoff evidence.
  These collaboration rules do not expand the user's authorization to commit,
  push, install dependencies, deploy, or mutate production data.

## Authority Roles and Supplier Sufficiency

- Require only the authorities declared by the consumer's contract. Do not
  invent a second upstream-authority or human-review gate when a validated
  supplier already binds that upstream source, its controlled derivation, row
  identity, coordinate system, validity, and content digests.
- A validated body-frame supplier is sufficient for consumers that need only
  anatomical origin, forward/left axes, heading, and validity. Its source
  keypoints or masks remain sealed lineage; those upstream authorities need to
  be resolved separately only by consumers that read their landmarks or pixels.
- `keypoint_authority=false` on a body-frame run prevents the derived cache from
  masquerading as canonical keypoints. It does not invalidate the run's own
  body-frame authority and does not imply that a separate human keypoint review
  is required.
- Do not infer a human-review requirement from words such as `candidate`,
  `unreviewed`, or `selector-ineligible`. Human review is an execution gate only
  when an applicable contract defines the reviewed claim, receipt schema, and
  runtime validator. A representative visual inspection may instead be a
  distinct promotion or publication-quality check.
- Keep `canonical` and `authoritative` claims separate. `canonical` identifies
  the validated standard schema, coordinate system, lineage, and normal
  production representation. `authoritative` identifies the exact accepted
  selection for a declared use. A canonical candidate need not be
  authoritative, and an authority activation must first validate the applicable
  canonical contract.
- Do not treat the mere presence of `authoritative_run` as proof of scientific
  review. The generic authority primitive proves that a named run exists, is
  complete, is selector-eligible, and has approval provenance; it does not by
  itself validate a stage-specific review payload. Claim reviewed authority
  only when the stage activation path also validates the required review state
  or receipt.
- `authoritative_run` is not the universal authority mechanism. Canonical raw
  detections use matching `latest`/`latest_complete` selectors plus the bound
  canonical-detection contract and digest; refined detections may use
  `refined_detect_runs.authoritative_run`; manually accepted refined subject
  masks may use `refined_subject_masks_runs.authoritative_run`; and modern
  atomic subject-mask publication selects a cross-family bundle through the
  root `subject_mask_authority` envelope.
- Detection authority is explicit and stage-specific: `detect_runs/<run>` may
  be the selected canonical raw/model detection authority, while the finalized
  `refined_detect_runs/<run>` (or its clipped finalized collection) is the
  downstream curated detection authority. Derived position, crop, or analysis
  runs must not relabel themselves as detection authorities.
- For modern segmentation, use the existing subject-mask vocabulary. A model
  produces a `subject_mask_runs/<run>` candidate; a recording-level bundle binds
  raw mask, refined dense mask, and quality members; explicit activation creates
  the root `subject_mask_authority` envelope. Do not create a parallel generic
  `segmentation_authority` when that subject-mask contract applies. A trained
  network or model artifact is not itself per-recording mask authority.
- Follow
  `docs/diagnostics/authority_acceptance_implementation_checklist_2026-08-27.md`
  when changing review, acceptance, approval, authority resolution, registry
  projection, or cohort-release behavior. Do not introduce another
  modality-specific acceptance schema without reconciling it there.

## Sandbox Zarr Fallback Rule

- If sync `zarr.open_group(...)` hangs in Codex sandbox, use metadata-file checks from `docs/sandbox_zarr_fallback.md`.
- For keypoint review status checks, prefer `zarr.json` + `jq` fallback over Python `zarr` reads when sandbox hangs are observed.

## Consolidated Metadata Read Policy

- Treat metadata mode as a lifecycle decision, not a universal reader option.
- Writers, edit tools, and readers inspecting an actively mutable or incomplete
  Palette Zarr must use `use_consolidated=False` so newly created groups are
  visible during mutation.
- For a selector-visible immutable publication, finish and validate all payload,
  attrs, and provenance writes; update the direct `latest`/manifest selection
  metadata; then consolidate the root as the final published visibility step.
  Validate that the consolidated generation contains the intended selector
  state before declaring publication complete.
- New readers of published immutable artifacts should use consolidated metadata
  by default and validate the published metadata generation/schema contract.
- Missing or stale consolidated metadata on a published immutable artifact is a
  publication defect. Do not silently normalize that state by making
  unconsolidated traversal the permanent reader default.
- Diagnostics and benchmarks that compare metadata paths must select
  consolidated or unconsolidated mode explicitly and record the selected mode
  in their result.

## Sandbox Zarr Test Policy

- Invoke Palette test suites only from a workstation checkout. Never run
  pytest on the campus `login1` or `login2` nodes, and do not submit test
  suites to LSF as a substitute for workstation validation.
- `scripts/py` rejects pytest invocations on those hosts before Python starts,
  and `tests/conftest.py` independently enforces the prohibition before
  collection.
- Run pytest-based validation outside the Codex sandbox by default for this repository; tests run normally there and sandbox execution can hang on zarr paths.
- Use `scripts/py -m pytest ...` with an outside-sandbox/escalated command when running focused or full test suites.
- Keep in-sandbox validation to static/non-zarr checks such as `scripts/py -m py_compile`, `git diff --check`, or explicitly safe in-memory tests.
- In Codex sandbox, prefer in-memory or fake-group test harnesses (with monkeypatch) for zarr-related unit tests.
- Do not rely on sync real-zarr integration tests in sandbox when equivalent logic can be validated with in-memory tests.
- If a real-zarr test hangs or is known to hang in sandbox:
  - stop that test path,
  - run non-hanging validation (`scripts/py -m py_compile` and relevant fast unit tests),
  - if the real-zarr test is important to the current change, rerun that exact focused test outside the sandbox with escalation,
  - if outside-sandbox execution is unavailable or still fails, report the skipped test as deferred local validation.
- For deferred local validation, provide exact commands for the user to run in their terminal.
- For new zarr-heavy tests, default to deterministic in-memory coverage first; add real-zarr integration checks only when required and mark them for local execution if sandbox stability is an issue.

## Dask / Parallel Zarr Write Rule

- Parallel writes to Zarr are safe only when each worker owns whole, non-overlapping physical Zarr chunks for every array it writes.
- Do not assume disjoint logical row slices are safe. If two workers write different row ranges inside the same physical chunk, Zarr chunk-level read-modify-write behavior can cause stale overwrites.
- When adding or changing Dask writes, align worker chunks to the physical chunk grid of the written arrays, serialize writes that cannot be chunk-aligned, or write per-worker temporary outputs and merge deterministically.
- Record both requested and effective worker chunking in provenance when Dask chunk sizes are adjusted for Zarr write safety.
- See `docs/dask_zarr_write_safety.md` before modifying Dask-backed Zarr writers.

## Subject Mask Direction

- Treat eye-mask-specific stages as legacy compatibility surfaces.
- For new mask work, prefer `subject_mask_runs` and `refined_subject_masks_runs`.
- For modern editable refined subject-mask outputs, dense `masks_roi` is the
  authoritative pixel surface and must be physically present. Compact
  `mask_bitpacked` and `mask_rle` stores are derived display/archive caches, not
  edit or training authorities.
- Review/edit paths should mutate dense `masks_roi` only. After accepted dense
  edits, mark derived bitpacked/RLE/metrics/contours stale and refresh them only
  during explicit validation, promotion, or maintenance steps.
- Historical compact-only refined subject-mask runs may be read through
  `MaskStore` for compatibility, but must be materialized to dense `masks_roi`
  before review/editing or training export.
- Do not add new workflows, docs, or model paths that make `eye_masks_runs` or `refined_eye_masks_runs` the primary source of truth.
- It is acceptable to read, migrate, validate, or materialize eye-mask compatibility data when supporting historical archives or legacy consumers.
- Canonical manual review/editing for body, swim bladder, and eye components should route through unified refined subject-mask tooling and component review state.
- Training/export paths should prefer subject-mask contracts; eye-mask training artifacts remain legacy or compatibility-specific unless the user explicitly asks for them.
- Subject-mask training zarrs are dense `uint8` export artifacts: compact analysis sources (`mask_bitpacked` or `mask_rle`) must be materialized through `MaskStore` into dense `subject_mask_runs/<run>/masks_roi`, with source encoding recorded in training provenance.

## Outside-Sandbox Validation Notes

- CUDA/GPU visibility may be unavailable in Codex sandbox even when available outside it.
- Run `scripts/py -m marimo check ...` outside the Codex sandbox by default. In the sandbox, marimo's checker can hang before diagnostics because its async filesystem path uses `asyncio.to_thread`; outside-sandbox execution has been observed to return normally.
- For CUDA checks, run outside the sandbox with `scripts/py` rather than `conda activate`, for example:
  `scripts/py -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'`
- Real-zarr training/export/inference smokes should run outside the sandbox with escalation, especially when they use CUDA or touch `/nvme1`.
- For U-Net subject-mask smoke validation, prefer the CUDA-capable outside-sandbox path over CPU sandbox execution. If the sandbox prints the startup banner but does not reach the artifact summary promptly, stop it and rerun outside the sandbox.
- When newly written Zarr groups are hidden during an active mutation, follow
  the Consolidated Metadata Read Policy above; do not use a stale consolidated
  view of a mutable archive.

## Examples

- `scripts/py --version`
- `scripts/py -m pytest`
- `scripts/py src/test_fisheye.py`
