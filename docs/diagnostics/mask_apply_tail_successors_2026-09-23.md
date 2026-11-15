# Mask Apply tail successors

Owner: `/root`. Worktree `/tmp/palette-mask-tail-apply-versioning-20260923`,
branch `agent/palette/mask-tail-apply-versioning-20260923`. The prerequisite
is the deployed preview commit `f772816a10ad1bc97b42e14fa986665e7aab5d23`,
whose required CI passed in
[run 35787456627](https://github.com/jmdelahanty/palette/actions/runs/35787456627).
The independently owned mask QC candidate is
`agent/palette/review-apply-performance-20260923`; integration must wait for
required CI on both incoming exact commits and then validate their combination.

## Behavior and preservation

The user chose to regenerate automatic tail landmarks after mask corrections
and preserve manual edits. This is a new, named publication policy:
`new_mask_seed_and_review_version_preserve_recorded_manual_points_v1`, with
source proof schema `palette.training.mask_apply_tail_successor.v1`. It retains
the existing recovered/native 18/19-landmark scientific recipes, coordinate
systems, row identities, crop images, and initial-contract digest grammar.

The publisher snapshots the edited dense masks and uses the maintained recovery
producer and atomic child publisher to create five new run children: crop,
immutable mask source, editable mask draft, immutable keypoint seed, and editable
keypoint draft. It never overwrites old runs or moves selectors. The source
proof binds the Apply ID, current mask revision, saved keypoints, manual flags,
origin codes, reason labels, row identities, original source bindings, and QC
parameters. Producing-code provenance uses the existing writer receipt.

Automatic tail stations and the automatic snout are regenerated. Current head
landmarks supply orientation; existing native head/fin/snout landmarks remain
bound to their source. Per-landmark manual edits override regenerated points,
including explicit NaN clears. The immutable seed and editable initial snapshot
have distinct content hashes. Operator reason tags survive; stale derivation
and geometry statuses are recomputed. Training eligibility requires a valid
automatic tail, all landmarks finite and inside the crop, and the existing
keypoint QC. This does not approve labels or activate training authority.

The browser adapter applies only to the supported training-review schemas and
body/swim-bladder mask tasks. It requires exactly one paired pose source and
refuses unapplied pose checkpoints or pending pose Apply effects. It offers
separate successor tasks rather than retargeting current tasks or closing
sessions. Failed rows carry ROI, source frame, and failure reason. An opened or
completed successor task is not reset by retry. New edits on an older task stay
on that older version; this change does not merge independent annotation drafts.

The caller holds the existing refined mask lock; the publisher additionally
holds the archive publication lock, which serializes canonical keypoint Apply.
Source revalidation guards publication. A same-source retry resumes partial
publication or reuses a completed version without replacing subsequent draft
edits. An Apply ID cannot silently bind different source labels after publishing
its first crop child. A conflicting source requires recovery, not an automatic rebase.

## Validation and cost

Real-Zarr tests run outside the sandbox, with explicit unconsolidated access to
mutable groups. Coverage includes recovered and native data, manual points and
clears, reason/confidence preservation, failed tail eligibility, unchanged
historical arrays/selectors, wrong schema/recipe/identity/revision/origin,
interrupted publication, changed sources, safe retry, and real SQLite task and
checkpoint behavior. The editor reopens the published versions through its
normal resolver; readback uses the final consolidated generation.

The full tail suite passed 28 tests in 95.42 seconds. Final strengthening of the
first-child retry binding is additionally checked by the four recovered/native
partial-publication and changed-source tests. Import boundaries, authority
access ratchets, metadata-mode policy, generated Zarr inventories, file-size and
contract checks also passed locally. These checks do not substitute for CI.

A private copy of the frozen 184-row recovered archive generated 184 valid
tails, retained 758 manual landmarks, and left the old mask and pose directories
byte-identical. Local elapsed time was 35.13 seconds, excluding the input-copy
setup and browser/registry/QC work. Evidence is under
`/tmp/palette-review-apply-validation-20260923/tail-refresh-smoke.json`.
The two final source-refusal/confidence checks were added after that smoke;
the final test report records the exact subsequent validation.

This reuses the recovery producer's whole-dataset staging and copies crop/mask
payloads into each version. It is intended for these small training archives;
it is not a streaming writer for arbitrarily large recordings. It adds real
derivation, copying, and validation cost to Apply. A later optimization must
preserve immutable source bindings and initial-content validation rather than
claiming freshness from revision markers alone.

## Integration and deployment status

The new publisher and adapter are separate from the browser route in this
candidate. After both candidates pass required CI, wire the adapter after mask
QC and before audit/registry/secondary-effects completion in both the initial
Apply and same-ID retry paths. Validate supported HTTP routes and the combined
commit. The receipt must remain pending if regeneration or publication fails.

No live dataset, SQLite store, task, session, preview process, production selector,
or shared `/groups` checkout has been changed by this implementation. Required
CI, route integration, and deployment remain separate gates; local test success
is not a merge-ready or deployed claim.
