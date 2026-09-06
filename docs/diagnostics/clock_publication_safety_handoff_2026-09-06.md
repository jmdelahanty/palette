# Clock-publication safety — 2026-09-06

## State and ownership

Local implementation and validation are prepared, **not merge-ready**.
This is the pre-publication implementation snapshot; the later draft PR records
the exact published candidate and live required checks. The user authorized
Palette-only implementation/review, commits, pushes, and CI on September 6.
Main merges, ruleset changes, installation, deployment, shared-checkout updates,
and production activation remain separate. Transfer-v2 adoption is not bundled.
Status belongs to `RID-CLOCK-001` in the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md).

- Owner: Palette clock-publication worker.
- Worktree: `/tmp/palette-clock-publication-safety-20260906`.
- Branch: `agent/palette/clock-publication-safety-20260906`.
- Initial implementation base/prerequisite: `3a85a8c9215894945685eaba6f0055730392db44`,
  [draft PR 143](https://github.com/jmdelahanty/palette/pull/143). That exact head
  had 23/23 successful checks at 2026-09-06 08:46 UTC; it was unmerged.
- At this pre-publication snapshot the six-path patch is uncommitted: the existing
  `src/fisheye/shared/acquisition_frame_clock.py`, new
  `tests/unit/fisheye/test_acquisition_frame_clock_publication.py`, regenerated
  `docs/diagnostics/zarr_production_writer_census.json`, this handoff, and the
  owning queue, plus temporary-log isolation in the existing two-session fixture.
  No other worktree's source or dirty files were integrated.
- Initial F3 implementation and parallel audit-doc reconciliation were local
  only. Subsequent preparation authorization permits commit/push/draft PR and
  full CI, but never main merge, deployment, registry/production mutation, or
  dependency installation. The next exact #143 prerequisite must pass all 24
  checks before it is integrated; the combined clock candidate then needs its
  own complete 24-check run.

Before implementation, relevant registered worktree heads, dirty scope, and
clock/queue ownership were checked. No dirty clock-source or test overlap was
found. The new branch deliberately preserves the stronger parser/frame-map
work from PR 143 instead of rebuilding it from the older audit checkout.

## Classification and preserved contracts

This is an **enforcement correction** to clock publication, not a scientific
schema, source-ID, or numerical identity migration. The existing family owns
clock validation; `selector_activation` remains the one selector-rollback
owner. Existing completion helpers, archive publication locking, and the
direct-tree consolidation helper are reused unchanged.

Preserved:

- All six array names, dtypes, values, row/frame identities, validity/sentinel
  semantics, source evidence, PTP interpretation, and camera clock semantics.
- `palette.acquisition_frame_clock.v1` record grammar and its SHA-256 bytes.
  The prerequisite golden digest remains
  `621faa07511174f0bbc02f0e6703430061b8710cebcb3c2451a7ed7ffc204771`.
- Strict public resolver payload checks, completion/eligibility requirements,
  matching selectors, and root bindings; no candidate-only validation path is
  exposed as a public reader bypass.
- PR 143's exact-integer/PTP classification and clipped frame-map corrections.
- Recording-time consumers' nominal-FPS numerical policy. No timestamps are
  copied into a second timing authority, and no hardware synchronization claim
  or new upstream/human-review gate is introduced.

## Implemented lifecycle and compatibility

The public `acquisition_frame_clock_source_sha256(source)` helper validates the
live source and uses this publisher's unchanged record/digest implementation.
It gives ingestion replay one maintained interface without exposing a private
record builder or adding a competing digest grammar.

1. Create an unoccupied public child with owner UUID and literal false
   eligibility in its first metadata operation. Shared lifecycle-helper attr
   operations freshly check that owner, including during failure cleanup.
2. Write and reload the exact payload before marking complete. Completion
   retains the parent epoch's provenance gate while eligibility remains false.
3. Use the existing owner/generation selector activation helper in deferred
   mode. Repeated proofs validate the same expected clock record and arrays.
4. Bind and reload the six root clock fields while the child remains
   ineligible; the public resolver refuses that intermediate state.
5. Commit eligibility as the final metadata operation. Return the previously
   validated result without a fallible post-commit resolver call. The shared
   helper handles a persisted final write whose acknowledgment raises.

The parent records the existing shared lease grammar under clock-specific
`acquisition_frame_clock_publication_{policy,generation,lease}` attrs, with
lease schema `palette.acquisition_frame_clock_publication_lease`, version 1.
The child owner attr is `acquisition_frame_clock_publication_owner_uuid`.
These are operational publication evidence, not scientific acceptance receipts
and not inputs to the clock record digest.

Failed owned public attempts remain failed, selector-ineligible tombstones
with no completion timestamp, failure details, and version-1
`palette.acquisition_frame_clock_publication_tombstone` evidence in
`acquisition_frame_clock_publication_tombstone`. Foreign replacements and
foreign selector/root mutations are not overwritten. Incomplete rollback is
reported as an error, not successful publication.

Root rollback restores only values belonging to the unique attempt reference
or its exact still-owned lease. The activation receipt alone owns selector
rollback. For an already-consolidated local archive, successful rollback of
the prior direct selection also uses the existing direct-tree consolidation
helper and verifies consolidated/default-reader visibility. This repairs the
observed Zarr 3.1.3 behavior in which mutable root attr writes discard inline
consolidation. It does not restore an unrelated older selector snapshot.

Retry naming is intentionally tightened:

- The first available name remains `acquisition_frame_clock_<digest[:16]>`.
- A repeat of the currently selected valid identical clock is read-only and
  returns the same identity, including legacy selected clocks without epochs.
- An occupied public name is never deleted, overwritten, or reinterpreted.
  Retry or reselection of an older source uses a new UUID-suffixed name with
  the same clock record/digest. Thus an execution path can change without
  changing logical clock content; consumers bind the actual returned path.
- Conflicting/tampered selected state is refused, not automatically repaired.
  Previously broken historical selectors require separately authorized repair.
- After a failed mutation, reopen the root with `use_consolidated=False`
  before retrying. A stale cached handle is refused rather than written back.
  A consolidated mutation handle is also refused explicitly.

This publisher operates inside an active ingestion archive. Successful
archive-wide consolidation remains the owning importer's final publication
step after its other payload and manifest writes. Failure visibility recovery
is implemented for local filesystem Zarr-v3 archives; synthetic in-memory
tests cover lifecycle mechanics, not remote-store publication. Hard process
termination and abandoned-lease administrative recovery are not made atomic
or automatically repaired by this exception-path correction.

## Validation evidence

Tests preceded the implementation: **17/17 new tests failed** against the
unchanged prerequisite, exposing unsafe write/loadback ordering, partial
metadata publication, missing failure cleanup, non-idempotent mutation, and
owner-loss behavior. Later real-storage tests found two additional failures
where direct rollback succeeded but consolidated visibility was lost; those
assertions were retained and the implementation corrected.

Final expanded selection: **277 passed in 25.41 s**, no failures or skips,
with 12 standard Zarr-v3 consolidation-specification warnings. This includes
52 new publication/interface cases, the original parser/digest preservation tests,
shared activation tests, real import -> resolver -> unmodified timing consumer
coverage, local filesystem failure/retry/consolidation, and the prerequisite
clipped frame-map/public producer tests.

```bash
scripts/py -m pytest \
  tests/unit/fisheye/test_acquisition_frame_clock_publication.py \
  tests/unit/fisheye/test_acquisition_frame_clock.py \
  tests/unit/fisheye/test_acquisition_frame_clock_validation.py \
  tests/unit/fisheye/test_selector_activation.py \
  tests/unit/fisheye/test_provider_recording_timing_authority.py \
  tests/unit/fisheye/test_import_recording_analysis.py \
  tests/unit/fisheye/test_import_organized_recordings_analysis.py \
  tests/unit/fisheye/test_recording_import_authority_integration.py \
  tests/unit/fisheye/test_two_session_four_camera_current_authority.py \
  tests/unit/fisheye/test_repair_missing_frame_clock_declarations.py \
  tests/unit/fisheye/test_create_clipped_training_zarr.py \
  tests/unit/fisheye/test_clipped_video_collection_frame_mapping.py \
  tests/unit/fisheye/test_create_clipped_analysis_zarr.py -q --tb=short
```

All pytest runs used workstation `scripts/py` outside the sandbox: Python
3.11.14, pytest 8.4.2, Zarr 3.1.3. Fixtures are temporary test data, not live
recordings or a hardware canary. Existing O(rows) array validation remains;
fresh proof passes add reads/hashes, and rollback reconsolidation traverses
archive metadata. No performance improvement or constant-memory claim is made.

The generated writer census was refreshed after the writer moved into its
locked implementation function: source line, enclosing symbol, discovered
receiver expression, derived census-site ID, and function count changed. No
runtime receipt or historical scientific digest was regenerated.

The two-session fixture now explicitly writes organizer logs beneath its pytest
temporary directory. Previous workstation runs used its inherited default
`/nvme1/recordings/logs`; no real recording payload or production registry was
changed, and existing logs were not removed. The fixture's identity, producer,
and registry assertions are unchanged.

Local source/test compilation, whitespace checks, generated census verification,
import-layer contracts, FPS/keypoint-motion/tail/paradigm access ratchets,
file-size and Zarr metadata-mode ratchets, observed-metadata checks, active
contract freshness, and the registry schema reference check passed. Existing
ratchet baselines were not widened or changed.

## Parallel documentation work

The separately owned draft remains at
`/tmp/palette-review-docs-rules-20260905`, branch
`agent/palette/review-docs-rules-20260905`, HEAD
`6af66c5a6ba3b35ea0bf00cfc74add7bb22da2b2`. Its handoff is
`docs/diagnostics/review_docs_rules_landing_handoff_2026-09-05.md`, section
“September 6 audit reconciliation.” It adds the completed instruction audit
and joint plan and narrows unsafe, outdated, and overstated audit claims while
preserving numerical evidence. The worker reported nine focused tests and
documentation/static gates passing, 86 valid local links across 21 touched
Markdown files, and two explicitly unavailable historical references. It did
not edit this queue, code, original-checkout reports, or the draft's unrelated
AGENTS/scripts/generated JSON. Its 30 dirty paths remain uncommitted; it was
not merged into this clock worktree. The coordinating owner reviewed its
handoff and recorded the dispositions in this branch's existing queue.

## Remaining gates

The implementation is local and validated in the stated scope; it is **not
committed, remotely CI-validated, integrated, deployed, or activated**.

Every required remote check for these changes is **unrun**:

- `generated artifacts`
- `import boundaries`
- `file-size ratchet`
- `zarr open metadata modes`
- `observed metadata literals`
- `active contract freshness`
- `package and collection`
- `non-gpu tests (shard 0)` through `non-gpu tests (shard 15)`
- `ci-required` after the planned validated #143 refresh

Local components do not substitute for those exact-commit required checks.
Full non-GPU CI, wheel/non-editable installed-package validation, full test
collection, and later combined-commit CI remain outstanding. Any authorized
commit/push/draft PR must identify itself as incomplete until every required
check succeeds; the prerequisite's green checks do not validate this patch.

Dispatch acknowledgment/recovery, crop-ledger integrity, parent-level clipped
intake, source-ID origin policy, hardware/clock equivalence, historical data
repair, broader consolidation, and production activation remain separate.
