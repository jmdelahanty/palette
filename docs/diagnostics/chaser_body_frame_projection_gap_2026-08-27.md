# Chaser body-frame projection gap (2026-08-27)

## Decision

Project body-frame observations onto the chaser relative-frame axis only by
exact `acquisition_frame_id` identity. A relative acquisition frame with no
exact body-frame observation is **missing-source-row evidence**. A frame with
an exact body-frame observation whose `axis_valid` value is false is
**present-but-invalid-axis evidence**. These states are distinct and must not
be collapsed into a generic "uncovered" category.

Neither state may be filled by interpolation, velocity or motion heading, a
neighboring anatomical frame, or another position provider. Position-only
distance, occupancy, and trajectory measurements remain independently usable
where their own validity contracts pass.

The body extension must persist, for every flattened frame-by-chaser row:

- the exact body-frame source-row identity and whether it is present;
- anatomical origin and forward/left axes only when valid;
- body heading and body-relative chaser coordinates only when valid;
- separate source-row, anatomical-axis, and derived-bearing validity; and
- a reason code that distinguishes source-row absence from invalid or
  nonfinite anatomical geometry.

## Why the current publication lacks the extension

The audited recording already contains the complete selector-ineligible body
frame
`analysis/body_frame_runs/body_frame_goodbatbadbat_pose_384_20260816_v1`.
It derives the anatomical forward axis from swim bladder to eye midpoint. Its
manifest payload digest is
`1df2f9a62aac4caa9984ae0d922cc313cb96f652e3de666e6e2b74af4e5742a5`.

The provider-motion publication records profile
`explicit_position_body_frame.v1` and proves exact ordered `instance_key`
equality between that body frame and
`position_goodbatbadbat_keypoint_triad_talk_20260818_v1`. Therefore the gap is
not missing anatomical data or a discovered position/body row conflict.

The gap is in chaser composition:

- `chaser_proxy_relative_frame_adapter.py` currently constructs
  `ChaserRelativeFrameInput` with `body_frame=None`;
- the resulting exact relative-frame manifest declares
  `body_extension_present: false` and `source_authorities.body_frame: null`;
  and
- the cohort runner invokes the generalized successors with
  `--no-body-extension`.

The existing position-only products are correct for their declared inputs.
They are not body-relative products and must remain immutable.

## Audited-recording coverage

Read-only exact-array comparison used recording
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` and relative-frame run
`chaser_relative_frame_keypoint_triad_cohort_20260825_exact_trials_session_time_activity_orthogonal_v3`.

| State | Acquisition frames | Fraction of 149,936 frames |
| --- | ---: | ---: |
| Exact body row and valid anatomical axis | 148,091 | 98.77% |
| No exact body-frame source row | 557 | 0.37% |
| Exact body row, invalid anatomical axis | 1,288 | 0.86% |

The 557 missing-source-row frames occur in 86 discontinuous ranges. The
largest include `5567-5660` (94 frames), `95750-95812` (63), `9338-9379`
(42), `126095-126133` (39), `14053-14088` (36), and `148717-148745` (29).
These are sparse internal holes, not an uncovered recording tail: the body
source spans acquisition frames 0 through 152,034, while the relative source
spans 1,019 through 151,018.

Because the relative product has two chasers, the projection would retain
1,114 flattened rows with no body source and 2,576 flattened rows with a
present but invalid anatomical axis.

All 1,445 bout onsets have a valid body frame. No onset bearing would be lost
to body coverage. At bout end, 196 bouts have an invalid anatomical axis (392
bout-by-chaser rows), so their whole-bout directed-turn result must remain
invalid. Among 7,961 exact logged active trial-member rows, none lacks a body
observation and one has an invalid anatomical axis.

These counts describe this one immutable recording and must not be generalized
to the cohort. Every successor publication must recompute and seal its own
coverage evidence.

The first strict read also exposed a portability defect in validation of the
derived `heading_deg` cache. Recomputing the cache from its authoritative
`float32` forward axes on this workstation differed from the stored cache on
52,084 valid rows, but every difference was at most
`0.000030517578125` degrees and none exceeded `0.0001` degrees. The body axes,
sign convention, and invalid-row behavior still validated. Body-frame v1 now
accepts only an absolute `0.00005` degree reproduction tolerance for this
derived cache, matching the existing downstream relative-frame validator;
relative calculations continue to derive from the digest-bound axes rather
than treating the scalar cache as authority.

## Safe implementation boundary

The first implementation should bind the explicit keypoint body-frame source,
reuse or reproduce its exact keypoint-position compatibility proof, project it
onto relative acquisition frames without interpolation, and publish a new
immutable body-extended relative-frame run. The generalized bout successor can
then run without `--no-body-extension`.

Before those body-relative results are treated as reviewed scientific
evidence, the chain must bind an existing applicable anatomical-orientation
review receipt or produce a chaser-specific one. A complete body-frame
manifest and exact row compatibility do not by themselves establish that
review claim.

Detection-centroid position plus a keypoint-derived body frame is a separate
mixed-provider composition. It must remain unavailable until an explicit
coordinate, frame, row-identity, review, and provider-disagreement contract is
bound; it must not silently inherit the keypoint-position proof.

## Structural implementation checkpoint (2026-08-27)

The exact keypoint composition is implemented as a new selector-ineligible
successor path. The adapter reopens the keypoint position source named by the
proxy, requires every digest-bound position field to match, composes it with
the explicitly named body-frame run, and projects onto the relative axis only
by exact acquisition-frame identity. The dry-run operator exposes the sealed
coverage record alongside its digest.

The full-recording no-write canary reproduced the audited counts above and
prepared manifest
`9506ab2e3920c393f047156a0848dff3afe02d20b748f07bea722002ad596254`.
Its body projection record is
`f6eec26ac739a30967db081208e968dcf28e242b67038fdb8ff84641483ae73f`
and retains `scientific_review_disposition=unreviewed_structural_candidate`.

The versioned cohort planner now resolves each body frame from the exact
provider-motion authority, freezes the body metadata and manifest digests as
worker inputs, passes the body source only to the keypoint relative-frame
operator, and runs the generalized bout successor without
`--no-body-extension`. A no-write replan of all 84 frozen GoodBatBadBat
recordings succeeded with task digest
`6cc1c46ec7372627794a66c06e0eab6c31faf6d8b3c3bd0499fae6eeb0ad0d02`.
Eighty-three recordings bind
`body_frame_goodbatbadbat_pose_384_20260816_v1`; the exact canary motion source
for one recording binds
`body_frame_goodbatbadbat_pose_384_heading_successor_20260818_v2`. No job was
submitted and no recording was modified during this replan.

The derived receipt-bound execution task is
`/tmp/goodbatbadbat_chaser_body_frame_receipt_bound_task_20260827.json`, with
task digest
`bbbcc9710d38c6bd5c0a611bc68b40c24ef908f1faaf93913b26b914a0256509`.
That `/tmp` file is diagnostic planning evidence, not yet a durable submission
handle.

This checkpoint establishes structural computation and immutable planning,
not reviewed scientific authority. The anatomical-orientation review receipt,
required CI, a commit-pinned deployment, and a controlled publication canary
remain gates before broader execution.

This diagnostic changes no selector, registry row, production authority, or
historical analysis publication.
