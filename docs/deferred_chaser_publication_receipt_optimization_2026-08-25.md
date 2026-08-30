# Deferred chaser publication-receipt optimization — 2026-08-25

## Goal

Eliminate the remaining one-time post-publication rehash used to create
reusable exact-child validation receipts. The current receipt-bound consumer
path already removes repeated archive-root consolidated-metadata parsing for
relative-frame, semantic-selection, composable-successor, radial/near-field,
and spatial-occupancy children. Consumers rehash the arrays they load; initial
receipt creation independently streams every manifest-declared array. This
deferred step would let each immutable producer emit the same evidence while
those arrays and their content digests are already in hand.

## Consumer-side progress — 2026-08-30

The interactive reader now has a separate, implemented optimization boundary.
One `palette.analysis.exact_chaser.projection_receipt` v1 record composes the
independent exact-child and relative-frame receipts by canonical path and
digest. The selected renderer revalidates direct metadata once per consumed
child and rehashes only arrays it actually displays. The composition record is
not a selector, production authority, cache authority, or replacement for the
lineage-specific receipts.

This does not complete the producer optimization below. Existing cohorts still
pay the one-time post-publication receipt-creation scan; future producer-bound
receipt emission can remove that cost without changing the reader contract.

## Proposed successor

Add a versioned common immutable-child publisher/finalizer that:

1. computes every declared array content digest during the existing staged
   write/validation pass;
2. completes the direct and consolidated child audit;
3. atomically publishes the selector-ineligible immutable child;
4. after final completion metadata exists, seals an external reusable
   validation receipt bound to the exact archive, child path, manifest,
   completion owner, direct metadata generation, and Palette commit; and
5. returns that receipt path and digest in the publication receipt without
   making it a selector or scientific authority.

The producer-authored receipt must have the same or a strictly versioned
stronger validation semantics as the applicable existing receipt:
`palette.analysis.chaser_relative_frame.reusable_validation_receipt` v1 or
`palette.analysis.exact_immutable_child.validation_receipt` v1. It must not be
synthesized before final completion metadata is durable.

## Acceptance evidence

- A cold benchmark must compare post-hoc receipt creation with producer-emitted
  receipt creation and record wall time, bytes read, peak RSS, metadata mode,
  filesystem, and cache state.
- An independent validator must rehash every declared array for at least one
  representative recording and obtain exactly the producer-sealed digests.
- Mutation tests must reject changed array chunks, child metadata, provenance,
  completion ownership, archive/run identity, recording identity, and software
  commit.
- Targeted consumer tests must prove archive-root metadata is never parsed on
  receipt reuse and that only requested arrays are opened and rehashed.
- The receipt must remain selector-ineligible, production-ineligible, and
  incapable of changing registry or selector state.
- Required CI and a commit-pinned high-metadata canary must pass before the
  producer-emitted path replaces post-hoc receipt creation in cohort jobs.

## Out of scope

This optimization does not weaken initial scientific validation, make mutable
children eligible for receipt reuse, treat a task/worker operational receipt as
scientific evidence, or change consolidated metadata lifecycle policy. It also
does not replace the exact plot-recipe provenance stored with every generated
figure bundle.
