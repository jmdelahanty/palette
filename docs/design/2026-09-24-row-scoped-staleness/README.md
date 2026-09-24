# Row-scoped staleness and stable row identity

- **Status:** draft
- **Owner:** labeling/Apply work (session palette-12, for the user); last reviewed 2026-09-24
- **Builds on:** `docs/manual_add_row_propagation_design.md` (append-only rows,
  per-row `pending_generation`, row-set digest, reconciler) and
  `docs/mutable_review_runs_contract.md` (mutable review surfaces; deletions
  are tombstones). This note does not repeat them; it adds the rule that
  connects them.
- **Related:** `docs/design/2026-09-24-background-apply-effects/README.md`

## The question

When something upstream changes, which downstream products are wrong, and how
much of each?

Today the answer is usually "the whole run". Staleness is recorded mostly as
run-level flags (`metrics_stale`, `contours_stale`, `derived_mask_caches_stale`)
and spread across three systems that don't coordinate (see "Staleness is three
uncoordinated systems" in the add-row design). So a one-row mask edit makes the
whole run's QC stale and triggers whole-run recomputation. A deleted or added
detection is hard to reason about because nothing says how far its effect
spreads.

## Two different kinds of upstream change

**1. Content change: the row stays, its values change.** For example a
corrected mask, moved keypoints, or a relabeled reason. Row identity is
unchanged, so the effect is local, and how far it spreads depends on the
downstream product (below).

**2. Identity change: rows appear or disappear.** Downstream arrays are aligned
to upstream rows by index. Removing a row in place shifts every later row, so
every aligned product is wrong at once. This is what makes upstream deletions
feel catastrophic. The rules below remove it as a failure mode.

## Rule 1: row identity never shifts

- **Removal is a tombstone,** never a physical delete. A deleted detection
  becomes `valid = false` or `decision = removed` in place, as the
  mutable-review contract already specifies. Indices stay stable, so a removal
  is a content change (kind 1) with a local footprint.
- **Additions append at the tail** with a fresh stable ID, as specified in the
  add-row design. Existing indices never move.
- **An observation row set never changes in place** (`detect_runs`,
  `crop_runs`, raw model output). A changed observation row set is a new run,
  and it gets a new row-set digest.

With these rules, the only change that makes a whole downstream product stale
is a **new upstream row set**. That is explicit, versioned, and detectable by
digest, rather than a silent shift.

## Rule 2: every derived product declares its change footprint

Each derived product states, in its contract, how far one changed source row
reaches:

| Footprint | Meaning | Examples |
|---|---|---|
| `row` | Output row *i* depends only on source row *i* | per-row mask metrics, contours (sampled and full), eye ellipses and separation, tail seed derived from a mask row |
| `window(w)` | Output rows *i±w* frames depend on source row *i* | smoothed kinematics, bout detection, speed filters |
| `run` | Any source row can change any output row | tracking linkage, arena fits, percentile QC grades, dataset splits, training exports |

The footprint is a property of the computation, recorded next to the
product's method and version. A product that doesn't declare one is treated as
`run`, which is today's behavior. That keeps adoption incremental and safe.

## Rule 3: freshness is pulled per row from source revisions

Instead of push-style stale flags, each derived row records which source it
was computed from:

- `source_row_revision[i]`: the source row's revision (the per-row
  `row_revision` that refined mask components already keep), for `row`
  products;
- the source window's revisions, or their maximum, for `window` products;
- the source run's `edit_revision` and row-set digest, for `run` products.

**A derived row is fresh when its recorded source revision equals the source's
current revision, and its row-set digest matches.** This is a comparison of
small arrays, not a recompute. It gives:

- **Row-scoped refresh:** recompute only the rows where the comparison fails.
  Row-scoped QC is this rule applied to `row` products.
- **One mechanism instead of three:** the registry, stale markers and lineage
  fingerprints can all derive their answer from the same comparison, instead
  of each being told separately.
- **No lost signals:** a crash between an edit and its stale-mark can no longer
  leave something marked fresh. Freshness is computed, not stored.

A run-level flag like `metrics_stale` becomes a derived summary ("any row
stale?") rather than a separate truth.

## What this means for the cases that worry you

- **A detection is deleted:** it's tombstoned. For `row` products, only that
  row's derived outputs change; they become "not applicable, source invalid".
  For `window` products, only frames within *w* of it. For `run` products, the
  whole product is stale, as it truly is: tracking really must re-link.
- **A detection is added:** it's appended. For `row` products, the new row is
  `pending_generation` (add-row design). Existing rows are untouched.
- **Detection is re-run and produces a new row set:** a new run with a new
  digest. Every downstream product built on the old digest is stale as a whole
  and says so explicitly. This is the one expensive case, and it's explicit.

## What must be true before relying on this

- **Footprints must be correct.** A product declared `row` that secretly uses
  neighbors gives wrong freshness. Every footprint declaration needs a test
  that edits one source row and asserts only the declared rows change, run on
  real data as well as fixtures.
- **Keep a full-recompute backstop:** a periodic or at-promotion full
  recompute that must match the row-scoped result byte for byte.
- **Some consumers still use physical order.** Consumers that scan by physical
  row order must use the frame-index lookup from the add-row design.

## Adoption order

1. **Mask QC** (`row` footprint). Switch browser QC to the sampled contours
   analytics already uses, which are row-aligned, and add
   `source_row_revision`. This is the row-scoped QC follow-up from the
   background-effects design.
2. **Tombstone audit:** find every path that physically deletes rows from a
   review surface and convert it to a tombstone.
3. **Keypoint-derived and tail products** (`row`).
4. **Kinematics and bouts** (`window`), then registry step status derived from
   freshness instead of stored separately.

## Decisions needed

1. Adopt the footprint vocabulary (`row`, `window(w)`, `run`), with undeclared
   meaning `run`?
2. Move freshness from stored flags to computed per-row revisions, keeping
   run-level flags as derived summaries?
3. Start with mask QC as the first adopter?

## Decision log

- 2026-09-24: opened from a discussion of why a one-row mask edit makes whole
  runs stale, and how upstream deletions propagate.
