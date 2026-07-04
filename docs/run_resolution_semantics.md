# Run resolution semantics

<!-- contract-meta
status: proposed
created: 2026-07-03
owner: jeremy
related: docs/palette_cli_narrow_waist_design.md,
         docs/identity_lineage_staleness_review.md
-->

## Why this exists

A stage can have many runs (`<stage>_runs/<run>`). "The current run" is asked by many
callers — the `palette` verbs, `status`/`plan`, downstream stages resolving their input,
training exports, registry inventory queries, the future `Recording` accessor. **Today
"current run" silently means different things in different places** (zarr completion
helpers vs registry inventory), which is the same ambiguity that produced the false-stale
bug (`docs/identity_lineage_staleness_review.md`). Before the `Recording` accessor is
built — whose entire job is resolving "which run" — the resolution modes must be **named
explicitly**, so a caller must *say which meaning it wants* instead of getting an
accidental default.

This is a design contract, not code yet. It defines the modes; the accessor and the verb
layer then take an explicit mode instead of an implicit one.

## The four modes

| Mode | Means | Answers | Backed by (today) |
|---|---|---|---|
| `AUTHORITATIVE` | the human-approved run; falls back to `LATEST_COMPLETE` if no pointer set | "what the science should use" | `resolve_authoritative_run_name` (`shared/zarr_run_completion.py:312`) |
| `LATEST_COMPLETE` | newest run carrying an on-disk completion marker | "most recent *finished* work on disk" | `resolve_latest_complete_run_name` (`:248`) |
| `INVENTORY_LATEST` | the registry's recorded latest for the stage (index view; may lag or lead disk) | "what the registry believes is current" | registry query paths |
| `SOURCE_MATCH` | the *specific* upstream run a given downstream artifact was built from, resolved by lineage (`source_*_run` / `source_crop_row_ids` attrs) | "the exact run this output derived from" (reproducibility) | `resolve_zarr_run` + lineage attrs (`shared/zarr_helpers.py:305`) |

### Semantics and when each applies

**`AUTHORITATIVE` — the consumption default.** Any path that feeds the *science* — a
downstream stage picking its input, a training export, `plan`'s "what to run next",
default reads through the accessor — uses this. It resolves the approved run if a pointer
exists, else the latest complete (so it is safe before anyone approves anything). This is
the mode that makes a later smoke run **not** hijack "current."

**`LATEST_COMPLETE` — recency, not authority.** For "show me the newest finished run"
regardless of approval — review UIs listing candidates, "did the run I just kicked off
finish." Must **not** be the default for consumption (that is exactly the false-stale
bug: latest ≠ what-to-use).

**`INVENTORY_LATEST` — the registry's index view.** For registry/inventory/maintenance
tooling that asks "what does the catalog say," which is deliberately distinct from disk
truth. Reconcile/audit tools use this *because* comparing it against disk is their job.
Never use it for consumption — the registry is a rebuildable index, not the source of
truth.

**`SOURCE_MATCH` — reproducibility / lineage.** For "which upstream run did *this*
artifact actually use," resolved by following the downstream run's recorded source
pointers, not by recency or approval. This is how you answer "reproduce exactly what
produced this output" and how a consumer verifies it is reading the same upstream a
downstream artifact was built against.

## The rule the accessor and verbs must follow

- **No implicit "current run."** Every resolution takes an explicit `RunResolution`
  value. The `Recording` accessor signature carries it:
  `rec.subject_masks(resolution=RunResolution.AUTHORITATIVE)`, defaulting to
  `AUTHORITATIVE` for the consumption case but never *hiding* the choice.
- **Consumption defaults to `AUTHORITATIVE`; inventory/maintenance opts into
  `INVENTORY_LATEST`; reproducibility opts into `SOURCE_MATCH`; recency opts into
  `LATEST_COMPLETE`.** This is the same consumption-vs-inventory line drawn in the
  authoritative-run-pointer slice's call-site classification — `RunResolution` names it as
  a type instead of leaving it to convention.
- **One resolver surface.** Introduce a single `resolve_run(parent, mode, *, ...)` (or a
  small `RunResolution` enum + dispatch) that wraps the four existing functions, so there
  is one place the modes are defined and one place a new mode would be added. The existing
  functions become its implementations, not competing entry points.

## Migration note (not this slice)

The existing ~40 `resolve_latest_complete_run_name` call sites (inventoried in
`docs/diagnostics/authoritative_run_resolution_callsite_inventory_2026-07-02.md`) already
split along the consumption-vs-inventory line. Reclassifying them onto explicit
`RunResolution` modes is incremental and can follow the accessor — the point of naming the
modes now is that the accessor is *built* against them from the start, rather than baking
in one meaning and having to unpick it later.

## Concrete reconciliation case: detect review authority (found 2026-07-04)

Detect review state is currently split across **two parallel parent-level "which run to
use" pointers**, written by different paths:

- `detect_review_status_latest` (parent attr) + `detect_review_status` (per-run verdict
  payload) — the **established** mechanism. Propagated through crop→keypoint lineage
  (`tracking/crop.py:793,1965`), and written both by the review backend and by
  `utils/backfill_detect_review_status.py:161` — the latter a writer that **bypasses
  `approve()` entirely** (no fail-closed guarantee, no authoritative pointer).
- `authoritative_run` (parent attr) — the **new** mechanism; the fail-closed approval
  slice (2026-07-04) wired detect review to set it.

These overlap: both answer "which reviewed detect run should downstream use." Today crop
does not yet resolve its detect input via `authoritative_run` — so the new pointer detect
review sets is **forward-looking**, not yet consumed. The reconciliation (part of the
`Recording` accessor / RunResolution work, not a standalone patch):

1. Make `authoritative_run` (resolved via `AUTHORITATIVE`) the single run-selection
   pointer that crop and other consumers use for their detect input.
2. Keep `detect_review_status` as the per-run *verdict*, feeding the pointer, not as a
   second selector; retire or subsume `detect_review_status_latest`.
3. Reconcile `backfill_detect_review_status.py` to set the authoritative pointer (or
   retire it) so no writer bypasses the fail-closed approval path.

This is a design reconciliation, not a utility-chase — it belongs in the accessor slice
where "which run does the pipeline consume" becomes an explicit `AUTHORITATIVE`
resolution.

## Non-goals

- Not changing what any existing resolver returns — this names and unifies them.
- Not building the accessor here — this is its precondition.
- Not reclassifying all call sites here — that is a follow-up, guided by the modes named
  above.
