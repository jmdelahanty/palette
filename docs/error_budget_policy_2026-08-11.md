# Palette error-budget policy

<!-- contract-meta
version: 1
status: draft
last_verified: 2026-08-11
implementation: specified-only
-->

Purpose: convert "how careful should we be" from an open-ended obligation into
a bounded, measurable allocation. Every SLO below has (a) a target, (b) a
mechanical indicator (a command, not an intention), (c) an exhaustion policy —
what happens when the budget is blown. Targets marked (p) are provisional
first guesses; the mechanism is the point, and numbers get recalibrated at
each quarterly review. 100% is a defect everywhere except Tier 0: budget spent
past the knee of the cost curve is stolen from science.

Guiding asymmetry: prevention is priced by tier; DETECTION LATENCY is priced
globally. A detected error is cheap at any severity; a silent error is
expensive at any size. When in doubt, buy detection, not prevention.

---

## Tier 0 — Irreversible (acquisition, primary pixels, publication integrity)

Zero error budget. This is the only tier where "what's better than 0" is the
right question, because retrofit is impossible.

| Surface | SLO | Indicator | Exhaustion policy |
|---|---|---|---|
| Acquisition metadata completeness on NEW recordings | 100% pass required-field import profile at import time | `check_import_profile` (required_missing == none) | STOP THE LINE: no further acquisition sessions until fixed; the failed recording is quarantined, never silently repaired |
| Pixel identity (color range, pixel contracts) on new imports | 100% observed-not-guessed (`*_observed` fields from ffprobe, never literals) | `audit_zarr_pixel_contracts`; grep-gate on hardcoded `*_observed` literals | Same as above |
| Primary-artifact publication (crops, masters): complete-or-tombstone, never partial | 100% of publications end in exactly one of {complete, tombstone} | zarr run-completion validation; publication receipts | Stop-the-line for the writer involved |
| Stimulus provenance on new protocols (schedule mode, seed, realized events) | 100% present per the chaser-schedule contract | importer QC (planned-vs-realized cross-check) | Recording flagged unusable-for-training-analyses until resolved |

Detection-latency target: same day (all Tier 0 indicators run at import/publication time, not in batch audits).

## Tier 1 — Expensive to redo (training data, refinements, registry, provenance chain)

Small, explicit budget. Errors here cost weeks, not years; they are permitted
in bounded, detected, attributed form.

| Surface | SLO (p) | Indicator | Exhaustion policy |
|---|---|---|---|
| Run provenance on newly finalized runs | ≥99% carry valid `run_provenance`; every bypass has a written reason; bypass count ≤5/quarter | `check_provenance_capture`; bypass-count query (W2.4 index when it lands) | Next agent wave is remediation, not features |
| Registry↔zarr consistency | Reconcile-sweep divergence <1% of rows; ledger freshness: newest `recording_step_status` row ≤7 days old while pipeline active | `reconcile_sweep --dry-run`; freshness query. NOTE: the 2026-06-18 ledger freeze went unnoticed for 7 weeks BECAUSE no indicator watched it — the freshness check is the lesson | Same |
| Stage-array/contract hard enforcement | Ratchet: enforced-stage count only increases (currently 7/25); no publishing stage added without shadow telemetry | `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` diff in review | Blocked additions queue for next wave |
| Training-data lineage (label origin, model hashes) | 100% of NEW training runs carry model content hash + dataset ids; historical backfill NOT owed | training_runs query | New-run gap = remediation wave; historical gaps = logged, not owed |
| Tier-1 review findings half-life | Confirmed Tier-1 findings fixed or explicitly waived within one quarter | fix-queue doc status | Quarterly review decides: fix, waive with reason, or re-tier |

## Tier 2 — Redoable (analysis interiors, viewers, tooling, docs)

Generous, explicit budget. Correctness is guaranteed ONLY at contracted
boundaries; interiors are allowed to be scrappy BY POLICY.

| Surface | SLO (p) | Indicator | Exhaustion policy |
|---|---|---|---|
| Contracted export boundaries (Arrow tables, cohort manifests, group stats) | Boundary columns present + honest: denominators, speed_source, geometry status, real FDR families | export contract validation; boundary tests | Fix before next export generation; exports are consumer-facing = boundary, not interior |
| CONSUMER-VISIBLE dashboards/figures | Must not display metrics from disavowed sources (raw-centroid speed class). A dashboard is a boundary, not an interior — the raw `immobile_fraction` incident defines this rule | speed_source/geometry-status propagation checks | Remove or label the surface within days; no obligation to fix the interior |
| Analysis interiors, one-off scripts, viewers | NO SLO. Deliberately. Scrappy permitted; honesty required (`implementation:` labels) | none — that is the point | n/a |
| Docs | Freshness enforced ONLY for `status: active` contracts (≤90 days); everything else may rot if labeled `draft`/`superseded`. Target: ~20 active contracts, not 115 | `check_contract_freshness` (once vocabulary fixed, W3.2) | Demote to draft/superseded rather than update — subtraction is compliance |

## Cross-cutting — the enforcement layer itself

The July outage is the defining incident: enforcement silently off for two
weeks, three gates down together, drift accumulating. The enforcement layer
gets its own SLO because every other budget is unmeasurable when it is down.

| Surface | SLO | Indicator | Exhaustion policy |
|---|---|---|---|
| CI on the integration branch | Green is the resting state; red age ≤48h | `gh run list` | RED >48h = GLOBAL FEATURE FREEZE: every agent brief pauses except the fix. This is the error-budget policy with the most teeth, and it is the one that was missing in July |
| Gate liveness | lint-imports, ratchet, collect, freshness each demonstrably RAN this week (not just "no failures") | CI step visibility (W0.3 independence) | Same as CI red |

---

## What the budget is spent ON (the point of having one)

Budget exists to be spent. Current authorized spends:

- Ship the escape/pursuit manuscript with enumerated caveats (known geometry
  audit pending, habituation underpowered) rather than after exhaustive
  remediation.
- Interior `ArrayContract` migration (W4.4) proceeds opportunistically, not as
  a blocker.
- ~400 bare `zarr.open_group` sites in analysis interiors: ratcheted (no new
  ones), not mass-fixed.
- Historical backfills (Bernoulli params, training lineage, eye-mask rows):
  logged, explicitly not owed.
- Tier-2 doc corpus: demoted, not repaired.

## Review cadence

Quarterly: recalibrate (p) targets against incidents; retire indicators nobody
consulted; add indicators for any error that reached Tier 0/1 undetected. An
error that occurred WITHIN budget and was detected within target latency is a
SUCCESS of this policy, not a failure of the operator.
