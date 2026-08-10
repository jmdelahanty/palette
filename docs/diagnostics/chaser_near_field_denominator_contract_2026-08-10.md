# Chaser near-field entry-rate denominator contract

Date: 2026-08-10

Status: implemented as a selector-ineligible schema candidate; downstream
profile/export/viewer allowlists must adopt v2 before production activation.

## Maintained contract

`palette.chaser.near_field_occupancy.v2` / method version `3` defaults to
`valid_tracked_observed_transitions_v2`. For each phase and chaser:

- an entry is counted only after a valid sample above `r_out_mm` establishes an
  observed outside state and a later valid sample falls below `r_in_mm`;
- the numerator is persisted in both `near_zone_entry_count` and the explicit
  `near_zone_entry_rate_numerator_count` array;
- `near_zone_valid_tracked_duration_s` is the number of valid finite
  fish-to-chaser distance samples divided by FPS;
- `near_zone_entry_rate_denominator_duration_s` equals that valid tracked
  duration under the maintained policy;
- `near_zone_entry_rate_per_min` is
  `60 * numerator / denominator_duration_s`, and is `NaN` when the denominator
  is zero.

The duplicate explicit numerator is intentional: consumers can verify the rate
without inferring which historical count array was used.

## Gaps and censoring

An invalid sample starts an invalid gap and resets transition state to unknown.
If a visit was active, its duration is censored. A later valid sample that is
already inside the zone is also censored; it cannot be counted as a new entry.
Only a later sample above `r_out_mm` re-arms entry detection. Therefore tracking
dropout cannot split one biological visit into two observed entries.

A phase beginning inside is left-censored and is not an observed entry. A phase
ending inside retains any entry that was actually observed, but the incomplete
dwell is excluded from complete-visit dwell summaries. Median and total visit
dwell include only visits with an observed entry and exit within one contiguous
valid interval.

The component persists:

- `near_zone_invalid_gap_count`;
- `near_zone_censor_event_count`;
- `near_zone_boundary_censor_event_count`; and
- `near_zone_invalid_gap_censor_event_count`.

The config, component parameters, diagnostics, summary, and per-chaser group
attributes carry the exact visit-policy, denominator, invalid-gap, boundary-
censor, and complete-dwell semantics.

## Historical reproduction

Existing v1 components retain their historical meaning. New computation uses
v2 by default. Reproducing the old behavior requires the explicit
`legacy_epoch_gap_split_v1` policy (CLI:
`--visit-policy-version legacy_epoch_gap_split_v1`). That policy closes a visit
at every invalid sample, permits a resumed inside sample to start another
visit, and divides entry count by full phase duration. It still publishes its
actual valid tracked duration separately from its selected rate denominator so
the discrepancy is visible.

The maintained default component name advances from
`chaser_relative_near_field_v2` to `chaser_relative_near_field_v3`; immutable
historical children are never overwritten.

## Validation

Focused tests cover:

- dropout inside a visit without synthetic re-entry;
- left- and right-boundary censoring;
- all-invalid phases with a zero denominator and `NaN` rate;
- exact valid-time rate arithmetic;
- explicit legacy reproduction; and
- physical persistence of numerator, denominator, censor arrays, and policy
  attributes.

Before selector activation, integration must update the chaser workflow profile,
surface-classification catalog, analytics exporter schema map, and visualization
schema allowlist from near-field v1 to v2. That compatibility wiring is outside
this isolated science-correctness lane; activating v2 without it must fail
closed.
