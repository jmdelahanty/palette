# Subject-Mask Full-Duration Canary

Date: 2026-07-31

Status: interrupted by a cluster-wide LSF execution-host outage during
refinement; incomplete and not promotion evidence

## Immutable execution

- Palette commit: `73f7bb5e9bf840f7a5ce697857f8130a6115bff4`
- Plan digest:
  `0fb4451df97b871f6eb187becf053305bf9d1139b817ef7d24e063306fc559a8`
- Run root:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/subject_mask_storage/full_duration/sleepyfish_cam2010095_20260731_73f7bb5e`
- Inference array: `153237178`
- Refinement array: `153237179`
- Recording publication: `153237180`

The plan covers 22 real clip windows, 1,188,000 acquisition frames, and
1,169,010 crop/keypoint rows. It is selector-ineligible and prohibits registry,
authority, and production-path mutation.

## Completed inference evidence

All 22 inference partitions completed successfully before the interruption.
Their ordered row intervals cover `[0, 1,169,010)` exactly. The 22 worker
bundles have distinct scientific-identity, attempt, semantic-receipt, and pixel
work-package identities. Every run reports `cuda:0`, completion status
`complete`, and `stage_selector_eligible=false`.

The workers processed 308,075,820,594 source-video bytes and materialized
306,448,957,440 logical ROI-cache bytes. Median per-window cache
materialization was 171.5 rows/s. Median U-Net inference was 82.4 rows/s.
These are correctness-canary timings: this driver uses one decoder per GPU task
and does not exercise the maintained production DAG's separately benchmarked
eight-session NVDEC cache bundles.

## Interruption evidence

Two refinement partitions completed before the outage, covering 107,359 rows.
They contain distinct scientific, attempt, and semantic-receipt identities,
dense editable `masks_roi`, sampled component contours, and no finalizer
warnings.

At approximately 10:47 EDT, active refinement elements 2, 3, 5, and 6 changed
from `RUN` to `UNKWN`. LSF reported both execution hosts, `h07u21` and
`h07u29`, as `unavail`; many unrelated cluster hosts were unavailable at the
same time. This is an infrastructure interruption, not a Palette validation
failure.

The outage caught two workers during atomic bundle copy. Only these hidden
temporary destinations became visible:

- `.clip_000001.291ee5f13f5640cdaea0766e3be7faca.tmp`
- `.clip_000004.66f7ee23d8b848cfbbdee71bbdc0958f.tmp`

Neither corresponding canonical bundle directory nor terminal `result.json`
exists. The other interrupted workers exposed no destination. Consequently:

- no partial refined bundle is consumable;
- pending array elements have not started;
- the all-success dependency has not released recording publication;
- `subject_mask_authority` remains absent; and
- no registry, selector, production archive, or production path changed.

## Recovery rule

Do not delete temporary paths or resubmit while LSF reports the attempts as
active or `UNKWN`. After LSF either recovers the jobs or gives them terminal
failure states, retry only the missing immutable partitions. A retry must
revalidate its inference predecessor and plan/window identity, publish through
a new hidden destination, and expose the canonical bundle only by final atomic
rename. Final recording publication remains forbidden until all 22 refinement
receipts validate and cover the ordered crop-row domain exactly once.

This document must be updated with terminal scheduler state, recovery actions,
final artifact validation, and the recording-level result digest before the
canary may be called complete.
