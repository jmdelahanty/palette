# SLEAP / Palette Storage Assessment

Date anchored: 2026-07-08

Status: assessment. This records the verified comparison between current
SLEAP/sleap-io storage choices and Palette's row-lineage design. It is intended
to guide the manual add-row and row-alignment work, not to propose a wholesale
storage rewrite.

## Executive Summary

The useful SLEAP lesson is not "replace Palette with a SLEAP-style project
file." Palette's recording Zarr remains the right canonical unit for
cluster-scale, per-recording provenance. The useful lesson is narrower:

- dense payloads justify out-of-core chunked storage;
- sparse annotation identity should not depend on physical row position across
  stage families;
- downstream missing rows should be explicit pending state, not row-count drift
  discovered by a late payload-shape assertion.

In Palette terms: keep the Zarr/run substrate, but make `instance_key` or other
stable row identity the sparse lineage contract. Physical row equality should be
a payload-materialization invariant only after the downstream row exists.

## Verified SLEAP Facts

Current external facts, checked against SLEAP/sleap-io documentation on
2026-07-08:

- The SLEAP Nature Methods paper describes the mature pose/tracking system:
  top-down and bottom-up multi-animal pose approaches, many model
  architectures, and motion/appearance identity tracking.
- `sleap-io` 0.8.0 was released on 2026-06-25. Segmentation is not a dev-only
  future feature in current `sleap-io`.
- Current SLP docs list bounding boxes, segmentation masks, and label images as
  first-class file-format surfaces. Older "bbox is not first-class" wording is
  therefore stale unless explicitly scoped to older SLEAP/SLEAP-IO versions.
- `SegmentationMask` stores RLE counts as `uint32` runs and carries optional
  links to `video`, `frame_idx`, `track`, and `instance`. This is loose
  reference, not a fused `box + keypoints + mask` row object.
- SLP format 2.2 supports chunked label-image storage:
  `/label_image_data` may be a 3D `(T, H, W)` `int32` dataset with per-frame
  chunks and lazy frame decompression.

The corrected interpretation is that SLEAP/sleap-io uses a hybrid split:
sparse annotation identity remains instance-oriented, while dense pixel payloads
can live in chunked/lazy storage.

## Palette Facts

Palette already has some of the intended identity machinery:

- `src/fisheye/shared/row_lineage.py` compares row-lineage sources by
  `instance_key` when both sides provide it, sorting by key before comparing
  per-row lineage arrays.
- `src/fisheye/shared/row_alignment.py` is still a raw leading-dimension check.
  It is small, but callers and nearby payload consumers still commonly require
  physical row-count equality before operating.
- Subject-mask stale markers already preserve pending row ids in
  `source_update_pending_rows` and run-level `source_subject_mask_stale`
  payloads. The dense `source_row_stale` boolean array only updates in-bounds
  rows, so it cannot represent appended downstream rows by itself.
- Manual detect additions are partially modeled by sparse refined instances,
  but current review surfaces still have important one-instance-per-frame or
  one-slot-per-context compatibility limits.

This means "switch to `instance_key` joins" is directionally right but too
small as an implementation description. Keyed lineage comparison exists; the
remaining work is to let downstream payload surfaces represent a source row
that does not yet have materialized keypoints or masks.

## What Was Overbuilt

The overbuilt part is not the whole repository, the Zarr substrate, or the
registry in isolation. The fragile part is the coupling of four concerns to one
physical row-order contract:

- sparse instance identity;
- dense payload storage;
- cross-stage lineage;
- review/stale lifecycle state.

When detect gains a row and keypoints or masks still have the old row count,
that coupling turns a clean "new source instance lacks downstream artifact" case
into an N+1 pipeline wedge. SLEAP avoids this class of issue because sparse
identity is instance-centric and dense segmentation payloads are referenced from
that identity layer, not positionally fused across stage outputs.

## Design Rule

Use stable identity for sparse lineage, and use physical row equality only for
materialized payloads that claim to be complete.

Concretely:

1. A source row-count change should create explicit per-row downstream artifact
   state, for example `pending_generation`, keyed by stable row id or
   `instance_key`.
2. Downstream consumers should distinguish three cases:
   - source row exists and downstream payload row exists: compare keyed lineage
     and payload shape normally;
   - source row exists and downstream payload row is explicitly pending: show or
     dispatch pending work, do not treat it as corruption;
   - row counts differ without explicit pending state: fail closed as today.
3. A reconciler should act on pending state and submit exact-run downstream work
   without resolving `latest` at execution time.
4. Rowset fingerprints should include `row_count` and an `instance_key` digest
   so read-time gates can detect row drift that push-style stale markers miss.

This is the targeted version of "be more SLEAP-like": identity-oriented sparse
annotations plus out-of-core dense payloads, not a single mutable project file
and not a rewrite away from recording Zarrs.

## Refactor Scope

The full version of this design is a substantial refactor.

The `instance_key` part is partly present already: keyed lineage comparison
exists in `row_lineage.py`. The larger change is explicit downstream pending
state. That state would need to be understood by schemas, validators, readers,
review UIs, registry projections, stale handling, and job dispatch. Any consumer
that currently treats "same row count" as safe and "different row count" as
corrupt would need a third state: the source row exists, but the downstream
artifact is intentionally not materialized yet.

So the correct framing is:

- the diagnosis is localized;
- the complete fix is cross-cutting;
- the first implementation should remove the N+1 wedge without teaching the
  entire repository partial materialization semantics in one pass.

## Practical Staging

Prefer this sequence unless a concrete use case proves incremental pending rows
are needed immediately:

1. Treat any source row-count change as requiring full downstream regeneration.
   This is the smallest policy that avoids the N+1 wedge while preserving the
   existing invariant that materialized downstream payloads have complete row
   coverage.
2. Add rowset fingerprints that include `row_count` and an `instance_key`
   digest. This makes row drift visible early and gives registry/query surfaces
   a source-aware freshness signal.
3. Build a reconciler that turns explicit missing/stale downstream state into
   submitted exact-run jobs. Its first implementation can regenerate whole
   downstream runs.
4. Add per-row `pending_generation` state only after there is an actuator and
   reader surface that consumes it.
5. Add incremental generate-for-specific-rows once whole-run regeneration is
   proven too expensive for the common path.

This staging keeps the current fail-closed behavior for unexpected mismatches.
It only relaxes physical row equality after pending state is explicit and
actively consumed.

## Interchange Scope

`sleap-io` is worth treating as an interchange/export target for shareable
subsets: pose, bbox, mask, label-image, COCO/NWB/DLC/YOLO-adjacent exchange, and
reviewable examples. It should not be framed as a registry replacement or as the
canonical store for full Palette recordings unless a separate scale test proves
that workflow.

Palette's canonical archive remains:

- exact per-recording Zarr runs for raw, refined, and derived artifacts;
- registry rows as rebuildable discovery/freshness projections;
- virtual collections or manifests for project/cohort selection;
- export products as rebuildable sidecars.

## References

- SLEAP Nature Methods paper: https://www.nature.com/articles/s41592-022-01426-1
- sleap-io release history: https://pypi.org/project/sleap-io/#history
- SLP format reference: https://io.sleap.ai/latest/formats/slp/
- sleap-io mask model reference:
  https://io.sleap.ai/latest/reference/sleap_io/model/mask/
- Local companion: `docs/manual_add_row_propagation_design.md`
- Local identity direction: `docs/v2_tabular_identity_migration_checklist.md`
- Local sparse detect target: `docs/refined_detect_sparse_instances_schema.md`
