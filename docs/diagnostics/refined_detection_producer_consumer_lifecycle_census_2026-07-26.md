# Refined-Detection Producer, Consumer, And Lifecycle Census

Status: read-only current-state census; not an accepted logical or physical
storage contract

Date: 2026-07-26

## Result

Palette has one intended refined-detection authority but not yet one end-to-end
implementation contract.

- The current non-clipped authority is
  `refined_detect_runs/<run>/instances`; `source_detections` is the bound raw
  candidate audit table. Clipped authority remains the finalized collection of
  concrete clip-local `instances` tables; a recording-level table is a derived
  consumer snapshot (`docs/current_pipeline_contract.md:65-80`).
- The sparse logical model already supports multiple rows per frame: instance
  rows are sorted by `(frame_indices, refined_row_ids)`, and an `F+1` offset
  array is rebuilt from frame counts
  (`src/fisheye/shared/refined_detect_curation.py:1685-1718`).
- The maintained manual reviewer is still a single-slot compatibility writer.
  It reconstructs one row per frame, rejects duplicate instance frames, and
  rewrites the curated surfaces after an edit
  (`src/fisheye/shared/refined_detect_curation.py:2785-2933`,
  `src/fisheye/shared/refined_detect_curation.py:2988-3096`). The current
  workflow explicitly says unconstrained multi-instance editing within one
  arena/ROI is unsupported (`docs/detection_refinement_workflow.md:145-156`).
- Immutable sharded publication exists for both an exact sibling snapshot and
  the clipped recording-level collection snapshot
  (`src/fisheye/utils/publish_tabular_snapshot.py:234-340`,
  `src/fisheye/utils/publish_clipped_refined_detect_snapshot.py:821-925`).
- A base-bound sparse-delta primitive defines `add_instance`, but its detection
  payload contains only a new bounding box and validity flag. It does not carry
  the new row's frame, acquisition-frame identity, class, confidence, refined
  row ID, or source-kind fields
  (`src/fisheye/shared/tabular_deltas.py:24-30`,
  `src/fisheye/shared/tabular_deltas.py:211-292`). There is no detection
  compactor or maintained review/Crimson routing through this layer; the delta
  contract identifies those integrations as follow-up work
  (`docs/tabular_delta_compaction_contract.md:3-5`,
  `docs/tabular_delta_compaction_contract.md:190-210`).

Therefore, adding a missing detection is central to the future refined-detection
contract, not to the immutable raw-detection contract. Palette can currently
replace or clear a single-slot frame and can express a low-level `add_instance`
operation, but it cannot yet complete the general lifecycle “add a second
subject -> display it -> compact it -> regenerate dependent products -> promote
it to training.” The earlier add-row design correctly labels that workflow as
unimplemented (`docs/manual_add_row_propagation_design.md:3-16`).

## Current Logical Surfaces

| Surface | Current role | Current arrays / identity | Lifecycle | Disposition |
| --- | --- | --- | --- | --- |
| `detect_runs/<run>` | Bound raw detector evidence | Canonical raw detection contract | Build then immutable | Reuse as immutable provenance; never edit during refinement. |
| `refined_detect_runs/<run>/instances` | Curated positive instance authority for a non-clipped run | `refined_row_ids`, frame rows and offsets, two bbox representations, source kind, manual flag, raw-row lineage, optional `instance_key`, score/class/notes | Initially built as a complete run; current compatibility reviewers later rewrite it | Future logical authority, but current mutation and physical policy are transitional. |
| `refined_detect_runs/<run>/source_detections` | Complete audit projection of the bound raw candidates | Raw row ID/key, frame, bboxes, decision, resolved refined row, score/class/reason/notes | Rebuilt with the curated projection | Retain as audit/provenance; do not make Crimson's hot display table. |
| Clip-local `.../refined_detect_runs/<run>/{instances,source_detections}` | Source-preserving clipped refined authority | Same semantic core plus clip-local identity | Per-clip evidence selected by a finalized collection | Preserve as authority for clipped workflows. |
| Top-level clipped recording snapshot | Consumer-facing projection of one finalized collection | 36 sharded arrays with recording, clip, raw-row, and refined-row lineage | Immutable, validate then promote | Keep as publication profile; do not confuse it with clip-local authoring authority. |
| `edit_delta_runs/<run>/generations/.../partitions` | Sparse edits bound by `instance_key` digest | Common revision/order fields plus bbox/valid for detections | Open generation -> immutable partitions -> frozen generation | Correct lifecycle direction, but detection add payload, overlay reads, compaction, and scheduling are incomplete. |

The checked-in `REFINED_DETECT_SPEC` declares 11 source-candidate bindings and
14 instance bindings (`src/fisheye/shared/zarr/stage_arrays.py:1089-1128`). It
does not describe the editable root compatibility arrays or the extra recording
and clip lineage written by the 36-array clipped snapshot. The generated schema
inventory records those as distinct runtime variants rather than treating the
StageSpec as complete (`docs/diagnostics/zarr_detection_schema_inventory.md`).

## Producers And Mutators

| Producer | Reads | Writes | Selection / publication behavior | Contract concern |
| --- | --- | --- | --- | --- |
| `fisheye.refinement.refine_detect` | One explicit canonical detect run, optional quality run, experiment setup, geometry | New sparse `instances` and `source_detections` run with source provenance | Marks started and pending, writes both projections, then marks complete (`src/fisheye/refinement/refine_detect.py:1568-1583`, `src/fisheye/refinement/refine_detect.py:1613-1702`, `src/fisheye/refinement/refine_detect.py:1702-1757`, `src/fisheye/refinement/refine_detect.py:1820-1825`) | Logical producer is sound; it bypasses the new canonical storage planner and writes float64 bbox rows. |
| `refined_detect_curation` | Sparse or legacy dense curated state plus the bound raw run | Recreates curated arrays and metadata | Sorts instances, derives image bboxes and `F+1` offsets, preserves/mints keys where possible (`src/fisheye/shared/refined_detect_curation.py:1637-1775`) | Central semantic writer, but not a central storage owner. |
| CLI/web detection review | Latest complete or explicit refined run and video frames | Compatibility edits back into the selected refined run | Opens the completed run writable and rewrites through `write_curated_refined_detect_root` (`src/fisheye/shared/refined_detect_curation.py:2988-3096`) | Single-slot only; mutating a completed run conflicts with the target immutable-base-plus-delta lifecycle. |
| Detection training promotion | Reviewed analysis frame state | Upserts per-recording training labels and mirrors positive labels into training `refined_detect_runs/<run>/instances` | Analysis remains editable truth; promotion is explicit or opt-in from the current web server (`docs/analysis_to_training_promotion_contract.md:47-76`, `docs/analysis_to_training_promotion_contract.md:184-228`) | Detection promotion exists, but arbitrary multi-instance identity and downstream keypoint/mask propagation remain incomplete. |
| `publish_tabular_snapshot` | Completed refined run | Exact sharded sibling | Dry-run by default; decoded hashes before completion and optional pointer promotion (`src/fisheye/utils/publish_tabular_snapshot.py:234-340`) | Preserves source inner chunks and adds a row-count-derived outer shard; it does not re-plan by bytes/access class. |
| `publish_clipped_refined_detect_snapshot` | Finalized clip collection plus recording-frame map | Recording-level `instances` and `source_detections` snapshot | Validates full frame map, unique keys, exact writes, refined identity, then completes/promotes (`src/fisheye/utils/publish_clipped_refined_detect_snapshot.py:943-1029`) | Mature snapshot publisher, but its 1,024/16,384 inner rows and 131,072 outer rows are hard-coded per column family. |
| `tabular_deltas` | Immutable base with unique `instance_key` | One immutable, one-chunk partition per writer/batch | Validates the base key digest and hint/key agreement; freezes generations (`src/fisheye/shared/tabular_deltas.py:54-82`, `src/fisheye/shared/tabular_deltas.py:85-163`, `src/fisheye/shared/tabular_deltas.py:193-196`, `src/fisheye/shared/tabular_deltas.py:211-330`) | Good concurrency boundary; detection edit schema is insufficient for a new row and no detection compactor consumes it. |

## Maintained Consumers

| Consumer class | Current behavior | Required future behavior |
| --- | --- | --- |
| Shared source resolution | Prefers an explicit/authoritative curated run and its `instances` subgroup, then falls back through legacy refined groups and finally raw detections (`src/fisheye/shared/refined_detect_resolution.py:58-171`). | Fail closed on the selected refined snapshot plus its declared delta generations; raw fallback must be an explicit compatibility policy, not silent production selection. |
| Crop | `auto`/`refined` can resolve the curated instances table and copies stable refined/raw lineage into crop rows (`src/fisheye/tracking/crop.py:3184-3502`; `docs/refined_detect_row_identity_contract.md:62-72`). | Resolve base+delta or the latest compacted refined authority, and treat rowset changes as topology changes. |
| Arena assignment / tracking | Can select `refined_detect_runs/<run>/instances` and propagates `source_refined_row_ids` (`src/fisheye/tracking/arena_assignment.py:681-736`, `src/fisheye/tracking/arena_assignment.py:802-804`). | Bind to an exact immutable snapshot/edit revision and preserve `instance_key` as observation identity. |
| Detection training export/loader | Requires/refers to refined `instances`, bbox/frame/source-kind/manual/count/raw-lineage columns and carries refined/raw row lineage into exports (`src/fisheye/utils/export_detect_training_zarr.py:589-629`, `src/fisheye/utils/export_detect_training_zarr.py:804-827`; `src/fisheye/training/zarr_yolo_dataset_loader.py:393-445`). | Consume reviewed materialized state, preserve all instances in a frame, and record the exact snapshot/delta revision. |
| Approval/profile/registry | Approves the refined authority and can create a detection profile for training use (`docs/detection_refinement_workflow.md:157-170`; `src/fisheye/tune/detect_review_backend.py:401-499`). | Select only a validated compacted snapshot or an explicitly supported base+delta view; never publish a partially extended rowset. |
| Crimson | The completed 2026-07-26 benchmark validates the canonical raw detection schema and retained `frame_row_offsets`; it does not validate refined selection or manual-add semantics (`docs/canonical_detection_storage_implementation_checklist.md`). | Prefer an explicitly selected validated refined/corrected authority, load its `F+1` offset index once, overlay active deltas if that profile is supported, and retain raw detections as provenance/fallback only. |
| Downstream keypoints/masks | Today, a newly added detection can leave fixed-length positional products short; the audited design reports no complete append/reconcile path (`docs/manual_add_row_propagation_design.md:54-111`). | A row-count change must create explicit pending work and trigger deterministic regeneration or a later incremental row workflow before the refined snapshot is promoted. |

## Identity And Index Findings

The future contract should share the raw detection core but add refined
semantics. These are different identities, not aliases:

| Field | Meaning | Current finding |
| --- | --- | --- |
| `instance_key` | Durable observation/edit identity, including manual rows | Required by delta binding and clipped snapshot publication, but optional in the current StageSpec and tolerated as missing by the normal curated writer (`src/fisheye/shared/zarr/stage_arrays.py:1091-1117`; `src/fisheye/shared/refined_detect_curation.py:1740-1755`). It must become required for the future contract. |
| `refined_row_ids` | Stable logical row ID within a refined artifact lineage | Edits retain IDs, additions receive new non-reused IDs, deletions omit the row without reuse (`docs/refined_detect_row_identity_contract.md:43-60`). It is not fish/track identity. |
| `source_detect_row_index` | Lineage into the bound raw detect rowset | `-1` represents a manual addition (`docs/refined_detect_row_identity_contract.md:28-35`). |
| `frame_indices` | Frame-local grouping key | Physical instances are sorted by frame then refined row ID (`docs/refined_detect_row_identity_contract.md:43-51`). |
| `frame_offsets` | Existing refined `F+1` CSR index | Created as `int64[F+1]` from `frame_counts` (`src/fisheye/shared/refined_detect_curation.py:1702-1705`). The shared future name should be decided once; the canonical raw contract calls the same concept `frame_row_offsets`. |
| recording/clip frame lineage | Acquisition and source-video identity | The clipped snapshot persists parent frame, recording frame ID, clip ordinal, and clip-local frame (`docs/clipped_refined_detection_snapshot_contract.md:24-40`). The ordinary refined table does not expose the same complete set. |

The cleanest shared rule is: `instance_key` identifies the observation across
publications; `refined_row_id` identifies the curated row inside one refined
lineage; offsets locate all rows for frame `f` as
`offsets[f]:offsets[f+1]`; assignment/track/subject identity belongs in later
layers.

## Physical-Storage Findings

There is no single refined-detection storage policy owner today.

- Normal curated writes set `65,536` rows only for the two fixed-width bbox
  arrays; most other arrays use Zarr defaults, and string arrays use one
  whole-array chunk (`src/fisheye/shared/refined_detect_curation.py:64-66`,
  `src/fisheye/shared/refined_detect_curation.py:214-247`). No shards are passed.
- The generic snapshot publisher keeps each source array's existing chunks and
  adds an aligned shard with a default request of 131,072 rows
  (`src/fisheye/utils/publish_tabular_snapshot.py:34-45`,
  `src/fisheye/utils/publish_tabular_snapshot.py:60-71`,
  `src/fisheye/utils/publish_tabular_snapshot.py:112-169`). This is row-count,
  not byte-budget, planning.
- The clipped publisher explicitly assigns 1,024-row payload chunks,
  16,384-row lineage chunks, and 131,072-row shards
  (`src/fisheye/utils/publish_clipped_refined_detect_snapshot.py:40-55`,
  `src/fisheye/utils/publish_clipped_refined_detect_snapshot.py:371-499`). This
  gives good object-count behavior but makes the per-column byte size vary by
  dtype and trailing shape.
- Sparse delta partitions deliberately use one ordinary chunk per column and no
  shard (`src/fisheye/shared/tabular_deltas.py:193-196`), which is appropriate
  while partitions remain small.
- None of the audited refined writers calls metadata consolidation; the mutable
  curation helper explicitly opens with `use_consolidated=False`, and training
  promotion removes inline consolidated metadata before and after mutable
  writes so it cannot remain stale
  (`src/fisheye/shared/refined_detect_curation.py:150-170`,
  `src/fisheye/tune/detect_training_promotion_backend.py:765-795`,
  `src/fisheye/tune/detect_training_promotion_backend.py:1011-1018`).
  Consolidated metadata therefore needs an explicit immutable-publication step,
  not an assumption that it already exists.

## Lifecycle Verdicts

| Question | Ground truth | Verdict |
| --- | --- | --- |
| Are raw and refined detections the same contract? | They share bbox/frame/class/score/key semantics, but raw is immutable evidence while refined adds decisions, manual rows, stable curated IDs, deletion/restore, review state, and downstream invalidation. | Share primitives and dtype definitions; keep separate versioned schemas and lifecycle profiles. |
| Are manual additions central? | Missing detections and second subjects are explicit requirements, while current single-slot review cannot express the general case. | Yes, central to refined detections; not a mutation of raw detection runs. |
| Should editable refined tables themselves be heavily sharded? | Large immutable snapshots benefit from sharding; sparse interactive edits would cause shard read-modify-write amplification. | Use immutable sharded base/snapshots plus small unsharded delta partitions. Do not append into published shards. |
| Is append-then-consolidate implemented? | Partition creation and generation freeze exist; arbitrary detection-add payload, overlay routing, and detection compaction do not. | Partially implemented infrastructure, not an operational detection lifecycle. |
| Is `frame_counts` sufficient? | The writer already derives `F+1` offsets; Crimson's accepted access model retains offsets once. | Make the offset index required and authoritative for frame-to-row lookup; counts may be derived/compatibility data. |
| Can downstream products tolerate N+1 rows? | Existing positional keypoint/mask surfaces and stale systems do not complete the workflow. | Initially force explicit full downstream regeneration on rowset change; incremental materialization can follow behind the same state machine. |

## Contract Decisions To Make Next

The first storage-contract checkpoint is frozen in
[`../refined_detection_storage_contract_v1.md`](../refined_detection_storage_contract_v1.md)
and its executable schema/storage declarations. It does not yet route a
production writer.

- [x] Define `palette.stage.refined_detection` v1 as a separate extension of
      the accepted canonical raw-detection primitives.
- [x] Make `instances/instance_key` required and require manual rows to carry a
      newly minted durable key before they enter a compacted snapshot. The
      minting algorithm belongs to the deferred delta/compactor contract.
- [x] Choose `frame_row_offsets` as the one canonical required `F+1` array name
      in both tables; reject `frame_offsets` and count aliases from v1 groups.
- [x] Lock exact dtypes, dimensions, fill/sentinel meanings, and the exact
      required fields for both `instances` and `source_detections`, including
      float32 continuous geometry and explicit score validity.
- [x] Specify full-acquisition and clipped-recording-snapshot lineage profiles.
- [ ] Define `palette.refined_detection.delta.v2` with complete operation-specific
      payloads. `add_instance` must include frame identity, bbox, class, score
      semantics, manual source kind, and a new durable `instance_key`; delete,
      restore, and replace must define field and tombstone behavior.
- [ ] Implement a delta-aware resolver/read overlay for Palette and Crimson.
- [ ] Implement a detection compactor that freezes generation `G`, opens `G+1`,
      streams whole output shards, rebuilds sorted rows and `F+1` offsets, validates
      decoded state, and publishes a new immutable snapshot without rewriting the
      old base.
- [ ] Define rowset-change state and the initial full-regeneration policy for
      crop, keypoints, and subject masks before enabling arbitrary additions.
- [ ] Route analysis-to-training promotion through the same resolved reviewed
      state, including multiple detections in one frame.
- [x] Give immutable refined-snapshot arrays exact byte-budgeted access rules
      and preserve the 128 KiB/1 MiB/8 MiB access-aware layout as an explicit
      unpromoted candidate. Authoring, delta, and training-export profiles remain
      separate follow-up contracts rather than inheriting snapshot row constants.
- [x] Require metadata consolidation only at immutable finalization, validated
      against direct metadata, with exact schema/dtype/code-map/storage/codec
      declarations in the run manifest consumed by Crimson.
- [ ] Benchmark every maintained producer and consumer at three boundaries:
      initial refined publication, sparse review apply, and delta compaction/read.
      Include missing-frame add, second-instance add, replace, delete/restore,
      random frame reads, whole hot-column residency, crop handoff, and training
      promotion.

## Recommended Implementation Order

1. Freeze the logical schema, identity/index rules, and immutable storage
   intent. **Complete for the versioned implementation target; production
   routing and physical-profile promotion remain gated.**
2. Extend and test the detection delta schema, including arbitrary manual
   additions, without changing production selectors.
3. Add a read-only base+delta resolver and deterministic materialized-state
   validator.
4. Implement the immutable detection compactor through the shared byte-budgeted
   storage planner and consolidated manifest writer.
5. Route review saves through partitions, then integrate Crimson overlay reads
   and explicit refined-run selection.
6. Add downstream rowset-change regeneration and training promotion gates.
7. Run the paired regular-versus-hybrid refined-snapshot practical promotion
   gate only after the refined logical and lifecycle contract is stable. Keep
   the reduced three-candidate optimization matrix deferred unless the paired
   check exposes a material problem or later optimization is justified;
   otherwise it benchmarks and tunes a transitional rowset.

This order keeps the accepted raw-detection contract intact, makes missing
detections a first-class refined operation, and uses large shards only where
they are beneficial: immutable publication rather than interactive mutation.
