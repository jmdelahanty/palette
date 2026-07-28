# Detection Snapshot DAG Integration

Status: implemented selector-ineligible production-placement boundary; no
selector or registry activation

Date: 2026-07-27

## Purpose

The canonical-detection v1 and refined-detection v1 logical/storage contracts
now have a reusable production DAG boundary. The boundary turns complete
full-acquisition compatibility runs into immutable, access-aware Zarr v3
snapshots, atomically places both run groups in the recording analysis archive,
and proves the copied result through direct and archive-root consolidated
metadata.

This is deliberately narrower than authority promotion. A successfully placed
snapshot remains `stage_selector_eligible=false`, does not change `latest`,
`latest_complete`, `latest_pending`, or `authoritative_run`, and does not update
the registry. A later reviewed activation change must remain a separate step.

## Workflow Shape

```text
complete raw compatibility run
              \
               +--> node-local canonical/refined v1 materialization
              /                |
complete refined run            +--> complete v1 validation
                                      |
                                      +--> atomic raw run-group import
                                      +--> atomic refined run-group import
                                      +--> archive-root reconsolidation
                                      +--> fresh direct/consolidated validation
                                      +--> selector-ineligible pair receipt
```

The reusable LSF fragment is
`fisheye.cluster.detection_snapshots.build_detection_snapshot_fragment`. It
accepts exact source group paths and output run IDs, depends on an upstream
artifact supplied by the caller, and provides one typed
`detection_snapshot_pair:<target>` artifact. Therefore a full workflow can use
the module without copying its command or scheduler policy.

`compose_detection_snapshot_workflow` also supports a snapshot-only workflow
for one or many recordings whose source runs are already complete. Those
source artifacts are declared as explicit external inputs rather than hidden
mutable `latest` lookups.

The worker entry point is:

```bash
scripts/py -m fisheye.utils.publish_detection_snapshots \
  --analysis-zarr /path/to/recording_analysis.zarr \
  --source-detect-group detect_runs/detect_source \
  --source-refined-group refined_detect_runs/refined_source \
  --recording-identity recording_id \
  --canonical-run detect_snapshot_v1 \
  --refined-run refined_snapshot_v1 \
  --scratch-root /scratch/$USER/$LSB_JOBID/detection_snapshot_publication \
  --result-json /path/to/workflow/detection_snapshots/recording.json
```

The DAG wrapper creates the job-specific scratch directory, requests a
single-host CPU allocation, records the expected raw/refined `zarr.json` files
and receipt, and removes job scratch through the common runtime envelope.

## Publication Guarantees

Before shared mutation, the worker:

- requires an existing Zarr v3 analysis archive and an exact root
  `recording_id` match;
- requires fresh canonical and refined destination run IDs;
- builds both complete snapshots under node-local scratch;
- uses the frozen schema, manifest, storage-plan, array-factory, and transition
  builders rather than reconstructing manifests in the DAG;
- writes the promoted `detection_published_access_aware_v1` physical profile;
- preserves source `instance_key` identity and fails on a mismatch with the
  deterministically derived canonical source rowset;
- validates decoded arrays, offsets, reason registries, source evidence,
  manifests, and physical declarations; and
- marks both local candidates selector-ineligible.

Each run group is then copied through the common atomic run-group publisher:

- one per-recording advisory lock serializes the public rename;
- copy-back uses a hidden sibling and exclusive atomic rename;
- source and destination physical files are content-hashed;
- the public child is owner-bound and selector-ineligible before it appears;
- any failure leaves an ineligible tombstone rather than a selectable partial
  result; and
- existing family selectors are verified not to reference the new run.

After both imports, the worker regenerates root consolidated metadata and
reopens the canonical and refined run declarations from the recording archive.
The final gate repeats full publication validation and reopens the immutable
raw compatibility source evidence. Success emits one strict-JSON receipt with
the two run-manifest digests, atomic-publication receipts, transition report,
storage profile IDs, and an explicit record that selector activation and
registry update did not occur.

## Compatibility Inputs

The first integration consumes current full-acquisition raw/refined runs. It is
an adoption bridge, not the final compute writer:

- the raw adapter converts current geometry to exact canonical v1 dtypes and
  derives `frame_row_offsets`;
- modern refined source keys must agree exactly with canonical derivation;
- missing historical source keys can be initialized only with the explicit
  `--allow-initialize-missing-source-keys` migration flag; and
- legacy manual scores can be reset only with the distinct explicit
  `--allow-manual-score-reset` flag.

The compatibility adapter currently materializes the small tabular arrays in
memory. Future native detection and compactor writers should emit the same
frozen v1 arrays by physical shard and then use this publication/receipt
boundary without the compatibility conversion.

## Intentional Blockers

Clipped refined detections are not flattened by this implementation. A clipped
source contains recording-frame, clip ordinal, local-frame, and per-clip
source-row lineage that must be bound to one exact finalized collection. The
full-acquisition transition fails before either destination is published when
it sees those arrays. The dedicated clipped-v1 transition must land before the
clipped inference DAG can attach this module.

This integration also does not:

- write or interpret delta partitions;
- compact base-plus-delta generations;
- route manual review to deltas;
- activate raw or refined selectors;
- update registry projections; or
- invalidate/rebuild crops, keypoints, masks, or training exports.

Those boundaries keep this branch independent from the concurrent delta writer,
generation digest gate, and compactor work.

## Validation

Focused coverage proves:

- the fragment composes through typed artifacts and preserves exact job
  dependencies;
- historical migration switches are explicit in the rendered command;
- a real Zarr v3 pair is atomically placed and visible through consolidated
  metadata without changing existing selectors;
- both output manifests carry the promoted storage profile;
- frame offsets and raw/refined source-audit cardinalities survive copy-back;
  and
- a clipped-lineage source fails before either target run exists.

The real-Zarr tests must run outside the Codex sandbox per repository policy.
