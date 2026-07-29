# Native Detection Production Integration

Status: implemented and exercised selector-ineligible raw publication boundary;
compatibility parity, Crimson acceptance, activation, and native refined
publication remain separate gates

Date: 2026-07-27

## Outcome

Native detection now has one explicit storage and publication path for clipped
and whole-recording inputs:

```text
one or more independent detector work units
        |
        +-- node-local YOLO output
        +-- immutable detection_artifact_runs package
        +-- atomic artifact import and work-unit report
                            |
all reports complete ------+
        |
        +-- fresh artifact/receipt/tree validation
        +-- exact local-frame -> recording-frame binding
        +-- deterministic recording-level instance_key minting
        +-- canonical logical detection schema v1
        +-- native run-provenance manifest v2
        +-- access-aware Zarr v3 sharded candidate on node-local scratch
        +-- complete local validation
        +-- atomic copy to detect_runs/<run>
        +-- archive reconsolidation and fresh validation
        +-- selector-ineligible publication receipt
```

The number of detector partitions does not change the public representation.
A clipped recording contributes several artifact members; a whole video can
contribute one member whose parent-frame map is `0..F-1`. Both publish the same
recording-level canonical array schema under `detect_runs/<run>`.

## Version Meanings

The version numbers identify different layers and must not be conflated:

- physical store: Zarr format 3;
- canonical detection logical arrays: schema v1;
- native canonical run/provenance manifest: v2;
- clipped/partition binding evidence: v1; and
- production placement receipt: v1.

Manifest v2 is required at the production copy boundary. Manifest v1 remains a
legacy-conversion envelope and is rejected by the native publisher.

## Artifact Versus Canonical Namespaces

`detection_artifact_runs` is an immutable, selector-free quarantine namespace.
Its arrays retain detector-local frame indices, float64 normalized boxes, dense
run-local `artifact_row_id`, and compatibility count arrays. It is not a public
canonical detection table and cannot supply quality/refinement directly.

`detect_runs` is the recording-level canonical namespace. The binder writes the
exact nine-array schema:

- `instances/frame_indices`;
- `instances/source_acquisition_frame_index`;
- `instances/instance_key`;
- `instances/bbox_norm_coords`;
- `instances/bbox_img_xyxy`;
- `instances/centers_img_xy`;
- `instances/scores`;
- `instances/class_ids`; and
- `instances/frame_row_offsets`.

The plan schema now emits
`zarr_paths.detection_artifact_target_group_path` explicitly. DAG inputs reject
a `detect_runs` path in that field. This prevents the old accidental assumption
that an artifact import and a canonical publication were the same operation.

## Binding And Lineage

Each artifact is rebound through the exact `recording_frame_index.parquet`
camera/clip mapping. The complete member set must cover every recording frame
exactly once, in canonical order, with no gap or overlap. Artifact arrays must
retain their exact dtypes, dense row IDs, count equivalence, frame ordering,
source extent, model digest, receipt identity, and tree digest.

Normalized boxes are cast once to canonical float32. Image-space geometry is
derived from those persisted values. `instance_key` is minted only after rows
are in the recording-wide frame domain, so partitioning the same detections into
one or several work units produces the same keys.

The native manifest embeds binding evidence that records every member's clip,
camera, parent-frame interval, canonical-row interval, artifact-manifest digest,
tree digest, frame-map digest, and the logical hashes of all canonical arrays.

## Publication Safety

`write_native_clipped_detection_candidate()` writes complete physical shards on
node-local scratch through the frozen byte-budget planner and shared array
factory. It publishes no selector and touches no registry.

`publish_native_canonical_detection_candidate()` then:

1. reopens the candidate and requires native manifest v2;
2. reconstructs the frozen logical and storage plans;
3. validates decoded arrays and direct/consolidated metadata;
4. resolves the exact frame and pixel authority records in the destination
   archive and verifies their content digests;
5. copies the fresh run through the common owner-bound atomic run publisher;
6. reconsolidates the archive root;
7. revalidates arrays, metadata, manifest, and source authorities; and
8. proves that no family selector references the new run.

Any authority drift detected before copy leaves no destination run. Published
runs remain `stage_selector_eligible=false`; activation and registry projection
are explicitly deferred.

## DAG Boundary

`fisheye.cluster.native_detection.build_native_detection_fragment()` creates:

1. one bounded LSF job array for independent detector artifacts; and
2. one recording-level CPU assembly/publication job depending on the complete
   array.

The fragment provides the typed artifact
`canonical_detection:<target>`. A quality or refinement module can depend on
that artifact without reconstructing storage policy or relying on `latest`.
The module is also independently composable as a detection-only workflow for
one or many recordings.

`fisheye.cluster.native_detection_campaign` is the production-facing planner.
It resolves the exact registered model and canonical acquisition authority,
materializes immutable plan evidence in dry-run mode, and submits only the two
native jobs through the Citrus login poller in apply mode.  Both source-evidence
roles may point to the same acquisition-camera record: that record owns the
recording frame domain as well as the native `width_px`/`height_px` extent.
The planner records that co-resolution explicitly and never synthesizes a
second pixel authority.

## Deliberately Open

This checkpoint does not activate the raw selector, mutate the registry, or
claim that the older clipped quality/refinement path already consumes the new
nested canonical arrays. The next integration must:

- teach recording-level detect-quality to read canonical `instances/*` arrays;
- publish refined detection snapshots through the frozen refined-v1 contract;
- bind downstream crop completeness to the refined snapshot; and
- perform the separately reviewed selector/registry activation transaction.

Until those gates land, the new raw fragment is safe to run as a detection-only
selector-ineligible canary. The existing full clipped inference workflow remains
a compatibility surface rather than being silently redirected.

## First Production-Scale Canary

The first complete native canary used the 22-clip Sleepyfish Cam2010094
recording on 2026-07-28. It deliberately published no selector and made no
registry update.

The exercise had three publication attempts:

1. `v001` failed before inference because strict JSON parsing was incorrectly
   applied to unrelated historical root metadata containing a bare `Infinity`.
   No output was mutated. Run identity discovery now tolerates unrelated legacy
   metadata while new native evidence remains strict JSON.
2. `v002` completed all 22 L4 inference array elements and built the canonical
   run, but post-copy validation hit the same enclosing-root issue. The exact
   owner-bound run was tombstoned as failed and selector-ineligible. Post-copy
   failures now fail closed automatically and cannot leave a run labeled
   complete.
3. `v003` reused the 22 immutable, validated detector artifacts and reran only
   the CPU assembly/publication job. LSF job `153226799` completed in 74 seconds
   with 939 MB peak RSS.

The completed run is:

```text
detect_runs/
  detect_native_sleepyfish_cam2010094_native_canary_20260728_v003_sleepyfish_cam2010094
```

Fresh, unconsolidated and consolidated reads validated:

- 1,188,000 recording frames;
- 1,168,175 detection rows and 1,168,175 unique `instance_key` values;
- source-frame coverage from 0 through 1,187,999, with 19,825 valid empty
  frames;
- exact `F+1` monotonic offsets and equality between canonical and acquisition
  frame indices;
- native manifest v2, logical schema v1, and
  `detection_published_access_aware_v1` storage;
- 24 MB stored in 27 files; and
- null `latest`, `latest_complete`, `latest_pending`, and `authoritative_run`
  selectors.

The retained publication receipt is:

```text
/groups/johnson/johnsonlab/jeremy/staging/.processing_logs/
  native_detection_canary_20260728_v003_republish/native_detection/
  sleepyfish_cam2010094.publication.json
```

Its SHA-256 is
`50d4e1b68b3b52c64f354a01f88a907271368bc9c70d6ae42f652b8a947f159c`.
The adjacent validation receipt records the independent fresh-read audit.

The logical content from the failed `v002` placement and successful `v003`
republish was identical, with digest
`8a841f6a866da96a6eee6a2eb483f0f897998742c38f374af22ec05f8d7f9431`.
This proves deterministic replay from the same immutable artifacts. It does not
yet prove compatibility-writer parity: the historical comparison run used a
different model version. The remaining raw-writer gate is an exact same-input
compatibility/native comparison followed by a Crimson read canary.

The GPU array produced one operational warning: concurrent workers could race
while creating the optional Ultralytics user-settings file. Inference was not
affected, but production jobs should isolate `YOLO_CONFIG_DIR` per worker.

## Focused Validation

The focused suite covers:

- split/unsplit deterministic key equivalence;
- exact artifact dtypes, row IDs, frame coverage, and camera identity;
- native manifest-v2/local logical-v1 construction;
- model-digest consistency across all work units;
- authority drift and manifest-v1 rejection before archive mutation;
- atomic copy-back with no selector or registry change;
- explicit artifact/canonical path separation in the LSF plan; and
- detection-only workflow composition.
