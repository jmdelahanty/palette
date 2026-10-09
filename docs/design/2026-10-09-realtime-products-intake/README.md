# Realtime products at intake (step 1)

- **Status:** accepted (scope and defaults decided by Jeremy, 2026-10-09)
- **Owner:** Palette intake. **Last reviewed:** 2026-10-09
- **Builds on:** [2026-10-07-intake-single-writer](../2026-10-07-intake-single-writer/README.md)

## What this covers

Orange records, per camera, what its realtime detector and pose model produced
for every recording frame, and declares those files in `recording_session.json`
under `realtime_products` (`orange.recording_realtime_products` v1). The sealed
start snapshot names the models that ran (`models.<serial>.detect`,
`models.<serial>.pose`) by content digest.

Step 1 makes Palette check and record this at intake. It does not turn the
detections or poses into Zarr arrays for analysis; that is step 2, decided
separately.

Step 1 does three things:

1. **Validate.** Check the declaration, both model blocks and every line of
   each v2 event log against Orange's pinned schemas, and check the declared
   counts and frame range against the log itself.
2. **Place by owner.** Organize each declared per-camera file into that
   camera's recording folder only. Today these files are session context and
   are linked into every camera's folder.
3. **Record.** Write one immutable realtime-products record into each camera's
   analysis Zarr and mirror it into the registry, including which detection and
   pose models ran.

## Sources of record

These were settled with Orange and Citrus on 2026-10-09:

- **Orange's per-camera event logs** are the per-frame source for what the
  models produced. They cover every recorded frame, are written by the
  producer, and carry model identity.
- **Citrus's unified H5** (`/observations/bounding_boxes`, `/observations/pose`)
  is "what reached this arena's experiment". It is a subset of Orange's
  output: it loses updates on counted queue overflows and uncounted
  reader-level drops, and an update with no boxes writes no bounding-box row.
- **What a stimulus acted on** is in that stimulus's own tables, for example the
  Chaser `target_source_*` fields.

Step 1 does not compare the H5 with Orange's logs. The join keys and the
expected subset rule are recorded in the step-2 notes below, so that check can
be built later.

## Contracts pinned

The schemas are copied byte for byte from Orange at `0927ecaf`. Those commits
live on github.com/jmdelahanty/orange; the JohnsonLabJanelia/orange repository
only mirrors it. Each schema is pinned by SHA-256 in
`src/fisheye/shared/contracts/README.md`:

| Schema | Used for | SHA-256 |
| --- | --- | --- |
| `orange_recording_realtime_products_v1` | `recording_session.json` `realtime_products` | `cfe02e6e…fc227` |
| `orange_recording_detect_model_v1` | `models.<serial>.detect` (from 2026-10-09) | `02be9656…1626f` |
| `orange_recording_pose_model_v2` | `models.<serial>.pose` (from 2026-10-09) | `f1916316…7198b` |
| `orange_recording_pose_model_v1` | `models.<serial>.pose` (before) | `778f172e…5a0ee` |
| `orange_yolo_event_v2` | `Cam<serial>_yolo_events.jsonl` lines | `393a77ba…650f9` |
| `orange_pose_event_v2` | `Cam<serial>_pose_events.jsonl` lines | `835d413c…74fb7` |

Version-1 event logs, from before Orange `d99e759`, never had a schema. For
those Palette records only the file-level facts that `realtime_products`
declares. Recordings sealed before Orange `de773b1` (2026-10-08 evening) have no
`realtime_products` at all, and their record says so.

## Validation (refusals are deterministic, exit 65)

For each camera that the transfer delivers, when `realtime_products` is present:

- The block validates against `orange_recording_realtime_products_v1`, and it
  declares exactly the delivered cameras.
- Every declared file is in the sealed snapshot inventory, with the declared
  `size_bytes` and `sha256`. The seal has already proven those bytes.
- For each product (`detections`, `pose`) whose `status` is `present`:
  - The `model_ref` digests equal the start snapshot's
    `models.<serial>.<detect|pose>.runtime` digests.
  - **Line schema version 2:** every line validates against Orange's schema:
    - line 1 is the `session_header`, for this camera and recording;
    - `spatial_mask_policy` lines may appear;
    - frame lines carry `event_sequence` 1..N with no gaps, and strictly
      increasing `frame.recording_frame_id` from the declared first frame to
      the declared last frame;
    - N equals `row_count`, and the header and policy lines equal
      `header_rows`;
    - the per-kind counts equal `rows_by_kind`, and the header's model digests
      equal `model_ref`.
  - **Line schema version 1:** the line count equals `row_count`, and there is
    no line validation.
- Each model block is recorded with the newest pinned schema it validates
  against: detect v1; pose v2, then pose v1. A block that validates against
  none, because it predates Orange's model schemas, is recorded as "not
  schema-validated" and is not refused. Its digests are still compared with
  `model_ref`.

A declared product with `status: absent` is recorded as absent. Any
disagreement between the declaration, the logs, the snapshot and the inventory
refuses the delivery, because it is a producer contradiction.

Every line is validated, by decision. Measured on the reference delivery,
validating both logs of one camera (24,002 lines) takes about 9 s. A 24-hour
camera is about 8.6 million lines per log, so roughly 1.8 h per camera; cameras
can run in parallel. If day-long sessions become routine, a faster validator,
such as a compiled schema check, is the place to recover time. It must not
reduce what is checked.

## Placement

The organizer reads `realtime_products.cameras.<serial>`. Every path listed
under `detections.files`, `pose.files`, `crop_files` and `acquisition_files` is
owned by that camera, with the new plan role `camera_realtime_product`. It is
placed only in that camera's folder, at the same relative path it has today
(`raw/acquisition/<name>`), so existing readers still find it there.

A file declared for two cameras, or a declared file that is missing from the
inventory, is refused. Undeclared files stay session context, as before.

## Record

The record is written once at import, per camera, into
`analysis/acquisition_realtime_products` with schema
`palette.acquisition_realtime_products.v1`. It holds:

- whether `realtime_products` was declared, and the producer schema;
- per product:
  - `status` and the files (organized path, size, sha256, role);
  - the line schema, `row_count`, `header_rows`, `rows_by_kind`,
    `rows_by_status`, and the first and last recording frame;
  - the validation mode (`every_line_v2` or `file_level_v1`);
- the model:
  - `models_key`, `model_id`, `engine_sha256`, `engine_bytes`,
    `weights_sha256`, `onnx_sha256`;
  - the engine manifest's `run_id`, `set_id`, `status`, `build_id`,
    `precision` and `sha256`;
  - the schema version that validated it;
- `record_sha256` over the canonical record.

The manifest `status` (today `candidate`) is recorded as given, as history of
what ran. It is never an admission gate, and Palette never requires an
activated model to import a recording.

The record is not added to the import receipt in step 1, because that would
change the receipt's grammar. It is published like the other acquisition
records, and validated against its own digest when read.

## Registry

A migration adds `recording_realtime_products`, one row per dataset and
product. It mirrors the record exactly, as current-source rows do (#290
decision 4).

A view, `recording_realtime_models`, joins each product to `training_runs` in
two ways:

- by content: Orange `weights_sha256` = `training_runs.model_sha256`;
- by name: the manifest `run_id` = `training_runs.run_id`.

It reports whether the two joins agree. On the reference delivery both models
join by content: detect `74458dc7…`, run
`detect_all_available_detect_training_v004_yolo11n_trt_20260520`; pose
`2edf67ad…`, run `pose_head_192_recovered_reviewed_v001_yolo11n_100e_20260915`.

## Engine provenance

The start snapshot names each engine's manifest by path and sha256, but the
manifest itself stays on the rig. Two things are requested from Orange
(2026-10-09):

- every delivery includes, declared with path, size and sha256, the manifest of
  each engine that ran, plus the calibration record and cache for INT8 engines;
- a closed schema for `orange.tensorrt_engine_manifest` v1. Today the writers
  define it, and the new `build.environment` and `int8_calibration.record_*`
  fields are optional.

Until that lands, step 1 records the manifest path and sha256 from the
snapshot. Engines are registered from manifests copied by hand. For example,
the int8mm detect engine `479e82d2…` was built on pancake0 with the TensorRT
min-max calibrator on 975 frames, with cache `54bd559f…` and record
`f586f5cb…`. Its files were copied to `/groups/…/jeremy/handoff/` on 2026-10-09
and verified against the snapshot's manifest digest `1ad86ffc…`.

## Delivery plan (small PRs)

1. This note, the pinned schemas, and a pure validator module
   (`fisheye.shared.orange_realtime_products`) with tests. No intake behaviour
   changes.
2. Organizer placement by owner.
3. Import: validate, refuse on contradiction, write the Zarr record.
4. Registry migration, extractor and view.

Each PR runs the reference fixture: delivery `d830510c`,
`rolling_crops_shadow_20261009_004349`, four cameras, v2 logs.

## Step-2 notes (not built here)

These are the cross-check keys, from Orange and Citrus on 2026-10-09:

- **H5 bounding-box rows** are keyed by `payload_frame_id`. That is Orange's
  `state_frame_id`, which the log line carries as `frame.local_frame_id`. The
  rows have no `recording_frame_id`, and a frame with zero boxes writes no row.
- **H5 pose updates** carry `recording_frame_id`, so they join to the pose log
  on `local_frame_id` and are cross-checked on `recording_frame_id`.
- **Frame offset:** `local_frame_id` equals `recording_frame_id` in headless
  runs, and differs by a constant in GUI sessions.
- **Expected subset:** every H5 update inside the recording window has exactly
  one log line. Every log line with `citrus_live_ipc.request_status` =
  `queued` reached Citrus, unless Orange's IPC drop counters
  (`frame_ipc_summary`) or Citrus's session totals
  (`tracking_queue_push_failures`, the IPC gap counters in the end-of-session
  Frame Statistics event) are non-zero.

## Decision log

- 2026-10-09:
  - Step 1 only.
  - Orange's logs are the source, and the H5 is the experiment's view.
  - Validate every v2 line.
  - Model status is recorded as history, never as a gate.
  - Jeremy decided all of the above, after consulting Citrus and Orange.
