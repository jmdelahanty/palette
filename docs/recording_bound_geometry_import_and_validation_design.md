# Recording-Bound Geometry Import and Independent Validation Design
<!-- contract-meta
version: 3
status: phases-0-through-3-implemented
last_verified: 2026-07-26
-->

## Purpose and status

This document defines Palette's intended handling of recording-bound geometry
produced by Orange and embedded or mirrored by Citrus. It covers immutable
acquisition evidence, normalized geometry candidates, independent fitting from
recorded imagery, comparison and review, operational selection, detection
gating, registry reporting, and staging cleanup.

The design is approved. Phases 0 through 3 are implemented: Palette can
preserve and verify the exact recording-local bundle during organization; the
shared read-only normalization boundary supports strict Orange-folder and
Citrus-H5 adapters; and verified acquisition geometry can be published as an
immutable, candidate-only analysis run. Independent fitting, comparison,
selection, detection gating, registry projection, and cleanup remain future
phases and are not activated by the loader or candidate publisher.

The implementation lives in `fisheye.shared.recording_geometry`. Organized
recordings retain the fixed version-1 subtree at:

```text
raw/recording_geometry_bundle/
  recording_snapshot.json
  recording_geometry_contract.json
  recording_geometry_assets/
```

The wrapper directory is Palette's relocated recording-folder root; the three
producer children retain their guaranteed names and relative relationships.
Organization verifies source bytes, copies to a same-filesystem temporary
directory, verifies the copy, atomically renames it, and verifies the published
directory again before ordinary recording moves proceed. The recording
manifest and organizer JSONL record the contract and asset-manifest digests.

The normalized mask declares Palette's canonical
`source_camera_image_px.top_left_y_down.v1` profile with pixel-center geometry.
It does not mint a coordinate authority from JSON. Before downstream
publication or consumption, `bind_registered_dish_mask_to_source_camera_frame`
requires the exact persisted Palette source-camera authority and rejects a
different camera, extent, semantic space, units, or pixel convention.

Two validation modes are intentionally distinct:

- bundle preservation verifies the fixed v1 subtree even for early producer
  fixtures whose snapshot omitted the contract pointer;
- the scientific Orange-folder adapter requires the checksummed snapshot
  pointer and reports those early fixtures as
  `legacy_missing_recording_bound_mask`.

This distinction lets Palette save irreplaceable acquisition evidence without
quietly upgrading incomplete producer metadata into scientific authority.

The central rule is:

> Preserve what acquisition reported, independently evaluate it, and use a
> geometry for downstream processing only through an explicit, provenance-bound
> selection.

An independently fitted Palette circle is not presumed correct. An
acquisition fit is not presumed correct merely because it was registered and
operator-confirmed. Agreement corroborates the candidates; disagreement is a
review condition and does not identify its own cause.

## Scope

The recording geometry bundle may contain more than a dish mask:

- recording-bound daily dish-rim observations;
- a physical inner-rim boundary;
- an outward-forgiving detection region;
- homographies and their directed coordinate contracts;
- projected-surface and dish-top-rim scales;
- daily-registration identity and runtime-selection evidence;
- tank-design geometry and optical-stack metadata;
- nominal experimental-area geometry;
- source-image, review, and quality evidence when materialized.

These surfaces have different meanings and must not be collapsed into one
unqualified circle. In particular:

- the observed dish-top inner rim is a physical-boundary candidate;
- a visible upper/top-rim edge fitted from recording images is a distinct
  image-observation candidate and must not be relabeled as the physical inner
  rim;
- `valid_detection_region` is a derived centroid gate;
- the nominal experimental area is not a substitute for an observed rim;
- dish-top-rim scale and projected-surface scale describe different physical
  planes;
- Citrus runtime registration identity is distinct from the recording-bound
  Orange geometry embedded for offline use.

## Authoritative producer references

The current producer contracts are maintained in the Orange checkout:

- `/home/jeremy/orange-gop-split-a16/docs/recording_geometry_contract.md`
- `/home/jeremy/orange-gop-split-a16/docs/dish_top_rim_observation_design.md`
- `/home/jeremy/orange-gop-split-a16/docs/schemas/orange_recording_geometry_contract.schema.json`
- `/home/jeremy/orange-gop-split-a16/docs/schemas/orange_recording_geometry_assets.schema.json`

The current supported versions are:

- `orange.recording.geometry_contract`, version 1;
- `orange.recording.geometry_assets`, version 1;
- `orange.calibration.dish_top_rim_observation`, version 2;
- `orange.recording.daily_registration_geometry`, version 1;
- `citrus.calibration.daily_registration`, version 1;
- `citrus.session.orange_recording_geometry_contract_scope`, version 1.

Unsupported future schema versions must be preserved as raw evidence and
handled fail-closed until Palette explicitly supports them.

## Conceptual architecture

Palette should separate evidence, interpretation, and operational decisions:

```text
Recording-bound evidence
  Orange folder + Citrus H5
          |
          | exact-byte checksum and schema validation
          v
Normalized immutable candidates
  acquisition physical rim + acquisition gate
  Palette recording-image observed-boundary fit
  optional manual-review fit
          |
          | candidate-to-candidate comparison
          v
Agreement or review decision
          |
          | explicit selection
          v
Operational geometry
  detection gate, physical arena, scale, transforms
          |
          v
Detection refinement, segmentation, tracking, and spatial analytics
```

The same normalized model and geometry kernel must serve both producer input
forms. Orange-folder and Citrus-H5 support are input adapters, not independent
implementations.

## Recording-folder authority and organization

### Required relative layout

For schema version 1, Orange guarantees this relative layout:

```text
recording_snapshot.json
recording_geometry_contract.json
recording_geometry_assets/
```

Palette's organizer may relocate the containing recording-folder root. It must
not rename, flatten, or independently move the three children. The relative
layout is part of recording-bound discovery and immutable checksum validation.

When one Orange session folder is split into Palette's per-arena recording
directories, the complete geometry subtree is small enough to be copied into
each affected recording. Each destination must receive the intact subtree and
must be verified before the staging authority can be retired. A camera-specific
loader then selects only the exact `(arena_id, camera_serial)` entry relevant to
that recording.

### Safe discovery

The folder adapter must:

1. Read `recording_snapshot.json`.
2. Read `recording_snapshot.recording_geometry_contract.relative_path`.
3. Reject an absolute path or a path escaping the recording root.
4. Read the exact referenced contract bytes.
5. Verify their SHA-256 against the snapshot declaration.
6. Require the supported contract schema ID and version.
7. Read `daily_registration_geometry` without searching any external
   calibration location.

It must never select an artifact by current active pointer, latest timestamp,
directory scan, or filename heuristic. It must never substitute another
camera's or arena's geometry.

The historical `valid_until_utc` is evaluated relative to acquisition. A valid
recording does not become invalid merely because the registration has expired
by the later analysis date.

### Asset verification

When `materialized_assets.status == "complete"`, Palette must:

- resolve the manifest inside the recording root;
- verify the manifest SHA-256;
- validate required manifest entries;
- verify each used asset's path, size, role, camera/arena context, and SHA-256;
- distinguish complete, partial, missing, and invalid local evidence.

The embedded checksummed recording-contract geometry remains the full-precision
numerical authority. The Palette-v2 compatibility export rounds center and
radius to integer pixels and must not replace the full-precision calculation
source. Circle centers and radii are continuous point geometry in the native
camera image plane; they are not discrete pixel indices or half-open box edges.

## Citrus H5 authority

The H5 adapter reads:

```text
/recording_geometry_contract/contract_json
/recording_geometry_contract/h5_scope_json
/runtime_geometry_contract/daily_registration_json
```

The first two datasets bind the recording to an exact Orange contract and the
H5 arena/camera scope. The runtime dataset records which daily registration
Citrus actually loaded. These meanings must remain separate.

### Exact H5 string checksum semantics

Variable-length scalar UTF-8 datasets may be returned by `h5py` as `bytes`,
`str`, or zero-dimensional NumPy values. Palette must unwrap the scalar once,
obtain the exact UTF-8 payload bytes, and hash those bytes before JSON parsing.

- Do not append an HDF5 NUL terminator.
- Do not strip whitespace.
- Do not normalize newlines.
- Do not parse and reserialize JSON before hashing.
- Preserve a source-file trailing newline when it is present.

`/recording_geometry_contract/contract_json` contains the exact Orange source
file bytes, including a final newline when the source file contains one.
`h5_scope_json` and the Citrus runtime contract JSON use the exact compact dump
stored in their datasets without an appended newline.

### Runtime application identity

Palette should preserve the literal recorded runtime source path as provenance,
but a path is not portable mask identity. Offline identity is determined by:

- schema identity;
- artifact and registration identities;
- exact camera and arena binding;
- exact content checksums.

Orange already records
`selected_daily_registration_applied_by_citrus` after applying its literal path
and checksum rules. Palette should retain that producer conclusion rather than
attempting to reinterpret equivalent relocated paths. Palette still verifies
the embedded contract, H5 scope, runtime JSON checksum, registration identity,
and exact camera/arena/rim-observation identity.

## Normalized identity and status model

The public conceptual loader result is:

```text
mapping[(rig_id, canvas_name, arena_id, camera_serial)]
    -> RegisteredDishMask
```

The normalized record retains at least:

- rig, canvas, arena, and camera identity;
- acquisition artifact ID and SHA-256;
- registration ID and checksum;
- source-contract and H5-scope checksums;
- native camera width and height;
- coordinate-space, origin, axis, and unit declarations;
- full-precision accepted inner-rim center and radius;
- full-precision valid-gate center and radius;
- physical radius and dish-top-rim scale;
- materialized-asset status;
- source validity timestamps;
- Citrus runtime application status, when available;
- producer review and quality evidence.

Statuses should remain orthogonal:

```text
mask_geometry_status:
  valid | missing | invalid | legacy_missing

materialized_asset_status:
  complete | partial | missing | invalid

citrus_registration_status:
  exact_match_applied | missing | checksum_mismatch |
  registration_id_mismatch | camera_arena_target_missing |
  rim_observation_mismatch | invalid

comparison_status:
  pending | agreement_passed | review_required |
  palette_fit_failed | acquisition_missing | coordinate_mismatch
```

`selected_partial` may be salvaged per camera under `if_available` when that
camera independently passes all schema, identity, checksum, raster, geometry,
operator-acceptance, and required-asset checks. Under `required`, an affected
camera fails. A caller may additionally require all participating cameras and
therefore fail the whole collection.

## Geometry and shape validation

For a schema-v2 registered rim, Palette requires a circle in native camera
pixels. The following declarations must agree:

- observation `camera.width` and `camera.height`;
- `accepted_mask.image_shape_px`;
- `circle_detection.image_shape_px`;
- the recording-contract snapshot dimensions;
- `recording_snapshot.camera_runtime[serial].coordinate_frame`;
- the decoded native/full-frame source dimensions.

Palette validates the physical boundary separately from the operational gate.

The physical boundary must declare:

- `target_plane == "dish_top_rim"`;
- `coordinate_space == "camera_native_pixels"`;
- circle geometry with finite center and positive radius;
- exact camera and arena identity;
- operator acceptance where required by the producer contract.

The valid detection region must declare:

- purpose `bounding_box_centroid_detection_gating`;
- outward offset direction;
- native-camera coordinate space;
- finite positive circle geometry;
- a center concentric with the accepted inner rim within numerical tolerance;
- radius greater than or equal to the inner-rim radius.

`valid_detection_region` is the final producer-selected tolerance. Palette must
use its center and radius exactly and must not add Palette's existing 0.5 mm
boundary tolerance.

Current schema-v2 observations and the Palm tank designs are circular. Palette
may retain its existing rectangle support for legacy/manual geometry, but it
must not coerce an unsupported registered shape into a circle. A future producer
shape or schema is unsupported until implemented explicitly.

## Current acquisition evidence inventory

A read-only inventory on 2026-07-26 found nine geometry contracts under the
current `/groups/.../staging` area:

- seven July 21 contracts with no `daily_registration_geometry`;
- two July 22 contracts with
  `mode == "selected_daily_registration"`,
  `status == "selected_resolved"`, four resolved cameras, and complete asset
  manifests.

The two July 22 recording batches bind the same accepted daily registration,
as expected for a daily calibration, while their recording-contract bytes and
checksums differ because each is a distinct recording snapshot.

The observed camera geometry is:

| Camera | Arena | Accepted rim center, native px | Rim radius, px | Detection-gate radius, px |
| --- | --- | ---: | ---: | ---: |
| 2010093 | arena_1 | (2243.057, 2234.795) | 2143.641 | 2160.756 |
| 2010094 | arena_2 | (2333.936, 2284.365) | 2137.582 | 2154.698 |
| 2010095 | arena_3 | (2347.705, 2300.889) | 2143.365 | 2160.481 |
| 2010096 | arena_4 | (2212.764, 2218.271) | 2137.307 | 2154.422 |

All four gates add approximately 17.116 camera pixels, or 0.319--0.320 mm,
beyond the physical inner rim.

The acquisition evidence includes:

- 4512 by 4512 native `Mono8` image dimensions;
- physical target `dish_top_rim` and the water-side inner-rim edge;
- circular physical geometry with 40 mm radius;
- dish-top-rim scale near 53.59 pixels/mm;
- a separately labeled projected-surface scale that is not authoritative for
  the top rim;
- a temporal mean of 60 source frames;
- controlled NIR illumination, exposure, filter, and projector state;
- Hough parameters, detection scaling, source frame IDs, and source checksums;
- operator confirmation, operator adjustment, and quality flags;
- registration, artifact, camera, arena, and contract identities;
- tank-design dimensions, materials, refractive indices, water depth, and
  optical-stack assumptions.

For these observations, operator adjustment is zero and quality flags are
empty. `dish_fill_state` and runtime rim verification are recorded as unknown.
Optional calibration source images and review overlays were not requested for
materialization. Their metadata and checksums exist, but the image bytes are
not present in the recording-local bundle.

The July 21 Batman artifacts are valid legacy-negative fixtures for registered
masks: the recording snapshot does not reference recording-bound daily mask
geometry, and the H5 reports that the recording geometry was not referenced.
Neither runtime registration nor nominal `experimental_area` may be used to
invent a recording-bound physical mask.

## Current Palette fit behavior

Palette's current canonical tuner is `fisheye.tune.mask_tuner`. It:

- selects one frame, using the middle frame by default;
- prefers `raw_video/images_ds` and otherwise generates a source-video preview
  of at most approximately 640 pixels on the longest dimension unless full
  resolution is explicitly requested;
- converts to grayscale and applies a 9 by 9 Gaussian blur;
- calls OpenCV `HoughCircles`;
- uses the first returned circle;
- exposes Hough thresholds and manual radius adjustment through an interactive
  UI;
- supports manual rectangular geometry;
- stores the chosen source frame, source array, image dimensions, normalized
  metrics, Hough parameters, and timestamp in
  `analysis_metadata.attrs["dish_mask"]`.

The existing automatic parameter search prefers parameter values near fixed
defaults when exactly one circle is returned. It does not score edge support or
fit residual. The chamber propagation utility reruns shared Hough parameters on
one frame from each recording, but similarly takes the first returned circle.

The current fitter does not persist:

- edge residual or inlier fraction;
- confidence;
- multi-frame or temporal-window stability;
- immutable candidate history;
- candidate comparison;
- a distinction between physical rim and operational detection gate.

This makes the current tool useful for interactive review and as an initial
independent signal, but not sufficient to automatically overrule an accepted
acquisition candidate.

## Independent Palette fit

The primary purpose of the independent fit is to assess whether the acquisition
fit itself was accurate. A fit from the recorded imagery also has the useful
secondary property of observing the dish during the experiment, but a
disagreement does not prove that the dish moved.

A disagreement may result from:

- acquisition fitting error;
- Palette fitting error;
- dish or camera movement between registration and recording;
- an incorrect coordinate mapping;
- illumination, focus, fill-state, or rim-visibility differences.

Palette should therefore report disagreement without assigning cause.

The target independent fitter should:

1. Run only after the recording and analysis Zarr import succeeds.
2. Use the actual recording source and prove its native coordinate mapping.
3. Sample early, middle, and late temporal windows.
4. Build robust temporal composites, such as a median or trimmed mean, to
   suppress fish and transient stimuli.
5. Detect at a reduced resolution and refine the boundary against
   full-resolution image evidence.
6. Avoid seeding the fit with the acquisition center or radius.
7. Persist edge support, residuals, inlier fraction, and between-window
   stability.
8. Optionally fit an ellipse as a diagnostic. An ellipse must not become an
   operational gate without a new explicit geometry contract.

The fitter's declared target and the feature actually supported by the image
must remain separate. A visually reviewed fit may be accepted as a
`visible_dish_top_rim_edge` observation even when the probe originally labeled
its target as an inner water-side edge. That semantic correction is part of
the immutable review record; it does not rewrite the frozen probe report or
turn the visible top edge into an acquisition physical-inner-rim observation.

An image-derived physical-inner-rim fit must not include the acquisition
gate's forgiveness in the fitted physical radius. A reviewed visible-top-rim
fit may instead derive an offline centroid gate directly from that observed
circle, with an inclusive boundary and zero additional Palette tolerance. The
two derivations have different semantics and must not be silently compared as
the same physical boundary.

## Candidate comparison

A physical-boundary comparison binds two exact immutable candidate IDs whose
observed features are semantically compatible and compares them in native
camera coordinates. It must not compare a Palette visible-top-rim observation
to Orange's physical inner rim as though they were measurements of the same
edge.

A detection-gate disagreement audit is a different comparison. It may compare
the exact `valid_detection_region` from each candidate against one exact
detection rowset. That audit reports the operational consequences of choosing
one gate or the other; it does not claim that the two source circles represent
the same physical feature.

Persist at least:

- center displacement in native pixels;
- center displacement using the dish-top-rim scale in millimetres;
- signed and absolute radius difference;
- maximum circle-boundary separation;
- circle intersection-over-union or mask-disagreement fraction;
- Palette edge residual and inlier fraction;
- temporal-window fit variation;
- source-dimension and coordinate-contract agreement;
- the exact candidate and source artifact checksums.

Initial production should report continuous measurements before adopting hard
pass/fail thresholds. Thresholds should be chosen from the July 22 four-camera
canary and repeat-recording variability, not guessed in advance.

## Immutable candidate and selection model

Palette uses a small versioned geometry-run surface:

```text
arena_geometry_runs/<candidate_run>/
arena_geometry_comparison_runs/<comparison_run>/
arena_geometry_selection/<selection_record>/
```

The first implemented surface is
`analysis/arena_geometry_runs/<candidate_id>`. Acquisition candidates use
`palette.arena_geometry_candidate_record` version 1 inside
`palette.arena_geometry_candidate_run` version 1. They are metadata-only Zarr
v3 groups. The candidate ID is derived from the canonical candidate-record
SHA-256, so retrying the same evidence is idempotent and changed evidence
produces a new immutable name.

Candidate publication is intentionally weaker than operational selection. A
published acquisition candidate is complete and `stage_selector_eligible`,
meaning that it is safe for comparison and review code to read. Publication
does not set `latest` or `latest_complete`, write
`analysis_metadata.attrs["dish_mask"]`, select a detection gate, or alter any
detection. The run records each of those negative assertions explicitly.

The publisher creates and validates the candidate in node-local Zarr storage,
copies it to a hidden same-parent sibling, verifies the copy, atomically
renames it, completes it, and only then marks it readable. Immediately after
the rename it revalidates the recovery receipt, Orange contract, and persisted
source-camera coordinate authority. A source change aborts publication rather
than publishing a candidate bound to mixed evidence.

The production checkpoint on 2026-07-26 revalidated both affected July 22
Batman session bundles against all eight recording targets. The recovery chain
contains eight verified immutable receipts and eight complete acquisition
candidates: one for each camera/arena in the `15:44:40Z` and `16:15:04Z`
four-arena batches. Every candidate remained `not_selected`; no `latest` or
`latest_complete` pointer, compatibility `dish_mask`, or detection gate was
written. The copied acquisition snapshots remained byte-identical to their
staging authorities. Staging inputs were deliberately retained pending the
independent-fit and comparison phase.

Candidate rereads preserve the original creation-time Git, software, host, and
runtime provenance. Validation across later Palette versions compares the
stable publication contract—command, configuration hash, parameters, input
run IDs, and input artifact identities and digests—rather than incorrectly
requiring the validator's current runtime provenance to equal the historical
producer context.

Reviewed Palette image candidates use the same immutable publication surface
with `candidate_kind = palette_recording_image_fit`. The record binds the
frozen fit-report bytes, review-montage bytes, exact source-video identity,
early/middle/late source-frame hashes, canonical continuous source-camera
pixel-frame authority, the semantic correction made during review, and the
reviewer decision. Publication remains pointerless and explicitly audit-only.

For the first Batman canary, visual review found that the blind Palette circle
followed the visible top of the rim well, while the acquisition observation
followed the inner surface. Palette therefore preserves the acquisition
physical inner-rim circle unchanged and publishes the Palette circle as
`visible_dish_top_rim_edge`, not as a replacement physical-inner-rim claim.
The Palette circle's offline gate is derived directly from that reviewed
circle; the legacy 0.5 mm expansion is not added.

`audit_arena_geometry_detection_gates` performs the next fail-closed step. It
binds exact Palette candidate, acquisition candidate, and raw-detect run names;
requires their native-camera coordinate authorities to agree; verifies every
stored center against its source box; preserves `instance_key`; calculates
both unrounded signed distances; writes every asymmetric decision to CSV; and
selects deterministic temporal samples from both `palette_only` and
`acquisition_only`. When there are no exclusive disagreements, it instead
selects the boundary-nearest detection from each of several temporal
partitions, so the visual review cannot silently collapse to an empty montage.
Optional PyNvVC rendering seeks to the preceding GOP keyframe for each selected
acquisition frame and decodes only through that target. For the current Orange
I/P-only 25-frame GOP contract, exactness is proven by the demuxer's target
relation, strictly increasing packet presentation timestamps, and an ordered
pending-packet/display-frame queue that accounts for NVDEC startup latency;
reordering or an unexpected seek result fails closed. It produces full-frame
plus local review panels on an LSF GPU worker. The diagnostic never selects a
candidate, filters detections, modifies the Zarr, or updates the registry.

The Batman Cam2010093 canary decoded eight review frames from 128 total GOP
packets in 1.72 seconds. The complete LSF job finished in 19 seconds with
278 MB peak memory. The superseded sequential implementation was canceled
after 681 seconds while traversing toward the same late-recording samples.

Each candidate records:

- `source_kind`, such as `orange_registered_observation`,
  `palette_recording_image_fit`, or `manual_review`;
- camera, arena, rig, canvas, and recording binding;
- physical boundary geometry;
- operational gate geometry, when one is explicitly supplied;
- coordinate and raster contracts;
- source artifact IDs and checksums;
- method, parameters, quality, and software version;
- completion and review state.

Candidates are immutable. A manual correction publishes a new candidate rather
than rewriting an acquisition or automated fit. A selection record chooses one
candidate for an operational role and records the decision source. Candidates
must never be silently averaged.

For compatibility, a completed selection may materialize
`analysis_metadata.attrs["dish_mask"]`, but that attribute becomes a derived
projection containing the selected run reference and digest. It is not the
primary evidence or candidate authority.

Geometry metadata is tiny. Its physical layout should favor clarity and
atomic publication; indexed sharding is not a relevant optimization for these
records.

## Operational policy

Loading/gating policy remains explicit:

- `off`: do not load or apply registered masks;
- `if_available`: accept independently valid per-camera recording-bound masks
  and report missing or partial cameras;
- `required`: fail an affected camera when its exact mask is unavailable or
  invalid.

Candidate selection adds an independent readiness state:

- acquisition imported, comparison pending;
- agreement passed and acquisition gate corroborated;
- review required;
- reviewer-selected acquisition candidate;
- reviewer-selected Palette candidate;
- selected manual replacement.

Existing workflows retain their old default until registered geometry is
explicitly requested. A production workflow may additionally require a
completed comparison or manual selection before refinement.

If the acquisition and Palette fits agree, the acquisition
`valid_detection_region` remains the natural gate because it contains the
producer-approved outward centroid forgiveness. If they disagree, retain both
and require review. If the Palette fit fails, the acquisition fit is valid but
uncorroborated; using it remains an explicit policy decision.

## Detection gating and auditability

Registered-mask gating belongs in detection quality/refinement, not the raw
YOLO writer. Raw detections remain immutable and recoverable.

The inclusive registered-gate calculation is:

```text
signed_distance_px = radius_px - hypot(x_native_px - cx, y_native_px - cy)
inside = signed_distance_px >= 0
```

Positive signed distance is inside, zero is accepted on the boundary, and
negative is outside. The canonical registered-gate rejection reason is
`outside_valid_detection_region`.

The selected gate identity and constant geometry belong once at run level.
Per detection, retain or make recoverable:

- `instance_key`;
- native-camera centroid;
- signed distance;
- gate result;
- rejection reason.

Run-level provenance records:

- selected geometry candidate and selection identities;
- mask artifact and registration identities;
- camera and arena binding;
- full-precision gate geometry;
- coordinate transform used;
- policy and comparison status;
- accepted and rejected counts.

Palette's `bbox_norm_coords` convention is center-X, center-Y, width, height,
normalized to the full frame. Its first two components are already the
centroid. A future generic gate must read the bound coordinate descriptor and
must not blindly apply a top-left `x + width / 2` convention to every array.

If detections are not already in native camera coordinates, map the centroid
back to native pixels with independently proven X and Y scales or an explicit
transform. Do not average anisotropic scales. If the mapping cannot be proven,
do not gate silently.

## Other geometry consumers

The normalized geometry import should make homography, physical scale, and
tank-design evidence available to arena assignment, tracking, subject shape,
tail kinematics, stimulus-response, and chaser metrics. Each consumer must bind
the exact geometry run and coordinate contract it used.

The following must remain distinguishable:

- acquisition physical rim;
- acquisition detection gate;
- Palette physical-rim fit;
- selected operational gate;
- nominal experimental area;
- dish-top-rim scale;
- projected-surface scale;
- camera-to-canvas homography.

`experiment_setup` may describe declared tank design and experimental layout.
It must not replace the recording-bound observed geometry or its provenance.

## Registry representation

The registry should expose enough normalized state to answer:

- Was recording-bound acquisition geometry present and valid?
- Was the local asset bundle complete?
- Did Citrus report applying the exact registration?
- Did the independent Palette fit complete?
- Did the candidates agree?
- Is review required?
- Which candidate and gate are operational?
- Was detection gating completed against that selection?

The existing `dish_mask=ok` step may remain a high-level readiness projection,
but it cannot represent these distinctions alone. New normalized fields or
stage details should preserve artifact ID, registration ID, comparison status,
selection status, and selected geometry-run identity.

## Staging disposition and cleanup

Geometry files are `retained_authority` while they are unconsumed acquisition
inputs. They may become cleanup-eligible only after:

1. The exact geometry subtree has been copied to every required recording.
2. Every copy has been verified against the snapshot and asset manifest.
3. The recording/camera/arena binding has been validated.
4. The acquisition candidate has been successfully published to the analysis
   Zarr with source checksums.
5. A fresh disposition manifest finds no retained or unknown geometry.

Independent Palette fitting and candidate agreement are not required to prove
that the staging bytes were safely preserved, but they may be required by a
later production-analysis policy before using the mask.

The cleanup tool remains separate, explicit, and dry-run-first. It revalidates
immediately before deletion and emits a cleanup receipt. Failed July 2 batches
and any legacy batch containing unorganized source video remain excluded.

## Legacy behavior

Older recordings may lack a snapshot pointer, embedded recording contract, or
schema-v2 rim observation. Their structured status is
`legacy_missing_recording_bound_mask`.

For these recordings Palette must not:

- follow an absolute calibration path;
- query a current active pointer;
- choose the newest observation;
- use another camera or arena;
- substitute Citrus runtime registration alone;
- substitute nominal `experimental_area` geometry;
- relabel a legacy/manual mask as recording-bound evidence.

An explicitly configured legacy/manual import may remain available as a
separate authority. The current July 21 Batman recordings are expected to
remain ungated under the registered-mask policy.

## Historical recovery receipt

The July 22 Batman acquisition bundles expose a narrower historical defect than
the July 21 recordings. Their exact Orange contract and complete 49-file asset
manifest are present and verify successfully, but the original
`recording_snapshot.json` does not contain the producer-declared pointer from
the snapshot to `recording_geometry_contract.json`. Their arena H5 files also
say that recording geometry was not referenced. Palette must not edit either
producer artifact to make that missing relationship appear native.

The approved recovery is therefore a separate immutable sidecar at:

```text
<recording>/raw/recording_geometry_recovery.json
```

The unmodified acquisition bundle is copied to:

```text
<recording>/raw/recording_geometry_bundle/
```

Each receipt binds exactly one target recording and contains:

- the original snapshot SHA-256, Orange recording ID, and explicit
  `contract_pointer_status = missing`;
- the exact contract and asset-manifest SHA-256 values;
- the exact target H5 relative path, full-file SHA-256, session UUID, camera
  serial, arena ID, and its explicit `not_referenced` geometry status;
- the exact resolved camera/arena entry proved by the contract;
- a human/operator approval identity and construction timestamp;
- the Palette Git identity and recovery algorithm version; and
- explicit negative claims: producer artifacts were not changed, Orange did
  not declare the missing snapshot link, and Citrus runtime application is not
  claimed.

The receipt is an attestation of an operator-approved historical association,
not a rewritten acquisition record and not a cryptographic human signature.
Its stable `receipt_id` is content-derived from the contract, H5, camera, and
arena. Its serialized bytes receive their own SHA-256 when loaded or published.

Receipt validation always re-hashes and revalidates the referenced evidence.
It fails closed unless:

1. the source bundle is a complete, checksummed Orange schema-v1 bundle;
2. the producer snapshot pointer is genuinely absent, rather than invalid or
   contradictory;
3. the target H5 has no verified producer-native geometry authority;
4. the H5 camera and arena select exactly one resolved contract entry;
5. native dimensions agree between snapshot and rim observation;
6. the full-precision rim observation agrees with the asset-manifest checksum;
7. every receipt path is relative and remains within the recording `raw/`
   directory, so moving or renaming the enclosing recording root does not
   invalidate the evidence; and
8. every recorded digest and identity still matches current bytes.

The normal Orange-folder loader remains unchanged and continues to classify
the copied pointerless bundle as `legacy_missing_recording_bound_mask`. A caller
must explicitly invoke the recovery-receipt loader. Recovered masks are labeled:

```text
source_kind = palette_recovered_recording_geometry
producer_contract_linkage_status = operator_approved_recovery_receipt
citrus_registration_status = missing
selected_daily_registration_applied_by_citrus = false
independent_fit_required_before_operational_use = true
```

Thus recovery permits acquisition-candidate import and comparison without
silently making the acquisition fit operational. Palette's independent
recording-image fit and the candidate comparison remain required before the
recovered acquisition gate may be selected for production use.

Publication is receipt-last. Palette first verifies and atomically publishes
the fixed-layout bundle, then builds and atomically writes the receipt, then
reopens and revalidates the complete chain. A copied bundle without a receipt
does not grant recovered reader authority. Existing verified receipts are
idempotent and are never overwritten.

The operator workflow is dry-run-first:

```bash
scripts/py -m fisheye.utils.recover_recording_geometry \
  --source-bundle <staging-session-root> \
  --target-recording <recording-root> \
  --approved-by <operator>

scripts/py -m fisheye.utils.recover_recording_geometry \
  --source-bundle <staging-session-root> \
  --target-recording <recording-root> \
  --approved-by <operator> \
  --apply
```

The command accepts repeated `--target-recording` arguments. Every target is
independently hashed and checked; the utility never infers a target by scanning
for a newest calibration, contract, or recording.

## Implementation sequence

The recommended slices are:

1. **Organization and preservation**
   - Preserve the version-1 geometry subtree exactly.
   - Verify fan-out copies.
   - Backfill the two July 22 Batman session batches without deleting staging
     inputs.

2. **Shared loader and model**
   - Implement the path-safe Orange-folder adapter.
   - Implement the exact-byte Citrus-H5 adapter.
   - Resolve both into one immutable normalized model.
   - For the bounded July 22 pointer omission only, preserve the original
     bundle and publish one explicit per-recording recovery receipt.

3. **Acquisition candidate publication**
   - Implemented as metadata-only
     `analysis/arena_geometry_runs/<candidate_id>` groups.
   - Physical rim and valid gate are published separately.
   - Asset completeness and runtime-registration status are retained.
   - Completion makes a candidate readable, never operationally selected.

4. **Independent fitter and comparison**
   - Multi-window robust fitting and a reviewed, pointerless Palette candidate
     are implemented for the first Batman canary.
   - The exact raw-detection disagreement audit and review rendering are
     implemented.
   - Repeat across the four-camera canary before defining any automatic
     agreement threshold or operational selection policy.

5. **Selection and compatibility projection**
   - Add explicit selection/review state.
   - Materialize the selected compatibility `dish_mask` without losing the
     candidate reference.

6. **Detection quality/refinement integration**
   - Bind by camera, arena, geometry candidate, and coordinate contract.
   - Preserve raw detections and keyed per-row decisions.
   - Bypass the legacy 0.5 mm expansion for registered valid gates.

7. **Registry and cleanup**
   - Expose normalized geometry/comparison/selection status.
   - Regenerate staging disposition manifests.
   - Canary explicit cleanup only after fresh verification.

## Test requirements

Focused unit and integration coverage should include:

- exact folder discovery and traversal rejection;
- exact-byte H5 checksum handling for bytes, strings, NumPy scalars, and
  trailing newlines;
- valid single- and multi-camera contracts;
- exact camera/arena selection from grouped registrations;
- selected-partial per-camera salvage under `if_available`;
- contract, scope, manifest, observation, and runtime checksum mismatches;
- schema-version rejection;
- native-dimension disagreement;
- physical rim versus outward valid-gate separation;
- concentricity, finite values, positive radius, and inclusive boundary tests;
- historical registration validity without comparison to current wall time;
- literal path preservation without treating the path as portable identity;
- independent multi-window fit stability and failure;
- agreement and review-required comparison outcomes;
- raw-detection auditability and `instance_key` preservation;
- explicit bbox-coordinate convention binding;
- registered-gate execution without the legacy 0.5 mm expansion;
- legacy Batman recordings remaining `legacy_missing_recording_bound_mask`;
- staging cleanup remaining blocked until verified preservation and candidate
  publication.

## Remaining decisions

Two implementation decisions intentionally remain open:

1. The final Zarr group names and schemas for comparisons and selections. The
   candidate family is now fixed at `analysis/arena_geometry_runs`.
2. Numerical agreement thresholds. These should be derived from the July 22
   four-camera canary and repeated-recording variability after the independent
   fitter exists.

No implementation should silently choose thresholds, overwrite one candidate
with another, or activate registered gating before those decisions are made
explicit.
