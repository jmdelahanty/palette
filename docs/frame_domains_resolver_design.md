# FrameDomains Resolver Design

<!-- contract-meta
status: approved (2026-07-05, maintainer; see Approval Record)
created: 2026-07-05
owner: jeremy
related: docs/identity_lineage_staleness_review.md, docs/palette_cli_narrow_waist_design.md,
         docs/run_resolution_semantics.md,
         docs/diagnostics/redscare_keypoint_frame_axis_diagnosis_2026-07-02.md
-->

## Executive Summary

Palette has enough recorded lineage to make frame-domain conversions explicit, but the
conversion logic is still distributed across producers, readers, validators, diagnostics,
and visualizers. This design introduces one read-only `FrameDomains` resolver attached to
the existing `Recording` accessor. It answers: "which frame domain is this value in?",
"how do I convert it?", and "how long is the corresponding per-frame axis?" without
letting each caller re-derive the math.

The main decision is strictness. The resolver should construct conversion edges from
recorded mapping arrays and explicit stage/schema semantics, never from array lengths or
`max()+1`. If a mapping is absent or ambiguous, conversion refuses loudly and
`capabilities()` explains what is available.

The `/nvme1` census supports the need for this resolver:

- 125 zarrs were readable without errors: 68 training zarrs and 57 analysis zarrs.
- 56 zarrs carry `raw_video/original_frame_indices`; observed `frame_step` values were
  `100` for 52 zarrs and `1700` for 4 zarrs.
- In sampled zarrs, stage runs contain both sample-local/stored-axis `frame_indices` and
  source/original-axis `source_frame_indices`. For example, sampled crop runs split 56
  stored-axis-like and 56 original-axis-like run-frame arrays in the same corpus.
- No `source_crop_video_frame_indices` or `source_crop_local_frame_ids` arrays were found
  under `/nvme1/recordings`; crop-video dropped-frame behavior is therefore verified from
  current writer/reader code paths, not from this real-store census. The implementation
  slice needs a synthetic fixture for drops plus subsampling unless a `/groups` corpus is
  separately approved for read-only validation.

## Current Map

The identity review declares the frame-domain problem as Rec 3: acquisition-local frame
ID, raw-video frame number, stored zarr index, and crop-video frame index all coexist, and
callers independently redo the conversions (`docs/identity_lineage_staleness_review.md:94`).
The same review explicitly ranks a `FrameDomains` resolver as the next fix after
authoritative runs and instance keys (`docs/identity_lineage_staleness_review.md:108`).

The read-side integration point now exists. `Recording` is a read-only zarr handle
(`src/fisheye/shared/recording.py:166`) and already resolves runs through the shared
`RunResolution` dispatcher (`src/fisheye/shared/recording.py:197`). Its `.detections()`,
`.keypoints()`, and `.subject_masks()` methods default to authoritative run resolution
(`src/fisheye/shared/recording.py:266`, `src/fisheye/shared/recording.py:284`,
`src/fisheye/shared/recording.py:302`). `FrameDomains` should hang off this handle as
`recording.frame_domains(...)`, not as another free function each caller wires up.

`RunResolution` is the right model for the resolver interface. It names resolution modes
centrally (`src/fisheye/shared/run_resolution.py:24`) and returns structured resolution
provenance (`src/fisheye/shared/run_resolution.py:33`). `FrameDomains` should similarly
return conversion provenance: which mapping arrays/attrs were used, which run was
resolved, and which domain edge failed if conversion was impossible.

One caveat: refined-detect authority is mid-conversion. `RunResolution` still bridges
`detect_review_status_latest` for refined-detect parents
(`src/fisheye/shared/run_resolution.py:20`, `src/fisheye/shared/run_resolution.py:293`).
Design text should treat that as transitional evidence, not a permanent frame-domain
contract.

### Producer and Reader Evidence

Sampled raw imports are produced by `import_video.py`. `_compute_frame_indices()` samples
`range(0, total_frames, frame_step)` (`src/fisheye/capture/import_video.py:68`), sampled
training mode computes `frame_indices_to_import` from that helper
(`src/fisheye/capture/import_video.py:784`), and sampled metadata records `import_mode`,
`frame_step`, `original_video_length`, and `imported_frame_count`
(`src/fisheye/capture/import_video.py:991`). The stored mapping array is
`raw_video/original_frame_indices` (`src/fisheye/capture/import_video.py:1146`). The zarr
structure doc states that this array maps sampled imports to original video indices
(`src/fisheye/docs/zarr_structure.md:219`) and that clipped training zarr
stage-level `frame_indices` remain sample-local while source provenance lives elsewhere
(`src/fisheye/docs/zarr_structure.md:235`).

Acquisition crop-video writers already persist row-aligned crop-video mappings. The
analysis crop builder stores `frame_indices`, `source_frame_indices`,
`source_recording_frame_ids`, `source_crop_meta_row_indices`,
`source_crop_video_frame_indices`, `source_crop_local_frame_ids`, and `frame_counts`
(`src/fisheye/utils/build_analysis_acquisition_crop_run.py:407`). Its attrs define
`source_crop_video_frame_indices` as zero-based acquisition crop-video frames and
`source_crop_local_frame_ids` as Orange acquisition local frame IDs, not video frame
indices (`src/fisheye/utils/build_analysis_acquisition_crop_run.py:470`).

The acquisition crop pose exporter follows the same pattern. It selects source frames and
crop-video/local-frame IDs into one `_Selection`
(`src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py:120`), fills those
fields from keypoint and crop metadata (`src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py:556`),
and writes both crop and keypoint arrays from that lineage
(`src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py:747`). Its current
`n_keypoints` sizing uses `frame_counts.shape[0]`, the pattern that fixed the RedScare
class of bug (`src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py:742`).

Hybrid acquisition/offline crop runs introduce supplemental pixel rows. The writer builds
`frame_indices` and `source_frame_indices` for both online and offline rows
(`src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:610`) and marks offline
rows with `source_crop_video_frame_indices=-1`
(`src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:636`). It separately
records `source_pixel_kind_codes` and `supplemental_cache_row_indices`
(`src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:686`,
`src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:702`). Therefore the
`-1` value is not "no source frame at all"; it is "no acquisition crop-video frame for
this row." The row can still have a canonical source frame.

`shared/crop_image_source.py` is the current crop-video reader. It requires
`source_crop_video_frame_indices` for acquisition crop-video sources
(`src/fisheye/shared/crop_image_source.py:780`) and validates the array length against ROI
count (`src/fisheye/shared/crop_image_source.py:786`). It refuses negative crop-video
indices for video-backed rows (`src/fisheye/shared/crop_image_source.py:1058`) and routes
supplemental rows through `supplemental_cache_row_indices`
(`src/fisheye/shared/crop_image_source.py:1077`). This fail-loud pattern is the right
model for the resolver.

Distributed local arithmetic remains. `detect_quality.py` re-detects sampled imports from
`import_mode`, `frame_step`, and `original_frame_indices`
(`src/fisheye/refinement/detect_quality.py:118`) and locally pads or recomputes
`frame_counts` (`src/fisheye/refinement/detect_quality.py:593`,
`src/fisheye/refinement/detect_quality.py:827`). `detect_keypoints_yolo.py` still has a
fallback that derives total frames from `frame_indices.max()+1` when no better source is
available (`src/fisheye/detection/detect_keypoints_yolo.py:827`). The RedScare diagnosis
shows why that pattern is risky: the affected runs were safe only because the malformed
array was a summary array and the row-level keypoint arrays stayed aligned
(`docs/diagnostics/redscare_keypoint_frame_axis_diagnosis_2026-07-02.md:13`). That doc
explicitly points to `FrameDomains` as the general fix
(`docs/diagnostics/redscare_keypoint_frame_axis_diagnosis_2026-07-02.md:123`).

## Census Results

The census was run read-only with
`/home/delahantyj@hhmi.org/miniconda3/envs/palette-py311/bin/python` against
`/nvme1/recordings/*/zarr/*.zarr`. It opened zarr stores in read mode and wrote only a
temporary JSON report under `/tmp`.

| Result | Count / observation | Interpretation |
| --- | ---: | --- |
| zarrs scanned | 125 | `/nvme1` was reachable; no substitute corpus was used. |
| analysis zarrs | 57 | Current corpus includes analysis outputs. |
| training zarrs | 68 | Current corpus includes training outputs. |
| zarr open/read errors | 0 | Census is complete for paths discovered. |
| `raw_video/original_frame_indices` present | 56 zarrs | Sampled stored-zarr to source-frame mapping is common. |
| sampled raw import `frame_step=100` | 52 zarrs | Dominant sampled-training stride. |
| sampled raw import `frame_step=1700` | 4 zarrs | Sparse sickyfish-style training imports. |
| `source_crop_video_frame_indices` present | 0 runs | `/nvme1` does not cover acquisition crop-video mapping arrays. |
| `source_crop_local_frame_ids` present | 0 runs | Orange local-ID mapping is code-path-only in this census. |
| supplemental pixel rows | 0 runs | Hybrid supplemental behavior is code-path-only in this census. |
| frame-count/source-max mismatches | 104 runs | Mostly sampled stores where source-frame values live on an original/source axis while `frame_counts` lives on the stored/run axis. This is evidence against `max()+1`, not automatically corruption. |

Focused sampled-zarr classification found mixed frame-value domains by stage:

| Stage parent | Runs with frame values | Stored-axis-like | Source/original-axis-like |
| --- | ---: | ---: | ---: |
| `detect_runs` | 70 | 66 | 4 |
| `crop_runs` | 112 | 56 | 56 |
| `keypoints_runs` | 119 | 105 | 14 |
| `refined_keypoints_runs` | 115 | 103 | 12 |
| `subject_mask_runs` | 284 | 284 | 0 |
| `refined_subject_masks_runs` | 238 | 238 | 0 |

Example: in sampled zarrs with `frame_step=100` and `raw_video/original_frame_indices`
length 231, some crop runs have `frame_indices` max 230 (stored axis) while paired
`source_frame_indices` reach 23000 (original/source axis). This is valid lineage but
invalidates any consumer that assumes an array name alone implies one global frame
domain.

No `/nvme1` recording showed both raw subsampling and acquisition crop-video dropped-frame
mappings. The implementation slice must therefore include a synthetic fixture exercising
subsampling plus non-identity crop-video mapping. A separate `/groups` read-only census
would be useful if the maintainer wants real acquisition-crop coverage before approving
consumer migration.

## Domain Contract

### Frame Domains

| Domain | Operational meaning | Recorded mappings / evidence | Notes |
| --- | --- | --- | --- |
| `acquisition_frame` | Canonical zero-based source frame on the acquisition timeline. | `source_frame_indices` when present; `frame_indices` only when the stage contract or explicit attr says it is acquisition/source-domain. | This is the internal canonical representation. `source_recording_frame_ids = acquisition_frame + 1` is a display/external ID, not the canonical array index. |
| `stored_zarr_frame` | Zero-based row in stored `raw_video/images_*` arrays. | `raw_video/original_frame_indices` maps stored row -> source/original frame for sampled imports. | For full imports without a stored identity map, conversion to acquisition must be unavailable unless a future explicit identity mapping is written. Do not infer identity from equal lengths. |
| `source_video_frame` | Frame number in the source full-frame video named by the zarr. | Usually the same integer space as `acquisition_frame` for current Orange full-frame MP4s; clipped zarr docs define `original_frame_indices` as parent frame indices. | Keep the name distinct so clipped/training zarrs can point to parent videos without overloading acquisition-local semantics. |
| `run_frame` | Contextual per-run frame axis for arrays shaped `(n_frames,)` such as `frame_counts` and `n_keypoints`. | `frame_counts.shape[0]` declares count only; row arrays plus explicit semantics declare conversion. | This domain is always scoped to a resolved run. It is not globally comparable across runs without a mapping edge. |
| `crop_video_frame` | Zero-based decoded frame in an acquisition crop video. | `source_crop_video_frame_indices`, row-aligned with ROI rows; semantics attr in acquisition crop writers. | Dropped crop-video frames make this non-identical to acquisition frame. Negative values mean unmappable for crop-video, not no acquisition frame. |

### Row Axes

ROI row axes are not frame domains, but the resolver must expose row-to-frame mapping
because most frame-domain values are row-aligned. A crop/keypoint/mask row can map to
`acquisition_frame`, `stored_zarr_frame`, `crop_video_frame`, or a supplemental cache row,
depending on the arrays present.

`supplemental_cache_row` is a pixel-source row axis, not a fifth frame domain. Hybrid
offline rows should convert to `acquisition_frame` through their `frame_indices` /
`source_frame_indices` but should fail if converted to `crop_video_frame`, because
`source_crop_video_frame_indices=-1` means no video frame exists for that source.

## Canonical Semantics

All cross-domain conversions go through `acquisition_frame` where a canonical mapping is
available:

1. `stored_zarr_frame -> acquisition_frame` uses `raw_video/original_frame_indices`.
2. `crop_video_frame -> acquisition_frame` is built from paired row-aligned
   `source_crop_video_frame_indices` and `source_frame_indices` / declared acquisition
   `frame_indices`.
3. `run_frame -> acquisition_frame` is available only when the run records an explicit
   mapping. `frame_counts.shape[0]` answers count, not conversion.
4. `row -> acquisition_frame` uses the row's recorded frame mapping.

`source_recording_frame_ids` should remain a display/import compatibility alias. It is
commonly `frame_indices + 1` in acquisition crop writers
(`src/fisheye/utils/build_analysis_acquisition_crop_run.py:325`,
`src/fisheye/utils/export_acquisition_crop_pose_training_zarr.py:562`) and should not be
the internal canonical domain unless the Orange acquisition contract later requires a
native, non-zero-based local ID as the primary key. `source_crop_local_frame_ids` should be
preserved as an external Orange local-frame ID and exposed in resolver metadata, but it is
not the crop-video frame domain; the current writer says so explicitly
(`src/fisheye/utils/build_analysis_acquisition_crop_run.py:471`).

## Resolver API

Home: `src/fisheye/shared/frame_domains.py` in a later implementation slice, exposed via
`Recording.frame_domains()`.

Sketch:

```python
class FrameDomain(str, Enum):
    ACQUISITION = "acquisition_frame"
    STORED_ZARR = "stored_zarr_frame"
    SOURCE_VIDEO = "source_video_frame"
    RUN_FRAME = "run_frame"
    CROP_VIDEO = "crop_video_frame"


@dataclass(frozen=True)
class FrameDomainEdge:
    source: FrameDomain
    target: FrameDomain
    source_path: str
    mapping_arrays: tuple[str, ...]
    run_name: str | None = None
    confidence: Literal["explicit", "stage_contract"] = "explicit"


@dataclass(frozen=True)
class FrameDomainCapabilities:
    domains: frozenset[FrameDomain]
    edges: tuple[FrameDomainEdge, ...]
    missing: tuple[str, ...]


class FrameDomains:
    def capabilities(self) -> FrameDomainCapabilities: ...
    def count(self, domain: FrameDomain, *, stage: str | None = None, run: str | None = None) -> int: ...
    def convert(
        self,
        values: npt.ArrayLike,
        source: FrameDomain,
        target: FrameDomain,
        *,
        stage: str | None = None,
        run: str | None = None,
    ) -> np.ndarray: ...
```

`Recording.frame_domains()` should accept the same run-resolution vocabulary already used
by `.detections()`, `.keypoints()`, and `.subject_masks()`. The resolver should record
which run was resolved and which mapping arrays were used. That makes a conversion
auditable in the same way `RunResolutionResult` makes "which run" auditable.

### Fail-Loud Policy

Use one policy: `convert()` raises `FrameDomainUnmappedError` if any requested value is
unmappable. The exception includes source domain, target domain, stage/run, mapping arrays
consulted, and the first few failing values/positions.

No sentinels and no masked return by default. Masked returns are easy to ignore, and
sentinel `-1` is already overloaded in existing crop-video metadata. Callers that need to
partition data can use `capabilities()` or a separate explicit helper such as
`is_mappable(values, source, target, ...)`.

### Missing-Mapping Degradation

Missing mappings disable only the affected conversions. They do not disable the whole
resolver.

Examples:

- A full raw import with no `raw_video/original_frame_indices` can expose
  `count(STORED_ZARR)` from stored arrays, but `STORED_ZARR -> ACQUISITION` is unavailable
  unless an explicit identity mapping or trusted stage/schema contract exists.
- A crop run with `source_crop_video_frame_indices` but no `source_frame_indices` can
  expose crop-video count and ROI-row to crop-video conversion, but cannot convert
  crop-video frames to acquisition frames.
- A hybrid row with `source_crop_video_frame_indices=-1` and `source_pixel_kind_codes != 0`
  can map row -> acquisition if `source_frame_indices` exists, but row -> crop-video
  raises.

The implementation may include table-driven legacy adapters for known historical writer
contracts, but those adapters must be explicit and test-covered. They may use stage name,
schema id, attrs, and array names. They must not classify a domain by comparing
`max(frame_indices)` with an array length at runtime.

## Migration Plan

| Phase | Change | Validation |
| --- | --- | --- |
| 1 | Add `FrameDomain`, `FrameDomains`, `Recording.frame_domains()`, and synthetic fixtures for sampled raw imports, crop-video drops, and hybrid supplemental rows. No production consumers. | Unit tests for each edge and missing-edge failure. Synthetic fixture must combine subsampling and non-identity crop-video mapping because `/nvme1` lacks that real case. |
| 2 | Add read-only diagnostics that compare resolver output with existing local arithmetic in `detect_quality.py`, `detect_keypoints_yolo.py`, and crop-image source paths. | Shadow mode only; emit warnings/report, no behavior change. |
| 3 | Migrate high-risk consumers: `detect_keypoints_yolo.py` total-frame selection, `detect_quality.py` coverage/count logic, and `shared/crop_image_source.py` crop-video row resolution. | Focused tests should assert removal of `max()+1` fallbacks and direct crop-video arithmetic from those paths. |
| 4 | Migrate crop-run builders and exporters so new runs stamp explicit frame-domain semantics for `frame_indices`, `source_frame_indices`, `frame_counts`, and crop-video mappings. | Writer tests verify attrs/arrays allow resolver reconstruction without stage-specific guessing. |
| 5 | Migrate diagnostics/visualizers and training loaders. | Existing CLI/notebook behavior preserved, but source frame labels come from resolver conversions. |
| 6 | Add forcing function. | Grep-based CI check with an allowlist for producers/resolver internals: flag `frame_indices.max()+1`, `np.bincount(frame_indices, minlength=...)`, direct `original_frame_indices` conversion, and direct `source_crop_video_frame_indices` frame reads outside approved modules. |

The forcing function cannot be import-linter alone because the hazard is local arithmetic,
not dependency direction. A simple static check is honest and maintainable if the allowlist
is small: `shared/frame_domains.py`, writer modules that create mappings, and tests.

## Non-Goals

- No historical data rewriting.
- No RedScare backfill; the RedScare keypoint mismatch was already diagnosed as safe for
  training but malformed in one summary array.
- No change to how `import_video.py` currently records sampled mappings. If future work
  adds explicit identity maps for full imports, that is a producer enhancement, not part
  of the resolver design itself.
- No retirement of legacy detect-review authority pointers. They are mid-conversion and
  relevant only because run resolution chooses which run's frame mappings the resolver
  reads.

## Decisions Needing Maintainer Eyes

| Decision | Recommendation | Alternative | Why it matters |
| --- | --- | --- | --- |
| Canonical internal frame ID | Use zero-based `acquisition_frame` as the canonical conversion hub. Keep `source_recording_frame_ids` and `source_crop_local_frame_ids` as external aliases/metadata. | Make Orange's native local frame ID the canonical key. | Current arrays and writers mostly use zero-based frame indices; switching to a 1-based/native ID would require wider producer changes. |
| Supplemental `-1` rows | Treat `-1` crop-video indices as unmappable to `crop_video_frame`, not as a fifth frame domain. | Model `supplemental_cache_row` as a separate `FrameDomain`. | Supplemental rows can still map to acquisition frames. Making them a frame domain would blur pixel-source rows with time/frame identity. |
| Missing mappings | `convert()` raises by default; `capabilities()` exposes available edges. | Return masked arrays for partially mappable conversions. | Raising prevents silent partial conversions. Masked arrays are useful but too easy to ignore in downstream science code. |
| Legacy adapters | Allow explicit, table-driven stage/schema adapters when old stores lack attrs, but never length-derived guesses. | Strict arrays-only mode for all legacy data. | Strict arrays-only is cleaner but may make too many existing stores unreadable through the accessor. Adapters preserve usability while keeping the rules auditable. |
| Real crop-drop validation | Require a synthetic fixture in implementation, and optionally approve a `/groups` read-only census for acquisition-crop recordings. | Approve design with only code-path evidence. | `/nvme1` did not contain crop-video mapping arrays, so the most important dropped-frame case was not observed in real stores. |

## Site-List Corrections

- The brief's "rows with no source frame at all" description for hybrid `-1` rows is too
  broad. Current hybrid rows still record `frame_indices` and `source_frame_indices`; the
  `-1` applies to `source_crop_video_frame_indices` for rows sourced from supplemental
  cache.
- The `/nvme1` corpus did not contain `source_crop_video_frame_indices` or
  `source_crop_local_frame_ids`, so it cannot validate dropped crop-video frames.
- The RedScare `n_keypoints` bug is now best used as a canonical example of the bug class,
  not as evidence that current `export_acquisition_crop_pose_training_zarr.py` is wrong;
  that exporter uses `frame_counts.shape[0]` for `n_keypoints`.
- Cite `shared/` homes (`shared/recording.py`, `shared/run_resolution.py`,
  `shared/crop_image_source.py`) rather than the old `utils/` shims added during the utils
  reorganization.

## Open Questions Blocking Approval

1. Should implementation be allowed to read a `/groups` acquisition-crop corpus for a
   second census, or is a synthetic fixture enough for the first implementation slice?
2. Should future full imports write an explicit identity mapping for
   `stored_zarr_frame -> acquisition_frame`, so the resolver can avoid legacy
   arrays-only degradation on full zarrs?
3. Which legacy stage/schema adapters are acceptable for old stores that do not stamp
   frame-domain attrs? The design recommends adapters, but the approved allowlist should
   be explicit before consumer migration starts.

## Approval Record (2026-07-05)

Maintainer approved the design with all five recommended decisions and these answers:

1. **Both, sequenced.** A synthetic drops+subsampling fixture is sufficient to BUILD the
   resolver (first implementation slice). A read-only `/groups` acquisition-crop census
   is a HARD GATE before migrating any crop-video consumer — building is unblocked,
   crop-video migration is not, until real dropped-frame stores are observed.
2. **Yes.** Future full imports write the explicit `stored_zarr_frame ->
   acquisition_frame` identity mapping (stamp-going-forward, same philosophy as
   completion-epoch provenance). Historical stores are not backfilled.
   *Implemented (slice 1, edca55c):* the map is written at
   `raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame` — a new
   namespace rather than overloading `original_frame_indices`, so sampled-import
   detection behavior is untouched. Consumer migrations should read it there.
3. **Deferred to consumer-migration slices.** No upfront adapter allowlist; each adapter
   is approved one at a time with per-store evidence when a migration slice needs it.
   The first implementation slice ships ZERO legacy adapters.
