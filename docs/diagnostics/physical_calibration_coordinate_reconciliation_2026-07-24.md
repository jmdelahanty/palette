# Physical-calibration and coordinate-contract reconciliation

Date: 2026-07-24

Status: read-only repository and archive assessment. No Zarr, registry, selector,
or cluster state was modified by this assessment.

## Executive conclusion

The Sleepyfish `v007` track-kinematics publications are not presently readable
from `sun`, but the evidence does **not** indicate corrupt track or speed data.
The failure is contract/version skew between two branches that diverged from
commit `4e76963a`:

- current `sun` contains the newer point-versus-half-open-bbox coordinate
  contracts; and
- `agent/physical-calibration-guard-20260720` contains recording-level physical
  calibration, merged collection-proxy support, and several track-publication
  fixes used to create the Sleepyfish sealed physical runs.

The physical branch is 12 unmatched commits ahead of the shared base, while
current `sun` is 41 commits ahead on the other side. All 12 physical commits
have distinct patch identities according to `git cherry`; they must not be
merged or cherry-picked as an undifferentiated stack.

The correct production repair is:

1. preserve every existing v1 publication unchanged;
2. retain an explicitly historical, exact-schema reader for validation and
   audit only;
3. port recording-level physical authority onto current `sun` semantics;
4. create a new immutable current-contract successor for merged proxy geometry;
5. materialize new track-kinematics publications from that successor; and
6. run swim-bout detection and regenerate the intern exports from the new
   tracks.

The existing `v007` arrays must not be relabeled as current v2 simply because
their values look plausible.

## Triggering production failure

Three guarded Citrus jobs were submitted from Palette commit `a244b848`:

| Camera | Job | Requested output |
|---|---:|---|
| 2010093 | 153168699 | `swim_bouts_sleepyfish_cam2010093_sealed_physical_20260724_v001` |
| 2010094 | 153168698 | `swim_bouts_sleepyfish_cam2010094_sealed_physical_20260724_v001` |
| 2010096 | 153168700 | `swim_bouts_sleepyfish_cam2010096_sealed_physical_20260724_v001` |

All three failed before publication with the same error:

```text
Canonical position rowset must declare exactly one detection-mapping or
crop-selection lineage.
```

No requested swim-bout run was published. The immutable execution directories,
reports, stderr logs, and failed status files remain as provenance.

## What is sealed in the three archives

The selected physical track runs are:

```text
track_kinematics_sleepyfish_cam2010093_sealed_physical_20260720_v007
track_kinematics_sleepyfish_cam2010094_sealed_physical_20260720_v007
track_kinematics_sleepyfish_cam2010096_sealed_physical_20260720_v007
```

Metadata-only inspection found the same important structure in all three:

| Property | Observed value |
|---|---|
| Track schema | `analysis.track_kinematics_runs`, schema version 1 |
| Track completion | `complete` |
| Track selector eligibility | `true` |
| Physical authority kind | `recording_calibration` |
| Source rowset schema | `palette_clipped_collection_merged_proxy_crop_run_v1` |
| Source rowset coordinate label | `canonical_v2` |
| Merged collection mapping | present |
| Detection acquisition mapping | absent |
| Ordinary crop selection | absent |
| Bbox projection record | schema version 1 |
| Bbox-center derivation record | schema version 1 |
| Separate half-open source-camera frame | absent |

The current `sun` resolver recognizes only detection-acquisition or ordinary
crop-selection lineage. The physical branch additionally recognizes merged
collection-proxy lineage. That older merged-proxy reader, however, was written
against bbox projection and center-derivation schema v1.

Current `sun` instead requires:

```text
DETECTION_BBOX_PROJECTION_SCHEMA_VERSION = 2
BBOX_CENTER_DERIVATION_SCHEMA_VERSION = 2
SOURCE_CAMERA_POINT_PIXEL_CONVENTION = continuous
SOURCE_CAMERA_BBOX_PIXEL_CONVENTION = pixel_edge_half_open
```

It also requires distinct point and bbox frame arguments when constructing
detection frame evidence. Therefore the old merged-proxy reader cannot simply
be cherry-picked: it is not compatible with the current point/edge API or the
persisted v1 records.

The rowset-level `coordinate_contract = canonical_v2` label is insufficient to
prove current semantics. The versioned projection, derivation, frame, lineage,
and digest-bound records are authoritative.

## Read-only compatibility proof

Cam 2010093 was loaded using the exact tip of the branch that produced and
understands these publications (`e257a135`). The normal strict
`load_track_kinematics_track` reader succeeded and reported:

```text
rows: 1,182,938
finite filtered physical-speed samples: 1,181,698
```

The same run fails under current `sun` while resolving its source coordinate
authority, before the logical speed table is returned.

This establishes a narrow conclusion:

- the sealed `v007` publication remains internally valid under its producing
  contract implementation; and
- current failure is caused by reader/contract divergence.

It does not authorize treating the old source geometry as current v2 or using
an unverified legacy array reader for new scientific products.

## Branch topology

```text
                          physical branch (12 unmatched commits)
                         /
shared base 4e76963a ----
                         \
                          sun (41 unmatched commits at assessment time)
```

A textual three-way merge shows overlapping production edits in at least:

- `scripts/submit_analysis_workflow_bsub.sh`;
- `src/fisheye/analysis/import_stimulus_to_zarr.py`;
- `src/fisheye/analysis/track_kinematics.py`;
- `src/fisheye/analysis_workflows/materializers/track_kinematics.py`;
- `src/fisheye/detection/detect_yolo.py`;
- `src/fisheye/shared/observation_coordinate_publication.py`;
- `src/fisheye/utils/backfill_analysis_calibration.py`; and
- their corresponding unit-test modules.

The new physical-authority and migration modules add conceptually necessary
behavior, but their APIs and tests were authored before the current coordinate
contract landed.

## Commit-by-commit disposition

### `1cfbd16f analysis: require sealed recording calibration`

Disposition: **port the design onto current `sun`; do not cherry-pick**.

This is the core missing capability for recordings without a stimulus H5. It
introduces a stimulus-neutral, recording-level source-camera physical authority
and lets track kinematics consume either stimulus-backed or recording-backed
calibration through one typed interface.

This behavior is required for Sleepyfish. Current `sun` only resolves the
stimulus physical-coordinate authority in normal track publication. The port
must bind the current source-camera point authority and current calibration
records without weakening either contract.

### `aac78041 migration: seal external video acquisition metadata`

Disposition: **port after review against current acquisition v2 APIs**.

This supplies an explicit migration for video-only recordings. Its purpose
remains valid, but its record construction must be compared field-for-field
with the current acquisition-camera and source-video metadata contracts.

### `423ebb00 fix: keep debug metadata out of acquisition identity`

Disposition: **retain as a small independent fix**.

Current `sun` still copies `imageio_metadata` into the input used to build the
canonical source-video record. The physical commit excludes this debug payload,
which may contain non-finite sentinels such as `nframes=inf`, while preserving
it separately for diagnostics.

### `01446eba cluster: accept git worktree analysis checkouts`

Disposition: **retain independently**.

Current `sun` still requires `$PALETTE_REPO/.git` to be a directory. A linked
worktree correctly uses a `.git` file, so the current wrapper rejects valid
worktrees. The repair should use Git's own repository discovery rather than a
filesystem-type assumption.

### `13bedc70 cluster: pin analysis imports to verified checkout`

Disposition: **skip; functionally superseded by `a244b848`**.

Current `sun` now pins `PYTHONPATH`, verifies the resolved `fisheye.__file__`
against the Git-verified checkout, and records that source file in runtime
provenance. The newer implementation should be retained.

### `4384e198 migration: seal merged proxy coordinates`

Disposition: **redesign for current v2; do not cherry-pick**.

The commit establishes the missing merged-proxy lineage concept and performs
exact source-proxy/refined-row validation. Those invariants are valuable. Its
published bbox and center records are nevertheless schema v1 and it constructs
only the earlier continuous camera frame. Current code requires distinct
continuous-point and half-open-bbox authorities and v2 derivations.

The new implementation must publish an immutable successor or overlay with
current records. It must not overwrite or relabel the historical rowset.

### `a8e69c91 fix: reuse sealed source camera frame`

Disposition: **port the invariant, adapted to both current camera frames**.

Reusing exact persisted frame authority instead of replacing it remains
correct. Current v2 needs this behavior for both continuous points and
half-open bbox edges, with exact acquisition-extent agreement.

### `e6f66c2a fix: resume partial proxy coordinate migration`

Disposition: **subsume into the redesigned successor publisher**.

The new publisher should be idempotent and transactionally resumable, but it
should resume a new immutable publication transaction—not mutate the v1 source
into a mixed v1/v2 state.

### `d0890eca fix: scale acceleration from stored pixel values`

Disposition: **skip; superseded on current `sun`**.

Current code already freezes public acceleration values to float32 before
deriving physical peers and uses the shared exact pixel-to-physical conversion
path. The physical branch's older patch identity differs, but its scientific
invariant is present in the newer implementation.

### `61c22de1 fix: keep structured lineage arrays unsharded`

Disposition: **port before any new track materialization**.

Current `sun` still applies ordinary sharding plans to structured-dtype arrays.
The physical commit records a Zarr-v3 structured-dtype workaround: one logical
chunk, no outer sharding. This is required to avoid unsupported or unsafe
structured lineage writes.

### `9db9b7ec fix: validate structured track storage fallback`

Disposition: **port immediately after `61c22de1`**.

The materializer must recognize and validate the explicit structured-array
single-chunk layout rather than reporting it as missing indexed sharding.

### `e257a135 fix: retain track lineage array handles`

Disposition: **port after reconciling current track-publication code**.

Real Zarr groups may return a new Python array wrapper for each lookup. The
time-lineage and row-identity sealing operations intentionally require exact
in-memory authority continuity. Retaining one handle per lineage array during
binding and loading is still necessary; current `sun` repeats the lookups.

## Required architecture

### 1. Historical schema-v1 reader

Implement a deliberately named historical reader that accepts only the exact
legacy tuple observed here:

- merged collection-proxy schema v1;
- collection-proxy acquisition mapping v1;
- bbox projection v1;
- bbox midpoint derivation v1;
- the exact persisted continuous source-camera authority; and
- all expected record and payload digests.

The historical reader must return a historical/audit type, not silently mint a
current `BoundSourceCameraPositionSurface`. It may prove old data and support
inspection, but normal future publications should not automatically consume it.

Hostile tests must show that changing any one schema, source row reference,
array digest, acquisition extent, or mapping record fails closed.

### 2. Recording-level physical authority on current contracts

Port the stimulus-neutral source-camera physical authority from the physical
branch. It should:

- accept exact selected-camera calibration evidence whether it originated from
  stimulus import or reviewed video-only acquisition metadata;
- bind the current continuous source-camera point frame;
- publish one typed physical-mm frame and exact reciprocal scale;
- record source kind and provenance without claiming an H5 source when none
  existed; and
- let track kinematics select stimulus authority first when explicitly pinned,
  otherwise the recording authority.

Track publication must fail closed if neither typed authority exists. Detached
`pixel_to_mm` arguments or inherited run-level scalars must remain insufficient.

### 3. Immutable current-contract merged-proxy successor

Create a new publication rather than editing the legacy auxiliary rowset in
place. The successor must prove:

- exact row identity and acquisition-frame mapping;
- exact linkage to every source proxy and refined-detection row;
- a current continuous source-camera point frame;
- a distinct current half-open source-camera bbox frame;
- normalized `cx,cy,w,h` to half-open `xyxy` projection v2;
- exact half-open-box midpoint to continuous-point derivation v2; and
- completion-last, digest-bound activation under a fresh immutable name.

If duplicating all row arrays is judged too expensive, a versioned coordinate
overlay may reference exact immutable source arrays by record and payload digest.
That overlay requires its own explicit reader contract; ordinary Zarr paths or
matching shapes are not sufficient references.

### 4. Current track successor

Materialize a new track run—do not modify `v007`—using:

- the current merged-proxy successor as position source;
- the current recording-level physical authority;
- current point and temporal bindings;
- structured-lineage single-chunk storage;
- stable array handles during binding; and
- the current full-motion publication manifest and atomic publisher.

The resulting track must load successfully from current `sun` through the
normal strict logical reader before any downstream stage is allowed to run.

## Recommended implementation order

1. Start a clean implementation worktree from the then-current `origin/sun`.
2. Port `423ebb00` and `01446eba` as independent small commits.
3. Port the structured-array storage/validation pair (`61c22de1`, `9db9b7ec`).
4. Port the stable lineage-handle invariant from `e257a135`.
5. Adapt and test recording-level physical authority from `1cfbd16f`.
6. Adapt the external-video acquisition migration (`aac78041`) and its debug
   metadata exclusion.
7. Implement the exact historical v1 reader, unavailable to normal publishing.
8. Design and implement the immutable current-contract merged-proxy successor,
   incorporating the useful invariants from `4384e198`, `a8e69c91`, and
   `e6f66c2a` without copying their old point/edge assumptions.
9. Materialize one new cam2010093 track canary.
10. Validate the canary under current `sun`, then run swim-bout detection.
11. Only after the canary passes, repeat for cams 2010094 and 2010096.
12. Regenerate the intern CSV exports and verify all four cameras explicitly.

## Canary acceptance criteria

For cam2010093, require all of the following before scaling out:

1. Every source and output run name is immutable and previously unused.
2. The source successor has distinct, verified point and bbox authorities.
3. The recording physical authority resolves without stimulus metadata.
4. The current strict track reader loads the new track publication.
5. Row, time, and source-instance lineage proofs pass.
6. Structured lineage arrays match their declared single-chunk/no-shard layout.
7. Physical arrays equal their persisted pixel peers times the exact sealed
   `mm_per_pixel`, using the declared storage dtype and rounding path.
8. Filtered physical speed contains finite samples.
9. The default exponential bout detector signal contains finite samples.
10. Swim-bout validation succeeds before atomic publication and selector update.
11. The old `v007` track remains unchanged and remains identifiable as a
    historical-contract publication.
12. A rerun with the same immutable output name fails closed.

The bout count itself should be reported, but a particular positive count must
not be hard-coded as a generic validator invariant: a scientifically valid
recording can contain zero detected bouts.

## Export acceptance criteria

After all three new bout publications complete, regenerate the intern exports
and inspect every camera independently:

- cam2010093: finite speed and an explicit bout event/interval table;
- cam2010094: finite speed and an explicit bout event/interval table;
- cam2010095: preserve the already valid source selection unless a deliberate
  cohort-wide rematerialization is chosen; and
- cam2010096: finite speed and an explicit bout event/interval table.

The export manifest should record the exact track and swim-bout run names,
detector contract/version, frame/time axis authority, calibration authority,
row counts, finite-sample counts, bout counts, and file hashes.

## Things not to do

- Do not merge all 12 physical commits blindly.
- Do not cherry-pick `4384e198` onto current `sun`.
- Do not relabel projection/derivation schema v1 as v2.
- Do not mutate `v007` or its source rowsets to make the current reader accept
  them.
- Do not bypass the strict reader with the legacy inspection loader to create
  new scientific publications.
- Do not infer physical scale from a familiar camera, filename, or prior run.
- Do not resubmit the failed 20260724 execution identifiers or output names.
- Do not update a `latest` pointer before full validation succeeds.

## Evidence commands used

Repository comparison:

```bash
git merge-base sun agent/physical-calibration-guard-20260720
git rev-list --left-right --count sun...agent/physical-calibration-guard-20260720
git cherry sun agent/physical-calibration-guard-20260720
git merge-tree 4e76963a sun agent/physical-calibration-guard-20260720
```

Production metadata was read from `zarr.json` files with `jq`; array data was
not opened during the three-archive metadata inventory. The separate cam2010093
compatibility proof used the normal strict logical track reader from the exact
physical branch and opened the archive read-only with consolidated metadata
disabled.

## Related documentation

- `docs/continuous_points_and_half_open_boxes.md`
- `docs/diagnostics/coordinate_contract_audit_2026-07-19.md`
- `docs/diagnostics/crimson_coordinate_implementation_work_package_2026-07-19.md`
- `docs/dask_zarr_write_safety.md`
- `docs/sandbox_zarr_fallback.md`
