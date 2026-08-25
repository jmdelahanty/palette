# Recording Identity Census — 2026-08-25

**Status:** Read-only implementation evidence; not a repair plan and not a
merge-readiness declaration
**Branch:** `agent/palette/recording-identity-evidence-20260825`
**Base commit:** `162b95ba` (`docs: record source-of-truth consolidation plan`)
**Registry:** `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`
**Registry UUID:** `fbba3e9f-f444-434d-b821-928c615a5754`
**Registry schema:** `PRAGMA user_version=71`

## 1. Verdict

The census found concrete identity contradictions in a narrow, explicitly
registry-marked evidence cohort. It did **not** establish which artifacts were
created by the current Palette writer, and it is not evidence of a store-wide
identity collapse.

- The 31 selected source-analysis artifacts contain 23
  `analysis_zarr` recordings and eight `rolling_clips` recordings using
  `palette.recording_frame_index.v1`.
- The count 31 is a count of per-camera artifact/registry entities, not 31
  independent acquisition sessions and not 31 million-row frame indexes. Only
  eight artifacts have frame-index Parquets, representing two long acquisition
  families with four camera-specific indexes each.
- All 30 stored registry identity-projection tables scanned successfully. The
  selected identity scope covered 9,001 projection rows and produced no projection
  mismatch finding.
- Eight selected clipped artifacts contain actionable cross-surface
  identity defects. They fall into two four-camera families, not 24 independent
  root causes.
- Twenty-eight Zarr scopes were completely readable. Three selected
  Sleepyfish roots contain non-finite `Infinity` in JSON metadata, so the
  declared artifact scope is honestly incomplete.
- All 31 declared recording-metadata scopes completed. The scanner covered the
  declared sidecars, declared clip manifests, and frame-index identity columns;
  it did not recursively inventory every directory entry.
- The eight frame-index Parquets are healthy identity evidence even though four
  contain 2,937,604 rows each and four contain 1,188,000 rows each. Row count is
  not a defect.
- Unmarked artifacts and compatibility reconciliation are deferred. An
  unmarked artifact is not thereby classified as legacy.

The Step 2 writer consolidation remains justified: new code must refuse these
contradictions before it creates or updates a Zarr/registry projection. This
census does not choose a winning identity and does not authorize repair.

## 2. Scope boundary

The default filesystem scope is `explicit_source_layout`:

```text
datasets.status = active
datasets.artifact_kind = source_recording
datasets.zarr_use = analysis
and either source_layout or source_frame_index_schema is explicitly populated
```

This is an evidence-selection boundary, not a calendar cutoff or a writer
generation. The markers say how the registry describes a source layout; they
do not bind the producing Palette commit. The registry schema and
identity-bearing views are still inventoried globally, while row-level
projection findings are restricted to the selected dataset, recording, and
session identities.

This distinction matters for the two recording families raised during review:

| Family | Active source-analysis rows | Registry layout markers | Root schema marker | Selected by default? |
|---|---:|---|---|---|
| Batman | 36 | null | `recording_analysis_v1` | no |
| Goodbatbadbat | 84 | null | `recording_analysis_v1` | no |

All 120 are excluded by the registry-layout predicate even though their roots
carry the same broad analysis schema marker. That does not prove that they came
from an older implementation. `recording_analysis_v1` identifies a data
contract, not a unique code revision, and these roots do not bind the Palette
commit that created or registered them.

The broader `active_source_analysis` mode remains available for a later corpus
or compatibility campaign. A preliminary all-296 pass was useful for finding
scanner noise and heterogeneous conventions, but it is superseded as
acceptance evidence. In particular, arbitrary deep analytics provenance is not
recording identity, run/profile copies are not merged into the root comparison
domain, and repeated `source_zarr` self-references are not treated as external
donors.

The following remain deferred:

- unmarked artifact reconciliation or repair;
- HDF5 identity attributes;
- Zarr array and frame-map payload equality;
- video decoding;
- consolidated-versus-direct metadata comparison, which belongs to the
  lifecycle-aware opener work; and
- unregistered filesystem discovery outside exact registry locators.

One compatibility guard remains in scope: a current reader must use a declared
compatibility boundary or reject an unsupported artifact clearly. It must not
silently reinterpret a legacy field as a current identity.

### 2.1 Producer-provenance boundary

Modern downstream components commonly record real producer evidence. Shared
`palette.run_provenance.v1` records the Git SHA and dirty state, configuration
hash, command, parameters, input run IDs/artifacts, and package/runtime context;
stage provenance records the producing stage and its inputs. Many detection,
crop, keypoint, mask, epoch, and motion runs use those contracts
(`src/fisheye/shared/run_provenance.py:114-269` and
`src/fisheye/shared/stage_provenance.py:136-201`).

That evidence answers which code produced a downstream run. It does not prove
which Palette revision originally created the recording root. The acquisition
authority records a versioned producer label, and Orange recording sidecars can
record the acquisition producer/version, but the source root and registry do
not consistently bind an exact Palette importer commit. Downstream provenance
must therefore remain component-local evidence, not be promoted into inferred
root provenance.

Current-writer acceptance must come from synthetic writer-to-unpatched-reader
round trips and a commit-pinned, selector-ineligible canary after consolidation.
The existing recording corpus can provide contradiction examples, but cannot
serve as a complete writer-generation oracle.

A related read-only check found a bounded enforcement loophole in recent
Goodbat diagnostic outputs: 356 runs have `run_provenance.git_sha=null`; 355 are
marked complete and one failed. All 356 are selector-ineligible, no parent
selector points to them, and none is an authoritative science output. The
three local materializers pass the source Zarr path as the Git working
directory, then use parentless completion, which skips the parent-scoped
provenance gate. This does not change the identity finding counts, but it is a
current engineering defect to close: “complete” must not imply valid producer
provenance unless the same validation gate actually ran. The bypass is visible
at `src/fisheye/shared/zarr_run_completion.py:197-209`; representative callers
are `src/fisheye/analysis_workflows/materializers/subject_position.py:399-410,811-816`,
`src/fisheye/analysis_workflows/materializers/provider_epoch_behavior_summary.py:955-975`,
and `src/fisheye/analysis_workflows/materializers/provider_track_motion.py:1012-1032,1497-1502`.

## 3. Reproducible evidence

Command:

```bash
scripts/py -m fisheye.registry.recording_identity_census \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --output /tmp/palette-recording-identity-census-explicit-layout-20260825-v5.json \
  --progress
```

The command returns status 1 because findings exist. That is a completed
diagnostic result, not a process failure.

Evidence bindings:

| Item | Value |
|---|---|
| Generated UTC | `2026-08-25T06:40:15.620555Z` |
| Registry snapshot SHA-256 | `f37c8f2155904becd2a61a613b1e6036fa5d58b02c505cc536f5136d2c598d1b` |
| Registry snapshot size | 58,486,784 bytes |
| Registry integrity | `ok` |
| Registry foreign-key findings | 0 |
| Deterministic census-body digest | `5f35c874f3cab45cfa0f7481e5b6a6f9bafcb9ddba3a4b8adf5d791ba5444420` |
| Serialized report SHA-256 | `99c428b9b623bde2ad79452ae7584bc5bf1619578bfb3eed03acf50296d41d9e` |
| Report size/location | 3,807,343 bytes in `/tmp`; intentionally not tracked |

The report's operational timestamp is outside the canonical census body. The
digest covers the deterministic body. The scanner uses a read-only SQLite URI,
a local SQLite backup snapshot, `PRAGMA query_only`, direct bounded metadata
reads, pre/post file fences, and exclusive report creation. It never opens Zarr
arrays or decodes video.

## 4. Selected-cohort findings by root cause

### 4.1 Aug-6 rolling-clips session identity — four artifacts

Affected family:

```text
2026_08_06_19_13_35_cam2010093
2026_08_06_19_13_35_cam2010094
2026_08_06_19_13_35_cam2010095
2026_08_06_19_13_35_cam2010096
```

For each camera artifact:

- `datasets.session_uuid` and `recordings.session_uuid` agree on the shared
  acquisition identity `2026_08_06_19_13_35`;
- `recording_manifest.json.session_uuid` contains the camera-specific
  recording ID instead; and
- the Zarr root has no exact `session_uuid` field.

This produces four artifact conflicts, four recording-sidecar conflicts, and
four missing-root findings. Those 12 findings describe four writer-boundary
failures. They must not be repaired by precedence. The current clipped-shell
writer should require agreement before publication.

### 4.2 Sleepyfish recording identity — four artifacts

Affected family:

```text
sleepyfish_2026_05_05_17_45_30_cam2010093
sleepyfish_2026_05_05_17_45_30_cam2010094
sleepyfish_2026_05_05_17_45_30_cam2010095
sleepyfish_2026_05_05_17_45_30_cam2010096
```

For each camera artifact, the recording manifest carries session-level
`recording_id=2026_05_05_17_45_30`, while the registry, frame-index evidence,
clip index/manifests, and readable Zarr evidence use the camera-specific
`sleepyfish_..._cam...` identity. This produces four artifact conflicts and
four sidecar conflicts.

Three roots (`cam2010093`, `cam2010094`, and `cam2010096`) also contain literal
JSON `Infinity`; they are therefore incomplete evidence, not silently accepted
metadata. The readable `cam2010095` root is missing `session_uuid`.

The registry marks these artifacts as `rolling_clips`, but their exact Palette
writer commit is not bound. The implementation lesson still applies to the
current clipped-shell contract: sidecar agreement and valid canonical metadata
must be checked before output creation. Repairing the existing artifacts
remains deferred.

### 4.3 Expected compatibility observation — 23 rows

Twenty-three selected `analysis_zarr` datasets have `dataset_id ==
session_uuid`. Equality alone does not prove that the fields are aliases and is
not a conformance failure. The census records it as expected compatibility
evidence while continuing to model `dataset_id` and `session_uuid` as different
semantic facts.

## 5. Large-file interpretation

### Frame-index Parquet

The scanner reads only `recording_id`, `session_uuid`, legacy `session_id`, and
`camera_serial`. It uses row-group statistics when an identity column is
constant and bounded batches otherwise. Each of the eight selected frame
indexes had:

- one distinct `recording_id`;
- one distinct legacy `session_id`;
- one distinct `camera_serial`;
- no malformed identity value; and
- no identity-cardinality overflow.

The eight files group as follows:

| Acquisition family | Camera-specific indexes | Rows per index | Byte-identical duplicates |
|---|---:|---:|---:|
| `2026_08_06_19_13_35` | 4 | 2,937,604 | 0 |
| `sleepyfish_2026_05_05_17_45_30` | 4 | 1,188,000 | 0 |

A separate read-only duplicate check—not the bounded census report—computed all
eight full-file SHA-256 digests and file sizes; all differed. Their equal row
counts within a family reflect synchronized cameras covering the same duration;
their recording IDs, camera serials, and source paths are camera-specific. The
layout intentionally repeats the frame axis per camera. A session-level shared
clock/index could be studied as a later storage optimization, but the combined
checks found no accidental duplicate file or duplicate registry/Zarr path.

Millions of rows are expected for long recordings reconstructed from clips.
The implementation never emits a finding based on row count. It separately
caps pathological identity cardinality so a malformed file cannot create an
unbounded report.

### Accidental inline root metadata

The superseded broad pass found the confirmed canary anomaly:

```text
2026-08-10T17-20-55Z_arena_2_goodbatbadbat_analysis.zarr/zarr.json
size: 1,458,574,027 bytes
```

This is metadata amplification, not legitimate frame-index scale. The scanner
read only a stable 8 MiB prefix containing the root identity attrs and did not
expand inline consolidated metadata. The current test suite locks in that
bounded behavior. The artifact is outside the explicit-layout cohort, and the
user confirmed this production shape was accidental and should not recur.

## 6. Implementation properties

`src/fisheye/registry/recording_identity_census.py` is diagnostic evidence, not
a new runtime authority. It guarantees:

1. exact `recording_id` binding from `datasets` to `recordings`; no
   `session_uuid` fallback for row association;
2. separate semantic treatment of recording, acquisition/session, legacy
   session, dataset, camera, clip, run, and profile facts;
3. dynamic inventory of every SQLite table/view with identity-named columns;
4. row-level comparison of every stored projection in the selected explicit
   layout scope;
5. bounded direct Zarr metadata traversal without array reads;
6. bounded, projection-digested frame-index Parquet reads independent of row
   count;
7. explicit missing, malformed, conflict, capped, unstable, and incomplete
   outcomes;
8. no emitted effective identity, precedence resolution, correction, or
   mutation plan; and
9. exclusive output creation outside every observed input root.

The implementation is deliberately additive in Step 1: approximately 3,080
lines of diagnostic source and 760 lines of focused tests at this commit. It is
not imported by ordinary production readers or writers. The campaign has not
yet achieved code reduction; Step 2 must use this evidence to consolidate and
then delete the duplicate production writers and precedence ladders. This
accounting should remain explicit rather than claiming an early net reduction.

## 7. Tests and readiness

Focused validation:

```text
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_recording_identity_census.py -q

20 passed
```

Static validation:

```text
scripts/py -m py_compile \
  src/fisheye/registry/recording_identity_census.py \
  tests/unit/fisheye/test_recording_identity_census.py

git diff --check
```

Both completed successfully. Required repository CI has not run, so the branch
must not be described as merge-ready.

## 8. Step 2 handoff

The next implementation should use this census as a failing-before-change
baseline and satisfy these gates:

1. one evidence resolver preserves `recording_id` and `session_uuid` as
   separate facts and rejects conflicting non-null evidence;
2. one registry projection writer owns both normal registration and
   maintenance updates;
3. routine imports cannot overwrite a known identity or erase one with null;
4. explicit correction uses revisioned, audited compare-and-swap semantics;
5. source-recording `datasets` and `recordings` identities remain in parity;
6. clipped-shell sidecars agree before root attrs are written;
7. donor identity, camera, and frame-map bindings are proven before copying;
8. every new source publication carries an immutable import receipt binding
   the exact Palette commit, dirty state, configuration, source evidence, and
   identity decision;
9. writer-to-unpatched-reader round trips cover both regular and clipped
   layouts;
10. a commit-pinned selector-ineligible canary proves the current writer and
    reader boundary before production activation;
11. parentless completion cannot mark a run complete with invalid producer
    provenance; and
12. the two four-camera failure families above fail before output creation; and
13. superseded registry SQL, source-precedence branches, and duplicate profile
    extraction are deleted after all callers migrate.

Existing-artifact repair is explicitly not part of those gates. Supported
compatibility readers remain behind a declared boundary until a separate
campaign is authorized.
