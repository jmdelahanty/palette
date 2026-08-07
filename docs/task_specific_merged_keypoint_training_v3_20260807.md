# Task-specific merged keypoint training v3 — 2026-08-07

Status: **IMPLEMENTED; CORRECTED V2 SOURCE RECEIPT PREFLIGHT PASSED; FINAL PUBLICATION PENDING**

## Goal

Publish the complete reviewed five-keypoint corpus without ambiguous frame
lineage or biological/acquisition leakage. This is a successor to v2. Existing
v1/v2 artifacts remain immutable evidence and are not rewritten.

## Required source contract

Every source manifest entry declares:

- `recording_id`;
- `leakage_group.id`;
- `leakage_group.source`, exactly one of `registered_subject`,
  `acquisition_start_fallback`, or `recording_fallback`;
- an exact reviewed keypoint run, crop run, skeleton, label order, ROI pixel
  contract, and row gate.

The group resolver prefers registered subject identity, then acquisition start
time for sibling-camera collections, then recording identity. A group is an
indivisible split unit. Immutable v3 publication rejects every other split
mode.

## Persisted row lineage

`source_index/source_frame_idx` is forbidden in v3. It is replaced by:

- `source_sample_row_index (N,) int64`: frame index in the source training
  archive's local frame domain;
- `source_acquisition_frame_index (N,) int64`: frame index in the original
  acquisition-camera domain.

`source_frame_mapping_json (S,) utf8` declares the exact source arrays and one
of three mapping methods: direct acquisition index, lookup through
`raw_video/original_frame_indices`, or identity fallback. V3 also persists
`leakage_group_id (S,) utf8` and `leakage_group_source (S,) utf8`.

The existing source dataset, ROI row, refined row, and raw-detection row arrays
remain required. The validator recomputes split group membership and rejects
any group present in more than one split.

## Pose-only lineage boundary

Historical rows without stable refined-detection or raw-detection identity are
valid only for pose training and pose evaluation. Their fallback identity is
`recording_id + source_acquisition_frame_index + source_roi_idx`. The
publication records their exact count and explicitly prohibits:

- joint detection-pose training;
- detection-edit propagation;
- claims of complete detection lineage.

## Full five-keypoint candidate

The frozen census identifies 61 compatible sources, 12,704 usable poses, and
63,520 landmark locations. Sixty sources are 512×512; the reviewed Batman
source is centered-zero-padded from 348×348 without resizing. Float64 sources
use the checked float32 conversion already validated by v2.

The publication remains selector-ineligible and absent from the registry until
training/evaluation review explicitly activates it.

Immutable publication copies the complete 61-source input manifest to a
versioned sibling of the Zarr and binds that durable path and SHA-256 inside
the sealed root metadata. A node-local or `/tmp` manifest path is not the final
provenance reference.

## Checklist

- [x] Add exact dual frame-domain arrays.
- [x] Remove the ambiguous frame alias from v3.
- [x] Add subject/acquisition/recording leakage-group resolution.
- [x] Make leakage groups the required immutable split unit.
- [x] Persist group and frame-mapping provenance.
- [x] Add fail-closed overlap and metadata validation.
- [x] Add the pose-only incomplete-detection-lineage policy.
- [x] Add adversarial split-tampering tests.
- [x] Compose the exact frozen census into one 61-source reviewed manifest.
- [x] Bind the reviewed Batman artifact through that census composition.
- [x] Preflight exactly 61 sources and 12,704 usable rows.
- [ ] Build on bounded node-local scratch and publish atomically.
- [ ] Validate direct/consolidated metadata, storage plans, row identities,
  split isolation, and exact source composition.
- [ ] Train and evaluate a candidate before registry activation.

## Exact preflight receipt

The source-composition digest remains:

`f07e3127f5b2c2e3fb55881962eccbf1df0ac7c0dacf41457d8d969460302323`

The composed manifest has composition digest:

`25ca6294d51e3e5a1f04c519cc34610b44ec728e6db98a1bc4cfcd8b40bf15da`

The fail-closed dry run resolved all 61 exact source paths and reviewed runs,
selected 12,704 usable rows, grouped them into 29 leakage groups, and produced
10,096 training plus 2,608 validation rows. It performed no writes. During
preflight, stale ancestor inline metadata was proven capable of hiding both a
current crop pixel contract and the census-selected five-keypoint refined run.
Source discovery now opens each mutable run group at its exact direct path and
never substitutes another refined run for an explicit census binding.

The selector-ineligible `v001` artifact proved the write and validation path,
but its source census conservatively missed 205 lineage IDs hidden by a stale
inline snapshot and its bound source-manifest path was node-local. It remains
immutable benchmark evidence and is superseded by `v002`; it is not eligible
for registration or model training claims. The corrected census reports 3,047
pose-only rows without detection identity, matching the actual merged arrays.
