# Source-video metadata migration

This migration makes `source_video_metadata` schema
`palette.source_video_metadata.v2` authoritative for ordinary single-video
recording Zarrs. The normative schema and conflict rules are defined in
`docs/source_video_metadata_contract.md`.

## Migration phases

### Phase 0: contract and new writers

Complete locally:

- new analysis/import writes emit the v2 object;
- videos inside a recording use a `recording_relative` locator;
- absolute root/raw-video paths remain compatibility mirrors;
- existing legacy objects are not partially upgraded when overwrite is false;
- no historical archive is rewritten implicitly.

### Phase 1: operational readers

Complete locally:

- `fisheye.shared.metadata.get_video_source_path`;
- `fisheye.shared.zarr_recording_context.infer_recording_context`, including
  direct `zarr.json`/`.zattrs` reads;
- `fisheye.shared.crop_image_source.CropImageSource` root-video fallback;
- dish-mask tuner metadata-only video fallback;
- flat ROI-cache eligibility and bundle planning;
- external ROI-cache materialization from a crop run.

These readers preserve legacy fallback but fail closed when populated video
path mirrors disagree. A missing locator remains an ordinary optional result
where the old API allowed no external video.

Crop-run-specific paths still take precedence for acquisition crop videos,
clipped inputs, and other explicitly bound pixel sources. The root
single-video resolver is consulted only when no more specific crop source is
present.

### Phase 2: remaining direct readers

Pending:

- training-crop regeneration and pixel-parity diagnostics;
- visualization/review surfaces that open a root video directly;
- legacy refinement utilities that copy root paths into new provenance;
- validation/audit tools, which should report canonical resolution and mirror
  conflicts separately.

Provenance-only fields are not operational locators and do not need wholesale
rewriting. Each remaining direct reader should be migrated with a parity test
that covers v2, consistent legacy metadata, conflicts, and any collection-aware
override.

### Phase 3: historical archive backfill

The read-only GoodCopBadCop preflight completed on 2026-07-17. It selected 40
active analysis rows representing 40 distinct recording IDs and Zarr paths. All
40 archives were Zarr v3, legacy-unversioned, and eligible for a
recording-relative v2 update. There were no conflicts, missing videos, warnings,
duplicate paths/recordings, or `/nvme1` operational fields.

Retained report:

`/groups/johnson/johnsonlab/jeremy/reports/source_video_metadata_migration/goodcopbadcop_v2_preflight_20260717T215730Z.json`

SHA-256:

`262187850029b81193a99fb1a72f51dc5d86cc8fe5803c9ccb7e660005c2660f`

Every row includes SHA-256 preconditions for the root and `raw_video` metadata
files plus source-video size and modification-time preconditions. The guarded
apply refused to write unless the report hash, registry cohort, metadata hashes,
and source-video stat preconditions still matched.

The guarded GoodCopBadCop apply completed on 2026-07-17:

- 40 canonical analysis archives targeted;
- 80 root/`raw_video` metadata files backed up and hash-verified before the
  first mutation;
- 40 root `zarr.json` files changed (`raw_video` compatibility mirrors were
  already correct and therefore remained byte-identical);
- 40/40 archives reopened and resolved through
  `source_video_metadata.locator` as `palette.source_video_metadata.v2`;
- no rollback was required.

Apply receipt:

`/groups/johnson/johnsonlab/jeremy/reports/source_video_metadata_migration/goodcopbadcop_v2_apply_receipt_20260717T220639Z.json`

Receipt SHA-256:

`e06da8df60a18700679ea0fc6c95d964b863eabfc695ffa3b1c26022e6ab703d`

Metadata backup:

`/groups/johnson/johnsonlab/jeremy/reports/source_video_metadata_migration/backups/goodcopbadcop_v2_apply_20260717T220639Z`

Backup-manifest SHA-256:

`7442c708b1bd09f9801518c37c31b38d7773c38aa09ffde7cc9f14f48118bc93`

An independent read-only post-apply census found 40 `already_v2` archives,
40 distinct Zarr paths, 40 distinct recording IDs, and zero cohort errors,
per-row errors, warnings, or duplicates:

`/groups/johnson/johnsonlab/jeremy/reports/source_video_metadata_migration/goodcopbadcop_v2_postapply_20260717T220704Z.json`

Post-apply report SHA-256:

`c130f0f27f9140bb3f669571273ab66f78a8afc765c76d1d494d9fbd05918324`

The registry was backed up before refresh at:

`/groups/johnson/johnsonlab/jeremy/registries/backups/palette_registry_before_source_video_v2_20260717T220639Z.sqlite`

Registry-backup SHA-256:

`4dde46823031b153bd180ec0fd8e6cf0804d4116319ffa93ff67d4e5ee77d8d9`

The standard registry scanner subsequently refreshed all 40 analysis rows.
The dataset IDs, recording IDs, Zarr paths, and active statuses remained
unchanged. SQLite `quick_check` returned `ok` and `foreign_key_check` returned
no rows. The live registry SHA-256 after refresh was
`e8e0a816603d831c07dfa2f34b3a725fe42e1ff236c416f9e1ea8fc305abb2d8`.

Clipped/multi-video collections are excluded from this single-video backfill.
They remain governed by the collection manifest and recording frame index.
