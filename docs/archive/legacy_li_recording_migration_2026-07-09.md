# Legacy Li Recording Migration — 2026-07-09

## Purpose

Migrate the original Blindfish recordings from `/nvme1/Li` into the shared
Palette staging, recordings, and registry workflow without treating locally
generated resized videos as acquisition authorities.

## Source and destination

- Source: `/nvme1/Li`
- Staging: `/groups/johnson/johnsonlab/jeremy/staging`
- Recordings: `/groups/johnson/johnsonlab/jeremy/recordings`
- Registry: `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`

The migration copies source data into staging and preserves the `/nvme1/Li`
source. No automatic Citrus completion marker is created. Organization and
registry writes are run explicitly after staged verification.

## Included acquisition sessions

- `2026_03_27_18_37_03`
- `2026_03_27_20_56_46`
- `2026_03_27_23_16_52`
- `2026_04_20_16_37_39`
- `2026_04_28_21_34_57`
- `2026_04_28_23_47_03`

The selected transfer contains 150 regular files totaling
`835,864,549,979` bytes. It includes:

- original `Cam<id>.mp4` camera videos;
- `Cam<id>_meta.csv` frame metadata;
- `Cam<id>_keyframe.json` encoded-frame/keyframe indexes;
- Citrus H5, stimulus replay MP4, and update-timing CSV artifacts;
- `recording_snapshot.json`;
- `ptp_sync_summary.json`.

## Explicit exclusions

- `/nvme1/Li/test`
- every `*_cropped_resized.mp4`
- every `*_resized.mp4`

The resized files were user-created convenience derivatives. They are not
staged, organized, registered, or treated as provenance-bearing inputs.

## Recording topology and metadata

The H5-to-camera mapping is consistent across the source:

- `arena_1` -> `Cam2010093`
- `arena_2` -> `Cam2010094`
- `arena_3` -> `Cam2010095`
- `arena_4` -> `Cam2010096`

Each camera image contains one Cedar dish with four separate wells and one fish
per well. The migration therefore applies operator-known metadata
`num_dishes=1` and `fish_per_dish=4` to organized recording manifests.

There are 22 H5-backed recordings. Session `2026_03_27_23_16_52` contains H5
records only for arenas 1 and 2; its Cam2010095 and Cam2010096 streams are
retained and imported separately as reviewed `video_only_v1` recordings rather
than being assigned stimulus context that does not exist.

## Legacy-layout compatibility

These sessions place camera keyframe JSON files and shared runtime context one
directory above `citrus/`. The H5 organizer must therefore:

- discover `Cam<id>_keyframe.json` beside the original camera video;
- copy the full snapshot into every recording as
  `raw/recording_snapshot_runtime.json`;
- copy `ptp_sync_summary.json` into every recording;
- retain the per-camera filtered snapshot under `derived/`;
- accept explicit `--num-dishes` and `--fish-per-dish` metadata overrides.

Shared session files are copied, not moved, because one acquisition session can
produce several organized recordings.

## Validation gates

1. Focused organizer unit tests and static checks pass.
2. Staged inventory matches the 150-file selection by size and relative path.
3. Organizer dry-runs have no missing required H5/camera artifacts.
4. Video and H5 diagnostics persist a non-failing manifest preflight.
5. Analysis Zarr import succeeds before registry synchronization.
6. The final audit finds 24 organized recordings: 22 H5-backed and 2
   camera-only, all with original source video metadata and four-fish setup
   context.

## Final status

Migration completed successfully.

- 24 recording directories and 24 registry recording rows are present.
- 24 analysis datasets are registered.
- 19 recordings retain H5/stimulus context.
- 5 recordings are explicitly `video_only_v1`: the three H5 recordings whose
  frame metadata had no positive camera IDs, plus two camera-only streams from
  the partial March session.
- All 24 manifests contain `num_dishes=1` and `fish_per_dish=4`.
- All 24 manifests contain the camera keyframe sidecar, PTP summary, and
  runtime snapshot.
- All camera/video diagnostics passed.
- The 19 H5-backed manifests retain their non-fatal alignment warnings. The
  three failed H5 contexts remain preserved under reversible
  `__h5_context_quarantine` directories, including the interrupted partial H5
  Zarr marked `h5_failed_import`.
- The original source still contains exactly the 150-file canonical payload
  (`835,864,549,979` bytes); no source files were removed.
- Focused organizer regression tests pass: 11 passed.
