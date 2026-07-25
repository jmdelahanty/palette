# Import profile and Zarr bootstrap drift TODO
<!-- contract-meta
status: working_checklist
last_verified: 2026-07-24
-->

Purpose: track the remaining mismatch between Palette's active import profiles
and the broad root shape described by the authoritative Zarr layout. This file
was refreshed after the import-profile validator and safe detection-publication
boundary landed; completed historical concerns are recorded below instead of
being left as open work.

## Current implementation

Palette now has explicit runtime import profiles in
`fisheye.shared.import_profile_contract`:

- `metadata_only_analysis`: acquisition metadata and source-video authority,
  without imported frame arrays;
- `sampled_training_pynvvc_luma`: materialized sampled training frames with an
  exact pixel/decode contract and original-frame mapping;
- `legacy_decord_training_or_full`: read-only compatibility classification for
  historical materialized imports.

`fisheye.utils.check_import_profile` classifies archives without mutation and
reports required and recommended gaps. Focused unit tests cover the active and
legacy profiles, incomplete archives, JSONL output, and bounded recordings-root
discovery.

Canonical analysis import uses `import_recording_analysis`, stamps analysis
purpose, and publishes acquisition metadata before detection. Canonical
detection no longer asks the numerical writer to create or overwrite that
metadata: it resolves a registered model, constructs a disposable node-local
candidate, and publishes the completed run atomically. The low-level detector's
standalone `production` archive behavior remains a non-canonical development and
compatibility surface.

## Remaining drift

1. `src/fisheye/docs/zarr_structure.md` still presents one broad set of root
   attrs and immediate child groups as though every valid archive materializes
   all of them at bootstrap. It needs a profile-aware required/optional table.
2. `ensure_analysis_archive` deliberately creates a minimal analysis authority,
   while legacy `create_palette_zarr` creates a much broader skeleton. The
   repository has not yet either consolidated those behind one profile-aware
   bootstrap helper or formally retired the broad helper.
3. The runtime classifier is the machine-readable authority for import-profile
   validation, but the human-readable Zarr spec does not yet point to it or
   describe its reason codes and compatibility boundary.
4. External-consumer guidance should explicitly distinguish missing optional
   groups in a valid metadata-only archive from a corrupt/incomplete archive.

## Completed work

- [x] Define active metadata-only analysis and sampled-training profiles in
  runtime code.
- [x] Keep historical Decord materialization in an explicit compatibility
  profile rather than inferring that it is current.
- [x] Add a read-only profile diagnostic with compact JSONL and bounded PRFS
  discovery.
- [x] Add deterministic in-memory tests for profile classification and CLI
  behavior.
- [x] Retire the duplicate `create_analysis_zarr` entry point in favor of
  `import_recording_analysis` (2026-07-24).
- [x] Prevent canonical detection from mutating acquisition metadata or changing
  an analysis archive's purpose.
- [x] Route canonical full-recording detection through node-local candidate
  construction and atomic publication.
- [x] Update the recording pipeline and operator detection guidance to use the
  registry-resolved publication path.

## Remaining plan

- [ ] Add an import-profile section to
  `src/fisheye/docs/zarr_structure.md`, including required root attrs, required
  groups, allowed omissions, and frame-universe semantics for each active
  profile.
- [ ] Decide whether to replace `create_palette_zarr` with a shared
  profile-aware bootstrap or mark it as a legacy training-only helper. Do not
  make metadata-only analysis archives eagerly create empty run families merely
  to resemble the old broad skeleton.
- [ ] Link the authoritative Zarr spec to
  `fisheye.shared.import_profile_contract` and document the classifier's
  required-versus-recommended distinction.
- [ ] Add a short external-consumer note describing valid optional omissions in
  metadata-only analysis archives.

## Exit criteria

- [ ] The human-readable Zarr layout and runtime classifier describe the same
  active profiles.
- [ ] Analysis bootstrap and sampled-training bootstrap each have one declared
  implementation authority.
- [ ] A consumer can distinguish a valid sparse profile from an incomplete
  archive without relying on filename suffixes or empty placeholder groups.
