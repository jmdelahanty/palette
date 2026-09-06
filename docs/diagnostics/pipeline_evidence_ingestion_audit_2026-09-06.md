# Pipeline evidence audit: ingestion and layout-neutral recording interfaces

Date: 2026-09-06. Status: living audit; first ingestion/import review and
repository-wide evidence-family index. This is evidence and design clarification,
not a new runtime contract, implementation queue, or declaration of pipeline
completion.

## September 6 PR-preparation reconciliation

The findings below describe the pinned audited source, not the state of every
later branch. Historical measurements, diagnostic outputs, source pins, and
numerical tables are unchanged. The user authorized Palette-only preparation,
commits, pushes, and CI; main merges and production changes remain separate.

- This audit candidate is refreshed from its clean prior head
  `0cbca7f5be5e8f2c1a21660147b831606295bf4e` onto current main
  `3d017867e79b14d11ddca3ee1916d50ac6499c78` by a conflict-free,
  history-preserving merge. Each incoming exact commit had all 23 required
  checks completed successfully. The refreshed candidate needs its own CI;
  the PR records its exact commit and results. No runtime fix branch is
  incorporated into this documentation-only change.
- [CI gate #140](https://github.com/jmdelahanty/palette/pull/140), exact head
  `74926b21cb2e587b1caae151aaa9c4df91e592b8`, passed all 24 checks in
  [run 34028051914](https://github.com/jmdelahanty/palette/actions/runs/34028051914).
  It remains unmerged; `ci-required` is not yet an active main requirement.
- [Ingestion validation #143](https://github.com/jmdelahanty/palette/pull/143)
  corrects the reproduced PTP-state, integer, and frame-map defects. Its old
  head `3a85a8c9215894945685eaba6f0055730392db44` passed 23 checks.
  Refreshed head `eb224a216ac7774770b33de021c599b741ce7927` includes the
  validated CI gate and has fresh CI pending. Those fixes are not merged into
  main, and do not close the independent clock-publication finding.
- Clock-publication safety is separately implemented in the owned
  `/tmp/palette-clock-publication-safety-20260906` worktree, based on the
  original #143 head: 277 focused tests pass, including failure/retry and the
  public source-digest interface. It is still uncommitted and lacks full CI.
- The separate ingestion-enforcement worktree at
  `/tmp/palette-ingestion-enforcement-20260906` now rejects recorded optional
  failures, source legacy intake, unverified replay, and false invocation
  acknowledgments. Required-context defaults and manual sampled-training
  identity/failure gates have been corrected through their own contracts.
  The 370-test regression and 40-test follow-up pass locally; full CI and
  integration with clock safety remain unrun. This is not full dispatcher
  recovery, parent-level clipped intake, or an all-entrypoint closure claim.
- The separately reviewable documentation/rules draft preserves the older
  architecture audits and narrows their scientific and authority claims.
  Transfer-v2 consumer adoption, hardware synchronization/equivalence,
  full crop-ledger payload integrity, historical mutation, and activation
  remain separately scoped work. No deployed code or live data was changed.

Implementation status continues to belong to the existing authority
consolidation queue. Later PR heads/checks supersede this dated preparation
snapshot; none of these local results is scientific acceptance.

## 1. Outcome and scope

The desired interface is already the right architectural direction: one scientific
recording interface, with whole-video and clipped adapters underneath it. Clipped
workloads own partitioning, exact frame mapping, worker evidence, and validated
recording-level assembly. Scientific consumers should not need a different API
because the pixels happen to live in 22 or 55 files.

The repository has substantial pieces of this design, but ingestion through
consumer admission is not yet one consistently enforced workflow. In particular:

- Current single-recording import has a strict identity claim, a clean-code import
  receipt, and receipt-bound registry finalization. These are stronger than some
  older audit descriptions.
- The two live clipped sets are structurally different. The May sleepyfish set is
  a whole-video source with materialized stream-copy clips; the August date-named
  set is a native rolling-clip collection. Both must be supported without treating
  their historical layouts or selectors as interchangeable.
- Two in-memory probes reproduced enforcement defects: clipped frame-map admission
  accepts duplicate/missing-frame combinations, and failed new-clock loadback can
  leave `latest` pointing at a run marked complete.
- Clock investigation also reproduced contradictory PTP-state admission and lossy
  Parquet integer coercion. Clock availability is not measured-time or
  synchronization sufficiency; the current provider timing contract deliberately
  uses nominal FPS and supports only single-video sources.
- Acquisition crop-ledger verification is narrower than a payload-integrity proof.
  It checks pointer/digest agreement and array lengths, not decoded array contents.
- Fresh rolling-clip intake, current import identity/registry binding, and final
  consolidated publication do not yet share the whole-import boundary.
- Installed cron automation uses older deployed code and stops after import. Its
  submission acknowledgment also has a separately reproduced false-success risk.

No recording was repaired, re-imported, rehashed in full, migrated, or activated.
No scientific defaults or receipt grammars were changed. The audit does not infer
that any live recording is corrupt from the code defects above.

### Review coverage

| Area | Evidence in this revision | Not established |
|---|---|---|
| Transfer, dispatch, organization | Installed-script inspection, deployed commit, current code paths, prior safe Bash reproduction | End-to-end fault/retry campaign or transfer-content verification |
| Whole import, identity, registry | Producer/receipt/consumer trace, focused tests, one historical live metadata reference | Fresh production import under current code; production SQLite acceptance |
| Clipped source and sampled import | Both live sets, source-map and publication code, focused tests, adversarial in-memory map probe | Encoded-video or decoded-pixel parity; live per-frame mapping revalidation |
| Clock and acquisition crop ledger | Digest scope, publication order, source/classifier probes, immediate timing consumers, focused tests | Full live array verification; recording-specific clock/synchronization acceptance; complete ledger payload integrity |
| Downstream stages | Current family/source index and selected convergence seams | Full producer → publication → resolver → unpatched consumer audit for every family |

The initial audit findings are usable now. The all-stage audit remains open; the
index in §9 is not a claim that every listed stage has passed a deep review.

## 2. Snapshot, ownership, and evidence limits

| Surface | Exact snapshot / ownership |
|---|---|
| Audited source and test base | `0a9531f9914f30a487704388a12a088a8d3365fe` |
| Reconciled producer handoff | `jmdelahanty/agent-contracts` commit `4d80fa92ff926baaa4e3c7408d39ed054a5df732`, supplied branch `agent/acquisition/producer-timing-audit-20260906`; see §5.7 for evidence limits |
| Audit worktree | `/tmp/palette-ingestion-provenance-audit-20260906` |
| Audit branch | `agent/palette/ingestion-provenance-audit-20260906` |
| Audit owner / publication scope | Codex; this document, metadata evidence JSON and clock diagnostic Markdown only. The documentation revision is its containing Git commit, distinct from the audited source base. User-requested parallel clock reviewer was read-only at the same base; root owns these documents. |
| Installed poller | `/home/delahantyj@hhmi.org/bin/citrus_staging_marker_poller.sh`; SHA-256 `6bbdcfc4e81278ddbf1b01efc3ad2af768d778c6c48079cf688a4784cabec172` |
| Poller-selected shared deployment | `/groups/johnson/johnsonlab/jeremy/gitrepos/palette`; clean commit `c1882f9b3565016a777acc5ab33e1a57b858eb06` at inspection, 299 commits behind the audited base |
| Live recording observations | 2026-09-06 UTC, read-only SSH to Citrus; direct JSON files plus inline consolidated metadata; no synchronous Zarr payload traversal |
| Prior audit comparison | `receipt_builder_census_2026-09-02.md` from the separate workstation checkout; its stated census base is `ea03eb76`, not this audit's base |

The implementation-status owner remains the
[authority consolidation queue](authority_consolidation_work_queue_2026-08-25.md).
Its dated status rows must be reconciled with exact current commits before work is
assigned; this document does not silently close them. Keep scientific and
performance queues separate. No other worktree or review draft was changed.

The existing [layout-neutral DAG design](../production_dag_recording_layout_design.md)
and [whole/clipped convergence checklist](../clipped_whole_detection_convergence_implementation_checklist.md)
already own this architectural direction. Their July implementation assessments
are historical evidence, not proof that every listed defect remains in current
code. This audit extends their source/import evidence rather than starting another
DAG framework or competing checklist.

The [metadata snapshot](evidence/ingestion_layout_metadata_2026-09-06.json) records
observed declarations. Root selector fields and consolidated node counts are
inventory, not verification of the selected artifacts. Reads were not an atomic
filesystem snapshot. No live registry was opened writable or accepted using a
system `sqlite3` binary.

## 3. One interface, with separate source and execution layouts

There are three relevant cases, not just a `clipped` boolean:

```text
whole source video ───────────────────────────────┐
whole source video + derived stream-copy clips ──┼─ source/layout adapter
native rolling-clip source collection ───────────┘          │
                                                   admitted recording
                                                   + exact work units
                                                          │
                                               shared scientific stages
                                                          │
                                      validate/bind/assemble as needed
                                                          │
                                        recording-level family publication
                                                          │
                                          the same downstream consumers
```

This is an interface target, not a newly persisted envelope schema. Existing
`RecordingTarget`, `VideoWorkUnit`, and `VideoFrameMapping` in
[recording_layout.py](../../src/fisheye/cluster/recording_layout.py) are the
planning owner to extend. Their constructors validate declarations; merely
constructing one is not dynamic receipt admission. `VideoFrameMapping` currently
holds a path/mode, not a content-bound frame-map proof.

The existing acquisition ownership/frame objects, stage-specific manifests, and
family resolvers should supply the scientific evidence. Do not introduce a second
generic authority that reinterprets them or weakens their validators.

| Common claim | Whole source | Clipped source / execution | Must remain invariant |
|---|---|---|---|
| Recording identity | Explicit session/camera/recording identity | Same parent identity; clips are partitions, not new recordings | No path/name-derived biological identity; no duplicate recording IDs per clip |
| Source membership | One video | Ordered, complete source members; derived clips separately bind their whole source | Exact source generation and declared derivation; no silent source substitution |
| Frame domain | Identity local → acquisition map under the declared profile | Verified clip-local → parent/acquisition map | Exact coverage and correspondence in the declared output domain; acquisition gaps, subsets and intentional blanks remain explicit, never silently rebased |
| Time | Explicit clock, units, origin, timescale, validity | Same clock across boundaries; missing/uncertain time stays explicit | Do not invent timestamps from frame count or infer UTC from PTP enablement |
| Pixel/coordinate domain | Bound native extent and pixel contract | Same per camera across members, with source-member lineage | Native scale, orientation, pixel-center/edge rules, validity, decoder and model preprocessing |
| Observation identity | Recording-scoped keys and row lineage | Same keys after assembly; preserve source clip/run/row identity | Exact key correspondence, not equal row counts or a freshly rebased local row number |
| Scientific parameters | Applicable named recipe, pinned model and effective parameters | Identical recipe and kernels | No hidden threshold, resize, smoothing, quality, or acceptance changes |
| Execution/storage | One or more planned compute partitions | Clip tasks, bounded bundles, caches, then finalizers | Resources/layout may vary; physical chunks have one writer |
| Publication/consumption | Family manifest and applicable selector | Same final family contract; extra partition receipts underneath | Same consumer validation and stage meaning, not identical receipt bytes |

Source media partitioning and compute partitioning are independent. A whole video
can use sharded work; a clipped recording can use one allocation. A new interface
does not require concatenating MP4s or putting every payload in one physical file.

For frame-local computation, assembly may be a deterministic ordered merge.
Temporal quality, tracking, smoothing, derivatives, swim-bout detection, and event
statistics need boundary semantics: validated state carry, overlap/halo trimming,
or recording-wide reconciliation. An event spanning a clip boundary must not be
split or counted twice. These are preservation requirements, not optional cleanup.

The common output should use each family's existing normal representation: e.g.
canonical `detect_runs`, refined detection snapshots, keypoint/crop manifests, and
modern subject-mask bundles with physically present dense `masks_roi`. Historical
`latest_collection` readers remain named compatibility adapters until migrated and
parity-tested. An absent `latest_collection` does not invalidate a newer top-level
publication; an absent `authoritative_run` does not imply absent raw-detection or
bundle authority.

Supplier sufficiency remains use-scoped. A validated body-frame supplier suffices
for a consumer reading only origin/axes/heading/validity. Do not reopen sealed
keypoints or masks, or invent human review, solely to make a diagram look uniform.
Human acceptance is required only for a declared reviewed claim and intended use.

## 4. What the live recordings actually contain

The recording root is `/groups/johnson/johnsonlab/jeremy/recordings`. A bounded
search for recording-level `recording_clip_index.json` found eight directories:
four cameras (`2010093`–`2010096`) in each of these two sets.

| Observation | May sleepyfish (`sleepyfish_2026_05_05_17_45_30_cam…`) | Date-named August set (`2026_08_06_19_13_35_cam…`) |
|---|---|---|
| Clip index | Historical `clips` array; no `schema_id`; mode `materialized_stream_copy` | `palette.orange_external_ipc_recording_clip_index.v1`, `rows` array, mode `rolling_clips` |
| Clips / frames per camera | 22 / 1,188,000 | 55 / 2,937,604 |
| Recording frame index | `palette.recording_frame_index.v1`; manifest declares `status=ok` | Same frame-index schema and declared status |
| Acquisition source metadata | `palette.source_video_metadata.v2`, `single_video`; locator into recording-level `cams/` | `palette.source_video_collection_metadata.v1`, `clipped_video_collection`; locator `recording_frame_index.parquet` |
| Acquisition mode | `external_video_v1` | `external_clipped_videos_v1` |
| Publication status declaration | `published_canonical_v1` | `published_canonical_v1` |
| Clip-camera namespaces | 22 | 55 |
| Historical finalized collections visible in consolidated metadata | One per camera; refined-detection `latest_collection` present | None; top-level raw detection `latest` present instead |
| Top-level keypoint selection | Present | Present |
| Root subject-mask bundle authority envelope | Absent in the observed roots | Present, generation 2 |
| Analysis root consolidated metadata | Present | Present |
| Current source identity profile on root / recording manifest | Absent / absent | Absent / absent |

May camera `2010095` also has a top-level raw-detection selection and lacks the
`analysis_layout` marker that the other three May roots have. Even within one set,
directory names and nominal cohort membership do not establish identical outputs.
May manifests use `recording_id=2026_05_05_17_45_30`, while their analysis roots use
camera-qualified sleepyfish IDs. These are historical, unprofiled declarations;
do not relabel them as valid current-v2 claims or repair IDs by string substitution.

The August downstream run labels contain `sleepyfish_2026_08_06…`, despite the
date-only recording directories. Labels are not scientific identity or reliable
cohort classification. Resolve exact roots, camera IDs, manifests, and receipts.

On sampled-training surfaces, all four May recording directories have both
`*_training.zarr` and `*_clipped_training.zarr`; their inspected root metadata has
no inline consolidation. The August directories exposed only their analysis Zarrs
in the bounded scan. This says nothing about whether those training archives have
been accepted or whether exports exist elsewhere. Do not double-count both May
sampling representations as independent biological data.

### Organization observed in one camera from each set

Both have recording-level manifests/frame indexes and:

```text
<recording>/
  recording_manifest.json
  recording_clip_index.json          # historical clips[] or native rows[]
  recording_frame_index.parquet
  recording_frame_index_manifest.json
  raw/                              # acquisition/session/protocol evidence
  derived/                          # derived diagnostics and retained evidence
  clips/clip_000000/
    Cam…mp4
    Cam…_meta.csv
    Cam…_keyframe.json
    clip_manifest.json
  zarr/<recording>_analysis.zarr/
    clips/<clip>/cameras/<camera>/…   # workload-specific evidence/output
    detect_runs/…                    # when recording-level output is published
    refined_detect_runs/…
    analysis/…
```

May also has `cams/` holding the whole-video locator. The inspected August root
has no `cams/` and retains `rolling_clip_consolidation_receipt.json` and
`rolling_clip_source_cleanup_receipt.json`. Their presence is operational history,
not permission to rerun cleanup or proof of every receipt's contents.

The whole-video reference
`2026-08-12T21-59-55Z_arena_4_goodbatbadbat` declares 182,440 source frames,
`single_video`, and published `external_video_v1` authority. It also contains
acquisition crop-video and geometry-bundle evidence. Its manifest/root recording
IDs differ, neither declares current-v2 identity, and its `.imports` directory is
absent. Thus it is a useful historical layout reference, not a production canary
of the newer receipt-bound importer.

## 5. Initial evidence ledger: exact claims and consuming boundaries

Terminology: a locator finds an artifact; a digest fingerprints a declared byte or
logical domain; a receipt binds claims/evidence; a publication marker advertises a
lifecycle state; a selector chooses a publication for a declared use. A digest is
not issuer authentication. `sha256` alone does not describe what was hashed.

### 5.1 Transfer and organization

| Artifact / owner | What it records or hashes | What consumes it / claim limit |
|---|---|---|
| Installed `_citrus_transfer_complete.json`, `citrus.transfer_completion_marker.v1` | Transfer status, destination, local-target flag, verification mode and payload kind; poller key hashes marker pathname | Poller checks declarations and destination agreement. `verify_mode=quick` is not an end-to-end content receipt; marker-path identity is not marker-generation/content identity. |
| Poller `.submitted` state, LSF job ID, per-dispatch logs | Submission acknowledgment and operational identifiers | Prevents ordinary duplicate dispatch; does not establish job completion or successful import. See F1. |
| Organizer `recording_manifest.json` | Current source identity fields, relative raw/cams/derived inventory, acquisition context, geometry references and diagnostics | Identity is preflighted before movements; the identity claim does not hash the entire manifest or every listed file. Partial file operations can still be warnings; see F6. |
| Geometry bundle | `palette.recording_geometry_bundle.v1`; contract/asset-manifest digests, snapshot pointer and materialized-asset status | `verify_recording_geometry_bundle` and H5/folder loaders in [recording_geometry.py](../../src/fisheye/shared/recording_geometry.py); later physical binding must match the exact camera/pixel frame. Import is not geometry selection or scientific review. |
| `_palette_batch_disposition.json` / run-local `batch_disposition.json` | `palette.staging_batch_disposition.v1`; moved, verified fanout copy, retained authority, disposable diagnostic, unknown | [staging_batch_disposition.py](../../src/fisheye/shared/staging_batch_disposition.py) uses file-content hashing for verified copy residues. It is cleanup-planning evidence, not permission to delete and not proof all sources were imported. |
| Rolling consolidation and cleanup receipts | Separate historical operation evidence; per-artifact/path and verification context | [consolidate_external_ipc_rolling_recordings.py](../../src/fisheye/utils/consolidate_external_ipc_rolling_recordings.py) is a repair/consolidation path, not the normal automatic ingest entry point. Source cleanup stays separately authorized. |

### 5.2 Whole-source metadata, identity, and provenance

| Artifact / schema | Exact binding and important exclusions | Producer / validator / consuming boundary |
|---|---|---|
| `palette.source_recording_identity.v2`; `palette.source_recording_identity_claim.v2` | Exact recording ID, session UUID, camera ID, optional declared mapping profile; verified manifest/direct-root roles and source-analysis classification. Not MP4/H5 byte identity or a full manifest hash. | [source_recording_identity.py](../../src/fisheye/shared/source_recording_identity.py): strict bounded JSON, duplicate/non-finite rejection, root/manifest agreement. Current declarations cannot fall through to legacy after a one-sided profile loss. |
| `palette.source_recording_id.session_camera_sha256.v1` | Declared mapping derives recording ID from canonical JSON of exact session UUID and camera ID, not pathname | Organizer mapping helpers and `SourceRecordingIdentity`; never retroactively infer this profile for an old human-readable ID. |
| `palette.source_video_metadata.v2` | Stream dimensions/count/FPS/codec/colorimetry and authoritative recording-relative locator, with compatibility mirrors | [source_video_metadata.py](../../src/fisheye/shared/source_video_metadata.py): resolver rejects traversal, collection layout, and conflicting mirrors. Absolute mirrors do not become stable identity. |
| `source_video_*` / `source_h5_*` `stat_v1` fingerprints | SHA-256 of resolved path, size, mtime and supplied metadata, **not file content** | [import_source_fingerprint.py](../../src/fisheye/shared/import_source_fingerprint.py); deliberately cheap, path-dependent diagnostics. Same-size/mtime-preserving replacement is outside this claim. |
| `palette.acquisition_import_ownership`, version 1 | Recording/camera identity; authority mode; digest of parsed source-video metadata; materialization evidence only in applicable mode | [pixel_frame_authority.py](../../src/fisheye/shared/pixel_frame_authority.py), `_acquisition_ownership_record`, stamp/load helpers. External-video mode has no decoded-frame or encoded-MP4 content claim. |
| `palette.acquisition_camera_frame` | Camera dimensions, source extent/frame domain and ownership-bound frame evidence | Same owner; `load_persisted_acquisition_camera_authority` returns verified ownership/frame objects. Geometry is not a claim that every media byte was hashed. |
| Acquisition materialization and physical-chunk manifests | Materialized frame/index identities, decode/import operation, array storage identity and encoded physical-object evidence | Same owner; applies to actual full materialization, not metadata-only import or an arbitrary same-size image array. Physical identity is layout-dependent. |
| `palette.acquisition_authority_publication_status`, version 2 | Exact status, reason, mode, authority path, completion and resumption semantics; duplicated at root and `raw_video` | [acquisition_publication_status.py](../../src/fisheye/shared/acquisition_publication_status.py) validates agreement; [import_video_metadata.py](../../src/fisheye/shared/import_video_metadata.py) publishes ownership/frame evidence before published status. The status record has no independent self-digest and is not a substitute for its referenced evidence. |
| `palette.recording_import_receipt.v2` | Closed canonical-JSON document: clean full producer commit, importer ID, import-config SHA, relative target, source identity claim, acquisition ownership and frame refs/digests. Self-digest excludes `receipt_sha256`. | [recording_import_receipt.py](../../src/fisheye/shared/recording_import_receipt.py): ≤256 KiB, strict fields/paths/JSON; immutable digest-named `.imports/<sha>.json` installed via no-overwrite hard-link publication. Registry validates exact sidecar and live evidence, not receipt presence. |

The v2 import receipt is a **bounded source/import witness**, not an umbrella
digest for every imported product. It does not directly bind frame-clock, subject
metadata/setup, stimulus, geometry, crop-ledger, whole-store content, or an
environment lock digest. Its `config_sha256` binds the effective import-options
object built under `palette.recording_import_config.v1`; the receipt itself does
not embed that object. Preserve invocation/log evidence and the pinned source
needed to reconstruct it. A downstream consumer must resolve whichever additional
supplier its own contract requires, not assume the import witness proves all of
them.

The importer checks its clean full commit before mutation and again before
minting, and rejects changed code identity. Literal `git_dirty=false` in a valid
receipt is therefore enforced on the real producer path, not merely invented by
the serializer. It is still self-reported provenance, not a cryptographic signature
authenticating who executed the code.

### 5.3 Timing, acquisition crop streams, and experiment context

| Artifact / owner | Digest scope | Validation / lifecycle assessment |
|---|---|---|
| `palette.acquisition_frame_clock.v1`, [acquisition_frame_clock.py](../../src/fisheye/shared/acquisition_frame_clock.py) | Record SHA; per-array dtype/shape/C-order digests for recording/parent frame IDs, camera/system nanoseconds, and validity; declared clock-domain semantics. Original CSV/Parquet evidence is locator/size/mtime, not a source-file content hash. | Resolver rehashes bound arrays and rejects tampering, but does not rerun source semantic validation. New-run selection precedes final loadback (F3); contradictory PTP status and lossy source coercion can be sealed (F8/F9). |
| `palette.acquisition_crop_stream_ledger.v1` and `.collection.v1`, [acquisition_crop_stream_ledger.py](../../src/fisheye/shared/acquisition_crop_stream_ledger.py) | Source metadata digest, source-sidecar identities, source stream contract, row coverage, video fingerprint; collection adds source-member mapping. Record is not a decoded-output-array digest manifest. | Complete rows include blank/no-detection states. Current validator checks selected run, matching digest strings, required array lengths, and collection member-table presence; F4. Pointer-last structure alone does not prove payload integrity. |
| `palette.subject_metadata.v1`, [subject_metadata.py](../../src/fisheye/shared/subject_metadata.py) | Canonical subject-metadata record digest | Immutable subject metadata publication and resolver. Recording subject count/identity must not be replaced by population count inferred from a dish. |
| `palette.experiment_setup.v2`, [experiment_setup.py](../../src/fisheye/shared/experiment_setup.py) | Setup record plus exact subject-metadata reference/digest; explicit/count-only/partial assignment semantics | `resolve_experiment_setup` verifies the selected subject-metadata binding. Current H5 import publishes both records before returning success. |
| Stimulus/calibration import | Separate stimulus run/provenance and coordinate contracts | `run_stimulus_import` invokes `analysis.import_stimulus_to_zarr`; optional only when workflow permits. `--stimulus-metadata-and-calibration-only` deliberately omits uncontracted positional surfaces. Full stimulus receipt/consumer trace is next-stage work, not implied by import receipt v2. |

`recording-only` means absent experiment/stimulus context is legitimate. It does
not forbid ordinary movement, shape, eye, or tail analysis. An analysis requiring
stimulus alignment must reject missing stimulus evidence; an unrelated movement
consumer should not acquire a fabricated stimulus or human-review gate.

#### Clock proof is use-scoped, not one availability flag

The importer can explicitly record `unavailable_no_camera_clock_source` and
continue. With a source, current validation requires a complete ordered parent
frame domain, contiguous recording IDs, aligned vectors and at least one usable
camera or system timestamp. It permits equal timestamps and sparse validity. That
can preserve honest acquisition evidence; it does not establish usable timing for
every analysis. `available=true`, a complete run, or a verified digest cannot
upgrade the supported scientific claim.

The following is an audit acceptance matrix, not a new persisted capability schema
or permission to bypass existing consumer contracts:

| Intended use | Necessary timing claim | What is not enough |
|---|---|---|
| Frame-local pixel computation | Exact recording/source/frame correspondence; absent or uncertain clock remains explicit | A guessed frame mapping; pretending absent timestamps were measured |
| Named nominal-FPS analysis | Exact acquisition frame domain, bound FPS and explicit approximation policy | Relabeling frame-index/FPS values as measured intervals or UTC |
| Measured within-recording timing | Exact integer clock values and units, valid required rows/transitions, positive intervals, and declared reset/gap policy | A single usable timestamp, duplicate time at a required transition, or a nondecreasing aggregate; absolute epoch is unnecessary |
| Temporal computation across clips | The same required clock and frame claims across each boundary, plus state carry/halo/reconciliation appropriate to the computation | Contiguous file numbering or blind concatenation; a source gap is not automatically continuous time |
| Stimulus/session alignment | Exact frame correspondence and bound producer/session-time meaning, with observed/interpolated rows distinguished | Controller logging time relabeled as visual presentation time, or a camera timestamp assigned an unsupported session origin |
| Cross-camera or absolute UTC analysis | A shared clock or validated transformation, applicable timescale/offset, synchronization evidence and uncertainty appropriate to the use | PTP enabled, host arrival time, a timestamp's magnitude, or the classifier label alone |
| Publication and reuse | Required payload and semantic checks before selection; exact root/consolidated generation bindings; safe failure/retry | A matching self-digest over semantically invalid input or a prematurely selected run |

Immediate maintained consumers are not uniform:

- [provider_recording_timing_authority.py](../../src/fisheye/analysis_workflows/provider_recording_timing_authority.py)
  binds the selected clock and source metadata, verifies camera/count and the exact
  parent domain, but explicitly implements
  `nominal_fps_bound_to_acquisition_frame_domain.v1`. Its numerical rule is frame
  difference divided by FPS. It refuses native clipped-collection metadata; it
  currently supports `source_video_metadata.v2` with `layout=single_video` only.
- [resolved_epoch_selection.py](../../src/fisheye/analysis_workflows/resolved_epoch_selection.py)
  optionally binds that authority and checks identity/count/FPS when present. Its
  persisted timing rule remains frame-index/FPS, not measured camera timestamps.
- [chaser_proxy_relative_frame_adapter.py](../../src/fisheye/analysis_workflows/chaser_proxy_relative_frame_adapter.py)
  separately requires exact integer, nonnegative producer-logged session times,
  equality across chasers for the same acquisition frame, and strictly increasing
  mapped times. It binds `citrus_session_monotonic_ns` and prohibits interpolation;
  it does not reinterpret inferred camera PTP semantics as this session clock.
- [recording_distribution_timebase_adapter.py](../../src/fisheye/analysis_workflows/recording_distribution_timebase_adapter.py)
  loads an exact sealed relative-frame child, maps by acquisition-frame keys, and
  leaves missing/invalid rows uncovered rather than interpolating them.

The strict clock resolver verifies integrity and selection. It is not a universal
proof of elapsed time, absolute epoch, stimulus alignment, or cross-camera sync.
Extending a layout adapter while preserving an existing nominal-FPS policy is a
different change from replacing that policy with measured timestamps; the latter
requires an explicit scientific/recipe compatibility decision and parity tests.

### 5.4 Clipped-source and sampled-training evidence

| Artifact / schema | Binding | Validation / claim limit |
|---|---|---|
| `recording_clip_index.json` plus per-clip metadata | Source membership and camera/session-continuous frame IDs; historical `clips[]` and native `rows[]` grammars | [build_recording_frame_index.py](../../src/fisheye/utils/build_recording_frame_index.py) accepts supported input layouts; original producer artifacts remain lineage. |
| `palette.recording_frame_index.v1` Parquet plus manifest | Parent frame → camera, clip, local frame, metadata/video paths and timestamps; manifest records checks/status | Derived recording-level map, not review state. The frame-index builder's manifest is not itself a full byte-bound publication receipt. |
| `palette.source_video_collection_metadata.v1` / `palette.source_video_collection.v1` | Collection digest over three indexed-file evidences (clip index, Parquet, frame-index manifest; each byte SHA/size/mtime/relative path) and ordered member stream facts/stat fingerprints | [clipped_video_collection.py](../../src/fisheye/shared/clipped_video_collection.py) validates one camera, indexed/probed extents, geometry/FPS consistency, membership and interval coverage; exact row-bijection gap F2. Videos are stat-bound, not content-hashed. |
| `verify_clipped_video_collection_live_files` | Rechecks indexed-file content digests and each member's live stat identity | Called by clipped crop pixel-source binding after canonical acquisition parsing. The helper alone is not the complete metadata-envelope parser; do not promote it into a universal source validator. |
| `palette.clipped_analysis_zarr_shell.v1` shell manifest | Planned/written paths, clip sources, frame summary, code/environment metadata, and acquisition publication refs | [create_clipped_analysis_zarr.py](../../src/fisheye/utils/create_clipped_analysis_zarr.py) publishes acquisition records but does not emit whole-import v2 receipt/profile or perform a final root consolidation. A shell is not a ready downstream analysis deliverable. |
| `palette.clipped_training_zarr.v1`; `palette.training_source_frame_index.v1` | Materialized sample rows, parent-frame indices, source clip/local-frame Parquet mapping, options and optional copied detection lineage | [create_clipped_training_zarr.py](../../src/fisheye/utils/create_clipped_training_zarr.py): source-map checks and optional sibling manifest; not a sealed decoded-sample-content or current source-recording receipt. |
| Clipped-training provenance report | Source-map row alignment, declared pixel contracts and downstream crop references | [validate_clipped_training_provenance.py](../../src/fisheye/utils/validate_clipped_training_provenance.py); a diagnostic/repair surface, not authority to rewrite pixels or relabel legacy decoding as canonical luma. |

Training indices must remain explicit:

```text
stage frame_indices = sample_index
  → raw_video/original_frame_indices = parent_frame_index
  → source_frame_index.parquet = camera + clip + clip_local_frame_index + video
```

Inside historical clip-local analysis runs, `frame_indices` is normally clip-local;
inside canonical recording snapshots it is the declared recording/acquisition
domain. Never join those arrays positionally. A legacy `refined_row_id` is scoped
to its exact run/clip; a canonical finalizer must validate recording-global row
identity and stable instance keys.

### 5.5 Registry publication

[recording_identity_authority.py](../../src/fisheye/registry/recording_identity_authority.py)
is the current identity owner. `synchronize_recording_import` dispatches current
profiles to either fresh receipt finalization or refresh of an already-bound
receipt. It does not discover an arbitrary sidecar and silently treat an unbound
archive as finalized. Unprofiled legacy artifacts use a distinct compatibility scan.

Fresh projection calls `_verify_live_import_receipt(...,
require_current_producer_code=True)` before/around persistence. It checks:

1. Exact validated receipt type, identity claim, recording-relative target and
   digest-named sidecar.
2. Clean binding checkout matching the producing commit.
3. Live manifest/direct-root identity, published external-video status, matching
   ownership/frame refs and digests, camera/recording identity, and stable status
   across the read boundary.
4. Registry schema, conflicts, unique receipt binding and normalized identity
   history/current state within a transaction.

The current routine requires `external_video_v1`; it is not a generic clipped
identity/admission API. Do not widen migration 73 implicitly just to reuse its
method name. A future convergence package needs an explicit compatible profile
decision, while preserving the same logical recording meaning.

[shadow_publish.py](../../src/fisheye/registry/shadow_publish.py) makes the import
registry mutation on a local SQLite candidate, preserves a backup, validates full
`integrity_check` and `foreign_key_check`, checks canonical/staged file hashes and
concurrent-change conditions, atomically replaces the canonical file, and validates
the result. It also checks the configured writer-host boundary. These checks use
Palette's Python SQLite runtime; a separate CLI result cannot satisfy acceptance.

Identity registry rows, dataset locator/path hashes, and explicit acquisition-batch
assignments are distinct. [Acquisition-batch membership](../acquisition_batch_registry_contract.md)
must be explicitly assigned; matching timestamps, camera suffixes, or similar names
do not establish a batch or biological subject identity.

### 5.6 Requested producer evidence from Orange and Citrus

Added 2026-09-06 following the request for acquisition-agent requirements. This is
a proposed producer handoff, not authorization to change acquisition, add a new
universal receipt, or rewrite historical datasets. Ask each agent to distinguish
**already emitted**, **implemented but not deployed**, **derivable with declared
limitations**, and **not recoverable / needs new instrumentation**. Require exact
producer commits, deployed versions and representative emitted artifacts for each
claim; source code presence alone does not establish what a recording contains.

Start with field semantics and small replayable examples, before redesigning
schemas. Extend the existing Orange session/clip/stream contracts and Citrus
session/stimulus provenance with one agreed owner per shared interface.

| Owner / priority | Confirm from the producing implementation | Make available in the acquired dataset |
|---|---|---|
| Orange — required source/frame identity | Where recording IDs are assigned; relation to camera hardware counter; 0/1 origin, reset/reconnect behavior; whether encoded presentation order matches metadata rows | Explicit session/camera/parent-recording identity; exact clip/local → recording/acquisition mapping; hardware counter if available; dropped/repeated/undelivered/blank frames and discontinuities distinguished. Preserve clip IDs; do not rename a gap away by rebasing. |
| Orange — required clock meaning and coverage | For `timestamp` and `timestamp_sys`: source API, unit, domain/epoch, actual sampling point, missing/sentinel rules, precision, wrap/reset behavior. Is hardware time exposure start, midpoint, end, or unspecified? | Exact integer values and validity tied to the same source-frame keys; explicit semantics and unknowns. Preserve hardware time separately from host-arrival time and MP4 PTS/DTS. FPS and packet counts do not supply a measured clock or frame map. |
| Orange — synchronization-dependent uses | Exact PTP state grammar; what enablement, lock, offset and latch samples prove; clock/grandmaster identity; which time interval each observation covers | State transitions or interval-qualified telemetry, source sample times, relevant offsets and uncertainties, sync loss/recovery, and applicable timescale/UTC-offset evidence. A session-wide summary must disclose coverage, not imply every exposure was synchronized. |
| Orange — whole/rolling completion and pixel identity | Clip rollover/encoder buffering rules, incomplete-final-clip behavior, presentation-order mapping, full-frame versus runtime crop roles | Closed ordered member inventory, first/last IDs plus row-level mapping, required/optional artifacts and terminal status; effective camera/decoder/encoder/pixel settings, crop geometry and blank/detection policy. A crop stream remains a derived acquisition input, not full-frame pixels. |
| Citrus — session time and source linkage | Where `timestamp_ns_session` is sampled; steady-clock origin; relation to logging/update/enqueue; restart behavior; whether the source camera ID denotes an input consumed or a displayed-state alignment | Exact stimulus-state/trial/protocol/subject-assignment identity and integer session time, tied to the actual source acquisition ID and camera. Preserve multiple valid stimulus-state rows for one camera frame; do not deduplicate or claim simultaneity from that join. |
| Citrus + Orange — only when cross-domain alignment is claimed | How camera, Citrus session and host clock domains can be related; measurement procedure, validity intervals and drift/uncertainty | A content-bound measured clock transformation or paired observations with declared method and uncertainty, including restart/reset boundaries. Do not synthesize a transform from matching frame numbers or nominal FPS. |
| Citrus — only when displayed/exposure-aligned stimulus is claimed | Difference between CPU state, queued raster, actually selected display buffer, swap/completion and physical presentation | State/buffer-bound presentation events and the calibration/evidence needed for the claimed physical timing; otherwise an explicit unsupported claim. Logging `glfwSwapBuffers` alone is not proof of scanout or exposure alignment. |
| Both — reproducibility and transfer | Exact running build/config; which artifacts are mandatory; when writers close/finalize; what transfer verification actually checks | Full producer commit/build identity and dirty state, effective acquisition/protocol settings, relevant SDK/firmware/runtime and model/calibration references; immutable final artifact inventory with relative locators, byte sizes and suitable content digests; transfer evidence bound to that exact source generation. |

Clock uncertainty should not erase good pixels. A source can preserve incomplete
timing honestly while timing-dependent consumers refuse unsupported uses under
§5.3. Conversely, retaining a hardware timestamp does not justify inventing an
epoch or exposure reference point.

There is already a specific Citrus/Orange question to resolve, not merely a
generic wish for more timestamps. The existing
[stimulus-to-camera temporal projection audit](../chaser_stimulus_camera_temporal_projection_audit_2026-08-20.md)
reports CPU-side session-state logging, no state-bound presentation timestamp, an
unspecified camera exposure reference, and no checksummed camera/session clock
transform for its inspected recordings. It also lacked the exact Citrus producing
commit because `software_version` was not persisted. These are dated findings:
ask the agents whether current deployed code has addressed them and provide actual
output evidence. They do not prove every later recording has the same limitations.

Requested handoff from each agent:

1. A versioned field dictionary with exact producer code anchors and a table of
   claim → artifact/path → emitted field → availability in historical/current data.
2. One small valid whole recording and one recording spanning multiple native
   clips, or equivalent deterministic producer fixtures, with emitted sidecars and
   expected exact frame/time joins. Include valid 0/1 origins and the supported
   physical row-order behavior.
3. Fault fixtures for a missing/duplicate frame, rollover, incomplete final member,
   process restart/clock reset, absent clock, and PTP loss or contradictory state;
   include partial transfer/retry fixtures at the transfer owner. Hardware fault
   injection or new acquisitions require their own authorization.
4. Exact code/config/runtime bindings and compatibility limits. Mark historical
   evidence that cannot be recovered; do not backfill invented clocks, provenance
   or receipts.

Keep writer completion, transfer completion, dispatch acknowledgment, Palette
import acceptance, and scientific-use acceptance as separate claims. F1's poller
acknowledgment and F2/F3/F8/F9's Palette validators remain Palette-side work even
if upstream artifacts are strengthened. A new acquisition receipt cannot repair
those consumers by itself.

### 5.7 Reconciliation with the acquisition-agent handoff

Reviewed 2026-09-06: the complete
[producer timing/provenance audit at `4d80fa92ff926baaa4e3c7408d39ed054a5df732`](https://github.com/jmdelahanty/agent-contracts/blob/4d80fa92ff926baaa4e3c7408d39ed054a5df732/orange-palette/producer_timing_provenance_audit_2026-09-06.md)
and its
[coordination index](https://github.com/jmdelahanty/agent-contracts/blob/4d80fa92ff926baaa4e3c7408d39ed054a5df732/docs/current-agent-coordination.md).
The audit blob is `288879f1f768546945d2dee5d2423d20dfb9ef55`. This is the shared
cross-repository evidence baseline, not an implemented schema, new acceptance
authority, or authorization to start its proposed corrections.

The producer reviewer inspected Citrus
`9071cfccbf9155b330ce491fabe16fc39b968852` and Orange revisions
`5cf21a9f9de0a86f58009722a39da3708f000403`,
`2f009be1d23671fc5c4f4b19b1bb4aa43e95f260`,
`f5ce5d09d6a102adffab6818aed505f82e52a66b` and
`b3e8aa60bc8a325ebcc3b5035ec8c3397cea80e4`. Exact deployed binaries were not
established. Producer tests were inspected but not run in that review. Those
revisions were absent from the local producer Git object stores; attempted pinned
GitHub file reads did not provide them. Thus the producer source/artifact findings
below are report-backed, not independently re-executed by this Palette audit.
Palette code references and tests remain independently checked at this document's
`0a9531…` base.

#### What changes or sharpens the current assessment

| Producer evidence at the pinned handoff | Palette reconciliation / consequence |
|---|---|
| Orange assigns a 1-based recording counter, with 0 meaning not recording; stream-local and hardware counters are distinct. Some resource rejection happens before recording-ID assignment. | A dense recording sequence proves coverage of that assigned domain, not capture of every hardware exposure. Require truthful disposition/coverage claims; do not infer a gap-free camera from dense IDs. Preserve supported historical 0-based profiles rather than globally changing their interpretation. |
| The example recording begins at `recording_frame_id=1`, `local_frame_id=90919`; its crop hardware counter is another value. | Orange `local_frame_id` must not be confused with Palette's zero-based `clip_local_frame_index`. The current frame-index builder derives clip-local position by metadata-row enumeration; retain that distinction and verify encoded presentation-order correspondence. |
| Orange's new acquisition-index seal covers qualifying external whole recordings, but its native header and top-level-only source traversal do not establish native/rolling parity. | F5 is a two-sided convergence gap, not just a missing Palette wrapper. Orange owns the producer extension; Citrus and Palette must review the same versioned grammar. A valid rolling recording is not automatically eligible for a sealed-map-only consumer. |
| PTP state evidence is sampled; the producer uses an exact `slave`/`master` whitelist. Exposure phase/precision remain unknown, and some classification inputs can be unfinalized or stale. | F8's substring admission remains a Palette defect. Do not copy a positive label into a continuous synchronization claim. Bind observation coverage/closure and preserve inference/traceability limits; do not silently adopt a new whitelist for all historical grammars without a compatibility decision. |
| Orange samples realtime after optional PTP work, not immediately upon camera-return; an earlier receive sample is in the steady-clock domain. | Palette currently persists `capture_point=immediately_after_EVT_CameraGetFrame_returns` in its clock descriptor. Reconcile that descriptor with the actual producer profile; retain separate host-receive and realtime samples. The producer report supplies a concrete semantic correction to review, not permission to rewrite old seals. |
| Citrus implements selected-state/buffer and software compose/swap boundaries, plus bracketed steady/system samples at session start/end. The inspected smoke has no live source and zero display-submission rows. | Update the older blanket “no submission evidence” description for these reviewed revisions: the machinery exists. But schema presence, an empty table and endpoint correlations do not prove live display mapping, continuous drift bounds, camera/session transformation, or physical exposure alignment. |
| Citrus state logging occurs after stimulus computation and before protocol advancement; held targets and source-input IDs are distinct. | The older producer description's relative protocol timing is not assumed current. Preserve native state identity and one-to-many state/input joins; neither state time nor input provenance establishes displayed-stimulus simultaneity. |
| Citrus can write raw `session_status=COMPLETE` after semantic finalization fails, while the finalized observation receipt refuses `close_complete=false`. | Audit receipt-aware and legacy import paths separately. Do not promote the raw string to proof of successful finalization or claim the receipt-aware path is broken merely because the raw status is misleading. |
| Producer runtime Git snapshots are not exact compiled build identities; whole-video closure lacks a video-content digest; transfer markers lack exact source-generation binding. | F7 remains open. Preserve stronger specialized ROI/H5 receipt contracts and separate source closure, transfer, dispatch and Palette acceptance. Do not replace existing family evidence with a weaker universal inventory. |

The handoff also identifies a live-example clock coverage inconsistency: an
unfinalized summary starts at recording ID 67794 while that recording's CSV starts
at 1, with differing stream-local counter context and older host-management
capture. This is a producer-reported example requiring investigation of reset,
scope and freshness. It is not a finding about the May/August recording roots
observed in §4, nor proof that their clocks are corrupt.

#### Existing Palette contract that constrains the proposed extension

The current v6 stimulus adapter already admits a closed
`orange.recording.acquisition_index_mapping` version 1 in
[stimulus_coordinate_contract.py](../../src/fisheye/shared/stimulus_coordinate_contract.py),
`_v6_validate_orange_source_record` around line 1960. It requires finalized status,
the declared 1-based `uint64 recording_frame_id` → 0-based `int64
source_acquisition_frame_index` conversion, exact camera/recording binding, one
CSV artifact reference per camera, and complete 1…N coverage with no gaps. That is
a specific existing profile, not a generic allowance for reordered clips, subsets
or arbitrary source-ID offsets. The raw Citrus H5 binding separately records an
exact finalized-receipt reference and H5 digest/size/relative path.

Therefore extending producer mapping to multiple clips or additional gap policies
requires a compatible/versioned consumer plan through this existing owner as well
as the source importer. Do not just loosen closed-key validation or equate native
recording validity with v6 readiness. Focused v6 adapter validation passed all 16
existing tests; it does not establish successful live Orange → Citrus → Palette
production with the proposed extension.

#### Coordination and remaining handoff items

Use `agent-contracts` for cross-repository interface decisions and evidence; keep
Palette implementation status in the existing authority queue. The proposed
ownership is coherent: Orange owns frame mapping/acquisition-clock/source
inventory; Citrus owns session-clock/display/transfer; Palette owns consumer
admission/publication. No item is marked implemented merely by this review.

The remaining evidence requests are:

1. Readable pinned producer source and exact deployed/build identity where it can
   be established, preserving “unknown” where it cannot.
2. A shared whole/native-rolling fixture matrix with the precise mapping fields,
   encoded presentation order, permitted source gaps and expected refusal cases.
3. A version/compatibility proposal against the existing v1/v6 consumers, including
   what the seal's “both ID columns must agree” requirement refers to; it must not
   imply that stream-local and recording counters have equal values.
4. Live nonempty software-display/session evidence before claiming that path has
   been exercised; physical timing remains a separate commissioned capability.
5. A durable published Palette audit/reproduction link. At producer-review time
   this draft existed only in the local audit worktree and was correctly reported
   inaccessible on the producer host. The user subsequently authorized
   documentation-only Palette and linked `agent-contracts` response PRs on
   2026-09-06, without merges, source fixes or deployment. The response links the
   exact published documentation revision. That publication is distinct from
   this read-only reconciliation and does not activate the proposed changes.

## 6. Actual lifecycle and outstanding findings

### Whole current-profile path

```text
preflight + identity/code checks
  → ensure archive and acquisition stream inventory
  → source video metadata + acquisition ownership/frame publication
  → acquisition clock
  → subject metadata/setup and requested stimulus import
  → consolidate and check current identity / selected crop-ledger metadata
  → immutable import receipt
  → receipt-bound registry shadow finalization (when requested)
```

The receipt is minted only after those requested steps succeed. This sequencing
does not make it a digest binding to all those products. A receipt-sealed source
cannot be casually re-imported; the supported path is receipt-bound registry
refresh. Failure before receipt installation can leave partially written state,
whose own publication/resumption rules must decide retry behavior.

### F1 — Submission acknowledgment can report false success

Classification: enforcement/operational correction required; reproduced latent
defect in the installed poller, not observed missing data. `process_session` runs
SSH and then an unconditional successful log call; its caller invokes the function
as an `if` condition. The safe reproduction below shows why `set -e` does not make
the SSH command a reliable gate in this structure:

```bash
bash -c 'set -euo pipefail; simulated_submit(){ false; printf "dispatch completed\n"; }; if simulated_submit; then printf "submission marked successful despite failed command\n"; else printf "failure detected\n"; fi'
```

Observed: both success messages, despite `false`. Installed-script anchors:
SSH around line 92; trailing log 109; conditional call 210; job-ID extraction
218–221; submitted-state write 234; retry/claim removal 246. The inspected state
census contained 68 submitted records, all with nonempty job IDs. That does not
exercise failure safety. Correction must also reconcile ambiguous LSF acceptance
before retry; returning nonzero and blindly resubmitting is insufficient.

Deployment is a separate open boundary: current main requires registry writer-host
configuration that the older deployed wrapper did not. The installed poller does
not pass it, and the inspected remote environment left it unset. A blind deployment
update could therefore break dispatch. Use required-CI-green exact code, reconcile
configuration, and record deployment path/full commit; do not update the shared
checkout as an incidental audit action.

### F2 — Clipped frame-map checks do not prove exact row correspondence

Classification: enforcement correction; in-memory reproduction at the audited
commit. `_frame_index_units` in `clipped_video_collection.py:314` uses grouped
min/max/count and interval coverage. Those summaries cannot prove uniqueness or
correct parent/local pairing.

| Four-row input | Parent indices | Local indices | Current result |
|---|---|---|---|
| Valid control | `0,1,2,3` | `0,1,2,3` | Accepted |
| Repeated parent / missing parent 2 | `0,1,1,3` | `0,1,2,3` | Accepted |
| Repeated local / missing local 2 | `0,1,2,3` | `0,1,1,3` | Accepted |
| Wrong correspondence | `0,1,2,3` | `0,2,1,3` | Accepted |

The first probe replaced only Parquet reads with in-memory Arrow tables and called
the real helper. A second probe used the maintained clipped-recording fixture,
changed its actual temporary Parquet table, then called the real
`create_clipped_analysis_zarr` and unpatched
`load_persisted_acquisition_camera_authority`. Only video probing was stubbed, as
in the maintained fixture tests. A valid control and all three malformed variants
returned `status=ok`, `published_canonical_v1`, and a five-frame acquisition
authority. The two-clip fixture used first-clip parent/local arrays `0,0,2` for
duplicate cases or local `0,2,1` for wrong pairing; the second clip was unchanged.

Thus a newly built collection can seal a semantically bad frame map and pass the
acquisition loader; hashing is not the missing semantic check. This does not claim
upstream builders normally generate bad maps or that live maps contain these
cases. Preserve legitimate row-order independence by validating keyed
correspondence, not merely demanding one arbitrary physical Parquet order.

Owner seam: existing frame-index/collection validation and `ADM-001`, `TEST-001`.
Before correction, add valid unsorted-row controls and duplicate, gap, wrong-camera,
wrong-video, wrong-pairing and source-change regressions through the real shell
producer and unpatched downstream resolver. Retain bounded-memory validation.

### F3 — New frame-clock publication selects before final loadback

Classification: enforcement correction; fake-group fault injection reproduced.
`publish_acquisition_frame_clock` calls `mark_run_complete` at line 899, then
`_validate_run` at line 923. The shared completion helper immediately sets
`latest`/`latest_complete` for an eligible run.

The probe changed one stored `parent_frame_index` value during `store_array` while
leaving producer input/expected digests intact. Observed:

```text
AcquisitionFrameClockError: parent_frame_index differs from its bound digest
parent.latest == parent.latest_complete == newly created clock run
run.palette_run_completion_status == complete
run.stage_selector_eligible == True
root acquisition_frame_clock_ref is absent
```

The strong resolver detects the problem, and the outer current importer returns
failure without minting its receipt. Nevertheless the local clock selector and
completion state are misleading. Require a test preserving the previous selector
through failed write/loadback and retry. Resolve the publication ordering through
the existing lifecycle owner; do not merely suppress the exception or weaken the
resolver. No production clock payload was read or modified by this probe.

### F4 — Crop-ledger validation is not decoded-payload validation

Classification: verified claim limit / enforcement gap if used as full integrity
admission. `_validate_current_run` at `acquisition_crop_stream_ledger.py:875`
compares digest strings, completion, array presence and first-axis lengths. It does
not recompute the record seal, rederive rows from the bound source metadata, or
verify decoded geometry/validity values. The current import consolidation check
calls this same validator.

Do not describe that successful call as proving every copied crop-ledger value.
Add same-shape tamper cases for crop geometry, frame IDs, blank/detection flags,
categorical mappings and collection member indices. Establish the output claim at
the writer through compatible receipts or a full-strength verifier under explicit
immutability assumptions. This is not a request for routine full-MP4 hashing or
for replacing product-specific crop semantics with a generic array-shape check.

### F5 — Fresh clipped intake and whole import do not close the same boundary

Classification: supported-path/compatibility and lifecycle gap. Current
`run_citrus_session_import` invokes the organizer and whole import-only wrapper.
The recording-only organizer explicitly rejects multiple rolling clip IDs
(`organize_recordings.py:1454`); it does not automatically organize each into a
proper parent and call the clipped shell producer. Rejecting split recording IDs
is correct; an ordinary parent-recording intake path is still needed.

The clipped shell creator publishes camera acquisition authority but does not
declare current source identity, emit receipt v2, run the whole context/clock
sequence, or finalize consolidated root metadata. The real-producer fixture probe
in F2 also confirmed `consolidated_metadata` absent/null after successful fresh
shell creation. The sampled-clipped training
creator likewise does not perform final consolidation or seal decoded sample
content. A later enclosing workflow may consolidate; these standalone entry points
do not prove final immutable publication on their own. Define the enclosing owner
and completion claim explicitly before calling them import-complete.

Preserve the distinction between source import, sampled training, historical repair,
and downstream admission. Do not force every profile through migration 73 or mint
current receipts for historical stores just by adding attributes. Queue seams:
`RID-001`, `ADM-001`–`003`, `PROD-DET-001`, `TEST-001`.

### F6 — Organization and preflight summaries have narrower success semantics

Classification: operational claim/required-input policy needs explicit closure.
Identity/geometry preflight rejects serious conflicts before movement, but ordinary
copy/move exceptions become warnings in `_apply_plan`; `recording_applied` can be
logged afterward and `main` can return zero. The manifest lists planned artifacts,
not a content-verified transaction of all successful copies.

`preflight_gate_reason` rejects recorded `fail` unless overridden; absent, unknown,
or warning-only preflight is not equivalent to a completed passing inspection.
This may be legitimate compatibility policy. Required-source failures must be
separated from optional diagnostics before tightening it. Test a failed required
copy, existing conflicting destination, malformed diagnostic, restart after partial
move, and successful optional-diagnostic omission. Do not demand every historical
optional file or rerun expensive diagnostics unconditionally.

### F7 — Source bytes and producer/options/environment are not universally sealed

Classification: evidence-scope and reproducibility decision, not permission for a
global digest migration. Current video/H5 `stat_v1` is explicitly weaker than
content identity. The clipped collection binds index bytes but not encoded member
bytes; regenerating a Parquet file or changing its mtime can change collection
evidence even if its logical mapping is unchanged.

For unattended reuse, declare the source immutability/generation assumption and
which transfer/writer evidence makes it trustworthy. If durable byte identity is
required, establish it at acquisition/transfer/write time with a versioned
compatibility decision. Do not manufacture a future digest or require a new
universal receipt to replace every existing family. The
[receipt hashing lifecycle](publication_receipt_hashing_lifecycle_2026-08-29.md)
already describes receipt composition and when deeper revalidation is needed.

Current `create_clipped_training_zarr._decode_clip_frames` uses OpenCV decode and
BGR-to-gray conversion. That is not automatically byte-equivalent to native
PyNvVideoCodec luma. Preserve the
[pixel/model-input contract](../video_pixel_model_input_contract.md), recorded
decoding lineage, and supported historical reads. Changing decoding, resizing, or
model preprocessing is a scientific/pixel-contract change requiring parity
evidence, not a layout cleanup.

### F8 — PTP status parsing admits explicit contradictory states

Classification: enforcement correction; independently reproduced by the parallel
reviewer and root with in-memory summary fixtures at the audited base.
`acquisition_frame_clock.py:299` accepts a status containing any of `locked`,
`slave`, `master`, or `synchronized`. This admits negative phrases as well:

| Explicit input status | Other required evidence | Resulting camera classification |
|---|---|---|
| `slave` / `locked` | Qualifying enablement, register, offset, latch and camera/system delta fixture | `absolute_epoch` |
| `unlocked`, `unsynchronized`, `not synchronized` | Same qualifying fixture | **`absolute_epoch`**; contradiction incorrectly accepted |
| `faulty`, `listening` | Same qualifying fixture | `device_defined_unknown_epoch` |

This is not merely an unused helper: both source loaders call the classifier and
the whole-import producer publishes its returned semantics. A perfectly matching
record/array digest can therefore bind the incorrect semantic classification.
Repository search found no downstream Python consumer interpreting these camera
epoch/PTP classification fields outside the clock module. The demonstrated defect
is false semantic publication, not an observed wrong UTC calculation or proof that
any recording contains one of these status strings.

The positive inference also has a deliberately limited claim: the record labels it
`inferred_from_recording_evidence_not_sdk_declared`, and the classifier encodes a
37-second expected camera/system offset with broad tolerances. Preserve that
uncertainty; do not upgrade it to traceable synchronization or silently convert
historical clocks. Define exact accepted state grammar and contradiction handling,
then test positive, negative, unknown and missing evidence. Durable
[read-only probe commands](evidence/ingestion_clock_diagnostic_probes_2026-09-06.md)
cover this finding and F9.

### F9 — Clock source admission can change malformed values before sealing

Classification: enforcement correction for exact numeric parsing and overflow;
explicit source-domain compatibility decision for frame-ID offsets.
`acquisition_frame_clock.py:495` and `:527` coerce Parquet values with `int(...)`
and `np.asarray(..., dtype=int64)` without first requiring exact integer input.

| In-memory source input | Current admitted result |
|---|---|
| Recording IDs `1.9,2.9,3.9`; parents `0.9,1.9,2.9`; camera times `100.9,110.9,120.9` | Silently truncated to `1,2,3`; `0,1,2`; `100,110,120` |
| Recording IDs `-3,-2,-1` or `400,401,402`; parents `0,1,2` | Accepted; only recording-ID contiguity is checked |
| Camera times `INT64_MAX,INT64_MIN,INT64_MIN+1` | Accepted as monotonic because `np.diff(int64)` wraps to `1,1` |

The CSV path separately requires its first recording ID to be 0 or 1; Parquet
does not. Decide and bind the supported source-ID offset rather than blindly
rebasing it or rejecting legitimate partial-source profiles by accident. Complete
parent-frame identity remains distinct from a source camera's session ID origin.
Exact integer/domain checks must precede conversion, and monotonic comparisons
must not depend on overflowing subtraction. The overflow probe is malformed-domain
robustness evidence, not a plausible normal timestamp reversal observed in data.

These probes exercised the real Parquet loader and source validator with only
`pq.read_table` replaced by an Arrow table. They did not modify a real Parquet file
or publish a live clock. F2 is the separate clip-local/parent pairing defect; a
correction must preserve both frame-map exactness and exact clock/source identity.
Route F3/F8/F9 through the existing acquisition-clock owner and `ADM-001` /
`TEST-001` boundary coverage, not a second clock helper or independent status queue.

### Prior finding dispositions

| Older assertion / tempting inference | Current disposition |
|---|---|
| Import receipt is checked only on registry read | Superseded for current-profile imports: fresh binding verifies live acquisition evidence and producer identity too. |
| Importer simply forces `git_dirty=false` | Superseded as a producer-path finding: the importer rejects dirty/unpinned code and rechecks identity before minting. |
| Frame-clock digest gate always precedes `latest` | Not true for the new-run path; F3 reproduces the failure state. Existing-run replay does validate first. |
| Every family must have the same envelope, code block, and human-review block | Not adopted. Require the consumer's declared claims and preserve each family's scientific grammar and stable digest bytes. |
| A missing bundle envelope means May masks are missing/invalid | Not established. May has older selected refined-mask surfaces; evaluate through their supported profile. |
| All outputs under a name containing sleepyfish belong to the May source set | False as a naming rule; August run labels also use sleepyfish. |
| A green helper/unit suite proves a green whole/clipped workflow | False. Tests below do not establish production parity, closure of all consumers, or required CI. |

## 7. Digest domains must not be conflated

| Domain | Examples | What can legitimately change its digest |
|---|---|---|
| Canonical document | `manifest_digest.canonical_json_bytes`: sorted keys, compact UTF-8 JSON, no non-finite values | Changed fields; excluded fields depend on the specific schema. A pretty-printed file hash is different. |
| Raw file content | Indexed JSON/Parquet evidence; model and transferred artifact SHA | Byte changes, including re-encoding/re-serializing logically equivalent data |
| Stat descriptor | `stat_v1` video/H5 fingerprints | Path, mtime, size, declared metadata; contents can change without detection if descriptor is preserved |
| Logical arrays | Family manifests and coordinate records | Values, dtype/shape/header semantics; preserve exact per-family byte grammar |
| Rowset vs row order | Sorted-key rowset fingerprints vs ordered identity/array digests | A row permutation may preserve a set fingerprint but violate a consumer's ordered join |
| Physical storage / Merkle roots | Payload receipts, encoded chunks, materialization manifests | Rechunking, sharding, compression, topology; even decoded Merkle identity can depend on partition grammar |
| Execution/provenance | Import config, worker attempt, command/runtime receipts | Commit, effective options, execution attempt, timestamps and resources according to the schema |
| Publication/selection | Family manifest document, authority envelope, metadata generation | New publication, manifest/selector generation, approval event; not necessarily changed scientific values |

`zarr_payload_receipt.py` explicitly separates decoded payload, physical payload,
and immutable metadata roots. Preserve that distinction. A flat whole-array SHA
cannot be reconstructed by concatenating per-chunk SHA strings. A validated
whole/split comparison should compare declared scientific invariants and logical
identities, not assert every receipt, storage digest, timestamp, or execution
attestation is byte-identical.

## 8. Acceptance tests for the clean interface

These are audit closure criteria to adopt in the existing owning packages, not a
second implementation-status queue. Capture preservation tests before changes.

| Boundary | Required positive evidence | Required refusal/failure evidence |
|---|---|---|
| Intake / dispatch | One complete whole batch; native rolling batch; explicit recording-only batch; exact deployment/recipe | Malformed/stale marker, failed SSH, ambiguous accepted job, duplicate polling, partial arrival, retry after process loss |
| Organization / identity | One parent per camera, preserved source/session IDs and required inventories | Conflicting existing manifest, failed required file operation, duplicate identity, wrong camera, stripped current profile; optional omissions remain explicit |
| Source map | One-member identity source; whole + derived clips; native clip set; shuffled physical map rows with correct keys | Duplicate/missing/overlapping frames; wrong parent/local pair; mismatched video, camera, schema or source generation |
| Clock / crop ledger | Complete frame domain, use-scoped timing proof from §5.3, exact integers, blank-row preservation and verified numerical ledger | Same-shape tamper, fractional/overflow values, contradictory PTP states, invalid required transitions, publication failure preserving old selectors, source change and safe retry |
| Source receipt / registry | Real producer → immutable sidecar → shadow finalization → verified read using Palette SQLite | Dirty/wrong producer, wrong target, stale/missing sidecar, conflicting identity, registry integrity/foreign-key failure, concurrent publication, orphan sidecar recovery policy |
| Split/unsplit numerical parity | Same source frames, recipe/model/digests, decoding, thresholds, coordinate conventions and canonical keys | Model/backend/config drift, mixed attempts, unexpected or duplicate worker receipts, one missing partition |
| Temporal boundaries | Event spanning artificial cut, continuous track, quality jump/blip and smoothing edge cases | Double-counted/split bouts, lost state, overlap counted twice, silent gap interpolation |
| Finalization / readers | Same family contracts through real publisher and unpatched generic reader/exporter | Wrong source/use, incomplete member, owner/lease loss, final-location mismatch, stale consolidation, invalid selected publication with tempting fallback |
| Sampled training | Exact sample → parent → source-member map, preserved pixel/model contract | Source-map permutation, sample-local/acquisition confusion, copied detections with wrong key/source lineage, mislabeled gray/luma pixels |
| Delivery / retry | Numerical and required presentation identities reported separately | Plot failure keeps a presentation-required deliverable incomplete; safe presentation-only retry does not recompute or invalidate accepted numerical products |

New outputs should first be selector-ineligible, commit-pinned canaries. All
required CI for each incoming and integrated exact commit must pass before merge,
shared-checkout update, production selection, or any merge-ready claim. Existing
validated scientific products must not be reinterpreted or rewritten to fit the
new interface.

## 9. All-stage evidence-family index for subsequent reviews

This index covers the vocabulary in
[stage_catalog.py](../../src/fisheye/registry/stage_catalog.py), plus receipts and
products outside that advisory status catalog. It locates current owners; it is
not a replacement executable producer catalog. `ADM-003` remains the owner of
producer → profile → resolver → boundary-test inventory generation.

Depth labels: `reviewed` = this revision's source/import trace; `seam` = selected
boundary inspected/tested; `indexed` = owner located, complete gate trace pending.
For every later row, record exact schema/version, digest bytes, producer/config/
environment/input bindings, validator caller and timing, lifecycle, compatibility,
live evidence and tests before upgrading its depth.

| Stage(s) / product | Evidence families and code owners (paths relative to `src/fisheye/`) | Depth |
|---|---|---|
| Transfer, organization, `raw` | `utils/run_citrus_session_import.py`, `utils/organize_recordings.py`, `shared/recording_import_receipt.py`, `shared/source_recording_identity.py`, `registry/recording_identity_authority.py`, `registry/shadow_publish.py` | reviewed |
| Source frame/pixel/time; acquisition crop ledger | `shared/pixel_frame_authority.py`, `shared/acquisition_publication_status.py`, `shared/source_camera_physical_authority.py`, `shared/coordinate_frame_record.py`, `shared/acquisition_frame_clock.py`, `shared/acquisition_crop_stream_ledger.py`, `shared/acquisition_crop_identity.py` | reviewed scopes in §5; broader coordinate consumers pending |
| `downsample`, `background`, sampled training | `shared/import_profile_contract.py`, `utils/import_recordings_training.py`, `utils/create_clipped_training_zarr.py`, image/materialization provenance | seam; full producer matrix pending |
| `calibration`, `recording_geometry_import`, `arena_geometry_offline_fit`, `arena_geometry_comparison`, `arena_geometry_selection` | `shared/recording_geometry.py`, acquisition geometry contracts, `cluster/arena_geometry.py`, geometry fit/comparison/selection publications | geometry import seam; selection/review trace pending |
| `stimulus`, `stimulus_epochs`, `stimulus_response` | `analysis/import_stimulus_to_zarr.py`, `analysis/stimulus_epoch_schema.py`, `analysis_workflows/resolved_epoch_selection.py`, `analysis/stimulus_response_coordinate_authority.py`, protocol semantic selection publications | import seam; downstream indexed |
| Subject/experiment context, acquisition-batch/subject registry identity | `shared/subject_metadata.py`, `shared/experiment_setup.py`, `registry/` normalized identity/assignment snapshots; acquisition-batch contract | import seam; assignment/export propagation pending |
| `detect` | `shared/zarr/canonical_detection_manifest.py`, `shared/zarr/detection_snapshot_publication.py`, `analysis_workflows/native_canonical_detection_publication.py`, `cluster/native_detection_authority.py`, unbound detection artifacts and model evidence | seam; complete producer/ingress/activation audit pending |
| `detect_quality`, `registered_detection_gate`, `registered_detection_gate_consumption`, `refined_detect` | `shared/zarr/refined_detection_manifest.py`, `shared/zarr/refined_detection_authority_activation.py`, `shared/zarr/clipped_refined_detection_finalization.py`, family quality/gate records; legacy finalized-collection manifests | seam; shared quality/refine and consumer parity pending |
| `crop` and pixel/cache suppliers | `shared/zarr/crop_manifest.py`, `shared/zarr/crop_snapshot_publication.py`, `shared/zarr/crop_pixel_authority.py`, `shared/manifest_crop_position_authority.py`, `shared/crop_signature.py`, ROI-cache work-package evidence | clipped source seam; all supplier profiles pending |
| `keypoints`, `keypoint_quality`, `refined_keypoints` | `shared/zarr/{keypoint,keypoint_quality,refined_keypoint}_manifest.py`, matching publication modules, `shared/keypoint_coordinate_publication.py`, strict clipped/terminal receipts and v2 finalization | finalization tests; full maintained planner/consumer trace pending |
| Body-frame suppliers | `shared/zarr/body_frame_manifest.py`, `shared/zarr/body_frame_publication.py`, `shared/keypoint_motion_authority.py`, `shared/keypoint_success_authority.py` | indexed; use-scoped sufficiency must be preserved |
| `subject_masks`, `refined_subject_masks` | `shared/subject_mask_worker_receipt.py`, `shared/zarr/subject_mask_validation_receipt.py`, `shared/zarr/subject_mask_core_publication.py`, bundle/cache/quality publication and manifests, contour-worker receipts | indexed; modern dense/bundle contracts, not a generic segmentation authority |
| Coordinate successors / assignment rebinding | `shared/zarr/coordinate_successor_authority.py`, coordinate-validation receipts, keypoint/refined-mask coordinate publications and rebinding evidence | indexed; historical equivalence is not a routine new pipeline stage |
| `arena_assignment`, `tracks` | `tracking/run_manifest.py`, `shared/track_coordinate_publication.py`, source position/assignment and row/frame bindings | indexed |
| `track_kinematics`, `swim_bouts`, `bout_kinematics` | Numerical/coordinate publications, `shared/zarr_payload_receipt.py`, stage scientific manifests, runtime verification/admission and exact source selections | indexed |
| `subject_shape`, `eye_angles`, `tail_kinematics`, `tail_posture_view` | `shared/subject_shape_coordinate_publication.py`, subject-shape storage/bundle-source receipts, eye access-aware validation receipts, `shared/tail_coordinate_publication.py` | indexed |
| `bout_classification`, `detection_occupancy`, `session_occupancy` | Family analysis runs, input/lineage fingerprints, derived export publications | indexed |
| `chaser_distance`, other chaser components | `analysis/chaser_component_publication.py`, `analysis/chaser_distance_coordinate_publication.py`, `analysis_workflows/chaser_component_receipt.py`, exact projection/child/relative-frame receipts, provider-position and composable-successor publications | indexed; later consumer evidence may be stronger than the Sept-2 census |
| Behavior distributions / provider views | `analysis_workflows/recording_behavior_distribution_publication.py`, core paradigm/authority roster, validated-behavior receipts and product catalog | indexed |
| `track_kinematics_visualization`, plots/reports/review proxies | `shared/plot_artifacts.py`, `reporting/manifest.py`, `reporting/export.py`, report registry and review-proxy manifests | indexed; presentation identity and failure state separate from numerical completion |
| Analytics / group statistics / cohort freeze and release | `analytics_exports/publication.py`, `derived_publication.py`, `chaser_authority.py`, per-table/Parquet manifests, `cohorts/registry.py`, `cohorts/release.py` | indexed |
| Training data, review/compaction, model training/export/pinning | `training/training_base_publication.py`, `training_review_artifact_publication.py`, `training_review_compaction_publication.py`, `training_crop_materialization_publication.py`, `shared/zarr/training_keypoint_review_publication.py`, training exporters, `shared/artifact_fingerprint.py`, model rows/artifact manifests | indexed; clipped sampling seam reviewed, not all training publication |
| `dish_mask`, `detection_tuning`, `keypoint_tuning`, `subject_mask_tuning`, `subdish_mask_tuning` | Review/edit attrs, immutable selected geometry/review publications, source rowset/edit revisions and acceptance policies | indexed; review-state string alone is not a receipt of an exact reviewed claim |
| Deprecated `eye_masks`, `refined_eye_masks`, `eye_mask_tuning` | Legacy eye-mask run/edit/training evidence | indexed compatibility only; no new primary workflow |
| Cross-cutting execution / publication / inventory | `shared/run_provenance.py`, `shared/stage_provenance.py`, `shared/run_lineage_fingerprint.py`, `shared/rowset_fingerprint.py`, `shared/zarr/manifest_digest.py`, `shared/zarr_run_completion.py`, `shared/selector_activation.py`, `shared/recording_artifact_inventory.py`, `cluster/lsf/`, `analysis_workflows/runtime_verification.py` | selected seams; no universal receipt validation implied |

Important consolidation candidates already exist. The strict refined-detection
finalizer publishes a recording snapshot from clip evidence, checks globally
allocated `refined_row_ids`, preserves source membership, and refuses to invent
identity during merging. The strict clipped keypoint finalizer also has focused
coverage. These are stronger bases than reimplementing historical collection
flattening. A finalizer's existence and unit tests do not prove every active
planner calls it or every public consumer admits its output.

## 10. Validation performed and next audit boundary

All tests used `scripts/py -m pytest` on the workstation outside the sandbox, with
fixture data only. Python was 3.11.14, pytest 8.4.2, Zarr 3.1.3. No dependencies
were installed. Commands are relative to the audit worktree.

```bash
scripts/py -m pytest -q \
  tests/unit/fisheye/test_recording_import_receipt.py \
  tests/unit/fisheye/test_source_recording_identity_profile.py \
  tests/unit/fisheye/test_registry_recording_import_receipt_bindings.py \
  tests/unit/fisheye/test_import_recording_analysis.py \
  tests/unit/fisheye/test_import_organized_recordings_analysis.py \
  tests/unit/fisheye/test_organize_recordings_external_ipc.py \
  tests/unit/fisheye/test_organize_recordings_video_only.py \
  tests/unit/fisheye/test_create_clipped_analysis_zarr.py \
  tests/unit/fisheye/test_create_clipped_training_zarr.py \
  tests/unit/fisheye/test_validate_clipped_training_provenance.py \
  tests/unit/fisheye/test_staging_batch_disposition.py
# 125 passed in 14.22s

scripts/py -m pytest -q --color=no \
  tests/unit/fisheye/test_registry_shadow_publish.py \
  tests/unit/fisheye/test_recording_identity_authority.py \
  tests/unit/fisheye/test_run_citrus_session_import.py \
  tests/unit/fisheye/test_acquisition_frame_clock.py \
  tests/unit/fisheye/test_acquisition_publication_status.py \
  tests/unit/fisheye/test_build_recording_frame_index.py \
  tests/unit/fisheye/test_acquisition_crop_stream_ledger.py \
  tests/unit/fisheye/test_acquisition_crop_stream_collection_ledger.py
# 105 passed in 29.23s

scripts/py -m pytest -q --color=no \
  tests/unit/fisheye/test_source_video_metadata.py \
  tests/unit/fisheye/test_import_profile_contract.py \
  tests/unit/fisheye/test_consolidate_external_ipc_rolling_recordings.py \
  tests/unit/fisheye/test_clipped_refined_detection_finalization.py \
  tests/unit/fisheye/test_clipped_keypoint_finalization.py
# 42 passed in 7.24s

scripts/py -m pytest -q \
  tests/unit/fisheye/test_acquisition_frame_clock.py \
  tests/unit/fisheye/test_provider_recording_timing_authority.py \
  tests/unit/fisheye/test_recording_distribution_timebase_adapter.py
# Parallel clock review: 15 passed in 3.00s; five Zarr-v3 consolidation warnings
# Includes six clock tests already present in the second batch above.

scripts/py -m pytest -q --color=no \
  tests/unit/fisheye/test_stimulus_coordinate_v6_adapter.py
# Producer-handoff reconciliation: 16 passed in 0.76s
```

Total: **297 distinct existing tests passed** (303 test executions including six
reruns). This does not include the additional adversarial diagnostics in the pass
count: those exposed F2/F3/F8/F9 and remain defects, not successful rejection tests.
F4/F6 are code-path findings, not claims of new behavioral regression coverage.

Documentation checks: all relative Markdown file links resolve; the evidence JSON
contains exactly eight clipped observations; the existing contract-freshness check
reports 122 checked documents and no issues. The audit has no contract-meta claim
of implemented runtime behavior. New-file whitespace checks passed for all three
audit artifacts; pre-publication worktree inspection showed only those documents.

Unrun at the audit checkpoint: full repository suite and all required remote
checks for the documentation candidate. Publication-time CI status belongs to the
exact PR commit and must be read there, not inferred from this test census.
Also unrun: current-code fresh intake/registry production canary;
full source-byte/decoded-pixel or all-live-receipt verification; split/unsplit
scientific and temporal parity; every downstream producer/consumer audit. The
missing canary/parity evidence is not replaced by the green focused tests.

Next review boundary: finish required-source disposition and source/map/clock
admission semantics, then trace canonical detection → quality/gate → refinement
through the existing layout adapter and strict recording-level finalizers. Use the
same whole, whole-plus-derived-clips, and native-clips fixture matrix. Record
current caller adoption before retiring any historical collection path.

### Handoff

Implemented: audit documentation, read-only metadata evidence and reproducible
in-memory clock diagnostic commands only. Validated:
the focused tests and diagnostic observations above. Publication scope: authorized
documentation-only branches/PRs; their exact commits and live check status are
recorded in the PR handoffs. Integrated into main: no. Deployed/activated: nothing.
No source interfaces, numerical parameters,
persisted identities, selectors, registry rows, or historical receipt bytes changed.
This audit does not establish merge readiness. Full required CI, integration, any
fixes, new canaries, historical migration and production activation remain
distinct steps. The downstream all-stage review remains open.

When extending this document, date each observation, name its exact code/artifact
snapshot and metadata mode, preserve historical evidence, and link implementation
status back to its existing owner. Do not turn a successful audit check or receipt
count into a new authority gate.
