# Crop → Keypoint → Tracking Contract Split: Audit and Decided Direction

**Date:** 2026-08-24
**Method:** four parallel read-only audit agents (producer contracts, consumer contracts, DAG/orchestration wiring, test coverage), synthesized. No files were modified during the audit.
**Repo state at audit:** branch `agent/palette/clipped-geometry-acquisition-authority-20260821`, HEAD `a2859cb0`.
**Status:** direction DECIDED (2026-08-24, see §8). The previously in-flight attr-stamped `crop_manifest_position_authority` bridge is **abandoned** — it would have been a fifth authority grammar and required republishing existing artifacts.

---

## 0. Verdict

The observed failure — a complete recording-level geometry-only crop v2 publication rejected by
`load_persisted_source_camera_position_surface` — is a genuine **split-canonical-authority** defect,
not a misconfiguration. The repo contains two incompatible authority grammars:

- **Old grammar:** mutable attr-stamped records on the rowset, revalidated against live arrays at load.
- **New grammar:** immutable digest-sealed `run_manifest` covering payload *and* metadata.

No artifact can satisfy both, by construction: the ordinary loaders require
`coordinate_contract == "canonical_v2"` (`src/fisheye/shared/keypoint_coordinate_publication.py:877`)
while the sealed-crop contract requires that same attribute to be **absent**
(`src/fisheye/shared/zarr/historical_geometry_only_crop_adapter.py:1015`).

The failure was *loud* (fail-closed refusal), not silent coordinate corruption — the gates did their
job. The defect is architectural: contracts were added without disposing of their predecessors.

---

## 1. The distinct definitions of "canonical" (four crop contracts + one position contract)

### A. Geometry-only crop v2 — the current producer

- Entry points: `publish_crop_geometry_from_explicit_refined_candidate`
  (`src/fisheye/shared/zarr/crop_snapshot_publication.py:95`) and
  `publish_crop_geometry_production_candidate` (`:482`), both via
  `publish_selector_ineligible_crop_geometry_snapshot` (`src/fisheye/shared/zarr/crop_shadow.py:489`).
- Group attrs (`crop_shadow.py:549–564`, `crop_snapshot_publication.py:324–360`):
  `status:"complete"`, `stage_selector_eligible: False`, `artifact_class:"geometry_only_analysis"`,
  `immutable_snapshot: True`, `production_candidate: True`,
  `production_selector_activation:"deferred"`, refined-source and gate bindings.
- Manifest: `run_manifest` attr (`CROP_RUN_MANIFEST_ATTRIBUTE`, `crop_manifest.py:55`), schema
  `palette.crop_geometry.run_manifest` v2 (`:54`), fields incl. `publication`, `logical_schema`,
  `storage_plan`, `source_refined_snapshot`, `source_pixel_authority`, `row_signature`,
  `logical_content` (per-array sha256, algorithm `sha256_c_contiguous_bytes_v1`, `:59`),
  `metadata_declarations_digest` (`:62–64`), and the v2-only `coordinate_contract` catalog envelope
  (`:705–708`).
- Lineage: `source_refined_snapshot` → **refined_detect** run identified by digests
  (`CropRefinedSourceIdentity`, `crop_manifest.py:116–180`); `source_pixel_authority`
  (`CropPixelAuthority`, `:257–309`, `frame_index_domain:"zero_based_acquisition_camera_frame"`);
  explicit-origin authority `palette.crop_geometry.explicit_origin_authority`
  (`crop_snapshot_publication.py:252–268`, `placement_mode: VERIFIED_EXPLICIT_PER_ROW`).
- Arrays: exactly `CROP_GEOMETRY_SCHEMA_V1.binding_paths` (13 arrays, `crop_shadow.py:226–264`);
  **no `roi_images`, no `crop_storage_mode` attr, no `coordinate_contract` group attr**.

### B. Materialized canonical_v2 — what the ordinary loaders enforce

- Written by `src/fisheye/tracking/crop.py:1052` and `tracking/incremental_crop.py:980,1398`:
  `coordinate_contract:"canonical_v2"`, `crop_storage_mode:"materialized"`, mandatory `roi_images`,
  `crop_geometry_selection` binding a **live, selector-eligible** `detect_runs` rowset.
- Enforced at: `keypoint_coordinate_publication.py:877–889`,
  `observation_coordinate_publication.py:4610–4620`.
- In-code policy declares B the future: `utils/crop_batch.py:298–304`
  ("future-canonical ordinary crop requires crop_storage_mode=materialized; geometry_only crop
  creation is an isolated legacy workflow") and `tracking/crop.py:3992–3993`. **By B's own labels,
  the current v2 producer's output is "legacy" on the day it is published.**

### C. Signed hybrid crop provider (pixel origin)

- `src/fisheye/shared/hybrid_crop_provider.py`, schema
  `palette.hybrid_acquisition_offline_crop_run.v3` (`:25`); `crop_storage_mode:"geometry_only"`
  stamped by `utils/build_hybrid_acquisition_offline_crop_run.py:1847`. Consumed by the v2 producer
  as origin authority (`crop_snapshot_publication.py:162–274`).

### D. Legacy collection-proxy crops

- `src/fisheye/shared/crop_snapshot_identity.py:70–80`: `stage:"crop_proxy"`,
  `proxy_crop_complete`, `crop_storage_mode:"geometry_only"`, selector-ineligible.

### Position authority — the consumer's definition

`load_persisted_source_camera_position_surface`
(`src/fisheye/shared/observation_coordinate_publication.py:4731–4759`) requires the rowset's own
zarr attrs to carry **exactly one** of three legacy lineage records:

- `detection_acquisition_frame_mapping` (`:159`)
- `crop_geometry_selection` (`:197`)
- `collection_proxy_coordinate_successor_mapping` (`:202–204`)

**This is the exact failing predicate** (`:4739–4746`): a v2 publication carries zero of the three
(its lineage is inside `run_manifest`), so `sum(...) == 0` fails before any array is opened. Then,
independently, the common gate `_require_complete_canonical_observation_rowset` (`:2677–2710`) would
also reject it on `coordinate_contract != "canonical_v2"` (`:2693`) and
`stage_selector_eligible is not True` (`:2708`, identity comparison; absent also fails).

Consumers with additional seals downstream of the loader:
`track_kinematics.load_canonical_offline_position_source` (`analysis/track_kinematics.py:1567–1661`;
content-sha equality between surface records and live arrays at `:1586–1634`),
`_require_stage_source_surface` (`:10662–10699`), the post-publication resolver (`:11305–11370`),
and motion-manifest v2 `position_lineage_mode` (`:6102–6215`, exactly two modes).

---

## 2. `stage_selector_eligible`: three live semantics

| Semantics | Meaning of `False`/absent | Representative sites |
|---|---|---|
| Discovery-only | skip candidate; **missing marker = eligible** | `shared/zarr_run_completion.py:427–443`; `reporting/discovery.py:364,402`; eye_angle/swim_bout/megabouts loaders |
| Consumption gate, must be `True` | **forbidden from exact-path scientific consumption** | `observation_coordinate_publication.py:2705–2710` (defaults `True` at `:4292,:4448,:4581,:4726`); chaser family; `visualization/interactive_track_kinematics.py:249` |
| Consumption gate, must be `False` | candidate/staging contract; `True` = corruption | adapter `:1013`; `tracking/incremental_crop.py:1248`; `crop_snapshot_publication.py:424–454` (`_require_unselected`); tail_kinematics/subject_shape/stimulus staging families |

Producer intent is unambiguous — v2 runs stamp `selection_contract:"none_shadow_direct_path_only"`
(`crop_shadow.py:539`) and `selector_activation:"deferred_separate_reviewed_change"`
(`crop_snapshot_publication.py:868`): *consume by exact path, don't discover*. But the manifest
freezes `publication.stage_selector_eligible: false` inside the digested payload
(`crop_manifest.py:685`), so "deferred activation" is unimplementable — the flag can never legally
become `True`. Producer says "deferred"; the digests say "never"; the must-be-`True` family says
"then never consumable."

Note also: `finalize_keypoint_shards.py:751` stamps merged keypoint runs
`stage_selector_eligible: True` even when their coordinate parent crop is `False`.

---

## 3. The "historical" adapter is a misnomer — it is the live production bridge

- Timeline (git): v2 producer landed **2026-07-28/29** (`c4cdd3be`, `d2975aa7`); clipped CLI
  2026-07-30/08-05; explicit-origin binding 2026-08-15 (`32307bff`). The adapter was created
  **2026-08-17** (`b1107dde` "fix: support sealed geometry-only coordinate successors") — *after*
  the producer, as the fix that makes producer output consumable.
- Every artifact the current CLIs emit (`utils/publish_clipped_crop_geometry_v2.py`,
  `utils/publish_crop_geometry_candidate.py`, `..._batch.py`) can reach keypoint/mask successor
  publication **only** through the adapter.
- Mechanism: `historical_geometry_only_crop_loader` (`adapter:1217–1314`) **monkey-patches module
  globals under an RLock** — replaces `keypoint_coordinate_publication.load_persisted_keypoint_crop_source`
  and `subject_mask_coordinate_publication.load_persisted_subject_mask_crop_source` (`:1278–1279`)
  plus three `CROP_PLACEMENT_*` attr-name constants (`:1280–1303`).
- Why tracking is excluded: `track_kinematics.py:99` **from-imports**
  `load_persisted_source_camera_position_surface`, freezing the reference at import time. The
  adapter is structurally unable to serve tracking even if extended. Its docstring (`:9–12`) frames
  the ongoing production path as a closed historical exception.
- `shared/pixel_frame_authority.py:174–175` hardcodes the adapter as the only legal producer of
  padded placement ownership (`CROP_PLACEMENT_PADDED_PRODUCER`).

---

## 4. Contradiction matrix (each a hard mutual exclusion)

| # | Old-grammar requirement | v2 artifact reality | Why unfixable in place |
|---|---|---|---|
| 1 | `coordinate_contract == "canonical_v2"` (obs `:2693`, kp `:877`) | attr must be **absent** (adapter `:1015`); contract lives in manifest payload | adding the attr breaks `metadata_declarations_digest` |
| 2 | `stage_selector_eligible is True` (obs `:2708`) | frozen `False` in digested payload; downstream requires it **stay** `False` (adapter `:1013`, `incremental_crop.py:1248`) | flipping = corruption per producer's own `_require_unselected` |
| 3 | one of three live attr-stamped lineage records (obs `:4739–4746`) | lineage = `source_refined_snapshot` (refined_detect, digest-identified) inside manifest | loader can't dispatch on the family; adding an attr breaks the digest |
| 4 | crop branch requires array `detection_indices` (obs `:4544`) | schema has `source_refined_row_ids` (`crop_schema.py:319`) | adding an array breaks adapter topology check (`:1021–1026`) and `logical_content` |
| 5 | `crop_storage_mode == "materialized"` + `roi_images` (obs `:4610–4620`) | geometry-only, no pixels stored | pixels live with the signed hybrid provider by design |

**Immutability vs consumability:** `metadata_declarations_digest` covers the entire attribute set
with only `run_manifest` redacted (`crop_manifest.py:435–524`). Any retrofit attribute
self-invalidates the publication (`crop_shadow.py:477–486`). The sanctioned successor bridge
(`publish_collection_proxy_successor_mapping`, obs `:3628–3646`) admits only old merged-proxy-v1
sources (`historical_collection_proxy_v1.py:52–64`) and cannot bridge v2.

---

## 5. DAG wiring: deliberate half-finished migration, documented

- **Clipped batch** (`cluster/clipped_inference.py`): keypoints AND masks bind to the merged proxy
  crop (`--target-crop-run <merged_proxy>`, `:1754–1766` and `:1879–1880`). The crop-v2 chain is
  built only in legacy gate-off mode (`:823–826, 1374`) and gates only the ROI cache
  (`:1465–1471`) — **never the coordinate parent of any keypoint/mask/tracking output**.
- **Whole-recording**: keypoint refinement always binds `cache.crop_run` — the hybrid pixel
  provider (`cluster/keypoints/common.py`, `build_refinement_job:1136–1249`, never passes
  `--terminal-crop-run`); in the finalizer, `terminal_crop_run or crop_run`
  (`utils/finalize_whole_recording_keypoint_v2.py:342`) collapses the distinction. Masks, by
  contrast, split roles explicitly (`cluster/subject_masks/recording.py:104–105`) and publication
  hard-requires crop-v2 (`publish_recording_bundle.py:679, 712–731`).
- **Deliberate**: commit `51f5b7b3` (2026-08-15, "keypoints: separate geometry and pixel crop
  authorities") landed the rebase mechanism + tests but changed no planner; `33f79603` (2026-08-16)
  wired the mask half end-to-end. The production checklist
  (`docs/goodbatbadbat_acquisition_crop_stream_production_checklist_20260815.md`) says verbatim:
  "completed keypoint provider consumption remains future work."
- **The rebase exists and is correct**: `_require_terminal_crop_provider_compatible`
  (`finalize_whole_recording_keypoint_v2.py:188–320`) — mode
  `signed_hybrid_pixels_with_strict_crop_v2_geometry`, exact `np.array_equal` over six ordered
  geometry arrays, provider fingerprints, matching `source_refined_run_id`. Coordinate-authority-
  first (pixel identity insufficient) — the right philosophy. Exercised only in unit tests.
- `build_clipped_storage_keypoint_chain_fragments`
  (`cluster/clipped_storage_finalization.py:260–298`) — the clipped crop-v2→keypoint binding —
  has **zero callers** outside the crimson benchmark (`crimson_storage_candidate.py:167–176`).

Existing mode-specific bypasses to be aware of (not to extend): `legacy_allow_missing` mask policy
(`publish_recording_bundle.py:712`), `geometry_crop_run or cache.crop_run` fallbacks
(`whole_recording_analysis.py:179`, `subject_masks/recording.py:104`), missing-eligibility-marker =
eligible (`zarr_run_completion.py:441–443`), single-clip finalization with no `--target-crop-run`
(`finalize_keypoint_shards.py:369–374`).

---

## 6. Test coverage: why both suites are green over a broken boundary

- Producing set (`test_crop_snapshot_publication.py`, `test_crop_manifest.py`,
  `test_crop_snapshot_workflow.py`, batch/fixture tests) and consuming set are **fully disjoint**
  (verified by cross-grep). Writer tests never import the tracking loader.
- `test_track_kinematics_coordinate_contract.py` **monkeypatches
  `load_persisted_source_camera_position_surface` itself in 12 places** (`:173,:211,:720,:801,
  :912,:980,:1036,:1090,:1164,:1213` + variants), feeding FakeGroup fixtures built by the OLD
  `publish_crop_observation_geometry`. `test_track_motion_publication.py` does the same
  (`:280,:338`) and hand-stamps `CROP_GEOMETRY_SELECTION_ATTR`.
- Adapter tests (`test_coordinate_successor_publication.py:826–1166`) monkeypatch
  `validate_crop_run_manifest → lambda: ()` over a hand-forged manifest (`"a"*64` digests,
  fixture `:532`, patch `:687`) — the adapter never sees real producer output in tests.
- `test_clipped_keypoint_finalization.py:369` no-ops `validate_crop_geometry_shadow_publication`.
- The only genuine publish→load round trips use the OLD writer:
  `test_detect_yolo_sharding.py:858` (real unpatched loader, tamper-reject leg at `:1048`) and
  `test_benchmark_track_kinematics_v2_candidate.py` (`_build_canonical_sealed_source:54`,
  full track load at `:262`). **These are the templates to clone.**
- All of the above runs in CI (16-shard matrix over `tests/`, `.github/workflows/ci.yml:267–273`;
  none marked `gpu`). CI green is fully compatible with the broken boundary.

Missing tests: (a) v2 publication → unpatched tracking loader (would fail today); (b) clipped
keypoint finalization → tracking; (c) adapter parity (anything provable for keypoints/masks is
provable for tracking); (d) behavioral `stage_selector_eligible=False` under direct-path
consumption (the only current guard is signature inspection,
`test_observation_coordinate_publication.py:1021`).

---

## 7. Design assessment (summary)

Local engineering is strong (digest-sealed manifests, fail-closed gates, coordinate-authority-first
rebase); the system-level failure is that **contracts are added without disposing of predecessors**,
and mutual strictness among disagreeing contracts produces deadlock, bridged by a monkey-patch.
Same disease as prior review waves ("capture outruns enforcement"; "validation thorough, evidence
evaporates") — depth perpendicular to the data flow, nothing along it. Additional hazard for the
multi-agent workflow: **the code lies about itself** ("historical" adapter for the live path,
"isolated legacy workflow" labels on the invested direction, "deferred" activation frozen
impossible) — agents read and believe these labels.

Process rules adopted going forward:

1. **No new authority contract without a disposition for its predecessor.** Valid dispositions:
   migrate, tombstone with reject-with-pointer, or declared coexistence — where coexistence means
   BOTH profiles are first-class in the shared resolver with their own boundary tests (§8).
   "Both coexist, adapter TBD" is not a disposition; it is the state that produced this.
2. **One real-writer→unpatched-reader boundary test per producer→consumer pair, in CI.**
   Monkeypatching the boundary function in reader tests is a review-blocking smell.
3. **Per-contract eligibility semantics**: consumption gates read the stated
   `selection_contract` where present; retire ambient three-way `stage_selector_eligible` reads.
4. **Resolvers, not adapters**: consumers ask one authority resolver per kind; the resolver
   dispatches on the artifact's declared grammar with full-strength validation per grammar.

---

## 8. Decided direction (2026-08-24)

**Abandoned:** the attr-stamped `crop_manifest_position_authority` bridge (would be a fifth grammar,
require republishing existing artifacts, and preserve the split).

**Adopted:** the shared position resolver recognizes the existing sealed crop-v2 manifest
**directly**, as a fourth mutually-exclusive lineage branch in
`load_persisted_source_camera_position_surface`:

- Dispatch on `run_manifest` presence + `artifact_class == "geometry_only_analysis"`. The
  `sum(...) != 1` exclusivity check stays (now over four).
- Validate at full strength via `open_persisted_crop_geometry_publication` (complete digest-sealed
  manifest validation — equal or greater strength than the attr-stamped path).
- Build the ordinary sealed position surface from `centers_img_xy` /
  `source_acquisition_frame_index` / `instance_key`; temporal authority bound to the live
  acquisition camera authority AND cross-checked against the manifest's `source_pixel_authority`.
- Eligibility checked at the **v2 polarity** for this branch only: exactly `False`, with
  `selection_contract:"none_shadow_direct_path_only"` present.
- **No artifact mutation, no republication.** Existing sealed artifacts become consumable as
  published — the decisive advantage over any successor/bridge design.
- Preserve: the clipped-DAG keypoint rebase (wire `--terminal-crop-run` / the crop-v2 binding into
  the planners) and the real-publisher→unpatched-reader regression test.
- Follow-up (same effort, separable PRs): keypoint/mask loaders grow the same manifest branch and
  the adapter's RLock global-swap is retired (its validation core *is* the branch implementation);
  already-published successors stay valid.
- Precedent in-repo for manifest-bound authority in a loader:
  `shared/subject_position_detection_source.py:211,499` (manifest v3, detect_runs).

**Explicit non-goals:** a tracking-specific adapter; an `allow_geometry_only=True` flag on existing
branches; any mode-specific bypass. Each would be a fifth definition.

**Policy decision (Jeremy, 2026-08-24): coexistence, not succession.** Materialized crops
(profile B) and geometry-only sealed crops (profile A) are BOTH supported publication profiles,
covering different workflows. Neither replaces the other. The contract statement is therefore:
**multiple publication profiles, one position-authority consumption interface** — which is exactly
what the resolver design implements (one branch per profile, full-strength validation per grammar).

What "supported profile" means (the forcing function that keeps coexistence from re-becoming
divergence):

- A supported profile is reachable through the shared resolver with a full-strength validation
  branch — no adapter, no monkey-patch, no bypass flag. A profile that needs an adapter to be
  consumed is not supported; it is abandoned-with-a-crutch.
- Each supported profile carries its own real-writer→unpatched-reader boundary test in CI.
- Adding profile N+1 costs exactly one resolver branch + one round-trip test. That cost is the
  admission fee, paid up front, never deferred to an adapter.
- Directional labels are banned in both directions: "future-canonical", "legacy",
  "historical", "isolated workflow" language comes out of comments/gates
  (`crop_batch.py:298–304`, `tracking/crop.py:3992–3993`, the adapter's framing) and is replaced
  with neutral profile names (the schema already has `artifact_profile`).

Profile roster under this policy (to confirm during implementation):

- **A. geometry_only sealed (crop-v2)** — supported; gains its resolver branch in this wave.
- **B. materialized canonical_v2** — supported; already has its resolver branch (the
  `crop_geometry_selection` path); comments claiming exclusivity get neutralized.
- **C. signed hybrid provider** — not a crop publication profile; it is a pixel-origin *input*
  authority consumed by A. Keep as-is, label accordingly.
- **D. collection-proxy crops** — legacy: keep readable (proxy-successor bridge stands), tombstone
  for new publication.

---

## 9. Implementation tripwires (for the implementing agent)

1. **Do not reuse `_require_complete_canonical_observation_rowset` for the fourth branch** — it
   demands `coordinate_contract=="canonical_v2"` (`:2693`) and eligibility `True` (`:2708`), both
   wrong for v2. Build the branch gate on `open_persisted_crop_geometry_publication` at full
   strength: manifest digests incl. `metadata_declarations_digest`, completion attrs, exact
   13-array topology, v2 eligibility polarity. Anything less is a weakened gate.
2. **Digest-algorithm parity is load-bearing.** `load_canonical_offline_position_source`
   (`track_kinematics.py:1586–1634`) requires content-sha equality between surface records and
   live arrays. The manifest uses `sha256_c_contiguous_bytes_v1`. Verify byte-identity with how
   the old path computes `content_sha256` **before** building on the manifest digests, or the
   branch passes the loader and fails the seal.
3. **The loader is the first of three consumption grammars.** Motion-manifest v2
   `position_lineage_mode` (`track_kinematics.py:6102–6215`) admits exactly two modes and
   hard-fails otherwise; `_require_stage_source_surface` (`:10662`) requires field-for-field
   equality incl. `coordinate_descriptor_sha256`, `row_identity_ref`. Publishing tracks from the
   new lineage needs a third manifest mode and a surface exposing all those fields. This is the
   second half of the feature, not an edge case.
4. **Temporal authority construction:** bind to live `load_persisted_acquisition_camera_authority`
   AND cross-check camera identity / `n_frames` / frame-index domain against the manifest's
   `source_pixel_authority`. Trust neither alone.
5. **Zero-row behavior is undefined for v2.** The detection branch has
   `empty_observation_declaration` (`obs:3428–3482`); v2 has no equivalent. Define it before it
   happens in production.
6. **Dispatch cleanliness:** no old-grammar rowset carries `run_manifest`, so dispatch is clean —
   but assert the negative: a rowset carrying both `run_manifest` and any of the three legacy
   attrs must FAIL, not pick a winner.
7. **Tests:** land the real-publisher→unpatched-reader round trip (clone
   `_build_canonical_sealed_source` with `publish_crop_geometry_production_candidate`) in the SAME
   PR as the branch; un-stub at least one of the 12 monkeypatched track_kinematics tests; add the
   adapter-parity assertion (c in §6).
8. **The keypoint rebase only becomes real when a planner passes `--terminal-crop-run`** — the
   finalizer's `terminal_crop_run or crop_run` default silently collapses it otherwise
   (`finalize_whole_recording_keypoint_v2.py:342`).
9. **Relabeling wave** (cheap, do it): per the §8 coexistence policy, remove directional labels in
   BOTH directions — the adapter's "historical" framing AND the "future-canonical requires
   materialized" / "isolated legacy workflow" comments — replacing them with neutral profile
   names. Neither profile is the past or the future; both are supported.

---

## 10. Affected workflows today (pre-fix)

- A single clipped recording currently carries **three coordinate lineages**: masks on crop-v2
  (contract A), keypoints on the merged proxy (D) or hybrid provider (C), tracking consumable only
  via B or a proxy-successor of D.
- Tracking cannot consume any artifact the current v2 CLIs emit; keypoint/mask successors can, but
  only through the RLock monkey-patch adapter.
- `stage_selector_eligible=False` artifacts are simultaneously "direct-path consumable" (producer
  intent) and "forbidden from exact-path consumption" (obs family) depending on which consumer
  opens them.
