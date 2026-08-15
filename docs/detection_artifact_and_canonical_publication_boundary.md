# Detection artifact and canonical publication boundary
<!-- contract-meta
version: 1
status: accepted
accepted: 2026-08-14
-->

## Decision

Production detection is one operator-facing workload with two immutable storage
roles:

```text
inference
  -> detection_artifact_runs/<work-unit-artifact>
  -> canonical assembly and validation
  -> detect_runs/<recording-level-canonical-v3-run>
  -> selector activation
  -> serial registry reconciliation
```

`detection_artifact_runs` contains detector-local evidence. It is never
selector eligible, never a registry `detect=ok` authority, and cannot be used
directly by geometry, quality, refinement, crops, pose, segmentation, or
training publication.

`detect_runs` contains only recording-level canonical detection publications.
A production child must carry a valid canonical-v3 `run_manifest`, coordinate
catalog, complete recording frame domain, stable recording-level instance
keys, exact source-frame and source-pixel authorities, immutable completion
state, and validated direct and consolidated metadata.

The number of input videos is a physical execution detail. A whole video is one
identity-mapped work unit. A one-clip recording is one indexed work unit. A
multi-clip recording is several indexed work units. All three layouts converge
to the same recording-level canonical detection contract.

## Publication transaction

1. Every inference worker writes a fresh non-selector artifact.
2. The recording finalizer revalidates every artifact, receipt, model digest,
   frame mapping, camera identity, native extent, and tree digest.
3. The finalizer binds the complete recording frame domain, deterministically
   mints recording-level `instance_key` values, and builds a canonical-v3
   candidate on node-local storage.
4. The candidate is deeply validated before archive mutation.
5. The finalizer atomically places the canonical child while it remains
   selector ineligible.
6. Direct and consolidated metadata, the coordinate catalog, logical hashes,
   source authorities, and manifest digest are revalidated from the archive.
7. Only then may the finalizer activate `latest`, `latest_complete`, and the
   run's selector-eligibility bit. Failed visibility validation restores the
   previous selectors and leaves an explicit failed, ineligible tombstone.
8. A serial reconciliation step projects completed canonical authorities into
   the registry. Parallel workers never write SQLite directly.

The intermediate artifact remains durable for audit, recovery, or canonical
reassembly. Retention is a separate policy; successful publication does not
delete it.

## Consumer rule

Every modern consumer must bind one exact `detect_runs/<run>` identity and its
canonical run-manifest payload digest. Planning or execution must fail before
writing output when the source is missing, stale, selector ineligible when
eligibility is required, not canonical-v3, lacks the coordinate catalog, has a
different frame or pixel authority, or has changed instance-key coverage.

This applies to:

- registered geometry comparison and keyed gates;
- detection quality and refinement;
- crop construction and crop completeness;
- keypoints and subject masks;
- tracking, analytics, visualization, and training/export publication.

Compatibility readers may inspect historical layouts only through an
explicitly named migration or recovery command. Compatibility fallback is not
a production selector policy.

When canonical publication occurs earlier in the same planned workflow, the
manifest digest does not exist yet. The plan therefore carries the immutable
native-publication receipt path. Quality, refinement, and finalization resolve
the digest from that receipt at execution time, verify that the receipt names
the exact expected `detect_runs/<run>` path, and then require that digest to be
the currently selected canonical-v3 authority. Plans for an already-published
source carry the digest directly. Exactly one form is required; neither an
unbound “latest” lookup nor both competing forms are accepted.

The enforcement boundary is transitive for downstream image products. Crop
construction directly validates and records the selected detection manifest
digest. Modern pose and subject-mask workflows consume that canonical crop or
the geometry-gated refined detection lineage; they do not independently reopen
detector artifacts. Direct raw-detection analytics, including chaser-distance
publication, call the same active canonical-source validator and bind the
manifest digest in their input authority record.

## Existing GoodBatBadBat evidence

The 84 analysis Zarrs contain immutable historical flat runs at:

```text
detect_runs/detect_goodbatbadbat_raw_detection_20260813_v1
```

Those runs must not be rewritten, deleted, or used as modern authority. Their
already-materialized canonical-v3 successors preserve the exact source
detections and instance keys. Geometry, quality, refinement, and later stages
must bind the successors.

## Implementation checklist

### Producer boundary

- [x] Whole-video inference writes only `detection_artifact_runs`.
- [x] Clipped native inference writes only `detection_artifact_runs`.
- [x] Whole-video identity mapping is explicit and independently validated.
- [x] Clipped local-to-recording frame mapping is explicit and independently
      validated.
- [x] The production native candidate defaults to canonical-v3 with a
      coordinate catalog.
- [x] The canonical finalizer is mandatory in whole-video, one-clip, and
      multi-clip production recipes.
- [x] Production recipes expose no path that can report detection completion
      from an artifact or historical flat run.

### Publication and registry

- [x] Canonical placement remains ineligible until deep post-copy validation.
- [x] Selector activation is rollback-safe and validates fresh consolidated
      visibility.
- [x] Publication receipts record the exact canonical manifest digest.
- [x] Registry extraction marks `detect=ok` only for an eligible, selected,
      canonical-v3 manifest-bearing run.
- [x] Registry reconciliation is serialized after parallel publications.

### Consumers

- [x] Registered detection gates require a canonical manifest and bind its
      payload digest.
- [x] Recording-level detection quality validates canonical manifests.
- [x] Shared planner inputs carry either the expected canonical manifest digest
      or the exact immutable publication receipt that resolves it.
- [x] Refinement verifies the same run ID and digest selected by quality.
- [x] Crop and direct analytics use the shared canonical-source validator;
      pose, mask, and export plans inherit the bound canonical crop/refined
      lineage and cannot reopen artifact or flat detection layouts.
- [x] Geometry-review discovery and approval expose only eligible canonical-v3
      sources.

### Compatibility and regression safety

- [x] Historical flat-to-canonical successor publication remains an explicit
      compatibility command.
- [x] The older direct `run_detection_local_publish` adapter is removed from
      production recipes and retained only as a named compatibility surface.
- [x] Existing immutable artifacts and canonical successors remain unchanged.
- [x] Empty-detection recordings retain all `F+1` frame offsets and a valid
      canonical manifest.
- [x] Focused tests cover whole-video identity, one-clip indexed, multi-clip,
      artifact rejection, canonical-v3 enforcement, selector rollback, stale
      manifest digests, and registry fail-closed behavior.

The historical clipped `off`-geometry tail and its detect-quality recovery
tooling remain explicit compatibility surfaces and may consume per-clip legacy
sources. They cannot claim active canonical authority, become production
selectors, enter the geometry-review queue, or satisfy the modern production
acceptance criteria.

## Acceptance criteria

- A new production inference run creates no flat-layout child under
  `detect_runs`.
- Every selector-eligible `detect_runs` child is a deeply valid canonical-v3
  publication.
- Artifact-only and historical flat runs cannot become `detect=ok`.
- Geometry, quality, refinement, crops, pose, masks, and exported analytics
  cannot bypass the exact selected canonical detection authority.
- Whole-video, one-clip, and multi-clip inputs produce the same public logical
  schema and coordinate contract.
- Existing GoodBatBadBat raw detections and canonical successors remain
  byte-for-byte unchanged.
