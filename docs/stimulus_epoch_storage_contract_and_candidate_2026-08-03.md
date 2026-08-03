# Stimulus-Epoch Exact Storage Contract and Candidate — 2026-08-03

Status: implemented selector-ineligible candidate contract; not a production
writer, selector, registry, or profile promotion.

## Census and authority boundary

The maintained producer is
`fisheye.analysis.stimulus_epoch_runs.write_stimulus_epoch_run`. It resolves a
versioned protocol profile against one exact stimulus run and writes
`analysis/stimulus_epoch_runs/<run>/windows`. The scientific inputs and output
dataclass are defined in `stimulus_epoch_runs.py`; the current direct writer
creates twelve arrays with hand-selected row chunk limits and identifies them
as `palette.stimulus_epoch_windows.v1`.

The v1 producer remains unchanged. It is now an explicit compatibility input,
not an exact v2 authority. It has no exact array manifest or executable storage
receipt. The candidate boundary in
`analysis_workflows/materializers/stimulus_epochs.py` accepts only one explicit,
complete, non-ineligible v1 name and validates its entire logical table before
copying it.

Current consumers are also compatibility readers:

- detection occupancy reads seven display/interval columns and casts them to
  expected NumPy dtypes;
- chaser distance and several chaser summaries reuse that interval reader or
  equivalent local readers;
- those readers do not yet require the v2 schema, exact manifest, or physical
  receipt.

This change deliberately does not modify those consumers. A future strict v2
adapter must require the exact schema and manifest; legacy v1 stays behind the
compatibility boundary.

## Exact v2 logical inventory

Schema: `palette.stimulus_epoch_windows.v2`, version `2`.

Layout: `exact_columnar_v1`.

Every array is required. There are no optional bundles or aliases.

| Path | Dtype | Shape | Units / meaning | Authority | Access |
| --- | --- | --- | --- | --- | --- |
| `windows/window_id` | `int32` | `[W]` | stable window ID | lineage index | eager |
| `windows/label_bytes` | `uint8` | `[W,96]` | NUL-padded UTF-8 label | semantic metadata | eager |
| `windows/start_frame` | `int64` | `[W]` | inclusive camera frame | scientific | eager |
| `windows/end_frame` | `int64` | `[W]` | inclusive camera frame | scientific | eager |
| `windows/start_time_s` | `float64` | `[W]` | `start_frame / fps` | scientific | eager |
| `windows/end_time_s` | `float64` | `[W]` | exclusive `(end_frame+1) / fps` | scientific | eager |
| `windows/duration_s` | `float64` | `[W]` | inclusive frame count / FPS | scientific | eager |
| `windows/source_start_event_name_bytes` | `uint8` | `[W,96]` | resolved source event | lineage | eager |
| `windows/source_end_event_name_bytes` | `uint8` | `[W,96]` | resolved source event | lineage | eager |
| `windows/source_start_event_frame` | `int64` | `[W]` | inclusive source boundary | lineage | eager |
| `windows/source_end_event_frame` | `int64` | `[W]` | exclusive source boundary | lineage | eager |
| `windows/source_policy_bytes` | `uint8` | `[W,160]` | boundary/fallback policy | semantic metadata | eager |

The payload is 508 uncompressed bytes per window across all arrays. A normal
three-window GoodCopBadCop run therefore contains only 1,524 logical payload
bytes. Eager whole-array reads are the correct access model. The published HTTP
profile produces one chunk and one payload object per array for the tested
three-window fixture; sharding provides no object-count benefit at this scale.
The physical receipt remains executable and byte-derived so an unusually large
epoch table is still planned from its actual shapes and dtypes.

## Semantic invariants

The executable validator in `analysis/stimulus_epoch_schema.py` requires:

- exactly the twelve arrays above, with exact rank, shape widths, and dtypes;
- at least one row;
- nonnegative, unique, strictly increasing `window_id` values;
- unique nonempty UTF-8 labels;
- NUL padding with no nonzero bytes after the first terminator;
- positive finite FPS and positive exact-integer `total_frames`;
- inclusive, nonempty frame intervals inside the recording;
- chronological row order and no overlapping windows;
- ordered source-event boundaries inside `[0, total_frames]`;
- finite time columns exactly derived from frame bounds and FPS within a small
  float64 arithmetic tolerance.

Each v2 candidate also owns a canonical
`stimulus_epoch_run_manifest`. Its digest-bound payload is reconstructed from
the current run and binds:

- recording identity, window count, total frames, and FPS;
- the exact source stimulus run/path and a decoded logical-tree fingerprint of
  that stimulus group;
- the exact v1 source epoch run/path, lineage hash, lineage-payload digest, and
  full logical-content digest;
- the protocol profile, adapter, role resolver, method, and window-policy
  identities;
- the newly computed v2 candidate lineage hash and payload digest;
- every decoded candidate array, the exact array-manifest and storage-plan
  digests, and the authoritative `windows` group declaration;
- exact `stage_selector_eligible=false` and
  `storage_candidate_profile_promoted=false` publication state.

The candidate never inherits the v1 `lineage_payload_json`. It constructs and
persists a new complete v2 lineage payload that binds the candidate schema,
both source identities, source fingerprints, protocol parameters, dimensions,
and materializer revision. Validators recompute both the lineage and run
manifest. Rehashing only a modified outer manifest therefore cannot make
recording, dimensions, source, policy, profile, or lineage tampering valid.

Empty labels, invalid UTF-8, aliases such as `frame_counts`, missing arrays,
wrong dtypes, overlapping intervals, duplicate IDs, and inconsistent time
columns fail closed.

## Physical and publication contract

The candidate materializer:

1. resolves a safe explicit v1 source name and a distinct immutable v2 name;
2. rejects source-run symlinks, target replacement, and source/scratch tree
   containment in either direction;
3. validates v1 logical semantics and hashes every decoded array;
4. plans actual fixed-width bytes using `published_http_v1`;
5. creates a fresh node-local Zarr v3 through the shared array factory, which
   pins the profile's bytes/Zstd codec chain;
6. writes complete non-overlapping physical units;
7. writes the exact logical manifest and executable storage receipt;
8. proves source/candidate logical hashes are identical;
9. writes and validates candidate-owned v2 lineage plus the canonical run-level
   scientific/lineage manifest;
10. consolidates local metadata and proves the complete direct/consolidated
   declaration tree is equal, including the run group, `windows` group attrs,
   and every array declaration;
11. uses the common atomic run-group publisher to copy into a hidden sibling,
    validate it, rename it atomically, and retain selector-ineligible state;
12. leaves `latest` and `latest_complete` unchanged;
13. consolidates the authoritative archive as the final visibility step and
    proves direct/consolidated equivalence;
14. on a post-consolidation failure, writes an owner-bound failed tombstone,
    reconsolidates, and requires exact failed metadata in both views.

The candidate records `storage_candidate_profile_promoted=false` and
`stage_selector_eligible=false`. It is not registry authority, production
selection evidence, or permission to change the v1 writer.

## Implementation checklist

- [x] Census the current v1 producer and permissive consumers.
- [x] Freeze exact paths, dtypes, widths, fills, null rules, and authority roles.
- [x] Freeze row-order, interval, and time-consistency semantics.
- [x] Declare v1 input-only compatibility and v2 strict-authority boundaries.
- [x] Add a digest-bound exact array manifest.
- [x] Add a deeply validated canonical run-level scientific/lineage manifest.
- [x] Replace inherited v1 lineage with candidate-owned complete v2 lineage.
- [x] Bind the actual source stimulus logical-tree fingerprint and exact source
  epoch lineage/content identities.
- [x] Replan actual bytes with the shared storage planner and factory.
- [x] Materialize on node-local scratch as Zarr v3.
- [x] Prove exact decoded equality and direct/consolidated equivalence.
- [x] Publish atomically without selector or registry changes.
- [x] Repair consolidated failed-tombstone visibility after injected failure.
- [x] Compare the complete direct/consolidated declaration tree, including
  authoritative child-group declarations.
- [x] Enforce `storage_candidate_profile_promoted` as exact JSON `false`.
- [x] Add adversarial inventory, dtype, ordering, containment, symlink, and
  failure-visibility tests, plus recomputed-digest manifest/lineage probes.
- [ ] Add a strict v2 consumer adapter; keep current v1 readers explicitly
  legacy-compatible.
- [ ] Run a real archive canary and record read/open/object-count telemetry.
- [ ] Obtain consumer review before considering writer adoption.
- [ ] If adoption is approved, make the scientific writer emit v2 directly on
  node-local scratch and retain v1 read compatibility.
- [ ] Treat profile promotion, selector activation, and registry eligibility as
  separate reviewed changes.

## Focused validation

The implementation is covered by:

- `tests/unit/fisheye/test_stimulus_epoch_schema.py`;
- `tests/unit/fisheye/test_stimulus_epoch_candidate_materializer.py`.

The suite includes a real post-consolidation injected failure and verifies that
the resulting public child is failed and selector-ineligible in both direct and
consolidated metadata while the prior parent pointers remain unchanged.
