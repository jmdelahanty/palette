# Canonical Detection Storage Benchmark Fixture — 2026-07-24

Status: published immutable benchmark input; noncanonical and ineligible for
registry, selector, analysis, or training use.

## Published Identity

Fixture ID:
`sleepyfish_cam2010095_detect_20260724_v1`

Cluster-visible root:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
  canonical_detection_storage/fixtures/
  sleepyfish_cam2010095_detect_20260724_v1/
```

The copied Zarr is the `source.zarr` child. Its sibling
`fixture_manifest.json` is the publication and validation record.

The publisher ran from commit
`f8995c0001de4c3e2949824bf04481efea1c96cd` in the locked deployment:

```text
/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/
  shared-zarr-storage-policy-20260723-f8995c00
```

The deploy helper verified through `login1-citrus-poller` that `scripts/py`
imported `fisheye` from this exact worktree rather than the primary checkout's
editable install.

## Source Lineage

The publication source was the previously verified disposable local copy:

```text
/tmp/palette-zarr-benchmarks/
  sleepyfish_cam2010095_detection_20260724/sources/
  detect_2026-05-14_15-39-11
```

That copy was derived read-only from the historical Sleepyfish detection run
and was already marked `canonical=false`, `registry_registered=false`, and
`selector_eligible=false`. The fixture publisher refuses a source manifest
unless all three declarations are explicitly false and its purpose identifies
benchmark use.

## Exact Copy Evidence

| Property | Source | Published copy |
| --- | ---: | ---: |
| Zarr format | 3 | 3 |
| Files | 5,809 | 5,809 |
| Apparent bytes | 8,317,265 | 8,317,265 |
| Tree SHA-256 | `7dbe2bf7b5517990609024923ddede439f61614d582918beeab56ce49c81657d` | same |

The digest schema is
`palette.tree_sha256.path_size_content.v1`: sorted relative paths, file sizes,
and complete file contents. The older local-copy manifest recorded
`de59cb09...` under an unstated digest procedure; the values are not compared
across schemes. Equality for this publication uses the named v1 scheme on both
trees.

The copy was written to an exclusive temporary sibling, inventoried, compared,
manifested, mode-frozen, and renamed to the final destination. The final
manifest records `exact_relative_path_size_content_match=true`.

## Safety And Immutability

- The fixture root and every child directory are mode `0555`; every file is
  mode `0444`.
- Benchmark jobs must open the fixture read-only and verify the recorded digest
  before and after staging. Mode freezing prevents accidental writes but is not
  a defense against the owning account deliberately changing permissions.
- Fixture destinations must be fresh and below a `fixtures` namespace under an
  explicitly supplied benchmark root.
- Candidate destinations remain exclusive-create benchmark paths. None of the
  implemented benchmark commands accepts a recording archive, registry,
  selector, or training path as a writable destination.

After publication, Citrus independently verified the exact Palette commit,
fixture flags, recorded digest, 5,809-file count, and all file/directory modes.
No canonical archive, registry row, selector, or training artifact changed.

## Next Use

An LSF benchmark block may stage this frozen `source.zarr` to node-local
scratch. All canonical conversion and candidate computation must use that local
copy. PRFS is read again only after a candidate is published back to a fresh
benchmark workflow destination for the explicit shared-storage read workloads.
