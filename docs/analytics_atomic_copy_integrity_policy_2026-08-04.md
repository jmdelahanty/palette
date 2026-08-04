# Analytics Atomic Copy-Integrity Policy

Date: 2026-08-04

Status: implemented as the fail-safe atomic-publisher default; no selector,
registry authority, physical profile, or canonical run was changed.

## Decision

Every maintained atomic run-group publication now verifies physical content by
default. `AtomicRunPublishSpec.content_checksum` defaults to `True` under
policy `content_checksum_required_v1`.

The exact backend behavior is:

- Python copy: hash every regular source and copied target file with SHA-256,
  require identical relative paths, sizes, inventory digest, and content
  digest, and record verification `sha256_all_physical_files`.
- rsync copy: hash the source inventory, copy with archive semantics, then run
  `rsync --dry-run --checksum --delete --itemize-changes`; any reported change
  fails publication. The receipt records `rsync_checksum_dry_run`.

The check occurs while the destination remains hidden and before atomic rename,
completion, selector mutation, or registry visibility. The mechanical receipt
is runtime/publication evidence and does not alter scientific array identity.

Historical compatibility callers may explicitly set `content_checksum=False`.
That opt-out retains relative-path/size inventory comparison and is not the
maintained default. A caller that needs the weaker mode must therefore make the
compatibility decision visible in code and in the physical-copy receipt.

Sealed chaser-component requests use the same safe default. This closes the
previous split where eye, tail, track, subject-shape, stimulus, and chaser-base
publishers opted into checksums while swim bouts, bout kinematics, exact-
tabular candidates, bout classification, and component publications silently
inherited path/size-only verification.

## Validation

The focused real-Zarr gate passed 60/60 across:

- the atomic publisher and rollback paths;
- sealed chaser components;
- computed swim bouts;
- both bout-kinematics publication modes;
- exact-tabular candidates; and
- bout-classification v2 candidate publication and fresh-process evidence.

The tests assert the persisted physical-copy receipt reports
`sha256_all_physical_files` with a nonempty content digest for Python-backed
publication. Static compilation, Ruff, and `git diff --check` also pass.

## Benchmark Boundary

Writer timings collected before this policy remain valid descriptions of the
historical code revision, but they are not current checksum-inclusive promotion
evidence for families that previously inherited `content_checksum=False`.
Those writer/publication phases must be rerun under the new default before a
physical profile can be promoted. Read-only matrices and decoded-equality
receipts are unaffected.
