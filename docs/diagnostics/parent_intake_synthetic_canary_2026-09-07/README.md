# Parent-intake synthetic execution evidence

This archive preserves the full-stimulus canary executed on clean Palette
`a5a6ff6dd83df3f49a8f1ed7108ea6846ef3a298`. It is evidence linked from
`INGEST-001`, not a new status authority, production receipt, or reusable
admission credential. The original producing commit passed all 24 required CI
checks. Future packaging commits require their own CI.

- [Positive execution](evidence/full_stimulus_positive.json): two parent
  recordings, two stimulus runs/four steps, exact frame/event/geometry and
  calibration readback, all 41 staged files preserved before retirement,
  empty staging, unchanged acquisition originals, byte-stable replay.
- [Malformed-H5 refusal](evidence/full_stimulus_negative.json): transport-valid
  synthetic delivery fails stimulus import; failed runs remain ineligible,
  staging remains intact, registry bytes remain unchanged.
- [Positive producer](evidence/positive_producer.json) and
  [negative producer](evidence/negative_producer.json): exact Citrus source pins,
  measured H264 counts, snapshot IDs, synthetic declarations and limitations.
- [Positive SQLite acceptance](evidence/positive_registry_integrity.json) and
  [negative SQLite acceptance](evidence/negative_registry_integrity.json): full
  integrity and foreign-key checks through Palette Python SQLite 3.52.0.

The JSON files are **byte-for-byte copies**, including original absolute paths,
timestamps, hashes and producer identities. Those paths describe where the test
ran; they are not current locators or permission to mutate those directories.
Original local payloads remain untouched. Zarr/media payloads are not committed
here; this is a report archive, not a promise that `/tmp` payloads survive local
cleanup. Generate a fresh fixture to repeat the canary.

`evidence/historical_source/*.py.txt` preserves the exact original external
generator, H5 builder and verifier source as non-executable historical evidence.
The maintained executable versions are in
[`tests/canaries/parent_intake`](../../../tests/canaries/parent_intake/README.md).
Do not execute or edit historical source copies as an alternative implementation.

The negative producer predates the added builder/base-fixture hash fields;
those absent fields remain absent. Its earlier generator was recovered from the
known four-line instrumentation delta and matches the original report's exact
`59fc75b6...` source hash (`historical_source/negative_fixture_generator.py.txt`).
The final positive builder hash is not retroactively asserted for that earlier
negative fixture. Fresh packaged executions record their own evidence.

Verify archived bytes from this directory with:

```bash
sha256sum --check SHA256SUMS
```

The checksum list is an ordinary archive inventory, not a scientific receipt.
The portable regression also pins the two execution-report hashes and their
original producing commit. New executions get new reports and preserve their
actual identities; old reports are never relabeled to a packaging commit.

The [acquisition readiness request](../acquisition_parent_intake_canary_request_2026-09-07.md)
is a draft only. No message, acquisition, transfer, deployment or production
activation is authorized by this archive.
