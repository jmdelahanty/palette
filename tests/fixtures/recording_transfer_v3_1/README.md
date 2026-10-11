# Sealer 3.1.0 bound-session fixtures — fake media only

The five `bound_*` deliveries made by the Citrus sealer
`citrus-recording-transfer` 3.1.0. That is JohnsonLabJanelia/pancake-plate, tag
`recording-transfer-v3.1.0` = `456d7fe947f47aaad797c8747638592155567732`, with
envelope schema sha256 `4865374c…0f5ee6a0`. They are copied verbatim from the
output of the tag's own generator:

    python3 tests/recording_transfer_fixture.py --output <new dir>

That run writes 145 files, and its sorted `sha256sum` listing hashes to
`92e7d906…`. The `whole`, `rolling` and `failed_optional_proof` bundles are
unchanged from 3.0.0 and already live in `../recording_transfer_v2/`. The 101
files here hash to
`fd8cc6def49b13f781103571254a18707885d69a3c58a603f229f0039738c608` using
`find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`, run in this
directory, excluding this README.

The MP4 and H5 files are labelled text placeholders, as `fixture_notice.txt`
says. **Never submit these to staging or treat them as media evidence.**

| Bundle | Receipts | Snapshot / marker | `citrus_artifacts` |
| --- | --- | --- | --- |
| `bound_v1_no_video` | v1 | v2 / v3 | none (3.0.0 form) |
| `bound_v2_video` | v2 | v3 / v4 | 8 (video, finalization, timing, diagnostic × 2) |
| `bound_v2_no_video` | v2 | v3 / v4 | 4 |
| `bound_v2_no_citrus_artifacts` | v2 | v3 / v4 | 0 |
| `bound_v1_upgraded_to_v2_pre310_diagnostic` | v1 → v2 (r2) | v3 / v4 | 7, revision chain, pre-3.1.0 diagnostic name |

Do not edit these files. To change them, regenerate them from a new sealer tag
and update `ENVELOPE_SCHEMA_SHA256` together with them.
