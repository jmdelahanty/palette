# Sealer 3.2.0 receipt-v3 fixtures — fake media only

These fixtures come from the Citrus sealer `citrus-recording-transfer` 3.2.0:
JohnsonLabJanelia/pancake-plate, tag `recording-transfer-v3.2.0`, commit
`00787e7c36ba9fa27adba1a80b49c3996afb5426`. Its wheel's sha256 is
`02cd77b7…bf35a`. Receipt v3 is receipt v2 plus `contract.subject_reference`
(citrus.subject_reference v1). It changes nothing on disk, so a v3 head seals
as snapshot v3 / marker v4, exactly like a v2 head.

The tag's `--output` generator emits no v3 bundles; its v3 cases are built
inside its tests. `make_v3_fixtures.py` (copied here) calls the tag's own
`tests/recording_transfer_fixture.py` functions to write the same cases to
disk:

    python3 -I make_v3_fixtures.py <pancake-plate checkout at 00787e7c> <new dir>

The tag's own `verify_bundle` admits both sealed bundles, and the tag's own
`build_snapshot` refuses each source tree with the message shown below. The
sorted `sha256sum` listing of every file except this README, the generator and
the schema hashes to
`caff89c4a9ca524faf259b46261e373fa43b006cd6915209faaca9fa750ddbc3` using
`find . -type f ! -name README.md ! -name make_v3_fixtures.py ! -name '*.schema.json' | LC_ALL=C sort | xargs sha256sum | sha256sum`,
run in this directory.

| Path | Receipts | Expected |
| --- | --- | --- |
| `bound_v3_video` | v3 × 2 | snapshot v3 / marker v4, 8 `citrus_artifacts` |
| `bound_v3_null_subject_objects` | v3 × 2, one with `dish`, `fish` and `zebrobot_version` null | admitted (the sealer checks key sets only), but fails the schema's conditionals |
| `refused_sources/mixed_v2_v3` | v2 + v3 | `the head collection mixes receipt versions` |
| `refused_sources/v3_without_subject_reference` | v3 | `receipt v3 keys are not exact` |
| `refused_sources/v3_extra_contract_key` | v3 | `receipt v3 keys are not exact` |
| `refused_sources/v2_with_subject_reference` | v2 | `receipt v2 keys are not exact` |

`citrus_recording_observation_finalized_receipt.v3.schema.json` is the
receipt v3 contract schema, byte-equal to the sealer's tests/fixtures copy
and to agent-contracts PR #77. Its sha256,
`c5a55e533d3cdb914392a5d0d4bf4e8bb83cd9e9b3a81e04428de0a20adaea27`, is pinned in
`tests/unit/fisheye/test_transfer_sealer_320.py`.

The MP4 and H5 files are labelled text placeholders. **Never submit these to
staging or treat them as media evidence.** Do not edit these files; regenerate
them from a new sealer tag instead.
