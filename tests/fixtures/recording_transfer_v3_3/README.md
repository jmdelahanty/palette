# Sealer 3.3.0 receipt-v4 fixtures — fake media only

These fixtures come from the Citrus sealer `citrus-recording-transfer` 3.3.0:
JohnsonLabJanelia/pancake-plate, tag `recording-transfer-v3.3.0`, commit
`9f6816e7f5a45632cef63e7db4da461604b92cd9`. Receipt v4 is receipt v3 with
`contract.subject_reference` citrus.subject_reference v2 (subject identity v4;
agent-contracts #105, merge 83f9ed5). It changes nothing on disk, so a v4 head
seals as snapshot v3 / marker v4, exactly like v2 and v3 heads.

The tag's `--output` generator emits no v4 bundles; its v4 cases are built
inside its tests (`tests/test_recording_transfer_v4_receipts.py`).
`make_v4_fixtures.py` (copied here) calls the tag's own
`tests/recording_transfer_fixture.py` functions to write the same cases to
disk:

    python3 -I make_v4_fixtures.py <pancake-plate checkout at 9f6816e7> <new dir>

The tag's own `verify_bundle` admits all three sealed bundles, and the tag's
own `build_snapshot` refuses each source tree with the message shown below. The
sorted `sha256sum` listing of every file except this README, the generator and
the schemas hashes to
`079deeed1e09205264a27dc7d923d840a852092a94a66a470a4b5d08fc77c2d4` using
`find . -type f ! -name README.md ! -name make_v4_fixtures.py ! -name '*.schema.json' | LC_ALL=C sort | xargs sha256sum | sha256sum`,
run in this directory.

| Path | Receipts | Expected |
| --- | --- | --- |
| `bound_v4_video` | v4 × 2 | snapshot v3 / marker v4, 8 `citrus_artifacts` |
| `bound_v4_dish_group` | v4 × 2, one a dish group (`subject_id` null) | snapshot v3 / marker v4 |
| `bound_v4_chain_v4_v4` | v4 revision 1 + v4 revision 2 | snapshot v3 / marker v4, 8 `citrus_artifacts` |
| `refused_sources/mixed_v3_v4_head` | v3 + v4 | `the head collection mixes receipt versions` |
| `refused_sources/v4_with_reference_v1` | v4 | `receipt v4 subject_reference is not citrus.subject_reference v2` |
| `refused_sources/v3_with_reference_v2` | v3 | `receipt v3 subject_reference is not citrus.subject_reference v1` |
| `refused_sources/v4_without_subject_reference` | v4 | `receipt v4 keys are not exact` |
| `refused_sources/v4_extra_contract_key` | v4 | `receipt v4 keys are not exact` |
| `refused_sources/chain_v3_then_v4` | r1 v3, r2 v4 | `the collection revision chain mixes receipt v3 and v4` |
| `refused_sources/chain_v4_then_v3` | r1 v4, r2 v3 | `the collection revision chain mixes receipt v3 and v4` |
| `refused_sources/chain_v2_then_v4` | r1 v2, r2 v4 | `a receipt v4 head has an earlier revision with another receipt version …` |
| `refused_sources/chain_v1_then_v4` | r1 v1, r2 v4 | `a receipt v4 head has an earlier revision with another receipt version …` |
| `refused_sources/v4_unreadable_superseded` | r1 v4 (not JSON), r2 v4 | `superseded finalized receipt is not readable JSON: …` |

`citrus_recording_observation_finalized_receipt.v4.schema.json` (sha256
`47d2251244ed6bf18cf5e65c360da2b4a809afbdee1864a3cef75b0b44a76b83`) and
`citrus_subject_reference_v2.schema.json` (sha256
`30f42c9d49d1ba040b509d6743144aefacda63365c5ca4261524506b35931eb9`) are
byte-equal to the sealer's tests/fixtures copies and to agent-contracts #105.
Both are pinned in `tests/unit/fisheye/test_transfer_sealer_330.py`.

The MP4 and H5 files are labelled text placeholders. **Never submit these to
staging or treat them as media evidence.** Do not edit these files; regenerate
them from a new sealer tag instead.
