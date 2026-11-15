# MetaZebrobot consumer contract (vendored, byte-identical)

These files are byte-identical copies of their owners' files, pinned by SHA-256.
They are not restatements: MetaZebrobot owns the API shapes and meaning, and
agent-contracts owns which fields each consumer relies on.

| File | Source | SHA-256 |
| --- | --- | --- |
| `consumers.json` | agent-contracts `metazebrobot-consumers/` @ `ae52c9bdabd2a65c660858d8d039e0e04c75d27d` | `6d2baa4f8776eba8f9c65a0d72c2b986f316c2334e249949687e3963390c7e05` |
| `verify_consumers.py` | agent-contracts `metazebrobot-consumers/` @ `ae52c9bdabd2a65c660858d8d039e0e04c75d27d` (unchanged since PR 54) | `0e5ec76c15aef37120eee3fb160a7f3b79a1ad7e37760313996414c1f091befc` |
| `consumer_openapi.json` | metazebrobot `docs/api/consumer_openapi.json` @ `b2f4b20751fd4206ce5cc85fe996444a95043fe9` | `d80e79a68c65a04c5ec3b519648b9cf82985bde987abdce378e7a7d07e92710f` |

`tests/unit/fisheye/test_metazebrobot_consumer_contract.py` checks the digests,
checks that `fisheye.shared.zebrobot_subject_reference.MZB_PIN` names the same
pin, and runs, fully offline:

    python verify_consumers.py --spec consumers.json \
        --openapi-file consumer_openapi.json --consumer palette

To move the pin, replace all three files from their sources at the new commits,
update the table, the test constants and `MZB_PIN` together. A recording sealed by an earlier build still
matches when its slice is listed in `COMPATIBLE_CONSUMER_OPENAPI_SHA256` (only add a slice the new pin
strictly extends). For a rig-day
live check (not CI), add `--live http://delahantyj-ws1.hhmi.org`.

## Which serving builds count as matching the pin

Each recording seals the consumer slice digest of the MetaZebrobot build that
served it (`zebrobot_service.consumer_schema_sha256` in the Citrus snapshot,
`zebrobot.consumer_schema_sha256` in Orange's subject reference). Palette
records `zebrobot_consumer_schema_matches_pin` from
`COMPATIBLE_CONSUMER_OPENAPI_SHA256` in
`fisheye.shared.zebrobot_subject_reference`. The flag is provenance only: a
recording whose build does not match is still admitted.

A slice is listed as compatible only when the pinned slice strictly extends it
for everything Palette relies on (its `consumers.json` entry). At the
`b2f4b20` pin, the pin and `c93b2530` qualify (b2f4b20 adds only
`X-Delivered-By` on usage links, which Palette does not read). Earlier slices
do not (decided by Jeremy, 2026-10-15), because `GET /dishes/{dish_id}/fish`, which Palette relies on, changed meaning
twice after `509a3eb8`:

| Slice | MetaZebrobot commit | `GET /dishes/{dish_id}/fish` lists | Matches |
| --- | --- | --- | --- |
| `f5280e43…` | `509a3eb8` | fish registered to the dish | no |
| `f120734c…` | `50e578ea` | fish registered to the dish (additive: fish create + revisions) | no |
| `463def80…` | `ed39b042` | fish in the dish now (origin/location; `dish_id` = current dish) | no |
| `a8cb8884…` | `c93b2530` | living fish in the dish now (ended fish hidden) | yes |
| `d80e79a6…` | `b2f4b20` | as `c93b2530` (adds `X-Delivered-By` on usage links) | yes (pin) |

So a fish list sealed by an earlier build (Orange `dish_fish`, Palette's
`dish_fish_ids`) means something different from one sealed at the pin, and
Palette's drift check (`dish_fish_changed_since_recording`) compares it with
today's list. Every recording served by a build before `c93b2530`,
including the 2026-10-09 snow sessions (served by `509a3eb8`), records
`matches_pin: false` for that reason, not because anything is wrong with it.
The flag never gates admission. Fish identity itself (`mzb_fish_id`) is stable
across all five slices.
