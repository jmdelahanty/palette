# MetaZebrobot consumer contract (vendored, byte-identical)

These files are byte-identical copies of their owners' files, pinned by SHA-256.
They are not restatements: MetaZebrobot owns the API shapes and meaning, and
agent-contracts owns which fields each consumer relies on.

| File | Source | SHA-256 |
| --- | --- | --- |
| `consumers.json` | agent-contracts `metazebrobot-consumers/` @ `bfdae22898e84678fd3b553ef9e3b1f1d71a43b8` (re-pin at 7cedf98) | `364c583184763966f65d61d55cc6df7b81f4fb37f39e10de729eb6ed0a43a7de` |
| `verify_consumers.py` | agent-contracts `metazebrobot-consumers/` @ `bfdae22898e84678fd3b553ef9e3b1f1d71a43b8` (unchanged since PR 54) | `0e5ec76c15aef37120eee3fb160a7f3b79a1ad7e37760313996414c1f091befc` |
| `consumer_openapi.json` | metazebrobot `docs/api/consumer_openapi.json` @ `7cedf98d1c81b2da3051f0e64eb52236de68449a` | `9b97bb72d9da50a59467f8cfa65b4d73a115a847cfd8004c13d0231d85ab790f` |

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
`7cedf98` pin, `c93b2530` and every later slice qualify: the two operations
Palette relies on (`GET /dishes/{dish_id}/citrus-snapshot` and
`GET /dishes/{dish_id}/fish`) and every schema they reference are identical in
each slice, and their server handlers are unchanged from `b2f4b20` to `7cedf98`.
Later builds add only what Palette does not read: `X-Delivered-By` on usage
links (`b2f4b20`), screening, timeline and construct endpoints (`5f3ac60`,
`c48f9a2`), usage-link `members` (`ecaa247`), and `PUT /usage-links/{recording_ref}/members` (`7cedf98`). Earlier slices
do not (decided by Jeremy, 2026-10-15), because `GET /dishes/{dish_id}/fish`, which Palette relies on, changed meaning
twice after `509a3eb8`:

| Slice | MetaZebrobot commit | `GET /dishes/{dish_id}/fish` lists | Matches |
| --- | --- | --- | --- |
| `f5280e43…` | `509a3eb8` | fish registered to the dish | no |
| `f120734c…` | `50e578ea` | fish registered to the dish (additive: fish create + revisions) | no |
| `463def80…` | `ed39b042` | fish in the dish now (origin/location; `dish_id` = current dish) | no |
| `a8cb8884…` | `c93b2530` | living fish in the dish now (ended fish hidden) | yes |
| `d80e79a6…` | `b2f4b20` | as `c93b2530` (adds `X-Delivered-By` on usage links) | yes |
| `950353ae…` | `5f3ac60` | as `c93b2530` (adds screening, timeline, constructs) | yes |
| `6d7d35ea…` | `c48f9a2` (served as `52ebae1`) | as `c93b2530` (adds mixed well plates, per-fish screening) | yes |
| `97c581df…` | `ecaa247` | as `c93b2530` (adds usage-link `members`) | yes |
| `9b97bb72…` | `7cedf98` | as `c93b2530` (adds `PUT /usage-links/{recording_ref}/members`) | yes (pin) |

So a fish list sealed by an earlier build (Orange `dish_fish`, Palette's
`dish_fish_ids`) means something different from one sealed at the pin, and
Palette's drift check (`dish_fish_changed_since_recording`) compares it with
today's list. Every recording served by a build before `c93b2530`,
including the 2026-10-09 snow sessions (served by `509a3eb8`), records
`matches_pin: false` for that reason, not because anything is wrong with it.
The flag never gates admission. Fish identity itself (`mzb_fish_id`) is stable
across all nine slices.
