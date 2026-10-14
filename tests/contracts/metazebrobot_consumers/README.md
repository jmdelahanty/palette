# MetaZebrobot consumer contract (vendored, byte-identical)

These files are byte-identical copies of their owners' files, pinned by SHA-256.
They are not restatements: MetaZebrobot owns the API shapes and meaning, and
agent-contracts owns which fields each consumer relies on.

| File | Source | SHA-256 |
| --- | --- | --- |
| `consumers.json` | agent-contracts `metazebrobot-consumers/` @ `5d9b17274d392d71ca80fd546cb3cab4d58f26f0` (fish endings, #113) | `511a0aae1c931c50329ccb894cce604c94118115699d8424f2ef6ee5334d7d02` |
| `verify_consumers.py` | agent-contracts `metazebrobot-consumers/` @ `5d9b17274d392d71ca80fd546cb3cab4d58f26f0` (unchanged since PR 54) | `0e5ec76c15aef37120eee3fb160a7f3b79a1ad7e37760313996414c1f091befc` |
| `consumer_openapi.json` | metazebrobot `docs/api/consumer_openapi.json` @ `c93b2530030bd80e65a008299a26442b1f5fb3ce` | `a8cb88841430739f16958777062e1c6ed1e01c98583e6a2b0bc14e9864752d03` |

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
