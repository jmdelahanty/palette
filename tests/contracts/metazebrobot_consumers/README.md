# MetaZebrobot consumer contract (vendored, byte-identical)

These files are byte-identical copies of their owners' files, pinned by SHA-256.
They are not restatements: MetaZebrobot owns the API shapes and meaning, and
agent-contracts owns which fields each consumer relies on.

| File | Source | SHA-256 |
| --- | --- | --- |
| `consumers.json` | agent-contracts `metazebrobot-consumers/` @ `5fc735fee7a013b016ab501177339b138f8931b7` (PR 54) | `8d2b38780ad08c7111cb6f5bd81723169e5ce96cad44f489e6cf442ee071bda3` |
| `verify_consumers.py` | agent-contracts `metazebrobot-consumers/` @ `5fc735fee7a013b016ab501177339b138f8931b7` | `0e5ec76c15aef37120eee3fb160a7f3b79a1ad7e37760313996414c1f091befc` |
| `consumer_openapi.json` | metazebrobot `docs/api/consumer_openapi.json` @ `509a3eb88d6ff44fe07e7ea20d212be5eafe46b7` | `f5280e430d4b5f10c3643cb89a6187eacc45fdac7af55b2e81f5e315b7b754dc` |

`tests/unit/fisheye/test_metazebrobot_consumer_contract.py` checks the digests,
checks that `fisheye.shared.zebrobot_subject_reference.MZB_PIN` names the same
pin, and runs, fully offline:

    python verify_consumers.py --spec consumers.json \
        --openapi-file consumer_openapi.json --consumer palette

To move the pin, replace all three files from their sources at the new commits,
update the table, the test constants and `MZB_PIN` together. For a rig-day
live check (not CI), add `--live http://delahantyj-ws1.hhmi.org`.
