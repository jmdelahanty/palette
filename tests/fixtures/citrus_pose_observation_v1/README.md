# Citrus pose observations v1 — frozen reference fixtures

Copied byte for byte from agent-contracts `citrus-pose-observation-v1/fixtures/`
(merged as `49142e6`, PR #72; Citrus companion JohnsonLabJanelia/citrus#7,
merged as `6c5bd74`). They come from Citrus `headless_h5_core_writer_fixture
--case <pose_intake|pose_intake_failed|pose_intake_absent> --experimental-core`.
They hold synthetic observations, not recordings.

**Frozen bytes.** The logger records real clock values, so regenerating them
gives the same structure but different bytes. Never regenerate or edit them.
`tests/unit/fisheye/test_unified_h5_pose.py` checks each digest.

| File | Pose | File status | sha256 |
| --- | --- | --- | --- |
| `pose_complete.h5` | declared, complete (4 updates, 3 objects, 9 keypoints) | complete | `4856d50e71c9197435f7470b3ef2d63bd69e84abffd6bd1529b08b77d2f67a25` |
| `pose_failed.h5` | declared, failed (`pose_keypoint_count_exceeds_v1_limit`) | FAILED; must be refused | `9183ffbe254b33c707d3659b769417d3cfc76299d2361af224b369398dd62e75` |
| `pose_absent.h5` | not declared | complete | `8abb6b52fcec95c7b9db603e231c52125f7348c0eadbb2a986808bad1fd5caf0` |

None is a bound recording observation session (no Orange collection or
receipt). Transfer admission is covered by `../recording_transfer_v3_1/`.
