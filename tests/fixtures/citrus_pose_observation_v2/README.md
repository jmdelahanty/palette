# Citrus pose observations v2 — frozen reference fixtures

Copied byte for byte from agent-contracts `citrus-pose-observation-v2/fixtures/`
(merged as `6b63aa2`, PR #86; Citrus companion JohnsonLabJanelia/citrus#23,
merged as `cf7a60b`). Synthetic observations, not recordings. Never regenerate
or edit them; `tests/unit/fisheye/test_unified_h5_pose_v2.py` checks each digest.

| File | Pose | File status | sha256 |
| --- | --- | --- | --- |
| `pose_complete.h5` | declared, complete (4 updates, 3 objects, 15 keypoints); headings 0° valid, one NaN row | complete | `7ca61b9277633e677dfedbb12a05a77b8cc16d4e20ba3b7504ecbefe961cc550` |
| `pose_failed.h5` | declared, failed | FAILED; must be refused | `31199ab626fd05f68d1612190b4dcc2d93f352db11b1b55b9e836456a0084e40` |
| `pose_absent.h5` | not declared | complete | `6df3fc7115592d5f795f4fa2833981d38861fd0db0a3543659a3bedf72f3c171` |
