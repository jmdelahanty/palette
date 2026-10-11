# Citrus unified H5 bounding_boxes v2 — frozen reference fixtures

Copied byte for byte from agent-contracts
`citrus-unified-h5-core-bounding-boxes-v2/fixtures/` (PR 73). They come from
Citrus at PR 8. Synthetic writer output, all finalized complete; never
regenerate or edit them (the logger records real clock values).
`tests/unit/fisheye/test_unified_h5_admission_v3.py` checks each digest.

| File | Contents | sha256 |
| --- | --- | --- |
| `bounding_boxes_v2_pose_on.h5` | v2 table, pose declared, no capacity preflight (catalog selection only) | `116ebfd97790c0cfdff0c161e300ae44dbfc946cf7a61f4b6dc679b6dea4db48` |
| `bounding_boxes_v2_pose_off.h5` | v2 table, no pose, no capacity preflight | `6920631211ea0f38bdcd0edd0f2c7edd3b82a6ec3491e06e545048628ca32722` |
| `bounding_boxes_v2_capacity_pose_on.h5` | bound synthetic pair, capacity preflight declaring `b2af7764…`, pose declared | `f7b1d6a663e630e4397544540b3e521e7bcd4b1f31975d57d49284f30a43bb0a` |
| `bounding_boxes_v2_capacity_pose_off.h5` | bound synthetic pair, capacity preflight declaring `b2af7764…`, no pose | `13a67cbffc3a54c9d5e661ca58dd80dd77a543aa24ab852605b8f7fc21aea663` |
