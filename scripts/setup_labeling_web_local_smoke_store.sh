#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORE="${1:-/tmp/palette_labeling_first_operator_test.sqlite}"

rm -f "$STORE" "$STORE-shm" "$STORE-wal"

"$ROOT/scripts/py" -m fisheye.utils.labeling_work --store "$STORE" init

"$ROOT/scripts/py" -m fisheye.utils.labeling_work --store "$STORE" assign \
  --recording-id recording-a \
  --user alice \
  --assigned-by operator \
  --notes "Alice should see only recording-a. Replace placeholder zarr paths before opening workflow UIs."

"$ROOT/scripts/py" -m fisheye.utils.labeling_work --store "$STORE" assign \
  --recording-id recording-b \
  --user bob \
  --assigned-by operator \
  --notes "Bob should see only recording-b. Replace placeholder zarr paths before opening workflow UIs."

"$ROOT/scripts/py" -m fisheye.utils.labeling_work --store "$STORE" add-task \
  --task-id task-alice-keypoints \
  --recording-id recording-a \
  --workflow-kind keypoints \
  --title "Alice keypoint placeholder" \
  --scope-json '{"zarr_path":"/replace/with/real/test.zarr"}'

"$ROOT/scripts/py" -m fisheye.utils.labeling_work --store "$STORE" add-task \
  --task-id task-bob-detect \
  --recording-id recording-b \
  --workflow-kind detect_training \
  --title "Bob detection placeholder" \
  --scope-json '{"zarr_path":"/replace/with/real/test.zarr"}'

cat <<EOF
Created local web-labeling smoke store:
  $STORE

Start Alice loopback service:
  scripts/py -m fisheye.utils.labeling_work --store "$STORE" serve --user alice --admin-user alice --host 127.0.0.1 --port 8795 --link-secret local-test-secret --access-log

Start Bob loopback service:
  scripts/py -m fisheye.utils.labeling_work --store "$STORE" serve --user bob --admin-user bob --host 127.0.0.1 --port 8795 --link-secret local-test-secret --access-log

These placeholder tasks are for dashboard/admin/session behavior. Replace
placeholder zarr paths or use the real-zarr smoke spec before testing workflow
save paths.
EOF
