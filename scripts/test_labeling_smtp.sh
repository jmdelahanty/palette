#!/usr/bin/env bash
set -euo pipefail

TO_EMAIL="${1:-${PALETTE_LABELING_TEST_EMAIL_TO:-delahantyj@hhmi.org}}"
TEST_USERNAME="${PALETTE_LABELING_TEST_USERNAME:-delahantyj}"
TEST_DISPLAY_NAME="${PALETTE_LABELING_TEST_DISPLAY_NAME:-Jeremy Delahanty}"
TEST_ROLE="${PALETTE_LABELING_TEST_ROLE:-admin}"

export PALETTE_LABELING_NOTIFICATION_MODE="${PALETTE_LABELING_NOTIFICATION_MODE:-smtp}"
export PALETTE_LABELING_SMTP_HOST="${PALETTE_LABELING_SMTP_HOST:-proofpointrelayjrc.hhmi.org}"
export PALETTE_LABELING_SMTP_PORT="${PALETTE_LABELING_SMTP_PORT:-25}"
export PALETTE_LABELING_SMTP_STARTTLS="${PALETTE_LABELING_SMTP_STARTTLS:-false}"
export PALETTE_LABELING_SMTP_SSL="${PALETTE_LABELING_SMTP_SSL:-false}"
export PALETTE_LABELING_NOTIFICATION_FROM="${PALETTE_LABELING_NOTIFICATION_FROM:-Jeremy Delahanty <delahantyj@hhmi.org>}"
export PALETTE_LABELING_BASE_URL="${PALETTE_LABELING_BASE_URL:-http://localhost:8765}"

scripts/py - "$TO_EMAIL" "$TEST_USERNAME" "$TEST_DISPLAY_NAME" "$TEST_ROLE" <<'PY'
import json
import sys

from fisheye.labeling.notifications import (
    LabelingNotificationConfig,
    send_labeler_added_notification,
)

to_email, username, display_name, role = sys.argv[1:5]
config = LabelingNotificationConfig.from_env()
result = send_labeler_added_notification(
    config=config,
    user={
        "user_id": username,
        "email": to_email,
        "display_name": display_name,
        "role": role,
        "status": "active",
    },
    actor_user=username,
)
print(json.dumps(result, indent=2, sort_keys=True))
if not bool(result.get("ok")):
    raise SystemExit(1)
PY
