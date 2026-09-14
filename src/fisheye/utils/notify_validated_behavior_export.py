"""Preview or explicitly deliver one validated export availability email."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_notifications import (
    deliver_validated_behavior_export_announcement,
    prepare_validated_behavior_export_announcement,
)
from fisheye.labeling.notifications import (
    LabelingNotificationConfig,
    NOTIFICATION_MODES,
)

SENDER_ENV_VAR = "PALETTE_EXPORT_NOTIFICATION_FROM"
OUTBOX_ENV_VAR = "PALETTE_EXPORT_NOTIFICATION_OUTBOX"
DEFAULT_OUTBOX = "~/.palette/export_email_outbox"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--export-run-id", required=True)
    parser.add_argument(
        "--to",
        action="append",
        required=True,
        metavar="EMAIL",
        help="One plain email address; repeat for more recipients.",
    )
    parser.add_argument("--audience", help="Optional intended audience label.")
    parser.add_argument(
        "--location",
        help="Recipient-facing access path or URL; defaults to publication root.",
    )
    parser.add_argument(
        "--handoff", help="Recipient-accessible reading guide path or URL."
    )
    parser.add_argument("--access-note", help="One-line access instructions.")
    parser.add_argument("--note", help="One-line message from the sender.")
    parser.add_argument(
        "--mode",
        choices=NOTIFICATION_MODES,
        help="Delivery transport when --deliver is set; otherwise preview only.",
    )
    parser.add_argument(
        "--deliver",
        action="store_true",
        help="Queue to outbox or send by SMTP. Without this flag, print a preview only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    announcement = prepare_validated_behavior_export_announcement(
        publication_root=args.publication_root,
        export_run_id=args.export_run_id,
        to=args.to,
        audience=args.audience,
        location=args.location,
        handoff=args.handoff,
        access_note=args.access_note,
        note=args.note,
    )
    if not args.deliver:
        print(f"To: {announcement.notification.to_email}")
        print(f"Subject: {announcement.notification.subject}")
        print()
        print(announcement.notification.text_body)
        return 0

    config = LabelingNotificationConfig.from_env(mode=args.mode or "outbox")
    config = replace(
        config,
        sender=os.environ.get(SENDER_ENV_VAR) or config.sender,
        outbox_dir=Path(os.environ.get(OUTBOX_ENV_VAR) or DEFAULT_OUTBOX).expanduser(),
    )
    result = deliver_validated_behavior_export_announcement(announcement, config=config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"queued", "sent"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
