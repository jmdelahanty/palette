"""One-shot geometry-review notification scanner for cron or systemd."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from fisheye.labeling.notifications import LabelingNotificationConfig
from fisheye.registry.geometry_review_notifications import (
    scan_geometry_review_notifications,
)


RECIPIENTS_ENV_VAR = "PALETTE_GEOMETRY_REVIEW_NOTIFICATION_TO"
STATE_DB_ENV_VAR = "PALETTE_GEOMETRY_REVIEW_NOTIFICATION_STATE_DB"
DEFAULT_STATE_DB = "~/.palette/geometry_review_notifications.sqlite"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read Palette registry geometry states once and send one deduplicated "
            "operator digest without modifying scientific data."
        )
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--state-db",
        type=Path,
        default=Path(os.environ.get(STATE_DB_ENV_VAR) or DEFAULT_STATE_DB).expanduser(),
        help=f"Operational deduplication SQLite (env: {STATE_DB_ENV_VAR}).",
    )
    parser.add_argument(
        "--to",
        default=os.environ.get(RECIPIENTS_ENV_VAR, ""),
        help=f"Comma-separated operator recipients (env: {RECIPIENTS_ENV_VAR}).",
    )
    parser.add_argument(
        "--mode",
        choices=("disabled", "outbox", "smtp"),
        help="Override PALETTE_LABELING_NOTIFICATION_MODE for this scan.",
    )
    parser.add_argument(
        "--base-url",
        help="Optional geometry-review app base URL included in digest links.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render transition counts without transport or dedup completion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = LabelingNotificationConfig.from_env(
        mode=args.mode,
        base_url=args.base_url,
    )
    result = scan_geometry_review_notifications(
        registry_path=args.registry,
        state_db=args.state_db,
        recipients=str(args.to or "").strip(),
        config=config,
        dry_run=bool(args.dry_run),
    )
    payload = result.to_json()
    print(json.dumps(payload, indent=2, sort_keys=True))
    status = str(result.delivery.get("status") or "")
    return 0 if status in {"queued", "sent", "skipped", "dry_run", "no_new_events"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
