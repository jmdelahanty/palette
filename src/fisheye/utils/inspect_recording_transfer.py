"""Inspect Citrus transfer-v2 parent plans without import or scheduler writes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from fisheye.shared.recording_transfer_snapshot import (
    TransferSnapshotError,
    plan_parent_recordings,
    verify_transfer_snapshot,
)


def inspect_recording_transfer(root: Path) -> dict:
    transfer = verify_transfer_snapshot(root)
    parents = plan_parent_recordings(transfer)
    return {
        "status": "parent_plan_only",
        "snapshot_id": transfer.snapshot_id,
        "delivery_attempt_id": transfer.attempt_id,
        "recording_layout": transfer.recording_layout,
        "recording_payload_kind": transfer.recording_payload_kind,
        "parent_recording_count": len(parents),
        "parents": [asdict(parent) for parent in parents],
        "import_complete": False,
        "registry_admitted": False,
        "source_cleanup_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_recording_transfer(args.recording_dir)
    except (TransferSnapshotError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
