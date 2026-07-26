"""Dry-run-first historical recording-geometry recovery.

This utility is intentionally narrow.  It only handles the known case where a
complete Orange geometry bundle exists but its original recording snapshot did
not contain the contract pointer.  It never edits producer artifacts and never
selects a current or newest calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.recording_geometry import RecordingGeometryError
from fisheye.shared.recording_geometry_recovery import (
    RECOVERY_RECEIPT_NAME,
    build_recording_geometry_recovery_receipt,
    publish_recording_geometry_recovery,
)


def _target_h5(recording_root: Path) -> Path:
    raw_root = recording_root / "raw"
    candidates = sorted(raw_root.glob("*.h5"))
    if len(candidates) != 1:
        raise RecordingGeometryError(
            f"Expected exactly one recording H5 under {raw_root}; found {len(candidates)}."
        )
    return candidates[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bundle",
        type=Path,
        required=True,
        help="Fixed-layout Orange geometry bundle root in staging.",
    )
    parser.add_argument(
        "--target-recording",
        action="append",
        type=Path,
        required=True,
        help="Recording directory to bind; repeat for each camera/arena.",
    )
    parser.add_argument(
        "--approved-by",
        required=True,
        help="Human/operator identity approving the historical association.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the verified bundle and atomically write the receipt. Default is dry-run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source_bundle.expanduser().resolve()
    rows: list[dict[str, object]] = []
    for raw_target in args.target_recording:
        recording = raw_target.expanduser().resolve()
        h5_path = _target_h5(recording)
        if args.apply:
            publication = publish_recording_geometry_recovery(
                source_bundle_root=source,
                recording_root=recording,
                target_h5_path=h5_path,
                approved_by=args.approved_by,
            )
            verified = publication.verified
            rows.append(
                {
                    "recording": recording.name,
                    "camera_serial": verified.evidence.camera_serial,
                    "arena_id": verified.evidence.arena_id,
                    "contract_sha256": (
                        verified.evidence.bundle_verification.contract_sha256
                    ),
                    "receipt_sha256": verified.receipt_sha256,
                    "bundle_published": publication.bundle_publication.published,
                    "receipt_published": publication.receipt_published,
                    "receipt_path": str(publication.receipt_path),
                    "status": "verified_recovered",
                }
            )
        else:
            receipt = build_recording_geometry_recovery_receipt(
                bundle_root=source,
                target_h5_path=h5_path,
                approved_by=args.approved_by,
            )
            target = receipt["target"]
            evidence = receipt["evidence"]
            assert isinstance(target, dict) and isinstance(evidence, dict)
            contract = evidence["recording_geometry_contract"]
            assert isinstance(contract, dict)
            rows.append(
                {
                    "recording": target["recording_directory_name_at_recovery"],
                    "camera_serial": target["camera_serial"],
                    "arena_id": target["arena_id"],
                    "contract_sha256": contract["sha256"],
                    "planned_receipt_path": str(
                        recording / "raw" / RECOVERY_RECEIPT_NAME
                    ),
                    "status": "dry_run_validated",
                }
            )
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "source_bundle": str(source),
                "target_count": len(rows),
                "targets": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
