"""Compatibility: the transfer-v2 import behind ``run_citrus_session_import``.

The import itself is :func:`fisheye.intake.import_delivery`, which takes the
delivery's workflow lock before any side effect, imports the parents
in-process through the recording import owner, and never writes the
registry. This module only translates the historical command line (session
directory, ``--resume-transfer-plan``, ``--status-json``) into that call and
keeps the dry-run plan report. Registration happens on the writer host
(``fisheye.intake register-delivery``); job-mode registration is retired.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid

from fisheye.intake.delivery import CANONICAL_REGISTRY, plan_recording_only
from fisheye.intake.importing import STATUS_SCHEMA, import_delivery
from fisheye.intake.outcomes import EXIT_DONE, IntakeRefused, exit_code_for
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    TRANSFER_PARENT_LAYOUTS,
    require,
    strict_json,
)


def _utc_timestamp_for_path() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _plan_for_report(args) -> dict:
    """The plan a dry run reports, built or validated without any write."""

    from fisheye.utils.organize_transfer_recordings import (
        _validate_plan,
        build_transfer_organization_plan,
    )

    source = args.session_dir.absolute()
    if args.resume_transfer_plan is None:
        plan = build_transfer_organization_plan(source, destination_root=args.dest_root)
    else:
        plan = strict_json(args.resume_transfer_plan)
        _validate_plan(plan, live_source=False)
        require(plan["source_dir"] == str(source.resolve()), "resume plan names another staging source")
        require(
            plan["destination_root"] == str(args.dest_root.resolve()),
            "resume plan names another destination root",
        )
    require(
        plan["recording_layout"] in TRANSFER_PARENT_LAYOUTS,
        "parent workflow supports rolling_clips or single_video only",
    )
    plan_recording_only(plan)
    return plan


def _snapshot_sha(args) -> str:
    """The delivery's identity: from the saved plan, else the live marker."""

    from fisheye.intake.discovery import MarkerRefusal, check_marker

    if args.resume_transfer_plan is not None:
        return str(strict_json(args.resume_transfer_plan)["snapshot_id"])
    try:
        marker = check_marker(args.session_dir.absolute() / MARKER_NAME)
    except MarkerRefusal as exc:
        raise IntakeRefused(f"not a sealed transfer-v2 delivery: {exc}") from exc
    if marker is None:
        raise IntakeRefused("legacy v1 transfer marker; transfer-v2 intake only")
    return str(marker["snapshot_id"])


def run_transfer_parent_workflow(args) -> int:
    """Invoke only through run_citrus_session_import's explicit v2 dispatch."""

    try:
        if not args.apply:
            payload = {
                "schema_id": STATUS_SCHEMA,
                "status": "planned",
                "import_complete": False,
                "staging_finalized": False,
                "plan": _plan_for_report(args),
                "apply": False,
                "registry": None,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return EXIT_DONE
        source = args.session_dir.absolute()
        run_dir = args.run_dir or (
            source.parent
            / ".processing_logs"
            / f"citrus_transfer_v2_{_utc_timestamp_for_path()}_{uuid.uuid4().hex[:8]}"
        )
        result = import_delivery(
            _snapshot_sha(args),
            run_dir,
            args.resume_transfer_plan,
            destination_root=args.dest_root,
            session_dir=source,
            status_path=args.status_json,
        )
        print(json.dumps(result.to_json(), indent=2, sort_keys=True))
        return EXIT_DONE
    except Exception as exc:
        print(f"Transfer parent intake failed: {exc}", file=sys.stderr)
        return exit_code_for(exc)


__all__ = ["CANONICAL_REGISTRY", "run_transfer_parent_workflow"]
