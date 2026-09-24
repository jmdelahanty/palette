"""Opt-in transfer-v2 branch of the maintained Citrus session import workflow.

Organization is explicit and inventory-complete. The maintained batch importer,
actual receipt resolver and optional registry owner provide admission; this
orchestrator cannot replace those claims with a log or a transport marker.
"""

from __future__ import annotations

from dataclasses import asdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import sys
import uuid

from fisheye.shared.batch_logging import JsonLogger, make_run_id
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.recording_transfer_snapshot import require, strict_json
from fisheye.utils.organize_transfer_recordings import (
    _separate_destination,
    _state_directory,
    _validate_plan,
    build_transfer_organization_plan,
    finalize_transfer_staging,
    prepare_transfer_parent_recordings,
    transfer_parent_workflow_lock,
)
from fisheye.utils.run_citrus_session_import import (
    _newest_jsonl,
    _read_zarr_paths_from_import_log,
    _run_command,
    _utc_timestamp_for_path,
    _verify_import_acknowledgments,
    build_import_command,
)


def run_transfer_parent_workflow(args) -> int:
    """Invoke only through run_citrus_session_import's explicit v2 dispatch."""
    payload = {
        "schema_id": "palette.citrus_transfer_parent_intake.status.v1",
        "status": "failed",
        "import_complete": False,
        "staging_finalized": False,
    }
    status_path = None
    status_identity = None
    resources = ExitStack()

    def publish_status():
        nonlocal status_identity
        require(status_path is not None, "no reserved status destination")
        require(
            not any(p.is_symlink() for p in (status_path, *status_path.parents)),
            "status destination contains a symlink",
        )
        current = status_path.lstat()
        require(
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == status_identity,
            "status destination ownership lost",
        )
        write_json_atomic(status_path, payload)
        current = status_path.lstat()
        status_identity = (current.st_dev, current.st_ino)

    try:
        source = args.session_dir.absolute()
        if args.resume_transfer_plan is None:
            plan = build_transfer_organization_plan(
                source,
                destination_root=args.dest_root,
                recording_type=args.recording_type,
                recording_subtype=args.recording_subtype,
                behavior_mode=args.behavior_mode,
            )
        else:
            plan = strict_json(args.resume_transfer_plan)
            _validate_plan(plan, live_source=False)
            require(
                plan["source_dir"] == str(source.resolve()),
                "resume plan names another staging source",
            )
            require(
                plan["destination_root"] == str(args.dest_root.resolve()),
                "resume plan names another destination root",
            )
            for field in ("recording_type", "recording_subtype", "behavior_mode"):
                value = getattr(args, field)
                require(
                    value is None or value == plan["context"][field],
                    "resume cannot change recording context",
                )
        require(
            plan["recording_layout"] == "rolling_clips",
            "opt-in parent workflow currently supports rolling_clips only",
        )
        source = Path(plan["source_dir"])
        payload.update(
            plan=plan,
            apply=bool(args.apply),
            registry=str(args.registry) if args.register else None,
        )
        if not args.apply:
            payload["status"] = "planned"
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        run_dir = args.run_dir or (
            source.parent
            / ".processing_logs"
            / f"citrus_transfer_v2_{_utc_timestamp_for_path()}_{uuid.uuid4().hex[:8]}"
        )
        run_dir = _separate_destination(source.resolve(), run_dir)
        for parent in plan["parents"]:
            _separate_destination(Path(parent["destination_dir"]), run_dir)
        state_directory = _state_directory(plan)
        _separate_destination(state_directory.parent, run_dir)
        selected_status = (
            args.status_json or run_dir / "citrus_session_import.status.json"
        ).absolute()
        require(
            not any(
                path.is_symlink()
                for path in (selected_status, *selected_status.parents)
            ),
            "status path contains a symlink",
        )
        selected_status = selected_status.resolve()
        for protected in (
            source,
            state_directory.parent,
            *(Path(p["destination_dir"]) for p in plan["parents"]),
        ):
            require(
                not selected_status.is_relative_to(protected),
                "status path would mutate protected intake evidence",
            )
        require(not selected_status.exists(), "status destination already exists")
        if selected_status.is_relative_to(run_dir):
            require(
                selected_status == run_dir / "citrus_session_import.status.json",
                "custom status path overlaps workflow outputs",
            )
        registry = args.registry.resolve() if args.register else None
        if registry is not None:
            require(selected_status != registry, "status destination overlaps registry")
            for protected in (
                source,
                state_directory.parent,
                *(Path(p["destination_dir"]) for p in plan["parents"]),
            ):
                require(
                    not registry.is_relative_to(protected),
                    "registry must not be inside staging or a source recording",
                )
        run_dir.mkdir(parents=True, exist_ok=False)
        selected_status.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            selected_status, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        try:
            reserved = os.fstat(descriptor)
            status_identity = (reserved.st_dev, reserved.st_ino)
        finally:
            os.close(descriptor)
        status_path = selected_status
        plan_path = run_dir / "organization_plan.json"
        write_json_atomic(plan_path, plan, overwrite=False)
        payload.update(run_dir=str(run_dir), organization_plan_path=str(plan_path))
        verify_workflow_lock, workflow_lock_fd = resources.enter_context(
            transfer_parent_workflow_lock(plan)
        )

        state_path = state_directory / "organization_state.json"
        prior_state = strict_json(state_path) if state_path.is_file() else None
        if prior_state is not None and prior_state.get("status") in {
            "retiring",
            "complete",
        }:
            # Source control files may already be gone: use exact journal replay,
            # never reconstruct or silently re-import an immutable publication.
            require(
                args.resume_transfer_plan is not None,
                "retirement replay requires --resume-transfer-plan",
            )
            final = finalize_transfer_staging(
                plan, registry_path=registry, require_stimulus=not args.recording_only
            )
        else:
            prepare_transfer_parent_recordings(
                plan, registry_path=registry, require_stimulus=not args.recording_only
            )
            organize_log = run_dir / "organized_parents.jsonl"
            logger = JsonLogger(organize_log, make_run_id())
            try:
                for parent in plan["parents"]:
                    logger.log(
                        "recording_applied",
                        dest_dir=parent["destination_dir"],
                        recording_id=parent["identity"]["recording_id"],
                        camera_id=parent["identity"]["camera_id"],
                        source_snapshot_id=plan["snapshot_id"],
                    )
            finally:
                logger.close()
            import_log_dir = run_dir / "imports"
            import_log_dir.mkdir()
            command = build_import_command(
                organize_log=organize_log,
                log_dir=import_log_dir,
                apply=True,
                recording_only=args.recording_only,
                registry=registry,
            )
            result = _run_command(
                command,
                name="import_parents",
                run_dir=run_dir,
                pass_fds=(workflow_lock_fd,),
                env={
                    **os.environ,
                    "PALETTE_RECORDING_IMPORT_LEASE_FD": str(workflow_lock_fd),
                },
            )
            payload["commands"] = [asdict(result)]
            require(
                result.returncode == 0,
                f"parent importer exited with return code {result.returncode}",
            )
            import_log = _newest_jsonl(
                import_log_dir,
                "import_organized_recordings_analysis_*.jsonl",
                before=set(),
            )
            zarr_paths = _read_zarr_paths_from_import_log(import_log)
            _verify_import_acknowledgments(
                import_log=import_log,
                recording_dirs=[Path(p["destination_dir"]) for p in plan["parents"]],
                zarr_paths=zarr_paths,
                recording_only=args.recording_only,
            )
            payload.update(
                import_log=str(import_log),
                zarr_paths=[str(p) for p in zarr_paths],
                import_complete=True,
            )
            verify_workflow_lock()
            final = finalize_transfer_staging(
                plan, registry_path=registry, require_stimulus=not args.recording_only
            )
        payload.update(
            status="complete",
            import_complete=True,
            staging_finalized=True,
            import_receipts=final["import_receipts"],
            retired_file_count=len(final["retired_files"]),
        )
        publish_status()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload["error"] = str(exc)
        if status_path is not None:
            try:
                publish_status()
            except Exception as status_exc:
                print(f"Status not written: {status_exc}", file=sys.stderr)
        print(f"Transfer parent intake failed: {exc}", file=sys.stderr)
        return 1
    finally:
        resources.close()
