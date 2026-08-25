"""Dry-run-first H5 subject/setup authority backfill for analysis Zarrs.

Targets come from the registry rather than a filesystem walk.  Only active
``source_recording`` analysis datasets are eligible, and source metadata is
accepted only from the recording's single ``raw/*.h5`` file.  The command
never infers subject identity from detections, tracks, filenames, or counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import zarr

from fisheye.registry.db import Registry, RegistryPaths, SQLITE_BUSY_TIMEOUT_MS
from fisheye.registry.prune_stale_datasets import create_backup
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.experiment_setup import (
    MissingExperimentSetupError,
    build_experiment_setup_record,
    experiment_setup_sha256,
    publish_experiment_setup,
    resolve_experiment_setup,
)
from fisheye.shared.subject_metadata import (
    MissingSubjectMetadataError,
    build_subject_metadata_record,
    publish_subject_metadata,
    read_h5_subject_metadata,
    resolve_subject_metadata,
    subject_metadata_sha256,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    load_source_recording_identity_profile,
)


REPORT_SCHEMA_ID = "palette.subject_experiment_setup_backfill.v1"


@dataclass(frozen=True)
class BackfillTarget:
    dataset_id: str
    recording_id: str
    zarr_path: Path
    registry_recording_path: Path | None
    artifact_kind: str
    zarr_use: str
    dataset_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "recording_id": self.recording_id,
            "zarr_path": str(self.zarr_path),
            "registry_recording_path": (
                str(self.registry_recording_path)
                if self.registry_recording_path is not None
                else None
            ),
            "artifact_kind": self.artifact_kind,
            "zarr_use": self.zarr_use,
            "dataset_status": self.dataset_status,
        }


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    conn = sqlite3.connect(
        f"file:{quote(str(resolved), safe='/')}?mode=ro",
        uri=True,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA query_only = ON;")
    return conn


def select_backfill_targets(
    registry_path: Path,
    *,
    recording_ids: Sequence[str] = (),
    path_contains: Sequence[str] = (),
    status: str = "active",
    limit: int | None = None,
    all_recordings: bool = False,
) -> list[BackfillTarget]:
    """Select canonical recording-owned analysis archives without registry writes."""

    if not all_recordings and not recording_ids and not path_contains:
        raise ValueError(
            "Provide --recording-id/--path-contains or pass --all-recordings."
        )
    sql = [
        "SELECT d.dataset_id, d.recording_id, d.zarr_path, d.artifact_kind,",
        "       d.zarr_use, d.status AS dataset_status,",
        "       r.recording_path AS registry_recording_path",
        "FROM datasets AS d",
        "LEFT JOIN recordings AS r ON r.recording_id = d.recording_id",
        "WHERE d.status = ? COLLATE NOCASE",
        "  AND d.zarr_use = 'analysis' COLLATE NOCASE",
        "  AND d.artifact_kind = 'source_recording' COLLATE NOCASE",
        "  AND NULLIF(TRIM(d.recording_id), '') IS NOT NULL",
    ]
    params: list[Any] = [str(status)]
    if recording_ids:
        placeholders = ", ".join("?" for _ in recording_ids)
        sql.append(f"AND d.recording_id IN ({placeholders})")
        params.extend(str(value) for value in recording_ids)
    if path_contains:
        predicates = " OR ".join("LOWER(d.zarr_path) LIKE ?" for _ in path_contains)
        sql.append(f"AND ({predicates})")
        params.extend(f"%{str(value).casefold()}%" for value in path_contains)
    sql.append("ORDER BY d.recording_id, d.dataset_id")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        sql.append("LIMIT ?")
        params.append(limit)

    with closing(_connect_read_only(registry_path)) as conn:
        rows = conn.execute("\n".join(sql), params).fetchall()

    targets = [
        BackfillTarget(
            dataset_id=str(row["dataset_id"]),
            recording_id=str(row["recording_id"]),
            zarr_path=Path(str(row["zarr_path"])).expanduser().resolve(strict=False),
            registry_recording_path=(
                Path(str(row["registry_recording_path"]))
                .expanduser()
                .resolve(strict=False)
                if row["registry_recording_path"] not in (None, "")
                else None
            ),
            artifact_kind=str(row["artifact_kind"]),
            zarr_use=str(row["zarr_use"]),
            dataset_status=str(row["dataset_status"]),
        )
        for row in rows
    ]
    paths: dict[Path, str] = {}
    for target in targets:
        prior = paths.setdefault(target.zarr_path, target.dataset_id)
        if prior != target.dataset_id:
            raise ValueError(
                "Multiple active registry datasets select the same analysis Zarr: "
                f"{prior!r}, {target.dataset_id!r} -> {target.zarr_path}"
            )
    return targets


def _open_root(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _recording_dir(target: BackfillTarget) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    physical = (
        target.zarr_path.parent.parent
        if target.zarr_path.parent.name == "zarr"
        else None
    )
    registered = target.registry_recording_path
    if physical is not None and registered is not None and physical != registered:
        warnings.append(
            "registry_recording_path_differs_from_zarr_parent:"
            f"{registered} != {physical}"
        )
    return (physical or registered), warnings


def _base_row(target: BackfillTarget) -> dict[str, Any]:
    return {
        **target.as_dict(),
        "disposition": "blocked",
        "action": "none",
        "reason": None,
        "detail": None,
        "warnings": [],
    }


def _finish(
    row: dict[str, Any],
    *,
    disposition: str,
    action: str,
    reason: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    row.update(
        {
            "disposition": disposition,
            "action": action,
            "reason": reason,
            "detail": detail,
        }
    )
    return row


def plan_target(target: BackfillTarget) -> dict[str, Any]:
    """Build one mutation-free target plan and validate any existing authority."""

    row = _base_row(target)
    if not target.zarr_path.is_dir():
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="zarr_missing",
        )
    if not (target.zarr_path / "zarr.json").is_file():
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="zarr_v3_metadata_missing",
        )
    if (
        load_source_recording_identity_profile(target.zarr_path)
        == SOURCE_RECORDING_IDENTITY_PROFILE
    ):
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="current_source_profile_unsupported",
        )

    recording_dir, warnings = _recording_dir(target)
    row["warnings"] = warnings
    row["recording_dir"] = str(recording_dir) if recording_dir is not None else None
    if recording_dir is None:
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="recording_dir_unresolved",
        )
    h5_paths = sorted(path for path in (recording_dir / "raw").glob("*.h5") if path.is_file())
    row["h5_candidates"] = [str(path) for path in h5_paths]
    if not h5_paths:
        return _finish(
            row,
            disposition="skipped",
            action="none",
            reason="source_h5_missing",
        )
    if len(h5_paths) != 1:
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="source_h5_ambiguous",
            detail=f"found {len(h5_paths)} raw H5 files",
        )
    h5_path = h5_paths[0]
    row["source_h5_path"] = str(h5_path)
    try:
        metadata = read_h5_subject_metadata(h5_path)
        subject_record = build_subject_metadata_record(metadata)
        subject_ids = [str(value) for value in subject_record["subject_ids"]]
        if not subject_ids:
            return _finish(
                row,
                disposition="skipped",
                action="none",
                reason="explicit_subject_identity_missing",
            )
        subject_digest = subject_metadata_sha256(subject_record)
        subject_run = f"subject_metadata_{subject_digest[:16]}"
        subject_ref = f"analysis/subject_metadata_runs/{subject_run}"
        setup_record = build_experiment_setup_record(
            metadata,
            source_h5_path=h5_path,
            subject_metadata_sha256=subject_digest,
            subject_metadata_ref=subject_ref,
        )
        setup_digest = experiment_setup_sha256(setup_record)
        setup_run = f"experiment_setup_{setup_digest[:16]}"
    except Exception as exc:
        return _finish(
            row,
            disposition="skipped",
            action="none",
            reason="source_subject_metadata_incomplete_or_invalid",
            detail=f"{type(exc).__name__}: {exc}",
        )

    row["desired"] = {
        "subject_metadata_run": subject_run,
        "subject_metadata_ref": subject_ref,
        "subject_metadata_sha256": subject_digest,
        "subject_ids": subject_ids,
        "subject_identity_kind": subject_record["subject_identity_kind"],
        "expected_subject_count": setup_record["expected_subject_count"],
        "assigned_subject_count": setup_record["assigned_subject_count"],
        "experiment_setup_run": setup_run,
        "experiment_setup_sha256": setup_digest,
    }

    try:
        root = _open_root(target.zarr_path, mode="r")
        try:
            existing_subject = resolve_subject_metadata(root, allow_legacy=False)
        except MissingSubjectMetadataError:
            existing_subject = None
        try:
            existing_setup = resolve_experiment_setup(root, allow_legacy=False)
        except MissingExperimentSetupError:
            existing_setup = None
    except Exception as exc:
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="existing_authority_invalid_or_unreadable",
            detail=f"{type(exc).__name__}: {exc}",
        )

    row["existing"] = {
        "subject_metadata_run": (
            existing_subject.run_name if existing_subject is not None else None
        ),
        "subject_metadata_sha256": (
            existing_subject.record_sha256 if existing_subject is not None else None
        ),
        "experiment_setup_run": (
            existing_setup.run_name if existing_setup is not None else None
        ),
        "experiment_setup_sha256": (
            existing_setup.record_sha256 if existing_setup is not None else None
        ),
    }
    if existing_subject is not None and existing_subject.record_sha256 != subject_digest:
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="subject_metadata_conflicts_with_h5",
        )
    if existing_setup is not None and existing_setup.record_sha256 != setup_digest:
        return _finish(
            row,
            disposition="blocked",
            action="none",
            reason="experiment_setup_conflicts_with_h5",
        )
    if existing_subject is not None and existing_setup is not None:
        return _finish(
            row,
            disposition="eligible",
            action="verify_existing",
        )
    return _finish(
        row,
        disposition="eligible",
        action="publish",
    )


def build_backfill_plan(targets: Sequence[BackfillTarget]) -> dict[str, Any]:
    rows = [plan_target(target) for target in targets]
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "created_utc": utc_now(),
        "mode": "dry_run",
        "dataset_count": len(rows),
        "recording_count": len({row["recording_id"] for row in rows}),
        "disposition_counts": dict(sorted(Counter(row["disposition"] for row in rows).items())),
        "action_counts": dict(sorted(Counter(row["action"] for row in rows).items())),
        "reason_counts": dict(
            sorted(Counter(row["reason"] for row in rows if row["reason"]).items())
        ),
        "datasets": rows,
    }


def _validate_published(root: zarr.Group, desired: Mapping[str, Any]) -> None:
    subject = resolve_subject_metadata(root, allow_legacy=False)
    setup = resolve_experiment_setup(root, allow_legacy=False)
    if subject.record_sha256 != desired["subject_metadata_sha256"]:
        raise ValueError("Published subject-metadata digest does not match the plan")
    if tuple(subject.subject_ids) != tuple(desired["subject_ids"]):
        raise ValueError("Published subject IDs do not match the plan")
    if setup.record_sha256 != desired["experiment_setup_sha256"]:
        raise ValueError("Published experiment-setup digest does not match the plan")
    if setup.expected_subject_count != desired["expected_subject_count"]:
        raise ValueError("Published expected subject count does not match the plan")


def apply_backfill_plan(
    registry_path: Path,
    plan: Mapping[str, Any],
    *,
    refresh_registry: bool = True,
) -> dict[str, Any]:
    """Apply eligible rows; failures are isolated and reported for safe reruns."""

    registry = Registry(registry_path) if refresh_registry else None
    applied_rows: list[dict[str, Any]] = []
    try:
        for planned in plan.get("datasets", []):
            row = dict(planned)
            if row.get("disposition") != "eligible":
                applied_rows.append(row)
                continue
            zarr_path = Path(str(row["zarr_path"]))
            h5_path = Path(str(row["source_h5_path"]))
            desired = row.get("desired")
            if not isinstance(desired, Mapping):
                row.update(
                    disposition="error",
                    action="none",
                    reason="desired_authority_plan_missing",
                )
                applied_rows.append(row)
                continue
            try:
                # Re-plan immediately before mutation so a changed H5, selector, or
                # archive cannot invalidate an older dry-run decision.
                fresh = plan_target(
                    BackfillTarget(
                        dataset_id=str(row["dataset_id"]),
                        recording_id=str(row["recording_id"]),
                        zarr_path=zarr_path,
                        registry_recording_path=(
                            Path(str(row["registry_recording_path"]))
                            if row.get("registry_recording_path")
                            else None
                        ),
                        artifact_kind=str(row["artifact_kind"]),
                        zarr_use=str(row["zarr_use"]),
                        dataset_status=str(row["dataset_status"]),
                    )
                )
                if fresh.get("disposition") != "eligible":
                    raise ValueError(
                        "Apply-time preflight changed: "
                        f"{fresh.get('reason') or fresh.get('disposition')}"
                    )
                if fresh.get("desired") != desired:
                    raise ValueError("Apply-time authority plan differs from dry run")
                if (
                    load_source_recording_identity_profile(zarr_path)
                    == SOURCE_RECORDING_IDENTITY_PROFILE
                ):
                    raise ValueError(
                        "historical subject/setup backfill does not mutate "
                        "current-profile source recordings"
                    )

                root = _open_root(zarr_path, mode="r+")
                metadata = read_h5_subject_metadata(h5_path)
                subject = publish_subject_metadata(
                    root,
                    metadata,
                    source_h5_path=h5_path,
                )
                setup_record = build_experiment_setup_record(
                    metadata,
                    source_h5_path=h5_path,
                    subject_metadata_sha256=subject.record_sha256,
                    subject_metadata_ref=subject.group_path,
                )
                publish_experiment_setup(
                    root,
                    setup_record,
                    source_h5_path=h5_path,
                )
                _validate_published(root, desired)
                refreshed_dataset_id = None
                if registry is not None:
                    refreshed_dataset_id = registry.scan_zarr(zarr_path)
                    if refreshed_dataset_id != row["dataset_id"]:
                        raise ValueError(
                            "Registry refresh changed dataset identity: "
                            f"planned={row['dataset_id']!r}, "
                            f"refreshed={refreshed_dataset_id!r}"
                        )
                row.update(
                    disposition="applied",
                    action=(
                        "verified_existing"
                        if fresh.get("action") == "verify_existing"
                        else "published"
                    ),
                    reason=None,
                    detail=None,
                    registry_refreshed=registry is not None,
                    refreshed_dataset_id=refreshed_dataset_id,
                )
            except Exception as exc:
                row.update(
                    disposition="error",
                    action="none",
                    reason="apply_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            applied_rows.append(row)
    finally:
        if registry is not None:
            registry.close()

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "created_utc": utc_now(),
        "mode": "apply",
        "dataset_count": len(applied_rows),
        "recording_count": len({row["recording_id"] for row in applied_rows}),
        "disposition_counts": dict(
            sorted(Counter(row["disposition"] for row in applied_rows).items())
        ),
        "action_counts": dict(
            sorted(Counter(row["action"] for row in applied_rows).items())
        ),
        "reason_counts": dict(
            sorted(Counter(row["reason"] for row in applied_rows if row["reason"]).items())
        ),
        "datasets": applied_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or apply canonical subject-metadata and experiment-setup runs "
            "from acquisition H5 metadata."
        )
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument(
        "--path-contains",
        "--cohort-contains",
        dest="path_contains",
        action="append",
        default=[],
    )
    parser.add_argument("--status", default="active")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-registry-refresh", action="store_true")
    parser.add_argument(
        "--backup",
        type=Path,
        help="Required for an apply that refreshes the registry.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    targets = select_backfill_targets(
        registry_path,
        recording_ids=args.recording_id,
        path_contains=args.path_contains,
        status=str(args.status),
        limit=args.limit,
        all_recordings=bool(args.all_recordings),
    )
    report = build_backfill_plan(targets)
    if args.apply:
        refresh_registry = not bool(args.no_registry_refresh)
        if refresh_registry:
            if args.backup is None:
                raise ValueError("--apply with registry refresh requires --backup")
            if args.backup.exists():
                raise FileExistsError(f"Backup already exists: {args.backup}")
            create_backup(registry_path, args.backup)
        report = apply_backfill_plan(
            registry_path,
            report,
            refresh_registry=refresh_registry,
        )
        if args.backup is not None:
            report["registry_backup"] = str(
                args.backup.expanduser().resolve(strict=False)
            )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.apply and report.get("disposition_counts", {}).get("error", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackfillTarget",
    "REPORT_SCHEMA_ID",
    "apply_backfill_plan",
    "build_backfill_plan",
    "plan_target",
    "select_backfill_targets",
]
