"""Plan, shard, publish, or validate a generic behavior-cohort export.

Profiles select closed table contracts and source adapters.  They never expose
formula switches: every scientific value is copied from an exact bundle-bound
source under its installed table contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.analytics_exports.validated_behavior_cohort import (
    build_validated_behavior_export_plan,
    publish_validated_behavior_cohort,
    read_validated_behavior_export_manifest,
    read_validated_behavior_export_plan,
    validated_behavior_manifest_path,
    write_validated_behavior_export_plan,
    write_validated_behavior_recording_shard,
)
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_METADATA_PROFILE_ID,
)
from fisheye.analytics_exports.validated_behavior_profiles import (
    INSTALLED_VALIDATED_BEHAVIOR_PROFILES,
    ValidatedBehaviorExportProfile,
    profile_id_from_record,
    resolve_validated_behavior_profile,
)


class ValidatedBehaviorExportCliError(ValueError):
    """The requested operation is not commit-pinned or contract-exact."""


def _repository() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_state(repository: Path) -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidatedBehaviorExportCliError(
            "Cannot resolve the executing Palette Git authority."
        ) from exc
    return commit, status


def _require_clean_current_authority(
    expected: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    repository = _repository()
    commit, status = _git_state(repository)
    if status:
        raise ValidatedBehaviorExportCliError(
            "Validated-behavior export execution requires a clean Palette worktree."
        )
    if expected is not None and (
        expected.get("repository") != "palette"
        or expected.get("commit") != commit
        or expected.get("deployment_path") != str(repository)
    ):
        raise ValidatedBehaviorExportCliError(
            "Executing Palette commit/path differs from the export plan authority."
        )
    return repository, commit


def _summary(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, ensure_ascii=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write one immutable export plan")
    plan.add_argument("--membership", type=Path, required=True)
    plan.add_argument("--bundle-set", type=Path, required=True)
    plan.add_argument("--export-run-id", required=True)
    plan.add_argument("--plan-output", type=Path, required=True)
    plan.add_argument("--shard-root", type=Path, required=True)
    plan.add_argument("--publication-root", type=Path, required=True)
    plan.add_argument(
        "--profile",
        choices=tuple(INSTALLED_VALIDATED_BEHAVIOR_PROFILES),
        default=CORE_METADATA_PROFILE_ID,
        help="Closed installed table/source profile (default: core metadata only).",
    )

    shard = subparsers.add_parser("shard", help="materialize one recording shard")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--member-ordinal", type=int, required=True)

    run_shards = subparsers.add_parser(
        "run-shards", help="materialize every planned shard serially for a canary"
    )
    run_shards.add_argument("--plan", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize", help="validate all shards and commit one manifest last"
    )
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--generation-id")

    validate = subparsers.add_parser(
        "validate", help="validate one exact selected publication"
    )
    validate.add_argument("--publication-root", type=Path, required=True)
    validate.add_argument("--export-run-id", required=True)
    validate.add_argument("--full-part-hashes", action="store_true")
    return parser


def _plan_command(args: argparse.Namespace) -> dict[str, Any]:
    repository, commit = _require_clean_current_authority()
    profile = resolve_validated_behavior_profile(args.profile)
    value = build_validated_behavior_export_plan(
        membership_path=args.membership,
        bundle_set_path=args.bundle_set,
        export_run_id=args.export_run_id,
        shard_root=args.shard_root,
        publication_root=args.publication_root,
        palette_commit=commit,
        palette_repo=repository,
        table_specs=profile.table_specs,
        export_profile_id=profile.profile_id,
    )
    target = write_validated_behavior_export_plan(args.plan_output, value)
    return {
        "operation": "plan",
        "profile_id": profile.profile_id,
        "plan_path": str(target),
        "plan_sha256": value["plan_sha256"],
        "finalization_evidence_profile_id": value["evidence_profile"]["profile_id"],
        "export_run_id": value["export_run_id"],
        "member_count": value["member_count"],
        "table_names": list(value["table_names"]),
        "safety": value["safety"],
    }


def _read_plan_for_execution(
    path: Path,
) -> tuple[Mapping[str, Any], ValidatedBehaviorExportProfile]:
    profile = resolve_validated_behavior_profile(
        profile_id_from_record(path, record_kind="export plan")
    )
    plan, _membership, _bundle_set = read_validated_behavior_export_plan(
        path,
        table_specs=profile.table_specs,
        require_current_evidence=True,
    )
    _require_clean_current_authority(plan["software_authority"])
    return plan, profile


def _shard_command(args: argparse.Namespace) -> dict[str, Any]:
    plan, profile = _read_plan_for_execution(args.plan)
    receipt = write_validated_behavior_recording_shard(
        plan_path=args.plan,
        member_ordinal=args.member_ordinal,
        table_specs=profile.table_specs,
        row_extractors=profile.row_extractors(),
    )
    return {
        "operation": "shard",
        "export_run_id": plan["export_run_id"],
        "member_ordinal": args.member_ordinal,
        "recording_id": receipt["member"]["recording_id"],
        "receipt_path": receipt["receipt_path"],
        "record_sha256": receipt["record_sha256"],
        "receipt_schema_version": receipt["schema_version"],
        "semantic_validation_record_sha256": receipt["semantic_validation"][
            "record_sha256"
        ],
        "reused": receipt["reused"],
        "row_counts_by_table": {
            name: receipt["parts_by_table"][name]["row_count"]
            for name in receipt["requested_tables"]
        },
    }


def _run_shards_command(args: argparse.Namespace) -> dict[str, Any]:
    plan, profile = _read_plan_for_execution(args.plan)
    extractors = profile.row_extractors()
    created = 0
    reused = 0
    for ordinal in range(1, int(plan["member_count"]) + 1):
        receipt = write_validated_behavior_recording_shard(
            plan_path=args.plan,
            member_ordinal=ordinal,
            table_specs=profile.table_specs,
            row_extractors=extractors,
        )
        if receipt["reused"]:
            reused += 1
        else:
            created += 1
    return {
        "operation": "run-shards",
        "export_run_id": plan["export_run_id"],
        "member_count": plan["member_count"],
        "created_shards": created,
        "reused_shards": reused,
    }


def _finalize_command(args: argparse.Namespace) -> dict[str, Any]:
    plan, profile = _read_plan_for_execution(args.plan)
    manifest = publish_validated_behavior_cohort(
        plan_path=args.plan,
        table_specs=profile.table_specs,
        generation_id=args.generation_id,
    )
    return {
        "operation": "finalize",
        "export_run_id": plan["export_run_id"],
        "manifest_path": manifest["manifest_path"],
        "record_sha256": manifest["record_sha256"],
        "manifest_schema_version": manifest["schema_version"],
        "validation_receipt_record_sha256": manifest["validation_receipt"][
            "record_sha256"
        ],
        "transfer_receipt_record_sha256": manifest["transfer_receipt"]["record_sha256"],
        "row_counts_by_table": manifest["row_counts_by_table"],
        "process_telemetry": manifest["process_telemetry"],
        "safety": manifest["safety"],
    }


def _validate_command(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = validated_behavior_manifest_path(
        args.publication_root, args.export_run_id
    )
    profile = resolve_validated_behavior_profile(
        profile_id_from_record(manifest_path, record_kind="export manifest")
    )
    manifest, membership, bundle_set = read_validated_behavior_export_manifest(
        args.publication_root,
        args.export_run_id,
        table_specs=profile.table_specs,
        validate_parts="full" if args.full_part_hashes else "receipt",
    )
    return {
        "operation": "validate",
        "validation_mode": "full" if args.full_part_hashes else "receipt",
        "export_run_id": manifest["export_run_id"],
        "record_sha256": manifest["record_sha256"],
        "manifest_schema_version": manifest["schema_version"],
        "member_count": membership["member_count"],
        "bundle_state_counts": dict(bundle_set["state_counts"]),
        "row_counts_by_table": manifest["row_counts_by_table"],
        "producer_software_authority": manifest["software_authority"],
        "safety": manifest["safety"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    operations = {
        "plan": _plan_command,
        "shard": _shard_command,
        "run-shards": _run_shards_command,
        "finalize": _finalize_command,
        "validate": _validate_command,
    }
    result = operations[args.command](args)
    print(_summary(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
