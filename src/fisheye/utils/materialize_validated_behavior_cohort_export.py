"""Plan, shard, publish, or validate a generic behavior-cohort export.

The initial installed profile contains only protocol-neutral cohort, bundle,
and capability tables.  Scientific tables are added through exact table specs
and recording-scoped adapters, not command-line formula switches.
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
    write_validated_behavior_export_plan,
    write_validated_behavior_recording_shard,
)
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_METADATA_PROFILE_ID,
    CORE_TABLE_NAMES,
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

    plan = subparsers.add_parser("plan", help="write one immutable core export plan")
    plan.add_argument("--membership", type=Path, required=True)
    plan.add_argument("--bundle-set", type=Path, required=True)
    plan.add_argument("--export-run-id", required=True)
    plan.add_argument("--plan-output", type=Path, required=True)
    plan.add_argument("--shard-root", type=Path, required=True)
    plan.add_argument("--publication-root", type=Path, required=True)

    shard = subparsers.add_parser("shard", help="materialize one recording shard")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--member-ordinal", type=int, required=True)

    run_shards = subparsers.add_parser(
        "run-shards", help="materialize every core shard serially for a canary"
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
    value = build_validated_behavior_export_plan(
        membership_path=args.membership,
        bundle_set_path=args.bundle_set,
        export_run_id=args.export_run_id,
        shard_root=args.shard_root,
        publication_root=args.publication_root,
        palette_commit=commit,
        palette_repo=repository,
        export_profile_id=CORE_METADATA_PROFILE_ID,
    )
    target = write_validated_behavior_export_plan(args.plan_output, value)
    return {
        "operation": "plan",
        "profile_id": CORE_METADATA_PROFILE_ID,
        "plan_path": str(target),
        "plan_sha256": value["plan_sha256"],
        "export_run_id": value["export_run_id"],
        "member_count": value["member_count"],
        "table_names": list(CORE_TABLE_NAMES),
        "safety": value["safety"],
    }


def _read_plan_for_execution(path: Path) -> Mapping[str, Any]:
    plan, _membership, _bundle_set = read_validated_behavior_export_plan(path)
    _require_clean_current_authority(plan["software_authority"])
    return plan


def _shard_command(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_plan_for_execution(args.plan)
    receipt = write_validated_behavior_recording_shard(
        plan_path=args.plan,
        member_ordinal=args.member_ordinal,
    )
    return {
        "operation": "shard",
        "export_run_id": plan["export_run_id"],
        "member_ordinal": args.member_ordinal,
        "recording_id": receipt["member"]["recording_id"],
        "receipt_path": receipt["receipt_path"],
        "record_sha256": receipt["record_sha256"],
        "reused": receipt["reused"],
        "row_counts_by_table": {
            name: receipt["parts_by_table"][name]["row_count"]
            for name in receipt["requested_tables"]
        },
    }


def _run_shards_command(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_plan_for_execution(args.plan)
    created = 0
    reused = 0
    for ordinal in range(1, int(plan["member_count"]) + 1):
        receipt = write_validated_behavior_recording_shard(
            plan_path=args.plan, member_ordinal=ordinal
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
    plan = _read_plan_for_execution(args.plan)
    manifest = publish_validated_behavior_cohort(
        plan_path=args.plan, generation_id=args.generation_id
    )
    return {
        "operation": "finalize",
        "export_run_id": plan["export_run_id"],
        "manifest_path": manifest["manifest_path"],
        "record_sha256": manifest["record_sha256"],
        "row_counts_by_table": manifest["row_counts_by_table"],
        "safety": manifest["safety"],
    }


def _validate_command(args: argparse.Namespace) -> dict[str, Any]:
    manifest, membership, bundle_set = read_validated_behavior_export_manifest(
        args.publication_root,
        args.export_run_id,
        validate_parts="full" if args.full_part_hashes else "receipt",
    )
    return {
        "operation": "validate",
        "validation_mode": "full" if args.full_part_hashes else "receipt",
        "export_run_id": manifest["export_run_id"],
        "record_sha256": manifest["record_sha256"],
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
