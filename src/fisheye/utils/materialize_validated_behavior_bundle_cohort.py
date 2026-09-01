"""Plan or validate generic membership and bundle-set manifests.

The command writes only external JSON contracts.  It never mutates source
Zarrs, resolves selectors, writes the Palette registry, or activates an export.
Protocol-specific inputs are handled by explicit adapters; the persisted
membership and bundle-set formats remain generic validated-behavior contracts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.analysis_workflows.validated_behavior_cohort import (
    policy_envelope,
    validate_validated_behavior_bundle_set,
)
from fisheye.analysis_workflows.validated_behavior_cohort_adapters import (
    build_bundle_set_from_validated_recording_behavior_bundles,
    build_membership_from_composable_chaser_task_v5,
    build_membership_from_frozen_cohort_v2,
    missing_acquisition_batch_policy,
    plan_composable_chaser_task_v5_dispositions,
    recording_scoped_analysis_unit_policy,
    sha256_file,
    validate_membership_current_sources,
    validate_recording_behavior_bundle_set_current_sources,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


BUNDLE_PATHS_SCHEMA_ID = "palette.analysis.validated_behavior_bundle_paths"
BUNDLE_PATHS_SCHEMA_VERSION = 1


class ValidatedBehaviorCohortCliError(ValueError):
    """A CLI input cannot produce one exact generic cohort contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: str | Path, *, field: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{field} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorCohortCliError(
            f"Cannot read strict JSON object from {source}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValidatedBehaviorCohortCliError(f"{field} must contain one object.")
    return source, value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidatedBehaviorCohortCliError(
            f"{field} must be one lowercase SHA-256 digest."
        )
    return value


def _load_policy(path: str | Path, *, field: str) -> dict[str, Any]:
    _, value = _read_object(path, field=field)
    if set(value) == {"record", "sha256"}:
        # The core builder performs the full envelope validation.
        return value
    return policy_envelope(value)


def _check_expected_digest(path: Path, expected: str | None, *, field: str) -> None:
    if expected is None:
        return
    observed = sha256_file(path)
    if observed != _digest(expected, field=field):
        raise ValidatedBehaviorCohortCliError(
            f"{field} mismatch: expected {expected}, observed {observed}."
        )


def _current_palette_git_state() -> tuple[str, str]:
    repository = Path(__file__).resolve().parents[3]
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
        raise ValidatedBehaviorCohortCliError(
            "Cannot verify the current Palette Git identity."
        ) from exc
    return commit, status


def _require_current_software_authority(expected_commit: str) -> None:
    expected = str(expected_commit).strip()
    current, status = _current_palette_git_state()
    if expected != current:
        raise ValidatedBehaviorCohortCliError(
            "--palette-commit must equal the exact commit executing this command; "
            f"expected {current}, received {expected}."
        )
    if status:
        raise ValidatedBehaviorCohortCliError(
            "Artifact planning requires a clean commit-pinned Palette worktree."
        )


def _check_counts(
    membership: Mapping[str, Any],
    *,
    parent: int | None,
    admitted: int | None,
    invalid: int | None,
) -> None:
    observed = {
        "parent": int(membership["member_count"]),
        "admitted": int(membership["state_counts"]["admitted"]),
        "invalid": int(membership["state_counts"]["invalid"]),
    }
    expected = {"parent": parent, "admitted": admitted, "invalid": invalid}
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if value is not None and value != observed[key]
    }
    if mismatches:
        raise ValidatedBehaviorCohortCliError(
            f"Membership count expectations failed: {mismatches!r}."
        )


def _ensure_json(
    output: Path,
    requested: Mapping[str, Any],
    *,
    validator: Any,
) -> tuple[dict[str, Any], str]:
    target = output.expanduser().resolve()
    if target.exists():
        _, current_raw = _read_object(target, field="existing output")
        current = validator(current_raw)
        if dict(current) != dict(validator(requested)):
            raise ValidatedBehaviorCohortCliError(
                f"Existing immutable output differs from the request: {target}"
            )
        return current_raw, "reused_exact"
    write_json_atomic(target, requested, overwrite=False)
    persisted_raw = json.loads(target.read_text(encoding="utf-8"))
    validator(persisted_raw)
    return persisted_raw, "created"


def _load_bundle_paths(
    path: str | Path, *, membership_record_sha256: str
) -> dict[str, str]:
    _, document = _read_object(path, field="bundle paths")
    persisted = _digest(document.get("record_sha256"), field="bundle paths digest")
    body = {key: value for key, value in document.items() if key != "record_sha256"}
    if canonical_json_sha256(body) != persisted:
        raise ValidatedBehaviorCohortCliError("Bundle-path document digest is stale.")
    if set(body) != {
        "schema_id",
        "schema_version",
        "membership_record_sha256",
        "entry_count",
        "entries",
    }:
        raise ValidatedBehaviorCohortCliError(
            "Bundle-path document field set is inexact."
        )
    if (
        body.get("schema_id") != BUNDLE_PATHS_SCHEMA_ID
        or body.get("schema_version") != BUNDLE_PATHS_SCHEMA_VERSION
        or body.get("membership_record_sha256") != membership_record_sha256
    ):
        raise ValidatedBehaviorCohortCliError(
            "Bundle paths bind another schema or membership generation."
        )
    entries = body.get("entries")
    if not isinstance(entries, list) or body.get("entry_count") != len(entries):
        raise ValidatedBehaviorCohortCliError("Bundle-path entry count is stale.")
    result: dict[str, str] = {}
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping) or set(raw) != {"recording_id", "bundle_path"}:
            raise ValidatedBehaviorCohortCliError(
                f"Bundle-path entry {index} is inexact."
            )
        recording_id = str(raw.get("recording_id") or "").strip()
        bundle_path = str(raw.get("bundle_path") or "").strip()
        if not recording_id or not bundle_path or recording_id in result:
            raise ValidatedBehaviorCohortCliError(
                f"Bundle-path entry {index} is empty or duplicated."
            )
        result[recording_id] = bundle_path
    return result


def _membership_parser(subparsers: Any, *, frozen: bool) -> None:
    command = (
        "membership-from-frozen-v2" if frozen else "membership-from-chaser-task-v5"
    )
    parser = subparsers.add_parser(command)
    parser.add_argument(
        "--source-membership",
        type=Path,
        required=True,
        help="Exact schema-v5 task or frozen-cohort-v2 manifest.",
    )
    parser.add_argument("--source-membership-file-sha256")
    if frozen:
        parser.add_argument(
            "--dispositions-json",
            type=Path,
            required=True,
            help="Explicit complete disposition mapping keyed by recording ID.",
        )
    else:
        parser.add_argument("--receipt-generation", required=True)
        parser.add_argument("--receipt-filename", required=True)
        parser.add_argument("--invalid-dispositions-json", type=Path, required=True)
    parser.add_argument("--membership-id", required=True)
    parser.add_argument("--analysis-zarr-root", type=Path, required=True)
    parser.add_argument("--admission-receipt-root", type=Path, required=True)
    parser.add_argument("--identity-decision-evidence", type=Path, required=True)
    parser.add_argument("--identity-decision-evidence-file-sha256")
    parser.add_argument("--identity-decision-timestamp-utc", required=True)
    parser.add_argument("--distinct-animal-count", type=int, required=True)
    parser.add_argument("--capture-subject-uuid-reuse-count", type=int, required=True)
    parser.add_argument("--temporal-alignment-policy-json", type=Path, required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-parent-count", type=int)
    parser.add_argument("--expected-admitted-count", type=int)
    parser.add_argument("--expected-invalid-count", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _membership_parser(subparsers, frozen=False)
    _membership_parser(subparsers, frozen=True)

    bundle = subparsers.add_parser("bundle-set")
    bundle.add_argument("--membership", type=Path, required=True)
    bundle.add_argument("--bundle-paths-json", type=Path, required=True)
    bundle.add_argument("--bundle-set-id", required=True)
    bundle.add_argument("--bundle-root", type=Path, required=True)
    bundle.add_argument("--palette-commit", required=True)
    bundle.add_argument("--output-json", type=Path, required=True)

    validate_membership = subparsers.add_parser("validate-membership")
    validate_membership.add_argument("--membership", type=Path, required=True)

    validate_bundle = subparsers.add_parser("validate-bundle-set")
    validate_bundle.add_argument("--membership", type=Path, required=True)
    validate_bundle.add_argument("--bundle-set", type=Path, required=True)
    return parser


def _membership_command(args: argparse.Namespace) -> dict[str, Any]:
    _require_current_software_authority(args.palette_commit)
    source = args.source_membership.expanduser().resolve()
    _check_expected_digest(
        source,
        args.source_membership_file_sha256,
        field="source membership file digest",
    )
    evidence = args.identity_decision_evidence.expanduser().resolve()
    _check_expected_digest(
        evidence,
        args.identity_decision_evidence_file_sha256,
        field="identity decision evidence file digest",
    )
    unit_policy = recording_scoped_analysis_unit_policy(
        distinct_animal_count=args.distinct_animal_count,
        decision_timestamp_utc=args.identity_decision_timestamp_utc,
        decision_evidence_path=evidence,
        decision_evidence_file_sha256=sha256_file(evidence),
        capture_subject_uuid_reuse_count=args.capture_subject_uuid_reuse_count,
    )
    temporal_policy = _load_policy(
        args.temporal_alignment_policy_json,
        field="temporal alignment policy",
    )
    if args.command == "membership-from-chaser-task-v5":
        dispositions = plan_composable_chaser_task_v5_dispositions(
            source,
            receipt_generation=args.receipt_generation,
            receipt_filename=args.receipt_filename,
            invalid_dispositions_path=args.invalid_dispositions_json,
        )
        builder = build_membership_from_composable_chaser_task_v5
    else:
        _, disposition_document = _read_object(
            args.dispositions_json, field="member dispositions"
        )
        if not all(
            isinstance(key, str) and isinstance(value, Mapping)
            for key, value in disposition_document.items()
        ):
            raise ValidatedBehaviorCohortCliError(
                "Frozen-v2 dispositions must be an object keyed by recording ID."
            )
        dispositions = disposition_document
        builder = build_membership_from_frozen_cohort_v2
    common = {
        "membership_id": args.membership_id,
        "dispositions_by_recording": dispositions,
        "analysis_zarr_root": args.analysis_zarr_root,
        "admission_receipt_root": args.admission_receipt_root,
        "analysis_unit_policy": unit_policy,
        "acquisition_batch_policy": missing_acquisition_batch_policy(),
        "temporal_alignment_policy": temporal_policy,
        "palette_commit": args.palette_commit,
    }
    output = args.output_json.expanduser().resolve()
    if output.exists():
        _, existing = _read_object(output, field="existing membership")
        created_at_utc = existing.get("created_at_utc")
    else:
        created_at_utc = _utc_now()
    requested = builder(source, created_at_utc=created_at_utc, **common)
    requested = dict(validate_membership_current_sources(requested))
    _check_counts(
        requested,
        parent=args.expected_parent_count,
        admitted=args.expected_admitted_count,
        invalid=args.expected_invalid_count,
    )
    persisted, mode = _ensure_json(
        output,
        requested,
        validator=validate_membership_current_sources,
    )
    return {
        "mode": mode,
        "path": str(output),
        "schema_id": persisted["schema_id"],
        "schema_version": persisted["schema_version"],
        "membership_id": persisted["membership_id"],
        "record_sha256": persisted["record_sha256"],
        "member_count": persisted["member_count"],
        "state_counts": persisted["state_counts"],
        "selector_eligible": persisted["safety"]["selector_eligible"],
        "production_authority": persisted["safety"]["production_authority"],
    }


def _bundle_set_command(args: argparse.Namespace) -> dict[str, Any]:
    _require_current_software_authority(args.palette_commit)
    membership_path, membership_raw = _read_object(args.membership, field="membership")
    membership = validate_membership_current_sources(membership_raw)
    bundle_paths = _load_bundle_paths(
        args.bundle_paths_json,
        membership_record_sha256=membership["record_sha256"],
    )
    output = args.output_json.expanduser().resolve()
    if output.exists():
        _, existing = _read_object(output, field="existing bundle set")
        created_at_utc = existing.get("created_at_utc")
    else:
        created_at_utc = _utc_now()
    requested = build_bundle_set_from_validated_recording_behavior_bundles(
        bundle_set_id=args.bundle_set_id,
        membership=membership,
        membership_path=membership_path,
        bundle_paths_by_recording=bundle_paths,
        bundle_root=args.bundle_root,
        palette_commit=args.palette_commit,
        created_at_utc=created_at_utc,
        validate_current_sources=True,
    )

    def validator(raw: object) -> Mapping[str, Any]:
        return validate_recording_behavior_bundle_set_current_sources(
            raw, membership=membership
        )

    persisted, mode = _ensure_json(output, requested, validator=validator)
    return {
        "mode": mode,
        "path": str(output),
        "schema_id": persisted["schema_id"],
        "schema_version": persisted["schema_version"],
        "bundle_set_id": persisted["bundle_set_id"],
        "record_sha256": persisted["record_sha256"],
        "member_count": persisted["member_count"],
        "state_counts": persisted["state_counts"],
        "capability_matrix_sha256": persisted["capability_matrix_sha256"],
        "selector_eligible": persisted["safety"]["selector_eligible"],
        "production_authority": persisted["safety"]["production_authority"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command.startswith("membership-from-"):
        summary = _membership_command(args)
    elif args.command == "bundle-set":
        summary = _bundle_set_command(args)
    elif args.command == "validate-membership":
        path, raw = _read_object(args.membership, field="membership")
        value = validate_membership_current_sources(raw)
        summary = {
            "status": "valid",
            "path": str(path),
            "record_sha256": value["record_sha256"],
            "member_count": value["member_count"],
            "state_counts": dict(value["state_counts"]),
        }
    else:
        membership_path, membership_raw = _read_object(
            args.membership, field="membership"
        )
        membership = validate_membership_current_sources(membership_raw)
        bundle_path, bundle_raw = _read_object(args.bundle_set, field="bundle set")
        # First prove the generic membership binding before the adapter re-opens
        # the complete bundle sources.
        validate_validated_behavior_bundle_set(bundle_raw, membership=membership)
        value = validate_recording_behavior_bundle_set_current_sources(
            bundle_raw, membership=membership
        )
        summary = {
            "status": "valid",
            "membership_path": str(membership_path),
            "bundle_set_path": str(bundle_path),
            "record_sha256": value["record_sha256"],
            "member_count": value["member_count"],
            "state_counts": dict(value["state_counts"]),
        }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_PATHS_SCHEMA_ID",
    "BUNDLE_PATHS_SCHEMA_VERSION",
    "ValidatedBehaviorCohortCliError",
    "main",
]
