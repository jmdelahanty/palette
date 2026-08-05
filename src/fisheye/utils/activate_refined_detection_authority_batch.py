"""Freeze or apply an analysis-only refined-detection authority cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_authority_activation import (
    activate_refined_detection_authority,
    inspect_active_refined_detection_authority,
    inspect_refined_detection_authority_candidate,
)
from fisheye.utils.publish_accept_all_refined_detection_batch import (
    validate_plan as validate_refined_publication_plan,
)


PLAN_SCHEMA_ID = "palette.refined_detection.authority_activation_batch_plan"
PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.refined_detection.authority_activation_batch_result"
RESULT_SCHEMA_VERSION = 1
_PLAN_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "source_refined_publication_plan_digest",
        "run_id",
        "approval",
        "candidate_count",
        "candidates",
        "digest_algorithm",
        "plan_digest",
    }
)
_APPROVAL_FIELDS = frozenset(
    {
        "approved_by",
        "approved_at_utc",
        "review_method",
        "intended_use",
        "git_sha",
        "note",
    }
)


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _plan_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"digest_algorithm", "plan_digest"}
    }


def validate_plan(plan: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if set(plan) != _PLAN_FIELDS:
        errors.append("activation plan field set is not exact")
    if plan.get("schema_id") != PLAN_SCHEMA_ID:
        errors.append("activation plan schema_id is invalid")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        errors.append("activation plan schema_version is invalid")
    if plan.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("activation plan digest_algorithm is invalid")
    approval = plan.get("approval")
    if not isinstance(approval, Mapping) or set(approval) != _APPROVAL_FIELDS:
        errors.append("activation approval field set is not exact")
        approval = {}
    if approval.get("intended_use") != "analysis":
        errors.append("activation intended_use must be analysis")
    for name in ("approved_by", "approved_at_utc", "review_method", "git_sha"):
        if not isinstance(approval.get(name), str) or not approval.get(name):
            errors.append(f"activation approval {name} must be nonempty")
    if not isinstance(approval.get("note"), str):
        errors.append("activation approval note must be text")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        errors.append("activation candidates must be a list")
        candidates = []
    if plan.get("candidate_count") != len(candidates):
        errors.append("activation candidate_count differs from candidates")
    paths: list[str] = []
    identities: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "analysis_zarr",
            "inspection",
        }:
            errors.append(f"activation candidate {index} field set is not exact")
            continue
        inspection = candidate.get("inspection")
        if not isinstance(inspection, Mapping):
            errors.append(f"activation candidate {index} inspection is invalid")
            continue
        if candidate.get("analysis_zarr") != inspection.get("analysis_zarr"):
            errors.append(f"activation candidate {index} archive binding differs")
        if inspection.get("run_id") != plan.get("run_id"):
            errors.append(f"activation candidate {index} run binding differs")
        if inspection.get("intended_use") != "analysis":
            errors.append(f"activation candidate {index} intended use differs")
        paths.append(str(candidate.get("analysis_zarr") or ""))
        identities.append(str(inspection.get("recording_identity") or ""))
    if len(paths) != len(set(paths)):
        errors.append("activation archive paths must be unique")
    if len(identities) != len(set(identities)):
        errors.append("activation recording identities must be unique")
    if plan.get("plan_digest") != canonical_json_sha256(_plan_payload(plan)):
        errors.append("activation plan_digest differs from its payload")
    return tuple(errors)


def build_plan(
    *,
    refined_publication_plan: Mapping[str, Any],
    approved_by: str,
    approved_at_utc: str,
    review_method: str,
    git_sha: str,
    note: str = "",
) -> dict[str, Any]:
    source_errors = validate_refined_publication_plan(refined_publication_plan)
    if source_errors:
        raise ValueError(
            "Refined publication plan is invalid: " + "; ".join(source_errors)
        )
    run_id = str(refined_publication_plan["refined_run_id"])
    candidates: list[dict[str, Any]] = []
    for source in refined_publication_plan["candidates"]:
        archive = Path(str(source["analysis_zarr"])).expanduser().resolve()
        inspection = inspect_refined_detection_authority_candidate(
            analysis_zarr=archive,
            run_id=run_id,
        )
        expected_identity = source["inspection"]["recording_identity"]
        if inspection["recording_identity"] != expected_identity:
            raise RuntimeError(
                "Activation candidate recording identity differs from the frozen "
                f"publication plan for {archive}."
            )
        candidates.append(
            {"analysis_zarr": str(archive), "inspection": inspection}
        )
    candidates.sort(key=lambda item: str(item["analysis_zarr"]))
    payload: dict[str, Any] = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_refined_publication_plan_digest": refined_publication_plan[
            "plan_digest"
        ],
        "run_id": run_id,
        "approval": {
            "approved_by": str(approved_by),
            "approved_at_utc": str(approved_at_utc),
            "review_method": str(review_method),
            "intended_use": "analysis",
            "git_sha": str(git_sha),
            "note": str(note),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    plan = {
        **payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "plan_digest": canonical_json_sha256(payload),
    }
    errors = validate_plan(plan)
    if errors:
        raise RuntimeError("Generated activation plan is invalid: " + "; ".join(errors))
    return plan


def apply_plan(
    plan: Mapping[str, Any],
    *,
    receipt_root: Path,
    only_recording_identities: frozenset[str] = frozenset(),
    exclude_recording_identities: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    errors = validate_plan(plan)
    if errors:
        raise ValueError("Refusing invalid activation plan: " + "; ".join(errors))
    overlap = only_recording_identities & exclude_recording_identities
    if overlap:
        raise ValueError(
            f"Recordings cannot be both selected and excluded: {overlap!r}"
        )
    selected = []
    for candidate in plan["candidates"]:
        identity = str(candidate["inspection"]["recording_identity"])
        if only_recording_identities and identity not in only_recording_identities:
            continue
        if identity in exclude_recording_identities:
            continue
        selected.append(candidate)
    if only_recording_identities:
        observed = {
            str(item["inspection"]["recording_identity"]) for item in selected
        }
        missing = sorted(only_recording_identities - observed)
        if missing:
            raise ValueError(f"Requested recordings are absent: {missing!r}")
    if not selected:
        raise ValueError("Activation plan selection contains no candidates.")

    receipt_root = receipt_root.expanduser().resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    approval = plan["approval"]
    results: list[dict[str, Any]] = []
    for candidate in selected:
        inspection = candidate["inspection"]
        identity = str(inspection["recording_identity"])
        result = activate_refined_detection_authority(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            run_id=str(plan["run_id"]),
            approved_by=str(approval["approved_by"]),
            approved_at_utc=str(approval["approved_at_utc"]),
            review_method=str(approval["review_method"]),
            git_sha=str(approval["git_sha"]),
            note=str(approval["note"]),
            expected_inspection=inspection,
        )
        write_json_atomic(receipt_root / f"{identity}.json", result)
        results.append(result)
    return {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "plan_digest": plan["plan_digest"],
        "requested_candidate_count": len(selected),
        "completed_candidate_count": len(results),
        "receipts": [
            str(receipt_root / f"{item['recording_identity']}.json")
            for item in results
        ],
        "intended_use": "analysis",
        "registry_updated": False,
    }


def verify_plan(
    plan: Mapping[str, Any],
    *,
    receipt_root: Path,
) -> dict[str, Any]:
    """Reopen every committed authority and reconcile it to plan and receipt."""

    errors = validate_plan(plan)
    if errors:
        raise ValueError("Refusing invalid activation plan: " + "; ".join(errors))
    receipt_root = receipt_root.expanduser().resolve()
    approval = plan["approval"]
    inspections: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        frozen = candidate["inspection"]
        identity = str(frozen["recording_identity"])
        active = inspect_active_refined_detection_authority(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            run_id=str(plan["run_id"]),
        )
        expected_authority_payload = {
            "run_id": str(plan["run_id"]),
            "run_manifest_digest": frozen["activation_manifest_digest"],
            "review_state": "approved",
            "review_method": approval["review_method"],
            "intended_use": "analysis",
            "approved_by": approval["approved_by"],
            "approved_at_utc": approval["approved_at_utc"],
            "git_sha": approval["git_sha"],
            "note": approval["note"],
        }
        expected = {
            "analysis_zarr": candidate["analysis_zarr"],
            "recording_identity": identity,
            "run_id": plan["run_id"],
            "manifest_digest": frozen["activation_manifest_digest"],
            "logical_content_digest": frozen["logical_content_digest"],
            "publication_owner_uuid": frozen["publication_owner_uuid"],
            "storage_profile_id": frozen["storage_profile_id"],
            "authority_payload": expected_authority_payload,
        }
        observed = {
            "analysis_zarr": active["analysis_zarr"],
            "recording_identity": active["recording_identity"],
            "run_id": active["run_id"],
            "manifest_digest": active["manifest_digest"],
            "logical_content_digest": active["logical_content_digest"],
            "publication_owner_uuid": active["publication_owner_uuid"],
            "storage_profile_id": active["storage_profile_id"],
            "authority_payload": active["authority_provenance"]["payload"],
        }
        if observed != expected:
            raise RuntimeError(
                f"Active refined authority differs from plan for {identity}."
            )
        receipt_path = receipt_root / f"{identity}.json"
        receipt = _strict_json_load(receipt_path)
        receipt_projection = {
            "analysis_zarr": receipt.get("analysis_zarr"),
            "recording_identity": receipt.get("recording_identity"),
            "run_id": receipt.get("run_id"),
            "manifest_digest": receipt.get("activated_manifest_digest"),
            "logical_content_digest": receipt.get("logical_content_digest"),
            "publication_owner_uuid": receipt.get("publication_owner_uuid"),
            "storage_profile_id": frozen["storage_profile_id"],
            "authority_payload": receipt.get("authority_provenance", {}).get(
                "payload"
            ),
        }
        if receipt_projection != expected:
            raise RuntimeError(
                f"Refined authority receipt differs from plan for {identity}."
            )
        if (
            receipt.get("status") != "complete"
            or receipt.get("selection_mode")
            != "approved_authoritative_refined_v1"
            or receipt.get("post_commit_archive_writes") != 0
            or receipt.get("registry_updated") is not False
        ):
            raise RuntimeError(
                f"Refined authority receipt state is invalid for {identity}."
            )
        inspections.append(active)
    return {
        "schema_id": "palette.refined_detection.authority_activation_batch_verification",
        "schema_version": 1,
        "status": "valid",
        "verified_at_utc": utc_now(),
        "plan_digest": plan["plan_digest"],
        "verified_candidate_count": len(inspections),
        "total_instance_count": sum(
            int(item["dimensions"]["n_instances"]) for item in inspections
        ),
        "intended_use": "analysis",
        "all_selection_modes": sorted(
            {str(item["selection_mode"]) for item in inspections}
        ),
        "all_storage_profiles": sorted(
            {str(item["storage_profile_id"]) for item in inspections}
        ),
        "registry_updated": False,
        "archives": inspections,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--refined-publication-plan", type=Path)
    parser.add_argument("--approved-by")
    parser.add_argument("--approved-at-utc")
    parser.add_argument("--review-method")
    parser.add_argument("--git-sha")
    parser.add_argument("--note", default="")
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--only-recording", action="append", default=[])
    parser.add_argument("--exclude-recording", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply and args.verify:
            raise ValueError("--apply and --verify are mutually exclusive.")
        if args.verify:
            if args.receipt_root is None:
                raise ValueError("--verify requires --receipt-root.")
            plan = _strict_json_load(args.plan_json.expanduser().resolve())
            result = verify_plan(plan, receipt_root=args.receipt_root)
        elif args.apply:
            if args.receipt_root is None:
                raise ValueError("--apply requires --receipt-root.")
            plan = _strict_json_load(args.plan_json.expanduser().resolve())
            result = apply_plan(
                plan,
                receipt_root=args.receipt_root,
                only_recording_identities=frozenset(args.only_recording),
                exclude_recording_identities=frozenset(args.exclude_recording),
            )
        else:
            required = {
                "--refined-publication-plan": args.refined_publication_plan,
                "--approved-by": args.approved_by,
                "--approved-at-utc": args.approved_at_utc,
                "--review-method": args.review_method,
                "--git-sha": args.git_sha,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "Plan construction is missing: " + ", ".join(missing)
                )
            source = _strict_json_load(
                args.refined_publication_plan.expanduser().resolve()
            )
            plan = build_plan(
                refined_publication_plan=source,
                approved_by=args.approved_by,
                approved_at_utc=args.approved_at_utc,
                review_method=args.review_method,
                git_sha=args.git_sha,
                note=args.note,
            )
            write_json_atomic(args.plan_json.expanduser().resolve(), plan)
            result = {
                "schema_id": RESULT_SCHEMA_ID,
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "plan_frozen",
                "completed_at_utc": utc_now(),
                "plan_digest": plan["plan_digest"],
                "candidate_count": plan["candidate_count"],
                "intended_use": "analysis",
                "registry_updated": False,
            }
        if args.result_json is not None:
            write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
