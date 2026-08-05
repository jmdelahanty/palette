"""Freeze or apply an all-accepted refined snapshot cohort plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.detection_snapshot_publication import (
    inspect_accept_all_refined_detection_source,
    publish_accept_all_refined_detection_successor,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.utils.publish_canonical_detection_successor_batch import (
    validate_plan as validate_canonical_successor_plan,
)


PLAN_SCHEMA_ID = "palette.accept_all_refined_detection.batch_plan"
PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.accept_all_refined_detection.batch_result"
RESULT_SCHEMA_VERSION = 1
_PLAN_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "canonical_successor_plan_digest",
        "canonical_run_id",
        "refined_run_id",
        "candidate_count",
        "candidates",
        "digest_algorithm",
        "plan_digest",
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
    unexpected = sorted(set(plan) - _PLAN_FIELDS)
    missing = sorted(_PLAN_FIELDS - set(plan))
    if unexpected:
        errors.append(f"unexpected plan fields: {unexpected!r}")
    if missing:
        errors.append(f"missing plan fields: {missing!r}")
    if plan.get("schema_id") != PLAN_SCHEMA_ID:
        errors.append("plan schema_id is invalid")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        errors.append("plan schema_version is invalid")
    if plan.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("plan digest_algorithm is invalid")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        errors.append("plan candidates must be a list")
        candidates = []
    if plan.get("candidate_count") != len(candidates):
        errors.append("plan candidate_count does not match candidates")
    identities: list[str] = []
    paths: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidate {index} must be an object")
            continue
        if set(candidate) != {"analysis_zarr", "inspection"}:
            errors.append(f"candidate {index} fields are not exact")
            continue
        inspection = candidate.get("inspection")
        if not isinstance(inspection, dict):
            errors.append(f"candidate {index} inspection must be an object")
            continue
        if candidate.get("analysis_zarr") != inspection.get("analysis_zarr"):
            errors.append(f"candidate {index} archive binding differs")
        source = inspection.get("source")
        target = inspection.get("target")
        if not isinstance(source, Mapping) or source.get("run_id") != plan.get(
            "canonical_run_id"
        ):
            errors.append(f"candidate {index} canonical run differs")
        if not isinstance(target, Mapping) or target.get("run_id") != plan.get(
            "refined_run_id"
        ):
            errors.append(f"candidate {index} refined run differs")
        identities.append(str(inspection.get("recording_identity") or ""))
        paths.append(str(candidate.get("analysis_zarr") or ""))
    if len(identities) != len(set(identities)):
        errors.append("recording identities must be unique")
    if len(paths) != len(set(paths)):
        errors.append("analysis Zarr paths must be unique")
    if plan.get("plan_digest") != canonical_json_sha256(_plan_payload(plan)):
        errors.append("plan_digest does not match canonical plan payload")
    return tuple(errors)


def build_plan(
    *,
    canonical_successor_plan: Mapping[str, Any],
    refined_run_id: str,
) -> dict[str, Any]:
    source_errors = validate_canonical_successor_plan(canonical_successor_plan)
    if source_errors:
        raise ValueError(
            "Canonical successor plan is invalid: " + "; ".join(source_errors)
        )
    canonical_run_id = str(canonical_successor_plan["successor_run_id"])
    candidates: list[dict[str, Any]] = []
    for candidate in canonical_successor_plan["candidates"]:
        source_inspection = candidate["inspection"]
        archive = Path(str(candidate["analysis_zarr"]))
        inspection = inspect_accept_all_refined_detection_source(
            analysis_zarr=archive,
            canonical_run_id=canonical_run_id,
            recording_identity=str(source_inspection["recording_identity"]),
            refined_run_id=refined_run_id,
        )
        candidates.append(
            {
                "analysis_zarr": str(archive.expanduser().resolve()),
                "inspection": inspection,
            }
        )
    candidates.sort(key=lambda item: str(item["analysis_zarr"]))
    payload: dict[str, Any] = {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "canonical_successor_plan_digest": canonical_successor_plan["plan_digest"],
        "canonical_run_id": canonical_run_id,
        "refined_run_id": str(refined_run_id),
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
        raise RuntimeError("Invalid generated plan: " + "; ".join(errors))
    return plan


def apply_plan(
    plan: Mapping[str, Any],
    *,
    scratch_root: Path,
    receipt_root: Path,
    only_recording_identities: frozenset[str] = frozenset(),
    exclude_recording_identities: frozenset[str] = frozenset(),
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    errors = validate_plan(plan)
    if errors:
        raise ValueError("Refusing invalid frozen plan: " + "; ".join(errors))
    overlap = only_recording_identities & exclude_recording_identities
    if overlap:
        raise ValueError(
            f"Recordings cannot be both selected and excluded: {overlap!r}"
        )
    selected: list[Mapping[str, Any]] = []
    for candidate in plan["candidates"]:
        identity = str(candidate["inspection"]["recording_identity"])
        if only_recording_identities and identity not in only_recording_identities:
            continue
        if identity in exclude_recording_identities:
            continue
        selected.append(candidate)
    if only_recording_identities:
        observed = {str(item["inspection"]["recording_identity"]) for item in selected}
        missing = sorted(only_recording_identities - observed)
        if missing:
            raise ValueError(f"Requested recordings are absent from plan: {missing!r}")
    if not selected:
        raise ValueError("Frozen plan selection contains no candidates.")

    receipt_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for candidate in selected:
        expected = dict(candidate["inspection"])
        observed = inspect_accept_all_refined_detection_source(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            canonical_run_id=str(plan["canonical_run_id"]),
            recording_identity=str(expected["recording_identity"]),
            refined_run_id=str(plan["refined_run_id"]),
        )
        if observed != expected:
            raise RuntimeError(
                "Frozen source inspection drifted before publication for "
                f"{expected['recording_identity']}."
            )
        identity = str(expected["recording_identity"])
        receipt_path = receipt_root / f"{identity}.json"
        result = publish_accept_all_refined_detection_successor(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            canonical_run_id=str(plan["canonical_run_id"]),
            recording_identity=identity,
            refined_run_id=str(plan["refined_run_id"]),
            scratch_root=scratch_root,
            copy_backend=copy_backend,
            keep_scratch=keep_scratch,
            result_json=receipt_path,
        )
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
        "selector_activation": "none",
        "registry_updated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--canonical-successor-plan", type=Path)
    parser.add_argument("--refined-run")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--only-recording", action="append", default=[])
    parser.add_argument("--exclude-recording", action="append", default=[])
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply:
            if args.scratch_root is None or args.receipt_root is None:
                raise ValueError("--apply requires --scratch-root and --receipt-root.")
            plan = _strict_json_load(args.plan_json.expanduser().resolve())
            result = apply_plan(
                plan,
                scratch_root=args.scratch_root,
                receipt_root=args.receipt_root,
                only_recording_identities=frozenset(args.only_recording),
                exclude_recording_identities=frozenset(args.exclude_recording),
                copy_backend=args.copy_backend,
                keep_scratch=args.keep_scratch,
            )
        else:
            if args.canonical_successor_plan is None or not args.refined_run:
                raise ValueError(
                    "Plan construction requires --canonical-successor-plan and "
                    "--refined-run."
                )
            source_plan = _strict_json_load(
                args.canonical_successor_plan.expanduser().resolve()
            )
            plan = build_plan(
                canonical_successor_plan=source_plan,
                refined_run_id=args.refined_run,
            )
            write_json_atomic(args.plan_json.expanduser().resolve(), plan)
            result = {
                "schema_id": RESULT_SCHEMA_ID,
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "planned",
                "mode": "dry_run",
                "plan_json": str(args.plan_json.expanduser().resolve()),
                "plan_digest": plan["plan_digest"],
                "candidate_count": plan["candidate_count"],
                "zarr_writes": False,
            }
        if args.result_json is not None:
            write_json_atomic(args.result_json.expanduser().resolve(), result)
    except Exception as exc:
        result = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "failed",
            "mode": "apply" if args.apply else "plan",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.result_json is not None:
            write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
