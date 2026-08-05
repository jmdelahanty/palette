"""Freeze or apply a cohort plan for canonical-v3 detection successors.

Plan construction is read-only. Application re-inspects every frozen source
before publishing its selector-ineligible successor and stops at the first
drift or publication failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.detection_snapshot_publication import (
    inspect_canonical_detection_successor_source,
    publish_canonical_detection_successor,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr_discovery import discover_registry_zarrs
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.tracking.crop import get_detection_source_info


PLAN_SCHEMA_ID = "palette.canonical_detection_successor.batch_plan"
PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.canonical_detection_successor.batch_result"
RESULT_SCHEMA_VERSION = 1
_PLAN_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "registry_path",
        "scope_paths",
        "path_contains",
        "successor_run_id",
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
        if inspection.get("successor_run_id") != plan.get("successor_run_id"):
            errors.append(f"candidate {index} successor run differs")
        identities.append(str(inspection.get("recording_identity") or ""))
        paths.append(str(candidate.get("analysis_zarr") or ""))
    if len(identities) != len(set(identities)):
        errors.append("recording identities must be unique")
    if len(paths) != len(set(paths)):
        errors.append("analysis Zarr paths must be unique")
    expected_digest = canonical_json_sha256(_plan_payload(plan))
    if plan.get("plan_digest") != expected_digest:
        errors.append("plan_digest does not match canonical plan payload")
    return tuple(errors)


def build_plan(
    *,
    registry_path: Path,
    scope_paths: Sequence[Path],
    path_contains: str,
    successor_run_id: str,
) -> dict[str, Any]:
    archives = discover_registry_zarrs(
        registry_path=registry_path,
        scope_paths=scope_paths,
        zarr_use="analysis",
        path_contains=path_contains,
        require_steps_ok=["detect"],
        zarr_suffix="_analysis.zarr",
    )
    candidates: list[dict[str, Any]] = []
    for archive in archives:
        root = open_zarr_root(archive, mode="r")
        recording_identity = str(root.attrs.get("recording_id") or "").strip()
        if not recording_identity:
            raise ValueError(f"Archive lacks root recording_id: {archive}")
        source_path, _source, _source_kind, resolved_type = get_detection_source_info(
            root, source_type="detect"
        )
        if resolved_type != "detect" or not source_path.startswith("detect_runs/"):
            raise ValueError(
                f"Expected a raw authoritative detect source for {archive}; "
                f"resolved {resolved_type!r} at {source_path!r}."
            )
        inspection = inspect_canonical_detection_successor_source(
            analysis_zarr=archive,
            source_detect_group_path=source_path,
            recording_identity=recording_identity,
            successor_run_id=successor_run_id,
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
        "registry_path": str(registry_path.expanduser().resolve()),
        "scope_paths": [str(path.expanduser().resolve()) for path in scope_paths],
        "path_contains": path_contains,
        "successor_run_id": successor_run_id,
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
        inspection = candidate["inspection"]
        identity = str(inspection["recording_identity"])
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
        observed = inspect_canonical_detection_successor_source(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            source_detect_group_path=str(expected["source_group_path"]),
            recording_identity=str(expected["recording_identity"]),
            successor_run_id=str(expected["successor_run_id"]),
        )
        if observed != expected:
            raise RuntimeError(
                "Frozen source inspection drifted before publication for "
                f"{expected['recording_identity']}."
            )
        identity = str(expected["recording_identity"])
        receipt_path = receipt_root / f"{identity}.json"
        result = publish_canonical_detection_successor(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            source_detect_group_path=str(expected["source_group_path"]),
            recording_identity=identity,
            successor_run_id=str(expected["successor_run_id"]),
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
            str(receipt_root / f"{item['recording_identity']}.json") for item in results
        ],
        "selector_activation": "none",
        "registry_updated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--scope", type=Path, action="append", default=[])
    parser.add_argument("--path-contains")
    parser.add_argument("--successor-run")
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
            for name in ("scratch_root", "receipt_root"):
                if getattr(args, name) is None:
                    raise ValueError(f"--apply requires --{name.replace('_', '-')}.")
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
            if args.registry is None or not args.scope:
                raise ValueError("Plan construction requires --registry and --scope.")
            if not args.path_contains or not args.successor_run:
                raise ValueError(
                    "Plan construction requires --path-contains and --successor-run."
                )
            plan = build_plan(
                registry_path=args.registry,
                scope_paths=args.scope,
                path_contains=args.path_contains,
                successor_run_id=args.successor_run,
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
            "mode": "apply" if args.apply else "dry_run",
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
