"""Serially publish derived-analysis completion receipts to the registry.

Parallel analysis workers own Zarr publication only.  This finalizer consumes
their immutable completion receipts, freshly revalidates the selected Zarr
runs, and performs the short SQLite transactions serially in one LSF job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.registry.db import Registry
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.derived_analysis_registry_status import (
    emit_derived_analysis_stage_completion,
    emit_eye_angle_stage_completion,
    emit_track_kinematics_stage_completion,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name

from .storage_contract_catalog import (
    DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE,
    SERIALIZED_REGISTRY_STAGE_IDS,
)


REPORT_SCHEMA = "palette.derived_analysis_registry_finalizer.v1"
TARGET_RECEIPT_SCHEMA = "palette.chaser_analytics_target_receipt.v1"
_DISABLE_REGISTRY_WRITES_ENV = "PALETTE_DISABLE_REGISTRY_WRITES"
_SUPPORTED_STAGES = SERIALIZED_REGISTRY_STAGE_IDS


@dataclass(frozen=True)
class RequestedPublication:
    zarr_path: Path
    stage_id: str
    requested_run: str
    receipt_path: Path
    receipt_sha256: str


@dataclass(frozen=True)
class PreparedPublication:
    request: RequestedPublication
    root: Any
    run_group: Any
    registry_run_name: str
    leaf_run_name: str
    run_type: str | None


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Receipt must contain one JSON object: {path}")
    return dict(payload), hashlib.sha256(payload_bytes).hexdigest()


def _require_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _execution_report_requests(path: Path) -> list[RequestedPublication]:
    payload, digest = _read_json_object(path)
    if payload.get("schema_id") != "palette.analysis_workflow_execution":
        raise ValueError(f"Unsupported analysis execution receipt: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported analysis execution receipt version: {path}")
    if payload.get("mode") != "apply" or payload.get("status") != "complete":
        raise RuntimeError(f"Analysis execution receipt is not complete: {path}")
    if payload.get("registry_write_mode") != "deferred_to_serial_finalizer":
        raise RuntimeError(
            f"Analysis execution receipt did not defer registry writes: {path}"
        )
    zarr_path = Path(_require_text(payload.get("zarr_path"), field="zarr_path"))

    execution_plan = payload.get("execution_plan")
    if not isinstance(execution_plan, Mapping):
        raise ValueError(f"Execution receipt lacks execution_plan: {path}")
    commands = execution_plan.get("commands")
    results = payload.get("node_results")
    if not isinstance(commands, list) or not isinstance(results, list):
        raise ValueError(f"Execution receipt lacks commands or node_results: {path}")
    result_by_node = {
        str(item.get("node_id")): item
        for item in results
        if isinstance(item, Mapping) and item.get("node_id")
    }

    requests: list[RequestedPublication] = []
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError(f"Execution command is not an object: {path}")
        stage_id = _require_text(command.get("stage_id"), field="stage_id")
        if stage_id not in _SUPPORTED_STAGES:
            continue
        node_id = _require_text(command.get("node_id"), field="node_id")
        run_name = _require_text(command.get("output_run"), field="output_run")
        result = result_by_node.get(node_id)
        if not isinstance(result, Mapping) or result.get("status") != "complete":
            raise RuntimeError(
                f"Execution node {node_id!r} is not complete in receipt {path}"
            )
        if str(result.get("run_name") or "") != run_name:
            raise RuntimeError(
                f"Execution node {node_id!r} run identity differs from its command"
            )
        requests.append(
            RequestedPublication(
                zarr_path=zarr_path,
                stage_id=stage_id,
                requested_run=(
                    f"offline/{run_name}"
                    if stage_id == "track_kinematics"
                    else run_name
                ),
                receipt_path=path,
                receipt_sha256=digest,
            )
        )
    return requests


def _parse_stage_selector(raw: str) -> tuple[str, str]:
    stage_id, separator, run_name = str(raw).partition("=")
    stage_id = stage_id.strip()
    run_name = run_name.strip()
    if not separator or stage_id not in _SUPPORTED_STAGES or not run_name:
        allowed = ", ".join(sorted(_SUPPORTED_STAGES))
        raise ValueError(
            f"--stage-selector must be STAGE=RUN for one of {allowed}: {raw!r}"
        )
    return stage_id, run_name


def _target_receipt_requests(
    target_list: Path,
    status_dir: Path,
    stage_selectors: Sequence[str],
) -> list[RequestedPublication]:
    targets = [
        Path(line.strip()).expanduser().resolve()
        for line in target_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not targets:
        raise ValueError(f"Target list is empty: {target_list}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"Target list contains duplicate Zarr paths: {target_list}")
    selectors = [_parse_stage_selector(raw) for raw in stage_selectors]
    if not selectors:
        raise ValueError("At least one --stage-selector is required")

    receipts_by_target: dict[Path, tuple[Path, str, dict[str, Any]]] = {}
    for receipt_path in sorted(status_dir.glob("*.json")):
        payload, digest = _read_json_object(receipt_path)
        if payload.get("schema_id") != TARGET_RECEIPT_SCHEMA:
            continue
        if payload.get("schema_version") != 1 or payload.get("status") != "complete":
            raise RuntimeError(f"Target receipt is not complete: {receipt_path}")
        if payload.get("registry_write_mode") != "deferred_to_serial_finalizer":
            raise RuntimeError(
                f"Target receipt did not defer registry writes: {receipt_path}"
            )
        zarr_path = (
            Path(_require_text(payload.get("zarr_path"), field="zarr_path"))
            .expanduser()
            .resolve()
        )
        if zarr_path in receipts_by_target:
            raise RuntimeError(f"Duplicate target receipt for {zarr_path}")
        receipts_by_target[zarr_path] = (receipt_path, digest, payload)

    expected = set(targets)
    actual = set(receipts_by_target)
    if expected != actual:
        missing = sorted(str(path) for path in expected - actual)
        unexpected = sorted(str(path) for path in actual - expected)
        raise RuntimeError(
            "Target completion receipts do not exactly match the manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )

    requests: list[RequestedPublication] = []
    for zarr_path in targets:
        receipt_path, digest, payload = receipts_by_target[zarr_path]
        requested_publications = payload.get("requested_publications")
        if not isinstance(requested_publications, Mapping):
            raise ValueError(
                f"Target receipt lacks requested_publications: {receipt_path}"
            )
        for stage_id, run_name in selectors:
            stage_receipt = requested_publications.get(stage_id)
            if (
                not isinstance(stage_receipt, Mapping)
                or stage_receipt.get("requested") is not True
            ):
                raise RuntimeError(
                    f"Target receipt did not request {stage_id}: {receipt_path}"
                )
            receipt_run_name = str(
                stage_receipt.get("run_name")
                or stage_receipt.get("output_run_name")
                or ""
            ).strip()
            if not receipt_run_name:
                raise RuntimeError(
                    f"Target receipt does not bind an exact {stage_id} output run: "
                    f"{receipt_path}"
                )
            requested_run = receipt_run_name if run_name == "latest" else run_name
            if requested_run != receipt_run_name:
                raise RuntimeError(
                    f"{stage_id} selector differs from the target receipt: "
                    f"selector={run_name!r}, receipt={receipt_run_name!r}"
                )
            requests.append(
                RequestedPublication(
                    zarr_path=zarr_path,
                    stage_id=stage_id,
                    requested_run=requested_run,
                    receipt_path=receipt_path,
                    receipt_sha256=digest,
                )
            )
    return requests


def _prepare(request: RequestedPublication) -> PreparedPublication:
    root = open_zarr_group_direct(request.zarr_path, mode="r")
    if request.stage_id == "eye_angles":
        parent = root.get("analysis/eye_angle_runs")
        if parent is None:
            raise RuntimeError(f"Missing eye-angle parent in {request.zarr_path}")
        selected = resolve_authoritative_run_name(parent)
        requested = (
            selected if request.requested_run == "latest" else request.requested_run
        )
        if not selected or requested != selected:
            raise RuntimeError(
                "Eye-angle receipt does not name the selected canonical run: "
                f"requested={requested!r}, selected={selected!r}, "
                f"zarr={request.zarr_path}"
            )
        if requested not in parent:
            raise RuntimeError(f"Missing selected eye-angle run {requested!r}")
        return PreparedPublication(
            request=request,
            root=root,
            run_group=parent[requested],
            registry_run_name=requested,
            leaf_run_name=requested,
            run_type=None,
        )

    if request.stage_id == "track_kinematics":
        parent = root.get("analysis/track_kinematics_runs")
        if parent is None:
            raise RuntimeError(
                f"Missing track-kinematics parent in {request.zarr_path}"
            )
        selected = resolve_authoritative_run_name(parent)
        requested = (
            selected if request.requested_run == "latest" else request.requested_run
        )
        if not selected or requested != selected:
            raise RuntimeError(
                "Track-kinematics receipt does not name the selected canonical run: "
                f"requested={requested!r}, selected={selected!r}, "
                f"zarr={request.zarr_path}"
            )
        run_type, separator, leaf_name = requested.partition("/")
        if not separator or run_type not in {"online", "offline"} or not leaf_name:
            raise RuntimeError(f"Invalid selected track run identity: {requested!r}")
        if (
            "/" in leaf_name
            or run_type not in parent
            or leaf_name not in parent[run_type]
        ):
            raise RuntimeError(f"Missing selected track run {requested!r}")
        return PreparedPublication(
            request=request,
            root=root,
            run_group=parent[run_type][leaf_name],
            registry_run_name=requested,
            leaf_run_name=leaf_name,
            run_type=run_type,
        )

    contract = DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE.get(request.stage_id)
    if contract is None:
        raise RuntimeError(f"Unsupported derived-analysis stage: {request.stage_id}")
    parent = root.get(contract.run_parent)
    if parent is None:
        raise RuntimeError(
            f"Missing {request.stage_id} parent {contract.run_parent!r} "
            f"in {request.zarr_path}"
        )
    selected = resolve_authoritative_run_name(parent)
    requested = selected if request.requested_run == "latest" else request.requested_run
    if not selected or requested != selected:
        raise RuntimeError(
            f"{request.stage_id} receipt does not name the selected canonical run: "
            f"requested={requested!r}, selected={selected!r}, "
            f"zarr={request.zarr_path}"
        )
    if not requested or "/" in requested or requested in {".", ".."}:
        raise RuntimeError(
            f"Invalid selected {request.stage_id} run identity: {requested!r}"
        )
    if requested not in parent:
        raise RuntimeError(f"Missing selected {request.stage_id} run {requested!r}")
    return PreparedPublication(
        request=request,
        root=root,
        run_group=parent[requested],
        registry_run_name=requested,
        leaf_run_name=requested,
        run_type=None,
    )


def _deduplicate(
    requests: Sequence[RequestedPublication],
) -> list[RequestedPublication]:
    by_key: dict[tuple[Path, str], RequestedPublication] = {}
    for request in requests:
        key = (request.zarr_path.expanduser().resolve(), request.stage_id)
        prior = by_key.get(key)
        if prior is not None and prior.requested_run != request.requested_run:
            raise RuntimeError(
                f"Conflicting receipts for {key}: "
                f"{prior.requested_run!r} versus {request.requested_run!r}"
            )
        by_key[key] = request
    return [
        by_key[key] for key in sorted(by_key, key=lambda item: (str(item[0]), item[1]))
    ]


def finalize_registry(
    requests: Sequence[RequestedPublication],
    *,
    registry_path: Path,
    apply: bool,
) -> dict[str, Any]:
    prepared = [_prepare(request) for request in _deduplicate(requests)]
    results: list[dict[str, Any]] = []
    registry: Registry | None = None
    previous_disable = os.environ.pop(_DISABLE_REGISTRY_WRITES_ENV, None)
    try:
        if apply:
            if not registry_path.is_file():
                raise FileNotFoundError(f"Registry does not exist: {registry_path}")
            registry = Registry(registry_path)
        for item in prepared:
            wrote = False
            if apply:
                assert registry is not None
                if item.request.stage_id == "eye_angles":
                    wrote = emit_eye_angle_stage_completion(
                        item.root,
                        item.request.zarr_path,
                        run_group=item.run_group,
                        run_name=item.leaf_run_name,
                        source="serial_derived_analysis_registry_finalizer",
                        registry=registry,
                        invalidate_on_ok=True,
                    )
                elif item.request.stage_id == "track_kinematics":
                    assert item.run_type is not None
                    wrote = emit_track_kinematics_stage_completion(
                        item.root,
                        item.request.zarr_path,
                        run_group=item.run_group,
                        run_name=item.leaf_run_name,
                        run_type=item.run_type,
                        source="serial_derived_analysis_registry_finalizer",
                        registry=registry,
                        invalidate_on_ok=True,
                    )
                else:
                    wrote = emit_derived_analysis_stage_completion(
                        item.root,
                        item.request.zarr_path,
                        stage_id=item.request.stage_id,
                        run_group=item.run_group,
                        run_name=item.leaf_run_name,
                        source="serial_derived_analysis_registry_finalizer",
                        registry=registry,
                        invalidate_on_ok=True,
                    )
                if not wrote:
                    raise RuntimeError(
                        f"Registry publication returned false for "
                        f"{item.request.stage_id} {item.registry_run_name}"
                    )
            results.append(
                {
                    "zarr_path": str(item.request.zarr_path),
                    "stage_id": item.request.stage_id,
                    "run_name": item.registry_run_name,
                    "receipt_path": str(item.request.receipt_path),
                    "receipt_sha256": item.request.receipt_sha256,
                    "status": "published" if wrote else "validated",
                }
            )
        integrity = "not_checked"
        if registry is not None:
            integrity = str(
                registry.conn.execute("PRAGMA integrity_check;").fetchone()[0]
            )
            if integrity != "ok":
                raise RuntimeError(f"Registry integrity_check failed: {integrity}")
    finally:
        if registry is not None:
            registry.close()
        if previous_disable is not None:
            os.environ[_DISABLE_REGISTRY_WRITES_ENV] = previous_disable

    return {
        "schema_id": REPORT_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "mode": "apply" if apply else "dry_run",
        "completed_at_utc": utc_now(),
        "registry": str(registry_path),
        "registry_integrity": integrity,
        "publication_count": len(results),
        "publications": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--execution-report", action="append", type=Path)
    source.add_argument("--target-list", type=Path)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--stage-selector", action="append", default=[])
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.execution_report:
        if args.status_dir is not None or args.stage_selector:
            raise ValueError(
                "--status-dir/--stage-selector are only valid with --target-list"
            )
        requests = [
            request
            for path in args.execution_report
            for request in _execution_report_requests(path.expanduser().resolve())
        ]
    else:
        if args.status_dir is None:
            raise ValueError("--target-list requires --status-dir")
        requests = _target_receipt_requests(
            args.target_list.expanduser().resolve(),
            args.status_dir.expanduser().resolve(),
            args.stage_selector,
        )
    report = finalize_registry(
        requests,
        registry_path=args.registry.expanduser().resolve(),
        apply=bool(args.apply),
    )
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite finalizer report: {output_path}"
            )
        write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA",
    "TARGET_RECEIPT_SCHEMA",
    "RequestedPublication",
    "finalize_registry",
    "main",
]
