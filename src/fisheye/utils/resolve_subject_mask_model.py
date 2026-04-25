#!/usr/bin/env python3
"""Resolve subject-mask U-Net checkpoints from registry model metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.batch_logging import utc_now


def _norm_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return [value] if value.strip() else []
    else:
        parsed = value
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes, bytearray)):
        return []
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class SubjectMaskModelCandidate:
    run_id: str
    set_id: Optional[str]
    model_path: str
    status: Optional[str]
    label_schema_id: Optional[str]
    coverage_class: Optional[str]
    component_coverage_key: Optional[str]
    mask_labels: tuple[str, ...]
    component_groups: tuple[str, ...]
    best_metric_name: Optional[str]
    best_metric_value: Optional[float]
    best_epoch: Optional[int]
    created_utc: Optional[str]
    model_sha256: Optional[str]
    metrics_path: Optional[str]
    metrics_sha256: Optional[str]
    metadata: dict[str, Any]


def _candidate_payload(candidate: SubjectMaskModelCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["mask_labels"] = list(candidate.mask_labels)
    payload["component_groups"] = list(candidate.component_groups)
    return payload


def load_subject_mask_model_candidates(
    registry: Registry,
    *,
    coverage_class: Optional[str] = "dense_all_components",
    component_coverage_key: Optional[str] = None,
    label_schema_id: Optional[str] = None,
    include_non_success: bool = False,
    require_existing_path: bool = True,
) -> list[SubjectMaskModelCandidate]:
    """Load ranked subject-mask model candidates from the registry view."""

    sql = [
        "SELECT",
        "  run_id, set_id, status, task_type, model_path, model_sha256,",
        "  metrics_path, metrics_sha256, label_schema_id, coverage_class,",
        "  component_coverage_key, mask_labels_json, component_groups_json,",
        "  best_metric_name, best_metric_value, best_epoch, created_utc, metadata_json",
        "FROM subject_mask_training_models",
        "WHERE model_path IS NOT NULL AND TRIM(model_path) <> ''",
    ]
    params: list[Any] = []
    if not include_non_success:
        sql.append("AND LOWER(COALESCE(status, '')) = 'success'")
    if coverage_class:
        sql.append("AND coverage_class = ?")
        params.append(str(coverage_class))
    if component_coverage_key:
        sql.append("AND component_coverage_key = ?")
        params.append(str(component_coverage_key))
    if label_schema_id:
        sql.append("AND label_schema_id = ?")
        params.append(str(label_schema_id))
    sql.append(
        "ORDER BY COALESCE(best_metric_value, -1.0) DESC, "
        "COALESCE(best_epoch, -1) DESC, COALESCE(created_utc, '') DESC, run_id DESC"
    )

    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    candidates: list[SubjectMaskModelCandidate] = []
    for row in rows:
        model_path = str(row["model_path"])
        if require_existing_path and not Path(model_path).expanduser().exists():
            continue
        candidates.append(
            SubjectMaskModelCandidate(
                run_id=str(row["run_id"]),
                set_id=_norm_text(row["set_id"]),
                model_path=model_path,
                status=_norm_text(row["status"]),
                label_schema_id=_norm_text(row["label_schema_id"]),
                coverage_class=_norm_text(row["coverage_class"]),
                component_coverage_key=_norm_text(row["component_coverage_key"]),
                mask_labels=tuple(_json_list(row["mask_labels_json"])),
                component_groups=tuple(_json_list(row["component_groups_json"])),
                best_metric_name=_norm_text(row["best_metric_name"]),
                best_metric_value=_as_float(row["best_metric_value"]),
                best_epoch=_as_int(row["best_epoch"]),
                created_utc=_norm_text(row["created_utc"]),
                model_sha256=_norm_text(row["model_sha256"]),
                metrics_path=_norm_text(row["metrics_path"]),
                metrics_sha256=_norm_text(row["metrics_sha256"]),
                metadata=_json_dict(row["metadata_json"]),
            )
        )
    return candidates


def resolve_best_subject_mask_model(
    registry: Registry,
    *,
    coverage_class: Optional[str] = "dense_all_components",
    component_coverage_key: Optional[str] = None,
    label_schema_id: Optional[str] = None,
    include_non_success: bool = False,
    require_existing_path: bool = True,
    require_unique: bool = False,
) -> tuple[SubjectMaskModelCandidate, list[SubjectMaskModelCandidate]]:
    candidates = load_subject_mask_model_candidates(
        registry,
        coverage_class=coverage_class,
        component_coverage_key=component_coverage_key,
        label_schema_id=label_schema_id,
        include_non_success=include_non_success,
        require_existing_path=require_existing_path,
    )
    if not candidates:
        raise SystemExit("No subject-mask model candidates found.")
    if require_unique and len(candidates) > 1:
        best_score = candidates[0].best_metric_value
        next_score = candidates[1].best_metric_value
        if best_score == next_score:
            raise SystemExit(
                "Top subject-mask model metric is tied. Re-run with a stricter filter."
            )
    return candidates[0], candidates


def build_resolution_payload(
    *,
    registry_path: Path,
    selected: SubjectMaskModelCandidate,
    candidates: Sequence[SubjectMaskModelCandidate],
    top_k: int,
    parameters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "mode": "registry",
        "task": "subject_masks",
        "registry_path": str(registry_path),
        "resolved_at_utc": utc_now(),
        "selected": _candidate_payload(selected),
        "candidates": [
            _candidate_payload(candidate)
            for candidate in candidates[: max(0, int(top_k))]
        ],
        "parameters": dict(parameters or {}),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--coverage-class", default="dense_all_components")
    parser.add_argument("--component-coverage-key")
    parser.add_argument("--label-schema-id")
    parser.add_argument("--include-non-success", action="store_true")
    parser.add_argument("--allow-missing-path", action="store_true")
    parser.add_argument("--require-unique", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        best, candidates = resolve_best_subject_mask_model(
            registry,
            coverage_class=args.coverage_class,
            component_coverage_key=args.component_coverage_key,
            label_schema_id=args.label_schema_id,
            include_non_success=bool(args.include_non_success),
            require_existing_path=not bool(args.allow_missing_path),
            require_unique=bool(args.require_unique),
        )
    finally:
        registry.close()

    payload = build_resolution_payload(
        registry_path=registry_path,
        selected=best,
        candidates=candidates,
        top_k=int(args.top_k),
        parameters={
            "coverage_class": args.coverage_class,
            "component_coverage_key": args.component_coverage_key,
            "label_schema_id": args.label_schema_id,
            "include_non_success": bool(args.include_non_success),
            "require_existing_path": not bool(args.allow_missing_path),
            "require_unique": bool(args.require_unique),
        },
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("Recommended subject-mask model")
    print(f"  run_id: {best.run_id}")
    print(f"  set_id: {best.set_id or '-'}")
    print(f"  status: {best.status or '-'}")
    print(f"  label_schema_id: {best.label_schema_id or '-'}")
    print(f"  coverage_class: {best.coverage_class or '-'}")
    print(f"  component_coverage_key: {best.component_coverage_key or '-'}")
    print(f"  {best.best_metric_name or 'metric'}: {best.best_metric_value}")
    print(f"  best_epoch: {best.best_epoch}")
    print(f"  model_path: {best.model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
