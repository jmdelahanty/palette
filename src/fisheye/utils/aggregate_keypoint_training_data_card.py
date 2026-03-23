#!/usr/bin/env python3
"""Aggregate a keypoint training data card from manifest + registry metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.registry.db import Registry, RegistryPaths

try:  # Optional until plotting utility lands.
    from fisheye.utils import plot_keypoint_training_data_card as plot_data_card
except Exception:  # pragma: no cover - optional dependency
    plot_data_card = None  # type: ignore


SCHEMA_NAME = "keypoint_training_data_card"
SCHEMA_VERSION = "v1"
COMPOSITION_FIELDS = (
    "rig_id",
    "camera_id",
    "arena_id",
    "dish_design",
    "canvas_name",
    "protocol_name",
    "keypoint_method",
)
GEOMETRY_ARRAY_NAMES = ("triangle_area", "min_angle", "heading")
REFINED_PARENT_NAMES = ("refined_keypoints_runs", "keypoints_refined_runs")
LANDMARK_HEATMAP_GRID_H = 32
LANDMARK_HEATMAP_GRID_W = 32
LANDMARK_EDGE_MARGIN_NORM = 0.05


@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    zarr_path: Path
    manifest_row: dict[str, Any]
    resolved_by_registry: bool


@dataclass(frozen=True)
class SubjectLineageCoverage:
    manifest_dataset_count: int
    lineage_covered_dataset_count: Optional[int]
    missing_lineage_dataset_ids: tuple[str, ...]
    coverage_unavailable_reason: Optional[str]


@dataclass(frozen=True)
class PoseSchema:
    kpt_shape: Optional[tuple[int, ...]]
    keypoint_labels: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...]


_utc_now = utc_now


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    text = _normalize_text(value)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, Mapping):
        return payload
    return None


def _numeric_stats(values: Sequence[Optional[float]]) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _numeric_histogram(
    values: Sequence[float],
    *,
    bins: int = 32,
) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    bins = max(1, int(bins))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v <= min_v:
        edges = np.asarray([min_v - 0.5, max_v + 0.5], dtype=np.float64)
        counts = np.asarray([int(arr.size)], dtype=np.int64)
    else:
        edges = np.linspace(min_v, max_v, bins + 1, dtype=np.float64)
        counts, _ = np.histogram(arr, bins=edges)
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.astype(np.int64).tolist()],
        "source_value_count": int(arr.size),
    }


def _weighted_mean(values: Sequence[tuple[Optional[float], Optional[float]]]) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in values:
        if value is None:
            continue
        effective_weight = float(weight) if weight is not None and float(weight) > 0 else 1.0
        weighted_sum += float(value) * effective_weight
        total_weight += effective_weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _parse_iso_ts(value: Any) -> datetime:
    text = _normalize_text(value)
    if text is None:
        return datetime.min
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _default_output_path(manifest_path: Path, set_id: Optional[str]) -> Path:
    base = _normalize_manifest_stem(set_id or manifest_path.stem) or "keypoint_training_data_card"
    return manifest_path.parent / f"{base}.data_card.json"


def _default_plot_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.plots"


def _add_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([str(flag), str(value)])


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    rows = payload.get("datasets")
    merged_export = payload.get("merged_export")
    has_source_rows = (
        isinstance(merged_export, Mapping)
        and isinstance(merged_export.get("source_datasets"), list)
        and bool(merged_export.get("source_datasets"))
    )
    if (not isinstance(rows, list) or not rows) and not has_source_rows:
        raise ValueError(f"Manifest has no datasets: {path}")
    return dict(payload)


def _manifest_source_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged_export = manifest.get("merged_export")
    if isinstance(merged_export, Mapping):
        source_rows = merged_export.get("source_datasets")
        if isinstance(source_rows, list) and source_rows:
            rows: list[dict[str, Any]] = []
            for row in source_rows:
                if isinstance(row, Mapping):
                    rows.append(dict(row))
            if rows:
                return rows
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in datasets:
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _resolve_dataset_id_from_registry(registry: Registry, zarr_path: Path) -> Optional[str]:
    candidates = [str(zarr_path)]
    try:
        resolved = str(zarr_path.resolve())
    except OSError:
        resolved = None
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    for candidate in candidates:
        row = registry.conn.execute(
            "SELECT dataset_id FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (candidate,),
        ).fetchone()
        if row is not None:
            return _normalize_text(row["dataset_id"])
    return None


def _manifest_dataset_refs(registry: Registry, manifest: Mapping[str, Any]) -> list[DatasetRef]:
    refs: list[DatasetRef] = []
    for row in _manifest_source_rows(manifest):
        zarr_path_text = _normalize_text(row.get("zarr_path"))
        if zarr_path_text is None:
            raise ValueError("Manifest dataset row missing zarr_path.")
        zarr_path = Path(zarr_path_text)
        resolved_dataset_id = _resolve_dataset_id_from_registry(registry, zarr_path)
        dataset_id = resolved_dataset_id or _normalize_text(row.get("dataset_id")) or _normalize_text(row.get("session_uuid"))
        if dataset_id is None:
            raise ValueError(f"Unable to resolve dataset_id for zarr path: {zarr_path}")
        refs.append(
            DatasetRef(
                dataset_id=dataset_id,
                zarr_path=zarr_path,
                manifest_row=dict(row),
                resolved_by_registry=resolved_dataset_id is not None,
            )
        )
    if not refs:
        raise ValueError("Manifest contains no usable dataset rows.")
    return refs


def _query_subject_lineage_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not dataset_ids:
        return []
    placeholders = ", ".join("?" for _ in dataset_ids)
    sql = (
        "SELECT dataset_id, genotype, dpf_at_acquisition "
        "FROM recording_subject_overview "
        f"WHERE dataset_id IN ({placeholders});"
    )
    rows = registry.conn.execute(sql, tuple(dataset_ids)).fetchall()
    return [dict(row) for row in rows]


def _evaluate_subject_lineage_coverage(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
    subject_lineage_policy: str,
) -> tuple[SubjectLineageCoverage, list[dict[str, Any]]]:
    policy = _normalize_text(subject_lineage_policy) or "warn"
    if policy not in {"warn", "require"}:
        raise ValueError(f"Unsupported subject lineage policy: {subject_lineage_policy}")

    unique_dataset_ids = list(dict.fromkeys(dataset_ids))
    total_count = len(unique_dataset_ids)
    try:
        lineage_rows = _query_subject_lineage_rows(registry, dataset_ids=unique_dataset_ids)
    except Exception as exc:
        if policy == "require":
            raise ValueError(
                f"Subject lineage coverage unavailable (recording_subject_overview query failed): {exc}"
            ) from exc
        reason = f"recording_subject_overview query failed: {exc}"
        print(f"Subject lineage coverage: unavailable ({reason})")
        return (
            SubjectLineageCoverage(
                manifest_dataset_count=int(total_count),
                lineage_covered_dataset_count=None,
                missing_lineage_dataset_ids=tuple(sorted(unique_dataset_ids)),
                coverage_unavailable_reason=reason,
            ),
            [],
        )

    covered_ids = {
        str(row["dataset_id"]).strip()
        for row in lineage_rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    missing_ids = sorted(dataset_id for dataset_id in unique_dataset_ids if dataset_id not in covered_ids)
    covered_count = total_count - len(missing_ids)
    print(f"Subject lineage coverage: {covered_count}/{total_count} datasets")
    if missing_ids:
        print("Subject lineage missing dataset_id(s): " + ", ".join(missing_ids))
    if policy == "require" and missing_ids:
        raise ValueError(
            "Missing subject lineage rows in recording_subject_overview for dataset_id(s): "
            + ", ".join(missing_ids)
        )
    return (
        SubjectLineageCoverage(
            manifest_dataset_count=int(total_count),
            lineage_covered_dataset_count=int(covered_count),
            missing_lineage_dataset_ids=tuple(missing_ids),
            coverage_unavailable_reason=None,
        ),
        lineage_rows,
    )


def _build_subject_coverage_payload(coverage: SubjectLineageCoverage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "manifest_dataset_count": int(coverage.manifest_dataset_count),
        "lineage_covered_dataset_count": (
            int(coverage.lineage_covered_dataset_count)
            if coverage.lineage_covered_dataset_count is not None
            else None
        ),
        "missing_lineage_dataset_ids": list(coverage.missing_lineage_dataset_ids),
    }
    if coverage.coverage_unavailable_reason:
        payload["coverage_unavailable_reason"] = str(coverage.coverage_unavailable_reason)
    return payload


def _build_genotype_counts(lineage_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in lineage_rows:
        genotype = _normalize_text(row.get("genotype"))
        if genotype is not None:
            counter[genotype] += 1
    return {key: int(counter[key]) for key in sorted(counter)}


def _build_dpf_histogram(
    dpf_values: Sequence[Optional[float]],
    *,
    source_dataset_count: int,
) -> Optional[dict[str, Any]]:
    arr = np.asarray([float(v) for v in dpf_values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    low = int(np.floor(float(np.min(arr))))
    high = int(np.ceil(float(np.max(arr))))
    if high < low:
        return None
    edges = np.arange(low - 0.5, high + 1.5, 1.0, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=edges)
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.astype(np.int64).tolist()],
        "source_dataset_count": int(arr.size),
        "skipped_missing_values": int(max(0, int(source_dataset_count) - int(arr.size))),
    }


def _query_dataset_context_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not dataset_ids:
        return {}
    placeholders = ", ".join("?" for _ in dataset_ids)
    sql = (
        "SELECT dataset_id, rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name "
        "FROM dataset_context_current "
        f"WHERE dataset_id IN ({placeholders});"
    )
    rows = registry.conn.execute(sql, tuple(dataset_ids)).fetchall()
    payload: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = _normalize_text(row["dataset_id"])
        if dataset_id is None:
            continue
        payload[dataset_id] = dict(row)
    return payload


def _normalize_kpt_shape(value: Any) -> Optional[tuple[int, ...]]:
    if not isinstance(value, (list, tuple)):
        return None
    normalized: list[int] = []
    for item in value:
        try:
            normalized.append(int(item))
        except Exception:
            return None
    if not normalized:
        return None
    return tuple(normalized)


def _normalize_labels(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    labels: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if text is not None:
            labels.append(text)
    return tuple(labels)


def _normalize_skeleton_edges(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    dedup: set[tuple[int, int]] = set()
    for edge in value:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        a = _as_int(edge[0])
        b = _as_int(edge[1])
        if a is None or b is None or a == b:
            continue
        i = int(min(a, b))
        j = int(max(a, b))
        dedup.add((i, j))
    return tuple(sorted(dedup))


def _parse_pose_schema_mapping(value: Any) -> PoseSchema:
    if not isinstance(value, Mapping):
        return PoseSchema(kpt_shape=None, keypoint_labels=(), skeleton=())
    return PoseSchema(
        kpt_shape=_normalize_kpt_shape(value.get("kpt_shape")),
        keypoint_labels=_normalize_labels(value.get("keypoint_labels")),
        skeleton=_normalize_skeleton_edges(value.get("skeleton")),
    )


def _pose_schema_identity_key(schema: PoseSchema) -> Optional[str]:
    if schema.kpt_shape is None and not schema.skeleton:
        return None
    payload = {
        "kpt_shape": list(schema.kpt_shape) if schema.kpt_shape is not None else None,
        "skeleton": [[int(i), int(j)] for i, j in schema.skeleton],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _resolved_pose_schema(manifest_schema: PoseSchema, observed_schema: PoseSchema) -> PoseSchema:
    return PoseSchema(
        kpt_shape=manifest_schema.kpt_shape or observed_schema.kpt_shape,
        keypoint_labels=manifest_schema.keypoint_labels or observed_schema.keypoint_labels,
        skeleton=manifest_schema.skeleton or observed_schema.skeleton,
    )


def _pose_schema_keypoint_count(schema: PoseSchema) -> Optional[int]:
    if schema.kpt_shape is not None and len(schema.kpt_shape) >= 1:
        count = _as_int(schema.kpt_shape[0])
        if count is not None and count > 0:
            return int(count)
    if schema.keypoint_labels:
        return int(len(schema.keypoint_labels))
    if schema.skeleton:
        return int(max(max(edge) for edge in schema.skeleton) + 1)
    return None


def _pose_schema_conflicts(expected: PoseSchema, observed: PoseSchema) -> bool:
    expected_count = _pose_schema_keypoint_count(expected)
    observed_count = _pose_schema_keypoint_count(observed)
    if expected_count is not None and observed_count is not None and expected_count != observed_count:
        return True
    if expected.skeleton and observed.skeleton and expected.skeleton != observed.skeleton:
        return True
    return False


def _pose_schema_info_score(schema: PoseSchema) -> tuple[int, int, int]:
    return (
        len(schema.skeleton),
        len(schema.keypoint_labels),
        len(schema.kpt_shape or ()),
    )


def _extract_pose_schema_from_dataset_root(
    root: Any,
    *,
    manifest_row: Mapping[str, Any],
    quality_row: Optional[Mapping[str, Any]],
) -> tuple[PoseSchema, Optional[str]]:
    keypoints_parent = root.get("keypoints_runs")
    if keypoints_parent is None:
        return PoseSchema(kpt_shape=None, keypoint_labels=(), skeleton=()), None

    requested_run = (
        _normalize_text(quality_row.get("source_keypoint_run")) if quality_row is not None else None
    ) or _normalize_text(manifest_row.get("keypoint_run_resolved")) or _normalize_text(manifest_row.get("keypoint_run"))
    keypoint_run = requested_run
    if keypoint_run is None or keypoint_run not in keypoints_parent:
        latest = _normalize_text(keypoints_parent.attrs.get("latest"))
        if latest and latest in keypoints_parent:
            keypoint_run = latest
    if keypoint_run is None:
        group_keys = list(keypoints_parent.group_keys()) if hasattr(keypoints_parent, "group_keys") else []
        if group_keys:
            keypoint_run = str(sorted(group_keys)[-1])
    if keypoint_run is None or keypoint_run not in keypoints_parent:
        return PoseSchema(kpt_shape=None, keypoint_labels=(), skeleton=()), None

    keypoint_group = keypoints_parent[keypoint_run]
    pose_schema_attr = _as_mapping(keypoint_group.attrs.get("pose_schema"))
    kpt_shape = _normalize_kpt_shape(keypoint_group.attrs.get("kpt_shape"))
    if kpt_shape is None and pose_schema_attr is not None:
        kpt_shape = _normalize_kpt_shape(pose_schema_attr.get("kpt_shape"))
    if kpt_shape is None and "keypoints_roi" in keypoint_group:
        shape = tuple(int(v) for v in keypoint_group["keypoints_roi"].shape[1:])
        if shape:
            kpt_shape = shape
    labels = _normalize_labels(keypoint_group.attrs.get("keypoint_labels"))
    if not labels and pose_schema_attr is not None:
        labels = _normalize_labels(pose_schema_attr.get("nodes"))
    skeleton = _normalize_skeleton_edges(keypoint_group.attrs.get("keypoint_skeleton"))
    if not skeleton and pose_schema_attr is not None:
        skeleton = _normalize_skeleton_edges(pose_schema_attr.get("edges"))
    return PoseSchema(kpt_shape=kpt_shape, keypoint_labels=labels, skeleton=skeleton), keypoint_run


def _open_zarr_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r", consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def _enforce_single_skeleton(
    *,
    dataset_refs: Sequence[DatasetRef],
    registry_quality_rows: Mapping[str, Mapping[str, Any]],
    manifest_schema: PoseSchema,
) -> PoseSchema:
    observed_identities: dict[str, str] = {}
    observed_schemas: dict[str, PoseSchema] = {}
    expected_key = _pose_schema_identity_key(manifest_schema)

    mismatches: list[str] = []
    for ref in dataset_refs:
        try:
            root = _open_zarr_group(ref.zarr_path)
        except Exception:
            continue
        schema, _ = _extract_pose_schema_from_dataset_root(
            root,
            manifest_row=ref.manifest_row,
            quality_row=registry_quality_rows.get(ref.dataset_id),
        )
        key = _pose_schema_identity_key(schema)
        if key is None:
            continue
        observed_identities[ref.dataset_id] = key
        observed_schemas[ref.dataset_id] = schema
        if expected_key is not None and key != expected_key and _pose_schema_conflicts(manifest_schema, schema):
            mismatches.append(ref.dataset_id)

    if expected_key is not None and mismatches:
        raise ValueError(
            "Mixed skeleton identities in manifest datasets: expected manifest pose_schema identity, "
            "mismatched dataset_id(s): " + ", ".join(sorted(set(mismatches)))
        )

    if expected_key is None and observed_schemas:
        observed_counts = {
            count
            for count in (_pose_schema_keypoint_count(schema) for schema in observed_schemas.values())
            if count is not None
        }
        if len(observed_counts) > 1:
            by_dataset = ", ".join(
                f"{dataset_id}:{_pose_schema_keypoint_count(schema)}"
                for dataset_id, schema in sorted(observed_schemas.items())
            )
            raise ValueError(f"Mixed skeleton identities across manifest datasets (keypoint_count): {by_dataset}")
        observed_edges = {schema.skeleton for schema in observed_schemas.values() if schema.skeleton}
        if len(observed_edges) > 1:
            by_dataset = ", ".join(
                f"{dataset_id}:{schema.skeleton}" for dataset_id, schema in sorted(observed_schemas.items())
            )
            raise ValueError(f"Mixed skeleton identities across manifest datasets (skeleton_edges): {by_dataset}")

    if observed_schemas:
        return max(observed_schemas.values(), key=_pose_schema_info_score)
    return manifest_schema


def _sanitize_alias_token(value: str) -> str:
    cleaned = []
    for ch in value.strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    token = "".join(cleaned).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "kpt"


def _build_graph_metric_specs(pose_schema: PoseSchema) -> dict[str, list[dict[str, Any]]]:
    edge_specs: list[dict[str, Any]] = []
    labels = list(pose_schema.keypoint_labels)
    for i, j in pose_schema.skeleton:
        key = f"edge_{int(i)}_{int(j)}"
        alias: Optional[str] = None
        if int(i) < len(labels) and int(j) < len(labels):
            alias = "edge_" + "_".join(
                (_sanitize_alias_token(labels[int(i)]), _sanitize_alias_token(labels[int(j)]))
            )
        edge_specs.append({"key": key, "alias": alias, "indices": (int(i), int(j))})

    adjacency: dict[int, set[int]] = {}
    for i, j in pose_schema.skeleton:
        adjacency.setdefault(int(i), set()).add(int(j))
        adjacency.setdefault(int(j), set()).add(int(i))
    angle_specs: list[dict[str, Any]] = []
    for center in sorted(adjacency):
        neighbors = sorted(adjacency[center])
        for idx_a in range(len(neighbors)):
            for idx_b in range(idx_a + 1, len(neighbors)):
                left = int(neighbors[idx_a])
                right = int(neighbors[idx_b])
                key = f"angle_{left}_{int(center)}_{right}"
                alias: Optional[str] = None
                if left < len(labels) and int(center) < len(labels) and right < len(labels):
                    alias = "angle_" + "_".join(
                        (
                            _sanitize_alias_token(labels[left]),
                            _sanitize_alias_token(labels[int(center)]),
                            _sanitize_alias_token(labels[right]),
                        )
                    )
                angle_specs.append(
                    {
                        "key": key,
                        "alias": alias,
                        "indices": (left, int(center), right),
                    }
                )
    return {"edges": edge_specs, "angles": angle_specs}


def _select_quality_rows(
    registry: Registry,
    *,
    dataset_refs: Sequence[DatasetRef],
) -> dict[str, dict[str, Any]]:
    dataset_ids = [ref.dataset_id for ref in dataset_refs]
    rows = registry.query_keypoint_quality_current(dataset_ids=dataset_ids)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload = dict(row)
        dataset_id = _normalize_text(payload.get("dataset_id"))
        if dataset_id is None:
            continue
        by_dataset.setdefault(dataset_id, []).append(payload)

    selected: dict[str, dict[str, Any]] = {}
    for ref in dataset_refs:
        candidates = by_dataset.get(ref.dataset_id, [])
        if not candidates:
            continue
        desired_run = _normalize_text(ref.manifest_row.get("keypoint_run_resolved")) or _normalize_text(
            ref.manifest_row.get("keypoint_run")
        )
        desired_method = _normalize_text(ref.manifest_row.get("quality_registry_keypoint_method")) or _normalize_text(
            ref.manifest_row.get("keypoint_method")
        )

        def _rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
            source_run = _normalize_text(candidate.get("source_keypoint_run"))
            method = _normalize_text(candidate.get("keypoint_method"))
            review_state = _normalize_text(candidate.get("review_state"))
            review_use = _normalize_text(candidate.get("review_intended_use"))
            return (
                1 if desired_run is not None and source_run == desired_run else 0,
                1 if desired_method is not None and method == desired_method else 0,
                1 if review_state == "approved" else 0,
                1 if review_use == "training" else 0,
                _parse_iso_ts(candidate.get("review_timestamp_utc")),
                _parse_iso_ts(candidate.get("refined_created_utc")),
                _normalize_text(candidate.get("refined_run")) or "",
            )

        selected[ref.dataset_id] = max(candidates, key=_rank)
    return selected


def _select_profile_rows(
    registry: Registry,
    *,
    dataset_refs: Sequence[DatasetRef],
    quality_rows_by_dataset: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    dataset_ids = [ref.dataset_id for ref in dataset_refs]
    rows = registry.query_keypoint_data_profile_latest(dataset_ids=dataset_ids)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload = dict(row)
        dataset_id = _normalize_text(payload.get("dataset_id"))
        if dataset_id is None:
            continue
        by_dataset.setdefault(dataset_id, []).append(payload)

    selected: dict[str, dict[str, Any]] = {}
    for ref in dataset_refs:
        candidates = by_dataset.get(ref.dataset_id, [])
        if not candidates:
            continue
        quality_row = quality_rows_by_dataset.get(ref.dataset_id)
        desired_source_run = (
            _normalize_text(quality_row.get("source_keypoint_run")) if quality_row is not None else None
        ) or _normalize_text(ref.manifest_row.get("keypoint_run_resolved")) or _normalize_text(
            ref.manifest_row.get("keypoint_run")
        )
        desired_method = (
            _normalize_text(quality_row.get("keypoint_method")) if quality_row is not None else None
        ) or _normalize_text(ref.manifest_row.get("quality_registry_keypoint_method")) or _normalize_text(
            ref.manifest_row.get("keypoint_method")
        )

        def _rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
            source_run = _normalize_text(candidate.get("source_keypoint_run"))
            method = _normalize_text(candidate.get("keypoint_method"))
            return (
                1 if desired_source_run is not None and source_run == desired_source_run else 0,
                1 if desired_method is not None and method == desired_method else 0,
                _parse_iso_ts(candidate.get("profile_created_utc")),
                _parse_iso_ts(candidate.get("updated_utc")),
                _normalize_text(candidate.get("profile_run")) or "",
            )

        selected[ref.dataset_id] = max(candidates, key=_rank)
    return selected


def _profile_stale_reason(
    profile_row: Mapping[str, Any],
    *,
    zarr_path: Path,
) -> Optional[str]:
    expected_mtime = _as_int(profile_row.get("zarr_mtime_ns"))
    if expected_mtime is None:
        return "missing zarr_mtime_ns"
    if not zarr_path.exists():
        return "zarr missing on disk"
    try:
        observed_mtime = int(zarr_path.stat().st_mtime_ns)
    except OSError:
        return "unable to stat zarr path"
    if observed_mtime != expected_mtime:
        return f"mtime mismatch (registry={expected_mtime}, actual={observed_mtime})"
    return None


def _profile_refresh_remediation_text(registry_path: Path) -> str:
    return (
        "Remediation: run "
        f"'scripts/py -m fisheye.registry.maintenance --registry {registry_path} --refresh-keypoint-profiles', "
        f"then 'scripts/py -m fisheye.utils.check_training_registry --registry {registry_path} "
        "--view keypoint-profile --no-rich', then rerun aggregation/pipeline."
    )


def _resolve_refined_parent(root: Any) -> tuple[Optional[str], Any]:
    for name in REFINED_PARENT_NAMES:
        parent = root.get(name)
        if parent is not None:
            return name, parent
    return None, None


def _resolve_refined_run_name(
    root: Any,
    *,
    keypoint_run: Optional[str],
    manifest_row: Mapping[str, Any],
    quality_row: Optional[Mapping[str, Any]],
) -> Optional[str]:
    parent_name, refined_parent = _resolve_refined_parent(root)
    if parent_name is None or refined_parent is None:
        return None

    requested = (
        _normalize_text(quality_row.get("refined_run")) if quality_row is not None else None
    ) or _normalize_text(manifest_row.get("refined_keypoint_run")) or _normalize_text(
        manifest_row.get("quality_registry_refined_run")
    )
    if requested and requested in refined_parent:
        candidate = refined_parent[requested]
        source_run = _normalize_text(candidate.attrs.get("source_keypoints_run")) or _normalize_text(
            candidate.attrs.get("source_keypoint_run")
        )
        if keypoint_run is None or source_run is None or source_run == keypoint_run:
            return requested

    candidates: list[tuple[datetime, str]] = []
    group_names = list(refined_parent.group_keys()) if hasattr(refined_parent, "group_keys") else []
    for run_name in group_names:
        run_group = refined_parent[run_name]
        source_run = _normalize_text(run_group.attrs.get("source_keypoints_run")) or _normalize_text(
            run_group.attrs.get("source_keypoint_run")
        )
        if keypoint_run is not None and source_run is not None and source_run != keypoint_run:
            continue
        ts = _parse_iso_ts(run_group.attrs.get("created_utc") or run_group.attrs.get("timestamp_utc"))
        candidates.append((ts, str(run_name)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return str(candidates[0][1])


def _resolve_roi_dimensions(root: Any, keypoint_group: Any) -> tuple[int, int]:
    source_crop_run = _normalize_text(keypoint_group.attrs.get("source_crop_run"))
    if source_crop_run is None:
        return (1, 1)
    crop_parent = root.get("crop_runs")
    if crop_parent is None or source_crop_run not in crop_parent:
        return (1, 1)
    crop_group = crop_parent[source_crop_run]
    if "roi_images" not in crop_group:
        return (1, 1)
    shape = tuple(int(v) for v in crop_group["roi_images"].shape)
    if len(shape) < 3:
        return (1, 1)
    roi_h = int(shape[1])
    roi_w = int(shape[2])
    if roi_h <= 0 or roi_w <= 0:
        return (1, 1)
    return (roi_h, roi_w)


def _resolve_roi_diagonal(root: Any, keypoint_group: Any) -> float:
    roi_h, roi_w = _resolve_roi_dimensions(root, keypoint_group)
    if roi_h <= 0 or roi_w <= 0:
        return 1.0
    diag = float(np.sqrt(float(roi_h * roi_h + roi_w * roi_w)))
    return diag if diag > 0 else 1.0


def _selection_row_counts(
    *,
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[Optional[int], Optional[int]]:
    pre_values: list[int] = []
    post_values: list[int] = []
    for row in source_rows:
        pre = _coalesce(
            _as_int(row.get("row_gate_total")),
            _as_int(row.get("source_sample_count")),
            _as_int(row.get("keypoints_total")),
        )
        post = _coalesce(
            _as_int(row.get("row_gate_selected")),
            _as_int(row.get("sample_count")),
            _as_int(row.get("keypoints_total")),
        )
        if pre is not None:
            pre_values.append(int(pre))
        if post is not None:
            post_values.append(int(post))
    rows_pre_gate = int(sum(pre_values)) if pre_values else None
    rows_post_gate = int(sum(post_values)) if post_values else None
    return rows_pre_gate, rows_post_gate


def _manifest_quality_exclusion_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    rows = manifest.get("quality_exclusions")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reason = _normalize_text(row.get("reason"))
        if reason is not None:
            counter[reason] += 1
    return {key: int(counter[key]) for key in sorted(counter)}


def _resolve_split_group(root: Any) -> Any:
    split_group = root.get("splits")
    if split_group is not None:
        return split_group
    return root.get("split")


def _split_counts_from_merged_zarr(path: Path) -> dict[str, int]:
    root = _open_zarr_group(path)
    split_group = _resolve_split_group(root)
    if split_group is None:
        return {}
    counts: dict[str, int] = {}
    if "train_indices" in split_group:
        counts["train"] = int(split_group["train_indices"].shape[0])
    if "val_indices" in split_group:
        counts["val"] = int(split_group["val_indices"].shape[0])
    if "test_indices" in split_group:
        counts["test"] = int(split_group["test_indices"].shape[0])
    return counts


def _split_counts_from_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    merged_export = manifest.get("merged_export")
    if not isinstance(merged_export, Mapping):
        return {}
    counts_payload = merged_export.get("counts")
    if not isinstance(counts_payload, Mapping):
        return {}
    counts: dict[str, int] = {}
    for key in ("train", "val", "test"):
        value = _as_int(counts_payload.get(key))
        if value is not None and value >= 0:
            counts[key] = int(value)
    return counts


def _build_split_counts(
    *,
    merged_zarr: Optional[Path],
    manifest: Mapping[str, Any],
    split_label: str,
    rows_post_gate: Optional[int],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if merged_zarr is not None:
        counts = _split_counts_from_merged_zarr(merged_zarr)
    if not counts:
        counts = _split_counts_from_manifest(manifest)
    if not counts and rows_post_gate is not None:
        counts = {split_label: int(rows_post_gate)}
    return counts


def _decode_string_array(values: np.ndarray) -> list[Optional[str]]:
    flat = np.asarray(values).reshape(-1)
    decoded: list[Optional[str]] = []
    for value in flat.tolist():
        decoded.append(_normalize_text(value))
    return decoded


def _split_dataset_counts_from_merged_zarr(path: Optional[Path]) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    try:
        root = _open_zarr_group(path)
    except Exception:
        return {}

    split_group = _resolve_split_group(root)
    source_index_group = root.get("source_index")
    if split_group is None or source_index_group is None:
        return {}
    if "source_dataset_idx" not in source_index_group or "source_dataset_id" not in source_index_group:
        return {}

    try:
        source_dataset_idx = np.asarray(source_index_group["source_dataset_idx"][:], dtype=np.int64).reshape(-1)
        source_dataset_ids = _decode_string_array(np.asarray(source_index_group["source_dataset_id"][:]))
    except Exception:
        return {}
    if source_dataset_idx.ndim != 1 or not source_dataset_ids:
        return {}

    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        array_name = f"{split_name}_indices"
        if array_name not in split_group:
            continue
        try:
            split_indices = np.asarray(split_group[array_name][:], dtype=np.int64).reshape(-1)
        except Exception:
            continue
        if split_indices.size == 0:
            split_counts[split_name] = {}
            continue

        valid = (split_indices >= 0) & (split_indices < source_dataset_idx.shape[0])
        if not np.any(valid):
            split_counts[split_name] = {}
            continue

        counts: Counter[str] = Counter()
        for dataset_idx in source_dataset_idx[split_indices[valid]].tolist():
            if not isinstance(dataset_idx, (int, np.integer)):
                continue
            idx = int(dataset_idx)
            if idx < 0 or idx >= len(source_dataset_ids):
                continue
            dataset_id = source_dataset_ids[idx]
            if dataset_id is None:
                continue
            counts[dataset_id] += 1
        split_counts[split_name] = {key: int(counts[key]) for key in sorted(counts)}

    return split_counts


def _weighted_metric_from_counts(
    *,
    dataset_counts: Mapping[str, int],
    profile_rows_by_dataset: Mapping[str, Mapping[str, Any]],
    field_name: str,
) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for dataset_id, raw_weight in dataset_counts.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        row = profile_rows_by_dataset.get(dataset_id)
        if row is None:
            continue
        value = _as_float(row.get(field_name))
        if value is None:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _weighted_lineage_dpf(
    *,
    dataset_counts: Mapping[str, int],
    dpf_by_dataset: Mapping[str, Optional[float]],
) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0
    for dataset_id, raw_weight in dataset_counts.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        value = dpf_by_dataset.get(dataset_id)
        if value is None:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _weighted_genotype_fractions(
    *,
    dataset_counts: Mapping[str, int],
    genotype_by_dataset: Mapping[str, Optional[str]],
) -> dict[str, float]:
    counter: Counter[str] = Counter()
    total_weight = 0.0
    for dataset_id, raw_weight in dataset_counts.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        genotype = _normalize_text(genotype_by_dataset.get(dataset_id))
        if genotype is None:
            continue
        counter[genotype] += int(raw_weight)
        total_weight += weight
    if total_weight <= 0:
        return {}
    return {key: float(counter[key]) / total_weight for key in sorted(counter)}


def _max_fraction_delta(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> Optional[float]:
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return None
    return max(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def _build_train_val_parity(
    *,
    merged_zarr: Optional[Path],
    profile_rows_by_dataset: Mapping[str, Mapping[str, Any]],
    lineage_rows: Sequence[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    split_dataset_counts = _split_dataset_counts_from_merged_zarr(merged_zarr)
    train_counts = split_dataset_counts.get("train") or {}
    val_counts = split_dataset_counts.get("val") or {}
    if not train_counts or not val_counts:
        return None

    metric_specs = (
        ("usable_keypoints_rate", "usable_rate"),
        ("confidence_valid_rate", "confidence_valid_rate"),
        ("geometry_valid_rate", "geometry_valid_rate"),
        ("triangle_area_p50", "triangle_area_p50"),
        ("triangle_area_p90", "triangle_area_p90"),
        ("min_angle_p50", "min_angle_p50"),
        ("min_angle_p90", "min_angle_p90"),
    )
    metric_payload: dict[str, dict[str, Optional[float]]] = {}
    for metric_name, field_name in metric_specs:
        train_value = _weighted_metric_from_counts(
            dataset_counts=train_counts,
            profile_rows_by_dataset=profile_rows_by_dataset,
            field_name=field_name,
        )
        val_value = _weighted_metric_from_counts(
            dataset_counts=val_counts,
            profile_rows_by_dataset=profile_rows_by_dataset,
            field_name=field_name,
        )
        metric_payload[metric_name] = {
            "train": float(train_value) if train_value is not None else None,
            "val": float(val_value) if val_value is not None else None,
            "delta": (
                float(abs(train_value - val_value))
                if train_value is not None and val_value is not None
                else None
            ),
        }

    genotype_by_dataset = {
        str(row.get("dataset_id")): _normalize_text(row.get("genotype"))
        for row in lineage_rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    dpf_by_dataset = {
        str(row.get("dataset_id")): _as_float(row.get("dpf_at_acquisition"))
        for row in lineage_rows
        if _normalize_text(row.get("dataset_id")) is not None
    }
    train_genotype_fraction = _weighted_genotype_fractions(
        dataset_counts=train_counts,
        genotype_by_dataset=genotype_by_dataset,
    )
    val_genotype_fraction = _weighted_genotype_fractions(
        dataset_counts=val_counts,
        genotype_by_dataset=genotype_by_dataset,
    )
    train_dpf_mean = _weighted_lineage_dpf(
        dataset_counts=train_counts,
        dpf_by_dataset=dpf_by_dataset,
    )
    val_dpf_mean = _weighted_lineage_dpf(
        dataset_counts=val_counts,
        dpf_by_dataset=dpf_by_dataset,
    )

    return {
        "split_dataset_row_counts": {
            "train": {key: int(train_counts[key]) for key in sorted(train_counts)},
            "val": {key: int(val_counts[key]) for key in sorted(val_counts)},
        },
        "metrics": metric_payload,
        "lineage": {
            "genotype_fraction": {
                "train": train_genotype_fraction,
                "val": val_genotype_fraction,
            },
            "genotype_mix_max_abs_delta": _max_fraction_delta(
                train_genotype_fraction,
                val_genotype_fraction,
            ),
            "dpf_mean": {
                "train": float(train_dpf_mean) if train_dpf_mean is not None else None,
                "val": float(val_dpf_mean) if val_dpf_mean is not None else None,
                "delta": (
                    float(abs(train_dpf_mean - val_dpf_mean))
                    if train_dpf_mean is not None and val_dpf_mean is not None
                    else None
                ),
            },
        },
    }


def _as_finite_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr[np.isfinite(arr)]


def _update_landmark_spatial_accumulators(
    *,
    dataset_id: str,
    keypoints_arr: np.ndarray,
    roi_h: int,
    roi_w: int,
    heatmap_counts: dict[int, np.ndarray],
    edge_hits: dict[int, int],
    total_points: dict[int, int],
    source_dataset_ids: dict[int, set[str]],
    grid_h: int,
    grid_w: int,
    edge_margin_norm: float,
) -> None:
    if keypoints_arr.ndim != 3 or int(keypoints_arr.shape[2]) != 2:
        return
    if grid_h <= 0 or grid_w <= 0:
        return

    x_raw = np.asarray(keypoints_arr[:, :, 0], dtype=np.float64)
    y_raw = np.asarray(keypoints_arr[:, :, 1], dtype=np.float64)
    finite_xy = np.isfinite(x_raw) & np.isfinite(y_raw)

    # Handle both normalized and pixel-space ROI coordinates.
    finite_values = np.concatenate(
        [x_raw[finite_xy], y_raw[finite_xy]],
        axis=0,
    )
    likely_normalized = bool(finite_values.size > 0 and float(np.nanmax(np.abs(finite_values))) <= 1.5)

    if likely_normalized:
        x_norm = np.clip(x_raw, 0.0, 1.0)
        y_norm = np.clip(y_raw, 0.0, 1.0)
    else:
        roi_w_eff = float(roi_w if roi_w > 0 else 1)
        roi_h_eff = float(roi_h if roi_h > 0 else 1)
        x_norm = np.clip(x_raw / roi_w_eff, 0.0, 1.0)
        y_norm = np.clip(y_raw / roi_h_eff, 0.0, 1.0)

    kpt_count = int(keypoints_arr.shape[1])
    for landmark_idx in range(kpt_count):
        valid = finite_xy[:, landmark_idx]
        if not np.any(valid):
            continue

        x_values = x_norm[valid, landmark_idx]
        y_values = y_norm[valid, landmark_idx]
        if x_values.size == 0:
            continue

        hist2d, _, _ = np.histogram2d(
            y_values,
            x_values,
            bins=[int(grid_h), int(grid_w)],
            range=[[0.0, 1.0], [0.0, 1.0]],
        )
        if landmark_idx not in heatmap_counts:
            heatmap_counts[landmark_idx] = np.zeros((int(grid_h), int(grid_w)), dtype=np.float64)
        heatmap_counts[landmark_idx] += hist2d

        edge_mask = (
            (x_values <= edge_margin_norm)
            | (x_values >= (1.0 - edge_margin_norm))
            | (y_values <= edge_margin_norm)
            | (y_values >= (1.0 - edge_margin_norm))
        )
        edge_hits[landmark_idx] = int(edge_hits.get(landmark_idx, 0) + int(np.sum(edge_mask)))
        total_points[landmark_idx] = int(total_points.get(landmark_idx, 0) + int(x_values.size))
        source_dataset_ids.setdefault(landmark_idx, set()).add(str(dataset_id))


def _build_spatial_payload(
    *,
    heatmap_counts: Mapping[int, np.ndarray],
    edge_hits: Mapping[int, int],
    total_points: Mapping[int, int],
    source_dataset_ids: Mapping[int, set[str]],
    pose_schema: PoseSchema,
    edge_margin_norm: float,
) -> Optional[dict[str, Any]]:
    if not heatmap_counts:
        return None

    labels = list(pose_schema.keypoint_labels)
    landmark_payload: dict[str, dict[str, Any]] = {}
    for landmark_idx in sorted(heatmap_counts):
        grid = np.asarray(heatmap_counts[landmark_idx], dtype=np.float64)
        if grid.ndim != 2:
            continue
        total = float(np.sum(grid))
        if total <= 0:
            continue
        density = grid / total
        alias = labels[landmark_idx] if landmark_idx < len(labels) else None
        point_count = int(total_points.get(landmark_idx, 0))
        hit_count = int(edge_hits.get(landmark_idx, 0))
        landmark_payload[str(int(landmark_idx))] = {
            "landmark_index": int(landmark_idx),
            "alias": alias,
            "grid_h": int(grid.shape[0]),
            "grid_w": int(grid.shape[1]),
            "density": [float(v) for v in density.reshape(-1).tolist()],
            "source_sample_count": point_count,
            "source_dataset_count": int(len(source_dataset_ids.get(landmark_idx, set()))),
            "edge_proximity_rate": (
                float(hit_count) / float(point_count)
                if point_count > 0
                else None
            ),
        }
    if not landmark_payload:
        return None

    return {
        "edge_margin_norm": float(edge_margin_norm),
        "landmark_center_heatmaps": landmark_payload,
    }


def _build_keypoint_training_data_card(
    *,
    registry: Registry,
    manifest: Mapping[str, Any],
    merged_zarr: Optional[Path],
    split: str,
    subject_lineage_policy: str,
    allow_profile_mtime_mismatch: bool,
    allow_profile_fallback_scan: bool,
) -> dict[str, Any]:
    source_rows = _manifest_source_rows(manifest)
    refs = _manifest_dataset_refs(registry, manifest)
    dataset_ids = [ref.dataset_id for ref in refs]
    quality_rows_by_dataset = _select_quality_rows(registry, dataset_refs=refs)
    profile_rows_by_dataset = _select_profile_rows(
        registry,
        dataset_refs=refs,
        quality_rows_by_dataset=quality_rows_by_dataset,
    )
    context_rows = _query_dataset_context_rows(registry, dataset_ids=dataset_ids)
    subject_lineage_coverage, lineage_rows = _evaluate_subject_lineage_coverage(
        registry,
        dataset_ids=dataset_ids,
        subject_lineage_policy=subject_lineage_policy,
    )

    manifest_pose_schema = _parse_pose_schema_mapping(manifest.get("pose_schema"))
    observed_pose_schema = _enforce_single_skeleton(
        dataset_refs=refs,
        registry_quality_rows=quality_rows_by_dataset,
        manifest_schema=manifest_pose_schema,
    )
    resolved_pose_schema = _resolved_pose_schema(manifest_pose_schema, observed_pose_schema)
    graph_specs = _build_graph_metric_specs(resolved_pose_schema)

    rows_pre_gate, rows_post_gate = _selection_row_counts(source_rows=source_rows)
    split_counts = _build_split_counts(
        merged_zarr=merged_zarr,
        manifest=manifest,
        split_label=str(split),
        rows_post_gate=rows_post_gate,
    )

    missing_profile_dataset_ids = [
        ref.dataset_id for ref in refs if ref.dataset_id not in profile_rows_by_dataset
    ]
    if missing_profile_dataset_ids and not allow_profile_fallback_scan:
        raise ValueError(
            "Missing keypoint_data_profile_latest rows for dataset_id(s): "
            + ", ".join(sorted(missing_profile_dataset_ids))
            + ". "
            + _profile_refresh_remediation_text(Path(registry.path))
            + " To continue without profile rows, rerun with --allow-profile-fallback-scan."
        )
    if missing_profile_dataset_ids and allow_profile_fallback_scan:
        print(
            "Keypoint profile fallback enabled: missing keypoint_data_profile_latest row(s) for dataset_id(s): "
            + ", ".join(sorted(missing_profile_dataset_ids))
        )

    stale_profile_rows: list[str] = []
    for ref in refs:
        profile_row = profile_rows_by_dataset.get(ref.dataset_id)
        if profile_row is None:
            continue
        stale_reason = _profile_stale_reason(profile_row, zarr_path=ref.zarr_path)
        if stale_reason is not None:
            stale_profile_rows.append(f"{ref.dataset_id}:{stale_reason}")
    if stale_profile_rows and not allow_profile_mtime_mismatch:
        raise ValueError(
            "Stale keypoint_data_profile_latest row(s): "
            + ", ".join(sorted(stale_profile_rows))
            + ". "
            + _profile_refresh_remediation_text(Path(registry.path))
            + " To bypass staleness checks, rerun with --allow-profile-mtime-mismatch."
        )

    composition_rows: list[dict[str, Any]] = []
    geometry_values: dict[str, list[float]] = {name: [] for name in GEOMETRY_ARRAY_NAMES}
    edge_values: dict[str, list[float]] = {spec["key"]: [] for spec in graph_specs["edges"]}
    angle_values: dict[str, list[float]] = {spec["key"]: [] for spec in graph_specs["angles"]}
    landmark_heatmap_counts: dict[int, np.ndarray] = {}
    landmark_edge_hits: dict[int, int] = {}
    landmark_total_points: dict[int, int] = {}
    landmark_source_dataset_ids: dict[int, set[str]] = {}

    usable_totals: list[Optional[float]] = []
    total_keypoints_values: list[Optional[float]] = []
    usable_rate_values: list[Optional[float]] = []
    raw_success_rate_values: list[Optional[float]] = []
    raw_successful_values: list[Optional[float]] = []
    raw_success_pairs: list[tuple[Optional[float], Optional[float]]] = []
    confidence_valid_pairs: list[tuple[Optional[float], Optional[float]]] = []
    geometry_valid_pairs: list[tuple[Optional[float], Optional[float]]] = []

    confidence_valid_true = 0
    confidence_valid_total = 0
    geometry_valid_true = 0
    geometry_valid_total = 0
    flips_corrected_total = 0
    flips_total_rows = 0

    zarr_mtime_mismatch_count = 0
    quality_stale_count = 0
    source_run_refs: list[dict[str, Any]] = []

    for ref in refs:
        quality_row = quality_rows_by_dataset.get(ref.dataset_id)
        profile_row = profile_rows_by_dataset.get(ref.dataset_id)
        dataset_context = context_rows.get(ref.dataset_id, {})
        manifest_row = ref.manifest_row

        keypoint_method = (
            _normalize_text(profile_row.get("keypoint_method")) if profile_row is not None else None
        ) or (
            _normalize_text(quality_row.get("keypoint_method")) if quality_row is not None else None
        ) or _normalize_text(manifest_row.get("quality_registry_keypoint_method")) or _normalize_text(
            manifest_row.get("keypoint_method")
        )
        composition_rows.append(
            {
                "rig_id": (
                    _normalize_text(profile_row.get("rig_id")) if profile_row is not None else None
                ) or _normalize_text(dataset_context.get("rig_id")) or _normalize_text(manifest_row.get("rig_id")),
                "camera_id": (
                    _normalize_text(profile_row.get("camera_id")) if profile_row is not None else None
                ) or _normalize_text(dataset_context.get("camera_id")) or _normalize_text(manifest_row.get("camera_id")),
                "arena_id": (
                    _normalize_text(profile_row.get("arena_id")) if profile_row is not None else None
                ) or _normalize_text(dataset_context.get("arena_id")) or _normalize_text(manifest_row.get("arena_id")),
                "dish_design": (
                    _normalize_text(profile_row.get("dish_design")) if profile_row is not None else None
                ) or _normalize_text(dataset_context.get("dish_design")) or _normalize_text(manifest_row.get("dish_design")),
                "canvas_name": (
                    _normalize_text(profile_row.get("canvas_name")) if profile_row is not None else None
                ) or _normalize_text(dataset_context.get("canvas_name")) or _normalize_text(manifest_row.get("canvas_name")),
                "protocol_name": (
                    _normalize_text(profile_row.get("protocol_name")) if profile_row is not None else None
                ) or _normalize_text(dataset_context.get("protocol_name")) or _normalize_text(manifest_row.get("protocol_name")),
                "keypoint_method": keypoint_method,
            }
        )

        usable_value = _coalesce(
            _as_float(profile_row.get("usable_keypoints_total")) if profile_row is not None else None,
            _as_float(quality_row.get("usable_keypoints")) if quality_row is not None else None,
            _as_float(manifest_row.get("usable_keypoints_total")),
        )
        total_value = _coalesce(
            _as_float(profile_row.get("rows_total")) if profile_row is not None else None,
            _as_float(quality_row.get("total_keypoints")) if quality_row is not None else None,
            _as_float(manifest_row.get("keypoints_total")),
        )
        usable_rate = _coalesce(
            _as_float(profile_row.get("usable_rate")) if profile_row is not None else None,
            _as_float(quality_row.get("usable_keypoints_rate")) if quality_row is not None else None,
            _as_float(manifest_row.get("usable_keypoints_rate")),
        )
        raw_success_rate = _coalesce(
            _as_float(quality_row.get("raw_keypoints_success_rate")) if quality_row is not None else None,
            _as_float(manifest_row.get("keypoints_success_rate")),
        )
        raw_successful = _coalesce(
            _as_float(quality_row.get("raw_keypoints_successful")) if quality_row is not None else None,
            _as_float(manifest_row.get("keypoints_successful")),
        )
        usable_totals.append(usable_value)
        total_keypoints_values.append(total_value)
        usable_rate_values.append(usable_rate)
        raw_success_rate_values.append(raw_success_rate)
        raw_successful_values.append(raw_successful)
        raw_success_pairs.append((raw_success_rate, total_value))
        confidence_valid_pairs.append(
            (
                _as_float(profile_row.get("confidence_valid_rate")) if profile_row is not None else None,
                total_value,
            )
        )
        geometry_valid_pairs.append(
            (
                _as_float(profile_row.get("geometry_valid_rate")) if profile_row is not None else None,
                total_value,
            )
        )

        expected_mtime = _coalesce(
            _as_int(profile_row.get("zarr_mtime_ns")) if profile_row is not None else None,
            _as_int(quality_row.get("zarr_mtime_ns")) if quality_row is not None else None,
        )
        observed_mtime = None
        if ref.zarr_path.exists():
            try:
                observed_mtime = int(ref.zarr_path.stat().st_mtime_ns)
            except OSError:
                observed_mtime = None
        mtime_matches = expected_mtime is None or observed_mtime is None or expected_mtime == observed_mtime
        if not mtime_matches:
            zarr_mtime_mismatch_count += 1

        profile_stale_reason = _profile_stale_reason(profile_row, zarr_path=ref.zarr_path) if profile_row is not None else "missing profile row"
        if profile_stale_reason is not None:
            quality_stale_count += 1

        source_ref_payload: dict[str, Any] = {
            "dataset_id": ref.dataset_id,
            "zarr_path": str(ref.zarr_path),
            "profile_row_present": profile_row is not None,
            "profile_run": _normalize_text(profile_row.get("profile_run")) if profile_row is not None else None,
            "profile_source_keypoint_path": (
                _normalize_text(profile_row.get("source_keypoint_path")) if profile_row is not None else None
            ),
            "profile_stale_reason": profile_stale_reason,
            "quality_row_present": quality_row is not None,
            "quality_row_refined_run": _normalize_text(quality_row.get("refined_run")) if quality_row is not None else None,
            "source_keypoint_run": (
                _normalize_text(profile_row.get("source_keypoint_run")) if profile_row is not None else None
            ) or (
                _normalize_text(quality_row.get("source_keypoint_run")) if quality_row is not None else None
            ) or _normalize_text(manifest_row.get("keypoint_run_resolved")) or _normalize_text(manifest_row.get("keypoint_run")),
            "zarr_mtime_expected_ns": expected_mtime,
            "zarr_mtime_observed_ns": observed_mtime,
            "zarr_mtime_matches": bool(mtime_matches),
        }

        try:
            root = _open_zarr_group(ref.zarr_path)
        except Exception as exc:
            source_ref_payload["warning"] = f"zarr_open_failed:{exc}"
            source_run_refs.append(source_ref_payload)
            continue

        keypoints_parent = root.get("keypoints_runs")
        if keypoints_parent is None:
            source_ref_payload["warning"] = "missing_keypoints_runs_group"
            source_run_refs.append(source_ref_payload)
            continue
        keypoint_run = source_ref_payload["source_keypoint_run"]
        if keypoint_run is None or keypoint_run not in keypoints_parent:
            latest = _normalize_text(keypoints_parent.attrs.get("latest"))
            if latest and latest in keypoints_parent:
                keypoint_run = latest
        if keypoint_run is None or keypoint_run not in keypoints_parent:
            source_ref_payload["warning"] = "missing_keypoint_run"
            source_run_refs.append(source_ref_payload)
            continue

        keypoint_group = keypoints_parent[keypoint_run]
        refined_parent_name, refined_parent = _resolve_refined_parent(root)
        refined_run = _resolve_refined_run_name(
            root,
            keypoint_run=keypoint_run,
            manifest_row=manifest_row,
            quality_row=quality_row,
        )
        refined_group = (
            refined_parent[refined_run]
            if refined_parent_name is not None and refined_parent is not None and refined_run is not None and refined_run in refined_parent
            else None
        )
        source_ref_payload["resolved_keypoint_run"] = keypoint_run
        source_ref_payload["resolved_refined_run"] = refined_run

        if refined_group is not None:
            for metric_name in GEOMETRY_ARRAY_NAMES:
                if metric_name in refined_group:
                    metric_values = _as_finite_array(refined_group[metric_name][:])
                    if metric_values.size > 0:
                        geometry_values[metric_name].extend(float(v) for v in metric_values.tolist())

            if "confidence_valid" in refined_group:
                confidence_values = np.asarray(refined_group["confidence_valid"][:], dtype=np.bool_)
                confidence_valid_true += int(np.sum(confidence_values))
                confidence_valid_total += int(confidence_values.size)
            if "geometry_valid" in refined_group:
                geom_values = np.asarray(refined_group["geometry_valid"][:], dtype=np.bool_)
                geometry_valid_true += int(np.sum(geom_values))
                geometry_valid_total += int(geom_values.size)

            summary_stats = refined_group.attrs.get("summary_statistics")
            if isinstance(summary_stats, Mapping):
                flips_corrected = _as_int(summary_stats.get("flips_corrected"))
                flips_total = _as_int(summary_stats.get("total_rois"))
                if flips_corrected is not None and flips_total is not None and flips_total > 0:
                    flips_corrected_total += int(flips_corrected)
                    flips_total_rows += int(flips_total)

        keypoints_source = None
        if refined_group is not None and "keypoints_roi" in refined_group:
            keypoints_source = refined_group["keypoints_roi"]
        elif "keypoints_roi" in keypoint_group:
            keypoints_source = keypoint_group["keypoints_roi"]

        if keypoints_source is not None:
            keypoints_arr = np.asarray(keypoints_source[:], dtype=np.float64)
            if keypoints_arr.ndim == 3 and int(keypoints_arr.shape[2]) == 2:
                roi_h, roi_w = _resolve_roi_dimensions(root, keypoint_group)
                _update_landmark_spatial_accumulators(
                    dataset_id=ref.dataset_id,
                    keypoints_arr=keypoints_arr,
                    roi_h=int(roi_h),
                    roi_w=int(roi_w),
                    heatmap_counts=landmark_heatmap_counts,
                    edge_hits=landmark_edge_hits,
                    total_points=landmark_total_points,
                    source_dataset_ids=landmark_source_dataset_ids,
                    grid_h=int(LANDMARK_HEATMAP_GRID_H),
                    grid_w=int(LANDMARK_HEATMAP_GRID_W),
                    edge_margin_norm=float(LANDMARK_EDGE_MARGIN_NORM),
                )
                diag = _resolve_roi_diagonal(root, keypoint_group)
                if diag <= 0:
                    diag = 1.0

                kpt_count = int(keypoints_arr.shape[1])
                for edge_spec in graph_specs["edges"]:
                    i, j = edge_spec["indices"]
                    if i >= kpt_count or j >= kpt_count:
                        continue
                    p_i = keypoints_arr[:, i, :]
                    p_j = keypoints_arr[:, j, :]
                    finite = np.isfinite(p_i).all(axis=1) & np.isfinite(p_j).all(axis=1)
                    if not np.any(finite):
                        continue
                    lengths = np.linalg.norm(p_i[finite] - p_j[finite], axis=1) / float(diag)
                    edge_values[edge_spec["key"]].extend(float(v) for v in lengths.tolist())

                for angle_spec in graph_specs["angles"]:
                    i, j, k = angle_spec["indices"]
                    if i >= kpt_count or j >= kpt_count or k >= kpt_count:
                        continue
                    p_i = keypoints_arr[:, i, :]
                    p_j = keypoints_arr[:, j, :]
                    p_k = keypoints_arr[:, k, :]
                    finite = (
                        np.isfinite(p_i).all(axis=1)
                        & np.isfinite(p_j).all(axis=1)
                        & np.isfinite(p_k).all(axis=1)
                    )
                    if not np.any(finite):
                        continue
                    v1 = p_i[finite] - p_j[finite]
                    v2 = p_k[finite] - p_j[finite]
                    n1 = np.linalg.norm(v1, axis=1)
                    n2 = np.linalg.norm(v2, axis=1)
                    valid = (n1 > 0) & (n2 > 0)
                    if not np.any(valid):
                        continue
                    dot = np.sum(v1[valid] * v2[valid], axis=1)
                    cos_theta = np.clip(dot / (n1[valid] * n2[valid]), -1.0, 1.0)
                    angles = np.degrees(np.arccos(cos_theta))
                    angle_values[angle_spec["key"]].extend(float(v) for v in angles.tolist())

        source_run_refs.append(source_ref_payload)

    usable_total_sum = sum(v for v in usable_totals if v is not None)
    total_keypoints_sum = sum(v for v in total_keypoints_values if v is not None)
    usable_rate_overall = (
        _safe_ratio(float(usable_total_sum), float(total_keypoints_sum))
        if total_keypoints_sum > 0
        else None
    )

    raw_success_sum = sum(v for v in raw_successful_values if v is not None)
    has_raw_successful = any(v is not None for v in raw_successful_values)
    raw_success_denom = sum(v for v in total_keypoints_values if v is not None)
    raw_success_rate_overall = (
        _safe_ratio(float(raw_success_sum), float(raw_success_denom))
        if has_raw_successful and raw_success_denom > 0
        else _weighted_mean(raw_success_pairs)
    )
    confidence_valid_rate_overall = (
        _safe_ratio(float(confidence_valid_true), float(confidence_valid_total))
        if confidence_valid_total > 0
        else _weighted_mean(confidence_valid_pairs)
    )
    geometry_valid_rate_overall = (
        _safe_ratio(float(geometry_valid_true), float(geometry_valid_total))
        if geometry_valid_total > 0
        else _weighted_mean(geometry_valid_pairs)
    )

    quality_payload = {
        "usable_keypoints_total": int(usable_total_sum) if total_keypoints_sum > 0 else None,
        "usable_keypoints_rate_overall": float(usable_rate_overall) if usable_rate_overall is not None else None,
        "usable_keypoints_rate_dataset_stats": _numeric_stats(usable_rate_values),
        "usable_keypoints_rate_histogram": _numeric_histogram(usable_rate_values),
        "raw_success_rate_overall": float(raw_success_rate_overall) if raw_success_rate_overall is not None else None,
        "confidence_valid_rate": (
            float(confidence_valid_rate_overall) if confidence_valid_rate_overall is not None else None
        ),
        "geometry_valid_rate": (
            float(geometry_valid_rate_overall) if geometry_valid_rate_overall is not None else None
        ),
        "flips_corrected_rate": _safe_ratio(float(flips_corrected_total), float(flips_total_rows)),
    }

    geometry_payload: dict[str, Any] = {}
    for metric_name in GEOMETRY_ARRAY_NAMES:
        values = geometry_values[metric_name]
        geometry_payload[metric_name] = {
            "stats": _numeric_stats(values),
            "histogram": _numeric_histogram(values),
        }

    edge_stats_payload: dict[str, Any] = {}
    for edge_spec in graph_specs["edges"]:
        values = edge_values[edge_spec["key"]]
        edge_stats_payload[edge_spec["key"]] = {
            "alias": edge_spec.get("alias"),
            "stats": _numeric_stats(values),
            "histogram": _numeric_histogram(values),
        }
    angle_stats_payload: dict[str, Any] = {}
    for angle_spec in graph_specs["angles"]:
        values = angle_values[angle_spec["key"]]
        angle_stats_payload[angle_spec["key"]] = {
            "alias": angle_spec.get("alias"),
            "stats": _numeric_stats(values),
            "histogram": _numeric_histogram(values),
        }

    composition_counts: dict[str, dict[str, int]] = {}
    for field in COMPOSITION_FIELDS:
        counter: Counter[str] = Counter()
        for row in composition_rows:
            value = _normalize_text(row.get(field))
            if value is not None:
                counter[value] += 1
        if counter:
            composition_counts[field] = {key: int(counter[key]) for key in sorted(counter)}

    spatial_payload = _build_spatial_payload(
        heatmap_counts=landmark_heatmap_counts,
        edge_hits=landmark_edge_hits,
        total_points=landmark_total_points,
        source_dataset_ids=landmark_source_dataset_ids,
        pose_schema=resolved_pose_schema,
        edge_margin_norm=float(LANDMARK_EDGE_MARGIN_NORM),
    )
    train_val_parity = _build_train_val_parity(
        merged_zarr=merged_zarr,
        profile_rows_by_dataset=profile_rows_by_dataset,
        lineage_rows=lineage_rows,
    )

    dpf_values = [_as_float(row.get("dpf_at_acquisition")) for row in lineage_rows]
    card: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "set_id": _normalize_text(manifest.get("set_id")),
        "set_version": (
            str(manifest.get("set_version"))
            if isinstance(manifest.get("set_version"), (int, str))
            else None
        ),
        "selection": {
            "dataset_count": int(len(refs)),
            "rows_pre_gate": rows_pre_gate,
            "rows_post_gate": rows_post_gate,
            "split": str(split),
            "split_counts": {key: int(value) for key, value in sorted(split_counts.items())},
            "quality_exclusions_by_reason": _manifest_quality_exclusion_counts(manifest),
        },
        "quality": quality_payload,
        "geometry": geometry_payload,
        "skeleton_graph_metrics": {
            "pose_schema": {
                "kpt_shape": list(resolved_pose_schema.kpt_shape) if resolved_pose_schema.kpt_shape is not None else None,
                "keypoint_labels": list(resolved_pose_schema.keypoint_labels),
                "skeleton": [[int(i), int(j)] for i, j in resolved_pose_schema.skeleton],
            },
            "edge_length_norm_stats": edge_stats_payload,
            "angle_stats": angle_stats_payload,
        },
        "spatial": spatial_payload,
        "composition_counts": composition_counts,
        "subject_coverage": _build_subject_coverage_payload(subject_lineage_coverage),
        "genotype_counts": _build_genotype_counts(lineage_rows),
        "dpf_stats": _numeric_stats(dpf_values),
        "dpf_histogram": _build_dpf_histogram(
            dpf_values,
            source_dataset_count=len(refs),
        ),
        "train_val_parity": train_val_parity,
        "audit_freshness": {
            "canonical_dataset_id_resolved_count": int(
                sum(1 for ref in refs if ref.resolved_by_registry)
            ),
            "zarr_mtime_mismatch_count": int(zarr_mtime_mismatch_count),
            "quality_stale_count": int(quality_stale_count),
            "source_run_refs": sorted(source_run_refs, key=lambda row: str(row.get("dataset_id") or "")),
        },
    }
    return card


def _generate_plots(
    *,
    card: Mapping[str, Any],
    output_dir: Path,
    prefix: str,
    heatmap_bin_factor: int,
) -> list[Path]:
    if plot_data_card is None:
        raise RuntimeError("plot_keypoint_training_data_card import is unavailable")

    generator = getattr(plot_data_card, "generate_keypoint_training_data_card_plots", None)
    if generator is None:
        raise RuntimeError("generate_keypoint_training_data_card_plots is unavailable")
    generated = generator(
        card_payload=dict(card),
        output_dir=output_dir,
        prefix=prefix,
        heatmap_bin_factor=int(heatmap_bin_factor),
    )
    return [Path(path) for path in generated]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate keypoint training-data card payload from manifest datasets, "
            "registry keypoint_quality rows, and refined keypoint arrays."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest JSON path.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: <set_id>.data_card.json).")
    parser.add_argument(
        "--merged-zarr",
        type=Path,
        help="Optional merged training Zarr path used for split-count extraction.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split label metadata (default: train).")
    parser.add_argument(
        "--subject-lineage-policy",
        choices=("warn", "require"),
        default="warn",
        help=(
            "Subject-lineage coverage policy for manifest datasets using "
            "recording_subject_overview (default: warn)."
        ),
    )
    parser.add_argument(
        "--allow-profile-mtime-mismatch",
        action="store_true",
        help=(
            "Allow stale keypoint_data_profile_latest rows where zarr_mtime_ns "
            "does not match filesystem mtime."
        ),
    )
    parser.add_argument(
        "--allow-profile-fallback-scan",
        action="store_true",
        help=(
            "Allow fallback to direct Zarr scanning when keypoint_data_profile_latest "
            "rows are missing."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot PNG generation for the aggregated data card.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Optional output directory for data-card plots (default: <card_stem>.plots).",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        help="Optional filename prefix for data-card plots (default: set_id).",
    )
    parser.add_argument(
        "--plot-heatmap-bin-factor",
        type=int,
        default=2,
        help="Coarsening factor for heatmap bins in plot PNGs (default: 2).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open generated/existing plot PNGs via xdg-open after aggregation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force plot regeneration even when existing plot PNGs are present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute card but do not write output JSON.")
    parser.add_argument("--json", action="store_true", help="Print card JSON to stdout.")
    args = parser.parse_args(argv)
    if args.plot_heatmap_bin_factor < 1:
        parser.error("--plot-heatmap-bin-factor must be >= 1.")
    if args.view and args.dry_run:
        parser.error("--view cannot be combined with --dry-run.")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Training data card aggregation failed: manifest not found: {manifest_path}")
        return 1

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    try:
        manifest = _load_manifest(manifest_path)
        merged_zarr = Path(args.merged_zarr) if args.merged_zarr is not None else None
        card = _build_keypoint_training_data_card(
            registry=registry,
            manifest=manifest,
            merged_zarr=merged_zarr,
            split=str(args.split),
            subject_lineage_policy=str(args.subject_lineage_policy),
            allow_profile_mtime_mismatch=bool(args.allow_profile_mtime_mismatch),
            allow_profile_fallback_scan=bool(args.allow_profile_fallback_scan),
        )
    except Exception as exc:
        print(f"Training data card aggregation failed: {exc}")
        return 1
    finally:
        registry.close()

    output_path = Path(args.output) if args.output is not None else _default_output_path(
        manifest_path,
        _normalize_text(manifest.get("set_id")),
    )
    if args.json:
        print(json.dumps(card, indent=2))
    if args.dry_run:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        print(
            "Keypoint training data card: mode=dry-run "
            f"datasets={card['selection']['dataset_count']} output={output_path} "
            f"plots={'off' if args.no_plots else plot_dir} "
            f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
        )
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "Keypoint training data card: mode=apply "
        f"datasets={card['selection']['dataset_count']} output={output_path}"
    )
    if not args.no_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else _default_plot_dir(output_path)
        prefix = _normalize_text(args.plot_prefix) or _normalize_text(card.get("set_id")) or output_path.stem
        if (args.view or args.force) and plot_data_card is not None:
            plot_main = getattr(plot_data_card, "main", None)
            if callable(plot_main):
                plot_cli: list[str] = [
                    "--card",
                    str(output_path),
                    "--heatmap-bin-factor",
                    str(int(args.plot_heatmap_bin_factor)),
                ]
                _add_arg(plot_cli, "--output-dir", plot_dir)
                _add_arg(plot_cli, "--prefix", prefix)
                if args.view:
                    plot_cli.append("--view")
                if args.force:
                    plot_cli.append("--force")
                try:
                    plot_rc = int(plot_main(plot_cli))
                except Exception as exc:
                    print(f"Warning: keypoint training data-card plot generation skipped: {exc}")
                else:
                    if plot_rc != 0:
                        print(
                            "Warning: keypoint training data-card plotting returned non-zero "
                            f"exit code: {plot_rc}"
                        )
            else:
                print("Warning: keypoint plotter main(argv) unavailable; falling back to direct generation.")
                try:
                    generated = _generate_plots(
                        card=card,
                        output_dir=plot_dir,
                        prefix=str(prefix),
                        heatmap_bin_factor=int(args.plot_heatmap_bin_factor),
                    )
                    print(
                        "Keypoint training data-card plots: mode=apply "
                        f"generated={len(generated)} output_dir={plot_dir} "
                        f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
                    )
                except Exception as exc:
                    print(f"Warning: keypoint training data-card plot generation skipped: {exc}")
        else:
            try:
                generated = _generate_plots(
                    card=card,
                    output_dir=plot_dir,
                    prefix=str(prefix),
                    heatmap_bin_factor=int(args.plot_heatmap_bin_factor),
                )
                print(
                    "Keypoint training data-card plots: mode=apply "
                    f"generated={len(generated)} output_dir={plot_dir} "
                    f"heatmap_bin_factor={int(args.plot_heatmap_bin_factor)}"
                )
            except Exception as exc:
                print(f"Warning: keypoint training data-card plot generation skipped: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
