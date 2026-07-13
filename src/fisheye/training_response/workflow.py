"""Batch publication of whole-training response analytics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from fisheye.analytics_exports.contracts import (
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
)
from fisheye.analytics_exports.validation import validate_export_run
from fisheye.utils.virtual_collection_manifest import verify_manifest_sha256

from .cohort import (
    classify_training_response_features,
    discover_training_response_clusters,
)
from .contracts import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
    TRAINING_RESPONSE_FEATURES_TABLE,
    TrainingResponseConfig,
    contract_fields,
)
from .features import derive_training_response_features
from .validation import validate_training_response_run


OUTPUT_TABLES = (
    TRAINING_RESPONSE_FEATURES_TABLE,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
)


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return payload


def _declared_parts(
    source_root: Path,
    source_run_id: str,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[Path, ...]:
    raw_parts = manifest.get("part_files_by_table", {}).get(table_name, [])
    if not isinstance(raw_parts, list):
        raise ValueError(f"source part list is not an array for {table_name}")
    table_root = source_root / "v1" / table_name / f"export_run_id={source_run_id}"
    parts: list[Path] = []
    for raw_part in raw_parts:
        path = (table_root / Path(str(raw_part)).name).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise PermissionError(f"source part resolves outside root: {raw_part}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(path)
    return tuple(parts)


def _read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(dict(row) for row in pq.ParquetFile(path).read().to_pylist())
    return rows


def _group_recordings(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        recording_id = str(row.get("recording_id") or "").strip()
        if recording_id:
            output[recording_id].append(dict(row))
    return dict(output)


def _recording_protocols(
    source_root: Path, export_manifest: Mapping[str, Any]
) -> dict[str, str]:
    collection = export_manifest.get("collection_manifest")
    if not isinstance(collection, Mapping) or not collection.get("path"):
        return {}
    path = (
        source_root
        / "v1"
        / "manifests"
        / "collections"
        / Path(str(collection["path"])).name
    ).resolve()
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise PermissionError("collection manifest resolves outside export root") from exc
    payload = _load_object(path)
    expected_sha256 = str(collection.get("manifest_sha256") or "").strip()
    if expected_sha256 and (
        str(payload.get("manifest_sha256") or "").strip() != expected_sha256
        or not verify_manifest_sha256(payload)
    ):
        raise ValueError("source collection manifest SHA-256 mismatch")
    output: dict[str, str] = {}
    for record in payload.get("records", []):
        if not isinstance(record, Mapping):
            continue
        recording_id = str(record.get("recording_id") or "").strip()
        protocol = record.get("protocol")
        protocol_name = (
            str(protocol.get("protocol_name") or "").strip()
            if isinstance(protocol, Mapping)
            else ""
        )
        if recording_id:
            output[recording_id] = protocol_name or "unknown"
    return output


def build_training_response_tables(
    *,
    source_export_run_id: str,
    behavior_rows: Sequence[Mapping[str, Any]],
    distance_rows: Sequence[Mapping[str, Any]],
    egocentric_rows: Sequence[Mapping[str, Any]] = (),
    speed_distance_rows: Sequence[Mapping[str, Any]] = (),
    recording_protocols: Mapping[str, str] | None = None,
    config: TrainingResponseConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    config = config or TrainingResponseConfig()
    config.validate()
    behavior_by_recording = _group_recordings(behavior_rows)
    distance_by_recording = _group_recordings(distance_rows)
    egocentric_by_recording = _group_recordings(egocentric_rows)
    speed_by_recording = _group_recordings(speed_distance_rows)
    protocols = recording_protocols or {}
    features = [
        derive_training_response_features(
            recording_id=recording_id,
            source_export_run_id=source_export_run_id,
            behavior_rows=rows,
            distance_rows=distance_by_recording.get(recording_id, ()),
            egocentric_rows=egocentric_by_recording.get(recording_id, ()),
            speed_distance_rows=speed_by_recording.get(recording_id, ()),
            protocol_name=protocols.get(recording_id),
            config=config,
        )
        for recording_id, rows in sorted(behavior_by_recording.items())
    ]
    classifications = classify_training_response_features(features, config=config)
    clusters = discover_training_response_clusters(classifications, config=config)
    return {
        TRAINING_RESPONSE_FEATURES_TABLE: features,
        TRAINING_RESPONSE_CLASSIFICATION_TABLE: classifications,
        TRAINING_RESPONSE_CLUSTERS_TABLE: clusters,
    }


def _write_table(
    *,
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    config: TrainingResponseConfig,
) -> tuple[int, list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_root = output_root / "v1" / table_name / f"analysis_run_id={analysis_run_id}"
    table_root.mkdir(parents=True, exist_ok=False)
    required = set(contract_fields(table_name))
    missing = sorted(required - {key for row in rows for key in row})
    if missing:
        raise ValueError(f"{table_name} rows are missing required fields: {missing}")
    enriched = [{**dict(row), "analysis_run_id": analysis_run_id} for row in rows]
    columns = list(dict.fromkeys(key for row in enriched for key in row))
    normalized = [{column: row.get(column) for column in columns} for row in enriched]
    table = pa.Table.from_pylist(normalized)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"palette.schema_id": SCHEMA_ID.encode(),
            b"palette.schema_version": str(SCHEMA_VERSION).encode(),
            b"palette.table_name": table_name.encode(),
            b"palette.training_response_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode(),
        }
    )
    table = table.cast(table.schema.with_metadata(metadata))
    temporary = table_root / ".part-00000.parquet.tmp"
    part_path = table_root / "part-00000.parquet"
    pq.write_table(table, temporary)
    os.replace(temporary, part_path)
    return len(rows), [str(part_path)]


def run_training_response_analytics(
    *,
    source_export_root: Path,
    source_export_run_id: str,
    output_root: Path,
    analysis_run_id: str,
    config: TrainingResponseConfig | None = None,
) -> dict[str, Any]:
    config = config or TrainingResponseConfig()
    config.validate()
    source_root = Path(source_export_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    source_run_id = _safe_component(source_export_run_id, label="source export run ID")
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    for candidate, parent in ((destination, source_root), (source_root, destination)):
        try:
            candidate.relative_to(parent)
        except ValueError:
            continue
        raise ValueError("output root and immutable source export root must not overlap")
    source_validation = validate_export_run(source_root, source_run_id)
    source_manifest_path = (
        source_root / "v1" / "manifests" / f"export_run_id={source_run_id}.json"
    )
    source_manifest = _load_object(source_manifest_path)
    required_tables = (
        CHASER_EPOCH_BEHAVIOR_TABLE,
        CHASER_DISTANCE_SUMMARY_TABLE,
        CHASER_EGOCENTRIC_SUMMARY_TABLE,
        CHASER_SPEED_DISTANCE_TABLE,
    )
    source_rows = {
        table_name: _read_rows(
            _declared_parts(source_root, source_run_id, source_manifest, table_name)
        )
        for table_name in required_tables
    }
    tables = build_training_response_tables(
        source_export_run_id=source_run_id,
        behavior_rows=source_rows[CHASER_EPOCH_BEHAVIOR_TABLE],
        distance_rows=source_rows[CHASER_DISTANCE_SUMMARY_TABLE],
        egocentric_rows=source_rows[CHASER_EGOCENTRIC_SUMMARY_TABLE],
        speed_distance_rows=source_rows[CHASER_SPEED_DISTANCE_TABLE],
        recording_protocols=_recording_protocols(source_root, source_manifest),
        config=config,
    )
    manifest_path = (
        destination / "v1" / "manifests" / f"analysis_run_id={run_id}.json"
    )
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    for table_name in OUTPUT_TABLES:
        if (
            destination / "v1" / table_name / f"analysis_run_id={run_id}"
        ).exists():
            raise FileExistsError(f"output table already exists: {table_name}")
    staging = destination / f".training_response_staging_{run_id}"
    if staging.exists():
        raise FileExistsError(staging)
    row_counts: dict[str, int] = {}
    staged_parts: dict[str, list[str]] = {}
    for table_name in OUTPUT_TABLES:
        row_counts[table_name], staged_parts[table_name] = _write_table(
            output_root=staging,
            analysis_run_id=run_id,
            table_name=table_name,
            rows=tables[table_name],
            config=config,
        )
    part_files = {
        table_name: [
            str(destination / Path(path).relative_to(staging)) for path in paths
        ]
        for table_name, paths in staged_parts.items()
    }
    collection = source_manifest.get("collection_manifest")
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "analysis_run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_export_root": str(source_root),
        "source_export_run_id": source_run_id,
        "source_export_manifest_sha256": hashlib.sha256(
            source_manifest_path.read_bytes()
        ).hexdigest(),
        "source_collection_manifest_sha256": (
            collection.get("manifest_sha256")
            if isinstance(collection, Mapping)
            else None
        ),
        "source_validation": source_validation,
        "feature_config": config.to_dict(),
        "output_tables": list(OUTPUT_TABLES),
        "row_counts_by_table": row_counts,
        "part_files_by_table": part_files,
        "source_export_mutated": False,
        "interpretation_guardrail": (
            "descriptive training response; causal avoidance, fear, anxiety, and "
            "escape success are not inferred"
        ),
        "temporal_adaptation_status": "unavailable_without_training_time_bins_or_samples",
    }
    staged_manifest = staging / "v1" / "manifests" / manifest_path.name
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for table_name in OUTPUT_TABLES:
        source_table = staging / "v1" / table_name / f"analysis_run_id={run_id}"
        destination_parent = destination / "v1" / table_name
        destination_parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_table, destination_parent / source_table.name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_manifest, manifest_path)
    shutil.rmtree(staging)
    output_validation = validate_training_response_run(destination, run_id)
    return {
        **payload,
        "manifest_path": str(manifest_path),
        "output_validation": output_validation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify whole-training chaser response profiles."
    )
    parser.add_argument("--source-export-root", type=Path, required=True)
    parser.add_argument("--source-export-run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--min-valid-position-fraction", type=float, default=0.75)
    parser.add_argument("--relative-score-threshold", type=float, default=0.75)
    parser.add_argument("--cluster-max-components", type=int, default=6)
    parser.add_argument("--cluster-min-rows-per-component", type=int, default=10)
    parser.add_argument("--cluster-stability-threshold", type=float, default=0.60)
    parser.add_argument("--cluster-stability-resamples", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = TrainingResponseConfig(
        min_valid_position_fraction=args.min_valid_position_fraction,
        relative_score_threshold=args.relative_score_threshold,
        cluster_max_components=args.cluster_max_components,
        cluster_min_rows_per_component=args.cluster_min_rows_per_component,
        cluster_stability_threshold=args.cluster_stability_threshold,
        cluster_stability_resamples=args.cluster_stability_resamples,
    )
    result = run_training_response_analytics(
        source_export_root=args.source_export_root,
        source_export_run_id=args.source_export_run_id,
        output_root=args.output_root,
        analysis_run_id=args.analysis_run_id,
        config=config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_TABLES",
    "build_training_response_tables",
    "run_training_response_analytics",
]
