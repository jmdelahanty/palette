"""Batch workflow for derived baseline strategy analytics.

The source analytics export is validated and opened read-only.  Outputs are
published beneath a separate root with their own manifest.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from fisheye.analytics_exports.contracts import (
    BASELINE_BEHAVIOR_SUMMARY_TABLE,
    BASELINE_BEHAVIOR_TIME_BINS_TABLE,
    BASELINE_KINEMATIC_SAMPLES_TABLE,
)
from fisheye.analytics_exports.validation import validate_export_run

from .cohort import classify_strategy_features, discover_strategy_clusters
from .contracts import (
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
    BASELINE_STRATEGY_FEATURES_TABLE,
    SCHEMA_ID,
    SCHEMA_VERSION,
    StrategyFeatureConfig,
    contract_fields,
)
from .features import derive_baseline_strategy_features
from .validation import validate_strategy_analytics_run


OUTPUT_TABLES = (
    BASELINE_STRATEGY_FEATURES_TABLE,
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
)


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _identity_key(row: Mapping[str, Any]) -> tuple[object, object, object]:
    return row.get("recording_id"), row.get("track_id"), row.get("baseline_window_id")


def _load_manifest(source_root: Path, source_run_id: str) -> dict[str, Any]:
    path = source_root / "v1" / "manifests" / f"export_run_id={source_run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source manifest is not an object: {path}")
    return payload


def _declared_parts(
    source_root: Path,
    source_run_id: str,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[Path, ...]:
    raw_by_table = manifest.get("part_files_by_table")
    if not isinstance(raw_by_table, Mapping):
        return ()
    raw_parts = raw_by_table.get(table_name)
    if not isinstance(raw_parts, list):
        return ()
    table_root = source_root / "v1" / table_name / f"export_run_id={source_run_id}"
    output = []
    for raw_part in raw_parts:
        part_name = Path(str(raw_part)).name
        path = (table_root / part_name).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise PermissionError(f"source part resolves outside export root: {raw_part}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(path)
    return tuple(output)


def _read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in paths:
        # ParquetFile avoids dataset-level Hive partition inference.  The
        # source rows already contain export_run_id, whose plain string type
        # otherwise conflicts with the dictionary-encoded partition column.
        rows.extend(dict(row) for row in pq.ParquetFile(path).read().to_pylist())
    return rows


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[object, object, object], list[dict[str, Any]]]:
    grouped: dict[tuple[object, object, object], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_identity_key(row)].append(dict(row))
    return dict(grouped)


def build_strategy_tables(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    time_bin_rows: Sequence[Mapping[str, Any]] = (),
    sample_rows_by_identity: Mapping[
        tuple[object, object, object], Sequence[Mapping[str, Any]]
    ] | None = None,
    config: StrategyFeatureConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build all derived tables from already loaded immutable export rows."""

    config = config or StrategyFeatureConfig()
    config.validate()
    time_by_identity = _group_rows(time_bin_rows)
    samples = sample_rows_by_identity or {}
    features = []
    episodes = []
    seen: set[tuple[object, object, object]] = set()
    for summary in summary_rows:
        key = _identity_key(summary)
        if key in seen:
            raise ValueError(f"duplicate baseline summary identity: {key}")
        seen.add(key)
        feature, feature_episodes = derive_baseline_strategy_features(
            summary,
            time_by_identity.get(key, ()),
            samples.get(key, ()),
            config=config,
        )
        features.append(feature)
        episodes.extend(feature_episodes)
    classifications = classify_strategy_features(features, config=config)
    clusters = discover_strategy_clusters(classifications, config=config)
    return {
        BASELINE_STRATEGY_FEATURES_TABLE: features,
        BASELINE_EXPLORATION_EPISODES_TABLE: episodes,
        BASELINE_STRATEGY_CLASSIFICATION_TABLE: classifications,
        BASELINE_STRATEGY_CLUSTERS_TABLE: clusters,
    }


def _write_table(
    *,
    output_root: Path,
    analysis_run_id: str,
    table_name: str,
    rows: Sequence[Mapping[str, Any]],
    config: StrategyFeatureConfig,
) -> tuple[int, list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table_root = output_root / "v1" / table_name / f"analysis_run_id={analysis_run_id}"
    table_root.mkdir(parents=True, exist_ok=False)
    if not rows:
        return 0, []
    required = set(contract_fields(table_name))
    missing = sorted(required - {key for row in rows for key in row})
    if missing:
        raise ValueError(f"{table_name} rows are missing required fields: {missing}")
    enriched = [
        {
            **dict(row),
            "analysis_run_id": analysis_run_id,
        }
        for row in rows
    ]
    table = pa.Table.from_pylist(enriched)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"palette.schema_id": SCHEMA_ID.encode("utf-8"),
            b"palette.schema_version": str(SCHEMA_VERSION).encode("utf-8"),
            b"palette.table_name": table_name.encode("utf-8"),
            b"palette.feature_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        }
    )
    table = table.cast(table.schema.with_metadata(metadata))
    part_path = table_root / "part-00000.parquet"
    temporary = table_root / ".part-00000.parquet.tmp"
    pq.write_table(table, temporary)
    os.replace(temporary, part_path)
    return len(rows), [str(part_path)]


def run_strategy_analytics(
    *,
    source_export_root: Path,
    source_export_run_id: str,
    output_root: Path,
    analysis_run_id: str,
    config: StrategyFeatureConfig | None = None,
) -> dict[str, Any]:
    """Validate one export, compute strategy analytics, and publish separately."""

    config = config or StrategyFeatureConfig()
    config.validate()
    source_root = Path(source_export_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    source_run_id = _safe_component(source_export_run_id, label="source export run ID")
    run_id = _safe_component(analysis_run_id, label="analysis run ID")
    roots_overlap = False
    for candidate, parent in ((destination, source_root), (source_root, destination)):
        try:
            candidate.relative_to(parent)
        except ValueError:
            continue
        roots_overlap = True
    if roots_overlap:
        raise ValueError(
            "output_root must not equal, contain, or be contained by the immutable source_export_root"
        )
    validation = validate_export_run(source_root, source_run_id)
    manifest = _load_manifest(source_root, source_run_id)
    summary_parts = _declared_parts(
        source_root, source_run_id, manifest, BASELINE_BEHAVIOR_SUMMARY_TABLE
    )
    if not summary_parts:
        raise ValueError("source export has no baseline_behavior_summary parts")
    summary_rows = _read_rows(summary_parts)
    time_rows = _read_rows(
        _declared_parts(
            source_root, source_run_id, manifest, BASELINE_BEHAVIOR_TIME_BINS_TABLE
        )
    )

    time_by_identity = _group_rows(time_rows)
    summary_by_identity: dict[tuple[object, object, object], dict[str, Any]] = {}
    for summary in summary_rows:
        key = _identity_key(summary)
        if key in summary_by_identity:
            raise ValueError(f"duplicate baseline summary identity: {key}")
        summary_by_identity[key] = dict(summary)

    features: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    processed: set[tuple[object, object, object]] = set()
    for sample_part in _declared_parts(
        source_root, source_run_id, manifest, BASELINE_KINEMATIC_SAMPLES_TABLE
    ):
        part_samples = _group_rows(_read_rows((sample_part,)))
        for key, identity_samples in part_samples.items():
            summary = summary_by_identity.get(key)
            if summary is None:
                continue
            if key in processed:
                raise ValueError(
                    f"baseline kinematic samples for one identity span multiple source parts: {key}"
                )
            feature, identity_episodes = derive_baseline_strategy_features(
                summary,
                time_by_identity.get(key, ()),
                identity_samples,
                config=config,
            )
            features.append(feature)
            episodes.extend(identity_episodes)
            processed.add(key)
    for key, summary in summary_by_identity.items():
        if key in processed:
            continue
        feature, identity_episodes = derive_baseline_strategy_features(
            summary,
            time_by_identity.get(key, ()),
            (),
            config=config,
        )
        features.append(feature)
        episodes.extend(identity_episodes)
    features.sort(key=lambda row: tuple(str(item) for item in _identity_key(row)))
    classifications = classify_strategy_features(features, config=config)
    clusters = discover_strategy_clusters(classifications, config=config)
    tables = {
        BASELINE_STRATEGY_FEATURES_TABLE: features,
        BASELINE_EXPLORATION_EPISODES_TABLE: episodes,
        BASELINE_STRATEGY_CLASSIFICATION_TABLE: classifications,
        BASELINE_STRATEGY_CLUSTERS_TABLE: clusters,
    }
    manifest_path = destination / "v1" / "manifests" / f"analysis_run_id={run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    for table_name in OUTPUT_TABLES:
        table_root = destination / "v1" / table_name / f"analysis_run_id={run_id}"
        if table_root.exists():
            raise FileExistsError(table_root)

    staging = destination / f".baseline_strategy_staging_{run_id}"
    if staging.exists():
        raise FileExistsError(staging)
    row_counts: dict[str, int] = {}
    staged_part_files: dict[str, list[str]] = {}
    for table_name in OUTPUT_TABLES:
        row_count, parts = _write_table(
            output_root=staging,
            analysis_run_id=run_id,
            table_name=table_name,
            rows=tables[table_name],
            config=config,
        )
        row_counts[table_name] = row_count
        staged_part_files[table_name] = parts
    part_files = {
        table_name: [
            str(destination / Path(path).relative_to(staging))
            for path in paths
        ]
        for table_name, paths in staged_part_files.items()
    }
    payload = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "analysis_run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_export_root": str(source_root),
        "source_export_run_id": source_run_id,
        "source_validation": validation,
        "feature_config": config.to_dict(),
        "output_tables": list(OUTPUT_TABLES),
        "row_counts_by_table": row_counts,
        "part_files_by_table": part_files,
        "source_export_mutated": False,
        "interpretation_guardrail": (
            "strategy labels are descriptive; anxiety inference is not permitted"
        ),
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
    validation = validate_strategy_analytics_run(destination, run_id)
    return {
        **payload,
        "manifest_path": str(manifest_path),
        "output_validation": validation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive fish/rodent open-field baseline strategy analytics."
    )
    parser.add_argument("--source-export-root", type=Path, required=True)
    parser.add_argument("--source-export-run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--active-speed-mm-s", type=float, default=1.0)
    parser.add_argument("--spatial-grid-size", type=int, default=12)
    parser.add_argument("--dwell-grid-size", type=int, default=8)
    parser.add_argument("--relative-score-threshold", type=float, default=0.75)
    parser.add_argument("--cluster-max-components", type=int, default=6)
    parser.add_argument("--cluster-stability-resamples", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = StrategyFeatureConfig(
        active_speed_mm_s=args.active_speed_mm_s,
        spatial_grid_size=args.spatial_grid_size,
        dwell_grid_size=args.dwell_grid_size,
        relative_score_threshold=args.relative_score_threshold,
        cluster_max_components=args.cluster_max_components,
        cluster_stability_resamples=args.cluster_stability_resamples,
    )
    result = run_strategy_analytics(
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


__all__ = ["OUTPUT_TABLES", "build_strategy_tables", "run_strategy_analytics"]
