from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.analytics_exports.derived_publication import (
    publish_derived_table_generation,
)
from fisheye.analytics_exports.arrow_contract_core import payload_sha256
from fisheye.analytics_exports.publication import sha256_file
from fisheye.baseline_strategy.contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_ARROW_CONTRACTS,
    BASELINE_STRATEGY_CLASSIFICATION_TABLE,
    BASELINE_STRATEGY_CLUSTERS_TABLE,
    BASELINE_STRATEGY_FEATURES_TABLE,
    BASELINE_STRATEGY_TABLES,
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    StrategyFeatureConfig,
    baseline_strategy_arrow_contract_envelope,
    normalize_baseline_strategy_rows,
)
from fisheye.baseline_strategy.qc import (
    discover_strategy_catalog,
    scan_recording_baseline_samples,
    scan_strategy_qc_rows,
    select_strategy_run_id,
    source_export_context,
)


def _write_part(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _complete_strategy_row(
    table_name: str,
    *,
    analysis_run_id: str,
    export_run_id: str,
) -> dict[str, object]:
    row: dict[str, object] = {}
    for item in BASELINE_STRATEGY_ARROW_CONTRACTS[table_name].fields:
        if item.nullable:
            continue
        if item.arrow_type == "string":
            row[item.name] = "value"
        elif item.arrow_type in {"int32", "int64"}:
            row[item.name] = 1
        elif item.arrow_type == "float64":
            row[item.name] = 1.0
        elif item.arrow_type == "bool":
            row[item.name] = False
        elif item.arrow_type == "list<string>":
            row[item.name] = ["value"]
        else:  # pragma: no cover - the exact contract closes this vocabulary.
            raise AssertionError(item.arrow_type)
    row.update(
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        table_name=table_name,
        method=METHOD,
        method_version=METHOD_VERSION,
        recording_id="recording_001",
        track_id=0,
        baseline_window_id=0,
        baseline_window_label="baseline",
        source_export_run_id=export_run_id,
        zarr_path="recording_001_analysis.zarr",
        analysis_run_id=analysis_run_id,
    )
    return row


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path, str]:
    strategy_root = tmp_path / "strategy"
    export_root = tmp_path / "exports"
    analysis_run_id = "strategy_001"
    export_run_id = "export_001"
    collection_path = (
        export_root / "v1" / "manifests" / "collections" / "collection.manifest.json"
    )
    collection_path.parent.mkdir(parents=True)
    collection_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "recording_id": "recording_001",
                        "protocol": {"protocol_name": "RedScare"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sample_part = (
        export_root
        / "v1"
        / ".generations"
        / f"export_run_id={export_run_id}"
        / "generation=test"
        / "tables"
        / "baseline_kinematic_samples"
        / "part-recording_001.parquet"
    )
    _write_part(
        sample_part,
        [
            {
                "recording_id": "recording_001",
                "relative_time_s": 0.1,
                "source_frame": 10,
                "x_arena_mm": 1.0,
                "y_arena_mm": 2.0,
                "speed_mm_s": 3.0,
                "wall": False,
                "position_valid": True,
                "sample_valid": True,
            },
            {
                "recording_id": "recording_002",
                "relative_time_s": 0.1,
                "source_frame": 10,
                "x_arena_mm": 4.0,
                "y_arena_mm": 5.0,
                "speed_mm_s": 6.0,
                "wall": True,
                "position_valid": True,
                "sample_valid": True,
            },
        ],
    )
    export_manifest_path = (
        export_root / "v1" / "manifests" / f"export_run_id={export_run_id}.json"
    )
    generation_path = (
        Path("v1")
        / ".generations"
        / f"export_run_id={export_run_id}"
        / "generation=test"
    )
    relative_sample = (
        generation_path
        / "tables"
        / "baseline_kinematic_samples"
        / sample_part.name
    ).as_posix()
    export_manifest_path.write_text(
        json.dumps(
            {
                "export_run_id": export_run_id,
                "collection_manifest": {"path": str(collection_path)},
                "part_files_by_table": {
                    "baseline_kinematic_samples": [relative_sample]
                },
                "publication": {
                    "schema_id": "palette.analytics_export.publication",
                    "schema_version": 1,
                    "state": "complete",
                    "generation_id": "test",
                    "generation_path": generation_path.as_posix(),
                    "parts_by_table": {
                        "baseline_kinematic_samples": [
                            {
                                "path": relative_sample,
                                "sha256": sha256_file(sample_part),
                                "size_bytes": sample_part.stat().st_size,
                                "row_count": 2,
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    export_manifest_sha256 = hashlib.sha256(
        export_manifest_path.read_bytes()
    ).hexdigest()
    feature = _complete_strategy_row(
        BASELINE_STRATEGY_FEATURES_TABLE,
        analysis_run_id=analysis_run_id,
        export_run_id=export_run_id,
    )
    feature.update(
        feature_status="complete",
        wall_fraction=0.4,
        occupancy_coverage_fraction=0.8,
    )
    classification = _complete_strategy_row(
        BASELINE_STRATEGY_CLASSIFICATION_TABLE,
        analysis_run_id=analysis_run_id,
        export_run_id=export_run_id,
    )
    classification.update(
        classification_status="complete",
        classification_reason=None,
        activity_state="typical_activity",
        boundary_strategy="mixed_boundary",
        spatial_organization="intermediate",
        temporal_pattern="stable_or_mixed",
        primary_strategy="mixed_or_uncertain",
        classification_confidence_score=0.2,
    )
    cluster = _complete_strategy_row(
        BASELINE_STRATEGY_CLUSTERS_TABLE,
        analysis_run_id=analysis_run_id,
        export_run_id=export_run_id,
    )
    cluster.update(
        cluster_status="complete",
        cluster_reason=None,
        cluster_id=1,
        cluster_probability=0.9,
    )
    rows_by_table = {
        BASELINE_STRATEGY_FEATURES_TABLE: [feature],
        BASELINE_EXPLORATION_EPISODES_TABLE: [],
        BASELINE_STRATEGY_CLASSIFICATION_TABLE: [classification],
        BASELINE_STRATEGY_CLUSTERS_TABLE: [cluster],
    }
    normalized = {
        table_name: normalize_baseline_strategy_rows(
            table_name,
            rows_by_table[table_name],
            analysis_run_id=analysis_run_id,
        )
        for table_name in BASELINE_STRATEGY_TABLES
    }
    config = StrategyFeatureConfig()
    publish_derived_table_generation(
        output_root=strategy_root,
        analysis_run_id=analysis_run_id,
        rows_by_table=normalized,
        table_names=BASELINE_STRATEGY_TABLES,
        contracts=BASELINE_STRATEGY_ARROW_CONTRACTS,
        arrow_contract_envelope=baseline_strategy_arrow_contract_envelope(),
        arrow_envelope_schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
        manifest_fields={
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": "2026-07-13T00:00:00+00:00",
            "source_export_root": str(export_root.resolve()),
            "source_export_run_id": export_run_id,
            "source_export_manifest_sha256": export_manifest_sha256,
            "source_collection_manifest_sha256": None,
            "row_provenance": {
                "source_export_run_id": export_run_id,
                "status": "complete",
            },
            "source_validation": {"status": "fixture"},
            "feature_config": config.to_dict(),
            "source_export_mutated": False,
            "interpretation_guardrail": (
                "strategy labels are descriptive; anxiety inference is not permitted"
            ),
        },
        footer_metadata={
            b"palette.schema_id": SCHEMA_ID.encode(),
            b"palette.schema_version": str(SCHEMA_VERSION).encode(),
            b"palette.feature_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode(),
        },
        generation_id="fixture-generation",
    )
    return strategy_root, export_root, analysis_run_id


def test_catalog_and_lazy_qc_join_use_manifest_declared_parts(tmp_path: Path) -> None:
    strategy_root, export_root, run_id = _fixture_roots(tmp_path)
    catalog = discover_strategy_catalog(strategy_root)

    assert select_strategy_run_id(catalog, "latest") == run_id
    assert catalog.entries[0].source_export_run_id == "export_001"
    context = source_export_context(
        strategy_root, run_id, authorized_export_root=export_root
    )
    assert context.recording_protocols == {"recording_001": "RedScare"}

    lazy = scan_strategy_qc_rows(
        strategy_root,
        run_id,
        recording_protocols=context.recording_protocols,
    )
    assert "SCAN" in lazy.explain().upper()
    row = lazy.collect().to_dicts()[0]
    assert row["primary_strategy"] == "mixed_or_uncertain"
    assert row["cluster_id"] == 1
    assert row["protocol_name"] == "RedScare"


def test_catalog_rejects_digest_valid_but_ineligible_v2_run(tmp_path: Path) -> None:
    strategy_root, _export_root, run_id = _fixture_roots(tmp_path)
    manifest_path = (
        strategy_root / "v2" / "manifests" / f"analysis_run_id={run_id}.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["publication"]["selector_eligible"] = False
    payload["manifest_payload_sha256"] = payload_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = discover_strategy_catalog(strategy_root)

    assert catalog.entries == ()
    assert len(catalog.diagnostics) == 1
    assert "not selector eligible" in catalog.diagnostics[0].message


def test_recording_sample_scan_pushes_filter_to_lazy_source(tmp_path: Path) -> None:
    strategy_root, export_root, run_id = _fixture_roots(tmp_path)
    context = source_export_context(
        strategy_root, run_id, authorized_export_root=export_root
    )
    lazy = scan_recording_baseline_samples(context, "recording_001")

    assert "recording_001" in lazy.explain()
    rows = lazy.collect().to_dicts()
    assert len(rows) == 1
    assert rows[0]["source_frame"] == 10


def test_source_context_binds_parsed_manifest_and_digest_to_same_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    strategy_root, export_root, run_id = _fixture_roots(tmp_path)
    import fisheye.baseline_strategy.qc as qc

    real_load = qc._load_object_snapshot

    def replace_after_snapshot(path: Path):
        snapshot = real_load(path)
        replacement = json.loads(path.read_text(encoding="utf-8"))
        replacement["replacement_generation"] = "newer"
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(qc, "_load_object_snapshot", replace_after_snapshot)
    context = source_export_context(
        strategy_root,
        run_id,
        authorized_export_root=export_root,
    )

    assert context.manifest.get("replacement_generation") is None
    assert scan_recording_baseline_samples(
        context,
        "recording_001",
    ).collect().height == 1


def test_source_context_rejects_unapproved_export_root(tmp_path: Path) -> None:
    strategy_root, _export_root, run_id = _fixture_roots(tmp_path)

    try:
        source_export_context(
            strategy_root,
            run_id,
            authorized_export_root=tmp_path / "different_export_root",
        )
    except PermissionError as exc:
        assert "authorized export root" in str(exc)
    else:
        raise AssertionError("unapproved source export root was accepted")


def test_source_context_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    strategy_root, export_root, run_id = _fixture_roots(tmp_path)
    manifest_path = (
        strategy_root / "v2" / "manifests" / f"analysis_run_id={run_id}.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_export_manifest_sha256"] = "0" * 64
    payload["manifest_payload_sha256"] = payload_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        source_export_context(
            strategy_root, run_id, authorized_export_root=export_root
        )
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("source export manifest hash mismatch was accepted")
