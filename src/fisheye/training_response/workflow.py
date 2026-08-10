"""Batch publication of whole-training response analytics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.analytics_exports.contracts import (
    CHASER_DISTANCE_SUMMARY_TABLE,
    CHASER_EGOCENTRIC_SUMMARY_TABLE,
    CHASER_EPOCH_BEHAVIOR_TABLE,
    CHASER_SPEED_DISTANCE_TABLE,
)
from fisheye.analytics_exports.publication import (
    export_manifest_path,
    manifest_selected_part_files_from_payload,
)
from fisheye.analytics_exports.derived_publication import (
    publish_derived_table_generation,
)
from fisheye.analytics_exports.validation import validate_export_payload
from fisheye.utils.virtual_collection_manifest import verify_manifest_sha256

from .cohort import (
    classify_training_response_features,
    discover_training_response_clusters,
)
from .contracts import (
    ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
    ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
    ARROW_TABLE_CONTRACTS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRAINING_RESPONSE_CLASSIFICATION_TABLE,
    TRAINING_RESPONSE_CLUSTERS_TABLE,
    TRAINING_RESPONSE_FEATURES_TABLE,
    TrainingResponseConfig,
    normalize_training_response_rows,
    training_response_arrow_contract_envelope,
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


def _load_object_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _declared_parts(
    source_root: Path,
    source_run_id: str,
    manifest: Mapping[str, Any],
    table_name: str,
) -> tuple[Path, ...]:
    if manifest.get("export_run_id") != source_run_id:
        raise ValueError("source manifest run identity mismatch")
    parts = manifest_selected_part_files_from_payload(
        source_root,
        manifest,
        table_name,
    )
    for path in parts:
        if not path.is_file():
            raise FileNotFoundError(path)
    return parts


def _read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(dict(row) for row in pq.ParquetFile(path).read().to_pylist())
    return rows


IdentityKey = tuple[str, str, str]


def _group_recordings(
    rows: Iterable[Mapping[str, Any]],
) -> dict[IdentityKey, list[dict[str, Any]]]:
    output: dict[IdentityKey, list[dict[str, Any]]] = defaultdict(list)
    binding_by_recording: dict[str, tuple[str, str]] = {}
    for row_index, row in enumerate(rows):
        values = tuple(
            str(row.get(name) or "").strip()
            for name in ("recording_id", "session_id", "subject_id")
        )
        if any(not value for value in values):
            raise ValueError(
                f"source row {row_index} is missing recording/session/subject identity"
            )
        recording_id, session_id, subject_id = values
        binding = (session_id, subject_id)
        previous = binding_by_recording.setdefault(recording_id, binding)
        if previous != binding:
            raise ValueError(
                f"recording {recording_id!r} has conflicting session/subject bindings"
            )
        output[(recording_id, session_id, subject_id)].append(dict(row))
    return dict(output)


def _source_registry_identity_receipt(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw = source_manifest.get("registry_identity_receipt")
    if raw is None:
        raw = source_manifest.get("registry_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("validated source export has no registry identity receipt")
    # The analytics-export validator owns receipt semantics and schema dispatch.
    # Training response preserves the validated receipt opaquely and binds it to
    # the immutable source-manifest digest rather than duplicating that contract.
    return json.loads(json.dumps(raw, sort_keys=True, separators=(",", ":")))


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
    all_groups = (
        behavior_by_recording,
        distance_by_recording,
        egocentric_by_recording,
        speed_by_recording,
    )
    binding_by_recording: dict[str, tuple[str, str]] = {}
    for grouped in all_groups:
        for recording_id, session_id, subject_id in grouped:
            binding = (session_id, subject_id)
            previous = binding_by_recording.setdefault(recording_id, binding)
            if previous != binding:
                raise ValueError(
                    f"recording {recording_id!r} has conflicting identities across source tables"
                )
    features = [
        derive_training_response_features(
            recording_id=recording_id,
            session_id=session_id,
            subject_id=subject_id,
            source_export_run_id=source_export_run_id,
            behavior_rows=rows,
            distance_rows=distance_by_recording.get(identity_key, ()),
            egocentric_rows=egocentric_by_recording.get(identity_key, ()),
            speed_distance_rows=speed_by_recording.get(identity_key, ()),
            protocol_name=protocols.get(recording_id),
            config=config,
        )
        for identity_key, rows in sorted(behavior_by_recording.items())
        for recording_id, session_id, subject_id in (identity_key,)
    ]
    classifications = classify_training_response_features(features, config=config)
    clusters = discover_training_response_clusters(classifications, config=config)
    return {
        TRAINING_RESPONSE_FEATURES_TABLE: features,
        TRAINING_RESPONSE_CLASSIFICATION_TABLE: classifications,
        TRAINING_RESPONSE_CLUSTERS_TABLE: clusters,
    }


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
    source_manifest_path = export_manifest_path(source_root, source_run_id)
    source_manifest, source_manifest_sha256 = _load_object_snapshot(
        source_manifest_path
    )
    source_validation = validate_export_payload(
        source_root,
        source_run_id,
        source_manifest,
    )
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
    normalized_tables = {
        table_name: normalize_training_response_rows(
            table_name,
            tables[table_name],
            analysis_run_id=run_id,
            config=config,
        )
        for table_name in OUTPUT_TABLES
    }
    collection = source_manifest.get("collection_manifest")
    source_registry_identity_receipt = _source_registry_identity_receipt(
        source_manifest
    )
    manifest_fields = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_export_root": str(source_root),
        "source_export_run_id": source_run_id,
        "source_export_manifest_sha256": source_manifest_sha256,
        "source_collection_manifest_sha256": (
            collection.get("manifest_sha256")
            if isinstance(collection, Mapping)
            else None
        ),
        "source_validation": source_validation,
        "source_registry_identity_receipt": source_registry_identity_receipt,
        "feature_config": config.to_dict(),
        "source_export_mutated": False,
        "interpretation_guardrail": (
            "descriptive training response; causal avoidance, fear, anxiety, and "
            "escape success are not inferred"
        ),
        "temporal_adaptation_status": "unavailable_without_training_time_bins_or_samples",
    }
    if hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() != source_manifest_sha256:
        raise ValueError("source export manifest changed during training-response planning")
    payload = publish_derived_table_generation(
        output_root=destination,
        analysis_run_id=run_id,
        rows_by_table=normalized_tables,
        table_names=OUTPUT_TABLES,
        contracts=ARROW_TABLE_CONTRACTS,
        arrow_contract_envelope=training_response_arrow_contract_envelope(),
        arrow_envelope_schema_id=ARROW_CONTRACT_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=ARROW_CONTRACT_ENVELOPE_SCHEMA_VERSION,
        manifest_fields=manifest_fields,
        footer_metadata={
            b"palette.schema_id": SCHEMA_ID.encode("utf-8"),
            b"palette.schema_version": str(SCHEMA_VERSION).encode("ascii"),
            b"palette.training_response_config": json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        },
    )
    output_validation = validate_training_response_run(destination, run_id)
    return {
        **payload,
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
