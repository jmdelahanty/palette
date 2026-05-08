"""Build virtual collection manifests from explicit Palette analysis Zarr paths."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.virtual_collection_manifest import (
    assert_valid_manifest,
    with_manifest_sha256,
    write_manifest,
)
from fisheye.utils.zarr_io import open_zarr_root


PROFILE_MOVEMENT_BOUTS = "movement_bouts"

RUN_FAMILIES = (
    "track_kinematics_run",
    "swim_bout_run",
    "bout_kinematics_run",
    "eye_angle_run",
    "tail_kinematics_run",
    "stimulus_run",
    "stimulus_response_run",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _recording_id_from_path(zarr_path: Path) -> str:
    name = zarr_path.name
    if name.endswith(".zarr"):
        name = name[:-5]
    if name.endswith("_analysis"):
        name = name[:-9]
    return name


def _profile_spec(profile_id: str) -> dict[str, Any]:
    if profile_id != PROFILE_MOVEMENT_BOUTS:
        raise ValueError(f"Unsupported export profile for v1 builder: {profile_id}")
    return {
        "profile_id": PROFILE_MOVEMENT_BOUTS,
        "required_run_families": [
            "track_kinematics_run",
            "swim_bout_run",
            "bout_kinematics_run",
        ],
        "optional_run_families": [
            "eye_angle_run",
            "tail_kinematics_run",
            "stimulus_run",
            "stimulus_response_run",
        ],
    }


def _jsonish_attr(value: Any) -> Any:
    normalized = normalize_attr(value)
    if isinstance(normalized, (str, int, float, bool)) or normalized is None:
        return normalized
    if isinstance(normalized, Mapping):
        return {str(key): _jsonish_attr(item) for key, item in normalized.items()}
    if isinstance(normalized, (list, tuple)):
        return [_jsonish_attr(item) for item in normalized]
    return str(normalized)


def _attr(group: Any, *names: str) -> Any:
    attrs = getattr(group, "attrs", {})
    for name in names:
        try:
            value = attrs.get(name)
        except AttributeError:
            value = None
        if value is not None:
            return _jsonish_attr(value)
    return None


def _string_attr(group: Any, *names: str) -> str | None:
    value = _attr(group, *names)
    if value is None:
        return None
    text = str(value)
    return text or None


def _schema_version(group: Any) -> int | None:
    value = _attr(group, "schema_version", "output_schema_version")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _present_run_entry(
    *,
    run_id: str,
    path: str,
    group: Any,
    required: bool,
    selection: str,
) -> dict[str, Any]:
    source_fingerprint = _string_attr(
        group,
        "source_fingerprint",
        "source_lineage_hash",
        "lineage_hash",
    )
    lineage_hash = _string_attr(group, "lineage_hash", "source_lineage_hash")
    fingerprint_status = "complete" if source_fingerprint else "best_effort"
    return {
        "present": True,
        "run_id": run_id,
        "path": path,
        "required": required,
        "selection": selection,
        "schema_id": _string_attr(group, "schema_id"),
        "schema_version": _schema_version(group),
        "method": _string_attr(group, "method", "detection_method"),
        "method_version": _string_attr(group, "method_version"),
        "source_revision": _string_attr(
            group,
            "source_revision",
            "authoring_revision",
            "row_revision",
        ),
        "source_fingerprint": source_fingerprint,
        "fingerprint_status": fingerprint_status,
        "lineage_hash": lineage_hash,
    }


def _absent_run_entry(*, required: bool, reason: str) -> dict[str, Any]:
    return {
        "present": False,
        "required": required,
        "reason": reason,
        "run_id": None,
        "path": None,
        "fingerprint_status": "not_applicable",
    }


def _resolve_run_entry(
    root: Any,
    *,
    parent_path: str | Sequence[str],
    run_name: str | None,
    run_path_prefix: str,
    required: bool,
    run_label: str,
) -> tuple[dict[str, Any], str | None]:
    selection = "explicit" if run_name else "resolved_latest"
    try:
        group, resolved = resolve_zarr_run(
            root,
            parent_path,
            run_name=run_name,
            fallback_to_latest=True,
            fallback_to_sorted=None,
            latest_aliases=("latest",),
            run_label=run_label,
        )
    except Exception as exc:
        reason = "required_run_missing" if required else "not_generated"
        entry = _absent_run_entry(required=required, reason=reason)
        return entry, f"{run_label}: {exc}"
    return (
        _present_run_entry(
            run_id=resolved,
            path=f"{run_path_prefix}/{resolved}",
            group=group,
            required=required,
            selection=selection,
        ),
        None,
    )


def _training_locator_for_analysis(zarr_path: Path, *, storage_tier: str, verified_utc: str) -> dict[str, str] | None:
    text = str(zarr_path)
    if "_analysis.zarr" not in text:
        return None
    training_path = Path(text.replace("_analysis.zarr", "_training.zarr"))
    if not training_path.exists():
        return None
    return {
        "uri": str(training_path.expanduser().resolve()),
        "storage_tier": storage_tier,
        "last_verified_utc": verified_utc,
    }


def _build_record(
    zarr_path: Path,
    *,
    profile: dict[str, Any],
    storage_tier: str,
    verified_utc: str,
    track_kinematics_run: str | None,
    swim_bout_run: str | None,
    bout_kinematics_run: str | None,
    eye_angle_run: str | None,
    tail_kinematics_run: str | None,
    stimulus_run: str | None,
    stimulus_response_run: str | None,
) -> dict[str, Any]:
    resolved_path = zarr_path.expanduser().resolve()
    root = open_zarr_root(resolved_path, mode="r")
    recording_id = _recording_id_from_path(resolved_path)
    required_families = set(profile["required_run_families"])
    warnings: list[str] = []
    exclusions: list[str] = []

    run_specs = {
        "track_kinematics_run": {
            "parent_path": ("analysis", "track_kinematics_runs", "offline"),
            "run_name": track_kinematics_run,
            "run_path_prefix": "analysis/track_kinematics_runs/offline",
            "label": "Track kinematics run",
        },
        "swim_bout_run": {
            "parent_path": ("analysis", "swim_bout_runs"),
            "run_name": swim_bout_run,
            "run_path_prefix": "analysis/swim_bout_runs",
            "label": "Swim-bout run",
        },
        "bout_kinematics_run": {
            "parent_path": ("analysis", "bout_kinematics_runs"),
            "run_name": bout_kinematics_run,
            "run_path_prefix": "analysis/bout_kinematics_runs",
            "label": "Bout-kinematics run",
        },
        "eye_angle_run": {
            "parent_path": ("analysis", "eye_angle_runs"),
            "run_name": eye_angle_run,
            "run_path_prefix": "analysis/eye_angle_runs",
            "label": "Eye-angle run",
        },
        "tail_kinematics_run": {
            "parent_path": ("analysis", "tail_kinematics_runs"),
            "run_name": tail_kinematics_run,
            "run_path_prefix": "analysis/tail_kinematics_runs",
            "label": "Tail-kinematics run",
        },
        "stimulus_run": {
            "parent_path": ("analysis", "stimulus_runs"),
            "run_name": stimulus_run,
            "run_path_prefix": "analysis/stimulus_runs",
            "label": "Stimulus run",
        },
        "stimulus_response_run": {
            "parent_path": ("analysis", "stimulus_response_runs"),
            "run_name": stimulus_response_run,
            "run_path_prefix": "analysis/stimulus_response_runs",
            "label": "Stimulus-response run",
        },
    }

    source_runs: dict[str, dict[str, Any]] = {}
    for family in RUN_FAMILIES:
        spec = run_specs[family]
        required = family in required_families
        entry, warning = _resolve_run_entry(
            root,
            parent_path=spec["parent_path"],
            run_name=spec["run_name"],
            run_path_prefix=spec["run_path_prefix"],
            required=required,
            run_label=spec["label"],
        )
        source_runs[family] = entry
        if warning:
            if required:
                exclusions.append(warning)
            else:
                warnings.append(warning)

    included = not exclusions
    training_locator = _training_locator_for_analysis(
        resolved_path,
        storage_tier=storage_tier,
        verified_utc=verified_utc,
    )

    record: dict[str, Any] = {
        "recording_id": recording_id,
        "dataset_id": f"analysis_{recording_id}",
        "artifact_kind": "analysis_zarr",
        "locator_at_selection": {
            "uri": str(resolved_path),
            "storage_tier": storage_tier,
            "last_verified_utc": verified_utc,
        },
        "recording_attrs": {
            "recording_id": recording_id,
        },
        "protocol": {
            "stimulus_run_id": source_runs["stimulus_run"].get("run_id"),
            "protocol_signature_hash": _attr(root, "protocol_signature_hash", "derived_protocol_hash"),
            "protocol_semantic_hash": _attr(root, "protocol_semantic_hash"),
            "protocol_snapshot_sha256": _attr(root, "protocol_snapshot_sha256"),
        },
        "stimulus_response_validation": {
            "required": False,
            "status": "not_applicable",
            "stimulus_alignment_status": "not_requested",
            "protocol_hash_status": "not_requested",
            "direction_mapping_status": "not_requested",
            "calibration_status": "not_requested",
            "warnings": [],
        },
        "source_runs": source_runs,
        "status": {
            "included": included,
            "warnings": warnings,
            "exclusions": exclusions,
        },
    }
    if training_locator is not None:
        training_recording_id = recording_id
        record["training_dataset_id"] = f"training_{training_recording_id}"
        record["training_locator_at_selection"] = training_locator
    return record


def build_manifest_from_zarrs(
    zarr_paths: Sequence[Path],
    *,
    collection_id: str,
    collection_name: str,
    profile_id: str = PROFILE_MOVEMENT_BOUTS,
    created_by: str = "palette",
    storage_tier: str = "hot_nvme",
    created_utc: str | None = None,
    track_kinematics_run: str | None = None,
    swim_bout_run: str | None = None,
    bout_kinematics_run: str | None = None,
    eye_angle_run: str | None = None,
    tail_kinematics_run: str | None = None,
    stimulus_run: str | None = None,
    stimulus_response_run: str | None = None,
) -> dict[str, Any]:
    if not zarr_paths:
        raise ValueError("At least one analysis Zarr path is required")
    created = created_utc or _utc_now()
    profile = _profile_spec(profile_id)
    records = [
        _build_record(
            Path(path),
            profile=profile,
            storage_tier=storage_tier,
            verified_utc=created,
            track_kinematics_run=track_kinematics_run,
            swim_bout_run=swim_bout_run,
            bout_kinematics_run=bout_kinematics_run,
            eye_angle_run=eye_angle_run,
            tail_kinematics_run=tail_kinematics_run,
            stimulus_run=stimulus_run,
            stimulus_response_run=stimulus_response_run,
        )
        for path in zarr_paths
    ]
    manifest = {
        "schema_id": "palette.virtual_collection_manifest",
        "schema_version": 1,
        "collection_id": collection_id,
        "collection_name": collection_name,
        "created_utc": created,
        "created_by": created_by,
        "purpose": "cross_recording_analytics_export",
        "selection_policy": {
            "latest_allowed_during_selection": True,
            "latest_resolved_before_export": True,
            "production_requires_explicit_runs": True,
            "missing_optional_runs": "warn",
            "missing_required_runs": "exclude",
        },
        "query": {
            "registry_path": None,
            "registry_snapshot_sha256": None,
            "registry_snapshot_status": "not_registry_derived",
            "filters": {
                "explicit_zarr_paths": [str(Path(path).expanduser()) for path in zarr_paths],
            },
            "trial_descriptor_filters": {
                "status": "deferred",
                "note": "Future protocol_trial_index_json-backed search goes here.",
            },
            "ordering": ["input_order"],
        },
        "export_profiles": [profile],
        "records": records,
        "manifest_canonicalization": "json_sorted_keys_no_hash_fields_v1",
        "manifest_sha256": "computed_after_canonicalization",
    }
    assert_valid_manifest(manifest)
    return with_manifest_sha256(manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Palette virtual collection manifest from explicit analysis Zarr paths.",
    )
    parser.add_argument("zarr_paths", nargs="+", type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--profile", default=PROFILE_MOVEMENT_BOUTS, choices=[PROFILE_MOVEMENT_BOUTS])
    parser.add_argument("--created-by", default="palette")
    parser.add_argument("--storage-tier", default="hot_nvme")
    parser.add_argument("--track-kinematics-run")
    parser.add_argument("--swim-bout-run")
    parser.add_argument("--bout-kinematics-run")
    parser.add_argument("--eye-angle-run")
    parser.add_argument("--tail-kinematics-run")
    parser.add_argument("--stimulus-run")
    parser.add_argument("--stimulus-response-run")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest_from_zarrs(
            args.zarr_paths,
            collection_id=args.collection_id,
            collection_name=args.collection_name,
            profile_id=args.profile,
            created_by=args.created_by,
            storage_tier=args.storage_tier,
            track_kinematics_run=args.track_kinematics_run,
            swim_bout_run=args.swim_bout_run,
            bout_kinematics_run=args.bout_kinematics_run,
            eye_angle_run=args.eye_angle_run,
            tail_kinematics_run=args.tail_kinematics_run,
            stimulus_run=args.stimulus_run,
            stimulus_response_run=args.stimulus_response_run,
        )
        write_manifest(args.output, manifest, overwrite=args.overwrite)
        print(manifest["manifest_sha256"])
        return 0
    except Exception as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
