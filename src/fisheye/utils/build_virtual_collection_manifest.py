"""Build virtual collection manifests from Palette analysis Zarr datasets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from fisheye.analysis.chaser_distance_io import (
    CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID,
    CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_VERSION,
    VERIFIED_AUTHORITY_STATUS,
    ChaserDistanceReadError,
    ChaserDistanceReadSnapshot,
    load_chaser_distance_run,
)
from fisheye.shared.batch_logging import utc_now_z as _utc_now
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.virtual_collection_manifest import (
    assert_valid_manifest,
    with_manifest_sha256,
    write_manifest,
)
from fisheye.shared.zarr_io import open_zarr_root


PROFILE_MOVEMENT_BOUTS = "movement_bouts"
PROFILE_CHASER = "chaser"
# Compatibility input for already documented explicit-path commands. New
# registry-derived collections should use the protocol-neutral ``chaser`` id.
PROFILE_GOODCOPBADCOP_CHASER = "goodcopbadcop_chaser"


@dataclass(frozen=True)
class RegistryStimulusDataset:
    """One analysis dataset selected through normalized stimulus metadata."""

    dataset_id: str
    recording_id: str
    zarr_path: Path
    stimulus_run_id: str
    protocol_hash: str
    protocol_name: str | None
    stimulus_mode: str
    step_count: int
    total_duration_s: float | None


def _recording_id_from_path(zarr_path: Path) -> str:
    name = zarr_path.name
    if name.endswith(".zarr"):
        name = name[:-5]
    if name.endswith("_analysis"):
        name = name[:-9]
    return name


def _profile_spec(profile_id: str) -> dict[str, Any]:
    if profile_id == PROFILE_MOVEMENT_BOUTS:
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
    if profile_id in {PROFILE_CHASER, PROFILE_GOODCOPBADCOP_CHASER}:
        protocol_neutral = profile_id == PROFILE_CHASER
        return {
            "profile_id": (
                PROFILE_CHASER if protocol_neutral else PROFILE_GOODCOPBADCOP_CHASER
            ),
            "required_run_families": [
                "detection_occupancy_run",
                "chaser_distance_run",
                *(["track_kinematics_run"] if protocol_neutral else []),
            ],
            "optional_run_families": [
                *(
                    [
                        "swim_bout_run",
                        "bout_kinematics_run",
                        "eye_angle_run",
                        "tail_kinematics_run",
                        "stimulus_response_run",
                    ]
                    if protocol_neutral
                    else []
                ),
                "stimulus_run",
            ],
        }
    raise ValueError(f"Unsupported export profile for v1 builder: {profile_id}")


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
    fingerprint_status = _string_attr(
        group,
        "fingerprint_status",
        "lineage_fingerprint_status",
    )
    if fingerprint_status is None:
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


def _verified_chaser_run_entry(
    snapshot: ChaserDistanceReadSnapshot,
    *,
    required: bool,
    selection: str,
) -> dict[str, Any]:
    """Describe a chaser run only from detached, payload-verified authority."""

    if snapshot.authority_status != VERIFIED_AUTHORITY_STATUS:
        raise ChaserDistanceReadError(
            "Virtual-collection selection requires verified chaser-distance authority."
        )
    return {
        "present": True,
        "run_id": snapshot.run_name,
        "path": snapshot.run_path,
        "required": required,
        "selection": selection,
        "schema_id": CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_ID,
        "schema_version": CHASER_DISTANCE_READ_AUTHORITY_SCHEMA_VERSION,
        "method": None,
        "method_version": None,
        "source_revision": None,
        "source_fingerprint": snapshot.publication_seal_sha256,
        "fingerprint_status": "complete",
        "lineage_hash": snapshot.surface_manifest_sha256,
        "authority_status": snapshot.authority_status,
        "publication_seal_ref": snapshot.publication_seal_ref,
        "publication_seal_sha256": snapshot.publication_seal_sha256,
        "surface_manifest_ref": snapshot.surface_manifest_ref,
        "surface_manifest_sha256": snapshot.surface_manifest_sha256,
        "row_identity_ref": snapshot.row_identity_ref,
        "row_identity_sha256": snapshot.row_identity_sha256,
    }


def _parent_path_text(parent_path: str | Sequence[str]) -> str:
    if isinstance(parent_path, str):
        return parent_path.strip("/")
    return "/".join(str(part).strip("/") for part in parent_path)


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
    if _parent_path_text(parent_path) == "analysis/chaser_distance_runs":
        # A collection manifest can become downstream discovery/scientific
        # authority. Chaser selection therefore pays the full canonical
        # payload-verification cost and records only detached typed evidence;
        # generic selectors and raw child attrs are not compatibility paths.
        try:
            snapshot = load_chaser_distance_run(
                root,
                run_name=str(run_name or "latest"),
            )
        except ChaserDistanceReadError as exc:
            reason = "required_run_incompatible" if required else "not_generated"
            entry = _absent_run_entry(required=required, reason=reason)
            return entry, f"{run_label}: canonical preflight failed closed: {exc}"
        return (
            _verified_chaser_run_entry(
                snapshot,
                required=required,
                selection=selection,
            ),
            None,
        )
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
    detection_occupancy_run: str | None,
    chaser_distance_run: str | None,
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
        "detection_occupancy_run": {
            "parent_path": ("analysis", "detection_occupancy_runs"),
            "run_name": detection_occupancy_run,
            "run_path_prefix": "analysis/detection_occupancy_runs",
            "label": "Detection-occupancy run",
        },
        "chaser_distance_run": {
            "parent_path": ("analysis", "chaser_distance_runs"),
            "run_name": chaser_distance_run,
            "run_path_prefix": "analysis/chaser_distance_runs",
            "label": "Chaser-distance run",
        },
    }

    source_runs: dict[str, dict[str, Any]] = {}
    selected_families = tuple(
        dict.fromkeys(
            list(profile["required_run_families"]) + list(profile["optional_run_families"])
        )
    )
    for family in selected_families:
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
            "stimulus_run_id": source_runs.get("stimulus_run", {}).get("run_id"),
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
    detection_occupancy_run: str | None = None,
    chaser_distance_run: str | None = None,
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
            detection_occupancy_run=detection_occupancy_run,
            chaser_distance_run=chaser_distance_run,
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


def _connect_registry_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def select_registry_stimulus_datasets(
    registry_path: Path,
    *,
    stimulus_mode: str,
    zarr_use: str = "analysis",
    dataset_status: str = "active",
) -> list[RegistryStimulusDataset]:
    """Select active analysis datasets by the latest normalized stimulus mode.

    Protocol names are deliberately not part of the predicate. This allows one
    collection to include RedScare, GoodCopBadCop, and future protocols that
    persist the same normalized stimulus capability.
    """

    mode = str(stimulus_mode).strip().upper()
    if not mode:
        raise ValueError("stimulus_mode must be non-empty")
    sql = """
        SELECT
            rsmc.dataset_id,
            COALESCE(NULLIF(TRIM(rsmc.recording_id), ''), dcc.recording_id) AS recording_id,
            dcc.zarr_path,
            rsmc.stimulus_run_id,
            rsmc.protocol_hash,
            COALESCE(NULLIF(TRIM(rsmc.protocol_name), ''), dcc.protocol_name) AS protocol_name,
            rsmc.stimulus_mode,
            rsmc.step_count,
            rsmc.total_duration_s
        FROM recording_stimulus_mode_counts rsmc
        JOIN dataset_context_current dcc ON dcc.dataset_id = rsmc.dataset_id
        WHERE UPPER(TRIM(rsmc.stimulus_mode)) = ?
          AND rsmc.is_latest = 1
          AND dcc.zarr_use = ? COLLATE NOCASE
          AND dcc.dataset_status = ? COLLATE NOCASE
        ORDER BY recording_id, rsmc.dataset_id, rsmc.stimulus_run_id
    """
    try:
        with closing(_connect_registry_read_only(registry_path)) as conn:
            rows = conn.execute(sql, (mode, zarr_use, dataset_status)).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "Registry stimulus-mode query failed; refresh/migrate the registry so "
            "recording_stimulus_mode_counts and dataset_context_current are available"
        ) from exc

    datasets: list[RegistryStimulusDataset] = []
    seen_dataset_ids: set[str] = set()
    for row in rows:
        dataset_id = str(row["dataset_id"])
        if dataset_id in seen_dataset_ids:
            raise RuntimeError(
                f"Registry has multiple latest {mode} stimulus rows for dataset {dataset_id!r}"
            )
        seen_dataset_ids.add(dataset_id)
        recording_id = str(row["recording_id"] or "").strip()
        if not recording_id:
            raise RuntimeError(f"Registry dataset {dataset_id!r} has no recording_id")
        datasets.append(
            RegistryStimulusDataset(
                dataset_id=dataset_id,
                recording_id=recording_id,
                zarr_path=Path(str(row["zarr_path"])).expanduser().resolve(strict=False),
                stimulus_run_id=str(row["stimulus_run_id"]),
                protocol_hash=str(row["protocol_hash"]),
                protocol_name=(
                    str(row["protocol_name"]) if row["protocol_name"] is not None else None
                ),
                stimulus_mode=str(row["stimulus_mode"]).strip().upper(),
                step_count=int(row["step_count"]),
                total_duration_s=(
                    float(row["total_duration_s"])
                    if row["total_duration_s"] is not None
                    else None
                ),
            )
        )
    return datasets


def build_manifest_from_registry(
    registry_path: Path,
    *,
    stimulus_mode: str,
    collection_id: str,
    collection_name: str,
    profile_id: str = PROFILE_CHASER,
    created_by: str = "palette",
    storage_tier: str = "shared_groups",
    created_utc: str | None = None,
    zarr_use: str = "analysis",
    dataset_status: str = "active",
    **run_overrides: str | None,
) -> dict[str, Any]:
    """Build a collection from registry-normalized stimulus capabilities."""

    registry = registry_path.expanduser().resolve(strict=True)
    selected = select_registry_stimulus_datasets(
        registry,
        stimulus_mode=stimulus_mode,
        zarr_use=zarr_use,
        dataset_status=dataset_status,
    )
    if not selected:
        raise ValueError(
            f"No {dataset_status!r} {zarr_use!r} datasets have latest normalized "
            f"stimulus mode {str(stimulus_mode).strip().upper()!r}"
        )
    manifest = build_manifest_from_zarrs(
        [item.zarr_path for item in selected],
        collection_id=collection_id,
        collection_name=collection_name,
        profile_id=profile_id,
        created_by=created_by,
        storage_tier=storage_tier,
        created_utc=created_utc,
        **run_overrides,
    )

    selections_by_path = {str(item.zarr_path): item for item in selected}
    for record in manifest["records"]:
        selection = selections_by_path[record["locator_at_selection"]["uri"]]
        record["dataset_id"] = selection.dataset_id
        record["recording_id"] = selection.recording_id
        record["recording_attrs"].update(
            {
                "recording_id": selection.recording_id,
                "protocol_name": selection.protocol_name,
                "selected_normalized_stimulus_mode": selection.stimulus_mode,
            }
        )
        record["protocol"].update(
            {
                "stimulus_run_id": selection.stimulus_run_id,
                "protocol_hash": selection.protocol_hash,
                "protocol_name": selection.protocol_name,
                "selected_stimulus_mode": selection.stimulus_mode,
                "selected_mode_step_count": selection.step_count,
                "selected_mode_total_duration_s": selection.total_duration_s,
            }
        )

    mode = str(stimulus_mode).strip().upper()
    manifest["query"] = {
        "registry_path": str(registry),
        "registry_snapshot_sha256": None,
        # The live read-only registry is query provenance, not an immutable copy.
        "registry_snapshot_status": "not_recorded",
        "filters": {
            "normalized_stimulus_mode": mode,
            "latest_stimulus_run_only": True,
            "zarr_use": zarr_use,
            "dataset_status": dataset_status,
            "protocol_name": None,
        },
        "trial_descriptor_filters": {
            "status": "not_applied",
            "note": "Cohort membership is based on normalized stimulus mode, not protocol name.",
        },
        "ordering": ["recording_id", "dataset_id", "stimulus_run_id"],
    }
    assert_valid_manifest(manifest)
    return with_manifest_sha256(manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Palette virtual collection manifest from explicit analysis Zarr paths "
            "or a read-only registry stimulus-mode query."
        ),
    )
    parser.add_argument("zarr_paths", nargs="*", type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--profile",
        default=None,
        choices=[PROFILE_MOVEMENT_BOUTS, PROFILE_CHASER, PROFILE_GOODCOPBADCOP_CHASER],
        help="Defaults to chaser for a CHASER registry query, otherwise movement_bouts.",
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--stimulus-mode",
        help="Normalized stimulus mode (for example CHASER); requires --registry.",
    )
    parser.add_argument("--zarr-use", default="analysis")
    parser.add_argument("--dataset-status", default="active")
    parser.add_argument("--created-by", default="palette")
    parser.add_argument(
        "--storage-tier",
        default=None,
        help="Defaults to shared_groups for registry queries and hot_nvme for explicit paths.",
    )
    parser.add_argument("--track-kinematics-run")
    parser.add_argument("--swim-bout-run")
    parser.add_argument("--bout-kinematics-run")
    parser.add_argument("--eye-angle-run")
    parser.add_argument("--tail-kinematics-run")
    parser.add_argument("--stimulus-run")
    parser.add_argument("--stimulus-response-run")
    parser.add_argument("--detection-occupancy-run")
    parser.add_argument("--chaser-distance-run")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.registry is not None or args.stimulus_mode is not None:
            if args.registry is None or args.stimulus_mode is None:
                raise ValueError("--registry and --stimulus-mode must be provided together")
            if args.zarr_paths:
                raise ValueError("Do not combine explicit zarr_paths with --registry")
            manifest_builder = build_manifest_from_registry
            source_args = {
                "registry_path": args.registry,
                "stimulus_mode": args.stimulus_mode,
                "zarr_use": args.zarr_use,
                "dataset_status": args.dataset_status,
            }
            default_profile = (
                PROFILE_CHASER
                if str(args.stimulus_mode).strip().upper() == "CHASER"
                else PROFILE_MOVEMENT_BOUTS
            )
            default_storage_tier = "shared_groups"
        else:
            if not args.zarr_paths:
                raise ValueError(
                    "Provide at least one zarr_path or --registry with --stimulus-mode"
                )
            manifest_builder = build_manifest_from_zarrs
            source_args = {"zarr_paths": args.zarr_paths}
            default_profile = PROFILE_MOVEMENT_BOUTS
            default_storage_tier = "hot_nvme"
        manifest = manifest_builder(
            **source_args,
            collection_id=args.collection_id,
            collection_name=args.collection_name,
            profile_id=args.profile or default_profile,
            created_by=args.created_by,
            storage_tier=args.storage_tier or default_storage_tier,
            track_kinematics_run=args.track_kinematics_run,
            swim_bout_run=args.swim_bout_run,
            bout_kinematics_run=args.bout_kinematics_run,
            eye_angle_run=args.eye_angle_run,
            tail_kinematics_run=args.tail_kinematics_run,
            stimulus_run=args.stimulus_run,
            stimulus_response_run=args.stimulus_response_run,
            detection_occupancy_run=args.detection_occupancy_run,
            chaser_distance_run=args.chaser_distance_run,
        )
        write_manifest(args.output, manifest, overwrite=args.overwrite)
        print(manifest["manifest_sha256"])
        return 0
    except Exception as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
