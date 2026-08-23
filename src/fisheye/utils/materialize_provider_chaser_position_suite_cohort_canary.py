"""Plan and publish a bounded provider-aware chaser position-suite cohort.

``plan`` resolves the deterministic earliest/latest recording in every arena and
emits a checksummed task containing exact provider, geometry-selection, and
physical-scale authorities.  It writes nothing.  ``run`` accepts only that
frozen task, revalidates every authority, and computes recording-balanced
cohort evidence.  ``run --apply`` atomically publishes selector-ineligible
operational artifacts; it never writes an analysis Zarr, registry, or selector.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fisheye.analysis.provider_chaser_distance_comparison_cohort import (  # noqa: E402
    select_arena_extremes,
)
from fisheye.shared.json_safety import json_attr_safe  # noqa: E402
from fisheye.shared.system_metadata import get_git_info  # noqa: E402
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256  # noqa: E402
from fisheye.utils.materialize_provider_chaser_position_suite_canary import (  # noqa: E402
    OUTPUT_FILES as RECORDING_OUTPUT_FILES,
    build_canary,
    publish_operational_canary,
)


SCHEMA_ID = "palette.provider_chaser_position_suite_cohort_canary"
SCHEMA_VERSION = 2
TASK_SCHEMA_ID = f"{SCHEMA_ID}.task"
TASK_SCHEMA_VERSION = 1
SELECTION_POLICY_ID = "earliest_latest_eligible_recording_per_arena_v1"
AGGREGATION_POLICY_ID = "recording_is_analysis_unit_no_frame_pooling_v1"
DISPOSITION = "selector_ineligible_operational_canary"
EXPECTED_ARENAS = ("arena_1", "arena_2", "arena_3", "arena_4")
COHORT_OUTPUT_FILES = (
    "cohort_report.json",
    "source_task.json",
    "recording_summary.csv",
    "per_epoch_chaser_metrics.csv",
    "role_contrasts.csv",
    "radial_recording_rows.csv",
    "radial_cohort_summary.csv",
    "distance_summary.png",
    "near_field_summary.png",
    "quadrant_summary.png",
    "radial_summary.png",
)
_ARENA_RE = re.compile(r"(?:^|_)arena_(?P<arena>[0-9]+)(?:_|$)")
_SUMMARY_METRICS = (
    "distance_p50_mm",
    "valid_distance_fraction",
    "same_quadrant_fraction_valid",
    "near_zone_fraction_valid",
    "near_zone_entry_rate_per_min_valid_time",
    "fish_arena_radius_mean_mm",
    "fish_wall_distance_mean_mm",
)


class ProviderChaserPositionSuiteCohortError(ValueError):
    """Raised when cohort planning, validation, or publication is unsafe."""


def _fail(message: str) -> None:
    raise ProviderChaserPositionSuiteCohortError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return _plain(value)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one nonempty canonical string.")
    return value


def _exact_name(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if (
        result in {"latest", "latest_complete", "selected", "current", ".", ".."}
        or "/" in result
        or "\\" in result
        or any(character.isspace() for character in result)
    ):
        _fail(f"{field} must be one exact immutable non-selector name.")
    return result


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _arena(recording_id: str) -> str:
    match = _ARENA_RE.search(recording_id)
    if match is None:
        _fail(f"Recording identity has no canonical arena token: {recording_id!r}.")
    return f"arena_{int(match.group('arena'))}"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zarr_attrs(group_path: Path, *, field: str) -> dict[str, Any]:
    metadata = group_path / "zarr.json"
    if not metadata.is_file():
        _fail(f"{field} metadata does not exist: {metadata}")
    document = _mapping(json.loads(metadata.read_bytes()), field=f"{field} metadata")
    return _mapping(document.get("attributes"), field=f"{field} attributes")


def _verified_record(
    attrs: Mapping[str, Any], *, record_key: str, digest_key: str, field: str
) -> tuple[dict[str, Any], str]:
    record = _mapping(attrs.get(record_key), field=f"{field} record")
    persisted = _digest(attrs.get(digest_key), field=f"{field} persisted digest")
    observed = canonical_json_sha256(record)
    if observed != persisted:
        _fail(f"{field} persisted digest is stale.")
    return record, observed


def _task_digest(task: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in task.items() if key != "task_sha256"}
    return canonical_json_sha256(payload)


def resolve_cohort_task(
    cohort_inputs: str | Path,
    *,
    provider_key: str = "detection_bbox_centroid",
    epoch_role_bindings: Sequence[tuple[str, int]] = (
        ("pre", 0),
        ("training", 1),
        ("post", 2),
    ),
    treatment_role: str = "aggressive",
    baseline_role: str = "inert",
    radial_bin_width_mm: float = 2.0,
    near_zone_radius_mm: float = 5.0,
    near_entry_radius_mm: float = 5.0,
    near_exit_radius_mm: float = 6.0,
    perimeter_band_mm: float = 5.0,
    min_expected_count: float = 5.0,
) -> dict[str, Any]:
    """Resolve mutable selectors once and return a frozen, no-write task."""

    source_path = Path(cohort_inputs).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Cohort input document does not exist: {source_path}")
    source_bytes = source_path.read_bytes()
    source = _mapping(json.loads(source_bytes), field="cohort input document")
    raw_entries = source.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        _fail("Cohort input document has no entries.")
    provider_key = _exact_name(provider_key, field="provider key")

    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_entries:
        entry = _mapping(raw, field="cohort input entry")
        recording_id = _text(entry.get("recording_id"), field="recording identity")
        if recording_id in seen:
            _fail(f"Duplicate recording identity in cohort input: {recording_id!r}.")
        seen.add(recording_id)
        providers = _mapping(entry.get("providers"), field="provider declarations")
        provider = _mapping(
            providers.get(provider_key), field=f"provider {provider_key!r}"
        )
        _exact_name(provider.get("sealed_run_name"), field="sealed provider run")
        eligible.append(
            {
                **entry,
                "recording_id": recording_id,
                "analysis_zarr": str(
                    Path(_text(entry.get("analysis_zarr"), field="analysis Zarr"))
                    .expanduser()
                    .resolve()
                ),
                "arena": _arena(recording_id),
            }
        )

    selected, selection = select_arena_extremes(eligible)
    if selection.get("policy_id") != SELECTION_POLICY_ID:
        _fail("Unexpected cohort selection policy.")
    if tuple(item["arena"] for item in selection["strata"]) != EXPECTED_ARENAS:
        _fail("Bounded canary must contain exactly arenas 1 through 4.")
    if len(selected) != 8:
        _fail("Bounded canary must contain exactly eight recordings.")

    planned_entries: list[dict[str, Any]] = []
    for selected_entry in selected:
        recording_id = str(selected_entry["recording_id"])
        arena = str(selected_entry["arena"])
        archive = Path(str(selected_entry["analysis_zarr"]))
        if not archive.is_dir():
            _fail(f"Analysis Zarr does not exist for {recording_id!r}: {archive}")
        expected_basename = f"{recording_id}_analysis.zarr"
        if archive.name != expected_basename:
            _fail(f"Analysis Zarr basename disagrees with {recording_id!r}.")

        provider = _mapping(
            _mapping(selected_entry["providers"], field="provider declarations").get(
                provider_key
            ),
            field=f"provider {provider_key!r}",
        )
        provider_run = _exact_name(
            provider.get("sealed_run_name"), field="sealed provider run"
        )
        provider_attrs = _zarr_attrs(
            archive / "analysis" / "provider_chaser_distance_runs" / provider_run,
            field="sealed provider run",
        )
        provider_manifest, provider_manifest_sha = _verified_record(
            provider_attrs,
            record_key="provider_chaser_distance_manifest",
            digest_key="provider_chaser_distance_manifest_sha256",
            field="sealed provider manifest",
        )
        if provider_manifest.get("recording_id") != recording_id:
            _fail(
                f"Provider manifest recording identity mismatch for {recording_id!r}."
            )
        if provider_manifest.get("run_name") != provider_run:
            _fail(f"Provider manifest run identity mismatch for {recording_id!r}.")

        selection_parent = _zarr_attrs(
            archive / "analysis" / "arena_geometry_selection",
            field="arena geometry selection parent",
        )
        latest = selection_parent.get("latest")
        latest_complete = selection_parent.get("latest_complete")
        if latest != latest_complete:
            _fail(f"Geometry selectors disagree for {recording_id!r}.")
        geometry_run = _exact_name(
            latest_complete, field="complete geometry selection run"
        )
        geometry_attrs = _zarr_attrs(
            archive / "analysis" / "arena_geometry_selection" / geometry_run,
            field="arena geometry selection run",
        )
        selection_record, selection_sha = _verified_record(
            geometry_attrs,
            record_key="selection_record",
            digest_key="selection_record_sha256",
            field="arena geometry selection",
        )
        if geometry_attrs.get("selection_id") != geometry_run:
            _fail(f"Geometry selection identity mismatch for {recording_id!r}.")
        selected_candidate = _mapping(
            selection_record.get("selected_candidate"), field="selected candidate"
        )
        arena_binding = _mapping(
            selected_candidate.get("arena_binding"), field="selected arena binding"
        )
        if arena_binding.get("arena_id") != arena:
            _fail(f"Selected geometry arena mismatch for {recording_id!r}.")
        camera_serial = _text(
            arena_binding.get("camera_serial"), field="selected camera serial"
        )

        physical_attrs = _zarr_attrs(
            archive / "analysis" / "calibration" / "coordinate_frames",
            field="source-camera physical authority",
        )
        physical_record, physical_sha = _verified_record(
            physical_attrs,
            record_key="source_camera_physical_authority",
            digest_key="source_camera_physical_authority_sha256",
            field="source-camera physical authority",
        )
        if str(physical_record.get("camera_id")) != camera_serial:
            _fail(f"Physical authority camera mismatch for {recording_id!r}.")

        planned_entries.append(
            {
                "recording_id": recording_id,
                "arena": arena,
                "camera_serial": camera_serial,
                "analysis_zarr": str(archive),
                "provider_run": provider_run,
                "provider_manifest_sha256": provider_manifest_sha,
                "geometry_selection_run": geometry_run,
                "geometry_selection_record_sha256": selection_sha,
                "physical_authority_sha256": physical_sha,
            }
        )

    task = json_attr_safe(
        {
            "schema_id": TASK_SCHEMA_ID,
            "schema_version": TASK_SCHEMA_VERSION,
            "disposition": DISPOSITION,
            "source_cohort_document": {
                "path": str(source_path),
                "sha256": _sha256_bytes(source_bytes),
                "schema_id": source.get("schema_id"),
                "schema_version": source.get("schema_version"),
                "declared_recording_count": source.get("recording_count"),
                "observed_recording_count": len(eligible),
                "plan_digest": source.get("plan_digest"),
            },
            "selection": selection,
            "provider_key": provider_key,
            "epoch_role_bindings": [
                {"analysis_role": role, "epoch_window_id": int(window_id)}
                for role, window_id in epoch_role_bindings
            ],
            "method_config": {
                "treatment_role": treatment_role,
                "baseline_role": baseline_role,
                "radial_bin_width_mm": radial_bin_width_mm,
                "cdf_threshold_policy": "derived_full_radial_edge_axis_v1",
                "near_zone_radius_mm": near_zone_radius_mm,
                "near_entry_radius_mm": near_entry_radius_mm,
                "near_exit_radius_mm": near_exit_radius_mm,
                "perimeter_band_mm": perimeter_band_mm,
                "min_expected_count": min_expected_count,
            },
            "aggregation_policy": {
                "policy_id": AGGREGATION_POLICY_ID,
                "analysis_unit": "recording",
                "frame_pooling_across_recordings": False,
                "animal_pooling": False,
                "inferential_statistics": False,
            },
            "entries": planned_entries,
            "safety": {
                "selector_eligible": False,
                "production_authority": False,
                "registry_update": False,
                "analysis_zarr_writes": False,
            },
        }
    )
    task["task_sha256"] = _task_digest(task)
    return task


def load_cohort_task(task_source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and strictly validate one frozen cohort task."""

    if isinstance(task_source, Mapping):
        task = _mapping(task_source, field="cohort task")
    else:
        path = Path(task_source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cohort task does not exist: {path}")
        task = _mapping(json.loads(path.read_bytes()), field="cohort task")
    if task.get("schema_id") != TASK_SCHEMA_ID or task.get("schema_version") != 1:
        _fail("Cohort task schema is unsupported.")
    persisted = _digest(task.get("task_sha256"), field="cohort task digest")
    if _task_digest(task) != persisted:
        _fail("Cohort task digest is stale.")
    source = _mapping(task.get("source_cohort_document"), field="source cohort")
    source_path = Path(_text(source.get("path"), field="source cohort path"))
    expected_source_sha = _digest(source.get("sha256"), field="source cohort digest")
    if not source_path.is_file() or _sha256_file(source_path) != expected_source_sha:
        _fail("Frozen source cohort document is absent or has changed.")
    safety = _mapping(task.get("safety"), field="task safety")
    expected_safety = {
        "selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "analysis_zarr_writes": False,
    }
    if safety != expected_safety:
        _fail("Cohort task does not retain the required no-authority safety state.")
    entries = task.get("entries")
    if not isinstance(entries, list) or len(entries) != 8:
        _fail("Cohort task must bind exactly eight recordings.")
    selected_ids = _mapping(task.get("selection"), field="selection").get(
        "selected_recording_ids"
    )
    if not isinstance(selected_ids, list):
        _fail("Cohort task selection lacks exact recording identities.")
    observed_ids: list[str] = []
    arena_counts: dict[str, int] = {}
    for raw in entries:
        entry = _mapping(raw, field="cohort task entry")
        recording_id = _text(entry.get("recording_id"), field="recording identity")
        arena = _text(entry.get("arena"), field="arena identity")
        if arena != _arena(recording_id):
            _fail(f"Arena identity mismatch for {recording_id!r}.")
        observed_ids.append(recording_id)
        arena_counts[arena] = arena_counts.get(arena, 0) + 1
        for field in ("provider_run", "geometry_selection_run"):
            _exact_name(entry.get(field), field=field)
        for field in (
            "provider_manifest_sha256",
            "geometry_selection_record_sha256",
            "physical_authority_sha256",
        ):
            _digest(entry.get(field), field=field)
    if observed_ids != selected_ids:
        _fail("Task entries do not preserve the deterministic selected order.")
    if arena_counts != {arena: 2 for arena in EXPECTED_ARENAS}:
        _fail("Cohort task does not bind two recordings in each expected arena.")
    return task


def _finite_summary(values: Sequence[object]) -> dict[str, float | int | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "recording_count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "recording_count": int(array.size),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _camera_from_report(report: Mapping[str, Any]) -> str:
    source = _mapping(
        report["source_bindings"]["provider_chaser_distance"][
            "source_position_provider"
        ],
        field="source position provider",
    )
    coordinate = _text(
        source.get("coordinate_authority_id"), field="coordinate authority"
    )
    match = re.search(r"/source_camera/(?P<camera>[^/]+)/", coordinate)
    if match is None:
        _fail("Provider coordinate authority has no camera identity.")
    return match.group("camera")


def _aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    recording_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    for report in reports:
        recording_id = _text(report.get("recording_id"), field="recording identity")
        arena = _arena(recording_id)
        camera = _camera_from_report(report)
        suite = _mapping(report.get("suite"), field="position suite")
        metrics = suite.get("per_epoch_chaser_metrics")
        contrasts = suite.get("role_contrasts")
        radial = suite.get("radial_occupancy")
        if not isinstance(metrics, list) or len(metrics) != 6:
            _fail(f"Expected six epoch/chaser rows for {recording_id!r}.")
        if not isinstance(contrasts, list) or not contrasts:
            _fail(f"Role contrasts are absent for {recording_id!r}.")
        if not isinstance(radial, list) or not radial:
            _fail(f"Radial occupancy is absent for {recording_id!r}.")
        prefix = {
            "recording_id": recording_id,
            "arena": arena,
            "camera_serial": camera,
        }
        local_metrics = [
            {**prefix, **_mapping(row, field="metric row")} for row in metrics
        ]
        metric_rows.extend(local_metrics)
        contrast_rows.extend(
            {**prefix, **_mapping(row, field="contrast row")} for row in contrasts
        )
        radial_rows.extend(
            {**prefix, **_mapping(row, field="radial row")} for row in radial
        )
        provider = report["source_bindings"]["provider_chaser_distance"]
        geometry = report["source_bindings"]["arena_geometry_and_scale"]
        recording_rows.append(
            {
                **prefix,
                "frame_count": int(suite["frame_count"]),
                "mm_per_pixel": float(suite["mm_per_pixel"]),
                "arena_radius_mm": float(suite["arena"]["radius_mm"]),
                "minimum_epoch_provider_frame_coverage_fraction": min(
                    float(row["epoch_provider_frame_coverage_fraction"])
                    for row in local_metrics
                ),
                "minimum_valid_distance_fraction": min(
                    float(row["valid_distance_fraction"]) for row in local_metrics
                ),
                "provider_manifest_sha256": provider["manifest_sha256"],
                "geometry_selection_record_sha256": geometry["selection"]["sha256"],
                "physical_authority_sha256": geometry[
                    "source_camera_physical_authority"
                ]["sha256"],
            }
        )

    distributions: list[dict[str, Any]] = []
    epoch_role_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = (
            str(row["analysis_role"]),
            int(row["epoch_window_id"]),
            str(row["behavior_role"]),
        )
        epoch_role_groups.setdefault(key, []).append(row)
    for (analysis_role, window_id, behavior_role), rows in sorted(
        epoch_role_groups.items()
    ):
        if len({str(row["recording_id"]) for row in rows}) != len(rows):
            _fail("Epoch/role aggregation would count one recording more than once.")
        for metric in _SUMMARY_METRICS:
            distributions.append(
                {
                    "analysis_role": analysis_role,
                    "epoch_window_id": window_id,
                    "behavior_role": behavior_role,
                    "metric": metric,
                    **_finite_summary([row.get(metric) for row in rows]),
                }
            )

    radial_summary: list[dict[str, Any]] = []
    radial_groups: dict[tuple[str, int, str, float, float], list[dict[str, Any]]] = {}
    for row in radial_rows:
        key = (
            str(row["analysis_role"]),
            int(row["epoch_window_id"]),
            str(row["behavior_role"]),
            float(row["bin_start_mm"]),
            float(row["bin_end_mm"]),
        )
        radial_groups.setdefault(key, []).append(row)
    for key, rows in sorted(radial_groups.items()):
        analysis_role, window_id, behavior_role, start, end = key
        if len({str(row["recording_id"]) for row in rows}) != len(rows):
            _fail("Radial aggregation would count one recording more than once.")
        radial_summary.append(
            {
                "analysis_role": analysis_role,
                "epoch_window_id": window_id,
                "behavior_role": behavior_role,
                "bin_start_mm": start,
                "bin_end_mm": end,
                **{
                    f"{metric}_{statistic}": value
                    for metric in (
                        "observed_fraction",
                        "expected_fraction_geometric",
                        "selection_index_geometric",
                    )
                    for statistic, value in _finite_summary(
                        [row.get(metric) for row in rows]
                    ).items()
                },
            }
        )

    contrast_distributions: list[dict[str, Any]] = []
    contrast_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in contrast_rows:
        key = (
            str(row["analysis_role"]),
            int(row["epoch_window_id"]),
            str(row["metric"]),
        )
        contrast_groups.setdefault(key, []).append(row)
    for (analysis_role, window_id, metric), rows in sorted(contrast_groups.items()):
        contrast_distributions.append(
            {
                "analysis_role": analysis_role,
                "epoch_window_id": window_id,
                "metric": metric,
                "contrast": "aggressive_minus_inert",
                **_finite_summary(
                    [row.get("treatment_minus_baseline") for row in rows]
                ),
            }
        )
    return (
        recording_rows,
        metric_rows,
        contrast_rows,
        radial_rows,
        distributions,
        radial_summary,
        contrast_distributions,
    )


def build_cohort_canary(
    task_source: str | Path | Mapping[str, Any], *, workers: int = 4
) -> dict[str, Any]:
    """Compute the exact frozen cohort without writing any source authority."""

    if workers < 1 or workers > 8:
        _fail("workers must be between 1 and 8.")
    task = load_cohort_task(task_source)
    config = _mapping(task.get("method_config"), field="method configuration")
    bindings = task.get("epoch_role_bindings")
    if not isinstance(bindings, list) or not bindings:
        _fail("Task has no epoch-role bindings.")
    epoch_roles = [
        (
            _exact_name(item.get("analysis_role"), field="analysis role"),
            int(item.get("epoch_window_id")),
        )
        for item in (_mapping(value, field="epoch-role binding") for value in bindings)
    ]

    def build(raw: Mapping[str, Any]) -> dict[str, Any]:
        entry = _mapping(raw, field="cohort task entry")
        report = build_canary(
            entry["analysis_zarr"],
            provider_run=entry["provider_run"],
            geometry_selection_run=entry["geometry_selection_run"],
            expected_selection_record_sha256=entry["geometry_selection_record_sha256"],
            expected_physical_authority_sha256=entry["physical_authority_sha256"],
            epoch_role_bindings=epoch_roles,
            treatment_role=str(config["treatment_role"]),
            baseline_role=str(config["baseline_role"]),
            radial_bin_width_mm=float(config["radial_bin_width_mm"]),
            cdf_thresholds_mm=None,
            near_zone_radius_mm=float(config["near_zone_radius_mm"]),
            near_entry_radius_mm=float(config["near_entry_radius_mm"]),
            near_exit_radius_mm=float(config["near_exit_radius_mm"]),
            perimeter_band_mm=float(config["perimeter_band_mm"]),
            min_expected_count=float(config["min_expected_count"]),
        )
        if report["recording_id"] != entry["recording_id"]:
            _fail("Computed report recording identity differs from the frozen task.")
        source = report["source_bindings"]["provider_chaser_distance"]
        geometry = report["source_bindings"]["arena_geometry_and_scale"]
        observed = {
            "provider_manifest_sha256": source["manifest_sha256"],
            "geometry_selection_record_sha256": geometry["selection"]["sha256"],
            "physical_authority_sha256": geometry["source_camera_physical_authority"][
                "sha256"
            ],
        }
        for field, value in observed.items():
            if value != entry[field]:
                _fail(f"Computed {field} differs from the frozen task.")
        return report

    entries = task["entries"]
    with ThreadPoolExecutor(max_workers=min(workers, len(entries))) as executor:
        reports = list(executor.map(build, entries))
    if [report["recording_id"] for report in reports] != [
        entry["recording_id"] for entry in entries
    ]:
        _fail("Cohort reports do not preserve frozen task order.")
    (
        recording_rows,
        metric_rows,
        contrast_rows,
        radial_rows,
        distributions,
        radial_summary,
        contrast_distributions,
    ) = _aggregate_reports(reports)
    camera_by_arena: dict[str, set[str]] = {}
    for row in recording_rows:
        camera_by_arena.setdefault(row["arena"], set()).add(row["camera_serial"])
    if any(len(cameras) != 1 for cameras in camera_by_arena.values()):
        _fail("An arena binds more than one source-camera authority in the cohort.")
    return json_attr_safe(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "disposition": DISPOSITION,
            "status": "computed_read_only",
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "source_task_sha256": task["task_sha256"],
            "source_cohort_document": task["source_cohort_document"],
            "selection": task["selection"],
            "provider_key": task["provider_key"],
            "method_config": config,
            "aggregation_policy": task["aggregation_policy"],
            "camera_authority_by_arena": {
                arena: next(iter(cameras))
                for arena, cameras in sorted(camera_by_arena.items())
            },
            "recording_summary": recording_rows,
            "recording_level_distribution_summary": distributions,
            "recording_level_role_contrast_summary": contrast_distributions,
            "cohort_plot_policies": {
                "recording_metric_panels": "one_value_per_recording_v1",
                "radial_summary": {
                    "policy_id": "complete_recording_support_only_v1",
                    "required_recording_count": len(recording_rows),
                    "reason": (
                        "Do not extend a cohort curve into radial tail bins that are "
                        "supported by only a subset of selected recordings."
                    ),
                },
            },
            "temporal_caveat": (
                "Controller-input-provenance proxy only; state presentation time and "
                "camera exposure alignment are unavailable. Cohort aggregation uses "
                "recordings as units and does not make frame-level inferential claims."
            ),
            "decision_status": "evidence_only_no_provider_or_analysis_promotion",
            "_source_task": task,
            "_per_recording_reports": reports,
            "_metric_rows": metric_rows,
            "_contrast_rows": contrast_rows,
            "_radial_rows": radial_rows,
            "_radial_summary": radial_summary,
        }
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _fail(f"Refusing to write empty table {path.name!r}.")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        _fail(f"Table {path.name!r} has inconsistent columns.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _short_label(recording_id: str) -> str:
    prefix, arena = recording_id.split("_arena_", 1)
    return f"{prefix[5:16]} A{arena.split('_', 1)[0]}"


def _plot_recording_metric(
    report: Mapping[str, Any], path: Path, *, metric: str, ylabel: str, title: str
) -> None:
    rows = report["_metric_rows"]
    recordings = [row["recording_id"] for row in report["recording_summary"]]
    roles = ("aggressive", "inert")
    epochs = ("pre", "training", "post")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    for ax, epoch in zip(axes, epochs, strict=True):
        for role in roles:
            index = {
                row["recording_id"]: row.get(metric)
                for row in rows
                if row["analysis_role"] == epoch and row["behavior_role"] == role
            }
            values = [
                np.nan if index.get(item) is None else index[item]
                for item in recordings
            ]
            ax.plot(range(len(recordings)), values, marker="o", label=role)
        ax.set_title(epoch)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xticks(
        range(len(recordings)),
        [_short_label(item) for item in recordings],
        rotation=30,
        ha="right",
    )
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_radial_summary(report: Mapping[str, Any], path: Path) -> None:
    rows = report["_radial_summary"]
    policy = report["cohort_plot_policies"]["radial_summary"]
    required_recording_count = int(policy["required_recording_count"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    for ax, epoch in zip(axes, ("pre", "training", "post"), strict=True):
        for role in ("aggressive", "inert"):
            present = [
                row
                for row in rows
                if row["analysis_role"] == epoch
                and row["behavior_role"] == role
                and row["selection_index_geometric_p50"] is not None
                and row["selection_index_geometric_recording_count"]
                == required_recording_count
            ]
            x = np.asarray(
                [0.5 * (row["bin_start_mm"] + row["bin_end_mm"]) for row in present]
            )
            p25 = np.asarray([row["selection_index_geometric_p25"] for row in present])
            p50 = np.asarray([row["selection_index_geometric_p50"] for row in present])
            p75 = np.asarray([row["selection_index_geometric_p75"] for row in present])
            ax.plot(x, p50, label=role)
            ax.fill_between(x, p25, p75, alpha=0.18)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(epoch)
        ax.set_ylabel("median observed / geometric expected")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set_xlabel("fish–chaser distance (mm)")
    fig.suptitle(
        "Recording-balanced moving-chaser radial occupancy\n"
        f"median and IQR; complete {required_recording_count}-recording support only"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def publish_cohort_canary(
    report: Mapping[str, Any], *, output_dir: str | Path
) -> dict[str, Any]:
    """Atomically publish compact cohort evidence outside scientific authorities."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing cohort output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        completed = _plain(report)
        source_task = completed.pop("_source_task")
        recording_reports = completed.pop("_per_recording_reports")
        metric_rows = completed.pop("_metric_rows")
        contrast_rows = completed.pop("_contrast_rows")
        radial_rows = completed.pop("_radial_rows")
        radial_summary = completed.pop("_radial_summary")
        recording_refs = []
        for recording_report in recording_reports:
            recording_id = recording_report["recording_id"]
            result = publish_operational_canary(
                recording_report,
                output_dir=temporary / "recordings" / recording_id,
            )
            recording_refs.append(
                {
                    "recording_id": recording_id,
                    "relative_path": f"recordings/{recording_id}",
                    "artifact_manifest_sha256": result["artifact_manifest"][
                        "manifest_sha256"
                    ],
                }
            )
        completed.update(
            {
                "status": "complete_selector_ineligible_operational_canary",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "software": get_git_info(),
                "per_recording_artifacts": recording_refs,
                "radial_cohort_summary_row_count": len(radial_summary),
            }
        )
        (temporary / "cohort_report.json").write_text(
            json.dumps(
                json_attr_safe(completed), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "source_task.json").write_text(
            json.dumps(source_task, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(temporary / "recording_summary.csv", completed["recording_summary"])
        _write_csv(temporary / "per_epoch_chaser_metrics.csv", metric_rows)
        _write_csv(temporary / "role_contrasts.csv", contrast_rows)
        _write_csv(temporary / "radial_recording_rows.csv", radial_rows)
        _write_csv(temporary / "radial_cohort_summary.csv", radial_summary)
        _plot_recording_metric(
            report,
            temporary / "distance_summary.png",
            metric="distance_p50_mm",
            ylabel="median fish–chaser distance (mm)",
            title="Fish–chaser distance by recording",
        )
        _plot_recording_metric(
            report,
            temporary / "near_field_summary.png",
            metric="near_zone_fraction_valid",
            ylabel="fraction of valid tracked frames",
            title="Near-chaser occupancy by recording",
        )
        _plot_recording_metric(
            report,
            temporary / "quadrant_summary.png",
            metric="same_quadrant_fraction_valid",
            ylabel="fraction of valid tracked frames",
            title="Same-quadrant occupancy by recording",
        )
        _plot_radial_summary(report, temporary / "radial_summary.png")
        for name in COHORT_OUTPUT_FILES:
            artifact = temporary / name
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                _fail(f"Expected cohort artifact {name!r} is absent or empty.")
        artifacts = []
        for artifact in sorted(item for item in temporary.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": artifact.relative_to(temporary).as_posix(),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256_file(artifact),
                }
            )
        manifest = {
            "schema_id": f"{SCHEMA_ID}.artifact_manifest",
            "schema_version": 1,
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return {
            "schema_id": f"{SCHEMA_ID}.publication_result",
            "schema_version": 1,
            "status": "published_selector_ineligible_operational_canary",
            "output_dir": str(target),
            "artifact_count": manifest["artifact_count"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_epoch_role(value: str) -> tuple[str, int]:
    role, separator, window = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("epoch roles must use ROLE=WINDOW_ID")
    try:
        window_id = int(window)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("epoch roles must use ROLE=WINDOW_ID") from exc
    if window_id < 0:
        raise argparse.ArgumentTypeError("epoch window IDs must be nonnegative")
    _exact_name(role, field="analysis role")
    return role, window_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="emit one frozen no-write task")
    plan.add_argument("cohort_inputs", type=Path)
    plan.add_argument("--provider-key", default="detection_bbox_centroid")
    plan.add_argument(
        "--epoch-role",
        action="append",
        type=_parse_epoch_role,
        default=None,
        help="ROLE=WINDOW_ID; defaults to pre=0, training=1, post=2",
    )
    plan.add_argument("--treatment-role", default="aggressive")
    plan.add_argument("--baseline-role", default="inert")
    plan.add_argument("--radial-bin-width-mm", type=float, default=2.0)
    plan.add_argument("--near-zone-radius-mm", type=float, default=5.0)
    plan.add_argument("--near-entry-radius-mm", type=float, default=5.0)
    plan.add_argument("--near-exit-radius-mm", type=float, default=6.0)
    plan.add_argument("--perimeter-band-mm", type=float, default=5.0)
    plan.add_argument("--min-expected-count", type=float, default=5.0)

    run = subparsers.add_parser("run", help="compute or publish one frozen task")
    run.add_argument("task_json", type=Path)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        bindings = args.epoch_role or [("pre", 0), ("training", 1), ("post", 2)]
        payload = resolve_cohort_task(
            args.cohort_inputs,
            provider_key=args.provider_key,
            epoch_role_bindings=bindings,
            treatment_role=args.treatment_role,
            baseline_role=args.baseline_role,
            radial_bin_width_mm=args.radial_bin_width_mm,
            near_zone_radius_mm=args.near_zone_radius_mm,
            near_entry_radius_mm=args.near_entry_radius_mm,
            near_exit_radius_mm=args.near_exit_radius_mm,
            perimeter_band_mm=args.perimeter_band_mm,
            min_expected_count=args.min_expected_count,
        )
    else:
        report = build_cohort_canary(args.task_json, workers=args.workers)
        if args.apply:
            if args.output_dir is None:
                raise SystemExit("run --apply requires --output-dir")
            payload = publish_cohort_canary(report, output_dir=args.output_dir)
        else:
            payload = {
                "schema_id": f"{SCHEMA_ID}.plan",
                "schema_version": 1,
                "status": "planned_no_writes",
                "source_task_sha256": report["source_task_sha256"],
                "selection": report["selection"],
                "camera_authority_by_arena": report["camera_authority_by_arena"],
                "recording_summary": report["recording_summary"],
                "recording_level_distribution_summary": report[
                    "recording_level_distribution_summary"
                ],
                "planned_output_dir": None
                if args.output_dir is None
                else str(args.output_dir.expanduser().resolve()),
                "planned_top_level_artifacts": [
                    *COHORT_OUTPUT_FILES,
                    "artifact_manifest.json",
                ],
                "per_recording_artifact_sets": len(report["recording_summary"]),
                "per_recording_artifacts": [
                    *RECORDING_OUTPUT_FILES,
                    "artifact_manifest.json",
                ],
                "writes": False,
            }
    print(
        json.dumps(json_attr_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AGGREGATION_POLICY_ID",
    "COHORT_OUTPUT_FILES",
    "ProviderChaserPositionSuiteCohortError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TASK_SCHEMA_ID",
    "TASK_SCHEMA_VERSION",
    "build_cohort_canary",
    "load_cohort_task",
    "publish_cohort_canary",
    "resolve_cohort_task",
]
