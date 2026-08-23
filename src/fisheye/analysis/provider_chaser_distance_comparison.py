"""Compare two exact provider-aware chaser-distance publications.

This module is a bounded canary, not a provider selector.  It loads two exact
selector-ineligible provider chaser-distance runs, binds their shared stimulus
epoch v2 authority and upstream chaser identity registries, and reports
distance summaries plus provider disagreement.  It never writes an analysis
Zarr, updates a registry, or promotes either provider.

Current recordings use a controller-input-provenance temporal proxy.  The
result therefore describes distance to the latest logged controller state for
the represented input acquisition frame; it does not claim physical display
presentation timing.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fisheye.analysis.provider_chaser_distance_candidates import (
    MANIFEST_ATTR as CANDIDATE_MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR as CANDIDATE_MANIFEST_DIGEST_ATTR,
)
from fisheye.analysis_workflows.provider_chaser_distance_publication import (
    ProviderChaserDistanceSourceHandle,
    load_provider_chaser_distance_source_handle,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root


SCHEMA_ID = "palette.provider_chaser_distance_comparison_canary"
SCHEMA_VERSION = 1
METHOD_ID = "exact_dual_provider_distance_epoch_comparison_v1"
TEMPORAL_ALIGNMENT_CLASS = "controller_input_provenance_proxy"
OUTPUT_FILES = (
    "canary_report.json",
    "per_epoch_metrics.csv",
    "provider_comparison.csv",
    "distance_cdf.png",
    "provider_disagreement.png",
)


class ProviderChaserDistanceComparisonError(ValueError):
    """Raised when an exact dual-provider comparison cannot be built."""


def _fail(message: str) -> None:
    raise ProviderChaserDistanceComparisonError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strict_name(value: object, *, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value in {".", "..", "latest", "latest_complete", "selected"}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        _fail(f"{label} must name one exact non-selector run.")
    return value


def _digest(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be one object.")
    return _plain(value)


def _declarations_by_path(
    value: object,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        _fail(f"{label} must be one declaration sequence.")
    result: dict[str, dict[str, Any]] = {}
    for raw in value:
        declaration = _mapping(raw, label=f"{label} entry")
        path = declaration.get("path")
        if type(path) is not str or not path or path in result:
            _fail(f"{label} contains an invalid or duplicate path.")
        result[path] = declaration
    return result


def _relative_semantic_binding(
    archive: Path,
    handle: ProviderChaserDistanceSourceHandle,
) -> dict[str, Any]:
    receipt = _mapping(handle.source_receipt, label="source receipt")
    relative = _mapping(receipt.get("relative_frame"), label="relative-frame receipt")
    run_path = relative.get("run_path")
    expected_digest = _digest(
        relative.get("manifest_sha256"),
        label="relative-frame manifest digest",
    )
    prefix = "analysis/chaser_relative_frame_runs/"
    if type(run_path) is not str or not run_path.startswith(prefix):
        _fail("Relative-frame receipt has no exact run path.")
    run_name = _strict_name(run_path[len(prefix) :], label="relative-frame run")
    if run_path != f"{prefix}{run_name}":
        _fail("Relative-frame receipt path is not canonical.")

    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[run_path]
    manifest = _mapping(
        run.attrs.get("chaser_relative_frame_manifest"),
        label="relative-frame manifest",
    )
    observed_digest = canonical_json_sha256(manifest)
    if observed_digest != expected_digest:
        _fail("Relative-frame semantic manifest differs from the sealed receipt.")
    if run.attrs.get("chaser_relative_frame_manifest_sha256") != expected_digest:
        _fail("Relative-frame manifest digest attribute is stale.")
    if manifest.get("recording_id") != handle.recording_id:
        _fail("Relative-frame semantic binding belongs to another recording.")

    registries = _mapping(
        manifest.get("identity_registries"),
        label="relative-frame identity registries",
    )
    expected_registry_names = {"fish", "chaser", "behavior_role", "active_state"}
    if set(registries) != expected_registry_names:
        _fail("Relative-frame identity registries are incomplete.")
    for name in expected_registry_names:
        registry = _mapping(registries[name], label=f"identity registry {name}")
        if not registry:
            _fail(f"Identity registry {name!r} is empty.")
        registries[name] = registry

    context = _mapping(manifest.get("context"), label="relative-frame context")
    occurrence = _mapping(
        context.get("chaser_occurrence"),
        label="chaser occurrence envelope",
    )
    occurrence_record = _mapping(
        occurrence.get("record"),
        label="chaser occurrence record",
    )
    occurrence_sha256 = _digest(
        occurrence.get("sha256"),
        label="chaser occurrence digest",
    )
    if canonical_json_sha256(occurrence_record) != occurrence_sha256:
        _fail("Chaser occurrence record digest is stale.")

    provider_declarations = _declarations_by_path(
        handle.manifest.get("array_declarations"),
        label="provider declarations",
    )
    relative_declarations = _declarations_by_path(
        manifest.get("array_declarations"),
        label="relative-frame declarations",
    )
    for name in (
        "acquisition_frame_id",
        "track_sample_id",
        "fish_identity_code",
        "chaser_identity_code",
        "chaser_behavior_role_code",
        "chaser_behavior_role_valid",
    ):
        provider_record = provider_declarations.get(name)
        relative_record = relative_declarations.get(f"base/{name}")
        if provider_record is None or relative_record is None:
            _fail(f"Required semantic declaration {name!r} is absent.")
        provider_digest = provider_record.get("content_sha256")
        relative_digest = relative_record.get("content_sha256")
        if provider_digest != relative_digest:
            _fail(f"Provider semantic array {name!r} differs from its bound source.")

    return {
        "run_name": run_name,
        "run_path": run_path,
        "manifest_sha256": observed_digest,
        "identity_registries": registries,
        "chaser_occurrence": occurrence_record,
        "chaser_occurrence_sha256": occurrence_sha256,
    }


def _candidate_epoch_binding(
    archive: Path,
    handle: ProviderChaserDistanceSourceHandle,
) -> dict[str, Any]:
    receipt = _mapping(handle.source_receipt, label="source receipt")
    native = _mapping(receipt.get("native_source"), label="native source receipt")
    run_path = native.get("run_path")
    expected_digest = _digest(
        native.get("manifest_sha256"),
        label="native candidate manifest digest",
    )
    prefix = "analysis/provider_chaser_distance_candidate_runs/"
    if type(run_path) is not str or not run_path.startswith(prefix):
        _fail("Native candidate receipt has no exact run path.")
    _strict_name(run_path[len(prefix) :], label="native candidate run")
    validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    run = root[run_path]
    manifest = _mapping(
        run.attrs.get(CANDIDATE_MANIFEST_ATTR),
        label="native candidate manifest",
    )
    payload = _mapping(manifest.get("payload"), label="native candidate payload")
    payload_digest = _digest(
        manifest.get("payload_digest"),
        label="native candidate payload digest",
    )
    if (
        canonical_json_sha256(payload) != payload_digest
        or payload_digest != expected_digest
    ):
        _fail("Native candidate manifest differs from the sealed source receipt.")
    if run.attrs.get(CANDIDATE_MANIFEST_DIGEST_ATTR) != payload_digest:
        _fail("Native candidate manifest digest attribute is stale.")
    source_authority = _mapping(
        payload.get("source_authority"),
        label="native candidate source authority",
    )
    epoch = _mapping(
        source_authority.get("stimulus_epoch"),
        label="stimulus epoch authority",
    )
    epoch_path = epoch.get("run_path")
    epoch_prefix = "analysis/stimulus_epoch_runs/"
    if type(epoch_path) is not str or not epoch_path.startswith(epoch_prefix):
        _fail("Native candidate has no exact stimulus-epoch run path.")
    epoch_name = _strict_name(
        epoch_path[len(epoch_prefix) :],
        label="stimulus epoch run",
    )
    return {
        "candidate_run_path": run_path,
        "candidate_manifest_sha256": payload_digest,
        "epoch_run_name": epoch_name,
        "epoch_run_path": epoch_path,
        # The producer calls this field ``manifest_sha256``, but stores the
        # stimulus-epoch manifest payload digest rather than the digest of the
        # complete manifest envelope. Keep those identities distinct.
        "epoch_manifest_payload_sha256": _digest(
            epoch.get("manifest_sha256"),
            label="stimulus epoch manifest payload digest",
        ),
    }


def _frame_chaser(handle: ProviderChaserDistanceSourceHandle, name: str) -> np.ndarray:
    values = np.asarray(handle.array(name))
    if values.ndim == 0 or values.shape[0] != handle.n_rows:
        _fail(f"Provider array {name!r} does not use the declared flat row axis.")
    return values.reshape((handle.n_frames, handle.n_chasers) + values.shape[1:])


def _require_repeated_frame_field(values: np.ndarray, *, name: str) -> np.ndarray:
    if values.ndim < 2:
        _fail(f"Frame field {name!r} is not frame-by-chaser.")
    first = values[:, :1, ...]
    if not np.array_equal(values, np.broadcast_to(first, values.shape), equal_nan=True):
        _fail(f"Frame field {name!r} differs across chaser rows.")
    return values[:, 0, ...]


def _provider_arrays(
    handle: ProviderChaserDistanceSourceHandle,
) -> dict[str, np.ndarray]:
    arrays = {
        name: _frame_chaser(handle, name)
        for name in (
            "acquisition_frame_id",
            "track_sample_id",
            "source_position_xy_px",
            "source_position_valid",
            "fish_identity_code",
            "selection_member",
            "chaser_identity_code",
            "chaser_behavior_role_code",
            "chaser_behavior_role_valid",
            "chaser_occurrence_member",
            "nearest_chaser_member",
            "distance_mm",
            "distance_mm_valid",
        )
    }
    for name in (
        "acquisition_frame_id",
        "track_sample_id",
        "source_position_xy_px",
        "source_position_valid",
        "fish_identity_code",
        "selection_member",
    ):
        arrays[name] = _require_repeated_frame_field(arrays[name], name=name)
    return arrays


def _percentile(values: np.ndarray, q: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, q))


def _mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _fraction(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=bool)
    return float(np.mean(array)) if array.size else None


def _interval_masks(
    frame_ids: np.ndarray,
    selection: ResolvedEpochSelection,
) -> list[tuple[Any, np.ndarray]]:
    result: list[tuple[Any, np.ndarray]] = []
    covered = np.zeros(frame_ids.shape, dtype=bool)
    for interval in selection.intervals:
        mask = (frame_ids >= interval.start_frame) & (frame_ids < interval.end_frame)
        if np.any(covered & mask):
            _fail("Resolved epoch intervals overlap on the provider frame axis.")
        covered |= mask
        result.append((interval, mask))
    return result


def build_comparison(
    archive: Path,
    *,
    first_run: str,
    second_run: str,
    first_label: str,
    second_label: str,
    cdf_thresholds_mm: Sequence[float],
) -> dict[str, Any]:
    """Build one read-only comparison over exact, explicitly named inputs."""

    archive = Path(archive).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    first_run = _strict_name(first_run, label="first provider run")
    second_run = _strict_name(second_run, label="second provider run")
    first_label = _strict_name(first_label, label="first provider label")
    second_label = _strict_name(second_label, label="second provider label")
    if first_run == second_run or first_label == second_label:
        _fail("Provider runs and labels must be distinct.")
    thresholds = np.asarray(cdf_thresholds_mm, dtype=np.float64)
    if (
        thresholds.ndim != 1
        or thresholds.size == 0
        or not np.isfinite(thresholds).all()
        or np.any(thresholds < 0)
        or np.any(np.diff(thresholds) <= 0)
    ):
        _fail("CDF thresholds must be finite, non-negative, and strictly increasing.")

    handles = {
        first_label: load_provider_chaser_distance_source_handle(
            archive,
            run_name=first_run,
            use_consolidated=True,
        ),
        second_label: load_provider_chaser_distance_source_handle(
            archive,
            run_name=second_run,
            use_consolidated=True,
        ),
    }
    first = handles[first_label]
    second = handles[second_label]
    if first.recording_id != second.recording_id:
        _fail("Provider runs belong to different recordings.")
    if first.n_frames != second.n_frames or first.n_chasers != second.n_chasers:
        _fail("Provider runs have different frame/chaser dimensions.")

    semantics = {
        label: _relative_semantic_binding(archive, handle)
        for label, handle in handles.items()
    }
    if (
        semantics[first_label]["identity_registries"]
        != semantics[second_label]["identity_registries"]
    ):
        _fail("Provider runs bind different fish/chaser/role identity registries.")
    if (
        semantics[first_label]["chaser_occurrence"]
        != semantics[second_label]["chaser_occurrence"]
    ):
        _fail("Provider runs bind different chaser occurrence authority.")

    epoch_bindings = {
        label: _candidate_epoch_binding(archive, handle)
        for label, handle in handles.items()
    }
    first_epoch = epoch_bindings[first_label]
    second_epoch = epoch_bindings[second_label]
    if (
        first_epoch["epoch_run_path"] != second_epoch["epoch_run_path"]
        or first_epoch["epoch_manifest_payload_sha256"]
        != second_epoch["epoch_manifest_payload_sha256"]
    ):
        _fail("Provider runs bind different stimulus-epoch authority.")
    validate_direct_consolidated_subtree(
        archive,
        subtree_path=first_epoch["epoch_run_path"],
    )
    selection = resolve_exact_stimulus_epoch_selection(
        archive,
        run_name=first_epoch["epoch_run_name"],
    )
    if (
        selection.run_manifest_payload_digest
        != first_epoch["epoch_manifest_payload_sha256"]
    ):
        _fail("Exact stimulus-epoch manifest payload differs from candidate authority.")
    arrays = {label: _provider_arrays(handle) for label, handle in handles.items()}

    for name in (
        "acquisition_frame_id",
        "track_sample_id",
        "fish_identity_code",
        "selection_member",
        "chaser_identity_code",
        "chaser_behavior_role_code",
        "chaser_behavior_role_valid",
        "chaser_occurrence_member",
    ):
        if not np.array_equal(arrays[first_label][name], arrays[second_label][name]):
            _fail(f"Provider row alignment differs for {name!r}.")

    frame_ids = np.asarray(arrays[first_label]["acquisition_frame_id"], dtype=np.int64)
    if frame_ids.ndim != 1 or np.any(np.diff(frame_ids) <= 0):
        _fail("Provider acquisition-frame axis is not strictly increasing.")
    interval_masks = _interval_masks(frame_ids, selection)

    registries = semantics[first_label]["identity_registries"]
    role_registry = registries["behavior_role"]
    chaser_registry = registries["chaser"]
    first_roles = np.asarray(arrays[first_label]["chaser_behavior_role_code"])
    first_chasers = np.asarray(arrays[first_label]["chaser_identity_code"])
    metrics: list[dict[str, Any]] = []
    cdf_rows: list[dict[str, Any]] = []
    for provider_label, provider_arrays in arrays.items():
        for interval, epoch_mask in interval_masks:
            for chaser_column in range(first.n_chasers):
                role_codes = first_roles[:, chaser_column]
                chaser_codes = first_chasers[:, chaser_column]
                unique_roles = np.unique(role_codes)
                unique_chasers = np.unique(chaser_codes)
                if unique_roles.size != 1 or unique_chasers.size != 1:
                    _fail(
                        "Canary requires stable chaser identity and role per axis column."
                    )
                role_code = int(unique_roles[0])
                chaser_code = int(unique_chasers[0])
                role_label = role_registry.get(str(role_code))
                chaser_identity = chaser_registry.get(str(chaser_code))
                if type(role_label) is not str or type(chaser_identity) is not str:
                    _fail("Chaser role or identity code is absent from its registry.")
                candidate = (
                    epoch_mask
                    & provider_arrays["selection_member"]
                    & provider_arrays["chaser_occurrence_member"][:, chaser_column]
                    & provider_arrays["chaser_behavior_role_valid"][:, chaser_column]
                )
                valid = (
                    candidate & provider_arrays["distance_mm_valid"][:, chaser_column]
                )
                distances = provider_arrays["distance_mm"][:, chaser_column][valid]
                denominator = int(np.count_nonzero(candidate))
                valid_count = int(np.count_nonzero(valid))
                common = {
                    "provider_label": provider_label,
                    "provider_run": handles[provider_label].run_name,
                    "provider_manifest_sha256": handles[provider_label].manifest_sha256,
                    "epoch_window_id": int(interval.window_id),
                    "epoch_label": str(interval.label),
                    "epoch_start_frame": int(interval.start_frame),
                    "epoch_end_frame_exclusive": int(interval.end_frame),
                    "source_interval_sha256": str(interval.source_interval_digest),
                    "chaser_column": chaser_column,
                    "chaser_identity_code": chaser_code,
                    "chaser_identity": chaser_identity,
                    "behavior_role_code": role_code,
                    "behavior_role": role_label,
                    "candidate_frame_count": denominator,
                    "valid_distance_frame_count": valid_count,
                    "valid_distance_fraction": (
                        float(valid_count / denominator) if denominator else None
                    ),
                }
                metrics.append(
                    {
                        **common,
                        "distance_mean_mm": _mean(distances),
                        "distance_p05_mm": _percentile(distances, 5),
                        "distance_p25_mm": _percentile(distances, 25),
                        "distance_p50_mm": _percentile(distances, 50),
                        "distance_p75_mm": _percentile(distances, 75),
                        "distance_p95_mm": _percentile(distances, 95),
                    }
                )
                for threshold in thresholds:
                    cdf_rows.append(
                        {
                            **common,
                            "threshold_mm": float(threshold),
                            "fraction_at_or_below": (
                                float(np.mean(distances <= threshold))
                                if distances.size
                                else None
                            ),
                        }
                    )

    first_arrays = arrays[first_label]
    second_arrays = arrays[second_label]
    position_common = (
        first_arrays["source_position_valid"]
        & second_arrays["source_position_valid"]
        & np.isfinite(first_arrays["source_position_xy_px"]).all(axis=1)
        & np.isfinite(second_arrays["source_position_xy_px"]).all(axis=1)
    )
    position_delta_px = np.linalg.norm(
        first_arrays["source_position_xy_px"][position_common].astype(np.float64)
        - second_arrays["source_position_xy_px"][position_common].astype(np.float64),
        axis=1,
    )
    comparison_rows: list[dict[str, Any]] = []
    nearest_first = first_arrays["nearest_chaser_member"]
    nearest_second = second_arrays["nearest_chaser_member"]
    all_distance_common = np.all(
        first_arrays["distance_mm_valid"] & second_arrays["distance_mm_valid"],
        axis=1,
    )
    nearest_agreement = np.all(nearest_first == nearest_second, axis=1)
    for interval, epoch_mask in interval_masks:
        epoch_position = epoch_mask & position_common
        epoch_all_distance = epoch_mask & all_distance_common
        comparison_rows.append(
            {
                "comparison_kind": "epoch_all_chasers",
                "epoch_window_id": int(interval.window_id),
                "epoch_label": str(interval.label),
                "chaser_column": None,
                "behavior_role": None,
                "common_position_frame_count": int(np.count_nonzero(epoch_position)),
                "position_delta_p50_px": _percentile(
                    np.linalg.norm(
                        first_arrays["source_position_xy_px"][epoch_position].astype(
                            np.float64
                        )
                        - second_arrays["source_position_xy_px"][epoch_position].astype(
                            np.float64
                        ),
                        axis=1,
                    ),
                    50,
                ),
                "position_delta_p95_px": _percentile(
                    np.linalg.norm(
                        first_arrays["source_position_xy_px"][epoch_position].astype(
                            np.float64
                        )
                        - second_arrays["source_position_xy_px"][epoch_position].astype(
                            np.float64
                        ),
                        axis=1,
                    ),
                    95,
                ),
                "common_all_chaser_distance_frame_count": int(
                    np.count_nonzero(epoch_all_distance)
                ),
                "nearest_chaser_agreement_fraction": _fraction(
                    nearest_agreement[epoch_all_distance]
                ),
                "common_distance_frame_count": None,
                "signed_distance_delta_mean_mm": None,
                "absolute_distance_delta_p50_mm": None,
                "absolute_distance_delta_p95_mm": None,
            }
        )
        for chaser_column in range(first.n_chasers):
            valid = (
                epoch_mask
                & first_arrays["distance_mm_valid"][:, chaser_column]
                & second_arrays["distance_mm_valid"][:, chaser_column]
            )
            delta = second_arrays["distance_mm"][:, chaser_column][valid].astype(
                np.float64
            ) - first_arrays["distance_mm"][:, chaser_column][valid].astype(np.float64)
            role_code = int(first_roles[0, chaser_column])
            comparison_rows.append(
                {
                    "comparison_kind": "epoch_chaser",
                    "epoch_window_id": int(interval.window_id),
                    "epoch_label": str(interval.label),
                    "chaser_column": chaser_column,
                    "behavior_role": role_registry[str(role_code)],
                    "common_position_frame_count": None,
                    "position_delta_p50_px": None,
                    "position_delta_p95_px": None,
                    "common_all_chaser_distance_frame_count": None,
                    "nearest_chaser_agreement_fraction": None,
                    "common_distance_frame_count": int(np.count_nonzero(valid)),
                    "signed_distance_delta_mean_mm": _mean(delta),
                    "absolute_distance_delta_p50_mm": _percentile(np.abs(delta), 50),
                    "absolute_distance_delta_p95_mm": _percentile(np.abs(delta), 95),
                }
            )

    temporal = {
        label: _mapping(
            handle.manifest.get("temporal_alignment"),
            label=f"{label} temporal alignment",
        )
        for label, handle in handles.items()
    }
    for label, record in temporal.items():
        if record.get("temporal_alignment_class") != TEMPORAL_ALIGNMENT_CLASS:
            _fail(f"Provider {label!r} does not declare the expected proxy caveat.")
        if record.get("physical_presentation_verified") is not False:
            _fail(f"Provider {label!r} has an invalid presentation-verification claim.")

    return json_attr_safe(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "recording_id": first.recording_id,
            "analysis_zarr": str(archive),
            "status": "computed_read_only",
            "selection": "none",
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "provider_default_promoted": False,
            "providers": {
                label: {
                    "run_name": handle.run_name,
                    "run_path": handle.run_path,
                    "manifest_sha256": handle.manifest_sha256,
                    "source_position_provider": _mapping(
                        handle.manifest.get("source_provider_authorities"),
                        label="source provider authorities",
                    )["source_position"],
                    "relative_semantic_binding": semantics[label],
                    "candidate_epoch_binding": epoch_bindings[label],
                    "temporal_alignment": temporal[label],
                    "valid_source_position_frame_count": int(
                        np.count_nonzero(arrays[label]["source_position_valid"])
                    ),
                }
                for label, handle in handles.items()
            },
            "epoch_selection": selection.selection_record,
            "cdf_thresholds_mm": thresholds.tolist(),
            "denominator_policy": {
                "epoch_membership": "acquisition_frame_id_in_exact_half_open_epoch_interval",
                "distance_fraction": "valid_distance_frames_over_candidate_epoch_chaser_frames",
                "provider_comparison": "exact_common_valid_acquisition_frames_only",
                "native_stimulus_samples_not_counted_as_camera_observations": True,
            },
            "temporal_caveat": (
                "Controller-input-provenance proxy only; state presentation time and "
                "camera exposure alignment are unavailable."
            ),
            "per_epoch_metrics": metrics,
            "distance_cdf": cdf_rows,
            "provider_comparison": comparison_rows,
            "overall_provider_comparison": {
                "common_position_frame_count": int(np.count_nonzero(position_common)),
                "position_delta_p50_px": _percentile(position_delta_px, 50),
                "position_delta_p95_px": _percentile(position_delta_px, 95),
                "position_delta_p99_px": _percentile(position_delta_px, 99),
                "common_all_chaser_distance_frame_count": int(
                    np.count_nonzero(all_distance_common)
                ),
                "nearest_chaser_agreement_fraction": _fraction(
                    nearest_agreement[all_distance_common]
                ),
            },
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


def _plot_cdf(report: Mapping[str, Any], path: Path) -> None:
    rows = report["distance_cdf"]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["provider_label"]),
            str(row["epoch_label"]),
            str(row["behavior_role"]),
        )
        groups.setdefault(key, []).append(row)
    for (provider, epoch, role), values in sorted(groups.items()):
        present = [
            value for value in values if value["fraction_at_or_below"] is not None
        ]
        if not present:
            continue
        x = [float(value["threshold_mm"]) for value in present]
        y = [float(value["fraction_at_or_below"]) for value in present]
        ax.plot(x, y, linewidth=1.3, label=f"{provider} · {epoch} · {role}")
    ax.set(
        xlabel="fish–chaser distance threshold (mm)",
        ylabel="fraction at or below",
        ylim=(0, 1),
    )
    ax.set_title("Provider-aware chaser-distance CDF canary")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_disagreement(report: Mapping[str, Any], path: Path) -> None:
    rows = [
        row
        for row in report["provider_comparison"]
        if row["comparison_kind"] == "epoch_chaser"
    ]
    present_rows = [
        row
        for row in rows
        if row["absolute_distance_delta_p50_mm"] is not None
        and row["absolute_distance_delta_p95_mm"] is not None
    ]
    labels = [f"{row['epoch_label']}\n{row['behavior_role']}" for row in present_rows]
    p50 = [float(row["absolute_distance_delta_p50_mm"]) for row in present_rows]
    p95 = [float(row["absolute_distance_delta_p95_mm"]) for row in present_rows]
    x = np.arange(len(present_rows))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(x - 0.18, p50, width=0.36, label="p50 |Δ distance|")
    axes[0].bar(x + 0.18, p95, width=0.36, label="p95 |Δ distance|")
    axes[0].set_ylabel("mm")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].legend()
    axes[0].set_title("Keypoint minus detection provider disagreement")
    epoch_rows = [
        row
        for row in report["provider_comparison"]
        if row["comparison_kind"] == "epoch_all_chasers"
    ]
    present_epoch_rows = [
        row
        for row in epoch_rows
        if row["nearest_chaser_agreement_fraction"] is not None
    ]
    ex = np.arange(len(present_epoch_rows))
    axes[1].bar(
        ex,
        [
            100.0 * float(row["nearest_chaser_agreement_fraction"])
            for row in present_epoch_rows
        ],
    )
    axes[1].set_xticks(
        ex,
        [str(row["epoch_label"]) for row in present_epoch_rows],
    )
    axes[1].set_ylabel("nearest-chaser agreement (%)")
    axes[1].set_ylim(99.0, 100.0)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_operational_canary(
    report: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Atomically publish compact operational evidence outside analysis Zarr."""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to replace existing canary output: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        completed = {
            **_plain(report),
            "status": "complete_selector_ineligible_operational_canary",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software": get_git_info(),
        }
        (temporary / "canary_report.json").write_text(
            json.dumps(completed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(temporary / "per_epoch_metrics.csv", completed["per_epoch_metrics"])
        _write_csv(
            temporary / "provider_comparison.csv",
            completed["provider_comparison"],
        )
        _plot_cdf(completed, temporary / "distance_cdf.png")
        _plot_disagreement(completed, temporary / "provider_disagreement.png")
        artifacts = []
        for name in OUTPUT_FILES:
            path = temporary / name
            if not path.is_file() or path.stat().st_size <= 0:
                _fail(f"Expected canary artifact {name!r} is absent or empty.")
            artifacts.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        manifest = {
            "schema_id": f"{SCHEMA_ID}.artifact_manifest",
            "schema_version": 1,
            "recording_id": completed["recording_id"],
            "selector_eligible": False,
            "production_authority": False,
            "registry_update": False,
            "artifacts": artifacts,
        }
        manifest["manifest_sha256"] = canonical_json_sha256(manifest)
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return {
            "schema_id": f"{SCHEMA_ID}.publication_result",
            "schema_version": 1,
            "status": "published_selector_ineligible_operational_canary",
            "output_dir": str(output_dir),
            "artifact_manifest": manifest,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated numbers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--first-run", required=True)
    parser.add_argument("--second-run", required=True)
    parser.add_argument("--first-label", default="detection")
    parser.add_argument("--second-label", default="keypoint")
    parser.add_argument(
        "--cdf-thresholds-mm",
        type=_parse_thresholds,
        default=tuple(float(value) for value in range(0, 101, 2)),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_comparison(
        args.zarr_path,
        first_run=args.first_run,
        second_run=args.second_run,
        first_label=args.first_label,
        second_label=args.second_label,
        cdf_thresholds_mm=args.cdf_thresholds_mm,
    )
    if args.apply:
        if args.output_dir is None:
            raise SystemExit("--apply requires --output-dir")
        payload = publish_operational_canary(report, output_dir=args.output_dir)
    else:
        payload = {
            "schema_id": f"{SCHEMA_ID}.plan",
            "schema_version": 1,
            "status": "planned_no_writes",
            "output_dir": None if args.output_dir is None else str(args.output_dir),
            "planned_artifacts": [*OUTPUT_FILES, "artifact_manifest.json"],
            "summary": {
                "recording_id": report["recording_id"],
                "providers": {
                    label: {
                        "run_name": value["run_name"],
                        "manifest_sha256": value["manifest_sha256"],
                        "valid_source_position_frame_count": value[
                            "valid_source_position_frame_count"
                        ],
                    }
                    for label, value in report["providers"].items()
                },
                "epoch_labels": [
                    interval["label"]
                    for interval in report["epoch_selection"]["intervals"]
                ],
                "overall_provider_comparison": report["overall_provider_comparison"],
                "temporal_caveat": report["temporal_caveat"],
            },
        }
    print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "METHOD_ID",
    "ProviderChaserDistanceComparisonError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "build_comparison",
    "publish_operational_canary",
]
