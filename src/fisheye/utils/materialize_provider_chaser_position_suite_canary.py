"""Materialize a selector-ineligible provider-aware chaser position canary.

The command accepts only explicit immutable run names and caller-declared
epoch-role bindings.  Without ``--apply`` it performs all read-only source
validation and computation, then prints the exact output plan.  With
``--apply`` it atomically writes compact evidence below an operator-supplied
operations directory; it never writes the analysis Zarr or registry.
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

from fisheye.analysis.provider_chaser_distance_comparison import (  # noqa: E402
    TEMPORAL_ALIGNMENT_CLASS,
    _candidate_epoch_binding,
    _mapping,
    _relative_semantic_binding,
    _require_repeated_frame_field,
)
from fisheye.analysis.provider_chaser_position_suite import (  # noqa: E402
    CircularArena,
    PositionSuiteConfig,
    PositionSuiteEpoch,
    compute_provider_chaser_position_suite,
)
from fisheye.analysis_workflows.provider_chaser_distance_publication import (  # noqa: E402
    ProviderChaserDistanceSourceHandle,
    load_provider_chaser_distance_source_handle,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (  # noqa: E402
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.shared.json_safety import json_attr_safe  # noqa: E402
from fisheye.shared.system_metadata import get_git_info  # noqa: E402
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256  # noqa: E402
from fisheye.shared.zarr.metadata_equivalence import (  # noqa: E402
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root  # noqa: E402
from fisheye.utils.materialize_provider_spatial_canary import (  # noqa: E402
    load_grid_and_transform_authority,
)


SCHEMA_ID = "palette.provider_chaser_position_suite_canary"
SCHEMA_VERSION = 1
DISPOSITION = "selector_ineligible_operational_canary"
OUTPUT_FILES = (
    "canary_report.json",
    "per_epoch_chaser_metrics.csv",
    "distance_cdf.csv",
    "radial_occupancy.csv",
    "quadrant_joint_occupancy.csv",
    "role_contrasts.csv",
    "role_radial_contrasts.csv",
    "distance_cdf.png",
    "radial_selection_index.png",
    "near_field_summary.png",
    "quadrant_summary.png",
)


class ProviderChaserPositionSuiteCanaryError(ValueError):
    """Raised when a canary source or requested publication is unsafe."""


def _fail(message: str) -> None:
    raise ProviderChaserPositionSuiteCanaryError(message)


def _strict_name(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", "..", "latest", "latest_complete", "selected", "current"}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        _fail(f"{field} must name one exact immutable non-selector identity.")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _frame_chaser(handle: ProviderChaserDistanceSourceHandle, name: str) -> np.ndarray:
    values = np.asarray(handle.array(name))
    if values.ndim == 0 or values.shape[0] != handle.n_rows:
        _fail(f"Provider array {name!r} does not use the declared flat row axis.")
    return values.reshape((handle.n_frames, handle.n_chasers) + values.shape[1:])


def _provider_arrays(
    handle: ProviderChaserDistanceSourceHandle,
) -> dict[str, np.ndarray]:
    values = {
        name: _frame_chaser(handle, name)
        for name in (
            "acquisition_frame_id",
            "source_position_xy_px",
            "source_position_valid",
            "selection_member",
            "chaser_position_xy_px",
            "chaser_position_valid",
            "chaser_identity_code",
            "chaser_behavior_role_code",
            "chaser_behavior_role_valid",
            "chaser_occurrence_member",
            "distance_px",
            "distance_px_valid",
            "distance_mm",
            "distance_mm_valid",
        )
    }
    for name in (
        "acquisition_frame_id",
        "source_position_xy_px",
        "source_position_valid",
        "selection_member",
    ):
        values[name] = _require_repeated_frame_field(values[name], name=name)
    return values


def _exact_attribute_record(
    archive: Path,
    *,
    record_ref: object,
    expected_sha256: str | None,
    field: str,
) -> dict[str, Any]:
    """Resolve one absolute group-attribute record in both metadata modes."""

    if type(record_ref) is not str or record_ref.count("@") != 1:
        _fail(f"{field} must be one exact group-attribute reference.")
    group_path, attr_name = record_ref.split("@", 1)
    if (
        not group_path.startswith("/")
        or group_path in {"", "/"}
        or not attr_name
        or "/" in attr_name
        or any(part in {"", ".", ".."} for part in group_path[1:].split("/"))
    ):
        _fail(f"{field} is not a canonical absolute group-attribute reference.")
    normalized_path = group_path[1:]
    validate_direct_consolidated_subtree(archive, subtree_path=normalized_path)
    observed_records = []
    persisted_digests = []
    for consolidated in (False, True):
        root = open_zarr_root(archive, mode="r", use_consolidated=consolidated)
        node = root[normalized_path]
        record = node.attrs.get(attr_name)
        if not isinstance(record, Mapping):
            _fail(f"{field} record is absent or malformed.")
        normalized = json.loads(
            json.dumps(
                json_attr_safe(dict(record)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        observed_records.append(normalized)
        persisted_digests.append(node.attrs.get(f"{attr_name}_sha256"))
    if observed_records[0] != observed_records[1]:
        _fail(f"{field} differs between direct and consolidated metadata.")
    observed_sha256 = canonical_json_sha256(observed_records[0])
    if any(value != observed_sha256 for value in persisted_digests):
        _fail(f"{field} persisted digest is stale.")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        _fail(f"{field} content differs from the reviewed physical authority.")
    return {
        "record_ref": record_ref,
        "record_sha256": observed_sha256,
        "record": observed_records[0],
    }


def _physical_frame_semantic_core(
    record: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    """Remove only path-scoped identity fields from one physical-frame record."""

    expected = {
        "schema_id",
        "schema_version",
        "kind",
        "frame_id",
        "coordinate_units",
        "origin",
        "source_origin_relation",
        "positive_directions",
        "compatible_profile_ids",
        "source_space_id",
        "source_camera_pixels",
        "selected_camera_evidence",
        "camera_id",
        "scale",
        "physical_extent",
    }
    if set(record) != expected:
        _fail(f"{field} has an unsupported physical-frame schema shape.")
    selected = record.get("selected_camera_evidence")
    if not isinstance(selected, Mapping) or set(selected) != {
        "record_ref",
        "record_sha256",
    }:
        _fail(f"{field} selected-camera evidence pointer is not closed.")
    core = {
        key: value
        for key, value in record.items()
        if key not in {"frame_id", "selected_camera_evidence"}
    }
    core["selected_camera_evidence_sha256"] = _digest(
        selected.get("record_sha256"),
        field=f"{field}.selected_camera_evidence.record_sha256",
    )
    return json_attr_safe(core)


def _validate_physical_frame_semantic_equivalence(
    provider: Mapping[str, Any],
    recording: Mapping[str, Any],
) -> dict[str, Any]:
    provider_core = _physical_frame_semantic_core(
        provider, field="provider physical frame"
    )
    recording_core = _physical_frame_semantic_core(
        recording, field="recording physical frame"
    )
    if provider_core != recording_core:
        _fail(
            "Provider and recording physical frames differ beyond their path-scoped "
            "frame identity and selected-evidence record_ref."
        )
    return {
        "policy_id": "physical_frame_path_scoped_identity_equivalence_v1",
        "semantic_core_sha256": canonical_json_sha256(provider_core),
        "allowed_differences": [
            "frame_id",
            "selected_camera_evidence.record_ref",
        ],
        "required_equalities": [
            "schema_and_coordinate_semantics",
            "source_camera_pixels_exact_pointer",
            "selected_camera_evidence_record_sha256",
            "camera_id",
            "scale",
            "physical_extent",
        ],
    }


def _parse_epoch_role(value: str) -> tuple[str, int]:
    role, separator, window = value.partition("=")
    try:
        window_id = int(window)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("epoch roles must use ROLE=WINDOW_ID") from exc
    if not separator or not role or role != role.strip() or window_id < 0:
        raise argparse.ArgumentTypeError("epoch roles must use ROLE=WINDOW_ID")
    _strict_name(role, field="epoch analysis role")
    return role, window_id


def _parse_float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one number")
    return result


def _epoch_specs(
    selection: Any, bindings: Sequence[tuple[str, int]]
) -> list[PositionSuiteEpoch]:
    if not bindings:
        _fail("At least one explicit --epoch-role binding is required.")
    roles = [role for role, _ in bindings]
    window_ids = [window_id for _, window_id in bindings]
    if len(set(roles)) != len(roles) or len(set(window_ids)) != len(window_ids):
        _fail("Epoch role names and window IDs must each be unique.")
    by_id = {int(interval.window_id): interval for interval in selection.intervals}
    missing = sorted(set(window_ids) - set(by_id))
    if missing:
        _fail(f"Caller-bound epoch window IDs are absent: {missing!r}.")
    return [
        PositionSuiteEpoch(
            analysis_role=role,
            window_id=window_id,
            source_label=str(by_id[window_id].label),
            start_frame=int(by_id[window_id].start_frame),
            end_frame=int(by_id[window_id].end_frame),
            source_interval_sha256=str(by_id[window_id].source_interval_digest),
        )
        for role, window_id in bindings
    ]


def build_canary(
    archive: str | Path,
    *,
    provider_run: str,
    geometry_selection_run: str,
    expected_selection_record_sha256: str,
    expected_physical_authority_sha256: str,
    epoch_role_bindings: Sequence[tuple[str, int]],
    treatment_role: str = "aggressive",
    baseline_role: str = "inert",
    radial_bin_width_mm: float = 2.0,
    cdf_thresholds_mm: Sequence[float] = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20),
    near_zone_radius_mm: float = 5.0,
    near_entry_radius_mm: float = 5.0,
    near_exit_radius_mm: float = 6.0,
    perimeter_band_mm: float = 5.0,
    min_expected_count: float = 5.0,
) -> dict[str, Any]:
    """Load all exact authorities and compute one read-only position suite."""

    path = Path(archive).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {path}")
    provider_run = _strict_name(provider_run, field="provider_run")
    geometry_selection_run = _strict_name(
        geometry_selection_run, field="geometry_selection_run"
    )
    selection_digest = _digest(
        expected_selection_record_sha256,
        field="expected_selection_record_sha256",
    )
    physical_digest = _digest(
        expected_physical_authority_sha256,
        field="expected_physical_authority_sha256",
    )
    handle = load_provider_chaser_distance_source_handle(
        path,
        run_name=provider_run,
        use_consolidated=True,
    )
    semantics = _relative_semantic_binding(path, handle)
    epoch_binding = _candidate_epoch_binding(path, handle)
    validate_direct_consolidated_subtree(
        path,
        subtree_path=epoch_binding["epoch_run_path"],
    )
    selection = resolve_exact_stimulus_epoch_selection(
        path,
        run_name=epoch_binding["epoch_run_name"],
        expected_run_manifest_digest=None,
    )
    if (
        selection.run_manifest_payload_digest
        != epoch_binding["epoch_manifest_payload_sha256"]
    ):
        _fail("Exact epoch manifest payload differs from the sealed provider source.")
    epochs = _epoch_specs(selection, epoch_role_bindings)

    geometry_task = {
        "analysis_zarr": str(path),
        "recording_id": handle.recording_id,
        "geometry_source": {
            "selection_run_name": geometry_selection_run,
            "selection_record_sha256": selection_digest,
            "physical_authority_sha256": physical_digest,
        },
        "grid": {
            "policy_id": "provider_chaser_position_suite_arena_grid_v1",
            "bin_width_mm": 1.0,
        },
    }
    grid_policy, transform, geometry_evidence = load_grid_and_transform_authority(
        geometry_task
    )
    provider_authorities = _mapping(
        handle.manifest.get("source_provider_authorities"),
        label="source provider authorities",
    )
    source_position = _mapping(
        provider_authorities.get("source_position"),
        label="source position provider authority",
    )
    selection_record = geometry_evidence["selection"]["record"]
    coordinate_binding = selection_record["selected_candidate"]["coordinate_binding"]
    if source_position.get("coordinate_authority_id") != coordinate_binding.get(
        "pixel_frame_record_ref"
    ):
        _fail(
            "Provider positions and selected arena geometry use different pixel frames."
        )
    if (
        coordinate_binding.get("pixel_frame_record_sha256")
        != grid_policy.geometry.coordinate_authority_id
    ):
        _fail("Selected geometry pixel-frame digest differs from the grid authority.")
    source_scale_ref = source_position.get("scale_authority_id")
    provider_physical_frame = _exact_attribute_record(
        path,
        record_ref=source_scale_ref,
        expected_sha256=None,
        field="source position scale authority",
    )
    recording_physical_frame = _exact_attribute_record(
        path,
        record_ref=geometry_evidence["source_camera_physical_authority"][
            "physical_frame_record_ref"
        ],
        expected_sha256=geometry_evidence["source_camera_physical_authority"][
            "physical_frame_record_sha256"
        ],
        field="recording physical scale authority",
    )
    physical_frame_equivalence = _validate_physical_frame_semantic_equivalence(
        provider_physical_frame["record"],
        recording_physical_frame["record"],
    )

    temporal = _mapping(
        handle.manifest.get("temporal_alignment"),
        label="provider temporal alignment",
    )
    if temporal.get("temporal_alignment_class") != TEMPORAL_ALIGNMENT_CLASS:
        _fail("Provider run lacks the expected explicit temporal proxy class.")
    if temporal.get("physical_presentation_verified") is not False:
        _fail("Provider temporal alignment makes an unsupported presentation claim.")

    arrays = _provider_arrays(handle)
    frame_ids = np.asarray(arrays["acquisition_frame_id"], dtype=np.int64)
    if frame_ids.ndim != 1 or np.any(np.diff(frame_ids) <= 0):
        _fail("Provider acquisition-frame axis is not strictly increasing.")
    registries = semantics["identity_registries"]
    computed = compute_provider_chaser_position_suite(
        frame_ids=frame_ids,
        fish_xy_px=arrays["source_position_xy_px"],
        fish_valid=arrays["source_position_valid"],
        chaser_xy_px=arrays["chaser_position_xy_px"],
        chaser_valid=arrays["chaser_position_valid"],
        distance_px=arrays["distance_px"],
        distance_px_valid=arrays["distance_px_valid"],
        distance_mm=arrays["distance_mm"],
        distance_mm_valid=arrays["distance_mm_valid"],
        selection_member=arrays["selection_member"],
        chaser_occurrence_member=arrays["chaser_occurrence_member"],
        chaser_role_codes=arrays["chaser_behavior_role_code"],
        chaser_role_valid=arrays["chaser_behavior_role_valid"],
        chaser_identity_codes=arrays["chaser_identity_code"],
        role_registry=registries["behavior_role"],
        chaser_registry=registries["chaser"],
        epochs=epochs,
        arena=CircularArena(
            center_x_px=grid_policy.geometry.center_x_px,
            center_y_px=grid_policy.geometry.center_y_px,
            radius_px=grid_policy.geometry.radius_px,
            boundary_role=grid_policy.geometry.boundary_role,
            observed_feature=str(grid_policy.geometry.observed_feature),
        ),
        mm_per_pixel=grid_policy.scale.mm_per_pixel,
        fps=selection.fps,
        config=PositionSuiteConfig(
            radial_bin_width_mm=radial_bin_width_mm,
            cdf_thresholds_mm=tuple(cdf_thresholds_mm),
            near_zone_radius_mm=near_zone_radius_mm,
            near_entry_radius_mm=near_entry_radius_mm,
            near_exit_radius_mm=near_exit_radius_mm,
            perimeter_band_mm=perimeter_band_mm,
            min_expected_count=min_expected_count,
            treatment_role=treatment_role,
            baseline_role=baseline_role,
        ),
    )
    return json_attr_safe(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "disposition": DISPOSITION,
            "status": "computed_read_only",
            "recording_id": handle.recording_id,
            "analysis_zarr": str(path),
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
            "source_bindings": {
                "provider_chaser_distance": {
                    "run_name": handle.run_name,
                    "run_path": handle.run_path,
                    "manifest_sha256": handle.manifest_sha256,
                    "source_receipt_sha256": handle.source_receipt_sha256,
                    "verification_mode": handle.verification_mode,
                    "source_position_provider": source_position,
                },
                "relative_frame": semantics,
                "epoch_candidate": epoch_binding,
                "epoch_selection": selection.selection_record,
                "arena_geometry_and_scale": geometry_evidence,
                "provider_physical_frame": provider_physical_frame,
                "recording_physical_frame": recording_physical_frame,
                "physical_frame_equivalence": physical_frame_equivalence,
                "source_camera_to_arena_mm_transform": transform.as_record(),
            },
            "temporal_alignment": temporal,
            "temporal_caveat": (
                "Controller-input-provenance proxy only; state presentation time and "
                "camera exposure alignment are unavailable. Acquisition-frame epochs "
                "remain useful proxy analyses but are not exact presentation-response timing."
            ),
            "suite": computed,
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
    rows = report["suite"]["distance_cdf"]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["analysis_role"]), str(row["behavior_role"])), []
        ).append(row)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for (epoch, role), values in sorted(groups.items()):
        present = [row for row in values if row["fraction_at_or_below"] is not None]
        if present:
            ax.plot(
                [row["threshold_mm"] for row in present],
                [row["fraction_at_or_below"] for row in present],
                label=f"{epoch} · {role}",
            )
    ax.set(
        xlabel="fish–chaser distance threshold (mm)",
        ylabel="fraction at or below",
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title("Provider-aware fish–chaser distance CDF")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_radial(report: Mapping[str, Any], path: Path) -> None:
    rows = report["suite"]["radial_occupancy"]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["analysis_role"]), str(row["behavior_role"])), []
        ).append(row)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for (epoch, role), values in sorted(groups.items()):
        present = [
            row for row in values if row["selection_index_geometric"] is not None
        ]
        if present:
            ax.plot(
                [0.5 * (row["bin_start_mm"] + row["bin_end_mm"]) for row in present],
                [row["selection_index_geometric"] for row in present],
                label=f"{epoch} · {role}",
            )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="fish–chaser distance (mm)", ylabel="observed / geometric expected")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title("Area-corrected moving-chaser radial occupancy")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metric_bars(
    report: Mapping[str, Any],
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    rows = report["suite"]["per_epoch_chaser_metrics"]
    epoch_roles = [item["analysis_role"] for item in report["suite"]["epoch_roles"]]
    behavior_roles = sorted({str(row["behavior_role"]) for row in rows})
    x = np.arange(len(epoch_roles), dtype=np.float64)
    width = 0.8 / max(1, len(behavior_roles))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index, role in enumerate(behavior_roles):
        values = []
        for epoch in epoch_roles:
            candidates = [
                row
                for row in rows
                if row["analysis_role"] == epoch and row["behavior_role"] == role
            ]
            value = candidates[0][metric] if len(candidates) == 1 else None
            values.append(np.nan if value is None else float(value))
        ax.bar(
            x + (index - (len(behavior_roles) - 1) / 2) * width,
            values,
            width,
            label=role,
        )
    ax.set_xticks(x, epoch_roles)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_operational_canary(
    report: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    """Atomically publish compact evidence outside all scientific authorities."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing canary output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        completed = json_attr_safe(
            {
                **dict(report),
                "status": "complete_selector_ineligible_operational_canary",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "software": get_git_info(),
            }
        )
        (temporary / "canary_report.json").write_text(
            json.dumps(completed, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        suite = completed["suite"]
        tables = {
            "per_epoch_chaser_metrics.csv": suite["per_epoch_chaser_metrics"],
            "distance_cdf.csv": suite["distance_cdf"],
            "radial_occupancy.csv": suite["radial_occupancy"],
            "quadrant_joint_occupancy.csv": suite["quadrant_joint_occupancy"],
            "role_contrasts.csv": suite["role_contrasts"],
            "role_radial_contrasts.csv": suite["role_radial_contrasts"],
        }
        for name, rows in tables.items():
            _write_csv(temporary / name, rows)
        _plot_cdf(completed, temporary / "distance_cdf.png")
        _plot_radial(completed, temporary / "radial_selection_index.png")
        _plot_metric_bars(
            completed,
            temporary / "near_field_summary.png",
            metric="near_zone_fraction_valid",
            ylabel="fraction of valid tracked frames",
            title="Near-chaser occupancy (≤ configured radius)",
        )
        _plot_metric_bars(
            completed,
            temporary / "quadrant_summary.png",
            metric="same_quadrant_fraction_valid",
            ylabel="fraction of valid tracked frames",
            title="Fish and chaser in the same selected-arena quadrant",
        )
        artifacts = []
        for name in OUTPUT_FILES:
            artifact = temporary / name
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                _fail(f"Expected canary artifact {name!r} is absent or empty.")
            artifacts.append(
                {
                    "path": name,
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256_file(artifact),
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
        os.replace(temporary, target)
        return {
            "schema_id": f"{SCHEMA_ID}.publication_result",
            "schema_version": 1,
            "status": "published_selector_ineligible_operational_canary",
            "output_dir": str(target),
            "artifact_manifest": manifest,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--provider-run", required=True)
    parser.add_argument("--geometry-selection-run", required=True)
    parser.add_argument("--expected-selection-record-sha256", required=True)
    parser.add_argument("--expected-physical-authority-sha256", required=True)
    parser.add_argument(
        "--epoch-role",
        type=_parse_epoch_role,
        action="append",
        required=True,
        help="Explicit analysis-role binding in ROLE=WINDOW_ID form; repeat as needed.",
    )
    parser.add_argument("--treatment-role", default="aggressive")
    parser.add_argument("--baseline-role", default="inert")
    parser.add_argument("--radial-bin-width-mm", type=float, default=2.0)
    parser.add_argument(
        "--cdf-thresholds-mm",
        type=_parse_float_list,
        default=(2, 3, 4, 5, 6, 8, 10, 12, 15, 20),
    )
    parser.add_argument("--near-zone-radius-mm", type=float, default=5.0)
    parser.add_argument("--near-entry-radius-mm", type=float, default=5.0)
    parser.add_argument("--near-exit-radius-mm", type=float, default=6.0)
    parser.add_argument("--perimeter-band-mm", type=float, default=5.0)
    parser.add_argument("--min-expected-count", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_canary(
        args.zarr_path,
        provider_run=args.provider_run,
        geometry_selection_run=args.geometry_selection_run,
        expected_selection_record_sha256=args.expected_selection_record_sha256,
        expected_physical_authority_sha256=args.expected_physical_authority_sha256,
        epoch_role_bindings=args.epoch_role,
        treatment_role=args.treatment_role,
        baseline_role=args.baseline_role,
        radial_bin_width_mm=args.radial_bin_width_mm,
        cdf_thresholds_mm=args.cdf_thresholds_mm,
        near_zone_radius_mm=args.near_zone_radius_mm,
        near_entry_radius_mm=args.near_entry_radius_mm,
        near_exit_radius_mm=args.near_exit_radius_mm,
        perimeter_band_mm=args.perimeter_band_mm,
        min_expected_count=args.min_expected_count,
    )
    if args.apply:
        if args.output_dir is None:
            raise SystemExit("--apply requires --output-dir")
        payload = publish_operational_canary(report, output_dir=args.output_dir)
    else:
        compact_metrics = [
            {
                key: row[key]
                for key in (
                    "analysis_role",
                    "epoch_window_id",
                    "behavior_role",
                    "chaser_identity",
                    "source_interval_frame_count",
                    "epoch_provider_frame_count",
                    "epoch_provider_frame_coverage_fraction",
                    "candidate_frame_count",
                    "valid_distance_frame_count",
                    "valid_distance_fraction",
                    "distance_p50_mm",
                    "same_quadrant_fraction_valid",
                    "near_zone_fraction_valid",
                    "near_zone_entry_count",
                    "near_zone_entry_rate_per_min_valid_time",
                    "fish_arena_radius_mean_mm",
                    "fish_wall_distance_mean_mm",
                )
            }
            for row in report["suite"]["per_epoch_chaser_metrics"]
        ]
        payload = {
            "schema_id": f"{SCHEMA_ID}.plan",
            "schema_version": 1,
            "status": "planned_no_writes",
            "recording_id": report["recording_id"],
            "provider_run": report["source_bindings"]["provider_chaser_distance"][
                "run_name"
            ],
            "epoch_roles": report["suite"]["epoch_roles"],
            "planned_output_dir": None
            if args.output_dir is None
            else str(args.output_dir.resolve()),
            "planned_artifacts": [*OUTPUT_FILES, "artifact_manifest.json"],
            "summary": {
                "source_authority": {
                    "provider_manifest_sha256": report["source_bindings"][
                        "provider_chaser_distance"
                    ]["manifest_sha256"],
                    "epoch_selection_sha256": report["source_bindings"][
                        "epoch_selection"
                    ]["selection_sha256"],
                    "geometry_selection_sha256": report["source_bindings"][
                        "arena_geometry_and_scale"
                    ]["selection"]["sha256"],
                    "physical_frame_equivalence": report["source_bindings"][
                        "physical_frame_equivalence"
                    ],
                },
                "arena": report["suite"]["arena"],
                "config": report["suite"]["config"],
                "per_epoch_chaser_row_count": len(
                    report["suite"]["per_epoch_chaser_metrics"]
                ),
                "per_epoch_chaser_metrics": compact_metrics,
                "radial_row_count": len(report["suite"]["radial_occupancy"]),
                "quadrant_joint_row_count": len(
                    report["suite"]["quadrant_joint_occupancy"]
                ),
                "role_contrast_row_count": len(report["suite"]["role_contrasts"]),
                "role_contrasts": report["suite"]["role_contrasts"],
                "temporal_caveat": report["temporal_caveat"],
                "selector_eligible": False,
                "registry_update": False,
                "analysis_zarr_write": False,
            },
        }
    print(
        json.dumps(json_attr_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DISPOSITION",
    "OUTPUT_FILES",
    "ProviderChaserPositionSuiteCanaryError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "build_canary",
    "publish_operational_canary",
]
