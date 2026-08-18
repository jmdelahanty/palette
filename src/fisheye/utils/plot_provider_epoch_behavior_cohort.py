"""Render bounded talk plots from one exact provider-epoch cohort generation.

This utility is deliberately downstream of
``export_provider_epoch_behavior_cohort``.  It accepts either the exact
immutable generation selected by its publication manifest or the exact two
Parquet parts named by that manifest.  It does not resolve a selector, write
to a Zarr archive, update the registry, or mutate the source tables.

The figures use the semantic epoch labels ``pre_event``, ``training_event``,
and ``post_event``.  Epoch colors are neutral presentation colors; they do
not encode a behavioral class or the color of an experimental stimulus.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.arrow_contract_core import validate_exact_schema
from fisheye.analytics_exports.derived_publication import (
    derived_manifest_path,
    derived_manifest_selected_parts,
    validate_derived_manifest_envelope,
)
from fisheye.analytics_exports.publication import sha256_file
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.export_provider_epoch_behavior_cohort import (
    ARROW_ENVELOPE_SCHEMA_ID,
    ARROW_ENVELOPE_SCHEMA_VERSION,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_BOUTS,
    TABLE_FISH,
    TABLE_NAMES,
    table_contracts_for_disposition,
)


PLOT_SCHEMA_ID = "palette.provider_epoch_behavior_cohort_plots"
PLOT_SCHEMA_VERSION = 1
EXPECTED_EPOCH_LABELS = ("pre_event", "training_event", "post_event")
EXPECTED_EPOCH_IDS = (0, 1, 2)
EXPECTED_EPOCH_IDENTITIES = tuple(zip(EXPECTED_EPOCH_IDS, EXPECTED_EPOCH_LABELS))
PLOT_METRICS = {
    "bout_rate_per_min": ("Bout rate", "Bout rate (events min$^{-1}$)"),
    "mean_speed_mm_s": ("Mean speed", "Mean speed (mm s$^{-1}$)"),
    "mean_bout_duration_s": ("Mean bout duration", "Mean bout duration (s)"),
}
NEUTRAL_EPOCH_COLORS = {
    "pre_event": "#6B7280",
    "training_event": "#8B8B72",
    "post_event": "#B0A37A",
}
_PREFIX_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


class ProviderEpochBehaviorPlotError(ValueError):
    """Raised when an exact cohort cannot be plotted safely."""


@dataclass(frozen=True)
class PlotUnit:
    """One recording/animal/track unit, intentionally not an animal-only row."""

    recording_id: str
    subject_id: str | None
    track_id: int
    values_by_metric: Mapping[str, tuple[float | None, ...]]

    @property
    def unit_id(self) -> str:
        subject = self.subject_id if self.subject_id is not None else "<missing-subject-id>"
        return f"{self.recording_id}::{subject}::track-{self.track_id}"


@dataclass(frozen=True)
class ValidatedCohort:
    """Validated in-memory view of one immutable cohort generation."""

    manifest: Mapping[str, Any]
    bouts_table: Any
    fish_table: Any
    units: tuple[PlotUnit, ...]
    source_tables: Mapping[str, Mapping[str, Any]]

    @property
    def cohort_id(self) -> str:
        return str(self.manifest["cohort_id"])

    @property
    def analysis_run_id(self) -> str:
        return str(self.manifest["analysis_run_id"])

    @property
    def generation_id(self) -> str:
        publication = self.manifest["publication"]
        assert isinstance(publication, Mapping)
        return str(publication["generation_id"])

    @property
    def metric_disposition(self) -> str:
        return str(self.manifest["metric_disposition"])

    @property
    def n_recording_animal_sessions(self) -> int:
        return len(self.units)

    @property
    def n_recordings(self) -> int:
        return len({unit.recording_id for unit in self.units})

    @property
    def n_subjects(self) -> int:
        return len({unit.subject_id for unit in self.units if unit.subject_id is not None})


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ProviderEpochBehaviorPlotError(
            "pyarrow is required to consume the cohort Parquet tables."
        ) from exc
    return pa, pq


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        # Matplotlib otherwise salts SVG element IDs from process state.  A
        # fixed salt makes the vector bytes reproducible across invocations.
        matplotlib.rcParams["svg.hashsalt"] = PLOT_SCHEMA_ID
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ProviderEpochBehaviorPlotError(
            "matplotlib is required to render provider-epoch cohort plots."
        ) from exc
    return plt


def _canonical_sha256(value: object) -> str:
    return canonical_json_sha256(value)


def _as_text(value: object, *, label: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="strict")
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProviderEpochBehaviorPlotError(f"{label} must be a non-empty string.")
    return value


def _as_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ProviderEpochBehaviorPlotError(f"{label} must be an integer.")
    return int(value)


def _as_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        number = float(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderEpochBehaviorPlotError(f"{label} must be numeric or null.") from exc
    if not np.isfinite(number):
        raise ProviderEpochBehaviorPlotError(f"{label} must be finite or null.")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderEpochBehaviorPlotError(f"Cannot read cohort publication manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ProviderEpochBehaviorPlotError("Cohort publication manifest must be a JSON object.")
    return payload


def _validate_publication_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the shared exact publication and the cohort-specific contract."""

    if manifest.get("export_schema_id") != EXPORT_SCHEMA_ID:
        raise ProviderEpochBehaviorPlotError("The source is not a provider-epoch cohort export.")
    if manifest.get("export_schema_version") != EXPORT_SCHEMA_VERSION:
        raise ProviderEpochBehaviorPlotError("The provider-epoch cohort export schema version is unsupported.")
    if manifest.get("selector_eligible") is not False:
        raise ProviderEpochBehaviorPlotError("Plotting requires a selector-ineligible cohort export.")
    if manifest.get("metric_disposition") != "linear_only":
        raise ProviderEpochBehaviorPlotError(
            "Talk plots require the explicitly declared linear_only metric disposition."
        )
    reason = manifest.get("metric_disposition_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ProviderEpochBehaviorPlotError("linear_only requires a non-empty disposition reason.")
    if not isinstance(manifest.get("cohort_id"), str) or not manifest["cohort_id"].strip():
        raise ProviderEpochBehaviorPlotError("The cohort manifest must identify one cohort.")
    if not isinstance(manifest.get("analysis_run_id"), str) or not manifest["analysis_run_id"].strip():
        raise ProviderEpochBehaviorPlotError("The cohort manifest must identify one analysis run.")
    if type(manifest.get("recording_count")) is not int or manifest["recording_count"] <= 0:
        raise ProviderEpochBehaviorPlotError("The cohort manifest must contain a positive recording count.")

    contracts = table_contracts_for_disposition("linear_only")
    try:
        validate_derived_manifest_envelope(
            manifest,
            analysis_run_id=str(manifest["analysis_run_id"]),
            table_names=TABLE_NAMES,
            contracts=contracts,
            arrow_envelope_schema_id=ARROW_ENVELOPE_SCHEMA_ID,
            arrow_envelope_schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderEpochBehaviorPlotError(
            f"The exact cohort publication manifest is invalid: {exc}"
        ) from exc

    publication = manifest.get("publication")
    if not isinstance(publication, Mapping) or publication.get("selector_eligible") is not False:
        raise ProviderEpochBehaviorPlotError("The exact cohort generation must be selector-ineligible.")
    if manifest.get("output_tables") != list(TABLE_NAMES):
        raise ProviderEpochBehaviorPlotError("The cohort must contain exactly the two provider-epoch tables.")
    return dict(manifest)


def _validate_table_schema(table: Any, table_name: str) -> None:
    contracts = table_contracts_for_disposition("linear_only")
    if table_name not in contracts:
        raise ProviderEpochBehaviorPlotError(f"Unknown cohort table: {table_name}")
    try:
        validate_exact_schema(contracts[table_name], table.schema)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProviderEpochBehaviorPlotError(
            f"{table_name}: Parquet schema is not the exact linear_only export contract."
        ) from exc


def _table_rows(table: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in table.to_pylist()]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProviderEpochBehaviorPlotError("Cohort tables must be PyArrow tables.") from exc


def _validate_rows(
    manifest: Mapping[str, Any],
    bouts_table: Any,
    fish_table: Any,
) -> tuple[PlotUnit, ...]:
    _validate_table_schema(bouts_table, TABLE_BOUTS)
    _validate_table_schema(fish_table, TABLE_FISH)
    expected_row_counts = manifest["row_counts_by_table"]
    if not isinstance(expected_row_counts, Mapping):
        raise ProviderEpochBehaviorPlotError("The cohort row-count inventory is missing.")
    for table_name, table in ((TABLE_BOUTS, bouts_table), (TABLE_FISH, fish_table)):
        if int(table.num_rows) != int(expected_row_counts[table_name]):
            raise ProviderEpochBehaviorPlotError(
                f"{table_name}: table row count differs from the immutable publication inventory."
            )
    fish_rows = _table_rows(fish_table)
    bout_rows = _table_rows(bouts_table)
    if not fish_rows:
        raise ProviderEpochBehaviorPlotError("The cohort contains no per-recording epoch rows.")

    expected_by_unit: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    subject_by_unit: dict[tuple[str, int], str | None] = {}
    for row in fish_rows:
        recording_id = _as_text(row.get("recording_id"), label="recording_id")
        assert recording_id is not None
        track_id = _as_int(row.get("track_id"), label="track_id")
        epoch_id = _as_int(row.get("epoch_id"), label="epoch_id")
        epoch_index = _as_int(row.get("epoch_index"), label="epoch_index")
        epoch_label = _as_text(row.get("epoch_label"), label="epoch_label")
        assert epoch_label is not None
        subject_id = _as_text(row.get("subject_id"), label="subject_id", allow_none=True)
        if epoch_index != epoch_id or (epoch_id, epoch_label) not in EXPECTED_EPOCH_IDENTITIES:
            raise ProviderEpochBehaviorPlotError(
                f"{recording_id}: epoch identity must be the ordered pre/training/post contract."
            )
        unit_key = (recording_id, track_id)
        if unit_key not in expected_by_unit:
            expected_by_unit[unit_key] = {}
            subject_by_unit[unit_key] = subject_id
        elif subject_by_unit[unit_key] != subject_id:
            raise ProviderEpochBehaviorPlotError(
                f"{recording_id}, track {track_id}: subject_id changes across epochs."
            )
        if epoch_id in expected_by_unit[unit_key]:
            raise ProviderEpochBehaviorPlotError(
                f"{recording_id}, track {track_id}: duplicate epoch row {epoch_id}."
            )
        expected_by_unit[unit_key][epoch_id] = row

    expected_epochs = set(EXPECTED_EPOCH_IDS)
    for unit_key, rows in expected_by_unit.items():
        if set(rows) != expected_epochs:
            raise ProviderEpochBehaviorPlotError(
                f"{unit_key[0]}, track {unit_key[1]}: expected exactly three pre/training/post epochs."
            )

    bout_keys: set[tuple[str, int, int, int]] = set()
    for row in bout_rows:
        recording_id = _as_text(row.get("recording_id"), label="recording_id")
        assert recording_id is not None
        track_id = _as_int(row.get("track_id"), label="track_id")
        epoch_id = _as_int(row.get("epoch_id"), label="epoch_id")
        epoch_label = _as_text(row.get("epoch_label"), label="epoch_label")
        assert epoch_label is not None
        bout_source_row = _as_int(row.get("bout_source_row"), label="bout_source_row")
        key = (recording_id, track_id, epoch_id, bout_source_row)
        if key in bout_keys:
            raise ProviderEpochBehaviorPlotError(f"Duplicate bout primary key: {key}")
        bout_keys.add(key)
        if (recording_id, track_id) not in expected_by_unit:
            raise ProviderEpochBehaviorPlotError("A bout row has no matching recording/track epoch summary.")
        if epoch_id not in expected_epochs or epoch_label != EXPECTED_EPOCH_LABELS[epoch_id]:
            raise ProviderEpochBehaviorPlotError("A bout row has an invalid epoch identity.")
        fish_subject = subject_by_unit[(recording_id, track_id)]
        bout_subject = _as_text(row.get("subject_id"), label="subject_id", allow_none=True)
        if fish_subject != bout_subject:
            raise ProviderEpochBehaviorPlotError("Bout and epoch-summary subject_id identities differ.")

    if len(expected_by_unit) != int(manifest["recording_count"]):
        raise ProviderEpochBehaviorPlotError(
            "The manifest recording_count does not match the distinct recording/track rows."
        )

    metric_names = tuple(name for name in PLOT_METRICS if name in fish_table.column_names)
    units: list[PlotUnit] = []
    for (recording_id, track_id), rows in sorted(expected_by_unit.items()):
        units.append(
            PlotUnit(
                recording_id=recording_id,
                subject_id=subject_by_unit[(recording_id, track_id)],
                track_id=track_id,
                values_by_metric={
                    metric: tuple(
                        _as_number(rows[epoch_id].get(metric), label=f"{metric} {recording_id} epoch {epoch_id}")
                        for epoch_id in EXPECTED_EPOCH_IDS
                    )
                    for metric in metric_names
                },
            )
        )
    return tuple(units)


def validate_cohort_tables(
    *,
    bouts_table: Any,
    fish_table: Any,
    manifest: Mapping[str, Any],
    source_tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> ValidatedCohort:
    """Validate in-memory exact tables and return the immutable plot view.

    This is also the seam used by unit tests.  Production callers should use
    :func:`plot_provider_epoch_behavior_cohort` or
    :func:`plot_provider_epoch_behavior_cohort_parquet` so the publication
    manifest's file digests are checked before this function is called.
    """

    checked_manifest = _validate_publication_manifest(manifest)
    units = _validate_rows(checked_manifest, bouts_table, fish_table)
    return ValidatedCohort(
        manifest=checked_manifest,
        bouts_table=bouts_table,
        fish_table=fish_table,
        units=units,
        source_tables=dict(source_tables or {}),
    )


def _manifest_output_root(manifest_path: Path) -> Path:
    path = manifest_path.expanduser().resolve()
    if path.parent.name != "manifests" or path.parent.parent.name != "v2":
        raise ProviderEpochBehaviorPlotError(
            "The cohort manifest must be located below <output-root>/v2/manifests."
        )
    return path.parent.parent.parent


def _load_manifest_and_paths(
    *,
    generation_root: Path | None = None,
    bouts_parquet: Path | None = None,
    fish_parquet: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Mapping[str, Any]]]:
    if generation_root is not None and (bouts_parquet is not None or fish_parquet is not None):
        raise ProviderEpochBehaviorPlotError("Choose a generation root or two Parquet paths, not both.")
    if (bouts_parquet is None) != (fish_parquet is None):
        raise ProviderEpochBehaviorPlotError("Both exact Parquet table paths are required.")

    if generation_root is not None:
        generation = generation_root.expanduser().resolve()
        if not generation.is_dir() or not generation.name.startswith("generation="):
            raise ProviderEpochBehaviorPlotError(
                "generation_root must be one exact v2/.generations/.../generation=<id> directory."
            )
        analysis_dir = generation.parent
        if not analysis_dir.name.startswith("analysis_run_id="):
            raise ProviderEpochBehaviorPlotError("The generation path is missing its analysis_run_id component.")
        analysis_run_id = analysis_dir.name.split("=", 1)[1]
        if manifest_path is None:
            manifest_path = derived_manifest_path(
                generation.parents[3],
                analysis_run_id,
            )
        manifest_path = manifest_path.expanduser().resolve()
        manifest = _validate_publication_manifest(_read_json(manifest_path))
        output_root = _manifest_output_root(manifest_path)
        expected_generation = (output_root / str(manifest["publication"]["generation_path"])).resolve()
        if expected_generation != generation:
            raise ProviderEpochBehaviorPlotError(
                "The supplied generation is not the generation named by the exact publication manifest."
            )
        selected_parts = {
            table_name: derived_manifest_selected_parts(
                output_root,
                manifest,
                table_name,
                table_names=TABLE_NAMES,
            )
            for table_name in TABLE_NAMES
        }
        if any(len(parts) != 1 for parts in selected_parts.values()):
            raise ProviderEpochBehaviorPlotError(
                "Talk plotting requires exactly one immutable Parquet part per table."
            )
        paths = {table_name: parts[0] for table_name, parts in selected_parts.items()}
    elif bouts_parquet is not None and fish_parquet is not None:
        if manifest_path is None:
            raise ProviderEpochBehaviorPlotError(
                "Exact Parquet table inputs require their publication manifest."
            )
        manifest_path = manifest_path.expanduser().resolve()
        manifest = _validate_publication_manifest(_read_json(manifest_path))
        output_root = _manifest_output_root(manifest_path)
        selected_part_sets = {
            TABLE_BOUTS: derived_manifest_selected_parts(
                output_root, manifest, TABLE_BOUTS, table_names=TABLE_NAMES
            ),
            TABLE_FISH: derived_manifest_selected_parts(
                output_root, manifest, TABLE_FISH, table_names=TABLE_NAMES
            ),
        }
        if any(len(parts) != 1 for parts in selected_part_sets.values()):
            raise ProviderEpochBehaviorPlotError(
                "Talk plotting requires exactly one immutable Parquet part per table."
            )
        selected_parts = {
            table_name: parts[0]
            for table_name, parts in selected_part_sets.items()
        }
        paths = {
            TABLE_BOUTS: bouts_parquet.expanduser().resolve(),
            TABLE_FISH: fish_parquet.expanduser().resolve(),
        }
        for table_name, path in paths.items():
            expected = selected_parts[table_name]
            inventory = manifest["publication"]["parts_by_table"][table_name][0]
            if not path.is_file() or path.stat().st_size != inventory["size_bytes"]:
                raise ProviderEpochBehaviorPlotError(f"{table_name}: exact Parquet part is missing or size differs.")
            if sha256_file(path) != inventory["sha256"]:
                raise ProviderEpochBehaviorPlotError(f"{table_name}: exact Parquet part digest differs.")
            if path != expected:
                # A byte-identical explicitly supplied table is acceptable, but
                # its immutable publication identity remains the manifest digest.
                pass
    else:
        raise ProviderEpochBehaviorPlotError("An exact generation root or exact Parquet pair is required.")

    source_tables: dict[str, Mapping[str, Any]] = {}
    publication = manifest["publication"]
    assert isinstance(publication, Mapping)
    inventory = publication["parts_by_table"]
    assert isinstance(inventory, Mapping)
    for table_name, path in paths.items():
        entry = inventory[table_name][0]
        source_tables[table_name] = {
            "path": str(path),
            "published_path": entry["path"],
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
            "row_count": entry["row_count"],
        }
    return manifest, paths, source_tables


def _load_exact_source(
    *,
    generation_root: Path | None = None,
    bouts_parquet: Path | None = None,
    fish_parquet: Path | None = None,
    manifest_path: Path | None = None,
) -> ValidatedCohort:
    manifest, paths, source_tables = _load_manifest_and_paths(
        generation_root=generation_root,
        bouts_parquet=bouts_parquet,
        fish_parquet=fish_parquet,
        manifest_path=manifest_path,
    )
    _, pq = _pyarrow()
    try:
        # ``pq.read_table(path)`` invokes dataset partition discovery and can
        # inject Hive directory fields such as ``analysis_run_id`` and
        # ``generation``.  These are exact files with already validated
        # digests, so read their physical schemas without dataset inference.
        bouts_table = pq.ParquetFile(paths[TABLE_BOUTS]).read()
        fish_table = pq.ParquetFile(paths[TABLE_FISH]).read()
    except Exception as exc:
        raise ProviderEpochBehaviorPlotError("Cannot read the exact cohort Parquet tables.") from exc
    return validate_cohort_tables(
        bouts_table=bouts_table,
        fish_table=fish_table,
        manifest=manifest,
        source_tables=source_tables,
    )


def _finite_stats(values: Sequence[float | None]) -> tuple[float | None, float | None, int]:
    observed = np.asarray([value for value in values if value is not None], dtype=np.float64)
    observed = observed[np.isfinite(observed)]
    count = int(observed.size)
    if count == 0:
        return None, None, 0
    mean = float(np.mean(observed))
    sem = float(np.std(observed, ddof=1) / np.sqrt(count)) if count > 1 else 0.0
    return mean, sem, count


def _subject_level_matrix(data: ValidatedCohort, metric: str) -> np.ndarray:
    """Aggregate repeated recording sessions to one row per subject first."""

    if any(unit.subject_id is None for unit in data.units):
        raise ProviderEpochBehaviorPlotError(
            "Grouped animal-level plots require subject_id for every recording session."
        )
    sessions_by_subject: dict[str, list[PlotUnit]] = {}
    for unit in data.units:
        assert unit.subject_id is not None
        sessions_by_subject.setdefault(unit.subject_id, []).append(unit)
    subject_rows: list[list[float]] = []
    for subject_id in sorted(sessions_by_subject):
        sessions = sessions_by_subject[subject_id]
        row: list[float] = []
        for epoch_index in range(len(EXPECTED_EPOCH_LABELS)):
            values = [
                unit.values_by_metric[metric][epoch_index]
                for unit in sessions
                if unit.values_by_metric[metric][epoch_index] is not None
            ]
            row.append(float(np.mean(values)) if values else np.nan)
        subject_rows.append(row)
    return np.asarray(subject_rows, dtype=np.float64)


def _configure_axis(ax: Any, *, ylabel: str) -> None:
    ax.set_xticks(np.arange(len(EXPECTED_EPOCH_LABELS)))
    ax.set_xticklabels(("Pre-event", "Training event", "Post-event"))
    ax.set_xlabel("Stimulus epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D1D5DB", alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)


def _draw_epoch_band(ax: Any, *, y_min: float, y_max: float) -> None:
    for index, label in enumerate(EXPECTED_EPOCH_LABELS):
        ax.axvspan(index - 0.5, index + 0.5, color=NEUTRAL_EPOCH_COLORS[label], alpha=0.08, linewidth=0)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
        ax.set_ylim(y_min, y_max)


def _render_grouped_metric(
    data: ValidatedCohort,
    metric: str,
    path: Path,
    *,
    plt: Any,
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    matrix = _subject_level_matrix(data, metric)
    finite_all = matrix[np.isfinite(matrix)]
    if finite_all.size:
        low = float(np.min(finite_all))
        high = float(np.max(finite_all))
    else:
        low, high = 0.0, 1.0
    for row in matrix:
        observed = np.isfinite(row)
        if observed.any():
            ax.plot(
                np.arange(3)[observed],
                row[observed],
                color="#9CA3AF",
                alpha=0.24,
                linewidth=0.75,
                marker="o",
                markersize=2.4,
            )
    means: list[float | None] = []
    sems: list[float | None] = []
    counts: list[int] = []
    for index in range(3):
        values = [None if not np.isfinite(row[index]) else float(row[index]) for row in matrix]
        mean, sem, count = _finite_stats(values)
        means.append(mean)
        sems.append(sem)
        counts.append(count)
    mean_array = np.asarray([np.nan if value is None else value for value in means], dtype=np.float64)
    sem_array = np.asarray([0.0 if value is None else value for value in sems], dtype=np.float64)
    observed = np.isfinite(mean_array)
    _draw_epoch_band(ax, y_min=low, y_max=high)
    if observed.any():
        ax.errorbar(
            np.arange(3)[observed],
            mean_array[observed],
            yerr=sem_array[observed],
            color="#1F2937",
            marker="o",
            markersize=5,
            linewidth=2.0,
            capsize=3,
            label="Mean ± SEM",
            zorder=5,
        )
        ax.legend(frameon=False, loc="best")
    _configure_axis(ax, ylabel=ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format=path.suffix.lstrip("."), dpi=160, metadata={"Date": None})
    plt.close(fig)
    return {
        "metric": metric,
        "finite_subject_counts_by_epoch": counts,
        "mean_by_epoch": means,
        "sem_by_epoch": sems,
        "uncertainty": "standard_error_of_mean",
    }


def _render_grouped_speed_duration(
    data: ValidatedCohort,
    metrics: Sequence[str],
    path: Path,
    *,
    plt: Any,
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0 * len(metrics), 5.2), squeeze=False, dpi=160)
    stats: dict[str, Any] = {}
    for axis, metric in zip(axes[0], metrics):
        # Render into the supplied axis using one row per subject, after
        # equal-weight averaging of that subject's repeated sessions.
        matrix = _subject_level_matrix(data, metric)
        finite_all = matrix[np.isfinite(matrix)]
        low = float(np.min(finite_all)) if finite_all.size else 0.0
        high = float(np.max(finite_all)) if finite_all.size else 1.0
        for row in matrix:
            observed = np.isfinite(row)
            if observed.any():
                axis.plot(np.arange(3)[observed], row[observed], color="#9CA3AF", alpha=0.24, linewidth=0.75, marker="o", markersize=2.4)
        means: list[float | None] = []
        sems: list[float | None] = []
        counts: list[int] = []
        for index in range(3):
            values = [None if not np.isfinite(row[index]) else float(row[index]) for row in matrix]
            mean, sem, count = _finite_stats(values)
            means.append(mean)
            sems.append(sem)
            counts.append(count)
        mean_array = np.asarray([np.nan if value is None else value for value in means], dtype=np.float64)
        sem_array = np.asarray([0.0 if value is None else value for value in sems], dtype=np.float64)
        observed = np.isfinite(mean_array)
        _draw_epoch_band(axis, y_min=low, y_max=high)
        if observed.any():
            axis.errorbar(np.arange(3)[observed], mean_array[observed], yerr=sem_array[observed], color="#1F2937", marker="o", markersize=5, linewidth=2.0, capsize=3, label="Mean ± SEM", zorder=5)
        _configure_axis(axis, ylabel=PLOT_METRICS[metric][1])
        axis.set_title(PLOT_METRICS[metric][0])
        stats[metric] = {
            "metric": metric,
            "finite_subject_counts_by_epoch": counts,
            "mean_by_epoch": means,
            "sem_by_epoch": sems,
            "uncertainty": "standard_error_of_mean",
        }
    if any(
        np.isfinite(
            np.asarray(
                [value if value is not None else np.nan for value in stats[metric]["mean_by_epoch"]]
            )
        ).any()
        for metric in metrics
    ):
        axes[0][0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, format=path.suffix.lstrip("."), dpi=160, metadata={"Date": None})
    plt.close(fig)
    return stats


def _save_individual_formats(data: ValidatedCohort, output_dir: Path, prefix: str, plt: Any) -> list[Path]:
    generated: list[Path] = []
    for extension in ("png", "svg"):
        path = output_dir / f"{prefix}.individual_bout_rate.{extension}"
        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
        all_values: list[float] = []
        for unit in data.units:
            values = np.asarray([np.nan if value is None else value for value in unit.values_by_metric["bout_rate_per_min"]], dtype=np.float64)
            all_values.extend(values[np.isfinite(values)].tolist())
            observed = np.isfinite(values)
            if observed.any():
                ax.plot(np.arange(3)[observed], values[observed], color="#4B5563", alpha=0.28, linewidth=0.8, marker="o", markersize=2.8)
        low = min(all_values) if all_values else 0.0
        high = max(all_values) if all_values else 1.0
        _draw_epoch_band(ax, y_min=low, y_max=high)
        _configure_axis(ax, ylabel=PLOT_METRICS["bout_rate_per_min"][1])
        ax.set_title("Bout rate for each recording × animal session")
        ax.text(0.99, 0.98, f"n = {data.n_recording_animal_sessions} recording×animal sessions", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#4B5563")
        fig.tight_layout()
        fig.savefig(path, format=extension, dpi=160, metadata={"Date": None})
        plt.close(fig)
        generated.append(path)
    return generated


def _save_grouped_metric_formats(
    data: ValidatedCohort,
    metric: str,
    output_dir: Path,
    prefix: str,
    plt: Any,
) -> tuple[list[Path], dict[str, Any]]:
    generated: list[Path] = []
    stats: dict[str, Any] | None = None
    for extension in ("png", "svg"):
        path = output_dir / f"{prefix}.grouped_{metric}.{extension}"
        # The renderer closes its figure; recomputing stats is deterministic.
        stats = _render_grouped_metric(data, metric, path, plt=plt, title=f"{PLOT_METRICS[metric][0]} across unique subjects", ylabel=PLOT_METRICS[metric][1])
        generated.append(path)
    assert stats is not None
    return generated, stats


def _save_speed_duration_formats(
    data: ValidatedCohort,
    metrics: Sequence[str],
    output_dir: Path,
    prefix: str,
    plt: Any,
) -> tuple[list[Path], dict[str, Any]]:
    generated: list[Path] = []
    stats: dict[str, Any] | None = None
    for extension in ("png", "svg"):
        path = output_dir / f"{prefix}.grouped_speed_duration.{extension}"
        stats = _render_grouped_speed_duration(data, metrics, path, plt=plt)
        generated.append(path)
    assert stats is not None
    return generated, stats


def render_provider_epoch_behavior_cohort(
    data: ValidatedCohort,
    *,
    output_dir: Path,
    prefix: str = "provider_epoch_behavior_cohort",
) -> dict[str, Any]:
    """Render deterministic figures and a receipt from a validated cohort."""

    if not isinstance(prefix, str) or not _PREFIX_RE.fullmatch(prefix):
        raise ProviderEpochBehaviorPlotError("prefix must be one safe portable filename component.")
    if any(unit.subject_id is None for unit in data.units):
        raise ProviderEpochBehaviorPlotError(
            "Grouped animal-level plots require subject_id for every recording session."
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = tuple(
        metric for metric in ("mean_speed_mm_s", "mean_bout_duration_s") if metric in data.units[0].values_by_metric
    )
    output_paths = [
        output_dir / f"{prefix}.individual_bout_rate.png",
        output_dir / f"{prefix}.individual_bout_rate.svg",
        output_dir / f"{prefix}.grouped_bout_rate.png",
        output_dir / f"{prefix}.grouped_bout_rate.svg",
        output_dir / f"{prefix}.receipt.json",
    ]
    if metric_names:
        output_paths.extend(
            [
                output_dir / f"{prefix}.grouped_speed_duration.png",
                output_dir / f"{prefix}.grouped_speed_duration.svg",
            ]
        )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing plot artifacts: {existing[0]}")

    plt = _matplotlib()
    generated: list[Path] = []
    plot_stats: dict[str, Any] = {}
    generated.extend(_save_individual_formats(data, output_dir, prefix, plt))
    grouped_paths, bout_stats = _save_grouped_metric_formats(data, "bout_rate_per_min", output_dir, prefix, plt)
    generated.extend(grouped_paths)
    plot_stats["bout_rate_per_min"] = bout_stats
    if metric_names:
        metric_paths, metric_stats = _save_speed_duration_formats(data, metric_names, output_dir, prefix, plt)
        generated.extend(metric_paths)
        plot_stats.update(metric_stats)

    source_manifest_sha256 = _canonical_sha256(data.manifest)
    session_counts_by_subject: dict[str, int] = {}
    for unit in data.units:
        assert unit.subject_id is not None
        session_counts_by_subject[unit.subject_id] = (
            session_counts_by_subject.get(unit.subject_id, 0) + 1
        )
    unsigned_receipt: dict[str, Any] = {
        "schema_id": PLOT_SCHEMA_ID,
        "schema_version": PLOT_SCHEMA_VERSION,
        "cohort_id": data.cohort_id,
        "analysis_run_id": data.analysis_run_id,
        "generation_id": data.generation_id,
        "metric_disposition": data.metric_disposition,
        "metric_disposition_reason": data.manifest["metric_disposition_reason"],
        "selector_eligible": False,
        "source_manifest_sha256": source_manifest_sha256,
        "source_tables": dict(data.source_tables),
        "expected_epoch_labels": list(EXPECTED_EPOCH_LABELS),
        "n_recordings": data.n_recordings,
        "n_subjects": data.n_subjects,
        "recording_count": data.n_recording_animal_sessions,
        "recording_animal_unit_count": data.n_recording_animal_sessions,
        "missing_subject_id_unit_count": sum(unit.subject_id is None for unit in data.units),
        "unit_identity": "recording_id_subject_id_track_id",
        "grouping_unit": "subject_id",
        "repeated_session_aggregation": "arithmetic_mean_within_subject_id_epoch",
        "session_weighting": "equal",
        "session_counts_by_subject": dict(sorted(session_counts_by_subject.items())),
        "grouped_estimate_level": "unique_subject_id",
        "uncertainty": "standard_error_of_mean",
        "epoch_colors": dict(NEUTRAL_EPOCH_COLORS),
        "metrics": plot_stats,
        "figures": [
            {
                "path": path.name,
                "media_type": "image/png" if path.suffix == ".png" else "image/svg+xml",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(generated)
        ],
        "scientific_notes": [
            "Repeated recordings remain separate recording×animal sessions in the source tables and session spaghetti figure.",
            "Grouped estimates first average repeated sessions within subject_id×epoch with equal session weighting, then summarize across unique subjects.",
            "The grouped overlay contains one line per unique subject after repeated-session aggregation.",
            "Epoch colors are neutral presentation colors and do not encode behavioral class or stimulus color.",
            "Group uncertainty is the standard error of the mean across unique subjects with finite metric values after within-subject session averaging.",
            "The linear_only source disposition excludes heading metrics; these figures report linear motion and bout metrics only.",
        ],
    }
    receipt = {
        **unsigned_receipt,
        "receipt_payload_sha256": _canonical_sha256(unsigned_receipt),
    }
    receipt_path = output_dir / f"{prefix}.receipt.json"
    write_json_atomic(receipt_path, receipt, overwrite=False)
    return {"receipt_path": str(receipt_path), "figure_paths": [str(path) for path in generated], "receipt": receipt}


def plot_provider_epoch_behavior_cohort_tables(
    *,
    bouts_table: Any,
    fish_table: Any,
    manifest: Mapping[str, Any],
    output_dir: Path,
    source_tables: Mapping[str, Mapping[str, Any]] | None = None,
    prefix: str = "provider_epoch_behavior_cohort",
) -> dict[str, Any]:
    """Plot exact in-memory tables after validating the publication manifest."""

    data = validate_cohort_tables(
        bouts_table=bouts_table,
        fish_table=fish_table,
        manifest=manifest,
        source_tables=source_tables,
    )
    return render_provider_epoch_behavior_cohort(data, output_dir=output_dir, prefix=prefix)


def plot_provider_epoch_behavior_cohort_parquet(
    *,
    bouts_parquet: Path,
    fish_parquet: Path,
    manifest_path: Path,
    output_dir: Path,
    prefix: str = "provider_epoch_behavior_cohort",
) -> dict[str, Any]:
    """Plot the exact two Parquet parts named by one publication manifest."""

    data = _load_exact_source(
        bouts_parquet=bouts_parquet,
        fish_parquet=fish_parquet,
        manifest_path=manifest_path,
    )
    return render_provider_epoch_behavior_cohort(data, output_dir=output_dir, prefix=prefix)


def plot_provider_epoch_behavior_cohort(
    generation_root: Path,
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
    prefix: str = "provider_epoch_behavior_cohort",
) -> dict[str, Any]:
    """Plot one exact immutable cohort generation selected by its manifest."""

    data = _load_exact_source(generation_root=generation_root, manifest_path=manifest_path)
    return render_provider_epoch_behavior_cohort(data, output_dir=output_dir, prefix=prefix)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--generation-root", type=Path)
    source.add_argument("--bouts-parquet", type=Path)
    parser.add_argument("--fish-parquet", type=Path)
    parser.add_argument("--manifest", type=Path, help="Required with explicit Parquet paths; optional for a generation root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="provider_epoch_behavior_cohort")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.generation_root is not None:
        result = plot_provider_epoch_behavior_cohort(
            args.generation_root,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            prefix=args.prefix,
        )
    else:
        if args.fish_parquet is None or args.manifest is None:
            raise SystemExit("--bouts-parquet requires --fish-parquet and --manifest")
        result = plot_provider_epoch_behavior_cohort_parquet(
            bouts_parquet=args.bouts_parquet,
            fish_parquet=args.fish_parquet,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_EPOCH_LABELS",
    "NEUTRAL_EPOCH_COLORS",
    "PLOT_SCHEMA_ID",
    "PLOT_SCHEMA_VERSION",
    "PlotUnit",
    "ProviderEpochBehaviorPlotError",
    "ValidatedCohort",
    "plot_provider_epoch_behavior_cohort",
    "plot_provider_epoch_behavior_cohort_parquet",
    "plot_provider_epoch_behavior_cohort_tables",
    "render_provider_epoch_behavior_cohort",
    "validate_cohort_tables",
]
