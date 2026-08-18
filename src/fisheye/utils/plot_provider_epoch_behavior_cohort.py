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
from datetime import datetime
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
PLOT_SCHEMA_VERSION = 2
ANALYSIS_UNIT_DECISION_SCHEMA_ID = "palette.provider_epoch_analysis_unit_decision"
ANALYSIS_UNIT_DECISION_SCHEMA_VERSION = 1
DEFAULT_ANALYSIS_UNIT_MODE = "subject_id"
RECORDING_ANALYSIS_UNIT_MODE = "recording"
RECORDING_ANALYSIS_UNIT_POLICY_ID = (
    "operator_asserted_recording_unit_for_duplicate_acquisition_uuid_incident_v1"
)
EXPECTED_EPOCH_LABELS = ("pre_event", "training_event", "post_event")
EXPECTED_EPOCH_IDS = (0, 1, 2)
EXPECTED_EPOCH_IDENTITIES = tuple(zip(EXPECTED_EPOCH_IDS, EXPECTED_EPOCH_LABELS))
PLOT_METRICS = {
    "bout_rate_per_min": ("Bout rate", "Bout rate (events min$^{-1}$)"),
    "mean_speed_mm_s": ("Mean speed", "Mean speed (mm s$^{-1}$)"),
    "mean_bout_duration_s": ("Mean bout duration", "Mean bout duration (s)"),
}
DISTRIBUTION_METRICS = {
    "bout_duration_s": ("Bout duration", "Bout duration (s)"),
    "bout_path_length_mm": ("Bout path length", "Bout path length (mm)"),
    "mean_bout_speed_mm_s": ("Mean bout speed", "Mean bout speed (mm s$^{-1}$)"),
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
class AnalysisUnitContext:
    """Explicit grouping authority for grouped and distribution plots."""

    mode: str
    label: str
    decision: Mapping[str, Any] | None = None
    decision_path: str | None = None
    decision_sha256: str | None = None
    policy_id: str | None = None


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


@dataclass(frozen=True)
class BoutDistributionData:
    """Validated descriptive distributions derived from exact bout rows."""

    pooled_values_by_metric_epoch: Mapping[str, tuple[np.ndarray, ...]]
    analysis_unit_values_by_metric_epoch: Mapping[str, tuple[np.ndarray, ...]]
    metrics: Mapping[str, Mapping[str, Any]]

    @property
    def subject_values_by_metric_epoch(self) -> Mapping[str, tuple[np.ndarray, ...]]:
        """Compatibility alias for the original subject-mode helper name."""

        return self.analysis_unit_values_by_metric_epoch


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


def _require_sha256(value: object, *, label: str) -> str:
    text = _as_text(value, label=label)
    assert text is not None
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProviderEpochBehaviorPlotError(f"{label} must be a lowercase SHA-256.")
    return text


def _validate_analysis_unit_decision(
    data: ValidatedCohort,
    *,
    mode: str,
    decision_path: Path | None,
) -> AnalysisUnitContext:
    if mode == DEFAULT_ANALYSIS_UNIT_MODE:
        if decision_path is not None:
            raise ProviderEpochBehaviorPlotError(
                "analysis_unit_decision is only accepted in recording analysis-unit mode."
            )
        return AnalysisUnitContext(mode=mode, label="unique subject_id")
    if mode != RECORDING_ANALYSIS_UNIT_MODE:
        raise ProviderEpochBehaviorPlotError(
            "analysis_unit_mode must be 'subject_id' or 'recording'."
        )
    if decision_path is None:
        raise ProviderEpochBehaviorPlotError(
            "recording analysis-unit mode requires an immutable analysis-unit decision JSON."
        )

    path = decision_path.expanduser().resolve()
    if not path.is_file():
        raise ProviderEpochBehaviorPlotError(
            f"The recording analysis-unit decision JSON does not exist: {path}"
        )
    decision = _read_json(path)
    expected_fields = {
        "schema_id",
        "schema_version",
        "decision_id",
        "cohort_id",
        "source_manifest_sha256",
        "source_manifest_payload_sha256",
        "analysis_unit",
        "policy_id",
        "operator_assertion",
        "operator_identity",
        "reviewed_at_utc",
        "canonical_subject_identity_corrected",
        "reason_code",
        "recording_count",
        "duplicate_source_subject_id_count",
        "affected_recording_count",
        "collisions",
        "decision_payload_sha256",
    }
    optional_decision_fields = {"source_manifest_file_sha256"}
    decision_fields = set(decision)
    if decision_fields != expected_fields and decision_fields != (expected_fields | optional_decision_fields):
        raise ProviderEpochBehaviorPlotError(
            "The recording analysis-unit decision has an unexpected field set."
        )
    if decision["schema_id"] != "palette.cohort_analysis_unit_decision":
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision schema_id is invalid.")
    if decision["schema_version"] != 1:
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision schema_version is invalid.")
    for field_name in ("decision_id", "operator_identity"):
        _as_text(decision[field_name], label=field_name)
    if decision["cohort_id"] != data.cohort_id:
        raise ProviderEpochBehaviorPlotError(
            "The analysis-unit decision cohort_id does not match the publication cohort."
        )
    if decision["analysis_unit"] != "recording_id":
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision must select recording_id.")
    if decision["policy_id"] != "operator_asserted_distinct_animal_per_recording_v1":
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision policy_id is invalid.")
    if decision["operator_assertion"] != "each_recording_contains_a_distinct_animal":
        raise ProviderEpochBehaviorPlotError("The distinct-animal operator assertion is required.")
    if decision["canonical_subject_identity_corrected"] is not False:
        raise ProviderEpochBehaviorPlotError(
            "Recording mode must not claim canonical subject identity correction."
        )
    if decision["reason_code"] != "acquisition_subject_id_reuse":
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision reason_code is invalid.")
    reviewed_at = _as_text(decision["reviewed_at_utc"], label="reviewed_at_utc")
    assert reviewed_at is not None
    if not reviewed_at.endswith("Z"):
        raise ProviderEpochBehaviorPlotError("reviewed_at_utc must be a UTC timestamp ending in Z.")
    try:
        datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
    except ValueError as exc:
        raise ProviderEpochBehaviorPlotError("reviewed_at_utc must be an ISO-8601 UTC timestamp.") from exc

    input_manifest_path_value = data.manifest.get("input_manifest_path")
    input_manifest_sha256 = _require_sha256(
        data.manifest.get("input_manifest_sha256"),
        label="publication input_manifest_sha256",
    )
    if input_manifest_path_value is None:
        raise ProviderEpochBehaviorPlotError(
            "The publication manifest does not bind its frozen input manifest path."
        )
    input_manifest_path = Path(_as_text(input_manifest_path_value, label="input_manifest_path") or "").expanduser().resolve()
    if not input_manifest_path.is_file():
        raise ProviderEpochBehaviorPlotError(
            "The publication input manifest is missing."
        )
    input_manifest = _read_json(input_manifest_path)
    if canonical_json_sha256(input_manifest) != input_manifest_sha256:
        raise ProviderEpochBehaviorPlotError(
            "The canonical input manifest identity does not match publication input_manifest_sha256."
        )
    if "source_manifest_file_sha256" in decision:
        source_manifest_file_sha256 = _require_sha256(
            decision["source_manifest_file_sha256"],
            label="source_manifest_file_sha256",
        )
        if sha256_file(input_manifest_path) != source_manifest_file_sha256:
            raise ProviderEpochBehaviorPlotError(
                "source_manifest_file_sha256 does not match the raw input manifest file."
            )
    input_payload_digest = _require_sha256(
        input_manifest.get("manifest_payload_sha256"),
        label="input manifest manifest_payload_sha256",
    )
    unsigned_input_manifest = {
        key: value for key, value in input_manifest.items() if key != "manifest_payload_sha256"
    }
    if canonical_json_sha256(unsigned_input_manifest) != input_payload_digest:
        raise ProviderEpochBehaviorPlotError("The frozen input manifest payload digest is invalid.")
    if decision["source_manifest_sha256"] != input_manifest_sha256:
        raise ProviderEpochBehaviorPlotError(
            "source_manifest_sha256 does not match publication input_manifest_sha256."
        )
    if decision["source_manifest_payload_sha256"] != input_payload_digest:
        raise ProviderEpochBehaviorPlotError(
            "source_manifest_payload_sha256 does not match the frozen input manifest payload."
        )

    recording_count = data.n_recording_animal_sessions
    if decision["recording_count"] != recording_count:
        raise ProviderEpochBehaviorPlotError("The decision recording_count does not match the cohort.")
    entries = input_manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != recording_count:
        raise ProviderEpochBehaviorPlotError("The frozen input manifest entry count does not match the cohort.")
    source_subject_by_recording: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ProviderEpochBehaviorPlotError("The frozen input manifest contains an invalid entry.")
        recording_id = _as_text(entry.get("recording_id"), label="input recording_id")
        subject_id = _as_text(entry.get("subject_id"), label="input subject_id")
        assert recording_id is not None and subject_id is not None
        if recording_id in source_subject_by_recording:
            raise ProviderEpochBehaviorPlotError("The frozen input manifest repeats a recording_id.")
        source_subject_by_recording[recording_id] = subject_id
    cohort_recordings = {unit.recording_id for unit in data.units}
    if set(source_subject_by_recording) != cohort_recordings:
        raise ProviderEpochBehaviorPlotError(
            "The frozen input manifest recording IDs do not match the published cohort."
        )
    expected_collisions = {
        subject_id: sorted(
            recording_id
            for recording_id, source_subject_id in source_subject_by_recording.items()
            if source_subject_id == subject_id
        )
        for subject_id in sorted(set(source_subject_by_recording.values()))
    }
    expected_collisions = {
        subject_id: recording_ids
        for subject_id, recording_ids in expected_collisions.items()
        if len(recording_ids) >= 2
    }
    if decision["duplicate_source_subject_id_count"] != len(expected_collisions):
        raise ProviderEpochBehaviorPlotError("The duplicate source-subject count does not match the cohort.")
    affected_recordings = sum(len(recording_ids) for recording_ids in expected_collisions.values())
    if decision["affected_recording_count"] != affected_recordings:
        raise ProviderEpochBehaviorPlotError("The affected recording count does not match the collision groups.")
    collisions = decision["collisions"]
    if not isinstance(collisions, list):
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision collisions field must be a list.")
    observed_collisions: dict[str, list[str]] = {}
    for collision in collisions:
        if not isinstance(collision, Mapping) or set(collision) != {"source_subject_id", "recording_ids"}:
            raise ProviderEpochBehaviorPlotError("Each collision must contain source_subject_id and recording_ids.")
        source_subject_id = _as_text(collision["source_subject_id"], label="collision source_subject_id")
        recording_ids = collision["recording_ids"]
        assert source_subject_id is not None
        if not isinstance(recording_ids, list) or len(recording_ids) < 2:
            raise ProviderEpochBehaviorPlotError("Each collision must contain at least two recordings.")
        if any(not isinstance(recording_id, str) or not recording_id.strip() for recording_id in recording_ids):
            raise ProviderEpochBehaviorPlotError("Collision recording_ids must be non-empty strings.")
        if len(set(recording_ids)) != len(recording_ids):
            raise ProviderEpochBehaviorPlotError("Collision recording_ids must not contain duplicates.")
        observed_collisions[source_subject_id] = sorted(recording_ids)
    if observed_collisions != expected_collisions:
        raise ProviderEpochBehaviorPlotError("The collision list is not the exact collision inventory for the cohort.")

    if decision["duplicate_source_subject_id_count"] != 8 or decision["affected_recording_count"] != 16:
        raise ProviderEpochBehaviorPlotError(
            "This incident decision must document 8 reused source subject IDs affecting 16 recordings."
        )
    decision_payload_sha256 = _require_sha256(
        decision.get("decision_payload_sha256"),
        label="decision_payload_sha256",
    )
    unsigned_decision = {
        key: value for key, value in decision.items() if key != "decision_payload_sha256"
    }
    if canonical_json_sha256(unsigned_decision) != decision_payload_sha256:
        raise ProviderEpochBehaviorPlotError("The analysis-unit decision payload digest is invalid.")
    return AnalysisUnitContext(
        mode=mode,
        label="recording×animal unit",
        decision=decision,
        decision_path=str(path),
        decision_sha256=sha256_file(path),
        policy_id=str(decision["policy_id"]),
    )


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


def _subject_level_matrix(
    data: ValidatedCohort,
    metric: str,
    *,
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> np.ndarray:
    """Return one row per selected analysis unit for the existing summary plots."""

    if analysis_unit_mode == RECORDING_ANALYSIS_UNIT_MODE:
        return np.asarray(
            [
                [
                    np.nan if value is None else float(value)
                    for value in unit.values_by_metric[metric]
                ]
                for unit in data.units
            ],
            dtype=np.float64,
        )
    if analysis_unit_mode != DEFAULT_ANALYSIS_UNIT_MODE:
        raise ProviderEpochBehaviorPlotError("Unknown analysis-unit mode.")

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


def _positive_observation(value: object) -> tuple[float | None, str | None]:
    """Return one strictly positive finite metric value or an audit reason."""

    if value is None:
        return None, "missing"
    if isinstance(value, np.generic):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "non_numeric"
    if not np.isfinite(number):
        return None, "nonfinite"
    if number < 0:
        return None, "negative"
    if number == 0:
        return None, "zero_or_nonpositive"
    return number, None


def _distribution_observation(row: Mapping[str, Any], metric: str) -> tuple[float | None, str | None]:
    if metric in {"bout_duration_s", "bout_path_length_mm"}:
        return _positive_observation(row.get(metric))
    if metric != "mean_bout_speed_mm_s":
        raise ProviderEpochBehaviorPlotError(f"Unknown bout distribution metric: {metric}")

    duration, duration_reason = _positive_observation(row.get("bout_duration_s"))
    if duration is None:
        return None, f"duration_{duration_reason}"
    path_length, path_reason = _positive_observation(row.get("bout_path_length_mm"))
    if path_length is None:
        return None, f"path_length_{path_reason}"
    return path_length / duration, None


def _empty_epoch_arrays() -> list[list[float]]:
    return [[] for _ in EXPECTED_EPOCH_IDS]


def _distribution_data(
    data: ValidatedCohort,
    *,
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> BoutDistributionData:
    """Compute pooled and subject-balanced values without clipping inputs.

    Duration and path length are considered plot-valid only when strictly
    positive and finite.  This makes the positive-domain contract explicit for
    the logarithmic pooled ECDF.  Zero, negative, missing, nonnumeric, and
    nonfinite values are retained in the audit counts and omitted from the
    corresponding metric; they are never replaced or clipped.  Mean bout
    speed additionally requires positive duration and positive path length.
    """

    if analysis_unit_mode not in {DEFAULT_ANALYSIS_UNIT_MODE, RECORDING_ANALYSIS_UNIT_MODE}:
        raise ProviderEpochBehaviorPlotError("Unknown analysis-unit mode.")
    rows = _table_rows(data.bouts_table)
    pooled_values: dict[str, list[list[float]]] = {
        metric: _empty_epoch_arrays() for metric in DISTRIBUTION_METRICS
    }
    session_values: dict[str, dict[tuple[str, int, int], list[float]]] = {
        metric: {} for metric in DISTRIBUTION_METRICS
    }
    metric_audit: dict[str, dict[str, Any]] = {}
    for metric in DISTRIBUTION_METRICS:
        metric_audit[metric] = {
            "total_bout_count_by_epoch": [0, 0, 0],
            "valid_bout_count_by_epoch": [0, 0, 0],
            "dropped_bout_count_by_epoch": [0, 0, 0],
            "dropped_reason_counts_by_epoch": [dict() for _ in EXPECTED_EPOCH_IDS],
        }

    for row in rows:
        epoch_id = _as_int(row.get("epoch_id"), label="epoch_id")
        recording_id = _as_text(row.get("recording_id"), label="recording_id")
        assert recording_id is not None
        track_id = _as_int(row.get("track_id"), label="track_id")
        subject_key = (recording_id, track_id, epoch_id)
        for metric in DISTRIBUTION_METRICS:
            audit = metric_audit[metric]
            audit["total_bout_count_by_epoch"][epoch_id] += 1
            value, reason = _distribution_observation(row, metric)
            if value is None:
                audit["dropped_bout_count_by_epoch"][epoch_id] += 1
                reasons = audit["dropped_reason_counts_by_epoch"][epoch_id]
                assert isinstance(reasons, dict)
                assert reason is not None
                reasons[reason] = int(reasons.get(reason, 0)) + 1
                continue
            pooled_values[metric][epoch_id].append(value)
            audit["valid_bout_count_by_epoch"][epoch_id] += 1
            session_values[metric].setdefault(subject_key, []).append(value)

    subject_by_unit = {
        (unit.recording_id, unit.track_id): unit.subject_id for unit in data.units
    }
    subject_values: dict[str, tuple[np.ndarray, ...]] = {}
    for metric in DISTRIBUTION_METRICS:
        medians_by_analysis_unit_epoch: dict[str, list[list[float]]] = {}
        valid_session_counts = [0, 0, 0]
        for (recording_id, track_id, epoch_id), values in sorted(session_values[metric].items()):
            subject_id = subject_by_unit[(recording_id, track_id)]
            if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE and subject_id is None:
                raise ProviderEpochBehaviorPlotError(
                    "Subject-balanced distributions require subject_id for every session."
                )
            median = float(np.median(np.asarray(values, dtype=np.float64)))
            analysis_unit_id = (
                subject_id
                if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE
                else f"{recording_id}::track-{track_id}"
            )
            assert analysis_unit_id is not None
            medians_by_analysis_unit_epoch.setdefault(analysis_unit_id, _empty_epoch_arrays())[epoch_id].append(median)
            valid_session_counts[epoch_id] += 1
        subject_arrays: list[np.ndarray] = []
        valid_analysis_unit_counts: list[int] = []
        for epoch_id in EXPECTED_EPOCH_IDS:
            values = [
                float(np.mean(medians_by_analysis_unit_epoch[analysis_unit_id][epoch_id]))
                for analysis_unit_id in sorted(medians_by_analysis_unit_epoch)
                if medians_by_analysis_unit_epoch[analysis_unit_id][epoch_id]
            ]
            subject_arrays.append(np.asarray(values, dtype=np.float64))
            valid_analysis_unit_counts.append(len(values))
        subject_values[metric] = tuple(subject_arrays)
        metric_audit[metric]["valid_analysis_unit_median_count_by_epoch"] = valid_session_counts
        metric_audit[metric]["valid_analysis_unit_count_by_epoch"] = valid_analysis_unit_counts
        # Retain the original subject-mode receipt vocabulary for valid
        # subject-grouped cohorts; recording mode uses the explicit
        # analysis-unit fields above instead.
        metric_audit[metric]["valid_session_median_count_by_epoch"] = valid_session_counts
        metric_audit[metric]["valid_subject_count_by_epoch"] = (
            valid_analysis_unit_counts
            if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE
            else None
        )
        metric_audit[metric]["dropped_session_median_count_by_epoch"] = [
            data.n_recording_animal_sessions - count for count in valid_session_counts
        ]

    return BoutDistributionData(
        pooled_values_by_metric_epoch={
            metric: tuple(np.asarray(values, dtype=np.float64) for values in epoch_values)
            for metric, epoch_values in pooled_values.items()
        },
        analysis_unit_values_by_metric_epoch=subject_values,
        metrics=metric_audit,
    )


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
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    matrix = _subject_level_matrix(data, metric, analysis_unit_mode=analysis_unit_mode)
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
        "finite_subject_counts_by_epoch": counts
        if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE
        else None,
        "finite_analysis_unit_counts_by_epoch": counts,
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
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.0 * len(metrics), 5.2), squeeze=False, dpi=160)
    estimate_label = "unique subjects" if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE else "recording×animal units"
    stats: dict[str, Any] = {}
    for axis, metric in zip(axes[0], metrics):
        # Render one row per selected analysis unit. Subject mode first
        # equal-weights repeated sessions within subject.
        matrix = _subject_level_matrix(data, metric, analysis_unit_mode=analysis_unit_mode)
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
            "finite_subject_counts_by_epoch": counts
            if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE
            else None,
            "finite_analysis_unit_counts_by_epoch": counts,
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
    fig.suptitle(f"Linear bout metrics across {estimate_label}", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> tuple[list[Path], dict[str, Any]]:
    generated: list[Path] = []
    stats: dict[str, Any] | None = None
    for extension in ("png", "svg"):
        path = output_dir / f"{prefix}.grouped_{metric}.{extension}"
        # The renderer closes its figure; recomputing stats is deterministic.
        estimate_label = "unique subjects" if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE else "recording×animal units"
        stats = _render_grouped_metric(data, metric, path, plt=plt, title=f"{PLOT_METRICS[metric][0]} across {estimate_label}", ylabel=PLOT_METRICS[metric][1], analysis_unit_mode=analysis_unit_mode)
        generated.append(path)
    assert stats is not None
    return generated, stats


def _save_speed_duration_formats(
    data: ValidatedCohort,
    metrics: Sequence[str],
    output_dir: Path,
    prefix: str,
    plt: Any,
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> tuple[list[Path], dict[str, Any]]:
    generated: list[Path] = []
    stats: dict[str, Any] | None = None
    for extension in ("png", "svg"):
        path = output_dir / f"{prefix}.grouped_speed_duration.{extension}"
        stats = _render_grouped_speed_duration(data, metrics, path, plt=plt, analysis_unit_mode=analysis_unit_mode)
        generated.append(path)
    assert stats is not None
    return generated, stats


def _render_pooled_bout_ecdf(
    distributions: BoutDistributionData,
    path: Path,
    *,
    plt: Any,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(DISTRIBUTION_METRICS),
        figsize=(7.0 * len(DISTRIBUTION_METRICS), 5.6),
        squeeze=False,
        dpi=160,
    )
    for axis, metric in zip(axes[0], DISTRIBUTION_METRICS):
        for epoch_id, epoch_label in enumerate(EXPECTED_EPOCH_LABELS):
            values = np.sort(distributions.pooled_values_by_metric_epoch[metric][epoch_id])
            if values.size == 0:
                continue
            axis.step(
                values,
                np.arange(1, values.size + 1, dtype=np.float64) / values.size,
                where="post",
                color=NEUTRAL_EPOCH_COLORS[epoch_label],
                linewidth=2.0,
                label=f"{epoch_label.replace('_', ' ')} (n={values.size:,})",
            )
        axis.set_xscale("log")
        axis.set_xlabel(f"{DISTRIBUTION_METRICS[metric][1]}; logarithmic x-axis")
        axis.set_ylabel("Empirical cumulative probability")
        axis.set_title(DISTRIBUTION_METRICS[metric][0])
        axis.set_ylim(0.0, 1.02)
        axis.grid(color="#D1D5DB", alpha=0.65, linewidth=0.7)
        axis.set_axisbelow(True)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Pooled bout distributions: descriptive ECDFs", y=0.99)
    fig.text(
        0.5,
        0.01,
        "Pooled bouts are descriptive only; bouts within an animal are not independent. No inferential uncertainty is shown.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, format=path.suffix.lstrip("."), dpi=160, metadata={"Date": None})
    plt.close(fig)


def _render_subject_balanced_distributions(
    distributions: BoutDistributionData,
    path: Path,
    *,
    plt: Any,
    title: str = "Subject-balanced bout distributions: one value per unique subject and epoch",
    note: str = "Each session contributes its median bout value; repeated sessions are averaged with equal weight within subject. Linear axes; no clipping.",
    unit_label: str = "subjects",
) -> None:
    fig, axes = plt.subplots(
        1,
        len(DISTRIBUTION_METRICS),
        figsize=(7.0 * len(DISTRIBUTION_METRICS), 5.8),
        squeeze=False,
        dpi=160,
    )
    for axis, metric in zip(axes[0], DISTRIBUTION_METRICS):
        values_by_epoch = distributions.analysis_unit_values_by_metric_epoch[metric]
        finite_values = np.concatenate([values for values in values_by_epoch if values.size]) if any(
            values.size for values in values_by_epoch
        ) else np.asarray([], dtype=np.float64)
        low = float(np.min(finite_values)) if finite_values.size else 0.0
        high = float(np.max(finite_values)) if finite_values.size else 1.0
        for epoch_id, values in enumerate(values_by_epoch):
            if values.size:
                box = axis.boxplot(
                    [values],
                    positions=[epoch_id],
                    widths=0.34,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#111827", "linewidth": 1.6},
                    whiskerprops={"color": "#4B5563", "linewidth": 1.0},
                    capprops={"color": "#4B5563", "linewidth": 1.0},
                    boxprops={"color": "#4B5563", "linewidth": 1.0},
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(NEUTRAL_EPOCH_COLORS[EXPECTED_EPOCH_LABELS[epoch_id]])
                    patch.set_alpha(0.55)
                jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
                axis.scatter(
                    np.full(values.size, epoch_id, dtype=np.float64) + jitter,
                    values,
                    s=15,
                    color="#1F2937",
                    alpha=0.72,
                    linewidths=0.25,
                    edgecolors="white",
                    zorder=3,
                )
        axis.set_xticks(np.arange(len(EXPECTED_EPOCH_LABELS)))
        axis.set_xticklabels(
            [
                f"Pre-event\nn={values_by_epoch[0].size:,} {unit_label}",
                f"Training event\nn={values_by_epoch[1].size:,} {unit_label}",
                f"Post-event\nn={values_by_epoch[2].size:,} {unit_label}",
            ]
        )
        axis.set_xlabel("Stimulus epoch")
        axis.set_ylabel(DISTRIBUTION_METRICS[metric][1])
        axis.set_title(DISTRIBUTION_METRICS[metric][0])
        axis.grid(axis="y", color="#D1D5DB", alpha=0.65, linewidth=0.7)
        axis.set_axisbelow(True)
        _draw_epoch_band(axis, y_min=low, y_max=high)
    fig.suptitle(title, y=0.99)
    fig.text(
        0.5,
        0.01,
        note,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, format=path.suffix.lstrip("."), dpi=160, metadata={"Date": None})
    plt.close(fig)


def _save_distribution_formats(
    distributions: BoutDistributionData,
    output_dir: Path,
    prefix: str,
    plt: Any,
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
) -> list[Path]:
    generated: list[Path] = []
    for extension in ("png", "svg"):
        ecdf_path = output_dir / f"{prefix}.pooled_bout_ecdf.{extension}"
        subject_path = output_dir / f"{prefix}.subject_balanced_bout_distributions.{extension}"
        _render_pooled_bout_ecdf(distributions, ecdf_path, plt=plt)
        if analysis_unit_mode == RECORDING_ANALYSIS_UNIT_MODE:
            _render_subject_balanced_distributions(
                distributions,
                subject_path,
                plt=plt,
                title="Recording-balanced bout distributions: one value per recording×animal unit and epoch",
                note="Each recording×animal unit contributes one median bout value per epoch. Linear axes; no clipping.",
                unit_label="recording×animal units",
            )
        else:
            _render_subject_balanced_distributions(distributions, subject_path, plt=plt)
        generated.extend((ecdf_path, subject_path))
    return generated


def render_provider_epoch_behavior_cohort(
    data: ValidatedCohort,
    *,
    output_dir: Path,
    prefix: str = "provider_epoch_behavior_cohort",
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
    analysis_unit_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Render deterministic figures and a receipt from a validated cohort."""

    if not isinstance(prefix, str) or not _PREFIX_RE.fullmatch(prefix):
        raise ProviderEpochBehaviorPlotError("prefix must be one safe portable filename component.")
    analysis_units = _validate_analysis_unit_decision(
        data,
        mode=analysis_unit_mode,
        decision_path=analysis_unit_decision_path,
    )
    if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE and any(unit.subject_id is None for unit in data.units):
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
        output_dir / f"{prefix}.pooled_bout_ecdf.png",
        output_dir / f"{prefix}.pooled_bout_ecdf.svg",
        output_dir / f"{prefix}.subject_balanced_bout_distributions.png",
        output_dir / f"{prefix}.subject_balanced_bout_distributions.svg",
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
    distributions = _distribution_data(data, analysis_unit_mode=analysis_unit_mode)
    generated: list[Path] = []
    plot_stats: dict[str, Any] = {}
    generated.extend(_save_individual_formats(data, output_dir, prefix, plt))
    grouped_paths, bout_stats = _save_grouped_metric_formats(
        data,
        "bout_rate_per_min",
        output_dir,
        prefix,
        plt,
        analysis_unit_mode=analysis_unit_mode,
    )
    generated.extend(grouped_paths)
    plot_stats["bout_rate_per_min"] = bout_stats
    if metric_names:
        metric_paths, metric_stats = _save_speed_duration_formats(
            data,
            metric_names,
            output_dir,
            prefix,
            plt,
            analysis_unit_mode=analysis_unit_mode,
        )
        generated.extend(metric_paths)
        plot_stats.update(metric_stats)
    generated.extend(
        _save_distribution_formats(
            distributions,
            output_dir,
            prefix,
            plt,
            analysis_unit_mode=analysis_unit_mode,
        )
    )

    source_manifest_sha256 = _canonical_sha256(data.manifest)
    session_counts_by_subject: dict[str, int] = {}
    for unit in data.units:
        if unit.subject_id is not None:
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
        "analysis_unit_mode": analysis_units.mode,
        "analysis_unit_label": analysis_units.label,
        "analysis_unit_count": data.n_recording_animal_sessions,
        "source_subject_id_count": data.n_subjects,
        "grouping_unit": "subject_id" if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE else "recording_id",
        "repeated_session_aggregation": (
            "arithmetic_mean_within_subject_id_epoch"
            if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE
            else "not_applicable_recording_id_is_analysis_unit"
        ),
        "session_weighting": (
            "equal"
            if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE
            else "not_applicable_one_recording_unit_per_recording"
        ),
        "session_counts_by_subject": dict(sorted(session_counts_by_subject.items())),
        "grouped_estimate_level": (
            "unique_subject_id" if analysis_unit_mode == DEFAULT_ANALYSIS_UNIT_MODE else "recording_id"
        ),
        "uncertainty": "standard_error_of_mean",
        "epoch_colors": dict(NEUTRAL_EPOCH_COLORS),
        "metrics": plot_stats,
        "distribution_metrics": dict(distributions.metrics),
        "analysis_unit_decision": (
            None
            if analysis_units.decision is None
            else {
                "path": analysis_units.decision_path,
                "sha256": analysis_units.decision_sha256,
                "payload": dict(analysis_units.decision),
            }
        ),
        "analysis_unit_policy_id": analysis_units.policy_id,
        "canonical_subject_identity_corrected": (
            None
            if analysis_units.decision is None
            else analysis_units.decision["canonical_subject_identity_corrected"]
        ),
        "duplicate_source_subject_id_count": (
            None
            if analysis_units.decision is None
            else analysis_units.decision["duplicate_source_subject_id_count"]
        ),
        "affected_recording_count": (
            None
            if analysis_units.decision is None
            else analysis_units.decision["affected_recording_count"]
        ),
        "distribution_figures": {
            "pooled_bout_ecdf": {
                "files": [
                    f"{prefix}.pooled_bout_ecdf.png",
                    f"{prefix}.pooled_bout_ecdf.svg",
                ],
                "semantics": "pooled_bouts_descriptive_only_nonindependent_within_subject",
                "uncertainty": "none_inferential_uncertainty_not_shown",
                "x_scale": "log10",
                "x_scale_label": "logarithmic x-axis",
                "clipping": "none",
            },
            "subject_balanced_bout_distributions": {
                "files": [
                    f"{prefix}.subject_balanced_bout_distributions.png",
                    f"{prefix}.subject_balanced_bout_distributions.svg",
                ],
                "semantics": "one_session_median_then_equal_weighted_subject_value",
                "uncertainty": "none_inferential_uncertainty_not_shown",
                "y_scale": "linear",
                "clipping": "none",
            },
        },
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
            "Pooled bout ECDFs retain every valid positive finite bout value and are descriptive only; bouts within animals are not independent and no inferential uncertainty is shown.",
            "Subject-balanced distributions take the median across valid bouts within each recording/session×epoch, then average repeated session medians with equal weight within subject×epoch.",
            "Duration and path-length values that are zero, negative, missing, nonnumeric, or nonfinite are counted by reason and omitted from the positive-domain distributions; no values are clipped or substituted.",
            "Mean bout speed is bout_path_length_mm / bout_duration_s only when both inputs are finite and strictly positive.",
        ],
    }
    if analysis_unit_mode == RECORDING_ANALYSIS_UNIT_MODE:
        unsigned_receipt["scientific_notes"].extend(
            [
                "Recording analysis-unit mode is an operator-asserted grouping workaround for reused acquisition subject IDs; it is not a canonical subject identity correction.",
                "The decision asserts that each recording contains a distinct animal; MetaZebrobot remains responsible for minting and verifying canonical replacement biological UUIDs.",
                "Recording-balanced distributions contain one median per recording×animal unit×epoch; no repeated-session averaging is performed.",
            ]
        )
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
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
    analysis_unit_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Plot exact in-memory tables after validating the publication manifest."""

    data = validate_cohort_tables(
        bouts_table=bouts_table,
        fish_table=fish_table,
        manifest=manifest,
        source_tables=source_tables,
    )
    return render_provider_epoch_behavior_cohort(
        data,
        output_dir=output_dir,
        prefix=prefix,
        analysis_unit_mode=analysis_unit_mode,
        analysis_unit_decision_path=analysis_unit_decision_path,
    )


def plot_provider_epoch_behavior_cohort_parquet(
    *,
    bouts_parquet: Path,
    fish_parquet: Path,
    manifest_path: Path,
    output_dir: Path,
    prefix: str = "provider_epoch_behavior_cohort",
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
    analysis_unit_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Plot the exact two Parquet parts named by one publication manifest."""

    data = _load_exact_source(
        bouts_parquet=bouts_parquet,
        fish_parquet=fish_parquet,
        manifest_path=manifest_path,
    )
    return render_provider_epoch_behavior_cohort(
        data,
        output_dir=output_dir,
        prefix=prefix,
        analysis_unit_mode=analysis_unit_mode,
        analysis_unit_decision_path=analysis_unit_decision_path,
    )


def plot_provider_epoch_behavior_cohort(
    generation_root: Path,
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
    prefix: str = "provider_epoch_behavior_cohort",
    analysis_unit_mode: str = DEFAULT_ANALYSIS_UNIT_MODE,
    analysis_unit_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Plot one exact immutable cohort generation selected by its manifest."""

    data = _load_exact_source(generation_root=generation_root, manifest_path=manifest_path)
    return render_provider_epoch_behavior_cohort(
        data,
        output_dir=output_dir,
        prefix=prefix,
        analysis_unit_mode=analysis_unit_mode,
        analysis_unit_decision_path=analysis_unit_decision_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--generation-root", type=Path)
    source.add_argument("--bouts-parquet", type=Path)
    parser.add_argument("--fish-parquet", type=Path)
    parser.add_argument("--manifest", type=Path, help="Required with explicit Parquet paths; optional for a generation root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="provider_epoch_behavior_cohort")
    parser.add_argument(
        "--analysis-unit-mode",
        choices=(DEFAULT_ANALYSIS_UNIT_MODE, RECORDING_ANALYSIS_UNIT_MODE),
        default=DEFAULT_ANALYSIS_UNIT_MODE,
        help="Grouping authority for grouped plots; recording mode requires --analysis-unit-decision.",
    )
    parser.add_argument(
        "--analysis-unit-decision",
        type=Path,
        help="Immutable decision JSON required for recording analysis-unit mode.",
    )
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
            analysis_unit_mode=args.analysis_unit_mode,
            analysis_unit_decision_path=args.analysis_unit_decision,
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
            analysis_unit_mode=args.analysis_unit_mode,
            analysis_unit_decision_path=args.analysis_unit_decision,
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
