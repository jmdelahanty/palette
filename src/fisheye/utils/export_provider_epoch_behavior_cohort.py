"""Export explicit selector-ineligible provider epoch summaries as a cohort.

This exporter is intentionally a separate, talk-support surface.  It accepts
only an operator-authored manifest which names one exact summary run per
recording and track.  It never resolves a selector, changes a Zarr archive, or
infers subject identity from a track number.  The resulting Parquet generation
is an immutable derived analysis product and is itself selector-ineligible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analytics_exports.arrow_contract_core import (
    ArrowTableContract,
    contract_envelope,
    field,
)
from fisheye.analytics_exports.derived_publication import (
    publish_derived_table_generation,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.shared.zarr_io import open_zarr_root


INPUT_SCHEMA_ID = "palette.provider_epoch_behavior_cohort_input"
INPUT_SCHEMA_VERSION = 1
EXPORT_SCHEMA_ID = "palette.provider_epoch_behavior_cohort"
EXPORT_SCHEMA_VERSION = 1
ARROW_ENVELOPE_SCHEMA_ID = "palette.provider_epoch_behavior_cohort.arrow_contracts"
ARROW_ENVELOPE_SCHEMA_VERSION = 1
SUMMARY_PARENT_PATH = "analysis/stimulus_epoch_behavior_summary_runs"
TABLE_BOUTS = "provider_epoch_bout_events"
TABLE_FISH = "provider_epoch_behavior_summary"
TABLE_NAMES = (TABLE_BOUTS, TABLE_FISH)
METRIC_DISPOSITIONS = ("linear_only", "full")
_FISH_ANGULAR_FIELDS = frozenset(
    {
        "bout_heading_sample_count",
        "mean_bout_net_heading_change_deg",
        "median_bout_net_heading_change_deg",
        "mean_abs_bout_net_heading_change_deg",
        "median_abs_bout_net_heading_change_deg",
        "mean_bout_heading_path_deg",
        "median_bout_heading_path_deg",
    }
)
_BOUT_ANGULAR_FIELDS = frozenset(
    {
        "bout_net_heading_change_deg",
        "abs_bout_net_heading_change_deg",
        "bout_heading_path_deg",
    }
)
class ProviderEpochBehaviorCohortError(ValueError):
    """Raised when an explicit cohort cannot be exported safely."""


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderEpochBehaviorCohortError(f"{label} must be one non-empty string.")
    return value


def _safe_run_name(value: object) -> str:
    run = _string(value, label="summary_run")
    if run in {".", "..", "latest", "latest_complete"} or "/" in run:
        raise ProviderEpochBehaviorCohortError(
            "summary_run must name one exact non-selector run, not a path or selector."
        )
    return run


def _sha256(value: object, *, label: str) -> str:
    digest = _string(value, label=label)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ProviderEpochBehaviorCohortError(f"{label} must be a lowercase SHA-256.")
    return digest


def _canonical_dtype(dtype: np.dtype[Any]) -> list[list[Any]]:
    return [list(item) for item in dtype.descr]


def _array_payload_sha256(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical_json_sha256(_canonical_dtype(data.dtype)).encode("ascii"))
    digest.update(str(tuple(int(v) for v in data.shape)).encode("ascii"))
    digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def _decode(value: object) -> object:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).rstrip(b"\x00").decode("utf-8", errors="strict")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _record(row: np.void) -> dict[str, object]:
    return {name: _decode(row[name]) for name in row.dtype.names or ()}


_FISH_SOURCE_FIELDS = (
    ("track_id", "int64"),
    ("window_id", "int32"),
    ("window_index", "int32"),
    ("window_label", "|S96"),
    ("start_frame", "int64"),
    ("end_frame", "int64"),
    ("start_time_s", "float64"),
    ("end_time_s", "float64"),
    ("duration_s", "float64"),
    ("total_span_frames", "int64"),
    ("provider_sample_count", "int64"),
    ("valid_tracked_frame_count", "int64"),
    ("missing_frame_count", "int64"),
    ("tracking_dropout_fraction", "float64"),
    ("valid_tracked_duration_s", "float64"),
    ("motion_valid_sample_count", "int64"),
    ("speed_sample_count", "int64"),
    ("mean_speed_mm_s", "float64"),
    ("median_speed_mm_s", "float64"),
    ("p05_speed_mm_s", "float64"),
    ("p95_speed_mm_s", "float64"),
    ("max_speed_mm_s", "float64"),
    ("total_path_mm", "float64"),
    ("bout_count", "int64"),
    ("bout_rate_per_min", "float64"),
    ("median_bout_duration_s", "float64"),
    ("mean_bout_duration_s", "float64"),
    ("median_bout_path_length_mm", "float64"),
    ("mean_bout_path_length_mm", "float64"),
    ("bout_heading_sample_count", "int64"),
    ("mean_bout_net_heading_change_deg", "float64"),
    ("median_bout_net_heading_change_deg", "float64"),
    ("mean_abs_bout_net_heading_change_deg", "float64"),
    ("median_abs_bout_net_heading_change_deg", "float64"),
    ("mean_bout_heading_path_deg", "float64"),
    ("median_bout_heading_path_deg", "float64"),
    ("inter_bout_interval_count", "int64"),
    ("mean_inter_bout_interval_s", "float64"),
    ("median_inter_bout_interval_s", "float64"),
    ("p05_inter_bout_interval_s", "float64"),
    ("p95_inter_bout_interval_s", "float64"),
    ("inter_bout_interval_rate_per_min", "float64"),
    ("rate_denominator", "|S64"),
    ("motion_validity_rule", "|S64"),
)
_BOUT_SOURCE_FIELDS = (
    ("track_id", "int64"),
    ("window_id", "int32"),
    ("window_index", "int32"),
    ("window_label", "|S96"),
    ("start_frame", "int64"),
    ("end_frame", "int64"),
    ("start_time_s", "float64"),
    ("end_time_s", "float64"),
    ("duration_s", "float64"),
    ("bout_source_row", "int64"),
    ("bout_id", "int64"),
    ("bout_event_frame", "int64"),
    ("bout_event_time_s", "float64"),
    ("bout_start_frame", "int64"),
    ("bout_end_frame", "int64"),
    ("bout_start_time_s", "float64"),
    ("bout_end_time_s", "float64"),
    ("bout_duration_s", "float64"),
    ("bout_path_length_mm", "float64"),
    ("bout_net_heading_change_deg", "float64"),
    ("abs_bout_net_heading_change_deg", "float64"),
    ("bout_heading_path_deg", "float64"),
)


def _source_dtype(spec: Sequence[tuple[str, str]]) -> np.dtype[Any]:
    return np.dtype(list(spec))


def _source_schema_digest(spec: Sequence[tuple[str, str]]) -> str:
    return canonical_json_sha256([[name, dtype] for name, dtype in spec])


def _contract_fields(
    disposition: str = "full",
) -> tuple[dict[str, str], dict[str, str]]:
    if disposition not in METRIC_DISPOSITIONS:
        raise ProviderEpochBehaviorCohortError(f"Unknown metric disposition: {disposition}")
    common = {
        "cohort_id": "string",
        "recording_id": "string",
        "analysis_zarr": "string",
        "subject_id": "string",
        "track_id": "int64",
        "summary_run": "string",
        "source_summary_sha256": "string",
        "source_refs_sha256": "string",
        "epoch_selection_sha256": "string",
        "provider_motion_manifest_sha256": "string",
        "provider_motion_verification_digest": "string",
        "swim_bout_lineage_sha256": "string",
        "swim_bout_frame_axis_sha256": "string",
    }
    bout = {
        **common,
        "epoch_id": "int32",
        "epoch_index": "int32",
        "epoch_label": "string",
        "epoch_start_frame": "int64",
        "epoch_end_frame": "int64",
        "epoch_start_time_s": "float64",
        "epoch_end_time_s": "float64",
        "epoch_duration_s": "float64",
        "bout_source_row": "int64",
        "bout_id": "int64",
        "bout_event_frame": "int64",
        "bout_event_time_s": "float64",
        "bout_start_frame": "int64",
        "bout_end_frame": "int64",
        "bout_start_time_s": "float64",
        "bout_end_time_s": "float64",
        "bout_duration_s": "float64",
        "bout_path_length_mm": "float64",
        "bout_net_heading_change_deg": "float64",
        "abs_bout_net_heading_change_deg": "float64",
        "bout_heading_path_deg": "float64",
    }
    fish = {
        **common,
        "epoch_id": "int32",
        "epoch_index": "int32",
        "epoch_label": "string",
    }
    for name, arrow_type in _FISH_SOURCE_FIELDS:
        if name in {"track_id", "window_id", "window_index", "window_label"}:
            continue
        if disposition == "linear_only" and name in _FISH_ANGULAR_FIELDS:
            continue
        output_name = {
            "window_id": "epoch_id",
            "window_index": "epoch_index",
            "window_label": "epoch_label",
        }.get(name, name)
        fish[output_name] = {
            "|S64": "string",
            "|S96": "string",
        }.get(arrow_type, arrow_type)
    if disposition == "linear_only":
        for name in _BOUT_ANGULAR_FIELDS:
            bout.pop(name, None)
    return bout, fish


def _contract(
    table_name: str,
    fields: Mapping[str, str],
    *,
    nullable: Sequence[str] = ("subject_id",),
    primary_key: Sequence[str],
) -> ArrowTableContract:
    return ArrowTableContract(
        table_name=table_name,
        fields=tuple(
            field(name, arrow_type, nullable=name in set(nullable))
            for name, arrow_type in fields.items()
        ),
        primary_key=tuple(primary_key),
    )


def table_contracts_for_disposition(
    disposition: str,
) -> dict[str, ArrowTableContract]:
    bout_fields, fish_fields = _contract_fields(disposition)
    return {
        TABLE_BOUTS: _contract(
            TABLE_BOUTS,
            bout_fields,
            primary_key=("recording_id", "track_id", "epoch_id", "bout_source_row"),
        ),
        TABLE_FISH: _contract(
            TABLE_FISH,
            fish_fields,
            primary_key=("recording_id", "track_id", "epoch_id"),
        ),
    }


TABLE_CONTRACTS = table_contracts_for_disposition("full")


@dataclass(frozen=True)
class CohortEntry:
    recording_id: str
    analysis_zarr: Path
    summary_run: str
    track_id: int
    subject_id: str | None


@dataclass(frozen=True)
class MetricDisposition:
    name: str
    reason: str

    @property
    def excluded_metrics(self) -> tuple[str, ...]:
        if self.name == "linear_only":
            return tuple(sorted(_FISH_ANGULAR_FIELDS | _BOUT_ANGULAR_FIELDS))
        return ()


@dataclass(frozen=True)
class LoadedSummary:
    entry: CohortEntry
    source_summary_sha256: str
    source_refs: Mapping[str, Any]
    source_refs_sha256: str
    per_epoch_bouts: np.ndarray
    per_epoch_fish: np.ndarray
    source_table_receipts: Mapping[str, Any]
    summary_attrs: Mapping[str, Any]


def _validate_manifest(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProviderEpochBehaviorCohortError("Cohort manifest must be a JSON object.")
    required = {
        "schema_id",
        "schema_version",
        "cohort_id",
        "metric_disposition",
        "metric_disposition_reason",
        "entries",
        "manifest_payload_sha256",
    }
    if set(payload) != required:
        raise ProviderEpochBehaviorCohortError("Cohort manifest has an unexpected field set.")
    if payload["schema_id"] != INPUT_SCHEMA_ID or payload["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ProviderEpochBehaviorCohortError("Cohort manifest schema identity is invalid.")
    disposition = payload["metric_disposition"]
    if disposition not in METRIC_DISPOSITIONS:
        raise ProviderEpochBehaviorCohortError(
            "metric_disposition must be explicitly one of 'linear_only' or 'full'."
        )
    reason = _string(payload["metric_disposition_reason"], label="metric_disposition_reason")
    if disposition == "linear_only" and not reason:
        raise ProviderEpochBehaviorCohortError(
            "linear_only requires an explicit exclusion reason."
        )
    entries = payload["entries"]
    if not isinstance(entries, list) or not entries:
        raise ProviderEpochBehaviorCohortError("Cohort manifest entries must be non-empty.")
    unsigned = {key: payload[key] for key in required if key != "manifest_payload_sha256"}
    if payload["manifest_payload_sha256"] != canonical_json_sha256(unsigned):
        raise ProviderEpochBehaviorCohortError("Cohort manifest payload digest is invalid.")
    seen_mappings: set[tuple[str, str, str, int]] = set()
    seen_animals: set[tuple[str, int]] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ProviderEpochBehaviorCohortError("Each cohort entry must be an object.")
        if set(raw) != {"recording_id", "analysis_zarr", "summary_run", "track_id", "subject_id"}:
            raise ProviderEpochBehaviorCohortError("Cohort entry fields are incomplete or unknown.")
        recording_id = _string(raw["recording_id"], label="recording_id")
        analysis_zarr = str(Path(_string(raw["analysis_zarr"], label="analysis_zarr")).expanduser().resolve())
        summary_run = _safe_run_name(raw["summary_run"])
        track_id = raw["track_id"]
        if type(track_id) is not int or track_id < 0:
            raise ProviderEpochBehaviorCohortError("track_id must be one nonnegative integer.")
        subject_id = raw["subject_id"]
        if subject_id is not None:
            _string(subject_id, label="subject_id")
        mapping_key = (recording_id, analysis_zarr, summary_run, track_id)
        if mapping_key in seen_mappings:
            raise ProviderEpochBehaviorCohortError(
                "The same recording/analysis_zarr/summary_run/track_id mapping is present more than once."
            )
        animal_key = (recording_id, track_id)
        if animal_key in seen_animals:
            raise ProviderEpochBehaviorCohortError(
                "A recording/track identity is mapped to more than one exact summary run."
            )
        seen_mappings.add(mapping_key)
        seen_animals.add(animal_key)
    return dict(payload)


def load_cohort_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[CohortEntry, ...], str, MetricDisposition]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderEpochBehaviorCohortError(f"Cannot read cohort manifest: {manifest_path}") from exc
    checked = _validate_manifest(payload)
    entries = tuple(
        CohortEntry(
            recording_id=str(raw["recording_id"]),
            analysis_zarr=Path(str(Path(str(raw["analysis_zarr"])).expanduser().resolve())),
            summary_run=str(raw["summary_run"]),
            track_id=int(raw["track_id"]),
            subject_id=str(raw["subject_id"]) if raw["subject_id"] is not None else None,
        )
        for raw in checked["entries"]
    )
    return (
        checked,
        entries,
        canonical_json_sha256(checked),
        MetricDisposition(
            name=str(checked["metric_disposition"]),
            reason=str(checked["metric_disposition_reason"]),
        ),
    )


def _group_at(root: Any, path: str) -> Any:
    group = root
    for part in path.split("/"):
        if not part:
            continue
        group = group.get(part)
        if group is None:
            raise ProviderEpochBehaviorCohortError(f"Missing source group: {path}")
    return group


def _validate_columnar_source(
    array: np.ndarray,
    attrs: Mapping[str, Any],
    *,
    name: str,
    spec: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    expected = _source_dtype(spec)
    if array.dtype != expected:
        raise ProviderEpochBehaviorCohortError(
            f"{name} source dtype differs from the provider summary contract."
        )
    if attrs.get("storage_layout") != "columnar" or attrs.get("field_names") != [item[0] for item in spec]:
        raise ProviderEpochBehaviorCohortError(f"{name} source columnar schema is invalid.")
    expected_dtypes = {field_name: str(expected.fields[field_name][0]) for field_name, _ in spec}
    if attrs.get("field_dtypes") != expected_dtypes:
        raise ProviderEpochBehaviorCohortError(f"{name} source field-dtype declaration is invalid.")
    return {
        "row_count": int(array.shape[0]),
        "schema_sha256": _source_schema_digest(spec),
        "payload_sha256": _array_payload_sha256(array),
    }


def _validate_summary(entry: CohortEntry) -> LoadedSummary:
    if not entry.analysis_zarr.is_dir():
        raise ProviderEpochBehaviorCohortError(f"Analysis Zarr is not a directory: {entry.analysis_zarr}")
    root = open_zarr_root(entry.analysis_zarr, mode="r", use_consolidated=True)
    run_path = f"{SUMMARY_PARENT_PATH}/{entry.summary_run}"
    run = _group_at(root, run_path)
    attrs = dict(run.attrs)
    if attrs.get("schema_id") != "palette.stimulus_epoch_behavior_summary" or attrs.get("schema_version") != 1:
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: summary schema identity is invalid.")
    if attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: summary is incomplete.")
    if attrs.get(RUN_NAME_ATTR) != entry.summary_run or attrs.get("stage_selector_eligible") is not False:
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: summary is not the exact complete selector-ineligible run.")
    if attrs.get("recording_id") != entry.recording_id or attrs.get("track_id") != entry.track_id:
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: summary identity does not match the explicit manifest.")
    source_refs = attrs.get("source_refs")
    if not isinstance(source_refs, Mapping):
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: source_refs are missing.")
    source_refs = json_attr_safe(dict(source_refs))
    source_refs_sha256 = _sha256(attrs.get("source_refs_sha256"), label="source_refs_sha256")
    if source_refs_sha256 != canonical_json_sha256(source_refs):
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: source_refs digest is stale.")
    for component in ("epoch_selection", "provider_motion", "swim_bouts"):
        if not isinstance(source_refs.get(component), Mapping):
            raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: source_refs.{component} is missing.")
    _sha256(source_refs["epoch_selection"].get("sha256"), label="epoch selection source digest")
    _sha256(source_refs["provider_motion"].get("manifest_sha256"), label="provider-motion manifest digest")
    _sha256(source_refs["provider_motion"].get("verification_digest"), label="provider-motion verification digest")
    _sha256(source_refs["swim_bouts"].get("lineage_hash"), label="swim-bout lineage digest")
    _sha256(source_refs["swim_bouts"].get("frame_axis_sha256"), label="swim-bout frame-axis digest")
    analysis_offer = attrs.get("analysis_offer")
    if not isinstance(analysis_offer, Mapping) or attrs.get("analysis_offer_sha256") != canonical_json_sha256(analysis_offer):
        raise ProviderEpochBehaviorCohortError(f"{entry.recording_id}: analysis offer digest is invalid.")
    bouts, bout_attrs = load_structured_dataset(run, "per_epoch_bouts")
    fish, fish_attrs = load_structured_dataset(run, "per_epoch_fish")
    receipts = {
        "per_epoch_bouts": _validate_columnar_source(bouts, bout_attrs, name="per_epoch_bouts", spec=_BOUT_SOURCE_FIELDS),
        "per_epoch_fish": _validate_columnar_source(fish, fish_attrs, name="per_epoch_fish", spec=_FISH_SOURCE_FIELDS),
    }
    source_record = {
        "run_path": run_path,
        "recording_id": entry.recording_id,
        "track_id": entry.track_id,
        "attrs": json_attr_safe(attrs),
        "tables": receipts,
    }
    source_summary_sha256 = canonical_json_sha256(source_record)
    return LoadedSummary(
        entry=entry,
        source_summary_sha256=source_summary_sha256,
        source_refs=source_refs,
        source_refs_sha256=source_refs_sha256,
        per_epoch_bouts=bouts,
        per_epoch_fish=fish,
        source_table_receipts=receipts,
        summary_attrs=json_attr_safe(attrs),
    )


def _lineage_columns(summary: LoadedSummary, cohort_id: str) -> dict[str, object]:
    refs = summary.source_refs
    motion = refs["provider_motion"]
    swim = refs["swim_bouts"]
    return {
        "cohort_id": cohort_id,
        "recording_id": summary.entry.recording_id,
        "analysis_zarr": str(summary.entry.analysis_zarr),
        "subject_id": summary.entry.subject_id,
        "track_id": summary.entry.track_id,
        "summary_run": summary.entry.summary_run,
        "source_summary_sha256": summary.source_summary_sha256,
        "source_refs_sha256": summary.source_refs_sha256,
        "epoch_selection_sha256": refs["epoch_selection"]["sha256"],
        "provider_motion_manifest_sha256": motion["manifest_sha256"],
        "provider_motion_verification_digest": motion["verification_digest"],
        "swim_bout_lineage_sha256": swim["lineage_hash"],
        "swim_bout_frame_axis_sha256": swim["frame_axis_sha256"],
    }


def _build_rows(
    cohort_id: str,
    summary: LoadedSummary,
    *,
    disposition: MetricDisposition,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lineage = _lineage_columns(summary, cohort_id)
    fish_rows: list[dict[str, object]] = []
    seen_fish: set[tuple[int, int]] = set()
    for raw in summary.per_epoch_fish:
        source = _record(raw)
        key = (int(source["track_id"]), int(source["window_id"]))
        if key in seen_fish or key[0] != summary.entry.track_id:
            raise ProviderEpochBehaviorCohortError(f"{summary.entry.recording_id}: duplicate or mismatched fish epoch row.")
        seen_fish.add(key)
        row = dict(lineage)
        for name, _ in _FISH_SOURCE_FIELDS:
            if name == "track_id":
                continue
            if disposition.name == "linear_only" and name in _FISH_ANGULAR_FIELDS:
                continue
            output_name = {"window_id": "epoch_id", "window_index": "epoch_index", "window_label": "epoch_label"}.get(name, name)
            row[output_name] = source[name]
        fish_rows.append(row)
    bout_rows: list[dict[str, object]] = []
    seen_bouts: set[tuple[int, int, int]] = set()
    for raw in summary.per_epoch_bouts:
        source = _record(raw)
        key = (int(source["track_id"]), int(source["window_id"]), int(source["bout_source_row"]))
        if key in seen_bouts or key[0] != summary.entry.track_id:
            raise ProviderEpochBehaviorCohortError(f"{summary.entry.recording_id}: duplicate or mismatched bout row.")
        seen_bouts.add(key)
        row = dict(lineage)
        for name, _ in _BOUT_SOURCE_FIELDS:
            if name == "track_id":
                continue
            if disposition.name == "linear_only" and name in _BOUT_ANGULAR_FIELDS:
                continue
            output_name = {
                "window_id": "epoch_id",
                "window_index": "epoch_index",
                "window_label": "epoch_label",
                "start_frame": "epoch_start_frame",
                "end_frame": "epoch_end_frame",
                "start_time_s": "epoch_start_time_s",
                "end_time_s": "epoch_end_time_s",
                "duration_s": "epoch_duration_s",
            }.get(name, name)
            row[output_name] = source[name]
        bout_rows.append(row)
    return bout_rows, fish_rows


def _lineage_manifest_record(summary: LoadedSummary) -> dict[str, Any]:
    return {
        "recording_id": summary.entry.recording_id,
        "analysis_zarr": str(summary.entry.analysis_zarr),
        "subject_id": summary.entry.subject_id,
        "track_id": summary.entry.track_id,
        "summary_run": summary.entry.summary_run,
        "summary_run_path": f"{SUMMARY_PARENT_PATH}/{summary.entry.summary_run}",
        "source_summary_sha256": summary.source_summary_sha256,
        "source_refs": summary.source_refs,
        "source_refs_sha256": summary.source_refs_sha256,
        "source_tables": summary.source_table_receipts,
        "summary_attrs": summary.summary_attrs,
    }


def build_provider_epoch_behavior_cohort_plan(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    analysis_run_id: str,
) -> dict[str, Any]:
    manifest, entries, manifest_sha256, disposition = load_cohort_manifest(manifest_path)
    run_id = _string(analysis_run_id, label="analysis_run_id")
    loaded = tuple(_validate_summary(entry) for entry in entries)
    loaded = tuple(sorted(loaded, key=lambda item: item.entry.recording_id))
    bout_rows: list[dict[str, object]] = []
    fish_rows: list[dict[str, object]] = []
    for summary in loaded:
        bouts, fish = _build_rows(
            str(manifest["cohort_id"]),
            summary,
            disposition=disposition,
        )
        bout_rows.extend(bouts)
        fish_rows.extend(fish)
    bout_rows.sort(key=lambda row: (row["recording_id"], row["track_id"], row["epoch_id"], row["bout_source_row"]))
    fish_rows.sort(key=lambda row: (row["recording_id"], row["track_id"], row["epoch_id"]))
    return {
        "schema_id": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "cohort_id": manifest["cohort_id"],
        "analysis_run_id": run_id,
        "input_manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "input_manifest_sha256": manifest_sha256,
        "metric_disposition": disposition.name,
        "metric_disposition_reason": disposition.reason,
        "excluded_metrics": list(disposition.excluded_metrics),
        "recording_count": len(loaded),
        "bout_row_count": len(bout_rows),
        "fish_epoch_row_count": len(fish_rows),
        "selector_eligible": False,
        "source_lineage": [_lineage_manifest_record(item) for item in loaded],
        "rows_by_table": {TABLE_BOUTS: bout_rows, TABLE_FISH: fish_rows},
    }


def export_provider_epoch_behavior_cohort(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    analysis_run_id: str,
    apply: bool = False,
    generation_id: str | None = None,
) -> dict[str, Any]:
    plan = build_provider_epoch_behavior_cohort_plan(
        manifest_path,
        output_root=output_root,
        analysis_run_id=analysis_run_id,
    )
    if not apply:
        return {key: value for key, value in plan.items() if key != "rows_by_table"}
    manifest_fields = {
        "export_schema_id": EXPORT_SCHEMA_ID,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "cohort_id": plan["cohort_id"],
        "input_manifest_path": plan["input_manifest_path"],
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "metric_disposition": plan["metric_disposition"],
        "metric_disposition_reason": plan["metric_disposition_reason"],
        "excluded_metrics": plan["excluded_metrics"],
        "recording_count": plan["recording_count"],
        "selector_eligible": False,
        "source_lineage": plan["source_lineage"],
    }
    contracts = table_contracts_for_disposition(str(plan["metric_disposition"]))
    envelope = contract_envelope(
        TABLE_NAMES,
        known_table_names=TABLE_NAMES,
        contracts=contracts,
        schema_id=ARROW_ENVELOPE_SCHEMA_ID,
        schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
    )
    published = publish_derived_table_generation(
        output_root=Path(output_root),
        analysis_run_id=str(plan["analysis_run_id"]),
        rows_by_table=plan["rows_by_table"],
        table_names=TABLE_NAMES,
        contracts=contracts,
        arrow_contract_envelope=envelope,
        arrow_envelope_schema_id=ARROW_ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=ARROW_ENVELOPE_SCHEMA_VERSION,
        manifest_fields=manifest_fields,
        footer_metadata={
            b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
            b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("ascii"),
            b"palette.selector_eligible": b"false",
        },
        selector_eligible=False,
        generation_id=generation_id,
    )
    return {
        **{key: value for key, value in plan.items() if key != "rows_by_table"},
        "publication": published,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--generation-id")
    parser.add_argument("--apply", action="store_true", help="Publish the immutable derived generation.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = export_provider_epoch_behavior_cohort(
        args.manifest,
        output_root=args.output_root,
        analysis_run_id=args.analysis_run_id,
        apply=args.apply,
        generation_id=args.generation_id,
    )
    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARROW_ENVELOPE_SCHEMA_ID",
    "ARROW_ENVELOPE_SCHEMA_VERSION",
    "CohortEntry",
    "EXPORT_SCHEMA_ID",
    "EXPORT_SCHEMA_VERSION",
    "ProviderEpochBehaviorCohortError",
    "MetricDisposition",
    "TABLE_BOUTS",
    "TABLE_CONTRACTS",
    "TABLE_FISH",
    "TABLE_NAMES",
    "build_provider_epoch_behavior_cohort_plan",
    "export_provider_epoch_behavior_cohort",
    "load_cohort_manifest",
    "table_contracts_for_disposition",
]
