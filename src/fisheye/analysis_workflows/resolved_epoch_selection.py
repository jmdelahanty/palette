"""Bounded compatibility adapter for exact stimulus-epoch v2 runs.

This module is intentionally smaller than the future composable selection
system.  It accepts one explicitly named, complete ``stimulus_epoch_runs`` v2
run and projects its windows into immutable half-open atomic intervals for
Phase 4 analysis offers.

The v2 run remains the authority.  Labels are descriptive metadata only; this
adapter never assigns analysis roles, resolves selectors, or combines
intervals.  The returned record is a value-bound compatibility product and is
not a Zarr publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import zarr

from fisheye.analysis.stimulus_epoch_consumer import (
    StimulusEpochSnapshot,
    read_stimulus_epoch_snapshot,
)
from fisheye.analysis.stimulus_epoch_schema import (
    STIMULUS_EPOCH_RUN_MANIFEST_ATTR,
    STIMULUS_EPOCH_RUN_SCHEMA_ID,
    STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
    STIMULUS_SOURCE_FINGERPRINT_ALGORITHM,
    stimulus_epoch_logical_content_sha256,
    stimulus_group_logical_fingerprint,
)
from fisheye.shared.stimulus_coordinate_contract import (
    COORDINATE_CONTRACT_EPOCH,
    STIMULUS_IMPORT_VERSION,
)
from fisheye.analysis_workflows.provider_recording_timing_authority import (
    load_provider_recording_timing_authority,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

RESOLVED_EPOCH_SELECTION_SCHEMA_ID = "palette.resolved_epoch_selection.v1"
RESOLVED_EPOCH_SELECTION_SCHEMA_VERSION = 1
ATOMIC_INTERVAL_SCHEMA_ID = "palette.resolved_epoch_atomic_interval.v1"
ATOMIC_INTERVAL_SCHEMA_VERSION = 1
HALF_OPEN_FRAME_INTERVAL_CONVENTION = "[start_frame,end_frame)"
V2_CONTIGUITY_POLICY = "chronological_non_overlapping_gaps_preserved_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEAL = object()


class ResolvedEpochSelectionError(ValueError):
    """Raised when an exact v2 run cannot be adapted safely."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ResolvedEpochSelectionError(
            f"{name} must be one nonempty canonical string."
        )
    return value


def _require_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ResolvedEpochSelectionError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResolvedEpochSelectionError(f"{name} must be one positive exact integer.")
    return value


def _require_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolvedEpochSelectionError(f"{name} must be one positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ResolvedEpochSelectionError(f"{name} must be one positive finite number.")
    return result


def _canonical_run_name(value: object) -> str:
    if type(value) is not str:
        raise ResolvedEpochSelectionError("run_name must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value in {".", "..", "latest", "latest_complete"}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ResolvedEpochSelectionError(
            "run_name must name one explicit non-selector v2 run."
        )
    return value


def _canonical_manifest(run_group: Any) -> tuple[dict[str, Any], str, str]:
    manifest = run_group.attrs.get(STIMULUS_EPOCH_RUN_MANIFEST_ATTR)
    if not isinstance(manifest, Mapping):
        raise ResolvedEpochSelectionError(
            "Exact v2 run manifest is absent or malformed."
        )
    manifest_record = _thaw(manifest)
    if not isinstance(manifest_record, dict):  # pragma: no cover - defensive
        raise ResolvedEpochSelectionError("Exact v2 run manifest is not an object.")
    payload = manifest_record.get("payload")
    payload_digest = manifest_record.get("payload_digest")
    if not isinstance(payload, dict):
        raise ResolvedEpochSelectionError(
            "Exact v2 run manifest payload is absent or malformed."
        )
    payload_digest = _require_digest(
        payload_digest,
        name="stimulus-epoch run manifest payload digest",
    )
    if canonical_json_sha256(payload) != payload_digest:
        raise ResolvedEpochSelectionError(
            "Stimulus-epoch run manifest payload digest is stale."
        )
    return (
        manifest_record,
        canonical_json_sha256(manifest_record),
        payload_digest,
    )


def _source_stimulus_format_identity(source_group: Any) -> dict[str, Any]:
    """Resolve a declared schema or one exact maintained legacy import contract.

    Historical canonical stimulus imports predate top-level ``schema_id`` and
    ``schema_version`` attributes.  They do, however, bind the maintained
    import version, coordinate-contract epoch, and writer command.  Treat that
    observed tuple as a format identity instead of inventing a schema name or
    mutating the immutable stimulus run in place.
    """

    attrs = source_group.attrs
    schema_id = attrs.get("schema_id")
    schema_version = attrs.get("schema_version")
    if schema_id is not None or schema_version is not None:
        return {
            "identity_kind": "declared_schema",
            "schema_id": _require_text(
                schema_id,
                name="source stimulus schema_id",
            ),
            "schema_version": _require_positive_int(
                schema_version,
                name="source stimulus schema_version",
            ),
        }

    import_version = attrs.get("import_version")
    coordinate_epoch = attrs.get("coordinate_contract_epoch")
    provenance = attrs.get("run_provenance")
    writer_command = (
        provenance.get("command") if isinstance(provenance, Mapping) else None
    )
    if (
        import_version != STIMULUS_IMPORT_VERSION
        or coordinate_epoch != COORDINATE_CONTRACT_EPOCH
        or writer_command != "fisheye.analysis.import_stimulus_to_zarr"
    ):
        raise ResolvedEpochSelectionError(
            "Source stimulus has neither a declared schema nor the exact "
            "maintained legacy import identity."
        )
    return {
        "identity_kind": "maintained_legacy_import_contract",
        "import_version": import_version,
        "coordinate_contract_epoch": int(coordinate_epoch),
        "writer_command": writer_command,
    }


def _source_timeline_binding(
    root: zarr.Group,
    run_group: Any,
    *,
    run_path: str,
) -> tuple[dict[str, Any], str, str]:
    attrs = run_group.attrs
    recording_id = _require_text(attrs.get("recording_id"), name="recording_id")
    source_run = _require_text(
        attrs.get("source_stimulus_run"), name="source_stimulus_run"
    )
    source_path = _require_text(
        attrs.get("source_stimulus_path"), name="source_stimulus_path"
    )
    expected_source_path = f"analysis/stimulus_runs/{source_run}"
    if source_path != expected_source_path:
        raise ResolvedEpochSelectionError(
            "source_stimulus_path does not bind source_stimulus_run."
        )
    source_group = root.get(source_path)
    if not isinstance(source_group, zarr.Group):
        raise ResolvedEpochSelectionError(
            "Exact source stimulus timeline group is absent."
        )
    source_group_run_name = source_group.attrs.get("run_name")
    if source_group_run_name is not None and source_group_run_name != source_run:
        raise ResolvedEpochSelectionError(
            "Source stimulus timeline run-name identity is stale."
        )
    source_format_identity = _source_stimulus_format_identity(source_group)
    fingerprint_algorithm = _require_text(
        attrs.get("source_stimulus_fingerprint_algorithm"),
        name="source_stimulus_fingerprint_algorithm",
    )
    if fingerprint_algorithm != STIMULUS_SOURCE_FINGERPRINT_ALGORITHM:
        raise ResolvedEpochSelectionError(
            "Source stimulus timeline fingerprint algorithm is unsupported."
        )
    declared_fingerprint = _require_digest(
        attrs.get("source_stimulus_fingerprint"),
        name="source_stimulus_fingerprint",
    )
    observed_fingerprint = stimulus_group_logical_fingerprint(source_group)
    if observed_fingerprint != declared_fingerprint:
        raise ResolvedEpochSelectionError(
            "Source stimulus timeline fingerprint is stale."
        )
    source_event_schema = attrs.get("source_event_schema")
    if not isinstance(source_event_schema, Mapping):
        raise ResolvedEpochSelectionError(
            "Exact source event/timeline schema identity is absent."
        )
    source_event_schema = _thaw(source_event_schema)
    if not isinstance(source_event_schema, dict):  # pragma: no cover - defensive
        raise ResolvedEpochSelectionError(
            "Exact source event/timeline schema identity is malformed."
        )
    timeline_identity = {
        "recording_id": recording_id,
        "source_stimulus_run": source_run,
        "source_stimulus_path": source_path,
        "source_stimulus_format_identity": source_format_identity,
        "source_event_schema": source_event_schema,
        "fingerprint_algorithm": fingerprint_algorithm,
        "fingerprint": declared_fingerprint,
        "source_epoch_run_path": run_path,
    }
    return timeline_identity, canonical_json_sha256(timeline_identity), source_path


def _validate_exact_timing(
    snapshot: StimulusEpochSnapshot,
    run_group: Any,
) -> tuple[int, float]:
    total_frames = _require_positive_int(
        run_group.attrs.get("total_frames"), name="native frame count"
    )
    fps = _require_positive_float(run_group.attrs.get("fps"), name="FPS")
    if not snapshot.segments:
        raise ResolvedEpochSelectionError("Exact v2 run has no atomic windows.")
    previous_end: int | None = None
    for segment in snapshot.segments:
        start = int(segment.start_frame)
        end_exclusive = int(segment.end_frame) + 1
        if start < 0 or end_exclusive <= start or end_exclusive > total_frames:
            raise ResolvedEpochSelectionError(
                "Exact v2 window bounds are invalid for the native frame count."
            )
        if previous_end is not None:
            if start < previous_end:
                raise ResolvedEpochSelectionError(
                    "Exact v2 windows overlap; the bounded adapter requires "
                    "chronological non-overlapping windows."
                )
        if (
            not math.isfinite(float(segment.start_time_s))
            or not math.isfinite(float(segment.end_time_s))
            or not math.isfinite(float(segment.duration_s))
        ):
            raise ResolvedEpochSelectionError(
                "Exact v2 timing authority contains a non-finite value."
            )
        expected_start = float(start) / fps
        expected_end = float(end_exclusive) / fps
        expected_duration = float(end_exclusive - start) / fps
        tolerance = max(math.ulp(max(total_frames, 1) / fps), 1e-12)
        if (
            abs(float(segment.start_time_s) - expected_start) > tolerance
            or abs(float(segment.end_time_s) - expected_end) > tolerance
            or abs(float(segment.duration_s) - expected_duration) > tolerance
        ):
            raise ResolvedEpochSelectionError(
                "Exact v2 window timing does not match its frame bounds and FPS."
            )
        previous_end = end_exclusive
    return total_frames, fps


@dataclass(frozen=True, init=False)
class AtomicStimulusEpochInterval:
    """One source v2 window represented as a half-open atomic interval."""

    window_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    source_metadata_identity: Mapping[str, Any] = field(repr=False)
    occurrence_identity: Mapping[str, Any] = field(repr=False)
    source_interval_digest: str
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _SEAL:
            raise ResolvedEpochSelectionError(
                "Atomic intervals must be created by the exact v2 adapter."
            )
        for name, value in values.items():
            if name in {"source_metadata_identity", "occurrence_identity"}:
                value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_id": ATOMIC_INTERVAL_SCHEMA_ID,
            "schema_version": ATOMIC_INTERVAL_SCHEMA_VERSION,
            "window_id": self.window_id,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "frame_interval_convention": HALF_OPEN_FRAME_INTERVAL_CONVENTION,
            "source_metadata_identity": _thaw(self.source_metadata_identity),
            "occurrence_identity": _thaw(self.occurrence_identity),
            "source_interval_digest": self.source_interval_digest,
        }


@dataclass(frozen=True, init=False)
class ResolvedEpochSelection:
    """Canonical, immutable Phase 4 selection projection of one exact v2 run."""

    run_name: str
    run_path: str
    run_schema_id: str
    run_schema_version: int
    run_manifest_digest: str
    run_manifest_payload_digest: str
    source_epoch_logical_content_digest: str
    source_epoch_lineage_hash: str
    source_epoch_lineage_payload_digest: str
    source_timeline_identity: Mapping[str, Any] = field(repr=False)
    source_timeline_digest: str
    native_frame_count: int
    fps: float
    timing_authority: Mapping[str, Any] = field(repr=False)
    recording_timing_authority: Mapping[str, Any] | None = field(repr=False)
    recording_timing_authority_sha256: str | None
    recording_timing_authority_status: str
    intervals: tuple[AtomicStimulusEpochInterval, ...]
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _SEAL:
            raise ResolvedEpochSelectionError(
                "Resolved selections must be created by the exact v2 adapter."
            )
        for name, value in values.items():
            if name in {
                "source_timeline_identity",
                "timing_authority",
                "recording_timing_authority",
            }:
                value = _freeze(value)
            elif name == "intervals":
                value = tuple(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    def to_record(self, *, include_digest: bool = True) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_id": RESOLVED_EPOCH_SELECTION_SCHEMA_ID,
            "schema_version": RESOLVED_EPOCH_SELECTION_SCHEMA_VERSION,
            "selection_kind": "atomic_stimulus_epoch_intervals",
            "run": {
                "name": self.run_name,
                "path": self.run_path,
                "schema_id": self.run_schema_id,
                "schema_version": self.run_schema_version,
                "manifest_sha256": self.run_manifest_digest,
                "manifest_payload_sha256": self.run_manifest_payload_digest,
                "logical_content_sha256": self.source_epoch_logical_content_digest,
                "lineage_hash": self.source_epoch_lineage_hash,
                "lineage_payload_sha256": self.source_epoch_lineage_payload_digest,
            },
            "source_timeline": _thaw(self.source_timeline_identity),
            "source_timeline_digest": self.source_timeline_digest,
            "timing": _thaw(self.timing_authority),
            "recording_timing_authority": {
                "status": self.recording_timing_authority_status,
                "record": _thaw(self.recording_timing_authority),
                "sha256": self.recording_timing_authority_sha256,
            },
            "native_frame_count": self.native_frame_count,
            "fps": self.fps,
            "frame_interval_convention": HALF_OPEN_FRAME_INTERVAL_CONVENTION,
            "contiguity_policy": V2_CONTIGUITY_POLICY,
            "intervals": [interval.to_record() for interval in self.intervals],
        }
        if include_digest:
            record["selection_sha256"] = canonical_json_sha256(record)
        return record

    @property
    def selection_record(self) -> dict[str, Any]:
        """Return the canonical record including its self-excluding digest."""

        return self.to_record(include_digest=True)

    @property
    def selection_digest(self) -> str:
        return canonical_json_sha256(self.to_record(include_digest=False))

    def assert_verified(self) -> None:
        if self._seal is not _SEAL:
            raise ResolvedEpochSelectionError("Selection verification seal is absent.")
        record = self.to_record(include_digest=False)
        if canonical_json_sha256(record) != self.selection_digest:
            raise ResolvedEpochSelectionError("Selection digest is stale.")


def _build_intervals(
    snapshot: StimulusEpochSnapshot,
    *,
    run_path: str,
    source_timeline_identity: Mapping[str, Any],
    source_epoch_logical_content_digest: str,
    source_epoch_lineage_hash: str,
    source_epoch_lineage_payload_digest: str,
) -> tuple[AtomicStimulusEpochInterval, ...]:
    intervals: list[AtomicStimulusEpochInterval] = []
    for segment in snapshot.segments:
        label = _require_text(segment.label, name="window label")
        source_start_event_name = _require_text(
            segment.source_start_event_name,
            name="source start event name",
        )
        source_end_event_name = _require_text(
            segment.source_end_event_name,
            name="source end event name",
        )
        if (
            segment.source_start_event_frame is None
            or segment.source_end_event_frame is None
        ):
            raise ResolvedEpochSelectionError(
                "Exact v2 window lacks complete source occurrence boundaries."
            )
        source_policy = _require_text(
            segment.source_policy,
            name="source window policy",
        )
        start_frame = int(segment.start_frame)
        end_frame = int(segment.end_frame) + 1
        occurrence_identity = {
            "schema_id": "palette.stimulus_epoch_occurrence.v1",
            "occurrence_id": f"{run_path}#window:{int(segment.segment_id)}",
            "source_run_path": run_path,
            "window_id": int(segment.segment_id),
            "source_start_event_name": source_start_event_name,
            "source_end_event_name": source_end_event_name,
            "source_start_event_frame": int(segment.source_start_event_frame),
            "source_end_event_frame": int(segment.source_end_event_frame),
        }
        source_metadata_identity = {
            "source_epoch_run_path": run_path,
            "source_epoch_schema_id": STIMULUS_EPOCH_RUN_SCHEMA_ID,
            "source_epoch_schema_version": STIMULUS_EPOCH_RUN_SCHEMA_VERSION,
            "source_epoch_logical_content_sha256": source_epoch_logical_content_digest,
            "source_epoch_lineage_hash": source_epoch_lineage_hash,
            "source_epoch_lineage_payload_sha256": source_epoch_lineage_payload_digest,
            "source_timeline_digest": canonical_json_sha256(source_timeline_identity),
            "source_start_event_name": source_start_event_name,
            "source_end_event_name": source_end_event_name,
            "source_start_event_frame": int(segment.source_start_event_frame),
            "source_end_event_frame": int(segment.source_end_event_frame),
            "source_policy": source_policy,
        }
        source_interval_payload = {
            "schema_id": ATOMIC_INTERVAL_SCHEMA_ID,
            "schema_version": ATOMIC_INTERVAL_SCHEMA_VERSION,
            "window_id": int(segment.segment_id),
            "label": label,
            "source_metadata_identity": source_metadata_identity,
            "occurrence_identity": occurrence_identity,
            "source_v2_frame_bounds": {
                "start_frame_inclusive": start_frame,
                "end_frame_inclusive": int(segment.end_frame),
            },
            "resolved_half_open_frame_bounds": {
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
            "source_v2_times": {
                "start_time_s": float(segment.start_time_s),
                "end_time_s": float(segment.end_time_s),
                "duration_s": float(segment.duration_s),
            },
        }
        source_interval_digest = canonical_json_sha256(source_interval_payload)
        intervals.append(
            AtomicStimulusEpochInterval(
                _verification_seal=_SEAL,
                window_id=int(segment.segment_id),
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_s=float(segment.start_time_s),
                end_time_s=float(segment.end_time_s),
                duration_s=float(segment.duration_s),
                source_metadata_identity=source_metadata_identity,
                occurrence_identity=occurrence_identity,
                source_interval_digest=source_interval_digest,
            )
        )
    return tuple(intervals)


def resolve_exact_stimulus_epoch_selection(
    archive_path: str | Path,
    *,
    run_name: str,
    expected_run_manifest_digest: str | None = None,
    expected_source_timeline_digest: str | None = None,
    expected_source_epoch_logical_content_digest: str | None = None,
) -> ResolvedEpochSelection:
    """Adapt one explicit complete v2 run into atomic half-open intervals.

    ``run_name`` is mandatory and is passed directly to the maintained strict
    snapshot reader.  The optional expected digests are caller bindings for a
    previously planned offer; a mismatch is stale input and fails closed.
    No selector, label role, timing fallback, or interval operation is used.
    """

    run_name = _canonical_run_name(run_name)
    try:
        snapshot = read_stimulus_epoch_snapshot(
            archive_path,
            run_name=run_name,
        )
        if snapshot.schema_id != STIMULUS_EPOCH_RUN_SCHEMA_ID:
            raise ResolvedEpochSelectionError(
                "Exact v2 snapshot schema identity differs."
            )
        if snapshot.schema_version != STIMULUS_EPOCH_RUN_SCHEMA_VERSION:
            raise ResolvedEpochSelectionError(
                "Exact v2 snapshot schema version differs."
            )
        archive = Path(archive_path).expanduser().resolve()
        root = zarr.open_group(
            str(archive),
            mode="r",
            zarr_format=3,
            use_consolidated=True,
        )
        run_group = root[snapshot.run_path]
        _manifest, run_manifest_digest, run_manifest_payload_digest = (
            _canonical_manifest(run_group)
        )
        if expected_run_manifest_digest is not None:
            expected_run_manifest_digest = _require_digest(
                expected_run_manifest_digest,
                name="expected run manifest digest",
            )
            if run_manifest_digest != expected_run_manifest_digest:
                raise ResolvedEpochSelectionError(
                    "Exact v2 run manifest digest is stale."
                )
        source_epoch_logical_content_digest = _require_digest(
            run_group.attrs.get("source_stimulus_epoch_logical_content_sha256"),
            name="source epoch logical content digest",
        )
        observed_content_digest = stimulus_epoch_logical_content_sha256(run_group)
        if observed_content_digest != source_epoch_logical_content_digest:
            raise ResolvedEpochSelectionError(
                "Source epoch logical content digest is stale."
            )
        if expected_source_epoch_logical_content_digest is not None:
            expected_source_epoch_logical_content_digest = _require_digest(
                expected_source_epoch_logical_content_digest,
                name="expected source epoch logical content digest",
            )
            if (
                source_epoch_logical_content_digest
                != expected_source_epoch_logical_content_digest
            ):
                raise ResolvedEpochSelectionError(
                    "Expected source epoch logical content digest is stale."
                )
        source_epoch_lineage_hash = _require_digest(
            run_group.attrs.get("source_stimulus_epoch_lineage_hash"),
            name="source epoch lineage hash",
        )
        source_epoch_lineage_payload_digest = _require_digest(
            run_group.attrs.get("source_stimulus_epoch_lineage_payload_sha256"),
            name="source epoch lineage payload digest",
        )
        source_timeline_identity, source_timeline_digest, _source_path = (
            _source_timeline_binding(
                root,
                run_group,
                run_path=snapshot.run_path,
            )
        )
        if expected_source_timeline_digest is not None:
            expected_source_timeline_digest = _require_digest(
                expected_source_timeline_digest,
                name="expected source timeline digest",
            )
            if source_timeline_digest != expected_source_timeline_digest:
                raise ResolvedEpochSelectionError(
                    "Expected source timeline digest is stale."
                )
        native_frame_count, fps = _validate_exact_timing(snapshot, run_group)
        recording_timing = load_provider_recording_timing_authority(
            archive,
            required=False,
            use_consolidated=True,
        )
        if recording_timing is not None:
            if (
                recording_timing.recording_id
                != source_timeline_identity["recording_id"]
            ):
                raise ResolvedEpochSelectionError(
                    "Stimulus epoch and recording timing identities disagree."
                )
            if recording_timing.frame_count != native_frame_count:
                raise ResolvedEpochSelectionError(
                    "Stimulus epoch and recording timing frame counts disagree."
                )
            if recording_timing.nominal_fps != fps:
                raise ResolvedEpochSelectionError(
                    "Stimulus epoch and recording timing nominal FPS values disagree."
                )
        intervals = _build_intervals(
            snapshot,
            run_path=snapshot.run_path,
            source_timeline_identity=source_timeline_identity,
            source_epoch_logical_content_digest=source_epoch_logical_content_digest,
            source_epoch_lineage_hash=source_epoch_lineage_hash,
            source_epoch_lineage_payload_digest=source_epoch_lineage_payload_digest,
        )
        timing_authority = {
            "authority": "stimulus_epoch_v2_run_attributes",
            "run_path": snapshot.run_path,
            "native_frame_count": native_frame_count,
            "fps": fps,
            "frame_index_origin": 0,
            "frame_time_rule": "frame_index_divided_by_fps",
            "end_time_rule": "exclusive_end_frame_divided_by_fps",
        }
        selection = ResolvedEpochSelection(
            _verification_seal=_SEAL,
            run_name=run_name,
            run_path=snapshot.run_path,
            run_schema_id=snapshot.schema_id,
            run_schema_version=snapshot.schema_version,
            run_manifest_digest=run_manifest_digest,
            run_manifest_payload_digest=run_manifest_payload_digest,
            source_epoch_logical_content_digest=source_epoch_logical_content_digest,
            source_epoch_lineage_hash=source_epoch_lineage_hash,
            source_epoch_lineage_payload_digest=source_epoch_lineage_payload_digest,
            source_timeline_identity=source_timeline_identity,
            source_timeline_digest=source_timeline_digest,
            native_frame_count=native_frame_count,
            fps=fps,
            timing_authority=timing_authority,
            recording_timing_authority=(
                None if recording_timing is None else _thaw(recording_timing.record)
            ),
            recording_timing_authority_sha256=(
                None if recording_timing is None else recording_timing.sha256
            ),
            recording_timing_authority_status=(
                "legacy_missing" if recording_timing is None else "bound"
            ),
            intervals=intervals,
        )
        selection.assert_verified()
        return selection
    except ResolvedEpochSelectionError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ResolvedEpochSelectionError(str(exc)) from exc


__all__ = [
    "ATOMIC_INTERVAL_SCHEMA_ID",
    "ATOMIC_INTERVAL_SCHEMA_VERSION",
    "HALF_OPEN_FRAME_INTERVAL_CONVENTION",
    "RESOLVED_EPOCH_SELECTION_SCHEMA_ID",
    "RESOLVED_EPOCH_SELECTION_SCHEMA_VERSION",
    "ResolvedEpochSelection",
    "ResolvedEpochSelectionError",
    "AtomicStimulusEpochInterval",
    "V2_CONTIGUITY_POLICY",
    "resolve_exact_stimulus_epoch_selection",
]
