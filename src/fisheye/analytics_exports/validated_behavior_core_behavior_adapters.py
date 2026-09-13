"""Bounded five-grain adapters for the core-behavior cohort profile.

Every extractor consumes the report-selected authorities sealed in the generic
bundle set.  The adapters reuse the maintained standalone projections; they do
not publish, discover ``latest`` selectors, or invent a second manifest.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from fisheye.analysis_workflows.core_behavior_cohort_adapter import (
    CORE_BEHAVIOR_BUNDLE_ADAPTER_ID,
    bind_core_behavior_cohort_sources,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .arrow_contracts import ARROW_TABLE_CONTRACTS
from .contracts import (
    EYE_TRACE_SAMPLES_TABLE,
    KINEMATICS_SAMPLES_TABLE,
    TAIL_TRACE_SAMPLES_TABLE,
)
from .eye_trace_samples import iter_projected_eye_trace_batches
from .kinematics_samples import (
    iter_projected_core_motion_sample_batches,
    iter_projected_kinematics_sample_batches,
)
from .tail_trace_samples import projected_tail_trace_sample_batch
from .validated_behavior_cohort import ValidatedBehaviorBatchSource
from .validated_behavior_core_behavior_contracts import (
    CANONICAL_SWIM_BOUTS_CAPABILITY,
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID,
    CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1,
    CORE_MOTION_KINEMATICS_PROJECTION_FIELDS,
    EYE_TRACE_CAPABILITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    KINEMATICS_SAMPLES,
    SUBJECT_BODY_FRAME_CAPABILITY,
    SUBJECT_BODY_FRAME_SAMPLES_TABLE,
    TAIL_TRACE_CAPABILITY,
)
from .validated_behavior_bout_kinematics_contracts import (
    BOUT_EYE_GAZE_TABLE,
    BOUT_HEADING_TABLE,
    BOUT_KINEMATICS_CAPABILITY,
    BOUT_KINEMATICS_CAPABILITY_KEYS,
    BOUT_KINEMATICS_EXPORT_PROFILE_ID,
    BOUT_MOVEMENT_TABLE,
    source_dtype,
    source_field_name,
)


class CoreBehaviorExportAdapterError(ValueError):
    """A core-behavior bundle or source changed after admission."""


def _fail(message: str) -> None:
    raise CoreBehaviorExportAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _repeat(value: Any, count: int) -> list[Any]:
    return [value] * count


class _CoreBehaviorContext:
    def __init__(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> None:
        if bundle_member.get("bundle_state") != "complete":
            _fail("Core-behavior extractors require one complete bundle member.")
        bundle = _mapping(bundle_member.get("bundle"), field="bundle member")
        if bundle.get("adapter_id") != CORE_BEHAVIOR_BUNDLE_ADAPTER_ID:
            _fail("Core-behavior extractor received another bundle profile.")
        if bundle_member.get("recording_id") != membership_member.get(
            "recording_id"
        ) or bundle_member.get("analysis_zarr") != membership_member.get(
            "analysis_zarr"
        ):
            _fail("Membership and bundle-set member identities differ.")
        self.plan = plan
        self.membership_member = membership_member
        self.bundle_member = bundle_member
        self.bundle = bundle
        export_profile = _mapping(
            plan.get("export_profile"), field="plan export_profile"
        )
        self.export_profile_id = str(export_profile.get("profile_id"))
        if self.export_profile_id not in {
            CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1,
            CORE_BEHAVIOR_EXPORT_PROFILE_ID,
            BOUT_KINEMATICS_EXPORT_PROFILE_ID,
        }:
            _fail("Core-behavior extractor received an unsupported export profile.")
        self.bound = bind_core_behavior_cohort_sources(
            bundle["path"],
            expected_analysis_zarr=membership_member["analysis_zarr"],
            expected_recording_id=membership_member["recording_id"],
            export_profile_id=self.export_profile_id,
        )
        if (
            bundle.get("file_sha256") != self.bound.report_binding["file_sha256"]
            or bundle.get("record_sha256") != self.bound.report["record_sha256"]
        ):
            _fail("Core-behavior execution report changed after bundle admission.")
        capabilities = _mapping(
            bundle_member.get("capabilities"), field="bundle capabilities"
        )
        capability_keys = (
            BOUT_KINEMATICS_CAPABILITY_KEYS
            if self.export_profile_id == BOUT_KINEMATICS_EXPORT_PROFILE_ID
            else CORE_BEHAVIOR_CAPABILITY_KEYS
        )
        if set(capabilities) != set(capability_keys):
            _fail("Core-behavior bundle capability roster is inexact.")
        for capability_id in capability_keys:
            capability = _mapping(
                capabilities[capability_id], field=f"capability {capability_id}"
            )
            if (
                capability.get("state") != "complete"
                or capability.get("reason_code") is not None
                or capability.get("detail") is not None
                or _plain(capability.get("binding"))
                != _plain(self.bound.capability_bindings[capability_id])
            ):
                _fail(f"Capability {capability_id!r} differs from its current source.")

    @property
    def recording_id(self) -> str:
        return str(self.membership_member["recording_id"])

    @property
    def row_group_rows(self) -> int:
        value = self.plan["parameters"]["effective_row_group_rows"]
        if type(value) is not int or value <= 0:
            _fail("Export plan row-group size is invalid.")
        return value

    def capability_binding(self, capability_id: str) -> Mapping[str, Any]:
        return _mapping(
            self.bundle_member["capabilities"][capability_id]["binding"],
            field=f"capability {capability_id} binding",
        )

    def common_columns(self, count: int) -> dict[str, list[Any]]:
        join_sha = str(self.bound.join_authority["payload_sha256"])
        return {
            "export_run_id": _repeat(str(self.plan["export_run_id"]), count),
            "recording_id": _repeat(self.recording_id, count),
            "membership_member_sha256": _repeat(
                str(self.membership_member["member_sha256"]), count
            ),
            "bundle_set_member_sha256": _repeat(
                str(self.bundle_member["member_sha256"]), count
            ),
            "bundle_record_sha256": _repeat(str(self.bundle["record_sha256"]), count),
            "cross_grain_join_authority_sha256": _repeat(join_sha, count),
        }

    @property
    def bundle_common(self) -> dict[str, Any]:
        return {
            "export_run_id": str(self.plan["export_run_id"]),
            "recording_id": self.recording_id,
            "membership_member_sha256": str(self.membership_member["member_sha256"]),
            "bundle_set_member_sha256": str(self.bundle_member["member_sha256"]),
            "bundle_record_sha256": str(self.bundle["record_sha256"]),
        }


class _LastCoreBehaviorContext:
    """Keep one recording's strict handles while all sibling grains are written."""

    def __init__(self) -> None:
        self._key: tuple[str, str, str] | None = None
        self._context: _CoreBehaviorContext | None = None

    def get(
        self,
        plan: Mapping[str, Any],
        membership_member: Mapping[str, Any],
        bundle_member: Mapping[str, Any],
    ) -> _CoreBehaviorContext:
        key = (
            str(plan["plan_sha256"]),
            str(membership_member["member_sha256"]),
            str(bundle_member["member_sha256"]),
        )
        if self._key != key:
            self._context = _CoreBehaviorContext(plan, membership_member, bundle_member)
            self._key = key
        assert self._context is not None
        return self._context


def _batch_source(
    batches: Iterable[Mapping[str, Any]],
) -> ValidatedBehaviorBatchSource:
    """Peek once so empty scientific sources carry an explicit typed reason."""

    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        return ValidatedBehaviorBatchSource((), "complete-no-rows")
    return ValidatedBehaviorBatchSource(chain((first,), iterator), None)


def _cohortize_standalone_batches(
    context: _CoreBehaviorContext,
    *,
    table_name: str,
    batches: Iterable[Mapping[str, Any]],
    source_fields: set[str] | None = None,
) -> Iterator[Mapping[str, Any]]:
    expected_source_fields = (
        {field.name for field in ARROW_TABLE_CONTRACTS[table_name].fields}
        if source_fields is None
        else set(source_fields)
    )
    omitted = {"export_schema_version", "table_name", "recording_id"}
    for index, columns in enumerate(batches):
        if set(columns) != expected_source_fields:
            _fail(
                f"{table_name}: standalone projection batch {index} has an "
                "inexact field roster."
            )
        lengths = {
            len(value)
            for name, value in columns.items()
            if name not in {"export_schema_version", "table_name", "recording_id"}
        }
        if len(lengths) != 1:
            _fail(f"{table_name}: projected columns do not share one row axis.")
        count = lengths.pop()
        yield {
            **context.common_columns(count),
            **{name: value for name, value in columns.items() if name not in omitted},
        }


def _kinematics_samples(context: _CoreBehaviorContext) -> ValidatedBehaviorBatchSource:
    capability = context.capability_binding(KINEMATICS_SAMPLES_CAPABILITY)
    source_fields: set[str] | None = None
    if context.export_profile_id == CORE_BEHAVIOR_EXPORT_PROFILE_ID_V1:
        batches = iter_projected_kinematics_sample_batches(
            context.bound.track,
            projection=capability["projection_contract"],
            source_window_rows=context.row_group_rows,
        )
    else:
        batches = iter_projected_core_motion_sample_batches(
            context.bound.track,
            projection=capability["projection_contract"],
            source_window_rows=context.row_group_rows,
            expected_arrow_schema_sha256=KINEMATICS_SAMPLES.payload_sha256,
        )
        source_fields = {item.name for item in CORE_MOTION_KINEMATICS_PROJECTION_FIELDS}
    return _batch_source(
        _cohortize_standalone_batches(
            context,
            table_name=KINEMATICS_SAMPLES_TABLE,
            batches=batches,
            source_fields=source_fields,
        )
    )


def _eye_trace_samples(context: _CoreBehaviorContext) -> ValidatedBehaviorBatchSource:
    capability = context.capability_binding(EYE_TRACE_CAPABILITY)
    batches = iter_projected_eye_trace_batches(
        context.bound.root,
        source_binding=capability["source_binding"],
        projection=capability["projection_contract"],
        row_group_rows=context.row_group_rows,
    )
    return _batch_source(
        _cohortize_standalone_batches(
            context, table_name=EYE_TRACE_SAMPLES_TABLE, batches=batches
        )
    )


def _tail_trace_samples(context: _CoreBehaviorContext) -> ValidatedBehaviorBatchSource:
    capability = context.capability_binding(TAIL_TRACE_CAPABILITY)
    source_rows = int(context.bound.tail.binding["tail_row_count"])
    sample_count = int(context.bound.tail.binding["source_tail_sample_count"])
    if sample_count <= 0:
        _fail("Tail source declares no normalized samples.")
    source_window_rows = max(1, context.row_group_rows // sample_count)

    def batches() -> Iterator[Mapping[str, Any]]:
        for start in range(0, source_rows, source_window_rows):
            stop = min(source_rows, start + source_window_rows)
            yield projected_tail_trace_sample_batch(
                context.bound.tail,
                start_row=start,
                stop_row=stop,
                projection=capability["projection_contract"],
            )

    return _batch_source(
        _cohortize_standalone_batches(
            context, table_name=TAIL_TRACE_SAMPLES_TABLE, batches=batches()
        )
    )


_BODY_ARRAYS: Mapping[str, tuple[np.dtype[Any], tuple[int, ...] | None]] = {
    "instance_key": (np.dtype("<u8"), ()),
    "source_crop_row_ids": (np.dtype("<i8"), ()),
    "source_acquisition_frame_index": (np.dtype("<i8"), ()),
    "body_frame/origin_xy": (np.dtype("<f4"), (2,)),
    "body_frame/forward_axis_xy": (np.dtype("<f4"), (2,)),
    "body_frame/left_axis_xy": (np.dtype("<f4"), (2,)),
    "body_frame/heading_deg": (np.dtype("<f4"), ()),
    "body_frame/axis_valid": (np.dtype("bool"), ()),
    "body_frame/failure_reason_bytes": (np.dtype("u1"), (64,)),
}


def _body_window(
    context: _CoreBehaviorContext, *, start: int, stop: int
) -> dict[str, np.ndarray]:
    run = context.bound.subject_shape._run
    count = stop - start
    result: dict[str, np.ndarray] = {}
    for path, (dtype, trailing) in _BODY_ARRAYS.items():
        values = np.asarray(run[path][start:stop])
        expected = (count, *trailing)
        if values.dtype != dtype or values.shape != expected:
            _fail(
                f"Subject body-frame array {path!r} changed from "
                f"{dtype.str}{expected!r}."
            )
        result[path] = values
    return result


def _subject_body_frame_samples(
    context: _CoreBehaviorContext,
) -> ValidatedBehaviorBatchSource:
    capability = context.capability_binding(SUBJECT_BODY_FRAME_CAPABILITY)
    source = capability["source_binding"]
    projection = capability["projection_contract"]
    row_count = int(source["row_count"])
    rate = float(source["source_sample_rate_hz"])
    lineage = canonical_json_sha256(
        {
            "source_binding_sha256": source["payload_sha256"],
            "projection_contract_sha256": projection["payload_sha256"],
        }
    )
    constants = {
        "zarr_path": str(context.membership_member["analysis_zarr"]),
        "source_lineage_hash": lineage,
        "source_subject_shape_run": source["run_name"],
        "source_subject_shape_path": source["run_path"],
        "source_subject_shape_schema_id": source["source_schema_id"],
        "source_subject_shape_schema_version": source["source_schema_version"],
        "source_subject_shape_publication_manifest_sha256": source[
            "publication_manifest_sha256"
        ],
        "source_binding_sha256": source["payload_sha256"],
        "projection_contract_sha256": projection["payload_sha256"],
        "row_identity_sha256": source["row_identity_sha256"],
        "temporal_authority_sha256": source["temporal_authority_sha256"],
        "acquisition_camera_frame_sha256": source["acquisition_camera_frame_sha256"],
        "camera_id": source["camera_id"],
        "source_sample_rate_hz": rate,
        "body_frame_record_sha256": source["body_frame_record_sha256"],
        "heading_semantics_sha256": source["heading_semantics_sha256"],
        "origin_coordinate_descriptor_sha256": source[
            "origin_coordinate_descriptor_sha256"
        ],
        "forward_coordinate_descriptor_sha256": source[
            "forward_coordinate_descriptor_sha256"
        ],
        "left_coordinate_descriptor_sha256": source[
            "left_coordinate_descriptor_sha256"
        ],
    }

    def batches() -> Iterator[Mapping[str, Any]]:
        for start in range(0, row_count, context.row_group_rows):
            stop = min(row_count, start + context.row_group_rows)
            count = stop - start
            values = _body_window(context, start=start, stop=stop)
            frame = values["source_acquisition_frame_index"]
            origin = values["body_frame/origin_xy"]
            forward = values["body_frame/forward_axis_xy"]
            left = values["body_frame/left_axis_xy"]
            yield {
                **context.common_columns(count),
                **{name: _repeat(value, count) for name, value in constants.items()},
                "subject_shape_row_index": np.arange(start, stop, dtype=np.int64),
                "instance_key": values["instance_key"],
                "source_crop_row_id": values["source_crop_row_ids"],
                "source_acquisition_frame_index": frame,
                "time_seconds": frame.astype(np.float64) / rate,
                "origin_x_px": origin[:, 0],
                "origin_y_px": origin[:, 1],
                "forward_x": forward[:, 0],
                "forward_y": forward[:, 1],
                "left_x": left[:, 0],
                "left_y": left[:, 1],
                "heading_deg": values["body_frame/heading_deg"],
                "body_frame_valid": values["body_frame/axis_valid"],
                "failure_reason": [
                    decode_null_terminated_text(row, errors="strict")
                    for row in values["body_frame/failure_reason_bytes"]
                ],
            }

    return _batch_source(batches())


def _canonical_swim_bouts(
    context: _CoreBehaviorContext,
) -> tuple[list[dict[str, Any]], str | None]:
    capability = context.capability_binding(CANONICAL_SWIM_BOUTS_CAPABILITY)
    source = capability["source_binding"]
    track_id = int(source["track_id"])
    bound = context.bound.bouts.bout_sources[track_id]
    events = np.asarray(bound.events.bouts)
    required = {
        "candidate_id",
        "signal_id",
        "track_id",
        "bout_id",
        "start_frame",
        "end_frame",
        "duration_s",
        "path_length_mm",
        "net_displacement_mm",
        "mean_speed_mm_s",
        "peak_physical_speed_mm_s",
    }
    if not required.issubset(events.dtype.names or ()):
        _fail(
            "Canonical swim-bout source lacks required physical event fields: "
            f"{sorted(required - set(events.dtype.names or ()))!r}."
        )
    if not events.size:
        return [], "complete-no-events"
    order = np.argsort(np.asarray(events["bout_id"], dtype=np.int64), kind="stable")
    rows: list[dict[str, Any]] = []
    child_common = {
        **context.bundle_common,
        "source_child_key": "canonical_swim_bouts",
        "source_run_path": source["run_path"],
        "source_manifest_sha256": source["source_array_manifest_sha256"],
        "source_payload_sha256": source["bout_content_sha256"],
        "source_receipt_sha256": source["payload_sha256"],
    }
    for source_index in order:
        item = events[int(source_index)]
        net = float(item["net_displacement_mm"])
        path = float(item["path_length_mm"])
        rows.append(
            {
                **child_common,
                "swim_bout_run_path": source["run_path"],
                "swim_bout_lineage_sha256": source["payload_sha256"],
                "track_id": track_id,
                "source_signal_id": int(item["signal_id"]),
                "bout_id": int(item["bout_id"]),
                "bout_row_id": int(source_index),
                "start_acquisition_frame_id": int(item["start_frame"]),
                "end_acquisition_frame_id": int(item["end_frame"]),
                "duration_s": float(item["duration_s"]),
                "path_length_mm": path,
                "net_displacement_mm": net,
                "mean_speed_mm_s": float(item["mean_speed_mm_s"]),
                "peak_speed_mm_s": float(item["peak_physical_speed_mm_s"]),
                "tortuosity": path / net if net > 1e-6 else float("nan"),
            }
        )
    return rows, None


def _bout_metric_samples(
    context: _CoreBehaviorContext,
    *,
    levels: tuple[str, ...],
) -> ValidatedBehaviorBatchSource:
    source = context.bound.bout_kinematics
    if source is None:
        _fail("Bout-metric projection requires the new bout-kinematics profile.")
    capability = context.capability_binding(BOUT_KINEMATICS_CAPABILITY)
    binding = _mapping(capability["source_binding"], field="bout source binding")
    if _plain(binding) != _plain(source.binding):
        _fail("Bout-metric source binding changed after admission.")
    projection = _mapping(
        capability["projection_contract"], field="bout projection contract"
    )

    def batches() -> Iterator[Mapping[str, Any]]:
        for level in levels:
            records = np.asarray(source.records_by_level[level])
            native_level = "heading" if level.startswith("heading_") else level
            dtype = source_dtype(native_level)
            if records.dtype != dtype:
                _fail(f"Bout-metric {level} dtype differs from the pinned contract.")
            for start in range(0, len(records), context.row_group_rows):
                chunk = records[start : start + context.row_group_rows]
                count = len(chunk)
                columns: dict[str, Any] = {
                    **context.common_columns(count),
                    "source_binding_sha256": _repeat(binding["payload_sha256"], count),
                    "projection_contract_sha256": _repeat(
                        projection["payload_sha256"], count
                    ),
                    "source_bout_kinematics_run": _repeat(binding["run_name"], count),
                    "source_bout_kinematics_path": _repeat(binding["run_path"], count),
                    "source_array_manifest_sha256": _repeat(
                        binding["source_array_manifest_sha256"], count
                    ),
                    "source_metric_content_sha256": _repeat(
                        binding["content_sha256_by_level"][level], count
                    ),
                    "track_id": _repeat(binding["source_track_id"], count),
                    "source_signal_id": _repeat(binding["source_signal_id"], count),
                }
                if level.startswith("heading_"):
                    columns["heading_level"] = _repeat(level, count)
                for name in dtype.names or ():
                    values = chunk[name]
                    columns[source_field_name(name)] = (
                        [
                            decode_null_terminated_text(value, errors="strict")
                            for value in values
                        ]
                        if values.dtype.kind == "S"
                        else values
                    )
                yield columns

    return _batch_source(batches())


def _bout_movement_metrics(
    context: _CoreBehaviorContext,
) -> ValidatedBehaviorBatchSource:
    return _bout_metric_samples(context, levels=("movement",))


def _bout_heading_metrics(
    context: _CoreBehaviorContext,
) -> ValidatedBehaviorBatchSource:
    return _bout_metric_samples(context, levels=("heading_raw", "heading_smoothed"))


def _bout_eye_gaze_metrics(
    context: _CoreBehaviorContext,
) -> ValidatedBehaviorBatchSource:
    return _bout_metric_samples(context, levels=("eye_gaze",))


_PRODUCERS: Mapping[str, Callable[[_CoreBehaviorContext], Any]] = MappingProxyType(
    {
        KINEMATICS_SAMPLES_TABLE: _kinematics_samples,
        SUBJECT_BODY_FRAME_SAMPLES_TABLE: _subject_body_frame_samples,
        EYE_TRACE_SAMPLES_TABLE: _eye_trace_samples,
        TAIL_TRACE_SAMPLES_TABLE: _tail_trace_samples,
        "canonical_swim_bouts": _canonical_swim_bouts,
    }
)

_BOUT_PRODUCERS: Mapping[str, Callable[[_CoreBehaviorContext], Any]] = MappingProxyType(
    {
        BOUT_MOVEMENT_TABLE: _bout_movement_metrics,
        BOUT_HEADING_TABLE: _bout_heading_metrics,
        BOUT_EYE_GAZE_TABLE: _bout_eye_gaze_metrics,
    }
)


def _build_row_extractors(
    producers: Mapping[str, Callable[[_CoreBehaviorContext], Any]],
) -> Mapping[str, Callable[..., Any]]:

    cache = _LastCoreBehaviorContext()

    def wrap(producer: Callable[[_CoreBehaviorContext], Any]) -> Callable[..., Any]:
        def extract(
            plan: Mapping[str, Any],
            membership_member: Mapping[str, Any],
            bundle_member: Mapping[str, Any],
        ) -> Any:
            return producer(cache.get(plan, membership_member, bundle_member))

        return extract

    return MappingProxyType(
        {name: wrap(producer) for name, producer in producers.items()}
    )


def build_core_behavior_row_extractors() -> Mapping[str, Callable[..., Any]]:
    """Return the frozen five-grain v1/v2 extractor roster."""

    return _build_row_extractors(_PRODUCERS)


def build_core_behavior_bout_row_extractors() -> Mapping[str, Callable[..., Any]]:
    """Return the additive bout-metric profile without changing older rosters."""

    return _build_row_extractors({**_PRODUCERS, **_BOUT_PRODUCERS})


__all__ = [
    "CoreBehaviorExportAdapterError",
    "build_core_behavior_bout_row_extractors",
    "build_core_behavior_row_extractors",
]
