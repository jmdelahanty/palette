"""Rebind chaser position facts to one selected core-motion authority.

The input chaser-relative publication is used only as a strict carrier of
chaser position, identity, occurrence, trial, and controller evidence.  Its
historical fish-position and body-frame authorities are deliberately ignored.
The adapter feeds the existing relative-frame computation, schema, and atomic
publisher; it does not introduce another scientific run or export surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CROSS_GRAIN_JOIN_AUTHORITY,
    KINEMATICS_SAMPLES_CAPABILITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .chaser_relative_frame import (
    AcquisitionFrameKeys,
    BodyFrameInput,
    ChaserObservations,
    ChaserRelativeFrameInput,
    CoordinatePolicy,
    ProviderSourceAuthority,
    ScalePolicy,
    TimingPolicy,
    compute_chaser_relative_frame,
)
from .chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
    require_chaser_relative_frame_source_handle,
)
from .chaser_relative_frame_storage import (
    ChaserRelativeFramePublicationContext,
    PreparedChaserRelativeFrame,
    prepare_chaser_relative_frame,
    validate_prepared_chaser_relative_frame,
)
from .chaser_proxy_relative_frame_adapter import (
    PreparedProxyRelativeFrame,
    prepare_proxy_relative_frame,
)
from .core_authority_roster import bind_core_motion_and_bouts_from_roster
from .core_motion_source_handle import (
    CoreMotionTrackSourceHandle,
    bind_core_motion_track_source_handle,
    require_core_motion_track_source_handle,
)
from .core_subject_body_frame_source_handle import (
    CoreSubjectBodyFrameSourceHandle,
    bind_core_subject_body_frame_source_handle,
    require_core_subject_body_frame_source_handle,
)
from .provider_recording_timing_authority import (
    ProviderRecordingTimingAuthorityError,
    load_provider_recording_timing_authority,
)

CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_ID = "palette.chaser.core_relative_frame_adapter"
CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_VERSION = 1
CORE_CHASER_RELATIVE_CONSUMER_ID = "palette.chaser.core_relative_frame.v1"
CORE_CHASER_RELATIVE_PROFILE_ID = "core_roster_chaser_relative_frame_v1"
CORE_CHASER_ROW_AXIS_POLICY_ID = "core_track_to_chaser_frame_exact_join_v1"
CORE_CHASER_TEMPORAL_SELECTION_ID = "core_track_chaser_frame_intersection_v1"


class CoreChaserRelativeFrameAdapterError(ValueError):
    """The selected core track cannot safely replace historical fish authority."""


def _fail(message: str) -> None:
    raise CoreChaserRelativeFrameAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be one exact object.")
    return value


def _context_record(
    context: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    value = context.get(name)
    if value is None:
        if required:
            _fail(f"Chaser source context lacks {name!r}.")
        return None
    envelope = _mapping(value, label=f"chaser context {name}")
    if set(envelope) != {"record", "sha256"}:
        _fail(f"Chaser context {name!r} is not a record-plus-digest envelope.")
    record = _plain(_mapping(envelope.get("record"), label=f"{name} record"))
    if canonical_json_sha256(record) != envelope.get("sha256"):
        _fail(f"Chaser context {name!r} digest is stale.")
    return record


def _same_across_chasers(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2:
        _fail(f"{label} must have frame and chaser axes.")
    if array.shape[0] and not np.all(array == array[:, :1]):
        _fail(f"{label} differs across chasers for one acquisition frame.")
    return np.array(array[:, 0], copy=True)


def _source_authority(
    record: Mapping[str, Any],
    *,
    recording_id: str,
    coordinate_authority_id: str,
    scale_authority_id: str,
    timing_authority_id: str,
    row_axis_authority_id: str,
    row_axis_authority_digest: str,
) -> ProviderSourceAuthority:
    if record.get("recording_id") != recording_id:
        _fail("Chaser position authority belongs to another recording.")
    expected = {
        "coordinate_authority_id": coordinate_authority_id,
        "scale_authority_id": scale_authority_id,
        "timing_authority_id": timing_authority_id,
    }
    for field_name, expected_value in expected.items():
        if record.get(field_name) != expected_value:
            _fail(f"Chaser position authority differs at {field_name!r}.")
    return ProviderSourceAuthority(
        recording_id=recording_id,
        source_authority_id=str(record["source_authority_id"]),
        source_digest=str(record["source_digest"]),
        provider_id=str(record["provider_id"]),
        provider_digest=str(record["provider_digest"]),
        coordinate_authority_id=coordinate_authority_id,
        scale_authority_id=scale_authority_id,
        timing_authority_id=timing_authority_id,
        row_axis_authority_id=row_axis_authority_id,
        row_axis_authority_digest=row_axis_authority_digest,
    )


_COMPOUND_BODY_KEY_DTYPE = np.dtype(
    [("acquisition_frame", "<i8"), ("instance_key", "<u8")]
)


def _compound_body_keys(
    frames: np.ndarray,
    instance_keys: np.ndarray,
) -> np.ndarray:
    if frames.ndim != 1 or instance_keys.shape != frames.shape:
        _fail("Body-frame compound join keys do not share one row axis.")
    result = np.empty(frames.shape, dtype=_COMPOUND_BODY_KEY_DTYPE)
    result["acquisition_frame"] = frames
    result["instance_key"] = instance_keys
    return result


def _project_core_body_frame(
    *,
    core: CoreMotionTrackSourceHandle,
    body: CoreSubjectBodyFrameSourceHandle,
    core_rows: np.ndarray,
    frame_keys: AcquisitionFrameKeys,
    coordinate_authority_id: str,
    scale_authority_id: str,
    timing_authority_id: str,
) -> tuple[BodyFrameInput, dict[str, Any]]:
    """Join canonical body observations by exact frame and instance identity."""

    source_instances = np.asarray(core.array("source_instance_key"))
    if source_instances.shape != (
        core.sample_count,
    ) or source_instances.dtype.names != ("valid", "instance_key"):
        _fail("Core track has no canonical nullable source-instance identity.")
    query_valid = np.asarray(source_instances["valid"], dtype=bool)[core_rows]
    query_instances = np.asarray(source_instances["instance_key"], dtype=np.uint64)[
        core_rows
    ]
    if np.any((~query_valid) & (query_instances != 0)):
        _fail("Core track nullable instance keys violate canonical zero fill.")

    source_frames = np.asarray(body.frame_indices, dtype=np.int64)
    source_instances = np.asarray(body.instance_keys, dtype=np.uint64)
    if source_frames.shape != (body.row_count,) or source_instances.shape != (
        body.row_count,
    ):
        _fail("Subject body-frame identity arrays do not share one source row axis.")
    source_keys = _compound_body_keys(source_frames, source_instances)
    order = np.argsort(source_keys, kind="stable")
    ordered = source_keys[order]
    if ordered.size > 1 and np.any(ordered[1:] == ordered[:-1]):
        _fail(
            "Subject body-frame source contains duplicate frame/instance keys; "
            "implicit observation selection is prohibited."
        )

    query_keys = _compound_body_keys(
        np.asarray(frame_keys.acquisition_frame_id, dtype=np.int64),
        query_instances,
    )
    insertion = np.searchsorted(ordered, query_keys)
    present = query_valid & (insertion < ordered.size)
    candidate_rows = np.flatnonzero(present)
    if candidate_rows.size:
        present[candidate_rows] &= (
            ordered[insertion[candidate_rows]] == query_keys[candidate_rows]
        )
    body_source_rows = np.full(query_keys.size, -1, dtype=np.int64)
    body_source_rows[present] = order[insertion[present]]

    source_origin = np.asarray(body.origin_xy, dtype=np.float64)
    source_forward = np.asarray(body.forward_axis_xy, dtype=np.float64)
    source_left = np.asarray(body.left_axis_xy, dtype=np.float64)
    source_axis_valid = np.asarray(body.axis_valid, dtype=bool)
    if (
        source_origin.shape != (body.row_count, 2)
        or source_forward.shape != (body.row_count, 2)
        or source_left.shape != (body.row_count, 2)
        or source_axis_valid.shape != (body.row_count,)
    ):
        _fail("Subject body-frame geometry does not share its identity row axis.")
    source_finite = (
        np.isfinite(source_origin).all(axis=1)
        & np.isfinite(source_forward).all(axis=1)
        & np.isfinite(source_left).all(axis=1)
    )
    if not np.array_equal(source_axis_valid, source_finite):
        _fail("Subject body-frame validity differs from its finite geometry.")

    origin = np.full((query_keys.size, 2), np.nan, dtype=np.float64)
    forward = np.full((query_keys.size, 2), np.nan, dtype=np.float64)
    left = np.full((query_keys.size, 2), np.nan, dtype=np.float64)
    axis_valid = np.zeros(query_keys.size, dtype=bool)
    if np.any(present):
        selected_rows = body_source_rows[present]
        selected_valid = source_axis_valid[selected_rows]
        selected_query_rows = np.flatnonzero(present)
        valid_query_rows = selected_query_rows[selected_valid]
        valid_source_rows = selected_rows[selected_valid]
        origin[valid_query_rows] = source_origin[valid_source_rows]
        forward[valid_query_rows] = source_forward[valid_source_rows]
        left[valid_query_rows] = source_left[valid_source_rows]
        axis_valid[valid_query_rows] = True

    authority = ProviderSourceAuthority(
        recording_id=core.recording_id,
        source_authority_id=body.run_path,
        source_digest=body.publication_manifest_sha256,
        provider_id="core_authority_roster",
        provider_digest=core.core_authority_roster_sha256,
        coordinate_authority_id=coordinate_authority_id,
        scale_authority_id=scale_authority_id,
        timing_authority_id=timing_authority_id,
        row_axis_authority_id=frame_keys.row_axis_authority_id,
        row_axis_authority_digest=frame_keys.row_axis_authority_digest,
    )
    projected = BodyFrameInput(
        frame_keys=frame_keys,
        origin_xy=origin,
        forward_axis_xy=forward,
        left_axis_xy=left,
        axis_valid=axis_valid,
        source_row_index=body_source_rows,
        authority=authority,
    )
    projection_record = {
        "schema_id": "palette.chaser_relative_frame.core_body_frame_projection",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "policy_id": "core_motion_to_subject_body_exact_compound_join_v1",
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "core_authority_consumption_receipt_sha256": core.consumption_receipt[
            "record_sha256"
        ],
        "source_body_frame_run_path": body.run_path,
        "source_body_frame_manifest_sha256": body.publication_manifest_sha256,
        "source_body_frame_binding_sha256": body.source_binding_sha256,
        "source_body_frame_row_identity_sha256": body.row_identity_sha256,
        "source_body_frame_record_sha256": body.body_frame_record_sha256,
        "relative_row_axis_authority_id": frame_keys.row_axis_authority_id,
        "relative_row_axis_authority_sha256": frame_keys.row_axis_authority_digest,
        "join_keys": [
            "recording_id",
            "source_acquisition_frame_index",
            "source_instance_key",
        ],
        "cardinality": ("zero_or_one_motion_row_to_zero_or_one_body_observation"),
        "requires_source_instance_key_valid": True,
        "source_acquisition_frame_sha256": array_values_sha256(source_frames),
        "source_instance_key_sha256": array_values_sha256(source_instances),
        "query_acquisition_frame_sha256": array_values_sha256(
            frame_keys.acquisition_frame_id
        ),
        "query_instance_key_sha256": array_values_sha256(query_instances),
        "query_instance_key_valid_sha256": array_values_sha256(query_valid),
        "projected_body_source_row_index_sha256": array_values_sha256(body_source_rows),
        "source_body_row_count": body.row_count,
        "relative_frame_count": int(query_keys.size),
        "query_instance_key_invalid_count": int(np.count_nonzero(~query_valid)),
        "exact_source_row_count": int(np.count_nonzero(present)),
        "missing_source_row_count": int(np.count_nonzero(~present)),
        "present_invalid_axis_count": int(np.count_nonzero(present & ~axis_valid)),
        "valid_axis_count": int(np.count_nonzero(axis_valid)),
        "missing_source_row_semantics": "explicit_minus_one_no_interpolation",
        "present_invalid_axis_semantics": "source_identity_retained_geometry_nan",
        "duplicate_source_key_policy": "prohibited_fail_closed",
        "interpolation": "prohibited",
        "motion_heading_fallback": "prohibited",
        "neighboring_body_frame_fallback": "prohibited",
    }
    return projected, projection_record


@dataclass(frozen=True)
class _ChaserFactsView:
    """Validated chaser-only facts from a persisted or transient carrier."""

    analysis_zarr_path: Path
    run_path: str
    recording_id: str
    selector_eligible: bool
    n_frames: int
    n_chasers: int
    n_rows: int
    run_manifest: Mapping[str, Any]
    source_authorities: Mapping[str, Any]
    context: Mapping[str, Any]
    identity_registries: Mapping[str, Any]
    base_arrays: Mapping[str, np.ndarray]
    manifest_sha256: str
    verification_digest: str

    def base_frame_chaser(self, name: str) -> np.ndarray:
        try:
            values = np.asarray(self.base_arrays[name])
        except KeyError as exc:
            raise CoreChaserRelativeFrameAdapterError(
                f"Chaser facts lack required array {name!r}."
            ) from exc
        if values.ndim == 0 or values.shape[0] != self.n_rows:
            _fail(f"Chaser facts array has another flat row axis: {name!r}.")
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])


def _facts_from_persisted_source(
    source: ChaserRelativeFrameSourceHandle,
) -> _ChaserFactsView:
    chaser = require_chaser_relative_frame_source_handle(source)
    return _ChaserFactsView(
        analysis_zarr_path=chaser.analysis_zarr_path,
        run_path=chaser.run_path,
        recording_id=chaser.recording_id,
        selector_eligible=chaser.selector_eligible,
        n_frames=chaser.n_frames,
        n_chasers=chaser.n_chasers,
        n_rows=chaser.n_rows,
        run_manifest=chaser.run_manifest,
        source_authorities=chaser.source_authorities,
        context=chaser.context,
        identity_registries=chaser.identity_registries,
        base_arrays=chaser.base_arrays,
        manifest_sha256=chaser.manifest_sha256,
        verification_digest=chaser.verification_digest,
    )


def _facts_from_prepared_proxy(
    source: PreparedProxyRelativeFrame,
    *,
    analysis_zarr: str | Path,
) -> _ChaserFactsView:
    if type(source) is not PreparedProxyRelativeFrame:
        raise TypeError("source must be one validated PreparedProxyRelativeFrame.")
    prepared = source.prepared
    validation = validate_prepared_chaser_relative_frame(prepared)
    if prepared.body_arrays is not None or any(
        value is not None
        for value in (
            source.body_frame_run_path,
            source.body_frame_manifest_sha256,
            source.body_frame_projection_sha256,
            source.body_frame_projection_record,
        )
    ):
        _fail("A transient proxy carrier must omit legacy body-frame geometry.")
    manifest = _mapping(prepared.manifest, label="prepared proxy manifest")
    context = _mapping(manifest.get("context"), label="prepared proxy context")
    publication = _context_record(
        context,
        "acquisition_projection_publication",
    )
    assert publication is not None
    expected_proxy = {
        "run_path": source.proxy_run_path,
        "manifest_sha256": source.proxy_manifest_sha256,
    }
    observed_proxy = {
        "run_path": publication.get("run_path"),
        "manifest_sha256": publication.get("manifest_sha256"),
    }
    if observed_proxy != expected_proxy:
        _fail("Prepared proxy facts bind another proxy publication.")
    if (
        validation.get("payload_digest") != prepared.payload_digest
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        _fail("Prepared proxy facts are not one validated selector-ineligible payload.")
    dimensions = _mapping(manifest.get("dimensions"), label="prepared proxy dimensions")
    n_frames = dimensions.get("n_frames")
    n_chasers = dimensions.get("n_chasers")
    n_rows = dimensions.get("n_rows")
    if (
        type(n_frames) is not int
        or type(n_chasers) is not int
        or type(n_rows) is not int
        or n_frames < 0
        or n_chasers <= 0
        or n_rows != n_frames * n_chasers
        or n_rows != prepared.dimensions.n_rows
    ):
        _fail("Prepared proxy frame/chaser dimensions are inconsistent.")
    return _ChaserFactsView(
        analysis_zarr_path=Path(analysis_zarr).expanduser().resolve(),
        run_path=source.proxy_run_path,
        recording_id=str(manifest["recording_id"]),
        selector_eligible=False,
        n_frames=n_frames,
        n_chasers=n_chasers,
        n_rows=n_rows,
        run_manifest=manifest,
        source_authorities=_mapping(
            manifest.get("source_authorities"),
            label="prepared proxy source authorities",
        ),
        context=context,
        identity_registries=_mapping(
            manifest.get("identity_registries"),
            label="prepared proxy identity registries",
        ),
        base_arrays=prepared.base_arrays,
        manifest_sha256=source.proxy_manifest_sha256,
        verification_digest=prepared.payload_digest,
    )


@dataclass(frozen=True)
class PreparedCoreChaserRelativeFrame:
    """Existing-schema relative-frame payload plus its core authority evidence."""

    prepared: PreparedChaserRelativeFrame
    core_authority_record: Mapping[str, Any]
    core_authority_record_sha256: str
    source_chaser_run_path: str
    source_chaser_manifest_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_ID,
            "schema_version": CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_VERSION,
            "core_authority_record_sha256": self.core_authority_record_sha256,
            "source_chaser_run_path": self.source_chaser_run_path,
            "source_chaser_manifest_sha256": self.source_chaser_manifest_sha256,
            "prepared_manifest_sha256": self.prepared.payload_digest,
            "publication_surface": "analysis/chaser_relative_frame_runs",
            "selector_eligible": False,
            "selection": "none",
        }


def _prepare_core_chaser_relative_frame(
    core_motion: CoreMotionTrackSourceHandle,
    core_body_frame: CoreSubjectBodyFrameSourceHandle,
    chaser: _ChaserFactsView,
) -> PreparedCoreChaserRelativeFrame:
    """Prepare a core-bound candidate from one validated chaser-facts view."""

    core = require_core_motion_track_source_handle(core_motion)
    body = require_core_subject_body_frame_source_handle(core_body_frame)
    if chaser.selector_eligible is not False:
        _fail("Core rebinding requires one exact selector-ineligible chaser source.")
    if core.analysis_zarr_path != chaser.analysis_zarr_path:
        _fail("Core and chaser sources do not belong to one analysis Zarr.")
    if core.recording_id != chaser.recording_id:
        _fail("Core and chaser sources name different recordings.")
    if (
        body.analysis_zarr_path != core.analysis_zarr_path
        or body.recording_id != core.recording_id
        or body.core_authority_roster_sha256 != core.core_authority_roster_sha256
    ):
        _fail("Core motion and subject body frame bind different authority rosters.")
    if body.consumption_receipt != core.consumption_receipt:
        _fail("Core motion and subject body frame lack one shared admission receipt.")

    receipt = _mapping(core.consumption_receipt, label="core consumption receipt")
    if (
        receipt.get("consumer_id") != CORE_CHASER_RELATIVE_CONSUMER_ID
        or receipt.get("selected_track_id") != core.track_id
        or set(receipt.get("required_capabilities", ()))
        != {
            CROSS_GRAIN_JOIN_AUTHORITY,
            KINEMATICS_SAMPLES_CAPABILITY,
            SUBJECT_BODY_FRAME_CAPABILITY,
        }
    ):
        _fail("Core track was not admitted for this exact relative-frame consumer.")
    roster = _mapping(core.core_authority_roster, label="core authority roster")
    join = _mapping(
        roster.get("cross_grain_join_authority"),
        label="core cross-grain join authority",
    )
    manifest = _mapping(chaser.run_manifest, label="chaser source manifest")
    coordinate = _mapping(
        manifest.get("coordinate_policy"), label="chaser coordinate policy"
    )
    scale = _mapping(manifest.get("scale_policy"), label="chaser scale policy")
    timing_policy_source = _mapping(
        manifest.get("timing_policy"), label="chaser timing policy"
    )
    coordinate_id = str(coordinate.get("coordinate_authority_id"))
    scale_id = str(scale.get("scale_authority_id"))
    timing_id = str(timing_policy_source.get("timing_authority_id"))
    if (
        coordinate.get("coordinate_frame") != "source_camera_continuous_pixel_xy"
        or coordinate.get("origin") != "top_left"
        or coordinate.get("x_axis_direction") != "right"
        or coordinate.get("y_axis_direction") != "down"
        or coordinate_id != join.get("acquisition_camera_frame_ref")
    ):
        _fail("Core and chaser sources do not share one source-camera authority.")
    pixels_per_mm = scale.get("pixels_per_unit")
    if (
        scale.get("unit") != "mm"
        or isinstance(pixels_per_mm, bool)
        or not isinstance(pixels_per_mm, (int, float))
        or not math.isfinite(float(pixels_per_mm))
        or float(pixels_per_mm) <= 0
    ):
        _fail("Chaser source lacks one valid pixels-per-millimeter authority.")
    position_surface = _mapping(
        core.selected_surfaces.get("positions_mm"),
        label="core physical-position surface",
    )
    if position_surface.get("physical_authority_sha256") != scale.get("scale_digest"):
        _fail("Core and chaser positions bind different physical-scale authority.")

    source_context = _mapping(chaser.context, label="chaser source context")
    controller = _context_record(source_context, "controller_state")
    assert controller is not None
    session_timing = _mapping(
        controller.get("session_timestamp_authority"),
        label="chaser session timestamp authority",
    )
    try:
        timing = load_provider_recording_timing_authority(
            core.analysis_zarr_path,
            required=True,
            use_consolidated=True,
            expected_sha256=session_timing.get("recording_timing_authority_sha256"),
        )
    except (ProviderRecordingTimingAuthorityError, TypeError, ValueError) as exc:
        raise CoreChaserRelativeFrameAdapterError(
            f"Chaser timing cannot be bound to the core frame domain: {exc}"
        ) from exc
    if timing is None:  # pragma: no cover - required=True contract
        _fail("Chaser timing authority is absent.")
    if (
        timing.recording_id != core.recording_id
        or timing.camera_id != join.get("camera_id")
        or timing.frame_count != join.get("source_total_frames")
        or timing.source_video_metadata_sha256
        != join.get("source_video_metadata_sha256")
    ):
        _fail("Core and chaser sources bind different acquisition timing authority.")

    frames = _same_across_chasers(
        chaser.base_frame_chaser("acquisition_frame_id"),
        label="chaser acquisition frame",
    ).astype(np.int64, copy=False)
    core_frames = np.asarray(core.frame_indices, dtype=np.int64)
    if core_frames.shape != (core.sample_count,) or (
        core_frames.size > 1 and np.any(np.diff(core_frames) <= 0)
    ):
        _fail("Core track frames are not one unique increasing identity axis.")
    positions = np.searchsorted(core_frames, frames)
    if np.any(positions >= core_frames.size) or not np.array_equal(
        core_frames[positions], frames
    ):
        _fail("Every chaser frame must resolve one exact selected core-track row.")
    core_rows = positions.astype(np.int64, copy=False)

    core_positions_mm = np.asarray(core.positions_mm, dtype=np.float64)
    core_sample_valid = np.asarray(core.array("sample_valid"), dtype=bool)
    core_position_finite = np.asarray(core.array("position_finite"), dtype=bool)
    if (
        core_positions_mm.shape != (core.sample_count, 2)
        or core_sample_valid.shape != (core.sample_count,)
        or core_position_finite.shape != (core.sample_count,)
    ):
        _fail("Core position arrays do not share the selected track row axis.")
    fish_mm = core_positions_mm[core_rows]
    fish_valid = (
        core_sample_valid[core_rows]
        & core_position_finite[core_rows]
        & np.isfinite(fish_mm).all(axis=1)
    )
    fish_px = fish_mm * float(pixels_per_mm)
    fish_px = np.array(fish_px, copy=True)
    fish_px[~fish_valid] = np.nan

    timestamp_ns = _same_across_chasers(
        chaser.base_frame_chaser("timestamp_ns"),
        label="chaser session timestamp",
    ).astype(np.int64, copy=False)
    timestamp_valid = _same_across_chasers(
        chaser.base_frame_chaser("timestamp_valid"),
        label="chaser timestamp validity",
    ).astype(bool, copy=False)
    if not np.all(timestamp_valid):
        _fail("Core-bound chaser frames require exact timestamps for every row.")

    row_axis_body = {
        "schema_id": "palette.chaser_relative_frame.core_row_axis",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "policy_id": CORE_CHASER_ROW_AXIS_POLICY_ID,
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "core_motion_run_path": core.run_path,
        "core_motion_manifest_sha256": core.source_manifest_sha256,
        "track_id": core.track_id,
        "acquisition_frame_sha256": array_values_sha256(frames),
        "core_track_sample_index_sha256": array_values_sha256(core_rows),
        "source_chaser_run_path": chaser.run_path,
        "source_chaser_manifest_sha256": chaser.manifest_sha256,
        "cardinality": "one_chaser_frame_to_one_selected_core_track_sample",
        "interpolation": "prohibited",
        "fallback": "prohibited",
    }
    row_axis_digest = canonical_json_sha256(row_axis_body)
    row_axis_id = f"{core.run_path}/tracks/id_{core.track_id}/track_sample_key"
    frame_keys = AcquisitionFrameKeys(
        recording_id=core.recording_id,
        acquisition_frame_id=frames,
        track_sample_id=core_rows,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
        timestamp_ns=timestamp_ns,
    )
    body_frame_input, body_frame_projection_record = _project_core_body_frame(
        core=core,
        body=body,
        core_rows=core_rows,
        frame_keys=frame_keys,
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
    )

    chaser_authority_record = _mapping(
        chaser.source_authorities.get("chaser_position"),
        label="chaser position authority",
    )
    chaser_authority = _source_authority(
        chaser_authority_record,
        recording_id=core.recording_id,
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
    )
    fish_authority = ProviderSourceAuthority(
        recording_id=core.recording_id,
        source_authority_id=core.run_path,
        source_digest=core.source_manifest_sha256,
        provider_id="core_authority_roster",
        provider_digest=core.core_authority_roster_sha256,
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
    )
    registries = _mapping(chaser.identity_registries, label="chaser registries")
    identity_registry = _mapping(
        registries.get("chaser"), label="chaser identity registry"
    )
    role_registry = _mapping(
        registries.get("behavior_role"), label="chaser role registry"
    )
    identities = tuple(
        str(identity_registry[str(index)]) for index in range(1, chaser.n_chasers + 1)
    )
    role_codes = np.asarray(chaser.base_frame_chaser("chaser_behavior_role_code"))
    try:
        roles = np.asarray(
            [role_registry[str(int(code))] for code in role_codes.reshape(-1)],
            dtype=np.dtypes.StringDType(),
        ).reshape(role_codes.shape)
    except KeyError as exc:
        raise CoreChaserRelativeFrameAdapterError(
            "Chaser role codes are not closed by the source registry."
        ) from exc
    selection = _same_across_chasers(
        chaser.base_frame_chaser("selection_member"),
        label="chaser selection membership",
    ).astype(bool, copy=False)
    occurrence = np.asarray(
        chaser.base_frame_chaser("chaser_occurrence_member"), dtype=bool
    )
    chaser_px = np.asarray(
        chaser.base_frame_chaser("chaser_position_xy_px"), dtype=np.float64
    )
    chaser_valid = np.asarray(
        chaser.base_frame_chaser("chaser_position_valid"), dtype=bool
    )
    chaser_rows = np.asarray(
        chaser.base_frame_chaser("chaser_source_row_id"), dtype=np.int64
    )
    trial_ids = np.asarray(chaser.base_frame_chaser("trial_id"), dtype=np.int64)
    active = (
        np.asarray(chaser.base_frame_chaser("active_state_code"), dtype=np.uint8) == 1
    )
    expected_pair_shape = (chaser.n_frames, chaser.n_chasers)
    if (
        chaser_px.shape != (*expected_pair_shape, 2)
        or chaser_valid.shape != expected_pair_shape
        or chaser_rows.shape != expected_pair_shape
        or occurrence.shape != expected_pair_shape
        or trial_ids.shape != expected_pair_shape
        or active.shape != expected_pair_shape
    ):
        _fail("Chaser facts do not share the declared frame/chaser axes.")

    coordinate_policy = CoordinatePolicy(
        coordinate_authority_id=coordinate_id,
        coordinate_frame="source_camera_continuous_pixel_xy",
        policy_id=str(coordinate.get("policy_id")),
    )
    scale_policy = ScalePolicy(
        scale_authority_id=scale_id,
        scale_digest=str(scale.get("scale_digest")),
        pixels_per_unit=float(pixels_per_mm),
        unit="mm",
        policy_id=str(scale.get("policy_id")),
    )
    timing_policy = TimingPolicy(
        timing_authority_id=timing_id,
        timing_digest=str(timing_policy_source.get("timing_digest")),
        recording_id=core.recording_id,
        timestamp_field=str(timing_policy_source.get("timestamp_field")),
        policy_id=str(timing_policy_source.get("policy_id")),
    )
    inputs = ChaserRelativeFrameInput(
        frame_keys=frame_keys,
        fish_xy=fish_px,
        fish_valid=fish_valid,
        fish_source_row_index=core_rows,
        fish_authority=fish_authority,
        chasers=ChaserObservations(
            identities=identities,
            behavior_roles=roles,
            xy=chaser_px,
            valid=chaser_valid,
            source_row_index=chaser_rows,
            authority=chaser_authority,
            trial_ids=trial_ids,
            active=active,
        ),
        selection_membership=selection,
        occurrence_membership=occurrence,
        coordinate_policy=coordinate_policy,
        scale_policy=scale_policy,
        timing_policy=timing_policy,
        body_frame=body_frame_input,
    )
    result = compute_chaser_relative_frame(inputs)

    core_authority_record = {
        "schema_id": "palette.chaser_relative_frame.core_authority_binding",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "core_authority_consumption_receipt": _plain(receipt),
        "core_motion": {
            "run_path": core.run_path,
            "source_manifest_sha256": core.source_manifest_sha256,
            "source_binding_sha256": core.source_binding_sha256,
            "track_id": core.track_id,
            "row_axis_sha256": row_axis_digest,
        },
        "core_subject_body_frame": {
            "run_path": body.run_path,
            "publication_manifest_sha256": body.publication_manifest_sha256,
            "source_binding_sha256": body.source_binding_sha256,
            "row_identity_sha256": body.row_identity_sha256,
            "body_frame_record_sha256": body.body_frame_record_sha256,
            "projection_record_sha256": canonical_json_sha256(
                body_frame_projection_record
            ),
        },
        "chaser_source": {
            "run_path": chaser.run_path,
            "manifest_sha256": chaser.manifest_sha256,
            "verification_digest": chaser.verification_digest,
            "consumed_authority": "chaser_position",
            "fish_position_authority": "not_used_core_roster_selected_instead",
            "body_frame_authority": "not_used_core_roster_selected_instead",
        },
        "fish_pixel_projection": {
            "source": "core_positions_mm",
            "formula": "positions_mm * pixels_per_mm",
            "physical_authority_sha256": scale.get("scale_digest"),
        },
        "core_motion_facts_repeated": False,
        "fallback": "prohibited",
    }
    temporal_record = {
        "schema_id": "palette.chaser_relative_frame.core_temporal_selection",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "selection_id": CORE_CHASER_TEMPORAL_SELECTION_ID,
        "row_axis_authority_id": row_axis_id,
        "row_axis_authority_sha256": row_axis_digest,
        "recording_timing_authority_sha256": timing.sha256,
        "selected_frame_count": chaser.n_frames,
        "selection": "source_chaser_frames_exactly_present_in_selected_core_track",
        "fallback": "prohibited",
    }
    analysis_profile = {
        "schema_id": "palette.chaser_relative_frame.core_analysis_profile",
        "schema_version": 1,
        "recording_id": core.recording_id,
        "profile_id": CORE_CHASER_RELATIVE_PROFILE_ID,
        "core_authority_roster_sha256": core.core_authority_roster_sha256,
        "source_chaser_profile_sha256": canonical_json_sha256(
            _plain(_context_record(source_context, "analysis_profile"))
        ),
        "body_frame": "core_roster_selected_subject_body_frame",
    }
    subject = _context_record(source_context, "subject_identity")
    occurrence_record = _context_record(source_context, "chaser_occurrence")
    acquisition = _context_record(source_context, "acquisition_projection")
    assert (
        subject is not None
        and occurrence_record is not None
        and acquisition is not None
    )
    publication = _context_record(
        source_context,
        "acquisition_projection_publication",
        required=False,
    )
    arena = _context_record(source_context, "arena_geometry", required=False)
    transform = _context_record(
        source_context,
        "arena_to_source_camera_transform",
        required=False,
    )
    publication_context = ChaserRelativeFramePublicationContext(
        fish_identity=str(source_context["fish_identity"]),
        subject_identity_record=subject,
        temporal_selection_record=temporal_record,
        chaser_occurrence_record=occurrence_record,
        acquisition_projection_record=acquisition,
        acquisition_projection_publication_record=publication,
        controller_state_record=controller,
        body_frame_projection_record=body_frame_projection_record,
        analysis_profile_record=analysis_profile,
        core_authority_record=core_authority_record,
        arena_geometry_record=arena,
        arena_to_source_camera_transform_record=transform,
    )
    prepared = prepare_chaser_relative_frame(result, context=publication_context)
    core_sha = canonical_json_sha256(core_authority_record)
    if prepared.manifest["context"]["core_authority"]["sha256"] != core_sha:
        _fail("Prepared relative frame lost its core authority binding.")
    return PreparedCoreChaserRelativeFrame(
        prepared=prepared,
        core_authority_record=MappingProxyType(core_authority_record),
        core_authority_record_sha256=core_sha,
        source_chaser_run_path=chaser.run_path,
        source_chaser_manifest_sha256=chaser.manifest_sha256,
    )


def prepare_core_chaser_relative_frame(
    core_motion: CoreMotionTrackSourceHandle,
    core_body_frame: CoreSubjectBodyFrameSourceHandle,
    chaser_source: ChaserRelativeFrameSourceHandle,
) -> PreparedCoreChaserRelativeFrame:
    """Rebind one already-published chaser carrier to selected core authority."""

    return _prepare_core_chaser_relative_frame(
        core_motion,
        core_body_frame,
        _facts_from_persisted_source(chaser_source),
    )


def prepare_core_chaser_relative_frame_from_proxy(
    core_motion: CoreMotionTrackSourceHandle,
    core_body_frame: CoreSubjectBodyFrameSourceHandle,
    proxy_source: PreparedProxyRelativeFrame,
    *,
    analysis_zarr: str | Path,
) -> PreparedCoreChaserRelativeFrame:
    """Publish core-bound geometry directly from transient proxy chaser facts.

    The ordinary proxy adapter performs the established chaser coordinate and
    controller-state extraction in memory, without publishing its historical
    fish/body result.  This boundary consumes only those chaser facts and
    substitutes the roster-selected motion and body authorities before the one
    existing relative-frame publisher is invoked.
    """

    return _prepare_core_chaser_relative_frame(
        core_motion,
        core_body_frame,
        _facts_from_prepared_proxy(proxy_source, analysis_zarr=analysis_zarr),
    )


def prepare_core_proxy_chaser_relative_frame(
    analysis_zarr: str | Path,
    *,
    core_authority_roster: Mapping[str, Any],
    core_track_id: int,
    proxy_run_name: str,
    analysis_profile_path: str | Path,
    expected_recording_id: str | None = None,
    expected_proxy_manifest_sha256: str | None = None,
    expected_subject_metadata_sha256: str | None = None,
    expected_timing_authority_sha256: str | None = None,
) -> PreparedCoreChaserRelativeFrame:
    """Resolve core and proxy sources, then prepare one canonical candidate.

    Proxy coordinate/controller extraction remains in memory. Only the final
    core-bound payload reaches the existing atomic relative-frame publisher.
    """

    archive = Path(analysis_zarr).expanduser().resolve()
    bound = bind_core_motion_and_bouts_from_roster(core_authority_roster)
    if Path(bound.roster["analysis_zarr"]) != archive:
        _fail("Core authority roster belongs to another analysis Zarr.")
    if (
        expected_recording_id is not None
        and bound.roster["recording_id"] != expected_recording_id
    ):
        _fail("Core authority roster belongs to another requested recording.")
    required_capabilities = (
        CROSS_GRAIN_JOIN_AUTHORITY,
        KINEMATICS_SAMPLES_CAPABILITY,
        SUBJECT_BODY_FRAME_CAPABILITY,
    )
    motion = bind_core_motion_track_source_handle(
        bound,
        consumer_id=CORE_CHASER_RELATIVE_CONSUMER_ID,
        required_capabilities=required_capabilities,
        track_id=core_track_id,
    )
    body = bind_core_subject_body_frame_source_handle(
        bound,
        consumer_id=CORE_CHASER_RELATIVE_CONSUMER_ID,
        required_capabilities=required_capabilities,
        track_id=core_track_id,
    )
    proxy = prepare_proxy_relative_frame(
        archive,
        proxy_run_name=proxy_run_name,
        analysis_profile_path=analysis_profile_path,
        expected_recording_id=expected_recording_id,
        expected_proxy_manifest_sha256=expected_proxy_manifest_sha256,
        expected_subject_metadata_sha256=expected_subject_metadata_sha256,
        expected_timing_authority_sha256=expected_timing_authority_sha256,
        body_frame_run_name=None,
    )
    return prepare_core_chaser_relative_frame_from_proxy(
        motion,
        body,
        proxy,
        analysis_zarr=archive,
    )


__all__ = [
    "CORE_CHASER_RELATIVE_ADAPTER_SCHEMA_ID",
    "CORE_CHASER_RELATIVE_CONSUMER_ID",
    "CORE_CHASER_RELATIVE_PROFILE_ID",
    "CoreChaserRelativeFrameAdapterError",
    "PreparedCoreChaserRelativeFrame",
    "prepare_core_chaser_relative_frame",
    "prepare_core_chaser_relative_frame_from_proxy",
    "prepare_core_proxy_chaser_relative_frame",
]
