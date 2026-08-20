"""Bind one published controller-state proxy to the common camera frame.

The proxy intentionally preserves Citrus arena-coordinate samples.  This
adapter reopens the exact native provider candidate named by that proxy,
revalidates its coordinate authorities, and applies the published typed
arena -> selected canvas -> source-camera transform chain.  It never relabels
arena coordinates, uses stimulus timestamps as camera timestamps, follows a
mutable proxy selector, or makes the resulting exploratory candidate
selector-eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.chaser_behavior import resolve_configured_chaser_behaviors
from fisheye.analysis.chaser_profiles import (
    ChaserAnalysisProfile,
    load_chaser_analysis_profile,
)
from fisheye.analysis.stimulus_response_coordinate_authority import (
    StimulusResponseCoordinateAuthority,
    load_stimulus_response_coordinate_authority,
)
from fisheye.analysis_workflows.chaser_input_provenance_proxy import PROXY_POLICY_ID
from fisheye.analysis_workflows.chaser_input_provenance_proxy_source_handle import (
    ChaserInputProvenanceProxySourceHandle,
    load_chaser_input_provenance_proxy_source_handle,
)
from fisheye.analysis_workflows.chaser_relative_frame import (
    AcquisitionFrameKeys,
    ChaserObservations,
    ChaserRelativeFrameInput,
    CoordinatePolicy,
    ProviderSourceAuthority,
    ScalePolicy,
    TimingPolicy,
    compute_chaser_relative_frame,
)
from fisheye.analysis_workflows.chaser_relative_frame_storage import (
    ChaserRelativeFramePublicationContext,
    PreparedChaserRelativeFrame,
    prepare_chaser_relative_frame,
)
from fisheye.analysis_workflows.provider_chaser_stimulus_source_handle import (
    ProviderChaserStimulusSourceHandle,
    load_provider_chaser_stimulus_source_handle,
)
from fisheye.analysis_workflows.provider_recording_timing_authority import (
    load_provider_recording_timing_authority,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.stimulus_physical_coordinate import (
    load_stimulus_physical_coordinate_authority,
    require_bound_stimulus_physical_coordinate_authority,
)
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root


ADAPTER_SCHEMA_ID = "palette.chaser_proxy_relative_frame_adapter"
ADAPTER_SCHEMA_VERSION = 1
ROW_AXIS_POLICY_ID = "proxy_represented_input_acquisition_frames_v1"
CHASER_IDENTITY_POLICY_ID = "stimulus_run_scoped_chaser_index_v1"
CHASER_OCCURRENCE_POLICY_ID = "native_sample_declared_chaser_axis_v1"
TEMPORAL_SELECTION_ID = "all_proxy_represented_input_acquisition_frames_v1"
COORDINATE_POLICY_ID = "typed_arena_to_source_camera_y_down_v1"
TIMING_POLICY_ID = "acquisition_frame_domain_without_camera_timestamps_v1"


class ChaserProxyRelativeFrameAdapterError(ValueError):
    """Raised when the proxy cannot be rebound without inference."""


def _fail(message: str) -> None:
    raise ChaserProxyRelativeFrameAdapterError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _record(schema_id: str, recording_id: str, **values: Any) -> dict[str, Any]:
    record = {
        "schema_id": schema_id,
        "schema_version": 1,
        "recording_id": recording_id,
        **values,
    }
    # Fail here, before the storage boundary, if a producer record cannot be
    # represented as strict readable JSON.
    canonical_json_sha256(record)
    return record


def _exact_run_name(path: object, *, parent: str, field: str) -> str:
    if type(path) is not str:
        _fail(f"{field} must be one exact run path.")
    prefix = f"{parent}/"
    if not path.startswith(prefix) or "/" in path[len(prefix) :] or not path[len(prefix) :]:
        _fail(f"{field} must name one exact child of {parent!r}.")
    return path[len(prefix) :]


def _pointer(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one coordinate-record pointer.")
    ref = value.get("record_ref")
    digest = value.get("record_sha256")
    if (
        type(ref) is not str
        or not ref
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        _fail(f"{field} is not an exact record_ref/record_sha256 pointer.")
    return {"record_ref": ref, "record_sha256": digest}


def _protocol_payload(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return _plain(value)
    normalized = normalize_attr(value)
    if isinstance(normalized, Mapping):
        return _plain(normalized)
    if isinstance(normalized, str) and normalized.strip():
        try:
            decoded = json.loads(normalized)
        except json.JSONDecodeError as exc:
            _fail(f"Exact stimulus protocol_json is malformed: {exc}.")
        if isinstance(decoded, Mapping):
            return decoded
    _fail("Exact stimulus run has no readable protocol_json object.")


def require_proxy_native_binding(
    proxy: ChaserInputProvenanceProxySourceHandle,
    native: ProviderChaserStimulusSourceHandle,
) -> None:
    projection = proxy.acquisition_projection_record
    expected_authority_digest = canonical_json_sha256(_plain(native.source_authority))
    expected = {
        "recording_id": native.recording_id,
        "source_run_path": native.run_path,
        "source_manifest_sha256": native.manifest_sha256,
        "source_verification_digest": native.verification_digest,
        "source_authority_digest": expected_authority_digest,
    }
    observed = {
        "recording_id": proxy.recording_id,
        "source_run_path": projection.get("source_run_path"),
        "source_manifest_sha256": projection.get("source_manifest_sha256"),
        "source_verification_digest": projection.get("source_verification_digest"),
        "source_authority_digest": projection.get("source_authority_digest"),
    }
    if observed != expected:
        _fail("Published proxy does not bind the exact reopened native provider candidate.")
    if projection.get("policy_id") != PROXY_POLICY_ID:
        _fail("Published proxy uses an unsupported projection policy.")


def _frame_source_rows(
    proxy: ChaserInputProvenanceProxySourceHandle,
    native: ProviderChaserStimulusSourceHandle,
) -> np.ndarray:
    selected = np.asarray(proxy.selected, dtype=bool)
    selected_rows = np.asarray(proxy.selected_native_sample_row_index, dtype=np.int64)
    candidate_rows = np.asarray(proxy.array("candidate_native_sample_row_index"), dtype=np.int64)
    offsets = np.asarray(proxy.candidate_offsets, dtype=np.int64)
    counts = np.asarray(proxy.candidate_sample_count, dtype=np.int64)
    if offsets.shape != (selected.size + 1,) or counts.shape != selected.shape:
        _fail("Proxy candidate ragged axes are inconsistent.")
    if np.any(counts <= 0) or offsets[0] != 0 or offsets[-1] != candidate_rows.size:
        _fail("Every represented proxy frame must retain at least one native candidate.")
    rows = selected_rows.copy()
    missing = ~selected
    rows[missing] = candidate_rows[offsets[:-1][missing]]
    if np.any(rows < 0) or np.any(rows >= native.dimensions.n_samples):
        _fail("Proxy frame-to-native-sample rows leave the exact native source.")
    frames = np.asarray(proxy.acquisition_frame_index, dtype=np.int64)
    if not np.array_equal(native.source_acquisition_frame_index[rows], frames):
        _fail("Proxy frame-to-native-sample rows do not preserve acquisition keys.")
    if np.any(selected_rows[selected] != rows[selected]) or np.any(selected_rows[missing] != -1):
        _fail("Proxy selected native sample row semantics are inconsistent.")
    return rows


def _configured_chasers(
    *,
    root: Any,
    stimulus_run_name: str,
    native: ProviderChaserStimulusSourceHandle,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
    run = root[f"analysis/stimulus_runs/{stimulus_run_name}"]
    payload = _protocol_payload(run.attrs.get("protocol_json"))
    try:
        configured = resolve_configured_chaser_behaviors(payload)
    except ValueError as exc:
        _fail(f"Configured chaser roles cannot be resolved: {exc}.")
    by_index = {item.chaser_index: item for item in configured}
    indices = tuple(int(value) for value in native.chaser_index.tolist())
    if len(by_index) != len(configured) or set(by_index) != set(indices):
        _fail("Protocol chaser indices do not exactly match the native provider axis.")
    identities = tuple(
        f"{stimulus_run_name}:chaser_index:{index}" for index in indices
    )
    role_values = tuple(by_index[index].behavior_class for index in indices)
    width = max(1, *(len(value) for value in role_values))
    roles = np.asarray(role_values, dtype=f"<U{width}")
    protocol_sha256 = canonical_json_sha256(_plain(payload))
    occurrence_record = _record(
        "palette.chaser_relative_frame.chaser_occurrence_binding",
        native.recording_id,
        occurrence_policy_id=CHASER_OCCURRENCE_POLICY_ID,
        chaser_identity_policy_id=CHASER_IDENTITY_POLICY_ID,
        source_stimulus_run_path=native.source_stimulus_run_path,
        source_protocol_sha256=protocol_sha256,
        chasers=[
            {
                "chaser_index": index,
                "identity": identity,
                "behavior_role": role,
            }
            for index, identity, role in zip(indices, identities, role_values)
        ],
        semantics=(
            "membership means the exact native sample exposes the declared "
            "chaser axis; it does not claim display presentation or visibility"
        ),
    )
    return identities, roles, occurrence_record


def _coordinate_records(
    coordinate: StimulusResponseCoordinateAuthority,
    *,
    recording_id: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    lineage = _plain(coordinate.record)
    arena_record = _record(
        "palette.chaser_relative_frame.arena_geometry_binding",
        recording_id,
        source_stimulus_run_ref=lineage["source_stimulus_run_ref"],
        arena_geometry=lineage["arena_geometry"],
        arena_frame=lineage["arena_frame"],
    )
    transform_record = _record(
        "palette.chaser_relative_frame.arena_to_source_camera_transform",
        recording_id,
        transform_policy_id=COORDINATE_POLICY_ID,
        from_coordinate_space="arena_relative_canvas_px",
        to_coordinate_space="source_camera_image_px",
        stimulus_frame_transform_manifest=lineage[
            "stimulus_frame_transform_manifest"
        ],
        selected_calibration=lineage["selected_calibration"],
        selected_canvas_frame=lineage["selected_canvas_frame"],
        source_camera_frame=lineage["source_camera_frame"],
        ordered_transform_chain=lineage[
            "arena_to_source_camera_transform_chain"
        ],
        no_reflection_or_heuristic_flip=True,
    )
    camera_pointer = _pointer(lineage["source_camera_frame"], field="source_camera_frame")
    physical_pointer = _pointer(
        lineage["physical_authority"]["physical_frame"],
        field="physical_frame",
    )
    return (
        arena_record,
        transform_record,
        camera_pointer["record_ref"],
        physical_pointer["record_ref"],
        physical_pointer["record_sha256"],
    )


@dataclass(frozen=True)
class PreparedProxyRelativeFrame:
    """Prepared candidate plus compact diagnostic bindings for the caller."""

    prepared: PreparedChaserRelativeFrame
    proxy_run_path: str
    proxy_manifest_sha256: str
    native_run_path: str
    native_manifest_sha256: str
    coordinate_lineage_sha256: str
    timing_authority_sha256: str
    subject_metadata_sha256: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": ADAPTER_SCHEMA_ID,
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "status": "prepared_selector_ineligible",
            "proxy_run_path": self.proxy_run_path,
            "proxy_manifest_sha256": self.proxy_manifest_sha256,
            "native_run_path": self.native_run_path,
            "native_manifest_sha256": self.native_manifest_sha256,
            "coordinate_lineage_sha256": self.coordinate_lineage_sha256,
            "timing_authority_sha256": self.timing_authority_sha256,
            "subject_metadata_sha256": self.subject_metadata_sha256,
            "prepared_manifest_sha256": self.prepared.payload_digest,
            "selector_eligible": False,
            "selection": "none",
        }


def prepare_proxy_relative_frame(
    analysis_zarr: str | Path,
    *,
    proxy_run_name: str,
    analysis_profile_path: str | Path,
    expected_recording_id: str | None = None,
    expected_proxy_manifest_sha256: str | None = None,
    expected_subject_metadata_sha256: str | None = None,
    expected_timing_authority_sha256: str | None = None,
) -> PreparedProxyRelativeFrame:
    """Load, rebind, compute, and prepare one exact exploratory candidate."""

    archive = Path(analysis_zarr).expanduser().resolve()
    proxy = load_chaser_input_provenance_proxy_source_handle(
        archive,
        run_name=proxy_run_name,
        expected_recording_id=expected_recording_id,
        expected_manifest_sha256=expected_proxy_manifest_sha256,
        use_consolidated=True,
    )
    native_name = _exact_run_name(
        proxy.acquisition_projection_record["source_run_path"],
        parent="analysis/provider_chaser_distance_candidate_runs",
        field="proxy source_run_path",
    )
    native = load_provider_chaser_stimulus_source_handle(
        archive,
        run_name=native_name,
        expected_recording_id=proxy.recording_id,
        expected_manifest_sha256=proxy.acquisition_projection_record[
            "source_manifest_sha256"
        ],
        use_consolidated=True,
    )
    require_proxy_native_binding(proxy, native)

    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    timing = load_provider_recording_timing_authority(
        archive,
        required=True,
        use_consolidated=True,
        expected_sha256=expected_timing_authority_sha256,
    )
    assert timing is not None
    if timing.recording_id != proxy.recording_id:
        _fail("Recording timing authority belongs to another recording.")
    if timing.frame_count != native.dimensions.total_frames:
        _fail("Timing and native provider frame domains differ.")

    stimulus_run_name = _exact_run_name(
        native.source_stimulus_run_path,
        parent="analysis/stimulus_runs",
        field="native stimulus run path",
    )
    physical = load_stimulus_physical_coordinate_authority(
        root,
        stimulus_run=stimulus_run_name,
    )
    physical = require_bound_stimulus_physical_coordinate_authority(physical)
    coordinate = load_stimulus_response_coordinate_authority(
        root,
        stimulus_run=stimulus_run_name,
        track_physical_authority=physical,
    )
    if coordinate.physical.camera_id != timing.camera_id:
        _fail("Coordinate and recording timing camera identities differ.")
    source_camera_pointer = _pointer(
        native.source_authority["stimulus"]["source_camera_frame"],
        field="native stimulus source_camera_frame",
    )
    position_camera_pointer = _pointer(
        native.source_authority["position"]["source_camera_frame"],
        field="native position source_camera_frame",
    )
    coordinate_camera_pointer = _pointer(
        coordinate.record["source_camera_frame"],
        field="coordinate source_camera_frame",
    )
    if not (
        source_camera_pointer
        == position_camera_pointer
        == coordinate_camera_pointer
    ):
        _fail(
            "Native position, stimulus, and rebound transform use different "
            "camera frames."
        )
    acquisition_pointer = _pointer(
        native.source_authority["acquisition_frame_authority"],
        field="native acquisition_frame_authority",
    )
    if acquisition_pointer != {
        "record_ref": timing.clock_run_path,
        "record_sha256": timing.clock_record_sha256,
    }:
        _fail("Native provider and recording timing use different frame domains.")

    frame_rows = _frame_source_rows(proxy, native)
    selected = np.asarray(proxy.selected, dtype=bool)
    if not np.array_equal(
        np.asarray(proxy.array("selected_chaser_index")),
        np.broadcast_to(native.chaser_index, proxy.array("selected_chaser_index").shape),
    ):
        _fail("Proxy selected chaser identity axis differs from the native source.")
    if np.any(selected):
        selected_rows = np.asarray(proxy.selected_native_sample_row_index)[selected]
        if not np.array_equal(
            proxy.selected_chaser_valid[selected], native.chaser_valid[selected_rows]
        ) or not np.array_equal(
            proxy.selected_chaser_position_xy[selected],
            native.chaser_position_arena_xy[selected_rows],
            equal_nan=True,
        ):
            _fail("Proxy selected chaser values differ from the exact native source.")

    chaser_camera_xy = np.full(proxy.selected_chaser_position_xy.shape, np.nan, dtype=np.float64)
    chaser_valid = np.asarray(proxy.selected_chaser_valid, dtype=bool).copy()
    if np.any(chaser_valid):
        converted = coordinate.arena_to_source_camera_px(
            proxy.selected_chaser_position_xy[chaser_valid]
        )
        if converted.shape != (int(np.count_nonzero(chaser_valid)), 2) or not np.isfinite(converted).all():
            _fail("Typed arena-to-camera transform returned invalid chaser coordinates.")
        chaser_camera_xy[chaser_valid] = converted

    fish_xy = np.asarray(native.fish_position_source_camera_xy[frame_rows], dtype=np.float64)
    fish_valid = np.asarray(native.fish_valid[frame_rows], dtype=bool)
    fish_rows = np.asarray(native.fish_source_position_run_row_index[frame_rows], dtype=np.int64)
    fish_xy = fish_xy.copy()
    fish_xy[~fish_valid] = np.nan
    if np.any(fish_valid & (fish_rows < 0)):
        _fail("A valid fish position lacks an exact source row.")

    identities, role_axis, occurrence_record = _configured_chasers(
        root=root,
        stimulus_run_name=stimulus_run_name,
        native=native,
    )
    n_frames = proxy.dimensions.n_frames
    n_chasers = proxy.dimensions.n_chasers
    roles = np.broadcast_to(role_axis, (n_frames, n_chasers)).copy()
    occurrence = np.ones((n_frames, n_chasers), dtype=bool)
    selection_membership = np.ones(n_frames, dtype=bool)

    frames = np.asarray(proxy.acquisition_frame_index, dtype=np.int64)
    row_axis_record = _record(
        "palette.chaser_relative_frame.row_axis_binding",
        proxy.recording_id,
        policy_id=ROW_AXIS_POLICY_ID,
        acquisition_frame_array_sha256=array_values_sha256(frames),
        proxy_run_path=proxy.run_path,
        proxy_manifest_sha256=proxy.manifest_sha256,
        represented_frame_count=n_frames,
    )
    row_axis_digest = canonical_json_sha256(row_axis_record)
    row_axis_id = f"{proxy.run_path}/arrays/acquisition_frame_index"

    arena_record, transform_record, coordinate_id, scale_id, scale_digest = (
        _coordinate_records(coordinate, recording_id=proxy.recording_id)
    )
    coordinate_policy = CoordinatePolicy(
        coordinate_authority_id=coordinate_id,
        coordinate_frame="source_camera_continuous_pixel_xy",
        policy_id=COORDINATE_POLICY_ID,
    )
    scale_policy = ScalePolicy(
        scale_authority_id=scale_id,
        scale_digest=scale_digest,
        pixels_per_unit=1.0 / coordinate.mm_per_pixel,
        unit="mm",
    )
    timing_id = timing.clock_run_path
    timing_policy = TimingPolicy(
        timing_authority_id=timing_id,
        timing_digest=timing.sha256,
        recording_id=proxy.recording_id,
        timestamp_field=None,
        policy_id=TIMING_POLICY_ID,
    )
    frame_keys = AcquisitionFrameKeys(
        recording_id=proxy.recording_id,
        acquisition_frame_id=frames,
        track_sample_id=frames.copy(),
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
        timestamp_ns=None,
    )
    position_authority = native.source_authority["position"]
    fish_authority = ProviderSourceAuthority(
        recording_id=proxy.recording_id,
        source_authority_id=position_authority["run_path"],
        source_digest=position_authority["manifest_sha256"],
        provider_id=position_authority["estimator_id"],
        provider_digest=position_authority["estimator_sha256"],
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
    )
    publication_binding = proxy.publication_binding_record
    chaser_authority = ProviderSourceAuthority(
        recording_id=proxy.recording_id,
        source_authority_id=proxy.run_path,
        source_digest=proxy.manifest_sha256,
        provider_id=PROXY_POLICY_ID,
        provider_digest=proxy.acquisition_projection_record_sha256,
        coordinate_authority_id=coordinate_id,
        scale_authority_id=scale_id,
        timing_authority_id=timing_id,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_digest=row_axis_digest,
    )
    inputs = ChaserRelativeFrameInput(
        frame_keys=frame_keys,
        fish_xy=fish_xy,
        fish_valid=fish_valid,
        fish_source_row_index=fish_rows,
        fish_authority=fish_authority,
        chasers=ChaserObservations(
            identities=identities,
            behavior_roles=roles,
            xy=chaser_camera_xy,
            valid=chaser_valid,
            source_row_index=np.asarray(
                proxy.selected_source_stimulus_run_row_index,
                dtype=np.int64,
            ),
            authority=chaser_authority,
        ),
        selection_membership=selection_membership,
        occurrence_membership=occurrence,
        coordinate_policy=coordinate_policy,
        scale_policy=scale_policy,
        timing_policy=timing_policy,
        body_frame=None,
    )
    result = compute_chaser_relative_frame(inputs)

    subject = resolve_subject_metadata(root, allow_legacy=False)
    if len(subject.subject_ids) != 1:
        _fail("Proxy relative-frame publication requires exactly one canonical subject ID.")
    if (
        expected_subject_metadata_sha256 is not None
        and subject.record_sha256 != expected_subject_metadata_sha256
    ):
        _fail("Canonical subject metadata differs from the expected digest.")
    subject_id = subject.subject_ids[0]
    subject_record = _record(
        "palette.chaser_relative_frame.subject_identity_binding",
        proxy.recording_id,
        subject_id=subject_id,
        subject_identity_kind=subject.subject_identity_kind,
        source_subject_metadata_run_path=subject.group_path,
        source_subject_metadata_sha256=subject.record_sha256,
    )
    temporal_record = _record(
        "palette.chaser_relative_frame.temporal_selection",
        proxy.recording_id,
        selection_id=TEMPORAL_SELECTION_ID,
        row_axis_policy_id=ROW_AXIS_POLICY_ID,
        row_axis_authority_id=row_axis_id,
        row_axis_authority_sha256=row_axis_digest,
        selected_frame_count=n_frames,
        semantics=(
            "all input-acquisition frames represented by native controller-state "
            "samples; no physical-presentation timing claim"
        ),
    )
    profile: ChaserAnalysisProfile = load_chaser_analysis_profile(
        analysis_profile_path
    )
    profile_payload = profile.to_dict()
    profile_record = _record(
        "palette.chaser_relative_frame.analysis_profile_binding",
        proxy.recording_id,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_sha256=profile.sha256,
        profile=profile_payload,
        temporal_alignment_policy=profile.policies.get("temporal_alignment"),
        candidate_use_class="exploratory_proxy",
    )
    context = ChaserRelativeFramePublicationContext(
        fish_identity=subject_id,
        subject_identity_record=subject_record,
        temporal_selection_record=temporal_record,
        chaser_occurrence_record=occurrence_record,
        acquisition_projection_record=_plain(proxy.acquisition_projection_record),
        acquisition_projection_publication_record=publication_binding,
        analysis_profile_record=profile_record,
        arena_geometry_record=arena_record,
        arena_to_source_camera_transform_record=transform_record,
    )
    prepared = prepare_chaser_relative_frame(result, context=context)
    coordinate_sha256 = canonical_json_sha256(_plain(coordinate.record))
    return PreparedProxyRelativeFrame(
        prepared=prepared,
        proxy_run_path=proxy.run_path,
        proxy_manifest_sha256=proxy.manifest_sha256,
        native_run_path=native.run_path,
        native_manifest_sha256=native.manifest_sha256,
        coordinate_lineage_sha256=coordinate_sha256,
        timing_authority_sha256=timing.sha256,
        subject_metadata_sha256=subject.record_sha256,
    )


__all__ = [
    "ADAPTER_SCHEMA_ID",
    "ADAPTER_SCHEMA_VERSION",
    "CHASER_IDENTITY_POLICY_ID",
    "CHASER_OCCURRENCE_POLICY_ID",
    "COORDINATE_POLICY_ID",
    "ROW_AXIS_POLICY_ID",
    "TEMPORAL_SELECTION_ID",
    "TIMING_POLICY_ID",
    "ChaserProxyRelativeFrameAdapterError",
    "PreparedProxyRelativeFrame",
    "prepare_proxy_relative_frame",
    "require_proxy_native_binding",
]
