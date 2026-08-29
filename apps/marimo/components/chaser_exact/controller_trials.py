"""Exact logged controller-trial membership and distance views."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .distance_traces import _trace_display_projection
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import TRACE_MAX_POINTS, freeze, plain

CONTROLLER_TRIAL_DISPLAY_RECIPE = "exact_logged_trial_membership_distance_view_v1"
CONTROLLER_TRIAL_MAX_PANELS = 32
CONTROLLER_TRIAL_MAX_GAP_MARKERS_PER_PANEL = 2_000


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Controller-trial successor lacks {label}.")
    return value


def _array(
    handle: Any,
    name: str,
    *,
    dtype: Any,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        values = np.asarray(handle.array(name))
    except KeyError as exc:
        raise ValueError(
            f"Controller-trial successor lacks required array {name!r}."
        ) from exc
    if values.dtype != np.dtype(dtype) or values.shape != shape:
        raise ValueError(
            f"Controller-trial array {name!r} has incompatible dtype or shape."
        )
    return values


def _controller_trial_values(
    projection: ExactChaserSuccessorProjection,
) -> Mapping[str, Any]:
    """Validate sealed trial membership, gaps, timing, and row-axis alignment."""

    handle = projection.controller_trials
    if handle is None or projection.relatives is None:
        raise ValueError(
            "Controller-trial display requires one exact trial successor and "
            "paired relative-frame sources."
        )
    if handle.successor_kind != "controller_chase_trials":
        raise ValueError("Controller-trial projection names another successor kind.")
    if handle.deep_audited is not True:
        raise ValueError("Controller-trial display requires a deep content audit.")
    scientific = _mapping(handle.scientific_manifest, label="scientific manifest")
    dimensions = _mapping(scientific.get("dimensions"), label="dimensions")
    n_frames = int(dimensions.get("n_frames", 0))
    n_chasers = int(dimensions.get("n_chasers", 0))
    n_rows = int(dimensions.get("n_source_rows", 0))
    n_trials = int(dimensions.get("n_trials", 0))
    relative = projection.relatives[0]
    if (
        n_frames != relative.n_frames
        or n_chasers != relative.n_chasers
        or n_rows != n_frames * n_chasers
        or n_trials <= 0
        or n_trials > CONTROLLER_TRIAL_MAX_PANELS
    ):
        raise ValueError(
            "Controller-trial dimensions exceed the exact display contract."
        )

    table_specs = {
        "trial_row_id": np.int64,
        "chaser_identity_code": np.uint16,
        "logged_trial_id": np.int64,
        "trial_ordinal": np.int32,
        "start_source_frame_row": np.int64,
        "end_source_frame_row_exclusive": np.int64,
        "start_acquisition_frame_id": np.int64,
        "end_acquisition_frame_id_inclusive": np.int64,
        "trigger_acquisition_frame_id": np.int64,
        "trigger_timestamp_ns": np.int64,
        "trigger_timestamp_valid": bool,
        "active_member_count": np.int64,
        "envelope_frame_count": np.int64,
        "gap_frame_count": np.int64,
        "gap_fraction": np.float64,
        "trigger_source_code": np.uint8,
        "fallback_used": bool,
    }
    dense_specs = {
        "source_relative_row_id": np.int64,
        "trial_row_id_by_source_row": np.int64,
        "trial_envelope_row_id_by_source_row": np.int64,
        "logged_active_trial_member": bool,
        "trial_envelope_member": bool,
        "trial_gap_member": bool,
        "trial_gap_reason_code_by_source_row": np.uint8,
        "logged_active_trial_id_unavailable": bool,
    }
    arrays = {
        name: _array(handle, name, dtype=dtype, shape=(n_trials,))
        for name, dtype in table_specs.items()
    }
    arrays.update(
        {
            name: _array(handle, name, dtype=dtype, shape=(n_rows,))
            for name, dtype in dense_specs.items()
        }
    )
    trial_row = arrays["trial_row_id_by_source_row"]
    envelope_row = arrays["trial_envelope_row_id_by_source_row"]
    member = arrays["logged_active_trial_member"]
    envelope = arrays["trial_envelope_member"]
    gap = arrays["trial_gap_member"]
    gap_reason = arrays["trial_gap_reason_code_by_source_row"]
    if (
        not np.array_equal(arrays["trial_row_id"], np.arange(n_trials))
        or not np.array_equal(arrays["source_relative_row_id"], np.arange(n_rows))
        or np.any(arrays["fallback_used"])
        or np.any(arrays["logged_active_trial_id_unavailable"])
        or not np.array_equal(member, trial_row >= 0)
        or not np.array_equal(envelope, envelope_row >= 0)
        or np.any(member & (~envelope | (trial_row != envelope_row)))
        or not np.array_equal(gap, envelope & ~member)
        or not np.array_equal(gap, gap_reason != 0)
        or np.any((trial_row < -1) | (trial_row >= n_trials))
        or np.any((envelope_row < -1) | (envelope_row >= n_trials))
        or np.any(arrays["logged_trial_id"] <= 0)
        or np.any(arrays["trial_ordinal"] <= 0)
        or np.any(arrays["trigger_source_code"] != 1)
        or not np.all(arrays["trigger_timestamp_valid"])
    ):
        raise ValueError(
            "Controller-trial membership or fail-closed evidence is inconsistent."
        )
    gap_registry = identity_registry(scientific, "trial_gap_reason")
    trigger_registry = identity_registry(scientific, "trigger_source")
    if gap_registry != {
        "0": "not_a_trial_gap",
        "1": "semantic_selection_nonmember",
        "2": "chaser_occurrence_unavailable",
        "3": "controller_active_state_unavailable",
        "4": "explicit_controller_inactive",
        "5": "logged_trial_id_unavailable",
        "6": "logged_trial_id_mismatch",
    } or np.any(gap_reason > 6):
        raise ValueError("Controller-trial gap-reason registry is unsupported.")
    if trigger_registry != {"1": "first_logged_active_member"}:
        raise ValueError("Controller-trial trigger-source registry is unsupported.")

    relative_chaser = relative.arrays["chaser_identity_code"]
    acquisition = relative.arrays["acquisition_frame_id"]
    timestamp = relative.arrays["timestamp_ns"]
    timestamp_valid = relative.arrays["timestamp_valid"]
    for trial_index in range(n_trials):
        member_rows = np.flatnonzero(trial_row == trial_index)
        envelope_rows = np.flatnonzero(envelope_row == trial_index)
        gap_rows = np.flatnonzero(gap & (envelope_row == trial_index))
        start = int(arrays["start_source_frame_row"][trial_index])
        end = int(arrays["end_source_frame_row_exclusive"][trial_index])
        code = int(arrays["chaser_identity_code"][trial_index])
        if (
            not member_rows.size
            or start < 0
            or end <= start
            or not np.array_equal(envelope_rows // n_chasers, np.arange(start, end))
            or np.any(relative_chaser[envelope_rows] != code)
            or int(arrays["active_member_count"][trial_index]) != member_rows.size
            or int(arrays["envelope_frame_count"][trial_index]) != envelope_rows.size
            or int(arrays["gap_frame_count"][trial_index]) != gap_rows.size
            or not np.isclose(
                float(arrays["gap_fraction"][trial_index]),
                float(gap_rows.size) / float(envelope_rows.size),
                rtol=1e-12,
                atol=1e-12,
            )
            or int(arrays["start_acquisition_frame_id"][trial_index])
            != int(acquisition[member_rows[0]])
            or int(arrays["end_acquisition_frame_id_inclusive"][trial_index])
            != int(acquisition[member_rows[-1]])
            or int(arrays["trigger_acquisition_frame_id"][trial_index])
            != int(acquisition[member_rows[0]])
            or not bool(timestamp_valid[member_rows[0]])
            or int(arrays["trigger_timestamp_ns"][trial_index])
            != int(timestamp[member_rows[0]])
        ):
            raise ValueError(
                f"Controller trial row {trial_index} has inconsistent sealed evidence."
            )
    return freeze(
        {
            **arrays,
            "n_frames": n_frames,
            "n_chasers": n_chasers,
            "n_rows": n_rows,
            "n_trials": n_trials,
            "gap_registry": gap_registry,
        }
    )


def _bounded_gap_indices(indices: np.ndarray) -> np.ndarray:
    if indices.size <= CONTROLLER_TRIAL_MAX_GAP_MARKERS_PER_PANEL:
        return indices
    positions = np.linspace(
        0,
        indices.size - 1,
        CONTROLLER_TRIAL_MAX_GAP_MARKERS_PER_PANEL,
        dtype=np.int64,
    )
    return indices[np.unique(positions)]


def _timed_gap_indices(
    gap_local: np.ndarray,
    *,
    timestamp_valid: np.ndarray,
    frame_indices: np.ndarray,
    chaser_column: int,
) -> np.ndarray:
    """Keep only gap rows whose frame-level timestamp is valid."""

    return gap_local[timestamp_valid[frame_indices[gap_local], chaser_column]]


def build_exact_controller_trials_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render full-session and trigger-aligned exact trial distance evidence."""

    from plotly.subplots import make_subplots

    values = _controller_trial_values(projection)
    assert projection.relatives is not None
    relative = projection.relatives[0]
    n_trials = int(values["n_trials"])
    n_chasers = int(values["n_chasers"])
    timestamp = relative.frame_chaser("timestamp_ns").astype(np.int64)
    timestamp_valid = relative.frame_chaser("timestamp_valid").astype(bool)
    frame_timestamp = relative.collapsed_frame("timestamp_ns").astype(np.int64)
    frame_timestamp_valid = relative.collapsed_frame("timestamp_valid").astype(bool)
    valid_frames = np.flatnonzero(frame_timestamp_valid)
    if not valid_frames.size:
        raise ValueError("Controller-trial source has no valid session timestamps.")
    session_time_s = (
        frame_timestamp.astype(np.float64) - float(frame_timestamp[valid_frames[0]])
    ) / 1e9
    member = np.asarray(values["logged_active_trial_member"], dtype=bool).reshape(
        int(values["n_frames"]), n_chasers
    )
    chaser_codes = relative.frame_chaser("chaser_identity_code")
    if not np.all(chaser_codes == chaser_codes[:1]):
        raise ValueError("Controller-trial chaser identity changes by source frame.")
    code_to_column = {
        int(chaser_codes[0, column]): column for column in range(n_chasers)
    }
    if len(code_to_column) != n_chasers:
        raise ValueError("Controller-trial chaser identities are not unique.")

    colors = ("#1f77b4", "#d95f02")
    full_titles = [
        f"full session · chaser {int(chaser_codes[0, column])}"
        for column in range(n_chasers)
    ]
    full = make_subplots(rows=1, cols=n_chasers, subplot_titles=full_titles)
    for column in range(n_chasers):
        for provider_index, (provider_id, provider_relative) in enumerate(
            zip(projection.provider_ids, projection.relatives, strict=True)
        ):
            distance = provider_relative.frame_chaser("relative_distance_physical")[
                :, column
            ]
            valid = (
                provider_relative.frame_chaser("relative_physical_valid")[:, column]
                & provider_relative.frame_chaser("chaser_occurrence_member")[:, column]
                & frame_timestamp_valid
            )
            x, y = _trace_display_projection(session_time_s, distance, valid)
            full.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines",
                    name=provider_id,
                    legendgroup=provider_id,
                    showlegend=column == 0,
                    connectgaps=False,
                    line={"color": colors[provider_index], "width": 1},
                    opacity=0.45,
                ),
                row=1,
                col=column + 1,
            )
            trial_x, trial_y = _trace_display_projection(
                session_time_s,
                distance,
                valid & member[:, column],
            )
            full.add_trace(
                go.Scattergl(
                    x=trial_x,
                    y=trial_y,
                    mode="lines",
                    name=f"{provider_id} · exact trial members",
                    legendgroup=f"{provider_id}-members",
                    showlegend=column == 0,
                    connectgaps=False,
                    line={"color": colors[provider_index], "width": 2.5},
                ),
                row=1,
                col=column + 1,
            )
    full.update_xaxes(title_text="session time from first valid timestamp (s)")
    full.update_yaxes(title_text="distance (mm)")
    full.update_layout(
        title=f"Full-session distance with exact logged trial membership · {projection.recording_id}",
        height=360,
    )

    trial_titles = [
        (
            f"trial {int(values['trial_ordinal'][index])} · logged ID "
            f"{int(values['logged_trial_id'][index])} · chaser "
            f"{int(values['chaser_identity_code'][index])}"
        )
        for index in range(n_trials)
    ]
    trials = make_subplots(rows=n_trials, cols=1, subplot_titles=trial_titles)
    gap_registry = values["gap_registry"]
    table_records: list[dict[str, Any]] = []
    for trial_index in range(n_trials):
        code = int(values["chaser_identity_code"][trial_index])
        column = code_to_column.get(code)
        if column is None:
            raise ValueError("Controller trial names an unknown chaser identity.")
        start = int(values["start_source_frame_row"][trial_index])
        end = int(values["end_source_frame_row_exclusive"][trial_index])
        frame_indices = np.arange(start, end, dtype=np.int64)
        source_rows = frame_indices * n_chasers + column
        trigger_ns = int(values["trigger_timestamp_ns"][trial_index])
        relative_time_s = (
            timestamp[frame_indices, column].astype(np.float64) - float(trigger_ns)
        ) / 1e9
        exact_member = np.asarray(values["trial_row_id_by_source_row"])[source_rows]
        exact_member = exact_member == trial_index
        finite_distances: list[np.ndarray] = []
        for provider_index, (provider_id, provider_relative) in enumerate(
            zip(projection.provider_ids, projection.relatives, strict=True)
        ):
            distance = provider_relative.frame_chaser("relative_distance_physical")[
                frame_indices, column
            ]
            valid = (
                provider_relative.frame_chaser("relative_physical_valid")[
                    frame_indices, column
                ]
                & provider_relative.frame_chaser("chaser_occurrence_member")[
                    frame_indices, column
                ]
                & timestamp_valid[frame_indices, column]
                & exact_member
            )
            x, y = _trace_display_projection(relative_time_s, distance, valid)
            trials.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines",
                    name=provider_id,
                    legendgroup=provider_id,
                    showlegend=trial_index == 0,
                    connectgaps=False,
                    line={"color": colors[provider_index], "width": 1.5},
                ),
                row=trial_index + 1,
                col=1,
            )
            finite_distances.append(distance[np.isfinite(distance) & valid])
        gap_local = np.flatnonzero(np.asarray(values["trial_gap_member"])[source_rows])
        timed_gap_local = _timed_gap_indices(
            gap_local,
            timestamp_valid=timestamp_valid,
            frame_indices=frame_indices,
            chaser_column=column,
        )
        displayed_gap_local = _bounded_gap_indices(timed_gap_local)
        finite_parts = [item for item in finite_distances if item.size]
        finite = (
            np.concatenate(finite_parts)
            if finite_parts
            else np.asarray([], dtype=np.float64)
        )
        gap_y = float(np.max(finite)) if finite.size else 0.0
        if displayed_gap_local.size:
            reasons = np.asarray(values["trial_gap_reason_code_by_source_row"])[
                source_rows[displayed_gap_local]
            ]
            trials.add_trace(
                go.Scattergl(
                    x=relative_time_s[displayed_gap_local],
                    y=np.full(displayed_gap_local.size, gap_y),
                    mode="markers",
                    name="retained nonmember gap",
                    legendgroup="trial-gaps",
                    showlegend=trial_index == 0,
                    marker={"color": "#666666", "symbol": "x", "size": 6},
                    customdata=np.asarray(
                        [gap_registry[str(int(reason))] for reason in reasons],
                        dtype=object,
                    ),
                    hovertemplate=(
                        "relative time=%{x:.4f} s<br>retained evidence: %{customdata}"
                        "<br>marker height is display-only<extra></extra>"
                    ),
                ),
                row=trial_index + 1,
                col=1,
            )
        table_records.append(
            {
                "trial_row_id": trial_index,
                "chaser_identity_code": code,
                "logged_trial_id": int(values["logged_trial_id"][trial_index]),
                "trial_ordinal": int(values["trial_ordinal"][trial_index]),
                "active_member_count": int(values["active_member_count"][trial_index]),
                "envelope_frame_count": int(
                    values["envelope_frame_count"][trial_index]
                ),
                "gap_frame_count": int(values["gap_frame_count"][trial_index]),
                "gap_fraction": float(values["gap_fraction"][trial_index]),
                "timed_gap_marker_candidate_count": int(timed_gap_local.size),
                "untimed_gap_count": int(gap_local.size - timed_gap_local.size),
                "trigger_timestamp_ns": trigger_ns,
                "fallback_used": False,
            }
        )
    trials.update_xaxes(title_text="time from first logged active member (s)")
    trials.update_yaxes(title_text="distance (mm)")
    display_meta = {
        "recipe_id": CONTROLLER_TRIAL_DISPLAY_RECIPE,
        "membership_array": "logged_active_trial_member",
        "envelope_array": "trial_envelope_member",
        "gap_array": "trial_gap_member",
        "gap_reason_array": "trial_gap_reason_code_by_source_row",
        "legacy_trial_reconstruction": "prohibited",
        "interpolation": "prohibited",
        "trace_projection_algorithm": (
            "source_order_bucket_first_last_min_max_missing_break_v1"
        ),
        "max_points_per_trace": TRACE_MAX_POINTS,
        "max_trial_panels": CONTROLLER_TRIAL_MAX_PANELS,
        "max_gap_markers_per_panel": CONTROLLER_TRIAL_MAX_GAP_MARKERS_PER_PANEL,
        "gap_marker_height_role": "display_only_panel_max_not_scientific_value",
        "gap_marker_timestamp_policy": (
            "valid_source_timestamps_only_all_gaps_retained_in_table"
        ),
    }
    figure_meta = {
        **plain(projection.provenance),
        "controller_trial_display": display_meta,
    }
    full.update_layout(meta=figure_meta)
    trials.update_layout(
        title="Trigger-aligned exact active trial members and retained gap evidence",
        height=max(360, 280 * n_trials),
        meta=figure_meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "Only producer-logged active rows are trial members. Gray × markers "
                "are retained envelope gaps with valid source timestamps and explicit "
                "exclusion reasons; the table retains all gaps, including untimed "
                "ones. Gaps are never converted into trial members.",
                kind="warn",
            ),
            full,
            trials,
            mo.md("### Exact controller-trial table"),
            mo.ui.table(table_records, selection=None, page_size=10),
        ]
    )


__all__ = [
    "CONTROLLER_TRIAL_DISPLAY_RECIPE",
    "CONTROLLER_TRIAL_MAX_GAP_MARKERS_PER_PANEL",
    "CONTROLLER_TRIAL_MAX_PANELS",
    "_controller_trial_values",
    "_timed_gap_indices",
    "build_exact_controller_trials_output",
]
