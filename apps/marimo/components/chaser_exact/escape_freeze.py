"""Persisted exact-trial escape/freeze outcomes and event evidence."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..chaser_exact_escape_freeze_contract import EXPECTED_REGISTRIES
from .array_requirements import ESCAPE_FREEZE_ARRAYS
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import freeze, plain

ESCAPE_FREEZE_DISPLAY_RECIPE = "persisted_exact_trial_escape_freeze_outcome_view_v1"
ESCAPE_FREEZE_DISPLAY_ALGORITHM = "source_order_uniform_endpoint_preserving_v1"
ESCAPE_FREEZE_MAX_EVENT_POINTS = 6_000
ESCAPE_FREEZE_MAX_EVENT_TABLE_ROWS = 1_000

_CLASS_COLORS = {
    0: "#9d9da1",
    1: "#e45756",
    2: "#4c78a8",
    3: "#54a24b",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Escape/freeze successor lacks {label}.")
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
            f"Escape/freeze successor lacks required array {name!r}."
        ) from exc
    if values.dtype != np.dtype(dtype) or values.shape != shape:
        raise ValueError(
            f"Escape/freeze array {name!r} has incompatible dtype or shape."
        )
    return values


def _finite_fraction(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(np.all((finite >= 0) & (finite <= 1)))


def _escape_freeze_values(
    projection: ExactChaserSuccessorProjection,
) -> Mapping[str, Any]:
    """Validate persisted identities, classifications, rates, and reason codes."""

    handle = projection.escape_freeze
    if handle is None:
        raise ValueError(
            "Escape/freeze display requires one exact persisted successor."
        )
    if handle.successor_kind != "chaser_escape_freeze":
        raise ValueError("Escape/freeze projection names another successor kind.")
    handle.require_verified_arrays(ESCAPE_FREEZE_ARRAYS)
    scientific = _mapping(handle.scientific_manifest, label="scientific manifest")
    dimensions = _mapping(scientific.get("dimensions"), label="dimensions")
    parameters = _mapping(scientific.get("parameters"), label="parameters")
    n_trials = int(dimensions.get("n_trials", -1))
    n_events = int(dimensions.get("n_events", -1))
    n_sweep_rows = int(dimensions.get("n_sweep_rows", -1))
    if not 0 < n_trials <= 32 or n_events < 0 or n_sweep_rows < 0:
        raise ValueError("Escape/freeze dimensions are incompatible.")

    event_specs = {
        "event_row_id": np.int64,
        "event_source_bout_chaser_row_id": np.int64,
        "event_bout_row_id": np.int64,
        "event_bout_id": np.int64,
        "event_controller_trial_row_id": np.int64,
        "event_chaser_identity_code": np.uint16,
        "event_onset_acquisition_frame_id": np.int64,
        "event_peak_speed_mm_s": np.float32,
        "event_distance_at_onset_mm": np.float32,
        "event_separation_gain_mm": np.float32,
        "event_directed_valid": bool,
        "event_turn_deg": np.float32,
        "event_high_turn": bool,
        "event_latency_from_trigger_s": np.float64,
        "event_trigger_distance_mm": np.float32,
        "event_recaptured": bool,
        "event_recapture_latency_s": np.float64,
        "event_trace_valid": bool,
        "event_trace_exclusion_reason_code": np.uint8,
    }
    trial_specs = {
        "trial_row_id": np.int64,
        "trial_chaser_identity_code": np.uint16,
        "trial_logged_id": np.int64,
        "trial_ordinal": np.int32,
        "trial_envelope_frame_count": np.int64,
        "trial_gap_frame_count": np.int64,
        "trial_gap_fraction": np.float64,
        "trial_logged_active_id_unavailable_count": np.int64,
        "trial_trigger_acquisition_frame_id": np.int64,
        "trial_trigger_distance_mm": np.float32,
        "trial_valid_time_s": np.float64,
        "trial_bout_count": np.int64,
        "trial_escape_event_count": np.int64,
        "trial_high_turn_escape_count": np.int64,
        "trial_escape_event_rate_per_min": np.float64,
        "trial_first_escape_latency_s": np.float64,
        "trial_mean_separation_gain_mm": np.float64,
        "trial_recapture_fraction": np.float64,
        "trial_freeze_valid_fraction": np.float64,
        "trial_freeze_low_speed_fraction": np.float64,
        "trial_escape_speed_class": bool,
        "trial_freeze_candidate": bool,
        "trial_response_class_code": np.uint8,
    }
    sweep_specs = {
        "sweep_row_id": np.int64,
        "sweep_trial_row_id": np.int64,
        "sweep_speed_threshold_mm_s": np.float32,
        "sweep_escape_event_count": np.int64,
        "sweep_escape_event_rate_per_min": np.float64,
    }
    recording_specs = {
        "recording_trial_count": np.int64,
        "recording_escape_trial_count": np.int64,
        "recording_freeze_trial_count": np.int64,
        "recording_escape_event_count": np.int64,
        "recording_high_turn_escape_event_count": np.int64,
        "recording_trace_usable_event_count": np.int64,
    }
    arrays = {
        name: _array(handle, name, dtype=dtype, shape=(n_events,))
        for name, dtype in event_specs.items()
    }
    arrays.update(
        {
            name: _array(handle, name, dtype=dtype, shape=(n_trials,))
            for name, dtype in trial_specs.items()
        }
    )
    arrays.update(
        {
            name: _array(handle, name, dtype=dtype, shape=(n_sweep_rows,))
            for name, dtype in sweep_specs.items()
        }
    )
    arrays.update(
        {
            name: _array(handle, name, dtype=dtype, shape=(1,))
            for name, dtype in recording_specs.items()
        }
    )

    if not np.array_equal(arrays["event_row_id"], np.arange(n_events)):
        raise ValueError("Escape/freeze event-row identity is not canonical.")
    if not np.array_equal(arrays["trial_row_id"], np.arange(n_trials)):
        raise ValueError("Escape/freeze trial-row identity is not canonical.")
    if np.unique(arrays["trial_row_id"]).size != n_trials:
        raise ValueError("Escape/freeze trial-row identities are duplicated.")
    event_trial = arrays["event_controller_trial_row_id"]
    if np.any((event_trial < 0) | (event_trial >= n_trials)):
        raise ValueError("Escape/freeze event attaches outside the exact trial table.")
    if n_events and np.any(
        arrays["event_chaser_identity_code"]
        != arrays["trial_chaser_identity_code"][event_trial]
    ):
        raise ValueError("Escape/freeze event attaches across chaser identities.")
    if np.unique(arrays["event_source_bout_chaser_row_id"]).size != n_events:
        raise ValueError("Escape/freeze source bout/chaser identities are duplicated.")

    escape_threshold = float(parameters.get("escape_speed_threshold_mm_s", np.nan))
    high_turn_threshold = float(parameters.get("high_turn_threshold_deg", np.nan))
    freeze_fraction_threshold = float(
        parameters.get("freeze_fraction_threshold", np.nan)
    )
    minimum_valid_fraction = float(
        parameters.get("minimum_freeze_valid_fraction", np.nan)
    )
    freeze_window_s = float(parameters.get("freeze_window_s", np.nan))
    freeze_speed_threshold = float(
        parameters.get("freeze_speed_threshold_mm_s", np.nan)
    )
    if (
        not np.isfinite(escape_threshold)
        or escape_threshold <= 0
        or not np.isfinite(high_turn_threshold)
        or high_turn_threshold <= 0
        or not np.isfinite(freeze_window_s)
        or freeze_window_s <= 0
        or not np.isfinite(freeze_speed_threshold)
        or freeze_speed_threshold < 0
        or not np.isfinite(freeze_fraction_threshold)
        or not 0 <= freeze_fraction_threshold <= 1
        or not np.isfinite(minimum_valid_fraction)
        or not 0 <= minimum_valid_fraction <= 1
    ):
        raise ValueError("Escape/freeze classifier parameters are invalid.")
    peak = arrays["event_peak_speed_mm_s"]
    directed = arrays["event_directed_valid"]
    turn = arrays["event_turn_deg"]
    expected_high_turn = (
        directed & np.isfinite(turn) & (np.abs(turn) >= high_turn_threshold)
    )
    trace_valid = arrays["event_trace_valid"]
    trace_reason = arrays["event_trace_exclusion_reason_code"]
    if (
        np.any(~np.isfinite(peak))
        or np.any(peak < escape_threshold)
        or np.any(~np.isfinite(arrays["event_distance_at_onset_mm"]))
        or np.any(~np.isfinite(arrays["event_separation_gain_mm"]))
        or not np.array_equal(arrays["event_high_turn"], expected_high_turn)
        or np.any(trace_reason > 2)
        or not np.array_equal(trace_valid, trace_reason == 0)
        or np.any(arrays["event_recaptured"] & ~trace_valid)
    ):
        raise ValueError("Escape/freeze event evidence is inconsistent.")

    nonnegative_trial_names = (
        "trial_envelope_frame_count",
        "trial_gap_frame_count",
        "trial_logged_active_id_unavailable_count",
        "trial_valid_time_s",
        "trial_bout_count",
        "trial_escape_event_count",
        "trial_high_turn_escape_count",
    )
    if any(np.any(arrays[name] < 0) for name in nonnegative_trial_names):
        raise ValueError("Escape/freeze trial evidence contains negative values.")
    envelope = arrays["trial_envelope_frame_count"]
    gap_count = arrays["trial_gap_frame_count"]
    expected_gap_fraction = np.divide(
        gap_count,
        envelope,
        out=np.zeros(n_trials, dtype=np.float64),
        where=envelope > 0,
    )
    observed_event_count = np.bincount(event_trial, minlength=n_trials)
    observed_high_turn_count = np.bincount(
        event_trial,
        weights=arrays["event_high_turn"].astype(np.int64),
        minlength=n_trials,
    ).astype(np.int64)
    valid_time = arrays["trial_valid_time_s"]
    trial_count = arrays["trial_escape_event_count"]
    expected_rate = np.divide(
        trial_count.astype(np.float64),
        valid_time / 60.0,
        out=np.full(n_trials, np.nan, dtype=np.float64),
        where=valid_time > 0,
    )
    coverage = arrays["trial_freeze_valid_fraction"]
    low_speed = arrays["trial_freeze_low_speed_fraction"]
    escape_class = trial_count > 0
    freeze_class = (
        ~escape_class
        & np.isfinite(coverage)
        & (coverage >= minimum_valid_fraction)
        & np.isfinite(low_speed)
        & (low_speed >= freeze_fraction_threshold)
    )
    expected_class = np.full(n_trials, 0, dtype=np.uint8)
    expected_class[
        ~escape_class & np.isfinite(coverage) & (coverage >= minimum_valid_fraction)
    ] = 3
    expected_class[freeze_class] = 2
    expected_class[escape_class] = 1
    if (
        np.any(gap_count > envelope)
        or np.any(arrays["trial_logged_active_id_unavailable_count"] > envelope)
        or not np.allclose(
            arrays["trial_gap_fraction"], expected_gap_fraction, equal_nan=False
        )
        or not np.array_equal(trial_count, observed_event_count)
        or not np.array_equal(
            arrays["trial_high_turn_escape_count"], observed_high_turn_count
        )
        or np.any(arrays["trial_high_turn_escape_count"] > trial_count)
        or np.any(arrays["trial_bout_count"] < trial_count)
        or not np.allclose(
            arrays["trial_escape_event_rate_per_min"],
            expected_rate,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        or not _finite_fraction(coverage)
        or not _finite_fraction(low_speed)
        or not _finite_fraction(arrays["trial_recapture_fraction"])
        or not np.array_equal(arrays["trial_escape_speed_class"], escape_class)
        or not np.array_equal(arrays["trial_freeze_candidate"], freeze_class)
        or not np.array_equal(arrays["trial_response_class_code"], expected_class)
    ):
        raise ValueError(
            "Escape/freeze persisted trial classification is inconsistent."
        )

    sweep = parameters.get("threshold_sweep_mm_s")
    if not isinstance(sweep, (list, tuple)) or not sweep:
        raise ValueError("Escape/freeze threshold sweep is invalid.")
    sweep_values = np.asarray(sweep, dtype=np.float64)
    n_thresholds = int(sweep_values.size)
    expected_sweep_rows = n_trials * n_thresholds
    expected_sweep_trial = np.repeat(np.arange(n_trials), n_thresholds)
    expected_sweep_threshold = np.tile(sweep_values, n_trials)
    sweep_count = arrays["sweep_escape_event_count"]
    sweep_rate = arrays["sweep_escape_event_rate_per_min"]
    expected_sweep_rate = np.divide(
        sweep_count.astype(np.float64),
        valid_time[expected_sweep_trial] / 60.0,
        out=np.full(n_sweep_rows, np.nan, dtype=np.float64),
        where=valid_time[expected_sweep_trial] > 0,
    )
    if (
        n_sweep_rows != expected_sweep_rows
        or not np.array_equal(arrays["sweep_row_id"], np.arange(n_sweep_rows))
        or not np.array_equal(arrays["sweep_trial_row_id"], expected_sweep_trial)
        or not np.allclose(
            arrays["sweep_speed_threshold_mm_s"], expected_sweep_threshold
        )
        or np.any(sweep_count < 0)
        or not np.allclose(
            sweep_rate,
            expected_sweep_rate,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    ):
        raise ValueError("Escape/freeze persisted threshold sweep is inconsistent.")

    scalar = {
        name: int(values[0])
        for name, values in arrays.items()
        if name in recording_specs
    }
    if (
        scalar["recording_trial_count"] != n_trials
        or scalar["recording_escape_trial_count"] != int(np.count_nonzero(escape_class))
        or scalar["recording_freeze_trial_count"] != int(np.count_nonzero(freeze_class))
        or scalar["recording_escape_event_count"] != n_events
        or scalar["recording_high_turn_escape_event_count"]
        != int(np.count_nonzero(arrays["event_high_turn"]))
        or scalar["recording_trace_usable_event_count"]
        != int(np.count_nonzero(trace_valid))
    ):
        raise ValueError("Escape/freeze recording totals are inconsistent.")
    if (
        identity_registry(scientific, "response_class")
        != EXPECTED_REGISTRIES["response_class"]
        or identity_registry(scientific, "trace_exclusion_reason")
        != EXPECTED_REGISTRIES["trace_exclusion_reason"]
    ):
        raise ValueError("Escape/freeze identity registries are unsupported.")
    return freeze(
        {
            **arrays,
            **scalar,
            "n_trials": n_trials,
            "n_events": n_events,
            "n_sweep_rows": n_sweep_rows,
            "n_thresholds": n_thresholds,
            "parameters": parameters,
            "response_registry": EXPECTED_REGISTRIES["response_class"],
            "trace_reason_registry": EXPECTED_REGISTRIES["trace_exclusion_reason"],
        }
    )


def _display_indices(size: int, *, limit: int) -> np.ndarray:
    """Bound persisted event rows for display without changing source order."""

    if size <= limit:
        return np.arange(size, dtype=np.int64)
    positions = np.linspace(0, size - 1, limit, dtype=np.int64)
    return np.unique(positions)


def _display_meta(
    projection: ExactChaserSuccessorProjection,
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    handle = projection.escape_freeze
    return {
        "escape_freeze_binding": {
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "scientific_payload_sha256": handle.scientific_payload_sha256,
            "sources": plain(handle.scientific_manifest.get("sources")),
            "deep_audited": handle.deep_audited,
            "verification_mode": handle.verification_mode,
            "receipt_digest": handle.receipt_digest,
            "verified_array_names": list(handle.verified_array_names),
        },
        "escape_freeze_display": {
            "recipe_id": ESCAPE_FREEZE_DISPLAY_RECIPE,
            "classifier_method_id": (
                "exact_trial_speed_escape_optional_high_turn_freeze_v1"
            ),
            "classifier_parameters": plain(values["parameters"]),
            "trial_rows": "persisted_exact_logged_controller_trials",
            "event_rows": "persisted_speed_thresholded_bout_x_chaser_events",
            "response_classes": "persisted_no_viewer_reclassification",
            "threshold_sweep": "persisted_no_viewer_threshold_recalculation",
            "event_trace_samples": "not_persisted_no_viewer_reconstruction",
            "display_projection_algorithm": ESCAPE_FREEZE_DISPLAY_ALGORITHM,
            "max_event_points": ESCAPE_FREEZE_MAX_EVENT_POINTS,
            "max_event_table_rows": ESCAPE_FREEZE_MAX_EVENT_TABLE_ROWS,
            "scientific_recomputation": False,
            "interpolation": "prohibited",
        },
        "projection_provenance": plain(projection.provenance),
    }


def _float_or_none(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def build_exact_escape_freeze_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render persisted exact trial outcomes, event facts, and sensitivity rows."""

    from plotly.subplots import make_subplots

    values = _escape_freeze_values(projection)
    parameters = values["parameters"]
    response_registry = values["response_registry"]
    trace_registry = values["trace_reason_registry"]
    meta = _display_meta(projection, values)
    ordinal = np.asarray(values["trial_ordinal"])
    response_code = np.asarray(values["trial_response_class_code"])

    outcomes = make_subplots(
        rows=2,
        cols=1,
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=(
            "Persisted response class by exact trial",
            "Persisted event rate and freeze-window evidence",
        ),
        vertical_spacing=0.18,
    )
    for code in range(4):
        rows = np.flatnonzero(response_code == code)
        outcomes.add_trace(
            go.Scatter(
                x=ordinal[rows],
                y=response_code[rows],
                mode="markers",
                name=response_registry[str(code)],
                marker={"color": _CLASS_COLORS[code], "size": 11},
                customdata=np.column_stack(
                    (
                        np.asarray(values["trial_logged_id"])[rows],
                        np.asarray(values["trial_chaser_identity_code"])[rows],
                        np.asarray(values["trial_escape_event_count"])[rows],
                        np.asarray(values["trial_gap_fraction"])[rows],
                    )
                ),
                hovertemplate=(
                    "ordinal=%{x}<br>class=%{text}<br>logged trial=%{customdata[0]}"
                    "<br>chaser=%{customdata[1]}<br>escape events=%{customdata[2]}"
                    "<br>gap fraction=%{customdata[3]:.3f}<extra></extra>"
                ),
                text=[response_registry[str(code)]] * rows.size,
            ),
            row=1,
            col=1,
        )
    outcomes.add_trace(
        go.Scatter(
            x=ordinal,
            y=np.asarray(values["trial_escape_event_rate_per_min"]),
            mode="lines+markers",
            name="escape event rate",
            line={"color": _CLASS_COLORS[1]},
            connectgaps=False,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    outcomes.add_trace(
        go.Scatter(
            x=ordinal,
            y=np.asarray(values["trial_freeze_low_speed_fraction"]),
            mode="lines+markers",
            name="low-speed fraction",
            line={"color": _CLASS_COLORS[2]},
            connectgaps=False,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    outcomes.add_trace(
        go.Scatter(
            x=ordinal,
            y=np.asarray(values["trial_freeze_valid_fraction"]),
            mode="lines+markers",
            name="freeze-window valid fraction",
            line={"color": "#72b7b2", "dash": "dash"},
            connectgaps=False,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    outcomes.add_hline(
        y=float(parameters["freeze_fraction_threshold"]),
        line_dash="dot",
        line_color=_CLASS_COLORS[2],
        row=2,
        col=1,
        secondary_y=True,
    )
    outcomes.add_hline(
        y=float(parameters["minimum_freeze_valid_fraction"]),
        line_dash="dot",
        line_color="#72b7b2",
        row=2,
        col=1,
        secondary_y=True,
    )
    outcomes.update_yaxes(
        title_text="persisted response class",
        tickmode="array",
        tickvals=[0, 1, 2, 3],
        ticktext=[response_registry[str(code)] for code in range(4)],
        row=1,
        col=1,
    )
    outcomes.update_xaxes(title_text="trial ordinal", row=1, col=1)
    outcomes.update_xaxes(title_text="trial ordinal", row=2, col=1)
    outcomes.update_yaxes(
        title_text="persisted escape events/min", row=2, col=1, secondary_y=False
    )
    outcomes.update_yaxes(
        title_text="persisted freeze-window fraction",
        range=[0, 1.05],
        row=2,
        col=1,
        secondary_y=True,
    )
    outcomes.update_layout(
        title="Exact controller-trial escape/freeze outcomes", meta=meta
    )

    event_rows = _display_indices(
        int(values["n_events"]), limit=ESCAPE_FREEZE_MAX_EVENT_POINTS
    )
    event_evidence = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Onset distance vs separation gain",
            "Peak speed vs body-frame turn",
            "Trigger latency vs recapture latency",
            "Trigger distance vs event-onset distance",
        ),
    )
    high_turn = np.asarray(values["event_high_turn"])
    for tier, label, color in (
        (False, "speed escape / dash", "#4c78a8"),
        (True, "speed escape / high turn", "#e45756"),
    ):
        rows = event_rows[high_turn[event_rows] == tier]
        event_evidence.add_trace(
            go.Scattergl(
                x=np.asarray(values["event_distance_at_onset_mm"])[rows],
                y=np.asarray(values["event_separation_gain_mm"])[rows],
                mode="markers",
                name=label,
                legendgroup=f"tier-{tier}",
                marker={"color": color, "size": 7, "opacity": 0.7},
                customdata=np.column_stack(
                    (
                        np.asarray(values["event_bout_id"])[rows],
                        np.asarray(values["event_controller_trial_row_id"])[rows],
                        np.asarray(values["event_trace_exclusion_reason_code"])[rows],
                    )
                ),
                hovertemplate=(
                    "onset=%{x:.3f}mm<br>gain=%{y:.3f}mm<br>bout=%{customdata[0]}"
                    "<br>trial row=%{customdata[1]}<br>trace reason=%{customdata[2]}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    directed_rows = event_rows[np.asarray(values["event_directed_valid"])[event_rows]]
    event_evidence.add_trace(
        go.Scattergl(
            x=np.asarray(values["event_peak_speed_mm_s"])[directed_rows],
            y=np.asarray(values["event_turn_deg"])[directed_rows],
            mode="markers",
            name="directed-valid event",
            marker={"color": "#f58518", "size": 7, "opacity": 0.7},
            customdata=np.asarray(values["event_bout_id"])[directed_rows],
            hovertemplate=(
                "peak=%{x:.3f}mm/s<br>turn=%{y:.3f}°<br>bout=%{customdata}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    latency_rows = event_rows[
        np.isfinite(np.asarray(values["event_latency_from_trigger_s"])[event_rows])
        & np.isfinite(np.asarray(values["event_recapture_latency_s"])[event_rows])
    ]
    event_evidence.add_trace(
        go.Scattergl(
            x=np.asarray(values["event_latency_from_trigger_s"])[latency_rows],
            y=np.asarray(values["event_recapture_latency_s"])[latency_rows],
            mode="markers",
            name="trace-usable recapture",
            marker={"color": "#54a24b", "size": 7, "opacity": 0.7},
            customdata=np.asarray(values["event_bout_id"])[latency_rows],
            hovertemplate=(
                "trigger→event=%{x:.3f}s<br>event→recapture=%{y:.3f}s"
                "<br>bout=%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    trigger_rows = event_rows[
        np.isfinite(np.asarray(values["event_trigger_distance_mm"])[event_rows])
    ]
    event_evidence.add_trace(
        go.Scattergl(
            x=np.asarray(values["event_trigger_distance_mm"])[trigger_rows],
            y=np.asarray(values["event_distance_at_onset_mm"])[trigger_rows],
            mode="markers",
            name="persisted trigger/event distance",
            marker={"color": "#b279a2", "size": 7, "opacity": 0.7},
            customdata=np.asarray(values["event_bout_id"])[trigger_rows],
            hovertemplate=(
                "trigger distance=%{x:.3f}mm<br>event onset=%{y:.3f}mm"
                "<br>bout=%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )
    event_evidence.add_hline(y=0.0, line_dash="dot", row=1, col=1)
    event_evidence.add_hline(y=0.0, line_dash="dot", row=1, col=2)
    event_evidence.update_xaxes(title_text="event-onset distance (mm)", row=1, col=1)
    event_evidence.update_yaxes(title_text="separation gain (mm)", row=1, col=1)
    event_evidence.update_xaxes(title_text="peak speed (mm/s)", row=1, col=2)
    event_evidence.update_yaxes(title_text="persisted turn (°)", row=1, col=2)
    event_evidence.update_xaxes(title_text="trigger→event latency (s)", row=2, col=1)
    event_evidence.update_yaxes(title_text="event→recapture latency (s)", row=2, col=1)
    event_evidence.update_xaxes(title_text="trigger distance (mm)", row=2, col=2)
    event_evidence.update_yaxes(title_text="event-onset distance (mm)", row=2, col=2)
    event_evidence.update_layout(
        title="Persisted escape-event outcome evidence", meta=meta
    )

    sensitivity = go.Figure()
    sweep_trial = np.asarray(values["sweep_trial_row_id"])
    for trial_row in range(int(values["n_trials"])):
        rows = np.flatnonzero(sweep_trial == trial_row)
        sensitivity.add_trace(
            go.Scatter(
                x=np.asarray(values["sweep_speed_threshold_mm_s"])[rows],
                y=np.asarray(values["sweep_escape_event_rate_per_min"])[rows],
                mode="lines+markers",
                name=f"trial row {trial_row}",
                customdata=np.asarray(values["sweep_escape_event_count"])[rows],
                hovertemplate=(
                    "threshold=%{x:.2f}mm/s<br>persisted rate=%{y:.3f}/min"
                    "<br>events=%{customdata}<extra></extra>"
                ),
                connectgaps=False,
            )
        )
    sensitivity.update_layout(
        title="Persisted escape-speed threshold sensitivity by exact trial",
        xaxis_title="persisted sweep threshold (mm/s)",
        yaxis_title="persisted escape event rate (/min)",
        meta=meta,
    )

    trial_records = []
    for row in range(int(values["n_trials"])):
        code = int(np.asarray(values["trial_response_class_code"])[row])
        trial_records.append(
            {
                "trial_row_id": row,
                "logged_trial_id": int(np.asarray(values["trial_logged_id"])[row]),
                "trial_ordinal": int(ordinal[row]),
                "chaser_identity_code": int(
                    np.asarray(values["trial_chaser_identity_code"])[row]
                ),
                "response_class": response_registry[str(code)],
                "escape_event_count": int(
                    np.asarray(values["trial_escape_event_count"])[row]
                ),
                "high_turn_escape_count": int(
                    np.asarray(values["trial_high_turn_escape_count"])[row]
                ),
                "escape_event_rate_per_min": _float_or_none(
                    np.asarray(values["trial_escape_event_rate_per_min"])[row]
                ),
                "freeze_valid_fraction": _float_or_none(
                    np.asarray(values["trial_freeze_valid_fraction"])[row]
                ),
                "freeze_low_speed_fraction": _float_or_none(
                    np.asarray(values["trial_freeze_low_speed_fraction"])[row]
                ),
                "first_escape_latency_s": _float_or_none(
                    np.asarray(values["trial_first_escape_latency_s"])[row]
                ),
                "trigger_distance_mm": _float_or_none(
                    np.asarray(values["trial_trigger_distance_mm"])[row]
                ),
                "mean_separation_gain_mm": _float_or_none(
                    np.asarray(values["trial_mean_separation_gain_mm"])[row]
                ),
                "recapture_fraction": _float_or_none(
                    np.asarray(values["trial_recapture_fraction"])[row]
                ),
                "valid_time_s": float(np.asarray(values["trial_valid_time_s"])[row]),
                "gap_fraction": float(np.asarray(values["trial_gap_fraction"])[row]),
                "logged_active_id_unavailable_count": int(
                    np.asarray(values["trial_logged_active_id_unavailable_count"])[row]
                ),
            }
        )
    table_rows = _display_indices(
        int(values["n_events"]), limit=ESCAPE_FREEZE_MAX_EVENT_TABLE_ROWS
    )
    event_records = []
    for row in table_rows:
        reason = int(np.asarray(values["event_trace_exclusion_reason_code"])[row])
        event_records.append(
            {
                "event_row_id": int(row),
                "bout_id": int(np.asarray(values["event_bout_id"])[row]),
                "trial_row_id": int(
                    np.asarray(values["event_controller_trial_row_id"])[row]
                ),
                "chaser_identity_code": int(
                    np.asarray(values["event_chaser_identity_code"])[row]
                ),
                "onset_acquisition_frame_id": int(
                    np.asarray(values["event_onset_acquisition_frame_id"])[row]
                ),
                "peak_speed_mm_s": float(
                    np.asarray(values["event_peak_speed_mm_s"])[row]
                ),
                "distance_at_onset_mm": float(
                    np.asarray(values["event_distance_at_onset_mm"])[row]
                ),
                "separation_gain_mm": float(
                    np.asarray(values["event_separation_gain_mm"])[row]
                ),
                "high_turn": bool(np.asarray(values["event_high_turn"])[row]),
                "trace_status": trace_registry[str(reason)],
                "recaptured": bool(np.asarray(values["event_recaptured"])[row]),
                "recapture_latency_s": _float_or_none(
                    np.asarray(values["event_recapture_latency_s"])[row]
                ),
            }
        )
    notice = mo.callout(
        (
            f"{values['recording_trial_count']} exact logged trials contain "
            f"{values['recording_escape_event_count']} persisted speed-defined escape "
            f"events at ≥{float(parameters['escape_speed_threshold_mm_s']):g} mm/s; "
            f"{values['recording_high_turn_escape_event_count']} also satisfy the "
            f"separate ≥{float(parameters['high_turn_threshold_deg']):g}° high-turn "
            "annotation. Freeze candidates use the persisted "
            f"{float(parameters['freeze_window_s']):g}s post-trigger window, ≤"
            f"{float(parameters['freeze_speed_threshold_mm_s']):g} mm/s low-speed "
            "fraction, and coverage gate. Response classes are never recomputed in "
            "the viewer. The successor does not persist aligned distance samples, so "
            "event trace trajectories are not reconstructed; only sealed recapture "
            "outcomes and trace-validity reasons are shown."
        ),
        kind="info",
    )
    return mo.vstack(
        [
            notice,
            outcomes,
            event_evidence,
            sensitivity,
            mo.md("### Persisted exact-trial outcome evidence"),
            mo.ui.table(trial_records, pagination=True, page_size=15),
            mo.md(
                "### Persisted escape-event evidence "
                f"(displaying {len(event_records):,} of {values['n_events']:,} rows)"
            ),
            mo.ui.table(event_records, pagination=True, page_size=15),
        ]
    )


__all__ = [
    "ESCAPE_FREEZE_DISPLAY_ALGORITHM",
    "ESCAPE_FREEZE_DISPLAY_RECIPE",
    "ESCAPE_FREEZE_MAX_EVENT_POINTS",
    "ESCAPE_FREEZE_MAX_EVENT_TABLE_ROWS",
    "_display_indices",
    "_escape_freeze_values",
    "build_exact_escape_freeze_output",
]
