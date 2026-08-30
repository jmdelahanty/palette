"""Persisted generalized bout-response summaries and bout-level views."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..chaser_exact_bout_response_contract import EXPECTED_REGISTRIES
from .array_requirements import BOUT_RESPONSE_ARRAYS
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import freeze, plain

BOUT_RESPONSE_DISPLAY_RECIPE = "persisted_exact_bout_by_chaser_response_view_v1"
BOUT_RESPONSE_DISPLAY_ALGORITHM = "source_order_uniform_endpoint_preserving_v1"
BOUT_RESPONSE_MAX_POINTS_PER_SERIES = 6_000

_ROLE_COLORS = {
    1: "#4c78a8",
    2: "#e45756",
    3: "#54a24b",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Generalized bout-response successor lacks {label}.")
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
            f"Generalized bout-response successor lacks required array {name!r}."
        ) from exc
    if values.dtype != np.dtype(dtype) or values.shape != shape:
        raise ValueError(
            f"Generalized bout-response array {name!r} has incompatible dtype or shape."
        )
    return values


def _distance_edges(value: Any) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= 65:
        raise ValueError("Generalized bout-response distance-bin edges are invalid.")
    edges: list[float] = []
    for index, item in enumerate(value):
        if item is None:
            if index != len(value) - 1:
                raise ValueError(
                    "Only the final bout-response distance-bin edge may be open."
                )
            edges.append(float("inf"))
        elif type(item) in {int, float} and np.isfinite(float(item)):
            edges.append(float(item))
        else:
            raise ValueError(
                "Generalized bout-response distance-bin edges are invalid."
            )
    result = np.asarray(edges, dtype=np.float64)
    if np.any(np.diff(result) <= 0):
        raise ValueError(
            "Generalized bout-response distance-bin edges are not increasing."
        )
    return result


def _bout_response_values(
    projection: ExactChaserSuccessorProjection,
) -> Mapping[str, Any]:
    """Validate persisted row identities, exact attachments, and summaries."""

    handle = projection.generalized_bout_response
    if handle is None:
        raise ValueError(
            "Generalized bout-response display requires one exact persisted successor."
        )
    if handle.successor_kind != "generalized_chaser_bout_response":
        raise ValueError("Bout-response projection names another successor kind.")
    handle.require_verified_arrays(BOUT_RESPONSE_ARRAYS)
    scientific = _mapping(handle.scientific_manifest, label="scientific manifest")
    schema = _mapping(scientific.get("scientific_schema"), label="scientific schema")
    dimensions = _mapping(scientific.get("dimensions"), label="dimensions")
    n_bouts = int(dimensions.get("n_bouts", -1))
    n_chasers = int(dimensions.get("n_chasers", -1))
    n_rows = int(dimensions.get("n_bout_chaser_rows", -1))
    n_summary = int(dimensions.get("n_summary_rows", -1))
    if n_bouts < 0 or n_chasers <= 0 or n_rows != n_bouts * n_chasers or n_summary <= 0:
        raise ValueError("Bout-response dimensions are incompatible.")
    row_specs = {
        "bout_chaser_row_id": np.int64,
        "bout_row_id": np.int64,
        "bout_id": np.int64,
        "chaser_identity_code": np.uint16,
        "source_signal_id": np.int32,
        "start_acquisition_frame_id": np.int64,
        "end_acquisition_frame_id": np.int64,
        "semantic_role_code": np.uint8,
        "controller_trial_row_id": np.int64,
        "controller_trial_envelope_row_id": np.int64,
        "controller_trial_gap_reason_code": np.uint8,
        "attachment_reason_code": np.uint8,
        "base_valid": bool,
        "directed_valid": bool,
        "distance_at_onset_mm": np.float32,
        "distance_at_end_mm": np.float32,
        "delta_distance_mm": np.float32,
        "bearing_at_onset_deg": np.float32,
        "turn_deg": np.float32,
        "turn_toward_chaser": bool,
        "bout_peak_speed_mm_s": np.float32,
        "bout_mean_speed_mm_s": np.float32,
        "bout_duration_s": np.float32,
        "bout_path_length_mm": np.float32,
        "bout_net_displacement_mm": np.float32,
        "bout_tortuosity": np.float32,
    }
    summary_specs = {
        "summary_role_code": np.uint8,
        "summary_chaser_identity_code": np.uint16,
        "summary_distance_bin_index": np.int16,
        "summary_distance_bin_start_mm": np.float32,
        "summary_distance_bin_end_mm": np.float32,
        "summary_valid_time_s": np.float64,
        "summary_bout_count": np.int64,
        "summary_bout_rate_per_min": np.float64,
        "summary_median_peak_speed_mm_s": np.float64,
        "summary_median_duration_s": np.float64,
        "summary_median_path_length_mm": np.float64,
        "summary_median_net_displacement_mm": np.float64,
    }
    arrays = {
        name: _array(handle, name, dtype=dtype, shape=(n_rows,))
        for name, dtype in row_specs.items()
    }
    arrays.update(
        {
            name: _array(handle, name, dtype=dtype, shape=(n_summary,))
            for name, dtype in summary_specs.items()
        }
    )
    if not np.array_equal(arrays["bout_chaser_row_id"], np.arange(n_rows)):
        raise ValueError("Bout-response row identity is not canonical.")
    expected_bout_rows = np.repeat(np.arange(n_bouts, dtype=np.int64), n_chasers)
    if not np.array_equal(arrays["bout_row_id"], expected_bout_rows):
        raise ValueError("Bout-response bout-row identity is inconsistent.")
    repeated_names = (
        "bout_id",
        "source_signal_id",
        "start_acquisition_frame_id",
        "end_acquisition_frame_id",
        "semantic_role_code",
        "bout_peak_speed_mm_s",
        "bout_mean_speed_mm_s",
        "bout_duration_s",
        "bout_path_length_mm",
        "bout_net_displacement_mm",
        "bout_tortuosity",
    )
    if n_bouts:
        for name in repeated_names:
            matrix = arrays[name].reshape(n_bouts, n_chasers)
            if not np.array_equal(
                matrix,
                np.broadcast_to(matrix[:, :1], matrix.shape),
                equal_nan=matrix.dtype.kind == "f",
            ):
                raise ValueError(
                    f"Bout-response array {name!r} changes within one bout."
                )
        bout_ids = arrays["bout_id"].reshape(n_bouts, n_chasers)[:, 0]
        if np.unique(bout_ids).size != n_bouts:
            raise ValueError("Bout-response selected bout identities are duplicated.")
        chaser_matrix = arrays["chaser_identity_code"].reshape(n_bouts, n_chasers)
        if not np.array_equal(
            chaser_matrix, np.broadcast_to(chaser_matrix[:1], chaser_matrix.shape)
        ):
            raise ValueError("Bout-response chaser identity changes between bouts.")
        chaser_codes = chaser_matrix[0]
    else:
        chaser_codes = np.unique(arrays["summary_chaser_identity_code"])
    if chaser_codes.size != n_chasers or np.unique(chaser_codes).size != n_chasers:
        raise ValueError("Bout-response chaser identity registry is inconsistent.")
    sources = _mapping(scientific.get("sources"), label="sources")
    swim_bouts = _mapping(sources.get("swim_bouts"), label="swim-bout source")
    if n_rows and (
        np.unique(arrays["source_signal_id"]).size != 1
        or int(arrays["source_signal_id"][0]) != int(swim_bouts.get("signal_id", -1))
    ):
        raise ValueError("Bout-response rows use another swim-bout signal.")
    roles = arrays["semantic_role_code"]
    trial = arrays["controller_trial_row_id"]
    envelope = arrays["controller_trial_envelope_row_id"]
    gap_reason = arrays["controller_trial_gap_reason_code"]
    attachment = arrays["attachment_reason_code"]
    member = trial >= 0
    envelope_member = envelope >= 0
    gap_member = envelope_member & ~member
    if (
        np.any(~np.isin(roles, np.asarray([0, 1, 2, 3], dtype=np.uint8)))
        or np.any(trial < -1)
        or np.any(envelope < -1)
        or np.any(gap_reason > 6)
        or np.any(attachment > 3)
        or np.any(member & (trial != envelope))
        or np.any(gap_member & (gap_reason == 0))
        or np.any(~gap_member & (gap_reason != 0))
        or np.any((attachment == 0) & ((roles == 0) | ~member))
        or np.any((attachment == 1) & ((roles != 0) | member | envelope_member))
        or np.any((attachment == 2) & (roles != 0))
        or np.any((attachment == 3) & ((roles == 0) | member))
    ):
        raise ValueError(
            "Bout-response semantic or controller attachment evidence is inconsistent."
        )
    onset = arrays["distance_at_onset_mm"]
    end = arrays["distance_at_end_mm"]
    delta = arrays["delta_distance_mm"]
    finite_pair = np.isfinite(onset) & np.isfinite(end)
    if np.any(arrays["base_valid"] & (~finite_pair | (roles == 0))) or np.any(
        finite_pair
        & ~np.isclose(
            delta.astype(np.float64),
            end.astype(np.float64) - onset.astype(np.float64),
            rtol=2e-5,
            atol=2e-5,
        )
    ):
        raise ValueError("Bout-response distance response evidence is inconsistent.")
    for name in (
        "bout_peak_speed_mm_s",
        "bout_mean_speed_mm_s",
        "bout_duration_s",
        "bout_path_length_mm",
        "bout_net_displacement_mm",
    ):
        values = arrays[name]
        if np.any(np.isfinite(values) & (values < 0)):
            raise ValueError(f"Bout-response metric {name!r} is negative.")
    directed = arrays["directed_valid"]
    bearing = arrays["bearing_at_onset_deg"]
    turn = arrays["turn_deg"]
    toward = arrays["turn_toward_chaser"]
    body_present = schema.get("body_extension_present")
    if type(body_present) is not bool:
        raise ValueError("Bout-response body-extension declaration is invalid.")
    if (
        np.any(
            directed
            & (~arrays["base_valid"] | ~np.isfinite(bearing) | ~np.isfinite(turn))
        )
        or np.any(~directed & toward)
        or np.any(directed & ((bearing < -180) | (bearing > 180)))
        or np.any(directed & ((turn < -180) | (turn > 180)))
        or (
            not body_present
            and (
                np.any(directed)
                or np.any(np.isfinite(bearing))
                or np.any(np.isfinite(turn))
                or np.any(toward)
            )
        )
    ):
        raise ValueError("Bout-response directed body-frame evidence is inconsistent.")
    if (
        identity_registry(scientific, "semantic_role")
        != EXPECTED_REGISTRIES["semantic_role"]
        or identity_registry(scientific, "attachment_reason")
        != EXPECTED_REGISTRIES["attachment_reason"]
    ):
        raise ValueError("Bout-response identity registries are unsupported.")
    edges = _distance_edges(scientific.get("distance_bin_edges_mm"))
    n_bands = edges.size - 1
    sorted_chasers = np.sort(chaser_codes.astype(np.uint16))
    expected_roles = np.repeat(
        np.asarray([1, 2, 3], dtype=np.uint8), n_chasers * n_bands
    )
    expected_chasers = np.tile(np.repeat(sorted_chasers, n_bands), 3)
    expected_bands = np.tile(np.arange(n_bands, dtype=np.int16), 3 * n_chasers)
    expected_starts = edges[:-1][expected_bands]
    expected_ends = edges[1:][expected_bands]
    valid_time = arrays["summary_valid_time_s"]
    count = arrays["summary_bout_count"]
    rate = arrays["summary_bout_rate_per_min"]
    expected_rate = np.divide(
        count.astype(np.float64),
        valid_time / 60.0,
        out=np.full(n_summary, np.nan, dtype=np.float64),
        where=valid_time > 0,
    )
    if (
        n_summary != 3 * n_chasers * n_bands
        or not np.array_equal(arrays["summary_role_code"], expected_roles)
        or not np.array_equal(arrays["summary_chaser_identity_code"], expected_chasers)
        or not np.array_equal(arrays["summary_distance_bin_index"], expected_bands)
        or not np.allclose(
            arrays["summary_distance_bin_start_mm"], expected_starts, equal_nan=True
        )
        or not np.allclose(
            arrays["summary_distance_bin_end_mm"], expected_ends, equal_nan=True
        )
        or np.any(valid_time < 0)
        or np.any(count < 0)
        or not np.allclose(rate, expected_rate, rtol=1e-12, atol=1e-12, equal_nan=True)
    ):
        raise ValueError(
            "Bout-response persisted distance-band summary is inconsistent."
        )
    return freeze(
        {
            **arrays,
            "n_bouts": n_bouts,
            "n_chasers": n_chasers,
            "n_rows": n_rows,
            "n_summary": n_summary,
            "n_bands": n_bands,
            "distance_edges_mm": edges,
            "chaser_codes": chaser_codes,
            "body_extension_present": body_present,
            "semantic_registry": EXPECTED_REGISTRIES["semantic_role"],
            "attachment_registry": EXPECTED_REGISTRIES["attachment_reason"],
            "source_signal_level": swim_bouts.get("signal_level"),
        }
    )


def _display_indices(indices: np.ndarray) -> np.ndarray:
    """Bound one raw persisted series without changing scientific summaries."""

    if indices.size <= BOUT_RESPONSE_MAX_POINTS_PER_SERIES:
        return indices
    positions = np.linspace(
        0,
        indices.size - 1,
        BOUT_RESPONSE_MAX_POINTS_PER_SERIES,
        dtype=np.int64,
    )
    return indices[np.unique(positions)]


def _band_label(low: float, high: float) -> str:
    return f"[{low:g}, ∞)" if np.isinf(high) else f"[{low:g}, {high:g})"


def _display_meta(
    projection: ExactChaserSuccessorProjection,
    values: Mapping[str, Any],
) -> Mapping[str, Any]:
    handle = projection.generalized_bout_response
    base_valid_count = int(np.count_nonzero(values["base_valid"]))
    directed_valid_count = int(np.count_nonzero(values["directed_valid"]))
    return {
        "bout_response_binding": {
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
            "scientific_payload_sha256": handle.scientific_payload_sha256,
            "sources": plain(handle.scientific_manifest.get("sources")),
            "deep_audited": handle.deep_audited,
            "verification_mode": handle.verification_mode,
            "receipt_digest": handle.receipt_digest,
            "verified_array_names": list(handle.verified_array_names),
        },
        "bout_response_display": {
            "recipe_id": BOUT_RESPONSE_DISPLAY_RECIPE,
            "summary_arrays": "persisted_no_rebinning_or_reaggregation",
            "bout_rows": "persisted_selected_swim_bout_x_chaser",
            "response_interval": "persisted_bout_start_to_end",
            "attachment": "exact_acquisition_frame_identity_at_bout_onset",
            "trial_membership": "exact_onset_row_only",
            "trial_envelope_gaps": "retained_nonmembers_not_events",
            "display_projection_algorithm": BOUT_RESPONSE_DISPLAY_ALGORITHM,
            "max_points_per_role_chaser_series": (BOUT_RESPONSE_MAX_POINTS_PER_SERIES),
            "body_extension_present": values["body_extension_present"],
            "base_valid_row_count": base_valid_count,
            "directed_valid_row_count": directed_valid_count,
            "base_valid_rows_without_directed_axis": (
                base_valid_count - directed_valid_count
            ),
            "body_frame_fallback": "prohibited",
            "bout_resegmentation": "prohibited",
            "scientific_recomputation": False,
            "interpolation": "prohibited",
        },
        "projection_provenance": plain(projection.provenance),
    }


def build_exact_bout_response_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render persisted distance-stratified and body-frame bout responses."""

    from plotly.subplots import make_subplots

    values = _bout_response_values(projection)
    n_chasers = int(values["n_chasers"])
    n_bands = int(values["n_bands"])
    chaser_codes = np.asarray(values["chaser_codes"])
    sorted_chasers = np.sort(chaser_codes)
    role_registry = values["semantic_registry"]
    edges = np.asarray(values["distance_edges_mm"])
    band_labels = [_band_label(edges[i], edges[i + 1]) for i in range(n_bands)]
    meta = _display_meta(projection, values)

    rate = make_subplots(
        rows=1,
        cols=n_chasers,
        subplot_titles=[f"chaser {int(code)}" for code in sorted_chasers],
    )
    summary_role = np.asarray(values["summary_role_code"])
    summary_chaser = np.asarray(values["summary_chaser_identity_code"])
    summary_band = np.asarray(values["summary_distance_bin_index"])
    for column, chaser_code in enumerate(sorted_chasers, start=1):
        for role_code in (1, 2, 3):
            mask = (summary_chaser == chaser_code) & (summary_role == role_code)
            order = np.argsort(summary_band[mask])
            rows = np.flatnonzero(mask)[order]
            custom = np.column_stack(
                (
                    np.asarray(values["summary_bout_count"])[rows],
                    np.asarray(values["summary_valid_time_s"])[rows],
                )
            )
            rate.add_trace(
                go.Scatter(
                    x=summary_band[rows],
                    y=np.asarray(values["summary_bout_rate_per_min"])[rows],
                    mode="lines+markers",
                    name=role_registry[str(role_code)],
                    legendgroup=f"role-{role_code}",
                    showlegend=column == 1,
                    line={"color": _ROLE_COLORS[role_code]},
                    customdata=custom,
                    hovertemplate=(
                        "band=%{x}<br>rate=%{y:.3f}/min<br>bouts=%{customdata[0]}"
                        "<br>valid time=%{customdata[1]:.2f}s<extra></extra>"
                    ),
                    connectgaps=False,
                ),
                row=1,
                col=column,
            )
        rate.update_xaxes(
            title_text="onset distance band (mm)",
            tickmode="array",
            tickvals=list(range(n_bands)),
            ticktext=band_labels,
            row=1,
            col=column,
        )
        rate.update_yaxes(title_text="persisted bout rate (/min)", row=1, col=column)
    rate.update_layout(
        title="Bout rate by exact semantic role and onset distance",
        meta=meta,
    )

    metric_specs = (
        ("summary_median_peak_speed_mm_s", "median peak speed (mm/s)"),
        ("summary_median_duration_s", "median duration (s)"),
        ("summary_median_path_length_mm", "median path length (mm)"),
        ("summary_median_net_displacement_mm", "median displacement (mm)"),
    )
    kinematics = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[label for _, label in metric_specs],
    )
    for metric_index, (name, label) in enumerate(metric_specs):
        plot_row = metric_index // 2 + 1
        plot_column = metric_index % 2 + 1
        for chaser_index, chaser_code in enumerate(sorted_chasers):
            for role_code in (1, 2, 3):
                mask = (summary_chaser == chaser_code) & (summary_role == role_code)
                rows = np.flatnonzero(mask)[np.argsort(summary_band[mask])]
                kinematics.add_trace(
                    go.Scatter(
                        x=summary_band[rows],
                        y=np.asarray(values[name])[rows],
                        mode="lines+markers",
                        name=f"{role_registry[str(role_code)]} · chaser {int(chaser_code)}",
                        legendgroup=f"role-{role_code}-chaser-{int(chaser_code)}",
                        showlegend=metric_index == 0,
                        line={
                            "color": _ROLE_COLORS[role_code],
                            "dash": "solid" if chaser_index == 0 else "dash",
                        },
                        connectgaps=False,
                    ),
                    row=plot_row,
                    col=plot_column,
                )
        kinematics.update_xaxes(
            title_text="onset distance band (mm)",
            tickmode="array",
            tickvals=list(range(n_bands)),
            ticktext=band_labels,
            row=plot_row,
            col=plot_column,
        )
        kinematics.update_yaxes(title_text=label, row=plot_row, col=plot_column)
    kinematics.update_layout(
        title="Persisted median bout kinematics by onset distance",
        meta=meta,
    )

    response = make_subplots(
        rows=2,
        cols=n_chasers,
        subplot_titles=[
            *[f"distance response · chaser {int(code)}" for code in chaser_codes],
            *[f"body-frame turn · chaser {int(code)}" for code in chaser_codes],
        ],
        vertical_spacing=0.16,
    )
    row_chaser = np.asarray(values["chaser_identity_code"])
    row_role = np.asarray(values["semantic_role_code"])
    for column, chaser_code in enumerate(chaser_codes, start=1):
        for role_code in (1, 2, 3):
            common = (row_chaser == chaser_code) & (row_role == role_code)
            base_rows = _display_indices(
                np.flatnonzero(
                    common
                    & np.asarray(values["base_valid"])
                    & np.isfinite(np.asarray(values["distance_at_onset_mm"]))
                    & np.isfinite(np.asarray(values["delta_distance_mm"]))
                )
            )
            response.add_trace(
                go.Scattergl(
                    x=np.asarray(values["distance_at_onset_mm"])[base_rows],
                    y=np.asarray(values["delta_distance_mm"])[base_rows],
                    mode="markers",
                    name=role_registry[str(role_code)],
                    legendgroup=f"response-role-{role_code}",
                    showlegend=column == 1,
                    marker={
                        "color": _ROLE_COLORS[role_code],
                        "size": 5,
                        "opacity": 0.55,
                    },
                    customdata=np.column_stack(
                        (
                            np.asarray(values["bout_id"])[base_rows],
                            np.asarray(values["start_acquisition_frame_id"])[base_rows],
                            np.asarray(values["end_acquisition_frame_id"])[base_rows],
                            np.asarray(values["controller_trial_row_id"])[base_rows],
                        )
                    ),
                    hovertemplate=(
                        "onset=%{x:.3f}mm<br>Δdistance=%{y:.3f}mm"
                        "<br>bout=%{customdata[0]}<br>frames=%{customdata[1]}–%{customdata[2]}"
                        "<br>exact trial row=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            directed_rows = _display_indices(
                np.flatnonzero(common & np.asarray(values["directed_valid"]))
            )
            response.add_trace(
                go.Scattergl(
                    x=np.asarray(values["bearing_at_onset_deg"])[directed_rows],
                    y=np.asarray(values["turn_deg"])[directed_rows],
                    mode="markers",
                    name=f"{role_registry[str(role_code)]} body frame",
                    legendgroup=f"directed-role-{role_code}",
                    showlegend=False,
                    marker={
                        "color": _ROLE_COLORS[role_code],
                        "size": 5,
                        "opacity": 0.55,
                    },
                    customdata=np.column_stack(
                        (
                            np.asarray(values["bout_id"])[directed_rows],
                            np.asarray(values["turn_toward_chaser"])[directed_rows],
                        )
                    ),
                    hovertemplate=(
                        "bearing=%{x:.2f}°<br>turn=%{y:.2f}°"
                        "<br>bout=%{customdata[0]}<br>turn toward=%{customdata[1]}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=column,
            )
        response.add_hline(y=0.0, line_dash="dot", line_color="#666", row=1, col=column)
        response.add_hline(y=0.0, line_dash="dot", line_color="#666", row=2, col=column)
        response.add_vline(x=0.0, line_dash="dot", line_color="#666", row=2, col=column)
        response.update_xaxes(title_text="onset distance (mm)", row=1, col=column)
        response.update_yaxes(title_text="end − onset distance (mm)", row=1, col=column)
        response.update_xaxes(
            title_text="chaser bearing at onset (°)", row=2, col=column
        )
        response.update_yaxes(title_text="bout turn (°)", row=2, col=column)
        if not values["body_extension_present"]:
            response.add_annotation(
                text="Body-frame extension absent; no heading fallback is allowed.",
                x=0.5,
                y=0.5,
                xref=f"x{n_chasers + column} domain",
                yref=f"y{n_chasers + column} domain",
                showarrow=False,
            )
    response.update_layout(
        title="Persisted bout-level distance and optional body-frame response",
        meta=meta,
    )

    summary_records = []
    for index in range(int(values["n_summary"])):
        role_code = int(np.asarray(values["summary_role_code"])[index])
        summary_records.append(
            {
                "semantic_role": role_registry[str(role_code)],
                "chaser_identity_code": int(
                    np.asarray(values["summary_chaser_identity_code"])[index]
                ),
                "onset_distance_band_mm": _band_label(
                    float(np.asarray(values["summary_distance_bin_start_mm"])[index]),
                    float(np.asarray(values["summary_distance_bin_end_mm"])[index]),
                ),
                "bout_count": int(np.asarray(values["summary_bout_count"])[index]),
                "valid_time_s": float(
                    np.asarray(values["summary_valid_time_s"])[index]
                ),
                "bout_rate_per_min": float(
                    np.asarray(values["summary_bout_rate_per_min"])[index]
                ),
                "median_peak_speed_mm_s": float(
                    np.asarray(values["summary_median_peak_speed_mm_s"])[index]
                ),
                "median_duration_s": float(
                    np.asarray(values["summary_median_duration_s"])[index]
                ),
                "median_path_length_mm": float(
                    np.asarray(values["summary_median_path_length_mm"])[index]
                ),
                "median_net_displacement_mm": float(
                    np.asarray(values["summary_median_net_displacement_mm"])[index]
                ),
            }
        )
    body_status = (
        (
            "present; "
            f"{int(np.count_nonzero(values['directed_valid']))} of "
            f"{int(np.count_nonzero(values['base_valid']))} base-valid rows have "
            "persisted valid directed axes, while present-invalid axes remain excluded"
        )
        if values["body_extension_present"]
        else "absent; directed plots remain empty and no motion-heading fallback is used"
    )
    notice = mo.callout(
        (
            f"Exact signal `{values['source_signal_level']}` contributes "
            f"{values['n_bouts']} persisted bouts × {n_chasers} chasers. "
            "Rates and medians are read from the sealed distance-band summaries; "
            "the scatter plots show persisted bout rows with display-only thinning. "
            f"Body-frame extension: {body_status}."
        ),
        kind="info",
    )
    return mo.vstack(
        [
            notice,
            rate,
            kinematics,
            response,
            mo.md("### Persisted distance-band summary evidence"),
            mo.ui.table(summary_records, pagination=True, page_size=15),
        ]
    )


__all__ = [
    "BOUT_RESPONSE_DISPLAY_ALGORITHM",
    "BOUT_RESPONSE_DISPLAY_RECIPE",
    "BOUT_RESPONSE_MAX_POINTS_PER_SERIES",
    "_bout_response_values",
    "_display_indices",
    "build_exact_bout_response_output",
]
