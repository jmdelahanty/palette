"""Persisted distance CDF and geometric radial-mass views."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .array_requirements import DISTANCE_DISTRIBUTION_ARRAYS
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain

DISTANCE_DISTRIBUTION_DISPLAY_RECIPE = (
    "paired_provider_persisted_distance_cdf_and_geometric_mass_v1"
)
_COLORS = ("#1f77b4", "#d95f02")
_Stratum = tuple[int, int, int, int]


def _array(handle: Any, name: str, *, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(handle.array(name), dtype=dtype)
    except KeyError as exc:
        raise ValueError(
            f"Distance distributions lack required persisted array {name!r}."
        ) from exc


def _key_arrays(handle: Any, prefix: str) -> tuple[np.ndarray, ...]:
    return (
        _array(handle, f"{prefix}_epoch_role_code", dtype=np.int64),
        _array(handle, f"{prefix}_epoch_window_id", dtype=np.int64),
        _array(handle, f"{prefix}_behavior_role_code", dtype=np.int64),
        _array(handle, f"{prefix}_chaser_identity_code", dtype=np.int64),
    )


def _keys(columns: tuple[np.ndarray, ...], *, label: str) -> tuple[_Stratum, ...]:
    size = columns[0].size
    if any(column.ndim != 1 or column.size != size for column in columns):
        raise ValueError(f"{label} identity columns have inconsistent row axes.")
    return tuple(
        tuple(int(column[index]) for column in columns)  # type: ignore[return-value]
        for index in range(size)
    )


def _metric_support(handle: Any) -> Mapping[_Stratum, Mapping[str, int]]:
    keys = _keys(_key_arrays(handle, "metric"), label="Metric")
    columns = {
        name: _array(handle, f"metric_{name}", dtype=np.int64)
        for name in (
            "candidate_frame_count",
            "valid_distance_frame_count",
            "wall_excluded_valid_frame_count",
        )
    }
    if any(values.shape != (len(keys),) for values in columns.values()):
        raise ValueError("Distance-distribution support columns are inconsistent.")
    if any(np.any(values < 0) for values in columns.values()):
        raise ValueError("Distance-distribution support contains negative counts.")
    rows: dict[_Stratum, Mapping[str, int]] = {}
    for index, key in enumerate(keys):
        if key in rows:
            raise ValueError("Distance-distribution metric strata are duplicated.")
        candidate = int(columns["candidate_frame_count"][index])
        valid = int(columns["valid_distance_frame_count"][index])
        wall = int(columns["wall_excluded_valid_frame_count"][index])
        if valid > candidate or wall > valid:
            raise ValueError("Distance-distribution support is not nested.")
        rows[key] = {name: int(values[index]) for name, values in columns.items()}
    return rows


def _group_rows(
    handle: Any,
    *,
    prefix: str,
    columns: Mapping[str, np.ndarray],
) -> Mapping[_Stratum, Mapping[str, np.ndarray]]:
    keys = _keys(_key_arrays(handle, prefix), label=prefix.capitalize())
    if any(values.ndim != 1 or values.size != len(keys) for values in columns.values()):
        raise ValueError(f"{prefix.capitalize()} columns have inconsistent row axes.")
    grouped: dict[_Stratum, dict[str, list[Any]]] = defaultdict(
        lambda: {name: [] for name in columns}
    )
    for index, key in enumerate(keys):
        for name, values in columns.items():
            grouped[key][name].append(values[index])
    return {
        key: {
            name: np.asarray(values, dtype=columns[name].dtype)
            for name, values in row.items()
        }
        for key, row in grouped.items()
    }


def _validate_fraction(values: np.ndarray, *, label: str) -> None:
    finite = np.isfinite(values)
    if np.any(finite & ((values < 0.0) | (values > 1.0))):
        raise ValueError(f"{label} must be persisted fractions in [0, 1].")


def _distribution_tables(handle: Any) -> Mapping[str, Any]:
    """Validate exact persisted rows without histogram reconstruction."""

    handle.require_verified_arrays(DISTANCE_DISTRIBUTION_ARRAYS)
    support = _metric_support(handle)
    cdf = _group_rows(
        handle,
        prefix="cdf",
        columns={
            "threshold_mm": _array(handle, "cdf_threshold_mm", dtype=np.float64),
            "fraction": _array(handle, "cdf_fraction_at_or_below", dtype=np.float64),
        },
    )
    radial = _group_rows(
        handle,
        prefix="radial",
        columns={
            "start_mm": _array(handle, "radial_bin_start_mm", dtype=np.float64),
            "end_mm": _array(handle, "radial_bin_end_mm", dtype=np.float64),
            "observed_count": _array(handle, "radial_observed_count", dtype=np.int64),
            "observed_fraction": _array(
                handle, "radial_observed_fraction", dtype=np.float64
            ),
            "expected_area": _array(
                handle,
                "radial_expected_available_area_mm2_frames",
                dtype=np.float64,
            ),
            "expected_fraction": _array(
                handle, "radial_expected_fraction_geometric", dtype=np.float64
            ),
            "selection_index": _array(
                handle, "radial_selection_index_geometric", dtype=np.float64
            ),
            "wall_observed_count": _array(
                handle, "radial_wall_excluded_observed_count", dtype=np.int64
            ),
            "wall_observed_fraction": _array(
                handle,
                "radial_wall_excluded_observed_fraction",
                dtype=np.float64,
            ),
            "wall_expected_area": _array(
                handle,
                "radial_wall_excluded_expected_available_area_mm2_frames",
                dtype=np.float64,
            ),
            "wall_expected_fraction": _array(
                handle,
                "radial_wall_excluded_expected_fraction_geometric",
                dtype=np.float64,
            ),
            "wall_selection_index": _array(
                handle,
                "radial_wall_excluded_selection_index_geometric",
                dtype=np.float64,
            ),
        },
    )
    if not support or set(cdf) != set(support) or set(radial) != set(support):
        raise ValueError(
            "Distance-distribution metric, CDF, and radial strata disagree."
        )
    ordered_cdf: dict[_Stratum, Mapping[str, np.ndarray]] = {}
    ordered_radial: dict[_Stratum, Mapping[str, np.ndarray]] = {}
    for key in support:
        cdf_row = cdf[key]
        threshold = cdf_row["threshold_mm"]
        fraction = cdf_row["fraction"]
        if (
            threshold.size == 0
            or np.any(~np.isfinite(threshold))
            or np.any(np.diff(threshold) <= 0)
        ):
            raise ValueError("CDF thresholds must be finite and strictly increasing.")
        _validate_fraction(fraction, label="CDF values")
        finite_fraction = fraction[np.isfinite(fraction)]
        if finite_fraction.size > 1 and np.any(np.diff(finite_fraction) < -1e-12):
            raise ValueError("Persisted CDF values are not monotone.")
        if support[key]["valid_distance_frame_count"] > 0 and not np.all(
            np.isfinite(fraction)
        ):
            raise ValueError("A supported CDF contains missing persisted values.")
        ordered_cdf[key] = {
            "threshold_mm": threshold,
            "fraction": fraction,
        }

        radial_row = radial[key]
        row = dict(radial_row)
        if (
            row["start_mm"].size == 0
            or np.any(~np.isfinite(row["start_mm"]))
            or np.any(~np.isfinite(row["end_mm"]))
            or np.any(row["end_mm"] <= row["start_mm"])
            or np.any(np.diff(row["start_mm"]) <= 0)
            or not np.allclose(
                row["end_mm"][:-1],
                row["start_mm"][1:],
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise ValueError("Persisted radial bins are invalid or noncontiguous.")
        if np.any(row["observed_count"] < 0) or np.any(row["wall_observed_count"] < 0):
            raise ValueError("Persisted radial counts cannot be negative.")
        for name in (
            "observed_fraction",
            "expected_fraction",
            "wall_observed_fraction",
            "wall_expected_fraction",
        ):
            _validate_fraction(row[name], label=name)
        valid_count = support[key]["valid_distance_frame_count"]
        wall_count = support[key]["wall_excluded_valid_frame_count"]
        if int(np.sum(row["observed_count"])) != valid_count:
            raise ValueError("Radial observed counts do not conserve valid support.")
        if int(np.sum(row["wall_observed_count"])) != wall_count:
            raise ValueError(
                "Wall-excluded radial counts do not conserve wall-excluded support."
            )
        if valid_count and not np.isclose(
            np.nansum(row["observed_fraction"]), 1.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError("Radial observed fractions do not sum to one.")
        if wall_count and not np.isclose(
            np.nansum(row["wall_observed_fraction"]),
            1.0,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Wall-excluded observed fractions do not sum to one.")
        ordered_radial[key] = row
    return {
        "support": support,
        "cdf": ordered_cdf,
        "radial": ordered_radial,
    }


def _stratum_label(handle: Any, key: _Stratum) -> str:
    epochs = identity_registry(handle.scientific_manifest, "epoch_role")
    behaviors = identity_registry(handle.scientific_manifest, "behavior_role")
    chasers = identity_registry(handle.scientific_manifest, "chaser")
    return (
        f"{epochs.get(str(key[0]), f'epoch {key[0]}')}"
        f" · window {key[1]}"
        f" · {behaviors.get(str(key[2]), f'role {key[2]}')}"
        f" · {chasers.get(str(key[3]), f'chaser {key[3]}')}"
    )


def _display_record(
    projection: ExactChaserSuccessorProjection,
    tables: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> Mapping[str, Any]:
    manifests = [handle.scientific_manifest for handle in projection.radials]
    configs = [
        manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
        for manifest in manifests
    ]
    perimeter_values = {
        float(config["perimeter_band_mm"])
        for config in configs
        if "perimeter_band_mm" in config
    }
    if len(perimeter_values) != 1:
        raise ValueError(
            "Paired radial products do not declare one shared perimeter band."
        )
    keys = sorted(tables[0]["support"])
    strata = []
    for key in keys:
        cdf = tables[0]["cdf"][key]
        radial = tables[0]["radial"][key]
        edges = np.concatenate((radial["start_mm"], radial["end_mm"][-1:]))
        strata.append(
            {
                "epoch_role_code": key[0],
                "epoch_window_id": key[1],
                "behavior_role_code": key[2],
                "chaser_identity_code": key[3],
                "cdf_thresholds_mm": cdf["threshold_mm"].tolist(),
                "radial_bin_edges_mm": edges.tolist(),
            }
        )
    return {
        "recipe_id": DISTANCE_DISTRIBUTION_DISPLAY_RECIPE,
        "source_arrays": list(DISTANCE_DISTRIBUTION_ARRAYS),
        "providers": list(projection.provider_ids),
        "strata": strata,
        "provider_denominators": [
            {
                "provider_id": provider_id,
                "strata": [
                    {
                        "epoch_role_code": key[0],
                        "epoch_window_id": key[1],
                        "behavior_role_code": key[2],
                        "chaser_identity_code": key[3],
                        **dict(table["support"][key]),
                    }
                    for key in keys
                ],
            }
            for provider_id, table in zip(projection.provider_ids, tables, strict=True)
        ],
        "cdf_thresholds": "persisted_exact_no_interpolation",
        "radial_bin_edges": "persisted_exact_no_rebinning",
        "ordinary_denominator": "metric_valid_distance_frame_count",
        "wall_excluded_denominator": "metric_wall_excluded_valid_frame_count",
        "perimeter_band_mm": perimeter_values.pop(),
        "scientific_recomputation": False,
        "interpolation": "prohibited",
        "rebinning": "prohibited",
    }


def build_exact_distance_distributions_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render persisted CDFs and ordinary/wall-excluded radial mass."""

    tables = tuple(_distribution_tables(handle) for handle in projection.radials)
    if set(tables[0]["support"]) != set(tables[1]["support"]):
        raise ValueError(
            "Paired distance-distribution products expose different strata."
        )
    keys = sorted(tables[0]["support"])
    if not keys:
        raise ValueError("Distance-distribution products expose no strata.")
    for key in keys:
        if not np.array_equal(
            tables[0]["cdf"][key]["threshold_mm"],
            tables[1]["cdf"][key]["threshold_mm"],
        ):
            raise ValueError("Paired provider CDF thresholds differ.")
        for name in ("start_mm", "end_mm"):
            if not np.array_equal(
                tables[0]["radial"][key][name],
                tables[1]["radial"][key][name],
            ):
                raise ValueError("Paired provider radial bin edges differ.")

    cdf_figure = go.Figure()
    ordinary_figure = go.Figure()
    wall_figure = go.Figure()
    selection_figure = go.Figure()
    for provider_index, (provider_id, handle, provider_tables) in enumerate(
        zip(projection.provider_ids, projection.radials, tables, strict=True)
    ):
        color = _COLORS[provider_index]
        for stratum_index, key in enumerate(keys):
            label = _stratum_label(handle, key)
            cdf = provider_tables["cdf"][key]
            radial = provider_tables["radial"][key]
            support = provider_tables["support"][key]
            centers = (radial["start_mm"] + radial["end_mm"]) / 2.0
            legendgroup = f"{provider_index}-{stratum_index}"
            cdf_figure.add_trace(
                go.Scatter(
                    x=cdf["threshold_mm"],
                    y=100.0 * cdf["fraction"],
                    mode="lines+markers",
                    name=f"{provider_id} · {label}",
                    legendgroup=legendgroup,
                    line={"color": color, "dash": ("solid", "dash")[provider_index]},
                    customdata=np.full(
                        (cdf["threshold_mm"].size, 1),
                        support["valid_distance_frame_count"],
                    ),
                    hovertemplate=(
                        "threshold=%{x:.3g} mm<br>"
                        "P(distance ≤ threshold)=%{y:.4f}%<br>"
                        "valid rows=%{customdata[0]:,.0f}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            ordinary_custom = np.column_stack(
                (radial["observed_count"], radial["expected_area"])
            )
            for value_name, dash, display_name in (
                ("observed_fraction", "solid", "observed"),
                ("expected_fraction", "dot", "geometric expected"),
            ):
                ordinary_figure.add_trace(
                    go.Scatter(
                        x=centers,
                        y=100.0 * radial[value_name],
                        mode="lines",
                        name=f"{provider_id} · {label} · {display_name}",
                        legendgroup=legendgroup,
                        line={"color": color, "dash": dash},
                        customdata=ordinary_custom,
                        hovertemplate=(
                            "distance=%{x:.3g} mm<br>mass=%{y:.4f}%/bin<br>"
                            "observed count=%{customdata[0]:,.0f}<br>"
                            "available area×frames=%{customdata[1]:.4g}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            wall_custom = np.column_stack(
                (radial["wall_observed_count"], radial["wall_expected_area"])
            )
            for value_name, dash, display_name in (
                ("wall_observed_fraction", "solid", "observed"),
                ("wall_expected_fraction", "dot", "geometric expected"),
            ):
                wall_figure.add_trace(
                    go.Scatter(
                        x=centers,
                        y=100.0 * radial[value_name],
                        mode="lines",
                        name=f"{provider_id} · {label} · {display_name}",
                        legendgroup=legendgroup,
                        line={"color": color, "dash": dash},
                        customdata=wall_custom,
                        hovertemplate=(
                            "distance=%{x:.3g} mm<br>wall-excluded mass=%{y:.4f}%/bin<br>"
                            "observed count=%{customdata[0]:,.0f}<br>"
                            "available area×frames=%{customdata[1]:.4g}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            selection_figure.add_trace(
                go.Scatter(
                    x=centers,
                    y=radial["selection_index"],
                    mode="lines",
                    name=f"{provider_id} · {label} · ordinary",
                    legendgroup=legendgroup,
                    line={"color": color, "dash": "solid"},
                )
            )
            selection_figure.add_trace(
                go.Scatter(
                    x=centers,
                    y=radial["wall_selection_index"],
                    mode="lines",
                    name=f"{provider_id} · {label} · wall-excluded",
                    legendgroup=legendgroup,
                    line={"color": color, "dash": "dot"},
                )
            )

    display = _display_record(projection, tables)
    base_meta = plain(projection.provenance)
    base_meta["distance_distribution_display"] = display
    cdf_figure.update_layout(
        title="Empirical fish–chaser distance CDF",
        xaxis_title="distance threshold (mm)",
        yaxis_title="fraction at or below threshold (%)",
        meta=base_meta,
    )
    ordinary_figure.update_layout(
        title="Observed versus moving-reference geometric radial mass",
        xaxis_title="distance-bin center (mm)",
        yaxis_title="probability mass (%/bin)",
        meta=base_meta,
    )
    wall_figure.update_layout(
        title=(
            "Observed versus geometric radial mass · "
            f"{display['perimeter_band_mm']:g} mm wall-excluded"
        ),
        xaxis_title="distance-bin center (mm)",
        yaxis_title="probability mass (%/bin)",
        meta=base_meta,
    )
    selection_figure.update_layout(
        title="Persisted geometric selection index · ordinary and wall-excluded",
        xaxis_title="distance-bin center (mm)",
        yaxis_title="geometric selection index",
        meta=base_meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "Every threshold, radial edge, observed mass, moving-reference "
                "geometric expectation, and wall-excluded twin is persisted. "
                "The viewer performs no histogram reconstruction or rebinning.",
                kind="info",
            ),
            cdf_figure,
            ordinary_figure,
            wall_figure,
            selection_figure,
        ]
    )


__all__ = [
    "DISTANCE_DISTRIBUTION_DISPLAY_RECIPE",
    "_distribution_tables",
    "build_exact_distance_distributions_output",
]
