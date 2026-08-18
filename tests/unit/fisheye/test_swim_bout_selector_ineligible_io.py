from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis import swim_bout_schema
from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    load_default_swim_bout_tables,
    load_exact_selector_ineligible_default_swim_bout_tables,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_CONTRACT,
)
from tests.unit.fisheye.test_swim_bout_exact_schemas import _set_columnar_attrs


RUN_NAME = "bouts_selector_ineligible"


def _assign_rows(table: zarr.Group, values: dict[str, object]) -> None:
    """Assign deterministic values to the predeclared exact-schema columns."""

    field_names = list(table.attrs["field_names"])
    row_count = int(table[field_names[0]].shape[0])
    for field_name in field_names:
        array = table[field_name]
        value = values.get(field_name, 0)
        logical_dtype = np.dtype(table.attrs["field_dtypes"][field_name])
        if logical_dtype.kind == "S" and array.dtype == np.uint8 and array.ndim == 2:
            row_values = (
                list(value)
                if isinstance(value, (list, tuple, np.ndarray))
                else [value] * row_count
            )
            if len(row_values) != row_count:
                raise AssertionError(
                    f"Fixture field {field_name!r} expected {row_count} rows; "
                    f"got {len(row_values)}."
                )
            encoded = np.zeros(array.shape, dtype=np.uint8)
            for index, item in enumerate(row_values):
                raw = item if isinstance(item, bytes) else str(item).encode("utf-8")
                if len(raw) > int(array.shape[1]):
                    raise AssertionError(
                        f"Fixture value for {field_name!r} exceeds its frozen width."
                    )
                encoded[index, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
            array[:] = encoded
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            data = np.asarray(value, dtype=array.dtype)
        else:
            data = np.full(row_count, value, dtype=array.dtype)
        array[:] = data


def _create_spec_array(group: zarr.Group, spec: object) -> None:
    """Create the exact physical width needed by a frozen column spec."""

    parent = group
    parts = spec.path.split("/")
    for name in parts[:-1]:
        parent = parent.require_group(name)
    first_extent = (
        2
        if spec.axes[0] == "detector_signal"
        else (7 if spec.axes[0] == "frame" else 3)
    )
    if len(spec.axes) == 1:
        shape = (first_extent,)
    elif spec.axes[1] == "utf8_byte":
        shape = (first_extent, np.dtype(spec.logical_dtype).itemsize)
    else:
        shape = (first_extent, 7)
    parent.create_array(
        parts[-1],
        data=np.zeros(shape, dtype=np.dtype(spec.dtype)),
    )


def _build_selector_ineligible_v8_root() -> zarr.Group:
    """Build one strict, complete, manifest-valid selector-ineligible run."""

    root = zarr.group()
    parent = root.require_group("analysis").require_group("swim_bout_runs")
    parent.attrs["palette_completion_epoch"] = COMPLETION_EPOCH_REQUIRE_PROVENANCE
    run = parent.create_group(RUN_NAME)

    required = swim_bout_schema._required_specs()
    for spec in required.values():
        _create_spec_array(run, spec)
    optional = swim_bout_schema._optional_bundles()["embedded_frame_axis"]
    for spec in optional.values():
        _create_spec_array(run, spec)
    _set_columnar_attrs(
        run,
        {**required, **optional},
        swim_bout_schema._COLUMNAR_TABLE_PATHS,
    )
    run.attrs.update(
        {
            "schema_id": swim_bout_schema.SWIM_BOUT_RUN_SCHEMA_ID,
            "schema_version": swim_bout_schema.SWIM_BOUT_RUN_SCHEMA_VERSION,
            "layout": swim_bout_schema.SWIM_BOUT_LAYOUT,
            "default_candidate_id": 0,
            "default_signal_id": 1,
            "palette_run_completion_contract": RUN_COMPLETION_CONTRACT,
            "palette_run_completion_status": "complete",
            "palette_run_completed_at_utc": "2026-08-18T12:00:00+00:00",
            "palette_run_name": RUN_NAME,
            "palette_run_stage": "swim_bout_detection",
            "stage_selector_eligible": False,
        }
    )
    swim_bout_schema.write_swim_bout_array_manifest(run)

    _assign_rows(
        run["indexes/candidates"],
        {
            "candidate_id": 0,
            "candidate_name": b"candidate_default",
            "is_default": True,
            "detection_method": b"peak_event",
        },
    )
    _assign_rows(
        run["indexes/signal_variants"],
        {
            "signal_id": [0, 1, 2],
            "speed_level": [
                b"speed_filtered",
                b"speed_exponential",
                b"speed_smoothed",
            ],
            "signal_name": [b"filtered", b"exponential", b"smoothed"],
            "role": [
                b"physical_estimator",
                b"detector_response",
                b"physical_estimator",
            ],
            "source_level": [
                b"speed_filtered",
                b"speed_filtered",
                b"speed_smoothed",
            ],
        },
    )
    _assign_rows(
        run["tables/bouts"],
        {
            "candidate_id": 0,
            "signal_id": [1, 1, 1],
            "estimator_signal_id": 0,
            "track_id": 0,
            "bout_id": [7, 8, 9],
            "start_frame": [10, 30, 50],
            "end_frame": [20, 42, 65],
            "duration_s": [0.16, 0.20, 0.24],
            "path_length_mm": [1.5, 2.25, 3.0],
        },
    )
    _assign_rows(
        run["tables/peak_events"],
        {
            "candidate_id": 0,
            "signal_id": 1,
            "peak_event_id": [0, 1, 2],
            "bout_id": [7, 8, 9],
        },
    )
    _assign_rows(
        run["tables/inter_bout_intervals"],
        {
            "candidate_id": 0,
            "signal_id": 1,
            "interval_id": [0, 1, 2],
            "prev_bout_id": [7, 8, 9],
            "next_bout_id": [8, 9, 10],
        },
    )
    _assign_rows(
        run["tables/summary_metrics"],
        {
            "candidate_id": 0,
            "signal_id": 1,
            "metric_name": [b"n_bouts", b"mean_duration_s", b"mean_path_mm"],
            "value": [3.0, 0.2, 2.25],
            "units": [b"count", b"s", b"mm"],
            "source_table": b"bouts",
        },
    )
    _assign_rows(
        run["tables/histograms"],
        {
            "candidate_id": 0,
            "signal_id": 1,
            "metric_name": b"inter_bout_interval_s",
            "bin_left": 0.1,
            "bin_right": 0.3,
            "count": 1,
            "density": 1.0,
            "units": b"s",
        },
    )
    _assign_rows(
        run["tables/bout_points"],
        {
            "candidate_id": 0,
            "signal_id": 1,
            "bout_id": [7, 8, 9],
        },
    )
    run["signals/detector_signal_mm_s"][:] = np.asarray(
        [[0.0, 5.0, 8.0, 0.0, 6.0, 9.0, 0.0], [0.0] * 7],
        dtype=np.float32,
    )
    run["signals/detector_signal_signal_ids"][:] = np.asarray(
        [1, 0], dtype=np.int32
    )
    run["signals/frame_indices"][:] = np.arange(7, dtype=np.int64)
    return root


@pytest.mark.parametrize(
    "requested_run",
    (
        None,
        "",
        "latest",
        " latest ",
        "/analysis/swim_bout_runs/bouts_selector_ineligible",
        "analysis/swim_bout_runs/bouts_selector_ineligible",
        "bouts_selector_ineligible/",
        "../bouts_selector_ineligible",
        "bouts_selector_ineligible/../other",
    ),
)
def test_selector_ineligible_loader_requires_one_exact_bare_run_name(
    requested_run: str | None,
) -> None:
    root = _build_selector_ineligible_v8_root()

    with pytest.raises((SwimBoutIOError, TypeError)):
        load_exact_selector_ineligible_default_swim_bout_tables(
            root, run_name=requested_run  # type: ignore[arg-type]
        )


def test_selector_ineligible_loader_returns_default_candidate_and_signal_tables() -> None:
    root = _build_selector_ineligible_v8_root()

    payload = load_exact_selector_ineligible_default_swim_bout_tables(
        root, run_name=RUN_NAME
    )

    assert payload.run_name == RUN_NAME
    assert payload.candidate.candidate_id == 0
    assert payload.candidate.candidate_name == "candidate_default"
    assert payload.signal.signal_id == 1
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.signal.is_default is True
    assert payload.bouts["bout_id"].tolist() == [7, 8, 9]


@pytest.mark.parametrize(
    "mutation",
    (
        "eligible",
        "numeric_false",
        "string_false",
        "incomplete",
        "wrong_schema_version",
        "missing_manifest",
        "tampered_manifest",
    ),
)
def test_selector_ineligible_loader_requires_current_exact_complete_ineligible_run(
    mutation: str,
) -> None:
    root = _build_selector_ineligible_v8_root()
    run = root[f"analysis/swim_bout_runs/{RUN_NAME}"]
    if mutation == "eligible":
        run.attrs["stage_selector_eligible"] = True
    elif mutation == "numeric_false":
        run.attrs["stage_selector_eligible"] = 0
    elif mutation == "string_false":
        run.attrs["stage_selector_eligible"] = "false"
    elif mutation == "incomplete":
        run.attrs["palette_run_completion_status"] = "running"
    elif mutation == "wrong_schema_version":
        run.attrs["schema_version"] = 7
    elif mutation == "missing_manifest":
        del run.attrs["array_schema_manifest"]
    elif mutation == "tampered_manifest":
        manifest = dict(run.attrs["array_schema_manifest"])
        manifest["payload"]["arrays"][0]["path"] = "invented"
        run.attrs["array_schema_manifest"] = manifest

    with pytest.raises(SwimBoutIOError):
        load_exact_selector_ineligible_default_swim_bout_tables(
            root, run_name=RUN_NAME
        )


def test_ordinary_default_loader_rejects_the_same_selector_ineligible_run() -> None:
    root = _build_selector_ineligible_v8_root()

    with pytest.raises(SwimBoutIOError, match="selector-eligible"):
        load_default_swim_bout_tables(root, run_name=RUN_NAME)
