from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis.exact_tabular_storage import (
    build_exact_tabular_storage_receipt,
    persist_exact_tabular_storage_receipt,
    rematerialize_exact_tabular_candidate,
    validate_exact_tabular_storage_receipt,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.array_contracts import FLOAT32, INT64, ArrayContract
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def _declaration(
    path: str,
    *,
    dtype: object,
    shape: tuple[str | int, ...],
    axes: tuple[str, ...],
    access: AccessPattern,
) -> AnalysisArrayDeclaration:
    return AnalysisArrayDeclaration(
        path=path,
        contract=ArrayContract(
            schema_id="palette.test.exact_tabular." + path.replace("/", "."),
            schema_version=1,
            dtype=dtype,
            shape_template=shape,
            axis_names=axes,
            description=f"Exact test array {path}.",
        ),
        required=True,
        access_pattern=access,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="every immutable value is written before validation",
        null_semantics="floating NaN represents unavailable values",
        physical_policy_owner="analysis_storage_planning_v1",
        byte_planner_adopted=True,
    )


def _source() -> zarr.Group:
    source = zarr.group(zarr_format=3)
    source.attrs["stage_selector_eligible"] = True
    signals = source.create_group("signals")
    signals.attrs["semantic_axis"] = "detector_signal_by_frame"
    signals.create_array(
        "values",
        data=np.arange(600_000, dtype=np.float32).reshape(2, 300_000),
    )
    table = source.create_group("table")
    table.create_array("frame_index", data=np.arange(300_000, dtype=np.int64))
    return source


def _declarations() -> tuple[AnalysisArrayDeclaration, ...]:
    return (
        _declaration(
            "signals/values",
            dtype=FLOAT32,
            shape=(2, "n_frame"),
            axes=("detector_signal", "frame"),
            access=AccessPattern.WINDOWED,
        ),
        _declaration(
            "table/frame_index",
            dtype=INT64,
            shape=("n_rows",),
            axes=("row",),
            access=AccessPattern.INDEXED,
        ),
    )


def test_exact_tabular_candidate_plans_time_matrix_on_frame_axis() -> None:
    source = _source()
    declarations = _declarations()
    receipt = build_exact_tabular_storage_receipt(
        source,
        declarations=declarations,
        profile=PUBLISHED_HTTP_V1,
    )
    entries = {entry.declaration.path: entry for entry in receipt.entries}

    signal = entries["signals/values"]
    assert signal.facts.growth_axis == 1
    assert signal.facts.access_unit_shape == (2, 1)
    assert signal.plan.chunk_shape == (2, 131_072)
    row = entries["table/frame_index"]
    assert row.facts.growth_axis == 0
    assert row.plan.chunk_shape == (131_072,)


def test_exact_tabular_candidate_rematerializes_and_validates() -> None:
    source = _source()
    declarations = _declarations()
    receipt = build_exact_tabular_storage_receipt(
        source,
        declarations=declarations,
        profile=PUBLISHED_HTTP_V1,
    )
    destination = zarr.group(zarr_format=3)

    rematerialize_exact_tabular_candidate(
        source,
        destination,
        receipt=receipt,
    )
    persist_exact_tabular_storage_receipt(destination, receipt)

    np.testing.assert_array_equal(
        destination["signals/values"][:],
        source["signals/values"][:],
    )
    np.testing.assert_array_equal(
        destination["table/frame_index"][:],
        source["table/frame_index"][:],
    )
    assert destination.attrs["stage_selector_eligible"] is False
    assert destination["signals"].attrs["semantic_axis"] == (
        "detector_signal_by_frame"
    )
    assert validate_exact_tabular_storage_receipt(
        destination,
        declarations=declarations,
    ) == ()


def test_exact_tabular_candidate_rejects_undeclared_scientific_array() -> None:
    source = _source()
    source.create_array("invented", data=np.zeros(3, dtype=np.int32))
    declarations = _declarations()
    receipt = build_exact_tabular_storage_receipt(
        source,
        declarations=declarations,
        profile=PUBLISHED_HTTP_V1,
    )

    with pytest.raises(ValueError, match="undeclared scientific arrays"):
        rematerialize_exact_tabular_candidate(
            source,
            zarr.group(zarr_format=3),
            receipt=receipt,
        )


def test_exact_tabular_candidate_preserves_explicit_report_artifacts() -> None:
    source = _source()
    visualizations = source.create_group("visualizations")
    visualizations.create_array(
        "summary_png",
        data=np.asarray([137, 80, 78, 71], dtype=np.uint8),
    )
    declarations = _declarations()
    receipt = build_exact_tabular_storage_receipt(
        source,
        declarations=declarations,
        profile=PUBLISHED_HTTP_V1,
    )
    destination = zarr.group(zarr_format=3)

    rematerialize_exact_tabular_candidate(
        source,
        destination,
        receipt=receipt,
    )

    np.testing.assert_array_equal(
        destination["visualizations/summary_png"][:],
        source["visualizations/summary_png"][:],
    )
