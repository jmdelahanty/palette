from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import fisheye.analytics_exports.validated_behavior_adapters as adapters
import fisheye.analytics_exports.validated_behavior_phase_c_contracts as contracts
import fisheye.analytics_exports.validated_behavior_profiles as profiles
import fisheye.visualization.chaser_appearance as appearance_module
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_NAMES,
    validate_table_specs,
)
from fisheye.analytics_exports.validated_behavior_phase_a_contracts import (
    CHASER_OCCURRENCES as PHASE_A_CHASER_OCCURRENCES,
)
from fisheye.analytics_exports.validated_behavior_phase_b_contracts import (
    PHASE_B_TABLE_SPECS,
)
from fisheye.group_statistics.validated_behavior_appearance import (
    NEUTRAL_MIXED_COLOR,
    ValidatedBehaviorAppearanceError,
    behavior_role_styles,
    build_chaser_appearance_dimension,
    validate_chaser_appearance_dimension,
)
from fisheye.visualization.chaser_appearance import (
    APPEARANCE_POLICY_ID,
    APPEARANCE_SCHEMA_ID,
    APPEARANCE_SCHEMA_VERSION,
    ChaserAppearance,
    ChaserAppearanceProjection,
)

_ADDED_APPEARANCE_FIELDS = (
    "behavior_role_code",
    "experimental_color_r",
    "experimental_color_g",
    "experimental_color_b",
    "experimental_color_a",
    "experimental_color_hex",
    "experimental_color_css",
    "contrast_outline_hex",
    "plotly_role_symbol",
    "matplotlib_role_marker",
    "appearance_schema_id",
    "appearance_schema_version",
    "appearance_policy_id",
    "appearance_projection_sha256",
    "occurrence_binding_sha256",
    "color_semantics",
    "role_semantics",
    "color_role_independence",
)


def _occurrence() -> dict[str, Any]:
    return {
        "schema_id": "palette.chaser_relative_frame.chaser_occurrence_binding",
        "schema_version": 1,
        "recording_id": "recording-1",
        "occurrence_policy_id": "native_sample_declared_chaser_axis_v1",
        "chaser_identity_policy_id": "stimulus_run_scoped_chaser_index_v1",
        "source_stimulus_run_path": "analysis/stimulus_runs/stimulus-exact-v1",
        "source_protocol_sha256": "c" * 64,
        "chasers": [
            {
                "chaser_index": 0,
                "identity": "stimulus-exact-v1:chaser_index:0",
                "behavior_role": "aggressive",
            },
            {
                "chaser_index": 1,
                "identity": "stimulus-exact-v1:chaser_index:1",
                "behavior_role": "inert",
            },
        ],
        "semantics": "exact native chaser axis",
    }


def _projection() -> ChaserAppearanceProjection:
    appearances = (
        ChaserAppearance(
            identity_code=1,
            chaser_index=0,
            identity="stimulus-exact-v1:chaser_index:0",
            behavior_role_code=1,
            behavior_role="aggressive",
            experimental_color_rgba=(0.0, 0.0, 1.0, 1.0),
            experimental_color_hex="#0000ff",
            experimental_color_css="rgba(0, 0, 255, 1)",
            plotly_role_symbol="star",
            matplotlib_role_marker="*",
            contrast_outline_hex="#ffffff",
        ),
        ChaserAppearance(
            identity_code=2,
            chaser_index=1,
            identity="stimulus-exact-v1:chaser_index:1",
            behavior_role_code=2,
            behavior_role="inert",
            experimental_color_rgba=(1.0, 0.0, 0.0, 0.5),
            experimental_color_hex="#ff0000",
            experimental_color_css="rgba(255, 0, 0, 0.5)",
            plotly_role_symbol="circle",
            matplotlib_role_marker="o",
            contrast_outline_hex="#ffffff",
        ),
    )
    return ChaserAppearanceProjection(
        recording_id="recording-1",
        source_stimulus_run_path="analysis/stimulus_runs/stimulus-exact-v1",
        source_protocol_sha256="c" * 64,
        occurrence_binding_sha256="b" * 64,
        appearances=appearances,
        projection_sha256="a" * 64,
    )


class _OccurrenceContext:
    def __init__(self, projection: ChaserAppearanceProjection | None = None) -> None:
        self.bundle_common = {
            "export_run_id": "export-1",
            "recording_id": "recording-1",
            "membership_member_sha256": "1" * 64,
            "bundle_set_member_sha256": "2" * 64,
            "bundle_record_sha256": "3" * 64,
        }
        self.bundle = {
            "source_bindings": {
                "row_axis_timing_and_scale": {
                    "authority": {"chaser_occurrence": _occurrence()}
                }
            }
        }
        self._projection = projection or _projection()

    def chaser_identity_maps(self) -> tuple[dict[int, str], dict[int, str]]:
        return (
            {
                1: "stimulus-exact-v1:chaser_index:0",
                2: "stimulus-exact-v1:chaser_index:1",
            },
            {1: "aggressive", 2: "inert"},
        )

    def chaser_appearance_projection(self) -> ChaserAppearanceProjection:
        return self._projection


class _AppearanceTable:
    spec = contracts.PHASE_C_TABLE_SPECS["chaser_occurrences"]

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.frame = pl.DataFrame(rows)

    def scan(self, *, columns=None, predicate=None):
        lazy = self.frame.lazy()
        if predicate is not None:
            lazy = lazy.filter(predicate)
        return lazy if columns is None else lazy.select(*columns)

    def query_identity(self, *, columns=None, predicate_description=None):
        return {
            "export_run_id": "export-1",
            "export_manifest_record_sha256": "4" * 64,
            "export_plan_sha256": "5" * 64,
            "table_name": "chaser_occurrences",
            "table_contract_sha256": self.spec.contract.payload_sha256,
            "grain": self.spec.grain,
            "selected_columns": list(columns or ()),
            "predicate_description": predicate_description,
            "analysis_unit_policy_sha256": "6" * 64,
            "capability_policy": self.spec.capability_policy,
            "semantic_metadata": dict(self.spec.semantic_metadata),
        }


class _AppearanceDataset:
    table_names = ("chaser_occurrences",)
    cache_identity = "4" * 64

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._table = _AppearanceTable(rows)

    def table(self, name: str) -> _AppearanceTable:
        assert name == "chaser_occurrences"
        return self._table


class _LegacyAppearanceDataset:
    table_names = ("chaser_occurrences",)

    class _Table:
        spec = PHASE_B_TABLE_SPECS["chaser_occurrences"]

    def table(self, name: str) -> _Table:
        assert name == "chaser_occurrences"
        return self._Table()


def test_phase_c_replaces_only_chaser_occurrence_schema() -> None:
    profile = profiles.resolve_validated_behavior_profile(contracts.PHASE_C_PROFILE_ID)

    assert profile.profile_id == contracts.PHASE_C_PROFILE_ID
    assert tuple(profile.table_specs) == tuple(PHASE_B_TABLE_SPECS)
    assert len(validate_table_specs(profile.table_specs)) == 30
    assert set(profile.row_extractors()) == set(profile.table_specs) - set(
        CORE_TABLE_NAMES
    )
    for name, phase_b_spec in PHASE_B_TABLE_SPECS.items():
        if name != "chaser_occurrences":
            assert profile.table_specs[name] is phase_b_spec

    old_contract = PHASE_B_TABLE_SPECS["chaser_occurrences"].contract
    new_contract = profile.table_specs["chaser_occurrences"].contract
    assert old_contract is PHASE_A_CHASER_OCCURRENCES
    assert old_contract.schema_version == 1
    assert new_contract.schema_version == 2
    assert new_contract.fields[: len(old_contract.fields)] == old_contract.fields
    assert tuple(
        field.name for field in new_contract.fields[len(old_contract.fields) :]
    ) == (_ADDED_APPEARANCE_FIELDS)


def test_phase_c_occurrence_semantics_prohibit_color_role_inference() -> None:
    metadata = dict(
        contracts.PHASE_C_TABLE_SPECS["chaser_occurrences"].semantic_metadata
    )

    assert metadata == {
        "appearance_source": (
            "exact_protocol_rgba_bound_through_relative_frame_receipt"
        ),
        "color_semantics": "experimental_protocol_rgba",
        "role_semantics": "independent_marker_shape_and_text",
        "color_role_independence": "true",
        "appearance_fallback": "prohibited",
        "appearance_join": "recording_id_and_chaser_identity_code",
    }


def test_phase_c_occurrences_export_color_and_role_as_independent_channels() -> None:
    rows, zero_reason = adapters._chaser_occurrences_v2(  # noqa: SLF001
        _OccurrenceContext()
    )

    assert zero_reason is None
    assert len(rows) == 2
    assert set(rows[0]) == {
        field.name for field in contracts.CHASER_OCCURRENCES_V2.fields
    }
    aggressive, inert = rows
    assert aggressive["behavior_role"] == "aggressive"
    assert aggressive["experimental_color_hex"] == "#0000ff"
    assert aggressive["plotly_role_symbol"] == "star"
    assert inert["behavior_role"] == "inert"
    assert inert["experimental_color_hex"] == "#ff0000"
    assert inert["plotly_role_symbol"] == "circle"
    assert inert["experimental_color_a"] == 0.5
    assert all(row["color_role_independence"] is True for row in rows)
    assert all(row["appearance_policy_id"] == APPEARANCE_POLICY_ID for row in rows)
    assert all(row["appearance_schema_id"] == APPEARANCE_SCHEMA_ID for row in rows)
    assert all(
        row["appearance_schema_version"] == APPEARANCE_SCHEMA_VERSION for row in rows
    )
    assert all(row["appearance_projection_sha256"] == "a" * 64 for row in rows)


def test_phase_c_occurrence_rejects_projection_identity_drift() -> None:
    projection = _projection()
    drifted = replace(
        projection.appearances[1],
        identity="stimulus-exact-v1:chaser_index:99",
    )
    projection = replace(projection, appearances=(projection.appearances[0], drifted))

    with pytest.raises(
        adapters.ValidatedBehaviorAdapterError,
        match="differs from its exact chaser occurrence row",
    ):
        adapters._chaser_occurrences_v2(_OccurrenceContext(projection))  # noqa: SLF001


def test_recording_context_resolves_appearance_from_exact_receipt_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence = _occurrence()
    relative_manifest = {
        "identity_registries": {
            "chaser": {
                "1": "stimulus-exact-v1:chaser_index:0",
                "2": "stimulus-exact-v1:chaser_index:1",
            },
            "behavior_role": {"1": "aggressive", "2": "inert"},
        },
        "context": {
            "chaser_occurrence": {
                "record": occurrence,
                "sha256": "b" * 64,
            }
        },
    }
    context = object.__new__(adapters._RecordingContext)  # noqa: SLF001
    context.analysis_zarr = Path("/tmp/fixture-analysis.zarr")
    context.recording_id = "recording-1"
    context.bundle = {
        "source_bindings": {
            "row_axis_timing_and_scale": {
                "authority": {"chaser_occurrence": occurrence}
            }
        }
    }
    context._chaser_appearance = None  # noqa: SLF001
    context.chaser_identity_receipt = lambda: {"run_manifest": relative_manifest}
    context.chaser_identity_maps = lambda: (
        {
            1: "stimulus-exact-v1:chaser_index:0",
            2: "stimulus-exact-v1:chaser_index:1",
        },
        {1: "aggressive", 2: "inert"},
    )

    observed: dict[str, Any] = {}

    def fake_load(
        analysis_zarr: Path,
        *,
        relative_manifest: dict[str, Any],
        identity_code_by_column: np.ndarray,
        behavior_role_code_by_column: np.ndarray,
        expected_recording_id: str,
    ) -> ChaserAppearanceProjection:
        observed.update(
            {
                "analysis_zarr": analysis_zarr,
                "relative_manifest": relative_manifest,
                "identity_codes": identity_code_by_column.tolist(),
                "role_codes": behavior_role_code_by_column.tolist(),
                "recording_id": expected_recording_id,
            }
        )
        return _projection()

    monkeypatch.setattr(
        appearance_module, "load_chaser_appearance_projection", fake_load
    )

    assert (
        context.chaser_appearance_projection() is context.chaser_appearance_projection()
    )
    assert observed == {
        "analysis_zarr": Path("/tmp/fixture-analysis.zarr"),
        "relative_manifest": relative_manifest,
        "identity_codes": [1, 2],
        "role_codes": [1, 2],
        "recording_id": "recording-1",
    }


def test_recording_context_rejects_bundle_receipt_occurrence_drift() -> None:
    occurrence = _occurrence()
    receipt_occurrence = {**occurrence, "source_protocol_sha256": "d" * 64}
    context = object.__new__(adapters._RecordingContext)  # noqa: SLF001
    context._chaser_appearance = None  # noqa: SLF001
    context.bundle = {
        "source_bindings": {
            "row_axis_timing_and_scale": {
                "authority": {"chaser_occurrence": occurrence}
            }
        }
    }
    context.chaser_identity_receipt = lambda: {
        "run_manifest": {
            "context": {
                "chaser_occurrence": {
                    "record": receipt_occurrence,
                    "sha256": "b" * 64,
                }
            }
        }
    }

    with pytest.raises(
        adapters.ValidatedBehaviorAdapterError,
        match="differs from the relative-frame receipt",
    ):
        context.chaser_appearance_projection()


def test_statistics_appearance_dimension_preserves_exact_colors_and_glyphs() -> None:
    rows, _zero_reason = adapters._chaser_occurrences_v2(  # noqa: SLF001
        _OccurrenceContext()
    )
    dimension = build_chaser_appearance_dimension(_AppearanceDataset(rows))

    assert dimension is not None
    assert validate_chaser_appearance_dimension(dimension) == dimension
    assert dimension["join_fields"] == ["recording_id", "chaser_identity_code"]
    styles = behavior_role_styles(dimension, legacy_display_colors={})
    assert styles["aggressive"] == {
        "aggregate_color_hex": "#0000ff",
        "aggregate_color_css": "rgba(0, 0, 255, 1)",
        "aggregate_color_policy": "unique_protocol_rgba_across_occurrences",
        "experimental_color_hex_values": ["#0000ff"],
        "experimental_color_css_values": ["rgba(0, 0, 255, 1)"],
        "plotly_role_symbol": "star",
        "matplotlib_role_marker": "*",
        "color_role_independence": True,
    }
    assert styles["inert"]["aggregate_color_hex"] == "#ff0000"


def test_phase_b_statistics_remain_valid_without_appearance_extension() -> None:
    assert build_chaser_appearance_dimension(_LegacyAppearanceDataset()) is None


def test_mixed_protocol_colors_use_neutral_aggregate_and_keep_role_glyph() -> None:
    rows, _zero_reason = adapters._chaser_occurrences_v2(  # noqa: SLF001
        _OccurrenceContext()
    )
    second_recording = []
    for row in rows:
        copied = {**row, "recording_id": "recording-2"}
        if copied["behavior_role"] == "aggressive":
            copied.update(
                {
                    "experimental_color_r": 0.0,
                    "experimental_color_g": 1.0,
                    "experimental_color_b": 0.0,
                    "experimental_color_hex": "#00ff00",
                    "experimental_color_css": "rgba(0, 255, 0, 1)",
                }
            )
        copied["appearance_projection_sha256"] = "d" * 64
        second_recording.append(copied)
    dimension = build_chaser_appearance_dimension(
        _AppearanceDataset([*rows, *second_recording])
    )
    assert dimension is not None

    styles = behavior_role_styles(dimension, legacy_display_colors={})
    aggressive = styles["aggressive"]
    assert aggressive["aggregate_color_hex"] == NEUTRAL_MIXED_COLOR
    assert aggressive["aggregate_color_policy"] == (
        "neutral_due_to_multiple_protocol_colors"
    )
    assert aggressive["experimental_color_hex_values"] == ["#0000ff", "#00ff00"]
    assert aggressive["experimental_color_css_values"] == [
        "rgba(0, 0, 255, 1)",
        "rgba(0, 255, 0, 1)",
    ]
    assert aggressive["plotly_role_symbol"] == "star"
    assert styles["inert"]["aggregate_color_hex"] == "#ff0000"


def test_statistics_appearance_dimension_rejects_digest_mutation() -> None:
    rows, _zero_reason = adapters._chaser_occurrences_v2(  # noqa: SLF001
        _OccurrenceContext()
    )
    dimension = build_chaser_appearance_dimension(_AppearanceDataset(rows))
    assert dimension is not None
    stale = {**dimension, "color_semantics": "invented_color_role_mapping"}

    with pytest.raises(ValidatedBehaviorAppearanceError, match="semantics"):
        validate_chaser_appearance_dimension(stale)

    with pytest.raises(ValidatedBehaviorAppearanceError, match="another source"):
        validate_chaser_appearance_dimension(
            dimension,
            expected_export_manifest_sha256="f" * 64,
        )
