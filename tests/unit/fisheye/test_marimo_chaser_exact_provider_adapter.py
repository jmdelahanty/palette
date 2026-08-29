from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.marimo.components import chaser_exact_successors as facade
from apps.marimo.components.analysis_catalog import (
    CHASER_CANDIDATE_PROVIDER,
    CHASER_EXACT_SUCCESSOR_PROVIDER,
)
from apps.marimo.components.chaser_exact.distance_traces import (
    _trace_display_projection,
)
from apps.marimo.components.chaser_exact.controller_trials import (
    build_exact_controller_trials_output,
)
from apps.marimo.components.chaser_exact.bout_response import (
    build_exact_bout_response_output,
)
from apps.marimo.components.chaser_exact.provider import (
    ANALYSIS_IDS,
    EXACT_CHASER_PROVIDER_ADAPTER,
    ExactChaserAnalysisUnavailableError,
    ExactChaserStaleSelectionError,
    ExactChaserUnknownAnalysisError,
    load_exact_chaser_successor_projection,
)
from apps.marimo.components.chaser_exact.trajectory_overlays import (
    _trajectory_display_indices,
)
from apps.marimo.components.registry import (
    CHASER_EXACT_SUCCESSOR_RENDERER,
    InteractiveSpecOption,
)


def _option(
    archive: Path,
    *,
    manifest_sha256: str = "a" * 64,
    run_name: str = "paired-spatial-v1",
    trace_max_points: int = 6_000,
    controller_manifest_sha256: str = "c" * 64,
    bout_manifest_sha256: str = "1" * 64,
) -> InteractiveSpecOption:
    run_path = f"analysis/chaser_spatial_occupancy_runs/{run_name}"
    spec = {
        "schema_id": "palette.chaser_exact_successor_explorer_spec",
        "schema_version": 5,
        "renderer": CHASER_EXACT_SUCCESSOR_RENDERER,
        "bundle_status": "exact_selector_ineligible",
        "bundle_manifest_sha256": manifest_sha256,
        "analysis_bindings": {
            "controller_trials": {
                "run_path": "analysis/controller_chase_trial_runs/controller-v1",
                "manifest_sha256": controller_manifest_sha256,
                "scientific_payload_sha256": "d" * 64,
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "semantic_selection": {
                    "run_path": (
                        "analysis/protocol_semantic_chaser_selection_runs/semantic-v1"
                    ),
                    "manifest_sha256": "f" * 64,
                },
            },
            "generalized_bout_response": {
                "run_path": ("analysis/generalized_chaser_bout_response_runs/bout-v1"),
                "manifest_sha256": bout_manifest_sha256,
                "scientific_payload_sha256": "2" * 64,
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "source_motion": {
                    "run_path": ("analysis/track_kinematics_runs/provider/motion-v1"),
                    "manifest_sha256": "3" * 64,
                    "relative_frame_projection": {
                        "schema_id": (
                            "palette.provider_motion.relative_frame_projection"
                        ),
                        "schema_version": 1,
                        "join_key": "exact_acquisition_frame_id",
                        "join_policy": (
                            "left_join_missing_provider_rows_invalid_no_interpolation"
                        ),
                        "provider_frame_count": 10,
                        "relative_frame_count": 10,
                        "matched_relative_frame_count": 10,
                        "missing_relative_frame_count": 0,
                        "provider_only_frame_count": 0,
                        "provider_frame_ids_sha256": "4" * 64,
                        "relative_frame_ids_sha256": "5" * 64,
                        "provider_row_index_by_relative_frame_sha256": "6" * 64,
                        "provider_frame_present_sha256": "7" * 64,
                        "fallback": "prohibited",
                    },
                },
                "source_swim_bouts": {
                    "run_path": "analysis/swim_bout_runs/bouts-v1",
                    "lineage_sha256": "8" * 64,
                    "signal_id": 4,
                    "signal_level": "speed_exponential",
                },
                "semantic_selection_manifest_sha256": "f" * 64,
                "controller_trial_payload_sha256": "d" * 64,
                "body_extension_present": True,
            },
        },
        "display_parameters": {
            "distance_traces": {
                "algorithm": (
                    "source_order_bucket_first_last_min_max_missing_break_v1"
                ),
                "max_points_per_series": trace_max_points,
            },
            "scientific_recomputation": False,
            "interpolation": "prohibited",
        },
    }
    return InteractiveSpecOption(
        zarr_path=archive,
        artifact_path=f"{run_path}/interactive",
        run_path=run_path,
        artifact_name="chaser_exact_successor_bundle",
        renderer=CHASER_EXACT_SUCCESSOR_RENDERER,
        schema_id=str(spec["schema_id"]),
        title="Exact chaser successors",
        run_name=run_name,
        label=run_name,
        is_supported=True,
        attrs={},
        spec=spec,
    )


def test_analysis_listing_is_metadata_only(tmp_path: Path, monkeypatch) -> None:
    option = _option(tmp_path / "recording.zarr")

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("analysis discovery must not open scientific arrays")

    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.provider.load_exact_chaser_projection",
        forbidden_loader,
    )
    assert (
        EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(option.zarr_path, option)
        == ANALYSIS_IDS
    )


def test_controller_trial_analysis_is_hidden_without_one_exact_binding(
    tmp_path: Path,
) -> None:
    option = _option(tmp_path / "recording.zarr")
    spec = dict(option.spec)
    spec["analysis_bindings"] = {}

    available = EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(
        option.zarr_path,
        replace(option, spec=spec),
    )

    assert "controller_trials" not in available
    assert "spatial_occupancy" in available


def test_provider_routes_are_closed_and_controls_are_explicit() -> None:
    assert ANALYSIS_IDS == (
        "radial_near_field",
        "distance_traces",
        "trajectory_overlays",
        "spatial_occupancy",
        "controller_trials",
        "generalized_bout_response",
        "provenance",
    )
    assert EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("distance_traces")
    assert not EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("provenance")
    assert EXACT_CHASER_PROVIDER_ADAPTER.build_controls("radial_near_field") is None
    with pytest.raises(ExactChaserUnknownAnalysisError, match="Unsupported"):
        EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("distance-ish")


def test_controller_trial_catalog_entry_belongs_to_exact_successors() -> None:
    exact_ids = tuple(
        item.analysis_id for item in CHASER_EXACT_SUCCESSOR_PROVIDER.analyses
    )
    candidate_ids = tuple(
        item.analysis_id for item in CHASER_CANDIDATE_PROVIDER.analyses
    )

    assert exact_ids == ANALYSIS_IDS
    assert "controller_trials" not in candidate_ids


def test_exact_source_defaults_only_when_discovery_is_unambiguous() -> None:
    assert EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(()) is None
    assert EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(("only",)) == "only"
    assert (
        EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(("newer", "older")) is None
    )


def test_only_selected_analysis_requests_relative_arrays(
    tmp_path: Path, monkeypatch
) -> None:
    option = _option(tmp_path / "recording.zarr")
    observed: list[tuple[str, bool, bool, bool]] = []

    def fake_loader(
        zarr_path,
        selected_option,
        *,
        selection_identity,
        load_relative,
        load_controller_trials,
        load_generalized_bout_response,
    ):
        assert zarr_path == option.zarr_path
        assert selected_option is option
        observed.append(
            (
                selection_identity.analysis_id,
                load_relative,
                load_controller_trials,
                load_generalized_bout_response,
            )
        )
        return selection_identity

    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.provider.load_exact_chaser_projection",
        fake_loader,
    )

    radial = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="radial_near_field"
    )
    spatial = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="spatial_occupancy"
    )
    distance = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="distance_traces"
    )
    controller = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="controller_trials"
    )
    bout_response = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="generalized_bout_response"
    )

    assert radial.analysis_id == "radial_near_field"
    assert spatial.analysis_id == "spatial_occupancy"
    assert distance.analysis_id == "distance_traces"
    assert controller.analysis_id == "controller_trials"
    assert bout_response.analysis_id == "generalized_bout_response"
    assert observed == [
        ("radial_near_field", False, False, False),
        ("spatial_occupancy", False, False, False),
        ("distance_traces", True, False, False),
        ("controller_trials", True, True, False),
        ("generalized_bout_response", True, True, True),
    ]


def test_selection_identity_binds_display_parameters_and_exact_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    option = _option(archive)
    identity = EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
        archive, option, analysis_id="distance_traces"
    )

    assert identity.archive_path == str(archive.resolve())
    assert identity.run_path == option.run_path
    assert identity.bundle_manifest_sha256 == "a" * 64
    assert identity.renderer == CHASER_EXACT_SUCCESSOR_RENDERER

    changed_display = _option(archive, trace_max_points=3_000)
    changed_manifest = _option(archive, manifest_sha256="b" * 64)
    changed_run = _option(archive, run_name="paired-spatial-v2")
    changed_controller = _option(archive, controller_manifest_sha256="9" * 64)
    changed_bout = _option(archive, bout_manifest_sha256="0" * 64)
    for changed in (
        changed_display,
        changed_manifest,
        changed_run,
        changed_controller,
        changed_bout,
    ):
        assert (
            EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
                archive, changed, analysis_id="distance_traces"
            )
            != identity
        )


def test_stale_projection_cannot_render_under_new_selection(tmp_path: Path) -> None:
    archive = tmp_path / "recording.zarr"
    option = _option(archive)
    identity = EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
        archive, option, analysis_id="distance_traces"
    )
    projection = SimpleNamespace(
        analysis_id="distance_traces", selection_identity=identity
    )

    EXACT_CHASER_PROVIDER_ADAPTER.require_current_projection(
        projection,
        zarr_path=archive,
        option=option,
        analysis_id="distance_traces",
    )
    with pytest.raises(ExactChaserStaleSelectionError, match="earlier archive"):
        EXACT_CHASER_PROVIDER_ADAPTER.require_current_projection(
            projection,
            zarr_path=archive,
            option=_option(archive, trace_max_points=3_000),
            analysis_id="distance_traces",
        )


def test_shared_provenance_route_is_typed_unavailable(tmp_path: Path) -> None:
    archive = tmp_path / "recording.zarr"
    option = _option(archive)
    identity = EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
        archive, option, analysis_id="provenance"
    )
    projection = SimpleNamespace(analysis_id="provenance", selection_identity=identity)

    with pytest.raises(ExactChaserAnalysisUnavailableError, match="shared"):
        EXACT_CHASER_PROVIDER_ADAPTER.render(
            None,
            None,
            projection,
            zarr_path=archive,
            option=option,
            analysis_id="provenance",
        )


def test_compatibility_facade_reexports_focused_components() -> None:
    assert (
        facade.load_exact_chaser_successor_projection
        is load_exact_chaser_successor_projection
    )
    assert facade._trace_display_projection is _trace_display_projection
    assert facade._trajectory_display_indices is _trajectory_display_indices
    assert (
        facade.build_exact_controller_trials_output
        is build_exact_controller_trials_output
    )
    assert facade.build_exact_bout_response_output is build_exact_bout_response_output
    assert facade.EXACT_CHASER_PROVIDER_ADAPTER is EXACT_CHASER_PROVIDER_ADAPTER


def test_palette_explorer_uses_one_exact_provider_load_and_render_boundary() -> None:
    source = Path("apps/marimo/palette_explorer.py").read_text(encoding="utf-8")

    assert "build_exact_distance_traces_output" not in source
    assert "build_exact_radial_near_field_output" not in source
    assert "build_exact_trajectory_overlays_output" not in source
    assert source.count("EXACT_CHASER_PROVIDER_ADAPTER.load_projection(") == 1
    assert source.count("EXACT_CHASER_PROVIDER_ADAPTER.render(") == 1
    assert "EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(source_labels)" in source
    assert "source_picker.value is not None" in source
    assert "no analysis arrays will load until" in source
