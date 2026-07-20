from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apps.marimo.components import core_behavior as marimo_core_module
from apps.marimo.components import goodcopbadcop_chaser as marimo_chaser_module
from apps.marimo.components import registry as marimo_registry_module
from fisheye.analysis import analyze_goodcopbadcop_escape as escape_module
from fisheye.analysis import analyze_goodcopbadcop_habituation as habituation_module
from fisheye.analysis import analyze_goodcopbadcop_bout_vigor_prepost as bout_vigor_module
from fisheye.analysis import analyze_goodcopbadcop_bout_kinematics_distance as bout_distance_module
from fisheye.analysis import analyze_goodcopbadcop_immobility_artifact as immobility_module
from fisheye.analysis import analyze_goodcopbadcop_lateral_gaze as lateral_gaze_module
from fisheye.analysis import analyze_goodcopbadcop_per_fish as per_fish_module
from fisheye.analysis import analyze_goodcopbadcop_radial_kinematics as radial_kinematics_module
from fisheye.analysis import analyze_goodcopbadcop_radial_turn_direction as radial_turn_module
from fisheye.analysis import analyze_goodcopbadcop_wall_mediator as wall_mediator_module
from fisheye.analysis import chaser_bout_response as bout_response_module
from fisheye.analysis import chaser_egocentric_bearing as egocentric_module
from fisheye.analysis import chaser_epoch_behavior_summary as epoch_behavior_module
from fisheye.analysis import chaser_escape_events as escape_events_module
from fisheye.analysis import chaser_escape_freeze_summary as escape_freeze_module
from fisheye.analysis import chaser_gaze_tracking as gaze_tracking_module
from fisheye.analysis import chaser_near_field_occupancy as near_field_module
from fisheye.analysis import chaser_quadrant_occupancy as quadrant_module
from fisheye.analysis import chaser_radial_occupancy as radial_module
from fisheye.analysis import chaser_response_regimes as response_regimes_module
from fisheye.analysis import plot_goodcopbadcop_bout_rate as bout_rate_module
from fisheye.analysis import plot_goodcopbadcop_occupancy_heatmaps as occupancy_module
from fisheye.analysis import plot_goodcopbadcop_trajectory_prepost as trajectory_module
from fisheye.analysis.chaser_distance_io import ChaserDistanceReadError
from fisheye.montage import render as montage_render_module
from fisheye.utils import export_goodcopbadcop_static_montage as montage_module
from fisheye.utils import export_cross_recording_analytics as export_module
from fisheye.utils import view_detection_chaser_overview as overview_module
from fisheye.visualization import chaser_analysis_figures as analysis_figures_module
from fisheye.visualization import chaser_habituation_figures as habituation_figures_module
from fisheye.visualization import chaser_ring_traversal as ring_traversal_module
from fisheye.visualization import chaser_visit_trajectories as visit_trajectories_module
from fisheye.visualization import goodcopbadcop_interactive as dashboard_module


class _RawNavigationTrap:
    """Root sentinel that rejects every attempt to navigate raw archive nodes."""

    def __getitem__(self, _key):
        raise AssertionError("raw Zarr indexing is forbidden")

    def get(self, _key, _default=None):
        raise AssertionError("raw Zarr discovery is forbidden")


def _snapshot(run_name: str = "canonical") -> SimpleNamespace:
    return SimpleNamespace(
        run_name=run_name,
        run_path=f"analysis/chaser_distance_runs/{run_name}",
    )


def _rejecting_snapshot(
    authority_calls: list[tuple[str, str | None]],
    *,
    run_name: str = "canonical",
) -> SimpleNamespace:
    def reject_derived(relative_path: str) -> None:
        authority_calls.append(("derived", relative_path))
        raise ChaserDistanceReadError("derived surface is intentionally unsealed")

    def reject_behavior() -> None:
        authority_calls.append(("behavior", None))
        raise ChaserDistanceReadError("behavior authority is intentionally unsealed")

    def reject_arena_geometry() -> None:
        authority_calls.append(("arena_geometry", None))
        raise ChaserDistanceReadError("arena geometry authority is intentionally unsealed")

    def reject_protocol(semantic_label: str) -> None:
        authority_calls.append(("stimulus_protocol", semantic_label))
        raise ChaserDistanceReadError("stimulus protocol authority is intentionally unsealed")

    return SimpleNamespace(
        run_name=run_name,
        run_path=f"analysis/chaser_distance_runs/{run_name}",
        require_derived_surface_authority=reject_derived,
        require_behavior_authority=reject_behavior,
        require_arena_geometry_authority=reject_arena_geometry,
        require_stimulus_protocol_authority=reject_protocol,
    )


@pytest.mark.parametrize(
    ("module", "loader_name", "authority_kind", "relative_path"),
    [
        (radial_turn_module, "load_rows", "derived", "chaser_bout_response"),
        (radial_kinematics_module, "load", "derived", "chaser_bout_response"),
        (lateral_gaze_module, "load", "derived", "egocentric_bearing"),
        (bout_rate_module, "load", "derived", "chaser_bout_response"),
        (escape_module, "load", "derived", "chaser_bout_response"),
        (per_fish_module, "learning_index", "derived", "chaser_escape_events"),
        (per_fish_module, "spatial_avoidance_did", "behavior", None),
        (occupancy_module, "load_positions", "behavior", None),
        (habituation_module, "load", "derived", "chaser_escape_events"),
        (bout_vigor_module, "load", "derived", "chaser_bout_response"),
        (immobility_module, "load", "derived", "chaser_bout_response"),
        (bout_distance_module, "load", "derived", "chaser_bout_response"),
        (wall_mediator_module, "load_angular", "arena_geometry", None),
        (wall_mediator_module, "load_partial", "derived", "chaser_escape_freeze"),
    ],
)
def test_analysis_loaders_reject_unsealed_authority_before_raw_navigation(
    monkeypatch: pytest.MonkeyPatch,
    module,
    loader_name: str,
    authority_kind: str,
    relative_path: str | None,
) -> None:
    root = _RawNavigationTrap()
    strict_calls: list[object] = []
    authority_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(module.zarr, "open_group", lambda *_a, **_k: root)

    def strict_load(root_node, **_kwargs):
        strict_calls.append(root_node)
        return _rejecting_snapshot(authority_calls)

    monkeypatch.setattr(module, "load_chaser_distance_run", strict_load)

    with pytest.raises(ChaserDistanceReadError, match="intentionally unsealed"):
        getattr(module, loader_name)("unused.zarr")

    assert strict_calls == [root]
    assert authority_calls == [(authority_kind, relative_path)]


@pytest.mark.parametrize(
    ("module", "builder_name", "authority_kind", "authority_value", "opener_name"),
    [
        (
            bout_response_module,
            "build_chaser_bout_response_result",
            "derived",
            "egocentric_bearing",
            "_open_root",
        ),
        (
            escape_events_module,
            "build_chaser_escape_events_result",
            "derived",
            "chaser_bout_response",
            "_open_root",
        ),
        (
            near_field_module,
            "build_chaser_near_field_occupancy_result",
            "derived",
            "chaser_quadrant_occupancy",
            "_open_root",
        ),
        (
            radial_module,
            "build_chaser_radial_occupancy_result",
            "arena_geometry",
            None,
            "_open_root",
        ),
        (
            response_regimes_module,
            "build_chaser_response_regimes_result",
            "stimulus_protocol",
            "chaser radii and position-transition timing",
            "_open_root",
        ),
        (
            escape_freeze_module,
            "build_chaser_escape_freeze_summary_result",
            "stimulus_protocol",
            "chaser controller states and trigger radius",
            "_open_root",
        ),
        (
            quadrant_module,
            "build_chaser_quadrant_occupancy_result",
            "behavior",
            None,
            "_open_root",
        ),
        (
            epoch_behavior_module,
            "build_chaser_epoch_behavior_summary_result",
            "behavior",
            None,
            "_open_root",
        ),
        (
            gaze_tracking_module,
            "build_chaser_gaze_tracking_result",
            "behavior",
            None,
            "open_zarr_root",
        ),
    ],
)
def test_component_builders_stop_at_unsealed_authority_before_raw_navigation(
    monkeypatch: pytest.MonkeyPatch,
    module,
    builder_name: str,
    authority_kind: str,
    authority_value: str | None,
    opener_name: str,
) -> None:
    root = _RawNavigationTrap()
    authority_calls: list[tuple[str, str | None]] = []
    snapshot = _rejecting_snapshot(authority_calls)
    monkeypatch.setattr(module, opener_name, lambda *_a, **_k: root)
    monkeypatch.setattr(
        module,
        "_resolve_chaser_distance_run",
        lambda root_node, _run_name: (
            (snapshot, snapshot.run_name, snapshot.run_path)
            if root_node is root
            else pytest.fail("strict selector received the wrong root")
        ),
    )

    with pytest.raises(ChaserDistanceReadError, match="intentionally unsealed"):
        getattr(module, builder_name)("/unused/archive.zarr")

    assert authority_calls == [(authority_kind, authority_value)]


@pytest.mark.parametrize(
    ("module", "writer_name", "opener_name"),
    [
        (bout_response_module, "write_chaser_bout_response_component", "_open_root"),
        (egocentric_module, "write_chaser_egocentric_bearing_component", "_open_root"),
        (epoch_behavior_module, "write_chaser_epoch_behavior_summary_component", "_open_root"),
        (escape_events_module, "write_chaser_escape_events_component", "_open_root"),
        (escape_freeze_module, "write_chaser_escape_freeze_summary_component", "_open_root"),
        (gaze_tracking_module, "write_chaser_gaze_tracking_component", "open_zarr_root"),
        (near_field_module, "write_chaser_near_field_occupancy_component", "_open_root"),
        (quadrant_module, "write_chaser_quadrant_occupancy_component", "_open_root"),
        (radial_module, "write_chaser_radial_occupancy_component", "_open_root"),
        (response_regimes_module, "write_chaser_response_regimes_component", "_open_root"),
    ],
)
def test_unsealed_component_writers_reject_before_group_or_selector_mutation(
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    opener_name: str,
) -> None:
    root = _RawNavigationTrap()
    calls: list[tuple[object, dict[str, str]]] = []
    monkeypatch.setattr(module, opener_name, lambda *_a, **_k: root)

    def reject(root_node, **kwargs):
        calls.append((root_node, kwargs))
        raise ChaserDistanceReadError("derived publication is intentionally disabled")

    monkeypatch.setattr(
        module,
        "reject_unsealed_chaser_derived_publication",
        reject,
    )
    result = SimpleNamespace(
        chaser_distance_run_name="canonical",
        chaser_distance_run_path="analysis/chaser_distance_runs/canonical",
        component_name="component",
    )

    with pytest.raises(ChaserDistanceReadError, match="intentionally disabled"):
        getattr(module, writer_name)("/unused/archive.zarr", result)

    assert calls == [
        (
            root,
            {
                "run_name": "canonical",
                "run_path": "analysis/chaser_distance_runs/canonical",
                "relative_path": f"{module.COMPONENT_PARENT_NAME}/component",
            },
        )
    ]


@pytest.mark.parametrize(
    ("module", "class_name", "relative_path"),
    [
        (analysis_figures_module, "RecordingData", "cra_primary_endpoint"),
        (habituation_figures_module, "HabituationData", "chaser_escape_events"),
    ],
)
def test_visualization_data_classes_reject_derived_reads_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
    module,
    class_name: str,
    relative_path: str,
) -> None:
    root = _RawNavigationTrap()
    authority_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(module.zarr, "open_group", lambda *_a, **_k: root)
    monkeypatch.setattr(
        module,
        "load_chaser_distance_run",
        lambda root_node: (
            _rejecting_snapshot(authority_calls)
            if root_node is root
            else pytest.fail("strict reader received the wrong root")
        ),
    )

    with pytest.raises(ChaserDistanceReadError, match="intentionally unsealed"):
        getattr(module, class_name)("/unused/archive.zarr")

    assert authority_calls == [("derived", relative_path)]


@pytest.mark.parametrize(
    ("module", "collector_name", "relative_path"),
    [
        (visit_trajectories_module, "collect_visits", "egocentric_bearing"),
        (ring_traversal_module, "collect_ring_entries", "chaser_bout_response"),
        (ring_traversal_module, "collect_chase_ring_entries", "chaser_bout_response"),
        (ring_traversal_module, "_load_bout_segments", "chaser_bout_response"),
    ],
)
def test_visualization_collectors_reject_derived_reads_before_navigation(
    monkeypatch: pytest.MonkeyPatch,
    module,
    collector_name: str,
    relative_path: str,
) -> None:
    root = _RawNavigationTrap()
    authority_calls: list[tuple[str, str | None]] = []
    snapshot = _rejecting_snapshot(authority_calls)
    monkeypatch.setattr(module, "_open_root", lambda *_a, **_k: root)
    monkeypatch.setattr(
        module,
        "_resolve_chaser_distance_run",
        lambda root_node, _run_name: (
            (snapshot, snapshot.run_name, snapshot.run_path)
            if root_node is root
            else pytest.fail("strict selector received the wrong root")
        ),
    )

    kwargs = {"chaser_distance_run": "canonical"}
    if collector_name == "_load_bout_segments":
        kwargs["component"] = "canonical_component"
    with pytest.raises(ChaserDistanceReadError, match="intentionally unsealed"):
        getattr(module, collector_name)(
            "/unused/archive.zarr",
            **kwargs,
        )

    assert authority_calls == [("derived", relative_path)]


def test_trajectory_preflights_behavior_authority_before_raw_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    authority_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        trajectory_module,
        "resolve_zarr",
        lambda _recording_like: ("recording", "unused.zarr"),
    )
    monkeypatch.setattr(
        trajectory_module,
        "open_zarr_root",
        lambda *_a, **_k: root,
    )
    monkeypatch.setattr(
        trajectory_module,
        "load_chaser_distance_run",
        lambda root_node: (
            _rejecting_snapshot(authority_calls)
            if root_node is root
            else pytest.fail("strict reader received the wrong root")
        ),
    )

    with pytest.raises(ChaserDistanceReadError, match="behavior authority"):
        trajectory_module.main(["--recording-id", "recording"])

    assert authority_calls == [("behavior", None)]


def test_overview_chaser_artifact_fails_before_png_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    strict_calls: list[tuple[object, str]] = []
    authority_calls: list[tuple[str, str | None]] = []

    def strict_load(root_node, *, run_name):
        strict_calls.append((root_node, run_name))
        return _rejecting_snapshot(authority_calls, run_name=run_name)

    monkeypatch.setattr(overview_module, "load_chaser_distance_run", strict_load)
    monkeypatch.setattr(
        overview_module,
        "load_png_artifact_bytes",
        lambda *_a, **_k: pytest.fail("unsealed PNG must never be read"),
    )

    with pytest.raises(ChaserDistanceReadError, match="intentionally unsealed"):
        overview_module._load_chaser_artifact(
            root,
            run_name_or_path="analysis/chaser_distance_runs/exact_run",
            artifact_name="chaser_distance_timeseries_png",
        )

    assert strict_calls == [(root, "exact_run")]
    assert authority_calls == [
        ("derived", "visualizations/chaser_distance_timeseries_png")
    ]


def test_static_montage_marks_chaser_artifact_unavailable_without_png_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    strict_calls: list[tuple[object, str]] = []
    authority_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(montage_module, "open_zarr_root", lambda *_a, **_k: root)

    def strict_load(root_node, *, run_name):
        strict_calls.append((root_node, run_name))
        return _rejecting_snapshot(authority_calls, run_name=run_name)

    monkeypatch.setattr(montage_module, "load_chaser_distance_run", strict_load)
    monkeypatch.setattr(
        montage_module,
        "load_png_artifact_bytes",
        lambda *_a, **_k: pytest.fail("unsealed PNG must never be read"),
    )
    source = montage_module.SourceRecording(
        recording_id="recording",
        zarr_path=montage_module.Path("/unused/archive.zarr"),
    )
    spec = montage_module.MontageArtifactSpec(
        artifact_id="chaser",
        label="Chaser",
        path=(
            "analysis/chaser_distance_runs/exact_run/visualizations/"
            "chaser_distance_timeseries_png"
        ),
    )

    tiles, missing = montage_module._load_recording_tiles(
        source,
        [spec],
        fail_on_missing=False,
    )

    assert strict_calls == [(root, "exact_run")]
    assert authority_calls == [
        ("derived", "visualizations/chaser_distance_timeseries_png")
    ]
    assert len(tiles) == 1 and tiles[0].image is None
    assert "preflight failed closed" in str(tiles[0].error)
    assert len(missing) == 1


def test_registry_montage_marks_chaser_artifact_unavailable_without_png_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    strict_calls: list[tuple[object, str]] = []
    authority_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(montage_render_module, "open_zarr_root", lambda *_a, **_k: root)

    def strict_load(root_node, *, run_name):
        strict_calls.append((root_node, run_name))
        return _rejecting_snapshot(authority_calls, run_name=run_name)

    monkeypatch.setattr(montage_render_module, "load_chaser_distance_run", strict_load)
    monkeypatch.setattr(
        montage_render_module,
        "load_png_artifact_bytes",
        lambda *_a, **_k: pytest.fail("unsealed PNG must never be read"),
    )
    recording = montage_render_module.RegistryRecording(
        recording_id="recording",
        zarr_path=montage_module.Path("/unused/archive.zarr"),
        dataset_id="dataset",
        protocol_name="GoodCopBadCop",
        arena_id=None,
        recording_started_utc=None,
        chaser_count=2,
        chaser_behaviors=("aggressive", "inert"),
    )
    spec = montage_render_module.MontageArtifactSpec(
        artifact_id="chaser",
        label="Chaser",
        path=(
            "analysis/chaser_distance_runs/exact_run/egocentric_bearing/"
            "component/visualizations/polar_png"
        ),
    )

    tiles, missing = montage_render_module.load_recording_tiles(
        recording,
        [spec],
        fail_on_missing=False,
    )

    assert strict_calls == [(root, "exact_run")]
    assert authority_calls == [
        ("derived", "egocentric_bearing/component/visualizations/polar_png")
    ]
    assert len(tiles) == 1 and tiles[0].image is None
    assert "preflight failed closed" in str(tiles[0].error)
    assert len(missing) == 1


def test_export_latest_run_preflights_exact_reader_without_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[tuple[object, str]] = []

    def load(root_node, *, run_name):
        calls.append((root_node, run_name))
        return _snapshot(run_name)

    monkeypatch.setattr(export_module, "load_chaser_distance_run", load)
    group, name, reason = export_module._latest_run(
        root,
        "analysis/chaser_distance_runs",
        requested="exact_run",
    )

    assert calls == [(root, "exact_run")]
    assert group is None
    assert name == "exact_run"
    assert "no independently verified sealed semantic authority" in str(reason)


def test_export_preflight_marks_each_unsealed_table_unavailable_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot()

    monkeypatch.setattr(export_module, "load_chaser_distance_run", load)
    derived = export_module._SOURCE_TABLE_BY_V2[
        export_module.CHASER_DISTANCE_SUMMARY_TABLE
    ]
    independent = export_module._SOURCE_TABLE_BY_V2[
        export_module.CHASER_SPATIAL_TABLE
    ]
    diagnostics: list[dict[str, object]] = []

    remaining = export_module._preflight_unsealed_chaser_exports(
        root,
        tables={derived, independent},
        diagnostics=diagnostics,
    )

    assert calls == ["latest"]
    assert remaining == {independent}
    assert diagnostics == [
        {
            "table": derived,
            "status": "unavailable",
            "reason": (
                "requested chaser-distance summary/component has no independently "
                "verified sealed semantic authority; raw Zarr export is unavailable "
                "(analysis/chaser_distance_runs/canonical)"
            ),
            "chaser_distance_run": "canonical",
            "chaser_distance_path": "analysis/chaser_distance_runs/canonical",
        }
    ]


def test_dashboard_discovery_preflights_then_fails_before_artifact_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []
    monkeypatch.setattr(dashboard_module, "open_zarr_root", lambda *_a, **_k: root)

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot()

    monkeypatch.setattr(dashboard_module, "load_chaser_distance_run", load)

    with pytest.raises(
        dashboard_module.ChaserDashboardUnavailableError,
        match="raw dashboard reads are unavailable",
    ):
        dashboard_module.discover_chaser_dashboard_options("unused.zarr")

    assert calls == ["latest"]


def test_dashboard_explicit_run_preflights_exact_child_before_artifact_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []
    monkeypatch.setattr(dashboard_module, "open_zarr_root", lambda *_a, **_k: root)

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot(run_name)

    monkeypatch.setattr(dashboard_module, "load_chaser_distance_run", load)

    with pytest.raises(
        dashboard_module.ChaserDashboardUnavailableError,
        match="raw dashboard reads are unavailable",
    ):
        dashboard_module.load_chaser_dashboard_data(
            "unused.zarr",
            run_path="/analysis/chaser_distance_runs/exact_run/",
        )

    assert calls == ["exact_run"]


@pytest.mark.parametrize(
    ("loader_name", "component_name"),
    [
        ("load_goodcopbadcop_cra_primary_endpoint_data", "primary"),
        ("load_goodcopbadcop_cra_near_field_data", "near_field"),
        ("load_goodcopbadcop_epoch_behavior_data", "epoch_behavior"),
        ("load_goodcopbadcop_escape_freeze_data", "escape_freeze"),
    ],
)
def test_derived_component_loaders_preflight_before_raw_navigation(
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    component_name: str,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []
    monkeypatch.setattr(dashboard_module, "open_zarr_root", lambda *_a, **_k: root)

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot(run_name)

    monkeypatch.setattr(dashboard_module, "load_chaser_distance_run", load)
    loader = getattr(dashboard_module, loader_name)

    with pytest.raises(
        dashboard_module.ChaserDashboardUnavailableError,
        match="raw component reads are unavailable",
    ):
        loader(
            "unused.zarr",
            run_path="analysis/chaser_distance_runs/exact_run",
            component_name=component_name,
        )

    assert calls == ["exact_run"]


@pytest.mark.parametrize(
    "resolver_name",
    [
        "resolve_latest_egocentric_bearing_component_path",
        "resolve_latest_cra_primary_endpoint_component_path",
        "resolve_latest_cra_near_field_component_path",
        "resolve_latest_epoch_behavior_summary_component_path",
        "resolve_latest_escape_freeze_component_path",
    ],
)
def test_public_component_resolvers_are_fail_closed_normal_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    resolver_name: str,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot(run_name)

    monkeypatch.setattr(dashboard_module, "load_chaser_distance_run", load)
    resolver = getattr(dashboard_module, resolver_name)

    with pytest.raises(
        dashboard_module.ChaserDashboardUnavailableError,
        match="raw component reads are unavailable",
    ):
        resolver(
            root,
            run_path="analysis/chaser_distance_runs/exact_run",
        )

    assert calls == ["exact_run"]


def test_dashboard_rejects_noncanonical_run_path_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    monkeypatch.setattr(dashboard_module, "open_zarr_root", lambda *_a, **_k: root)
    monkeypatch.setattr(
        dashboard_module,
        "load_chaser_distance_run",
        lambda *_a, **_k: pytest.fail("invalid path must not reach the reader"),
    )

    with pytest.raises(
        dashboard_module.ChaserDashboardUnavailableError,
        match="run_path must be exactly",
    ):
        dashboard_module.load_chaser_dashboard_data(
            "unused.zarr",
            run_path="analysis/chaser_distance_runs/run/derived/component",
        )


def test_marimo_registry_hides_chaser_specs_after_strict_base_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot(run_name)

    monkeypatch.setattr(marimo_registry_module, "load_chaser_distance_run", load)

    options = marimo_registry_module._discover_goodcopbadcop_chaser_specs_fast(  # noqa: SLF001
        root,
        marimo_registry_module.Path("/unused/archive.zarr"),
        run_path_filter=None,
        artifact_filter=None,
    )

    assert options == []
    assert calls == ["latest"]


def test_marimo_registry_never_reads_explicit_unsealed_chaser_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot(run_name)

    monkeypatch.setattr(marimo_registry_module, "load_chaser_distance_run", load)

    option = marimo_registry_module._read_option(  # noqa: SLF001
        root,
        marimo_registry_module.Path("/unused/archive.zarr"),
        "analysis/chaser_distance_runs/exact/visualizations/dashboard",
    )

    assert option is None
    assert calls == ["exact"]


def test_marimo_chaser_component_discovery_stops_after_base_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    calls: list[str] = []
    monkeypatch.setattr(marimo_chaser_module, "open_zarr_root", lambda *_a, **_k: root)

    def load(_root_node, *, run_name):
        calls.append(run_name)
        return _snapshot(run_name)

    monkeypatch.setattr(marimo_chaser_module, "load_chaser_distance_run", load)

    rows = marimo_chaser_module.discover_chaser_gaze_tracking_components(
        "/unused/archive.zarr",
        distance_run_path="analysis/chaser_distance_runs/exact",
    )

    assert rows == ()
    assert calls == ["exact"]


def test_marimo_chaser_component_load_rejects_before_raw_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    authority_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(marimo_chaser_module, "open_zarr_root", lambda *_a, **_k: root)
    monkeypatch.setattr(
        marimo_chaser_module,
        "load_chaser_distance_run",
        lambda _root_node, *, run_name: _rejecting_snapshot(
            authority_calls,
            run_name=run_name,
        ),
    )

    with pytest.raises(ChaserDistanceReadError, match="intentionally unsealed"):
        marimo_chaser_module.load_chaser_gaze_tracking_view(
            "/unused/archive.zarr",
            "analysis/chaser_distance_runs/exact/gaze_tracking/component",
        )

    assert authority_calls == [("derived", "gaze_tracking/component")]


def test_marimo_core_baseline_uses_sealed_epoch_table_without_dashboard_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RawNavigationTrap()
    snapshot = SimpleNamespace(
        epoch_labels=("training_event", "pre_event"),
        epoch_start_frame=np.asarray([100, 10], dtype=np.int64),
        epoch_end_frame=np.asarray([199, 29], dtype=np.int64),
        fps=10.0,
    )
    option = marimo_core_module.CoreBehaviorOption(
        zarr_path=marimo_core_module.Path("/unused/archive.zarr"),
        run_path="analysis/track_kinematics_runs/offline/run",
        run_name="run",
        label="run",
        track_id=0,
        source_paths={},
        attrs={},
    )
    source = marimo_core_module.CoreBehaviorSource("/unused/archive.zarr", option)
    monkeypatch.setattr(source, "_root", lambda: root)
    monkeypatch.setattr(
        marimo_core_module,
        "load_chaser_distance_run",
        lambda root_node, *, run_name: (
            snapshot
            if root_node is root and run_name == "latest"
            else pytest.fail("strict baseline preflight received the wrong selection")
        ),
    )

    baseline = source.baseline_interval()

    assert baseline == marimo_core_module.BaselineInterval(
        label="pre_event",
        start_s=1.0,
        stop_s=3.0,
    )
