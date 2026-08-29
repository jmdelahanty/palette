from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.marimo.components import chaser_exact_successors as facade
from apps.marimo.components.chaser_exact.distance_traces import (
    _trace_display_projection,
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
) -> InteractiveSpecOption:
    run_path = f"analysis/chaser_spatial_occupancy_runs/{run_name}"
    spec = {
        "schema_id": "palette.chaser_exact_successor_explorer_spec",
        "schema_version": 2,
        "renderer": CHASER_EXACT_SUCCESSOR_RENDERER,
        "bundle_status": "exact_selector_ineligible",
        "bundle_manifest_sha256": manifest_sha256,
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


def test_provider_routes_are_closed_and_controls_are_explicit() -> None:
    assert ANALYSIS_IDS == (
        "radial_near_field",
        "distance_traces",
        "trajectory_overlays",
        "provenance",
    )
    assert EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("distance_traces")
    assert not EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("provenance")
    assert EXACT_CHASER_PROVIDER_ADAPTER.build_controls("radial_near_field") is None
    with pytest.raises(ExactChaserUnknownAnalysisError, match="Unsupported"):
        EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("distance-ish")


def test_only_selected_analysis_requests_relative_arrays(
    tmp_path: Path, monkeypatch
) -> None:
    option = _option(tmp_path / "recording.zarr")
    observed: list[tuple[str, bool]] = []

    def fake_loader(
        zarr_path,
        selected_option,
        *,
        selection_identity,
        load_relative,
    ):
        assert zarr_path == option.zarr_path
        assert selected_option is option
        observed.append((selection_identity.analysis_id, load_relative))
        return selection_identity

    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.provider.load_exact_chaser_projection",
        fake_loader,
    )

    radial = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="radial_near_field"
    )
    distance = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="distance_traces"
    )

    assert radial.analysis_id == "radial_near_field"
    assert distance.analysis_id == "distance_traces"
    assert observed == [
        ("radial_near_field", False),
        ("distance_traces", True),
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
    for changed in (changed_display, changed_manifest, changed_run):
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
    assert facade.EXACT_CHASER_PROVIDER_ADAPTER is EXACT_CHASER_PROVIDER_ADAPTER


def test_palette_explorer_uses_one_exact_provider_load_and_render_boundary() -> None:
    source = Path("apps/marimo/palette_explorer.py").read_text(encoding="utf-8")

    assert "build_exact_distance_traces_output" not in source
    assert "build_exact_radial_near_field_output" not in source
    assert "build_exact_trajectory_overlays_output" not in source
    assert source.count("EXACT_CHASER_PROVIDER_ADAPTER.load_projection(") == 1
    assert source.count("EXACT_CHASER_PROVIDER_ADAPTER.render(") == 1
