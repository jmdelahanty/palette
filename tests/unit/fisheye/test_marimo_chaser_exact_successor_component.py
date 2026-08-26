from __future__ import annotations

from typing import Any

import numpy as np

from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.chaser_exact_successors import (
    _trace_display_projection,
    _trajectory_display_indices,
)
from apps.marimo.components.registry import (
    CHASER_EXACT_SUCCESSOR_RENDERER,
    discover_exact_chaser_successor_options,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


class _Group(dict[str, Any]):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def group_keys(self) -> tuple[str, ...]:
        return tuple(self)


def _relative(path: str, provider_id: str, provider_digest: str) -> _Group:
    manifest = {
        "recording_id": "recording-1",
        "selector_eligible": False,
        "selection": "none",
        "source_authorities": {
            "fish_position": {
                "provider_id": provider_id,
                "provider_digest": provider_digest,
            }
        },
    }
    digest = canonical_json_sha256(manifest)
    return _Group(
        attrs={
            "schema_id": "palette.analysis.chaser_relative_frame",
            "schema_version": 1,
            "run_path": path,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "chaser_relative_frame_manifest": manifest,
            "chaser_relative_frame_manifest_sha256": digest,
        }
    )


def _successor(
    path: str,
    kind: str,
    scientific: dict[str, Any],
) -> _Group:
    manifest = {
        "successor_kind": kind,
        "run_path": path,
        "recording_id": "recording-1",
        "selector_eligible": False,
        "selection": "none",
        "scientific_manifest": scientific,
    }
    digest = canonical_json_sha256(manifest)
    return _Group(
        attrs={
            "schema_id": "palette.analysis.composable_chaser_successor.run",
            "schema_version": 1,
            "successor_kind": kind,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "composable_chaser_successor_manifest": manifest,
            "composable_chaser_successor_manifest_sha256": digest,
        }
    )


def _archive() -> _Group:
    providers = (
        ("keypoint", "keypoint.v1", "a" * 64),
        ("detection", "detection.v1", "b" * 64),
    )
    root = _Group()
    records = []
    for role, provider_id, provider_digest in providers:
        relative_path = f"analysis/chaser_relative_frame_runs/{role}-relative"
        radial_path = f"analysis/chaser_radial_near_field_runs/{role}-radial"
        relative = _relative(relative_path, provider_id, provider_digest)
        relative_binding = {
            "run_path": relative_path,
            "manifest_sha256": relative.attrs["chaser_relative_frame_manifest_sha256"],
        }
        radial = _successor(
            radial_path,
            "chaser_radial_near_field",
            {
                "position_provider": {
                    "provider_id": provider_id,
                    "provider_digest": provider_digest,
                    "status": "first_class_explicit_authority",
                },
                "sources": {"relative_frame": relative_binding},
            },
        )
        root[relative_path] = relative
        root[radial_path] = radial
        records.append(
            {
                "provider_role": role,
                "provider_id": provider_id,
                "provider_digest": provider_digest,
                "relative_frame": relative_binding,
                "radial_near_field": {
                    "run_path": radial_path,
                    "manifest_sha256": radial.attrs[
                        "composable_chaser_successor_manifest_sha256"
                    ],
                },
            }
        )
    spatial_name = "paired-spatial-v1"
    spatial_path = f"analysis/chaser_spatial_occupancy_runs/{spatial_name}"
    spatial = _successor(
        spatial_path,
        "chaser_spatial_occupancy",
        {"sources": {"position_providers": records}},
    )
    parent = _Group({spatial_name: spatial})
    root["analysis/chaser_spatial_occupancy_runs"] = parent
    root[spatial_path] = spatial
    return root


def test_exact_successor_discovery_uses_spatial_bundle_and_exact_children(
    monkeypatch,
) -> None:
    root = _archive()
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert options[0].renderer == CHASER_EXACT_SUCCESSOR_RENDERER
    assert options[0].spec["bundle_status"] == "exact_selector_ineligible"
    assert options[0].spec["provider_ids"] == ["keypoint.v1", "detection.v1"]
    assert group_specs_by_provider(options) == {
        "stimulus_chaser_exact_successors": options
    }


def test_exact_successor_discovery_hides_stale_child_provider_binding(
    monkeypatch,
) -> None:
    root = _archive()
    radial = root["analysis/chaser_radial_near_field_runs/detection-radial"]
    manifest = radial.attrs["composable_chaser_successor_manifest"]
    manifest["scientific_manifest"]["position_provider"]["provider_digest"] = "c" * 64
    radial.attrs["composable_chaser_successor_manifest_sha256"] = canonical_json_sha256(
        manifest
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_trace_display_projection_preserves_extrema_and_missing_breaks() -> None:
    x = np.arange(100, dtype=np.float64)
    y = np.sin(x / 10.0)
    y[40] = 50.0
    valid = np.ones(100, dtype=bool)
    valid[48:53] = False

    display_x, display_y = _trace_display_projection(x, y, valid, max_points=24)

    assert 40.0 in display_x
    assert 50.0 in display_y
    assert np.isnan(display_y).any()
    finite_x = display_x[np.isfinite(display_y)]
    assert not np.any((finite_x >= 48) & (finite_x <= 52))


def test_trajectory_display_projection_retains_coordinate_extrema() -> None:
    xy = np.column_stack((np.arange(1000), -np.arange(1000))).astype(np.float64)
    valid = np.ones(1000, dtype=bool)

    indices = _trajectory_display_indices(xy, valid, max_points=40)

    assert indices.size <= 44
    assert {0, 999}.issubset(indices.tolist())
