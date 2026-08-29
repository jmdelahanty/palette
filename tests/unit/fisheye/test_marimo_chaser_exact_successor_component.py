from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.chaser_exact_successors import (
    _verify_bundle_children,
    _trace_display_projection,
    _trajectory_display_indices,
)
from apps.marimo.components.registry import (
    CHASER_EXACT_SUCCESSOR_RENDERER,
    discover_exact_chaser_successor_options,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    MINIMAL_EXACT_CHILD_PROFILE,
    RECEIPT_BOUND_PROFILE,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
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
        receipt_bound_relative_binding = {
            **relative_binding,
            "validation_receipt_sha256": "f" * 64,
            "verification_mode": "receipt_bound_targeted_array_rehash_v1",
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
                "relative_frame": receipt_bound_relative_binding,
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


def _redigest_spatial(root: _Group) -> None:
    spatial = root["analysis/chaser_spatial_occupancy_runs/paired-spatial-v1"]
    spatial.attrs["composable_chaser_successor_manifest_sha256"] = (
        canonical_json_sha256(spatial.attrs["composable_chaser_successor_manifest"])
    )


def _provider_records(root: _Group) -> list[dict[str, Any]]:
    spatial = root["analysis/chaser_spatial_occupancy_runs/paired-spatial-v1"]
    return spatial.attrs["composable_chaser_successor_manifest"]["scientific_manifest"][
        "sources"
    ]["position_providers"]


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
    assert options[0].spec["schema_version"] == 2
    assert options[0].spec["provider_ids"] == ["keypoint.v1", "detection.v1"]
    proofs = options[0].spec["relative_frame_binding_proofs"]
    assert len(proofs) == 2
    assert proofs[0]["spatial_binding_profile"] == RECEIPT_BOUND_PROFILE
    assert proofs[0]["radial_binding_profile"] == MINIMAL_EXACT_CHILD_PROFILE
    assert proofs[0]["validation_receipt_sha256"] == "f" * 64
    assert proofs[0]["verification_mode"] == ("receipt_bound_targeted_array_rehash_v1")
    assert group_specs_by_provider(options) == {
        "stimulus_chaser_exact_successors": options
    }


def test_production_binding_shapes_are_not_whole_object_equal() -> None:
    root = _archive()
    spatial_binding = _provider_records(root)[0]["relative_frame"]
    radial = root["analysis/chaser_radial_near_field_runs/keypoint-radial"]
    radial_binding = radial.attrs["composable_chaser_successor_manifest"][
        "scientific_manifest"
    ]["sources"]["relative_frame"]

    assert spatial_binding != radial_binding
    proof = require_same_exact_relative_frame_child(spatial_binding, radial_binding)

    assert dict(proof.normalized_identity) == {
        "run_path": spatial_binding["run_path"],
        "manifest_sha256": spatial_binding["manifest_sha256"],
    }
    assert proof.expected.profile_id == RECEIPT_BOUND_PROFILE
    assert proof.observed.profile_id == MINIMAL_EXACT_CHILD_PROFILE


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("validation_receipt_sha256", "not-a-digest", "lowercase SHA-256"),
        ("verification_mode", "deep_audit", "unsupported"),
    ],
)
def test_receipt_bound_binding_rejects_invalid_evidence(
    field: str,
    value: str,
    match: str,
) -> None:
    binding = dict(_provider_records(_archive())[0]["relative_frame"])
    binding[field] = value

    with pytest.raises(ExactRelativeFrameBindingError, match=match):
        validate_exact_relative_frame_binding(binding)


def test_relative_binding_rejects_unrecognized_field_set() -> None:
    binding = dict(_provider_records(_archive())[0]["relative_frame"])
    binding["receipt_path"] = "/not/an/admitted/field"

    with pytest.raises(ExactRelativeFrameBindingError, match="unrecognized"):
        validate_exact_relative_frame_binding(binding)


@pytest.mark.parametrize(
    "run_name",
    ["latest", "latest_complete", "authoritative_run", "selected", "current_run"],
)
def test_relative_binding_rejects_selector_like_child_names(run_name: str) -> None:
    binding = dict(_provider_records(_archive())[0]["relative_frame"])
    binding["run_path"] = f"analysis/chaser_relative_frame_runs/{run_name}"

    with pytest.raises(ExactRelativeFrameBindingError, match="exact child"):
        validate_exact_relative_frame_binding(binding)


@pytest.mark.parametrize("field", ["run_path", "manifest_sha256"])
def test_relative_binding_equivalence_rejects_different_child_identity(
    field: str,
) -> None:
    root = _archive()
    spatial_binding = dict(_provider_records(root)[0]["relative_frame"])
    radial = root["analysis/chaser_radial_near_field_runs/keypoint-radial"]
    radial_binding = dict(
        radial.attrs["composable_chaser_successor_manifest"]["scientific_manifest"][
            "sources"
        ]["relative_frame"]
    )
    radial_binding[field] = (
        "analysis/chaser_relative_frame_runs/another-exact-run"
        if field == "run_path"
        else "e" * 64
    )

    with pytest.raises(ExactRelativeFrameBindingError, match="different exact"):
        require_same_exact_relative_frame_child(spatial_binding, radial_binding)


def test_loader_bundle_validation_accepts_enriched_spatial_and_minimal_radial() -> None:
    root = _archive()
    records = _provider_records(root)
    semantic = {"run_path": "analysis/protocol_semantic/run", "sha256": "1" * 64}
    geometry = {"authority_id": "reviewed-arena", "sha256": "2" * 64}
    epochs = [{"epoch_id": "pre", "start_frame": 0, "end_frame": 10}]
    arena = {"radius_mm": 10.0}
    radials = []
    for record in records:
        role = record["provider_role"]
        radial_group = root[f"analysis/chaser_radial_near_field_runs/{role}-radial"]
        radial_manifest = radial_group.attrs["composable_chaser_successor_manifest"]
        scientific = dict(radial_manifest["scientific_manifest"])
        scientific["sources"] = {
            **scientific["sources"],
            "protocol_semantic_selection": semantic,
            "arena_geometry_and_scale": geometry,
        }
        scientific["epoch_records"] = epochs
        scientific["arena"] = arena
        radials.append(
            SimpleNamespace(
                run_path=record["radial_near_field"]["run_path"],
                manifest_sha256=record["radial_near_field"]["manifest_sha256"],
                scientific_manifest=scientific,
            )
        )
    spatial = SimpleNamespace(
        scientific_manifest={
            "sources": {
                "position_providers": records,
                "protocol_semantic_selection": semantic,
                "arena_geometry_and_scale": geometry,
            },
            "epoch_records": epochs,
        }
    )

    provider_ids, proofs = _verify_bundle_children(
        spatial,
        radials,
        tuple(record["relative_frame"] for record in records),
    )

    assert provider_ids == ("keypoint.v1", "detection.v1")
    assert tuple(proof.expected.profile_id for proof in proofs) == (
        RECEIPT_BOUND_PROFILE,
        RECEIPT_BOUND_PROFILE,
    )
    assert tuple(proof.observed.profile_id for proof in proofs) == (
        MINIMAL_EXACT_CHILD_PROFILE,
        MINIMAL_EXACT_CHILD_PROFILE,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_receipt_sha256", "invalid"),
        ("verification_mode", "unsupported"),
    ],
)
def test_discovery_hides_invalid_receipt_binding(
    monkeypatch,
    field: str,
    value: str,
) -> None:
    root = _archive()
    _provider_records(root)[0]["relative_frame"][field] = value
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "run_path",
            "analysis/chaser_relative_frame_runs/another-exact-relative-run",
        ),
        ("manifest_sha256", "e" * 64),
    ],
)
def test_discovery_hides_wrong_relative_child_identity(
    monkeypatch,
    field: str,
    value: str,
) -> None:
    root = _archive()
    _provider_records(root)[0]["relative_frame"][field] = value
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_relative_child_from_another_recording(monkeypatch) -> None:
    root = _archive()
    relative = root["analysis/chaser_relative_frame_runs/keypoint-relative"]
    manifest = relative.attrs["chaser_relative_frame_manifest"]
    manifest["recording_id"] = "recording-2"
    digest = canonical_json_sha256(manifest)
    relative.attrs["chaser_relative_frame_manifest_sha256"] = digest
    _provider_records(root)[0]["relative_frame"]["manifest_sha256"] = digest
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_incomplete_relative_child(monkeypatch) -> None:
    root = _archive()
    relative = root["analysis/chaser_relative_frame_runs/keypoint-relative"]
    relative.attrs["palette_run_completion_status"] = "writing"
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_reversed_provider_roles(monkeypatch) -> None:
    root = _archive()
    _provider_records(root).reverse()
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_forbidden_parent_selector(monkeypatch) -> None:
    root = _archive()
    root["analysis/chaser_spatial_occupancy_runs"].attrs["latest"] = "paired-spatial-v1"
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_does_not_retry_unconsolidated_metadata(monkeypatch) -> None:
    calls = []

    def fail_open(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("missing consolidated generation")

    monkeypatch.setattr("apps.marimo.components.registry.open_zarr_root", fail_open)

    assert discover_exact_chaser_successor_options("recording.zarr") == []
    assert calls == [{"mode": "r", "use_consolidated": True}]


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
