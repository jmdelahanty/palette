from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_relative_distance_view import (
    ChaserRelativeDistanceView,
)
import fisheye.analysis_workflows.provider_chaser_distance_successor as successor
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.provider_chaser_distance_schema import (
    PROVIDER_CHASER_DISTANCE_SCHEMA_V1,
    ProviderChaserDistanceDimensions,
    ProviderChaserDistanceSchemaError,
)
from tests.unit.fisheye.test_chaser_relative_distance_view import _prepared_base
from tests.unit.fisheye.test_chaser_relative_frame_storage import (
    _proxy_projection_record,
    _proxy_publication_binding,
)


_DIGEST = "a" * 64


def _authority(name: str) -> dict[str, str]:
    return {
        "recording_id": "recording-1",
        "source_authority_id": f"{name}-source",
        "source_digest": f"{name}-source-digest",
        "provider_id": f"{name}-provider",
        "provider_digest": f"{name}-provider-digest",
        "coordinate_authority_id": "camera-native-v1",
        "scale_authority_id": "scale-v1",
        "timing_authority_id": "camera-time-v1",
        "row_axis_authority_id": "camera-rows-v1",
        "row_axis_authority_digest": "camera-rows-digest",
    }


def _envelope(record: dict[str, object]) -> dict[str, object]:
    return {"record": record, "sha256": canonical_json_sha256(record)}


def _source(
    *,
    include_trial: bool = True,
    unit: str = "mm",
    include_scale: bool = True,
):
    base, registries = _prepared_base(include_optional=include_trial)
    view = ChaserRelativeDistanceView.from_base_arrays(
        recording_id="recording-1",
        source_run_path="analysis/chaser_relative_frame_runs/source-v1",
        source_run_digest="d" * 64,
        n_frames=3,
        n_chasers=2,
        base_arrays=base,
        registries=registries,
    )
    projection = _proxy_projection_record()
    publication = _proxy_publication_binding(projection)
    context = {
        "acquisition_projection": _envelope(projection),
        "acquisition_projection_publication": _envelope(publication),
    }
    run_manifest = {
        "coordinate_policy": {
            "policy_id": "source_camera_y_down_v1",
            "coordinate_authority_id": "camera-native-v1",
            "coordinate_frame": "source_camera_pixels",
            "origin": "top_left",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
        },
        "timing_policy": {
            "policy_id": "acquisition_camera_timing_v1",
            "timing_authority_id": "camera-time-v1",
            "timing_digest": _DIGEST,
            "frame_key_name": "acquisition_frame_id",
            "track_sample_key_name": "track_sample_id",
            "timestamp_field": "timestamp_ns",
        },
    }
    if include_scale:
        run_manifest["scale_policy"] = {
            "policy_id": "source_camera_scale_v1",
            "scale_authority_id": "scale-v1",
            "scale_digest": _DIGEST,
            "pixels_per_unit": 10.0,
            "unit": unit,
        }
    handle = SimpleNamespace(
        context=context,
        source_authorities={
            "fish_position": _authority("fish"),
            "chaser_position": _authority("chaser"),
        },
        manifest_sha256="e" * 64,
        payload_digest="f" * 64,
        run_manifest=run_manifest,
    )
    return handle, view


def test_prepare_preserves_flat_axis_and_binds_proxy_and_denominators(monkeypatch):
    handle, view = _source()
    observed = []

    def loader(value):
        observed.append(value)
        return view

    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", loader)
    prepared = successor.prepare_provider_chaser_distance_successor(handle)

    assert observed == [handle]
    assert prepared.dimensions == ProviderChaserDistanceDimensions(3, 2)
    assert prepared.array("acquisition_frame_id").tolist() == [10, 10, 11, 11, 12, 12]
    assert prepared.array("chaser_identity_code").tolist() == [1, 2, 1, 2, 1, 2]
    np.testing.assert_allclose(
        prepared.array("distance_px"),
        [3.0, 4.0, 3.0, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        prepared.array("distance_mm"),
        [0.3, 0.4, 0.3, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    assert prepared.array("trial_id").tolist() == [4, 4, 4, 4, 5, 5]
    assert prepared.manifest["selector_eligible"] is False
    assert prepared.manifest["production_authority"] is False
    assert prepared.manifest["registry_update"] is False
    assert prepared.manifest["selection"] == "none"
    assert prepared.manifest["temporal_alignment"]["temporal_alignment_class"] == (
        "controller_input_provenance_proxy"
    )
    assert prepared.manifest["temporal_alignment"]["timestamp_matching_performed"] is False
    assert prepared.manifest["denominators"] == {
        "unique_acquisition_frame_count": 3,
        "frame_x_chaser_relation_row_count": 6,
        "valid_source_position_frame_count": 3,
        "valid_distance_relation_row_count": 3,
        "native_stimulus_sample_count": 5,
        "selected_input_acquisition_frame_count": 3,
        "native_sample_multiplicity": {
            "numerator": "native_stimulus_sample_count",
            "denominator": "unique_acquisition_frame_count",
            "ratio": 5 / 3,
        },
    }
    assert prepared.manifest["optional_fields"]["distance_mm_triple_present"] is True
    assert all("content_sha256" in declaration for declaration in prepared.manifest["array_declarations"])
    assert all(not array.flags.writeable for array in prepared.arrays.values())


def test_prepare_non_mm_scale_produces_pixel_only_success(monkeypatch):
    handle, view = _source(unit="px")
    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", lambda _: view)

    prepared = successor.prepare_provider_chaser_distance_successor(handle)

    assert "distance_px" in prepared.arrays
    assert not {"distance_mm", "distance_mm_valid", "distance_mm_reason_code"} & set(prepared.arrays)
    assert prepared.manifest["scale_policy"]["available"] is True
    assert prepared.manifest["scale_policy"]["unit"] == "px"
    assert prepared.manifest["scale_policy"]["mm_derivation_available"] is False
    assert prepared.manifest["optional_fields"]["distance_mm_triple_present"] is False


def test_prepare_absent_scale_produces_pixel_only_success(monkeypatch):
    handle, view = _source(include_scale=False)
    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", lambda _: view)

    prepared = successor.prepare_provider_chaser_distance_successor(handle)

    assert "distance_px" in prepared.arrays
    assert not {"distance_mm", "distance_mm_valid", "distance_mm_reason_code"} & set(prepared.arrays)
    assert prepared.manifest["scale_policy"] == {
        "available": False,
        "unit": None,
        "mm_derivation_available": False,
        "authority": None,
    }
    assert prepared.manifest["optional_fields"]["distance_mm_triple_present"] is False


def test_prepare_keeps_trial_triple_optional(monkeypatch):
    handle, view = _source(include_trial=False)
    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", lambda _: view)

    prepared = successor.prepare_provider_chaser_distance_successor(handle)

    assert not {"trial_id", "trial_valid", "trial_reason_code"} & set(prepared.arrays)
    assert prepared.manifest["optional_fields"]["trial_triple_present"] is False


def test_schema_rejects_frame_evidence_that_is_not_repeated(monkeypatch):
    handle, view = _source()
    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", lambda _: view)
    prepared = successor.build_provider_chaser_distance_successor(handle)
    arrays = {name: np.array(values, copy=True) for name, values in prepared.arrays.items()}
    arrays["acquisition_frame_id"][1] = 99

    with pytest.raises(ProviderChaserDistanceSchemaError, match="frame-level evidence"):
        PROVIDER_CHASER_DISTANCE_SCHEMA_V1.require(
            arrays,
            dimensions=ProviderChaserDistanceDimensions(3, 2),
        )


def test_schema_rejects_partial_mm_triple(monkeypatch):
    handle, view = _source()
    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", lambda _: view)
    prepared = successor.build_provider_chaser_distance_successor(handle)
    arrays = {name: np.array(values, copy=True) for name, values in prepared.arrays.items()}
    del arrays["distance_mm_reason_code"]

    with pytest.raises(ProviderChaserDistanceSchemaError, match="distance_mm"):
        PROVIDER_CHASER_DISTANCE_SCHEMA_V1.require(
            arrays,
            dimensions=ProviderChaserDistanceDimensions(3, 2),
        )


def test_array_content_changes_bind_declaration_and_payload_digests(monkeypatch):
    handle, view = _source()
    monkeypatch.setattr(successor, "load_chaser_relative_distance_view", lambda _: view)
    prepared = successor.build_provider_chaser_distance_successor(handle)
    changed_arrays = {
        name: np.array(values, copy=True) for name, values in prepared.arrays.items()
    }
    changed_arrays["distance_px"][0] = 99.0

    prepared_manifest = prepared.to_json()
    original_declarations_digest = canonical_json_sha256(
        prepared_manifest["array_declarations"]
    )
    changed_declarations = successor._declarations(changed_arrays)
    changed_declarations_digest = canonical_json_sha256(changed_declarations)
    assert changed_declarations_digest != original_declarations_digest
    original_distance_declaration = next(
        declaration
        for declaration in prepared_manifest["array_declarations"]
        if declaration["path"] == "distance_px"
    )
    changed_distance_declaration = next(
        declaration
        for declaration in changed_declarations
        if declaration["path"] == "distance_px"
    )
    assert changed_distance_declaration["content_sha256"] != original_distance_declaration["content_sha256"]

    changed_payload = prepared.to_json()
    changed_payload["array_declarations"] = changed_declarations
    changed_payload.pop("payload_digest")
    assert canonical_json_sha256(changed_payload) != prepared.payload_digest
