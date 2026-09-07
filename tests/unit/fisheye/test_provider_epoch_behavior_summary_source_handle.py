from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle as subject
from apps.marimo.components.chaser_exact_epoch_behavior_discovery import (
    compatible_epoch_behavior_binding,
)
from fisheye.analysis_workflows.core_motion_source_handle import (
    bind_core_motion_track_source_handle,
    core_motion_dependency_record,
)
from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    ANALYSIS_CLASS_ID,
    ANALYSIS_CLASS_VERSION,
    CORE_EPOCH_BEHAVIOR_CONSUMER_ID,
    CORE_EPOCH_REQUIRED_CAPABILITIES,
    CORE_SEMANTIC_METHOD_VERSION,
    CORE_SEMANTIC_SCHEMA_VERSION,
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    METHOD_ID,
    PARENT_PATH,
    SCHEMA_ID,
    SEMANTIC_EPOCH_BINDING_MODE,
    SEMANTIC_METHOD_VERSION,
    SEMANTIC_SCHEMA_VERSION,
    ProviderEpochBehaviorSummaryError,
    build_provider_epoch_behavior_summary_plan,
)
from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
    ProviderEpochBehaviorSummarySourceError,
    load_provider_epoch_behavior_summary_source_handle,
    validate_provider_epoch_behavior_summary_metadata,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
)


class _Group(dict[str, Any]):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def group_keys(self) -> tuple[str, ...]:
        return tuple(self)


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.attrs = {
            "palette_storage_schema_id": "palette.columnar_zarr_storage.v1",
            "palette_storage_writer": "fisheye.shared.zarr.columnar.store_array",
        }

    def __getitem__(self, key: object) -> np.ndarray:
        return self.values[key]


def _encoded(values: np.ndarray, *, width: int) -> np.ndarray:
    encoded = np.zeros((values.shape[0], width), dtype=np.uint8)
    for index, value in enumerate(values):
        payload = bytes(value).rstrip(b"\x00")
        encoded[index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return encoded


def _semantic() -> dict[str, Any]:
    roles = [
        {
            "analysis_role": role,
            "source_window_id": index,
            "source_interval_sha256": str(index + 1) * 64,
            "selected_start_frame": index * 10,
            "selected_end_frame_exclusive": (index + 1) * 10,
            "protocol_semantic_step_index": 1,
            "protocol_semantic_step_ref": (
                "protocol_semantic_snapshot@recipe.steps[1]"
            ),
        }
        for index, role in enumerate(CHASER_WINDOW_ROLES)
    ]
    return {
        "run_path": ("analysis/protocol_semantic_chaser_selection_runs/semantic-v2"),
        "manifest_sha256": "a" * 64,
        "protocol_semantic_hash": f"sha256:{'b' * 64}",
        "roles": list(CHASER_WINDOW_ROLES),
        "semantic_role_bindings": roles,
        "selector_eligible": False,
        "production_authority": False,
    }


def _attrs(*, run_name: str = "epoch-v2") -> dict[str, Any]:
    run_path = f"{PARENT_PATH}/{run_name}"
    semantic = _semantic()
    sources = {
        "epoch_binding_mode": SEMANTIC_EPOCH_BINDING_MODE,
        "epoch_selection": {"record": {"run": "epoch"}, "sha256": "c" * 64},
        "provider_motion": {
            "run_path": "analysis/track_kinematics_runs/provider/motion-v1",
            "manifest_sha256": "d" * 64,
            "verification_digest": "e" * 64,
            "track_id": 0,
        },
        "swim_bouts": {
            "run_path": "analysis/swim_bout_runs/bouts-v1",
            "lineage_hash": "f" * 64,
            "track_id": 0,
        },
        "protocol_semantic_selection": semantic,
    }
    parameters = {
        "track_id": 0,
        "epoch_binding_mode": SEMANTIC_EPOCH_BINDING_MODE,
        "physical_speed_level": "filtered",
        "rate_denominator": "valid_tracked_duration_s",
        "spatial_metrics": ("omitted_requires_separately_selected_position_provider"),
        "protocol_to_acquisition_alignment": (
            "sealed_epoch_selection_proxy_not_physical_presentation"
        ),
    }
    offer = {
        "selector_eligible": False,
        "readiness": {"scientific": "ready"},
    }
    offer_sha = canonical_json_sha256(offer)
    body = {
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": SEMANTIC_SCHEMA_VERSION,
        },
        "method_id": METHOD_ID,
        "method_version": SEMANTIC_METHOD_VERSION,
        "epoch_binding_mode": SEMANTIC_EPOCH_BINDING_MODE,
        "run_path": run_path,
        "recording_id": "recording-1",
        "dimensions": {
            "n_epoch_rows": 3,
            "n_bout_rows": 2,
            "n_bout_histogram_rows": 30,
            "n_inter_bout_interval_histogram_rows": 6,
        },
        "sources": sources,
        "parameters": parameters,
        "analysis_offer_sha256": offer_sha,
        "array_declarations": [
            {
                "path": "per_epoch_fish/window_id",
                "dtype": "<i4",
                "shape": [3],
                "content_sha256": "1" * 64,
            }
        ],
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    manifest = {**body, "payload_digest": canonical_json_sha256(body)}
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "method_version": SEMANTIC_METHOD_VERSION,
        "epoch_binding_mode": SEMANTIC_EPOCH_BINDING_MODE,
        "run_path": run_path,
        "recording_id": "recording-1",
        RUN_COMPLETION_CONTRACT_ATTR: RUN_COMPLETION_CONTRACT,
        RUN_COMPLETION_STATUS_ATTR: RUN_STATUS_COMPLETE,
        RUN_NAME_ATTR: run_name,
        "stage_selector_eligible": False,
        "production_authority": False,
        "registry_update": False,
        "selection": "none",
        "source_refs": sources,
        "source_refs_sha256": canonical_json_sha256(sources),
        "parameters": parameters,
        "analysis_offer": offer,
        "analysis_offer_sha256": offer_sha,
        MANIFEST_ATTR: manifest,
        MANIFEST_DIGEST_ATTR: canonical_json_sha256(manifest),
    }


def _redigest(
    attrs: dict[str, Any],
    *,
    sources: dict[str, Any] | None = None,
    offer: dict[str, Any] | None = None,
) -> None:
    manifest = deepcopy(attrs[MANIFEST_ATTR])
    if sources is not None:
        attrs["source_refs"] = sources
        attrs["source_refs_sha256"] = canonical_json_sha256(sources)
        manifest["sources"] = sources
    if offer is not None:
        offer_sha256 = canonical_json_sha256(offer)
        attrs["analysis_offer"] = offer
        attrs["analysis_offer_sha256"] = offer_sha256
        manifest["analysis_offer_sha256"] = offer_sha256
    body = {key: value for key, value in manifest.items() if key != "payload_digest"}
    manifest = {**body, "payload_digest": canonical_json_sha256(body)}
    attrs[MANIFEST_ATTR] = manifest
    attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)


def _core_attrs(tmp_path: Path, *, run_name: str = "epoch-core-v3") -> dict[str, Any]:
    from fisheye.analysis_workflows.provider_analysis_offers import (
        TemporalSelectionIdentity,
    )
    from tests.unit.fisheye.test_core_authority_roster import _bound_core_motion

    bound = _bound_core_motion(tmp_path)
    handle = bind_core_motion_track_source_handle(
        bound,
        consumer_id=CORE_EPOCH_BEHAVIOR_CONSUMER_ID,
        required_capabilities=CORE_EPOCH_REQUIRED_CAPABILITIES,
        track_id=7,
    )
    dependency = dict(core_motion_dependency_record(handle))
    attrs = _attrs(run_name=run_name)
    attrs["schema_version"] = CORE_SEMANTIC_SCHEMA_VERSION
    attrs["method_version"] = CORE_SEMANTIC_METHOD_VERSION
    attrs["recording_id"] = "recording-a"
    parameters = {**attrs["parameters"], "track_id": 7}
    attrs["parameters"] = parameters
    epoch_record = {
        "run": {"path": "analysis/stimulus_epoch_runs/epochs-v2"},
        "source_timeline": {"recording_id": "recording-a"},
        "source_timeline_digest": "1" * 64,
        "recording_timing_authority": {"sha256": "2" * 64},
        "selection_sha256": "3" * 64,
    }
    temporal = TemporalSelectionIdentity(
        selection_id="stimulus_epoch_compatibility.v1",
        run_path=epoch_record["run"]["path"],
        recording_id="recording-a",
        source_timeline_sha256=epoch_record["source_timeline_digest"],
        resolved_sha256=epoch_record["selection_sha256"],
        timing_authority_sha256=epoch_record["recording_timing_authority"]["sha256"],
    )
    sources = {
        "epoch_binding_mode": SEMANTIC_EPOCH_BINDING_MODE,
        "epoch_selection": {"record": epoch_record, "sha256": "3" * 64},
        "core_motion": dependency,
        "swim_bouts": {
            "run_path": dependency["swim_bout_run_path"],
            "payload_sha256": dependency["swim_bout_source_binding_sha256"],
            "source_track_motion_manifest_sha256": dependency["motion_manifest_sha256"],
            "track_id": dependency["track_id"],
        },
        "protocol_semantic_selection": _semantic(),
    }
    offer = {
        "schema_id": "palette.core_epoch_behavior_summary.analysis_offer",
        "schema_version": 1,
        "analysis_class_id": ANALYSIS_CLASS_ID,
        "analysis_class_version": ANALYSIS_CLASS_VERSION,
        "computation_id": METHOD_ID,
        "computation_version": CORE_SEMANTIC_METHOD_VERSION,
        "scientific_readiness": "ready",
        "temporal_selection_sha256": temporal.sha256,
        "core_motion_dependency": dependency,
    }
    manifest = deepcopy(attrs[MANIFEST_ATTR])
    scientific = dict(manifest["scientific_schema"])
    scientific["schema_version"] = CORE_SEMANTIC_SCHEMA_VERSION
    manifest["scientific_schema"] = scientific
    manifest["method_version"] = CORE_SEMANTIC_METHOD_VERSION
    manifest["recording_id"] = "recording-a"
    manifest["parameters"] = parameters
    attrs[MANIFEST_ATTR] = manifest
    _redigest(attrs, sources=sources, offer=offer)
    return attrs


def test_metadata_validation_preserves_exact_semantic_v2_binding() -> None:
    attrs = _attrs()

    binding = validate_provider_epoch_behavior_summary_metadata(
        attrs,
        run_path=f"{PARENT_PATH}/epoch-v2",
        run_name="epoch-v2",
        expected_recording_id="recording-1",
        expected_semantic_selection=_semantic(),
    )

    assert binding["run_path"] == f"{PARENT_PATH}/epoch-v2"
    assert binding["parameters"]["physical_speed_level"] == "filtered"
    assert binding["source_protocol_semantic_selection"]["roles"] == tuple(
        CHASER_WINDOW_ROLES
    )
    assert binding["array_declaration_count"] == 1
    assert set(binding) == {
        "run_path",
        "manifest_sha256",
        "payload_digest",
        "source_protocol_semantic_selection",
        "source_provider_motion",
        "source_swim_bouts",
        "parameters",
        "dimensions",
        "array_declaration_count",
    }


def test_metadata_validation_admits_exact_core_semantic_v3_binding(
    tmp_path: Path,
) -> None:
    attrs = _core_attrs(tmp_path)

    binding = validate_provider_epoch_behavior_summary_metadata(
        attrs,
        run_path=f"{PARENT_PATH}/epoch-core-v3",
        run_name="epoch-core-v3",
        expected_recording_id="recording-a",
        expected_semantic_selection=_semantic(),
    )

    assert binding["run_path"] == f"{PARENT_PATH}/epoch-core-v3"
    assert binding["summary_profile"] == "core_semantic_v3"
    assert binding["source_core_motion"]["track_id"] == 7
    assert binding["source_provider_motion"] is None


def test_metadata_validation_rejects_core_offer_source_substitution(
    tmp_path: Path,
) -> None:
    attrs = _core_attrs(tmp_path)
    offer = deepcopy(attrs["analysis_offer"])
    dependency = deepcopy(offer["core_motion_dependency"])
    dependency["motion_run_path"] = "analysis/track_kinematics_runs/offline/other"
    dependency_body = {
        key: value for key, value in dependency.items() if key != "record_sha256"
    }
    dependency["record_sha256"] = canonical_json_sha256(dependency_body)
    offer["core_motion_dependency"] = dependency
    _redigest(attrs, offer=offer)

    with pytest.raises(
        ProviderEpochBehaviorSummarySourceError,
        match="differs from its source binding",
    ):
        validate_provider_epoch_behavior_summary_metadata(
            attrs,
            run_path=f"{PARENT_PATH}/epoch-core-v3",
            run_name="epoch-core-v3",
        )


def test_metadata_validation_rejects_stale_core_motion_dependency(
    tmp_path: Path,
) -> None:
    attrs = _core_attrs(tmp_path)
    sources = deepcopy(attrs["source_refs"])
    dependency = deepcopy(sources["core_motion"])
    dependency["record_sha256"] = "0" * 64
    sources["core_motion"] = dependency
    offer = {**deepcopy(attrs["analysis_offer"]), "core_motion_dependency": dependency}
    _redigest(attrs, sources=sources, offer=offer)

    with pytest.raises(
        ProviderEpochBehaviorSummarySourceError,
        match="core motion dependency",
    ):
        validate_provider_epoch_behavior_summary_metadata(
            attrs,
            run_path=f"{PARENT_PATH}/epoch-core-v3",
            run_name="epoch-core-v3",
        )


def test_metadata_validation_rejects_legacy_offer_on_core_semantic_v3(
    tmp_path: Path,
) -> None:
    attrs = _core_attrs(tmp_path)
    _redigest(
        attrs,
        offer={
            "selector_eligible": False,
            "readiness": {"scientific": "ready"},
        },
    )

    with pytest.raises(
        ProviderEpochBehaviorSummarySourceError,
        match="core analysis offer",
    ):
        validate_provider_epoch_behavior_summary_metadata(
            attrs,
            run_path=f"{PARENT_PATH}/epoch-core-v3",
            run_name="epoch-core-v3",
        )


def test_metadata_validation_rejects_another_core_consumer_receipt(
    tmp_path: Path,
) -> None:
    attrs = _core_attrs(tmp_path)
    sources = deepcopy(attrs["source_refs"])
    dependency = deepcopy(sources["core_motion"])
    receipt = dependency["core_authority_consumption_receipt"]
    receipt["consumer_id"] = "goodbatbadbat.another_consumer_v1"
    receipt_body = {
        key: value for key, value in receipt.items() if key != "record_sha256"
    }
    receipt["record_sha256"] = canonical_json_sha256(receipt_body)
    dependency_body = {
        key: value for key, value in dependency.items() if key != "record_sha256"
    }
    dependency["record_sha256"] = canonical_json_sha256(dependency_body)
    sources["core_motion"] = dependency
    offer = {**deepcopy(attrs["analysis_offer"]), "core_motion_dependency": dependency}
    _redigest(attrs, sources=sources, offer=offer)

    with pytest.raises(
        ProviderEpochBehaviorSummarySourceError,
        match="core consumer receipt is incompatible",
    ):
        validate_provider_epoch_behavior_summary_metadata(
            attrs,
            run_path=f"{PARENT_PATH}/epoch-core-v3",
            run_name="epoch-core-v3",
        )


def test_metadata_validation_rejects_core_dependency_from_another_archive(
    tmp_path: Path,
) -> None:
    attrs = _core_attrs(tmp_path)

    with pytest.raises(
        ProviderEpochBehaviorSummarySourceError,
        match="belongs to another archive",
    ):
        validate_provider_epoch_behavior_summary_metadata(
            attrs,
            run_path=f"{PARENT_PATH}/epoch-core-v3",
            run_name="epoch-core-v3",
            expected_analysis_zarr=tmp_path / "other.zarr",
        )


def test_metadata_validation_rejects_raw_speed_even_when_redigested() -> None:
    attrs = _attrs()
    manifest = dict(attrs[MANIFEST_ATTR])
    parameters = {**manifest["parameters"], "physical_speed_level": "raw"}
    body = {
        **{key: value for key, value in manifest.items() if key != "payload_digest"},
        "parameters": parameters,
    }
    manifest = {**body, "payload_digest": canonical_json_sha256(body)}
    attrs["parameters"] = parameters
    attrs[MANIFEST_ATTR] = manifest
    attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)

    with pytest.raises(ProviderEpochBehaviorSummarySourceError, match="prohibit raw"):
        validate_provider_epoch_behavior_summary_metadata(
            attrs,
            run_path=f"{PARENT_PATH}/epoch-v2",
            run_name="epoch-v2",
        )


def test_metadata_discovery_requires_one_unambiguous_exact_child() -> None:
    parent = _Group({"epoch-v2": _Group(attrs=_attrs())})
    root = _Group({PARENT_PATH: parent})

    binding = compatible_epoch_behavior_binding(
        root,
        recording_id="recording-1",
        spatial_sources={"protocol_semantic_selection": _semantic()},
    )

    assert binding is not None
    assert binding["run_path"].endswith("/epoch-v2")
    parent["epoch-v2-copy"] = _Group(attrs=_attrs(run_name="epoch-v2-copy"))
    assert (
        compatible_epoch_behavior_binding(
            root,
            recording_id="recording-1",
            spatial_sources={"protocol_semantic_selection": _semantic()},
        )
        is None
    )


def test_semantic_v2_discovery_does_not_mix_core_semantic_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apps.marimo.components.chaser_exact_epoch_behavior_discovery as discovery

    parent = _Group(
        {
            "epoch-v2": _Group(attrs=_attrs()),
            "epoch-core-v3": _Group(
                attrs={
                    "schema_version": CORE_SEMANTIC_SCHEMA_VERSION,
                    "method_version": CORE_SEMANTIC_METHOD_VERSION,
                }
            ),
        }
    )
    root = _Group({PARENT_PATH: parent})
    validated: list[str] = []

    def validate(*_args: object, run_path: str, **_kwargs: object) -> dict[str, str]:
        validated.append(run_path)
        return {"run_path": run_path}

    monkeypatch.setattr(
        discovery,
        "validate_provider_epoch_behavior_summary_metadata",
        validate,
    )

    binding = discovery.compatible_epoch_behavior_binding(
        root,
        recording_id="recording-1",
        spatial_sources={"protocol_semantic_selection": _semantic()},
    )

    assert binding is not None
    assert binding["run_path"] == f"{PARENT_PATH}/epoch-v2"
    assert validated == [f"{PARENT_PATH}/epoch-v2"]


def test_semantic_plan_rejects_raw_speed_before_source_loading(tmp_path: Path) -> None:
    archive = tmp_path / "recording.zarr"
    archive.mkdir()

    with pytest.raises(ProviderEpochBehaviorSummaryError, match="rejects raw"):
        build_provider_epoch_behavior_summary_plan(
            archive,
            scratch_root=tmp_path / "scratch",
            run_name="epoch-v2",
            epoch_run_name="semantic-epochs-v2",
            protocol_semantic_selection_run_name="semantic-v2",
            motion_run="analysis/track_kinematics_runs/provider/motion-v1",
            swim_bout_run_name="bouts-v1",
            speed_level="raw",
        )


def test_targeted_loader_reconstructs_logical_fixed_byte_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording.zarr"
    archive.mkdir()
    receipt_path = tmp_path / "receipt.json"
    logical = np.asarray(
        [b"chaser_pre", b"chaser_training", b"chaser_post"],
        dtype="S32",
    )
    path = "per_epoch_fish/analysis_role"
    attrs = _attrs()
    original = attrs[MANIFEST_ATTR]
    body = {
        **{key: value for key, value in original.items() if key != "payload_digest"},
        "array_declarations": [
            {
                "path": path,
                "dtype": logical.dtype.str,
                "shape": list(logical.shape),
                "content_sha256": array_values_sha256(logical),
            }
        ],
    }
    manifest = {**body, "payload_digest": canonical_json_sha256(body)}
    attrs[MANIFEST_ATTR] = manifest
    attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)
    run = _Group(
        {path: _Array(_encoded(logical, width=16))},
        attrs=attrs,
    )

    monkeypatch.setattr(subject, "open_zarr_root", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda *_args, **_kwargs: {
            "record_sha256": "a" * 64,
            "direct_metadata_inventory": {"inventory_sha256": "b" * 64},
        },
    )

    handle = load_provider_epoch_behavior_summary_source_handle(
        archive,
        run_name="epoch-v2",
        expected_recording_id="recording-1",
        direct_validation_receipt=receipt_path,
        required_array_paths=(path,),
    )

    np.testing.assert_array_equal(handle.array(path), logical)
    assert handle.array(path).dtype == logical.dtype
    assert handle.verified_array_paths == (path,)


def test_targeted_loader_accepts_exact_core_semantic_v3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording-a.zarr"
    archive.mkdir()
    receipt_path = tmp_path / "receipt.json"
    path = "per_epoch_fish/window_id"
    logical = np.asarray([0, 1, 2], dtype=np.int32)
    attrs = _core_attrs(tmp_path)
    manifest = deepcopy(attrs[MANIFEST_ATTR])
    body = {
        **{key: value for key, value in manifest.items() if key != "payload_digest"},
        "array_declarations": [
            {
                "path": path,
                "dtype": logical.dtype.str,
                "shape": list(logical.shape),
                "content_sha256": array_values_sha256(logical),
            }
        ],
    }
    manifest = {**body, "payload_digest": canonical_json_sha256(body)}
    attrs[MANIFEST_ATTR] = manifest
    attrs[MANIFEST_DIGEST_ATTR] = canonical_json_sha256(manifest)
    run = _Group({path: _Array(logical)}, attrs=attrs)

    monkeypatch.setattr(subject, "open_zarr_root", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda *_args, **_kwargs: {
            "record_sha256": "a" * 64,
            "direct_metadata_inventory": {"inventory_sha256": "b" * 64},
        },
    )

    handle = load_provider_epoch_behavior_summary_source_handle(
        archive,
        run_name="epoch-core-v3",
        expected_recording_id="recording-a",
        direct_validation_receipt=receipt_path,
        required_array_paths=(path,),
    )

    np.testing.assert_array_equal(handle.array(path), logical)
    assert handle.manifest["scientific_schema"]["schema_version"] == 3
    assert handle.verified_array_paths == (path,)
