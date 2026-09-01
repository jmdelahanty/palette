from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle as subject
from apps.marimo.components.chaser_exact_epoch_behavior_discovery import (
    compatible_epoch_behavior_binding,
)
from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
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
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
    ProviderEpochBehaviorSummarySourceError,
    load_provider_epoch_behavior_summary_source_handle,
    validate_provider_epoch_behavior_summary_metadata,
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


def test_metadata_validation_admits_only_exact_semantic_v2_binding() -> None:
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
