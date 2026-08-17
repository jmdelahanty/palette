from __future__ import annotations

from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.shared.subject_position_keypoint_source as source_mod
from fisheye.shared.anatomy_profile import AnatomyProfile
from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.subject_position_keypoint_source import (
    KeypointPositionSourceError,
    KeypointPositionSourcePolicy,
    load_bound_keypoint_position_source,
    revalidate_bound_keypoint_position_source,
)


_PROFILE_PATH = (
    Path(__file__).parents[3]
    / "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
)
_BINDING_ID = "zebrafish_larva_keypoint_traditional_v3_v1"
_V2_BINDING_ID = "zebrafish_larva_keypoint_traditional_v2_v1"
_RUN_PATH = "keypoints_runs/current_keypoints"
_RUN_ID = "current_keypoints"
_SKELETON_DIGEST = "a" * 64
_CANARY_RUN_PATH = "keypoints_runs/sealed_keypoints"
_CANARY_RUN_ID = "sealed_keypoints"
_CANARY_OWNER = "1" * 32


class _FakeArray:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __getitem__(self, key):
        if key is Ellipsis:
            return self.value
        return self.value[key]


class _FakeGroup:
    def __init__(self, *, attrs=None, children=None) -> None:
        self.attrs = dict(attrs or {})
        self.children = dict(children or {})

    def __getitem__(self, key):
        return self.children[key]


class _FakeIdentity:
    def __init__(self, leading_dimension: int) -> None:
        self.leading_dimension = leading_dimension
        self.contract = SimpleNamespace(digest=lambda: "b" * 64)

    def assert_verified(self) -> None:
        return None


def _profile() -> AnatomyProfile:
    return AnatomyProfile.from_json(_PROFILE_PATH)


def _v2_pose_binding(*, reverse_snout_edges: bool = False) -> dict[str, object]:
    edges = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 4]]
    if reverse_snout_edges:
        edges[3:5] = [[3, 1], [3, 2]]
    return build_explicit_pose_model_schema_binding(
        model_sha256="9" * 64,
        assertion_id="subject-position-v2-skeleton-fixture",
        skeleton_id="pose_skel_traditional_v2",
        model_kpt_shape=[5, 3],
        keypoint_labels=[
            "swim_bladder",
            "eye_left",
            "eye_right",
            "snout_tip",
            "tail_tip",
        ],
        edges=edges,
    )


def test_v2_anatomy_binding_requires_exact_skeleton_semantics() -> None:
    binding = _profile().binding(_V2_BINDING_ID)

    skeleton_id, _pose_digest, labels, role_to_index = source_mod._bind_source_schema(
        binding,
        _v2_pose_binding(),
    )

    assert skeleton_id == "pose_skel_traditional_v2"
    assert labels == (
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
    )
    assert role_to_index == {"eye_left": 1, "eye_right": 2, "swim_bladder": 0}

    with pytest.raises(KeypointPositionSourceError, match="skeleton semantics"):
        source_mod._bind_source_schema(
            binding,
            _v2_pose_binding(reverse_snout_edges=True),
        )


def _arrays(n_rows: int = 2) -> dict[str, np.ndarray]:
    n_keypoints = 10
    points = np.arange(n_rows * n_keypoints * 2, dtype=np.float32).reshape(
        n_rows, n_keypoints, 2
    )
    image_points = points + np.float32(100.0)
    arrays = {
        "instance_key": np.arange(n_rows, dtype=np.uint64),
        "source_crop_row_ids": np.arange(n_rows, dtype=np.int64),
        "source_acquisition_frame_index": np.arange(10, 10 + n_rows, dtype=np.int64),
        "frame_indices": np.arange(n_rows, dtype=np.int64),
        "frame_row_offsets": np.arange(n_rows + 1, dtype=np.int64),
        "source_crop_row_signature": np.zeros((n_rows, 32), dtype=np.uint8),
        "keypoint_row_signature": np.zeros((n_rows, 32), dtype=np.uint8),
        "keypoints_roi": points,
        "keypoints_img": image_points,
        "keypoint_confidences": np.full(
            (n_rows, n_keypoints), np.float32(0.9), dtype=np.float32
        ),
        "keypoint_valid": np.ones((n_rows, n_keypoints), dtype=bool),
        "pose_confidence": np.full(n_rows, np.float32(0.9), dtype=np.float32),
        "pose_bbox_xyxy_roi": np.ones((n_rows, 4), dtype=np.float32),
        "pose_bbox_xyxy_img": np.ones((n_rows, 4), dtype=np.float32),
        "pose_success": np.ones(n_rows, dtype=bool),
    }
    return arrays


def _fake_surfaces(arrays: dict[str, np.ndarray]) -> SimpleNamespace:
    identity = _FakeIdentity(len(arrays["instance_key"]))
    source_crop = _FakeGroup(
        children={name: _FakeArray(np.zeros(len(arrays["instance_key"]))) for name in (
            "instance_key",
            "frame_indices",
            "source_acquisition_frame_index",
            "source_row_signature",
            "roi_coordinates_full",
            "roi_sizes_full",
        )}
    )
    descriptor = SimpleNamespace(
        profile_id=source_mod.SOURCE_CAMERA_PROFILE_ID,
        space_id="source_camera_image_px",
        geometry_type="point_xy",
        pixel_convention="continuous",
        digest=lambda: "c" * 64,
    )
    coordinate = SimpleNamespace(
        descriptor=descriptor,
        reference_frame_authority=SimpleNamespace(record_sha256="g" * 64),
        coordinate_node=_FakeArray(arrays["keypoints_img"]),
    )
    context = SimpleNamespace(
        row_identity=identity,
        run_path=_RUN_PATH,
        source=SimpleNamespace(_rowset_node=source_crop),
    )
    return SimpleNamespace(
        keypoints_img=coordinate,
        context=context,
    )


def _fixture(monkeypatch: pytest.MonkeyPatch):
    profile = _profile()
    binding = profile.binding(_BINDING_ID)
    package = binding["source_schema"]["package_payload"]
    arrays = _arrays()
    run_attrs = {
        source_mod.RUN_COMPLETION_CONTRACT_ATTR: source_mod.RUN_COMPLETION_CONTRACT,
        source_mod.RUN_COMPLETION_STATUS_ATTR: source_mod.RUN_STATUS_COMPLETE,
        "stage_selector_eligible": True,
    }
    run = _FakeGroup(
        attrs=run_attrs,
        children={name: _FakeArray(value) for name, value in arrays.items()},
    )
    parent = _FakeGroup(
        attrs={"latest": _RUN_ID, "latest_complete": _RUN_ID}
    )
    root = _FakeGroup(
        attrs={"current_keypoint_group_path": _RUN_PATH},
        children={"keypoints_runs": parent, _RUN_PATH: run},
    )
    pose_binding = {
        "pose_schema": package,
        "binding_sha256": "d" * 64,
    }
    payload = {
        "run_id": _RUN_ID,
        "pose_model_schema_binding": pose_binding,
        "logical_schema": {
            "dimensions": {
                "n_frames": 2,
                "n_instances": 2,
                "n_keypoints": 10,
                "source_width": 640,
                "source_height": 480,
            }
        },
        "logical_content": {
            "digest": "e" * 64,
            "document": {"skeleton_digest": _SKELETON_DIGEST},
        },
        "storage_plan": {"storage_profile": {}},
        "publication": {"metadata_declarations_digest": "f" * 64},
    }
    manifest = {"payload": payload}
    surfaces = _fake_surfaces(arrays)

    monkeypatch.setattr(source_mod, "_manifest_and_payload", lambda run, run_id: (manifest, payload))
    monkeypatch.setattr(source_mod, "_validate_published_metadata", lambda *args, **kwargs: "f" * 64)
    monkeypatch.setattr(source_mod, "_require_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(source_mod, "keypoint_skeleton_digest", lambda binding: _SKELETON_DIGEST)
    monkeypatch.setattr(source_mod, "load_persisted_keypoint_coordinate_surfaces", lambda root, path: surfaces)
    monkeypatch.setattr(source_mod, "require_bound_keypoint_coordinate_surfaces", lambda value: value)
    monkeypatch.setattr(source_mod, "require_bound_row_identity_contract", lambda value: value)
    monkeypatch.setattr(source_mod, "require_source_camera_pixel_frame_authority", lambda value: value)
    return root, profile, arrays, pose_binding


def _canary_fixture(monkeypatch: pytest.MonkeyPatch):
    root, profile, arrays, pose_binding = _fixture(monkeypatch)
    base_run = root[_RUN_PATH]
    payload = {
        "run_id": _CANARY_RUN_ID,
        "pose_model_schema_binding": pose_binding,
        "logical_schema": {
            "dimensions": {
                "n_frames": 2,
                "n_instances": 2,
                "n_keypoints": 10,
                "source_width": 640,
                "source_height": 480,
            }
        },
        "logical_content": {
            "digest": "e" * 64,
            "document": {"skeleton_digest": _SKELETON_DIGEST},
        },
    }
    manifest = {
        "schema_id": "palette.keypoint.run_manifest",
        "schema_version": 1,
        "payload_digest": "p" * 64,
        "payload": payload,
    }
    canary_run = _FakeGroup(
        attrs={
            source_mod.RUN_COMPLETION_CONTRACT_ATTR: source_mod.RUN_COMPLETION_CONTRACT,
            source_mod.RUN_COMPLETION_STATUS_ATTR: source_mod.RUN_STATUS_COMPLETE,
            "status": source_mod.RUN_STATUS_COMPLETE,
            "palette_run_completion_status": source_mod.RUN_STATUS_COMPLETE,
            "stage_selector_eligible": False,
            "production_candidate": True,
            "production_selector_activation": source_mod.SEALED_BUNDLE_PRODUCTION_SELECTOR_ACTIVATION,
            source_mod.ATOMIC_PUBLICATION_OWNER_ATTR: _CANARY_OWNER,
            source_mod.KEYPOINT_RUN_MANIFEST_ATTRIBUTE: manifest,
        },
        children=base_run.children,
    )
    root.children[_CANARY_RUN_PATH] = canary_run
    root["keypoints_runs"].children[_CANARY_RUN_ID] = canary_run
    root.attrs["current_keypoint_group_path"] = _RUN_PATH

    raw_member = {
        "role": "raw_keypoints",
        "run_id": _CANARY_RUN_ID,
        "run_path": _CANARY_RUN_PATH,
        "publication_owner_uuid": _CANARY_OWNER,
        "manifest_payload_digest": manifest["payload_digest"],
        "manifest_document_digest": source_mod.canonical_json_sha256(manifest),
        "logical_content_digest": payload["logical_content"]["digest"],
    }
    authority = {
        "schema_id": source_mod.KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_ID,
        "schema_version": source_mod.KEYPOINT_BUNDLE_AUTHORITY_SCHEMA_VERSION,
        "generation": 1,
        "base_generation": 0,
        "policy": "sealed_four_surface_root_authority_then_consolidated_visibility_v1",
        "activation_plan_payload_digest": "a" * 64,
        "prior_authority_present": False,
        "prior_authority_digest": None,
        "crop": {"run_id": "crop", "run_path": "crop_runs/crop"},
        "members": {"raw_keypoints": raw_member},
        "activated_at_utc": "2026-08-17T00:00:00+00:00",
        "activation_owner_uuid": "b" * 32,
    }
    root.attrs[source_mod.KEYPOINT_BUNDLE_AUTHORITY_ATTR] = authority
    root.attrs[source_mod.KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR] = 1
    surfaces = _fake_surfaces(arrays)
    surfaces.context.run_path = _CANARY_RUN_PATH
    monkeypatch.setattr(
        source_mod,
        "_manifest_and_payload",
        lambda run, run_id: (manifest, payload),
    )
    monkeypatch.setattr(
        source_mod,
        "resolve_active_keypoint_bundle_from_root",
        lambda root: {"authority": authority},
    )
    monkeypatch.setattr(
        source_mod,
        "_validate_bundle_authority_direct_consolidated",
        lambda analysis_zarr, *, authority: authority,
    )
    monkeypatch.setattr(
        source_mod,
        "load_persisted_ineligible_keypoint_coordinate_surfaces",
        lambda root, path: surfaces,
    )
    monkeypatch.setattr(
        source_mod,
        "require_bound_ineligible_keypoint_coordinate_surfaces",
        lambda value: value,
    )
    monkeypatch.setattr(
        source_mod,
        "load_persisted_keypoint_coordinate_surfaces",
        lambda root, path: pytest.fail(
            "bundle-member mode called the selector-eligible coordinate loader"
        ),
    )
    return root, profile, pose_binding, authority


def test_loader_builds_exact_role_bindings_without_inventing_confidence_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, arrays, _ = _fixture(monkeypatch)

    source = load_bound_keypoint_position_source(
        root,
        run_path=_RUN_PATH,
        policy=KeypointPositionSourcePolicy(profile, _BINDING_ID),
    )

    assert source.source_modality == "keypoint"
    assert source.source_kind == "canonical_keypoint_coordinate_selector"
    assert source.instance_key.dtype == np.dtype("<u8")
    assert source.source_acquisition_frame_index.dtype == np.dtype("<i8")
    assert np.array_equal(source.source_row_index, np.arange(2, dtype=np.int64))
    assert source.source_binding_digest == profile.binding(_BINDING_ID)["binding_sha256"]
    assert source.confidence_valid is None
    assert source.expression_bindings.keypoints["eye_left"].values.shape == (2, 2)
    assert source.expression_bindings.keypoints["eye_left"].confidence_valid is None
    assert np.array_equal(
        source.expression_bindings.keypoints["eye_left"].values,
        arrays["keypoints_img"][:, 1, :],
    )


def test_default_policy_uses_selector_eligible_coordinate_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, _, _ = _fixture(monkeypatch)
    calls: list[str] = []
    surfaces = _fake_surfaces(_arrays())
    monkeypatch.setattr(
        source_mod,
        "load_persisted_keypoint_coordinate_surfaces",
        lambda root, path: (calls.append("eligible") or surfaces),
    )
    monkeypatch.setattr(
        source_mod,
        "load_persisted_ineligible_keypoint_coordinate_surfaces",
        lambda root, path: pytest.fail("default policy used the canary loader"),
    )

    source = load_bound_keypoint_position_source(
        root,
        run_path=_RUN_PATH,
        policy=KeypointPositionSourcePolicy(profile, _BINDING_ID),
    )

    assert calls == ["eligible"]
    assert source.authority_mode == source_mod.KEYPOINT_AUTHORITY_MODE_CANONICAL_SELECTOR
    assert source.keypoint_bundle_authority is None
    assert source.keypoint_bundle_authority_digest is None


def test_sealed_bundle_canary_uses_root_authority_and_ineligible_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, _, authority = _canary_fixture(monkeypatch)
    revalidation_calls: list[object] = []
    monkeypatch.setattr(
        source_mod,
        "require_bound_ineligible_keypoint_coordinate_surfaces",
        lambda value: (revalidation_calls.append(value) or value),
    )

    source = load_bound_keypoint_position_source(
        root,
        run_path=_CANARY_RUN_PATH,
        policy=KeypointPositionSourcePolicy(
            profile,
            _BINDING_ID,
            authority_mode=source_mod.KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY,
        ),
    )

    assert source.source_kind == source_mod.SEALED_BUNDLE_SOURCE_KIND
    assert source.authority_mode == source_mod.KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY
    assert source.keypoint_bundle_authority_digest == source_mod.canonical_json_sha256(
        authority
    )
    assert source.keypoint_bundle_authority["members"]["raw_keypoints"]["run_path"] == (
        _CANARY_RUN_PATH
    )
    assert len(revalidation_calls) == 1


def test_coordinate_successor_canary_uses_explicit_ineligible_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, arrays, _ = _fixture(monkeypatch)
    surfaces = _fake_surfaces(arrays)
    authority = {
        "schema_id": "palette.coordinate_successor_authority",
        "schema_version": 1,
    }
    monkeypatch.setattr(
        source_mod,
        "_require_coordinate_successor_authority",
        lambda *args, **kwargs: (authority, "7" * 64),
    )
    monkeypatch.setattr(
        source_mod,
        "load_persisted_ineligible_keypoint_coordinate_surfaces",
        lambda root, path: surfaces,
    )
    monkeypatch.setattr(
        source_mod,
        "require_bound_ineligible_keypoint_coordinate_surfaces",
        lambda value: value,
    )
    monkeypatch.setattr(
        source_mod,
        "load_persisted_keypoint_coordinate_surfaces",
        lambda *args, **kwargs: pytest.fail(
            "coordinate-successor mode used the eligible coordinate loader"
        ),
    )

    source = load_bound_keypoint_position_source(
        root,
        run_path=_RUN_PATH,
        policy=KeypointPositionSourcePolicy(
            profile,
            _BINDING_ID,
            authority_mode=(
                source_mod.KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY
            ),
        ),
    )

    assert source.source_kind == source_mod.COORDINATE_SUCCESSOR_SOURCE_KIND
    assert source.coordinate_successor_authority == authority
    assert source.coordinate_successor_authority_digest == "7" * 64
    assert source.keypoint_bundle_authority is None


def test_coordinate_successor_canary_does_not_fallback_when_authority_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, _arrays_value, _ = _fixture(monkeypatch)

    def reject(*args, **kwargs):
        raise KeypointPositionSourceError("synthetic stale successor")

    monkeypatch.setattr(
        source_mod,
        "_require_coordinate_successor_authority",
        reject,
    )
    with pytest.raises(KeypointPositionSourceError, match="stale successor"):
        load_bound_keypoint_position_source(
            root,
            run_path=_RUN_PATH,
            policy=KeypointPositionSourcePolicy(
                profile,
                _BINDING_ID,
                authority_mode=(
                    source_mod.KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY
                ),
            ),
        )


def test_root_bundle_authority_compares_two_open_metadata_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = source_mod._validate_bundle_authority_direct_consolidated
    _, _, _, authority = _canary_fixture(monkeypatch)
    direct = _FakeGroup(
        attrs={
            source_mod.KEYPOINT_BUNDLE_AUTHORITY_ATTR: authority,
            source_mod.KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR: 1,
        }
    )
    consolidated = _FakeGroup(
        attrs={
            source_mod.KEYPOINT_BUNDLE_AUTHORITY_ATTR: authority,
            source_mod.KEYPOINT_BUNDLE_AUTHORITY_GENERATION_ATTR: 1,
        }
    )
    opened: list[bool] = []

    def fake_open(path, mode="r", *, use_consolidated=False):
        opened.append(bool(use_consolidated))
        return consolidated if use_consolidated else direct

    monkeypatch.setattr(source_mod, "open_zarr_root", fake_open)

    assert validator("/tmp/fake.zarr", authority=authority) == authority
    assert opened == [False, True]

    consolidated.attrs[source_mod.KEYPOINT_BUNDLE_AUTHORITY_ATTR] = {
        **authority,
        "generation": 2,
    }
    with pytest.raises(KeypointPositionSourceError, match="state differs"):
        validator("/tmp/fake.zarr", authority=authority)


@pytest.mark.parametrize(
    "tamper",
    (
        lambda authority: authority["members"]["raw_keypoints"].update(
            {"logical_content_digest": "z" * 64}
        ),
        lambda authority: authority["members"]["raw_keypoints"].update(
            {"run_path": _RUN_PATH}
        ),
    ),
)
def test_sealed_bundle_member_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tamper,
) -> None:
    root, profile, _, authority = _canary_fixture(monkeypatch)
    tamper(authority)

    with pytest.raises(KeypointPositionSourceError, match="exact requested|logical"):
        load_bound_keypoint_position_source(
            root,
            run_path=_CANARY_RUN_PATH,
            policy=KeypointPositionSourcePolicy(
                profile,
                _BINDING_ID,
                authority_mode=source_mod.KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY,
            ),
        )


def test_sealed_bundle_authority_digest_is_rechecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, _, _ = _canary_fixture(monkeypatch)
    source = load_bound_keypoint_position_source(
        root,
        run_path=_CANARY_RUN_PATH,
        policy=KeypointPositionSourcePolicy(
            profile,
            _BINDING_ID,
            authority_mode=source_mod.KEYPOINT_AUTHORITY_MODE_SEALED_BUNDLE_CANARY,
        ),
    )
    changed = copy(source)
    object.__setattr__(changed, "keypoint_bundle_authority_digest", "c" * 64)
    monkeypatch.setattr(
        source_mod,
        "load_bound_keypoint_position_source",
        lambda *args, **kwargs: changed,
    )

    with pytest.raises(KeypointPositionSourceError, match="authority changed at"):
        revalidate_bound_keypoint_position_source(source)


@pytest.mark.parametrize("field", ["keypoint_labels", "nodes"])
def test_reordered_or_missing_anatomy_labels_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root, profile, _, pose_binding = _fixture(monkeypatch)
    altered = dict(pose_binding["pose_schema"])
    if field == "keypoint_labels":
        altered[field] = list(reversed(altered[field]))
    else:
        altered[field] = altered[field][:-1]
    pose_binding["pose_schema"] = altered

    with pytest.raises(KeypointPositionSourceError, match="source binding disagrees"):
        load_bound_keypoint_position_source(
            root,
            run_path=_RUN_PATH,
            policy=KeypointPositionSourcePolicy(profile, _BINDING_ID),
        )


def test_stale_current_selector_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    root, profile, _, _ = _fixture(monkeypatch)
    root.attrs["current_keypoint_group_path"] = "keypoints_runs/older"

    with pytest.raises(KeypointPositionSourceError, match="current_keypoint_group_path"):
        load_bound_keypoint_position_source(
            root,
            run_path=_RUN_PATH,
            policy=KeypointPositionSourcePolicy(profile, _BINDING_ID),
        )


def test_wrong_coordinate_surface_fails_closed() -> None:
    surfaces = SimpleNamespace(
        keypoints_img=SimpleNamespace(
            descriptor=SimpleNamespace(
                profile_id="roi_local_px.top_left_y_down.v1",
                space_id="roi_local_px",
                geometry_type="point_xy",
                pixel_convention="continuous",
            ),
            reference_frame_authority=object(),
        )
    )

    with pytest.raises(KeypointPositionSourceError, match="source-camera continuous"):
        source_mod._validate_source_camera_surface(surfaces)


def test_missing_explicit_validity_array_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, arrays, _ = _fixture(monkeypatch)
    del root[_RUN_PATH].children["keypoint_valid"]
    arrays.pop("keypoint_valid")

    with pytest.raises(KeypointPositionSourceError, match="keypoint array"):
        load_bound_keypoint_position_source(
            root,
            run_path=_RUN_PATH,
            policy=KeypointPositionSourcePolicy(profile, _BINDING_ID),
        )


def test_revalidation_detects_changed_source_array(monkeypatch: pytest.MonkeyPatch) -> None:
    root, profile, _, _ = _fixture(monkeypatch)
    source = load_bound_keypoint_position_source(
        root,
        run_path=_RUN_PATH,
        policy=KeypointPositionSourcePolicy(profile, _BINDING_ID),
    )
    changed = copy(source)
    image = np.array(source.keypoints_img, copy=True)
    image[0, 0, 0] += np.float32(1.0)
    image.setflags(write=False)
    object.__setattr__(changed, "keypoints_img", image)
    monkeypatch.setattr(source_mod, "load_bound_keypoint_position_source", lambda *args, **kwargs: changed)

    with pytest.raises(KeypointPositionSourceError, match="array changed at keypoints_img"):
        revalidate_bound_keypoint_position_source(source)
