from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.detection import detect_keypoints_traditional as keypoint_mod
from fisheye.detection import detect_traditional as detect_mod
from fisheye.shared.crop_image_source import resolve_materialized_crop_run
from fisheye.shared import detection_producer_lifecycle as lifecycle_mod
from fisheye.shared.detection_producer_lifecycle import (
    ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR,
    DETECTION_ARTIFACT_FAMILY_CONTRACT,
    EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR,
    STRICT_ARTIFACT_INTEGRITY_CONTRACT,
    UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR,
    DetectionProducerAttempt,
    UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
    publish_empty_artifact_observation_proof,
    validate_artifact_payload_inventory_seal,
    validate_unbound_artifact_numeric_semantics,
)
from fisheye.shared.observation_coordinate_publication import (
    load_persisted_detection_observation_geometry,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_FAILED,
    mark_run_complete,
    mark_run_started,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.attrs: dict[str, Any] = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value


class _FakeGroup:
    def __init__(self, children: dict[str, Any] | None = None) -> None:
        self._children: dict[str, Any] = children or {}
        self.attrs: dict[str, Any] = {}
        self.delete_error: BaseException | None = None

    def create_group(self, name: str, **kwargs) -> "_FakeGroup":
        if name in self._children:
            raise ValueError(f"child already exists: {name}")
        child = _FakeGroup()
        child.attrs.update(dict(kwargs.get("attributes") or {}))
        self._children[name] = child
        return child

    def create_array(self, name: str, data, **_kwargs) -> _FakeArray:
        array = _FakeArray(np.asarray(data))
        self._children[name] = array
        return array

    def get(self, name: str) -> Any:
        return self._children.get(name)

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> Any:
        if "/" not in key:
            return self._children[key]
        current: Any = self
        for token in key.split("/"):
            if not isinstance(current, _FakeGroup):
                raise KeyError(key)
            current = current._children[token]
        return current

    def __delitem__(self, key: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        del self._children[key]


class _FailingUpdateAttrs(dict[str, Any]):
    def update(self, *args, **kwargs) -> None:
        incoming = dict(*args, **kwargs)
        if incoming:
            name = next(iter(incoming))
            self[name] = incoming[name]
        raise RuntimeError("injected run attrs update failure")


class _IgnoringSelectorDeleteAttrs(dict[str, Any]):
    def __delitem__(self, key: str) -> None:
        if key == "authoritative_run":
            return
        super().__delitem__(key)


def _begin_unbound_artifact(root, **kwargs):
    return DetectionProducerAttempt.begin_unbound_artifact(
        root,
        semantic_manifest_id="traditional_detection.v1",
        **kwargs,
    )


def test_resolve_materialized_crop_run_prefers_latest_materialized() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_geometry"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"

    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    crop_geometry.attrs["roi_size"] = [4, 4]
    crop_geometry.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_geometry.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs["crop_storage_mode"] = "materialized"
    crop_materialized.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    crop_materialized.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_materialized.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    _parent, _group, run_name = resolve_materialized_crop_run(root)

    assert run_name == "crop_materialized"


def test_resolve_materialized_crop_run_rejects_geometry_only_latest_any() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_geometry"
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    crop_geometry.attrs["roi_size"] = [4, 4]
    crop_geometry.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_geometry.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="geometry-only|materialized crop run"):
        resolve_materialized_crop_run(root)


def test_require_imported_detection_inputs_rejects_missing_images_ds() -> None:
    root = _FakeGroup()
    background_parent = root.create_group("background_runs")
    background_run = background_parent.create_group("background_001")
    background_run.create_array("background_ds", data=np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="raw_video/images_ds"):
        detect_mod._require_imported_detection_inputs(root, "background_001")


def test_require_imported_detection_inputs_rejects_mismatched_hw() -> None:
    root = _FakeGroup()
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((2, 4, 5), dtype=np.uint8))
    background_parent = root.create_group("background_runs")
    background_run = background_parent.create_group("background_001")
    background_run.create_array(
        "background_ds",
        data=np.zeros((4, 4), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="exact images_ds/background_ds H/W"):
        detect_mod._require_imported_detection_inputs(root, "background_001")


def test_detection_attempt_rejects_every_selectable_output_before_creation() -> None:
    root = _FakeGroup()

    with pytest.raises(ValueError, match="artifact-only"):
        DetectionProducerAttempt.begin(
            root,
            run_name="bad",
            output_parent="detection_artifact_runs",
            selector_eligible=True,
            coordinate_contract=UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            stage="detect",
            semantic_manifest_id="traditional_detection.v1",
        )

    assert "detection_artifact_runs" not in root


@pytest.mark.parametrize("selector_eligible", [None, 0, "false"])
def test_detection_attempt_requires_exact_false_selector_marker(
    selector_eligible,
) -> None:
    root = _FakeGroup()

    with pytest.raises(ValueError, match="artifact-only"):
        DetectionProducerAttempt.begin(
            root,
            run_name="bad-marker",
            output_parent="detection_artifact_runs",
            selector_eligible=selector_eligible,
            coordinate_contract=UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            stage="detection_artifact",
            semantic_manifest_id="traditional_detection.v1",
        )

    assert "detection_artifact_runs" not in root


@pytest.mark.parametrize(
    "selector_name",
    [
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ],
)
def test_detection_artifact_parent_rejects_every_selector(selector_name) -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
            selector_name: "forbidden",
        }
    )

    with pytest.raises(ValueError, match="selector-free"):
        _begin_unbound_artifact(
            root,
            run_name="candidate",
        )

    assert "candidate" not in parent


def test_detection_artifact_begin_stamps_family_contract() -> None:
    root = _FakeGroup()

    attempt = _begin_unbound_artifact(
        root,
        run_name="candidate",
    )

    parent = root["detection_artifact_runs"]
    assert (
        parent.attrs["artifact_family_contract"]
        == DETECTION_ARTIFACT_FAMILY_CONTRACT
    )
    assert parent.attrs["stage_selector_eligible"] is False
    assert attempt.run.attrs[lifecycle_mod._PUBLICATION_OWNER_ATTR] == (
        attempt.owner_token
    )
    attempt.fail(RuntimeError("cleanup"))


def test_detection_artifact_rejects_transitional_integrity_bypass() -> None:
    root = _FakeGroup()

    with pytest.raises(ValueError, match="strict_integrity_required=True"):
        _begin_unbound_artifact(
            root,
            run_name="legacy-bypass",
            strict_integrity_required=False,
        )

    assert "detection_artifact_runs" not in root


def test_unbound_artifact_semantics_rejects_unknown_profile() -> None:
    array = _FakeArray(np.empty((0,), dtype=np.int32))

    with pytest.raises(ValueError, match="not registered"):
        lifecycle_mod.stamp_unbound_artifact_numeric_semantics(
            array,
            semantic_profile_id="plausible_but_unregistered_profile.v1",
            reference_node_path="raw_video/images_ds",
            reference_width=4,
            reference_height=4,
            source_frame_count=2,
            source_sha256="0" * 64,
        )

    assert array.attrs == {}


_REGISTERED_PROFILE_FIXED_FIELD_TAMPERS = [
    ("semantic_profile_id", "training.bbox_norm_cxcywh.v1"),
    ("numeric_space_id", "manifest_full_frame_normalized_xy"),
    ("geometry_type", "bbox_xyxy"),
    ("components", ["x_min", "y_min", "x_max", "y_max"]),
    ("component_units", ["px", "px", "px", "px"]),
    ("origin", "not_applicable"),
    ("positive_x_direction", "not_applicable"),
    ("positive_y_direction", "not_applicable"),
    ("pixel_convention", "pixel_edge_half_open"),
    ("axis_0_domain", "dense_frame_rows"),
    ("row_frame_binding_kind", "axis_0_index_equals_temporal_frame_index"),
    ("temporal_domain_id", "training_selected_frame_row_v1"),
    ("reference.kind", "selected_training_frame_array"),
    (
        "source_sha256_kind",
        "canonical_json_artifact_frame_source_lineage_v1",
    ),
    ("source_mapping_sha256_policy", "required"),
    ("dtype", np.dtype("float32").str),
    ("rank", 3),
    ("trailing_shape", [1, 4]),
    ("derivation.operation_id", "ultralytics_xyxy_to_normalized_cxcywh_v1"),
]


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    _REGISTERED_PROFILE_FIXED_FIELD_TAMPERS,
    ids=[path for path, _replacement in _REGISTERED_PROFILE_FIXED_FIELD_TAMPERS],
)
def test_registered_semantic_profile_rejects_every_fixed_field_tamper(
    field_path,
    replacement,
) -> None:
    array = _FakeArray(np.zeros((2, 4), dtype=np.float64))
    lifecycle_mod.stamp_unbound_artifact_numeric_semantics(
        array,
        semantic_profile_id="traditional.bbox_norm_cxcywh.v1",
        reference_node_path="raw_video/images_ds",
        reference_width=4,
        reference_height=4,
        source_frame_count=2,
        source_sha256="0" * 64,
    )
    attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    record = json.loads(json.dumps(array.attrs[attr]))
    target = record
    path = field_path.split(".")
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = replacement
    array.attrs[attr] = record
    array.attrs[f"{attr}_sha256"] = lifecycle_mod._canonical_sha256(record)

    with pytest.raises(ValueError):
        validate_unbound_artifact_numeric_semantics(array)


def test_unbound_numeric_profile_registry_is_immutable_and_axis_explicit() -> None:
    profiles = lifecycle_mod.UNBOUND_NUMERIC_PROFILES
    manifests = lifecycle_mod.UNBOUND_PRODUCER_MANIFESTS

    with pytest.raises(TypeError):
        profiles["forged.v1"] = profiles["traditional.frame_indices.v1"]
    with pytest.raises(AttributeError):
        profiles["traditional.frame_indices.v1"].origin = "top_left"
    with pytest.raises(TypeError):
        manifests["forged.v1"] = manifests["traditional_detection.v1"]
    with pytest.raises(AttributeError):
        manifests["traditional_detection.v1"].producer_family_id = "forged"

    observation = profiles["traditional.bbox_norm_cxcywh.v1"]
    dense_count = profiles["traditional.frame_counts.v1"]
    assert observation.axis_0_domain == "observation_rows"
    assert dense_count.axis_0_domain == "dense_frame_rows"
    assert observation.row_frame_binding_kind != dense_count.row_frame_binding_kind
    assert observation.dtype == np.dtype("float64").str
    assert observation.rank == 2
    assert observation.trailing_shape == (4,)
    assert dense_count.dtype == np.dtype("int32").str
    assert dense_count.rank == 1
    assert dense_count.trailing_shape == ()
    assert observation.source_mapping_sha256_policy == "forbidden"
    assert (
        profiles["training.source_frame_indices.v1"].source_mapping_sha256_policy
        == "required"
    )
    manifest = manifests["traditional_detection.v1"]
    assert dict(manifest.array_profiles)["scores"] == "traditional.scores.v1"
    assert manifest.row_array_names == (
        "artifact_row_id",
        "frame_indices",
        "bbox_norm_coords",
        "scores",
        "class_ids",
    )
    assert manifest.source_mapping_array_names == ()
    assert manifests[
        "training_detection_with_source_mapping.v1"
    ].source_mapping_array_names == ("source_frame_indices",)


@pytest.mark.parametrize(
    ("profile_id", "values"),
    [
        (
            "traditional.scores.v1",
            np.zeros((2,), dtype=np.float64),
        ),
        (
            "traditional.scores.v1",
            np.zeros((2, 1), dtype=np.float32),
        ),
        (
            "traditional.bbox_norm_cxcywh.v1",
            np.zeros((2, 1, 4), dtype=np.float64),
        ),
    ],
    ids=["wrong_dtype", "scalar_wrong_rank", "geometry_wrong_rank"],
)
def test_registered_semantic_profile_rejects_wrong_dtype_or_rank(
    profile_id,
    values,
) -> None:
    array = _FakeArray(values)

    with pytest.raises(ValueError, match="dtype, rank, or trailing shape"):
        lifecycle_mod.stamp_unbound_artifact_numeric_semantics(
            array,
            semantic_profile_id=profile_id,
            reference_node_path="raw_video/images_ds",
            reference_width=4,
            reference_height=4,
            source_frame_count=2,
            source_sha256="0" * 64,
        )

    assert array.attrs == {}


def test_detection_attempt_does_not_delete_concurrent_foreign_child() -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )

    def concurrent_create(name, **_kwargs):
        foreign = _FakeGroup()
        foreign.attrs[lifecycle_mod._PUBLICATION_OWNER_ATTR] = "foreign-owner"
        foreign.attrs["stage_selector_eligible"] = False
        parent._children[name] = foreign
        raise RuntimeError("concurrent name collision")

    parent.create_group = concurrent_create

    with pytest.raises(RuntimeError, match="concurrent name collision"):
        _begin_unbound_artifact(
            root,
            run_name="candidate",
        )

    assert parent["candidate"].attrs[lifecycle_mod._PUBLICATION_OWNER_ATTR] == (
        "foreign-owner"
    )
    assert parent["candidate"].attrs.get(RUN_COMPLETION_STATUS_ATTR) != (
        RUN_STATUS_FAILED
    )


def test_detection_attempt_begin_rolls_back_parent_creation_failure(
    monkeypatch,
) -> None:
    root = _FakeGroup()

    def create_parent_then_fail(target, family):
        parent = target.create_group(family)
        parent.attrs["latest"] = "leaked"
        raise RuntimeError("injected parent creation failure")

    monkeypatch.setattr(
        lifecycle_mod,
        "require_runs_parent",
        create_parent_then_fail,
    )

    with pytest.raises(RuntimeError, match="parent creation failure"):
        _begin_unbound_artifact(root, run_name="candidate")

    assert "detection_artifact_runs" not in root


def test_detection_attempt_begin_restores_existing_parent_when_require_fails(
    monkeypatch,
) -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )

    def mutate_parent_then_fail(_target, _family):
        parent.attrs.update(
            {
                "latest": "leaked",
                "latest_complete": "leaked",
                "latest_pending": "leaked",
                "authoritative_run": "leaked",
                "authoritative_run_provenance": {"leaked": True},
            }
        )
        raise RuntimeError("injected existing parent setup failure")

    monkeypatch.setattr(
        lifecycle_mod,
        "require_runs_parent",
        mutate_parent_then_fail,
    )

    with pytest.raises(RuntimeError, match="existing parent setup failure"):
        _begin_unbound_artifact(root, run_name="candidate")

    for name in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert name not in parent.attrs
    assert "candidate" not in parent


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_detection_attempt_begin_rolls_back_run_creation_failure(exception_type) -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )
    original_create = parent.create_group

    def create_run_then_fail(name, **kwargs):
        original_create(name, **kwargs)
        parent.attrs["latest"] = "leaked"
        raise exception_type("injected run creation failure")

    parent.create_group = create_run_then_fail

    with pytest.raises(exception_type, match="run creation failure"):
        _begin_unbound_artifact(root, run_name="candidate")

    assert "candidate" not in parent
    assert "latest" not in parent.attrs


def test_detection_attempt_begin_rolls_back_mark_started_failure(
    monkeypatch,
) -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )

    def mark_then_fail(run, **_kwargs):
        run.attrs[RUN_COMPLETION_STATUS_ATTR] = "running"
        parent.attrs["latest"] = "leaked"
        raise RuntimeError("injected mark started failure")

    monkeypatch.setattr(lifecycle_mod, "mark_run_started", mark_then_fail)

    with pytest.raises(RuntimeError, match="mark started failure"):
        _begin_unbound_artifact(root, run_name="candidate")

    assert "candidate" not in parent
    assert "latest" not in parent.attrs


def test_detection_attempt_begin_rolls_back_run_attrs_failure() -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )

    def create_with_failing_attrs(name, **kwargs):
        child = _FakeGroup()
        child.attrs = _FailingUpdateAttrs(dict(kwargs.get("attributes") or {}))
        parent._children[name] = child
        return child

    parent.create_group = create_with_failing_attrs

    with pytest.raises(RuntimeError, match="run attrs update failure"):
        _begin_unbound_artifact(root, run_name="candidate")

    assert "candidate" not in parent


def test_detection_attempt_begin_keeps_failed_child_when_delete_unavailable(
    monkeypatch,
) -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )

    parent.delete_error = RuntimeError("injected delete failure")

    def fail_mark_started(_run, **_kwargs):
        parent.attrs["latest"] = "leaked"
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(lifecycle_mod, "mark_run_started", fail_mark_started)

    with pytest.raises(RuntimeError, match="injected setup failure"):
        _begin_unbound_artifact(root, run_name="candidate")

    assert parent["candidate"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert "latest" not in parent.attrs


def test_detection_attempt_begin_reports_unsafe_cleanup_when_fail_and_delete_break(
    monkeypatch,
) -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )
    parent.delete_error = RuntimeError("injected delete failure")

    def fail_mark_started(_run, **_kwargs):
        raise RuntimeError("injected setup failure")

    def fail_mark_failed(*_args, **_kwargs):
        raise RuntimeError("injected failed-marker failure")

    monkeypatch.setattr(lifecycle_mod, "mark_run_started", fail_mark_started)
    monkeypatch.setattr(lifecycle_mod, "mark_run_failed", fail_mark_failed)

    with pytest.raises(RuntimeError, match="could not be rolled back safely"):
        _begin_unbound_artifact(root, run_name="candidate")

    assert "latest" not in parent.attrs
    assert parent["candidate"].attrs.get(RUN_COMPLETION_STATUS_ATTR) != (
        RUN_STATUS_COMPLETE
    )
    assert parent["candidate"].attrs["stage_selector_eligible"] is False
    assert (
        parent["candidate"].attrs["coordinate_contract_mode"]
        == "setup_incomplete_fail_closed"
    )


def test_detection_attempt_rejects_selectable_canonical_branch() -> None:
    root = _FakeGroup()

    with pytest.raises(ValueError, match="staged canonical publisher"):
        DetectionProducerAttempt.begin(
            root,
            run_name="candidate",
            output_parent="detect_runs",
            selector_eligible=True,
            coordinate_contract="canonical_v2",
            stage="detect",
            semantic_manifest_id="traditional_detection.v1",
        )

    assert "detect_runs" not in root


def test_detection_artifact_attempt_failure_restores_exact_selector_state() -> None:
    root = _FakeGroup()
    parent = root.create_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )
    attempt = _begin_unbound_artifact(
        root,
        run_name="failed",
    )
    parent.attrs.update(
        {
            "latest": "leaked",
            "latest_complete": "leaked",
            "latest_pending": "leaked",
            "authoritative_run": "leaked",
            "authoritative_run_provenance": {"leaked": True},
        }
    )

    attempt.fail(RuntimeError("injected"))

    for name in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert name not in parent.attrs
    assert (
        parent["failed"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    )


def test_detection_artifact_attempt_failure_is_idempotent(monkeypatch) -> None:
    root = _FakeGroup()
    attempt = _begin_unbound_artifact(
        root,
        run_name="failed",
    )
    original_mark_failed = lifecycle_mod.mark_run_failed
    failure_calls = 0

    def counting_mark_failed(*args, **kwargs):
        nonlocal failure_calls
        failure_calls += 1
        return original_mark_failed(*args, **kwargs)

    monkeypatch.setattr(lifecycle_mod, "mark_run_failed", counting_mark_failed)

    attempt.fail(KeyboardInterrupt("first interruption"))
    attempt.fail(SystemExit("second interruption"))

    assert failure_calls == 1
    assert (
        root["detection_artifact_runs"]["failed"].attrs[
            RUN_COMPLETION_STATUS_ATTR
        ]
        == RUN_STATUS_FAILED
    )


def test_detection_artifact_attempt_verifies_exact_selector_restoration() -> None:
    root = _FakeGroup()
    attempt = _begin_unbound_artifact(
        root,
        run_name="failed-restore",
    )
    parent = root["detection_artifact_runs"]
    parent.attrs = _IgnoringSelectorDeleteAttrs(parent.attrs)
    parent.attrs["authoritative_run"] = "leaked"

    with pytest.raises(RuntimeError, match="exact selector state"):
        attempt.fail(KeyboardInterrupt("injected"))

    assert parent.attrs["authoritative_run"] == "leaked"


def test_detection_artifact_attempt_rejects_trusted_coordinate_authority() -> None:
    root = _FakeGroup()
    attempt = _begin_unbound_artifact(
        root,
        run_name="invalid-authority",
    )
    attempt.run.attrs["row_identity_contract"] = {
        "schema_id": "palette.observation_row_identity.v1"
    }

    with pytest.raises(ValueError, match="identity or coordinate claims"):
        attempt.complete(run_provenance={})
    attempt.fail(RuntimeError("artifact carried canonical authority"))

    parent = root["detection_artifact_runs"]
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert (
        parent["invalid-authority"].attrs[RUN_COMPLETION_STATUS_ATTR]
        == RUN_STATUS_FAILED
    )


_CANONICAL_ARTIFACT_CLAIM_CASES = (
    tuple(
        ("run_attribute", name)
        for name in sorted(lifecycle_mod._TRUSTED_RUN_ATTRS)
    )
    + tuple(
        ("array_attribute", name)
        for name in sorted(lifecycle_mod._TRUSTED_NODE_ATTRS)
    )
    + tuple(
        ("identity_array", name)
        for name in sorted(lifecycle_mod._IDENTITY_ARRAY_NAMES)
    )
    + (
        ("run_attribute", "instance_key_recording_identity"),
        ("array_attribute", "acquisition_frame_mapping_v2"),
    )
)


@pytest.mark.parametrize(
    ("claim_kind", "claim_name"),
    _CANONICAL_ARTIFACT_CLAIM_CASES,
    ids=lambda value: str(value),
)
def test_detection_artifact_attempt_rejects_every_canonical_claim(
    claim_kind,
    claim_name,
) -> None:
    root = _FakeGroup()
    attempt = _begin_unbound_artifact(
        root,
        run_name="invalid-array-claim",
    )
    if claim_kind == "identity_array":
        attempt.run.create_array(
            claim_name,
            data=np.empty((0,), dtype=np.uint64),
        )
    elif claim_kind == "run_attribute":
        attempt.run.attrs[claim_name] = {"forbidden": True}
    else:
        array = attempt.run.create_array(
            "bbox_norm_coords",
            data=np.empty((0, 4), dtype=np.float64),
        )
        array.attrs[claim_name] = {"forbidden": True}

    with pytest.raises(ValueError, match="identity or coordinate claims"):
        attempt.complete(run_provenance={})
    attempt.fail(RuntimeError("artifact carried an array claim"))


def test_detection_artifact_attempt_rejects_selector_eligibility_tampering() -> None:
    root = _FakeGroup()
    attempt = _begin_unbound_artifact(
        root,
        run_name="tampered",
    )
    attempt.run.attrs["stage_selector_eligible"] = True

    with pytest.raises(ValueError, match="publication invariants"):
        attempt.complete(run_provenance={})
    attempt.fail(RuntimeError("artifact eligibility was tampered"))

    parent = root["detection_artifact_runs"]
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert parent["tampered"].attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert parent["tampered"].attrs["stage_selector_eligible"] is False


def test_detection_artifact_attempt_rejects_stale_zero_observation_proof() -> None:
    root = _FakeGroup()
    attempt = _begin_unbound_artifact(
        root,
        run_name="stale-empty-proof",
    )
    run = attempt.run
    run.create_array("artifact_row_id", data=np.empty((0,), dtype=np.uint64))
    run.create_array("frame_indices", data=np.empty((0,), dtype=np.int32))
    run.create_array("bbox_norm_coords", data=np.empty((0, 4), dtype=np.float64))
    run.create_array("scores", data=np.empty((0,), dtype=np.float32))
    run.create_array("class_ids", data=np.empty((0,), dtype=np.int32))
    run.create_array("frame_counts", data=np.zeros((2,), dtype=np.int32))
    run.create_array("n_detections", data=np.zeros((2,), dtype=np.int32))
    publish_empty_artifact_observation_proof(
        run,
        source_frame_count=2,
        row_array_names=(
            "artifact_row_id",
            "frame_indices",
            "bbox_norm_coords",
            "scores",
            "class_ids",
        ),
        full_domain_evidence={
            "coverage_status": "full_source_domain_validated",
            "source_frame_count": 2,
        },
    )
    run["n_detections"]._data[0] = 1

    with pytest.raises(ValueError, match="count array"):
        attempt.complete(run_provenance={})
    attempt.fail(RuntimeError("stale empty proof"))


def test_traditional_detection_default_fails_before_creating_archive(tmp_path) -> None:
    zarr_path = tmp_path / "must_not_exist.zarr"

    with pytest.raises(ValueError, match="artifact_only=True"):
        detect_mod.detect_fish(str(zarr_path), config_path=None)

    assert not zarr_path.exists()


def _write_traditional_inputs(
    zarr_path,
    *,
    recording_id: str | None = "traditional-recording",
    images_shape: tuple[int, int, int] = (2, 4, 4),
    background_shape: tuple[int, int] = (4, 4),
) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    if recording_id is not None:
        root.attrs["recording_id"] = recording_id
    raw = root.require_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros(images_shape, dtype=np.uint8),
        chunks=(max(1, images_shape[0]), *images_shape[1:]),
    )
    background_parent = root.require_group("background_runs")
    background = background_parent.create_group("background_001")
    background.create_array(
        "background_ds",
        data=np.zeros(background_shape, dtype=np.uint8),
    )
    mark_run_started(background, run_name="background_001", stage="background")
    mark_run_complete(
        background,
        parent_group=background_parent,
        run_name="background_001",
    )


def _patch_traditional_runtime(
    monkeypatch,
    *,
    with_detections: bool,
) -> None:
    worker_result = (
        (
            slice(0, 2),
            [0, 1],
            [
                [0.25, 0.5, 0.2, 0.4],
                [0.75, 0.5, 0.2, 0.4],
            ],
        )
        if with_detections
        else (slice(0, 2), [], [])
    )
    monkeypatch.setattr(
        detect_mod.dask,
        "compute",
        lambda *_tasks: (worker_result,),
    )
    monkeypatch.setattr(
        detect_mod,
        "create_dish_mask",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        detect_mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        detect_mod,
        "get_platform_info",
        lambda **_kwargs: {"hostname": "test", "system": "test"},
    )


def _replace_record_profile(record, profile_id: str) -> None:
    profile = lifecycle_mod.UNBOUND_NUMERIC_PROFILES[profile_id]
    record["semantic_profile_id"] = profile_id
    record.update(lifecycle_mod._profile_fixed_record(profile))
    record["reference"]["kind"] = profile.reference_kind
    record["derivation"]["operation_id"] = profile.derivation_operation_id


def _rewrite_semantics_and_reseal(run, array_name: str, mutator) -> None:
    semantics_attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
    semantics_digest_attr = f"{semantics_attr}_sha256"
    record = json.loads(json.dumps(run[array_name].attrs[semantics_attr]))
    mutator(record)
    semantics_digest = lifecycle_mod._canonical_sha256(record)
    run[array_name].attrs[semantics_attr] = record
    run[array_name].attrs[semantics_digest_attr] = semantics_digest

    seal = json.loads(json.dumps(run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR]))
    seal_array = seal["arrays"][array_name]
    seal_array["numeric_semantics_sha256"] = semantics_digest
    seal_array["semantic_profile_id"] = record["semantic_profile_id"]
    seal_array["numeric_space_id"] = record["numeric_space_id"]
    run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR] = seal
    run.attrs[f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256"] = (
        lifecycle_mod._canonical_sha256(seal)
    )


def test_traditional_detection_rejects_mismatched_hw_before_compute(
    monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / "mismatched.zarr"
    _write_traditional_inputs(zarr_path, background_shape=(3, 4))
    compute_called = False

    def forbidden_compute(*_args, **_kwargs):
        nonlocal compute_called
        compute_called = True
        raise AssertionError("Dask compute must not run")

    monkeypatch.setattr(detect_mod.dask, "compute", forbidden_compute)

    with pytest.raises(ValueError, match="exact images_ds/background_ds H/W"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    assert compute_called is False
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detection_artifact_runs" not in reopened


def test_traditional_detection_rejects_zero_frame_source_before_compute(
    monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / "zero-frames.zarr"
    _write_traditional_inputs(zarr_path, images_shape=(0, 4, 4))
    compute_called = False

    def forbidden_compute(*_args, **_kwargs):
        nonlocal compute_called
        compute_called = True
        raise AssertionError("Dask compute must not run")

    monkeypatch.setattr(detect_mod.dask, "compute", forbidden_compute)

    with pytest.raises(ValueError, match="at least one source frame"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    assert compute_called is False
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detection_artifact_runs" not in reopened


@pytest.mark.parametrize(
    ("changed_call", "match"),
    [
        (3, "raw_video/images_ds changed"),
        (4, "background_runs/background_001/background_ds changed"),
    ],
)
def test_traditional_detection_rejects_source_fingerprint_change_before_output(
    changed_call, match, monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / "changed-source.zarr"
    _write_traditional_inputs(zarr_path)
    fingerprint_calls = 0

    def changing_fingerprint(_array):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return ("b" if fingerprint_calls == changed_call else "a") * 64

    monkeypatch.setattr(
        detect_mod,
        "_array_content_fingerprint",
        changing_fingerprint,
    )
    monkeypatch.setattr(
        detect_mod.dask,
        "compute",
        lambda *_tasks: ((slice(0, 2), [], []),),
    )
    monkeypatch.setattr(
        detect_mod,
        "create_dish_mask",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match=match):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detection_artifact_runs" not in reopened


@pytest.mark.parametrize(
    ("worker_results", "match"),
    [
        ((), "exactly one worker result"),
        (((slice(0, 1), [], []),), "wrong source slice"),
        (
            ((slice(0, 2), [2], [[0.5, 0.5, 0.2, 0.2]]),),
            "outside its exact source slice/domain",
        ),
        (
            ((slice(0, 2), [0.0], [[0.5, 0.5, 0.2, 0.2]]),),
            "frame indices must be exact integers",
        ),
        (
            ((slice(0, 2), [0], np.empty((0, 4), dtype=np.float64)),),
            "bbox/frame cardinality",
        ),
        (
            ((slice(0, 2), [0], [[np.nan, 0.5, 0.2, 0.2]]),),
            "non-finite normalized bbox",
        ),
        (
            ((slice(0, 2), [0], [[0.5, 0.5, 0.0, 0.2]]),),
            "invalid normalized bbox extent",
        ),
        (
            ((slice(0, 2), [0], [[0.1, 0.5, 0.4, 0.2]]),),
            "invalid normalized bbox extent",
        ),
        (
            (
                (
                    slice(0, 2),
                    [1, 0],
                    [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
                ),
            ),
            "not ordered by source frame",
        ),
    ],
)
def test_traditional_detection_rejects_malformed_worker_results_before_output(
    worker_results, match, monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / "malformed-worker-result.zarr"
    _write_traditional_inputs(zarr_path)
    monkeypatch.setattr(
        detect_mod.dask,
        "compute",
        lambda *_tasks: worker_results,
    )
    monkeypatch.setattr(
        detect_mod,
        "create_dish_mask",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match=match):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detection_artifact_runs" not in reopened


def test_resolve_traditional_crop_background_inputs_rejects_missing_images_full() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_materialized"] = "crop_001"
    crop_run = crop_parent.create_group("crop_001")
    crop_run.attrs["crop_storage_mode"] = "materialized"
    crop_run.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    crop_run.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_run.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    background_parent = root.create_group("background_runs")
    background_parent.attrs["latest"] = "background_001"
    background_run = background_parent.create_group("background_001")
    background_run.create_array("background_full", data=np.zeros((6, 6), dtype=np.uint8))

    with pytest.raises(ValueError, match="raw_video/images_full"):
        keypoint_mod._resolve_traditional_crop_background_inputs(root)


def test_traditional_detection_isolated_as_nonselector_artifact(
    monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / "traditional.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "traditional-recording"
    raw = root.require_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.zeros((2, 4, 4), dtype=np.uint8),
        chunks=(2, 4, 4),
    )
    background_parent = root.require_group("background_runs")
    background = background_parent.create_group("background_001")
    background.create_array(
        "background_ds",
        data=np.zeros((4, 4), dtype=np.uint8),
    )
    mark_run_started(background, run_name="background_001", stage="background")
    mark_run_complete(
        background,
        parent_group=background_parent,
        run_name="background_001",
    )

    monkeypatch.setattr(
        detect_mod.dask,
        "compute",
        lambda *_tasks: (
            (
                slice(0, 2),
                [0],
                [[0.25, 0.5, 0.2, 0.4]],
            ),
        ),
    )
    monkeypatch.setattr(detect_mod, "create_dish_mask", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        detect_mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        detect_mod,
        "get_platform_info",
        lambda **_kwargs: {"hostname": "test", "system": "test"},
    )

    result = detect_mod.detect_fish(
        str(zarr_path),
        config_path=None,
        show_progress=False,
        artifact_only=True,
    )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_runs" not in reopened
    parent = reopened["detection_artifact_runs"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    assert (
        parent.attrs["artifact_family_contract"]
        == DETECTION_ARTIFACT_FAMILY_CONTRACT
    )
    assert parent.attrs["stage_selector_eligible"] is False
    run = parent[result["run_name"]]
    assert result["run_path"] == f"detection_artifact_runs/{result['run_name']}"
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert run.attrs["stage_selector_eligible"] is False
    assert (
        run.attrs["coordinate_contract"]
        == UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT
    )
    assert (
        run.attrs["artifact_integrity_contract"]
        == STRICT_ARTIFACT_INTEGRITY_CONTRACT
    )
    assert (
        run.attrs[lifecycle_mod.UNBOUND_NUMERIC_MANIFEST_ATTR]
        == "traditional_detection.v1"
    )
    assert run["artifact_row_id"].dtype == np.dtype("uint64")
    assert run["artifact_row_id"][:].tolist() == [0]
    expected_profiles = {
        "artifact_row_id": "traditional.artifact_row_id.v1",
        "frame_indices": "traditional.frame_indices.v1",
        "bbox_norm_coords": "traditional.bbox_norm_cxcywh.v1",
        "scores": "traditional.scores.v1",
        "class_ids": "traditional.class_ids.v1",
        "frame_counts": "traditional.frame_counts.v1",
        "n_detections": "traditional.n_detections.v1",
    }
    for name in run.keys():
        semantics = validate_unbound_artifact_numeric_semantics(run[name])
        assert semantics["canonical_binding_status"] == "unbound"
        assert semantics["semantic_profile_id"] == expected_profiles[name]
        assert semantics["reference"] == {
            "kind": "raw_video_images_ds_array",
            "node_path": "raw_video/images_ds",
            "width": 4,
            "height": 4,
        }
    bbox_semantics = validate_unbound_artifact_numeric_semantics(
        run["bbox_norm_coords"]
    )
    assert bbox_semantics["component_units"] == ["normalized"] * 4
    assert bbox_semantics["pixel_convention"] == "continuous"
    assert bbox_semantics["derivation"]["operation_id"] == (
        "skimage_regionprops_max_exclusive_bbox_to_normalized_cxcywh_v1"
    )
    assert bbox_semantics["axis_0_domain"] == "observation_rows"
    assert (
        validate_unbound_artifact_numeric_semantics(run["frame_counts"])[
            "axis_0_domain"
        ]
        == "dense_frame_rows"
    )
    payload_seal = validate_artifact_payload_inventory_seal(run)
    assert payload_seal["row_count"] == 1
    assert set(payload_seal["arrays"]) == set(run.keys())
    assert payload_seal["unbound_numeric_manifest_id"] == (
        "traditional_detection.v1"
    )
    assert {
        name: evidence["semantic_profile_id"]
        for name, evidence in payload_seal["arrays"].items()
    } == expected_profiles
    assert ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR in run.attrs
    assert "instance_key" not in run
    assert not any(name.startswith("instance_key_") for name in run.attrs)
    assert len(run.attrs["source_images_ds_content_sha256"]) == 64
    assert len(run.attrs["source_background_ds_content_sha256"]) == 64
    with pytest.raises(ValueError, match="exact detect_runs/<run> rowset"):
        load_persisted_detection_observation_geometry(
            reopened,
            result["run_path"],
        )


def test_traditional_detection_allows_zero_detections_from_nonempty_source(
    monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / "zero-detections.zarr"
    _write_traditional_inputs(zarr_path, recording_id=None)
    monkeypatch.setattr(
        detect_mod.dask,
        "compute",
        lambda *_tasks: ((slice(0, 2), [], []),),
    )
    monkeypatch.setattr(
        detect_mod,
        "create_dish_mask",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        detect_mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        detect_mod,
        "get_platform_info",
        lambda **_kwargs: {"hostname": "test", "system": "test"},
    )

    result = detect_mod.detect_fish(
        str(zarr_path),
        config_path=None,
        show_progress=False,
        artifact_only=True,
    )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = reopened["detection_artifact_runs"][result["run_name"]]
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["total_frames"] == 2
    assert result["total_detections"] == 0
    assert run.attrs["summary_statistics"]["total_detections"] == 0
    assert run["artifact_row_id"].dtype == np.dtype("uint64")
    assert run["artifact_row_id"].shape == (0,)
    assert run["bbox_norm_coords"].shape == (0, 4)
    assert run["frame_counts"][:].tolist() == [0, 0]
    assert run["n_detections"][:].tolist() == [0, 0]
    assert run["frame_counts"].dtype == np.dtype("int32")
    proof = run.attrs[EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR]
    assert proof["status"] == "verified_no_observations"
    assert proof["source_frame_count"] == 2
    assert proof["full_domain_evidence"]["worker_slice_plan"] == [[0, 2]]
    assert proof["full_domain_evidence"]["validated_worker_result_count"] == 1
    assert set(proof["array_inventory"]) == set(run.keys())
    assert set(proof["row_arrays"]) == {
        "artifact_row_id",
        "frame_indices",
        "bbox_norm_coords",
        "scores",
        "class_ids",
    }
    assert len(run.attrs[f"{EMPTY_ARTIFACT_OBSERVATION_PROOF_ATTR}_sha256"]) == 64
    lifecycle_mod.validate_empty_artifact_observation_proof(run)
    assert validate_artifact_payload_inventory_seal(run)["row_count"] == 0


@pytest.mark.parametrize(
    "tamper_case",
    [
        "missing_semantics",
        "tampered_semantics",
        "wrong_reference_extent",
        "artifact_row_id",
        "array_payload",
        "row_cardinality",
        "frame_counts",
    ],
)
def test_traditional_detection_rejects_post_seal_artifact_drift(
    tamper_case,
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / f"tampered-{tamper_case}.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=True)
    original_write = detect_mod.write_stage_provenance

    def write_then_tamper(run, provenance):
        original_write(run, provenance)
        if tamper_case == "missing_semantics":
            del run["bbox_norm_coords"].attrs[
                UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
            ]
        elif tamper_case == "tampered_semantics":
            attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
            record = json.loads(json.dumps(run["bbox_norm_coords"].attrs[attr]))
            record["numeric_space_id"] = "forged_coordinate_space"
            run["bbox_norm_coords"].attrs[attr] = record
        elif tamper_case == "wrong_reference_extent":
            attr = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
            digest_attr = f"{attr}_sha256"
            record = json.loads(json.dumps(run["bbox_norm_coords"].attrs[attr]))
            record["reference"]["width"] += 1
            run["bbox_norm_coords"].attrs[attr] = record
            run["bbox_norm_coords"].attrs[digest_attr] = (
                lifecycle_mod._canonical_sha256(record)
            )
        elif tamper_case == "artifact_row_id":
            run["artifact_row_id"][0] = np.uint64(7)
        elif tamper_case == "array_payload":
            run["scores"][0] = np.float32(0.125)
        elif tamper_case == "row_cardinality":
            old = run["scores"]
            attrs = dict(old.attrs)
            values = np.append(old[:], np.float32(0.5))
            del run["scores"]
            replacement = run.create_array(
                "scores",
                data=values,
                chunks=(values.shape[0],),
            )
            replacement.attrs.update(attrs)
        else:
            run["frame_counts"][0] = np.int32(2)

    monkeypatch.setattr(detect_mod, "write_stage_provenance", write_then_tamper)

    with pytest.raises(ValueError):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    assert len(list(parent.group_keys())) == 1
    failed = parent[next(iter(parent.group_keys()))]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize(
    "tamper_case",
    [
        "same_family_role_profile",
        "cross_family_profile",
        "reference_extent",
        "source_frame_count",
        "source_sha256",
        "source_mapping_sha256",
    ],
)
def test_traditional_manifest_rejects_recomputed_semantics_and_seal_bypass(
    tamper_case,
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / f"recomputed-{tamper_case}.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=True)
    original_write = detect_mod.write_stage_provenance

    def write_then_recompute(run, provenance):
        original_write(run, provenance)
        array_name = (
            "scores" if tamper_case == "same_family_role_profile" else "bbox_norm_coords"
        )

        def mutate(record):
            if tamper_case == "same_family_role_profile":
                _replace_record_profile(record, "traditional.class_ids.v1")
            elif tamper_case == "cross_family_profile":
                _replace_record_profile(record, "training.bbox_norm_cxcywh.v1")
            elif tamper_case == "reference_extent":
                record["reference"]["width"] += 1
            elif tamper_case == "source_frame_count":
                record["temporal_evidence"]["source_frame_count"] += 1
            elif tamper_case == "source_sha256":
                record["source_sha256"] = "1" * 64
            else:
                record["temporal_evidence"]["source_mapping_sha256"] = "2" * 64

        _rewrite_semantics_and_reseal(run, array_name, mutate)

    monkeypatch.setattr(detect_mod, "write_stage_provenance", write_then_recompute)

    with pytest.raises(ValueError):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    failed = parent[next(iter(parent.group_keys()))]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_traditional_attempt_rejects_recomputed_cross_family_manifest_swap(
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / "cross-family-manifest-swap.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=True)
    original_write = detect_mod.write_stage_provenance

    def write_then_swap_manifest(run, provenance):
        original_write(run, provenance)
        manifest_id = "training_detection_without_source_mapping.v1"
        manifest = lifecycle_mod.UNBOUND_PRODUCER_MANIFESTS[manifest_id]
        run.attrs[lifecycle_mod.UNBOUND_NUMERIC_MANIFEST_ATTR] = manifest_id
        manifest_digest = lifecycle_mod._canonical_sha256(
            lifecycle_mod._manifest_record(manifest_id, manifest)
        )
        run.attrs[lifecycle_mod.UNBOUND_NUMERIC_MANIFEST_DIGEST_ATTR] = (
            manifest_digest
        )
        for array_name, profile_id in manifest.array_profiles:
            _rewrite_semantics_and_reseal(
                run,
                array_name,
                lambda record, target=profile_id: _replace_record_profile(
                    record,
                    target,
                ),
            )
        seal = json.loads(
            json.dumps(run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR])
        )
        seal["unbound_numeric_manifest_id"] = manifest_id
        seal["unbound_numeric_manifest_sha256"] = manifest_digest
        run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR] = seal
        run.attrs[f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256"] = (
            lifecycle_mod._canonical_sha256(seal)
        )

    monkeypatch.setattr(
        detect_mod,
        "write_stage_provenance",
        write_then_swap_manifest,
    )

    with pytest.raises(ValueError, match="publication invariants"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    failed = parent[next(iter(parent.group_keys()))]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_traditional_rejects_coherent_all_array_source_reference_rewrite(
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / "coherent-source-reference-rewrite.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=True)
    original_write = detect_mod.write_stage_provenance

    def write_then_rewrite_all_semantics(run, provenance):
        original_write(run, provenance)
        seal = json.loads(
            json.dumps(run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR])
        )
        attr_name = UNBOUND_ARTIFACT_NUMERIC_SEMANTICS_ATTR
        digest_name = f"{attr_name}_sha256"
        for array_name in run.keys():
            record = json.loads(json.dumps(run[array_name].attrs[attr_name]))
            record["reference"]["width"] += 1
            record["source_sha256"] = "f" * 64
            semantics_digest = lifecycle_mod._canonical_sha256(record)
            run[array_name].attrs[attr_name] = record
            run[array_name].attrs[digest_name] = semantics_digest
            seal["arrays"][array_name][
                "numeric_semantics_sha256"
            ] = semantics_digest
        run.attrs[ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR] = seal
        run.attrs[f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256"] = (
            lifecycle_mod._canonical_sha256(seal)
        )

    monkeypatch.setattr(
        detect_mod,
        "write_stage_provenance",
        write_then_rewrite_all_semantics,
    )

    with pytest.raises(ValueError, match="run-owned reference, source"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = reopened["detection_artifact_runs"][
        next(iter(reopened["detection_artifact_runs"].group_keys()))
    ]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


def test_traditional_detection_rejects_zero_row_identity_tampering(
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / "tampered-zero-row-id.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=False)
    original_write = detect_mod.write_stage_provenance

    def write_then_replace_zero_row_id(run, provenance):
        original_write(run, provenance)
        del run["artifact_row_id"]
        run.create_array(
            "artifact_row_id",
            data=np.empty((0,), dtype=np.int64),
            chunks=(1,),
        )

    monkeypatch.setattr(
        detect_mod,
        "write_stage_provenance",
        write_then_replace_zero_row_id,
    )

    with pytest.raises(ValueError):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    failed = parent[next(iter(parent.group_keys()))]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize("tamper_case", ["payload", "publication_invariant"])
def test_traditional_detection_revalidates_fresh_child_after_mark_complete(
    tamper_case,
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / f"post-completion-{tamper_case}.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=True)
    original_mark_complete = lifecycle_mod.mark_run_complete

    def mark_complete_then_tamper(run, **kwargs):
        original_mark_complete(run, **kwargs)
        if tamper_case == "payload":
            run["scores"][0] = np.float32(0.375)
        else:
            run.attrs["coordinate_contract_mode"] = "forged_after_completion"

    monkeypatch.setattr(
        lifecycle_mod,
        "mark_run_complete",
        mark_complete_then_tamper,
    )

    with pytest.raises(ValueError, match="changed after sealing|publication invariants"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    failed = parent[next(iter(parent.group_keys()))]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_traditional_detection_base_exception_during_final_integrity_validation(
    exception_type,
    monkeypatch,
    tmp_path,
) -> None:
    zarr_path = tmp_path / f"final-{exception_type.__name__}.zarr"
    _write_traditional_inputs(zarr_path)
    _patch_traditional_runtime(monkeypatch, with_detections=True)
    original_validate = lifecycle_mod.validate_artifact_payload_inventory_seal
    validation_calls = 0

    def interrupt_completed_child_validation(run):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 3:
            raise exception_type("injected final artifact validation interruption")
        return original_validate(run)

    monkeypatch.setattr(
        lifecycle_mod,
        "validate_artifact_payload_inventory_seal",
        interrupt_completed_child_validation,
    )

    with pytest.raises(exception_type, match="final artifact validation"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    assert validation_calls == 3
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    failed = parent[next(iter(parent.group_keys()))]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_traditional_detection_base_exception_after_attempt_fails_closed(
    exception_type, monkeypatch, tmp_path
) -> None:
    zarr_path = tmp_path / f"interrupted-{exception_type.__name__}.zarr"
    _write_traditional_inputs(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    parent = root.require_group("detection_artifact_runs")
    parent.attrs.update(
        {
            "artifact_family_contract": DETECTION_ARTIFACT_FAMILY_CONTRACT,
            "stage_selector_eligible": False,
        }
    )
    monkeypatch.setattr(
        detect_mod.dask,
        "compute",
        lambda *_tasks: ((slice(0, 2), [], []),),
    )
    monkeypatch.setattr(
        detect_mod,
        "create_dish_mask",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        detect_mod,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "short_hash": "abc123",
            "branch": "test",
            "is_dirty": False,
            "remote_url": None,
        },
    )
    monkeypatch.setattr(
        detect_mod,
        "get_platform_info",
        lambda **_kwargs: {"hostname": "test", "system": "test"},
    )

    def interrupt_after_attempt(*_args, **_kwargs):
        raise exception_type("injected post-attempt interruption")

    monkeypatch.setattr(
        detect_mod,
        "publish_empty_artifact_observation_proof",
        interrupt_after_attempt,
    )

    with pytest.raises(exception_type, match="post-attempt interruption"):
        detect_mod.detect_fish(
            str(zarr_path),
            config_path=None,
            show_progress=False,
            artifact_only=True,
        )

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = reopened["detection_artifact_runs"]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "authoritative_run_provenance",
    ):
        assert selector not in parent.attrs
    child_names = list(parent.group_keys())
    assert len(child_names) == 1
    failed = parent[child_names[0]]
    assert failed.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_FAILED
    assert failed.attrs["stage_selector_eligible"] is False
