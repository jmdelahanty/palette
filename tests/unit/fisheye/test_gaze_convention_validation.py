from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis.gaze_convention_validation as validation_module
from fisheye.analysis.gaze_convention_validation import (
    _resolve_eye_run,
    _resolve_review_geometry,
    _resolve_review_masks,
    _resolve_review_roi_offsets,
    body_frame_angles_from_vectors,
    validate_gaze_geometry_arrays,
    wrap_degrees_signed,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _synthetic_geometry() -> dict[str, np.ndarray]:
    heading = np.asarray([0.0, 90.0, -90.0, 180.0], dtype=np.float64)
    radians = np.deg2rad(heading)
    forward = np.column_stack((np.cos(radians), -np.sin(radians)))
    left = np.column_stack((forward[:, 1], -forward[:, 0]))
    left_major = np.asarray([-20.0, -10.0, 5.0, 15.0])
    right_major = np.asarray([20.0, 12.0, -4.0, -14.0])
    left_gaze = wrap_degrees_signed(left_major + 90.0)
    right_gaze = wrap_degrees_signed(right_major - 90.0)

    def vectors(angles: np.ndarray) -> np.ndarray:
        rad = np.deg2rad(angles)
        return np.cos(rad)[:, None] * forward + np.sin(rad)[:, None] * left

    return {
        "left_major_signed_deg": left_major,
        "right_major_signed_deg": right_major,
        "left_eye_angle_deg": -left_major,
        "right_eye_angle_deg": right_major,
        "vergence_eye_angle_deg": -left_major + right_major,
        "left_gaze_signed_deg": left_gaze,
        "right_gaze_signed_deg": right_gaze,
        "left_gaze_xy": vectors(left_gaze),
        "right_gaze_xy": vectors(right_gaze),
        "forward_axis_xy": forward,
        "left_axis_xy": left,
        "heading_deg": heading,
        "valid": np.ones(heading.shape, dtype=bool),
    }


def test_gaze_geometry_gate_accepts_canonical_conventions() -> None:
    checks = validate_gaze_geometry_arrays(**_synthetic_geometry())
    assert checks
    assert all(check.passed for check in checks)


def test_gaze_geometry_gate_rejects_left_eye_sign_inversion() -> None:
    geometry = _synthetic_geometry()
    geometry["left_eye_angle_deg"] = geometry["left_major_signed_deg"].copy()
    checks = {check.name: check for check in validate_gaze_geometry_arrays(**geometry)}
    assert not checks["left_eye_angle_nasal_sign"].passed
    assert checks["right_eye_angle_nasal_sign"].passed


def test_gaze_geometry_gate_rejects_left_right_gaze_swap() -> None:
    geometry = _synthetic_geometry()
    geometry["left_gaze_xy"], geometry["right_gaze_xy"] = (
        geometry["right_gaze_xy"].copy(),
        geometry["left_gaze_xy"].copy(),
    )
    checks = {check.name: check for check in validate_gaze_geometry_arrays(**geometry)}
    assert not checks["left_gaze_vector_body_angle"].passed
    assert not checks["right_gaze_vector_body_angle"].passed


def test_body_frame_vector_angles_use_anatomical_left_positive() -> None:
    forward = np.asarray([[1.0, 0.0]] * 3)
    left = np.asarray([[0.0, -1.0]] * 3)
    vectors = np.asarray([[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        body_frame_angles_from_vectors(vectors, forward, left),
        np.asarray([0.0, 90.0, -90.0]),
    )


def test_review_masks_follow_subject_shape_refined_subject_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = np.zeros((5, 2, 8, 8), dtype=np.uint8)
    subject_shape = SimpleNamespace(
        masks_roi=None,
        ellipse_params=np.zeros((5, 2, 5), dtype=np.float32),
        group_path="analysis/subject_shape_runs/shape_a",
        source_refined_subject_run="refined_subject_a",
    )
    refined_subject = SimpleNamespace(
        masks_roi=masks,
        group_path="refined_subject_masks_runs/refined_subject_a",
    )
    calls: list[str] = []

    def _fake_resolve(_root: object, *, refined_subject_run: str):
        calls.append(refined_subject_run)
        return refined_subject

    monkeypatch.setattr(
        validation_module,
        "resolve_eye_geometry_source",
        _fake_resolve,
    )

    resolved, source_path = _resolve_review_masks(object(), subject_shape)

    assert resolved is masks
    assert source_path == "refined_subject_masks_runs/refined_subject_a"
    assert calls == ["refined_subject_a"]


def test_review_masks_follow_sealed_subject_shape_bundle_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = np.zeros((5, 4, 8, 8), dtype=np.uint8)
    dense[:, 1, :, :4] = 1
    dense[:, 2, :, 4:] = 1

    class _FakeBundleSource:
        authority = SimpleNamespace(
            refined_run=object(),
            refined_run_path="refined_subject_masks_runs/refined_subject_a",
        )

    class _FakeMaskStore:
        shape = dense.shape
        storage_path = "refined_subject_masks_runs/refined_subject_a/masks_roi"

        @staticmethod
        def component_index(component: str) -> int:
            return {"subject_body": 0, "eye_left": 1, "eye_right": 2, "swim_bladder": 3}[
                component
            ]

        @staticmethod
        def read_dense(rows=None, channels=None):
            row_indices = np.arange(dense.shape[0])[rows]
            if np.isscalar(row_indices):
                row_indices = np.asarray([row_indices])
            channel_indices = np.asarray(channels, dtype=np.int64)
            return dense[np.ix_(np.asarray(row_indices), channel_indices)]

    bundle = _FakeBundleSource()
    geometry = SimpleNamespace(
        masks_roi=None,
        ellipse_params=np.zeros((5, 2, 5), dtype=np.float32),
        group_path="analysis/subject_shape_runs/shape_a",
        source_refined_subject_run="refined_subject_a",
        subject_shape_coordinate_publication=SimpleNamespace(source=bundle),
    )
    monkeypatch.setattr(
        validation_module,
        "BoundSubjectShapeBundleSource",
        _FakeBundleSource,
    )
    monkeypatch.setattr(
        validation_module,
        "require_bound_subject_shape_bundle_source",
        lambda source: source,
    )
    monkeypatch.setattr(
        validation_module,
        "open_mask_store",
        lambda *_args, **_kwargs: _FakeMaskStore(),
    )
    monkeypatch.setattr(
        validation_module,
        "resolve_eye_geometry_source",
        lambda *_args, **_kwargs: pytest.fail(
            "sealed bundle masks fell through to the canonical refined-mask reader"
        ),
    )

    resolved, source_path = _resolve_review_masks(object(), geometry)

    assert source_path == "refined_subject_masks_runs/refined_subject_a/masks_roi"
    assert resolved.shape == (5, 2, 8, 8)
    np.testing.assert_array_equal(np.asarray(resolved[0, :, 0, 0]), [1, 0])
    np.testing.assert_array_equal(np.asarray(resolved[0, :, 0, 7]), [0, 1])


def test_review_masks_reject_row_mismatch() -> None:
    geometry = SimpleNamespace(
        masks_roi=np.zeros((4, 2, 8, 8), dtype=np.uint8),
        ellipse_params=np.zeros((5, 2, 5), dtype=np.float32),
        group_path="analysis/subject_shape_runs/shape_a",
        source_refined_subject_run="refined_subject_a",
    )

    with pytest.raises(ValueError, match="not row-aligned"):
        _resolve_review_masks(object(), geometry)


def test_review_roi_offsets_use_exact_bound_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuous = np.asarray(
        [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]],
        dtype=np.float64,
    )
    geometry = SimpleNamespace(
        stage_group=validation_module.EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
        ellipse_params=np.zeros((3, 2, 5), dtype=np.float32),
        subject_shape_coordinate_publication=SimpleNamespace(source=object()),
    )
    monkeypatch.setattr(
        validation_module,
        "require_translation_only_subject_shape_placement",
        lambda _source: (continuous, continuous.copy()),
    )

    np.testing.assert_array_equal(
        _resolve_review_roi_offsets(geometry),
        continuous,
    )


def test_review_roi_offsets_reject_nonidentical_edge_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuous = np.zeros((3, 2), dtype=np.float64)
    edge = continuous.copy()
    edge[1, 0] = 0.5
    geometry = SimpleNamespace(
        stage_group=validation_module.EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
        ellipse_params=np.zeros((3, 2, 5), dtype=np.float32),
        subject_shape_coordinate_publication=SimpleNamespace(source=object()),
    )
    monkeypatch.setattr(
        validation_module,
        "require_translation_only_subject_shape_placement",
        lambda _source: (continuous, edge),
    )

    with pytest.raises(ValueError, match="translation-only placement"):
        _resolve_review_roi_offsets(geometry)


def _candidate_review_attrs() -> tuple[dict[str, object], dict[str, object]]:
    run_name = "shape_candidate"
    admission: dict[str, object] = {
        "source_subject_shape_run": run_name,
        "normal_reader_authority": False,
        "selector_activation": False,
        "record_sha256": "a" * 64,
    }
    authority_body: dict[str, object] = {
        "schema_id": validation_module.EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID,
        "schema_version": validation_module.EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": validation_module.EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_CANDIDATE_AUTHORITY_SCOPE,
        "source_subject_shape_run": run_name,
        "source_subject_shape_run_ref": f"/analysis/subject_shape_runs/{run_name}",
        "row_count": 5,
        "canonical_publication": {},
        "source_contract_attrs": {},
        "allowed_arrays": {},
        "closed_array_inventory": True,
        "normal_reader_authority": False,
        "candidate_admission": admission,
    }
    authority = {
        **authority_body,
        "record_sha256": canonical_json_sha256(authority_body),
    }
    return (
        {
            "source_eye_geometry_stage": "analysis/subject_shape_runs",
            "source_eye_geometry_run": run_name,
            "source_eye_geometry_authority_mode": "digest_bound_staged_subset",
            "eye_angle_source_contracts": {
                "eye_geometry": {
                    "stage_group": "analysis/subject_shape_runs",
                    "run_name": run_name,
                    "source_authority": authority,
                }
            },
        },
        admission,
    )


def test_candidate_review_geometry_reuses_exact_sealed_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attrs, admission = _candidate_review_attrs()
    resolved = object()
    calls: list[dict[str, object]] = []

    def _fake_resolve(_root: object, **kwargs: object):
        calls.append(kwargs)
        return resolved

    monkeypatch.setattr(validation_module, "resolve_eye_geometry_source", _fake_resolve)

    assert (
        _resolve_review_geometry(
            object(),
            attrs,
            allow_ineligible_candidate=True,
        )
        is resolved
    )
    assert calls == [
        {
            "subject_shape_run": "shape_candidate",
            "_completed_ineligible_subject_shape_candidate": admission,
        }
    ]


def test_candidate_review_geometry_rejects_stale_embedded_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attrs, _admission = _candidate_review_attrs()
    authority = attrs["eye_angle_source_contracts"]["eye_geometry"]["source_authority"]
    authority["record_sha256"] = "b" * 64
    monkeypatch.setattr(
        validation_module,
        "resolve_eye_geometry_source",
        lambda *_args, **_kwargs: pytest.fail("stale authority reached resolver"),
    )

    with pytest.raises(ValueError, match="invalid or stale"):
        _resolve_review_geometry(
            object(),
            attrs,
            allow_ineligible_candidate=True,
        )


def test_normal_review_geometry_does_not_admit_candidate_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attrs, _admission = _candidate_review_attrs()
    resolved = object()
    calls: list[dict[str, object]] = []

    def _fake_resolve(_root: object, **kwargs: object):
        calls.append(kwargs)
        return resolved

    monkeypatch.setattr(validation_module, "resolve_eye_geometry_source", _fake_resolve)

    assert _resolve_review_geometry(object(), attrs) is resolved
    assert calls == [{"subject_shape_run": "shape_candidate"}]


def test_eye_run_resolution_uses_canonical_eye_angle_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_group = object()
    calls: list[tuple[object, object, bool]] = []

    def _fake_resolve(root, run_name, *, legacy_compatibility):
        calls.append((root, run_name, legacy_compatibility))
        return run_group, "eye_current", "analysis/eye_angle_runs/eye_current"

    monkeypatch.setattr(validation_module, "resolve_eye_angle_run", _fake_resolve)
    root = object()

    resolved, group = _resolve_eye_run(
        root,
        "analysis/eye_angle_runs/eye_current",
    )

    assert resolved == "eye_current"
    assert group is run_group
    assert calls == [
        (root, "analysis/eye_angle_runs/eye_current", False)
    ]


def test_eye_run_resolution_accepts_only_exact_storage_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_group = SimpleNamespace(attrs={"stage_selector_eligible": False})
    parent = {"eye_candidate": run_group}
    root = {"analysis/eye_angle_runs": parent}
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        validation_module,
        "is_run_complete_in_parent",
        lambda observed_parent, observed_run, *, legacy_default: (
            observed_parent is parent
            and observed_run is run_group
            and legacy_default is False
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "validate_eye_angle_compact_run",
        lambda observed: calls.append(("compact", observed)) or (),
    )
    monkeypatch.setattr(
        validation_module,
        "eye_angle_dimensions_from_run_attrs",
        lambda attrs: calls.append(("dimensions", attrs)) or "dimensions",
    )
    monkeypatch.setattr(
        validation_module,
        "validate_eye_angle_candidate_storage",
        lambda observed, *, dimensions: (
            calls.append(("storage", (observed, dimensions))) or ()
        ),
    )

    resolved, group = _resolve_eye_run(
        root,
        "analysis/eye_angle_runs/eye_candidate",
        allow_ineligible_candidate=True,
    )

    assert resolved == "eye_candidate"
    assert group is run_group
    assert [name for name, _value in calls] == ["compact", "dimensions", "storage"]

    with pytest.raises(ValueError, match="explicit run name"):
        _resolve_eye_run(root, None, allow_ineligible_candidate=True)
    with pytest.raises(ValueError, match="exact child name"):
        _resolve_eye_run(root, "latest", allow_ineligible_candidate=True)


def test_eye_run_resolution_rejects_non_candidate_ineligible_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_group = SimpleNamespace(attrs={"stage_selector_eligible": False})
    root = {"analysis/eye_angle_runs": {"eye_candidate": run_group}}
    monkeypatch.setattr(
        validation_module,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        validation_module,
        "validate_eye_angle_compact_run",
        lambda _run: (),
    )
    monkeypatch.setattr(
        validation_module,
        "eye_angle_dimensions_from_run_attrs",
        lambda _attrs: "dimensions",
    )
    monkeypatch.setattr(
        validation_module,
        "validate_eye_angle_candidate_storage",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                code="candidate_profile",
                path="analysis/eye_angle_runs/eye_candidate",
                message="not a candidate",
            ),
        ),
    )

    with pytest.raises(ValueError, match="candidate storage is invalid"):
        _resolve_eye_run(
            root,
            "eye_candidate",
            allow_ineligible_candidate=True,
        )
