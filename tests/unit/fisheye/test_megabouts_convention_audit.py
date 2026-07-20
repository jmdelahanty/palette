from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import megabouts_convention_audit as mod


def _angle_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = np.sum(a * b, axis=1)
    return np.arctan2(cross, dot)


def _fake_compute_angles_from_keypoints(
    *,
    head_x: np.ndarray,
    head_y: np.ndarray,
    tail_x: np.ndarray,
    tail_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    head = np.stack([head_x, head_y], axis=1).astype(np.float64)
    tail = np.stack([tail_x, tail_y], axis=2).astype(np.float64)
    start_vector = tail[:, 0, :] - head
    head_yaw = np.arctan2(-start_vector[:, 1], -start_vector[:, 0])
    segments = np.diff(tail, axis=1)
    relative = np.zeros((tail.shape[0], tail.shape[1] - 1), dtype=np.float64)
    relative[:, 0] = _angle_between_vectors(start_vector, segments[:, 0, :])
    for idx in range(segments.shape[1] - 1):
        relative[:, idx + 1] = _angle_between_vectors(segments[:, idx, :], segments[:, idx + 1, :])
    return np.cumsum(relative, axis=1), head_yaw


def test_resample_tail_keypoints_preserves_base_tip_order() -> None:
    source_s = np.asarray([0.0, 0.25, 1.0], dtype=np.float32)
    tail_xy = np.asarray(
        [
            [[0.0, 0.0], [-2.5, 1.0], [-10.0, 2.0]],
            [[1.0, 0.0], [1.0, -2.5], [1.0, -10.0]],
        ],
        dtype=np.float32,
    )

    out = mod.resample_tail_keypoints(
        source_tail_sample_s=source_s,
        tail_sample_xy=tail_xy,
        target_count=11,
    )

    assert out.shape == (2, 11, 2)
    np.testing.assert_allclose(out[:, 0, :], tail_xy[:, 0, :])
    np.testing.assert_allclose(out[:, -1, :], tail_xy[:, -1, :])


def test_compute_megabouts_angles_uses_injected_keypoint_converter() -> None:
    head_xy = np.asarray([[1.0, 0.0]], dtype=np.float32)
    tail_xy = np.zeros((1, 11, 2), dtype=np.float32)
    tail_xy[0, :, 0] = np.linspace(0.0, -10.0, 11)

    angle, yaw = mod.compute_megabouts_angles_from_tail_keypoints(
        head_xy=head_xy,
        tail_keypoints_xy=tail_xy,
        compute_angles_fn=_fake_compute_angles_from_keypoints,
    )

    assert angle.shape == (1, 10)
    np.testing.assert_allclose(angle, 0.0, atol=1e-8)
    np.testing.assert_allclose(yaw, 0.0, atol=1e-8)


def test_compare_angles_reports_sign_flipped_mapping() -> None:
    palette = np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32)
    megabouts = -palette

    summary = mod.compare_megabouts_to_palette_angles(
        megabouts_tail_angle_rad=megabouts,
        palette_tail_angle_rad=palette,
        valid=np.asarray([True, True], dtype=bool),
        frame_index=np.asarray([10, 11], dtype=np.int64),
    )

    assert summary["best_mapping"] == "sign_flipped"
    assert summary["best_palette_to_megabouts_sign"] == -1
    assert summary["sign_flipped"]["max_abs_rad"] == 0.0


def _build_audit_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    shape_parent = analysis.create_group("subject_shape_runs")
    shape_parent.attrs["latest"] = "shape_001"
    shape = shape_parent.create_group("shape_001")
    shape.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([10, 11], dtype=np.int64),
        overwrite=True,
    )
    components = shape.create_group("components")
    body = components.create_group("subject_body")

    source_s = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    tail_xy = np.zeros((2, 4, 2), dtype=np.float32)
    tail_xy[:, :, 0] = -np.linspace(0.0, 10.0, 4, dtype=np.float32)[None, :]
    body.create_array("tail_sample_s", data=source_s, overwrite=True)
    body.create_array("tail_sample_xy", data=tail_xy, overwrite=True)
    body.create_array("head_endpoint_xy", data=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32), overwrite=True)
    body.create_array("tail_sample_valid", data=np.asarray([True, True], dtype=bool), overwrite=True)
    body.create_array("bspline_valid", data=np.asarray([True, True], dtype=bool), overwrite=True)

    tail_parent = analysis.create_group("tail_kinematics_runs")
    tail_parent.attrs["latest"] = "tail_001"
    tail = tail_parent.create_group("tail_001")
    tail.create_array("tail_angle_rad", data=np.zeros((2, 10), dtype=np.float32), overwrite=True)
    tail.create_array("valid", data=np.asarray([True, True], dtype=bool), overwrite=True)
    tail.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([10, 11], dtype=np.int64),
        overwrite=True,
    )
    return root


def _install_canonical_readers(
    monkeypatch: pytest.MonkeyPatch,
    root: zarr.Group,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    shape_path = "analysis/subject_shape_runs/shape_001"
    tail_path = "analysis/tail_kinematics_runs/tail_001"
    shape_publication = SimpleNamespace(
        run_path=shape_path,
        manifest=SimpleNamespace(
            record_ref=f"{shape_path}@subject_shape_publication_manifest",
            record_sha256="a" * 64,
        ),
    )
    tail_publication = SimpleNamespace(
        run_path=tail_path,
        manifest=SimpleNamespace(
            record_ref=f"{tail_path}@tail_coordinate_publication_manifest",
            record_sha256="b" * 64,
        ),
        source=shape_publication,
    )

    def _resolve_shape(
        source_root: zarr.Group,
        run_name: str | None = None,
    ) -> tuple[zarr.Group, str, str, SimpleNamespace]:
        del source_root
        if run_name not in (None, "", "latest", "shape_001"):
            raise ValueError(f"Unknown shape run {run_name!r}.")
        return root[shape_path], "shape_001", shape_path, shape_publication

    def _resolve_tail(
        source_root: zarr.Group,
        run_name: str | None = None,
    ) -> tuple[zarr.Group, str, str]:
        del source_root
        if run_name not in (None, "", "latest", "tail_001"):
            raise ValueError(f"Unknown tail run {run_name!r}.")
        return root[tail_path], "tail_001", tail_path

    monkeypatch.setattr(mod, "resolve_canonical_subject_shape_run", _resolve_shape)
    monkeypatch.setattr(mod, "resolve_tail_kinematics_run", _resolve_tail)
    monkeypatch.setattr(
        mod,
        "load_tail_kinematics_coordinate_publication",
        lambda source_root, run_path: tail_publication,
    )
    return shape_publication, tail_publication


def test_audit_group_is_read_only_and_reports_direct_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_audit_root()
    _install_canonical_readers(monkeypatch, root)

    summary = mod.audit_megabouts_tail_convention_group(
        root,
        subject_shape_run="shape_001",
        tail_kinematics_run="tail_001",
        compute_angles_fn=_fake_compute_angles_from_keypoints,
    )

    assert summary["status"] == "ok"
    assert summary["mutates_archive"] is False
    assert summary["megabouts_keypoint_count"] == 11
    assert summary["megabouts_segment_count"] == 10
    assert summary["comparison"]["best_mapping"] == "direct"
    assert summary["comparison"]["valid_row_count"] == 2
    assert summary["source_refs"]["source_acquisition_frame_index"].endswith(
        "/source_acquisition_frame_index"
    )
    assert "bout_classification_runs" not in root["analysis"]


def test_audit_requires_direct_source_acquisition_frame_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_audit_root()
    _install_canonical_readers(monkeypatch, root)
    del root[
        "analysis/tail_kinematics_runs/tail_001/source_acquisition_frame_index"
    ]

    with pytest.raises(ValueError, match="source_acquisition_frame_index"):
        mod.audit_megabouts_tail_convention_group(
            root,
            subject_shape_run="shape_001",
            tail_kinematics_run="tail_001",
            compute_angles_fn=_fake_compute_angles_from_keypoints,
        )


def test_audit_requires_exact_subject_shape_publication_linkage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_audit_root()
    shape_publication, tail_publication = _install_canonical_readers(
        monkeypatch,
        root,
    )
    wrong_source = SimpleNamespace(
        run_path=shape_publication.run_path,
        manifest=SimpleNamespace(
            record_ref=shape_publication.manifest.record_ref,
            record_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        mod,
        "load_tail_kinematics_coordinate_publication",
        lambda source_root, run_path: SimpleNamespace(
            run_path=tail_publication.run_path,
            manifest=tail_publication.manifest,
            source=wrong_source,
        ),
    )

    with pytest.raises(ValueError, match="exact selected subject-shape"):
        mod.audit_megabouts_tail_convention_group(
            root,
            subject_shape_run="shape_001",
            tail_kinematics_run="tail_001",
            compute_angles_fn=_fake_compute_angles_from_keypoints,
        )


def test_audit_revalidates_publications_after_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_audit_root()
    _shape_publication, tail_publication = _install_canonical_readers(
        monkeypatch,
        root,
    )
    call_count = 0

    def _changing_tail_publication(
        source_root: zarr.Group,
        run_path: str,
    ) -> SimpleNamespace:
        del source_root, run_path
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tail_publication
        return SimpleNamespace(
            run_path=tail_publication.run_path,
            manifest=SimpleNamespace(
                record_ref=tail_publication.manifest.record_ref,
                record_sha256="f" * 64,
            ),
            source=tail_publication.source,
        )

    monkeypatch.setattr(
        mod,
        "load_tail_kinematics_coordinate_publication",
        _changing_tail_publication,
    )

    with pytest.raises(ValueError, match="changed while Megabouts audit inputs"):
        mod.audit_megabouts_tail_convention_group(
            root,
            subject_shape_run="shape_001",
            tail_kinematics_run="tail_001",
            compute_angles_fn=_fake_compute_angles_from_keypoints,
        )
