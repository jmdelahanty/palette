from __future__ import annotations

import numpy as np
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
    tail.create_array("frame_index", data=np.asarray([10, 11], dtype=np.int64), overwrite=True)
    return root


def test_audit_group_is_read_only_and_reports_direct_mapping() -> None:
    root = _build_audit_root()

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
    assert "bout_classification_runs" not in root["analysis"]
