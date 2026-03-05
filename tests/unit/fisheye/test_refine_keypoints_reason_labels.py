import numpy as np
import zarr

from fisheye.refinement.refine_keypoints import _extract_skeleton_edges, _write_reason_arrays
from fisheye.shared.detect_reason_codec import decode_reason_bytes


class _StubGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}


def test_write_reason_arrays_creates_reason_and_reason_bytes(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_reasons.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")

    labels = np.array(
        [
            "clean",
            "flip_corrected|geometry_issue",
            "manual_correction",
        ],
        dtype=object,
    )
    _write_reason_arrays(refined, labels, chunk_size=2)

    stored_text = np.asarray(refined["reason"][:], dtype=object).tolist()
    assert stored_text == labels.tolist()

    stored_bytes = np.asarray(refined["reason_bytes"][:], dtype=np.uint8)
    decoded = decode_reason_bytes(stored_bytes).tolist()
    assert decoded == labels.tolist()
    assert refined.attrs["reason_encoding"] == "utf8-null-terminated"
    assert refined.attrs["reason_fallback_order"] == ["reason_bytes", "reason", "detection_source"]


def test_extract_skeleton_edges_prefers_pose_schema_and_dedupes() -> None:
    run = _StubGroup()
    run.attrs["pose_schema"] = {
        "edges": [[0, 1], [1, 0], [2, 3], [4, 1], ["a", 2], [2, 2]],
    }

    edges, source = _extract_skeleton_edges(run, n_keypoints=4)

    assert source == "pose_schema"
    assert edges.tolist() == [[0, 1], [2, 3]]


def test_extract_skeleton_edges_defaults_to_triangle_for_three_keypoints() -> None:
    run = _StubGroup()

    edges, source = _extract_skeleton_edges(run, n_keypoints=3)

    assert source == "default_triangle"
    assert edges.tolist() == [[0, 1], [0, 2], [1, 2]]
