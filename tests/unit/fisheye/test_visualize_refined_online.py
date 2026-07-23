from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.refinement.refine_online_detect import CanonicalOnlineRefinementError
from fisheye.visualization import visualize_refined_online as mod


def _fixture() -> tuple[zarr.Group, zarr.Group]:
    root = zarr.group()
    root.attrs["fps"] = 20.0
    parent = root.require_group("refined_online_runs")
    parent.attrs["latest_complete"] = "canonical"
    parent.attrs["latest"] = "unsafe_alias"
    run = parent.require_group("canonical")
    frames = np.asarray([10, 11, 12], dtype=np.int64)
    run.create_array("camera_frame_ids", data=frames)
    run.create_array(
        "original_valid_mask",
        data=np.asarray([True, False, True], dtype=bool),
    )
    filtered = run.require_group("filtered")
    filtered.create_array(
        "positions_px",
        data=np.asarray([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
    )
    filtered.create_array(
        "valid_mask",
        data=np.asarray([True, False, True], dtype=bool),
    )
    interpolated = run.require_group("interpolated")
    interpolated.create_array(
        "positions_px",
        data=np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    )
    interpolated.create_array(
        "valid_mask",
        data=np.asarray([True, True, True], dtype=bool),
    )
    interpolated.create_array(
        "interpolation_mask",
        data=np.asarray([False, True, False], dtype=bool),
    )
    return root, run


def test_visualizer_uses_canonical_preflight_and_post_copy_revalidation(
    monkeypatch,
) -> None:
    root, run = _fixture()
    calls: list[str] = []
    proof = SimpleNamespace(assert_verified=lambda: calls.append("post_copy"))

    def _load(candidate_root, candidate_run):
        assert candidate_root is root
        assert candidate_run.path == run.path
        calls.append("preflight")
        return proof

    monkeypatch.setattr(
        mod,
        "load_bound_refined_online_coordinate_evidence",
        _load,
    )

    result = mod.load_refined_online_visualization_inputs(root)

    assert result["run_name"] == "canonical"
    assert calls == ["preflight", "post_copy"]
    np.testing.assert_array_equal(result["frames"], [10, 11, 12])
    assert result["datasets"]["interpolated"]["coverage_percent"] == 100.0


def test_visualizer_discards_copies_when_publication_changes(
    monkeypatch,
) -> None:
    root, _run = _fixture()

    def _changed() -> None:
        raise CanonicalOnlineRefinementError(
            "Refined-online evidence changed after the copied values were read."
        )

    monkeypatch.setattr(
        mod,
        "load_bound_refined_online_coordinate_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(assert_verified=_changed),
    )

    with pytest.raises(CanonicalOnlineRefinementError, match="changed after"):
        mod.load_refined_online_visualization_inputs(root)


def test_visualizer_fails_closed_when_canonical_preflight_rejects_run(
    monkeypatch,
) -> None:
    root, _run = _fixture()
    monkeypatch.setattr(
        mod,
        "load_bound_refined_online_coordinate_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CanonicalOnlineRefinementError("selector-ineligible")
        ),
    )

    with pytest.raises(CanonicalOnlineRefinementError, match="selector-ineligible"):
        mod.load_refined_online_visualization_inputs(root)
