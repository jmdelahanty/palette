"""Pose observation v3 heading recomputation: statuses the witnesses do not show.

The revision 7 complete witness carries statuses 7, 9, 7 at the gate, 5 one
ulp below it, 6, 9 (missing eye) and 3. These tests edit scratch copies to
reach statuses 2 and 4 and the v3 binding rule; they show Palette is
consistent with its reading of ``heading_arithmetic``, not that it agrees with
Citrus.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.pose import CONTRACT, GROUP, require_pose_version_for_binding
from fisheye.shared.unified_h5.pose import _json as pose_json
from fisheye.shared.unified_h5.pose_v3 import check_v3_headings

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
COMPLETE = FIXTURES / "citrus_unified_h5_video_overlay_v1" / "video_overlay_v1_complete.h5"
PHOTOTAXIS = FIXTURES / "citrus_unified_h5_phototaxis_v1" / "phototaxis_v1_complete.h5"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "pose.h5"
    shutil.copyfile(COMPLETE, target)
    target.chmod(0o644)
    return target


def _check(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        check_v3_headings(
            h5, pose_json(h5, CONTRACT), h5[GROUP + "/updates"], h5[GROUP + "/objects"], h5[GROUP + "/keypoints"]
        )


def _set_status(h5, row: int, status: int) -> None:
    objects = h5[GROUP + "/objects"][()]
    objects["heading_status"][row] = status
    for name in ("heading_origin_x_px", "heading_origin_y_px", "heading_direction_x",
                 "heading_direction_y", "heading_angle_degrees"):
        objects[name][row] = np.nan
    h5[GROUP + "/objects"][...] = objects


def test_a_repeated_label_is_status_4(tmp_path):
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        keypoints = h5[GROUP + "/keypoints"][()]
        keypoints["label_id"][4] = 3  # object 0's eye_right relabelled eye_left
        h5[GROUP + "/keypoints"][...] = keypoints
    with pytest.raises(UnifiedH5ContractError, match="status:row=0:stored=7:expected=4"):
        _check(path)
    with h5py.File(path, "r+") as h5:
        _set_status(h5, 0, 4)
    _check(path)


def test_an_update_hash_other_than_the_sidecar_is_status_2(tmp_path):
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        updates = h5[GROUP + "/updates"][()]
        updates["pose_skeleton_id_hash"][0] ^= 1
        h5[GROUP + "/updates"][...] = updates
    with pytest.raises(UnifiedH5ContractError, match="status:row=0:stored=7:expected=2"):
        _check(path)


def test_a_valid_heading_whose_origin_differs_by_one_ulp_is_refused(tmp_path):
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        objects = h5[GROUP + "/objects"][()]
        objects["heading_origin_x_px"][0] = np.nextafter(np.float32(20.0), np.float32(21.0))
        h5[GROUP + "/objects"][...] = objects
    with pytest.raises(UnifiedH5ContractError, match="pose_heading_not_reproduced:heading_origin_x_px"):
        _check(path)


def test_an_unreferenced_sidecar_is_refused(tmp_path):
    path = _copy(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5.create_dataset(GROUP + "/sidecars/" + "0" * 64, data="{}", dtype=h5py.string_dtype())
    with pytest.raises(UnifiedH5ContractError, match="pose_sidecar_set_mismatch"):
        _check(path)


def test_video_overlay_v1_requires_pose_v3():
    with h5py.File(PHOTOTAXIS, "r") as h5:  # carries pose v2
        require_pose_version_for_binding(h5, {"admits_video_overlay": False})
        with pytest.raises(UnifiedH5ContractError, match="pose_v3_required_by_video_overlay_v1"):
            require_pose_version_for_binding(h5, {"admits_video_overlay": True})
