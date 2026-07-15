import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.shared.detect_reason_codec import decode_reason_bytes, write_reason_columns
from fisheye.utils.backfill_keypoint_reason_bytes import _backfill_reason_columns


def test_backfill_reason_columns_writes_reason_bytes(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_backfill.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    refined.create_array("heading", shape=(3,), chunks=(2,), dtype="f8")
    reason = refined.create_array(
        "reason",
        shape=(3,),
        chunks=(2,),
        dtype=VariableLengthUTF8(),
        fill_value="",
        overwrite=True,
    )
    reason[:] = np.array(["clean", "manual_correction", "geometry_issue"], dtype=object)

    result = _backfill_reason_columns(refined, overwrite_existing=False, apply=True)
    assert result.status == "ok"
    assert "reason_bytes" in refined
    decoded = decode_reason_bytes(np.asarray(refined["reason_bytes"][:], dtype=np.uint8)).tolist()
    assert decoded == ["clean", "manual_correction", "geometry_issue"]
    assert "reason" not in refined


def test_backfill_reason_columns_skips_when_present_without_overwrite(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_backfill_skip.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    write_reason_columns(
        refined,
        np.array(["clean", "manual_correction"], dtype=object),
        chunk_size=2,
        overwrite=True,
    )

    result = _backfill_reason_columns(refined, overwrite_existing=False, apply=True)
    assert result.status == "skipped_existing"
