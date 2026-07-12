from fisheye.detection.detect_keypoints_yolo import (
    DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
)
from fisheye.inference import predict_pose


def test_predict_pose_defaults_to_keypoint_sharding_and_supports_opt_out() -> None:
    parser = predict_pose._build_arg_parser()
    default_args = parser.parse_args(["--model", "pose.pt", "--zarr", "archive.zarr"])
    regular_args = parser.parse_args(
        [
            "--model",
            "pose.pt",
            "--zarr",
            "archive.zarr",
            "--no-keypoint-sharding",
        ]
    )

    assert default_args.keypoint_roi_shard_rows == DEFAULT_KEYPOINT_ROI_SHARD_ROWS
    assert default_args.keypoint_frame_shard_rows == DEFAULT_KEYPOINT_FRAME_SHARD_ROWS
    assert regular_args.keypoint_roi_shard_rows is None
