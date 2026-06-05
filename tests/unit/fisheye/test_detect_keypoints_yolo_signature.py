from inspect import signature

from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo


def test_detect_keypoints_yolo_accepts_mask_threshold() -> None:
    params = signature(detect_keypoints_yolo).parameters
    assert "mask_threshold" in params
    assert "registry" in params
    assert "roi_cache_policy" in params
    assert "roi_cache_dir" in params
    assert "roi_live_acceleration" in params
    assert "roi_live_gpu_chunk_frames" in params
    assert "input_mode" in params
    assert "profile_timings" in params
