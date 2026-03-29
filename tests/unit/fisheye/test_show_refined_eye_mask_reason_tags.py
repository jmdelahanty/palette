from __future__ import annotations

import json

import zarr

from fisheye.diagnostics import show_refined_eye_mask_reason_tags as mod


def _build_archive(tmp_path):
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_a"
    run = refined_parent.create_group("refined_a")
    run.attrs.update(
        {
            "successful_roi_pairs": 12,
            "successful_roi_pair_rate": 0.75,
            "mask_probability_threshold": 0.45,
            "mask_probability_threshold_source": "default",
            "success_min_eye_area_px": 50.0,
            "metrics_summary": {
                "reason_tag_counts": {
                    "ellipse_fail_pair": 3,
                    "filtered_pair": 1,
                    "union_source": 16,
                }
            },
        }
    )
    return tmp_path / "archive.zarr"


def test_show_refined_eye_mask_reason_tags_prints_json(tmp_path, capsys):
    archive = _build_archive(tmp_path)

    mod.show_refined_eye_mask_reason_tags(archive, "refined_a")

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {
        "ellipse_fail_pair": 3,
        "filtered_pair": 1,
        "union_source": 16,
    }


def test_show_refined_eye_mask_reason_tags_prints_summary(tmp_path, capsys):
    archive = _build_archive(tmp_path)

    mod.show_refined_eye_mask_reason_tags(archive, "refined_a", show_summary=True)

    out = capsys.readouterr().out
    assert "run_name: refined_a" in out
    assert "successful_roi_pairs: 12" in out
    assert "mask_probability_threshold: 0.45" in out
    payload = json.loads(out.split("\n\n", 1)[1])
    assert payload["ellipse_fail_pair"] == 3


def test_show_refined_eye_mask_reason_tags_requires_refined_group(tmp_path):
    archive = tmp_path / "missing.zarr"
    zarr.open_group(str(archive), mode="w")

    try:
        mod.show_refined_eye_mask_reason_tags(archive, "refined_a")
    except ValueError as exc:
        assert "No refined_eye_masks_runs group found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_show_refined_eye_mask_reason_tags_requires_run(tmp_path):
    archive = _build_archive(tmp_path)

    try:
        mod.show_refined_eye_mask_reason_tags(archive, "missing")
    except ValueError as exc:
        assert "Run 'missing' not found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
