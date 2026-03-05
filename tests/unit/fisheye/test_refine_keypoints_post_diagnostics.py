from __future__ import annotations

import json
from pathlib import Path

from fisheye.refinement import refine_keypoints as mod


def test_dataset_report_stem_sanitizes_path() -> None:
    stem = mod._dataset_report_stem("/tmp/2026-01-28T19:22:28Z arena_1_analysis.zarr")
    assert stem == "2026-01-28T19_22_28Z_arena_1_analysis"


def test_post_refinement_diagnostics_writes_audit_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def _audit(**kwargs):  # noqa: ANN003
        calls.update(kwargs)
        return {"status_counts": {"aligned": 123}, "bad_frame_ranges": []}

    monkeypatch.setattr(mod, "_analyze_coordinate_space", _audit)
    monkeypatch.setattr(
        mod,
        "_analyze_bad_row_overlap",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("overlap should not run")),  # noqa: ANN003
    )

    attrs: dict[str, object] = {}
    result = mod._run_post_refinement_diagnostics(
        zarr_path="/data/recording_analysis.zarr",
        run_name="refined_keypoints_001",
        source_crop_run="crop_001",
        config={"refine_keypoints": {"post_refinement_output_dir": str(tmp_path)}},
        run_attrs=attrs,
        console=None,
    )

    audit_path = tmp_path / "recording_analysis_audit.json"
    assert result["enabled"] is True
    assert result["audit_json"] == str(audit_path)
    assert result["overlap_json"] is None
    assert audit_path.exists()
    assert json.loads(audit_path.read_text(encoding="utf-8"))["status_counts"]["aligned"] == 123
    assert attrs["post_refinement_audit_json"] == str(audit_path)
    assert isinstance(attrs["post_refinement_audit_generated_utc"], str)
    assert calls["run_name"] == "refined_keypoints_001"


def test_post_refinement_diagnostics_writes_overlap_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "_analyze_coordinate_space",
        lambda **_kwargs: {"status_counts": {"aligned": 2}},  # noqa: ANN003
    )
    monkeypatch.setattr(
        mod,
        "_analyze_bad_row_overlap",
        lambda **_kwargs: {"bad_row_count": 7, "bad_frame_ranges": [[0, 1]]},  # noqa: ANN003
    )

    attrs: dict[str, object] = {}
    result = mod._run_post_refinement_diagnostics(
        zarr_path="/data/recording_analysis.zarr",
        run_name="refined_keypoints_001",
        source_crop_run="crop_001",
        config={
            "refine_keypoints": {
                "post_refinement_output_dir": str(tmp_path),
                "post_refinement_overlap": True,
            }
        },
        run_attrs=attrs,
        console=None,
    )

    audit_path = tmp_path / "recording_analysis_audit.json"
    overlap_path = tmp_path / "recording_analysis_overlap.json"
    assert result["audit_json"] == str(audit_path)
    assert result["overlap_json"] == str(overlap_path)
    assert audit_path.exists()
    assert overlap_path.exists()
    assert attrs["post_refinement_overlap_json"] == str(overlap_path)
    assert attrs["post_refinement_overlap_bad_row_count"] == 7


def test_post_refinement_diagnostics_respects_disable_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "_analyze_coordinate_space",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("audit should not run")),  # noqa: ANN003
    )

    attrs: dict[str, object] = {}
    result = mod._run_post_refinement_diagnostics(
        zarr_path="/data/recording_analysis.zarr",
        run_name="refined_keypoints_001",
        source_crop_run=None,
        config={
            "refine_keypoints": {
                "post_refinement_audit": False,
                "post_refinement_output_dir": str(tmp_path),
            }
        },
        run_attrs=attrs,
        console=None,
    )

    assert result["enabled"] is False
    assert result["audit_json"] is None
    assert result["overlap_json"] is None
    assert attrs == {}
