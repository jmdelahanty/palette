from __future__ import annotations

from pathlib import Path
from typing import Any

from fisheye.utils import export_keypoint_quality_overview as mod


class _FakeGroup:
    def __init__(self, *, attrs: dict[str, Any] | None = None, children: dict[str, Any] | None = None) -> None:
        self.attrs = attrs or {}
        self._children = children or {}

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        return self._children[key]

    def get(self, key: str, default=None):
        return self._children.get(key, default)

    def keys(self):
        return self._children.keys()

    def group_keys(self):
        return self._children.keys()


def test_extract_temporal_heading_summary_prefers_postprocess() -> None:
    run_group = _FakeGroup(
        attrs={
            "summary_statistics": {
                "heading_temporal_outlier": 1,
                "heading_temporal_evaluable": 3,
                "postprocess": {
                    "heading_temporal_outlier": 4,
                    "heading_temporal_evaluable": 10,
                    "heading_temporal_outlier_rate_percent": 40.0,
                },
            }
        }
    )

    outlier, evaluable, rate = mod._extract_temporal_heading_summary(run_group)

    assert outlier == 4
    assert evaluable == 10
    assert rate == 40.0


def test_collect_rows_sorts_by_temporal_outlier_count(monkeypatch, tmp_path: Path) -> None:
    low_path = tmp_path / "rec_low_training.zarr"
    high_path = tmp_path / "rec_high_training.zarr"
    roots = [tmp_path]

    fake_roots = {
        str(low_path): _FakeGroup(
            children={
                "refined_keypoints_runs": _FakeGroup(
                    attrs={"latest": "rk_low"},
                    children={
                        "rk_low": _FakeGroup(
                            attrs={
                                "summary_statistics": {
                                    "postprocess": {
                                        "heading_temporal_outlier": 1,
                                        "heading_temporal_evaluable": 10,
                                        "heading_temporal_outlier_rate_percent": 10.0,
                                    }
                                }
                            },
                            children={
                                "visualizations": _FakeGroup(
                                    children={mod.QUALITY_ARTIFACT_NAME: object()}
                                )
                            },
                        )
                    },
                )
            }
        ),
        str(high_path): _FakeGroup(
            children={
                "refined_keypoints_runs": _FakeGroup(
                    attrs={"latest": "rk_high"},
                    children={
                        "rk_high": _FakeGroup(
                            attrs={
                                "summary_statistics": {
                                    "postprocess": {
                                        "heading_temporal_outlier": 5,
                                        "heading_temporal_evaluable": 10,
                                        "heading_temporal_outlier_rate_percent": 50.0,
                                    }
                                }
                            },
                            children={
                                "visualizations": _FakeGroup(
                                    children={mod.QUALITY_ARTIFACT_NAME: object()}
                                )
                            },
                        )
                    },
                )
            }
        ),
    }

    monkeypatch.setattr(mod, "_iter_zarr", lambda _roots, recursive: [low_path, high_path])
    monkeypatch.setattr(mod, "_read_zarr_attrs", lambda _path: {"zarr_purpose": "training"})
    monkeypatch.setattr(mod.zarr, "open_group", lambda path, mode="r": fake_roots[str(path)])

    rows = mod._collect_rows(
        roots,
        recursive=False,
        zarr_use_filter="training",
        refined_run=None,
        output_dir=tmp_path / "exports",
        artifact_name=mod.QUALITY_ARTIFACT_NAME,
        sort_by="temporal-outliers",
    )

    assert [Path(row.zarr_path).name for row in rows] == ["rec_high_training.zarr", "rec_low_training.zarr"]
    assert rows[0].heading_temporal_outlier == 5
    assert rows[1].heading_temporal_outlier == 1
