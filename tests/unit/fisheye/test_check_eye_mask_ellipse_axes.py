from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.diagnostics import check_eye_mask_ellipse_axes as mod


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    def group_keys(self) -> list[str]:
        keys: list[str] = []
        for key, value in self.items():
            if isinstance(value, _FakeGroup):
                keys.append(str(key))
        return keys


def test_analyze_run_group_reports_axis_issues_and_samples() -> None:
    ellipse_params = np.zeros((3, 2, 5), dtype=np.float32)
    ellipse_success = np.zeros((3, 2), dtype=bool)

    # noncanonical (major < minor)
    ellipse_params[0, 0, 2] = 2.0
    ellipse_params[0, 0, 3] = 4.0
    ellipse_success[0, 0] = True

    # canonical
    ellipse_params[0, 1, 2] = 5.0
    ellipse_params[0, 1, 3] = 3.0
    ellipse_success[0, 1] = True

    # nonpositive
    ellipse_params[1, 0, 2] = 0.0
    ellipse_params[1, 0, 3] = 2.0
    ellipse_success[1, 0] = True

    # nonfinite
    ellipse_params[1, 1, 2] = np.nan
    ellipse_params[1, 1, 3] = 2.0
    ellipse_success[1, 1] = True

    # equal axes
    ellipse_params[2, 1, 2] = 3.0
    ellipse_params[2, 1, 3] = 3.0
    ellipse_success[2, 1] = True

    run_group = _FakeGroup(
        {
            "ellipse_params": ellipse_params,
            "ellipse_success": ellipse_success,
            "frame_indices": np.array([100, 101, 102], dtype=np.int32),
        }
    )

    report = mod._analyze_run_group(
        "refined_eye_masks_runs",
        "refined_eye_masks_001",
        run_group,
        tolerance=0.0,
        sample_limit=4,
        chunk_size=2,
    )

    assert report.success_eyes == 5
    assert report.validated_eyes == 4
    assert report.positive_valid_eyes == 3
    assert report.noncanonical_eyes == 1
    assert report.nonpositive_eyes == 1
    assert report.nonfinite_eyes == 1
    assert report.equal_axis_eyes == 1
    assert len(report.violation_samples) == 1
    sample = report.violation_samples[0]
    assert sample["roi_idx"] == 0
    assert sample["eye_idx"] == 0
    assert sample["frame_idx"] == 100


def test_analyze_run_group_respects_tolerance() -> None:
    ellipse_params = np.zeros((1, 1, 5), dtype=np.float32)
    ellipse_success = np.array([[True]], dtype=bool)
    ellipse_params[0, 0, 2] = 1.0
    ellipse_params[0, 0, 3] = 1.0005

    run_group = _FakeGroup(
        {
            "ellipse_params": ellipse_params,
            "ellipse_success": ellipse_success,
        }
    )

    report = mod._analyze_run_group(
        "eye_masks_runs",
        "eye_masks_001",
        run_group,
        tolerance=0.001,
        sample_limit=2,
        chunk_size=128,
    )

    assert report.noncanonical_eyes == 0
    assert report.has_issues is False


def test_resolve_run_names_prefers_latest_unless_all_runs() -> None:
    parent = _FakeGroup(
        {
            "eye_masks_001": _FakeGroup(),
            "eye_masks_002": _FakeGroup(),
        },
        attrs={"latest": "eye_masks_002"},
    )

    assert mod._resolve_run_names(parent, explicit_run=None, all_runs=False) == ["eye_masks_002"]
    assert mod._resolve_run_names(parent, explicit_run=None, all_runs=True) == [
        "eye_masks_001",
        "eye_masks_002",
    ]


def test_run_strict_exit_code_tracks_axis_violations(monkeypatch) -> None:
    params_ok = np.zeros((2, 2, 5), dtype=np.float32)
    success = np.ones((2, 2), dtype=bool)
    params_ok[:, :, 2] = 4.0
    params_ok[:, :, 3] = 2.0

    params_bad = params_ok.copy()
    params_bad[0, 1, 2] = 1.0
    params_bad[0, 1, 3] = 3.0

    root_ok = _FakeGroup(
        {
            "refined_eye_masks_runs": _FakeGroup(
                {
                    "refined_eye_masks_001": _FakeGroup(
                        {
                            "ellipse_params": params_ok,
                            "ellipse_success": success,
                        }
                    )
                },
                attrs={"latest": "refined_eye_masks_001"},
            )
        }
    )
    root_bad = _FakeGroup(
        {
            "refined_eye_masks_runs": _FakeGroup(
                {
                    "refined_eye_masks_001": _FakeGroup(
                        {
                            "ellipse_params": params_bad,
                            "ellipse_success": success,
                        }
                    )
                },
                attrs={"latest": "refined_eye_masks_001"},
            )
        }
    )

    args = mod.build_parser().parse_args(
        [
            "dummy.zarr",
            "--stage",
            "refined_eye_masks_runs",
            "--strict",
        ]
    )

    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: root_ok)
    assert mod.run(args) == 0

    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: root_bad)
    assert mod.run(args) == 1
