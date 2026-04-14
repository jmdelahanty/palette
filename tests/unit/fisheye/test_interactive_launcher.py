from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from fisheye.cli import interactive_launcher as mod


class _FakeArray:
    def __init__(self, n_rows: int) -> None:
        self.shape = (n_rows,)


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict[str, object] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}


def test_build_refine_status_lines_prefers_sparse_instances() -> None:
    refined_group = _FakeGroup(
        {
            "instances": _FakeGroup({"frame_indices": _FakeArray(12)}),
            "source_detections": _FakeGroup({"frame_indices": _FakeArray(20)}),
            "filtered": _FakeGroup(attrs={"total_detections": 30, "dropped_detections": 4}),
            "interpolated": _FakeGroup(
                attrs={"total_detections": 33, "interpolated_detections": 3, "gaps_filled": 2}
            ),
        },
        attrs={
            "summary_statistics": {
                "rows_present": 12,
                "rows_filtered_out": 3,
                "rows_manual_edited": 2,
                "frames_multi_instance": 1,
                "source_detection_candidates": 20,
                "source_detection_accepted": 12,
            }
        },
    )

    lines = mod._build_refine_status_lines(refined_group)

    assert lines == [
        "✓ Complete",
        "  └─ instances: 12 curated rows (3 filtered out, 2 manual edits, 1 multi-instance frames)",
        "  └─ source_detections: 20 candidates (12 accepted)",
    ]


def test_build_command_omits_deprecated_refine_interpolation_args(monkeypatch) -> None:
    app = mod.PipelineLauncherApp()
    app.selected_zarr = "/tmp/archive.zarr"
    app.selected_config = "configs/fisheye/default.yaml"
    app.progress_log = None
    app.stage_checkboxes = {"refine": SimpleNamespace(value=True)}

    widgets = {
        "#refine_remove_jumps_checkbox": SimpleNamespace(value=True),
        "#refine_remove_blips_checkbox": SimpleNamespace(value=False),
        "#scheduler_select": SimpleNamespace(value=mod.Select.BLANK),
    }

    def _fake_query_one(selector: str, *_args, **_kwargs):
        return widgets[selector]

    monkeypatch.setattr(app, "query_one", _fake_query_one)

    cmd = app._build_command()

    assert cmd is not None
    assert "--refine-remove-jumps" in cmd
    assert "--refine-keep-blips" in cmd
    assert "--refine-max-gap" not in cmd
    assert "--refine-method" not in cmd
