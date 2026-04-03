from __future__ import annotations

from pathlib import Path

from fisheye.utils import find_subject_mask_tuning_sources as mod


class _FakeGroup(dict):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: object = None) -> object:
        return super().get(key, default)


def _make_root(
    *,
    zarr_use: str | None = None,
    subject_mask_tuning: dict[str, object] | None = None,
) -> _FakeGroup:
    root = _FakeGroup(attrs={})
    if zarr_use is not None:
        root.attrs["zarr_use"] = zarr_use
    if subject_mask_tuning is not None:
        root["analysis_metadata"] = _FakeGroup(attrs={"subject_mask_tuning": subject_mask_tuning})
    return root


def test_scan_zarr_path_matches_selected_subject_mask_component(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = _make_root(
        zarr_use="training",
        subject_mask_tuning={
            "version": "2.0",
            "latest_component": "swim_bladder",
            "latest_timestamp": "2026-04-02T12:00:00+00:00",
            "components": {
                "subject_body": {"method": "traditional_subject_mask_seed"},
                "swim_bladder": {
                    "subject_method_family": "swim_bladder_polar_boundary_v1",
                    "tuned_timestamp": "2026-04-01T18:30:00+00:00",
                },
            },
        },
    )

    monkeypatch.setattr(mod, "open_zarr_root", lambda path, mode="r": root)
    monkeypatch.setattr(mod, "_camera_id_for_zarr", lambda path, root: "2010093")

    row = mod._scan_zarr_path(  # noqa: SLF001
        zarr_path,
        zarr_use_filter="training",
        subject_mask_components=("swim_bladder",),
    )

    assert row.status == "match"
    assert row.camera_id == "2010093"
    assert row.component_names == ("subject_body", "swim_bladder")
    assert row.matched_components == ("swim_bladder",)
    assert row.latest_component == "swim_bladder"
    assert row.latest_timestamp == "2026-04-02T12:00:00+00:00"
    assert row.component_details == (
        "component[swim_bladder]: method=swim_bladder_polar_boundary_v1 tuned_timestamp=2026-04-01T18:30:00+00:00",
    )


def test_scan_zarr_path_reports_missing_selected_component(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = _make_root(
        zarr_use="training",
        subject_mask_tuning={
            "version": "2.0",
            "components": {
                "subject_body": {"method": "traditional_subject_mask_seed"},
            },
        },
    )

    monkeypatch.setattr(mod, "open_zarr_root", lambda path, mode="r": root)
    monkeypatch.setattr(mod, "_camera_id_for_zarr", lambda path, root: "2010093")

    row = mod._scan_zarr_path(  # noqa: SLF001
        zarr_path,
        zarr_use_filter="training",
        subject_mask_components=("swim_bladder",),
    )

    assert row.status == "missing_components"
    assert row.component_names == ("subject_body",)
    assert row.reason == "swim_bladder"


def test_main_lists_matches_and_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    match_path = tmp_path / "recording_a_training.zarr"
    miss_path = tmp_path / "recording_b_training.zarr"
    analysis_path = tmp_path / "recording_c_analysis.zarr"
    roots = {
        match_path: _make_root(
            zarr_use="training",
            subject_mask_tuning={
                "version": "2.0",
                "latest_component": "swim_bladder",
                "components": {
                    "swim_bladder": {
                        "subject_method_family": "swim_bladder_polar_boundary_v1",
                        "tuned_timestamp": "2026-04-01T18:30:00+00:00",
                    }
                },
            },
        ),
        miss_path: _make_root(
            zarr_use="training",
            subject_mask_tuning={
                "version": "2.0",
                "components": {
                    "subject_body": {"method": "traditional_subject_mask_seed"},
                },
            },
        ),
        analysis_path: _make_root(
            zarr_use="analysis",
            subject_mask_tuning={
                "version": "2.0",
                "components": {
                    "swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"},
                },
            },
        ),
    }

    monkeypatch.setattr(mod, "_iter_zarr", lambda roots, recursive: [match_path, miss_path, analysis_path])
    monkeypatch.setattr(mod, "open_zarr_root", lambda path, mode="r": roots[Path(path)])
    monkeypatch.setattr(mod, "_camera_id_for_zarr", lambda path, root: "2010093")

    rc = mod.main([str(tmp_path), "--recursive", "--subject-mask-components", "swim_bladder"])
    out = capsys.readouterr().out

    assert rc == 0
    assert f"MATCH {match_path}" in out
    assert str(miss_path) not in out
    assert "matched=1" in out
    assert "filtered_zarr_use=1" in out
    assert "missing_components=1" in out
