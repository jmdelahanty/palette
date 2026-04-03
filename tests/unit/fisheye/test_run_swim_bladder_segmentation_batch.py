from __future__ import annotations

from pathlib import Path

from fisheye.utils import run_swim_bladder_segmentation_batch as mod


class _FakeRoot:
    def __init__(
        self,
        *,
        zarr_use: str | None = None,
        subject_mask_runs: set[str] | None = None,
        subject_mask_tuning: dict[str, object] | None = None,
    ) -> None:
        self.attrs: dict[str, object] = {}
        if zarr_use is not None:
            self.attrs["zarr_use"] = zarr_use
        self._groups: dict[str, object] = {}
        if subject_mask_runs is not None:
            self._groups["subject_mask_runs"] = set(subject_mask_runs)
        if subject_mask_tuning is not None:
            self._groups["analysis_metadata"] = _FakeRootAnalysis(subject_mask_tuning)

    def get(self, key: str) -> object | None:
        return self._groups.get(key)


class _FakeRootAnalysis:
    def __init__(self, subject_mask_tuning: dict[str, object]) -> None:
        self.attrs = {"subject_mask_tuning": subject_mask_tuning}


def _install_fake_roots(monkeypatch, roots: dict[Path, _FakeRoot]) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(mod, "open_zarr_root", lambda path, mode="r": roots[Path(path)])
    monkeypatch.setattr(mod, "_camera_id_for_zarr", lambda zarr_path, root: "2010093")


def _default_options(**overrides) -> mod.BatchOptions:
    base = mod.BatchOptions(
        apply=False,
        zarr_use_filter="training",
        output_run="traditional_swim_bladder_masks_batch_001",
        overwrite=False,
        require_swim_tuning=True,
        config_dict={},
    )
    values = dict(base.__dict__)
    values.update(overrides)
    return mod.BatchOptions(**values)


def test_process_zarr_path_plans_training_archive_with_swim_tuning(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(
        monkeypatch,
        {
            zarr_path: _FakeRoot(
                zarr_use="training",
                subject_mask_tuning={
                    "version": "2.0",
                    "components": {
                        "swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"},
                    },
                },
            )
        },
    )

    row = mod._process_zarr_path(zarr_path, _default_options())  # noqa: SLF001

    assert row.status == "planned"
    assert row.camera_id == "2010093"
    assert row.output_run == "traditional_swim_bladder_masks_batch_001"
    assert row.matched_components == ("swim_bladder",)


def test_process_zarr_path_skips_missing_swim_tuning_by_default(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(monkeypatch, {zarr_path: _FakeRoot(zarr_use="training")})

    row = mod._process_zarr_path(zarr_path, _default_options())  # noqa: SLF001

    assert row.status == "skipped_missing_tuning"
    assert row.reason == "analysis_metadata.subject_mask_tuning.components.swim_bladder missing"


def test_process_zarr_path_skips_existing_output_without_overwrite(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(
        monkeypatch,
        {
            zarr_path: _FakeRoot(
                zarr_use="training",
                subject_mask_runs={"traditional_swim_bladder_masks_batch_001"},
                subject_mask_tuning={
                    "version": "2.0",
                    "components": {"swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"}},
                },
            )
        },
    )

    row = mod._process_zarr_path(zarr_path, _default_options())  # noqa: SLF001

    assert row.status == "skipped_existing"
    assert row.reason == "subject_mask_runs/traditional_swim_bladder_masks_batch_001 already exists"


def test_process_zarr_path_apply_runs_segmentation(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(
        monkeypatch,
        {
            zarr_path: _FakeRoot(
                zarr_use="training",
                subject_mask_tuning={
                    "version": "2.0",
                    "components": {"swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"}},
                },
            )
        },
    )
    captured: dict[str, object] = {}

    def _fake_segment(zarr_path_arg, *, config_dict, console, output_run, overwrite):  # type: ignore[no-untyped-def]
        captured["zarr_path"] = zarr_path_arg
        captured["config_dict"] = config_dict
        captured["console"] = console
        captured["output_run"] = output_run
        captured["overwrite"] = overwrite
        return "traditional_swim_bladder_masks_batch_001"

    monkeypatch.setattr(mod.swim_mod, "segment_swim_bladder_masks", _fake_segment)

    row = mod._process_zarr_path(  # noqa: SLF001
        zarr_path,
        _default_options(apply=True, config_dict={"roi_padding": 12}),
    )

    assert row.status == "updated"
    assert row.output_run == "traditional_swim_bladder_masks_batch_001"
    assert captured["zarr_path"] == zarr_path
    assert captured["config_dict"] == {"roi_padding": 12}
    assert captured["console"] is None
    assert captured["output_run"] == "traditional_swim_bladder_masks_batch_001"
    assert captured["overwrite"] is False


def test_main_scans_roots_and_reports_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    training_path = tmp_path / "recording_training.zarr"
    analysis_path = tmp_path / "recording_analysis.zarr"
    roots = {
        training_path: _FakeRoot(
            zarr_use="training",
            subject_mask_tuning={
                "version": "2.0",
                "components": {"swim_bladder": {"subject_method_family": "swim_bladder_polar_boundary_v1"}},
            },
        ),
        analysis_path: _FakeRoot(zarr_use="analysis"),
    }
    _install_fake_roots(monkeypatch, roots)
    monkeypatch.setattr(mod, "_iter_zarr", lambda roots, recursive: [training_path, analysis_path])

    rc = mod.main([str(tmp_path), "--recursive", "--run-name", "traditional_swim_bladder_masks_batch_001"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "planned" in out
    assert "scope=training" in out
    assert "Results: updated=0 planned=1 skipped_existing=0 skipped_missing_tuning=0" in out
