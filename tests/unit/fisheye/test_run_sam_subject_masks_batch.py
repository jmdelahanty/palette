from __future__ import annotations

from pathlib import Path

from fisheye.utils import run_sam_subject_masks_batch as mod


class _FakeRoot:
    def __init__(
        self,
        *,
        zarr_use: str | None = None,
        subject_mask_runs: set[str] | None = None,
    ) -> None:
        self.attrs: dict[str, object] = {}
        if zarr_use is not None:
            self.attrs["zarr_use"] = zarr_use
        self._groups: dict[str, object] = {}
        if subject_mask_runs is not None:
            self._groups["subject_mask_runs"] = set(subject_mask_runs)

    def get(self, key: str) -> object | None:
        return self._groups.get(key)


def _install_fake_roots(monkeypatch, roots: dict[Path, _FakeRoot]) -> None:  # type: ignore[no-untyped-def]
    def _fake_open_zarr_root(path: str | Path, mode: str = "r") -> _FakeRoot:
        return roots[Path(path)]

    monkeypatch.setattr(mod, "open_zarr_root", _fake_open_zarr_root)


def _default_options(**overrides) -> mod.BatchOptions:
    base = mod.BatchOptions(
        apply=False,
        zarr_use_filter="training",
        crop_run=None,
        keypoint_run=None,
        keypoint_group="auto",
        prepare_count=0,
        output_run=None,
        sam3_root=None,
        checkpoint=None,
        batch_size=8,
        device=None,
        multimask_output=True,
        use_box_prompt=True,
        box_prompt_source="detect",
        roi_inset_fraction=0.05,
        pose_box_expand_fraction=0.10,
        negative_point_policy="none",
        negative_point_margin_fraction=0.05,
        overwrite=False,
        no_hf_download=False,
        include_interpolated=False,
        positive_keypoint_labels=None,
        allow_zero_eligible=False,
    )
    values = dict(base.__dict__)
    values.update(overrides)
    return mod.BatchOptions(**values)


def _inspect_summary(*, output_run: str = "sam_subject_masks_from_refined_001", eligible_rows: int = 5) -> dict[str, object]:
    return {
        "crop_run": "crop_001",
        "keypoint_group": "refined_keypoints_runs",
        "keypoint_run": "refined_001",
        "eligibility": {
            "eligible_rows": eligible_rows,
        },
        "planned_output": {
            "output_run": output_run,
        },
    }


def test_process_zarr_path_plans_training_archive(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(monkeypatch, {zarr_path: _FakeRoot(zarr_use="training")})

    monkeypatch.setattr(mod, "inspect_sam_subject_archive_path", lambda *_args, **_kwargs: _inspect_summary())

    row = mod._process_zarr_path(zarr_path, _default_options())  # noqa: SLF001

    assert row.status == "planned"
    assert row.observed_use == "training"
    assert row.crop_run == "crop_001"
    assert row.keypoint_group == "refined_keypoints_runs"
    assert row.keypoint_run == "refined_001"
    assert row.output_run == "sam_subject_masks_from_refined_001"
    assert row.eligible_rows == 5


def test_process_zarr_path_skips_existing_output_without_overwrite(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(
        monkeypatch,
        {
            zarr_path: _FakeRoot(
                zarr_use="training",
                subject_mask_runs={"sam_subject_masks_from_refined_001"},
            )
        },
    )

    monkeypatch.setattr(mod, "inspect_sam_subject_archive_path", lambda *_args, **_kwargs: _inspect_summary())

    row = mod._process_zarr_path(zarr_path, _default_options())  # noqa: SLF001

    assert row.status == "skipped_existing"
    assert row.reason == "subject_mask_runs/sam_subject_masks_from_refined_001 already exists"
    assert row.eligible_rows == 5


def test_process_zarr_path_skips_zero_eligible_rows_by_default(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(monkeypatch, {zarr_path: _FakeRoot(zarr_use="training")})

    monkeypatch.setattr(
        mod,
        "inspect_sam_subject_archive_path",
        lambda *_args, **_kwargs: _inspect_summary(eligible_rows=0),
    )

    row = mod._process_zarr_path(zarr_path, _default_options())  # noqa: SLF001

    assert row.status == "skipped_no_eligible"
    assert row.reason == "inspect reported zero eligible rows"
    assert row.eligible_rows == 0


def test_process_zarr_path_apply_runs_inference(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _install_fake_roots(monkeypatch, {zarr_path: _FakeRoot(zarr_use="training")})

    monkeypatch.setattr(mod, "inspect_sam_subject_archive_path", lambda *_args, **_kwargs: _inspect_summary())
    captured: dict[str, object] = {}

    def _fake_apply(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "crop_run": "crop_001",
            "keypoint_group": "refined_keypoints_runs",
            "keypoint_run": "refined_001",
            "output_run": "sam_subject_masks_from_refined_001",
            "rows_eligible": 5,
            "rows_segmented": 5,
            "rows_with_nonempty_masks": 4,
        }

    monkeypatch.setattr(mod, "run_sam_subject_mask_inference", _fake_apply)

    row = mod._process_zarr_path(
        zarr_path,
        _default_options(apply=True, batch_size=16, positive_keypoint_labels=("swim_bladder",)),
    )  # noqa: SLF001

    assert row.status == "updated"
    assert row.segmented_rows == 5
    assert row.nonempty_rows == 4
    assert captured["args"] == (zarr_path,)
    assert captured["kwargs"]["batch_size"] == 16
    assert captured["kwargs"]["positive_keypoint_labels"] == ("swim_bladder",)


def test_main_scans_roots_and_reports_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    training_path = tmp_path / "recording_training.zarr"
    analysis_path = tmp_path / "recording_analysis.zarr"
    _install_fake_roots(
        monkeypatch,
        {
            training_path: _FakeRoot(zarr_use="training"),
            analysis_path: _FakeRoot(zarr_use="analysis"),
        },
    )
    monkeypatch.setattr(mod, "_iter_zarr", lambda _roots, recursive: [training_path, analysis_path])

    monkeypatch.setattr(mod, "inspect_sam_subject_archive_path", lambda *_args, **_kwargs: _inspect_summary())

    rc = mod.main([str(tmp_path), "--recursive"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "planned" in out
    assert "filtered_zarr_use" in out
    assert "scope=training" in out
    assert "Results: updated=0 planned=1 skipped_existing=0 skipped_no_eligible=0" in out
