from pathlib import Path

from fisheye.utils import materialize_provider_behavior_chain as mod


def test_epoch_summary_uses_physical_filtered_speed(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    captured: dict[str, object] = {}

    def fake_materialize(source_zarr: Path, **kwargs: object) -> dict[str, object]:
        captured["source_zarr"] = source_zarr
        captured.update(kwargs)
        return {"status": "published"}

    monkeypatch.setattr(
        mod,
        "materialize_provider_epoch_behavior_summary",
        fake_materialize,
    )

    result = mod._summary(  # noqa: SLF001 - focused orchestration contract test
        {
            "analysis_zarr": str(archive),
            "output_runs": {
                "epoch_summary": "summary_v1",
                "stimulus_epochs_v2": "epochs_v2",
                "motion": "motion_v1",
                "swim_bouts": "bouts_v1",
            },
        },
        tmp_path / "scratch",
    )

    assert captured["source_zarr"] == archive
    assert captured["speed_level"] == mod.SUMMARY_SPEED_LEVEL == "filtered"
    assert result["status"] == "published"
