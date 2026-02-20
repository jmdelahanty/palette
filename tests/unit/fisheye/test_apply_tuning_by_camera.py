from __future__ import annotations

from pathlib import Path

import h5py
import zarr

from fisheye.utils import apply_tuning_by_camera as mod


def _make_zarr(
    path: Path,
    *,
    camera_id: str | None = None,
    zarr_use: str | None = None,
    tuning: dict[str, object] | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    if camera_id is not None:
        root.attrs["camera_id"] = camera_id
    if zarr_use is not None:
        root.attrs["zarr_use"] = zarr_use
    if tuning:
        analysis = root.create_group("analysis_metadata")
        for key, value in tuning.items():
            analysis.attrs[key] = value


def _read_tuning(path: Path, key: str) -> object:
    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    return analysis.attrs.get(key)


def test_candidate_h5_stems_adds_suffixless_variants() -> None:
    assert mod._candidate_h5_stems("recording_training") == ["recording_training", "recording"]
    assert mod._candidate_h5_stems("recording_analysis") == ["recording_analysis", "recording"]
    assert mod._candidate_h5_stems("recording") == ["recording"]


def test_camera_id_for_zarr_falls_back_to_suffixless_h5(tmp_path: Path) -> None:
    recording_dir = tmp_path / "rec"
    raw_dir = recording_dir / "raw"
    zarr_dir = recording_dir / "zarr"
    raw_dir.mkdir(parents=True)
    zarr_dir.mkdir(parents=True)

    h5_path = raw_dir / "example.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["camera_id"] = "2010093"

    zarr_path = zarr_dir / "example_training.zarr"
    _make_zarr(zarr_path, zarr_use="training")

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert mod._camera_id_for_zarr(zarr_path, root) == "2010093"


def test_main_apply_defaults_to_source_use_scope(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target_training = tmp_path / "rec_a" / "zarr" / "rec_a_training.zarr"
    target_analysis = tmp_path / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    source.parent.mkdir(parents=True)
    target_training.parent.mkdir(parents=True)
    target_analysis.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={"eye_mask_tuning": {"tuned_parameters": {"min_circularity": 0.77}}},
    )
    _make_zarr(target_training, camera_id="2010093", zarr_use="training")
    _make_zarr(target_analysis, camera_id="2010093", zarr_use="analysis")

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--keys",
            "eye_mask_tuning",
        ]
    )
    assert rc == 0
    assert _read_tuning(target_training, "eye_mask_tuning") == {
        "tuned_parameters": {"min_circularity": 0.77}
    }
    assert _read_tuning(target_analysis, "eye_mask_tuning") is None


def test_main_apply_can_target_analysis_use(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target_training = tmp_path / "rec_a" / "zarr" / "rec_a_training.zarr"
    target_analysis = tmp_path / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    source.parent.mkdir(parents=True)
    target_training.parent.mkdir(parents=True)
    target_analysis.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={"keypoint_tuning": {"threshold": 0.12}},
    )
    _make_zarr(target_training, camera_id="2010093", zarr_use="training")
    _make_zarr(target_analysis, camera_id="2010093", zarr_use="analysis")

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--zarr-use",
            "analysis",
            "--keys",
            "keypoint_tuning",
        ]
    )
    assert rc == 0
    assert _read_tuning(target_analysis, "keypoint_tuning") == {"threshold": 0.12}
    assert _read_tuning(target_training, "keypoint_tuning") is None

