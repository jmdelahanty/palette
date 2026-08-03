from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.columnar import read_columnar_dataset
from fisheye.shared.zarr_run_completion import mark_run_complete
from fisheye.analysis.megabouts_classifier import (
    MEGABOUTS_PREPROCESSED_INPUT_MODE,
    PALETTE_PREPARED_INPUT_MODE,
    MegaboutsRuntime,
    build_per_bout_classification_table,
    classify_megabouts_input_pack,
    write_megabouts_classification_run,
)
from fisheye.analysis.megabouts_classifier_inputs import build_megabouts_classifier_input_pack
from tests.unit.fisheye.test_megabouts_classifier_inputs import (
    _build_root,
    _install_verified_source_readers,
)


@pytest.fixture(autouse=True)
def _verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_source_readers(monkeypatch)


def _build_classifier_root() -> zarr.Group:
    root = _build_root()
    root["analysis/track_kinematics_runs/offline/tk_001"].attrs["fps"] = 60.0
    swim_parent = root["analysis/swim_bout_runs"]
    swim_run = swim_parent["bouts_001"]
    swim_run["speed_filtered"].attrs["fps"] = 60.0
    swim_run.attrs["stage_selector_eligible"] = True
    mark_run_complete(
        swim_run,
        parent_group=swim_parent,
        run_name="bouts_001",
    )
    return root


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8")
    return str(value)


class _FakeTrackingConfig:
    def __init__(self, *, fps: int, tracking: str) -> None:
        self.fps = int(fps)
        self.tracking = str(tracking)


class _FakeSegmentationConfig:
    def __init__(self, *, fps: int, bout_duration_ms: float) -> None:
        self.fps = int(fps)
        self.bout_duration_ms = float(bout_duration_ms)

    @property
    def bout_duration(self) -> int:
        return int(self.bout_duration_ms / 1000.0 * float(self.fps))


class _FakeClassifier:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __init__(self, tracking_cfg, segmentation_cfg, *, exclude_CS: bool = False, device=None) -> None:
        self.tracking_cfg = tracking_cfg
        self.segmentation_cfg = segmentation_cfg
        self.exclude_CS = bool(exclude_CS)
        self.device = device

    def run_classification(self, *, tail_array: np.ndarray, traj_array: np.ndarray) -> dict[str, np.ndarray]:
        self.calls.append((tail_array.copy(), traj_array.copy()))
        n = int(tail_array.shape[0])
        return {
            "cat": np.arange(n, dtype=np.int32) + 2,
            "subcat": np.arange(n, dtype=np.int32) + 12,
            "sign": np.full((n,), -1, dtype=np.int32),
            "proba": np.full((n,), 0.875, dtype=np.float32),
            "first_half_beat": np.full((n,), 3, dtype=np.int32),
        }


def _fake_runtime() -> MegaboutsRuntime:
    return MegaboutsRuntime(
        classifier_class=_FakeClassifier,
        tracking_config_class=_FakeTrackingConfig,
        segmentation_config_class=_FakeSegmentationConfig,
        category_names=("approach_swim", "slow1", "slow2"),
        package_version="fake-0",
        package_path="/fake/megabouts",
        source_repo="/fake/megabouts_repo",
        git_commit="fakecommit",
    )


def test_classify_megabouts_input_pack_sends_only_valid_windows() -> None:
    root = _build_classifier_root()
    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.9,
        max_consecutive_invalid_frames=1,
    )
    assert pack.valid_bout.tolist() == [True, False]
    _FakeClassifier.calls.clear()

    result = classify_megabouts_input_pack(pack, runtime=_fake_runtime())

    assert result.classified_indices.tolist() == [0]
    assert result.classif_results["cat"].tolist() == [2]
    assert len(_FakeClassifier.calls) == 1
    tail_array, traj_array = _FakeClassifier.calls[0]
    assert tail_array.shape == (1, 10, 4)
    assert traj_array.shape == (1, 3, 4)
    np.testing.assert_allclose(tail_array[0], pack.tail_array[0])
    np.testing.assert_allclose(traj_array[0], pack.traj_array[0])


def test_build_per_bout_classification_table_marks_invalid_rows_skipped() -> None:
    root = _build_classifier_root()
    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.9,
        max_consecutive_invalid_frames=1,
    )
    result = classify_megabouts_input_pack(pack, runtime=_fake_runtime())

    table = build_per_bout_classification_table(pack, result)

    assert table["source_bout_id"].tolist() == [11, 12]
    assert table["source_window_valid"].tolist() == [True, False]
    assert table["classified"].tolist() == [True, False]
    assert table["category_id"].tolist() == [2, -1]
    assert [_decode(value) for value in table["category_label_bytes"]] == [
        "slow2",
        "skipped_invalid_window",
    ]
    assert table["HB1_frame"].tolist() == [4, -1]
    assert np.isclose(table["probability"][0], 0.875)
    assert np.isnan(table["probability"][1])
    assert _decode(table["failure_reason_bytes"][0]) == "ok"
    assert _decode(table["failure_reason_bytes"][1]) == "traj_valid_fraction_below_threshold"


def test_write_megabouts_classification_run_persists_columnar_per_bout_table() -> None:
    source_root = _build_classifier_root()
    pack = build_megabouts_classifier_input_pack(
        source_root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.9,
        max_consecutive_invalid_frames=1,
    )
    result = classify_megabouts_input_pack(pack, runtime=_fake_runtime())
    out_root = zarr.group()

    run_name = write_megabouts_classification_run(
        out_root,
        run_name="megabouts_classifier_test",
        pack=pack,
        result=result,
    )

    assert run_name == "megabouts_classifier_test"
    parent = out_root["analysis/bout_classification_runs"]
    assert parent.attrs["latest"] == "megabouts_classifier_test"
    run = parent["megabouts_classifier_test"]
    assert run.attrs["schema_id"] == "analysis.bout_classification_runs"
    assert run.attrs["classifier_family"] == "megabouts"
    assert run.attrs["classifier_version"] == "fake-0"
    assert run.attrs["adapter_method"] == "palette_megabouts_direct_classifier"
    assert run.attrs["classifier_input_mode"] == PALETTE_PREPARED_INPUT_MODE
    assert run.attrs["megabouts_preprocessing"] is False
    assert run.attrs["megabouts_segmentation"] is False
    assert run.attrs["source_fps"] == 60.0
    assert run.attrs["window_frames"] == 4
    assert np.isclose(run.attrs["window_duration_s"], 4.0 / 60.0)
    assert run.attrs["megabouts_time_sampling"] is True
    assert run.attrs["parameters"]["classifier_input_mode"] == PALETTE_PREPARED_INPUT_MODE
    assert run.attrs["invalid_window_policy"] == "skip_invalid_windows"
    assert run.attrs["trajectory_conversion"]["alignment"] == "onset_translation_rotation"
    assert run.attrs["tail_angle_conversion"]["convention"] == "megabouts_cumulative_segment_angle"
    assert run.attrs["invalid_frame_policy"]["policy"] == "skip_invalid_windows"
    assert run.attrs["source_bout_count"] == 2
    assert run.attrs["classified_bout_count"] == 1
    persisted = read_columnar_dataset(run["per_bout"])
    assert persisted["category_id"].tolist() == [2, -1]
    assert persisted["valid"].tolist() == [True, False]
    assert [_decode(value) for value in persisted["category_label_bytes"]] == [
        "slow2",
        "skipped_invalid_window",
    ]


def test_write_megabouts_classification_run_preserves_preprocessed_input_mode() -> None:
    source_root = _build_classifier_root()
    pack = build_megabouts_classifier_input_pack(
        source_root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.9,
        max_consecutive_invalid_frames=1,
    )
    pack = replace(
        pack,
        parameters={
            **pack.parameters,
            "classifier_input_mode": MEGABOUTS_PREPROCESSED_INPUT_MODE,
            "megabouts_preprocessing": True,
            "megabouts_segmentation": False,
        },
    )
    result = classify_megabouts_input_pack(pack, runtime=_fake_runtime())
    out_root = zarr.group()

    write_megabouts_classification_run(
        out_root,
        run_name="megabouts_preprocessed_classifier_test",
        pack=pack,
        result=result,
    )

    run = out_root["analysis/bout_classification_runs/megabouts_preprocessed_classifier_test"]
    assert run.attrs["classifier_input_mode"] == MEGABOUTS_PREPROCESSED_INPUT_MODE
    assert run.attrs["megabouts_preprocessing"] is True
    assert run.attrs["megabouts_segmentation"] is False
    assert run.attrs["parameters"]["classifier_input_mode"] == MEGABOUTS_PREPROCESSED_INPUT_MODE
    assert run.attrs["parameters"]["megabouts_preprocessing"] is True


def test_classify_megabouts_input_pack_with_no_valid_windows_does_not_require_runtime() -> None:
    root = _build_classifier_root()
    pack = build_megabouts_classifier_input_pack(root, bout_duration_frames=4)
    assert pack.valid_bout.tolist() == [False, False]
    pack = replace(pack, tail_array=pack.tail_array.copy(), traj_array=pack.traj_array.copy())

    result = classify_megabouts_input_pack(pack)

    assert result.classified_indices.tolist() == []
    assert result.runtime is None
    assert result.classif_results["cat"].shape == (0,)
