"""Tests for registry-driven detect preflight wrapper."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_detect_training_from_registry as wrapper


def _seed_dataset(db: Registry, dataset_id: str, zarr_path: Path) -> None:
    db.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        dataset_id,
        provenance={},
        context={"canvas_name": "DefaultScreen"},
        protocol_name=None,
        protocol_hash=None,
        acquisition={
            "dish_design": "cedar",
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    _seed_dataset(db, "dataset_1", zarr_path)
    db.close()


def _upsert_detect_quality(
    db: Registry,
    *,
    dataset_id: str,
    review_state: str = "approved",
    review_intended_use: str = "training",
    interpolated_detections_rate: float | None = 0.1,
    zarr_mtime_ns: int | None = 101,
) -> None:
    db.upsert_detect_quality(
        dataset_id=dataset_id,
        refined_run="refined_run_1",
        refined_created_utc=None,
        source_detect_run="detect_run_1",
        detect_method="detect",
        review_state=review_state,
        review_intended_use=review_intended_use,
        review_reviewer=None,
        review_timestamp_utc=None,
        review_resolved_group="filtered",
        total_detections=10,
        real_detections=9,
        interpolated_detections=1,
        interpolated_detections_rate=interpolated_detections_rate,
        zarr_mtime_ns=zarr_mtime_ns,
    )


def test_auto_set_name_is_generated_when_missing(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: list[list[str]] = []

    def fake_prepare(cli: list[str]) -> None:
        calls.append(list(cli))

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)

    rc = wrapper.main(["--registry", str(registry_path), "--dry-run"])
    assert rc == 0
    rc = wrapper.main(["--registry", str(registry_path), "--dry-run"])
    assert rc == 0
    assert len(calls) == 2

    first_cli = calls[0]
    second_cli = calls[1]
    assert "--set-name" in first_cli
    first_name = first_cli[first_cli.index("--set-name") + 1]
    second_name = second_cli[second_cli.index("--set-name") + 1]
    assert first_name == second_name
    assert first_name.startswith("cedar_defaultscreen_manual_gray_")
    assert len(first_name.rsplit("_", 1)[-1]) == 8


def test_auto_out_manifest_is_set_when_out_config_is_given(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    out_config = tmp_path / "prep" / "detect.yaml"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: list[list[str]] = []

    def fake_prepare(cli: list[str]) -> None:
        calls.append(list(cli))

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--out-config",
            str(out_config),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert calls
    cli = calls[0]
    assert "--out-manifest" in cli
    out_manifest = Path(cli[cli.index("--out-manifest") + 1])
    assert out_manifest == out_config.with_suffix(".manifest.json")


def test_registry_is_forwarded_to_preflight_without_register_flag(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry(registry_path, tmp_path / "dataset_1.zarr")

    calls: list[list[str]] = []

    def fake_prepare(cli: list[str]) -> None:
        calls.append(list(cli))

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)

    rc = wrapper.main(["--registry", str(registry_path), "--dry-run"])
    assert rc == 0
    assert calls
    cli = calls[0]
    assert "--registry" in cli
    assert str(registry_path) in cli


def test_rejects_legacy_train_flag() -> None:
    try:
        wrapper.main(["--train"])
    except SystemExit as exc:
        msg = str(exc)
        assert "prepare-only" in msg
        assert "run_detect_training_pipeline" in msg
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when --train is passed to prepare wrapper.")


def test_rejects_legacy_export_flags() -> None:
    try:
        wrapper.main(["--export-merged", "--merge-split", "0.8/0.2"])
    except SystemExit as exc:
        msg = str(exc)
        assert "prepare-only" in msg
        assert "--export-merged" in msg
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when merge flags are passed to prepare wrapper.")


@pytest.mark.parametrize(
    ("observed_patch", "expected_error"),
    [
        ({"source_detect_run": "wrong_detect_run"}, "source_detect_run divergence"),
        ({"zarr_mtime_ns": 202}, "detect_quality row is stale"),
    ],
)
def test_detect_quality_gate_rejects_stale_or_divergent_rows(
    tmp_path: Path,
    monkeypatch,
    observed_patch: dict[str, object],
    expected_error: str,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    dataset_path = tmp_path / "dataset_1.zarr"
    db = Registry(registry_path)
    _seed_dataset(db, "dataset_1", dataset_path)
    _upsert_detect_quality(db, dataset_id="dataset_1")
    db.close()

    observed = {
        "source_detect_run": "detect_run_1",
        "review_state": "approved",
        "review_intended_use": "training",
        "review_resolved_group": "filtered",
        "total_detections": 10,
        "real_detections": 9,
        "interpolated_detections": 1,
        "interpolated_detections_rate": 0.1,
        "zarr_mtime_ns": 101,
    }
    observed.update(observed_patch)

    def fake_resolve(zarr_path: Path, *, expected_refined_run: str) -> dict[str, object]:
        assert zarr_path == dataset_path
        assert expected_refined_run == "refined_run_1"
        return dict(observed)

    def fail_prepare(_cli: list[str]) -> None:  # pragma: no cover - defensive branch
        raise AssertionError("prepare_detect_training should not run when quality validation fails.")

    monkeypatch.setattr(wrapper, "_resolve_detect_quality_from_zarr", fake_resolve)
    monkeypatch.setattr(wrapper.pdt, "main", fail_prepare)

    with pytest.raises(ValueError, match=expected_error):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--max-interpolated-detections-rate",
                "0.2",
                "--dry-run",
            ]
        )


def test_detect_quality_exclusion_reasons_are_concrete(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    dataset_ids = [
        "missing_quality",
        "review_state_mismatch",
        "review_use_mismatch",
        "missing_interp_rate",
        "interp_rate_high",
    ]
    for dataset_id in dataset_ids:
        _seed_dataset(db, dataset_id, tmp_path / f"{dataset_id}.zarr")
    _upsert_detect_quality(
        db,
        dataset_id="review_state_mismatch",
        review_state="pending",
        review_intended_use="training",
        interpolated_detections_rate=0.1,
    )
    _upsert_detect_quality(
        db,
        dataset_id="review_use_mismatch",
        review_state="approved",
        review_intended_use="full_recording",
        interpolated_detections_rate=0.1,
    )
    _upsert_detect_quality(
        db,
        dataset_id="missing_interp_rate",
        review_state="approved",
        review_intended_use="training",
        interpolated_detections_rate=None,
    )
    _upsert_detect_quality(
        db,
        dataset_id="interp_rate_high",
        review_state="approved",
        review_intended_use="training",
        interpolated_detections_rate=0.8,
    )
    db.close()

    with pytest.raises(SystemExit, match="No datasets remain after detect quality filtering."):
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--require-review-state",
                "approved",
                "--require-review-intended-use",
                "training",
                "--max-interpolated-detections-rate",
                "0.3",
            ]
        )

    output = capsys.readouterr().out
    assert "Detect quality SQL filter excluded 5 dataset(s):" in output
    assert "missing_quality_row" in output
    assert "review_state_mismatch:pending!=approved" in output
    assert "review_use_mismatch:full_recording!=training" in output
    assert "missing_interpolated_detections_rate" in output
    assert "interpolated_rate_above_threshold:0.800000>0.300000" in output
