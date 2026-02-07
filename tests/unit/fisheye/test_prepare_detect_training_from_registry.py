"""Tests for registry-driven detect preflight wrapper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_detect_training_from_registry as wrapper


def _seed_registry(registry_path: Path, zarr_path: Path) -> None:
    db = Registry(registry_path)
    db.upsert_dataset(
        "dataset_1",
        session_uuid="dataset_1",
        zarr_path=zarr_path,
    )
    db.upsert_provenance(
        "dataset_1",
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
    db.close()


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
