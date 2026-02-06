"""Tests for one-command preflight + merged export wrapper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils import prepare_detect_training_from_registry as wrapper


def test_export_merged_invokes_exporter(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    manifest_path = tmp_path / "out.manifest.json"
    db = Registry(registry_path)
    db.upsert_dataset(
        "dataset_1",
        session_uuid="dataset_1",
        zarr_path=tmp_path / "dataset_1.zarr",
    )
    db.upsert_provenance(
        "dataset_1",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={
            "has_images_ds": True,
            "has_images_ds_rgb": False,
            "downsample_formats_json": '["gray"]',
        },
        zarr_purpose=None,
    )
    db.close()

    calls: dict[str, list[str]] = {}

    def fake_prepare(cli: list[str]) -> None:
        calls["prepare"] = list(cli)
        manifest_path.write_text("{}", encoding="utf-8")

    def fake_export(cli: list[str]) -> int:
        calls["export"] = list(cli)
        return 0

    monkeypatch.setattr(wrapper.pdt, "main", fake_prepare)
    monkeypatch.setattr(wrapper.export_zarr, "main", fake_export)

    rc = wrapper.main(
        [
            "--registry",
            str(registry_path),
            "--out-manifest",
            str(manifest_path),
            "--export-merged",
            "--merge-out-zarr",
            str(tmp_path / "merged.zarr"),
            "--merge-split",
            "0.7/0.3",
            "--merge-seed",
            "123",
            "--merge-overwrite",
        ]
    )
    assert rc == 0

    prepare_cli = calls["prepare"]
    export_cli = calls["export"]
    assert str(tmp_path / "dataset_1.zarr") in prepare_cli
    assert "--out-manifest" in prepare_cli
    assert "--merge" in export_cli
    assert "--manifest" in export_cli
    assert str(manifest_path) in export_cli
    assert "--registry" in export_cli
    assert str(registry_path) in export_cli
    assert "--out-zarr" in export_cli
    assert str(tmp_path / "merged.zarr") in export_cli
    assert "--split" in export_cli
    assert "0.7/0.3" in export_cli
    assert "--seed" in export_cli
    assert "123" in export_cli
    assert "--overwrite" in export_cli


def test_export_merged_requires_out_manifest(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    db = Registry(registry_path)
    db.close()
    try:
        wrapper.main(
            [
                "--registry",
                str(registry_path),
                "--export-merged",
            ]
        )
    except SystemExit as exc:
        assert "--out-manifest" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for missing --out-manifest.")
