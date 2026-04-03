from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import zstandard

from fisheye.utils import pack_zarr_transfer_artifact as mod


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _create_sample_zarr(zarr_path: Path) -> None:
    _write_file(zarr_path / "zarr.json", b'{"zarr_format":3}')
    _write_file(zarr_path / "raw_video" / "images_full" / "c" / "000", b"A" * 1024)
    _write_file(zarr_path / "subject_mask_runs" / "run_a" / "masks_roi" / "c" / "000", b"B" * 2048)
    _write_file(zarr_path / "refined_eye_masks_runs" / "run_b" / "masks_roi" / "c" / "000", b"C" * 512)


def _tar_names(artifact_path: Path) -> list[str]:
    with artifact_path.open("rb") as raw_fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(raw_fh) as reader:
            with tarfile.open(mode="r|", fileobj=reader) as tar:
                return [member.name for member in tar]


def test_process_zarr_path_plans_artifact(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _create_sample_zarr(zarr_path)
    options = mod.PackOptions(
        apply=False,
        recursive=False,
        overwrite=False,
        compression_level=3,
        out_dir=None,
        excluded_top_level=(),
        json=False,
    )

    row = mod._process_zarr(zarr_path, options)  # noqa: SLF001

    assert row.status == "planned"
    assert row.artifact_path.endswith(".tar.zst")
    assert row.source_files == 4
    assert row.source_bytes > 0


def test_apply_creates_artifact_manifest_and_checksum(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _create_sample_zarr(zarr_path)

    rc = mod.main([str(zarr_path), "--apply"])

    assert rc == 0
    artifact_path = zarr_path.with_name(f"{zarr_path.name}.tar.zst")
    manifest_path = artifact_path.with_name(f"{artifact_path.name}.manifest.json")
    checksum_path = artifact_path.with_name(f"{artifact_path.name}.sha256")
    assert artifact_path.exists()
    assert manifest_path.exists()
    assert checksum_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_format"] == "tar.zst"
    assert manifest["source_root_name"] == zarr_path.name
    assert manifest["source_files"] == 4
    assert manifest["compression"]["codec"] == "zstd"

    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert manifest["artifact_sha256"] == digest
    assert checksum_path.read_text(encoding="utf-8") == f"{digest}  {artifact_path.name}\n"

    names = _tar_names(artifact_path)
    assert f"{zarr_path.name}/zarr.json" in names
    assert f"{zarr_path.name}/raw_video/images_full/c/000" in names
    assert f"{zarr_path.name}/subject_mask_runs/run_a/masks_roi/c/000" in names


def test_apply_excludes_named_top_level_group(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _create_sample_zarr(zarr_path)

    rc = mod.main([str(zarr_path), "--exclude-top-level", "refined_eye_masks_runs", "--apply"])

    assert rc == 0
    artifact_path = zarr_path.with_name(f"{zarr_path.name}.tar.zst")
    manifest_path = artifact_path.with_name(f"{artifact_path.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["excluded_top_level_requested"] == ["refined_eye_masks_runs"]
    assert manifest["excluded_top_level_present"] == ["refined_eye_masks_runs"]
    assert "refined_eye_masks_runs" not in manifest["included_top_level"]
    assert manifest["source_files"] == 3

    names = _tar_names(artifact_path)
    assert f"{zarr_path.name}/refined_eye_masks_runs" not in names
    assert not any(name.startswith(f"{zarr_path.name}/refined_eye_masks_runs/") for name in names)


def test_existing_artifact_is_skipped_without_overwrite(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    _create_sample_zarr(zarr_path)
    artifact_path = zarr_path.with_name(f"{zarr_path.name}.tar.zst")
    artifact_path.write_bytes(b"existing")

    rc = mod.main([str(zarr_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "skipped_existing" in out
    assert "Results: updated=0 planned=0 skipped_existing=1" in out
