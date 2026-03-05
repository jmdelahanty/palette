"""Tests for eye masks batch registry discovery and CLI integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.utils import run_eye_masks_batch as mod


# ---------------------------------------------------------------------------
# _discover_zarrs_from_registry tests
# ---------------------------------------------------------------------------


def test_discover_zarrs_from_registry_skip_existing_passes_exclude_step_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When skip_existing=True, exclude_step_ok='eye_masks' and require_steps_ok=['crop', 'keypoints'] are passed."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    mod._discover_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
        skip_existing=True,
    )

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["exclude_step_ok"] == "eye_masks"
    assert captured_kwargs[0]["require_steps_ok"] == ["crop", "keypoints"]


def test_discover_zarrs_from_registry_no_skip_omits_exclude_step_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When skip_existing=False (default), exclude_step_ok is not passed."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr("fisheye.registry.db.Registry", _FakeRegistry)

    mod._discover_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
        skip_existing=False,
    )

    assert len(captured_kwargs) == 1
    assert "exclude_step_ok" not in captured_kwargs[0]
    assert captured_kwargs[0]["require_steps_ok"] == ["crop", "keypoints"]


# ---------------------------------------------------------------------------
# main() registry mode tests
# ---------------------------------------------------------------------------


def test_main_source_registry_missing_registry_fails(tmp_path: Path) -> None:
    """--source registry with missing registry file returns exit code 1."""
    rc = mod.main(
        [
            "--source",
            "registry",
            "--registry",
            str(tmp_path / "nonexistent.sqlite"),
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 1


def test_main_emit_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--emit-paths prints discovered paths and exits 0."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "_discover_zarrs_from_registry",
        lambda **_kw: [Path("/data/rec_a_analysis.zarr"), Path("/data/rec_b_analysis.zarr")],
    )

    rc = mod.main(
        [
            "--source",
            "registry",
            "--emit-paths",
            "--registry",
            str(registry_path),
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert "/data/rec_a_analysis.zarr" in lines
    assert "/data/rec_b_analysis.zarr" in lines
