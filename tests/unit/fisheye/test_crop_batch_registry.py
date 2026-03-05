"""Tests for crop_batch registry discovery and CLI integration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# crop_batch imports from tracking.crop which imports decord at module level.
# Mock decord before importing crop_batch to avoid the OSError on systems
# where the native library is unavailable.
if "decord" not in sys.modules:
    sys.modules["decord"] = MagicMock()

from fisheye.utils import crop_batch as mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_row(zarr_path: str) -> dict:
    return {"zarr_path": zarr_path}


# ---------------------------------------------------------------------------
# _discover_zarrs_from_registry tests
# ---------------------------------------------------------------------------


def test_discover_zarrs_from_registry_skip_existing_passes_exclude_step_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When skip_existing=True, exclude_step_ok='crop' and require_steps_ok=['detect'] are passed."""
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
    assert captured_kwargs[0]["exclude_step_ok"] == "crop"
    assert captured_kwargs[0]["require_steps_ok"] == ["detect"]


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
    assert captured_kwargs[0]["require_steps_ok"] == ["detect"]


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
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert "/data/rec_a_analysis.zarr" in lines
    assert "/data/rec_b_analysis.zarr" in lines
