from __future__ import annotations

from pathlib import Path

import pytest

import fisheye.visualization.eye_angle_timeseries as timeseries


class _StopAfterPolicyHandoff(RuntimeError):
    pass


@pytest.mark.parametrize("legacy_compatibility", (False, True))
def test_eye_angle_timeseries_wrappers_propagate_legacy_policy(
    monkeypatch: pytest.MonkeyPatch,
    legacy_compatibility: bool,
) -> None:
    root = object()
    calls: list[tuple[str, object, bool]] = []
    monkeypatch.setattr(
        timeseries,
        "open_zarr_root",
        lambda _path, *, mode: root if mode == "r" else None,
    )

    def discover(
        received_root: object,
        *,
        legacy_compatibility: bool = False,
    ) -> list[object]:
        calls.append(("discover", received_root, legacy_compatibility))
        return []

    def catalog(
        received_root: object,
        *,
        run_name: str | None,
        prefer_frame: bool,
        legacy_compatibility: bool = False,
    ) -> object:
        del run_name, prefer_frame
        calls.append(("catalog", received_root, legacy_compatibility))
        return object()

    def window(
        received_root: object,
        **kwargs: object,
    ) -> None:
        calls.append(
            (
                "window",
                received_root,
                bool(kwargs["legacy_compatibility"]),
            )
        )
        raise _StopAfterPolicyHandoff

    def tables(
        received_root: object,
        **kwargs: object,
    ) -> None:
        calls.append(
            (
                "tables",
                received_root,
                bool(kwargs["legacy_compatibility"]),
            )
        )
        raise _StopAfterPolicyHandoff

    monkeypatch.setattr(
        timeseries,
        "discover_eye_angle_run_options_from_root",
        discover,
    )
    monkeypatch.setattr(timeseries, "catalog_eye_angle_series", catalog)
    monkeypatch.setattr(timeseries, "load_eye_angle_series_window", window)
    monkeypatch.setattr(timeseries, "load_eye_angle_run_tables", tables)

    assert timeseries.discover_eye_angle_run_options(
        Path("archive.zarr"),
        legacy_compatibility=legacy_compatibility,
    ) == []
    timeseries.catalog_eye_angle_timeseries_data(
        Path("archive.zarr"),
        legacy_compatibility=legacy_compatibility,
    )
    with pytest.raises(_StopAfterPolicyHandoff):
        timeseries.load_eye_angle_timeseries_window(
            Path("archive.zarr"),
            legacy_compatibility=legacy_compatibility,
        )
    with pytest.raises(_StopAfterPolicyHandoff):
        timeseries.load_eye_angle_timeseries_data(
            Path("archive.zarr"),
            legacy_compatibility=legacy_compatibility,
        )

    assert calls == [
        ("discover", root, legacy_compatibility),
        ("catalog", root, legacy_compatibility),
        ("window", root, legacy_compatibility),
        ("tables", root, legacy_compatibility),
    ]
