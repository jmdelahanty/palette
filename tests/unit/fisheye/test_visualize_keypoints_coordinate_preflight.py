from __future__ import annotations

from types import SimpleNamespace

import pytest

from fisheye.visualization import visualize_keypoints as mod


class _Group(dict):
    def __init__(self, *args, attrs=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})


def _root(
    *,
    latest: object = "run-a",
    latest_complete: object = "run-a",
    pointer: object = "keypoints_runs/run-a",
) -> _Group:
    parent = _Group(
        {"run-a": _Group(), "run-b": _Group()},
        attrs={"latest": latest, "latest_complete": latest_complete},
    )
    return _Group(
        {"keypoints_runs": parent},
        attrs={"current_keypoint_group_path": pointer},
    )


def test_implicit_selection_requires_one_matching_selector_tuple() -> None:
    assert mod._resolve_canonical_keypoint_selection(_root(), None) == "run-a"
    assert mod._resolve_canonical_keypoint_selection(_root(), "latest") == "run-a"


@pytest.mark.parametrize(
    ("latest", "latest_complete", "pointer"),
    [
        ("run-b", "run-a", "keypoints_runs/run-a"),
        ("run-a", None, "keypoints_runs/run-a"),
        ("run-a", "run-a", "keypoints_runs/run-b"),
        ("run-a", "run-a", None),
    ],
)
def test_implicit_selection_fails_closed_on_torn_or_missing_authority(
    latest: object,
    latest_complete: object,
    pointer: object,
) -> None:
    with pytest.raises(RuntimeError, match="Canonical keypoint selection"):
        mod._resolve_canonical_keypoint_selection(
            _root(
                latest=latest,
                latest_complete=latest_complete,
                pointer=pointer,
            ),
            None,
        )


@pytest.mark.parametrize(
    "shortcut",
    ["latest_traditional", "traditional", "latest_yolo", "yolo"],
)
def test_method_shortcuts_are_not_coordinate_authority(shortcut: str) -> None:
    with pytest.raises(RuntimeError, match="legacy inference"):
        mod._resolve_canonical_keypoint_selection(_root(), shortcut)


def test_explicit_historical_canonical_child_does_not_mean_latest() -> None:
    assert mod._resolve_canonical_keypoint_selection(_root(), "run-b") == "run-b"


def test_explicit_missing_child_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="is absent"):
        mod._resolve_canonical_keypoint_selection(_root(), "missing")


def test_interactive_helper_rejects_raw_default_before_root_navigation() -> None:
    with pytest.raises(RuntimeError, match="requires sealed canonical surfaces"):
        mod._preflight_interactive_keypoint_view(
            keypoint_run="run-a",
            crop_run="crop-a",
            canonical_surfaces=None,
            legacy_unverified=False,
        )


def test_raw_per_frame_helper_requires_explicit_legacy_mode() -> None:
    with pytest.raises(RuntimeError, match="legacy-unverified"):
        mod.get_record_for_frame(
            object(),  # type: ignore[arg-type]
            0,
            "run-a",
            "crop-a",
            ("a",),
        )


def test_interactive_helper_allows_raw_only_when_visibly_legacy() -> None:
    assert (
        mod._preflight_interactive_keypoint_view(
            keypoint_run="run-a",
            crop_run="crop-a",
            canonical_surfaces=None,
            legacy_unverified=True,
        )
        is None
    )


def test_interactive_helper_binds_exact_run_and_crop(monkeypatch) -> None:
    surfaces = SimpleNamespace(
        context=SimpleNamespace(
            run_path="keypoints_runs/run-a",
            source=SimpleNamespace(crop_path="crop_runs/crop-a"),
        )
    )
    monkeypatch.setattr(
        mod,
        "require_bound_keypoint_coordinate_surfaces",
        lambda value: value,
    )

    assert (
        mod._preflight_interactive_keypoint_view(
            keypoint_run="run-a",
            crop_run="crop-a",
            canonical_surfaces=surfaces,  # type: ignore[arg-type]
            legacy_unverified=False,
        )
        is surfaces
    )

    with pytest.raises(RuntimeError, match="requested keypoint run"):
        mod._preflight_interactive_keypoint_view(
            keypoint_run="run-b",
            crop_run="crop-a",
            canonical_surfaces=surfaces,  # type: ignore[arg-type]
            legacy_unverified=False,
        )
    with pytest.raises(RuntimeError, match="requested source crop"):
        mod._preflight_interactive_keypoint_view(
            keypoint_run="run-a",
            crop_run="crop-b",
            canonical_surfaces=surfaces,  # type: ignore[arg-type]
            legacy_unverified=False,
        )
