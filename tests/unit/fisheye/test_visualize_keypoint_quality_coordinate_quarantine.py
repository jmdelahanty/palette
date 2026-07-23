from __future__ import annotations

import pytest

from fisheye.visualization.visualize_keypoint_quality import _resolve_refined_run


class _Group(dict):
    def __init__(self, *args, attrs=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = dict(attrs or {})

    def group_keys(self):
        return self.keys()


def test_quality_diagnostic_requires_exact_run_by_default() -> None:
    parent = _Group({"a": object()}, attrs={"latest": "a"})

    with pytest.raises(ValueError, match="require one exact run name"):
        _resolve_refined_run(parent, None)


def test_exact_quality_diagnostic_run_is_not_implicit_authority() -> None:
    parent = _Group({"a": object()}, attrs={"latest": "a"})

    assert _resolve_refined_run(parent, "a") == "a"
    assert _resolve_refined_run(parent, "missing") is None


def test_legacy_latest_fallback_requires_explicit_opt_in() -> None:
    parent = _Group({"a": object(), "b": object()}, attrs={"latest": "a"})

    assert (
        _resolve_refined_run(
            parent,
            None,
            allow_legacy_latest_fallback=True,
        )
        == "a"
    )


def test_legacy_sorted_child_fallback_is_explicit_and_deterministic() -> None:
    parent = _Group({"a": object(), "b": object()})

    assert (
        _resolve_refined_run(
            parent,
            None,
            allow_legacy_latest_fallback=True,
        )
        == "b"
    )
