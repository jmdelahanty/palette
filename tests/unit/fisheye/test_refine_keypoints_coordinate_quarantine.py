from __future__ import annotations

from types import SimpleNamespace

import pytest

from fisheye.refinement import refine_keypoints


def test_future_normal_refinement_fails_before_archive_open(monkeypatch) -> None:
    opened = False

    def _unexpected_open(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("archive must not be opened")

    monkeypatch.setattr(refine_keypoints.zarr, "open_group", _unexpected_open)

    with pytest.raises(
        refine_keypoints.RefinedKeypointCoordinatePublicationUnavailable,
        match="disabled for future-normal processing",
    ):
        refine_keypoints.create_refined_keypoint_run("would-be-mutated.zarr")

    assert opened is False


@pytest.mark.parametrize("value", [False, None, 0, 1, "true"])
def test_diagnostic_opt_in_requires_exact_true(value: object) -> None:
    with pytest.raises(
        refine_keypoints.RefinedKeypointCoordinatePublicationUnavailable
    ):
        refine_keypoints._require_refined_keypoint_publication_mode(
            allow_legacy_unverified_diagnostic_output=value,  # type: ignore[arg-type]
        )


def test_diagnostic_output_is_explicitly_unverified_and_ineligible() -> None:
    group = SimpleNamespace(attrs={})

    refine_keypoints._stamp_legacy_unverified_diagnostic_output(group)

    assert group.attrs == {
        "stage_selector_eligible": False,
        "coordinate_contract": (
            "palette.refined_keypoints.legacy_unverified_nonselector.v1"
        ),
        "legacy_unverified_diagnostic_output": True,
        "publication_scope": "historical_diagnostic_only",
    }
