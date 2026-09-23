"""Scoped component tests are not globally finalized artifact acceptance."""

from __future__ import annotations

import h5py
import pytest

from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.appearance import validate_appearance_component
from fisheye.shared.unified_h5.correspondence import validate_component_correspondence
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture


@pytest.mark.parametrize(
    "name", ["appearance_global_success", "appearance_reused_profile_new_chaser"]
)
def test_step_local_appearance_owner_positive(tmp_path, name):
    with h5py.File(emit_fixture(tmp_path, name), "r") as source:
        assert validate_appearance_component(source).row_count > 0
        if name.endswith("new_chaser"):
            assert (
                int(
                    source["/components/visual_appearance/states"][0][
                        "stimulus_frame_num"
                    ]
                )
                > 2**63
            )


@pytest.mark.parametrize(
    "name,reason",
    [
        ("appearance_duplicate_runtime_key", "appearance_runtime_key_duplicate"),
        ("appearance_duplicate_step_owner", "appearance_authored_step_owner_duplicate"),
        ("appearance_wrong_chaser_state", "appearance_runtime_chaser_state_unresolved"),
        ("appearance_wrong_profile", "appearance_runtime_profile_wrong_step"),
        ("appearance_wrong_reference", "appearance_replay_reference_mismatch"),
        ("appearance_wrong_step", "appearance_runtime_profile_wrong_step"),
    ],
)
def test_supplied_component_negatives(tmp_path, name, reason):
    with h5py.File(emit_fixture(tmp_path, name), "r") as source:
        with pytest.raises(UnifiedH5ContractError, match=reason):
            validate_appearance_component(source)


@pytest.mark.parametrize(
    "name", ["frames_only", "independent_motion_grid", "moving_grating"]
)
def test_non_chaser_correspondence_is_sufficient(tmp_path, name):
    with h5py.File(emit_fixture(tmp_path, name), "r") as source:
        summary = validate_component_correspondence(source)
        assert summary.mapped_frame_count > 0
        assert "chaser" not in summary.component_rows
        assert (name in summary.component_rows) == (name != "frames_only")
