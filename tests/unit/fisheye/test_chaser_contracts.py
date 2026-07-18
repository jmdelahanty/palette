from __future__ import annotations

import pytest

from fisheye.analysis.chaser_contracts import (
    CanonicalChaserSet,
    ChaserIdentity,
    ChaserRoleInterval,
    canonical_chaser_set_from_protocol_payload,
)


def _payload(chasers: list[dict[str, object]]) -> dict[str, object]:
    return {"steps": [{"parameters": {"chasers": chasers}}]}


@pytest.mark.parametrize("chaser_count", [1, 2, 3])
def test_protocol_chaser_contract_supports_variable_length(chaser_count: int) -> None:
    chasers = [
        {
            "chaser_index": index,
            "enable_chase": index == 0,
            "enable_random_movement": index == 1,
            "color_r": index / max(1, chaser_count),
        }
        for index in range(chaser_count)
    ]

    result = canonical_chaser_set_from_protocol_payload(
        _payload(chasers),
        total_frames=101,
    )

    assert len(result.identities) == chaser_count
    assert len(result.role_intervals) == chaser_count
    assert [row["stimulus_instance_id"] for row in result.identity_rows()] == [
        f"chaser:{index}" for index in range(chaser_count)
    ]
    assert all(row["start_frame"] == 0 for row in result.role_rows())
    assert all(row["end_frame"] == 100 for row in result.role_rows())


def test_role_is_not_a_unique_key() -> None:
    result = canonical_chaser_set_from_protocol_payload(
        _payload(
            [
                {"chaser_index": 0, "enable_chase": False},
                {"chaser_index": 1, "enable_chase": False},
                {"chaser_index": 2, "enable_chase": False},
            ]
        )
    )

    assert [row.role for row in result.role_intervals] == ["inert", "inert", "inert"]


def test_role_intervals_may_change_but_cannot_overlap() -> None:
    identity = ChaserIdentity(
        stimulus_instance_id="chaser:0",
        chaser_index=0,
        source_track_key="chaser_index:0",
        raw_color_rgba=(1.0, 0.0, 0.0, 1.0),
    )
    valid = CanonicalChaserSet(
        identities=(identity,),
        role_intervals=(
            ChaserRoleInterval("chaser:0", 3, "inert", 0, 49, "test"),
            ChaserRoleInterval("chaser:0", 1, "aggressive", 50, None, "test"),
        ),
    )
    assert len(valid.role_intervals) == 2

    with pytest.raises(ValueError, match="overlapping role intervals"):
        CanonicalChaserSet(
            identities=(identity,),
            role_intervals=(
                ChaserRoleInterval("chaser:0", 3, "inert", 0, 50, "test"),
                ChaserRoleInterval("chaser:0", 1, "aggressive", 50, None, "test"),
            ),
        )
