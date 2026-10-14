from __future__ import annotations

from fisheye.shared import citrus_enums
from fisheye.shared.protocol_semantic_contract import (
    PROTOCOL_TRIAL_INDEX_ANGLE_V1_SCHEMA_VERSION,
    PROTOCOL_TRIAL_INDEX_SCHEMA_VERSIONS,
)
from fisheye.shared.stimulus_vocabulary import (
    STIMULUS_FAMILY,
    STIMULUS_MODE,
    STIMULUS_MODE_PLANNED,
    TRIAL_INDEX_ACCEPTED_VERSIONS,
)


def test_solid_color_is_live() -> None:
    assert STIMULUS_MODE_PLANNED == {}
    assert STIMULUS_MODE[22] == citrus_enums.STIMULUS_MODE[22] == "SOLID_COLOR"


def test_citrus_enums_mirror_is_the_live_vocabulary() -> None:
    assert citrus_enums.STIMULUS_MODE == dict(STIMULUS_MODE)
    assert set(STIMULUS_MODE) == {-1, *range(2, 23), 99}


def test_semantic_contract_accepts_the_contract_trial_index_versions() -> None:
    # v1/v2 are the legacy contract; v3 is accepted under the angle_v1 binding.
    assert (
        PROTOCOL_TRIAL_INDEX_SCHEMA_VERSIONS | {PROTOCOL_TRIAL_INDEX_ANGLE_V1_SCHEMA_VERSION}
        == TRIAL_INDEX_ACCEPTED_VERSIONS
    )


def test_solid_and_chaser_families() -> None:
    assert STIMULUS_FAMILY[4] == STIMULUS_FAMILY[5] == STIMULUS_FAMILY[22] == "solid_color"
    assert STIMULUS_FAMILY[12] == "chaser"
