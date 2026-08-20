from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.zarr.metadata_cardinality import (
    MetadataCardinalityError,
    require_cardinality_independent_metadata,
)


def test_readable_fixed_policy_and_array_reference_are_allowed() -> None:
    profile = require_cardinality_independent_metadata(
        {
            "policy": {
                "policy_id": "occupancy_fraction_difference_v1",
                "formula": "treatment_fraction - baseline_fraction",
                "units": "fraction",
            },
            "source": {
                "run_path": "analysis/provider_occupancy_runs/pre",
                "array_path": "pooled/occupancy_fraction",
                "dtype": "<f8",
                "shape": [32, 32],
                "content_sha256": "a" * 64,
            },
        },
        forbidden_fields=("source_manifest_bindings", "acquisition_frames"),
        label="compact_provenance",
    )

    assert profile.serialized_bytes > 0
    assert profile.mapping_entry_count > 0


@pytest.mark.parametrize(
    "value, match",
    [
        (
            {"acquisition_frames": [0, 1, 2]},
            "cardinality-scaled",
        ),
        (
            {"values": np.asarray([0, 1, 2], dtype=np.int64)},
            "NumPy array",
        ),
    ],
)
def test_row_payloads_and_dense_arrays_are_rejected(value, match: str) -> None:
    with pytest.raises(MetadataCardinalityError, match=match):
        require_cardinality_independent_metadata(
            value,
            forbidden_fields=("acquisition_frames",),
            label="bad_provenance",
        )
