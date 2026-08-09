"""Analysis adapter for the shared eye-angle logical schema."""

from fisheye.shared.eye_angle_schema import *  # noqa: F403
from fisheye.shared.eye_angle_schema import (
    validate_eye_angle_compact_run as _validate_eye_angle_compact_run,
)


def validate_eye_angle_compact_run(run_group):
    """Validate the logical contract plus the analysis-owned physical profile."""

    from fisheye.analysis.eye_angle_storage import validate_eye_angle_candidate_storage

    return _validate_eye_angle_compact_run(
        run_group,
        candidate_storage_validator=validate_eye_angle_candidate_storage,
    )
