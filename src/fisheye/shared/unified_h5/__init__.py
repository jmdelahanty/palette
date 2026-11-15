"""Explicit, selector-ineligible admission for Citrus unified experimental H5 v1."""

from .common import PROFILE, UnifiedH5ContractError, contract_errors


UNIFIED_DEVELOPMENT_SCHEMA_ID = "citrus.experimental_h5_core_writer_test"


def declared_unified_profile(h5) -> str | None:
    """Return the unified artifact profile an open H5 declares, if any.

    Recognition only: a declared profile says the file is not a legacy v5/v6
    stimulus H5. It is not admission; ``validate_unified_h5_artifact`` still
    decides whether the exact profile and schema epoch are supported.
    """

    session = h5.get("/metadata/session")
    if session is not None and "recording_artifact_profile" in session.attrs:
        return str(session.attrs["recording_artifact_profile"])
    if h5.attrs.get("development_schema_id") == UNIFIED_DEVELOPMENT_SCHEMA_ID:
        return PROFILE
    return None


@contract_errors
def validate_unified_h5_artifact(h5, *, source_h5, finalization_receipt):
    from .admission import validate_artifact

    return validate_artifact(
        h5, source_h5=source_h5, finalization_receipt=finalization_receipt
    )


__all__ = [
    "PROFILE",
    "UNIFIED_DEVELOPMENT_SCHEMA_ID",
    "UnifiedH5ContractError",
    "declared_unified_profile",
    "validate_unified_h5_artifact",
]
