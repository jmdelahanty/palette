"""Explicit, selector-ineligible admission for Citrus unified experimental H5 v1."""

from .common import PROFILE, UnifiedH5ContractError, contract_errors


@contract_errors
def validate_unified_h5_artifact(h5, *, source_h5, finalization_receipt):
    from .admission import validate_artifact

    return validate_artifact(
        h5, source_h5=source_h5, finalization_receipt=finalization_receipt
    )


__all__ = ["PROFILE", "UnifiedH5ContractError", "validate_unified_h5_artifact"]
