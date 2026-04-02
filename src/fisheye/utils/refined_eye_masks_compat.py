"""Helpers for identifying refined-eye compatibility artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Optional


DERIVED_REFINED_EYE_MASKS_COMPAT_ROLE = "derived_from_refined_subject_masks"


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_attrs(source: Any) -> Mapping[str, object]:
    attrs = getattr(source, "attrs", source)
    if isinstance(attrs, Mapping):
        return attrs
    return {}


def refined_eye_masks_compat_context(
    source: Any,
    *,
    base_stage: str = "refined_eye_masks_runs",
) -> dict[str, object]:
    attrs = _coerce_attrs(source)
    compatibility_role = _normalize_text(attrs.get("compatibility_role"))
    source_refined_subject_masks_run = _normalize_text(attrs.get("source_refined_subject_masks_run"))
    source_subject_mask_run = _normalize_text(attrs.get("source_subject_mask_run"))
    source_refined_eye_masks_run = _normalize_text(attrs.get("source_refined_eye_masks_run"))
    compat_refined_eye_masks_run = _normalize_text(attrs.get("compat_refined_eye_masks_run"))

    is_derived_compat = bool(
        compatibility_role == DERIVED_REFINED_EYE_MASKS_COMPAT_ROLE
        or source_refined_subject_masks_run is not None
    )

    return {
        "is_derived_compat": is_derived_compat,
        "compatibility_role": compatibility_role,
        "source_refined_subject_masks_run": source_refined_subject_masks_run,
        "source_subject_mask_run": source_subject_mask_run,
        "source_refined_eye_masks_run": source_refined_eye_masks_run,
        "compat_refined_eye_masks_run": compat_refined_eye_masks_run,
        "authority_stage_group": "refined_subject_masks_runs" if is_derived_compat else base_stage,
        "stage_role": "derived_compat" if is_derived_compat else "canonical",
        "stage_label": (
            f"{base_stage} (derived compat)"
            if is_derived_compat
            else base_stage
        ),
    }


def refined_eye_masks_redirect_hint(
    *,
    refined_run: Optional[str],
    source: Any,
    zarr_path: object | None = None,
) -> str:
    context = refined_eye_masks_compat_context(source)
    stage_label = str(context["stage_label"])
    run_label = refined_run or "<latest>"
    canonical_run = context.get("source_refined_subject_masks_run")
    zarr_label = str(zarr_path) if zarr_path is not None else "<zarr>"

    detail = f"{stage_label}/{run_label}"
    if canonical_run:
        detail += f" is derived from refined_subject_masks_runs/{canonical_run}"
    else:
        detail += " is a derived compatibility artifact"

    return (
        f"{detail}. Use `scripts/py -m fisheye.tune.eye_mask_review {zarr_label} --manual "
        f"--refined-run {run_label}` for canonical eye review/editing."
    )
