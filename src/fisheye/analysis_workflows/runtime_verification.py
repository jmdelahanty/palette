"""Full-strength runtime verification for workflow-produced artifacts.

Metadata-only availability remains the planning boundary. After a command
returns successfully, this module reopens the exact output through the normal
scientific authority resolver and any profile-specific physical validator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from fisheye.analysis.subject_shape_io import resolve_canonical_subject_shape_run
from fisheye.analysis.subject_shape_storage import (
    is_subject_shape_access_aware_storage,
    validate_subject_shape_direct_consolidated_storage,
)
from fisheye.registry.stage_catalog import canonical_stage_id
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID,
    SUBJECT_SHAPE_BUNDLE_ID_ATTR,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
    SUBJECT_SHAPE_SOURCE_KIND_ATTR,
    SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR,
)
from fisheye.shared.zarr_io import open_zarr_root

from .availability import StageAvailability, discover_stage_availability


RuntimeStageVerifier = Callable[
    [Path, StageAvailability, Mapping[str, str]], StageAvailability
]


def _verify_subject_shape(
    zarr_path: Path,
    availability: StageAvailability,
    dependency_runs: Mapping[str, str],
) -> StageAvailability:
    run_name = availability.run_name
    if not availability.available or not isinstance(run_name, str):
        return availability
    try:
        root = open_zarr_root(zarr_path, mode="r")
        run, resolved, run_path, publication = resolve_canonical_subject_shape_run(
            root,
            run_name,
        )
        if (
            resolved != run_name
            or run_path != f"analysis/subject_shape_runs/{run_name}"
            or publication.run_path != run_path
            or publication.selector_eligible is not True
        ):
            raise ValueError(
                "Subject-shape resolver returned another run or lifecycle state."
            )
        storage_profile_id = run.attrs.get(SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR)
        source_authority = dependency_runs.get("refined_subject_masks")
        if not isinstance(source_authority, str) or not source_authority:
            raise ValueError(
                "Subject-shape verification requires its refined-mask authority."
            )
        if source_authority.startswith("bundle/"):
            bundle_id = source_authority.removeprefix("bundle/")
            if not bundle_id or "/" in bundle_id:
                raise ValueError("Subject-shape bundle dependency is malformed.")
            if (
                run.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
                != SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
                or run.attrs.get(SUBJECT_SHAPE_BUNDLE_ID_ATTR) != bundle_id
                or storage_profile_id
                != SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID
            ):
                raise ValueError(
                    "Subject-shape publication does not match the planned bundle "
                    "authority and supported storage profile."
                )
        elif (
            run.attrs.get(
                SUBJECT_SHAPE_SOURCE_KIND_ATTR,
                SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
            )
            != SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND
            or run.attrs.get("source_refined_subject_masks_run")
            != source_authority
        ):
            raise ValueError(
                "Subject-shape publication does not match the planned refined-mask "
                "authority."
            )
        if storage_profile_id is not None:
            profile_id = str(storage_profile_id)
            if not is_subject_shape_access_aware_storage(profile_id):
                raise ValueError(
                    f"Unsupported subject-shape storage profile {profile_id!r}."
                )
            errors = validate_subject_shape_direct_consolidated_storage(
                zarr_path,
                run_path=run_path,
                phase="bound",
                expected_profile_id=profile_id,
            )
            if errors:
                raise ValueError(
                    "Subject-shape storage validation failed: " + "; ".join(errors)
                )
    except Exception as exc:
        return StageAvailability(
            stage_id=availability.stage_id,
            available=False,
            artifact_path=availability.artifact_path,
            run_name=run_name,
            reason=(
                "strict subject-shape authority verification failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            completion_status=availability.completion_status,
        )
    return StageAvailability(
        stage_id=availability.stage_id,
        available=True,
        artifact_path=run_path,
        run_name=run_name,
        reason="strict canonical subject-shape authority is available",
        completion_status=availability.completion_status,
    )


_RUNTIME_STAGE_VERIFIERS: Mapping[str, RuntimeStageVerifier] = {
    "subject_shape": _verify_subject_shape,
}


def verify_persisted_stage_output(
    zarr_path: str | Path,
    stage_id: str,
    *,
    requested_run: str,
    dependency_runs: Mapping[str, str],
) -> StageAvailability:
    """Verify one exact completed workflow output through the shared gate."""

    archive = Path(zarr_path).expanduser().resolve()
    canonical = canonical_stage_id(stage_id)
    availability = discover_stage_availability(
        archive,
        canonical,
        requested_run=requested_run,
        dependency_runs=dependency_runs,
    )
    verifier = _RUNTIME_STAGE_VERIFIERS.get(canonical)
    if verifier is None or not availability.available:
        return availability
    return verifier(archive, availability, dependency_runs)


__all__ = ["verify_persisted_stage_output"]
