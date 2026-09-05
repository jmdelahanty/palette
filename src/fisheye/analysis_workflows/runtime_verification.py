"""Full-strength runtime verification for workflow-produced artifacts.

Metadata-only availability remains the planning boundary. After a command
returns successfully, this module reopens the exact output through the normal
scientific authority resolver and any profile-specific physical validator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fisheye.analysis.eye_angle_analysis import (
    validate_eye_angle_persisted_contract_manifests,
)
from fisheye.analysis.eye_angle_io import resolve_eye_angle_run
from fisheye.analysis.eye_angle_schema import validate_eye_angle_compact_run
from fisheye.analysis.subject_shape_io import resolve_canonical_subject_shape_run
from fisheye.analysis.subject_shape_storage import (
    is_subject_shape_access_aware_storage,
    validate_subject_shape_direct_consolidated_storage,
)
from fisheye.registry.stage_catalog import canonical_stage_id
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR,
    SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID,
    SUBJECT_SHAPE_BUNDLE_ID_ATTR,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_HISTORICAL_SOURCE_KIND,
    SUBJECT_SHAPE_SOURCE_KIND_ATTR,
    SUBJECT_SHAPE_STORAGE_PROFILE_ID_ATTR,
    SealedSubjectShapePublicationMetadataProof,
    validate_sealed_subject_shape_publication_metadata,
)
from fisheye.shared.eye_geometry_source import (
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    validate_staged_subject_shape_eye_geometry_authority,
)
from fisheye.shared.zarr_io import open_zarr_root

from .availability import StageAvailability, discover_stage_availability


@dataclass
class RuntimeVerificationSession:
    """Process-local cache for one immutable pre-execution admission pass."""

    zarr_path: Path
    _root: Any = field(default=None, init=False, repr=False)
    subject_shape_metadata_proofs: dict[
        str, SealedSubjectShapePublicationMetadataProof
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.zarr_path = Path(self.zarr_path).expanduser().resolve()

    def open_root(self, zarr_path: Path) -> Any:
        archive = Path(zarr_path).expanduser().resolve()
        if archive != self.zarr_path:
            raise ValueError("Runtime verification session cannot cross archives.")
        if self._root is None:
            self._root = open_zarr_root(archive, mode="r")
        return self._root


RuntimeStageVerifier = Callable[
    [
        Path,
        StageAvailability,
        Mapping[str, str],
        RuntimeVerificationSession | None,
    ],
    StageAvailability,
]


def _verify_subject_shape(
    zarr_path: Path,
    availability: StageAvailability,
    dependency_runs: Mapping[str, str],
    session: RuntimeVerificationSession | None,
) -> StageAvailability:
    run_name = availability.run_name
    if not availability.available or not isinstance(run_name, str):
        return availability
    try:
        root = (
            session.open_root(zarr_path)
            if session is not None
            else open_zarr_root(zarr_path, mode="r")
        )
        run_path = f"analysis/subject_shape_runs/{run_name}"
        run = root.get(run_path)
        if run is None:
            raise ValueError("Subject-shape publication is missing.")
        if SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR in run.attrs:
            publication_owner = run.attrs.get("subject_shape_publication_owner_uuid")
            if not isinstance(publication_owner, str) or not publication_owner:
                raise ValueError("Subject-shape publication lacks its immutable owner.")
            metadata_proof = validate_sealed_subject_shape_publication_metadata(
                root,
                run_path,
                expected_selector_eligible=True,
                expected_publication_owner=publication_owner,
            )
            if (
                metadata_proof.run_path != run_path
                or metadata_proof.selector_eligible is not True
                or metadata_proof.publication_owner != publication_owner
            ):
                raise ValueError(
                    "Subject-shape metadata resolver returned another publication."
                )
            if session is not None:
                session.subject_shape_metadata_proofs[run_name] = metadata_proof
        else:
            # Compatibility for publications predating sealed payload receipts.
            # The full scientific resolver remains fail-closed, albeit slower.
            run, resolved, resolved_path, publication = (
                resolve_canonical_subject_shape_run(root, run_name)
            )
            if (
                resolved != run_name
                or resolved_path != run_path
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
                or storage_profile_id != SUBJECT_SHAPE_ACCESS_AWARE_SUPPORTED_PROFILE_ID
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
            or run.attrs.get("source_refined_subject_masks_run") != source_authority
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


def _verify_eye_angles(
    zarr_path: Path,
    availability: StageAvailability,
    dependency_runs: Mapping[str, str],
    session: RuntimeVerificationSession | None,
) -> StageAvailability:
    run_name = availability.run_name
    if not availability.available or not isinstance(run_name, str):
        return availability
    try:
        source_subject_shape_run = dependency_runs.get("subject_shape")
        if (
            not isinstance(source_subject_shape_run, str)
            or not source_subject_shape_run
        ):
            raise ValueError(
                "Eye-angle verification requires its exact subject-shape authority."
            )
        root = (
            session.open_root(zarr_path)
            if session is not None
            else open_zarr_root(zarr_path, mode="r")
        )
        run, resolved, run_path = resolve_eye_angle_run(root, run_name)
        if resolved != run_name or run_path != f"analysis/eye_angle_runs/{run_name}":
            raise ValueError("Eye-angle resolver returned another run.")

        schema_issues = validate_eye_angle_compact_run(run)
        if schema_issues:
            summary = "; ".join(
                f"{issue.code}:{issue.path}:{issue.message}" for issue in schema_issues
            )
            raise ValueError(f"eye-angle payload contract failed: {summary}")
        attrs = dict(run.attrs)
        manifest_errors = validate_eye_angle_persisted_contract_manifests(attrs)
        if manifest_errors:
            raise ValueError(
                "eye-angle executable contract failed: " + "; ".join(manifest_errors)
            )
        if (
            attrs.get("source_eye_geometry_stage") != EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
            or attrs.get("source_eye_geometry_run") != source_subject_shape_run
            or attrs.get("source_subject_shape_run") != source_subject_shape_run
        ):
            raise ValueError(
                "Eye-angle publication does not match the planned subject-shape "
                "authority."
            )

        source_contracts = attrs.get("eye_angle_source_contracts")
        if not isinstance(source_contracts, Mapping):
            raise ValueError("Eye-angle source contracts are absent or malformed.")
        eye_geometry = source_contracts.get("eye_geometry")
        expected_source_path = (
            f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{source_subject_shape_run}"
        )
        if (
            not isinstance(eye_geometry, Mapping)
            or eye_geometry.get("available") is not True
            or eye_geometry.get("path") != expected_source_path
            or eye_geometry.get("stage_group") != EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
            or eye_geometry.get("run_name") != source_subject_shape_run
            or eye_geometry.get("source_subject_shape_run") != source_subject_shape_run
        ):
            raise ValueError(
                "Eye-angle geometry source contract names another subject-shape run."
            )
        source_authority = eye_geometry.get("source_authority")
        if not isinstance(source_authority, Mapping):
            raise ValueError(
                "Eye-angle geometry source contract lacks its staged authority."
            )
        validate_staged_subject_shape_eye_geometry_authority(
            root,
            run_name=source_subject_shape_run,
            authority=source_authority,
            verify_payload=False,
            publication_metadata_proof=(
                session.subject_shape_metadata_proofs.get(source_subject_shape_run)
                if session is not None
                else None
            ),
        )
    except Exception as exc:
        return StageAvailability(
            stage_id=availability.stage_id,
            available=False,
            artifact_path=availability.artifact_path,
            run_name=run_name,
            reason=(
                "strict eye-angle authority verification failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            completion_status=availability.completion_status,
        )
    return StageAvailability(
        stage_id=availability.stage_id,
        available=True,
        artifact_path=run_path,
        run_name=run_name,
        reason="strict dependency-bound eye-angle authority is available",
        completion_status=availability.completion_status,
    )


_RUNTIME_STAGE_VERIFIERS: Mapping[str, RuntimeStageVerifier] = {
    "eye_angles": _verify_eye_angles,
    "subject_shape": _verify_subject_shape,
}


def verify_persisted_stage_output(
    zarr_path: str | Path,
    stage_id: str,
    *,
    requested_run: str,
    dependency_runs: Mapping[str, str],
    session: RuntimeVerificationSession | None = None,
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
    return verifier(archive, availability, dependency_runs, session)


__all__ = ["RuntimeVerificationSession", "verify_persisted_stage_output"]
