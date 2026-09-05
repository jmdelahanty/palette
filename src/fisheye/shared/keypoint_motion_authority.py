"""Resolve one keypoint/crop/body-frame authority for motion consumers.

The current keypoint bundle profile deliberately separates reviewed keypoint
observations from derived body-frame geometry.  Historical keypoint profiles
store both lineage attributes and ``heading`` on the keypoint run itself.
Consumers must not guess between those layouts.  This module owns the profile
dispatch and returns one common, fully identified motion source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import zarr

from fisheye.shared.zarr.body_frame_manifest import (
    body_frame_source_from_manifest,
    validate_body_frame_run_manifest,
)
from fisheye.shared.zarr.body_frame_schema import BODY_FRAME_SCHEMA_V1
from fisheye.shared.zarr.crop_manifest import validate_crop_run_manifest
from fisheye.shared.zarr.keypoint_bundle_activation import (
    resolve_active_keypoint_bundle_from_root,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_SCHEMA_ID,
    keypoint_crop_source_from_persisted,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_quality_manifest import (
    validate_keypoint_quality_run_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID,
    refined_keypoint_source_bindings_from_manifest,
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
)


class KeypointMotionAuthorityError(ValueError):
    """Raised when a keypoint selection cannot prove one motion authority."""


@dataclass(frozen=True)
class KeypointLineageDeclaration:
    """Profile-neutral keypoint-to-crop lineage parsed from persisted metadata."""

    run_name: str
    run_path: str
    is_refined: bool
    base_run_name: str
    base_run_path: str
    crop_run: str
    crop_path: str
    row_count: int | None
    profile_id: str
    run_manifest_digest: str | None = None
    base_manifest_digest: str | None = None
    crop_manifest_digest: str | None = None
    quality_run_name: str | None = None
    quality_run_path: str | None = None
    quality_manifest_digest: str | None = None


@dataclass(frozen=True)
class KeypointLineageAuthority:
    """One selected keypoint rowset plus its exact crop authority."""

    group: Any
    run_name: str
    is_refined: bool
    base_run_name: str
    crop_run: str
    profile_id: str = "legacy_keypoint_lineage_v1"


@dataclass(frozen=True)
class KeypointMotionAuthority(KeypointLineageAuthority):
    """One selected keypoint rowset plus its crop and heading authorities."""

    heading_group: Any | None = None
    heading_array_name: str = "heading"
    heading_group_path: str | None = None
    body_frame_run_name: str | None = None


@dataclass(frozen=True)
class _ResolvedLineageContext:
    authority: KeypointLineageAuthority
    declaration: KeypointLineageDeclaration
    active_bundle: Mapping[str, Any] | None
    active_refined: bool


def _run_name(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise KeypointMotionAuthorityError(f"{label} must be one safe run name.")
    return text


def _manifest(attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "run_manifest" not in attrs:
        return None
    value = attrs.get("run_manifest")
    if not isinstance(value, Mapping):
        raise KeypointMotionAuthorityError(
            "Declared run_manifest must be one persisted object."
        )
    return value


def keypoint_lineage_from_attributes(
    *,
    family: str,
    run_name: str,
    attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any] | None = None,
) -> KeypointLineageDeclaration:
    """Parse one supported keypoint lineage without opening array payloads."""

    name = _run_name(run_name, label="keypoint run")
    if family not in {"keypoints_runs", "refined_keypoints_runs"}:
        raise KeypointMotionAuthorityError(
            f"Unsupported keypoint run family: {family!r}."
        )
    path = f"{family}/{name}"
    manifest = _manifest(attrs)

    if family == "refined_keypoints_runs":
        if manifest is not None:
            if manifest.get("schema_id") != REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID:
                raise KeypointMotionAuthorityError(
                    "Refined keypoint run declares an unsupported manifest profile."
                )
            errors = validate_refined_keypoint_run_manifest(manifest)
            if errors:
                raise KeypointMotionAuthorityError(
                    "Refined keypoint manifest is invalid: " + "; ".join(errors)
                )
            payload = manifest.get("payload")
            if not isinstance(payload, Mapping) or payload.get("run_id") != name:
                raise KeypointMotionAuthorityError(
                    "Refined keypoint manifest names another run."
                )
            source_value = payload.get("source_bindings")
            if not isinstance(source_value, Mapping):
                raise KeypointMotionAuthorityError(
                    "Refined keypoint manifest lacks source bindings."
                )
            try:
                source = refined_keypoint_source_bindings_from_manifest(
                    source_value
                )
            except (TypeError, ValueError) as exc:
                raise KeypointMotionAuthorityError(
                    f"Refined keypoint source bindings are invalid: {exc}."
                ) from exc
            return KeypointLineageDeclaration(
                run_name=name,
                run_path=path,
                is_refined=True,
                base_run_name=source.raw_run_id,
                base_run_path=f"keypoints_runs/{source.raw_run_id}",
                crop_run=source.crop_run_id,
                crop_path=f"crop_runs/{source.crop_run_id}",
                row_count=int(source.dimensions.n_instances),
                profile_id="sealed_refined_keypoint_manifest_v1",
                run_manifest_digest=canonical_json_sha256(manifest),
                base_manifest_digest=source.raw_manifest_digest,
                crop_manifest_digest=source.crop_manifest_digest,
                quality_run_name=source.quality_run_id,
                quality_run_path=(
                    f"keypoint_quality_runs/{source.quality_run_id}"
                ),
                quality_manifest_digest=source.quality_manifest_digest,
            )

        base_name = _run_name(
            attrs.get("source_keypoints_run"),
            label="refined keypoint base run",
        )
        if raw_attrs is None:
            raise KeypointMotionAuthorityError(
                "Legacy refined keypoint lineage requires its base-run metadata."
            )
        base = keypoint_lineage_from_attributes(
            family="keypoints_runs",
            run_name=base_name,
            attrs=raw_attrs,
        )
        row_count_value = attrs.get("keypoints_processed")
        row_count = (
            int(row_count_value)
            if type(row_count_value) is int and row_count_value >= 0
            else base.row_count
        )
        return KeypointLineageDeclaration(
            run_name=name,
            run_path=path,
            is_refined=True,
            base_run_name=base.run_name,
            base_run_path=base.run_path,
            crop_run=base.crop_run,
            crop_path=base.crop_path,
            row_count=row_count,
            profile_id="legacy_refined_keypoint_attrs_v1",
            base_manifest_digest=base.run_manifest_digest,
            crop_manifest_digest=base.crop_manifest_digest,
        )

    if manifest is not None:
        if manifest.get("schema_id") != KEYPOINT_RUN_MANIFEST_SCHEMA_ID:
            raise KeypointMotionAuthorityError(
                "Raw keypoint run declares an unsupported manifest profile."
            )
        errors = validate_keypoint_run_manifest(manifest)
        if errors:
            raise KeypointMotionAuthorityError(
                "Raw keypoint manifest is invalid: " + "; ".join(errors)
            )
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping) or payload.get("run_id") != name:
            raise KeypointMotionAuthorityError(
                "Raw keypoint manifest names another run."
            )
        source_value = payload.get("source_crop_snapshot")
        if not isinstance(source_value, Mapping):
            raise KeypointMotionAuthorityError(
                "Raw keypoint manifest lacks its source crop snapshot."
            )
        try:
            source = keypoint_crop_source_from_persisted(source_value)
        except (TypeError, ValueError) as exc:
            raise KeypointMotionAuthorityError(
                f"Raw keypoint crop binding is invalid: {exc}."
            ) from exc
        return KeypointLineageDeclaration(
            run_name=name,
            run_path=path,
            is_refined=False,
            base_run_name=name,
            base_run_path=path,
            crop_run=source.run_id,
            crop_path=f"crop_runs/{source.run_id}",
            row_count=int(source.n_instances),
            profile_id="sealed_raw_keypoint_manifest_v2",
            run_manifest_digest=canonical_json_sha256(manifest),
            base_manifest_digest=canonical_json_sha256(manifest),
            crop_manifest_digest=source.manifest_digest,
        )

    crop_name = _run_name(
        attrs.get("source_crop_run"),
        label="keypoint source crop run",
    )
    row_count_value = attrs.get("keypoints_processed")
    row_count = (
        int(row_count_value)
        if type(row_count_value) is int and row_count_value >= 0
        else None
    )
    return KeypointLineageDeclaration(
        run_name=name,
        run_path=path,
        is_refined=False,
        base_run_name=name,
        base_run_path=path,
        crop_run=crop_name,
        crop_path=f"crop_runs/{crop_name}",
        row_count=row_count,
        profile_id="legacy_raw_keypoint_attrs_v1",
    )


def _group(root: Any, path: str, *, label: str) -> Any:
    try:
        value = root
        for part in (part for part in path.split("/") if part):
            value = value[part]
    except (AttributeError, KeyError, TypeError) as exc:
        raise KeypointMotionAuthorityError(f"{label} is absent: {path}.") from exc
    return value


def _optional_group(root: Any, path: str) -> Any | None:
    try:
        value = root
        for part in (part for part in path.split("/") if part):
            value = value[part]
        return value
    except (AttributeError, KeyError, TypeError):
        return None


def _require_complete(parent: Any, group: Any, *, label: str) -> None:
    try:
        complete = is_run_complete_in_parent(parent, group, legacy_default=False)
    except Exception as exc:
        raise KeypointMotionAuthorityError(
            f"Unable to validate {label} completion: {exc}."
        ) from exc
    if complete is not True:
        raise KeypointMotionAuthorityError(f"{label} is incomplete.")


def _selection(requested: str | None, active: Mapping[str, Any] | None, root: Any) -> tuple[str, str]:
    value = str(requested or "").strip().strip("/")
    if not value and active is not None:
        member = active.get("refined_keypoints")
        if isinstance(member, Mapping):
            return "refined_keypoints_runs", _run_name(
                member.get("run_id"), label="active refined keypoint run"
            )
    if value.startswith("refined/"):
        return "refined_keypoints_runs", _run_name(
            value.split("/", 1)[1], label="refined keypoint run"
        )
    if value.startswith("refined_keypoints_runs/"):
        return "refined_keypoints_runs", _run_name(
            value.split("/", 1)[1], label="refined keypoint run"
        )
    if value.startswith("keypoints_runs/"):
        return "keypoints_runs", _run_name(
            value.split("/", 1)[1], label="keypoint run"
        )
    if value:
        refined_parent = _optional_group(root, "refined_keypoints_runs")
        if refined_parent is not None and value in refined_parent:
            return "refined_keypoints_runs", _run_name(
                value, label="refined keypoint run"
            )
        return "keypoints_runs", _run_name(value, label="keypoint run")

    refined_parent = _optional_group(root, "refined_keypoints_runs")
    if refined_parent is not None:
        latest = refined_parent.attrs.get("latest")
        if latest:
            return "refined_keypoints_runs", _run_name(
                latest, label="latest refined keypoint run"
            )
    raw_parent = _optional_group(root, "keypoints_runs")
    if raw_parent is not None:
        latest = raw_parent.attrs.get("latest")
        if latest:
            return "keypoints_runs", _run_name(
                latest, label="latest keypoint run"
            )
    raise KeypointMotionAuthorityError("No keypoint authority is selected.")


def _manifest_digest_matches(
    group: Any,
    expected: str | None,
    *,
    label: str,
) -> None:
    if expected is None:
        return
    manifest = _manifest(group.attrs)
    if manifest is None or canonical_json_sha256(manifest) != expected:
        raise KeypointMotionAuthorityError(
            f"{label} manifest differs from the selected keypoint binding."
        )


def _resolve_keypoint_lineage_context(
    root: zarr.Group,
    requested: str | None,
) -> _ResolvedLineageContext:
    """Resolve the shared selection and keypoint-to-crop proof once."""

    try:
        active = resolve_active_keypoint_bundle_from_root(root)
    except Exception as exc:
        raise KeypointMotionAuthorityError(
            f"Active keypoint bundle authority is invalid: {exc}."
        ) from exc
    family, name = _selection(requested, active, root)
    parent = _group(root, family, label="keypoint run parent")
    group = _group(root, f"{family}/{name}", label="selected keypoint run")
    _require_complete(parent, group, label="Selected keypoint run")

    active_refined = False
    if active is not None:
        member = active.get("refined_keypoints")
        active_refined = (
            family == "refined_keypoints_runs"
            and isinstance(member, Mapping)
            and member.get("run_path") == f"{family}/{name}"
        )
    if not active_refined and not is_run_selector_eligible(group):
        raise KeypointMotionAuthorityError(
            f"Selected keypoint run is neither active nor selector-eligible: "
            f"{family}/{name}."
        )

    raw_attrs: Mapping[str, Any] | None = None
    if family == "refined_keypoints_runs" and _manifest(group.attrs) is None:
        raw_name = _run_name(
            group.attrs.get("source_keypoints_run"),
            label="refined keypoint base run",
        )
        raw_attrs = _group(
            root,
            f"keypoints_runs/{raw_name}",
            label="refined keypoint base run",
        ).attrs
    lineage = keypoint_lineage_from_attributes(
        family=family,
        run_name=name,
        attrs=group.attrs,
        raw_attrs=raw_attrs,
    )

    raw_parent = _group(root, "keypoints_runs", label="raw keypoint parent")
    raw_group = _group(root, lineage.base_run_path, label="raw keypoint source")
    _require_complete(raw_parent, raw_group, label="Raw keypoint source")
    raw_lineage = keypoint_lineage_from_attributes(
        family="keypoints_runs",
        run_name=lineage.base_run_name,
        attrs=raw_group.attrs,
    )
    if (
        raw_lineage.crop_run != lineage.crop_run
        or (
            raw_lineage.row_count is not None
            and lineage.row_count is not None
            and raw_lineage.row_count != lineage.row_count
        )
    ):
        raise KeypointMotionAuthorityError(
            "Raw and refined keypoint lineage do not identify one crop rowset."
        )
    crop_parent = _group(root, "crop_runs", label="crop run parent")
    crop_group = _group(root, lineage.crop_path, label="keypoint source crop")
    _require_complete(crop_parent, crop_group, label="Keypoint source crop")
    _manifest_digest_matches(
        raw_group,
        lineage.base_manifest_digest,
        label="Raw keypoint source",
    )
    _manifest_digest_matches(
        crop_group,
        lineage.crop_manifest_digest,
        label="Keypoint source crop",
    )
    if lineage.crop_manifest_digest is not None:
        crop_manifest = _manifest(crop_group.attrs)
        assert crop_manifest is not None
        crop_errors = validate_crop_run_manifest(crop_manifest)
        if crop_errors:
            raise KeypointMotionAuthorityError(
                "Keypoint source crop manifest is invalid: "
                + "; ".join(crop_errors)
            )

    if active_refined:
        assert active is not None
        expected_active_paths = {
            "raw_keypoints": lineage.base_run_path,
            "keypoint_quality": lineage.quality_run_path,
            "crop": lineage.crop_path,
        }
        for role, expected_path in expected_active_paths.items():
            member = active.get(role)
            if (
                not isinstance(member, Mapping)
                or expected_path is None
                or member.get("run_path") != expected_path
            ):
                raise KeypointMotionAuthorityError(
                    "Active keypoint bundle members differ from the selected "
                    f"refined-keypoint {role} binding."
                )
        quality_member = active["keypoint_quality"]
        assert isinstance(quality_member, Mapping)
        quality_group = quality_member.get("group")
        if quality_group is None:
            raise KeypointMotionAuthorityError(
                "Active keypoint bundle lacks its quality group."
            )
        _manifest_digest_matches(
            quality_group,
            lineage.quality_manifest_digest,
            label="Keypoint quality source",
        )
        quality_manifest = _manifest(quality_group.attrs)
        if quality_manifest is None:
            raise KeypointMotionAuthorityError(
                "Active keypoint quality source lacks its sealed manifest."
            )
        quality_errors = validate_keypoint_quality_run_manifest(quality_manifest)
        if quality_errors:
            raise KeypointMotionAuthorityError(
                "Active keypoint quality manifest is invalid: "
                + "; ".join(quality_errors)
            )

    return _ResolvedLineageContext(
        authority=KeypointLineageAuthority(
            group=group,
            run_name=name,
            is_refined=(family == "refined_keypoints_runs"),
            base_run_name=lineage.base_run_name,
            crop_run=lineage.crop_run,
            profile_id=(
                "active_refined_bundle_lineage_v1"
                if active_refined
                else lineage.profile_id
            ),
        ),
        declaration=lineage,
        active_bundle=active,
        active_refined=active_refined,
    )


def resolve_keypoint_lineage_authority(
    root: zarr.Group,
    requested: str | None,
) -> KeypointLineageAuthority:
    """Resolve one authorized keypoint rowset and its exact crop lineage."""

    return _resolve_keypoint_lineage_context(root, requested).authority


def resolve_keypoint_motion_authority(
    root: zarr.Group,
    requested: str | None,
) -> KeypointMotionAuthority:
    """Resolve one authorized keypoint source and its exact heading provider."""

    context = _resolve_keypoint_lineage_context(root, requested)
    authority = context.authority
    lineage = context.declaration
    active = context.active_bundle
    active_refined = context.active_refined
    group = authority.group
    name = authority.run_name

    if active_refined:
        assert active is not None
        body = active.get("body_frame")
        if not isinstance(body, Mapping):
            raise KeypointMotionAuthorityError(
                "Active keypoint bundle lacks its body-frame member."
            )
        body_group = body.get("group")
        body_run = _run_name(body.get("run_id"), label="body-frame run")
        body_path = f"analysis/body_frame_runs/{body_run}"
        if body_group is None or body.get("run_path") != body_path:
            raise KeypointMotionAuthorityError(
                "Active keypoint bundle body-frame path is invalid."
            )
        body_manifest = _manifest(body_group.attrs)
        if body_manifest is None:
            raise KeypointMotionAuthorityError(
                "Active body-frame run lacks its sealed manifest."
            )
        body_errors = validate_body_frame_run_manifest(body_manifest)
        if body_errors:
            raise KeypointMotionAuthorityError(
                "Active body-frame manifest is invalid: " + "; ".join(body_errors)
            )
        body_payload = body_manifest.get("payload")
        source_value = (
            body_payload.get("source_keypoint_snapshot")
            if isinstance(body_payload, Mapping)
            else None
        )
        if not isinstance(source_value, Mapping):
            raise KeypointMotionAuthorityError(
                "Active body-frame manifest lacks its keypoint source."
            )
        try:
            body_source = body_frame_source_from_manifest(source_value)
        except (TypeError, ValueError) as exc:
            raise KeypointMotionAuthorityError(
                f"Active body-frame keypoint binding is invalid: {exc}."
            ) from exc
        selected_manifest = _manifest(group.attrs)
        if (
            selected_manifest is None
            or body_source.run_path != lineage.run_path
            or body_source.run_name != lineage.run_name
            or body_source.manifest_digest
            != canonical_json_sha256(selected_manifest)
        ):
            raise KeypointMotionAuthorityError(
                "Active body-frame run does not bind the selected refined keypoints."
            )
        expected_arrays = set(BODY_FRAME_SCHEMA_V1.binding_paths)
        observed_arrays = set(str(value) for value in body_group.array_keys())
        observed_groups = set(str(value) for value in body_group.group_keys())
        if observed_arrays != expected_arrays or observed_groups:
            raise KeypointMotionAuthorityError(
                "Active body-frame run does not have the exact v1 topology."
            )
        logical_content = body_payload.get("logical_content")
        document = (
            logical_content.get("document")
            if isinstance(logical_content, Mapping)
            else None
        )
        declarations = (
            document.get("arrays") if isinstance(document, Mapping) else None
        )
        dimensions = (
            document.get("dimensions") if isinstance(document, Mapping) else None
        )
        if not isinstance(declarations, Mapping) or not isinstance(
            dimensions, Mapping
        ):
            raise KeypointMotionAuthorityError(
                "Active body-frame manifest lacks exact array dimensions."
            )
        body_rows = dimensions.get("n_instances")
        if (
            type(body_rows) is not int
            or body_rows < 0
            or (lineage.row_count is not None and body_rows != lineage.row_count)
        ):
            raise KeypointMotionAuthorityError(
                "Active body-frame row count differs from selected refined keypoints."
            )
        for leaf in expected_arrays:
            declaration = declarations.get(leaf)
            node = body_group[leaf]
            if (
                not isinstance(declaration, Mapping)
                or list(node.shape) != declaration.get("shape")
                or str(node.dtype) != declaration.get("dtype")
            ):
                raise KeypointMotionAuthorityError(
                    f"Active body-frame array {leaf!r} differs from its manifest."
                )
        return KeypointMotionAuthority(
            group=group,
            run_name=name,
            is_refined=True,
            base_run_name=lineage.base_run_name,
            crop_run=lineage.crop_run,
            heading_group=body_group,
            heading_array_name="heading_deg",
            heading_group_path=body_path,
            body_frame_run_name=body_run,
            profile_id="active_refined_bundle_body_frame_v1",
        )

    if "heading" not in group or "instance_key" not in group:
        raise KeypointMotionAuthorityError(
            "Selected legacy keypoint profile lacks its row-aligned heading authority."
        )
    return KeypointMotionAuthority(
        group=group,
        run_name=name,
        is_refined=authority.is_refined,
        base_run_name=lineage.base_run_name,
        crop_run=lineage.crop_run,
        heading_group=group,
        heading_array_name="heading",
        heading_group_path=lineage.run_path,
        body_frame_run_name=None,
        profile_id=lineage.profile_id,
    )


__all__ = [
    "KeypointLineageAuthority",
    "KeypointLineageDeclaration",
    "KeypointMotionAuthority",
    "KeypointMotionAuthorityError",
    "keypoint_lineage_from_attributes",
    "resolve_keypoint_lineage_authority",
    "resolve_keypoint_motion_authority",
]
