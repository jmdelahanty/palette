#!/usr/bin/env python3
"""Audit refined mask metric surfaces in a Palette Zarr archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import zarr

from fisheye.shared.zarr.stage_arrays import (
    REFINED_EYE_MASKS_SPEC,
    REFINED_SUBJECT_COMPONENT_ARRAYS,
    REFINED_SUBJECT_COMPONENT_METRICS,
    REFINED_SUBJECT_EYE_CONTOURS,
    REFINED_SUBJECT_EYE_GEOMETRY,
    REFINED_SUBJECT_EYE_PAIR_METRICS,
    REFINED_SUBJECT_MASKS_SPEC,
    StageSpec,
    validate_run,
)

REFINED_MASK_STAGES = ("refined_eye_masks", "refined_subject_masks")
_EYE_COMPONENTS = ("eye_left", "eye_right")


def _group_keys(group: zarr.Group | None) -> list[str]:
    if group is None:
        return []
    return sorted(str(key) for key in group.keys())


def _selected_run_names(parent: zarr.Group | None, run_name: Optional[str], *, latest_only: bool) -> list[str]:
    if parent is None:
        return []
    if run_name:
        return [str(run_name)] if str(run_name) in parent else []
    if latest_only:
        latest = parent.attrs.get("latest")
        if latest is not None and str(latest) in parent:
            return [str(latest)]
        keys = _group_keys(parent)
        return keys[-1:] if keys else []
    return _group_keys(parent)


def _append_validation(
    *,
    group: zarr.Group,
    spec: StageSpec,
    path: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    result = validate_run(group, spec)
    errors.extend(f"{path}: {message}" for message in result.errors)
    warnings.extend(f"{path}: {message}" for message in result.warnings)


def _component_stage_spec(component_name: str) -> StageSpec:
    return StageSpec(
        stage_name=f"refined_subject_masks/components/{component_name}",
        zarr_group=f"components/{component_name}/",
        specs=REFINED_SUBJECT_COMPONENT_ARRAYS,
        subgroups={"metrics": REFINED_SUBJECT_COMPONENT_METRICS},
    )


def _eye_geometry_stage_spec(component_name: str) -> StageSpec:
    return StageSpec(
        stage_name=f"refined_subject_masks/components/{component_name}/geometry",
        zarr_group=f"components/{component_name}/geometry/",
        specs=REFINED_SUBJECT_EYE_GEOMETRY,
    )


def _eye_contour_stage_spec(component_name: str) -> StageSpec:
    return StageSpec(
        stage_name=f"refined_subject_masks/components/{component_name}/contours",
        zarr_group=f"components/{component_name}/contours/",
        specs=REFINED_SUBJECT_EYE_CONTOURS,
    )


def _eye_pair_metrics_stage_spec() -> StageSpec:
    return StageSpec(
        stage_name="refined_subject_masks/relations/eye_pair/metrics",
        zarr_group="relations/eye_pair/metrics/",
        specs=REFINED_SUBJECT_EYE_PAIR_METRICS,
    )


def _mask_labels(run_group: zarr.Group) -> set[str]:
    labels = run_group.attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)):
        return set()
    return {str(label) for label in labels}


def _audit_refined_eye_runs(
    root: zarr.Group,
    *,
    run_name: Optional[str],
    latest_only: bool,
    errors: list[str],
    warnings: list[str],
) -> list[str]:
    parent = root.get("refined_eye_masks_runs")
    run_names = _selected_run_names(parent, run_name, latest_only=latest_only)
    if parent is None:
        warnings.append("refined_eye_masks_runs: stage group missing")
        return []
    if run_name and not run_names:
        errors.append(f"refined_eye_masks_runs/{run_name}: run not found")
        return []

    for name in run_names:
        _append_validation(
            group=parent[name],
            spec=REFINED_EYE_MASKS_SPEC,
            path=f"refined_eye_masks_runs/{name}",
            errors=errors,
            warnings=warnings,
        )
    return run_names


def _audit_refined_subject_run(
    run_group: zarr.Group,
    *,
    run_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    run_path = f"refined_subject_masks_runs/{run_name}"
    _append_validation(
        group=run_group,
        spec=REFINED_SUBJECT_MASKS_SPEC,
        path=run_path,
        errors=errors,
        warnings=warnings,
    )

    components = run_group.get("components")
    if not isinstance(components, zarr.Group):
        errors.append(f"{run_path}: missing required subgroup 'components'")
        return

    component_names = _group_keys(components)
    if not component_names:
        errors.append(f"{run_path}/components: no component groups found")
        return

    for component_name in component_names:
        component_group = components[component_name]
        if not isinstance(component_group, zarr.Group):
            errors.append(f"{run_path}/components/{component_name}: expected group")
            continue
        _append_validation(
            group=component_group,
            spec=_component_stage_spec(component_name),
            path=f"{run_path}/components/{component_name}",
            errors=errors,
            warnings=warnings,
        )

    if not set(_EYE_COMPONENTS).issubset(_mask_labels(run_group)):
        return

    for component_name in _EYE_COMPONENTS:
        if component_name not in components:
            errors.append(f"{run_path}/components/{component_name}: missing eye component group")
            continue
        component_group = components[component_name]
        geometry = component_group.get("geometry")
        contours = component_group.get("contours")
        if not isinstance(geometry, zarr.Group):
            errors.append(f"{run_path}/components/{component_name}: missing required subgroup 'geometry'")
        else:
            _append_validation(
                group=geometry,
                spec=_eye_geometry_stage_spec(component_name),
                path=f"{run_path}/components/{component_name}/geometry",
                errors=errors,
                warnings=warnings,
            )
        if not isinstance(contours, zarr.Group):
            errors.append(f"{run_path}/components/{component_name}: missing required subgroup 'contours'")
        else:
            _append_validation(
                group=contours,
                spec=_eye_contour_stage_spec(component_name),
                path=f"{run_path}/components/{component_name}/contours",
                errors=errors,
                warnings=warnings,
            )

    relation_metrics = run_group.get("relations/eye_pair/metrics")
    if not isinstance(relation_metrics, zarr.Group):
        errors.append(f"{run_path}: missing required subgroup 'relations/eye_pair/metrics'")
    else:
        _append_validation(
            group=relation_metrics,
            spec=_eye_pair_metrics_stage_spec(),
            path=f"{run_path}/relations/eye_pair/metrics",
            errors=errors,
            warnings=warnings,
        )


def _audit_refined_subject_runs(
    root: zarr.Group,
    *,
    run_name: Optional[str],
    latest_only: bool,
    errors: list[str],
    warnings: list[str],
) -> list[str]:
    parent = root.get("refined_subject_masks_runs")
    run_names = _selected_run_names(parent, run_name, latest_only=latest_only)
    if parent is None:
        warnings.append("refined_subject_masks_runs: stage group missing")
        return []
    if run_name and not run_names:
        errors.append(f"refined_subject_masks_runs/{run_name}: run not found")
        return []

    for name in run_names:
        _audit_refined_subject_run(
            parent[name],
            run_name=name,
            errors=errors,
            warnings=warnings,
        )
    return run_names


def audit_refined_mask_metrics(
    root: zarr.Group,
    *,
    stage: str = "all",
    run_name: Optional[str] = None,
    latest_only: bool = False,
) -> dict[str, Any]:
    """Return a metadata-only audit summary for refined mask metric surfaces."""

    stage_key = str(stage).strip().lower()
    if stage_key not in {"all", *REFINED_MASK_STAGES}:
        raise ValueError(f"Unsupported stage {stage!r}; expected all, refined_eye_masks, or refined_subject_masks.")
    if run_name and stage_key == "all":
        raise ValueError("run_name requires stage='refined_eye_masks' or stage='refined_subject_masks'.")

    errors: list[str] = []
    warnings: list[str] = []
    audited: dict[str, list[str]] = {}

    if stage_key in {"all", "refined_eye_masks"}:
        audited["refined_eye_masks"] = _audit_refined_eye_runs(
            root,
            run_name=run_name,
            latest_only=latest_only,
            errors=errors,
            warnings=warnings,
        )

    if stage_key in {"all", "refined_subject_masks"}:
        audited["refined_subject_masks"] = _audit_refined_subject_runs(
            root,
            run_name=run_name,
            latest_only=latest_only,
            errors=errors,
            warnings=warnings,
        )

    return {
        "valid": not errors,
        "stage": stage_key,
        "latest_only": bool(latest_only),
        "audited_runs": audited,
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
        "errors": errors,
        "warnings": warnings,
    }


def audit_refined_mask_metrics_zarr(
    zarr_path: Path | str,
    *,
    stage: str = "all",
    run_name: Optional[str] = None,
    latest_only: bool = False,
) -> dict[str, Any]:
    root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="r")
    summary = audit_refined_mask_metrics(
        root,
        stage=stage,
        run_name=run_name,
        latest_only=latest_only,
    )
    summary["zarr_path"] = str(Path(zarr_path).expanduser().resolve())
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette Zarr archive to audit.")
    parser.add_argument(
        "--stage",
        choices=("all", *REFINED_MASK_STAGES),
        default="all",
        help="Refined mask stage family to audit.",
    )
    parser.add_argument("--run", dest="run_name", help="Specific run name to audit in the selected stage family.")
    parser.add_argument("--latest-only", action="store_true", help="Audit only the parent latest run.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON summary.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.run_name and args.stage == "all":
        parser.error("--run requires --stage refined_eye_masks or --stage refined_subject_masks.")

    summary = audit_refined_mask_metrics_zarr(
        args.zarr_path,
        stage=args.stage,
        run_name=args.run_name,
        latest_only=bool(args.latest_only),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif summary["valid"]:
        print(
            "Refined mask metric audit passed "
            f"({summary['error_count']} errors, {summary['warning_count']} warnings)."
        )
    else:
        print("Refined mask metric audit failed.")
        for message in summary["errors"]:
            print(f"- {message}")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
