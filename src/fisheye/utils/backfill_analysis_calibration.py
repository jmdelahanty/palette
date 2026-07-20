#!/usr/bin/env python3
"""Backfill normalized ``analysis/calibration`` groups from recording H5 files.

Default mode is dry-run. Use ``--apply`` to write updates. The intended batch
target is ``/nvme1/recordings``; each analysis archive is matched to its
stimulus H5 via ``analysis/stimulus_runs/<run>.attrs["source_h5"]`` when
available, then by the affiliated ``raw/*.h5`` file under the recording folder.

When an archive has no H5 source but was acquired with a known-matching rig
configuration, ``--donor-zarr`` can copy an existing calibration group from a
reference archive. Donor-derived backfills are explicitly stamped in attrs and
should be used only when the operator has verified the physical configuration.
"""

from __future__ import annotations

from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
    infer_zarr_use as _infer_zarr_use,
)
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import zarr

from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.selected_calibration import (
    VerifiedSelectedCameraSourceEvidence,
    build_selected_camera_source_evidence_from_h5_values,
)
from fisheye.shared.source_camera_physical_authority import (
    publish_source_camera_physical_authority,
)


LEGACY_CALIBRATION_ATTRS = (
    "measured_stimulus_fps",
    "measured_fps",
    "camera_id",
    "camera_model",
    "water_depth_mm",
)


@dataclass(frozen=True)
class BackfillPlan:
    zarr_path: Path
    status: str
    h5_path: Optional[Path] = None
    donor_zarr_path: Optional[Path] = None
    run_name: Optional[str] = None
    message: str = ""


def _resolve_roots(paths: Sequence[Path]) -> list[Path]:
    if paths:
        return [Path(path).expanduser() for path in paths]
    return [Path("/nvme1/recordings")]


def _open_zarr(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _group_keys(group: zarr.Group) -> list[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            pass
    return sorted(str(key) for key in group.keys() if isinstance(group.get(key), zarr.Group))


def _get_analysis_calibration(root: zarr.Group) -> Optional[zarr.Group]:
    analysis = root.get("analysis")
    if analysis is None:
        return None
    calibration = analysis.get("calibration")
    return calibration if isinstance(calibration, zarr.Group) else None


def _get_calibration_group(root: zarr.Group) -> Optional[zarr.Group]:
    calibration = _get_analysis_calibration(root)
    if calibration is not None:
        return calibration
    legacy = root.get("calibration")
    return legacy if isinstance(legacy, zarr.Group) else None


def _analysis_calibration_complete(root: zarr.Group) -> bool:
    calibration = _get_analysis_calibration(root)
    if calibration is None:
        return False
    attrs = calibration.attrs
    has_source = attrs.get("source_h5") not in (None, "")
    has_run = attrs.get("source_stimulus_run") not in (None, "")
    has_scale = attrs.get("pixel_to_mm") is not None or attrs.get("pixels_per_mm_camera") is not None
    has_homography = "homography_matrix" in calibration or attrs.get("homography_status") not in (None, "")
    return bool(has_source and has_run and has_scale and has_homography)


def _calibration_has_usable_scale(calibration: zarr.Group) -> bool:
    attrs = calibration.attrs
    return any(
        attrs.get(key) is not None
        for key in ("pixel_to_mm", "pixels_per_mm_camera", "pixels_per_mm")
    )


def _copy_attrs(source: Any, target: Any) -> None:
    target.attrs.update({str(key): value for key, value in source.attrs.items()})


def _copy_array(source: zarr.Array, target_group: zarr.Group, name: str) -> None:
    data = np.asarray(source[:])
    chunks = getattr(source, "chunks", None)
    kwargs: dict[str, Any] = {
        "data": data,
        "overwrite": True,
    }
    if chunks is not None:
        kwargs["chunks"] = chunks
    target_group.create_array(name, **kwargs)


def _copy_calibration_group_recursive(source: zarr.Group, target: zarr.Group) -> None:
    _copy_attrs(source, target)

    array_names = sorted(str(name) for name in source.array_keys())
    for name in array_names:
        _copy_array(source[name], target, name)

    group_names = sorted(str(name) for name in source.group_keys())
    for name in group_names:
        if name in target:
            del target[name]
        child = target.create_group(name)
        _copy_calibration_group_recursive(source[name], child)


def _source_label_for_calibration(root: zarr.Group, calibration: zarr.Group) -> str:
    source = calibration.attrs.get("source")
    source_h5 = calibration.attrs.get("source_h5")
    source_run = calibration.attrs.get("source_stimulus_run")
    zarr_id = root.attrs.get("recording_id") or root.attrs.get("recording_name")
    parts = []
    if zarr_id not in (None, ""):
        parts.append(f"recording={zarr_id}")
    if source not in (None, ""):
        parts.append(f"source={source}")
    if source_h5 not in (None, ""):
        parts.append(f"source_h5={source_h5}")
    if source_run not in (None, ""):
        parts.append(f"source_stimulus_run={source_run}")
    return "; ".join(parts)


def _donor_source_camera_evidence(
    donor_calibration: zarr.Group,
) -> VerifiedSelectedCameraSourceEvidence:
    source_h5_raw = donor_calibration.attrs.get("source_h5")
    camera_id_raw = (
        donor_calibration.attrs.get("active_camera_id")
        or donor_calibration.attrs.get("primary_camera_id")
    )
    if source_h5_raw in (None, "") or camera_id_raw in (None, ""):
        raise ValueError(
            "Donor calibration lacks source_h5 or active/primary camera identity."
        )
    source_h5 = Path(str(source_h5_raw)).expanduser().resolve()
    camera_id = str(camera_id_raw).strip()
    camera_path = f"/calibration_snapshot/{camera_id}"
    if not source_h5.is_file():
        raise FileNotFoundError(f"Donor calibration source H5 not found: {source_h5}")
    with h5py.File(source_h5, "r") as h5:
        arena_path = "/calibration_snapshot/arena_config_json"
        if arena_path not in h5 or camera_path not in h5:
            raise ValueError(
                "Donor source H5 lacks exact arena-config or selected-camera nodes."
            )
        arena_node = h5[arena_path]
        camera_node = h5[camera_path]
        if not isinstance(arena_node, h5py.Dataset) or not isinstance(
            camera_node, h5py.Group
        ):
            raise ValueError("Donor source H5 calibration nodes have invalid types.")
        return build_selected_camera_source_evidence_from_h5_values(
            source_h5_path=str(source_h5),
            arena_config_raw=arena_node[()],
            camera_group_path=camera_path,
            camera_group_attrs=dict(camera_node.attrs),
            expected_camera_id=camera_id,
        )


def _donor_calibration_plan(
    zarr_path: Path,
    *,
    root: zarr.Group,
    donor_zarr_path: Path,
    apply: bool,
    overwrite_existing: bool,
    operator_note: Optional[str],
    consolidate_metadata: bool,
) -> BackfillPlan:
    existing_complete = _analysis_calibration_complete(root)
    if existing_complete and not overwrite_existing:
        return BackfillPlan(zarr_path=zarr_path, status="skipped_existing", donor_zarr_path=donor_zarr_path)

    donor_root = _open_zarr(donor_zarr_path.expanduser(), mode="r")
    donor_calibration = _get_calibration_group(donor_root)
    if donor_calibration is None:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="missing_donor_calibration",
            donor_zarr_path=donor_zarr_path,
        )
    if not _calibration_has_usable_scale(donor_calibration):
        return BackfillPlan(
            zarr_path=zarr_path,
            status="donor_calibration_missing_scale",
            donor_zarr_path=donor_zarr_path,
        )
    if not isinstance(operator_note, str) or not operator_note.strip():
        return BackfillPlan(
            zarr_path=zarr_path,
            status="donor_operator_verification_required",
            donor_zarr_path=donor_zarr_path,
            message=(
                "A nonempty --donor-note is required to attest that camera, "
                "optics, resolution, and physical configuration match."
            ),
        )
    try:
        donor_evidence = _donor_source_camera_evidence(donor_calibration)
        donor_ppm_raw = donor_calibration.attrs.get("pixels_per_mm_camera")
        if donor_ppm_raw is None:
            pixel_to_mm_raw = donor_calibration.attrs.get("pixel_to_mm")
            donor_ppm_raw = (
                None
                if pixel_to_mm_raw is None
                else 1.0 / float(pixel_to_mm_raw)
            )
        if (
            donor_evidence.pixels_per_mm_camera is None
            or donor_ppm_raw is None
            or float(donor_ppm_raw) != donor_evidence.pixels_per_mm_camera
        ):
            raise ValueError(
                "Donor normalized scale differs from exact source-H5 camera evidence."
            )
        load_persisted_acquisition_camera_authority(
            root,
            expected_camera_id=donor_evidence.active_camera_id,
        )
    except Exception as exc:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="donor_physical_authority_unavailable",
            donor_zarr_path=donor_zarr_path,
            message=str(exc),
        )

    planned_status = "would_copy_donor_overwrite" if existing_complete else "would_copy_donor"
    written_status = "copied_donor_overwrite" if existing_complete else "copied_donor"
    if not apply:
        return BackfillPlan(
            zarr_path=zarr_path,
            status=planned_status,
            donor_zarr_path=donor_zarr_path,
            message=_source_label_for_calibration(donor_root, donor_calibration),
        )

    analysis = root.require_group("analysis")
    if "calibration" in analysis:
        del analysis["calibration"]
    target_calibration = analysis.create_group("calibration")
    _copy_calibration_group_recursive(donor_calibration, target_calibration)
    target_calibration.attrs["source"] = "donor_zarr_calibration"
    target_calibration.attrs["donor_zarr"] = str(donor_zarr_path.expanduser().resolve())
    target_calibration.attrs["donor_calibration_path"] = (
        "analysis/calibration" if _get_analysis_calibration(donor_root) is not None else "calibration"
    )
    target_calibration.attrs["donor_calibration_source"] = _source_label_for_calibration(
        donor_root,
        donor_calibration,
    )
    target_calibration.attrs["donor_configuration_verified_by_operator"] = True
    target_calibration.attrs["donor_backfill_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if operator_note:
        target_calibration.attrs["donor_backfill_note"] = str(operator_note)
    publish_source_camera_physical_authority(
        root,
        source_camera_evidence=donor_evidence,
        source_kind="operator_verified_donor_h5_calibration",
        provenance={
            "operator_verified": True,
            "donor_zarr": str(donor_zarr_path.expanduser().resolve()),
            "donor_source_h5": donor_evidence.source_h5_path,
            "operator_note": str(operator_note or ""),
        },
    )
    if consolidate_metadata:
        consolidate_metadata_capture_expected_warnings(zarr_path)

    return BackfillPlan(
        zarr_path=zarr_path,
        status=written_status,
        donor_zarr_path=donor_zarr_path,
        message=_source_label_for_calibration(donor_root, donor_calibration),
    )


def _select_stimulus_run(
    root: zarr.Group,
    *,
    requested_run: Optional[str] = None,
) -> tuple[Optional[str], Optional[zarr.Group]]:
    analysis = root.get("analysis")
    if analysis is None:
        return None, None
    runs_parent = analysis.get("stimulus_runs")
    if runs_parent is None:
        return None, None

    if requested_run:
        if requested_run in runs_parent:
            run = runs_parent[requested_run]
            return requested_run, run if isinstance(run, zarr.Group) else None
        return requested_run, None

    names: list[str] = []
    latest = runs_parent.attrs.get("latest")
    if latest is not None:
        names.append(str(latest))
    for name in reversed(_group_keys(runs_parent)):
        if name not in names:
            names.append(name)

    first_available: tuple[Optional[str], Optional[zarr.Group]] = (None, None)
    for name in names:
        if name not in runs_parent:
            continue
        run = runs_parent[name]
        if not isinstance(run, zarr.Group):
            continue
        if first_available == (None, None):
            first_available = (name, run)
        if run.attrs.get("source_h5") not in (None, ""):
            return name, run
    return first_available


def _source_h5_from_existing_calibration(root: zarr.Group) -> Optional[Path]:
    calibration = _get_analysis_calibration(root)
    if calibration is None:
        return None
    source = calibration.attrs.get("source_h5")
    if source in (None, ""):
        return None
    path = Path(str(source)).expanduser()
    return path if path.exists() and path.is_file() else None


def _source_h5_from_stimulus_run(run_group: Optional[zarr.Group]) -> Optional[Path]:
    if run_group is None:
        return None
    source = run_group.attrs.get("source_h5")
    if source in (None, ""):
        return None
    path = Path(str(source)).expanduser()
    return path if path.exists() and path.is_file() else None


def _recording_dir_for_zarr(zarr_path: Path) -> Optional[Path]:
    try:
        parent = zarr_path.resolve().parent
    except OSError:
        parent = zarr_path.parent
    if parent.name == "zarr":
        return parent.parent
    return None


def _source_h5_from_recording_raw(zarr_path: Path) -> tuple[Optional[Path], str]:
    recording_dir = _recording_dir_for_zarr(zarr_path)
    if recording_dir is None:
        return None, "zarr is not under a recording/zarr directory"

    raw_dir = recording_dir / "raw"
    candidates = sorted(path for suffix in ("*.h5", "*.hdf5") for path in raw_dir.glob(suffix))
    if not candidates:
        return None, f"no H5 files found under {raw_dir}"

    zarr_stem = zarr_path.name
    recording_stem = None
    if zarr_stem.endswith("_analysis.zarr"):
        recording_stem = zarr_stem[: -len("_analysis.zarr")]
    elif zarr_stem.endswith("_training.zarr"):
        recording_stem = zarr_stem[: -len("_training.zarr")]
    if recording_stem:
        preferred = [path for path in candidates if path.stem == recording_stem]
        if len(preferred) == 1:
            return preferred[0], ""

    if len(candidates) == 1:
        return candidates[0], ""

    joined = ", ".join(str(path) for path in candidates[:5])
    suffix = "" if len(candidates) <= 5 else f", ... ({len(candidates)} total)"
    return None, f"ambiguous H5 candidates: {joined}{suffix}"


def _resolve_source_h5(
    zarr_path: Path,
    root: zarr.Group,
    run_group: Optional[zarr.Group],
    *,
    explicit_h5_path: Optional[Path],
) -> tuple[Optional[Path], str]:
    if explicit_h5_path is not None:
        path = explicit_h5_path.expanduser()
        if path.exists() and path.is_file():
            return path, ""
        return None, f"explicit H5 path does not exist: {path}"

    for candidate in (
        _source_h5_from_stimulus_run(run_group),
        _source_h5_from_existing_calibration(root),
    ):
        if candidate is not None:
            return candidate, ""

    return _source_h5_from_recording_raw(zarr_path)


def _h5_has_calibration_snapshot(h5_path: Path) -> bool:
    with h5py.File(h5_path, "r") as h5:
        return "calibration_snapshot" in h5


def _materialize_analysis_calibration(
    root: zarr.Group,
    h5: h5py.File,
    *,
    source_h5: Path,
    run_name: str,
) -> None:
    """Write the legacy normalized calibration surface used by this repair CLI."""

    arena_node = h5["/calibration_snapshot/arena_config_json"]
    raw = arena_node[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    arena = json.loads(str(raw))
    if not isinstance(arena, dict):
        raise ValueError("Calibration arena config is not a JSON object.")
    camera_id = str(arena.get("active_camera_id") or "").strip()
    camera_records = arena.get("camera_calibrations")
    matches = [
        item
        for item in camera_records or []
        if isinstance(item, dict) and str(item.get("camera_id")) == camera_id
    ]
    if not camera_id or len(matches) != 1:
        raise ValueError("Calibration snapshot lacks one exact active-camera record.")
    camera_record = matches[0]
    camera_group = h5[f"/calibration_snapshot/{camera_id}"]

    analysis = root.require_group("analysis")
    calibration = analysis.require_group("calibration")
    calibration.attrs.update(
        {
            "schema_version": 1,
            "source": "h5_calibration_snapshot",
            "source_h5": str(source_h5.resolve()),
            "source_stimulus_run": run_name,
            "active_camera_id": camera_id,
            "primary_camera_id": camera_id,
        }
    )
    for name in (
        "pixels_per_mm_camera",
        "pixels_per_mm_projector",
        "real_world_ref_mm",
        "native_width_px",
        "native_height_px",
    ):
        value = camera_record.get(name, camera_group.attrs.get(name))
        if value is not None:
            calibration.attrs[name] = value
    ppm = calibration.attrs.get("pixels_per_mm_camera")
    if ppm is not None and float(ppm) > 0:
        calibration.attrs["pixel_to_mm"] = 1.0 / float(ppm)
    for name in (
        "experimental_area_center_x_px",
        "experimental_area_center_y_px",
        "experimental_area_radius_px",
        "experimental_area_radius_mm",
        "experimental_area_width_px",
        "experimental_area_height_px",
        "sub_arena_x_px",
        "sub_arena_y_px",
        "sub_arena_width_px",
        "sub_arena_height_px",
        "sub_arena_width_mm",
        "sub_arena_height_mm",
        "calculated_z_eff_mm",
        "experimental_area_shape",
    ):
        if arena.get(name) is not None:
            calibration.attrs[name] = arena[name]

    yaml_node = camera_group.get("homography_matrix_yml")
    if not isinstance(yaml_node, h5py.Dataset):
        calibration.attrs["homography_status"] = "missing_numeric_matrix"
        return
    yaml_raw = yaml_node[()]
    text = yaml_raw.decode("utf-8") if isinstance(yaml_raw, bytes) else str(yaml_raw)
    match = re.search(r"data\s*:\s*\[([^\]]+)\]", text, flags=re.DOTALL)
    if match is None:
        calibration.attrs["homography_status"] = "missing_numeric_matrix"
        return
    values = np.asarray(
        [float(item.strip()) for item in match.group(1).split(",")],
        dtype=np.float64,
    )
    if values.size != 9:
        raise ValueError("Calibration homography YAML does not contain nine values.")
    calibration.create_array(
        "homography_matrix",
        data=values.reshape(3, 3),
        chunks=(3, 3),
        overwrite=True,
    )
    calibration.attrs["homography_source"] = (
        f"calibration_snapshot/{camera_id}/homography_matrix_yml"
    )


def _copy_legacy_calibration_attrs(root: zarr.Group, *, overwrite_existing: bool) -> None:
    analysis_calibration = _get_analysis_calibration(root)
    legacy_calibration = root.get("calibration")
    if analysis_calibration is None or legacy_calibration is None:
        return
    for key in LEGACY_CALIBRATION_ATTRS:
        if key not in legacy_calibration.attrs:
            continue
        if not overwrite_existing and key in analysis_calibration.attrs:
            continue
        analysis_calibration.attrs[key] = legacy_calibration.attrs[key]


def plan_or_backfill_one(
    zarr_path: Path,
    *,
    apply: bool,
    overwrite_existing: bool,
    requested_run: Optional[str] = None,
    explicit_h5_path: Optional[Path] = None,
    donor_zarr_path: Optional[Path] = None,
    donor_note: Optional[str] = None,
    zarr_use: str = "analysis",
    consolidate_metadata: bool = False,
) -> BackfillPlan:
    mode = "a" if apply else "r"
    root = _open_zarr(zarr_path, mode=mode)

    observed_use = _infer_zarr_use(root, zarr_path)
    if zarr_use in {"analysis", "training"} and observed_use != zarr_use:
        return BackfillPlan(zarr_path=zarr_path, status="filtered_zarr_use", message=str(observed_use))

    if donor_zarr_path is not None:
        return _donor_calibration_plan(
            zarr_path,
            root=root,
            donor_zarr_path=donor_zarr_path,
            apply=apply,
            overwrite_existing=overwrite_existing,
            operator_note=donor_note,
            consolidate_metadata=consolidate_metadata,
        )

    existing_complete = _analysis_calibration_complete(root)
    if existing_complete and not overwrite_existing:
        return BackfillPlan(zarr_path=zarr_path, status="skipped_existing")

    run_name, run_group = _select_stimulus_run(root, requested_run=requested_run)
    if requested_run and run_group is None:
        return BackfillPlan(zarr_path=zarr_path, status="missing_stimulus_run", run_name=requested_run)

    h5_path, h5_message = _resolve_source_h5(
        zarr_path,
        root,
        run_group,
        explicit_h5_path=explicit_h5_path,
    )
    if h5_path is None:
        status = "ambiguous_h5" if h5_message.startswith("ambiguous") else "missing_h5"
        return BackfillPlan(zarr_path=zarr_path, status=status, run_name=run_name, message=h5_message)

    try:
        has_calibration_snapshot = _h5_has_calibration_snapshot(h5_path)
    except Exception as exc:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="h5_error",
            h5_path=h5_path,
            run_name=run_name,
            message=str(exc),
        )
    if not has_calibration_snapshot:
        return BackfillPlan(
            zarr_path=zarr_path,
            status="missing_calibration_snapshot",
            h5_path=h5_path,
            run_name=run_name,
        )

    planned_status = "would_overwrite" if existing_complete else "would_backfill"
    written_status = "overwritten" if existing_complete else "backfilled"
    source_run_name = run_name or "unknown"
    if not apply:
        return BackfillPlan(
            zarr_path=zarr_path,
            status=planned_status,
            h5_path=h5_path,
            run_name=source_run_name,
        )

    with h5py.File(h5_path, "r") as h5:
        _materialize_analysis_calibration(
            root,
            h5,
            source_h5=h5_path,
            run_name=source_run_name,
        )
    _copy_legacy_calibration_attrs(root, overwrite_existing=overwrite_existing)
    if consolidate_metadata:
        consolidate_metadata_capture_expected_warnings(zarr_path)

    return BackfillPlan(
        zarr_path=zarr_path,
        status=written_status,
        h5_path=h5_path,
        run_name=source_run_name,
    )


def _print_result(result: BackfillPlan, *, verbose: bool) -> None:
    noisy_statuses = {
        "would_backfill",
        "would_overwrite",
        "would_copy_donor",
        "would_copy_donor_overwrite",
        "backfilled",
        "overwritten",
        "copied_donor",
        "copied_donor_overwrite",
        "ambiguous_h5",
        "missing_h5",
        "missing_calibration_snapshot",
        "missing_donor_calibration",
        "donor_calibration_missing_scale",
        "donor_operator_verification_required",
        "donor_physical_authority_unavailable",
        "h5_error",
        "missing_stimulus_run",
    }
    if not verbose and result.status not in noisy_statuses:
        return

    parts = [result.status, str(result.zarr_path)]
    if result.h5_path is not None:
        parts.append(f"h5={result.h5_path}")
    if result.donor_zarr_path is not None:
        parts.append(f"donor_zarr={result.donor_zarr_path}")
    if result.run_name:
        parts.append(f"run={result.run_name}")
    if result.message:
        parts.append(result.message)
    print(": ".join(parts[:2]) + (" " + " ".join(parts[2:]) if len(parts) > 2 else ""))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or analysis Zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Archive scope (default: analysis). Use training or any to process training archives.",
    )
    parser.add_argument("--run-name", help="Use a specific analysis/stimulus_runs/<run> for provenance.")
    parser.add_argument("--h5-path", type=Path, help="Explicit H5 path. Intended for single-archive repairs.")
    parser.add_argument(
        "--donor-zarr",
        type=Path,
        help=(
            "Copy calibration from a known-matching donor Zarr and publish sealed "
            "recording physical authority. Requires --donor-note."
        ),
    )
    parser.add_argument(
        "--donor-note",
        help="Provenance note recorded when --donor-zarr is applied.",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="Rewrite complete analysis/calibration groups.")
    parser.add_argument("--apply", action="store_true", help="Write updates (default: dry-run).")
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="After each write, refresh Zarr consolidated metadata.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print skipped archives as well as candidates/errors.")
    args = parser.parse_args(argv)

    if args.consolidate_metadata and not args.apply:
        parser.error("--consolidate-metadata requires --apply")
    if args.h5_path is not None and len(args.paths) != 1:
        parser.error("--h5-path requires exactly one zarr path")
    if args.h5_path is not None and args.donor_zarr is not None:
        parser.error("--h5-path and --donor-zarr are mutually exclusive")
    if args.donor_note and args.donor_zarr is None:
        parser.error("--donor-note requires --donor-zarr")

    roots = _resolve_roots(args.paths)
    counts: dict[str, int] = {
        "zarr_scanned": 0,
        "errors": 0,
    }

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            result = plan_or_backfill_one(
                zarr_path,
                apply=bool(args.apply),
                overwrite_existing=bool(args.overwrite_existing),
                requested_run=args.run_name,
                explicit_h5_path=args.h5_path,
                donor_zarr_path=args.donor_zarr,
                donor_note=args.donor_note,
                zarr_use=str(args.zarr_use),
                consolidate_metadata=bool(args.consolidate_metadata),
            )
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")
            continue

        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status in {
            "h5_error",
            "donor_operator_verification_required",
            "donor_physical_authority_unavailable",
        }:
            counts["errors"] += 1
        _print_result(result, verbose=bool(args.verbose))

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    summary_keys = [
        "zarr_scanned",
        "would_backfill",
        "would_overwrite",
        "would_copy_donor",
        "would_copy_donor_overwrite",
        "backfilled",
        "overwritten",
        "copied_donor",
        "copied_donor_overwrite",
        "skipped_existing",
        "filtered_zarr_use",
        "missing_h5",
        "ambiguous_h5",
        "missing_stimulus_run",
        "missing_calibration_snapshot",
        "missing_donor_calibration",
        "donor_calibration_missing_scale",
        "donor_operator_verification_required",
        "donor_physical_authority_unavailable",
        "h5_error",
        "errors",
    ]
    summary = " ".join(f"{key}={counts.get(key, 0)}" for key in summary_keys)
    print(f"Analysis calibration backfill {mode}: {summary}")
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
