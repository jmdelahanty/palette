"""Pipeline-aware inspector for raw and refined subject-mask runs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.mask_store import MaskStore, MaskStoreError, open_mask_store
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.zarr_io import open_zarr_root

WINDOW_NAME = "Subject Mask Inspector"
DISPLAY_SCALE = 2.0
COMPONENT_ORDER = ("subject_body", "eyes_union", "eye_left", "eye_right", "swim_bladder")
COMPONENT_ALIASES = {
    "body": "subject_body",
    "subject": "subject_body",
    "subject_body": "subject_body",
    "subject-body": "subject_body",
    "whole_subject": "subject_body",
    "whole-subject": "subject_body",
    "eye": "eyes_union",
    "eyes": "eyes_union",
    "eyes_union": "eyes_union",
    "eye-union": "eyes_union",
    "eye_left": "eye_left",
    "left_eye": "eye_left",
    "left-eye": "eye_left",
    "eye_right": "eye_right",
    "right_eye": "eye_right",
    "right-eye": "eye_right",
    "swimbladder": "swim_bladder",
    "swim_bladder": "swim_bladder",
    "swim-bladder": "swim_bladder",
}
COMPONENT_COLORS: dict[str, tuple[int, int, int]] = {
    "subject_body": (0, 220, 0),
    "swim_bladder": (255, 220, 0),
    "eyes_union": (255, 0, 255),
    "eye_left": (0, 255, 255),
    "eye_right": (255, 128, 255),
}
BBOX_THICKNESS = 1
CENTROID_RADIUS = 3
SUMMARY_LINE_HEIGHT = 18
SUMMARY_START_Y = 26
CONTOUR_THICKNESS = 1
HULL_THICKNESS = 1
HULL_COLOR = (80, 180, 255)
CONTOUR_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class LoadedMaskRun:
    stage_name: str
    run_name: str
    group: zarr.Group
    crop_run: str
    mask_labels: tuple[str, ...]
    available_channels: np.ndarray
    mask_store: MaskStore
    metrics: Any | None
    method: str | None
    source_subject_mask_run: str | None
    review_status: Mapping[str, object] | None
    component_reviews: Mapping[str, Mapping[str, object]]

    def read_component_mask(self, roi_idx: int, component_idx: int) -> np.ndarray:
        """Read one ROI-local component mask through the logical mask store."""

        return np.asarray(
            self.mask_store.read_dense(rows=int(roi_idx), channels=int(component_idx))[0, 0],
            dtype=np.uint8,
        )


@dataclass(frozen=True)
class InspectorThresholds:
    largest_component_fraction_threshold: float = 0.98
    sigma_noise_threshold: float = 2.5
    ipr_threshold: float = 10.0
    solidity_threshold: float = 0.85


def _require_gui_display() -> None:
    display = str(os.environ.get("DISPLAY") or "").strip()
    wayland_display = str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    if display or wayland_display:
        return
    raise RuntimeError(
        "No GUI display is available for the OpenCV subject-mask inspector. DISPLAY and WAYLAND_DISPLAY are unset."
    )


def _normalize_component_name(name: object) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    return COMPONENT_ALIASES.get(text, text)


def _resolve_crop_run(root: zarr.Group, preferred: str | None) -> str:
    crop_group, crop_run = resolve_zarr_run(
        root,
        "crop_runs",
        preferred,
        fallback_to_latest=True,
        run_label="Crop run",
    )
    if "roi_images" not in crop_group:
        raise RuntimeError(f"crop_runs/{crop_run} missing roi_images.")
    return crop_run


def _load_mask_run(
    root: zarr.Group,
    *,
    parent_path: str,
    run_name: str | None,
    run_label: str,
    required: bool,
) -> LoadedMaskRun | None:
    try:
        group, resolved_name = resolve_zarr_run(
            root,
            parent_path,
            run_name,
            fallback_to_latest=True,
            run_label=run_label,
        )
    except ValueError:
        if required:
            raise
        return None

    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"{parent_path}/{resolved_name} missing usable mask_labels attr.")
    available = group.get("available_channels")
    if available is None:
        raise RuntimeError(f"{parent_path}/{resolved_name} missing available_channels.")
    try:
        mask_store = open_mask_store(
            group,
            source_path=f"{parent_path}/{resolved_name}",
            prefer="dense",
        )
    except MaskStoreError as exc:
        raise RuntimeError(f"{parent_path}/{resolved_name} missing usable mask storage: {exc}") from exc
    crop_run = str(group.attrs.get("source_crop_run") or "")
    if not crop_run:
        crop_run = _resolve_crop_run(root, None)
    return LoadedMaskRun(
        stage_name=parent_path,
        run_name=resolved_name,
        group=group,
        crop_run=crop_run,
        mask_labels=tuple(str(item) for item in labels_raw),
        available_channels=np.asarray(available[:], dtype=bool),
        mask_store=mask_store,
        metrics=group.get("metrics"),
        method=str(group.attrs.get("method")) if group.attrs.get("method") is not None else None,
        source_subject_mask_run=(
            str(group.attrs.get("source_subject_mask_run")) if group.attrs.get("source_subject_mask_run") is not None else None
        ),
        review_status=(
            dict(group.attrs.get("refined_subject_mask_review_status") or {})
            if parent_path == "refined_subject_masks_runs"
            else None
        ),
        component_reviews=(
            dict(group.attrs.get("component_review_statuses") or {})
            if parent_path == "refined_subject_masks_runs"
            else {}
        ),
    )


def _load_runs(
    root: zarr.Group,
    *,
    subject_run: str | None,
    refined_run: str | None,
) -> tuple[LoadedMaskRun | None, LoadedMaskRun | None]:
    subject = _load_mask_run(
        root,
        parent_path="subject_mask_runs",
        run_name=subject_run,
        run_label="Subject-mask run",
        required=(subject_run is not None or refined_run is None),
    )

    refined = _load_mask_run(
        root,
        parent_path="refined_subject_masks_runs",
        run_name=refined_run,
        run_label="Refined subject-mask run",
        required=refined_run is not None,
    )

    if refined is None and refined_run is None and subject is not None:
        candidate = _load_mask_run(
            root,
            parent_path="refined_subject_masks_runs",
            run_name=None,
            run_label="Refined subject-mask run",
            required=False,
        )
        if candidate is not None and str(candidate.source_subject_mask_run or "") == subject.run_name:
            refined = candidate

    if subject is None and refined is not None and refined.source_subject_mask_run:
        subject = _load_mask_run(
            root,
            parent_path="subject_mask_runs",
            run_name=refined.source_subject_mask_run,
            run_label="Subject-mask run",
            required=True,
        )

    return subject, refined


def _to_bgr(roi_image: np.ndarray) -> np.ndarray:
    image = np.asarray(roi_image, dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and int(image.shape[2]) == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and int(image.shape[2]) == 3:
        return image.copy()
    raise ValueError(f"Unsupported ROI image shape for visualization: {tuple(int(v) for v in image.shape)}")


def _overlay_components(
    roi_image: np.ndarray,
    component_names: Sequence[str],
    masks: Sequence[np.ndarray],
) -> np.ndarray:
    display = _to_bgr(roi_image)
    overlay = display.copy()
    for component_name, mask in zip(component_names, masks):
        color = COMPONENT_COLORS.get(component_name, (255, 255, 255))
        overlay[np.asarray(mask, dtype=np.uint8) > 0] = color
    return cv2.addWeighted(overlay, 0.45, display, 0.55, 0)


def _panel(title: str, image: np.ndarray, footer_lines: Sequence[str] = ()) -> np.ndarray:
    panel = np.asarray(image, dtype=np.uint8).copy()
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    line_y = int(panel.shape[0]) - 16 - (18 * max(0, len(footer_lines) - 1))
    for line in footer_lines:
        cv2.putText(panel, str(line), (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        line_y += 18
    return panel


def _blank_panel_like(roi_image: np.ndarray, title: str, note: str) -> np.ndarray:
    panel = np.zeros((int(roi_image.shape[0]), int(roi_image.shape[1]), 3), dtype=np.uint8)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(panel, note, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return panel


def _component_index(run: LoadedMaskRun, component_name: str) -> int | None:
    if component_name not in run.mask_labels:
        return None
    idx = int(run.mask_labels.index(component_name))
    if idx >= int(run.available_channels.shape[0]) or not bool(run.available_channels[idx]):
        return None
    return idx


def _mask_for_component(run: LoadedMaskRun | None, component_name: str, roi_idx: int) -> np.ndarray | None:
    if run is None:
        return None
    comp_idx = _component_index(run, component_name)
    if comp_idx is None:
        return None
    return run.read_component_mask(int(roi_idx), comp_idx)


def _extract_largest_external_contour(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(np.count_nonzero(binary)) <= 0:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    if int(contour.shape[0]) < 3:
        return None
    return contour


def _draw_component_shape_overlay(
    image: np.ndarray,
    *,
    component_name: str,
    mask: np.ndarray | None,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    contour = _extract_largest_external_contour(mask)
    if contour is None:
        return canvas
    hull = cv2.convexHull(contour)
    if hull is not None and int(hull.shape[0]) >= 3:
        cv2.polylines(canvas, [np.asarray(hull, dtype=np.int32)], True, HULL_COLOR, HULL_THICKNESS, cv2.LINE_8)
    cv2.polylines(canvas, [np.asarray(contour, dtype=np.int32)], True, CONTOUR_COLOR, CONTOUR_THICKNESS, cv2.LINE_8)
    return canvas


def _component_geometry(
    run: LoadedMaskRun | None,
    component_name: str,
    roi_idx: int,
) -> tuple[np.ndarray, bool, np.ndarray, bool]:
    centroid_xy = np.asarray([0.0, 0.0], dtype=np.float32)
    bbox_xyxy = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if run is None or run.metrics is None:
        return centroid_xy, False, bbox_xyxy, False
    comp_idx = _component_index(run, component_name)
    if comp_idx is None:
        return centroid_xy, False, bbox_xyxy, False
    centroid_valid = False
    bbox_valid = False
    if "centroid_xy" in run.metrics:
        centroid_xy = np.asarray(run.metrics["centroid_xy"][int(roi_idx), comp_idx], dtype=np.float32)
    if "centroid_valid" in run.metrics:
        centroid_valid = bool(np.asarray(run.metrics["centroid_valid"][int(roi_idx), comp_idx], dtype=bool))
    if "bbox_xyxy" in run.metrics:
        bbox_xyxy = np.asarray(run.metrics["bbox_xyxy"][int(roi_idx), comp_idx], dtype=np.float32)
    if "bbox_valid" in run.metrics:
        bbox_valid = bool(np.asarray(run.metrics["bbox_valid"][int(roi_idx), comp_idx], dtype=bool))
    return centroid_xy, centroid_valid, bbox_xyxy, bbox_valid


def _draw_component_geometry(
    image: np.ndarray,
    *,
    run: LoadedMaskRun | None,
    component_name: str,
    roi_idx: int,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    centroid_xy, centroid_valid, bbox_xyxy, bbox_valid = _component_geometry(run, component_name, roi_idx)
    color = COMPONENT_COLORS.get(component_name, (255, 255, 255))
    if bbox_valid:
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox_xyxy.tolist()]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, BBOX_THICKNESS)
    if centroid_valid:
        cx, cy = [int(round(float(v))) for v in centroid_xy.tolist()]
        cv2.circle(canvas, (cx, cy), CENTROID_RADIUS, color, -1)
        cv2.circle(canvas, (cx, cy), CENTROID_RADIUS + 3, color, 1)
    return canvas


def _stage_summary_lines(run: LoadedMaskRun | None, component_name: str, roi_idx: int) -> list[str]:
    if run is None:
        return ["stage=missing"]
    comp_idx = _component_index(run, component_name)
    header = f"{run.stage_name}/{run.run_name}"
    method = str(run.method or "unknown")
    if comp_idx is None:
        return [header, f"method={method}", f"{component_name}: unavailable"]

    mask = run.read_component_mask(int(roi_idx), comp_idx)
    present = bool(np.count_nonzero(mask) > 0)
    area = float(np.count_nonzero(mask))
    centroid_xy, centroid_valid, bbox_xyxy, bbox_valid = _component_geometry(run, component_name, roi_idx)
    if run.metrics is not None:
        if "mask_present" in run.metrics:
            present = bool(np.asarray(run.metrics["mask_present"][int(roi_idx), comp_idx], dtype=bool))
        if "area_px" in run.metrics:
            area = float(np.asarray(run.metrics["area_px"][int(roi_idx), comp_idx], dtype=np.float32))

    lines = [
        header,
        f"method={method}",
        f"present={int(present)} area_px={area:.1f}",
        (
            f"centroid=({float(centroid_xy[0]):.1f}, {float(centroid_xy[1]):.1f})"
            if centroid_valid
            else "centroid=--"
        ),
        (
            "bbox=["
            f"{float(bbox_xyxy[0]):.1f},{float(bbox_xyxy[1]):.1f},"
            f"{float(bbox_xyxy[2]):.1f},{float(bbox_xyxy[3]):.1f}]"
            if bbox_valid
            else "bbox=--"
        ),
    ]

    if run.stage_name == "refined_subject_masks_runs":
        component_group = run.group.require_group("components").require_group(component_name)
        component_metrics = component_group.get("metrics")
        review_state = str(dict(run.component_reviews.get(component_name) or {}).get("state") or "pending")
        edit_applied = bool(np.asarray(component_group["edit_applied"][int(roi_idx)], dtype=bool))
        reason_labels = read_reason_labels(component_group)
        reason_label = str(reason_labels[int(roi_idx)]) if reason_labels is not None else "unknown"
        lines.extend(
            [
                f"review={review_state} edit_applied={int(edit_applied)}",
                f"reason={reason_label}",
            ]
        )
        if component_metrics is not None:
            component_count = int(np.asarray(component_metrics["component_count"][int(roi_idx)], dtype=np.int32))
            largest_fraction = float(
                np.asarray(component_metrics["largest_component_fraction"][int(roi_idx)], dtype=np.float32)
            )
            hole_count = int(np.asarray(component_metrics["hole_count"][int(roi_idx)], dtype=np.int32))
            hole_area_fraction = float(
                np.asarray(component_metrics["hole_area_fraction"][int(roi_idx)], dtype=np.float32)
            )
            sigma_noise = float(np.asarray(component_metrics["sigma_noise"][int(roi_idx)], dtype=np.float32))
            curvature_var = float(np.asarray(component_metrics["curvature_var"][int(roi_idx)], dtype=np.float32))
            ipr = float(np.asarray(component_metrics["ipr"][int(roi_idx)], dtype=np.float32))
            solidity = float(np.asarray(component_metrics["solidity"][int(roi_idx)], dtype=np.float32))
            lines.extend(
                [
                    f"components={component_count} largest_frac={largest_fraction:.3f}",
                    f"holes={hole_count} hole_area_frac={hole_area_fraction:.3f}",
                    f"sigma_noise={sigma_noise:.3f} curvature_var={curvature_var:.6f}",
                    f"ipr={ipr:.3f} solidity={solidity:.3f}",
                ]
            )
    return lines


def _component_flag_reasons(
    refined: LoadedMaskRun | None,
    component_name: str,
    roi_idx: int,
    *,
    thresholds: InspectorThresholds,
) -> list[str]:
    if refined is None or refined.stage_name != "refined_subject_masks_runs":
        return []
    comp_idx = _component_index(refined, component_name)
    if comp_idx is None:
        return []
    component_group = refined.group.require_group("components").require_group(component_name)
    component_metrics = component_group.get("metrics")
    if component_metrics is None:
        return []
    present = bool(np.asarray(refined.group["metrics/mask_present"][int(roi_idx), comp_idx], dtype=bool))
    if not present:
        return []
    reasons: list[str] = []
    component_count = int(np.asarray(component_metrics["component_count"][int(roi_idx)], dtype=np.int32))
    largest_fraction = float(np.asarray(component_metrics["largest_component_fraction"][int(roi_idx)], dtype=np.float32))
    hole_count = int(np.asarray(component_metrics["hole_count"][int(roi_idx)], dtype=np.int32))
    sigma_noise = float(np.asarray(component_metrics["sigma_noise"][int(roi_idx)], dtype=np.float32))
    ipr = float(np.asarray(component_metrics["ipr"][int(roi_idx)], dtype=np.float32))
    solidity = float(np.asarray(component_metrics["solidity"][int(roi_idx)], dtype=np.float32))
    if component_count > 1:
        reasons.append("component_count>1")
    if largest_fraction < float(thresholds.largest_component_fraction_threshold):
        reasons.append(f"largest_frac<{float(thresholds.largest_component_fraction_threshold):.2f}")
    if hole_count > 0:
        reasons.append("hole_count>0")
    if sigma_noise > float(thresholds.sigma_noise_threshold):
        reasons.append(f"sigma_noise>{float(thresholds.sigma_noise_threshold):.2f}")
    if ipr > float(thresholds.ipr_threshold):
        reasons.append(f"ipr>{float(thresholds.ipr_threshold):.2f}")
    if solidity < float(thresholds.solidity_threshold):
        reasons.append(f"solidity<{float(thresholds.solidity_threshold):.2f}")
    return reasons


def _flagged_roi_indices(
    refined: LoadedMaskRun | None,
    component_name: str,
    *,
    thresholds: InspectorThresholds,
) -> list[int]:
    if refined is None:
        return []
    total_rois = int(refined.mask_store.n_rows)
    return [
        roi_idx
        for roi_idx in range(total_rois)
        if _component_flag_reasons(refined, component_name, roi_idx, thresholds=thresholds)
    ]


def _summary_panel(
    roi_image: np.ndarray,
    *,
    component_name: str,
    roi_idx: int,
    total_rois: int,
    subject: LoadedMaskRun | None,
    refined: LoadedMaskRun | None,
    flag_reasons: Sequence[str],
) -> np.ndarray:
    panel = np.zeros((int(roi_image.shape[0]), int(roi_image.shape[1]), 3), dtype=np.uint8)
    lines = [
        f"ROI {int(roi_idx) + 1}/{int(total_rois)} component={component_name}",
        (
            f"flags={', '.join(str(reason) for reason in flag_reasons)}"
            if flag_reasons
            else "flags=none"
        ),
        "",
        *[f"RAW  {line}" for line in _stage_summary_lines(subject, component_name, roi_idx)],
        "",
        *[f"REF  {line}" for line in _stage_summary_lines(refined, component_name, roi_idx)],
    ]
    for idx, line in enumerate(lines):
        if not line:
            continue
        cv2.putText(
            panel,
            str(line),
            (10, SUMMARY_START_Y + idx * SUMMARY_LINE_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255) if not str(line).startswith("flags=") else (0, 220, 255),
            1,
        )
    cv2.putText(panel, "Summary", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return panel


def _component_names(subject: LoadedMaskRun | None, refined: LoadedMaskRun | None) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for component_name in COMPONENT_ORDER:
        declared = False
        for run in (subject, refined):
            if run is None:
                continue
            if component_name in run.mask_labels:
                declared = True
                break
        if declared and component_name not in seen:
            seen.add(component_name)
            result.append(component_name)
    for run in (subject, refined):
        if run is None:
            continue
        for component_name in run.mask_labels:
            if component_name not in seen:
                seen.add(component_name)
                result.append(component_name)
    return tuple(result)


def _cycle_component_index(active_idx: int, component_count: int, step: int) -> int:
    if component_count <= 0:
        return 0
    return int((int(active_idx) + int(step)) % int(component_count))


def launch_inspector(
    zarr_path: str | Path,
    *,
    subject_run: str | None = None,
    refined_run: str | None = None,
    crop_run: str | None = None,
    component: str | None = None,
    roi_index: int = 0,
    display_scale: float = DISPLAY_SCALE,
    thresholds: InspectorThresholds = InspectorThresholds(),
) -> None:
    root = open_zarr_root(zarr_path, mode="r")
    subject, refined = _load_runs(root, subject_run=subject_run, refined_run=refined_run)
    if subject is None and refined is None:
        raise RuntimeError("No subject_mask_runs or refined_subject_masks_runs available to inspect.")

    crop_run_name = crop_run or (subject.crop_run if subject is not None else refined.crop_run)  # type: ignore[union-attr]
    crop_group, crop_run_name = resolve_zarr_run(
        root,
        "crop_runs",
        crop_run_name,
        fallback_to_latest=True,
        run_label="Crop run",
    )
    if "roi_images" not in crop_group:
        raise RuntimeError(f"crop_runs/{crop_run_name} missing roi_images.")
    roi_images = crop_group["roi_images"]

    component_names = _component_names(subject, refined)
    if not component_names:
        raise RuntimeError("No overlapping available components found for subject-mask inspection.")
    active_component = _normalize_component_name(component) or component_names[0]
    if active_component not in component_names:
        raise RuntimeError(f"Component '{active_component}' is not available in the selected runs.")

    total_rois = int(roi_images.shape[0])
    current_pos = max(0, min(int(roi_index), total_rois - 1))
    active_idx = int(component_names.index(active_component))
    overlay_mode = "all"

    _require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1900, 1100)

    print("\nSubject Mask Inspector")
    if subject is not None:
        print(f"  Raw run: subject_mask_runs/{subject.run_name}")
    if refined is not None:
        print(f"  Refined run: refined_subject_masks_runs/{refined.run_name}")
    print(f"  Crop run: {crop_run_name}")
    print(f"  Components: {', '.join(component_names)}")
    print("Controls:")
    print("  1..9: select active component")
    print("  [/]: previous/next active component")
    print("  n/p: next/previous ROI")
    print("  j/k: jump +10 / -10 ROIs")
    print("  o: toggle overlay mode (all vs active component)")
    print("  f: jump to next flagged ROI for active component")
    print("  q/ESC: quit")

    def update_display() -> None:
        roi_img = np.asarray(roi_images[current_pos], dtype=np.uint8)
        if overlay_mode == "active":
            overlay_components = (component_names[active_idx],)
        else:
            overlay_components = component_names
        active_component_name = component_names[active_idx]
        raw_masks_lookup = {
            component_name: _mask_for_component(subject, component_name, current_pos)
            for component_name in overlay_components
        }
        refined_masks_lookup = {
            component_name: _mask_for_component(refined, component_name, current_pos)
            for component_name in overlay_components
        }
        raw_masks = [
            raw_masks_lookup[component_name]
            if raw_masks_lookup[component_name] is not None
            else np.zeros((int(roi_img.shape[0]), int(roi_img.shape[1])), dtype=np.uint8)
            for component_name in overlay_components
        ]
        refined_masks = [
            refined_masks_lookup[component_name]
            if refined_masks_lookup[component_name] is not None
            else np.zeros((int(roi_img.shape[0]), int(roi_img.shape[1])), dtype=np.uint8)
            for component_name in overlay_components
        ]

        crop_panel = _panel(
            "Crop ROI",
            _to_bgr(roi_img),
            footer_lines=(
                f"overlay_mode={overlay_mode}",
                f"active_component={active_component_name} ({active_idx + 1}/{len(component_names)})",
            ),
        )
        raw_panel = (
            _panel(
                f"Raw Overlay: {subject.run_name}",
                _draw_component_shape_overlay(
                    _draw_component_geometry(
                        _overlay_components(roi_img, overlay_components, raw_masks),
                        run=subject,
                        component_name=active_component_name,
                        roi_idx=current_pos,
                    ),
                    component_name=active_component_name,
                    mask=raw_masks_lookup.get(active_component_name),
                ),
                footer_lines=(f"method={subject.method or 'unknown'}",),
            )
            if subject is not None
            else _blank_panel_like(roi_img, "Raw Overlay", "No raw subject-mask run")
        )
        refined_panel = (
            _panel(
                f"Refined Overlay: {refined.run_name}",
                _draw_component_shape_overlay(
                    _draw_component_geometry(
                        _overlay_components(roi_img, overlay_components, refined_masks),
                        run=refined,
                        component_name=active_component_name,
                        roi_idx=current_pos,
                    ),
                    component_name=active_component_name,
                    mask=refined_masks_lookup.get(active_component_name),
                ),
                footer_lines=(
                    f"review={str(dict(refined.component_reviews.get(active_component_name) or {}).get('state') or 'pending')}",
                    "active_outline=contour+hull",
                ),
            )
            if refined is not None
            else _blank_panel_like(roi_img, "Refined Overlay", "No matching refined subject-mask run")
        )
        flags = _component_flag_reasons(refined, active_component_name, current_pos, thresholds=thresholds)
        summary_panel = _summary_panel(
            roi_img,
            component_name=active_component_name,
            roi_idx=current_pos,
            total_rois=total_rois,
            subject=subject,
            refined=refined,
            flag_reasons=flags,
        )
        combined = np.vstack(
            [
                np.hstack([crop_panel, raw_panel]),
                np.hstack([refined_panel, summary_panel]),
            ]
        )
        if display_scale != 1.0:
            combined = cv2.resize(combined, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW_NAME, combined)

    update_display()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord("9"):
            choice = key - ord("1")
            if choice < len(component_names):
                active_idx = choice
                update_display()
        elif key == ord("["):
            active_idx = _cycle_component_index(active_idx, len(component_names), -1)
            update_display()
        elif key == ord("]"):
            active_idx = _cycle_component_index(active_idx, len(component_names), 1)
            update_display()
        elif key == ord("n"):
            if current_pos < total_rois - 1:
                current_pos += 1
                update_display()
        elif key == ord("p"):
            if current_pos > 0:
                current_pos -= 1
                update_display()
        elif key == ord("j"):
            current_pos = min(total_rois - 1, current_pos + 10)
            update_display()
        elif key == ord("k"):
            current_pos = max(0, current_pos - 10)
            update_display()
        elif key == ord("o"):
            overlay_mode = "active" if overlay_mode == "all" else "all"
            update_display()
        elif key == ord("f"):
            flagged = _flagged_roi_indices(refined, component_names[active_idx], thresholds=thresholds)
            if flagged:
                for idx in flagged:
                    if idx > current_pos:
                        current_pos = idx
                        break
                else:
                    current_pos = flagged[0]
                update_display()

    cv2.destroyAllWindows()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-mostly inspector for raw and refined subject-mask runs.")
    parser.add_argument("zarr_path", help="Path to Palette zarr archive.")
    parser.add_argument("--subject-run", help="Raw subject_mask_runs/<run> to inspect (default: latest or inferred).")
    parser.add_argument("--refined-run", help="Refined refined_subject_masks_runs/<run> to inspect (default: latest matching raw run).")
    parser.add_argument("--crop-run", help="Crop run to use for ROI images (default: source crop run).")
    parser.add_argument("--component", help="Initial active component.")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index.")
    parser.add_argument("--display-scale", type=float, default=DISPLAY_SCALE)
    parser.add_argument("--largest-component-fraction-threshold", type=float, default=0.98)
    parser.add_argument("--sigma-noise-threshold", type=float, default=2.5)
    parser.add_argument("--ipr-threshold", type=float, default=10.0)
    parser.add_argument("--solidity-threshold", type=float, default=0.85)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    launch_inspector(
        args.zarr_path,
        subject_run=args.subject_run,
        refined_run=args.refined_run,
        crop_run=args.crop_run,
        component=args.component,
        roi_index=args.roi_index,
        display_scale=float(args.display_scale),
        thresholds=InspectorThresholds(
            largest_component_fraction_threshold=float(args.largest_component_fraction_threshold),
            sigma_noise_threshold=float(args.sigma_noise_threshold),
            ipr_threshold=float(args.ipr_threshold),
            solidity_threshold=float(args.solidity_threshold),
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
