from __future__ import annotations

import h5py
import numpy as np

from .models import DatasetSummary, Finding, TrackingInfo
from .reader import dataset_fields, dataset_row_count

TRACKING_REQUIREMENTS = {
    "bounding_boxes": ("payload_frame_id", "x_min", "y_min", "width", "height"),
    "chaser_states": ("stimulus_frame_num", "chaser_pos_x", "chaser_pos_y"),
    "independent_motion_grid_states": ("stimulus_frame_num",),
    "moving_grating_states": ("stimulus_frame_num",),
}


def _summarize_chaser_motion(data: np.ndarray) -> tuple[bool, list[str]]:
    if data.size == 0:
        return True, ["dataset is empty"]
    x = data["chaser_pos_x"].astype(np.float64, copy=False)
    y = data["chaser_pos_y"].astype(np.float64, copy=False)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return False, ["no finite chaser positions"]
    x_valid = x[valid]
    y_valid = y[valid]
    range_x = float(np.nanmax(x_valid) - np.nanmin(x_valid))
    range_y = float(np.nanmax(y_valid) - np.nanmin(y_valid))
    if x_valid.size < 2:
        return True, [f"range_x={range_x:.3f}", f"range_y={range_y:.3f}"]
    step = np.sqrt(np.diff(x_valid) ** 2 + np.diff(y_valid) ** 2)
    nonzero_steps = int(np.count_nonzero(step > 1e-6))
    varies = nonzero_steps > 0 and (range_x > 1e-3 or range_y > 1e-3)
    return varies, [
        f"range_x={range_x:.3f}",
        f"range_y={range_y:.3f}",
        f"nonzero_steps={nonzero_steps}",
    ]


def inspect_tracking(handle: h5py.File) -> tuple[TrackingInfo, list[Finding]]:
    tracking = handle.get("/tracking_data")
    if tracking is None:
        return TrackingInfo(status="skip", tracking_group_present=False), []

    info = TrackingInfo(status="pass", tracking_group_present=True)
    findings: list[Finding] = []
    dataset_statuses: list[str] = []

    for name, required_fields in TRACKING_REQUIREMENTS.items():
        if name not in tracking:
            info.datasets[name] = DatasetSummary(name=name, status="skip", present=False)
            continue

        dataset = tracking[name]
        fields = dataset_fields(dataset)
        rows = dataset_row_count(dataset)
        missing_fields = [field for field in required_fields if field not in fields]
        summary = DatasetSummary(
            name=name,
            status="pass",
            present=True,
            rows=rows,
            fields=fields,
            missing_fields=missing_fields,
        )
        if missing_fields:
            summary.status = "fail"
            findings.append(
                Finding(
                    severity="fail",
                    code=f"h5.tracking_{name}_fields_missing",
                    summary=f"tracking_data/{name} is missing required fields.",
                    details=", ".join(missing_fields),
                    component="tracking",
                    kind="optional",
                )
            )
        elif name == "chaser_states":
            data = dataset[:]
            varies, notes = _summarize_chaser_motion(data)
            summary.notes.extend(notes)
            info.chaser_position_varies = varies
            if not varies and rows > 0:
                summary.status = "warn"
                findings.append(
                    Finding(
                        severity="warn",
                        code="h5.chaser_positions_constant",
                        summary="chaser_states positions do not vary over time.",
                        details=", ".join(notes),
                        component="tracking",
                        kind="optional",
                    )
                )
        info.datasets[name] = summary
        dataset_statuses.append(summary.status)

    if not dataset_statuses:
        info.status = "skip"
        return info, findings

    statuses = set(dataset_statuses)
    if "fail" in statuses:
        info.status = "fail"
    elif "warn" in statuses:
        info.status = "warn"
    else:
        info.status = "pass"
    return info, findings
