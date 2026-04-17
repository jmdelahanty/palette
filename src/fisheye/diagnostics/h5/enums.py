from __future__ import annotations

import h5py

from .models import EnumsInfo, Finding
from .reader import dataset_fields, dataset_row_count

EXPECTED_ENUM_DATASETS = (
    "events",
    "stimulus_modes",
    "chaser_trial_states",
    "chaser_loom_modes",
    "chaser_loom_phases",
)


def inspect_enums(handle: h5py.File) -> tuple[EnumsInfo, list[Finding]]:
    enums_group = handle.get("/enums")
    if enums_group is None:
        return EnumsInfo(status="skip"), []

    info = EnumsInfo(status="pass")
    findings: list[Finding] = []
    for name, node in enums_group.items():
        if not isinstance(node, h5py.Dataset):
            continue
        fields = dataset_fields(node)
        if "id" not in fields or "name" not in fields:
            info.malformed_datasets.append(name)
            findings.append(
                Finding(
                    severity="warn",
                    code="h5.enum_dataset_malformed",
                    summary=f"Enum dataset {name} does not expose id/name fields.",
                    component="enums",
                    kind="optional",
                )
            )
            continue
        info.dataset_counts[name] = dataset_row_count(node)

    info.missing_expected = [name for name in EXPECTED_ENUM_DATASETS if name not in info.dataset_counts]
    if info.malformed_datasets:
        info.status = "warn"
    elif info.dataset_counts:
        info.status = "pass"
    else:
        info.status = "skip"
    return info, findings
