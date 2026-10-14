"""Citrus subject identity v3, 2026-10-14 amendment: the Citrus-written witnesses.

agent-contracts PR 110 head 3b3f214 ``citrus-palette/fixtures/subject_v3/``,
generated at Citrus PR 43 ``533ce76`` by the production DishAPI code from canned
MetaZebrobot responses; vendored verbatim with their outer finalization receipts.
The complete files go through the full path (receipt-gated admission, native
import, subject projection). Refused files were edited with h5py after sealing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import h5py
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import import_stimulus_to_zarr
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.shared.unified_h5 import UnifiedH5ContractError
from fisheye.shared.unified_h5.citrus_subject_snapshot import (
    SCHEMA_SHA256,
    SNAPSHOT_PATH,
    admit_subject_snapshot,
    subject_attributes,
)
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.metadata import native_attributes
from fisheye.utils import import_recording_analysis as mod

WITNESSES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_unified_h5_subject_v3"
README = (WITNESSES / "README.md").read_text()
DIGESTS = dict(re.findall(r"\| `([^`]+\.h5)` \| `([0-9a-f]{64})`", README))

# case -> (fish_reference status, reason, mzb_fish_unit_id, subject_lookup status)
COMPLETE = {
    "fish_collected": ("collected", None, "19220_1_WP1:A1", "collected"),
    "fish_collected_no_unit": ("collected", None, None, "collected"),
    "dish_collected": ("not_collected", "operator_declined_pick", None, "collected"),
    "no_registered_fish": ("not_collected", "no_registered_fish", None, "collected"),
    "fish_list_unavailable": ("not_collected", "fish_list_unavailable", None, "collected"),
    "lookup_failed": ("lookup_failed", "dish_lookup_failed", None, "lookup_failed"),
    "dish_group": ("not_collected", "dish_group_subject", None, "collected"),
    "no_dish": ("not_collected", "no_dish_declared", None, "not_collected"),
    "version_read_failed": ("not_collected", "operator_declined_pick", None, "collected"),
}
# refused file -> (the complete file it was edited from, the named check's reason,
#                  whether the edit touched a sealed dataset)
REFUSED = {
    "unit_without_collected": ("dish_collected", "citrus_snapshot_schema:'19220_1_WP1:A1' is not of type 'null'", True),
    "new_reason_without_unit_key": ("no_registered_fish", "citrus_snapshot_schema:'mzb_fish_unit_id' is a required property", True),
    "list_unavailable_without_error": ("fish_list_unavailable", "citrus_snapshot_schema:\\[\\] does not contain items", True),
    "list_unavailable_other_dish_error": ("fish_list_unavailable", "fish_list_unavailable_error_not_for_dish:/dishes/19220_1/fish", True),
    "declined_pick_under_lookup_failed": ("lookup_failed", "citrus_snapshot_schema:'operator_declined_pick' is not valid", True),
    # Attribute-only edits: HDF5 attributes are outside the finalization digest,
    # so integrity passes and the snapshot pairing is what refuses them.
    "unit_attr_mismatch": ("fish_collected", "citrus_subject_attr_mismatch:mzb_fish_unit_id", False),
    "unit_attr_without_value": ("fish_collected_no_unit", "citrus_subject_attr_without_snapshot_value:mzb_fish_unit_id", False),
}


def _h5(name: str) -> Path:
    suffix = "_refused" if name in REFUSED else ""
    return WITNESSES / f"subject_v3_{name}{suffix}.h5"


def _snapshot_and_attrs(path: Path):
    with h5py.File(path, "r") as h5:
        return (
            json.loads(h5[SNAPSHOT_PATH][()]),
            subject_attributes(native_attributes(h5["/metadata/subject"])),
        )


def test_vendored_witnesses_are_the_published_bytes():
    assert f"`{SCHEMA_SHA256}`" in README
    assert len(DIGESTS) == len(COMPLETE) + len(REFUSED) == 16
    for name, digest in DIGESTS.items():
        assert hashlib.sha256((WITNESSES / name).read_bytes()).hexdigest() == digest, name
    for name in COMPLETE:
        receipt = json.loads((WITNESSES / f"subject_v3_{name}.receipt.json").read_text())
        assert receipt["contract"]["h5_artifact"]["sha256"] == "sha256:" + DIGESTS[_h5(name).name]


@pytest.mark.parametrize("name", sorted(COMPLETE))
def test_complete_witness_is_admitted_imported_and_projected(tmp_path, name):
    status, reason, unit, lookup = COMPLETE[name]
    snapshot, attrs = _snapshot_and_attrs(_h5(name))
    assert admit_subject_snapshot(snapshot, attrs)["citrus_snapshot_status"] == "admitted"
    assert snapshot["fish_reference"]["status"] == status
    assert snapshot["fish_reference"]["reason"] == reason
    assert snapshot["fish_reference"]["mzb_fish_unit_id"] == unit
    assert snapshot["subject_lookup_status"] == lookup
    assert "fish_id" not in attrs
    assert {k for k, v in attrs.items() if not isinstance(v, str)} <= {"dish_revision", "mzb_fish_revision"}
    if name == "version_read_failed":
        assert snapshot["zebrobot_service"]["service_commit"] is None

    h5_path = tmp_path / _h5(name).name
    shutil.copyfile(_h5(name), h5_path)
    zarr_path = tmp_path / "analysis.zarr"
    receipt = WITNESSES / f"subject_v3_{name}.receipt.json"
    import_stimulus_to_zarr(
        h5_path, zarr_path, run_name="candidate", overwrite=False, verbose=False,
        source_profile="unified_experimental_h5_v1", finalization_receipt=receipt,
    )
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path, h5_path=h5_path, cam_video=None, zarr_path=zarr_path,
        finalization_receipt_path=receipt,
    )
    mod.project_unified_subject_metadata(plan, "candidate")
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    subject = resolve_subject_metadata(root, allow_legacy=False).subject
    assert subject["fish_reference"] == {"status": status, "reason": reason or ""}
    assert subject.get("mzb_fish_unit_id") == unit


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_refused_witness_is_refused_by_its_named_check(name):
    source, reason, sealed = REFUSED[name]
    with h5py.File(_h5(name), "r") as h5:
        if sealed:
            with pytest.raises(UnifiedH5ContractError, match="internal_dependency_mismatch:/metadata/zebrobot/snapshot_json"):
                validate_internal_integrity(h5)
        else:
            validate_internal_integrity(h5)
    snapshot, attrs = _snapshot_and_attrs(_h5(name))
    with pytest.raises(UnifiedH5ContractError, match=reason):
        admit_subject_snapshot(snapshot, attrs)
    # The file it was edited from is admitted.
    assert admit_subject_snapshot(*_snapshot_and_attrs(_h5(source)))["citrus_snapshot_status"] == "admitted"


def _resealed(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Untrusted test reseal over the producer receipt, to get past the byte gate."""

    source, _, _ = REFUSED[name]
    h5_path = tmp_path / _h5(name).name
    shutil.copyfile(_h5(name), h5_path)
    receipt = json.loads((WITNESSES / f"subject_v3_{source}.receipt.json").read_text())
    artifact = receipt["contract"]["h5_artifact"]
    artifact["size_bytes"] = h5_path.stat().st_size
    artifact["sha256"] = "sha256:" + hashlib.sha256(h5_path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(
        receipt["contract"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()
    receipt.update(contract_sha256="sha256:" + digest, receipt_id="obsbindfin_" + digest)
    path = tmp_path / "resealed.receipt.json"
    path.write_text(json.dumps(receipt))
    return h5_path, path


@pytest.mark.parametrize("name", ["unit_attr_mismatch", "unit_attr_without_value"])
def test_attribute_only_edits_are_refused_at_subject_projection(tmp_path, name):
    """Integrity cannot see attrs, so the full path must refuse at the pairing."""

    h5_path, receipt = _resealed(tmp_path, name)
    zarr_path = tmp_path / "analysis.zarr"
    import_stimulus_to_zarr(
        h5_path, zarr_path, run_name="candidate", overwrite=False, verbose=False,
        source_profile="unified_experimental_h5_v1", finalization_receipt=receipt,
    )
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path, h5_path=h5_path, cam_video=None, zarr_path=zarr_path,
        finalization_receipt_path=receipt,
    )
    with pytest.raises(UnifiedH5ContractError, match=REFUSED[name][1]):
        mod.project_unified_subject_metadata(plan, "candidate")
