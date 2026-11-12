"""Citrus subject identity v4: the Citrus-written sealed witnesses, end to end.

agent-contracts #115 (merged 2aaa7bc) ``citrus-palette/fixtures/subject_v4/``,
generated at Citrus ``368dc42`` by the production writer and v4 start module on
canned MetaZebrobot answers; vendored verbatim (H5, finalized receipt v4,
acceptance v3, README). Admitted files go through the full path: receipt-gated
admission (sealer 3.3.0 receipt v4 rules, snapshot v4 ↔ receipt agreement),
native import and subject projection.
"""

from __future__ import annotations

import copy
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
    SCHEMAS,
    SNAPSHOT_PATH,
    admit_subject_snapshot,
    check_receipt_subject_reference,
    subject_attributes,
)
from fisheye.shared.unified_h5.integrity import validate_internal_integrity
from fisheye.shared.unified_h5.metadata import native_attributes
from fisheye.utils import import_recording_analysis as mod

WITNESSES = Path(__file__).resolve().parents[2] / "fixtures" / "citrus_unified_h5_subject_v4"
README = (WITNESSES / "README.md").read_text()
# h5 name -> (h5, receipt, acceptance) sha256 from the README table.
DIGESTS = {
    name: (h5, receipt, acceptance)
    for name, h5, receipt, acceptance in re.findall(
        r"\| `([^`]+\.h5)` \| `([0-9a-f]{64})`<br>`([0-9a-f]{64})`<br>`([0-9a-f]{64})`", README
    )
}
# case -> (registration mode, status, reason, subject_type, listed count, unit)
ADMITTED = {
    "individual_existing_registered": ("existing", "registered", None, "individual", None, "19220_1_WP1:A1"),
    "individual_existing_unverified": ("existing", "unverified", "fish_lookup_transport_failure", "individual", None, None),
    "individual_minted_registered": ("created_at_start", "registered", None, "individual", 2, None),
    "individual_minted_queued": ("outbox", "queued", "create_transport_failure", "individual", 0, None),
    "individual_minted_failed": ("created_at_start", "failed", "create_fish_id_conflict", "individual", 1, None),
    "dish_group": (None, None, None, "dish_group", None, None),
    "no_dish": (None, None, None, None, None, None),
}
REFUSED = ("listed_count_as_string", "link_revision_differs")


def _h5(name: str) -> Path:
    suffix = "_refused" if name in REFUSED else ""
    return WITNESSES / f"subject_v4_{name}{suffix}.h5"


def _receipt_path(name: str) -> Path:
    return _h5(name).with_suffix(".receipt.json")


def _snapshot_and_attrs(path: Path):
    with h5py.File(path, "r") as h5:
        return (
            json.loads(h5[SNAPSHOT_PATH][()]),
            subject_attributes(native_attributes(h5["/metadata/subject"])),
        )


def _import(tmp_path: Path, name: str) -> mod.RecordingAnalysisPlan:
    h5_path = tmp_path / _h5(name).name
    shutil.copyfile(_h5(name), h5_path)
    zarr_path = tmp_path / "analysis.zarr"
    import_stimulus_to_zarr(
        h5_path, zarr_path, run_name="candidate", overwrite=False, verbose=False,
        source_profile="unified_experimental_h5_v1", finalization_receipt=_receipt_path(name),
    )
    return mod.RecordingAnalysisPlan(
        recording_dir=tmp_path, h5_path=h5_path, cam_video=None, zarr_path=zarr_path,
        finalization_receipt_path=_receipt_path(name),
    )


def test_vendored_witnesses_are_the_published_bytes():
    assert f"`{SCHEMAS[4][1][:8]}…`" in README
    assert len(DIGESTS) == len(ADMITTED) + len(REFUSED) == 9
    for name, (h5, receipt, acceptance) in DIGESTS.items():
        stem = name.removesuffix(".h5")
        assert hashlib.sha256((WITNESSES / name).read_bytes()).hexdigest() == h5, name
        assert hashlib.sha256((WITNESSES / f"{stem}.receipt.json").read_bytes()).hexdigest() == receipt, name
        assert hashlib.sha256((WITNESSES / f"{stem}.acceptance.json").read_bytes()).hexdigest() == acceptance, name


@pytest.mark.parametrize("name", sorted(ADMITTED))
def test_admitted_witness_is_admitted_imported_and_projected(tmp_path, name):
    mode, status, reason, subject_type, listed, unit = ADMITTED[name]
    snapshot, attrs = _snapshot_and_attrs(_h5(name))
    admitted = admit_subject_snapshot(snapshot, attrs)
    assert admitted["citrus_snapshot_schema_version"] == 4
    fish = snapshot["fish_reference"]
    if mode is None:
        assert fish is None
    else:
        assert (fish["mzb_registration_mode"], fish["mzb_registration_status"],
                fish["mzb_registration_reason"]) == (mode, status, reason)
        assert snapshot["subject"]["subject_id"] == fish["mzb_fish_id"]
    assert {k for k, v in attrs.items() if not isinstance(v, str)} <= {
        "dish_revision", "mzb_fish_revision", "mzb_dish_listed_fish_count_at_start"}
    assert attrs.get("mzb_dish_listed_fish_count_at_start") == listed
    assert attrs.get("mzb_fish_unit_id") == unit
    receipt = json.loads(_receipt_path(name).read_text())
    assert receipt["schema_version"] == 4
    check_receipt_subject_reference(snapshot, receipt)

    plan = _import(tmp_path, name)
    published = mod.project_unified_subject_metadata(plan, "candidate")
    root = zarr.open_group(str(plan.zarr_path), mode="r", use_consolidated=False)
    resolved = resolve_subject_metadata(root, allow_legacy=False)
    if subject_type is None:
        assert published is None  # no dish declared: statuses only
        return
    assert resolved.subject["subject_type"] == subject_type
    if subject_type == "individual":
        assert resolved.subject_ids == (fish["mzb_fish_id"],)
        assert resolved.subject["mzb_registration"] == {
            "mode": mode, "status": status, "reason": reason or ""}
        assert resolved.subject.get("mzb_dish_listed_fish_count_at_start") == listed
        assert resolved.subject.get("mzb_fish_unit_id") == unit
    else:
        assert resolved.subject_ids == ()
        assert published["expected_subject_count"] == snapshot["subject"]["subject_count"]


def test_listed_count_as_string_is_refused_at_subject_projection(tmp_path):
    """Its receipt is re-issued over the edited file and attrs are outside the
    digest, so the full path imports it and the attr type is what refuses."""

    plan = _import(tmp_path, "listed_count_as_string")
    with pytest.raises(UnifiedH5ContractError, match="attr_not_int64:mzb_dish_listed_fish_count_at_start"):
        mod.project_unified_subject_metadata(plan, "candidate")


def test_link_revision_differs_is_refused_at_integrity_and_by_r6(tmp_path):
    with h5py.File(_h5("link_revision_differs"), "r") as h5:
        with pytest.raises(UnifiedH5ContractError, match="internal_dependency_mismatch:/metadata/zebrobot/snapshot_json"):
            validate_internal_integrity(h5)
    snapshot, attrs = _snapshot_and_attrs(_h5("link_revision_differs"))
    with pytest.raises(UnifiedH5ContractError, match="R6_usage_link_fish_differs"):
        admit_subject_snapshot(snapshot, attrs)
    # Its receipt names the unedited file: the byte gate refuses before anything else.
    with pytest.raises(UnifiedH5ContractError, match="external_h5_byte_digest_mismatch"):
        _import(tmp_path, "link_revision_differs")


def test_admission_refuses_a_receipt_whose_reference_disagrees(tmp_path):
    """The agreement check runs inside receipt-gated admission, not only here."""

    receipt = json.loads(_receipt_path("individual_existing_registered").read_text())
    receipt["contract"]["subject_reference"]["fish"]["revision"] = 4
    digest = hashlib.sha256(json.dumps(
        receipt["contract"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()
    receipt.update(contract_sha256="sha256:" + digest, receipt_id="obsbindfin_" + digest)
    path = tmp_path / "resealed.receipt.json"
    path.write_text(json.dumps(receipt))
    h5_path = tmp_path / _h5("individual_existing_registered").name
    shutil.copyfile(_h5("individual_existing_registered"), h5_path)
    with pytest.raises(UnifiedH5ContractError, match="subject_reference_disagrees:fish"):
        import_stimulus_to_zarr(
            h5_path, tmp_path / "analysis.zarr", run_name="candidate", overwrite=False,
            verbose=False, source_profile="unified_experimental_h5_v1", finalization_receipt=path,
        )


def _registered():
    snapshot, _ = _snapshot_and_attrs(_h5("individual_existing_registered"))
    return snapshot, json.loads(_receipt_path("individual_existing_registered").read_text())


@pytest.mark.parametrize(
    "edit, name",
    [
        (lambda r: r.update(subject_id="0" * 8 + "-0000-4000-8000-" + "0" * 12), "subject_id"),
        (lambda r: r.update(subject_count=2), "subject_count"),
        (lambda r: r["dish"].update(dish_uuid="0" * 8 + "-0000-4000-8000-" + "0" * 12), "dish_uuid"),
        (lambda r: r["fish"].update(revision=4), "fish"),
        (lambda r: r.update(fish=None), "fish"),
        (lambda r: r.update(fish_status="not_collected"), "fish_status"),
        (lambda r: r.update(fish_reason="fish_lookup_transport_failure"), "fish_reason"),
        (lambda r: r.update(lookup_status="lookup_failed"), "lookup_status"),
    ],
)
def test_receipt_reference_must_agree_with_the_snapshot(edit, name):
    snapshot, receipt = _registered()
    edited = copy.deepcopy(receipt)
    edit(edited["contract"]["subject_reference"])
    with pytest.raises(UnifiedH5ContractError, match=f"subject_reference_disagrees:{name}"):
        check_receipt_subject_reference(snapshot, edited)


def test_snapshot_v4_pairs_only_with_receipt_v4():
    snapshot, receipt = _registered()
    for version in (1, 2, 3):
        older = copy.deepcopy(receipt)
        older["schema_version"] = version
        with pytest.raises(UnifiedH5ContractError, match=f"version_pairing:receipt_v{version}"):
            check_receipt_subject_reference(snapshot, older)
    v3_snapshot = copy.deepcopy(snapshot)
    v3_snapshot["schema_version"] = 3
    for older_snapshot in (v3_snapshot, None):
        with pytest.raises(UnifiedH5ContractError, match="version_pairing:receipt_v4"):
            check_receipt_subject_reference(older_snapshot, receipt)
    v3_receipt = copy.deepcopy(receipt)
    v3_receipt["schema_version"] = 3
    check_receipt_subject_reference(v3_snapshot, v3_receipt)  # earlier pairs unchanged
