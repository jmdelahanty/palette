"""Subject declaration v2 in /metadata/session (agent-contracts #126, option ii)."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files as resource_files
from pathlib import Path

import h5py
import pytest

from fisheye.shared.unified_h5 import admission
from fisheye.shared.unified_h5.common import UnifiedH5ContractError, canonical_json
from fisheye.shared.unified_h5.subject_declaration import (
    DECLARATION,
    RUN_PLAN_REFERENCE,
    SCHEMA,
    SENDER,
    invariant_errors,
    validate_subject_declaration,
)

FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "citrus_subject_declaration_v2"
SNAPSHOT = "/metadata/zebrobot/snapshot_json"
SENDER_DOC = {"gid": 1000, "pid": 4242, "request_id": "req-1", "uid": 1000, "user": "fixture"}


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / "accepted" / f"{name}.json").read_text())


def _snapshot_for(declaration: dict) -> dict:
    """The snapshot v4 fields the pairing reads, as Citrus serves them."""

    individual = declaration["subject_type"] == "individual"
    return {
        "schema_version": 4,
        "dish_id": declaration["dish_id"],
        "subject": {
            "subject_type": declaration["subject_type"],
            "subject_count": declaration["subject_count"],
            "subject_id": declaration["fish_id"] if individual else None,
        },
        "fish_reference": (
            {"mzb_fish_id": declaration["fish_id"], "mzb_fish_unit_id": declaration["unit_id"]}
            if individual
            else None
        ),
        "members": [
            {"mzb_fish_id": fish, "mzb_fish_revision": 1, "provenance": "load_record"}
            for fish in declaration["member_fish_ids"]
        ],
    }


def _write(group, name: str, value, *, attrs=None, raw: bytes | None = None) -> None:
    data = raw if raw is not None else canonical_json(value)
    dataset = group.create_dataset(name, data=data.decode("utf-8"), dtype=h5py.string_dtype("utf-8"))
    checksum = "sha256:" + hashlib.sha256(data).hexdigest()
    dataset.attrs["checksum_algorithm"] = "sha256"
    dataset.attrs["checksum_sha256"] = checksum
    for key, item in (attrs or {}).items():
        dataset.attrs[key] = checksum if item is None else item


def _h5(tmp_path, declaration, *, snapshot=None, reference=True, sender=True, edit=None):
    path = tmp_path / "declaration.h5"
    with h5py.File(path, "w") as h5:
        session = h5.create_group("/metadata/session")
        if declaration is not None:
            _write(session, "subject_declaration_json", declaration)
            orchard = declaration["source"] == "orchard_run_plan"
            if orchard and reference:
                _write(session, "run_plan_reference_json", declaration["run_plan"],
                       attrs={"run_plan_reference_sha256": None})
            if orchard and sender:
                _write(session, "subject_declaration_sender_json", SENDER_DOC)
        zebrobot = h5.create_group("/metadata/zebrobot")
        doc = snapshot if snapshot is not None else (_snapshot_for(declaration) if declaration else None)
        if doc is not None:
            zebrobot.create_dataset("snapshot_json", data=json.dumps(doc), dtype=h5py.string_dtype("utf-8"))
        if edit:
            edit(h5)
    return h5py.File(path, "r")


def _refused(h5, code: str) -> None:
    with h5, pytest.raises(UnifiedH5ContractError, match=code):
        validate_subject_declaration(h5)


ACCEPTED = sorted(path.stem for path in (FIXTURES / "accepted").glob("*.json"))


def test_the_schema_is_pinned_byte_for_byte():
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(SCHEMA[0]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SCHEMA[1]


def test_the_reference_invariants_agree_with_the_contract_fixtures():
    assert len(ACCEPTED) == 7
    for name in ACCEPTED:
        assert invariant_errors(_fixture(name)) == [], name
    expected = json.loads((FIXTURES / "refused" / "expected.json").read_text())
    for name, kind in expected.items():
        doc = json.loads((FIXTURES / "refused" / f"{name}.json").read_text())
        if kind == "invariant":
            assert invariant_errors(doc), name


@pytest.mark.parametrize("name", ACCEPTED)
def test_every_accepted_declaration_is_admitted_with_its_paired_snapshot(tmp_path, name):
    declaration = _fixture(name)
    with _h5(tmp_path, declaration) as h5:
        assert validate_subject_declaration(h5) == declaration


@pytest.mark.parametrize(
    "name",
    sorted(path.stem for path in (FIXTURES / "refused").glob("*.json") if path.stem != "expected"),
)
def test_every_refused_declaration_is_refused(tmp_path, name):
    declaration = json.loads((FIXTURES / "refused" / f"{name}.json").read_text())
    with _h5(tmp_path, declaration, snapshot={"schema_version": 4}) as h5:
        with pytest.raises(UnifiedH5ContractError, match="subject_declaration_(schema|invariant)"):
            validate_subject_declaration(h5)


def test_a_file_without_a_declaration_is_unaffected(tmp_path):
    with _h5(tmp_path, None) as h5:
        assert validate_subject_declaration(h5) is None


def test_a_reference_or_sender_without_a_declaration_is_refused(tmp_path):
    def stray(h5):
        _write(h5["/metadata/session"], "subject_declaration_sender_json", SENDER_DOC)

    _refused(_h5(tmp_path, None, edit=stray), "subject_declaration_companion_without_declaration")


def test_a_wrong_or_malformed_checksum_is_refused(tmp_path):
    declaration = _fixture("dish_group_members_from_plan")

    def tamper(value):
        def edit(h5):
            h5[DECLARATION].attrs["checksum_sha256"] = value
        return edit

    _refused(_h5(tmp_path, declaration, edit=tamper("sha256:" + "0" * 64)),
             "subject_declaration_checksum_mismatch")
    _refused(_h5(tmp_path, declaration, edit=tamper("0" * 64)), "subject_declaration_checksum_form")

    def algorithm(h5):
        h5[DECLARATION].attrs["checksum_algorithm"] = "md5"

    _refused(_h5(tmp_path, declaration, edit=algorithm), "subject_declaration_checksum_algorithm")

    def reference_attr(h5):
        h5[RUN_PLAN_REFERENCE].attrs["run_plan_reference_sha256"] = "sha256:" + "1" * 64

    _refused(_h5(tmp_path, declaration, edit=reference_attr),
             "subject_declaration_checksum_mismatch:.*run_plan_reference_sha256")


def test_stored_bytes_must_be_canonical(tmp_path):
    declaration = _fixture("operator_without_plan")

    def pretty(h5):
        session = h5["/metadata/session"]
        del session["subject_declaration_json"]
        _write(session, "subject_declaration_json", None,
               raw=json.dumps(declaration, indent=1).encode("utf-8"))

    _refused(_h5(tmp_path, declaration, edit=pretty), "noncanonical_json")


def test_a_dataset_that_is_not_a_scalar_utf8_string_is_refused(tmp_path):
    declaration = _fixture("operator_without_plan")

    def as_bytes(h5):
        session = h5["/metadata/session"]
        data = canonical_json(declaration)
        del session["subject_declaration_json"]
        dataset = session.create_dataset("subject_declaration_json", data=list(data), dtype="u1")
        dataset.attrs["checksum_algorithm"] = "sha256"
        dataset.attrs["checksum_sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()

    _refused(_h5(tmp_path, declaration, edit=as_bytes), "subject_declaration_not_scalar_utf8")


def test_the_run_plan_reference_and_sender_go_with_a_plan_declaration_only(tmp_path):
    plan = _fixture("dish_group_members_from_plan")
    _refused(_h5(tmp_path, plan, reference=False), "subject_declaration_run_plan_reference_presence")
    _refused(_h5(tmp_path, plan, sender=False), "subject_declaration_sender_presence")

    override = _fixture("operator_override_of_plan")

    def add_sender(h5):
        _write(h5["/metadata/session"], "subject_declaration_sender_json", SENDER_DOC)

    _refused(_h5(tmp_path, override, edit=add_sender), "subject_declaration_sender_presence")


def test_the_run_plan_reference_equals_the_declarations_run_plan(tmp_path):
    plan = _fixture("dish_group_members_from_plan")

    def other_plan(h5):
        session = h5["/metadata/session"]
        changed = copy.deepcopy(plan["run_plan"])
        changed["plan_revision"] = changed["plan_revision"] + 1
        del session["run_plan_reference_json"]
        _write(session, "run_plan_reference_json", changed, attrs={"run_plan_reference_sha256": None})

    _refused(_h5(tmp_path, plan, edit=other_plan), "subject_declaration_run_plan_reference_mismatch")


def test_the_sender_is_closed_and_typed(tmp_path):
    plan = _fixture("dish_group_members_from_plan")
    for doc in ({**SENDER_DOC, "extra": 1}, {**SENDER_DOC, "uid": "1000"}, {**SENDER_DOC, "user": ""}):
        def swap(h5, doc=doc):
            session = h5["/metadata/session"]
            del session["subject_declaration_sender_json"]
            _write(session, "subject_declaration_sender_json", doc)

        _refused(_h5(tmp_path, plan, edit=swap), "subject_declaration_sender_(keys|value)")


def test_the_snapshot_v4_must_pair_with_the_declaration(tmp_path):
    group = _fixture("dish_group_members_from_plan")
    snapshot = _snapshot_for(group)
    snapshot["members"] = snapshot["members"][:-1] + [
        {"mzb_fish_id": "00000000-0000-4000-8000-000000000999", "mzb_fish_revision": 1, "provenance": "x"}
    ]
    _refused(_h5(tmp_path, group, snapshot=snapshot), "subject_declaration_pairing:member_fish_ids")
    duplicated = _snapshot_for(group)
    duplicated["members"] = duplicated["members"][:1] * len(group["member_fish_ids"])
    _refused(_h5(tmp_path, group, snapshot=duplicated), "subject_declaration_pairing:member_fish_ids")
    count = _snapshot_for(group)
    count["subject"]["subject_count"] += 1
    _refused(_h5(tmp_path, group, snapshot=count), "subject_declaration_pairing:subject_count")
    dish = _snapshot_for(group)
    dish["dish_id"] = "other_dish"
    _refused(_h5(tmp_path, group, snapshot=dish), "subject_declaration_pairing:dish_id")

    well = _fixture("individual_plate_well_from_plan")
    fish = _snapshot_for(well)
    fish["subject"]["subject_id"] = "00000000-0000-4000-8000-000000000999"
    _refused(_h5(tmp_path, well, snapshot=fish), "subject_declaration_pairing:fish_id")
    unit = _snapshot_for(well)
    unit["fish_reference"]["mzb_fish_unit_id"] = "other:A2"
    _refused(_h5(tmp_path, well, snapshot=unit), "subject_declaration_pairing:unit_id")

    _refused(_h5(tmp_path, group, snapshot={**_snapshot_for(group), "schema_version": 3}),
             "subject_declaration_requires_snapshot_v4")

    def no_snapshot(h5):
        del h5[SNAPSHOT]

    _refused(_h5(tmp_path, group, edit=no_snapshot), "subject_declaration_without_snapshot")


def test_whole_file_admission_runs_the_declaration_check_after_integrity(monkeypatch):
    calls = []
    monkeypatch.setattr(admission, "validate_external_receipt", lambda *a, **k: calls.append("receipt"))
    monkeypatch.setattr(admission, "validate_internal_integrity", lambda h5: calls.append("integrity"))
    monkeypatch.setattr(admission, "validate_rig_provenance", lambda h5: calls.append("rig"))

    def refuse(h5):
        calls.append("declaration")
        raise UnifiedH5ContractError("subject_declaration_schema:stop")

    monkeypatch.setattr(admission, "validate_subject_declaration", refuse)
    with pytest.raises(UnifiedH5ContractError, match="subject_declaration_schema:stop"):
        admission.validate_artifact(object(), source_h5=Path("x.h5"), finalization_receipt={})
    assert calls == ["receipt", "integrity", "rig", "declaration"]
