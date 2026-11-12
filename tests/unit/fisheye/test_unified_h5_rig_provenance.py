"""Citrus #75 adds scalar UTF-8 datasets to the existing open metadata tree.

These are synthetic extensions of the pinned base H5. Its legacy rig ID is
retained throughout. The fixture exercises Palette admission, not Citrus start.
"""

import hashlib
import json

import h5py
import pytest

from fisheye.shared.unified_h5 import (
    UnifiedH5ContractError,
    validate_unified_h5_artifact,
)
from fisheye.shared.unified_h5.common import canonical_json, digest
from fisheye.shared.unified_h5.integrity import FINALIZATION, MANIFEST
from tests.unit.fisheye.unified_h5_fixtures import (
    emit_fixture,
    synthetic_receipt_for_mutated_test_file,
)

GROUP = "/metadata/rig/pancakebatter_rig_file"
FILES = ("rig_file", "events_file", "sidecar")
EVENTS = b"events: []\n"
COMMIT = "a" * 40


def _entry(name, raw):
    # Producer's scalar VL-string encoding. Keep this independent of Palette's
    # describe_internal_dataset, so an admission regression cannot reseal itself.
    path = f"{GROUP}/{name}"
    entry = {
        "kind": "internal_dataset",
        "component_id": "dataset:" + path,
        "path": path,
        "encoding": "hdf5_closed_logical_values_v1",
        "shape": [],
        "type": {"class": "string", "variable_length": True, "character_set": "utf8"},
        "logical_payload_size_bytes": len(raw),
    }
    header = canonical_json(entry)
    framed = (
        len(header).to_bytes(8, "little")
        + header
        + len(raw).to_bytes(8, "little")
        + raw
    )
    entry["content_sha256"] = digest(
        b"citrus.experimental_h5.internal_dataset_digest.v1\n" + framed
    )
    return entry


def _seal_manifest(h5, manifest):
    final = json.loads(h5[FINALIZATION][()])
    raw = canonical_json(manifest)
    h5[MANIFEST][()] = raw
    final["dependency_count"] = len(manifest["dependencies"])
    final["dependency_manifest_sha256"] = digest(raw)
    h5[FINALIZATION][()] = canonical_json(final)


def _fixture(
    tmp_path,
    *,
    status="pinned",
    events=EVENTS,
    listed_events=EVENTS,
    commit=COMMIT,
    sidecar_present=True,
    source_commit=None,
    sidecar_extra=b"",
):
    path = emit_fixture(tmp_path, "base")
    with h5py.File(path, "r+") as h5:
        group = h5.create_group(GROUP)
        rig_id = h5["/metadata/session"].attrs["rig_id"]
        rig = (
            f"schema: {{name: rig_config, version: 1}}\nrig_id: {rig_id}\n"
            "rig_geometry_revision: 10\ncameras: {'CAM-42': {}}\n"
        ).encode()
        installed = "2026-10-30T00:00:00Z"
        sidecar = (
            f"commit: {commit}\ninstalled: {installed}\nsha256: {hashlib.sha256(rig).hexdigest()}\n"
            "rig_geometry_revision: 10\n"
        ).encode()
        if listed_events is not None:
            sidecar += (
                f"events_sha256: {hashlib.sha256(listed_events).hexdigest()}\n".encode()
            )
        sidecar += sidecar_extra
        if source_commit is None:
            # What PinRigFile records: the sidecar's commit, only when full.
            source_commit = commit if sidecar_present and len(commit) == 40 else ""
        group.attrs.update(
            {
                "status": status,
                "status_detail": "" if status == "pinned" else "synthetic diagnostic",
                "rig_id": rig_id,
                "rig_geometry_revision": 10,
                "lookup": "installed",
                "source_commit": source_commit,
                "installed_at": installed if sidecar_present else "",
            }
        )
        manifest = json.loads(h5[MANIFEST][()])
        root = f"/etc/pancakebatter/rigs/{rig_id}"
        files = (
            ("rig_file", rig, ".yml"),
            ("events_file", events, ".events.yml"),
            ("sidecar", sidecar if sidecar_present else None, ".yml.source"),
        )
        for name, raw, suffix in files:
            group.attrs[name + "_path"] = root + suffix
            group.attrs[name + "_sha256"] = digest(raw) if raw is not None else ""
            if raw is not None and b"\0" in raw:
                group.attrs[name + "_not_stored"] = (
                    "contains a NUL byte, which an HDF5 string would cut short"
                )
            elif raw is not None:
                ds = group.create_dataset(
                    name, data=raw, dtype=h5py.string_dtype("utf-8")
                )
                ds.attrs.update(
                    {"checksum_algorithm": "sha256", "checksum_sha256": digest(raw)}
                )
                manifest["dependencies"].append(_entry(name, raw))
        _seal_manifest(h5, manifest)
    return path


def _admit(path, receipt=None):
    if receipt is None:
        receipt = synthetic_receipt_for_mutated_test_file(path, "base")
    with h5py.File(path) as h5:
        return validate_unified_h5_artifact(
            h5, source_h5=path, finalization_receipt=receipt
        )


@pytest.mark.parametrize(
    ("status", "options", "present"),
    [
        ("pinned", {}, FILES),
        ("pinned", {"events": None, "listed_events": None}, ("rig_file", "sidecar")),
        (
            "pinned",
            {"events": b"#\0", "listed_events": b"#\0"},
            ("rig_file", "sidecar"),
        ),
        (
            "rig_file_sidecar_missing",
            {"sidecar_present": False},
            ("rig_file", "events_file"),
        ),
        ("rig_file_sidecar_mismatch", {"events": None}, ("rig_file", "sidecar")),
        ("rig_file_sidecar_mismatch", {"events": EVENTS + b"# appended\n"}, FILES),
        ("rig_file_sidecar_invalid", {"commit": "abcdef0"}, FILES),
    ],
)
def test_technical_admission_of_captured_files(tmp_path, status, options, present):
    path = _fixture(tmp_path, status=status, **options)
    result = _admit(path)
    assert result.dependency_count == 90 + len(present)
    assert {name for name in FILES if f"{GROUP}/{name}" in result.node_kinds} == set(
        present
    )
    assert not result.selector_eligible


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("missing_dataset", "duplicate_or_missing_dependency"),
        ("missing_entry", "global_dependency_graph_not_closed"),
        ("duplicate_entry", "duplicate_or_missing_dependency"),
        ("wrong_hash", "internal_dependency_mismatch"),
        ("changed_bytes", "internal_dependency_mismatch"),
    ],
)
def test_provenance_manifest_refusals(tmp_path, name, defect, reason):
    path = _fixture(tmp_path)
    target = f"{GROUP}/{name}"
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5[MANIFEST][()])
        entries = manifest["dependencies"]
        row = next(item for item in entries if item.get("path") == target)
        if defect == "missing_dataset":
            del h5[target]
        elif defect == "missing_entry":
            entries.remove(row)
        elif defect == "duplicate_entry":
            entries.append(row)
        elif defect == "wrong_hash":
            row["content_sha256"] = "sha256:" + "0" * 64
        else:
            raw = h5[target][()]
            h5[target][()] = b"X" + raw[1:]
        _seal_manifest(h5, manifest)
    with pytest.raises(UnifiedH5ContractError, match=reason):
        _admit(path)


def test_attribute_edit_breaks_original_whole_file_receipt(tmp_path):
    path = _fixture(tmp_path)
    receipt = synthetic_receipt_for_mutated_test_file(path, "base")
    with h5py.File(path, "r+") as h5:
        h5[GROUP].attrs.modify("rig_geometry_revision", 11)
    with pytest.raises(
        UnifiedH5ContractError, match="external_h5_byte_digest_mismatch"
    ):
        _admit(path, receipt)


def test_scalar_string_fixture_matches_citrus_digest_vector():
    # DescribeInternalDataset at Citrus 2d6bfb7, on a scalar UTF-8 VL string
    # containing the eleven bytes b"events: []\n" at the production path.
    assert _entry("events_file", EVENTS)["content_sha256"] == (
        "sha256:1ad00b4ff07d4710f23896c7b9723cfaf995f1cd901e7e7b038e0ddedac21f0f"
    )


# Enforcement correction: ``source_commit`` is "the sidecar's ``commit``,
# recorded only when it is a full sha" (Citrus docs/Understanding_H5_Log.md and
# RigBinding::PinRigFile at dbf2ee9). The sidecar dataset is covered by the
# sealed dependency manifest; the group attribute is not, so the attribute
# must agree with the sealed bytes.
CONTRADICTS = "rig_provenance_source_commit_contradicts_sidecar"
UNVERIFIABLE = "rig_provenance_source_commit_unverifiable"
PINNED_WITHOUT = "rig_provenance_pinned_without_source_commit"


def _replace_sidecar(h5, raw):
    target = f"{GROUP}/sidecar"
    manifest = json.loads(h5[MANIFEST][()])
    del h5[target]
    ds = h5.create_dataset(target, data=raw, dtype=h5py.string_dtype("utf-8"))
    ds.attrs.update({"checksum_algorithm": "sha256", "checksum_sha256": digest(raw)})
    entries = manifest["dependencies"]
    index = next(i for i, row in enumerate(entries) if row.get("path") == target)
    entries[index] = _entry("sidecar", raw)
    _seal_manifest(h5, manifest)


def test_absent_rig_provenance_is_unchanged(tmp_path):
    path = emit_fixture(tmp_path, "base")
    with h5py.File(path) as h5:
        assert "/metadata/rig" not in h5 or GROUP not in h5
    result = _admit(path)
    assert result.dependency_count == 90
    assert not any(node.startswith(GROUP) for node in result.node_kinds)


@pytest.mark.parametrize(
    ("status", "options"),
    [
        ("pinned", {}),
        ("pinned", {"commit": "1" * 40}),  # all digits stays a YAML string
        ("rig_file_sidecar_mismatch", {"events": EVENTS + b"# appended\n"}),
        ("rig_file_sidecar_invalid", {"commit": "abcdef0"}),
        ("rig_file_sidecar_missing", {"sidecar_present": False}),
    ],
)
def test_consistent_source_commit_is_admitted(tmp_path, status, options):
    result = _admit(_fixture(tmp_path, status=status, **options))
    assert not result.selector_eligible


def test_quoted_sidecar_commit_is_the_same_scalar(tmp_path):
    path = _fixture(tmp_path, commit=f'"{COMMIT}"', source_commit=COMMIT)
    _admit(path)


@pytest.mark.parametrize("status", ["pinned", "rig_file_sidecar_mismatch"])
@pytest.mark.parametrize(
    "attribute",
    ["b" * 40, COMMIT.upper(), COMMIT[:7], COMMIT + "\n", " " + COMMIT[1:]],
)
def test_contradictory_source_commit_is_refused(tmp_path, status, attribute):
    path = _fixture(tmp_path, status=status, source_commit=attribute)
    with pytest.raises(UnifiedH5ContractError, match=CONTRADICTS):
        _admit(path)


@pytest.mark.parametrize(
    "sidecar_options",
    [
        {"commit": "abcdef0"},  # the sidecar's own commit is not a full sha
        {"sidecar_extra": b"commit: [unterminated\n"},  # does not parse
    ],
)
def test_source_commit_beyond_an_invalid_sidecar_is_refused(tmp_path, sidecar_options):
    path = _fixture(
        tmp_path,
        status="rig_file_sidecar_invalid",
        source_commit=COMMIT,
        **sidecar_options,
    )
    with pytest.raises(UnifiedH5ContractError, match=CONTRADICTS):
        _admit(path)


@pytest.mark.parametrize(
    "raw",
    [b"- " + COMMIT.encode() + b"\n", b"commit: {sha: x}\n", b""],
)
def test_source_commit_against_a_sidecar_without_a_commit_is_refused(tmp_path, raw):
    path = _fixture(tmp_path)
    with h5py.File(path, "r+") as h5:
        _replace_sidecar(h5, raw)
    with pytest.raises(UnifiedH5ContractError, match=CONTRADICTS):
        _admit(path)


@pytest.mark.parametrize(
    "options",
    [
        {"status": "rig_file_sidecar_missing", "sidecar_present": False},
        {"sidecar_extra": b"#\0"},  # recorded only as a digest, never as bytes
    ],
)
def test_source_commit_without_stored_sidecar_is_refused(tmp_path, options):
    path = _fixture(tmp_path, source_commit=COMMIT, **options)
    with h5py.File(path) as h5:
        assert f"{GROUP}/sidecar" not in h5
    with pytest.raises(UnifiedH5ContractError, match=UNVERIFIABLE):
        _admit(path)


def test_pinned_without_source_commit_is_refused(tmp_path):
    path = _fixture(tmp_path, source_commit="")
    with pytest.raises(UnifiedH5ContractError, match=PINNED_WITHOUT):
        _admit(path)


def test_attribute_tamper_with_fresh_receipt_is_refused(tmp_path):
    # PR #424's separate probe: an attribute edit after sealing, then a fresh
    # whole-file receipt. That passed before; the cross-check now refuses it.
    path = _fixture(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5[GROUP].attrs["source_commit"] = "b" * 40
    with pytest.raises(UnifiedH5ContractError, match=CONTRADICTS):
        _admit(path)


def test_attribute_tamper_with_original_receipt_still_fails_byte_gate(tmp_path):
    path = _fixture(tmp_path)
    receipt = synthetic_receipt_for_mutated_test_file(path, "base")
    with h5py.File(path, "r+") as h5:
        h5[GROUP].attrs["source_commit"] = "b" * 40
    with pytest.raises(
        UnifiedH5ContractError, match="external_h5_(size|byte_digest)_mismatch"
    ):
        _admit(path, receipt)


def test_resealed_sidecar_tamper_is_refused(tmp_path):
    # The sidecar bytes are swapped and the manifest resealed, so the inner
    # integrity gate passes. The unchanged attribute now names another commit.
    path = _fixture(tmp_path)
    with h5py.File(path, "r+") as h5:
        raw = h5[f"{GROUP}/sidecar"][()].replace(COMMIT.encode(), b"c" * 40)
        _replace_sidecar(h5, raw)
    with pytest.raises(UnifiedH5ContractError, match=CONTRADICTS):
        _admit(path)


def test_admitted_claims_keep_their_shape(tmp_path):
    # The check only refuses: an admitted file's claim keys are unchanged.
    base = _admit(emit_fixture(tmp_path / "base", "base"))
    pinned = _admit(_fixture(tmp_path / "pinned"))
    assert set(pinned.manifest_claims()) == set(base.manifest_claims())
    assert pinned.dependency_count == base.dependency_count + 3
