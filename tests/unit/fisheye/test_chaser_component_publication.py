from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis.chaser_component_publication import (
    COMPONENT_MANIFEST_ATTR,
    COMPONENT_MANIFEST_DIGEST_ATTR,
    COMPONENT_SELECTOR_ATTR,
    COMPONENT_SELECTOR_DIGEST_ATTR,
    ChaserComponentContract,
    ChaserComponentPublicationError,
    build_chaser_component_handle,
    component_record_sha256,
    load_chaser_component_handle_json,
    load_explicit_chaser_component,
    load_selected_chaser_component,
    open_explicit_chaser_component_group,
    persist_chaser_component_manifest,
    persist_chaser_component_selector,
    validate_chaser_component_manifest,
    validate_chaser_component_handle,
    validate_chaser_component_selector,
)


class _Array:
    def __init__(self, values, *, attrs=None):
        self.values = np.asarray(values)
        self.attrs = dict(attrs or {})

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, item):
        return self.values[item]


class _Group:
    def __init__(self, *, attrs=None, arrays=None, groups=None):
        self.attrs = dict(attrs or {})
        self.arrays = dict(arrays or {})
        self.groups = dict(groups or {})

    def array_keys(self):
        return self.arrays.keys()

    def group_keys(self):
        return self.groups.keys()

    def __getitem__(self, name):
        if "/" in name:
            current = self
            for part in name.split("/"):
                current = current[part]
            return current
        if name in self.arrays:
            return self.arrays[name]
        return self.groups[name]


def _snapshot(*, digest_byte: str = "a"):
    return SimpleNamespace(
        run_path="analysis/chaser_distance_runs/canonical",
        publication_seal_ref=(
            "/analysis/chaser_distance_runs/canonical@chaser_distance_publication_seal"
        ),
        publication_seal_sha256=digest_byte * 64,
        surface_manifest_ref=(
            "/analysis/chaser_distance_runs/canonical@chaser_distance_surface_manifest"
        ),
        surface_manifest_sha256="b" * 64,
        row_identity_ref=(
            "/analysis/chaser_distance_runs/canonical@row_identity_contract"
        ),
        row_identity_sha256="c" * 64,
        authority_record=lambda: {
            "schema_id": "palette.chaser_distance_read_authority",
            "schema_version": 1,
            "run_ref": "/analysis/chaser_distance_runs/canonical",
        },
    )


def _component():
    return _Group(
        attrs={
            "coordinate_space": "arena_relative_canvas_px",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        },
        arrays={
            "frame_index": _Array(np.asarray([2, 7], dtype=np.int64)),
        },
        groups={
            "events": _Group(
                attrs={"row_axis": "event"},
                arrays={
                    "distance_mm": _Array(
                        np.asarray([1.25, 3.5], dtype=np.float32),
                        attrs={"units": "mm"},
                    ),
                    "valid": _Array(np.asarray([True, False], dtype=bool)),
                },
            )
        },
    )


def _contract(*, threshold: float = 4.0):
    return ChaserComponentContract(
        component_family="chaser_escape_events",
        component_name="escape_v2",
        semantic_schema_id="palette.chaser_escape_events",
        semantic_schema_version=2,
        method_id="palette.chaser_escape_event_detector",
        method_version="2.1.0",
        parameters={"threshold_mm": threshold},
        source_authorities={
            "bout_run": {
                "run_ref": "/analysis/swim_bout_runs/bouts",
                "manifest_sha256": "d" * 64,
            }
        },
    )


def test_seal_then_select_exact_component() -> None:
    component = _component()
    snapshot = _snapshot()
    manifest, manifest_digest = persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )

    assert manifest["selector_eligible"] is False
    assert manifest_digest == component_record_sha256(manifest)
    assert [row["path"] for row in manifest["payload"]["arrays"]] == [
        "frame_index",
        "events/distance_mm",
        "events/valid",
    ]
    assert manifest["payload"]["arrays"][1]["dtype"] == "<f4"

    parent = _Group(attrs={"latest": "legacy_name"})
    selector, selector_digest = persist_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )

    assert parent.attrs["latest"] == "legacy_name"
    assert selector["selected_component"] == "escape_v2"
    assert selector["component_manifest_sha256"] == manifest_digest
    assert selector_digest == component_record_sha256(selector)


def test_payload_tampering_fails_after_seal() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    component.groups["events"].arrays["distance_mm"].values[0] = 99.0

    with pytest.raises(
        ChaserComponentPublicationError,
        match="payload or semantic contract changed",
    ):
        validate_chaser_component_manifest(
            component,
            snapshot=_snapshot(),
            expected_relative_path="chaser_escape_events/escape_v2",
        )


def test_non_scientific_publisher_attrs_do_not_invalidate_payload() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    component.attrs.update(
        {
            "atomic_publication_owner_uuid": "attempt-1",
            "stage_selector_eligible": False,
            "palette_run_completion_status": "complete",
            "cluster_output_staging": {"copy_seconds": 1.0},
        }
    )

    validate_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        expected_relative_path="chaser_escape_events/escape_v2",
    )


def test_unclassified_root_attribute_change_invalidates_payload() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    component.attrs["scientific_threshold_mm"] = 7.0

    with pytest.raises(
        ChaserComponentPublicationError,
        match="payload or semantic contract changed",
    ):
        validate_chaser_component_manifest(
            component,
            snapshot=_snapshot(),
            expected_relative_path="chaser_escape_events/escape_v2",
        )


def test_rehashed_nested_manifest_tampering_fails_exact_fields() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    tampered = copy.deepcopy(component.attrs[COMPONENT_MANIFEST_ATTR])
    tampered["component"]["unexpected"] = "accepted_by_loose_validator"
    component.attrs[COMPONENT_MANIFEST_ATTR] = tampered
    component.attrs[COMPONENT_MANIFEST_DIGEST_ATTR] = component_record_sha256(tampered)

    with pytest.raises(ChaserComponentPublicationError, match="exactly"):
        validate_chaser_component_manifest(
            component,
            snapshot=_snapshot(),
            expected_relative_path="chaser_escape_events/escape_v2",
        )


def test_expected_contract_rejects_rehashed_parameter_tampering() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    tampered = copy.deepcopy(component.attrs[COMPONENT_MANIFEST_ATTR])
    tampered["parameters"]["threshold_mm"] = 400.0
    component.attrs[COMPONENT_MANIFEST_ATTR] = tampered
    component.attrs[COMPONENT_MANIFEST_DIGEST_ATTR] = component_record_sha256(tampered)

    with pytest.raises(
        ChaserComponentPublicationError,
        match="payload or semantic contract changed",
    ):
        validate_chaser_component_manifest(
            component,
            snapshot=_snapshot(),
            expected_relative_path="chaser_escape_events/escape_v2",
            expected_contract=_contract(),
        )


def test_base_authority_change_fails_closed() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )

    with pytest.raises(ChaserComponentPublicationError, match="different base"):
        validate_chaser_component_manifest(
            component,
            snapshot=_snapshot(digest_byte="e"),
            expected_relative_path="chaser_escape_events/escape_v2",
        )


def test_manifest_is_immutable() -> None:
    component = _component()
    persist_chaser_component_manifest(
        component,
        snapshot=_snapshot(),
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )

    with pytest.raises(ChaserComponentPublicationError, match="Refusing to rewrite"):
        persist_chaser_component_manifest(
            component,
            snapshot=_snapshot(),
            relative_path="chaser_escape_events/escape_v2",
            contract=_contract(),
        )


def test_selector_rejects_rehashed_manifest_replacement() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    parent = _Group()
    persist_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    tampered = copy.deepcopy(component.attrs[COMPONENT_MANIFEST_ATTR])
    tampered["parameters"]["threshold_mm"] = 6.0
    component.attrs[COMPONENT_MANIFEST_ATTR] = tampered
    component.attrs[COMPONENT_MANIFEST_DIGEST_ATTR] = component_record_sha256(tampered)

    with pytest.raises(ChaserComponentPublicationError, match="digest changed"):
        validate_chaser_component_selector(
            parent,
            component=component,
            snapshot=snapshot,
            expected_family="chaser_escape_events",
        )


def test_selector_rejects_rehashed_unknown_field() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    parent = _Group()
    persist_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    tampered = copy.deepcopy(parent.attrs[COMPONENT_SELECTOR_ATTR])
    tampered["latest"] = "fallback"
    parent.attrs[COMPONENT_SELECTOR_ATTR] = tampered
    parent.attrs[COMPONENT_SELECTOR_DIGEST_ATTR] = component_record_sha256(tampered)

    with pytest.raises(ChaserComponentPublicationError, match="exactly"):
        validate_chaser_component_selector(
            parent,
            component=component,
            snapshot=snapshot,
            expected_family="chaser_escape_events",
        )


def test_object_arrays_are_forbidden() -> None:
    component = _component()
    component.arrays["labels"] = _Array(np.asarray(["bad"], dtype=object))

    with pytest.raises(ChaserComponentPublicationError, match="object dtype"):
        persist_chaser_component_manifest(
            component,
            snapshot=_snapshot(),
            relative_path="chaser_escape_events/escape_v2",
            contract=_contract(),
        )


def test_selected_reader_returns_detached_exact_payload_without_latest_fallback() -> (
    None
):
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    parent = _Group(
        attrs={"latest": "unsealed_newer_child"},
        groups={"escape_v2": component},
    )
    persist_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    component.attrs["stage_selector_eligible"] = True
    root = _Group(
        groups={
            "analysis": _Group(
                groups={
                    "chaser_distance_runs": _Group(
                        groups={
                            "canonical": _Group(groups={"chaser_escape_events": parent})
                        }
                    )
                }
            )
        }
    )

    selected = load_selected_chaser_component(
        root,
        snapshot=snapshot,
        component_family="chaser_escape_events",
        expected_semantic_schema_id="palette.chaser_escape_events",
        expected_semantic_schema_version=2,
    )

    assert selected.component_name == "escape_v2"
    assert selected.array("events/distance_mm").tolist() == [1.25, 3.5]
    assert selected.array("events/distance_mm").flags.writeable is False
    assert selected.group_attributes["events"]["row_axis"] == "event"
    with pytest.raises(TypeError):
        selected.manifest["publication_state"] = "tampered"


def test_selected_reader_rejects_selector_committed_before_eligibility() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    parent = _Group(groups={"escape_v2": component})
    persist_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    root = _Group(
        groups={
            "analysis": _Group(
                groups={
                    "chaser_distance_runs": _Group(
                        groups={
                            "canonical": _Group(
                                groups={"chaser_escape_events": parent}
                            )
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(
        ChaserComponentPublicationError,
        match="not selector eligible",
    ):
        load_selected_chaser_component(
            root,
            snapshot=snapshot,
            component_family="chaser_escape_events",
            expected_semantic_schema_id="palette.chaser_escape_events",
            expected_semantic_schema_version=2,
        )


def test_explicit_handle_loads_ineligible_component_without_selector() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    parent = _Group(
        attrs={"latest": "unsealed_newer_child"},
        groups={"escape_v2": component},
    )
    root = _Group(
        groups={
            "analysis": _Group(
                groups={
                    "chaser_distance_runs": _Group(
                        groups={
                            "canonical": _Group(
                                groups={"chaser_escape_events": parent}
                            )
                        }
                    )
                }
            )
        }
    )
    handle = build_chaser_component_handle(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )

    selected = load_explicit_chaser_component(
        root,
        snapshot=snapshot,
        handle=handle,
        expected_semantic_schema_id="palette.chaser_escape_events",
        expected_semantic_schema_version=2,
    )

    assert selected.component_name == "escape_v2"
    assert selected.array("events/distance_mm").tolist() == [1.25, 3.5]
    assert COMPONENT_SELECTOR_ATTR not in parent.attrs
    assert parent.attrs["latest"] == "unsealed_newer_child"

    opened = open_explicit_chaser_component_group(
        root,
        snapshot=snapshot,
        handle=handle,
        expected_semantic_schema_id="palette.chaser_escape_events",
        expected_semantic_schema_version=2,
    )
    assert opened.group is component
    assert opened.manifest_sha256 == selected.manifest_sha256
    assert opened.dependency_handle["record_sha256"] == handle["record_sha256"]


def test_explicit_handle_rejects_incomplete_and_tombstoned_components() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    component.attrs["palette_run_completion_status"] = "failed"
    with pytest.raises(ChaserComponentPublicationError, match="runtime-complete"):
        build_chaser_component_handle(
            component,
            snapshot=snapshot,
            relative_path="chaser_escape_events/escape_v2",
        )

    component.attrs["palette_run_completion_status"] = "complete"
    handle = build_chaser_component_handle(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    component.attrs["atomic_publication_tombstone"] = {"failed": True}
    root = _Group(
        groups={
            "analysis": _Group(
                groups={
                    "chaser_distance_runs": _Group(
                        groups={
                            "canonical": _Group(
                                groups={
                                    "chaser_escape_events": _Group(
                                        groups={"escape_v2": component}
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(ChaserComponentPublicationError, match="tombstone"):
        open_explicit_chaser_component_group(
            root,
            snapshot=snapshot,
            handle=handle,
            expected_semantic_schema_id="palette.chaser_escape_events",
            expected_semantic_schema_version=2,
        )


def test_explicit_handle_rejects_rehashed_unknown_field_and_base_change() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    handle = build_chaser_component_handle(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    tampered = copy.deepcopy(handle)
    tampered["unexpected"] = "fallback"
    body = {
        key: value
        for key, value in tampered.items()
        if key not in {"record_sha256", "unexpected"}
    }
    tampered["record_sha256"] = component_record_sha256(body)

    with pytest.raises(ChaserComponentPublicationError, match="exactly"):
        validate_chaser_component_handle(tampered, snapshot=snapshot)
    with pytest.raises(ChaserComponentPublicationError, match="different base"):
        validate_chaser_component_handle(handle, snapshot=_snapshot(digest_byte="e"))


def test_explicit_handle_rejects_manifest_replacement() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    handle = build_chaser_component_handle(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    replacement = copy.deepcopy(component.attrs[COMPONENT_MANIFEST_ATTR])
    replacement["parameters"]["threshold_mm"] = 8.0
    component.attrs[COMPONENT_MANIFEST_ATTR] = replacement
    component.attrs[COMPONENT_MANIFEST_DIGEST_ATTR] = component_record_sha256(
        replacement
    )
    root = _Group(
        groups={
            "analysis": _Group(
                groups={
                    "chaser_distance_runs": _Group(
                        groups={
                            "canonical": _Group(
                                groups={
                                    "chaser_escape_events": _Group(
                                        groups={"escape_v2": component}
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(ChaserComponentPublicationError, match="digest changed"):
        load_explicit_chaser_component(
            root,
            snapshot=snapshot,
            handle=handle,
            expected_semantic_schema_id="palette.chaser_escape_events",
            expected_semantic_schema_version=2,
        )


def test_component_handle_json_loader_is_strict(tmp_path: Path) -> None:
    source = tmp_path / "handle.json"
    source.write_text('{"schema_id":"example","schema_version":1}', encoding="utf-8")
    assert dict(load_chaser_component_handle_json(source)) == {
        "schema_id": "example",
        "schema_version": 1,
    }

    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ChaserComponentPublicationError, match="one object"):
        load_chaser_component_handle_json(source)

    source.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ChaserComponentPublicationError, match="non-finite"):
        load_chaser_component_handle_json(source)


def test_selected_reader_rejects_schema_mismatch() -> None:
    component = _component()
    snapshot = _snapshot()
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
        contract=_contract(),
    )
    parent = _Group(groups={"escape_v2": component})
    persist_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        relative_path="chaser_escape_events/escape_v2",
    )
    component.attrs["stage_selector_eligible"] = True
    root = _Group(
        groups={
            "analysis": _Group(
                groups={
                    "chaser_distance_runs": _Group(
                        groups={
                            "canonical": _Group(groups={"chaser_escape_events": parent})
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(ChaserComponentPublicationError, match="incompatible"):
        load_selected_chaser_component(
            root,
            snapshot=snapshot,
            component_family="chaser_escape_events",
            expected_semantic_schema_id="palette.chaser_escape_events",
            expected_semantic_schema_version=3,
        )
