from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import pytest

import fisheye.shared.subject_mask_coordinate_publication as publication_module
from fisheye.shared.coordinate_record import bind_persisted_coordinate_record
from fisheye.shared.zarr.coordinate_successor_authority import (
    build_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_coordinate_validation_receipt import (
    RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE,
    build_subject_mask_coordinate_validation_receipt,
    stamp_subject_mask_coordinate_validation_receipt,
)
from tests.unit.fisheye.test_subject_mask_coordinate_publication import (
    _subject_fixture,
)


_SOURCE_RUN_PATH = "subject_mask_runs/source_v1"
_SOURCE_MANIFEST = {
    "schema_id": "palette.subject_mask.core_run_manifest",
    "schema_version": 5,
    "digest_algorithm": "sha256_canonical_json_v1",
    "payload_digest": "a" * 64,
    "payload": {
        "run_id": "source_v1",
        "stage_family": "subject_mask_runs",
        "kind": "raw_probability_uint8",
        "logical_content": {"digest": "b" * 64},
    },
}
_BUNDLE_MANIFEST = {
    "schema_id": "palette.subject_mask.bundle_manifest",
    "schema_version": 3,
    "payload": {
        "members": {
            "raw": {
                "family": "subject_mask_runs",
                "run_path": _SOURCE_RUN_PATH,
                "manifest_schema_id": _SOURCE_MANIFEST["schema_id"],
                "manifest_schema_version": _SOURCE_MANIFEST["schema_version"],
                "manifest_payload_digest": _SOURCE_MANIFEST["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(_SOURCE_MANIFEST),
                "logical_content_digest": "b" * 64,
            }
        }
    },
}
_PAYLOAD_FILE_EQUIVALENCE = {
    "schema_id": "palette.coordinate_successor_payload_file_equivalence",
    "schema_version": 1,
    "receipt_digest": "c" * 64,
    "inventory_digest": "d" * 64,
    "payload_file_count": 0,
}


class _DenseReadTrap:
    """Metadata-compatible array proxy that fails on scientific payload reads."""

    def __init__(
        self,
        node: Any,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        path: str | None = None,
    ) -> None:
        self._node = node
        self.shape = node.shape if shape is None else shape
        self.dtype = node.dtype if dtype is None else dtype
        self.path = node.path if path is None else path
        self.attrs = node.attrs
        self._coordinate_archive_token = node._coordinate_archive_token

    def __getitem__(self, key: Any) -> Any:
        raise AssertionError(f"dense or companion payload was indexed: {key!r}")


def test_complete_historical_successor_reinstalls_its_sealed_crop_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True, fresh=True)
    run.attrs["coordinate_successor_historical_crop_adapter"] = {
        "schema_id": "test.adapter",
        "schema_version": 1,
    }
    binding = object()
    sentinel = object()
    active = False

    monkeypatch.setattr(
        publication_module,
        "_load_persisted_historical_crop_successor_binding",
        lambda *_args, **_kwargs: binding,
    )

    @contextmanager
    def adapter_scope(value: Any) -> Any:
        nonlocal active
        assert value is binding
        active = True
        try:
            yield
        finally:
            active = False

    import fisheye.shared.zarr.historical_geometry_only_crop_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module,
        "historical_geometry_only_crop_loader",
        adapter_scope,
    )

    def load_impl(*_args: Any, **_kwargs: Any) -> Any:
        assert active is True
        return sentinel

    monkeypatch.setattr(
        publication_module,
        "_load_subject_mask_coordinate_context_impl",
        load_impl,
    )
    result = publication_module._load_subject_mask_coordinate_context(
        root,
        run.path,
        require_complete=True,
        expected_selector_eligible=False,
    )
    assert result is sentinel
    assert active is False


def _pointer(value: Any) -> dict[str, str]:
    return {"record_ref": value.record_ref, "record_sha256": value.record_sha256}


def _prepare_receipt_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, dict[str, Any]]:
    root, parent, run = _subject_fixture(monkeypatch, published=True, fresh=True)
    monkeypatch.setattr(
        publication_module,
        "validate_subject_mask_core_run_manifest",
        lambda _manifest: (),
    )
    monkeypatch.setattr(
        publication_module,
        "validate_subject_mask_bundle_manifest",
        lambda _manifest: (),
    )
    source_run = parent.create_group("source_v1")
    source_run.attrs["run_manifest"] = copy.deepcopy(_SOURCE_MANIFEST)
    target_manifest = {
        "schema_id": "palette.subject_mask.core_run_manifest",
        "schema_version": 5,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": "e" * 64,
        "payload": {
            "run_id": "s1",
            "stage_family": "subject_mask_runs",
            "kind": "raw_probability_uint8",
            "source": {
                "run_path": _SOURCE_RUN_PATH,
                "manifest_digest": canonical_json_sha256(_SOURCE_MANIFEST),
                "manifest": copy.deepcopy(_SOURCE_MANIFEST),
            },
            "logical_content": {"digest": "b" * 64},
        },
    }
    run.attrs["run_manifest"] = target_manifest
    padded = {"schema_id": "palette.padded_crop_lineage", "schema_version": 1}
    run.attrs["coordinate_successor_padded_crop_lineage"] = padded
    run.attrs["coordinate_successor_padded_crop_lineage_sha256"] = (
        canonical_json_sha256(padded)
    )
    context = publication_module._load_subject_mask_coordinate_context(
        root,
        "subject_mask_runs/s1",
        require_complete=False,
        expected_selector_eligible=False,
    )
    records = {
        "context": _pointer(context.context_record),
        "derivation": _pointer(
            bind_persisted_coordinate_record(
                run,
                attr_name=publication_module.SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
            )
        ),
        "padded_crop_lineage": _pointer(
            bind_persisted_coordinate_record(
                run,
                attr_name="coordinate_successor_padded_crop_lineage",
            )
        ),
        "row_identity": _pointer(context.row_identity),
        "surface_inventory": _pointer(
            bind_persisted_coordinate_record(
                run,
                attr_name=publication_module.SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
            )
        ),
        "temporal_authority": _pointer(context.temporal_authority),
    }
    receipt = build_subject_mask_coordinate_validation_receipt(
        kind=RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
        successor_run_path="subject_mask_runs/s1",
        source={
            "run_path": _SOURCE_RUN_PATH,
            "core_manifest_payload_digest": _SOURCE_MANIFEST["payload_digest"],
            "core_manifest_document_digest": canonical_json_sha256(_SOURCE_MANIFEST),
            "logical_content_digest": "b" * 64,
        },
        source_validation={
            "schema_id": "palette.subject_mask.source_validation_receipt",
            "schema_version": 2,
            "payload_digest": "f" * 64,
            "document_sha256": "1" * 64,
            "semantic_unit_count": 1,
        },
        bundle_authority={
            "kind": "inactive_subject_mask_bundle_v3",
            "document_digest": canonical_json_sha256(_BUNDLE_MANIFEST),
        },
        coordinate_records=records,
        coordinate_record_names=publication_module._RAW_COORDINATE_VALIDATION_RECORD_NAMES,
        payload_equivalence=_PAYLOAD_FILE_EQUIVALENCE,
        validator_identity={"package": "fisheye.shared", "version": "test"},
    )
    stamp_subject_mask_coordinate_validation_receipt(
        run,
        receipt,
        expected_kind=RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
        expected_successor_run_path="subject_mask_runs/s1",
        expected_coordinate_record_names=publication_module._RAW_COORDINATE_VALIDATION_RECORD_NAMES,
    )
    receipt_pointer = _pointer(
        bind_persisted_coordinate_record(
            run,
            attr_name=SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
        )
    )
    authority_records = {**records, "coordinate_validation_receipt": receipt_pointer}
    authority = build_coordinate_successor_authority(
        kind="raw_subject_mask_coordinate_successor",
        source_family="subject_mask_runs",
        source_run_path=_SOURCE_RUN_PATH,
        source_manifest=_SOURCE_MANIFEST,
        source_authority_kind="inactive_subject_mask_bundle_v3",
        source_authority=_BUNDLE_MANIFEST,
        successor_family="subject_mask_runs",
        successor_run_path="subject_mask_runs/s1",
        payload_equivalence={
            "policy": "same_filesystem_hardlink_payload_exact_logical_digest_v2",
            "source_logical_content_digest": "b" * 64,
            "payload_file_equivalence": _PAYLOAD_FILE_EQUIVALENCE,
        },
        coordinate_records=authority_records,
    )
    stamp_coordinate_successor_authority(run, authority)
    return root, run, receipt


def _trap_dense_and_companion_arrays(run: Any, *, mask_probs_node: Any | None = None) -> None:
    dense_paths = (
        "mask_probs_roi",
        "masks_roi",
        "metrics/centroid_xy",
        "metrics/bbox_xyxy",
        "available_channels",
        "metrics/prob_max",
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_valid",
        "metrics/bbox_valid",
    )
    for path in dense_paths:
        parent = run
        parts = path.split("/")
        for part in parts[:-1]:
            parent = parent[part]
        node = parent[parts[-1]]
        if path == "mask_probs_roi" and mask_probs_node is not None:
            node = mask_probs_node
        parent.children[parts[-1]] = _DenseReadTrap(node)


def test_no_receipt_loader_keeps_full_scan_path(monkeypatch: pytest.MonkeyPatch) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True, fresh=True)
    called = 0
    original = publication_module._validate_companion_metadata_and_values

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        publication_module,
        "_validate_companion_metadata_and_values",
        counted,
    )
    loaded = publication_module._load_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        require_complete=False,
        expected_selector_eligible=False,
    )
    assert type(loaded) is publication_module.BoundSubjectMaskCoordinateSurfaces
    assert called == 1
    assert run.attrs.get(SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE) is None


def test_valid_receipt_loader_never_reads_dense_or_companion_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _receipt = _prepare_receipt_successor(monkeypatch)
    _trap_dense_and_companion_arrays(run)

    loaded = publication_module._load_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        require_complete=False,
        expected_selector_eligible=False,
    )

    assert type(loaded) is publication_module.BoundSubjectMaskCoordinateSurfaces
    assert loaded.inventory.record == loaded.context._run_group.attrs[
        publication_module.SUBJECT_MASK_SURFACE_INVENTORY_ATTR
    ]


def test_receipt_loader_requires_the_live_source_core_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _receipt = _prepare_receipt_successor(monkeypatch)
    del root["subject_mask_runs/source_v1"]
    _trap_dense_and_companion_arrays(run)

    with pytest.raises(ValueError, match="source core is unavailable"):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


@pytest.mark.parametrize("tamper", ("receipt", "authority", "record"))
def test_receipt_loader_rejects_stale_authority_or_record_without_dense_read(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    root, run, receipt = _prepare_receipt_successor(monkeypatch)
    if tamper == "receipt":
        run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE] = "0" * 64
    elif tamper == "authority":
        authority = copy.deepcopy(run.attrs["coordinate_successor_authority"])
        authority["payload"]["source"]["logical_content_digest"] = "f" * 64
        run.attrs["coordinate_successor_authority"] = authority
    else:
        updated = copy.deepcopy(receipt)
        updated["payload"]["coordinate_records"]["context"]["record_sha256"] = "f" * 64
        body = {key: value for key, value in updated.items() if key != "payload_digest"}
        updated["payload_digest"] = canonical_json_sha256(body)
        run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE] = updated
        run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE] = (
            canonical_json_sha256(updated)
        )
    _trap_dense_and_companion_arrays(run)

    with pytest.raises(ValueError):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


@pytest.mark.parametrize("field", ("shape", "dtype", "path"))
def test_receipt_loader_rejects_live_inventory_metadata_mismatch_without_dense_read(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root, run, _receipt = _prepare_receipt_successor(monkeypatch)
    original = run["mask_probs_roi"]
    if field == "shape":
        replacement = _DenseReadTrap(
            original,
            shape=(original.shape[0] + 1, *original.shape[1:]),
        )
    elif field == "dtype":
        replacement = _DenseReadTrap(original, dtype="<f4")
    else:
        replacement = _DenseReadTrap(original, path="subject_mask_runs/s1/other")
    run.children["mask_probs_roi"] = replacement
    _trap_dense_and_companion_arrays(run, mask_probs_node=replacement)

    with pytest.raises(ValueError):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )
