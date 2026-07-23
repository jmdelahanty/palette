from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from fisheye.shared.coordinate_reference import (
    ATTRS_REFERENCE_EXTENT_SCHEMA_ID,
    BoundReferenceExtent,
    CoordinateReferenceError,
    bind_array_reference_extent,
    bind_attrs_reference_extent,
    bind_persisted_record_reference_extent,
    canonical_node_path,
    verify_bound_reference_extent,
)


class _Node:
    _archive = object()

    def __init__(
        self,
        *,
        path: str,
        shape: tuple[int, ...] = (3, 4512, 4512),
        dtype: str = "u1",
        attrs: dict[str, object] | None = None,
    ) -> None:
        self.path = path
        self._coordinate_archive_token = self._archive
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.attrs = {} if attrs is None else attrs


def test_array_reference_is_derived_from_exact_path_shape_and_dtype() -> None:
    node = _Node(path="raw_video/images_full")
    bound = bind_array_reference_extent(node, units="px")

    assert bound.record_ref == "/raw_video/images_full@zarr_metadata"
    assert bound.selector == "shape[-2:]"
    assert (bound.width, bound.height, bound.units) == (4512, 4512, "px")
    assert bound.authority_record["shape"] == [3, 4512, 4512]
    assert verify_bound_reference_extent(bound) is bound

    changed = bind_array_reference_extent(
        _Node(path="raw_video/images_full", shape=(3, 4512, 4511)),
        units="px",
    )
    assert changed.record_sha256 != bound.record_sha256


def test_attrs_reference_evaluates_exact_selected_values() -> None:
    node = _Node(
        path="analysis/arena_frame",
        attrs={"width_mm": 84.5, "height_mm": 83.25},
    )
    bound = bind_attrs_reference_extent(
        node,
        width_attr="width_mm",
        height_attr="height_mm",
        units="mm",
    )

    assert bound.authority_kind == ATTRS_REFERENCE_EXTENT_SCHEMA_ID
    assert bound.record_ref == "/analysis/arena_frame"
    assert bound.selector == "attrs[width_mm,height_mm]"
    assert (bound.width, bound.height) == (84.5, 83.25)

    node.attrs["width_mm"] = 99.0
    with pytest.raises(CoordinateReferenceError, match="changed after it was bound"):
        verify_bound_reference_extent(bound)


@pytest.mark.parametrize(
    "path",
    (
        "analysis/../wrong",
        "analysis/./wrong",
        "/analysis/wrong",
        "analysis//wrong",
        "analysis/wrong/",
        " analysis/wrong",
    ),
)
def test_reference_nodes_reject_noncanonical_paths(path: str) -> None:
    with pytest.raises(CoordinateReferenceError):
        canonical_node_path(_Node(path=path))


@pytest.mark.parametrize(
    "invalid_width",
    (4512.5, 4512.0, np.float64(4512.0), np.int64(4512)),
)
def test_pixel_extent_requires_exact_python_integer_authority(
    invalid_width: object,
) -> None:
    node = _Node(
        path="analysis/frame",
        attrs={"width_px": invalid_width, "height_px": 4512},
    )
    with pytest.raises(CoordinateReferenceError, match="exact positive integer"):
        bind_attrs_reference_extent(
            node,
            width_attr="width_px",
            height_attr="height_px",
            units="px",
        )


def test_array_pixel_extent_rejects_non_python_integer_shape_metadata() -> None:
    node = _Node(path="raw_video/images_full")
    node.shape = (3, np.int64(4512), 4512)
    with pytest.raises(CoordinateReferenceError, match="exact Python integer"):
        bind_array_reference_extent(node, units="px")


def test_free_constructed_bound_authority_is_not_writer_evidence() -> None:
    forged = BoundReferenceExtent(
        record_ref="/analysis/unrelated@zarr_metadata",
        record_sha256="0" * 64,
        selector="shape[-2:]",
        width=4512,
        height=4512,
        units="px",
        authority_kind="invented",
        authority_record={},
        _archive_identity=bind_array_reference_extent(
            _Node(path="analysis/forged", shape=(1, 4512, 4512)),
            units="px",
        ).archive_identity,
        _authority_node=_Node(path="analysis/unrelated"),
        _verification_seal=object(),
    )
    with pytest.raises(CoordinateReferenceError, match="BoundReferenceExtent"):
        verify_bound_reference_extent(forged)


def test_bound_reference_subclass_is_not_canonical_authority() -> None:
    class _Subclass(BoundReferenceExtent):
        pass

    bound = bind_array_reference_extent(
        _Node(path="analysis/reference", shape=(1, 10, 20)),
        units="px",
    )
    subclass = _Subclass(
        record_ref=bound.record_ref,
        record_sha256=bound.record_sha256,
        selector=bound.selector,
        width=bound.width,
        height=bound.height,
        units=bound.units,
        authority_kind=bound.authority_kind,
        authority_record=bound.authority_record,
        _archive_identity=bound.archive_identity,
        _authority_node=bound._authority_node,
        _verification_seal=bound._verification_seal,
    )
    with pytest.raises(CoordinateReferenceError, match="BoundReferenceExtent"):
        verify_bound_reference_extent(subclass)
    with pytest.raises(CoordinateReferenceError, match="verified authority loader"):
        subclass.assert_verified()


def test_attrs_reference_rejects_missing_or_ambiguous_attr_names() -> None:
    node = _Node(path="analysis/frame", attrs={"same": 10})
    with pytest.raises(CoordinateReferenceError):
        bind_attrs_reference_extent(
            node,
            width_attr="same",
            height_attr="same",
            units="px",
        )


def test_persisted_reference_record_binds_digest_and_direct_attrs() -> None:
    record = {
        "schema_id": "palette.arena_geometry_reference",
        "schema_version": 1,
        "arena_region_width_px": 358,
        "arena_region_height_px": 358,
        "units": "px",
    }
    import hashlib
    import json

    digest = hashlib.sha256(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    node = _Node(
        path="analysis/stimulus_runs/s1/calibration/arena_geometry",
        attrs={
            "arena_geometry_record": record,
            "arena_geometry_record_sha256": digest,
            "arena_region_width_px": 358,
            "arena_region_height_px": 358,
        },
    )
    bound = bind_persisted_record_reference_extent(
        node,
        record_attr="arena_geometry_record",
        digest_attr="arena_geometry_record_sha256",
        width_field="arena_region_width_px",
        height_field="arena_region_height_px",
        units_field="units",
    )
    assert bound.record_sha256 == digest
    assert bound.record_ref.endswith("@arena_geometry_record")
    assert (bound.width, bound.height) == (358, 358)
    assert verify_bound_reference_extent(bound) is bound

    node.attrs["arena_region_width_px"] = 999
    with pytest.raises(CoordinateReferenceError, match="conflicts"):
        verify_bound_reference_extent(bound)
    with pytest.raises(CoordinateReferenceError):
        bind_attrs_reference_extent(
            node,
            width_attr="unknown",
            height_attr="height_px",
            units="px",
        )


def test_persisted_reference_rejects_ordered_dict_and_numpy_scalars() -> None:
    import hashlib
    import json

    ordered = OrderedDict(
        (
            ("schema_id", "palette.arena_geometry_reference"),
            ("schema_version", 1),
            ("arena_region_width_px", 358),
            ("arena_region_height_px", 358),
            ("units", "px"),
        )
    )
    digest = hashlib.sha256(
        json.dumps(
            ordered,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    node = _Node(
        path="analysis/ordered",
        attrs={
            "arena_geometry_record": ordered,
            "arena_geometry_record_sha256": digest,
            "arena_region_width_px": 358,
            "arena_region_height_px": 358,
        },
    )
    with pytest.raises(CoordinateReferenceError, match="exact built-in dict"):
        bind_persisted_record_reference_extent(
            node,
            record_attr="arena_geometry_record",
            digest_attr="arena_geometry_record_sha256",
            width_field="arena_region_width_px",
            height_field="arena_region_height_px",
            units_field="units",
        )

    noncanonical_scalar = dict(ordered)
    noncanonical_scalar["arena_region_width_px"] = np.int64(358)
    node.attrs["arena_geometry_record"] = noncanonical_scalar
    with pytest.raises(CoordinateReferenceError, match="exact built-in JSON"):
        bind_persisted_record_reference_extent(
            node,
            record_attr="arena_geometry_record",
            digest_attr="arena_geometry_record_sha256",
            width_field="arena_region_width_px",
            height_field="arena_region_height_px",
            units_field="units",
        )


def test_persisted_reference_record_roundtrip_detects_record_drift() -> None:
    import hashlib
    import json

    record = {
        "schema_id": "palette.arena_geometry_reference",
        "schema_version": 1,
        "arena_region_width_px": 320,
        "arena_region_height_px": 240,
        "units": "px",
    }
    digest = hashlib.sha256(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    node = _Node(
        path="analysis/arena_geometry",
        attrs={
            "arena_geometry_record": record,
            "arena_geometry_record_sha256": digest,
            "arena_region_width_px": 320,
            "arena_region_height_px": 240,
        },
    )
    bound = bind_persisted_record_reference_extent(
        node,
        record_attr="arena_geometry_record",
        digest_attr="arena_geometry_record_sha256",
        width_field="arena_region_width_px",
        height_field="arena_region_height_px",
        units_field="units",
    )
    assert verify_bound_reference_extent(bound) is bound


def test_persisted_reference_requires_one_canonical_digested_units_field() -> None:
    import hashlib
    import json

    missing_units = {
        "schema_id": "palette.arena_geometry_reference",
        "schema_version": 1,
        "arena_region_width_px": 320,
        "arena_region_height_px": 240,
    }

    def digest(record: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    node = _Node(
        path="analysis/arena_geometry",
        attrs={
            "arena_geometry_record": missing_units,
            "arena_geometry_record_sha256": digest(missing_units),
            "arena_region_width_px": 320,
            "arena_region_height_px": 240,
        },
    )
    with pytest.raises(CoordinateReferenceError, match="lacks 'units'"):
        bind_persisted_record_reference_extent(
            node,
            record_attr="arena_geometry_record",
            digest_attr="arena_geometry_record_sha256",
            width_field="arena_region_width_px",
            height_field="arena_region_height_px",
            units_field="units",
        )

    ambiguous = dict(missing_units, units="px", alternate_units="mm")
    node.attrs["arena_geometry_record"] = ambiguous
    node.attrs["arena_geometry_record_sha256"] = digest(ambiguous)
    with pytest.raises(CoordinateReferenceError, match="canonical digested 'units'"):
        bind_persisted_record_reference_extent(
            node,
            record_attr="arena_geometry_record",
            digest_attr="arena_geometry_record_sha256",
            width_field="arena_region_width_px",
            height_field="arena_region_height_px",
            units_field="alternate_units",
        )

    bound = bind_persisted_record_reference_extent(
        node,
        record_attr="arena_geometry_record",
        digest_attr="arena_geometry_record_sha256",
        width_field="arena_region_width_px",
        height_field="arena_region_height_px",
        units_field="units",
    )
    assert bound.units == "px"
    assert bound.authority_scope == "extent_only"

    node.attrs["arena_geometry_record"]["arena_region_width_px"] = 321
    node.attrs["arena_region_width_px"] = 321
    node.attrs["arena_geometry_record_sha256"] = hashlib.sha256(
        json.dumps(
            node.attrs["arena_geometry_record"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CoordinateReferenceError, match="changed after it was bound"):
        verify_bound_reference_extent(bound)
