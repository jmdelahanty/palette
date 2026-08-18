from __future__ import annotations

import numpy as np
import pytest

import fisheye.shared.refined_subject_mask_coordinate_publication as module


class _CountingArray:
    _archive = object()

    def __init__(self, *, path: str, values: object) -> None:
        self.path = path
        self._coordinate_archive_token = self._archive
        self.attrs: dict[str, object] = {}
        self._values = np.asarray(values)
        self._reported_shape = self._values.shape
        self.reads = 0

    @property
    def shape(self) -> tuple[int, ...]:
        return self._reported_shape

    @property
    def dtype(self) -> np.dtype:
        return self._values.dtype

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads += 1
        return self._values[key]


def test_payload_cache_reuses_only_complete_evidence_and_returns_copies() -> None:
    node = _CountingArray(
        path="refined_subject_masks_runs/run/metrics/area_px",
        values=np.arange(4, dtype=np.float32),
    )

    with module._payload_cache_scope():
        first = module._payload(node)
        first["shape"][0] = 999
        second = module._payload(node)

    assert node.reads == 1
    assert second["shape"] == [4]
    assert second is not first


def test_payload_cache_is_call_scoped() -> None:
    node = _CountingArray(
        path="refined_subject_masks_runs/run/metrics/area_px",
        values=np.arange(4, dtype=np.float32),
    )

    module._payload(node)
    module._payload(node)
    assert node.reads == 2

    with module._payload_cache_scope():
        module._payload(node)
        module._payload(node)
    assert node.reads == 3

    with module._payload_cache_scope():
        module._payload(node)
        module._payload(node)
    assert node.reads == 4


def test_payload_cache_rechecks_path_and_misses_after_metadata_change() -> None:
    node = _CountingArray(
        path="refined_subject_masks_runs/run/metrics/area_px",
        values=np.arange(4, dtype=np.float32),
    )

    with module._payload_cache_scope():
        module._payload(node)
        node.path = "refined_subject_masks_runs/other/metrics/area_px"
        changed = module._payload(node)

    assert node.reads == 2
    assert changed["array_ref"] == "/refined_subject_masks_runs/other/metrics/area_px"


def test_payload_cache_rechecks_shape_and_fails_closed_after_metadata_change() -> None:
    node = _CountingArray(
        path="refined_subject_masks_runs/run/metrics/area_px",
        values=np.arange(4, dtype=np.float32),
    )

    with module._payload_cache_scope():
        module._payload(node)
        node._reported_shape = (5,)
        with pytest.raises(
            module.RefinedSubjectMaskCoordinatePublicationError,
            match="changed shape",
        ):
            module._payload(node)

    assert node.reads == 2
