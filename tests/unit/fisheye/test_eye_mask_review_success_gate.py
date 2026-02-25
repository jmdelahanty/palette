from __future__ import annotations

import numpy as np

from fisheye.tune import eye_mask_review as mod


def test_compute_success_mask_applies_min_eye_area_gate() -> None:
    ellipse_success = np.array(
        [
            [True, True],
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )
    eye_separation = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    area_refined = np.array(
        [
            [40.0, 75.0],  # left eye below threshold
            [80.0, 95.0],  # both eyes above threshold
            [90.0, 90.0],  # one eye failed ellipse_success
        ],
        dtype=np.float32,
    )

    success = mod._compute_success_mask(
        ellipse_success,
        eye_separation,
        min_sep=None,
        max_sep=None,
        area_refined=area_refined,
        min_eye_area_px=50.0,
    )

    np.testing.assert_array_equal(success, np.array([False, True, False], dtype=bool))


def test_load_area_refined_prefers_masks_over_metrics(monkeypatch) -> None:
    class _FakeArray:
        def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
            self._data = np.asarray(data)
            self.shape = self._data.shape
            self.chunks = chunks

        def __getitem__(self, item):
            return self._data[item]

    class _FakeGroup(dict):
        def __init__(self, *args, attrs: dict[str, object] | None = None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.attrs = attrs or {}

        def get(self, key: str, default=None):
            return super().get(key, default)

    monkeypatch.setattr(mod.zarr, "Group", _FakeGroup)

    masks = np.zeros((2, 2, 4, 4), dtype=np.uint8)
    masks[0, 0, 0:2, 0:2] = 1  # left area=4
    masks[0, 1, 2:4, 2:4] = 1  # right area=4
    masks[1, 0, 1, 1] = 1      # left area=1
    # right eye remains empty on row 1

    refined = _FakeGroup(
        {
            "masks_roi": _FakeArray(masks, chunks=(1, 2, 4, 4)),
            "metrics": _FakeGroup(
                {
                    "area_refined": _FakeArray(
                        np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)
                    )
                }
            ),
        },
        attrs={},
    )

    area = mod._load_area_refined(refined)
    assert area is not None
    np.testing.assert_array_equal(area, np.array([[4.0, 4.0], [1.0, 0.0]], dtype=np.float32))
