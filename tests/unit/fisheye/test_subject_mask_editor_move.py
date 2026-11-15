"""Move tool in the shipped mask editor: drag the active mask, nudge with arrows."""

from __future__ import annotations

import numpy as np

from tests.unit.fisheye.test_subject_mask_editor_pieces import _as_mask, _run

SHAPE = (12, 14)


def _mask():
    mask = np.zeros(SHAPE, np.uint8)
    mask[3:6, 4:8] = 1
    return mask


def _shift(mask, dx, dy):
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    keep = (xs + dx >= 0) & (xs + dx < mask.shape[1]) & (ys + dy >= 0) & (ys + dy < mask.shape[0])
    out[ys[keep] + dy, xs[keep] + dx] = 1
    return out


def _event(x, y):
    return f"{{x:{x},y:{y},shiftKey:false,preventDefault(){{}}}}"


def test_translate_mask_shifts_and_counts_pixels_outside():
    pure = (
        "(()=>{const a=translateMask(mask,maskWidth,maskHeight,3,-2);"
        "const b=translateMask(mask,maskWidth,maskHeight,9,0);"
        "return {a:Array.from(a.mask),lostA:a.lost,b:Array.from(b.mask),lostB:b.lost};})()"
    )
    out = _run(_mask(), [], pure=pure)["pure"]
    np.testing.assert_array_equal(_as_mask(out["a"], SHAPE), _shift(_mask(), 3, -2))
    assert out["lostA"] == 0
    np.testing.assert_array_equal(_as_mask(out["b"], SHAPE), _shift(_mask(), 9, 0))
    assert out["lostB"] == int(_mask().sum() - _shift(_mask(), 9, 0).sum()) > 0


def test_drag_moves_the_whole_mask_and_one_undo_reverts_drag_and_nudges():
    out = _run(_mask(), [
        {"name": "tool", "press": {"key": "m"}},
        {"name": "drag", "run": f"beginCanvasEdit({_event(5, 4)});moveCanvasEdit({_event(7, 5)});endCanvasEdit()"},
        {"name": "nudge", "press": {"key": "ArrowRight"}},
        {"name": "undo", "run": "undoBulkEdit()"},
    ])
    np.testing.assert_array_equal(_as_mask(out["drag"]["mask"], SHAPE), _shift(_mask(), 2, 1))
    assert "Moved the mask 2, 1 px" in out["drag"]["status"] and out["drag"]["undoDisabled"] is False
    assert out["drag"]["unloadBlocked"] is True
    np.testing.assert_array_equal(_as_mask(out["nudge"]["mask"], SHAPE), _shift(_mask(), 3, 1))
    np.testing.assert_array_equal(_as_mask(out["undo"]["mask"], SHAPE), _mask())
    assert out["undo"]["unloadBlocked"] is False


def test_arrows_nudge_only_in_move_mode_and_shift_steps_ten():
    out = _run(_mask(), [
        {"name": "paint_arrow", "press": {"key": "ArrowDown"}},
        {"name": "tool", "press": {"key": "m"}},
        {"name": "down", "press": {"key": "ArrowDown"}},
        {"name": "shift_left", "press": {"key": "ArrowLeft", "mods": {"shiftKey": True}}},
    ])
    np.testing.assert_array_equal(_as_mask(out["paint_arrow"]["mask"], SHAPE), _mask())
    np.testing.assert_array_equal(_as_mask(out["down"]["mask"], SHAPE), _shift(_mask(), 0, 1))
    np.testing.assert_array_equal(_as_mask(out["shift_left"]["mask"], SHAPE), _shift(_mask(), -10, 1))
    assert "outside the crop" in out["shift_left"]["status"]


def test_pixels_pushed_past_the_edge_return_when_moved_back():
    out = _run(_mask(), [
        {"name": "tool", "press": {"key": "m"}},
        {"name": "far", "run": "nudgeMask(-6, 0)"},
        {"name": "back", "run": "nudgeMask(6, 0)"},
    ])
    assert int(np.sum(out["far"]["mask"])) < int(_mask().sum())
    np.testing.assert_array_equal(_as_mask(out["back"]["mask"], SHAPE), _mask())


def test_switching_tools_ends_the_move_so_undo_reverts_only_the_next_one():
    out = _run(_mask(), [
        {"name": "tool", "press": {"key": "m"}},
        {"name": "first", "press": {"key": "ArrowRight"}},
        {"name": "paint", "press": {"key": "b"}},
        {"name": "tool2", "press": {"key": "m"}},
        {"name": "second", "press": {"key": "ArrowDown"}},
        {"name": "undo", "run": "undoBulkEdit()"},
    ])
    np.testing.assert_array_equal(_as_mask(out["second"]["mask"], SHAPE), _shift(_mask(), 1, 1))
    np.testing.assert_array_equal(_as_mask(out["undo"]["mask"], SHAPE), _shift(_mask(), 1, 0))
