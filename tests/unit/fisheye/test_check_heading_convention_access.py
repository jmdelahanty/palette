from __future__ import annotations

from pathlib import Path

from scripts import check_heading_convention_access as mod


def _write_module(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _reasons(tmp_path: Path) -> list[tuple[str, str, str]]:
    return [
        (item.relative_path, item.symbol, item.reason.split(";")[0])
        for item in mod.collect_heading_findings(Path("src/fisheye"), repo_root=tmp_path)
    ]


def test_repository_has_no_private_heading_copies() -> None:
    assert mod.check_heading_convention_access() == 0


def test_shared_owner_and_legacy_tree_are_not_scanned(tmp_path: Path) -> None:
    body = "def w(d):\n    return (d + 180.0) % 360.0 - 180.0\n"
    _write_module(tmp_path / mod.CANONICAL_OWNER, body)
    _write_module(tmp_path / "src/chaser_analysis/old.py", body)

    assert _reasons(tmp_path) == []
    assert mod.EXCLUDED_LEGACY_ROOTS == ("src/chaser_analysis",)


def test_degree_and_radian_wraps_are_rejected(tmp_path: Path) -> None:
    _write_module(
        tmp_path / "src/fisheye/a.py",
        "import math\n"
        "import numpy as np\n"
        "def deg(d):\n    return ((d + 180) % 360.0) - 180.0\n"
        "def rad(d):\n    return (d + np.pi) % (2 * np.pi) - np.pi\n"
        "def rad_math(d):\n    return (d + math.pi) % math.tau - math.pi\n"
        "def mod_call(d):\n    return np.mod(d + 180.0, 360.0) - 180.0\n"
        "def trig(d):\n    return np.arctan2(np.sin(d), np.cos(d))\n"
        "def inline(a, b):\n    x = (a - b + 180.0) % 360.0 - 180.0\n    return abs(x)\n"
        "def not_a_wrap(d):\n    return (d + 90.0) % 360.0 - 90.0, d % 360.0\n",
    )

    assert [symbol for _, symbol, _ in _reasons(tmp_path)] == [
        "deg",
        "rad",
        "rad_math",
        "mod_call",
        "trig",
        "inline",
    ]


def test_flipped_heading_expressions_are_rejected(tmp_path: Path) -> None:
    _write_module(
        tmp_path / "src/fisheye/b.py",
        "import math\n"
        "import numpy as np\n"
        "class C:\n"
        "    def heading(self, f):\n        return np.rad2deg(np.arctan2(-f[:, 1], f[:, 0]))\n"
        "def scalar(fx, fy):\n    return math.degrees(math.atan2(-fy, fx))\n"
        "def negated(dx, dy):\n    return -np.degrees(np.arctan2(dy, dx))\n"
        "def fine(left, forward):\n    return np.degrees(np.arctan2(left, forward))\n",
    )

    assert _reasons(tmp_path) == [
        ("src/fisheye/b.py", "C.heading", "private heading expression"),
        ("src/fisheye/b.py", "scalar", "private heading expression"),
        ("src/fisheye/b.py", "negated", "private heading expression"),
    ]


def test_allowlist_permits_exact_count_and_fails_when_stale(tmp_path: Path, capsys) -> None:
    _write_module(
        tmp_path / "src/fisheye/c.py",
        "import numpy as np\n"
        "def draw(v):\n    return np.degrees(np.arctan2(-v[1], v[0]))\n",
    )
    key = "src/fisheye/c.py::draw"

    assert mod.check_heading_convention_access(repo_root=tmp_path, allowlist={key: 1}) == 0
    assert mod.check_heading_convention_access(repo_root=tmp_path, allowlist={}) == 1
    assert "c.py:3 (draw)" in capsys.readouterr().err
    assert mod.check_heading_convention_access(repo_root=tmp_path, allowlist={key: 1, "x.py::gone": 1}) == 1
    assert "stale" in capsys.readouterr().err
