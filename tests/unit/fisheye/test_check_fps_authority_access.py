from __future__ import annotations

from pathlib import Path

from scripts import check_fps_authority_access as mod


def _write_module(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_ast_census_accepts_canonical_resolver_and_finds_direct_reads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/fisheye"
    _write_module(
        source / "reader.py",
        "from fisheye.shared.metadata import resolve_fps\n"
        "def canonical(root):\n"
        "    return resolve_fps(root)\n"
        "def direct(root, root_attrs):\n"
        "    first = root.attrs.get('fps', 60.0)\n"
        "    second = root_attrs['video_fps']\n"
        "    return helper(root.attrs, 'frames_per_second')\n",
    )

    reads, forbidden = mod.collect_fps_authority_accesses(
        source,
        repo_root=tmp_path,
    )

    assert forbidden == []
    assert len(reads) == 3
    assert {read.symbol for read in reads} == {"direct"}


def test_legacy_find_fps_is_forbidden_outside_declared_owner(tmp_path: Path) -> None:
    source = tmp_path / "src/fisheye"
    _write_module(
        source / "consumer.py",
        "from fisheye.analysis.compute_speed import find_fps as legacy_fps\n"
        "def run(root, console):\n"
        "    return legacy_fps(root, console)\n",
    )
    _write_module(
        source / "analysis/compute_speed.py",
        "def find_fps(root, console):\n"
        "    return 60.0\n"
        "def main(root, console):\n"
        "    return find_fps(root, console)\n",
    )

    _, forbidden = mod.collect_fps_authority_accesses(
        source,
        repo_root=tmp_path,
    )

    assert [(item.relative_path, item.reason) for item in forbidden] == [
        ("src/fisheye/consumer.py", "imports legacy find_fps"),
    ]


def test_new_private_fps_resolver_is_forbidden(tmp_path: Path) -> None:
    source = tmp_path / "src/fisheye"
    _write_module(
        source / "consumer.py",
        "def _resolve_fps(root):\n"
        "    return root.attrs.get('fps', 60.0)\n",
    )

    _, forbidden = mod.collect_fps_authority_accesses(
        source,
        repo_root=tmp_path,
    )

    assert [(item.relative_path, item.reason) for item in forbidden] == [
        ("src/fisheye/consumer.py", "defines noncanonical FPS resolver"),
    ]


def test_ratchet_rejects_new_reads_and_tightens_removed_reads(tmp_path: Path) -> None:
    source = tmp_path / "src/fisheye"
    module = source / "reader.py"
    baseline = tmp_path / "baseline.json"
    _write_module(
        module,
        "def read(root):\n    return root.attrs.get('fps')\n",
    )
    reads, forbidden = mod.collect_fps_authority_accesses(
        source,
        repo_root=tmp_path,
    )
    assert forbidden == []
    mod._write_baseline(baseline, reads)

    assert mod.check_fps_authority_access(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 0

    _write_module(
        module,
        "def read(root):\n"
        "    first = root.attrs.get('fps')\n"
        "    return root.attrs.get('video_fps')\n",
    )
    assert mod.check_fps_authority_access(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 1

    _write_module(module, "def read(root):\n    return 30.0\n")
    assert mod.check_fps_authority_access(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 0
    assert mod._read_baseline(baseline) == {}
