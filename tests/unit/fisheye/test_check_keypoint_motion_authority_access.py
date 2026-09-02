from __future__ import annotations

from pathlib import Path

from scripts import check_keypoint_motion_authority_access as mod


def _write_module(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_ast_census_accepts_shared_resolver_and_finds_direct_lineage_reads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/fisheye"
    _write_module(
        source / "shared/keypoint_motion_authority.py",
        "def resolve_keypoint_motion_authority(root, requested):\n"
        "    return root.attrs.get('source_crop_run')\n",
    )
    _write_module(
        source / "reader.py",
        "def direct(group, attrs):\n"
        "    first = group.attrs.get('source_keypoints_run')\n"
        "    return attrs['source_crop_run']\n",
    )

    reads, forbidden = mod.collect_keypoint_motion_authority_accesses(
        source,
        repo_root=tmp_path,
    )

    assert forbidden == []
    assert len(reads) == 2
    assert {read.symbol for read in reads} == {"direct"}


def test_new_private_resolver_and_strict_runtime_bypass_are_forbidden(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/fisheye"
    _write_module(
        source / "consumer.py",
        "def resolve_keypoint_group(root):\n"
        "    return root\n",
    )
    _write_module(
        source / "analysis/track_kinematics.py",
        "def run(keypoints):\n"
        "    crop = keypoints.attrs.get('source_crop_run')\n"
        "    return keypoints['heading'], crop\n",
    )

    _, forbidden = mod.collect_keypoint_motion_authority_accesses(
        source,
        repo_root=tmp_path,
    )

    assert {
        (item.relative_path, item.reason)
        for item in forbidden
    } == {
        (
            "src/fisheye/consumer.py",
            "defines a private keypoint-motion resolver",
        ),
        (
            "src/fisheye/analysis/track_kinematics.py",
            "runtime consumer reads keypoint lineage attrs directly",
        ),
        (
            "src/fisheye/analysis/track_kinematics.py",
            "runtime consumer reads legacy embedded heading directly",
        ),
    }


def test_canonical_resolver_names_cannot_be_redefined_outside_owner(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/fisheye"
    names = (
        "resolve_keypoint_lineage_authority",
        "resolve_keypoint_motion_authority",
    )
    for index, name in enumerate(names):
        _write_module(
            source / f"consumer_{index}.py",
            f"def {name}(root):\n    return root\n",
        )

    _, forbidden = mod.collect_keypoint_motion_authority_accesses(
        source,
        repo_root=tmp_path,
    )

    assert {
        (item.relative_path, item.symbol, item.reason)
        for item in forbidden
    } == {
        (
            f"src/fisheye/consumer_{index}.py",
            name,
            "defines a private keypoint-motion resolver",
        )
        for index, name in enumerate(names)
    }


def test_ratchet_rejects_new_reads_and_tightens_removed_reads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/fisheye"
    module = source / "reader.py"
    baseline = tmp_path / "baseline.json"
    _write_module(
        module,
        "def read(group):\n    return group.attrs.get('source_crop_run')\n",
    )
    reads, forbidden = mod.collect_keypoint_motion_authority_accesses(
        source,
        repo_root=tmp_path,
    )
    assert forbidden == []
    mod._write_baseline(baseline, reads)

    assert mod.check_keypoint_motion_authority_access(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 0

    _write_module(
        module,
        "def read(group):\n"
        "    first = group.attrs.get('source_crop_run')\n"
        "    return group.attrs.get('source_keypoints_run')\n",
    )
    assert mod.check_keypoint_motion_authority_access(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 1

    _write_module(module, "def read(group):\n    return None\n")
    assert mod.check_keypoint_motion_authority_access(
        source_root=source,
        baseline_path=baseline,
        repo_root=tmp_path,
    ) == 0
    assert mod._read_baseline(baseline) == {}
