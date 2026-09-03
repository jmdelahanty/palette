from __future__ import annotations

from pathlib import Path

from scripts import check_tail_payload_receipt_access as mod


def _write(repo: Path, relative_path: str, source: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _violations(repo: Path) -> list[mod.TailReceiptAccessViolation]:
    return mod.collect_tail_payload_receipt_access_violations(
        Path("src/fisheye"),
        repo_root=repo,
    )


def test_allows_exact_kinds_on_closed_candidate_paths(tmp_path) -> None:
    _write(
        tmp_path,
        "src/fisheye/diagnostics/tail_kinematics_candidate_execution.py",
        """
def check(publication, root, path):
    return publication._load_tail_coordinate_publication(
        root,
        path,
        expected_selector_eligible=False,
        expected_kind="tail_kinematics",
        require_complete=True,
    )
""",
    )
    _write(
        tmp_path,
        "src/fisheye/diagnostics/tail_posture_candidate_execution.py",
        """
def check(publication, root, path):
    return publication._load_tail_coordinate_publication(
        root,
        path,
        expected_selector_eligible=False,
        expected_kind="tail_posture_view",
        require_complete=True,
    )
""",
    )

    assert _violations(tmp_path) == []


def test_rejects_caller_selectable_receipt_policy(tmp_path) -> None:
    _write(
        tmp_path,
        "src/fisheye/diagnostics/tail_kinematics_candidate_execution.py",
        """
def check(publication, root, path):
    return publication._load_tail_coordinate_publication(
        root,
        path,
        expected_kind="tail_kinematics",
        require_payload_receipt=False,
    )
""",
    )

    assert [item.reason for item in _violations(tmp_path)] == [
        "attempts to make the receipt contract caller-selectable"
    ]


def test_rejects_legacy_loader_and_low_level_writer_in_maintained_code(
    tmp_path,
) -> None:
    _write(
        tmp_path,
        "src/fisheye/analysis/consumer.py",
        """
from fisheye.analysis.tail_kinematics_runs import write_tail_kinematics_run_group
from fisheye.shared.tail_coordinate_publication import (
    load_legacy_tail_kinematics_coordinate_publication,
)

def consume(root, path):
    load_legacy_tail_kinematics_coordinate_publication(root, path)
    return write_tail_kinematics_run_group(root)
""",
    )

    reasons = [item.reason for item in _violations(tmp_path)]
    assert sorted(reasons) == sorted(
        [
            "imports the receipt-free tail-kinematics compatibility loader",
            "imports the low-level tail writer outside its atomic materializer",
            "calls the receipt-free tail-kinematics compatibility loader",
            "calls the low-level tail writer outside its atomic materializer",
        ]
    )


def test_rejects_private_receipt_optional_publisher(tmp_path) -> None:
    _write(
        tmp_path,
        "src/fisheye/analysis/consumer.py",
        """
def publish(module, root, run):
    return module._publish_tail_coordinate_surfaces(
        root,
        run,
        kind="tail_kinematics",
    )
""",
    )

    assert [item.reason for item in _violations(tmp_path)] == [
        "calls the private receipt-optional publisher"
    ]


def test_rejects_public_publisher_outside_atomic_materializer(tmp_path) -> None:
    _write(
        tmp_path,
        "src/fisheye/analysis/consumer.py",
        """
from fisheye.shared.tail_coordinate_publication import (
    publish_tail_kinematics_coordinate_surfaces,
)

def publish(root, run, evidence):
    return publish_tail_kinematics_coordinate_surfaces(
        root,
        run,
        **evidence,
    )
""",
    )

    assert [item.reason for item in _violations(tmp_path)] == [
        "imports the tail publisher outside its atomic materializer",
        "calls the tail publisher outside its atomic materializer",
    ]
