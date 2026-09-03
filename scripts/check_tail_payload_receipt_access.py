#!/usr/bin/env python3
"""Keep maintained tail-kinematics access behind sealed payload receipts."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys


DEFAULT_SOURCE_ROOT = Path("src/fisheye")
CANONICAL_OWNER = "src/fisheye/shared/tail_coordinate_publication.py"
TAIL_RUN_WRITER = "src/fisheye/analysis/tail_kinematics_runs.py"
TAIL_MATERIALIZER = "src/fisheye/analysis_workflows/materializers/tail_kinematics.py"
STRICT_PUBLISHER_CALLERS = {TAIL_RUN_WRITER, TAIL_MATERIALIZER}
PRIVATE_LOADER_KINDS = {
    "src/fisheye/analysis_workflows/materializers/tail_posture.py": "tail_posture_view",
    "src/fisheye/diagnostics/tail_posture_candidate_execution.py": "tail_posture_view",
    "src/fisheye/diagnostics/benchmark_tail_posture_view_v3_candidate.py": (
        "tail_posture_view"
    ),
    "src/fisheye/diagnostics/tail_kinematics_candidate_execution.py": "tail_kinematics",
    "src/fisheye/diagnostics/benchmark_tail_kinematics_candidate_reads.py": (
        "tail_kinematics"
    ),
}
LEGACY_LOADER_NAMES = {
    "load_legacy_tail_kinematics_coordinate_publication",
    "_load_legacy_tail_kinematics_coordinate_publication",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TailReceiptAccessViolation:
    relative_path: str
    line: int
    reason: str


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _literal_keyword(node: ast.Call, name: str) -> object:
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


class _TailReceiptVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.violations: list[TailReceiptAccessViolation] = []

    def _record(self, node: ast.AST, reason: str) -> None:
        self.violations.append(
            TailReceiptAccessViolation(
                relative_path=self.relative_path,
                line=int(getattr(node, "lineno", 0)),
                reason=reason,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in LEGACY_LOADER_NAMES:
            self._record(node, "defines a receipt-free tail-kinematics loader")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name in LEGACY_LOADER_NAMES:
            self._record(node, "defines a receipt-free tail-kinematics loader")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names = {alias.name for alias in node.names}
        if self.relative_path != CANONICAL_OWNER and names & LEGACY_LOADER_NAMES:
            self._record(
                node,
                "imports the receipt-free tail-kinematics compatibility loader",
            )
        if (
            self.relative_path not in {TAIL_RUN_WRITER, TAIL_MATERIALIZER}
            and "write_tail_kinematics_run_group" in names
        ):
            self._record(
                node,
                "imports the low-level tail writer outside its atomic materializer",
            )
        if (
            self.relative_path not in STRICT_PUBLISHER_CALLERS
            and "publish_tail_kinematics_coordinate_surfaces" in names
        ):
            self._record(
                node,
                "imports the tail publisher outside its atomic materializer",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if self.relative_path != CANONICAL_OWNER:
            if name in LEGACY_LOADER_NAMES:
                self._record(
                    node,
                    "calls the receipt-free tail-kinematics compatibility loader",
                )
            elif name == "_publish_tail_coordinate_surfaces":
                self._record(node, "calls the private receipt-optional publisher")
            elif (
                name == "publish_tail_kinematics_coordinate_surfaces"
                and self.relative_path not in STRICT_PUBLISHER_CALLERS
            ):
                self._record(
                    node,
                    "calls the tail publisher outside its atomic materializer",
                )
            elif name == "_load_tail_coordinate_publication":
                expected = PRIVATE_LOADER_KINDS.get(self.relative_path)
                observed = _literal_keyword(node, "expected_kind")
                if expected is None:
                    self._record(
                        node,
                        "calls the private tail loader outside its closed allowlist",
                    )
                elif any(
                    keyword.arg == "require_payload_receipt"
                    for keyword in node.keywords
                ):
                    self._record(
                        node,
                        "attempts to make the receipt contract caller-selectable",
                    )
                elif observed != expected:
                    self._record(
                        node,
                        "private tail loader does not declare its exact artifact kind",
                    )
            elif (
                name == "write_tail_kinematics_run_group"
                and self.relative_path not in {TAIL_RUN_WRITER, TAIL_MATERIALIZER}
            ):
                self._record(
                    node,
                    "calls the low-level tail writer outside its atomic materializer",
                )
        self.generic_visit(node)


def collect_tail_payload_receipt_access_violations(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    repo_root: Path | None = None,
) -> list[TailReceiptAccessViolation]:
    repo = (repo_root or _repo_root()).resolve()
    source = source_root if source_root.is_absolute() else repo / source_root
    violations: list[TailReceiptAccessViolation] = []
    for path in sorted(source.rglob("*.py")):
        relative_path = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        visitor = _TailReceiptVisitor(relative_path=relative_path)
        visitor.visit(tree)
        violations.extend(visitor.violations)
    return violations


def check_tail_payload_receipt_access(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    repo_root: Path | None = None,
) -> int:
    violations = collect_tail_payload_receipt_access_violations(
        source_root,
        repo_root=repo_root,
    )
    if violations:
        print(
            "Noncanonical tail-kinematics payload access is forbidden; use the "
            "receipt-backed loader and atomic materializer:",
            file=sys.stderr,
        )
        for item in violations:
            print(
                f"  {item.relative_path}:{item.line}: {item.reason}",
                file=sys.stderr,
            )
        return 1
    print("Tail payload-receipt access ratchet passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args(argv)
    return check_tail_payload_receipt_access(source_root=args.source_root)


if __name__ == "__main__":
    raise SystemExit(main())
