#!/usr/bin/env python3
"""Keep the maintained chaser planner on the shared core-authority resolver."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys

DEFAULT_PLANNER = Path(
    "src/fisheye/utils/materialize_composable_chaser_successor_cohort.py"
)
FORBIDDEN_NAMES = frozenset({"MOTION_BOUT_PAIRS", "_resolve_motion_bouts"})
FORBIDDEN_CALLS = frozenset(
    {
        "load_provider_track_motion_source_handle",
        "load_exact_selector_ineligible_default_swim_bout_tables",
        "load_default_swim_bout_tables",
    }
)
FORBIDDEN_ARGUMENTS = frozenset(
    {
        "--motion-run",
        "--provider-motion-run-path",
        "--swim-bout-run",
        "--swim-bout-run-name",
        "--track-id",
    }
)
REQUIRED_CALLS = frozenset(
    {
        "selected_core_track_id_from_roster",
        "_core_authority_plan_binding",
        "_revalidate_core_bundle_selection",
    }
)
REQUIRED_ARGUMENTS = frozenset(
    {
        "--core-authority-roster",
        "--expected-core-authority-roster-sha256",
        "--core-track-id",
    }
)


@dataclass(frozen=True)
class ParadigmCoreAuthorityViolation:
    line: int
    reason: str


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class _PlannerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[ParadigmCoreAuthorityViolation] = []
        self.calls: set[str] = set()
        self.strings: set[str] = set()

    def _record(self, node: ast.AST, reason: str) -> None:
        self.violations.append(
            ParadigmCoreAuthorityViolation(
                line=int(getattr(node, "lineno", 0)),
                reason=reason,
            )
        )

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self._record(node, f"uses retired core-source selector {node.id!r}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in FORBIDDEN_NAMES:
            self._record(node, f"defines retired core-source selector {node.name!r}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name in FORBIDDEN_NAMES:
            self._record(node, f"defines retired core-source selector {node.name!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name is not None:
            self.calls.add(name)
        if name in FORBIDDEN_CALLS:
            self._record(node, f"calls independent core-source resolver {name!r}")
        for keyword in node.keywords:
            if (
                keyword.arg in {"track_id", "selected_track_id"}
                and isinstance(keyword.value, ast.Constant)
                and type(keyword.value.value) is int
                and keyword.value.value == 0
            ):
                self._record(
                    keyword,
                    "hard-codes implicit core track zero instead of roster selection",
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.add(node.value)
            if node.value in FORBIDDEN_ARGUMENTS:
                self._record(
                    node,
                    f"invokes retired independent core-source argument {node.value!r}",
                )
        self.generic_visit(node)


def collect_paradigm_core_authority_violations(
    planner_path: Path = DEFAULT_PLANNER,
    *,
    repo_root: Path | None = None,
) -> list[ParadigmCoreAuthorityViolation]:
    repo = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    planner = planner_path if planner_path.is_absolute() else repo / planner_path
    if not planner.is_file():
        return [
            ParadigmCoreAuthorityViolation(
                line=0,
                reason=f"maintained chaser planner is absent: {planner}",
            )
        ]
    tree = ast.parse(planner.read_text(encoding="utf-8"), filename=str(planner))
    visitor = _PlannerVisitor()
    visitor.visit(tree)
    for name in sorted(REQUIRED_CALLS - visitor.calls):
        visitor.violations.append(
            ParadigmCoreAuthorityViolation(
                line=0,
                reason=f"does not invoke shared core-authority boundary {name!r}",
            )
        )
    for argument in sorted(REQUIRED_ARGUMENTS - visitor.strings):
        visitor.violations.append(
            ParadigmCoreAuthorityViolation(
                line=0,
                reason=f"does not propagate frozen core argument {argument!r}",
            )
        )
    return sorted(visitor.violations, key=lambda item: (item.line, item.reason))


def check_paradigm_core_authority_access(
    *,
    planner_path: Path = DEFAULT_PLANNER,
    repo_root: Path | None = None,
) -> int:
    violations = collect_paradigm_core_authority_violations(
        planner_path,
        repo_root=repo_root,
    )
    if violations:
        print(
            "Maintained paradigm core-authority access is not canonical:",
            file=sys.stderr,
        )
        for item in violations:
            print(f"  line {item.line}: {item.reason}", file=sys.stderr)
        return 1
    print("Paradigm core-authority access ratchet passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    args = parser.parse_args(argv)
    return check_paradigm_core_authority_access(planner_path=args.planner)


if __name__ == "__main__":
    raise SystemExit(main())
