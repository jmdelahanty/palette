#!/usr/bin/env python3
"""Reject private copies of the heading convention outside its shared owner.

The heading angle, the left axis and the angle-difference wrap live in
``fisheye.shared.heading_convention``. This AST census of ``src/fisheye``
fails on:

* the degree or radian wrap formula -- ``(x + 180) % 360 - 180``,
  ``(x + pi) % (2 * pi) - pi``, their ``np.mod``/``np.remainder`` spellings
  and ``arctan2(sin(x), cos(x))`` -- anywhere outside the owner;
* a y-flipped ``arctan2(-y, x)`` / ``atan2(-y, x)`` heading expression, or
  the equivalent ``-degrees(arctan2(y, x))``, outside the owner.

Each allowlisted site is genuinely not the Palette heading or cannot move
without changing published values; an allowlist entry that no longer matches
anything fails too, so the list only shrinks.

``src/chaser_analysis`` is legacy code outside ``src/fisheye`` and is
deliberately not scanned.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys


DEFAULT_SOURCE_ROOT = Path("src/fisheye")
CANONICAL_OWNER = "src/fisheye/shared/heading_convention.py"
# Legacy tree, not under src/fisheye and not scanned (listed so the
# exclusion is explicit rather than an accident of the scan root).
EXCLUDED_LEGACY_ROOTS = ("src/chaser_analysis",)

# "path::qualname" -> number of findings the site is allowed. Keep minimal.
ALLOWLIST: dict[str, int] = {
    # Drawing angle in the chaser-centred (y-up) frame, with a direction-of-
    # travel fallback where the body heading is missing: not a Palette
    # heading. Documented as non-conforming in heading_convention.
    "src/fisheye/visualization/chaser_ring_traversal.py::_chase_frame_heading": 2,
    # Audit residual wrapped with arctan2(sin, cos) into the closed [-pi, pi];
    # wrap_delta_radians returns [-pi, pi) and differs in the last bits, so
    # moving it would change the audit's reported residuals.
    "src/fisheye/analysis/megabouts_convention_audit.py::_angle_residual": 1,
}

_ARCTAN2 = frozenset({"arctan2", "atan2"})
_DEGREES = frozenset({"degrees", "rad2deg"})
_MOD = frozenset({"mod", "remainder"})


@dataclass(frozen=True)
class HeadingFinding:
    relative_path: str
    symbol: str
    line: int
    reason: str

    @property
    def key(self) -> str:
        return f"{self.relative_path}::{self.symbol}"


def _callee(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
    return None


def _number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _is_pi(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "pi") or (
        isinstance(node, ast.Name) and node.id == "pi"
    )


def _is_two_pi(node: ast.AST) -> bool:
    if (isinstance(node, ast.Attribute) and node.attr == "tau") or (
        isinstance(node, ast.Name) and node.id in {"tau", "TWO_PI", "TAU"}
    ):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return (_number(node.left) == 2.0 and _is_pi(node.right)) or (
            _is_pi(node.left) and _number(node.right) == 2.0
        )
    return False


def _half_turn(node: ast.AST) -> str | None:
    if _number(node) == 180.0:
        return "degrees"
    if _is_pi(node):
        return "radians"
    return None


def _full_turn_matches(node: ast.AST, unit: str) -> bool:
    return _number(node) == 360.0 if unit == "degrees" else _is_two_pi(node)


def _shifted_mod(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """Return (shifted, modulus) for ``shifted % modulus`` or ``np.mod(...)``."""

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return node.left, node.right
    if _callee(node) in _MOD and len(node.args) == 2:  # type: ignore[union-attr]
        return node.args[0], node.args[1]  # type: ignore[union-attr]
    return None


def _is_wrap(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        unit = _half_turn(node.right)
        parts = _shifted_mod(node.left)
        if unit is None or parts is None:
            return False
        shifted, modulus = parts
        return (
            isinstance(shifted, ast.BinOp)
            and isinstance(shifted.op, ast.Add)
            and (_half_turn(shifted.right) == unit or _half_turn(shifted.left) == unit)
            and _full_turn_matches(modulus, unit)
        )
    if _callee(node) in _ARCTAN2 and len(node.args) == 2:  # type: ignore[union-attr]
        first, second = node.args  # type: ignore[union-attr]
        return _callee(first) == "sin" and _callee(second) == "cos"
    return False


def _is_flipped_heading(node: ast.AST) -> bool:
    if _callee(node) in _ARCTAN2 and node.args:  # type: ignore[union-attr]
        first = node.args[0]  # type: ignore[union-attr]
        return isinstance(first, ast.UnaryOp) and isinstance(first.op, ast.USub)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        return _callee(inner) in _DEGREES and bool(inner.args) and _callee(inner.args[0]) in _ARCTAN2  # type: ignore[union-attr]
    return False


class _HeadingVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.findings: list[HeadingFinding] = []

    def _symbol(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, node.name)

    def _record(self, node: ast.AST, reason: str) -> None:
        self.findings.append(
            HeadingFinding(self.relative_path, self._symbol(), int(getattr(node, "lineno", 0)), reason)
        )

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if _is_wrap(node):
            self._record(node, "private angle-wrap formula; use wrap_delta_degrees/wrap_delta_radians")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if _is_flipped_heading(node):
            self._record(node, "private heading expression; use heading_degrees/image_angle_degrees")
            return  # the inner arctan2 is the same finding
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_wrap(node):
            self._record(node, "private angle-wrap formula; use wrap_delta_degrees/wrap_delta_radians")
        elif _is_flipped_heading(node):
            self._record(node, "private heading expression; use heading_degrees/image_angle_degrees")
        self.generic_visit(node)


def collect_heading_findings(source_root: Path, *, repo_root: Path) -> list[HeadingFinding]:
    repo = repo_root.resolve()
    source = source_root if source_root.is_absolute() else repo / source_root
    findings: list[HeadingFinding] = []
    for path in sorted(source.rglob("*.py")):
        relative_path = path.relative_to(repo).as_posix()
        if relative_path == CANONICAL_OWNER or relative_path.startswith(
            tuple(root + "/" for root in EXCLUDED_LEGACY_ROOTS)
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        visitor = _HeadingVisitor(relative_path=relative_path)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


def check_heading_convention_access(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    repo_root: Path | None = None,
    allowlist: dict[str, int] | None = None,
) -> int:
    repo = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    allowed = Counter(ALLOWLIST if allowlist is None else allowlist)
    findings = collect_heading_findings(source_root, repo_root=repo)
    current = Counter(finding.key for finding in findings)
    excess = current - allowed
    stale = allowed - current
    status = 0
    if excess:
        print(
            "Heading convention ratchet failed; use fisheye.shared.heading_convention "
            "instead of a private copy (or, for a genuinely different angle, add a "
            "justified ALLOWLIST entry):",
            file=sys.stderr,
        )
        for finding in findings:
            if finding.key in excess:
                print(f"  {finding.relative_path}:{finding.line} ({finding.symbol}): {finding.reason}", file=sys.stderr)
        status = 1
    if stale:
        print("Heading convention allowlist is stale; shrink these entries:", file=sys.stderr)
        for key, count in sorted(stale.items()):
            print(f"  {key} (allowed {allowed[key]}, found {current[key]})", file=sys.stderr)
        status = 1
    if status == 0:
        print(f"Heading convention ratchet passed: {sum(current.values())} allowlisted site(s).")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args(argv)
    return check_heading_convention_access(source_root=args.source_root)


if __name__ == "__main__":
    raise SystemExit(main())
