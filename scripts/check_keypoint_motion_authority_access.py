#!/usr/bin/env python3
"""Ratchet direct keypoint-motion lineage reads and private resolvers."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys


BASELINE_SCHEMA_VERSION = 1
DEFAULT_SOURCE_ROOT = Path("src/fisheye")
DEFAULT_BASELINE = Path(__file__).with_name(
    "keypoint_motion_authority_access_ratchet_baseline.json"
)
CANONICAL_OWNER = "src/fisheye/shared/keypoint_motion_authority.py"
STRICT_RUNTIME_CONSUMERS = frozenset(
    {
        "src/fisheye/analysis/track_kinematics.py",
        "src/fisheye/tracking/arena_assignment.py",
    }
)
LINEAGE_KEYS = frozenset(
    {
        "source_crop_run",
        "source_keypoint_run",
        "source_keypoints_run",
    }
)
PRIVATE_RESOLVER_NAMES = frozenset(
    {
        "_resolve_keypoint_group",
        "resolve_keypoint_group",
        "_resolve_keypoint_lineage",
        "resolve_keypoint_lineage",
        "_resolve_keypoint_lineage_context",
        "resolve_keypoint_lineage_authority",
        "_resolve_keypoint_motion_authority",
        "resolve_keypoint_motion_authority",
    }
)
GRANDFATHERED_PRIVATE_RESOLVERS = frozenset(
    {
        "src/fisheye/analysis/gaze_convention_validation.py::_resolve_keypoint_group",
        "src/fisheye/diagnostics/benchmark_eye_assignment_split.py::_resolve_keypoint_group",
        "src/fisheye/visualization/visualize_swim_bladder_mask_patches.py::_resolve_keypoint_group",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DirectLineageRead:
    key: str
    relative_path: str
    symbol: str
    line: int
    expression: str


@dataclass(frozen=True)
class ForbiddenAuthorityAccess:
    relative_path: str
    symbol: str
    line: int
    reason: str


def _literal_key(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    return node.value


def _is_attrs_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "attrs" or node.id.endswith("_attrs")
    return isinstance(node, ast.Attribute) and node.attr == "attrs"


class _AuthorityVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.symbol_stack: list[str] = []
        self.direct_reads: list[DirectLineageRead] = []
        self.forbidden: list[ForbiddenAuthorityAccess] = []

    @property
    def _symbol(self) -> str:
        return ".".join(self.symbol_stack) or "<module>"

    def _record_direct_read(self, node: ast.AST) -> None:
        if self.relative_path == CANONICAL_OWNER:
            return
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        digest = sha256(normalized.encode("utf-8")).hexdigest()[:20]
        key = f"{self.relative_path}::{self._symbol}::{digest}"
        self.direct_reads.append(
            DirectLineageRead(
                key=key,
                relative_path=self.relative_path,
                symbol=self._symbol,
                line=int(getattr(node, "lineno", 0)),
                expression=normalized,
            )
        )
        if self.relative_path in STRICT_RUNTIME_CONSUMERS:
            self._record_forbidden(
                node,
                reason="runtime consumer reads keypoint lineage attrs directly",
            )

    def _record_forbidden(self, node: ast.AST, *, reason: str) -> None:
        self.forbidden.append(
            ForbiddenAuthorityAccess(
                relative_path=self.relative_path,
                symbol=self._symbol,
                line=int(getattr(node, "lineno", 0)),
                reason=reason,
            )
        )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = f"{self.relative_path}::{node.name}"
        self.symbol_stack.append(node.name)
        try:
            if (
                node.name in PRIVATE_RESOLVER_NAMES
                and self.relative_path != CANONICAL_OWNER
                and qualified not in GRANDFATHERED_PRIVATE_RESOLVERS
            ):
                self._record_forbidden(
                    node,
                    reason="defines a private keypoint-motion resolver",
                )
            self.generic_visit(node)
        finally:
            self.symbol_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == "KeypointResolution" and self.relative_path != CANONICAL_OWNER:
            self._record_forbidden(
                node,
                reason="defines the retired consumer-local KeypointResolution",
            )
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "fisheye.analysis.track_kinematics" and any(
            alias.name in {"resolve_keypoint_group", "KeypointResolution"}
            for alias in node.names
        ):
            self._record_forbidden(
                node,
                reason="imports a retired track-local keypoint resolver",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        literal = _literal_key(node.slice)
        if (
            isinstance(node.ctx, ast.Load)
            and _is_attrs_expression(node.value)
            and literal in LINEAGE_KEYS
        ):
            self._record_direct_read(node)
        if (
            isinstance(node.ctx, ast.Load)
            and self.relative_path in STRICT_RUNTIME_CONSUMERS
            and literal == "heading"
        ):
            self._record_forbidden(
                node,
                reason="runtime consumer reads legacy embedded heading directly",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "get"
            and _is_attrs_expression(target.value)
            and node.args
            and _literal_key(node.args[0]) in LINEAGE_KEYS
        ):
            self._record_direct_read(node)
        elif (
            node.args
            and _is_attrs_expression(node.args[0])
            and any(_literal_key(value) in LINEAGE_KEYS for value in node.args[1:])
        ):
            self._record_direct_read(node)
        if (
            self.relative_path in STRICT_RUNTIME_CONSUMERS
            and isinstance(target, ast.Attribute)
            and target.attr == "get"
            and node.args
            and _literal_key(node.args[0]) == "heading"
        ):
            self._record_forbidden(
                node,
                reason="runtime consumer reads legacy embedded heading directly",
            )
        self.generic_visit(node)


def collect_keypoint_motion_authority_accesses(
    source_root: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[list[DirectLineageRead], list[ForbiddenAuthorityAccess]]:
    repo = (repo_root or _repo_root()).resolve()
    source = source_root if source_root.is_absolute() else repo / source_root
    reads: list[DirectLineageRead] = []
    forbidden: list[ForbiddenAuthorityAccess] = []
    for path in sorted(source.rglob("*.py")):
        relative_path = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        visitor = _AuthorityVisitor(relative_path=relative_path)
        visitor.visit(tree)
        reads.extend(visitor.direct_reads)
        forbidden.extend(visitor.forbidden)
    return reads, forbidden


def _counter(reads: list[DirectLineageRead]) -> Counter[str]:
    return Counter(read.key for read in reads)


def _read_baseline(path: Path) -> Counter[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != BASELINE_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported keypoint-motion authority baseline: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(
            f"Keypoint-motion authority baseline entries must be an object: {path}"
        )
    result: Counter[str] = Counter()
    for key, count in entries.items():
        if not isinstance(key, str) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid keypoint-motion baseline entry: {key!r}")
        result[key] = count
    if payload.get("direct_read_count") != sum(result.values()):
        raise ValueError(f"Keypoint-motion baseline count is inconsistent: {path}")
    return result


def _write_baseline(path: Path, reads: list[DirectLineageRead]) -> None:
    counts = _counter(reads)
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "source_root": DEFAULT_SOURCE_ROOT.as_posix(),
        "direct_read_count": sum(counts.values()),
        "entries": dict(sorted(counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_keypoint_motion_authority_access(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    baseline_path: Path = DEFAULT_BASELINE,
    repo_root: Path | None = None,
    update_on_shrink: bool = True,
) -> int:
    repo = (repo_root or _repo_root()).resolve()
    baseline = baseline_path if baseline_path.is_absolute() else repo / baseline_path
    reads, forbidden = collect_keypoint_motion_authority_accesses(
        source_root,
        repo_root=repo,
    )
    if forbidden:
        print(
            "Noncanonical keypoint-motion authority access is forbidden; use "
            "fisheye.shared.keypoint_motion_authority:",
            file=sys.stderr,
        )
        for item in forbidden:
            print(
                f"  {item.relative_path}:{item.line} ({item.symbol}): {item.reason}",
                file=sys.stderr,
            )
        return 1

    current = _counter(reads)
    expected = _read_baseline(baseline)
    added = current - expected
    removed = expected - current
    if added:
        details = {read.key: read for read in reads}
        print(
            "Keypoint-motion authority ratchet failed; new direct lineage reads "
            "must use the shared resolver or a typed producer validator:",
            file=sys.stderr,
        )
        for key, count in sorted(added.items()):
            read = details[key]
            print(
                f"  {read.relative_path}:{read.line} ({read.symbol}) x{count}",
                file=sys.stderr,
            )
        return 1

    if removed and update_on_shrink:
        _write_baseline(baseline, reads)
        print(
            "Tightened keypoint-motion authority baseline by "
            f"{sum(removed.values())} read(s)."
        )
    elif removed:
        print(
            "Keypoint-motion authority baseline can shrink by "
            f"{sum(removed.values())} read(s)."
        )
    print(
        "Keypoint-motion authority access ratchet passed: "
        f"{sum(current.values())} grandfathered direct read(s)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--no-update-on-shrink", action="store_true")
    args = parser.parse_args(argv)
    if args.write_baseline:
        reads, forbidden = collect_keypoint_motion_authority_accesses(
            args.source_root
        )
        if forbidden:
            for item in forbidden:
                print(
                    f"{item.relative_path}:{item.line}: {item.reason}",
                    file=sys.stderr,
                )
            return 1
        baseline = args.baseline
        if not baseline.is_absolute():
            baseline = _repo_root() / baseline
        _write_baseline(baseline, reads)
        print(
            "Wrote keypoint-motion authority baseline with "
            f"{len(reads)} direct read(s): {baseline}"
        )
        return 0
    return check_keypoint_motion_authority_access(
        source_root=args.source_root,
        baseline_path=args.baseline,
        update_on_shrink=not args.no_update_on_shrink,
    )


if __name__ == "__main__":
    raise SystemExit(main())
