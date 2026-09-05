#!/usr/bin/env python3
"""Reject legacy FPS resolution and ratchet direct timing-metadata reads."""

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
    "fps_authority_access_ratchet_baseline.json"
)
CANONICAL_FPS_OWNER = "src/fisheye/shared/metadata.py"
LEGACY_FPS_OWNER = "src/fisheye/analysis/compute_speed.py"
FPS_KEYS = frozenset({"fps", "video_fps", "frames_per_second"})
GRANDFATHERED_LOCAL_FPS_RESOLVERS = frozenset(
    {
        "src/fisheye/analysis/megabouts_classifier.py::_resolve_fps",
        "src/fisheye/analysis/tail_kinematics_io.py::_resolve_fps",
        "src/fisheye/tune/video_detect_review_backend.py::_resolve_fps",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DirectFpsRead:
    key: str
    relative_path: str
    symbol: str
    line: int
    expression: str


@dataclass(frozen=True)
class ForbiddenLegacyAccess:
    relative_path: str
    symbol: str
    line: int
    reason: str


def _literal_timing_key(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    return node.value if node.value in FPS_KEYS else None


def _contains_timing_key(nodes: list[ast.AST]) -> bool:
    return any(_literal_timing_key(node) is not None for node in nodes)


def _is_attrs_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "attrs" or node.id.endswith("_attrs")
    return isinstance(node, ast.Attribute) and node.attr == "attrs"


class _FpsAuthorityVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.symbol_stack: list[str] = []
        self.direct_reads: list[DirectFpsRead] = []
        self.forbidden_legacy_accesses: list[ForbiddenLegacyAccess] = []

    @property
    def _symbol(self) -> str:
        return ".".join(self.symbol_stack) or "<module>"

    def _record_direct_read(self, node: ast.AST) -> None:
        if self.relative_path == CANONICAL_FPS_OWNER:
            return
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        digest = sha256(normalized.encode("utf-8")).hexdigest()[:20]
        key = f"{self.relative_path}::{self._symbol}::{digest}"
        self.direct_reads.append(
            DirectFpsRead(
                key=key,
                relative_path=self.relative_path,
                symbol=self._symbol,
                line=int(getattr(node, "lineno", 0)),
                expression=normalized,
            )
        )

    def _record_legacy(self, node: ast.AST, *, reason: str) -> None:
        if self.relative_path == LEGACY_FPS_OWNER:
            return
        self.forbidden_legacy_accesses.append(
            ForbiddenLegacyAccess(
                relative_path=self.relative_path,
                symbol=self._symbol,
                line=int(getattr(node, "lineno", 0)),
                reason=reason,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "find_fps":
            self._record_legacy(node, reason="defines legacy find_fps")
        elif (
            node.name in {"get_fps", "resolve_fps", "_resolve_fps"}
            and self.relative_path != CANONICAL_FPS_OWNER
            and f"{self.relative_path}::{node.name}"
            not in GRANDFATHERED_LOCAL_FPS_RESOLVERS
        ):
            self._record_legacy(node, reason="defines noncanonical FPS resolver")
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == "find_fps":
            self._record_legacy(node, reason="defines legacy find_fps")
        elif (
            node.name in {"get_fps", "resolve_fps", "_resolve_fps"}
            and self.relative_path != CANONICAL_FPS_OWNER
            and f"{self.relative_path}::{node.name}"
            not in GRANDFATHERED_LOCAL_FPS_RESOLVERS
        ):
            self._record_legacy(node, reason="defines noncanonical FPS resolver")
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "find_fps" for alias in node.names):
            self._record_legacy(node, reason="imports legacy find_fps")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and _is_attrs_expression(node.value)
            and _literal_timing_key(node.slice) is not None
        ):
            self._record_direct_read(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        if (
            isinstance(target, ast.Name)
            and target.id == "find_fps"
        ) or (
            isinstance(target, ast.Attribute)
            and target.attr == "find_fps"
        ):
            self._record_legacy(node, reason="calls legacy find_fps")

        if (
            isinstance(target, ast.Attribute)
            and target.attr == "get"
            and _is_attrs_expression(target.value)
            and node.args
            and _literal_timing_key(node.args[0]) is not None
        ):
            self._record_direct_read(node)
        elif (
            node.args
            and _is_attrs_expression(node.args[0])
            and _contains_timing_key(list(node.args[1:]))
        ):
            self._record_direct_read(node)
        self.generic_visit(node)


def collect_fps_authority_accesses(
    source_root: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[list[DirectFpsRead], list[ForbiddenLegacyAccess]]:
    repo = (repo_root or _repo_root()).resolve()
    source = source_root if source_root.is_absolute() else repo / source_root
    direct_reads: list[DirectFpsRead] = []
    forbidden: list[ForbiddenLegacyAccess] = []
    for path in sorted(source.rglob("*.py")):
        relative_path = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        visitor = _FpsAuthorityVisitor(relative_path=relative_path)
        visitor.visit(tree)
        direct_reads.extend(visitor.direct_reads)
        forbidden.extend(visitor.forbidden_legacy_accesses)
    return direct_reads, forbidden


def _counter(reads: list[DirectFpsRead]) -> Counter[str]:
    return Counter(read.key for read in reads)


def _read_baseline(path: Path) -> Counter[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != BASELINE_SCHEMA_VERSION
    ):
        raise ValueError(f"Unsupported FPS authority baseline: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"FPS authority baseline entries must be an object: {path}")
    counter: Counter[str] = Counter()
    for key, count in entries.items():
        if not isinstance(key, str) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid FPS authority baseline entry: {key!r}")
        counter[key] = count
    if payload.get("direct_read_count") != sum(counter.values()):
        raise ValueError(f"FPS authority baseline count is inconsistent: {path}")
    return counter


def _write_baseline(path: Path, reads: list[DirectFpsRead]) -> None:
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


def check_fps_authority_access(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    baseline_path: Path = DEFAULT_BASELINE,
    repo_root: Path | None = None,
    update_on_shrink: bool = True,
) -> int:
    repo = (repo_root or _repo_root()).resolve()
    baseline = baseline_path if baseline_path.is_absolute() else repo / baseline_path
    reads, forbidden = collect_fps_authority_accesses(
        source_root,
        repo_root=repo,
    )
    if forbidden:
        print(
            "Legacy FPS resolution is forbidden outside its compatibility owner; "
            "use fisheye.shared.metadata.get_fps/resolve_fps:",
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
            "FPS authority ratchet failed; new direct timing-metadata reads must "
            "use the shared recording resolver or a typed artifact resolver:",
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
        print(f"Tightened FPS authority baseline by {sum(removed.values())} read(s).")
    elif removed:
        print(f"FPS authority baseline can shrink by {sum(removed.values())} read(s).")
    print(
        "FPS authority access ratchet passed: "
        f"{sum(current.values())} grandfathered direct read(s)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Replace the baseline with the exact current AST census.",
    )
    parser.add_argument(
        "--no-update-on-shrink",
        action="store_true",
        help="Report removed direct reads without tightening the baseline.",
    )
    args = parser.parse_args(argv)
    if args.write_baseline:
        reads, forbidden = collect_fps_authority_accesses(args.source_root)
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
        print(f"Wrote FPS authority baseline with {len(reads)} direct read(s): {baseline}")
        return 0
    return check_fps_authority_access(
        source_root=args.source_root,
        baseline_path=args.baseline,
        update_on_shrink=not args.no_update_on_shrink,
    )


if __name__ == "__main__":
    raise SystemExit(main())
