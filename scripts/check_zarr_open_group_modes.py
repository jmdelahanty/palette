#!/usr/bin/env python3
"""Ratchet direct/consolidated intent on every ``zarr.open_group`` call."""

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
DEFAULT_BASELINE = Path(__file__).with_name("zarr_open_group_mode_ratchet_baseline.json")
MODE_KEYWORDS = frozenset({"use_consolidated", "consolidated"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BareOpenGroupCall:
    key: str
    relative_path: str
    symbol: str
    line: int


class _OpenGroupVisitor(ast.NodeVisitor):
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self.symbol_stack: list[str] = []
        self.calls: list[BareOpenGroupCall] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        is_zarr_open_group = (
            isinstance(target, ast.Attribute)
            and target.attr == "open_group"
            and isinstance(target.value, ast.Name)
            and target.value.id == "zarr"
        )
        if is_zarr_open_group and not any(
            keyword.arg in MODE_KEYWORDS for keyword in node.keywords
        ):
            symbol = ".".join(self.symbol_stack) or "<module>"
            normalized_call = ast.dump(node, annotate_fields=True, include_attributes=False)
            digest = sha256(normalized_call.encode("utf-8")).hexdigest()[:20]
            key = f"{self.relative_path}::{symbol}::{digest}"
            self.calls.append(
                BareOpenGroupCall(
                    key=key,
                    relative_path=self.relative_path,
                    symbol=symbol,
                    line=int(node.lineno),
                )
            )
        self.generic_visit(node)


def collect_bare_open_group_calls(
    source_root: Path,
    *,
    repo_root: Path | None = None,
) -> list[BareOpenGroupCall]:
    repo = (repo_root or _repo_root()).resolve()
    source = source_root if source_root.is_absolute() else repo / source_root
    calls: list[BareOpenGroupCall] = []
    for path in sorted(source.rglob("*.py")):
        relative_path = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        visitor = _OpenGroupVisitor(relative_path=relative_path)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return calls


def _counter(calls: list[BareOpenGroupCall]) -> Counter[str]:
    return Counter(call.key for call in calls)


def _read_baseline(path: Path) -> Counter[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Zarr open-mode baseline: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Zarr open-mode baseline entries must be an object: {path}")
    counter: Counter[str] = Counter()
    for key, count in entries.items():
        if not isinstance(key, str) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid Zarr open-mode baseline entry: {key!r}")
        counter[key] = count
    if payload.get("bare_call_count") != sum(counter.values()):
        raise ValueError(f"Zarr open-mode baseline count is inconsistent: {path}")
    return counter


def _write_baseline(path: Path, calls: list[BareOpenGroupCall]) -> None:
    counts = _counter(calls)
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "source_root": DEFAULT_SOURCE_ROOT.as_posix(),
        "bare_call_count": sum(counts.values()),
        "entries": dict(sorted(counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_zarr_open_group_modes(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    baseline_path: Path = DEFAULT_BASELINE,
    repo_root: Path | None = None,
    update_on_shrink: bool = True,
) -> int:
    repo = (repo_root or _repo_root()).resolve()
    baseline = baseline_path if baseline_path.is_absolute() else repo / baseline_path
    calls = collect_bare_open_group_calls(source_root, repo_root=repo)
    current = _counter(calls)
    expected = _read_baseline(baseline)
    added = current - expected
    removed = expected - current

    if added:
        details = {call.key: call for call in calls}
        print(
            "Zarr open-group metadata-mode ratchet failed; new or modified calls "
            "must pass use_consolidated explicitly:",
            file=sys.stderr,
        )
        for key, count in sorted(added.items()):
            call = details[key]
            print(f"  {call.relative_path}:{call.line} ({call.symbol}) x{count}", file=sys.stderr)
        return 1

    if removed and update_on_shrink:
        _write_baseline(baseline, calls)
        print(f"Tightened Zarr open-group baseline by {sum(removed.values())} call(s).")
    elif removed:
        print(f"Zarr open-group baseline can shrink by {sum(removed.values())} call(s).")

    print(
        "Zarr open-group metadata-mode ratchet passed: "
        f"{sum(current.values())} grandfathered bare call(s)."
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
        help="Report removed bare calls without tightening the baseline.",
    )
    args = parser.parse_args(argv)
    if args.write_baseline:
        calls = collect_bare_open_group_calls(args.source_root)
        baseline = args.baseline
        if not baseline.is_absolute():
            baseline = _repo_root() / baseline
        _write_baseline(baseline, calls)
        print(f"Wrote Zarr open-group baseline with {len(calls)} bare call(s): {baseline}")
        return 0
    return check_zarr_open_group_modes(
        source_root=args.source_root,
        baseline_path=args.baseline,
        update_on_shrink=not args.no_update_on_shrink,
    )


if __name__ == "__main__":
    raise SystemExit(main())
